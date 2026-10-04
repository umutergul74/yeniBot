import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from yenibot.training import advisor_final as final
from yenibot.training.advisor import read_config, sha256


def protocol():
    return read_config('configs/advisor_final_execution.yaml')


def reservation():
    return read_config('configs/advisor_final_test.yaml')


def test_final_dates_preserve_reserved_window_and_original_counts():
    final.validate_plan(protocol(),reservation())
    fold=final.fit_indices(protocol())
    assert len(fold.train)==5040 and len(fold.val)==1080
    assert fold.val[0]-fold.train[-1]-1==24


def test_selected_contract_checks_method_and_freezes_full_plan(tmp_path,monkeypatch):
    cfg=protocol();cfg['output']=str(tmp_path/'final');cfg['development_root']=str(tmp_path/'development')
    root=Path(cfg['development_root']);(root/'data').mkdir(parents=True)
    feature_cfg=read_config('config.yaml');feature_cfg['features']['wavelet']['enabled']=False
    (root/'data/A_reference.manifest.json').write_text(json.dumps({'feature_config':feature_cfg,
        'source_hashes':{'1h':'fixed-hourly-source','4h':'fixed-4h-source'}}))
    control={'config':{key:copy.deepcopy(cfg[key]) for key in ['training','model','labeling']}}
    control['config']['walk_forward']={'train_bars':5040,'val_bars':1080,'purge_bars':24}
    decision={'winner':'A','architecture':'gru','features':cfg['selected_method']['features']}
    losses={'winner':'L_BCE','wavelet_selection':{'winner':'W_OFF'}}
    monkeypatch.setattr(final,'select_architecture',lambda *args:(decision,control))
    monkeypatch.setattr(final,'finalize_losses',lambda *args:losses)
    contract=final.selected_contract(cfg)
    assert (Path(cfg['output'])/'data/final_plan.json').exists()
    assert final.selected_contract(cfg)==contract
    changed=copy.deepcopy(cfg);changed['raw']['history_start']='2022-01-02T00:00:00Z'
    with pytest.raises(ValueError,match='Frozen stage selection'):final.selected_contract(changed)
    losses['winner']='L_FOCAL'
    with pytest.raises(ValueError,match='Final method differs'):final.selected_contract(cfg)


@pytest.mark.parametrize('change',['short_gap','shift_val','refit','threshold','count'])
def test_invalid_final_plans_rejected(change):
    cfg=protocol()
    if change=='short_gap':cfg['fit']['purge_bars']=6
    if change=='shift_val':
        cfg['fit']['validation_start']='2026-07-18T00:00:00Z'
        cfg['fit']['validation_end']='2026-08-31T23:00:00Z'
    if change=='refit':cfg['fit']['refit_on_train_plus_validation']=True
    if change=='threshold':cfg['training']['threshold']=.4
    if change=='count':cfg['fit']['train_bars']=5039
    with pytest.raises(ValueError):final.validate_plan(cfg,reservation())


def raw_frame():
    times=pd.date_range('2026-07-01','2026-10-01 09:00',freq='h',tz='UTC')
    x=np.arange(len(times));price=100+np.sin(x/3)*2+np.cos(x/7)
    return pd.DataFrame({'timestamp':times,'open':price,'close':price+.1,'high':price+1.5,
        'low':price-1.5,'volume':100+np.cos(x/5)*20,'quote_volume':(100+np.cos(x/5)*20)*price,
        'num_trades':np.full(len(times),20),'taker_buy_base_vol':(100+np.cos(x/5)*20)*.5,
        'taker_buy_quote_vol':(100+np.cos(x/5)*20)*price*.5})


def fixture(tmp_path,monkeypatch):
    cfg=protocol();cfg['output']=str(tmp_path/'final');cfg['raw']['base_snapshot']=str(tmp_path/'snapshot')
    cfg['raw']['history_start']='2026-07-01T00:00:00Z'
    cfg['fit']['train_bars']=96;cfg['fit']['validation_bars']=80
    end=pd.Timestamp(cfg['fit']['validation_end']);start=end-pd.Timedelta(hours=79)
    train_end=start-pd.Timedelta(hours=25);train_start=train_end-pd.Timedelta(hours=95)
    for key,value in [('validation_start',start),('train_end',train_end),('train_start',train_start)]:cfg['fit'][key]=value.isoformat()
    cfg['model'].update(gru_hidden=4,gru_layers=1,fusion_hidden=4)
    cfg['training'].update(epochs=2,batch_size=16,torch_threads=2)
    final.validate_plan(cfg,reservation())
    raw=raw_frame();snapshot=Path(cfg['raw']['base_snapshot']);snapshot.mkdir()
    base=snapshot/'btc_1h.parquet';raw.iloc[:1000].to_parquet(base,index=False)
    (snapshot/'snapshot_manifest.json').write_text(json.dumps({'files':{'1h':{'sha256':sha256(base)}}}))
    feature_cfg=read_config('config.yaml');feature_cfg['features']['wavelet']['enabled']=False
    contract={'config':cfg,'reservation':reservation(),'feature_config':feature_cfg,
              'environment':{'device':'cpu'},'development_raw_hashes':{'1h':sha256(base)}}
    root=Path(cfg['output']);(root/'data').mkdir(parents=True)
    def selected(unused):
        final.immutable_json(root/'data/final_plan.json',contract)
        return contract
    monkeypatch.setattr(final,'selected_contract',selected)
    calls=[]
    def download(symbol,interval,start,end,**kwargs):
        calls.append((start,end))
        if pd.Timestamp(end)>pd.Timestamp(reservation()['test']['start']):
            assert (root/'artifacts/frozen_manifest.json').exists()
        return raw.loc[raw.timestamp.ge(start)&raw.timestamp.lt(end)].copy()
    monkeypatch.setattr(final,'download_full_klines',download)
    return cfg,root,calls


def test_complete_final_flow_blocks_early_test_resumes_and_preserves_weights(tmp_path,monkeypatch):
    cfg,root,calls=fixture(tmp_path,monkeypatch)
    for action in [final.prepare_test,final.evaluate_test]:
        with pytest.raises(ValueError,match='Freeze'):action(cfg)
    assert not calls
    final.prepare_fit(cfg);count=len(calls);final.prepare_fit(cfg);assert len(calls)==count
    fit=pd.read_parquet(root/'data/fit_frame.parquet')
    assert len(fit)==200 and fit.timestamp.max()==pd.Timestamp(cfg['fit']['validation_end'])
    final.train(cfg);final.freeze(cfg)
    before={p.name:sha256(p) for p in (root/'artifacts').glob('*.pt')}
    final.prepare_test(cfg)
    test=pd.read_parquet(root/'data/test_frame.parquet');assert len(test)==783
    original=final.score_frame;scored=[]
    def interrupt(model,artifact,*args):
        scored.append(artifact['seed'])
        if artifact['seed']==43:raise RuntimeError('interruption')
        return original(model,artifact,*args)
    monkeypatch.setattr(final,'score_frame',interrupt)
    with pytest.raises(RuntimeError,match='interruption'):final.evaluate_test(cfg)
    assert (root/'runs/test/seed_42_metrics.json').exists()
    scored.clear()
    def resumed(model,artifact,*args):
        scored.append(artifact['seed']);return original(model,artifact,*args)
    monkeypatch.setattr(final,'score_frame',resumed)
    final.evaluate_test(cfg);assert scored==[43,44]
    scored.clear();final.evaluate_test(cfg);assert not scored
    result=json.loads((root/'runs/test/test_results.json').read_text())
    assert result['test_fit_operations']==0 and result['test_evaluations']==1
    assert len(result['per_seed'])==3 and all(row['samples']==720 for row in result['per_seed'])
    assert before=={p.name:sha256(p) for p in (root/'artifacts').glob('*.pt')}
    zip_path=final.review_bundle(cfg)
    import zipfile
    with zipfile.ZipFile(zip_path) as z:
        assert not any(name.endswith('.pt') for name in z.namelist())
        assert len([name for name in z.namelist() if name.endswith('_test_predictions.parquet')])==3
    import runpy
    reviewer=runpy.run_path('scripts/review_advisor_final_bundle.py')['review']
    verified=reviewer(zip_path,tmp_path/'review')
    assert verified['test_fit_operations']==0
    weight=root/'artifacts/seed_42.pt';weight.write_bytes(weight.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='checksum'):final.prepare_test(cfg)


def test_fit_resume_has_identical_validation_checkpoint(tmp_path,monkeypatch):
    cfg,root,calls=fixture(tmp_path,monkeypatch)
    final.prepare_fit(cfg)
    original=final.run_stage_fold
    def fail_second(*args,**kwargs):
        if args[5]==43:raise RuntimeError('stopped between seeds')
        return original(*args,**kwargs)
    monkeypatch.setattr(final,'run_stage_fold',fail_second)
    with pytest.raises(RuntimeError,match='between seeds'):final.train(cfg)
    checkpoint=root/'runs/final/seed_42/last.pt';saved=sha256(checkpoint)
    monkeypatch.setattr(final,'run_stage_fold',original)
    final.train(cfg);assert sha256(checkpoint)==saved
    final.freeze(cfg);saved=sha256(root/'artifacts/frozen_manifest.json')
    final.freeze(cfg);assert sha256(root/'artifacts/frozen_manifest.json')==saved
