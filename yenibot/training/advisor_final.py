"""Separate final fitting/freeze/test commands. No test download before freeze.

The selected validation checkpoint is the final model; no train+val refit.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import log_loss, accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from yenibot.data import download_full_klines
from yenibot.data.advisor_validation import validate_advisor_klines
from yenibot.features.builder import compute_bar_features
from yenibot.labeling.triple_barrier import add_long_only_labels
from yenibot.training import advisor
from yenibot.training.advisor import atomic_json, atomic_checkpoint, atomic_text, sha256, read_config
from yenibot.training.advisor_holdout import audit_reservation, assert_fitting_scope
from yenibot.training.advisor_models import build_advisor_model
from yenibot.training.advisor_stages import (immutable_json, runtime_identity,
    select_architecture, finalize_losses, run_stage_fold)
from yenibot.training.dataset import SequenceDataset
from yenibot.training.walk_forward import FoldIndices


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def validate_plan(config, reservation):
    audit_reservation(reservation)
    fit, test = config['fit'], reservation['test']
    train_start,train_end,val_start,val_end = [pd.Timestamp(fit[key]) for key in
        ['train_start','train_end','validation_start','validation_end']]
    if any(t.tzinfo is None or t.utcoffset().total_seconds() or t!=t.floor('h')
           for t in [train_start,train_end,val_start,val_end]):
        raise ValueError('Fit timestamps must be whole UTC hours')
    if train_end-train_start!=pd.Timedelta(hours=fit['train_bars']-1) or val_end-val_start!=pd.Timedelta(hours=fit['validation_bars']-1):
        raise ValueError('Final fit counts and dates disagree')
    horizon = test['label_horizon_bars']
    if fit['purge_bars']<horizon or val_start-train_end!=pd.Timedelta(hours=fit['purge_bars']+1):
        raise ValueError('Final train-validation purge does not cover labels')
    if val_end!=pd.Timestamp(reservation['boundaries']['latest_fitting_sample']):
        raise ValueError('Final validation end must match reserved fit cutoff')
    if fit['refit_on_train_plus_validation'] or fit['fold_seed_offset']!=0:
        raise ValueError('This protocol retains the selected validation checkpoint, without refit')
    if config['training']['seeds']!=reservation['selection']['seeds'] or config['training']['threshold']!=reservation['selection']['threshold']:
        raise ValueError('Reserved seeds/threshold changed')
    if config['model']['seq_len']!=test['sequence_length'] or config['labeling']!={
        'max_holding_bars':horizon,'tp_multiplier':test['tp_multiplier'],'sl_multiplier':test['sl_multiplier']}:
        raise ValueError('Reserved model context or label definition changed')
    if config['selected_method']['comparator_models']:
        raise ValueError('Model comparators require a separately frozen protocol')


def selected_contract(config):
    reservation = read_config(config['reservation_config'])
    validate_plan(config,reservation)
    root = Path(config['development_root'])
    stage = read_config(config['stage_config'])
    decision,control = select_architecture(root,stage)
    losses = finalize_losses(root,stage,decision)
    method = config['selected_method']
    if (losses['winner']!='L_BCE' or losses['wavelet_selection']['winner']!='W_OFF' or
        decision['winner']!='A' or method['architecture']!=decision['architecture'] or
        method['features']!=decision['features'] or method['loss']!='bce' or method['wavelet']):
        raise ValueError('Final method differs from frozen development selection')
    for key in ['training','model','labeling']:
        if config[key]!=control['config'][key]:
            raise ValueError(f'Final common setting differs from selected control: {key}')
    for name,key in [('train_bars','train_bars'),('validation_bars','val_bars'),('purge_bars','purge_bars')]:
        if config['fit'][name]!=control['config']['walk_forward'][key]:
            raise ValueError('Final window lengths differ from selected development protocol')
    reference = json.loads((root/'data/A_reference.manifest.json').read_text())
    features = copy.deepcopy(reference['feature_config'])
    if features['features']['wavelet']['enabled']:
        raise ValueError('Expected unfiltered feature reference')
    files = [Path(__file__),Path(advisor.__file__),Path(__file__).with_name('advisor_stages.py'),
             Path(__file__).with_name('advisor_objectives.py'),Path(__file__).with_name('advisor_models.py'),
             Path(__file__).with_name('dataset.py'),Path(__file__).with_name('advisor_holdout.py'),
             Path(__file__).parents[1]/'features/builder.py',Path(__file__).parents[1]/'labeling/triple_barrier.py',
             Path(__file__).parents[1]/'data/advisor_validation.py']
    contract = {'config':config,'reservation':reservation,'development_loss_selection':losses,
                'feature_config':features,'development_raw_hashes':reference['source_hashes'],
                'source_hashes':{str(path):sha256(path) for path in files},
                'environment':runtime_identity(),'method_selected_without_test':True}
    directory = Path(config['output'])
    immutable_json(directory/'data/final_plan.json',contract)
    return contract


def validate_raw_range(frame,start,end,policy):
    frame=frame.copy()
    frame.timestamp=pd.to_datetime(frame.timestamp,utc=True)
    frame=frame.loc[frame.timestamp.between(start,end)].reset_index(drop=True)
    clean,audit=validate_advisor_klines(frame,'1h',zero_activity_policy=policy)
    advisor.assert_hourly(clean)
    if clean.timestamp.iloc[0]!=start or clean.timestamp.iloc[-1]!=end:
        raise ValueError('Raw range incomplete; cannot shorten final scope')
    return clean,audit


def extend_source(base_path,destination,end,config,downloader=None):
    """Immutable clipped monthly caches; no interpolation/deletion of raw hours."""
    downloader = downloader or download_full_klines
    destination.mkdir(parents=True,exist_ok=True)
    target=destination/'btc_1h.parquet'
    marker=target.with_suffix('.manifest.json')
    start=pd.Timestamp(config['raw']['history_start'])
    identity={'start':start.isoformat(),'end':end.isoformat(),'base_sha256':sha256(base_path),
              'zero_activity_policy':config['raw']['zero_activity_policy']}
    if marker.exists():
        saved=json.loads(marker.read_text())
        if saved['identity']!=identity or sha256(target)!=saved['sha256']:
            raise ValueError('Frozen final raw source changed')
        return target
    base=pd.read_parquet(base_path)
    base.timestamp=pd.to_datetime(base.timestamp,utc=True)
    base=base.loc[base.timestamp<=end].copy()
    base,_=validate_raw_range(base,start,base.timestamp.iloc[-1],config['raw']['zero_activity_policy'])
    chunks=[base]
    lower=base.timestamp.iloc[-1]+pd.Timedelta(hours=1)
    cache=destination/'monthly_cache';cache.mkdir(exist_ok=True)
    while lower<=end:
        upper=min(lower.normalize().replace(day=1)+pd.offsets.MonthBegin(1),end+pd.Timedelta(hours=1))
        chunk_path=cache/f'{lower:%Y%m}.parquet';chunk_meta=chunk_path.with_suffix('.manifest.json')
        bounds={'start':lower.isoformat(),'end_exclusive':upper.isoformat()}
        if chunk_meta.exists():
            saved=json.loads(chunk_meta.read_text())
            if saved['bounds']!=bounds or sha256(chunk_path)!=saved['sha256']:
                raise ValueError('Final monthly cache changed')
            chunk=pd.read_parquet(chunk_path)
        else:
            print('Download final 1H chunk:',bounds,flush=True)
            chunk=downloader('BTCUSDT','1h',lower,upper,data_source='auto')
            chunk,quality=validate_raw_range(chunk,lower,upper-pd.Timedelta(hours=1),config['raw']['zero_activity_policy'])
            tmp=chunk_path.with_suffix('.tmp.parquet');chunk.to_parquet(tmp,index=False);os.replace(tmp,chunk_path)
            atomic_json(chunk_meta,{'bounds':bounds,'sha256':sha256(chunk_path),'quality':quality})
        chunks.append(chunk);lower=upper
    joined,audit=validate_raw_range(pd.concat(chunks,ignore_index=True),start,end,config['raw']['zero_activity_policy'])
    tmp=target.with_suffix('.tmp.parquet');joined.to_parquet(tmp,index=False);os.replace(tmp,target)
    atomic_json(marker,{'identity':identity,'sha256':sha256(target),'rows':len(joined),'audit':audit})
    return target


def feature_frame(raw,contract):
    """Same hourly feature functions; inactive 4H/OI channels are not required."""
    config=contract['config']
    frame=compute_bar_features(raw,contract['feature_config']).frame
    keep=list(dict.fromkeys(['timestamp','open','high','low','close','atr_14',*config['selected_method']['features']]))
    frame=frame[keep].copy().ffill()
    frame=add_long_only_labels(frame,**config['labeling'])
    frame['label_end_timestamp']=frame.timestamp+pd.Timedelta(hours=config['labeling']['max_holding_bars'])
    return frame


def exact_scope(frame,start,end,features):
    section=frame.loc[frame.timestamp.between(start,end)].reset_index(drop=True)
    expected=pd.date_range(start,end,freq='h')
    if not section.timestamp.equals(pd.Series(expected,name='timestamp')):
        raise ValueError('Feature/label scope lost hours')
    columns=features+['label','fwd_return_10h']
    if not np.isfinite(section[columns].to_numpy()).all() or not set(section.label.unique()).issubset({0,1}):
        raise ValueError('Invalid final features/labels; no neutral filling allowed')
    return section


def fit_indices(config):
    fit=config['fit'];val_start=fit['train_bars']+fit['purge_bars']
    n=val_start+fit['validation_bars']
    return FoldIndices(0,np.arange(fit['train_bars']),np.arange(val_start,n),np.array([n]))


def prepare_fit(config):
    contract=selected_contract(config);root=Path(config['output'])
    plan_sha=digest(contract)
    target=root/'data/fit_frame.parquet';marker=target.with_suffix('.manifest.json')
    if marker.exists():
        saved=json.loads(marker.read_text())
        if saved['plan_sha256']!=plan_sha or sha256(target)!=saved['sha256']:
            raise ValueError('Frozen fitting frame changed')
        return
    snapshot=Path(config['raw']['base_snapshot'])
    manifest=json.loads((snapshot/'snapshot_manifest.json').read_text())
    base=snapshot/'btc_1h.parquet'
    if sha256(base)!=manifest['files']['1h']['sha256'] or sha256(base)!=contract['development_raw_hashes']['1h']:
        raise ValueError('Development raw source changed')
    end=pd.Timestamp(config['fit']['validation_end'])+pd.Timedelta(hours=config['labeling']['max_holding_bars'])
    if end>=pd.Timestamp(contract['reservation']['test']['start']):
        raise ValueError('Fit source would enter reserved test')
    raw_path=extend_source(base,root/'inputs/fit_raw',end,config)
    frame=feature_frame(pd.read_parquet(raw_path),contract)
    frame=exact_scope(frame,pd.Timestamp(config['fit']['train_start']),pd.Timestamp(config['fit']['validation_end']),config['selected_method']['features'])
    fold=fit_indices(config)
    assert_fitting_scope(frame.iloc[fold.train],contract['reservation'])
    assert_fitting_scope(frame.iloc[fold.val],contract['reservation'])
    advisor.audit_boundary(frame,fold.train,fold.val,config['labeling']['max_holding_bars'])
    tmp=target.with_suffix('.tmp.parquet');frame.to_parquet(tmp,index=False);os.replace(tmp,target)
    atomic_json(marker,{'plan_sha256':plan_sha,'sha256':sha256(target),'raw_sha256':sha256(raw_path),
                        'rows':len(frame),'fit_operations_on_test':0,'test_evaluations':0})
    print('Final fit frame ready:',len(frame),'rows; test has not been downloaded',flush=True)


def train(config):
    contract=selected_contract(config);root=Path(config['output'])
    frame_path=root/'data/fit_frame.parquet'
    info=json.loads(frame_path.with_suffix('.manifest.json').read_text())
    if info['plan_sha256']!=digest(contract) or sha256(frame_path)!=info['sha256']:
        raise ValueError('Fit data changed')
    frame=pd.read_parquet(frame_path);fold=fit_indices(config)
    # Calendar marker only, without OHLC/labels/features. Never a test input.
    marker=pd.DataFrame({'timestamp':[pd.Timestamp(contract['reservation']['test']['start'])]})
    frame=pd.concat([frame,marker],ignore_index=True)
    signature=digest({'contract':contract,'fit_data':info})
    features=config['selected_method']['features']
    torch.set_num_threads(config['training']['torch_threads'])
    device=torch.device(contract['environment']['device'])
    with advisor.exclusive_run(root/'runs/final'):
        immutable_json(root/'runs/final/protocol.json',{'contract':contract,'signature':signature,'fit_data':info})
        progress={'status':'training','completed':[],'test_evaluations':0}
        try:
            for seed in config['training']['seeds']:
                progress['current_seed']=seed;atomic_json(root/'runs/final/status.json',progress)
                result=run_stage_fold(frame,fold,config,{'architecture':'gru','objective':{'kind':'bce'}},
                    features,seed,root/f'runs/final/seed_{seed}',signature,device)
                progress['completed'].append({'seed':seed,'best_epoch':result['best_epoch'],'validation':result['validation']})
            progress['status']='complete';progress.pop('current_seed',None)
        except BaseException as exc:
            progress.update(status='interrupted_or_failed',error=repr(exc));raise
        finally:
            atomic_json(root/'runs/final/status.json',progress)
    print('Final fitting complete; test evaluations: 0',flush=True)


def verify_freeze(config):
    root=Path(config['output']);contract=selected_contract(config)
    path=root/'artifacts/frozen_manifest.json'
    if not path.exists():
        raise ValueError('Freeze all three validation-selected models before accessing test')
    saved=json.loads(path.read_text())
    if saved['contract_sha256']!=digest(contract):
        raise ValueError('Frozen final contract changed')
    if saved['seeds']!=config['training']['seeds']:
        raise ValueError('Frozen seed scope changed')
    for record in saved['artifacts']:
        if sha256(root/record['path'])!=record['sha256']:
            raise ValueError('Frozen model/scaler checksum mismatch')
    return contract,saved


def freeze(config):
    contract=selected_contract(config);root=Path(config['output'])
    if (root/'artifacts/frozen_manifest.json').exists():
        verify_freeze(config);print('Frozen models verified; not overwritten',flush=True);return
    status=json.loads((root/'runs/final/status.json').read_text())
    if status['status']!='complete' or [r['seed'] for r in status['completed']]!=config['training']['seeds']:
        raise ValueError('All final seeds must complete before freeze')
    protocol=json.loads((root/'runs/final/protocol.json').read_text())
    frame_path=root/'data/fit_frame.parquet'
    if protocol['contract']!=contract or sha256(frame_path)!=protocol['fit_data']['sha256']:
        raise ValueError('Final fitting identity changed')
    frame=pd.read_parquet(frame_path);fold=fit_indices(config);seq=config['model']['seq_len']
    prior=float(frame.iloc[fold.train].label.iloc[seq-1:].mean())
    directory=root/'artifacts';directory.mkdir(exist_ok=True)
    records=[]
    for seed in config['training']['seeds']:
        checkpoint=torch.load(root/f'runs/final/seed_{seed}/last.pt',map_location='cpu',weights_only=False)
        result=json.loads((root/f'runs/final/seed_{seed}/validation_metrics.json').read_text())
        if checkpoint['signature']!=protocol['signature'] or result['signature']!=protocol['signature'] or checkpoint['best_epoch']!=result['best_epoch']:
            raise ValueError('Final checkpoint/result identity mismatch')
        path=directory/f'seed_{seed}.pt'
        atomic_checkpoint(path,{'model':checkpoint['best_state'],'scaler_center':checkpoint['scaler_center'],
            'scaler_scale':checkpoint['scaler_scale'],'seed':seed,'features':config['selected_method']['features'],
            'contract_sha256':digest(contract),'best_epoch':checkpoint['best_epoch'],'training_positive_rate':prior})
        records.append({'seed':seed,'path':path.relative_to(root).as_posix(),'sha256':sha256(path),
                        'best_epoch':result['best_epoch'],'validation_metrics':result['validation']})
    atomic_json(directory/'frozen_manifest.json',{'contract_sha256':digest(contract),
        'fit_frame_sha256':sha256(frame_path),'training_positive_rate':prior,'artifacts':records,
        'seeds':config['training']['seeds'],'frozen_at':datetime.now(timezone.utc).isoformat(),
        'test_data_downloaded_before_freeze':False,'test_evaluations':0})
    verify_freeze(config)
    print('All 3 model/scaler artifacts frozen. Test access is now permitted.',flush=True)


def prepare_test(config):
    contract,frozen=verify_freeze(config);root=Path(config['output'])
    target=root/'data/test_frame.parquet';marker=target.with_suffix('.manifest.json')
    freeze_sha=sha256(root/'artifacts/frozen_manifest.json')
    if marker.exists():
        saved=json.loads(marker.read_text())
        if saved['freeze_sha256']!=freeze_sha or sha256(target)!=saved['sha256']:
            raise ValueError('Frozen test frame identity changed')
        return
    reservation=contract['reservation'];audit=audit_reservation(reservation)
    raw_path=extend_source(root/'inputs/fit_raw/btc_1h.parquet',root/'inputs/test_raw',
                          pd.Timestamp(audit['required_last_1h_label_source_bar']),config)
    frame=feature_frame(pd.read_parquet(raw_path),contract)
    features=config['selected_method']['features']
    fit=pd.read_parquet(root/'data/fit_frame.parquet')
    recreated=exact_scope(frame,fit.timestamp.iloc[0],fit.timestamp.iloc[-1],features)
    pd.testing.assert_frame_equal(fit,recreated)
    start=pd.Timestamp(audit['first_sequence_context_timestamp']);end=pd.Timestamp(reservation['test']['end'])
    section=exact_scope(frame,start,end,features)
    if len(section)!=reservation['test']['expected_prediction_timestamps']+config['model']['seq_len']-1:
        raise ValueError('Test context/count contract violated')
    tmp=target.with_suffix('.tmp.parquet');section.to_parquet(tmp,index=False);os.replace(tmp,target)
    atomic_json(marker,{'freeze_sha256':freeze_sha,'sha256':sha256(target),'raw_sha256':sha256(raw_path),
        'rows':len(section),'predictions':reservation['test']['expected_prediction_timestamps'],
        'fit_history_unchanged':True,'test_fit_operations':0})
    print('Test frame ready: 63 past context rows + 720 prediction endpoints',flush=True)


def score_frame(model,artifact,frame,config,device):
    features=config['selected_method']['features']
    frame=frame.copy()
    scaler=RobustScaler();scaler.center_=artifact['scaler_center'];scaler.scale_=artifact['scaler_scale']
    scaler.n_features_in_=len(features)
    frame.loc[:,features]=scaler.transform(frame[features].to_numpy())
    dataset=SequenceDataset(frame[features].to_numpy(np.float32),frame.label.to_numpy(np.float32),
        frame.fwd_return_10h.to_numpy(np.float32),seq_len=config['model']['seq_len'])
    loader=DataLoader(dataset,batch_size=config['training']['batch_size'],shuffle=False,num_workers=0)
    scores,predictions=advisor.evaluate(model,loader,device,config['training']['threshold'])
    predictions['timestamp']=frame.iloc[predictions.source_row_position].timestamp.to_numpy()
    return scores,predictions


def constant_reference(labels,probability,threshold):
    """A constant score has undefined Rank IC and AP equal to prevalence."""
    probabilities=np.full(len(labels),probability,dtype=float)
    precision,recall,f1,_=precision_recall_fscore_support(labels,probabilities>=threshold,
        average='binary',zero_division=0)
    return {'bce':float(log_loss(labels,probabilities,labels=[0,1])),
            'average_precision':float(np.mean(labels)) if np.any(labels==1) else None,
            'precision':float(precision),'recall':float(recall),'f1':float(f1),
            'accuracy':float(accuracy_score(labels,probabilities>=threshold)),
            'rank_ic':None,'prevalence':float(np.mean(labels)),
            'threshold':threshold,'samples':len(labels)}


def evaluate_test(config):
    contract,frozen=verify_freeze(config);root=Path(config['output'])
    path=root/'data/test_frame.parquet';metadata=json.loads(path.with_suffix('.manifest.json').read_text())
    freeze_sha=sha256(root/'artifacts/frozen_manifest.json')
    if metadata['freeze_sha256']!=freeze_sha or sha256(path)!=metadata['sha256']:
        raise ValueError('Test frame changed')
    directory=root/'runs/test';directory.mkdir(parents=True,exist_ok=True)
    test_identity={'freeze_sha256':freeze_sha,'test_frame_sha256':sha256(path),'contract_sha256':digest(contract)}
    with advisor.exclusive_run(directory):
        immutable_json(directory/'protocol.json',test_identity)
        result_path=directory/'test_results.json'
        if result_path.exists():
            saved=json.loads(result_path.read_text())
            if saved['identity']!=test_identity:
                raise ValueError('Cached test result identity changed')
            for record in saved['prediction_files']:
                if sha256(directory/record['path'])!=record['sha256']:
                    raise ValueError('Cached test predictions changed')
            print('Completed fixed test reused; no new evaluation',flush=True);return
        frame=pd.read_parquet(path);device=torch.device(contract['environment']['device'])
        expected=pd.Series(pd.date_range(contract['reservation']['test']['start'],contract['reservation']['test']['end'],freq='h'),name='timestamp')
        results=[];files=[]
        atomic_json(directory/'status.json',{'status':'evaluating','test_fit_operations':0})
        for record in frozen['artifacts']:
            artifact=torch.load(root/record['path'],map_location='cpu',weights_only=False)
            if artifact['contract_sha256']!=digest(contract) or artifact['features']!=config['selected_method']['features']:
                raise ValueError('Artifact internal contract mismatch')
            seed=record['seed'];prediction_path=directory/f'seed_{seed}_test_predictions.parquet'
            score_path=directory/f'seed_{seed}_metrics.json'
            if score_path.exists():
                cached=json.loads(score_path.read_text())
                if cached['identity']!=test_identity or sha256(prediction_path)!=cached['prediction_sha256']:
                    raise ValueError('Partial test checkpoint changed')
                scores=cached['metrics']
            else:
                model=build_advisor_model(len(artifact['features']),'gru',config['model']).to(device)
                model.load_state_dict(artifact['model'])
                scores,predictions=score_frame(model,artifact,frame,config,device)
                if not pd.to_datetime(predictions.timestamp,utc=True).equals(expected):
                    raise ValueError('Test prediction timestamps/count differ from reserved 720')
                tmp=prediction_path.with_suffix('.tmp.parquet');predictions.to_parquet(tmp,index=False);os.replace(tmp,prediction_path)
                atomic_json(score_path,{'identity':test_identity,'metrics':scores,'prediction_sha256':sha256(prediction_path)})
            results.append({'seed':seed,**scores})
            files.append({'path':prediction_path.name,'sha256':sha256(prediction_path)})
        table=pd.DataFrame(results)
        keys=['bce','average_precision','precision','recall','f1','accuracy','rank_ic','prevalence']
        aggregate={key:{'mean':None if table[key].isna().all() else float(table[key].mean()),
                        'std':None if table[key].count()<2 else float(table[key].std(ddof=1))} for key in keys}
        endpoints=frame.iloc[config['model']['seq_len']-1:]
        baseline=constant_reference(endpoints.label.to_numpy(),frozen['training_positive_rate'],config['training']['threshold'])
        negative=constant_reference(endpoints.label.to_numpy(),0.,.5)
        # Always-negative is a class-only baseline; its probability BCE/AP are not a fitted forecast.
        negative={key:negative[key] for key in ['precision','recall','f1','accuracy','samples','prevalence']}
        result={'identity':test_identity,'per_seed':results,'aggregate_mean_std':aggregate,
                'baselines':{'training_frequency_probability':baseline,'always_negative_classification':negative},
                'prediction_files':files,'test_fit_operations':0,'test_evaluations':1,
                'reporting':'one fixed 720-hour holdout, three seeds; no ensemble or independent-replicate claim'}
        atomic_json(result_path,result);table.to_csv(directory/'test_metrics_per_seed.csv',index=False)
        pd.DataFrame(aggregate).T.to_csv(directory/'test_metrics_mean_std.csv')
        atomic_json(directory/'status.json',{'status':'complete','test_evaluations':1,'test_fit_operations':0,
                    'seeds_scored':config['training']['seeds'],'test_timestamps':len(endpoints)})
        atomic_text(directory/'STATUS.md','# Final test complete\n\n720 timestamps, 3 fixed seeds. No test fitting.\n')
    print(json.dumps(result,indent=2),flush=True)


def review_bundle(config):
    root=Path(config['output']);reports=root/'reports';reports.mkdir(parents=True,exist_ok=True)
    allowed={'.json','.yaml','.yml','.csv','.md','.log','.txt'}
    files=[p for folder in ['data','runs','artifacts','sessions','inputs'] for p in (root/folder).rglob('*')
           if p.is_file() and (p.suffix in allowed or p.name=='validation_predictions.parquet' or p.name.endswith('_test_predictions.parquet'))]
    # Includes manifests of raw inputs and weights, never raw arrays or serialized models.
    inventory=[{'path':p.relative_to(root).as_posix(),'sha256':sha256(p),'size':p.stat().st_size} for p in files]
    result=root/'runs/test/test_results.json'
    role='final_holdout_scored' if result.exists() else 'final_fitting_or_partial_test'
    manifest={'files':inventory,'role':role,'weights_included':False,'test_evaluations':1 if result.exists() else 0}
    target=reports/'advisor_final_latest_review_bundle.zip';tmp=target.with_suffix('.tmp.zip')
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(root).as_posix())
        z.writestr('review_manifest.json',json.dumps(manifest,indent=2))
    os.replace(tmp,target)
    print('Final review ZIP:',target,flush=True)
    return target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare-fit','train','freeze','prepare-test','evaluate-test','bundle'])
    parser.add_argument('--config',required=True)
    args=parser.parse_args();config=read_config(args.config)
    actions={'prepare-fit':prepare_fit,'train':train,'freeze':freeze,'prepare-test':prepare_test,
             'evaluate-test':evaluate_test,'bundle':review_bundle}
    actions[args.action](config)


if __name__=='__main__':
    main()
