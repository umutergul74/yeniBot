import copy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from yenibot.training import advisor, advisor_stages as stages
from yenibot.training.advisor_objectives import AdvisorObjective
from yenibot.training.walk_forward import PurgedWalkForwardCV


@pytest.mark.parametrize('kind',['bce','weighted_bce','focal','bce_pearson','focal_pearson'])
def test_objectives_finite_gradients_and_training_only_weight(kind):
    training = torch.tensor([0.,0.,0.,1.])
    objective = AdvisorObjective({'kind':kind,'pearson_weight':.05},training)
    logits = torch.tensor([-.3,.1,.5,-.1],requires_grad=True)
    targets = torch.tensor([1.,1.,0.,0.])  # Different validation/batch rate must not set class weight.
    loss = objective(logits,targets,torch.tensor([.01,-.02,.03,-.01]))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert objective.audit['training_sequence_positives']==1
    if kind=='weighted_bce':
        assert objective.pos_weight.item()==3
        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits,targets,pos_weight=torch.tensor(3.))
        assert torch.equal(loss,expected)


def test_invalid_weighted_class_and_parameters_rejected():
    with pytest.raises(ValueError,match='both classes'):
        AdvisorObjective({'kind':'weighted_bce'},torch.ones(3))
    with pytest.raises(ValueError,match='parameters'):
        AdvisorObjective({'kind':'focal','gamma':-1},torch.tensor([0.,1.]))


def test_wavelet_channels_causal_base_and_labels_unchanged():
    n=430
    raw=pd.DataFrame({'timestamp':pd.date_range('2022-01-01',periods=n,freq='h',tz='UTC'),
                      'close':100+np.sin(np.arange(n)/3)+np.arange(n)*.01,
                      'volume':1000+np.sin(np.arange(n)/4)*100})
    base=raw.iloc[360:380].reset_index(drop=True)
    base['label']=np.arange(len(base))%2
    base['fwd_return_10h']=np.arange(len(base))*.001
    settings={'window':256,'wavelet':'db4','level':2,'threshold_scale':.5,
              'replacements':{'log_return':'close_denoised_log_return','volume_log_zscore':'volume_denoised_log_zscore'}}
    config={'features':{'stationarity':{'normalization_window':100}}}
    first=stages.add_wavelet_channels(base,raw.iloc[:380],config,settings)
    raw.loc[380:,['close','volume']]*=10
    second=stages.add_wavelet_channels(base,raw,config,settings)
    pd.testing.assert_frame_equal(first,second)
    pd.testing.assert_frame_equal(first[base.columns],base)
    assert stages.features_for(['log_return','volume_log_zscore','atr_14_pct'],settings,True)==[
        'close_denoised_log_return','volume_denoised_log_zscore','atr_14_pct']


def fixture_frame(n=100):
    timestamps=pd.date_range('2022-01-01',periods=n,freq='h',tz='UTC')
    return pd.DataFrame({'timestamp':timestamps,'label_end_timestamp':timestamps+pd.Timedelta(hours=3),
        'exit_timestamp':timestamps+pd.Timedelta(hours=1),'x':np.sin(np.arange(n)),
        'label':np.arange(n)%2,'fwd_return_3h':np.cos(np.arange(n))*.001})


def small_config():
    return dict(labeling={'max_holding_bars':3},model=dict(seq_len=4,tcn_channels=4,tcn_kernel_size=3,
        tcn_dilations=[1],gru_hidden=4,gru_layers=1,dropout=.2,fusion_hidden=4),
        training=dict(batch_size=8,deterministic=True,learning_rate=.001,weight_decay=.0001,
                      epochs=3,patience=10,threshold=.5,grad_clip=1.))


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_stage_bce_matches_original_control_exactly(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(2)
    fold=next(PurgedWalkForwardCV(train_bars=32,val_bars=12,test_bars=12,step_bars=12,
        purge_bars=3,embargo_bars=3,label_horizon_bars=3).split(100))
    kwargs=dict(frame=fixture_frame(),fold=fold,config=small_config(),features=['x'],seed=42,
                signature='control-equivalence',device=torch.device(device))
    a=advisor.run_fold(**kwargs,experiment={'architecture':'gru'},directory=tmp_path/'old')
    b=stages.run_stage_fold(**kwargs,experiment={'architecture':'gru','objective':{'kind':'bce'}},directory=tmp_path/'new')
    assert a['validation']==b['validation'] and a['best_epoch']==b['best_epoch']
    left=torch.load(tmp_path/'old/last.pt',weights_only=False)
    right=torch.load(tmp_path/'new/last.pt',weights_only=False)
    assert all(torch.equal(value,right['model'][key]) for key,value in left['model'].items())


def test_non_bce_interrupted_resume_preserves_weights_and_epoch_choice(tmp_path,monkeypatch):
    torch.set_num_threads(2)
    fold=next(PurgedWalkForwardCV(train_bars=32,val_bars=12,test_bars=12,step_bars=12,
        purge_bars=3,embargo_bars=3,label_horizon_bars=3).split(100))
    kwargs=dict(frame=fixture_frame(),fold=fold,config=small_config(),features=['x'],seed=42,
        signature='focal-resume',device=torch.device('cpu'),
        experiment={'architecture':'gru','objective':{'kind':'focal_pearson','pearson_weight':.05}})
    uninterrupted=stages.run_stage_fold(**kwargs,directory=tmp_path/'whole')
    original=stages.evaluate
    calls=0
    def interrupt(*args,**kw):
        nonlocal calls
        calls+=1
        if calls==2:
            raise RuntimeError('interrupt')
        return original(*args,**kw)
    monkeypatch.setattr(stages,'evaluate',interrupt)
    with pytest.raises(RuntimeError,match='interrupt'):
        stages.run_stage_fold(**kwargs,directory=tmp_path/'resume')
    monkeypatch.setattr(stages,'evaluate',original)
    resumed=stages.run_stage_fold(**kwargs,directory=tmp_path/'resume')
    assert resumed==uninterrupted
    a=torch.load(tmp_path/'whole/last.pt',weights_only=False)
    b=torch.load(tmp_path/'resume/last.pt',weights_only=False)
    assert a['history']==b['history']
    assert all(torch.equal(value,b['model'][key]) for key,value in a['model'].items())


def fake_completed(root,name,score,stage):
    directory=root/'runs'/name
    directory.mkdir(parents=True)
    protocol={**stages.runtime_identity(),'folds':[0,1],'seeds':[42],'features':['log_return','volume_log_zscore'],
              'config':{'training':{'loss':'bce'},'data':{'wavelet':False},'model':{},'development':{},
                        'labeling':{},'walk_forward':{},'experiments':{name:{'architecture':'gru'}}},
              'test_evaluations':0}
    # Name equals experiment in this isolated fixture.
    rows=[{'seed':42,'fold':fold,'validation':{'bce':score}} for fold in range(2)]
    (directory/'protocol.json').write_text(json.dumps(protocol))
    (directory/'status.json').write_text(json.dumps({'status':'complete','completed':rows,
        'signature':name,'test_evaluations':0}))
    pd.DataFrame([{'seed':42,'fold':fold,'bce':score} for fold in range(2)]).to_csv(directory/'validation_summary.csv',index=False)
    for fold in range(2):
        folder=directory/f'seed_42/fold_{fold:03d}'
        folder.mkdir(parents=True)
        (folder/'validation_metrics.json').write_text(json.dumps({'signature':name,'validation':{'bce':score}}))


def test_selection_requires_complete_scopes_and_freezes_decision(tmp_path):
    stage={'protocol':'fixture','reference_runs':{name:name for name in 'ABCD'},
           'selection':{'expected_folds':2,'expected_seeds':[42], 'architecture_tie_order':list('ABCD'),
                        'expected_architecture_winner':'A'}}
    for index,name in enumerate('ABCD'):
        fake_completed(tmp_path,name,.5+index*.1,stage)
    decision,_=stages.select_architecture(tmp_path,stage)
    assert decision['winner']=='A'
    assert stages.select_architecture(tmp_path,stage)[0]==decision
    path=tmp_path/'runs/D/status.json'
    status=json.loads(path.read_text())
    status['completed'].pop()
    path.write_text(json.dumps(status))
    with pytest.raises(ValueError,match='Incomplete'):
        stages.select_architecture(tmp_path,stage)


@pytest.mark.parametrize('on_score,expected',[(.4,'W_ON'),(.5,'W_OFF'),(.6,'W_OFF')])
def test_wavelet_selection_and_loss_completion_gate(tmp_path,on_score,expected):
    stage={'protocol':'fixture','reference_runs':{name:name for name in 'ABCD'},
        'selection':{'expected_folds':2,'expected_seeds':[42],'architecture_tie_order':list('ABCD'),
                     'expected_architecture_winner':'A','wavelet_tie_order':['W_OFF','W_ON']},
        'losses':{'L_FOCAL':{'kind':'focal','alpha':.6,'gamma':2}},'loss_tie_order':['L_BCE','L_FOCAL']}
    for index,name in enumerate('ABCD'):
        fake_completed(tmp_path,name,.5+index*.1,stage)
    decision,_=stages.select_architecture(tmp_path,stage)
    fake_completed(tmp_path,'W_ON',on_score,stage)
    path=tmp_path/'runs/W_ON/protocol.json'
    protocol=json.loads(path.read_text())
    protocol.update(experiment='W_ON',stage_architecture_selection=decision,data_sha256='shared_frame')
    path.write_text(json.dumps(protocol))
    (tmp_path/'data/stages_wavelet_run.json').write_text(json.dumps({'run':'W_ON','signature':'W_ON'}))
    choice=stages.select_wavelet(tmp_path,stage,decision)
    assert choice['winner']==expected
    with pytest.raises(ValueError,match='complete consistent loss'):
        stages.finalize_losses(tmp_path,stage,decision)
    fake_completed(tmp_path,'L_FOCAL',.3,stage)
    path=tmp_path/'runs/L_FOCAL/protocol.json'
    protocol=json.loads(path.read_text())
    protocol.update(stage_architecture_selection=decision,objective=stage['losses']['L_FOCAL'],data_sha256='shared_frame')
    protocol['config']['stage']={'wavelet_selection':choice}
    protocol['config']['training']['loss']='focal'
    path.write_text(json.dumps(protocol))
    path.parent.rename(tmp_path/'runs/L_FOCAL_fixture')
    selected=stages.finalize_losses(tmp_path,stage,decision)
    assert selected['winner']=='L_FOCAL'
