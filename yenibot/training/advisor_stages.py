"""Wavelet/loss development stages, with immutable validation-only selections.

The first-stage advisor runner is kept unchanged. No test data are read here.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.preprocessing import RobustScaler
from torch import nn
from torch.utils.data import DataLoader

from yenibot.training import advisor
from yenibot.training.advisor import (atomic_json, atomic_checkpoint, atomic_text, audit_boundary,
    assert_hourly, evaluate, exclusive_run, read_config, restore_rng, rng_state, seed_everything, sha256)
from yenibot.training.advisor_models import build_advisor_model
from yenibot.training.advisor_objectives import AdvisorObjective
from yenibot.training.dataset import SequenceDataset
from yenibot.training.walk_forward import PurgedWalkForwardCV
from yenibot.training.advisor_holdout import audit_reservation
from yenibot.features.builder import _log_return, _rolling_zscore
from yenibot.features.wavelet import causal_wavelet_denoise


def runtime_identity():
    return {'torch':torch.__version__, 'numpy':np.__version__, 'device':'cuda' if torch.cuda.is_available() else 'cpu',
            'python':sys.version, 'pandas':pd.__version__, 'sklearn':sklearn.__version__,
            'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'cuda':torch.version.cuda, 'cudnn':torch.backends.cudnn.version()}


def immutable_json(path, value):
    if path.exists():
        if json.loads(path.read_text())!=value:
            raise ValueError(f'Frozen stage selection changed: {path}; use the original code/config for resume')
    else:
        atomic_json(path,value)


def completed_scope(directory, expected_folds, expected_seeds):
    protocol = json.loads((directory/'protocol.json').read_text())
    status = json.loads((directory/'status.json').read_text())
    expected = {(seed,fold) for seed in expected_seeds for fold in range(expected_folds)}
    actual = [(row['seed'],row['fold']) for row in status['completed']]
    table = pd.read_csv(directory/'validation_summary.csv')
    if (status['status']!='complete' or len(actual)!=len(expected) or set(actual)!=expected or
        set(zip(table.seed,table.fold))!=expected or len(table)!=len(expected) or
        protocol['seeds']!=expected_seeds or protocol['folds']!=list(range(expected_folds))):
        raise ValueError(f'Incomplete/changed fold-seed scope: {directory}')
    if status['test_evaluations']!=0 or protocol['test_evaluations']!=0 or not np.isfinite(table.bce).all():
        raise ValueError('Stage choices require finite validation-only results')
    for item in status['completed']:
        score = table.loc[(table.seed==item['seed'])&(table.fold==item['fold']),'bce'].iloc[0]
        if abs(score-item['validation']['bce'])>1e-7:
            raise ValueError('Status and summary BCE differ')
        result = json.loads((directory/f"seed_{item['seed']}"/f"fold_{item['fold']:03d}"/'validation_metrics.json').read_text())
        if result['signature']!=status['signature'] or abs(result['validation']['bce']-score)>1e-7:
            raise ValueError('Fold result identity/score mismatch')
    return {'run':directory.name,'signature':status['signature'],'mean_bce':float(table.bce.mean()),
            'protocol_sha256':sha256(directory/'protocol.json')},protocol


def select_architecture(root, stage):
    rule = stage['selection']
    references, protocols = {},{}
    for name,run in stage['reference_runs'].items():
        references[name],protocols[name] = completed_scope(root/'runs'/run,rule['expected_folds'],rule['expected_seeds'])
    winner = min(rule['architecture_tie_order'],key=lambda name:references[name]['mean_bce'])
    if winner!=rule['expected_architecture_winner']:
        raise ValueError('Architecture winner differs from reviewed A-D decision')
    base = protocols[winner]
    for name,protocol in protocols.items():
        for key in ('training','model','development','labeling','walk_forward'):
            if base['config'][key]!=protocol['config'][key]:
                raise ValueError(f'A-D common protocol differs: {name}/{key}')
    for key,value in runtime_identity().items():
        if base[key]!=value:
            raise ValueError(f'Cached baseline environment differs ({key}); use its original runtime or a new paired-control protocol')
    if base['config']['training']['loss']!='bce' or base['config']['data']['wavelet']:
        raise ValueError('Expected unfiltered BCE control')
    decision = {'protocol':stage['protocol'],'references':references,'winner':winner,
                'features':base['features'],'architecture':base['config']['experiments'][winner]['architecture'],
                'stage_config':stage,'test_evaluations':0}
    immutable_json(root/'data/stages_architecture_selection.json',decision)
    return decision,base


def add_wavelet_channels(base, raw, feature_config, wavelet):
    settings = {key:wavelet[key] for key in ['window','wavelet','level','threshold_scale']}
    close = causal_wavelet_denoise(raw.close,**settings)
    volume = causal_wavelet_denoise(raw.volume,**settings)
    window = feature_config['features']['stationarity']['normalization_window']
    channels = pd.DataFrame({'timestamp':pd.to_datetime(raw.timestamp,utc=True),
        'close_denoised_log_return':_log_return(close),
        'volume_denoised_log_zscore':_rolling_zscore(np.log1p(volume.clip(lower=0)),window)})
    result = base.merge(channels,on='timestamp',how='left',validate='one_to_one',sort=False)
    pd.testing.assert_frame_equal(base,result[base.columns])
    transformed = list(wavelet['replacements'].values())
    if result[transformed].replace([np.inf,-np.inf],np.nan).isna().any().any():
        raise ValueError('Wavelet warmup/non-finite channels: do not silently shorten data')

    return result


def run(config, name):
    stage,root = config['stage']['specification'],Path(config['output']).parent
    decision,base_protocol = select_architecture(root,stage)
    if config['stage']['architecture_selection']!=decision:
        raise ValueError('Runtime stage architecture selection changed')
    for key in ('training','model','development','labeling','walk_forward'):
        if config[key]!=base_protocol['config'][key]:
            raise ValueError(f'Stage common configuration differs from cached control: {key}')
    if config['training']['selection_metric']!='validation_bce':
        raise ValueError('All objectives require common validation BCE epoch selection')
    phase = config['stage']['phase']
    if phase=='wavelet' and name=='W_ON':
        enabled,spec = True,{'kind':'bce'}
    elif phase=='loss' and name in stage['losses']:
        choice = select_wavelet(root,stage,decision)
        if config['stage'].get('wavelet_selection')!=choice:
            raise ValueError('Runtime wavelet selection differs')
        enabled,spec = choice['winner']=='W_ON',stage['losses'][name]
    else:
        raise ValueError('Unsupported experiment/phase; cached BCE controls are not retrained')
    features = features_for(decision['features'],stage['wavelet'],enabled)
    architecture = decision['architecture']
    frame_path = Path(config['data']['frame'])
    metadata = json.loads(frame_path.with_suffix('.manifest.json').read_text())
    data_hash = sha256(frame_path)
    if data_hash!=metadata['frame_sha256']:
        raise ValueError('Stage data hash changed')
    if phase=='loss' and enabled:
        selected_protocol = json.loads((root/'runs'/choice['candidates']['W_ON']['run']/'protocol.json').read_text())
        if data_hash!=selected_protocol['data_sha256']:
            raise ValueError('Loss input frame differs from selected wavelet control')
    for key in ('data','development','labeling'):
        if config[key]!=metadata['protocol'][key]:
            raise ValueError(f'Stage preparation/runtime contract differs: {key}')
    if metadata['preparation_identity']['base_frame_sha256']!=base_protocol['data_sha256']:
        raise ValueError('Stage base frame does not match selected control')
    holdout = read_config(config['stage']['holdout_config'])
    audit_reservation(holdout)
    if pd.Timestamp(metadata['end'])+pd.Timedelta(hours=config['labeling']['max_holding_bars'])>=pd.Timestamp(holdout['test']['start']):
        raise ValueError('Reserved test period in stage frame')
    frame = pd.read_parquet(frame_path)
    assert_hourly(frame)
    required = features+['label',f"fwd_return_{config['labeling']['max_holding_bars']}h"]
    if frame[required].replace([np.inf,-np.inf],np.nan).isna().any().any() or not set(frame.label.unique()).issubset({0,1}):
        raise ValueError('Invalid stage inputs or targets')
    folds = list(PurgedWalkForwardCV(**config['walk_forward'],label_horizon_bars=config['labeling']['max_holding_bars']).split(len(frame)))
    if len(folds)!=stage['selection']['expected_folds'] or config['training']['seeds']!=stage['selection']['expected_seeds']:
        raise ValueError('Stage fold/seed scope changed')
    experiment = {'architecture':architecture,'objective':spec}
    effective = copy.deepcopy(config)
    effective['training']['loss'] = spec['kind']
    effective['experiments'][name] = {'architecture':architecture,'features':'selected_fixed_dimension'}
    files = [Path(__file__),Path(advisor.__file__),Path(__file__).with_name('advisor_objectives.py'),
             Path(__file__).with_name('advisor_models.py'),Path(__file__).with_name('dataset.py'),
             Path(__file__).with_name('walk_forward.py'),Path(__file__).parents[1]/'losses.py',
             Path(__file__).parents[1]/'models/tcn.py',Path(__file__).parents[1]/'models/hybrid.py']
    protocol = {'config':effective,'experiment':name,'features':features,'data_sha256':data_hash,
        'source_hashes':{str(path):sha256(path) for path in files},'folds':[fold.fold for fold in folds],
        'seeds':config['training']['seeds'],**runtime_identity(),'selection':'validation_bce_only',
        'stage_architecture_selection':decision,'objective':spec,'wavelet_enabled':enabled,
        'role':'advisor_development_stage','test_evaluations':0}
    signature = hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    directory = Path(config['output'])/f'{name}_{signature[:12]}'
    torch.set_num_threads(config['training']['torch_threads'])
    device = torch.device(protocol['device'])
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    with exclusive_run(directory):
        immutable_json(directory/'protocol.json',protocol)
        progress = {'status':'running','experiment':name,'signature':signature,'total_folds':len(folds),
                    'seeds':protocol['seeds'],'completed':[],'test_evaluations':0}
        try:
            for seed in protocol['seeds']:
                for fold in folds:
                    progress['current'] = {'seed':seed,'fold':fold.fold}
                    atomic_json(directory/'status.json',progress)
                    atomic_text(directory/'STATUS.md',f"# {name}: çalışıyor\n\n{len(progress['completed'])}/114 tamamlandı.\n\nSeed {seed}, fold {fold.fold}; epoch kaydı alt klasörde.\n\nTest değerlendirmesi: 0.\n")
                    result = run_stage_fold(frame,fold,effective,experiment,features,seed,
                        directory/f'seed_{seed}'/f'fold_{fold.fold:03d}',signature,device)
                    progress['completed'].append({'seed':seed,'fold':fold.fold,'validation':result['validation'],'best_epoch':result['best_epoch']})
                    atomic_json(directory/'status.json',progress)
                    pd.DataFrame([{**item['validation'],'seed':item['seed'],'fold':item['fold'],'best_epoch':item['best_epoch']}
                        for item in progress['completed']]).to_csv(directory/'validation_summary.csv',index=False)
            progress['status'] = 'complete'
            progress.pop('current',None)
            atomic_json(directory/'status.json',progress)
            atomic_text(directory/'STATUS.md',f'# {name}: tamamlandı\n\n114 fold/seed. Yalnızca validation.\n')
        except BaseException as exc:
            progress.update(status='interrupted_or_failed',error=repr(exc))
            atomic_json(directory/'status.json',progress)
            atomic_text(directory/'STATUS.md',f"# {name}: kesildi\n\n{progress.get('current')}\n\n{exc!r}\n\nCheckpoint korunuyor; aynı ortam/kodla devam edin.\n")
            raise
    if name=='W_ON':
        ref = {'run':directory.name,'signature':signature}
        immutable_json(root/'data/stages_wavelet_run.json',ref)
        choice = select_wavelet(root,stage,decision)
        print('Wavelet selection:',json.dumps(choice['candidates']),choice['winner'],flush=True)
    print('Stage complete:',directory,'validation only; test evaluations: 0',flush=True)


def finalize_losses(root,stage,decision):
    choice = select_wavelet(root,stage,decision)
    candidates = {'L_BCE':choice['candidates'][choice['winner']]}
    base = json.loads((root/'runs'/decision['references'][decision['winner']]['run']/'protocol.json').read_text())
    hashes = set()
    for name in stage['losses']:
        matches = []
        for path in (root/'runs').glob(name+'_*'):
            try:
                score,protocol = completed_scope(path,stage['selection']['expected_folds'],stage['selection']['expected_seeds'])
            except (FileNotFoundError,ValueError):
                continue
            expected_training = copy.deepcopy(base['config']['training'])
            expected_training['loss'] = stage['losses'][name]['kind']
            if (protocol.get('stage_architecture_selection')==decision and
                protocol['config'].get('stage',{}).get('wavelet_selection')==choice and
                protocol.get('objective')==stage['losses'][name] and
                protocol['config']['training']==expected_training and
                all(protocol['config'][key]==base['config'][key]
                    for key in ('model','development','labeling','walk_forward')) and
                all(protocol.get(key)==value for key,value in runtime_identity().items())):
                matches.append(score)
                hashes.add(protocol['data_sha256'])
        if len(matches)!=1:
            raise ValueError(f'Need exactly one complete consistent loss scope for {name}; found {len(matches)}')
        candidates[name] = matches[0]
    if len(hashes)!=1:
        raise ValueError('Loss runs must use one common prepared input frame')
    winner = min(stage['loss_tie_order'],key=lambda name:candidates[name]['mean_bce'])
    selection = {'wavelet_selection':choice,'candidates':candidates,'winner':winner,'test_evaluations':0}
    immutable_json(root/'data/stages_loss_selection.json',selection)
    print('Loss selection:',json.dumps(candidates),winner,flush=True)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','finalize'])
    parser.add_argument('--config',required=True)
    parser.add_argument('--experiment')
    args = parser.parse_args()
    config = read_config(args.config)
    if args.action=='prepare':
        prepare_data(config)
    elif args.action=='run':
        run(config,args.experiment)
    else:
        root = Path(config['output']).parent
        decision,_ = select_architecture(root,config['stage']['specification'])
        finalize_losses(root,config['stage']['specification'],decision)


def prepare_data(config):
    """Reconstruct A exactly, add only two channels, keep raw labels untouched."""
    target = Path(config['data']['frame'])
    reference = json.loads(Path(config['stage']['reference_manifest']).read_text())
    stage = config['stage']['specification']
    import pywt
    current_features = copy.deepcopy(read_config(config['data']['feature_config']))
    current_features['features']['wavelet']['enabled'] = False
    current_features['features']['active_profile'] = reference['feature_config']['features']['active_profile']
    if current_features!=reference['feature_config']:
        raise ValueError('Feature config differs from A reference')
    holdout = read_config(config['stage']['holdout_config'])
    audit_reservation(holdout)
    end = pd.Timestamp(config['development']['end'])+pd.Timedelta(hours=config['labeling']['max_holding_bars'])
    if end>=pd.Timestamp(holdout['test']['start']):
        raise ValueError('Development data enter reserved holdout')
    identity = {'base_frame_sha256':reference['frame_sha256'],'source_hashes':reference['source_hashes'],
                'wavelet':stage['wavelet'],'feature_config':reference['feature_config'],
                'preparer_sha256':sha256(Path(__file__)),
                'pywavelets':pywt.__version__,
                'builder_sha256':sha256(Path(__file__).parents[1]/'features/builder.py'),
                'wavelet_source_sha256':sha256(Path(__file__).parents[1]/'features/wavelet.py')}
    manifest = target.with_suffix('.manifest.json')
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if saved['preparation_identity']!=identity or sha256(target)!=saved['frame_sha256']:
            raise ValueError('Stage frame source/config/hash changed; do not overwrite the frozen frame')
        snapshot = Path(config['data']['snapshot'])
        if any(sha256(snapshot/f'btc_{interval}.parquet')!=digest for interval,digest in reference['source_hashes'].items()):
            raise ValueError('Cached stage frame raw source differs from A')
        print('Stage frame reused and verified:',target,flush=True)
        return
    base_cfg = copy.deepcopy(config)
    base_cfg.pop('stage')
    probe = target.with_name('stages_base_probe.parquet')
    base_cfg['data']['frame'] = str(probe)
    advisor.prepare(base_cfg)
    if sha256(probe)!=reference['frame_sha256']:
        raise ValueError('Reconstructed base does not match A; do not rerun A silently')
    base_meta = json.loads(probe.with_suffix('.manifest.json').read_text())
    if base_meta['source_hashes']!=reference['source_hashes']:
        raise ValueError('Stage raw inputs differ from A')
    base = pd.read_parquet(probe)
    raw = pd.read_parquet(Path(config['data']['snapshot'])/'btc_1h.parquet')
    raw.timestamp = pd.to_datetime(raw.timestamp,utc=True)
    raw = raw.loc[raw.timestamp<=end].reset_index(drop=True)
    frame = add_wavelet_channels(base,raw,base_meta['feature_config'],stage['wavelet'])
    target.parent.mkdir(parents=True,exist_ok=True)
    temporary = target.with_suffix('.tmp.parquet')
    frame.to_parquet(temporary,index=False)
    os.replace(temporary,target)
    atomic_json(manifest,{'rows':len(frame),'start':frame.timestamp.iloc[0].isoformat(),
        'end':frame.timestamp.iloc[-1].isoformat(),'frame_sha256':sha256(target),
        'protocol':config,'feature_config':base_meta['feature_config'],
        'source_hashes':base_meta['source_hashes'],'raw_quality_audits':base_meta['raw_quality_audits'],
        'preparation_identity':identity,'wavelet_audit':{'base_columns_unchanged':True,
        'labels_from_raw_prices':True,'added_channels':list(stage['wavelet']['replacements'].values()),
        'settings':stage['wavelet'],'future_rows_used':False},'test_evaluations':0})
    probe.unlink()
    probe.with_suffix('.manifest.json').unlink()
    print(f'Stage frame prepared: {len(frame)} unchanged base rows; two extra wavelet channels',flush=True)


def features_for(base_features, wavelet, enabled):
    return [wavelet['replacements'].get(name,name) if enabled else name for name in base_features]


def select_wavelet(root,stage,decision):
    path = root/'data/stages_wavelet_run.json'
    if not path.exists():
        raise ValueError('Run 08 wavelet notebook to completion before the loss stage')
    reference = json.loads(path.read_text())
    on,protocol = completed_scope(root/'runs'/reference['run'],stage['selection']['expected_folds'],stage['selection']['expected_seeds'])
    if on['signature']!=reference['signature'] or protocol['experiment']!='W_ON':
        raise ValueError('Wavelet run reference changed')
    if protocol['stage_architecture_selection']!=decision:
        raise ValueError('Wavelet run used a different architecture decision')
    for key,value in runtime_identity().items():
        if protocol[key]!=value:
            raise ValueError(f'Wavelet control environment differs ({key})')
    off = decision['references'][decision['winner']]
    candidates = {'W_OFF':off,'W_ON':on}
    winner = min(stage['selection']['wavelet_tie_order'],key=lambda name:candidates[name]['mean_bce'])
    result = {'candidates':candidates,'winner':winner,'architecture_selection':decision,'test_evaluations':0}
    immutable_json(root/'data/stages_wavelet_selection.json',result)
    return result


def run_stage_fold(frame: pd.DataFrame, fold, config: dict, experiment: dict, features: list[str],
             seed: int, directory: Path, signature: str, device: torch.device) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "validation_metrics.json"
    checkpoint_path = directory / "last.pt"
    if result_path.exists():
        completed = json.loads(result_path.read_text())
        if completed["signature"] != signature:
            raise ValueError("Completed fold signature mismatch; use a new run directory")
        return completed
    horizon = config["labeling"]["max_holding_bars"]
    audit = {"train_validation": audit_boundary(frame, fold.train, fold.val, horizon),
             "validation_test_section": audit_boundary(frame, fold.val, fold.test, horizon)}
    train = frame.iloc[fold.train].copy()
    val = frame.iloc[fold.val].copy()
    scaler = RobustScaler().fit(train[features])
    train.loc[:, features] = scaler.transform(train[features])
    val.loc[:, features] = scaler.transform(val[features])
    seq_len = config["model"]["seq_len"]
    forward = f"fwd_return_{horizon}h"
    def dataset(part):
        return SequenceDataset(part[features].to_numpy(np.float32), part.label.to_numpy(np.float32),
                               part[forward].to_numpy(np.float32), seq_len=seq_len)
    batch = config["training"]["batch_size"]
    train_loader = DataLoader(dataset(train), batch_size=batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset(val), batch_size=batch, shuffle=False, num_workers=0)
    seed_everything(seed + fold.fold, config["training"]["deterministic"])
    model = build_advisor_model(len(features), experiment["architecture"], config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                                 weight_decay=config["training"]["weight_decay"])
    objective = AdvisorObjective(experiment['objective'],
        torch.tensor(train_loader.dataset.labels[train_loader.dataset.end_positions])).to(device)
    history, best_state = [], None
    best_loss, patience, start_epoch, best_epoch = float("inf"), 0, 0, 0
    if checkpoint_path.exists():
        # Only self-created, local checkpoints are loaded; never untrusted weights.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["signature"] != signature or checkpoint["device"] != str(device):
            raise ValueError("Checkpoint protocol/data/source/device mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        history, best_state = checkpoint["history"], checkpoint["best_state"]
        best_loss, patience = checkpoint["best_loss"], checkpoint["patience"]
        start_epoch, best_epoch = checkpoint["epoch"], checkpoint["best_epoch"]
        restore_rng(checkpoint["rng"])
        print(f"Resume fold {fold.fold}, seed {seed}, after epoch {start_epoch}", flush=True)
    for epoch in range(start_epoch, config["training"]["epochs"]):
        if patience >= config["training"]["patience"]:
            break
        model.train()
        total_loss, count = 0.0, 0
        for x, y, forward_returns, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model(x, return_logits=True), y, forward_returns.to(device))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip"])
            optimizer.step()
            total_loss += float(loss.detach()) * len(y)
            count += len(y)
        validation, _ = evaluate(model, val_loader, device, config["training"]["threshold"])
        if not np.isfinite(validation["bce"]):
            raise ValueError("Non-finite validation BCE")
        if validation["bce"] < best_loss:
            best_loss, best_epoch, patience = validation["bce"], epoch + 1, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            patience += 1
        history.append({"epoch": epoch + 1, "train_loss": total_loss / count, "validation": validation})
        atomic_checkpoint(checkpoint_path, {"signature": signature, "device": str(device), "epoch": epoch + 1,
                          "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "best_state": best_state, "best_loss": best_loss, "best_epoch": best_epoch,
                          "patience": patience, "history": history, "rng": rng_state(),
                          "scaler_center": scaler.center_, "scaler_scale": scaler.scale_, "features": features})
        atomic_json(directory / "progress.json", {"status": "training", "fold": fold.fold, "seed": seed,
                    "last_saved_epoch": epoch + 1, "best_epoch": best_epoch, "best_validation_bce": best_loss,
                    "updated_at": datetime.now(timezone.utc).isoformat(), "signature": signature})
        atomic_text(directory / "STATUS.md", f"# Eğitim durumu\n\nFold: {fold.fold}; seed: {seed}.\n\n"
                    f"Son kaydedilen epoch: {epoch + 1}; en iyi epoch: {best_epoch}.\n\n"
                    f"En iyi validation BCE: {best_loss:.6f}. Test değerlendirmesi: 0.\n\n"
                    "İşlem kesilirse aynı komutla son kaydedilen epoch'tan devam edin.\n")
        print(f"fold={fold.fold} seed={seed} epoch={epoch+1} train_objective={total_loss/count:.6f} val_BCE={validation['bce']:.6f} best={best_epoch}", flush=True)
    model.load_state_dict(best_state)
    validation, predictions = evaluate(model, val_loader, device, config["training"]["threshold"])
    predictions["timestamp"] = val.iloc[predictions.source_row_position].timestamp.to_numpy()
    predictions.to_parquet(directory / "validation_predictions.parquet", index=False)
    result = {"status": "complete", "signature": signature, "fold": fold.fold, "seed": seed,
              "best_epoch": best_epoch, "epochs_trained": len(history), "validation": validation,
              "boundary_audit": audit, "objective_audit": objective.audit, "parameters": sum(p.numel() for p in model.parameters()),
              "train_rows": len(train), "train_sequences": len(train_loader.dataset),
              "validation_rows": len(val), "features": features, "test_evaluations": 0}
    atomic_json(result_path, result)
    atomic_json(directory / "progress.json", result)
    atomic_text(directory / "STATUS.md", f"# Fold tamamlandı\n\nFold: {fold.fold}; seed: {seed}.\n\n"
                f"Eğitilen epoch: {len(history)}; seçilen epoch: {best_epoch}.\n\n"
                f"Validation BCE: {validation['bce']:.6f}. Test değerlendirmesi: 0.\n")
    pd.DataFrame([{"epoch": row["epoch"], "train_loss": row["train_loss"], **row["validation"]} for row in history]).to_csv(directory / "history.csv", index=False)
    return result



if __name__ == "__main__":
    main()
