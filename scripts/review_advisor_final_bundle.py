"""Read-only final ZIP audit. Recompute scores; never load serialized weights."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_fscore_support


def review(bundle,output):
    with zipfile.ZipFile(bundle) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts for n in names):
            raise ValueError('Unexpected final ZIP paths')
        manifest=json.loads(z.read('review_manifest.json'))
        if manifest['role']!='final_holdout_scored' or manifest['test_evaluations']!=1:
            raise ValueError('Final test scope incomplete')
        for item in manifest['files']:
            value=z.read(item['path'])
            if len(value)!=item['size'] or hashlib.sha256(value).hexdigest()!=item['sha256']:
                raise ValueError('Final ZIP inventory checksum mismatch')
        read=lambda path:json.loads(z.read(path))
        plan=read('data/final_plan.json');config=plan['config'];reservation=plan['reservation']
        freeze=read('artifacts/frozen_manifest.json');results=read('runs/test/test_results.json')
        if hashlib.sha256(z.read('artifacts/frozen_manifest.json')).hexdigest()!=results['identity']['freeze_sha256']:
            raise ValueError('Frozen artifact manifest changed')
        contract_sha=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
        if freeze['contract_sha256']!=contract_sha or results['identity']['contract_sha256']!=contract_sha:
            raise ValueError('Final contract identity mismatch')
        if results['test_fit_operations']!=0 or results['test_evaluations']!=1:
            raise ValueError('Final test fitting/evaluation contract violated')
        expected=pd.Series(pd.date_range(reservation['test']['start'],reservation['test']['end'],freq='h'),name='timestamp')
        if len(expected)!=720 or [row['seed'] for row in results['per_seed']]!=config['training']['seeds']:
            raise ValueError('Final seed/date scope changed')
        rows=[];truth=None
        for recorded in results['per_seed']:
            seed=recorded['seed'];path=f'runs/test/seed_{seed}_test_predictions.parquet'
            predictions=pd.read_parquet(io.BytesIO(z.read(path)))
            if not pd.to_datetime(predictions.timestamp,utc=True).equals(expected):
                raise ValueError('Final test timestamps differ')
            if truth is None:truth=predictions[['timestamp','label','forward_return','source_row_position']]
            else:pd.testing.assert_frame_equal(truth,predictions[truth.columns])
            y=predictions.label.to_numpy();p=predictions.probability.to_numpy().astype(float)
            r=predictions.forward_return.to_numpy()
            if not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not set(y).issubset({0,1}):
                raise ValueError('Invalid final predictions')
            precision,recall,f1,_=precision_recall_fscore_support(y,p>=config['training']['threshold'],average='binary',zero_division=0)
            q=np.clip(p,1e-12,1-1e-12)
            ic=float(spearmanr(p,r).statistic) if np.ptp(p)>0 and np.ptp(r)>0 else None
            calculated={'bce':float(-np.mean(y*np.log(q)+(1-y)*np.log1p(-q))),
                'average_precision':float(average_precision_score(y,p)) if np.any(y==1) else None,
                'precision':float(precision),'recall':float(recall),'f1':float(f1),
                'accuracy':float(accuracy_score(y,p>=config['training']['threshold'])),
                'rank_ic':ic,'prevalence':float(y.mean())}
            for key,value in calculated.items():
                if value is None:
                    if recorded[key] is not None:raise ValueError('Undefined metric differs')
                elif abs(value-recorded[key])>1e-6:raise ValueError(f'Test metric mismatch: {seed}/{key}')
            if recorded['samples']!=720 or recorded['threshold']!=config['training']['threshold']:
                raise ValueError('Final count/threshold differs')
            # Verify pre-test epoch selection against validation history.
            history=pd.read_csv(io.BytesIO(z.read(f'runs/final/seed_{seed}/history.csv')))
            selected=next(item for item in freeze['artifacts'] if item['seed']==seed)
            if int(history.loc[history.bce.idxmin(),'epoch'])!=selected['best_epoch']:
                raise ValueError('Frozen epoch was not selected by validation BCE')
            rows.append({'seed':seed,**calculated})
        table=pd.DataFrame(rows)
        for key,recorded in results['aggregate_mean_std'].items():
            for name,value in [('mean',table[key].mean()),('std',table[key].std(ddof=1))]:
                if pd.isna(value):
                    if recorded[name] is not None:raise ValueError('Undefined aggregate differs')
                elif abs(value-recorded[name])>1e-6:raise ValueError('Final seed aggregation differs')
        prior=freeze['training_positive_rate'];labels=truth.label.to_numpy()
        reference=results['baselines']['training_frequency_probability']
        q=float(np.clip(prior,np.finfo(float).eps,1-np.finfo(float).eps))
        bce=float(-np.mean(labels*np.log(q)+(1-labels)*np.log1p(-q)))
        if abs(reference['bce']-bce)>1e-6:raise ValueError('Training frequency baseline differs')
        if abs(results['baselines']['always_negative_classification']['accuracy']-(1-labels.mean()))>1e-6:
            raise ValueError('Always-negative reference differs')
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        table.to_csv(output/'verified_test_metrics.csv',index=False)
        (output/'review_summary.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
        print(json.dumps(results['aggregate_mean_std'],indent=2))
        return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();review(args.bundle,args.output)
