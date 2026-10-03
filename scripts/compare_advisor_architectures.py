"""Compare audited scopes on matching validation targets; no model selection."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def compare(bundle, output, experiments, previous=None):
    records = {name: json.loads((output/name/'review_summary.json').read_text()) for name in experiments}
    roots = {name: record['run'] for name, record in records.items()}
    with zipfile.ZipFile(bundle) as archive:
        protocols = {name: json.loads(archive.read(root+'protocol.json')) for name, root in roots.items()}
        base = protocols['A']
        for name, protocol in protocols.items():
            for key in ['folds','seeds','torch','numpy','device','python','pandas','sklearn','gpu','cuda','cudnn']:
                assert protocol[key] == base[key], (name,key)
            for section in ['training','model','labeling','walk_forward','development']:
                assert protocol['config'][section] == base['config'][section], (name,section)
            if name != 'A':
                assert protocol['features'] == protocols['B']['features'], name
                assert protocol['data_sha256'] == protocols['B']['data_sha256'], name
        paired_groups = 0
        for seed in base['seeds']:
            for fold in base['folds']:
                suffix = f'seed_{seed}/fold_{fold:03d}/validation_predictions.parquet'
                targets = {name: pd.read_parquet(io.BytesIO(archive.read(root+suffix)))[
                    ['timestamp','label','forward_return','source_row_position']] for name,root in roots.items()}
                for name in experiments:
                    pd.testing.assert_frame_equal(targets['A'],targets[name])
                paired_groups += 1
        unchanged = {}
        if previous:
            with zipfile.ZipFile(previous) as old:
                for name, root in roots.items():
                    paths = [path for path in old.namelist() if path.startswith(root)]
                    for path in paths:
                        assert hashlib.sha256(old.read(path)).digest() == hashlib.sha256(archive.read(path)).digest(), path
                    if paths:
                        unchanged[name] = len(paths)
    keys = ['bce','average_precision','precision','recall','f1','accuracy','rank_ic','prediction_rate']
    table = pd.DataFrame({name:{key:records[name]['macro_metrics'][key] for key in keys} for name in experiments})
    prefix = ''.join(experiments)
    table.to_csv(output/f'{prefix}_validation_comparison.csv')
    tables = {name:pd.read_csv(output/name/'fold_seed_metrics.csv').set_index(['seed','fold']) for name in experiments}
    wins = {}
    for before, after in [('A','C'),('B','C')]:
        if after not in experiments:
            continue
        delta = tables[after][keys]-tables[before][keys]
        delta.to_csv(output/f'{after}_minus_{before}_paired_differences.csv')
        means = delta.groupby('fold').mean()
        wins[f'{after}_vs_{before}'] = {key:{'fold_seed':int((delta[key]<0 if key=='bce' else delta[key]>0).sum()),
            'seed_averaged_folds':int((means[key]<0 if key=='bce' else means[key]>0).sum())}
            for key in ['bce','average_precision','f1','rank_ic']}
    audit = {'paired_groups':paired_groups,'previous_run_files_byte_identical':unchanged,
             'same_targets_splits_environment':True,'full_scopes_same_features_and_data':True,
             'descriptive_wins':wins,'test_evaluations':0,'statistical_significance_claim':False}
    (output/f'{prefix}_comparison_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    descriptions = {'A':'6 / GRU','B':'34 / GRU','C':'34 / TCN','D':'34 / TCN–GRU'}
    for ax,key,title in zip(axes.flat,['bce','average_precision','f1','precision','recall','rank_ic'],
                           ['BCE (düşük daha iyi)','Average Precision','F1 — eşik 0,5','Precision — eşik 0,5','Recall — eşik 0,5','Spearman Rank IC']):
        values=[table.loc[key,name] for name in experiments]
        bars=ax.bar([f'{name}: {descriptions[name]}' for name in experiments],values,color=['#287e8c','#d28843','#607fba','#9b6f9f'][:len(experiments)])
        for bar,value in zip(bars,values):
            ax.annotate(f'{value:.4f}',(bar.get_x()+bar.get_width()/2,value),xytext=(0,5),textcoords='offset points',ha='center',fontsize=10)
        ax.set_title(title)
        ax.set_ylim(min(0,min(values)*1.25),max(values)*1.25)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f'{" – ".join(experiments)}: özellik sayısı / mimari\nAynı BCE, aynı foldlar; yalnızca validation',fontsize=15)
    fig.text(.5,.02,'38 fold × 3 seed ortalamaları. Nihai test ve istatistiksel anlamlılık sonucu değildir.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.045,1,.91])
    fig.savefig(output/f'{prefix}_validation_comparison.png',dpi=160)
    plt.close(fig)
    print(table.to_string())
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--experiments',nargs='+',default=['A','B','C'],choices=['A','B','C','D'])
    parser.add_argument('--previous',type=Path)
    args=parser.parse_args()
    compare(args.bundle,args.output,args.experiments,args.previous)
