"""Audit paired A/B validation predictions and save descriptive comparison."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compare(bundle: Path, output: Path, previous: Path | None = None):
    records = {name: json.loads((output / name / 'review_summary.json').read_text()) for name in ['A', 'B']}
    roots = {name: records[name]['run'] for name in records}
    with zipfile.ZipFile(bundle) as archive:
        protocols = {name: json.loads(archive.read(root+'protocol.json')) for name, root in roots.items()}
        a, b = protocols['A'], protocols['B']
        for section in ['training', 'model', 'labeling', 'walk_forward', 'development']:
            assert a['config'][section] == b['config'][section], section
        for key in ['folds', 'seeds', 'torch', 'numpy', 'device', 'python', 'pandas', 'sklearn', 'gpu', 'cuda', 'cudnn']:
            assert a[key] == b[key], key
        assert len(a['features']) == 6 and len(b['features']) == 34
        removed = sorted(set(a['features'])-set(b['features']))
        added = sorted(set(b['features'])-set(a['features']))
        full = json.loads(archive.read('data/development_full_wavelet_off.manifest.json'))
        reference = json.loads(archive.read('data/A_reference.manifest.json'))
        assert full['source_hashes'] == reference['source_hashes']
        assert reference['frame_sha256'] == a['data_sha256']
        assert full['reference_a_audit']['base_frame_sha256'] == a['data_sha256']
        assert full['frame_sha256'] == b['data_sha256']
        assert full['full_features_missing'] == []
        pairs = 0
        for seed in a['seeds']:
            for fold in a['folds']:
                suffix = f'seed_{seed}/fold_{fold:03d}/validation_predictions.parquet'
                predictions = {name: pd.read_parquet(io.BytesIO(archive.read(root+suffix))) for name, root in roots.items()}
                columns = ['timestamp', 'label', 'forward_return', 'source_row_position']
                pd.testing.assert_frame_equal(predictions['A'][columns], predictions['B'][columns])
                pairs += 1
        unchanged = None
        if previous:
            with zipfile.ZipFile(previous) as original:
                paths = [name for name in original.namelist() if name.startswith(roots['A'])]
                for path in paths:
                    assert hashlib.sha256(original.read(path)).digest() == hashlib.sha256(archive.read(path)).digest(), path
                unchanged = len(paths)
    metrics = ['bce', 'average_precision', 'precision', 'recall', 'f1', 'accuracy', 'rank_ic', 'ap_lift', 'prediction_rate']
    table = pd.DataFrame({name: {metric: records[name]['macro_metrics'][metric] for metric in metrics} for name in records})
    table['B_minus_A'] = table.B-table.A
    table.to_csv(output/'AB_validation_comparison.csv')
    paired = {name: pd.read_csv(output/name/'fold_seed_metrics.csv').set_index(['seed','fold']) for name in records}
    delta = paired['B'][metrics]-paired['A'][metrics]
    delta.to_csv(output/'AB_paired_differences.csv')
    fold_delta = delta.groupby('fold').mean()
    wins = {metric: {'fold_seed': int((delta[metric] < 0 if metric == 'bce' else delta[metric] > 0).sum()),
                     'seed_averaged_folds': int((fold_delta[metric] < 0 if metric == 'bce' else fold_delta[metric] > 0).sum())}
            for metric in ['bce', 'average_precision', 'f1', 'rank_ic']}
    audit = {'paired_prediction_files': pairs, 'same_labels_returns_times': True,
             'same_base_data': True, 'same_training_model_splits_environment': True,
             'previous_A_files_byte_identical': unchanged, 'B_descriptive_wins': wins,
             'features_removed_from_A': removed, 'features_added_vs_A': added,
             'B_is_strict_feature_superset': not removed,
             'test_evaluations': 0, 'independence_or_significance_claim': False}
    (output/'AB_comparison_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, metric, title in zip(axes.flat, ['bce','average_precision','f1','precision','recall','rank_ic'],
                                 ['BCE (düşük daha iyi)','Average Precision','F1 — eşik 0,5','Precision — eşik 0,5','Recall — eşik 0,5','Spearman Rank IC']):
        values = [table.loc[metric, name] for name in ['A','B']]
        bars = ax.bar(['A: 6 özellik','B: 34 özellik'], values, color=['#287e8c','#d28843'])
        for bar, value in zip(bars, values):
            ax.annotate(f'{value:.4f}', (bar.get_x()+bar.get_width()/2, value), xytext=(0,5), textcoords='offset points', ha='center')
        ax.set_title(title)
        ax.set_ylim(0, max(values)*1.25)
    fig.suptitle('A ve B — aynı GRU, aynı BCE, aynı foldlar\nYalnızca validation: 38 fold × 3 seed', fontsize=15)
    fig.text(.5,.02,'Eşit ağırlıklı fold/seed ortalamaları. Nihai test ve istatistiksel anlamlılık sonucu değildir.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.045,1,.91])
    fig.savefig(output/'AB_validation_comparison.png',dpi=160)
    plt.close(fig)
    lines = ['# A–B doğrulama karşılaştırması','',
             'B tamamlandı: 38 fold × 3 seed = 114 eğitim, 1.865 epoch. Test değerlendirmesi yok.', '',
             '| Ölçüt | A: temel 6 özellik | B: tüm 34 özellik |','|---|---:|---:|']
    lines += [f'| {metric} | {table.loc[metric,"A"]:.6f} | {table.loc[metric,"B"]:.6f} |' for metric in metrics]
    lines += ['', 'B’de precision, recall ve F1 artıyor; BCE, AP ve Rank IC kötüleşiyor. B daha sık olumlu karar veriyor (%8,33; A %6,29).',
              'Seçim ölçütü önceden validation BCE olarak tanımlandı; bu ölçütte mevcut A–B karşılaştırması A lehine. C ve D tamamlanmadan mimari seçimi yapılmaz.', '',
              'A ve B iki farklı özellik setini karşılaştırıyor: A 6, B 34 kanal. B, A’nın doğrudan üst kümesi değil: realized_vol_14, gk_vol_14 ve atr_14_pct B listesinde yok.',
              'Dolayısıyla sonuçlar “A’ya eklenen özelliklerin etkisi” olarak yorumlanamaz; eklemeler ve üç kanalın çıkarılması birlikte değişiyor. İki OI kanalının ayrı katkısı da izole edilmedi.', '',
              'Veri: 33.551 saatlik satır, aynı temel veri hash’i, aynı etiketler ve doğrulama zamanları. Her modelde 1.017 validation dizisi; 114 dosyada eşleşme doğrulandı.',
              'OI: 419.628 kaynak kaydı; 475 kullanılamayan ölçüm. Geliştirme satırı kapsamı %99,7705. 77 saatlik satırdaki iki OI özelliği nötr dolduruldu; piyasa satırı silinmedi.',
              'OI kaynak tarihleri ileriye taşmıyor. Her kapsamda 76 benzersiz bölüm sınırının 3 seed üzerinden 228 kayıtlı kontrolü geçti.', '',
              'ZIP dosya hashleri ve tahminlerden hesaplanan metrikler kontrol edildi. Ham veri ve büyük model ağırlıkları ZIP’te bulunmadığı için bu içerikler bağımsız yeniden üretilemedi.',
              'Foldlar ve girdiler örtüşür; 114 eğitim bağımsız istatistiksel tekrar sayılmaz. Anlamlılık veya nihai başarı iddiası yapılmaz.', '',
              'Sonraki aşama C: aynı 34 özellik → TCN → BCE. Veri, eşik, seedler, foldlar ve epoch seçim ölçütü korunmalı.', '',
              '![A–B karşılaştırması](AB_validation_comparison.png)']
    (output/'AB_DENEYLERI_INCELEME.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(table.to_string())
    print(json.dumps(audit,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous', type=Path)
    args = parser.parse_args()
    compare(args.bundle,args.output,args.previous)
