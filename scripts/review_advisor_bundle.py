"""Read-only ZIP audit and reproducible validation-only report; never load weights."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, accuracy_score, precision_recall_fscore_support


def review(bundle: Path, output: Path, experiment: str = "A"):
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP members")
        for entry in archive.infolist():
            if PurePosixPath(entry.filename).is_absolute() or ".." in PurePosixPath(entry.filename).parts or entry.file_size > 100_000_000:
                raise ValueError("Unexpected ZIP path or member size")
        manifest = json.loads(archive.read("review_manifest.json"))
        for record in manifest["files"]:
            content = archive.read(record["path"])
            assert len(content) == record["size"]
            assert hashlib.sha256(content).hexdigest() == record["sha256"], record["path"]
        roots = [name[:-len("status.json")] for name in names if name.startswith("runs/") and name.endswith("/status.json")]
        roots = [root for root in roots if json.loads(archive.read(root + 'protocol.json'))['experiment'] == experiment]
        assert len(roots) == 1, "Specify one uniquely identified experimental scope"
        root = roots[0]
        read_json = lambda name: json.loads(archive.read(name))
        read_csv = lambda name: pd.read_csv(io.BytesIO(archive.read(name)))
        protocol = read_json(root + "protocol.json")
        status = read_json(root + "status.json")
        candidates = [read_json(name) for name in names if name.startswith('data/') and name.endswith('.manifest.json')]
        data = next(item for item in candidates if item.get('frame_sha256') == protocol['data_sha256'])
        session_candidates = [read_json(name) for name in names if name.startswith('sessions/') and name.endswith('/session.json')]
        sessions = [item for item in session_candidates if item.get('status') == 'complete' and experiment in item.get('experiments', [])]
        session = max(sessions, key=lambda item: item['session'])
        calendar = read_csv("data/fold_calendar.csv")
        summary = read_csv(root + "validation_summary.csv")
        cfg = protocol["config"]
        expected = {(seed, fold) for seed in protocol["seeds"] for fold in protocol["folds"]}
        assert set(zip(summary.seed, summary.fold)) == expected
        assert not summary.duplicated(["seed", "fold"]).any()
        assert status["status"] == manifest["session"]["status"] == "complete"
        assert len(status["completed"]) == len(expected)
        assert all(value == 0 for value in [status["test_evaluations"], protocol["test_evaluations"], manifest["session"]["test_evaluations"]])
        assert protocol["experiment"] == experiment and cfg["training"]["loss"] == "bce"
        assert cfg["training"]["selection_metric"] == "validation_bce"
        assert not cfg["data"]["wavelet"]
        assert cfg["walk_forward"]["purge_bars"] >= 10 and cfg["walk_forward"]["embargo_bars"] >= 10
        details, unique_timestamps, label_cache = [], set(), {}
        epochs = 0
        boundary_count = 0
        for seed, fold in sorted(expected):
            folder = root + f"seed_{seed}/fold_{fold:03d}/"
            result = read_json(folder + "validation_metrics.json")
            history = read_csv(folder + "history.csv")
            predictions = pd.read_parquet(io.BytesIO(archive.read(folder + "validation_predictions.parquet")))
            assert result["status"] == "complete" and result["test_evaluations"] == 0
            assert result["signature"] == status["signature"]
            assert result["features"] == protocol["features"]
            assert int(history.loc[history.bce.idxmin(), "epoch"]) == result["best_epoch"]
            assert abs(history.bce.min() - result["validation"]["bce"]) < 1e-7
            assert len(history) == result["epochs_trained"]
            assert len(history) == min(cfg['training']['epochs'], result["best_epoch"] + cfg["training"]["patience"])
            epochs += len(history)
            for check in result["boundary_audit"].values():
                assert check["passed"] and pd.Timestamp(check["last_target_timestamp"]) < pd.Timestamp(check["next_section_start"])
                boundary_count += 1
            y = predictions.label.to_numpy()
            probability = predictions.probability.to_numpy()
            fwd = predictions.forward_return.to_numpy()
            assert len(y) == 1017 and set(np.unique(y)).issubset({0, 1})
            assert np.isfinite(probability).all() and np.isfinite(fwd).all()
            assert ((probability >= 0) & (probability <= 1)).all()
            p, r, f1, _ = precision_recall_fscore_support(y, probability >= .5, average="binary", zero_division=0)
            computed = {"average_precision": average_precision_score(y, probability), "precision": p,
                        "recall": r, "f1": f1, "accuracy": accuracy_score(y, probability >= .5),
                        "rank_ic": spearmanr(probability, fwd).statistic}
            for key, value in computed.items():
                assert abs(value - result["validation"][key]) < 1e-6, (seed, fold, key)
            # Probabilities are float32; stable logit BCE itself needs the excluded weights/logits.
            q = np.clip(probability.astype(float), 1e-12, 1 - 1e-12)
            bce_probability = -np.mean(y * np.log(q) + (1-y) * np.log1p(-q))
            assert abs(bce_probability - result["validation"]["bce"]) < 1e-6
            times = pd.to_datetime(predictions.timestamp, utc=True)
            row = calendar.loc[calendar.fold == fold + 1].iloc[0]
            assert times.iloc[0] == pd.Timestamp(row.validation_start) + pd.Timedelta(hours=63)
            assert times.iloc[-1] == pd.Timestamp(row.validation_end)
            assert times.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all()
            assert np.array_equal(predictions.source_row_position, np.arange(63, 1080))
            for timestamp, label, forward in zip(times, y, fwd):
                key = timestamp.isoformat()
                if key in label_cache:
                    assert label_cache[key] == (float(label), float(forward))
                label_cache[key] = (float(label), float(forward))
                unique_timestamps.add(key)
            metric_row = summary.loc[(summary.seed == seed) & (summary.fold == fold)].iloc[0]
            for key in computed:
                assert abs(metric_row[key] - result["validation"][key]) < 1e-6
            assert result["train_rows"] == 5040 and result["train_sequences"] == 4977
            details.append({**metric_row.to_dict(), "prediction_rate": float((probability >= .5).mean()),
                            "ap_lift": float(computed["average_precision"] / y.mean()),
                            "always_negative_accuracy": float(1-y.mean()),
                            "epochs_trained": len(history)})
    table = pd.DataFrame(details)
    table.to_csv(output / "fold_seed_metrics.csv", index=False)
    seed_table = table.groupby("seed").mean(numeric_only=True)
    seed_table.to_csv(output / "seed_summary.csv")
    avg = table.mean(numeric_only=True)
    fold_table = table.groupby("fold").mean(numeric_only=True)
    record = {"bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
              "run": root, "session": session, "audited_inventory_files": len(manifest["files"]),
              "completed_fold_seed": len(table), "epochs_trained": epochs,
              "boundary_checks_in_artifacts": boundary_count, "unique_boundary_checks": len(protocol["folds"])*2,
              "validation_prediction_rows_with_repeats": int(table.samples.sum()),
              "unique_validation_timestamps": len(unique_timestamps),
              "positive_rank_ic_runs": int((table.rank_ic > 0).sum()),
              "positive_rank_ic_seed_averaged_folds": int((fold_table.rank_ic > 0).sum()),
              "zero_positive_prediction_runs": int((table.prediction_rate == 0).sum()),
              "macro_metrics": {key: float(avg[key]) for key in ["bce", "average_precision", "precision", "recall", "f1", "accuracy", "rank_ic", "prevalence", "ap_lift", "prediction_rate", "always_negative_accuracy"]},
              "raw_quality_audits": data["raw_quality_audits"],
              "open_interest_audit": data.get('open_interest_audit'),
              "reference_a_audit": data.get('reference_a_audit'), "test_evaluations": 0}
    (output / "review_summary.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax = axes[0, 0]
    values = [avg.precision, avg.recall, avg.f1, avg.accuracy, avg.always_negative_accuracy]
    ax.barh(["Precision", "Recall", "F1", "Accuracy", "Hep olumsuz: accuracy"], np.array(values)*100,
            color=["#287e8c", "#287e8c", "#287e8c", "#287e8c", "#a9b4bc"])
    for i, value in enumerate(values):
        ax.text(value*100+1, i, f"%{value*100:.2f}", va="center")
    ax.set_xlim(0, 100)
    ax.set_title("Sabit 0,5 eşikte sınıflandırma")
    ax.invert_yaxis()
    x = np.arange(1, 39)
    for ax, metric, title, reference in [(axes[0,1], "ap_lift", "AP / olumlu sınıf oranı", 1),
                                         (axes[1,0], "rank_ic", "İleri getiri sıralaması: Rank IC", 0),
                                         (axes[1,1], "bce", "Seçilen epoch: validation BCE", None)]:
        grouped = table.groupby("fold")[metric]
        ax.fill_between(x, grouped.min(), grouped.max(), color="#287e8c", alpha=.15)
        ax.plot(x, grouped.mean(), color="#287e8c", lw=1.8)
        if reference is not None:
            ax.axhline(reference, color="#9d5b45", ls="--")
        ax.set_title(title)
        ax.set_xlabel("Fold")
        ax.set_xlim(1, 38)
        ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 38])
    description = 'temel fiyat/hacim' if experiment == 'A' else 'tüm wavelet dışı özellikler'
    architecture = cfg['experiments'][experiment]['architecture'].upper().replace('_', '–')
    fig.suptitle(f"{experiment} deneyi — {description} + {architecture} + BCE\nYalnızca validation; 38 fold × 3 seed", fontsize=16)
    fig.text(.5, .015, "Gölge: 3 seed'in min–max aralığı; güven aralığı değildir. Test değerlendirmesi: 0.", ha="center", fontsize=10)
    fig.tight_layout(rect=[0, .045, 1, .93])
    fig.savefig(output / f"{experiment}_validation_dashboard.png", dpi=160)
    plt.close(fig)
    report = f"""# {experiment} deneyi: Colab sonuç incelemesi

Deney tamamlandı: 38 fold × 3 seed = {len(table)} eğitim. Toplam {epochs} epoch.
Kaynak commit: `{session['commit']}`; GPU: {session['gpu']}.
ZIP SHA256: `{record['bundle_sha256']}`.
Bu sonuçların tamamı validation sonuçlarıdır; nihai test değerlendirmesi yapılmadı.

## Dosya ve yöntem denetimi

- Manifestteki {len(manifest['files'])} dosyanın boyut ve SHA256 değerleri doğrulandı.
- Fold/seed kapsamı eksiksiz, yinelenmiş fold/seed kaydı yok.
- Her epoch seçimi minimum validation BCE ile uyumlu; en iyi epoch sonrası 15 epoch erken durma.
- Kaydedilen AP, precision, recall, F1, accuracy ve Rank IC tahminlerden yeniden hesaplandı.
- Olasılıklardan hesaplanan BCE, kayıtlı logit BCE ile 1e-6 toleransında eşleşiyor.
- 76 benzersiz bölüm sınırı kontrolü geçti; seedler dahil kayıtlarda {boundary_count} kontrol var.
- Her validation bölümünde 1.017 dizi, her eğitim bölümünde 4.977 dizi var.
- Tahmin tarihleri bölümün 64. gözleminde başlıyor; saatlik grid ve farklı seed/foldlarda etiketler tutarlı.
- Büyük model/optimizer ağırlıkları ve ham veri ZIP'te bulunmadığı için bunların içeriği bağımsız doğrulanmadı.

## Ortak özet

Sayılar 114 fold/seed sonucunun eşit ağırlıklı ortalamasıdır; havuzlanmış tek tahmin kümesi değildir.

| Ölçüt | Sonuç |
|---|---:|
| Validation BCE | {avg.bce:.6f} |
| Average Precision (AP) | {avg.average_precision:.6f} |
| Olumlu sınıf oranı | %{avg.prevalence*100:.2f} |
| AP / sınıf oranı (fold/seed ortalaması) | {avg.ap_lift:.3f}× |
| Precision, eşik 0,5 | %{avg.precision*100:.2f} |
| Recall, eşik 0,5 | %{avg.recall*100:.2f} |
| F1, eşik 0,5 | {avg.f1:.6f} |
| Accuracy | %{avg.accuracy*100:.2f} |
| Her örneğe olumsuz diyen referansın accuracy'si | %{avg.always_negative_accuracy*100:.2f} |
| Olumlu tahmin oranı | %{avg.prediction_rate*100:.2f} |
| Rank IC | {avg.rank_ic:.6f} |

AP'nin sınıf oranının üzerinde olması olumlu örnekleri sıralamada bir beceri işaretidir.
Bu sonuç, epoch seçiminin yapıldığı doğrulama verisinde ölçülmüştür; bağımsız nihai başarı veya
istatistiksel anlamlılık kanıtı değildir. AP, trapez PR-AUC ile aynı hesaplama değildir.
0,5 eşikte model az olumlu karar veriyor; recall ve F1 düşük. Accuracy basit hep-olumsuz
referansını aşmıyor. Bu nedenle yaklaşık %69 accuracy tek başına iyi model demek değildir.
Rank IC ortalaması küçük; {record['positive_rank_ic_seed_averaged_folds']}/38 foldun seed-ortalama IC'si pozitiftir.
Eşik veya ölçütü bu sonuçlara bakıp değiştirmiyoruz; A–D karşılaştırmasının ortak kuralları korunuyor.

## Örnek sayıları ve eğitim davranışı

Toplam {record['validation_prediction_rows_with_repeats']:,} tahmin kaydı var; aynı tarih farklı seedler ve
örtüşen validation foldlarında tekrar ediyor. Benzersiz validation saati {len(unique_timestamps):,}.
Bu saatler de 64 gözlemli örtüşen girdiler ve 10 saatlik hedeflerden dolayı bağımsız gözlemler sayılmaz.
114 bağımsız deney veya 115.938 bağımsız örnek varmış gibi hata aralığı hesaplanmadı.
En iyi epoch ortalaması {avg.best_epoch:.2f}; aralık {int(table.best_epoch.min())}–{int(table.best_epoch.max())}.
{int((table.best_epoch==1).sum())} eğitimde ilk epoch seçildi. Bu, daha uzun eğitimin validation
BCE'yi düzenli iyileştirmediğini gösterir; tek başına bunun sebebinin hangi bileşen olduğunu kanıtlamaz.

## Ham veri hatasının gerçek nedeni

Paketin kalite kaydı, 1H verisinde 2024-10-28 20:00 UTC zamanında bir doğrulanmış
sıfır işlem/hacim barı olduğunu gösteriyor. Yeni politika bu kaynak barını korumuş;
silinen veya yapay doldurulan satır sayısı sıfır. 4H verisinde böyle bir bar yok.
Bu kayıt mevcut ham kaynağın doğrulama sonucudur; sağlayıcının gerçek işlem akışı ayrıca denetlenmedi.

## Sıradaki aşama

B: tüm mevcut wavelet dışı girdiler → aynı GRU → aynı BCE. Önce açık pozisyonun iki
eksik girdisi tamamlanmalı. A'nın Drive'da dondurulan ham kaynakları korunmalı;
yereldeki farklı ham paketle karşılaştırma yapılmamalı. B hazırlanırken tarih, etiket
ve A'nın 6 girdi kanalının aynılığı kontrol edilmeli. Train frekansından üretilen sabit
olasılık referansı da rapor altyapısına eklenmeli; BCE için referans olmadan 'beceri'
iddiası kurulamaz. Bunlar validation odaklı geliştirmedir, nihai test henüz ayrılmadı.

![Doğrulama görünümü](A_validation_dashboard.png)
"""
    if experiment != 'A':
        report = report.split('## Sıradaki aşama')[0] + f'''## Sıradaki aşama

{ {'B': 'C: aynı 34 girdi ve aynı BCE ile TCN.', 'C': 'D: aynı 34 girdi ve aynı BCE ile paralel TCN–GRU.', 'D': 'Önceden tanımlanan validation ölçütüyle seçim, ardından wavelet karşılaştırması.'}[experiment] } Model seçimi A–D tamamlanmadan yapılmaz.
OI kalite ve A eşleşme ayrıntıları `review_summary.json` içinde bulunur.

![Doğrulama görünümü]({experiment}_validation_dashboard.png)
'''
    (output / f"{experiment}_DENEYI_INCELEME.md").write_text(report, encoding="utf-8")
    print(json.dumps({key: record[key] for key in ['run', 'completed_fold_seed', 'epochs_trained', 'macro_metrics', 'test_evaluations']}, indent=2))
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", choices=['A', 'B', 'C', 'D'], default='A')
    args = parser.parse_args()
    review(args.bundle, args.output, args.experiment)
