"""Rebuild the clean-output Colab notebook from reviewable cell sources."""
from __future__ import annotations

import json
import copy
from pathlib import Path
from textwrap import dedent


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": dedent(source).strip().splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


cells = [
cell("markdown", """
# 06 — Danışman Deneyleri: Colab + Drive

Mevcut notebooklarla aynı düzen: GPU → Drive → Git → bağımlılıklar → veri → eğitim → sonuç ZIP'i.
Bu notebook **ayrı deney dalını** çeker; eski otomatik deney matrisini veya test değerlendirmesini çağırmaz.

İlk çalışma: **A = 6 temel fiyat/hacim özelliği → GRU → saf BCE**.
38 fold, 3 seed, validation BCE ile epoch seçimi, her iki sınırda 24 saat boşluk.
B/C/D için OI kaynağı ve A veri eşleşmesi kontrol edilir; B aşamasında ayrı 07 notebookunu kullanın.
Wavelet/loss aşamaları sonraki ayrı adımlardır. İlk çalışmada `EXPERIMENTS = ['A']` bırakın.

Colab menüsünden **Runtime → Change runtime type → GPU** seçip hücreleri sırayla çalıştırın.
Checkpointler Drive'a gider. Yeniden bağlandıktan sonra aynı notebooku çalıştırmak,
aynı kod/veri/ayar/ortam imzasında kaldığı yerden devam eder. GPU veya kütüphane sürümü değişirse
ayrı çalışma oluşur; eski epoch'larla yeni ortamın sonuçları karıştırılmaz.
Yerel Windows deneyi korunmuştur; sürüm ve kaynak imzası farklı olduğundan Colab'da yeni çalışma başlar.
"""),
cell("code", """
import os, sys, subprocess, json, shutil, hashlib
from pathlib import Path
from datetime import datetime, timezone
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['ADVISOR_LOCK_ROOT'] = '/content/advisor_runtime/locks'
import torch
assert torch.cuda.is_available(), 'Runtime → Change runtime type → GPU seçin.'
print('GPU:', torch.cuda.get_device_name(0), '| Torch:', torch.__version__)

# İlk aşamada yalnızca A. Sonuçlara bakarak seed/fold veya ölçüt değiştirmeyin.
EXPERIMENTS = ['A']
REPO_URL = 'https://github.com/umutergul74/yeniBot.git'
REPO_BRANCH = 'codex/advisor-ablation-protocol'
REPO_COMMIT = ''  # Boş: dalın güncel commit'i. Devam için eski commit'i buraya sabitleyebilirsiniz.
REPO_DIR = Path('/content/yenibot_advisor_repo')
DRIVE_BASE = Path('/content/drive/MyDrive/yeniBot')
LOCAL_BASE = Path('/content/advisor_runtime')
AUTO_UNASSIGN = False  # True: başarıyla eğitim + ZIP tamamlandıktan sonra GPU oturumunu bırakır.
DOWNLOAD_MISSING_RAW = True  # Yalnızca gerekli 1H/4H ham kaynak Drive'da yoksa indir.
"""),
cell("code", """
from google.colab import drive
drive.mount('/content/drive')
PERSIST = DRIVE_BASE / 'advisor_experiments' / 'colab_v1'
for folder in [PERSIST / 'inputs', PERSIST / 'data', PERSIST / 'runs', PERSIST / 'sessions', PERSIST / 'reports', LOCAL_BASE]:
    folder.mkdir(parents=True, exist_ok=True)
print('Kalıcı çıktı alanı:', PERSIST)
"""),
cell("markdown", """
## Git'ten güncel deney kodunu çek

Doğru dal açıkça seçilir; commit ekrana ve sonuç paketine kaydedilir. Özel depo için Git kimlik
doğrulaması gerekirse mevcut hesabınızla erişim sağlayın; notebooka token yazmayın.
Çekilen kodu eğitim ayrı Python işleminde yükler. Önceki hücrelerden `yenibot` modülleri
yüklenmişse kodu güncellemeden önce oturumu yeniden başlatmanız istenir.
"""),
cell("code", """
def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO_DIR), *args], text=True).strip()

if any(name == 'yenibot' or name.startswith('yenibot.') for name in sys.modules):
    raise RuntimeError('Kod hücresini tekrar çalıştırmadan önce Runtime → Restart session; sonra baştan çalıştırın.')
if not (REPO_DIR / '.git').exists():
    subprocess.run(['git', 'clone', '--branch', REPO_BRANCH, '--single-branch', REPO_URL, str(REPO_DIR)], check=True)
else:
    assert not git('status', '--porcelain', '--untracked-files=no'), 'Colab çalışma kopyasında değişiklik var; otomatik silinmez.'
    subprocess.run(['git', '-C', str(REPO_DIR), 'fetch', 'origin', REPO_BRANCH], check=True)
    if REPO_COMMIT:
        subprocess.run(['git', '-C', str(REPO_DIR), 'checkout', '--detach', REPO_COMMIT], check=True)
    else:
        subprocess.run(['git', '-C', str(REPO_DIR), 'checkout', REPO_BRANCH], check=True)
        subprocess.run(['git', '-C', str(REPO_DIR), 'merge', '--ff-only', 'origin/' + REPO_BRANCH], check=True)
if REPO_COMMIT:
    subprocess.run(['git', '-C', str(REPO_DIR), 'checkout', '--detach', REPO_COMMIT], check=True)
repo_commit = git('rev-parse', 'HEAD')
assert (REPO_DIR / 'yenibot/training/advisor.py').exists(), 'Yanlış dal veya eksik deney kodu.'
print('Dal:', REPO_BRANCH, '| Commit:', repo_commit)
sys.path.insert(0, str(REPO_DIR))
os.chdir(REPO_DIR)
"""),
cell("code", """
# Colab'ın mevcut GPU Torch paketini zorla değiştirmeyin; upgrade kullanılmıyor.
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(REPO_DIR / 'requirements.txt')], check=True)
import yaml
from yenibot.training.advisor import sha256, atomic_json
from yenibot.training.advisor_colab import (freeze_market_inputs, cache_frozen_inputs, create_review_bundle,
                                          freeze_oi_inputs, pin_completed_a_reference)
from yenibot.data import download_full_klines
from yenibot.data.binance import download_futures_metrics_from_vision
session_stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
SESSION = PERSIST / 'sessions' / session_stamp
SESSION.mkdir(parents=True, exist_ok=True)
packages = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
(SESSION / 'packages.txt').write_text(packages)
session_meta = {'session': session_stamp, 'branch': REPO_BRANCH, 'commit': repo_commit,
                'gpu': torch.cuda.get_device_name(0), 'python': sys.version, 'torch': torch.__version__,
                'experiments': EXPERIMENTS, 'status': 'setup', 'test_evaluations': 0}
atomic_json(SESSION / 'session.json', session_meta)
print('Ortam ve kaynak kaydı:', SESSION)
"""),
cell("markdown", """
## Ham veriyi bir kez sabitle; Colab diskine kopyala

Drive'daki mevcut `data/raw/snapshots/20260830_integrity_v2` kaynakları önceliklidir;
yoksa mevcut `data/raw` dosyaları kullanılır. Eksik dosya varsa mevcut indiriciyle
sabit tarih aralığı için indirilir; eski ham dosyaların üzerine yazılmaz.
Girdiler bu araştırmanın ayrı alanında bir kez dondurulur, hashleri sonraki oturumlarda kontrol edilir.
Eksik saatler veya yetersiz tarih kapsamı eğitimi durdurur; tarih/örnek sayısı sessizce küçültülmez.
İşlem olmayan bir kaynak barı yalnızca işlem/hacim alanlarının tamamı sıfır,
OHLC fiyatları eşit ve önceki kapanışla aynıysa korunur; tarihleri kalite raporuna yazılır.
Sıfır fiyat, negatif değer veya tutarsız sıfır işlem kaydı kabul edilmez; satır uydurulmaz/silinmez.
Bu ilk A çalışması için açık pozisyon indirilmez. B–D için kaynak ayrı ve sabit bir girdiye kaydedilir.
"""),
cell("code", """
import pandas as pd
base_cfg = yaml.safe_load((REPO_DIR / 'configs/advisor_ablation.yaml').read_text())
RAW_START = '2022-01-01T00:00:00Z'
RAW_END = (pd.Timestamp(base_cfg['development']['end']) + pd.Timedelta(hours=10)).isoformat()
frozen_snapshot = PERSIST / 'inputs' / 'market_snapshot_v1'
if not (frozen_snapshot / 'snapshot_manifest.json').exists():
    candidates = [DRIVE_BASE / 'data/raw/snapshots/20260830_integrity_v2', DRIVE_BASE / 'data/raw']
    source = next((p for p in candidates if all((p / f'btc_{i}.parquet').exists() for i in ['1h','4h'])), None)
    if source is None:
        if not DOWNLOAD_MISSING_RAW:
            raise FileNotFoundError('Drive ham veri yok; 01_data_preparation veya DOWNLOAD_MISSING_RAW kullanın.')
        source = LOCAL_BASE / 'downloads'
        source.mkdir(parents=True, exist_ok=True)
        for interval in ['1h','4h']:
            frame = download_full_klines('BTCUSDT', interval, RAW_START,
                        pd.Timestamp(RAW_END) + pd.Timedelta(hours=4), data_source='auto')
            frame.to_parquet(source / f'btc_{interval}.parquet', index=False)
            print('İndirildi:', interval, len(frame))
    freeze_market_inputs(source, frozen_snapshot, start=RAW_START, end=RAW_END)
else:
    freeze_market_inputs(frozen_snapshot, frozen_snapshot, start=RAW_START, end=RAW_END)
cache_frozen_inputs(frozen_snapshot, LOCAL_BASE / 'raw')
shutil.copyfile(frozen_snapshot / 'snapshot_manifest.json', PERSIST / 'data/input_snapshot.manifest.json')
print('Sabit ham girdiler:', frozen_snapshot)
if any(name != 'A' for name in EXPERIMENTS):
    reference_a = pin_completed_a_reference(PERSIST)
    oi_candidates = [DRIVE_BASE / 'data/raw/btc_futures_metrics.parquet',
                     DRIVE_BASE / 'data/raw/snapshots/20260830_integrity_v2/btc_futures_metrics.parquet']
    oi_source = None
    for candidate in oi_candidates:
        if candidate.exists():
            timestamps = pd.read_parquet(candidate, columns=['timestamp']).timestamp
            timestamps = pd.to_datetime(timestamps, utc=True)
            if timestamps.min() <= pd.Timestamp(RAW_START) and timestamps.max() >= pd.Timestamp(RAW_END):
                oi_source = candidate
                break
    oi_file = freeze_oi_inputs(oi_source, PERSIST / 'inputs/oi_snapshot_v1',
        start=RAW_START, end=RAW_END,
        downloader=lambda start, end: download_futures_metrics_from_vision('BTCUSDT', start, end))
    for source_file in [oi_file, oi_file.with_suffix('.manifest.json')]:
        shutil.copyfile(source_file, LOCAL_BASE / 'raw' / source_file.name)
    shutil.copyfile(oi_file.with_suffix('.manifest.json'), PERSIST / 'data/open_interest_input.manifest.json')
    print('Sabit açık pozisyon girdisi:', oi_file)
"""),
cell("code", """
# Sabit yollar: sonraki Colab oturumunda da aynı imza kullanılabilsin.
cfg = base_cfg
cfg['data']['snapshot'] = str(LOCAL_BASE / 'raw')
cfg['data']['frame'] = str(LOCAL_BASE / 'development_wavelet_off.parquet')
cfg['data']['feature_config'] = str(REPO_DIR / 'config.yaml')
cfg['data']['feature_columns_file'] = str(REPO_DIR / 'configs/advisor_full_features.txt')
if any(name != 'A' for name in EXPERIMENTS):
    cfg['data']['frame'] = str(LOCAL_BASE / 'development_full_wavelet_off.parquet')
    cfg['data']['futures_metrics'] = str(LOCAL_BASE / 'raw/btc_futures_metrics.parquet')
    cfg['data']['reference_a_manifest'] = str(reference_a)
    cfg['data']['oi_policy'] = {'tolerance_minutes': 90, 'max_diff_gap_minutes': 15, 'min_coverage': .99}
cfg['output'] = str(PERSIST / 'runs')  # Checkpointler her epoch sonunda doğrudan Drive'a yazılır.
runtime_config = LOCAL_BASE / 'advisor_colab.yaml'
runtime_config.write_text(yaml.safe_dump(cfg, sort_keys=False))
shutil.copyfile(runtime_config, SESSION / 'advisor_colab.yaml')
shutil.copyfile(REPO_DIR / 'configs/advisor_full_features.txt', SESSION / 'full_features.txt')
print('Purge/boşluk:', cfg['walk_forward']['purge_bars'], cfg['walk_forward']['embargo_bars'])
print('Ufuk:', cfg['labeling']['max_holding_bars'], '| Epoch seçimi:', cfg['training']['selection_metric'])
print('Seedler:', cfg['training']['seeds'], '| Deneyler:', EXPERIMENTS)
assert cfg['walk_forward']['purge_bars'] >= 10 and cfg['walk_forward']['embargo_bars'] >= 10
assert EXPERIMENTS and all(name in cfg['experiments'] for name in EXPERIMENTS)

def execute_logged(command, log_name):
    # Ayrı süreç güncel kodu yükler; hata kodu başarısızlığı gizlemez.
    with (SESSION / log_name).open('a', buffering=1) as log:
        with subprocess.Popen(command, cwd=REPO_DIR, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                return_code = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
"""),
cell("code", """
execute_logged([sys.executable, '-u', '-m', 'yenibot.training.advisor', 'prepare',
                '--config', str(runtime_config)], 'prepare.log')
frame_path = Path(cfg['data']['frame'])
manifest_path = frame_path.with_suffix('.manifest.json')
shutil.copyfile(manifest_path, PERSIST / 'data' / manifest_path.name)
prepared = json.loads(manifest_path.read_text())
print('Geliştirme satırı:', prepared['rows'], '| Tarihler:', prepared['start'], prepared['end'])
print('Eksik tam özellikler:', prepared['full_features_missing'])
if prepared.get('open_interest_audit'):
    print('A ile temel veri eşleşmesi:', prepared['reference_a_audit'])
    print('Açık pozisyon kalite denetimi:', prepared['open_interest_audit'])
if any(name != 'A' for name in EXPERIMENTS) and prepared['full_features_missing']:
    raise RuntimeError('B–D için eksik özellik kaynağı var. Eksik sütunlarla tam özellik deneyi yapılmaz.')

# Fold tarihleri ve hedef taşması: eğitimden önce bütün sınırları kontrol et.
from yenibot.training.walk_forward import PurgedWalkForwardCV
from yenibot.training.advisor import audit_boundary
data = pd.read_parquet(frame_path)
folds = list(PurgedWalkForwardCV(**cfg['walk_forward'], label_horizon_bars=10).split(len(data)))
assert len(folds) == 38, f'Beklenen 38 fold; bulunan {len(folds)}. Kaynağı inceleyin.'
calendar = []
for fold in folds:
    for before, after in [(fold.train, fold.val), (fold.val, fold.test)]:
        audit_boundary(data, before, after, 10)
    row = {'fold': fold.fold + 1}
    for section, indices in [('train',fold.train),('validation',fold.val),('reserved_section',fold.test)]:
        row[section + '_start'] = data.timestamp.iloc[indices[0]].isoformat()
        row[section + '_end'] = data.timestamp.iloc[indices[-1]].isoformat()
    calendar.append(row)
pd.DataFrame(calendar).to_csv(PERSIST / 'data/fold_calendar.csv', index=False)
del data
print('38 fold / 76 sınır kontrolü geçti. Test tahmini: 0.')
"""),
cell("markdown", """
## Eğitim ve kalıcı devam kaydı

Bu hücre eski `run_experiment_matrix` akışını çağırmaz. Yalnızca validation ölçümleri üretilir.
Drive'daki `runs/<deney_imzası>/STATUS.md` mevcut fold/seed'i, fold klasöründeki
`STATUS.md` son kaydedilen epoch'u gösterir. Ortasında kesilen epoch yeniden çalışır;
tamamlanmış foldlar atlanır. Aynı runtime içinde ikinci eğitim kilitle engellenir.
Aynı Drive çıktısına yazan iki ayrı Colab oturumunu eşzamanlı çalıştırmayın.

Colab oturumu kapanırsa yeniden bağlanıp hücreleri aynı ayarlarla çalıştırın.
Checkpointler kalıcıdır; çalışma imzası değişirse yeni çalışma açılır.
Drive yazma hatası varsa başarısızlık gizlenmez; eğitim durur.
"""),
cell("code", """
bundle_path = None
session_meta['status'] = 'training'
atomic_json(SESSION / 'session.json', session_meta)
try:
    for experiment in EXPERIMENTS:
        session_meta['current_experiment'] = experiment
        atomic_json(SESSION / 'session.json', session_meta)
        execute_logged([sys.executable, '-u', '-m', 'yenibot.training.advisor', 'run',
                        '--config', str(runtime_config), '--experiment', experiment], f'train_{experiment}.log')
    session_meta['status'] = 'complete'
    session_meta.pop('current_experiment', None)
except BaseException as exc:
    session_meta['status'] = 'interrupted_or_failed'
    session_meta['error'] = repr(exc)
    raise
finally:
    session_meta['updated_at'] = datetime.now(timezone.utc).isoformat()
    atomic_json(SESSION / 'session.json', session_meta)
    bundle_path = create_review_bundle(PERSIST, session_meta)
    print('Drive inceleme ZIP:', bundle_path)
    print('İşlem durumu:', session_meta['status'])
"""),
cell("code", """
# Her klasörün kapsamını ayrı göster: tamamlanmamış sonuçlardan kazanan seçme.
for directory in sorted((PERSIST / 'runs').glob('*')):
    summary = directory / 'validation_summary.csv'
    status_path = directory / 'status.json'
    if not status_path.exists():
        continue
    status = json.loads(status_path.read_text())
    print(directory.name, '| durum:', status['status'], '| tamamlanan:', len(status['completed']),
          '| test değerlendirmesi:', status['test_evaluations'])
    if summary.exists():
        display(pd.read_csv(summary))
print('İnceleme için paylaşılacak küçük ZIP:', PERSIST / 'reports/advisor_latest_review_bundle.zip')
if AUTO_UNASSIGN and session_meta['status'] == 'complete' and bundle_path and bundle_path.exists():
    from google.colab import runtime
    runtime.unassign()
"""),
cell("markdown", """
## Sonuçları burada incelememiz için

Drive: `MyDrive/yeniBot/advisor_experiments/colab_v1/reports/advisor_latest_review_bundle.zip`.
ZIP'i buraya ekleyebilirsiniz veya bilgisayara senkronize edilen dosyanın yolunu paylaşabilirsiniz.
Bu sohbetin Colab'a bağlanan Drive'ı kendiliğinden görmesi beklenmemelidir.
ZIP; commit, ayarlar, ortam sürümleri, sınır takvimi, epoch geçmişi, validation
tahminleri ve ilerleme durumunu içerir. Büyük model ağırlıkları Drive'da kalır.
Yerel deney ile Colab deneyi farklı imzalarla korunur; sonuçlar tek deneymiş gibi birleştirilmez.
"""),
]

notebook = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"name": "06_advisor_ablation_colab.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
if __name__ == "__main__":
    for index, entry in enumerate(cells):
        entry["id"] = f"advisor-{index:02d}"
        if entry["cell_type"] == "code":
            compile("".join(entry["source"]), f"cell_{index}", "exec")
    target = Path(__file__).resolve().parents[1] / "notebooks/06_advisor_ablation_colab.ipynb"
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Generated {target.name}: {len(cells)} cells; code syntax validated")
    full = copy.deepcopy(notebook)
    full['metadata']['colab']['name'] = '07_advisor_full_features_colab.ipynb'
    full['cells'][0] = cell('markdown', '''
    # 07 — B Deneyi: Tüm Özellikler → GRU → BCE

    Bu çalışma tamamlanmış A deneyindeki aynı saatlik satırları, etiketleri,
    foldları ve eğitim ayarlarını kullanır. Wavelet kapalıdır; 34 özellik vardır.
    Açık pozisyonun iki log-değişim özelliği gerçek arşiv verisinden eklenir.
    **EXPERIMENTS = ['B'] bırakın; A'yı yeniden eğitmeyin.**

    GPU seçip hücreleri baştan sırayla çalıştırın. A'nın tamamlandığı Drive alanını
    kullanın. Kaynak yoksa Binance Vision arşivi indirilir; tamamlanan aylar
    Drive'a kaydedilir ve kesinti sonrası yeniden indirilmez. İlk kurulum uzun sürebilir.
    Kaynak hashleri ve yeniden üretilen temel veri A ile eşleşmezse eğitim durur.

    OI geçmişe doğru eşleştirilir (en fazla 90 dakika). 15 dakikayı aşan kaynak
    boşluğunda fark hesaplanmaz. En az %99 geçerli özellik kapsamı gerekir;
    kalan sınırlı eksikler nötr sıfırla doldurulur ve tarihleri raporlanır.
    Eksik kaynak veya düşük kapsamla eğitim başlatılmaz.

    38 fold × 3 seed; eğitim/validation arası 24 saat, validation/ayrılmış bölüm
    arası 24 saat. Epoch seçimi validation BCE ile yapılır. Test değerlendirilmez.
    Checkpointler ve ilerleme kayıtları Drive'da; oturum kesilirse aynı kod/veri/
    ortamla baştan çalıştırınca kayıttan devam edilir. ZIP A ve B kayıtlarını içerir.
    ''')
    full['cells'][0]['id'] = 'advisor-00'
    full['cells'][1]['source'] = [line.replace("EXPERIMENTS = ['A']", "EXPERIMENTS = ['B']")
                                .replace('# İlk aşamada yalnızca A.', '# Bu aşamada yalnızca B.')
                                for line in full['cells'][1]['source']]
    full['cells'][6]['source'] = [line for line in full['cells'][6]['source']
                                if 'Bu ilk A çalışması' not in line]
    full_target = target.with_name('07_advisor_full_features_colab.ipynb')
    for index, entry in enumerate(full['cells']):
        if entry['cell_type'] == 'code':
            compile(''.join(entry['source']), f'full_cell_{index}', 'exec')
    full_target.write_text(json.dumps(full, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(f'Generated {full_target.name}: {len(full["cells"])} cells; code syntax validated')
