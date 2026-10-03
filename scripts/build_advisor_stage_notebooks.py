"""08/09 Colab notebooks reuse the reviewed Git/Drive/session conventions."""
import copy
import json
from pathlib import Path

from build_advisor_notebook import cell, cells, notebook


def build(phase):
    is_wavelet = phase=='wavelet'
    name = '08_advisor_wavelet_colab.ipynb' if is_wavelet else '09_advisor_losses_colab.ipynb'
    result = copy.deepcopy(notebook)
    result['metadata']['colab']['name'] = name
    content = copy.deepcopy(cells[:6])
    content[0] = cell('markdown', f'''
    # {'08 — Wavelet karşılaştırması' if is_wavelet else '09 — Loss karşılaştırmaları'}

    Aynı Git → GPU → Drive → hazırlık → eğitim → ZIP düzeni.
    A–D'nin 38 fold × 3 seed sonuçları eksiksiz olmalı. Seçim ölçütü validation BCE;
    kazanan A (6 özellik / GRU). Aynı 6 boyut, ham fiyat/ATR etiketleri ve splitler korunur.
    Eylül 2026 test dönemi okunmaz; bütün ölçümler validation.

    {'W_OFF tamamlanmış A kontrolüdür ve yeniden eğitilmez. W_ON yalnızca getiri ve hacim kanallarını geçmişe dayalı wavelet karşılıklarıyla değiştirir. En düşük ortalama validation BCE seçilir; eşitlikte W_OFF.' if is_wavelet else 'Önce 08 tamamlanmış olmalı. Wavelet kazananı otomatik sabitlenir. Saf BCE kontrolü seçilen A veya W_ON sonucundan alınır; yeniden eğitilmez. Ağırlıklı BCE, Focal, BCE + Pearson ve Focal + Pearson denenir.'}

    GPU ve ortam cached kontrolle eşleşmelidir; uyumsuzlukta durur.
    Tüm losslar için epoch ve yöntem seçimi ortak **validation BCE**, eşik **0,5**.
    Parametreler önceden `configs/advisor_stages.yaml` içinde sabittir.
    Checkpointler Drive'da; kesinti sonrası aynı commit/ayar/ortamla baştan çalıştırın.
    Başka bir runtime'ı aynı çıktı alanına eşzamanlı yazdırmayın.
    ''')
    content[1] = cell('code', ''.join(content[1]['source']).replace("EXPERIMENTS = ['A']",
        "EXPERIMENTS = ['W_ON']" if is_wavelet else "EXPERIMENTS = ['L_WEIGHTED_BCE', 'L_FOCAL', 'L_BCE_PEARSON', 'L_FOCAL_PEARSON']")
        + f"\nPHASE = '{phase}'\n")
    content[1]['source'] = [line.replace('# İlk aşamada yalnızca A.', '# Bu notebookun varsayılan deneyleri.') for line in content[1]['source']]
    content[1]['source'] = [line for line in content[1]['source'] if not line.startswith('DOWNLOAD_MISSING_RAW =')]
    content[5] = cell('code',''.join(content[5]['source'])+'''
from yenibot.training.advisor_stages import select_architecture, select_wavelet
session_meta['phase'] = PHASE
atomic_json(SESSION / 'session.json', session_meta)
''')
    content += [cell('markdown','''
    ## Tamamlanmış kontrollere bağlan; sadece sabit geliştirme verisini hazırla

    İndirme yapılmaz. Aynı Drive `colab_v1` alanındaki A–D ve market snapshot kullanılır.
    Ham girdiler ile yeniden üretilen temel veri A ile hash düzeyinde eşleşmelidir.
    Wavelet 256 geçmiş gözlem, db4, seviye 2, eşik ölçeği 0,5. Diğer dört girdi
    ve etiketler değişmez. Oluşan iki kanallı ek veri bir kez Drive'a kaydedilir.
    Açık pozisyon kaynağı bu altı özellikli modelde kullanılmadığından indirilmez.
    '''),cell('code','''
from yenibot.training.advisor_colab import cache_frozen_inputs, pin_completed_a_reference
stage_cfg = yaml.safe_load((REPO_DIR / 'configs/advisor_stages.yaml').read_text())
decision, control = select_architecture(PERSIST, stage_cfg)
print('Mimari seçimi:', decision['winner'], decision['architecture'], decision['features'])
print('A–D validation BCE:', {key: value['mean_bce'] for key,value in decision['references'].items()})
reference_a = pin_completed_a_reference(PERSIST)
snapshot = PERSIST / 'inputs/market_snapshot_v1'
assert (snapshot / 'snapshot_manifest.json').exists(), 'A deneyinin sabit snapshot alanı bulunamadı.'
cache_frozen_inputs(snapshot, LOCAL_BASE / 'raw')
cfg = copy.deepcopy(control['config'])
cfg['data']['snapshot'] = str(LOCAL_BASE / 'raw')
cfg['data']['feature_config'] = str(REPO_DIR / 'config.yaml')
cfg['data']['feature_columns_file'] = str(REPO_DIR / 'configs/advisor_full_features.txt')
cfg['output'] = str(PERSIST / 'runs')
frame_key = hashlib.sha256(json.dumps(stage_cfg['wavelet'],sort_keys=True).encode()).hexdigest()[:12]
source_key = sha256(REPO_DIR / 'yenibot/training/advisor_stages.py')[:12]
# Yeni kaynak imzası eski hazırlanmış frame'in üzerine yazmaz.
cfg['data']['frame'] = str(PERSIST / 'data' / f'stages_{frame_key}_{source_key}.parquet')
cfg['stage'] = {'phase': PHASE, 'specification': stage_cfg,
                'reference_manifest': str(reference_a), 'architecture_selection': decision,
                'holdout_config': str(REPO_DIR / 'configs/advisor_final_test.yaml')}
if PHASE == 'loss':
    choice = select_wavelet(PERSIST, stage_cfg, decision)
    cfg['stage']['wavelet_selection'] = choice
    print('Seçilmiş wavelet:', choice['winner'])
assert EXPERIMENTS and all(name == 'W_ON' if PHASE == 'wavelet' else name in stage_cfg['losses'] for name in EXPERIMENTS)
runtime_config = LOCAL_BASE / f'advisor_{PHASE}.yaml'
runtime_config.write_text(yaml.safe_dump(cfg,sort_keys=False))
shutil.copyfile(runtime_config, SESSION / runtime_config.name)
shutil.copyfile(REPO_DIR / 'configs/advisor_stages.yaml', SESSION / 'advisor_stages.yaml')
shutil.copyfile(REPO_DIR / 'configs/advisor_final_test.yaml', PERSIST / 'data/reserved_test_protocol.yaml')
print('Purge/embargo:', cfg['walk_forward']['purge_bars'],cfg['walk_forward']['embargo_bars'])
print('Checkpoint kapsamı:', EXPERIMENTS)
''')]
    content[1]['source'].insert(0,'import copy\n')
    # The logging helper is the same reviewed subprocess/interrupt implementation.
    helper = ''.join(cells[8]['source']).split('def execute_logged(')[1]
    content.append(cell('code','def execute_logged('+helper))
    content.append(cell('code','''
execute_logged([sys.executable, '-u', '-m', 'yenibot.training.advisor_stages', 'prepare',
                '--config', str(runtime_config)], 'prepare_stages.log')
manifest_path = Path(cfg['data']['frame']).with_suffix('.manifest.json')
prepared = json.loads(manifest_path.read_text())
print('Satır:', prepared['rows'], '| Tarih:', prepared['start'],prepared['end'])
print('Wavelet denetimi:', prepared['wavelet_audit'])
assert prepared['rows'] == 33551
'''))
    train = ''.join(cells[11]['source']).replace('yenibot.training.advisor', 'yenibot.training.advisor_stages')
    train = train.replace("    session_meta['status'] = 'complete'", '''    if PHASE == 'loss' and set(EXPERIMENTS) == set(stage_cfg['losses']):
        execute_logged([sys.executable, '-u', '-m', 'yenibot.training.advisor_stages', 'finalize',
                        '--config', str(runtime_config)], 'loss_selection.log')
    elif PHASE == 'loss':
        print('Seçili alt kapsam bitti; tüm losslar tamamlanmadan kazanan seçilmez.')
    session_meta['status'] = 'complete' ''')
    content += [cell('markdown','''
    ## Eğitim ve kaldığı yerden devam

    Her epoch'ta Drive checkpoint'i ve ilerleme kaydı yazılır. Bitmiş foldlar atlanır.
    `runs/<deney>/STATUS.md` ve foldun `progress.json` dosyası kalan aşamayı gösterir.
    09'da tek loss çalıştırmak için EXPERIMENTS listesini daraltabilirsiniz; sonrasında
    varsayılan tüm listeyi çalıştırmak bitmişleri atlar ve eksikleri tamamlar.
    Tüm losslar tamamlanmadan loss seçimi kaydedilmez. ZIP başarısızlıkta da hazırlanır.
    '''),cell('code',train),copy.deepcopy(cells[12]),copy.deepcopy(cells[13])]
    for i,entry in enumerate(content):
        entry['id']=f'{phase}-{i:02d}'
        if entry['cell_type']=='code':
            compile(''.join(entry['source']),f'{name}_cell_{i}','exec')
    result['cells']=content
    target=Path(__file__).resolve().parents[1]/'notebooks'/name
    target.write_text(json.dumps(result,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(f'Generated {name}: {len(content)} cells; syntax OK')


if __name__=='__main__':
    build('wavelet')
    build('loss')
