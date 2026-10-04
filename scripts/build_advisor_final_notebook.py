"""Generate the separate Colab final fit/freeze/holdout notebook."""
import copy
import json
from pathlib import Path

from build_advisor_notebook import cell, cells, notebook


def build():
    name='10_advisor_final_test_colab.ipynb'
    result=copy.deepcopy(notebook);result['metadata']['colab']['name']=name
    content=copy.deepcopy(cells[:6])
    content[0]=cell('markdown','''
    # 10 — Final eğitim → model/scaler kilitleme → Eylül 2026 testi

    A–D, 08 ve 09 tamamlanmış olmalı. Seçilmiş yöntem: 6 özellik / GRU /
    wavelet kapalı / saf BCE. T4 kullanın; eski geliştirme ortamı korunmalı.

    **Eğitim:** 18 Aralık 2025–15 Temmuz 2026 (210 gün).
    **Purge:** 16 Temmuz (24 saat).
    **Validation:** 17 Temmuz–30 Ağustos (45 gün).
    **Test öncesi boşluk:** 31 Ağustos (24 saat).
    **Test:** 1–30 Eylül 2026, UTC; 720 tahmin, seed 42/43/44 ayrı.

    Üç final model yalnızca train'de öğrenir, epoch'u validation BCE seçer.
    Train+validation ile yeniden fit yapılmaz. Sonraki hücre bütün model/scaler
    hashlerini kilitler. **Test kaynağı yalnızca bu kilit oluşturulduktan sonra
    indirilir.** Testte eğitim, model/seed/eşik seçimi veya ağırlık güncelleme yok.

    Her adımın checkpoint/durum/ZIP'i Drive'da. Kesinti sonrası aynı commit,
    ayar ve ortamla baştan çalıştırın. Bitmiş adımlar/seeds tekrar eğitilmez.
    İki runtime aynı çıktı alanına yazmasın. Bu tamamlanmış dönem üzerinde
    geriye dönük dokunulmamış holdout'tur; ileriye dönük deney iddiası değildir.
    ''')
    content[1]['source']=[line.replace("EXPERIMENTS = ['A']","EXPERIMENTS = ['FINAL']")
                         for line in content[1]['source'] if not line.startswith('DOWNLOAD_MISSING_RAW =')]
    content += [cell('markdown','''
    ## Yöntem kararını ve final takvimini sabitle

    Aynı `colab_v1` Drive alanından A–D, wavelet ve loss kararları denetlenir.
    Final çıktılar ayrı `final_test_v1/` alanına kaydedilir; geliştirme kayıtları
    değişmez. Saatlik temel özellikler için 4H/OI kaynağı indirilmez.
    '''),cell('code','''
    import pandas as pd
    from yenibot.training.advisor_final import selected_contract
    cfg=yaml.safe_load((REPO_DIR/'configs/advisor_final_execution.yaml').read_text())
    cfg['reservation_config']=str(REPO_DIR/'configs/advisor_final_test.yaml')
    cfg['stage_config']=str(REPO_DIR/'configs/advisor_stages.yaml')
    cfg['development_root']=str(PERSIST)
    cfg['raw']['base_snapshot']=str(PERSIST/'inputs/market_snapshot_v1')
    FINAL_ROOT=PERSIST/'final_test_v1'
    cfg['output']=str(FINAL_ROOT)
    runtime_config=LOCAL_BASE/'advisor_final.yaml'
    runtime_config.write_text(yaml.safe_dump(cfg,sort_keys=False))
    SESSION=FINAL_ROOT/'sessions'/session_stamp
    SESSION.mkdir(parents=True,exist_ok=True)
    (SESSION/'packages.txt').write_text(packages)
    shutil.copyfile(runtime_config,SESSION/runtime_config.name)
    session_meta.update(phase='final',status='plan_frozen')
    atomic_json(SESSION/'session.json',session_meta)
    contract=selected_contract(cfg)
    print('Yöntem:',cfg['selected_method'])
    print('Eğitim/doğrulama:',cfg['fit'])
    print('Test rezervasyonu:',contract['reservation']['test'])
    print('Final çıktı:',FINAL_ROOT)
    ''')]
    helper=''.join(cells[8]['source']).split('def execute_logged(')[1]
    content.append(cell('code','def execute_logged('+helper+'''

def execute_final(action):
    session_meta.update(status='running',current_action=action)
    atomic_json(SESSION/'session.json',session_meta)
    try:
        execute_logged([sys.executable,'-u','-m','yenibot.training.advisor_final',action,
                        '--config',str(runtime_config)], action+'.log')
        session_meta.update(status='step_complete',last_completed_action=action)
        session_meta.pop('error',None)
    except BaseException as exc:
        session_meta.update(status='interrupted_or_failed',error=repr(exc))
        raise
    finally:
        session_meta['updated_at']=datetime.now(timezone.utc).isoformat()
        atomic_json(SESSION/'session.json',session_meta)
        subprocess.run([sys.executable,'-m','yenibot.training.advisor_final','bundle',
                        '--config',str(runtime_config)],check=True)
'''))
    for action,title,description in [
        ('prepare-fit','1 — Test öncesi saatlik kaynağı ve girdileri hazırla',
         '2022 başlangıcındaki sabit ham geçmiş korunur, eksik devam aylık cache ile yalnızca 31 Ağustos 09:00 UTC barına kadar indirilir. Eylül kaynağı okunmaz. Altı özellik eski saatlik hesaplayıcıyla üretilir.'),
        ('train','2 — Üç final modeli eğit',
         '5.040 train satırı → 4.977 dizi; 1.080 validation satırı → 1.017 dizi. Epoch seçimi normal validation BCE. Her epoch Drive checkpoint; seedler 42/43/44. Test sonucu yok.'),
        ('freeze','3 — Model, scaler ve ayar hashlerini kilitle',
         'Üç seed tamamlanmadan kilit oluşturulmaz. Her modelin en iyi validation epoch ağırlığı ve sadece train ile fit edilen scaler saklanır. Test verisini indirmek için bu manifest zorunludur.'),
        ('prepare-test','4 — Kilitten sonra Eylül test kaynağını hazırla',
         'İleri getiriler için son gereken saat 1 Ekim 09:00 UTC. İlk tahmin için 63 geçmiş gözlem eklenir: tablo 783 satır, tahmin sayısı 720. Fit geçmişi yeniden hesaplandığında birebir aynı olmalı.'),
        ('evaluate-test','5 — Donmuş modellerle final testi değerlendir',
         'Testte fit yok. Her seed ayrı 720 tahmin; BCE/AP/F1/precision/recall/accuracy/Rank IC ve mean/std. Mean/std seed kararlılığıdır, bağımsız örnek güven aralığı değildir. Sabit train-sınıf-frekansı ve hep-negatif referansları da raporlanır.')]:
        content += [cell('markdown',f'## {title}\n\n{description}'),cell('code',f"execute_final('{action}')")]
    content.append(cell('code','''
    test_results=json.loads((FINAL_ROOT/'runs/test/test_results.json').read_text())
    session_meta.update(status='complete',test_evaluations=1)
    atomic_json(SESSION/'session.json',session_meta)
    subprocess.run([sys.executable,'-m','yenibot.training.advisor_final','bundle',
                    '--config',str(runtime_config)],check=True)
    display(pd.DataFrame(test_results['per_seed']))
    display(pd.DataFrame(test_results['aggregate_mean_std']).T)
    print('Basit referanslar:',test_results['baselines'])
    print('İnceleme ZIP:',FINAL_ROOT/'reports/advisor_final_latest_review_bundle.zip')
    if AUTO_UNASSIGN:
        from google.colab import runtime
        runtime.unassign()
    '''))
    content.append(cell('markdown','''
    ## İnceleme için getirilecek ZIP

    `MyDrive/yeniBot/advisor_experiments/colab_v1/final_test_v1/reports/advisor_final_latest_review_bundle.zip`

    ZIP: final plan, kaynak/kilit hashleri, validation ve test tahminleri,
    metrikler, logs ve durumlar. Ağırlıklar ve büyük ham veri ZIP'e girmez.
    Başarısız hücrede de kısmi ZIP hazırlanır; son checkpoint Drive'da kalır.
    Test tamamlandıktan sonra aynı notebook aynı kimlikte raporu yeniden kullanır.
    Test sonuçlarına göre modeli/eşiği değiştirmek bu holdout'u geliştirme verisine
    dönüştürür; böyle bir değişiklik bu protokolün kapsamında değildir.
    '''))
    for i,entry in enumerate(content):
        entry['id']=f'final-{i:02d}'
        if entry['cell_type']=='code':compile(''.join(entry['source']),name+str(i),'exec')
    result['cells']=content
    target=Path(__file__).resolve().parents[1]/'notebooks'/name
    target.write_text(json.dumps(result,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print('Generated',name,len(content),'cells; syntax OK')


if __name__=='__main__':build()
