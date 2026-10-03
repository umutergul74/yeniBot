# Danışman deneyleri — çalışma ve devam kaydı

Başlangıç: 2026-10-03. Dal: `codex/advisor-ablation-protocol`.
Kaynak dal: `main`; geçmiş deney ve sonuçlar korunur.

## Amaç ve sıra

1. Bölüm sınırlarını 10 saatlik etiket ufkuna uygun yap; otomatik denetle.
2. A: temel fiyat/hacim özellikleri + GRU + saf BCE.
3. B: açıkça dondurulmuş mevcut özellik listesi + GRU + saf BCE.
4. C: aynı özellikler + TCN + saf BCE.
5. D: aynı özellikler + paralel TCN-GRU + saf BCE.
6. Validation sonucuyla seçilen mimaride wavelet kapalı/açık karşılaştırması.
7. Aynı yapıda önceden belirlenen loss karşılaştırması.
8. Yapı ve seçim kuralları sabitlendikten sonra bağımsız nihai test.

## Araştırma kuralları

- Train/validation ve validation/test sınırlarında 24 boş saatlik bar.
- Sınır kontrolü: önceki bölümde hedef için kullanılan son zaman sonraki
  bölümün başlangıcından önce olmalı. 24 sayısı optimum iddiası değildir.
- Gözlem sırası ve saatlik grid denetlenir; eksik saatler sıkıştırılmaz.
- Ölçekleyici yalnızca foldun eğitim verisinde fit edilir.
- Diziler bölüm ayrımından sonra, her bölüm içinde oluşturulur; uzunluk 64.
- İlk deneylerde wavelet kapalı; etiket ham OHLC ve ATR üzerinden sabit.
- Saf BCE; yardımcı korelasyon/regresyon/pairwise kayıpları kapalı.
- Epoch seçimi ve erken durma validation BCE. Eşik 0.5, sonradan ayarlanmaz.
- AP, precision, recall, F1, accuracy ve Rank IC doğrulamada raporlanır.
- Geliştirme komutu test tahmini üretmez. Tarihsel testler görülmüş veridir;
  yeni bir çalıştırma bunları bağımsız nihai teste dönüştürmez.
- Bütün foldların validation sonuçlarından seçim yapılırsa bunlarla örtüşen
  eski testler nihai bağımsız test olarak sunulmaz.
- Nihai test tarihleri henüz belirlenmedi; sonuçlara bakarak seçilmez.

## Kesinti sonrası devam

Her eğitim epoch'u atomik checkpoint ile saklanacak: model, optimizer,
en iyi validation modeli, erken durma sayacı ve rastgele sayı durumları.
Veri ve ayar imzaları değişmişse eski checkpoint kullanılmayacak.
Çalışma günlüğü epoch/fold durumunu gösterecek; biten foldlar atlanacak.
`docs/advisor-experiments-status.md` dosyası insan tarafından okunabilen
devam noktasıdır. Eğitim çıktıları `output/advisor_experiments/` altında.
Veri, model ağırlıkları ve çalışma anı JSON dosyaları Git'e eklenmez.

## Mevcut durum

- [x] Yeni dal açıldı.
- [x] Eski kod, model, veri envanteri ve deney kuralları incelendi.
- [x] Güvenli sınır denetimi ve regresyon testleri.
- [x] GRU/TCN mimari seçenekleri ve validation odaklı eğitim komutu.
- [x] Veri hazırlığı, kaynak hashleri ve başlangıç denetimi.
- [x] Kesinti/devam testi ve küçük teknik eğitim kontrolü.
- [ ] A deneyinin tam eğitimine başlama.
- [ ] B–D; wavelet ve loss deneyleri.
- [ ] Ayrı nihai test.

## İlk aşama sonuçları

- `python -m pytest tests/test_advisor_protocol.py tests/test_training_integration.py -q`:
  15 test geçti. Bunlar arasında CPU ve CUDA'da bilerek kesilen eğitimin
  kesintisiz eğitimle birebir eşleşmesi var.
- İnsan tarafından okunabilen otomatik `STATUS.md` kayıtları eklendikten
  sonra yeni protokolün 7 testi tekrar geçti.
- Hazırlanan geliştirme tablosu 33.551 satır ve kesintisiz saatlik grid.
- 38 fold için 76 bölüm sınırı kontrolü geçti. Fold takvimi:
  `output/advisor_experiments/data/fold_calendar.csv`.
- İlk fold: train 2022-03-01 04:00–2022-09-27 03:00;
  validation 2022-09-28 04:00–2022-11-12 03:00;
  ayrılan test bölümü 2022-11-13 04:00–2022-12-13 03:00 UTC.
  Eski 6 saatlik boşluktaki test takvimi aynen kullanılmaz.
- Gerçek veri teknik kontrolü: 1 fold, seed 42, 2 epoch; seçilen epoch 1,
  validation BCE 0.562323. **Bu kontrol nihai performans sonucu değildir.**
- Teknik kontrol çıktısı:
  `output/advisor_experiments/technical_check/A_3ba7077570fc/`.
- Test değerlendirmesi: 0.
- B–D için açık listede 34 wavelet dışı özellik var; açık pozisyonun iki
  sütunu eksik. Kaynak tamamlanmadan tam özellik deneyi başlatılmayacak.
- Wavelet ve farklı loss aşamalarının çalıştırılması henüz uygulanmadı.

Sonraki adım: A deneyini 38 fold × 3 seed, en fazla 100 epoch ve
validation BCE / patience 15 ile ayrı arka plan işleminde başlatmak.
Komut: `powershell -NoProfile -File scripts/run_advisor_experiment.ps1 -Experiment A`.
Devam ayrıntıları: `docs/advisor-experiment-runbook.md`.
Bu belge eğitim tamamlandı veya akademik başarı sağlandı anlamına gelmez.
