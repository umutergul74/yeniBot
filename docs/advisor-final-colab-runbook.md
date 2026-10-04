# Final Colab eğitim ve Eylül 2026 holdout

2026-10-04: `10_advisor_final_test_colab.ipynb` hazırlandı. Gerçek final
eğitim/test henüz çalıştırılmadı; Eylül 2026 piyasa verisi yerelde okunmadı.
Teknik testlerde kullanılan fiyatlar sentetikti.

Teknik doğrulama: 9 yeni final testi ve 50 mevcut ilgili test geçti.
Plan/sızıntı/kilit korumaları, train/test kesinti sonrası
devam, testte ağırlık hashlerinin değişmemesi ve final ZIP metrik denetimi
kontrol edildi. Saatlik-only hazırlık yolu gerçek eski 33.551 satırda A'nın
altı girdisi, ATR, etiketler, ileri getiriler ve hedef ufku ile birebir eşleşti.
Notebook kod hücreleri/nbformat şeması ve boş çıktıları doğrulandı.

## Sabit yöntem ve takvim

A–D, wavelet ve loss karşılaştırmaları validation BCE ile tamamlandı.
Seçilen yöntem 6 adet saatlik fiyat/hacim özelliği, 2×128 tek yönlü GRU,
wavelet kapalı ve saf BCE. 168.193 öğrenilen parametre; girdi 64×6.
Final karşılaştırma kapsamı bu seçilmiş yöntem ve iki basit referanstır.
Bütün A–D/loss adayları testte yeniden denenip kazanan seçilmeyecek.

| Bölüm | İlk bar (UTC) | Son bar (UTC) | Saatlik satır |
|---|---|---|---:|
| Eğitim | 18.12.2025 00:00 | 15.07.2026 23:00 | 5.040 |
| Purge | 16.07.2026 00:00 | 16.07.2026 23:00 | 24 |
| Validation | 17.07.2026 00:00 | 30.08.2026 23:00 | 1.080 |
| Test öncesi boşluk | 31.08.2026 00:00 | 31.08.2026 23:00 | 24 |
| Test | 01.09.2026 00:00 | 30.09.2026 23:00 | 720 |

210 günlük train/45 günlük validation ve 24 saatlik boşluklar önceki
geliştirme düzeninden korundu; yeni tarih seçimi test sonuçlarına bakılmadan
yapıldı. Train dizisi 4.977, validation dizisi 1.017. Tek final temporal
ayrım vardır; 38-fold geliştirme tekrar edilmez. RobustScaler yalnızca
train ile fit edilir. Testte scaler yeniden fit edilmez.

Zamanlar kaynak bar zaman damgalarıdır; kapanıştan üretilen özellikler
ilgili bar tamamlanınca erişilebilir. Hedef bundan sonraki 10 bardır.
Etiket kaynak fiyatları girişe dahil değildir. Son test etiketi için
01.10.2026 09:00 barı (10:00 UTC'de tamamlanır) gerekir.

Seedler 42/43/44; final fold offset 0. AdamW 0.001, weight_decay 0.0001,
batch 256, max 100 epoch/patience 15, clipping 1, dropout 0.2, eşik 0.5.
Epoch seçimi yalnızca pre-test validation BCE. Her seedin bu ölçütteki
en iyi checkpoint'i final ağırlıktır. Train+validation birleştirilerek
yeniden eğitim yapılmaz; validation ağırlıkları öğrenmek için kullanılmaz.

Rezervasyonun tarihsel kaydı `configs/advisor_final_test.yaml` korunur.
Çalıştırılabilir final plan `configs/advisor_final_execution.yaml` içindedir.
Final plan, eski rezervasyondaki henüz-kilitlenmedi alanlarını değiştirmeden
ayrı dosyada somutlaştırır. A–D ve 08/09 protokol imzaları korunur.

## Notebook sırası

10'u Colab Tesla T4 ile açıp hücreleri sırayla çalıştırın. Aynı
`MyDrive/yeniBot/advisor_experiments/colab_v1` kullanılmalı; A–D ve 08/09
kararları ve kontrol kayıtları burada bulunmalı. Ortam sürümleri A kontrolüyle
eşleşmezse işlem durur. Eski sonuçlarla farklı ortam sessizce karıştırılmaz.

1. **Plan:** tamamlanmış geliştirme kararları yeniden denetlenir, GRU/
   W_OFF/BCE seçimi doğrulanır; final ayarlar/veri/code/environment sözleşmesi
   `data/final_plan.json` içine değiştirilemez biçimde yazılır.
2. **prepare-fit:** dondurulmuş 2022 başlangıçlı 1H ham geçmiş korunur;
   eksik devam aylık kalıcı cache ile 31 Ağustos 09:00 UTC'ye kadar indirilir.
   Son validation hedefini hesaplamaya yeterlidir; Eylül indirilmez.
   Saatlik eski feature fonksiyonları kullanılır. Bu altı girdide 4H/OI
   olmadığından yeni 4H/OI indirilmez. Piyasa boş saat politikası korunur.
3. **train:** üç seedin final modelleri ayrı öğrenilir. Her epoch model/
   optimizer/RNG/en iyi ağırlık/scaler checkpoint'i Drive'a yazılır.
4. **freeze:** üç seed tamamlanmadan kilit oluşmaz. Seçilmiş ağırlıklar
   ve scaler parametreleri `artifacts/seed_<seed>.pt` içine kaydedilir;
   hashler ve train sınıf frekansı `frozen_manifest.json` içinde kilitlenir.
5. **prepare-test:** kilidin sözleşmesi ve üç ağırlık dosyasının hashleri
   doğrulanmadan bu komut veri indirmez. Kaynak Eylül/etiket devamına
   genişletilir. Yeniden hesaplanan fit geçmişi önceki tabloyla birebir
   aynı olmalı. İlk test dizisi için 29 Ağustos 09:00'dan başlayan 63
   önceki saat eklenir: 783 satırdan 720 tahmin çıkar, ilk 63 test saati atılmaz.
6. **evaluate-test:** donmuş ağırlık/scaler ile her seed için aynı 720
   zaman damgası değerlendirilir. Testte fit, epoch/eşik/seed seçimi yok.
   Seed sonuçları ayrı, sonra aritmetik mean ve sample std (ddof=1).
   Olasılık ensemble'ı yapılmaz; seed std güven aralığı değildir.
7. **ZIP:** final plan, metadata, validation/test tahminleri, metrikler ve
   ilerleme/log kayıtları inceleme paketine alınır. Ağırlıklar/ham veriler
   Drive'da kalır. Basit referanslar: hep-negatif sınıf kararı ve yalnızca
   training sequence etiket frekansından gelen sabit olasılık.

## Kalıcılık ve hangi aşamada kaldığı

Final çıktı alanı:
`MyDrive/yeniBot/advisor_experiments/colab_v1/final_test_v1/`.

- `data/final_plan.json`: yöntem, ayar/takvim, kaynak kod ve ortam kimliği.
- `data/fit_frame.manifest.json`: fit veri hash'i; Eylül değerlendirmesi 0.
- `runs/final/status.json`: bitmiş seedler ve o anki seed.
- `runs/final/seed_42/progress.json`, `STATUS.md`, `last.pt`: epoch checkpoint.
- `artifacts/frozen_manifest.json`: üç model/scaler kilidi ve en iyi epochlar.
- `runs/test/status.json`: test değerlendirme durumu.
- `runs/test/seed_<seed>_metrics.json`: tamamlanan seed testi ve tahmin hash'i.
- `runs/test/test_results.json`: tam kapsam sonuç ve baselines.
- `sessions/<oturum>/session.json`: mevcut/son tamamlanan notebook adımı.
- `reports/advisor_final_latest_review_bundle.zip`: burada incelenecek paket.

Aynı commit/ayar/veri/ortamla notebooku baştan çalıştırmak tamamlanan
hazırlıkları/seeds atlar, checkpoint'ten devam eder. Yarım epoch tekrar
yapılır. Test ortasında kesilirse tamamlanmış seed tahminleri korunur;
yalnızca eksikler hesaplanır. Tamamlanmış test tekrar çalıştırılınca hashleri
doğrulanıp aynı sonuç kullanılır, yeni bir yöntem denemesi yapılmaz.
GPU/ortam/source/ayar değişirse sessiz devam edilmez. Resume için ilk kod
hücresindeki `REPO_COMMIT` alanına kullanılan commit sabitlenebilir.
İki ayrı Colab runtime aynı sonuç alanına eşzamanlı yazmamalı.

Final train/val eğitim kodunda test sınırını denetlemek için boş bir takvim
işaretçisi vardır. Bu işaretçi test OHLC/etiket/özellik içermez, modeli
beslemez ve veri tablosuna test örneği eklemez. Test sınırı denetimi ile
gerçek test verisi erişimi ayrı işlemlerdir.

## Sonuçları incelemeye getirme

ZIP'i paylaşın:
`.../final_test_v1/reports/advisor_final_latest_review_bundle.zip`.
Bu sohbetin Drive'a otomatik erişimi yoktur. Yerel salt-okunur denetim:

```text
python scripts/review_advisor_final_bundle.py --bundle "<ZIP>" --output output/advisor_experiments/final_review
```

Araç ZIP hashlerini, 720 tarih/3 seed kapsamını, metrik yeniden hesaplamayı,
mean/std, validation epoch seçimini, kilit sözleşmesini ve basit referansları
kontrol eder. Model ağırlıkları ZIP'te bulunmadığından yeniden çıkarım yapmaz.

Bu dönem kullanıcı teyidiyle daha önce incelenmemiş geriye dönük holdout'tur.
Test sonucunun olumlu çıkması garanti değildir. Sonuca göre model/eşik
değiştirilirse aynı dönem artık geliştirme verisi olur; bu protokol bunu yapmaz.
