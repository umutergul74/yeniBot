# Wavelet karşılaştırması — doğrulama sonuçları

İnceleme tarihi: 2026-10-04. Kaynak: kullanıcının
`advisor_latest_review_bundle (4).zip` dosyası. Çalışma:
`W_ON_e170a14b3a9d`, Colab commit `021d59a91969e82d88308c01fd1941091d65f482`.

## Karar

Önceden belirlenen ortak ölçüt 114 fold/seed'in eşit ağırlıklı ortalama
validation BCE'sidir. Daha düşük BCE tercih edilir. Bu ölçüte göre
**wavelet kapalı A kontrolü seçildi**. Drive karar dosyası da W_OFF diyor.

| Ölçüt | Wavelet kapalı (A) | Wavelet açık |
|---|---:|---:|
| BCE ↓ | 0.594507 | 0.595259 |
| Average Precision ↑ | 0.420872 | 0.419435 |
| Precision ↑ | 0.410499 | 0.426119 |
| Recall ↑ | 0.096839 | 0.088158 |
| F1 ↑ | 0.142418 | 0.129448 |
| Accuracy ↑ | 0.687868 | 0.687186 |
| Spearman Rank IC ↑ | 0.018313 | 0.020638 |

Değerler fold/seed metriklerinin aritmetik ortalamasıdır; bütün tahminleri
birleştirerek hesaplanan tek bir confusion-matrix metriği değildir.
Precision/recall/F1/accuracy eşik 0.5 ile hesaplandı. AP, trapez PR-AUC
değildir. Bütün değerler validation verisindendir; nihai test sonucu yoktur.

Wavelet açık BCE farkı +0.0007522796'dır. 114 fold/seed'in 49'unda,
seedler fold içinde ortalandığında 38 foldun 16'sında daha iyi BCE vardır.
Precision ve Rank IC yükselse de recall, F1 ve AP iyileşmemiştir.
Fark küçüktür; bu inceleme istatistiksel anlamlılık testi yapmaz.
'Wavelet genel olarak işe yaramaz' sonucu çıkarılamaz; yalnızca bu iki
kanal, sabit dönüşüm ayarları, mimari ve seçim ölçütü için yarar görülmedi.

Her iki accuracy değeri de hep-olumsuz kararın yaklaşık 0.689541
accuracy değerini aşmıyor. Accuracy tek başına model becerisi kanıtı değildir.

## Karşılaştırma ve kayıt denetimi

- 38 fold × seed 42/43/44 = 114 eğitim tamamlanmış; toplam 2.189 epoch.
- Yeni ZIP'in manifestinde listelenen dosyaların boyut/hashleri doğrulandı.
- Önceki ZIP'teki 2.296 A–D çalışma dosyası byte düzeyinde aynı kaldı.
- 114 eşleşen validation tahmin grubunun zaman damgaları, etiketleri,
  ileri getirileri ve dizi sonu satır konumları birebir eşleşti.
- Kaydedilen metrikler tahminlerden tekrar hesaplanarak doğrulandı;
  seçilen epoch her foldun minimum validation BCE epoch'u.
- Sabit ham 1H/4H kaynak hashleri ve temel A tablosunun hash'i korundu.
  Hazırlık manifesti temel sütun/etiketlerin değişmediğini kaydediyor;
  büyük girdi tablosu ZIP'te bulunmadığından bu incelemede yeniden üretilmedi.
- Geliştirme kapsamı 33.551 saatlik satır, 2022-03-01 04:00–2025-12-28
  02:00 UTC. Aktif girdi 64 × 6; yalnızca getiri ve hacim kanalları
  geçmişe dayalı wavelet karşılıklarıyla değişti.
- db4, geçmiş pencere 256, level=2, threshold_scale=0.5. Ham OHLC/ATR
  etiketleri kullanıldı. Purge ve validation sonrası boşluk 24'er saat.
- Ortam Tesla T4 ve A kontrolünün aynı Python/PyTorch/NumPy/pandas/
  sklearn/CUDA/cuDNN sürümleri. Test değerlendirmesi kaydı 0.

Örtüşen girdiler, hedefler ve validation dönemleri bağımsız gözlemler
oluşturmaz. 114 eğitim bağımsız istatistiksel tekrar olarak yorumlanmaz.

## Sonraki aşama

09 loss notebooku W_OFF seçimini otomatik okuyacak. Girdi A'nın altı ham
özelliği, mimari GRU olacak. Saf BCE kontrolü A'dan alınacak; dört yeni
loss (ağırlıklı BCE, Focal, BCE+Pearson, Focal+Pearson) aynı protokolle
denenecek. Epoch/erken durdurma ve yöntem seçimi normal validation BCE;
diğer metrikler ikincil. Eylül 2026 testi henüz açılmaz.

Yerel inceleme çıktıları:
`output/advisor_experiments/wavelet_review/` altında
`review_summary.json`, `paired_comparison.json`, `wavelet_comparison.csv`,
`W_ON_DENEYI_INCELEME.md` ve `W_ON_validation_dashboard.png`.
