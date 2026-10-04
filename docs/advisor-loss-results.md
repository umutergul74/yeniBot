# Loss karşılaştırması — doğrulama analizi

2026-10-04; kaynak `advisor_latest_review_bundle (5).zip`. Dört yeni adayın
her biri 38 fold × seed 42/43/44 = 114 eğitim tamamladı. Saf BCE kontrolü
tamamlanan A'dan alındı. Yeni eğitim toplamı 456, epoch toplamı 10.615.

## Karar ve ortak kapsam

Önceden belirlenen seçim ölçütü eşit ağırlıklı ortalama validation BCE.
Bu ölçüte göre **L_BCE** seçildi; Drive karar dosyası da bunu kaydediyor.
Son yöntem seçimi: **6 temel fiyat/hacim özelliği → GRU → saf BCE;
wavelet kapalı**. Nihai test metrikleri henüz yok.

Tüm adaylarda 64 × 6 girdi, 2 × 128 tek yönlü GRU, 168.193 parametre,
dropout 0.2, AdamW lr=0.001/weight_decay=0.0001, batch=256, en çok 100
epoch/patience=15, gradient clipping=1.0, eşik=0.5. Train 5.040, purge 24,
validation 1.080, sonrası boşluk 24, takvim test bölümü 720 saat;
ilerleme 720 saat. Epoch seçimi her loss için normal validation BCE.

## Ortalama metrikler

| Eğitim kaybı | BCE ↓ | AP ↑ | Precision | Recall | F1 | Accuracy | Rank IC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Saf BCE | **0.594507** | **0.420872** | 0.410499 | 0.096839 | 0.142418 | 0.687868 | 0.018313 |
| Ağırlıklı BCE | 0.647174 | 0.409225 | 0.405004 | **0.537246** | **0.451214** | 0.607575 | 0.002063 |
| Focal | 0.630117 | 0.402705 | 0.412398 | 0.303510 | 0.333079 | 0.654522 | 0.012364 |
| BCE + Pearson | 0.597070 | 0.417264 | **0.426498** | 0.087385 | 0.132101 | **0.688523** | 0.017330 |
| Focal + Pearson | 0.644000 | 0.377118 | 0.368402 | 0.151038 | 0.195846 | 0.663372 | **0.018734** |

Her sayı 114 fold/seed metriğinin ortalamasıdır. Birleştirilmiş confusion
matrix metriği değildir. AP average precision'dır, trapez PR-AUC değildir.
Sınıflandırma metriklerinin eşiği 0.5. Rank IC Spearman değerlendirmesidir;
yardımcı eğitim kaybı negatif Pearson korelasyonudur.

## Bulguların anlamı

Saf BCE, olasılık tahminlerini değerlendiren BCE ve pozitif sınıfı skorla
sıralamayı değerlendiren AP'de en iyi ortalamayı verdi. Üç seedin her
birinin fold-ortalama BCE'sinde de en iyi aday saf BCE oldu. Ancak bu,
her foldta veya her metrikte üstünlük demek değildir; bağımsız test
genellemesi henüz ölçülmedi.

Ağırlıklı BCE pozitif sınıfa eğitimde daha fazla ağırlık verdi. Foldlara
göre train dizi sayılarından hesaplanan pos_weight 1.996388–2.678492.
Pozitif tahmin oranı saf BCE'de %6.29, weighted BCE'de %41.58; pozitif
etiket oranı fold ortalamasında yaklaşık %31.05. Weighted BCE çok daha
fazla pozitif karar vererek recall/F1'i yükseltti; precision benzer kaldı.
Bu 0.5 eşikte belirgin bir sınıflandırma farkıdır. Buna karşın AP ve BCE
kötüleşti: F1 artışı daha iyi olasılık tahmini veya sıralama becerisi
anlamına tek başına gelmez. Doğrudan kalibrasyon eğrisi/ECE analizi
yapılmadığından kalibrasyon hatasının büyüklüğü hakkında ayrı iddia yoktur.

Focal (alpha=0.6, gamma=2) recall/F1'i artırdı; AP/BCE'yi iyileştirmedi.
BCE+Pearson (ağırlık 0.05) precision/accuracy'de küçük artış, BCE/AP/F1
ve Rank IC'de düşüş gösterdi. Focal+Pearson, Rank IC ortalamasında küçük
artış gösterse de AP en düşük olan adaydı. Korelasyon kaybının varlığı
Spearman metriğinin tutarlı biçimde iyileşmesini garanti etmedi.

Bu sabit ayarlar için bulgular geçerlidir. Focal veya yardımcı kayıp
ailesinin her ayarda yararsız olduğu sonucu çıkarılamaz. Sonuçları gördükten
sonra seçimi F1'e çevirmedik, eşik ayarlamadık veya yeni hiperparametre
araması yapmadık. Araştırma hedefi değişecekse bu ayrı protokol olmalıdır.

## Basit referans ve dönem tutarlılığı

İncelemede ayrıca her foldun yalnızca eğitim dizilerindeki pozitif etiket
frekansını sabit olasılık tahmini olarak kullanan referans hesaplandı.
Validation BCE ortalaması 0.620458, AP 0.310459. Saf BCE'nin 0.594507
BCE ve 0.420872 AP değerleri bu basit referanstan daha iyi. Weighted BCE,
Focal ve Focal+Pearson'ın ortalama BCE'si bu referanstan kötü. Bu hesaplama
train etiket frekansını kullanır; validation'dan sabit olasılık seçilmez.

Her adayın accuracy'si hep-olumsuz referansın yaklaşık 0.689541 değerinin
altında. Bu, accuracy'yi tek başına iyi model kanıtı olarak yorumlamamayı
gerektirir; BCE/AP ile accuracy farklı davranışları ölçüyor.

Saf BCE'den daha iyi BCE veren seed-ortalama fold sayıları:

| Aday | Daha iyi fold / 38 | Saf BCE'ye göre ortalama BCE farkı |
|---|---:|---:|
| Weighted BCE | 2 | +0.052667 |
| Focal | 3 | +0.035610 |
| BCE + Pearson | 9 | +0.002563 |
| Focal + Pearson | 1 | +0.049493 |

Foldlar ve diziler örtüştüğü için 114 sonuç bağımsız istatistiksel tekrar
değildir. Güven aralığı veya anlamlılık testi hesaplanmadı; küçük
korelasyon farkları istatistiksel üstünlük olarak sunulmaz.

## Dosya ve deney denetimi

- ZIP manifestinde listelenen bütün dosyaların boyut ve hashleri doğrulandı.
- Önceki paketin 2.870 çalışma dosyası byte düzeyinde korundu.
- Yeni 456 tahmin grubunun tarih/etiket/ileri getiri ve kaynak satır
  konumları A kontrolüyle aynı.
- Bütün yeni losslar aynı hazırlanmış tablo hashini, aynı altı aktif
  özelliği, aynı mimari/parametre sayısını ve aynı çalışma ortamını kullandı.
- Ortak model/training/tarih/etiket ayarları kontrolle eşleşti; yalnızca
  önceden belirlenen loss ayarları değişti. Wavelet aktif değil.
- Tahminlerden kaydedilen metrikler tekrar hesaplanarak doğrulandı.
  Her foldta minimum validation BCE epoch'u seçilmiş.
- Weighted BCE ağırlıkları sadece eğitim dizisi etiketlerinden üretilmiş;
  training sequence sayısı 4.977. Validation sequence sayısı 1.017.
- Test değerlendirmesi bütün çalışma ve karar kayıtlarında 0.

## Çalışma kimlikleri ve eğitim bütçesi

| Aday | Run | Epoch toplamı |
|---|---|---:|
| Saf BCE kontrolü | A_81029f6d6d48 | 2.248 (önceden tamamlanan) |
| Weighted BCE | L_WEIGHTED_BCE_0903d92403a6 | 2.336 |
| Focal | L_FOCAL_e2bc949922f9 | 3.187 |
| BCE + Pearson | L_BCE_PEARSON_37dfd5e3e6f9 | 2.214 |
| Focal + Pearson | L_FOCAL_PEARSON_74fcb88b8ffb | 2.878 |

İnceleme çıktıları `output/advisor_experiments/loss_review/`:
her adayın review_summary/dashboard dosyaları, ortak loss_comparison.json,
loss_comparison.csv, training_frequency_baseline.csv ve
loss_comparison_dashboard.png. Büyük girdi tabloları ve model ağırlıkları
ZIP'te bulunmadığından burada model yeniden eğitilmedi veya yüklenmedi.

## Bundan sonraki adım

Yöntem seçimi tamamlandı. Final train/validation tarihleri, eğitim kapsamı,
raporlanacak karşılaştırmalar ve artifact kilitleme kuralları teste
bakmadan kesinleştirilmeli. Final model/scaler test öncesi veride üretilip
kilitlendikten sonra Eylül 2026 holdout değerlendirmesi yapılabilir.
Bu inceleme final eğitim veya bağımsız testin tamamlandığı anlamına gelmez.
