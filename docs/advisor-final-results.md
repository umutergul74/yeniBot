# Eylül 2026 final holdout incelemesi

2026-10-04. Kaynak: `advisor_final_latest_review_bundle.zip`.
Çalışma commit `af6d81e3899443a0ff87026b7b3d6abf4ae7db65`, Tesla T4.
Seçilen yöntem: 6 özellik/GRU/wavelet kapalı/saf BCE; eşik 0.5.

## Teknik sonuç

Paket tam bir final değerlendirmesi içeriyor. ZIP'in 64 üyesi var;
manifestte listelenen dosyaların boyut ve hashleri doğrulandı. Final plan
hash'i, model kilit manifesti, sonuç kimliği, validation epoch seçimi,
test tahminlerinden yeniden hesaplanan metrikler ve mean/std eşleşti.
Çalıştırma hatası veya metrik tutarsızlığı bulunmadı. Ham girdi tabloları
ve model ağırlıkları ZIP'te olmadığı için yerelde yeniden çıkarım yapılmadı;
dosya kayıtlarının ötesinde bütün veri/model doğruluğu iddia edilmez.

Eğitim 18.12.2025–15.07.2026, validation 17.07.2026–30.08.2026; purge 24
saat, test öncesi boşluk 24 saat. Train 5.040 satır/4.977 dizi,
validation 1.080 satır/1.017 dizi. Raw kalite kayıtlarında bozuk/silinen/
doldurulan bar yok; tarihsel bir doğrulanmış boş bar korunmuş.

Seed 42/43/44 için en iyi epoch 4/2/1, toplam epoch 19/17/16. Bu 15
epoch patience ile uyumlu erken durmadır; 100 epoch üst sınırdır.
Train loss düşerken validation BCE'nin yükselmesi, bu ayrımda eğitim
ilerledikçe validation genellemesinin kötüleştiğini gösterir. Tek başına
veri hatası kanıtı değildir; aşırı uyumla tutarlı bir eğitim davranışıdır.

Üç model/scaler kilit kaydı test indirmeden önce oluşmuş. Test fit
işlemi 0. Her seed 01.09.2026 00:00–30.09.2026 23:00 UTC arasında aynı
720 zaman damgasını değerlendiriyor. Testte 231 pozitif, 489 negatif
etiket var. 63 geçmiş bağlam satırıyla 783 satırlık giriş tablosundan
720 tahmin çıkarılmış. 2.160 tahmin kaydı 2.160 bağımsız test örneği değildir.

İki oturum kaydı bulunuyor. İkinci oturum frozen modeli üzerine yazmadan
ve yeniden eğitim yapmadan mevcut tamamlanmış test sonuçlarını kullandı.
Log: `Completed fixed test reused; no new evaluation`. Bu, farklı
yöntemlerin test üzerinde iki kez karşılaştırıldığı anlamına gelmiyor.
Train/kilit kayıtlarındaki test_evaluations=0 o aşamanın kaydıdır;
otoritatif final durum `runs/test/status.json` içinde complete/test_evaluations=1.

## Seed sonuçları

| Seed | BCE ↓ | AP ↑ | Precision | Recall | F1 | Accuracy | Rank IC |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.590138 | 0.428092 | 0.470588 | 0.242424 | 0.320000 | 0.669444 | -0.021241 |
| 43 | 0.597146 | 0.432755 | 0.318182 | 0.030303 | 0.055336 | 0.668056 | 0.006191 |
| 44 | 0.584854 | 0.470555 | 0.350000 | 0.030303 | 0.055777 | 0.670833 | 0.036833 |
| Ortalama | 0.590713 | 0.443801 | 0.379590 | 0.101010 | 0.143704 | 0.669444 | 0.007261 |
| Seed std (ddof=1) | 0.006166 | 0.023287 | 0.080397 | 0.122468 | 0.152677 | 0.001389 | 0.029052 |

Seed ortalaması probability ensemble değildir. Aynı ayın ve aynı
etiketlerin üç ayrı eğitim başlangıcına ilişkin değişkenliğidir; std
güven aralığı veya bağımsız dönem genellemesi ölçüsü olarak yorumlanmaz.
AP average precision'dır; trapez PR-AUC değildir. Rank IC burada skor ile
10 saatlik ileri getiri arasındaki Spearman korelasyonudur.

## Düşük recall/F1'in somut nedeni

| Seed | TP | FP | FN | TN | Pozitif karar sayısı / 720 |
|---|---:|---:|---:|---:|---:|
| 42 | 56 | 63 | 175 | 426 | 119 |
| 43 | 7 | 15 | 224 | 474 | 22 |
| 44 | 7 | 13 | 224 | 476 | 20 |

Gerçek 231 pozitiften seed 42 56'sını, diğer seedler yalnızca 7'şer
tanesini yakalamış. Seed 43/44 olasılıklarının en yüksek değerleri yaklaşık
0.5215/0.5159; 0.5 üstüne çok az örnek çıkıyor. Bu yüzden pozitif karar ve
recall düşük. Skorların tümü sabit veya sıfır değil; eşik altında farklı
skorlar üretmeleri AP'nin daha iyi olmasına imkân veriyor.

Seed 42 F1'de daha iyi, seed 44 BCE/AP'de daha iyi. Teste bakıp sadece
birini seçmek önceden tanımlanan üç-seed raporlama kuralını bozar. Sonuç
belirgin seed duyarlılığı gösteriyor; henüz güçlü ve kararlı bir karar
sistemi olarak tanımlanmamalı.

## Basit referanslarla karşılaştırma

Training sequence pozitif oranı 1550/4977 = 0.31143259. Her test örneğine
bu sabit olasılık verildiğinde BCE 0.627701, AP 0.320833. Modelin mean
BCE 0.590713/AP 0.443801 değerleri bu referanstan daha iyi. Bu, bu tek
ayda referansa göre olasılık/sıralama başarımı hakkında olumlu bir bulgu;
istatistiksel anlamlılık veya işlem kârlılığı kanıtı değildir.

Hep-negatif sınıf kararı accuracy 489/720 = 0.679167 verir; model ortalaması
0.669444 ile bunu geçmiyor. Fakat hep-negatifin recall/F1'i 0'dır. Farklı
metrikler farklı davranışları ölçer: modelde sıralama/olasılık katkısı var,
0.5 eşikte pozitifleri yakalama zayıf ve istikrarsız; accuracy tek başına
başarı kanıtı değildir. Rank IC ortalaması 0.007261 ve seedler arasında
işaret değiştiriyor; tutarlı ileri-getiri sıralaması gösterdiği söylenemez.

## Bundan sonra

Eylül 2026 sonucu artık görüldü ve bu protokolün final sonucudur.
Düşük F1'i düzeltmek için bu ayda eşik denemek, seed seçmek veya weighted
BCE'yi yeniden eğitip sonuca göre yöntem değiştirmek aynı holdout'u
araştırma/seçim verisine dönüştürür. Mevcut sonuç olduğu gibi raporlanmalı.

Hocaya anlatılacak bulgu: validation ile seçilen model, görülmemiş tek
ayda sabit train-frekansı referansından daha iyi BCE/AP üretmiş; ancak
pozitif sınıf recall/F1'i düşük ve seedlere duyarlı, Rank IC çok zayıf.
İleride sınıflandırma eşiği/olasılık kalibrasyonu, eğitim kararlılığı ve
değişen dönem davranışları ayrı geliştirme protokolünde araştırılabilir;
yeni yöntem için yeni dokunulmamış değerlendirme gerekir. Bu inceleme
eşik veya modeli değiştirmedi, yeni eğitim/test başlatmadı.

Yerel doğrulama çıktıları:
`output/advisor_experiments/final_review/verified_test_metrics.csv` ve
`review_summary.json`. İnceleme aracı
`scripts/review_advisor_final_bundle.py` başarıyla tamamlandı.
