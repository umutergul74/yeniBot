# A–B–C: doğrulama sonuç incelemesi

C tamamlandı: 38 fold × üç seed = 114 eğitim, toplam 1.892 epoch.
Girdi: B ile aynı 34 özellik; mimari TCN; saf BCE. Wavelet kapalı.
Tüm ölçümler validation'dır; test değerlendirmesi yapılmadı.

| Ölçüt, fold/seed ortalaması | A: 6 / GRU | B: 34 / GRU | C: 34 / TCN |
|---|---:|---:|---:|
| BCE, düşük daha iyi | 0,594507 | 0,612412 | 0,607080 |
| Average Precision | 0,420872 | 0,397423 | 0,397981 |
| Precision, eşik 0,5 | %41,05 | %44,78 | %37,99 |
| Recall, eşik 0,5 | %9,68 | %11,49 | %8,69 |
| F1 | 0,142418 | 0,159500 | 0,121095 |
| Accuracy | %68,79 | %67,77 | %68,58 |
| Rank IC | 0,018313 | 0,001562 | 0,007430 |

Önceden belirlenen seçim ölçütü validation BCE'dir. Bu ölçütte mevcut sıra
A, C, B. C, B'ye göre 38 foldun seed ortalamasında 25 foldda daha iyi;
A'ya göre sadece 8 foldda daha iyi. Bunlar betimsel sayımlardır;
istatistiksel anlamlılık veya bağımsız tekrar iddiası kurulmadı.
C'nin AP'si B'ye çok yakın; precision/recall/F1 daha düşük. Rank IC küçük.
D tamamlanmadan nihai yapı seçilmez; eşik ve ölçüt sonuçlara göre değiştirilmez.

## Doğrulanan kayıtlar

- ZIP envanterindeki 1.764 dosyanın boyutu ve SHA256'sı eşleşiyor.
- Önceki ZIP'e göre A'daki 574 ve B'deki 574 çalışma dosyası değişmemiş.
- A/B/C'nin 114 tahmin grubunda tarihler, etiketler, ileri getiriler ve satır konumları aynı.
- B/C özellik listesi ve veri hash'i aynı; splitler, seedler, genel eğitim ayarları ve ortam eşleşiyor.
- Epoch seçimleri minimum validation BCE ile uyumlu; erken durma patience 15.
- Tahminlerden AP, precision, recall, F1, accuracy, Rank IC ve BCE yeniden hesaplandı.
- C'deki 228 kayıtlı sınır denetimi geçti; 76 benzersiz bölüm sınırı var.
- Her validation fold/seed dosyasında 1.017 dizi; eğitimde 4.977 dizi.
- OI kalite kaydı B ile aynı: %99,7705 kapsam ve 77 saatlik satırda iki OI kanalının nötr doldurulması.

C'de en iyi epoch ortalaması 1,60; 60/114 eğitimde ilk epoch seçilmiş.
İlk fold örneğinde eğitim BCE'si azalırken validation BCE'si sonraki
epochlarda artıyor; bu aşırı uyumla tutarlı bir örüntüdür. Bu gözlem tek
başına sorunun nedenini veya gerekli hiperparametre değişikliğini belirlemez.

B'nin GRU modeli 178.945, C'nin TCN modeli 67.457 parametreli. Deney,
bu önceden sabitlenen mimari ayarlarını karşılaştırır; parametre sayısı
eşitlenmediği için sonuç bütün TCN/GRU modellerine genellenemez.
A ile tam özellik seti karşılaştırması da üç oynaklık kanalının çıkarılmasını
içerir: `realized_vol_14`, `gk_vol_14`, `atr_14_pct` B/C'de yoktur.

Kaynak veriler ve model ağırlıkları ZIP'te bulunmadığından bunlar bağımsız
yeniden üretilemedi. Hazırlık audit'leri paketteki kayıtlardır. Örtüşen foldlar,
64 gözlemli girdiler ve 10 saatlik hedefler nedeniyle örnekler bağımsız sayılmaz.
Doğrulama performansı bağımsız nihai test başarısı olarak sunulmamalıdır.

## Devam

07 notebookunda `EXPERIMENTS = ['D']` seçilerek aynı girdiler, aynı BCE ve
aynı seçim kurallarıyla paralel TCN–GRU denenir. A/B/C yeniden çalıştırılmaz.
Ardından validation sonuçlarına göre yapı seçimi, wavelet ve loss karşılaştırmaları.
Nihai bağımsız test dönemi ayrıca sabitlenmelidir; henüz tanımlanmamıştır.

C için önceki oturum eğitim loglarını içeriyor; son kısa tekrar tamamlanmış
kayıtları kullanmış. Son oturum süresi tüm eğitimin süresi değildir.
İnceleme araçları: `review_advisor_bundle.py --experiment C` ve
`compare_advisor_architectures.py`. Yerel çıktı: `output/advisor_experiments/review_C_20261003`.
