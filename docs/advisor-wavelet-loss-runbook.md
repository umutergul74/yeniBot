# Wavelet ve loss karşılaştırmaları

## Durum ve deney sırası

2026-10-04: Kod ve Colab notebookları hazırlandı. Gerçek Colab wavelet/loss
eğitimleri henüz yapılmadı. Yerelde yalnızca teknik doğrulama yapıldı.
Eylül 2026 nihai testi bu notebooklarda okunmaz ve değerlendirilmez.

1. `08_advisor_wavelet_colab.ipynb` dosyasını GPU/T4 ile baştan çalıştır.
2. W_ON'un 38 fold × 3 seed = 114 eğitimi tamamlanınca wavelet kararı
   otomatik kaydedilir. İnceleme ZIP'ini getir.
3. `09_advisor_losses_colab.ipynb` dosyasını aynı Drive alanı ve ortamda çalıştır.
4. Dört yeni loss deneyi de 114 eğitim tamamladığında loss kararı kaydedilir.
   İnceleme ZIP'ini getir. Bundan sonra nihai eğitim/test adımı hazırlanır.

07 notebooku A–D içindir; yeni deneyler için onu düzenlemeye gerek yoktur.
08/09 aynı `MyDrive/yeniBot/advisor_experiments/colab_v1/` alanını kullanır;
A–D kayıtları ve sabit ham girdiler korunur. Yeni veri indirilmez. Kontrol
verisi/ortamı veya tamamlanan fold sayıları değişirse karşılaştırma durur.

## Neden A temel alınıyor?

Önceden belirlenen birincil seçim ölçütü, 114 fold/seed'in eşit ağırlıklı
ortalama **validation BCE** değeridir. Tamamlanan sonuçlar:

| Deney | Girdi ve mimari | Ortalama validation BCE |
|---|---|---:|
| A | 6 temel özellik, GRU | 0.594507 |
| B | 34 mevcut özellik, GRU | 0.612412 |
| C | 34 mevcut özellik, TCN | 0.607080 |
| D | 34 mevcut özellik, paralel TCN–GRU | 0.605371 |

Daha düşük değer daha iyidir; seçilen yapı **A**'dır. F1/AP sonuçlarına
bakarak bu kararı değiştirmiyoruz. Parametre sayıları eşit değildir;
A–B iki farklı özellik kümesidir, yalnızca ek özelliklerin katkısı olarak
yorumlanmaz. Bunlar validation sonuçlarıdır, nihai test başarımı değildir.

## Wavelet deneyinin kapsamı

W_OFF kontrolü, tamamlanan A sonucudur; tekrar eğitilmez. W_ON aynı GRU,
6 girdi, hedefler, fold tarihleri, seedler ve BCE ile çalışır. Yalnızca:

| Ham girdi | W_ON girdisi |
|---|---|
| `log_return` | Geçmiş pencereye uygulanan wavelet sonrası kapanış getirisi |
| `volume_log_zscore` | Wavelet sonrası hacmin log1p kayan z-skoru |

Diğer dört girdi (gerçekleşen oynaklık, Garman–Klass oynaklığı, ATR yüzdesi,
VWAP uzaklığı) ham veriden hesaplanan değerlerini korur. A 4H girdileri
kullanmadığı için burada 4H kanallarını dönüştürmüyoruz. Bu deney,
**iki giriş kanalını filtrelemenin etkisini** ölçer; bütün girdilerin veya
etiketlerin filtrelendiği bir deney değildir.

Her hesaplama yalnızca geçmişteki 256 saatlik pencereyi kullanır:
db4, level=2, threshold_scale=0.5. Hacim dönüşümündeki negatif değerler
log1p öncesi sıfıra sınırlandırılır. Başlangıç geçmişi sabit ham kaynaktan
alınır; geliştirme satırları düşürülmez. Ham OHLC/ATR üzerinden hesaplanan
triple-barrier etiketleri ve ileri getiriler değişmez. İki ek sütun tabloda
saklanır; modelin aktif girdi boyutu yine 64 × 6'dır.

W_ON/W_OFF aynı validation BCE ölçütüyle karşılaştırılır. Tam eşitlikte
önceden tanımlanan tercih W_OFF'tur. Wavelet parametreleri için arama
yapılmaz; burada tek sabit dönüşüm denenir.

## Loss deneyleri

09, 08'de seçilen wavelet durumunu otomatik kullanır. Girdi, mimari,
seedler, optimizer ve eğitim bütçesi sabit tutulur:

| Kimlik | Eğitim kaybı | Sabit ayar |
|---|---|---|
| L_BCE | Binary Cross-Entropy | Seçilmiş tamamlanan kontrol yeniden kullanılır |
| L_WEIGHTED_BCE | Pozitif sınıf ağırlıklı BCE | Her foldun eğitim dizilerinde negatif/pozitif oranı |
| L_FOCAL | Focal Loss | alpha=0.6, gamma=2 |
| L_BCE_PEARSON | BCE + yardımcı korelasyon kaybı | Yardımcı ağırlık 0.05 |
| L_FOCAL_PEARSON | Focal + yardımcı korelasyon kaybı | alpha=0.6, gamma=2, yardımcı ağırlık 0.05 |

Sınıf ağırlığı yalnızca eğitim dizilerinin son satırındaki etiketlerden
hesaplanır; validation/test dağılımı kullanılmaz. Focal Loss, kolay
örneklerin ağırlığını azaltır. Yardımcı kayıp, sigmoid skorları ile
10 saatlik ileri getirilerin batch içi **negatif Pearson korelasyonudur**.
Mevcut sınıfın adı `RankICLoss` olsa da bu, doğrudan Spearman kaybı değildir.
Spearman Rank IC yalnızca raporlanan yardımcı değerlendirme ölçütüdür.

Bütün kayıplarda epoch seçimi ve erken durdurma **normal, ağırlıksız
validation BCE** üzerinden yapılır. Farklı eğitim kayıplarının ham sayısal
değerleri birbirleriyle karşılaştırılmaz. Nihai loss seçimi de aynı
114 fold/seed kapsamındaki ortalama validation BCE ile yapılır.
F1/precision/recall 0.5 eşikte; AP, accuracy ve Rank IC ayrıca raporlanır.
Ağırlıklı/Focal kayıplar olasılık kalibrasyonunu etkileyebilir; BCE ölçütü
buna duyarlıdır. Eşik veya kalibrasyon bu aşamada ayrıca optimize edilmez.

Varsayılan dört yeni loss deneyi sırayla çalışır; W_ON ile birlikte
toplam **570 yeni fold/seed eğitimi** vardır. GPU süresi için 09 içindeki
`EXPERIMENTS` listesi tek bir loss kimliğine indirilebilir. Diğerlerini
sonraki oturumda çalıştır; sonunda dört kimlikli varsayılan listeyle tekrar
çalıştırmak tamamlananları atlar ve ortak seçimi tamamlar. Eksik kapsamla
'en iyi loss' seçilmez. Aynı Drive alanına iki runtime eşzamanlı yazmasın.

## Kalıcılık ve inceleme

Her epoch sonunda model, optimizer, RNG ve en iyi model `last.pt` içine
kaydedilir. `progress.json`/`STATUS.md` son epoch'u; çalışma düzeyindeki
`status.json` tamamlanan fold/seed sayısını gösterir. Aynı kod, ayar, veri
ve ortamla yeniden çalıştırınca son kaydedilen epoch'tan devam eder.
Epoch ortasında kesilirse o epoch tekrar yapılır.

Kontrollerin Colab Python/PyTorch/NumPy/pandas/sklearn/CUDA/cuDNN ve GPU
ortamıyla eşleşme zorunludur. Ortam değişirse sessizce eski kontrolle yeni
sonuç birleştirilmez; eşleşen ortam veya ayrı kontrol protokolü gerekir.
Eğitim sürerken kodu/ayarları güncellemeyin; oturumun commit kaydı saklanır.

Drive `data/` altında kalıcı kararlar:

- `stages_architecture_selection.json`: A–D karşılaştırması ve seçilen A.
- `stages_wavelet_run.json`: tamamlanan W_ON çalışma kimliği.
- `stages_wavelet_selection.json`: W_OFF/W_ON karşılaştırması ve kararı.
- `stages_loss_selection.json`: beş kaybın karşılaştırması ve kararı.

Karar dosyaları farklı içerikle üzerine yazılmaz. Her çalışma protokolü
girdi/kod hashlerini, gerçek eğitim kaybını ve test değerlendirmesi=0
kaydını içerir. Validation tahminleri, metrikler, eğitim geçmişleri ve
kararlar `reports/advisor_latest_review_bundle.zip` içine girer.
Ağırlıklar ve büyük girdi tabloları Drive'da kalır. Yerel inceleme aracı
W_ON ve dört yeni loss kimliğini de denetleyebilir.

## Teknik doğrulamanın sınırı

Yerelde küçük CPU/CUDA deneylerinde yeni BCE yolu eski BCE yolu ile aynı
ağırlıkları/sonuçları verdi. Focal+Pearson kesinti sonrası devam testi,
gelecek veriyi değiştirince geçmiş wavelet değerlerinin sabit kalması,
seçim kapıları ve gerçek 33.551 satırlık hazırlık kontrol edildi.
Bunlar gerçek Colab eğitimlerinin tamamlandığı anlamına gelmez.
