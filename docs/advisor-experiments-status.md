# Danışman deneyleri — çalışma ve devam kaydı

Başlangıç: 2026-10-03. Dal: `codex/advisor-ablation-protocol`.
Kaynak dal: `main`; geçmiş deney ve sonuçlar korunur.

## Güncel durak — A–D tamamlandı; bağımsız test dönemi kararı bekleniyor

Paket: `C:/Users/Umut/Downloads/advisor_latest_review_bundle (3).zip`.
SHA256: `a96dde6cfbe5defd7078814060a187675ac237cdb1df6c629df97ca0f4db0aef`.
D kapsamı `D_e4e9d0b149a6`: 38 fold × seed 42/43/44, 114 tamamlanmış eğitim,
1.904 epoch, paralel TCN–GRU / 34 girdi / BCE. Kayıtlı parametre sayısı 246.145.
2.344 envanter dosyasının hash/boyutları doğrulandı. D metrikleri tahminlerden
yeniden hesaplandı; minimum validation BCE epoch seçimi ve 228 sınır denetimi
geçti. Test değerlendirmesi 0.

A/B/C'nin önceki paketteki 574'er çalışma dosyası değişmemiş. A–D'nin 114
tahmin grubunda tarihler/etiketler/getiriler/satır konumları aynı; B/C/D'nin
34 özellik listesi ve veri hash'i aynı. OI audit'i aynı (%99,7705 kapsam,
77 saatlik satırda iki nötr OI kanalı). Fold/seed/eğitim ayarları/ortam eşleşiyor.

D validation: BCE 0,605371; AP 0,408339; precision %37,72; recall %9,64;
F1 0,132345; accuracy %68,50; Rank IC 0,013251. BCE sırası A → D → C → B.
B F1 bakımından önde. D'nin BCE'si seed-ortalama 38 foldun 29'unda B'den,
25'inde C'den, 10'unda A'dan iyi. İstatistiksel anlamlılık iddiası kurulmadı.

Rapora göre A validation BCE ile bir sonraki aşama için seçim adayıdır;
henüz nihai test başarısı veya genel GRU üstünlüğü iddia edilemez. A'nın
6 girdisi ile B/C/D'nin 34 girdisi farklı setlerdir; kapasite de farklıdır.
56/114 D eğitiminde ilk epoch seçilmiş; en iyi epoch ortalaması 1,70.

Sıradaki gerekli karar: dokunulmamış nihai test dönemi. Kullanıcıdan Eylül
2026'nın önceki eğitim/backtest/performans değerlendirmelerinde görülüp
görülmediği soruldu; henüz yanıt yok. Yerel ham snapshot 30 Ağustos 2026'ya
kadar uzanıyor; 2026 verileri eski araştırmalarda kullanılmış olabileceğinden
2026'nın tamamı otomatik bağımsız test sayılamaz. Test tarihi henüz sabitlenmedi.
Yeni wavelet/loss veya test eğitimi başlatılmadı; mevcut sonuçlar korunuyor.

Ayrıntı: `docs/advisor-abcd-validation-review.md`.
JSON/CSV/grafikler: `output/advisor_experiments/review_D_20261003`.

## Önceki durak — A, B ve C tamamlandı

Yeni kullanıcı paketi: `C:/Users/Umut/Downloads/advisor_latest_review_bundle (2).zip`.
SHA256: `bd762d77567ec36e1032ca8b3f5b16eb3e183909505c1f620ec8cd24d2f96fba`.
C `C_de8fa1965470`: 38 fold × seed 42/43/44, 114 eğitim, 1.892 epoch.
1.764 envanter dosyasının hash/boyutları doğrulandı. C metrikleri tahminlerden
yeniden hesaplandı; epoch seçimleri validation BCE ile uyumlu; 228 kayıtlı
sınır kontrolü geçti. Test değerlendirmesi 0.

Önceki paketle A'nın 574, B'nin 574 çalışma dosyası byte düzeyinde aynı.
B/C aynı 34 girdiye ve veri hash'ine sahip; A/B/C'nin 114 tahmin grubunda
etiket, tarih, getiri ve satır konumları eşleşti. Split/seed/eğitim ayarları
ve ortam aynı. A'nın farklı altı özellik setiyle ilgili yorum sınırı korunur.

C validation: BCE 0,607080; AP 0,397981; precision %37,99; recall %8,69;
F1 0,121095; accuracy %68,58; Rank IC 0,007430. BCE'de C, B'den iyi fakat
A'dan zayıf. Seed ortalamasında 38 foldun 25'inde B'den, 8'inde A'dan daha iyi.
C 67.457, B 178.945 parametreli; kapasite eşitlenmiş mimari karşılaştırması değildir.
C'de 60/114 eğitim için ilk epoch seçilmiş. Uzun eğitimin faydası varsayılmamalı;
erken durdurma kaydı korunur, mevcut deneyin ayarları sonuçla değiştirilmez.

C iki oturum kaydında var: 18:30:15 UTC oturumu eğitim loglarını içeriyor;
19:07:03 UTC tekrarı tamamlanmış scope'u yeniden kullanmış. İkinci kısa
oturumun süresi tüm eğitimin süresi olarak yorumlanmamalı.

Rapor: `docs/advisor-abc-validation-review.md`.
Yerel JSON/CSV/grafik: `output/advisor_experiments/review_C_20261003`.
Sonraki aşama: 07 notebookunda `EXPERIMENTS = ['D']`, aynı 34 girdi → paralel
TCN–GRU → BCE. D henüz çalıştırılmadı. Nihai test dönemi hâlâ tanımlanmadı.

## Önceki durak — A ve B tamamlandı

Yeni kullanıcı paketi: `C:/Users/Umut/Downloads/advisor_latest_review_bundle (1).zip`.
B kapsamı `B_67a0db4c5eb3`: 38 fold × seed 42/43/44 = 114 tamamlanmış eğitim,
1.865 epoch. T4 oturumu, kaynak commit `43e6692`. Test değerlendirmesi 0.
Paketin 1.178 envanter dosyasının SHA256/boyutları ve tahminlerden AP,
precision/recall/F1/accuracy/Rank IC/BCE yeniden hesaplamaları doğrulandı.
A'nın önceki paketteki 574 çalışma dosyası yeni pakette byte düzeyinde aynı.

A–B: aynı etiket/zaman/getiri kayıtları 114 çift dosyada eşleşiyor. Eğitim,
model ayarları, foldlar, seedler ve ortam aynı. A veri referans hash'i eşleşmiş.
OI: 419.628 kaynak kaydı, 475 kullanılamayan kaynak ölçümü; geliştirme satırı
kapsamı %99,7705, iki OI kanalında 77 nötr doldurma, piyasa satırı silinmedi.
Sınır kontrolleri her deneyde 228 kayıt / 76 benzersiz sınır; hepsi geçti.

Validation ortalamaları: A/B BCE 0,594507/0,612412; AP 0,420872/0,397423;
F1 0,142418/0,159500; Rank IC 0,018313/0,001562. B'de eşik 0,5 precision ve
recall artıyor; önceden belirlenen seçim ölçütü BCE ise A lehine. 38 foldun
seed ortalamasında BCE bakımından B yalnızca 7 foldda daha iyi.

Yorum sınırı: B A'nın üst kümesi değil. A'daki `realized_vol_14`, `gk_vol_14`,
`atr_14_pct` B'nin dondurulmuş 34 özellik listesinde yok. Bu, iki özellik seti
karşılaştırması; yalnızca ek özelliklerin veya OI'nin marjinal katkısı değil.
C/D, B'nin aynı 34 girdisini kullanmalı; liste sonuçlara göre değiştirilmemeli.

Yerel yeniden üretilebilir inceleme:
`scripts/review_advisor_bundle.py --experiment A|B` ve
`scripts/compare_advisor_ab.py`; çıktı `output/advisor_experiments/review_B_20261003`.
Ayrıntı: `docs/advisor-ab-validation-review.md`.
Sıradaki deney C (34 girdi → TCN → BCE); eğitim henüz başlatılmadı.
Nihai bağımsız test dönemi sabitlenmeli; mevcut sonuçlar validation'dır.

## Geçmiş durak — B kaynak hatalarının çözülmesi

İkinci B hazırlık hatası (aynı gün): Mart 2022 OI arşivindeki sıfır ölçümler
katı kaynak kontrolünde reddedildi. Gerçek arşiv indirildi ve doğrulandı:
8.928 beş dakikalık kaydın 117'sinde iki OI alanı sıfır. Yeni politika bu
kayıtları silmeden ham snapshot'ta korur; ölçüm ve onu izleyen log-değişim
kullanılamaz sayılır. Başka bir eski satıra geri doldurularak gizlenmez.
Kaynak kalite sayıları, örnek değerler ve tarihler manifest/audit'e yazılır.
%99 geliştirme satırı kapsam şartı korunur; düşük kapsam eğitim başlatmaz.
Ocak/şubat gibi önceki sürümle tamamlanan aylık cache dosyaları yeniden kullanılır.
13 OI/Colab testi geçti; gerçek Mart verisinde ayrıca ileriye eşleşme ve temel
sütun değişimi olmadığı doğrulandı. Tüm dönem kapsamı hâlâ Colab'da ölçülecek.

2026-10-03: B hazırlığı iki OI sütunu bulunmadığı için doğru biçimde durmuştu.
`07_advisor_full_features_colab.ipynb` artık mevcut Drive OI kaynağını alır veya
Binance Vision arşivini aylık devam kayıtlarıyla indirir. Tamamlanmış A referansı
korunur; temel veri hash'i eşleşmeden B başlamaz. OI kapsamı en az %99 olmalıdır;
küçük kalan eksikler açıkça raporlanan nötr doldurmadır. Tam özellik sayısı 34.

26 protokol, ham veri, Colab ve OI testi geçti. İki notebookun nbformat şeması
ve hücre sözdizimi doğrulandı. Ayrıca mevcut yerel temel veri ve **sentetik OI**
ile hazırlık entegrasyonu doğrulandı: 33.551 temel satır değişmedi, eksik tam
girdi kalmadı. Bu kontrol gerçek OI kapsamı veya B performans sonucu değildir.
Yerel uzun eğitim başlatılmadı. B/C/D, wavelet ve loss eğitimleri bekleniyor.

Sıradaki işlem: kullanıcı yeni 07 notebookunu aynı Drive alanında baştan
çalıştırır, OI audit'i kontrol edilir, B tamamlanınca güncel ZIP incelenir.
Nihai bağımsız test dönemi henüz sabitlenmedi; bu akış test değerlendirmez.

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
- [x] A deneyinin tam eğitimine başlama.
- [x] A deneyinin 38 fold × 3 seed kapsamını tamamlama ve validation raporu.
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

## Aktif aşama — A tamamlandı, B için veri hazırlığı sırada

Kullanıcının `advisor_latest_review_bundle.zip` paketi 2026-10-03 tarihinde
incelendi. Colab A çalışması `A_81029f6d6d48` tamamlandı: 38 fold × 3 seed
= 114 eğitim, 2.248 epoch, Tesla T4. Kaynak commit `d9fc084`.
ZIP SHA256: `9b25fb39472545ba337afbd0185a18f8b7aad67a1947eac26181f12442e82a74`.
588 dosyanın hash/boyutu, tüm tahminlerin metrikleri, epoch seçimleri ve
76 benzersiz sınır kontrolü doğrulandı. Model ağırlıkları ZIP'e dahil değil;
bu yüzden checkpoint içeriği yeniden yüklenip bağımsız doğrulanmadı.

114 fold/seed ortalaması: validation BCE 0.594507; AP 0.420872;
AP/sınıf oranı 1.355×; precision 0.410499; recall 0.096839;
F1 0.142418; accuracy 0.687868; Rank IC 0.018313.
Olumlu tahmin oranı %6.29, gerçek olumlu oranı %31.05. Hep-olumsuz
referans accuracy'si %68.95; model %68.79. Sonuçlar validation'dır, test değil.
115.938 tahmin kaydı 27.657 benzersiz validation saatini kapsar; seedler
ve örtüşen foldlar bağımsız gözlem gibi sayılmadı.

Ham audit: 2024-10-28 20:00 UTC'de tek bir doğrulanmış boş 1H barı korundu.
Silinen/yapay doldurulan satır yok. İlk hazırlık hatasının kaynağı böylece
pakette görüldü; önceki belirsizlik giderildi.

Yerel rapor: `output/advisor_experiments/review_A_20261003/A_DENEYI_INCELEME.md`.
Grafik: aynı dizinde `A_validation_dashboard.png`.
Tekrarlama scripti: `scripts/review_advisor_bundle.py`.

**Sıradaki iş:** B için eksik iki açık pozisyon girdisinin Drive kaynağını
tamamlamak ve hazırlayıcıya eklemek. A'nın dondurulmuş ham girdileri, etiketleri,
6 temel kanalı ve fold takvimi B ile eşleştirilmeli. Yerel farklı ham snapshot
karşılaştırmaya kaynak yapılmamalı. B hazırlayıcısı henüz bu kaynağı okumuyor;
sadece `EXPERIMENTS=['B']` yaparak eğitime geçme. Eşik/loss/mimariyi A sonucunu
iyileştirmek için değiştirme; sırayı koru. Nihai test hâlâ ayrılmadı/değerlendirilmedi.

## Colab geçişinin tarihsel kaydı

Kullanıcı eğitimin Colab'a taşınmasını istedi. Yerel PID 14372 doğrulanıp
durduruldu; 42/114 tamamlanmış fold/seed ve son checkpointler korundu.
Yerel eğitim artık çalışmıyor. Eski durum satırları aşağıda tarihsel kayıttır.

Yeni çalışma girişi: `notebooks/06_advisor_ablation_colab.ipynb`.
Notebook ayrı dalın güncel kodunu çeker, commit ve ortamı kaydeder, Drive'a
kalıcı checkpoint/rapor yazar. Colab sürümleri ve kod imzası farklı
olduğundan yerel kısmi eğitimle Colab sonuçları tek çalışma gibi birleştirilmez.
Colab'da A deneyi yeni bir imza ile başlar; sonraki aynı ortamlı Colab
oturumları Drive'daki epoch checkpointinden devam eder.

Kalıcı Colab alanı:
`MyDrive/yeniBot/advisor_experiments/colab_v1/`.
İnceleme ZIP'i: `reports/advisor_latest_review_bundle.zip`.
Sonraki adım: GitHub'daki notebooku Colab'da GPU runtime ile çalıştır;
Drive bağlantısını ver; ilk A çıktısını ZIP olarak bu çalışma alanına getir.
Bu yerel sohbetin Drive'a otomatik dosya erişimi yoktur.

Colab hazırlığı:
- [x] Mevcut 00/01/04 notebooklarının hücre akışı incelendi.
- [x] 14 hücreli yeni Colab notebooku ve kaynak üretici script oluşturuldu.
- [x] Sabit ham kaynak hashleri, yerel SSD veri kopyası ve Drive checkpointleri.
- [x] Ortam/cihaz sürümleri eğitim imzasına eklendi.
- [x] Drive yerine runtime diskinde OS kilidi; iki ayrı runtime eşzamanlı kullanılmaz.
- [x] Ağırlıksız inceleme ZIP'i ve ortam/commit kayıtları.
- [x] 17 protokol/Colab/eğitim testi; ayrıca yeni kilit testini içeren
  3 Colab yardımcı testi geçti.
- [x] Notebook şeması nbformat ile doğrulandı; bütün kod hücreleri derleniyor,
  notebookta kayıtlı çıktı/kişisel çalışma sonucu bulunmuyor.
- [x] Notebookun gerçek kullanıcı Colab/Drive oturumunda çalıştırılması.

2026-10-03 kullanıcı Colab hazırlık hücresinde `Non-positive price/trade
count` hatası bildirdi. Önceki kontrol fiyat ve işlem sayısını aynı hata
altında reddediyordu; Drive'daki sorunlu satırın değeri/tarihi henüz görülmedi.
Yeni `validate_advisor_klines` yalnızca doğrulanmış boş barları açık politika
ile korur, satırları düşürmez; gerçek bozuk veride tarih/değer örnekleri verir.
Ham kalite audit'i ve hazırlık kaynak hashleri çıktı manifestine kaydedilir.
Kullanıcı en yeni kodu çekmek için Colab oturumunu yeniden başlatıp
notebooku baştan çalıştırmalı; mevcut sabit ham snapshot silinmez.

## Yerel A deneyi — durdurulmuş tarihsel kayıt

A deneyi 2026-10-03 13:54:56 yerel saatte ayrı Python işlemiyle başlatıldı.
Başlatıldığı sıradaki PID: 14372 (yeniden başlatmada değişir).
Çalışma dizini: `output/advisor_experiments/runs/A_bb0042fa3dde/`.
Kapsam: 38 fold × seed 42/43/44 = 114 fold eğitimi.
Üst sınır 100 epoch; validation BCE / patience 15.
Başlatma kaydı: `output/advisor_experiments/logs/A-20261003_135456.launch.json`.
Stdout: `output/advisor_experiments/logs/A-20261003_135456.stdout.log`.
Stderr: `output/advisor_experiments/logs/A-20261003_135456.stderr.log`.

Bu belgeden **daha güncel canlı durum**, çalışma dizinindeki `STATUS.md`
ve `status.json` dosyalarında bulunur. Her foldun `STATUS.md` ve
`progress.json` dosyası son kaydedilen epoch'u gösterir. İlk sağlık
kontrolünde seed 42 için fold 0–4 tamamlandı, fold 5 çalışıyordu;
bu anlık sayı güncel tamamlanma durumu olarak kullanılmamalı.

Yerel kısmi sonuçlar sadece korunur; kullanıcı Colab tercih ettiği için
yerel GPU eğitimini yeniden başlatma. B–D öncesinde açık pozisyon kaynağını
ve aynı tarih/etiket/grid eşleşmesini tamamla. Hiçbir aşamada validation
sonuçlarını nihai test sonucu diye sunma.

Eski yerel çalıştırma komutu (Colab geçişinden sonra otomatik kullanma):
Komut: `powershell -NoProfile -File scripts/run_advisor_experiment.ps1 -Experiment A`.
Devam ayrıntıları: `docs/advisor-experiment-runbook.md`.
Bu belge eğitim tamamlandı veya akademik başarı sağlandı anlamına gelmez.
