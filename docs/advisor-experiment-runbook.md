# Danışman deneyleri: protokol ve çalıştırma

## Deney tasarımı

Bu araştırma, hocanın 2026-10-03 tarihinde iletilen önerisini uygular.
Önceki test sonuçları yöntem seçiminde kullanıldığı için geçmiş dönem yeni
çalışmada geliştirme verisidir. Bu komutlar nihai test sonucu üretmez.
İlk A–D karşılaştırması wavelet olmadan yapılır. A'da 6 temel fiyat/hacim
özelliği bulunur; RSI/MACD eklenmez. Bunlar getiri, hacim z-skoru, gerçekleşen
oynaklık, Garman–Klass oynaklığı, ATR yüzdesi ve VWAP uzaklığıdır.

Mevcut modelin 38 girdisinden 4'ü wavelet fiyat/hacim dönüşümleridir.
Wavelet kapalı karşılaştırmada bunların ayrıca ham girdileri zaten bulunduğu
için B–D için 34 benzersiz, açıkça listelenmiş özellik donduruldu.
Bu, bütün mevcut girdilerin wavelet kapalı karşılığıdır; eski 38 boyutlu
modelin aynısı değildir. `configs/advisor_full_features.txt` sütun sırasını
sabitler. İki açık pozisyon sütunu yerel kaynakta eksiktir; tamamlanmadan
B–D çalıştırılamaz. Eksik sütunları atlamak ve 32 girdiyi 'tüm özellikler'
diye sunmak yasaktır. Wavelet karşılaştırmasında seçilen 34 kanallı yapıda
1H/4H getiri ve hacim kanalları filtrelenmiş karşılıklarıyla değiştirilecek;
boyut, etiketler ve ortak değerlendirme tarihleri sabit tutulacak. Bu aşama
henüz uygulanmadı ve öncesinde ayrı protokol kaydı yapılacak.

Her modelde ortak 128 boyutlu karar katmanı vardır. GRU, tek yönlü 2 katman
128 birim; TCN, 64 kanal ve [1,2,4,8,16] genişlemeler; hibrit paraleldir.
Parametre sayıları eşit değildir ve her foldun raporunda açıkça saklanır.
Batch 256, AdamW, öğrenme oranı 0.001, weight decay 0.0001, üst sınır
100 epoch ve 15 epoch erken durma sabittir. Scheduler eklenmez.
Seed 42/43/44; fold içi seed = temel seed + fold numarası.
Birincil seçim ölçütü validation BCE; mimari karşılaştırması aynı fold/seed
kapsamında ortalama validation BCE ile yapılacak. AP ve F1 ikincil,
Rank IC yardımcı ölçüttür. Sonuç görüldükten sonra birincil ölçüt değişmez.
F1, precision ve recall sabit 0.5 eşikte raporlanır. AP, trapez PR-AUC
ile aynı sayı değildir; çıktıda `average_precision` adıyla saklanır.

## Zaman sınırları

5.040 train → 24 boş bar → 1.080 validation → 24 boş bar → 720 test bölümü.
İlerleme 720 bar. Test bölümü burada takvim oluşturmak ve validation
etiketlerinin test sınırına taşmadığını kontrol etmek için tutulur; model
bu bölümde çalıştırılmaz. Training/validation hedefleri 10 saat sonra
kesinleşen sabit ufuk metadatası üzerinden denetlenir; bariyer erken
vurulsa bile ileri getiri tanısı için tam ufuk korunur.
Diziler ayrımdan sonra kurulup ilk 63 gözlem bağlam olarak kullanılır.
RobustScaler yalnızca ilgili train bölümüyle fit edilir.

Geliştirme sınırı: 2022-03-01 04:00–2025-12-28 02:00 UTC.
Ham verinin son etiketleri hesaplamak için 10 saatlik devamı kullanılır;
bu devam girdi veya değerlendirme örneği yapılmaz. Ham kaynak daha yeni
veri içerse de yeni dönem bu deneyde değerlendirilmez. Nihai testin
tarihleri, kullanıcı onayıyla Eylül 2026 olarak ayrı
`configs/advisor_final_test.yaml` dosyasında ayrıldı; mevcut A–D config
imzaları değiştirilmedi. Final eğitim ve test adaptörü henüz hazır değil.
Sonraki fold validation'ları eski testlerle
örtüşebildiği için tarihsel fold testleri nihai bağımsız kanıt değildir.

## Çalıştırma ve devam

Proje kökünde:

```powershell
python -m yenibot.training.advisor prepare --config configs/advisor_ablation.yaml
python -m pytest tests/test_advisor_protocol.py -q
powershell -NoProfile -File scripts/run_advisor_experiment.ps1 -Experiment A
```

Son komut gizli bir arka plan Python işlemi açar. Codex yanıtı/kredisi
bittiğinde işlem ayrı olarak devam eder; bilgisayarın kapanması, uyku,
GPU/işletim sistemi hataları çalışmayı kesebilir. GPU belleği 4 GB olan
yerel bilgisayarda sırasıyla tek deney çalıştırılır. B/C/D ilk A ve veri
eksikliği çözülmeden otomatik başlatılmaz.

Çıktı: `output/advisor_experiments/runs/A_<imza>/`.
- `protocol.json`: veri, kod, ayar, fold ve seed imzaları.
- `status.json`: mevcut fold/seed ve tamamlananlar.
- `seed_42/fold_000/progress.json`: son kaydedilen epoch ve en iyi epoch.
- `last.pt`: model, optimizer, en iyi model, scaler ve RNG durumları.
- `history.csv`: epoch başına train/validation ölçümleri.
- `validation_metrics.json` ve `validation_predictions.parquet`: yalnızca validation.
- `validation_summary.csv`: biten fold/seed sonuçlarının özeti.
- `output/advisor_experiments/logs/`: stdout, hata ve PID kayıtları.

Aynı komutu tekrar çalıştırmak, **aynı veri/kod/ayarlarla** son kaydedilen
epoch'tan devam eder. Epoch ortasında kesilirse yalnızca o epoch tekrar
yapılır. Biten foldlar tekrar eğitilmez. İşlem hâlâ çalışırken ikinci
başlatma OS kilidiyle reddedilir. Veri/ayar/kod değişimi farklı imzalı yeni
bir çalışma oluşturur; eski ağırlıklarla yeni protokol karıştırılmaz.
Kaydedilen PyTorch sürümü ve cihaz da eşleşmelidir. CUDA üzerinde birebir
yeniden üretilebilirlik CPU'da başarılı olan kesinti testinden ayrı olarak
gerçek veri teknik kontrolünde doğrulanır.

Teknik kontrol farklı çıktı dizinine ve 2 epoch bütçeye sahiptir; akademik
deney sonucu veya A deneyinin tamamlanması olarak sunulmaz.

## Kredi veya oturum kesildiğinde

1. `docs/advisor-experiments-status.md` dosyasını oku.
2. En yeni `*.launch.json`, `status.json`, `progress.json` ve stderr'i oku.
3. PID'nin hâlâ çalışıp çalışmadığını kontrol et. Çalışıyorsa tekrar başlatma.
4. Çalışmıyorsa hatanın nedenini çöz; aynı resume komutunu kullan.
5. Veri/kod/ayar değişmişse eski imza ile devam ettiğini iddia etme.
6. Sıradaki aşamayı ve sonuçları bu durum belgesine yaz.
