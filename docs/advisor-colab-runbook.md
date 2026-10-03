# Danışman deneylerini Colab'da çalıştırma

Notebook: `notebooks/06_advisor_ablation_colab.ipynb`.
Mevcut 00/01/04 notebookları incelenerek aynı GPU, Drive, Git ve çıktı
düzeni kullanıldı. Eski notebookların deney/test değerlendirmesi çağrılmaz.
Colab notebooku için ham veri/etiket ve sonuç alanı ayrıdır.

## Kullanım

1. Notebooku Colab'da aç; GPU runtime seç.
2. İlk denemede `EXPERIMENTS = ['A']` ayarını koru.
3. Hücreleri sırayla çalıştır; Drive bağlama iznini ver.
4. Ekrandaki dal/commit, 33.551 satır, 38 fold ve 76 sınır kontrolünü gör.
5. Eğitim çıktıları Drive'a kaydedilir. Sonuç ZIP'ini buraya getir.

Notebook dalın güncel commit'ini çeker; her oturumun tam commit'i ve paket
sürümleri kaydedilir. Kod değişikliği geldiğinde eski importlarla devam
edilmez. `REPO_COMMIT` belirli bir sürümü yeniden çalıştırmak için kullanılabilir.
Depo herkese açık olduğundan clone için token gerekmez.

## Veri ve hız

Drive'daki mevcut 1H/4H ham kaynaklar tercih edilir. Eksikse yalnızca gerekli
sabit aralık indirilir; önceki Drive veri dosyaları değiştirilmez.
Kaynak 2022-01-01 00:00–2025-12-28 12:00 UTC olarak sabitlenir; son 10 saat
hedefin olgunlaşması içindir. Geliştirme gözlemleri önceki deneyle aynı
2022-03-01 04:00–2025-12-28 02:00 UTC aralığıdır.
Eksik saat veya eksik tarih kapsamı varsa eğitim başlamaz.

### Sıfır işlemli kaynak barları

`data.zero_activity_policy=preserve_verified_empty` açık politika olarak
belirlendi. İşlem/hacim/taker/quote alanlarının tümü sıfır, OHLC tamamen
aynı pozitif fiyat ve önceki kapanışla aynı olduğunda kaynak barı korunur.
Satır silinmez, yapay fiyat/işlem üretilmez; 10 bar = 10 saat eşitliği korunur.
İlk barın önceki kapanışı olmadığı için sıfır işlemli ilk bar doğrulanamaz
ve reddedilir. Negatif değerler, NaN/inf, sıfır fiyat, bozuk OHLC aralığı,
taker tutarsızlığı ve hacim var/işlem yok gibi durumlar yine hata verir.
Hata tarih ve sütun değerlerini gösterir. Uygulanan politika ve korunan
bar tarihleri hazırlanmış veri manifestinde `raw_quality_audits` altında
ve Drive inceleme ZIP'inde bulunur. Eski global `zero_volume_policy=drop`
değiştirilmedi; bu saatlik deney ayrı sözleşmeyi kullanır.
Mevcut özellik hesaplayıcısının sıfır bölme için 0 dönüşü değişmedi;
sıfır işlemde bu değer gerçek bir işlem başına ortalama olarak yorumlanmaz.
Kodun reddettiği gerçek bozuk satırları bu politika onarmış saymayız.

Ham veri Colab'ın `/content` diskine kopyalanır; hazırlanan tablo buradan
okunur. Büyük veri okumaları Drive'a yüklenmez. Küçük checkpointler her
epoch sonunda Drive'a atomik dosya değişimiyle yazılır. Bu kalıcılık
karşılığında bir miktar Drive yazma maliyeti vardır; performans ölçülmeden
checkpoint sıklığı düşürülmez. Drive erişim hataları saklanmaz.

A deneyi 6 fiyat/hacim özelliği + GRU + BCE'dir. Bu notebookun ilk sürümü
B/C/D için eksik açık pozisyon girdilerini üretmez; onları eksik sütunlarla
çalıştırmak yerine durur. Wavelet/loss aşaması da daha sonra ayrı adımda.

## Kesinti ve ortam değişikliği

`MyDrive/yeniBot/advisor_experiments/colab_v1/runs/<imza>/` altında:
- `STATUS.md` ve `status.json`: fold/seed kapsamı ve tamamlananlar.
- Fold klasörlerinde `STATUS.md`, `progress.json`, `last.pt`: son epoch.
- `validation_summary.csv`: tamamlanan validation sonuçları.

Aynı notebooku aynı ayarlarla yeniden çalıştırmak aynı imzada devam eder.
Epoch ortasında kesilirse o epoch yeniden yapılır. Ortam/cihaz/veri/kod
değişirse yeni imzalı çalışma açılır; farklı sürümlü eğitimler birleştirilmez.
Bu nedenle yerel Windows çalışmasının 42 biten fold/seed'i otomatik
Colab'a taşınmaz. Yerel checkpointler korunur ve yerel süreç durdurulmuştur.

Dosya kilidi Colab'ın yerel diskinde tutulur: Drive dosya sistemi POSIX
kilitlerini desteklemeyebilir. Aynı runtime içinde ikinci başlatma
engellenir; iki ayrı Colab runtime'ı aynı Drive sonuçlarına yazdırmayın.

`AUTO_UNASSIGN=False` varsayılandır. True yapılırsa eğitim ve sonuç ZIP'i
başarıyla tamamlandıktan sonra GPU oturumu bırakılır. Hata hâlinde otomatik
kapatılmaz; hata günlüğü ve checkpoint durumu incelenebilir.

## Çıktıyı incelemeye getirme

Drive yolu:
`MyDrive/yeniBot/advisor_experiments/colab_v1/reports/advisor_latest_review_bundle.zip`.
ZIP; kullanılan kod/ayar/ortam, kaynak hashleri, fold takvimi, epoch geçmişi,
validation tahminleri ve durum dosyalarını içerir. Model ağırlıkları ZIP'e
girmez ve Drive'da kalır. Başarısız eğitim hücresi de mümkünse kısmi ZIP
üretir; runtime aniden tamamen silinirse önceden yazılan checkpointler
kalır, sonraki oturum bunlardan devam eder ve ZIP'i yeniler.

ZIP'i sohbete ekleyin veya bilgisayara senkronize edilen dosyanın yolunu
paylaşın. Bu sohbetin Google Drive'a otomatik erişimi bulunmuyor.
Notebook hazırlandı ve yerel kontroller geçti; kullanıcı Drive oturumunda
henüz çalıştırılmadı. Hazır notebook ile tamamlanmış Colab deneyi ayrıdır.
