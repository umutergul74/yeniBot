# Eylül 2026 bağımsız test rezervasyonu

Güncelleme 2026-10-04: Aşağıdaki metin rezervasyon anının tarihsel kaydıdır.
Yöntem seçimi tamamlandı: 6 özellik/GRU/wavelet kapalı/saf BCE. Ayrı
`configs/advisor_final_execution.yaml` ve 10 Colab notebooku final eğitim,
artifact freeze ve test akışını somutlaştırır. Güncel çalıştırma ayrıntıları
`docs/advisor-final-colab-runbook.md`; gerçek final eğitim/test henüz yapılmadı.

2026-10-03 tarihinde kullanıcı Eylül 2026 ile daha önce eğitim, backtest
veya performans değerlendirmesi yapmadığını doğruladı. Bu dönem model,
özellik ve yöntem seçimlerinden ayrı tutulmak üzere ayrıldı. Bu, dönem
tamamlandıktan sonra performansına bakılmadan ayrılan geriye dönük
dokunulmamış holdout'tur; ileriye dönük ön kayıt iddiası değildir.

Kaynak protokol: `configs/advisor_final_test.yaml`. Mevcut A–D ayarları ve
çalışma imzaları değiştirilmez. Yeni dosya rezervasyon kurallarını taşır;
henüz eğitilmiş bir model veya hazır nihai değerlendirme komutu değildir.

| Kural | Ayrılan değer |
|---|---|
| Varlık | Binance BTCUSDT USDT-M sürekli vadeli |
| Zaman çözünürlüğü | 1 saat; UTC bar zaman damgaları |
| İlk test örneği | 01.09.2026 00:00 UTC |
| Son test örneği | 30.09.2026 23:00 UTC |
| Planlanan örnek sayısı | 720; kaynak/özellik kalite kontrolü gerektirir |
| Dizi uzunluğu | 64; önceki 63 gözlem test öncesinden alınabilir |
| Hedef | 10 saat, ATR TP 2 / SL 5, mevcut etiket tanımı |
| Test öncesi boşluk | 24 saat; 31 Ağustos |
| En son eğitim/validation örneği | 30.08.2026 23:00 UTC |
| Son etikette gereken son 1H kaynak barı | 01.10.2026 09:00 UTC; 10:00'da tamamlanır |
| Sınıflandırma eşiği | 0,5; teste göre ayarlanmaz |
| Seedler | 42, 43, 44; teste göre seçilmez |
| Testte fit/ağırlık güncelleme | Yok |

24 saat boşluk eğitim/validation örneklerinden ayrılır. Tahmin sırasında
o güne kadar bilinen fiyat/hacim verilerinin geçmiş bağlamda kullanılması
mümkündür. Bu nedenle ilk test tahmini için 29 Ağustos 09:00–1 Eylül 00:00
dizisi kullanılabilir. Özelliklerin uzun rolling/wavelet pencereleri için
daha uzun tarihsel kaynak gerekir; bu 63 saat özellik warmup uzunluğu değildir.
Geçmiş bağlam sağlanırsa 720 yerine 657 örneğe düşürülmez. Gelecekteki
etiket kaynak barları girdiye konulmaz ve ek test örneği sayılmaz.

## Şu an ne yapılmış durumda?

Test tarihleri ve genel sınır/raporlama kuralları ayrıldı. Test verisi okunmadı,
performansı hesaplanmadı; yeni eğitim başlatılmadı. Model, wavelet/loss,
final eğitim ve pre-test validation pencereleri ile ağırlıklar henüz
kilitlenmedi. Rezervasyon denetleyicisi bu nedenle hazır değerlendirme
durumu bildirmez. Salt denetim:

```text
python -m yenibot.training.advisor_holdout --config configs/advisor_final_test.yaml
```

`assert_fitting_scope` gelecekteki final eğitim adaptöründe kullanılmak
üzere sample/etiket sınırlarını reddeden bir yardımcıdır. Henüz final
eğitim notebook'una bağlanmadı. Mevcut 07 notebooku zaten yalnızca
2022–2025 geliştirme bölümünü kullanır; rezervasyon yeni bir test modu eklemez.

## Bundan sonraki sıra

1. Geliştirme/validation sonuçlarıyla wavelet ve loss aşamalarının protokolünü
   netleştir. A–D'nin BCE ölçütünde A seçim adayı; test bu kararı etkilemez.
2. Final eğitim penceresini, test öncesi validation penceresini, aynı koşullarda
   raporlanacak karşılaştırma modellerini ve nihai epoch seçimini kaydet.
   Ağustos sonuna kadar kaynak kapsamının genişletilmesi gerekebilir;
   bu işlem Eylül test metriklerini açmaz.
3. Seçilen modelleri testten önceki veride eğit; scaler yalnızca ilgili train
   verisine fit edilir. Model/ağırlık/özellik/ayar/source hash manifestini kilitle.
4. Test girdilerini kalite kontrolü ve geçmişe dayalı özellik üretimiyle hazırla;
   donmuş modellerle tahmin et. Testte tekrar fit/epoch/threshold seçimi yapılmaz.
5. BCE, AP, precision, recall, F1, accuracy, Rank IC ve sınıf oranını raporla.
   Her seed ayrı; sonra aritmetik ortalama/std. Seed ortalaması yeni bir
   olasılık ensemble'ı değildir. Std istatistiksel anlamlılık kanıtı sayılmaz.
   Hep-olumsuz sınıflandırma ve train sınıf oranından sabit olasılık referanslarını ekle.

Kötü test sonucu yeni yöntemin aynı testte seçilmesiyle düzeltilmez. Bu dönem
yöntem değişikliklerinde kullanılmaya başlanırsa bağımsız doğrulama rolü
sona erer; sonraki yöntem için yeni dokunulmamış dönem gerekir.
Bir aylık tek varlık testi farklı piyasa rejimlerine genellenebilirlik kanıtı değildir.
