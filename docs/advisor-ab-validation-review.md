# A ve B: doğrulama sonuçları

2026-10-03 kullanıcı ZIP'i incelendi. A: temel 6 özellik + GRU + BCE;
B: dondurulmuş 34 wavelet dışı özellik + aynı GRU + aynı BCE.
Her biri 38 fold × 3 seed, 114 tamamlanmış eğitim. Test değerlendirmesi yok.

| Ölçüt (eşit ağırlıklı fold/seed ortalaması) | A | B |
|---|---:|---:|
| BCE, düşük daha iyi | 0,594507 | 0,612412 |
| Average Precision | 0,420872 | 0,397423 |
| Precision, eşik 0,5 | %41,05 | %44,78 |
| Recall, eşik 0,5 | %9,68 | %11,49 |
| F1, eşik 0,5 | 0,142418 | 0,159500 |
| Accuracy | %68,79 | %67,77 |
| Rank IC | 0,018313 | 0,001562 |
| Olumlu tahmin oranı | %6,29 | %8,33 |

B daha sık olumlu tahmin ediyor; precision/recall/F1 artışı var. BCE ve AP
kötüleşiyor; ileri getiri sıralamasıyla ilişki çok küçük. Önceden belirlenen
validation BCE ölçütü bu iki yapı arasında A lehine. C ve D tamamlanmadan
nihai mimari seçilmeyecek. Eşik veya seçim ölçütü bu sonuçlarla değiştirilmeyecek.
Her örneğe olumsuz diyen referansın accuracy'si %68,95; iki model de bunun altında.
Bu referans BCE/AP için kullanılmaz; train sınıf oranından sabit olasılık
referansı ayrıca raporlanmadan BCE'ye dayalı mutlak beceri iddiası kurulmaz.

## Veri ve yöntem kontrolü

- ZIP envanterindeki 1.178 dosyanın boyut ve hash'i doğrulandı.
- A'nın önceki ZIP'teki 574 çalışma dosyası byte düzeyinde aynı.
- 114 çift tahmin dosyasında tarihler, etiketler, getiriler ve satır konumları eşleşti.
- Eğitim, model ayarları, foldlar, seedler ve GPU/kütüphane ortamı eşleşiyor.
- A'nın temel veri hash'i B hazırlık referansıyla aynı. Tam veri 33.551 satır.
- Her eğitimde 4.977, validation'da 1.017 dizi. Her kapsamda 228 kayıtlı sınır denetimi geçti.
- Epoch seçimi minimum validation BCE ile uyumlu. B toplam 1.865 epoch eğitildi.
- Kaydedilen metrikler tahminlerden yeniden hesaplanıp doğrulandı.
- OI: 419.628 kayıt, 475 kullanılamayan kaynak ölçümü. Saatlik geliştirme satırlarında
  %99,7705 geçerli özellik kapsamı; 77 satırda iki OI kanalı nötr doldurulmuş.
  Kullanılamayan ölçümlerden log-değişim hesaplanmıyor; piyasa satırları silinmiyor.

B, A'nın doğrudan üst kümesi değildir: A'daki `realized_vol_14`, `gk_vol_14`
ve `atr_14_pct`, B listesinde bulunmuyor. Sonuç, farklı iki özellik setinin
karşılaştırmasıdır. Eklenen özelliklerin veya yalnızca OI'nin ayrı etkisini
göstermez. C ve D karşılaştırmasında B'nin 34 kanalı aynen korunacaktır.

Ham kaynak ve ağırlıklar ZIP'te bulunmadığından bu içerikleri bağımsız yeniden
üretmedik; hazırlık audit'i ve hash referansı paketin kaydıdır. Validation
foldları ve girdileri örtüşür; 114 bağımsız istatistiksel tekrar varsayılmadı.
Bu değerlendirme nihai test sonucu veya istatistiksel anlamlılık kanıtı değildir.

## Devam kaydı

Sonraki aşama C: aynı 34 özellik → TCN → BCE. Aynı split, seed, eşik ve
validation BCE seçim kuralı korunacak. Ardından D: paralel TCN–GRU.
Wavelet ve loss karşılaştırması A–D doğrulama değerlendirmesinden sonra gelir.
Nihai bağımsız test dönemi henüz tanımlanmadı; yöntem seçiminden önce sabitlenmelidir.

Yeniden inceleme araçları: `scripts/review_advisor_bundle.py` (A veya B kapsamı),
`scripts/compare_advisor_ab.py` (etiket eşleşmeleri ve karşılaştırma).
Yerel ayrıntılı rapor, JSON, CSV ve grafikler:
`output/advisor_experiments/review_B_20261003`.
