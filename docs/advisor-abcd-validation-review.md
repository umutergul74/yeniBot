# A–D tamamlanmış doğrulama karşılaştırması

İncelenen paket: `advisor_latest_review_bundle (3).zip`.
SHA256: `a96dde6cfbe5defd7078814060a187675ac237cdb1df6c629df97ca0f4db0aef`.
Her kapsam 38 fold × seed 42/43/44 = 114 eğitim. Toplam 456 fold/seed eğitimi
vardır; bunlar bağımsız istatistiksel tekrarlar değildir. Tüm sonuçlar validation.

| Ölçüt, eşit ağırlıklı fold/seed ortalaması | A: 6 / GRU | B: 34 / GRU | C: 34 / TCN | D: 34 / TCN–GRU |
|---|---:|---:|---:|---:|
| BCE, düşük daha iyi | 0,594507 | 0,612412 | 0,607080 | 0,605371 |
| Average Precision | 0,420872 | 0,397423 | 0,397981 | 0,408339 |
| Precision, eşik 0,5 | %41,05 | %44,78 | %37,99 | %37,72 |
| Recall, eşik 0,5 | %9,68 | %11,49 | %8,69 | %9,64 |
| F1 | 0,142418 | 0,159500 | 0,121095 | 0,132345 |
| Accuracy | %68,79 | %67,77 | %68,58 | %68,50 |
| Rank IC | 0,018313 | 0,001562 | 0,007430 | 0,013251 |

Önceden belirlenmiş validation BCE ölçütünde sıra A, D, C, B. Aynı 34 girdili
modeller arasında D, B ve C'ye göre BCE/AP/Rank IC ortalamalarında ilerleme
gösteriyor; temel altı girdili A'yı aşmıyor. F1'de B önde. Sonuçlara bakarak
seçim ölçütü veya eşik değiştirilmemeli. A, sonraki karşılaştırmalar için bu
ölçütle seçim adayıdır; bağımsız test başarısı henüz bilinmiyor.

D'nin BCE'si 38 foldun seed ortalamasında B'ye göre 29, C'ye göre 25 ve A'ya
göre 10 foldda daha iyi. AP'si C'ye göre 30 foldda daha iyi. Bunlar betimsel
sayılardır; örtüşen foldlarla istatistiksel anlamlılık sonucu çıkarılmadı.
Her örneğe olumsuz diyen referansın accuracy'si %68,95; tüm modeller bunun
altında. Bu referans BCE için uygun referans değildir; train sınıf oranından
sabit olasılık referansı ayrıca değerlendirilmeden mutlak BCE becerisi iddiası yok.

## Denetim

- 2.344 envanter dosyasının boyut ve SHA256 değerleri eşleşti.
- A/B/C'nin önceki ZIP'teki 574'er çalışma dosyası byte düzeyinde aynı.
- D tamamlandı: 114 eğitim, 1.904 epoch; her eğitimde 4.977 dizi, validation'da 1.017 dizi.
- AP/precision/recall/F1/accuracy/Rank IC/BCE, tahmin dosyalarından yeniden hesaplandı ve eşleşti.
- En iyi epoch minimum validation BCE ile uyumlu; patience 15. 228 kayıtlı sınır denetimi geçti.
- A–D'nin 114 tahmin grubunda tarih, etiket, ileri getiri ve satır konumu aynı.
- B/C/D'nin özellik listesi ve veri hash'i aynı; split, seed, genel eğitim ayarları ve ortam eşleşiyor.
- OI audit'i aynı: 419.628 kaynak kaydı, 475 kullanılamayan ölçüm, %99,7705 saatlik kapsam;
  77 saatlik satırda iki OI özelliği nötr dolduruldu, piyasa satırı silinmedi.
- Test değerlendirmesi tüm kapsamlarda sıfır.

Paket kaynak commit'i `d55573c`; sonraki commit'ler inceleme araçları ve
raporları değiştirdi, bu tamamlanmış deneyin kodunu yeniden eğitmedi.
Ham kaynaklar ve model ağırlıkları ZIP'te bulunmadığından bu içerikler bağımsız
yeniden üretilemedi; hazırlık ve veri eşleşme audit'leri paketteki kayıtlardır.

## Yorum sınırları ve eğitim davranışı

D'de 246.145 parametre var; A/B/C'de sırasıyla 168.193 / 178.945 / 67.457.
Parametre bütçesi eşitlenmedi. Sonuç, sabitlenen bu mimari ayarları için geçerli;
genel bir GRU veya TCN üstünlüğü sonucu değil. A'nın `realized_vol_14`, `gk_vol_14`,
`atr_14_pct` kanalları B/C/D'de yok; A ile karşılaştırma yalnızca ek özelliklerin
etkisini izole etmez. B/C/D'nin özellik seti aynı olduğundan kendi aralarındaki
karşılaştırma farklı mimarileri aynı veri temsiliyle değerlendirir.

D'de en iyi epoch ortalaması 1,70 (1–5); 56 eğitimde ilk epoch seçilmiş.
22 eğitimde eşik 0,5 üzerinde hiç olumlu tahmin yok. Rank IC 65/114 eğitimde,
seed-ortalama 21/38 foldda pozitif. Recall düşük; hibrit karmaşıklığı tek başına
performans artışı sağlamıyor. Sonuçlar model seçimi yapılan doğrulama verisinde
ölçüldü; kârlılık veya nihai genelleme kanıtı olarak sunulmamalı.

## Devam için karar

Test dönemi hâlâ tanımlanmadı. Eylül 2026 daha önce performans/araştırma kararlarında
kullanılmadıysa aday olabilir; kullanıcıdan bu bilgi istenmiş, yanıt gelmemiştir.
2026 verileri eski çalışmalarda değerlendirilmiş olabilir; tüm 2026 otomatik
dokunulmamış test değildir. Test sınırları ve kuralları sabitlenmeden yeni
wavelet/loss veya nihai test çalışması başlatılmadı. A–D kayıtları korunur.

Yerel karşılaştırma CSV/JSON/grafikleri: `output/advisor_experiments/review_D_20261003`.
Tekrar inceleme: `review_advisor_bundle.py --experiment A|B|C|D`,
`compare_advisor_architectures.py --experiments A B C D`.
