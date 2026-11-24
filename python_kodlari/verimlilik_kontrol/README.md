# DEA girdi kontrolleri

```bash
python python_kodlari/dea_girdi_kontrol.py run --config python_kodlari/ayarlar/dea_girdi_kontrol.json
python python_kodlari/dea_girdi_kontrol.py verify --run KONTROL_KOSU_KLASORU
python python_kodlari/dashboard_veri_hazirla.py
```

Müşteri tabanına göre üretim amacında mevcut müşteri girdisi ve buna personel
kapasitesinin eklendiği senaryo karşılaştırılır. Üç çıktı ve kayıtlı ana küme
atamaları aynıdır. Ana DEA modeli veya `outputs/dea_verimlilik/latest.json`
pointer'ı değiştirilmez. Son kontrol `outputs/dea_girdi_kontrol/latest.json`
içindedir. Var olan koşu ezilmez; `--output`, `--dea-run` kullanılabilir.

Kontroller:

- Bütün kaynak aylarda aday girdi/çıktı kalite özeti, anahtar ve personel değişimi.
- Banka geneli ve küme içinde Pearson/Spearman; korelasyon bir eleme kanıtı değildir.
- Kayıtlı aylarda sekiz tam DEA koşusu; her koşuda bütün şubelerin kapsanması zorunludur.
- Aynı şubelerde ek girdinin mekanik skor etkisi ve optimal dual katkıları.
- Son ayda 20 tekrarlı ±%1 yerel girdi/çıktı stresi, ortak alanlarda eşit rassal değişim.
- Her emsali tek tek çıkarıp kalan şubeleri yeniden çözme; hedef şube emsal setinde kalır.
- Aylık skor değişimlerini kümede kalan/ölçülen profili az değişen alt örneklemlerde ayırma.

Tüm ana senaryo koşuları mevcut bağımsız DEA dual/slack doğrulamasından geçer.
Stres özetlerinin tamamı kayıtlardan yeniden hesaplanır. Her küme/senaryonun ilk
rassal tekrarı ve en etkili emsal çıkarıldığında kalan şubelerin skorları ayrı
dual LP ile kontrol edilir. Bütün stres tekrarlarının bağımsız yeniden çözümü
iddia edilmez. ZIP `code/dea_girdi_kontrol.py verify --run PAKET_KLASORU` ile
proje dışında da doğrulanabilir.

Kapsama ve inceleme eşikleri pilot politikasıdır. Yüksek DEA ortalaması seçim
amacı değildir. Müşteri girdisini koruma önerisi bütün olası girdiler arasında
istatistiksel optimum iddiası taşımaz. Kaynak veriler sentetiktir ve personel
kapasitesi aylarda sabittir; gerçek personel hareketleri bu veride sınanamaz.
Emsal etkisi yüksekse çıktıdaki `peer_removal.csv` dosyası incelenir; etkili emsal
otomatik silinmez. Deneyler yeni test, güven aralığı veya gelecek ay tahmini değildir.

Tanımlı ek girdiler kümeleme seçiminden önce `dea_coverage` sözleşmesine
alınır. Senaryoların biri küçük küme nedeniyle şube kaybedecekse seçim adayı
elenir. Bu nedenle önce otomatik kümeleme seçimi, sonra ana DEA, sonra girdi
kontrolü çalıştırılır. Girdi kontrolü küme yapısını kendiliğinden değiştirmez.
