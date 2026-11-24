# Dinamik kümeleme değişken seçimi

CSV kolonları, hedefler, anahtarlar ve alan sayısı kodda sabit değildir. CatBoost
algoritması bütün göstergelerde ortaktır. Önceki tahmin pipeline'ı tarihsel olarak
korunur; bu paket stok ekleyen tahmin modeli yerine yapısal adaylarla önem analizi yapar.

## Kurulum

Proje kökünde Python 3.12 ile:

```bash
python3 -m venv .venv_flaml
.venv_flaml/bin/python -m pip install -r python_kodlari/requirements_flaml.txt
.venv_flaml/bin/python python_kodlari/dinamik_feature_selection.py --help
```

Mevcut FLAML ortamı yeterlidir. Bu modül FLAML araması başlatmaz.

## Sözlük

V4 örneği `python_kodlari/ayarlar/dinamik_v4.json` içindedir. JSON kullanılır;
YAML okuyucu bağımlılığı gerekmez. `input.path` sözlük dosyasının konumuna göre
çözülür. CSV ayırıcı/encoding ve iş birimi filtresi input altında tanımlanabilir.

Örnek alanlar:

```json
[
  {"name":"gelir_profili", "source":"AYLIK_GELIR", "label":"Ortalama aylık gelir",
   "role":"candidate", "dtype":"numeric", "known_by_period_end":true,
   "family":"gelir", "priority":10, "aggregation":"mean"},
  {"name":"meslek", "source":"MESLEK", "role":"candidate", "dtype":"categorical",
   "known_by_period_end":true, "aggregation":"category_shares", "levels":["UCRETLI","EMEKLI"]},
  {"name":"vadesiz", "source":"TL_VADESIZ_AYLIK_ORT", "role":"target",
   "dtype":"numeric", "aggregation":"sum", "required":true}
]
```

`source` fiziksel kolon, `name` modeldeki mantıksal addır. Aynı hedef kaynağı
aday olarak tekrar eşlenemez. Sayısal kodların anlamı otomatik çıkarılmaz.
İş tanımı, bilinirlik ve production_proxy işaretleri verinin sahibi tarafından
doğru tanımlanmalıdır. Örneğin başka isimli bir bakiye kopyasını sistemin yalnızca
veri tipine bakarak sızıntı olarak tanıması beklenmez.

Roller: candidate, target, context, excluded, review. Kimlik/dönem/grup alanları
input.keys altında tutulur. Aday/çıktı alanlarının anahtar kaynağı kullanması reddedilir.
`availability_lag_months` sıfır veya pozitif olabilir; tam geçmiş takvim ayına bağlanır.
`required` isteğe bağlı adaylarda false, hedeflerde varsayılan true'dur. Olmayan
isteğe bağlı aday missing_source kararıyla raporlanır. Uygulamada seçilmiş kaynaklar zorunludur.

`priority` küçük olan alan korelasyon temsilcisi olarak önce değerlendirilir.
Eşitlikte mantıksal ad sırası kullanılır. `must_keep:true` açık iş gerekçesiyle
koruma sağlar; eksiklik/sabitlik/korelasyon elemesini geçersiz kılamaz.

## Veri düzeyi ve toplulaştırma

- branch_month: şube–ay tekildir; adaylar zaten şube düzeyindedir.
- customer_month: müşteri–ay tekildir; şube–aya sözlükteki kurallarla toplanır.
- customer_branch_month: müşteri–şube–ay tekildir; devir katkıları önceden doğru
  hazırlanmış aylık parçalardır. Aynı müşteri/şubeye birden fazla giriş parçaları önce birleştirilmelidir.

Müşteri düzeyinde her aday ve çıktı için aggregation zorunludur:
sum, mean, weighted_mean (+ weight kaynak kolonu), mode, first_exact,
count_distinct veya category_shares (+ sabit levels). Kategori payları OTHER/MISSING
paylarıyla genişletilir. Branch_month için genişletme yapılmaz; kategoriler native kalır.
Oranlar 0/1 müşteri işaretlerinin mean'iyle üretilebilir. first_exact aynı şube–ayda
farklı değer varsa hata verir. Ağırlıklar sayısal, eksiksiz ve negatif olmayan değerlerdir.

Bu araç günlük mevduat muhasebesi/devir simülasyonu yapmaz. Aylık ortalamalar ve
müşteri–şube katkıları kaynağında doğru gün ağırlığıyla hazırlanmalıdır.
Bir müşteri başına aynı toplam tutarı her parçaya kopyalamak hatalı toplama yol açar.

## Komutlar

```bash
python python_kodlari/dinamik_feature_selection.py discover --input banka.csv --output sozluk.json --branch SUBE --period AY --group GRUP
python python_kodlari/dinamik_feature_selection.py run --config sozluk.json
python python_kodlari/dinamik_feature_selection.py check --run SECIM_KLASORU
python python_kodlari/dinamik_feature_selection.py report --run SECIM_KLASORU
python python_kodlari/dinamik_feature_selection.py apply --schema SECIM_KLASORU/selected_schema.json --input yeni_ay.csv --output outputs/uygulama
```

discover bütün açıklayıcı alanları başlangıçta review yapar; hedef ve adayları
eşlemeden run çalışmaz. Müşteri düzeyinde discover için --grain customer_month
ve --customer MUSTERI eklenir. `run --as-of YYYY-MM` yalnızca bu tarihe kadar
veriyi işler. --output farklı deney klasörü; --baseline önceki şemayla karşılaştırma içindir.
Mevcut çıktı klasörü sessizce ezilmez. Sözlük değiştiğinde yeni aday sürüm üretilir.

## Çıktılar ve aylık çalışma

- decisions.json/csv: seçilen/elenen/tanımsız alanlar ve gerekçeleri.
- selected_schema.json: ortak adaylar, tipler, sıralama, dondurulmuş medyan/kategoriler.
- cluster_input_latest.csv: son ay şube başına tek satır, yalnızca seçilmiş alanlar.
- cluster_input_panel.csv: aylık izleme paneli; tek dönem şubeleri gibi kümelenmez.
- model_metrics.csv / importance_repeats.csv: dönem kontrolleri ve tüm önem tekrarları.
- schema_profile.json/csv: tüm kaynak kolonları ve eşleme durumu.
- drift.json / schema_comparison.json: dağılım ve önceki şemayla alan farkları.
- SONUCLAR.md / METODOLOJI.md / report.json: sonuç, protokol ve dashboard girdisi.
- models/: önem kontrolü için hedef/dönem özelinde CatBoost kayıtları.

run en son aylardan yapılandırılmış holdout_months kadarını seçim dışında tutar.
Genişleyen dönemler, eşikler ve CatBoost ayarları selection altındadır. Kurallar
evrensel doğrular değildir; yeni veri için tarihsel dönemlerde değerlendirilmelidir.
Alan sayısı sabitlenmez. V4 örneğinde 23 adaydan 22'si seçildi; tüm alanlar
pozitif/kararlı çıktı diye daha az alan seçmek için sonradan eşik değiştirilmedi.

apply eğitim yapmaz ve alan listesi değiştirmez. Kaynak kaybı/yüksek eksiklikte
durur; bilinmeyen kategorileri UNKNOWN'a dönüştürüp oranlarını raporlar. Medyan
kayması eğitim IQR'ı cinsindendir; bu bir istatistiksel anlamlılık testi değildir.
Yeni run aktif sürümü otomatik değiştirmez. URF/K-Medoids için kategori kodlama
ve uzaklık tasarımı sonraki aşamadır. Henüz küme/DEA/şube hedefi yoktur.

## Doğrulama

```bash
PYTHONPATH=python_kodlari .venv_flaml/bin/python -m unittest discover -s python_kodlari/tests -p 'test_dinamik_secim.py' -v
```

Testler yeni alan/isimleri, tam takvim bağlantısını, toplulaştırmayı, kategori
sözlüğünü, anahtar/çıktı sızıntısı kontrollerini ve gelecek aylar değişse de aynı
seçim sonucunu sınar. Her run ayrıca bütün native modelleri, tüm permütasyon
kayıplarını, kararları, çıktı hashlerini ve şema uygulamasını yeniden kontrol eder.
