# Performansa göre ortak kümeleme seçimi

Doğrulanmış URF / Gower / K-Prototypes parametre gridini okuyup yöntem, alan
oranı ve banka toplam küme sayısını birlikte seçer. Tek aday bütün göstergeler
ve coğrafi gruplar için ortaktır. Referans modelleri yeniden eğitilmez.

Proje kökünden, gerekli paketlerin kurulu olduğu Python ile:

```bash
python python_kodlari/otomatik_kumeleme.py select --config python_kodlari/ayarlar/otomatik_kumeleme.json
python python_kodlari/otomatik_kumeleme.py verify --run outputs/otomatik_kumeleme/KOSU_KODU
python python_kodlari/otomatik_kumeleme.py apply --run outputs/otomatik_kumeleme/KOSU_KODU --panel hazirlanmis_sube_paneli.csv --period 2027-01 --output yeni_ay_atamalari.csv
```

`select --source GRID_KLASORU` ile farklı doğrulanmış grid kullanılabilir.
`--output` mevcut olmayan ayrı bir koşu klasörü belirtir. Varsayılan koşu
kimliği kaynak manifesti, politika ve kod içeriklerinden üretilir; var olan
koşu üzerine yazılmaz. `latest.json` tamamlanmış ve doğrulanmış koşuyu gösterir.

Aylık akış: kaynak paneli ve karşılaştırma gridini güncelle, seçim ve ayrı
değerlendirme aylarını config içinde ilerlet, `select`, `verify`, ardından
dashboard snapshot betiğini çalıştır. Kaynak modelin referans tarihini ilerletmek
ve yeniden eğitim yapmak grid aşamasının sorumluluğudur. Şube müşteri devirleri
hazırlanmış güncel aylık girdiye yansır. `apply` yalnızca sabit modellerle atar.

Eşik ve ağırlıklar sentetik pilot politikasıdır. Skor doğruluk yüzdesi değildir.
Hiçbir aday uygun değilse ana model seçilmez ve `apply` çalışmaz. Mevcut aday
uygunsa minimum_gain olmadan değiştirilmez. Farklı bir mevcut adayı korumak için
config içindeki incumbent alanına `senaryo::yöntem` kodunu yazın.

Taşınabilir çıktı ZIP'i config, kanıt metrikleri, doğrulama kodu, seçilen native
modeller ve hazırlanmış panel içerir. Açılan pakette `code/otomatik_kumeleme.py`
ile `apply --run PAKET_KLASORU ...` veya `verify --run PAKET_KLASORU` çalışır.
Yeni grid seçimi için kaynak grid de gereklidir. requirements.txt içindeki
paketleri kurun; joblib modellerini yalnızca güvenilir kaynaktan yükleyin.

DEA ve hedef dağıtımı bu aşamaya dahil değildir.

Tam şube kapsamı zorunludur. `ayarlar/otomatik_kumeleme.json` içindeki
`dea_coverage`, ana DEA sözlüğünü ve girdi kontrol sözlüğünü okur. Küme alt
sınırı sabit 12 olarak bırakılmaz; bütün senaryoların gereksinimi birlikte
hesaplanır (şu an 15). Yeni girdi eklenince önce select tekrar çalıştırılır.

Referans/doğrulama ayları seçimde kullanılır. Karar sabitlendikten sonra
bütün kayıtlı atamalar kaynak şube evreniyle birebir karşılaştırılır. Sonraki
ayın büyüklük kontrolü başarısızsa aktivasyon durur; sonraki ayın verisiyle
aday sırası geriye dönük değiştirilmez. apply da aynı kapsam ve boyut
kontrolünden geçer. Sözleşmesiz eski pakete yeni kodla apply yerine yeniden
select gerekir; eski arşivler kendi kodlarıyla tarihsel olarak doğrulanabilir.

Coğrafi ayrışma koşulu da zorunludur: `thresholds.worst_group_silhouette`
her grubun doğrulama dönemi ortalama silhouette değerine aynı şekilde
uygulanır. Mevcut pilot sınır 0.10'dur; evrensel veya otomatik kalibre edilmiş
bir sınır değildir. Bir grubun yetersiz değeri banka ortalamasıyla örtülemez.
Eksik ölçü koşulu geçmez; değerlendirme ayları bu seçime katılmaz.
`geographic_quality.json/csv` bütün adayların grup koşullarını gösterir.

Bu kuralla mevcut 8 uygulanabilir adayın tamamı elendi. `selected=null` güncel
karardır; apply ve varsayılan DEA komutu durur. Önceki 5 kümeli model ve DEA
paketleri tarihsel olarak korunur; yeni politikaya uygunmuş gibi aktive edilmez.
