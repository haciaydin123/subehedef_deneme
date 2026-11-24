# Üç algoritmanın karşılaştırma metodolojisi

Bu deneyde finansal veriler sentetiktir. Birim şube-ay; tahmin ufku bir aydır.
İlk LightGBM deneyi ve dosyaları ayrı korunmuştur. Bu koşuda üç algoritma da
aynı kaydedilmiş CSV ayrımlarıyla yeniden eğitildi.

## Ortak veri ve tahmin sorusu

- 200 şube; İstanbul/Ankara/İzmir 80, diğer iller 120. Gruplar ayrı eğitilir.
- TL vadesiz, TL vadeli ve kredi aylık ortalamaları ayrı hedeflerdir.
- Eğitim: hedef ayı Şubat 2025–Haziran 2026, 3.400 satır.
- Doğrulama: Temmuz–Eylül 2026, 600 satır.
- Test: Ekim–Aralık 2026, 600 satır. Üç ardışık bir ay ileri tahmindir.
- Ocak 2027: Aralık 2026 bilgilerini içeren 200 girdide etiketsiz tahmin.
- Yapısal görünüm: 25 özellik, doğrudan sonraki ayın bakiyesi.
- Geçmiş bakiyeli görünüm: aynı 25 + 5 varlık/HP payı + 4 ilgili bakiye alanı.
  Model sonraki ay ile mevcut ay arasındaki farkı öğrenir;
  tahmin = max(0, mevcut bakiye + tahmin edilen fark).
- Müşteri/şube kimlikleri, gelecek gerçekleşmeler ve banka hedef büyümeleri
  modele verilmez. Özellikler mevcut ilk çalışmanın `pipeline.features` tanımını kullanır.

3 algoritma × 3 gösterge × 2 grup × 2 görünüm = 36 model.
Her modelin ayrı evaluation ve production kaydı vardır: toplam 72 dosya.

## Ayar seçimi

Her algoritma için üç aday yalnızca doğrulama MAE ile karşılaştırılır.
Aday sayısı eşittir; ağaç sayısı veya hesaplama bütçesi eşitlenmiş değildir.
Bu, sınırlı bir parametre taramasıdır; algoritmaların ulaşabileceği en iyi sonuç
olarak yorumlanmamalıdır. CPU iş parçacığı sınırı 2, rastgele tohum 20261120.

| Algoritma | A | B | C |
|---|---|---|---|
| LightGBM | 7 yaprak, derinlik 3, min 40, L2 10 | 15 yaprak, derinlik 4, min 40, L2 10 | 15 yaprak, derinlik 4, min 20, L2 25 |
| Random Forest | 200 ağaç, derinlik 6, min yaprak 10, max_features 0,9 | 300 ağaç, derinlik 10, min yaprak 5, max_features 1 | 300 ağaç, sınırsız derinlik, min yaprak 3, max_features 0,8 |
| CatBoost | derinlik 3, L2 10 | derinlik 4, L2 10 | derinlik 6, L2 25 |

LightGBM ve CatBoost için öğrenme oranı 0,04, üst sınır 800 ağaç ve
doğrulama MAE'sinde 50 turluk erken durdurma kullanılır. LightGBM kare hata
regresyonu, CatBoost RMSE ve Random Forest squared_error ile öğrenir;
aday seçimi her üçünde de MAE ile yapılır.

Ekonomik yapı LightGBM ve CatBoost'ta native kategoriktir. Random Forest için
one-hot dönüştürücü yalnızca eğitimde fit edilir; görülmeyen kategori yok sayılır.
Sayısal ölçekleme yapılmaz. RF'nin dönüştürücüsü modelle birlikte saklanır.

## Test ve üretim modelleri

Ayarlar ve ağaç sayısı doğrulamayla dondurulunca eğitim + doğrulama dönemleri
birleştirilerek evaluation modeli eğitilir ve test tahminleri saklanır.
Test dondurulduktan sonra aynı ayarlarla tüm etiketli dönemler kullanılarak
Ocak 2027 için ayrı production modeli eğitilir. Bu modellerin eğitim içi
skoru test başarısı olarak sunulmaz. Testteki sıralamaya bakılarak yeniden
ayar taraması veya özellik seçimi yapılmadı.

İki ortak referans: mevcut ay bakiyesini korumak ve eğitim hedef ortalamasını
kullanmak. Bunlar ilk çalışmadan aynı keyed tahminlerdir; tahmin etiketleri
üç algoritmayla bağımsız olarak eşleştirilir.

## Hata ve süre ölçümü

MAE, RMSE, WAPE, R² ve toplam sapması raporlanır. WAPE tüm satırlardan
mutlak hata toplamı / gerçekleşme toplamı olarak yeniden hesaplanır;
grup oranlarının ortalaması değildir. Aynı görüntüdeki bütün modeller
aynı gerçek değerler ve satır anahtarlarıyla karşılaştırılır.

Fit süresi, `.fit` çağrısını kapsar; RF'de one-hot fit/dönüşümü dahildir.
Doğrulama görünümünde seçilen TRAIN modeli, test görünümünde TRAIN+VALIDATION
yeniden eğitimi gösterilir. İki grup seçiliyse fit süreleri toplanır.
Üç adayın arama toplamı ayrıca gösterilir; doğrulama tahmini ve puanlamasını
da içerir. Önem hesaplama, model kaydı ve production yeniden eğitimi fit
çubuklarının dışında; tüm pipeline duvar süresinin içindedir.
Test tahmin gecikmesi ısınmadan sonra beş çağrının medyanıdır ve çıktı CSV'sinde
1.000 satıra ölçeklenir; az satırlı bu ölçüm bir yük testi değildir.
Süreler aynı makinedeki tek CPU koşusuna aittir; tekrarlı hız benchmark'ı
veya farklı donanım için hız garantisi değildir.

## Değişken önemi

Her model için TRAIN'de fit edilmiş seçilmiş model ve yalnızca Temmuz–Eylül
doğrulaması kullanılır. Özellikler aynı ay içindeki şubeler arasında karıştırılır;
aynı eşleştirme üç ay boyunca korunur. Aynı tohum ve sekiz tekrar kullanılır.
İlgili güncel bakiye karıştırılırsa dışarıdan eklenen anchor da birlikte değişir.
Tekil ve özellik grubu permütasyonları saklanır. Negatif ve sıfır değerler korunur.
Normalize önem = ΔMAE / doğrulama gerçekleşme ortalaması.

Ortak permütasyon ölçüsü algoritmalar arasında aynı tanıma sahiptir. Native
ölçüler farklıdır: LightGBM gain, RF impurity decrease ve CatBoost
PredictionValuesChange. Bu üç native yüzde birbirinin eşdeğeri değildir.
RF'nin one-hot sütun önemleri orijinal kategori alanına geri toplanır.
Geçmiş bakiyeli modellerde native önem dışarıdan eklenen mevcut bakiyeyi kapsamaz.

Üç tarihsel pencerede kararlılık incelemesi ilk LightGBM çalışmasında bulunur.
Bu yeni karşılaştırmada RF/CatBoost için tekrar edilmedi. Bunlara kararlı aday
etiketi verilmez. Kümeleme, DEA ve şube hedef dağıtımı henüz çalıştırılmadı.

## Doğrulama ve yeniden üretim

- Kaydedilmiş eğitim/doğrulama/test/girdi dosyalarının SHA-256 özeti korunur.
- 72 kayıtlı model yeniden yüklenerek hem test hem Ocak tahminleriyle eşleştirilir.
- Şube, banka toplamı ve ay bazındaki metrikler bağımsız kodla yeniden hesaplanır.
- Sekiz tahminleyicinin her gözlemde aynı gerçekle karşılaştırıldığı doğrulanır.
- Kütüphane sürümleri, donanım platformu ve süre notları manifest.json'dadır.

Proje kökünden:

```bash
python python_kodlari/model_karsilastir.py
python python_kodlari/model_karsilastir_dogrula.py
python python_kodlari/model_karsilastirma_raporla.py
python python_kodlari/dashboard_veri_hazirla.py
```

Bağımsız model paketini açtıktan sonra:

```bash
python -m pip install -r kod/requirements.txt
python kod/model_karsilastir.py --source . --output yeniden_calisma
python kod/model_karsilastir_dogrula.py --source . --output yeniden_calisma
```

## Teknik başvuru kaynakları

- [CatBoost fit ve erken durdurma](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor_fit)
- [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
