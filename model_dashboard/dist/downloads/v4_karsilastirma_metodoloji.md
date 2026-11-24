# V4 eğitim ve karşılaştırma protokolü

200 şube ve üç gösterge (TL vadesiz, TL vadeli, kredi aylık ortalaması) için
İstanbul/Ankara/İzmir 80 şube ve diğer iller 120 şube olarak ayrı eğitilir.
Her algoritma aynı satırları, aynı özellikleri ve aynı zaman ayrımını kullanır.

## Zaman ayrımı ve aidiyet

- Bilgi t ayı sonundaki görünüm, etiket t+1 ayının ortalama bakiyesidir.
- Eğitim hedef ayları Şubat 2025–Haziran 2026: 3.400 satır.
- Doğrulama Temmuz–Eylül 2026: 600 satır. Test Ekim–Aralık 2026: 600 satır.
- İlk iki ay geçmiş pencere için kullanılır. Aralık 2026 bilgileriyle Ocak 2027
  için 200 girdide etiketsiz tahmin üretilir.
- Şube demografileri sabit müşteri ana tablosundan yeniden birleştirilmez;
  V4'ün tarihli, ay sonu aktif müşteri kitlesinden hesaplanmış tablosu kullanılır.
- Üretim/bakiye gerçekleşmeleri, gün ağırlıklı müşteri-şube parçalarından gelir.
- Gelecek ayın olayları özelliklere katılmaz. Parametre seçiminde test kullanılmaz.

## Üç özellik görünümü

1. Yapısal (25): demografi, müşteri sayısı, gelir, kapasite, bölge ve takvim.
   Ürün bakiyesi ve varlık/HP türevleri yoktur. Doğrudan bakiye öğrenir.
2. Geçmiş bakiyeli (34): önceki karşılaştırmayla aynı 25 + 5 varlık/HP alanı
   + ilgili göstergenin mevcut/önceki/son üç ay/büyüme alanları.
   Tahmin = max(0, mevcut aylık ortalama + model(y_next − mevcut ortalama)).
3. Hareket + ay sonu (49): aynı 34 alan + 15 güncel alan. Bunlar ilişki skoru,
   yeni/ayrılan/devir gelen/devir giden müşteri sayıları, dış kaynak giriş/çıkışı,
   müşteri-günü, ay içi hizmet verilen müşteri sayısı, müşteri sayısı değişimi,
   müşteri-günü/ay sonu müşteri oranı, ilgili göstergenin kapanış stoku,
   kapanış−ortalama farkı ve mevcut ayın devir/devir dışı stok değişimleridir.
   Tahmin = max(0, ay sonu stoku + model(y_next − ay sonu stoku)).

Üç algoritma × üç gösterge × iki grup × üç görünüm = 54 model ailesi.
Her birinin ayrı evaluation ve production kaydı vardır: 108 model dosyası.

## Algoritmalar ve seçim

LightGBM, Random Forest ve CatBoost için önceki V3 karşılaştırmasındaki aynı
üç aday yapılandırma korunur. Öğrenme oranı boosting modellerinde 0,04,
üst sınır 800 ağaç ve doğrulama MAE ile 50 turluk erken durdurmadır.
RF adayları 200/300 ağaç, derinlik 6/10/sınırsız ve min yaprak 10/5/3'tür.
Tohum 20261120, CPU iş parçacığı 2. Ayrıntılı adaylar ayar_arama.csv içindedir.
LightGBM/CatBoost native kategorik, RF yalnızca eğitimde fit edilen one-hot
kodlama kullanır. Sayısal ölçekleme yapılmaz. Aday seçimi doğrulama MAE'dir.
Testten önce ayar ve ağaç sayısı dondurulup TRAIN+VALIDATION ile yeniden fit edilir.
Test skorları saklandıktan sonra tüm etiketli dönemlerle ayrı Ocak modeli eğitilir.

## Referanslar, hata ve süre

Üç referans V4'ün kendi satırlarından yeniden hesaplanır: mevcut aylık ortalama,
mevcut ay sonu stoku ve eğitim hedef ortalaması. V3 tahminleri V4'e taşınmaz.
Ay sonu referansı hareket görünümündeki anchor bilgisine erişir.
MAE/RMSE TL, WAPE mutlak hata toplamı / gerçekleşme toplamı, R² ve toplam
sapması raporlanır. Banka WAPE'si grup yüzdelerini ortalayarak hesaplanmaz.
Fit süreleri seçilen modelin `.fit` çağrısıdır; RF dönüştürücüsünü içerir.
Arama, önem ve dosya işlemleri ayrıca tam pipeline süresine dahildir.
Süreler tek yerel koşudur; genel hız garantisi değildir.

## Önem ve bağımsız kontroller

Her modelde TRAIN'de fit edilmiş seçilmiş modelle yalnızca doğrulamada sekiz
tekrarlı permutation importance hesaplanır. Aynı ay içinde şubeler karıştırılır,
aynı eşleştirme üç ay boyunca korunur. Anchor alanı karıştırılırsa dışarıdan
eklenen anchor da birlikte karıştırılır. Native gain, RF impurity decrease ve
CatBoost PredictionValuesChange farklı tanımlardır; ortak ölçü permütasyondur.
Geçmiş/ay sonu bilgileri otomatik kümeleme veya DEA girdisi sayılmaz.

108 kayıtlı model yeniden yüklenip test ve Ocak tahminleriyle eşleştirilir.
Toplam/grup/aylık metrikler satırlardan bağımsız hesaplanır. Kaynak hashleri,
gelecek etiketleri, mevcut ortalama/kapanış stokları ve dinamik müşteri kitlesi
raw V4 tablosuyla eşleştirilir. CSV dosyalarındaki satır anahtarları kontrol edilir.

## Yeniden çalıştırma

Proje kökünde, requirements.txt sürümlerini içeren Python 3.12 ortamıyla:

```bash
python python_kodlari/v4_egitim_seti_hazirla.py
python python_kodlari/model_karsilastir.py --protocol v4 --source outputs/model_karsilastirma_v4 --output outputs/model_karsilastirma_v4
python python_kodlari/model_karsilastir_dogrula.py --source outputs/model_karsilastirma_v4 --output outputs/model_karsilastirma_v4
python python_kodlari/model_v4_raporla.py
python python_kodlari/dashboard_veri_hazirla.py
```

Veri ve sonuçlar sentetiktir. Banka üst hedef büyümeleri modele verilmez.
Şube hedef dağıtımı bu tahmin karşılaştırmasından ayrı bir tasarım aşamasıdır.
