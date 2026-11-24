# LightGBM ile ilk model ve değişken önemi çalışması

Bu çalışma banka hedeflerini şubelere dağıtmanın **değişken değerlendirme** aşamasıdır.
Üç göstergenin sonraki ay gerçekleşmesini açıklayan modeller kurulur. Kümeleme,
DEA ve Ocak bütçesini şubelere paylaştırma bu aşamada yapılmaz.

## 1. Veri ve tahmin sorusu

Kaynak V3: 10.127 müşteri, 200 şube, 26 ay (2024-11 / 2026-12).
Finansal tutarlar, şubeler ve ekonomik koşullar sentetiktir; kaynak demografiler
Kaggle profillerinden gelir. Kaynak profil alanları, simülasyon başlangıcında
biliniyor ve sabit kabul edilmiştir. Gerçek uygulamada tarihli müşteri profili gerekir.

Bir satır bir şubenin bir ayını temsil eder. Tahmin ufku **bir ay**:

`t ayında bilinen müşteri/şube bilgileri → t+1 ayının ortalama bakiyesi`

| Gösterge | Gerçekleşme alanı | Etiket |
|---|---|---|
| TL vadesiz mevduat | tl_vadesiz_ort_bakiye_tl | y_tl_vadesiz_tl |
| TL vadeli mevduat | tl_vadeli_ort_bakiye_tl | y_tl_vadeli_tl |
| Kredi | kredi_ort_bakiye_tl | y_kredi_tl |

Kredi ihtiyaç + konut + taşıt + KMH toplamıdır; kredi kartı borcu ayrı tutulur.
Aylık müşteri bakiyesi ortalamaları aynı şube ve ayda toplanır. Bu, ay sonu
bakiyesi veya aylar arası bakiye toplamı değildir. Eğitimde hedef gerçekleşmiş
sentetik bakiyedir; kullanıcının %15/%10/%5 bütçe artışları etiket veya özellik değildir.

## 2. Şube grupları ve iki model görünümü

İstanbul–Ankara–İzmir **UC_BUYUK_IL** (80 şube), kalan iller **DIGER_ILLER**
(120 şube) olarak ayrı modellenir. Üç gösterge × iki grup = **6 yapısal model**.
Karşılaştırma için aynı gruplarda **6 gecikmeli bakiye modeli** daha kurulur.
Müşteri satırları bağımsız şube örnekleri gibi kullanılmaz.

**Yapısal model:** 25 özellik. Müşteri sayısı, personel, gelir profili, demografi,
ilişki/aktivite oranları, bölgesel koşullar ve bilinen takvim bilgileri. Amaç,
benzer şubeleri tanımlamakta kullanılabilecek değişkenleri değerlendirmektir.
Takvim alanları tahmine yardımcıdır, kümeleme adayı değildir.

**Gecikmeli model:** 34 özellik. Yapısal özelliklere t bakiyesi, t−1 bakiyesi,
t dahil son üç ay ortalaması ve son aylık büyüme eklenir. HP/segment oranları
ve ortalama banka varlığı da bu görünümde yer alır. Bunlar mevcut banka
bakiyelerinden türetildiği için bağımsız müşteri potansiyeli gibi sunulmaz.

Gecikmeli model **bakiye değişimini** öğrenir: `d = y(t+1) − bakiye(t)`.
Nihai tahmin `max(0, bakiye(t) + tahmin(d))` olur. Bu, ağacın aynı bakiyeyi
yeniden öğrenmesine gerek bırakmaz. Yapısal model doğrudan bakiye öğrenir.

Şube/müşteri/ilçe kimlikleri, aynı hedef ayın gerçekleşmeleri, banka bütçesi,
hazır model skorları ve mükerrer ürün toplamları özelliklere sokulmaz.
GST/GST2 gösterge tanımıdır; her göstergeye ayrı model kurulduğu için modele
sayısal açıklayıcı değişken olarak verilmez. HP müşteride tanımlıdır; bu pilot
şubeyi bir bütün olarak modeller, HP payını yalnızca gecikmeli görünümde kullanır.
Personel henüz HP gruplarına paylaştırılmadığından bağımsız şube-HP modelleri kurulmaz.

## 3. Zaman ayrımı

İlk iki gözlem ayı üç aylık geçmiş hesaplarının başlangıcıdır. Son gözlem ayının
sonraki ay etiketi yoktur. 5.200 şube-aydan **4.600 etiketli satır** kalır.

| Ayrım | Özellik ayları | Tahmin edilen hedef ayları | Satır |
|---|---|---|---:|
| TRAIN | 2025-01 / 2026-05 | 2025-02 / 2026-06 | 3.400 |
| VALIDATION | 2026-06 / 2026-08 | 2026-07 / 2026-09 | 600 |
| TEST | 2026-09 / 2026-11 | 2026-10 / 2026-12 | 600 |
| FORECAST | 2026-12 | 2027-01, etiketi henüz yok | 200 |

Rastgele satır bölünmez. Aynı hedef ayın bütün şubeleri aynı ayrımda kalır.
Test, ardışık üç **bir ay ileri** tahmini içerir: Kasım tahmini için Ekim
gerçekleşmesinin artık biliniyor olması normaldir. Bu, Eylül'den tek seferde
üç ay ileri tahmin değildir. Aynı şubeler geçmiş ve gelecek dönemlerde bulunur;
yeni bir şubeye genelleme bu deneyde ölçülmez.

## 4. LightGBM ayar seçimi

GBDT regresyon, öğrenme oranı 0,04, en çok 800 ağaç, iki CPU iş parçacığı,
tohum 20261120. Sayısal özellikler ölçeklenmez. Ekonomik yapı native categorical
olarak kullanılır; kategori sözlüğü yalnızca ilk eğitim döneminden öğrenilir.
Gelecekte bilinmeyen kategori eksik kategoriye dönüşür; hedefe göre kodlama yapılmaz.

| Aday | Yaprak | Azami derinlik | Asgari yaprak örneği | L2 |
|---|---:|---:|---:|---:|
| A | 7 | 3 | 40 | 10 |
| B | 15 | 4 | 40 | 10 |
| C | 15 | 4 | 20 | 25 |

Her aday TRAIN üzerinde eğitilir. VALIDATION MAE 50 tur iyileşmezse erken
durdurma uygulanır. En düşük doğrulama MAE'si seçilir. Ağaç sayısı ve parametreler
**test görülmeden dondurulur**. TRAIN+VALIDATION ile bu sabit ağaç sayısında
yeniden eğitim yapılarak TEST değerlendirilir.

Test sonuçları kaydedildikten sonra, Ocak tahmini için tüm 4.600 etiketli
gözlemle ayrı `production` modelleri eğitilir. Bunların içinde eski test ayları
artık eğitim verisidir. Bu nedenle raporlanan test başarısı `evaluation`
modellerine aittir; production modellerinin eğitim içi skoru gibi sunulmaz.

## 5. Karşılaştırma ve başarı

İki basit referans hesaplanır: **son ayın bakiyesini aynen korumak** ve eğitim
örneklerinin ortalama bakiyesini tahmin etmek. LightGBM bunlardan iyi çıkmak
zorunda değildir; sonuçlar olduğu gibi raporlanır. Random Forest çalıştırılmadığı
için ona göre hız veya başarı üstünlüğü iddia edilmez.

- MAE: şube başına ortalama mutlak hata, TL.
- RMSE: büyük hatalara daha duyarlı hata, TL.
- WAPE: `mutlak hataların toplamı / gerçek bakiyelerin toplamı`.
- R²: testteki açıklama başarısı; gerekirse negatif kalır.
- Toplam sapması: `tahmin toplamı / gerçek toplam − 1`.

WAPE, şube gruplarının yüzdelerini aritmetik ortalayarak hesaplanmaz; satır
düzeyindeki tahmin ve gerçekleşmeler yeniden toplanır. Eğitim süreleri yerel
ortamda ölçülür; kurulum/indirme süresi modelin fit süresine dahil değildir.

## 6. Değişken önemi ve dönemsel kararlılık

LightGBM gain önemleriyle birlikte **doğrulama MAE'sine dayalı permutation
importance** hesaplanır. Pozitif `delta_mae_tl`, o alanın karıştırılmasının
hatayı artırdığını gösterir. Negatif değerler sıfırlanmaz. Normalize önem,
delta MAE'nin doğrulama gerçek bakiye ortalamasına oranıdır; nedensel katkı değildir.

Panel yapısını korumak için şubeler aynı ay içinde yer değiştirir; aynı şube
eşleştirmesi bütün doğrulama aylarında korunur. Böylece aylar karışmaz ve üç
aylık özellik yolu birlikte taşınır. Son doğrulamada sekiz tekrar yapılır.
Takvim alanları aynı ayda bütün şubelerde aynı olduğundan bu permütasyonda
önemleri 0 çıkabilir; bu onların zaman etkisi olmadığını göstermez.

Birbiriyle ilişkili alanların öneminin bölünebilmesi nedeniyle kapasite,
demografi, gelir, ilişki, bölge ve geçmiş bakiye grupları **birlikte** de
karıştırılır. Gecikmeli modelde t bakiyesi karıştırıldığında tahminin dışarıdan
eklenen t bakiyesi de karıştırılır. Gecikmeli modelin gain skoru ise yalnızca
değişim öğrenen ağaçlara aittir; tüm tahmin fonksiyonunun önem skoru değildir.

Yapısal model için iki ek geçmiş pencere kullanılır:
1. 2025-12'ye kadar eğitim → 2026-01/03 doğrulama.
2. 2026-03'e kadar eğitim → 2026-04/06 doğrulama.
3. Ana pencere: 2026-06'ya kadar eğitim → 2026-07/09 doğrulama.

Her geçmiş pencere kendi adaylarını kendi doğrulamasında seçer; daha sonraki
dönemle seçilmiş ayar önceki pencereye aktarılmaz. Geçmiş iki pencerede beş
permütasyon tekrarı vardır. Üç pencereden en az ikisinde pozitif olan ve ortalama
normalize önemi pozitif kalan yapısal alanlar **kararlı aday** işaretlenir.
Bu işaret anlamlılık testi değildir; tekrar standart sapması güven aralığı değildir.
Takvim, geçmiş bakiye ve varlıktan türetilen alanlar kümeleme listesine otomatik girmez.
Zayıf modelin düşük önem verdiği değişken otomatik silinmez. Nihai özellik seçimi
henüz yapılmamıştır; dosya sonraki kümeleme aşaması için aday listesidir.

## 7. Sınırlar ve dosyalar

Sentetik veriyle bu deneyin akışı ve tutarlılığı ölçülür. Gerçek bankadaki
satış potansiyeli, nedensel etki veya Ocak hedefinin gerçekleştirilebilirliği
kanıtlanmaz. Özellikle Kasım–Aralık 2026 sonradan ayrı bir sentetik devam
mekanizmasıyla üretildiğinden test aylık performansı ayrıca raporlanır.

`veri/`: train, validation, test, etiketli birleşik tablo, Ocak girdileri ve özellik sözlüğü.
`modeller/`: native LightGBM metin modelleri; evaluation/production ayrı, 12 metadata.
`sonuclar/`: hatalar, süreler, tahminler, önemler, grup önemleri ve kararlılık.
`kod/`: veriyi hazırlayan/eğiten pipeline, kaydedilmiş modelden tahmin yardımcısı.

## Kaynaklar

- [LightGBM regresyon API](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html)
- [LightGBM erken durdurma](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.early_stopping.html)
- [Permutation importance ve ilişkili değişkenler](https://scikit-learn.org/stable/modules/permutation_importance.html)
- [Zaman sıralı doğrulama](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)
