# Hedef — Model Laboratuvarı

Metodoloji, pipeline, değişken seçimi ve şube kümelerinin Türkçe dashboard'u.
“Ana model seçimi” güncel açılış ekranıdır. Coğrafi grupların her birinde
doğrulama dönemi ortalama silhouette değeri için 0.10 pilot alt sınır uygulanır.
Mevcut 8 uygulanabilir aday bu koşulları sağlamadığı için onaylı ana model yoktur.
Yeni atama/DEA durur; şubeler elenmez ve önceki model otomatik geri alınmaz.
Eşik yapılandırılabilir; veriyle optimize edilmiş veya evrensel bir sınır değildir.

“Verimlilik” ekranı önceki 5 kümeli modelin 200 şubeyi kapsayan DEA koşusunu
**tarihsel sonuç** olarak gösterir; güncel politikayla onaylanmış skor değildir.
Girdi aylık ortalama aktif müşteri sayısıdır (müşteri-gün / takvim günü); çıktılar
TL vadesiz, TL vadeli ve kredi aylık ortalama bakiyeleridir. VRS skoru ölçek
varsayımına duyarlılık için gösterilir. Skor, radyal artış mesafesi, ürün bazında
sınır projeksiyonları, emsal şubeler ve λ katsayıları birlikte incelenir.
Gruplar, kümeler, gösterge ve şube aramasıyla filtrelenir. Aynı sayfadaki “DEA
girdi kontrolleri” bölümü müşteri / müşteri+personel senaryolarını veri kalitesi,
kapsam, aynı şubelerde skor farkı ve duyarlılıkla karşılaştırır. Kontrol bölümü
banka genelindedir; görünüm filtrelerinden bağımsızdır. Ana model değişmez.
Etkili emsaller tek tek gösterilir; ±%1 ölçüm stresi ile emsal çıkarma deneyleri
ayrıdır. Personel verisi sentetik ve aylarda sabit olduğundan gerçek personel
hareketleri bu çalışmada doğrulanamaz. Boyut ön kontrolü
geçmeyen küme varsa işlem durur; hiçbir şube elenerek kısmi sonuç üretilmez. DEA projeksiyonu tahmin veya bütçe hedefi
değildir; hedef dağıtımı sıradaki aşamadır. Görünen ortalama skor farklı küme
sınırlarının betimleyici ortalamasıdır; banka genelinde tek başarı sırası değildir.

“Ana model seçimi” önceki aşamadır. Yöntem, %25/%50 alan seçimi ve 5/7
banka toplam küme senaryosu birlikte seçilir. Kalite ve boyut koşullarını geçen
adaylar silhouette ve üç kararlılık ölçüsüyle sıralanır. Hiçbir aday uygun değilse
ana model onaylanmaz. Önceki sentetik koşuda Gower + K-Medoids, %50 / 11 alan /
5 küme seçilmişti. Güncel coğrafi kalite politikasıyla onaylı aday yoktur. İki coğrafi grup kendi modellerini kullanır; üç göstergede ortak
yöntem ve senaryo korunur. Ekim–Kasım 2026 seçim, Aralık ayrı geçmiş dönem
kontrolüdür. Aylar daha önce incelendiği için yeni dokunulmamış test iddiası yoktur.

“Şube kümeleri” seçim onaylıysa o yöntem/senaryoyu, onaylı aday yoksa
referans denemelerini inceleme amacıyla açar.
Diğer yöntem ve senaryolar yerel seçicilerden incelenebilir. Önceki URF referans
koşusu kaynak dosyalarında korunur. DEA sonuçları Verimlilik ekranındadır; hedef dağıtımı henüz yapılmadı.
Kodlama sonrası %25 mtry yalnızca URF için geçerlidir. Ortak değerlendirme her
senaryoda tam ebeveyn alan setidir. K-Prototypes'in uygulanamadığı senaryolar
etiketlenir. Seçim puanı doğruluk yüzdesi değildir; ortak Gower silhouette ölçüsü
Gower yöntemini avantajlı kılabilir. Koşullar ve ağırlıklar ayar dosyasındadır.

“Değişken seçimi” alan sözlüğünü, dinamik aday/seçilen
sayılarını, karar gerekçelerini, üç geçmiş dönemdeki katkıları ve şema dosyalarını
gösterir. Bu ekranın göstergeleri ve alanları run manifestinden okunur. Önceki
stok tahmin deneyleri FLAML/karşılaştırma ekranlarında tarihsel olarak korunur.
Tüm sayfalardaki güncel çalışma özeti bu ekrana bağlanır. Seçilmiş alanların
dönem kararlılığı ve fiziksel kaynak–model alanı sözlüğü aynı sayfadadır.
Anlık görüntü oluşturulunca script/stil adresleri içerik sürümü alır; tarayıcı
önceki çalışma dosyalarını önbellekten yüklemeye devam etmez.
LightGBM için ayrı sonuç/önem sekmeleri yoktur. Üç algoritmanın sonuçları ve
değişken önemleri “Algoritma karşılaştırması” içinde birlikte gösterilir.
Eski #results ve #importance bağlantıları bu ekrana yönlendirilir.
Genel bakış, veri kapsamı ve metodoloji güncel V4 ortak modeline aittir.
“Algoritma karşılaştırması” sayfasında aynı veriyle yeniden eğitilmiş LightGBM,
Random Forest ve CatBoost'un hata, süre, parametre ve değişken önemleri gösterilir.
V3/V4 seçimiyle deneyler ayrı görüntülenir. V3'te 25/34, V4'te 25/34/49 alanlı
görünümler vardır. V4'ün 49 alanlı görünümü güncel müşteri/ürün hareketlerini ve
ay sonu stokunu kullanır; ay sonu stok referansı da ölçülür.
HTML/CSS/JavaScript ile çalışır; tarayıcıda model eğitimi yapmaz.
Filtreler, metrikler, önem sıraları ve CSV dışa aktarma yerel tarayıcıda çalışır.

“FLAML AutoML” sekmesi bütün göstergelerde ve şube gruplarında ortak algoritmayı
gösterir. Üç göstergenin banka geneli doğrulama WAPE ortalamasında tek seçim
yapılır; güncel seçim CatBoost'tur. Gösterge/grup için ayrı eğitimler vardır,
algoritma ailesi değişmez. Referanslar yalnızca karşılaştırma içindir.
Önceki altı aramanın kayıtları yeniden kullanıldı. Arama eğrileri seçilen
algoritmanın eski denemelerde iyileşen doğrulama noktalarıdır; yeni arama değildir.
Bu pilotta FLAML özellik önemleri yeniden hesaplanmadı. Önceki önemler önceki
modellere aittir. Ekim–Aralık test verisi daha önce incelendi; yeniden kullanılan
tarihsel karşılaştırma olarak etiketlendi.

Veri anlık görüntüsünü proje kökünden güncellemek için:

```bash
python3 python_kodlari/dashboard_veri_hazirla.py
```

Yerel inceleme için `dist/index.html` doğrudan açılabilir veya:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory model_dashboard/dist
```

Tamamlanmış aşamalar veri hazırlama, üretim modellerinin karşılaştırılması,
dinamik değişken seçimi, URF/PAM kümeleme ve k adaylarının karşılaştırılmasıdır.
Önceki stok tahmin modelinin önemleri yeniden hesaplanmadı; dinamik seçimdeki
yapısal CatBoost katkıları ayrı çalışmadır. Önceki DEA koşusu tarihsel olarak korunur; güncel politika altında yeni DEA durduruldu.
Veri ve model performansı sentetik bir deneye aittir. Dashboard gerçek banka
başarısı veya neden-sonuç etkisi iddia etmez. İlk V3 testindeki son ay referansı
ve V4'ün bütün referansları gerçek ölçümleriyle korunur. Deneyler farklı veri
üretimleri olduğundan V3/V4 hata farkı tek başına model gelişimi olarak yorumlanmaz.

Doğrulama:

```bash
node --check model_dashboard/dist/app.js
node model_dashboard/scripts/verify.mjs
```

Doğrulama betiği iki veri sürümü, üç gösterge, üç şube grubu, iki dönem ve özellik görünümleri
için ekrandaki metrikleri kaydedilmiş sonuçlarla karşılaştırır; üç algoritmanın
iki önem yöntemini ve indirme dosyalarını da kontrol eder. Programatik kontroller bütün sayfa ve filtreleri kapsar. Ana model seçimi
ekranındaki tam kapsam tablosu ile Verimlilik ekranındaki 200/200 şube sonucu
gerçek tarayıcıda ayrıca kontrol edildi. Diğer ekranlar için programatik
kontroller geçerlidir. Çevrimiçi yayın yapılmadı; yerel dosyalar hazırdır.

Tam şube kapsamı seçim ön koşuludur. DEA ana girdisi ve tanımlı ek girdi
senaryolarının boyut gereksinimleri birlikte okunur; mevcut ortak küme alt
sınırı 15 şubedir. Seçim ekranı her ay kaynak/atanan şube sayılarını ve en
küçük kümeyi gösterir. Yeni ayda kapsam bozulursa apply/DEA durur; girdi
tanımı değişirse önce kümeleme seçimi yeniden çalıştırılır.
