# Parametre senaryoları protokolü

Önce doğrulanmış değişken seçiminin ortak alanları kendi geçmiş doğrulama
mean_delta_wape sırasına konur. Her oran için ceil(P × oran) orijinal alan alınır.
must_keep alanlar bu bütçenin içinde korunur; sayıları bütçeyi aşarsa işlem durur.
Bu oran sütun kodlamasından önce uygulanır. mtry ise RF’ye verilen kodlanmış
matrisin gerçek sütun sayısına göre max(1, floor(P_encoded × mtry_ratio)) olur.
Her modelin encoded_columns, encoded_feature_count ve effective_mtry bilgisi saklanır.
Her ağacın max_features_ değeri bağımsız doğrulamada yeniden kontrol edilir.
Bu RF aday bütçesidir; geçerli bölünme ararken kütüphane ek alan inceleyebilir.

Toplam banka küme sayısı coğrafi gruplara referans şube sayılarıyla orantılı
dağıtılır. Her grupta en az iki küme, en fazla floor(n/minimum_küme_boyutu)
olacak şekilde en büyük kota açığı sırayla doldurulur. Bu sayısal kapasite kontrolü,
gerçek kümelerin boyut kuralını sağlayacağını garanti etmez; ölçülen sayılar ayrıca kontrol edilir.

Her alan oranı × banka toplam küme sayısı bir ayrı senaryodur. Üç yöntem aynı
senaryodaki aynı alanları kullanır. K-Prototypes karma alan koşulu yoksa açıkça
uygulanamaz sayılır. Seçim sonrası bir kategorik alan eklenerek oran bozulmaz.

Her grup ve senaryonun ortak silhouette ölçüsü, ebeveyn şemanın TÜM alanlarıyla
referans ayda fit edilmiş aynı Gower mesafesinden gelir. Böylece alan sayısını
azaltarak değerlendirme geometrisi değiştirilmez. Bu yine Gower geometrisini
tercih edebilir ve nihai iş başarısı/DEA kanıtı değildir.

Bankadaki senaryo puanı grup silhouette'larının referans şube sayısıyla ağırlıklı
ortalamasıdır. Yalnızca ana yöntem URF'nin referans puanlarıyla global senaryo
seçilir. Önce tüm gruplarda yakınsama/boyut kuralı aranır; hiçbiri sağlamazsa sonuç
exploratory_size_review olarak açıkça işaretlenir. Eşitlikte küçük toplam küme
sayısı ve ardından küçük alan oranı tercih edilir. İzleme ARI'si bu seçime katılmaz.
Küme senaryosu her gösterge ve grup için ortak, grup içindeki k kota dağılımıdır.

Her alt çalışmanın METODOLOJI.md dosyası URF yapay veri üretimi, PAM, ölçekleme,
tohum/alt örneklem ve aylık yeniden kümeleme detaylarını içerir. Referans öncesi
doğrulama önemleri kullanılır; gelecek aylar seçim için kullanılmaz. Veriler
sentetiktir ve bu dönemler daha önce incelenmiştir; yeni bağımsız test iddiası yoktur.

Önceki tam alan/sqrt çalışması korunur. Bu paket yeni bir parametre deneyidir.
CLI check ve apply, grid kökünde de çalışır; apply referansta seçilen aktif alt
senaryonun dondurulmuş modellerine yönlenir ve yeniden eğitim yapmaz.
