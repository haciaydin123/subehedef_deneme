# Dinamik alan sözlüğü ve seçim protokolü

Bu çalışma sözlükten okunan değişken sayısı/adlarıyla çalışır. Kaynak CSV, anahtarlar,
veri düzeyi, alan rolleri, tipleri, toplulaştırma, zamanında bilinirlik ve seçim
eşikleri config.json içindedir. Kaynağı tanımsız alanlar profile/karar raporunda
review_unmapped olarak tutulur; otomatik açıklayıcı alan yapılmaz. Kimlik ve çıktı
kaynakları aday olarak çift eşlenemez. İş anlamı otomatik veri tipinden çıkarılmaz.

Desteklenen düzeyler branch_month, customer_month, customer_branch_month.
Müşteri düzeyinde ortalama, toplam, ağırlıklı ortalama, tekil sayım, kategori payları,
mod ve şubede sabit değer toplulaştırmaları tanımlanabilir. Girdi aylık ve tarihli
olmalıdır. Günlük bakiye/devir muhasebesi bu modülün görevi değildir; müşteri-şube
parçaları doğru gün ağırlıklı aylık katkılarla hazırlanmalıdır. İlan edilen tekil
anahtarlar, şube-grup tutarlılığı, sayılar ve YYYY-MM tarihleri doğrulanır.

Her şubenin t özelliklerine t+1 aylık üretimi tam takvim eşlemesiyle bağlanır;
eksik ayda sonraki gözlem yanlışlıkla bir ay ileri sayılmaz. Availability lag
varsa yalnızca geçmiş takvim ayı alınır. İlgili ay sonunda bilinmeyen alan aday olamaz.

## Seçim

Veri sonu 2026-12; son 3 etiket ayı seçim dışında.
Seçim sonu 2026-09. 3 genişleyen eğitim
penceresi, her birinde 3 doğrulama ayı.
Tüm hedeflerde aynı CatBoost ailesi/ayarları kullanılır; model her hedef/dönemde
ayrı eğitilir. Şube gruplarına göre algoritma seçimi yapılmaz.

Kalite/korelasyon kuralları yalnızca en erken eğitim penceresinde fit edilir.
Eksiklik, baskın/sabit değer ve kategori sayısı incelenir. Sayısal Spearman
korelasyonu eşik üstündeyse alan sözlüğü önceliği, ardından ad sırası temsilciyi
belirler. Bu tercih iş kuralıdır; üretim başarısıyla eşdeğer değildir. Üretim
toplamları/kapanışlar ve sözlükte proxy olarak işaretli alanlar X'e alınmaz.
HP/segment alanları V4 örneğinde mevcut banka varlığından türediği için review
rolündedir. Gerçek veride bağımsız potansiyel bilgisi tanımıyla yeniden değerlendirilebilir.

Eksik sayısalların medyanı ve kategori sözlüğü her modelin kendi TRAIN verisinden
öğrenilir. Tanımsız kategoriler ayrı unknown değerine çevrilir. Doğrulama bilgisi
bu parametrelere girmez. Model üretimi doğrudan açıklar; stok ekleyen anchor yoktur.

Permutation importance 8 tekrar ile
aynı ay/coğrafi grup içinde şubeler arasında karıştırılır. Her hedefte ΔWAPE
hesaplanır; hedeflerin eşit ağırlıklı ortalaması alınır. Karıştırma tekrar std'si
güven aralığı değildir. Pozitif ortalama katkı ve en az
2 pozitif dönem gerekir. Güvenilirlik
kontrolünde modellerin eğitim ortalaması referansından yeterince iyi olması
aranır; zayıf model halinde alanlara review_model_weak verilir. must_keep yalnızca
sözlükte açık iş kararı varsa uygulanır; veri kalitesi hatasını geçersiz kılamaz.

Korelasyonlu alanlar tekil önemi paylaşabilir. Bu seçim küme kalitesinin, nedensel
potansiyelin veya adil hedef dağıtımının kanıtı değildir. Kararlar ve temsilciler
raporlanır; seçilmiş set küme kararlılığı/iş profilleriyle ayrıca sınanmalıdır.

## Aylık kullanım

selected_schema.json alanları, sırası, tipleri ve en erken TRAIN'den dondurulmuş
medyan/kategori sözlüğünü taşır. apply komutu yeni ayı yeniden fit etmeden hazırlar.
Gerekli alan kayıpsa durur; yeni sütunları raporlar. Drift raporu eksiklik,
yeni kategori oranı ve eğitim IQR'ına göre medyan kaymasını gösterir; istatistiksel
anlamlılık veya otomatik ret kararı iddia edilmez. run yeni bir aday sürüm çıkarır.
--baseline ile eski şemaya göre eklenen/çıkarılan alanlar raporlanır. Yeni sürüm
aktif şemayı veya önceki tahmin modelini sessizce değiştirmez.

cluster_input_latest.csv tek dönemde şube başına bir satır ve yalnızca seçilmiş
alanları içerir. cluster_input_panel.csv aylık izleme içindir; 26 ayı tek dönem
şubeleri gibi kümelememek gerekir. Kategori alanları metin olarak korunur;
URF/uzaklık yöntemine uygun kodlama sonraki aşamada yapılmalıdır.

V4 verisinin finansalları ve bölgesel alanları sentetiktir. Dışarıda bırakılan
son aylar önceki çalışmalarda görüldü; yeni dokunulmamış test oldukları iddia edilmez.

Bağımsız doğrulama bütün kayıtlı modelleri yükler, tüm permütasyon kayıplarını ve
seçim kararlarını yeniden hesaplar; kaynak/kod/çıktı hashleri ile dondurulmuş
kümeleme dosyalarını kontrol eder. Bu işlem kümeleme çalıştırmaz.
