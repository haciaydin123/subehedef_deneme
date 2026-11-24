# V4 model karşılaştırması

Veri: 26 ayın tamamında aynı mekanizmayla üretilmiş müşteri/ürün hareketleri. Test: Ekim–Aralık 2026.
Müşteri demografileri güncel aktif şube aidiyetinden, bakiye toplamları gün ağırlıklı aidiyet parçalarından gelir.
Her tabloda aynı hedef ve aynı test satırları karşılaştırılır. Finansal gözlemler sentetiktir.

## Yapısal: 25 özellik

| Gösterge | LightGBM WAPE | RF WAPE | CatBoost WAPE | Son ay ortalaması | Ay sonu stokunu koru |
|---|---:|---:|---:|---:|---:|
| TL vadesiz | %10.646 | %12.409 | %11.039 | %4.250 | %2.950 |
| TL vadeli | %13.948 | %16.451 | %13.325 | %4.707 | %3.426 |
| Kredi | %10.029 | %10.336 | %9.751 | %3.466 | %2.267 |

## Geçmiş bakiyeli: 34 özellik

| Gösterge | LightGBM WAPE | RF WAPE | CatBoost WAPE | Son ay ortalaması | Ay sonu stokunu koru |
|---|---:|---:|---:|---:|---:|
| TL vadesiz | %3.847 | %3.857 | %3.906 | %4.250 | %2.950 |
| TL vadeli | %4.458 | %4.525 | %4.439 | %4.707 | %3.426 |
| Kredi | %3.390 | %3.418 | %3.357 | %3.466 | %2.267 |

## Hareket + ay sonu: 49 özellik

| Gösterge | LightGBM WAPE | RF WAPE | CatBoost WAPE | Son ay ortalaması | Ay sonu stokunu koru |
|---|---:|---:|---:|---:|---:|
| TL vadesiz | %2.927 | %2.908 | %2.933 | %4.250 | %2.950 |
| TL vadeli | %3.466 | %3.485 | %3.452 | %4.707 | %3.426 |
| Kredi | %2.292 | %2.306 | %2.290 | %3.466 | %2.267 |

## Doğrulamayla seçilen tahminleyicilerin ayrı test sonucu

Seçim için yalnızca Temmuz–Eylül doğrulama MAE kullanıldı. Test sıralamasıyla parametre değiştirilmedi.

| Gösterge | Doğrulamada seçilen | Doğrulama WAPE | Test WAPE |
|---|---|---:|---:|
| TL vadesiz | ay_sonu_koru | %2.715 | %2.950 |
| TL vadeli | catboost__hareketli | %3.306 | %3.452 |
| Kredi | ay_sonu_koru | %2.142 | %2.267 |

## Ölçülen eğitim süreleri

| Algoritma | 18 test modeli fit toplamı |
|---|---:|
| LightGBM | 2.873 sn |
| Random Forest | 25.045 sn |
| CatBoost | 11.926 sn |

Tüm koşu: 381.2 sn; parametre araması, önem, kaydetme ve Ocak modelleri dahil.
Süreler tek yerel CPU koşusudur, 2 iş parçacığı kullanılmıştır. V3 ve V4 farklı tarihlerde çalıştırılmıştır.

## Yorumlama sınırları

- Yapısal görünüm bakiyeleri bilinçli dışarıda bırakır; stok bilgisine erişen referanslarla aynı bilgi kümesini kullanmaz.
- Geçmiş bakiyeli görünüm önceki deneydeki aynı 34 alanı korur. Hareket görünümü 15 ek güncel alan ve ay sonu anchor kullanır.
- Ay sonu stok referansı, hareket modelindeki anchor bilgisine erişir. Böylece olası fayda sadece bilinen kapanış stokunu eklemekten kaynaklanıyormuş gibi sunulmaz.
- Gelecek ayın devir/ayrılma olayları girdilere verilmez. Bilinmeyen yeni müşteri ve kaynak hareketleri tahmin hatası oluşturabilir.
- V3 ve V4 test değerleri farklı veri üretimlerine aittir. Aradaki hata farkı gerçek banka performansında iyileşme kanıtı değildir.
- Teste bakılarak yeni parametre taraması veya veri üretimi yapılmadı. Üç dönemlik özellik kararlılığı bu koşuda hesaplanmadı.
- Bu çalışma bir ay ileri tahmin karşılaştırmasıdır. URF/K-Medoids, DEA veya hedef dağıtımı tamamlanmış değildir.
