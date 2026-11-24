# FLAML — bütün göstergelerde tek algoritma

**Ortak seçilen algoritma: CatBoost.**
TL vadesiz, TL vadeli ve kredide; İstanbul/Ankara/İzmir ve diğer illerde aynı algoritma kullanılır.
Tek algoritma ailesi seçilir. Her gösterge–grup için ayrı eğitim/parametre vardır; tek bir çok çıktılı model nesnesi değildir.

## Yalnızca doğrulamaya dayalı ortak seçim

| Algoritma | TL vadesiz WAPE | TL vadeli WAPE | Kredi WAPE | Eşit ağırlıklı ortalama |
|---|---:|---:|---:|---:|
| LightGBM | %2.930 | %3.319 | %2.119 | %2.790 |
| Random Forest | %3.044 | %3.336 | %2.176 | %2.852 |
| CatBoost | %2.883 | %3.304 | %2.161 | %2.783 |

Seçim Temmuz–Eylül 2026 üzerinde yapılır. Önce şube grupları birleştirilerek gösterge WAPE hesaplanır; ardından üç gösterge eşit ağırlıkla ortalanır. Basit referanslar ve eski manuel modeller seçime katılmaz.

## Yeniden kullanılan tarihsel test

| Gösterge | Ortak model WAPE | Ay sonu referansı WAPE |
|---|---:|---:|
| TL vadesiz | %2.939 | %2.950 |
| TL vadeli | %3.470 | %3.426 |
| Kredi | %2.300 | %2.267 |

Ekim–Aralık 2026 daha önce incelendi. Bu test tekrar kullanılan karşılaştırmadır; yeni bağımsız test değildir. Finansallar sentetiktir.

30 model paketi yeniden yüklenerek doğrulandı. Ortak seçim yalnızca doğrulama tahminlerinden bağımsız hesaplandı. Önceki karışık model sonuçları outputs/flaml_v4 altında tarihsel olarak korunur.
