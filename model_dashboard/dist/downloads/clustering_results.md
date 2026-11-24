# Değişken oranı ve kodlama sonrası mtry karşılaştırması

Ana yöntem URF + K-Medoids. mtry oranı %25, payda RF’ye verilen kodlama sonrası sütun sayısı.
Değişken seçim oranı ise önceki doğrulama önem sırasındaki orijinal alan sayısına uygulanır. Alan sayısı yukarı, mtry aşağı yuvarlanır; mtry en az 1 olur.
Üst aday havuzu 22 alan. Referans 2026-09; aylık izleme ['2026-10', '2026-11', '2026-12'].

| Senaryo | Alan oranı | Ham alan | Toplam küme | URF kodlanmış sütun | URF mtry |
|---|---:|---:|---:|---|---|
| top25_k5 | %25 | 6 | 5 | [6] | [1] |
| top25_k7 | %25 | 6 | 7 | [6] | [1] |
| top50_k5 | %50 | 11 | 5 | [11] | [2] |
| top50_k7 | %50 | 11 | 7 | [11] | [2] |

## Aynı tam alan setiyle ortak referans ölçümü

| Senaryo | Yöntem | Ortak silhouette | Boyut kuralı | Aylık yeniden kümeleme ARI |
|---|---|---:|---|---:|
| top25_k5 | URF + K-Medoids | 0.109 | True | 0.434 |
| top25_k5 | Gower + K-Medoids | 0.129 | True | 0.757 |
| top25_k5 | K-Prototypes | — | Uygulanamaz: karma veri yok | — |
| top25_k7 | URF + K-Medoids | 0.069 | True | 0.560 |
| top25_k7 | Gower + K-Medoids | 0.140 | True | 0.907 |
| top25_k7 | K-Prototypes | — | Uygulanamaz: karma veri yok | — |
| top50_k5 | URF + K-Medoids | 0.111 | True | 0.345 |
| top50_k5 | Gower + K-Medoids | 0.138 | True | 0.742 |
| top50_k5 | K-Prototypes | — | Uygulanamaz: karma veri yok | — |
| top50_k7 | URF + K-Medoids | 0.051 | True | 0.461 |
| top50_k7 | Gower + K-Medoids | 0.118 | True | 0.654 |
| top50_k7 | K-Prototypes | — | Uygulanamaz: karma veri yok | — |

**Referans döneminde URF ile seçilen ortak senaryo: top50_k5.**
Bu seçim üç göstergede ve bütün coğrafi gruplarda aynı alan oranı/toplam küme senaryosunu kullanır. Alternatif yönteme otomatik geçiş yoktur.
Bankadaki ortak skor, grup silhouette puanlarının referans şube sayısıyla ağırlıklı ortalamasıdır. Gruplar arası mesafeyi içeren tek bir banka silhouette puanı değildir.
Senaryoların silhouette değerlendirmesi, modelin daha az alan kullanmasından bağımsız olarak ebeveyn şemanın bütün alanlarıyla aynı sabit Gower mesafesine dayanır.
K-Prototypes iki veri tipi de gerektirir. Seçilen üst alanlar yalnızca sayısalsa yöntem uygulanamaz; gizlice başka algoritma kullanılmaz.
Küme başına minimum şube sayısı pilot kuraldır; DEA yeterlilik kanıtı değildir. Veriler sentetiktir. DEA ve hedef dağıtımı çalıştırılmadı.
