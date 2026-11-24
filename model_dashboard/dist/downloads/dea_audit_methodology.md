# DEA girdi kontrolleri

Ana amaç müşteri tabanına göre üretimdir. Karşılaştırma müşteri girdisine personel eklenmesinin etkisini sınar; farklı amaçtaki personel başına üretim modeli otomatik aday değildir. Çıktılar ve Gower/PAM kümeleri bütün senaryolarda aynıdır. Ana DEA modeli veya küme atamaları değiştirilmedi.

İncelenen aylar daha önce kullanılan sentetik geçmiş verilerdir. Her ayın kendi DEA sınırı yeniden hesaplandı; yeni dokunulmamış test veya gelecek ay tahmini iddiası yoktur. Personel kapasitesi sentetiktir ve bütün şubelerde bütün aylar sabittir; gerçek personel devirleri/iş yükü değişimleri sınanamaz.

## Veri ve kapsam

5200 şube-ay, 200 şube, 26 ay kaynak kalite kontrolünde incelendi. Eksik, sayısal olmayan, negatif ve sıfır değerler ayrı raporlanır. DEA karşılaştırması ana modelin kayıtlı atamalarının bulunduğu aylarda yapılır. Müşteri girdisi müşteri-gün / takvim günüdür.

Bir girdi ve üç çıktı için mevcut ön kontrol sınırı 12, iki girdi ve üç çıktı için 15 şubedir. Bu bir politika olarak uygulanan literatür pratik kuralıdır; yeterlilik teoremi değildir. Küçük küme varsa işlem durur; bazı şubelere null skor verilip kalanlarla devam edilmez. Kümeleme adayları ana girdi ve tanımlı ek girdilerin boyut gereksinimlerini önceden karşılamalıdır. Bu koşul kalite ölçütleriyle birlikte uygulanır; tek amaç şube sayısına uygun kümeler bulmak değildir.

2026-12 ayında müşteri senaryosu 200, müşteri+personel senaryosu 200 şubeyi kapsar. Skor farkı yalnızca aynı 200 şubede karşılaştırılır. Bu ortak örneklemde ortalamalar 0.728611 ve 0.763777. Fark performans kazanımı değildir: iç içe DEA girdileri skoru mekanik olarak artırabilir. Yüksek skor seçim hedefi değildir.

## Ek bilgi ve duyarlılık

Pearson/Spearman hem banka genelinde hem ayrı kümelerde hesaplanır. Korelasyon tek başına girdinin gereksizliğini veya nedensel katkısını kanıtlamaz. Ek girdinin tek bir optimal dual çözümdeki normalize katkı payı raporlanır; alternatif optimum ağırlıkları olabileceğinden bu pay yapısal önem veya tekil gerçek katkı değildir.

Son ayda kümeler sabit tutularak 20 tekrarda girdiler ve çıktılar bağımsız, göreli ±%1 aralığında bozulur. Ortak alanlara iki senaryoda aynı rassal değişim verilir. Sıfır çıktı sıfır kalır. Bu deney ölçüm hatası için yerel stres testidir; bootstrap, örneklem güven aralığı veya kanıtlanmış maksimum kararlılık yarıçapı değildir.

Her kümeden sırayla tek bir emsal çıkarılır; yalnız kalan şubeler yeniden değerlendirilir. Değerlendirilen şube kendi emsal setinde kalır: super-efficiency hesaplanmaz. Kalan bağımsız şube sayısı ön kontrolü sağlamıyorsa deney yapılmaz. Emsal çıkarınca skorun yükselmesi diğer şubenin hatalı olduğunu göstermez; sınırın o gözleme duyarlılığını gösterir. En yüksek etkiye sahip emsaller ayrıca iş incelemesi gerektirir.

Aylık değişimler ayrı raporlanır. Aynı kümede kalan ve seçili girdi/çıktılarının tamamı en fazla %5 değişen şubeler ayrıca gösterilir. Bu gözlemlenen profil kontrolüdür; tam müşteri karmasının sabitliği iddiası yoktur. Müşteri devirleri ve gerçek üretim değişimleri skor değişikliğine yol açabilir; aylık oynama tek başına model hatası sayılmaz. Sıra ilişkileri yalnız aynı küme içinde hesaplanır.

## Öneri kuralı

Kapsama oranı, veri kalitesi ve duyarlılık raporlanır. Ana amaç müşteri tabanı olduğundan mevcut girdiyi koruma önerisi bu iki anlamlı senaryo ile sınırlıdır; bütün olası girdiler arasında optimal seçim değildir. Mevcut bütün şubeler için tek ortak politika hedeflendiğinden kapsama alt sınırı %100'dür. Küme bazında yerel stresin p95 mutlak skor farkı 0.03, tek emsal çıkarmada maksimum fark 0.10 üstündeyse inceleme işareti konur. Bunlar önceden tanımlanan pilot eşiklerdir, evrensel doğruluk eşikleri değildir. Skor farkları yüzde artış değil [0,1] skor ölçeğinde puandır.

Öneri: {"recommended_scenario": "customer", "status": "baseline_with_review", "review_reasons": ["influential_peer_review"], "augmented_coverage": 1.0, "augmented_withheld": 0, "augmented_global_eligible": true, "main_model_changed": false, "higher_efficiency_is_selection_objective": false, "business_roles_inferred_from_data": false, "scope": "customer_base_relative_production", "real_staff_changes_tested": false, "all_results_reused_historical_synthetic": true}

## Kaynaklar

- [DEA değişken seçimi ve boyut etkisi](https://www.sciencedirect.com/science/article/pii/S0377221711006011)
- [DEA emsal setine duyarlılık](https://www.sciencedirect.com/science/article/pii/0377221795000542)
