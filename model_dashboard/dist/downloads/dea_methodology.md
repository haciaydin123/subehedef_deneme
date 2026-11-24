# Küme içi verimlilik metodolojisi

Dönem 2026-12. Her şube sadece seçilmiş ana yöntemin aynı ay, aynı coğrafi grup ve aynı kümesindeki şubelerle karşılaştırılır. Şube, DEA karar birimidir; müşteriler tek tek DEA birimi değildir. Girdi/çıktı rolleri feature importance sırasından türetilmez; ayar sözlüğünde açık tanımlanır.

Girdi: Aylık ortalama aktif müşteri sayısı (ortalama_aktif_musteri). Varsayılan müşteri girdisi müşteri-gün / ayın bütün takvim günleridir; ay sonu müşteri sayısı kullanılmaz. Bu tanım müşteri tabanına göre göreli bilanço üretimini ölçer. Müşteri sayısı idari olarak azaltılacak bir maliyet değildir; çıktı odaklı modelde sabit tutulan maruziyet/potansiyel ölçüsüdür. Maliyet, kârlılık veya risk ayarlı verimlilik iddiası yoktur.

Çıktılar aynı ayın TL vadesiz, TL vadeli ve kredi aylık ortalama bakiyeleridir. Pilot hizmet/üretim yaklaşımıyla mevduat ve kredi birlikte çıktı sayılır. Mevduatı fonlama girdisi sayan aracılık yaklaşımı seçilmedi. V4 kredi toplamı KMH dahil ihtiyaç/konut/taşıt alt ürünlerini içerir; kart borcu bu toplamın dışında tutulur. Kaynak alan eşleştirmesi config.json içindedir.

## Matematik

Çıktı odaklı CRS için max φ; Xλ ≤ x₀, Yλ ≥ φy₀, λ ≥ 0. CRS'de toplam λ = 1 koşulu yoktur. Şubenin kendisi referans kümesinde olduğundan φ ≥ 1 sağlanır. Verimlilik E = 1/φ, ortak radyal artış mesafesi φ−1. Örneğin E=0.80 ise ortak radyal mesafe %25'tir; %20 değildir. Sabit getiri varsayımı, girdilerle çıktıların orantılı ölçeklenebileceğini kabul eder.

İkinci doğrusal program φ optimumunu sabitler, medyanla ölçeklenmiş girdi/çıktı boşlukları toplamını maksimize eder. Projeksiyon Yλ = φy₀ + s⁺; ürün bazındaki artış bu projeksiyon ile gerçekleşme arasındaki farktır. İlk aşamanın ortak çarpanı üç göstergede aynıdır; ikinci aşamada ürün boşlukları farklılaşabilir. λ ve projeksiyonlar alternatif optimumlarda tekil olmayabilir. CRS λ katsayıları toplamı 1 olmak zorunda olmadığından, emsaller basit müşteri/portföy transfer planı veya aritmetik ortalama değildir.

Radyal skor 1 olsa bile pozitif boşluk bulunabilir; bu durum radially_efficient_with_slack olarak ayrıca işaretlenir. Yalnız skor 1 ve normalize toplam boşluk tolerans altında ise strongly_efficient denir. Esnek DEA ağırlıkları bir çıktıyı fiilen çok az ağırlıklandırabilir; skor 1 bütün ürünlerde liderlik anlamına gelmez. İş önceliklerine göre ağırlık kısıtları bu ilk koşuda eklenmedi.

CRS ana modelin yanında aynı şube/emsallerle VRS duyarlılığı hesaplanır (ek koşul: toplam λ = 1). CRS skoru VRS skorundan büyük olamaz; E_CRS/E_VRS çıktı yönelimli ölçek verimliliği olarak raporlanır. VRS otomatik ana modele çevrilmez.

## Kontroller ve kullanım sınırı

n ≥ max(m×s, 3(m+s)) bir ayırt etme ön kontrolü olarak uygulanır. Bu literatürdeki pratik kural yeterlilik veya istatistiksel güven garantisi değildir. 1 girdi / 3 çıktı için 12, 2 girdi / 3 çıktı için 15 emsal gerekir. Küçük kümeler oluşursa bütün DEA işlemi durur; bazı şubeleri eleyip kısmi skor yayımlanmaz. Kümeleme seçimi, ana girdi ve tanımlı girdi senaryolarının tamamının boyut gereksinimini önceden karşılamalıdır. Mevcut sözleşme bu senaryolar için en az 15 şube gerektirir. Girdi/çıktı tanımı değişirse kümeleme seçimi yeniden çalıştırılır. Geçmiş aylar ek bağımsız şubeymiş gibi sayılmaz. Geçersiz/eksik/negatif değerler otomatik doldurulmaz. Pozitif girdiler, negatif olmayan ve her şubede en az bir pozitif çıktı gerekir; tamamı sıfır bir çıktı varsa tanımın düzeltilmesi istenir. Sıfır gerçekleşmede yüzde artış null, tutar projeksiyonu açık kalır.

Şubeler aynı ay gözlemleriyle sınır oluşturur; bu bir tahmin modeli veya dokunulmamış test başarısı değildir. Aykırı üretimler DEA sınırını etkileyebilir. %100'ü aşan radyal mesafeler ve radyal etkin şube payı %50'yi aşan kümeler inceleme işareti alır; veri kırpılmaz. Bu bayraklar politika tercihleridir. Farklı kümelerin farklı sınırları olduğundan banka ortalama skoru betimleyicidir; bütün şubeler için tek ortak verimlilik sıralaması iddiası yoktur.

Projeksiyonlar ve artış mesafeleri gelecek ay büyüme tahmini, müşteri talebi veya banka hedefi değildir. Ocak 2027 bütçesindeki %15/%10/%5 oranlar bu aşamada şubelere dağıtılmadı. Risk, mevduat maliyeti, vade yapısı ve uygulanabilirlik sonraki iş incelemesinde ele alınmalıdır. Finansal veri sentetiktir.

Doğrulama her CRS skorunu bağımsız çarpan/dual LP ile yeniden çözer, ilk aşama primal/dual fizibilite ve amaç eşitliğini denetler. İkinci aşama emsal katsayılarıyla girdi/çıktı projeksiyonları ve boşluklar tekrar hesaplanır. VRS LP ayrı kurularak kontrol edilir. Kaynak müşteri-gün/gün dönüşümü, alan eşleştirmeleri ve bir şubenin başka kümeden emsal almaması ayrıca doğrulanır.

## Kaynaklar

- [Output-oriented CCR formulation (primary research)](https://www.scielo.br/j/pope/a/HGkM4mcBhxfhpLJ8t3H5nRd/?lang=en)
- [DEA variable selection and sample-size heuristic (primary research)](https://www.mdpi.com/2073-8994/13/7/1155)
- [SciPy HiGHS linear programming documentation](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html)
