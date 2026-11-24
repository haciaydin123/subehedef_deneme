# Ana kümeleme modeli seçimi

Aday, yöntem + orijinal alan seçim oranı + banka toplam küme sayısı birleşimidir. URF korunması zorunlu değildir. Tek ortak aday seçilir; coğrafi grupların kendi referans modelleri korunur.

Referans ay modelleri yeniden eğitilmez. Seçimde yalnızca config.json içindeki validation_periods ve referans ay tohum/alt örneklem tekrarları kullanılır. Evaluation ayları karar dosyası yazıldıktan sonra raporlanır; seçime, elemelere veya puana girmez. Bu çalışmanın ayları daha önce incelendiğinden evaluation dokunulmamış test değildir. Değişken seçimi ve kaynak veri geçmişi önceki çalışmaya aittir; yeni uçtan uca test iddiası yoktur.

Önce uygulanabilirlik, tam grup/ay/tekrar kaydı, yakınsama, referans ve sabit doğrulama atamalarında küme boyutu koşulları aranır. Sonra banka ortalama silhouette, aylık yeniden kümeleme ARI, referans alt örneklem ARI, referans tohum ARI ve en zayıf grubun ortalama yeniden kümeleme ARI eşikleri uygulanır. Eşikler pilot politika tercihleridir; evrensel DEA yeterlilik ölçütü değildir.

Her ay eşit ağırlıklıdır. Coğrafi gruplar referans şube sayılarına göre ağırlıklanır. Ortak silhouette bütün adaylarda aynı tam 22 alanlı Gower geometrisinde ölçülmüştür; Gower yöntemini avantajlı kılabilir. İş profilleri ve sonraki gerçek veri dönemleriyle inceleme gereklidir.

Puan = 0.30 × silhouette + 0.30 × yeniden kümeleme ARI + 0.25 × alt örneklem ARI + 0.15 × tohum ARI. Her bileşen [0,1] aralığına kırpılır. Yapılandırılabilir ağırlıklar ham ölçeklerde bir fayda puanıdır; eşit yüzdeli katkı veya doğruluk yüzdesi değildir. Sabit atama ARI puana katılmaz. Eğitim süresi seçim ölçütü değildir; mevcut adaylar zaten küçük ve önceden eğitilmiştir.

Koşulları geçen en yüksek puanlı aday seçilir. Eşitlikte daha az küme, daha az alan, sonra aday kodu kullanılır. Mevcut aday hâlâ uygunsa minimum_gain kadar artış olmadan değiştirilmez. Hiçbir aday uygun değilse selected=null olur ve ana modelle atama engellenir. Evaluation'daki bozulma seçimi geriye dönük değiştirmez; ayrı inceleme sinyali olarak gösterilir.

apply komutu hazırlanmış şube-ay panelini alır. Seçilmiş tek yöntemin bölgesel native modellerini kullanır; yeniden fit etmez. Ham müşteri verisi önce dinamik veri sözlüğündeki aynı hazırlama adımlarından geçirilmelidir. Yeni aylar için önce karşılaştırma paneli/grid güncellenir, sonra seçim yapılandırmasındaki dönemler ilerletilir; değerlendirme ayı seçime alınırsa yeni bir ileri dönem ayrıca ayrılır.

## Bütün şubelerin kapsanması

DEA alanları ve girdi kontrol senaryoları seçimden önce okunur. Gerekli küme alt sınırı bu senaryoların max(m×s,3(m+s)) değerlerinin en büyüğü ve tanımlı pilot alt sınırın maksimumudur: bu koşuda 15. Alanlar değişirse seçim yeniden çalıştırılır. Referans/doğrulama boyut koşullarını geçmeyen aday elenir. Seçim dondurulduktan sonra bütün kayıtlı ayların gerçek atamaları kaynak şube evreniyle birebir karşılaştırılır. Sonraki ayda kapsama bozulursa başka aday o aya bakılarak seçilmez; aktivasyon durur. Yeni aya apply işlemi de aynı kontrolü yapar. Bu ön kontrol istatistiksel yeterlilik garantisi değildir.

## Coğrafi grupların ayrı kalite koşulu

Her coğrafi grubun doğrulama dönemindeki sabit atama silhouette ortalaması ayrı hesaplanır. Bütün gruplar aynı 0.10 alt sınırını geçmelidir. En düşük grup ortalaması bu sınırın altındaysa banka ortalaması veya birleşik puan yüksek olsa da aday elenir. Eksik grup/ay ölçüsü de geçerli sayılmaz. Eşik yapılandırılabilir bir pilot politikasıdır; evrensel veya bu veriden optimize edilmiş eşik değildir. İki grup için ayrı algoritma seçilmez. Kontrol yalnız config.validation_periods aylarını kullanır; evaluation ayları bu koşula ve aday sırasına girmez. Uygun aday yoksa seçilen model null kalır; önceki model yeni politika kapsamında otomatik geri alınmaz. Önceki DEA sonuçları yalnız tarihsel olarak korunur.
