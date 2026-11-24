# Küme içi DEA verimlilik analizi

Proje kökünden:

```bash
python python_kodlari/dea_verimlilik.py run --config python_kodlari/ayarlar/dea_v4.json
python python_kodlari/dea_verimlilik.py verify --run outputs/dea_verimlilik/KOSU_KODU
python python_kodlari/dashboard_veri_hazirla.py
```

Paketler: `python -m pip install -r python_kodlari/requirements_dea.txt`.
Son doğrulanmış koşu `outputs/dea_verimlilik/latest.json` içindedir. Var olan koşu
ezilmez. Farklı bir çıktı için `--output`, referans ayı için `--period YYYY-MM`,
başka bir doğrulanmış ana küme seçimi için `--selection KLASOR` verilir.

Varsayılan girdi `musteri_gun / takvim_gunu` ile aylık ortalama aktif müşteridir.
Üç çıktı TL vadesiz, TL vadeli ve kredi aylık ortalama bakiyesidir. Alan adları,
girdi/çıktı listeleri ve veri filtreleri JSON sözlüğünden gelir. Farklı kaynak
kolonları için sözlüğü değiştirin; model kodunu değiştirmek gerekmez. FTE girdisi
istenirse sözlüğe `personel_fte_sentetik` identity dönüşümüyle eklenebilir veya
tek girdi olarak tanımlanabilir. İki girdi / üç çıktıda 15 şubelik ön kontrol
sınırı vardır. Küme seçimi bütün tanımlı senaryoların en büyük gereksinimine
göre yapılır. Küçük küme varsa bütün DEA işlemi durur; şube elenmez.

Ana model çıktı odaklı iki aşamalı CRS-DEA'dır; aynı emsallerde VRS skoru
duyarlılık olarak saklanır. İlk aşama ortak çarpanı, ikinci aşama ilave ürün
boşluklarını üretir. Şube emsal katsayıları ve primal/dual LP sertifikaları
kaydedilir. `verify` ayrı dual LP ve ayrı ikinci aşama LP kurarak yeniden çözer;
kaynak alan dönüşümleri ve küme dışına emsal alınmaması da kontrol edilir.

Geçersiz/eksik/negatif değerler doldurulmaz. Girdiler kesin pozitif olmalıdır;
çıktılar negatif olamaz ve bir şubenin bütün çıktıları sıfır olamaz. Bir ürün
bütün emsallerde sıfırsa tanımın düzeltilmesi istenir. Sıfır gerçekleşmede yüzde
artış tanımsızdır; yapay epsilon veya sahte büyüme yüzdesi eklenmez.

Her ay güncel müşteri devirlerini ve ürün ortalamalarını kaynak panele yansıtın,
ana kümeleme seçimini/atamalarını güncelleyin, sonra DEA'yı aynı ayda yeniden
çalıştırın. Yeni aya ait onaylı ana küme ataması bulunmadan DEA çalışmaz. Müşteri
devirlerinin etkisi hem küme girdilerine hem müşteri-gün ve bakiyelere yansır.

Çıktılar `scores`, `potentials`, `peers`, `clusters` dosyaları; tam kanıtlar
`certificates.json`, kaynak/DEA kesitleri ve hashli manifest içindedir. ZIP'te
`code/dea_verimlilik.py verify --run PAKET_KLASORU` proje dışında çalışır.

Skorlar farklı emsal kümeleri arasında tek banka sıralaması değildir. DEA
projeksiyonları gelecek ay tahmini veya bütçe hedefi değildir. Bankanın
%15/%10/%5 büyüme bütçesine şube payı dağıtımı sonraki ayrı aşamadır.

Üretim kuralı `insufficient_cluster_policy=error` olarak sabittir. Önce
`otomatik_kumeleme.py select` çalışır; `dea_coverage_contract.json` ana ve ek
girdileri kaydeder. DEA alanları bu sözleşmeyle eşleşmiyorsa yeniden seçim
istenir. Kısmi/null skorlu yeni koşu yayımlanmaz. Tarihsel koşular korunur.
