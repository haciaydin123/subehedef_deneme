# Ortak FLAML algoritması protokolü

Güncel kural: bütün göstergelerde ve iki coğrafi grupta aynı algoritma ailesi.
Gösterge veya grup bazında LightGBM/CatBoost/Random Forest/referans geçişi yoktur.
Tek aile kullanılır; farklı hedefler için altı ayrı eğitim ve parametre seti vardır.
Tek bir çok çıktılı model nesnesi oluşturulmamıştır.

## Veri ve arama kayıtları

V4'ün 3.400 eğitim, 600 doğrulama, 600 test şube-ay satırı değişmedi. Eğitim
Şubat 2025–Haziran 2026, doğrulama Temmuz–Eylül, tarihsel test Ekim–Aralık 2026.
49 alanlı hareket görünümü ve ay sonu stokuna göre fark hedefi kullanılır.
Tahmin max(0, t ayı kapanış stoku + öğrenilen fark) olarak hesaplanır.

Önceki altı FLAML araması ve 15.247 denemesi yeniden kullanılır; yeni arama
yapılmamıştır. Her hücrede her algoritmanın en düşük kayıtlı doğrulama MAE
konfigürasyonu alınır: 6 hücre × 3 algoritma = 18 aday.
Bu konfigürasyonlar tüm TRAIN üzerinde yeniden eğitilir. Yeniden eğitimde
arama süresinin ağaçları kesmesi kaldırılır; bu nedenle kayıtlı trial kaybı ile
yeniden eğitilen kayıp farklı olabilir. Karar yeniden hesaplanan tahminlere dayanır.
Arama, algoritmalara farklı sayıda deneme ayırmıştır; eşit hesaplama bütçesi
verilmiş bağımsız algoritma yarışması olduğu iddia edilmez.

CatBoost gizli iç doğrulama bölmesi yapmaz (use_best_model=False).
LightGBM/CatBoost native kategorik, FLAML RF kategori kodları kullanır.
Önceki manuel RF one-hot kullandığından fark yalnızca ayar aramasına atfedilmez.

## Tek seçim

Her algoritma için önce iki şube grubunun doğrulama tahminleri birleştirilir.
Her gösterge için WAPE = toplam mutlak hata / toplam mutlak gerçek bakiye.
Seçim skoru = (vadesiz WAPE + vadeli WAPE + kredi WAPE) / 3.
En düşük skorlu tek aile bütün hücrelerde kullanılır. Tam eşitlikte algoritma
kodu alfabetik sıralanır. Referanslar ve eski manuel modeller yalnızca ölçüm içindir.

Karar ortak_model_secimi.json dosyasına TEST değerlendirmesinden önce yazılır.
Karardan sonra seçilen ailenin altı modeli TRAIN+VALIDATION ile evaluation için,
tüm etiketli aylarla Ocak 2027 tahmini için yeniden eğitilir.
Production modeliyle test skoru hesaplanmaz. Ocak girdileri etiketsizdir.

## Kayıt ve sınırlar

18 aday validation + 6 ortak evaluation + 6 ortak production = 30 model paketi.
Seçilen sistem ile FLAML adayı aynı tahminlerdir; karışık yöntem yönlendirmesi yoktur.
Önceki deney outputs/flaml_v4 altında değiştirilmeden korunur.
Arama eğrileri önceki denemelerin seçilen algoritmaya ait doğrulama noktalarıdır;
yeniden eğitilen modelin yeni bir araması değildir. Tüm denemeler CSV'de korunur.

Ekim–Aralık 2026 testi daha önce görüldü; sonuçlar tekrar kullanılan tarihsel
karşılaştırmadır. Sentetik veri gerçek banka başarısını kanıtlamaz.
Özellik önemleri bu değişiklikte yeniden hesaplanmadı. DEA/hedef dağıtımı sonraki aşamadır.

## Çalıştırma

Proje kökünde requirements_flaml.txt bağımlılıkları kurulu Python ile:

```bash
python python_kodlari/flaml_ortak_model.py
python python_kodlari/flaml_ortak_dogrula.py
python python_kodlari/flaml_ortak_raporla.py
python python_kodlari/dashboard_veri_hazirla.py
```

İlk komut önceki V4 ve FLAML arama kayıtlarını gerektirir. Mevcut sonuç klasörünü
ezmez; yeni koşu için --output ile farklı klasör verilebilir. Doğrulama ve rapor
komutlarının varsayılanı outputs/flaml_ortak_v4 klasörüdür.
Tahmin flaml_tahmin.py ile herhangi bir __common_production.joblib dosyasından yapılır.
