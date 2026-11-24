# Banka modeli Python kodları

Güncel seçim politikası her coğrafi grupta en az 0.10 doğrulama silhouette
ortalaması arar. Bu pilot koşulu mevcut adaylar sağlamadığı için onaylı ana
model yoktur; yeni atama/DEA başlatılmaz. Önceki sonuçlar tarihsel olarak
korunur. Grup koşulları Ana model seçimi ekranındadır.

Tam şube kapsamı zorunludur: önce `otomatik_kumeleme.py select`, sonra
`dea_verimlilik.py run`, ardından `dea_girdi_kontrol.py run` çalıştırılır.
Ana ve ek girdi senaryoları için ortak boyut sınırı seçimden önce hesaplanır.
Küçük küme veya eksik atama varsa işlem durur; şubeler elenerek devam edilmez.

DEA girdilerinin kalite ve duyarlılık kontrolü:

```bash
.venv_flaml/bin/python python_kodlari/dea_girdi_kontrol.py run
.venv_flaml/bin/python python_kodlari/dea_girdi_kontrol.py verify --run KONTROL_KOSU_KLASORU
python3 python_kodlari/dashboard_veri_hazirla.py
```

Müşteri / müşteri+personel senaryoları aynı emsallerde karşılaştırılır. Veri
kalitesi, kapsam, aynı şubelerde ek girdi etkisi, küçük veri değişimi, tek emsal
çıkarma ve aylık skor hareketleri incelenir. Ana DEA modeli değiştirilmez.
Ayrıntı: [verimlilik_kontrol/README.md](verimlilik_kontrol/README.md).

Güncel aşama: küme içi DEA verimlilik analizi.

```bash
.venv_flaml/bin/python python_kodlari/dea_verimlilik.py run --config python_kodlari/ayarlar/dea_v4.json
.venv_flaml/bin/python python_kodlari/dea_verimlilik.py verify --run DEA_KOSU_KLASORU
python3 python_kodlari/dashboard_veri_hazirla.py
```

Aynı ayın ortalama aktif müşteri sayısı ve üç aylık ortalama bakiyesi, seçilmiş
ana modelin emsal kümelerinde karşılaştırılır. İki aşamalı çıktı odaklı CRS-DEA
ve VRS duyarlılığı birlikte hesaplanır. Kaynak gün ağırlığı, ayrı dual LP,
ürün boşlukları ve emsal izolasyonu bağımsız doğrulanır. Şube bütçesi dağıtılmaz.
Ayrıntı: [verimlilik/README.md](verimlilik/README.md). Son koşu:
`outputs/dea_verimlilik/latest.json`.

Güncel ana kümeleme modeli artık performansa göre seçilir:

```bash
.venv_flaml/bin/python python_kodlari/otomatik_kumeleme.py select --config python_kodlari/ayarlar/otomatik_kumeleme.json
python3 python_kodlari/dashboard_veri_hazirla.py
```

Yöntem + alan oranı + toplam küme sayısı tek ortak aday olarak seçilir. Önce kalite
ve boyut koşulları, sonra ortak puan uygulanır. Uygun aday yoksa ana model seçilmez.
Son karar `outputs/otomatik_kumeleme/latest.json` içindedir. Tek ana yöntemle yeni
ay atamak için **otomatik_kumeleme.py apply** kullanın; eski karşılaştırma apply
komutu bütün yöntemleri döndürür ve güncel ana model seçimini uygulamaz.
Kurallar, aylık akış ve komutlar: [kumeleme_secimi/README.md](kumeleme_secimi/README.md).

Kaynak karşılaştırma gridini üretme — URF başlangıç referansı:

```bash
.venv_flaml/bin/python -m pip install -r python_kodlari/requirements_kumeleme.txt
.venv_flaml/bin/python python_kodlari/kumeleme_karsilastir.py run --config python_kodlari/ayarlar/kumeleme.json
python3 python_kodlari/dashboard_veri_hazirla.py
```

Son doğrulanmış değişken seçiminin aynı alanlarıyla URF + K-Medoids, Gower +
K-Medoids ve K-Prototypes çalışır. Bu kaynak grid eski URF referansını korur;
üzerindeki otomatik_kumeleme.py aşaması güncel ana modeli seçer. Referans
ayda k seçilir; sonraki aylarda sabit atama ve yeniden kümeleme ayrı izlenir.
Tohum ve ortak şube alt örneklemleriyle kararlılık ölçülür. İşlem 3 gösterge için
aynı şube kümelerini üretir; DEA ve hedef dağıtımı içermez. Mevcut çıktı ezilmez.
Son çalışma `outputs/kumeleme_karsilastirma/latest.json` dosyasındadır.

Güncel ayarlar `feature_selection_ratios=[0.25,0.5]`, `bank_cluster_counts=[5,7]`
ve `mtry_ratio=0.25` şeklindedir. Alan seçimi önceki doğrulama önem sırasındaki
orijinal alanlara uygulanır: sayı ceil ile yukarı yuvarlanır. mtry ise RF'ye
verilen kodlama sonrası sütun sayısına göre max(1,floor(P_encoded*0.25)) olur.
Her native model gerçek sütun adlarını, encoded_feature_count ve effective_mtry
değerini saklar; bütün ağaçlarda bu değer doğrulanır.

200 şubelik mevcut kesitte banka genelinde 5 küme = büyük illerde 2 / diğerlerinde 3;
7 küme = 3 / 4. Grup dağılımı referans şube sayılarına göre dinamik hesaplanır.
Her alan oranı × toplam küme sayısı ayrı senaryodur. Senaryoların ortak ölçümü
model alt setiyle değil, ebeveyn şemanın bütün alanlarıyla aynı sabit Gower
uzaklığında yapılır. Kaynak gridin eski referans senaryosu yalnızca URF referans puanına dayanır.
Güncel yöntem/senaryo seçimi otomatik seçim aşamasında ayrı doğrulama aylarıyla
yapılır; kaynak geçmişinin üzerine yazılmaz.

Üst %25/%50 listelerinde kategorik alan kalmadığından mevcut veride K-Prototypes
uygulanamaz. Gizlice başka algoritma kullanılmaz veya oranı bozarak kategori eklenmez.
Önceki tam 22 alan / sqrt çalışması tarihsel klasöründe korunur. Parametre grid'i
tek pakettir; `check`, `report`, `apply` komutları grid kökünü de kabul eder.

```bash
.venv_flaml/bin/python python_kodlari/kumeleme_karsilastir.py check --run KUMELEME_KLASORU
.venv_flaml/bin/python python_kodlari/kumeleme_karsilastir.py apply --run KUMELEME_KLASORU --panel HAZIR_YENI_AY_PANEL.csv --output outputs/yeni_ay_kumeleri
```

apply, dinamik seçimin dondurulmuş şemasıyla hazırlanmış paneli bekler. Orman,
medoid ve prototipler yeniden eğitilmez. Yeni coğrafi grup için referans gerekir.
Parametreler `python_kodlari/ayarlar/kumeleme.json` içindedir. Küme başına 12 üye
yalnızca pilot kontrol sınırıdır; DEA girdi/çıktı boyutlarına göre yeniden değerlendirilir.

Önceki aşama: sözlükten çalışan dinamik kümeleme değişken seçimi.

```bash
.venv_flaml/bin/python python_kodlari/dinamik_feature_selection.py run --config python_kodlari/ayarlar/dinamik_v4.json
python3 python_kodlari/dashboard_veri_hazirla.py
```

Bu komut kalite/korelasyon kontrolü, geçmiş dönemli CatBoost destekli seçim,
bütün model/permütasyon kontrolleri ve rapor/paket üretimini birlikte yapar.
Çıktı tarihli ve hashli bir klasöre yazılır; eskisi ezilmez. Son çalışma
`outputs/dinamik_feature_selection/latest.json` içindedir. Bu iş kümeleme,
DEA veya hedef dağıtımı çalıştırmaz. Önceki 49 alanlı tahmin deneyleri korunur.

Gerçek veri için önce alan sözlüğü taslağı oluşturulur:

```bash
.venv_flaml/bin/python python_kodlari/dinamik_feature_selection.py discover --input gercek_veri.csv --output alan_sozlugu.json --branch SUBE_KODU --period DONEM --group SUBE_GRUBU
```

Alanların rolü, tipi, zamanında bilinirliği ve gerekiyorsa toplulaştırması
sözlükte tanımlanır; tanımsız alanlar otomatik aday değildir. Model kodunda
değişken adlarını/sayısını değiştirmek gerekmez. Gerçek veriye ait sütun adları
yukarıdaki örnektekilerden farklı olabilir; komut argümanlarına kendi adlarını yazın.

Aylık sabit şema uygulaması:

```bash
.venv_flaml/bin/python python_kodlari/dinamik_feature_selection.py apply --schema SECIM_KLASORU/selected_schema.json --input yeni_ay.csv --output outputs/yeni_ay_kumeleme_girdisi
```

Bu komut yeniden fit etmez, gerekli alan kayıpsa durur. Dağılım/yeni kategori
raporu ve yalnızca seçilmiş alanlarla şube girdileri oluşturur. Yeni aday set
çıkarmak için `run` kullanılır; `--baseline eski_selected_schema.json` ile eklenen
ve çıkarılan alanlar karşılaştırılır. Aktif şema otomatik değiştirilmez.
Kurulum, sözlük örnekleri ve veri düzeyleri: [dinamik_secim/README.md](dinamik_secim/README.md).

FLAML AutoML — güncel kural bütün göstergelerde tek algoritma:

```bash
python3 -m venv .venv_flaml
.venv_flaml/bin/python -m pip install -r python_kodlari/requirements_flaml.txt
.venv_flaml/bin/python python_kodlari/flaml_ortak_model.py
.venv_flaml/bin/python python_kodlari/flaml_ortak_dogrula.py
python3 python_kodlari/flaml_ortak_raporla.py
python3 python_kodlari/dashboard_veri_hazirla.py
```

`flaml_ortak_model.py` mevcut FLAML kayıtlarından üç algoritmanın adaylarını
yeniden eğitir. Üç göstergenin banka geneli doğrulama WAPE ortalamasıyla tek
algoritma seçer. Şu an ortak seçim **CatBoost**. Aynı aile tüm gösterge ve şube
gruplarında kullanılır; altı ayrı eğitim/parametre seti vardır. Referanslar ve
eski manuel modeller yalnızca karşılaştırmadır. Güncel çıktı `outputs/flaml_ortak_v4`.
Mevcut çıktı ezilmez; farklı klasör için `--output` verilebilir. Doğrulama ve rapor
betikleri varsayılan sonuç klasörünü kullanır.

`flaml_ortak_dogrula.py` 30 modeli yeniden yükler, ortak algoritma seçimini ve
bütün metrikleri bağımsız hesaplar. `flaml_ortak_raporla.py` rapor/CSV/ZIP üretir.

Önceki, hücreye göre karışık seçim yapan deney tarihsel olarak korunur:

- flaml_pipeline.py: V4'te altı arama yapar. Üç algoritmayı 49 alanla dener;
  doğrulamada FLAML, mevcut modeller ve referanslar arasından kazananı seçer.
- flaml_tahmin.py: kaydedilmiş native paketle arama yapmadan tahmin üretir.
- flaml_dogrula.py: 30 paketi, doğrulama seçimlerini, metrikleri ve deneme kayıtlarını doğrular.
- flaml_raporla.py: sonuç/metodoloji raporu, arama geçmişi CSV ve ZIP üretir.
- flaml_uyumluluk_kontrol.py: yalnızca TRAIN/VALIDATION ile üç algoritmanın entegrasyonunu sınar.

Çıktı `outputs/flaml_v4` içindedir. `--budget` arama başına saniyedir.
Kesilen koşu aynı veri/kod/ayarlarla `--resume` kullanılarak sürdürülür.
Yalnızca bu eski deneyde referans da seçilebilirdi. Ekim–Aralık 2026
daha önce incelendiğinden sonuçlar yeniden kullanılan tarihsel benchmark'tır.
Yeni, el değmemiş son test başarısı iddia edilmez.

V4 model karşılaştırması:

```bash
python3 python_kodlari/v4_egitim_seti_hazirla.py
python python_kodlari/model_karsilastir.py --protocol v4 --source outputs/model_karsilastirma_v4 --output outputs/model_karsilastirma_v4
python python_kodlari/model_karsilastir_dogrula.py --source outputs/model_karsilastirma_v4 --output outputs/model_karsilastirma_v4
python3 python_kodlari/model_v4_raporla.py
python3 python_kodlari/dashboard_veri_hazirla.py
```

Model komutları için Python 3.12 ortamına `outputs/model_karsilastirma_v4/requirements.txt`
sürümlerini yükleyin. V4 modeli 25/34/49 alanlı üç görünüm içerir. 49 alanlı
görünümde anchor ay sonu stokudur; aynı stok bilgisine erişen referans da ölçülür.
V4 sonuçları ayrı klasördedir. V3 modelleri ve veri dosyaları korunur.

- v4_egitim_seti_hazirla.py: dinamik şube kitlesi ve gün ağırlıklı gerçekleşmelerle zaman ayrımını kurar.
- model_v4_protokol.py: üç algoritmanın ortak alanlarını, anchor ve önem kurallarını tanımlar.
- model_v4_raporla.py: V4 metodolojisi, sonuç tablosu ve model paketini oluşturur.

V4 hareketli test verisi:

- test_verisi_v4_uret.py: 26 ayın tamamını müşteri devirleri, yeni müşteri,
  kaynak giriş/çıkışı, ürün değişimleri ve tam ayrılmalarla yeniden üretir.
- test_verisi_v4_dogrula.py: olay defterini baştan okuyup günlük ortalamaları,
  şube aidiyetlerini, demografiyi ve stok mutabakatını bağımsız doğrular.
- test_verisi_v4_paketle.py: doğrulanmış veri için Excel özeti girdisi ve ZIP oluşturur.

```bash
python3 python_kodlari/test_verisi_v4_uret.py
python3 python_kodlari/test_verisi_v4_dogrula.py
python3 python_kodlari/test_verisi_v4_paketle.py --prepare-summary
node .work/build_v4_summary.mjs
python3 python_kodlari/test_verisi_v4_paketle.py
```

Üretim ve doğrulama Python standart kütüphanesini kullanır. Kaynak V3 dosyaları
korunur; V4 yeni klasöre yazılır. Eski `pipeline.py` veri hazırlığı sabit müşteri–şube
eşleştirmesi yaptığı için V4'e doğrudan uygulanmamalıdır; yukarıdaki V4 hazırlığı
kullanılır. V4'te aylık şube özelliklerini `sube_model_girdileri_aylik.csv`,
gün ağırlıklı şube bakiyelerini `musteri_sube_aylik.csv` sağlar.

- pipeline.py: şube-ay verisini hazırlar; eğitim/doğrulama/test setlerini oluşturur;
  LightGBM modellerini eğitir; değişken önemlerini ve tahminleri kaydeder.
- tahmin.py: kaydedilmiş native LightGBM modelinden tahmin üretir.
- grafikler.py: model sonuçlarının grafiklerini ve sonuç özetini oluşturur.
- dogrula.py: veri zamanlamasını, hedefleri, kaydedilen modelleri ve metrikleri doğrular.
- requirements.txt: çalışmada kullanılan paket sürümleri.
- model_karsilastir.py: aynı veri ayrımlarıyla üç algoritmayı eğitir. V3 protokolü
  36, V4 protokolü 54 model üretir; hata, süre, önem ve tahminleri kaydeder.
- tahmin_karsilastir.py: üç algoritmanın kaydedilmiş modellerinden tahmin üretir.
- model_karsilastir_dogrula.py: protokoldeki bütün model dosyalarını yeniden yükler; metrikleri
  ve ortak satır anahtarlarını bağımsız olarak kontrol eder.
- model_karsilastirma_raporla.py: karşılaştırma metodolojisi, sonuç özeti ve ZIP üretir.
- dashboard_veri_hazirla.py: tamamlanmış raporlardan yerel dashboard verisini günceller.

Üç algoritmanın karşılaştırmasını çalıştırmak için:

```bash
.work/ml_env/bin/python python_kodlari/model_karsilastir.py
.work/ml_env/bin/python python_kodlari/model_karsilastir_dogrula.py
.work/ml_env/bin/python python_kodlari/model_karsilastirma_raporla.py
.work/ml_env/bin/python python_kodlari/dashboard_veri_hazirla.py
```

Yeni sonuçlar `outputs/model_karsilastirma` klasörüne yazılır. İlk LightGBM
deneyi korunur. Dashboard'da “Algoritma karşılaştırması” sayfasını açın.

Komutları hedef proje klasöründe çalıştırın. Önceki V3 veri ve model klasörleri
mevcut konumlarında kullanılır. Özgün paket içindeki kod dosyaları korunmuştur.

Hazır yerel ortam ile yalnızca veri hazırlama:

```bash
.work/ml_env/bin/python python_kodlari/pipeline.py --source outputs/banka_test_verisi_v3 --output outputs/lightgbm_ilk_model --prepare-only
```

Modelleri yeniden eğitme:

```bash
.work/ml_env/bin/python python_kodlari/pipeline.py --source outputs/banka_test_verisi_v3 --output outputs/lightgbm_ilk_model
```

Sonuçları doğrulama:

```bash
.work/ml_env/bin/python python_kodlari/dogrula.py --source outputs/banka_test_verisi_v3 --output outputs/lightgbm_ilk_model
```

Grafikleri oluşturma:

```bash
.work/ml_env/bin/python python_kodlari/grafikler.py --output outputs/lightgbm_ilk_model
```

Yeni ortam kurmak için Python 3.12 ile bir sanal ortam oluşturup requirements.txt
paketlerini yükleyin. macOS'ta LightGBM için libomp gerekir; mevcut ortamda vardır.
