# Jupyter notebook rehberi

Önce **[00_baslangic.ipynb](00_baslangic.ipynb)** dosyasını açın.
Projenin tamamını indirin; notebook’lar `python_kodlari/` modüllerini kullanır.
`.py` dosyalarını ayrı hesaplama hücrelerine kopyalamak yerine aynı iş kuralları
notebook’tan çağrılır. Her notebook bağımsız kernel’de açılabilir.

| Notebook | İçerik |
| --- | --- |
| [00 · Başlangıç](00_baslangic.ipynb) | Kernel, paketler, çalışma sırası, mevcut karar |
| [01 · Veri](01_veri_ve_egitim_setleri.ipynb) | Şube paneli, kalite, bakiye grafiği, tarih ayrımı, eğitim seti hazırlama |
| [02 · Modeller](02_modeller_ve_flaml.ipynb) | LightGBM / Random Forest / CatBoost, FLAML, tek ortak algoritma, kayıtlı modelden tahmin |
| [03 · Değişken seçimi](03_dinamik_degisken_secimi.ipynb) | Dinamik alan sözlüğü, CatBoost seçimi, permütasyon katkıları, donmuş şema |
| [04 · Kümeleme](04_kumeleme_ve_model_secimi.ipynb) | URF / Gower / K-Prototypes gridi, ortak aday seçimi, coğrafi kalite ve tam kapsam |
| [05 · DEA](05_dea_ve_girdi_kontrolleri.ipynb) | Verimlilik, ürün mesafeleri, emsaller, müşteri / müşteri+personel kontrolleri |
| [06 · Yeni ay](06_yeni_ay_ve_dashboard.ipynb) | Şema uygulama, donmuş modelle atama, drift, dashboard |

## Çalıştırma

Python 3.12 ortamında proje kökünden:

```bash
python -m pip install -r python_kodlari/requirements_jupyter.txt
python -m ipykernel install --user --name hedef --display-name "Hedef (Python)"
python -m jupyter lab notebooks/00_baslangic.ipynb
```

Jupyter zaten açıksa dosyayı açıp **Hedef (Python)** kernel’ini seçin.
00’daki paket kurulum hücresi de aktif kernel’in Python’unu kullanır. Notebook’u
repo kökünden veya `notebooks/` klasöründen açabilirsiniz; makineye özel sabit yol yoktur.

İlk açılışta **Restart Kernel and Run All Cells** yalnızca örnek veriyi ve hazır
sonuçları inceler. Eğitim / uygulama için ilgili ayar hücresindeki `False`
değerini `True` yapıp baştan sırayla çalıştırın. Varsayılan hücreler paket kurmaz,
model eğitmez, model / atama çıktısı yazmaz ve sunucu başlatmaz. 01 bir bakiye grafiği de çizer.

Yeni eğitimler `outputs/jupyter/` altında benzersiz koşu klasörlerine yazılır.
Sonraki notebook önce burada tamamlanmış koşuyu, yoksa mevcut ana çıktıyı arar.
Kaynak koşular Git klonunda yoksa hazır dashboard CSV/JSON tabloları gösterilir.
Eski mutlak `latest.json` yolları doğrudan kullanılmaz; yollar proje kökünden çözülür.

Model arşivi açma seçeneği için önce `git lfs pull` gerekir. ZIP paketleri ayrı
klasöre açılır; veri / model klasörlerinin üzerine yazılmaz. 01’deki müşteri
hareketlerini baştan üretme adımı tam V3 müşteri kaynaklarını gerektirir. Buna
karşılık şube düzeyindeki örnek veriyle eğitim seti hazırlama ve güncel dinamik
seçim / kümeleme akışı doğrudan repodan çalıştırılabilir.

## Mevcut model politikası

Her coğrafi grup için 0,10 pilot doğrulama silhouette sınırı ve tam şube kapsamı
zorunludur. Şu an uygun ana aday yoktur. 05/06’daki çalıştırma ayarı `True`
olsa da uygun model olmadan yeni DEA veya atama üretilemez. Gösterilen önceki
DEA skorları tarihsel olarak işaretlenir. Banka bütçesi dağıtımı henüz yoktur.

Jupyter denemeleri ana dashboard’u kendiliğinden değiştirmez veya aktif modele
terfi etmez. 06’daki isteğe bağlı dashboard yenileme hücresi yalnızca ana
`outputs/` kaynaklarından çalışır; tam tarihsel kaynaklar gerekir.

## Notebook kontrolü

```bash
python notebooks/verify_notebooks.py
python notebooks/verify_notebooks.py --execute --kernel python3
```

İkinci komut ayrı bir temiz proje kopyasında yedi notebook’u gerçek kernel’de
çalıştırır. Varsayılan akışın sonuç yazmadığını, uygun model yokken DEA / atama
engellerini ve eğitim setinin 3.400 / 600 / 600 + 200 satırını kontrol eder.
Kontrol çıktıları `.work/` altındadır; repo notebook’larına çalışma çıktıları eklenmez.
