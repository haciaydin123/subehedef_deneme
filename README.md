# Şube hedef dağıtım çalışması

Bireysel bankacılık için sentetik verilerle geliştirilen model ve metodoloji
prototipi. Müşteri yapısına göre değişken seçimi, şube kümeleme yöntemlerinin
karşılaştırılması, ortak model seçimi ve küme içi DEA verimlilik analizi içerir.
Şube hedef dağıtımı henüz uygulanmadı.

Üç gösterge aylık ortalama TL vadesiz mevduat, TL vadeli mevduat ve kredi
bakiyesidir. İstanbul, Ankara ve İzmir bir coğrafi grup; diğer iller ikinci
gruptur. Yöntem ve senaryo bütün göstergeler ve iki grup için ortak seçilir.

## Jupyter ile çalıştırma

**[00_baslangic.ipynb](notebooks/00_baslangic.ipynb)** ile başlayın. Yedi notebook;
veri hazırlama, model karşılaştırma / FLAML, dinamik değişken seçimi, kümeleme,
DEA ve yeni ay uygulamasını açıklamalı hücrelerle sunar. İlk çalıştırmada hazır
sonuçlar incelenir; eğitimler ayar hücrelerinden açılır ve `outputs/jupyter/`
altına kaydedilir. Mevcut `.py` modülleri hesaplama kaynağı olarak korunur.

```bash
python -m pip install -r python_kodlari/requirements_jupyter.txt
python -m jupyter lab notebooks/00_baslangic.ipynb
```

Ayrıntılar ve kernel seçimi: [Jupyter rehberi](notebooks/README.md).

## Dashboard'u açma

Python 3 ve tarayıcı yeterlidir; dashboard modeli yeniden eğitmez. İndirme
arşivleri Git LFS ile saklanır. Klonlamadan önce Git LFS kurulu olmalıdır:

```bash
git lfs install
git clone https://github.com/haciaydin123/subehedef_deneme.git
cd subehedef_deneme
git lfs pull
python3 -m http.server 8765 --bind 127.0.0.1 --directory model_dashboard/dist
```

Tarayıcıdan <http://127.0.0.1:8765/> adresini açın. `dist/index.html` dosyası
doğrudan da açılabilir. GitHub'ın “Download ZIP” çıktısı LFS arşivlerini
içermeyebilir; indirme paketleri için `git lfs pull` kullanın.

## Güncel sonuç

Her coğrafi grupta Ekim–Kasım 2026 doğrulama dönemi ortalama silhouette değeri
için **0,10** pilot alt sınırı aranır. Mevcut sekiz uygulanabilir adayın hiçbiri
bütün koşulları sağlamadığı için **onaylı ana kümeleme modeli yoktur**. Yeni
atama ve DEA bu durumda durur; önceki model otomatik geri alınmaz. Dashboard,
önceki 200 şubelik DEA sonuçlarını tarihsel olarak gösterir.

Tam şube kapsamı zorunludur. Küme boyutları ana DEA ve tanımlı ek girdi
senaryolarına birlikte yeterli olmalıdır; mevcut ortak alt sınır 15 şubedir.
Eksik atama veya küçük küme varsa şube elemek yerine işlem durur.

0,10 ayarlanabilir bir pilot politika değeridir. Sentetik sonuçlar gerçek
banka performansı olarak yorumlanmaz. Aralık 2026 daha önce incelenmiş bir
tarihsel kontrol dönemidir; dokunulmamış test değildir.

## Kod ve örnek veri

- `python_kodlari/`: Python komutları, model modülleri, testler ve ayarlar.
- `notebooks/`: Jupyter `.ipynb` dosyaları ve ortak çalışma yardımcıları.
- `model_dashboard/dist/`: hazır HTML/CSS/JavaScript dashboard ve sonuç anlık görüntüsü.
- `model_dashboard/dist/downloads/`: raporlar, sözlükler ve Git LFS model arşivleri.
- `outputs/banka_test_verisi_v4/`: 200 şube × 26 aylık sentetik şube girdisi,
  şube/bölge tanımları, stok mutabakatı, veri manifesti ve örnek banka hedefleri.

Sanal ortamlar, önbellekler, müşteri düzeyindeki büyük ara tablolar ve bütün
tarihsel koşu klasörleri Git'e alınmaz. Hazır dashboard, Python eğitim ortamı
kurmadan incelenebilir. Kaynak veriden bütün tarihsel deneyleri yeniden üretmek
için ilgili veri ve model paketleri ayrıca gereklidir.

Python modelleri için önerilen ortam Python 3.12'dir:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r python_kodlari/requirements_kumeleme.txt
```

macOS'ta LightGBM için `libomp` gerekebilir. Şube düzeyindeki örnek girdiyle
aşamalar sırasıyla çalıştırılabilir:

```bash
python python_kodlari/dinamik_feature_selection.py run --config python_kodlari/ayarlar/dinamik_v4.json
python python_kodlari/kumeleme_karsilastir.py run --config python_kodlari/ayarlar/kumeleme.json
python python_kodlari/otomatik_kumeleme.py select --config python_kodlari/ayarlar/otomatik_kumeleme.json
```

Son aşama uygun aday bulamazsa durmak beklenen davranıştır. DEA ancak doğrulanmış
ve uygun bir ana model varsa başlatılır. Dashboard'un veri hazırlama betiği bütün
tarihsel kaynak koşuları bekler; yalnızca örnek panelle çalıştırmak yerine
repodaki hazır anlık görüntü kullanılabilir.

## Kontroller

Dashboard kontrolü Node.js ile çalışır; harici npm paketi gerekmez. Önce LFS
dosyalarının indirildiğinden emin olun:

```bash
node model_dashboard/scripts/verify.mjs
PYTHONPATH=python_kodlari python -m unittest discover -s python_kodlari/tests -p 'test_kumeleme_secimi.py'
PYTHONPATH=python_kodlari python -m unittest discover -s python_kodlari/tests -p 'test_full_branch_coverage.py'
```

Ayrıntılı komutlar ve metodoloji için [Python rehberi](python_kodlari/README.md)
ve [dashboard rehberine](model_dashboard/README.md) bakın.
