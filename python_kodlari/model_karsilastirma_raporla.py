"""Write the comparison methodology/results and a portable local model package."""
from pathlib import Path
import json
import shutil
from zipfile import ZipFile,ZIP_DEFLATED
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/model_karsilastirma'

def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert manifest['checks']['independent_all_72_model_reload_and_inference']
    methodology="""# Üç algoritmanın karşılaştırma metodolojisi

Bu deneyde finansal veriler sentetiktir. Birim şube-ay; tahmin ufku bir aydır.
İlk LightGBM deneyi ve dosyaları ayrı korunmuştur. Bu koşuda üç algoritma da
aynı kaydedilmiş CSV ayrımlarıyla yeniden eğitildi.

## Ortak veri ve tahmin sorusu

- 200 şube; İstanbul/Ankara/İzmir 80, diğer iller 120. Gruplar ayrı eğitilir.
- TL vadesiz, TL vadeli ve kredi aylık ortalamaları ayrı hedeflerdir.
- Eğitim: hedef ayı Şubat 2025–Haziran 2026, 3.400 satır.
- Doğrulama: Temmuz–Eylül 2026, 600 satır.
- Test: Ekim–Aralık 2026, 600 satır. Üç ardışık bir ay ileri tahmindir.
- Ocak 2027: Aralık 2026 bilgilerini içeren 200 girdide etiketsiz tahmin.
- Yapısal görünüm: 25 özellik, doğrudan sonraki ayın bakiyesi.
- Geçmiş bakiyeli görünüm: aynı 25 + 5 varlık/HP payı + 4 ilgili bakiye alanı.
  Model sonraki ay ile mevcut ay arasındaki farkı öğrenir;
  tahmin = max(0, mevcut bakiye + tahmin edilen fark).
- Müşteri/şube kimlikleri, gelecek gerçekleşmeler ve banka hedef büyümeleri
  modele verilmez. Özellikler mevcut ilk çalışmanın `pipeline.features` tanımını kullanır.

3 algoritma × 3 gösterge × 2 grup × 2 görünüm = 36 model.
Her modelin ayrı evaluation ve production kaydı vardır: toplam 72 dosya.

## Ayar seçimi

Her algoritma için üç aday yalnızca doğrulama MAE ile karşılaştırılır.
Aday sayısı eşittir; ağaç sayısı veya hesaplama bütçesi eşitlenmiş değildir.
Bu, sınırlı bir parametre taramasıdır; algoritmaların ulaşabileceği en iyi sonuç
olarak yorumlanmamalıdır. CPU iş parçacığı sınırı 2, rastgele tohum 20261120.

| Algoritma | A | B | C |
|---|---|---|---|
| LightGBM | 7 yaprak, derinlik 3, min 40, L2 10 | 15 yaprak, derinlik 4, min 40, L2 10 | 15 yaprak, derinlik 4, min 20, L2 25 |
| Random Forest | 200 ağaç, derinlik 6, min yaprak 10, max_features 0,9 | 300 ağaç, derinlik 10, min yaprak 5, max_features 1 | 300 ağaç, sınırsız derinlik, min yaprak 3, max_features 0,8 |
| CatBoost | derinlik 3, L2 10 | derinlik 4, L2 10 | derinlik 6, L2 25 |

LightGBM ve CatBoost için öğrenme oranı 0,04, üst sınır 800 ağaç ve
doğrulama MAE'sinde 50 turluk erken durdurma kullanılır. LightGBM kare hata
regresyonu, CatBoost RMSE ve Random Forest squared_error ile öğrenir;
aday seçimi her üçünde de MAE ile yapılır.

Ekonomik yapı LightGBM ve CatBoost'ta native kategoriktir. Random Forest için
one-hot dönüştürücü yalnızca eğitimde fit edilir; görülmeyen kategori yok sayılır.
Sayısal ölçekleme yapılmaz. RF'nin dönüştürücüsü modelle birlikte saklanır.

## Test ve üretim modelleri

Ayarlar ve ağaç sayısı doğrulamayla dondurulunca eğitim + doğrulama dönemleri
birleştirilerek evaluation modeli eğitilir ve test tahminleri saklanır.
Test dondurulduktan sonra aynı ayarlarla tüm etiketli dönemler kullanılarak
Ocak 2027 için ayrı production modeli eğitilir. Bu modellerin eğitim içi
skoru test başarısı olarak sunulmaz. Testteki sıralamaya bakılarak yeniden
ayar taraması veya özellik seçimi yapılmadı.

İki ortak referans: mevcut ay bakiyesini korumak ve eğitim hedef ortalamasını
kullanmak. Bunlar ilk çalışmadan aynı keyed tahminlerdir; tahmin etiketleri
üç algoritmayla bağımsız olarak eşleştirilir.

## Hata ve süre ölçümü

MAE, RMSE, WAPE, R² ve toplam sapması raporlanır. WAPE tüm satırlardan
mutlak hata toplamı / gerçekleşme toplamı olarak yeniden hesaplanır;
grup oranlarının ortalaması değildir. Aynı görüntüdeki bütün modeller
aynı gerçek değerler ve satır anahtarlarıyla karşılaştırılır.

Fit süresi, `.fit` çağrısını kapsar; RF'de one-hot fit/dönüşümü dahildir.
Doğrulama görünümünde seçilen TRAIN modeli, test görünümünde TRAIN+VALIDATION
yeniden eğitimi gösterilir. İki grup seçiliyse fit süreleri toplanır.
Üç adayın arama toplamı ayrıca gösterilir; doğrulama tahmini ve puanlamasını
da içerir. Önem hesaplama, model kaydı ve production yeniden eğitimi fit
çubuklarının dışında; tüm pipeline duvar süresinin içindedir.
Test tahmin gecikmesi ısınmadan sonra beş çağrının medyanıdır ve çıktı CSV'sinde
1.000 satıra ölçeklenir; az satırlı bu ölçüm bir yük testi değildir.
Süreler aynı makinedeki tek CPU koşusuna aittir; tekrarlı hız benchmark'ı
veya farklı donanım için hız garantisi değildir.

## Değişken önemi

Her model için TRAIN'de fit edilmiş seçilmiş model ve yalnızca Temmuz–Eylül
doğrulaması kullanılır. Özellikler aynı ay içindeki şubeler arasında karıştırılır;
aynı eşleştirme üç ay boyunca korunur. Aynı tohum ve sekiz tekrar kullanılır.
İlgili güncel bakiye karıştırılırsa dışarıdan eklenen anchor da birlikte değişir.
Tekil ve özellik grubu permütasyonları saklanır. Negatif ve sıfır değerler korunur.
Normalize önem = ΔMAE / doğrulama gerçekleşme ortalaması.

Ortak permütasyon ölçüsü algoritmalar arasında aynı tanıma sahiptir. Native
ölçüler farklıdır: LightGBM gain, RF impurity decrease ve CatBoost
PredictionValuesChange. Bu üç native yüzde birbirinin eşdeğeri değildir.
RF'nin one-hot sütun önemleri orijinal kategori alanına geri toplanır.
Geçmiş bakiyeli modellerde native önem dışarıdan eklenen mevcut bakiyeyi kapsamaz.

Üç tarihsel pencerede kararlılık incelemesi ilk LightGBM çalışmasında bulunur.
Bu yeni karşılaştırmada RF/CatBoost için tekrar edilmedi. Bunlara kararlı aday
etiketi verilmez. Kümeleme, DEA ve şube hedef dağıtımı henüz çalıştırılmadı.

## Doğrulama ve yeniden üretim

- Kaydedilmiş eğitim/doğrulama/test/girdi dosyalarının SHA-256 özeti korunur.
- 72 kayıtlı model yeniden yüklenerek hem test hem Ocak tahminleriyle eşleştirilir.
- Şube, banka toplamı ve ay bazındaki metrikler bağımsız kodla yeniden hesaplanır.
- Sekiz tahminleyicinin her gözlemde aynı gerçekle karşılaştırıldığı doğrulanır.
- Kütüphane sürümleri, donanım platformu ve süre notları manifest.json'dadır.

Proje kökünden:

```bash
python python_kodlari/model_karsilastir.py
python python_kodlari/model_karsilastir_dogrula.py
python python_kodlari/model_karsilastirma_raporla.py
python python_kodlari/dashboard_veri_hazirla.py
```

Bağımsız model paketini açtıktan sonra:

```bash
python -m pip install -r kod/requirements.txt
python kod/model_karsilastir.py --source . --output yeniden_calisma
python kod/model_karsilastir_dogrula.py --source . --output yeniden_calisma
```

## Teknik başvuru kaynakları

- [CatBoost fit ve erken durdurma](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor_fit)
- [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
"""
    (OUT/'METODOLOJI.md').write_text(methodology,encoding='utf-8')
    metrics=pd.read_csv(OUT/'sonuclar/toplam_metrikler.csv',encoding='utf-8-sig')
    per_group=pd.read_csv(OUT/'sonuclar/model_metrikleri.csv',encoding='utf-8-sig')
    names={'tl_vadesiz':'TL vadesiz','tl_vadeli':'TL vadeli','kredi':'Kredi'}
    algs={'lightgbm':'LightGBM','random_forest':'Random Forest','catboost':'CatBoost'}
    lines=['# Üç algoritma — ölçülen sonuçlar','',
      'Aynı ayrımlar ve özelliklerle yeniden eğitim. Test: Ekim–Aralık 2026. Finansal veriler sentetiktir.',
      '', '| Görünüm | Gösterge | LightGBM WAPE | RF WAPE | CatBoost WAPE | Son ayı koru |',
      '|---|---|---:|---:|---:|---:|']
    for view in ['yapisal','gecikmeli']:
        for target,label in names.items():
            p=metrics[metrics.gosterge.eq(target)&metrics.split.eq('TEST')].set_index('model')
            values=[p.loc[a+'__'+view,'wape'] for a in algs]+[p.loc['son_ayi_koru','wape']]
            lines.append('| '+('Yapısal' if view=='yapisal' else 'Geçmiş bakiyeli')+' | '+label+' | '+' | '.join(f'%{x*100:.2f}' for x in values)+' |')
    lines+=['','| Algoritma | 12 test modeli fit toplamı | 36 aday fit toplamı |',
            '|---|---:|---:|']
    search=pd.read_csv(OUT/'sonuclar/ayar_arama.csv',encoding='utf-8-sig')
    for a,label in algs.items():
        fit=per_group[per_group.algoritma.eq(a)&per_group.split.eq('TEST')].egitim_saniye.sum()
        tuning=search[search.algoritma.eq(a)].fit_seconds.sum()
        lines.append(f'| {label} | {fit:.3f} sn | {tuning:.3f} sn |')
    lines+=['',f'Tüm pipeline: {manifest["pipeline_seconds"]:.1f} sn. Süreler tek yerel koşudur; önemler ve dosya işlemleri fit toplamlarının dışında kalır.',
      '', 'Her modelin üç adayından seçim yalnızca doğrulama MAE ile yapıldı. Bu test tablosu sınırlı aday taramasının sonucudur; genel algoritma üstünlüğü iddiası değildir.',
      '', 'Kontroller: 72 modelin yeniden yükleme/tahmin eşleşmesi; aynı keyed gerçekler; değişmeyen veri hashleri; toplam, grup ve aylık metrik mutabakatı geçti.',
      '', 'URF/K-Medoids, DEA ve şube hedef dağıtımı sonraki aşamalardır. RF/CatBoost için üç dönemlik özellik kararlılığı ölçülmedi.']
    (OUT/'SONUCLAR.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for name in ['model_karsilastir.py','model_karsilastir_dogrula.py','model_karsilastirma_raporla.py',
                 'tahmin_karsilastir.py','pipeline.py','requirements.txt']:
        shutil.copy2(ROOT/'python_kodlari'/name,OUT/'kod'/name)
    archive=ROOT/'outputs/model_karsilastirma_paketi.zip'
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,Path(OUT.name)/p.relative_to(OUT))
    with ZipFile(archive) as z:assert z.testzip() is None
    print('Report and model package:',archive,archive.stat().st_size,'bytes')

if __name__=='__main__':main()
