"""Readable V4 protocol/results, model package and V3/V4 side-by-side metrics."""
import ast
import csv
import json
from pathlib import Path
import shutil
from zipfile import ZipFile,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/model_karsilastirma_v4'
ALGS={'lightgbm':'LightGBM','random_forest':'Random Forest','catboost':'CatBoost'}
VIEWS={'yapisal':'Yapısal','gecikmeli':'Geçmiş bakiyeli','hareketli':'Hareket + ay sonu'}
TARGETS={'tl_vadesiz':'TL vadesiz','tl_vadeli':'TL vadeli','kredi':'Kredi'}

def read(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def main():
    m=json.loads((OUT/'manifest.json').read_text());assert all(m['checks'].values())
    assert m['checks']['independent_saved_models_reload_and_inference']
    totals=read(OUT/'sonuclar/toplam_metrikler.csv');per_group=read(OUT/'sonuclar/model_metrikleri.csv')
    lookup={(r['gosterge'],r['model'],r['split']):r for r in totals}
    lines=['# V4 model karşılaştırması','',
      'Veri: 26 ayın tamamında aynı mekanizmayla üretilmiş müşteri/ürün hareketleri. Test: Ekim–Aralık 2026.',
      'Müşteri demografileri güncel aktif şube aidiyetinden, bakiye toplamları gün ağırlıklı aidiyet parçalarından gelir.',
      'Her tabloda aynı hedef ve aynı test satırları karşılaştırılır. Finansal gözlemler sentetiktir.','']
    for view,label in VIEWS.items():
        lines += [f'## {label}: {m["feature_counts"][view]} özellik','',
          '| Gösterge | LightGBM WAPE | RF WAPE | CatBoost WAPE | Son ay ortalaması | Ay sonu stokunu koru |',
          '|---|---:|---:|---:|---:|---:|']
        for target,name in TARGETS.items():
            keys=[a+'__'+view for a in ALGS]+['son_ayi_koru','ay_sonu_koru']
            values=[float(lookup[(target,key,'TEST')]['wape']) for key in keys]
            lines.append('| '+name+' | '+' | '.join(f'%{v*100:.3f}' for v in values)+' |')
        lines.append('')
    lines += ['## Doğrulamayla seçilen tahminleyicilerin ayrı test sonucu','',
      'Seçim için yalnızca Temmuz–Eylül doğrulama MAE kullanıldı. Test sıralamasıyla parametre değiştirilmedi.',
      '', '| Gösterge | Doğrulamada seçilen | Doğrulama WAPE | Test WAPE |', '|---|---|---:|---:|']
    for target,name in TARGETS.items():
        records=[r for r in totals if r['gosterge']==target and r['split']=='VALIDATION']
        chosen=min(records,key=lambda r:float(r['mae_tl']))
        lines.append(f'| {name} | {chosen["model"]} | %{float(chosen["wape"])*100:.3f} | %{float(lookup[(target,chosen["model"],"TEST")]["wape"])*100:.3f} |')
    lines+=['','## Ölçülen eğitim süreleri','',
      '| Algoritma | 18 test modeli fit toplamı |', '|---|---:|']
    for alg,name in ALGS.items():
        seconds=sum(float(r['egitim_saniye']) for r in per_group if r['algoritma']==alg and r['split']=='TEST')
        lines.append(f'| {name} | {seconds:.3f} sn |')
    lines+=['',f'Tüm koşu: {m["pipeline_seconds"]:.1f} sn; parametre araması, önem, kaydetme ve Ocak modelleri dahil.',
      'Süreler tek yerel CPU koşusudur, 2 iş parçacığı kullanılmıştır. V3 ve V4 farklı tarihlerde çalıştırılmıştır.',
      '', '## Yorumlama sınırları','',
      '- Yapısal görünüm bakiyeleri bilinçli dışarıda bırakır; stok bilgisine erişen referanslarla aynı bilgi kümesini kullanmaz.',
      '- Geçmiş bakiyeli görünüm önceki deneydeki aynı 34 alanı korur. Hareket görünümü 15 ek güncel alan ve ay sonu anchor kullanır.',
      '- Ay sonu stok referansı, hareket modelindeki anchor bilgisine erişir. Böylece olası fayda sadece bilinen kapanış stokunu eklemekten kaynaklanıyormuş gibi sunulmaz.',
      '- Gelecek ayın devir/ayrılma olayları girdilere verilmez. Bilinmeyen yeni müşteri ve kaynak hareketleri tahmin hatası oluşturabilir.',
      '- V3 ve V4 test değerleri farklı veri üretimlerine aittir. Aradaki hata farkı gerçek banka performansında iyileşme kanıtı değildir.',
      '- Teste bakılarak yeni parametre taraması veya veri üretimi yapılmadı. Üç dönemlik özellik kararlılığı bu koşuda hesaplanmadı.',
      '- Bu çalışma bir ay ileri tahmin karşılaştırmasıdır. URF/K-Medoids, DEA veya hedef dağıtımı tamamlanmış değildir.']
    (OUT/'SONUCLAR.md').write_text('\n'.join(lines)+'\n')
    methodology='''# V4 eğitim ve karşılaştırma protokolü

200 şube ve üç gösterge (TL vadesiz, TL vadeli, kredi aylık ortalaması) için
İstanbul/Ankara/İzmir 80 şube ve diğer iller 120 şube olarak ayrı eğitilir.
Her algoritma aynı satırları, aynı özellikleri ve aynı zaman ayrımını kullanır.

## Zaman ayrımı ve aidiyet

- Bilgi t ayı sonundaki görünüm, etiket t+1 ayının ortalama bakiyesidir.
- Eğitim hedef ayları Şubat 2025–Haziran 2026: 3.400 satır.
- Doğrulama Temmuz–Eylül 2026: 600 satır. Test Ekim–Aralık 2026: 600 satır.
- İlk iki ay geçmiş pencere için kullanılır. Aralık 2026 bilgileriyle Ocak 2027
  için 200 girdide etiketsiz tahmin üretilir.
- Şube demografileri sabit müşteri ana tablosundan yeniden birleştirilmez;
  V4'ün tarihli, ay sonu aktif müşteri kitlesinden hesaplanmış tablosu kullanılır.
- Üretim/bakiye gerçekleşmeleri, gün ağırlıklı müşteri-şube parçalarından gelir.
- Gelecek ayın olayları özelliklere katılmaz. Parametre seçiminde test kullanılmaz.

## Üç özellik görünümü

1. Yapısal (25): demografi, müşteri sayısı, gelir, kapasite, bölge ve takvim.
   Ürün bakiyesi ve varlık/HP türevleri yoktur. Doğrudan bakiye öğrenir.
2. Geçmiş bakiyeli (34): önceki karşılaştırmayla aynı 25 + 5 varlık/HP alanı
   + ilgili göstergenin mevcut/önceki/son üç ay/büyüme alanları.
   Tahmin = max(0, mevcut aylık ortalama + model(y_next − mevcut ortalama)).
3. Hareket + ay sonu (49): aynı 34 alan + 15 güncel alan. Bunlar ilişki skoru,
   yeni/ayrılan/devir gelen/devir giden müşteri sayıları, dış kaynak giriş/çıkışı,
   müşteri-günü, ay içi hizmet verilen müşteri sayısı, müşteri sayısı değişimi,
   müşteri-günü/ay sonu müşteri oranı, ilgili göstergenin kapanış stoku,
   kapanış−ortalama farkı ve mevcut ayın devir/devir dışı stok değişimleridir.
   Tahmin = max(0, ay sonu stoku + model(y_next − ay sonu stoku)).

Üç algoritma × üç gösterge × iki grup × üç görünüm = 54 model ailesi.
Her birinin ayrı evaluation ve production kaydı vardır: 108 model dosyası.

## Algoritmalar ve seçim

LightGBM, Random Forest ve CatBoost için önceki V3 karşılaştırmasındaki aynı
üç aday yapılandırma korunur. Öğrenme oranı boosting modellerinde 0,04,
üst sınır 800 ağaç ve doğrulama MAE ile 50 turluk erken durdurmadır.
RF adayları 200/300 ağaç, derinlik 6/10/sınırsız ve min yaprak 10/5/3'tür.
Tohum 20261120, CPU iş parçacığı 2. Ayrıntılı adaylar ayar_arama.csv içindedir.
LightGBM/CatBoost native kategorik, RF yalnızca eğitimde fit edilen one-hot
kodlama kullanır. Sayısal ölçekleme yapılmaz. Aday seçimi doğrulama MAE'dir.
Testten önce ayar ve ağaç sayısı dondurulup TRAIN+VALIDATION ile yeniden fit edilir.
Test skorları saklandıktan sonra tüm etiketli dönemlerle ayrı Ocak modeli eğitilir.

## Referanslar, hata ve süre

Üç referans V4'ün kendi satırlarından yeniden hesaplanır: mevcut aylık ortalama,
mevcut ay sonu stoku ve eğitim hedef ortalaması. V3 tahminleri V4'e taşınmaz.
Ay sonu referansı hareket görünümündeki anchor bilgisine erişir.
MAE/RMSE TL, WAPE mutlak hata toplamı / gerçekleşme toplamı, R² ve toplam
sapması raporlanır. Banka WAPE'si grup yüzdelerini ortalayarak hesaplanmaz.
Fit süreleri seçilen modelin `.fit` çağrısıdır; RF dönüştürücüsünü içerir.
Arama, önem ve dosya işlemleri ayrıca tam pipeline süresine dahildir.
Süreler tek yerel koşudur; genel hız garantisi değildir.

## Önem ve bağımsız kontroller

Her modelde TRAIN'de fit edilmiş seçilmiş modelle yalnızca doğrulamada sekiz
tekrarlı permutation importance hesaplanır. Aynı ay içinde şubeler karıştırılır,
aynı eşleştirme üç ay boyunca korunur. Anchor alanı karıştırılırsa dışarıdan
eklenen anchor da birlikte karıştırılır. Native gain, RF impurity decrease ve
CatBoost PredictionValuesChange farklı tanımlardır; ortak ölçü permütasyondur.
Geçmiş/ay sonu bilgileri otomatik kümeleme veya DEA girdisi sayılmaz.

108 kayıtlı model yeniden yüklenip test ve Ocak tahminleriyle eşleştirilir.
Toplam/grup/aylık metrikler satırlardan bağımsız hesaplanır. Kaynak hashleri,
gelecek etiketleri, mevcut ortalama/kapanış stokları ve dinamik müşteri kitlesi
raw V4 tablosuyla eşleştirilir. CSV dosyalarındaki satır anahtarları kontrol edilir.

## Yeniden çalıştırma

Proje kökünde, requirements.txt sürümlerini içeren Python 3.12 ortamıyla:

```bash
python python_kodlari/v4_egitim_seti_hazirla.py
python python_kodlari/model_karsilastir.py --protocol v4 --source outputs/model_karsilastirma_v4 --output outputs/model_karsilastirma_v4
python python_kodlari/model_karsilastir_dogrula.py --source outputs/model_karsilastirma_v4 --output outputs/model_karsilastirma_v4
python python_kodlari/model_v4_raporla.py
python python_kodlari/dashboard_veri_hazirla.py
```

Veri ve sonuçlar sentetiktir. Banka üst hedef büyümeleri modele verilmez.
Şube hedef dağıtımı bu tahmin karşılaştırmasından ayrı bir tasarım aşamasıdır.
'''
    (OUT/'METODOLOJI.md').write_text(methodology)
    old=read(ROOT/'outputs/model_karsilastirma/sonuclar/toplam_metrikler.csv')
    comparison=[]
    for version,data in [(3,old),(4,totals)]:
        for row in data:
            if row['split']=='TEST':comparison.append({'veri_surumu':version,**row})
    with (OUT/'sonuclar/v3_v4_test_karsilastirma.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(comparison[0]));w.writeheader();w.writerows(comparison)
    specs={}
    for filename,names in [('pipeline.py',['STRUCTURAL','PROXIES']),('model_v4_protokol.py',['EXTRAS'])]:
        tree=ast.parse((ROOT/'python_kodlari'/filename).read_text())
        for node in tree.body:
            if isinstance(node,ast.Assign) and any(isinstance(n,ast.Name) and n.id in names for n in node.targets):specs.update(ast.literal_eval(node.value))
    metadata=[json.loads(p.read_text()) for p in sorted((OUT/'modeller').glob('*__metadata.json'))]
    fields=sorted({f for meta in metadata for f in meta['features']})
    dictionary=[]
    for f in fields:
        family,desc,unit=specs.get(f,('donem_sonu' if '_ay_sonu_' in f or '_kapanis_ortalama_' in f else 'kaynak_hareketi' if '_devir_' in f else 'gecmis_bakiye','t ayında bilinen gösterge veya geçmiş pencere alanı','oran' if 'buyume' in f else 'TL'))
        dictionary.append({'alan':f,'ozellik_grubu':family,'aciklama':desc,'birim':unit,
          'kumeleme_adayi':family not in ['takvim','varlik_segmenti','gecmis_bakiye','donem_sonu','kaynak_hareketi','musteri_hareketi'],
          **{view:any(meta['gorunum']==view and f in meta['features'] for meta in metadata) for view in VIEWS}})
    with (OUT/'veri/ozellik_sozlugu.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dictionary[0]));w.writeheader();w.writerows(dictionary)
    for name in ['model_karsilastir.py','model_karsilastir_dogrula.py','model_v4_protokol.py','v4_egitim_seti_hazirla.py',
                 'model_v4_raporla.py','tahmin_karsilastir.py','pipeline.py','requirements.txt']:
        shutil.copy2(OUT/name if name=='requirements.txt' else ROOT/'python_kodlari'/name,OUT/'kod'/name)
    archive=ROOT/'outputs/model_karsilastirma_v4_paketi.zip'
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,Path(OUT.name)/p.relative_to(OUT))
    with ZipFile(archive) as z:assert z.testzip() is None
    print('V4 report and package ready:',archive,archive.stat().st_size)

if __name__=='__main__':main()
