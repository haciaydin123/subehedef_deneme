"""Produce a verified FLAML report, bounded trial history and portable artifacts."""
import csv
import json
from pathlib import Path
import shutil
from zipfile import ZipFile,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/flaml_v4'
NAMES={'flaml_automl':'FLAML adayı','secilen_sistem':'Doğrulamayla seçilen sistem',
 'lightgbm__hareketli':'Manuel LightGBM','random_forest__hareketli':'Manuel Random Forest',
 'catboost__hareketli':'Manuel CatBoost','ay_sonu_koru':'Ay sonu stokunu koru',
 'son_ayi_koru':'Son ay ortalamasını koru','egitim_ortalamasi':'Eğitim ortalaması'}
TARGETS={'tl_vadesiz':'TL vadesiz','tl_vadeli':'TL vadeli','kredi':'Kredi'}
GROUPS={'UC_BUYUK_IL':'İstanbul / Ankara / İzmir','DIGER_ILLER':'Diğer iller'}

def main():
    m=json.loads((OUT/'manifest.json').read_text())
    assert all(m['checks'].values()) and m['checks']['independent_30_bundles_reloaded']
    report=json.loads((OUT/'sonuclar/rapor_verisi.json').read_text());report['manifest']=m
    lookup={(r['gosterge'],r['model'],r['split']):r for r in report['total_metrics']}
    history=[];updates=[]
    for meta in report['model_metadata']:
        cell=f'{meta["target"]}__{meta["group"]}'
        actual_mean=meta['validation_candidates']['flaml_automl']['gercek_toplam_tl']/meta['validation_candidates']['flaml_automl']['n']
        best=float('inf')
        for line in (OUT/'aramalar'/f'{cell}.jsonl').read_text().splitlines():
            row=json.loads(line)
            if 'learner' not in row:continue
            value=float(row['validation_loss']);improved=value<best;best=min(best,value)
            result={'gosterge':meta['target'],'sube_grubu':meta['group'],'record_id':row['record_id'],
              'algoritma':row['learner'],'arama_saniye':row['wall_clock_time'],'deneme_saniye':row['trial_time'],
              'valid_mae_tl':value,'valid_wape':value/actual_mean,'en_iyi_valid_wape':best/actual_mean,
              'yeni_en_iyi':improved,'egitim_satiri':row['sample_size'],'config_json':json.dumps(row['config'],ensure_ascii=False)}
            history.append(result)
            if improved:updates.append(result)
    with (OUT/'sonuclar/arama_gecmisi.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
    report['search_updates']=updates
    (OUT/'sonuclar/rapor_verisi.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    lines=['# V4 FLAML AutoML denemesi','',
      'Aynı V4 eğitim/doğrulama/test ayrımları ve 49 alanlı hareket görünümü. Üç gösterge × iki şube grubu için altı arama.',
      f'Her aramaya {m["settings"]["time_budget_per_search"]:.0f} saniye bütçe, 2 CPU iş parçacığı. Toplam {m["trial_count"]:,} deneme.',
      '', '**Ekim–Aralık 2026 daha önce incelendi. Buradaki test sonuçları tekrar kullanılan tarihsel karşılaştırmadır; el değmemiş son test değildir.**',
      '', '| Gösterge | FLAML adayı WAPE | Seçilen sistem WAPE | Ay sonu referansı | Son ay ortalaması |',
      '|---|---:|---:|---:|---:|']
    for target,label in TARGETS.items():
        values=[lookup[(target,k,'TEST')]['wape'] for k in ['flaml_automl','secilen_sistem','ay_sonu_koru','son_ayi_koru']]
        lines.append('| '+label+' | '+' | '.join(f'%{x*100:.3f}' for x in values)+' |')
    lines += ['', '## Doğrulamada dondurulan seçimler','',
       '| Gösterge | Şube grubu | FLAML algoritması | Deneme | Seçilen aday |', '|---|---|---|---:|---|']
    learner={'lgbm':'LightGBM','rf':'Random Forest','catboost':'CatBoost'}
    for meta in report['model_metadata']:
        lines.append(f'| {TARGETS[meta["target"]]} | {GROUPS[meta["group"]]} | {learner[meta["best_learner"]]} | {meta["trials"]} | {NAMES[meta["selected_candidate"]]} |')
    lines += ['', 'Seçim her gösterge–şube grubu için yalnızca Temmuz–Eylül doğrulama MAE ile yapıldı. Banka geneli seçilen sistem, grupların farklı yöntemlerini birleştirebilir.',
       '', f'Arama toplamı: {m["search_seconds_total"]:.1f} sn. Tam pipeline: {m["pipeline_seconds"]:.1f} sn. Süreler tek yerel koşuya aittir.',
       '', 'Kontroller: 30 native paket yeniden yüklenerek tahminlerle eşleştirildi; seçimler yalnızca dışa aktarılan doğrulama satırlarından yeniden hesaplandı; bütün toplam/grup/aylık metrikler ve kaynak hashleri doğrulandı.',
       '', 'FLAML daha iyi sonucu garanti etmez. Basit referans ve mevcut model kazanabilir. Sentetik verideki sıralama gerçek banka başarısı anlamına gelmez.']
    (OUT/'SONUCLAR.md').write_text('\n'.join(lines)+'\n')
    methodology='''# FLAML V4 protokolü

Bu paket bir AutoML pilotudur. FLAML 2.8.0 yerel çalışır; bulut servisi değildir.
Kişi/şube kimlikleri ve gelecek gerçekleşmeler girdi değildir. Finansallar sentetiktir.

## Veri ve hedef

V4'ün aynı 3.400 eğitim, 600 doğrulama, 600 test satırı korunur. Üç gösterge
(TL vadesiz, TL vadeli ve KMH dahil nakdi kredi) ve iki şube grubu ayrı aranır.
Girdi, t ayı sonundaki 49 alanlı hareket görünümüdür; hedef t+1 aylık ortalamasıdır.
Eğitim Şubat 2025–Haziran 2026 hedef ayları, doğrulama Temmuz–Eylül,
tarihsel karşılaştırma Ekim–Aralık 2026'dır. Ocak 2027 girdilerinde etiket yoktur.
Veri hashleri mevcut V4 modellerinin hashleriyle eşleşmezse işlem durur.

FLAML mevcut ay sonu stokuna göre farkı öğrenir. Nihai tahmin:
max(0, mevcut ay sonu stoku + tahmin edilen fark).
Arama kaybı bu toplamdan hesaplanan gerçek TL MAE'dir. Yalnızca farkı karıştıran
yanlış bir metrik kullanılmaz; doğrulama MAE ve WAPE birlikte kaydedilir.

## Arama

- Altı bağımsız arama, her biri 120 sn hedef bütçe; tek deneme en fazla 20 sn
  kaynak bütçesi. Duvar süresi eğitim/işletim ayrıntılarıyla sınırı aşabilir.
- Üç aday: lgbm, rf, catboost. XGBoost FLAML kurulum bağımlılığıdır, aday değildir.
- 2 CPU iş parçacığı, bir seferde bir deneme, tohum 20261120.
- Dış doğrulama açık X_val/y_val ile verilir. Rastgele dış bölme yoktur.
- sample=False: adaylar bütün eğitim satırlarıyla fit edilir.
- skip_transform=True: 49 alanın sırası ve mevcut kategorik sözlük korunur.
- ensemble=False, retrain_full=False: yalnızca arama yapılır. Seçilmiş modelin
  sonraki eğitimleri ayrı ve açık biçimde yürütülür.
- CatBoost use_best_model=False ve early_stopping_rounds=None: gizli bir
  iç doğrulama bölmesiyle eğitim satırı ayırmaz. Ağaç sayısı dış doğrulamayla aranır.
- LightGBM 8–1024 ağaç ve 4–64 yaprak; RF 16–512 ağaç ve 4–128 yaprak;
  CatBoost 16–800 ağaç, derinlik 3/4/6, L2 3/10/25. Diğer aranan alanlar
  FLAML varsayılan arama uzayındadır. Bütün denemeler JSONL/CSV'de saklanır.
- LightGBM/CatBoost native kategorik kullanır. FLAML'nin yerleşik RF'si
  kategorileri kodlara dönüştürür. Önceki manuel RF one-hot kullandığından,
  aradaki fark sadece hiperparametre aramasına mal edilmemelidir.

Zamana bağlı arama aynı tohumla farklı makinelerde farklı sayıda deneme
tamamlayabilir. Kesin çıktı yeniden üretimi için kaydedilmiş modeller kullanılır.

## Doğrulama seçimi ve kayıtlar

Her grupta yedi aday karşılaştırılır: FLAML'nin bulduğu model; önceki 49 alanlı
LightGBM, Random Forest ve CatBoost; ay sonu stokunu koru, son ay ortalamasını
koru ve eğitim ortalaması. En düşük doğrulama MAE seçilir. Tam eşitlikte
basit referans tercih edilir. Seçilen sistem bir yönlendiricidir; banka genelinde
iki grup farklı yöntemler kullanabilir.

Karar TEST tahmininden önce ayrı selection.json dosyasına yazılır. Test veya
Ocak gerçekleşmesi parametre/seçim için kullanılmaz. Ancak Ekim–Aralık testinin
önceki çalışmada görüldüğü açıktır; yeni, dokunulmamış test başarısı iddia edilmez.
Gerçek veri kullanımında ileriye dönük yeni aylarda bağımsız izleme gerekir.

FLAML'nin native modeli ve gerçek eğitilmiş ağaç sayısı dışarı alınır.
Native paket doğrulamada FLAML predict ile eşleşmek zorundadır. Bu ağaç sayısı
ve parametrelerle TRAIN+VALIDATION üzerinde evaluation modeli fit edilir.
Test skorları sabitlenince tüm etiketli dönemlerle ayrı production modeli fit edilir.
Seçilen mevcut modellerde yalnızca önceki evaluation/production dosyaları kendi
amaçları için kullanılır. Production modeliyle eğitim içi test skoru raporlanmaz.

Toplam 30 paket: altı hücre için FLAML validation/evaluation/production (18)
ve seçilen sistem evaluation/production (12). Native tahmin kodu FLAML aramasını
yeniden başlatmaz. Paketler kendi kategorik kodlamasını ve anchor bilgisini taşır.

## Kullanım

Proje kökünde Python 3.12 ile requirements.txt paketlerini kurduktan sonra:

```bash
python python_kodlari/flaml_pipeline.py --budget 120
python python_kodlari/flaml_dogrula.py
python python_kodlari/flaml_raporla.py
python python_kodlari/dashboard_veri_hazirla.py
```

Kesilmiş bir koşuyu aynı veri, kod ve ayarlarla sürdürmek için --resume kullanılır.
Kod/veri/ayar imzası farklıysa eski hücreler sessizce kullanılmaz. Bütçe --budget
ile ayarlanabilir. Gerçek aylık veride tarihli hazırlık ve mevcut modellerin
aynı girdilerle eğitilmiş olması ayrıca sağlanmalıdır; zamanlayıcı kurulmamıştır.

Tahmin:

```bash
python python_kodlari/flaml_tahmin.py --model outputs/flaml_v4/modeller/tl_vadesiz__UC_BUYUK_IL__champion_production.joblib --input outputs/flaml_v4/veri/ocak2027_tahmin_girdisi.csv --output tahmin.csv
```

## Kontroller

Üç algoritmanın entegrasyonu yalnızca TRAIN/VALIDATION ile sınanır. Tüm 30 paket
dosyadan yüklenip kayıtlı validation/test/forecast tahminleriyle eşleştirilir.
Seçim doğrulama satırlarından bağımsız yeniden hesaplanır; trial kaybı native
çıktıyla eşleştirilir. Sekiz tahminleyicinin her gözlemde aynı gerçekle
karşılaştırıldığı ve toplam/grup/aylık metriklerin mutabakatı kontrol edilir.

Özellik önemi ve üç dönemlik kararlılık bu AutoML pilotunda yeniden hesaplanmadı.
Önceki V4 önemleri önceki modellerindir; FLAML'ye ait gibi gösterilmez.
URF/K-Medoids, DEA ve şube hedef dağıtımı ayrı aşamalardır.

Kaynak: [FLAML görev ve metrik ayarları](https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML/),
[FLAML AutoML API](https://microsoft.github.io/FLAML/docs/reference/automl/automl/).
'''
    (OUT/'METODOLOJI.md').write_text(methodology)
    for name in ['flaml_pipeline.py','flaml_tahmin.py','flaml_dogrula.py','flaml_raporla.py',
                 'flaml_uyumluluk_kontrol.py','model_v4_protokol.py','pipeline.py']:
        shutil.copy2(ROOT/'python_kodlari'/name,OUT/'kod'/name)
    shutil.copy2(OUT/'requirements.txt',OUT/'kod/requirements.txt')
    archive=ROOT/'outputs/flaml_v4_paketi.zip'
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for path in sorted(OUT.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:z.write(path,Path(OUT.name)/path.relative_to(OUT))
    with ZipFile(archive) as z:assert z.testzip() is None
    print('FLAML report, trace and package ready:',archive,archive.stat().st_size)

if __name__=='__main__':main()
