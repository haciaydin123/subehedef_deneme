"""Publish local reports for the single shared FLAML algorithm policy."""
import csv
import json
from pathlib import Path
import shutil
from zipfile import ZipFile,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/flaml_ortak_v4'
NAMES={'lgbm':'LightGBM','rf':'Random Forest','catboost':'CatBoost'}
TARGETS={'tl_vadesiz':'TL vadesiz','tl_vadeli':'TL vadeli','kredi':'Kredi'}

def main():
    m=json.loads((OUT/'manifest.json').read_text());assert all(m['checks'].values()) and m['checks']['global_selection_recomputed_from_validation']
    report=json.loads((OUT/'sonuclar/rapor_verisi.json').read_text());report['manifest']=m
    decision=report['global_selection'];learner=m['selected_learner'];history=[];updates=[]
    for meta in report['model_metadata']:
        best=float('inf');cell=f'{meta["target"]}__{meta["group"]}'
        for r in map(json.loads,(OUT/'aramalar'/f'{cell}.jsonl').read_text().splitlines()):
            if 'learner' not in r:continue
            row={'gosterge':meta['target'],'sube_grubu':meta['group'],'record_id':r['record_id'],
                 'algoritma':r['learner'],'arama_saniye':r['wall_clock_time'],'deneme_saniye':r['trial_time'],
                 'valid_mae_tl':r['validation_loss'],'valid_wape':r['logged_metric']['validation_wape'],
                 'config_json':json.dumps(r['config'],ensure_ascii=False)}
            history.append(row)
            if r['learner']==learner and row['valid_wape']<best:
                best=row['valid_wape'];updates.append({**row,'en_iyi_valid_wape':best})
    with (OUT/'sonuclar/arama_gecmisi.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
    report['search_updates']=updates
    (OUT/'sonuclar/rapor_verisi.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    lines=['# FLAML — bütün göstergelerde tek algoritma','',f'**Ortak seçilen algoritma: {NAMES[learner]}.**',
      'TL vadesiz, TL vadeli ve kredide; İstanbul/Ankara/İzmir ve diğer illerde aynı algoritma kullanılır.',
      'Tek algoritma ailesi seçilir. Her gösterge–grup için ayrı eğitim/parametre vardır; tek bir çok çıktılı model nesnesi değildir.',
      '', '## Yalnızca doğrulamaya dayalı ortak seçim','',
      '| Algoritma | TL vadesiz WAPE | TL vadeli WAPE | Kredi WAPE | Eşit ağırlıklı ortalama |',
      '|---|---:|---:|---:|---:|']
    for s in decision['algorithm_scores']:
        values=[s['indicator_wape'][t] for t in TARGETS]+[s['mean_indicator_wape']]
        lines.append('| '+NAMES[s['learner']]+' | '+' | '.join(f'%{v*100:.3f}' for v in values)+' |')
    lines+=['','Seçim Temmuz–Eylül 2026 üzerinde yapılır. Önce şube grupları birleştirilerek gösterge WAPE hesaplanır; ardından üç gösterge eşit ağırlıkla ortalanır. Basit referanslar ve eski manuel modeller seçime katılmaz.',
       '', '## Yeniden kullanılan tarihsel test','',
       '| Gösterge | Ortak model WAPE | Ay sonu referansı WAPE |','|---|---:|---:|']
    for target,label in TARGETS.items():
        get=lambda model:next(r['wape'] for r in report['total_metrics'] if r['gosterge']==target and r['model']==model and r['split']=='TEST')
        lines.append(f'| {label} | %{get("flaml_automl")*100:.3f} | %{get("ay_sonu_koru")*100:.3f} |')
    lines+=['','Ekim–Aralık 2026 daha önce incelendi. Bu test tekrar kullanılan karşılaştırmadır; yeni bağımsız test değildir. Finansallar sentetiktir.',
       '', '30 model paketi yeniden yüklenerek doğrulandı. Ortak seçim yalnızca doğrulama tahminlerinden bağımsız hesaplandı. Önceki karışık model sonuçları outputs/flaml_v4 altında tarihsel olarak korunur.']
    (OUT/'SONUCLAR.md').write_text('\n'.join(lines)+'\n')
    method='''# Ortak FLAML algoritması protokolü

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
'''
    (OUT/'METODOLOJI.md').write_text(method)
    for name in m['code_hashes']:shutil.copy2(ROOT/'python_kodlari'/name,OUT/'kod'/name)
    shutil.copy2(OUT/'requirements.txt',OUT/'kod/requirements.txt')
    archive=ROOT/'outputs/flaml_ortak_v4_paketi.zip'
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,Path(OUT.name)/p.relative_to(OUT))
    with ZipFile(archive) as z:assert z.testzip() is None
    print('Common FLAML report and package:',archive)

if __name__=='__main__':main()
