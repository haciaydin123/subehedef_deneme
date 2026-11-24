"""Local report/snapshot and portable code/config bundle, without deployment."""
import json
from pathlib import Path
import shutil
from zipfile import ZipFile,ZIP_DEFLATED
import pandas as pd
from .schema import write_json

def report(out):
    out=Path(out);m=json.loads((out/'manifest.json').read_text());assert m['checks'] and all(m['checks'].values())
    decisions=json.loads((out/'decisions.json').read_text());schema=json.loads((out/'selected_schema.json').read_text())
    config=json.loads((out/'config.json').read_text())
    result={'manifest':m,'decisions':decisions,'selected_schema':schema,
        'metrics':pd.read_csv(out/'model_metrics.csv').to_dict('records'),
        'window_scores':pd.read_csv(out/'window_scores.csv').to_dict('records'),
        'profile':json.loads((out/'schema_profile.json').read_text()),
        'drift':json.loads((out/'drift.json').read_text()),
        'comparison':json.loads((out/'schema_comparison.json').read_text())}
    write_json(out/'report.json',result)
    lines=['# Dinamik değişken seçimi','',f'Çalışma: {m["run_id"]}. Veri sonu: {m["as_of"]}; seçim etiketi sonu: {m["selection_end"]}.',
       f'{m["candidate_count"]} tanımlı aday → {m["eligible_count"]} kalite/korelasyon sonrası aday → **{m["selected_count"]} seçilen alan**.',
       '', 'Amaç müşteri kitlesi/potansiyeli benzer şubeler için ortak kümeleme girdisi hazırlamaktır. Kümeleme, DEA ve hedef dağıtımı çalıştırılmadı.',
       'CatBoost yalnızca geçmiş üretim ilişkisini sınamak ve değişken seçimini desteklemek için kullanılır. Ay sonu bakiye anchor’ı kullanılmaz. Bu önemler önceki 49 alanlı tahmin modelinin önemleri değildir.',
       '', '| Alan | Karar | Ortalama ΔWAPE | Pozitif dönem | Gerekçe |','|---|---|---:|---:|---|']
    for d in sorted(decisions,key=lambda d:(d['status']!='selected',-d.get('mean_delta_wape',0))):
        score=f'{d["mean_delta_wape"]*100:.4f} yüzde puan' if 'mean_delta_wape' in d else '—'
        lines.append(f'| {d["name"]} | {d["status"]} | {score} | {d.get("positive_windows","—")} | {d["reason"]} |')
    (out/'SONUCLAR.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    method=f'''# Dinamik alan sözlüğü ve seçim protokolü

Bu çalışma sözlükten okunan değişken sayısı/adlarıyla çalışır. Kaynak CSV, anahtarlar,
veri düzeyi, alan rolleri, tipleri, toplulaştırma, zamanında bilinirlik ve seçim
eşikleri config.json içindedir. Kaynağı tanımsız alanlar profile/karar raporunda
review_unmapped olarak tutulur; otomatik açıklayıcı alan yapılmaz. Kimlik ve çıktı
kaynakları aday olarak çift eşlenemez. İş anlamı otomatik veri tipinden çıkarılmaz.

Desteklenen düzeyler branch_month, customer_month, customer_branch_month.
Müşteri düzeyinde ortalama, toplam, ağırlıklı ortalama, tekil sayım, kategori payları,
mod ve şubede sabit değer toplulaştırmaları tanımlanabilir. Girdi aylık ve tarihli
olmalıdır. Günlük bakiye/devir muhasebesi bu modülün görevi değildir; müşteri-şube
parçaları doğru gün ağırlıklı aylık katkılarla hazırlanmalıdır. İlan edilen tekil
anahtarlar, şube-grup tutarlılığı, sayılar ve YYYY-MM tarihleri doğrulanır.

Her şubenin t özelliklerine t+1 aylık üretimi tam takvim eşlemesiyle bağlanır;
eksik ayda sonraki gözlem yanlışlıkla bir ay ileri sayılmaz. Availability lag
varsa yalnızca geçmiş takvim ayı alınır. İlgili ay sonunda bilinmeyen alan aday olamaz.

## Seçim

Veri sonu {m['as_of']}; son {m['holdout_months']} etiket ayı seçim dışında.
Seçim sonu {m['selection_end']}. {config['selection']['windows']} genişleyen eğitim
penceresi, her birinde {config['selection']['validation_months']} doğrulama ayı.
Tüm hedeflerde aynı CatBoost ailesi/ayarları kullanılır; model her hedef/dönemde
ayrı eğitilir. Şube gruplarına göre algoritma seçimi yapılmaz.

Kalite/korelasyon kuralları yalnızca en erken eğitim penceresinde fit edilir.
Eksiklik, baskın/sabit değer ve kategori sayısı incelenir. Sayısal Spearman
korelasyonu eşik üstündeyse alan sözlüğü önceliği, ardından ad sırası temsilciyi
belirler. Bu tercih iş kuralıdır; üretim başarısıyla eşdeğer değildir. Üretim
toplamları/kapanışlar ve sözlükte proxy olarak işaretli alanlar X'e alınmaz.
HP/segment alanları V4 örneğinde mevcut banka varlığından türediği için review
rolündedir. Gerçek veride bağımsız potansiyel bilgisi tanımıyla yeniden değerlendirilebilir.

Eksik sayısalların medyanı ve kategori sözlüğü her modelin kendi TRAIN verisinden
öğrenilir. Tanımsız kategoriler ayrı unknown değerine çevrilir. Doğrulama bilgisi
bu parametrelere girmez. Model üretimi doğrudan açıklar; stok ekleyen anchor yoktur.

Permutation importance {config['selection']['permutation_repeats']} tekrar ile
aynı ay/coğrafi grup içinde şubeler arasında karıştırılır. Her hedefte ΔWAPE
hesaplanır; hedeflerin eşit ağırlıklı ortalaması alınır. Karıştırma tekrar std'si
güven aralığı değildir. Pozitif ortalama katkı ve en az
{config['selection']['min_stable_windows']} pozitif dönem gerekir. Güvenilirlik
kontrolünde modellerin eğitim ortalaması referansından yeterince iyi olması
aranır; zayıf model halinde alanlara review_model_weak verilir. must_keep yalnızca
sözlükte açık iş kararı varsa uygulanır; veri kalitesi hatasını geçersiz kılamaz.

Korelasyonlu alanlar tekil önemi paylaşabilir. Bu seçim küme kalitesinin, nedensel
potansiyelin veya adil hedef dağıtımının kanıtı değildir. Kararlar ve temsilciler
raporlanır; seçilmiş set küme kararlılığı/iş profilleriyle ayrıca sınanmalıdır.

## Aylık kullanım

selected_schema.json alanları, sırası, tipleri ve en erken TRAIN'den dondurulmuş
medyan/kategori sözlüğünü taşır. apply komutu yeni ayı yeniden fit etmeden hazırlar.
Gerekli alan kayıpsa durur; yeni sütunları raporlar. Drift raporu eksiklik,
yeni kategori oranı ve eğitim IQR'ına göre medyan kaymasını gösterir; istatistiksel
anlamlılık veya otomatik ret kararı iddia edilmez. run yeni bir aday sürüm çıkarır.
--baseline ile eski şemaya göre eklenen/çıkarılan alanlar raporlanır. Yeni sürüm
aktif şemayı veya önceki tahmin modelini sessizce değiştirmez.

cluster_input_latest.csv tek dönemde şube başına bir satır ve yalnızca seçilmiş
alanları içerir. cluster_input_panel.csv aylık izleme içindir; 26 ayı tek dönem
şubeleri gibi kümelememek gerekir. Kategori alanları metin olarak korunur;
URF/uzaklık yöntemine uygun kodlama sonraki aşamada yapılmalıdır.

V4 verisinin finansalları ve bölgesel alanları sentetiktir. Dışarıda bırakılan
son aylar önceki çalışmalarda görüldü; yeni dokunulmamış test oldukları iddia edilmez.

Bağımsız doğrulama bütün kayıtlı modelleri yükler, tüm permütasyon kayıplarını ve
seçim kararlarını yeniden hesaplar; kaynak/kod/çıktı hashleri ile dondurulmuş
kümeleme dosyalarını kontrol eder. Bu işlem kümeleme çalıştırmaz.
'''
    (out/'METODOLOJI.md').write_text(method,encoding='utf-8')
    code=out/'code/dinamik_secim';code.mkdir(parents=True,exist_ok=True)
    for path in Path(__file__).parent.glob('*.py'):shutil.copy2(path,code/path.name)
    code_parent=Path(__file__).resolve().parent.parent
    shutil.copy2(code_parent/'dinamik_feature_selection.py',out/'code/dinamik_feature_selection.py')
    requirements=code_parent/'requirements_flaml.txt'
    if not requirements.exists():requirements=code_parent/'requirements.txt'
    shutil.copy2(requirements,out/'code/requirements.txt')
    if (Path(__file__).parent/'README.md').exists():shutil.copy2(Path(__file__).parent/'README.md',code/'README.md')
    archive=out.parent/(out.name+'.zip')
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,Path(out.name)/p.relative_to(out))
    with ZipFile(archive) as z:assert z.testzip() is None
    print(f'Report and package: {archive}',flush=True)
