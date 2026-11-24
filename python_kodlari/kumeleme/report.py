"""Comparison report with URF kept as the baseline, not automatically replaced."""
import json
from pathlib import Path
import shutil
from zipfile import ZipFile,ZIP_DEFLATED
from .models import METHODS
from .study import save

def report(out):
    out=Path(out);m=json.loads((out/'manifest.json').read_text());assert m['checks'] and all(m['checks'].values())
    selected=json.loads((out/'selected.json').read_text());monthly=json.loads((out/'monthly.json').read_text())
    sensitivity=json.loads((out/'sensitivity.json').read_text());config=json.loads((out/'config.json').read_text())
    summary=[]
    def mean(values):
        values=[v for v in values if v is not None]
        return sum(values)/len(values) if values else None
    for r in selected:
        rows=[x for x in monthly if x['group']==r['group'] and x['method']==r['method']]
        sens=[x for x in sensitivity if x['group']==r['group'] and x['method']==r['method']]
        summary.append({**r,'mean_frozen_ari':mean([x['ari'] for x in rows if x['mode']=='frozen']),
           'mean_refit_ari':mean([x['ari'] for x in rows if x['mode']=='monthly_refit']),
           'mean_seed_ari':mean([x['ari'] for x in sens if x['case']=='seed']),
           'mean_subsample_ari':mean([x['ari'] for x in sens if x['case']=='subsample']),
           'mean_frozen_switch_fraction':mean([x['aligned_switch_fraction'] for x in rows if x['mode']=='frozen']),
           'latest_frozen_sizes':next(x['cluster_sizes'] for x in rows if x['mode']=='frozen' and x['period']==m['as_of']),
           'latest_size_rule_passed':next(x['size_feasible'] for x in rows if x['mode']=='frozen' and x['period']==m['as_of'])})
    data={'manifest':m,'settings':config,'summary':summary,'method_names':METHODS,
      'selected_schema':json.loads((out/'input/selected_schema.json').read_text())}
    for name in ['candidates','selected','monthly','sensitivity','profiles','drift','assignments']:
        data[name]=json.loads((out/(name+'.json')).read_text())
    save(out/'report.json',data);save(out/'summary.json',summary)
    lines=['# Şube kümeleme karşılaştırması','',
      '**Ana yöntem URF + K-Medoids olarak korunmuştur.** Alternatifler Gower + K-Medoids ve K-Prototypes. Otomatik yöntem değişimi yapılmadı.',
      f'Referans: {m["reference_period"]}; izleme: {", ".join(m["monitoring_periods"])}; ortak {m["feature_count"]} alan; son ay {m["latest_branch_count"]} şube.',
      'İstanbul/Ankara/İzmir ve diğer iller ayrı çalışılır. Aynı şube kümeleri üç üretim göstergesi için ortaktır.',
      '', '| Grup | Yöntem | k | Referans büyüklükleri | Ortak Gower silhouette | Aylık yeniden kümeleme ARI | Alt örneklem ARI |',
      '|---|---|---:|---|---:|---:|---:|']
    fmt=lambda x:'—' if x is None else f'{x:.3f}'
    for r in summary:lines.append(f'| {r["group"]} | {METHODS[r["method"]]} | {r["k"]} | {r["cluster_sizes"]} | {fmt(r["common_gower_silhouette"])} | {fmt(r["mean_refit_ari"])} | {fmt(r["mean_subsample_ari"])} |')
    lines+=['','Sabit referans atamasındaki kararlılık, merkezler sabit tutulduğu için yüksek olabilir. Ayrı aylık yeniden kümeleme ve alt örneklem sonuçları da incelenmelidir.',
       'Ortak silhouette aynı Gower matrisiyle hesaplanır; Gower geometrisini tercih eder ve tek başına nihai başarı ölçütü değildir.',
       f'Küme başına {config["min_cluster_size"]} şube yalnızca pilot kontrol sınırıdır. DEA girdi/çıktıları henüz kesinleşmedi; bu sınır DEA yeterlilik kanıtı değildir.',
       f'{m["saved_models"]} model dosyası yeniden yüklendi; atamalar, referans seçimi, büyüklükler, ARI ve profiller bağımsız doğrulandı.',
       '', 'Kümeler oluşturuldu; DEA ve banka hedefinin şubelere dağıtımı henüz çalıştırılmadı. Veri sentetiktir; sonuçlar gerçek banka başarısı değildir.']
    (out/'SONUCLAR.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    method=f'''# URF + K-Medoids başlangıcı ve alternatif karşılaştırması

Alanlar selected_schema.json ve onun doğrulanmış aylık panelinden okunur.
Müşteri sayısı/profili, kapasite ve bölgesel adaylar kullanılır. Üretim toplamları,
ay sonu stokları ve gelecek gerçekleşmeler girdide yoktur. Alan listesi/sayısı ve
coğrafi grup adları kodda sabitlenmez. Üç gösterge için aynı kümeler kullanılır.

## Referans ve ortak hazırlık

Özellik seçiminin son etiket ayı {m['reference_period']} referans olarak alınır.
Yalnızca bu ayın her gruptaki şubeleriyle sayısal min/max aralıkları fit edilir.
Kategoriler önceki seçilmiş şemadan gelir. Sayısallar referans min/max ile 0–1'e
getirilir; sonraki ay aralık dışına çıkanlar mesafe hesabında bu sınırlara kırpılır.
Kırpılan oranlar drift dosyasında görünür. Bu tercih bilgi kaybı yaratabilir;
yüksek aralık dışı oranında referans güncellemesi ayrıca değerlendirilmelidir.
Sabit referans alanları sıfır katkılıdır. Değişkenler eşit ağırlıklı; CatBoost
önem puanları uzaklık ağırlığı yapılmaz. Çok alanı olan ailelerin daha fazla
ağırlığı olabileceği iş profillerinde incelenmelidir.

## URF ve PAM

Her orijinal alanın değerleri bağımsız permüte edilerek eşit büyüklükte yapay
şube örnekleri oluşturulur; tekil one-hot sütunları ayrı karıştırılmaz.
Gerçek/yapay sınıfını ayıran RandomForestClassifier eğitilir. Kategoriler sabit
sözlükle one-hot olarak verilir. RF yalnızca benzerlik öğrenir; üretim tahmini değildir.
Referans gerçek şubelerin tüm ağaçlardaki ortak yaprak oranı P hesaplanır.
Uzaklık D=sqrt(1-P). OOB proximity kullanılmaz; tüm ağaç yaklaşımıdır. OOB sınıflama
doğruluğu sadece gerçek/yapay ayrım teşhisidir; küme doğruluğu değildir.
Orman ayarları config.json'dadır; yöntem adayıdır, en iyi orman olduğu iddia edilmez.
Varsa mtry_ratio={config.get('mtry_ratio','tanımlı değil')} RF'ye verilen kodlama
sonrası sütun sayısına uygulanır: max(1, floor(oran × P_encoded)). Gerçek sütun
isimleri ve effective_mtry her native pakette saklanır; her ağacın max_features_
değeri doğrulanır. Kategorik dummy ve sayısal kolonlar bu sayımın içindedir.

PAM: greedy BUILD başlangıcı ve ek rastgele başlangıçlar; her adımda toplam en
yakın medoid uzaklığını azaltan en iyi tek SWAP uygulanır. Gerçek gözlemler medoid
olur. Bu yerel optimizasyondur; genel optimum garantisi yoktur. Medoidler referans
ayının profilleridir; aynı şubenin sonraki ay profiliyle sessizce değiştirilmez.

## Alternatifler

Gower: normalize sayısal mutlak fark ve kategorik eşleşmeme göstergesinin alanlar
üzerinde eşit ağırlıklı ortalaması. Aynı PAM uygulamasına verilir.
K-Prototypes: kmodes paketinin Huang başlangıcı, sayısal karesel fark + gamma
çarpı kategorik eşleşmeme maliyeti. Gamma={config['gamma']}, başlangıç sayısı
{config['starts']}; sayısal ve kategorik ağırlık duyarlılığı henüz ayrıca aranmadı.
Yalnızca sayısal veya yalnızca kategorik şemada bu aday uygulanamaz olarak raporlanır.
K-Prototypes temsilci şubesi prototipe en yakın gerçek şubedir; prototipin kendisi
gerçek şube olmak zorunda değildir. Sayısal/kategorik olmayan algoritmayla gizli
bir ikame yapılmaz.

## k ve boyut kontrolü

Genel k listesi {config['k_values']}. Bu senaryoda özel grup eşlemesi varsa
yalnızca o grubun k değeri denenir: {config.get('k_by_group','özel eşleme yok')}.
Referans ayın aynı
Gower mesafesiyle silhouette hesaplanır. Yöntemlerin farklı kendi uzaklıklarındaki
silhouette puanları doğrudan karşılaştırılmaz. Ortak ölçü Gower geometrisine
yakın bölünmeleri tercih edebilir; nihai üstünlük kanıtı değildir.
Değerlendirme alanları {len(m.get('common_evaluation_features',m['features']))}
alandır; modelin kullandığı {m['feature_count']} alandan farklı olabilir. Oran
senaryolarında ortak mesafe tam ebeveyn şemasından gelir, model alt kümeden öğrenir.
Önce yakınsayan ve en az {config['min_cluster_size']} üyeli adaylar dikkate alınır,
sonra ortak silhouette en yüksek k seçilir. Eşitlikte küçük k tercih edilir.
Hiçbiri boyut kuralını geçmezse sonuç keşif amaçlı/inceleme gerekli işaretlenir;
boyut koşulu sessizce gevşetilmez. Bu eşik DEA boyutundan türetilmemiş pilot
kuralıdır. Sonraki aylarda küçülen kümeler de raporlanır; k yeniden ayarlanmaz.

## Kararlılık ve aylar

Referans ay seçimi reference_decisions.json'a izleme aylarından önce kaydedilir.
İzleme ayları {m['monitoring_periods']}. İki ayrı değerlendirme:

1. Dondurulmuş orman/medoid/prototiple yeni ayın şubelerini en yakın kümeye atama.
2. Aynı k/ayarlar ve sabit ölçekle her ay yeniden kümeleme (yalnızca teşhis).

Etiket numaralarına duyarsız Adjusted Rand Index (ARI) ve maksimum örtüşme için
Hungarian eşleştirmesi sonrası küme değiştiren şube oranı hesaplanır. Ortak şubeler
üzerinden karşılaştırılır; müşteri devri nedeniyle gerçek profil değişimleri hata
olarak yorumlanmaz. Sabit atama ARI'sının yüksekliği tek başına öğrenme kararlılığı
değildir. Aylık yeniden kümeleme etiketi referans atamasını veya k seçimini değiştirmez.

Ayrıca referansta {config['seed_repeats']} farklı tohum ve {config['subsample_repeats']}
kez %{config['subsample_fraction']*100:.0f} şube alt örneklemiyle yeniden fit edilir;
her model bütün referans şubelerini yeniden atar. Aynı alt örneklem indeksleri
bütün yöntemlerde kullanılır; ölçekler burada da dondurulmuştur. Bu tam preprocessing
belirsizliğini ölçmez. Native modeller, eğitim şube kimlikleri ve atamalar saklanır.

Yöntem değişimi otomatik değildir: ana yöntem URF + K-Medoids kalır. Alternatifler
aynı aday setinde karşılaştırılır. Minimum küme büyüklüğü, iş profilleri, aylık ve
alt örneklem kararlılığı birlikte değerlendirilir. İş grupları ve mesafe varsayımları
DEA için benzer üretim koşullarını garanti etmez. Veri sentetik, izleme ayları daha
önceki deneylerde incelenmiştir; yeni bağımsız gerçek veri testi değildir.

## Kullanım

```bash
python python_kodlari/kumeleme_karsilastir.py run --config python_kodlari/ayarlar/kumeleme.json
python python_kodlari/kumeleme_karsilastir.py check --run CALISMA_KLASORU
python python_kodlari/kumeleme_karsilastir.py apply --run CALISMA_KLASORU --panel YENI_AY_HAZIR_PANEL.csv --output outputs/aylik_atama
```

run varsayılan olarak son doğrulanmış değişken seçimini okur. --selection ile
başka bir seçim klasörü, --output ile yeni çıktı klasörü verilebilir. Eski sonuç
ezilmez. apply öncesinde ham müşteri verisi seçilmiş şemayla şube-ay paneline
hazırlanmalıdır; apply öğrenilmiş uzaklık ve referansları yeniden fit etmez.
Çift anahtar, eksik alan veya referansı olmayan yeni grup hatadır.

Küme profilleri kaynak birimlerindedir. Şube atamaları ve temsilciler gerçek
etiketli kaynak kimliklerini taşır. DEA ve hedef dağıtımı ayrı aşamalardır;
bu çalışma herhangi bir şubeye hedef yazmaz.

Kaynaklar: [URF](https://doi.org/10.1198/106186006X94072),
[PAM](https://stat.ethz.ch/R-manual/R-devel/library/cluster/html/pam.html),
[Gower](https://stat.ethz.ch/R-manual/R-devel/library/cluster/html/daisy.html),
[K-Prototypes](https://github.com/nicodv/kmodes),
[ARI](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html).
'''
    (out/'METODOLOJI.md').write_text(method,encoding='utf-8')
    code=out/'code/kumeleme';code.mkdir(exist_ok=True)
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,code/p.name)
    parent=Path(__file__).parent.parent
    shutil.copy2(parent/'kumeleme_karsilastir.py',out/'code/kumeleme_karsilastir.py')
    requirements=parent/'requirements_kumeleme.txt'
    if not requirements.exists():requirements=parent/'requirements.txt'
    shutil.copy2(requirements,out/'code/requirements.txt')
    archive=out.parent/(out.name+'.zip')
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,Path(out.name)/p.relative_to(out))
    with ZipFile(archive) as z:assert z.testzip() is None
    print(f'Report/package ready: {archive}',flush=True)
