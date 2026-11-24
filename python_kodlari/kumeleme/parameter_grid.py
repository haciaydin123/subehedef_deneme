"""Joint feature-count / bank-wide k scenarios; post-encoding mtry is fixed."""
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
from zipfile import ZipFile,ZIP_DEFLATED
import pandas as pd
from .study import run,sha,save,csv,ROOT
from .models import METHODS

def allocate_clusters(counts,total,min_size):
    if not isinstance(total,int) or total<2*len(counts):raise ValueError('Total clusters must allow at least two clusters per group')
    upper={g:min(n-1,n//min_size) for g,n in counts.items()}
    if any(v<2 for v in upper.values()) or sum(upper.values())<total:raise ValueError('Cluster total is incompatible with group sizes and pilot minimum')
    allocation={g:2 for g in counts};quota={g:total*n/sum(counts.values()) for g,n in counts.items()}
    for _ in range(total-sum(allocation.values())):
        eligible=[g for g in counts if allocation[g]<upper[g]]
        chosen=min(eligible,key=lambda g:(-(quota[g]-allocation[g]),g))
        allocation[chosen]+=1
    return allocation

def feature_subset(schema,decisions,ratio):
    if not 0<ratio<=1:raise ValueError('Feature selection fraction must be in (0,1]')
    lookup={d['name']:d for d in decisions};pool=schema['features']
    if not all(n in lookup and 'mean_delta_wape' in lookup[n] for n in pool):raise ValueError('A verified importance score is required for every parent feature')
    ranked=sorted(pool,key=lambda n:(-lookup[n]['mean_delta_wape'],n))
    n=max(1,math.ceil(len(pool)*ratio));required=[f['name'] for f in schema['fields'] if f.get('must_keep')]
    if len(required)>n:raise ValueError('must_keep features exceed the requested fraction budget')
    names=[x for x in ranked if x in required]+[x for x in ranked if x not in required][:n-len(required)]
    names=sorted(names,key=ranked.index)
    subset=copy.deepcopy(schema);subset['run_id']=schema['run_id']+f'__top{ratio:g}'
    subset['features']=names;fields={f['name']:f for f in schema['fields']};subset['fields']=[fields[n] for n in names]
    subset['transformer']['features']=names;subset['transformer']['columns']={n:schema['transformer']['columns'][n] for n in names}
    subset['evaluation_schema']=copy.deepcopy(schema)
    subset['selection_basis']={'parent_selection_run_id':schema['run_id'],'fraction':ratio,'pool_count':len(pool),
        'selected_count':n,'count_basis':'original_fields_before_encoding','rounding':'ceil','ranked_fields':ranked,
        'ranking_metric':'prior validation mean_delta_wape','mtry_count_basis':'post_encoding_columns'}
    return subset

def bank_scores(cases,config):
    rows=[]
    for case in cases:
        report=json.loads((Path(case['path'])/'report.json').read_text())
        for method in config['methods']:
            values=[r for r in report['summary'] if r['method']==method]
            if len(values)!=len(case['group_counts']):
                rows.append({'case_id':case['case_id'],'feature_ratio':case['feature_ratio'],'bank_clusters':case['bank_clusters'],
                    'method':method,'applicable':False,'size_feasible':False,'reason':'Numeric/categorical requirements are not satisfied',
                    'reference_score':None,'mean_refit_ari':None,'mean_subsample_ari':None});continue
            weights=case['group_counts'];den=sum(weights.values())
            avg=lambda key:sum(r[key]*weights[r['group']] for r in values)/den
            rows.append({'case_id':case['case_id'],'feature_ratio':case['feature_ratio'],'bank_clusters':case['bank_clusters'],
               'method':method,'applicable':True,'size_feasible':all(r['size_feasible'] and r['converged'] for r in values),
               'reference_score':avg('common_gower_silhouette'),'mean_refit_ari':avg('mean_refit_ari'),
               'mean_subsample_ari':avg('mean_subsample_ari'),'encoded_feature_counts':sorted(set(r['encoded_feature_count'] for r in values if r['encoded_feature_count'] is not None)),
               'effective_mtry_values':sorted(set(r['effective_mtry'] for r in values if r['effective_mtry'] is not None))})
    return rows

def verify_grid(out):
    from .verify import verify
    out=Path(out);plan=json.loads((out/'grid_plan.json').read_text());m=json.loads((out/'manifest.json').read_text())
    for name,digest in m['artifact_hashes'].items():assert sha(out/name)==digest,name
    for name,digest in m['code_hashes'].items():assert sha(Path(__file__).parent/name)==digest,name
    for name,digest in m['source_hashes'].items():assert sha(Path(m['source_path'])/name)==digest,name
    parent=json.loads((out/'input/parent_schema.json').read_text());decisions=json.loads((out/'input/parent_decisions.json').read_text())
    for case in plan['cases']:
        path=out/case['relative_path'];schema=json.loads((path/'input/selected_schema.json').read_text())
        expected=feature_subset(parent,decisions,case['feature_ratio'])
        assert schema==expected and len(schema['features'])==math.ceil(len(parent['features'])*case['feature_ratio'])
        assert case['k_by_group']==allocate_clusters(case['group_counts'],case['bank_clusters'],plan['settings']['min_cluster_size'])
        assert sum(case['k_by_group'].values())==case['bank_clusters']
        cm=verify(path)
        assert cm['common_evaluation_features']==parent['features']
        selected=json.loads((path/'selected.json').read_text())
        assert all(r['k']==case['k_by_group'][r['group']] for r in selected)
        assert json.loads((path/'config.json').read_text())['mtry_ratio']==plan['settings']['mtry_ratio']
    scores=bank_scores([{**c,'path':str(out/c['relative_path'])} for c in plan['cases']],plan['settings'])
    eligible=[r for r in scores if r['method']==plan['settings']['primary_method'] and r['applicable'] and r['size_feasible']]
    pool=eligible or [r for r in scores if r['method']==plan['settings']['primary_method'] and r['applicable']]
    winner=max(pool,key=lambda r:(r['reference_score'],-r['bank_clusters'],-r['feature_ratio']))
    assert winner['case_id']==plan['selected_case_id']
    m['checks']={'all_case_native_models_and_metrics_verified':True,'fraction_counts_and_prior_rank_verified':True,
        'post_encoding_mtry_verified_in_every_urf':True,'same_full_schema_common_metric_verified':True,
        'bank_cluster_totals_verified':True,'global_case_uses_reference_primary_urf_only':True,
        'no_automatic_method_switch':True,'no_dea_or_target_allocation':True}
    save(out/'manifest.json',m)
    print(f'PASS: parameter grid, {m["saved_models"]} model bundles, raw field fractions and encoded mtry',flush=True)
    return m

def report_grid(out):
    out=Path(out);plan=json.loads((out/'grid_plan.json').read_text());m=json.loads((out/'manifest.json').read_text())
    assert m['checks'] and all(m['checks'].values())
    cases=[]
    for case in plan['cases']:
        payload=json.loads((out/case['relative_path']/'report.json').read_text())
        cases.append({**case,'report':payload})
    active=next(c for c in cases if c['case_id']==plan['selected_case_id']);data=copy.deepcopy(active['report'])
    data['manifest']=m
    data['parameter_grid']={'cases':cases,'bank_scores':plan['bank_scores'],'selected_case_id':plan['selected_case_id'],
        'selection_status':plan['selection_status'],'mtry_ratio':plan['settings']['mtry_ratio'],
        'parent_feature_count':plan['parent_feature_count'],'parent_selection_run_id':plan['parent_selection_run_id']}
    save(out/'report.json',data)
    lines=['# Değişken oranı ve kodlama sonrası mtry karşılaştırması','',
      f'Ana yöntem URF + K-Medoids. mtry oranı %{plan["settings"]["mtry_ratio"]*100:g}, payda RF’ye verilen kodlama sonrası sütun sayısı.',
      'Değişken seçim oranı ise önceki doğrulama önem sırasındaki orijinal alan sayısına uygulanır. Alan sayısı yukarı, mtry aşağı yuvarlanır; mtry en az 1 olur.',
      f'Üst aday havuzu {plan["parent_feature_count"]} alan. Referans {m["reference_period"]}; aylık izleme {m["monitoring_periods"]}.',
      '', '| Senaryo | Alan oranı | Ham alan | Toplam küme | URF kodlanmış sütun | URF mtry |', '|---|---:|---:|---:|---|---|']
    for c in cases:
        score=next(r for r in plan['bank_scores'] if r['case_id']==c['case_id'] and r['method']=='urf_kmedoids')
        lines.append(f'| {c["case_id"]} | %{c["feature_ratio"]*100:g} | {c["feature_count"]} | {c["bank_clusters"]} | {score["encoded_feature_counts"]} | {score["effective_mtry_values"]} |')
    lines+=['','## Aynı tam alan setiyle ortak referans ölçümü','',
       '| Senaryo | Yöntem | Ortak silhouette | Boyut kuralı | Aylık yeniden kümeleme ARI |','|---|---|---:|---|---:|']
    for r in plan['bank_scores']:
        if not r['applicable']:lines.append(f'| {r["case_id"]} | {METHODS[r["method"]]} | — | Uygulanamaz: karma veri yok | — |');continue
        lines.append(f'| {r["case_id"]} | {METHODS[r["method"]]} | {r["reference_score"]:.3f} | {r["size_feasible"]} | {r["mean_refit_ari"]:.3f} |')
    lines+=['',f'**Referans döneminde URF ile seçilen ortak senaryo: {plan["selected_case_id"]}.**',
       'Bu seçim üç göstergede ve bütün coğrafi gruplarda aynı alan oranı/toplam küme senaryosunu kullanır. Alternatif yönteme otomatik geçiş yoktur.',
       'Bankadaki ortak skor, grup silhouette puanlarının referans şube sayısıyla ağırlıklı ortalamasıdır. Gruplar arası mesafeyi içeren tek bir banka silhouette puanı değildir.',
       'Senaryoların silhouette değerlendirmesi, modelin daha az alan kullanmasından bağımsız olarak ebeveyn şemanın bütün alanlarıyla aynı sabit Gower mesafesine dayanır.',
       'K-Prototypes iki veri tipi de gerektirir. Seçilen üst alanlar yalnızca sayısalsa yöntem uygulanamaz; gizlice başka algoritma kullanılmaz.',
       'Küme başına minimum şube sayısı pilot kuraldır; DEA yeterlilik kanıtı değildir. Veriler sentetiktir. DEA ve hedef dağıtımı çalıştırılmadı.']
    (out/'SONUCLAR.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    method='''# Parametre senaryoları protokolü

Önce doğrulanmış değişken seçiminin ortak alanları kendi geçmiş doğrulama
mean_delta_wape sırasına konur. Her oran için ceil(P × oran) orijinal alan alınır.
must_keep alanlar bu bütçenin içinde korunur; sayıları bütçeyi aşarsa işlem durur.
Bu oran sütun kodlamasından önce uygulanır. mtry ise RF’ye verilen kodlanmış
matrisin gerçek sütun sayısına göre max(1, floor(P_encoded × mtry_ratio)) olur.
Her modelin encoded_columns, encoded_feature_count ve effective_mtry bilgisi saklanır.
Her ağacın max_features_ değeri bağımsız doğrulamada yeniden kontrol edilir.
Bu RF aday bütçesidir; geçerli bölünme ararken kütüphane ek alan inceleyebilir.

Toplam banka küme sayısı coğrafi gruplara referans şube sayılarıyla orantılı
dağıtılır. Her grupta en az iki küme, en fazla floor(n/minimum_küme_boyutu)
olacak şekilde en büyük kota açığı sırayla doldurulur. Bu sayısal kapasite kontrolü,
gerçek kümelerin boyut kuralını sağlayacağını garanti etmez; ölçülen sayılar ayrıca kontrol edilir.

Her alan oranı × banka toplam küme sayısı bir ayrı senaryodur. Üç yöntem aynı
senaryodaki aynı alanları kullanır. K-Prototypes karma alan koşulu yoksa açıkça
uygulanamaz sayılır. Seçim sonrası bir kategorik alan eklenerek oran bozulmaz.

Her grup ve senaryonun ortak silhouette ölçüsü, ebeveyn şemanın TÜM alanlarıyla
referans ayda fit edilmiş aynı Gower mesafesinden gelir. Böylece alan sayısını
azaltarak değerlendirme geometrisi değiştirilmez. Bu yine Gower geometrisini
tercih edebilir ve nihai iş başarısı/DEA kanıtı değildir.

Bankadaki senaryo puanı grup silhouette'larının referans şube sayısıyla ağırlıklı
ortalamasıdır. Yalnızca ana yöntem URF'nin referans puanlarıyla global senaryo
seçilir. Önce tüm gruplarda yakınsama/boyut kuralı aranır; hiçbiri sağlamazsa sonuç
exploratory_size_review olarak açıkça işaretlenir. Eşitlikte küçük toplam küme
sayısı ve ardından küçük alan oranı tercih edilir. İzleme ARI'si bu seçime katılmaz.
Küme senaryosu her gösterge ve grup için ortak, grup içindeki k kota dağılımıdır.

Her alt çalışmanın METODOLOJI.md dosyası URF yapay veri üretimi, PAM, ölçekleme,
tohum/alt örneklem ve aylık yeniden kümeleme detaylarını içerir. Referans öncesi
doğrulama önemleri kullanılır; gelecek aylar seçim için kullanılmaz. Veriler
sentetiktir ve bu dönemler daha önce incelenmiştir; yeni bağımsız test iddiası yoktur.

Önceki tam alan/sqrt çalışması korunur. Bu paket yeni bir parametre deneyidir.
CLI check ve apply, grid kökünde de çalışır; apply referansta seçilen aktif alt
senaryonun dondurulmuş modellerine yönlenir ve yeniden eğitim yapmaz.
'''
    (out/'METODOLOJI.md').write_text(method,encoding='utf-8')
    csv(out/'grid_scores.csv',plan['bank_scores'])
    archive=out.parent/(out.name+'.zip')
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file() and p.suffix!='.zip' and '__pycache__' not in p.parts:z.write(p,Path(out.name)/p.relative_to(out))
    with ZipFile(archive) as z:assert z.testzip() is None
    print(f'Grid report/package ready: {archive}',flush=True)

def run_grid(source,config_path,out=None):
    source=Path(source).resolve();settings=json.loads(Path(config_path).read_text())
    parent=json.loads((source/'selected_schema.json').read_text());pm=json.loads((source/'manifest.json').read_text())
    decisions=json.loads((source/'decisions.json').read_text())
    if not pm['checks'] or not all(pm['checks'].values()):raise ValueError('Unverified parent selection')
    for name in ['selected_schema.json','cluster_input_panel.csv','decisions.json']:
        if sha(source/name)!=pm['artifact_hashes'][name]:raise ValueError('Parent selection artifacts changed')
    types={f['name']:str for f in parent['fields'] if f['dtype']=='categorical'}
    panel=pd.read_csv(source/'cluster_input_panel.csv',dtype={'__branch':str,'__period':str,'__group':str,**types})
    reference=settings.get('reference_period') or parent['selection_end']
    counts=panel[panel.__period.eq(reference)].groupby('__group').size().to_dict()
    code_hashes={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    signature=hashlib.sha256(json.dumps({'settings':settings,'parent':sha(source/'selected_schema.json'),
        'panel':sha(source/'cluster_input_panel.csv'),'decisions':sha(source/'decisions.json'),'code':code_hashes},sort_keys=True).encode()).hexdigest()[:12]
    out=Path(out).resolve() if out else ROOT/'outputs/kumeleme_karsilastirma'/f'{parent["as_of"]}_grid_{signature}'
    if out.exists():raise FileExistsError(out)
    (out/'input').mkdir(parents=True);(out/'cases').mkdir();(out/'sources').mkdir()
    for name,new_name in [('selected_schema.json','parent_schema.json'),('decisions.json','parent_decisions.json')]:shutil.copy2(source/name,out/'input'/new_name)
    save(out/'settings.json',settings);cases=[]
    for ratio in settings['feature_selection_ratios']:
        subset=feature_subset(parent,decisions,ratio);derived=out/'sources'/f'top{ratio*100:g}';derived.mkdir()
        save(derived/'selected_schema.json',subset)
        # All parent fields remain available ONLY for the fixed common evaluation metric.
        shutil.copy2(source/'cluster_input_panel.csv',derived/'cluster_input_panel.csv')
        save(derived/'manifest.json',{'parent_selection_run_id':parent['run_id'],'selection_basis':subset['selection_basis'],
             'checks':{'verified_parent':True,'prior_validation_rank':True,'explicit_fraction_projection':True},
             'artifact_hashes':{n:sha(derived/n) for n in ['selected_schema.json','cluster_input_panel.csv']}})
        for total in settings['bank_cluster_counts']:
            case_id=f'top{ratio*100:g}_k{total}';config=copy.deepcopy(settings)
            config.pop('feature_selection_ratios',None);config.pop('bank_cluster_counts',None)
            config['k_by_group']=allocate_clusters(counts,total,settings['min_cluster_size']);config['k_values']=sorted(set(config['k_by_group'].values()))
            config['forest']['max_features']=settings['mtry_ratio']
            cp=out/'sources'/f'{case_id}.json';save(cp,config)
            print(f'GRID {case_id}: {len(subset["features"])} original fields; k_by_group={config["k_by_group"]}',flush=True)
            path=run(derived,cp,out/'cases'/case_id)
            cases.append({'case_id':case_id,'relative_path':str(path.relative_to(out)),'path':str(path),
                'feature_ratio':ratio,'feature_count':len(subset['features']),'features':subset['features'],
                'bank_clusters':total,'k_by_group':config['k_by_group'],'group_counts':counts})
    scores=bank_scores(cases,settings)
    valid=[r for r in scores if r['method']==settings['primary_method'] and r['applicable'] and r['size_feasible']]
    pool=valid or [r for r in scores if r['method']==settings['primary_method'] and r['applicable']]
    winner=max(pool,key=lambda r:(r['reference_score'],-r['bank_clusters'],-r['feature_ratio']))
    active=next(c for c in cases if c['case_id']==winner['case_id']);active_path=out/active['relative_path']
    plan={'settings':settings,'cases':[{k:v for k,v in c.items() if k!='path'} for c in cases],
       'bank_scores':scores,'selected_case_id':winner['case_id'],'selection_status':'size_rule_passed' if valid else 'exploratory_size_review',
       'parent_selection_run_id':parent['run_id'],'parent_feature_count':len(parent['features'])}
    save(out/'grid_plan.json',plan)
    for name in ['candidates','selected','monthly','sensitivity','profiles','drift','assignments']:
        for ext in ['json','csv']:
            p=active_path/(name+'.'+ext)
            if p.exists():shutil.copy2(p,out/p.name)
    model_index=[];audit=[]
    for case in cases:
        for item in json.loads((Path(case['path'])/'model_index.json').read_text()):
            model_index.append({**item,'case_id':case['case_id'],'model_id':case['case_id']+'::'+item['model_id'],
                'file':str(Path(case['relative_path'])/'models'/item['file'])})
        for item in json.loads((Path(case['path'])/'audit_assignments.json').read_text()):
            audit.append({**item,'case_id':case['case_id'],'model_id':case['case_id']+'::'+item['model_id']})
    save(out/'model_index.json',model_index);csv(out/'model_index.csv',model_index)
    save(out/'audit_assignments.json',audit);csv(out/'audit_assignments.csv',audit)
    active_report=json.loads((active_path/'report.json').read_text());m=copy.deepcopy(active_report['manifest'])
    m.update(run_id=out.name,is_parameter_grid=True,selected_case_id=winner['case_id'],active_case_path=active['relative_path'],
       feature_selection_run_id=parent['run_id'],model_feature_schema_run_id=active_report['manifest']['feature_selection_run_id'],
       source_path=str(source),code_hashes=code_hashes,source_hashes={n:sha(source/n) for n in ['selected_schema.json','cluster_input_panel.csv','decisions.json']},
       saved_models=len(model_index),config_sha256=sha(config_path),selection_status=plan['selection_status'],checks={})
    m['artifact_hashes']={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.suffix!='.zip' and p.name!='manifest.json'}
    save(out/'manifest.json',m);verify_grid(out);report_grid(out)
    save(out.parent/'latest.json',{'run_id':out.name,'path':str(out),'primary_method':settings['primary_method']})
    print(f'GRID COMPLETE: {out}; selected_case={winner["case_id"]}',flush=True)
    return out
