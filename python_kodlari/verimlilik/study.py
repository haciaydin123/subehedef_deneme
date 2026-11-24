"""Dynamic field dictionary, frozen primary clusters and one same-month DEA frontier per cluster."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
import pandas as pd
from .solver import minimum_units,solve_peer_set,validate
from kumeleme_secimi.coverage import assert_spec_covered

ROOT=Path(__file__).resolve().parents[2]
SOURCES=[
 {'title':'Output-oriented CCR formulation (primary research)','url':'https://www.scielo.br/j/pope/a/HGkM4mcBhxfhpLJ8t3H5nRd/?lang=en'},
 {'title':'DEA variable selection and sample-size heuristic (primary research)','url':'https://www.mdpi.com/2073-8994/13/7/1155'},
 {'title':'SciPy HiGHS linear programming documentation','url':'https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html'}]
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    def convert(x):
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,np.ndarray):return x.tolist()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=convert),encoding='utf-8')
def table(path,rows,columns=None):pd.DataFrame(rows,columns=columns).to_csv(path,index=False,encoding='utf-8-sig',float_format='%.17g')


def prepare(source,config,assignments,period):
    keys=config['input']['keys'];fields=config['inputs']+config['outputs']
    names=[f['name'] for f in fields]
    if not config['inputs'] or not config['outputs'] or len(set(names))!=len(names):raise ValueError('Unique, nonempty DEA input/output definitions required')
    if set(names)&{'__branch','__period','__group','cluster_id','cluster','method','mode','model_id'}:raise ValueError('Reserved DEA field name')
    frame=pd.read_csv(source,dtype={keys['branch']:str,keys['period']:str,keys['group']:str})
    required=set(keys.values())|{f['source'] for f in fields}|set(config['input'].get('filters',{}))
    if not required<=set(frame):raise ValueError('Missing source columns: '+str(sorted(required-set(frame))))
    for name,values in config['input'].get('filters',{}).items():frame=frame[frame[name].isin(values)]
    frame=frame[frame[keys['period']].eq(period)].copy()
    frame=frame.rename(columns={keys['branch']:'__branch',keys['period']:'__period',keys['group']:'__group'})
    if frame.empty or frame.duplicated(['__branch','__period']).any() or frame[['__branch','__period','__group']].isna().any().any():raise ValueError('Empty, duplicate or missing branch-month keys')
    if not pd.Series([period]).str.fullmatch(r'\d{4}-(0[1-9]|1[0-2])').all():raise ValueError('Period must be YYYY-MM')
    days=pd.Period(period,freq='M').days_in_month
    derived=frame[['__branch','__period','__group']].copy()
    for field in fields:
        values=pd.to_numeric(frame[field['source']],errors='raise')
        transform=field.get('transform','identity')
        if transform=='divide_by_calendar_days':values=values/days
        elif transform!='identity':raise ValueError('Unknown transform: '+transform)
        derived[field['name']]=values
    selected=pd.DataFrame([r for r in assignments if r['__period']==period and r['mode']=='frozen'])
    if selected.empty or selected.duplicated(['__branch','__period']).any() or selected['method'].nunique()!=1:raise ValueError('Exactly one frozen primary assignment per branch required')
    if set(derived['__branch'])!=set(selected['__branch']):raise ValueError('DEA data and primary assignments must cover exactly the same branches')
    result=selected.merge(derived,on=['__branch','__period','__group'],how='outer',validate='one_to_one',indicator=True)
    if not result['_merge'].eq('both').all():raise ValueError('Source and cluster geography/keys disagree')
    result=result.drop(columns='_merge').sort_values(['__group','cluster_id','__branch']).reset_index(drop=True)
    validate(result[[f['name'] for f in config['inputs']]],result[[f['name'] for f in config['outputs']]])
    return result


def analyze(panel,config):
    inputs=[f['name'] for f in config['inputs']];outputs=[f['name'] for f in config['outputs']];required=minimum_units(len(inputs),len(outputs))
    if panel.empty or panel['__period'].nunique()!=1 or panel.duplicated('__branch').any():raise ValueError('DEA requires the complete one-row-per-branch population for one month')
    sizes=panel.groupby(['__group','cluster_id']).size()
    bad=sizes[sizes<required]
    if len(bad):raise ValueError(f'DEA tam kapsam koşulu sağlanmadı: küme başına en az {required} şube gerekiyor. Küçük kümeler: {bad.to_dict()}. Şubeler elenmedi; kümeleme seçimini yeniden çalıştırın.')
    scores=[];potentials=[];peers=[];certificates=[];clusters=[]
    for (group,cluster),frame in panel.groupby(['__group','cluster_id'],sort=True):
        frame=frame.sort_values('__branch').reset_index(drop=True);ids=frame['__branch'].tolist();n=len(frame);eligible=n>=required
        summary={'group':group,'cluster_id':cluster,'n':n,'required_n':required,'size_rule_passed':eligible,'status':'calculated'}
        X,Y=frame[inputs].to_numpy(float),frame[outputs].to_numpy(float)
        results=solve_peer_set(X,Y,config['tolerance'])
        for i,(branch,result) in enumerate(zip(ids,results)):
            base={'branch':branch,'group':group,'cluster_id':cluster,'period':str(frame['__period'].iloc[0])}
            row={**base,'n_peers':n,**{k:result[k] for k in ['efficiency','phi','radial_growth','status','vrs_efficiency','scale_efficiency','normalized_slack_sum']},
                 'high_growth_flag':result['radial_growth']>config['high_radial_growth_flag']}
            scores.append(row);certificates.append({**base,'peer_branch_order':ids,**result})
            for j,field in enumerate(config['outputs']):
                actual=float(Y[i,j]);projection=float(result['output_projection'][j]);radial=float(result['radial_projection'][j]);gap=projection-actual
                if gap < -config['tolerance']*max(1,actual):raise RuntimeError('DEA output projection below actual')
                potentials.append({**base,'indicator':field['name'],'actual':actual,'radial_projection':radial,'output_slack':float(result['output_slack'][j]),
                                   'frontier_projection':projection,'improvement_amount':max(0,gap),'improvement_ratio':max(0,gap)/actual if actual>0 else None,
                                   'zero_actual':actual==0,'is_budget_target':False})
            for other,weight in zip(ids,result['lambda']):
                if weight>config['tolerance']:peers.append({**base,'peer_branch':other,'lambda':weight,'is_self':branch==other})
        good=[s for s in scores if s['cluster_id']==cluster]
        radial_count=sum(s['status'] in ['strongly_efficient','radially_efficient_with_slack'] for s in good)
        summary.update(mean_efficiency=float(np.mean([s['efficiency'] for s in good])),median_efficiency=float(np.median([s['efficiency'] for s in good])),
                       min_efficiency=min(s['efficiency'] for s in good),strongly_efficient=sum(s['status']=='strongly_efficient' for s in good),
                       radially_efficient_with_slack=sum(s['status']=='radially_efficient_with_slack' for s in good),radially_efficient_share=radial_count/n,
                       discrimination_review=radial_count/n>config['high_radial_efficiency_share_flag'],high_growth_count=sum(s['high_growth_flag'] for s in good))
        clusters.append(summary)
    if len(scores)!=len(panel) or {s['branch'] for s in scores}!=set(panel['__branch']) or len(potentials)!=len(panel)*len(outputs):
        raise RuntimeError('Full DEA coverage failed; no partial score output may be published')
    return {'scores':scores,'potentials':potentials,'peers':peers,'certificates':certificates,'clusters':clusters}


def run(selection,config_path,period=None,output=None):
    selection=Path(selection).resolve();config_path=Path(config_path).resolve();cfg=read(config_path)
    if cfg['model']!='output_oriented_CRS_two_phase' or not cfg['vrs_sensitivity'] or cfg['min_units_policy']!='max(m*s,3*(m+s))' or cfg['insufficient_cluster_policy']!='error':raise ValueError('DEA requires full coverage and insufficient_cluster_policy=error')
    if not 0<cfg['tolerance']<=1e-5:raise ValueError('Invalid numerical tolerance')
    sm=read(selection/'manifest.json');decision=read(selection/'decision.json')
    if not sm['checks'] or not all(sm['checks'].values()) or not decision['selected']:raise ValueError('A verified, qualified primary clustering selection is required')
    if not (selection/'dea_coverage_contract.json').exists():raise ValueError('DEA coverage contract missing; rerun automatic clustering selection first')
    contract=read(selection/'dea_coverage_contract.json');assert_spec_covered(contract,cfg)
    if not sm.get('full_coverage_verified'):raise ValueError('Primary clustering has not passed the full-population coverage check')
    for name,digest in sm['artifact_hashes'].items():
        target=(selection/name).resolve()
        if not target.is_relative_to(selection) or sha(target)!=digest:raise ValueError('Stale cluster selection: '+name)
    source=(config_path.parent/cfg['input']['path']).resolve();period=period or sm['as_of']
    assignments=read(selection/'primary_assignments.json')
    if {r['method'] for r in assignments}!={decision['selected']['method']}:raise ValueError('Primary method mismatch')
    panel=prepare(source,cfg,assignments,period)
    digest=hashlib.sha256((sha(source)+sha(selection/'manifest.json')+json.dumps(cfg,sort_keys=True)+period+''.join(sha(p) for p in sorted(Path(__file__).parent.glob('*.py')))).encode()).hexdigest()[:12]
    out=Path(output) if output else ROOT/'outputs/dea_verimlilik'/f'{period}_{digest}'
    if out.exists():raise FileExistsError(out)
    result=analyze(panel,cfg);out.mkdir(parents=True);save(out/'config.json',cfg)
    panel.to_csv(out/'dea_input.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    # Source subset retained for independent mapping/day-weight verification and portability.
    raw=pd.read_csv(source,dtype={cfg['input']['keys']['branch']:str,cfg['input']['keys']['period']:str})
    raw=raw[raw[cfg['input']['keys']['period']].eq(period)]
    needed=list(dict.fromkeys(list(cfg['input']['keys'].values())+list(cfg['input'].get('filters',{}))+[f['source'] for f in cfg['inputs']+cfg['outputs']]))
    raw[needed].to_csv(out/'source_rows.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    save(out/'source_cluster_assignments.json',[r for r in assignments if r['__period']==period]);save(out/'source_selection_decision.json',decision)
    for name,rows in result.items():save(out/(name+'.json'),rows)
    for name in ['scores','potentials','peers','clusters']:table(out/(name+'.csv'),result[name])
    valid=[r for r in result['scores'] if r['efficiency'] is not None]
    summary={'branches':len(panel),'calculated_branches':len(valid),'withheld_branches':len(panel)-len(valid),'clusters':len(result['clusters']),
             'eligible_clusters':sum(r['size_rule_passed'] for r in result['clusters']),'required_cluster_n':minimum_units(len(cfg['inputs']),len(cfg['outputs'])),
             'mean_efficiency':float(np.mean([r['efficiency'] for r in valid])) if valid else None,
             'median_efficiency':float(np.median([r['efficiency'] for r in valid])) if valid else None,
             'strongly_efficient':sum(r['status']=='strongly_efficient' for r in valid),'radially_efficient_with_slack':sum(r['status']=='radially_efficient_with_slack' for r in valid),
             'high_growth_count':sum(r['high_growth_flag'] for r in valid)}
    save(out/'summary.json',summary);save(out/'sources.json',SOURCES)
    code=out/'code';(code/'verimlilik').mkdir(parents=True)
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,code/'verimlilik'/p.name)
    (code/'kumeleme_secimi').mkdir()
    for name in ['__init__.py','coverage.py']:shutil.copy2(Path(__file__).parents[1]/'kumeleme_secimi'/name,code/'kumeleme_secimi'/name)
    save(out/'dea_coverage_contract.json',contract)
    shutil.copy2(Path(__file__).parents[1]/'dea_verimlilik.py',code/'dea_verimlilik.py')
    shutil.copy2(Path(__file__).parents[1]/'requirements_dea.txt',out/'requirements.txt')
    write_methodology(out,cfg,period,contract['required_min_cluster_size'])
    (out/'SONUCLAR.md').write_text(f"# Küme içi DEA sonuçları\n\nDönem: {period}. Ana kümeleme: {decision['selected']['method']} / {decision['selected']['case_id']}.\n\n"+json.dumps(summary,ensure_ascii=False,indent=2)+"\n\nSkorlar kendi kümesi içindeki göreli karşılaştırmadır; banka genelinde tek bir verimlilik sırası oluşturmaz. Projeksiyonlar Ocak 2027 hedefi veya büyüme tahmini değildir. Hedef dağıtımı yapılmadı.\n",encoding='utf-8')
    manifest={'run_id':out.name,'as_of':period,'source_selection_run_id':sm['run_id'],'source_selection_path':str(selection),
              'source_selection_manifest_sha256':sha(selection/'manifest.json'),'source_data_path':str(source),'source_data_sha256':sha(source),
              'primary_method':decision['selected']['method'],'selected_case_id':decision['selected']['case_id'],'input_names':[f['name'] for f in cfg['inputs']],
              'full_coverage_required':True,'full_coverage_verified':len(valid)==len(panel),'coverage_contract_minimum':contract['required_min_cluster_size'],
              'output_names':[f['name'] for f in cfg['outputs']],'dea_executed':True,'target_allocation_executed':False,'same_month_within_cluster_frontier':True,
              'not_a_forecast':True,'score_aggregation':'descriptive_branch_mean_not_global_frontier_efficiency','summary':summary,
              'software':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy']},'checks':{}}
    manifest['artifact_hashes']={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    save(out/'manifest.json',manifest)
    from .verify import verify
    manifest['checks']=verify(out);save(out/'manifest.json',manifest)
    save(out/'report.json',{'manifest':manifest,'config':cfg,'summary':summary,'scores':result['scores'],'potentials':result['potentials'],
                          'clusters':result['clusters'],'peers':result['peers'],'sources':SOURCES})
    archive=out.parent/(out.name+'.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('DEA package corrupted')
    save(out.parent/'latest.json',{'run_id':out.name,'path':str(out.resolve())})
    print(json.dumps({'path':str(out),'summary':summary,'checks':manifest['checks']},ensure_ascii=False));return out


def write_methodology(out,cfg,period,coverage_minimum):
    inputs='; '.join(f['label']+' ('+f['name']+')' for f in cfg['inputs'])
    text=f'''# Küme içi verimlilik metodolojisi

Dönem {period}. Her şube sadece seçilmiş ana yöntemin aynı ay, aynı coğrafi grup ve aynı kümesindeki şubelerle karşılaştırılır. Şube, DEA karar birimidir; müşteriler tek tek DEA birimi değildir. Girdi/çıktı rolleri feature importance sırasından türetilmez; ayar sözlüğünde açık tanımlanır.

Girdi: {inputs}. Varsayılan müşteri girdisi müşteri-gün / ayın bütün takvim günleridir; ay sonu müşteri sayısı kullanılmaz. Bu tanım müşteri tabanına göre göreli bilanço üretimini ölçer. Müşteri sayısı idari olarak azaltılacak bir maliyet değildir; çıktı odaklı modelde sabit tutulan maruziyet/potansiyel ölçüsüdür. Maliyet, kârlılık veya risk ayarlı verimlilik iddiası yoktur.

Çıktılar aynı ayın TL vadesiz, TL vadeli ve kredi aylık ortalama bakiyeleridir. Pilot hizmet/üretim yaklaşımıyla mevduat ve kredi birlikte çıktı sayılır. Mevduatı fonlama girdisi sayan aracılık yaklaşımı seçilmedi. V4 kredi toplamı KMH dahil ihtiyaç/konut/taşıt alt ürünlerini içerir; kart borcu bu toplamın dışında tutulur. Kaynak alan eşleştirmesi config.json içindedir.

## Matematik

Çıktı odaklı CRS için max φ; Xλ ≤ x₀, Yλ ≥ φy₀, λ ≥ 0. CRS'de toplam λ = 1 koşulu yoktur. Şubenin kendisi referans kümesinde olduğundan φ ≥ 1 sağlanır. Verimlilik E = 1/φ, ortak radyal artış mesafesi φ−1. Örneğin E=0.80 ise ortak radyal mesafe %25'tir; %20 değildir. Sabit getiri varsayımı, girdilerle çıktıların orantılı ölçeklenebileceğini kabul eder.

İkinci doğrusal program φ optimumunu sabitler, medyanla ölçeklenmiş girdi/çıktı boşlukları toplamını maksimize eder. Projeksiyon Yλ = φy₀ + s⁺; ürün bazındaki artış bu projeksiyon ile gerçekleşme arasındaki farktır. İlk aşamanın ortak çarpanı üç göstergede aynıdır; ikinci aşamada ürün boşlukları farklılaşabilir. λ ve projeksiyonlar alternatif optimumlarda tekil olmayabilir. CRS λ katsayıları toplamı 1 olmak zorunda olmadığından, emsaller basit müşteri/portföy transfer planı veya aritmetik ortalama değildir.

Radyal skor 1 olsa bile pozitif boşluk bulunabilir; bu durum radially_efficient_with_slack olarak ayrıca işaretlenir. Yalnız skor 1 ve normalize toplam boşluk tolerans altında ise strongly_efficient denir. Esnek DEA ağırlıkları bir çıktıyı fiilen çok az ağırlıklandırabilir; skor 1 bütün ürünlerde liderlik anlamına gelmez. İş önceliklerine göre ağırlık kısıtları bu ilk koşuda eklenmedi.

CRS ana modelin yanında aynı şube/emsallerle VRS duyarlılığı hesaplanır (ek koşul: toplam λ = 1). CRS skoru VRS skorundan büyük olamaz; E_CRS/E_VRS çıktı yönelimli ölçek verimliliği olarak raporlanır. VRS otomatik ana modele çevrilmez.

## Kontroller ve kullanım sınırı

n ≥ max(m×s, 3(m+s)) bir ayırt etme ön kontrolü olarak uygulanır. Bu literatürdeki pratik kural yeterlilik veya istatistiksel güven garantisi değildir. 1 girdi / 3 çıktı için 12, 2 girdi / 3 çıktı için 15 emsal gerekir. Küçük kümeler oluşursa bütün DEA işlemi durur; bazı şubeleri eleyip kısmi skor yayımlanmaz. Kümeleme seçimi, ana girdi ve tanımlı girdi senaryolarının tamamının boyut gereksinimini önceden karşılamalıdır. Mevcut sözleşme bu senaryolar için en az {coverage_minimum} şube gerektirir. Girdi/çıktı tanımı değişirse kümeleme seçimi yeniden çalıştırılır. Geçmiş aylar ek bağımsız şubeymiş gibi sayılmaz. Geçersiz/eksik/negatif değerler otomatik doldurulmaz. Pozitif girdiler, negatif olmayan ve her şubede en az bir pozitif çıktı gerekir; tamamı sıfır bir çıktı varsa tanımın düzeltilmesi istenir. Sıfır gerçekleşmede yüzde artış null, tutar projeksiyonu açık kalır.

Şubeler aynı ay gözlemleriyle sınır oluşturur; bu bir tahmin modeli veya dokunulmamış test başarısı değildir. Aykırı üretimler DEA sınırını etkileyebilir. %100'ü aşan radyal mesafeler ve radyal etkin şube payı %50'yi aşan kümeler inceleme işareti alır; veri kırpılmaz. Bu bayraklar politika tercihleridir. Farklı kümelerin farklı sınırları olduğundan banka ortalama skoru betimleyicidir; bütün şubeler için tek ortak verimlilik sıralaması iddiası yoktur.

Projeksiyonlar ve artış mesafeleri gelecek ay büyüme tahmini, müşteri talebi veya banka hedefi değildir. Ocak 2027 bütçesindeki %15/%10/%5 oranlar bu aşamada şubelere dağıtılmadı. Risk, mevduat maliyeti, vade yapısı ve uygulanabilirlik sonraki iş incelemesinde ele alınmalıdır. Finansal veri sentetiktir.

Doğrulama her CRS skorunu bağımsız çarpan/dual LP ile yeniden çözer, ilk aşama primal/dual fizibilite ve amaç eşitliğini denetler. İkinci aşama emsal katsayılarıyla girdi/çıktı projeksiyonları ve boşluklar tekrar hesaplanır. VRS LP ayrı kurularak kontrol edilir. Kaynak müşteri-gün/gün dönüşümü, alan eşleştirmeleri ve bir şubenin başka kümeden emsal almaması ayrıca doğrulanır.

## Kaynaklar

'''
    text+='\n'.join('- ['+s['title']+']('+s['url']+')' for s in SOURCES)+'\n'
    (out/'METODOLOJI.md').write_text(text,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__);subs=parser.add_subparsers(dest='command',required=True)
    p=subs.add_parser('run');p.add_argument('--selection');p.add_argument('--config',default=str(ROOT/'python_kodlari/ayarlar/dea_v4.json'));p.add_argument('--period');p.add_argument('--output')
    p=subs.add_parser('verify');p.add_argument('--run',required=True)
    args=parser.parse_args()
    if args.command=='run':run(args.selection or read(ROOT/'outputs/otomatik_kumeleme/latest.json')['path'],args.config,args.period,args.output)
    else:
        from .verify import verify
        print(json.dumps(verify(args.run),ensure_ascii=False))
