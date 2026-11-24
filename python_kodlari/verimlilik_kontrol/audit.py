"""Empirical input checks; stress experiments are diagnostic, not confidence intervals."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from verimlilik.solver import radial,scales,minimum_units
from verimlilik.study import prepare,run as run_dea,read,save,sha,table

ROOT=Path(__file__).resolve().parents[2]
NAMES={'customer':'Ortalama aktif müşteri','customer_staff':'Ortalama aktif müşteri + personel'}


def describe_changes(values):
    a=np.asarray(values,float)
    return {'n':len(a),'mean_abs_change':float(np.mean(np.abs(a))) if len(a) else None,
            'p95_abs_change':float(np.quantile(np.abs(a),.95)) if len(a) else None,
            'max_abs_change':float(np.max(np.abs(a))) if len(a) else None}


def correlation(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)<3 or len(np.unique(x))<2 or len(np.unique(y))<2:return {'pearson':None,'spearman':None}
    return {'pearson':float(np.corrcoef(x,y)[0,1]),'spearman':float(spearmanr(x,y).statistic)}


def measured_profile_change(before,after):
    before=np.asarray(before,float);after=np.asarray(after,float)
    zero_to_positive=bool(((before==0)&(after>0)).any())
    relative=np.abs(after[before>0]/before[before>0]-1)
    return (float(relative.max()) if len(relative) else 0.),zero_to_positive


def deterministic_noise(seed,cluster,repeat,n,p,amplitude):
    local=int(hashlib.sha256(f'{seed}/{cluster}/{repeat}'.encode()).hexdigest()[:16],16)
    return np.random.default_rng(local).uniform(-amplitude,amplitude,size=(n,p))


def radial_values(X,Y):
    xn,yn=X/scales(X),Y/scales(Y)
    return np.array([1/radial(xn,yn,i)['phi'] for i in range(len(X))])


def quality(raw,base,additional):
    keys=base['input']['keys'];fields=base['inputs']+additional+base['outputs'];rows=[]
    for field in fields:
        s=pd.to_numeric(raw[field['source']],errors='coerce');arr=s.to_numpy(float);finite=np.isfinite(arr)
        rows.append({'field':field['name'],'source':field['source'],'role':'output' if field in base['outputs'] else 'input_candidate',
                     'n':len(s),'missing_or_non_numeric':int(s.isna().sum()),'nonfinite_nonmissing':int((~finite&~s.isna()).sum()),
                     'zero':int((arr==0).sum()),'negative':int((arr<0).sum()),'unique':int(s.nunique()),
                     'min':float(s[finite].min()) if finite.any() else None,'max':float(s[finite].max()) if finite.any() else None})
    duplicates=int(raw.duplicated([keys['branch'],keys['period']]).sum());missing_keys=int(raw[list(keys.values())].isna().any(axis=1).sum())
    return rows,{'duplicate_branch_months':duplicates,'missing_keys':missing_keys,'rows':len(raw),'periods':int(raw[keys['period']].nunique()),'branches':int(raw[keys['branch']].nunique())}


def compare_same_branches(left,right,tolerance,period):
    a=pd.DataFrame(left);b=pd.DataFrame(right)
    joined=a[['branch','cluster_id','efficiency']].merge(b[['branch','cluster_id','efficiency']],on=['branch','cluster_id'],suffixes=('_base','_aug'),validate='one_to_one')
    common=joined.dropna(subset=['efficiency_base','efficiency_aug']).copy();common['delta']=common.efficiency_aug-common.efficiency_base
    if (common['delta'] < -1e-6).any():raise ValueError('Nested-input DEA efficiency decreased on an identical cohort')
    rows=[{'period':period,**r} for r in common.to_dict('records')]
    return {'period':period,'base_scored':int(a.efficiency.notna().sum()),'augmented_scored':int(b.efficiency.notna().sum()),'common_scored':len(common),
            'base_mean_same_cohort':float(common.efficiency_base.mean()) if len(common) else None,
            'augmented_mean_same_cohort':float(common.efficiency_aug.mean()) if len(common) else None,
            'mean_mechanical_score_increase':float(common['delta'].mean()) if len(common) else None,
            'material_change_count':int((common['delta']>tolerance).sum()),'metric_is_performance_accuracy':False},rows


def temporal_changes(scores,panels,policy):
    records=[];within=[]
    for scenario in NAMES:
        periods=sorted(p for (s,p) in panels if s==scenario)
        for previous,current in zip(periods,periods[1:]):
            a=pd.DataFrame(scores[(scenario,previous)]);b=pd.DataFrame(scores[(scenario,current)])
            merged=a[['branch','cluster_id','efficiency']].merge(b[['branch','cluster_id','efficiency']],on='branch',suffixes=('_prev','_now'),validate='one_to_one').dropna(subset=['efficiency_prev','efficiency_now'])
            old=panels[(scenario,previous)].set_index('__branch');new=panels[(scenario,current)].set_index('__branch')
            features=[c for c in old.columns if c.startswith('audit_value__')]
            for r in merged.to_dict('records'):
                largest,zero_change=measured_profile_change(old.loc[r['branch'],features],new.loc[r['branch'],features])
                stable=bool(features) and not zero_change and largest<=policy['stable_profile_relative_change']
                records.append({'scenario':scenario,'previous_period':previous,'period':current,'branch':r['branch'],'previous_cluster':r['cluster_id_prev'],'cluster_id':r['cluster_id_now'],
                                'cluster_stayer':r['cluster_id_prev']==r['cluster_id_now'],'stable_measured_profile':stable,
                                'max_measured_relative_change':None if zero_change else largest,'zero_to_positive':zero_change,'score_change':r['efficiency_now']-r['efficiency_prev']})
            stay=merged[merged.cluster_id_prev.eq(merged.cluster_id_now)]
            for cluster,block in stay.groupby('cluster_id_now'):
                within.append({'scenario':scenario,'previous_period':previous,'period':current,'cluster_id':cluster,'n':len(block),**correlation(block.efficiency_prev,block.efficiency_now)})
    summary=[]
    for scenario in NAMES:
        for label,predicate in [('all_common',lambda r:True),('cluster_stayers',lambda r:r['cluster_stayer']),('stable_profile_stayers',lambda r:r['cluster_stayer'] and r['stable_measured_profile'])]:
            rows=[r for r in records if r['scenario']==scenario and predicate(r)]
            summary.append({'scenario':scenario,'cohort':label,**describe_changes([r['score_change'] for r in rows])})
    return records,summary,within


def stress(panel,cfg,policy,scenario,baseline_scores):
    features=[f['name'] for f in cfg['inputs']];outputs=[f['name'] for f in cfg['outputs']];canonical=list(dict.fromkeys(features+outputs))
    # Shared named draws: common customer/output fields get identical perturbations across scenarios.
    all_names=sorted({f['name'] for f in cfg['inputs']+cfg['outputs']}|{f['name'] for f in policy['additional_inputs']})
    baseline={r['branch']:r['efficiency'] for r in baseline_scores};local=[];removal=[];cluster_summaries=[]
    for cluster,block in panel.groupby('cluster_id',sort=True):
        block=block.sort_values('__branch').reset_index(drop=True);ids=block['__branch'].tolist();n=len(block);required=minimum_units(len(features),len(outputs))
        if n<required:raise ValueError('Duyarlılık deneyi için tam şube kapsamı sağlanmadı; kısmi deney yayımlanmaz')
        X,Y=block[features].to_numpy(float),block[outputs].to_numpy(float);original=np.array([baseline[x] for x in ids]);measure=block[all_names].to_numpy(float)
        for repeat in range(policy['perturbation_repeats']):
            noise=deterministic_noise(policy['seed'],cluster,repeat,n,len(all_names),policy['relative_perturbation'])
            changed=measure*(1+noise);xx=changed[:,[all_names.index(f) for f in features]];yy=changed[:,[all_names.index(f) for f in outputs]]
            scores=radial_values(xx,yy)
            local.extend({'scenario':scenario,'cluster_id':cluster,'repeat':repeat,'branch':branch,'baseline':float(old),'stressed':float(new),'score_change':float(new-old)} for branch,old,new in zip(ids,original,scores))
        if n-1>=required:
            for removed in range(n):
                keep=np.array([i for i in range(n) if i!=removed]);after=radial_values(X[keep],Y[keep])
                if (after<original[keep]-1e-6).any():raise ValueError('Removing a peer improved the frontier')
                removal.extend({'scenario':scenario,'cluster_id':cluster,'removed_peer':ids[removed],'branch':ids[i],'baseline':float(original[i]),'stressed':float(new),'score_change':float(new-original[i]),'remaining_n':n-1} for i,new in zip(keep,after))
        lr=[r['score_change'] for r in local if r['cluster_id']==cluster]
        rr=[r['score_change'] for r in removal if r['cluster_id']==cluster]
        cluster_summaries.append({'scenario':scenario,'cluster_id':cluster,'n':n,'local':describe_changes(lr),'peer_removal':describe_changes(rr),'peer_removal_size_rule_passed':n-1>=required})
        print(json.dumps({'stage':'sensitivity','scenario':scenario,'cluster':cluster,'local_scores':len(lr),'peer_removal_scores':len(rr)}),flush=True)
    return local,removal,cluster_summaries


def recommend(policy,latest,coverage,stress_summary):
    base=next(c for c in coverage if c['scenario']=='customer' and c['period']==latest)
    aug=next(c for c in coverage if c['scenario']=='customer_staff' and c['period']==latest)
    base_stress=[s for s in stress_summary if s['scenario']=='customer'];review=[]
    if base['coverage']<policy['minimum_global_coverage']:review.append('baseline_coverage_insufficient')
    if any(s['local']['p95_abs_change']>policy['local_score_p95_review'] for s in base_stress):review.append('local_perturbation_review')
    if any(s['peer_removal']['max_abs_change'] is not None and s['peer_removal']['max_abs_change']>policy['peer_removal_max_review'] for s in base_stress):review.append('influential_peer_review')
    if any(not s['peer_removal_size_rule_passed'] for s in base_stress):review.append('peer_removal_test_limited_by_size')
    return {'recommended_scenario':'customer' if base['coverage']==1 else None,'status':'baseline_with_review' if review else 'baseline_supported_in_synthetic_pilot',
            'review_reasons':review,'augmented_coverage':aug['coverage'],'augmented_withheld':aug['withheld_branches'],
            'augmented_global_eligible':aug['coverage']>=policy['minimum_global_coverage'],'main_model_changed':False,
            'higher_efficiency_is_selection_objective':False,'business_roles_inferred_from_data':False,
            'scope':'customer_base_relative_production','real_staff_changes_tested':False,'all_results_reused_historical_synthetic':True}


def run(policy_path,dea_run=None,output=None):
    policy_path=Path(policy_path).resolve();policy=read(policy_path);base=read(policy_path.parent/policy['base_config'])
    if policy['perturbation_repeats']<2 or not 0<policy['relative_perturbation']<.1:raise ValueError('Invalid stress policy')
    dea_run=Path(dea_run or read(ROOT/'outputs/dea_verimlilik/latest.json')['path']).resolve();dm=read(dea_run/'manifest.json')
    for name,digest in dm['artifact_hashes'].items():
        if sha(dea_run/name)!=digest:raise ValueError('Stale main DEA artifact: '+name)
    if not dm['checks'] or not all(dm['checks'].values()) or base!=read(dea_run/'config.json'):raise ValueError('Audited base config must match current verified DEA')
    source=(policy_path.parent/base['input']['path']).resolve()
    if sha(source)!=dm['source_data_sha256']:raise ValueError('DEA input source changed; rerun DEA before this audit')
    selection=Path(dm['source_selection_path']);sm=read(selection/'manifest.json')
    if sha(selection/'manifest.json')!=dm['source_selection_manifest_sha256']:raise ValueError('Stale primary clustering')
    periods=sorted({r['__period'] for r in read(selection/'primary_assignments.json')});latest=dm['as_of'];periods=[p for p in periods if p<=latest]
    code=list(Path(__file__).parent.glob('*.py'));signature=hashlib.sha256((json.dumps(policy,sort_keys=True)+sha(dea_run/'manifest.json')+sha(source)+''.join(sha(p) for p in sorted(code))).encode()).hexdigest()[:12]
    out=Path(output) if output else ROOT/'outputs/dea_girdi_kontrol'/f'{latest}_{signature}'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);(out/'input').mkdir();shutil.copy2(source,out/'input/source_panel.csv');save(out/'policy.json',policy)
    shutil.copy2(dea_run/'scores.json',out/'input/main_dea_scores.json');shutil.copy2(dea_run/'manifest.json',out/'input/main_dea_manifest.json')
    shutil.copy2(selection/'primary_assignments.json',out/'input/primary_assignments.json');save(out/'base_config.json',base)
    raw=pd.read_csv(source,dtype={v:str for v in base['input']['keys'].values()})
    for key,values in base['input'].get('filters',{}).items():raw=raw[raw[key].isin(values)]
    quality_rows,quality_keys=quality(raw,base,policy['additional_inputs']);save(out/'quality.json',{'fields':quality_rows,'keys':quality_keys})
    assignments=read(selection/'primary_assignments.json');configs={};case_index=[];scores={};panels={};coverage=[];correlations=[];contributions=[]
    for scenario in NAMES:
        cfg=deepcopy(base);cfg['input']['path']=str((out/'input/source_panel.csv').resolve())
        if scenario=='customer_staff':cfg['inputs']+=policy['additional_inputs']
        configs[scenario]=cfg;cfg_path=out/(scenario+'_config.json');save(cfg_path,cfg)
        for period in periods:
            folder=out/'cases'/scenario/period
            run_dea(selection,cfg_path,period=period,output=folder)
            report=read(folder/'report.json');case_index.append({'scenario':scenario,'period':period,'path':str(folder.relative_to(out)),'manifest_sha256':sha(folder/'manifest.json')})
            scores[(scenario,period)]=report['scores'];frame=pd.read_csv(folder/'dea_input.csv',dtype={'__branch':str,'__period':str,'__group':str},float_precision='round_trip')
            # All candidates are retained for shared stress draws; model inputs still come only from cfg.
            extra=raw[raw[base['input']['keys']['period']].eq(period)].set_index(base['input']['keys']['branch'])
            for field in policy['additional_inputs']:
                if field['name'] not in frame:frame[field['name']]=frame['__branch'].map(extra[field['source']])
            for field in cfg['inputs']+cfg['outputs']:frame['audit_value__'+field['name']]=frame[field['name']]
            panels[(scenario,period)]=frame
            summary=report['summary'];coverage.append({'scenario':scenario,'period':period,'branches':summary['branches'],'calculated_branches':summary['calculated_branches'],'withheld_branches':summary['withheld_branches'],
                'eligible_clusters':summary['eligible_clusters'],'clusters':summary['clusters'],'required_n':summary['required_cluster_n'],'coverage':summary['calculated_branches']/summary['branches'],
                'mean_efficiency_own_coverage':summary['mean_efficiency'],'strongly_efficient':summary['strongly_efficient'],
                'withheld_cluster_ids':[c['cluster_id'] for c in report['clusters'] if not c['size_rule_passed']]})
            if scenario=='customer':
                for scope,block in [('ALL',frame)]+list(frame.groupby('cluster_id',sort=True)):
                    for field in policy['additional_inputs']:
                        correlations.append({'period':period,'cluster_id':scope,'n':len(block),'base_input':base['inputs'][0]['name'],'additional_input':field['name'],
                            'additional_unique':int(block[field['name']].nunique()),**correlation(block[base['inputs'][0]['name']],block[field['name']])})
            else:
                for cert in read(folder/'certificates.json'):
                    row=frame[frame['__branch'].eq(cert['branch'])].iloc[0];x=np.array([row[f['name']] for f in cfg['inputs']])/cert['input_scales'];v=np.array(cert['dual_v']);parts=v*x
                    for j,field in enumerate(cfg['inputs']):contributions.append({'period':period,'branch':cert['branch'],'cluster_id':cert['cluster_id'],'field':field['name'],
                        'one_optimal_dual_contribution_share':float(parts[j]/parts.sum()),'optimal_duals_may_be_nonunique':True})
    comparisons=[];paired=[]
    for period in periods:
        summary,rows=compare_same_branches(scores[('customer',period)],scores[('customer_staff',period)],policy['material_score_change'],period);comparisons.append(summary);paired+=rows
    temporal,temporal_summary,rank=temporal_changes(scores,panels,policy)
    local=[];removal=[];stress_summary=[]
    for scenario in NAMES:
        a,b,c=stress(panels[(scenario,latest)],configs[scenario],policy,scenario,scores[(scenario,latest)]);local+=a;removal+=b;stress_summary+=c
    staff_static=[]
    for field in policy['additional_inputs']:
        counts=raw.groupby(base['input']['keys']['branch'])[field['source']].nunique()
        staff_static.append({'field':field['name'],'branches':len(counts),'constant_across_all_months':int((counts==1).sum()),'real_staff_movements_available':bool((counts>1).any()),'synthetic':field['source'].endswith('_sentetik')})
    recommendation=recommend(policy,latest,coverage,stress_summary)
    results={'coverage':coverage,'correlations':correlations,'comparisons':comparisons,'paired_scores':paired,'dual_contributions':contributions,'temporal':temporal,
             'temporal_summary':temporal_summary,'temporal_within_cluster_ranks':rank,'local_stress':local,'peer_removal':removal,'stress_summary':stress_summary,'staff_profile':staff_static}
    save(out/'case_index.json',case_index);save(out/'recommendation.json',recommendation)
    for name,rows in results.items():save(out/(name+'.json'),rows)
    for name in ['quality','coverage','correlations','comparisons','paired_scores','dual_contributions','temporal_summary','staff_profile']:
        table(out/(name+'.csv'),quality_rows if name=='quality' else results[name])
    table(out/'stress_summary.csv',[{'scenario':r['scenario'],'cluster_id':r['cluster_id'],'n':r['n'],**{'local_'+k:v for k,v in r['local'].items()},**{'peer_removal_'+k:v for k,v in r['peer_removal'].items()}} for r in stress_summary])
    table(out/'peer_removal.csv',removal);table(out/'local_stress.csv',local)
    save(out/'sources.json',[{'title':'DEA variable selection: adding inputs is not predictive performance','url':'https://www.sciencedirect.com/science/article/pii/S0377221711006011'},
                              {'title':'DEA sensitivity to reference-set changes','url':'https://www.sciencedirect.com/science/article/pii/0377221795000542'}])
    write_report(out,policy,latest,results,recommendation,quality_rows,quality_keys)
    (out/'code/verimlilik_kontrol').mkdir(parents=True);(out/'code/verimlilik').mkdir()
    for directory in ['verimlilik_kontrol','verimlilik']:
        for p in (Path(__file__).parents[1]/directory).glob('*.py'):shutil.copy2(p,out/'code'/directory/p.name)
    (out/'code/kumeleme_secimi').mkdir()
    for name in ['__init__.py','coverage.py']:shutil.copy2(Path(__file__).parents[1]/'kumeleme_secimi'/name,out/'code/kumeleme_secimi'/name)
    shutil.copy2(Path(__file__).parents[1]/'dea_girdi_kontrol.py',out/'code/dea_girdi_kontrol.py')
    shutil.copy2(Path(__file__).parents[1]/'requirements_dea.txt',out/'requirements.txt')
    manifest={'run_id':out.name,'as_of':latest,'source_dea_run_id':dm['run_id'],'source_dea_path':str(dea_run),'source_dea_manifest_sha256':sha(dea_run/'manifest.json'),
        'source_selection_run_id':sm['run_id'],'source_selection_manifest_sha256':sha(selection/'manifest.json'),'source_data_sha256':sha(source),'periods':periods,
        'scenario_names':NAMES,'main_model_changed':False,'dea_latest_pointer_unchanged':read(ROOT/'outputs/dea_verimlilik/latest.json')['path']==str(dea_run),
        'local_stress_records':len(local),'peer_removal_records':len(removal),'checks':{}}
    manifest['artifact_hashes']={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    save(out/'manifest.json',manifest)
    from .verify import verify
    manifest['checks']=verify(out);save(out/'manifest.json',manifest)
    save(out/'report.json',{'manifest':manifest,'policy':policy,'recommendation':recommendation,'quality':{'fields':quality_rows,'keys':quality_keys},'case_index':case_index,
          **{k:results[k] for k in ['coverage','correlations','comparisons','temporal_summary','temporal_within_cluster_ranks','stress_summary','staff_profile']}})
    archive=out.parent/(out.name+'.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(out))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Audit package corrupted')
    save(out.parent/'latest.json',{'run_id':out.name,'path':str(out.resolve())})
    print(json.dumps({'path':str(out),'recommendation':recommendation,'checks':manifest['checks']},ensure_ascii=False),flush=True);return out


def write_report(out,policy,latest,results,recommendation,quality_rows,quality_keys):
    c=next(r for r in results['comparisons'] if r['period']==latest)
    text=f'''# DEA girdi kontrolleri

Ana amaç müşteri tabanına göre üretimdir. Karşılaştırma müşteri girdisine personel eklenmesinin etkisini sınar; farklı amaçtaki personel başına üretim modeli otomatik aday değildir. Çıktılar ve Gower/PAM kümeleri bütün senaryolarda aynıdır. Ana DEA modeli veya küme atamaları değiştirilmedi.

İncelenen aylar daha önce kullanılan sentetik geçmiş verilerdir. Her ayın kendi DEA sınırı yeniden hesaplandı; yeni dokunulmamış test veya gelecek ay tahmini iddiası yoktur. Personel kapasitesi sentetiktir ve bütün şubelerde bütün aylar sabittir; gerçek personel devirleri/iş yükü değişimleri sınanamaz.

## Veri ve kapsam

{quality_keys['rows']} şube-ay, {quality_keys['branches']} şube, {quality_keys['periods']} ay kaynak kalite kontrolünde incelendi. Eksik, sayısal olmayan, negatif ve sıfır değerler ayrı raporlanır. DEA karşılaştırması ana modelin kayıtlı atamalarının bulunduğu aylarda yapılır. Müşteri girdisi müşteri-gün / takvim günüdür.

Bir girdi ve üç çıktı için mevcut ön kontrol sınırı 12, iki girdi ve üç çıktı için 15 şubedir. Bu bir politika olarak uygulanan literatür pratik kuralıdır; yeterlilik teoremi değildir. Küçük küme varsa işlem durur; bazı şubelere null skor verilip kalanlarla devam edilmez. Kümeleme adayları ana girdi ve tanımlı ek girdilerin boyut gereksinimlerini önceden karşılamalıdır. Bu koşul kalite ölçütleriyle birlikte uygulanır; tek amaç şube sayısına uygun kümeler bulmak değildir.

{latest} ayında müşteri senaryosu {c['base_scored']}, müşteri+personel senaryosu {c['augmented_scored']} şubeyi kapsar. Skor farkı yalnızca aynı {c['common_scored']} şubede karşılaştırılır. Bu ortak örneklemde ortalamalar {c['base_mean_same_cohort']:.6f} ve {c['augmented_mean_same_cohort']:.6f}. Fark performans kazanımı değildir: iç içe DEA girdileri skoru mekanik olarak artırabilir. Yüksek skor seçim hedefi değildir.

## Ek bilgi ve duyarlılık

Pearson/Spearman hem banka genelinde hem ayrı kümelerde hesaplanır. Korelasyon tek başına girdinin gereksizliğini veya nedensel katkısını kanıtlamaz. Ek girdinin tek bir optimal dual çözümdeki normalize katkı payı raporlanır; alternatif optimum ağırlıkları olabileceğinden bu pay yapısal önem veya tekil gerçek katkı değildir.

Son ayda kümeler sabit tutularak {policy['perturbation_repeats']} tekrarda girdiler ve çıktılar bağımsız, göreli ±%{policy['relative_perturbation']*100:g} aralığında bozulur. Ortak alanlara iki senaryoda aynı rassal değişim verilir. Sıfır çıktı sıfır kalır. Bu deney ölçüm hatası için yerel stres testidir; bootstrap, örneklem güven aralığı veya kanıtlanmış maksimum kararlılık yarıçapı değildir.

Her kümeden sırayla tek bir emsal çıkarılır; yalnız kalan şubeler yeniden değerlendirilir. Değerlendirilen şube kendi emsal setinde kalır: super-efficiency hesaplanmaz. Kalan bağımsız şube sayısı ön kontrolü sağlamıyorsa deney yapılmaz. Emsal çıkarınca skorun yükselmesi diğer şubenin hatalı olduğunu göstermez; sınırın o gözleme duyarlılığını gösterir. En yüksek etkiye sahip emsaller ayrıca iş incelemesi gerektirir.

Aylık değişimler ayrı raporlanır. Aynı kümede kalan ve seçili girdi/çıktılarının tamamı en fazla %5 değişen şubeler ayrıca gösterilir. Bu gözlemlenen profil kontrolüdür; tam müşteri karmasının sabitliği iddiası yoktur. Müşteri devirleri ve gerçek üretim değişimleri skor değişikliğine yol açabilir; aylık oynama tek başına model hatası sayılmaz. Sıra ilişkileri yalnız aynı küme içinde hesaplanır.

## Öneri kuralı

Kapsama oranı, veri kalitesi ve duyarlılık raporlanır. Ana amaç müşteri tabanı olduğundan mevcut girdiyi koruma önerisi bu iki anlamlı senaryo ile sınırlıdır; bütün olası girdiler arasında optimal seçim değildir. Mevcut bütün şubeler için tek ortak politika hedeflendiğinden kapsama alt sınırı %{policy['minimum_global_coverage']*100:g}'dür. Küme bazında yerel stresin p95 mutlak skor farkı {policy['local_score_p95_review']:.2f}, tek emsal çıkarmada maksimum fark {policy['peer_removal_max_review']:.2f} üstündeyse inceleme işareti konur. Bunlar önceden tanımlanan pilot eşiklerdir, evrensel doğruluk eşikleri değildir. Skor farkları yüzde artış değil [0,1] skor ölçeğinde puandır.

Öneri: {json.dumps(recommendation,ensure_ascii=False)}

## Kaynaklar

- [DEA değişken seçimi ve boyut etkisi](https://www.sciencedirect.com/science/article/pii/S0377221711006011)
- [DEA emsal setine duyarlılık](https://www.sciencedirect.com/science/article/pii/0377221795000542)
'''
    (out/'METODOLOJI.md').write_text(text,encoding='utf-8')
    (out/'SONUCLAR.md').write_text('# DEA girdi kontrol sonuçları\n\n'+json.dumps(recommendation,ensure_ascii=False,indent=2)+'\n\n'+json.dumps(c,ensure_ascii=False,indent=2)+'\n\nAyrıntılar quality, coverage, correlations, comparisons, stress_summary ve temporal_summary dosyalarındadır.\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__);subs=parser.add_subparsers(dest='command',required=True)
    p=subs.add_parser('run');p.add_argument('--config',default=str(ROOT/'python_kodlari/ayarlar/dea_girdi_kontrol.json'));p.add_argument('--dea-run');p.add_argument('--output')
    p=subs.add_parser('verify');p.add_argument('--run',required=True)
    args=parser.parse_args()
    if args.command=='run':run(args.config,args.dea_run,args.output)
    else:
        from .verify import verify
        print(json.dumps(verify(args.run),ensure_ascii=False))
