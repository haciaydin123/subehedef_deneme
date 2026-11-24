"""Reference-period selection, frozen assignment and separately labelled refits."""
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path
import shutil
import time
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import silhouette_score,adjusted_rand_score
from threadpoolctl import threadpool_limits
from .models import METHODS,fit_geometry,normalize,gower,fit_base,fit_partition,predict

ROOT=Path(__file__).resolve().parents[2]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    def convert(x):
        if isinstance(x,np.generic):return x.item()
        if isinstance(x,np.ndarray):return x.tolist()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=convert),encoding='utf-8')
def csv(path,rows):pd.DataFrame(rows).to_csv(path,index=False,encoding='utf-8-sig',float_format='%.17g')
def seed_for(seed,group):return int(hashlib.sha256(f'{seed}/{group}'.encode()).hexdigest()[:8],16)%1000000000
def group_key(group):return re.sub(r'[^A-Za-z0-9_-]','_',str(group))[:40]+'_'+hashlib.sha256(str(group).encode()).hexdigest()[:8]

def partition_score(labels,common,k,min_size):
    counts=np.bincount(np.asarray(labels,int),minlength=k);unique=len(np.unique(labels))
    silhouette=float(silhouette_score(common,labels,metric='precomputed')) if 1<unique<len(labels) else None
    return {'common_gower_silhouette':silhouette,'min_cluster_size':int(counts.min()),
       'max_cluster_size':int(counts.max()),'cluster_sizes':counts.tolist(),'distinct_clusters':unique,
       'size_feasible':bool(unique==k and counts.min()>=min_size)}

def stability(left,right):
    common=left[['__branch','cluster']].merge(right[['__branch','cluster']],on='__branch',suffixes=('_a','_b'),validate='one_to_one')
    if len(common)<2:return {'shared_branches':len(common),'ari':None,'aligned_switch_fraction':None}
    a,b=common.cluster_a.to_numpy(int),common.cluster_b.to_numpy(int)
    contingency=pd.crosstab(a,b);rr,cc=linear_sum_assignment(-contingency.to_numpy())
    matched=int(contingency.to_numpy()[rr,cc].sum())
    return {'shared_branches':len(common),'ari':float(adjusted_rand_score(a,b)),
            'aligned_switch_fraction':1-matched/len(common)}

def assignment_rows(frame,labels,group,method,mode,model_id):
    return [{'__branch':r['__branch'],'__period':r['__period'],'__group':group,'method':method,
       'mode':mode,'cluster':int(label),'cluster_id':f'{group}__C{int(label)+1}','model_id':model_id}
       for r,label in zip(frame.to_dict('records'),labels)]

def run(source,config_path,out=None):
    started=time.perf_counter();source=Path(source).resolve();config=json.loads(Path(config_path).read_text())
    schema=json.loads((source/'selected_schema.json').read_text());selection=json.loads((source/'manifest.json').read_text())
    if not selection['checks'] or not all(selection['checks'].values()):raise ValueError('Feature selection must be independently verified')
    if not schema['features']:raise ValueError('No selected features')
    for name in ['selected_schema.json','cluster_input_panel.csv']:
        if sha(source/name)!=selection['artifact_hashes'][name]:raise ValueError('Stale selected schema or panel')
    if config['primary_method']!='urf_kmedoids' or config['methods'][0]!='urf_kmedoids':raise ValueError('Original URF/PAM baseline must remain primary and run first')
    if any(x not in METHODS for x in config['methods']):raise ValueError('Unknown method')
    if config['min_cluster_size']<2 or not all(isinstance(k,int) and k>1 for k in config['k_values']):raise ValueError('Invalid cluster-size/k settings')
    if 'mtry_ratio' in config and not 0<config['mtry_ratio']<=1:raise ValueError('mtry_ratio must be in (0,1]')
    categories={f['name']:str for f in schema['fields']+schema.get('evaluation_schema',schema)['fields'] if f['dtype']=='categorical'}
    panel=pd.read_csv(source/'cluster_input_panel.csv',dtype={'__branch':str,'__period':str,'__group':str,**categories})
    if panel.duplicated(['__branch','__period']).any():raise ValueError('Each branch must be one row per month')
    if not set(schema['features'])<=set(panel):raise ValueError('Schema feature absent')
    if panel[schema['features']].isna().any().any():raise ValueError('Apply selected schema preprocessing before clustering')
    for f in schema['fields']:
        if f['dtype']=='numeric' and not np.isfinite(panel[f['name']].to_numpy(float)).all():raise ValueError('Non-finite feature')
    as_of=panel.__period.max();reference=config.get('reference_period') or schema['selection_end']
    if reference<schema['selection_end'] or reference>=as_of:raise ValueError('Reference must follow feature selection and precede monitoring')
    ref=panel[panel.__period.eq(reference)].sort_values(['__group','__branch']).reset_index(drop=True)
    if ref.empty:raise ValueError('No reference snapshot')
    periods=sorted(p for p in panel.__period.unique() if reference<p<=as_of)[-config['evaluation_months']:]
    groups=sorted(ref.__group.unique())
    if config.get('k_by_group') and set(config['k_by_group'])!=set(groups):raise ValueError('k_by_group must map every reference group')
    evaluation_schema=schema.get('evaluation_schema',schema)
    if not set(evaluation_schema['features'])<=set(panel):raise ValueError('Common evaluation fields absent')
    if set(panel[panel.__period.isin(periods)].__group)-set(groups):raise ValueError('New group has no reference fit')
    code_hashes={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    signature=hashlib.sha256(json.dumps({'schema':sha(source/'selected_schema.json'),'panel':sha(source/'cluster_input_panel.csv'),
       'settings':config,'code':code_hashes},sort_keys=True).encode()).hexdigest()[:12]
    run_id=f'{as_of}_{signature}';out=Path(out).resolve() if out else ROOT/'outputs/kumeleme_karsilastirma'/run_id
    if out.exists():raise FileExistsError(f'Use a new output folder: {out}')
    for folder in ['models','input','distances','code']:(out/folder).mkdir(parents=True,exist_ok=True)
    shutil.copy2(source/'selected_schema.json',out/'input/selected_schema.json');shutil.copy2(source/'cluster_input_panel.csv',out/'input/panel.csv')
    save(out/'config.json',config)
    candidates=[];selected=[];models=[];audit=[];assignments=[];sensitivity=[];monthly=[];profiles=[];drift=[];not_applicable=[]
    def persist(bundle,frame,group,method,kind,tag,train_ids):
        file=f'{group_key(group)}__{method}__{tag}.joblib';model_id=file.removesuffix('.joblib')
        labels=predict(bundle,frame);joblib.dump(bundle,out/'models'/file,compress=3)
        models.append({'model_id':model_id,'file':file,'group':group,'method':method,'kind':kind,
            'fit_period':bundle['fit_period'],'evaluation_period':str(frame.__period.iloc[0]),'train_branch_ids':list(train_ids)})
        audit.extend(assignment_rows(frame,labels,group,method,kind,model_id))
        return model_id,labels
    for group in groups:
        reference_frame=ref[ref.__group.eq(group)].sort_values('__branch').reset_index(drop=True)
        group_seed=seed_for(config['seed'],group);geometry=fit_geometry(reference_frame,schema)
        evaluation_geometry=fit_geometry(reference_frame,evaluation_schema)
        a,b=normalize(reference_frame,evaluation_geometry);common=gower(a,b,a,b);np.fill_diagonal(common,0)
        np.save(out/'distances'/f'{group_key(group)}__common_gower.npy',common)
        sample_sets=[np.sort(np.random.default_rng(group_seed+1000+r).choice(len(reference_frame),max(3,int(len(reference_frame)*config['subsample_fraction'])),replace=False)) for r in range(config['subsample_repeats'])]
        for method in config['methods']:
            tick=time.perf_counter()
            if method=='kprototypes' and (not geometry['numeric'] or not geometry['categorical']):
                not_applicable.append({'group':group,'method':method,'reason':'K-Prototypes requires both numeric and categorical fields'});continue
            base,native=fit_base(method,reference_frame,geometry,group_seed,config)
            base_seconds=time.perf_counter()-tick
            if native is not None:np.save(out/'distances'/f'{group_key(group)}__{method}.npy',native)
            options=[]
            k_values=[config['k_by_group'][group]] if config.get('k_by_group') else config['k_values']
            for k in k_values:
                if k>=len(reference_frame):continue
                bundle=fit_partition(base,native,k,group_seed,config)
                model_id,labels=persist(bundle,reference_frame,group,method,'candidate',f'k{k}',reference_frame.__branch)
                score=partition_score(labels,common,k,config['min_cluster_size'])
                row={'group':group,'method':method,'k':k,**score,'native_objective':bundle['native_objective'],
                   'converged':bundle['converged'],'base_fit_seconds':base_seconds,'partition_fit_seconds':bundle['partition_fit_seconds'],
                   'model_id':model_id,'model_file':model_id+'.joblib','urf_oob_accuracy':base.get('urf_oob_accuracy'),
                   'encoded_feature_count':base.get('encoded_feature_count'),'effective_mtry':base.get('effective_mtry'),
                   'mtry_basis':base.get('mtry_basis'),'common_evaluation_feature_count':len(evaluation_schema['features'])}
                candidates.append(row);options.append((row,bundle,labels))
            if not options:raise ValueError('No candidate k values fit group size')
            valid=[v for v in options if v[0]['size_feasible'] and v[0]['converged'] and v[0]['common_gower_silhouette'] is not None]
            pool=valid or [v for v in options if v[0]['distinct_clusters']>1 and v[0]['common_gower_silhouette'] is not None]
            if not pool:raise ValueError(f'No nondegenerate partition: {group}/{method}')
            chosen,bundle,labels=max(pool,key=lambda v:(v[0]['common_gower_silhouette'],-v[0]['k']))
            k=chosen['k'];chosen={**chosen,'selection_status':'size_rule_passed' if valid else 'exploratory_size_review',
                 'selection_period':reference,'primary':method==config['primary_method']}
            selected.append(chosen)
            print(f'REFERENCE {group}/{method}: k={k}, sizes={chosen["cluster_sizes"]}, common silhouette={chosen["common_gower_silhouette"]:.3f}',flush=True)
            # Persist the reference-only decision before any monitoring-month model fits.
            save(out/'reference_decisions.json',selected)
            baseline=pd.DataFrame({'__branch':reference_frame.__branch,'cluster':labels})
            assignments.extend(assignment_rows(reference_frame,labels,group,method,'frozen',chosen['model_id']))
            for r in range(config['seed_repeats']):
                seed=group_seed+2000+r;start=time.perf_counter()
                temp,d=fit_base(method,reference_frame,geometry,seed,config);fitted=fit_partition(temp,d,k,seed,config)
                mid,lab=persist(fitted,reference_frame,group,method,'seed',f'seed{r}',reference_frame.__branch)
                sensitivity.append({'group':group,'method':method,'case':'seed','repeat':r,'model_id':mid,
                   'train_size':len(reference_frame),'seed':seed,'seconds':time.perf_counter()-start,
                   **stability(baseline,pd.DataFrame({'__branch':reference_frame.__branch,'cluster':lab}))})
            for r,indices in enumerate(sample_sets):
                sample=reference_frame.iloc[indices].reset_index(drop=True);seed=group_seed+3000+r;start=time.perf_counter()
                temp,d=fit_base(method,sample,geometry,seed,config);fitted=fit_partition(temp,d,k,seed,config)
                mid,lab=persist(fitted,reference_frame,group,method,'subsample',f'sub{r}',sample.__branch)
                sensitivity.append({'group':group,'method':method,'case':'subsample','repeat':r,'model_id':mid,
                   'train_size':len(sample),'seed':seed,'seconds':time.perf_counter()-start,
                   **stability(baseline,pd.DataFrame({'__branch':reference_frame.__branch,'cluster':lab}))})
            previous_frozen=baseline;previous_refit=baseline
            for period in periods:
                frame=panel[panel.__group.eq(group)&panel.__period.eq(period)].sort_values('__branch').reset_index(drop=True)
                if len(frame)<=k:raise ValueError('Monitoring group too small for selected k')
                frozen=predict(bundle,frame);assignments.extend(assignment_rows(frame,frozen,group,method,'frozen',chosen['model_id']))
                start=time.perf_counter();temp,d=fit_base(method,frame,geometry,group_seed,config)
                fitted=fit_partition(temp,d,k,group_seed,config)
                mid,refit=persist(fitted,frame,group,method,'monthly_refit',f'refit_{period}',frame.__branch)
                assignments.extend(assignment_rows(frame,refit,group,method,'monthly_refit',mid))
                current=pd.DataFrame({'__branch':frame.__branch,'cluster':frozen});current_refit=pd.DataFrame({'__branch':frame.__branch,'cluster':refit})
                an,ac=normalize(frame,evaluation_geometry);current_distance=gower(an,ac,an,ac);np.fill_diagonal(current_distance,0)
                for mode,lab,left,right in [('frozen',frozen,previous_frozen,current),('monthly_refit',refit,previous_refit,current_refit)]:
                    monthly.append({'group':group,'method':method,'period':period,'mode':mode,
                       **partition_score(lab,current_distance,k,config['min_cluster_size']),**stability(left,right),
                       'ari_vs_reference':stability(baseline,right)['ari'],'monthly_refit_seconds':time.perf_counter()-start if mode=='monthly_refit' else None})
                previous_frozen=current;previous_refit=current_refit
            # Profiles remain in original units, using current frozen assignments.
            for period in [reference,as_of]:
                frame=panel[panel.__group.eq(group)&panel.__period.eq(period)].sort_values('__branch').reset_index(drop=True)
                lab=predict(bundle,frame)
                for cluster in range(k):
                    block=frame.iloc[np.flatnonzero(lab==cluster)]
                    for f in schema['fields']:
                        s=block[f['name']]
                        if f['dtype']=='numeric':
                            profiles.append({'group':group,'method':method,'period':period,'cluster':cluster,'cluster_id':f'{group}__C{cluster+1}',
                                'n':len(block),'feature':f['name'],'dtype':'numeric','median':float(s.median()) if len(s) else None,
                                'mean':float(s.mean()) if len(s) else None,'p25':float(s.quantile(.25)) if len(s) else None,'p75':float(s.quantile(.75)) if len(s) else None,
                                'representative_branch_id':bundle['representative_branch_ids'][cluster]})
                        else:
                            for category,share in s.value_counts(normalize=True).items():profiles.append({'group':group,'method':method,'period':period,'cluster':cluster,
                                'cluster_id':f'{group}__C{cluster+1}','n':len(block),'feature':f['name'],'dtype':'categorical','category':str(category),'share':float(share),
                                'representative_branch_id':bundle['representative_branch_ids'][cluster]})
            for period in [reference]+periods:
                frame=panel[panel.__group.eq(group)&panel.__period.eq(period)]
                for name in geometry['numeric']:
                    bounds=geometry['ranges'][name];s=frame[name]
                    drift.append({'group':group,'method':method,'period':period,'feature':name,'ref_low':bounds['low'],'ref_high':bounds['high'],
                       'outside_reference_fraction':float((s.lt(bounds['low'])|s.gt(bounds['high'])).mean())})
    for name,data in {'candidates':candidates,'selected':selected,'model_index':models,'sensitivity':sensitivity,'monthly':monthly,
        'profiles':profiles,'drift':drift,'assignments':assignments,'audit_assignments':audit}.items():
        save(out/(name+'.json'),data);csv(out/(name+'.csv'),data)
    manifest={'run_id':run_id,'reference_period':reference,'as_of':as_of,'monitoring_periods':periods,'groups':groups,
        'features':schema['features'],'feature_count':len(schema['features']),
        'common_evaluation_features':evaluation_schema['features'],'primary_method':config['primary_method'],
        'methods':config['methods'],'feature_selection_run_id':schema['run_id'],'source_path':str(source),
        'source_hashes':{n:sha(source/n) for n in ['selected_schema.json','cluster_input_panel.csv']},
        'config_sha256':sha(config_path),'code_hashes':code_hashes,'selected_models':len(selected),'saved_models':len(models),
        'latest_branch_count':int(panel[panel.__period.eq(as_of)].__branch.nunique()),'not_applicable':not_applicable,
        'reference_selection_only':True,'method_switch_automatic':False,'dea_executed':False,'target_allocation_executed':False,
        'software':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','kmodes','joblib']},
        'seconds_before_verification':time.perf_counter()-started,'checks':{}}
    manifest['artifact_hashes']={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
    save(out/'manifest.json',manifest)
    from .verify import verify
    verify(out)
    from .report import report
    report(out)
    save(out.parent/'latest.json',{'run_id':run_id,'path':str(out),'primary_method':config['primary_method']})
    print(f'COMPLETE: {out}',flush=True)
    return out

def apply(run_path,panel_path,out,period=None):
    run_path=Path(run_path);out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    schema=json.loads((run_path/'input/selected_schema.json').read_text());types={n:str for n,s in schema['transformer']['columns'].items() if s['dtype']=='categorical'}
    frame=pd.read_csv(panel_path,dtype={'__branch':str,'__period':str,'__group':str,**types});period=period or frame.__period.max()
    frame=frame[frame.__period.eq(period)].copy()
    if frame.duplicated('__branch').any() or frame[schema['features']].isna().any().any():raise ValueError('Invalid prepared branch-month input')
    selections=json.loads((run_path/'selected.json').read_text());groups={r['group'] for r in selections}
    if set(frame.__group)-groups:raise ValueError('New group requires a reference model')
    rows=[]
    for item in selections:
        bundle=joblib.load(run_path/'models'/item['model_file']);block=frame[frame.__group.eq(item['group'])].sort_values('__branch')
        if block.empty:continue
        rows+=assignment_rows(block,predict(bundle,block),item['group'],item['method'],'frozen',item['model_id'])
    csv(out/'assignments.csv',rows);save(out/'manifest.json',{'period':period,'reference_run':run_path.name,'refitted':False,'dea_executed':False})
    return out
