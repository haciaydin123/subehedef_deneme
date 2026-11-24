"""Run a versioned selection study or apply its frozen schema to a new month."""
import copy
import json
from pathlib import Path
import time
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
from .schema import load_config,read_input,profile,digest,write_json,write_csv
from .prepare import prepare
from .selection import folds,quality_filter,fit_window,fit_transformer,transform,select_features

def export_inputs(panel,transformer,out):
    keys=['__branch','__period','__group']
    values=transform(panel,transformer);data=pd.concat([panel[keys].reset_index(drop=True),values.reset_index(drop=True)],axis=1)
    write_csv(out/'cluster_input_panel.csv',data)
    latest=data[data.__period.eq(data.__period.max())].copy()
    write_csv(out/'cluster_input_latest.csv',latest)
    return data,latest

def drift(frame,transformer):
    records=[]
    for name in transformer['features']:
        spec=transformer['columns'][name];s=frame[name]
        row={'name':name,'missing_fraction':float(s.isna().mean())}
        if spec['dtype']=='numeric':
            median=float(s.median()) if s.notna().any() else None
            row.update(median=median,training_median=spec['median'],
                median_shift_in_training_iqr=abs(median-spec['median'])/spec['iqr'] if median is not None else None)
        else:row['unseen_category_fraction']=float((s.notna()&~s.isin(spec['levels'])).mean())
        records.append(row)
    return records

def run(config_path,out=None,as_of=None,baseline=None):
    c=load_config(config_path);raw=read_input(c);panel,fields,missing,as_of=prepare(raw,c,as_of)
    clean={k:v for k,v in c.items() if not k.startswith('_')}
    config_hash=__import__('hashlib').sha256(json.dumps(clean,sort_keys=True).encode()).hexdigest()
    input_hash=digest(c['_source_path']);rules=c['selection'];windows,end=folds(panel,as_of,rules)
    code_paths=sorted(Path(__file__).parent.glob('*.py'))
    code_hashes={p.name:digest(p) for p in code_paths}
    code_signature=__import__('hashlib').sha256(json.dumps(code_hashes,sort_keys=True).encode()).hexdigest()
    run_id=f'{as_of}_{input_hash[:8]}_{config_hash[:8]}_{code_signature[:8]}'
    out=Path(out).resolve() if out else Path(__file__).resolve().parents[2]/'outputs/dinamik_feature_selection'/run_id
    if out.exists():raise FileExistsError(f'Use another output folder; existing run is preserved: {out}')
    out.mkdir(parents=True);started=time.perf_counter()
    write_json(out/'config.json',clean);write_json(out/'schema_profile.json',profile(raw,c))
    write_csv(out/'schema_profile.csv',pd.DataFrame(profile(raw,c)))
    write_csv(out/'analysis_panel.csv',panel)
    candidates=[f for f in fields if f['role']=='candidate'];targets=[f for f in fields if f['role']=='target']
    eligible,decisions=quality_filter(windows[0][1],candidates,rules['quality'])
    print(f'{len(candidates)} declared candidates; {len(eligible)} pass TRAIN-only controls; {len(windows)} windows; {len(targets)} targets',flush=True)
    all_importance=[];all_predictions=[];metrics=[];models=[]
    (out/'models').mkdir()
    for name,train,valid in windows:
        for target in targets:
            bundle,pred,importance,metric=fit_window(train,valid,target,eligible,rules,name)
            file=f'{name}__{target["name"]}.joblib';joblib.dump(bundle,out/'models'/file,compress=3)
            models.append({'file':file,'window':name,'target':target['name']})
            all_importance+=importance;all_predictions.append(pred);metrics.append(metric)
            print(f'{name}/{target["name"]}: WAPE={metric["wape"]:.3%}; informative={metric["informative"]}',flush=True)
    decisions,window_scores=select_features(decisions,all_importance,metrics,rules)
    for f in c['fields']:
        if f['role']=='candidate' and f['name'] in missing:
            decisions.append({'name':f['name'],'label':f.get('label',f['name']),'family':f.get('family','other'),
                'dtype':f['dtype'],'status':'missing_source','reason':'İsteğe bağlı kaynak alanı bu dosyada yok'})
        elif f['role']!='candidate':
            decisions.append({'name':f['name'],'label':f.get('label',f['name']),'family':f.get('family','other'),
                'dtype':f['dtype'],'status':'output_only' if f['role']=='target' else 'review_role' if f['role']=='review' else 'excluded_role',
                'reason':f.get('reason',f'Rol: {f["role"]}')})
    mapped=set(c['input']['keys'].values())|{f['source'] for f in c['fields']}
    for col in raw.columns:
        if col not in mapped:decisions.append({'name':col,'label':col,'family':'unmapped','dtype':'unknown',
               'status':'review_unmapped','reason':'Alan sözlüğünde tanımsız; otomatik modele alınmadı'})
    selected=[d['name'] for d in decisions if d['status']=='selected']
    selected_fields=[f for f in eligible if f['name'] in selected]
    # Freeze imputation/category vocabulary on the earliest training window only.
    transformer=fit_transformer(windows[0][1],selected_fields)
    schema={'version':1,'run_id':run_id,'as_of':as_of,'selection_end':end,'algorithm':'catboost',
        'purpose':'branch_clustering_candidates','features':selected,'fields':selected_fields,
        'transformer':transformer,'config':clean,'status':'candidate' if selected else 'review_required',
        'clustering_executed':False,'automatically_replaces_active_schema':False}
    write_json(out/'selected_schema.json',schema)
    exported,latest=export_inputs(panel,transformer,out)
    latest_raw=panel[panel.__period.eq(panel.__period.max())]
    comparison={'baseline_run_id':None,'added':selected,'removed':[],'dtype_changes':[]}
    if baseline:
        previous=json.loads(Path(baseline).read_text());before=set(previous['features'])
        comparison={'baseline_run_id':previous['run_id'],'added':sorted(set(selected)-before),'removed':sorted(before-set(selected)),
            'dtype_changes':[n for n in set(selected)&before if transformer['columns'][n]['dtype']!=previous['transformer']['columns'][n]['dtype']]}
    write_json(out/'schema_comparison.json',comparison);write_json(out/'drift.json',drift(latest_raw,transformer))
    write_json(out/'decisions.json',decisions);write_csv(out/'decisions.csv',pd.DataFrame(decisions).drop(columns=['window_scores'],errors='ignore'))
    write_csv(out/'importance_repeats.csv',pd.DataFrame(all_importance));write_csv(out/'window_scores.csv',pd.DataFrame(window_scores))
    write_csv(out/'validation_predictions.csv',pd.concat(all_predictions,ignore_index=True));write_csv(out/'model_metrics.csv',pd.DataFrame(metrics))
    write_json(out/'model_index.json',models)
    manifest={'version':1,'run_id':run_id,'as_of':as_of,'selection_end':end,'source_path':c['_source_path'],
        'input_sha256':input_hash,'config_sha256':config_hash,'algorithm':'catboost','purpose':schema['purpose'],
        'candidate_count':len(candidates),'eligible_count':len(eligible),'selected_count':len(selected),
        'models':len(models),'targets':[{'name':t['name'],'label':t.get('label',t['name'])} for t in targets],
        'groups':sorted(panel.__group.unique().tolist()),'feature_count_is_dynamic':True,'unknown_fields_quarantined':True,
        'selection_rows':len(panel[panel.__label_period.le(end)]),'panel_rows':len(panel),'latest_branch_count':len(latest),
        'quality_fit_label_end':windows[0][1].__label_period.max(),'holdout_months':rules['holdout_months'],
        'software':{n:importlib.metadata.version(n) for n in ['catboost','pandas','numpy','joblib']},
        'code_hashes':code_hashes,'checks':{},'run_seconds':time.perf_counter()-started}
    manifest['artifact_hashes']={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()}
    write_json(out/'manifest.json',manifest)
    from .verify import verify
    verify(out)
    from .report import report
    report(out)
    write_json(out.parent/'latest.json',{'run_id':run_id,'path':str(out),'status':schema['status'],'active_schema_changed':False})
    print(f'COMPLETE: {out}; selected={len(selected)}',flush=True)
    return out

def apply_schema(schema_path,input_path,out,as_of=None):
    saved=json.loads(Path(schema_path).read_text());out=Path(out).resolve()
    if not saved['features']:raise ValueError('Cannot apply empty candidate schema')
    if out.exists():raise FileExistsError(f'Output already exists: {out}')
    out.mkdir(parents=True);c=copy.deepcopy(saved['config']);c['_source_path']=str(Path(input_path).resolve())
    # Inference needs no production labels; only selected feature sources are mandatory.
    selected_sources={f['source'] for f in saved['fields']}
    for f in c['fields']:
        if f['role']=='target':f['role']='excluded'
        if f['role']=='candidate':f['required']=f['source'] in selected_sources
    raw=read_input(c);write_json(out/'schema_profile.json',profile(raw,c))
    panel,_,_,as_of=prepare(raw,c,as_of)
    latest=panel[panel.__period.eq(as_of)]
    for name in saved['features']:
        if name not in latest or latest[name].isna().mean()>c['selection']['quality']['max_missing_fraction']:
            raise ValueError(f'Frozen feature missing/too incomplete in current month: {name}')
    export_inputs(panel,saved['transformer'],out)
    write_json(out/'drift.json',drift(latest,saved['transformer']))
    write_json(out/'manifest.json',{'applied_schema_run_id':saved['run_id'],'as_of':as_of,
          'input_sha256':digest(input_path),'features':saved['features'],'refitted':False,'clustering_executed':False})
    print(f'Frozen schema applied without refitting: {out}')
    return out
