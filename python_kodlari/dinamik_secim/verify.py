"""Read back every saved model, permutation loss and frozen selection decision."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .schema import digest,write_json
from .selection import transform

def verify(out):
    out=Path(out);m=json.loads((out/'manifest.json').read_text());c=json.loads((out/'config.json').read_text())
    assert digest(m['source_path'])==m['input_sha256']
    for name,value in m['artifact_hashes'].items():assert digest(out/name)==value,name
    for name,value in m['code_hashes'].items():assert digest(Path(__file__).parent/name)==value,name
    category_types={f['name']:str for f in c['fields'] if f['role']=='candidate' and f['dtype']=='categorical' and f.get('aggregation')!='category_shares'}
    panel=pd.read_csv(out/'analysis_panel.csv',dtype={'__branch':str,'__period':str,'__label_period':str,'__group':str,**category_types})
    assert not panel.duplicated(['__branch','__period']).any()
    assert ((pd.PeriodIndex(panel.__period,freq='M')+1).astype(str)==panel.__label_period).all()
    index=json.loads((out/'model_index.json').read_text());pred=pd.read_csv(out/'validation_predictions.csv',dtype={'__branch':str,'__period':str,'__group':str,'__label_period':str})
    importance=pd.read_csv(out/'importance_repeats.csv');perf=pd.read_csv(out/'model_metrics.csv')
    schema=json.loads((out/'selected_schema.json').read_text());decisions=json.loads((out/'decisions.json').read_text())
    assert len(index)==m['models']==c['selection']['windows']*len(m['targets'])
    assert schema['algorithm']==m['algorithm']=='catboost'
    target_sources={f['source'] for f in c['fields'] if f['role']=='target'}
    assert all(f['role']=='candidate' and f['source'] not in target_sources and not f.get('production_proxy',False) for f in schema['fields'])
    assert schema['features']==[d['name'] for d in decisions if d['status']=='selected']
    eligibility={d['name'] for d in decisions if d['status'] in {'selected','not_selected','review_model_weak'}}
    native_count=0
    for item in index:
        bundle=joblib.load(out/'models'/item['file']);name,target=item['window'],item['target']
        p=pred[pred.window.eq(name)&pred.target.eq(target)].copy()
        assert not p.duplicated(['__branch','__label_period']).any()
        assert bundle['train_label_end']<bundle['validation_start']<=bundle['validation_end']<=m['selection_end']
        assert set(bundle['transformer']['features'])==eligibility
        columns=['__branch','__period','__group','__label_period']
        valid=p[columns].merge(panel,on=columns,how='left',validate='one_to_one').reset_index(drop=True)
        y=valid['__y_'+target].to_numpy(float);np.testing.assert_allclose(y,p.actual,rtol=0,atol=1e-6)
        xv=transform(valid,bundle['transformer']);got=np.maximum(0,bundle['model'].predict(xv,thread_count=2))
        np.testing.assert_allclose(got,p.prediction,rtol=0,atol=1e-6)
        denominator=np.abs(y).sum();base=np.abs(y-got).sum()/denominator
        metric=perf[perf.window.eq(name)&perf.target.eq(target)].iloc[0]
        np.testing.assert_allclose([base,np.abs(y-got).mean()],[metric.wape,metric.mae_tl],rtol=0,atol=1e-6)
        train=panel[panel.__label_period.le(bundle['train_label_end'])&panel['__y_'+target].notna()]
        np.testing.assert_allclose(p.train_mean_prediction,float(train['__y_'+target].mean()),rtol=0,atol=1e-6)
        for field,spec in bundle['transformer']['columns'].items():
            if spec['dtype']=='numeric':np.testing.assert_allclose(spec['median'],train[field].median(),rtol=0,atol=1e-6)
            else:assert spec['levels']==sorted(set(train[field].dropna().astype(str)))
        for repeat in range(c['selection']['permutation_repeats']):
            # Independent construction of the within-month/group shuffling indices.
            rng=np.random.default_rng(bundle['permutation_seed']+repeat);take=np.arange(len(valid))
            for ids in valid.groupby(['__period','__group'],sort=True).indices.values():take[ids]=rng.permutation(ids)
            assert valid.__period.to_numpy()[take].tolist()==valid.__period.tolist()
            assert valid.__group.to_numpy()[take].tolist()==valid.__group.tolist()
            for field in bundle['transformer']['features']:
                changed=xv.copy();changed[field]=xv[field].iloc[take].to_numpy()
                h=np.maximum(0,bundle['model'].predict(changed,thread_count=2))
                delta=float(np.abs(y-h).sum()/denominator-base)
                r=importance[importance.window.eq(name)&importance.target.eq(target)&importance.feature.eq(field)&importance.repeat.eq(repeat)]
                assert len(r)==1
                np.testing.assert_allclose(delta,r.iloc[0].delta_wape,rtol=0,atol=1e-12)
        native_count+=1
    expected_repeats=len(index)*len(eligibility)*c['selection']['permutation_repeats'];assert len(importance)==expected_repeats
    by_target=importance.groupby(['window','target','feature']).delta_wape.mean()
    by_window=by_target.groupby(['window','feature']).mean()
    reliable=perf.groupby('window').informative.sum().ge(c['selection'].get('min_informative_targets',1)).to_dict()
    for d in decisions:
        if d['name'] not in eligibility:continue
        values=by_window.xs(d['name'],level='feature');mean=float(values.mean())
        positive=sum(value>c['selection']['min_delta_wape'] and reliable[win] for win,value in values.items())
        np.testing.assert_allclose(mean,d['mean_delta_wape'],atol=1e-12,rtol=0);assert positive==d['positive_windows']
        expected=d['must_keep'] or (all(reliable.values()) and positive>=c['selection']['min_stable_windows'] and mean>c['selection']['min_delta_wape'])
        assert (d['status']=='selected')==bool(expected)
    for filename in ['cluster_input_panel.csv','cluster_input_latest.csv']:
        exported=pd.read_csv(out/filename,dtype={'__branch':str,'__period':str,'__group':str,
            **{n:str for n,s in schema['transformer']['columns'].items() if s['dtype']=='categorical'}})
        assert exported.columns.tolist()==['__branch','__period','__group']+schema['features']
        expected_panel=panel if filename=='cluster_input_panel.csv' else panel[panel.__period.eq(m['as_of'])]
        expected=transform(expected_panel,schema['transformer']).reset_index(drop=True)
        assert len(exported)==len(expected)
        for name in schema['features']:
            if schema['transformer']['columns'][name]['dtype']=='numeric':np.testing.assert_allclose(exported[name],expected[name],rtol=0,atol=1e-6)
            else:assert exported[name].tolist()==expected[name].tolist()
    m['checks']={'source_code_artifact_hashes':True,'all_models_reloaded':native_count==len(index),
        'all_permutation_losses_recomputed':True,'validation_only_selection_recomputed':True,
        'no_target_or_production_proxy_in_selected_schema':True,'exact_calendar_label_alignment':True,
        'frozen_preprocessing_and_cluster_exports_verified':True,'no_clustering_executed':True}
    write_json(out/'manifest.json',m)
    print(f'PASS: {native_count} native models, {expected_repeats} permutation losses, common selection and frozen exports',flush=True)
    return m
