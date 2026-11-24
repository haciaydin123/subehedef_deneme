"""Independent artifact readback, partition/ARI metrics and reference-only choices."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score,adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from .models import normalize,gower,onehot,fit_geometry
from .study import sha,save,group_key

def verify(out):
    out=Path(out);m=json.loads((out/'manifest.json').read_text());config=json.loads((out/'config.json').read_text())
    schema=json.loads((out/'input/selected_schema.json').read_text())
    for name,digest in m['artifact_hashes'].items():assert sha(out/name)==digest,name
    for name,digest in m['code_hashes'].items():assert sha(Path(__file__).parent/name)==digest,name
    for name,digest in m['source_hashes'].items():assert sha(Path(m['source_path'])/name)==digest
    types={f['name']:str for f in schema['fields']+schema.get('evaluation_schema',schema)['fields'] if f['dtype']=='categorical'}
    panel=pd.read_csv(out/'input/panel.csv',dtype={'__branch':str,'__period':str,'__group':str,**types})
    audit=pd.read_csv(out/'audit_assignments.csv',dtype={'__branch':str,'__period':str,'__group':str})
    assignments=pd.read_csv(out/'assignments.csv',dtype={'__branch':str,'__period':str,'__group':str})
    index=json.loads((out/'model_index.json').read_text());candidates=json.loads((out/'candidates.json').read_text());selected=json.loads((out/'selected.json').read_text())
    assert len(index)==m['saved_models'] and not assignments.duplicated(['__branch','__period','method','mode']).any()
    assert len(selected)==len(m['groups'])*len(m['methods'])-len(m['not_applicable'])
    assert json.loads((out/'reference_decisions.json').read_text())==selected
    assert all(r['selection_period']==m['reference_period'] for r in selected)
    assert m['primary_method']=='urf_kmedoids' and m['method_switch_automatic'] is False
    bundles={}
    evaluation_schema=schema.get('evaluation_schema',schema)
    evaluation_geometries={}
    for group in m['groups']:
        reference=panel[panel.__period.eq(m['reference_period'])&panel.__group.eq(group)].sort_values('__branch').reset_index(drop=True)
        evaluation_geometries[group]=fit_geometry(reference,evaluation_schema)
        a,b=normalize(reference,evaluation_geometries[group]);expected=gower(a,b,a,b);np.fill_diagonal(expected,0)
        np.testing.assert_allclose(expected,np.load(out/'distances'/f'{group_key(group)}__common_gower.npy'),atol=1e-12,rtol=0)
    for item in index:
        bundle=joblib.load(out/'models'/item['file']);bundles[item['model_id']]=bundle
        assert bundle['method']==item['method'] and bundle['fit_period']==item['fit_period']
        assert set(bundle['geometry']['features'])==set(schema['features'])
        assert bundle['reference_branch_ids']==item['train_branch_ids']
        reference=panel[panel.__period.eq(m['reference_period'])&panel.__group.eq(item['group'])]
        for name,bounds in bundle['geometry']['ranges'].items():
            np.testing.assert_allclose([bounds['low'],bounds['high']],[reference[name].min(),reference[name].max()],atol=1e-8,rtol=0)
        expected=audit[audit.model_id.eq(item['model_id'])].copy()
        frame=expected[['__branch','__period','__group']].merge(panel,on=['__branch','__period','__group'],validate='one_to_one',how='left')
        num,cat=normalize(frame,bundle['geometry'])
        # Independent native prediction, including a separately computed URF proximity.
        if item['method']=='urf_kmedoids':
            encoded=onehot(num,cat,bundle['geometry']);forest=bundle['forest']
            assert bundle['encoded_feature_count']==encoded.shape[1]==forest.n_features_in_==len(bundle['encoded_columns'])
            if 'mtry_ratio' in config:
                expected_mtry=max(1,int(config['mtry_ratio']*encoded.shape[1]))
                assert forest.get_params()['max_features']==config['mtry_ratio']
                assert bundle['effective_mtry']==expected_mtry
                assert all(tree.max_features_==expected_mtry for tree in forest.estimators_)
            leaves=bundle['forest'].apply(onehot(num,cat,bundle['geometry']))
            distance=np.sqrt(np.maximum(0,1-(leaves[:,None,:]==bundle['medoid_leaves'][None,:,:]).mean(axis=2)))
            got=distance.argmin(axis=1)
        elif item['method']=='gower_kmedoids':
            d=np.zeros((len(frame),bundle['k']))
            for j in range(num.shape[1]):d+=np.abs(num[:,j,None]-bundle['medoid_numeric'][None,:,j])
            for j in range(cat.shape[1]):d+=(cat[:,j,None]!=bundle['medoid_categorical'][None,:,j])
            got=d.argmin(axis=1)
        else:
            matrix=np.column_stack([num.astype(object),cat]);got=bundle['prototype_model'].predict(matrix,categorical=list(range(num.shape[1],matrix.shape[1])))
        np.testing.assert_array_equal(got,expected.cluster.to_numpy())
        if item['kind']!='monthly_refit':assert item['fit_period']==m['reference_period']
        else:assert item['fit_period'] in m['monitoring_periods']
    def cluster_metrics(labels,d,k):
        counts=np.bincount(labels.astype(int),minlength=k)
        score=silhouette_score(d,labels,metric='precomputed') if 1<len(np.unique(labels))<len(labels) else None
        return counts,score
    for row in candidates:
        bundle=bundles[row['model_id']];p=audit[audit.model_id.eq(row['model_id'])]
        labels=p.cluster.to_numpy();d=np.load(out/'distances'/f'{group_key(row["group"])}__common_gower.npy')
        counts,score=cluster_metrics(labels,d,row['k'])
        assert counts.tolist()==row['cluster_sizes']
        assert row['size_feasible']==bool(len(np.unique(labels))==row['k'] and counts.min()>=config['min_cluster_size'])
        if score is None:assert row['common_gower_silhouette'] is None
        else:np.testing.assert_allclose(score,row['common_gower_silhouette'],atol=1e-12,rtol=0)
        if row['method']!='kprototypes':
            native=np.load(out/'distances'/f'{group_key(row["group"])}__{row["method"]}.npy')
            assert np.allclose(native,native.T) and np.allclose(np.diag(native),0)
            np.testing.assert_allclose(native[:,bundle['medoid_indices']].min(axis=1).sum(),row['native_objective'],atol=1e-10,rtol=0)
    for row in selected:
        pool=[r for r in candidates if r['group']==row['group'] and r['method']==row['method']]
        good=[r for r in pool if r['size_feasible'] and r['converged'] and r['common_gower_silhouette'] is not None]
        pool=good or [r for r in pool if r['distinct_clusters']>1 and r['common_gower_silhouette'] is not None]
        winner=max(pool,key=lambda r:(r['common_gower_silhouette'],-r['k']))
        assert winner['model_id']==row['model_id']
    def compare(left,right):
        p=left[['__branch','cluster']].merge(right[['__branch','cluster']],on='__branch',suffixes=('_a','_b'),validate='one_to_one')
        if len(p)<2:return None,None,len(p)
        table=pd.crosstab(p.cluster_a,p.cluster_b).to_numpy();a,b=linear_sum_assignment(-table)
        return adjusted_rand_score(p.cluster_a,p.cluster_b),1-table[a,b].sum()/len(p),len(p)
    sensitivity=json.loads((out/'sensitivity.json').read_text())
    for row in sensitivity:
        chosen=next(s for s in selected if (s['group'],s['method'])==(row['group'],row['method']))
        left=audit[audit.model_id.eq(chosen['model_id'])];right=audit[audit.model_id.eq(row['model_id'])]
        ari,switch,n=compare(left,right)
        np.testing.assert_allclose([ari,switch],[row['ari'],row['aligned_switch_fraction']],atol=1e-12,rtol=0)
        assert n==row['shared_branches']
    monthly=json.loads((out/'monthly.json').read_text())
    for selected_row in selected:
        for mode in ['frozen','monthly_refit']:
            previous=assignments[assignments.method.eq(selected_row['method'])&assignments.__group.eq(selected_row['group'])&assignments.__period.eq(m['reference_period'])&assignments['mode'].eq('frozen')]
            reference=previous
            for period in m['monitoring_periods']:
                current=assignments[assignments.method.eq(selected_row['method'])&assignments.__group.eq(selected_row['group'])&assignments.__period.eq(period)&assignments['mode'].eq(mode)]
                row=next(r for r in monthly if (r['group'],r['method'],r['period'],r['mode'])==(selected_row['group'],selected_row['method'],period,mode))
                ari,switch,n=compare(previous,current)
                if ari is None:assert row['ari'] is None
                else:np.testing.assert_allclose([ari,switch],[row['ari'],row['aligned_switch_fraction']],atol=1e-12,rtol=0)
                frame=panel[panel.__group.eq(selected_row['group'])&panel.__period.eq(period)].sort_values('__branch').reset_index(drop=True)
                en,ec=normalize(frame,evaluation_geometries[selected_row['group']]);d=gower(en,ec,en,ec);np.fill_diagonal(d,0)
                num,cat=normalize(frame,bundles[selected_row['model_id']]['geometry'])
                current=current.sort_values('__branch');counts,score=cluster_metrics(current.cluster.to_numpy(),d,selected_row['k'])
                if mode=='frozen':
                    b=bundles[selected_row['model_id']]
                    if b['method']=='urf_kmedoids':
                        leaves=b['forest'].apply(onehot(num,cat,b['geometry']))
                        expected=np.sqrt(np.maximum(0,1-(leaves[:,None,:]==b['medoid_leaves'][None,:,:]).mean(axis=2))).argmin(axis=1)
                    elif b['method']=='gower_kmedoids':expected=gower(num,cat,b['medoid_numeric'],b['medoid_categorical']).argmin(axis=1)
                    else:
                        x=np.column_stack([num.astype(object),cat]);expected=b['prototype_model'].predict(x,categorical=list(range(num.shape[1],x.shape[1])))
                    np.testing.assert_array_equal(expected,current.cluster.to_numpy())
                assert counts.tolist()==row['cluster_sizes']
                if score is not None:np.testing.assert_allclose(score,row['common_gower_silhouette'],atol=1e-12,rtol=0)
                previous=current
    # Independently check every numeric/categorical current/reference profile.
    profiles=json.loads((out/'profiles.json').read_text())
    for r in profiles:
        members=assignments[assignments.method.eq(r['method'])&assignments.__group.eq(r['group'])&assignments.__period.eq(r['period'])&assignments['mode'].eq('frozen')&assignments.cluster.eq(r['cluster'])]
        s=panel[panel.__branch.isin(members.__branch)&panel.__period.eq(r['period'])][r['feature']]
        assert len(s)==r['n']
        if r['dtype']=='numeric' and len(s):np.testing.assert_allclose([s.median(),s.mean(),s.quantile(.25),s.quantile(.75)],
             [r['median'],r['mean'],r['p25'],r['p75']],atol=1e-6,rtol=1e-12)
        elif r['dtype']=='categorical':np.testing.assert_allclose(s.astype(str).eq(r['category']).mean(),r['share'],atol=1e-12,rtol=0)
    m['checks']={'source_code_and_artifact_hashes':True,'all_native_models_reloaded':True,
        'independent_urf_proximity_and_assignments':True,'common_silhouettes_and_size_rules':True,
        'reference_only_k_choices':True,'label_invariant_stability_recomputed':True,
        'monthly_metrics_and_profiles_recomputed':True,'all_indicators_share_clusters':True,
        'dea_and_target_allocation_not_executed':True}
    save(out/'manifest.json',m)
    print(f'PASS: {len(index)} native models; all assignments, k choices, ARI, profiles and hashes',flush=True)
    return m
