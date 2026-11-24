import itertools
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from kumeleme.models import fit_geometry,normalize,gower,onehot,synthetic,pam,fit_base,fit_partition,predict
from kumeleme.study import run,apply,sha,stability,group_key
from kumeleme.parameter_grid import allocate_clusters,feature_subset,run_grid

ROOT=Path(__file__).resolve().parents[2]
def settings():
    c=json.loads((ROOT/'python_kodlari/ayarlar/kumeleme.json').read_text())
    c.update(k_values=[2,3],min_cluster_size=3,starts=2,seed_repeats=1,subsample_repeats=1,evaluation_months=2)
    c['forest'].update(n_estimators=30,max_depth=4,min_samples_leaf=2)
    return c

def source(folder,change_future=False):
    folder.mkdir();rows=[]
    for period in ['2026-03','2026-04','2026-05']:
        for group in ['G/A','G_B']:
            for i in range(14):
                rows.append({'__branch':group+str(i),'__period':period,'__group':group,
                    'x':(i%7)+(30 if i>6 else 0)+(100 if change_future and period>'2026-03' else 0),
                    'other_income':i*2.,'sector':'A' if i<=6 else 'B'})
    pd.DataFrame(rows).to_csv(folder/'cluster_input_panel.csv',index=False)
    fields=[{'name':'x','label':'Yeni sayısal','source':'x','dtype':'numeric','role':'candidate'},
            {'name':'other_income','label':'Gelir','source':'other_income','dtype':'numeric','role':'candidate'},
            {'name':'sector','label':'Sektör','source':'sector','dtype':'categorical','role':'candidate'}]
    schema={'version':1,'run_id':'test-selection','selection_end':'2026-03','as_of':'2026-05','features':[f['name'] for f in fields],
        'fields':fields,'transformer':{'columns':{'x':{'dtype':'numeric'},'other_income':{'dtype':'numeric'},'sector':{'dtype':'categorical','levels':['A','B']}}}}
    (folder/'selected_schema.json').write_text(json.dumps(schema))
    decisions=[{'name':f['name'],'status':'selected','mean_delta_wape':3-i} for i,f in enumerate(fields)]
    (folder/'decisions.json').write_text(json.dumps(decisions))
    manifest={'checks':{'verified':True},'artifact_hashes':{n:sha(folder/n) for n in ['selected_schema.json','cluster_input_panel.csv','decisions.json']}}
    (folder/'manifest.json').write_text(json.dumps(manifest))
    return folder

class ClusteringContracts(unittest.TestCase):
    def test_gower_mixed_values(self):
        n=np.array([[0.],[1.]]);c=np.array([['A'],['B']]);d=gower(n,c,n,c)
        np.testing.assert_allclose(d,[[0,1],[1,0]])
        self.assertEqual(gower(np.array([[.5]]),np.array([['A']]),n,c).tolist(),[[.25,.75]])

    def test_pam_matches_small_bruteforce(self):
        x=np.array([0.,.1,.9,1.]);d=np.abs(x[:,None]-x[None,:]);r=pam(d,2,12,starts=3)
        optimum=min(d[:,list(indices)].min(axis=1).sum() for indices in itertools.combinations(range(4),2))
        self.assertAlmostEqual(r['cost'],optimum);self.assertTrue(r['converged'])
        with self.assertRaises(ValueError):pam(np.ones((4,4)),2,1)

    def test_synthetic_preserves_valid_categorical_rows(self):
        n=np.arange(12).reshape(6,2).astype(float);c=np.array([['A'],['B'],['C'],['A'],['B'],['C']])
        sn,sc=synthetic(n,c,101)
        for j in range(2):np.testing.assert_array_equal(np.sort(n[:,j]),np.sort(sn[:,j]))
        np.testing.assert_array_equal(np.sort(c[:,0]),np.sort(sc[:,0]))
        encoded=onehot(sn,sc,{'categorical':['sector'],'levels':{'sector':['A','B','C','__UNKNOWN__']}})
        np.testing.assert_array_equal(encoded[:,2:].sum(axis=1),np.ones(6))

    def test_ari_and_churn_ignore_label_names(self):
        a=pd.DataFrame({'__branch':['1','2','3','4'],'cluster':[0,0,1,1]})
        b=pd.DataFrame({'__branch':['1','2','3','4'],'cluster':[7,7,3,3]})
        score=stability(a,b);self.assertEqual(score['ari'],1.);self.assertEqual(score['aligned_switch_fraction'],0.)

    def test_geometry_does_not_refit_on_new_month(self):
        frame=pd.DataFrame({'x':[0.,10.],'sector':['A','B']})
        schema={'features':['x','sector'],'transformer':{'columns':{'x':{'dtype':'numeric'},'sector':{'dtype':'categorical','levels':['A','B']}}}}
        g=fit_geometry(frame,schema);n,c=normalize(pd.DataFrame({'x':[100.],'sector':['NEW']}),g)
        self.assertEqual(n[0,0],1.);self.assertEqual(c[0,0],'__UNKNOWN__');self.assertEqual(g['ranges']['x']['high'],10.)

    def test_mtry_uses_encoded_columns_not_original_fields(self):
        frame=pd.DataFrame({'__branch':[str(i) for i in range(24)],'__period':['2026-03']*24,
            'x':np.arange(24,dtype=float),'sector':['A','B','C','D','E','F']*4})
        schema={'features':['x','sector'],'transformer':{'columns':{'x':{'dtype':'numeric'},
            'sector':{'dtype':'categorical','levels':['A','B','C','D','E','F']}}}}
        geometry=fit_geometry(frame,schema);c=settings();c['mtry_ratio']=.25
        c['forest']['max_features']='sqrt'
        base,_=fit_base('urf_kmedoids',frame,geometry,23,c)
        self.assertEqual(base['encoded_feature_count'],9) # 1 numeric + 6 categories + unknown/missing
        self.assertEqual(base['effective_mtry'],2)
        self.assertEqual(base['forest'].get_params()['max_features'],.25)
        self.assertTrue(all(tree.max_features_==2 for tree in base['forest'].estimators_))

    def test_fraction_rounding_and_bank_allocation(self):
        schema={'run_id':'parent','features':[f'x{i}' for i in range(22)],
            'fields':[{'name':f'x{i}'} for i in range(22)],'transformer':{'features':[],
                'columns':{f'x{i}':{'dtype':'numeric'} for i in range(22)}}}
        d=[{'name':f'x{i}','mean_delta_wape':22-i} for i in range(22)]
        self.assertEqual(len(feature_subset(schema,d,.25)['features']),6)
        self.assertEqual(len(feature_subset(schema,d,.5)['features']),11)
        self.assertEqual(allocate_clusters({'big':80,'other':120},5,12),{'big':2,'other':3})
        self.assertEqual(allocate_clusters({'big':80,'other':120},7,12),{'big':3,'other':4})
        with self.assertRaises(ValueError):allocate_clusters({'big':80,'other':120},20,12)

    def test_parameter_grid_preserves_full_common_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=source(root/'source');c=settings();c['feature_selection_ratios']=[.25,.5];c['bank_cluster_counts']=[5,7]
            cp=root/'config.json';cp.write_text(json.dumps(c));out=run_grid(src,cp,root/'grid')
            plan=json.loads((out/'grid_plan.json').read_text());m=json.loads((out/'manifest.json').read_text())
            self.assertEqual(len(plan['cases']),4)
            self.assertTrue(m['checks']['post_encoding_mtry_verified_in_every_urf'])
            self.assertTrue(m['checks']['bank_cluster_totals_verified'])
            for case in plan['cases']:
                report=json.loads((out/case['relative_path']/'report.json').read_text())
                self.assertEqual(report['manifest']['common_evaluation_features'],['x','other_income','sector'])
                self.assertEqual(len(report['manifest']['not_applicable']),2)
            self.assertEqual(len(json.loads((out/'model_index.json').read_text())),m['saved_models'])

    def test_end_to_end_reference_choices_are_future_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);s1=source(root/'source1');s2=source(root/'source2',True)
            c=settings();config=root/'config.json';config.write_text(json.dumps(c))
            first=run(s1,config,root/'first');second=run(s2,config,root/'second')
            a=json.loads((first/'selected.json').read_text());b=json.loads((second/'selected.json').read_text())
            self.assertEqual([(r['group'],r['method'],r['k'],r['cluster_sizes'],r['common_gower_silhouette']) for r in a],
                             [(r['group'],r['method'],r['k'],r['cluster_sizes'],r['common_gower_silhouette']) for r in b])
            for group in ['G/A','G_B']:
                for method in ['urf_kmedoids','gower_kmedoids']:
                    np.testing.assert_array_equal(np.load(first/'distances'/f'{group_key(group)}__{method}.npy'),np.load(second/'distances'/f'{group_key(group)}__{method}.npy'))
            result=apply(first,s1/'cluster_input_panel.csv',root/'application')
            got=pd.read_csv(result/'assignments.csv');expected=pd.read_csv(first/'assignments.csv')
            expected=expected[expected['__period'].eq('2026-05')&expected['mode'].eq('frozen')]
            keys=['__branch','method'];joined=got.merge(expected,on=keys,suffixes=('_got','_expected'),validate='one_to_one')
            self.assertEqual(len(joined),28*3);np.testing.assert_array_equal(joined.cluster_got,joined.cluster_expected)

if __name__=='__main__':unittest.main()
