"""Contracts that matter for new schemas, monthly alignment and leakage."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

from dinamik_secim.cli import default_rules
from dinamik_secim.schema import load_config,profile
from dinamik_secim.prepare import prepare,aggregate
from dinamik_secim.selection import fit_transformer,transform,quality_filter,folds,blocked_take
from dinamik_secim.runner import run,apply_schema

def config():
    rules=default_rules();rules.update(windows=2,validation_months=2,min_train_months=4,
        holdout_months=2,permutation_repeats=2,min_stable_windows=1,min_informative_targets=1)
    rules['catboost'].update(iterations=25,depth=3)
    return {'version':1,'input':{'path':'input.csv','grain':'branch_month',
          'keys':{'branch':'office_code','period':'month','group':'region_class'}},
      'fields':[{'name':'signal','source':'customer_income','role':'candidate','dtype':'numeric',
                 'known_by_period_end':True,'must_keep':True,'priority':1},
                {'name':'job_profile','source':'employment','role':'candidate','dtype':'categorical','known_by_period_end':True},
                {'name':'product_A','source':'output_a','role':'target','dtype':'numeric'},
                {'name':'product_B','source':'output_b','role':'target','dtype':'numeric'}],
       'selection':rules}

def fixture():
    rows=[]
    for j,month in enumerate(pd.period_range('2024-01',periods=16,freq='M').astype(str)):
        for b in range(8):
            signal=10+b*7+j*.2
            rows.append({'office_code':f'{b:03d}','month':month,'region_class':'G1' if b<4 else 'G2',
                'customer_income':signal,'employment':str(b%2),'output_a':100*signal+5*j,'output_b':55*signal+3*j,
                'mystery_new_column':b*j})
    return pd.DataFrame(rows)

class Contracts(unittest.TestCase):
    def test_exact_calendar_join_not_next_observed_row(self):
        raw=fixture();raw=raw[~(raw.office_code.eq('000')&raw.month.eq('2024-02'))]
        p,_,_,_=prepare(raw,config())
        self.assertTrue(np.isnan(p[p['__branch'].eq('000')&p['__period'].eq('2024-01')]['__y_product_A'].iloc[0]))

    def test_duplicate_grain_is_rejected(self):
        raw=fixture()
        with self.assertRaisesRegex(ValueError,'Duplicate'):prepare(pd.concat([raw,raw.iloc[:1]]),config())

    def test_dictionary_addition_without_code_changes(self):
        c=config();c['fields'].append({'name':'any_new_metric','source':'mystery_new_column','dtype':'numeric','role':'candidate','known_by_period_end':True})
        p,fields,_,_=prepare(fixture(),c)
        self.assertIn('any_new_metric',p);self.assertEqual(sum(f['role']=='candidate' for f in fields),3)
        self.assertEqual(next(r for r in profile(fixture(),config()) if r['source']=='mystery_new_column')['status'],'unmapped')

    def test_availability_lag_uses_past_calendar(self):
        c=config();c['fields'][0]['availability_lag_months']=1;p,_,_,_=prepare(fixture(),c)
        january=p[p['__branch'].eq('000')&p['__period'].eq('2024-01')].iloc[0]
        feb=p[p['__branch'].eq('000')&p['__period'].eq('2024-02')].iloc[0]
        self.assertTrue(np.isnan(january.signal));self.assertEqual(feb.signal,10.)

    def test_customer_aggregation_is_explicit(self):
        block=pd.DataFrame({'balance':[100.,300.],'income':[20.,80.],'days':[1.,3.],'job':['A','Z']})
        self.assertEqual(aggregate(block,{'name':'x','source':'balance','aggregation':'sum'})['x'],400.)
        self.assertEqual(aggregate(block,{'name':'x','source':'income','aggregation':'weighted_mean','weight':'days'})['x'],65.)
        shares=aggregate(block,{'name':'job','source':'job','aggregation':'category_shares','levels':['A','B']})
        self.assertEqual(shares['job__A'],.5);self.assertEqual(shares['job____OTHER__'],.5)
        with self.assertRaises(ValueError):aggregate(block,{'name':'x','source':'income','aggregation':'first_exact'})

    def test_customer_month_prepare_is_dynamic(self):
        c=config();c['input']['grain']='customer_month';c['input']['keys']['customer']='customer_code'
        for f in c['fields']:f['aggregation']='mean' if f['dtype']=='numeric' and f['role']=='candidate' else 'sum' if f['role']=='target' else 'category_shares'
        c['fields'][1]['levels']=['0','1']
        raw=pd.DataFrame([{'office_code':'001','month':month,'region_class':'G','customer_code':str(i),
            'customer_income':income,'employment':str(i),'output_a':value,'output_b':value*2}
            for month in ['2026-01','2026-02'] for i,income,value in [(0,20.,100.),(1,80.,300.)]])
        p,fields,_,_=prepare(raw,c)
        self.assertEqual(len(p),2);self.assertEqual(p.signal.tolist(),[50.,50.])
        self.assertEqual(p.product_A.tolist(),[400.,400.]);self.assertEqual(p['job_profile__0'].tolist(),[.5,.5])
        self.assertEqual(p['__y_product_A'].iloc[0],400.)
        self.assertEqual(sum(f['role']=='candidate' for f in fields),5)

    def test_numeric_category_and_unseen_are_stable(self):
        fields=[{'name':'cat','dtype':'categorical'},{'name':'v','dtype':'numeric'}]
        params=fit_transformer(pd.DataFrame({'cat':['0','1'],'v':[1.,3.]}),fields)
        x=transform(pd.DataFrame({'cat':pd.Series([0,2,None],dtype=object),'v':[None,1000.,5.]}),params)
        self.assertEqual(x.cat.tolist(),['0','__UNKNOWN__','__MISSING__']);self.assertEqual(x.v.iloc[0],2.)

    def test_training_only_quality_and_preprocessing(self):
        train=pd.DataFrame({'a':[1.,2.,3.,4.],'b':[2.,4.,6.,8.],'constant':[1.,1.,1.,1.]})
        fields=[{'name':n,'dtype':'numeric','priority':i} for i,n in enumerate(train)]
        retained,decisions=quality_filter(train,fields,default_rules()['quality'])
        self.assertEqual([r['name'] for r in retained],['a'])
        self.assertEqual(next(r for r in decisions if r['name']=='b')['status'],'excluded_correlated')
        params=fit_transformer(train,retained);transform(pd.DataFrame({'a':[99999.,None]}),params)
        self.assertEqual(params['columns']['a']['median'],2.5)

    def test_blocked_permutation_stays_in_period_and_group(self):
        p,_,_,_=prepare(fixture(),config());take=blocked_take(p,123)
        self.assertEqual(p['__period'].iloc[take].tolist(),p['__period'].tolist())
        self.assertEqual(p['__group'].iloc[take].tolist(),p['__group'].tolist())

    def test_semantic_contract_rejects_keys_targets_and_future(self):
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'c.json'
            for alteration in ['key','target','future']:
                c=config()
                if alteration=='key':c['fields'][0]['source']='office_code'
                if alteration=='target':c['fields'][0]['source']='output_a'
                if alteration=='future':c['fields'][0]['known_by_period_end']=False
                file.write_text(json.dumps(c))
                with self.assertRaises(ValueError):load_config(file)

    def test_end_to_end_future_changes_do_not_change_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c=config();cp=root/'config.json';cp.write_text(json.dumps(c));data=root/'input.csv'
            raw=fixture();raw.to_csv(data,index=False)
            first=run(cp,root/'first')
            schema=json.loads((first/'selected_schema.json').read_text())
            self.assertTrue(schema['features']);self.assertEqual(schema['algorithm'],'catboost')
            # The last two held-out label months and current month profiles are not selection inputs.
            changed=raw.copy();mask=changed.month.ge('2025-03')
            changed.loc[mask,['output_a','output_b','customer_income']]*=100
            changed.loc[mask,'employment']='FUTURE_ONLY'
            changed.to_csv(data,index=False);second=run(cp,root/'second')
            self.assertEqual(json.loads((first/'decisions.json').read_text()),json.loads((second/'decisions.json').read_text()))
            self.assertEqual(pd.read_csv(first/'importance_repeats.csv').to_dict('records'),pd.read_csv(second/'importance_repeats.csv').to_dict('records'))
            # Applying a saved schema must not require production labels or refit.
            fresh=raw[raw.month.eq('2025-04')].drop(columns=['output_a','output_b']);fresh.to_csv(root/'fresh.csv',index=False)
            applied=apply_schema(first/'selected_schema.json',root/'fresh.csv',root/'applied')
            self.assertFalse(json.loads((applied/'manifest.json').read_text())['refitted'])
            fresh.drop(columns=['customer_income']).to_csv(root/'broken.csv',index=False)
            with self.assertRaises(ValueError):apply_schema(first/'selected_schema.json',root/'broken.csv',root/'broken_output')

if __name__=='__main__':unittest.main()
