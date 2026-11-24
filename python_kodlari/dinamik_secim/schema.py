"""Explicit field semantics, schema discovery and reproducible IO."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROLES={'candidate','target','context','excluded','review'}
AGGS={'sum','mean','weighted_mean','mode','first_exact','category_shares','count_distinct'}

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def write_csv(path,frame):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(path,index=False,encoding='utf-8-sig',float_format='%.17g')

def period(value):
    value=str(value)
    if not re.fullmatch(r'\d{4}-\d{2}',value):raise ValueError(f'Period must be YYYY-MM: {value}')
    return str(pd.Period(value,freq='M'))

def load_config(path):
    path=Path(path).resolve();c=json.loads(path.read_text(encoding='utf-8'))
    if c.get('version')!=1:raise ValueError('Unsupported dictionary version')
    i=c['input'];keys=i['keys']
    if i['grain'] not in {'branch_month','customer_month','customer_branch_month'}:raise ValueError('Unknown grain')
    if not {'branch','period','group'}<=keys.keys():raise ValueError('branch, period and group mappings required')
    if i['grain']!='branch_month' and 'customer' not in keys:raise ValueError('Customer key required')
    if len(set(keys.values()))!=len(keys):raise ValueError('Key mappings must be distinct')
    fields=c['fields'];names=[f['name'] for f in fields]
    if len(names)!=len(set(names)) or any(n.startswith('__') for n in names):raise ValueError('Field names must be unique, without __ prefix')
    if not any(f['role']=='target' for f in fields):raise ValueError('Map at least one target before running')
    target_sources={f['source'] for f in fields if f['role']=='target'}
    for f in fields:
        if f['role'] not in ROLES:raise ValueError(f'Unknown role: {f}')
        if f.get('dtype') not in {'numeric','categorical'}:raise ValueError(f'Declare dtype: {f["name"]}')
        if f['source'] in keys.values() and f['role'] in {'candidate','target'}:raise ValueError('Keys cannot be candidate/target sources')
        if f['role']=='target' and f['dtype']!='numeric':raise ValueError('Targets must be numeric')
        if f['role']=='candidate':
            if f.get('known_by_period_end') is not True:raise ValueError(f'Availability must be declared: {f["name"]}')
            if f.get('production_proxy',False) or f['source'] in target_sources:raise ValueError(f'Production source cannot be a candidate: {f["name"]}')
            lag=f.get('availability_lag_months',0)
            if not isinstance(lag,int) or lag<0:raise ValueError('Availability lag must be a nonnegative integer')
        if i['grain']!='branch_month' and f['role'] in {'candidate','target'}:
            if f.get('aggregation') not in AGGS:raise ValueError(f'Aggregation required: {f["name"]}')
            if f['aggregation']=='weighted_mean' and not f.get('weight'):raise ValueError('weighted_mean needs weight column')
            if f['aggregation']=='category_shares' and not f.get('levels'):raise ValueError('category_shares needs fixed levels')
            if f['aggregation']=='category_shares':
                levels=list(map(str,f['levels']))
                if len(levels)!=len(set(levels)) or set(levels)&{'__OTHER__','__MISSING__'}:raise ValueError('Invalid/reserved category levels')
    rules=c['selection']
    if rules.get('algorithm')!='catboost':raise ValueError('One common algorithm is required: catboost')
    if not 0<rules['quality']['max_missing_fraction']<1:raise ValueError('Invalid missingness threshold')
    if not 0<rules['quality']['max_dominance']<=1:raise ValueError('Invalid dominance threshold')
    if not 0<rules['quality']['correlation_threshold']<=1:raise ValueError('Invalid correlation threshold')
    for k in ['validation_months','windows','min_train_months','permutation_repeats','min_stable_windows']:
        if not isinstance(rules[k],int) or rules[k]<1:raise ValueError(f'Positive integer required: {k}')
    if rules['min_stable_windows']>rules['windows']:raise ValueError('min_stable_windows exceeds windows')
    if not isinstance(rules['holdout_months'],int) or rules['holdout_months']<0:raise ValueError('Invalid holdout_months')
    c['_config_path']=str(path)
    c['_source_path']=str((path.parent/i['path']).resolve())
    return c

def read_input(c):
    # Read strings first: identifiers retain leading zeroes; numbers are validated explicitly.
    frame=pd.read_csv(c['_source_path'],encoding=c['input'].get('encoding','utf-8-sig'),
                      sep=c['input'].get('delimiter',','),dtype=str)
    frame=frame.replace(r'^\s*$',np.nan,regex=True)
    for name,values in c['input'].get('filters',{}).items():
        if name not in frame:raise ValueError(f'Filter source missing: {name}')
        frame=frame[frame[name].isin(list(map(str,values)))]
    if frame.empty:raise ValueError('Input is empty after filters')
    return frame

def profile(frame,c):
    mapping={}
    for f in c['fields']:mapping.setdefault(f['source'],[]).append(f)
    keys=set(c['input']['keys'].values());rows=[]
    for col in frame.columns:
        fs=mapping.get(col,[]);s=frame[col];v=pd.to_numeric(s,errors='coerce')
        rows.append({'source':col,'roles':','.join(f['role'] for f in fs) if fs else 'key' if col in keys else 'review',
          'fields':','.join(f['name'] for f in fs),'status':'mapped' if fs or col in keys else 'unmapped',
          'missing_fraction':float(s.isna().mean()),'unique':int(s.nunique()),
          'suggested_dtype':'numeric' if v.notna().sum()==s.notna().sum() and s.notna().any() else 'categorical'})
    for f in c['fields']:
        if f['source'] not in frame:rows.append({'source':f['source'],'roles':f['role'],'fields':f['name'],
             'status':'missing','missing_fraction':1.,'unique':0,'suggested_dtype':f['dtype']})
    return rows

def numeric(series,name):
    value=pd.to_numeric(series,errors='coerce')
    if (series.notna()&value.isna()).any() or np.isinf(value.to_numpy(float)).any():
        raise ValueError(f'Invalid numeric values: {name}')
    return value.astype(float)
