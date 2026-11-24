"""Declared grain/aggregation and exact calendar joins; no future event joins."""
import numpy as np
import pandas as pd
from .schema import numeric,period

def aggregate(block,f):
    s=block[f['source']];agg=f['aggregation']
    if agg=='category_shares':
        known=s.fillna('__MISSING__').astype(str)
        categories=list(map(str,f['levels']))+['__OTHER__','__MISSING__']
        known=known.where(known.isin(categories),'__OTHER__')
        return {f['name']+'__'+str(k):float(known.eq(str(k)).mean()) for k in categories}
    if agg=='count_distinct':return {f['name']:float(s.nunique())}
    if agg=='first_exact':
        if s.dropna().nunique()>1:raise ValueError(f'Branch attribute is not constant: {f["name"]}')
        value=s.dropna().iloc[0] if s.notna().any() else np.nan
    elif agg=='mode':
        mode=s.dropna().mode();value=mode.iloc[0] if len(mode) else np.nan
    else:
        s=numeric(s,f['source'])
        if agg=='sum':value=s.sum(min_count=1)
        elif agg=='mean':value=s.mean()
        elif agg=='weighted_mean':
            w=numeric(block[f['weight']],f['weight'])
            if w.isna().any() or w.lt(0).any():raise ValueError('Weights must be finite and nonnegative')
            keep=s.notna()&w.gt(0);den=w[keep].sum()
            value=float((s[keep]*w[keep]).sum()/den) if den else np.nan
        else:raise ValueError(agg)
    return {f['name']:value}

def prepare(raw,c,as_of=None):
    keys=c['input']['keys'];data=raw.copy()
    for source in keys.values():
        if source not in data or data[source].isna().any():raise ValueError(f'Missing key: {source}')
    data['__period']=data[keys['period']].map(period)
    as_of=period(as_of) if as_of else data['__period'].max()
    data=data[data['__period'].le(as_of)].copy()
    if data.empty:raise ValueError('No rows on/before as_of')
    data['__branch']=data[keys['branch']].astype(str);data['__group']=data[keys['group']].astype(str)
    grain=c['input']['grain'];unique=[keys['period'],keys['branch']]
    if grain=='customer_month':unique=[keys['period'],keys['customer']]
    if grain=='customer_branch_month':unique+=[keys['customer']]
    if data.duplicated(unique).any():raise ValueError(f'Duplicate rows at declared grain {grain}: {unique}')
    if data.groupby(['__branch','__period'])['__group'].nunique().gt(1).any():raise ValueError('Multiple groups for one branch-month')
    fields=[];missing=[]
    for f in c['fields']:
        if f['role'] not in {'candidate','target'}:continue
        if f['source'] not in data:
            if f.get('required',f['role']=='target'):raise ValueError(f'Required source missing: {f["source"]}')
            missing.append(f['name']);continue
        fields.append(f)
    identities=['__branch','__period','__group']
    expanded=[]
    if grain=='branch_month':
        panel=data[identities].copy()
        for f in fields:
            panel[f['name']]=numeric(data[f['source']],f['source']) if f['dtype']=='numeric' else data[f['source']]
            expanded.append(f.copy())
    else:
        records=[]
        for _,block in data.groupby(['__branch','__period'],sort=True):
            row={key:block[key].iloc[0] for key in identities}
            for f in fields:row.update(aggregate(block,f))
            records.append(row)
        panel=pd.DataFrame(records)
        for f in fields:
            if f['aggregation']=='category_shares':
                for level in list(f['levels'])+['__OTHER__','__MISSING__']:
                    expanded.append({**f,'name':f['name']+'__'+str(level),'dtype':'numeric'})
            else:expanded.append(f.copy())
    if panel.columns.duplicated().any() or len({f['name'] for f in expanded})!=len(expanded):raise ValueError('Expanded field names collide')
    panel=panel.sort_values(['__period','__branch']).reset_index(drop=True)
    for f in expanded:
        if f['dtype']=='numeric':panel[f['name']]=numeric(panel[f['name']],f['name'])
        if f['role']=='target' and (panel[f['name']].isna().any() or panel[f['name']].lt(0).any()):raise ValueError(f'Invalid target: {f["name"]}')
        lag=f.get('availability_lag_months',0) if f['role']=='candidate' else 0
        if lag:
            lookup=panel.set_index(['__branch','__period'])[f['name']]
            lookup_keys=pd.MultiIndex.from_arrays([panel['__branch'],(pd.PeriodIndex(panel['__period'],freq='M')-lag).astype(str)])
            panel[f['name']]=lookup.reindex(lookup_keys).to_numpy()
    target_names=[f['name'] for f in expanded if f['role']=='target']
    panel['__label_period']=(pd.PeriodIndex(panel['__period'],freq='M')+1).astype(str)
    lookup=panel.set_index(['__branch','__period'])
    future_keys=pd.MultiIndex.from_arrays([panel['__branch'],panel['__label_period']])
    for name in target_names:panel['__y_'+name]=lookup[name].reindex(future_keys).to_numpy()
    return panel,expanded,missing,as_of
