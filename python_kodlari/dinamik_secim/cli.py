import argparse
import os
from pathlib import Path
import pandas as pd
from .schema import write_json,write_csv,profile

def default_rules():
    return {'algorithm':'catboost','seed':20261121,'threads':2,'holdout_months':3,
        'validation_months':3,'windows':3,'min_train_months':9,'permutation_repeats':8,
        'min_stable_windows':2,'min_delta_wape':0.00001,'min_baseline_improvement':0.01,
        'min_informative_targets':2,
        'quality':{'max_missing_fraction':.35,'max_dominance':.995,'max_categories':100,'correlation_threshold':.90},
        'catboost':{'iterations':300,'depth':4,'learning_rate':.05,'loss_function':'MAE','l2_leaf_reg':10.}}

def main():
    parser=argparse.ArgumentParser(description='Dictionary-driven common feature selection for branch clustering')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('discover',help='Write a schema profile and editable dictionary; no automatic semantic assignment')
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--branch',required=True);p.add_argument('--period',required=True);p.add_argument('--group',required=True)
    p.add_argument('--customer');p.add_argument('--grain',choices=['branch_month','customer_month','customer_branch_month'],default='branch_month')
    p=sub.add_parser('run');p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path)
    p.add_argument('--as-of');p.add_argument('--baseline',type=Path)
    p=sub.add_parser('check');p.add_argument('--run',type=Path,required=True)
    p=sub.add_parser('report');p.add_argument('--run',type=Path,required=True)
    p=sub.add_parser('apply');p.add_argument('--schema',type=Path,required=True);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--as-of')
    a=parser.parse_args()
    if a.command=='discover':
        if a.output.exists():raise FileExistsError(a.output)
        raw=pd.read_csv(a.input,dtype=str,encoding='utf-8-sig');keys={'branch':a.branch,'period':a.period,'group':a.group}
        if a.customer:keys['customer']=a.customer
        if not set(keys.values())<=set(raw):raise ValueError('Key column not found')
        if a.grain!='branch_month' and not a.customer:raise ValueError('--customer required')
        c={'version':1,'input':{'path':os.path.relpath(a.input.resolve(),a.output.resolve().parent),'grain':a.grain,'keys':keys},
           'fields':[],'selection':default_rules()}
        for name in raw:
            if name in keys.values():continue
            n=pd.to_numeric(raw[name],errors='coerce');dtype='numeric' if n.notna().sum()==raw[name].notna().sum() and raw[name].notna().any() else 'categorical'
            c['fields'].append({'name':name,'source':name,'label':name,'role':'review','dtype':dtype,
                'reason':'İş tanımı, çıktı/adayı rolü ve zamanında bilinirlik eşlenmeli'})
        write_json(a.output,c);write_csv(a.output.with_suffix('.profile.csv'),pd.DataFrame(profile(raw,c)))
        print(f'Dictionary template ready: {a.output}. Assign target/candidate roles before run.')
    elif a.command=='run':
        from .runner import run
        run(a.config,a.output,a.as_of,a.baseline)
    elif a.command=='apply':
        from .runner import apply_schema
        apply_schema(a.schema,a.input,a.output,a.as_of)
    elif a.command=='check':
        from .verify import verify
        verify(a.run)
    elif a.command=='report':
        from .report import report
        report(a.run)
