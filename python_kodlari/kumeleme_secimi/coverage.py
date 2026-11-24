"""DEA-aware coverage contract, shared by selection, inference and DEA execution."""
from collections import Counter


def required_units(inputs,outputs):
    if not isinstance(inputs,int) or not isinstance(outputs,int) or inputs<1 or outputs<1:
        raise ValueError('Positive input/output dimensions required')
    return max(inputs*outputs,3*(inputs+outputs))


def field_definition(field):
    return {k:field.get(k,'identity') for k in ['name','source','transform']}


def build_contract(base,audit,configured_minimum):
    if not isinstance(configured_minimum,int) or configured_minimum<2:raise ValueError('Invalid configured cluster minimum')
    if audit.get('minimum_global_coverage')!=1.0:raise ValueError('DEA input audit must require 100% branch coverage')
    inputs=base['inputs'];outputs=base['outputs'];extra=audit['additional_inputs']
    if not inputs or not outputs:raise ValueError('Empty DEA input/output specification')
    if len({f['name'] for f in inputs+extra+outputs})!=len(inputs+extra+outputs):raise ValueError('DEA field names overlap')
    scenarios=[]
    for name,fields in [('customer',inputs),('customer_staff',inputs+extra)]:
        scenarios.append({'scenario':name,'inputs':[field_definition(f) for f in fields],'input_count':len(fields),
                          'output_count':len(outputs),'required_n':required_units(len(fields),len(outputs))})
    required=max(configured_minimum,*(s['required_n'] for s in scenarios))
    return {'version':1,'require_full_coverage':True,'configured_min_cluster_size':configured_minimum,'required_min_cluster_size':required,
            'outputs':[field_definition(f) for f in outputs],'scenarios':scenarios,'on_incomplete_coverage':'error',
            'size_rule_is_pilot_heuristic':True,'coverage_is_not_statistical_adequacy':True}


def assert_spec_covered(contract,config):
    if not contract or not contract.get('require_full_coverage'):raise ValueError('Full-coverage clustering contract missing; reselect the main clustering model')
    out=[field_definition(f) for f in config['outputs']];inputs=[field_definition(f) for f in config['inputs']]
    if out!=contract['outputs'] or not any(inputs==s['inputs'] for s in contract['scenarios']):
        raise ValueError('DEA fields changed; rerun clustering selection with the updated DEA input scenarios')
    if config.get('insufficient_cluster_policy')!='error':raise ValueError('Partial DEA scoring is disabled; insufficient_cluster_policy must be error')


def assignment_coverage(assignments,expected,minimum,k_by_group=None):
    """Require exactly one assignment for every source branch and no small/empty learned clusters."""
    identity=lambda r:(str(r['__period']),str(r['__branch']),str(r['__group']))
    expected_keys=[identity(r) for r in expected];actual_keys=[identity(r) for r in assignments]
    if not expected_keys or len({(p,b) for p,b,g in expected_keys})!=len(expected_keys):raise ValueError('Empty or duplicate source branch-month population')
    if len({(p,b) for p,b,g in actual_keys})!=len(actual_keys) or set(actual_keys)!=set(expected_keys):raise ValueError('Full branch coverage failed: missing, duplicate or extra assignments')
    periods=sorted({p for p,b,g in expected_keys});result=[]
    for period in periods:
        rows=[r for r in assignments if str(r['__period'])==period]
        counts=Counter((str(r['__group']),int(r['cluster'])) for r in rows)
        if k_by_group:
            for group in {str(r['__group']) for r in rows}:
                if group not in k_by_group:raise ValueError('Unknown cluster geography')
                for cluster in range(k_by_group[group]):counts.setdefault((group,cluster),0)
        bad={f'{g}__C{c+1}':n for (g,c),n in counts.items() if n<minimum}
        if bad:raise ValueError(f'Full DEA coverage failed for {period}; need at least {minimum} branches per cluster: {bad}. Reselect or rebuild clustering; no branches were scored partially.')
        result.append({'period':period,'source_branches':len(rows),'assigned_branches':len(rows),'clusters':len(counts),
                       'min_cluster_size':min(counts.values()),'required_min_cluster_size':minimum,'coverage':1.0,
                       'cluster_sizes':{f'{g}__C{c+1}':n for (g,c),n in sorted(counts.items())}})
    return result
