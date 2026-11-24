"""Independent dual LPs, projected peer combinations, phase-II objective and source mapping."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linprog


def verify(run):
    from .study import read,sha
    run=Path(run);m=read(run/'manifest.json');cfg=read(run/'config.json');tol=2e-7
    for name,digest in m['artifact_hashes'].items():
        path=(run/name).resolve();assert path.is_relative_to(run.resolve()) and sha(path)==digest,name
    panel=pd.read_csv(run/'dea_input.csv',dtype={'__branch':str,'__period':str,'__group':str},float_precision='round_trip')
    raw=pd.read_csv(run/'source_rows.csv',dtype={cfg['input']['keys']['branch']:str,cfg['input']['keys']['period']:str},float_precision='round_trip')
    keys=cfg['input']['keys']
    for name,values in cfg['input'].get('filters',{}).items():raw=raw[raw[name].isin(values)]
    assert len(raw)==len(panel) and not raw.duplicated([keys['branch'],keys['period']]).any()
    raw=raw.sort_values(keys['branch']);pp=panel.sort_values('__branch')
    assert raw[keys['branch']].tolist()==pp['__branch'].tolist() and raw[keys['group']].tolist()==pp['__group'].tolist()
    for field in cfg['inputs']+cfg['outputs']:
        expected=raw[field['source']].to_numpy(float)
        if field.get('transform')=='divide_by_calendar_days':expected=expected/pd.Period(m['as_of'],freq='M').days_in_month
        np.testing.assert_allclose(pp[field['name']],expected,rtol=1e-12,atol=1e-8)
    original=pd.DataFrame(read(run/'source_cluster_assignments.json')).sort_values('__branch')
    assert set(original['method'])=={m['primary_method']} and set(original['__period'])=={m['as_of']}
    np.testing.assert_array_equal(original[['__branch','__group','cluster_id']].to_numpy(),pp[['__branch','__group','cluster_id']].to_numpy())
    scores=read(run/'scores.json');certs=read(run/'certificates.json');pots=read(run/'potentials.json');clusters=read(run/'clusters.json');peers=read(run/'peers.json')
    assert len(scores)==len(panel) and len({s['branch'] for s in scores})==len(panel)
    if m.get('full_coverage_required'):
        assert cfg['insufficient_cluster_policy']=='error' and m['full_coverage_verified']
        assert all(s['efficiency'] is not None for s in scores) and m['summary']['withheld_branches']==0
        contract=read(run/'dea_coverage_contract.json')
        from kumeleme_secimi.coverage import assert_spec_covered
        assert_spec_covered(contract,cfg)
    inputs=[f['name'] for f in cfg['inputs']];outputs=[f['name'] for f in cfg['outputs']];a,b=len(inputs),len(outputs);required=max(a*b,3*(a+b))
    for cluster in clusters:
        block=panel[panel['cluster_id'].eq(cluster['cluster_id'])].sort_values('__branch');ids=block['__branch'].tolist();n=len(block)
        assert n==cluster['n'] and cluster['required_n']==required and cluster['size_rule_passed']==(n>=required)
        if n<required:
            ss=[s for s in scores if s['cluster_id']==cluster['cluster_id']]
            assert len(ss)==n and all(s['efficiency'] is None and s['status']=='insufficient_peer_count' for s in ss)
            assert not any(c['cluster_id']==cluster['cluster_id'] for c in certs+pots+peers);continue
        X=block[inputs].to_numpy(float);Y=block[outputs].to_numpy(float)
        xs=np.array([np.quantile(c[c>0],.5) for c in X.T]);ys=np.array([np.quantile(c[c>0],.5) for c in Y.T]);xn,yn=X/xs,Y/ys
        for i,branch in enumerate(ids):
            c=next(c for c in certs if c['branch']==branch);score=next(s for s in scores if s['branch']==branch)
            assert c['peer_branch_order']==ids and c['group']==str(block['__group'].iloc[0])
            phi=c['phi'];weights=np.array(c['lambda']);first=np.array(c['radial_lambda']);u=np.array(c['dual_u']);v=np.array(c['dual_v'])
            np.testing.assert_allclose(c['input_scales'],xs,rtol=1e-12);np.testing.assert_allclose(c['output_scales'],ys,rtol=1e-12)
            assert phi>=1-tol and (first>=-tol).all() and (xn.T@first<=xn[i]+tol).all() and (yn.T@first>=phi*yn[i]-tol).all()
            assert (u>=-tol).all() and (v>=-tol).all() and (yn@u<=xn@v+tol).all()
            np.testing.assert_allclose([u@yn[i],v@xn[i]],[1,phi],atol=tol,rtol=tol)
            # Separate multiplier formulation, with a free intercept for the VRS diagnostic.
            for returns in ['CRS','VRS']:
                vrs=returns=='VRS';obj=np.r_[np.zeros(b),xn[i],1.] if vrs else np.r_[np.zeros(b),xn[i]]
                A=np.c_[yn,-xn,-np.ones(n)] if vrs else np.c_[yn,-xn]
                eq=np.r_[yn[i],np.zeros(a+int(vrs))][None,:]
                bounds=[(0,None)]*(a+b)+([(None,None)] if vrs else [])
                res=linprog(obj,A_ub=A,b_ub=np.zeros(n),A_eq=eq,b_eq=[1],bounds=bounds,method='highs-ds')
                assert res.success,res.message
                expected_phi=phi if returns=='CRS' else c['vrs_phi']
                np.testing.assert_allclose(res.fun,expected_phi,rtol=tol,atol=tol)
            assert (weights>=-tol).all() and (xn.T@weights<=xn[i]+tol).all() and (yn.T@weights>=phi*yn[i]-tol).all()
            xp,yp=weights@X,weights@Y
            np.testing.assert_allclose(c['input_projection'],xp,rtol=tol,atol=tol)
            np.testing.assert_allclose(c['output_projection'],yp,rtol=tol,atol=1e-5)
            np.testing.assert_allclose(np.array(c['input_slack'])/xs,xn[i]-weights@xn,rtol=tol,atol=tol)
            np.testing.assert_allclose(np.array(c['output_slack'])/ys,weights@yn-phi*yn[i],rtol=tol,atol=tol)
            # Independent phase II with only lambda, without explicit slack variables.
            res=linprog(xn.sum(axis=1)-yn.sum(axis=1),A_ub=np.r_[xn.T,-yn.T],b_ub=np.r_[xn[i],-phi*yn[i]],bounds=(0,None),method='highs-ds')
            assert res.success
            independent_slack=xn[i].sum()-phi*yn[i].sum()-res.fun
            np.testing.assert_allclose(independent_slack,c['normalized_slack_sum'],atol=tol,rtol=tol)
            np.testing.assert_allclose([score['efficiency'],score['radial_growth'],score['vrs_efficiency'],score['scale_efficiency']],
                                       [1/phi,phi-1,1/c['vrs_phi'],c['vrs_phi']/phi],atol=tol,rtol=tol)
            assert 0<score['efficiency']<=score['vrs_efficiency']+tol<=1+2*tol
            radial=abs(phi-1)<=cfg['tolerance'];strong=radial and c['normalized_slack_sum']<=cfg['tolerance']
            assert score['status']==('strongly_efficient' if strong else 'radially_efficient_with_slack' if radial else 'inefficient')
            for j,name in enumerate(outputs):
                p=next(p for p in pots if p['branch']==branch and p['indicator']==name)
                np.testing.assert_allclose([p['actual'],p['radial_projection'],p['frontier_projection'],p['output_slack']],
                                           [Y[i,j],phi*Y[i,j],yp[j],yp[j]-phi*Y[i,j]],atol=1e-4,rtol=tol)
                np.testing.assert_allclose(p['improvement_amount'],max(0,yp[j]-Y[i,j]),atol=1e-4,rtol=tol)
                assert p['is_budget_target'] is False
                if p['actual']==0:assert p['improvement_ratio'] is None
                else:np.testing.assert_allclose(p['improvement_ratio'],p['improvement_amount']/p['actual'],atol=tol)
            listed=[r for r in peers if r['branch']==branch];expected=[(peer,float(w)) for peer,w in zip(ids,weights) if w>cfg['tolerance']]
            assert [(r['peer_branch'],r['lambda']) for r in listed]==expected
            assert all(r['peer_branch'] in ids for r in listed)
    assert len(pots)==len(certs)*b and m['summary']['calculated_branches']==len(certs)
    good=[s for s in scores if s['efficiency'] is not None]
    if good:np.testing.assert_allclose(m['summary']['mean_efficiency'],np.mean([s['efficiency'] for s in good]),atol=tol)
    assert m['dea_executed'] and not m['target_allocation_executed']
    return {'artifact_hashes':True,'source_fields_and_calendar_customer_average':True,'one_primary_cluster_per_branch':True,
            'cluster_size_rule_verified':True,'independent_CRS_dual_optimum_and_certificates':True,'independent_VRS_dual_optimum':True,
            'independent_phase_two_slack_optimum':True,'native_unit_projections_and_peer_isolation':True,'scores_not_forecasts_or_budget_targets':True,
            'complete_population_scored':not m.get('full_coverage_required') or all(s['efficiency'] is not None for s in scores)}
