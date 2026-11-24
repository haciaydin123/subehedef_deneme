"""Independent case LP readback, paired cohorts, experiment aggregates and sampled stress dual LPs."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import spearmanr


def independent_radial(X,Y):
    X=np.asarray(X,float);Y=np.asarray(Y,float);X=X/np.median(X,axis=0)
    ys=np.array([np.median(c[c>0]) for c in Y.T]);Y=Y/ys
    values=[];a,b=X.shape[1],Y.shape[1]
    for x,y in zip(X,Y):
        # Multiplier/dual formulation, separate from the envelopment stress implementation.
        result=linprog(np.r_[np.zeros(b),x],A_ub=np.c_[Y,-X],b_ub=np.zeros(len(X)),
                       A_eq=np.r_[y,np.zeros(a)][None,:],b_eq=[1.],bounds=(0,None),method='highs-ds')
        assert result.success,result.message;values.append(1/result.fun)
    return np.array(values)


def verify(directory):
    from verimlilik.study import read,sha
    from verimlilik.verify import verify as verify_case
    from .audit import deterministic_noise
    root=Path(directory);m=read(root/'manifest.json');policy=read(root/'policy.json');base=read(root/'base_config.json');latest=m['as_of']
    for name,digest in m['artifact_hashes'].items():
        p=(root/name).resolve();assert p.is_relative_to(root.resolve()) and sha(p)==digest,name
    assert sha(root/'input/source_panel.csv')==m['source_data_sha256']
    assert sha(root/'input/main_dea_manifest.json')==m['source_dea_manifest_sha256']
    index=read(root/'case_index.json');assert len(index)==2*len(m['periods'])
    reports={}
    for case in index:
        path=root/case['path'];assert sha(path/'manifest.json')==case['manifest_sha256']
        checks=verify_case(path);assert all(checks.values())
        reports[(case['scenario'],case['period'])]=read(path/'report.json')
    original=read(root/'input/main_dea_scores.json');rerun=reports[('customer',latest)]['scores']
    assert len(original)==len(rerun)
    for a,b in zip(sorted(original,key=lambda r:r['branch']),sorted(rerun,key=lambda r:r['branch'])):
        assert a['branch']==b['branch'] and a['cluster_id']==b['cluster_id']
        np.testing.assert_allclose(a['efficiency'],b['efficiency'],atol=1e-8,rtol=0)
    coverage=read(root/'coverage.json');comparisons=read(root/'comparisons.json');paired=read(root/'paired_scores.json')
    for row in coverage:
        report=reports[(row['scenario'],row['period'])];summary=report['summary']
        assert row['calculated_branches']==sum(r['efficiency'] is not None for r in report['scores'])
        assert row['withheld_branches']==summary['branches']-row['calculated_branches']
        np.testing.assert_allclose(row['coverage'],row['calculated_branches']/summary['branches'],atol=1e-12)
        assert row['withheld_cluster_ids']==[r['cluster_id'] for r in report['clusters'] if r['n']<r['required_n']]
    for row in comparisons:
        p=row['period'];a={s['branch']:s for s in reports[('customer',p)]['scores'] if s['efficiency'] is not None};b={s['branch']:s for s in reports[('customer_staff',p)]['scores'] if s['efficiency'] is not None}
        common=sorted(set(a)&set(b));assert row['common_scored']==len(common)
        before=[a[x]['efficiency'] for x in common];after=[b[x]['efficiency'] for x in common];delta=np.array(after)-before
        np.testing.assert_allclose([row['base_mean_same_cohort'],row['augmented_mean_same_cohort'],row['mean_mechanical_score_increase']],
                                   [np.mean(before),np.mean(after),np.mean(delta)],atol=1e-12,rtol=0)
        assert row['material_change_count']==sum(delta>policy['material_score_change'])
        assert set(r['branch'] for r in paired if r['period']==p)==set(common)
    raw=pd.read_csv(root/'input/source_panel.csv',dtype={v:str for v in base['input']['keys'].values()})
    for k,values in base['input'].get('filters',{}).items():raw=raw[raw[k].isin(values)]
    quality=read(root/'quality.json')
    assert quality['keys']['rows']==len(raw) and quality['keys']['duplicate_branch_months']==int(raw.duplicated(list(base['input']['keys'].values())[:2]).sum())
    for row in quality['fields']:
        x=pd.to_numeric(raw[row['source']],errors='coerce').to_numpy(float)
        assert row['missing_or_non_numeric']==int(np.isnan(x).sum()) and row['zero']==int((x==0).sum()) and row['negative']==int((x<0).sum())
    for row in read(root/'staff_profile.json'):
        field=next(f for f in policy['additional_inputs'] if f['name']==row['field']);counts=raw.groupby(base['input']['keys']['branch'])[field['source']].nunique()
        assert row['constant_across_all_months']==int((counts==1).sum())
    local=read(root/'local_stress.json');removal=read(root/'peer_removal.json');summaries=read(root/'stress_summary.json')
    assert len(local)==m['local_stress_records'] and len(removal)==m['peer_removal_records']
    for row in local+removal:np.testing.assert_allclose(row['stressed']-row['baseline'],row['score_change'],atol=1e-12)
    sampled_local=0;sampled_removal=0
    for summary in summaries:
        s,cluster=summary['scenario'],summary['cluster_id'];report=reports[(s,latest)];cfg=report['config'];folder=root/next(r['path'] for r in index if r['scenario']==s and r['period']==latest)
        frame=pd.read_csv(folder/'dea_input.csv',dtype={'__branch':str},float_precision='round_trip');frame=frame[frame.cluster_id.eq(cluster)].sort_values('__branch').reset_index(drop=True)
        # Add unused candidate columns solely to reproduce common named random draws.
        src=raw[raw[base['input']['keys']['period']].eq(latest)].set_index(base['input']['keys']['branch'])
        for field in policy['additional_inputs']:
            if field['name'] not in frame:frame[field['name']]=frame['__branch'].map(src[field['source']])
        inputs=[f['name'] for f in cfg['inputs']];outputs=[f['name'] for f in cfg['outputs']];all_names=sorted(set(inputs+outputs)|{f['name'] for f in policy['additional_inputs']});n=len(frame)
        lr=[r for r in local if r['scenario']==s and r['cluster_id']==cluster];rr=[r for r in removal if r['scenario']==s and r['cluster_id']==cluster]
        assert len(lr)==n*policy['perturbation_repeats']
        for name,rows in [('local',lr),('peer_removal',rr)]:
            changes=np.abs([r['score_change'] for r in rows]);expected=summary[name]
            assert expected['n']==len(rows)
            if len(rows):np.testing.assert_allclose([expected['mean_abs_change'],expected['p95_abs_change'],expected['max_abs_change']],[np.mean(changes),np.quantile(changes,.95),np.max(changes)],atol=1e-12)
        noise=deterministic_noise(policy['seed'],cluster,0,n,len(all_names),policy['relative_perturbation']);changed=frame[all_names].to_numpy(float)*(1+noise)
        expected=independent_radial(changed[:,[all_names.index(f) for f in inputs]],changed[:,[all_names.index(f) for f in outputs]])
        records={r['branch']:r['stressed'] for r in lr if r['repeat']==0};np.testing.assert_allclose(expected,[records[x] for x in frame['__branch']],atol=2e-7,rtol=2e-7);sampled_local+=n
        if rr:
            assert len(rr)==n*(n-1) and all(r['removed_peer']!=r['branch'] for r in rr)
            worst=max(rr,key=lambda r:r['score_change'])['removed_peer'];block=frame[frame['__branch'].ne(worst)]
            expected=independent_radial(block[inputs].to_numpy(float),block[outputs].to_numpy(float))
            records={r['branch']:r['stressed'] for r in rr if r['removed_peer']==worst};np.testing.assert_allclose(expected,[records[x] for x in block['__branch']],atol=2e-7,rtol=2e-7);sampled_removal+=len(block)
    # Correlations and temporal summaries are recalculated from saved same-month data/records.
    for row in read(root/'correlations.json'):
        folder=root/next(r['path'] for r in index if r['scenario']=='customer_staff' and r['period']==row['period'])
        frame=pd.read_csv(folder/'dea_input.csv');frame=frame if row['cluster_id']=='ALL' else frame[frame.cluster_id.eq(row['cluster_id'])]
        x,y=frame[row['base_input']],frame[row['additional_input']]
        if x.nunique()>1 and y.nunique()>1:
            np.testing.assert_allclose([row['pearson'],row['spearman']],[np.corrcoef(x,y)[0,1],spearmanr(x,y).statistic],atol=1e-10)
        else:assert row['pearson'] is None and row['spearman'] is None
    temporal=read(root/'temporal.json')
    for row in read(root/'temporal_summary.json'):
        records=[r for r in temporal if r['scenario']==row['scenario'] and (row['cohort']=='all_common' or r['cluster_stayer']) and (row['cohort']!='stable_profile_stayers' or r['stable_measured_profile'])]
        assert row['n']==len(records)
        if records:np.testing.assert_allclose(row['mean_abs_change'],np.mean([abs(r['score_change']) for r in records]),atol=1e-12)
    recommendation=read(root/'recommendation.json');base_cov=next(r for r in coverage if r['scenario']=='customer' and r['period']==latest);aug_cov=next(r for r in coverage if r['scenario']=='customer_staff' and r['period']==latest)
    assert recommendation['recommended_scenario']==('customer' if base_cov['coverage']==1 else None)
    assert recommendation['augmented_global_eligible']==(aug_cov['coverage']>=policy['minimum_global_coverage'])
    assert not recommendation['main_model_changed'] and not recommendation['higher_efficiency_is_selection_objective']
    assert m['dea_latest_pointer_unchanged']
    return {'artifact_hashes':True,'all_scenario_cases_independently_verified':True,'baseline_scores_reproduced':True,'identical_cohort_comparisons':True,
            'quality_and_coverage_readback':True,'all_stress_aggregates_recomputed':True,'stratified_local_dual_checks':sampled_local>0,
            'worst_peer_removal_dual_checks':sampled_removal>0,'correlations_and_temporal_summary_readback':True,'main_model_not_replaced':True}
