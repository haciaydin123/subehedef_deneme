"""Readback checks independent weighted arithmetic, gates, decision and native inference."""
import json
from pathlib import Path
from statistics import mean
import joblib
import numpy as np
import pandas as pd


def verify(run):
    from .runner import read,check_hashes,infer
    run=Path(run);m=read(run/'manifest.json');check_hashes(run,m)
    cfg=read(run/'config.json');plan=read(run/'grid_plan.json');evidence=read(run/'source_evidence.json')
    contract=read(run/'dea_coverage_contract.json')
    definitions=contract['source_definitions'];expected_min=max(contract['configured_min_cluster_size'],
        *(max(s['input_count']*s['output_count'],3*(s['input_count']+s['output_count'])) for s in contract['scenarios']))
    assert contract['require_full_coverage'] and expected_min==cfg['thresholds']['min_cluster_size']==contract['required_min_cluster_size']
    assert [s['input_count'] for s in contract['scenarios']]==[len(definitions['base']['inputs']),len(definitions['base']['inputs'])+len(definitions['input_audit']['additional_inputs'])]
    candidates=read(run/'candidate_scores.json');decision=read(run/'decision.json')
    assert not set(cfg['validation_periods'])&set(cfg['evaluation_periods'])
    assert len(candidates)==len(plan['cases'])*len(plan['settings']['methods'])
    assert len({c['candidate_id'] for c in candidates})==len(candidates)
    for candidate in candidates:
        case=next(c for c in plan['cases'] if c['case_id']==candidate['case_id']);counts=case['group_counts'];e=evidence[case['case_id']];method=candidate['method']
        reference=[r for r in e['selected'] if r['method']==method]
        assert candidate['applicable']==bool(reference)
        if not reference:
            assert not candidate['qualified'] and candidate['score'] is None;continue
        components={};by_group={}
        for g in counts:
            values={}
            for key,mode,field in [('silhouette','frozen','common_gower_silhouette'),('refit_ari','monthly_refit','ari')]:
                rows=[r for r in e['monthly'] if r['method']==method and r['group']==g and r['period'] in cfg['validation_periods'] and r['mode']==mode]
                assert {r['period'] for r in rows}==set(cfg['validation_periods']) and len(rows)==len(cfg['validation_periods'])
                values[key]=sum(r[field] for r in rows)/len(rows)
            for key,kind in [('subsample_ari','subsample'),('seed_ari','seed')]:
                rows=[r for r in e['sensitivity'] if r['method']==method and r['group']==g and r['case']==kind]
                assert len(rows)==e['repeat_counts'][kind]
                values[key]=sum(r['ari'] for r in rows)/len(rows)
            by_group[g]=values
        for key in cfg['weights']:
            components[key]=sum(counts[g]*by_group[g][key] for g in counts)/sum(counts.values())
            np.testing.assert_allclose(candidate['metrics'][key],components[key],rtol=0,atol=1e-12)
        expected=sum(cfg['weights'][k]*np.clip(v,0,1) for k,v in components.items())
        np.testing.assert_allclose(candidate['score'],expected,rtol=0,atol=1e-12)
        t=cfg['thresholds'];passed={}
        passed['complete_reference_groups']=len(reference)==len(counts) and {r['group'] for r in reference}==set(counts)
        passed['complete_validation_and_repeats']=True
        passed['reference_converged']=all(r['converged'] for r in reference)
        passed['reference_size']=all(r['min_cluster_size']>=t['min_cluster_size'] and r['distinct_clusters']==case['k_by_group'][r['group']] and sum(r['cluster_sizes'])==counts[r['group']] for r in reference)
        vals=[r for r in e['monthly'] if r['method']==method and r['period'] in cfg['validation_periods'] and r['mode']=='frozen']
        passed['validation_size']=all(r['min_cluster_size']>=t['min_cluster_size'] and r['distinct_clusters']==case['k_by_group'][r['group']] and sum(r['cluster_sizes'])==counts[r['group']] for r in vals)
        passed.update({k:v>=t[k] for k,v in components.items()})
        weakest=min(v['silhouette'] for v in by_group.values())
        np.testing.assert_allclose(candidate['metrics']['worst_group_silhouette'],weakest,atol=1e-12,rtol=0)
        expected_quality=[{'group':g,'silhouette':by_group[g]['silhouette'],'threshold':t['worst_group_silhouette'],'passed':by_group[g]['silhouette']>=t['worst_group_silhouette']} for g in counts]
        assert [r['group'] for r in candidate['group_quality']]==[r['group'] for r in expected_quality]
        for got,want in zip(candidate['group_quality'],expected_quality):
            assert got['passed']==want['passed'] and got['threshold']==want['threshold']
            np.testing.assert_allclose(got['silhouette'],want['silhouette'],atol=1e-12,rtol=0)
        passed['worst_group_silhouette']=weakest>=t['worst_group_silhouette']
        passed['worst_group_refit_ari']=min(v['refit_ari'] for v in by_group.values())>=t['worst_group_refit_ari']
        assert {g['name']:g['passed'] for g in candidate['gates']}==passed
        assert candidate['qualified']==all(passed.values())
    ranked=sorted((c for c in candidates if c['qualified']),key=lambda c:(-c['score'],c['bank_clusters'],c['feature_count'],c['candidate_id']))
    expected=ranked[0] if ranked else None
    incumbent=next((c for c in ranked if c['candidate_id']==cfg.get('incumbent')),None)
    if incumbent and expected['score']-incumbent['score']<cfg['minimum_gain']:expected=incumbent
    assert decision['selected']==expected
    winner=decision['selected'];index=read(run/'selected_model_index.json');native_count=0
    if winner:
        assert set(x['method'] for x in index)=={winner['method']}
        expected_rows=pd.DataFrame(read(run/'primary_assignments.json'))
        panel=pd.read_csv(run/'prepared_panel.csv',dtype={'__branch':str,'__period':str,'__group':str})
        expected_coverage=[]
        for period in sorted(panel['__period'].unique()):
            source_rows=panel[panel['__period'].eq(period)];assigned=expected_rows[expected_rows['__period'].eq(period)]
            assert not assigned.duplicated(['__branch','__period']).any()
            assert set(map(tuple,source_rows[['__branch','__group']].to_numpy()))==set(map(tuple,assigned[['__branch','__group']].to_numpy()))
            counts=assigned.groupby(['__group','cluster']).size()
            assert int(counts.min())>=expected_min and len(counts)==winner['bank_clusters']
            expected_coverage.append({'period':period,'source_branches':len(source_rows),'assigned_branches':len(assigned),'clusters':len(counts),
                'min_cluster_size':int(counts.min()),'required_min_cluster_size':expected_min,'coverage':1.0,
                'cluster_sizes':{f'{group}__C{cluster+1}':int(n) for (group,cluster),n in counts.sort_index().items()}})
        assert read(run/'full_coverage.json')==expected_coverage
        assert m['full_coverage_required'] and m['full_coverage_verified']
        for item in index:
            bundle=joblib.load(run/item['path']);assert bundle['method']==winner['method'] and bundle['fit_period']==m['reference_period']
            frame=panel[panel['__group'].eq(item['group'])].sort_values(['__period','__branch'])
            geometry=bundle['geometry'];num=[];cat=[]
            for name in geometry['numeric']:
                bounds=geometry['ranges'][name];span=bounds['high']-bounds['low'];s=frame[name].to_numpy(float)
                num.append(np.clip((s-bounds['low'])/span,0,1) if span else np.zeros(len(s)))
            num=np.column_stack(num) if num else np.empty((len(frame),0))
            for name in geometry['categorical']:
                vals=frame[name].astype(str).to_numpy();cat.append(np.where(np.isin(vals,geometry['levels'][name]),vals,'__UNKNOWN__'))
            cat=np.column_stack(cat) if cat else np.empty((len(frame),0),dtype=object)
            if winner['method']=='gower_kmedoids':
                distance=sum((np.abs(num[:,j,None]-bundle['medoid_numeric'][None,:,j]) for j in range(num.shape[1])),np.zeros((len(frame),bundle['k'])))
                for j in range(cat.shape[1]):distance+=(cat[:,j,None]!=bundle['medoid_categorical'][None,:,j])
                labels=distance.argmin(axis=1)
            elif winner['method']=='urf_kmedoids':
                parts=[num]+[np.column_stack([(cat[:,j]==level).astype(float) for level in geometry['levels'][name]]) for j,name in enumerate(geometry['categorical'])]
                leaves=bundle['forest'].apply(np.column_stack(parts));labels=(1-(leaves[:,None,:]==bundle['medoid_leaves'][None,:,:]).mean(axis=2)).argmin(axis=1)
            else:
                labels=bundle['prototype_model'].predict(np.column_stack([num.astype(object),cat]),categorical=list(range(num.shape[1],num.shape[1]+cat.shape[1])))
            recorded=expected_rows[expected_rows['__group'].eq(item['group'])].sort_values(['__period','__branch'])
            np.testing.assert_array_equal(frame[['__branch','__period']].to_numpy(),recorded[['__branch','__period']].to_numpy())
            np.testing.assert_array_equal(labels,recorded['cluster'].to_numpy());native_count+=1
        # Public apply uses exactly one method and reproduces all recorded reference/monitoring assignments.
        for period in expected_rows['__period'].unique():
            actual=pd.DataFrame(infer(run,run/'prepared_panel.csv',period)).sort_values('__branch')
            recorded=expected_rows[expected_rows['__period'].eq(period)].sort_values('__branch')
            np.testing.assert_array_equal(actual['cluster'].to_numpy(),recorded['cluster'].to_numpy())
            assert len(actual)==actual['__branch'].nunique() and set(actual['method'])=={winner['method']}
        latest=pd.read_csv(run/'primary_latest.csv');assert len(latest)==m['latest_branch_count'] and set(latest['__period'])=={m['as_of']}
        evaluation=read(run/'evaluation.json')
        assert evaluation['periods']==cfg['evaluation_periods'] and not evaluation['changes_decision']
        if cfg['evaluation_periods']:
            case=next(c for c in plan['cases'] if c['case_id']==winner['case_id']);counts=case['group_counts']
            for key,mode,field in [('silhouette','frozen','common_gower_silhouette'),('refit_ari','monthly_refit','ari'),('frozen_ari','frozen','ari')]:
                grouped={g:mean(r[field] for r in evidence[case['case_id']]['monthly'] if r['method']==winner['method'] and r['group']==g and r['mode']==mode and r['period'] in cfg['evaluation_periods']) for g in counts}
                expected_metric=sum(counts[g]*grouped[g] for g in counts)/sum(counts.values())
                np.testing.assert_allclose(evaluation['metrics'][key],expected_metric,atol=1e-12,rtol=0)
    else:
        assert not index and not m['full_coverage_verified'] and read(run/'full_coverage.json')==[]
        assert read(run/'primary_assignments.json')==[] and pd.read_csv(run/'primary_latest.csv').empty
        assert not list((run/'models').glob('*.joblib')) and not ranked
        try:infer(run,run/'does_not_exist.csv')
        except ValueError as exc:assert str(exc)=='No qualified main model; inference is disabled'
        else:raise AssertionError('Inference must stop without a qualified model')
    assert read(run/'geographic_quality.json')==[{'candidate_id':c['candidate_id'],'case_id':c['case_id'],'method':c['method'],**g} for c in candidates for g in c['group_quality']]
    # Deliberately corrupt evaluation values: this must not influence any candidate or winner.
    from copy import deepcopy
    from .selection import score_candidate,choose
    changed=deepcopy(evidence)
    for e in changed.values():
        for r in e['monthly']:
            if r['period'] in cfg['evaluation_periods']:r.update(ari=-999,common_gower_silhouette=999,min_cluster_size=0)
    recomputed=[score_candidate(case,changed[case['case_id']]['selected'],changed[case['case_id']]['monthly'],changed[case['case_id']]['sensitivity'],method,cfg,changed[case['case_id']]['repeat_counts']) for case in plan['cases'] for method in plan['settings']['methods']]
    assert recomputed==candidates and choose(recomputed,cfg)['selected']==winner
    return {'artifact_hashes':True,'independent_weighted_metrics_and_gates':True,'independent_global_winner':True,'independent_evaluation_readback':True,'evaluation_mutation_cannot_change_decision':True,
            'native_selected_models_reloaded':native_count==len(index),'single_method_apply_matches_all_recorded_months':True,
            'dea_dimensions_set_cluster_minimum':True,'all_source_branches_covered_without_small_clusters':True,
            'independent_geographic_quality_gates':True,'no_unqualified_model_can_be_applied':True,
            'dea_and_allocation_not_executed':not m['dea_executed'] and not m['target_allocation_executed']}
