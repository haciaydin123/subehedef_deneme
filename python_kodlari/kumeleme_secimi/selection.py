"""Pure selection logic: evaluation-month records never enter the decision."""
import math
from statistics import mean


def validate_config(config, reference, available):
    val=config['validation_periods']; evaluation=config['evaluation_periods']
    if not val or len(set(val))!=len(val) or len(set(evaluation))!=len(evaluation):
        raise ValueError('Periods must be nonempty validation and unique lists')
    if set(val)&set(evaluation) or any(p<=reference for p in val):
        raise ValueError('Reference < validation < evaluation; periods must not overlap')
    if evaluation and min(evaluation)<=max(val):raise ValueError('Evaluation must follow validation')
    if not set(val+evaluation)<=set(available):raise ValueError('Period absent from completed study')
    if set(config['weights'])!={'silhouette','refit_ari','subsample_ari','seed_ari'}:
        raise ValueError('Unknown score components')
    if any(not math.isfinite(w) or w<0 for w in config['weights'].values()) or not math.isclose(sum(config['weights'].values()),1):
        raise ValueError('Score weights must be nonnegative and sum to one')
    if config['minimum_gain']<0:raise ValueError('Negative switch margin')
    t=config['thresholds']
    if not isinstance(t['min_cluster_size'],int) or t['min_cluster_size']<2:raise ValueError('Invalid minimum size')
    quality_keys=['silhouette','refit_ari','subsample_ari','seed_ari','worst_group_refit_ari','worst_group_silhouette']
    if not all(k in t for k in quality_keys):raise ValueError('Missing global or geographic quality threshold')
    if any(not math.isfinite(t[k]) or not -1<=t[k]<=1 for k in quality_keys):
        raise ValueError('Invalid quality threshold')


def monthly_metrics(monthly, method, periods, counts):
    """Equal month weights, then fixed reference branch-count stratum weights."""
    parts={}; errors=[]
    for group in counts:
        part={}
        for mode,key,metric in [('frozen','silhouette','common_gower_silhouette'),('frozen','frozen_ari','ari'),('monthly_refit','refit_ari','ari')]:
            values=[]
            for period in periods:
                rows=[r for r in monthly if r['method']==method and r['group']==group and r['period']==period and r['mode']==mode]
                if len(rows)!=1 or rows[0].get(metric) is None or not math.isfinite(rows[0][metric]):
                    errors.append(f'{group}/{period}/{key}: missing or duplicate metric');continue
                values.append(rows[0][metric])
            part[key]=mean(values) if len(values)==len(periods) and values else None
        frozen=[r for r in monthly if r['method']==method and r['group']==group and r['period'] in periods and r['mode']=='frozen']
        part['min_cluster_size']=min((r['min_cluster_size'] for r in frozen),default=None)
        part['complete_cluster_count']=bool(frozen) and all(r['distinct_clusters']==len(r['cluster_sizes']) and sum(r['cluster_sizes'])==counts[group] for r in frozen)
        parts[group]=part
    def weighted(key):
        return sum(counts[g]*parts[g][key] for g in counts)/sum(counts.values()) if all(parts[g][key] is not None for g in counts) else None
    return {**{k:weighted(k) for k in ['silhouette','refit_ari','frozen_ari']},'by_group':parts,'errors':errors}


def score_candidate(case, selected, monthly, sensitivity, method, config, repeats):
    counts=case['group_counts']; rows=[r for r in selected if r['method']==method]
    result={k:case[k] for k in ['case_id','feature_ratio','feature_count','bank_clusters','k_by_group']}
    result.update(candidate_id=case['case_id']+'::'+method,method=method,applicable=bool(rows),qualified=False,score=None,gates=[],metrics={},group_quality=[])
    if not rows:
        result['reasons']=['Karma sayısal/kategorik girdi yok; yöntem bu senaryoda uygulanamaz.'];return result
    metrics=monthly_metrics(monthly,method,config['validation_periods'],counts)
    errors=list(metrics['errors'])
    for kind in ['subsample','seed']:
        for group in counts:
            values=[r for r in sensitivity if r['method']==method and r['group']==group and r['case']==kind]
            repeat_ids={r['repeat'] for r in values}
            if len(values)!=repeats[kind] or repeat_ids!=set(range(repeats[kind])) or any(r.get('ari') is None or not math.isfinite(r['ari']) for r in values):
                errors.append(f'{group}/{kind}: incomplete repeats');metrics['by_group'][group][kind+'_ari']=None
            else:metrics['by_group'][group][kind+'_ari']=mean(r['ari'] for r in values)
        metrics[kind+'_ari']=sum(counts[g]*metrics['by_group'][g][kind+'_ari'] for g in counts)/sum(counts.values()) if all(metrics['by_group'][g][kind+'_ari'] is not None for g in counts) else None
    t=config['thresholds']; gates=result['gates']
    def gate(name,passed,actual=None,threshold=None):gates.append({'name':name,'passed':bool(passed),'actual':actual,'threshold':threshold})
    complete=len(rows)==len(counts) and {r['group'] for r in rows}==set(counts)
    gate('complete_reference_groups',complete)
    gate('complete_validation_and_repeats',not errors)
    gate('reference_converged',complete and all(r['converged'] for r in rows))
    reference_size=complete and all(r['distinct_clusters']==case['k_by_group'][r['group']]==len(r['cluster_sizes']) and sum(r['cluster_sizes'])==counts[r['group']] and r['min_cluster_size']>=t['min_cluster_size'] for r in rows)
    gate('reference_size',reference_size,min((r['min_cluster_size'] for r in rows),default=None),t['min_cluster_size'])
    val_size=all(p['min_cluster_size'] is not None and p['min_cluster_size']>=t['min_cluster_size'] and p['complete_cluster_count'] for p in metrics['by_group'].values())
    gate('validation_size',val_size,min((p['min_cluster_size'] for p in metrics['by_group'].values() if p['min_cluster_size'] is not None),default=None),t['min_cluster_size'])
    for key in config['weights']:gate(key,metrics[key] is not None and metrics[key]>=t[key],metrics[key],t[key])
    # Every geography must clear the same declared floor; branch-count weights cannot hide a weak group.
    result['group_quality']=[{'group':g,'silhouette':metrics['by_group'][g]['silhouette'],'threshold':t['worst_group_silhouette'],
                             'passed':metrics['by_group'][g]['silhouette'] is not None and metrics['by_group'][g]['silhouette']>=t['worst_group_silhouette']}
                            for g in counts]
    group_scores=[p['silhouette'] for p in metrics['by_group'].values()]
    weakest=min(group_scores) if all(v is not None for v in group_scores) else None
    metrics['worst_group_silhouette']=weakest
    gate('worst_group_silhouette',weakest is not None and weakest>=t['worst_group_silhouette'],weakest,t['worst_group_silhouette'])
    worst=min((p['refit_ari'] for p in metrics['by_group'].values() if p['refit_ari'] is not None),default=None)
    gate('worst_group_refit_ari',worst is not None and worst>=t['worst_group_refit_ari'],worst,t['worst_group_refit_ari'])
    if all(metrics[k] is not None for k in config['weights']):
        result['score']=sum(w*max(0,min(1,metrics[k])) for k,w in config['weights'].items())
    result.update(metrics=metrics,qualified=all(g['passed'] for g in gates),reasons=[g['name'] for g in gates if not g['passed']]+errors)
    return result


def choose(candidates,config):
    eligible=sorted((c for c in candidates if c['qualified'] and c['score'] is not None),key=lambda c:(-c['score'],c['bank_clusters'],c['feature_count'],c['candidate_id']))
    if not eligible:return {'status':'no_qualified_model','selected':None,'reason':'Hiçbir aday bütün pilot kalite ve boyut koşullarını sağlamadı.'}
    best=eligible[0];current=next((c for c in eligible if c['candidate_id']==config.get('incumbent')),None)
    winner=best;reason='highest_qualified_score'
    if current and best['score']-current['score']<config['minimum_gain']:
        winner=current;reason='qualified_incumbent_within_switch_margin'
    return {'status':'selected','selected':winner,'best_candidate_id':best['candidate_id'],'reason':reason,
            'qualified_count':len(eligible),'incumbent':config.get('incumbent'),'minimum_gain':config['minimum_gain']}
