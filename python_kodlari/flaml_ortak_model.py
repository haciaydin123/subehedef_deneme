"""Select one FLAML algorithm globally, then use it in every indicator/group.

Reuses logged searches. No per-cell algorithm/reference routing is permitted.
One algorithm family is selected; six target/group-specific fits are retained.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import pandas as pd
from flaml.automl.model import LGBMEstimator, RandomForestEstimator, CatBoostEstimator
from threadpoolctl import threadpool_limits

import flaml_pipeline as old
import model_v4_protokol as protocol
from flaml_tahmin import design, predict_bundle

ROOT=Path(__file__).resolve().parents[1]
CLASSES={'lgbm':LGBMEstimator,'rf':RandomForestEstimator,'catboost':CatBoostEstimator}

def scored_rows(frame,target,group,name,pred,split):
    return [{'gosterge':target,'sube_grubu':group,'model':name,'split':split,
       'sube_id':r.sube_id,'hedef_donemi':r.hedef_donemi,
       'gercek_tl':getattr(r,'y_'+target+'_tl'),'tahmin_tl':float(p)}
       for r,p in zip(frame.itertuples(),pred)]

def metrics_tables(pred):
    metrics=[];totals=[];monthly=[]
    for keys,p in pred.groupby(['gosterge','model','split','sube_grubu'],sort=False):
        metrics.append(dict(zip(['gosterge','model','split','sube_grubu'],keys))|protocol.metrics(p.gercek_tl,p.tahmin_tl))
    for keys,p in pred.groupby(['gosterge','model','split'],sort=False):
        totals.append(dict(zip(['gosterge','model','split'],keys))|protocol.metrics(p.gercek_tl,p.tahmin_tl))
    for group in ['ALL']+protocol.GROUPS:
        block=pred if group=='ALL' else pred[pred.sube_grubu.eq(group)]
        for keys,p in block.groupby(['gosterge','model','split','hedef_donemi'],sort=False):
            monthly.append(dict(zip(['gosterge','model','split','hedef_donemi'],keys))|{'sube_grubu':group}|protocol.metrics(p.gercek_tl,p.tahmin_tl))
    return metrics,totals,monthly

def run(source,search,out):
    started=time.perf_counter()
    if (out/'manifest.json').exists():raise FileExistsError(f'Output already exists: {out}')
    frames,hashes=old.prepare(source,out)
    legacy=json.loads((search/'manifest.json').read_text())
    assert legacy['training_input_hashes']==hashes
    assert legacy['checks']['independent_30_bundles_reloaded']
    shutil.copy2(search/'requirements.txt',out/'requirements.txt')
    validations=[];candidates=[];cached={};search_hashes={}
    # Read and fit only TRAIN; the decision is frozen before opening TEST labels.
    for group in protocol.GROUPS:
        for target in protocol.TARGETS:
            cell=f'{target}__{group}'
            train=frames['train'][frames['train'].buyuk_sehir_grubu.eq(group)].copy()
            valid=frames['validation'][frames['validation'].buyuk_sehir_grubu.eq(group)].copy()
            cols=protocol.features(target,'hareketli');anchor=protocol.anchor_field(target,'hareketli')
            levels=old.categories(train);x=design(train,cols,levels)
            log=search/'aramalar'/f'{cell}.jsonl';search_hashes[log.name]=old.sha(log)
            shutil.copy2(log,out/'aramalar'/log.name)
            trials=[r for r in map(json.loads,log.read_text().splitlines()) if 'learner' in r]
            for learner in old.LEARNERS:
                trial=min((r for r in trials if r['learner']==learner),key=lambda r:(r['validation_loss'],r['record_id']))
                wrapper=CLASSES[learner](task='regression',n_jobs=2,**trial['config'])
                kwargs={'use_best_model':False} if learner=='catboost' else {}
                wrapper.fit(x,train['y_'+target+'_tl'].to_numpy()-train[anchor].to_numpy(),**kwargs)
                bundle={'format_version':1,'kind':'model','source':'flaml_common','target':target,
                    'group':group,'features':cols,'categories':levels,'encoding':'codes' if learner=='rf' else 'native',
                    'anchor_feature':anchor,'learner':learner,'model':copy.deepcopy(wrapper.estimator)}
                pred=predict_bundle(bundle,valid)
                np.testing.assert_allclose(pred,np.maximum(0,valid[anchor].to_numpy()+wrapper.predict(design(valid,cols,levels))),atol=1e-6,rtol=0)
                file=f'{cell}__{learner}__validation.joblib';old.export(out/'modeller'/file,bundle,valid)
                validations+=scored_rows(valid,target,group,learner,pred,'VALIDATION')
                m={'target':target,'group':group,'learner':learner,'record_id':trial['record_id'],
                   'config':trial['config'],'logged_validation_mae_tl':trial['validation_loss'],
                   'validation_mae_tl':protocol.metrics(valid['y_'+target+'_tl'],pred)['mae_tl'],
                   'actual_trees':old.tree_count(bundle['model']),'validation_file':file}
                candidates.append(m);cached[(target,group,learner)]=(bundle,trials)
                print(f'VALIDATION {cell} {learner}',flush=True)
    val=pd.DataFrame(validations);scores=[]
    for learner in old.LEARNERS:
        by_target={}
        for target in protocol.TARGETS:
            p=val[val.gosterge.eq(target)&val.model.eq(learner)]
            by_target[target]=protocol.metrics(p.gercek_tl,p.tahmin_tl)['wape']
        scores.append({'learner':learner,'mean_indicator_wape':float(np.mean(list(by_target.values()))),
                       'indicator_wape':by_target})
    selected=min(scores,key=lambda r:(r['mean_indicator_wape'],r['learner']))['learner']
    decision={'selection_policy':'one_algorithm_all_indicators_and_groups','selected_learner':selected,
       'selection_metric':'unweighted mean of three bank-wide VALIDATION WAPEs',
       'algorithm_scores':scores,'candidates':candidates,'reference_eligible':False,
       'selection_uses_validation_only':True,'test_is_reused_historical_benchmark':True,
       'training_input_hashes':hashes,'search_log_hashes':search_hashes}
    old.save_csv(out/'sonuclar/algoritma_dogrulama_tahminleri.csv',val)
    old.save_json(out/'ortak_model_secimi.json',decision)
    print(f'GLOBAL WINNER: {selected}; frozen before TEST scoring',flush=True)
    # TEST and forecast use the one frozen family in all six cells.
    prior=pd.read_csv(source/'sonuclar/test_ve_validasyon_tahminleri.csv',encoding='utf-8-sig')
    rows=[];forecast=[];metadata=[]
    for group in protocol.GROUPS:
        for target in protocol.TARGETS:
            cell=f'{target}__{group}';bundle,trials=cached[(target,group,selected)]
            parts={k:f[f.buyuk_sehir_grubu.eq(group)].copy() for k,f in frames.items()}
            block=pd.concat([parts['train'],parts['validation']],ignore_index=True)
            evaluation,es=old.refit(bundle,block,target)
            ep=f'{cell}__common_evaluation.joblib';old.export(out/'modeller'/ep,evaluation,parts['test'])
            production,ps=old.refit(bundle,pd.concat([block,parts['test']],ignore_index=True),target)
            pp=f'{cell}__common_production.joblib';future=old.export(out/'modeller'/pp,production,parts['ocak2027_tahmin_girdisi'])
            for split,key,b in [('VALIDATION','validation',bundle),('TEST','test',evaluation)]:
                frame=parts[key];pred=predict_bundle(b,frame)
                for name in ['flaml_automl','secilen_sistem']:rows+=scored_rows(frame,target,group,name,pred,split)
                for name in old.REFERENCES:
                    h=predict_bundle(old.reference_bundle(name,parts['train'],target,group),frame)
                    rows+=scored_rows(frame,target,group,name,h,split)
                for name in old.MANUAL:
                    p=prior[prior.gosterge.eq(target)&prior.sube_grubu.eq(group)&prior.model.eq(name)&prior.split.eq(split)]
                    joined=frame[['sube_id','hedef_donemi','y_'+target+'_tl']].merge(p,on=['sube_id','hedef_donemi'],validate='one_to_one',how='left',sort=False)
                    assert joined.tahmin_tl.notna().all()
                    np.testing.assert_allclose(joined.gercek_tl,joined['y_'+target+'_tl'],atol=1e-6,rtol=0)
                    rows+=scored_rows(frame,target,group,name,joined.tahmin_tl.to_numpy(),split)
            for name in ['flaml_automl','secilen_sistem']:
                for obs,p in zip(parts['ocak2027_tahmin_girdisi'].itertuples(),future):
                    forecast.append({'gosterge':target,'sube_grubu':group,'model':name,'sube_id':obs.sube_id,
                                     'hedef_donemi':obs.hedef_donemi,'tahmin_aylik_ortalama_tl':float(p)})
            c=next(c for c in candidates if (c['target'],c['group'],c['learner'])==(target,group,selected))
            old_meta=json.loads((search/'modeller'/f'{cell}__metadata.json').read_text())
            m={**old_meta,'best_learner':selected,'selected_candidate':'flaml_automl','best_config':c['config'],
               'actual_trees':c['actual_trees'],'validation_file':c['validation_file'],'evaluation_file':ep,
               'production_file':pp,'champion_evaluation_file':ep,'champion_production_file':pp,
               'selection_metric':decision['selection_metric'],'common_algorithm':True,
               'selection_scope':'all indicators and groups','eval_refit_seconds':es,'production_refit_seconds':ps,
               'encoding':bundle['encoding'],'native_parameters':bundle['model'].get_params()}
            # Old per-cell decisions and TEST scores must not survive into current metadata.
            for k in ['validation_candidates','test_results']:m.pop(k,None)
            metadata.append(m);old.save_json(out/'modeller'/f'{cell}__metadata.json',m)
    pred=pd.DataFrame(rows);metrics,totals,monthly=metrics_tables(pred)
    for name,data in {'model_metrikleri':metrics,'toplam_metrikler':totals,'aylik_metrikler':monthly,
                      'tahminler':pred,'ocak2027_tahminleri':forecast}.items():old.save_csv(out/'sonuclar'/f'{name}.csv',data)
    code_names=['flaml_ortak_model.py','flaml_ortak_dogrula.py','flaml_ortak_raporla.py',
                'flaml_pipeline.py','flaml_tahmin.py','model_v4_protokol.py','pipeline.py']
    manifest={**legacy,'selection_policy':decision['selection_policy'],'selected_learner':selected,
       'algorithm_scores':scores,'new_searches':0,'reused_searches':6,'saved_native_bundles':30,
       'pipeline_seconds':time.perf_counter()-started,'search_log_hashes':search_hashes,
       'code_hashes':{n:old.sha(ROOT/'python_kodlari'/n) for n in code_names if (ROOT/'python_kodlari'/n).exists()},
       'checks':{'one_algorithm_all_indicators_and_groups':True,'selection_uses_validation_only':True,
                 'references_are_benchmarks_only':True}}
    old.save_json(out/'manifest.json',manifest)
    old.save_json(out/'sonuclar/rapor_verisi.json',{'manifest':manifest,'metrics':metrics,'total_metrics':totals,
         'monthly_metrics':monthly,'model_metadata':metadata,'global_selection':decision})
    for name in code_names:
        if (ROOT/'python_kodlari'/name).exists():shutil.copy2(ROOT/'python_kodlari'/name,out/'kod'/name)
    print(json.dumps({'selected_learner':selected,'algorithm_scores':scores,'seconds':manifest['pipeline_seconds']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=ROOT/'outputs/model_karsilastirma_v4')
    p.add_argument('--search',type=Path,default=ROOT/'outputs/flaml_v4')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/flaml_ortak_v4');a=p.parse_args()
    with threadpool_limits(limits=2):run(a.source.resolve(),a.search.resolve(),a.output.resolve())
