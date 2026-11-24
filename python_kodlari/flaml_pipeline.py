"""Time-bounded, validation-only AutoML challenger for the existing V4 panel.

Six searches; native exports; frozen validation champion may be a reference or
an existing model. Already inspected Oct-Dec data remains a reused benchmark.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import shutil
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from flaml import AutoML,tune
from sklearn.base import clone
from threadpoolctl import threadpool_limits

import model_v4_protokol as protocol
from flaml_tahmin import design,predict_bundle

ROOT=Path(__file__).resolve().parents[1]
LEARNERS=['lgbm','rf','catboost']
NAMES={'lgbm':'LightGBM','rf':'Random Forest','catboost':'CatBoost'}
MANUAL=['lightgbm__hareketli','random_forest__hareketli','catboost__hareketli']
REFERENCES=['son_ayi_koru','ay_sonu_koru','egitim_ortalamasi']
SEED=protocol.SEED

def encode(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,Path):return str(value)
    raise TypeError(type(value).__name__)
def save_json(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=encode))
def save_csv(path,rows):pd.DataFrame(rows).to_csv(path,index=False,encoding='utf-8-sig',float_format='%.17g')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def search_space():
    # Bounded CPU pilot. Same choices in every cell; never changed after TEST.
    return {
      'lgbm':{'n_estimators':{'domain':tune.lograndint(8,1025),'init_value':32,'low_cost_init_value':8},
              'num_leaves':{'domain':tune.lograndint(4,65),'init_value':15,'low_cost_init_value':4},
              'random_state':{'domain':SEED}},
      'rf':{'n_estimators':{'domain':tune.lograndint(16,513),'init_value':32,'low_cost_init_value':16},
            'max_leaves':{'domain':tune.lograndint(4,129),'init_value':8,'low_cost_init_value':4},
            'random_state':{'domain':SEED}},
      'catboost':{'n_estimators':{'domain':tune.lograndint(16,801),'init_value':64,'low_cost_init_value':16},
                  'early_stopping_rounds':{'domain':None},
                  'depth':{'domain':tune.choice([3,4,6]),'init_value':4,'low_cost_init_value':3},
                  'l2_leaf_reg':{'domain':tune.choice([3.,10.,25.]),'init_value':10.},
                  'allow_writing_files':{'domain':False},'random_seed':{'domain':SEED}}}

def level_mae(X_val,y_val,estimator,labels,X_train,y_train,weight_val=None,weight_train=None,
              config=None,groups_val=None,groups_train=None):
    anchors=[c for c in X_val.columns if c.endswith('_ay_sonu_bakiye_tl')]
    # Only this target's closing stock is in the 49-column design.
    assert len(anchors)==1
    anchor=X_val[anchors[0]].to_numpy(float)
    actual=np.asarray(y_val,float)+anchor
    pred=np.maximum(0,np.asarray(estimator.predict(X_val),float)+anchor)
    mae=float(np.abs(actual-pred).mean())
    return mae,{'validation_mae_tl':mae,'validation_wape':float(np.abs(actual-pred).sum()/actual.sum())}

def make_search(budget,log_file,max_iter=10000):
    return AutoML(time_budget=budget,task='regression',metric=level_mae,
      estimator_list=LEARNERS,eval_method='holdout',split_type='time',sample=False,
      skip_transform=True,retrain_full=False,ensemble=False,n_jobs=2,n_concurrent_trials=1,
      train_time_limit=min(20.,budget),seed=SEED,max_iter=max_iter,early_stop=False,
      log_type='all',log_file_name=str(log_file),verbose=0,mlflow_logging=False,
      custom_hp=search_space(),fit_kwargs_by_estimator={'catboost':{'use_best_model':False}})

def categories(train):
    return {'ekonomik_yapi_sentetik':sorted(train['ekonomik_yapi_sentetik'].unique().tolist())+['__BILINMIYOR__']}

def from_search(automl,frame,target,levels):
    native=automl.model.estimator
    learner=automl.best_estimator
    bundle={'format_version':1,'kind':'model','source':'flaml','target':target,
      'group':frame['buyuk_sehir_grubu'].iloc[0],'features':protocol.features(target,'hareketli'),
      'categories':levels,'encoding':'codes' if learner=='rf' else 'native',
      'anchor_feature':protocol.anchor_field(target,'hareketli'),'learner':learner,'model':copy.deepcopy(native)}
    x=design(frame,bundle['features'],levels)
    expected=np.maximum(0,frame[bundle['anchor_feature']].to_numpy(float)+automl.predict(x))
    # Proves native export faithfully reproduces FLAML preprocessing/prediction.
    np.testing.assert_allclose(predict_bundle(bundle,frame),expected,rtol=0,atol=1e-6)
    return bundle

def tree_count(model):
    if isinstance(model,CatBoostRegressor):return int(model.tree_count_)
    if hasattr(model,'booster_'):return int(model.booster_.num_trees())
    if hasattr(model,'estimators_'):return len(model.estimators_)
    raise TypeError(type(model))

def refit(bundle,frame,target):
    result=copy.deepcopy(bundle);model=clone(bundle['model']);n=tree_count(bundle['model'])
    if isinstance(model,CatBoostRegressor):
        params=model.get_params()
        for name in ['iterations','n_estimators','num_boost_round','num_trees']:params.pop(name,None)
        params.pop('early_stopping_rounds',None)
        model=CatBoostRegressor(**params,iterations=n)
    else:model.set_params(n_estimators=n,n_jobs=2)
    x=design(frame,bundle['features'],bundle['categories'],bundle['encoding'])
    y=frame['y_'+target+'_tl'].to_numpy(float)-frame[bundle['anchor_feature']].to_numpy(float)
    tick=time.perf_counter()
    if isinstance(model,CatBoostRegressor):model.fit(x,y,cat_features=list(bundle['categories']),verbose=False)
    else:model.fit(x,y)
    result['model']=model
    assert tree_count(model)==n
    return result,time.perf_counter()-tick

def reference_bundle(name,train,target,group):
    result={'format_version':1,'source':name,'group':group,'target':target}
    if name=='egitim_ortalamasi':return {**result,'kind':'constant','constant_tl':float(train['y_'+target+'_tl'].mean())}
    return {**result,'kind':'feature','reference_feature':target+('_bakiye_t' if name=='son_ayi_koru' else '_ay_sonu_bakiye_tl')}

def incumbent_bundle(source,name,target,group,artifact):
    alg=name.split('__')[0]
    meta_path=source/'modeller'/f'{alg}__{target}__{group}__hareketli__metadata.json'
    m=json.loads(meta_path.read_text());path=meta_path.parent/m[artifact+'_file']
    if alg=='lightgbm':model=lgb.Booster(model_file=str(path))
    elif alg=='catboost':model=CatBoostRegressor();model.load_model(str(path))
    else:model=joblib.load(path)
    return {'format_version':1,'kind':'model','source':name,'group':group,'target':target,
      'features':m['features'],'categories':m['categorical_levels'],'encoding':'native',
      'anchor_feature':m['anchor_feature'],'learner':alg,'model':model}

def export(path,bundle,frame):
    expected=predict_bundle(bundle,frame);joblib.dump(bundle,path,compress=3)
    np.testing.assert_allclose(predict_bundle(joblib.load(path),frame),expected,rtol=0,atol=1e-6)
    return expected

def prepare(source,out):
    frames={name:pd.read_csv(source/'veri'/f'{name}.csv',encoding='utf-8-sig')
            for name in ['train','validation','test','ocak2027_tahmin_girdisi']}
    hashes={name+'.csv':sha(source/'veri'/f'{name}.csv') for name in frames}
    incumbent_manifest=json.loads((source/'manifest.json').read_text())
    assert incumbent_manifest['source_version']==4
    assert hashes==incumbent_manifest['training_input_hashes'],'Stale incumbent models: input hashes differ.'
    assert incumbent_manifest['checks']['independent_saved_models_reload_and_inference']
    for a,b in [('train','validation'),('validation','test')]:
        assert frames[a]['hedef_donemi'].max()<frames[b]['hedef_donemi'].min()
    for key,frame in frames.items():
        assert not frame.duplicated(['sube_id','hedef_donemi']).any()
        assert ((pd.PeriodIndex(frame['ozellik_donemi'],freq='M')+1).astype(str)==frame['hedef_donemi']).all()
        for target in protocol.TARGETS:
            cols=protocol.features(target,'hareketli');assert len(cols)==49
            assert frame[cols].notna().all().all()
    for name in ['modeller','sonuclar','aramalar','kod','veri','hucreler']:(out/name).mkdir(parents=True,exist_ok=True)
    for name in frames:shutil.copy2(source/'veri'/f'{name}.csv',out/'veri'/f'{name}.csv')
    for name in ['ozellik_sozlugu.csv','zaman_ayrimi.csv']:
        if (source/'veri'/name).exists():shutil.copy2(source/'veri'/name,out/'veri'/name)
    return frames,hashes

def run_cell(source,out,frames,target,group,budget):
    cell=f'{target}__{group}';cols=protocol.features(target,'hareketli');anchor=protocol.anchor_field(target,'hareketli')
    train=frames['train'][frames['train']['buyuk_sehir_grubu'].eq(group)].copy()
    valid=frames['validation'][frames['validation']['buyuk_sehir_grubu'].eq(group)].copy()
    levels=categories(train);x=design(train,cols,levels);xv=design(valid,cols,levels)
    y=train['y_'+target+'_tl'].to_numpy(float)-train[anchor].to_numpy(float)
    yv=valid['y_'+target+'_tl'].to_numpy(float)-valid[anchor].to_numpy(float)
    print(f'SEARCH {cell}, budget={budget}s',flush=True)
    auto=make_search(budget,out/'aramalar'/f'{cell}.jsonl')
    tick=time.perf_counter();auto.fit(X_train=x,y_train=y,X_val=xv,y_val=yv)
    search_seconds=time.perf_counter()-tick
    selected=from_search(auto,valid,target,levels)
    candidates={};val_predictions={};selected_pred=predict_bundle(selected,valid)
    val_predictions['flaml_automl']=selected_pred
    candidates['flaml_automl']=protocol.metrics(valid['y_'+target+'_tl'],selected_pred)
    np.testing.assert_allclose(auto.best_loss,candidates['flaml_automl']['mae_tl'],rtol=0,atol=1e-5)
    for name in REFERENCES:
        pred=predict_bundle(reference_bundle(name,train,target,group),valid)
        val_predictions[name]=pred;candidates[name]=protocol.metrics(valid['y_'+target+'_tl'],pred)
    incumbent=pd.read_csv(source/'sonuclar/test_ve_validasyon_tahminleri.csv',encoding='utf-8-sig')
    for name in MANUAL:
        previous=incumbent[incumbent.gosterge.eq(target)&incumbent.sube_grubu.eq(group)&incumbent.model.eq(name)&incumbent.split.eq('VALIDATION')]
        joined=valid[['sube_id','hedef_donemi','y_'+target+'_tl']].merge(previous,on=['sube_id','hedef_donemi'],validate='one_to_one',how='left',sort=False)
        assert len(joined)==len(valid) and joined['tahmin_tl'].notna().all()
        np.testing.assert_allclose(joined['gercek_tl'],joined['y_'+target+'_tl'],rtol=0,atol=1e-6)
        val_predictions[name]=joined['tahmin_tl'].to_numpy();candidates[name]=protocol.metrics(valid['y_'+target+'_tl'],val_predictions[name])
    winner=min(candidates,key=lambda name:(candidates[name]['mae_tl'],0 if name in REFERENCES else 1,name))
    decision={'target':target,'group':group,'selected_candidate':winner,'selection_metric':'VALIDATION_MAE_TL',
      'validation_candidates':candidates,'best_learner':auto.best_estimator,'best_config':auto.best_config,
      'actual_trees':tree_count(selected['model']),'native_parameters':selected['model'].get_params(),
      'search_seconds':search_seconds,'search_budget_seconds':budget,
      'selection_frozen_before_test':True,'test_is_reused_historical_benchmark':True}
    # Record the decision before test predictions or TEST rows enter selection.
    save_json(out/'hucreler'/f'{cell}__selection.json',decision)
    val_path=out/'modeller'/f'{cell}__flaml_validation.joblib';export(val_path,selected,valid)
    block=pd.concat([train,valid],ignore_index=True)
    evaluation,eval_seconds=refit(selected,block,target)
    test=frames['test'][frames['test']['buyuk_sehir_grubu'].eq(group)].copy()
    eval_path=out/'modeller'/f'{cell}__flaml_evaluation.joblib'
    test_predictions={'flaml_automl':export(eval_path,evaluation,test)}
    all_observed=pd.concat([block,test],ignore_index=True)
    production,production_seconds=refit(selected,all_observed,target)
    future=frames['ocak2027_tahmin_girdisi'][frames['ocak2027_tahmin_girdisi']['buyuk_sehir_grubu'].eq(group)].copy()
    prod_path=out/'modeller'/f'{cell}__flaml_production.joblib';next_pred=export(prod_path,production,future)
    champion_eval=evaluation if winner=='flaml_automl' else reference_bundle(winner,train,target,group) if winner in REFERENCES else incumbent_bundle(source,winner,target,group,'evaluation')
    champion_prod=production if winner=='flaml_automl' else reference_bundle(winner,train,target,group) if winner in REFERENCES else incumbent_bundle(source,winner,target,group,'production')
    champ_eval_path=out/'modeller'/f'{cell}__champion_evaluation.joblib'
    champ_prod_path=out/'modeller'/f'{cell}__champion_production.joblib'
    test_predictions['secilen_sistem']=export(champ_eval_path,champion_eval,test)
    champion_next=export(champ_prod_path,champion_prod,future)
    for name in REFERENCES:test_predictions[name]=predict_bundle(reference_bundle(name,train,target,group),test)
    for name in MANUAL:
        prior=incumbent[incumbent.gosterge.eq(target)&incumbent.sube_grubu.eq(group)&incumbent.model.eq(name)&incumbent.split.eq('TEST')]
        joined=test[['sube_id','hedef_donemi','y_'+target+'_tl']].merge(prior,on=['sube_id','hedef_donemi'],validate='one_to_one',how='left',sort=False)
        assert len(joined)==len(test) and joined['tahmin_tl'].notna().all()
        np.testing.assert_allclose(joined.gercek_tl,joined['y_'+target+'_tl'],rtol=0,atol=1e-6)
        test_predictions[name]=joined['tahmin_tl'].to_numpy()
    val_predictions['secilen_sistem']=val_predictions[winner]
    context={'gosterge':target,'sube_grubu':group}
    metrics=[];rows=[];forecasts=[]
    for split,frame,preds in [('VALIDATION',valid,val_predictions),('TEST',test,test_predictions)]:
        for name,pred in preds.items():
            metrics.append({**context,'model':name,'split':split,**protocol.metrics(frame['y_'+target+'_tl'],pred)})
            for obs,yhat in zip(frame.itertuples(),pred):
                rows.append({**context,'model':name,'split':split,'sube_id':obs.sube_id,
                  'hedef_donemi':obs.hedef_donemi,'gercek_tl':getattr(obs,'y_'+target+'_tl'),'tahmin_tl':float(yhat)})
    for name,pred in [('flaml_automl',next_pred),('secilen_sistem',champion_next)]:
        for obs,yhat in zip(future.itertuples(),pred):
            forecasts.append({**context,'model':name,'sube_id':obs.sube_id,'hedef_donemi':obs.hedef_donemi,
                              'tahmin_aylik_ortalama_tl':float(yhat)})
    trials=[]
    for line in (out/'aramalar'/f'{cell}.jsonl').read_text().splitlines():
        record=json.loads(line)
        if 'learner' in record:trials.append(record)
    counts={learner:sum(r['learner']==learner for r in trials) for learner in LEARNERS}
    decision.update(validation_file=val_path.name,evaluation_file=eval_path.name,production_file=prod_path.name,
      champion_evaluation_file=champ_eval_path.name,champion_production_file=champ_prod_path.name,
      features=cols,categorical_levels=levels,anchor_feature=anchor,encoding=selected['encoding'],
      eval_refit_seconds=eval_seconds,production_refit_seconds=production_seconds,trial_counts=counts,
      trials=len(trials),test_results={name:protocol.metrics(test['y_'+target+'_tl'],pred) for name,pred in test_predictions.items()})
    save_json(out/'modeller'/f'{cell}__metadata.json',decision)
    result={'metadata':decision,'metrics':metrics,'predictions':rows,'forecasts':forecasts}
    save_json(out/'hucreler'/f'{cell}__complete.json',result)
    print(f'DONE {cell}: learner={auto.best_estimator}, trials={len(trials)}, champion={winner}, AutoML TEST WAPE={decision["test_results"]["flaml_automl"]["wape"]:.4%}',flush=True)
    return result

def run(source,out,budget,resume):
    started=time.perf_counter();frames,hashes=prepare(source,out)
    settings={'time_budget_per_search':budget,'n_jobs':2,'seed':SEED,'learners':LEARNERS,'features':49,
      'metric':'MAE after adding closing-stock anchor and clipping at zero','no_random_outer_split':True,
      'sample':False,'skip_transform':True,'retrain_full':False,'catboost_use_best_model':False,
      'note':'RF categorical codes follow FLAML built-in preprocessing; earlier manual RF uses one-hot encoding.',
      'test_policy':'Oct-Dec 2026 was already inspected. Reused benchmark, not an untouched final test.'}
    code_hashes={name:sha(Path(__file__).parent/name) for name in ['flaml_pipeline.py','flaml_tahmin.py','model_v4_protokol.py']}
    incumbent_hashes={name:sha(source/name) for name in ['manifest.json','sonuclar/test_ve_validasyon_tahminleri.csv']}
    signature=hashlib.sha256(json.dumps({'hashes':hashes,'settings':settings,'code':code_hashes,'incumbents':incumbent_hashes},sort_keys=True).encode()).hexdigest()
    config_path=out/'run_config.json'
    if resume and config_path.exists():assert json.loads(config_path.read_text())['signature']==signature
    save_json(config_path,{'signature':signature,'training_input_hashes':hashes,'settings':settings,'code_hashes':code_hashes,'incumbent_hashes':incumbent_hashes})
    cells=[]
    for group in protocol.GROUPS:
        for target in protocol.TARGETS:
            complete=out/'hucreler'/f'{target}__{group}__complete.json'
            if resume and complete.exists():cells.append(json.loads(complete.read_text()));continue
            cells.append(run_cell(source,out,frames,target,group,budget))
    pred=pd.DataFrame([r for cell in cells for r in cell['predictions']]);perf=[r for cell in cells for r in cell['metrics']]
    forecast=[r for cell in cells for r in cell['forecasts']];metadata=[cell['metadata'] for cell in cells]
    totals=[];monthly=[]
    for (target,name,split),p in pred.groupby(['gosterge','model','split'],sort=False):
        totals.append({'gosterge':target,'model':name,'split':split,**protocol.metrics(p.gercek_tl,p.tahmin_tl)})
    for group in ['ALL']+protocol.GROUPS:
        block=pred if group=='ALL' else pred[pred.sube_grubu.eq(group)]
        for (target,name,split,period),p in block.groupby(['gosterge','model','split','hedef_donemi'],sort=False):
            monthly.append({'gosterge':target,'model':name,'split':split,'hedef_donemi':period,'sube_grubu':group,**protocol.metrics(p.gercek_tl,p.tahmin_tl)})
    assert len(metadata)==6 and len(pred)==28800 and len(forecast)==1200
    for name,data in {'model_metrikleri':perf,'toplam_metrikler':totals,'aylik_metrikler':monthly,
                      'tahminler':pred,'ocak2027_tahminleri':forecast}.items():save_csv(out/'sonuclar'/f'{name}.csv',data)
    manifest={'source_version':4,'synthetic_financial_data':True,'training_input_hashes':hashes,
      'code_hashes':code_hashes,'incumbent_hashes':incumbent_hashes,
      'searches':6,'saved_native_bundles':30,'feature_count':49,'seed':SEED,
      'pipeline_seconds':time.perf_counter()-started,'settings':settings,
      'search_seconds_total':sum(m['search_seconds'] for m in metadata),'trial_count':sum(m['trials'] for m in metadata),
      'software':{name:importlib.metadata.version(name) for name in ['flaml','lightgbm','catboost','scikit-learn','numpy','pandas','xgboost','joblib']},
      'checks':{'input_hashes_match_v4_incumbents':True,'selection_uses_validation_only':True,
        'native_export_matches_flaml_validation_predictions':True,'model_reload_matches_predictions':True,
        'independent_verification_pending':True}}
    save_json(out/'manifest.json',manifest)
    save_json(out/'sonuclar/rapor_verisi.json',{'manifest':manifest,'metrics':perf,'total_metrics':totals,
      'monthly_metrics':monthly,'model_metadata':metadata})
    for name in ['flaml_pipeline.py','flaml_tahmin.py','flaml_dogrula.py','model_v4_protokol.py','pipeline.py']:
        p=Path(__file__).parent/name
        if p.exists():shutil.copy2(p,out/'kod'/name)
    print(json.dumps({'status':'complete','trials':manifest['trial_count'],'seconds':manifest['pipeline_seconds']},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=ROOT/'outputs/model_karsilastirma_v4')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/flaml_v4');p.add_argument('--budget',type=float,default=120.)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    logging.getLogger('flaml').setLevel(logging.ERROR)
    with threadpool_limits(limits=2):run(a.source.resolve(),a.output.resolve(),a.budget,a.resume)
