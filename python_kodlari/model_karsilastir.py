"""Same temporal splits/features for LightGBM, Random Forest and CatBoost.

Run from project root:
  .work/ml_env/bin/python python_kodlari/model_karsilastir.py
The original LightGBM experiment is preserved. All financial data is synthetic.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

import pipeline as pilot

ROOT=Path(__file__).resolve().parents[1]
ALGORITHMS=['lightgbm','random_forest','catboost']
ALGORITHM_NAMES={'lightgbm':'LightGBM','random_forest':'Random Forest','catboost':'CatBoost'}
CAT='ekonomik_yapi_sentetik'
CONFIGS={
 'lightgbm':pilot.CONFIGS,
 'random_forest':[
   {'config_id':'A','n_estimators':200,'max_depth':6,'min_samples_leaf':10,'max_features':.9},
   {'config_id':'B','n_estimators':300,'max_depth':10,'min_samples_leaf':5,'max_features':1.},
   {'config_id':'C','n_estimators':300,'max_depth':None,'min_samples_leaf':3,'max_features':.8}],
 'catboost':[
   {'config_id':'A','depth':3,'l2_leaf_reg':10.},
   {'config_id':'B','depth':4,'l2_leaf_reg':10.},
   {'config_id':'C','depth':6,'l2_leaf_reg':25.}],
}

def fit(algorithm,train,valid,target,view,config,levels,n=None):
    cols=pilot.features(target,view)
    x=pilot.design(train,cols,levels)
    y=train['y_'+target+'_tl'].to_numpy(float)-pilot.anchor(train,target,view)
    params={k:v for k,v in config.items() if k!='config_id'}
    if algorithm=='lightgbm':
        model=pilot.make_model(config,n or 800)
    elif algorithm=='random_forest':
        # Vocabulary is fitted on TRAIN only; unknown future categories are ignored.
        prep=ColumnTransformer([('numeric','passthrough',[c for c in cols if c!=CAT]),
              ('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False),[CAT])])
        model=Pipeline([('encode',prep),('forest',RandomForestRegressor(
              random_state=pilot.SEED,n_jobs=2,criterion='squared_error',**params))])
    else:
        model=CatBoostRegressor(iterations=n or 800,learning_rate=.04,loss_function='RMSE',
          eval_metric='MAE',random_seed=pilot.SEED,thread_count=2,task_type='CPU',
          boosting_type='Plain',cat_features=[CAT],allow_writing_files=False,verbose=False,**params)
    start=time.perf_counter()
    if algorithm=='lightgbm' and valid is not None:
        xv=pilot.design(valid,cols,levels)
        yv=valid['y_'+target+'_tl'].to_numpy(float)-pilot.anchor(valid,target,view)
        model.fit(x,y,eval_X=xv,eval_y=yv,eval_metric='l1',
                  callbacks=[lgb.early_stopping(50,first_metric_only=True,verbose=False)])
    elif algorithm=='catboost' and valid is not None:
        xv=pilot.design(valid,cols,levels)
        yv=valid['y_'+target+'_tl'].to_numpy(float)-pilot.anchor(valid,target,view)
        model.fit(x,y,eval_set=(xv,yv),early_stopping_rounds=50,use_best_model=True)
    else:
        model.fit(x,y)
    return model,time.perf_counter()-start

def raw_predict(model,algorithm,x):
    if algorithm=='catboost':return model.predict(x,thread_count=2)
    if algorithm=='lightgbm':return model.predict(x,num_threads=2)
    return model.predict(x)

def predict(model,algorithm,frame,target,view,levels):
    return np.maximum(0,pilot.anchor(frame,target,view)+raw_predict(
        model,algorithm,pilot.design(frame,pilot.features(target,view),levels)))

class PredictionAdapter:
    def __init__(self,model,algorithm):self.model,self.algorithm=model,algorithm
    def predict(self,x):return raw_predict(self.model,self.algorithm,x)

def native_importance(model,algorithm,cols):
    if algorithm=='lightgbm':
        values=model.booster_.feature_importance(importance_type='gain')
        method='gain'
    elif algorithm=='catboost':
        values=model.get_feature_importance(type='PredictionValuesChange',thread_count=2)
        method='PredictionValuesChange'
    else:
        encoded=model.named_steps['forest'].feature_importances_
        numeric=[c for c in cols if c!=CAT]
        mapped=dict(zip(numeric,encoded[:len(numeric)]))
        mapped[CAT]=float(encoded[len(numeric):].sum())
        values=np.array([mapped[c] for c in cols]);method='impurity_decrease'
    values=np.asarray(values,float)
    return 100*values/max(float(values.sum()),1e-30),method

def save_and_verify(model,algorithm,path,frame,target,view,levels,expected):
    if algorithm=='lightgbm':
        model.booster_.save_model(str(path))
        restored=lgb.Booster(model_file=str(path))
    elif algorithm=='catboost':
        model.save_model(str(path));restored=CatBoostRegressor();restored.load_model(str(path))
    else:
        joblib.dump(model,path,compress=3);restored=joblib.load(path)
    actual=predict(restored,algorithm,frame,target,view,levels)
    np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-6)

def main(source,out):
    pilot.lgb=lgb
    start=time.perf_counter()
    for folder in ['sonuclar','modeller','kod','veri']: (out/folder).mkdir(parents=True,exist_ok=True)
    for p in (source/'veri').glob('*.csv'):
        if (out/'veri'/p.name).resolve()!=p.resolve():shutil.copy2(p,out/'veri'/p.name)
    hashes={}
    frames={}
    for split in ['train','validation','test','ocak2027_tahmin_girdisi']:
        path=source/'veri'/f'{split}.csv'
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        frames[split]=pd.read_csv(path,encoding='utf-8-sig')
    data=pd.concat([frames[x] for x in ['train','validation','test']],ignore_index=True)
    future=frames['ocak2027_tahmin_girdisi']
    assert [len(frames[x]) for x in ['train','validation','test']]==[3400,600,600]
    assert data[['sube_id','hedef_donemi']].duplicated().sum()==0
    assert data.groupby('split').hedef_donemi.agg(['min','max']).to_dict()=={
      'min':{'TEST':'2026-10','TRAIN':'2025-02','VALIDATION':'2026-07'},
      'max':{'TEST':'2026-12','TRAIN':'2026-06','VALIDATION':'2026-09'}}
    assert len(future)==200 and future['hedef_donemi'].eq('2027-01').all()
    views=getattr(pilot,'VIEWS',['yapisal','gecikmeli'])
    is_v4=hasattr(pilot,'VIEWS')
    baselines=['son_ayi_koru','egitim_ortalamasi']+(['ay_sonu_koru'] if is_v4 else [])
    # Recompute all reference forecasts from THIS dataset. No V3 forecasts are reused for V4.
    predictions=[];scores=[];search=[];importance=[];group_importance=[];metadata=[];forecasts=[]
    for group in pilot.GROUPS:
        block=data[data['buyuk_sehir_grubu'].eq(group)]
        train=block[block['split'].eq('TRAIN')]
        for target in pilot.TARGETS:
            label='y_'+target+'_tl'
            for split in ['VALIDATION','TEST']:
                frame=block[block['split'].eq(split)]
                candidates={'son_ayi_koru':frame[target+'_bakiye_t'].to_numpy(float),
                  'egitim_ortalamasi':np.repeat(train[label].mean(),len(frame))}
                if is_v4:candidates['ay_sonu_koru']=frame[target+'_ay_sonu_bakiye_tl'].to_numpy(float)
                for baseline,pred in candidates.items():
                    ctx={'sube_grubu':group,'gosterge':target,'algoritma':'referans','gorunum':'referans','model':baseline}
                    scores.append({**ctx,'split':split,**pilot.metrics(frame[label],pred),'egitim_saniye':0.,
                                   'agac_sayisi':0,'ayar_arama_saniye':0.,'tahmin_1000_satir_ms':0.})
                    for row,yhat in zip(frame.itertuples(),pred):
                        predictions.append({**ctx,'split':split,'sube_id':row.sube_id,'hedef_donemi':row.hedef_donemi,
                                            'gercek_tl':getattr(row,label),'tahmin_tl':float(yhat)})
    for group in pilot.GROUPS:
        block=data[data['buyuk_sehir_grubu'].eq(group)]
        train=block[block['split'].eq('TRAIN')]
        valid=block[block['split'].eq('VALIDATION')]
        test=block[block['split'].eq('TEST')]
        train_valid=block[block['split'].isin(['TRAIN','VALIDATION'])]
        levels=sorted(train[CAT].unique().tolist())
        for target in pilot.TARGETS:
            label='y_'+target+'_tl'
            for view in views:
                cols=pilot.features(target,view)
                for algorithm in ALGORITHMS:
                    model_id=f'{algorithm}__{target}__{group}__{view}'
                    ctx={'sube_grubu':group,'gosterge':target,'algoritma':algorithm,
                         'gorunum':view,'model':algorithm+'__'+view}
                    candidates=[];tuning_start=time.perf_counter()
                    for config in CONFIGS[algorithm]:
                        model,seconds=fit(algorithm,train,valid,target,view,config,levels)
                        val_pred=predict(model,algorithm,valid,target,view,levels)
                        score=pilot.metrics(valid[label],val_pred)
                        n=int(model.best_iteration_) if algorithm=='lightgbm' else int(model.tree_count_) if algorithm=='catboost' else config['n_estimators']
                        search.append({**ctx,**config,'best_iteration':n,'valid_mae_tl':score['mae_tl'],
                                       'valid_wape':score['wape'],'fit_seconds':seconds})
                        candidates.append((score['mae_tl'],config,model,seconds,n,val_pred))
                    tuning_seconds=time.perf_counter()-tuning_start
                    _,config,selected,seconds,n,val_pred=min(candidates,key=lambda x:x[0])
                    scores.append({**ctx,'split':'VALIDATION',**pilot.metrics(valid[label],val_pred),
                                   'egitim_saniye':seconds,'agac_sayisi':n,'ayar_arama_saniye':tuning_seconds})
                    # All parameter/iteration choices precede TEST access.
                    evaluation,eval_seconds=fit(algorithm,train_valid,None,target,view,config,levels,n=n)
                    test_pred=predict(evaluation,algorithm,test,target,view,levels)
                    # Warm once, then median of five complete frame predictions (incl. transform).
                    latencies=[]
                    for _ in range(5):
                        tick=time.perf_counter();predict(evaluation,algorithm,test,target,view,levels)
                        latencies.append(time.perf_counter()-tick)
                    speed=1000*1000*float(np.median(latencies))/len(test)
                    scores.append({**ctx,'split':'TEST',**pilot.metrics(test[label],test_pred),
                                   'egitim_saniye':eval_seconds,'agac_sayisi':n,
                                   'ayar_arama_saniye':tuning_seconds,'tahmin_1000_satir_ms':speed})
                    ext={'lightgbm':'txt','catboost':'cbm','random_forest':'joblib'}[algorithm]
                    eval_path=out/'modeller'/f'{model_id}__evaluation.{ext}'
                    save_and_verify(evaluation,algorithm,eval_path,test,target,view,levels,test_pred)
                    for split,frame,pred in [('VALIDATION',valid,val_pred),('TEST',test,test_pred)]:
                        for row,yhat in zip(frame.itertuples(),pred):
                            predictions.append({**ctx,'split':split,'sube_id':row.sube_id,
                              'hedef_donemi':row.hedef_donemi,'gercek_tl':getattr(row,label),'tahmin_tl':float(yhat)})
                    # Importance uses the TRAIN-fitted selected model and VALIDATION only.
                    adapter=PredictionAdapter(selected,algorithm)
                    single=pilot.permutations(adapter,valid,target,view,levels,{c:[c] for c in cols},repeats=8)
                    native,native_method=native_importance(selected,algorithm,cols)
                    for i,row in enumerate(single):
                        f=row['alan_veya_grup']
                        family=pilot.feature_group(f) if hasattr(pilot,'feature_group') else pilot.STRUCTURAL[f][0] if f in pilot.STRUCTURAL else pilot.PROXIES[f][0] if f in pilot.PROXIES else 'gecmis_bakiye'
                        importance.append({**ctx,**row,'native_importance_pct':float(native[i]),
                          'native_method':native_method,'ozellik_grubu':family})
                    feature_groups={}
                    for f in cols:
                        family=pilot.feature_group(f) if hasattr(pilot,'feature_group') else pilot.STRUCTURAL[f][0] if f in pilot.STRUCTURAL else pilot.PROXIES[f][0] if f in pilot.PROXIES else 'gecmis_bakiye'
                        feature_groups.setdefault(family,[]).append(f)
                    for row in pilot.permutations(adapter,valid,target,view,levels,feature_groups,repeats=8):
                        group_importance.append({**ctx,**row})
                    production,prod_seconds=fit(algorithm,block,None,target,view,config,levels,n=n)
                    nxt=future[future['buyuk_sehir_grubu'].eq(group)]
                    next_pred=predict(production,algorithm,nxt,target,view,levels)
                    prod_path=out/'modeller'/f'{model_id}__production.{ext}'
                    save_and_verify(production,algorithm,prod_path,nxt,target,view,levels,next_pred)
                    for row,yhat in zip(nxt.itertuples(),next_pred):
                        forecasts.append({**ctx,'sube_id':row.sube_id,'ozellik_donemi':row.ozellik_donemi,
                          'hedef_donemi':row.hedef_donemi,'referans_aylik_ortalama_tl':getattr(row,target+'_bakiye_t'),
                          'tahmin_aylik_ortalama_tl':float(yhat)})
                    offset_field=pilot.anchor_field(target,view) if hasattr(pilot,'anchor_field') else target+'_bakiye_t' if view=='gecikmeli' else None
                    meta={**ctx,'model_id':model_id,'features':cols,'categorical_levels':{CAT:levels},
                      'best_config':config,'n_estimators':n,'prediction_mode':'delta_plus_anchor' if offset_field else 'absolute_balance',
                      'anchor_feature':offset_field,
                      'eval_training_rows':len(train_valid),'production_training_rows':len(block),
                      'selected_fit_seconds':seconds,'eval_refit_seconds':eval_seconds,'production_fit_seconds':prod_seconds,
                      'tuning_seconds':tuning_seconds,'prediction_ms_per_1000_rows':speed,
                      'evaluation_file':eval_path.name,'production_file':prod_path.name}
                    metadata.append(meta);pilot.save_json(out/'modeller'/f'{model_id}__metadata.json',meta)
                    print(f'{model_id}: TEST WAPE={scores[-1]["wape"]:.4%}, fit={eval_seconds:.3f}s, trees={n}',flush=True)
    perf=pd.DataFrame(scores);pred=pd.DataFrame(predictions)
    totals=[];monthly=[]
    for (target,model,split),part in pred.groupby(['gosterge','model','split'],sort=False):
        totals.append({'gosterge':target,'model':model,'algoritma':part['algoritma'].iloc[0],
                       'gorunum':part['gorunum'].iloc[0],'split':split,**pilot.metrics(part.gercek_tl,part.tahmin_tl)})
    for group in ['ALL']+pilot.GROUPS:
        subset=pred if group=='ALL' else pred[pred.sube_grubu.eq(group)]
        for (target,model,split,period),part in subset.groupby(['gosterge','model','split','hedef_donemi'],sort=False):
            monthly.append({'gosterge':target,'model':model,'split':split,'hedef_donemi':period,
                            'sube_grubu':group,**pilot.metrics(part.gercek_tl,part.tahmin_tl)})
    n_models=len(ALGORITHMS)*len(pilot.GROUPS)*len(pilot.TARGETS)*len(views)
    n_predictors=len(ALGORITHMS)*len(views)+len(baselines)
    assert len(pred)==200*6*3*n_predictors and len(totals)==3*2*n_predictors
    assert len(metadata)==n_models and len(forecasts)==200*3*len(ALGORITHMS)*len(views)
    assert len(importance)==len(pilot.GROUPS)*len(ALGORITHMS)*sum(len(pilot.features(t,v)) for t in pilot.TARGETS for v in views)
    assert len(monthly)==3*3*6*n_predictors
    for name,rows in {'model_metrikleri':perf,'toplam_metrikler':totals,'aylik_donem_sonuclari':monthly,
      'ayar_arama':search,'test_ve_validasyon_tahminleri':pred,'degisken_onemi':importance,
      'grup_onemi':group_importance,'ocak2027_model_tahminleri':forecasts}.items():
        pilot.save_csv(out/'sonuclar'/f'{name}.csv',rows)
    manifest={'seed':pilot.SEED,'synthetic_financial_data':True,'algorithms':ALGORITHMS,'source_version':4 if is_v4 else 3,
      'training_input_hashes':hashes,'original_lightgbm_experiment_preserved':True,
      'models':n_models,'saved_models':n_models*2,'structural_models':18,'lagged_models':18,
      'views':views,'baselines':baselines,'feature_counts':{v:len(pilot.features('kredi',v)) for v in views},'thread_limit':2,
      'search_candidates_per_model':3,'permutation_repeats':8,
      'pipeline_seconds':time.perf_counter()-start,'platform':platform.platform(),
      'software':{p:importlib.metadata.version(p) for p in ['lightgbm','catboost','scikit-learn','numpy','pandas','joblib']},
      'checks':{'same_splits_and_features':True,'test_excluded_from_tuning_and_importance':True,
                'native_model_reload_matches_predictions':True,'production_refit_after_test_frozen':True},
      'timing_note':'Single local CPU run, two threads. Fit includes category encoder training for RF; excludes importance and reload. Full wall time includes tuning, importance, forecasts and saving. Not a repeated speed benchmark.',
      'stability_note':'Three-window stability has not been measured in this run. Historical V3 LightGBM stability is not transferred to V4.'}
    if is_v4:manifest['movement_models']=18
    if (source/'hazirlik_manifest.json').exists():
        preparation=json.loads((source/'hazirlik_manifest.json').read_text())
        manifest['preparation']=preparation
        if source!=out:shutil.copy2(source/'hazirlik_manifest.json',out/'hazirlik_manifest.json')
    pilot.save_json(out/'manifest.json',manifest)
    def records(rows):return pd.DataFrame(rows).replace({np.nan:None}).to_dict('records')
    report={'manifest':manifest,'metrics':records(perf),'total_metrics':records(totals),
      'monthly_metrics':records(monthly),'importance':records(importance),'group_importance':records(group_importance),
      'model_metadata':metadata,'search':records(search)}
    pilot.save_json(out/'sonuclar/rapor_verisi.json',report)
    for name in ['model_karsilastir.py','model_v4_protokol.py','v4_egitim_seti_hazirla.py','pipeline.py','model_karsilastir_dogrula.py','tahmin_karsilastir.py','requirements.txt']:
        p=Path(__file__).parent/name
        if p.exists():shutil.copy2(p,out/'kod'/name)
    print(json.dumps({'status':'complete','models':len(metadata),'pipeline_seconds':manifest['pipeline_seconds'],
       'test_metrics':[r for r in totals if r['split']=='TEST']},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--source',type=Path,default=ROOT/'outputs/lightgbm_ilk_model')
    cli.add_argument('--output',type=Path,default=ROOT/'outputs/model_karsilastirma')
    cli.add_argument('--protocol',choices=['v3','v4'],default='v3')
    args=cli.parse_args()
    if args.protocol=='v4':
        import model_v4_protokol as pilot
    with threadpool_limits(limits=2):main(args.source.resolve(),args.output.resolve())
