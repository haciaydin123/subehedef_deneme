"""Reproducible, time-ordered LightGBM pilot for three branch-level balances.

Run: python kod/pipeline.py --source ../banka_test_verisi_v3 --output .
All financial observations in the supplied V3 data are synthetic.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

SEED=20261120
TARGETS={
 'tl_vadesiz':('tl_vadesiz_ort_bakiye_tl','TL vadesiz mevduat'),
 'tl_vadeli':('tl_vadeli_ort_bakiye_tl','TL vadeli mevduat'),
 'kredi':('kredi_ort_bakiye_tl','Kredi'),
}
GROUPS=['UC_BUYUK_IL','DIGER_ILLER']
STRUCTURAL={
 'musteri_sayisi':('kapasite','Şubedeki müşteri sayısı','kişi'),
 'personel_fte_sentetik':('kapasite','Test portföyüne ait personel kapasitesi','FTE'),
 'ortalama_aylik_gelir_tl':('gelir_profili','Müşterilerin aylık gelir ortalaması','TL'),
 'ortalama_yas':('demografi','Müşteri yaş ortalaması','yıl'),
 'yas_std':('demografi','Yaş standart sapması','yıl'),
 'yas_35_alti_orani':('demografi','35 yaş altı müşteri oranı','0–1'),
 'yas_60_ustu_orani':('demografi','60 yaş ve üzeri müşteri oranı','0–1'),
 'bakim_yukumlulugu_ort':('demografi','Bakmakla yükümlü kişi ortalaması','kişi'),
 'yuksek_egitim_orani':('demografi','Lisans/yüksek lisans/doktora oranı','0–1'),
 'egitim_bilinmiyor_orani':('demografi','Eğitim bilgisi bilinmeyen oranı','0–1'),
 'evli_orani':('demografi','Evli müşteri oranı','0–1'),
 'kaynak_ust_gelir_orani':('gelir_profili','Kaynakta yıllık 80 bin USD ve üzeri bandının oranı; TL gelire çevrilmez','0–1'),
 'kaynak_gelir_bilinmiyor_orani':('gelir_profili','Kaynak gelir bandı bilinmeyen oranı','0–1'),
 'kaynak_pasif_ay_ort':('iliski_aktivite','Kaynak profilindeki son 12 ay pasif ay ortalaması; simülasyonda başlangıçta bilinen profil','ay'),
 'mobil_aktif_orani':('iliski_aktivite','Mobil aktif müşteri oranı','0–1'),
 'maas_musterisi_orani':('iliski_aktivite','Maaş müşterisi oranı','0–1'),
 'emekli_maasi_orani':('iliski_aktivite','Emekli maaşı müşterisi oranı','0–1'),
 'ana_banka_orani':('iliski_aktivite','Ana banka ilişkisi oranı','0–1'),
 'ekonomik_yapi_sentetik':('bolgesel_kosullar','Sentetik ekonomik yapı; native categorical','kategori'),
 'sosyoekonomik_skor_sentetik':('bolgesel_kosullar','Sentetik ekonomik çevre skoru; SEGE değildir','0–100'),
 'rakip_sube_sayisi_sentetik':('bolgesel_kosullar','Sentetik rakip şube sayısı','adet'),
 'nufus_sentetik':('bolgesel_kosullar','Sentetik ilçe nüfusu','kişi'),
 'takvim_ay_sin':('takvim','Tahmin edilecek ayın bilinen takvim sinüsü','-1–1'),
 'takvim_ay_cos':('takvim','Tahmin edilecek ayın bilinen takvim kosinüsü','-1–1'),
 'zaman_indeksi':('takvim','Bilinen takvim sıra numarası','ay'),
}
PROXIES={
 'portfoy_musteri_orani':('varlik_segmenti','t ayındaki portföy müşteri payı; mevcut banka varlığından türetilir','0–1'),
 'segment_s1_orani':('varlik_segmenti','t ayındaki S1 payı; varlık eşiğinden türetilir','0–1'),
 'segment_s2_orani':('varlik_segmenti','t ayındaki S2 payı; varlık eşiğinden türetilir','0–1'),
 'segment_s5_orani':('varlik_segmenti','t ayındaki S5 payı; varlık eşiğinden türetilir','0–1'),
 'ortalama_varlik_tl':('varlik_segmenti','t ayındaki banka mevduat ve yatırım varlık ortalaması; mevcut üretim göstergesidir','TL'),
}
CONFIGS=[
 {'config_id':'A','num_leaves':7,'max_depth':3,'min_child_samples':40,'reg_lambda':10.},
 {'config_id':'B','num_leaves':15,'max_depth':4,'min_child_samples':40,'reg_lambda':10.},
 {'config_id':'C','num_leaves':15,'max_depth':4,'min_child_samples':20,'reg_lambda':25.},
]

def save_json(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def save_csv(path,data):
    pd.DataFrame(data).to_csv(path,index=False,encoding='utf-8-sig',float_format='%.17g')

def metrics(y,pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y=np.asarray(y,float);pred=np.maximum(0,np.asarray(pred,float))
    return {'n':len(y),'mae_tl':float(mean_absolute_error(y,pred)),
      'rmse_tl':float(math.sqrt(mean_squared_error(y,pred))),
      'wape':float(np.abs(y-pred).sum()/np.abs(y).sum()),
      'r2':float(r2_score(y,pred)),
      'gercek_toplam_tl':float(y.sum()),'tahmin_toplam_tl':float(pred.sum()),
      'toplam_sapma_orani':float((pred.sum()-y.sum())/y.sum())}

def prepare(source,out):
    b=pd.read_csv(source/'sube_aylik.csv',encoding='utf-8-sig').sort_values(['sube_id','donem']).reset_index(drop=True)
    p=pd.read_csv(source/'musteriler.csv',encoding='utf-8-sig')
    districts=pd.read_csv(source/'bolgeler_sentetik.csv',encoding='utf-8-sig')
    assert len(b)==5200 and not b.duplicated(['sube_id','donem']).any()
    assert b.groupby('sube_id').size().eq(26).all() and b['sube_id'].nunique()==200
    assert b['is_birimi'].eq('BIREYSEL').all()
    f=p.assign(
      yas_std=p['yas'],yas_35_alti_orani=(p['yas']<35).astype(float),
      yas_60_ustu_orani=(p['yas']>=60).astype(float),
      bakim_yukumlulugu_ort=p['bakmakla_yukumlu_kisi_sayisi'],
      yuksek_egitim_orani=p['egitim_duzeyi'].isin(['LISANS','YUKSEK_LISANS','DOKTORA']).astype(float),
      egitim_bilinmiyor_orani=p['egitim_duzeyi'].eq('BILINMIYOR').astype(float),
      evli_orani=p['medeni_durum'].eq('EVLI').astype(float),
      kaynak_ust_gelir_orani=p['kaynak_gelir_grubu'].isin(['80_120K_USD','GE_120K_USD']).astype(float),
      kaynak_gelir_bilinmiyor_orani=p['kaynak_gelir_grubu'].eq('BILINMIYOR').astype(float),
      kaynak_pasif_ay_ort=p['kaynak_pasif_ay_sayisi_12ay'])
    static_cols=list(STRUCTURAL)[4:14]
    agg={c:('std' if c=='yas_std' else 'mean') for c in static_cols}
    dem=f.groupby('sube_id',as_index=False).agg(agg)
    assert dem.notna().all().all()
    b=b.merge(dem,on='sube_id',how='left',validate='many_to_one')
    b=b.merge(districts[['ilce_id','nufus_sentetik']],on='ilce_id',how='left',validate='many_to_one')
    pcounts=p.groupby('sube_id').size()
    assert b['musteri_sayisi'].eq(b['sube_id'].map(pcounts)).all()
    b['ortalama_aylik_gelir_tl']=b['aylik_gelir_tl']/b['musteri_sayisi']
    b['emekli_maasi_orani']=b['emekli_maasi_musterisi_musteri_sayisi']/b['musteri_sayisi']
    b['ana_banka_orani']=b['ana_banka_musterisi_musteri_sayisi']/b['musteri_sayisi']
    b['portfoy_musteri_orani']=b['portfoy_musteri_sayisi']/b['musteri_sayisi']
    b['ortalama_varlik_tl']=b['toplam_varlik_ort_tl']/b['musteri_sayisi']
    seg=pd.read_csv(source/'musteri_aylik.csv',encoding='utf-8-sig',usecols=['donem','sube_id','musteri_segmenti'])
    c=pd.crosstab([seg['sube_id'],seg['donem']],seg['musteri_segmenti']).reset_index()
    for segment,field in [('S1_BASLANGIC','segment_s1_orani'),('S2_BIRIKIM','segment_s2_orani'),('S5_OZEL','segment_s5_orani')]:
        c[field]=c[segment]
    b=b.merge(c[['sube_id','donem','segment_s1_orani','segment_s2_orani','segment_s5_orani']],
              on=['sube_id','donem'],validate='one_to_one')
    for col in ['segment_s1_orani','segment_s2_orani','segment_s5_orani']:b[col]/=b['musteri_sayisi']
    b=b.sort_values(['sube_id','donem']).reset_index(drop=True).copy()
    b['ozellik_donemi']=b['donem']
    periods=pd.PeriodIndex(b['donem'],freq='M')
    b['hedef_donemi']=(periods+1).astype(str)
    month=(periods+1).month
    b['takvim_ay_sin']=np.sin(2*np.pi*month/12)
    b['takvim_ay_cos']=np.cos(2*np.pi*month/12)
    b['zaman_indeksi']=[x.ordinal-pd.Period('2024-11',freq='M').ordinal for x in periods]
    lag_cols=[]
    for name,(col,_) in TARGETS.items():
        g=b.groupby('sube_id',sort=False)[col]
        b['y_'+name+'_tl']=g.shift(-1)
        b[name+'_bakiye_t']=b[col]
        b[name+'_bakiye_t_eksi_1']=g.shift(1)
        b[name+'_son3ay_ort_tl']=g.transform(lambda s:s.rolling(3,min_periods=3).mean())
        previous=b[name+'_bakiye_t_eksi_1']
        b[name+'_son_ay_buyume']=(b[col]/previous-1).where(previous>0,0.)
        lag_cols += [name+'_bakiye_t',name+'_bakiye_t_eksi_1',name+'_son3ay_ort_tl',name+'_son_ay_buyume']
    labels=['y_'+name+'_tl' for name in TARGETS]
    inputs=list(STRUCTURAL)+list(PROXIES)+lag_cols
    keep=['sube_id','buyuk_sehir_grubu','ozellik_donemi','hedef_donemi']+inputs+labels
    data=b.loc[b['ozellik_donemi'].ge('2025-01') & b[labels].notna().all(axis=1),keep].copy()
    assert len(data)==4600
    data['split']=np.select([data['hedef_donemi'].le('2026-06'),data['hedef_donemi'].le('2026-09')],['TRAIN','VALIDATION'],default='TEST')
    data=data.sort_values(['hedef_donemi','sube_id']).reset_index(drop=True)
    future=b.loc[b['ozellik_donemi'].eq('2026-12'),[c for c in keep if c not in labels]].copy()
    future['split']='FORECAST'
    assert len(future)==200 and future['hedef_donemi'].eq('2027-01').all()
    assert data.groupby('split').size().to_dict()=={'TEST':600,'TRAIN':3400,'VALIDATION':600}
    assert data[inputs].notna().all().all() and future[inputs].notna().all().all()
    assert all(pd.Period(r.hedef_donemi,freq='M')==pd.Period(r.ozellik_donemi,freq='M')+1 for r in data.itertuples())
    for split in ['TRAIN','VALIDATION','TEST']:
        save_csv(out/'veri'/f'{split.lower()}.csv',data[data['split'].eq(split)])
    save_csv(out/'veri'/'tum_etiketli_veri.csv',data)
    save_csv(out/'veri'/'ocak2027_tahmin_girdisi.csv',future)
    specs=[]
    for name,(group,desc,unit) in {**STRUCTURAL,**PROXIES}.items():
        specs.append({'alan':name,'ozellik_grubu':group,'aciklama':desc,'birim':unit,
          'yapisal_modelde':name in STRUCTURAL,'gecikmeli_modelde':True,
          'kumeleme_adayi':group not in ('takvim','varlik_segmenti'),
          'zaman':'t ayı; takvim alanlarında t+1 ayı önceden bilinir'})
    for name in lag_cols:
        specs.append({'alan':name,'ozellik_grubu':'gecmis_bakiye','aciklama':'Göstergeye ait t/t-1/geçmiş 3 ay veya son aylık büyüme; yalnızca ilgili gösterge modeline girer',
          'birim':'oran' if name.endswith('buyume') else 'TL','yapisal_modelde':False,
          'gecikmeli_modelde':True,'kumeleme_adayi':False,'zaman':'en geç t ayı'})
    save_csv(out/'veri'/'ozellik_sozlugu.csv',specs)
    split_rows=[]
    for split,frame in data.groupby('split',sort=False):
        split_rows.append({'split':split,'satir':len(frame),'sube':frame['sube_id'].nunique(),
          'ilk_ozellik_ayi':frame['ozellik_donemi'].min(),'son_ozellik_ayi':frame['ozellik_donemi'].max(),
          'ilk_hedef_ayi':frame['hedef_donemi'].min(),'son_hedef_ayi':frame['hedef_donemi'].max()})
    save_csv(out/'veri'/'zaman_ayrimi.csv',split_rows)
    return data,future,pd.DataFrame(specs),split_rows

def features(target,view):
    columns=list(STRUCTURAL)
    if view=='gecikmeli':
        columns+=list(PROXIES)+[target+'_bakiye_t',target+'_bakiye_t_eksi_1',target+'_son3ay_ort_tl',target+'_son_ay_buyume']
    assert len(columns)==len(set(columns))
    assert not any(c.startswith('y_') or c in ('sube_id','ilce_id','hedef_donemi') for c in columns)
    return columns

def design(frame,columns,levels):
    x=frame[columns].copy()
    x['ekonomik_yapi_sentetik']=pd.Categorical(x['ekonomik_yapi_sentetik'],categories=levels)
    return x

def anchor(frame,target,view):
    return frame[target+'_bakiye_t'].to_numpy(float) if view=='gecikmeli' else np.zeros(len(frame))

def make_model(config,n=800):
    params={k:v for k,v in config.items() if k!='config_id'}
    return lgb.LGBMRegressor(objective='regression',metric='l1',learning_rate=.04,n_estimators=n,
      random_state=SEED,n_jobs=2,deterministic=True,force_col_wise=True,verbosity=-1,
      colsample_bytree=.90,subsample=.85,subsample_freq=1,importance_type='gain',**params)

def fit_model(train,valid,target,view,config,levels,n=None):
    columns=features(target,view)
    x=design(train,columns,levels);y=train['y_'+target+'_tl'].to_numpy()-anchor(train,target,view)
    model=make_model(config,n or 800)
    started=time.perf_counter()
    if valid is not None:
        xv=design(valid,columns,levels);yv=valid['y_'+target+'_tl'].to_numpy()-anchor(valid,target,view)
        model.fit(x,y,eval_X=xv,eval_y=yv,eval_metric='l1',callbacks=[lgb.early_stopping(50,first_metric_only=True,verbose=False)])
    else:model.fit(x,y)
    return model,time.perf_counter()-started

def predict(model,frame,target,view,levels):
    return np.maximum(0,anchor(frame,target,view)+model.predict(design(frame,features(target,view),levels)))

def permutations(model,frame,target,view,levels,feature_groups,repeats=8):
    from sklearn.metrics import mean_absolute_error
    frame=frame.sort_values(['hedef_donemi','sube_id']).reset_index(drop=True)
    cols=features(target,view);x=design(frame,cols,levels)
    y=frame['y_'+target+'_tl'].to_numpy(float);base=anchor(frame,target,view)
    base_pred=np.maximum(0,base+model.predict(x))
    baseline=mean_absolute_error(y,base_pred)
    n_branch=frame['sube_id'].nunique();n_month=frame['hedef_donemi'].nunique()
    assert len(frame)==n_branch*n_month
    ids=frame['sube_id'].to_numpy().reshape(n_month,n_branch)
    assert (ids==ids[0]).all()
    rng=np.random.default_rng(SEED)
    takes=[np.arange(len(frame)).reshape(n_month,n_branch)[:,rng.permutation(n_branch)].ravel() for _ in range(repeats)]
    results=[]
    for label,group_cols in feature_groups.items():
        diffs=[]
        for take in takes:
            changed=x.copy()
            for col in group_cols:changed[col]=x[col].iloc[take].array
            # For residual forecasts, the anchor is part of the predictor. If
            # current balance is shuffled, shuffle its anchor too.
            shifted=base[take] if view=='gecikmeli' and target+'_bakiye_t' in group_cols else base
            pred=np.maximum(0,shifted+model.predict(changed))
            diffs.append(mean_absolute_error(y,pred)-baseline)
        results.append({'alan_veya_grup':label,'delta_mae_tl':float(np.mean(diffs)),
          'delta_mae_std_tl':float(np.std(diffs,ddof=1)),
          'normalize_delta_mae':float(np.mean(diffs)/y.mean()),'tekrar':repeats})
    return results

def run(source,out):
    global lgb
    import lightgbm as lgb
    wall_start=time.perf_counter()
    for name in ('veri','sonuclar','modeller','grafikler'): (out/name).mkdir(parents=True,exist_ok=True)
    prepared_start=time.perf_counter()
    data,future,specs,splits=prepare(source,out)
    preprocess_seconds=time.perf_counter()-prepared_start
    print(f'Prepared {len(data)} labeled branch-months, {len(future)} January inputs',flush=True)
    scores=[];search=[];pred_rows=[];forecast_rows=[];importance=[];group_importance=[];fold_imp=[];fold_scores=[];model_meta=[]
    for group in GROUPS:
        block=data[data['buyuk_sehir_grubu'].eq(group)].copy()
        train=block[block['split'].eq('TRAIN')];valid=block[block['split'].eq('VALIDATION')];test=block[block['split'].eq('TEST')]
        levels=sorted(train['ekonomik_yapi_sentetik'].unique().tolist())
        for target,(_,display) in TARGETS.items():
            label='y_'+target+'_tl'
            context={'sube_grubu':group,'gosterge':target}
            for split,frame in [('VALIDATION',valid),('TEST',test)]:
                for baseline,pred in [('son_ayi_koru',frame[target+'_bakiye_t'].to_numpy()),('egitim_ortalamasi',np.repeat(train[label].mean(),len(frame)))]:
                    scores.append({**context,'model':baseline,'split':split,**metrics(frame[label],pred),'egitim_saniye':0.,'agac_sayisi':0})
                    for row,yhat in zip(frame.itertuples(),pred):
                        pred_rows.append({**context,'model':baseline,'split':split,'sube_id':row.sube_id,
                           'hedef_donemi':row.hedef_donemi,'gercek_tl':getattr(row,label),'tahmin_tl':max(0,float(yhat))})
            for view in ('yapisal','gecikmeli'):
                all_fit_start=time.perf_counter();candidates=[]
                for config in CONFIGS:
                    model,seconds=fit_model(train,valid,target,view,config,levels)
                    score=metrics(valid[label],predict(model,valid,target,view,levels))
                    search.append({**context,'model':view,**config,'best_iteration':int(model.best_iteration_),
                                   'valid_mae_tl':score['mae_tl'],'valid_wape':score['wape'],'fit_seconds':seconds})
                    candidates.append((score['mae_tl'],config,model,seconds))
                _,best_config,selected,selected_seconds=min(candidates,key=lambda c:c[0])
                n=max(1,int(selected.best_iteration_));columns=features(target,view)
                valid_pred=predict(selected,valid,target,view,levels)
                scores.append({**context,'model':view,'split':'VALIDATION',**metrics(valid[label],valid_pred),
                               'egitim_saniye':selected_seconds,'agac_sayisi':n})
                gains=selected.booster_.feature_importance(importance_type='gain')
                gain_pct=100*gains/max(1,gains.sum())
                single=permutations(selected,valid,target,view,levels,{c:[c] for c in columns})
                for i,row in enumerate(single):
                    importance.append({**context,'model':view,'fold':'2026_07_09',**row,
                      'gain_yuzde':float(gain_pct[i]),'ozellik_grubu':dict(zip(specs['alan'],specs['ozellik_grubu']))[row['alan_veya_grup']]})
                fg={}
                for col in columns:
                    g=STRUCTURAL[col][0] if col in STRUCTURAL else PROXIES[col][0] if col in PROXIES else 'gecmis_bakiye'
                    fg.setdefault(g,[]).append(col)
                for row in permutations(selected,valid,target,view,levels,fg):
                    group_importance.append({**context,'model':view,**row})
                if view=='yapisal':
                    for row in single:fold_imp.append({**context,'fold':'2026_07_09',**row})
                    for end,first,last,fold in [('2025-12','2026-01','2026-03','2026_01_03'),('2026-03','2026-04','2026-06','2026_04_06')]:
                        tr=block[block['hedef_donemi'].le(end)]
                        va=block[block['hedef_donemi'].between(first,last)]
                        # Only historical TRAIN periods participate in these checks.
                        assert va['split'].eq('TRAIN').all()
                        # Select within THIS fold; later validation periods do
                        # not choose parameters for earlier historical folds.
                        local=[]
                        for cfg in CONFIGS:
                            candidate,seconds=fit_model(tr,va,target,view,cfg,levels)
                            err=metrics(va[label],predict(candidate,va,target,view,levels))['mae_tl']
                            local.append((err,cfg,candidate,seconds))
                        _,fold_config,fm,fs=min(local,key=lambda v:v[0])
                        fold_scores.append({**context,'fold':fold,'train_rows':len(tr),'validation_rows':len(va),
                          'config_id':fold_config['config_id'],**metrics(va[label],predict(fm,va,target,view,levels)),'egitim_saniye':fs})
                        for row in permutations(fm,va,target,view,levels,{c:[c] for c in columns},repeats=5):
                            fold_imp.append({**context,'fold':fold,**row})
                # Freeze parameters and tree count before touching TEST labels.
                train_valid=block[block['split'].isin(['TRAIN','VALIDATION'])]
                evaluation,eval_seconds=fit_model(train_valid,None,target,view,best_config,levels,n=n)
                test_pred=predict(evaluation,test,target,view,levels)
                scores.append({**context,'model':view,'split':'TEST',**metrics(test[label],test_pred),
                               'egitim_saniye':eval_seconds,'agac_sayisi':n})
                model_id=f'{target}__{group}__{view}'
                eval_path=out/'modeller'/f'{model_id}__evaluation.txt'
                evaluation.booster_.save_model(str(eval_path))
                restored=lgb.Booster(model_file=str(eval_path))
                check=np.maximum(0,anchor(test,target,view)+restored.predict(design(test,columns,levels),num_threads=2))
                assert np.allclose(check,test_pred,rtol=0,atol=1e-6)
                for split,frame,pred in [('VALIDATION',valid,valid_pred),('TEST',test,test_pred)]:
                    for row,yhat in zip(frame.itertuples(),pred):
                        pred_rows.append({**context,'model':view,'split':split,'sube_id':row.sube_id,
                          'hedef_donemi':row.hedef_donemi,'gercek_tl':getattr(row,label),'tahmin_tl':float(yhat)})
                # The test scores above stay frozen. The forecast-only artifact
                # can now refit on all observed labeled periods through Dec 2026.
                production,production_seconds=fit_model(block,None,target,view,best_config,levels,n=n)
                prod_path=out/'modeller'/f'{model_id}__production.txt'
                production.booster_.save_model(str(prod_path))
                next_frame=future[future['buyuk_sehir_grubu'].eq(group)]
                next_pred=predict(production,next_frame,target,view,levels)
                for row,yhat in zip(next_frame.itertuples(),next_pred):
                    forecast_rows.append({**context,'model':view,'sube_id':row.sube_id,
                      'ozellik_donemi':row.ozellik_donemi,'hedef_donemi':row.hedef_donemi,
                      'referans_aylik_ortalama_tl':getattr(row,target+'_bakiye_t'),
                      'tahmin_aylik_ortalama_tl':float(yhat)})
                meta={**context,'model':view,'model_id':model_id,'display':display,'features':columns,
                  'categorical_levels':{'ekonomik_yapi_sentetik':levels},'best_config':best_config,'n_estimators':n,
                  'prediction_mode':'delta_plus_current_balance' if view=='gecikmeli' else 'absolute_balance',
                  'anchor_feature':target+'_bakiye_t' if view=='gecikmeli' else None,
                  'eval_training_rows':len(train_valid),'production_training_rows':len(block),
                  'selected_fit_seconds':selected_seconds,'eval_refit_seconds':eval_seconds,
                  'production_fit_seconds':production_seconds,'model_pipeline_seconds':time.perf_counter()-all_fit_start}
                save_json(out/'modeller'/f'{model_id}__metadata.json',meta)
                model_meta.append(meta)
                print(f'{model_id}: trees={n}, TEST WAPE={scores[-1]["wape"]:.4f}, fit={eval_seconds:.3f}s',flush=True)

    perf=pd.DataFrame(scores);predictions=pd.DataFrame(pred_rows);fi=pd.DataFrame(importance);folds=pd.DataFrame(fold_imp)
    stable=folds.groupby(['sube_grubu','gosterge','alan_veya_grup'],as_index=False).agg(
      pozitif_fold_sayisi=('delta_mae_tl',lambda s:int((s>0).sum())),
      ortalama_normalize_onem=('normalize_delta_mae','mean'),fold_sayisi=('fold','nunique'))
    stable['kumeleme_adayi']=~stable['alan_veya_grup'].isin([c for c in STRUCTURAL if STRUCTURAL[c][0]=='takvim'])
    stable['kararli_aday']=stable['kumeleme_adayi'] & stable['pozitif_fold_sayisi'].ge(2) & stable['ortalama_normalize_onem'].gt(0)
    stable=stable.sort_values(['sube_grubu','gosterge','ortalama_normalize_onem'],ascending=[True,True,False])
    # Aggregate metrics from row-level predictions, never average group WAPEs.
    aggregate=[]
    for (target,view,split),part in predictions.groupby(['gosterge','model','split'],sort=False):
        aggregate.append({'gosterge':target,'model':view,'split':split,**metrics(part['gercek_tl'],part['tahmin_tl'])})
    per_month=[]
    for (target,view,period),part in predictions.groupby(['gosterge','model','hedef_donemi'],sort=False):
        assert part['split'].nunique()==1
        per_month.append({'gosterge':target,'model':view,'hedef_donemi':period,'split':part['split'].iloc[0],**metrics(part['gercek_tl'],part['tahmin_tl'])})
    outputs={'model_metrikleri.csv':perf,'toplam_metrikler.csv':aggregate,'aylik_donem_sonuclari.csv':per_month,
      'ayar_arama.csv':search,'test_ve_validasyon_tahminleri.csv':predictions,'degisken_onemi.csv':fi,
      'grup_onemi.csv':group_importance,'donemsel_onem.csv':folds,'donemsel_performans.csv':fold_scores,
      'kararli_degisken_adaylari.csv':stable,'ocak2027_model_tahminleri.csv':forecast_rows}
    for filename,frame in outputs.items():save_csv(out/'sonuclar'/filename,frame)
    # Check that the row-level metric files can be independently reconciled.
    assert len(predictions)==14400 # 200 branches × 6 months × 3 indicators × 4 predictors.
    assert len(forecast_rows)==1200
    assert len(model_meta)==12 and len(list((out/'modeller').glob('*.txt')))==24
    assert len(stable)==150
    source_manifest=json.loads((source/'manifest.json').read_text())
    source_hashes={}
    for file in ['sube_aylik.csv','musteriler.csv','musteri_aylik.csv','bolgeler_sentetik.csv']:
        digest=hashlib.sha256((source/file).read_bytes()).hexdigest()
        assert digest==source_manifest['csv_files'][file]['sha256']
        source_hashes[file]=digest
    manifest={'seed':SEED,'source_version':3,'source_hashes':source_hashes,'synthetic_financial_data':True,
      'forecast_horizon_months':1,'unit':'branch-month','groups':GROUPS,'target_mapping':TARGETS,
      'labeled_rows':len(data),'forecast_rows':len(future),'splits':splits,
      'structural_features':len(STRUCTURAL),'forecast_features_per_indicator':len(STRUCTURAL)+len(PROXIES)+4,
      'structural_models':6,'comparison_models':6,'total_saved_boosters':24,
      'preprocessing_seconds':preprocess_seconds,'pipeline_seconds':time.perf_counter()-wall_start,
      'software':{p:importlib.metadata.version(p) for p in ['lightgbm','scikit-learn','numpy','pandas']},
      'checks':{'no_random_time_split':True,'one_month_label_alignment':True,'no_target_columns_in_features':True,
        'test_excluded_from_parameter_and_feature_selection':True,'native_model_reload_matches_predictions':True,
        'production_refit_after_test_metrics_frozen':True,'budget_growth_not_used_for_training':True}}
    save_json(out/'manifest.json',manifest)
    # Compact, typed intermediate for the human-readable workbook report.
    def records(frame):return pd.DataFrame(frame).replace({np.nan:None}).to_dict('records')
    report={'manifest':manifest,'splits':splits,'metrics':records(perf),'total_metrics':records(aggregate),
      'features':records(specs),'importance':records(fi),'stability':records(stable),
      'group_importance':records(group_importance),'fold_scores':records(fold_scores)}
    save_json(out/'sonuclar'/'rapor_verisi.json',report)
    print(json.dumps({'total_test_metrics':[r for r in aggregate if r['split']=='TEST'],
       'pipeline_seconds':manifest['pipeline_seconds']},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--source',type=Path,required=True);cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--prepare-only',action='store_true')
    args=cli.parse_args()
    if args.prepare_only:
        data,future,specs,splits=prepare(args.source.resolve(),args.output.resolve())
        print(json.dumps({'labeled_rows':len(data),'forecast_rows':len(future),'splits':splits},ensure_ascii=False,indent=2))
    else:run(args.source.resolve(),args.output.resolve())
