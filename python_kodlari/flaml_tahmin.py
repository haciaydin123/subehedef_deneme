"""Inference from a trusted local native model bundle; FLAML is not needed."""
from pathlib import Path
import argparse
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

def design(frame,features,categories,encoding='native'):
    x=frame[features].copy()
    for col,levels in categories.items():
        values=x[col].astype(object).where(x[col].isin(levels),'__BILINMIYOR__')
        x[col]=pd.Categorical(values,categories=levels)
        if encoding=='codes':x[col]=x[col].cat.codes
    return x

def predict_bundle(bundle,frame):
    if bundle['kind']=='feature':return np.maximum(0,frame[bundle['reference_feature']].to_numpy(float))
    if bundle['kind']=='constant':return np.repeat(bundle['constant_tl'],len(frame))
    x=design(frame,bundle['features'],bundle['categories'],bundle['encoding'])
    model=bundle['model']
    if isinstance(model,lgb.Booster):pred=model.predict(x,num_threads=2)
    elif isinstance(model,CatBoostRegressor):pred=model.predict(x,thread_count=2)
    else:pred=model.predict(x)
    anchor=frame[bundle['anchor_feature']].to_numpy(float) if bundle['anchor_feature'] else 0.
    return np.maximum(0,anchor+np.asarray(pred,float))

def predict_saved(path,frame):
    bundle=joblib.load(path)
    selected=frame[frame['buyuk_sehir_grubu'].eq(bundle['group'])]
    result=selected[['sube_id','ozellik_donemi','hedef_donemi']].copy()
    result['tahmin_aylik_ortalama_tl']=predict_bundle(bundle,selected)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();predict_saved(a.model,pd.read_csv(a.input,encoding='utf-8-sig')).to_csv(a.output,index=False,encoding='utf-8-sig')
