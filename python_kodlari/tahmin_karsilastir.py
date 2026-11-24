"""Inference for a trusted local comparison model; RF uses joblib serialization."""
import argparse
import json
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

def predict_saved(metadata_path,frame,artifact='production'):
    path=Path(metadata_path);meta=json.loads(path.read_text())
    data=frame[frame['buyuk_sehir_grubu'].eq(meta['sube_grubu'])].copy()
    x=data[meta['features']].copy()
    for col,levels in meta['categorical_levels'].items():x[col]=pd.Categorical(x[col],categories=levels)
    model_path=path.parent/meta[artifact+'_file']
    if meta['algoritma']=='lightgbm':
        model=lgb.Booster(model_file=str(model_path));raw=model.predict(x,num_threads=2)
    elif meta['algoritma']=='catboost':
        model=CatBoostRegressor();model.load_model(str(model_path));raw=model.predict(x,thread_count=2)
    else:
        model=joblib.load(model_path);raw=model.predict(x)
    anchor=data[meta['anchor_feature']].to_numpy(float) if meta['anchor_feature'] else np.zeros(len(data))
    result=data[['sube_id','ozellik_donemi','hedef_donemi']].copy()
    result['tahmin_aylik_ortalama_tl']=np.maximum(0,anchor+raw)
    return result

if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--metadata',type=Path,required=True)
    cli.add_argument('--input',type=Path,required=True);cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--artifact',choices=['evaluation','production'],default='production')
    args=cli.parse_args()
    predict_saved(args.metadata,pd.read_csv(args.input,encoding='utf-8-sig'),args.artifact).to_csv(
        args.output,index=False,encoding='utf-8-sig')
