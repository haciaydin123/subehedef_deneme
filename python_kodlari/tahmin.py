"""Predict from a saved native LightGBM booster and its explicit metadata.

python kod/tahmin.py --metadata modeller/NAME__metadata.json \
 --input veri/ocak2027_tahmin_girdisi.csv --output tahmin.csv
"""
import argparse
import json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

def predict_saved(metadata,input_frame,artifact='production'):
    metadata=Path(metadata)
    spec=json.loads(metadata.read_text(encoding='utf-8'))
    frame=input_frame[input_frame['buyuk_sehir_grubu'].eq(spec['sube_grubu'])].copy()
    columns=spec['features']
    if not set(columns).issubset(frame):raise ValueError('Required feature columns are missing')
    x=frame[columns].copy()
    for col,levels in spec['categorical_levels'].items():x[col]=pd.Categorical(x[col],categories=levels)
    booster=lgb.Booster(model_file=str(metadata.with_name(spec['model_id']+f'__{artifact}.txt')))
    assert booster.feature_name()==columns
    values=booster.predict(x,num_threads=2)
    if spec['prediction_mode']=='delta_plus_current_balance':values+=frame[spec['anchor_feature']].to_numpy(float)
    frame=frame[['sube_id','ozellik_donemi','hedef_donemi']].copy()
    frame['gosterge']=spec['gosterge'];frame['model']=spec['model']
    frame['tahmin_aylik_ortalama_tl']=np.maximum(0,values)
    return frame

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--metadata',required=True,type=Path)
    parser.add_argument('--input',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    predict_saved(args.metadata,pd.read_csv(args.input,encoding='utf-8-sig')).to_csv(args.output,index=False,encoding='utf-8-sig')
