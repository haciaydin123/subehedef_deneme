"""TRAIN/VALIDATION-only integration check for all three FLAML learners."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from flaml_pipeline import make_search,from_search,refit,categories,LEARNERS
from flaml_tahmin import design,predict_bundle
import model_v4_protokol as protocol

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=ROOT/'outputs/model_karsilastirma_v4/veri';out=ROOT/'.work/flaml_smoke';out.mkdir(exist_ok=True)
    train=pd.read_csv(p/'train.csv',encoding='utf-8-sig');valid=pd.read_csv(p/'validation.csv',encoding='utf-8-sig')
    train=train[train.buyuk_sehir_grubu.eq('UC_BUYUK_IL')];valid=valid[valid.buyuk_sehir_grubu.eq('UC_BUYUK_IL')]
    target='tl_vadesiz';cols=protocol.features(target,'hareketli');levels=categories(train);anchor=protocol.anchor_field(target,'hareketli')
    x=design(train,cols,levels);xv=design(valid,cols,levels)
    for learner in LEARNERS:
        auto=make_search(10,out/(learner+'.jsonl'),max_iter=3)
        auto.fit(X_train=x,y_train=train['y_'+target+'_tl'].to_numpy()-train[anchor].to_numpy(),
          X_val=xv,y_val=valid['y_'+target+'_tl'].to_numpy()-valid[anchor].to_numpy(),estimator_list=[learner])
        assert auto.best_estimator==learner
        bundle=from_search(auto,valid,target,levels)
        np.testing.assert_allclose(auto.best_loss,protocol.metrics(valid['y_'+target+'_tl'],predict_bundle(bundle,valid))['mae_tl'],rtol=0,atol=1e-5)
        fitted,seconds=refit(bundle,train,target)
        file=out/(learner+'.joblib');joblib.dump(fitted,file)
        np.testing.assert_allclose(predict_bundle(joblib.load(file),valid),predict_bundle(fitted,valid),rtol=0,atol=1e-6)
        print('PASS',learner,'native export, refit and reload; fit seconds',round(seconds,4),flush=True)
    print('PASS: no TEST labels used in this compatibility check.',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
