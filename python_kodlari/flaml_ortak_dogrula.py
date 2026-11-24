"""Independent readback of the global algorithm decision and saved models."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from flaml_pipeline import sha
from flaml_tahmin import predict_saved

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/flaml_ortak_v4'

def main(output=None,source=None,search=None):
    OUT=Path(output or ROOT/'outputs/flaml_ortak_v4').resolve()
    source=Path(source or ROOT/'outputs/model_karsilastirma_v4').resolve()
    search=Path(search or ROOT/'outputs/flaml_v4').resolve()
    m=json.loads((OUT/'manifest.json').read_text());d=json.loads((OUT/'ortak_model_secimi.json').read_text())
    for name,digest in m['training_input_hashes'].items():
        assert sha(OUT/'veri'/name)==sha(source/'veri'/name)==digest
    for name,digest in m['incumbent_hashes'].items():assert sha(source/name)==digest
    for name,digest in m['code_hashes'].items():assert sha(ROOT/'python_kodlari'/name)==digest
    for name,digest in m['search_log_hashes'].items():
        assert sha(OUT/'aramalar'/name)==sha(search/'aramalar'/name)==digest
    frames={key:pd.read_csv(OUT/'veri'/f'{key}.csv') for key in ['train','validation','test','ocak2027_tahmin_girdisi']}
    assert frames['train'].hedef_donemi.max()<frames['validation'].hedef_donemi.min()
    assert frames['validation'].hedef_donemi.max()<frames['test'].hedef_donemi.min()
    val=pd.read_csv(OUT/'sonuclar/algoritma_dogrulama_tahminleri.csv')
    assert set(val.model)=={'lgbm','rf','catboost'} and len(val)==5400 and set(val.split)=={'VALIDATION'}
    assert not val.duplicated(['gosterge','sube_id','hedef_donemi','model']).any()
    shared=val.groupby(['gosterge','sube_id','hedef_donemi']).gercek_tl.agg(['min','max','count'])
    assert shared['count'].eq(3).all();np.testing.assert_allclose(shared['min'],shared['max'],rtol=0,atol=1e-6)
    scores={}
    for learner in sorted(val.model.unique()):
        target_scores={}
        for target in ['tl_vadesiz','tl_vadeli','kredi']:
            p=val[val.model.eq(learner)&val.gosterge.eq(target)]
            target_scores[target]=float(np.abs(p.gercek_tl-p.tahmin_tl).sum()/np.abs(p.gercek_tl).sum())
        scores[learner]=float(np.mean(list(target_scores.values())))
        recorded=next(s for s in d['algorithm_scores'] if s['learner']==learner)
        np.testing.assert_allclose(recorded['mean_indicator_wape'],scores[learner],atol=1e-12,rtol=0)
        for target,value in target_scores.items():np.testing.assert_allclose(recorded['indicator_wape'][target],value,atol=1e-12,rtol=0)
    winner=min(scores,key=lambda k:(scores[k],k))
    assert winner==d['selected_learner']==m['selected_learner'] and d['reference_eligible'] is False
    pred=pd.read_csv(OUT/'sonuclar/tahminler.csv');future=pd.read_csv(OUT/'sonuclar/ocak2027_tahminleri.csv')
    assert len(pred)==28800 and len(future)==1200
    assert not pred.duplicated(['gosterge','sube_id','hedef_donemi','model']).any()
    shared=pred.groupby(['gosterge','sube_id','hedef_donemi']).gercek_tl.agg(['min','max','count'])
    assert shared['count'].eq(8).all();np.testing.assert_allclose(shared['min'],shared['max'],atol=1e-6,rtol=0)
    count=0
    for c in d['candidates']:
        target,group,learner=c['target'],c['group'],c['learner']
        logs=[r for r in map(json.loads,(OUT/'aramalar'/f'{target}__{group}.jsonl').read_text().splitlines()) if r.get('learner')==learner]
        record=min(logs,key=lambda r:(r['validation_loss'],r['record_id']))
        assert c['record_id']==record['record_id'] and c['config']==record['config']
        got=predict_saved(OUT/'modeller'/c['validation_file'],frames['validation'])
        expected=val[val.gosterge.eq(target)&val.sube_grubu.eq(group)&val.model.eq(learner)]
        merged=got.merge(expected,on=['sube_id','hedef_donemi'],validate='one_to_one')
        assert len(merged)==len(expected)>0
        np.testing.assert_allclose(merged.tahmin_aylik_ortalama_tl,merged.tahmin_tl,rtol=0,atol=1e-6);count+=1
    metas=[json.loads(p.read_text()) for p in (OUT/'modeller').glob('*__metadata.json')]
    assert len(metas)==6
    for meta in metas:
        assert meta['best_learner']==winner and meta['selected_candidate']=='flaml_automl' and meta['common_algorithm']
        assert len(meta['features'])==49
        for split,key,field in [('TEST','test','evaluation_file'),(None,'ocak2027_tahmin_girdisi','production_file')]:
            got=predict_saved(OUT/'modeller'/meta[field],frames[key]);data=pred if split else future
            data=data[data.gosterge.eq(meta['target'])&data.sube_grubu.eq(meta['group'])]
            if split:data=data[data.split.eq(split)]
            for name in ['flaml_automl','secilen_sistem']:
                expected=data[data.model.eq(name)]
                value='tahmin_tl' if split else 'tahmin_aylik_ortalama_tl'
                joined=got.merge(expected[['sube_id','hedef_donemi',value]].rename(columns={value:'expected'}),on=['sube_id','hedef_donemi'],validate='one_to_one')
                assert len(joined)==len(expected)>0
                np.testing.assert_allclose(joined.tahmin_aylik_ortalama_tl,joined.expected,atol=1e-6,rtol=0)
            count+=1
        p=pred[pred.gosterge.eq(meta['target'])&pred.sube_grubu.eq(meta['group'])&pred.split.eq('VALIDATION')]
        expected=val[val.gosterge.eq(meta['target'])&val.sube_grubu.eq(meta['group'])&val.model.eq(winner)]
        for name in ['flaml_automl','secilen_sistem']:
            joined=p[p.model.eq(name)].merge(expected,on=['sube_id','hedef_donemi'],suffixes=('_p','_v'),validate='one_to_one')
            np.testing.assert_allclose(joined.tahmin_tl_p,joined.tahmin_tl_v,atol=1e-6,rtol=0)
    assert count==30 and len(list((OUT/'modeller').glob('*.joblib')))==30
    for filename,keys in [('toplam_metrikler.csv',['gosterge','model','split']),
         ('model_metrikleri.csv',['gosterge','model','split','sube_grubu']),
         ('aylik_metrikler.csv',['gosterge','model','split','sube_grubu','hedef_donemi'])]:
        for r in pd.read_csv(OUT/'sonuclar'/filename).to_dict('records'):
            p=pred
            for key in keys:
                if key=='sube_grubu' and r[key]=='ALL':continue
                p=p[p[key].eq(r[key])]
            y=p.gercek_tl.to_numpy();h=p.tahmin_tl.to_numpy();assert len(y)==r['n']
            values=[mean_absolute_error(y,h),np.sqrt(mean_squared_error(y,h)),np.abs(y-h).sum()/np.abs(y).sum(),r2_score(y,h),(h.sum()-y.sum())/y.sum()]
            np.testing.assert_allclose(values,[r[k] for k in ['mae_tl','rmse_tl','wape','r2','toplam_sapma_orani']],atol=1e-6,rtol=1e-12)
    m['checks'].update(independent_30_bundles_reloaded=True,global_selection_recomputed_from_validation=True,
        one_family_verified_in_all_six_cells=True,all_metrics_recomputed=True,input_code_and_search_hashes_verified=True)
    (OUT/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2))
    print('PASS: single global algorithm, equal-weight validation selection, 30 model reloads, all metrics and hashes.')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--source',type=Path,default=ROOT/'outputs/model_karsilastirma_v4')
    p.add_argument('--search',type=Path,default=ROOT/'outputs/flaml_v4')
    a=p.parse_args();main(a.output,a.source,a.search)
