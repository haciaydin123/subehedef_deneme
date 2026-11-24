"""Independent readback of FLAML exports, validation decisions and all metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from flaml_tahmin import predict_saved

ROOT=Path(__file__).resolve().parents[1]
REFERENCES={'son_ayi_koru','ay_sonu_koru','egitim_ortalamasi'}

def run(out,source):
    manifest=json.loads((out/'manifest.json').read_text());config=json.loads((out/'run_config.json').read_text())
    for name,digest in manifest['training_input_hashes'].items():
        assert hashlib.sha256((source/'veri'/name).read_bytes()).hexdigest()==digest
        assert hashlib.sha256((out/'veri'/name).read_bytes()).hexdigest()==digest
    for name,digest in manifest['incumbent_hashes'].items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest
    for name,digest in manifest['code_hashes'].items():assert hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()==digest
    frames={name:pd.read_csv(out/'veri'/f'{name}.csv',encoding='utf-8-sig') for name in ['train','validation','test','ocak2027_tahmin_girdisi']}
    assert frames['train'].hedef_donemi.max()<frames['validation'].hedef_donemi.min()
    assert frames['validation'].hedef_donemi.max()<frames['test'].hedef_donemi.min()
    pred=pd.read_csv(out/'sonuclar/tahminler.csv',encoding='utf-8-sig')
    forecasts=pd.read_csv(out/'sonuclar/ocak2027_tahminleri.csv',encoding='utf-8-sig')
    assert len(pred)==28800 and len(forecasts)==1200
    assert not pred.duplicated(['gosterge','sube_id','hedef_donemi','model']).any()
    keyed=pred.groupby(['gosterge','sube_id','hedef_donemi']).gercek_tl.agg(['min','max','count'])
    assert keyed['count'].eq(8).all()
    np.testing.assert_allclose(keyed['min'],keyed['max'],atol=1e-6,rtol=0)
    counts=0;search_count=0
    for path in sorted((out/'modeller').glob('*__metadata.json')):
        m=json.loads(path.read_text());target=m['target'];group=m['group']
        assert len(m['features'])==49 and len(set(m['features']))==49
        assert not any(c.startswith('y_') or c in ['sube_id','hedef_donemi','split'] for c in m['features'])
        # Selection is independently recomputed solely from exported VALIDATION rows.
        p=pred[pred.gosterge.eq(target)&pred.sube_grubu.eq(group)&pred.split.eq('VALIDATION')]
        mae={name:float(np.abs(r.gercek_tl-r.tahmin_tl).mean()) for name,r in p.groupby('model') if name!='secilen_sistem'}
        winner=min(mae,key=lambda name:(mae[name],0 if name in REFERENCES else 1,name))
        assert winner==m['selected_candidate']
        recorded=json.loads((out/'hucreler'/f'{target}__{group}__selection.json').read_text())
        assert 'test_results' not in recorded and recorded['selected_candidate']==winner
        decision=recorded['validation_candidates']
        assert set(decision)==set(mae)
        for name,value in mae.items():np.testing.assert_allclose(value,decision[name]['mae_tl'],atol=1e-6,rtol=0)
        for split in ['VALIDATION','TEST']:
            p=pred[pred.gosterge.eq(target)&pred.sube_grubu.eq(group)&pred.split.eq(split)]
            champion=p[p.model.eq('secilen_sistem')].sort_values(['sube_id','hedef_donemi'])
            chosen=p[p.model.eq(winner)].sort_values(['sube_id','hedef_donemi'])
            np.testing.assert_allclose(champion.tahmin_tl,chosen.tahmin_tl,atol=1e-6,rtol=0)
        artifacts=[('validation_file','validation','VALIDATION','flaml_automl','tahmin_tl'),
          ('evaluation_file','test','TEST','flaml_automl','tahmin_tl'),
          ('production_file','ocak2027_tahmin_girdisi',None,'flaml_automl','tahmin_aylik_ortalama_tl'),
          ('champion_evaluation_file','test','TEST','secilen_sistem','tahmin_tl'),
          ('champion_production_file','ocak2027_tahmin_girdisi',None,'secilen_sistem','tahmin_aylik_ortalama_tl')]
        for field,frame_name,split,name,value in artifacts:
            result=predict_saved(out/'modeller'/m[field],frames[frame_name])
            expected=pred if split else forecasts
            mask=expected.gosterge.eq(target)&expected.sube_grubu.eq(group)&expected.model.eq(name)
            if split:mask &= expected.split.eq(split)
            expected=expected[mask][['sube_id','hedef_donemi',value]].rename(columns={value:'expected'})
            joined=result.merge(expected,on=['sube_id','hedef_donemi'],validate='one_to_one')
            assert len(joined)==len(expected)>0
            np.testing.assert_allclose(joined.tahmin_aylik_ortalama_tl,joined.expected,rtol=0,atol=1e-6)
            counts+=1
        log=out/'aramalar'/f'{target}__{group}.jsonl'
        trials=[r for r in map(json.loads,log.read_text().splitlines()) if 'learner' in r]
        assert len(trials)==m['trials'] and all(r['sample_size']==len(frames['train'][frames['train'].buyuk_sehir_grubu.eq(group)]) for r in trials)
        assert all(r['learner'] in ['lgbm','rf','catboost'] for r in trials)
        assert all(np.isfinite(r['validation_loss']) for r in trials)
        assert all(m['trial_counts'][learner]>0 for learner in ['lgbm','rf','catboost']),m['trial_counts']
        # Best recorded loss is the reported native-export validation MAE.
        np.testing.assert_allclose(min(r['validation_loss'] for r in trials),mae['flaml_automl'],rtol=0,atol=1e-5)
        search_count+=1
    assert counts==30 and search_count==6
    for filename,keys in [('toplam_metrikler.csv',['gosterge','model','split']),
                          ('model_metrikleri.csv',['gosterge','model','split','sube_grubu']),
                          ('aylik_metrikler.csv',['gosterge','model','split','sube_grubu','hedef_donemi'])]:
        data=pd.read_csv(out/'sonuclar'/filename,encoding='utf-8-sig')
        for r in data.to_dict('records'):
            mask=np.ones(len(pred),dtype=bool)
            for key in keys:
                if key=='sube_grubu' and r[key]=='ALL':continue
                mask &= pred[key].eq(r[key]).to_numpy()
            p=pred[mask];y=p.gercek_tl.to_numpy();h=p.tahmin_tl.to_numpy();assert len(y)==r['n']
            actual=[mean_absolute_error(y,h),np.sqrt(mean_squared_error(y,h)),np.abs(y-h).sum()/np.abs(y).sum(),
                    r2_score(y,h),(h.sum()-y.sum())/y.sum()]
            expected=[r[k] for k in ['mae_tl','rmse_tl','wape','r2','toplam_sapma_orani']]
            np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-12)
    manifest['checks'].pop('independent_verification_pending',None)
    manifest['checks'].update(independent_30_bundles_reloaded=True,independent_validation_winners_verified=True,
      independent_all_metric_levels_recomputed=True,all_eight_predictors_share_actuals=True,
      all_three_learners_evaluated_in_each_search=True,trial_losses_match_native_export=True,
      immutable_input_and_incumbent_hashes_verified=True)
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print('PASS: 30 native bundles, 6 validation-only decisions, all 3 learners/cell, source hashes, all metrics and trial losses.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs/flaml_v4')
    p.add_argument('--source',type=Path,default=ROOT/'outputs/model_karsilastirma_v4');a=p.parse_args()
    run(a.output.resolve(),a.source.resolve())
