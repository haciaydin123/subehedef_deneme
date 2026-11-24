"""Independent metric recomputation and saved-artifact inference verification."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from tahmin_karsilastir import predict_saved

ROOT=Path(__file__).resolve().parents[1]

def run(source,out,raw_source=None):
    manifest=json.loads((out/'manifest.json').read_text())
    for filename,digest in manifest['training_input_hashes'].items():
        assert hashlib.sha256((source/'veri'/filename).read_bytes()).hexdigest()==digest
    test=pd.read_csv(source/'veri/test.csv',encoding='utf-8-sig')
    future=pd.read_csv(source/'veri/ocak2027_tahmin_girdisi.csv',encoding='utf-8-sig')
    pred=pd.read_csv(out/'sonuclar/test_ve_validasyon_tahminleri.csv',encoding='utf-8-sig')
    forecast=pd.read_csv(out/'sonuclar/ocak2027_model_tahminleri.csv',encoding='utf-8-sig')
    model_count=manifest['models'];predictors=len(pred['model'].unique())
    assert len(pred)==200*6*3*predictors and len(forecast)==200*model_count//2
    assert not pred.duplicated(['gosterge','model','sube_id','hedef_donemi']).any()
    # Every model sees exactly the same keyed actuals.
    actuals=pred.groupby(['gosterge','sube_id','hedef_donemi']).gercek_tl.agg(['min','max','count'])
    assert actuals['count'].eq(predictors).all()
    np.testing.assert_allclose(actuals['min'],actuals['max'],atol=1e-6,rtol=0)
    count=0
    for path in sorted((out/'modeller').glob('*__metadata.json')):
        m=json.loads(path.read_text());cols=m['features']
        assert len(cols)==manifest['feature_counts'][m['gorunum']]
        assert not any(c.startswith('y_') or c in ['sube_id','hedef_donemi','split'] for c in cols)
        if m['gorunum']=='yapisal':assert not any(any(s in c for s in ['bakiye','varlik','portfoy','segment']) for c in cols)
        for frame,expected,artifact,value in [
          (test,pred[pred.split.eq('TEST')],'evaluation','tahmin_tl'),
          (future,forecast,'production','tahmin_aylik_ortalama_tl')]:
            result=predict_saved(path,frame,artifact)
            e=expected[expected.gosterge.eq(m['gosterge'])&expected.model.eq(m['model'])&expected.sube_grubu.eq(m['sube_grubu'])]
            merged=result.merge(e[['sube_id','hedef_donemi',value]].rename(columns={value:'expected'}),
                                on=['sube_id','hedef_donemi'],validate='one_to_one')
            assert len(merged)==len(e)>0
            np.testing.assert_allclose(merged['tahmin_aylik_ortalama_tl'],merged.expected,rtol=0,atol=1e-6)
            count+=1
    assert count==model_count*2
    if manifest.get('source_version')==4:
        preparation=json.loads((out/'hazirlik_manifest.json').read_text())
        for name,digest in preparation['source_hashes'].items():
            raw_source=raw_source or ROOT/'outputs/banka_test_verisi_v4'
            assert hashlib.sha256((raw_source/name).read_bytes()).hexdigest()==digest
        raw=pd.read_csv(raw_source/'sube_model_girdileri_aylik.csv',encoding='utf-8-sig').set_index(['sube_id','donem'])
        prepared=pd.concat([pd.read_csv(source/'veri'/f'{s}.csv',encoding='utf-8-sig') for s in ['train','validation','test']],ignore_index=True)
        xkeys=pd.MultiIndex.from_frame(prepared[['sube_id','ozellik_donemi']])
        ykeys=pd.MultiIndex.from_frame(prepared[['sube_id','hedef_donemi']])
        for target,col in [('tl_vadesiz','tl_vadesiz_ort_bakiye_tl'),('tl_vadeli','tl_vadeli_ort_bakiye_tl'),('kredi','kredi_ort_bakiye_tl')]:
            np.testing.assert_allclose(prepared['y_'+target+'_tl'],raw.loc[ykeys,col],rtol=0,atol=1e-6)
            np.testing.assert_allclose(prepared[target+'_bakiye_t'],raw.loc[xkeys,col],rtol=0,atol=1e-6)
            np.testing.assert_allclose(prepared[target+'_ay_sonu_bakiye_tl'],raw.loc[xkeys,target+'_ay_sonu_bakiye_tl'],rtol=0,atol=1e-6)
        for col in ['musteri_sayisi','ortalama_yas','ortalama_aylik_gelir_tl','evli_orani','devir_giris_musteri_sayisi','devir_cikis_musteri_sayisi']:
            np.testing.assert_allclose(prepared[col],raw.loc[xkeys,col],rtol=0,atol=1e-6)
        assert not any(c.startswith('y_') for c in future.columns)
        manifest['checks']['v4_raw_labels_current_owner_features_and_closing_stocks_verified']=True
    for filename,keys in [('toplam_metrikler.csv',['gosterge','model','split']),
                           ('model_metrikleri.csv',['gosterge','model','split','sube_grubu']),
                           ('aylik_donem_sonuclari.csv',['gosterge','model','split','sube_grubu','hedef_donemi'])]:
        table=pd.read_csv(out/'sonuclar'/filename,encoding='utf-8-sig')
        for score in table.to_dict('records'):
            mask=np.ones(len(pred),dtype=bool)
            for key in keys:
                if key=='sube_grubu' and score[key]=='ALL':continue
                mask &= pred[key].eq(score[key]).to_numpy()
            p=pred[mask];y=p.gercek_tl.to_numpy();h=p.tahmin_tl.to_numpy()
            assert len(p)==score['n']
            np.testing.assert_allclose([mean_absolute_error(y,h),np.sqrt(mean_squared_error(y,h)),
                np.abs(y-h).sum()/np.abs(y).sum(),r2_score(y,h),(h.sum()-y.sum())/y.sum()],
                [score[k] for k in ['mae_tl','rmse_tl','wape','r2','toplam_sapma_orani']],atol=1e-6,rtol=1e-12)
    manifest['checks'].update({'independent_saved_models_reload_and_inference':True,
      'all_metric_levels_recomputed_from_exported_predictions':True,'source_split_hashes_unchanged':True,
      'all_predictors_share_same_actuals':True})
    if count==72:manifest['checks']['independent_all_72_model_reload_and_inference']=True
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(f'PASS: {count} model reloads, {predictors} predictors with identical keyed actuals, unchanged split hashes, all group/total/monthly metrics.')

if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--source',type=Path,default=ROOT/'outputs/lightgbm_ilk_model')
    cli.add_argument('--output',type=Path,default=ROOT/'outputs/model_karsilastirma')
    cli.add_argument('--raw-source',type=Path)
    a=cli.parse_args();run(a.source.resolve(),a.output.resolve(),a.raw_source.resolve() if a.raw_source else None)
