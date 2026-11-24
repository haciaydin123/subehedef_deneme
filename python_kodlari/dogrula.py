"""Independent readback checks for saved data, native models and reported metrics."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd

def run(source,out):
    spec=importlib.util.spec_from_file_location('saved_inference',out/'kod/tahmin.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    frames={name:pd.read_csv(out/'veri'/f'{name}.csv',encoding='utf-8-sig') for name in ['train','validation','test']}
    assert [len(frames[x]) for x in frames]==[3400,600,600]
    all_rows=pd.concat(frames.values(),ignore_index=True)
    assert len(all_rows)==4600 and not all_rows.duplicated(['sube_id','hedef_donemi']).any()
    assert all_rows['sube_id'].nunique()==200
    assert frames['train']['hedef_donemi'].max()<frames['validation']['hedef_donemi'].min()
    assert frames['validation']['hedef_donemi'].max()<frames['test']['hedef_donemi'].min()
    periods=pd.PeriodIndex(all_rows['ozellik_donemi'],freq='M')
    assert ((periods+1).astype(str)==all_rows['hedef_donemi']).all()
    actual=pd.read_csv(source/'sube_aylik.csv',encoding='utf-8-sig').set_index(['sube_id','donem'])
    for name,col in [('tl_vadesiz','tl_vadesiz_ort_bakiye_tl'),('tl_vadeli','tl_vadeli_ort_bakiye_tl'),('kredi','kredi_ort_bakiye_tl')]:
        ykeys=pd.MultiIndex.from_frame(all_rows[['sube_id','hedef_donemi']])
        xkeys=pd.MultiIndex.from_frame(all_rows[['sube_id','ozellik_donemi']])
        assert np.allclose(actual.loc[ykeys,col].to_numpy(),all_rows['y_'+name+'_tl'],rtol=0,atol=1e-6)
        assert np.allclose(actual.loc[xkeys,col].to_numpy(),all_rows[name+'_bakiye_t'],rtol=0,atol=1e-6)
    forecast_input=pd.read_csv(out/'veri/ocak2027_tahmin_girdisi.csv',encoding='utf-8-sig')
    assert len(forecast_input)==200 and forecast_input['hedef_donemi'].eq('2027-01').all()
    assert not any(c.startswith('y_') for c in forecast_input)
    predictions=pd.read_csv(out/'sonuclar/test_ve_validasyon_tahminleri.csv',encoding='utf-8-sig')
    forecasts=pd.read_csv(out/'sonuclar/ocak2027_model_tahminleri.csv',encoding='utf-8-sig')
    total=pd.read_csv(out/'sonuclar/toplam_metrikler.csv',encoding='utf-8-sig')
    checked=0
    for path in sorted((out/'modeller').glob('*__metadata.json')):
        meta=json.loads(path.read_text())
        assert len(meta['features'])==(25 if meta['model']=='yapisal' else 34)
        assert not any(c.startswith('y_') or c in ('sube_id','ilce_id','hedef_donemi') for c in meta['features'])
        if meta['model']=='yapisal':
            assert not any('bakiye' in c or 'varlik' in c or 'portfoy' in c or 'segment' in c for c in meta['features'])
        for frame,expected,artifact in [(frames['test'],predictions[predictions['split'].eq('TEST')],'evaluation'),
                                        (forecast_input,forecasts,'production')]:
            result=module.predict_saved(path,frame,artifact=artifact)
            e=expected[expected['gosterge'].eq(meta['gosterge'])&expected['model'].eq(meta['model'])&expected['sube_grubu'].eq(meta['sube_grubu'])]
            merged=result.merge(e,on=['sube_id','hedef_donemi'],validate='one_to_one')
            predicted='tahmin_tl' if artifact=='evaluation' else 'tahmin_aylik_ortalama_tl_y'
            inferred='tahmin_aylik_ortalama_tl' if artifact=='evaluation' else 'tahmin_aylik_ortalama_tl_x'
            assert len(merged)==len(e)
            assert np.allclose(merged[inferred],merged[predicted],rtol=0,atol=1e-6)
        checked+=1
    assert checked==12
    for score in total.to_dict('records'):
        p=predictions[predictions['gosterge'].eq(score['gosterge'])&predictions['model'].eq(score['model'])&predictions['split'].eq(score['split'])]
        error=np.abs(p['gercek_tl']-p['tahmin_tl'])
        assert abs(error.mean()-score['mae_tl'])<1e-6
        assert abs(error.sum()/p['gercek_tl'].sum()-score['wape'])<1e-12
    feature=pd.read_csv(out/'veri/ozellik_sozlugu.csv',encoding='utf-8-sig')
    stable=pd.read_csv(out/'sonuclar/kararli_degisken_adaylari.csv',encoding='utf-8-sig')
    calendars=set(feature[feature['ozellik_grubu'].eq('takvim')]['alan'])
    assert not stable[stable['kararli_aday']]['alan_veya_grup'].isin(calendars).any()
    manifest=json.loads((out/'manifest.json').read_text())
    manifest['checks'].update({'independent_csv_label_and_lag_alignment':True,
      'all_24_saved_models_reproduce_exported_predictions':True,'aggregate_metrics_recomputed_from_rows':True})
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PASS: 4600 rows, chronology, all labels/current lags, 24 model reloads, forecasts, and aggregate metrics.')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();run(a.source.resolve(),a.output.resolve())
