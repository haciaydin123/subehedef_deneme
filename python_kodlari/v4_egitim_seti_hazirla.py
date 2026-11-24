"""Build date-aligned V4 training rows from dynamic branch snapshots, not master joins."""
import argparse
import calendar
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TARGETS={'tl_vadesiz':'tl_vadesiz_ort_bakiye_tl','tl_vadeli':'tl_vadeli_ort_bakiye_tl','kredi':'kredi_ort_bakiye_tl'}

def read(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def save(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def next_month(period):
    y,m=map(int,period.split('-'));return f'{y+(m==12):04d}-{m%12+1:02d}'

def run(source,out):
    m=json.loads((source/'manifest.json').read_text());assert m['version']==4 and all(m['checks'].values())
    hashes={}
    for name in ['sube_model_girdileri_aylik.csv','sube_stok_mutabakat.csv','sube_aylik.csv']:
        digest=hashlib.sha256((source/name).read_bytes()).hexdigest()
        assert digest==m['csv_files'][name]['sha256'];hashes[name]=digest
    snapshots=read(source/'sube_model_girdileri_aylik.csv')
    reconciliation={(r['donem'],r['sube_id'],r['gosterge']):r for r in read(source/'sube_stok_mutabakat.csv')}
    by_branch={}
    for row in snapshots:by_branch.setdefault(row['sube_id'],[]).append(row)
    data=[];future=[]
    for bid,series in sorted(by_branch.items()):
        series.sort(key=lambda r:r['donem']);assert len(series)==26
        for i,raw in enumerate(series):
            if i<2:continue
            period=raw['donem'];target_period=next_month(period)
            row={**raw,'ozellik_donemi':period,'hedef_donemi':target_period}
            row['musteri_sayisi_degisim']=int(raw['musteri_sayisi'])-int(series[i-1]['musteri_sayisi'])
            year,month=map(int,period.split('-'));days=calendar.monthrange(year,month)[1]
            row['musteri_gun_ay_sonu_orani']=int(raw['musteri_gun'])/days/max(1,int(raw['musteri_sayisi']))
            target_month=int(target_period[-2:]);row['takvim_ay_sin']=math.sin(2*math.pi*target_month/12)
            row['takvim_ay_cos']=math.cos(2*math.pi*target_month/12);row['zaman_indeksi']=i
            for target,col in TARGETS.items():
                current=float(raw[col]);previous=float(series[i-1][col])
                row[target+'_bakiye_t']=current;row[target+'_bakiye_t_eksi_1']=previous
                row[target+'_son3ay_ort_tl']=sum(float(r[col]) for r in series[i-2:i+1])/3
                row[target+'_son_ay_buyume']=current/previous-1 if previous else 0.
                closing=float(raw[target+'_ay_sonu_bakiye_tl'])
                row[target+'_kapanis_ortalama_farki_tl']=closing-current
                rec=reconciliation[(period,bid,target)]
                row[target+'_devir_net_degisim_tl']=int(rec['devir_net_degisim_kurus'])/100
                row[target+'_devir_haric_net_degisim_tl']=int(rec['devir_haric_net_degisim_kurus'])/100
                if i<25:row['y_'+target+'_tl']=float(series[i+1][col])
            row['split']='FORECAST' if i==25 else 'TRAIN' if target_period<='2026-06' else 'VALIDATION' if target_period<='2026-09' else 'TEST'
            (future if i==25 else data).append(row)
    data.sort(key=lambda r:(r['hedef_donemi'],r['sube_id']));future.sort(key=lambda r:r['sube_id'])
    counts={split:sum(r['split']==split for r in data) for split in ['TRAIN','VALIDATION','TEST']}
    assert counts=={'TRAIN':3400,'VALIDATION':600,'TEST':600} and len(future)==200
    folder=out/'veri';folder.mkdir(parents=True,exist_ok=True)
    for split in counts:save(folder/(split.lower()+'.csv'),[r for r in data if r['split']==split])
    save(folder/'tum_etiketli_veri.csv',data);save(folder/'ocak2027_tahmin_girdisi.csv',future)
    splits=[]
    for split in counts:
        part=[r for r in data if r['split']==split]
        splits.append({'split':split,'satir':len(part),'sube':len({r['sube_id'] for r in part}),
          'ilk_ozellik_ayi':min(r['ozellik_donemi'] for r in part),'son_ozellik_ayi':max(r['ozellik_donemi'] for r in part),
          'ilk_hedef_ayi':min(r['hedef_donemi'] for r in part),'son_hedef_ayi':max(r['hedef_donemi'] for r in part)})
    save(folder/'zaman_ayrimi.csv',splits)
    preparation={'source_version':4,'source_hashes':hashes,'splits':splits,'labeled_rows':len(data),'forecast_rows':len(future),
      'dynamic_owner_demographics':True,'calendar_weighted_branch_balances':True,
      'past_only_lags':True,'future_events_not_joined':True,'source_data_manifest':m}
    (out/'hazirlik_manifest.json').write_text(json.dumps(preparation,ensure_ascii=False,indent=2))
    print(json.dumps({'status':'prepared','splits':counts,'forecast_rows':len(future)},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=ROOT/'outputs/banka_test_verisi_v4')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/model_karsilastirma_v4')
    a=p.parse_args();run(a.source.resolve(),a.output.resolve())
