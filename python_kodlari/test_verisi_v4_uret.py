"""Event-driven synthetic retail bank panel. Standard library only.

Regenerates all 26 months using one mechanism; preserves V3 unchanged.
Amounts in the event ledger are integer kuruş. Events apply before that day's
closing balance; monthly averages use every calendar day, including inactive days.
"""
from __future__ import annotations
import argparse
import calendar
import collections
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import shutil
import time

ROOT=Path(__file__).resolve().parents[1]
SEED=20261121
PRODUCTS=['tl_vadesiz','doviz_vadesiz','altin_hesabi','tl_vadeli','doviz_vadeli',
          'ihtiyac_kredi','konut_kredi','tasit_kredi','kmh','fon','hisse','tahvil_bono',
          'kart_borc','kart_limit','bes_birikim']
COLS=[p+'_ort_bakiye_tl' for p in PRODUCTS[:12]]+['kart_borc_ort_tl','kart_limit_tl','bes_donem_sonu_birikim_tl']
ASSETS=[0,1,2,3,4,9,10,11]
LOANS=[5,6,7,8]
TARGETS={'tl_vadesiz':[0],'tl_vadeli':[3],'kredi':LOANS}
FAMILIES={'vadesiz':[0,1,2],'vadeli':[3,4],'kredi':LOANS,'yatirim':[9,10,11]}
SEGMENTS=['S1_BASLANGIC','S2_BIRIKIM','S3_GELISEN','S4_VARLIKLI','S5_OZEL']
RATES={'sube_devir_olasiligi':.012,'yeni_musteri_orani':.008,
       'tam_ayrilma_taban_olasiligi':.003,'diger_bankaya_kismi_cikis_olasiligi':.022,
       'diger_bankadan_giris_olasiligi':.014,'yeni_urun_olasiligi':.015,
       'urun_kapama_olasiligi':.006,'vade_donusu_olasiligi':.025}

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def cents(value):return int(round(float(value)*100))
def rounded(n,d):return (2*n+d)//(2*d)
def money(n):return round(n/100,2)
def clamp(x,a,b):return min(b,max(a,x))
def total(v,idx):return sum(v[i] for i in idx)

def months():
    return [f'{2024+(10+i)//12:04d}-{(10+i)%12+1:02d}' for i in range(26)]

def save(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

class Sink:
    def __init__(self,path):self.path=path;self.f=path.open('w',encoding='utf-8-sig',newline='');self.writer=None;self.count=0
    def put(self,row):
        if self.writer is None:self.writer=csv.DictWriter(self.f,fieldnames=list(row));self.writer.writeheader()
        self.writer.writerow(row);self.count+=1
    def close(self):self.f.close()

def run(source,out):
    started=time.perf_counter();rng=random.Random(SEED);out.mkdir(parents=True,exist_ok=True)
    profiles=read(source/'musteriler.csv');branches=read(source/'subeler.csv');regions=read(source/'bolgeler_sentetik.csv')
    region_by_id={r['ilce_id']:r for r in regions};branch_by_id={r['sube_id']:r for r in branches}
    branch_ids=list(branch_by_id);initial={}
    with (source/'musteri_aylik.csv').open(encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            if row['donem']!='2024-11':break
            initial[row['musteri_id']]=row
    assert len(initial)==10127
    quality={b:rng.uniform(.86,1.14) for b in branch_ids}
    clients={};next_id=len(initial)+1
    def make_client(profile,base,cid,branch,joined,initial_client):
        p={**profile};p.pop('sube_id',None)
        p.update(musteri_id=cid,kaynak_musteri_id=profile['kaynak_musteri_id'] if initial_client else '',
          ilk_sube_id=branch,katilim_tarihi=joined,
          profil_kokeni='KAGGLE_PROFILI' if initial_client else 'SENTETIK_YENI_MUSTERI',
          dogum_yili=2024-int(profile['yas']))
        scale=1. if initial_client else rng.uniform(.45,1.2)
        vector=[max(0,round(cents(base[col])*scale)) for col in COLS]
        vector[13]=max(vector[13],vector[12])
        income=max(1000.,float(base['aylik_gelir_tl'])*scale)
        return {'profile':p,'id':cid,'branch':branch,'active':initial_client,
          'v':vector[:] if initial_client else [0]*15,'initial':vector,'base_income':income,'income':income,
          'base_assets':max(10000,total(vector,ASSETS)),'base_risk':cents(base['diger_banka_kredi_ort_risk_tl']),
          'join':joined,'relationship':int(base['iliski_suresi_ay']) if initial_client else 0,
          'score':rng.uniform(45,90),'mobile':int(base['mobil_aktif']),
          'salary':int(base['maas_musterisi']),'pension':int(base['emekli_maasi_musterisi']),
          'mainbank':int(base['ana_banka_musterisi']),'base':base,'initial_client':initial_client}
    for p in profiles:
        c=make_client(p,initial[p['musteri_id']],p['musteri_id'],p['sube_id'],'2024-11-01',True);clients[c['id']]=c
    opening=[]
    for c in clients.values():
        opening.append({'musteri_id':c['id'],'sube_id':c['branch'],'aktif':1,
          **{p+'_bakiye_kurus':c['v'][i] for i,p in enumerate(PRODUCTS)}})
    save(out/'acilis_bakiyeleri.csv',opening)
    sinks={name:Sink(out/(name+'.csv')) for name in ['musteri_aylik','musteri_sube_aylik','musteri_olaylari']}
    all_branch=[];all_hp=[];bank_rows=[];threshold_rows=[];reconciliations=[];movement_rows=[];feature_rows=[]
    examples=[];example_counts=collections.Counter();all_types=collections.Counter();latest=[];event_id=0
    prior_close={b:{t:sum(total(c['v'],idx) for c in clients.values() if c['branch']==b) for t,idx in TARGETS.items()} for b in branch_ids}
    historical_counts={}
    for t,period in enumerate(months()):
        year,month=map(int,period.split('-'));days=calendar.monthrange(year,month)[1]
        start_count=sum(c['active'] for c in clients.values())
        branch_shock={b:rng.gauss(0,.018)+.012*math.sin(2*math.pi*month/12)+.02*(quality[b]-1) for b in branch_ids}
        acquisitions=max(1,round(start_count*RATES['yeni_musteri_orani']*rng.uniform(.8,1.2)))
        for _ in range(acquisitions):
            template=profiles[rng.randrange(len(profiles))];base=initial[template['musteri_id']]
            bid=rng.choices(branch_ids,weights=[int(branch_by_id[b]['personel_fte_sentetik'])*quality[b] for b in branch_ids])[0]
            cid=f'M{next_id:05d}';next_id+=1;joined=f'{period}-{rng.randint(2,days-1):02d}'
            clients[cid]=make_client(template,base,cid,bid,joined,False)
        monthly=[];bridges=[];counter=collections.Counter()
        moves={b:collections.Counter() for b in branch_ids}
        balance_change={b:collections.Counter() for b in branch_ids}
        transfer_change={b:collections.Counter() for b in branch_ids}
        for c in clients.values():
            cid=c['id'];is_new=c['join'].startswith(period) and not c['initial_client']
            eligible=c['active'] or is_new
            c['income']=round(c['base_income']*math.exp(.0035*t+rng.gauss(0,.025)),2) if eligible else c['income']
            if eligible:
                c['score']=clamp(c['score']+.6*(quality[c['branch']]-1)+rng.gauss(0,2.5),10,98)
                if rng.random()<.012:c['salary']=1-c['salary']
                if rng.random()<.008:c['mobile']=1-c['mobile']
                c['mainbank']=int(c['score']>=58)
            salary_during_month=c['salary'];pension_during_month=c['pension']
            new_day=int(c['join'][-2:]) if is_new else 1
            schedule=[]
            def plan(kind,low=2,high=None,payload=None):
                if low<days:schedule.append((rng.randint(max(low,new_day),max(max(low,new_day),high or days)),len(schedule),kind,payload))
            if is_new:schedule.append((new_day,-1,'YENI_MUSTERI',None))
            if eligible:
                plan('RUTIN_BAKIYE_HAREKETI',max(2,new_day),min(days,max(12,new_day)))
                plan('KREDI_TAKSIT_ODEMESI',max(2,new_day))
                plan('KART_ODEME_VE_KULLANIM',max(2,new_day))
                plan('BES_KATKI_VE_GETIRI',max(2,new_day))
                if rng.random()<RATES['sube_devir_olasiligi']:
                    same=[b for b in branch_ids if b!=c['branch'] and branch_by_id[b]['ilce_id']==branch_by_id[c['branch']]['ilce_id']]
                    dest=rng.choice(same if same and rng.random()<.65 else [b for b in branch_ids if b!=c['branch']])
                    plan('SUBE_DEVRI',max(2,new_day),payload=dest)
                risk_multiplier=clamp((85-c['score'])/30,.35,2.)
                if rng.random()<RATES['diger_bankaya_kismi_cikis_olasiligi']*risk_multiplier:plan('DIGER_BANKAYA_KAYNAK_CIKISI',max(2,new_day))
                if rng.random()<RATES['diger_bankadan_giris_olasiligi']*(.6+c['score']/100):plan('DIGER_BANKADAN_KAYNAK_GIRISI',max(2,new_day))
                if rng.random()<RATES['yeni_urun_olasiligi']:plan('YENI_URUN_ACILISI',max(2,new_day))
                if rng.random()<RATES['urun_kapama_olasiligi']:plan('URUN_KAPAMA_VE_IC_AKTARIM',max(2,new_day))
                if rng.random()<RATES['vade_donusu_olasiligi']:plan('VADE_DONUSU',max(2,new_day))
                tenure=c['relationship']+t-months().index(c['join'][:7])
                if not is_new and tenure>=3 and rng.random()<RATES['tam_ayrilma_taban_olasiligi']*risk_multiplier:
                    exit_day=rng.randint(3,days)
                    schedule.extend([(exit_day,100,'AYRILIS_BORC_KAPAMA',None),
                                     (exit_day,101,'BANKADAN_TAM_AYRILMA',None)])
            integral=[0]*15;by_branch={};active_days=0;last_day=1;types=collections.Counter()
            def integrate(span):
                nonlocal active_days
                if not span:return
                for i in range(15):integral[i]+=c['v'][i]*span
                if c['active']:
                    part=by_branch.setdefault(c['branch'],{'days':0,'integral':[0]*13})
                    part['days']+=span;active_days+=span
                    for i in range(13):part['integral'][i]+=c['v'][i]*span
            for day,order,kind,payload in sorted(schedule):
                integrate(day-last_day);last_day=day
                if not c['active'] and kind!='YENI_MUSTERI':continue
                old=c['v'][:];from_branch=c['branch'];old_active=c['active']
                if kind=='YENI_MUSTERI':c['active']=True;c['v']=c['initial'][:]
                elif kind=='RUTIN_BAKIYE_HAREKETI':
                    target_asset=c['base_assets']*math.exp(.0035*t)*(0.85+c['score']/250)*quality[c['branch']]
                    current=max(1,total(c['v'],ASSETS));pull=clamp(math.log(max(1,target_asset)/current),-.5,.5)*.04
                    for i in ASSETS:
                        if c['v'][i]>0:c['v'][i]=max(0,round(c['v'][i]*math.exp(.001+pull+branch_shock[c['branch']]+rng.gauss(0,.045))))
                elif kind=='KREDI_TAKSIT_ODEMESI':
                    for i,rate in zip(LOANS,[.018,.004,.022,.08]):c['v'][i]-=min(c['v'][i],max(0,round(c['v'][i]*rate*rng.uniform(.85,1.15))))
                elif kind=='KART_ODEME_VE_KULLANIM':
                    limit=max(cents(c['income']*1.7),c['v'][12]);c['v'][13]=limit
                    c['v'][12]=min(limit,max(0,round(c['v'][12]*rng.uniform(.70,.95)+cents(c['income']*rng.uniform(.03,.18)))))
                elif kind=='BES_KATKI_VE_GETIRI':
                    if c['v'][14]>0:c['v'][14]=round(c['v'][14]*1.002)+cents(float(c['base']['bes_aylik_katki_tl'])*active_days/max(1,day-1))
                elif kind=='SUBE_DEVRI':c['branch']=payload
                elif kind=='DIGER_BANKAYA_KAYNAK_CIKISI':
                    fraction=rng.uniform(.15,.65)
                    for i in ASSETS:c['v'][i]-=round(c['v'][i]*fraction)
                    c['score']=max(10,c['score']-rng.uniform(3,9))
                elif kind=='DIGER_BANKADAN_KAYNAK_GIRISI':
                    amount=cents(c['income']*rng.uniform(.5,4));c['v'][0]+=amount*2//5;c['v'][3]+=amount-amount*2//5
                    c['score']=min(98,c['score']+rng.uniform(1,4))
                elif kind=='YENI_URUN_ACILISI':
                    options=[i for i in range(12) if c['v'][i]==0 and i not in [0,6]]
                    if not options:continue
                    i=rng.choice(options);multiple=rng.uniform(3,12) if i in LOANS else rng.uniform(.3,3)
                    c['v'][i]=cents(c['income']*multiple)
                elif kind in ['URUN_KAPAMA_VE_IC_AKTARIM','VADE_DONUSU']:
                    options=[i for i in ([3,4] if kind=='VADE_DONUSU' else [1,2,3,4,9,10,11]) if c['v'][i]>0]
                    if not options:continue
                    i=rng.choice(options);c['v'][0]+=c['v'][i];c['v'][i]=0
                elif kind=='AYRILIS_BORC_KAPAMA':
                    for i in LOANS+[12]:c['v'][i]=0
                elif kind=='BANKADAN_TAM_AYRILMA':
                    assert total(c['v'],LOANS+[12])==0
                    c['v']=[0]*15;c['active']=False;c['salary']=0;c['mobile']=0;c['mainbank']=0
                delta=[c['v'][i]-old[i] for i in range(15)]
                if not any(delta) and kind not in ['SUBE_DEVRI','BANKADAN_TAM_AYRILMA','YENI_MUSTERI']:continue
                assert min(c['v'])>=0 and c['v'][12]<=c['v'][13]
                event_id+=1;date=f'{period}-{day:02d}'
                row={'olay_id':f'O{event_id:08d}','donem':period,'musteri_id':cid,'olay_tarihi':date,
                  'bilinirlik_tarihi':date,'olay_sirasi':order,'olay_tipi':kind,'onceki_sube_id':from_branch,
                  'sonraki_sube_id':c['branch'],'sonraki_aktif':int(c['active']),
                  **{p+'_delta_kurus':delta[i] for i,p in enumerate(PRODUCTS)},
                  **{'devir_'+name+'_kurus':total(old,idx) if kind=='SUBE_DEVRI' else 0 for name,idx in TARGETS.items()}}
                sinks['musteri_olaylari'].put(row);counter[kind]+=1;types[kind]+=1;all_types[kind]+=1
                if kind not in ['RUTIN_BAKIYE_HAREKETI','KREDI_TAKSIT_ODEMESI','KART_ODEME_VE_KULLANIM','BES_KATKI_VE_GETIRI'] and example_counts[kind]<3:
                    examples.append(row);example_counts[kind]+=1
                for target,idx in TARGETS.items():
                    balance_change[from_branch][target]+=total(delta,idx)
                    if kind=='SUBE_DEVRI':
                        transfer_change[from_branch][target]-=total(old,idx);transfer_change[c['branch']][target]+=total(old,idx)
                if kind=='SUBE_DEVRI':moves[from_branch]['devir_cikis']+=1;moves[c['branch']]['devir_giris']+=1
                if kind=='YENI_MUSTERI':moves[c['branch']]['yeni_musteri']+=1
                if kind=='BANKADAN_TAM_AYRILMA':moves[from_branch]['ayrilan_musteri']+=1
                if kind=='DIGER_BANKAYA_KAYNAK_CIKISI':moves[from_branch]['diger_banka_cikis_kurus']-=total(delta,ASSETS)
                if kind=='DIGER_BANKADAN_KAYNAK_GIRISI':moves[from_branch]['diger_banka_giris_kurus']+=total(delta,ASSETS)
            integrate(days+1-last_day)
            averages=[rounded(value,days) for value in integral]
            # Allocate rounded customer means to owners with at most 1 kuruş rounding error.
            allocations={bid:[x//days for x in part['integral']] for bid,part in by_branch.items()}
            for i in range(13):
                remainder=averages[i]-sum(v[i] for v in allocations.values())
                ranked=sorted(by_branch,key=lambda bid:by_branch[bid]['integral'][i]%days,reverse=True)
                for bid in ranked[:remainder]:allocations[bid][i]+=1
            base=c['base'];exposure=active_days/days
            values={COLS[i]:money(averages[i]) for i in range(13)}
            for family,idx in FAMILIES.items():values[family+'_ort_bakiye_tl']=money(total(averages,idx))
            owned={p+'_sahibi':int(c['active'] and c['v'][i]>0) for i,p in enumerate(PRODUCTS[:12])}
            owned['tl_vadesiz_sahibi']=int(c['active'])
            flows={}
            for name in ['kart_harcama_tl','kart_islem_adedi','banka_karti_harcama_tl','banka_karti_islem_adedi',
              'bes_aylik_katki_tl','hayat_aylik_prim_tl','saglik_aylik_prim_tl','konut_sigorta_aylik_prim_tl','kasko_aylik_prim_tl',
              'otomatik_fatura_adet','otomatik_fatura_tutar_tl','duzenli_odeme_adet','duzenli_odeme_tutar_tl',
              'eft_adet','eft_tutar_tl','havale_adet','havale_tutar_tl','fast_adet','fast_tutar_tl']:
                value=float(base[name])*exposure*(c['income']/max(1,float(base['aylik_gelir_tl'])))*rng.uniform(.85,1.15)
                flows[name]=round(value,2) if name.endswith('_tl') else max(0,round(value))
            if averages[14]==0:flows['bes_aylik_katki_tl']=0.
            flags={name:int(c['active'] and int(base[name])) for name in ['banka_karti_sahibi','hayat_sahibi','saglik_sahibi','konut_sigorta_sahibi','kasko_sahibi','otomatik_fatura_talimati_var','duzenli_odeme_talimati_var']}
            flags.update(kredi_karti_sahibi=int(c['active']),bes_sahibi=int(c['active'] and c['v'][14]>0))
            risk=money(round(c['base_risk']*math.exp(.002*t))) if c['active'] else None
            row={'donem':period,'musteri_id':cid,'sube_id':c['branch'],'is_birimi':'BIREYSEL',
              'ay_sonu_aktif':int(c['active']),'aktif_gun_sayisi':active_days,'ay_gun_sayisi':days,
              'katilim_tarihi':c['join'],'yas':year-int(c['profile']['dogum_yili']),
              'aylik_gelir_tl':c['income'] if c['active'] else 0.,'iliski_skoru':round(c['score'],2) if c['active'] else None,
              **values,'kart_limit_tl':money(c['v'][13]),'bes_donem_sonu_birikim_tl':money(c['v'][14]),
              'diger_banka_kredi_ort_risk_tl':risk,
              'memzuc_ort_nakdi_risk_tl':round(values['kredi_ort_bakiye_tl']+values['kart_borc_ort_tl']+risk,2) if risk is not None else None,
              'memzuc_bilgi_tarihi':f'{period}-{days:02d}' if c['active'] else '',
              **owned,**flags,**flows,'mobil_aktif':c['mobile'] if c['active'] else 0,
              'maas_musterisi':c['salary'] if c['active'] else 0,'emekli_maasi_musterisi':c['pension'] if c['active'] else 0,
              'ana_banka_musterisi':c['mainbank'] if c['active'] else 0,
              'kredi_karti_aktif':int(flows['kart_islem_adedi']>0),'banka_karti_aktif':int(flows['banka_karti_islem_adedi']>0),
              'maas_yatis_tutari_tl':round(c['income']*exposure*salary_during_month,2),
              'emekli_maasi_yatis_tutari_tl':round(c['income']*exposure*pension_during_month,2),
              'iliski_suresi_ay':c['relationship']+max(0,t-months().index(c['join'][:7])),
              'devir_sayisi':types['SUBE_DEVRI'],'yeni_musteri':int(is_new),'ayrilan_musteri':types['BANKADAN_TAM_AYRILMA'],
              'diger_bankaya_cikis_sayisi':types['DIGER_BANKAYA_KAYNAK_CIKISI'],
              'diger_bankadan_giris_sayisi':types['DIGER_BANKADAN_KAYNAK_GIRISI'],
              'toplam_varlik_ort_tl':money(total(averages,ASSETS)),
              'toplam_varlik_ay_sonu_tl':money(total(c['v'],ASSETS)),
              **{p+'_ay_sonu_bakiye_tl':money(c['v'][i]) for i,p in enumerate(PRODUCTS[:12])},
              'kart_borc_ay_sonu_tl':money(c['v'][12])}
            row['aktif_urun_sayisi']=sum(owned.values())+sum(flags.values())
            monthly.append(row)
            for bid,part in by_branch.items():
                v=allocations[bid]
                br={'donem':period,'musteri_id':cid,'sube_id':bid,'aktif_gun_sayisi':part['days'],
                  'ay_gun_sayisi':days,'ay_sonu_bu_subede_aktif':int(c['active'] and c['branch']==bid),
                  **{COLS[i]:money(v[i]) for i in range(13)}}
                for family,idx in FAMILIES.items():br[family+'_ort_bakiye_tl']=money(total(v,idx))
                br['toplam_varlik_ort_tl']=money(total(v,ASSETS));bridges.append(br)
        eligible_assets=sorted(cents(r['toplam_varlik_ort_tl']) for r in monthly if r['ay_sonu_aktif'])
        n=len(eligible_assets);port_n=round(n/16)
        cuts=[eligible_assets[n//2],eligible_assets[int(n*.8)],eligible_assets[-port_n],eligible_assets[int(n*.99)]]
        assert cuts==sorted(set(cuts))
        for i,name in enumerate(SEGMENTS):
            threshold_rows.append({'donem':period,'segment':name,'hp':'HAVUZ' if i<3 else 'PORTFOY',
             'alt_sinir_dahil_tl':money(cuts[i-1]) if i else 0.,'ust_sinir_haric_tl':money(cuts[i]) if i<4 else None,
             'kalibrasyon_evreni':'AY_SONU_AKTIF_MUSTERILER','hedef_havuz_portfoy_orani':15})
        by_id={r['musteri_id']:r for r in monthly};customer_totals=collections.Counter()
        eom_by_branch=collections.defaultdict(list);br_by_branch=collections.defaultdict(list)
        for r in monthly:
            index=sum(cents(r['toplam_varlik_ort_tl'])>=cut for cut in cuts)
            r['musteri_segmenti']=SEGMENTS[index];r['hp']='HAVUZ' if index<3 else 'PORTFOY'
            r['segment_durumu']='AKTIF' if r['ay_sonu_aktif'] else 'SON_GORUNUM_AYRILMIS'
            sinks['musteri_aylik'].put(r)
            if r['ay_sonu_aktif']:eom_by_branch[r['sube_id']].append(r)
            for name in ['tl_vadesiz_ort_bakiye_tl','tl_vadeli_ort_bakiye_tl','kredi_ort_bakiye_tl']:
                customer_totals[name]+=cents(r[name])
        for r in bridges:
            r['hp']=by_id[r['musteri_id']]['hp'];r['musteri_segmenti']=by_id[r['musteri_id']]['musteri_segmenti']
            sinks['musteri_sube_aylik'].put(r);br_by_branch[r['sube_id']].append(r)
        closing={b:{target:0 for target in TARGETS} for b in branch_ids}
        month_branch=[]
        for bid in branch_ids:
            members=eom_by_branch[bid];parts=br_by_branch[bid];branch=branch_by_id[bid];region=region_by_id[branch['ilce_id']]
            count=len(members)
            sums={name:sum(cents(r[name]) for r in parts) for name in list(COLS[:13])+[f+'_ort_bakiye_tl' for f in FAMILIES]+['toplam_varlik_ort_tl']}
            h=sum(r['hp']=='HAVUZ' for r in members);p=count-h
            row={'donem':period,'sube_id':bid,'is_birimi':'BIREYSEL',
              **{k:branch[k] for k in ['ilce_id','il','buyuk_sehir_grubu','sube_segmenti']},
              **{k:region[k] for k in ['ekonomik_yapi_sentetik','sosyoekonomik_skor_sentetik','rakip_sube_sayisi_sentetik','nufus_sentetik']},
              'personel_fte_sentetik':int(branch['personel_fte_sentetik']),
              'musteri_sayisi':count,'ay_ici_hizmet_verilen_musteri_sayisi':len(parts),
              'musteri_gun':sum(r['aktif_gun_sayisi'] for r in parts),
              'ortalama_yas':round(sum(r['yas'] for r in members)/count,6) if count else 0.,
              'ortalama_aylik_gelir_tl':round(sum(r['aylik_gelir_tl'] for r in members)/count,2) if count else 0.,
              'mobil_aktif_orani':round(sum(r['mobil_aktif'] for r in members)/count,6) if count else 0.,
              'maas_musterisi_orani':round(sum(r['maas_musterisi'] for r in members)/count,6) if count else 0.,
              'ortalama_iliski_skoru':round(sum(r['iliski_skoru'] for r in members)/count,6) if count else 0.,
              'havuz_musteri_sayisi':h,'portfoy_musteri_sayisi':p,'havuz_portfoy_orani':round(h/p,6) if p else None,
              **{name:money(v) for name,v in sums.items()},
              'yeni_musteri_sayisi':moves[bid]['yeni_musteri'],'ayrilan_musteri_sayisi':moves[bid]['ayrilan_musteri'],
              'devir_giris_musteri_sayisi':moves[bid]['devir_giris'],'devir_cikis_musteri_sayisi':moves[bid]['devir_cikis'],
              'diger_bankaya_kaynak_cikisi_tl':money(moves[bid]['diger_banka_cikis_kurus']),
              'diger_bankadan_kaynak_girisi_tl':money(moves[bid]['diger_banka_giris_kurus'])}
            for target,idx in TARGETS.items():
                closing[bid][target]=sum(total(clients[r['musteri_id']]['v'],idx) for r in members)
                row[target+'_ay_sonu_bakiye_tl']=money(closing[bid][target])
                reconciliations.append({'donem':period,'sube_id':bid,'gosterge':target,
                  'acilis_kurus':prior_close[bid][target],'devir_haric_net_degisim_kurus':balance_change[bid][target],
                  'devir_net_degisim_kurus':transfer_change[bid][target],'kapanis_kurus':closing[bid][target]})
                assert prior_close[bid][target]+balance_change[bid][target]+transfer_change[bid][target]==closing[bid][target]
            all_branch.append(row);month_branch.append(row)
            movement_rows.append({k:row[k] for k in ['donem','sube_id','yeni_musteri_sayisi','ayrilan_musteri_sayisi',
              'devir_giris_musteri_sayisi','devir_cikis_musteri_sayisi','diger_bankaya_kaynak_cikisi_tl','diger_bankadan_kaynak_girisi_tl']})
            for hp in ['HAVUZ','PORTFOY']:
                bp=[r for r in parts if r['hp']==hp]
                all_hp.append({'donem':period,'sube_id':bid,'hp':hp,
                  'musteri_sayisi':sum(r['hp']==hp for r in members),
                  **{name:money(sum(cents(r[name]) for r in bp)) for name in ['tl_vadesiz_ort_bakiye_tl','tl_vadeli_ort_bakiye_tl','kredi_ort_bakiye_tl','toplam_varlik_ort_tl']}})
            # These rows use the current owner of active customers, never the master initial owner.
            profile_for=lambda r:clients[r['musteri_id']]['profile']
            mean=lambda fn:sum(fn(r) for r in members)/count if count else 0.
            model={**row,'yas_std':math.sqrt(mean(lambda r:(r['yas']-row['ortalama_yas'])**2)),
              'yas_35_alti_orani':mean(lambda r:r['yas']<35),'yas_60_ustu_orani':mean(lambda r:r['yas']>=60),
              'bakim_yukumlulugu_ort':mean(lambda r:int(profile_for(r)['bakmakla_yukumlu_kisi_sayisi'])),
              'yuksek_egitim_orani':mean(lambda r:profile_for(r)['egitim_duzeyi'] in ['LISANS','YUKSEK_LISANS','DOKTORA']),
              'egitim_bilinmiyor_orani':mean(lambda r:profile_for(r)['egitim_duzeyi']=='BILINMIYOR'),
              'evli_orani':mean(lambda r:profile_for(r)['medeni_durum']=='EVLI'),
              'kaynak_ust_gelir_orani':mean(lambda r:profile_for(r)['kaynak_gelir_grubu'] in ['80_120K_USD','GE_120K_USD']),
              'kaynak_gelir_bilinmiyor_orani':mean(lambda r:profile_for(r)['kaynak_gelir_grubu']=='BILINMIYOR'),
              'kaynak_pasif_ay_ort':mean(lambda r:int(profile_for(r)['kaynak_pasif_ay_sayisi_12ay'])),
              'emekli_maasi_orani':mean(lambda r:r['emekli_maasi_musterisi']),
              'ana_banka_orani':mean(lambda r:r['ana_banka_musterisi']),
              'portfoy_musteri_orani':p/count if count else 0.,
              'ortalama_varlik_tl':mean(lambda r:r['toplam_varlik_ort_tl']),
              **{f'segment_s{i}_orani':mean(lambda r,i=i:r['musteri_segmenti']==SEGMENTS[i-1]) for i in [1,2,5]}}
            feature_rows.append(model)
        end_count=sum(c['active'] for c in clients.values())
        assert start_count+counter['YENI_MUSTERI']-counter['BANKADAN_TAM_AYRILMA']==end_count
        for name,value in customer_totals.items():assert sum(cents(r[name]) for r in month_branch)==value
        for target in TARGETS:assert sum(transfer_change[b][target] for b in branch_ids)==0
        port=sum(r['ay_sonu_aktif'] and r['hp']=='PORTFOY' for r in monthly)
        bank_rows.append({'donem':period,'ay_basi_aktif_musteri':start_count,'yeni_musteri':counter['YENI_MUSTERI'],
          'tam_ayrilan_musteri':counter['BANKADAN_TAM_AYRILMA'],'ay_sonu_aktif_musteri':end_count,
          'sube_devir_sayisi':counter['SUBE_DEVRI'],'diger_bankaya_kismi_cikis_sayisi':counter['DIGER_BANKAYA_KAYNAK_CIKISI'],
          'diger_bankadan_giris_sayisi':counter['DIGER_BANKADAN_KAYNAK_GIRISI'],
          'yeni_urun_acilis_sayisi':counter['YENI_URUN_ACILISI'],'urun_kapama_sayisi':counter['URUN_KAPAMA_VE_IC_AKTARIM'],
          'havuz_musteri_sayisi':end_count-port,'portfoy_musteri_sayisi':port,'havuz_portfoy_orani':round((end_count-port)/port,6),
          **{name:money(v) for name,v in customer_totals.items()},
          **{target+'_devir_banka_net_etki_kurus':sum(transfer_change[b][target] for b in branch_ids) for target in TARGETS}})
        prior_close=closing;latest=monthly;historical_counts[period]=dict(counter)
        print(f'{period}: active={end_count}, new={counter["YENI_MUSTERI"]}, transfers={counter["SUBE_DEVRI"]}, exits={counter["BANKADAN_TAM_AYRILMA"]}',flush=True)
    for sink in sinks.values():sink.close()
    save(out/'musteriler.csv',[c['profile'] for c in clients.values()])
    save(out/'musteri_son_donem.csv',latest)
    save(out/'sube_aylik.csv',all_branch);save(out/'sube_hp_aylik.csv',all_hp)
    save(out/'sube_model_girdileri_aylik.csv',feature_rows);save(out/'sube_hareket_ozeti.csv',movement_rows)
    save(out/'banka_aylik.csv',bank_rows);save(out/'segment_esikleri.csv',threshold_rows)
    save(out/'sube_stok_mutabakat.csv',reconciliations);save(out/'olay_ornekleri.csv',examples)
    for name in ['subeler.csv','bolgeler_sentetik.csv']:shutil.copy2(source/name,out/name)
    budgets=[]
    for name,field,rate in [('TL_VADESIZ_ORT','tl_vadesiz_ort_bakiye_tl',.15),('TL_VADELI_ORT','tl_vadeli_ort_bakiye_tl',.10),('BIREYSEL_KREDI_ORT','kredi_ort_bakiye_tl',.05)]:
        amount=sum(cents(r[field]) for r in latest);target=rounded(amount*(100+round(rate*100)),100)
        budgets.append({'hedef_donem':'2027-01','referans_donem':'2026-12','gosterge_kodu':name,'veri_alani':field,
          'referans_aylik_ortalama_tl':money(amount),'artis_orani':rate,'hedef_aylik_ortalama_tl':money(target),'artis_tutari_tl':money(target-amount)})
    save(out/'ocak2027_banka_hedefleri.csv',budgets)
    assumptions={'seed':SEED,'monthly_base_probabilities_synthetic_not_estimated_from_bank':RATES,
      'relationship_adjustment':'outflow/exit probabilities scaled by current relationship score',
      'event_timing':'event applies before end-of-day balance; all calendar days in denominator',
      'transfer_rule':'entire simulated retail relationship transfers at event date, balance delta exactly zero',
      'full_exit_rule':'relationship tenure >=3 months; cash loans/card debt fully repaid before closing; assets and BES leave, limits close',
      'segmentation':'monthly contemporaneous asset quantiles of EOM-active customers, pool/portfolio count ratio about 15',
      'initial_balance_assumption':'V3 first-month averages used solely as synthetic initial day-zero stocks',
      'no_budget_in_data_generation':True,'no_realistic_rate_calibration_claim':True}
    (out/'varsayimlar.json').write_text(json.dumps(assumptions,ensure_ascii=False,indent=2))
    dictionary=[];old_dict={(r['tablo'],r['alan']):r for r in read(source/'veri_sozlugu.csv')}
    explicit={'sube_id':'Müşteri aylık tabloda ay sonu/son sorumlu şube. Ay içi bakiyeyi bu kimlikle tek şubeye toplamayın; musteri_sube_aylik kullanın.',
      'ay_sonu_aktif':'Ay sonunda açık banka ilişkisi. Ayrılmış müşterilerin sonraki ayları sıfır banka bakiyesiyle korunur.',
      'aktif_gun_sayisi':'Ay içinde aktif olunan/hizmet verilen takvim günü sayısı; olayı izleyen gün değil, olay günü yeni durum geçerlidir.',
      'bilinirlik_tarihi':'Olayın bankaca bilindiği tarih; bu senaryoda gerçekleşme tarihi. Sonraki ay olayları geçmiş özelliklere katılmaz.',
      'ilk_sube_id':'İlk atama; güncel aidiyet veya geçmiş bütün aylara uygulanacak şube değildir.',
      'musteri_sayisi':'Ay sonu aktif müşteri sayısı. Bakiyeler ay içindeki gün bazlı aidiyete göre toplanır.',
      'memzuc_ort_nakdi_risk_tl':'Aktif müşteride simüle dönem bilgisi: banka kredi ortalaması + kart borç ortalaması + diğer banka risk ortalaması. İlişki kapalıysa bilgi yok/boş; gerçek risk sıfır kabul edilmez.',
      'profil_kokeni':'Başlangıç profili Kaggle; yeni müşteri profili sentetik ve örnek profilden türetilmiş. Yeni müşterinin Kaggle kimliği boş.',
      'toplam_varlik_ort_tl':'Mevduat ve yatırım aylık ortalamaları toplamı. Kredi, kart borcu, limit, memzuç ve dönem sonu BES hariç.'}
    for p in sorted(out.glob('*.csv')):
        with p.open(encoding='utf-8-sig') as f:fields=next(csv.reader(f))
        for field in fields:
            old=old_dict.get((p.name,field),{})
            unit='kurus' if field.endswith('_kurus') else 'TL' if field.endswith('_tl') else old.get('birim','alan')
            desc=explicit.get(field,old.get('aciklama','V4 olay simülasyonu çıktısı; tablo anahtarı ve hesaplama tanımı README içinde.'))
            if field.endswith('_sahibi') or field.endswith('_talimati_var'):desc='Ay sonundaki ürün/talimat sahipliği; ay içi ortalama pozitif olsa da kapanışta 0 olabilir.'
            if p.name=='musteri_olaylari.csv' and field.endswith('_delta_kurus'):desc='Ürün stokundaki imzalı değişim; tamsayı kuruş. Şube devrinde bütün stok değişimleri 0.'
            dictionary.append({'tablo':p.name,'alan':field,'birim':unit,'koken':'SENTETIK_V4','aciklama':desc})
    save(out/'veri_sozlugu.csv',dictionary)
    manifest={'version':4,'seed':SEED,'source_version':3,'synthetic_financial_data':True,
      'initial_customers':10127,'registered_customers':len(clients),'latest_active_customers':sum(c['active'] for c in clients.values()),
      'branches':200,'months':26,'start':'2024-11','end':'2026-12',
      'customer_month_rows':sinks['musteri_aylik'].count,'customer_branch_month_rows':sinks['musteri_sube_aylik'].count,
      'event_rows':sinks['musteri_olaylari'].count,'branch_month_rows':len(all_branch),'event_counts':dict(all_types),
      'one_generation_mechanism_for_all_months':True,'source_v3_unchanged':True,
      'source_profile_sha256':hashlib.sha256((source/'musteriler.csv').read_bytes()).hexdigest(),
      'source_input_hashes':{name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ['musteriler.csv','musteri_aylik.csv','subeler.csv','bolgeler_sentetik.csv','veri_sozlugu.csv']},
      'target_growth':{'TL_VADESIZ_ORT':.15,'TL_VADELI_ORT':.10,'BIREYSEL_KREDI_ORT':.05},
      'models_retrained':False,'checks':{'independent_verification_pending':True},
      'generation_seconds':time.perf_counter()-started}
    manifest['csv_files']={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(out.glob('*.csv'))}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    (out/'README.md').write_text('''# V4: aylık müşteri ve ürün hareketleri içeren sentetik banka verisi

26 ayın tamamı (2024-11–2026-12) tek olay mekanizmasıyla yeniden üretildi.
V3 korunur. İlk 10.127 demografik profil kaynak veriden gelir; yeni müşteriler,
finansallar, şubeler, hareket oranları ve coğrafi koşullar sentetiktir.
Olasılıklar gerçek bankadan tahmin edilmemiştir; varsayimlar.json içinde bulunur.

## Kullanılacak tablolar

- musteri_aylik.csv: bir müşteri ve ay için tek satır. sube_id ay sonu/son
  sorumlu şubedir. katilim_tarihi paneldeki başlangıçtır; ilk müşteri kitlesi
  simülasyon öncesinden gelen ilişki süresine sahip olabilir. Yeni müşteri
  katılmadan önce satır yok; ayrılan müşterinin
  sonraki ayları sıfır banka bakiyesi ve ay_sonu_aktif=0 ile korunur.
- musteri_sube_aylik.csv: aylık bakiyelerin ay içi gerçek şube aidiyetine göre
  parçaları. Şube bakiyesini BU TABLODAN toplayın. Devredilen müşteri ay içinde
  iki şubede görünebilir; banka toplamında yalnızca kendi aylık ortalaması vardır.
- musteri_olaylari.csv: bütün stok değişimleri ve devirler. Olay tarihi ve
  bilinirlik tarihi aynı; ürün değişimleri tamsayı kuruş. Olay gününde yeni stok
  geçerlidir. Şube devri bütün ürün stoklarını korur; yalnızca sahibi değişir.
- acilis_bakiyeleri.csv: ilk günün öncesindeki stoklar. V3 ilk ay ortalamaları
  yeni simülasyonun başlangıç stokları olarak varsayılmıştır; eski gerçekleşme değildir.
- musteriler.csv: başlangıç profili ve katılım tarihi. ilk_sube_id geçmiş bütün
  aylara uygulanmaz. Dinamik yaş, gelir, aidiyet ve HP aylık dosyadadır.
- sube_aylik.csv: takvim günleriyle ağırlıklandırılmış müşteri bakiyesi parçalarının
  toplamı. Müşteri sayısı ay sonu aktiftir; ay_ici_hizmet_verilen_musteri_sayisi
  ayrı tutulur. Bir şubenin müşteri sayısı ve müşteri-günü farklı kavramlardır.
- sube_model_girdileri_aylik.csv: güncel aktif müşteri kitlesinden hesaplanmış
  demografi/ilişki/segment özellikleri. Kimlik, hedef ve olay sonrası etiketler
  otomatik özellik seçilmez. t bilgisi t+1 etiketiyle ayrıca eşleştirilmelidir.
- sube_hp_aylik.csv: aylık HP grubuna göre bakiye parçaları ve ay sonu aktif sayıları.
- sube_stok_mutabakat.csv: açılış + devir dışı stok değişimi + net devir = kapanış.
  Bu AY SONU STOK eşitliğidir; aylık ortalamalara akım ekleme formülü değildir.
- banka_aylik.csv / sube_hareket_ozeti.csv: yeni müşteri, devir, ayrılma ve dış kaynak hareketleri.
- segment_esikleri.csv: ay sonu aktif kitlenin o ayki mevduat + yatırım
  ortalamalarına göre kalibrasyon; havuz/portföy kişi oranı yaklaşık 15.
  Ayrılmış satırların segmenti son aylık görünüm olarak saklanır; oran evrenine girmez.
- ocak2027_banka_hedefleri.csv: yeni Aralık ortalamalarına %15/%10/%5 uygulanır.
  Hedefler veri üretiminde kullanılmaz. Ocak gerçekleşmesi oluşturulmadı.

## Zaman ve muhasebe kuralları

Aylık ortalama = günlük kapanış stokları toplamı / ayın bütün takvim günleri.
Katılım öncesi ve ayrılış sonrası günler 0'dır; aktif günlere bölünmez.
Ayın 16'sındaki devirde 1–15 eski şubeye, 16–ayın sonu yeni şubeye yazılır.
İlişkinin tamamı devredilir; gerçek bankadaki ürün/hesap bazlı aidiyet ayrıca uyarlanmalıdır.
Müşteri ortalaması kuruşa yuvarlanır, şube parçalarına en büyük kalan yöntemiyle
paylaştırılır; parçaların toplamı müşteri ortalamasını kuruşu kuruşuna verir.

Tam ayrılmadan önce kredi ve kart borcu kapatılır. Diğer bankaya kısmi kaynak
çıkışında ilişki ve kredi devam edebilir. Vade dönüşü ve ürün kapama/iç aktarım
bankanın toplam mevduat+yatırım stokunu korur, alt ürün dağılımını değiştirebilir.
Faiz/anapara ayrımı, kur değerlemesi ve tam banka muhasebesi simüle edilmez;
olay defteri ürün stoklarını izler, çift taraflı muhasebe defteri değildir.
Diğer banka riski aktif ilişki için sentetiktir; ayrılmış müşteri için bilinmiyor/boş.
BES dönem sonu stoktur. İşlem/prim/katkı alanları aylık akımdır ve aktif süreye
göre ölçeklenir; hedeflenen üç aylık ortalama günlük stok defterinden hesaplanır.

## Model kullanım sınırı

Eski modeller ve dashboard sonuçları V3'e aittir; V4 ile yeniden eğitilmedi.
V3 pipeline'ındaki sabit müşteri–şube birleştirmesi V4 için kullanılamaz.
Demografi için sube_model_girdileri_aylik, bakiye için gün bazlı aidiyet tablosu
kullanılmalı; gelecek ay olayları geçmiş girdilere eklenmemelidir.
İş akışını sınamak için üretilmiş veridir. Algoritma üstünlüğü sağlamak için
sonuçlara göre olay veya oran ayarı yapılmadı. Dönemler arasında aynı mekanizma geçerlidir.
''',encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ['registered_customers','latest_active_customers','customer_month_rows','event_rows','generation_seconds']},indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=ROOT/'outputs/banka_test_verisi_v3')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/banka_test_verisi_v4')
    args=parser.parse_args();run(args.source.resolve(),args.output.resolve())
