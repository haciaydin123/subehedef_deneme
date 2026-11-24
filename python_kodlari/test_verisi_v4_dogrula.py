"""Independent replay of integer-kuruş events; never imports the generator."""
import argparse
import calendar
import collections
import csv
from decimal import Decimal,ROUND_HALF_UP
import hashlib
import itertools
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
PRODUCTS=['tl_vadesiz','doviz_vadesiz','altin_hesabi','tl_vadeli','doviz_vadeli',
 'ihtiyac_kredi','konut_kredi','tasit_kredi','kmh','fon','hisse','tahvil_bono','kart_borc','kart_limit','bes_birikim']
AVERAGES=[x+'_ort_bakiye_tl' for x in PRODUCTS[:12]]+['kart_borc_ort_tl']
ASSETS=[0,1,2,3,4,9,10,11];LOANS=[5,6,7,8]
TARGETS={'tl_vadesiz':[0],'tl_vadeli':[3],'kredi':LOANS}
FAMILIES={'vadesiz':[0,1,2],'vadeli':[3,4],'kredi':LOANS,'yatirim':[9,10,11]}

def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:yield from csv.DictReader(f)
def C(value):
    n=Decimal(value)*100;assert n==int(n);return int(n)
def T(vector,index):return sum(vector[i] for i in index)
def avg(numerator,days):
    q,r=divmod(numerator,days);return q+int(2*r>=days)
def assert_close(a,b,tol=1e-6):assert abs(float(a)-float(b))<=tol,(a,b)

def boundary_cases():
    # 30-day month; events effective at the start of day 16.
    assert avg(100_000_000*15,30)==50_000_000
    assert avg(100_000_000*15,30)+avg(100_000_000*15,30)==100_000_000
    # Entry on final day is one day of exposure, not a full month.
    assert avg(3_100_000,31)==100_000
    # First-day exit removes all days; last-day exit retains 30 days in December.
    assert avg(0,31)==0 and avg(3_100_000*30,31)==3_000_000

def run(out,source):
    started=time.perf_counter();boundary_cases()
    manifest=json.loads((out/'manifest.json').read_text())
    source_manifest=json.loads((source/'manifest.json').read_text())
    for name,spec in source_manifest['csv_files'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==spec['sha256'],name
    for name,spec in manifest['csv_files'].items():
        assert hashlib.sha256((out/name).read_bytes()).hexdigest()==spec['sha256'],name
    profiles={r['musteri_id']:r for r in rows(out/'musteriler.csv')}
    positions={r['musteri_id']:{'v':[int(r[p+'_bakiye_kurus']) for p in PRODUCTS],
              'branch':r['sube_id'],'active':int(r['aktif'])} for r in rows(out/'acilis_bakiyeleri.csv')}
    branch_ids=[r['sube_id'] for r in rows(out/'subeler.csv')]
    previous_close={(b,t):sum(T(c['v'],idx) for c in positions.values() if c['branch']==b and c['active'])
                    for b in branch_ids for t,idx in TARGETS.items()}
    customer_months=itertools.groupby(rows(out/'musteri_aylik.csv'),lambda r:r['donem'])
    bridge_months=iter(itertools.groupby(rows(out/'musteri_sube_aylik.csv'),lambda r:r['donem']))
    event_stream=iter(rows(out/'musteri_olaylari.csv'));event=next(event_stream,None)
    branch_table={(r['donem'],r['sube_id']):r for r in rows(out/'sube_aylik.csv')}
    hp_table={(r['donem'],r['sube_id'],r['hp']):r for r in rows(out/'sube_hp_aylik.csv')}
    bank_table={r['donem']:r for r in rows(out/'banka_aylik.csv')}
    rollforward={(r['donem'],r['sube_id'],r['gosterge']):r for r in rows(out/'sube_stok_mutabakat.csv')}
    features={(r['donem'],r['sube_id']):r for r in rows(out/'sube_model_girdileri_aylik.csv')}
    count=0;event_count=0;bridge_count=0;all_counts=collections.Counter();prior_active=10127
    examples=[];example_types=set();months_seen=[];previous_keys=set()
    for period,frame in customer_months:
        months_seen.append(period);days=calendar.monthrange(*map(int,period.split('-')))[1]
        monthly=list(frame);current_keys=set();expected_parts={};monthly_by_id={}
        movement=collections.defaultdict(collections.Counter);delta_by_branch=collections.Counter();transfer=collections.Counter()
        month_counts=collections.Counter();active_owners=collections.defaultdict(list)
        for row in monthly:
            cid=row['musteri_id'];assert cid not in current_keys;current_keys.add(cid);monthly_by_id[cid]=row
            assert profiles[cid]['katilim_tarihi'][:7]<=period
            if cid not in positions:
                positions[cid]={'v':[0]*15,'branch':profiles[cid]['ilk_sube_id'],'active':0}
                assert profiles[cid]['katilim_tarihi'][:7]==period
            state=positions[cid];numerator=[0]*15;parts={};day_cursor=1;active_days=0;customer_counts=collections.Counter()
            def add_days(length):
                nonlocal active_days
                assert length>=0
                for i,value in enumerate(state['v']):numerator[i]+=value*length
                if state['active'] and length:
                    part=parts.setdefault(state['branch'],{'days':0,'n':[0]*13})
                    part['days']+=length;active_days+=length
                    for i in range(13):part['n'][i]+=state['v'][i]*length
            while event and event['donem']==period and event['musteri_id']==cid:
                assert event['olay_tarihi'][:7]==period and event['bilinirlik_tarihi']==event['olay_tarihi']
                day=int(event['olay_tarihi'][-2:]);assert 1<=day<=days
                add_days(day-day_cursor);day_cursor=day
                v=state['v'];before=v[:];origin=state['branch'];kind=event['olay_tipi']
                assert event['onceki_sube_id']==origin
                change=[int(event[p+'_delta_kurus']) for p in PRODUCTS]
                if kind=='SUBE_DEVRI':
                    assert state['active']==1 and not any(change)
                    assert event['sonraki_sube_id']!=origin and int(event['sonraki_aktif'])==1
                    for target,idx in TARGETS.items():
                        amount=T(before,idx);assert int(event['devir_'+target+'_kurus'])==amount
                        transfer[(origin,target)]-=amount;transfer[(event['sonraki_sube_id'],target)]+=amount
                    movement[origin]['devir_cikis_musteri_sayisi']+=1
                    movement[event['sonraki_sube_id']]['devir_giris_musteri_sayisi']+=1
                if kind=='YENI_MUSTERI':
                    assert state['active']==0 and not any(before)
                    movement[origin]['yeni_musteri_sayisi']+=1
                if kind=='BANKADAN_TAM_AYRILMA':
                    assert T(before,LOANS+[12])==0
                    movement[origin]['ayrilan_musteri_sayisi']+=1
                if kind in ['VADE_DONUSU','URUN_KAPAMA_VE_IC_AKTARIM']:assert T(change,ASSETS)==0
                if kind=='DIGER_BANKAYA_KAYNAK_CIKISI':
                    assert T(change,ASSETS)<0 and all(change[i]==0 for i in LOANS)
                    movement[origin]['diger_bankaya_kaynak_cikisi_tl']-=T(change,ASSETS)
                if kind=='DIGER_BANKADAN_KAYNAK_GIRISI':movement[origin]['diger_bankadan_kaynak_girisi_tl']+=T(change,ASSETS)
                for target,idx in TARGETS.items():delta_by_branch[(origin,target)]+=T(change,idx)
                state['v']=[a+b for a,b in zip(before,change)]
                assert min(state['v'])>=0 and state['v'][12]<=state['v'][13]
                state['branch']=event['sonraki_sube_id'];state['active']=int(event['sonraki_aktif'])
                if not state['active']:assert not any(state['v'])
                month_counts[kind]+=1;customer_counts[kind]+=1;all_counts[kind]+=1;event_count+=1
                event=next(event_stream,None)
            add_days(days+1-day_cursor)
            assert row['sube_id']==state['branch'] and int(row['ay_sonu_aktif'])==state['active']
            assert int(row['aktif_gun_sayisi'])==active_days<=days
            for i,col in enumerate(AVERAGES):assert C(row[col])==avg(numerator[i],days),(period,cid,col)
            for i,p in enumerate(PRODUCTS[:12]):assert C(row[p+'_ay_sonu_bakiye_tl'])==state['v'][i]
            assert C(row['kart_borc_ay_sonu_tl'])==state['v'][12]
            assert C(row['kart_limit_tl'])==state['v'][13] and C(row['bes_donem_sonu_birikim_tl'])==state['v'][14]
            for family,idx in FAMILIES.items():assert C(row[family+'_ort_bakiye_tl'])==sum(C(row[AVERAGES[i]]) for i in idx)
            assert C(row['toplam_varlik_ort_tl'])==sum(C(row[AVERAGES[i]]) for i in ASSETS)
            for field,kind in [('devir_sayisi','SUBE_DEVRI'),('ayrilan_musteri','BANKADAN_TAM_AYRILMA'),('yeni_musteri','YENI_MUSTERI')]:
                assert int(row[field])==customer_counts[kind]
            if not state['active']:
                assert row['diger_banka_kredi_ort_risk_tl']==row['memzuc_ort_nakdi_risk_tl']==''
            else:
                assert C(row['memzuc_ort_nakdi_risk_tl'])==C(row['kredi_ort_bakiye_tl'])+C(row['kart_borc_ort_tl'])+C(row['diger_banka_kredi_ort_risk_tl'])
                active_owners[state['branch']].append(row)
            for bid,part in parts.items():expected_parts[(cid,bid)]=part
            count+=1
        assert previous_keys<=current_keys;previous_keys=current_keys
        bridge_period,bridge_rows=next(bridge_months);assert bridge_period==period
        customer_sums=collections.defaultdict(lambda:[0]*13)
        branch_sums=collections.defaultdict(collections.Counter);hp_sums=collections.defaultdict(collections.Counter)
        seen_parts=set()
        for row in bridge_rows:
            cid,bid=row['musteri_id'],row['sube_id'];key=(cid,bid);assert key not in seen_parts;seen_parts.add(key)
            expected=expected_parts[key];assert int(row['aktif_gun_sayisi'])==expected['days']
            assert int(row['ay_sonu_bu_subede_aktif'])==int(positions[cid]['active'] and positions[cid]['branch']==bid)
            assert row['hp']==monthly_by_id[cid]['hp']
            for i,col in enumerate(AVERAGES):
                amount=C(row[col]);assert abs(amount*days-expected['n'][i])<days
                customer_sums[cid][i]+=amount;branch_sums[bid][col]+=amount;hp_sums[(bid,row['hp'])][col]+=amount
            for family,idx in FAMILIES.items():assert C(row[family+'_ort_bakiye_tl'])==sum(C(row[AVERAGES[i]]) for i in idx)
            bridge_count+=1
        assert seen_parts==set(expected_parts)
        for cid,row in monthly_by_id.items():
            assert customer_sums[cid]==[C(row[c]) for c in AVERAGES]
        closing={}
        for bid in branch_ids:
            branch=branch_table[(period,bid)];members=active_owners[bid];n=len(members)
            assert int(branch['musteri_sayisi'])==n
            assert int(branch['havuz_musteri_sayisi'])+int(branch['portfoy_musteri_sayisi'])==n
            assert_close(branch['ortalama_yas'],sum(int(r['yas']) for r in members)/n if n else 0)
            assert_close(branch['ortalama_aylik_gelir_tl'],sum(float(r['aylik_gelir_tl']) for r in members)/n if n else 0,tol=.0051)
            for col in AVERAGES:assert C(branch[col])==branch_sums[bid][col]
            for col in ['yeni_musteri_sayisi','ayrilan_musteri_sayisi','devir_giris_musteri_sayisi','devir_cikis_musteri_sayisi']:
                assert int(branch[col])==movement[bid][col]
            for col in ['diger_bankaya_kaynak_cikisi_tl','diger_bankadan_kaynak_girisi_tl']:
                assert C(branch[col])==movement[bid][col]
            for hp in ['HAVUZ','PORTFOY']:
                hp_row=hp_table[(period,bid,hp)]
                assert int(hp_row['musteri_sayisi'])==sum(r['hp']==hp for r in members)
                for target,idx in TARGETS.items():
                    col=target+'_ort_bakiye_tl'
                    assert C(hp_row[col])==sum(hp_sums[(bid,hp)][AVERAGES[i]] for i in idx)
            for target,idx in TARGETS.items():
                closing[(bid,target)]=sum(T(positions[r['musteri_id']]['v'],idx) for r in members)
                rec=rollforward[(period,bid,target)]
                assert int(rec['acilis_kurus'])==previous_close[(bid,target)]
                assert int(rec['devir_haric_net_degisim_kurus'])==delta_by_branch[(bid,target)]
                assert int(rec['devir_net_degisim_kurus'])==transfer[(bid,target)]
                assert int(rec['kapanis_kurus'])==closing[(bid,target)]
                assert previous_close[(bid,target)]+delta_by_branch[(bid,target)]+transfer[(bid,target)]==closing[(bid,target)]
            f=features[(period,bid)]
            assert int(f['musteri_sayisi'])==n
            assert_close(f['evli_orani'],sum(profiles[r['musteri_id']]['medeni_durum']=='EVLI' for r in members)/n if n else 0)
        bank=bank_table[period];active=sum(len(v) for v in active_owners.values())
        assert int(bank['ay_basi_aktif_musteri'])==prior_active
        assert int(bank['ay_sonu_aktif_musteri'])==active==prior_active+month_counts['YENI_MUSTERI']-month_counts['BANKADAN_TAM_AYRILMA']
        assert int(bank['sube_devir_sayisi'])==month_counts['SUBE_DEVRI']>0
        for target,idx in TARGETS.items():
            col=target+'_ort_bakiye_tl'
            assert C(bank[col])==sum(C(r[col]) for r in monthly)
            assert sum(transfer[(b,target)] for b in branch_ids)==0
        h=sum(r['ay_sonu_aktif']=='1' and r['hp']=='HAVUZ' for r in monthly)
        p=active-h;assert abs(h/p-15)<.03
        assert int(bank['havuz_musteri_sayisi'])==h and int(bank['portfoy_musteri_sayisi'])==p
        if period=='2026-12':assert list(rows(out/'musteri_son_donem.csv'))==monthly
        # Capture one complete transfer, entry and exit example with weighted ownership.
        for kind,field in [('SUBE_DEVRI','devir_sayisi'),('YENI_MUSTERI','yeni_musteri'),('BANKADAN_TAM_AYRILMA','ayrilan_musteri')]:
            if kind in example_types:continue
            found=next((r for r in monthly if int(r[field])>0),None)
            if found:
                cid=found['musteri_id'];example_types.add(kind)
                examples.append({'olay':kind,'donem':period,'musteri_id':cid,'ay_sonu_sube':found['sube_id'],
                  'aktif_gun':found['aktif_gun_sayisi'],'tl_vadesiz_ort_tl':float(found['tl_vadesiz_ort_bakiye_tl']),
                  'tl_vadesiz_ay_sonu_tl':float(found['tl_vadesiz_ay_sonu_bakiye_tl']),
                  'ay_sonu_aktif':int(found['ay_sonu_aktif']),
                  'sube_parcalari':[{'sube_id':bid,'gun':part['days'],'tl_vadesiz_ortalama_ham_tl':part['n'][0]/days/100}
                    for (customer,bid),part in expected_parts.items() if customer==cid]})
        previous_close=closing;prior_active=active
        print('Verified',period,'customer rows',len(monthly),flush=True)
    assert event is None and next(bridge_months,None) is None
    assert len(months_seen)==26 and len(branch_table)==5200 and len(rollforward)==15600
    assert count==manifest['customer_month_rows'] and event_count==manifest['event_rows'] and bridge_count==manifest['customer_branch_month_rows']
    assert dict(all_counts)==manifest['event_counts']
    bank=bank_table['2026-12']
    for b in rows(out/'ocak2027_banka_hedefleri.csv'):
        reference=C(bank[b['veri_alani']]);assert reference==C(b['referans_aylik_ortalama_tl'])
        expected=int((Decimal(reference)*(1+Decimal(b['artis_orani']))).quantize(Decimal('1'),rounding=ROUND_HALF_UP))
        assert C(b['hedef_aylik_ortalama_tl'])==expected
    manifest['checks']={'independent_event_replay_all_customer_months':True,'all_stock_nonnegative_and_card_within_limit':True,
      'daily_calendar_weighted_averages':True,'customer_branch_parts_reconcile_exactly':True,
      'branch_transfer_preserves_bank_stock':True,'all_15600_branch_stock_rollforwards':True,
      'new_customer_exit_count_rollforward':True,'post_exit_zero_bank_balances_and_unknown_external_risk':True,
      'dynamic_branch_demographics':True,'hp_ratio_approximately_15_active_customers':True,
      'source_v3_hashes_unchanged':True,'all_output_csv_hashes':True,'january_bank_targets':True,'boundary_timing_cases':True}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    result={'status':'passed','customer_months_replayed':count,'events_replayed':event_count,
      'customer_branch_parts':bridge_count,'stock_rollforwards':15600,'seconds':time.perf_counter()-started,'examples':examples}
    (out/'dogrulama_raporu.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='examples'},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs/banka_test_verisi_v4')
    p.add_argument('--source',type=Path,default=ROOT/'outputs/banka_test_verisi_v3')
    args=p.parse_args();run(args.output.resolve(),args.source.resolve())
