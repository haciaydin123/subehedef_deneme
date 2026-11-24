"""V4 feature/anchor definitions, shared without changes by all algorithms."""
import lightgbm as lgb
import numpy as np
import pipeline as base

base.lgb=lgb
SEED=base.SEED;TARGETS=base.TARGETS;GROUPS=base.GROUPS
STRUCTURAL=base.STRUCTURAL;PROXIES=base.PROXIES
CONFIGS=base.CONFIGS;metrics=base.metrics;save_csv=base.save_csv;save_json=base.save_json
design=base.design;make_model=base.make_model
VIEWS=['yapisal','gecikmeli','hareketli']
EXTRAS={
 'ortalama_iliski_skoru':('iliski_aktivite','Ay sonu aktif müşterilerin ilişki skoru ortalaması','skor'),
 'yeni_musteri_sayisi':('musteri_hareketi','t ayında şubeye kazanılan müşteri','kişi'),
 'ayrilan_musteri_sayisi':('musteri_hareketi','t ayında bankadan tam ayrılan müşteri','kişi'),
 'devir_giris_musteri_sayisi':('musteri_hareketi','t ayında başka şubeden gelen müşteri','kişi'),
 'devir_cikis_musteri_sayisi':('musteri_hareketi','t ayında başka şubeye devredilen müşteri','kişi'),
 'diger_bankaya_kaynak_cikisi_tl':('kaynak_hareketi','t ayında diğer bankalara kısmi varlık çıkışı','TL'),
 'diger_bankadan_kaynak_girisi_tl':('kaynak_hareketi','t ayında diğer bankalardan varlık girişi','TL'),
 'musteri_gun':('musteri_hareketi','t ayında şubedeki aktif müşteri-günü','müşteri-gün'),
 'ay_ici_hizmet_verilen_musteri_sayisi':('musteri_hareketi','Ay içinde herhangi bir gün hizmet verilen müşteri','kişi'),
 'musteri_sayisi_degisim':('musteri_hareketi','Aktif müşteri sayısının t ile t-1 farkı','kişi'),
 'musteri_gun_ay_sonu_orani':('musteri_hareketi','Müşteri-günü / takvim günü / ay sonu aktif müşteri','oran'),
}

def features(target,view):
    cols=base.features(target,'yapisal' if view=='yapisal' else 'gecikmeli')
    if view=='hareketli':
        cols+=list(EXTRAS)+[target+'_ay_sonu_bakiye_tl',target+'_kapanis_ortalama_farki_tl',
                          target+'_devir_net_degisim_tl',target+'_devir_haric_net_degisim_tl']
    assert len(cols)==len(set(cols))
    return cols

def anchor_field(target,view):
    return target+'_ay_sonu_bakiye_tl' if view=='hareketli' else target+'_bakiye_t' if view=='gecikmeli' else None

def anchor(frame,target,view):
    col=anchor_field(target,view)
    return frame[col].to_numpy(float) if col else np.zeros(len(frame))

def feature_group(field):
    for specs in [STRUCTURAL,PROXIES,EXTRAS]:
        if field in specs:return specs[field][0]
    if '_ay_sonu_' in field or '_kapanis_ortalama_' in field:return 'donem_sonu'
    if '_devir_' in field:return 'kaynak_hareketi'
    return 'gecmis_bakiye'

def permutations(model,frame,target,view,levels,feature_groups,repeats=8):
    frame=frame.sort_values(['hedef_donemi','sube_id']).reset_index(drop=True)
    x=design(frame,features(target,view),levels)
    y=frame['y_'+target+'_tl'].to_numpy(float);offset=anchor(frame,target,view)
    baseline=np.abs(y-np.maximum(0,offset+model.predict(x))).mean()
    n_branch=frame['sube_id'].nunique();n_month=frame['hedef_donemi'].nunique()
    ids=frame['sube_id'].to_numpy().reshape(n_month,n_branch)
    assert (ids==ids[0]).all()
    rng=np.random.default_rng(SEED)
    permutations=[np.arange(len(frame)).reshape(n_month,n_branch)[:,rng.permutation(n_branch)].ravel() for _ in range(repeats)]
    rows=[]
    for label,cols in feature_groups.items():
        diffs=[]
        for take in permutations:
            changed=x.copy()
            for col in cols:changed[col]=x[col].iloc[take].array
            shifted=offset[take] if anchor_field(target,view) in cols else offset
            pred=np.maximum(0,shifted+model.predict(changed))
            diffs.append(float(np.abs(y-pred).mean()-baseline))
        rows.append({'alan_veya_grup':label,'delta_mae_tl':float(np.mean(diffs)),
          'delta_mae_std_tl':float(np.std(diffs,ddof=1)),'normalize_delta_mae':float(np.mean(diffs)/y.mean()),'tekrar':repeats})
    return rows
