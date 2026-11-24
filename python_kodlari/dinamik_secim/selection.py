"""Past-only quality filters, common CatBoost and blocked permutation stability."""
import hashlib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

MISSING='__MISSING__';UNKNOWN='__UNKNOWN__'

def folds(panel,as_of,rules):
    end=str(pd.Period(as_of,freq='M')-rules['holdout_months'])
    first=pd.Period(end,freq='M')-rules['windows']*rules['validation_months']+1
    result=[]
    for i in range(rules['windows']):
        start=first+i*rules['validation_months'];stop=start+rules['validation_months']-1
        train=panel[panel.__label_period.lt(str(start))&panel.__label_period.le(end)].copy()
        valid=panel[panel.__label_period.between(str(start),str(stop))].copy()
        if train.__label_period.nunique()<rules['min_train_months'] or valid.empty:raise ValueError('Insufficient history for configured rolling windows')
        result.append((f'W{i+1}',train.reset_index(drop=True),valid.reset_index(drop=True)))
    return result,end

def quality_filter(train,fields,rules):
    retained=[];decisions=[]
    for f in sorted(fields,key=lambda f:(f.get('priority',50),f['name'])):
        name=f['name'];s=train[name];missing=float(s.isna().mean());dominance=float(s.value_counts(normalize=True).max()) if s.notna().any() else 1.
        reason='';status='eligible'
        if missing>rules['max_missing_fraction']:reason='Yüksek eğitim dönemi eksikliği';status='excluded_missing'
        elif s.nunique(dropna=True)<2:reason='Eğitim döneminde sabit veya tamamen eksik';status='excluded_constant'
        elif dominance>=rules['max_dominance']:reason='Eğitim döneminde baskın tek değer';status='excluded_dominance'
        elif f['dtype']=='categorical' and s.nunique()>rules.get('max_categories',100):reason='Yüksek kategori sayısı; kimlik/serbest metin incelemesi gerekli';status='review_cardinality'
        representative=None;correlation=None
        if not reason and f['dtype']=='numeric':
            for other in retained:
                if other['dtype']!='numeric':continue
                corr=train[[name,other['name']]].corr(method='spearman').iloc[0,1]
                if np.isfinite(corr) and abs(corr)>=rules['correlation_threshold']:
                    representative=other['name'];correlation=float(corr)
                    reason=f'Benzer bilgiyi {representative} temsil ediyor';status='excluded_correlated';break
        if not reason:retained.append(f)
        if f.get('must_keep') and reason:raise ValueError(f'must_keep field fails quality checks: {name}: {reason}')
        decisions.append({'name':name,'label':f.get('label',name),'family':f.get('family','other'),
            'dtype':f['dtype'],'status':status,'reason':reason,'missing_fraction_train':missing,
            'dominance_train':dominance,'representative':representative,'correlation':correlation,
            'must_keep':bool(f.get('must_keep',False))})
    if not retained:raise ValueError('No eligible candidates after quality controls')
    return retained,decisions

def fit_transformer(frame,fields):
    params={}
    for f in fields:
        name=f['name'];s=frame[name]
        if f['dtype']=='numeric':
            median=float(s.median());q=s.quantile([.25,.75]);iqr=float(q.iloc[1]-q.iloc[0])
            params[name]={'dtype':'numeric','median':median,'iqr':iqr if iqr>0 else 1.}
        else:
            values=sorted(set(s.dropna().astype(str)))
            if MISSING in values or UNKNOWN in values:raise ValueError('Input category collides with reserved missing/unknown tokens')
            params[name]={'dtype':'categorical','levels':values}
    return {'features':[f['name'] for f in fields],'columns':params}

def transform(frame,params):
    x=pd.DataFrame(index=frame.index)
    for name in params['features']:
        if name not in frame:raise ValueError(f'Saved schema requires field: {name}')
        spec=params['columns'][name];s=frame[name]
        if spec['dtype']=='numeric':x[name]=s.fillna(spec['median']).astype(float)
        else:
            s=s.astype('string')
            x[name]=s.where(s.isna()|s.isin(spec['levels']),UNKNOWN).fillna(MISSING).astype(str)
    return x

def blocked_take(frame,seed):
    rng=np.random.default_rng(seed);take=np.arange(len(frame))
    for indices in frame.groupby(['__period','__group'],sort=True).indices.values():
        take[indices]=rng.permutation(indices)
    return take

def mae_wape(y,pred):
    y=np.asarray(y,float);p=np.maximum(0,np.asarray(pred,float));den=float(np.abs(y).sum())
    if den<=0:raise ValueError('Target has zero absolute total; WAPE undefined')
    return {'n':len(y),'mae_tl':float(np.abs(y-p).mean()),'wape':float(np.abs(y-p).sum()/den)}

def fit_window(train,valid,target,fields,rules,window):
    label='__y_'+target['name'];train=train[train[label].notna()].reset_index(drop=True);valid=valid[valid[label].notna()].reset_index(drop=True)
    if len(train)<3 or len(valid)<2:raise ValueError('Insufficient labeled rows')
    prep=fit_transformer(train,fields);x=transform(train,prep);xv=transform(valid,prep)
    model=CatBoostRegressor(**rules['catboost'],thread_count=rules['threads'],random_seed=rules['seed'],
                            allow_writing_files=False,verbose=False)
    model.fit(x,train[label],cat_features=[f['name'] for f in fields if f['dtype']=='categorical'])
    pred=np.maximum(0,model.predict(xv,thread_count=rules['threads']));actual=valid[label].to_numpy(float);score=mae_wape(actual,pred)
    baseline=np.repeat(float(train[label].mean()),len(valid));baseline_score=mae_wape(actual,baseline)
    informative=score['wape']<baseline_score['wape']*(1-rules.get('min_baseline_improvement',.01))
    seed=int(hashlib.sha256(f'{rules["seed"]}/{window}/{target["name"]}'.encode()).hexdigest()[:8],16)
    importance=[]
    for name in prep['features']:
        for repeat in range(rules['permutation_repeats']):
            take=blocked_take(valid,seed+repeat);changed=xv.copy();changed[name]=xv[name].iloc[take].to_numpy()
            shifted=np.maximum(0,model.predict(changed,thread_count=rules['threads']));loss=mae_wape(actual,shifted)
            importance.append({'window':window,'target':target['name'],'feature':name,'repeat':repeat,
                'delta_wape':loss['wape']-score['wape'],'delta_mae_tl':loss['mae_tl']-score['mae_tl'],
                'permuted_wape':loss['wape'],'baseline_wape':score['wape']})
    bundle={'model':model,'transformer':prep,'target':target['name'],'window':window,'permutation_seed':seed,
            'train_label_end':train.__label_period.max(),'validation_start':valid.__label_period.min(),
            'validation_end':valid.__label_period.max(),'algorithm':'catboost'}
    predictions=valid[['__branch','__period','__group','__label_period']].copy()
    predictions['window']=window;predictions['target']=target['name'];predictions['actual']=actual
    predictions['prediction']=pred;predictions['train_mean_prediction']=baseline
    metric={'window':window,'target':target['name'],**score,'train_mean_wape':baseline_score['wape'],
        'informative':bool(informative),'train_rows':len(train),'train_start':train.__label_period.min(),
        'train_end':train.__label_period.max(),'validation_start':valid.__label_period.min(),'validation_end':valid.__label_period.max()}
    return bundle,predictions,importance,metric

def select_features(decisions,importance,metrics,rules):
    importance=pd.DataFrame(importance);metrics=pd.DataFrame(metrics)
    means=importance.groupby(['window','target','feature']).delta_wape.mean().reset_index()
    # Targets share dimensionless WAPE units and receive equal weight, regardless of balance size.
    windows=means.groupby(['window','feature']).delta_wape.mean().reset_index()
    trusted=metrics.groupby('window').informative.sum().ge(rules.get('min_informative_targets',1)).to_dict()
    result=[]
    for d in decisions:
        d=d.copy()
        if d['status']=='eligible':
            values=windows[windows.feature.eq(d['name'])];avg=float(values.delta_wape.mean())
            positive=sum(v.delta_wape>rules['min_delta_wape'] and trusted[v.window] for v in values.itertuples())
            d.update(mean_delta_wape=avg,positive_windows=positive,windows=len(values),window_scores={r.window:float(r.delta_wape) for r in values.itertuples()})
            if d['must_keep']:d.update(status='selected',reason='Alan sözlüğünde iş gerekçesiyle zorunlu tutuldu')
            elif not all(trusted.values()):d.update(status='review_model_weak',reason='Bazı dönemlerde model referansa göre yeterince açıklayıcı değil')
            elif positive>=rules['min_stable_windows'] and avg>rules['min_delta_wape']:d.update(status='selected',reason='Pozitif katkı ve dönem kararlılığı koşulları sağlandı')
            else:d.update(status='not_selected',reason='Pozitif katkı/dönem kararlılığı koşulları sağlanmadı; iş incelemesiyle yeniden değerlendirilebilir')
        result.append(d)
    return result,windows.to_dict('records')
