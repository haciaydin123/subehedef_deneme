"""Fixed-reference mixed-data geometry, authentic URF proximity and PAM."""
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from kmodes.kprototypes import KPrototypes

METHODS={'urf_kmedoids':'URF + K-Medoids','gower_kmedoids':'Gower + K-Medoids','kprototypes':'K-Prototypes'}

def fit_geometry(frame,schema):
    numeric=[n for n in schema['features'] if schema['transformer']['columns'][n]['dtype']=='numeric']
    categorical=[n for n in schema['features'] if n not in numeric]
    ranges={n:{'low':float(frame[n].min()),'high':float(frame[n].max())} for n in numeric}
    levels={n:list(dict.fromkeys(schema['transformer']['columns'][n]['levels']+['__UNKNOWN__','__MISSING__'])) for n in categorical}
    return {'features':schema['features'],'numeric':numeric,'categorical':categorical,'ranges':ranges,'levels':levels}

def normalize(frame,geometry):
    columns=[]
    for n in geometry['numeric']:
        s=frame[n].to_numpy(float);bounds=geometry['ranges'][n]
        span=bounds['high']-bounds['low']
        columns.append(np.clip((s-bounds['low'])/span,0,1) if span>0 else np.zeros(len(s)))
    num=np.column_stack(columns) if columns else np.empty((len(frame),0))
    cats=[]
    for n in geometry['categorical']:
        values=frame[n].astype(str).to_numpy();known=np.isin(values,geometry['levels'][n])
        cats.append(np.where(known,values,'__UNKNOWN__'))
    cat=np.column_stack(cats) if cats else np.empty((len(frame),0),dtype=object)
    return num,cat

def onehot(num,cat,geometry):
    parts=[num]
    for j,n in enumerate(geometry['categorical']):
        parts.append(np.column_stack([(cat[:,j]==level).astype(float) for level in geometry['levels'][n]]))
    return np.column_stack(parts)

def synthetic(num,cat,seed):
    # Shuffle original variables, not individual one-hot columns.
    rng=np.random.default_rng(seed);a=num.copy();b=cat.copy()
    for j in range(a.shape[1]):a[:,j]=rng.permutation(a[:,j])
    for j in range(b.shape[1]):b[:,j]=rng.permutation(b[:,j])
    return a,b

def gower(a_num,a_cat,b_num,b_cat):
    p=a_num.shape[1]+a_cat.shape[1]
    if not p:raise ValueError('No usable fields')
    result=np.zeros((len(a_num),len(b_num)))
    for j in range(a_num.shape[1]):result+=np.abs(a_num[:,j,None]-b_num[None,:,j])
    for j in range(a_cat.shape[1]):result+=(a_cat[:,j,None]!=b_cat[None,:,j])
    return result/p

def proximity_distance(leaves_a,leaves_b):
    proximity=np.zeros((len(leaves_a),len(leaves_b)))
    for j in range(leaves_a.shape[1]):proximity+=(leaves_a[:,j,None]==leaves_b[None,:,j])
    return np.sqrt(np.maximum(0,1-proximity/leaves_a.shape[1]))

def pam(distance,k,seed,starts=3,max_iter=50):
    n=len(distance)
    if not 1<k<n:raise ValueError('PAM requires 1 < k < n')
    if not np.isfinite(distance).all() or not np.allclose(distance,distance.T) or not np.allclose(np.diag(distance),0):raise ValueError('Invalid pairwise distance matrix')
    rng=np.random.default_rng(seed);best=None
    for start in range(starts):
        if start==0:
            chosen=[int(np.argmin(distance.sum(axis=0)))];current=distance[:,chosen].min(axis=1)
            while len(chosen)<k:
                options=[i for i in range(n) if i not in chosen]
                selected=min(options,key=lambda i:(np.minimum(current,distance[:,i]).sum(),i))
                chosen.append(selected);current=np.minimum(current,distance[:,selected])
        else:chosen=rng.choice(n,k,replace=False).tolist()
        chosen=sorted(chosen);cost=float(distance[:,chosen].min(axis=1).sum());converged=False
        for iteration in range(max_iter):
            proposal=None
            for pos in range(k):
                for candidate in range(n):
                    if candidate in chosen:continue
                    swapped=sorted(chosen[:pos]+chosen[pos+1:]+[candidate])
                    new_cost=float(distance[:,swapped].min(axis=1).sum())
                    if new_cost<cost-1e-12 and (proposal is None or (new_cost,swapped)<proposal):proposal=(new_cost,swapped)
            if proposal is None:converged=True;break
            cost,chosen=proposal
        item=(cost,chosen,converged)
        if best is None or (cost,chosen)<(best[0],best[1]):best=item
    cost,chosen,converged=best
    labels=np.argmin(distance[:,chosen],axis=1)
    return {'medoid_indices':chosen,'labels':labels,'cost':cost,'converged':converged}

def fit_base(method,frame,geometry,seed,settings):
    num,cat=normalize(frame,geometry)
    base={'method':method,'geometry':geometry,'reference_branch_ids':frame['__branch'].astype(str).tolist(),
          'fit_period':str(frame['__period'].iloc[0]),'seed':int(seed)}
    if method=='urf_kmedoids':
        fake_num,fake_cat=synthetic(num,cat,seed)
        real=onehot(num,cat,geometry);fake=onehot(fake_num,fake_cat,geometry)
        forest_settings=dict(settings['forest'])
        if 'mtry_ratio' in settings:forest_settings['max_features']=settings['mtry_ratio']
        forest=RandomForestClassifier(**forest_settings,random_state=seed,n_jobs=settings['threads'],oob_score=True)
        forest.fit(np.vstack([real,fake]),np.r_[np.zeros(len(real),int),np.ones(len(fake),int)])
        leaves=forest.apply(real)
        encoded_columns=list(geometry['numeric'])+[name+'=='+str(level) for name in geometry['categorical'] for level in geometry['levels'][name]]
        assert len(encoded_columns)==real.shape[1]
        base.update(forest=forest,reference_leaves=leaves,urf_oob_accuracy=float(forest.oob_score_),
            encoded_columns=encoded_columns,encoded_feature_count=int(real.shape[1]),
            effective_mtry=int(forest.estimators_[0].max_features_),mtry_basis='post_encoding_columns')
        distance=proximity_distance(leaves,leaves)
    elif method=='gower_kmedoids':distance=gower(num,cat,num,cat)
    elif method=='kprototypes':
        if not num.shape[1] or not cat.shape[1]:raise ValueError('K-Prototypes needs both numeric and categorical fields')
        distance=None
    else:raise ValueError(method)
    base.update(reference_numeric=num,reference_categorical=cat)
    return base,distance

def fit_partition(base,distance,k,seed,settings):
    tick=time.perf_counter();bundle=dict(base);bundle['k']=int(k);method=base['method']
    if method in {'urf_kmedoids','gower_kmedoids'}:
        found=pam(distance,k,seed,settings['starts'],settings['pam_max_iter'])
        chosen=found['medoid_indices'];bundle.update(medoid_indices=chosen,native_objective=found['cost'],converged=found['converged'])
        bundle['medoid_numeric']=base['reference_numeric'][chosen];bundle['medoid_categorical']=base['reference_categorical'][chosen]
        if method=='urf_kmedoids':bundle['medoid_leaves']=base['reference_leaves'][chosen]
        bundle['representative_branch_ids']=[base['reference_branch_ids'][i] for i in chosen]
    else:
        num,cat=base['reference_numeric'],base['reference_categorical'];matrix=np.column_stack([num.astype(object),cat])
        model=KPrototypes(n_clusters=k,init='Huang',n_init=settings['starts'],max_iter=settings['kprototypes_max_iter'],
           gamma=settings['gamma'],random_state=seed,n_jobs=1,verbose=0)
        labels=model.fit_predict(matrix,categorical=list(range(num.shape[1],matrix.shape[1])))
        bundle.update(prototype_model=model,native_objective=float(model.cost_),converged=bool(model.n_iter_<settings['kprototypes_max_iter']))
        representatives=[]
        for cluster in range(k):
            ids=np.flatnonzero(labels==cluster)
            if not len(ids):representatives.append(None);continue
            # Actual branch representative, not the synthetic numeric/categorical prototype.
            center_num=model.cluster_centroids_[cluster,:num.shape[1]].astype(float)
            center_cat=model.cluster_centroids_[cluster,num.shape[1]:].astype(str)
            costs=((num[ids]-center_num)**2).sum(axis=1)+settings['gamma']*(cat[ids]!=center_cat).sum(axis=1)
            representatives.append(base['reference_branch_ids'][int(ids[int(np.argmin(costs))])])
        bundle['representative_branch_ids']=representatives
    bundle['partition_fit_seconds']=time.perf_counter()-tick
    return bundle

def predict(bundle,frame):
    num,cat=normalize(frame,bundle['geometry'])
    if bundle['method']=='urf_kmedoids':
        leaves=bundle['forest'].apply(onehot(num,cat,bundle['geometry']))
        return proximity_distance(leaves,bundle['medoid_leaves']).argmin(axis=1)
    if bundle['method']=='gower_kmedoids':return gower(num,cat,bundle['medoid_numeric'],bundle['medoid_categorical']).argmin(axis=1)
    matrix=np.column_stack([num.astype(object),cat])
    return bundle['prototype_model'].predict(matrix,categorical=list(range(num.shape[1],matrix.shape[1])))
