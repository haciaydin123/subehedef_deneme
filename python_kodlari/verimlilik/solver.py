"""Two-stage output-oriented CCR/CRS DEA and a BCC/VRS sensitivity score."""
import numpy as np
from scipy.optimize import linprog


def minimum_units(inputs,outputs):
    """Declared discrimination heuristic, not a statistical adequacy theorem."""
    return max(inputs*outputs,3*(inputs+outputs))


def validate(X,Y):
    X=np.asarray(X,dtype=float);Y=np.asarray(Y,dtype=float)
    if X.ndim!=2 or Y.ndim!=2 or len(X)!=len(Y) or not len(X) or not X.shape[1] or not Y.shape[1]:
        raise ValueError('Nonempty, aligned input/output matrices required')
    if not np.isfinite(X).all() or not np.isfinite(Y).all():raise ValueError('Missing/nonfinite DEA data; no implicit imputation')
    if (X<=0).any():raise ValueError('This pilot requires strictly positive inputs')
    if (Y<0).any() or (Y.sum(axis=1)<=0).any():raise ValueError('Outputs must be nonnegative with at least one positive output per branch')
    if (Y.sum(axis=0)<=0).any():raise ValueError('An output is zero for the entire peer set; revise the specification')
    return X,Y


def scales(A):
    return np.array([np.median(col[col>0]) for col in A.T])


def radial(X,Y,index,returns='CRS',method='highs'):
    """max phi : X' lambda <= x0, Y' lambda >= phi*y0, lambda>=0."""
    n=len(X);m=X.shape[1];s=Y.shape[1]
    A=np.zeros((m+s,n+1));A[:m,:n]=X.T;A[m:,:n]=-Y.T;A[m:,-1]=Y[index]
    b=np.r_[X[index],np.zeros(s)];c=np.r_[np.zeros(n),-1.]
    eq=np.r_[np.ones(n),0.][None,:] if returns=='VRS' else None
    if returns not in ['CRS','VRS']:raise ValueError('Unknown returns to scale')
    result=linprog(c,A_ub=A,b_ub=b,A_eq=eq,b_eq=[1.] if eq is not None else None,
                   bounds=[(0,None)]*(n+1),method=method,
                   options={'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8})
    if not result.success:raise RuntimeError(f'{returns} DEA failed ({result.status}): {result.message}')
    phi=float(result.x[-1]);weights=result.x[:-1]
    # Dual certificate: multipliers u>=0,v>=0, u*y0=1 and Y*u <= X*v.
    # In-sample self-inclusion guarantees phi >= 1 without an active artificial bound.
    v=-result.ineqlin.marginals[:m];u=-result.ineqlin.marginals[m:]
    return {'phi':phi,'lambda':weights,'dual_v':v,'dual_u':u,'solver_status':int(result.status)}


def solve_peer_set(X,Y,tolerance=1e-7):
    X,Y=validate(X,Y);xs,ys=scales(X),scales(Y);xn,yn=X/xs,Y/ys;n,m,s=len(X),X.shape[1],Y.shape[1]
    results=[]
    # Phase II is lexicographic: fix the radial optimum, then maximize normalized slacks.
    Aeq=np.zeros((m+s,n+m+s));Aeq[:m,:n]=xn.T;Aeq[:m,n:n+m]=np.eye(m)
    Aeq[m:,:n]=yn.T;Aeq[m:,n+m:]=-np.eye(s)
    objective=np.r_[np.zeros(n),-np.ones(m+s)]
    for i in range(n):
        crs=radial(xn,yn,i);vrs=radial(xn,yn,i,returns='VRS');phi=crs['phi']
        phase2=linprog(objective,A_eq=Aeq,b_eq=np.r_[xn[i],phi*yn[i]],bounds=(0,None),method='highs',
                       options={'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8})
        if not phase2.success:raise RuntimeError(f'Phase II failed ({phase2.status}): {phase2.message}')
        weights=phase2.x[:n];input_slack=phase2.x[n:n+m]*xs;output_slack=phase2.x[n+m:]*ys
        projection=weights@Y;input_projection=weights@X;score=1/phi;vrs_score=1/vrs['phi']
        slack_sum=float(phase2.x[n:].sum());radial_efficient=abs(phi-1)<=tolerance
        status='strongly_efficient' if radial_efficient and slack_sum<=tolerance else 'radially_efficient_with_slack' if radial_efficient else 'inefficient'
        if score>vrs_score+tolerance:raise RuntimeError('CRS efficiency exceeds VRS efficiency')
        results.append({'phi':phi,'efficiency':score,'vrs_efficiency':vrs_score,'vrs_phi':vrs['phi'],'scale_efficiency':score/vrs_score,
                        'radial_growth':phi-1,'status':status,'normalized_slack_sum':slack_sum,
                        'lambda':weights.tolist(),'radial_lambda':crs['lambda'].tolist(),
                        'dual_v':crs['dual_v'].tolist(),'dual_u':crs['dual_u'].tolist(),
                        'input_scales':xs.tolist(),'output_scales':ys.tolist(),
                        'input_slack':input_slack.tolist(),'output_slack':output_slack.tolist(),
                        'input_projection':input_projection.tolist(),'output_projection':projection.tolist(),
                        'radial_projection':(phi*Y[i]).tolist(),'solver_status':crs['solver_status']})
    return results
