from __future__ import annotations
import math, csv
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from skimage.morphology import disk

COVERAGES=[0.5,0.6,0.7,0.8,0.9]

def overlap_pairs(gt,pred):
    gt=np.asarray(gt); pred=np.asarray(pred)
    gi=np.unique(gt); gi=gi[gi>0]
    pi=np.unique(pred); pi=pi[pi>0]
    if not len(gi) or not len(pi):
        return [],len(gi),len(pi)
    mask=(gt>0)&(pred>0)
    gg=gt[mask];pp=pred[mask]
    gidx=np.searchsorted(gi,gg);pidx=np.searchsorted(pi,pp)
    inter=np.bincount(gidx*len(pi)+pidx,minlength=len(gi)*len(pi)).reshape(len(gi),len(pi))
    ga=np.bincount(gt.ravel().astype(np.int64),minlength=int(gi.max())+1)[gi]
    pa=np.bincount(pred.ravel().astype(np.int64),minlength=int(pi.max())+1)[pi]
    union=ga[:,None]+pa[None,:]-inter
    iou=np.divide(inter,union,out=np.zeros_like(inter,dtype=float),where=union>0)
    rr,cc=linear_sum_assignment(-iou)
    return [(int(gi[r]),int(pi[c]),float(iou[r,c])) for r,c in zip(rr,cc) if inter[r,c]>0],len(gi),len(pi)

def sape(a,b,eps=1e-8):
    a=float(a);b=float(b)
    return 2*abs(a-b)/(abs(a)+abs(b)+eps)

def span(a,b,eps=1e-8):
    a=float(a);b=float(b)
    return 2*abs(a-b)/(abs(a)+abs(b)+eps)

def expand_slice(sl,shape,pad):
    return (slice(max(0,sl[0].start-pad),min(shape[0],sl[0].stop+pad)),
            slice(max(0,sl[1].start-pad),min(shape[1],sl[1].stop+pad)))

def cell_rows(pred,gt,img,mpp,case,organ,radius_um=1.0):
    pred=np.asarray(pred,np.int32);gt=np.asarray(gt,np.int32);img=np.asarray(img,float)
    pairs,_,_=overlap_pairs(gt,pred)
    match={p:(g,iou) for g,p,iou in pairs if iou>=0.5}
    if not match:return []
    maxg=int(gt.max());maxp=int(pred.max())
    ga=np.bincount(gt.ravel().astype(np.int64),minlength=maxg+1)
    pa=np.bincount(pred.ravel().astype(np.int64),minlength=maxp+1)
    gs=np.bincount(gt.ravel().astype(np.int64),weights=img.ravel(),minlength=maxg+1)
    ps=np.bincount(pred.ravel().astype(np.int64),weights=img.ravel(),minlength=maxp+1)
    slices=ndi.find_objects(pred)
    rpx=max(1,int(round(radius_um/float(mpp))))
    foot=disk(rpx).astype(bool)
    out=[]
    for p,(g,iou) in match.items():
        if p-1>=len(slices) or slices[p-1] is None:continue
        sl=expand_slice(slices[p-1],pred.shape,rpx+1)
        loc=pred[sl]==p
        inner=ndi.binary_erosion(loc,structure=foot,border_value=0)
        outer=ndi.binary_dilation(loc,structure=foot,border_value=0)
        vals=img[sl]
        ai=float(inner.sum());ao=float(outer.sum())
        a0=float(pa[p]);ag=float(ga[g])
        if min(ai,ao,a0,ag)<=0:continue
        mi=float(vals[inner].mean()) if inner.any() else np.nan
        mo=float(vals[outer].mean()) if outer.any() else np.nan
        m0=float(ps[p]/a0); mg=float(gs[g]/ag)
        out.append(dict(
            case=case,organ=organ,gt_label=int(g),pred_label=int(p),matched_iou=float(iou),
            sms_mean=span(mi,mo),
            geometry_span=span(ai,ao),
            inverse_sqrt_area=1.0/math.sqrt(max(a0,1.0)),
            actual_mean_error=sape(m0,mg),
            raw_mean=m0,gt_mean=mg,inner_mean=mi,outer_mean=mo,
            raw_area=a0,gt_area=ag,radius_um=float(radius_um),radius_px=int(rpx),mpp=float(mpp)
        ))
    return out

def evaluate_group(rr,score):
    x=np.asarray([r[score] for r in rr],float)
    y=np.asarray([r["actual_mean_error"] for r in rr],float)
    k=np.isfinite(x)&np.isfinite(y);x=x[k];y=y[k]
    if len(y)<10 or np.std(x)==0:return None
    order=np.argsort(x,kind="mergesort");base=float(y.mean())
    d={"n_cells":int(len(y)),"spearman":float(spearmanr(x,y).statistic),"overall_risk":base}
    for c in COVERAGES:
        n=max(1,int(np.floor(c*len(y))))
        risk=float(y[order[:n]].mean())
        d[f"risk_{int(c*100)}"]=risk
        d[f"risk_ratio_{int(c*100)}"]=risk/base if base>0 else np.nan
    q=max(1,len(y)//4)
    lo=float(y[order[:q]].mean());hi=float(y[order[-q:]].mean())
    d["stable_q25_risk"]=lo;d["unstable_q25_risk"]=hi
    d["stable_to_unstable_ratio"]=lo/hi if hi>0 else np.nan
    return d

def case_metrics(cells):
    groups={}
    for r in cells:groups.setdefault(r["case"],[]).append(r)
    rows=[]
    for case,rr in groups.items():
        z={"case":case,"organ":rr[0]["organ"],"n_matched_cells":len(rr)}
        for score in ["sms_mean","geometry_span","inverse_sqrt_area"]:
            e=evaluate_group(rr,score)
            if e:
                for k,v in e.items():z[f"{score}__{k}"]=v
        rows.append(z)
    return rows

def aggregate(rows,score):
    def mf(k):
        a=np.asarray([r.get(f"{score}__{k}",np.nan) for r in rows],float)
        return float(np.nanmean(a)) if np.isfinite(a).any() else np.nan
    return {
        "n_cases":len(rows),"n_cells":int(sum(r["n_matched_cells"] for r in rows)),
        "mean_spearman":mf("spearman"),
        "risk_ratio_50":mf("risk_ratio_50"),"risk_ratio_60":mf("risk_ratio_60"),
        "risk_ratio_70":mf("risk_ratio_70"),"risk_ratio_80":mf("risk_ratio_80"),
        "risk_ratio_90":mf("risk_ratio_90"),
        "stable_to_unstable_ratio":mf("stable_to_unstable_ratio")
    }

def bootstrap(rows,score="sms_mean",B=5000,seed=20261102):
    rng=np.random.default_rng(seed);n=len(rows);v=[]
    for _ in range(B):
        idx=rng.integers(0,n,n)
        rr=np.asarray([rows[i].get(f"{score}__risk_ratio_70",np.nan) for i in idx],float)
        sp=np.asarray([rows[i].get(f"{score}__spearman",np.nan) for i in idx],float)
        v.append((np.nanmean(rr),np.nanmean(sp)))
    a=np.asarray(v,float)
    return {
      "risk_ratio_70_95ci":[float(np.nanpercentile(a[:,0],2.5)),float(np.nanpercentile(a[:,0],97.5))],
      "spearman_95ci":[float(np.nanpercentile(a[:,1],2.5)),float(np.nanpercentile(a[:,1],97.5))]
    }

def write_csv(path,rows):
    if not rows:
        open(path,"w").close();return
    fields=sorted({k for r in rows for k in r})
    with open(path,"w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
