from __future__ import annotations
from pathlib import Path
import csv, math, numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr, rankdata
from skimage.morphology import disk

MPP=0.16
RADIUS_UM=1.0
COVERAGES=np.arange(0.10,1.001,0.05)

def read_tif(path):
    import tifffile
    return np.asarray(tifffile.imread(path))

def to_hwc(a):
    a=np.asarray(a)
    if a.ndim==2: return a[...,None]
    if a.ndim==3 and a.shape[-1]<=4: return a
    if a.ndim==3 and a.shape[0]<=4: return np.moveaxis(a,0,-1)
    raise ValueError(f"Unsupported TIFF shape {a.shape}")

def unseg_channels(a):
    x=to_hwc(a)
    if x.shape[-1] < 3: raise ValueError(f"Expected >=3 channels, got {x.shape}")
    return np.asarray(x[...,0],np.float32), np.asarray(x[...,2],np.float32)

def load_label_txt(path):
    last=None
    for delim in (None,",","\t"):
        try:
            x=np.loadtxt(path,delimiter=delim)
            if x.size: return np.rint(x).astype(np.int32)
        except Exception as e: last=e
    raise RuntimeError(f"Cannot parse {path}: {last}")

def overlap_pairs(gt,pred):
    gt=np.asarray(gt); pred=np.asarray(pred)
    gi=np.unique(gt); gi=gi[gi>0]
    pi=np.unique(pred); pi=pi[pi>0]
    if not len(gi) or not len(pi): return [],len(gi),len(pi)
    m=(gt>0)&(pred>0); gg=gt[m]; pp=pred[m]
    gidx=np.searchsorted(gi,gg); pidx=np.searchsorted(pi,pp)
    inter=np.bincount(gidx*len(pi)+pidx,minlength=len(gi)*len(pi)).reshape(len(gi),len(pi))
    ga=np.bincount(gt.ravel().astype(np.int64),minlength=int(gi.max())+1)[gi]
    pa=np.bincount(pred.ravel().astype(np.int64),minlength=int(pi.max())+1)[pi]
    union=ga[:,None]+pa[None,:]-inter
    iou=np.divide(inter,union,out=np.zeros_like(inter,dtype=float),where=union>0)
    rr,cc=linear_sum_assignment(-iou)
    return [(int(gi[r]),int(pi[c]),float(iou[r,c])) for r,c in zip(rr,cc) if inter[r,c]>0],len(gi),len(pi)

def sape(a,b,eps=1e-8):
    return 2*abs(float(a)-float(b))/(abs(float(a))+abs(float(b))+eps)

def span(a,b,eps=1e-8):
    return 2*abs(float(a)-float(b))/(abs(float(a))+abs(float(b))+eps)

def expand_slice(sl,shape,pad):
    return (slice(max(0,sl[0].start-pad),min(shape[0],sl[0].stop+pad)),
            slice(max(0,sl[1].start-pad),min(shape[1],sl[1].stop+pad)))

def cell_rows(pred,gt,img,case,mpp=MPP,radius_um=RADIUS_UM):
    pairs,_,_=overlap_pairs(gt,pred)
    match={p:(g,iou) for g,p,iou in pairs if iou>=0.5}
    if not match: return []
    maxg=int(gt.max()); maxp=int(pred.max())
    ga=np.bincount(gt.ravel().astype(np.int64),minlength=maxg+1)
    pa=np.bincount(pred.ravel().astype(np.int64),minlength=maxp+1)
    gs=np.bincount(gt.ravel().astype(np.int64),weights=img.ravel(),minlength=maxg+1)
    ps=np.bincount(pred.ravel().astype(np.int64),weights=img.ravel(),minlength=maxp+1)
    slices=ndi.find_objects(pred); rpx=max(1,int(round(radius_um/float(mpp))))
    foot=disk(rpx).astype(bool); out=[]
    for p,(g,iou) in match.items():
        if p-1>=len(slices) or slices[p-1] is None: continue
        sl=expand_slice(slices[p-1],pred.shape,rpx+1); loc=pred[sl]==p
        inner=ndi.binary_erosion(loc,structure=foot,border_value=0)
        outer=ndi.binary_dilation(loc,structure=foot,border_value=0)
        vals=img[sl]; ai=float(inner.sum()); ao=float(outer.sum())
        a0=float(pa[p]); ag=float(ga[g])
        if min(ai,ao,a0,ag)<=0: continue
        mi=float(vals[inner].mean()); mo=float(vals[outer].mean())
        m0=float(ps[p]/a0); mg=float(gs[g]/ag)
        out.append(dict(case=case,gt_label=int(g),pred_label=int(p),matched_iou=float(iou),
                        sms_mean=span(mi,mo),geometry_span=span(ai,ao),
                        inverse_sqrt_area=1.0/math.sqrt(max(a0,1.0)),
                        actual_mean_error=sape(m0,mg),raw_mean=m0,gt_mean=mg,
                        inner_mean=mi,outer_mean=mo,raw_area=a0,gt_area=ag,
                        radius_um=float(radius_um),radius_px=int(rpx),mpp=float(mpp)))
    return out

def eval_group(rr,score):
    x=np.asarray([r[score] for r in rr],float); y=np.asarray([r["actual_mean_error"] for r in rr],float)
    keep=np.isfinite(x)&np.isfinite(y); x=x[keep]; y=y[keep]
    if len(y)<10 or np.std(x)==0: return None
    order=np.argsort(x,kind="mergesort"); base=float(y.mean())
    d={"n_cells":int(len(y)),"spearman":float(spearmanr(x,y).statistic),"overall_risk":base}
    for c in COVERAGES:
        k=max(1,int(np.floor(c*len(y)))); risk=float(y[order[:k]].mean())
        d[f"risk_ratio_{int(round(c*100))}"]=risk/base if base>0 else np.nan
    return d

def case_metrics(cells):
    groups={}
    for r in cells: groups.setdefault(r["case"],[]).append(r)
    out=[]
    for case,rr in groups.items():
        z={"case":case,"n_matched_cells":len(rr)}
        for score in ["sms_mean","geometry_span","inverse_sqrt_area"]:
            e=eval_group(rr,score)
            if e:
                for k,v in e.items(): z[f"{score}__{k}"]=v
        out.append(z)
    return out

def aggregate(rows,score):
    def mf(k):
        a=np.asarray([r.get(f"{score}__{k}",np.nan) for r in rows],float)
        return float(np.nanmean(a)) if np.isfinite(a).any() else np.nan
    out={"n_evaluable_cases":int(sum(np.isfinite(r.get(f"{score}__spearman",np.nan)) for r in rows)),
         "n_cells":int(sum(r.get("n_matched_cells",0) for r in rows)),
         "mean_spearman":mf("spearman")}
    for c in COVERAGES: out[f"risk_ratio_{int(round(c*100))}"]=mf(f"risk_ratio_{int(round(c*100))}")
    return out

def bootstrap(rows,score="sms_mean",B=5000,seed=20261104):
    valid=[r for r in rows if np.isfinite(r.get(f"{score}__risk_ratio_70",np.nan))]
    if len(valid)<2: return {"risk_ratio_70_95ci":[np.nan,np.nan],"spearman_95ci":[np.nan,np.nan]}
    rng=np.random.default_rng(seed); n=len(valid); v=[]
    for _ in range(B):
        idx=rng.integers(0,n,n)
        rr=np.asarray([valid[i][f"{score}__risk_ratio_70"] for i in idx],float)
        sp=np.asarray([valid[i][f"{score}__spearman"] for i in idx],float)
        v.append((np.mean(rr),np.mean(sp)))
    a=np.asarray(v,float)
    return {"risk_ratio_70_95ci":[float(np.percentile(a[:,0],2.5)),float(np.percentile(a[:,0],97.5))],
            "spearman_95ci":[float(np.percentile(a[:,1],2.5)),float(np.percentile(a[:,1],97.5))]}

def partial_rank(x,y,controls):
    cols=[np.asarray(x,float),np.asarray(y,float)]+[np.asarray(c,float) for c in controls]
    keep=np.ones(len(cols[0]),bool)
    for a in cols: keep &= np.isfinite(a)
    cols=[a[keep] for a in cols]
    if len(cols[0])<20:return np.nan
    rx=rankdata(cols[0]); ry=rankdata(cols[1])
    C=np.column_stack([rankdata(c) for c in cols[2:]])
    C=np.column_stack([np.ones(len(C)),C])
    bx=np.linalg.lstsq(C,rx,rcond=None)[0]; by=np.linalg.lstsq(C,ry,rcond=None)[0]
    ex=rx-C@bx; ey=ry-C@by
    return float(np.corrcoef(ex,ey)[0,1]) if ex.std()>0 and ey.std()>0 else np.nan

def geometry_strata(cells):
    rows=[]; groups={}
    for r in cells: groups.setdefault(r["case"],[]).append(r)
    for case,rr in groups.items():
        if len(rr)<30: continue
        g=np.asarray([r["geometry_span"] for r in rr]); order=np.argsort(g,kind="mergesort")
        for label,idx in zip(["low","middle","high"],np.array_split(order,3)):
            sub=[rr[i] for i in idx]; e=eval_group(sub,"sms_mean")
            if e: rows.append({"case":case,"stratum":label,**e})
    summary={}
    for label in ["low","middle","high"]:
        rr=[r for r in rows if r["stratum"]==label]
        summary[label]={"n_cases":len(rr),
                        "risk_ratio_70":float(np.mean([r["risk_ratio_70"] for r in rr])) if rr else np.nan,
                        "spearman":float(np.mean([r["spearman"] for r in rr])) if rr else np.nan}
    return summary,rows

def write_csv(path,rows):
    path=Path(path)
    if not rows: path.write_text("",encoding="utf-8"); return
    fields=sorted({k for r in rows for k in r})
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
