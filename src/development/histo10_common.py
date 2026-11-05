from __future__ import annotations
from pathlib import Path
import json, math, csv
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from skimage.morphology import disk

PRIMARY_RADIUS_UM = 1.0
COVERAGES = [0.5,0.6,0.7,0.8,0.9]

def overlap_pairs(gt,pred):
    gt=np.asarray(gt); pred=np.asarray(pred)
    gi=np.unique(gt); gi=gi[gi>0]
    pi=np.unique(pred); pi=pi[pi>0]
    if not len(gi) or not len(pi):
        return [],len(gi),len(pi)
    mask=(gt>0)&(pred>0)
    gg=gt[mask]; pp=pred[mask]
    gidx=np.searchsorted(gi,gg); pidx=np.searchsorted(pi,pp)
    flat=gidx*len(pi)+pidx
    inter=np.bincount(flat,minlength=len(gi)*len(pi)).reshape(len(gi),len(pi))
    ga=np.bincount(gt.ravel().astype(np.int64),minlength=int(gi.max())+1)[gi]
    pa=np.bincount(pred.ravel().astype(np.int64),minlength=int(pi.max())+1)[pi]
    union=ga[:,None]+pa[None,:]-inter
    iou=np.divide(inter,union,out=np.zeros_like(inter,dtype=float),where=union>0)
    rr,cc=linear_sum_assignment(-iou)
    return [(int(gi[r]),int(pi[c]),float(iou[r,c])) for r,c in zip(rr,cc) if inter[r,c]>0],len(gi),len(pi)

def sape(a,b,eps=1e-8):
    a=float(a);b=float(b)
    return 2.0*abs(a-b)/(abs(a)+abs(b)+eps)

def span_score(lo,hi,eps=1e-8):
    lo=float(lo);hi=float(hi)
    return 2.0*abs(hi-lo)/(abs(hi)+abs(lo)+eps)

def expand_slice(sl,shape,pad):
    y0=max(0,sl[0].start-pad);y1=min(shape[0],sl[0].stop+pad)
    x0=max(0,sl[1].start-pad);x1=min(shape[1],sl[1].stop+pad)
    return (slice(y0,y1),slice(x0,x1))

def cell_uncertainty_rows(pred,gt,img,mpp,case,group,organ=None,iou_min=0.5,radius_um=PRIMARY_RADIUS_UM):
    pred=np.asarray(pred,np.int32);gt=np.asarray(gt,np.int32);img=np.asarray(img,float)
    pairs,ng,npred=overlap_pairs(gt,pred)
    match={p:(g,iou) for g,p,iou in pairs if iou>=iou_min}
    if not match:
        return []
    maxg=int(gt.max()); maxp=int(pred.max())
    gt_area=np.bincount(gt.ravel().astype(np.int64),minlength=maxg+1)
    pr_area=np.bincount(pred.ravel().astype(np.int64),minlength=maxp+1)
    gt_sum=np.bincount(gt.ravel().astype(np.int64),weights=img.ravel(),minlength=maxg+1)
    pr_sum=np.bincount(pred.ravel().astype(np.int64),weights=img.ravel(),minlength=maxp+1)
    slices=ndi.find_objects(pred)
    rpx=max(1,int(round(radius_um/float(mpp))))
    foot=disk(rpx).astype(bool)
    rows=[]
    for p,(g,iou) in match.items():
        if p-1>=len(slices) or slices[p-1] is None:
            continue
        sl=expand_slice(slices[p-1],pred.shape,rpx+1)
        local=(pred[sl]==p)
        if not local.any(): continue
        inner=ndi.binary_erosion(local,structure=foot,border_value=0)
        outer=ndi.binary_dilation(local,structure=foot,border_value=0)
        intens=img[sl]
        a0=float(pr_area[p]); ag=float(gt_area[g])
        s0=float(pr_sum[p]); sg=float(gt_sum[g])
        ai=float(inner.sum()); ao=float(outer.sum())
        si=float(intens[inner].sum()) if inner.any() else 0.0
        so=float(intens[outer].sum()) if outer.any() else 0.0
        m0=s0/a0 if a0>0 else np.nan
        mg=sg/ag if ag>0 else np.nan
        mi=si/ai if ai>0 else np.nan
        mo=so/ao if ao>0 else np.nan
        rows.append(dict(
            case=case,group=group,organ=(organ or ""),gt_label=int(g),pred_label=int(p),
            matched_iou=float(iou),mpp=float(mpp),radius_um=float(radius_um),radius_px=int(rpx),
            raw_area=a0,gt_area=ag,inner_area=ai,outer_area=ao,
            raw_signal=s0,gt_signal=sg,inner_signal=si,outer_signal=so,
            raw_mean=float(m0),gt_mean=float(mg),inner_mean=float(mi),outer_mean=float(mo),
            actual_sum_error=sape(s0,sg),
            actual_mean_error=sape(m0,mg) if np.isfinite(m0) and np.isfinite(mg) else np.nan,
            sms_sum=span_score(si,so),
            sms_mean=span_score(mi,mo) if np.isfinite(mi) and np.isfinite(mo) else np.nan,
            geometry_span=span_score(ai,ao),
            inverse_sqrt_area=1.0/math.sqrt(max(a0,1.0)),
        ))
    return rows

def per_case_eval(rows,score_key,error_key):
    if len(rows)<10:
        return None
    score=np.asarray([r[score_key] for r in rows],float)
    err=np.asarray([r[error_key] for r in rows],float)
    keep=np.isfinite(score)&np.isfinite(err)
    score=score[keep];err=err[keep]
    if len(err)<10 or np.std(score)==0:
        return None
    sp=float(spearmanr(score,err).statistic)
    base=float(np.mean(err))
    out=dict(n_cells=int(len(err)),spearman=sp,overall_risk=base)
    order=np.argsort(score,kind="mergesort")
    for c in COVERAGES:
        k=max(1,int(math.floor(c*len(err))))
        risk=float(np.mean(err[order[:k]]))
        out[f"risk_{int(c*100)}"]=risk
        out[f"risk_ratio_{int(c*100)}"]=risk/base if base>0 else np.nan
    q=max(1,len(err)//4)
    stable=float(np.mean(err[order[:q]]))
    unstable=float(np.mean(err[order[-q:]]))
    out["stable_q25_risk"]=stable
    out["unstable_q25_risk"]=unstable
    out["stable_to_unstable_ratio"]=stable/unstable if unstable>0 else np.nan
    return out

def summarize_case_metrics(cell_rows,group_key="case"):
    groups={}
    for r in cell_rows:
        groups.setdefault(r[group_key],[]).append(r)
    case_rows=[]
    for g,rr in groups.items():
        base={"case":g,"group":rr[0].get("group",""),"organ":rr[0].get("organ",""),"n_matched_cells":len(rr)}
        for score in ["sms_sum","geometry_span","inverse_sqrt_area","sms_mean"]:
            ev=per_case_eval(rr,score,"actual_sum_error" if score!="sms_mean" else "actual_mean_error")
            if ev:
                for k,v in ev.items():
                    base[f"{score}__{k}"]=v
        case_rows.append(base)
    return case_rows

def aggregate_case_metrics(case_rows,prefix="sms_sum"):
    def mf(key):
        a=np.asarray([r.get(f"{prefix}__{key}",np.nan) for r in case_rows],float)
        return float(np.nanmean(a)) if np.isfinite(a).any() else np.nan
    return dict(
        n_cases=len(case_rows),
        n_cells=int(sum(r.get("n_matched_cells",0) for r in case_rows)),
        mean_spearman=mf("spearman"),
        mean_risk_ratio_80=mf("risk_ratio_80"),
        mean_stable_to_unstable_ratio=mf("stable_to_unstable_ratio"),
        mean_risk_ratio_50=mf("risk_ratio_50"),
        mean_risk_ratio_90=mf("risk_ratio_90"),
    )

def aggregate_by_parent(case_rows):
    parents={}
    for r in case_rows:
        parents.setdefault(r.get("group",""),[]).append(r)
    parent_rows=[]
    keys=set()
    for r in case_rows: keys.update(r.keys())
    metric_keys=[k for k in keys if "__" in k]
    for p,rr in parents.items():
        pr={"parent":p,"n_tiles":len(rr),"n_matched_cells":sum(x.get("n_matched_cells",0) for x in rr)}
        for k in metric_keys:
            a=np.asarray([x.get(k,np.nan) for x in rr],float)
            pr[k]=float(np.nanmean(a)) if np.isfinite(a).any() else np.nan
        parent_rows.append(pr)
    return parent_rows

def aggregate_parent_metrics(parent_rows,prefix="sms_sum"):
    def mf(key):
        a=np.asarray([r.get(f"{prefix}__{key}",np.nan) for r in parent_rows],float)
        return float(np.nanmean(a)) if np.isfinite(a).any() else np.nan
    return dict(
        n_parents=len(parent_rows),
        n_cells=int(sum(r.get("n_matched_cells",0) for r in parent_rows)),
        mean_spearman=mf("spearman"),
        mean_risk_ratio_80=mf("risk_ratio_80"),
        mean_stable_to_unstable_ratio=mf("stable_to_unstable_ratio"),
    )

def bootstrap_case_risk(case_rows,prefix="sms_sum",B=2000,seed=20261102):
    rng=np.random.default_rng(seed)
    vals=[]
    n=len(case_rows)
    for _ in range(B):
        ss=rng.integers(0,n,n)
        rr=np.asarray([case_rows[i].get(f"{prefix}__risk_ratio_80",np.nan) for i in ss],float)
        sp=np.asarray([case_rows[i].get(f"{prefix}__spearman",np.nan) for i in ss],float)
        vals.append((np.nanmean(rr),np.nanmean(sp)))
    a=np.asarray(vals,float)
    return {
        "risk_ratio_80_95ci":[float(np.nanpercentile(a[:,0],2.5)),float(np.nanpercentile(a[:,0],97.5))],
        "spearman_95ci":[float(np.nanpercentile(a[:,1],2.5)),float(np.nanpercentile(a[:,1],97.5))]
    }

def bootstrap_parent_risk(parent_rows,prefix="sms_sum",B=2000,seed=20261102):
    rng=np.random.default_rng(seed)
    vals=[];n=len(parent_rows)
    for _ in range(B):
        ss=rng.integers(0,n,n)
        rr=np.asarray([parent_rows[i].get(f"{prefix}__risk_ratio_80",np.nan) for i in ss],float)
        sp=np.asarray([parent_rows[i].get(f"{prefix}__spearman",np.nan) for i in ss],float)
        vals.append((np.nanmean(rr),np.nanmean(sp)))
    a=np.asarray(vals,float)
    return {
        "risk_ratio_80_95ci":[float(np.nanpercentile(a[:,0],2.5)),float(np.nanpercentile(a[:,0],97.5))],
        "spearman_95ci":[float(np.nanpercentile(a[:,1],2.5)),float(np.nanpercentile(a[:,1],97.5))]
    }

def write_csv(path,rows):
    path=Path(path)
    if not rows:
        path.write_text("",encoding="utf-8");return
    fields=sorted({k for r in rows for k in r})
    with path.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
