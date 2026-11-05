
from pathlib import Path
import argparse, json, itertools, math
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
import matplotlib as mpl
import matplotlib.pyplot as plt

RADII = [0.5, 1.0, 1.5, 2.0]
COVERAGE = 0.70
MIN_CELLS_PER_IMAGE = 10
N_BOOT = 5000
SEED = 20261105
ID_COLS = ["case", "gt_label", "pred_label"]

def ci(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return [np.nan, np.nan]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

def bootstrap_mean(v, rng):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return [np.nan, np.nan]
    n = len(v)
    out = np.empty(N_BOOT, float)
    for i in range(N_BOOT):
        out[i] = np.mean(v[rng.integers(0, n, n)])
    return ci(out)

def jaccard(a, b):
    a, b = set(a), set(b)
    u = len(a | b)
    return float(len(a & b) / u) if u else np.nan

def build_decisions(df):
    out, cuts = [], []
    for (case, radius), g in df.groupby(["case", "radius_um"], sort=True):
        g = g.sort_values(["sms_mean", "gt_label", "pred_label"], kind="mergesort").copy()
        n = len(g)
        k = max(1, int(np.floor(COVERAGE * n)))
        g["retained70"] = False
        g.iloc[:k, g.columns.get_loc("retained70")] = True
        g["sms_rank_percentile"] = (
            rankdata(g["sms_mean"].to_numpy(float), method="average") - 1
        ) / max(n - 1, 1)
        out.append(g)
        cuts.append({
            "case": case,
            "radius_um": float(radius),
            "n_cells": int(n),
            "n_retained": int(k),
            "retained_fraction": float(k/n),
            "max_sms_retained": float(g.loc[g.retained70, "sms_mean"].max()),
            "min_sms_rejected": float(g.loc[~g.retained70, "sms_mean"].min()) if k < n else np.nan,
        })
    return pd.concat(out, ignore_index=True), pd.DataFrame(cuts)

def pairwise_per_image(dec):
    rows = []
    for case, cg in dec.groupby("case", sort=True):
        by_r = {float(r): g.set_index(["gt_label","pred_label"]).sort_index()
                for r, g in cg.groupby("radius_um", sort=True)}
        for r1, r2 in itertools.combinations(RADII, 2):
            g1, g2 = by_r[r1], by_r[r2]
            if not g1.index.equals(g2.index):
                raise RuntimeError(f"{case}: identity mismatch {r1} vs {r2}")
            s1, s2 = g1.sms_mean.to_numpy(float), g2.sms_mean.to_numpy(float)
            rho = float(spearmanr(s1,s2).statistic) if len(s1)>=3 and np.std(s1)>0 and np.std(s2)>0 else np.nan
            d1, d2 = g1.retained70.to_numpy(bool), g2.retained70.to_numpy(bool)
            ids = list(g1.index)
            ret1 = [ids[i] for i in np.where(d1)[0]]
            ret2 = [ids[i] for i in np.where(d2)[0]]
            rej1 = [ids[i] for i in np.where(~d1)[0]]
            rej2 = [ids[i] for i in np.where(~d2)[0]]
            shift = np.abs(
                g1.sms_rank_percentile.to_numpy(float)
                - g2.sms_rank_percentile.to_numpy(float)
            )
            rows.append({
                "case": case,
                "radius1_um": r1,
                "radius2_um": r2,
                "adjacent_pair": abs(r2-r1)==0.5,
                "n_cells": len(g1),
                "sms_rank_spearman": rho,
                "retained70_jaccard": jaccard(ret1,ret2),
                "rejected30_jaccard": jaccard(rej1,rej2),
                "decision_switch_rate": float(np.mean(d1 != d2)),
                "mean_abs_rank_percentile_shift": float(np.mean(shift)),
                "median_abs_rank_percentile_shift": float(np.median(shift)),
            })
    return pd.DataFrame(rows)

def summarize_pairwise(per_image):
    rng = np.random.default_rng(SEED)
    metrics = [
        "sms_rank_spearman","retained70_jaccard","rejected30_jaccard",
        "decision_switch_rate","mean_abs_rank_percentile_shift"
    ]
    rows = []
    for (r1,r2), g in per_image.groupby(["radius1_um","radius2_um"], sort=True):
        row = {
            "radius1_um": float(r1), "radius2_um": float(r2),
            "adjacent_pair": abs(r2-r1)==0.5,
            "n_images": int(len(g)), "n_cells_total": int(g.n_cells.sum())
        }
        for m in metrics:
            v = pd.to_numeric(g[m],errors="coerce").to_numpy(float)
            row[m] = float(np.nanmean(v))
            lo,hi = bootstrap_mean(v,rng)
            row[m+"_ci_low"], row[m+"_ci_high"] = lo,hi
        rows.append(row)
    return pd.DataFrame(rows)

def matrix_from(summary, metric, diag):
    m = pd.DataFrame(np.nan,index=RADII,columns=RADII)
    for r in RADII: m.loc[r,r]=diag
    for _,x in summary.iterrows():
        a,b=float(x.radius1_um),float(x.radius2_um)
        m.loc[a,b]=m.loc[b,a]=float(x[metric])
    m.index.name="radius_um"
    return m

def build_classes(dec):
    idx = ID_COLS
    d = dec.pivot_table(index=idx,columns="radius_um",values="retained70",aggfunc="first")
    p = dec.pivot_table(index=idx,columns="radius_um",values="sms_rank_percentile",aggfunc="first")
    er = dec.groupby(idx).actual_mean_error.agg(["min","max","first"])
    if float((er["max"]-er["min"]).abs().max()) > 1e-12:
        raise RuntimeError("actual_mean_error changed across radii.")
    meta = dec.sort_values(idx+["radius_um"]).drop_duplicates(idx)[idx+["organ","actual_mean_error"]].set_index(idx)
    out = meta.copy()
    for r in RADII:
        out[f"retained_{r:.1f}"] = d[r].astype(bool)
        out[f"rankpct_{r:.1f}"] = p[r].astype(float)
    dc = [f"retained_{r:.1f}" for r in RADII]
    out["n_radii_retained"] = out[dc].sum(axis=1).astype(int)
    out["decision_pattern"] = out[dc].apply(lambda x:"".join("R" if bool(v) else "X" for v in x),axis=1)
    out["decision_class"] = np.where(
        out.n_radii_retained.eq(4),"always_retained",
        np.where(out.n_radii_retained.eq(0),"always_rejected","radius_dependent")
    )
    arr=out[dc].to_numpy(bool)
    out["adjacent_switch_count"]=np.sum(arr[:,1:]!=arr[:,:-1],axis=1).astype(int)
    rc=[f"rankpct_{r:.1f}" for r in RADII]
    ra=out[rc].to_numpy(float)
    out["rank_percentile_range"]=np.max(ra,axis=1)-np.min(ra,axis=1)
    out["rank_percentile_sd"]=np.std(ra,axis=1)
    return out.reset_index()

def per_image_classes(classes):
    rows=[]
    order=["always_retained","radius_dependent","always_rejected"]
    for case,g in classes.groupby("case",sort=True):
        n=len(g)
        for c in order:
            h=g[g.decision_class.eq(c)]
            rows.append({
                "case":case,"decision_class":c,"n_cells":len(h),
                "proportion":len(h)/n,
                "mean_actual_error":float(h.actual_mean_error.mean()) if len(h) else np.nan,
                "median_actual_error":float(h.actual_mean_error.median()) if len(h) else np.nan,
            })
    return pd.DataFrame(rows)

def summarize_classes(classes, per_img):
    rng=np.random.default_rng(SEED+1)
    rows=[]
    total=len(classes)
    for c in ["always_retained","radius_dependent","always_rejected"]:
        h=classes[classes.decision_class.eq(c)]
        p=per_img[per_img.decision_class.eq(c)]
        props=p.proportion.to_numpy(float)
        errs=p.mean_actual_error.to_numpy(float)
        plo,phi=bootstrap_mean(props,rng)
        elo,ehi=bootstrap_mean(errs,rng)
        rows.append({
            "decision_class":c,
            "n_cells":len(h),
            "pooled_proportion":len(h)/total,
            "image_balanced_proportion":float(np.nanmean(props)),
            "proportion_ci_low":plo,"proportion_ci_high":phi,
            "n_images_with_group":int(np.isfinite(errs).sum()),
            "pooled_mean_actual_error":float(h.actual_mean_error.mean()) if len(h) else np.nan,
            "pooled_median_actual_error":float(h.actual_mean_error.median()) if len(h) else np.nan,
            "image_balanced_mean_actual_error":float(np.nanmean(errs)),
            "actual_error_ci_low":elo,"actual_error_ci_high":ehi,
            "mean_adjacent_switch_count":float(h.adjacent_switch_count.mean()) if len(h) else np.nan,
            "mean_rank_percentile_range":float(h.rank_percentile_range.mean()) if len(h) else np.nan,
        })
    return pd.DataFrame(rows)

def contrasts(per_img):
    rng=np.random.default_rng(SEED+2)
    p=per_img.pivot(index="case",columns="decision_class",values="mean_actual_error")
    pairs=[
        ("radius_dependent","always_retained"),
        ("always_rejected","always_retained"),
        ("always_rejected","radius_dependent"),
    ]
    rows=[]
    for hi,lo in pairs:
        if hi not in p.columns or lo not in p.columns: continue
        d=(p[hi]-p[lo]).dropna().to_numpy(float)
        clo,chi=bootstrap_mean(d,rng)
        rows.append({
            "contrast":f"{hi} minus {lo}",
            "n_paired_images":len(d),
            "mean_error_difference":float(np.mean(d)) if len(d) else np.nan,
            "ci_low":clo,"ci_high":chi
        })
    return pd.DataFrame(rows)

def make_figures(rank_mat, class_summary, out):
    mpl.rcParams.update({
        "svg.fonttype":"none","font.family":"sans-serif","font.size":7.5,
        "axes.labelsize":7.5,"xtick.labelsize":7.0,"ytick.labelsize":7.0
    })
    vals=rank_mat.to_numpy(float)
    fig,ax=plt.subplots(figsize=(3.5,2.0))
    fig.subplots_adjust(left=.22,right=.90,bottom=.25,top=.93)
    mesh=ax.pcolormesh(np.arange(5),np.arange(5),vals,shading="flat")
    ax.set_xlim(0,4); ax.set_ylim(4,0)
    labs=["0.5","1.0","1.5","2.0"]
    ax.set_xticks(np.arange(4)+.5,labs); ax.set_yticks(np.arange(4)+.5,labs)
    ax.set_xlabel("Perturbation radius (μm)"); ax.set_ylabel("Perturbation radius (μm)")
    ax.text(.01,1.02,"(a)",transform=ax.transAxes,fontweight="bold",va="bottom")
    for i in range(4):
        for j in range(4):
            ax.text(j+.5,i+.5,f"{vals[i,j]:.2f}",ha="center",va="center",fontsize=7)
    cb=fig.colorbar(mesh,ax=ax,fraction=.05,pad=.04); cb.set_label("Image-balanced Spearman ρ")
    fig.savefig(out/"Fig5a_rank_stability.svg"); fig.savefig(out/"Fig5a_rank_stability_600dpi.png",dpi=600)
    plt.close(fig)

    order=["always_retained","radius_dependent","always_rejected"]
    labels=["Always\nretained","Radius-\ndependent","Always\nrejected"]
    s=class_summary.set_index("decision_class").loc[order]
    y=s.image_balanced_mean_actual_error.to_numpy(float)
    lo=s.actual_error_ci_low.to_numpy(float); hi=s.actual_error_ci_high.to_numpy(float)
    n=s.n_cells.to_numpy(int)
    fig,ax=plt.subplots(figsize=(3.5,1.95))
    fig.subplots_adjust(left=.21,right=.98,bottom=.32,top=.93)
    xx=np.arange(3)
    ax.errorbar(xx,y,yerr=np.vstack([y-lo,hi-y]),marker="o",linestyle="none",capsize=2.5)
    ax.set_xticks(xx,[f"{labels[i]}\n(n={n[i]:,})" for i in range(3)])
    ax.set_ylabel("Manual-reference\nmean-intensity error")
    ax.text(.01,1.02,"(b)",transform=ax.transAxes,fontweight="bold",va="bottom")
    ax.tick_params(direction="out")
    fig.savefig(out/"Fig5b_decision_class_error.svg"); fig.savefig(out/"Fig5b_decision_class_error_600dpi.png",dpi=600)
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser(description="Experiment 3: same-cell SMS decision stability across physical radii.")
    ap.add_argument("--input",type=Path,default=Path(r"E:\Histo_Pilot\sms_backbone_transfer\physical_scale_mesmer\mesmer_physical_scale_commoncell_rows.csv"))
    ap.add_argument("--out",type=Path,default=Path(r"E:\Histo_Pilot\sms_backbone_transfer\same_cell_radius_stability"))
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    if not args.input.exists(): raise FileNotFoundError(args.input)

    df=pd.read_csv(args.input)
    req=set(ID_COLS+["organ","radius_um","sms_mean","actual_mean_error","matched_iou","raw_mean","gt_mean"])
    miss=sorted(req-set(df.columns))
    if miss: raise RuntimeError(f"Missing columns: {miss}")
    df["radius_um"]=df.radius_um.astype(float).round(6)
    if sorted(df.radius_um.unique().tolist())!=RADII: raise RuntimeError("Unexpected radii.")
    nrad=df.groupby(ID_COLS).radius_um.nunique()
    if not (nrad==4).all(): raise RuntimeError("Not every identity appears at all four radii.")
    if len(nrad)!=6437: raise RuntimeError(f"Expected 6437 common identities, got {len(nrad)}.")
    if df.duplicated(ID_COLS+["radius_um"]).any(): raise RuntimeError("Duplicate cell-radius rows.")
    er=df.groupby(ID_COLS).actual_mean_error.agg(lambda x:float(np.max(x)-np.min(x)))
    if float(er.max())>1e-12: raise RuntimeError("actual_mean_error varies across radii.")

    case_n=df[df.radius_um.eq(1.0)].groupby("case").size()
    eval_cases=sorted(case_n[case_n>=MIN_CELLS_PER_IMAGE].index.tolist())
    excluded=sorted(case_n[case_n<MIN_CELLS_PER_IMAGE].index.tolist())
    ev=df[df.case.isin(eval_cases)].copy()
    n_eval=ev[ev.radius_um.eq(1.0)][ID_COLS].drop_duplicates().shape[0]
    if len(eval_cases)!=72 or n_eval!=6423:
        raise RuntimeError(f"Expected 72 images/6423 cells, got {len(eval_cases)}/{n_eval}.")

    print("EXPERIMENT 3 — SAME-CELL DECISION STABILITY ACROSS RADIUS")
    print("="*76)
    print("Common-cell pool:",len(nrad))
    print("Primary evaluable images:",len(eval_cases))
    print("Primary evaluable cells:",n_eval)
    print("Excluded <10-cell images:",len(excluded),excluded)
    print("Frozen coverage:",COVERAGE)
    print("Radii:",RADII)
    print("No threshold or radius is reselected.\n")

    dec,cuts=build_decisions(ev)
    perpair=pairwise_per_image(dec)
    psum=summarize_pairwise(perpair)
    rankm=matrix_from(psum,"sms_rank_spearman",1.0)
    retm=matrix_from(psum,"retained70_jaccard",1.0)
    rejm=matrix_from(psum,"rejected30_jaccard",1.0)
    swm=matrix_from(psum,"decision_switch_rate",0.0)

    cls=build_classes(dec)
    percls=per_image_classes(cls)
    csum=summarize_classes(cls,percls)
    con=contrasts(percls)

    dec.to_csv(args.out/"cell_radius_decisions.csv",index=False)
    cuts.to_csv(args.out/"per_image_radius_cutoffs.csv",index=False)
    perpair.to_csv(args.out/"per_image_pairwise_stability.csv",index=False)
    psum.to_csv(args.out/"pairwise_stability_summary.csv",index=False)
    rankm.to_csv(args.out/"rank_spearman_matrix.csv")
    retm.to_csv(args.out/"retained70_jaccard_matrix.csv")
    rejm.to_csv(args.out/"rejected30_jaccard_matrix.csv")
    swm.to_csv(args.out/"decision_switch_rate_matrix.csv")
    cls.to_csv(args.out/"cell_decision_classes.csv",index=False)
    percls.to_csv(args.out/"per_image_decision_class.csv",index=False)
    csum.to_csv(args.out/"decision_class_summary.csv",index=False)
    con.to_csv(args.out/"decision_class_error_contrasts.csv",index=False)
    make_figures(rankm,csum,args.out)

    s=csum.set_index("decision_class")
    orderflag=bool(
        s.loc["always_retained","image_balanced_mean_actual_error"]
        < s.loc["radius_dependent","image_balanced_mean_actual_error"]
        < s.loc["always_rejected","image_balanced_mean_actual_error"]
    )
    report={
        "protocol":"Experiment 3: post-hoc same-cell SMS decision stability across physical perturbation radius",
        "status":"Secondary robustness/mechanism analysis. Frozen 70% coverage and the four Experiment-2 radii are unchanged; no operating point is reselected.",
        "source_file":str(args.input),
        "identity_key":ID_COLS,
        "common_cell_pool":int(len(nrad)),
        "primary_evaluable_images":int(len(eval_cases)),
        "primary_evaluable_cells":int(n_eval),
        "excluded_images_lt10_common_cells":excluded,
        "coverage":COVERAGE,"radii_um":RADII,
        "decision_rule":"Within each image and radius, retain floor(0.70*n) cells with lowest SMS; ties are deterministically broken by gt_label then pred_label.",
        "bootstrap":{"unit":"image","resamples":N_BOOT,"seed":SEED},
        "pairwise_stability":psum.to_dict("records"),
        "decision_class_summary":csum.to_dict("records"),
        "decision_class_error_contrasts":con.to_dict("records"),
        "error_order_always_retained_lt_radius_dependent_lt_always_rejected":orderflag,
        "interpretation_rule":[
            "High pairwise SMS-rank correlation indicates preserved reliability ordering across physical scales.",
            "High retained/rejected Jaccard overlap and low switch rate indicate stable individual-cell QC decisions.",
            "If always-retained cells have lower true measurement error than radius-dependent cells, which have lower error than always-rejected cells, cross-radius consensus is meaningful rather than merely self-consistent.",
            "If aggregate RR is robust but decision overlap is low, report population-level robustness with cell-level instability."
        ]
    }
    (args.out/"experiment3_summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")

    print("PAIRWISE IMAGE-BALANCED STABILITY")
    print("-"*76)
    print(psum[["radius1_um","radius2_um","sms_rank_spearman","retained70_jaccard",
                "rejected30_jaccard","decision_switch_rate","mean_abs_rank_percentile_shift"]]
          .to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\nDECISION-CLASS SUMMARY")
    print("-"*76)
    print(csum[["decision_class","n_cells","image_balanced_proportion",
                "image_balanced_mean_actual_error","actual_error_ci_low","actual_error_ci_high"]]
          .to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\nPAIRED IMAGE-LEVEL ERROR CONTRASTS")
    print("-"*76)
    print(con.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("\nError ordering retained < dependent < rejected:",orderflag)
    print("\nSaved to:",args.out)

if __name__=="__main__":
    main()