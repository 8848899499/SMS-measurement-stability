
from pathlib import Path
import argparse,json,sys,numpy as np
from sms_backbone_common import *
def safe(x):
    if isinstance(x,dict): return {k:safe(v) for k,v in x.items()}
    if isinstance(x,list): return [safe(v) for v in x]
    if isinstance(x,np.generic): return x.item()
    return x
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=Path(r"E:\UNSEG_GIT"))
    ap.add_argument("--pred",type=Path,default=Path(r"E:\Histo_Pilot\sms_backbone_transfer\cellpose_predictions"))
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    tifs=sorted(a.root.glob("*.tif")); cells=[]; diag=[]
    for i,t in enumerate(tifs):
        pred=np.asarray(np.load(a.pred/(t.stem+".npz"),allow_pickle=False)["masks"],np.int32)
        gt=load_label_txt(a.root/(t.stem+"_annotation_cell.txt")); mem,_=unseg_channels(read_tif(t))
        rr=cell_rows(pred,gt,mem,t.stem); cells.extend(rr)
        diag.append({"case":t.stem,"manual_cells":int(gt.max()),"cellpose_cells":int(pred.max()),"matched_iou50":len(rr)})
        print("Evaluate",i+1,"/",len(tifs),t.stem,"matched",len(rr),flush=True)
    cases=case_metrics(cells); sms=aggregate(cases,"sms_mean"); geom=aggregate(cases,"geometry_span")
    area=aggregate(cases,"inverse_sqrt_area"); boot=bootstrap(cases,"sms_mean",5000,20261104)
    groups={}; partials=[]
    for r in cells: groups.setdefault(r["case"],[]).append(r)
    for case,rr in groups.items():
        if len(rr)<20: continue
        pr=partial_rank([x["sms_mean"] for x in rr],[x["actual_mean_error"] for x in rr],
                        [[x["geometry_span"] for x in rr],[x["raw_area"] for x in rr],[x["raw_mean"] for x in rr]])
        partials.append({"case":case,"partial_spearman_geometry_area_intensity":pr,"n_cells":len(rr)})
    partial_mean=float(np.nanmean([x["partial_spearman_geometry_area_intensity"] for x in partials])) if partials else np.nan
    strata,strata_rows=geometry_strata(cells)
    rule={"matched_cells_min":500,"evaluable_images_min":30,"image_balanced_spearman_min":0.20,
          "risk_ratio70_max":0.90,"risk_ratio70_bootstrap_upper95_lt":1.0,"sms_beats_geometry_by_at_least":0.05}
    support=bool(sms["n_cells"]>=500 and sms["n_evaluable_cases"]>=30 and sms["mean_spearman"]>=0.20 and
                 sms["risk_ratio_70"]<=0.90 and boot["risk_ratio_70_95ci"][1]<1.0 and
                 sms["risk_ratio_70"]<=geom["risk_ratio_70"]-0.05)
    result={"protocol":"SMS backbone transfer on UNSEG using Cellpose-SAM cpsam_v2",
            "sms":sms,"geometry":geom,"inverse_area":area,"bootstrap_95ci":boot,
            "partial_spearman_controlling_geometry_area_intensity":partial_mean,
            "within_geometry_tertiles":strata,"predeclared_rule":rule,
            "decision":"BACKBONE_TRANSFER_SUPPORTED" if support else "BACKBONE_TRANSFER_NOT_SUPPORTED",
            "interpretation":"Robustness analysis on the same external dataset, not a new independent dataset validation."}
    write_csv(a.out/"cell_rows.csv",cells); write_csv(a.out/"case_metrics.csv",cases)
    write_csv(a.out/"diagnostics.csv",diag); write_csv(a.out/"partial_rank_by_image.csv",partials)
    write_csv(a.out/"geometry_strata_by_image.csv",strata_rows)
    (a.out/"summary.json").write_text(json.dumps(safe(result),indent=2),encoding="utf-8")
    (a.out/"report.txt").write_text(json.dumps(safe(result),indent=2),encoding="utf-8")
    print(json.dumps(safe(result),indent=2),flush=True); sys.exit(0 if support else 30)
if __name__=="__main__": main()
