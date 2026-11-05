from pathlib import Path
import argparse,json,sys
import numpy as np
from histo10b_common import cell_rows,case_metrics,aggregate,bootstrap,write_csv

def read_tif(p):
    import tifffile
    return np.asarray(tifffile.imread(p))

def load_label(p):
    last=None
    for d in (None,",","\t"):
        try:
            x=np.loadtxt(p,delimiter=d)
            if x.size:return np.rint(x).astype(np.int32)
        except Exception as e:last=e
    raise RuntimeError(f"Cannot parse {p}: {last}")

def organ_from_case(c):
    x=c.lower()
    if "smallintestine" in x or "small_intestine" in x or "small intestine" in x:return "small_intestine"
    for o in ["appendix","colon","esophagus","gallbladder","liver","pancreas","stomach"]:
        if o in x:return o
    return "unknown"

def membrane(img):
    a=np.asarray(img)
    if a.ndim==3 and a.shape[-1]>=3:return np.asarray(a[...,0],float)
    if a.ndim==3 and a.shape[0]>=3 and a.shape[0]<=4:return np.asarray(a[0],float)
    raise ValueError(f"Unexpected TIFF shape {a.shape}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=Path(r"E:\UNSEG_GIT"))
    ap.add_argument("--mesmer",type=Path,default=Path(r"E:\UNSEG_GIT\histo09_run\mesmer_predictions.npz"))
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)

    tifs=sorted(a.root.glob("*.tif"))
    if len(tifs)!=75:raise RuntimeError(f"Expected 75 TIFFs, found {len(tifs)}")
    with np.load(a.mesmer,allow_pickle=False) as z:
        names=z["case_names"].astype(str).tolist()
        pred=np.asarray(z["whole_cell"],np.int32)
        mpp=np.asarray(z["pixel_size"],float)
    if names != [t.stem for t in tifs]:raise RuntimeError("Mesmer case order mismatch")
    if not np.allclose(mpp,0.16):raise RuntimeError("Mesmer MPP mismatch")

    cells=[]
    for i,t in enumerate(tifs):
        gt=load_label(a.root/(t.stem+"_annotation_cell.txt"))
        mem=membrane(read_tif(t))
        rr=cell_rows(pred[i],gt,mem,0.16,t.stem,organ_from_case(t.stem),1.0)
        cells.extend(rr)
        print("UNSEG",i+1,"/",len(tifs),"matched cells",len(rr),flush=True)

    cases=case_metrics(cells)
    sms=aggregate(cases,"sms_mean")
    geom=aggregate(cases,"geometry_span")
    area=aggregate(cases,"inverse_sqrt_area")
    boot=bootstrap(cases,"sms_mean",5000,20261102)

    organ_rows=[]
    for o in sorted(set(r["organ"] for r in cases)):
        rr=[r for r in cases if r["organ"]==o]
        for score in ["sms_mean","geometry_span","inverse_sqrt_area"]:
            organ_rows.append({"organ":o,"score":score,**aggregate(rr,score)})

    go=bool(
        sms["n_cells"]>=500 and
        sms["mean_spearman"]>=0.20 and
        sms["risk_ratio_70"]<=0.85 and
        sms["risk_ratio_70"]<=geom["risk_ratio_70"]-0.10 and
        boot["risk_ratio_70_95ci"][1] < 1.0
    )
    out={
      "protocol":"Histo10b frozen mean-intensity SMS external test on UNSEG GIT",
      "n_images":75,
      "sms_mean":sms,
      "geometry_baseline":geom,
      "inverse_area_baseline":area,
      "bootstrap_95ci":boot,
      "external_rule":{
        "matched_cells_min":500,
        "image_balanced_spearman_min":0.20,
        "image_balanced_risk_ratio70_max":0.85,
        "sms_must_beat_geometry_by":0.10,
        "bootstrap_upper95_risk_ratio70_lt":1.0
      },
      "decision":"CONTINUE_HISTO10B" if go else "STOP_HISTO10B_NO_GENERALIZATION",
      "guardrails":[
        "UNSEG manual masks are used only to compute evaluation error and matches.",
        "SMS itself uses only raw Mesmer mask, red membrane image, and 0.16 um/pixel.",
        "No threshold, radius, coverage, or score was tuned on UNSEG."
      ]
    }
    write_csv(a.out/"unseg_cell_rows.csv",cells)
    write_csv(a.out/"unseg_case_metrics.csv",cases)
    write_csv(a.out/"unseg_organ_summary.csv",organ_rows)
    (a.out/"summary.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    (a.out/"report.txt").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out,indent=2),flush=True)
    sys.exit(0 if go else 30)

if __name__=="__main__":
    main()
