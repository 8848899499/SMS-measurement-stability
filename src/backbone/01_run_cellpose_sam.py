
from pathlib import Path
import argparse,json,numpy as np
from sms_backbone_common import read_tif,unseg_channels
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,default=Path(r"E:\UNSEG_GIT"))
    ap.add_argument("--out",type=Path,default=Path(r"E:\Histo_Pilot\sms_backbone_transfer\cellpose_predictions"))
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    from cellpose import models
    model=models.CellposeModel(gpu=True,pretrained_model="cpsam_v2")
    tifs=sorted(a.root.glob("*.tif"))
    if len(tifs)!=75: raise RuntimeError(f"Expected 75 TIFFs, found {len(tifs)}")
    for i,t in enumerate(tifs):
        op=a.out/(t.stem+".npz")
        if op.exists(): print("Reuse",i+1,"/",len(tifs),t.name,flush=True); continue
        mem,nuc=unseg_channels(read_tif(t))
        img=np.stack([mem,nuc],axis=-1).astype(np.float32)
        res=model.eval(img,channel_axis=-1,diameter=None,flow_threshold=0.4,
                       cellprob_threshold=0.0,min_size=15,normalize=True)
        masks=np.asarray(res[0]).squeeze().astype(np.int32)
        np.savez_compressed(op,masks=masks)
        print("Cellpose-SAM",i+1,"/",len(tifs),t.name,"cells",int(masks.max()),flush=True)
    (a.out/"MODEL_INFO.json").write_text(json.dumps({"model":"cpsam_v2","flow_threshold":0.4,
      "cellprob_threshold":0.0,"diameter":None,"min_size":15,"normalize":True},indent=2),encoding="utf-8")
if __name__=="__main__": main()
