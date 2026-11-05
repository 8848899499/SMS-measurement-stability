from pathlib import Path
import argparse,json,sys
import numpy as np
from histo10_common import cell_uncertainty_rows,summarize_case_metrics,aggregate_by_parent,aggregate_parent_metrics,bootstrap_parent_risk,write_csv

def load_parent_map(selection):
    s=json.loads(Path(selection).read_text(encoding='utf-8'))
    mp={}
    for p in s.get('parents',[]):
        for i in p.get('indices',[]):
            mp[int(i)]=str(p.get('parent','unknown'))
    return mp

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cache',type=Path,default=Path(r'E:\Histo_Pilot\histo08\cache\train'))
    ap.add_argument('--selection',type=Path,default=Path(r'E:\Histo_Pilot\histo08\selection\train_selection.json'))
    ap.add_argument('--mesmer',type=Path,default=Path(r'E:\Histo_Pilot\histo08\mesmer\train_mesmer.npz'))
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)

    X=np.load(a.cache/'X.npy',mmap_mode='r')
    Y=np.load(a.cache/'y.npy',mmap_mode='r')
    pmap=load_parent_map(a.selection)
    with np.load(a.mesmer,allow_pickle=False) as z:
        idx=np.asarray(z['indices'],int)
        pred=np.asarray(z['whole_cell'],np.int32)
        mpp=np.asarray(z['pixel_size'],float)
    if len(idx)!=len(pred): raise RuntimeError('TissueNet Mesmer index mismatch')

    cells=[]
    for j,i in enumerate(idx):
        parent=pmap.get(int(i),'unknown')
        rr=cell_uncertainty_rows(pred[j],np.asarray(Y[i,...,0],np.int32),
                                 np.asarray(X[i,...,1],float),float(mpp[j]),
                                 case=f'tissuenet_{int(i)}',group=parent)
        cells.extend(rr)
        print('TissueNet',j+1,'/',len(idx),'matched_cells',len(rr),flush=True)

    cases=summarize_case_metrics(cells)
    parents=aggregate_by_parent(cases)
    sms=aggregate_parent_metrics(parents,'sms_sum')
    geom=aggregate_parent_metrics(parents,'geometry_span')
    area=aggregate_parent_metrics(parents,'inverse_sqrt_area')
    mean_sms=aggregate_parent_metrics(parents,'sms_mean')
    boot=bootstrap_parent_risk(parents,'sms_sum',B=2000,seed=20261102)

    go=bool(
        sms['n_cells']>=1000 and
        sms['mean_spearman']>=0.25 and
        sms['mean_risk_ratio_80']<=0.80 and
        sms['mean_risk_ratio_80']<=geom['mean_risk_ratio_80']-0.05
    )
    summary=dict(
      protocol='Histo10 SMS development on TissueNet TRAIN only',
      method='Training-free per-cell measurement stability under +/-1.0 um boundary perturbation',
      primary_endpoint='integrated membrane/cell signal',
      development_unit='TissueNet parent (parent-balanced)',
      sms_sum=sms,geometry_baseline=geom,inverse_area_baseline=area,
      secondary_sms_mean=mean_sms,bootstrap_95ci=boot,
      go_rule={
        'matched_cells_min':1000,
        'parent_balanced_spearman_min':0.25,
        'parent_balanced_risk_ratio80_max':0.80,
        'sms_must_beat_geometry_risk_ratio80_by':0.05
      },
      decision='GO_EXTERNAL_UNSEG' if go else 'STOP_HISTO10_BEFORE_UNSEG'
    )
    write_csv(a.out/'tissuenet_cell_rows.csv',cells)
    write_csv(a.out/'tissuenet_case_metrics.csv',cases)
    write_csv(a.out/'tissuenet_parent_metrics.csv',parents)
    (a.out/'development_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (a.out/'development_report.txt').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    sys.exit(0 if go else 20)

if __name__=='__main__':
    main()
