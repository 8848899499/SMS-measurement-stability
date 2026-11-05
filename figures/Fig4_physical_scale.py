from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import xml.etree.ElementTree as ET
from PIL import Image

HERE = Path(__file__).resolve().parent
native = pd.read_csv(HERE / "mesmer_physical_scale_native_summary.csv").sort_values("requested_radius_um").reset_index(drop=True)
common = pd.read_csv(HERE / "mesmer_physical_scale_commoncell_summary.csv").sort_values("requested_radius_um").reset_index(drop=True)

mpl.rcParams.update({
    "svg.fonttype":"none",
    "font.family":"sans-serif",
    "font.size":7.5,
    "axes.labelsize":7.5,
    "xtick.labelsize":7.0,
    "ytick.labelsize":7.0,
    "legend.fontsize":6.4,
})

WIDTH, HEIGHT = 3.5, 1.62
x = native["requested_radius_um"].to_numpy(float)
xoff = 0.015
xticklabels = ["0.5","1.0*","1.5","2.0"]

def err(y, lo, hi):
    y=np.asarray(y,float); lo=np.asarray(lo,float); hi=np.asarray(hi,float)
    return np.vstack([y-lo, hi-y])

def make_a():
    fig,ax=plt.subplots(figsize=(WIDTH,HEIGHT))
    fig.subplots_adjust(left=0.19,right=0.985,bottom=0.20,top=0.97)
    yn=native["risk_ratio_70"].to_numpy(float)
    yc=common["risk_ratio_70"].to_numpy(float)
    ax.errorbar(x-xoff,yn,yerr=err(yn,native["rr70_ci_low"],native["rr70_ci_high"]),
                marker="o",markersize=3.6,capsize=2.0,label="Native per-radius")
    ax.errorbar(x+xoff,yc,yerr=err(yc,common["rr70_ci_low"],common["rr70_ci_high"]),
                marker="s",markersize=3.6,capsize=2.0,label="Common-cell (n=6,437)")
    ax.set_xlim(0.38,2.12); ax.set_ylim(0.79,0.895)
    ax.set_xticks(x,xticklabels); ax.set_ylabel("RR(0.70)")
    ax.text(0.015,0.955,"(a)",transform=ax.transAxes,fontweight="bold",va="top")
    ax.legend(loc="upper left",frameon=False,bbox_to_anchor=(0.09,0.98),
              handlelength=2.0,borderaxespad=0.0)
    ax.tick_params(direction="out")
    fig.savefig(HERE/"Fig4a_RR70.svg")
    fig.savefig(HERE/"Fig4a_RR70_600dpi.png",dpi=600)
    plt.close(fig)

def make_b():
    fig,ax=plt.subplots(figsize=(WIDTH,HEIGHT))
    fig.subplots_adjust(left=0.19,right=0.985,bottom=0.31,top=0.97)
    yn=native["mean_spearman"].to_numpy(float)
    yc=common["mean_spearman"].to_numpy(float)
    ax.errorbar(x-xoff,yn,yerr=err(yn,native["spearman_ci_low"],native["spearman_ci_high"]),
                marker="o",markersize=3.6,capsize=2.0,label="Native per-radius")
    ax.errorbar(x+xoff,yc,yerr=err(yc,common["spearman_ci_low"],common["spearman_ci_high"]),
                marker="s",markersize=3.6,capsize=2.0,label="Common-cell (n=6,437)")
    ax.set_xlim(0.38,2.12); ax.set_ylim(0.22,0.415)
    ax.set_xticks(x,xticklabels)
    ax.set_xlabel("Nominal perturbation radius (μm)")
    ax.set_ylabel("Spearman ρ")
    ax.text(0.015,0.955,"(b)",transform=ax.transAxes,fontweight="bold",va="top")
    ax.tick_params(direction="out")
    fig.savefig(HERE/"Fig4b_Spearman.svg")
    fig.savefig(HERE/"Fig4b_Spearman_600dpi.png",dpi=600)
    plt.close(fig)

def prefix_ids(root,prefix):
    ids={}
    for el in root.iter():
        if "id" in el.attrib:
            old=el.attrib["id"]; new=prefix+old; ids[old]=new; el.set("id",new)
    for el in root.iter():
        for k,v in list(el.attrib.items()):
            for old,new in ids.items():
                v=v.replace(f"url(#{old})",f"url(#{new})").replace(f"#{old}",f"#{new}")
            el.set(k,v)
    return root

def combine_svg():
    ns="http://www.w3.org/2000/svg"; ET.register_namespace("",ns)
    def parse(p):
        root=ET.parse(p).getroot()
        vb=[float(v) for v in root.attrib["viewBox"].split()]
        return root,vb
    a,va=parse(HERE/"Fig4a_RR70.svg"); b,vb=parse(HERE/"Fig4b_Spearman.svg")
    a=prefix_ids(a,"a_"); b=prefix_ids(b,"b_")
    W=max(va[2],vb[2]); gap=3.0; H=va[3]+gap+vb[3]
    root=ET.Element(f"{{{ns}}}svg",{"width":f"{WIDTH}in","height":f"{WIDTH*H/W:.4f}in",
                                   "viewBox":f"0 0 {W:.6f} {H:.6f}","version":"1.1"})
    sa=ET.SubElement(root,f"{{{ns}}}svg",{"x":"0","y":"0","width":str(W),"height":str(va[3]),
                                         "viewBox":f"{va[0]} {va[1]} {va[2]} {va[3]}","id":"panel_a"})
    for child in list(a): sa.append(child)
    sb=ET.SubElement(root,f"{{{ns}}}svg",{"x":"0","y":str(va[3]+gap),"width":str(W),"height":str(vb[3]),
                                         "viewBox":f"{vb[0]} {vb[1]} {vb[2]} {vb[3]}","id":"panel_b"})
    for child in list(b): sb.append(child)
    ET.ElementTree(root).write(HERE/"Fig4_physical_scale_column.svg",
                               encoding="utf-8",xml_declaration=True)

def combine_png():
    a=Image.open(HERE/"Fig4a_RR70_600dpi.png").convert("RGB")
    b=Image.open(HERE/"Fig4b_Spearman_600dpi.png").convert("RGB")
    out=Image.new("RGB",(max(a.width,b.width),a.height+b.height),"white")
    out.paste(a,(0,0)); out.paste(b,(0,a.height))
    out.save(HERE/"Fig4_physical_scale_column_600dpi.png",dpi=(600,600))

make_a(); make_b(); combine_svg(); combine_png()
print("Fig. 4 rebuilt.")