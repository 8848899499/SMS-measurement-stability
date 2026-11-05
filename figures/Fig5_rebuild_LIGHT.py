from pathlib import Path
import pandas as pd, numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

HERE=Path(__file__).resolve().parent
RESULTS=HERE.parent/"results_reference"
pair=pd.read_csv(RESULTS/"pairwise_stability_summary.csv")
cls=pd.read_csv(RESULTS/"decision_class_summary.csv")
radii=[0.5,1.0,1.5,2.0]
mat=np.eye(4)
for _,r in pair.iterrows():
    i=radii.index(float(r.radius1_um)); j=radii.index(float(r.radius2_um))
    mat[i,j]=mat[j,i]=float(r.sms_rank_spearman)

mpl.rcParams.update({
    "svg.fonttype":"none","font.family":"sans-serif","font.size":7.2,
    "axes.labelsize":7.2,"xtick.labelsize":6.8,"ytick.labelsize":6.8
})
light_rank=LinearSegmentedColormap.from_list(
    "light_rank",["#D8EAF5","#E4F0F7","#EEF6F2","#FFF8D8"]
)

fig=plt.figure(figsize=(3.5,3.45))
ax=fig.add_axes([0.21,0.59,0.56,0.35])
mesh=ax.imshow(mat,vmin=0.78,vmax=1.0,aspect="equal",
               interpolation="nearest",cmap=light_rank)
ax.set_xticks(range(4),["0.5","1.0","1.5","2.0"])
ax.set_yticks(range(4),["0.5","1.0","1.5","2.0"])
ax.set_xlabel("Perturbation radius (µm)",labelpad=2)
ax.set_ylabel("Perturbation radius (µm)",labelpad=2)
ax.text(-0.13,1.03,"(a)",transform=ax.transAxes,fontweight="bold",va="bottom")
for i in range(4):
    for j in range(4):
        ax.text(j,i,f"{mat[i,j]:.2f}",ha="center",va="center",fontsize=6.8,color="black")
cax=fig.add_axes([0.81,0.59,0.035,0.35]); cb=fig.colorbar(mesh,cax=cax)
cb.set_label("Image-balanced Spearman ρ",fontsize=6.5,labelpad=3)
cb.ax.tick_params(labelsize=6.1)

ax2=fig.add_axes([0.20,0.18,0.76,0.25])
order=["always_retained","radius_dependent","always_rejected"]
labels=["Always\nretained","Radius-\ndependent","Always\nrejected"]
s=cls.set_index("decision_class").loc[order]
y=s.image_balanced_mean_actual_error.to_numpy(float)
lo=s.actual_error_ci_low.to_numpy(float); hi=s.actual_error_ci_high.to_numpy(float)
n=s.n_cells.to_numpy(int); x=np.arange(3)
ax2.errorbar(x,y,yerr=np.vstack([y-lo,hi-y]),marker="o",linestyle="none",
             capsize=2.5,markersize=4.0)
ax2.set_xticks(x,[f"{labels[i]}\n(n={n[i]:,})" for i in range(3)])
ax2.set_ylabel("Manual-reference\nmean-intensity error",labelpad=3)
ax2.set_ylim(0.10,0.315); ax2.set_xlim(-0.15,2.15)
ax2.text(-0.12,1.03,"(b)",transform=ax2.transAxes,fontweight="bold",va="bottom")
ax2.tick_params(direction="out")

fig.savefig(HERE/"Fig5_same_cell_stability_column_LIGHT.svg")
fig.savefig(HERE/"Fig5_same_cell_stability_column_LIGHT_600dpi.png",dpi=600)
plt.close(fig)
im=Image.open(HERE/"Fig5_same_cell_stability_column_LIGHT_600dpi.png").convert("RGB")
im.save(HERE/"Fig5_same_cell_stability_column_LIGHT_600dpi.tiff",
        dpi=(600,600),compression="tiff_lzw")
