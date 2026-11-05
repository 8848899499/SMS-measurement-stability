# Segmentation-to-Measurement Stability (SMS)

Code and compact reference outputs for the manuscript:

**Segmentation-to-Measurement Stability for Training-Free Reliability Assessment of Single-Cell Mean Fluorescence Measurements**

Authors: **Chih Mao Tsai** and **Jiann-Der Lee**.

## Overview

Segmentation-to-measurement stability (SMS) is a training-free, per-cell reliability score for quantitative fluorescence measurements. Rather than estimating whether a predicted cell mask is geometrically correct, SMS asks whether the reported **mean fluorescence** changes materially under a small, physically defined inward/outward displacement of the predicted boundary.

This public release contains the code needed for the manuscript's main reproducibility path:

1. TissueNet development and endpoint selection.
2. Frozen external validation on UNSEG gastrointestinal tissue images.
3. Cellpose-SAM segmentation-backbone transfer stress test.
4. Physical-scale robustness at 0.5, 1.0, 1.5, and 2.0 µm.
5. Same-cell decision stability across perturbation radii.

Additional explanatory/third-domain audits reported in the manuscript did not define the frozen operating point. The legacy CL3 archive is intentionally not included in v0.1.0 because an early protocol metadata label was superseded during the manuscript audit; no frozen result is silently rewritten here.

The **1.0 µm perturbation radius and 70% coverage operating point remain the frozen primary setting**. The later scale and same-cell analyses are post-hoc robustness/mechanism analyses and are not used to reselect an operating point.

## Repository layout

```text
configs/                 frozen protocol/evidence JSON files
src/development/         TissueNet development code
src/external/            frozen UNSEG external SMS evaluation
src/backbone/            Cellpose-SAM transfer and cross-backbone audit
src/robustness/          physical-scale and same-cell stability experiments
figures/                 figure rebuild scripts
results_reference/       compact aggregate reference outputs
```

## Data

Source microscopy data are **not redistributed** in this repository.

- **TissueNet**: obtain from the original TissueNet/DeepCell distribution under its terms. The paper development analysis used 172 training tiles from 24 parent image groups and yielded 47,854 IoU-matched cells after the frozen Mesmer predictions were matched to manual whole-cell annotations.
- **UNSEG GIT**: obtain the public gastrointestinal tissue release from its original distributor (Synapse accession `syn61804540`; DOI `10.7303/syn61804540`). The analysis uses 75 images at 0.16 µm/pixel.
- **CL3**: obtain the original release from Figshare (DOI `10.6084/m9.figshare.21702068`). The preregistered third-domain test was inconclusive because the frozen pipeline produced too few matched cells.

See [`docs/DATA.md`](docs/DATA.md) for expected local inputs and file semantics.

## Environment

The core SMS analyses use NumPy, pandas, SciPy, scikit-image, tifffile, and matplotlib. A minimal environment is provided in [`environment.yml`](environment.yml).

The Cellpose-SAM stress test used **Cellpose 4.2.1.1** (`cpsam_v2`) and is separated into [`environment_cellpose.yml`](environment_cellpose.yml) because PyTorch/CUDA installation is platform-specific.

## Reproduction order

The archived scripts retain the frozen analysis logic used for the manuscript. Some scripts include Windows paths as defaults from the original analysis workstation; supply the corresponding command-line path arguments for your local installation.

A concise reproduction guide is in [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

### Main reported checkpoints

The compact reference outputs in `results_reference/` should reproduce the manuscript-level summaries. Key checkpoints include:

- TissueNet / Mesmer: parent-balanced Spearman correlation ≈ **0.485**; `RR(0.70)` ≈ **0.749**.
- UNSEG / Mesmer: image-balanced Spearman correlation ≈ **0.336**; `RR(0.70)` ≈ **0.829**.
- UNSEG / Cellpose-SAM: image-balanced Spearman correlation ≈ **0.198**; `RR(0.70)` ≈ **0.902**; the prespecified backbone-transfer rule was **not met**.
- Common-cell physical-scale `RR(0.70)` at 0.5/1.0/1.5/2.0 µm: approximately **0.821 / 0.821 / 0.830 / 0.862**.
- Same-cell adjacent-radius rank correlations: approximately **0.958–0.964**.
- Cross-radius decision classes show increasing image-balanced manual-reference error from always retained to radius-dependent to always rejected.

These values are verification targets, not parameters to tune against.

## Code status and interpretation

This repository is intended to support reproducibility of the manuscript's retrospective analyses. It does **not** provide a clinical device, autonomous acceptance rule, or prospective safety claim. Manual-reference masks are used only for evaluation, not as SMS inputs at deployment.

## License

Code in this repository is released under the MIT License. Dataset access and pretrained-model use remain subject to the licenses/terms of their original providers.

## Citation

If you use this code, please cite the accompanying manuscript. Repository citation metadata are provided in [`CITATION.cff`](CITATION.cff).
