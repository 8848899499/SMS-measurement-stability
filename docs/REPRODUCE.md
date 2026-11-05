# Reproduction guide

This guide follows the manuscript analysis order. The original scripts are intentionally close to the frozen analysis code; some default paths reflect the authors' Windows workstation. Prefer explicit command-line path arguments where supported.

## 1. Environment

```bash
conda env create -f environment.yml
conda activate sms-analysis
```

For the Cellpose-SAM stress test:

```bash
conda env create -f environment_cellpose.yml
conda activate sms-cellpose
```

Install the PyTorch build appropriate for the target GPU/CPU platform separately.

## 2. TissueNet development

Run `src/development/histo10_01_develop_tissuenet.py` with the local TissueNet inputs and frozen Mesmer prediction source expected by the script. Verification target: `results_reference/development_summary.json`.

## 3. Frozen UNSEG external validation

Provide the frozen Mesmer prediction NPZ used by the analysis. It must contain `case_names`, `whole_cell`, and `pixel_size`, with the 75 UNSEG cases in sorted TIFF order and pixel size 0.16 µm/pixel. Then run:

```bash
python src/external/histo10b_external_unseg.py --root <UNSEG_ROOT> --mesmer <MESMER_PREDICTIONS_NPZ> --out <OUTPUT_DIR>
```

The exact frozen protocol and development evidence are in `configs/PROTOCOL_HISTO10B_MEAN_SMS.json` and `configs/FROZEN_DEVELOPMENT_EVIDENCE.json`. Verification target: `results_reference/unseg_summary.json`.

## 4. Cellpose-SAM backbone transfer

With the Cellpose environment active:

```bash
python src/backbone/01_run_cellpose_sam.py
python src/backbone/02_evaluate_backbone_transfer.py
```

The manuscript's prespecified transfer rule was not met. Do not reinterpret the post-hoc mechanism audit as overturning that result.

Optional explanatory audit:

```bash
python src/backbone/03_same_cell_cross_backbone.py
```

## 5. Physical-scale robustness

```bash
python src/robustness/04_physical_scale_mesmer.py
```

This script first reconstructs the frozen 1.0 µm analysis and hard-stops unless all 6,658 reference rows reproduce exactly. Only after that gate passes does it evaluate 0.5, 1.5, and 2.0 µm.

Verification targets:

- `results_reference/physical_scale_native_summary.csv`
- `results_reference/physical_scale_commoncell_summary.csv`

## 6. Same-cell decision stability

```bash
python src/robustness/05_same_cell_decision_stability.py
```

Primary analysis uses the same 6,423 cells from 72 evaluable images (≥10 common cells/image) and retains the frozen 70% coverage rule.

Verification targets:

- `results_reference/experiment3_summary.json`
- `results_reference/pairwise_stability_summary.csv`
- `results_reference/decision_class_summary.csv`
- `results_reference/decision_class_error_contrasts.csv`

## 7. Figures

Figure 1 is stored as the final editable SVG. Figures 4 and 5 include rebuild scripts:

```bash
python figures/Fig4_physical_scale.py
python figures/Fig5_rebuild_LIGHT.py
```

The submitted artwork may include author-side layout edits made in Inkscape; the numerical values must remain identical to the reference outputs.
