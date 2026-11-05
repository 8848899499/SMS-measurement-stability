# Data and local input layout

The repository does not redistribute TissueNet, UNSEG, CL3, or pretrained-model files.

## TissueNet

The development code expects TissueNet image/annotation data and archived Mesmer predictions produced by the preceding segmentation stage. The manuscript development subset contained 172 tiles from 24 parent groups. The exact development protocol is frozen in `configs/PROTOCOL_HISTO10_SMS.json`.

Because TissueNet distribution formats can change, inspect the command-line arguments at the bottom of `src/development/histo10_01_develop_tissuenet.py` and point them to your local copy rather than relying on the historical Windows defaults.

## UNSEG GIT

The frozen external analysis used 75 expert-annotated 1000 × 1000 gastrointestinal tissue images at 0.16 µm/pixel. Obtain the data from the original distributor (Synapse accession `syn61804540`; DOI `10.7303/syn61804540`).

The upstream Mesmer segmentation utility writes an NPZ containing at least `case_names`, `whole_cell`, and `pixel_size`. The physical-scale robustness script verifies the expected 75 cases and 0.16 µm/pixel scale before proceeding.

## Cellpose-SAM

The backbone-transfer analysis resegments the same 75 UNSEG images using Cellpose 4.2.1.1 `cpsam_v2`. The protocol is in `configs/PROTOCOL_BACKBONE_TRANSFER.json`.

## CL3 applicability audit

The third-domain audit uses the CL3 public release from Figshare DOI `10.6084/m9.figshare.21702068`. The final audit was classified as inconclusive because the frozen Mesmer pipeline yielded too few IoU-matched cells relative to the preregistered minimum. No parameters were changed to rescue this result.

## No source data in Git

Do not commit raw images, annotations, pretrained checkpoints, patient-linked metadata, generated segmentation stacks, or large per-cell intermediate files. `.gitignore` contains common extensions/directories for these items.
