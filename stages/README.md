# Vendored Inference Code

This directory vendors the inference-time core code used by ArtifactWorld.

- `stage1/`: Stage-1 predictor (heatmap / noisemap)
- `stage2/`: Stage-2 restorer (AATF-guided restoration)

Each stage contains:

- `scripts/validate.py`
- `src/ltxv_trainer/`
- upstream license files (`LICENSE.txt`, `LTX-Video-Open-Weights-License-0.X.txt`)

The launchers in `../../scripts/` use these local stage directories by default.
You can still override with environment variables:

- `LTX_STAGE1_ROOT`
- `LTX_STAGE2_ROOT`
