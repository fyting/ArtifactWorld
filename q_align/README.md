# Q-Align Artifact Labeling

This directory contains the Q-Align-based inference code used to classify nine 3DGS artifact types in videos:

`Aliasing`, `Blurring`, `Color`, `Crack`, `Dilation`, `Floater`, `Ghosting`, `Needles`, and `Popping`.

The inference entry point is [`tools/infer_q_align.py`](https://github.com/fyting/ArtifactWorld/blob/main/tools/infer_q_align.py). The model weights are hosted separately on Hugging Face:

**[buaadwxl/Q-Align-Artifacts](https://huggingface.co/buaadwxl/Q-Align-Artifacts)**

## Download weights

The Hugging Face repository contains both the `one-align` base model and the `q-align-artifacts` SFT adapter. Download it into the repository root:

```bash
hf download buaadwxl/Q-Align-Artifacts --local-dir ./q_align_weights
```

The resulting layout should be:

```text
q_align_weights/
├── one-align/
└── q-align-artifacts/
```

The weights are large and are intentionally not stored in GitHub. To use another location, set `QALIGN_WEIGHTS_ROOT` or pass `--model_base` and `--model_path` explicitly.

## Environment

Q-Align inference requires an NVIDIA GPU and CUDA. The tested environment uses Python 3.10, PyTorch 2.0.1 with CUDA 11.8, `transformers==4.36.1`, and `peft==0.4.0`.

Create the environment with:

```bash
bash tools/q_align/setup_env.sh
conda activate qalign-artifacts
```

The pinned dependencies are listed in [`tools/q_align/requirements.txt`](https://github.com/fyting/ArtifactWorld/blob/main/tools/q_align/requirements.txt). The original Q-Align-related license is preserved in [`LICENSE`](https://github.com/fyting/ArtifactWorld/blob/main/q_align/LICENSE).

## Inference

Run one video:

```bash
python tools/infer_q_align.py \
  --video /path/to/clip.mp4 \
  --output labels.json
```

Run a list of videos, one path per line:

```bash
python tools/infer_q_align.py \
  --video_list examples/q_align_video_list.txt \
  --output labels.json
```

Use multiple GPUs by assigning one full model replica to each GPU:

```bash
python tools/infer_q_align.py \
  --video_list examples/q_align_video_list.txt \
  --output labels.json \
  --num_gpus 8
```

To classify only selected artifact types:

```bash
python tools/infer_q_align.py \
  --video /path/to/clip.mp4 \
  --artifacts Blurring Ghosting
```

The output JSON contains a `Yes`/`No` prediction and the yes/no probabilities for each artifact type. If GPU memory is limited, add `--use_8bit` or reduce `--max_frames`.

## Attribution and licenses

The code is based on Q-Align / one-align and includes model components from mPLUG-Owl2 and LLaVA. See [`LICENSE`](LICENSE) and the upstream model terms before redistribution or commercial use. The base and SFT model weights retain their respective upstream license terms.
