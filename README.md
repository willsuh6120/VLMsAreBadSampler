# VLM Random-Choice PoC (choice-only + choice+color)

This repo builds a small synthetic benchmark to audit whether a VLM can sample uniformly from visual candidates, and how that bias propagates to a downstream "choose + describe" task. It includes:

- Synthetic image generator with controlled visual factors.
- Choice-only sampling audit with chi-square tests.
- Choice+color downstream task with joint accuracy metrics.
- Per-image logit calibration to enforce uniformity without training.
- One-click scripts for local 3090 debugging and 4xGPU PoC runs.

## Setup

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You will need access to the model weights (default: `llava-hf/llava-1.5-7b-hf`).

## Local debug on a single 3090

This is a tiny run to verify the pipeline end-to-end.

```bash
bash run_local_debug_3090.sh
```

Defaults (can override via env vars):

- `PER_COND=2` (images per condition)
- `SAMPLES=50` (samples per image)
- `BATCH=8`
- `DTYPE=fp16`

Outputs go to `poc_outputs_dbg/`.

If you hit OOM, lower `BATCH` or enable 4-bit:

```bash
pip install bitsandbytes
LOAD_4BIT=1 BATCH=4 bash run_local_debug_3090.sh
```

## Full PoC on a 4xGPU node (B200)

```bash
bash run_poc_all_4gpu.sh
```

Defaults (override via env vars):

- `PER_COND=10`
- `SAMPLES=1000`
- `BATCH=128`
- `DTYPE=auto`

Example with custom scale:

```bash
PER_COND=20 SAMPLES=2000 BATCH=192 DTYPE=bf16 bash run_poc_all_4gpu.sh
```

Outputs go to `poc_outputs/`.

## Expected runtime (rough)

These are conservative ranges for LLaVA-1.5 7B:

- 3090 local debug (default settings): ~5–20 minutes.
- B200 4xGPU full PoC (default settings, 4 runs): ~30–90 minutes.

Actual time depends on driver, I/O, and model cache state.

## Outputs

Each run produces JSONL with per-image stats and a CSV summary.

- Choice-only summary: `poc_outputs/choice_only_*.csv`
- Choice+color summary: `poc_outputs/choice_color_*.csv`

Key fields:

- `reject_rate_chi2`, `mean_cramers_v` for uniformity.
- `mean_p_special` for special-digit bias.
- `mean_joint_accuracy_valid` for downstream correctness (choice+color).

## Files

- `generate_synth_dataset_attr.py`: synthetic image generator with visual factors + color labels.
- `run_vlm_choice_only.py`: choice-only sampling audit (uniformity + calibration).
- `run_vlm_choice_color.py`: choice+color downstream task (uniformity + joint accuracy).
- `analyze_choice_only.py`, `analyze_choice_color.py`: aggregation and CSV summaries.
- `run_local_debug_3090.sh`: small local debug run.
- `run_poc_all_4gpu.sh`: full PoC on 4 GPUs.
