#!/usr/bin/env bash
set -euo pipefail

# Small local debug run for a single 3090 (24GB).

DATA_DIR=${DATA_DIR:-poc_data_attr_dbg}
OUT_DIR=${OUT_DIR:-poc_outputs_dbg}
MODEL_ID=${MODEL_ID:-llava-hf/llava-1.5-7b-hf}

K=${K:-4}
PER_COND=${PER_COND:-2}
SAMPLES=${SAMPLES:-50}
BATCH=${BATCH:-8}
DTYPE=${DTYPE:-fp16}
LOAD_4BIT=${LOAD_4BIT:-0}

EXTRA_ARGS=()
if [[ "${LOAD_4BIT}" == "1" ]]; then
  EXTRA_ARGS+=("--load_4bit")
fi

mkdir -p "${OUT_DIR}"

echo "[1/4] Generate dataset"
python generate_synth_dataset_attr.py \
  --out_dir "${DATA_DIR}" \
  --k "${K}" \
  --per_condition "${PER_COND}" \
  --img_size 512 \
  --conditions sym,center,size,contrast,occlusion

DATA_JSONL="${DATA_DIR}/metadata.jsonl"

echo "[2/4] Choice-only (baseline)"
CUDA_VISIBLE_DEVICES=0 python run_vlm_choice_only.py \
  --model_id "${MODEL_ID}" \
  --data_jsonl "${DATA_JSONL}" \
  --out_jsonl "${OUT_DIR}/choice_only_none.jsonl" \
  --samples_per_image "${SAMPLES}" \
  --batch_size "${BATCH}" \
  --restrict_format \
  --calibration none \
  --num_shards 1 --shard_id 0 \
  --dtype "${DTYPE}" \
  "${EXTRA_ARGS[@]}"

python analyze_choice_only.py \
  --results_jsonl "${OUT_DIR}/choice_only_none.jsonl" \
  --out_csv "${OUT_DIR}/choice_only_none.csv"

echo "[3/4] Choice+Color (baseline)"
CUDA_VISIBLE_DEVICES=0 python run_vlm_choice_color.py \
  --model_id "${MODEL_ID}" \
  --data_jsonl "${DATA_JSONL}" \
  --out_jsonl "${OUT_DIR}/choice_color_none.jsonl" \
  --samples_per_image "${SAMPLES}" \
  --batch_size "${BATCH}" \
  --restrict_format \
  --calibration none \
  --num_shards 1 --shard_id 0 \
  --dtype "${DTYPE}" \
  "${EXTRA_ARGS[@]}"

python analyze_choice_color.py \
  --results_jsonl "${OUT_DIR}/choice_color_none.jsonl" \
  --out_csv "${OUT_DIR}/choice_color_none.csv"

echo "[4/4] Done. See ${OUT_DIR}/"
