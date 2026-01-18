#!/usr/bin/env bash
set -euo pipefail

# End-to-end PoC runner (4 GPUs).

DATA_DIR=${DATA_DIR:-poc_data_attr}
OUT_DIR=${OUT_DIR:-poc_outputs}
MODEL_ID=${MODEL_ID:-llava-hf/llava-1.5-7b-hf}

K=${K:-4}
PER_COND=${PER_COND:-10}
SAMPLES=${SAMPLES:-1000}
BATCH=${BATCH:-128}
DTYPE=${DTYPE:-auto}
LOAD_4BIT=${LOAD_4BIT:-0}

EXTRA_ARGS=()
if [[ "${LOAD_4BIT}" == "1" ]]; then
  EXTRA_ARGS+=("--load_4bit")
fi

mkdir -p "${OUT_DIR}"

echo "[1/5] Generate synthetic dataset"
python generate_synth_dataset_attr.py \
  --out_dir "${DATA_DIR}" \
  --k "${K}" \
  --per_condition "${PER_COND}" \
  --img_size 512 \
  --conditions sym,center,size,contrast,occlusion

DATA_JSONL="${DATA_DIR}/metadata.jsonl"

run_sharded() {
  local script=$1
  local tag=$2
  local calib=$3

  echo "[RUN] ${tag} | calibration=${calib}"

  for shard in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=${shard} python ${script} \
      --model_id "${MODEL_ID}" \
      --data_jsonl "${DATA_JSONL}" \
      --out_jsonl "${OUT_DIR}/${tag}_${calib}_s${shard}.jsonl" \
      --samples_per_image "${SAMPLES}" \
      --batch_size "${BATCH}" \
      --restrict_format \
      --calibration "${calib}" \
      --num_shards 4 --shard_id ${shard} \
      --dtype "${DTYPE}" \
      "${EXTRA_ARGS[@]}" &
  done
  wait

  cat "${OUT_DIR}/${tag}_${calib}_s0.jsonl" \
      "${OUT_DIR}/${tag}_${calib}_s1.jsonl" \
      "${OUT_DIR}/${tag}_${calib}_s2.jsonl" \
      "${OUT_DIR}/${tag}_${calib}_s3.jsonl" \
      > "${OUT_DIR}/${tag}_${calib}.jsonl"
}

echo "[2/5] Choice-only baseline"
run_sharded run_vlm_choice_only.py choice_only none
python analyze_choice_only.py \
  --results_jsonl "${OUT_DIR}/choice_only_none.jsonl" \
  --out_csv "${OUT_DIR}/choice_only_none.csv"

echo "[3/5] Choice-only calibrated"
run_sharded run_vlm_choice_only.py choice_only per_image_logit_bias
python analyze_choice_only.py \
  --results_jsonl "${OUT_DIR}/choice_only_per_image_logit_bias.jsonl" \
  --out_csv "${OUT_DIR}/choice_only_per_image_logit_bias.csv"

echo "[4/5] Choice+Color baseline"
run_sharded run_vlm_choice_color.py choice_color none
python analyze_choice_color.py \
  --results_jsonl "${OUT_DIR}/choice_color_none.jsonl" \
  --out_csv "${OUT_DIR}/choice_color_none.csv"

echo "[5/5] Choice+Color calibrated"
run_sharded run_vlm_choice_color.py choice_color per_image_logit_bias
python analyze_choice_color.py \
  --results_jsonl "${OUT_DIR}/choice_color_per_image_logit_bias.jsonl" \
  --out_csv "${OUT_DIR}/choice_color_per_image_logit_bias.csv"

echo "[DONE] See ${OUT_DIR}/"
