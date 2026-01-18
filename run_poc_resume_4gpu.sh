#!/usr/bin/env bash
set -euo pipefail

# Resume-friendly PoC runner (4 GPUs). Skips steps if outputs already exist.

DATA_DIR=${DATA_DIR:-poc_data_attr}
OUT_DIR=${OUT_DIR:-poc_outputs}
MODEL_ID=${MODEL_ID:-llava-hf/llava-1.5-7b-hf}

K=${K:-4}
PER_COND=${PER_COND:-10}
SAMPLES=${SAMPLES:-1000}
BATCH=${BATCH:-128}
DTYPE=${DTYPE:-auto}
LOAD_4BIT=${LOAD_4BIT:-0}
FORCE=${FORCE:-0}

EXTRA_ARGS=()
if [[ "${LOAD_4BIT}" == "1" ]]; then
  EXTRA_ARGS+=("--load_4bit")
fi

mkdir -p "${OUT_DIR}"

if [[ "${FORCE}" == "1" || ! -s "${DATA_DIR}/metadata.jsonl" ]]; then
  echo "[1/5] Generate synthetic dataset"
  python generate_synth_dataset_attr.py \
    --out_dir "${DATA_DIR}" \
    --k "${K}" \
    --per_condition "${PER_COND}" \
    --img_size 512 \
    --conditions sym,center,size,contrast,occlusion
else
  echo "[1/5] Skip dataset (found ${DATA_DIR}/metadata.jsonl)"
fi

DATA_JSONL="${DATA_DIR}/metadata.jsonl"
if [[ ! -s "${DATA_JSONL}" ]]; then
  echo "[ERROR] Missing ${DATA_JSONL}" >&2
  exit 1
fi

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
}

merge_shards() {
  local tag=$1
  local calib=$2
  local out_jsonl="${OUT_DIR}/${tag}_${calib}.jsonl"
  local missing=0

  for shard in 0 1 2 3; do
    local shard_path="${OUT_DIR}/${tag}_${calib}_s${shard}.jsonl"
    if [[ ! -s "${shard_path}" ]]; then
      missing=1
    fi
  done

  if [[ "${missing}" == "1" ]]; then
    echo "[ERROR] Missing shard outputs for ${tag} | calibration=${calib}" >&2
    exit 1
  fi

  cat "${OUT_DIR}/${tag}_${calib}_s0.jsonl" \
      "${OUT_DIR}/${tag}_${calib}_s1.jsonl" \
      "${OUT_DIR}/${tag}_${calib}_s2.jsonl" \
      "${OUT_DIR}/${tag}_${calib}_s3.jsonl" \
      > "${out_jsonl}"
}

run_choice_only() {
  local calib=$1
  local tag="choice_only"
  local out_jsonl="${OUT_DIR}/${tag}_${calib}.jsonl"
  local out_csv="${OUT_DIR}/${tag}_${calib}.csv"

  if [[ "${FORCE}" == "1" || ! -s "${out_jsonl}" ]]; then
    run_sharded run_vlm_choice_only.py "${tag}" "${calib}"
    merge_shards "${tag}" "${calib}"
  else
    echo "[SKIP] ${tag} | calibration=${calib} (found ${out_jsonl})"
  fi

  if [[ -s "${out_jsonl}" ]]; then
    if [[ "${FORCE}" == "1" || ! -s "${out_csv}" ]]; then
      python analyze_choice_only.py \
        --results_jsonl "${out_jsonl}" \
        --out_csv "${out_csv}"
    else
      echo "[SKIP] ${tag} analysis (found ${out_csv})"
    fi
  else
    echo "[SKIP] ${tag} analysis (missing ${out_jsonl})"
  fi
}

run_choice_color() {
  local calib=$1
  local tag="choice_color"
  local out_jsonl="${OUT_DIR}/${tag}_${calib}.jsonl"
  local out_csv="${OUT_DIR}/${tag}_${calib}.csv"

  if [[ "${FORCE}" == "1" || ! -s "${out_jsonl}" ]]; then
    run_sharded run_vlm_choice_color.py "${tag}" "${calib}"
    merge_shards "${tag}" "${calib}"
  else
    echo "[SKIP] ${tag} | calibration=${calib} (found ${out_jsonl})"
  fi

  if [[ -s "${out_jsonl}" ]]; then
    if [[ "${FORCE}" == "1" || ! -s "${out_csv}" ]]; then
      python analyze_choice_color.py \
        --results_jsonl "${out_jsonl}" \
        --out_csv "${out_csv}"
    else
      echo "[SKIP] ${tag} analysis (found ${out_csv})"
    fi
  else
    echo "[SKIP] ${tag} analysis (missing ${out_jsonl})"
  fi
}

echo "[2/5] Choice-only baseline"
run_choice_only none

echo "[3/5] Choice-only calibrated"
run_choice_only per_image_logit_bias

echo "[4/5] Choice+Color baseline"
run_choice_color none

echo "[5/5] Choice+Color calibrated"
run_choice_color per_image_logit_bias

echo "[DONE] See ${OUT_DIR}/"
