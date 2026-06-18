#!/usr/bin/env bash
set -euo pipefail

PROMPT_FILE="${PROMPT_FILE:-prompts/rollback_audit_prompts.txt}"
SEEDS="${SEEDS:-0 1}"
NUM_FRAMES="${NUM_FRAMES:-150}"
ROLLBACK_DINO_TRIGGER_Z="${ROLLBACK_DINO_TRIGGER_Z:-3.0}"
RUN_ID="${RUN_ID:-rollback-depth-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-rollback_dataset/${RUN_ID}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/rolling_forcing_dmd.pt}"
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-configs/vast_rollback_trigger_dataset_1_3b.yaml}"
BASE_CONFIG="${BASE_CONFIG:-configs/vast_base_1_3b.yaml}"
RUN_BASELINE="${RUN_BASELINE:-1}"
AZURE_STORAGE_ACCOUNT="${AZURE_STORAGE_ACCOUNT:-}"
AZURE_CONTAINER="${AZURE_CONTAINER:-}"
AZURE_PREFIX="${AZURE_PREFIX:-rollback-dataset}"

mkdir -p "${RUN_ROOT}"
cp "${PROMPT_FILE}" "${RUN_ROOT}/prompts.txt"

for seed in ${SEEDS}; do
  shard="${RUN_ROOT}/seed_${seed}"
  mkdir -p "${shard}"

  if [[ -f "${shard}/DONE" ]]; then
    echo "Skipping completed shard: ${shard}"
    continue
  fi

  export ROLLBACK_RUN_ID="${RUN_ID}"
  export ROLLBACK_SEED="${seed}"
  export ROLLBACK_MANIFEST_PATH="${shard}/events.jsonl"
  export ROLLBACK_DINO_TRIGGER_Z
  rm -f "${ROLLBACK_MANIFEST_PATH}"

  PYTHONUNBUFFERED=1 python inference.py \
    --config_path "${EXPERIMENT_CONFIG}" \
    --output_folder "${shard}/experiment_videos" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --data_path "${PROMPT_FILE}" \
    --num_output_frames "${NUM_FRAMES}" \
    --seed "${seed}" \
    --num_samples 1 \
    --reset_seed_per_prompt \
    --save_with_index \
    --use_ema 2>&1 | tee "${shard}/experiment.log"

  if [[ "${RUN_BASELINE}" == "1" ]]; then
    PYTHONUNBUFFERED=1 python inference.py \
      --config_path "${BASE_CONFIG}" \
      --output_folder "${shard}/baseline_videos" \
      --checkpoint_path "${CHECKPOINT_PATH}" \
      --data_path "${PROMPT_FILE}" \
      --num_output_frames "${NUM_FRAMES}" \
      --seed "${seed}" \
      --num_samples 1 \
      --reset_seed_per_prompt \
      --save_with_index \
      --use_ema 2>&1 | tee "${shard}/baseline.log"
  fi

  touch "${shard}/DONE"

  python scripts/summarize_rollback_manifest.py \
    "${ROLLBACK_MANIFEST_PATH}" \
    --json-out "${shard}/summary.json" \
    2>&1 | tee "${shard}/summary.txt"

  if [[ -n "${AZURE_STORAGE_ACCOUNT}" && -n "${AZURE_CONTAINER}" ]]; then
    az storage blob upload-batch \
      --account-name "${AZURE_STORAGE_ACCOUNT}" \
      --destination "${AZURE_CONTAINER}" \
      --destination-path "${AZURE_PREFIX}/${RUN_ID}/seed_${seed}" \
      --source "${shard}" \
      --overwrite \
      --auth-mode login
  fi
done

python scripts/summarize_rollback_manifest.py \
  "${RUN_ROOT}"/seed_*/events.jsonl \
  --json-out "${RUN_ROOT}/summary.json" \
  2>&1 | tee "${RUN_ROOT}/summary.txt"
