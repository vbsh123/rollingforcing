#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir-use-symlinks False \
  --local-dir wan_models/Wan2.1-T2V-1.3B

huggingface-cli download TencentARC/RollingForcing checkpoints/rolling_forcing_dmd.pt \
  --local-dir .

cat <<'EOF'

Checkpoints downloaded.
Baseline smoke command, when ready:
  python inference.py \
    --config_path configs/rolling_forcing_dmd.yaml \
    --output_folder videos/smoke_baseline \
    --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
    --data_path prompts/example_prompts.txt \
    --num_output_frames 21 \
    --use_ema
EOF
