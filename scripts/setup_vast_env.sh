#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
RF_VENV_DIR="${RF_VENV_DIR:-rfenv}"
VBENCH_VENV_DIR="${VBENCH_VENV_DIR:-vbenv}"

DOWNLOAD_CHECKPOINTS="${DOWNLOAD_CHECKPOINTS:-1}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"
PATCH_SDPA_FALLBACK="${PATCH_SDPA_FALLBACK:-0}"
INSTALL_VBENCH="${INSTALL_VBENCH:-1}"
CREATE_VAST_CONFIGS="${CREATE_VAST_CONFIGS:-1}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "$PYTHON_BIN was not found. Set PYTHON_BIN=/path/to/python before running this script."
  exit 1
fi

echo "Creating RollingForcing environment: $RF_VENV_DIR"
"$PYTHON_BIN" -m venv "$RF_VENV_DIR"
source "$RF_VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# Vast images often provide Python 3.12. The repo requirements pin numpy==1.24.4,
# which does not support Python 3.12, and TensorRT/PyCUDA packages are not needed
# for these inference experiments. Keep this list explicit so setup is repeatable.
python -m pip install \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  numpy==1.26.4 \
  opencv-python-headless==4.10.0.84 \
  diffusers==0.31.0 \
  "transformers>=4.49.0,<5" \
  "tokenizers>=0.20.3" \
  "accelerate>=1.1.1" \
  tqdm imageio easydict ftfy dashscope imageio-ffmpeg wandb omegaconf einops \
  av==13.1.0 open_clip_torch starlette pycocotools lmdb matplotlib sentencepiece \
  pydantic==2.10.6 scikit-image "huggingface_hub[cli]" dominate \
  flask flask-socketio torchao tensorboard ninja packaging "gradio>=4.44.0"

if [[ "$INSTALL_FLASH_ATTN" == "1" ]]; then
  echo "Installing flash-attn. If this fails, rerun with INSTALL_FLASH_ATTN=0 PATCH_SDPA_FALLBACK=1."
  python -m pip install flash-attn --no-build-isolation
fi

deactivate

if [[ "$PATCH_SDPA_FALLBACK" == "1" ]]; then
  echo "Patching Wan model import to use the SDPA attention fallback."
  python - <<'PY'
from pathlib import Path

p = Path("wan/modules/model.py")
s = p.read_text()
old = "from .attention import flash_attention"
new = "from .attention import attention as flash_attention"
if old in s:
    p.write_text(s.replace(old, new))
    print(f"patched {p}")
else:
    print(f"{p} already patched or import pattern not found")
PY
fi

if [[ "$DOWNLOAD_CHECKPOINTS" == "1" ]]; then
  echo "Downloading Wan 1.3B base model and RollingForcing/Self-Forcing checkpoints."
  source "$RF_VENV_DIR/bin/activate"
  huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
    --local-dir-use-symlinks False \
    --local-dir wan_models/Wan2.1-T2V-1.3B
  huggingface-cli download TencentARC/RollingForcing checkpoints/rolling_forcing_dmd.pt \
    --local-dir .
  huggingface-cli download gdhe17/Self-Forcing checkpoints/ode_init.pt \
    --local-dir .
  deactivate
fi

if [[ "$CREATE_VAST_CONFIGS" == "1" ]]; then
  echo "Creating Vast experiment configs and prompt file."
  mkdir -p prompts
  if [[ ! -f prompts/vast_eval_5prompts.txt ]]; then
    printf '%s\n' \
      "a helicopter flying over a forest" \
      "a slime humanoid ripples as it walks" \
      "a llama dances on a stage with disco lights" \
      "a robot learning to walk" \
      "a bright red kite pulling forward against a light blue sky" \
      > prompts/vast_eval_5prompts.txt
  fi

  if [[ ! -f configs/vast_base_1_3b.yaml ]]; then
    cp configs/rolling_forcing_dmd.yaml configs/vast_base_1_3b.yaml
    perl -0pi -e 's/real_name: Wan2\.1-T2V-14B/real_name: Wan2.1-T2V-1.3B/' configs/vast_base_1_3b.yaml
    perl -0pi -e 's#data_path: .*#data_path: prompts/vast_eval_5prompts.txt#' configs/vast_base_1_3b.yaml
  fi

  if [[ ! -f configs/vast_tokentrim_baseline_1_3b.yaml ]]; then
    cp configs/vast_base_1_3b.yaml configs/vast_tokentrim_baseline_1_3b.yaml
    printf '\n' >> configs/vast_tokentrim_baseline_1_3b.yaml
    cat >> configs/vast_tokentrim_baseline_1_3b.yaml <<'EOF'
tokentrim_enabled: true
tokentrim_pruning_fraction: 0.30
tokentrim_lambda_threshold: 0.5
tokentrim_warmup_steps: 0
tokentrim_sink_blocks: 1
tokentrim_max_rerolls: 1
tokentrim_debug: true
EOF
  fi

  if [[ ! -f configs/vast_tokentrim_rollback2_suppress_1_3b.yaml ]]; then
    cp configs/vast_base_1_3b.yaml configs/vast_tokentrim_rollback2_suppress_1_3b.yaml
    printf '\n' >> configs/vast_tokentrim_rollback2_suppress_1_3b.yaml
    cat >> configs/vast_tokentrim_rollback2_suppress_1_3b.yaml <<'EOF'
tokentrim_enabled: true
tokentrim_pruning_fraction: 0.30
tokentrim_lambda_threshold: 0.5
tokentrim_warmup_steps: 0
tokentrim_sink_blocks: 1
tokentrim_max_rerolls: 0
tokentrim_debug: true

tokentrim_rollback_windows: 2
tokentrim_rollback_max_attempts: 4
tokentrim_rollback_experimental: true
tokentrim_rollback_suppress_cache: false
tokentrim_rollback_reset_rng: false

tokentrim_checkpoint_count: 2
tokentrim_checkpoint_device: cpu
tokentrim_checkpoint_interval: 1

tokentrim_rollback_depths:
  - 1
  - 2
tokentrim_rollback_interventions:
  - none
  - suppress
tokentrim_rollback_best_of_n: 1
EOF
  fi

  if [[ ! -f configs/vast_tokentrim_rollback_softsearch_1_3b.yaml ]]; then
    cp configs/vast_base_1_3b.yaml configs/vast_tokentrim_rollback_softsearch_1_3b.yaml
    printf '\n' >> configs/vast_tokentrim_rollback_softsearch_1_3b.yaml
    cat >> configs/vast_tokentrim_rollback_softsearch_1_3b.yaml <<'EOF'
tokentrim_enabled: true
tokentrim_pruning_fraction: 0.15
tokentrim_lambda_threshold: 0.5
tokentrim_warmup_steps: 0
tokentrim_sink_blocks: 1
tokentrim_max_rerolls: 0
tokentrim_debug: true
tokentrim_trigger_mode: rate_anomaly

tokentrim_rollback_windows: 2
tokentrim_rollback_max_attempts: 2
tokentrim_rollback_experimental: true
tokentrim_rollback_suppress_cache: false
tokentrim_rollback_reset_rng: false
tokentrim_rollback_include_original: true
tokentrim_rollback_selector: rate_anomaly
tokentrim_selector_subject_weight: 1.0
tokentrim_selector_boundary_weight: 1.0
tokentrim_selector_motion_weight: 0.5
tokentrim_selector_drift_weight: 0.05
tokentrim_selector_rate_weight: 1.0
tokentrim_selector_context_frames: 6
tokentrim_rate_history_size: 8
tokentrim_rate_warmup_steps: 6
tokentrim_rate_z_threshold: 8.0
tokentrim_rate_top_fraction: 0.10

tokentrim_checkpoint_count: 2
tokentrim_checkpoint_device: cpu
tokentrim_checkpoint_interval: 1

tokentrim_rollback_depths:
  - 1
  - 2
tokentrim_rollback_interventions:
  - none
  - rate_normalize:0.5
  - soft_suppress:0.8
tokentrim_rollback_best_of_n: 1
EOF
  fi
fi

if [[ "$INSTALL_VBENCH" == "1" ]]; then
  echo "Creating VBench environment: $VBENCH_VENV_DIR"
  "$PYTHON_BIN" -m venv "$VBENCH_VENV_DIR"
  source "$VBENCH_VENV_DIR/bin/activate"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install --no-deps vbench==0.1.5
  python -m pip install \
    decord==0.6.0 opencv-python-headless==4.10.0.84 \
    scipy scikit-image scikit-learn pandas matplotlib seaborn \
    tqdm easydict omegaconf ftfy regex yacs iopath fvcore fairscale filterpy \
    openai-clip timm==1.0.12 transformers==4.57.6 "tokenizers>=0.20.3" \
    boto3 cython lvis pycocoevalcap tensorboard
  python -m pip install --no-deps pyiqa
  python -m pip install \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    numpy==1.26.4

  python - <<'PY'
from pathlib import Path

import vbench.motion_smoothness as ms

p = Path(ms.__file__)
s = p.read_text()
old = 'ckpt = torch.load(ckpt_path, map_location="cpu")'
new = 'ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)'
if old in s:
    p.write_text(s.replace(old, new))
    print(f"patched {p} for PyTorch 2.6+ checkpoint loading behavior")
else:
    print(f"{p} patch not needed or pattern not found")
PY
  deactivate
fi

cat <<EOF

Vast setup complete.

Generation env:
  source $RF_VENV_DIR/bin/activate

VBench env:
  source $VBENCH_VENV_DIR/bin/activate

Useful smoke command:
  python inference.py \\
    --config_path configs/vast_base_1_3b.yaml \\
    --output_folder videos/vast_smoke_baseline \\
    --checkpoint_path checkpoints/rolling_forcing_dmd.pt \\
    --data_path prompts/vast_eval_5prompts.txt \\
    --num_output_frames 30 \\
    --use_ema
EOF
