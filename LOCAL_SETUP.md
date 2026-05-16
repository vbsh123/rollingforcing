# Local RTX 3080 Setup Notes

This repo is patched to optionally import the local TokenTrim package from `../TokenTrim`.

## Current Status

This shell is WSL2. `nvidia-smi` currently reports:

```text
GPU access blocked by the operating system
```

So the first blocker is WSL GPU visibility, not Rolling Forcing code.

## Setup, Without Running Inference

Rolling Forcing recommends Python 3.10. If `python3.10` is available:

```bash
cd /mnt/c/Users/User/code/RollingForcing
bash scripts/setup_local_env.sh
```

Then check CUDA:

```bash
source .venv/bin/activate
python scripts/check_local_gpu.py
```

Only after CUDA is visible, download checkpoints:

```bash
bash scripts/download_checkpoints.sh
```

## Baseline Smoke Command

Do not run this until GPU and checkpoints are ready:

```bash
python inference.py \
  --config_path configs/rolling_forcing_dmd.yaml \
  --output_folder videos/smoke_baseline \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path prompts/example_prompts.txt \
  --num_output_frames 21 \
  --use_ema
```

## TokenTrim Smoke

Set `tokentrim_enabled: true` in `configs/default_config.yaml` or make a separate config copy before running.
Use `tokentrim_sink_blocks` to keep more cache history alive around pruning, and `tokentrim_max_rerolls` to allow more reroll attempts before accepting the result.

The current TokenTrim hook supports `batch_size=1` / `num_samples=1`.
