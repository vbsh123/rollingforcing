# Kaggle Smoke Test Plan

This is the lowest-risk path for testing Rolling Forcing / Wan on Kaggle.

## 1. Create The Notebook

1. Go to Kaggle -> Code -> New Notebook.
2. In notebook settings, enable:
   - Accelerator: GPU
   - Internet: On
3. Use a fresh session before running model downloads.

## 2. Get This Repo Into Kaggle

Use one of these:

- Best for current local edits: zip this folder and upload it as a Kaggle Dataset.
- Best for repeatability: push this branch/fork to GitHub and clone it in Kaggle.

If using a zip dataset, unzip it in the first cell:

```bash
!mkdir -p /kaggle/working/RollingForcing
!cp -r /kaggle/input/rollingforcing/* /kaggle/working/RollingForcing/
%cd /kaggle/working/RollingForcing
```

If using GitHub:

```bash
!git clone https://github.com/YOUR_USER/RollingForcing.git /kaggle/working/RollingForcing
%cd /kaggle/working/RollingForcing
```

## 3. Check GPU

```bash
!nvidia-smi
```

In Python:

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_properties(0).total_memory / 1024**3, "GiB")
```

## 4. Install Dependencies

Start with the repo requirements. If `flash-attn` fails, skip it for the first smoke test unless the import path requires it.

```bash
!pip install -q -U pip setuptools wheel
!pip install -q -r requirements.txt
!pip install -q huggingface_hub
```

If testing the local TokenTrim hook, install TokenTrim too. Upload it as a second Kaggle Dataset or clone/push it separately.

```bash
# Example if TokenTrim was uploaded as a dataset:
!pip install -e /kaggle/input/tokentrim
```

## 5. Download Wan First

Do this before downloading the Rolling Forcing checkpoint.

```bash
!huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir-use-symlinks False \
  --local-dir wan_models/Wan2.1-T2V-1.3B
```

Expected size: about 17.6 GB.

## 6. Download Rolling Forcing Checkpoint

Only do this after the Wan download and basic imports work.

```bash
!huggingface-cli download TencentARC/RollingForcing checkpoints/rolling_forcing_dmd.pt \
  --local-dir .
```

Expected size: about 17 GB.

## 7. Make A Tiny Prompt File

```bash
!mkdir -p prompts
!printf 'a quiet cinematic shot of a red car driving down an empty road at sunset\n' > prompts/kaggle_smoke.txt
```

## 8. Baseline Smoke Run

Start with TokenTrim disabled.

```bash
!python inference.py \
  --config_path configs/rolling_forcing_dmd.yaml \
  --output_folder videos/kaggle_baseline \
  --checkpoint_path checkpoints/rolling_forcing_dmd.pt \
  --data_path prompts/kaggle_smoke.txt \
  --num_output_frames 21 \
  --use_ema \
  --save_with_index
```

## 9. Show Result

```python
from IPython.display import Video, display
display(Video("/kaggle/working/RollingForcing/videos/kaggle_baseline/0-0_ema.mp4", embed=True))
```

## 10. First Ablations

After baseline works:

1. `tokentrim_enabled: true`
2. `tokentrim_sink_blocks: 1`
3. `tokentrim_sink_blocks: 2`
4. `tokentrim_max_rerolls: 2`
5. rollback/checkpoint-memory variant after the baseline and current TokenTrim path are stable

Keep each run to one prompt and 21 frames until memory behavior is known.
