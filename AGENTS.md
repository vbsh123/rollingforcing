# Agent Notes

GitHub CLI auth for this workspace is stored in the shared WSL config directory:

```bash
export GH_CONFIG_DIR=/mnt/c/Users/User/.ghconfig
```

Use that `GH_CONFIG_DIR` before running `gh auth status`, `gh auth login`, `gh repo create`, or `git push` from future sessions. The active token is in the `hosts.yml` file under that config directory, so later sessions should reuse the same path rather than creating a new auth location.

## RollingForcing VBench Baseline Reference

Baseline videos used for rollback comparisons:

```text
rollback_dataset/rollback-dino-audit-v1/seed_0/baseline_videos
```

Seed 0, 12 prompts, 150 frames. Aggregate VBench scores:

```text
subject_consistency:     0.9418562886907283
background_consistency:  0.9593366584521812
motion_smoothness:       0.9867401970010814
temporal_flickering:     0.978369026713901
dynamic_degree:          0.4166666666666667
```

Use these as the first sanity baseline when testing rollback experiment runs.

## TokenTrim VideoJAM-Aligned Seed 0 Reference

Prompt file:

```text
prompts/tokentrim_videojam_aligned_12.txt
```

Video folders:

```text
videos/tokentrim_videojam_baseline_seed0
videos/tokentrim_videojam_paper_seed0
videos/tokentrim_videojam_rollback1_seed0
```

Plain paper TokenTrim config:

```text
configs/vast_tokentrim_paper_1_3b.yaml
```

Retrospective rollback-1 TokenTrim config:

```text
configs/vast_tokentrim_paper_rollback1_1_3b.yaml
```

Rollback-1 parameters:

```text
p=0.10
lambda=2.0
warmup=2
tokentrim_max_rerolls=0
trigger=tokentrim
rollback_depths=[1]
rollback_interventions=[suppress]
rollback_max_attempts=1
include_original=false
selector=drift
checkpoint_count=1
```

Aggregate VBench scores observed so far:

```text
motion_smoothness:
  baseline:        0.9871062105070253
  plain TokenTrim: 0.9873888996985402
  rollback-1:      0.9878116168426397

temporal_flickering:
  baseline:        0.9748611186064925
  plain TokenTrim: 0.9757071659455891
  rollback-1:      0.9766510498289969

subject_consistency:
  baseline:        0.9126296128110931
  plain TokenTrim: 0.9140908030013843
  rollback-1:      0.9082649997758452
```

## TokenTrim Guarded vs Forced Rollback Runs

Do not mix these folders in the same result table without labeling the semantics.

Guarded rollback means: try the retrospective rollback repair, but if the repaired
candidate still triggers TokenTrim, keep the original current trajectory instead.
These folders were generated from the accidental guarded variant:

```text
videos/tokentrim_videojam_rollback1_soft08_seed1_rerun
videos/tokentrim_videojam_rollback2_soft08_seed1_rerun
videos/tokentrim_videojam_rollback2_soft08_seed2_rerun
```

Forced rollback means: apply the rollback once even if the repaired candidate still
triggers TokenTrim. Use new output folders for this corrected/intended variant:

```text
videos/tokentrim_videojam_rollback1_soft08_forced_seed1
videos/tokentrim_videojam_rollback2_soft08_forced_seed1
videos/tokentrim_videojam_rollback2_soft08_forced_seed2
```

The first commit with corrected forced semantics is:

```text
5cde17a Commit forced TokenTrim rollback once
```

## Vast Instance Setup Cheat Sheet

On a fresh Vast instance:

```bash
cd /workspace/RollingForcing
source rfenv/bin/activate
git fetch origin
git checkout tokentrim-retro-soft
git pull
bash scripts/setup_vast_env.sh
git log --oneline -5
nvidia-smi
wc -l prompts/tokentrim_videojam_aligned_20_b.txt
```

Expected code state for the forced rollback experiments:

```text
5cde17a Commit forced TokenTrim rollback once
```

Use `--mode custom_input` for VBench. Without it, VBench expects full benchmark
prompt-named videos and prints missing-video warnings.

## TokenTrim 20-Prompt Extension

Second-batch prompt file:

```text
prompts/tokentrim_videojam_aligned_20_b.txt
```

Combined 32-prompt file:

```text
prompts/tokentrim_videojam_aligned_32.txt
```

For the 20-prompt extension, run baseline, plain TokenTrim, and forced
rollback-1 soft suppression:

```text
videos/tokentrim_videojam_baseline_extra20_seed0
videos/tokentrim_videojam_paper_extra20_seed0
videos/tokentrim_videojam_rollback1_soft08_forced_extra20_seed0
```

## Latest TokenTrim Result Ledger

Valid 12-prompt seed2 comparison:

```text
videos/tokentrim_videojam_baseline_seed2
  motion_smoothness      0.9881668390
  temporal_flickering    0.9769732198
  subject_consistency    0.8875581974
  background_consistency 0.9326411510

videos/tokentrim_videojam_paper_seed2
  motion_smoothness      0.9879541125
  temporal_flickering    0.9765869315
  subject_consistency    0.8864720697
  background_consistency 0.9306903472

videos/tokentrim_videojam_rollback1_soft08_forced_seed2
  motion_smoothness      0.9875962821
  temporal_flickering    0.9769371821
  subject_consistency    0.9274101722
  background_consistency 0.9601846648
```

Seed2 forced rollback-1 deltas:

```text
vs baseline:
  motion_smoothness      -0.0005705569
  temporal_flickering    -0.0000360377
  subject_consistency    +0.0398519748
  background_consistency +0.0275435138

vs plain TokenTrim:
  motion_smoothness      -0.0003578303
  temporal_flickering    +0.0003502506
  subject_consistency    +0.0409381025
  background_consistency +0.0294943176
```
