# miniVLA-T

miniVLA-T extends `miniVLA_v3` with a moving-object trajectory token for
LIBERO-Safety. It supports either causal history or privileged future
trajectory windows. The original v3 baseline remains available through the
same training entry point for controlled comparisons.

## Model

```text
agent view ─┐
wrist view ─┤
state ──────┤
language ───┼─> modality embeddings -> transformer fusion -> action decoder
task ID ────┤
trajectory ─┘  miniVLA-T only: [N frames × (x, y, dx, dy)] -> 3-layer MLP
```

`MiniVLA` has five fusion tokens. `MiniVLAT` adds a sixth token produced by a
three-linear-layer MLP in `models/trajectory_encoder.py`. History windows are
left padded; future windows are right padded. Both use a validity mask. Images
may be 128×128 or the benchmark's native 256×256 resolution.

The extension review, including remaining scalability limitations, is in
`docs/architecture_review.md`.

## Local setup

This checkout intentionally ignores the large raw benchmark, episode videos,
assets, checkpoints, and result directories.

```bash
conda env create -f environment.yml
conda activate miniVLA-T

# Clone LIBERO-Safety with sparse checkout as described below, then:
pip install -e third_party/libero_safety/third_party/robosuite-1.4
git -C third_party/libero_safety apply ../../patches/libero_safety_optional_wand.patch
python scripts/configure_libero_safety.py

export LIBERO_CONFIG_PATH="$PWD/config/libero_safety"
export PYTHONPATH="$PWD/third_party/libero_safety/third_party/robosuite-1.4:$PWD/third_party/libero_safety:$PYTHONPATH"
```

The official repository is expected at `third_party/libero_safety`. Only the
root benchmark code, vendored robosuite, and the BDDL/init files for
`obstacle_avoidance_human` and `obstacle_avoidance` are needed. Verify them with:

```bash
python scripts/configure_libero_safety.py
python scripts/smoke_libero_safety.py
python scripts/smoke_libero_safety.py --create-env --tasks all
```

## Selected L1 data

The official LeRobot release stores both agent and wrist RGB streams at
256×256, 20 FPS, using AV1/yuv420p. `IMAGE_SIZE` near the top of `train.py`
controls model preprocessing and now defaults to the native 256 resolution;
use 128 for a substantially faster, lower-memory local run. The resolved value
is saved in the checkpoint, and both offline and simulator evaluation restore
it automatically. `--image-size` can override it explicitly.

The official dataset metadata provides one split only—`train: 0:19664`—and
does not publish separate demonstration validation or test splits. The local
pilot and five-task manifests create episode-disjoint `train` and `val` subsets
for loss/MAE measurement. Closed-loop benchmark evaluation is different: it
runs trained checkpoints in the simulator on configured initial states and
reports success and collision-free rates; it does not use a third demonstration
`eval` folder.

The pilot uses **one task per suite**, not one task ID shared across the whole
experiment:

- Free-Space Hand–Object Avoidance, task 1: 20 train + 5 validation episodes.
- Tabletop Spatial Avoidance, task 1: 20 train + 5 validation episodes.

The public LeRobot metadata does not contain suite or safety-level fields. The
downloader therefore assigns the two suites using conservative episode-index
blocks cross-checked against representative videos. This is suitable for the
local pilot, but the inferred mapping must be validated against an authoritative
release mapping before reporting benchmark-comparable results.

```bash
# One task in each suite (50 episodes total)
python scripts/download_libero_safety_subset.py \
  --tasks 1 --train-per-task 20 --val-per-task 5 \
  --output data/libero_safety/pilot

python scripts/extract_object_trajectories.py \
  --manifest data/libero_safety/pilot/manifest.jsonl

# Quality-control videos
python scripts/extract_object_trajectories.py \
  --manifest data/libero_safety/pilot/manifest.jsonl \
  --limit 2 --overlay-dir results/trajectory_overlays
```

To expand to **all five tasks in each suite**, the local preset uses 10 train +
2 held-out episodes per task (120 episodes total):

```bash
python scripts/download_libero_safety_subset.py \
  --tasks all --train-per-task 10 --val-per-task 2 \
  --output data/libero_safety/five_tasks
python scripts/extract_object_trajectories.py \
  --manifest data/libero_safety/five_tasks/manifest.jsonl

# Use episode-aware batching so 100 training videos are not repeatedly decoded.
python train.py --model v3 \
  --manifest data/libero_safety/five_tasks/manifest.jsonl \
  --output-dir results/five_tasks_v3 --epochs 6 --frame-stride 16 \
  --episode-batching
python train.py --model t \
  --manifest data/libero_safety/five_tasks/manifest.jsonl \
  --output-dir results/five_tasks_t --epochs 6 --frame-stride 16 \
  --episode-batching --trajectory-mode history --trajectory-length 8
```

`scripts/download_libero_safety_assets.py` range-reads the official 10.7 GB ZIP
and extracts only the assets used by the selected tasks. The local default now
covers all five L1 tasks in both requested suites:

```bash
python scripts/download_libero_safety_assets.py --tasks all --prune
```

The verified subset contains 2,443 files (445.8 MiB expanded). All five L1
tasks in both suites pass environment construction, reset, and initial-state
loading with this subset.

## Training and evaluation

Both comparison runs use identical episode splits, sampling, normalization,
architecture size, and seed. The reproduced results below used six epochs;
the editable `NUM_EPOCHS` value near the top of `train.py` controls new runs.

The training scope is also editable at the top of `train.py` through
`TRAINING_SUITES` and `TASK_IDS`; the number of trained suite/task combinations
is `len(TRAINING_SUITES) * len(TASK_IDS)`. CLI values override them. The pilot
manifest supports task 1 in both suites, while the five-task manifest supports
tasks 1-5 in both suites:

```bash
# One task in each suite
python train.py --manifest data/libero_safety/pilot/manifest.jsonl \
  --suites obstacle_avoidance_human obstacle_avoidance --tasks 1

# Five tasks in each suite
python train.py --manifest data/libero_safety/five_tasks/manifest.jsonl \
  --suites obstacle_avoidance_human obstacle_avoidance --tasks 1 2 3 4 5
```

`FRAME_STRIDE=1` uses every 20 FPS video frame. It gives the densest supervision
but makes adjacent samples highly correlated. For the currently selected pilot
suite/task it produces 3,598 train and 904 validation samples, compared with
460/115 at stride 8. Stride 1 is reasonable for a focused one-task run; stride
8 or 16 is more practical for the five-task expansion. `--checkpoint-every N`
saves a numbered checkpoint every N completed epochs in addition to `best.pt`
and `last.pt`.

Training prints the selected episode split, sample and batch counts, model and
optimizer configuration, and a legacy-style epoch table with loss, MAE,
learning rate, backbone stage, duration, and best-checkpoint markers. Offline
and simulator evaluation print per-task and overall result tables.

```bash
python train.py --model v3 \
  --manifest data/libero_safety/pilot/manifest.jsonl \
  --output-dir results/pilot_v3 --epochs 6 --frame-stride 8

python train.py --model t \
  --manifest data/libero_safety/pilot/manifest.jsonl \
  --output-dir results/pilot_t --epochs 6 --frame-stride 8 \
  --trajectory-mode history --trajectory-length 8

python evaluate_offline.py --checkpoint results/pilot_v3/best.pt \
  --manifest data/libero_safety/pilot/manifest.jsonl \
  --output results/evaluation/v3.json
python evaluate_offline.py --checkpoint results/pilot_t/best.pt \
  --manifest data/libero_safety/pilot/manifest.jsonl \
  --output results/evaluation/t.json
python scripts/compare_results.py --v3 results/evaluation/v3.json \
  --trajectory results/evaluation/t.json --output-dir results/comparison
```

### Privileged future-trajectory training

The full extracted track is already stored in every episode's `trajectory.npz`;
no new data download or extraction is required. At observation frame `t`,
future mode supplies frames `t+1` through `t+N`. Missing frames at the end of
an episode are right padded and masked. Train 8-frame and 16-frame variants in
separate directories:

```bash
python train.py --model t \
  --manifest data/libero_safety/five_tasks/manifest.jsonl \
  --suites obstacle_avoidance_human obstacle_avoidance --tasks 1 2 3 4 5 \
  --output-dir results/five_tasks_t_future8 --epochs 6 --frame-stride 16 \
  --episode-batching --trajectory-mode future --trajectory-length 8

python train.py --model t \
  --manifest data/libero_safety/five_tasks/manifest.jsonl \
  --suites obstacle_avoidance_human obstacle_avoidance --tasks 1 2 3 4 5 \
  --output-dir results/five_tasks_t_future16 --epochs 6 --frame-stride 16 \
  --episode-batching --trajectory-mode future --trajectory-length 16
```

Offline evaluation restores the trajectory mode and length from the checkpoint.
Closed-loop simulator evaluation intentionally rejects future-trajectory
checkpoints until an explicitly labeled oracle trajectory source is connected.

The comparison reports held-out imitation MSE and action MAE. These are not
simulator success or collision rates; simulator evaluation should be reported
separately after the asset smoke test passes.

Closed-loop success/collision evaluation is available after exporting the two
LIBERO paths from setup:

```bash
python evaluate_simulator.py --checkpoint results/pilot_t/best.pt \
  --tasks 1 --rollouts 5 --action-horizon 12 \
  --save-every 5 --generate-video --resume \
  --output results/simulator/pilot_t.json \
  --video-dir results/simulator/videos/pilot_t
```

`train.py`, `evaluate_offline.py`, and `evaluate_simulator.py` each have an
editable configuration section near the top, matching the legacy scripts.
CLI arguments remain available and override those values. For a full five-task
closed-loop run, the simulator evaluator defaults to the five-task miniVLA-T
checkpoint, both suites, tasks 1-5, and 10 rollouts per suite/task.
The evaluator terminates immediately on a reported safety cost and only counts
goal completion before a collision as success.

`ACTION_HORIZON` controls how many sequential actions are consumed from each
predicted chunk before the policy replans; it must not exceed the checkpoint's
chunk size. miniVLA-T still updates its causal trajectory history on every
simulator step. `SAVE_EVERY_ROLLOUTS` atomically checkpoints JSON and CSV
results, and `RESUME` skips rollout keys already saved after verifying that the
checkpoint and evaluation settings match. Videos are streamed to MP4 during
each rollout and renamed with their final `success`, `collision`, `horizon`, or
`interrupted` status. These settings can also be edited directly in the
configuration section near the top of `evaluate_simulator.py`.

### Current local pilot result

Six epochs, seed 42, frame stride 8, 40 train and 10 complete held-out
episodes. Lower is better for the offline columns.

| Model | Normalized action MSE | Raw action MAE | Simulator success | Collision-free |
|---|---:|---:|---:|---:|
| miniVLA v3 | 0.508543 | 0.040266 | 0/2 | 2/2 |
| miniVLA-T | 0.518726 | 0.041152 | 0/2 | 2/2 |

The simulator figures are one rollout per suite and are only a smoke-scale
estimate. The trajectory token did not improve this small pilot; more episodes,
multiple seeds, and longer training are needed before drawing a model-level
conclusion. The generated files are under `results/comparison`.

The all-five-task run used 100 train and 20 held-out episodes, six epochs,
frame stride 16, and the same seed/model size:

| Model | Normalized action MSE | Raw action MAE |
|---|---:|---:|
| miniVLA v3 | 0.695600 | 0.074837 |
| miniVLA-T | 0.695117 | 0.073594 |

miniVLA-T is marginally better in this expansion (0.07% lower MSE and 1.66%
lower MAE), but a multi-seed run is still required for a reliable conclusion.
This five-task comparison is held-out offline evaluation across all five tasks
in both suites. The required assets are now available for a later full
closed-loop evaluation. The generated files are under
`results/comparison_five_tasks`.

## Tests

```bash
pytest -q tests/test_trajectory_model.py
```

Legacy `train3.py` and `evaulate.py` are retained for v3 compatibility. New
work should use `train.py`, `evaluate_offline.py`, and the scripts above.

## Upstream references

- [LIBERO-Safety benchmark](https://github.com/LIBERO-SAFETY/LIBERO-Safety)
- [Official LeRobot dataset](https://huggingface.co/datasets/LIBERO-Safety/libero_safety)
- [Official simulator assets](https://huggingface.co/datasets/LIBERO-Safety/libero_safety_assets)
- [LIBERO-Safety paper](https://arxiv.org/abs/2606.23686)
# miniVLA-T
