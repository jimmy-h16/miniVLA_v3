# train3.py  —  miniVLA v3 training script
# Changes vs v2:
#   1. Device: defaults to "cuda" (AutoDL). Falls back to "mps" then "cpu".
#   2. task_idx fed to model.forward() via batch["task_idx"]
#   3. model.set_backbone_stage() called at epoch milestones (3-stage freeze protocol)
#   4. num_workers=4 (AutoDL multi-worker DataLoader)
#   5. Removed hardcoded MPS device string
#   6. train3.py is the canonical entry point (README updated to match)

import os
import glob
import json
import csv
import torch
import h5py
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.mini_vla    import MiniVLA
from data.libero_dataset import LiberoDataset
from data.task_map      import NUM_TASKS, get_task_idx

# ── Config ────────────────────────────────────────────────────────────────
# DATA_DIR      = "./data/libero_10"     # path to your .hdf5 demo files

DATA_DIR  = os.environ.get("LIBERO_DATASET_DIR", os.path.expanduser("~/.robosuite/datasets/datasets/libero_spatial"))
all_hdf5 = sorted(glob.glob(os.path.join(DATA_DIR, "*.hdf5")))
TASK_RANGE = range(1)

CHECKPOINT_DIR = "./checkpoints"
HISTORY_CSV   = os.path.join(CHECKPOINT_DIR, "training_history.csv")
LOSS_PLOT     = os.path.join(CHECKPOINT_DIR, "training_loss.png")
NUM_EPOCHS    = 50
BATCH_SIZE    = 32
LR            = 1e-4
LAYER4_LR     = 1e-5
BACKBONE_LR   = 1e-6
WARMUP_EPOCHS = 5
PERIOD        = 10                     # checkpoint every N epochs
TRAIN_EPISODES_PER_TASK = 45
VAL_EPISODES_PER_TASK   = 5
SPLIT_SEED             = 42

# Backbone freeze schedule (stages applied at these epoch boundaries)
BACKBONE_STAGE2_EPOCH = 20   # unfreeze layer4 after this many epochs
BACKBONE_STAGE3_EPOCH = 50   # full backbone unfreeze after this many epochs

# Model hyperparameters
DIM_MODEL   = 256
CHUNK_SIZE  = 16
ACTION_DIM  = 7
STATE_DIM   = 8
NHEAD       = 4

# ── Device ────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")   # M2 Mac fallback
else:
    DEVICE = torch.device("cpu")

OUTPUT_WIDTH = 72
print("=" * OUTPUT_WIDTH)
print("  miniVLA v3 — Training")
print("=" * OUTPUT_WIDTH)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────

assert 0 <= TASK_RANGE.start <= TASK_RANGE.stop <= NUM_TASKS, (
    f"TASK_RANGE must lie within [0, {NUM_TASKS}], got "
    f"range({TASK_RANGE.start}, {TASK_RANGE.stop})"
)

selected_task_indices = set(TASK_RANGE)
selected_hdf5 = []

for hdf5_path in all_hdf5:
    with h5py.File(hdf5_path, "r") as f:
        problem_info = json.loads(f["data"].attrs["problem_info"])
        instruction = problem_info["language_instruction"]
        task_idx = get_task_idx(instruction)  # canonical index: 0 to 9

    if task_idx in selected_task_indices:
        selected_hdf5.append(hdf5_path)

assert len(selected_hdf5) == len(selected_task_indices), (
    f"Expected {len(selected_task_indices)} selected task files, "
    f"but found {len(selected_hdf5)}. Check DATA_DIR and task_map.py."
)

print(f"\n[Dataset] Found {len(selected_hdf5)} selected task(s):")
for selection_index, hdf5_path in enumerate(selected_hdf5):
    print(f"  [{selection_index}] {os.path.basename(hdf5_path)}")
    

full_dataset = LiberoDataset(
    hdf5_files=selected_hdf5,
    chunk_size=CHUNK_SIZE,
    image_size=128,
)

# Split by complete demonstration episode, not individual timesteps. This
# prevents adjacent observations from the same rollout leaking into validation.
split_generator = torch.Generator().manual_seed(SPLIT_SEED)
train_episode_keys = set()
val_episode_keys = set()

for hdf5_path in selected_hdf5:
    episode_keys = sorted({
        episode_key
        for sample_path, episode_key, *_ in full_dataset.samples
        if sample_path == hdf5_path
    })
    expected_episodes = TRAIN_EPISODES_PER_TASK + VAL_EPISODES_PER_TASK
    assert len(episode_keys) == expected_episodes, (
        f"Expected {expected_episodes} episodes in {os.path.basename(hdf5_path)}, "
        f"but found {len(episode_keys)}."
    )

    permutation = torch.randperm(
        len(episode_keys), generator=split_generator
    ).tolist()
    shuffled_keys = [episode_keys[i] for i in permutation]

    train_episode_keys.update(
        (hdf5_path, key)
        for key in shuffled_keys[:TRAIN_EPISODES_PER_TASK]
    )
    val_episode_keys.update(
        (hdf5_path, key)
        for key in shuffled_keys[TRAIN_EPISODES_PER_TASK:]
    )

train_indices = []
val_indices = []
for sample_index, (hdf5_path, episode_key, *_) in enumerate(full_dataset.samples):
    episode_id = (hdf5_path, episode_key)
    if episode_id in train_episode_keys:
        train_indices.append(sample_index)
    elif episode_id in val_episode_keys:
        val_indices.append(sample_index)

assert train_episode_keys.isdisjoint(val_episode_keys)
assert len(train_indices) + len(val_indices) == len(full_dataset)

train_ds = Subset(full_dataset, train_indices)
val_ds = Subset(full_dataset, val_indices)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, pin_memory=(DEVICE.type == "cuda"),
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=(DEVICE.type == "cuda"),
)
print("\n[Dataset] Episode split per task:")
print(f"  Train episodes : {TRAIN_EPISODES_PER_TASK}")
print(f"  Val episodes   : {VAL_EPISODES_PER_TASK}")
print(f"  Split seed     : {SPLIT_SEED}")
print("\n[Dataset] Samples after chunking:")
print(f"  Train : {len(train_ds):,} samples")
print(f"  Val   : {len(val_ds):,} samples")

# ── Model ─────────────────────────────────────────────────────────────────
model = MiniVLA(
    dim_model=DIM_MODEL,
    state_dim=STATE_DIM,
    chunk_size=CHUNK_SIZE,
    action_dim=ACTION_DIM,
    nhead=NHEAD,
    num_tasks=NUM_TASKS,            # from task_map.py (10 for libero_10)
    pretrained_backbone=True,
).to(DEVICE)

# Stage 1: freeze ResNet backbone at start of training
model.set_backbone_stage(1)
print("\n[Model] MiniVLA v3")
print(f"  Trainable parameters : {model.count_parameters():,}")
print(f"  Device               : {DEVICE}")
print("  Backbone stage       : 1 (frozen)")

# ── Optimiser + Scheduler ─────────────────────────────────────────────────
# Keep one optimiser for the complete run so AdamW momentum is retained when
# backbone stages are unfrozen. Parameters with requires_grad=False remain
# untouched until model.set_backbone_stage() enables them.
image_encoders = (model.image_encoder, model.wrist_encoder)

layer4_params = []
early_backbone_params = []
for encoder in image_encoders:
    layer4_params.extend(encoder.backbone[-1].parameters())
    early_backbone_params.extend(encoder.backbone[:-1].parameters())

backbone_param_ids = {
    id(parameter)
    for encoder in image_encoders
    for parameter in encoder.backbone.parameters()
}
task_head_params = [
    parameter
    for parameter in model.parameters()
    if id(parameter) not in backbone_param_ids
]

optimiser = AdamW(
    [
        {
            "name": "task_and_heads",
            "params": task_head_params,
            "lr": LR,
        },
        {
            "name": "resnet_layer4",
            "params": layer4_params,
            "lr": LAYER4_LR,
        },
        {
            "name": "resnet_early",
            "params": early_backbone_params,
            "lr": BACKBONE_LR,
        },
    ],
    weight_decay=1e-2,
)
scheduler = CosineAnnealingLR(optimiser, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)

def warmup_lr(epoch: int) -> float:
    """Linear warmup for the first WARMUP_EPOCHS epochs."""
    return float(epoch + 1) / float(WARMUP_EPOCHS) if epoch < WARMUP_EPOCHS else 1.0

warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=warmup_lr)

criterion = nn.MSELoss(reduction="none")

print("\n[Scheduler]")
print(f"  Task, fusion and action LR : {LR:.2e}")
print(f"  ResNet layer4 LR           : {LAYER4_LR:.2e}")
print(f"  Earlier ResNet layers LR   : {BACKBONE_LR:.2e}")
print(
    f"  Linear warmup ({WARMUP_EPOCHS} epochs) -> "
    f"cosine annealing ({NUM_EPOCHS - WARMUP_EPOCHS} epochs)"
)
print(
    f"  Backbone stage 2 at epoch {BACKBONE_STAGE2_EPOCH + 1}; "
    f"stage 3 at epoch {BACKBONE_STAGE3_EPOCH + 1}"
)

# ── Training helpers ──────────────────────────────────────────────────────
def run_epoch(loader, train: bool) -> float:
    model.train(train)
    total_loss = 0.0
    n_batches  = 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            image       = batch["agentview_rgb"].to(DEVICE)
            wrist_image = batch["eye_in_hand_rgb"].to(DEVICE)
            state       = batch["robot_state"].to(DEVICE)
            action_gt   = batch["action_chunk"].to(DEVICE)   # [B, T, 7]
            action_mask = batch["action_mask"].to(DEVICE)    # [B, T]
            task_idx    = batch["task_idx"].to(DEVICE)       # [B]  long
            tokens = batch["tokens"].to(DEVICE, non_blocking=True)          # [B, 77], int64
            text_mask = batch["text_mask"].to(DEVICE, non_blocking=True)    # [B, 77], float32

            pred = model(
                image=image,
                wrist_image=wrist_image,
                tokens=tokens,
                mask=text_mask,
                state=state,
                task_idx=task_idx,
            )  # [B, chunk_size, action_dim]

            # Masked MSE: ignore padded timesteps at episode end
            loss_per_step = criterion(pred, action_gt)              # [B, T, 7]
            loss_per_step = loss_per_step.mean(dim=-1)              # [B, T]
            loss = (loss_per_step * action_mask).sum() / action_mask.sum().clamp(min=1)

            if train:
                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()

            total_loss += loss.item()
            n_batches  += 1

    return total_loss / max(n_batches, 1)


def save_training_history(history: list[dict]) -> None:
    """Atomically save completed epochs as CSV and a train/validation plot."""
    csv_tmp = HISTORY_CSV + ".tmp"
    with open(csv_tmp, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["epoch", "train_loss", "val_loss", "lr", "backbone_stage"],
        )
        writer.writeheader()
        writer.writerows(history)
    os.replace(csv_tmp, HISTORY_CSV)

    if not history:
        return

    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    val_losses = [row["val_loss"] for row in history]

    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(epochs, train_losses, label="Training loss", linewidth=2)
    axis.plot(epochs, val_losses, label="Validation loss", linewidth=2)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Masked MSE loss")
    axis.set_title("miniVLA v3 training and validation loss")
    axis.grid(True, alpha=0.3)
    axis.legend()
    if len(epochs) == 1:
        axis.set_xlim(0.5, 1.5)
    else:
        axis.set_xlim(1, epochs[-1])
    figure.tight_layout()

    plot_tmp = LOSS_PLOT + ".tmp"
    figure.savefig(plot_tmp, format="png", dpi=160)
    plt.close(figure)
    os.replace(plot_tmp, LOSS_PLOT)


# ── Main loop ─────────────────────────────────────────────────────────────
best_val_loss = float("inf")
best_ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
backbone_stage = "1/frozen"
history = []
interrupted = False

header = (
    f"{'Epoch':>9}  {'Train Loss':>11}  {'Val Loss':>10}  "
    f"{'LR':>10}  {'Backbone':>11}  {'Best?':>6}"
)
separator = "-" * len(header)

print(
    f"\n[Training] {NUM_EPOCHS} epochs | batch={BATCH_SIZE} | "
    f"initial lr={LR:.0e}"
)
print(separator)
print(header)
print(separator)

try:
    for epoch in range(NUM_EPOCHS):
        # ── Backbone stage transitions ─────────────────────────────────
        if epoch == BACKBONE_STAGE2_EPOCH:
            model.set_backbone_stage(2)
            backbone_stage = "2/layer4"
            print(separator)
            print(
                f"[Backbone] Epoch {epoch + 1}: stage 2, layer4 unfrozen "
                f"(configured lr={LAYER4_LR:.2e})"
            )
            print(separator)

        elif epoch == BACKBONE_STAGE3_EPOCH:
            model.set_backbone_stage(3)
            backbone_stage = "3/full"
            print(separator)
            print(
                f"[Backbone] Epoch {epoch + 1}: stage 3, full backbone unfrozen "
                f"(configured lr={BACKBONE_LR:.2e})"
            )
            print(separator)

        # ── Train / val ────────────────────────────────────────────────
        train_loss = run_epoch(train_loader, train=True)
        val_loss = run_epoch(val_loader, train=False)

        # ── LR schedule ────────────────────────────────────────────────
        if epoch < WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            scheduler.step()

        # ── Best checkpoint ────────────────────────────────────────────
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_ckpt_path)

        current_lr = optimiser.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr,
                "backbone_stage": backbone_stage,
            }
        )
        save_training_history(history)

        marker = "yes" if is_best else ""
        print(
            f"{epoch + 1:>3}/{NUM_EPOCHS:<3}  {train_loss:>11.5f}  "
            f"{val_loss:>10.5f}  {current_lr:>10.2e}  "
            f"{backbone_stage:>11}  {marker:>6}"
        )

        if is_best:
            print(f"          -> Saved best checkpoint: {best_ckpt_path}")

        # ── Periodic checkpoint ────────────────────────────────────────
        if (epoch + 1) % PERIOD == 0:
            ckpt_path = os.path.join(
                CHECKPOINT_DIR, f"ckpt_epoch{epoch + 1:03d}.pt"
            )
            torch.save(model.state_dict(), ckpt_path)
            print(f"          -> Periodic checkpoint: {ckpt_path}")

except KeyboardInterrupt:
    interrupted = True
    save_training_history(history)
    interrupted_ckpt_path = os.path.join(CHECKPOINT_DIR, "interrupted_model.pt")
    torch.save(model.state_dict(), interrupted_ckpt_path)
    print("\n[Stopped] Training interrupted by user.")
    print(f"  Completed epochs      : {len(history)}")
    print(f"  Interrupted checkpoint: {interrupted_ckpt_path}")

if not interrupted:
    final_ckpt_path = os.path.join(CHECKPOINT_DIR, "final_model.pt")
    torch.save(model.state_dict(), final_ckpt_path)

print(separator)
print("\n[Done] Training history saved.")
if history:
    print(f"  Best val loss   : {best_val_loss:.5f}")
    print(f"  Best checkpoint : {best_ckpt_path}")
print(f"  History CSV     : {HISTORY_CSV}")
print(f"  Loss graph      : {LOSS_PLOT}")
if not interrupted:
    print(f"  Final checkpoint: {final_ckpt_path}")
print("=" * OUTPUT_WIDTH)
