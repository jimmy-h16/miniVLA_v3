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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.mini_vla    import MiniVLA
from data.libero_dataset import LiberoDataset
from data.task_map      import NUM_TASKS

# ── Config ────────────────────────────────────────────────────────────────
DATA_DIR      = "./data/libero_10"     # path to your .hdf5 demo files
CHECKPOINT_DIR = "./checkpoints"
NUM_EPOCHS    = 100
BATCH_SIZE    = 64
LR            = 1e-4
WARMUP_EPOCHS = 5
PERIOD        = 10                     # checkpoint every N epochs
VAL_FRAC      = 0.1

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
print(f"[train3] Using device: {DEVICE}")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────
all_hdf5 = sorted(glob.glob(os.path.join(DATA_DIR, "*.hdf5")))
assert len(all_hdf5) > 0, f"No .hdf5 files found in {DATA_DIR}"
print(f"[train3] Found {len(all_hdf5)} HDF5 files")

full_dataset = LiberoDataset(
    hdf5_files=all_hdf5,
    chunk_size=CHUNK_SIZE,
    image_size=128,
)

val_size   = max(1, int(len(full_dataset) * VAL_FRAC))
train_size = len(full_dataset) - val_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=(DEVICE.type == "cuda"),
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=4, pin_memory=(DEVICE.type == "cuda"),
)
print(f"[train3] Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

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
print(f"[train3] Backbone stage 1 (frozen). Trainable params: {model.count_parameters():,}")

# ── Optimiser + Scheduler ─────────────────────────────────────────────────
optimiser = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
scheduler = CosineAnnealingLR(optimiser, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)

def warmup_lr(epoch: int) -> float:
    """Linear warmup for the first WARMUP_EPOCHS epochs."""
    return float(epoch + 1) / float(WARMUP_EPOCHS) if epoch < WARMUP_EPOCHS else 1.0

warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda=warmup_lr)

criterion = nn.MSELoss(reduction="none")

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

            # TextEncoder tokens are built inside MiniVLA.forward in v3;
            # we pass dummy ids here — replace with real CLIP token ids
            # once you integrate the tokeniser into the DataLoader.
            # TODO: integrate CLIP tokenisation into LiberoDataset.__getitem__
            B = image.shape[0]
            tokens = torch.zeros(B, 77, dtype=torch.long,  device=DEVICE)
            mask   = torch.ones( B, 77, dtype=torch.float, device=DEVICE)

            pred = model(
                image=image,
                wrist_image=wrist_image,
                tokens=tokens,
                mask=mask,
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


# ── Main loop ─────────────────────────────────────────────────────────────
best_val_loss = float("inf")

for epoch in range(NUM_EPOCHS):
    # ── Backbone stage transitions ─────────────────────────────────────
    if epoch == BACKBONE_STAGE2_EPOCH:
        model.set_backbone_stage(2)
        # Rebuild optimiser to include newly unfrozen params
        optimiser = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR * 0.1)
        print(f"[train3] Epoch {epoch}: backbone stage 2 (layer4 unfrozen, LR={LR*0.1:.2e})")

    elif epoch == BACKBONE_STAGE3_EPOCH:
        model.set_backbone_stage(3)
        optimiser = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR * 0.01)
        print(f"[train3] Epoch {epoch}: backbone stage 3 (full backbone, LR={LR*0.01:.2e})")

    # ── LR schedule ────────────────────────────────────────────────────
    if epoch < WARMUP_EPOCHS:
        warmup_scheduler.step()
    else:
        scheduler.step()

    # ── Train / val ────────────────────────────────────────────────────
    train_loss = run_epoch(train_loader, train=True)
    val_loss   = run_epoch(val_loader,   train=False)

    current_lr = optimiser.param_groups[0]["lr"]
    print(f"Epoch {epoch+1:03d}/{NUM_EPOCHS} | "
          f"train={train_loss:.4f} | val={val_loss:.4f} | lr={current_lr:.2e}")

    # ── Best checkpoint ────────────────────────────────────────────────
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(),
                   os.path.join(CHECKPOINT_DIR, "best_model.pt"))
        print(f"  → New best val loss: {best_val_loss:.4f} (saved)")

    # ── Periodic checkpoint ────────────────────────────────────────────
    if (epoch + 1) % PERIOD == 0:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"ckpt_epoch{epoch+1:03d}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"  → Checkpoint saved: {ckpt_path}")

print(f"[train3] Training complete. Best val loss: {best_val_loss:.4f}")
