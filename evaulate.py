# evaluate/eval_online.py
#
# Online LIBERO evaluation for miniVLA v3.
#
# Run from the repository root:
#   python evaluate/eval_online.py
#
# v3 differences from v2:
#   - MiniVLA.forward(...) requires task_idx [B]
#   - Both camera images are resized to 128x128 and ImageNet-normalised
#     for the pretrained ResNet18 visual backbones
#   - Device is selected automatically: CUDA -> MPS -> CPU

import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import robosuite.utils.transform_utils as T
from transformers import CLIPTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from data.task_map import NUM_TASKS, get_task_idx
from models.mini_vla import MiniVLA


# ---------------------------------------------------------------------------
# PyTorch compatibility patch
# ---------------------------------------------------------------------------
_original_load = torch.load


def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)


torch.load = _patched_load


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TASK_SUITE = "libero_spatial"
TASK_INDICES = list(range(1))

NUM_EPISODES = 50
MAX_STEPS = 300

CHUNK_SIZE = 16
ACTION_HORIZON = CHUNK_SIZE

SEQ_LEN = 77
MODEL_IMAGE_SIZE = 128

# Render at 256x256 for videos, then resize model inputs to 128x128.
CAMERA_H = 256
CAMERA_W = 256

CHECKPOINT = "checkpoints/ckpt_epoch020.pt"
GENERATE_VIDEO = True
VIDEO_DIR = "videos"

# Must match train3.py / v3 architecture.
DIM_MODEL = 256
STATE_DIM = 8
ACTION_DIM = 7
NHEAD = 4

# ImageNet preprocessing required by pretrained ResNet18.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"[Eval] Using device: {DEVICE}")


# ---------------------------------------------------------------------------
# Tokenizer
# Must stay consistent with LiberoDataset._tokenize().
# ---------------------------------------------------------------------------
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")


def tokenize_instruction(text: str):
    enc = tokenizer(
        text,
        max_length=SEQ_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    tokens = enc["input_ids"].long()
    text_mask = enc["attention_mask"].float()
    return tokens, text_mask


# ---------------------------------------------------------------------------
# Observation preprocessing
# ---------------------------------------------------------------------------
def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """
    Converts a LIBERO RGB image to the visual input expected by v3.

    Input:
        image: [H, W, 3], uint8 RGB

    Output:
        tensor: [1, 3, 128, 128], float32, ImageNet-normalised
    """
    tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float()
    tensor = tensor.unsqueeze(0) / 255.0

    tensor = F.interpolate(
        tensor,
        size=(MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )

    mean = IMAGENET_MEAN.to(device=tensor.device, dtype=tensor.dtype)
    std = IMAGENET_STD.to(device=tensor.device, dtype=tensor.dtype)
    tensor = (tensor - mean) / std

    return tensor.to(DEVICE)


def observation_to_state(obs: dict) -> torch.Tensor:
    """
    Reproduces v3's 8-D proprioceptive state:
        [eef_pos(3), eef_axis_angle(3), gripper_qpos(2)]
    """
    eef_pos = obs["robot0_eef_pos"]
    eef_ori = T.quat2axisangle(obs["robot0_eef_quat"])
    gripper = obs["robot0_gripper_qpos"]

    state = np.concatenate([eef_pos, eef_ori, gripper])
    return torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = MiniVLA(
    dim_model=DIM_MODEL,
    state_dim=STATE_DIM,
    chunk_size=CHUNK_SIZE,
    action_dim=ACTION_DIM,
    nhead=NHEAD,
    num_tasks=NUM_TASKS,
    pretrained_backbone=True,
).to(DEVICE)

checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
model.load_state_dict(checkpoint)
model.eval()

print(f"[Eval] Loaded checkpoint: {CHECKPOINT}")


# ---------------------------------------------------------------------------
# LIBERO benchmark
# ---------------------------------------------------------------------------
benchmark_dict = benchmark.get_benchmark_dict()
task_suite_obj = benchmark_dict[TASK_SUITE]()

if GENERATE_VIDEO:
    os.makedirs(VIDEO_DIR, exist_ok=True)

all_results = {}

for task_index in TASK_INDICES:
    task = task_suite_obj.get_task(task_index)

    # v3's TaskEncoder requires a stable global task index.
    canonical_task_idx = get_task_idx(task.language)
    task_idx_tensor = torch.tensor(
        [canonical_task_idx],
        dtype=torch.long,
        device=DEVICE,
    )

    tokens, text_mask = tokenize_instruction(task.language)
    tokens = tokens.to(DEVICE)
    text_mask = text_mask.to(DEVICE)

    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )

    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=CAMERA_H,
        camera_widths=CAMERA_W,
    )
    env.seed(0)

    init_states = task_suite_obj.get_task_init_states(task_index)
    num_init_states = len(init_states)

    print(f"\n[Task {task_index:>2}] {task.language}")
    print(f"  Canonical task index: {canonical_task_idx}")

    successes = []

    with torch.no_grad():
        for ep_index in range(NUM_EPISODES):
            obs = env.reset()
            env.set_init_state(init_states[ep_index % num_init_states])

            ep_success = False
            action_chunk = None
            step_in_chunk = ACTION_HORIZON
            frames = []

            for step in range(MAX_STEPS):
                if GENERATE_VIDEO:
                    frames.append(obs["agentview_image"].copy())

                # Predict a fresh chunk at rollout start and every 16 steps.
                if step_in_chunk >= ACTION_HORIZON:
                    agentview = image_to_tensor(obs["agentview_image"])
                    wrist = image_to_tensor(obs["robot0_eye_in_hand_image"])
                    state = observation_to_state(obs)

                    output_actions = model(
                        image=agentview,
                        wrist_image=wrist,
                        tokens=tokens,
                        mask=text_mask,
                        state=state,
                        task_idx=task_idx_tensor,
                    )

                    action_chunk = output_actions.squeeze(0).cpu().numpy()
                    step_in_chunk = 0

                obs, reward, done, info = env.step(
                    action_chunk[step_in_chunk].tolist()
                )
                step_in_chunk += 1

                if reward > 0:
                    ep_success = True
                    break

                if done:
                    break

            successes.append(ep_success)

            status = "SUCCESS" if ep_success else "FAIL"

            if GENERATE_VIDEO and frames:
                video_path = os.path.join(
                    VIDEO_DIR,
                    f"task{task_index:02d}_ep{ep_index + 1:03d}_{status}.mp4",
                )

                height, width = frames[0].shape[:2]
                writer = cv2.VideoWriter(
                    video_path,
                    cv2.VideoWriter_fourcc(*"avc1"),
                    20,
                    (width, height),
                )

                for frame in frames:
                    # LIBERO images are vertically flipped relative to display.
                    frame = cv2.flip(frame, 0)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    writer.write(frame)

                writer.release()

            print(
                f"  Episode {ep_index + 1:>3}/{NUM_EPISODES} "
                f"| {status} | steps={step + 1}"
            )

    env.close()

    success_rate = sum(successes) / len(successes)
    all_results[task.name] = success_rate

    print(
        f"  -> Task success rate: {success_rate * 100:.1f}% "
        f"({sum(successes)}/{NUM_EPISODES})"
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print(f"  Multi-task Eval | Suite: {TASK_SUITE}")
print("=" * 65)

for name, rate in all_results.items():
    print(f"  {name:<55} {rate * 100:5.1f}%")

print("-" * 65)

if all_results:
    average_success_rate = sum(all_results.values()) / len(all_results)
    print(f"  Average{'':<50} {average_success_rate * 100:5.1f}%")

print("=" * 65)