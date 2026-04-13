"""
Convert Scale and MARS robot episodes into a unified ACT dataset
using the 8D shared representation:

  [ee_x, ee_y, ee_z, pitch, yaw, grip_width, v, omega]

Both qpos (current state) and action (next state) use this format.

Usage:
  # Convert Scale episodes (exported from Colab as .npz)
  python convert_to_shared_dataset.py scale \
    --input-dir /path/to/scale_exports/ \
    --output-dir /path/to/shared_dataset/

  # Convert MARS robot episodes (existing HDF5)
  python convert_to_shared_dataset.py robot \
    --input-dir /path/to/robot_episodes/ \
    --output-dir /path/to/shared_dataset/

  # The output directory can be the same for both — episodes are numbered
  # sequentially and a single metadata.json is maintained.

Output HDF5 format (per episode):
  action:                    (T, 8)  float64 — next frame's shared representation
  observations/qpos:         (T, 8)  float64 — current frame's shared representation
  observations/images/camera_1: (T, 224, 224, 3) uint8 — egocentric camera
  observations/images/camera_2: (T, 224, 224, 3) uint8 — second camera (arm cam or duplicated)
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import h5py
import numpy as np

from shared_representation import (
    robot_fk,
    head_pose_to_base_motion,
    CAMERA_OFFSET_FROM_BASE,
)

SHARED_DIM = 8
IMAGE_SIZE = (224, 224)


# ---------------------------------------------------------------------------
# Scale → Shared HDF5
# ---------------------------------------------------------------------------

def convert_scale_episode(npz_path, output_h5_path):
    """
    Convert a single Scale episode (.npz exported from Colab) to shared HDF5.

    Expected .npz keys:
        right_ee_cam:    (T, 3+) — right hand EE xyz in camera frame
        head_poses:      (T, 7)  — head pose for v, omega
        fps:             scalar
        images:          (T, H, W, 3) uint8 — egocentric frames (optional)
        right_ee_world:  (T, 7) — if right_ee_cam not available
    """
    data = np.load(npz_path, allow_pickle=True)
    fps = float(data.get('fps', 30))

    # --- EE position in camera frame ---
    if 'right_ee_cam' in data:
        ee_cam = data['right_ee_cam']  # (T, 3) or (T, 6) with orientation
    elif 'right_ee_world' in data and 'head_poses' in data:
        from pyquaternion import Quaternion
        ee_world = data['right_ee_world']
        head_world = data['head_poses']
        T = min(len(ee_world), len(head_world))
        ee_cam = np.zeros((T, 3))
        for t in range(T):
            h_pos = head_world[t, :3]
            h_q = Quaternion(head_world[t, 3], head_world[t, 4],
                             head_world[t, 5], head_world[t, 6])
            disp = ee_world[t, :3] - h_pos
            ee_cam[t] = h_q.inverse.rotate(disp)
    else:
        raise ValueError(f"Need 'right_ee_cam' or 'right_ee_world'+'head_poses' in {npz_path}")

    # --- Orientation from EE pose ---
    if 'right_ee_world' in data:
        from pyquaternion import Quaternion
        ee_world = data['right_ee_world']
        orientations = np.zeros((len(ee_world), 2))  # pitch, yaw
        for t in range(len(ee_world)):
            q = Quaternion(ee_world[t, 3], ee_world[t, 4],
                           ee_world[t, 5], ee_world[t, 6])
            R = q.rotation_matrix
            pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
            yaw = np.arctan2(R[1, 0], R[0, 0])
            orientations[t] = [pitch, yaw]
    else:
        orientations = np.zeros((len(ee_cam), 2))

    # --- Base motion from head pose ---
    if 'head_poses' in data:
        head_poses = data['head_poses']
        base_motion = head_pose_to_base_motion(head_poses, fps=fps)
        # base_motion is T-1 frames, pad last frame
        base_motion = np.vstack([base_motion, base_motion[-1:]])
    else:
        base_motion = np.zeros((len(ee_cam), 2))

    # --- Grip width ---
    # Scale data doesn't have gripper info from EE pose alone
    # If MediaPipe was run, it would be in 'grip_widths'
    if 'grip_widths' in data:
        grip_widths = data['grip_widths']
    else:
        grip_widths = np.zeros(len(ee_cam))

    # --- Build shared representation ---
    T = min(len(ee_cam), len(orientations), len(base_motion), len(grip_widths))
    shared = np.zeros((T, SHARED_DIM))
    shared[:, 0:3] = ee_cam[:T, :3]           # ee xyz (camera frame)
    shared[:, 3] = orientations[:T, 0]         # pitch
    shared[:, 4] = orientations[:T, 1]         # yaw
    shared[:, 5] = grip_widths[:T]             # grip width
    shared[:, 6] = base_motion[:T, 0]          # v
    shared[:, 7] = base_motion[:T, 1]          # omega

    # --- qpos = 6D, action = 10D ---
    qpos = shared[:-1, :6]
    act_8d = shared[1:]
    action = np.zeros((len(act_8d), 10))
    action[:, :6] = act_8d[:, :6]
    action[:, 8] = act_8d[:, 6]   # v
    action[:, 9] = act_8d[:, 7]   # omega
    T_out = len(qpos)

    # --- Images ---
    if 'images' in data:
        images = data['images'][:T_out]
        # Resize to 224x224
        images_resized = np.zeros((T_out, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8)
        for t in range(T_out):
            images_resized[t] = cv2.resize(images[t], IMAGE_SIZE)
    else:
        # Placeholder black images
        images_resized = np.zeros((T_out, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8)

    # --- Write HDF5 ---
    with h5py.File(output_h5_path, 'w') as f:
        f.create_dataset('action', data=action.astype(np.float64))
        f.create_dataset('observations/qpos', data=qpos.astype(np.float64))
        f.create_dataset('observations/images/camera_1', data=images_resized, dtype=np.uint8)
        # Duplicate camera_1 for camera_2 (Scale only has one egocentric camera)
        f.create_dataset('observations/images/camera_2', data=images_resized, dtype=np.uint8)

    print(f"  Wrote {T_out} frames to {output_h5_path}")
    return T_out


# ---------------------------------------------------------------------------
# Robot → Shared HDF5
# ---------------------------------------------------------------------------

def convert_robot_episode(input_h5_path, output_h5_path):
    """
    Convert a single MARS robot episode (original HDF5) to shared representation.

    Input HDF5 keys:
        action:                    (T, 10 or 12) — leader command + optional v/omega
        observations/qpos:         (T, 6)  — joint angles
        observations/images/camera_1: (T, H, W, 3)
        observations/images/camera_2: (T, H, W, 3)
    """
    with h5py.File(input_h5_path, 'r') as f:
        action_raw = f['action'][:]
        qpos_raw = f['observations/qpos'][:]
        has_images = 'observations/images/camera_1' in f
        if has_images:
            img1 = f['observations/images/camera_1'][:]
            img2 = f['observations/images/camera_2'][:]

    T = len(qpos_raw)

    # --- Extract base motion from action ---
    if action_raw.shape[1] >= 12:
        cmd_vel = action_raw[:, 10:12]  # [v, omega]
    else:
        cmd_vel = np.zeros((T, 2))

    # --- Convert each frame to shared representation ---
    shared = np.zeros((T, SHARED_DIM))
    for t in range(T):
        # Physical servos: [j1, j2, j3, j4, gripper, (padding)]
        joint_angles = [
            qpos_raw[t, 0],  # joint1
            qpos_raw[t, 1],  # joint2
            qpos_raw[t, 2],  # joint3
            qpos_raw[t, 3],  # joint4
            qpos_raw[t, 4],  # gripper (servo 5)
        ]
        shared[t] = robot_fk(
            joint_angles,
            v=cmd_vel[t, 0],
            omega=cmd_vel[t, 1],
        )

    # --- qpos = 6D, action = 10D ---
    # qpos: [ee_x, ee_y, ee_z, pitch, yaw, grip_width]
    qpos = shared[:-1, :6]
    # action: [ee_x, ee_y, ee_z, pitch, yaw, grip_width, 0, 0, v, omega]
    act_8d = shared[1:]
    action = np.zeros((len(act_8d), 10))
    action[:, :6] = act_8d[:, :6]
    action[:, 8] = act_8d[:, 6]   # v
    action[:, 9] = act_8d[:, 7]   # omega
    T_out = len(qpos)

    # --- Write HDF5 ---
    with h5py.File(output_h5_path, 'w') as f:
        f.create_dataset('action', data=action.astype(np.float64))
        f.create_dataset('observations/qpos', data=qpos.astype(np.float64))

        if has_images:
            # Resize images if needed
            if img1.shape[1:3] != IMAGE_SIZE:
                img1_resized = np.zeros((T_out, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8)
                img2_resized = np.zeros((T_out, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8)
                for t in range(T_out):
                    img1_resized[t] = cv2.resize(img1[t], IMAGE_SIZE)
                    img2_resized[t] = cv2.resize(img2[t], IMAGE_SIZE)
            else:
                img1_resized = img1[:T_out]
                img2_resized = img2[:T_out]
            f.create_dataset('observations/images/camera_1', data=img1_resized, dtype=np.uint8)
            f.create_dataset('observations/images/camera_2', data=img2_resized, dtype=np.uint8)

    print(f"  Wrote {T_out} frames to {output_h5_path}")
    return T_out


# ---------------------------------------------------------------------------
# Metadata management
# ---------------------------------------------------------------------------

def load_or_create_metadata(output_dir):
    """Load existing metadata.json or create a new one."""
    meta_path = os.path.join(output_dir, 'metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            return json.load(f)
    return {
        "task_name": "cross_embodiment_shared",
        "dataset_type": "h5",
        "shared_dim": SHARED_DIM,
        "representation": ["ee_x", "ee_y", "ee_z", "pitch", "yaw", "grip_width", "v", "omega"],
        "episodes": [],
    }


def save_metadata(output_dir, metadata):
    """Save metadata.json."""
    meta_path = os.path.join(output_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {meta_path}")


def next_episode_id(metadata):
    """Get the next available episode ID."""
    if not metadata['episodes']:
        return 0
    return max(ep['episode_id'] for ep in metadata['episodes']) + 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def convert_scale_dir(input_dir, output_dir):
    """Convert all .npz files in input_dir."""
    os.makedirs(output_dir, exist_ok=True)
    metadata = load_or_create_metadata(output_dir)

    npz_files = sorted(Path(input_dir).glob('*.npz'))
    if not npz_files:
        print(f"No .npz files found in {input_dir}")
        return

    print(f"Found {len(npz_files)} Scale episodes to convert")

    for npz_path in npz_files:
        ep_id = next_episode_id(metadata)
        h5_name = f"episode_{ep_id}.h5"
        h5_path = os.path.join(output_dir, h5_name)

        print(f"\nConverting {npz_path.name} → {h5_name}")
        try:
            n_frames = convert_scale_episode(str(npz_path), h5_path)
            metadata['episodes'].append({
                "episode_id": ep_id,
                "file_name": h5_name,
                "source": "scale",
                "source_file": npz_path.name,
                "num_frames": n_frames,
            })
        except Exception as e:
            print(f"  ERROR: {e}")

    save_metadata(output_dir, metadata)
    print(f"\nDone. {len(metadata['episodes'])} total episodes in dataset.")


def convert_robot_dir(input_dir, output_dir):
    """Convert all episode_*.h5 files in input_dir."""
    os.makedirs(output_dir, exist_ok=True)
    metadata = load_or_create_metadata(output_dir)

    h5_files = sorted(Path(input_dir).glob('episode_*.h5'))
    if not h5_files:
        print(f"No episode_*.h5 files found in {input_dir}")
        return

    print(f"Found {len(h5_files)} robot episodes to convert")

    for h5_path in h5_files:
        ep_id = next_episode_id(metadata)
        h5_name = f"episode_{ep_id}.h5"
        out_path = os.path.join(output_dir, h5_name)

        print(f"\nConverting {h5_path.name} → {h5_name}")
        try:
            n_frames = convert_robot_episode(str(h5_path), out_path)
            metadata['episodes'].append({
                "episode_id": ep_id,
                "file_name": h5_name,
                "source": "robot",
                "source_file": h5_path.name,
                "num_frames": n_frames,
            })
        except Exception as e:
            print(f"  ERROR: {e}")

    save_metadata(output_dir, metadata)
    print(f"\nDone. {len(metadata['episodes'])} total episodes in dataset.")


def main():
    parser = argparse.ArgumentParser(
        description='Convert Scale or Robot episodes to shared 8D ACT dataset')
    subparsers = parser.add_subparsers(dest='source', required=True)

    # Scale subcommand
    sp_scale = subparsers.add_parser('scale', help='Convert Scale .npz episodes')
    sp_scale.add_argument('--input-dir', required=True, help='Directory with .npz files')
    sp_scale.add_argument('--output-dir', required=True, help='Output dataset directory')

    # Robot subcommand
    sp_robot = subparsers.add_parser('robot', help='Convert MARS robot HDF5 episodes')
    sp_robot.add_argument('--input-dir', required=True, help='Directory with episode_*.h5 files')
    sp_robot.add_argument('--output-dir', required=True, help='Output dataset directory')

    args = parser.parse_args()

    if args.source == 'scale':
        convert_scale_dir(args.input_dir, args.output_dir)
    elif args.source == 'robot':
        convert_robot_dir(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()
