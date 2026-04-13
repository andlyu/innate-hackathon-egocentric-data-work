#!/usr/bin/env python3
"""
Self-contained script to build the shared 8D dataset from EgoVerse Scale episodes.
Run on a GPU machine with fast disk for best performance.

Usage:
    pip install boto3 zarr==3.1.5 pyquaternion numpy opencv-python h5py zstandard
    python build_dataset_gpu.py --n-episodes 100 --output-dir ./shared_dataset

Output: HDF5 files with 8D shared representation + 224x224 egocentric images.
"""

import argparse
import json
import os
import random
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import h5py
import numpy as np
import zstandard

# ---------------------------------------------------------------------------
# R2 / S3 setup
# ---------------------------------------------------------------------------

def get_s3_client():
    import boto3
    os.environ["AWS_ACCESS_KEY_ID"] = "os.environ.get("AWS_ACCESS_KEY_ID", "")"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "os.environ.get("AWS_SECRET_ACCESS_KEY", "")"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-2"

    sm = boto3.client("secretsmanager", region_name="us-east-2")
    r2_json = json.loads(
        sm.get_secret_value(SecretId="r2/rldb/public/credentials")["SecretString"]
    )

    s3 = boto3.client(
        "s3",
        aws_access_key_id=r2_json["access_key_id"],
        aws_secret_access_key=r2_json["secret_access_key"],
        endpoint_url=r2_json["endpoint_url"],
        region_name="auto",
    )
    return s3


def list_all_episodes(s3):
    """List all Scale episode names in R2."""
    episodes = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(
        Bucket="rldb", Prefix="processed_v3/scale/", Delimiter="/"
    ):
        for p in page.get("CommonPrefixes", []):
            ep = p["Prefix"].replace("processed_v3/scale/", "").rstrip("/")
            episodes.append(ep)
    return sorted(episodes)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

NEEDED_FILES = [
    "right.obs_keypoints/c/0/0",
    "right.obs_keypoints/zarr.json",
    "obs_head_pose/c/0/0",
    "obs_head_pose/zarr.json",
    "images.front_1/c/0",
    "images.front_1/zarr.json",
    "zarr.json",
]


def download_episode(s3, ep_name, download_dir):
    """Download a single episode's needed files."""
    ep_dir = os.path.join(download_dir, ep_name)
    os.makedirs(ep_dir, exist_ok=True)

    for fname in NEEDED_FILES:
        local_path = os.path.join(ep_dir, fname)
        if os.path.exists(local_path):
            continue
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            s3.download_file("rldb", f"processed_v3/scale/{ep_name}/{fname}", local_path)
        except Exception:
            pass

    return ep_dir


def download_all(s3, episode_names, download_dir, max_workers=8):
    """Download all episodes in parallel."""
    print(f"Downloading {len(episode_names)} episodes with {max_workers} threads...")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_episode, s3, ep, download_dir): ep
            for ep in episode_names
        }
        done = 0
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 10 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (len(episode_names) - done) / rate
                print(f"  [{done}/{len(episode_names)}] {rate:.1f} eps/s, ETA {eta:.0f}s")

    print(f"Download complete in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------------------
# Image decoding from sharded zarr
# ---------------------------------------------------------------------------

def decode_sharded_images(ep_dir, n_frames, frame_indices):
    """Decode specific JPEG frames from zarr shard, resize to 224x224."""
    data_path = f"{ep_dir}/images.front_1/c/0"
    if not os.path.exists(data_path):
        return None

    with open(data_path, "rb") as f:
        raw = f.read()

    # Shard index at end: n_frames * 16 bytes + 4 bytes CRC
    index_size = n_frames * 16 + 4
    index_data = raw[-index_size:-4]
    all_offsets = np.frombuffer(index_data, dtype=np.uint64).reshape(-1, 2)

    dctx = zstandard.ZstdDecompressor()
    images = np.zeros((len(frame_indices), 224, 224, 3), dtype=np.uint8)

    for i, t in enumerate(frame_indices):
        offset, nbytes = int(all_offsets[t, 0]), int(all_offsets[t, 1])
        dec = dctx.decompress(raw[offset : offset + nbytes])
        # 8-byte prefix + JPEG data
        img = cv2.imdecode(np.frombuffer(dec[8:], dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            images[i] = cv2.resize(img, (224, 224))

    return images


# ---------------------------------------------------------------------------
# Shared representation computation (batched numpy)
# ---------------------------------------------------------------------------

def quat_to_rotation_matrix_batch(quats):
    """Convert (N, 4) quaternions [w, x, y, z] to (N, 3, 3) rotation matrices."""
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

    R = np.zeros((len(quats), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def quat_yaw_batch(quats):
    """Extract yaw from (N, 4) quaternions [w, x, y, z]."""
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    siny = 2 * (w * z + x * y)
    cosy = 1 - 2 * (y * y + z * z)
    return np.arctan2(siny, cosy)


def compute_shared_representation(right_kp, head_pose, fps=30.0):
    """
    Batched computation of 8D shared representation.

    Args:
        right_kp: (T, 21, 3) hand keypoints in world frame
        head_pose: (T, 7) head pose [x,y,z,qw,qx,qy,qz]
        fps: recording frame rate

    Returns:
        (T, 8) shared representation, valid_mask (T,) bool
    """
    T = min(len(right_kp), len(head_pose))
    right_kp = right_kp[:T]
    head_pose = head_pose[:T]

    # Find valid frames
    quat_norms = np.linalg.norm(head_pose[:, 3:7], axis=1)
    valid = quat_norms > 0.1
    if valid.sum() < 10:
        return None, None

    # Normalize quaternions
    quats = head_pose[:, 3:7].copy()
    quats[valid] /= quat_norms[valid, None]

    # Compute inverse rotation matrices for valid frames
    # Inverse of unit quaternion [w,x,y,z] is [w,-x,-y,-z]
    inv_quats = quats.copy()
    inv_quats[:, 1:] *= -1
    R_inv = quat_to_rotation_matrix_batch(inv_quats)

    # Transform keypoints to camera frame: R_inv @ (kp - head_pos)
    h_pos = head_pose[:, :3]  # (T, 3)

    wrist_w = right_kp[:, 0, :]  # (T, 3)
    thumb_w = right_kp[:, 4, :]
    index_w = right_kp[:, 8, :]

    # Displacements
    wrist_disp = wrist_w - h_pos
    thumb_disp = thumb_w - h_pos
    index_disp = index_w - h_pos

    # Batched rotation: R_inv @ disp (using einsum)
    wrist_cam = np.einsum("tij,tj->ti", R_inv, wrist_disp)
    thumb_cam = np.einsum("tij,tj->ti", R_inv, thumb_disp)
    index_cam = np.einsum("tij,tj->ti", R_inv, index_disp)

    # Wrist → thumb direction
    direction = thumb_cam - wrist_cam
    dx, dy, dz = direction[:, 0], direction[:, 1], direction[:, 2]
    pitch = np.arctan2(-dz, np.sqrt(dx**2 + dy**2))
    yaw_hand = np.arctan2(dy, dx)

    # Grip width
    grip_width = np.linalg.norm(thumb_cam - index_cam, axis=1)

    # Base motion from head pose deltas
    v = np.zeros(T)
    omega = np.zeros(T)
    head_yaw = quat_yaw_batch(quats)

    dt_arr = np.ones(T) / fps
    # For consecutive valid frames
    for i in range(T - 1):
        if valid[i] and valid[i + 1]:
            dx_h = head_pose[i + 1, 0] - head_pose[i, 0]
            dy_h = head_pose[i + 1, 1] - head_pose[i, 1]
            y0 = head_yaw[i]
            v[i] = (dx_h * np.cos(y0) + dy_h * np.sin(y0)) * fps
            d_yaw = (head_yaw[i + 1] - y0 + np.pi) % (2 * np.pi) - np.pi
            omega[i] = d_yaw * fps

    # Assemble shared representation
    shared = np.column_stack(
        [
            thumb_cam[:, 0],
            thumb_cam[:, 1],
            thumb_cam[:, 2],
            pitch,
            yaw_hand,
            grip_width,
            v,
            omega,
        ]
    )

    return shared, valid


# ---------------------------------------------------------------------------
# Process single episode
# ---------------------------------------------------------------------------

def process_episode(ep_dir, ep_hash, output_dir, ep_id):
    """Process one episode: read data, compute shared repr, write HDF5."""
    import zarr

    try:
        right_kp_flat = zarr.open_array(f"{ep_dir}/right.obs_keypoints", mode="r")[:]
        head_pose = zarr.open_array(f"{ep_dir}/obs_head_pose", mode="r")[:]
    except Exception as e:
        return None, f"read error: {e}"

    right_kp = right_kp_flat.reshape(-1, 21, 3)
    T = min(len(right_kp), len(head_pose))

    shared, valid = compute_shared_representation(right_kp[:T], head_pose[:T])
    if shared is None:
        return None, "too few valid frames"

    valid_indices = np.where(valid)[0]
    T_valid = len(valid_indices)
    T_out = T_valid - 1

    # Get shared repr for valid frames only
    shared_valid = shared[valid_indices]

    # Decode images for valid frames
    images = decode_sharded_images(ep_dir, T, valid_indices[:T_out])

    # Write HDF5
    h5_name = f"episode_{ep_id}.h5"
    h5_path = os.path.join(output_dir, h5_name)

    # qpos = 6D: [ee_x, ee_y, ee_z, pitch, yaw, grip_width]
    qpos = shared_valid[:-1, :6]

    # action = 10D: [ee_x, ee_y, ee_z, pitch, yaw, grip_width, 0, 0, v, omega]
    act_8d = shared_valid[1:]
    action = np.zeros((len(act_8d), 10))
    action[:, :6] = act_8d[:, :6]   # ee_xyz, pitch, yaw, grip
    action[:, 8] = act_8d[:, 6]     # v
    action[:, 9] = act_8d[:, 7]     # omega

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("action", data=action.astype(np.float64))
        f.create_dataset("observations/qpos", data=qpos.astype(np.float64))

    # Write images as mp4 (much faster for WebDataset conversion later)
    video_files = []
    if images is not None:
        for cam in ["camera_1", "camera_2"]:
            mp4_name = f"episode_{ep_id}_{cam}.mp4"
            mp4_path = os.path.join(output_dir, mp4_name)
            writer = cv2.VideoWriter(
                mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (224, 224)
            )
            for frame in images:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            video_files.append(mp4_name)

    return {
        "episode_id": ep_id,
        "file_name": h5_name,
        "source": "scale",
        "source_hash": ep_hash,
        "num_frames": T_out,
        "video_files": video_files,
    }, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build shared 8D dataset from EgoVerse Scale")
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="./shared_dataset")
    parser.add_argument("--download-dir", type=str, default="./scale_episodes")
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--process-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.download_dir, exist_ok=True)

    # Connect to R2
    print("Connecting to R2...")
    s3 = get_s3_client()

    # List and select episodes
    print("Listing episodes...")
    all_episodes = list_all_episodes(s3)
    print(f"Found {len(all_episodes)} Scale episodes")

    random.seed(args.seed)
    selected = random.sample(all_episodes, min(args.n_episodes, len(all_episodes)))
    print(f"Selected {len(selected)} episodes")

    # Download
    download_all(s3, selected, args.download_dir, max_workers=args.download_workers)

    # Process
    print(f"\nProcessing {len(selected)} episodes...")
    t0 = time.time()

    metadata = {
        "task_name": "cross_embodiment_shared",
        "dataset_type": "h264",
        "data_frequency": 30,
        "shared_dim": 8,
        "representation": [
            "thumb_x", "thumb_y", "thumb_z",
            "pitch", "yaw", "grip_width", "v", "omega",
        ],
        "number_of_episodes": 0,
        "episodes": [],
    }

    ep_id = 0
    skipped = 0

    # Process with thread pool (I/O bound: reading zarr + writing HDF5)
    with ThreadPoolExecutor(max_workers=args.process_workers) as executor:
        futures = {}
        for ep_name in selected:
            ep_dir = os.path.join(args.download_dir, ep_name)
            ep_hash = ep_name.replace(".zarr", "")
            futures[executor.submit(process_episode, ep_dir, ep_hash, args.output_dir, ep_id)] = ep_name
            ep_id += 1

        done = 0
        for future in as_completed(futures):
            result, error = future.result()
            done += 1
            if error:
                skipped += 1
            else:
                metadata["episodes"].append(result)

            if done % 10 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (len(selected) - done) / rate
                total_size = sum(
                    os.path.getsize(os.path.join(args.output_dir, f))
                    for f in os.listdir(args.output_dir)
                    if f.endswith(".h5")
                )
                print(
                    f"  [{done}/{len(selected)}] "
                    f"{total_size / 1e9:.1f}GB — "
                    f"{rate:.1f} eps/s — "
                    f"ETA {eta:.0f}s"
                )

    # Sort episodes by ID
    metadata["episodes"].sort(key=lambda x: x["episode_id"])
    metadata["number_of_episodes"] = len(metadata["episodes"])

    # Write both metadata.json and dataset_metadata.json for compatibility
    for fname in ["metadata.json", "dataset_metadata.json"]:
        with open(os.path.join(args.output_dir, fname), "w") as f:
            json.dump(metadata, f, indent=2)

    total_frames = sum(ep["num_frames"] for ep in metadata["episodes"])
    total_size = sum(
        os.path.getsize(os.path.join(args.output_dir, f))
        for f in os.listdir(args.output_dir)
        if f.endswith(".h5")
    )
    frame_counts = [ep["num_frames"] for ep in metadata["episodes"]]
    elapsed = time.time() - t0

    print(f"\n{'='*50}")
    print(f"Dataset Complete!")
    print(f"{'='*50}")
    print(f"Episodes:     {len(metadata['episodes'])}")
    print(f"Skipped:      {skipped}")
    print(f"Total frames: {total_frames}")
    print(f"Frames/ep:    min={min(frame_counts)}, max={max(frame_counts)}, avg={np.mean(frame_counts):.0f}")
    print(f"Dataset size: {total_size / 1e9:.2f} GB")
    print(f"Time:         {elapsed / 60:.1f} min")
    print(f"Output:       {args.output_dir}")


if __name__ == "__main__":
    main()
