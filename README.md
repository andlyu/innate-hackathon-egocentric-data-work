# Cross-Embodiment Learning: EgoVerse + MARS Robot

Training a robot manipulation policy using human egocentric video data from the [EgoVerse](https://egoverse.ai/) dataset, combined with robot teleoperation data from the MARS robot.

## Demo

https://github.com/user-attachments/assets/scale_visualization.mp4

*Left: Egocentric video from Scale EgoVerse dataset. Right: 3D visualization of extracted hand keypoints (wrist, thumb tip, index fingertip) alongside the MARS robot arm, both in camera-relative coordinates.*

## Overview

We built a cross-embodiment training pipeline that enables a small mobile robot (MARS) to learn manipulation skills from thousands of human demonstration videos captured with head-mounted cameras.

### The Problem

Training robot manipulation policies typically requires hundreds of teleoperated demonstrations on the target robot. This is expensive and time-consuming. Meanwhile, the [EgoVerse](https://egoverse.ai/) dataset contains 80,000+ episodes of humans performing manipulation tasks, captured egocentrically with stereo cameras and 3D hand tracking.

### Our Approach

**1. Shared Representation**

We defined a unified 6D observation / 10D action representation that maps between human hands and robot grippers:

| | Human (EgoVerse) | Robot (MARS) |
|---|---|---|
| **Position** | Thumb tip xyz (camera frame) | End-effector xyz via FK (camera frame) |
| **Orientation** | Wrist-to-thumb direction (pitch, yaw) | EE orientation from FK (pitch, yaw) |
| **Grip** | Thumb-to-index distance | Gripper finger distance from FK |
| **Base motion** | Head pose deltas (v, omega) | Wheel commands (v, omega) |

- `qpos` (6D): `[ee_x, ee_y, ee_z, pitch, yaw, grip_width]`
- `action` (10D): `[ee_x, ee_y, ee_z, pitch, yaw, grip_width, 0, 0, v, omega]`

The robot's joint angles are converted to this shared space using forward kinematics on the MARS URDF. The human hand keypoints (21 per hand from EgoVerse's stereo hand tracking) are transformed to camera frame and mapped to the same representation.

**2. Cross-Embodiment Pre-training**

We downloaded 1,500 Scale EgoVerse episodes and converted them to the shared representation. These were filtered to episodes under 1 minute (~1,440 episodes, ~1M frames) and combined with 5x-duplicated robot data (~500 episodes) for balanced training.

The [ACT (Action Chunking with Transformers)](https://github.com/innate-inc/ACT) model was trained on this combined dataset, learning visual features and manipulation patterns from both human and robot demonstrations.

**3. Frankenstein Checkpoint: Body Transplant**

The key insight: the cross-embodiment model's **vision backbone and transformer layers** learned rich manipulation representations from the diverse human data, but the **input/output projection layers** were trained on the shared representation format (EE positions), not the robot's native joint angles.

We created a "Frankenstein" checkpoint:
- **Body** (280 layers): ResNet18 backbone + transformer encoder/decoder from cross-embodiment training (1,440 human + 500 robot episodes)
- **Head** (18 layers): Input/output projections + normalization from a model trained purely on robot joint angle data

This gave us a model with cross-embodiment visual understanding that speaks the robot's native action language.

**4. Fine-tuning on Robot Data**

The Frankenstein checkpoint was fine-tuned on the robot's task-specific data (101 episodes across 8 task types + 3 demonstration sets), converging quickly thanks to the pre-trained backbone.

## Pipeline

```
EgoVerse R2 Bucket (7,592 Scale episodes)
    |
    v
build_dataset_gpu.py          # Download + extract keypoints + write mp4s
    |
    v
shared_dataset_1500/           # 1,500 episodes in shared 6D/10D format
    |
    +-- convert_to_shared_dataset.py  # Convert robot HDF5 (joint angles) via FK
    |       |
    |       v
    |   merged_full_v1_8D/     # Robot data in shared format
    |
    v
merge_datasets.py              # Combine human + robot (5x upsampled)
    |
    v
fast_webdataset.py             # Zero-copy parallel tar shard writer
    |                          # (64 workers, streaming, no OOM)
    v
merged_1500_1min/webdataset/   # Training-ready WebDataset
    |
    v
ACT training (H200 GPU)        # Cross-embodiment pre-training
    |
    v
Frankenstein checkpoint         # Cross-embodiment body + robot head
    |
    v
Fine-tune on robot data        # Task-specific adaptation
    |
    v
Deploy on Jetson (MARS robot)  # ros2 action send_goal /execute_skill ...
```

## Scripts

| Script | Purpose |
|---|---|
| `build_dataset_gpu.py` | Download Scale episodes from R2, extract 21 hand keypoints, compute shared representation, write h5 + mp4 |
| `convert_to_shared_dataset.py` | Convert robot HDF5 episodes from joint angles to shared representation via forward kinematics |
| `fast_webdataset.py` | Zero-copy parallel WebDataset converter. Each worker writes its own tar shards directly - no memory accumulation |
| `shared_representation.py` | Forward kinematics utilities, camera offset computation, URDF chain loading |
| `visualize_combined.py` | Interactive 3D matplotlib viewer showing MARS arm FK + human hand trajectory in camera-relative coordinates |
| `data_viz_scale.py` | Colab notebook for browsing EgoVerse episodes, selecting training data, and batch export |

## Key Technical Details

### Forward Kinematics
The MARS robot has a 5-DOF arm (4 arm joints + gripper). Joint angles are converted to 3D EE position using `pytorch_kinematics` with the robot's URDF. The position is computed relative to the head camera to match the egocentric view of the human data.

### Hand Keypoint Extraction
EgoVerse provides 21 keypoints per hand in world frame. We extract:
- Keypoint 0 (wrist) and keypoint 4 (thumb tip) for position + orientation
- Keypoint 8 (index fingertip) for grip width
- Head pose quaternion for world-to-camera-frame transformation

### Base Motion
Human head translation maps to robot forward velocity (v), head yaw rotation maps to turn rate (omega). Both computed as frame-to-frame deltas at 30 FPS.

### WebDataset Conversion
We solved a critical memory issue: the original converter held all samples in RAM before writing (~1.4TB for 2,000 episodes, crashed an H200 node with 1.5TB RAM). The zero-copy version has each worker write tar shards directly, keeping memory usage flat regardless of dataset size.

## Future Work

- **Task-specific human data**: Currently using general household manipulation videos. Using the [EgoVerse flagship tasks](https://partners.mecka.ai/egoverse?task=flagship_object_in_container&robot_name=scale_right_arm#explorer) (e.g., object-in-container) that closely match the robot's pick-and-place tasks should significantly improve transfer
- **Scaling to 5,000+ episodes**: Pipeline is built and tested, limited only by download time (~40 min) and processing (~1 hour with 64 CPU workers)
- **Dexterous manipulation**: EgoVerse provides full 21-keypoint hand tracking. With a dexterous gripper, the Aina-style fingertip policy approach could enable fine-grained manipulation
- **Real-time domain adaptation**: Using EgoBridge-style optimal transport alignment during training to better bridge the visual domain gap between human and robot egocentric views

## Infrastructure

- **Training**: 8x NVIDIA H200 (Nebius), 1.5TB RAM, 128 CPU cores
- **Robot**: MARS mobile robot with 5-DOF arm, stereo head cameras, Jetson compute
- **Data**: [EgoVerse](https://egoverse.ai/) dataset via Cloudflare R2

## References

- [EgoVerse](https://egoverse.ai/) - Egocentric human dataset for robot learning
- [EgoMimic](https://egomimic.github.io/) - Scaling imitation learning via egocentric video
- [ACT](https://github.com/innate-inc/ACT) - Action Chunking with Transformers
- [Aina](https://github.com/facebookresearch/AINA) - Dexterity from smart lenses
- [EgoScale](https://research.nvidia.com/labs/gear/egoscale/) - Scaling dexterous manipulation with egocentric data
