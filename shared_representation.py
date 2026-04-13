"""
Shared 8D representation for cross-embodiment training.

Maps both robot joint data and human egocentric video data into a common format:
  [thumb_x, thumb_y, thumb_z, pitch, yaw, grip_width, v, omega]

Robot side:  joint angles → FK → (ee_xyz, orientation, gripper aperture, base vel)
Human side:  video frames → MediaPipe → (thumb_xyz, wrist→thumb direction, grip width, head motion)
"""

import numpy as np
import torch
import pytorch_kinematics as pk
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Robot FK: joint angles → shared representation
# ---------------------------------------------------------------------------

URDF_PATH = Path(__file__).parent / "ros2_ws/src/maurice_bot/maurice_sim/urdf/maurice.urdf"

# Camera position relative to base_link (from URDF)
# base_link → head joint origin + head → head_camera_left
_HEAD_OFFSET = np.array([-0.040751, -0.0002, 0.25882])
_CAM_OFFSET_FROM_HEAD = np.array([0.04327, 0.0297, -0.000275])
CAMERA_OFFSET_FROM_BASE = _HEAD_OFFSET + _CAM_OFFSET_FROM_HEAD  # total offset at head_tilt=0

_chain_ee = None
_chain_f1 = None
_chain_f2 = None


def _load_chains():
    global _chain_ee, _chain_f1, _chain_f2
    if _chain_ee is not None:
        return
    urdf_str = URDF_PATH.read_text()
    _chain_ee = pk.build_serial_chain_from_urdf(urdf_str, end_link_name="ee_link")
    _chain_f1 = pk.build_serial_chain_from_urdf(urdf_str, end_link_name="link61")
    _chain_f2 = pk.build_serial_chain_from_urdf(urdf_str, end_link_name="link62")


def robot_fk(joint_angles, v=0.0, omega=0.0, head_tilt=0.0):
    """
    Convert robot joint angles to the shared 8D representation.
    EE position is relative to the head camera.

    Args:
        joint_angles: (5,) array — [joint1, joint2, joint3, joint4, gripper]
                      Note: physical servo 5 = gripper = URDF joint6
        v:     forward velocity from /cmd_vel
        omega: yaw rate from /cmd_vel
        head_tilt: head tilt angle in radians (joint_head, pitch axis)

    Returns:
        (8,) array — [ee_x, ee_y, ee_z, pitch, yaw, grip_width, v, omega]
        where ee_xyz is relative to the head camera
    """
    _load_chains()

    # Map physical servos to URDF joints
    # Physical: [j1, j2, j3, j4, gripper]
    # URDF ee_link chain: [joint1, joint2, joint3, joint4, joint5]
    #   joint5 (wrist roll) doesn't exist physically → set to 0
    # URDF finger chains: [joint1, joint2, joint3, joint4, joint5, joint6]
    #   joint6 = gripper

    arm_q = torch.tensor([[
        joint_angles[0],  # joint1
        joint_angles[1],  # joint2
        joint_angles[2],  # joint3
        joint_angles[3],  # joint4
        0.0,              # joint5 (wrist roll, doesn't exist)
    ]], dtype=torch.float32)

    finger_q = torch.tensor([[
        joint_angles[0],  # joint1
        joint_angles[1],  # joint2
        joint_angles[2],  # joint3
        joint_angles[3],  # joint4
        0.0,              # joint5 (doesn't exist)
        joint_angles[4],  # joint6 = gripper
    ]], dtype=torch.float32)

    # FK for EE position + orientation (in base_link frame)
    tf_ee = _chain_ee.forward_kinematics(arm_q)
    mat = tf_ee.get_matrix()[0].detach().numpy()
    ee_pos_base = mat[:3, 3]

    # Convert EE position to camera frame
    # Camera position in base_link frame (accounting for head tilt)
    cam_pos = _HEAD_OFFSET.copy()
    # Head tilts around Y axis (from URDF: axis xyz="0 -1 0")
    cos_t, sin_t = np.cos(-head_tilt), np.sin(-head_tilt)
    cam_from_head = np.array([
        cos_t * _CAM_OFFSET_FROM_HEAD[0] + sin_t * _CAM_OFFSET_FROM_HEAD[2],
        _CAM_OFFSET_FROM_HEAD[1],
        -sin_t * _CAM_OFFSET_FROM_HEAD[0] + cos_t * _CAM_OFFSET_FROM_HEAD[2],
    ])
    cam_pos += cam_from_head

    # EE relative to camera
    ee_pos = ee_pos_base - cam_pos

    # Extract pitch and yaw from rotation matrix
    R = mat[:3, :3]
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    yaw = np.arctan2(R[1, 0], R[0, 0])

    # FK for gripper fingers → aperture
    tf_f1 = _chain_f1.forward_kinematics(finger_q)
    tf_f2 = _chain_f2.forward_kinematics(finger_q)
    f1_pos = tf_f1.get_matrix()[0, :3, 3].detach().numpy()
    f2_pos = tf_f2.get_matrix()[0, :3, 3].detach().numpy()
    grip_width = np.linalg.norm(f1_pos - f2_pos)

    return np.array([
        ee_pos[0], ee_pos[1], ee_pos[2],  # xyz relative to camera
        pitch, yaw,                        # orientation
        grip_width,                        # gripper aperture
        v, omega,                          # base motion
    ])


def robot_fk_batch(joint_angles_seq, cmd_vel_seq=None):
    """
    Batch FK for a full episode.

    Args:
        joint_angles_seq: (T, 5) array of joint angles
        cmd_vel_seq:      (T, 2) array of [v, omega], or None for zeros

    Returns:
        (T, 8) array in shared representation
    """
    T = len(joint_angles_seq)
    if cmd_vel_seq is None:
        cmd_vel_seq = np.zeros((T, 2))

    result = np.zeros((T, 8))
    for t in range(T):
        result[t] = robot_fk(
            joint_angles_seq[t],
            v=cmd_vel_seq[t, 0],
            omega=cmd_vel_seq[t, 1],
        )
    return result


# ---------------------------------------------------------------------------
# 2. Human hand: video frames → shared representation
# ---------------------------------------------------------------------------

_mp_hands = None


def _load_mediapipe():
    global _mp_hands
    if _mp_hands is not None:
        return
    import mediapipe as mp
    _mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
    )


def human_hand_from_frame(frame_rgb, depth_frame=None, fx=None, fy=None, cx=None, cy=None):
    """
    Extract hand representation from a single video frame.

    Args:
        frame_rgb:   (H, W, 3) uint8 RGB image
        depth_frame: (H, W) depth map in meters (optional, for 3D positions)
        fx, fy, cx, cy: camera intrinsics (needed if using depth)

    Returns:
        dict with keys: thumb_xyz, pitch, yaw, grip_width
        or None if no hand detected
    """
    _load_mediapipe()

    results = _mp_hands.process(frame_rgb)
    if not results.multi_hand_landmarks:
        return None

    hand = results.multi_hand_landmarks[0]
    h, w = frame_rgb.shape[:2]

    # Landmark indices: 0=wrist, 4=thumb tip, 8=index tip
    wrist = np.array([hand.landmark[0].x * w, hand.landmark[0].y * h, hand.landmark[0].z * w])
    thumb = np.array([hand.landmark[4].x * w, hand.landmark[4].y * h, hand.landmark[4].z * w])
    index = np.array([hand.landmark[8].x * w, hand.landmark[8].y * h, hand.landmark[8].z * w])

    # If depth is available, use real 3D coordinates
    if depth_frame is not None and fx is not None:
        def unproject(landmark):
            px, py = int(landmark[0]), int(landmark[1])
            px = np.clip(px, 0, w - 1)
            py = np.clip(py, 0, h - 1)
            z = depth_frame[py, px]
            x = (px - cx) * z / fx
            y = (py - cy) * z / fy
            return np.array([x, y, z])
        wrist = unproject(wrist)
        thumb = unproject(thumb)
        index = unproject(index)

    # Direction vector: wrist → thumb
    direction = thumb - wrist
    dx, dy, dz = direction

    pitch = np.arctan2(-dz, np.sqrt(dx**2 + dy**2))
    yaw = np.arctan2(dy, dx)

    # Grip width: thumb to index distance
    grip_width = np.linalg.norm(thumb - index)

    return {
        "thumb_xyz": thumb,
        "pitch": pitch,
        "yaw": yaw,
        "grip_width": grip_width,
    }


def human_hand_from_scale_data(obs_ee_pose, obs_head_pose=None):
    """
    Extract hand representation from Scale EgoVerse zarr data.

    Args:
        obs_ee_pose: (7,) array in xyzwxyz format — right hand EE pose
        obs_head_pose: not used for hand, but included for API consistency

    Returns:
        dict with keys: thumb_xyz, pitch, yaw, grip_width
        Note: grip_width is not available from EE pose alone (set to 0)
    """
    pos = obs_ee_pose[:3]  # xyz
    quat = obs_ee_pose[3:]  # wxyz quaternion

    # Convert quaternion to rotation matrix
    # xyzwxyz format: [x, y, z, w, qx, qy, qz] or [x, y, z, qw, qx, qy, qz]
    from pyquaternion import Quaternion
    q = Quaternion(quat[0], quat[1], quat[2], quat[3])
    R = q.rotation_matrix

    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    yaw = np.arctan2(R[1, 0], R[0, 0])

    return {
        "thumb_xyz": pos,
        "pitch": pitch,
        "yaw": yaw,
        "grip_width": 0.0,  # not available from EE pose
    }


# ---------------------------------------------------------------------------
# 3. Base motion: head pose → (v, omega)
# ---------------------------------------------------------------------------

def head_pose_to_base_motion(head_poses, fps=30):
    """
    Extract (v, omega) from a sequence of head poses.

    Args:
        head_poses: (T, 7) array in xyzwxyz format
        fps: recording frame rate

    Returns:
        (T-1, 2) array of [v, omega] per timestep
    """
    from pyquaternion import Quaternion

    dt = 1.0 / fps
    cmds = np.zeros((len(head_poses) - 1, 2))

    for t in range(len(head_poses) - 1):
        # Position: ground-plane displacement projected along heading
        dx = head_poses[t + 1, 0] - head_poses[t, 0]
        dy = head_poses[t + 1, 1] - head_poses[t, 1]

        q0 = Quaternion(head_poses[t, 3], head_poses[t, 4], head_poses[t, 5], head_poses[t, 6])
        q1 = Quaternion(head_poses[t + 1, 3], head_poses[t + 1, 4], head_poses[t + 1, 5], head_poses[t + 1, 6])

        yaw0 = q0.yaw_pitch_roll[0]
        yaw1 = q1.yaw_pitch_roll[0]

        # Project displacement into body frame → forward velocity
        v = (dx * np.cos(yaw0) + dy * np.sin(yaw0)) / dt

        # Yaw rate
        d_yaw = (yaw1 - yaw0 + np.pi) % (2 * np.pi) - np.pi
        omega = d_yaw / dt

        cmds[t] = [v, omega]

    return cmds


# ---------------------------------------------------------------------------
# 4. Full pipeline helpers
# ---------------------------------------------------------------------------

def robot_episode_to_shared(qpos_seq, action_seq):
    """
    Convert a full robot episode (from HDF5) to shared representation.

    Args:
        qpos_seq:   (T, 6) from observations/qpos — but only 5 real servos
        action_seq: (T, 12) from action — [6 leader + 4 pad + v + omega]
                    or (T, 10) if base motion not recorded

    Returns:
        (T, 8) shared representation
    """
    T = len(qpos_seq)

    # Extract physical joint angles: servos 1-4 + servo 5 (gripper)
    # qpos is [j1, j2, j3, j4, gripper, 0]
    joint_angles = np.column_stack([
        qpos_seq[:, 0],  # joint1
        qpos_seq[:, 1],  # joint2
        qpos_seq[:, 2],  # joint3
        qpos_seq[:, 3],  # joint4
        qpos_seq[:, 4],  # gripper (servo 5)
    ])

    # Extract base motion from action if available
    if action_seq.shape[1] >= 12:
        cmd_vel = action_seq[:, 10:12]  # [v, omega]
    else:
        cmd_vel = np.zeros((T, 2))

    return robot_fk_batch(joint_angles, cmd_vel)


def scale_episode_to_shared(ee_poses, head_poses, fps=30):
    """
    Convert a Scale EgoVerse episode to shared representation.

    Args:
        ee_poses:   (T, 7) right hand obs_ee_pose in xyzwxyz format
        head_poses: (T, 7) obs_head_pose in xyzwxyz format
        fps:        recording frame rate

    Returns:
        (T-1, 8) shared representation (one fewer frame due to velocity diff)
    """
    # Base motion from head pose
    base_motion = head_pose_to_base_motion(head_poses, fps=fps)
    T = len(base_motion)  # T-1 frames

    result = np.zeros((T, 8))
    for t in range(T):
        hand = human_hand_from_scale_data(ee_poses[t])
        result[t] = [
            hand["thumb_xyz"][0],
            hand["thumb_xyz"][1],
            hand["thumb_xyz"][2],
            hand["pitch"],
            hand["yaw"],
            hand["grip_width"],
            base_motion[t, 0],
            base_motion[t, 1],
        ]

    return result


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Robot FK test ===")
    # Zero pose
    r = robot_fk([0, 0, 0, 0, 0])
    print(f"Zero pose:  ee=({r[0]:.4f}, {r[1]:.4f}, {r[2]:.4f})  pitch={r[3]:.3f}  yaw={r[4]:.3f}  grip={r[5]*1000:.1f}mm")

    # Arm reaching forward and down
    r = robot_fk([0, 0.5, -1.0, 0.5, -0.3], v=0.1, omega=0.2)
    print(f"Reaching:   ee=({r[0]:.4f}, {r[1]:.4f}, {r[2]:.4f})  pitch={r[3]:.3f}  yaw={r[4]:.3f}  grip={r[5]*1000:.1f}mm  v={r[6]:.1f}  ω={r[7]:.1f}")

    print("\n=== Scale data test ===")
    # Fake EE pose: position + quaternion (identity)
    fake_ee = np.array([0.3, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0])
    hand = human_hand_from_scale_data(fake_ee)
    print(f"Scale hand: xyz=({hand['thumb_xyz'][0]:.3f}, {hand['thumb_xyz'][1]:.3f}, {hand['thumb_xyz'][2]:.3f})  pitch={hand['pitch']:.3f}  yaw={hand['yaw']:.3f}")

    print("\nDone.")
