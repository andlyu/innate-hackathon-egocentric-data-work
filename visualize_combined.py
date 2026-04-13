"""
Combined 3D visualizer: MARS arm FK + Scale human hand trajectory.

Both are shown in camera-relative coordinates so they can be directly compared.

Usage:
  1. Export Scale data from Colab notebook (adds export cell to data_viz_scale.py)
  2. Run: python visualize_combined.py [--scale-data path/to/scale_episode.npz]
  3. Without Scale data, shows MARS arm with sliders + synthetic human trajectory
"""

import argparse
import numpy as np
import torch
import pytorch_kinematics as pk
import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button
import warnings
warnings.filterwarnings('ignore')

from shared_representation import CAMERA_OFFSET_FROM_BASE, _HEAD_OFFSET, _CAM_OFFSET_FROM_HEAD

# ---------------------------------------------------------------------------
# URDF / FK setup
# ---------------------------------------------------------------------------
URDF_PATH = 'ros2_ws/src/maurice_bot/maurice_sim/urdf/maurice.urdf'
urdf_str = open(URDF_PATH).read()

link_names = ['link1', 'link2', 'link3', 'link4', 'link5', 'ee_link', 'link61', 'link62']
chains = {}
for name in link_names:
    chains[name] = pk.build_serial_chain_from_urdf(urdf_str, end_link_name=name)


def compute_arm_positions(joints):
    """Compute all link positions in camera-relative frame."""
    # Joint1 origin in base frame
    base_positions = [np.array([0.086, -0.05285, 0.04025])]

    for name in link_names:
        chain = chains[name]
        n = len(chain.get_joint_parameter_names())
        q = torch.tensor([joints[:n]], dtype=torch.float32)
        tf = chain.forward_kinematics(q)
        pos = tf.get_matrix()[0, :3, 3].detach().numpy()
        base_positions.append(pos)

    # Convert all to camera-relative
    cam_offset = CAMERA_OFFSET_FROM_BASE
    cam_positions = [p - cam_offset for p in base_positions]

    return cam_positions


def scale_world_to_camera(ee_pose_world, head_pose_world):
    """
    Transform Scale EE pose from world frame to camera frame.
    Matches what PoseCoordinateFrameTransform does in data_viz_scale.py.

    Args:
        ee_pose_world:   (7,) xyzwxyz — [x, y, z, qw, qx, qy, qz]
        head_pose_world: (7,) xyzwxyz — [x, y, z, qw, qx, qy, qz]

    Returns:
        (3,) xyz position in camera frame
    """
    from pyquaternion import Quaternion

    # Head pose
    head_pos = head_pose_world[:3]
    head_q = Quaternion(head_pose_world[3], head_pose_world[4],
                        head_pose_world[5], head_pose_world[6])

    # EE pose
    ee_pos = ee_pose_world[:3]

    # Transform to camera frame: rotate the displacement by inverse head rotation
    displacement = ee_pos - head_pos
    ee_cam = head_q.inverse.rotate(displacement)

    return ee_cam


def load_scale_data(npz_path):
    """
    Load exported Scale episode data.

    Expected .npz keys:
        right_ee_cam: (T, 6) — right hand EE in camera frame [x,y,z,yaw,pitch,roll]
        head_poses:   (T, 7) — head pose in world frame (for v, omega extraction)
        fps:          scalar — recording frame rate

    Or raw world-frame data:
        right_ee_world: (T, 7) — right hand EE in world frame [x,y,z,qw,qx,qy,qz]
        head_poses:     (T, 7) — head pose in world frame
    """
    data = np.load(npz_path, allow_pickle=True)

    if 'right_ee_cam' in data:
        # Already in camera frame
        return data['right_ee_cam'][:, :3]  # just xyz
    elif 'right_ee_world' in data:
        # Need to transform
        ee_world = data['right_ee_world']
        head_world = data['head_poses']
        T = len(ee_world)
        cam_positions = np.zeros((T, 3))
        for t in range(T):
            cam_positions[t] = scale_world_to_camera(ee_world[t], head_world[t])
        return cam_positions
    else:
        raise ValueError(f"Expected 'right_ee_cam' or 'right_ee_world' in {npz_path}")


def generate_synthetic_human_trajectory(n_frames=200):
    """
    Generate a synthetic human hand trajectory for testing.
    Simulates a reaching + grasping motion in camera-relative coordinates.
    Scaled to roughly match MARS workspace.
    """
    t = np.linspace(0, 2 * np.pi, n_frames)

    # Start from resting position, reach forward, grasp, retract
    x = 0.25 + 0.10 * np.sin(t)           # forward/back
    y = -0.05 + 0.08 * np.sin(t * 0.5)    # left/right
    z = -0.05 + 0.06 * np.sin(t * 1.5)    # up/down

    return np.column_stack([x, y, z])


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Combined MARS + Scale 3D Visualizer')
    parser.add_argument('--scale-data', type=str, default=None,
                        help='Path to exported Scale episode .npz file')
    parser.add_argument('--scale-factor', type=float, default=1.0,
                        help='Scale factor to apply to human trajectory (to match MARS workspace)')
    args = parser.parse_args()

    # Load or generate human trajectory
    if args.scale_data:
        human_traj = load_scale_data(args.scale_data)
        print(f"Loaded Scale data: {len(human_traj)} frames")
    else:
        human_traj = generate_synthetic_human_trajectory()
        print("Using synthetic human trajectory (pass --scale-data to use real data)")

    # Apply scale factor
    human_traj = human_traj * args.scale_factor

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.40)

    # Joint limits from URDF
    joint_limits = [
        (-1.5708, 1.5708),   # joint1 (base rotation)
        (-1.5708, 1.22),     # joint2 (shoulder)
        (-1.5708, 1.7453),   # joint3 (elbow)
        (-1.9199, 1.7453),   # joint4 (wrist pitch)
        (-0.8727, 0.3491),   # joint5 → physical gripper
    ]

    # Sliders for robot joints
    sliders = []
    for i in range(5):
        label = ['Base', 'Shoulder', 'Elbow', 'Wrist', 'Gripper'][i]
        ax_slider = plt.axes([0.15, 0.25 - i * 0.035, 0.55, 0.025])
        s = Slider(ax_slider, label, joint_limits[i][0], joint_limits[i][1], valinit=0.0)
        sliders.append(s)

    # Slider for human trajectory frame
    ax_frame = plt.axes([0.15, 0.25 - 5 * 0.035, 0.55, 0.025])
    frame_slider = Slider(ax_frame, 'Frame', 0, len(human_traj) - 1, valinit=0, valstep=1)

    # Scale factor slider
    ax_scale = plt.axes([0.15, 0.25 - 6 * 0.035, 0.55, 0.025])
    scale_slider = Slider(ax_scale, 'Scale', 0.1, 3.0, valinit=args.scale_factor)

    def update(val):
        # Map physical 5 servos to URDF 6 joints
        phys_joints = [s.val for s in sliders]
        urdf_joints = [
            phys_joints[0],  # joint1
            phys_joints[1],  # joint2
            phys_joints[2],  # joint3
            phys_joints[3],  # joint4
            0.0,             # joint5 (doesn't exist)
            phys_joints[4],  # joint6 = gripper
        ]

        ax.cla()

        # --- MARS arm ---
        positions = compute_arm_positions(urdf_joints)
        arm_pos = np.array(positions[:7])
        f1 = np.array(positions[7])
        f2 = np.array(positions[8])
        ee = np.array(positions[6])

        ax.plot(arm_pos[:, 0], arm_pos[:, 1], arm_pos[:, 2],
                'b-o', linewidth=3, markersize=8, label='MARS arm')
        ax.plot([ee[0], f1[0]], [ee[1], f1[1]], [ee[2], f1[2]],
                'r-s', linewidth=3, markersize=10, label='Finger 1')
        ax.plot([ee[0], f2[0]], [ee[1], f2[1]], [ee[2], f2[2]],
                'g-s', linewidth=3, markersize=10, label='Finger 2')

        # --- Camera origin ---
        ax.scatter([0], [0], [0], c='black', s=200, marker='*', label='Camera', zorder=10)

        # --- Human hand trajectory ---
        sf = scale_slider.val
        traj_scaled = human_traj * sf
        frame_idx = int(frame_slider.val)

        # Full trajectory (faded)
        ax.plot(traj_scaled[:, 0], traj_scaled[:, 1], traj_scaled[:, 2],
                'orange', linewidth=1, alpha=0.3, label='Human trajectory')

        # Trail up to current frame
        trail_start = max(0, frame_idx - 30)
        trail = traj_scaled[trail_start:frame_idx + 1]
        if len(trail) > 1:
            ax.plot(trail[:, 0], trail[:, 1], trail[:, 2],
                    'orange', linewidth=2, alpha=0.7)

        # Current human hand position
        h = traj_scaled[frame_idx]
        ax.scatter([h[0]], [h[1]], [h[2]], c='orange', s=200, marker='D',
                   edgecolors='red', linewidth=2, label=f'Human hand (frame {frame_idx})',
                   zorder=10)

        # --- Distance line between robot EE and human hand ---
        dist = np.linalg.norm(ee - h)
        ax.plot([ee[0], h[0]], [ee[1], h[1]], [ee[2], h[2]],
                '--', color='gray', linewidth=1, alpha=0.5)

        gripper_dist = np.linalg.norm(f1 - f2) * 1000
        ax.set_title(
            f'MARS + Human (Camera Frame)\n'
            f'Robot EE: ({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f})m  '
            f'Human: ({h[0]:.3f}, {h[1]:.3f}, {h[2]:.3f})m\n'
            f'Distance: {dist*100:.1f}cm  Gripper: {gripper_dist:.1f}mm'
        )
        ax.set_xlabel('X (forward)')
        ax.set_ylabel('Y (left/right)')
        ax.set_zlabel('Z (up/down)')
        ax.set_xlim(-0.2, 0.5)
        ax.set_ylim(-0.3, 0.3)
        ax.set_zlim(-0.3, 0.3)
        ax.legend(loc='upper left', fontsize=8)
        fig.canvas.draw_idle()

    for s in sliders:
        s.on_changed(update)
    frame_slider.on_changed(update)
    scale_slider.on_changed(update)

    update(None)
    plt.show()


if __name__ == '__main__':
    main()
