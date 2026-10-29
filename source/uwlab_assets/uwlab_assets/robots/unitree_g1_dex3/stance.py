# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""How the real G1 stands and drives its Dex3 hands: numbers copied from the robot's code (ludo-research).

Every value here is measured on or commanded to the physical robot and is copied so that the simulated robot
matches it; nothing is tuned for the simulator. Sources, all in the ``ludo-research`` repository at commit
``d031ab04f`` (2026-10-28):

* The stance: ``research/projects/ludi02-vla/sim/src/g1_sim/core/start_pose.json``. The joint positions are the
  per-joint medians of the first frame of 139 Quest teleoperation episodes recorded on 2026-10-20 (the ready pose
  the robot's teleoperation starts from: forearms raised forward, hands open, waist upright, knees bent). The
  pelvis pitch is the median of luda2's pelvis IMU over a take (positive is nose down), the pelvis height the one
  that puts the soles of the sagging legs on the floor.
* The Dex3 open and shut poses: ``core/body_sdk/src/ludo_body/g1/dex3.py`` (``DEX3_OPEN_RIGHT`` and
  ``DEX3_SHUT_RIGHT``), the binary hand poses the ludi 0.2 data was recorded with, measured as the right hand
  reaches them.
* The Dex3 position-loop gains: ``core/vla_controller/src/ludo_vla_controller/dex3.py`` (``DEX3_KP``,
  ``DEX3_KD``), the gains of the rig that recorded that data.

Joint positions are in rad, the Dex3 joints in Dex3 motor order (thumb 0 to 2, middle 0 and 1, index 0 and 1),
which the URDF lists as middle before index as well.
"""

# ====================================================================================================
# The standing stance: the pelvis pose over the floor and every joint of the start pose, by joint group.
# ====================================================================================================

PELVIS_HEIGHT = 0.7738
"""Height [m] of the pelvis link origin above the floor in the stance."""

PELVIS_PITCH = 0.078
"""Pitch [rad] of the pelvis about its y axis in the stance, positive nose down (world frame)."""

STANCE_LEGS = {
    "left_hip_pitch_joint": -0.3774,
    "left_hip_roll_joint": -0.0124,
    "left_hip_yaw_joint": 0.0106,
    "left_knee_joint": 0.5785,
    "left_ankle_pitch_joint": -0.2442,
    "left_ankle_roll_joint": -0.0165,
    "right_hip_pitch_joint": -0.4029,
    "right_hip_roll_joint": 0.006,
    "right_hip_yaw_joint": -0.0115,
    "right_knee_joint": 0.6311,
    "right_ankle_pitch_joint": -0.2925,
    "right_ankle_roll_joint": -0.0091,
}
"""Leg joint positions [rad] of the stance (knees bent, as the robot stands)."""

STANCE_WAIST = {
    "waist_yaw_joint": -0.0002,
    "waist_roll_joint": 0.002,
    "waist_pitch_joint": 0.015,
}
"""Waist joint positions [rad] of the stance (upright)."""

STANCE_LEFT_ARM = {
    "left_shoulder_pitch_joint": 0.0093,
    "left_shoulder_roll_joint": 0.6479,
    "left_shoulder_yaw_joint": -0.0483,
    "left_elbow_joint": -0.5682,
    "left_wrist_roll_joint": -0.0029,
    "left_wrist_pitch_joint": 0.0018,
    "left_wrist_yaw_joint": -0.0045,
}
"""Left-arm joint positions [rad] of the stance: forearm raised forward beside the torso."""

STANCE_RIGHT_ARM = {
    "right_shoulder_pitch_joint": 0.0036,
    "right_shoulder_roll_joint": -0.5887,
    "right_shoulder_yaw_joint": 0.0976,
    "right_elbow_joint": -0.5648,
    "right_wrist_roll_joint": -0.0049,
    "right_wrist_pitch_joint": -0.0102,
    "right_wrist_yaw_joint": 0.0013,
}
"""Right-arm joint positions [rad] of the stance: the left arm's mirror image."""

STANCE_LEFT_HAND = {
    "left_hand_thumb_0_joint": -0.0228,
    "left_hand_thumb_1_joint": 0.078,
    "left_hand_thumb_2_joint": 0.0916,
    "left_hand_middle_0_joint": -0.0915,
    "left_hand_middle_1_joint": -0.1353,
    "left_hand_index_0_joint": -0.1377,
    "left_hand_index_1_joint": -0.0926,
}
"""Left Dex3 joint positions [rad] of the stance (open)."""

STANCE_RIGHT_HAND = {
    "right_hand_thumb_0_joint": -0.0257,
    "right_hand_thumb_1_joint": -0.0919,
    "right_hand_thumb_2_joint": -0.1102,
    "right_hand_middle_0_joint": 0.0966,
    "right_hand_middle_1_joint": 0.1515,
    "right_hand_index_0_joint": 0.1305,
    "right_hand_index_1_joint": 0.104,
}
"""Right Dex3 joint positions [rad] of the stance (open)."""

# =======================================================================================
# The right Dex3 as the robot drives it: the binary open / shut poses and the drive gains.
# =======================================================================================

DEX3_JOINT_SUFFIXES = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")
"""The seven Dex3 joints in motor order; the URDF joint of hand ``side`` is ``<side>_hand_<suffix>_joint``."""

DEX3_RIGHT_JOINT_NAMES = [f"right_hand_{s}_joint" for s in DEX3_JOINT_SUFFIXES]
"""URDF names of the right Dex3 joints, in motor order."""

DEX3_OPEN_RIGHT = dict(zip(DEX3_RIGHT_JOINT_NAMES, (-0.040, -0.030, -0.036, 0.026, 0.030, 0.048, 0.025)))
"""Right Dex3 joint positions [rad] of the binary *open* pose, as the hand reaches it."""

DEX3_SHUT_RIGHT = dict(zip(DEX3_RIGHT_JOINT_NAMES, (-0.040, -0.605, -1.426, 0.982, 1.630, 0.978, 1.631)))
"""Right Dex3 joint positions [rad] of the binary *shut* pose, as the empty hand reaches it. ``thumb_0`` (thumb
abduction) is the same open and shut."""

DEX3_KP = 0.5
"""Dex3 position-loop stiffness [N m/rad] the data-recording rig drives the hand joints with."""

DEX3_KD = 0.1
"""Dex3 position-loop damping [N m s/rad] the data-recording rig drives the hand joints with."""
