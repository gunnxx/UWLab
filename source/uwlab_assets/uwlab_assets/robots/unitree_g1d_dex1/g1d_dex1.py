# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Unitree G1-D + Dex1-1 (right arm only).

The following configurations are available:

* :obj:`G1D_DEX1_ARTICULATION`: Base articulation (USD, init state), no actuators.
* :obj:`IMPLICIT_G1D_DEX1`: ImplicitActuator arm with zero joint gains (pure effort passthrough for the task-space
  OSC) and a PD gripper. Stage 1 RL training.
* :obj:`EXPLICIT_G1D_DEX1`: DelayedPDActuator arm (motor delay) for sim2real fine-tuning.
* :obj:`DEX1_GRIPPER`: the floating Dex1-1 alone, for grasp sampling.

Assets live in ``source/uwlab_assets/data/Robots/Unitree/G1dDex1`` (gitignored); they are produced by
:mod:`uwlab_assets.robots.unitree_g1d_dex1.build_urdf` and the Isaac Lab URDF converter.

Frames: the articulation root is ``AGV_link`` (chassis), fixed to the world with its origin 0.111 m above the
floor (wheel axle height plus wheel radius). The waist, lift, wheels and left arm are frozen inside the USD.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from .paths import DEX1_GRIPPER_USD_PATH, G1D_DEX1_ASSET_DIR, G1D_DEX1_METADATA_PATH, G1D_DEX1_USD_PATH  # noqa: F401

AGV_ROOT_HEIGHT = 0.0265 + 0.0848
"""Height of the AGV_link origin above the floor: wheel joint offset + wheel radius (from g1_d.urdf)."""

G1D_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
G1D_ARM_JOINT_REGEX = ["right_shoulder_.*", "right_elbow_joint", "right_wrist_.*"]
DEX1_FINGER_JOINT_NAMES = ["right_dex1_finger_joint_1", "right_dex1_finger_joint_2"]
DEX1_FINGER_JOINT_REGEX = ["right_dex1_finger_joint_.*"]

G1D_EE_BODY_NAME = "right_wrist_yaw_link"
"""Flange the OSC controls (the Dex1-1 is bolted 41.5 mm along its x axis)."""
DEX1_BASE_BODY_NAME = "right_dex1_base_link"
"""Gripper body used for grasp offsets (metadata ``gripper_offset`` is in this frame)."""

DEX1_FINGER_OPEN = 0.0245
DEX1_FINGER_CLOSED = -0.02

# Default right-arm posture: upper arm forward-down, elbow bent, gripper tilted up and ~25 cm above the table top
# so a default-reset scene never starts in contact (see docs/notes on the G1-D workspace).
G1D_DEFAULT_ARM_JOINT_POS = {
    "right_shoulder_pitch_joint": -1.0,
    "right_shoulder_roll_joint": -0.4,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 1.0,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": -0.8,
    "right_wrist_yaw_joint": 0.0,
}
DEX1_DEFAULT_JOINT_POS = {n: DEX1_FINGER_OPEN for n in DEX1_FINGER_JOINT_NAMES}
G1D_DEX1_DEFAULT_JOINT_POS = {**G1D_DEFAULT_ARM_JOINT_POS, **DEX1_DEFAULT_JOINT_POS}

# Official per-joint limits (g1_d.urdf; wrist pitch/yaw 5 N m per the G1-D spec sheet)
G1D_EFFORT_LIMITS = {
    "right_shoulder_pitch_joint": 25.0,
    "right_shoulder_roll_joint": 25.0,
    "right_shoulder_yaw_joint": 25.0,
    "right_elbow_joint": 25.0,
    "right_wrist_roll_joint": 25.0,
    "right_wrist_pitch_joint": 5.0,
    "right_wrist_yaw_joint": 5.0,
}
G1D_VELOCITY_LIMITS = {
    "right_shoulder_pitch_joint": 37.0,
    "right_shoulder_roll_joint": 37.0,
    "right_shoulder_yaw_joint": 37.0,
    "right_elbow_joint": 37.0,
    "right_wrist_roll_joint": 37.0,
    "right_wrist_pitch_joint": 22.0,
    "right_wrist_yaw_joint": 22.0,
}

DEX1_GRIPPER_ACTUATOR = ImplicitActuatorCfg(
    joint_names_expr=DEX1_FINGER_JOINT_REGEX,
    # prismatic fingers: stiffness in N/m. A 30 mm peg stops the fingers 12 mm short of the closed command, so
    # 2000 N/m saturates at the 20 N effort limit, roughly the Dex1-1's default grip force.
    stiffness=2000.0,
    damping=50.0,
    effort_limit_sim=20.0,
    velocity_limit_sim=0.2,
)

G1D_DEX1_ARTICULATION = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=G1D_DEX1_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=36,
            solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, AGV_ROOT_HEIGHT), rot=(0.0, 0.0, 0.0, 1.0), joint_pos=G1D_DEX1_DEFAULT_JOINT_POS
    ),
    soft_joint_pos_limit_factor=1,
)

EXPLICIT_G1D_DEX1 = G1D_DEX1_ARTICULATION.copy()  # type: ignore
EXPLICIT_G1D_DEX1.actuators = {
    "arm": DelayedPDActuatorCfg(
        joint_names_expr=G1D_ARM_JOINT_REGEX,
        stiffness=0.0,
        damping=0.0,
        effort_limit=G1D_EFFORT_LIMITS,
        effort_limit_sim=G1D_EFFORT_LIMITS,
        velocity_limit=G1D_VELOCITY_LIMITS,
        velocity_limit_sim=G1D_VELOCITY_LIMITS,
        min_delay=0,
        max_delay=1,
    ),
    "gripper": DEX1_GRIPPER_ACTUATOR,
}

IMPLICIT_G1D_DEX1 = G1D_DEX1_ARTICULATION.copy()  # type: ignore
IMPLICIT_G1D_DEX1.actuators = {
    "arm": ImplicitActuatorCfg(
        joint_names_expr=G1D_ARM_JOINT_REGEX,
        stiffness=0.0,
        damping=0.0,
        effort_limit_sim=G1D_EFFORT_LIMITS,
        velocity_limit_sim=G1D_VELOCITY_LIMITS,
    ),
    "gripper": DEX1_GRIPPER_ACTUATOR,
}

DEX1_GRIPPER = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Dex1Gripper",
    spawn=sim_utils.UsdFileCfg(
        usd_path=DEX1_GRIPPER_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=36, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 0.1), rot=(0, 0, 0, 1), joint_pos=DEX1_DEFAULT_JOINT_POS),
    actuators={"gripper": DEX1_GRIPPER_ACTUATOR},
    soft_joint_pos_limit_factor=1,
)
