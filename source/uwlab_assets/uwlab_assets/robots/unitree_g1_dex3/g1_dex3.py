# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Unitree G1 (29 DoF) + Dex3, right arm and right hand only.

The following configurations are available:

* :obj:`G1_DEX3_ARTICULATION`: Base articulation (USD, init state), no actuators.
* :obj:`IMPLICIT_G1_DEX3`: ImplicitActuator arm with zero joint stiffness (pure effort passthrough for the
  task-space OSC) and the Dex3 at the robot's position-loop gains. Stage 1 RL training.
* :obj:`EXPLICIT_G1_DEX3`: DelayedPDActuator arm (motor delay) for sim2real fine-tuning.
* :obj:`DEX3_HAND`: the floating right Dex3 alone, for grasp sampling.

Assets live in ``source/uwlab_assets/data/Robots/Unitree/G1Dex3`` (gitignored); they are produced by
:mod:`uwlab_assets.robots.unitree_g1_dex3.build_urdf` and the Isaac Lab URDF converter.

Frames: the articulation root is ``pelvis``, fixed to the world :data:`~.stance.PELVIS_HEIGHT` above the floor
and pitched :data:`~.stance.PELVIS_PITCH` nose down, as the real robot stands (:mod:`.stance`). The legs, the
waist, the left arm and the left hand are frozen inside the USD in that stance, so the feet hover 3 mm above the
floor and never touch it. Joint limits, efforts and velocities are the robot URDF's.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from .paths import DEX3_HAND_USD_PATH, G1_DEX3_ASSET_DIR, G1_DEX3_METADATA_PATH, G1_DEX3_USD_PATH  # noqa: F401
from .stance import DEX3_KD, DEX3_KP, DEX3_OPEN_RIGHT, DEX3_RIGHT_JOINT_NAMES, PELVIS_HEIGHT, PELVIS_PITCH

# ===========================================================================================================
# Root pose and joint groups. Quaternions are (x, y, z, w), the Isaac Lab 3.0 convention.
# ===========================================================================================================

PELVIS_ROOT_POS = (0.0, 0.0, PELVIS_HEIGHT)
"""Position [m] of the pelvis origin in the world: on the floor's normal through the environment origin."""

PELVIS_ROOT_ROT = (0.0, math.sin(PELVIS_PITCH / 2), 0.0, math.cos(PELVIS_PITCH / 2))
"""Orientation (x, y, z, w) of the pelvis in the world: pitched :data:`~.stance.PELVIS_PITCH` nose down."""

G1_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
"""The seven right-arm joints, shoulder to wrist."""

G1_ARM_JOINT_REGEX = ["right_shoulder_.*", "right_elbow_joint", "right_wrist_.*"]
"""Joint-name patterns matching exactly :data:`G1_ARM_JOINT_NAMES`."""

DEX3_JOINT_NAMES = list(DEX3_RIGHT_JOINT_NAMES)
"""The seven right Dex3 joints, in motor order."""

DEX3_JOINT_REGEX = ["right_hand_.*_joint"]
"""Joint-name pattern matching exactly :data:`DEX3_JOINT_NAMES`."""

G1_EE_BODY_NAME = "right_wrist_yaw_link"
"""Flange the OSC controls (the Dex3 palm is bolted 41.5 mm along its x axis)."""

DEX3_PALM_BODY_NAME = "right_hand_palm_link"
"""Hand body used for grasp offsets (metadata ``gripper_offset`` is in this frame)."""

# ===========================================================================================================
# Default joint positions: the robot's stance for the arm, the binary open pose for the hand.
# ===========================================================================================================

G1_DEFAULT_ARM_JOINT_POS = {
    "right_shoulder_pitch_joint": -0.44,
    "right_shoulder_roll_joint": -0.41,
    "right_shoulder_yaw_joint": -0.40,
    "right_elbow_joint": -0.44,
    "right_wrist_roll_joint": -0.69,
    "right_wrist_pitch_joint": -0.39,
    "right_wrist_yaw_joint": 0.98,
}
"""Default right-arm posture [rad] and null-space target of the OSC: the palm facing down 20 cm above the table
top of the OmniReset layout (palm origin 0.26 m ahead and 0.20 m to the right of the pelvis), fingers heading 30
deg left of straight ahead, elbow in front of the torso; solved by IK on the robot URDF on 2026-10-28. The
posture the real robot starts from is :data:`~.stance.STANCE_RIGHT_ARM` (forearm raised, palm facing inward)."""

DEX3_DEFAULT_JOINT_POS = dict(DEX3_OPEN_RIGHT)
"""Default Dex3 joint positions [rad]: the binary open pose."""

G1_DEX3_DEFAULT_JOINT_POS = {**G1_DEFAULT_ARM_JOINT_POS, **DEX3_DEFAULT_JOINT_POS}
"""Default positions [rad] of all 14 movable joints."""

# ===========================================================================================================
# Per-joint limits of the robot URDF (effort in N m, velocity in rad/s).
# ===========================================================================================================

G1_EFFORT_LIMITS = {
    "right_shoulder_pitch_joint": 25.0,
    "right_shoulder_roll_joint": 25.0,
    "right_shoulder_yaw_joint": 25.0,
    "right_elbow_joint": 25.0,
    "right_wrist_roll_joint": 25.0,
    "right_wrist_pitch_joint": 5.0,
    "right_wrist_yaw_joint": 5.0,
}
"""Effort limits [N m] of the right-arm joints."""

G1_VELOCITY_LIMITS = {
    "right_shoulder_pitch_joint": 37.0,
    "right_shoulder_roll_joint": 37.0,
    "right_shoulder_yaw_joint": 37.0,
    "right_elbow_joint": 37.0,
    "right_wrist_roll_joint": 37.0,
    "right_wrist_pitch_joint": 22.0,
    "right_wrist_yaw_joint": 22.0,
}
"""Velocity limits [rad/s] of the right-arm joints."""

DEX3_EFFORT_LIMITS = {n: 2.45 if n.endswith("thumb_0_joint") else 1.4 for n in DEX3_JOINT_NAMES}
"""Effort limits [N m] of the Dex3 joints: 2.45 for the thumb abduction, 1.4 for the six flexion joints."""

DEX3_VELOCITY_LIMITS = {n: 6.857 if n.endswith("thumb_0_joint") else 12.0 for n in DEX3_JOINT_NAMES}
"""Velocity limits [rad/s] of the Dex3 joints."""

# ===========================================================================================================
# Actuators and articulations.
# ===========================================================================================================

DEX3_HAND_ACTUATOR = ImplicitActuatorCfg(
    joint_names_expr=DEX3_JOINT_REGEX,
    stiffness=DEX3_KP,
    damping=DEX3_KD,
    effort_limit_sim=DEX3_EFFORT_LIMITS,
    velocity_limit_sim=DEX3_VELOCITY_LIMITS,
)
"""The Dex3 position loop at the gains the data-recording rig drives the real hand with (0.5 N m/rad, 0.1 N m
s/rad): soft, so the fingers stop on an object well short of the shut pose, which is how the robot's code tells a
holding hand from an empty one."""

G1_DEX3_ARTICULATION = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=G1_DEX3_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=36,
            solver_velocity_iteration_count=0,
            # Fixed base: ``convert_urdf_kit.py --fix_base`` authors a world ``root_joint`` and puts the articulation
            # root on the default prim, so PhysX treats the asset as a true fixed-base articulation. Leave
            # ``fix_root_link`` unset (see the G1-D configuration).
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=PELVIS_ROOT_POS, rot=PELVIS_ROOT_ROT, joint_pos=G1_DEX3_DEFAULT_JOINT_POS
    ),
    soft_joint_pos_limit_factor=1,
)
"""The standing G1 with the right Dex3, no actuators."""

EXPLICIT_G1_DEX3 = G1_DEX3_ARTICULATION.copy()  # type: ignore
EXPLICIT_G1_DEX3.actuators = {
    "arm": DelayedPDActuatorCfg(
        joint_names_expr=G1_ARM_JOINT_REGEX,
        stiffness=0.0,
        damping=0.0,
        effort_limit=G1_EFFORT_LIMITS,
        effort_limit_sim=G1_EFFORT_LIMITS,
        velocity_limit=G1_VELOCITY_LIMITS,
        velocity_limit_sim=G1_VELOCITY_LIMITS,
        min_delay=0,
        max_delay=1,
    ),
    "gripper": DEX3_HAND_ACTUATOR,
}
"""Delayed PD arm for sim2real fine-tuning, Dex3 at the robot's gains."""

IMPLICIT_G1_DEX3 = G1_DEX3_ARTICULATION.copy()  # type: ignore
IMPLICIT_G1_DEX3.actuators = {
    "arm": ImplicitActuatorCfg(
        joint_names_expr=G1_ARM_JOINT_REGEX,
        stiffness=0.0,
        # The same arm as the G1-D's: joint damping lives here (PhysX applies it implicitly) instead of in the OSC,
        # whose explicit task-space damping diverges on the light wrist links (see IMPLICIT_G1D_DEX1).
        damping=1.0,
        effort_limit_sim=G1_EFFORT_LIMITS,
        velocity_limit_sim=G1_VELOCITY_LIMITS,
    ),
    "gripper": DEX3_HAND_ACTUATOR,
}
"""Implicit arm actuator (zero stiffness, implicit damping) for stage 1 training, Dex3 at the robot's gains."""

DEX3_HAND = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Dex3Hand",
    spawn=sim_utils.UsdFileCfg(
        usd_path=DEX3_HAND_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=36,
            solver_velocity_iteration_count=0,
            # Anchored like the Dex1 grasp-sampling gripper (fixed-base articulation authored by the converter's
            # ``--fix_base``); the sampler moves it by writing the root pose. ``fix_root_link`` must stay unset.
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 0.1), rot=(0, 0, 0, 1), joint_pos=DEX3_DEFAULT_JOINT_POS),
    actuators={"gripper": DEX3_HAND_ACTUATOR},
    soft_joint_pos_limit_factor=1,
)
"""The floating right Dex3 (palm link as root) for grasp sampling."""
