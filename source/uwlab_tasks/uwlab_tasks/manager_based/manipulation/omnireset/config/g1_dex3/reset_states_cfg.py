# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset-state generation environments for the standing G1 + Dex3.

These are the UR5e reset-state environments with the robot, the table placement, the spawn regions and the
dataset location swapped, the G1-D + Dex1 pattern. The robot stands on the floor in front of the plain table
block, so object spawn ranges are absolute (env frame) rather than relative to a mounting plate. The hand's
approach direction is its palm normal, so the end-effector orientation ranges point the palm down instead of the
fingers.
"""

from __future__ import annotations

import numpy as np
from typing import Any

from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from uwlab_assets.robots.unitree_g1_dex3 import (
    DEX3_JOINT_REGEX,
    DEX3_PALM_BODY_NAME,
    G1_ARM_JOINT_REGEX,
    IMPLICIT_G1_DEX3,
)

from ... import mdp as task_mdp
from ..ur5e_robotiq_2f85 import reset_states_cfg as ur5e
from .actions import G1Dex3RelativeOSCAction
from .layout import (
    EE_X,
    EE_Y,
    EE_Z_ROOT,
    G1_DEX3_DATASET_DIR,
    G1_DEX3_TABLE,
    INSERTIVE_X,
    INSERTIVE_Y,
    INSERTIVE_Z_ABOVE_TABLE,
    RECEPTIVE_X,
    RECEPTIVE_Y,
    TABLE_TOP_Z,
)

ROBOT_IK_CFG = SceneEntityCfg("robot", joint_names=G1_ARM_JOINT_REGEX, body_names=DEX3_PALM_BODY_NAME)
"""The right-arm joints and the Dex3 palm body the IK placement of the end effector solves for."""

GRIPPER_CFG = SceneEntityCfg("robot", joint_names=DEX3_JOINT_REGEX)
"""The Dex3 joints closed on the object by the grasp-dataset resets."""

EE_POSE_RANGE_B = {
    "x": EE_X,
    "y": EE_Y,
    "z": EE_Z_ROOT,
    "roll": (-2 * np.pi / 3, -np.pi / 3),
    "pitch": (-np.pi / 6, np.pi / 6),
    "yaw": (-np.pi / 6, 2 * np.pi / 3),
}
"""Palm reset region in the robot root frame. Euler angles compose as ``Rz(yaw) Ry(pitch) Rx(roll)``: a roll of
-90 deg points the palm normal (+y) down, the roll and pitch ranges tilt it by up to about 40 deg, the yaw heads
the fingers from 30 deg right of straight ahead to 120 deg left, the headings the IK probe reaches reliably."""

G1_DEX3_COLLISION_BODY_REGEX = ["right_shoulder_.*_link", "right_elbow_link", "right_wrist_.*_link", "right_hand_.*"]
"""Robot links the collision analyzers of the success test sample points on.

Only the right arm and the right Dex3 can touch the objects; the legs, torso and the frozen left arm never come
near the object regions, so the analyzers sample these links alone."""


# ===============
# Private Helpers
# ===============
def _with_params(term: EventTerm, **updates: Any) -> EventTerm:
    """Copy an event term with some of its parameters overridden.

    Args:
        term: The event term to copy.
        **updates: Parameter values that replace the entries of ``term.params`` with the same key.

    Returns:
        A copy of ``term`` whose ``params`` are the original ones updated with ``updates``.
    """
    return term.replace(params={**term.params, **updates})


def _robot_links_only(analyzer_cfg: task_mdp.CollisionAnalyzerCfg) -> task_mdp.CollisionAnalyzerCfg:
    """Restrict a collision analyzer that samples the robot to the links of :data:`G1_DEX3_COLLISION_BODY_REGEX`.

    Args:
        analyzer_cfg: A collision analyzer configuration of the UR5e success test.

    Returns:
        The analyzer unchanged if it does not sample the robot, otherwise a copy sampling only the right-arm and
        Dex3 links.
    """
    asset_cfg = analyzer_cfg.asset_cfg
    if asset_cfg.name != "robot":
        return analyzer_cfg
    return analyzer_cfg.replace(asset_cfg=SceneEntityCfg("robot", body_names=G1_DEX3_COLLISION_BODY_REGEX))


# configclass stores mutable defaults as factories, so parent defaults are read from instances, not classes
_UR5E_SCENE = ur5e.ResetStatesSceneCfg(num_envs=1, env_spacing=1.5)
_UR5E_EE_ANYWHERE = ur5e.ObjectAnywhereEEAnywhereEventCfg()
_UR5E_RESTING_GRASPED = ur5e.ObjectRestingEEGraspedEventCfg()
_UR5E_ANYWHERE_GRASPED = ur5e.ObjectAnywhereEEGraspedEventCfg()
_UR5E_PARTIAL_ANYWHERE = ur5e.ObjectPartiallyAssembledEEAnywhereEventCfg()
_UR5E_PARTIAL_GRASPED = ur5e.ObjectPartiallyAssembledEEGraspedEventCfg()
_UR5E_TERMINATIONS = ur5e.ResetStatesTerminationCfg()


# ------------------------------------------------------------------------------------------------------- scene
@configclass
class G1Dex3ResetStatesSceneCfg(ur5e.ResetStatesSceneCfg):
    """The UR5e reset-state scene with the standing G1 in front of the plain table block."""

    robot = IMPLICIT_G1_DEX3.replace(prim_path="{ENV_REGEX_NS}/Robot")
    """The G1 with the Dex3 hand, implicit actuators."""

    table = G1_DEX3_TABLE
    """The plain table block of the G1 + Dex3 layout."""

    ur5_metal_support = None
    """The UR5e's mounting support is not part of this scene."""

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=_UR5E_SCENE.ground.spawn,
    )
    """The UR5e ground plane moved to z = 0: the robot stands on the floor, not on the table."""


# ------------------------------------------------------------------------------------------------------ events
@configclass
class G1Dex3ResetStatesBaseEventCfg(ur5e.ResetStatesBaseEventCfg):
    """Events shared by every G1 + Dex3 reset-state set."""

    reset_robot_pose = None
    """No robot-pose jitter: the robot is fixed to the floor."""

    reset_receptive_object_pose = EventTerm(
        func=task_mdp.reset_root_states_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": RECEPTIVE_X,
                "y": RECEPTIVE_Y,
                "z": (TABLE_TOP_Z, TABLE_TOP_Z),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-np.pi / 12, np.pi / 12),
            },
            "velocity_range": {},
            "asset_cfgs": {"receptive_object": SceneEntityCfg("receptive_object")},
            "offset_asset_cfg": None,
            "use_bottom_offset": True,
        },
    )
    """The receptive object placed upright on the table top within its spawn region, yaw within +-15 deg."""


_INSERTIVE_ANYWHERE = EventTerm(
    func=task_mdp.reset_root_states_uniform,
    mode="reset",
    params={
        "pose_range": {
            "x": INSERTIVE_X,
            "y": INSERTIVE_Y,
            "z": (TABLE_TOP_Z + INSERTIVE_Z_ABOVE_TABLE[0], TABLE_TOP_Z + INSERTIVE_Z_ABOVE_TABLE[1]),
            "roll": (-np.pi, np.pi),
            "pitch": (-np.pi, np.pi),
            "yaw": (-np.pi, np.pi),
        },
        "velocity_range": {},
        "asset_cfgs": {"insertive_object": SceneEntityCfg("insertive_object")},
        "offset_asset_cfg": None,
        "use_bottom_offset": True,
    },
)
"""The insertive object dropped anywhere in its spawn region above the table top with a random orientation."""


@configclass
class G1Dex3ObjectAnywhereEEAnywhereEventCfg(G1Dex3ResetStatesBaseEventCfg, ur5e.ObjectAnywhereEEAnywhereEventCfg):
    """ObjectAnywhereEEAnywhere events: insertive object anywhere, palm IK-placed anywhere in its region."""

    reset_insertive_object_pose = _INSERTIVE_ANYWHERE
    reset_end_effector_pose = _with_params(
        _UR5E_EE_ANYWHERE.reset_end_effector_pose,
        pose_range_b=EE_POSE_RANGE_B,
        robot_ik_cfg=ROBOT_IK_CFG,
    )


@configclass
class G1Dex3ObjectRestingEEGraspedEventCfg(G1Dex3ResetStatesBaseEventCfg, ur5e.ObjectRestingEEGraspedEventCfg):
    """ObjectRestingEEGrasped events: object from the resting dataset, hand closed on it from the grasp dataset."""

    reset_insertive_object_pose_from_reset_states = _with_params(
        _UR5E_RESTING_GRASPED.reset_insertive_object_pose_from_reset_states,
        dataset_dir=G1_DEX3_DATASET_DIR,
    )
    reset_end_effector_pose_from_grasp_dataset = _with_params(
        _UR5E_RESTING_GRASPED.reset_end_effector_pose_from_grasp_dataset,
        dataset_dir=G1_DEX3_DATASET_DIR,
        robot_ik_cfg=ROBOT_IK_CFG,
        gripper_cfg=GRIPPER_CFG,
    )


@configclass
class G1Dex3ObjectAnywhereEEGraspedEventCfg(G1Dex3ResetStatesBaseEventCfg, ur5e.ObjectAnywhereEEGraspedEventCfg):
    """ObjectAnywhereEEGrasped events: insertive object anywhere, hand closed on it from the grasp dataset."""

    reset_insertive_object_pose = _INSERTIVE_ANYWHERE
    reset_end_effector_pose_from_grasp_dataset = _with_params(
        _UR5E_ANYWHERE_GRASPED.reset_end_effector_pose_from_grasp_dataset,
        dataset_dir=G1_DEX3_DATASET_DIR,
        robot_ik_cfg=ROBOT_IK_CFG,
        gripper_cfg=GRIPPER_CFG,
    )


@configclass
class G1Dex3ObjectPartiallyAssembledEEAnywhereEventCfg(
    G1Dex3ResetStatesBaseEventCfg, ur5e.ObjectPartiallyAssembledEEAnywhereEventCfg
):
    """ObjectPartiallyAssembledEEAnywhere events: assembly from its dataset, palm at the top of its region."""

    reset_insertive_object_pose_from_partial_assembly_dataset = _with_params(
        _UR5E_PARTIAL_ANYWHERE.reset_insertive_object_pose_from_partial_assembly_dataset,
        dataset_dir=G1_DEX3_DATASET_DIR,
    )
    reset_end_effector_pose = _with_params(
        _UR5E_PARTIAL_ANYWHERE.reset_end_effector_pose,
        pose_range_b={**EE_POSE_RANGE_B, "z": (EE_Z_ROOT[1], EE_Z_ROOT[1])},
        robot_ik_cfg=ROBOT_IK_CFG,
    )


@configclass
class G1Dex3ObjectPartiallyAssembledEEGraspedEventCfg(
    G1Dex3ResetStatesBaseEventCfg, ur5e.ObjectPartiallyAssembledEEGraspedEventCfg
):
    """ObjectPartiallyAssembledEEGrasped events: assembly from its dataset, hand closed on the insertive object."""

    reset_insertive_object_pose_from_partial_assembly_dataset = _with_params(
        _UR5E_PARTIAL_GRASPED.reset_insertive_object_pose_from_partial_assembly_dataset,
        dataset_dir=G1_DEX3_DATASET_DIR,
    )
    reset_end_effector_pose_from_grasp_dataset = _with_params(
        _UR5E_PARTIAL_GRASPED.reset_end_effector_pose_from_grasp_dataset,
        dataset_dir=G1_DEX3_DATASET_DIR,
        robot_ik_cfg=ROBOT_IK_CFG,
        gripper_cfg=GRIPPER_CFG,
    )


# ------------------------------------------------------------------------------------------------ terminations
@configclass
class G1Dex3ResetStatesTerminationCfg(ur5e.ResetStatesTerminationCfg):
    """The UR5e reset-state terminations with the success test adapted to the G1 + Dex3."""

    success = _UR5E_TERMINATIONS.success.replace(
        params={
            **_UR5E_TERMINATIONS.success.params,
            "collision_analyzer_cfgs": [
                _robot_links_only(c) for c in _UR5E_TERMINATIONS.success.params["collision_analyzer_cfgs"]
            ],
            "ee_body_name": DEX3_PALM_BODY_NAME,
            # objects (and the hand) must stay above the table top: a peg that rolled off the edge onto the
            # floor is not a usable reset state (the UR5e value is relative to its mounting plate)
            "pos_z_threshold": TABLE_TOP_Z - 0.02,
        }
    )
    """The UR5e success test sampling the right-arm and Dex3 links only, with the palm as the hand body and the
    height threshold just below the table top."""


# ------------------------------------------------------------------------------------------------- environments
@configclass
class G1Dex3ResetStatesCfg(ur5e.UR5eRobotiq2f85ResetStatesCfg):
    """Reset-state environment with the G1 right arm and Dex3 hand."""

    scene: G1Dex3ResetStatesSceneCfg = G1Dex3ResetStatesSceneCfg(num_envs=1, env_spacing=1.5)
    terminations: G1Dex3ResetStatesTerminationCfg = G1Dex3ResetStatesTerminationCfg()
    actions: G1Dex3RelativeOSCAction = G1Dex3RelativeOSCAction()


# ``configclass`` binds each class's ``__post_init__`` to whatever ``cls.__post_init__`` resolves to at decoration
# time. For the variants below that is the one inherited from ``G1Dex3ResetStatesCfg`` (first base), which skips
# the UR5e variant's ``__post_init__`` that fills in the success-term parameters (``max_object_pos_deviation``,
# assembly thresholds). Chain it explicitly.
@configclass
class G1Dex3ObjectAnywhereEEAnywhereResetStatesCfg(G1Dex3ResetStatesCfg, ur5e.ObjectAnywhereEEAnywhereResetStatesCfg):
    """ObjectAnywhereEEAnywhere reset states for the G1 + Dex3."""

    events: G1Dex3ObjectAnywhereEEAnywhereEventCfg = G1Dex3ObjectAnywhereEEAnywhereEventCfg()

    def __post_init__(self):
        """Apply the UR5e variant's success-term parameters, which the configclass post-init binding skips."""
        ur5e.ObjectAnywhereEEAnywhereResetStatesCfg.__post_init__(self)


@configclass
class G1Dex3ObjectRestingEEGraspedResetStatesCfg(G1Dex3ResetStatesCfg, ur5e.ObjectRestingEEGraspedResetStatesCfg):
    """ObjectRestingEEGrasped reset states for the G1 + Dex3."""

    events: G1Dex3ObjectRestingEEGraspedEventCfg = G1Dex3ObjectRestingEEGraspedEventCfg()

    def __post_init__(self):
        """Apply the UR5e variant's success-term parameters, which the configclass post-init binding skips."""
        ur5e.ObjectRestingEEGraspedResetStatesCfg.__post_init__(self)


@configclass
class G1Dex3ObjectAnywhereEEGraspedResetStatesCfg(G1Dex3ResetStatesCfg, ur5e.ObjectAnywhereEEGraspedResetStatesCfg):
    """ObjectAnywhereEEGrasped reset states for the G1 + Dex3."""

    events: G1Dex3ObjectAnywhereEEGraspedEventCfg = G1Dex3ObjectAnywhereEEGraspedEventCfg()

    def __post_init__(self):
        """Apply the UR5e variant's success-term parameters, which the configclass post-init binding skips."""
        ur5e.ObjectAnywhereEEGraspedResetStatesCfg.__post_init__(self)


@configclass
class G1Dex3ObjectPartiallyAssembledEEAnywhereResetStatesCfg(
    G1Dex3ResetStatesCfg, ur5e.ObjectPartiallyAssembledEEAnywhereResetStatesCfg
):
    """ObjectPartiallyAssembledEEAnywhere reset states for the G1 + Dex3."""

    events: G1Dex3ObjectPartiallyAssembledEEAnywhereEventCfg = G1Dex3ObjectPartiallyAssembledEEAnywhereEventCfg()

    def __post_init__(self):
        """Apply the UR5e variant's success-term parameters, which the configclass post-init binding skips."""
        ur5e.ObjectPartiallyAssembledEEAnywhereResetStatesCfg.__post_init__(self)


@configclass
class G1Dex3ObjectPartiallyAssembledEEGraspedResetStatesCfg(
    G1Dex3ResetStatesCfg, ur5e.ObjectPartiallyAssembledEEGraspedResetStatesCfg
):
    """ObjectPartiallyAssembledEEGrasped reset states for the G1 + Dex3."""

    events: G1Dex3ObjectPartiallyAssembledEEGraspedEventCfg = G1Dex3ObjectPartiallyAssembledEEGraspedEventCfg()

    def __post_init__(self):
        """Apply the UR5e variant's success-term parameters, which the configclass post-init binding skips."""
        ur5e.ObjectPartiallyAssembledEEGraspedResetStatesCfg.__post_init__(self)
