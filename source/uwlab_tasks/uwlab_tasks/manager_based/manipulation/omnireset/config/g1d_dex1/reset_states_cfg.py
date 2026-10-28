# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset-state generation environments for the G1-D + Dex1-1.

These are the UR5e reset-state environments with the robot, the table placement, the spawn regions and the
dataset location swapped. The robot stands on the floor in front of the same Vention table instead of being
mounted on it, so object spawn ranges are absolute (env frame) rather than relative to a mounting plate.
"""

from __future__ import annotations

import numpy as np
from typing import Any

from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from uwlab_assets.robots.unitree_g1d_dex1 import (
    DEX1_BASE_BODY_NAME,
    DEX1_FINGER_JOINT_REGEX,
    G1D_ARM_JOINT_REGEX,
    IMPLICIT_G1D_DEX1,
)

from ... import mdp as task_mdp
from ..ur5e_robotiq_2f85 import reset_states_cfg as ur5e
from .actions import G1dDex1RelativeOSCAction
from .layout import (
    EE_X,
    EE_Y,
    EE_Z_ROOT,
    G1D_DATASET_DIR,
    G1D_TABLE,
    INSERTIVE_X,
    INSERTIVE_Y,
    INSERTIVE_Z_ABOVE_TABLE,
    RECEPTIVE_X,
    RECEPTIVE_Y,
    TABLE_TOP_Z,
)

ROBOT_IK_CFG = SceneEntityCfg("robot", joint_names=G1D_ARM_JOINT_REGEX, body_names=DEX1_BASE_BODY_NAME)
"""The right-arm joints and the Dex1-1 base body the IK placement of the end effector solves for."""
GRIPPER_CFG = SceneEntityCfg("robot", joint_names=DEX1_FINGER_JOINT_REGEX)
"""The Dex1-1 finger joints closed on the object by the grasp-dataset resets."""
EE_POSE_RANGE_B = {
    "x": EE_X,
    "y": EE_Y,
    "z": EE_Z_ROOT,
    "roll": (0.0, 0.0),
    "pitch": (np.pi / 4, 3 * np.pi / 4),
    "yaw": (np.pi / 2, 3 * np.pi / 2),
}
"""End-effector reset region in the robot root frame: the layout's box, gripper pointing within 45 deg of down."""

G1D_COLLISION_BODY_REGEX = ["right_shoulder_.*_link", "right_elbow_link", "right_wrist_.*_link", "right_dex1_.*"]
"""Robot links the collision analyzers of the success test sample points on.

Only the right arm and the Dex1 can touch the objects; the chassis, lift, torso and the frozen left arm never
come near the table top, so the analyzers sample these links alone (31 links otherwise)."""


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
    """Restrict a collision analyzer that samples the robot to the links of :data:`G1D_COLLISION_BODY_REGEX`.

    Args:
        analyzer_cfg: A collision analyzer configuration of the UR5e success test.

    Returns:
        The analyzer unchanged if it does not sample the robot, otherwise a copy sampling only the right-arm and
        Dex1-1 links.
    """
    asset_cfg = analyzer_cfg.asset_cfg
    if asset_cfg.name != "robot":
        return analyzer_cfg
    return analyzer_cfg.replace(asset_cfg=SceneEntityCfg("robot", body_names=G1D_COLLISION_BODY_REGEX))


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
class G1dResetStatesSceneCfg(ur5e.ResetStatesSceneCfg):
    """The UR5e reset-state scene with the G1-D standing on the floor in front of the plain table block."""

    robot = IMPLICIT_G1D_DEX1.replace(prim_path="{ENV_REGEX_NS}/Robot")
    """The G1-D with the Dex1-1 gripper, implicit actuators."""

    table = G1D_TABLE
    """The plain table block of the G1-D layout."""

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
class G1dResetStatesBaseEventCfg(ur5e.ResetStatesBaseEventCfg):
    """Events shared by every G1-D reset-state set."""

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
class G1dObjectAnywhereEEAnywhereEventCfg(G1dResetStatesBaseEventCfg, ur5e.ObjectAnywhereEEAnywhereEventCfg):
    """ObjectAnywhereEEAnywhere events: insertive object anywhere, end effector IK-placed anywhere in its region."""

    reset_insertive_object_pose = _INSERTIVE_ANYWHERE
    reset_end_effector_pose = _with_params(
        _UR5E_EE_ANYWHERE.reset_end_effector_pose,
        pose_range_b=EE_POSE_RANGE_B,
        robot_ik_cfg=ROBOT_IK_CFG,
    )


@configclass
class G1dObjectRestingEEGraspedEventCfg(G1dResetStatesBaseEventCfg, ur5e.ObjectRestingEEGraspedEventCfg):
    """ObjectRestingEEGrasped events: object from the resting dataset, gripper closed on it from the grasp dataset."""

    reset_insertive_object_pose_from_reset_states = _with_params(
        _UR5E_RESTING_GRASPED.reset_insertive_object_pose_from_reset_states,
        dataset_dir=G1D_DATASET_DIR,
    )
    reset_end_effector_pose_from_grasp_dataset = _with_params(
        _UR5E_RESTING_GRASPED.reset_end_effector_pose_from_grasp_dataset,
        dataset_dir=G1D_DATASET_DIR,
        robot_ik_cfg=ROBOT_IK_CFG,
        gripper_cfg=GRIPPER_CFG,
    )


@configclass
class G1dObjectAnywhereEEGraspedEventCfg(G1dResetStatesBaseEventCfg, ur5e.ObjectAnywhereEEGraspedEventCfg):
    """ObjectAnywhereEEGrasped events: insertive object anywhere, gripper closed on it from the grasp dataset."""

    reset_insertive_object_pose = _INSERTIVE_ANYWHERE
    reset_end_effector_pose_from_grasp_dataset = _with_params(
        _UR5E_ANYWHERE_GRASPED.reset_end_effector_pose_from_grasp_dataset,
        dataset_dir=G1D_DATASET_DIR,
        robot_ik_cfg=ROBOT_IK_CFG,
        gripper_cfg=GRIPPER_CFG,
    )


@configclass
class G1dObjectPartiallyAssembledEEAnywhereEventCfg(
    G1dResetStatesBaseEventCfg, ur5e.ObjectPartiallyAssembledEEAnywhereEventCfg
):
    """ObjectPartiallyAssembledEEAnywhere events: assembly from its dataset, end effector at the top of its region."""

    reset_insertive_object_pose_from_partial_assembly_dataset = _with_params(
        _UR5E_PARTIAL_ANYWHERE.reset_insertive_object_pose_from_partial_assembly_dataset,
        dataset_dir=G1D_DATASET_DIR,
    )
    reset_end_effector_pose = _with_params(
        _UR5E_PARTIAL_ANYWHERE.reset_end_effector_pose,
        pose_range_b={**EE_POSE_RANGE_B, "z": (EE_Z_ROOT[1], EE_Z_ROOT[1])},
        robot_ik_cfg=ROBOT_IK_CFG,
    )


@configclass
class G1dObjectPartiallyAssembledEEGraspedEventCfg(
    G1dResetStatesBaseEventCfg, ur5e.ObjectPartiallyAssembledEEGraspedEventCfg
):
    """ObjectPartiallyAssembledEEGrasped events: assembly from its dataset, gripper closed on the insertive object."""

    reset_insertive_object_pose_from_partial_assembly_dataset = _with_params(
        _UR5E_PARTIAL_GRASPED.reset_insertive_object_pose_from_partial_assembly_dataset,
        dataset_dir=G1D_DATASET_DIR,
    )
    reset_end_effector_pose_from_grasp_dataset = _with_params(
        _UR5E_PARTIAL_GRASPED.reset_end_effector_pose_from_grasp_dataset,
        dataset_dir=G1D_DATASET_DIR,
        robot_ik_cfg=ROBOT_IK_CFG,
        gripper_cfg=GRIPPER_CFG,
    )


# ------------------------------------------------------------------------------------------------ terminations
@configclass
class G1dResetStatesTerminationCfg(ur5e.ResetStatesTerminationCfg):
    """The UR5e reset-state terminations with the success test adapted to the G1-D."""

    success = _UR5E_TERMINATIONS.success.replace(
        params={
            **_UR5E_TERMINATIONS.success.params,
            "collision_analyzer_cfgs": [
                _robot_links_only(c) for c in _UR5E_TERMINATIONS.success.params["collision_analyzer_cfgs"]
            ],
            "ee_body_name": DEX1_BASE_BODY_NAME,
            # objects (and the hand) must stay above the table top: a peg that rolled off the edge onto the
            # floor is not a usable reset state (the UR5e value is relative to its mounting plate)
            "pos_z_threshold": TABLE_TOP_Z - 0.02,
        }
    )
    """The UR5e success test sampling the right-arm and Dex1-1 links only, with the Dex1-1 base as the hand body
    and the height threshold just below the table top."""


# ------------------------------------------------------------------------------------------------- environments
@configclass
class G1dDex1ResetStatesCfg(ur5e.UR5eRobotiq2f85ResetStatesCfg):
    """Reset-state environment with the G1-D right arm and Dex1-1 gripper."""

    scene: G1dResetStatesSceneCfg = G1dResetStatesSceneCfg(num_envs=1, env_spacing=1.5)
    terminations: G1dResetStatesTerminationCfg = G1dResetStatesTerminationCfg()
    actions: G1dDex1RelativeOSCAction = G1dDex1RelativeOSCAction()


# ``configclass`` binds each class's ``__post_init__`` to whatever ``cls.__post_init__`` resolves to at decoration
# time. For the variants below that is the one inherited from ``G1dDex1ResetStatesCfg`` (first base), which skips
# the UR5e variant's ``__post_init__`` that fills in the success-term parameters (``max_object_pos_deviation``,
# assembly thresholds). Chain it explicitly.
@configclass
class G1dObjectAnywhereEEAnywhereResetStatesCfg(G1dDex1ResetStatesCfg, ur5e.ObjectAnywhereEEAnywhereResetStatesCfg):
    """ObjectAnywhereEEAnywhere reset states for the G1-D + Dex1-1."""

    events: G1dObjectAnywhereEEAnywhereEventCfg = G1dObjectAnywhereEEAnywhereEventCfg()

    def __post_init__(self):
        """Apply the UR5e variant's success-term parameters, which the configclass post-init binding skips."""
        ur5e.ObjectAnywhereEEAnywhereResetStatesCfg.__post_init__(self)


@configclass
class G1dObjectRestingEEGraspedResetStatesCfg(G1dDex1ResetStatesCfg, ur5e.ObjectRestingEEGraspedResetStatesCfg):
    """ObjectRestingEEGrasped reset states for the G1-D + Dex1-1."""

    events: G1dObjectRestingEEGraspedEventCfg = G1dObjectRestingEEGraspedEventCfg()

    def __post_init__(self):
        """Apply the UR5e variant's success-term parameters, which the configclass post-init binding skips."""
        ur5e.ObjectRestingEEGraspedResetStatesCfg.__post_init__(self)


@configclass
class G1dObjectAnywhereEEGraspedResetStatesCfg(G1dDex1ResetStatesCfg, ur5e.ObjectAnywhereEEGraspedResetStatesCfg):
    """ObjectAnywhereEEGrasped reset states for the G1-D + Dex1-1."""

    events: G1dObjectAnywhereEEGraspedEventCfg = G1dObjectAnywhereEEGraspedEventCfg()

    def __post_init__(self):
        """Apply the UR5e variant's success-term parameters, which the configclass post-init binding skips."""
        ur5e.ObjectAnywhereEEGraspedResetStatesCfg.__post_init__(self)


@configclass
class G1dObjectPartiallyAssembledEEAnywhereResetStatesCfg(
    G1dDex1ResetStatesCfg, ur5e.ObjectPartiallyAssembledEEAnywhereResetStatesCfg
):
    """ObjectPartiallyAssembledEEAnywhere reset states for the G1-D + Dex1-1."""

    events: G1dObjectPartiallyAssembledEEAnywhereEventCfg = G1dObjectPartiallyAssembledEEAnywhereEventCfg()

    def __post_init__(self):
        """Apply the UR5e variant's success-term parameters, which the configclass post-init binding skips."""
        ur5e.ObjectPartiallyAssembledEEAnywhereResetStatesCfg.__post_init__(self)


@configclass
class G1dObjectPartiallyAssembledEEGraspedResetStatesCfg(
    G1dDex1ResetStatesCfg, ur5e.ObjectPartiallyAssembledEEGraspedResetStatesCfg
):
    """ObjectPartiallyAssembledEEGrasped reset states for the G1-D + Dex1-1."""

    events: G1dObjectPartiallyAssembledEEGraspedEventCfg = G1dObjectPartiallyAssembledEEGraspedEventCfg()

    def __post_init__(self):
        """Apply the UR5e variant's success-term parameters, which the configclass post-init binding skips."""
        ur5e.ObjectPartiallyAssembledEEGraspedResetStatesCfg.__post_init__(self)
