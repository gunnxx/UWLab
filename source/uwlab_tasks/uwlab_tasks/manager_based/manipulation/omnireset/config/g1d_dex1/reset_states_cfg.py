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

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
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
    INSERTIVE_X,
    INSERTIVE_Y,
    INSERTIVE_Z_ABOVE_TABLE,
    RECEPTIVE_X,
    RECEPTIVE_Y,
    TABLE_CENTER_X,
    TABLE_ORIGIN_Z,
    TABLE_TOP_Z,
)

ROBOT_IK_CFG = SceneEntityCfg("robot", joint_names=G1D_ARM_JOINT_REGEX, body_names=DEX1_BASE_BODY_NAME)
GRIPPER_CFG = SceneEntityCfg("robot", joint_names=DEX1_FINGER_JOINT_REGEX)
EE_POSE_RANGE_B = {
    "x": EE_X,
    "y": EE_Y,
    "z": EE_Z_ROOT,
    "roll": (0.0, 0.0),
    "pitch": (np.pi / 4, 3 * np.pi / 4),
    "yaw": (np.pi / 2, 3 * np.pi / 2),
}


def _with_params(term: EventTerm, **updates) -> EventTerm:
    return term.replace(params={**term.params, **updates})


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
    robot = IMPLICIT_G1D_DEX1.replace(prim_path="{ENV_REGEX_NS}/Robot")

    table = _UR5E_SCENE.table.replace(
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(TABLE_CENTER_X, 0.0, TABLE_ORIGIN_Z), rot=(0.0, 0.0, -0.707, 0.707)
        )
    )

    ur5_metal_support = None

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=_UR5E_SCENE.ground.spawn,
    )


# ------------------------------------------------------------------------------------------------------ events
@configclass
class G1dResetStatesBaseEventCfg(ur5e.ResetStatesBaseEventCfg):
    # the robot is fixed to the floor; nothing to jitter
    reset_robot_pose = None

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


@configclass
class G1dObjectAnywhereEEAnywhereEventCfg(G1dResetStatesBaseEventCfg, ur5e.ObjectAnywhereEEAnywhereEventCfg):
    reset_insertive_object_pose = _INSERTIVE_ANYWHERE
    reset_end_effector_pose = _with_params(
        _UR5E_EE_ANYWHERE.reset_end_effector_pose,
        pose_range_b=EE_POSE_RANGE_B,
        robot_ik_cfg=ROBOT_IK_CFG,
    )


@configclass
class G1dObjectRestingEEGraspedEventCfg(G1dResetStatesBaseEventCfg, ur5e.ObjectRestingEEGraspedEventCfg):
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
    success = _UR5E_TERMINATIONS.success.replace(
        params={**_UR5E_TERMINATIONS.success.params, "ee_body_name": DEX1_BASE_BODY_NAME}
    )


# ------------------------------------------------------------------------------------------------- environments
@configclass
class G1dDex1ResetStatesCfg(ur5e.UR5eRobotiq2f85ResetStatesCfg):
    """Reset-state environment with the G1-D right arm and Dex1-1 gripper."""

    scene: G1dResetStatesSceneCfg = G1dResetStatesSceneCfg(num_envs=1, env_spacing=1.5)
    terminations: G1dResetStatesTerminationCfg = G1dResetStatesTerminationCfg()
    actions: G1dDex1RelativeOSCAction = G1dDex1RelativeOSCAction()


@configclass
class G1dObjectAnywhereEEAnywhereResetStatesCfg(G1dDex1ResetStatesCfg, ur5e.ObjectAnywhereEEAnywhereResetStatesCfg):
    events: G1dObjectAnywhereEEAnywhereEventCfg = G1dObjectAnywhereEEAnywhereEventCfg()


@configclass
class G1dObjectRestingEEGraspedResetStatesCfg(G1dDex1ResetStatesCfg, ur5e.ObjectRestingEEGraspedResetStatesCfg):
    events: G1dObjectRestingEEGraspedEventCfg = G1dObjectRestingEEGraspedEventCfg()


@configclass
class G1dObjectAnywhereEEGraspedResetStatesCfg(G1dDex1ResetStatesCfg, ur5e.ObjectAnywhereEEGraspedResetStatesCfg):
    events: G1dObjectAnywhereEEGraspedEventCfg = G1dObjectAnywhereEEGraspedEventCfg()


@configclass
class G1dObjectPartiallyAssembledEEAnywhereResetStatesCfg(
    G1dDex1ResetStatesCfg, ur5e.ObjectPartiallyAssembledEEAnywhereResetStatesCfg
):
    events: G1dObjectPartiallyAssembledEEAnywhereEventCfg = G1dObjectPartiallyAssembledEEAnywhereEventCfg()


@configclass
class G1dObjectPartiallyAssembledEEGraspedResetStatesCfg(
    G1dDex1ResetStatesCfg, ur5e.ObjectPartiallyAssembledEEGraspedResetStatesCfg
):
    events: G1dObjectPartiallyAssembledEEGraspedEventCfg = G1dObjectPartiallyAssembledEEGraspedEventCfg()
