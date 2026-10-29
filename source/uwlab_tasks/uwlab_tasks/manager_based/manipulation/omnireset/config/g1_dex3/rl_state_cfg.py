# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""State-based RL environments (stage 1 train / eval) for the G1 right arm + Dex3.

Everything that is not robot- or layout-specific is inherited from the UR5e configuration; see that module for
the MDP. Changes: robot asset, table placement (robot on the floor), end-effector and hand body names, hand
actuator randomization joints, and the reset-state dataset location.
"""

from __future__ import annotations

from typing import Any, TypeVar

from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from uwlab_assets.robots.unitree_g1_dex3 import (
    DEX3_JOINT_REGEX,
    DEX3_PALM_BODY_NAME,
    G1_ARM_JOINT_REGEX,
    G1_EE_BODY_NAME,
    IMPLICIT_G1_DEX3,
)

from ... import mdp as task_mdp
from ..ur5e_robotiq_2f85 import rl_state_cfg as ur5e
from .actions import G1Dex3RelativeOSCAction, G1Dex3RelativeOSCEvalAction
from .layout import G1_DEX3_DATASET_DIR, G1_DEX3_TABLE

TermT = TypeVar("TermT", EventTerm, RewTerm)
"""A manager term configuration that :func:`_with_params` copies: an event or a reward term."""


# ===============
# Private Helpers
# ===============
def _with_params(term: TermT, **updates: Any) -> TermT:
    """Copy a manager term with some of its parameters overridden.

    Args:
        term: The event or reward term to copy.
        **updates: Parameter values that replace the entries of ``term.params`` with the same key.

    Returns:
        A copy of ``term`` whose ``params`` are the original ones updated with ``updates``.
    """
    return term.replace(params={**term.params, **updates})


# configclass stores mutable defaults as factories, so parent defaults are read from instances, not classes
_UR5E_SCENE = ur5e.RlStateSceneCfg(num_envs=1, env_spacing=1.5)
_UR5E_BASE_EVENTS = ur5e.BaseEventCfg()
_UR5E_TRAIN_EVENTS = ur5e.TrainEventCfg()
_UR5E_TRAIN_EVAL_EVENTS = ur5e.TrainEvalEventCfg()
_UR5E_REWARDS = ur5e.RewardsCfg()


# ------------------------------------------------------------------------------------------------------- scene
@configclass
class G1Dex3RlStateSceneCfg(ur5e.RlStateSceneCfg):
    """The UR5e RL scene with the standing G1 in front of the plain table block."""

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
_GRIPPER_GAINS = _with_params(
    _UR5E_BASE_EVENTS.randomize_gripper_actuator_parameters,
    asset_cfg=SceneEntityCfg("robot", joint_names=DEX3_JOINT_REGEX),
)
"""The UR5e gripper-actuator randomization (gains scaled by 0.5 to 2) applied to the Dex3 joints."""


@configclass
class G1Dex3TrainEventCfg(ur5e.TrainEventCfg):
    """The UR5e training events with the Dex3 joints and the G1 + Dex3 reset-state datasets."""

    randomize_gripper_actuator_parameters = _GRIPPER_GAINS
    reset_from_reset_states = _with_params(_UR5E_TRAIN_EVENTS.reset_from_reset_states, dataset_dir=G1_DEX3_DATASET_DIR)


@configclass
class G1Dex3TrainEvalEventCfg(ur5e.TrainEvalEventCfg):
    """The UR5e evaluation events with the Dex3 joints and the G1 + Dex3 reset-state datasets."""

    randomize_gripper_actuator_parameters = _GRIPPER_GAINS
    reset_from_reset_states = _with_params(
        _UR5E_TRAIN_EVAL_EVENTS.reset_from_reset_states, dataset_dir=G1_DEX3_DATASET_DIR
    )


# ------------------------------------------------------------------------------------------------ observations
_EE = SceneEntityCfg("robot", body_names=G1_EE_BODY_NAME)
"""The G1 end-effector body (the wrist flange the OSC controls) the pose and velocity observations use."""


@configclass
class G1Dex3ObservationsCfg(ur5e.ObservationsCfg):
    """The UR5e observations with the G1 end-effector body."""

    @configclass
    class PolicyCfg(ur5e.ObservationsCfg.PolicyCfg):
        """Policy observations: the UR5e terms with the end-effector frame of the G1."""

        end_effector_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={"target_asset_cfg": _EE, "root_asset_cfg": SceneEntityCfg("robot"), "rotation_repr": "axis_angle"},
        )
        insertive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": _EE,
                "rotation_repr": "axis_angle",
            },
        )
        receptive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("receptive_object"),
                "root_asset_cfg": _EE,
                "rotation_repr": "axis_angle",
            },
        )

    @configclass
    class CriticCfg(ur5e.ObservationsCfg.CriticCfg):
        """Critic observations: the UR5e terms with the end-effector frame of the G1."""

        end_effector_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={"target_asset_cfg": _EE, "root_asset_cfg": SceneEntityCfg("robot"), "rotation_repr": "axis_angle"},
        )
        insertive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": _EE,
                "rotation_repr": "axis_angle",
            },
        )
        receptive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("receptive_object"),
                "root_asset_cfg": _EE,
                "rotation_repr": "axis_angle",
            },
        )
        end_effector_vel_lin_ang_b = ObsTerm(
            func=task_mdp.asset_link_velocity_in_root_asset_frame,
            params={"target_asset_cfg": _EE, "root_asset_cfg": SceneEntityCfg("robot")},
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg | None = None


# ----------------------------------------------------------------------------------------------------- rewards
@configclass
class G1Dex3RewardsCfg(ur5e.RewardsCfg):
    """The UR5e rewards with the G1 right-arm joints and the Dex3 pinch point (``gripper_offset`` of the palm)."""

    joint_vel = _with_params(_UR5E_REWARDS.joint_vel, asset_cfg=SceneEntityCfg("robot", joint_names=G1_ARM_JOINT_REGEX))
    ee_asset_distance = _with_params(
        _UR5E_REWARDS.ee_asset_distance, root_asset_cfg=SceneEntityCfg("robot", body_names=DEX3_PALM_BODY_NAME)
    )


# ------------------------------------------------------------------------------------------------ environments
@configclass
class G1Dex3RlStateCfg(ur5e.Ur5eRobotiq2f85RlStateCfg):
    """State-based RL environment with the G1 right arm and Dex3 hand."""

    scene: G1Dex3RlStateSceneCfg = G1Dex3RlStateSceneCfg(num_envs=32, env_spacing=1.5)
    observations: G1Dex3ObservationsCfg = G1Dex3ObservationsCfg()
    actions: G1Dex3RelativeOSCAction = G1Dex3RelativeOSCAction()
    rewards: G1Dex3RewardsCfg = G1Dex3RewardsCfg()


@configclass
class G1Dex3RelCartesianOSCTrainCfg(G1Dex3RlStateCfg):
    """Stage 1 training: implicit actuator, soft gains, four reset-state distributions."""

    events: G1Dex3TrainEventCfg = G1Dex3TrainEventCfg()
    actions: G1Dex3RelativeOSCAction = G1Dex3RelativeOSCAction()


@configclass
class G1Dex3RelCartesianOSCEvalCfg(G1Dex3RlStateCfg):
    """Evaluation after stage 1: ObjectAnywhereEEAnywhere resets only."""

    events: G1Dex3TrainEvalEventCfg = G1Dex3TrainEvalEventCfg()
    actions: G1Dex3RelativeOSCAction = G1Dex3RelativeOSCAction()


@configclass
class G1Dex3RelCartesianOSCStiffEvalCfg(G1Dex3RlStateCfg):
    """Evaluation with the stiff (end-of-curriculum) gains and small action scale."""

    events: G1Dex3TrainEvalEventCfg = G1Dex3TrainEvalEventCfg()
    actions: G1Dex3RelativeOSCEvalAction = G1Dex3RelativeOSCEvalAction()
