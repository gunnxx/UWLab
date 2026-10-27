# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""State-based RL environments (stage 1 train / eval) for the G1-D right arm + Dex1-1.

Everything that is not robot- or layout-specific is inherited from the UR5e configuration; see that module for
the MDP. Changes: robot asset, table placement (robot on the floor), end-effector and gripper body names, gripper
actuator randomization joints, and the reset-state dataset location.
"""

from __future__ import annotations

from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from uwlab_assets.robots.unitree_g1d_dex1 import (
    DEX1_BASE_BODY_NAME,
    DEX1_FINGER_JOINT_REGEX,
    G1D_EE_BODY_NAME,
    IMPLICIT_G1D_DEX1,
)

from ... import mdp as task_mdp
from ..ur5e_robotiq_2f85 import rl_state_cfg as ur5e
from .actions import G1dDex1RelativeOSCAction, G1dDex1RelativeOSCEvalAction
from .layout import G1D_DATASET_DIR, G1D_TABLE


def _with_params(term, **updates):
    return term.replace(params={**term.params, **updates})


# configclass stores mutable defaults as factories, so parent defaults are read from instances, not classes
_UR5E_SCENE = ur5e.RlStateSceneCfg(num_envs=1, env_spacing=1.5)
_UR5E_BASE_EVENTS = ur5e.BaseEventCfg()
_UR5E_TRAIN_EVENTS = ur5e.TrainEventCfg()
_UR5E_TRAIN_EVAL_EVENTS = ur5e.TrainEvalEventCfg()
_UR5E_REWARDS = ur5e.RewardsCfg()


# ------------------------------------------------------------------------------------------------------- scene
@configclass
class G1dRlStateSceneCfg(ur5e.RlStateSceneCfg):
    robot = IMPLICIT_G1D_DEX1.replace(prim_path="{ENV_REGEX_NS}/Robot")

    table = G1D_TABLE

    ur5_metal_support = None

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=_UR5E_SCENE.ground.spawn,
    )


# ------------------------------------------------------------------------------------------------------ events
_GRIPPER_GAINS = _with_params(
    _UR5E_BASE_EVENTS.randomize_gripper_actuator_parameters,
    asset_cfg=SceneEntityCfg("robot", joint_names=DEX1_FINGER_JOINT_REGEX),
)


@configclass
class G1dTrainEventCfg(ur5e.TrainEventCfg):
    randomize_gripper_actuator_parameters = _GRIPPER_GAINS
    reset_from_reset_states = _with_params(_UR5E_TRAIN_EVENTS.reset_from_reset_states, dataset_dir=G1D_DATASET_DIR)


@configclass
class G1dTrainEvalEventCfg(ur5e.TrainEvalEventCfg):
    randomize_gripper_actuator_parameters = _GRIPPER_GAINS
    reset_from_reset_states = _with_params(
        _UR5E_TRAIN_EVAL_EVENTS.reset_from_reset_states, dataset_dir=G1D_DATASET_DIR
    )


# ------------------------------------------------------------------------------------------------ observations
_EE = SceneEntityCfg("robot", body_names=G1D_EE_BODY_NAME)


@configclass
class G1dObservationsCfg(ur5e.ObservationsCfg):
    @configclass
    class PolicyCfg(ur5e.ObservationsCfg.PolicyCfg):
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
class G1dRewardsCfg(ur5e.RewardsCfg):
    joint_vel = _with_params(
        _UR5E_REWARDS.joint_vel, asset_cfg=SceneEntityCfg("robot", joint_names=["right_shoulder_.*", "right_elbow_joint", "right_wrist_.*"])
    )
    ee_asset_distance = _with_params(
        _UR5E_REWARDS.ee_asset_distance, root_asset_cfg=SceneEntityCfg("robot", body_names=DEX1_BASE_BODY_NAME)
    )


# ------------------------------------------------------------------------------------------------ environments
@configclass
class G1dDex1RlStateCfg(ur5e.Ur5eRobotiq2f85RlStateCfg):
    scene: G1dRlStateSceneCfg = G1dRlStateSceneCfg(num_envs=32, env_spacing=1.5)
    observations: G1dObservationsCfg = G1dObservationsCfg()
    actions: G1dDex1RelativeOSCAction = G1dDex1RelativeOSCAction()
    rewards: G1dRewardsCfg = G1dRewardsCfg()


@configclass
class G1dDex1RelCartesianOSCTrainCfg(G1dDex1RlStateCfg):
    """Stage 1 training: implicit actuator, soft gains, four reset-state distributions."""

    events: G1dTrainEventCfg = G1dTrainEventCfg()
    actions: G1dDex1RelativeOSCAction = G1dDex1RelativeOSCAction()


@configclass
class G1dDex1RelCartesianOSCEvalCfg(G1dDex1RlStateCfg):
    """Evaluation after stage 1: ObjectAnywhereEEAnywhere resets only."""

    events: G1dTrainEvalEventCfg = G1dTrainEvalEventCfg()
    actions: G1dDex1RelativeOSCAction = G1dDex1RelativeOSCAction()


@configclass
class G1dDex1RelCartesianOSCStiffEvalCfg(G1dDex1RlStateCfg):
    """Evaluation with the stiff (end-of-curriculum) gains and small action scale."""

    events: G1dTrainEvalEventCfg = G1dTrainEvalEventCfg()
    actions: G1dDex1RelativeOSCEvalAction = G1dDex1RelativeOSCEvalAction()
