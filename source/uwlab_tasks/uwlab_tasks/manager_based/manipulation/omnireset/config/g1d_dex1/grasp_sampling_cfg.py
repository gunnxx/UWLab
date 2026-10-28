# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Grasp sampling with a floating Dex1-1 gripper (same procedure as the Robotiq one, different gripper)."""

from __future__ import annotations

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from uwlab_assets.robots.unitree_g1d_dex1 import DEX1_BASE_BODY_NAME, DEX1_GRIPPER, Dex1BinaryGripperAction

from ..ur5e_robotiq_2f85 import grasp_sampling_cfg as robotiq


@configclass
class Dex1GraspSamplingSceneCfg(robotiq.GraspSamplingSceneCfg):
    """Grasp-sampling scene with the floating Dex1-1 gripper in place of the Robotiq 2F-85."""

    robot = DEX1_GRIPPER.replace(prim_path="{ENV_REGEX_NS}/Robot")
    """The floating Dex1-1 gripper."""


# configclass stores mutable defaults as factories, so defaults are read from an instance, not the class
_ROBOTIQ_EVENTS = robotiq.GraspSamplingEventCfg()


@configclass
class Dex1GraspSamplingEventCfg(robotiq.GraspSamplingEventCfg):
    """Grasp-sampling events of the Robotiq procedure with the Dex1-1 base body as the gripper frame."""

    grasp_sampling = _ROBOTIQ_EVENTS.grasp_sampling.replace(
        params={
            **_ROBOTIQ_EVENTS.grasp_sampling.params,
            "gripper_cfg": SceneEntityCfg("robot", body_names=DEX1_BASE_BODY_NAME),
            # half of the surface samples uniform over the object: the top-biased cap grasps only work on a
            # standing peg, a lying one needs a grasp across its diameter approached from above
            "top_bias_fraction": 0.5,
        }
    )
    """The grasp-sampling event with the Dex1-1 gripper frame and half of the surface samples uniform."""


@configclass
class Dex1GraspSamplingCfg(robotiq.Robotiq2f85GraspSamplingCfg):
    """Grasp sampling environment with the Dex1-1 parallel gripper."""

    scene: Dex1GraspSamplingSceneCfg = Dex1GraspSamplingSceneCfg(num_envs=32, env_spacing=1.5)
    events: Dex1GraspSamplingEventCfg = Dex1GraspSamplingEventCfg()
    actions: Dex1BinaryGripperAction = Dex1BinaryGripperAction()
