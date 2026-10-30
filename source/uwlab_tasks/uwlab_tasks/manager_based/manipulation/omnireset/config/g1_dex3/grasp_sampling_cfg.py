# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Grasp sampling with a floating Dex3 hand (same procedure as the Robotiq one, different gripper).

The hand approaches along its palm normal and pinches between the thumb and the two fingertips; the asset's
``metadata.yaml`` holds that geometry (approach +y of the palm, pinch point 9.8 cm along the fingers and 7.2 cm
along the approach), and the sampler reads it from there. Unlike the Robotiq and the Dex1, whose approach is laid
along the sampled pinch line, the Dex3 lays an axis 45 degrees from its squeeze axis in the palm plane along the
line, so that the squeeze axis (palm x, thumb to fingertips) crosses the pinch line at 45 degrees, across the
diagonal of a square peg, and it sweeps the hand around the pinch line with that yaw kept, so the palm comes from
above, from the side or anywhere between. Found by inspection on the big peg (2026-10-29); the standoffs stay
within the pocket's depth.
"""

from __future__ import annotations

import math

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from uwlab_assets.robots.unitree_g1_dex3 import DEX3_HAND, DEX3_PALM_BODY_NAME, Dex3BinaryHandAction

from ..ur5e_robotiq_2f85 import grasp_sampling_cfg as robotiq


@configclass
class Dex3GraspSamplingSceneCfg(robotiq.GraspSamplingSceneCfg):
    """Grasp-sampling scene with the floating Dex3 hand in place of the Robotiq 2F-85."""

    robot = DEX3_HAND.replace(prim_path="{ENV_REGEX_NS}/Robot")
    """The floating Dex3 hand."""


# configclass stores mutable defaults as factories, so defaults are read from an instance, not the class
_ROBOTIQ_EVENTS = robotiq.GraspSamplingEventCfg()

@configclass
class Dex3GraspSamplingEventCfg(robotiq.GraspSamplingEventCfg):
    """Grasp-sampling events of the Robotiq procedure with the Dex3 palm as the gripper frame."""

    grasp_sampling = _ROBOTIQ_EVENTS.grasp_sampling.replace(
        params={
            **_ROBOTIQ_EVENTS.grasp_sampling.params,
            "gripper_cfg": SceneEntityCfg("robot", body_names=DEX3_PALM_BODY_NAME),
            # surface samples uniform over the object: with the squeeze axis along the ray every face yields pinch
            # lines across the object, and the top bias only served hands that approach along the ray
            "top_bias_fraction": 0.0,
            # the hand axis laid along the pinch line: 45 degrees from the squeeze axis (palm x) toward the back of
            # the palm (-y), so the squeeze crosses the line at 45 degrees, on the square peg along its diagonal
            "grasp_align_axis": (math.sqrt(0.5), -math.sqrt(0.5), 0.0),
            # the spin axis, the same as the alignment axis: the hand sweeps around the pinch line keeping the yaw.
            # (1, 0, 0) instead would roll the hand about the squeeze axis and turn the yawed axis off the line.
            "orientation_sample_axis": (math.sqrt(0.5), -math.sqrt(0.5), 0.0),
            # standoffs over the pocket's depth about the pinch point, instead of the sweep out by an object
            # diagonal that suits hands which approach along the pinch line; the counts are the Robotiq's
            "standoffs_within_clearance": True,
        }
    )
    """The grasp-sampling event with the Dex3 palm frame: antipodal candidates with the squeeze axis across the pinch
    line at 45 degrees and the hand swept around the line."""


@configclass
class Dex3GraspSamplingCfg(robotiq.Robotiq2f85GraspSamplingCfg):
    """Grasp sampling environment with the Dex3 hand."""

    scene: Dex3GraspSamplingSceneCfg = Dex3GraspSamplingSceneCfg(num_envs=32, env_spacing=1.5)
    events: Dex3GraspSamplingEventCfg = Dex3GraspSamplingEventCfg()
    actions: Dex3BinaryHandAction = Dex3BinaryHandAction()
