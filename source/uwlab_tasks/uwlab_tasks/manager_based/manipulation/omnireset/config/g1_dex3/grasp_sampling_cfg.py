# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Grasp sampling with a floating Dex3 hand (same procedure as the Robotiq one, different gripper).

The hand approaches along its palm normal and pinches between the thumb and the two fingertips; the asset's
``metadata.yaml`` holds that geometry (approach +y of the palm, pinch point 9.8 cm along the fingers and 7.2 cm
along the approach), and the sampler reads it from there.
"""

from __future__ import annotations

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

# =================================================================================================================
# Nominal Dex3 grasps of the standing peg, in the object frame (z up along the peg axis, origin at its centre).
# Tuned by hand in inspect_env.py (2026-10-28): the palm comes in at 45 degrees from above, which is the closest
# the standing G1's right arm gets to a top grasp; pitching that by 90 degrees gives the sideways grasp.
# =================================================================================================================

DEX3_PINCH_HEIGHT = 0.025
"""z (object frame, m) of the pinch point of both nominal grasps."""

DEX3_NOMINAL_TOP_GRASP = {
    "center": (0.0, 0.0, DEX3_PINCH_HEIGHT),
    "approach": (0.0, 0.0, -1.0),
    "rpy_deg": (45.0, 0.0, 0.0),
}
"""Grasp from above, palm tilted 45 degrees about the object's x axis."""

DEX3_NOMINAL_SIDE_GRASP = {
    "center": (0.0, 0.0, DEX3_PINCH_HEIGHT),
    "approach": (0.0, 0.0, -1.0),
    "rpy_deg": (45.0, 90.0, 0.0),
}
"""Sideways grasp: the top grasp pitched by 90 degrees about the object's y axis."""


@configclass
class Dex3GraspSamplingEventCfg(robotiq.GraspSamplingEventCfg):
    """Grasp-sampling events of the Robotiq procedure with the Dex3 palm as the gripper frame."""

    grasp_sampling = _ROBOTIQ_EVENTS.grasp_sampling.replace(
        params={
            **_ROBOTIQ_EVENTS.grasp_sampling.params,
            "gripper_cfg": SceneEntityCfg("robot", body_names=DEX3_PALM_BODY_NAME),
            # half of the surface samples uniform over the object: the top-biased cap grasps only work on a
            # standing peg, a lying one needs a grasp across its diameter approached from above
            "top_bias_fraction": 0.5,
            # the candidates are perturbations of the two nominal grasps (antipodal sampling cannot produce the
            # oblique top approach); the fixed grasp defaults to the top one for inspect_env.py
            "nominal_grasps": [DEX3_NOMINAL_TOP_GRASP, DEX3_NOMINAL_SIDE_GRASP],
            "fixed_grasp": {**_ROBOTIQ_EVENTS.grasp_sampling.params["fixed_grasp"], **DEX3_NOMINAL_TOP_GRASP},
        }
    )
    """The grasp-sampling event with the Dex3 palm frame, sampling around :data:`DEX3_NOMINAL_TOP_GRASP` and
    :data:`DEX3_NOMINAL_SIDE_GRASP`."""


@configclass
class Dex3GraspSamplingCfg(robotiq.Robotiq2f85GraspSamplingCfg):
    """Grasp sampling environment with the Dex3 hand."""

    scene: Dex3GraspSamplingSceneCfg = Dex3GraspSamplingSceneCfg(num_envs=32, env_spacing=1.5)
    events: Dex3GraspSamplingEventCfg = Dex3GraspSamplingEventCfg()
    actions: Dex3BinaryHandAction = Dex3BinaryHandAction()
