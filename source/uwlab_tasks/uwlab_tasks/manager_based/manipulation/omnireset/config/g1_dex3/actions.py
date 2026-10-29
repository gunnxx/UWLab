# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Relative Cartesian OSC + binary Dex3 hand actions for the G1 right arm."""

from __future__ import annotations

from isaaclab.utils import configclass

from uwlab_assets.robots.unitree_g1_dex3 import (
    DEX3_HAND_BINARY_ACTIONS,
    G1_ARM_JOINT_NAMES,
    G1_DEFAULT_ARM_JOINT_POS,
    G1_EE_BODY_NAME,
    G1_EFFORT_LIMITS,
)

from ...mdp.actions.actions_cfg import RelCartesianOSCActionCfg

G1_JACOBIAN_FN = "uwlab_assets.robots.unitree_g1_dex3.kinematics:compute_jacobian_analytical"
"""Import path of the analytical Jacobian of the G1 right arm used by the OSC action."""

G1_TORQUE_LIMIT = tuple(G1_EFFORT_LIMITS[n] for n in G1_ARM_JOINT_NAMES)
"""Per-joint torque clamp of the OSC action: the motor effort limits of the seven right-arm joints."""

G1_DEX3_RELATIVE_OSC = RelCartesianOSCActionCfg(
    asset_name="robot",
    joint_names=G1_ARM_JOINT_NAMES,
    body_name=G1_EE_BODY_NAME,
    scale_xyz_axisangle=(0.02, 0.02, 0.02, 0.02, 0.02, 0.2),
    motion_stiffness=(200.0, 200.0, 200.0, 3.0, 3.0, 3.0),
    # No explicit task-space damping: unstable on the light G1 wrist links; the damping is provided implicitly by
    # the arm actuator (see IMPLICIT_G1_DEX3 in uwlab_assets).
    motion_damping_ratio=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    torque_limit=G1_TORQUE_LIMIT,
    jacobian_fn=G1_JACOBIAN_FN,
    # 7-DoF arm: hold the default elbow configuration in the null space of the task
    nullspace_stiffness=2.0,
    nullspace_damping_ratio=1.0,
    nullspace_joint_pos=G1_DEFAULT_ARM_JOINT_POS,
)
"""Pre-train gains: the G1-D recipe, this being the same arm."""

G1_DEX3_RELATIVE_OSC_EVAL = G1_DEX3_RELATIVE_OSC.replace(
    scale_xyz_axisangle=(0.01, 0.01, 0.002, 0.02, 0.02, 0.2),
    motion_stiffness=(1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0),
    motion_damping_ratio=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
)
"""Eval / sim2real gains (end-of-curriculum values, mirrors the UR5e recipe)."""

G1_DEX3_RELATIVE_OSC_UNSCALED = G1_DEX3_RELATIVE_OSC_EVAL.replace(
    scale_xyz_axisangle=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
)
"""Unscaled (sysid scripts)."""


@configclass
class G1Dex3RelativeOSCAction:
    """Analytical OSC on the 7 right-arm joints + binary Dex3 hand."""

    arm = G1_DEX3_RELATIVE_OSC
    """Relative Cartesian OSC action of the right arm with the pre-train gains."""
    gripper = DEX3_HAND_BINARY_ACTIONS
    """Binary open / shut action of the Dex3 hand."""


@configclass
class G1Dex3RelativeOSCEvalAction:
    """Eval / sim2real variant: OSC with the stiff end-of-curriculum gains + binary Dex3 hand."""

    arm = G1_DEX3_RELATIVE_OSC_EVAL
    """Relative Cartesian OSC action of the right arm with the eval gains and action scale."""
    gripper = DEX3_HAND_BINARY_ACTIONS
    """Binary open / shut action of the Dex3 hand."""


@configclass
class G1Dex3SysidOSCAction:
    """Sysid variant: OSC with the eval gains and an unscaled action + binary Dex3 hand."""

    arm = G1_DEX3_RELATIVE_OSC_UNSCALED
    """Relative Cartesian OSC action of the right arm with the eval gains and a unit action scale."""
    gripper = DEX3_HAND_BINARY_ACTIONS
    """Binary open / shut action of the Dex3 hand."""
