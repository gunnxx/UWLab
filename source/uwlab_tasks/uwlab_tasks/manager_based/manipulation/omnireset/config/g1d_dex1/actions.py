# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Relative Cartesian OSC + binary Dex1-1 gripper actions for the G1-D right arm."""

from __future__ import annotations

from isaaclab.utils import configclass

from uwlab_assets.robots.unitree_g1d_dex1 import (
    DEX1_GRIPPER_BINARY_ACTIONS,
    G1D_ARM_JOINT_NAMES,
    G1D_DEFAULT_ARM_JOINT_POS,
    G1D_EE_BODY_NAME,
    G1D_EFFORT_LIMITS,
)

from ...mdp.actions.actions_cfg import RelCartesianOSCActionCfg

G1D_JACOBIAN_FN = "uwlab_assets.robots.unitree_g1d_dex1.kinematics:compute_jacobian_analytical"
"""Import path of the analytical Jacobian of the G1-D right arm used by the OSC action."""
G1D_TORQUE_LIMIT = tuple(G1D_EFFORT_LIMITS[n] for n in G1D_ARM_JOINT_NAMES)
"""Per-joint torque clamp of the OSC action: the motor effort limits of the seven right-arm joints."""

G1D_DEX1_RELATIVE_OSC = RelCartesianOSCActionCfg(
    asset_name="robot",
    joint_names=G1D_ARM_JOINT_NAMES,
    body_name=G1D_EE_BODY_NAME,
    scale_xyz_axisangle=(0.02, 0.02, 0.02, 0.02, 0.02, 0.2),
    motion_stiffness=(200.0, 200.0, 200.0, 3.0, 3.0, 3.0),
    # Translational task-space damping as on the UR5e (ratio 3 gives the G1-D the UR5e's ~40 mm/s cruise speed
    # instead of ~230 mm/s). No rotational task-space damping: on the light G1-D wrist links it chatters
    # (Kd dt / I ~ 7, stable below 2), so the rotation is damped by the implicit wrist joint damping instead
    # (G1D_ARM_JOINT_DAMPING in uwlab_assets). See docs/notes/omnireset_osc_two_arms.html.
    motion_damping_ratio=(3.0, 3.0, 3.0, 0.0, 0.0, 0.0),
    torque_limit=G1D_TORQUE_LIMIT,
    jacobian_fn=G1D_JACOBIAN_FN,
    # 7-DoF arm: hold the default elbow configuration in the null space of the task
    nullspace_stiffness=2.0,
    nullspace_damping_ratio=1.0,
    nullspace_joint_pos=G1D_DEFAULT_ARM_JOINT_POS,
)
"""Pre-train gains. Same stiffness, scale and translational damping as the UR5e recipe. The rotational damping is
in the arm actuator, and the torque clamp is the per-joint motor limit (25 N m, wrists 5 N m)."""

G1D_DEX1_RELATIVE_OSC_EVAL = G1D_DEX1_RELATIVE_OSC.replace(
    scale_xyz_axisangle=(0.01, 0.01, 0.002, 0.02, 0.02, 0.2),
    motion_stiffness=(1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0),
    motion_damping_ratio=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
)
"""Eval / sim2real gains (end-of-curriculum values, mirrors the UR5e recipe)."""

G1D_DEX1_RELATIVE_OSC_UNSCALED = G1D_DEX1_RELATIVE_OSC_EVAL.replace(
    scale_xyz_axisangle=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
)
"""Unscaled (sysid scripts)."""


@configclass
class G1dDex1RelativeOSCAction:
    """Analytical OSC on the 7 right-arm joints + binary Dex1-1 gripper."""

    arm = G1D_DEX1_RELATIVE_OSC
    """Relative Cartesian OSC action of the right arm with the pre-train gains."""
    gripper = DEX1_GRIPPER_BINARY_ACTIONS
    """Binary open / close action of the Dex1-1 gripper."""


@configclass
class G1dDex1RelativeOSCEvalAction:
    """Eval / sim2real variant: OSC with the stiff end-of-curriculum gains + binary Dex1-1 gripper."""

    arm = G1D_DEX1_RELATIVE_OSC_EVAL
    """Relative Cartesian OSC action of the right arm with the eval gains and action scale."""
    gripper = DEX1_GRIPPER_BINARY_ACTIONS
    """Binary open / close action of the Dex1-1 gripper."""


@configclass
class G1dDex1SysidOSCAction:
    """Sysid variant: OSC with the eval gains and an unscaled action + binary Dex1-1 gripper."""

    arm = G1D_DEX1_RELATIVE_OSC_UNSCALED
    """Relative Cartesian OSC action of the right arm with the eval gains and a unit action scale."""
    gripper = DEX1_GRIPPER_BINARY_ACTIONS
    """Binary open / close action of the Dex1-1 gripper."""
