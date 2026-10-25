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
G1D_TORQUE_LIMIT = tuple(G1D_EFFORT_LIMITS[n] for n in G1D_ARM_JOINT_NAMES)

# Pre-train gains. Same task-space gains as the UR5e recipe: the G1 arm is ~5x lighter, so the loop is if
# anything faster; the torque clamp is the per-joint motor limit (25 N m, wrists 5 N m).
G1D_DEX1_RELATIVE_OSC = RelCartesianOSCActionCfg(
    asset_name="robot",
    joint_names=G1D_ARM_JOINT_NAMES,
    body_name=G1D_EE_BODY_NAME,
    scale_xyz_axisangle=(0.02, 0.02, 0.02, 0.02, 0.02, 0.2),
    motion_stiffness=(200.0, 200.0, 200.0, 3.0, 3.0, 3.0),
    motion_damping_ratio=(3.0, 3.0, 3.0, 1.0, 1.0, 1.0),
    torque_limit=G1D_TORQUE_LIMIT,
    jacobian_fn=G1D_JACOBIAN_FN,
    # 7-DoF arm: hold the default elbow configuration in the null space of the task
    nullspace_stiffness=2.0,
    nullspace_damping_ratio=1.0,
    nullspace_joint_pos=G1D_DEFAULT_ARM_JOINT_POS,
)

# Eval / sim2real gains (end-of-curriculum values, mirrors the UR5e recipe)
G1D_DEX1_RELATIVE_OSC_EVAL = G1D_DEX1_RELATIVE_OSC.replace(
    scale_xyz_axisangle=(0.01, 0.01, 0.002, 0.02, 0.02, 0.2),
    motion_stiffness=(1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0),
    motion_damping_ratio=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
)

# Unscaled (sysid scripts)
G1D_DEX1_RELATIVE_OSC_UNSCALED = G1D_DEX1_RELATIVE_OSC_EVAL.replace(
    scale_xyz_axisangle=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
)


@configclass
class G1dDex1RelativeOSCAction:
    """Analytical OSC on the 7 right-arm joints + binary Dex1-1 gripper."""

    arm = G1D_DEX1_RELATIVE_OSC
    gripper = DEX1_GRIPPER_BINARY_ACTIONS


@configclass
class G1dDex1RelativeOSCEvalAction:
    arm = G1D_DEX1_RELATIVE_OSC_EVAL
    gripper = DEX1_GRIPPER_BINARY_ACTIONS


@configclass
class G1dDex1SysidOSCAction:
    arm = G1D_DEX1_RELATIVE_OSC_UNSCALED
    gripper = DEX1_GRIPPER_BINARY_ACTIONS
