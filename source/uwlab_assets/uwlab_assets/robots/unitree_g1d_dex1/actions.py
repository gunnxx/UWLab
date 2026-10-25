# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action configurations for the G1-D right arm + Dex1-1 gripper."""

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.utils import configclass

from .g1d_dex1 import DEX1_FINGER_CLOSED, DEX1_FINGER_JOINT_NAMES, DEX1_FINGER_OPEN

DEX1_GRIPPER_BINARY_ACTIONS = BinaryJointPositionActionCfg(
    asset_name="robot",
    joint_names=DEX1_FINGER_JOINT_NAMES,
    open_command_expr={n: DEX1_FINGER_OPEN for n in DEX1_FINGER_JOINT_NAMES},
    close_command_expr={n: DEX1_FINGER_CLOSED for n in DEX1_FINGER_JOINT_NAMES},
)
"""Both prismatic fingers driven to the same target: fully open (95 mm gap) or fully closed (6 mm gap)."""


@configclass
class Dex1BinaryGripperAction:
    gripper = DEX1_GRIPPER_BINARY_ACTIONS
