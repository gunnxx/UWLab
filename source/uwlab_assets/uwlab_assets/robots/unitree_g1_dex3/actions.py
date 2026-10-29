# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action configurations for the G1 right arm + Dex3 hand."""

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.utils import configclass

from .stance import DEX3_OPEN_RIGHT, DEX3_RIGHT_JOINT_NAMES, DEX3_SHUT_RIGHT

DEX3_HAND_BINARY_ACTIONS = BinaryJointPositionActionCfg(
    asset_name="robot",
    joint_names=DEX3_RIGHT_JOINT_NAMES,
    open_command_expr=dict(DEX3_OPEN_RIGHT),
    close_command_expr=dict(DEX3_SHUT_RIGHT),
)
"""The seven Dex3 joints driven to the robot's binary poses: open or shut (:mod:`.stance`). ``thumb_0`` has the same
target in both, so the closure checks of the reset pipeline skip it."""


@configclass
class Dex3BinaryHandAction:
    """Action terms of the Dex3 hand: one binary open/shut command."""

    gripper = DEX3_HAND_BINARY_ACTIONS
