# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analytical kinematics of the G1 right arm, read from the ``kinematic_chain`` block of this robot's ``metadata.yaml``.

The right arm of the G1 is the right arm of the G1-D (same joints, same flange link ``right_wrist_yaw_link``);
only the fixed transform from the articulation root (``pelvis``, with the frozen waist) to the shoulder differs,
and that is what the metadata records. The chain walker of :mod:`uwlab_assets.robots.unitree_g1d_dex1.kinematics`
is therefore reused with this package's metadata; outputs are in the articulation root frame, batched over
environments, on the GPU.
"""

from __future__ import annotations

import torch

from uwlab_assets.robots.unitree_g1d_dex1 import kinematics as _g1d

from .paths import G1_DEX3_METADATA_PATH

ARM_JOINT_NAMES = _g1d.ARM_JOINT_NAMES
"""The seven right-arm joints, shoulder to wrist, the order of every joint tensor here."""

EE_BODY_NAME = _g1d.EE_BODY_NAME
"""Flange link whose origin the kinematics describe."""

NUM_ARM_JOINTS = _g1d.NUM_ARM_JOINTS
"""Number of arm joints."""


def compute_fk(joint_angles: torch.Tensor, device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """Forward kinematics of the end effector (``right_wrist_yaw_link`` origin) in the articulation root frame.

    Args:
        joint_angles: Arm joint positions (N, 7) in :data:`ARM_JOINT_NAMES` order.
        device: Device the chain tensors are created on.

    Returns:
        The end-effector position (N, 3) and rotation matrix (N, 3, 3).
    """
    return _g1d.compute_fk(joint_angles, device, G1_DEX3_METADATA_PATH)


def compute_jacobian_analytical(joint_angles: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """Geometric Jacobian of the EE link origin, [linear; angular], in the articulation root frame.

    Args:
        joint_angles: Arm joint positions (N, 7) in :data:`ARM_JOINT_NAMES` order.
        device: Device the chain tensors are created on.

    Returns:
        The Jacobian (N, 6, 7).
    """
    return _g1d.compute_jacobian_analytical(joint_angles, device, G1_DEX3_METADATA_PATH)


def joint_limits(device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """Position limits of the arm joints as recorded in ``metadata.yaml``.

    Args:
        device: Device the chain tensors are created on.

    Returns:
        The lower and upper limits, each (7,), in :data:`ARM_JOINT_NAMES` order.
    """
    return _g1d.joint_limits(device, G1_DEX3_METADATA_PATH)
