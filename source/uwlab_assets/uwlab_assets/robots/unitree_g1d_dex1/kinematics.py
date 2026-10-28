# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Analytical kinematics of the G1-D right arm, read from the ``kinematic_chain`` block of ``metadata.yaml``.

The chain is a serial revolute chain described URDF-style: a fixed transform from the articulation root
(``AGV_link``) to the parent frame of the first arm joint, then per joint an origin (xyz, rpy) and a unit axis
expressed in the previous joint frame. All outputs are in the articulation root frame, which is the frame the
:class:`RelCartesianOSCAction` uses for the end-effector pose, so the Jacobian and the pose error are consistent.

Functions are batched over environments and run on the GPU.
"""

from __future__ import annotations

import functools
import torch
import yaml

ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
EE_BODY_NAME = "right_wrist_yaw_link"
NUM_ARM_JOINTS = 7


def compute_fk(
    joint_angles: torch.Tensor, device: str = "cuda", meta_path: str | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward kinematics of the end effector (``right_wrist_yaw_link`` origin) in the articulation root frame.

    Args:
        joint_angles: Arm joint positions (N, 7) in :data:`ARM_JOINT_NAMES` order.
        device: Device the chain tensors are created on.
        meta_path: Path to ``metadata.yaml``; ``None`` uses the packaged asset metadata.

    Returns:
        The end-effector position (N, 3) and rotation matrix (N, 3, 3).
    """
    T, _, _ = _forward_frames(joint_angles.to(torch.float32), device, meta_path)
    return T[:, :3, 3], T[:, :3, :3]


def compute_jacobian_analytical(
    joint_angles: torch.Tensor, device: str = "cuda", meta_path: str | None = None
) -> torch.Tensor:
    """Geometric Jacobian of the EE link origin, [linear; angular], in the articulation root frame.

    Column i is (z_i x (p_ee - o_i), z_i) for revolute joint i with world axis z_i through o_i.

    Args:
        joint_angles: Arm joint positions (N, 7) in :data:`ARM_JOINT_NAMES` order.
        device: Device the chain tensors are created on.
        meta_path: Path to ``metadata.yaml``; ``None`` uses the packaged asset metadata.

    Returns:
        The Jacobian (N, 6, 7).
    """
    T, axes_w, orig_w = _forward_frames(joint_angles.to(torch.float32), device, meta_path)
    p_ee = T[:, :3, 3].unsqueeze(1)  # (N,1,3)
    lin = torch.cross(axes_w, p_ee - orig_w, dim=-1)  # (N,J,3)
    J = torch.cat([lin, axes_w], dim=-1)  # (N,J,6)
    return J.transpose(1, 2).contiguous()  # (N,6,J)


def joint_limits(device: str = "cuda", meta_path: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Position limits of the arm joints as recorded in ``metadata.yaml``.

    Args:
        device: Device the chain tensors are created on.
        meta_path: Path to ``metadata.yaml``; ``None`` uses the packaged asset metadata.

    Returns:
        The lower and upper limits, each (7,), in :data:`ARM_JOINT_NAMES` order.
    """
    ch = _load_chain(device, meta_path)
    return ch["lower"], ch["upper"]


# ===============
# Private Helpers
# ===============


def _rpy_to_matrix(rpy: torch.Tensor) -> torch.Tensor:
    """Rotation matrix of URDF roll/pitch/yaw angles, ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``.

    Args:
        rpy: Roll, pitch and yaw [rad] as a (3,) tensor.

    Returns:
        The (3, 3) rotation matrix on the same device.
    """
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = torch.cos(r), torch.sin(r), torch.cos(p), torch.sin(p), torch.cos(y), torch.sin(y)
    Rz = torch.stack(
        [torch.stack([cy, -sy, cy * 0]), torch.stack([sy, cy, cy * 0]), torch.stack([cy * 0, cy * 0, cy * 0 + 1])]
    )
    Ry = torch.stack(
        [torch.stack([cp, cp * 0, sp]), torch.stack([cp * 0, cp * 0 + 1, cp * 0]), torch.stack([-sp, cp * 0, cp])]
    )
    Rx = torch.stack(
        [torch.stack([cr * 0 + 1, cr * 0, cr * 0]), torch.stack([cr * 0, cr, -sr]), torch.stack([cr * 0, sr, cr])]
    )
    return Rz @ Ry @ Rx


@functools.lru_cache(maxsize=8)
def _load_chain(device: str, meta_path: str | None = None) -> dict[str, torch.Tensor]:
    """Load the fixed transforms of the chain as tensors on ``device`` (cached per device and path).

    Args:
        device: Device the tensors are created on.
        meta_path: Path to ``metadata.yaml``; ``None`` uses the packaged asset metadata.

    Returns:
        ``T_base`` (4, 4), the per-joint ``origins`` (J, 4, 4), unit ``axes`` (J, 3) and the ``lower`` / ``upper``
        position limits (J,).

    Raises:
        RuntimeError: If the joint names in the metadata differ from :data:`ARM_JOINT_NAMES`.
    """
    if meta_path is None:
        from .paths import G1D_DEX1_METADATA_PATH

        meta_path = G1D_DEX1_METADATA_PATH
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    chain = meta["kinematic_chain"]
    names = [j["name"] for j in chain["joints"]]
    if names != ARM_JOINT_NAMES:
        raise RuntimeError(f"metadata chain {names} does not match ARM_JOINT_NAMES")
    dev = torch.device(device)
    T_base = torch.eye(4, device=dev)
    T_base[:3, :3] = _rpy_to_matrix(torch.tensor(chain["base_rpy"], device=dev))
    T_base[:3, 3] = torch.tensor(chain["base_xyz"], device=dev)
    origins = torch.eye(4, device=dev).repeat(len(names), 1, 1)
    axes = torch.zeros(len(names), 3, device=dev)
    for i, j in enumerate(chain["joints"]):
        origins[i, :3, :3] = _rpy_to_matrix(torch.tensor(j["rpy"], device=dev))
        origins[i, :3, 3] = torch.tensor(j["xyz"], device=dev)
        a = torch.tensor(j["axis"], device=dev, dtype=torch.float32)
        axes[i] = a / a.norm()
    lower = torch.tensor([j["lower"] for j in chain["joints"]], device=dev)
    upper = torch.tensor([j["upper"] for j in chain["joints"]], device=dev)
    return {"T_base": T_base, "origins": origins, "axes": axes, "lower": lower, "upper": upper}


def _axis_angle_matrix(axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Rodrigues' formula for a fixed unit axis and batched angles.

    Args:
        axis: Unit rotation axis (3,).
        theta: Rotation angles (N,) [rad].

    Returns:
        The rotation matrices (N, 3, 3).
    """
    N = theta.shape[0]
    K = torch.zeros(3, 3, device=theta.device, dtype=theta.dtype)
    K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -axis[2], axis[1], axis[2], -axis[0], -axis[1], axis[0]
    eye = torch.eye(3, device=theta.device, dtype=theta.dtype)
    s = torch.sin(theta).view(N, 1, 1)
    c = torch.cos(theta).view(N, 1, 1)
    return eye + s * K + (1 - c) * (K @ K)


def _forward_frames(
    joint_angles: torch.Tensor, device: str, meta_path: str | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Walk the chain and return the EE transform together with every joint's world axis and origin.

    Args:
        joint_angles: Arm joint positions (N, J), float32.
        device: Device the chain tensors are created on.
        meta_path: Path to ``metadata.yaml``; ``None`` uses the packaged asset metadata.

    Returns:
        The EE transform (N, 4, 4) and, per joint, the axis (N, J, 3) and origin (N, J, 3) in the root frame.
    """
    ch = _load_chain(device, meta_path)
    N, J = joint_angles.shape
    T = ch["T_base"].unsqueeze(0).expand(N, 4, 4).clone()
    axes_w = torch.zeros(N, J, 3, device=T.device, dtype=T.dtype)
    orig_w = torch.zeros(N, J, 3, device=T.device, dtype=T.dtype)
    for i in range(J):
        T = T @ ch["origins"][i]
        axes_w[:, i] = T[:, :3, :3] @ ch["axes"][i]
        orig_w[:, i] = T[:, :3, 3]
        R = torch.eye(4, device=T.device, dtype=T.dtype).repeat(N, 1, 1)
        R[:, :3, :3] = _axis_angle_matrix(ch["axes"][i], joint_angles[:, i])
        T = T @ R
    return T, axes_w, orig_w
