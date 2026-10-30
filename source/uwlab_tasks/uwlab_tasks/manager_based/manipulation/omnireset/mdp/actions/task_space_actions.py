# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib
import torch
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers.action_manager import ActionTerm

from . import actions_cfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class RelCartesianOSCAction(ActionTerm):
    """Relative Cartesian OSC action term using analytical Jacobian and PD control.

    Matches the real robot's OSC implementation using calibrated analytical kinematics:
        tau = J^T @ (Kp * pose_error + Kd * vel_error)

    No inertial dynamics decoupling. Velocity is computed from J @ dq for consistency
    with the analytical Jacobian. Designed to work with DelayedDCMotor actuator.

    The flow per policy step:
        1. process_actions: scale raw 6-DOF delta, compute desired EE pose
        2. apply_actions (every physics step): compute current state, analytical J,
           PD torques, clamp, and apply as joint effort targets

    Frame convention: both EE pose and analytical Jacobian are in the robot's
    base_link frame (REP-103), consistent with the calibrated USD model.
    """

    cfg: actions_cfg.RelCartesianOSCActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""

    def __init__(self, cfg: actions_cfg.RelCartesianOSCActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Resolve joints
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_dof = len(self._joint_ids)
        # Avoid slice-vs-list indexing overhead when all joints match
        if self._num_dof == self._asset.num_joints:
            self._joint_ids = slice(None)

        # Resolve EE body
        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one match for body_name '{self.cfg.body_name}', got {len(body_ids)}: {body_names}"
            )
        self._ee_body_idx = body_ids[0]

        # Controller gains (per-env for domain randomization): Kd = 2 * sqrt(Kp) * damping_ratio
        kp = torch.tensor(cfg.motion_stiffness, device=self.device, dtype=torch.float32)
        damping_ratio = torch.tensor(cfg.motion_damping_ratio, device=self.device, dtype=torch.float32)
        kd = 2.0 * torch.sqrt(kp) * damping_ratio
        # Store defaults (1D) and expand to per-env (N, 6)
        self._kp_default = kp
        self._kd_default = kd
        self._damping_ratio_default = damping_ratio
        self._kp = kp.unsqueeze(0).expand(self.num_envs, -1).clone()
        self._kd = kd.unsqueeze(0).expand(self.num_envs, -1).clone()
        self._torque_max = torch.tensor(cfg.torque_limit, device=self.device, dtype=torch.float32)
        if self._torque_max.numel() != self._num_dof:
            raise ValueError(
                f"torque_limit has {self._torque_max.numel()} entries for {self._num_dof} joints {self._joint_names}"
            )

        # Analytical Jacobian of the controlled chain, (N, 6, num_dof) in the articulation root frame
        self._jacobian_fn = _resolve_jacobian_fn(cfg.jacobian_fn)

        # Optional null-space posture task (redundant arms)
        self._nullspace_kp = float(cfg.nullspace_stiffness)
        self._nullspace_kd = 2.0 * (self._nullspace_kp**0.5) * float(cfg.nullspace_damping_ratio)
        if self._nullspace_kp > 0.0:
            default_pos = self._asset.data.default_joint_pos.torch[0, self._joint_ids].clone()
            if cfg.nullspace_joint_pos is not None:
                for i, name in enumerate(self._joint_names):
                    if name in cfg.nullspace_joint_pos:
                        default_pos[i] = cfg.nullspace_joint_pos[name]
            self._nullspace_q = default_pos.unsqueeze(0).expand(self.num_envs, -1).clone()

        # Action scaling
        self._scale = torch.tensor(cfg.scale_xyz_axisangle, device=self.device, dtype=torch.float32)
        if cfg.input_clip is not None:
            self._input_clip = torch.tensor(cfg.input_clip, device=self.device, dtype=torch.float32)
        else:
            self._input_clip = None

        # Buffers
        self._raw_actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._ee_pos_des = torch.zeros(self.num_envs, 3, device=self.device)
        self._ee_quat_des = torch.zeros(self.num_envs, 4, device=self.device)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def process_actions(self, actions: torch.Tensor):
        """Scale raw 6-DOF deltas and compute desired EE pose for the PD tracker.

        Called once per policy step. The desired pose is held fixed while
        apply_actions recomputes torques at each physics step.
        """
        self._raw_actions[:] = actions
        scaled = actions * self._scale
        if self._input_clip is not None:
            scaled = torch.clamp(scaled, min=self._input_clip[0], max=self._input_clip[1])
        self._processed_actions[:] = scaled

        # Anchor of the delta: the previous desired pose when latched, otherwise the current EE pose in the root
        # (base_link) frame
        if self.cfg.latch_target:
            ee_pos_b, ee_quat_b = self._ee_pos_des.clone(), self._ee_quat_des.clone()
        else:
            ee_pos_b, ee_quat_b = self._get_ee_pose_root_frame()

        # Desired position = anchor + delta
        self._ee_pos_des[:] = ee_pos_b + scaled[:, :3]

        # Desired orientation: axis-angle delta -> quaternion -> compose
        delta_rot = scaled[:, 3:6]
        angle = torch.norm(delta_rot, dim=-1, keepdim=True)
        safe_angle = torch.where(angle > 1e-6, angle, torch.ones_like(angle))
        axis = delta_rot / safe_angle
        axis = torch.where(angle > 1e-6, axis, torch.zeros_like(axis))
        half = angle / 2.0
        delta_quat = torch.cat([axis * torch.sin(half), torch.cos(half)], dim=-1)
        self._ee_quat_des[:] = math_utils.quat_mul(delta_quat, ee_quat_b)

    def apply_actions(self):
        """Compute PD torques using analytical Jacobian and apply as joint efforts.

        Called every physics step (decimation times per policy step).
        """
        # Current state
        ee_pos_b, ee_quat_b = self._get_ee_pose_root_frame()
        joint_pos = self._asset.data.joint_pos.torch[:, self._joint_ids]
        joint_vel = self._asset.data.joint_vel.torch[:, self._joint_ids]

        # Analytical Jacobian (root frame, matching EE pose frame)
        jacobian = self._jacobian_fn(joint_pos, device=str(self.device))

        # EE velocity from J @ dq (consistent with analytical Jacobian)
        ee_vel = torch.bmm(jacobian, joint_vel.unsqueeze(-1)).squeeze(-1)  # (N, 6)

        # Pose error
        pos_error = self._ee_pos_des - ee_pos_b
        quat_error = math_utils.quat_mul(self._ee_quat_des, math_utils.quat_inv(ee_quat_b))
        axis_angle_error = math_utils.axis_angle_from_quat(quat_error)
        pose_error = torch.cat([pos_error, axis_angle_error], dim=-1)  # (N, 6)

        # PD control: tau = J^T @ (Kp * err + Kd * (-vel))
        vel_error = -ee_vel
        task_force = self._kp * pose_error + self._kd * vel_error
        joint_torques = torch.bmm(jacobian.transpose(-1, -2), task_force.unsqueeze(-1)).squeeze(-1)

        # Null-space posture task: tau_null projected with N = I - J^T (J J^T + lambda^2 I)^-1 J, so it never
        # disturbs the end effector (up to the damping). Only meaningful for redundant arms.
        if self._nullspace_kp > 0.0:
            tau_posture = self._nullspace_kp * (self._nullspace_q - joint_pos) - self._nullspace_kd * joint_vel
            jjt = torch.bmm(jacobian, jacobian.transpose(-1, -2))
            jjt = jjt + 1e-4 * torch.eye(6, device=self.device).unsqueeze(0)
            jt_pinv = torch.bmm(jacobian.transpose(-1, -2), torch.linalg.solve(jjt, jacobian))  # J^T (J J^T)^-1 J
            projector = torch.eye(self._num_dof, device=self.device).unsqueeze(0) - jt_pinv
            joint_torques = joint_torques + torch.bmm(projector, tau_posture.unsqueeze(-1)).squeeze(-1)

        joint_torques = torch.clamp(joint_torques, -self._torque_max, self._torque_max)

        self._asset.set_joint_effort_target(joint_torques, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset targets to current EE pose to avoid transients."""
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        ee_pos_b, ee_quat_b = self._get_ee_pose_root_frame()
        self._ee_pos_des[env_ids] = ee_pos_b[env_ids]
        self._ee_quat_des[env_ids] = ee_quat_b[env_ids]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_ee_pose_root_frame(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get EE pose in root (base_link) frame from sim state."""
        ee_pos_w = self._asset.data.body_pos_w.torch[:, self._ee_body_idx]
        ee_quat_w = self._asset.data.body_quat_w.torch[:, self._ee_body_idx]
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            self._asset.data.root_pos_w.torch,
            self._asset.data.root_quat_w.torch,
            ee_pos_w,
            ee_quat_w,
        )
        return ee_pos_b, ee_quat_b


# ===============
# Private Helpers
# ===============


def _resolve_jacobian_fn(path: str | None) -> Callable[..., torch.Tensor]:
    """Resolve the analytical Jacobian function from its import path.

    Args:
        path: Import path of the form ``"package.module:function"``, or None for the calibrated UR5e kinematics
            (:func:`uwlab_assets.robots.ur5e_robotiq_gripper.kinematics.compute_jacobian_analytical`).

    Returns:
        The Jacobian function, called as ``fn(joint_pos, device=...)``.

    Raises:
        ValueError: If ``path`` is not of the form ``"package.module:function"``.
    """
    if path is None:
        from uwlab_assets.robots.ur5e_robotiq_gripper.kinematics import compute_jacobian_analytical

        return compute_jacobian_analytical
    module_name, _, attr = path.partition(":")
    if not attr:
        raise ValueError(f"jacobian_fn must be 'package.module:function', got {path!r}")
    return getattr(importlib.import_module(module_name), attr)
