# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP functions for manipulation tasks."""

import torch

from isaaclab.assets import BaseArticulation, BaseRigidObject, BaseRigidObjectCollection
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg
from isaaclab.sim.utils import enable_extension
from isaaclab.utils import math as math_utils

from uwlab_tasks.manager_based.manipulation.omnireset.mdp import utils

from ..assembly_keypoints import Offset
from .collision_analyzer_cfg import CollisionAnalyzerCfg


def _check_obb_overlap(centroids_a, axes_a, half_extents_a, centroids_b, axes_b, half_extents_b) -> torch.Tensor:
    """
    OBB overlap check.

    Args:
        centroids_a: Centers of OBB A for all envs (num_envs, 3) - torch tensor on GPU
        axes_a: Orientation axes of OBB A for all envs (num_envs, 3, 3) - torch tensor on GPU
        half_extents_a: Half extents of OBB A (3,) - torch tensor on GPU
        centroids_b: Centers of OBB B for all envs (num_envs, 3) - torch tensor on GPU
        axes_b: Orientation axes of OBB B for all envs (num_envs, 3, 3) - torch tensor on GPU
        half_extents_b: Half extents of OBB B (3,) - torch tensor on GPU

    Returns:
        torch.Tensor: Boolean tensor (num_envs,) indicating overlap for each environment
    """
    num_envs = centroids_a.shape[0]
    device = centroids_a.device

    # Vector between centroids for all envs (num_envs, 3)
    d = centroids_b - centroids_a

    # Matrix C = A^T * B (rotation from A to B) for all envs (num_envs, 3, 3)
    C = torch.bmm(axes_a.transpose(1, 2), axes_b)
    abs_C = torch.abs(C)

    # Initialize overlap results (assume all overlap initially)
    overlap_results = torch.ones(num_envs, device=device, dtype=torch.bool)

    # Test all axes of A at once (vectorized across all 3 axes and all environments)
    # axes_a: (num_envs, 3, 3), d: (num_envs, 3) -> projections: (num_envs, 3)
    projections_on_axes_a = torch.abs(torch.bmm(d.unsqueeze(1), axes_a).squeeze(1))  # (num_envs, 3)
    ra_all = half_extents_a.unsqueeze(0).expand(num_envs, -1)  # (num_envs, 3)
    rb_all = torch.sum(half_extents_b.unsqueeze(0).unsqueeze(0) * abs_C, dim=2)  # (num_envs, 3)
    no_overlap_a = projections_on_axes_a > (ra_all + rb_all)  # (num_envs, 3)
    overlap_results &= ~torch.any(no_overlap_a, dim=1)  # (num_envs,)

    # Test all axes of B at once (vectorized across all 3 axes and all environments)
    # axes_b: (num_envs, 3, 3), d: (num_envs, 3) -> projections: (num_envs, 3)
    projections_on_axes_b = torch.abs(torch.bmm(d.unsqueeze(1), axes_b).squeeze(1))  # (num_envs, 3)
    ra_all_b = torch.sum(half_extents_a.unsqueeze(0).unsqueeze(0) * abs_C.transpose(1, 2), dim=2)  # (num_envs, 3)
    rb_all_b = half_extents_b.unsqueeze(0).expand(num_envs, -1)  # (num_envs, 3)
    no_overlap_b = projections_on_axes_b > (ra_all_b + rb_all_b)  # (num_envs, 3)
    overlap_results &= ~torch.any(no_overlap_b, dim=1)  # (num_envs,)

    # Test all cross products at once (9 cross products per environment)
    # Reshape axes for broadcasting: axes_a (num_envs, 3, 1, 3), axes_b (num_envs, 1, 3, 3)
    axes_a_expanded = axes_a.unsqueeze(2)  # (num_envs, 3, 1, 3)
    axes_b_expanded = axes_b.unsqueeze(1)  # (num_envs, 1, 3, 3)

    # Compute all 9 cross products at once: (num_envs, 3, 3, 3)
    cross_products = torch.cross(axes_a_expanded, axes_b_expanded, dim=3)  # (num_envs, 3, 3, 3)

    # Compute norms and filter out near-parallel axes: (num_envs, 3, 3)
    cross_norms = torch.norm(cross_products, dim=3)  # (num_envs, 3, 3)
    valid_crosses = cross_norms > 1e-6  # (num_envs, 3, 3)

    # Normalize cross products (set invalid ones to zero)
    normalized_crosses = torch.where(
        valid_crosses.unsqueeze(3),
        cross_products / cross_norms.unsqueeze(3).clamp(min=1e-6),
        torch.zeros_like(cross_products),
    )  # (num_envs, 3, 3, 3)

    # Project d onto all cross product axes: (num_envs, 3, 3)
    d_expanded = d.unsqueeze(1).unsqueeze(1)  # (num_envs, 1, 1, 3)
    projections_cross = torch.abs(torch.sum(d_expanded * normalized_crosses, dim=3))  # (num_envs, 3, 3)

    # Compute ra for all cross products: (num_envs, 3, 3)
    # half_extents_a: (3,), axes_a: (num_envs, 3, 3), normalized_crosses: (num_envs, 3, 3, 3)
    axes_a_cross_dots = torch.abs(
        torch.sum(axes_a.unsqueeze(1).unsqueeze(1) * normalized_crosses.unsqueeze(3), dim=4)
    )  # (num_envs, 3, 3, 3)
    ra_cross = torch.sum(
        half_extents_a.unsqueeze(0).unsqueeze(0).unsqueeze(0) * axes_a_cross_dots, dim=3
    )  # (num_envs, 3, 3)

    # Compute rb for all cross products: (num_envs, 3, 3)
    axes_b_cross_dots = torch.abs(
        torch.sum(axes_b.unsqueeze(1).unsqueeze(1) * normalized_crosses.unsqueeze(4), dim=4)
    )  # (num_envs, 3, 3, 3)
    rb_cross = torch.sum(
        half_extents_b.unsqueeze(0).unsqueeze(0).unsqueeze(0) * axes_b_cross_dots, dim=3
    )  # (num_envs, 3, 3)

    # Check separating condition for all cross products: (num_envs, 3, 3)
    no_overlap_cross = projections_cross > (ra_cross + rb_cross)  # (num_envs, 3, 3)
    # Only consider valid cross products
    no_overlap_cross_valid = no_overlap_cross & valid_crosses  # (num_envs, 3, 3)
    overlap_results &= ~torch.any(no_overlap_cross_valid.view(num_envs, -1), dim=1)  # (num_envs,)

    return overlap_results


class check_grasp_success(ManagerTermBase):
    """Check if grasp is successful based on object stability, gripper closure, and collision detection."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._env = env

        self.object_cfg = cfg.params.get("object_cfg")
        self.gripper_cfg = cfg.params.get("gripper_cfg")
        self.collision_analyzer_cfg = cfg.params.get("collision_analyzer_cfg")
        self.collision_analyzer = self.collision_analyzer_cfg.class_type(self.collision_analyzer_cfg, self._env)
        self.max_pos_deviation = cfg.params.get("max_pos_deviation")
        self.pos_z_threshold = cfg.params.get("pos_z_threshold")
        self.consecutive_stability_steps = cfg.params.get("consecutive_stability_steps", 5)

        # The fingers must have travelled at least this fraction of the open-to-close range: an object lying
        # across fully open fingers does not move either, but it is not grasped. Uses the task's gripper action.
        self.min_close_fraction = cfg.params.get("min_close_fraction", 0.1)
        self._closure_joint_ids: list[int] | None = None
        gripper_action_cfg = getattr(getattr(env.cfg, "actions", None), "gripper", None)
        close_expr = getattr(gripper_action_cfg, "close_command_expr", None)
        open_expr = getattr(gripper_action_cfg, "open_command_expr", None)
        if close_expr and open_expr and self.min_close_fraction > 0:
            gripper_asset = env.scene[self.gripper_cfg.name]
            names = [n for n in close_expr if n in open_expr and abs(open_expr[n] - close_expr[n]) > 1e-6]
            if names:
                self._closure_joint_ids, _ = gripper_asset.find_joints(names, preserve_order=True)
                self._closure_open = torch.tensor([open_expr[n] for n in names], device=env.device)
                self._closure_close = torch.tensor([close_expr[n] for n in names], device=env.device)

        self.stability_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)

        object_asset = self._env.scene[self.object_cfg.name]
        if not hasattr(object_asset, "initial_pos"):
            object_asset.initial_pos = object_asset.data.root_pos_w.torch.clone()
            object_asset.initial_quat = object_asset.data.root_quat_w.torch.clone()
        else:
            object_asset.initial_pos[env_ids] = object_asset.data.root_pos_w.torch[env_ids].clone()
            object_asset.initial_quat[env_ids] = object_asset.data.root_quat_w.torch[env_ids].clone()

        if env_ids is None:
            self.stability_counter.zero_()
        else:
            self.stability_counter[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedEnv,
        object_cfg: SceneEntityCfg,
        gripper_cfg: SceneEntityCfg,
        collision_analyzer_cfg: CollisionAnalyzerCfg,
        max_pos_deviation: float = 0.05,
        pos_z_threshold: float = 0.05,
        consecutive_stability_steps: int = 5,
        min_close_fraction: float = 0.1,
    ) -> torch.Tensor:
        # Get object and gripper from scene
        object_asset = env.scene[self.object_cfg.name]
        gripper_asset = env.scene[self.gripper_cfg.name]

        # Check time out
        time_out = env.episode_length_buf >= env.max_episode_length

        # Check for abnormal gripper state (excessive joint velocities)
        abnormal_gripper_state = (
            gripper_asset.data.joint_vel.torch.abs() > (gripper_asset.data.joint_vel_limits.torch * 2)
        ).any(dim=1)

        # Check if asset velocities are small
        current_step_stable = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        # Check gripper (articulation) velocities
        current_step_stable &= gripper_asset.data.joint_vel.torch.abs().sum(dim=1) < 5.0
        # Check object (rigid object) velocities
        if isinstance(object_asset, BaseRigidObject):
            current_step_stable &= object_asset.data.body_lin_vel_w.torch.abs().sum(dim=2).sum(dim=1) < 0.05
            current_step_stable &= object_asset.data.body_ang_vel_w.torch.abs().sum(dim=2).sum(dim=1) < 1.0
        elif isinstance(object_asset, BaseRigidObjectCollection):
            current_step_stable &= object_asset.data.object_lin_vel_w.torch.abs().sum(dim=2).sum(dim=1) < 0.05
            current_step_stable &= object_asset.data.object_ang_vel_w.torch.abs().sum(dim=2).sum(dim=1) < 1.0

        self.stability_counter = torch.where(
            current_step_stable,
            self.stability_counter + 1,  # Increment counter if stable
            torch.zeros_like(self.stability_counter),  # Reset counter if not stable
        )

        stability_reached = self.stability_counter >= self.consecutive_stability_steps

        # Skip if position or quaternion is NaN
        pos_is_nan = torch.isnan(object_asset.data.root_pos_w.torch).any(dim=1)
        quat_is_nan = torch.isnan(object_asset.data.root_quat_w.torch).any(dim=1)
        skip_check = pos_is_nan | quat_is_nan

        # Object has excessive pose deviation if position exceeds thresholds
        pos_deviation = (object_asset.data.root_pos_w.torch - object_asset.initial_pos).norm(dim=1)
        valid_pos_deviation = torch.where(~skip_check, pos_deviation, torch.zeros_like(pos_deviation))
        excessive_pose_deviation = valid_pos_deviation > self.max_pos_deviation

        # Object is above ground if position is greater than z threshold
        pos_above_ground = object_asset.data.root_pos_w.torch[:, 2] >= self.pos_z_threshold

        # Check for collisions between gripper and object
        all_env_ids = torch.arange(env.num_envs, device=env.device)
        collision_free = self.collision_analyzer(env, all_env_ids)

        grasp_success = (
            (~abnormal_gripper_state)
            & stability_reached
            & (~excessive_pose_deviation)
            & pos_above_ground
            & collision_free
            & time_out
        )

        if self._closure_joint_ids is not None:
            finger_pos = gripper_asset.data.joint_pos.torch[:, self._closure_joint_ids]
            closed_fraction = (finger_pos - self._closure_open) / (self._closure_close - self._closure_open)
            # the mean over the closure joints: on a multi-finger hand (Dex3) the object can lever one joint past
            # its open target while the others close on it, and that is still a grasp
            grasp_success &= closed_fraction.mean(dim=1) > self.min_close_fraction

        return grasp_success


class check_reset_state_success(ManagerTermBase):
    """Check if grasp is successful based on object stability, gripper closure, and collision detection."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._env = env

        self.object_cfgs = cfg.params.get("object_cfgs")
        self.robot_cfg = cfg.params.get("robot_cfg")
        self.ee_body_name = cfg.params.get("ee_body_name")
        self.collision_analyzer_cfgs = cfg.params.get("collision_analyzer_cfgs")
        self.collision_analyzers = [
            collision_analyzer_cfg.class_type(collision_analyzer_cfg, self._env)
            for collision_analyzer_cfg in self.collision_analyzer_cfgs
        ]
        self.max_robot_pos_deviation = cfg.params.get("max_robot_pos_deviation")
        self.max_object_pos_deviation = cfg.params.get("max_object_pos_deviation")
        self.pos_z_threshold = cfg.params.get("pos_z_threshold")
        self.consecutive_stability_steps = cfg.params.get("consecutive_stability_steps", 5)
        self.approach_z_max = cfg.params.get("approach_z_max", -0.5)

        # Load gripper_approach_direction from metadata
        robot_asset = env.scene[self.robot_cfg.name]
        usd_path = robot_asset.cfg.spawn.usd_path
        metadata = utils.read_metadata_from_usd_directory(usd_path)
        self.gripper_approach_direction = tuple(metadata.get("gripper_approach_direction"))

        # Initialize stability counter for consecutive stability checking
        self.stability_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)

        self.object_assets = [env.scene[cfg.name] for cfg in self.object_cfgs]
        self.robot_asset = env.scene[self.robot_cfg.name]
        self.assets_to_check = self.object_assets + [self.robot_asset]
        self.ee_body_idx = self.robot_asset.data.body_names.index(self.ee_body_name)

        # Optional hold check for the "EE grasped" sets. Three clauses: the gripper must be stopped short of its
        # closed command (something is between the fingers), the grasped object must sit near the end effector,
        # and the end effector must still be at the grasp pose it was placed at, expressed in the object's frame
        # (target recorded by reset_end_effector_from_grasp_dataset). The last clause is what tells a hand
        # closed on the peg from a hand closed on the hole, the table edge or thin air next to the peg; the
        # first two alone accept those whenever the object happens not to move.
        self.require_grasp = cfg.params.get("require_grasp", False)
        if self.require_grasp:
            grasp_object_cfg = cfg.params.get("grasp_object_cfg") or self.object_cfgs[0]
            self.grasp_object = env.scene[grasp_object_cfg.name]
            self.grasp_max_object_ee_dist = cfg.params.get("grasp_max_object_ee_dist", 0.2)
            self.grasp_min_open_fraction = cfg.params.get("grasp_min_open_fraction", 0.08)
            # and not (nearly) fully open either: an open hand hovering over the object is not a grasp
            self.grasp_max_open_fraction = cfg.params.get("grasp_max_open_fraction", 0.9)
            self.grasp_event_name = cfg.params.get("grasp_event_name", "reset_end_effector_pose_from_grasp_dataset")
            self.grasp_pos_tol = cfg.params.get("grasp_pos_tol", 0.015)
            self.grasp_ang_tol = cfg.params.get("grasp_ang_tol", 0.26)
            self._grasp_event = None  # resolved on first call, the event manager may not exist yet
            gripper_action_cfg = env.cfg.actions.gripper
            # joints whose open and closed targets coincide (the Dex3 thumb abduction) say nothing about closure
            joint_names = [
                n
                for n in gripper_action_cfg.close_command_expr
                if abs(gripper_action_cfg.open_command_expr.get(n, 0.0) - gripper_action_cfg.close_command_expr[n])
                > 1e-6
            ]
            self.grasp_joint_ids, _ = self.robot_asset.find_joints(joint_names, preserve_order=True)
            self.grasp_closed = torch.tensor(
                [gripper_action_cfg.close_command_expr[n] for n in joint_names], device=env.device
            )
            self.grasp_open = torch.tensor(
                [gripper_action_cfg.open_command_expr.get(n, 0.0) for n in joint_names], device=env.device
            )

        # Optional per-clause bookkeeping (debug_reset_env.py --funnel): counts, over the envs whose hold just
        # ended, how many pass each clause. Collected inside the term because the env is reset right after.
        self.collect_funnel = cfg.params.get("collect_funnel", False)
        self.funnel: dict[str, int] = {}

        # Optional assembly alignment filter
        self.assembly_success_prob = cfg.params.get("assembly_success_prob")
        if self.assembly_success_prob is not None:
            insertive_asset_cfg = cfg.params.get("insertive_asset_cfg")
            receptive_asset_cfg = cfg.params.get("receptive_asset_cfg")
            self.insertive_asset = env.scene[insertive_asset_cfg.name]
            self.receptive_asset = env.scene[receptive_asset_cfg.name]

            insertive_meta = utils.read_asset_metadata(self.insertive_asset)
            receptive_meta = utils.read_asset_metadata(self.receptive_asset)
            self.insertive_asset_offset = Offset(
                pos=tuple(insertive_meta.get("assembled_offset").get("pos")),
                quat=tuple(insertive_meta.get("assembled_offset").get("quat")),
            )
            self.receptive_asset_offset = Offset(
                pos=tuple(receptive_meta.get("assembled_offset").get("pos")),
                quat=tuple(receptive_meta.get("assembled_offset").get("quat")),
            )
            assembly_threshold_scale = cfg.params.get("assembly_threshold_scale", 1.0)
            self.assembly_pos_threshold: float = (
                receptive_meta.get("success_thresholds").get("position") * assembly_threshold_scale
            )
            self.assembly_ori_threshold: float = (
                receptive_meta.get("success_thresholds").get("orientation") * assembly_threshold_scale
            )
            self.require_assembly_success = torch.rand(env.num_envs, device=env.device) < self.assembly_success_prob
            self._pending_reflip = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)

        for asset in self.assets_to_check:
            if asset is self.robot_asset:
                asset_pos = asset.data.body_link_pos_w.torch[:, self.ee_body_idx].clone()
            else:
                asset_pos = asset.data.root_pos_w.torch.clone()
            if not hasattr(asset, "initial_pos") or env_ids is None:
                asset.initial_pos = asset_pos
            else:
                asset.initial_pos[env_ids] = asset_pos[env_ids].clone()

        if env_ids is None:
            self.stability_counter.zero_()
        else:
            self.stability_counter[env_ids] = 0

        if self.assembly_success_prob is not None:
            if env_ids is None:
                self.require_assembly_success = (
                    torch.rand(self._env.num_envs, device=self._env.device) < self.assembly_success_prob
                )
                self._pending_reflip.zero_()
            else:
                reflip_mask = self._pending_reflip[env_ids]
                if reflip_mask.any():
                    reflip_ids = env_ids[reflip_mask]
                    self.require_assembly_success[reflip_ids] = (
                        torch.rand(reflip_ids.shape[0], device=self._env.device) < self.assembly_success_prob
                    )
                self._pending_reflip[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedEnv,
        object_cfgs: list[SceneEntityCfg],
        robot_cfg: SceneEntityCfg,
        ee_body_name: str,
        collision_analyzer_cfgs: list[CollisionAnalyzerCfg],
        max_robot_pos_deviation: float = 0.1,
        max_object_pos_deviation: float = 0.1,
        pos_z_threshold: float = -0.01,
        consecutive_stability_steps: int = 5,
        approach_z_max: float = -0.5,
        insertive_asset_cfg: SceneEntityCfg | None = None,
        receptive_asset_cfg: SceneEntityCfg | None = None,
        assembly_success_prob: float | None = None,
        assembly_threshold_scale: float = 1.0,
        require_grasp: bool = False,
        grasp_object_cfg: SceneEntityCfg | None = None,
        grasp_max_object_ee_dist: float = 0.2,
        grasp_min_open_fraction: float = 0.08,
        grasp_max_open_fraction: float = 0.9,
        grasp_event_name: str = "reset_end_effector_pose_from_grasp_dataset",
        grasp_pos_tol: float = 0.015,
        grasp_ang_tol: float = 0.26,
        collect_funnel: bool = False,
    ) -> torch.Tensor:

        # Check time out
        time_out = env.episode_length_buf >= env.max_episode_length

        # Check for abnormal gripper state (excessive joint velocities)
        abnormal_gripper_state = (
            self.robot_asset.data.joint_vel.torch.abs() > (self.robot_asset.data.joint_vel_limits.torch * 2)
        ).any(dim=1)

        # Check the hand's approach direction against a cone around straight down. ``approach_z_max`` is the cosine
        # of the widest angle allowed: -0.5 is the 60 degree cone, 0.0 admits a horizontal palm, 1.0 removes the test
        ee_quat = self.robot_asset.data.body_link_quat_w.torch[:, self.ee_body_idx]
        gripper_approach_local = torch.tensor(
            self.gripper_approach_direction, device=env.device, dtype=torch.float32
        ).expand(env.num_envs, -1)
        gripper_approach_world = math_utils.quat_apply(ee_quat, gripper_approach_local)
        gripper_orientation_within_range = gripper_approach_world[:, 2] < self.approach_z_max

        # Check if asset velocities are small
        current_step_stable = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        for asset in self.assets_to_check:
            if isinstance(asset, BaseArticulation):
                current_step_stable &= asset.data.joint_vel.torch.abs().sum(dim=1) < 5.0
            elif isinstance(asset, BaseRigidObject):
                current_step_stable &= asset.data.body_lin_vel_w.torch.abs().sum(dim=2).sum(dim=1) < 0.1
                current_step_stable &= asset.data.body_ang_vel_w.torch.abs().sum(dim=2).sum(dim=1) < 1.0
            elif isinstance(asset, BaseRigidObjectCollection):
                current_step_stable &= asset.data.object_lin_vel_w.torch.abs().sum(dim=2).sum(dim=1) < 0.1
                current_step_stable &= asset.data.object_ang_vel_w.torch.abs().sum(dim=2).sum(dim=1) < 1.0

        self.stability_counter = torch.where(
            current_step_stable,
            self.stability_counter + 1,  # Increment counter if stable
            torch.zeros_like(self.stability_counter),  # Reset counter if not stable
        )

        stability_reached = self.stability_counter >= self.consecutive_stability_steps

        # Reset initial positions on first check or after env reset
        excessive_pose_deviation = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        pos_below_threshold = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        deviations: dict[str, torch.Tensor] = {}
        for asset in self.assets_to_check:
            if asset is self.robot_asset:
                asset_pos = asset.data.body_link_pos_w.torch[:, self.ee_body_idx].clone()
            else:
                asset_pos = asset.data.root_pos_w.torch.clone()

            # Skip if position or quaternion is NaN
            pos_is_nan = torch.isnan(asset.data.root_pos_w.torch).any(dim=1)
            quat_is_nan = torch.isnan(asset.data.root_quat_w.torch).any(dim=1)
            skip_check = pos_is_nan | quat_is_nan

            # Asset has excessive pose deviation if position exceeds thresholds
            pos_deviation = (asset_pos - asset.initial_pos).norm(dim=1)
            valid_pos_deviation = torch.where(~skip_check, pos_deviation, torch.zeros_like(pos_deviation))
            deviations[asset.cfg.prim_path.split("/")[-1]] = valid_pos_deviation
            if asset is self.robot_asset:
                excessive_pose_deviation |= valid_pos_deviation > self.max_robot_pos_deviation
            else:
                excessive_pose_deviation |= valid_pos_deviation > self.max_object_pos_deviation

            # Asset is above ground if position is greater than z threshold
            pos_below_threshold |= asset_pos[:, 2] < self.pos_z_threshold

        # The cheap clauses are evaluated every step for every environment. The expensive ones (signed-distance
        # collision queries over the sampled point clouds, the grasp-pose comparison, the assembly check) can only
        # change the outcome on the final step of the hold, and only for environments that passed the cheap clauses,
        # so they run on that subset alone. With collect_funnel they run on every environment at the final step so
        # that the per-clause counts stay comparable.
        cheap_ok = (
            (~abnormal_gripper_state)
            & gripper_orientation_within_range
            & stability_reached
            & (~excessive_pose_deviation)
            & (~pos_below_threshold)
        )
        eval_mask = time_out if self.collect_funnel else (time_out & cheap_ok)
        eval_ids = eval_mask.nonzero(as_tuple=False).squeeze(-1)
        collision_free = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        if eval_ids.numel() > 0:
            collision_free[eval_ids] = torch.all(
                torch.stack([collision_analyzer(env, eval_ids) for collision_analyzer in self.collision_analyzers]),
                dim=0,
            )

        reset_success = cheap_ok & collision_free & time_out

        clauses: dict[str, torch.Tensor] = {}
        if self.collect_funnel:
            clauses = {
                "no_abnormal_gripper": ~abnormal_gripper_state,
                "orientation_down": gripper_orientation_within_range,
                "stable": stability_reached,
                "pose_deviation_ok": ~excessive_pose_deviation,
                "above_z_threshold": ~pos_below_threshold,
                "collision_free": collision_free,
            }
            for name, dev in deviations.items():
                is_robot = name == self.robot_asset.cfg.prim_path.split("/")[-1]
                limit = self.max_robot_pos_deviation if is_robot else self.max_object_pos_deviation
                clauses[f"  dev_ok {name}"] = dev <= limit
                clauses[f"  dev<3cm {name}"] = dev <= 0.03

        if self.require_grasp:
            finger_pos = self.robot_asset.data.joint_pos.torch[:, self.grasp_joint_ids]
            # 0 = at the closed command (nothing between the fingers), 1 = at the open command
            open_fraction = ((finger_pos - self.grasp_closed) / (self.grasp_open - self.grasp_closed)).abs()
            # mean over the closure joints instead of checking individual joints to accommodate
            # multi-finger hands and it does not change the behavior for two-finger gripper
            mean_open_fraction = open_fraction.mean(dim=1)
            fingers_blocked = (mean_open_fraction > self.grasp_min_open_fraction) & (
                mean_open_fraction < self.grasp_max_open_fraction
            )
            ee_pos = self.robot_asset.data.body_link_pos_w.torch[:, self.ee_body_idx]
            object_near_ee = (self.grasp_object.data.root_pos_w.torch - ee_pos).norm(dim=1) < (
                self.grasp_max_object_ee_dist
            )
            reset_success = reset_success & fingers_blocked & object_near_ee
            if self.collect_funnel:
                clauses["fingers_blocked"] = fingers_blocked
                clauses["  fingers open>20%"] = (open_fraction > 0.2).all(dim=1)
                clauses["  fingers open>50%"] = (open_fraction > 0.5).all(dim=1)
                clauses["object_near_ee"] = object_near_ee
            grasp_event = self._resolve_grasp_event(env)
            if grasp_event is not None:
                at_grasp_pose = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
                pos_err = torch.full((env.num_envs,), float("inf"), device=env.device)
                ang_err = torch.full((env.num_envs,), float("inf"), device=env.device)
                if eval_ids.numel() > 0:
                    ee_quat = self.robot_asset.data.body_link_quat_w.torch[eval_ids, self.ee_body_idx]
                    ee_pos_obj, ee_quat_obj = math_utils.subtract_frame_transforms(
                        self.grasp_object.data.root_pos_w.torch[eval_ids],
                        self.grasp_object.data.root_quat_w.torch[eval_ids],
                        ee_pos[eval_ids],
                        ee_quat,
                    )
                    pos_err[eval_ids] = (ee_pos_obj - grasp_event.target_pos_obj[eval_ids]).norm(dim=1)
                    ang_err[eval_ids] = math_utils.quat_error_magnitude(
                        ee_quat_obj, grasp_event.target_quat_obj[eval_ids]
                    )
                    at_grasp_pose[eval_ids] = (pos_err[eval_ids] < self.grasp_pos_tol) & (
                        ang_err[eval_ids] < self.grasp_ang_tol
                    )
                reset_success = reset_success & at_grasp_pose
                if self.collect_funnel:
                    clauses["grasp_pos_ok"] = pos_err < self.grasp_pos_tol
                    clauses["grasp_pos_within_3cm"] = pos_err < 0.03
                    clauses["grasp_ang_ok"] = ang_err < self.grasp_ang_tol

        if self.collect_funnel:
            counts = torch.stack([time_out.sum()] + [(time_out & m).sum() for m in clauses.values()]).tolist()
            for name, n in zip(["hold_end"] + list(clauses), counts):
                self.funnel[name] = self.funnel.get(name, 0) + int(n)

        if self.assembly_success_prob is not None:
            assembly_match = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
            if eval_ids.numel() > 0:
                ins_pos_w, ins_quat_w = self.insertive_asset_offset.apply(self.insertive_asset)
                rec_pos_w, rec_quat_w = self.receptive_asset_offset.apply(self.receptive_asset)
                rel_pos, rel_quat = math_utils.subtract_frame_transforms(
                    rec_pos_w[eval_ids], rec_quat_w[eval_ids], ins_pos_w[eval_ids], ins_quat_w[eval_ids]
                )
                e_x, e_y, _ = math_utils.euler_xyz_from_quat(rel_quat)
                euler_xy_dist = math_utils.wrap_to_pi(e_x).abs() + math_utils.wrap_to_pi(e_y).abs()
                xyz_dist = torch.norm(rel_pos, dim=1)
                assembly_success = (xyz_dist < self.assembly_pos_threshold) & (
                    euler_xy_dist < self.assembly_ori_threshold
                )
                assembly_match[eval_ids] = torch.where(
                    self.require_assembly_success[eval_ids], assembly_success, ~assembly_success
                )
            reset_success = reset_success & assembly_match
            self._pending_reflip |= reset_success
            if self.collect_funnel:
                self.funnel["assembly_match"] = self.funnel.get("assembly_match", 0) + int(
                    (time_out & assembly_match).sum()
                )

        if self.collect_funnel:
            self.funnel["SUCCESS"] = self.funnel.get("SUCCESS", 0) + int(reset_success.sum())

        return reset_success

    def _resolve_grasp_event(self, env: ManagerBasedEnv) -> ManagerTermBase | None:
        """Resolve the grasp placement event term of this environment (looked up once).

        Args:
            env: The environment whose event manager is searched for ``grasp_event_name``.

        Returns:
            The event term that records ``target_pos_obj`` / ``target_quat_obj``, or None if the environment
            has no such term.
        """
        if self._grasp_event is None and self.grasp_event_name is not None:
            try:
                self._grasp_event = env.event_manager.get_term_cfg(self.grasp_event_name).func
            except (ValueError, KeyError, AttributeError):
                self.grasp_event_name = None
            if self._grasp_event is not None and not hasattr(self._grasp_event, "target_pos_obj"):
                self._grasp_event, self.grasp_event_name = None, None
        return self._grasp_event


class check_obb_no_overlap_termination(ManagerTermBase):
    """Termination condition that checks if OBBs of two objects no longer overlap."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._env = env

        self.insertive_object_cfg = cfg.params.get("insertive_object_cfg")
        self.insertive_object = env.scene[self.insertive_object_cfg.name]

        self.enable_visualization = cfg.params.get("enable_visualization", False)
        enable_extension("isaacsim.core.experimental.utils")
        import isaacsim.core.experimental.utils.bounds as bounds_utils

        self._bounds_utils = bounds_utils

        # Initialize OBB computation cache and compute OBBs once
        self._bbox_cache = bounds_utils.create_bbox_cache()
        self._compute_object_obbs()

        # Initialize placeholders for initial poses (will be set in reset)
        self._insertive_initial_pos = None
        self._insertive_initial_quat = None

        # Store debug draw interface if visualization is enabled
        if self.enable_visualization:
            enable_extension("isaacsim.util.debug_draw")
            import isaacsim.util.debug_draw._debug_draw as omni_debug_draw

            self._omni_debug_draw = omni_debug_draw
        else:
            self._omni_debug_draw = None

    def _compute_object_obbs(self):
        """Compute OBB for insertive object and convert to body frame."""
        # Get prim path (use env 0 as template)
        insertive_prim_path = utils.RigidObjectHasher.resolve_prim_paths(
            self._env.num_envs, self.insertive_object.cfg.prim_path
        )[0]

        # Compute OBB in world frame using Isaac Sim's built-in functions
        insertive_centroid_world, insertive_axes_world, insertive_half_extents = self._bounds_utils.compute_obb(
            insertive_prim_path, bbox_cache=self._bbox_cache
        )

        # Get current world pose of object (env 0) to convert OBB to body frame
        insertive_pos_world = self.insertive_object.data.root_pos_w.torch[0]  # (3,)
        insertive_quat_world = self.insertive_object.data.root_quat_w.torch[0]  # (4,)

        device = self._env.device

        # Convert world frame OBB data to torch tensors
        insertive_centroid_world_tensor = torch.tensor(insertive_centroid_world, device=device, dtype=torch.float32)
        insertive_axes_world_tensor = torch.tensor(insertive_axes_world, device=device, dtype=torch.float32)

        # Convert centroid from world frame to body frame
        insertive_centroid_body = math_utils.quat_apply_inverse(
            insertive_quat_world, insertive_centroid_world_tensor - insertive_pos_world
        )

        # Convert axes from world frame to body frame
        insertive_rot_matrix_world = math_utils.matrix_from_quat(insertive_quat_world.unsqueeze(0))[0]  # (3, 3)

        # Transform axes: R_world_to_body @ world_axes = R_world^T @ world_axes
        # Note: Isaac Sim's compute_obb returns axes as column vectors, so we need to transpose
        insertive_axes_body = torch.matmul(insertive_rot_matrix_world.T, insertive_axes_world_tensor.T).T

        # Cache OBB data in body frame as torch tensors on device for fast access
        self._insertive_obb_centroid = insertive_centroid_body
        self._insertive_obb_axes = insertive_axes_body
        self._insertive_obb_half_extents = torch.tensor(insertive_half_extents, device=device, dtype=torch.float32)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Store initial pose of insertive object when environments are reset."""
        super().reset(env_ids)

        insertive_pos = self.insertive_object.data.root_pos_w.torch.clone()
        insertive_quat = self.insertive_object.data.root_quat_w.torch.clone()

        if self._insertive_initial_pos is None or self._insertive_initial_quat is None or env_ids is None:
            # First time initialization or reset all environments
            self._insertive_initial_pos = insertive_pos
            self._insertive_initial_quat = insertive_quat
        else:
            # Update only the reset environments
            self._insertive_initial_pos[env_ids] = insertive_pos[env_ids]
            self._insertive_initial_quat[env_ids] = insertive_quat[env_ids]

    @torch.no_grad()
    def _compute_obb_corners_batch(
        self, centroids: torch.Tensor, axes: torch.Tensor, half_extents: torch.Tensor
    ) -> torch.Tensor:
        """Compute batched OBB corners on-device in Isaac Sim's corner order.

        Args:
            centroids: Centers [m], shape (num_envs, 3).
            axes: Orientation axes, shape (num_envs, 3, 3), with one axis per row.
            half_extents: Half lengths [m], shape (3,).

        Returns:
            Detached float32 corner positions [m], shape (num_envs, 8, 3).
        """
        num_envs = centroids.shape[0]
        device = centroids.device

        # Keep corner offsets on the input device.
        signs = torch.tensor(
            [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]],
            device=device,
            dtype=centroids.dtype,
        )

        # Compute offsets for all environments at once.
        offsets = (signs * half_extents).expand(num_envs, -1, -1)
        # Match Isaac Sim's ordering and row-vector axes.
        corners_tensor = centroids.unsqueeze(1) + torch.bmm(offsets, axes)  # (num_envs, 8, 3)

        # Preserve the existing float32 output contract without a host transfer.
        return corners_tensor.to(dtype=torch.float32)  # (num_envs, 8, 3)

    def _visualize_bounding_boxes(self, env: ManagerBasedEnv):
        """Visualize oriented bounding boxes for initial and current insertive object positions using wireframe edges."""
        # Clear previous debug lines
        draw_interface = self._omni_debug_draw.acquire_debug_draw_interface()
        draw_interface.clear_lines()

        # Get current world poses of insertive object for all environments
        insertive_pos = self.insertive_object.data.root_pos_w.torch  # (num_envs, 3)
        insertive_quat = self.insertive_object.data.root_quat_w.torch  # (num_envs, 4)

        # Transform current insertive object OBB centroid from body frame to world coordinates for all environments
        insertive_obb_centroid_body = self._insertive_obb_centroid
        insertive_current_world_centroids = insertive_pos + math_utils.quat_apply(
            insertive_quat, insertive_obb_centroid_body.unsqueeze(0).expand(env.num_envs, -1)
        )  # (num_envs, 3)

        # Transform current insertive object OBB orientation from body frame to world coordinates for all environments
        insertive_current_rot_matrices = math_utils.matrix_from_quat(insertive_quat)  # (num_envs, 3, 3)
        insertive_obb_axes_body = self._insertive_obb_axes
        insertive_current_world_axes = torch.bmm(
            insertive_current_rot_matrices,
            insertive_obb_axes_body.unsqueeze(0).expand(env.num_envs, -1, -1).transpose(1, 2),
        ).transpose(
            1, 2
        )  # (num_envs, 3, 3)

        # Compute OBB corners for current position visualization in one batch.
        insertive_current_corners = self._compute_obb_corners_batch(
            insertive_current_world_centroids, insertive_current_world_axes, self._insertive_obb_half_extents
        )  # (num_envs, 8, 3)

        # Transform initial insertive object OBB centroid from body frame to world coordinates for all environments
        insertive_initial_world_centroids = self._insertive_initial_pos + math_utils.quat_apply(
            self._insertive_initial_quat, insertive_obb_centroid_body.unsqueeze(0).expand(env.num_envs, -1)
        )  # (num_envs, 3)

        # Transform initial insertive object OBB orientation from body frame to world coordinates
        insertive_initial_rot_matrices = math_utils.matrix_from_quat(self._insertive_initial_quat)  # (num_envs, 3, 3)
        insertive_initial_world_axes = torch.bmm(
            insertive_initial_rot_matrices,
            insertive_obb_axes_body.unsqueeze(0).expand(env.num_envs, -1, -1).transpose(1, 2),
        ).transpose(
            1, 2
        )  # (num_envs, 3, 3)

        # Compute OBB corners for initial position visualization in one batch.
        insertive_initial_corners = self._compute_obb_corners_batch(
            insertive_initial_world_centroids, insertive_initial_world_axes, self._insertive_obb_half_extents
        )  # (num_envs, 8, 3)

        # Draw wireframe boxes for the environment batch.
        # Draw current insertive object bounding box edges (blue)
        self._draw_obb_wireframe(
            insertive_current_corners,  # (num_envs, 8, 3)
            color=(0.0, 0.5, 1.0, 1.0),  # Bright blue
            line_width=4.0,
            draw_interface=draw_interface,
        )

        # Draw initial insertive object bounding box edges (red)
        self._draw_obb_wireframe(
            insertive_initial_corners,  # (num_envs, 8, 3)
            color=(1.0, 0.2, 0.0, 1.0),  # Bright red
            line_width=4.0,
            draw_interface=draw_interface,
        )

    def _draw_obb_wireframe(
        self, corners: torch.Tensor, color: tuple = (1.0, 1.0, 1.0, 1.0), line_width: float = 2.0, draw_interface=None
    ):
        """
        Draw wireframe edges of an oriented bounding box.

        Args:
            corners: Corner positions [m], shape (num_envs, 8, 3) or (8, 3).
            color: RGBA color tuple for the lines
            line_width: Width of the lines
            draw_interface: Debug draw interface (optional, will acquire if not provided)
        """
        # Define the edges of a cube by connecting corner indices
        # Corners are ordered as: [0-3] bottom face, [4-7] top face
        edge_indices = [
            # Bottom face edges
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            # Top face edges
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            # Vertical edges connecting bottom to top
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
            # Diagonal X on top face (4-5-6-7)
            (4, 6),
            (5, 7),
            # Diagonal X on bottom face (0-1-2-3)
            (0, 2),
            (1, 3),
            # Diagonal X on front face (0-1-4-5)
            (0, 5),
            (1, 4),
            # Diagonal X on back face (2-3-6-7)
            (2, 7),
            (3, 6),
            # Diagonal X on left face (0-3-4-7)
            (0, 7),
            (3, 4),
            # Diagonal X on right face (1-2-5-6)
            (1, 6),
            (2, 5),
        ]

        # Create all line segments on-device, then transfer once for the debug-draw API.
        edges = torch.tensor(edge_indices, device=corners.device)
        lines = corners.reshape(-1, 8, 3)[:, edges].reshape(-1, 2, 3).detach().cpu()
        line_starts = lines[:, 0].tolist()
        line_ends = lines[:, 1].tolist()

        # Use provided interface or acquire new one
        if draw_interface is None:
            draw_interface = self._omni_debug_draw.acquire_debug_draw_interface()

        colors = [list(color)] * len(line_starts)
        line_thicknesses = [line_width] * len(line_starts)

        # Draw all edges at once
        draw_interface.draw_lines(line_starts, line_ends, colors, line_thicknesses)

    def __call__(
        self,
        env: ManagerBasedEnv,
        insertive_object_cfg: SceneEntityCfg,
        enable_visualization: bool = False,
    ) -> torch.Tensor:
        """Check if OBB overlap condition is violated between initial and current insertive object positions."""

        # Get current world poses of insertive object for all environments
        insertive_pos = self.insertive_object.data.root_pos_w.torch  # (num_envs, 3)
        insertive_quat = self.insertive_object.data.root_quat_w.torch  # (num_envs, 4)

        # Transform current insertive object centroid from body frame to world coordinates for all environments
        insertive_obb_centroid_body = self._insertive_obb_centroid
        insertive_current_world_centroids = insertive_pos + math_utils.quat_apply(
            insertive_quat, insertive_obb_centroid_body.unsqueeze(0).expand(env.num_envs, -1)
        )  # (num_envs, 3)

        # Transform initial insertive object centroid from body frame to world coordinates for all environments
        insertive_initial_world_centroids = self._insertive_initial_pos + math_utils.quat_apply(
            self._insertive_initial_quat, insertive_obb_centroid_body.unsqueeze(0).expand(env.num_envs, -1)
        )  # (num_envs, 3)

        # Transform OBB axes to world coordinates
        insertive_current_rot_matrices = math_utils.matrix_from_quat(insertive_quat)  # (num_envs, 3, 3)
        insertive_initial_rot_matrices = math_utils.matrix_from_quat(self._insertive_initial_quat)  # (num_envs, 3, 3)

        insertive_obb_axes_body = self._insertive_obb_axes

        # Transform axes from body frame to world frame: R @ body_axes for all environments
        # Since axes are stored as row vectors, we need to handle the transpose properly
        insertive_current_world_axes = torch.bmm(
            insertive_current_rot_matrices,
            insertive_obb_axes_body.unsqueeze(0).expand(env.num_envs, -1, -1).transpose(1, 2),
        ).transpose(
            1, 2
        )  # (num_envs, 3, 3)

        insertive_initial_world_axes = torch.bmm(
            insertive_initial_rot_matrices,
            insertive_obb_axes_body.unsqueeze(0).expand(env.num_envs, -1, -1).transpose(1, 2),
        ).transpose(
            1, 2
        )  # (num_envs, 3, 3)

        # Check OBB overlap between initial and current insertive object positions for all environments
        obb_overlap = _check_obb_overlap(
            insertive_current_world_centroids,
            insertive_current_world_axes,
            self._insertive_obb_half_extents,
            insertive_initial_world_centroids,
            insertive_initial_world_axes,
            self._insertive_obb_half_extents,
        )

        # Visualize bounding boxes if enabled
        if self.enable_visualization:
            self._visualize_bounding_boxes(env)

        return ~obb_overlap


class grasp_placement_failed(ManagerTermBase):
    """Terminate at once when the IK placement of ``reset_end_effector_from_grasp_dataset`` missed its target.

    A reset whose end effector never reached the sampled grasp pose cannot become a valid "EE grasped" state, so
    holding it for the full episode only costs time. Envs without that event term never terminate here.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedEnv):
        """Constructor."""
        super().__init__(cfg, env)
        self.event_name = cfg.params.get("event_name", "reset_end_effector_pose_from_grasp_dataset")
        self.pos_tol = cfg.params.get("pos_tol", 0.01)
        self.ang_tol = cfg.params.get("ang_tol", 0.17)
        self._event: ManagerTermBase | None = None
        self._checked = False

    def __call__(
        self,
        env: ManagerBasedEnv,
        event_name: str = "reset_end_effector_pose_from_grasp_dataset",
        pos_tol: float = 0.01,
        ang_tol: float = 0.17,
    ) -> torch.Tensor:
        """Flag the environments whose last IK placement ended outside the tolerances.

        Args:
            env: The environment.
            event_name: Name of the grasp placement event term whose IK errors are read.
            pos_tol: Position tolerance [m] on the IK error.
            ang_tol: Orientation tolerance [rad] on the IK error.

        Returns:
            Boolean tensor of shape (num_envs,), True where the placement missed its target.
        """
        if not self._checked:
            self._checked = True
            try:
                term = env.event_manager.get_term_cfg(self.event_name).func
                self._event = term if hasattr(term, "ik_pos_error") else None
            except (ValueError, KeyError, AttributeError):
                self._event = None
        if self._event is None:
            return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        return (self._event.ik_pos_error > self.pos_tol) | (self._event.ik_ang_error > self.ang_tol)


class reset_hold_lost(ManagerTermBase):
    """Terminate a reset-state hold as soon as it can no longer pass ``check_reset_state_success``.

    The success term compares every tracked body with the position it had right after the reset events and
    rejects, at the end of the hold, any hold in which the end effector moved more than its allowed deviation,
    an object moved more than the family's allowed deviation, or a body dropped below the height threshold.
    The height condition never recovers within a hold. A deviation can in principle shrink again (an object
    nudged and settling back), so the deviation limits are applied with a margin, ``deviation_margin`` times the
    success term's limit, which still catches the common case of a hand snapped away from its placement on the
    first physics step. It reads the references and thresholds from the success term (``success_term_name``) and
    is a no-op in environments without one.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedEnv):
        """Constructor."""
        super().__init__(cfg, env)
        self.success_term_name = cfg.params.get("success_term_name", "success")
        self.deviation_margin = float(cfg.params.get("deviation_margin", 2.0))
        self._term: ManagerTermBase | None = None
        self._checked = False

    def __call__(
        self, env: ManagerBasedEnv, success_term_name: str = "success", deviation_margin: float = 2.0
    ) -> torch.Tensor:
        """Flag the environments whose hold can no longer pass the success term.

        Args:
            env: The environment.
            success_term_name: Name of the ``check_reset_state_success`` term whose references and limits are read.
            deviation_margin: Multiplier applied to the success term's deviation limits before comparing.

        Returns:
            Boolean tensor of shape (num_envs,), True where a tracked body dropped below the height threshold or
            moved beyond the margin-scaled deviation limit.
        """
        if not self._checked:
            self._checked = True
            try:
                term = env.termination_manager.get_term_cfg(self.success_term_name).func
                self._term = term if hasattr(term, "assets_to_check") else None
            except (ValueError, KeyError, AttributeError):
                self._term = None
        lost = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        term = self._term
        if term is None:
            return lost
        for asset in term.assets_to_check:
            if not hasattr(asset, "initial_pos"):
                return lost  # before the first reset
            if asset is term.robot_asset:
                pos = asset.data.body_link_pos_w.torch[:, term.ee_body_idx]
                limit = term.max_robot_pos_deviation
            else:
                pos = asset.data.root_pos_w.torch
                limit = term.max_object_pos_deviation
            valid = ~torch.isnan(pos).any(dim=1)
            deviation = (pos - asset.initial_pos).norm(dim=1)
            lost |= valid & (deviation > self.deviation_margin * limit)
            lost |= valid & (pos[:, 2] < term.pos_z_threshold)
        return lost


def consecutive_success_state(env: ManagerBasedRLEnv, num_consecutive_successes: int = 10):
    # Get the progress context to access assets and offsets
    context_term = env.reward_manager.get_term_cfg("progress_context").func  # type: ignore
    continuous_success_counter = getattr(context_term, "continuous_success_counter")

    return continuous_success_counter >= num_consecutive_successes


def consecutive_success_state_with_min_length(
    env: ManagerBasedRLEnv, num_consecutive_successes: int = 10, min_episode_length: int = 0
):
    """Like consecutive_success_state but rejects episodes shorter than min_episode_length.

    Episodes that start already assembled will reach num_consecutive_successes quickly,
    but won't be marked as success until min_episode_length steps have passed.
    Combined with a separate early termination, these episodes get terminated as failures.
    """
    context_term = env.reward_manager.get_term_cfg("progress_context").func  # type: ignore
    continuous_success_counter = getattr(context_term, "continuous_success_counter")
    success = continuous_success_counter >= num_consecutive_successes
    if min_episode_length > 0:
        success = success & (env.episode_length_buf >= min_episode_length)
    return success


def early_success_termination(env: ManagerBasedRLEnv, num_consecutive_successes: int = 5, min_episode_length: int = 10):
    """Terminates episodes that achieve success before min_episode_length steps.

    Paired with consecutive_success_state_with_min_length as the 'success' term:
    that term gates success until min_episode_length, while this term terminates
    the episode early (as a non-success failure) to avoid wasting sim time.
    """
    context_term = env.reward_manager.get_term_cfg("progress_context").func  # type: ignore
    continuous_success_counter = getattr(context_term, "continuous_success_counter")
    is_successful = continuous_success_counter >= num_consecutive_successes
    is_too_short = env.episode_length_buf < min_episode_length
    return is_successful & is_too_short


def corrupted_camera_detected(
    env: ManagerBasedRLEnv, camera_names: list[str], std_threshold: float = 10.0
) -> torch.Tensor:
    """
    Detect corrupted camera images by checking if standard deviation is below threshold.

    Corrupted cameras typically show uniform gray/black images with very low variance.
    This function checks all specified cameras and returns True for environments where
    any camera shows corruption (std < threshold).

    Args:
        env: The environment instance.
        camera_names: List of camera sensor names to check (e.g., ["front_camera", "wrist_camera"]).
        std_threshold: Standard deviation threshold below which image is considered corrupted.
                      Default 10.0 is conservative - normal images have std > 20.

    Returns:
        Boolean tensor of shape (num_envs,) indicating which environments have corrupted cameras.
    """
    num_envs = env.num_envs
    device = env.device

    # Initialize as no corruption
    is_corrupted = torch.zeros(num_envs, dtype=torch.bool, device=device)

    # Check each camera
    for camera_name in camera_names:
        # Get camera sensor from scene
        camera = env.scene[camera_name]

        # Get RGB data: shape (num_envs, height, width, 3)
        rgb_data = camera.data.output["rgb"]

        # Compute standard deviation across spatial and channel dimensions
        # Reshape to (num_envs, -1) to compute std per environment
        rgb_flat = rgb_data.reshape(num_envs, -1).float()
        std_per_env = torch.std(rgb_flat, dim=1)

        # Mark as corrupted if std is below threshold
        is_corrupted |= std_per_env < std_threshold

    return is_corrupted
