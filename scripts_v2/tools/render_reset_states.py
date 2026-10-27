# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render recorded reset states to images, headless.

One environment per state: the first ``--num_states`` states of each reset set are written into the
environments exactly as recorded (no random sampling), a camera per environment renders them, and the
result is saved as one image grid per set plus the individual tiles. Works without a display.

    python scripts_v2/tools/render_reset_states.py --task OmniReset-G1dDex1-RelCartesianOSC-State-Play-v0 \\
        --dataset_dir source/uwlab_assets/data/Datasets/OmniReset_g1d --out_dir /tmp/reset_renders
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

ALL_RESET_TYPES = [
    "ObjectAnywhereEEAnywhere",
    "ObjectRestingEEGrasped",
    "ObjectAnywhereEEGrasped",
    "ObjectPartiallyAssembledEEGrasped",
]

parser = argparse.ArgumentParser(description="Render recorded reset states to images.")
parser.add_argument("--task", type=str, required=True, help="Task whose events contain `reset_from_reset_states`.")
parser.add_argument("--dataset_dir", type=str, required=True, help="Dataset directory (contains Resets/<Pair>/).")
parser.add_argument("--reset_types", nargs="+", default=ALL_RESET_TYPES, help="Reset sets to render.")
parser.add_argument("--num_states", type=int, default=8, help="States per set (= number of environments).")
parser.add_argument("--out_dir", type=str, required=True, help="Where the images are written.")
parser.add_argument("--width", type=int, default=720)
parser.add_argument("--height", type=int, default=540)
parser.add_argument("--eye", type=float, nargs=3, default=(0.95, -0.85, 1.3), help="Camera position (env frame).")
parser.add_argument("--target", type=float, nargs=3, default=(0.37, -0.14, 0.9), help="Camera look-at (env frame).")
parser.add_argument("--focal_length", type=float, default=24.0, help="Pinhole focal length in mm (default 24).")
parser.add_argument("--settle_steps", type=int, default=2, help="Physics steps (targets held) before capturing.")
parser.add_argument(
    "--gripper",
    choices=["recorded", "close", "open"],
    default="recorded",
    help="Gripper command during the settle: the task's close / open command through its binary gripper action "
    "(the recorder commands close while holding the grasped sets), or the recorded finger positions held as targets. "
    "The arm is always held by the task's own controller at a zero action, as during recording.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

# the --enable_cameras CLI flag is gone in Isaac Lab 3.0; the intent is passed to the launcher directly
app_launcher = AppLauncher(args_cli, enable_cameras=True)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import inspect
import json
import numpy as np
import os
import torch
from PIL import Image
from typing import cast

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase
from isaaclab.sensors import CameraCfg

from uwlab_tasks.manager_based.manipulation.omnireset.mdp.events import sample_from_nested_dict
from uwlab_tasks.utils.hydra import hydra_task_compose


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_states
    env_cfg.scene.env_spacing = 6.0  # keep neighbouring environments out of the frame
    env_cfg.seed = 0
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    term = env_cfg.events.reset_from_reset_states
    term.params["dataset_dir"] = args_cli.dataset_dir
    term.params["reset_types"] = list(args_cli.reset_types)
    term.params["probs"] = [1.0] * len(args_cli.reset_types)

    env_cfg.scene.camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.0,
        width=args_cli.width,
        height=args_cli.height,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=args_cli.focal_length, clipping_range=(0.05, 30.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), convention="world"),
    )

    env = cast(ManagerBasedRLEnv, gym.make(args_cli.task, cfg=env_cfg)).unwrapped

    # class-based event terms are created lazily; make sure the reset manager exists (same as visualize_reset_states)
    for mode_cfgs in env.event_manager._mode_term_cfgs.values():
        for tc in mode_cfgs:
            if inspect.isclass(tc.func) and issubclass(tc.func, ManagerTermBase):
                tc.func = tc.func(cfg=tc, env=env)
    env.reset()

    device = env.device
    origins = env.scene.env_origins
    camera = env.scene["camera"]
    eye = origins + torch.tensor(args_cli.eye, device=device)
    target = origins + torch.tensor(args_cli.target, device=device)
    camera.set_world_poses_from_view(eye, target)

    manager = env.event_manager.get_term_cfg("reset_from_reset_states").func
    robot = env.scene["robot"]
    os.makedirs(args_cli.out_dir, exist_ok=True)

    # The hold replicates what the recorder does while it waits for its success check: the task's own action
    # terms run with a zero arm action, so the operational-space controller holds the end effector where the
    # reset put it (joint position targets would do nothing, the arm actuators have no joint stiffness), and the
    # binary gripper action drives the fingers to the open or close command. With --gripper recorded the finger
    # targets are overridden with the recorded positions instead.
    gripper_cfg = env.cfg.actions.gripper
    gripper_joint_ids = robot.find_joints(list(gripper_cfg.close_command_expr.keys()))[0]
    actions = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=device)
    actions[:, -1] = {"close": -1.0, "open": 1.0}.get(args_cli.gripper, 0.0)

    def hold_and_render(num_steps: int):
        hold_pos = robot.data.joint_pos[:, gripper_joint_ids].clone()
        env.action_manager.reset()
        for i in range(num_steps):
            if i % env.cfg.decimation == 0:
                env.action_manager.process_action(actions)
            env.action_manager.apply_action()
            if args_cli.gripper == "recorded":
                robot.set_joint_position_target(hold_pos, joint_ids=gripper_joint_ids)
            env.scene.write_data_to_sim()
            env.sim.step(render=True)
            # asset data buffers refresh lazily on the scene timestamp; without this the poses read
            # below would still be the values written at reset, whatever physics did in between
            env.scene.update(dt=env.physics_dt)
        for _ in range(6):
            env.sim.render()
        camera.update(dt=0.0, force_recompute=True)

    hold_and_render(1)  # warm up shaders / materials before the first capture
    poses: dict = {}
    for set_idx, reset_type in enumerate(args_cli.reset_types):
        available = int(manager.num_states[set_idx])
        count = min(available, args_cli.num_states)
        env_ids = torch.arange(count, device=device)
        states = sample_from_nested_dict(manager.datasets[set_idx], env_ids)
        manager._reset_to(states["initial_state"], env_ids=env_ids, is_relative=True)
        hold_and_render(args_cli.settle_steps)

        rgb = camera.data.output["rgb"][:count, :, :, :3].cpu().numpy().astype(np.uint8)
        cols = min(4, count)
        rows = (count + cols - 1) // cols
        grid = np.zeros((rows * args_cli.height, cols * args_cli.width, 3), dtype=np.uint8)
        for i in range(count):
            r, c = divmod(i, cols)
            grid[r * args_cli.height : (r + 1) * args_cli.height, c * args_cli.width : (c + 1) * args_cli.width] = rgb[i]
            Image.fromarray(rgb[i]).save(os.path.join(args_cli.out_dir, f"{reset_type}_{i:02d}.png"))
        Image.fromarray(grid).save(os.path.join(args_cli.out_dir, f"{reset_type}_grid.png"))
        ins = env.scene["insertive_object"].data.root_pos_w[:count] - origins[:count]
        rec = env.scene["receptive_object"].data.root_pos_w[:count] - origins[:count]
        ee_body = robot.find_bodies(env.cfg.actions.arm.body_name)[0][0]
        ee = robot.data.body_pos_w[:count, ee_body] - origins[:count]
        poses[reset_type] = {
            "settle_steps": args_cli.settle_steps,
            "gripper": args_cli.gripper,
            "insertive_pos": [[round(v, 4) for v in row] for row in ins.tolist()],
            "receptive_pos": [[round(v, 4) for v in row] for row in rec.tolist()],
            "ee_pos": [[round(v, 4) for v in row] for row in ee.tolist()],
        }
        with open(os.path.join(args_cli.out_dir, "poses.json"), "w") as f:
            json.dump(poses, f, indent=1)
        print(
            f"[render] {reset_type}: {count}/{available} states -> {reset_type}_grid.png | insertive z "
            f"{[round(v, 3) for v in ins[:, 2].tolist()]} | receptive z {[round(v, 3) for v in rec[:, 2].tolist()]}"
        )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
