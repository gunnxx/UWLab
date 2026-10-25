# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Step a grasp-sampling environment with the gripper closing and print what the gripper and object do.

Used to debug a new gripper (e.g. the Dex1-1) when ``record_grasps.py`` never reports a successful grasp.

Example::

    python scripts_v2/tools/debug_grasp_env.py --task OmniReset-Dex1-GraspSampling-v0 --num_envs 16 --steps 300 \
        --visualizer none env.scene.object=peg
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Probe a grasp-sampling environment.")
parser.add_argument("--task", type=str, default="OmniReset-Dex1-GraspSampling-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--print_every", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from typing import cast  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

import uwlab_tasks  # noqa: E402, F401
from uwlab_tasks.utils.hydra import hydra_task_compose  # noqa: E402


def _fmt(t: torch.Tensor, n: int = 4, d: int = 4):
    return t[:n].detach().cpu().numpy().round(d).tolist()


def _safe(fn, default="n/a"):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return f"{default} ({type(e).__name__}: {e})"


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = None
    env = cast(ManagerBasedRLEnv, gym.make(args_cli.task, cfg=env_cfg)).unwrapped
    robot = env.scene["robot"]
    obj = env.scene["object"]

    print("[probe] robot joints :", robot.joint_names)
    print("[probe] robot bodies :", robot.body_names)
    print("[probe] joint limits :", _safe(lambda: _fmt(robot.data.joint_pos_limits.torch[0], n=8)))
    print("[probe] actuators    :", {k: type(v).__name__ for k, v in robot.actuators.items()})
    print("[probe] act stiffness:", _safe(lambda: {k: _fmt(v.stiffness[0], n=8) for k, v in robot.actuators.items()}))
    print("[probe] act effort   :", _safe(lambda: {k: _fmt(v.effort_limit[0], n=8) for k, v in robot.actuators.items()}))
    print("[probe] action terms :", {k: v.action_dim for k, v in env.action_manager._terms.items()})
    print("[probe] event terms  :", env.event_manager.active_terms)
    print("[probe] termination  :", env.termination_manager.active_terms)

    env.reset()
    actions = -torch.ones(env.action_space.shape, device=env.device, dtype=torch.float32)
    for t in range(args_cli.steps):
        _, _, terminated, truncated, _ = env.step(actions)
        if t % args_cli.print_every == 0 or bool(terminated.any()) or bool(truncated.any()):
            q = robot.data.joint_pos.torch
            tgt = _safe(lambda: _fmt(robot.data.joint_pos_target.torch[0], n=8))
            tau = _safe(lambda: _fmt(robot.data.applied_torque.torch[0], n=8))
            p_obj = obj.data.root_pos_w.torch
            p_g = robot.data.root_pos_w.torch
            dist = (p_obj - p_g).norm(dim=1)
            ok = _safe(lambda: env.recorder_manager.exported_successful_episode_count)
            print(
                f"[probe] t={t:4d} q[env0]={_fmt(q[0], n=8)} target[env0]={tgt} tau[env0]={tau}\n"
                f"[probe]        obj z={_fmt(p_obj[:, 2], d=3)} grip z={_fmt(p_g[:, 2], d=3)} |obj-grip|={_fmt(dist, d=3)}\n"
                f"[probe]        terminated={int(terminated.sum())} truncated={int(truncated.sum())}"
                f" successes_exported={ok} gravity={_safe(lambda: env.sim.get_physics_context().get_gravity())}"
            )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
