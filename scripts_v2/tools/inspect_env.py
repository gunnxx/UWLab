# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open a registered environment, hold it still, and reset it on a key press.

Loads any gym-registered task (UR5e + Robotiq 2F-85, G1-D + Dex1, reset-state or RL variants), steps it with a
zero arm action (the OSC holds the current pose) and a constant gripper command, and listens to the Kit window's
keyboard. This is the quickest way to look at a scene after a config change: the objects, the table, the reset
distribution (every ``R`` draws a new reset state), the collision behaviour.

Keys (need the Kit window, i.e. ``--viz kit``):

    R  reset all environments        P  pause / resume stepping (the viewport keeps rendering)
    N  step once while paused        H  print this help

Examples::

    # UR5e + Robotiq, 4 envs, reset distribution of the RL task
    python scripts_v2/tools/inspect_env.py --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
        --num_envs 4 --viz kit env.scene.insertive_object=peg env.scene.receptive_object=peghole

    # G1-D + Dex1, one of the reset-state sampling envs, never time out (R is the only reset)
    python scripts_v2/tools/inspect_env.py --task OmniReset-G1dDex1-ObjectAnywhereEEAnywhere-v0 \
        --num_envs 4 --viz kit --no_timeout

    # headless smoke check: 100 steps, auto-reset every 2 s, then exit
    python scripts_v2/tools/inspect_env.py --task OmniReset-G1dDex1-RelCartesianOSC-State-Play-v0 \
        --num_envs 2 --viz none --max_steps 100 --auto_reset_s 2
"""

from __future__ import annotations

import argparse
import functools
import time

from isaaclab.app import AppLauncher

# stdout is often a file or a pipe (run_local.sh, logs); keep the status lines unbuffered.
print = functools.partial(print, flush=True)  # noqa: A001

parser = argparse.ArgumentParser(description="Inspect a registered environment and reset it from the keyboard.")
parser.add_argument("--task", type=str, required=True, help="Registered task id (see scripts/environments/list_envs.py).")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
parser.add_argument(
    "--gripper",
    choices=["keep", "open", "closed"],
    default="keep",
    help="Gripper command while holding: keep its state after the reset (default), force open, or force closed.",
)
parser.add_argument("--no_timeout", action="store_true", help="Disable the time_out termination so only R resets.")
parser.add_argument("--episode_length_s", type=float, default=None, help="Override the episode length (seconds).")
parser.add_argument("--auto_reset_s", type=float, default=0.0, help="Also reset every N seconds of wall clock (0 = off).")
parser.add_argument("--max_steps", type=int, default=0, help="Exit after this many env steps (0 = run until closed).")
parser.add_argument("--print_every", type=int, default=50, help="Print termination counts every N steps (0 = off).")
parser.add_argument(
    "--render_substeps",
    action="store_true",
    help="Keep the task's render interval (renders every physics substep). Default: one render per env step.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import contextlib  # noqa: E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from typing import cast  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

import uwlab_tasks  # noqa: E402, F401
from uwlab_tasks.utils.hydra import hydra_task_compose  # noqa: E402

HELP = "keys: [R] reset all envs  [P] pause/resume  [N] single step while paused  [H] help"


class KeyboardControl:
    """Flags toggled from the Kit window's keyboard; falls back to no-op without a window (``--viz none``)."""

    def __init__(self):
        self.reset_requested = False
        self.paused = False
        self.step_once = False
        self._sub = None
        try:
            import carb
            import omni.appwindow

            self._carb = carb
            self._input = carb.input.acquire_input_interface()
            self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
            self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_event)
            print(f"[inspect] keyboard attached. {HELP}")
        except Exception as exc:  # no app window (headless) or no carb input
            print(f"[inspect] keyboard unavailable ({exc!r}); use --viz kit for key presses.")

    def _on_event(self, event, *args, **kwargs):
        if event.type != self._carb.input.KeyboardEventType.KEY_PRESS:
            return True
        key = event.input.name
        if key == "R":
            self.reset_requested = True
        elif key == "P":
            self.paused = not self.paused
            print(f"[inspect] {'paused' if self.paused else 'resumed'}")
        elif key == "N":
            self.step_once = True
        elif key == "H":
            print(f"[inspect] {HELP}")
        return True

    def close(self):
        if self._sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
            self._sub = None


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=hydra_args)
def main(env_cfg, agent_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = None  # a different reset draw every time
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = args_cli.episode_length_s
    # The OmniReset configs render every physics substep (render_interval 1, decimation 12): 12 RTX frames per
    # env step. For looking at a scene one frame per step is enough and ~10x faster with a window.
    if not args_cli.render_substeps and env_cfg.sim.render_interval < env_cfg.decimation:
        env_cfg.sim.render_interval = env_cfg.decimation
    if args_cli.no_timeout and getattr(env_cfg, "terminations", None) is not None:
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        else:
            print("[inspect] --no_timeout: this task has no 'time_out' termination term")

    env = cast(ManagerBasedRLEnv, gym.make(args_cli.task, cfg=env_cfg)).unwrapped
    env.reset()

    print(f"[inspect] task {args_cli.task}: {env.num_envs} envs, episode {env_cfg.episode_length_s} s,"
          f" action dim {env.action_space.shape[-1]}")
    print(f"[inspect] terminations: {env.termination_manager.active_terms}")
    if hasattr(env, "reward_manager"):
        print(f"[inspect] rewards: {env.reward_manager.active_terms}")
    print(f"[inspect] {HELP}")

    # Gripper command for the last action dimension. "keep" measures the first gripper joint against the binary
    # action's open/close targets (Robotiq finger_joint, Dex1 prismatic fingers) and commands what it already is.
    robot = env.scene["robot"]
    gripper_cfg = getattr(env_cfg.actions, "gripper", None)
    keep_gripper = args_cli.gripper == "keep" and gripper_cfg is not None and hasattr(gripper_cfg, "close_command_expr")
    if keep_gripper:
        joint = next(iter(gripper_cfg.close_command_expr))
        joint_id = robot.find_joints([joint])[0][0]
        g_open = float(gripper_cfg.open_command_expr.get(joint, 0.0))
        g_closed = float(gripper_cfg.close_command_expr[joint])

    def hold_action() -> torch.Tensor:
        action = torch.zeros((env.num_envs, env.action_space.shape[-1]), device=env.device)
        if keep_gripper:
            frac = (robot.data.joint_pos[:, joint_id] - g_open) / (g_closed - g_open)
            action[:, -1] = torch.where(frac > 0.1, -1.0, 1.0)
        else:
            action[:, -1] = -1.0 if args_cli.gripper == "closed" else 1.0
        return action

    keys = KeyboardControl()
    step = 0
    last_auto_reset = time.monotonic()
    with contextlib.suppress(KeyboardInterrupt):
        while simulation_app.is_running():
            if keys.reset_requested:
                keys.reset_requested = False
                env.reset()
                print(f"[inspect] step {step}: reset all {env.num_envs} envs")
            if args_cli.auto_reset_s > 0 and time.monotonic() - last_auto_reset >= args_cli.auto_reset_s:
                last_auto_reset = time.monotonic()
                env.reset()
                print(f"[inspect] step {step}: auto reset")
            if keys.paused and not keys.step_once:
                env.sim.render()
                continue
            keys.step_once = False

            _, _, terminated, truncated, _ = env.step(hold_action())
            step += 1
            if args_cli.print_every and step % args_cli.print_every == 0:
                done = (terminated | truncated).sum().item()
                print(f"[inspect] step {step}: {done} env(s) auto-reset by termination this step")
            if args_cli.max_steps and step >= args_cli.max_steps:
                break

    keys.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
