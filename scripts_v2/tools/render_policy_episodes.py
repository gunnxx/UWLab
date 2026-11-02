# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Roll out a trained RSL-RL policy and render each episode to a video, headless.

One environment per episode: ``--num_episodes`` environments start from the task's own resets and run the policy
for one full episode each, a camera per environment records it, and every episode is saved as an MP4 with a poster
frame. Frames are captured at the simulation's render interval inside the decimation loop (physics at 120 Hz,
``--render_interval`` 4 gives 30 fps), so the videos play in real time. An episode ends at its first termination or
time-out; frames after that belong to the next episode and are dropped. ``episodes.json`` records, per episode, the
number of policy steps, whether it timed out or terminated early, and the task's success flag at the last step.
The camera pose is given in the environment frame.

    python scripts_v2/tools/render_policy_episodes.py --task OmniReset-G1dDex1-RelCartesianOSC-State-Play-v0 \\
        --checkpoint logs/rsl_rl/g1d_dex1_omnireset_agent/<run>/model_2200.pt --out_dir /tmp/episodes \\
        env.scene.insertive_object=peg env.scene.receptive_object=peghole
"""

from __future__ import annotations

import argparse
import logging

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Roll out a trained policy and render each episode to a video.")
parser.add_argument("--task", type=str, required=True, help="Play task (resets from the evaluation reset set).")
parser.add_argument("--checkpoint", type=str, required=True, help="RSL-RL checkpoint (``model_<iteration>.pt``).")
parser.add_argument("--out_dir", type=str, required=True, help="Where the videos are written.")
parser.add_argument("--num_episodes", type=int, default=10, help="Episodes to render (= number of environments).")
parser.add_argument("--dataset_dir", type=str, default=None, help="Dataset directory overriding the task's own.")
parser.add_argument("--seed", type=int, default=0, help="Environment seed; it picks the reset states.")
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--eye", type=float, nargs=3, default=(0.95, -0.85, 1.3), help="Camera position (env frame).")
parser.add_argument("--target", type=float, nargs=3, default=(0.37, -0.14, 0.9), help="Camera look-at (env frame).")
parser.add_argument("--focal_length", type=float, default=24.0, help="Pinhole focal length in mm (default 24).")
parser.add_argument("--render_interval", type=int, default=4, help="Physics steps per video frame.")
parser.add_argument("--crf", type=int, default=26, help="H.264 constant rate factor (lower is better quality).")
parser.add_argument(
    "--arm_damping",
    type=str,
    default=None,
    help="JSON object {joint regex: damping} replacing the joint damping of the robot's ``arm`` actuator.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

# the --enable_cameras CLI flag is gone in Isaac Lab 3.0; the intent is passed to the launcher directly
app_launcher = AppLauncher(args_cli, enable_cameras=True)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import imageio.v2 as imageio
import inspect
import json
import numpy as np
import os
import torch
from importlib import metadata
from typing import cast

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.sensors import CameraCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from PIL import Image
from rsl_rl import algorithms
from rsl_rl.runners import OnPolicyRunner

from uwlab.utils.log_utils import configure_console_logging

import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_compose

logger = logging.getLogger(__name__)


def sanitize_agent_cfg(agent_cfg: RslRlOnPolicyRunnerCfg) -> RslRlOnPolicyRunnerCfg:
    """Make the runner configuration loadable by the installed rsl-rl, as ``rsl_rl/cli_args.py`` does for play.

    Args:
        agent_cfg: Runner configuration composed by Hydra.

    Returns:
        The configuration without the algorithm keys the installed algorithm class does not accept.
    """
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    alg_class = getattr(algorithms, getattr(agent_cfg.algorithm, "class_name", ""), None)
    if alg_class is not None:
        accepted = set(inspect.signature(alg_class.__init__).parameters)
        for key in list(vars(agent_cfg.algorithm)):
            if key != "class_name" and key not in accepted:
                delattr(agent_cfg.algorithm, key)
    return agent_cfg


@hydra_task_compose(args_cli.task, "rsl_rl_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Run one episode per environment with the policy and write a video per episode.

    Args:
        env_cfg: Environment configuration composed by Hydra from the task's entry point.
        agent_cfg: Runner configuration composed by Hydra from the task's RSL-RL entry point.
    """
    configure_console_logging()
    num = args_cli.num_episodes
    env_cfg.scene.num_envs = num
    env_cfg.scene.env_spacing = 6.0  # keep neighbouring environments out of the frame
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.render_interval = args_cli.render_interval
    if args_cli.dataset_dir is not None:
        env_cfg.events.reset_from_reset_states.params["dataset_dir"] = args_cli.dataset_dir
    if args_cli.arm_damping is not None:
        env_cfg.scene.robot.actuators["arm"].damping = json.loads(args_cli.arm_damping)
    env_cfg.scene.camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.0,
        width=args_cli.width,
        height=args_cli.height,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=args_cli.focal_length, clipping_range=(0.05, 30.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), convention="world"),
    )

    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    env = cast(ManagerBasedRLEnv, gym_env.unwrapped)
    agent_cfg = sanitize_agent_cfg(agent_cfg)
    vec_env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.device)

    device = env.device
    origins = env.scene.env_origins
    camera = env.scene["camera"]
    camera.set_world_poses_from_view(
        origins + torch.tensor(args_cli.eye, device=device), origins + torch.tensor(args_cli.target, device=device)
    )

    os.makedirs(args_cli.out_dir, exist_ok=True)
    fps = round(1.0 / (env.physics_dt * args_cli.render_interval))
    writers = [
        imageio.get_writer(
            os.path.join(args_cli.out_dir, f"episode_{i:02d}.mp4"),
            fps=fps,
            codec="libx264",
            quality=None,
            pixelformat="yuv420p",
            macro_block_size=16,
            ffmpeg_params=["-crf", str(args_cli.crf), "-preset", "slow", "-movflags", "+faststart"],
        )
        for i in range(num)
    ]
    frames_per_step = env.cfg.decimation // args_cli.render_interval
    recording = torch.ones(num, dtype=torch.bool)
    frame_budget = [0]
    poster_saved = [False]

    def grab(poster: bool = False):
        """Read the camera and append a frame to the video of every episode still running.

        Args:
            poster: Also save the frame as each episode's poster image.
        """
        camera.update(dt=0.0, force_recompute=True)
        rgb = camera.data.output["rgb"][:, :, :, :3].cpu().numpy().astype(np.uint8)
        for i in range(num):
            if recording[i]:
                writers[i].append_data(rgb[i])
            if poster:
                Image.fromarray(rgb[i]).save(os.path.join(args_cli.out_dir, f"episode_{i:02d}.jpg"), quality=88)

    # The environment renders inside its decimation loop every ``render_interval`` physics steps; a frame is taken
    # after each of those renders. Renders beyond ``frames_per_step`` in one step are the re-renders after a reset,
    # which already show the next episode. The first frame is taken after the first physics step: before it, the
    # renderer still shows the objects where the scene spawned them, not where the reset placed them.
    sim_render = env.sim.render

    def render_and_grab(*args, **kwargs):
        """Render, then capture a frame while the current step's frame budget lasts."""
        sim_render(*args, **kwargs)
        if frame_budget[0] > 0:
            frame_budget[0] -= 1
            grab(poster=not poster_saved[0])
            poster_saved[0] = True

    env.sim.render = render_and_grab

    obs = vec_env.get_observations()
    for _ in range(6):
        sim_render()  # settle materials and lighting before the first frame

    success_term = env.reward_manager.get_term_cfg("progress_context").func
    max_steps = int(env.max_episode_length)
    steps = torch.zeros(num, dtype=torch.long)
    timed_out = torch.zeros(num, dtype=torch.bool)
    success = torch.zeros(num, dtype=torch.bool)
    with torch.inference_mode():
        for step in range(max_steps):
            frame_budget[0] = frames_per_step
            obs, _, dones, _ = vec_env.step(policy(obs))
            done = dones.bool().cpu() & recording
            if done.any():
                ids = done.nonzero().squeeze(-1)
                steps[ids] = step + 1
                timed_out[ids] = env.termination_manager.time_outs.cpu()[ids]
                success[ids] = success_term.success.bool().cpu()[ids]
                recording &= ~done
            if not recording.any():
                break
            if (step + 1) % 20 == 0:
                logger.info(f"[episodes] step {step + 1}/{max_steps}, {int(recording.sum())} episodes running")
    env.sim.render = sim_render

    for w in writers:
        w.close()
    episodes = [
        {"episode": i, "steps": int(steps[i]), "timed_out": bool(timed_out[i]), "success": bool(success[i])}
        for i in range(num)
    ]
    summary = {
        "task": args_cli.task,
        "checkpoint": args_cli.checkpoint,
        "overrides": remaining_args,
        "arm_damping": json.loads(args_cli.arm_damping) if args_cli.arm_damping else None,
        "seed": args_cli.seed,
        "fps": fps,
        "step_dt": env.step_dt,
        "max_steps": max_steps,
        "episodes": episodes,
    }
    with open(os.path.join(args_cli.out_dir, "episodes.json"), "w") as f:
        json.dump(summary, f, indent=1)
    logger.info(f"[episodes] {int(success.sum())}/{num} successful | steps {steps.tolist()}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
