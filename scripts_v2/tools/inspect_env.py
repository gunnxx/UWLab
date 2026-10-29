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
    C  close / open the hand         G  gravity on / off (only sticks with --no_gravity, see below)
    T  print the hand pose in the object frame (what record_grasps.py would record)

    Grasp-sampling tasks with ``fixed_grasp`` enabled: the keys below turn the nominal grasp about the object's
    axes by ``--angle_step`` degrees (SHIFT: a fifth of that) and reset at once; ``0`` restores the start values.

    J / L  roll  (about x)  -/+       I / K  pitch (about y)  +/-       U / O  yaw (about z)  -/+

Examples::

    # UR5e + Robotiq, 4 envs, reset distribution of the RL task
    python scripts_v2/tools/inspect_env.py --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
        --num_envs 4 --viz kit env.scene.insertive_object=peg env.scene.receptive_object=peghole

    # G1-D + Dex1, one of the reset-state sampling envs, never time out (R is the only reset)
    python scripts_v2/tools/inspect_env.py --task OmniReset-G1dDex1-ObjectAnywhereEEAnywhere-v0 \
        --num_envs 4 --viz kit --no_timeout

    # Dex3 grasp sampling pinned to one nominal grasp: palm approaching the standing peg from above, pinch point
    # 1.5 cm below the top, hand open, no gravity, frame marker at the palm. Tune the orientation with the
    # J/L I/K U/O keys, close the hand with C, switch gravity on with G to test the hold, print the pose with T.
    python scripts_v2/tools/inspect_env.py --task OmniReset-Dex3-GraspSampling-v0 --num_envs 1 --viz kit \
        --gripper open --no_timeout --no_gravity env.scene.object=peg \
        env.events.grasp_sampling.params.visualize_grasps=true \
        env.events.grasp_sampling.params.fixed_grasp.enabled=true \
        env.events.grasp_sampling.params.fixed_grasp.center=[0,0,0.015] \
        env.events.grasp_sampling.params.fixed_grasp.approach=[0,0,-1] \
        env.events.grasp_sampling.params.fixed_grasp.rpy_deg=[0,0,0]

    # headless smoke check: 100 steps, auto-reset every 2 s, then exit
    python scripts_v2/tools/inspect_env.py --task OmniReset-G1dDex1-RelCartesianOSC-State-Play-v0 \
        --num_envs 2 --viz none --max_steps 100 --auto_reset_s 2
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import TYPE_CHECKING, cast

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect a registered environment and reset it from the keyboard.")
parser.add_argument(
    "--task", type=str, required=True, help="Registered task id (see scripts/environments/list_envs.py)."
)
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
parser.add_argument(
    "--gripper",
    choices=["keep", "open", "closed"],
    default="keep",
    help="Gripper command while holding: keep its state after the reset (default), force open, or force closed.",
)
parser.add_argument("--no_timeout", action="store_true", help="Disable the time_out termination so only R resets.")
parser.add_argument(
    "--no_gravity",
    action="store_true",
    help=(
        "Start without gravity and without the OmniReset force/torque pokes on the object: the task's physics event"
        " never switches either on and sim gravity is zero; G toggles gravity."
    ),
)
parser.add_argument("--angle_step", type=float, default=5.0, help="Degrees per key press when turning a fixed grasp.")
parser.add_argument("--episode_length_s", type=float, default=None, help="Override the episode length (seconds).")
parser.add_argument(
    "--auto_reset_s", type=float, default=0.0, help="Also reset every N seconds of wall clock (0 = off)."
)
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

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg  # noqa: E402

from uwlab.utils.log_utils import configure_console_logging  # noqa: E402

import uwlab_tasks  # noqa: E402, F401
from uwlab_tasks.utils.hydra import hydra_task_compose  # noqa: E402

if TYPE_CHECKING:
    import carb

logger = logging.getLogger(__name__)

HELP = (
    "keys: [R] reset  [P] pause/resume  [N] single step  [C] close/open hand  [G] gravity on/off"
    "  [T] print hand pose in object frame  [H] help | fixed grasp: [J]/[L] roll -/+  [I]/[K] pitch +/-"
    "  [U]/[O] yaw -/+  (SHIFT: fine)  [0] restore"
)
"""Key bindings, logged when the keyboard is attached and on ``H``."""

RPY_KEYS = {"J": (0, -1.0), "L": (0, 1.0), "I": (1, 1.0), "K": (1, -1.0), "U": (2, -1.0), "O": (2, 1.0)}
"""Fixed-grasp orientation keys: key name to (Euler index, sign) of the angle step it applies."""


class KeyboardControl:
    """Flags toggled from the Kit window's keyboard.

    Subscribes to the default app window's keyboard on construction: ``R`` requests a reset of all environments,
    ``P`` toggles ``paused``, ``N`` requests a single step while paused and ``H`` logs the key bindings. Without
    a window (``--viz none``) or without carb input the subscription fails, a warning is logged and the flags
    simply keep their defaults, so the main loop runs unattended.
    """

    def __init__(self):
        """Constructor."""
        self.reset_requested = False
        self.paused = False
        self.step_once = False
        self.gripper_toggle_requested = False
        self.gravity_toggle_requested = False
        self.print_pose_requested = False
        self.restore_grasp_requested = False
        self.rpy_delta: list[float] = [0.0, 0.0, 0.0]
        """Pending change (degrees) of the fixed grasp's roll, pitch and yaw, consumed by the main loop."""
        self.angle_step = 5.0
        self._sub = None
        try:
            import carb
            import omni.appwindow

            self._carb = carb
            self._input = carb.input.acquire_input_interface()
            self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
            self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_event)
            logger.info(f"[inspect] keyboard attached. {HELP}")
        except Exception as exc:  # no app window (headless) or no carb input
            logger.warning(f"[inspect] keyboard unavailable ({exc!r}); use --viz kit for key presses.")

    def close(self):
        """Unsubscribe from the keyboard; a no-op when nothing was subscribed."""
        if self._sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
            self._sub = None

    # ===============
    # Private Methods
    # ===============

    def _on_event(self, event: carb.input.KeyboardEvent, *args: object, **kwargs: object) -> bool:
        """Keyboard callback: set the flag of a handled key press and let every event through.

        Args:
            event: Keyboard event from carb; only key presses are handled.
            *args: Ignored; carb passes extra positional arguments on some Kit versions.
            **kwargs: Ignored; see ``args``.

        Returns:
            Always ``True`` so the event keeps propagating.
        """
        if event.type != self._carb.input.KeyboardEventType.KEY_PRESS:
            return True
        key = event.input.name
        if key == "R":
            self.reset_requested = True
        elif key == "P":
            self.paused = not self.paused
            logger.info(f"[inspect] {'paused' if self.paused else 'resumed'}")
        elif key == "N":
            self.step_once = True
        elif key == "C":
            self.gripper_toggle_requested = True
        elif key == "G":
            self.gravity_toggle_requested = True
        elif key == "T":
            self.print_pose_requested = True
        elif key in ("KEY_0", "NUMPAD_0"):
            self.restore_grasp_requested = True
        elif key in RPY_KEYS:
            index, sign = RPY_KEYS[key]
            shift = bool(getattr(event, "modifiers", 0) & self._carb.input.KEYBOARD_MODIFIER_FLAG_SHIFT)
            self.rpy_delta[index] += sign * (self.angle_step / 5.0 if shift else self.angle_step)
        elif key == "H":
            logger.info(f"[inspect] {HELP}")
        return True


def log_object_physics(prim_path: str):
    """Log the authored mass and the physics material bound to every collider under a prim.

    Args:
        prim_path: Stage path of the object's root prim in environment 0.
    """
    import omni.usd
    from pxr import Usd, UsdPhysics, UsdShade

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        logger.warning(f"[inspect] no prim at {prim_path}")
        return
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.MassAPI):
            mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
            logger.info(f"[inspect] {prim.GetPath()}: authored physics:mass = {mass}")
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")
            if material:
                mat_prim = material.GetPrim()
                get = lambda name: mat_prim.GetAttribute(name).Get()  # noqa: E731
                logger.info(
                    f"[inspect] collider {prim.GetPath()}: material {mat_prim.GetPath()} static/dynamic friction"
                    f" {get('physics:staticFriction')}/{get('physics:dynamicFriction')}"
                    f" combine {get('physxMaterial:frictionCombineMode')}"
                )
            else:
                logger.info(f"[inspect] collider {prim.GetPath()}: no physics material bound (PhysX default 0.5)")


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=hydra_args)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: object):
    """Create the task, hold it still and reset it on the keyboard's or the wall clock's request.

    Args:
        env_cfg: Environment configuration composed by Hydra from the task's entry point.
        agent_cfg: Unused; the decorator passes the configuration of the second entry point.
    """
    configure_console_logging()
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
            logger.warning("[inspect] --no_timeout: this task has no 'time_out' termination term")
    if args_cli.no_gravity:
        # the OmniReset grasp/reset-state tasks switch gravity on from an event after a settling interval;
        # pushing that interval out of reach leaves the event's reset (gravity off) in force, and G toggles it
        gravity_event = getattr(getattr(env_cfg, "events", None), "global_physics_control_event", None)
        if gravity_event is not None:
            gravity_event.params["gravity_on_interval"] = (float("inf"), float("inf"))
            gravity_event.params["force_torque_on_interval"] = (float("inf"), float("inf"))
        env_cfg.sim.gravity = (0.0, 0.0, 0.0)

    env = cast(ManagerBasedRLEnv, gym.make(args_cli.task, cfg=env_cfg)).unwrapped
    env.reset()

    logger.info(
        f"[inspect] task {args_cli.task}: {env.num_envs} envs, episode {env_cfg.episode_length_s} s,"
        f" action dim {env.action_space.shape[-1]}"
    )
    logger.info(f"[inspect] terminations: {env.termination_manager.active_terms}")
    if "object" in env.scene.keys():  # grasp-sampling tasks: what the object variant ended up with in the stage
        masses = env.scene["object"].data.default_mass
        masses = masses.torch if hasattr(masses, "torch") else masses
        logger.info(f"[inspect] object mass per body (kg, env 0): {[round(m, 4) for m in masses[0].tolist()]}")
        # the scene has resolved the prim path to a regex over the environments; pick environment 0
        object_path = env.scene["object"].cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
        log_object_physics(object_path.replace("env_[^/]+", "env_0").replace("env_.*", "env_0"))
    if hasattr(env, "reward_manager"):
        logger.info(f"[inspect] rewards: {env.reward_manager.active_terms}")
    logger.info(f"[inspect] {HELP}")

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

    # None: follow --gripper; True/False after C was pressed: command closed/open
    gripper_closed: bool | None = None if args_cli.gripper == "keep" else args_cli.gripper == "closed"

    def hold_action() -> torch.Tensor:
        """Zero arm action plus the gripper command, so the OSC holds the current pose.

        Returns:
            Action tensor of shape ``(num_envs, action_dim)``.
        """
        action = torch.zeros((env.num_envs, env.action_space.shape[-1]), device=env.device)
        if gripper_closed is None and keep_gripper:
            frac = (robot.data.joint_pos[:, joint_id] - g_open) / (g_closed - g_open)
            action[:, -1] = torch.where(frac > 0.1, -1.0, 1.0)
        else:
            action[:, -1] = -1.0 if gripper_closed else 1.0
        return action

    # Fixed grasp of a grasp-sampling task, turned from the keyboard: the event instance holds the spec and the
    # cached candidate; clearing the cache makes the next reset rebuild the pose from the edited spec.
    grasp_term = None
    if "grasp_sampling" in getattr(env.event_manager, "active_terms", {}).get("reset", []):
        grasp_term = env.event_manager.get_term_cfg("grasp_sampling").func
        if not (getattr(grasp_term, "fixed_grasp", None) or {}).get("enabled", False):
            grasp_term = None
    grasp_spec_start = dict(grasp_term.fixed_grasp) if grasp_term is not None else None
    if grasp_term is not None:
        logger.info(f"[inspect] fixed grasp {grasp_spec_start}; J/L I/K U/O turn it, 0 restores it")

    def set_fixed_grasp_rpy(rpy_deg: list[float]):
        """Store new roll, pitch and yaw in the fixed grasp, drop the cached pose and reset.

        Args:
            rpy_deg: Roll, pitch and yaw in degrees about the object's x, y and z axes.
        """
        grasp_term.fixed_grasp = {**grasp_term.fixed_grasp, "rpy_deg": tuple(round(a, 3) for a in rpy_deg)}
        grasp_term.grasp_candidates = None
        env.reset()
        logger.info(f"[inspect] fixed grasp rpy_deg={grasp_term.fixed_grasp['rpy_deg']} (reset)")

    def log_hand_pose():
        """Log the gripper body's pose in the object frame, the quantity record_grasps.py records."""
        from isaaclab.utils import math as math_utils

        obj = env.scene["object"] if "object" in env.scene.keys() else None
        if obj is None or grasp_term is None:
            logger.warning("[inspect] T needs a grasp-sampling task (scene entity 'object' and event 'grasp_sampling')")
            return
        body_id = grasp_term.gripper_cfg.body_ids[0] if grasp_term.gripper_cfg.body_ids else 0
        pos, quat = math_utils.subtract_frame_transforms(
            obj.data.root_pos_w.torch,
            obj.data.root_quat_w.torch,
            robot.data.body_link_pos_w.torch[:, body_id],
            robot.data.body_link_quat_w.torch[:, body_id],
        )
        rpy = torch.stack(math_utils.euler_xyz_from_quat(quat), dim=1).rad2deg()
        for i in range(env.num_envs):
            logger.info(
                f"[inspect] env {i}: {robot.body_names[body_id]} in object frame: pos={pos[i].tolist()}"
                f" quat(w,x,y,z)={quat[i].tolist()} rpy_deg={rpy[i].tolist()}"
            )

    keys = KeyboardControl()
    keys.angle_step = args_cli.angle_step
    gravity_on = not args_cli.no_gravity
    step = 0
    last_auto_reset = time.monotonic()
    with contextlib.suppress(KeyboardInterrupt):
        while simulation_app.is_running():
            if keys.reset_requested:
                keys.reset_requested = False
                env.reset()
                logger.info(f"[inspect] step {step}: reset all {env.num_envs} envs")
            if keys.gripper_toggle_requested:
                keys.gripper_toggle_requested = False
                gripper_closed = not (gripper_closed if gripper_closed is not None else False)
                logger.info(f"[inspect] hand command: {'close' if gripper_closed else 'open'}")
            if keys.gravity_toggle_requested:
                keys.gravity_toggle_requested = False
                import carb

                gravity_on = not gravity_on
                env.sim.physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, -9.81 if gravity_on else 0.0))
                if not args_cli.no_gravity:
                    logger.warning("[inspect] the task's gravity event may switch gravity back; use --no_gravity")
                logger.info(f"[inspect] gravity {'on' if gravity_on else 'off'}")
            if keys.print_pose_requested:
                keys.print_pose_requested = False
                log_hand_pose()
            if any(abs(d) > 0 for d in keys.rpy_delta):
                delta, keys.rpy_delta = keys.rpy_delta, [0.0, 0.0, 0.0]
                if grasp_term is None:
                    logger.warning("[inspect] orientation keys need a grasp-sampling task with fixed_grasp enabled")
                else:
                    current = list(grasp_term.fixed_grasp.get("rpy_deg", (0.0, 0.0, 0.0)))
                    set_fixed_grasp_rpy([c + d for c, d in zip(current, delta)])
            if keys.restore_grasp_requested:
                keys.restore_grasp_requested = False
                if grasp_term is not None:
                    set_fixed_grasp_rpy(list(grasp_spec_start.get("rpy_deg", (0.0, 0.0, 0.0))))
            if args_cli.auto_reset_s > 0 and time.monotonic() - last_auto_reset >= args_cli.auto_reset_s:
                last_auto_reset = time.monotonic()
                env.reset()
                logger.info(f"[inspect] step {step}: auto reset")
            if keys.paused and not keys.step_once:
                env.sim.render()
                continue
            keys.step_once = False

            _, _, terminated, truncated, _ = env.step(hold_action())
            step += 1
            if args_cli.print_every and step % args_cli.print_every == 0:
                done = (terminated | truncated).sum().item()
                motion = ""
                if "object" in env.scene.keys():  # grasp-sampling tasks: is the object still?
                    obj = env.scene["object"]
                    lin = obj.data.root_lin_vel_w.torch.norm(dim=1).max().item()
                    ang = obj.data.root_ang_vel_w.torch.norm(dim=1).max().item()
                    motion = f"; object max speed {lin:.4f} m/s, {ang:.4f} rad/s"
                logger.info(f"[inspect] step {step}: {done} env(s) terminated this step{motion}")
            if args_cli.max_steps and step >= args_cli.max_steps:
                break

    keys.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
