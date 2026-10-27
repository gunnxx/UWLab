"""Probe a reset-state environment: per-step breakdown of the success term's sub-conditions.

Companion of ``debug_grasp_env.py`` for when ``record_reset_states.py`` never reports a success. Prints, every
few steps, how many envs pass each clause of ``check_reset_state_success`` (gripper state, approach orientation,
stability, collision freedom, pose deviation) and the joint-velocity magnitudes. Flags isolate causes:
``--zero_gains`` (OSC off), ``--no_self_collisions``, ``--no_table``, ``--no_ik`` (skip the EE reset event),
``--kin_check`` (analytical FK/Jacobian vs PhysX), ``--substeps`` (torque/velocity per physics substep of the
first policy step), ``--mass_check``, ``--damping_ratio``, ``--arm_damping``, ``--armature``.

Example::

    python scripts_v2/tools/debug_reset_env.py --task OmniReset-G1dDex1-ObjectAnywhereEEAnywhere-v0 \
        --num_envs 16 --steps 45 --visualizer none env.scene.insertive_object=peg env.scene.receptive_object=peghole
"""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="OmniReset-G1dDex1-ObjectAnywhereEEAnywhere-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=45)
parser.add_argument("--zero_gains", action="store_true")
parser.add_argument("--no_self_collisions", action="store_true")
parser.add_argument("--no_table", action="store_true")
parser.add_argument("--no_ik", action="store_true")
parser.add_argument("--kin_check", action="store_true")
parser.add_argument("--detail", action="store_true")
parser.add_argument("--stale_check", action="store_true")
parser.add_argument("--torque_terms", action="store_true")
parser.add_argument("--substeps", action="store_true")
parser.add_argument("--mass_check", action="store_true")
parser.add_argument("--arm_damping", type=float, default=None)
parser.add_argument("--armature", type=float, default=None)
parser.add_argument("--damping_ratio", type=float, default=None)
parser.add_argument("--stiffness_scale", type=float, default=None)
parser.add_argument("--funnel", action="store_true", help="Accumulate, over all episodes that reach the hold end, how many pass each success clause.")
parser.add_argument("--close_gripper", action="store_true", help="Command the gripper closed (binary action -1), as the recorder does for grasped sets.")
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
import gymnasium as gym, torch  # noqa: E402
from typing import cast  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
import uwlab_tasks  # noqa: E402, F401
from uwlab_tasks.utils.hydra import hydra_task_compose  # noqa: E402

def f(t, n=4, d=3):
    return t[:n].detach().cpu().numpy().round(d).tolist()

@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.funnel:
        env_cfg.terminations.success.params["collect_funnel"] = True
    env_cfg.seed = None
    if args_cli.zero_gains:
        env_cfg.actions.arm.motion_stiffness = (0.0,) * 6
        env_cfg.actions.arm.nullspace_stiffness = 0.0
    if args_cli.no_self_collisions:
        env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
    if args_cli.no_table:
        env_cfg.scene.table = None
    if args_cli.no_ik:
        env_cfg.events.reset_end_effector_pose = None
    if args_cli.damping_ratio is not None:
        env_cfg.actions.arm.motion_damping_ratio = (args_cli.damping_ratio,) * 6
    if args_cli.stiffness_scale is not None:
        env_cfg.actions.arm.motion_stiffness = tuple(k * args_cli.stiffness_scale for k in env_cfg.actions.arm.motion_stiffness)
    if args_cli.arm_damping is not None:
        env_cfg.scene.robot.actuators["arm"].damping = args_cli.arm_damping
    if args_cli.armature is not None:
        env_cfg.scene.robot.actuators["arm"].armature = args_cli.armature
    print("[probe] arm actuator damping:", env_cfg.scene.robot.actuators["arm"].damping, "armature:", getattr(env_cfg.scene.robot.actuators["arm"], "armature", None))
    print("[probe] damping ratio:", env_cfg.actions.arm.motion_damping_ratio)
    print("[probe] table:", env_cfg.scene.table is not None, "ik event:", getattr(env_cfg.events, "reset_end_effector_pose", None) is not None or getattr(env_cfg.events, "reset_end_effector_pose_from_grasp_dataset", None) is not None)
    print("[probe] arm gains:", env_cfg.actions.arm.motion_stiffness, env_cfg.actions.arm.nullspace_stiffness, "self-coll:", env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions)
    env = cast(ManagerBasedRLEnv, gym.make(args_cli.task, cfg=env_cfg)).unwrapped
    robot = env.scene["robot"]
    term = env.termination_manager.get_term_cfg("success").func
    print("[probe] is_fixed_base:", robot.is_fixed_base, "joints:", robot.joint_names)
    print("[probe] ee body:", term.ee_body_name, "idx", term.ee_body_idx, "approach", term.gripper_approach_direction)
    print("[probe] max dev robot/object:", term.max_robot_pos_deviation, term.max_object_pos_deviation, "z thr", term.pos_z_threshold, "stab steps", term.consecutive_stability_steps)
    print("[probe] terminations:", env.termination_manager.active_terms, "max_episode_length", env.max_episode_length)
    env.reset()
    if args_cli.kin_check:
        arm = env.action_manager.get_term("arm")
        jid = arm._joint_ids if not isinstance(arm._joint_ids, slice) else list(range(robot.num_joints))[arm._joint_ids]
        ee = arm._ee_body_idx
        q = robot.data.joint_pos.torch[:, jid]
        pos_b, quat_b = arm._get_ee_pose_root_frame()
        from uwlab_assets.robots.unitree_g1d_dex1 import kinematics as kin
        fk_pos, fk_rot = kin.compute_fk(q, device=str(env.device))
        rot_b = math_utils.matrix_from_quat(quat_b)
        ang_err = torch.acos(torch.clamp(((fk_rot.transpose(1, 2) @ rot_b).diagonal(dim1=1, dim2=2).sum(-1) - 1) / 2, -1, 1))
        print("[kin] ee body:", robot.body_names[ee], "joint ids:", jid, "fixed base:", robot.is_fixed_base)
        print("[kin] physx ee pos (root frame):", f(pos_b, n=3), " analytical FK:", f(fk_pos, n=3))
        print("[kin] |pos err| per env:", f((fk_pos - pos_b).norm(dim=1), n=8, d=4), " rot err [rad]:", f(ang_err, n=8, d=3))
        Ja = arm._jacobian_fn(q, device=str(env.device))  # (N,6,7) root frame
        Jw = robot.data.body_link_jacobian_w.torch[:, ee - 1 if robot.is_fixed_base else ee][:, :, jid]  # (N,6,7) world
        R = math_utils.matrix_from_quat(robot.data.root_quat_w.torch)  # root->world
        Rt = R.transpose(1, 2)
        Jp = torch.cat([Rt @ Jw[:, :3], Rt @ Jw[:, 3:]], dim=1)
        print("[kin] physx J shape:", tuple(robot.data.body_link_jacobian_w.torch.shape))
        for i in range(Ja.shape[2]):
            ca = Ja[0, :, i]; cp = Jp[0, :, i]
            cos = torch.dot(ca, cp) / (ca.norm() * cp.norm() + 1e-9)
            print(f"[kin] joint {robot.joint_names[jid[i]]:28s} |Ja|={ca.norm():.3f} |Jphysx|={cp.norm():.3f} cos={cos:.3f}  Ja={f(ca, n=6)}  Jp={f(cp, n=6)}")
        # finite-difference check of the analytical Jacobian (position rows)
        eps = 1e-4
        Jfd = torch.zeros_like(Ja[:, :3])
        for i in range(q.shape[1]):
            dq = torch.zeros_like(q); dq[:, i] = eps
            Jfd[:, :, i] = (kin.compute_fk(q + dq, device=str(env.device))[0] - fk_pos) / eps
        print("[kin] analytical vs finite-diff (pos rows) max abs err:", float((Jfd - Ja[:, :3]).abs().max()))
    actions = torch.zeros(env.action_space.shape, device=env.device)
    actions[:, -1] = 1.0
    if args_cli.torque_terms:
        arm = env.action_manager.get_term("arm")
        jid = arm._joint_ids
        arm.process_actions(actions[:, :6])
        ee_pos_b, ee_quat_b = arm._get_ee_pose_root_frame()
        q = robot.data.joint_pos.torch[:, jid]; qd = robot.data.joint_vel.torch[:, jid]
        J = arm._jacobian_fn(q, device=str(env.device))
        ee_vel = torch.bmm(J, qd.unsqueeze(-1)).squeeze(-1)
        pos_error = arm._ee_pos_des - ee_pos_b
        quat_error = math_utils.quat_mul(arm._ee_quat_des, math_utils.quat_inv(ee_quat_b))
        aa = math_utils.axis_angle_from_quat(quat_error)
        pose_error = torch.cat([pos_error, aa], dim=-1)
        task_force = arm._kp * pose_error - arm._kd * ee_vel
        tau_task = torch.bmm(J.transpose(-1, -2), task_force.unsqueeze(-1)).squeeze(-1)
        print("[tau] ee_quat_b   =", f(ee_quat_b[0:1], d=4)[0], " ee_quat_des =", f(arm._ee_quat_des[0:1], d=4)[0])
        print("[tau] quat_error  =", f(quat_error[0:1], d=4)[0], " axis-angle =", f(aa[0:1], d=4)[0])
        print("[tau] pose_error  =", f(pose_error[0:1], d=4)[0], " ee_vel =", f(ee_vel[0:1], d=4)[0])
        print("[tau] task_force  =", f(task_force[0:1], d=3)[0], " tau_task =", f(tau_task[0:1], d=3)[0])
        if arm._nullspace_kp > 0:
            tau_post = arm._nullspace_kp * (arm._nullspace_q - q) - arm._nullspace_kd * qd
            print("[tau] nullspace q =", f(arm._nullspace_q[0:1], n=1, d=2)[0], " q =", f(q[0:1], n=1, d=2)[0], " tau_posture =", f(tau_post[0:1], d=3)[0])
        arm.apply_actions()
        print("[tau] effort target after apply_actions =", f(robot.data.joint_effort_target.torch[0:1], n=1, d=2)[0])
    if args_cli.mass_check:
        d = robot.data
        m = d.default_mass.torch[0] if hasattr(d, "default_mass") else None
        I = d.default_inertia.torch[0] if hasattr(d, "default_inertia") else None
        print("[mass] body_names:", len(robot.body_names), "mass attr:", m is not None, "inertia attr:", I is not None)
        for b, name in enumerate(robot.body_names):
            mi = float(m[b]) if m is not None else float("nan")
            Ii = I[b].reshape(3, 3).diagonal().tolist() if I is not None else None
            print(f"[mass] {name:34s} mass={mi:9.4f} I_diag={[round(x, 7) for x in Ii] if Ii else None}")
        try:
            print("[mass] view masses:", robot.root_view.get_masses().numpy()[0].round(4).tolist())
        except Exception as e:
            print("[mass] view masses: n/a", type(e).__name__, e)
    if args_cli.substeps:
        arm = env.action_manager.get_term("arm")
        jid = arm._joint_ids
        env.action_manager.process_action(actions)
        for i in range(env.cfg.decimation):
            env.action_manager.apply_action()
            tgt = robot.data.joint_effort_target.torch[0].clone()
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            ee_pos_b, ee_quat_b = arm._get_ee_pose_root_frame()
            perr = (arm._ee_pos_des - ee_pos_b)[0]
            qe = math_utils.axis_angle_from_quat(math_utils.quat_mul(arm._ee_quat_des, math_utils.quat_inv(ee_quat_b)))[0]
            print(f"[sub] {i:2d} effort_tgt={f(tgt, n=9, d=2)} applied={f(robot.data.applied_torque.torch[0], n=9, d=2)} q_dot={f(robot.data.joint_vel.torch[0], n=9, d=2)} |perr|={float(perr.norm()):.4f} |rot_err|={float(qe.norm()):.4f} finger_q={f(robot.data.joint_pos.torch[0, 7:9], d=4)} finger_tgt={f(robot.data.joint_pos_target.torch[0, 7:9], d=4)}")
    if args_cli.stale_check:
        from uwlab_assets.robots.unitree_g1d_dex1 import kinematics as kin
        arm = env.action_manager.get_term("arm")
        jid = arm._joint_ids
        def report(tag):
            q = robot.data.joint_pos.torch[:, jid]
            fk_pos, _ = kin.compute_fk(q, device=str(env.device))
            pos_b, _ = arm._get_ee_pose_root_frame()
            print(f"[stale] {tag}: q0={f(q[0], n=7, d=2)} FK(q)={f(fk_pos[0:1])[0]} data.ee_pos_b={f(pos_b[0:1])[0]} ee_pos_des={f(arm._ee_pos_des[0:1])[0]} |des-FK|={float((arm._ee_pos_des - fk_pos).norm(dim=1)[0]):.3f} |data-FK|={float((pos_b - fk_pos).norm(dim=1)[0]):.3f}")
        report("after env.reset()")
        env.step(actions)
        report("after step 0")
        env.reset()
        report("after 2nd env.reset()")
    funnel: dict[str, int] = {}
    def tally(name, mask):
        funnel[name] = funnel.get(name, 0) + int(mask.sum())
    grasp_event = term._resolve_grasp_event(env) if getattr(term, "require_grasp", False) else None
    if args_cli.close_gripper:
        actions[:, -1] = -1.0
    for t in range(args_cli.steps):
        _, _, terminated, truncated, _ = env.step(actions)
        N = env.num_envs
        jv = robot.data.joint_vel.torch
        abnormal_gripper = (jv.abs() > robot.data.joint_vel_limits.torch * 2).any(dim=1)
        ee_quat = robot.data.body_link_quat_w.torch[:, term.ee_body_idx]
        ee_pos = robot.data.body_link_pos_w.torch[:, term.ee_body_idx]
        appr = math_utils.quat_apply(ee_quat, torch.tensor(term.gripper_approach_direction, device=env.device, dtype=torch.float32).expand(N, -1))
        orient_ok = appr[:, 2] < -0.5
        stab = term.stability_counter >= term.consecutive_stability_steps
        dev_robot = (ee_pos - robot.initial_pos).norm(dim=1) if hasattr(robot, "initial_pos") else torch.full((N,), float("nan"), device=env.device)
        devs_obj = {a.cfg.prim_path.split("/")[-1]: (a.data.root_pos_w.torch - a.initial_pos).norm(dim=1) for a in term.object_assets if hasattr(a, "initial_pos")}
        all_ids = torch.arange(N, device=env.device)
        coll = [c(env, all_ids) for c in term.collision_analyzers]
        coll_free = torch.all(torch.stack(coll), dim=0)
        tm = env.termination_manager
        if args_cli.funnel and "grasp_placement_failed" in tm.active_terms:
            tally("ik_failed_early_out", tm.get_term("grasp_placement_failed"))
        if args_cli.detail:
            arm = env.action_manager.get_term("arm")
            pos_b, quat_b = arm._get_ee_pose_root_frame()
            perr = (arm._ee_pos_des - pos_b).norm(dim=1)
            tq = robot.data.applied_torque.torch if hasattr(robot.data, "applied_torque") else robot.data.joint_effort_target.torch
            print(f"[detail] t={t:3d} env0 q_dot={f(jv[0], n=9, d=2)} torque={f(tq[0], n=9, d=1)} |pos_err|={float(perr[0]):.3f} ee_pos_b={f(pos_b[0:1])[0]} q={f(robot.data.joint_pos.torch[0], n=7, d=2)}")
        if t % 5 == 0 or (terminated | truncated).any():
            print(f"[probe] t={t:3d} ep={int(env.episode_length_buf[0])} term={int(terminated.sum())} trunc={int(truncated.sum())} "
                  f"time_out={int(tm.get_term('time_out').sum())} abnormal={int(tm.get_term('abnormal_robot').sum())} success={int(tm.get_term('success').sum())}")
            print(f"[probe]   ok-counts/{N}: !abn_grip={int((~abnormal_gripper).sum())} orient={int(orient_ok.sum())} stable={int(stab.sum())} "
                  f"coll_free={int(coll_free.sum())} per-analyzer={[int(c.sum()) for c in coll]} "
                  f"dev_robot<max={int((dev_robot <= term.max_robot_pos_deviation).sum())} dev_obj<max={[int((d <= term.max_object_pos_deviation).sum()) for d in devs_obj.values()]} "
                  f"z_ok={int((ee_pos[:, 2] >= term.pos_z_threshold).sum())}")
            print(f"[probe]   ee_pos={f(ee_pos)} appr_z={f(appr[:, 2])} |jv|sum={f(jv.abs().sum(1))} dev_robot={f(dev_robot)} stabcnt={f(term.stability_counter.float())}")
            print(f"[probe]   q={f(robot.data.joint_pos.torch[0], n=9)} obj_z={[f(a.data.root_pos_w.torch[:, 2]) for a in term.object_assets]}")
    if args_cli.funnel:
        print("[funnel] counts over the run: early-outs, then episodes reaching the hold end and each clause at that step")
        for k, v in list(funnel.items()) + list(term.funnel.items()):
            print(f"[funnel] {k:28s} {v}")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
