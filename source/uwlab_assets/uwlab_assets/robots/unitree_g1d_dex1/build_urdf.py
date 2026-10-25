# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build the fixed-base, right-arm-only Unitree G1-D + Dex1-1 URDFs used by OmniReset.

Inputs (both from https://github.com/unitreerobotics/unitree_ros):
  robots/g1_d_description/g1_d.urdf                       official G1-D (wheeled chassis, lift, 2-axis waist, Dex3 hands)
  robots/g1_description/g1_29dof_mode_15_with_dex1_1.urdf G1 with Dex1-1 grippers; only the Dex1 links, joints and
                                                          meshes are taken from it

Outputs, in ``--out-dir``:
  g1d_dex1_right_arm.urdf   the robot: wheels, lift, waist and left arm frozen; right arm (7 revolute) + right Dex1-1
                            (2 prismatic fingers) movable; Dex3 hands replaced by Dex1-1 on both wrists
  dex1_gripper.urdf         the right Dex1-1 alone, rooted at its base link (floating gripper for grasp sampling)
  metadata.yaml             gripper offsets for the OmniReset reset pipeline and the right-arm kinematic chain for
                            the analytical Jacobian (see ``kinematics.py``)
  meshes/                   the referenced STL files, flattened into one folder

Frozen joints become fixed joints with the displacement folded into the joint origin, so the frozen posture is
part of the kinematic tree and survives the USD import unchanged.

Example::

    python -m uwlab_assets.robots.unitree_g1d_dex1.build_urdf \
        --unitree-ros /path/to/unitree_ros --out-dir source/uwlab_assets/data/Robots/Unitree/G1dDex1
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import yaml

# ---------------------------------------------------------------------------------------------------------------
# Defaults chosen for the OmniReset table-top setup (Vention table, top 0.855 m above the floor).
# ---------------------------------------------------------------------------------------------------------------
DEFAULT_WAIST_PITCH = 0.10
"""Frozen waist pitch [rad] (the G1-D joint named ``Yaw_Joint``, axis y). Small on purpose: the real waist carries
the upper body and the user asked to keep that load low. 0.1 rad adds ~3.5 cm of forward reach."""
DEFAULT_LIFT = 0.0
"""Frozen lift-column height [m], 0..0.42. Lowest is best: the table is already below the shoulders."""
DEFAULT_LEFT_ARM = (0.0, 0.15, 0.0, 1.5, 0.0, 0.0, 0.0)
"""Frozen left arm (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw) [rad]: forearm hanging straight down
beside the torso, well behind the table edge."""
DEFAULT_LEFT_FINGER = 0.0245
"""Frozen left Dex1-1 finger joints [m] (fully open)."""
WRIST_SMALL_EFFORT = 5.0
"""Effort limit [N m] of the wrist pitch/yaw motors (official G1 spec; the G1 URDF says 13.4)."""

RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
LEFT_ARM_JOINTS = [n.replace("right_", "left_") for n in RIGHT_ARM_JOINTS]
EE_LINK = "right_wrist_yaw_link"
GRIPPER_BASE_LINK = "right_dex1_base_link"
GRIPPER_FINGER_JOINTS = ["right_dex1_finger_joint_1", "right_dex1_finger_joint_2"]

# Dex1-1 geometry in the Dex1 base frame (x = approach, fingers slide along +-y). Measured on the official meshes:
# finger pads span x = 0.077..0.143 with inner faces at |y| = 0.0229 when q = 0; q in [-0.02, 0.0245] per finger.
DEX1_TCP_X = 0.12
DEX1_PAD_LENGTH = 0.066
DEX1_GAP_AT_ZERO = 0.0458
DEX1_FINGER_OPEN = 0.0245
DEX1_FINGER_CLOSED = -0.02


# ---------------------------------------------------------------------------------------------------------------
# Small rotation helpers (URDF rpy = Rz(yaw) Ry(pitch) Rx(roll))
# ---------------------------------------------------------------------------------------------------------------
def rpy_to_R(rpy) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def R_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    sy = -R[2, 0]
    cy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if cy > 1e-9:
        return (math.atan2(R[2, 1], R[2, 2]), math.atan2(sy, cy), math.atan2(R[1, 0], R[0, 0]))
    return (math.atan2(-R[1, 2], R[1, 1]), math.atan2(sy, cy), 0.0)


def axis_angle_R(axis, th: float) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def fmt(v) -> str:
    return " ".join(f"{float(x):.9g}" for x in v)


def parse_vec(s: str | None, n: int = 3) -> list[float]:
    return [float(x) for x in s.split()] if s else [0.0] * n


def origin_T(j: ET.Element) -> np.ndarray:
    o = j.find("origin")
    T = np.eye(4)
    if o is not None:
        T[:3, :3] = rpy_to_R(parse_vec(o.get("rpy")))
        T[:3, 3] = parse_vec(o.get("xyz"))
    return T


# ---------------------------------------------------------------------------------------------------------------
# URDF surgery
# ---------------------------------------------------------------------------------------------------------------
def freeze_joint(j: ET.Element, q: float) -> None:
    """Turn a revolute/prismatic joint into a fixed joint displaced by ``q`` about/along its axis."""
    origin = j.find("origin")
    if origin is None:
        origin = ET.SubElement(j, "origin")
        origin.set("xyz", "0 0 0")
        origin.set("rpy", "0 0 0")
    axis_el = j.find("axis")
    axis = parse_vec(axis_el.get("xyz")) if axis_el is not None else [0.0, 0.0, 1.0]
    xyz = np.array(parse_vec(origin.get("xyz")))
    R = rpy_to_R(parse_vec(origin.get("rpy")))
    if j.get("type") in ("revolute", "continuous"):
        R = R @ axis_angle_R(axis, q)
    elif j.get("type") == "prismatic":
        xyz = xyz + R @ (np.asarray(axis, float) * q)
    origin.set("xyz", fmt(xyz))
    origin.set("rpy", fmt(R_to_rpy(R)))
    j.set("type", "fixed")
    for tag in ("axis", "limit", "dynamics", "safety_controller", "calibration", "mimic"):
        el = j.find(tag)
        if el is not None:
            j.remove(el)


def subtree(root: ET.Element, start_link: str) -> tuple[set[str], set[str]]:
    """Links and joints strictly below ``start_link`` (links include ``start_link`` itself)."""
    children: dict[str, list[ET.Element]] = {}
    for j in root.findall("joint"):
        children.setdefault(j.find("parent").get("link"), []).append(j)
    links, joints, stack = {start_link}, set(), [start_link]
    while stack:
        link = stack.pop()
        for j in children.get(link, []):
            joints.add(j.get("name"))
            child = j.find("child").get("link")
            links.add(child)
            stack.append(child)
    return links, joints


def parent_chain(root: ET.Element, link: str) -> list[ET.Element]:
    """Joints from the tree root down to ``link`` (inclusive of the joint whose child is ``link``)."""
    by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
    chain = []
    while link in by_child:
        j = by_child[link]
        chain.append(j)
        link = j.find("parent").get("link")
    return list(reversed(chain))


def build_robot(g1d_urdf: str, g1_dex1_urdf: str, waist_pitch: float, waist_yaw: float, lift: float,
                left_arm, left_finger: float, name: str) -> tuple[ET.ElementTree, set[str]]:
    tree = ET.parse(g1d_urdf)
    root = tree.getroot()
    root.set("name", name)
    for el in list(root):
        if el.tag == "mujoco":
            root.remove(el)

    # 1. drop the Dex3 hands (everything below the two palm joints)
    removed_links, removed_joints = set(), set()
    for side in ("left", "right"):
        pj = root.find(f"joint[@name='{side}_hand_palm_joint']")
        if pj is None:
            raise SystemExit(f"{side}_hand_palm_joint not found in {g1d_urdf}")
        links, joints = subtree(root, pj.find("child").get("link"))
        removed_links |= links
        removed_joints |= joints | {pj.get("name")}
    for j in list(root.findall("joint")):
        if j.get("name") in removed_joints:
            root.remove(j)
    for link in list(root.findall("link")):
        if link.get("name") in removed_links:
            root.remove(link)

    # 2. splice the Dex1-1 (base + two fingers) from the G1 file onto both wrist_yaw links
    src = ET.parse(g1_dex1_urdf).getroot()
    dex1_meshes: set[str] = set()
    for side in ("left", "right"):
        bj = src.find(f"joint[@name='{side}_base_joint']")
        if bj is None:
            raise SystemExit(f"{side}_base_joint not found in {g1_dex1_urdf}")
        links, joints = subtree(src, bj.find("child").get("link"))
        for link in src.findall("link"):
            if link.get("name") in links:
                root.append(copy.deepcopy(link))
                for m in link.iter("mesh"):
                    dex1_meshes.add(os.path.basename(m.get("filename")))
        root.append(copy.deepcopy(bj))
        for j in src.findall("joint"):
            if j.get("name") in joints:
                root.append(copy.deepcopy(j))

    # 3. freeze everything that must not move
    frozen = {
        "Left_Wheel_Joint": 0.0,
        "Right_Wheel_Joint": 0.0,
        "LZ_mt_Joint": lift / 2.0,
        "LZ_it_Joint": lift / 2.0,
        "Yaw_Joint": waist_pitch,  # axis y: the waist pitch, despite its name
        "torso_Joint": waist_yaw,  # axis z: the waist yaw
        "left_dex1_finger_joint_1": left_finger,
        "left_dex1_finger_joint_2": left_finger,
    }
    frozen.update(dict(zip(LEFT_ARM_JOINTS, left_arm)))
    for j in root.findall("joint"):
        if j.get("name") in frozen:
            freeze_joint(j, frozen[j.get("name")])

    # 4. wrist pitch/yaw effort limit (official spec)
    for n in ("right_wrist_pitch_joint", "right_wrist_yaw_joint"):
        root.find(f"joint[@name='{n}']").find("limit").set("effort", f"{WRIST_SMALL_EFFORT:g}")
    return tree, dex1_meshes


def build_gripper(robot_root: ET.Element, name: str) -> ET.ElementTree:
    """The right Dex1-1 alone, rooted at its base link."""
    links, joints = subtree(robot_root, GRIPPER_BASE_LINK)
    g = ET.Element("robot", name=name)
    for link in robot_root.findall("link"):
        if link.get("name") in links:
            g.append(copy.deepcopy(link))
    for j in robot_root.findall("joint"):
        if j.get("name") in joints:
            g.append(copy.deepcopy(j))
    return ET.ElementTree(g)


def kinematic_chain(root: ET.Element) -> dict:
    """Root -> EE chain for the analytical Jacobian: the fixed transform from the articulation root to the first
    arm joint's parent frame, then one entry per arm joint (origin xyz/rpy and axis in the parent joint frame)."""
    chain = parent_chain(root, EE_LINK)
    T_base = np.eye(4)
    joints = []
    for j in chain:
        if j.get("name") in RIGHT_ARM_JOINTS:
            o = j.find("origin")
            joints.append({
                "name": j.get("name"),
                "xyz": [float(v) for v in parse_vec(o.get("xyz"))],
                "rpy": [float(v) for v in parse_vec(o.get("rpy"))],
                "axis": [float(v) for v in parse_vec(j.find("axis").get("xyz"))],
                "lower": float(j.find("limit").get("lower")),
                "upper": float(j.find("limit").get("upper")),
                "effort": float(j.find("limit").get("effort")),
                "velocity": float(j.find("limit").get("velocity")),
            })
        elif not joints:
            if j.get("type") != "fixed":
                raise SystemExit(f"joint {j.get('name')} above the arm is not fixed")
            T_base = T_base @ origin_T(j)
        else:
            raise SystemExit(f"unexpected joint {j.get('name')} between arm joints and {EE_LINK}")
    root_link = [link.get("name") for link in root.findall("link")
                 if link.get("name") not in {jj.find("child").get("link") for jj in root.findall("joint")}][0]
    return {
        "root_link": root_link,
        "ee_link": EE_LINK,
        "base_xyz": [float(v) for v in T_base[:3, 3]],
        "base_rpy": [float(v) for v in R_to_rpy(T_base[:3, :3])],
        "joints": joints,
    }


def write_metadata(path: str, root: ET.Element, frozen: dict) -> None:
    meta = {
        # ---- gripper (same keys as the UR5e/Robotiq metadata; frame = right_dex1_base_link, x = approach)
        "gripper_body_name": GRIPPER_BASE_LINK,
        "gripper_offset": {"pos": [DEX1_TCP_X, 0.0, 0.0], "quat": [0.5, 0.5, 0.5, 0.5]},
        "finger_offset": DEX1_TCP_X,
        "finger_clearance": DEX1_PAD_LENGTH,
        "maximum_aperture": round(DEX1_GAP_AT_ZERO + 2 * DEX1_FINGER_OPEN, 4),
        "gripper_approach_direction": [1, 0, 0],
        "grasp_align_axis": [1, 0, 0],
        "orientation_sample_axis": [1, 0, 0],
        "finger_open_joint_angle": DEX1_FINGER_OPEN,
        "gripper_open_joint_config": {n: DEX1_FINGER_OPEN for n in GRIPPER_FINGER_JOINTS},
        "gripper_closed_joint_config": {n: DEX1_FINGER_CLOSED for n in GRIPPER_FINGER_JOINTS},
        # ---- arm
        "arm_joint_names": RIGHT_ARM_JOINTS,
        "kinematic_chain": kinematic_chain(root),
        "frozen_joints": {k: float(v) for k, v in frozen.items()},
        "quat_convention": "xyzw",
    }
    with open(path, "w") as f:
        f.write("# Generated by uwlab_assets.robots.unitree_g1d_dex1.build_urdf -- do not edit by hand.\n")
        f.write("# Quaternions are (x, y, z, w), the Isaac Lab 3.0 convention.\n")
        yaml.safe_dump(meta, f, sort_keys=False)


def copy_meshes(root: ET.Element, out_meshes: str, src_dirs: list[str]) -> list[str]:
    os.makedirs(out_meshes, exist_ok=True)
    used = []
    for link in root.findall("link"):
        for m in link.iter("mesh"):
            fn = os.path.basename(m.get("filename"))
            for d in src_dirs:
                p = os.path.join(d, fn)
                if os.path.exists(p):
                    shutil.copy2(p, os.path.join(out_meshes, fn))
                    break
            else:
                raise SystemExit(f"mesh not found in {src_dirs}: {fn}")
            m.set("filename", f"meshes/{fn}")
            used.append(fn)
    return used


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unitree-ros", required=True, help="path to a checkout of unitreerobotics/unitree_ros")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--waist-pitch", type=float, default=DEFAULT_WAIST_PITCH)
    ap.add_argument("--waist-yaw", type=float, default=0.0)
    ap.add_argument("--lift", type=float, default=DEFAULT_LIFT)
    ap.add_argument("--left-arm", type=float, nargs=7, default=list(DEFAULT_LEFT_ARM))
    ap.add_argument("--left-finger", type=float, default=DEFAULT_LEFT_FINGER)
    args = ap.parse_args(argv)

    g1d = os.path.join(args.unitree_ros, "robots/g1_d_description/g1_d.urdf")
    g1 = os.path.join(args.unitree_ros, "robots/g1_description/g1_29dof_mode_15_with_dex1_1.urdf")
    for p in (g1d, g1):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
    os.makedirs(args.out_dir, exist_ok=True)

    tree, _ = build_robot(g1d, g1, args.waist_pitch, args.waist_yaw, args.lift, args.left_arm, args.left_finger,
                          name="g1d_dex1_right_arm")
    root = tree.getroot()
    gripper = build_gripper(root, name="dex1_gripper")

    mesh_dirs = [os.path.join(args.unitree_ros, "robots/g1_d_description/meshes"),
                 os.path.join(args.unitree_ros, "robots/g1_description/meshes")]
    used = copy_meshes(root, os.path.join(args.out_dir, "meshes"), mesh_dirs)
    copy_meshes(gripper.getroot(), os.path.join(args.out_dir, "meshes"), mesh_dirs)

    for t, fn in ((tree, "g1d_dex1_right_arm.urdf"), (gripper, "dex1_gripper.urdf")):
        ET.indent(t, space="  ")
        t.write(os.path.join(args.out_dir, fn), encoding="utf-8", xml_declaration=True)

    frozen = {"Yaw_Joint": args.waist_pitch, "torso_Joint": args.waist_yaw, "LZ_mt_Joint": args.lift / 2,
              "LZ_it_Joint": args.lift / 2, **dict(zip(LEFT_ARM_JOINTS, args.left_arm)),
              "left_dex1_finger_joint_1": args.left_finger, "left_dex1_finger_joint_2": args.left_finger}
    write_metadata(os.path.join(args.out_dir, "metadata.yaml"), root, frozen)

    movable = [j.get("name") for j in root.findall("joint") if j.get("type") != "fixed"]
    print(f"[build_urdf] wrote {args.out_dir}")
    print(f"[build_urdf] robot: {len(root.findall('link'))} links, {len(movable)} movable joints: {movable}")
    print(f"[build_urdf] gripper: {len(gripper.getroot().findall('link'))} links")
    size = sum(os.path.getsize(os.path.join(args.out_dir, 'meshes', f)) for f in set(used)) / 1e6
    print(f"[build_urdf] meshes: {len(set(used))} files, {size:.1f} MB")


if __name__ == "__main__":
    main()
