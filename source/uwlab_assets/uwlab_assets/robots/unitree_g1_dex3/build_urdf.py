# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build the fixed-base, right-arm-only Unitree G1 (29 DoF) + Dex3 URDFs used by OmniReset.

Input, vendored next to this module as ``g1_body29_hand14.urdf`` (BSD-3-Clause, ``LICENSE.unitree_ros``): the
robot's own URDF, the one its command path computes the arm and waist gravity feedforward from (ludo-research
``core/g1_command/src/ludo_g1_command/assets/g1_body29_hand14.urdf``, commit ``f1df78bd1`` of 2026-10-20, sha256
``63b8d1178113a1311de3c89a1d0db98d4366c076d0ec57727ae85f307d553b1e``). It is Unitree's
``g1_29dof_with_hand_rev_1_0.urdf`` with the robot's sensor frames and the joint limits the robot runs with.
Checked against ``unitree_ros`` ``deeb607f`` (2026-10-28): every link inertial is identical (34.394 kg in total,
0.697 kg per Dex3), the upstream file has since changed some joint limits (ankle and waist effort 35 instead of
50 N m and velocity 30 instead of 37 rad/s, ``thumb_0`` velocity 3.14 instead of 6.857 rad/s, ``thumb_1`` range
+-0.611 instead of +-0.724 rad) and the mid360 frame; the robot's values are kept.

Building the simulation asset from this file is what keeps the simulated robot and the gravity solver on one mass
model: the Isaac Lab URDF importer takes mass, centre of mass and inertia of every link from the URDF (link
density 0). The hand masses are the part that went wrong before: ludo-research's MuJoCo rig had grafted the Dex3
onto Unitree's MJCF without the finger inertials (0.146 kg instead of 0.324 kg of fingers per hand, the palm at
0.255 kg instead of 0.457 kg) and the arms rose under the feedforward of the correct URDF (commit ``b5c9c5e5c``).
``scripts_v2/tools/check_usd_inertials.py`` compares the converted USD with the URDF link by link.

Meshes are not vendored; ``--meshes`` points at ``robots/g1_description/meshes`` of a ``unitree_ros`` checkout
(the URDF references them by basename).

Outputs, in ``--out-dir``:
  g1_dex3_right_arm.urdf  the robot: legs, waist, left arm and left hand frozen in the stance of :mod:`.stance`;
                          right arm (7 revolute) + right Dex3 (7 revolute) movable; rooted at ``pelvis``
  dex3_hand.urdf          the right Dex3 alone, rooted at ``right_hand_palm_link`` (floating hand for grasp sampling)
  metadata.yaml           gripper offsets for the OmniReset reset pipeline and the right-arm kinematic chain for
                          the analytical Jacobian (see ``kinematics.py``)
  meshes/                 the referenced STL files, flattened into one folder

Frozen joints become fixed joints with the displacement folded into the joint origin (see
:mod:`uwlab_assets.robots.unitree_g1d_dex1.build_urdf`, whose URDF helpers this module reuses).

Example::

    python -m uwlab_assets.robots.unitree_g1_dex3.build_urdf \
        --meshes /path/to/unitree_ros/robots/g1_description/meshes \
        --out-dir source/uwlab_assets/data/Robots/Unitree/G1Dex3
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import xml.etree.ElementTree as ET
import yaml

from uwlab_assets.robots.unitree_g1d_dex1.build_urdf import _freeze_joint, _subtree, copy_meshes, kinematic_chain

from uwlab.utils.log_utils import configure_console_logging

from .paths import G1_SOLVER_URDF_PATH
from .stance import (
    DEX3_OPEN_RIGHT,
    DEX3_RIGHT_JOINT_NAMES,
    DEX3_SHUT_RIGHT,
    STANCE_LEFT_ARM,
    STANCE_LEFT_HAND,
    STANCE_LEGS,
    STANCE_WAIST,
)

logger = logging.getLogger(__name__)

# ==========================================================================================================
# Links and joints of the movable part. The arm joint names and the flange link are the ones of the G1-D, so
# the kinematic-chain writer of that package applies unchanged.
# ==========================================================================================================

RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
"""The seven right-arm joints, shoulder to wrist."""

EE_LINK = "right_wrist_yaw_link"
"""Flange link the kinematic chain ends at (the Dex3 palm is bolted to it)."""

PALM_LINK = "right_hand_palm_link"
"""Root link of the right Dex3; the gripper body of the reset pipeline."""

# ==========================================================================================================
# Dex3 pinch geometry in the palm frame: x along the fingers, +y the side the thumb sits on and the fingers curl
# toward (the palm normal), z across the index and middle fingers. Measured on the official meshes while the hand
# closes from DEX3_OPEN_RIGHT to DEX3_SHUT_RIGHT (kinematic probe of 2026-10-28): the thumb comes from the palm
# side and the two fingertips curl in from the front, so an object is squeezed along x and sits in a pocket in
# front of the palm's thumb edge; the pocket centre is at (0.100, 0.072) for a 50 mm object and (0.099, 0.072) for
# a 30 mm one, (0.083, 0.041) with the hand shut on nothing. The hand therefore approaches along +y (palm down
# for a table-top grasp) and the pinch point is offset from the palm origin both along the approach and along x.
# ==========================================================================================================

DEX3_PINCH_X = 0.098
"""x [m] (palm frame) of the pinch point, the object centre of a 30 to 50 mm object held between thumb and fingers."""

DEX3_PINCH_Y = 0.072
"""y [m] (palm frame) of the pinch point: its distance from the palm origin along the approach direction."""

DEX3_POCKET_DEPTH = 0.03
"""Depth [m] along the approach over which an object still ends between thumb and fingertips (the pocket spans
y = 0.05 to 0.085 m)."""

DEX3_MAX_APERTURE = 0.09
"""Largest object extent [m] the sampler accepts along the approach ray (the ``maximum_aperture`` of the
UR5e/Robotiq metadata; the pocket itself takes objects up to about 8 cm along x)."""

DEX3_TCP_QUAT_XYZW = (-0.70710678, 0.0, 0.0, 0.70710678)
"""Orientation (x, y, z, w) of the pinch frame in the palm frame: z along the approach (+y of the palm), x along
the squeeze direction (+x of the palm), the convention of the Robotiq and Dex1 metadata."""


def build_robot(urdf: str, name: str) -> ET.ElementTree:
    """Assemble the right-arm-only G1 + Dex3 from the robot's URDF.

    Steps: drop the ``<mujoco>`` block, freeze the legs, the waist, the left arm and the left hand in the stance
    and leave the right arm and the right Dex3 movable. Every link keeps its inertial as written.

    Args:
        urdf: Path to the robot's URDF (:data:`~.paths.G1_SOLVER_URDF_PATH`).
        name: Name attribute of the generated ``<robot>`` element.

    Returns:
        The robot element tree.
    """
    tree = ET.parse(urdf)
    root = tree.getroot()
    root.set("name", name)
    for el in list(root):
        if el.tag == "mujoco":
            root.remove(el)
    frozen = frozen_joints()
    for j in root.findall("joint"):
        if j.get("name") in frozen:
            _freeze_joint(j, frozen[j.get("name")])
    movable = [j.get("name") for j in root.findall("joint") if j.get("type") != "fixed"]
    expected = RIGHT_ARM_JOINTS + DEX3_RIGHT_JOINT_NAMES
    if sorted(movable) != sorted(expected):
        raise SystemExit(f"movable joints {movable} differ from the expected {expected}")
    return tree


def frozen_joints() -> dict[str, float]:
    """The joints frozen in the robot and the positions they are frozen at.

    Returns:
        Joint name -> position [rad]: legs, waist, left arm and left hand of the stance.
    """
    return {**STANCE_LEGS, **STANCE_WAIST, **STANCE_LEFT_ARM, **STANCE_LEFT_HAND}


def build_hand(robot_root: ET.Element, name: str) -> ET.ElementTree:
    """Extract the right Dex3 alone, rooted at its palm link.

    Args:
        robot_root: ``<robot>`` element of the assembled robot (see :func:`build_robot`).
        name: Name attribute of the generated ``<robot>`` element.

    Returns:
        Element tree holding the links and joints below :data:`PALM_LINK`.
    """
    links, joints = _subtree(robot_root, PALM_LINK)
    g = ET.Element("robot", name=name)
    for link in robot_root.findall("link"):
        if link.get("name") in links:
            g.append(copy.deepcopy(link))
    for j in robot_root.findall("joint"):
        if j.get("name") in joints:
            g.append(copy.deepcopy(j))
    return ET.ElementTree(g)


def check_inertials(source_urdf: str, root: ET.Element) -> tuple[float, float]:
    """Verify that every link of the assembled robot carries the inertial of the source URDF.

    Args:
        source_urdf: Path to the URDF the robot was assembled from.
        root: ``<robot>`` element of the assembled robot.

    Returns:
        The total mass and the mass of the right Dex3 (palm and fingers) [kg].

    Raises:
        SystemExit: If a link is missing or its inertial differs from the source.
    """
    source = {link.get("name"): link.find("inertial") for link in ET.parse(source_urdf).getroot().findall("link")}
    total, hand = 0.0, 0.0
    for link in root.findall("link"):
        name = link.get("name")
        if name not in source:
            raise SystemExit(f"link {name} is not in {source_urdf}")
        src, dst = source[name], link.find("inertial")
        if (src is None) != (dst is None) or (src is not None and ET.tostring(src) != ET.tostring(dst)):
            raise SystemExit(f"inertial of {name} differs from {source_urdf}")
        if dst is not None:
            mass = float(dst.find("mass").get("value"))
            total += mass
            if name.startswith("right_hand_"):
                hand += mass
    return total, hand


def write_metadata(path: str, root: ET.Element, frozen: dict[str, float]):
    """Write ``metadata.yaml``: gripper offsets, the arm kinematic chain and the frozen joint values.

    The gripper keys are the ones of the UR5e/Robotiq and G1-D/Dex1 metadata; their frame is
    ``right_hand_palm_link`` with +y the approach direction (see the pinch-geometry constants).

    Args:
        path: Output file path.
        root: ``<robot>`` element of the assembled robot.
        frozen: Frozen joint name -> value, recorded for reference.
    """
    meta = {
        "gripper_body_name": PALM_LINK,
        "gripper_offset": {"pos": [DEX3_PINCH_X, DEX3_PINCH_Y, 0.0], "quat": list(DEX3_TCP_QUAT_XYZW)},
        "finger_offset": DEX3_PINCH_Y,
        "finger_clearance": DEX3_POCKET_DEPTH,
        "maximum_aperture": DEX3_MAX_APERTURE,
        "gripper_approach_direction": [0, 1, 0],
        "grasp_align_axis": [0, 1, 0],
        "orientation_sample_axis": [0, 1, 0],
        "gripper_open_joint_config": dict(DEX3_OPEN_RIGHT),
        "gripper_closed_joint_config": dict(DEX3_SHUT_RIGHT),
        "arm_joint_names": RIGHT_ARM_JOINTS,
        "kinematic_chain": kinematic_chain(root),
        "frozen_joints": {k: float(v) for k, v in frozen.items()},
        "quat_convention": "xyzw",
    }
    with open(path, "w") as f:
        f.write("# Generated by uwlab_assets.robots.unitree_g1_dex3.build_urdf -- do not edit by hand.\n")
        f.write("# Quaternions are (x, y, z, w), the Isaac Lab 3.0 convention.\n")
        yaml.safe_dump(meta, f, sort_keys=False)


def main(argv: list[str] | None = None):
    """Command-line entry point: build both URDFs, the mesh folder and ``metadata.yaml`` in ``--out-dir``.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.
    """
    configure_console_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meshes", required=True, help="robots/g1_description/meshes of a unitree_ros checkout")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--urdf", default=G1_SOLVER_URDF_PATH, help="the robot URDF (default: the vendored one)")
    args = ap.parse_args(argv)
    for p in (args.urdf, args.meshes):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
    os.makedirs(args.out_dir, exist_ok=True)

    tree = build_robot(args.urdf, name="g1_dex3_right_arm")
    root = tree.getroot()
    hand = build_hand(root, name="dex3_hand")
    total, hand_mass = check_inertials(args.urdf, root)

    used = copy_meshes(root, os.path.join(args.out_dir, "meshes"), [args.meshes])
    copy_meshes(hand.getroot(), os.path.join(args.out_dir, "meshes"), [args.meshes])
    for t, fn in ((tree, "g1_dex3_right_arm.urdf"), (hand, "dex3_hand.urdf")):
        ET.indent(t, space="  ")
        t.write(os.path.join(args.out_dir, fn), encoding="utf-8", xml_declaration=True)
    write_metadata(os.path.join(args.out_dir, "metadata.yaml"), root, frozen_joints())

    movable = [j.get("name") for j in root.findall("joint") if j.get("type") != "fixed"]
    massless = [link.get("name") for link in root.findall("link") if link.find("inertial") is None]
    logger.info(f"[build_urdf] wrote {args.out_dir}")
    logger.info(f"[build_urdf] robot: {len(root.findall('link'))} links, {len(movable)} movable joints: {movable}")
    logger.info(
        f"[build_urdf] mass: {total:.4f} kg in total, right Dex3 {hand_mass:.4f} kg; massless links: {massless}"
    )
    logger.info(f"[build_urdf] hand: {len(hand.getroot().findall('link'))} links")
    size = sum(os.path.getsize(os.path.join(args.out_dir, "meshes", f)) for f in set(used)) / 1e6
    logger.info(f"[build_urdf] meshes: {len(set(used))} files, {size:.1f} MB")


if __name__ == "__main__":
    main()
