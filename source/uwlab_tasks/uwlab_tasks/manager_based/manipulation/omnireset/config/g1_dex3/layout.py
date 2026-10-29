# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Table-top layout for the standing Unitree G1 + Dex3 OmniReset scenes.

World frame: the environment origin is on the floor under the pelvis, x forward, y to the robot's left, z up. The
pelvis sits at z = :data:`~uwlab_assets.robots.unitree_g1_dex3.PELVIS_HEIGHT`, pitched 0.078 rad nose down, with the
legs bent as the robot stands (its teleoperation stance); the knees are the forward-most part of the body, at
x = :data:`KNEE_X`. The table is the plain block of the G1-D scene (the Vention top height, 0.684 m along x,
1.0 m along y) standing in front of the robot with its near edge 20 cm ahead of the pelvis, the standoff the
robot's own simulator keeps from a rectangular table.

Workspace numbers come from a kinematic probe of 2026-10-28 (damped-least-squares IK on the robot URDF, palm
normal pointing down within 45 deg, the reset event's 1 cm / 10 deg success thresholds): the palm origin is
reachable for x in [0.10, 0.38] and y in [-0.46, 0.18] at 10 to 30 cm above the table top, most reliably for
x in [0.14, 0.34] and with the fingers heading between -30 and +120 deg from straight ahead. The Dex3 pinch point
sits 9.8 cm ahead of the palm origin along the fingers and 7.2 cm below it when the palm faces down (see the
asset's ``build_urdf``), which is why the object regions sit ahead of the palm region. The left hand hangs in the
stance 7 cm above the table top at y in [0.08, 0.22], so the regions stop at y = 0.06.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

from uwlab_assets import UWLAB_ASSETS_DATA_DIR
from uwlab_assets.robots.unitree_g1_dex3 import PELVIS_HEIGHT

# ======================================================================================
# Table block: the G1-D block in front of the standing robot, 20 cm ahead of the pelvis.
# ======================================================================================

TABLE_TOP_Z = 0.855
"""Height of the table surface above the floor."""

TABLE_HALF_LENGTH = 0.342
"""Half of the table's long side (0.684 m), which runs along x."""

TABLE_HALF_WIDTH = 0.5
"""Half of the table's short side (1.0 m), along y."""

KNEE_X = 0.165
"""x of the forward-most point of the standing body (the bent right knee) in the stance."""

TABLE_EDGE_X = 0.20
"""x of the near table edge: 3.5 cm ahead of the knees, the robot simulator's standoff from a rectangular table."""

TABLE_CENTER_X = TABLE_EDGE_X + TABLE_HALF_LENGTH
"""x of the table block's centre: the near edge plus half of the long side."""

G1_DEX3_TABLE = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Table",
    init_state=RigidObjectCfg.InitialStateCfg(pos=(TABLE_CENTER_X, 0.0, TABLE_TOP_Z / 2)),
    spawn=sim_utils.CuboidCfg(
        size=(2 * TABLE_HALF_LENGTH, 2 * TABLE_HALF_WIDTH, TABLE_TOP_Z),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=40.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5, dynamic_friction=0.4),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.36, 0.38, 0.4)),
    ),
)
"""The plain table block (kinematic, reaches the floor)."""

# ================================================================================================================
# Object spawn regions are defined in the environment frame, which is the world frame with z = 0 at the table top.
# ================================================================================================================

RECEPTIVE_X = (0.26, 0.40)
"""x range (env frame) in which the receptive object is spawned on the table top."""

RECEPTIVE_Y = (-0.36, 0.04)
"""y range (env frame) in which the receptive object is spawned on the table top."""

INSERTIVE_X = (0.25, 0.41)
"""x range (env frame) in which the insertive object is spawned, slightly wider than the receptive one."""

INSERTIVE_Y = (-0.38, 0.06)
"""y range (env frame) in which the insertive object is spawned, slightly wider than the receptive one."""

INSERTIVE_Z_ABOVE_TABLE = (0.0, 0.25)
"""Height range above the table top in which the insertive object is spawned (object-anywhere sets)."""

# =======================================================================================================
# End-effector reset region is defined in the robot root frame: the pelvis frame, PELVIS_HEIGHT above the
# floor and pitched 0.078 rad nose down, so a root-frame point 0.3 m ahead sits 2.3 cm lower in the world
# than its root-frame z says. The region is the palm-origin strip of the IK probe.
# =======================================================================================================

EE_X = (0.14, 0.34)
"""x range (robot root frame) of the palm reset region: the reliable strip of the IK probe."""

EE_Y = (-0.42, 0.06)
"""y range (robot root frame) of the palm reset region, stopping short of the left hand."""

EE_Z_ROOT = (TABLE_TOP_Z - PELVIS_HEIGHT + 0.12, TABLE_TOP_Z - PELVIS_HEIGHT + 0.32)
"""z range (robot root frame) of the palm reset region: 12 cm to 32 cm above the table top, so the pinch point
(7 cm below a downward-facing palm) stays above the table."""

# ========
# Datasets
# ========

G1_DEX3_DATASET_DIR = os.path.join(UWLAB_ASSETS_DATA_DIR, "Datasets", "OmniReset_g1_dex3")
"""Reset states, grasps and partial assemblies recorded for this robot (robot-specific, generated by the
``datasets`` stage of ``docker/forge/g1d_pipeline.sh`` with ``ROBOT=g1_dex3``)."""
