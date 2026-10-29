# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Asset locations of the G1 + Dex3 robot (kept free of simulator imports)."""

import os

from uwlab_assets import UWLAB_ASSETS_DATA_DIR

G1_DEX3_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
"""Folder of this package: holds the vendored robot URDF the assets are built from."""

G1_SOLVER_URDF_PATH = os.path.join(G1_DEX3_PACKAGE_DIR, "g1_body29_hand14.urdf")
"""The robot's own URDF (the gravity-feedforward solver's model); see :mod:`.build_urdf` for its provenance."""

G1_DEX3_ASSET_DIR = os.path.join(UWLAB_ASSETS_DATA_DIR, "Robots", "Unitree", "G1Dex3")
"""Folder holding the generated URDF/USD, meshes and ``metadata.yaml`` (gitignored, built by the pipeline)."""

G1_DEX3_USD_PATH = os.path.join(G1_DEX3_ASSET_DIR, "g1_dex3_right_arm.usd")
"""USD of the fixed-base robot with the right arm and right Dex3 movable."""

DEX3_HAND_USD_PATH = os.path.join(G1_DEX3_ASSET_DIR, "dex3_hand.usd")
"""USD of the right Dex3 alone, rooted at its palm link (grasp sampling)."""

G1_DEX3_METADATA_PATH = os.path.join(G1_DEX3_ASSET_DIR, "metadata.yaml")
"""Gripper offsets and the right-arm kinematic chain written by :mod:`.build_urdf`."""
