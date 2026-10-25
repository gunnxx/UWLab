# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Asset locations of the G1-D + Dex1-1 robot (kept free of simulator imports)."""

import os

from uwlab_assets import UWLAB_ASSETS_DATA_DIR

G1D_DEX1_ASSET_DIR = os.path.join(UWLAB_ASSETS_DATA_DIR, "Robots", "Unitree", "G1dDex1")
"""Folder holding the generated URDF/USD, meshes and ``metadata.yaml``."""

G1D_DEX1_USD_PATH = os.path.join(G1D_DEX1_ASSET_DIR, "g1d_dex1_right_arm.usd")
DEX1_GRIPPER_USD_PATH = os.path.join(G1D_DEX1_ASSET_DIR, "dex1_gripper.usd")
G1D_DEX1_METADATA_PATH = os.path.join(G1D_DEX1_ASSET_DIR, "metadata.yaml")
