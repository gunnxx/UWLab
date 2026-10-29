# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree G1 (29 DoF humanoid) with Dex3 hands, standing, right arm and right hand movable, for OmniReset.

The robot description is the robot's own URDF (the gravity-feedforward solver's model, vendored here) assembled by
:mod:`.build_urdf` and converted to USD with ``scripts_v2/tools/convert_urdf_kit.py``; the stance and the hand
poses come from the robot's code (:mod:`.stance`). See ``docker/forge/g1d_pipeline.sh`` with ``ROBOT=g1_dex3``.
"""

from .actions import *  # noqa: F401, F403
from .g1_dex3 import *  # noqa: F401, F403
from .kinematics import *  # noqa: F401, F403
from .stance import *  # noqa: F401, F403
