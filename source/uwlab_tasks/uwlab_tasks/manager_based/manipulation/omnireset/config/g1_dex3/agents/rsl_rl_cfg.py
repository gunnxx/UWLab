# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO configuration for the G1 + Dex3 OmniReset tasks (same recipe as the UR5e)."""

from isaaclab.utils import configclass

from ...ur5e_robotiq_2f85.agents.rsl_rl_cfg import Base_PPORunnerCfg


@configclass
class G1Dex3_PPORunnerCfg(Base_PPORunnerCfg):
    """PPO runner for the G1 + Dex3 tasks: the UR5e recipe under its own experiment name."""

    experiment_name = "g1_dex3_omnireset_agent"
    """Name of the experiment directory the runner logs to."""
