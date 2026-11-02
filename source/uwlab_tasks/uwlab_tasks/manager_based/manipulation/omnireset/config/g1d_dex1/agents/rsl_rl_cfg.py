# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO configuration for the G1-D + Dex1-1 OmniReset tasks (the UR5e recipe plus an action clip)."""

from isaaclab.utils import configclass

from ...ur5e_robotiq_2f85.agents.rsl_rl_cfg import Base_PPORunnerCfg


@configclass
class G1dDex1_PPORunnerCfg(Base_PPORunnerCfg):
    """PPO runner for the G1-D + Dex1-1 tasks: the UR5e recipe with clipped actions, under its own experiment name."""

    experiment_name = "g1d_dex1_omnireset_agent"
    """Name of the experiment directory the runner logs to."""

    clip_actions = 7.0
    """Symmetric clip of the raw policy actions before scaling. With the OSC scale and gains it caps the commanded
    hand speed near the 0.32 m/s p99 of the UR5e policies (0.32 / 0.047 per unit action), see
    docs/notes/omnireset_osc_two_arms.html."""
