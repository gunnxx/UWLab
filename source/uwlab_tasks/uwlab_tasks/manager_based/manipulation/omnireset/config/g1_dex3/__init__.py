# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OmniReset tasks for the standing Unitree G1 (right arm) with a Dex3 hand.

Pipeline (see ``docker/forge/g1d_pipeline.sh`` with ``ROBOT=g1_dex3``):
  1. ``OmniReset-PartialAssemblies-v0`` (robot-independent, from the UR5e package) -> partial assemblies
  2. ``OmniReset-Dex3-GraspSampling-v0`` -> grasps for the Dex3
  3. ``OmniReset-G1Dex3-<ResetType>-v0`` x4 -> reset-state datasets
  4. ``OmniReset-G1Dex3-RelCartesianOSC-State-v0`` -> stage 1 PPO
"""

import gymnasium as gym

from . import agents

gym.register(
    id="OmniReset-Dex3-GraspSampling-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.grasp_sampling_cfg:Dex3GraspSamplingCfg"},
    disable_env_checker=True,
)

gym.register(
    id="OmniReset-G1Dex3-ObjectAnywhereEEAnywhere-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:G1Dex3ObjectAnywhereEEAnywhereResetStatesCfg"},
)

gym.register(
    id="OmniReset-G1Dex3-ObjectRestingEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:G1Dex3ObjectRestingEEGraspedResetStatesCfg"},
)

gym.register(
    id="OmniReset-G1Dex3-ObjectAnywhereEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:G1Dex3ObjectAnywhereEEGraspedResetStatesCfg"},
)

gym.register(
    id="OmniReset-G1Dex3-ObjectPartiallyAssembledEEAnywhere-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reset_states_cfg:G1Dex3ObjectPartiallyAssembledEEAnywhereResetStatesCfg"
    },
)

gym.register(
    id="OmniReset-G1Dex3-ObjectPartiallyAssembledEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reset_states_cfg:G1Dex3ObjectPartiallyAssembledEEGraspedResetStatesCfg"
    },
)

gym.register(
    id="OmniReset-G1Dex3-RelCartesianOSC-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:G1Dex3RelCartesianOSCTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:G1Dex3_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-G1Dex3-RelCartesianOSC-State-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:G1Dex3RelCartesianOSCEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:G1Dex3_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-G1Dex3-RelCartesianOSC-State-StiffPlay-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:G1Dex3RelCartesianOSCStiffEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:G1Dex3_PPORunnerCfg",
    },
)
