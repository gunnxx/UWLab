# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OmniReset tasks for the Unitree G1-D (right arm) with a Dex1-1 gripper.

Pipeline (see ``docker/forge/submit_g1d.sh``):
  1. ``OmniReset-PartialAssemblies-v0`` (robot-independent, from the UR5e package) -> partial assemblies
  2. ``OmniReset-Dex1-GraspSampling-v0`` -> grasps for the Dex1-1
  3. ``OmniReset-G1dDex1-<ResetType>-v0`` x4 -> reset-state datasets
  4. ``OmniReset-G1dDex1-RelCartesianOSC-State-v0`` -> stage 1 PPO
"""

import gymnasium as gym

from . import agents

gym.register(
    id="OmniReset-Dex1-GraspSampling-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.grasp_sampling_cfg:Dex1GraspSamplingCfg"},
    disable_env_checker=True,
)

gym.register(
    id="OmniReset-G1dDex1-ObjectAnywhereEEAnywhere-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:G1dObjectAnywhereEEAnywhereResetStatesCfg"},
)

gym.register(
    id="OmniReset-G1dDex1-ObjectRestingEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:G1dObjectRestingEEGraspedResetStatesCfg"},
)

gym.register(
    id="OmniReset-G1dDex1-ObjectAnywhereEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:G1dObjectAnywhereEEGraspedResetStatesCfg"},
)

gym.register(
    id="OmniReset-G1dDex1-ObjectPartiallyAssembledEEAnywhere-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:G1dObjectPartiallyAssembledEEAnywhereResetStatesCfg"},
)

gym.register(
    id="OmniReset-G1dDex1-ObjectPartiallyAssembledEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:G1dObjectPartiallyAssembledEEGraspedResetStatesCfg"},
)

gym.register(
    id="OmniReset-G1dDex1-RelCartesianOSC-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:G1dDex1RelCartesianOSCTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:G1dDex1_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-G1dDex1-RelCartesianOSC-State-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:G1dDex1RelCartesianOSCEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:G1dDex1_PPORunnerCfg",
    },
)

gym.register(
    id="OmniReset-G1dDex1-RelCartesianOSC-State-StiffPlay-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:G1dDex1RelCartesianOSCStiffEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:G1dDex1_PPORunnerCfg",
    },
)
