# 4. Training and playing a policy

A policy trains in a robot's `RelCartesianOSC-State-v0` task and is watched in the matching `State-Play-v0`
task. Training reads the `ObjectAnywhereEEAnywhere` reset set, so for the Unitree robots step 3 of
[page 3](03_datasets.md) has to exist first. This page runs a short smoke test, then a real run, and then
plays the result.

## Step 1. Smoke-test the training stage

The pipeline script has a training stage that runs from the cached assets and datasets. From the host, a
handful of iterations on a few hundred environments confirms that the task builds and the optimiser steps:

```bash
NUM_ENVS=256 MAX_ITER=20 docker/forge/run_local.sh smoke-train
ROBOT=g1_dex3 NUM_ENVS=256 MAX_ITER=20 docker/forge/run_local.sh smoke-train
```

What to expect: a TensorBoard log and a checkpoint under `logs/rsl_rl/<experiment>/<run>/`, where the
experiment is `g1d_dex1_omnireset_agent`, `g1_dex3_omnireset_agent` or `ur5e_robotiq_2f85_omnireset_agent`,
and the run is a timestamped folder.

## Step 2. Train by hand

Inside the container, the RSL-RL trainer takes the task, the number of environments and the iteration count:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task OmniReset-G1dDex1-RelCartesianOSC-State-v0 --num_envs 1024 --max_iterations 3000 \
    env.scene.insertive_object=peg env.scene.receptive_object=peghole
```

Swap the task id for `OmniReset-G1Dex3-RelCartesianOSC-State-v0` or
`OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0` for the other robots. The run is headless, and a
checkpoint is written every few hundred iterations into the run folder named above.

Training randomises the insertive object's mass at start-up, by default anywhere between 20 and 200 g. For
the big peg add `env.events.randomize_insertive_object_mass=big-peg` next to the two object choices, which
draws between 150 and 250 g around its 200 g. The pipeline and the Forge submit script add it by themselves
when `INSERTIVE=big-peg`.

For a run on the cluster, `docker/forge/README.md` section 4 has the submission flow. Forge checkpoints land
under `/mnt/robotics/<user>/omni-reset/<experiment>/<run>/`, and the download of one is described there too.

## Step 3. Play a checkpoint

The player opens the `Play` task with a window and drives it with the policy:

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task OmniReset-G1dDex1-RelCartesianOSC-State-Play-v0 --num_envs 16 --viz kit \
    --checkpoint logs/rsl_rl/g1d_dex1_omnireset_agent/<run>/model_<iteration>.pt \
    env.scene.insertive_object=peg env.scene.receptive_object=peghole
```

For the UR5e a stage 1 checkpoint plays in `...-State-Play-v0`, and a stage 2 one, fine-tuned, in
`...-State-Finetune-Play-v0`.

Three options change what you see:

- Leave out `--checkpoint` and the newest run under the experiment folder is used. `--load_run <dir>` picks
  a particular run and still takes its newest checkpoint.
- `--real-time` plays at wall-clock speed instead of as fast as the simulation goes.
- `--video --video_length 400` records an MP4 of that many steps next to the checkpoint.

What to expect: the usual minute of start-up, then the policy running in every environment. The window
accepts the camera controls of Kit but no keyboard commands. To hold a scene still and reset it by hand, use
the inspector of [page 5](05_inspection.md) instead.
