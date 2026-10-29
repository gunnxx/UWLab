# 5. Looking at a scene and finding out why it fails

Four tools cover the questions that come up while working on these environments. The inspector shows you a
scene and lets you reset it by hand. The replay tool cycles through a recorded dataset. The probe tells you
which clause of a success test a reset-state family keeps failing. And the inspector again, in a special
mode, lets you tune a grasp with the keyboard. This page takes them in that order, from the question you
would ask first to the one you ask last.

## Step 1. Look at a scene and reset it by hand

`scripts_v2/tools/inspect_env.py` opens any registered task, holds the arm still with a zero action, and
listens to the keyboard of the Kit window. It is the quickest way to look at a scene after a configuration
change: the objects, the table, what a reset draws, and how things collide.

```bash
# the RL task's reset distribution for the UR5e, four environments side by side
python scripts_v2/tools/inspect_env.py --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
    --num_envs 4 --viz kit env.scene.insertive_object=peg env.scene.receptive_object=peghole

# the same for a humanoid, swap in OmniReset-G1Dex3-... for the standing G1
python scripts_v2/tools/inspect_env.py --task OmniReset-G1dDex1-RelCartesianOSC-State-Play-v0 \
    --num_envs 4 --viz kit env.scene.insertive_object=peg env.scene.receptive_object=peghole
```

Once the window is up, these keys work:

| Key | What it does |
|---|---|
| `R` | reset every environment, which draws a new state from the reset distribution or the dataset |
| `P` | pause and resume stepping, the viewport keeps rendering so you can move the camera |
| `N` | step once while paused |
| `C` | close or open the hand |
| `G` | gravity off and on, which only sticks when the run started with `--no_gravity` |
| `T` | print the hand's pose in the object frame, which is what the grasp recorder would record |
| `H` | print the key help |

Reset-state sampling tasks are worth opening with `--no_timeout`, so that an episode never ends on its own
and `R` is the only reset. A pose can then be examined for as long as you like. The anywhere family needs no
dataset, the others need the files of [page 3](03_datasets.md):

```bash
python scripts_v2/tools/inspect_env.py --task OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEAnywhere-v0 \
    --num_envs 4 --viz kit --no_timeout
python scripts_v2/tools/inspect_env.py --task OmniReset-G1dDex1-ObjectRestingEEGrasped-v0 \
    --num_envs 4 --viz kit --no_timeout --gripper closed
```

The flags you will use most:

- `--gripper {keep,open,closed}` decides what the hand is told while holding. The default keeps whatever
  state the reset left it in.
- `--no_gravity` starts with gravity off and without the force and torque pokes the task applies to the
  object, so a pose can be examined undisturbed. `G` then switches gravity on when you want to test a hold.
- `--episode_length_s` overrides the episode length, and `--auto_reset_s N` also resets every N seconds of
  wall clock.
- `--max_steps` and `--print_every` make headless smoke checks possible, as in the example below.
- `--render_substeps` renders every physics substep as the task configurations do, twelve frames per step,
  which is slower but shows the motion within a step.

When the scene starts, the log reports the object's mass per body and the friction of its colliders, and the
periodic line reports the object's top speed, which is a quick way to see whether something is still moving.

A headless check that exits by itself, useful after a configuration change to confirm the task still builds:

```bash
python scripts_v2/tools/inspect_env.py --task OmniReset-G1dDex1-ObjectAnywhereEEAnywhere-v0 \
    --num_envs 2 --viz none --max_steps 100 --auto_reset_s 2 --no_timeout
```

## Step 2. Replay a recorded dataset

`scripts_v2/tools/visualize_reset_states.py` opens a task, cycles through the states of a recorded family one
reset at a time, and prints the success flag stored with each state:

```bash
python scripts_v2/tools/visualize_reset_states.py --task OmniReset-G1dDex1-RelCartesianOSC-State-Play-v0 \
    --dataset_dir source/uwlab_assets/data/Datasets/OmniReset_g1d --reset_type ObjectAnywhereEEAnywhere \
    --num_envs 4 --reset_interval 2 --viz kit env.scene.insertive_object=peg env.scene.receptive_object=peghole
```

Leave out `--reset_type` to load all four families in turn. `--reset_interval` is the time between resets in
seconds. This is the tool to reach for when a trained policy behaves oddly from the start, since it shows you
the exact states the policy is being reset into.

## Step 3. Probe a family that records nothing

When `record_reset_states.py` runs for a long time and reports no successes, the question is which clause of
the success test is failing. `scripts_v2/tools/debug_reset_env.py` runs the family and logs, every few steps,
how many environments pass each clause: the gripper state, the approach orientation, the stability, the
freedom from collisions and the pose deviation. With `--funnel` it accumulates those counts over every episode,
which gives you the yield of the family and the clause that costs the most:

```bash
python scripts_v2/tools/debug_reset_env.py --task OmniReset-G1dDex1-ObjectAnywhereEEGrasped-v0 \
    --num_envs 256 --steps 45 --funnel --viz none env.scene.insertive_object=peg env.scene.receptive_object=peghole
```

For a grasped family add `--close_gripper`, which commands the hand closed the way the recorder does. The
other flags isolate a cause one at a time: `--zero_gains` turns the controller off, `--no_self_collisions`
and `--no_table` remove two sources of contact, `--no_ik` skips the end-effector reset event, `--kin_check`
compares the analytical kinematics with PhysX, `--mass_check` prints the link masses, and `--arm_damping`,
`--armature`, `--damping_ratio` and `--stiffness_scale` try other drive settings without editing the asset.

What to expect: a table of pass counts per clause at the end. If the orientation or the collision clause
removes almost everything, the table layout or the grasp set is the usual culprit, and
`docker/forge/README.md` has a worked example under "Checking why a reset set yields nothing".

## Step 4. Tune a grasp by hand

The grasp sampler of the Dex3 starts from grasps written down in its configuration, and those were found
with the inspector. The grasp-sampling task takes a `fixed_grasp` parameter. When it is enabled, the sampler
produces that one pose instead of its usual candidates, and the inspector lets you turn it with the keyboard.

```bash
python scripts_v2/tools/inspect_env.py --task OmniReset-Dex3-GraspSampling-v0 --num_envs 1 --viz kit \
    --gripper open --no_timeout --no_gravity env.scene.object=peg \
    env.events.grasp_sampling.params.visualize_grasps=true \
    env.events.grasp_sampling.params.fixed_grasp.enabled=true \
    env.events.grasp_sampling.params.fixed_grasp.center=[0,0,0.015] \
    env.events.grasp_sampling.params.fixed_grasp.approach=[0,0,-1] \
    env.events.grasp_sampling.params.fixed_grasp.rpy_deg=[0,0,0]
```

This opens one environment with the hand open, no gravity, a frame marker at the palm, and the palm
approaching the standing peg from above with the pinch point 1.5 cm below the top. Then:

1. Turn the grasp with `J` and `L` for roll, `I` and `K` for pitch, and `U` and `O` for yaw. Each press turns
   by `--angle_step` degrees, five by default, about the object's own axes, and resets the scene at once.
   Hold SHIFT for a fifth of the step. `0` restores the starting values.
2. Press `C` to close the hand and watch where the fingers land.
3. Press `G` to switch gravity on and see whether the hold survives.
4. Press `T` to print the hand pose in the object frame. That printed pose, together with the roll, pitch
   and yaw you ended on, is what goes into the configuration as a nominal grasp.

Once a nominal grasp is written into the task configuration, the recorder of [page 3](03_datasets.md) scatters
its candidates around it. [How the grasp sampler works](../omnireset_grasp_sampling.html) explains the spec,
the scatter and the reasons behind them.
