# 2. Which environment is which

OmniReset registers a few dozen environments. They are easier to navigate once you see that there are three
robots, and that each robot has the same three families of tasks. This page gives you that map, and then says
which dataset files each environment reads when it resets, because that is what decides which recording steps
of [page 3](03_datasets.md) you need.

## Step 1. List what is registered

```bash
python scripts/environments/list_envs.py --keyword OmniReset
python scripts/environments/list_envs.py --keyword G1dDex1
python scripts/environments/list_envs.py --keyword G1Dex3
```

The first command lists everything. The other two narrow the list to one robot.

## Step 2. The three robots

- **UR5e + Robotiq 2F-85.** The fixed arm with the two-finger gripper. Its datasets are downloaded from the
  UW Lab cloud assets, so it works out of the box.
- **G1-D + Dex1.** The wheeled Unitree G1-D with the Dex1-1 gripper. Its datasets are recorded locally.
- **G1 + Dex3.** The standing 29 degree-of-freedom Unitree G1 with the Dex3 hand. The legs, waist, left arm
  and left hand are frozen in the robot's teleoperation stance, with the pelvis pitched 0.078 rad as measured
  on the robot, and the right arm and right hand move. The URDF is the robot's own gravity-solver model,
  vendored under `source/uwlab_assets/uwlab_assets/robots/unitree_g1_dex3/`, so the simulated link masses
  are the solver's. The hand approaches along its palm normal and pinches between the thumb and the two
  fingertips. Its datasets are recorded locally too.

## Step 3. The three families

Every robot has the same three kinds of task, which you can tell apart from the task id.

**Reinforcement learning tasks** are the ones a policy trains and plays in. Their ids contain
`RelCartesianOSC` and end in `State-v0` for training or `State-Play-v0` for playing. The UR5e also has RGB
variants and fine-tuning variants.

**Reset-state sampling tasks** are not trained in. Each one samples a kind of starting situation, lets it
settle, and keeps the ones that pass a success test. The recorder of page 3 runs them to produce the reset
datasets. Their ids name the situation, for example `ObjectRestingEEGrasped`, which means the object rests on
the table and the end effector already holds it.

**Tooling tasks** serve one recorder or one check. The grasp-sampling tasks move a bare hand around an object
to find grasps that hold. The partial-assemblies task slides the peg into the hole to collect half-inserted
poses.

| Robot | Family | Task ids |
|---|---|---|
| UR5e + Robotiq 2F-85 | RL, state | `OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0`, `...-State-Play-v0`, `...-State-Finetune-v0`, `...-State-Finetune-Play-v0` |
| UR5e + Robotiq 2F-85 | RL, RGB | `OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-Play-v0`, `...-RGB-DataCollection-v0`, `...-RGB-OOD-*` |
| UR5e + Robotiq 2F-85 | reset-state sampling | `OmniReset-UR5eRobotiq2f85-<situation>-v0`, note the capital `UR5e` |
| UR5e + Robotiq 2F-85 | tooling | `OmniReset-Robotiq2f85-GraspSampling-v0`, `OmniReset-PartialAssemblies-v0`, `OmniReset-Ur5eRobotiq2f85-Sysid-v0`, `OmniReset-Ur5eRobotiq2f85-CameraAlign-v0` |
| G1-D + Dex1 | RL, state | `OmniReset-G1dDex1-RelCartesianOSC-State-v0`, `...-State-Play-v0`, `...-State-StiffPlay-v0` |
| G1-D + Dex1 | reset-state sampling | `OmniReset-G1dDex1-<situation>-v0` |
| G1-D + Dex1 | tooling | `OmniReset-Dex1-GraspSampling-v0` |
| G1 + Dex3 | RL, state | `OmniReset-G1Dex3-RelCartesianOSC-State-v0`, `...-State-Play-v0`, `...-State-StiffPlay-v0` |
| G1 + Dex3 | reset-state sampling | `OmniReset-G1Dex3-<situation>-v0` |
| G1 + Dex3 | tooling | `OmniReset-Dex3-GraspSampling-v0` |

The five situations are `ObjectAnywhereEEAnywhere`, `ObjectRestingEEGrasped`, `ObjectAnywhereEEGrasped`,
`ObjectPartiallyAssembledEEAnywhere` and `ObjectPartiallyAssembledEEGrasped`.

## Step 4. What each environment reads at reset

Most environments do not sample their starting state from scratch. They draw it from a recorded dataset, and
the file they need depends on the family and the situation.

| Environment | Dataset files it reads |
|---|---|
| RL tasks, `...-State-*` | `Resets/<pair>/resets_ObjectAnywhereEEAnywhere.pt` |
| `ObjectAnywhereEEAnywhere` | none, it samples everything itself |
| `ObjectAnywhereEEGrasped` | `Grasps/<object>/grasps.pt` |
| `ObjectRestingEEGrasped` | `Resets/<pair>/resets_ObjectAnywhereEEAnywhere.pt` and `Grasps/<object>/grasps.pt` |
| `ObjectPartiallyAssembled*` | `Resets/<pair>/partial_assemblies.pt`, plus `grasps.pt` for the grasped one |

Here `<pair>` is the sorted pair of USD folder names, for example `Peg__PegHole`, and `<object>` is the
insertive object's folder name, for example `Peg`. A scaled object variant gets its own folders, so the big
peg records to `Grasps/Peg_x2`, and the big peg in the big hole to `Resets/Peg_x2__PegHole_x2`. The pair is
chosen with `env.scene.insertive_object=big-peg env.scene.receptive_object=big-peghole` in every task that
takes the pair, and the hole has to be scaled with the peg for the two to fit.

The files live under one dataset root per robot.

| Robot | Dataset root |
|---|---|
| UR5e + Robotiq 2F-85 | `Datasets/OmniReset_isaaclab3` in the cloud assets, downloaded on first use |
| G1-D + Dex1 | `source/uwlab_assets/data/Datasets/OmniReset_g1d`, recorded by page 3 |
| G1 + Dex3 | `source/uwlab_assets/data/Datasets/OmniReset_g1_dex3`, recorded by page 3 |

## One thing to remember about recorded states

A recorded reset state restores every rigid object in the scene that is not kinematic. The table is kinematic
and is skipped, so you can move the table without invalidating a dataset. Changing an object, its scale or its
physics, or changing the robot, does invalidate it, and the affected sets have to be recorded again.
