# 3. Recording the datasets

The UR5e environments download their datasets. The two Unitree robots do not, because their datasets depend
on the robot's own hand and reach, so you record them yourself. This page takes you through that recording
in the order the steps depend on each other. You can run every step by hand, which is the best way to learn
what each one does, or hand the whole chain to the pipeline script, which is described at the end.

## How the steps depend on each other

There are seven steps. The first builds the robot assets, and the remaining six each write one dataset file.
Later steps read what earlier steps wrote, which fixes the order.

| Step | What it records | Writes | Needs |
|---|---|---|---|
| 0 | the robot's URDF converted to USD | `data/Robots/Unitree/<robot>/*.usd` | the `unitree_ros` meshes |
| 1 | half-inserted poses of the peg in the hole | `Resets/<pair>/partial_assemblies.pt` | 0 |
| 2 | hand poses on the object that hold when the hand closes | `Grasps/<object>/grasps.pt` | 0 |
| 3 | object anywhere, end effector anywhere | `Resets/<pair>/resets_ObjectAnywhereEEAnywhere.pt` | 0 |
| 4 | object anywhere, end effector holding it | `Resets/<pair>/resets_ObjectAnywhereEEGrasped.pt` | 2 |
| 5 | object resting on the table, end effector holding it | `Resets/<pair>/resets_ObjectRestingEEGrasped.pt` | 2 and 3 |
| 6 | object half inserted, end effector holding it | `Resets/<pair>/resets_ObjectPartiallyAssembledEEGrasped.pt` | 1 and 2 |

Steps 3 to 6 all use the same recorder. Each one samples its situation thousands of times, lets the scene
settle, keeps the states whose success test holds, and writes them to its file. The three grasped families
need the grasps of step 2 to put the hand on the object, and they are reach-limited by the table layout, so
they succeed far less often than step 3. [Page 5](05_inspection.md) shows how to measure a family's yield on
a small run before you commit to a long one.

## Before you start

Open a shell in the container, as in [page 1](01_getting_in.md), and pick the robot. The commands below use
four shell variables so that the same commands serve both robots.

For the G1-D + Dex1:

```bash
ROBOT_ASSETS=source/uwlab_assets/data/Robots/Unitree/G1dDex1
DATA=source/uwlab_assets/data/Datasets/OmniReset_g1d
TASK=OmniReset-G1dDex1
GRASP_TASK=OmniReset-Dex1-GraspSampling-v0
GRIPPER_BODY=right_dex1_base_link
```

For the G1 + Dex3:

```bash
ROBOT_ASSETS=source/uwlab_assets/data/Robots/Unitree/G1Dex3
DATA=source/uwlab_assets/data/Datasets/OmniReset_g1_dex3
TASK=OmniReset-G1Dex3
GRASP_TASK=OmniReset-Dex3-GraspSampling-v0
GRIPPER_BODY=right_hand_palm_link
```

And for both, the object pair:

```bash
OBJ="env.scene.insertive_object=peg env.scene.receptive_object=peghole"
```

For the big peg, the peg at twice the size made for the Dex3, the pair is `big-peg` and `big-peghole`, and the
grasp step below takes `env.scene.object=big-peg`. Every file then lands under `Peg_x2__PegHole_x2` and
`Peg_x2` instead of `Peg__PegHole` and `Peg`, so the two sizes never overwrite each other. The big peg is
authored at 200 g, the weight of the printed one, and training draws its mass from a range around that, see
[page 4](04_training.md).

The sizes in the commands are the ones the recorded sets were made with. Smaller numbers are fine for a first
pass, and the pipeline's smoke sizes at the end of this page finish in minutes.

## Step 0. Build the robot assets

This step runs once per machine, not once per dataset. The converted assets are gitignored, so a fresh clone does not have them, but once they exist they stay valid until the builder or the URDF changes. If the two USD files named below are already in the assets folder, together with a `metadata.yaml`, skip to step 1.

Both robots are assembled from the Unitree ROS description, so clone it once:

```bash
git clone --depth 1 https://github.com/unitreerobotics/unitree_ros.git /workspace/unitree_ros
```

For the G1-D + Dex1, the builder writes two URDFs, the right arm and the bare gripper, and each is converted
to USD inside a Kit application:

```bash
python -m uwlab_assets.robots.unitree_g1d_dex1.build_urdf --unitree-ros /workspace/unitree_ros --out-dir $ROBOT_ASSETS
python scripts_v2/tools/convert_urdf_kit.py $ROBOT_ASSETS/g1d_dex1_right_arm.urdf $ROBOT_ASSETS/g1d_dex1_right_arm.usd \
    --fix_base --joint_stiffness 0 --joint_damping 0 --viz none
python scripts_v2/tools/convert_urdf_kit.py $ROBOT_ASSETS/dex1_gripper.urdf $ROBOT_ASSETS/dex1_gripper.usd \
    --fix_base --joint_stiffness 0 --joint_damping 0 --viz none
```

For the G1 + Dex3, the builder reads the vendored robot URDF and only needs the meshes from the clone. After
the conversion, a link-by-link check confirms that the USD carries the URDF's masses, centres of mass and
inertias, which matters because those are the gravity solver's numbers:

```bash
python -m uwlab_assets.robots.unitree_g1_dex3.build_urdf --meshes /workspace/unitree_ros/robots/g1_description/meshes --out-dir $ROBOT_ASSETS
python scripts_v2/tools/convert_urdf_kit.py $ROBOT_ASSETS/g1_dex3_right_arm.urdf $ROBOT_ASSETS/g1_dex3_right_arm.usd \
    --fix_base --joint_stiffness 0 --joint_damping 0 --viz none
python scripts_v2/tools/convert_urdf_kit.py $ROBOT_ASSETS/dex3_hand.urdf $ROBOT_ASSETS/dex3_hand.usd \
    --fix_base --joint_stiffness 0 --joint_damping 0 --viz none
python scripts_v2/tools/check_usd_inertials.py $ROBOT_ASSETS/g1_dex3_right_arm.urdf $ROBOT_ASSETS/g1_dex3_right_arm.usd
```

What to expect: two USD files per robot in the assets folder, plus a `metadata.yaml` next to the gripper USD
that the grasp sampler reads. The inertial check prints one line per link and exits with an error if anything
differs beyond tolerance.

## Step 1. Record the partial assemblies

This step runs physics trajectories of the peg sliding into the hole and keeps the distinct poses it passes
through. It uses the UR5e tooling task, since no robot is in the scene, and it is the same for both robots.

```bash
python scripts_v2/tools/record_partial_assemblies.py --task OmniReset-PartialAssemblies-v0 \
    --num_envs 10 --num_trajectories 10 --dataset_dir $DATA --viz none $OBJ
```

What to expect: `Resets/Peg__PegHole/partial_assemblies.pt` under the dataset root.

## Step 2. Record the grasps

The grasp sampler drops the arm and moves the bare hand around the object. It places the open hand at a
candidate pose, closes it, lets physics settle, pokes the object, and keeps the poses where the object is
still held. Each kept grasp is the pose of the gripper body in the object frame, plus the finger joint
positions, which is why the body name is a flag.

```bash
python scripts_v2/tools/record_grasps.py --task $GRASP_TASK \
    --num_envs 8192 --num_grasps 1000 --dataset_dir $DATA --gripper_body_name $GRIPPER_BODY \
    --viz none env.scene.object=peg
```

Both hands sample their candidates from the object mesh with the antipodal sampler. The Dex1 lays its approach
along the sampled pinch line, as the Robotiq does. The Dex3 lays an axis 45 degrees from its squeeze axis along
the line, so that the squeeze, from thumb to fingertips, crosses the line at 45 degrees and takes a square peg
across its diagonal, and it sweeps the hand around the line, so the 45 degree palm tilt of the hand-tuned grasp
is among the candidates. Its standoffs stay within the pocket's depth. These choices sit in
`config/g1_dex3/grasp_sampling_cfg.py`, next to the two hand-written grasps kept for the inspector.
[The antipodal sampler with the squeeze axis](../omnireset_antipodal_sampler.html) explains the mechanism and
where the hand-tuned 45 degree tilt sits in it, and [How the grasp sampler works](../omnireset_grasp_sampling.html)
explains the sampler's mechanics and the earlier nominal-grasp source.

The scene's object is a Hydra choice. Besides `peg` there are `fbleg`, `fbdrawerbottom` and `big-peg`, which
is the peg at twice the size with an authored mass of 150 g and a friction of 0.6, made for the Dex3. A
scaled variant records to its own folder, so the big peg writes `Grasps/Peg_x2/grasps.pt`.

What to expect: `Grasps/Peg/grasps.pt` under the dataset root. The recorder writes the file at most once a
minute, so a run killed early keeps what it had.

## Steps 3 to 6. Record the reset-state families

One recorder serves the four families. Run them in this order, because the resting family needs the anywhere
family's file and all three grasped families need the grasps:

```bash
for RT in ObjectAnywhereEEAnywhere ObjectAnywhereEEGrasped ObjectRestingEEGrasped ObjectPartiallyAssembledEEGrasped; do
    python scripts_v2/tools/record_reset_states.py --task ${TASK}-${RT}-v0 \
        --num_envs 2048 --num_reset_states 10000 --dataset_dir $DATA --viz none $OBJ
done
```

Setting `--num_reset_states 0` records until you stop the process. Each recorder starts its output file over,
so a family can only be made larger by recording more states in one run, not by running twice.

What to expect: one `resets_<situation>.pt` per family under `Resets/Peg__PegHole/`. The anywhere family
fills quickly. The grasped families can take hours at these sizes, which is why the pipeline below records
them one per job.

## All at once with the pipeline

The script `docker/forge/g1d_pipeline.sh` runs steps 0 to 6 and then a training stage, and it caches each
stage so that a later run can skip it. The cache lives under `~/.cache/uwlab-local/robotics/<user>/omni-reset/<robot>`
locally and under `/mnt/robotics/<user>/omni-reset/<robot>` on Forge. You start it from the host through
`run_local.sh`, and `ROBOT` picks the robot:

```bash
# smoke-test sizes, a few minutes: assets, all datasets, then five PPO iterations
docker/forge/run_local.sh smoke
ROBOT=g1_dex3 docker/forge/run_local.sh smoke
```

The knobs are environment variables. These are the ones you will reach for:

| Knob | Meaning |
|---|---|
| `STAGES=assets,datasets,train` | which stages to run, any subset in this order |
| `INSERTIVE=big-peg RECEPTIVE=big-peghole` | the object pair, `peg` and `peghole` by default |
| `FORCE=1` | rebuild or re-record a stage even if the cache already has its files |
| `RESET_ENVS`, `NUM_RESETS` | environments and states per reset-state family, 2048 and 10000 by default |
| `GRASP_ENVS`, `NUM_GRASPS` | environments and grasps for step 2, 8192 and 1000 by default |
| `RESET_TYPES="A B"` | record only these families |
| `AUX_FROM_CACHE=1` | take the partial assemblies, the grasps and earlier families from the cache, for one family per job |
| `RESET_SYNC_S=300` | while recording, mirror the files to the cache this often, so a killed run keeps what it had |

Two runs you will want:

```bash
# the real sizes, only the dataset stages, re-recording whatever the cache already has
FORCE=1 STAGES=assets,datasets RESET_ENVS=16384 NUM_RESETS=10000 GRASP_ENVS=8192 NUM_GRASPS=1000 docker/forge/run_local.sh smoke

# one family on its own, everything it needs taken from the cache
AUX_FROM_CACHE=1 RESET_TYPES=ObjectRestingEEGrasped FORCE=1 STAGES=datasets docker/forge/run_local.sh smoke
```

`docker/forge/README.md` section 3 shows the same runs submitted as Forge jobs.

## When do I have to record again?

A recorded state restores every rigid object in the scene that is not kinematic. So:

- Moving the table does not invalidate anything. The table is kinematic and is skipped.
- Changing an object, its scale or its physics invalidates every set that contains it. Re-record steps 1 to 6.
- Changing the robot or its hand invalidates the grasps and the three grasped families. Re-record steps 2,
  4, 5 and 6. The anywhere family and the partial assemblies survive.
- Changing only the hand's finger gains changes which grasps hold. Re-record step 2 and then the grasped
  families.

[Page 5](05_inspection.md) shows how to replay a finished set and how to probe a family that records nothing.
