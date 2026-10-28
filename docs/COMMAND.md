# Command reference: inspecting and running the OmniReset environments

Every command below runs **inside the Forge image** (`uwlab-omnireset:local`), where `python` is Isaac
Sim's `python.sh`. Two equivalent ways to get there:

```bash
docker/forge/run_local.sh                      # interactive shell in the container (repo at /workspace/uwlab)
docker/forge/run_local.sh python <script> ...  # one command
# or: VS Code -> F1 -> "Dev Containers: Reopen in Container", then use the integrated terminal
```

Windowed commands (`--viz kit`) need the X display: `run_local.sh` forwards `DISPLAY` automatically, the
dev container too; run `xhost +si:localuser:root` once per login on the host. Without `--viz kit` Isaac
Lab 3.0 runs headless (`--viz none` is the default). Run one Kit window at a time on a 16 GB GPU.

The object pair is a Hydra override appended to any command:
`env.scene.insertive_object=peg env.scene.receptive_object=peghole`.

## 1. List the registered environments

```bash
python scripts/environments/list_envs.py --keyword OmniReset
python scripts/environments/list_envs.py --keyword G1dDex1
```

| Robot | Env family | Task ids |
|---|---|---|
| UR5e + Robotiq 2F-85 | RL (state) | `OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0`, `...-State-Play-v0`, `...-State-Finetune-v0`, `...-State-Finetune-Play-v0` |
| UR5e + Robotiq 2F-85 | RL (RGB) | `OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-RGB-Play-v0`, `...-RGB-DataCollection-v0`, `...-RGB-OOD-*` |
| UR5e + Robotiq 2F-85 | reset-state sampling | `OmniReset-UR5eRobotiq2f85-{ObjectAnywhereEEAnywhere,ObjectRestingEEGrasped,ObjectAnywhereEEGrasped,ObjectPartiallyAssembledEEAnywhere,ObjectPartiallyAssembledEEGrasped}-v0` (note the `UR5e` capitalisation) |
| UR5e + Robotiq 2F-85 | tooling | `OmniReset-Robotiq2f85-GraspSampling-v0`, `OmniReset-PartialAssemblies-v0`, `OmniReset-Ur5eRobotiq2f85-Sysid-v0`, `OmniReset-Ur5eRobotiq2f85-CameraAlign-v0` |
| G1-D + Dex1 | RL (state) | `OmniReset-G1dDex1-RelCartesianOSC-State-v0`, `...-State-Play-v0`, `...-State-StiffPlay-v0` |
| G1-D + Dex1 | reset-state sampling | `OmniReset-G1dDex1-{ObjectAnywhereEEAnywhere,ObjectRestingEEGrasped,ObjectAnywhereEEGrasped,ObjectPartiallyAssembledEEAnywhere,ObjectPartiallyAssembledEEGrasped}-v0` |
| G1-D + Dex1 | tooling | `OmniReset-Dex1-GraspSampling-v0` |

Most envs reset from recorded datasets. The UR5e ones download from the UW Lab cloud assets
(`Datasets/OmniReset_isaaclab3`); the G1-D ones read `source/uwlab_assets/data/Datasets/OmniReset_g1d`, which
section 2 generates.

| Env | Dataset files |
|---|---|
| RL (`...-State-*`) | `Resets/<pair>/resets_ObjectAnywhereEEAnywhere.pt` |
| `ObjectAnywhereEEAnywhere` | none (samples everything itself) |
| `ObjectRestingEEGrasped` | `Resets/<pair>/resets_ObjectAnywhereEEAnywhere.pt`, `Grasps/<object>/grasps.pt` |
| `ObjectAnywhereEEGrasped` | `Grasps/<object>/grasps.pt` |
| `ObjectPartiallyAssembled*` | `Resets/<pair>/partial_assemblies.pt` (+ `grasps.pt` for `EEGrasped`) |

Note: a recorded reset state restores every non-kinematic rigid object in the scene. The table is kinematic and
is skipped, so moving it does not invalidate the sets; changing the objects or the robot does.

## 2. Generate the G1-D datasets

The G1-D sets are robot-specific and gitignored, so they are recorded, not downloaded. Four recorders write
into one `--dataset_dir`; each later step reads what the earlier ones wrote, so the order matters:

| Step | Script / task | Writes | Needs |
|---|---|---|---|
| 0 | `build_urdf.py` + `convert_urdf_kit.py` (assets stage) | `data/Robots/Unitree/G1dDex1/*.usd` | `unitree_ros` checkout |
| 1 | `record_partial_assemblies.py`, `OmniReset-PartialAssemblies-v0` | `Resets/<pair>/partial_assemblies.pt` | 0 |
| 2 | `record_grasps.py`, `OmniReset-Dex1-GraspSampling-v0` | `Grasps/<object>/grasps.pt` | 0 |
| 3 | `record_reset_states.py`, `...-ObjectAnywhereEEAnywhere-v0` | `Resets/<pair>/resets_ObjectAnywhereEEAnywhere.pt` | 0 |
| 4 | `record_reset_states.py`, `...-ObjectAnywhereEEGrasped-v0` | `Resets/<pair>/resets_ObjectAnywhereEEGrasped.pt` | 2 |
| 5 | `record_reset_states.py`, `...-ObjectRestingEEGrasped-v0` | `Resets/<pair>/resets_ObjectRestingEEGrasped.pt` | 2, 3 |
| 6 | `record_reset_states.py`, `...-ObjectPartiallyAssembledEEGrasped-v0` | `Resets/<pair>/resets_ObjectPartiallyAssembledEEGrasped.pt` | 1, 2 |

`<pair>` is the sorted pair of USD folder names (`Peg__PegHole`), `<object>` the insertive object's (`Peg`).
The grasp sampler drops the arm and moves the bare Dex1-1 gripper around the object, so step 2 runs on the
gripper USD alone and records the pose of `right_dex1_base_link` relative to the object. Steps 3 to 6 sample
poses, let the scene settle and keep the states whose success term holds; the grasped families are
reach-limited by the table layout (see section 5 to measure a family's yield before a long run).

**All at once.** `docker/forge/g1d_pipeline.sh` runs steps 0 to 6 in this order and caches each stage under
`${CACHE}` (`~/.cache/uwlab-local/robotics/twguntara/omni-reset/g1d` locally, `/mnt/robotics/twguntara/omni-reset/g1d` on
Forge). From the host:

```bash
# smoke-test sizes, a few minutes: assets -> all datasets -> 5 PPO iterations
docker/forge/run_local.sh smoke
# the real sizes, only the dataset stages; FORCE=1 re-records even if the cache already has the files
FORCE=1 STAGES=assets,datasets RESET_ENVS=16384 NUM_RESETS=10000 GRASP_ENVS=8192 NUM_GRASPS=1000 docker/forge/run_local.sh smoke
# one family per job/GPU: partial assemblies, grasps and the other families come from the cache
AUX_FROM_CACHE=1 RESET_TYPES=ObjectRestingEEGrasped FORCE=1 STAGES=datasets docker/forge/run_local.sh smoke
```

The reset-state files are mirrored to the cache every `RESET_SYNC_S` (300) seconds while recording, so a
killed run keeps what it had. `docker/forge/README.md` section 3 has the same runs as Forge jobs.

**Step by step**, inside the container (`DATA=source/uwlab_assets/data/Datasets/OmniReset_g1d`,
`OBJ="env.scene.insertive_object=peg env.scene.receptive_object=peghole"`). Sizes are the pipeline's defaults;
the ones used for the recorded sets are in the comments:

```bash
# 0. assets: G1-D right arm + Dex1-1 URDFs from unitree_ros, converted to USD inside a Kit app
git clone --depth 1 https://github.com/unitreerobotics/unitree_ros.git /workspace/unitree_ros
ASSETS=source/uwlab_assets/data/Robots/Unitree/G1dDex1
python -m uwlab_assets.robots.unitree_g1d_dex1.build_urdf --unitree-ros /workspace/unitree_ros --out-dir $ASSETS
python scripts_v2/tools/convert_urdf_kit.py $ASSETS/g1d_dex1_right_arm.urdf $ASSETS/g1d_dex1_right_arm.usd \
    --fix_base --joint_stiffness 0 --joint_damping 0 --viz none
python scripts_v2/tools/convert_urdf_kit.py $ASSETS/dex1_gripper.urdf $ASSETS/dex1_gripper.usd \
    --fix_base --joint_stiffness 0 --joint_damping 0 --viz none

# 1. partial assemblies: physics trajectories of the peg sliding into the hole, de-duplicated poses
python scripts_v2/tools/record_partial_assemblies.py --task OmniReset-PartialAssemblies-v0 \
    --num_envs 10 --num_trajectories 10 --dataset_dir $DATA --viz none $OBJ

# 2. grasps: Dex1-1 gripper poses on the peg that hold when the fingers close (recorded set: 8192 envs, 1000 grasps)
python scripts_v2/tools/record_grasps.py --task OmniReset-Dex1-GraspSampling-v0 \
    --num_envs 8192 --num_grasps 1000 --dataset_dir $DATA --gripper_body_name right_dex1_base_link \
    --viz none env.scene.object=peg

# 3-6. reset states, one family per run (recorded sets: 16384 envs, 10000 states each; --num_reset_states 0 = until killed)
for RT in ObjectAnywhereEEAnywhere ObjectAnywhereEEGrasped ObjectRestingEEGrasped ObjectPartiallyAssembledEEGrasped; do
    python scripts_v2/tools/record_reset_states.py --task OmniReset-G1dDex1-${RT}-v0 \
        --num_envs 2048 --num_reset_states 10000 --dataset_dir $DATA --viz none $OBJ
done
```

Each recorder starts its output file over, so a family can only be extended by recording more states in one
run, not by running twice. A state restores every non-kinematic rigid object, so the reset sets must be
re-recorded after the objects or the robot change but not after the table moves. Section 4 replays a
finished set.

## 3. Inspect a scene and reset it from the keyboard

`scripts_v2/tools/inspect_env.py` opens any registered task, holds the arm still (zero OSC action) and
listens to the Kit window:

| Key | Action |
|---|---|
| `R` | reset all environments (a new draw from the reset distribution / reset-state dataset) |
| `P` | pause / resume stepping (the viewport keeps rendering, so the camera can be moved) |
| `N` | step once while paused |
| `H` | print the key help |

```bash
# UR5e + Robotiq: the RL task's reset distribution, 4 envs side by side
python scripts_v2/tools/inspect_env.py --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 \
    --num_envs 4 --viz kit env.scene.insertive_object=peg env.scene.receptive_object=peghole

# G1-D + Dex1: same for the humanoid
python scripts_v2/tools/inspect_env.py --task OmniReset-G1dDex1-RelCartesianOSC-State-Play-v0 \
    --num_envs 4 --viz kit env.scene.insertive_object=peg env.scene.receptive_object=peghole

# Reset-state sampling envs. --no_timeout: episodes never end on their own, so R is the only reset
# and a pose can be examined for as long as needed. ObjectAnywhereEEAnywhere needs no dataset.
python scripts_v2/tools/inspect_env.py --task OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEAnywhere-v0 \
    --num_envs 4 --viz kit --no_timeout
python scripts_v2/tools/inspect_env.py --task OmniReset-G1dDex1-ObjectRestingEEGrasped-v0 \
    --num_envs 4 --viz kit --no_timeout --gripper closed
```

Flags: `--gripper {keep,open,closed}` (default `keep`: command whatever state the reset left the gripper
in), `--episode_length_s`, `--auto_reset_s N` (also reset every N seconds of wall clock), `--max_steps`
and `--print_every` (headless smoke checks), plus every AppLauncher flag (`--viz`, `--device`,
`--enable_cameras`). The script renders one frame per env step; the task configs render every physics
substep (12 frames per step), which `--render_substeps` restores if the motion within a step matters.
Expect about a minute of startup before the window shows the scene (longer on the first run while
the shader cache fills), then a few env steps per second with a window. Headless example that exits
by itself:

```bash
python scripts_v2/tools/inspect_env.py --task OmniReset-G1dDex1-ObjectAnywhereEEAnywhere-v0 \
    --num_envs 2 --viz none --max_steps 100 --auto_reset_s 2 --no_timeout
```

## 4. Replay recorded reset states

Cycles through a saved reset-state dataset (one reset every `--reset_interval` seconds) and prints the
success flag of each loaded state:

```bash
python scripts_v2/tools/visualize_reset_states.py --task OmniReset-G1dDex1-RelCartesianOSC-State-Play-v0 \
    --dataset_dir source/uwlab_assets/data/Datasets/OmniReset_g1d --reset_type ObjectAnywhereEEAnywhere \
    --num_envs 4 --reset_interval 2 --viz kit env.scene.insertive_object=peg env.scene.receptive_object=peghole
```

## 5. Probe why a reset-state set yields nothing

Per-step breakdown of the success term's clauses (gripper, approach orientation, stability, collisions,
pose deviation); `--funnel` accumulates the pass counts over all episodes:

```bash
python scripts_v2/tools/debug_reset_env.py --task OmniReset-G1dDex1-ObjectAnywhereEEGrasped-v0 \
    --num_envs 256 --steps 45 --funnel --viz none env.scene.insertive_object=peg env.scene.receptive_object=peghole
```

## 6. Play a trained policy

```bash
# UR5e + Robotiq (Stage 1 checkpoint; use ...-State-Finetune-Play-v0 for a Stage 2 one)
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 --num_envs 16 --viz kit \
    --checkpoint logs/rsl_rl/ur5e_robotiq_2f85_omnireset_agent/<run>/model_2999.pt \
    env.scene.insertive_object=peg env.scene.receptive_object=peghole

# G1-D + Dex1 (Forge checkpoints land under /mnt/robotics/<user>/omni-reset/g1d_dex1_omnireset_agent/<run>/)
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task OmniReset-G1dDex1-RelCartesianOSC-State-Play-v0 --num_envs 16 --viz kit \
    --checkpoint logs/rsl_rl/g1d_dex1_omnireset_agent/<run>/model_<iter>.pt \
    env.scene.insertive_object=peg env.scene.receptive_object=peghole
```

Without `--checkpoint` the newest run under `logs/rsl_rl/<experiment>` is used (`--load_run <dir>` narrows
it). `--real-time` plays at wall-clock speed, `--video --video_length 400` records an MP4 next to the
checkpoint.

## 7. Train

```bash
# local smoke test of the whole G1-D pipeline (assets -> datasets -> 5 PPO iterations), from the host
docker/forge/run_local.sh smoke
NUM_ENVS=256 MAX_ITER=20 docker/forge/run_local.sh smoke-train

# a training run by hand, inside the container
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task OmniReset-G1dDex1-RelCartesianOSC-State-v0 --num_envs 1024 --max_iterations 3000 \
    env.scene.insertive_object=peg env.scene.receptive_object=peghole
```

See `docker/forge/README.md` for the Forge (cluster) submission flow and the VS Code dev container.

## 8. Tests and formatting

```bash
python -m pytest source/uwlab_tasks/test -x          # or any test file / directory
uwlab --format                                       # black + flake8 (pre-commit hooks)
```
