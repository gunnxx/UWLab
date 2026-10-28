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
(`Datasets/OmniReset_isaaclab3`); the G1-D ones read `source/uwlab_assets/data/Datasets/OmniReset_g1d`, generated
by `docker/forge/run_local.sh smoke` (stages `assets,datasets`).

| Env | Dataset files |
|---|---|
| RL (`...-State-*`) | `Resets/<pair>/resets_ObjectAnywhereEEAnywhere.pt` |
| `ObjectAnywhereEEAnywhere` | none (samples everything itself) |
| `ObjectRestingEEGrasped` | `Resets/<pair>/resets_ObjectAnywhereEEAnywhere.pt`, `Grasps/<object>/grasps.pt` |
| `ObjectAnywhereEEGrasped` | `Grasps/<object>/grasps.pt` |
| `ObjectPartiallyAssembled*` | `Resets/<pair>/partial_assemblies.pt` (+ `grasps.pt` for `EEGrasped`) |

Note: a recorded reset state restores every rigid object in the scene, the table included. After changing
the table layout, envs that read `resets_*.pt` keep the recorded table pose until the datasets are re-recorded.

## 2. Inspect a scene and reset it from the keyboard

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

## 3. Replay recorded reset states

Cycles through a saved reset-state dataset (one reset every `--reset_interval` seconds) and prints the
success flag of each loaded state:

```bash
python scripts_v2/tools/visualize_reset_states.py --task OmniReset-G1dDex1-RelCartesianOSC-State-Play-v0 \
    --dataset_dir source/uwlab_assets/data/Datasets/OmniReset_g1d --reset_type ObjectAnywhereEEAnywhere \
    --num_envs 4 --reset_interval 2 --viz kit env.scene.insertive_object=peg env.scene.receptive_object=peghole
```

## 4. Probe why a reset-state set yields nothing

Per-step breakdown of the success term's clauses (gripper, approach orientation, stability, collisions,
pose deviation); `--funnel` accumulates the pass counts over all episodes:

```bash
python scripts_v2/tools/debug_reset_env.py --task OmniReset-G1dDex1-ObjectAnywhereEEGrasped-v0 \
    --num_envs 256 --steps 45 --funnel --viz none env.scene.insertive_object=peg env.scene.receptive_object=peghole
```

## 5. Play a trained policy

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

## 6. Train

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

## 7. Tests and formatting

```bash
python -m pytest source/uwlab_tasks/test -x          # or any test file / directory
uwlab --format                                       # black + flake8 (pre-commit hooks)
```
