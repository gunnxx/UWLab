# UW Lab on Forge

Build context and helpers for running UW Lab / OmniReset on the Forge cluster
without rebuilding the image on every code change.

## How it works

* `Dockerfile` bakes **all dependencies** on top of `nvcr.io/nvidia/isaac-sim:6.1.0`:
  Isaac Lab 3.0 EA (pin inside `uwlab.sh`), torch 2.11+cu128, RSL-RL (pin in
  `source/uwlab_rl/setup.py`), and the four `uwlab*` extensions installed
  **editable** from `/workspace/uwlab`.
* The code is **not** in the image. At job start a snapshot of your working copy
  (uploaded by `push_code.sh`, see section 2) is extracted into `/workspace/uwlab`,
  the pod's local scratch. The editable installs resolve to whatever is at that path,
  so the extracted code is what runs. No `pip install`, no rebuild, no push.
* Everything that has to survive that replacement lives outside `/workspace/uwlab`:
  Isaac Sim (`/isaac-sim`), the Isaac Lab clone (`/opt/uwlab-deps/_isaaclab`), and
  the `python` / `pip` / `uwlab` wrappers in `/usr/local/bin`.

The image is a single squashed layer and skips Isaac Lab's `isaacsim` pip extra (the container already is Isaac Sim), which keeps it small enough for Forge nodes to unpack. Isaac Sim's runtime caches are symlinked into `/tmp`; run `uwlab-init-cache` first in any entrypoint that does not go through bash's login profile (the helpers do).

Rebuild only when a dependency changes (Isaac Sim / Isaac Lab / RSL-RL / torch
pins, a new `install_requires` entry in a `setup.py`, a new apt package).

## 0. Local smoke test (same image, this machine)

The Forge image runs unchanged on a workstation with an NVIDIA GPU and the container toolkit;
`run_local.sh` reproduces the job layout (working copy mounted over `/workspace/uwlab`, a local
folder in place of the blob disk at `/mnt/robotics`, persistent Isaac Sim caches):

```bash
docker build -f docker/forge/Dockerfile -t uwlab-omnireset:local docker/forge   # once, ~1 h
docker/forge/run_local.sh smoke          # G1-D assets -> tiny datasets -> 5 PPO iterations (1 GPU, ~20 min)
docker/forge/run_local.sh smoke-train    # train stage only, from the cached assets/datasets
docker/forge/run_local.sh                # interactive shell
NUM_ENVS=256 MAX_ITER=20 docker/forge/run_local.sh smoke-train   # any g1d_pipeline.sh knob overrides
```

Cache and checkpoints land in `~/.cache/uwlab-local/robotics/twguntara/omni-reset/`, mirroring the
cluster paths; TensorBoard files in `logs/rsl_rl/g1d_dex1_omnireset_agent/`.

### Watching an environment with a window

`run_local.sh` passes the X display through when `DISPLAY` is set (run `xhost +si:localuser:root`
once per login). Isaac Lab 3.0 runs headless unless a Kit visualizer is requested, so pass
`--viz kit` (there is no `--headless` flag any more; headless is `--viz none`, which the pipeline uses).
Example, the UR5e peg-insertion reset states (dataset downloaded from HuggingFace on first use):

```bash
docker/forge/run_local.sh python scripts_v2/tools/visualize_reset_states.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 --num_envs 16 --reset_interval 1.0 --viz kit \
    --dataset_dir https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/83860532010b2737aa80d6e8621235ee554186f0/Datasets/OmniReset_isaaclab3
```

Add `--reset_type ObjectRestingEEGrasped` (or one of the other three set names) to cycle a single set.

The G1-D variant reads the locally generated sets (the `datasets` stage of `g1d_pipeline.sh` writes them
under `source/uwlab_assets/data/Datasets/OmniReset_g1d`, i.e. after `run_local.sh smoke`). The sets store the
root pose of every rigid object including the table, so after any scene change re-record them with
`FORCE=1 STAGES=datasets docker/forge/run_local.sh smoke` (and `FORCE=1` on the next Forge job):

```bash
docker/forge/run_local.sh python scripts_v2/tools/visualize_reset_states.py \
    --task OmniReset-G1dDex1-RelCartesianOSC-State-Play-v0 --num_envs 16 --reset_interval 1.0 --viz kit \
    --dataset_dir source/uwlab_assets/data/Datasets/OmniReset_g1d
```

Without a display, `scripts_v2/tools/render_reset_states.py` writes the first N states of every set as images
(one camera per environment, RTX, `--viz none`): `--task ... --dataset_dir ... --out_dir logs/reset_renders`.
By default it shows the state as recorded (two physics steps). `--settle_steps 240 --gripper close` shows the
same states after two seconds with the task's own controller holding the arm at a zero action and the gripper
commanded closed, which is what the recorder does during its hold (joint position targets would not hold the
arm: the arm actuators have no joint stiffness, the OSC supplies the torques), and `poses.json` next to the images holds the peg, hole and end-effector positions so two
renders can be compared numerically. A recorded grasp holds the peg only while the gripper is *commanded*
closed: the dataset stores finger positions, the clamping force comes from the position target being past the
contact, so a policy that opens the gripper on its first step drops the peg by design.

### Checking why a reset set yields nothing

`scripts_v2/tools/debug_reset_env.py --funnel --close_gripper --task OmniReset-G1dDex1-<Set>-v0 --num_envs 256
--steps 150 --viz none env.scene.insertive_object=peg env.scene.receptive_object=peghole` prints, at the end,
how many resets were discarded at once because the IK placement missed the grasp pose
(`grasp_placement_failed`), how many holds were abandoned early because a body had already moved to twice its
allowed deviation or dropped below the height threshold (`hold_lost`), how many reached the end of the hold, and
how many of those passed each clause of the success term (approach from above, stability, pose deviation per
asset, collision freedom, fingers blocked, end effector still at the grasp pose). The tool also prints the
environment step rate. The expensive clauses (signed-distance collision queries, grasp-pose comparison, assembly
check) run only on the final step of a hold and only for environments that passed the cheap clauses; on the
G1-D the collision analyzers sample the right arm and Dex1 links only. The "EE grasped" sets need the end effector to stay within 1.5 cm and
15 degrees of the sampled grasp pose in the object's frame; the IK iterates are clamped to the joint limits
(an out-of-range wrist is snapped back by PhysX on the first step and throws the hand off the peg); grasps
whose fingers never closed are dropped when the grasp dataset is loaded.

To play a trained policy with the window, run `play.py` with `--viz kit`, the `-Play-v0` task id and a
checkpoint (a path or URL; without `--checkpoint` the newest run under `logs/rsl_rl/<experiment>` is used,
narrow it with `--load_run <run dir>`). The hydra overrides at the end pick the object pair:

```bash
# UR5e + Robotiq (Stage 1 checkpoint; use ...-State-Finetune-Play-v0 for a Stage 2 one)
docker/forge/run_local.sh python scripts/reinforcement_learning/rsl_rl/play.py \
    --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 --num_envs 16 --viz kit \
    --checkpoint logs/rsl_rl/ur5e_robotiq_2f85_omnireset_agent/<run>/model_2999.pt \
    env.scene.insertive_object=peg env.scene.receptive_object=peghole

# G1-D + Dex1 (checkpoints synced from a Forge job land under /mnt/robotics/<user>/omni-reset/g1d_dex1_omnireset_agent/<run>/)
docker/forge/run_local.sh python scripts/reinforcement_learning/rsl_rl/play.py \
    --task OmniReset-G1dDex1-RelCartesianOSC-State-Play-v0 --num_envs 16 --viz kit \
    --checkpoint logs/rsl_rl/g1d_dex1_omnireset_agent/<run>/model_<iter>.pt \
    env.scene.insertive_object=peg env.scene.receptive_object=peghole
```

`--real-time` slows playback to wall-clock speed; `--video --video_length 400` records an MP4 next to the
checkpoint. Run one Kit instance at a time: the env reserves a 2 GB PhysX collision stack and two instances
overflow a 16 GB card.

### Browsing and debugging the code in VS Code

Isaac Sim, Isaac Lab and the pinned dependencies exist only inside the image, so on the host Pylance
cannot resolve most imports (the `_isaac_sim` / `_isaaclab` symlinks dangle). `.devcontainer/` opens
this working copy *inside* `uwlab-omnireset:local` with the `run_local.sh` layout (repo at
`/workspace/uwlab`, `~/.cache/uwlab-local/robotics` at `/mnt/robotics`, persistent Isaac Sim caches,
GPU, X display):

1. Build the image as above (once), then in VS Code: `F1` -> **Dev Containers: Reopen in Container**
   (the `ms-vscode-remote.remote-containers` extension). The first start installs the VS Code server and
   the Python extensions into the container and runs `.devcontainer/post_create.sh`, which recreates the
   two symlinks, whitelists the repo for git, and generates `.vscode/settings.json` (Pylance
   `extraPaths` for Isaac Sim's Kit extensions, Isaac Lab under `_isaaclab/`, the four uwlab
   extensions) from `.vscode/tools/settings.template.json`.
2. "Go to Definition" then follows imports into Isaac Lab and Isaac Sim; the integrated terminal is a
   shell in the container (`python` is Isaac Sim's `python.sh`), and the `launch.json` configurations
   (play / train OmniReset G1-D, pytest) debug under the same interpreter. `xhost +si:localuser:root`
   once per login for the Kit window.

The container runs as root; files it creates in the working copy are chowned back to the host user on
every container start. Point the config at another image with `UWLAB_DEVCONTAINER_IMAGE=<image> code .`.
Rebuild the image -> **Dev Containers: Rebuild Container** picks it up.

## 1. Build the image (once)

```bash
# from the repo root
forge image build --name uwlab-omnireset --tag slim-$(git rev-parse --short HEAD) \
    --context docker/forge --build-timeout 4 --build-memory 32Gi --build-cpu 8
forge image logs <image-id>
forge image list --mine
```

`forge image build` has no `--build-arg`; edit the `ARG UWLAB_COMMIT` default in
the Dockerfile to change the baked commit (only matters as a fallback when no disk
is mounted).

## 2. How code reaches the job

**Default: a tarball of your working copy, extracted at job start.** `push_code.sh`
snapshots the laptop checkout (tracked files plus untracked files that are not
gitignored, so uncommitted edits ship and logs/datasets do not; ~22 MB) and uploads it to
`/mnt/robotics/twguntara/code/uwlab-src.tar.gz` on the team disk through an SSH
notebook that mounts the disk. `submit_train.sh` (`CODE_SOURCE=tar`) makes the job mount
that disk (it already does, for checkpoints), extract the tarball into `/workspace/uwlab`
on the pod's local scratch and run from there. No image rebuild, no commit, no push for a
code change; two jobs never share a working tree, and a job keeps the snapshot it started
with even if you push again. The job log starts with the snapshot's `[code] commit …`
stamp. The helper warns at submit time when the working tree differs from the last push.

```bash
# once: a CPU-only SSH notebook as the upload channel (lives up to 168 h)
forge notebook create --name uwlab-code-push --type ssh --gpu 0 --duration 168 \
    --disk-mount blob-robotics-scus:/mnt/robotics -m "UW Lab code upload channel (push_code.sh)"
# every time the code changed
docker/forge/push_code.sh                 # or: docker/forge/push_code.sh <other-notebook>
docker/forge/submit_train.sh peg peghole
```

**Alternative: clone at job start** (`CODE_SOURCE=git`). The job runs
`git clone --depth 1 -b <branch> <fork>` into `/workspace/uwlab` (~2 s). Push to the fork
before submitting; the job never sees your laptop. Forge mounts its own ephemeral volume
at `/workspace`, which hides the copy baked into the image, so some code source is
always required.

**Why not run from a mounted disk directly.** The first attempt mounted a shared `blob`
(blobfuse) disk at `/workspace/uwlab`. After one successful run the mount stopped
enumerating directories (`ls` empty, `find` found 0 files) while files were still
readable by path, and Python could no longer import sibling modules such as `cli_args`.
Blobfuse is not a filesystem for git trees; reading a single tarball by path and
extracting it locally sidesteps the problem entirely. (The team disk `blob-robotics-scus`
is in fact served over NFS, which lists fine, but local extraction is still faster and
isolates concurrent jobs.) The `disk` (Premium SSD, RWO) profile would work for a direct
mount but is single-attach, so parallel seeds would each need their own disk;
`CODE_SOURCE=disk DISK=<name>` remains available for that case.

**Interactive editing on the cluster.** Start an SSH notebook on the image, clone the
fork there, edit, commit, push, then submit jobs. `code_sync.sh` clones or
fast-forwards a checkout in place:

```bash
forge notebook create --name uwlab-dev --type ssh --gpu 0 --custom-image uwlab-omnireset:slim-4bfca44 \
    -m "UW Lab dev shell"
bash docker/forge/code_sync.sh feature/forge-omnireset     # inside the notebook
```

Checkpoints and TensorBoard files are written to local scratch and rsynced every 10 minutes
(and at exit) to `/mnt/robotics/twguntara/omni-reset/<timestamp>_<run name>/` on the
team disk `blob-robotics-scus`. Training never blocks on the shared mount, and the
destination mirrors RSL-RL's log root, so a run that hit its duration cap continues with
`RESUME_PATH=/mnt/robotics/twguntara/omni-reset/<run dir>/model_<n>.pt docker/forge/submit_train.sh <objs>`
(`MAX_ITER` stays the intended total; the helper subtracts the checkpoint's iteration because RSL-RL counts `--max_iterations` as additional iterations after a resume). Weights & Biases files stay local.

## 4. Train

`submit_train.sh` dry-runs, prints the entrypoint, and submits only after you
confirm. Defaults match the paper setup for a single task: 4 GPUs x 16384 envs,
logged to Weights & Biases project `omnireset` with run name
`<task>-<insertive>-<receptive>-s<seed>`. The API key is read from the environment
or `~/.zshenv` and passed to the job as `--env WANDB_API_KEY`.

```bash
docker/forge/submit_train.sh peg peghole                 # seed 42, 4 GPUs
SEED=43 docker/forge/submit_train.sh peg peghole
GPUS=1 NUM_ENVS=16384 docker/forge/submit_train.sh peg peghole   # smaller run
LOGGER=tensorboard docker/forge/submit_train.sh peg peghole                   # local TensorBoard only
```

The equivalent raw command:

```bash
forge job dry-run --name omnireset-peg-s42 -m "OmniReset peg insertion, seed 42" \
    --image uwlab-omnireset:slim-4bfca44 --gpu 4 --max-duration 24 \
    --disk-mount blob-robotics-scus:/mnt/robotics \
    --entrypoint "uwlab-init-cache; mkdir -p /workspace/uwlab && tar xzf /mnt/robotics/twguntara/code/uwlab-src.tar.gz -C /workspace/uwlab && cd /workspace/uwlab && python -m torch.distributed.run --nnodes 1 --nproc_per_node 4 \
      scripts/reinforcement_learning/rsl_rl/train.py \
      --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0 --num_envs 16384 \
      --logger tensorboard --visualizer none --distributed agent.seed=42 \
      env.scene.insertive_object=peg env.scene.receptive_object=peghole"
```

Notes

* The cluster is single-node (`forge cluster info`: MULTI-NODE no), so use
  `--gpu N` and `python -m torch.distributed.run` yourself; `--num-nodes` stays 1.
* `--num_envs` is **per process**. 16384 x 4 GPUs is the paper configuration.
* Logs land in `/workspace/uwlab/logs/rsl_rl/...` on local scratch and are
  rsynced to the team disk (see section 2); the pod's scratch is gone when it ends.
* `python` in the image is Isaac Sim's `python.sh`. Use it for everything.
* Evaluation: same image, `--gpu 1`, entrypoint
  `cd /workspace/uwlab && python scripts/reinforcement_learning/rsl_rl/play.py --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 --num_envs 1 --visualizer none --checkpoint <path> env.scene.insertive_object=peg env.scene.receptive_object=peghole`.
