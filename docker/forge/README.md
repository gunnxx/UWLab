# UW Lab on Forge

Build context and helpers for running UW Lab / OmniReset on the Forge cluster
without rebuilding the image on every code change.

## How it works

* `Dockerfile` bakes **all dependencies** on top of `nvcr.io/nvidia/isaac-sim:6.1.0`:
  Isaac Lab 3.0 EA (pin inside `uwlab.sh`), torch 2.11+cu128, RSL-RL (pin in
  `source/uwlab_rl/setup.py`), and the four `uwlab*` extensions installed
  **editable** from `/workspace/uwlab`.
* A Forge disk holding your working copy is mounted **over** `/workspace/uwlab`
  at job time. The editable installs resolve to whatever is at that path, so the
  mounted code is what runs. No `pip install`, no rebuild.
* Everything that has to survive that mount lives outside `/workspace/uwlab`:
  Isaac Sim (`/isaac-sim`), the Isaac Lab clone (`/opt/uwlab-deps/_isaaclab`), and
  the `python` / `pip` / `uwlab` wrappers in `/usr/local/bin`.

Rebuild only when a dependency changes (Isaac Sim / Isaac Lab / RSL-RL / torch
pins, a new `install_requires` entry in a `setup.py`, a new apt package).

## 1. Build the image (once)

```bash
# from the repo root
forge image build --name uwlab-omnireset --tag $(git rev-parse --short HEAD) \
    --context docker/forge --build-timeout 4 --build-memory 32Gi --build-cpu 8
forge image logs <image-id>
forge image list --mine
```

`forge image build` has no `--build-arg`; edit the `ARG UWLAB_COMMIT` default in
the Dockerfile to change the baked commit (only matters as a fallback when no disk
is mounted).

## 2. Create the code disk (once)

Forge mounts its own ephemeral volume at `/workspace` in every job, which hides the
copy baked into the image. A disk (or a symlink into one) at `/workspace/uwlab` is
therefore required, not optional.

Use the shared `blob` profile so a notebook (for editing) and training jobs can
mount the same disk at the same time. The private `disk` profile is faster but
single-attach: a job cannot start while a notebook holds it.

```bash
forge disk create --name uwlab-code --size 20 --profile blob
```

**Alternative: the team's shared disk.** `blob-robotics-scus` (profile `blob`,
RWX, mounted by the team's training jobs at `/mnt/robotics`) can hold the code too,
e.g. under `/mnt/robotics/code/uwlab/<you>/UWLab`. The job then symlinks
`/workspace/uwlab` to it:

```bash
DISK=blob-robotics-scus MOUNT_PATH=/mnt/robotics CODE_DIR=code/uwlab/<you>/UWLab \
    docker/forge/submit_train.sh peg peghole
```

Trade-off: one disk for everything, but git churn (many small files) lands on the
volume that teammates' 8-GPU runs read datasets from, and a stale blobfuse cache
on one node can serve old code for a while after a pull on another. A dedicated
`uwlab-code` disk isolates that; keep datasets and checkpoints on the shared one.

## 3. Put code on the disk

Two ways. Both leave a real git checkout on the disk, so `git pull` works there.

**a) git from inside Forge** (no local copy needed, also usable as a job
`--setup-command`). `code_sync.sh` clones on an empty disk, otherwise fetches and
checks out the requested branch or commit. It refuses to touch a tree with
uncommitted changes and takes a lock so two jobs never pull at once.

```bash
# first time, from a notebook or a 0-GPU job that mounts the disk
forge job submit --name uwlab-code-bootstrap -m "clone UWLab onto the code disk" \
    --image uwlab-omnireset --gpu 0 --disk-mount uwlab-code:/workspace/uwlab --max-duration 1 \
    --entrypoint "wget -qO- https://raw.githubusercontent.com/gunnxx/UWLab/feature/forge-omnireset/docker/forge/code_sync.sh | bash -s -- feature/forge-omnireset"

# later: pull the latest of a branch, or pin a commit, in a notebook terminal
bash /workspace/uwlab/docker/forge/code_sync.sh feature/forge-omnireset
bash /workspace/uwlab/docker/forge/code_sync.sh 2b40afb

# or let the training job do it right before it starts
GIT_REF=feature/forge-omnireset docker/forge/submit_train.sh peg peghole
```

The disk tracks the fork `gunnxx/UWLab` (upstream is `UW-Lab/UWLab`). For a private fork set `UWLAB_REPO=https://<user>:<token>@github.com/<user>/UWLab.git`
(pass it with `--env` on the job).

**b) rsync from your laptop** through an SSH notebook on the image. Includes
`.git` by default so (a) keeps working afterwards; `NO_GIT=1` skips it.

```bash
forge notebook create --name uwlab-dev --type ssh --gpu 0 \
    --custom-image uwlab-omnireset --disk-mount uwlab-code:/workspace/uwlab \
    -m "UW Lab code disk editor (rsync target)"
forge notebook get uwlab-dev           # SSH host / port
docker/forge/sync_code.sh root@<host> -p <port>
```

Blob storage has no file locking and is slow on many-small-file operations, so:
sync once, then launch your batch of jobs. Do not run `code_sync.sh` from several
jobs that start simultaneously on the same disk.

## 4. Train

`submit_train.sh` dry-runs, prints the entrypoint, and submits only after you
confirm. Defaults match the paper setup for a single task: 4 GPUs x 16384 envs.

```bash
docker/forge/submit_train.sh peg peghole                 # seed 42, 4 GPUs
SEED=43 docker/forge/submit_train.sh peg peghole
GPUS=1 NUM_ENVS=16384 docker/forge/submit_train.sh peg peghole   # smaller run
WANDB_API_KEY=... LOGGER=wandb docker/forge/submit_train.sh peg peghole
```

The equivalent raw command:

```bash
forge job dry-run --name omnireset-peg-s42 -m "OmniReset peg insertion, seed 42" \
    --image uwlab-omnireset --gpu 4 --max-duration 24 \
    --disk-mount uwlab-code:/workspace/uwlab \
    --entrypoint "cd /workspace/uwlab && python -m torch.distributed.run --nnodes 1 --nproc_per_node 4 \
      scripts/reinforcement_learning/rsl_rl/train.py \
      --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0 --num_envs 16384 \
      --logger tensorboard --visualizer none --distributed agent.seed=42 \
      env.scene.insertive_object=peg env.scene.receptive_object=peghole"
```

Notes

* The cluster is single-node (`forge cluster info`: MULTI-NODE no), so use
  `--gpu N` and `python -m torch.distributed.run` yourself; `--num-nodes` stays 1.
* `--num_envs` is **per process**. 16384 x 4 GPUs is the paper configuration.
* Logs land in `/workspace/uwlab/logs/rsl_rl/...` on the code disk. Each job
  writes its own run directory, which is safe on blob.
* `python` in the image is Isaac Sim's `python.sh`. Use it for everything.
* Evaluation: same image, `--gpu 1`, entrypoint
  `cd /workspace/uwlab && python scripts/reinforcement_learning/rsl_rl/play.py --task OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0 --num_envs 1 --visualizer none --checkpoint <path> env.scene.insertive_object=peg env.scene.receptive_object=peghole`.
