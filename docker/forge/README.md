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

The image is a single squashed layer and skips Isaac Lab's `isaacsim` pip extra (the container already is Isaac Sim), which keeps it small enough for Forge nodes to unpack. Isaac Sim's runtime caches are symlinked into `/tmp`; run `uwlab-init-cache` first in any entrypoint that does not go through bash's login profile (the helpers do).

Rebuild only when a dependency changes (Isaac Sim / Isaac Lab / RSL-RL / torch
pins, a new `install_requires` entry in a `setup.py`, a new apt package).

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

**Default: clone at job start.** `submit_train.sh` makes every job run
`git clone --depth 1 -b <branch> <fork>` into `/workspace/uwlab`, the pod's local
scratch volume (hundreds of GB, local SSD). The repo is ~30 MB, so this costs about
two seconds. The editable installs in the image point at `/workspace/uwlab`, so the
cloned code is what runs. Push to the fork before submitting; the job never sees your
laptop. Forge mounts its own ephemeral volume at `/workspace`, which hides the copy
baked into the image, so some code source is always required.

**Why not a mounted disk.** The first attempt used a shared `blob` disk mounted at
`/workspace/uwlab`. After one successful run the blobfuse mount stopped enumerating
directories (`ls` empty, `find` found 0 files) while files were still readable by
path, and Python could no longer import sibling modules such as `cli_args`. Blobfuse
is not a filesystem for git trees. The `disk` (Premium SSD, RWO) profile would work,
but it is single-attach, so parallel seeds would each need their own disk.
`CODE_SOURCE=disk DISK=<name>` remains available for that case.

**Interactive editing on the cluster.** Start an SSH notebook on the image, clone the
fork there, edit, commit, push, then submit jobs. `code_sync.sh` clones or
fast-forwards a checkout in place:

```bash
forge notebook create --name uwlab-dev --type ssh --gpu 0 --custom-image uwlab-omnireset:slim-4bfca44 \
    -m "UW Lab dev shell"
bash docker/forge/code_sync.sh feature/forge-omnireset     # inside the notebook
```

Checkpoints and logs are written under `/workspace/uwlab/logs`, which is scratch and
disappears with the pod. For runs you want to keep, add `--disk-mount <data-disk>:/mnt/data`
on a shareable disk and pass `agent.log_dir`/`--logger wandb`, or copy the run
directory out at the end of the entrypoint.

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
    --entrypoint "uwlab-init-cache; git clone -q --depth 1 -b feature/forge-omnireset https://github.com/gunnxx/UWLab.git /workspace/uwlab && cd /workspace/uwlab && python -m torch.distributed.run --nnodes 1 --nproc_per_node 4 \
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
