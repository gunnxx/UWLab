#!/usr/bin/env bash
# Dry-run, then (on confirmation) submit an OmniReset RSL-RL training job on Forge.
#
#   docker/forge/submit_train.sh peg peghole                       # 4 GPUs, paper setup
#   GPUS=1 NUM_ENVS=16384 docker/forge/submit_train.sh peg peghole
#   GIT_REF=my-branch docker/forge/submit_train.sh peg peghole     # train another branch of the fork
#
# Code source (CODE_SOURCE):
#   git  (default) clone ${GIT_REPO}@${GIT_REF} into /workspace/uwlab at job start.
#                  /workspace is the pod's local scratch volume; the clone takes ~2 s.
#                  Push before you submit: the job sees the fork, not your laptop.
#   disk           mount a Forge disk at ${MOUNT_PATH} and symlink /workspace/uwlab to
#                  ${MOUNT_PATH}/${CODE_DIR}. Only for a `disk` (fast, RWO) profile.
#                  Blob (blobfuse) disks lost directory listings under a git tree and
#                  broke Python imports (2026-10-24); do not use them for code.
#
# Env overrides: IMAGE, GPUS, NUM_ENVS, SEED, HOURS, TASK, JOB_NAME, LOGGER, EXTRA,
#                CODE_SOURCE, GIT_REPO, GIT_REF, DISK, MOUNT_PATH, CODE_DIR, WANDB_API_KEY
set -euo pipefail

INSERTIVE="${1:?insertive object (e.g. peg)}"
RECEPTIVE="${2:?receptive object (e.g. peghole)}"

IMAGE="${IMAGE:-uwlab-omnireset:slim-4bfca44}"
GPUS="${GPUS:-4}"
NUM_ENVS="${NUM_ENVS:-16384}"       # per GPU process; the paper used 16384 x 4 GPUs
SEED="${SEED:-42}"
HOURS="${HOURS:-24}"
TASK="${TASK:-OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0}"
JOB_NAME="${JOB_NAME:-omnireset-${INSERTIVE}-${RECEPTIVE}-s${SEED}}"
LOGGER="${LOGGER:-tensorboard}"
EXTRA="${EXTRA:-}"
CODE=/workspace/uwlab

CODE_SOURCE="${CODE_SOURCE:-git}"
GIT_REPO="${GIT_REPO:-https://github.com/gunnxx/UWLab.git}"
GIT_REF="${GIT_REF:-feature/forge-omnireset}"
DISK="${DISK:-}"
MOUNT_PATH="${MOUNT_PATH:-/mnt/uwlab-code}"
CODE_DIR="${CODE_DIR:-}"

TRAIN="python scripts/reinforcement_learning/rsl_rl/train.py \
  --task ${TASK} --num_envs ${NUM_ENVS} --logger ${LOGGER} --visualizer none \
  agent.seed=${SEED} env.scene.insertive_object=${INSERTIVE} env.scene.receptive_object=${RECEPTIVE} ${EXTRA}"
if (( GPUS > 1 )); then
    TRAIN="python -m torch.distributed.run --nnodes 1 --nproc_per_node ${GPUS} \
  ${TRAIN#python } --distributed"
fi

ARGS=(--name "${JOB_NAME}" --image "${IMAGE}" --gpu "${GPUS}" --max-duration "${HOURS}")

case "${CODE_SOURCE}" in
  git)
    PREP="rm -rf ${CODE} && git clone -q --depth 1 -b ${GIT_REF} ${GIT_REPO} ${CODE} && \
ln -s /isaac-sim ${CODE}/_isaac_sim && ln -s /opt/uwlab-deps/_isaaclab ${CODE}/_isaaclab && \
cd ${CODE} && git log -1 --oneline"
    SRC_DESC="code ${GIT_REF}@fork"
    ;;
  disk)
    [[ -n "${DISK}" ]] || { echo "CODE_SOURCE=disk needs DISK=<name>" >&2; exit 1; }
    PREP="rm -rf ${CODE} && ln -sfn ${MOUNT_PATH}${CODE_DIR:+/${CODE_DIR}} ${CODE} && cd ${CODE}"
    ARGS+=(--disk-mount "${DISK}:${MOUNT_PATH}")
    SRC_DESC="code on disk ${DISK}"
    ;;
  *) echo "unknown CODE_SOURCE=${CODE_SOURCE}" >&2; exit 1;;
esac

ENTRY="uwlab-init-cache; ${PREP} && ${TRAIN}"
ARGS+=(-m "OmniReset ${INSERTIVE}/${RECEPTIVE} RSL-RL PPO seed ${SEED}, ${GPUS}x${NUM_ENVS} envs, ${SRC_DESC}")
ARGS+=(--entrypoint "${ENTRY}")
[[ -n "${WANDB_API_KEY:-}" ]] && ARGS+=(--env "WANDB_API_KEY=${WANDB_API_KEY}")

echo "== dry-run =="
forge job dry-run "${ARGS[@]}"
echo
echo "entrypoint: ${ENTRY}"
read -r -p "Submit this job? [y/N] " yn
[[ "${yn}" == [yY] ]] || { echo "not submitted"; exit 0; }
forge job submit "${ARGS[@]}"
