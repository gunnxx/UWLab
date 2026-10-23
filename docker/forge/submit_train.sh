#!/usr/bin/env bash
# Dry-run, then (on confirmation) submit an OmniReset RSL-RL training job on Forge.
#
#   docker/forge/submit_train.sh peg peghole                    # 4 GPUs, paper setup
#   GPUS=1 NUM_ENVS=16384 docker/forge/submit_train.sh peg peghole
#   GIT_REF=main docker/forge/submit_train.sh peg peghole       # git pull on the disk first
#
# Env overrides: IMAGE, DISK, GPUS, NUM_ENVS, SEED, HOURS, TASK, JOB_NAME, LOGGER,
#                EXTRA (appended to train.py), GIT_REF (branch/commit to sync before training),
#                WANDB_API_KEY (passed through when set)
set -euo pipefail

INSERTIVE="${1:?insertive object (e.g. peg)}"
RECEPTIVE="${2:?receptive object (e.g. peghole)}"

IMAGE="${IMAGE:-uwlab-omnireset}"
DISK="${DISK:-uwlab-code}"
# Where the disk is mounted and where the repo sits inside it. Default: a dedicated
# code disk mounted directly at /workspace/uwlab. For the shared team disk use e.g.
#   DISK=blob-robotics-scus MOUNT_PATH=/mnt/robotics CODE_DIR=code/uwlab/<you>/UWLab
# and the job symlinks /workspace/uwlab -> ${MOUNT_PATH}/${CODE_DIR} before training.
MOUNT_PATH="${MOUNT_PATH:-/workspace/uwlab}"
CODE_DIR="${CODE_DIR:-}"
GPUS="${GPUS:-4}"
NUM_ENVS="${NUM_ENVS:-16384}"       # per GPU process; the paper used 16384 x 4 GPUs
SEED="${SEED:-42}"
HOURS="${HOURS:-24}"
TASK="${TASK:-OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0}"
JOB_NAME="${JOB_NAME:-omnireset-${INSERTIVE}-${RECEPTIVE}-s${SEED}}"
LOGGER="${LOGGER:-tensorboard}"
EXTRA="${EXTRA:-}"
GIT_REF="${GIT_REF:-}"
CODE=/workspace/uwlab

TRAIN="python scripts/reinforcement_learning/rsl_rl/train.py \
  --task ${TASK} --num_envs ${NUM_ENVS} --logger ${LOGGER} --visualizer none \
  agent.seed=${SEED} env.scene.insertive_object=${INSERTIVE} env.scene.receptive_object=${RECEPTIVE} ${EXTRA}"

if (( GPUS > 1 )); then
    TRAIN="python -m torch.distributed.run --nnodes 1 --nproc_per_node ${GPUS} \
  ${TRAIN#python } --distributed"
fi

LINK=""
if [[ "${MOUNT_PATH}" != "${CODE}" ]]; then
    LINK="rm -rf ${CODE} && ln -sfn ${MOUNT_PATH}${CODE_DIR:+/${CODE_DIR}} ${CODE} && "
fi
ENTRY="uwlab-init-cache; ${LINK}cd ${CODE} && ${TRAIN}"
DESC="OmniReset ${INSERTIVE}/${RECEPTIVE} RSL-RL PPO seed ${SEED}, ${GPUS}x${NUM_ENVS} envs"

ARGS=(
  --name "${JOB_NAME}"
  -m "${DESC}"
  --image "${IMAGE}"
  --gpu "${GPUS}"
  --disk-mount "${DISK}:${MOUNT_PATH}"
  --max-duration "${HOURS}"
  --entrypoint "${ENTRY}"
)

# Optional: update the code disk from git before training. Do not combine with
# other jobs that are starting on the same disk at the same moment (blob has no
# locking); sync once, then launch the batch.
if [[ -n "${GIT_REF}" ]]; then
    ARGS+=(--setup-command "UWLAB_PATH=${MOUNT_PATH}${CODE_DIR:+/${CODE_DIR}} bash ${MOUNT_PATH}${CODE_DIR:+/${CODE_DIR}}/docker/forge/code_sync.sh ${GIT_REF}")
fi
[[ -n "${WANDB_API_KEY:-}" ]] && ARGS+=(--env "WANDB_API_KEY=${WANDB_API_KEY}")

echo "== dry-run =="
forge job dry-run "${ARGS[@]}"
echo
echo "entrypoint: ${ENTRY}"
[[ -n "${GIT_REF}" ]] && echo "setup:      code_sync.sh ${GIT_REF}"
read -r -p "Submit this job? [y/N] " yn
[[ "${yn}" == [yY] ]] || { echo "not submitted"; exit 0; }
forge job submit "${ARGS[@]}"
