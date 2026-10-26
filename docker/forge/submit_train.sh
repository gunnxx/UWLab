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
# Logging: wandb by default, project "omnireset", run name "<task>-<objects>-s<seed>".
#   WANDB_API_KEY is taken from the environment, else from ~/.zshenv. LOGGER=tensorboard disables it.
#
# Checkpoints: training writes to local scratch (logs/rsl_rl/<experiment>/<ts>_<run>). Every
#   SYNC_MINUTES the run directories are rsynced (wandb files excluded) to
#   ${CKPT_MOUNT}/${CKPT_DIR} on the shareable disk CKPT_DISK, and once more at exit, so a run
#   survives the pod. Defaults: blob-robotics-scus mounted at /mnt/robotics, dir twguntara/omni-reset.
#   CKPT_DISK="" disables it. MAX_ITER sets --max_iterations (default 3000; repo cap is 40000).
#
# Resume: RESUME_PATH=/mnt/robotics/twguntara/omni-reset/<run dir>/model_<n>.pt continues that run.
#   RSL-RL restores the iteration counter but treats --max_iterations as *additional* iterations,
#   so the helper passes MAX_ITER - n (MAX_ITER stays the intended total; n is parsed from the
#   checkpoint file name). The checkpoint disk is mounted at CKPT_MOUNT; the path must live under it.
#
# Non-interactive: YES=1 skips the confirmation prompt, DRY_RUN=1 stops after the dry-run.
#
# Env overrides: IMAGE, GPUS, NUM_ENVS, SEED, HOURS, TASK, JOB_NAME, LOGGER, LOG_PROJECT, RUN_NAME,
#                MAX_ITER, EXTRA, CODE_SOURCE, GIT_REPO, GIT_REF, DISK, MOUNT_PATH, CODE_DIR,
#                CKPT_DISK, CKPT_MOUNT, CKPT_DIR, SYNC_MINUTES, WANDB_API_KEY, WANDB_ENTITY, YES, DRY_RUN
set -euo pipefail

INSERTIVE="${1:?insertive object (e.g. peg)}"
RECEPTIVE="${2:?receptive object (e.g. peghole)}"

IMAGE="${IMAGE:-uwlab-omnireset:slim-4bfca44}"
GPUS="${GPUS:-4}"
NUM_ENVS="${NUM_ENVS:-16384}"       # per GPU process; the paper used 16384 x 4 GPUs
SEED="${SEED:-42}"
HOURS="${HOURS:-24}"
TASK="${TASK:-OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0}"
JOB_NAME="${JOB_NAME:-omnireset-${INSERTIVE}-${RECEPTIVE}-s${SEED}${RESUME_PATH:+-resume}}"
LOGGER="${LOGGER:-wandb}"
LOG_PROJECT="${LOG_PROJECT:-omnireset}"
RUN_NAME="${RUN_NAME:-${TASK}-${INSERTIVE}-${RECEPTIVE}-s${SEED}${RESUME_PATH:+-resume}}"

if [[ "${LOGGER}" == "wandb" ]]; then
    if [[ -z "${WANDB_API_KEY:-}" && -f "${HOME}/.zshenv" ]]; then
        WANDB_API_KEY="$(grep -E '^(export )?WANDB_API_KEY=' "${HOME}/.zshenv" | tail -1 | sed -E 's/^(export )?WANDB_API_KEY=//; s/^["'"'"']//; s/["'"'"']$//')"
    fi
    [[ -n "${WANDB_API_KEY:-}" ]] || { echo "LOGGER=wandb but no WANDB_API_KEY in env or ~/.zshenv" >&2; exit 1; }
fi
EXTRA="${EXTRA:-}"
MAX_ITER="${MAX_ITER:-3000}"
RESUME_PATH="${RESUME_PATH:-}"
ITERS_ARG="${MAX_ITER}"
if [[ -n "${RESUME_PATH}" ]]; then
    if [[ "${RESUME_PATH}" =~ model_([0-9]+)\.pt$ ]]; then
        ITERS_ARG=$(( MAX_ITER - BASH_REMATCH[1] ))
        (( ITERS_ARG > 0 )) || { echo "checkpoint iteration ${BASH_REMATCH[1]} >= MAX_ITER ${MAX_ITER}; nothing to do" >&2; exit 1; }
        echo "resume from iteration ${BASH_REMATCH[1]}: ${ITERS_ARG} more iterations to reach ${MAX_ITER}"
    else
        echo "RESUME_PATH does not end in model_<n>.pt; passing MAX_ITER=${MAX_ITER} as additional iterations" >&2
    fi
fi
CODE=/workspace/uwlab

CKPT_DISK="${CKPT_DISK-blob-robotics-scus}"
CKPT_MOUNT="${CKPT_MOUNT:-/mnt/robotics}"
CKPT_DIR="${CKPT_DIR:-twguntara/omni-reset}"
SYNC_MINUTES="${SYNC_MINUTES:-10}"
EXPERIMENT_DIR="logs/rsl_rl/ur5e_robotiq_2f85_omnireset_agent"   # experiment_name in rsl_rl_cfg.py

CODE_SOURCE="${CODE_SOURCE:-git}"
GIT_REPO="${GIT_REPO:-https://github.com/gunnxx/UWLab.git}"
GIT_REF="${GIT_REF:-feature/forge-omnireset}"
DISK="${DISK:-}"
MOUNT_PATH="${MOUNT_PATH:-/mnt/uwlab-code}"
CODE_DIR="${CODE_DIR:-}"

TRAIN="python scripts/reinforcement_learning/rsl_rl/train.py \
  --task ${TASK} --num_envs ${NUM_ENVS} --max_iterations ${ITERS_ARG} \
  --logger ${LOGGER} --log_project_name ${LOG_PROJECT} --run_name ${RUN_NAME} --visualizer none \
  ${RESUME_PATH:+--resume_path ${RESUME_PATH}} \
  agent.seed=${SEED} env.scene.insertive_object=${INSERTIVE} env.scene.receptive_object=${RECEPTIVE} ${EXTRA}"
if (( GPUS > 1 )); then
    TRAIN="python -m torch.distributed.run --standalone --nnodes 1 --nproc_per_node ${GPUS} \
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

if [[ -n "${CKPT_DISK}" ]]; then
    DEST="${CKPT_MOUNT}/${CKPT_DIR}"
    SYNC="rsync -a --exclude 'wandb/' ${EXPERIMENT_DIR}/ ${DEST}/"
    ARGS+=(--disk-mount "${CKPT_DISK}:${CKPT_MOUNT}")
    # background sync loop; final sync after training regardless of its exit code
    TRAIN="mkdir -p ${DEST} ${EXPERIMENT_DIR} && (while true; do sleep $((SYNC_MINUTES*60)); ${SYNC} 2>/dev/null; done &) ; \
${TRAIN}; rc=\$?; ${SYNC}; echo \"[ckpt] synced ${EXPERIMENT_DIR} -> ${DEST}\"; ls ${DEST}; exit \$rc"
fi
ENTRY="uwlab-init-cache; ${PREP} && ${TRAIN}"
ARGS+=(-m "OmniReset ${INSERTIVE}/${RECEPTIVE} RSL-RL PPO seed ${SEED}, ${GPUS}x${NUM_ENVS} envs, ${SRC_DESC}")
ARGS+=(--entrypoint "${ENTRY}")
if [[ "${LOGGER}" == "wandb" ]]; then
    ARGS+=(--env "WANDB_API_KEY=${WANDB_API_KEY}")
    [[ -n "${WANDB_ENTITY:-}" ]] && ARGS+=(--env "WANDB_ENTITY=${WANDB_ENTITY}")
fi

echo "== dry-run: ${JOB_NAME} =="
forge job dry-run "${ARGS[@]}"
echo
echo "entrypoint: ${ENTRY}"
[[ "${DRY_RUN:-0}" == "1" ]] && exit 0
if [[ "${YES:-0}" != "1" ]]; then
    read -r -p "Submit this job? [y/N] " yn
    [[ "${yn}" == [yY] ]] || { echo "not submitted"; exit 0; }
fi
forge job submit "${ARGS[@]}"
