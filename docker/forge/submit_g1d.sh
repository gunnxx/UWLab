#!/usr/bin/env bash
# Dry-run, then (on confirmation) submit a Forge job that runs docker/forge/g1d_pipeline.sh: build the
# Unitree G1-D + Dex1-1 assets, record the OmniReset datasets for it, and train stage 1.
#
#   docker/forge/submit_g1d.sh                                  # peg/peghole, all stages, 1 GPU
#   STAGES=assets,datasets docker/forge/submit_g1d.sh           # only prepare (and cache) assets + datasets
#   STAGES=train GPUS=4 NUM_ENVS=16384 docker/forge/submit_g1d.sh   # train from the cached datasets
#   STAGES=assets,datasets,train MAX_ITER=5 HOURS=3 docker/forge/submit_g1d.sh   # end-to-end smoke test
#
# The job clones ${GIT_REPO}@${GIT_REF} (push first!) and mounts the shared blob disk at /mnt/robotics for the
# asset/dataset cache and checkpoints. Env knobs: IMAGE, GPUS, HOURS, STAGES, INSERTIVE, RECEPTIVE, SEED, NUM_ENVS,
# MAX_ITER, LOGGER, LOG_PROJECT, RUN_NAME, FORCE, BUILD_ARGS, RESET_ENVS, NUM_RESETS, GIT_REPO, GIT_REF, JOB_NAME,
# WANDB_API_KEY, WANDB_ENTITY, YES, DRY_RUN.
set -euo pipefail

IMAGE="${IMAGE:-uwlab-omnireset:slim-4bfca44}"
GPUS="${GPUS:-1}"
HOURS="${HOURS:-24}"
STAGES="${STAGES:-assets,datasets,train}"
INSERTIVE="${INSERTIVE:-peg}"
RECEPTIVE="${RECEPTIVE:-peghole}"
SEED="${SEED:-42}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITER="${MAX_ITER:-3000}"
LOGGER="${LOGGER:-wandb}"
LOG_PROJECT="${LOG_PROJECT:-omnireset}"
JOB_NAME="${JOB_NAME:-g1d-${INSERTIVE}-${RECEPTIVE}-$(echo "${STAGES}" | tr ',' '-')-s${SEED}}"
GIT_REPO="${GIT_REPO:-https://github.com/gunnxx/UWLab.git}"
GIT_REF="${GIT_REF:-feature/forge-omnireset}"
CKPT_DISK="${CKPT_DISK:-blob-robotics-scus}"
CKPT_MOUNT="${CKPT_MOUNT:-/mnt/robotics}"
CODE=/workspace/uwlab

if [[ "${LOGGER}" == "wandb" && ",${STAGES}," == *",train,"* ]]; then
    if [[ -z "${WANDB_API_KEY:-}" && -f "${HOME}/.zshenv" ]]; then
        WANDB_API_KEY="$(grep -E '^(export )?WANDB_API_KEY=' "${HOME}/.zshenv" | tail -1 | sed -E 's/^(export )?WANDB_API_KEY=//; s/^["'"'"']//; s/["'"'"']$//')"
    fi
    [[ -n "${WANDB_API_KEY:-}" ]] || { echo "LOGGER=wandb but no WANDB_API_KEY in env or ~/.zshenv" >&2; exit 1; }
fi

# knobs forwarded verbatim into the job's environment
FWD=(STAGES INSERTIVE RECEPTIVE SEED NUM_ENVS MAX_ITER LOGGER LOG_PROJECT GPUS)
for v in RUN_NAME FORCE BUILD_ARGS RESET_ENVS NUM_RESETS GRASP_ENVS NUM_GRASPS EXTRA SYNC_MINUTES CACHE CKPT_DEST; do
    [[ -n "${!v:-}" ]] && FWD+=("$v")
done
ENVSTR=""
for v in "${FWD[@]}"; do ENVSTR+="${v}=$(printf '%q' "${!v}") "; done

PREP="rm -rf ${CODE} && git clone -q --depth 1 -b ${GIT_REF} ${GIT_REPO} ${CODE} && \
ln -s /isaac-sim ${CODE}/_isaac_sim && ln -s /opt/uwlab-deps/_isaaclab ${CODE}/_isaaclab && \
cd ${CODE} && git log -1 --oneline"
ENTRY="uwlab-init-cache; ${PREP} && ${ENVSTR}bash docker/forge/g1d_pipeline.sh"

ARGS=(--name "${JOB_NAME}" --image "${IMAGE}" --gpu "${GPUS}" --max-duration "${HOURS}"
      --disk-mount "${CKPT_DISK}:${CKPT_MOUNT}"
      -m "OmniReset G1-D/Dex1 ${INSERTIVE}/${RECEPTIVE}: ${STAGES} (${GPUS}x${NUM_ENVS} envs, seed ${SEED}, ${GIT_REF}@fork)"
      --entrypoint "${ENTRY}")
if [[ "${LOGGER}" == "wandb" && ",${STAGES}," == *",train,"* ]]; then
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
