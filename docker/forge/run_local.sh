#!/usr/bin/env bash
# Run the Forge image on this machine with the same layout a Forge job sees, so a change can be
# smoke-tested locally before it is submitted:
#   * this working copy is bind-mounted over /workspace/uwlab (the editable installs resolve to it),
#   * ${LOCAL_ROOT}/robotics stands in for the team blob disk at /mnt/robotics (asset/dataset cache,
#     checkpoint sync destination), so g1d_pipeline.sh runs with its default CACHE / CKPT_DEST,
#   * ${LOCAL_ROOT}/isaac-cache persists the Isaac Sim shader/Kit caches between runs.
#
#   docker/forge/run_local.sh                 # interactive bash in the container
#   docker/forge/run_local.sh smoke           # end-to-end G1-D pipeline at smoke-test sizes (~minutes)
#   docker/forge/run_local.sh smoke-train     # only the train stage, from the cached assets/datasets
#   docker/forge/run_local.sh python -c 'import uwlab_tasks'      # any command
#
# Knobs: IMAGE (default uwlab-omnireset:local, build with `docker build -f docker/forge/Dockerfile
# -t uwlab-omnireset:local docker/forge`), LOCAL_ROOT (default ~/.cache/uwlab-local), GPUS (docker
# --gpus value, default all), plus every g1d_pipeline.sh knob (STAGES, INSERTIVE, RECEPTIVE, NUM_ENVS,
# MAX_ITER, SEED, LOGGER, RESET_ENVS, NUM_RESETS, GRASP_ENVS, NUM_GRASPS, FORCE, EXTRA, ...), which
# are forwarded when set; ROBOT=g1_dex3 runs the standing G1 + Dex3 pipeline instead of the G1-D + Dex1.
# WANDB_API_KEY is forwarded when set.
#
# Rendering: when DISPLAY is set the X socket is passed through, so commands run with --viz kit
# (Isaac Lab 3.0 is headless otherwise) open the Kit window on this desktop (run `xhost +si:localuser:root` once per login so the root
# user inside the container may connect).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-uwlab-omnireset:local}"
LOCAL_ROOT="${LOCAL_ROOT:-${HOME}/.cache/uwlab-local}"
GPUS="${GPUS:-all}"
CODE=/workspace/uwlab

mkdir -p "${LOCAL_ROOT}/robotics" "${LOCAL_ROOT}/isaac-cache"

# Smoke-test sizes: tiny datasets and a handful of PPO iterations, TensorBoard only.
# 8 reset states per set: the two "EE grasped" sets succeed in well under 1% of episodes (2026-10-26), so larger
# targets turn a smoke test into an hour-long run.
SMOKE=(RESET_ENVS=256 NUM_RESETS=8 GRASP_ENVS=128 NUM_GRASPS=32 NUM_ENVS=64 MAX_ITER=5 LOGGER=tensorboard)
case "${1:-}" in
    smoke)       shift; MODE=pipeline; SMOKE+=(STAGES=assets,datasets,train) ;;
    smoke-train) shift; MODE=pipeline; SMOKE+=(STAGES=train) ;;
    *)           MODE=cmd ;;
esac

# g1d_pipeline.sh knobs set in the caller's environment win over the smoke defaults.
FWD=()
for kv in "${SMOKE[@]}"; do
    v="${kv%%=*}"
    [[ "${MODE}" == pipeline && -z "${!v:-}" ]] && FWD+=(-e "${kv}")
done
for v in ROBOT STAGES INSERTIVE RECEPTIVE SEED NUM_ENVS MAX_ITER LOGGER LOG_PROJECT RUN_NAME FORCE BUILD_ARGS \
         RESET_ENVS NUM_RESETS GRASP_ENVS NUM_GRASPS RESET_TYPES AUX_FROM_CACHE RESET_SYNC_S EXTRA SYNC_MINUTES \
         CACHE CKPT_DEST DEBUG_ENVS DEBUG_STEPS WANDB_API_KEY WANDB_ENTITY; do
    [[ -n "${!v:-}" ]] && FWD+=(-e "${v}=${!v}")
done
[[ "${MODE}" == pipeline ]] && FWD+=(-e GPUS="${GPUS/all/1}")

# Same preparation a Forge job does after cloning: cache dirs, the two symlinks the mount hides
# (both are gitignored). Files the container writes into the working copy are handed back to the
# host user at exit.
PREP="uwlab-init-cache; ln -sfn /isaac-sim ${CODE}/_isaac_sim; ln -sfn /opt/uwlab-deps/_isaaclab ${CODE}/_isaaclab; cd ${CODE}"
FIX_OWNER="chown -R $(id -u):$(id -g) ${CODE}/logs ${CODE}/source/uwlab_assets/data/Robots/Unitree ${CODE}/source/uwlab_assets/data/Datasets 2>/dev/null || true"

if [[ "${MODE}" == pipeline ]]; then
    CMD="${PREP}; trap '${FIX_OWNER}' EXIT; bash docker/forge/g1d_pipeline.sh"
elif [[ $# -eq 0 ]]; then
    CMD="${PREP}; trap '${FIX_OWNER}' EXIT; exec bash"
else
    CMD="${PREP}; trap '${FIX_OWNER}' EXIT; $(printf '%q ' "$@")"
fi

TTY=()
[[ -t 0 && -t 1 ]] && TTY=(-it)

X11=()
[[ -n "${DISPLAY:-}" && -d /tmp/.X11-unix ]] && X11=(-e "DISPLAY=${DISPLAY}" -v /tmp/.X11-unix:/tmp/.X11-unix)

exec docker run --rm "${TTY[@]}" --gpus "${GPUS}" --network host --ipc host \
    -v "${REPO}:${CODE}" \
    -v "${LOCAL_ROOT}/robotics:/mnt/robotics" \
    -v "${LOCAL_ROOT}/isaac-cache:/tmp/isaac-cache" \
    "${X11[@]}" "${FWD[@]}" "${IMAGE}" bash -c "${CMD}"
