#!/usr/bin/env bash
# Runs INSIDE a Forge job (after the fork is cloned to ${CODE}): builds the G1-D + Dex1-1 assets, records the
# OmniReset datasets for that robot, and trains stage 1. Each stage is cached on the shared blob disk so later
# jobs can skip it.
#
#   STAGES=assets,datasets,train   comma list; any subset, in this order
#   INSERTIVE=peg RECEPTIVE=peghole
#   CACHE=/mnt/robotics/twguntara/omni-reset/g1d   blob cache root ("" disables caching)
#   FORCE=1                        rebuild/re-record even if cached
#   GPUS NUM_ENVS SEED MAX_ITER LOGGER LOG_PROJECT RUN_NAME    training knobs (see submit_train.sh)
#   RESET_ENVS=2048 NUM_RESETS=10000 GRASP_ENVS=8192 NUM_GRASPS=1000
#   BUILD_ARGS="--waist-pitch 0.1"  extra flags for build_urdf.py
set -euo pipefail

CODE="${CODE:-/workspace/uwlab}"
cd "${CODE}"
STAGES="${STAGES:-assets,datasets,train}"
INSERTIVE="${INSERTIVE:-peg}"
RECEPTIVE="${RECEPTIVE:-peghole}"
CACHE="${CACHE-/mnt/robotics/twguntara/omni-reset/g1d}"
FORCE="${FORCE:-0}"
UNITREE_ROS="${UNITREE_ROS:-/workspace/unitree_ros}"
ASSET_DIR="${CODE}/source/uwlab_assets/data/Robots/Unitree/G1dDex1"
DATA_DIR="${CODE}/source/uwlab_assets/data/Datasets/OmniReset_g1d"
RESET_ENVS="${RESET_ENVS:-2048}"
NUM_RESETS="${NUM_RESETS:-10000}"
GRASP_ENVS="${GRASP_ENVS:-8192}"
NUM_GRASPS="${NUM_GRASPS:-1000}"
VIS=(--visualizer none)

log() { echo "[g1d $(date +%H:%M:%S)] $*"; }
has_stage() { [[ ",${STAGES}," == *",$1,"* ]]; }

# ------------------------------------------------------------------------------------------------------ assets
if has_stage assets; then
    if [[ "${FORCE}" != 1 && -n "${CACHE}" && -f "${CACHE}/assets/g1d_dex1_right_arm.usd" ]]; then
        log "assets: using cached ${CACHE}/assets"
        mkdir -p "${ASSET_DIR}" && rsync -a "${CACHE}/assets/" "${ASSET_DIR}/"
    else
        log "assets: cloning unitree_ros (G1-D description + Dex1-1 meshes)"
        rm -rf "${UNITREE_ROS}"
        git clone -q --depth 1 --filter=blob:none --no-checkout https://github.com/unitreerobotics/unitree_ros.git "${UNITREE_ROS}"
        (cd "${UNITREE_ROS}" && git sparse-checkout init --no-cone && git sparse-checkout set \
            '/robots/g1_d_description/*' '/robots/g1_d_description/meshes/*' \
            '/robots/g1_description/g1_29dof_mode_15_with_dex1_1.urdf' \
            '/robots/g1_description/meshes/Dex1_*' '/robots/g1_description/meshes/dex1_col_*' && git checkout -q)
        log "assets: building URDFs"
        python -m uwlab_assets.robots.unitree_g1d_dex1.build_urdf --unitree-ros "${UNITREE_ROS}" --out-dir "${ASSET_DIR}" ${BUILD_ARGS:-}
        # the converter uses the new launcher args; accept either spelling of the headless flag
        convert() { python scripts/tools/convert_urdf.py "$@" --visualizer none || python scripts/tools/convert_urdf.py "$@" --headless; }
        log "assets: converting to USD (robot, fixed base)"
        convert "${ASSET_DIR}/g1d_dex1_right_arm.urdf" "${ASSET_DIR}/g1d_dex1_right_arm.usd" --fix_base --joint_stiffness 0 --joint_damping 0
        log "assets: converting to USD (floating gripper)"
        convert "${ASSET_DIR}/dex1_gripper.urdf" "${ASSET_DIR}/dex1_gripper.usd" --joint_stiffness 0 --joint_damping 0
        test -f "${ASSET_DIR}/g1d_dex1_right_arm.usd" && test -f "${ASSET_DIR}/dex1_gripper.usd"
        ls -la "${ASSET_DIR}"
        if [[ -n "${CACHE}" ]]; then mkdir -p "${CACHE}/assets" && rsync -a "${ASSET_DIR}/" "${CACHE}/assets/" && log "assets: cached"; fi
    fi
fi

# ---------------------------------------------------------------------------------------------------- datasets
if has_stage datasets; then
    # dataset folder name = sorted USD parent-directory names of the two objects (see mdp/utils.compute_pair_dir)
    PAIR="$(python3 - "${INSERTIVE}" "${RECEPTIVE}" <<'PY'
import sys
names = {"peg": "Peg", "peghole": "PegHole", "fbleg": "SquareLeg", "fbtabletop": "SquareTableTop",
         "fbdrawerbottom": "DrawerBottom", "fbdrawerbox": "DrawerBox", "cupcake": "CupCake", "plate": "Plate",
         "rectangle": "Rectangle", "wall": "Wall"}
ins, rec = sys.argv[1], sys.argv[2]
a = names.get(ins, "InsertiveCube" if ins == "cube" else None)
b = names.get(rec, "ReceptiveCube" if rec == "cube" else None)
if a is None or b is None:
    raise SystemExit(f"unknown object variant: {ins} / {rec}")
print("__".join(sorted([a, b])))
PY
)"
    DONE_MARK="Resets/${PAIR}/resets_ObjectPartiallyAssembledEEGrasped.pt"
    if [[ "${FORCE}" != 1 && -n "${CACHE}" && -f "${CACHE}/datasets/${DONE_MARK}" ]]; then
        log "datasets: using cached ${CACHE}/datasets (${PAIR})"
        mkdir -p "${DATA_DIR}" && rsync -a "${CACHE}/datasets/" "${DATA_DIR}/"
    else
        OBJ=(env.scene.insertive_object="${INSERTIVE}" env.scene.receptive_object="${RECEPTIVE}")
        log "datasets: partial assemblies (${PAIR})"
        python scripts_v2/tools/record_partial_assemblies.py --task OmniReset-PartialAssemblies-v0 \
            --num_envs 10 --num_trajectories 10 --dataset_dir "${DATA_DIR}" "${VIS[@]}" "${OBJ[@]}"
        log "datasets: Dex1-1 grasps on ${INSERTIVE}"
        python scripts_v2/tools/record_grasps.py --task OmniReset-Dex1-GraspSampling-v0 \
            --num_envs "${GRASP_ENVS}" --num_grasps "${NUM_GRASPS}" --dataset_dir "${DATA_DIR}" \
            --gripper_body_name right_dex1_base_link "${VIS[@]}" env.scene.object="${INSERTIVE}"
        for RT in ObjectAnywhereEEAnywhere ObjectRestingEEGrasped ObjectAnywhereEEGrasped ObjectPartiallyAssembledEEGrasped; do
            log "datasets: reset states ${RT}"
            python scripts_v2/tools/record_reset_states.py --task "OmniReset-G1dDex1-${RT}-v0" \
                --num_envs "${RESET_ENVS}" --num_reset_states "${NUM_RESETS}" --dataset_dir "${DATA_DIR}" "${VIS[@]}" "${OBJ[@]}"
        done
        find "${DATA_DIR}" -name '*.pt' -exec ls -la {} \;
        if [[ -n "${CACHE}" ]]; then mkdir -p "${CACHE}/datasets" && rsync -a "${DATA_DIR}/" "${CACHE}/datasets/" && log "datasets: cached"; fi
    fi
fi

# ------------------------------------------------------------------------------------------------------- train
if has_stage train; then
    GPUS="${GPUS:-1}"
    NUM_ENVS="${NUM_ENVS:-4096}"
    SEED="${SEED:-42}"
    MAX_ITER="${MAX_ITER:-3000}"
    LOGGER="${LOGGER:-wandb}"
    LOG_PROJECT="${LOG_PROJECT:-omnireset}"
    TASK="OmniReset-G1dDex1-RelCartesianOSC-State-v0"
    RUN_NAME="${RUN_NAME:-${TASK}-${INSERTIVE}-${RECEPTIVE}-s${SEED}}"
    EXPERIMENT_DIR="logs/rsl_rl/g1d_dex1_omnireset_agent"
    CKPT_DEST="${CKPT_DEST-/mnt/robotics/twguntara/omni-reset/g1d_dex1_omnireset_agent}"
    SYNC_MINUTES="${SYNC_MINUTES:-10}"

    TRAIN=(python scripts/reinforcement_learning/rsl_rl/train.py --task "${TASK}" --num_envs "${NUM_ENVS}"
           --max_iterations "${MAX_ITER}" --logger "${LOGGER}" --log_project_name "${LOG_PROJECT}" --run_name "${RUN_NAME}"
           "${VIS[@]}" agent.seed="${SEED}" env.scene.insertive_object="${INSERTIVE}" env.scene.receptive_object="${RECEPTIVE}" ${EXTRA:-})
    if (( GPUS > 1 )); then
        TRAIN=(python -m torch.distributed.run --standalone --nnodes 1 --nproc_per_node "${GPUS}" "${TRAIN[@]:1}" --distributed)
    fi
    if [[ -n "${CKPT_DEST}" ]]; then
        mkdir -p "${CKPT_DEST}" "${EXPERIMENT_DIR}"
        SYNC="rsync -a --exclude wandb/ ${EXPERIMENT_DIR}/ ${CKPT_DEST}/"
        (while true; do sleep $((SYNC_MINUTES * 60)); ${SYNC} 2>/dev/null; done) &
        SYNC_PID=$!
    fi
    log "train: ${TRAIN[*]}"
    set +e; "${TRAIN[@]}"; rc=$?; set -e
    if [[ -n "${CKPT_DEST}" ]]; then kill "${SYNC_PID}" 2>/dev/null || true; ${SYNC}; log "train: synced ${EXPERIMENT_DIR} -> ${CKPT_DEST}"; ls "${CKPT_DEST}"; fi
    exit "${rc}"
fi
