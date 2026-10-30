#!/usr/bin/env bash
# Runs INSIDE a Forge job (after the fork is cloned to ${CODE}): builds the robot assets, records the OmniReset
# datasets for that robot, and trains stage 1. Each stage is cached on the shared blob disk so later jobs can
# skip it. Two robots share the pipeline; ROBOT picks the asset builder, the task ids and the cache folder:
#
#   ROBOT=g1d_dex1 (default)       Unitree G1-D (wheeled) + Dex1-1 gripper
#   ROBOT=g1_dex3                  Unitree G1 (standing humanoid) + Dex3 hand
#
#   STAGES=assets,datasets,train   comma list; any subset, in this order
#   INSERTIVE=peg RECEPTIVE=peghole
#   CACHE=/mnt/robotics/twguntara/omni-reset/g1d   blob cache root ("" disables caching)
#   FORCE=1                        rebuild/re-record even if cached
#   RESET_TYPES="A B"              record only these reset-state families (default: all four)
#   AUX_FROM_CACHE=1               take partial assemblies, grasps and earlier families from the cache (one job per family)
#   STAGES=assets,bench BENCH_ENVS="4096 16384"   step-rate / memory benchmark of a reset-state env
#   GPUS NUM_ENVS SEED MAX_ITER LOGGER LOG_PROJECT RUN_NAME    training knobs (see submit_train.sh)
#   RESET_ENVS=2048 NUM_RESETS=10000 GRASP_ENVS=8192 NUM_GRASPS=1000
#   PA_ENVS=32 PA_TRAJ=32          partial-assembly recorder: environments and trajectories
#   GRASP_ARGS="env.events.grasp_sampling.params.standoffs_within_clearance=true"   extra Hydra overrides for record_grasps.py
#   RESET_SYNC_S=300               while recording reset states, mirror the files to the cache this often (seconds)
#   BUILD_ARGS="--waist-pitch 0.1"  extra flags for build_urdf.py
set -euo pipefail

CODE="${CODE:-/workspace/uwlab}"
cd "${CODE}"
STAGES="${STAGES:-assets,datasets,train}"
INSERTIVE="${INSERTIVE:-peg}"
RECEPTIVE="${RECEPTIVE:-peghole}"
ROBOT="${ROBOT:-g1d_dex1}"
FORCE="${FORCE:-0}"
UNITREE_ROS="${UNITREE_ROS:-/workspace/unitree_ros}"
# per-robot names: asset folder and files, dataset folder, task ids, the gripper body the grasps are recorded for
case "${ROBOT}" in
    g1d_dex1)
        CACHE="${CACHE-/mnt/robotics/twguntara/omni-reset/g1d}"
        ASSET_DIR="${CODE}/source/uwlab_assets/data/Robots/Unitree/G1dDex1"
        DATA_DIR="${CODE}/source/uwlab_assets/data/Datasets/OmniReset_g1d"
        ROBOT_USD=g1d_dex1_right_arm; GRIPPER_USD=dex1_gripper; GRIPPER_BODY=right_dex1_base_link
        TASK_PREFIX=OmniReset-G1dDex1; GRASP_TASK=OmniReset-Dex1-GraspSampling-v0
        EXPERIMENT=g1d_dex1_omnireset_agent ;;
    g1_dex3)
        CACHE="${CACHE-/mnt/robotics/twguntara/omni-reset/g1_dex3}"
        ASSET_DIR="${CODE}/source/uwlab_assets/data/Robots/Unitree/G1Dex3"
        DATA_DIR="${CODE}/source/uwlab_assets/data/Datasets/OmniReset_g1_dex3"
        ROBOT_USD=g1_dex3_right_arm; GRIPPER_USD=dex3_hand; GRIPPER_BODY=right_hand_palm_link
        TASK_PREFIX=OmniReset-G1Dex3; GRASP_TASK=OmniReset-Dex3-GraspSampling-v0
        EXPERIMENT=g1_dex3_omnireset_agent ;;
    *) echo "ROBOT must be g1d_dex1 or g1_dex3, got ${ROBOT}" >&2; exit 1 ;;
esac
RESET_ENVS="${RESET_ENVS:-2048}"
NUM_RESETS="${NUM_RESETS:-10000}"
GRASP_ENVS="${GRASP_ENVS:-8192}"
NUM_GRASPS="${NUM_GRASPS:-1000}"
PA_ENVS="${PA_ENVS:-32}"
PA_TRAJ="${PA_TRAJ:-32}"
# reset-state families to record (space separated); split them across jobs to use several GPUs in parallel
RESET_TYPES="${RESET_TYPES:-ObjectAnywhereEEAnywhere ObjectRestingEEGrasped ObjectAnywhereEEGrasped ObjectPartiallyAssembledEEGrasped}"
VIS=(--visualizer none)

log() { echo "[${ROBOT} $(date +%H:%M:%S)] $*"; }
has_stage() { [[ ",${STAGES}," == *",$1,"* ]]; }

# Forge keeps only the tail of a job's log; persist the full transcript on the blob disk.
if [[ -n "${CACHE}" ]]; then
    mkdir -p "${CACHE}/logs"
    LOGFILE="${CACHE}/logs/$(date +%Y%m%d-%H%M%S)-${STAGES//,/-}.log"
    exec > >(tee -a "${LOGFILE}") 2>&1
    log "full log: ${LOGFILE}"
fi

# ------------------------------------------------------------------------------------------------------ assets
if has_stage assets; then
    if [[ "${FORCE}" != 1 && -n "${CACHE}" && -f "${CACHE}/assets/${ROBOT_USD}.usd" ]]; then
        log "assets: using cached ${CACHE}/assets"
        mkdir -p "${ASSET_DIR}" && rsync -a "${CACHE}/assets/" "${ASSET_DIR}/"
    else
        rm -rf "${UNITREE_ROS}"
        git clone -q --depth 1 --filter=blob:none --no-checkout https://github.com/unitreerobotics/unitree_ros.git "${UNITREE_ROS}"
        # the importer appends _1, _2, ... when its output folder exists; start from a clean slate
        rm -rf "${ASSET_DIR}/${ROBOT_USD}" "${ASSET_DIR}/${ROBOT_USD}"_* "${ASSET_DIR}/${GRIPPER_USD}" "${ASSET_DIR}/${GRIPPER_USD}"_* "${ASSET_DIR}"/*.usd
        if [[ "${ROBOT}" == g1d_dex1 ]]; then
            log "assets: cloning unitree_ros (G1-D description + Dex1-1 meshes)"
            (cd "${UNITREE_ROS}" && git sparse-checkout init --no-cone && git sparse-checkout set \
                '/robots/g1_d_description/*' '/robots/g1_d_description/meshes/*' \
                '/robots/g1_description/g1_29dof_mode_15_with_dex1_1.urdf' \
                '/robots/g1_description/meshes/Dex1_*' '/robots/g1_description/meshes/dex1_col_*' && git checkout -q)
            log "assets: building URDFs"
            python -m uwlab_assets.robots.unitree_g1d_dex1.build_urdf --unitree-ros "${UNITREE_ROS}" --out-dir "${ASSET_DIR}" ${BUILD_ARGS:-}
        else
            # the G1 URDF is vendored with the package (the robot's own, the gravity solver's); only the meshes come from unitree_ros
            log "assets: cloning unitree_ros (G1 meshes)"
            (cd "${UNITREE_ROS}" && git sparse-checkout init --no-cone && git sparse-checkout set '/robots/g1_description/meshes/*' && git checkout -q)
            log "assets: building URDFs"
            python -m uwlab_assets.robots.unitree_g1_dex3.build_urdf --meshes "${UNITREE_ROS}/robots/g1_description/meshes" --out-dir "${ASSET_DIR}" ${BUILD_ARGS:-}
        fi
        # convert inside a Kit app (scripts/tools/convert_urdf.py imports pxr before launching and breaks here)
        log "assets: converting to USD (robot, fixed base)"
        python scripts_v2/tools/convert_urdf_kit.py "${ASSET_DIR}/${ROBOT_USD}.urdf" "${ASSET_DIR}/${ROBOT_USD}.usd" \
            --fix_base --joint_stiffness 0 --joint_damping 0 "${VIS[@]}"
        log "assets: converting to USD (gripper, fixed base: the sampler moves it by writing the root pose)"
        python scripts_v2/tools/convert_urdf_kit.py "${ASSET_DIR}/${GRIPPER_USD}.urdf" "${ASSET_DIR}/${GRIPPER_USD}.usd" \
            --fix_base --joint_stiffness 0 --joint_damping 0 "${VIS[@]}"
        [[ -f "${ASSET_DIR}/${ROBOT_USD}.usd" && -f "${ASSET_DIR}/${GRIPPER_USD}.usd" ]] || { log "assets: USD conversion failed"; exit 1; }
        # the USD must carry the URDF's link masses (the alignment with the robot's gravity solver rests on it)
        log "assets: checking the USD inertials against the URDF"
        python scripts_v2/tools/check_usd_inertials.py "${ASSET_DIR}/${ROBOT_USD}.urdf" "${ASSET_DIR}/${ROBOT_USD}.usd"
        python scripts_v2/tools/check_usd_inertials.py "${ASSET_DIR}/${GRIPPER_USD}.urdf" "${ASSET_DIR}/${GRIPPER_USD}.usd"
        ls -la "${ASSET_DIR}"
        if [[ -n "${CACHE}" ]]; then mkdir -p "${CACHE}/assets" && rsync -a "${ASSET_DIR}/" "${CACHE}/assets/" && log "assets: cached"; fi
    fi
fi

# ------------------------------------------------------------------------------------------------------- debug
if has_stage debug; then
    log "debug: probing the grasp environment (${INSERTIVE})"
    [[ -f "${ASSET_DIR}/${GRIPPER_USD}.usd" ]] || { mkdir -p "${ASSET_DIR}" && rsync -a "${CACHE}/assets/" "${ASSET_DIR}/"; }
    python scripts_v2/tools/debug_grasp_env.py --task "${GRASP_TASK}" --num_envs "${DEBUG_ENVS:-16}" \
        --steps "${DEBUG_STEPS:-300}" "${VIS[@]}" env.scene.object="${INSERTIVE}" || log "debug: probe exited with $?"
fi

# ---------------------------------------------------------------------------------------------------- datasets
if has_stage datasets; then
    # dataset folder name = USD parent-directory names of the two objects, sorted by the plain name and carrying a
    # scale suffix for a scaled variant (see mdp/utils.pair_dataset_name)
    PAIR="$(python - "${INSERTIVE}" "${RECEPTIVE}" <<'PY'
import sys
names = {"peg": "Peg", "peghole": "PegHole", "big-peg": "Peg_x2", "big-peghole": "PegHole_x2", "peg-long": "Peg_x1x1x2",
         "fbleg": "SquareLeg", "fbtabletop": "SquareTableTop",
         "fbdrawerbottom": "DrawerBottom", "fbdrawerbox": "DrawerBox", "cupcake": "CupCake", "plate": "Plate",
         "rectangle": "Rectangle", "wall": "Wall"}
ins, rec = sys.argv[1], sys.argv[2]
a = names.get(ins, "InsertiveCube" if ins == "cube" else None)
b = names.get(rec, "ReceptiveCube" if rec == "cube" else None)
if a is None or b is None:
    raise SystemExit(f"unknown object variant: {ins} / {rec}")
print("__".join(sorted([a, b], key=lambda n: n.split("_x")[0])))
PY
)"
    DONE_MARK="Resets/${PAIR}/resets_${RESET_TYPES##* }.pt"
    if [[ "${FORCE}" != 1 && -n "${CACHE}" && -f "${CACHE}/datasets/${DONE_MARK}" ]]; then
        log "datasets: using cached ${CACHE}/datasets (${PAIR})"
        mkdir -p "${DATA_DIR}" && rsync -a "${CACHE}/datasets/" "${DATA_DIR}/"
    else
        OBJ=(env.scene.insertive_object="${INSERTIVE}" env.scene.receptive_object="${RECEPTIVE}")
        if [[ "${AUX_FROM_CACHE:-0}" == 1 ]]; then
            # one job per family: take the partial assemblies, grasps and already recorded families from the cache
            # (a previous job recorded them) and record only ${RESET_TYPES}
            [[ -n "${CACHE}" && -d "${CACHE}/datasets" ]] || { log "AUX_FROM_CACHE=1 but no ${CACHE}/datasets"; exit 1; }
            mkdir -p "${DATA_DIR}" && rsync -a "${CACHE}/datasets/" "${DATA_DIR}/"
            compgen -G "${DATA_DIR}/Grasps/*/grasps.pt" > /dev/null || { log "no grasp file in the cache"; exit 1; }
            log "datasets: partial assemblies and grasps taken from ${CACHE}/datasets"
        else
            log "datasets: partial assemblies (${PAIR})"
            python scripts_v2/tools/record_partial_assemblies.py --task OmniReset-PartialAssemblies-v0 \
                --num_envs "${PA_ENVS}" --num_trajectories "${PA_TRAJ}" --dataset_dir "${DATA_DIR}" "${VIS[@]}" "${OBJ[@]}"
            log "datasets: ${ROBOT} grasps on ${INSERTIVE}"
            python scripts_v2/tools/record_grasps.py --task "${GRASP_TASK}" \
                --num_envs "${GRASP_ENVS}" --num_grasps "${NUM_GRASPS}" --dataset_dir "${DATA_DIR}" \
                --gripper_body_name "${GRIPPER_BODY}" "${VIS[@]}" env.scene.object="${INSERTIVE}" ${GRASP_ARGS:-}
        fi
        # The recorder writes into the job's own file system, which disappears with the job: a max-duration kill
        # or a cancel used to lose every state recorded so far. Mirror the reset-state files to the cache every
        # RESET_SYNC_S seconds while recording (the recorder renames complete snapshots into place, so a copy is
        # never partial; *.tmp is the snapshot being written), then once more when the recording ends.
        sync_resets() {
            [[ -n "${CACHE}" && -d "${DATA_DIR}/Resets" ]] || return 0
            mkdir -p "${CACHE}/datasets/Resets" && rsync -a --exclude '*.tmp' "${DATA_DIR}/Resets/" "${CACHE}/datasets/Resets/"
        }
        SYNC_PID=""
        if [[ -n "${CACHE}" ]]; then
            ( while sleep "${RESET_SYNC_S:-300}"; do sync_resets || true; done ) & SYNC_PID=$!
            trap 'kill ${SYNC_PID} 2>/dev/null || true; sync_resets || true' EXIT
        fi
        for RT in ${RESET_TYPES}; do
            log "datasets: reset states ${RT}"
            python scripts_v2/tools/record_reset_states.py --task "${TASK_PREFIX}-${RT}-v0" \
                --num_envs "${RESET_ENVS}" --num_reset_states "${NUM_RESETS}" --dataset_dir "${DATA_DIR}" "${VIS[@]}" "${OBJ[@]}"
            sync_resets && log "datasets: ${RT} mirrored to the cache"
        done
        [[ -n "${SYNC_PID}" ]] && { kill "${SYNC_PID}" 2>/dev/null || true; trap - EXIT; }
        find "${DATA_DIR}" -name '*.pt' -exec ls -la {} \;
        if [[ -n "${CACHE}" ]]; then mkdir -p "${CACHE}/datasets" && rsync -a "${DATA_DIR}/" "${CACHE}/datasets/" && log "datasets: cached"; fi
    fi
fi

# ------------------------------------------------------------------------------------------------------- bench
# Step-rate / memory benchmark of a reset-state environment at several environment counts (debug_reset_env.py
# prints env-steps/s and the success funnel); the peak GPU memory is sampled while each probe runs.
if has_stage bench; then
    BENCH_TASK="${BENCH_TASK:-ObjectAnywhereEEGrasped}"
    BENCH_ENVS="${BENCH_ENVS:-4096 8192 16384}"
    BENCH_STEPS="${BENCH_STEPS:-100}"
    OBJ=(env.scene.insertive_object="${INSERTIVE}" env.scene.receptive_object="${RECEPTIVE}")
    if [[ -n "${CACHE}" && -d "${CACHE}/datasets" ]]; then mkdir -p "${DATA_DIR}" && rsync -a "${CACHE}/datasets/" "${DATA_DIR}/"; fi
    if ! compgen -G "${DATA_DIR}/Grasps/*/grasps.pt" > /dev/null; then
        log "bench: no grasp file cached, recording a small one"
        python scripts_v2/tools/record_grasps.py --task "${GRASP_TASK}" --num_envs 512 --num_grasps 200 \
            --dataset_dir "${DATA_DIR}" --gripper_body_name "${GRIPPER_BODY}" "${VIS[@]}" env.scene.object="${INSERTIVE}"
    fi
    for n in ${BENCH_ENVS}; do
        log "bench: ${BENCH_TASK} with ${n} envs, ${BENCH_STEPS} steps"
        python scripts_v2/tools/debug_reset_env.py --task "${TASK_PREFIX}-${BENCH_TASK}-v0" --num_envs "${n}" \
            --steps "${BENCH_STEPS}" --funnel --close_gripper "${VIS[@]}" "${OBJ[@]}" > "/tmp/bench_${n}.log" 2>&1 &
        BPID=$!; PEAK=0
        while kill -0 "${BPID}" 2>/dev/null; do
            sleep 5; M=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); (( M > PEAK )) && PEAK=$M
        done
        wait "${BPID}" && RC=0 || RC=$?
        log "bench: ${n} envs exit ${RC}, peak GPU memory ${PEAK} MiB"
        grep -E "\[timing\]|\[funnel\]|out of memory|RuntimeError" "/tmp/bench_${n}.log" || true
        [[ -n "${CACHE}" ]] && cp "/tmp/bench_${n}.log" "${CACHE}/logs/bench_${BENCH_TASK}_${n}.log"
    done
fi

# ------------------------------------------------------------------------------------------------------- train
if has_stage train; then
    GPUS="${GPUS:-1}"
    NUM_ENVS="${NUM_ENVS:-4096}"
    SEED="${SEED:-42}"
    MAX_ITER="${MAX_ITER:-3000}"
    LOGGER="${LOGGER:-wandb}"
    LOG_PROJECT="${LOG_PROJECT:-omnireset}"
    TASK="${TASK_PREFIX}-RelCartesianOSC-State-v0"
    RUN_NAME="${RUN_NAME:-${TASK}-${INSERTIVE}-${RECEPTIVE}-s${SEED}}"
    EXPERIMENT_DIR="logs/rsl_rl/${EXPERIMENT}"
    CKPT_DEST="${CKPT_DEST-/mnt/robotics/twguntara/omni-reset/${EXPERIMENT}}"
    SYNC_MINUTES="${SYNC_MINUTES:-10}"

    # the big peg draws its training mass from its own range (see the variants of rl_state_cfg.py)
    MASS=(); [[ "${INSERTIVE}" == big-peg ]] && MASS=(env.events.randomize_insertive_object_mass=big-peg)
    TRAIN=(python scripts/reinforcement_learning/rsl_rl/train.py --task "${TASK}" --num_envs "${NUM_ENVS}"
           --max_iterations "${MAX_ITER}" --logger "${LOGGER}" --log_project_name "${LOG_PROJECT}" --run_name "${RUN_NAME}"
           "${VIS[@]}" agent.seed="${SEED}" env.scene.insertive_object="${INSERTIVE}" env.scene.receptive_object="${RECEPTIVE}"
           "${MASS[@]}" ${EXTRA:-})
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
