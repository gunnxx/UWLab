#!/usr/bin/env bash
# Clone or update the UW Lab working copy on the Forge code disk. Runs INSIDE Forge
# (a notebook terminal, or a job's --setup-command).
#
#   bash code_sync.sh                 # clone if empty, else fetch + fast-forward current branch
#   bash code_sync.sh main            # clone if empty, else fetch + checkout origin/main
#   bash code_sync.sh feature/x       # any branch
#   bash code_sync.sh 2b40afb         # or a commit (detached)
#
# Env: UWLAB_REPO (default https://github.com/gunnxx/UWLab.git, the working fork)
#      UWLAB_PATH (default /workspace/uwlab)
#
# First-time bootstrap on an empty disk, from a notebook or job that mounts it
# (the script is not on the disk yet, so fetch it from the repo):
#   wget -qO- https://raw.githubusercontent.com/gunnxx/UWLab/feature/forge-omnireset/docker/forge/code_sync.sh | bash -s -- feature/forge-omnireset
#
# Private fork: pass a token in the URL, e.g.
#   UWLAB_REPO=https://<user>:<token>@github.com/<user>/UWLab.git bash code_sync.sh my-branch
# or add --env GIT_TOKEN=... to the job and set UWLAB_REPO accordingly.
set -euo pipefail

REPO="${UWLAB_REPO:-https://github.com/gunnxx/UWLab.git}"
DEST="${UWLAB_PATH:-/workspace/uwlab}"
REF="${1:-}"

# On shared (blob) storage git is slow and has no locking: never run two syncs at once.
LOCK="${DEST}.sync.lock"
mkdir -p "$(dirname "${DEST}")"
exec 9>"${LOCK}"
# blobfuse may not support flock; treat that as "no lock available" and continue.
if ! flock -w 600 9 2>/dev/null; then
    echo "[code_sync] warning: could not take ${LOCK} (held by another sync, or locking unsupported on this mount)" >&2
fi

if [[ ! -d "${DEST}/.git" ]]; then
    if [[ -n "$(ls -A "${DEST}" 2>/dev/null)" ]]; then
        echo "[code_sync] ${DEST} is non-empty but not a git repo; refusing to clone over it" >&2
        exit 1
    fi
    echo "[code_sync] cloning ${REPO} -> ${DEST}"
    git clone "${REPO}" "${DEST}"
    [[ -n "${REF}" ]] && git -C "${DEST}" checkout "${REF}"
else
    echo "[code_sync] updating ${DEST}"
    if [[ -n "$(git -C "${DEST}" status --porcelain)" ]]; then
        echo "[code_sync] working tree has local changes; refusing to move it. Commit/stash them first:" >&2
        git -C "${DEST}" status --short >&2
        exit 1
    fi
    git -C "${DEST}" fetch --prune origin
    if [[ -n "${REF}" ]]; then
        if git -C "${DEST}" show-ref --verify --quiet "refs/remotes/origin/${REF}"; then
            git -C "${DEST}" checkout -B "${REF}" "origin/${REF}"
        else
            git -C "${DEST}" checkout --detach "${REF}"
        fi
    else
        git -C "${DEST}" pull --ff-only
    fi
fi

# Isaac Sim / Isaac Lab symlinks so ./uwlab.sh keeps working from the disk copy.
[[ -e "${DEST}/_isaac_sim" ]] || ln -s /isaac-sim "${DEST}/_isaac_sim"
[[ -e "${DEST}/_isaaclab" ]]  || ln -s /opt/uwlab-deps/_isaaclab "${DEST}/_isaaclab"

echo "[code_sync] now at $(git -C "${DEST}" log -1 --format='%h %s')"
