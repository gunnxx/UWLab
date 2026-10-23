#!/usr/bin/env bash
# Push this local working copy (including .git, so `git pull` keeps working on the
# disk) to the Forge code disk through an SSH notebook that mounts it.
#
#   forge notebook create --name uwlab-dev --type ssh --gpu 0 \
#       --custom-image uwlab-omnireset --disk-mount uwlab-code:/workspace/uwlab \
#       -m "UW Lab code disk editor"
#   forge notebook get uwlab-dev          # read the SSH host/port
#   docker/forge/sync_code.sh root@<host> -p <port>
#
# Logs, caches, datasets and the Isaac Sim / Isaac Lab symlinks are excluded; the
# image provides those paths. Set NO_GIT=1 to leave .git out (faster on blob).
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <ssh-target> [extra ssh options...]" >&2
    exit 1
fi

TARGET="$1"; shift
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_PATH="${UWLAB_REMOTE_PATH:-/workspace/uwlab}"
SSH_CMD="ssh $*"

EXCLUDES=(
    --exclude '_isaac_sim' --exclude '_isaaclab'
    --exclude 'logs/' --exclude 'outputs/' --exclude 'wandb/'
    --exclude 'Datasets/' --exclude 'datasets/'
    --exclude '__pycache__/' --exclude '*.egg-info/'
    --exclude '.vscode/' --exclude 'env_uwlab/' --exclude 'docs/_build/'
)
[[ "${NO_GIT:-0}" == "1" ]] && EXCLUDES+=(--exclude '.git/')

rsync -az --delete --info=progress2 -e "${SSH_CMD}" "${EXCLUDES[@]}" \
    "${REPO_ROOT}/" "${TARGET}:${REMOTE_PATH}/"

echo "synced ${REPO_ROOT} -> ${TARGET}:${REMOTE_PATH}"
