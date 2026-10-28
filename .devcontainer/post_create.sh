#!/usr/bin/env bash
# Dev Container lifecycle hook (see devcontainer.json). Runs inside the container as root.
#
#   post_create.sh          after the container is created: everything below
#   post_create.sh --start  on every start: cache dirs, symlinks, ownership fix only
#
# Mirrors the preparation docker/forge/run_local.sh does before a command: Isaac Sim cache dirs,
# the two gitignored symlinks the bind mount hides, then the VS Code settings (Pylance extraPaths
# for Isaac Sim's Kit extensions, Isaac Lab and the uwlab extensions) and a git exception for the
# bind-mounted repository, which is owned by the host user, not root.
set -euo pipefail

CODE="${UWLAB_PATH:-/workspace/uwlab}"
cd "${CODE}"

uwlab-init-cache
ln -sfn /isaac-sim "${CODE}/_isaac_sim"
ln -sfn /opt/uwlab-deps/_isaaclab "${CODE}/_isaaclab"

# Files the container created in the working copy (logs, generated assets, .vscode/settings.json)
# go back to the host user, whose uid the bind mount preserves on the repository root.
fix_owner() {
    local owner
    owner="$(stat -c '%u:%g' "${CODE}")"
    [[ "${owner}" == "0:0" ]] && return 0
    find "${CODE}" -xdev \( -path "${CODE}/.git" -prune \) -o \( ! -user "${owner%%:*}" -print0 \) 2>/dev/null \
        | xargs -0 -r chown -h "${owner}"
}

if [[ "${1:-}" == "--start" ]]; then
    fix_owner
    exit 0
fi

# git refuses a repository owned by another user ("dubious ownership").
git config --system --add safe.directory "${CODE}"
git config --system --add safe.directory "${CODE}/.git"

# .vscode/settings.json (+ launch.json when absent) from the templates in .vscode/tools; the
# interpreter stays ${workspaceFolder}/_isaac_sim/python.sh, which the symlink above resolves.
/isaac-sim/python.sh .vscode/tools/setup_vscode.py

fix_owner
echo "[devcontainer] ready: $(/isaac-sim/python.sh -c 'import isaaclab, uwlab_tasks; print("isaaclab", isaaclab.__file__); print("uwlab_tasks", uwlab_tasks.__file__)' 2>/dev/null | tr '\n' ' ')"
