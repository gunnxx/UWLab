#!/usr/bin/env bash
# Snapshot this working copy and upload it to the team disk, so Forge jobs run exactly
# what is on your laptop (uncommitted edits included) without an image rebuild or a push.
#
#   docker/forge/push_code.sh                 # upload via the notebook named $NOTEBOOK
#   docker/forge/push_code.sh g1d-logs        # via another running SSH notebook that mounts the disk
#
# What is shipped: every tracked file plus untracked files that are not gitignored (so
# logs/, outputs/, caches and datasets stay out), ~22 MB compressed. `.git` is not
# shipped; a `.forge-code-stamp` file records commit, branch, dirty files and the tree hash.
# The tarball lands at ${REMOTE_DIR}/${TAR_NAME} (written to .tmp and renamed, so a job
# starting mid-upload never sees a partial file). Jobs extract it to local scratch at start
# (submit_train.sh CODE_SOURCE=tar, the default).
#
# Transport: an SSH notebook that mounts the disk. A CPU-only notebook lives up to 168 h:
#   forge notebook create --name uwlab-code-push --type ssh --gpu 0 --duration 168 \
#       --disk-mount blob-robotics-scus:/mnt/robotics -m "UW Lab code upload channel (push_code.sh)"
#
# Env: NOTEBOOK (uwlab-code-push), REMOTE_DIR (/mnt/robotics/twguntara/code), TAR_NAME (uwlab-src.tar.gz),
#      DRY_RUN=1 builds the tarball in the current directory (run it outside the repo) and stops before uploading.
set -euo pipefail

NOTEBOOK="${1:-${NOTEBOOK:-uwlab-code-push}}"
REMOTE_DIR="${REMOTE_DIR:-/mnt/robotics/twguntara/code}"
TAR_NAME="${TAR_NAME:-uwlab-src.tar.gz}"

ORIG_PWD="${PWD}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
# shellcheck source=_lib.sh
source "${HERE}/_lib.sh"
cd "${REPO_ROOT}"

TMP="$(mktemp -d)"; trap 'rm -rf "${TMP}" "${REPO_ROOT}/.forge-code-stamp"' EXIT
# local bookkeeping files never enter git
for f in .forge-code-stamp .forge-last-push; do
    grep -qxF "$f" .git/info/exclude 2>/dev/null || echo "$f" >> .git/info/exclude
done
HASH="$(uwlab_tree_hash .)"
DIRTY="$(git status --porcelain | wc -l | tr -d ' ')"
STAMP="commit $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD)), ${DIRTY} uncommitted change(s), tree ${HASH}, pushed $(date '+%Y-%m-%d %H:%M %Z') from $(hostname -s)"
printf '%s\n' "${STAMP}" > .forge-code-stamp      # removed again on exit

# file list: tracked + untracked-not-ignored, existing regular files only
git ls-files -co --exclude-standard -z \
    | while IFS= read -r -d '' f; do [[ -f "$f" ]] && printf '%s\0' "$f"; done > "${TMP}/files"
printf '.forge-code-stamp\0' >> "${TMP}/files"
N=$(tr -cd '\0' < "${TMP}/files" | wc -c | tr -d ' ')
COPYFILE_DISABLE=1 tar -czf "${TMP}/${TAR_NAME}" --null -T "${TMP}/files"
SIZE_MB=$(( $(stat -f %z "${TMP}/${TAR_NAME}" 2>/dev/null || stat -c %s "${TMP}/${TAR_NAME}") / 1048576 ))
echo "[push_code] ${N} files, ${SIZE_MB} MB: ${STAMP}"
(( SIZE_MB < 300 )) || { echo "[push_code] tarball is ${SIZE_MB} MB; something untracked and large is in the tree (git status)" >&2; exit 1; }
LOCAL_MD5="$( (md5 -q "${TMP}/${TAR_NAME}" 2>/dev/null || md5sum "${TMP}/${TAR_NAME}" | cut -d' ' -f1) )"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    cp "${TMP}/${TAR_NAME}" "${ORIG_PWD}/${TAR_NAME}"; echo "[push_code] DRY_RUN=1: tarball left at ${ORIG_PWD}/${TAR_NAME} (md5 ${LOCAL_MD5}), nothing uploaded"; exit 0
fi
uwlab_notebook_ssh "${NOTEBOOK}"
echo "[push_code] uploading to ${UWLAB_SSH_HOST}:${REMOTE_DIR}/${TAR_NAME}"
ssh "${UWLAB_SSH_OPTS[@]}" "${UWLAB_SSH_HOST}" "mkdir -p '${REMOTE_DIR}'"
STAMP_NAME="${TAR_NAME%.tar.gz}.stamp"
scp -q "${UWLAB_SSH_OPTS[@]}" "${TMP}/${TAR_NAME}" "${UWLAB_SSH_HOST}:${REMOTE_DIR}/${TAR_NAME}.tmp"
scp -q "${UWLAB_SSH_OPTS[@]}" .forge-code-stamp "${UWLAB_SSH_HOST}:${REMOTE_DIR}/${STAMP_NAME}.tmp"
REMOTE_MD5="$(ssh "${UWLAB_SSH_OPTS[@]}" "${UWLAB_SSH_HOST}" "cd '${REMOTE_DIR}' && mv ${TAR_NAME}.tmp ${TAR_NAME} && mv ${STAMP_NAME}.tmp ${STAMP_NAME} && md5sum ${TAR_NAME} | cut -d' ' -f1")"
[[ "${LOCAL_MD5}" == "${REMOTE_MD5}" ]] || { echo "[push_code] checksum mismatch after upload (${LOCAL_MD5} vs ${REMOTE_MD5})" >&2; exit 1; }

# remember what was pushed so submit_train.sh can warn when the tree moved on
printf 'hash=%s\nstamp=%s\n' "${HASH}" "${STAMP}" > "${REPO_ROOT}/.forge-last-push"
echo "[push_code] ok, md5 ${REMOTE_MD5}. Submit with CODE_SOURCE=tar (default): docker/forge/submit_train.sh <insertive> <receptive>"
