# Shared helpers for the docker/forge scripts. Source, do not execute.

# Content hash of the working tree as push_code.sh ships it: every tracked file plus
# untracked-but-not-ignored files, deleted files skipped. Same on macOS and Linux.
uwlab_tree_hash() {
    local root="${1:-.}"
    git -C "${root}" ls-files -co --exclude-standard -z \
        | (cd "${root}" && while IFS= read -r -d '' f; do [[ -f "$f" ]] && printf '%s\0' "$f"; done) \
        | sort -z | (cd "${root}" && xargs -0 shasum -a 1) | shasum -a 1 | cut -c1-12
}

# "ssh" argument list for a Forge SSH notebook, resolved through `forge notebook ssh`.
# Forge prints `ssh ... -J bastion@host:22 root@nb-svc-...`; options given on the command
# line do not apply to a -J jump host, so the bastion gets its own options via ProxyCommand.
# Sets UWLAB_SSH_OPTS (array) and UWLAB_SSH_HOST (user@host).
uwlab_notebook_ssh() {
    local nb="$1" line jump
    line="$(forge notebook ssh "${nb}" 2>&1)" || { echo "forge notebook ssh ${nb}: ${line}" >&2; return 1; }
    UWLAB_SSH_HOST="$(grep -oE 'root@[^ ]+' <<<"${line}" | head -1)"
    jump="$(grep -oE '\-J [^ ]+' <<<"${line}" | awk '{print $2}')"
    [[ -n "${UWLAB_SSH_HOST}" ]] || { echo "could not parse an ssh target from: ${line}" >&2; return 1; }
    local common=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o BatchMode=yes)
    UWLAB_SSH_OPTS=("${common[@]}" -o ConnectTimeout=30)
    [[ -n "${jump}" ]] && UWLAB_SSH_OPTS+=(-o "ProxyCommand=ssh ${common[*]} -W %h:%p ${jump%:22}")
    return 0
}
