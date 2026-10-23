# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Forge (Kubernetes) image for UW Lab / OmniReset training.
#
# Design: every dependency (Isaac Sim 6.1, Isaac Lab 3.0 EA, RSL-RL, torch cu128,
# the four uwlab extensions) is baked in, but the uwlab extensions are installed
# *editable* from ${UWLAB_PATH}. At job time a Forge disk holding a working copy of
# this repository is mounted over ${UWLAB_PATH}; the editable installs then resolve
# to the mounted code, so code changes need no image rebuild.
#
# Everything that must survive that mount (Isaac Sim, the Isaac Lab clone, the
# python wrappers) deliberately lives outside ${UWLAB_PATH}.
#
# Size matters on Forge: nodes evict pods whose image unpack exhausts the node OS
# disk. Two measures keep this image small:
#   1. `uwlab.sh --install` always requests Isaac Lab's `isaacsim` extra, i.e. the
#      full pip Isaac Sim (~9 GB). On the Isaac Sim container that is a second copy
#      of Isaac Sim, so the baked installer is patched to skip that extra.
#   2. The final stage copies the builder filesystem into a single layer, so files
#      deleted along the way (the base's bundled torch, pip/apt caches) do not
#      remain in lower layers.
# Runtime caches that Isaac Sim writes (shader/Kit/Omniverse caches) are redirected
# to /tmp, which Forge backs with a per-pod volume instead of the node OS disk.
#
# Build (from the repo root):
#   forge image build --name uwlab-omnireset --tag <git-sha> --context docker/forge \
#       --build-timeout 4 --build-memory 32Gi --build-cpu 8
#
# `forge image build` has no --build-arg; change the ARG defaults below to re-pin.

# ---------------------------------------------------------------------------
FROM nvcr.io/nvidia/isaac-sim:6.1.0 AS builder

SHELL ["/bin/bash", "-c"]

ARG UWLAB_REPO=https://github.com/UW-Lab/UWLab.git
ARG UWLAB_COMMIT=2b40afb

ENV LANG=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    ISAACSIM_ROOT_PATH=/isaac-sim \
    UWLAB_PATH=/workspace/uwlab

USER root

# System packages (mirrors docker/Dockerfile.base, plus rsync/openssh for code sync).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git git-lfs libglib2.0-0 ncurses-term rsync openssh-client wget && \
    apt-get -y autoremove && apt-get clean && rm -rf /var/lib/apt/lists/*

# Repository at the pinned commit (fallback / dependency source only; runtime code
# comes from the mounted disk).
RUN git clone "${UWLAB_REPO}" "${UWLAB_PATH}" && \
    git -C "${UWLAB_PATH}" checkout --detach "${UWLAB_COMMIT}" && \
    chmod +x "${UWLAB_PATH}/uwlab.sh"

RUN ln -sf "${ISAACSIM_ROOT_PATH}" "${UWLAB_PATH}/_isaac_sim"

# Isaac Lab clone lives in /opt so a disk mounted over ${UWLAB_PATH} cannot hide it.
# uwlab.sh resolves the path before installing, so the editable finders point at /opt.
RUN mkdir -p /opt/uwlab-deps/_isaaclab && \
    ln -s /opt/uwlab-deps/_isaaclab "${UWLAB_PATH}/_isaaclab"

# uwlab.sh --install skips the interactive VS Code setup only when it detects a
# container; BuildKit on Kubernetes does not always expose the markers it checks.
RUN touch /.dockerenv

# Patch the baked installer: do not pip-install a second Isaac Sim on top of the
# container's own. The patched line must exist, otherwise fail loudly.
RUN grep -q 'for extra in ("isaacsim", "video"):' "${UWLAB_PATH}/uwlab.sh" && \
    sed -i 's/for extra in ("isaacsim", "video"):/for extra in ("video",):/' "${UWLAB_PATH}/uwlab.sh"

RUN "${UWLAB_PATH}/uwlab.sh" -p -m pip install toml

RUN apt-get update && \
    "${UWLAB_PATH}/uwlab.sh" -p "${UWLAB_PATH}/tools/install_deps.py" apt "${UWLAB_PATH}/source" && \
    apt-get -y autoremove && apt-get clean && rm -rf /var/lib/apt/lists/*

# Isaac Lab 3.0 EA (pinned inside uwlab.sh), torch 2.11+cu128, RSL-RL (pinned), and
# the four uwlab extensions in editable mode. Caches are purged in the same layer.
RUN "${UWLAB_PATH}/uwlab.sh" --install && \
    "${UWLAB_PATH}/uwlab.sh" -p -m pip uninstall -y quadprog || true; \
    "${UWLAB_PATH}/uwlab.sh" -p -m pip cache purge || true; \
    rm -rf /root/.cache/pip /tmp/* /var/tmp/*

# Fail early if the install left something out, or if a second Isaac Sim slipped in.
RUN "${UWLAB_PATH}/uwlab.sh" -p -m pip show uwlab uwlab_assets uwlab_rl uwlab_tasks isaaclab isaaclab_rl isaaclab_tasks rsl-rl-lib torch > /dev/null && \
    "${UWLAB_PATH}/uwlab.sh" -p -c 'import torch, sys; v = torch.__version__; print("torch", v); sys.exit(0 if "+cu128" in v else 1)' && \
    if "${UWLAB_PATH}/uwlab.sh" -p -m pip list 2>/dev/null | grep -qiE '^isaacsim(-|\s)'; then \
        echo "ERROR: pip Isaac Sim was installed on top of the container Isaac Sim (image bloat)"; exit 1; fi

# `python` / `pip` / `uwlab` on PATH for non-interactive job entrypoints. Isaac Sim's
# python.sh sets up the Omniverse environment before exec'ing the real interpreter.
RUN printf '#!/bin/bash\nexec /isaac-sim/python.sh "$@"\n' > /usr/local/bin/python && \
    printf '#!/bin/bash\nexec /isaac-sim/python.sh -m pip "$@"\n' > /usr/local/bin/pip && \
    printf '#!/bin/bash\nexec "${UWLAB_PATH:-/workspace/uwlab}/uwlab.sh" "$@"\n' > /usr/local/bin/uwlab && \
    chmod +x /usr/local/bin/python /usr/local/bin/pip /usr/local/bin/uwlab

# Redirect Isaac Sim / Omniverse runtime caches and logs to /tmp (Forge gives every
# pod its own /tmp volume; the container root counts against the node OS disk).
# Same directories the repo's docker-compose mounts as volumes.
RUN for d in /root/.cache/ov /root/.cache/nvidia /root/.nv /root/.nvidia-omniverse /root/.local/share/ov /isaac-sim/kit/cache /isaac-sim/kit/logs /isaac-sim/kit/data; do \
        rm -rf "$d"; mkdir -p "$(dirname "$d")"; ln -s "/tmp/isaac-cache${d}" "$d"; \
    done && \
    printf '#!/bin/bash\nfor d in /root/.cache/ov /root/.cache/nvidia /root/.nv /root/.nvidia-omniverse /root/.local/share/ov /isaac-sim/kit/cache /isaac-sim/kit/logs /isaac-sim/kit/data; do mkdir -p "/tmp/isaac-cache$d"; done\n' > /usr/local/bin/uwlab-init-cache && \
    chmod +x /usr/local/bin/uwlab-init-cache

RUN echo "export UWLAB_PATH=${UWLAB_PATH}" >> /root/.bashrc && \
    echo "uwlab-init-cache" >> /root/.bashrc && \
    echo "cd ${UWLAB_PATH} 2>/dev/null || true" >> /root/.bashrc

# Trim the shallow Isaac Lab clone's git objects; the checkout is all that is used.
RUN rm -rf /opt/uwlab-deps/_isaaclab/IsaacLab/.git

# ---------------------------------------------------------------------------
# Single-layer final image. Metadata is not inherited from scratch, so the base
# image's ENV is restated here (read from nvcr.io/nvidia/isaac-sim:6.1.0).
FROM scratch

COPY --from=builder / /

ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    MUJOCO_GL=egl \
    VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json \
    MIN_DRIVER_VERSION=570.169 \
    HUB__CACHE__PATH=/var/cache/hub \
    HUB__ARGS__DETECT_ONLY=true \
    OMNICLIENT_HUB_EXE=/usr/local/bin/hub \
    LANG=C.UTF-8 \
    ISAACSIM_ROOT_PATH=/isaac-sim \
    UWLAB_PATH=/workspace/uwlab \
    ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    OMNI_KIT_ALLOW_ROOT=1

LABEL description="UW Lab 2.0 (Isaac Lab 3.0 EA / Isaac Sim 6.1) for Forge, single layer, code mounted at runtime."

USER root
WORKDIR /workspace/uwlab
