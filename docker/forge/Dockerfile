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
# Build (from the repo root):
#   forge image build --name uwlab-omnireset --tag <git-sha> --context docker/forge \
#       --build-timeout 4 --build-memory 32Gi --build-cpu 8
#
# `forge image build` has no --build-arg; change the ARG defaults below to re-pin.

FROM nvcr.io/nvidia/isaac-sim:6.1.0

SHELL ["/bin/bash", "-c"]

LABEL description="UW Lab 2.0 (Isaac Lab 3.0 EA / Isaac Sim 6.1) for Forge, code mounted at runtime."

# Repository and commit baked into the image. The baked copy is only a fallback /
# dependency source; the runtime code comes from the mounted disk.
ARG UWLAB_REPO=https://github.com/UW-Lab/UWLab.git
ARG UWLAB_COMMIT=2b40afb

ENV LANG=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    ISAACSIM_ROOT_PATH=/isaac-sim \
    UWLAB_PATH=/workspace/uwlab \
    ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    OMNI_KIT_ALLOW_ROOT=1

USER root

# System packages (mirrors docker/Dockerfile.base, plus rsync/openssh for code sync).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        git-lfs \
        libglib2.0-0 \
        ncurses-term \
        rsync \
        openssh-client \
        wget && \
    apt-get -y autoremove && apt-get clean && rm -rf /var/lib/apt/lists/*

# Clone the repository at the pinned commit.
RUN git clone "${UWLAB_REPO}" "${UWLAB_PATH}" && \
    git -C "${UWLAB_PATH}" checkout --detach "${UWLAB_COMMIT}" && \
    chmod +x "${UWLAB_PATH}/uwlab.sh"

# Isaac Sim symlink expected by uwlab.sh.
RUN ln -sf "${ISAACSIM_ROOT_PATH}" "${UWLAB_PATH}/_isaac_sim"

# uwlab.sh --install clones Isaac Lab into ${UWLAB_PATH}/_isaaclab and installs it
# editable. Redirect that directory to /opt so the clone survives when a disk is
# mounted over ${UWLAB_PATH}. uwlab.sh resolves the path before installing, so the
# editable finders point at /opt/uwlab-deps/_isaaclab, not at the symlink.
RUN mkdir -p /opt/uwlab-deps/_isaaclab && \
    ln -s /opt/uwlab-deps/_isaaclab "${UWLAB_PATH}/_isaaclab"

# uwlab.sh --install skips the interactive VS Code setup only when it detects a
# container. BuildKit on Kubernetes does not always expose the markers it checks,
# so create the classic one explicitly.
RUN touch /.dockerenv

# toml is needed by tools/install_deps.py.
RUN "${UWLAB_PATH}/uwlab.sh" -p -m pip install toml

# apt dependencies declared in each extension's extension.toml.
RUN apt-get update && \
    "${UWLAB_PATH}/uwlab.sh" -p "${UWLAB_PATH}/tools/install_deps.py" apt "${UWLAB_PATH}/source" && \
    apt-get -y autoremove && apt-get clean && rm -rf /var/lib/apt/lists/*

# Isaac Lab 3.0 EA (pinned inside uwlab.sh), torch 2.11+cu128, RSL-RL (pinned),
# and the four uwlab extensions in editable mode.
RUN "${UWLAB_PATH}/uwlab.sh" --install

# Same workaround as docker/Dockerfile.base.
RUN "${UWLAB_PATH}/uwlab.sh" -p -m pip uninstall -y quadprog || true

# Fail the build early if the install silently left something out.
RUN "${UWLAB_PATH}/uwlab.sh" -p -m pip show uwlab uwlab_assets uwlab_rl uwlab_tasks isaaclab isaaclab_rl isaaclab_tasks rsl-rl-lib torch > /dev/null && \
    "${UWLAB_PATH}/uwlab.sh" -p -c 'import torch, sys; v = torch.__version__; print("torch", v); sys.exit(0 if "+cu128" in v else 1)'

# `python` / `pip` / `uwlab` on PATH for non-interactive job entrypoints. Isaac
# Sim's python.sh sets up the Omniverse environment before exec'ing the real
# interpreter, so torchrun workers (which re-exec sys.executable) inherit it.
RUN printf '#!/bin/bash\nexec /isaac-sim/python.sh "$@"\n' > /usr/local/bin/python && \
    printf '#!/bin/bash\nexec /isaac-sim/python.sh -m pip "$@"\n' > /usr/local/bin/pip && \
    printf '#!/bin/bash\nexec "${UWLAB_PATH:-/workspace/uwlab}/uwlab.sh" "$@"\n' > /usr/local/bin/uwlab && \
    chmod +x /usr/local/bin/python /usr/local/bin/pip /usr/local/bin/uwlab

# Interactive conveniences (notebooks / ssh).
RUN echo "export UWLAB_PATH=${UWLAB_PATH}" >> /root/.bashrc && \
    echo "cd ${UWLAB_PATH} 2>/dev/null || true" >> /root/.bashrc

WORKDIR ${UWLAB_PATH}
