# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Torch Dataset File Handler
This module provides a dataset file handler that saves data directly in the torch format.
"""

from __future__ import annotations

import os
import time
import torch
from collections.abc import Iterable
from typing import Any

from isaaclab.utils.datasets.dataset_file_handler_base import DatasetFileHandlerBase, EpisodeData


class TorchDatasetFileHandler(DatasetFileHandlerBase):
    """
    Dataset file handler that saves data directly in torch format as a torch file.
    """

    flush_interval_s: float = float(os.environ.get("UWLAB_DATASET_FLUSH_INTERVAL", "60"))
    """Minimum time between two writes of the file in :meth:`flush` (seconds; env ``UWLAB_DATASET_FLUSH_INTERVAL``).

    Isaac Lab's ``RecorderManager`` calls :meth:`flush` after every export, i.e. on every environment step
    that ends at least one successful episode, and each write serializes the whole accumulated dataset.
    Unthrottled, a reset-state recording at 16k environments spent ever more of each step rewriting the file
    (measured on Forge: the step rate fell by half over the first ~4,500 states). A pending write is never
    dropped: :meth:`close` always writes, and writes are atomic (temporary file + rename), so the file on disk
    is a complete snapshot at most ``flush_interval_s`` old.
    """

    def __init__(self):
        """Constructor."""
        self._file_path: str | None = None
        self._episode_data: dict[str, Any] = {}
        self._episode_count: int = 0
        self._dirty: bool = False
        self._last_flush: float = -float("inf")

    def create(self, file_path: str, env_name: str | None = None):
        """Create a new dataset file.

        Args:
            file_path: Path of the dataset file to write.
            env_name: Name of the environment the data comes from (not stored by this handler).
        """
        self._file_path = file_path
        self._episode_data = {}
        self._episode_count = 0
        self._dirty = False
        self._last_flush = -float("inf")

        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    def open(self, file_path: str, mode: str = "r"):
        """Open an existing dataset file."""
        if mode == "r":
            if os.path.exists(file_path):
                self._episode_data = torch.load(file_path)
                self._file_path = file_path
            else:
                raise FileNotFoundError(f"Dataset file not found: {file_path}")
        else:
            self._file_path = file_path
            self._episode_data = {}

    def get_env_name(self) -> str | None:
        """Get the environment name."""
        return

    def get_episode_names(self) -> Iterable[str]:
        """Get the names of the episodes in the file."""
        return []

    def get_num_episodes(self) -> int:
        """Get number of episodes in the file."""
        return self._episode_count

    def write_episode(self, episode: EpisodeData, demo_id: int | None = None, dataset_compression: bool = True):
        """Add an episode to the dataset.

        Args:
            episode: The episode data to add.
            demo_id: Custom index for the episode. If None, uses default index.
            dataset_compression: Accepted for interface compatibility and ignored.
                Isaac Lab 3.0's RecorderManager passes ``cfg.dataset_compression``
                positionally to every handler (see
                :meth:`isaaclab.managers.RecorderManager.export_episodes`); omitting
                it raises ``TypeError: write_episode() takes from 2 to 3 positional
                arguments but 4 were given`` at the first episode export. It is
                meaningful only for the HDF5 handler, which maps it onto gzip
                per-dataset compression. This handler accumulates episodes in memory
                and serializes them with ``torch.save`` in :meth:`flush` (at most every
                :attr:`flush_interval_s`), which has no equivalent option.
        """
        if episode.is_empty() or not episode.success:
            return

        # Extract only the last entry from this episode using extend_dicts_last_entry logic
        self._extend_dicts_last_entry(self._episode_data, episode.data)
        self._dirty = True

        # Only increment episode count if using default indexing
        if demo_id is None:
            self._episode_count += 1

    def _extend_dicts_last_entry(self, dest: dict[str, Any], src: dict[str, Any], store_data: bool = True) -> None:
        """Extend destination dictionary with last entry from source dictionary."""
        for key in src.keys():
            if isinstance(src[key], dict):
                if key not in dest:
                    dest[key] = {}
                self._extend_dicts_last_entry(dest[key], src[key], store_data)
            else:
                if key not in dest:
                    dest[key] = []
                if store_data:
                    dest[key].extend(src[key][-1:])  # Only take the last entry from src[key]

    def load_episode(self, episode_name: str, device: str = "cpu") -> EpisodeData | None:
        """Load episode data from the file."""
        # Not applicable for this handler since we store aggregated data
        raise NotImplementedError("Load episode not supported for preprocessed format")

    def flush(self, force: bool = False):
        """Write the accumulated dataset to disk.

        Skipped while nothing new was written since the last flush, or while the last write is less than
        :attr:`flush_interval_s` old (unless ``force``). The write goes to a temporary file that is then renamed
        over the dataset file, so a reader (or a copy taken while recording) never sees a partial file.

        Args:
            force: Write even if the last write is less than :attr:`flush_interval_s` old.
        """
        if not (self._file_path and self._episode_data and self._dirty):
            return
        now = time.monotonic()
        if not force and now - self._last_flush < self.flush_interval_s:
            return
        # Stamp the quaternion convention; loaders refuse files without it.
        tmp_path = self._file_path + ".tmp"
        torch.save({**self._episode_data, "quat_convention": "xyzw"}, tmp_path)
        os.replace(tmp_path, self._file_path)
        self._last_flush = now
        self._dirty = False

    def close(self):
        """Close the dataset file handler, writing whatever is still pending."""
        self.flush(force=True)
        self._episode_data = {}
        self._file_path = None

    def add_env_args(self, env_args: dict):
        pass
