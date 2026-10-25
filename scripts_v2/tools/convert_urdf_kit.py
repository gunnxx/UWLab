# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert a URDF to USD inside a running Isaac Sim (Kit) app.

``scripts/tools/convert_urdf.py`` imports ``isaaclab.sim`` and ``pxr`` before it launches the simulator, which
is fine for its kit-less importer path but breaks on images where the Kit app must be started first (the pxr
Python wrappers then fail to initialise). This variant follows the pattern of every other script in this repo:
launch the app, then import.

Example::

    python scripts_v2/tools/convert_urdf_kit.py robot.urdf robot.usd --fix_base --joint_stiffness 0 --joint_damping 0 --visualizer none
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert a URDF to USD (Kit app).")
parser.add_argument("input", type=str, help="Path to the input URDF file.")
parser.add_argument("output", type=str, help="Path of the USD entry layer to write.")
parser.add_argument("--merge_joints", "--merge-joints", action="store_true", default=False,
                    help="Consolidate links connected by fixed joints.")
parser.add_argument("--fix_base", "--fix-base", action="store_true", default=False, help="Fix the base link.")
parser.add_argument("--joint_stiffness", "--joint-stiffness", type=float, default=100.0)
parser.add_argument("--joint_damping", "--joint-damping", type=float, default=1.0)
parser.add_argument("--joint_target_type", "--joint-target-type", type=str, default="position",
                    choices=["position", "velocity", "none"])
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os  # noqa: E402
import sys  # noqa: E402

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402
from isaaclab.utils.assets import check_file_path  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", "tools"))
from usd_output import write_usd_entry_layer  # noqa: E402


def main():
    urdf_path = os.path.abspath(args_cli.input)
    if not check_file_path(urdf_path):
        raise ValueError(f"Invalid file path: {urdf_path}")
    dest_path = os.path.abspath(args_cli.output)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=os.path.dirname(dest_path),
        fix_base=args_cli.fix_base,
        merge_fixed_joints=args_cli.merge_joints,
        force_usd_conversion=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=args_cli.joint_stiffness, damping=args_cli.joint_damping
            ),
            target_type=args_cli.joint_target_type,
        ),
    )
    print("-" * 80)
    print(f"Input URDF file: {urdf_path}")
    print("URDF importer config:")
    print_dict(cfg.to_dict(), nesting=0)
    print("-" * 80)

    converter = UrdfConverter(cfg)
    output_path = write_usd_entry_layer(converter.usd_path, dest_path)
    print(f"Generated USD file: {converter.usd_path}")
    print(f"USD entry layer: {output_path}")
    print("-" * 80)


if __name__ == "__main__":
    main()
    simulation_app.close()
