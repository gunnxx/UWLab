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


def anchor_articulation_root(usd_path: str) -> None:
    """Turn the importer's fixed-base output into a real PhysX fixed-base articulation.

    With ``fix_base`` the Isaac Sim 6.1 importer keeps ``PhysicsArticulationRootAPI`` on the base *link* and adds a
    fixed ``root_joint`` whose body0 is the asset's default prim. PhysX then treats the asset as a floating-base
    articulation held by a constraint, not as a fixed-base one: teleporting the root (``write_root_pose_to_sim``,
    as the grasp sampler does) fights the joint and blows up, and Isaac Lab does not recognise the joint as a world
    anchor either (it looks for a joint with a missing body target). Isaac Lab's own ``fix_root_link=True`` repair
    cannot fix this at spawn time because the attributes live in a referenced layer.

    This rewrites the layer that authors the articulation root the way hand-made assets (e.g. the Robotiq grasp
    gripper) are authored: the articulation root and its authored root-level attributes move to the default prim,
    and the root joint's body0 is removed so it anchors the base link to the world.
    """
    from pxr import Sdf, Usd, UsdPhysics  # noqa: PLC0415

    def _edit_tokens(spec: Sdf.PrimSpec, add: set[str], remove: set[str]) -> None:
        op = spec.GetInfo("apiSchemas") if spec.HasInfo("apiSchemas") else Sdf.TokenListOp()
        new_op = Sdf.TokenListOp()
        if op.isExplicit:
            new_op.explicitItems = [t for t in op.explicitItems if t not in remove] + sorted(add)
        else:
            new_op.prependedItems = [t for t in op.prependedItems if t not in remove] + sorted(add)
            new_op.appendedItems = [t for t in op.appendedItems if t not in remove]
            new_op.deletedItems = list(op.deletedItems)
        spec.SetInfo("apiSchemas", new_op)

    stage = Usd.Stage.Open(usd_path)
    root = stage.GetDefaultPrim()
    link = next((p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.ArticulationRootAPI)), None)
    if link is None:
        raise RuntimeError(f"No PhysicsArticulationRootAPI found under '{root.GetPath()}' in {usd_path}")
    if link == root:
        print("[anchor] articulation root already on the default prim; nothing to do")
        return
    world_joint = None
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdPhysics.FixedJoint):
            continue
        joint = UsdPhysics.FixedJoint(prim)
        body0, body1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if body1 == [link.GetPath()] and (not body0 or body0 == [root.GetPath()]):
            world_joint = prim
            break
    if world_joint is None:
        raise RuntimeError(f"No fixed joint anchoring '{link.GetPath()}' to the world; was --fix_base given?")

    root_schemas = {"PhysicsArticulationRootAPI", "PhysxArticulationAPI", "NewtonArticulationRootAPI"}
    root_props: set[str] = {"newton:selfCollisionEnabled"}
    schema_def = Usd.SchemaRegistry().FindAppliedAPIPrimDefinition("NewtonArticulationRootAPI")
    if schema_def is not None:
        root_props.update(schema_def.GetPropertyNames())
    depth = len(link.GetPath().GetPrefixes()) - len(root.GetPath().GetPrefixes())

    # the layer(s) that author the articulation root on the link; usually exactly one (payloads/Physics/physics.usda)
    touched = []
    for link_spec in link.GetPrimStack():
        op = link_spec.GetInfo("apiSchemas") if link_spec.HasInfo("apiSchemas") else None
        if op is None or "PhysicsArticulationRootAPI" not in op.GetAddedOrExplicitItems():
            continue
        layer = link_spec.layer
        root_path = link_spec.path
        for _ in range(depth):
            root_path = root_path.GetParentPath()
        root_spec = layer.GetPrimAtPath(root_path) or Sdf.CreatePrimInLayer(layer, root_path)
        _edit_tokens(link_spec, add=set(), remove=root_schemas)
        _edit_tokens(root_spec, add=root_schemas & set(op.GetAddedOrExplicitItems()), remove=set())
        moved = []
        for name in list(link_spec.properties.keys()):
            if name.startswith("physxArticulation:") or name in root_props:
                src = link_spec.properties[name]
                dst = root_spec.properties.get(name) or Sdf.AttributeSpec(root_spec, name, src.typeName)
                dst.default = src.default
                link_spec.RemoveProperty(src)
                moved.append(name)
        touched.append(layer)
        print(f"[anchor] {layer.identifier}: articulation root {link_spec.path} -> {root_path}; moved {moved}")
    # drop every authored body0 of the world joint so that it anchors to the world
    for joint_spec in world_joint.GetPrimStack():
        rel = joint_spec.relationships.get("physics:body0")
        if rel is not None:
            joint_spec.RemoveProperty(rel)
            touched.append(joint_spec.layer)
            print(f"[anchor] {joint_spec.layer.identifier}: cleared body0 of {joint_spec.path}")
    for layer in dict.fromkeys(touched):
        layer.Save()


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
    if args_cli.fix_base:
        anchor_articulation_root(converter.usd_path)
    output_path = write_usd_entry_layer(converter.usd_path, dest_path)
    print(f"Generated USD file: {converter.usd_path}")
    print(f"USD entry layer: {output_path}")
    print("-" * 80)


if __name__ == "__main__":
    main()
    simulation_app.close()
