# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check that a converted robot USD carries the link masses of the URDF it was built from.

The Isaac Lab URDF importer authors ``physics:mass``, ``physics:centerOfMass``, ``physics:diagonalInertia`` and
``physics:principalAxes`` on every link. This script opens the USD without launching Kit (``pxr`` alone), reads
those attributes off every rigid body and compares them with the URDF's ``<inertial>`` blocks: mass, centre of
mass (in the link frame) and the principal moments of inertia. It prints a table and the totals and exits 1 on
any mismatch, so an asset whose hand lost its inertials (as the Dex3 MJCF of ludo-research once did) never
reaches a training run.

Example::

    python scripts_v2/tools/check_usd_inertials.py robot.urdf robot.usd [--tol-mass 1e-6] [--tol-com 1e-5] [--tol-inertia 1e-4]
"""

from __future__ import annotations

import argparse
import logging
import math
import numpy as np
import sys
import xml.etree.ElementTree as ET

from uwlab.utils.log_utils import configure_console_logging

logger = logging.getLogger(__name__)


def urdf_inertials(path: str) -> dict[str, tuple[float, np.ndarray, np.ndarray]]:
    """Read the inertial block of every link of a URDF.

    Args:
        path: Path to the URDF.

    Returns:
        Link name -> (mass [kg], centre of mass [m] in the link frame, sorted principal moments [kg m^2]) for the
        links that have an ``<inertial>`` element.
    """
    out = {}
    for link in ET.parse(path).getroot().findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass = float(inertial.find("mass").get("value"))
        origin = inertial.find("origin")
        com = np.array([float(v) for v in origin.get("xyz").split()]) if origin is not None else np.zeros(3)
        i = inertial.find("inertia")
        ixx, ixy, ixz, iyy, iyz, izz = (float(i.get(k)) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"))
        tensor = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
        if origin is not None and origin.get("rpy"):
            r, p, y = (float(v) for v in origin.get("rpy").split())
            R = _rpy_to_matrix(r, p, y)
            tensor = R @ tensor @ R.T
        out[link.get("name")] = (mass, com, np.sort(np.linalg.eigvalsh(tensor)))
    return out


def usd_inertials(path: str) -> dict[str, tuple[float, np.ndarray, np.ndarray]]:
    """Read the authored mass properties of every rigid body of a USD.

    Args:
        path: Path to the USD entry layer.

    Returns:
        Body name -> (mass [kg], centre of mass [m] in the body frame, sorted principal moments [kg m^2]) for the
        prims that carry ``PhysicsMassAPI`` with an authored mass.
    """
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(path)
    out = {}
    for prim in Usd.PrimRange(stage.GetDefaultPrim()):
        if not prim.HasAPI(UsdPhysics.MassAPI):
            continue
        mass_api = UsdPhysics.MassAPI(prim)
        mass = mass_api.GetMassAttr().Get()
        if mass is None:
            continue
        com = np.array(mass_api.GetCenterOfMassAttr().Get() or (0.0, 0.0, 0.0), dtype=float)
        diag = np.array(mass_api.GetDiagonalInertiaAttr().Get() or (0.0, 0.0, 0.0), dtype=float)
        out[prim.GetName()] = (float(mass), com, np.sort(diag))
    return out


def main() -> int:
    """Compare the URDF and USD named on the command line.

    Returns:
        0 when every link matches within the tolerances, 1 otherwise.
    """
    configure_console_logging()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("urdf")
    ap.add_argument("usd")
    ap.add_argument("--tol-mass", type=float, default=1e-6, help="absolute tolerance on the mass [kg]")
    ap.add_argument("--tol-com", type=float, default=1e-5, help="absolute tolerance on the centre of mass [m]")
    ap.add_argument("--tol-inertia", type=float, default=1e-4, help="relative tolerance on the principal moments")
    args = ap.parse_args()

    urdf, usd = urdf_inertials(args.urdf), usd_inertials(args.usd)
    failures = []
    logger.info(f"{'link':32s} {'urdf kg':>10s} {'usd kg':>10s} {'|dcom| mm':>10s} {'inertia rel':>12s}")
    for name, (mass, com, moments) in sorted(urdf.items()):
        if name not in usd:
            failures.append(f"{name}: no authored mass in the USD")
            logger.info(f"{name:32s} {mass:10.5f} {'missing':>10s}")
            continue
        u_mass, u_com, u_moments = usd[name]
        dcom = float(np.linalg.norm(com - u_com))
        rel = float(np.max(np.abs(moments - u_moments) / np.maximum(np.abs(moments), 1e-12)))
        logger.info(f"{name:32s} {mass:10.5f} {u_mass:10.5f} {1000 * dcom:10.4f} {rel:12.2e}")
        if not math.isclose(mass, u_mass, abs_tol=args.tol_mass):
            failures.append(f"{name}: mass {mass} (URDF) vs {u_mass} (USD)")
        if dcom > args.tol_com:
            failures.append(f"{name}: centre of mass off by {1000 * dcom:.3f} mm")
        if rel > args.tol_inertia:
            failures.append(f"{name}: principal moments differ by {100 * rel:.2f}%")
    extra = sorted(set(usd) - set(urdf))
    for name in extra:
        logger.info(f"{name:32s} {'(no urdf)':>10s} {usd[name][0]:10.5f}")
    total_urdf = sum(m for m, _, _ in urdf.values())
    total_usd = sum(m for m, _, _ in usd.values())
    hand_urdf = sum(m for n, (m, _, _) in urdf.items() if "_hand_" in n)
    hand_usd = sum(m for n, (m, _, _) in usd.items() if "_hand_" in n)
    logger.info(
        f"total mass: URDF {total_urdf:.4f} kg, USD {total_usd:.4f} kg; hand links: URDF {hand_urdf:.4f} kg, USD"
        f" {hand_usd:.4f} kg"
    )
    if failures:
        for f in failures:
            logger.error(f"[check_usd_inertials] {f}")
        return 1
    logger.info(f"[check_usd_inertials] OK: {len(urdf)} links match {args.usd}")
    return 0


# ===============
# Private Helpers
# ===============


def _rpy_to_matrix(r: float, p: float, y: float) -> np.ndarray:
    """Rotation matrix of URDF roll/pitch/yaw angles, ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``.

    Args:
        r: Roll [rad].
        p: Pitch [rad].
        y: Yaw [rad].

    Returns:
        The (3, 3) rotation matrix.
    """
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


if __name__ == "__main__":
    sys.exit(main())
