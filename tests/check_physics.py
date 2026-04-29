#!/usr/bin/env python3
"""
FastEddy Tier-3 physics conservation checks.

Validates physics invariants on a FastEddy NetCDF output snapshot without
requiring golden-reference data.  All checks are derived from first principles
that must hold regardless of hardware, compiler, or floating-point rounding:

  1. Finite fields       - no NaN or Inf in any variable
  2. Density positivity  - rho > 0 everywhere (air mass cannot vanish)
  3. Theta positivity    - total potential temperature (perturbation + base
                           state) > 0 K everywhere
  4. TKE non-negativity  - subgrid TKE >= floor (kinetic energy is positive
                           definite; a small numerical floor allows for
                           solver round-off)
  5. Moisture bounds     - qv and ql >= floor (mixing ratios are non-negative)
  6. Velocity bounds     - |u|, |v|, |w| < limit (rules out numerical blow-up)
"""

import argparse
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FastEddy physics conservation checks (Tier 3)"
    )
    p.add_argument(
        "output",
        help="Path to a FastEddy NetCDF output file (.nc)",
    )
    p.add_argument(
        "--velocity-limit",
        type=float,
        default=150.0,
        metavar="M_S",
        help="Maximum physically plausible wind speed in m/s (default: 150)",
    )
    p.add_argument(
        "--tke-floor",
        type=float,
        default=-1e-4,
        metavar="M2_S2",
        help="Minimum acceptable TKE in m²/s² (default: -1e-4, allows solver round-off)",
    )
    p.add_argument(
        "--moisture-floor",
        type=float,
        default=-1e-6,
        metavar="KG_KG",
        help="Minimum acceptable moisture mixing ratio in kg/kg (default: -1e-6)",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-field statistics even when checks pass",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_finite(ds: nc.Dataset, failures: list, verbose: bool) -> None:
    for name, var in ds.variables.items():
        if var.dtype.kind not in ("f", "d"):
            continue
        data = var[:]
        n_bad = int(np.sum(~np.isfinite(data)))
        if n_bad:
            failures.append(
                f"finite: '{name}' contains {n_bad} non-finite value(s) (NaN/Inf)"
            )
        elif verbose:
            print(f"  [finite   ] {name}: OK")


def check_density_positive(ds: nc.Dataset, failures: list, verbose: bool) -> None:
    if "rho" not in ds.variables:
        return
    rho = ds.variables["rho"][:]
    n_bad = int(np.sum(rho <= 0))
    if n_bad:
        failures.append(
            f"rho>0: {n_bad} non-positive density value(s) "
            f"(min={float(np.nanmin(rho)):.4g} kg/m³)"
        )
    elif verbose:
        print(f"  [rho>0    ] min={float(np.nanmin(rho)):.4g} max={float(np.nanmax(rho)):.4g} kg/m³")


def check_theta_positive(ds: nc.Dataset, failures: list, verbose: bool) -> None:
    """
    FastEddy stores theta as a perturbation from the base state (BS_1).
    The physically meaningful quantity is the total potential temperature:
        theta_total = theta_perturbation + BS_1
    which must be strictly positive everywhere.
    """
    if "theta" not in ds.variables or "BS_1" not in ds.variables:
        return
    theta_pert = ds.variables["theta"][:]
    theta_base = ds.variables["BS_1"][:]
    theta_total = theta_pert + theta_base
    n_bad = int(np.sum(theta_total <= 0))
    if n_bad:
        failures.append(
            f"theta>0: {n_bad} non-positive total potential temperature value(s) "
            f"(min={float(np.nanmin(theta_total)):.4g} K)"
        )
    elif verbose:
        print(
            f"  [theta>0  ] total theta min={float(np.nanmin(theta_total)):.4g} "
            f"max={float(np.nanmax(theta_total)):.4g} K"
        )


def check_tke(ds: nc.Dataset, failures: list, floor: float, verbose: bool) -> None:
    tke_vars = [n for n in ds.variables if n.startswith("TKE_")]
    for name in tke_vars:
        tke = ds.variables[name][:]
        n_bad = int(np.sum(tke < floor))
        if n_bad:
            failures.append(
                f"TKE>=0: '{name}' has {n_bad} value(s) below floor {floor} "
                f"(min={float(np.nanmin(tke)):.4g} m²/s²)"
            )
        elif verbose:
            print(
                f"  [TKE>=0   ] {name}: min={float(np.nanmin(tke)):.4g} "
                f"max={float(np.nanmax(tke)):.4g} m²/s²"
            )


def check_moisture(ds: nc.Dataset, failures: list, floor: float, verbose: bool) -> None:
    for name in ("qv", "ql"):
        if name not in ds.variables:
            continue
        data = ds.variables[name][:]
        n_bad = int(np.sum(data < floor))
        if n_bad:
            failures.append(
                f"moisture>=0: '{name}' has {n_bad} value(s) below floor {floor} "
                f"(min={float(np.nanmin(data)):.6g} kg/kg)"
            )
        elif verbose:
            print(
                f"  [moist>=0 ] {name}: min={float(np.nanmin(data)):.6g} "
                f"max={float(np.nanmax(data)):.6g} kg/kg"
            )


def check_velocity_bounds(
    ds: nc.Dataset, failures: list, limit: float, verbose: bool
) -> None:
    for name in ("u", "v", "w"):
        if name not in ds.variables:
            continue
        data = ds.variables[name][:]
        max_abs = float(np.nanmax(np.abs(data)))
        if max_abs > limit:
            failures.append(
                f"velocity: '{name}' max magnitude {max_abs:.2f} m/s exceeds "
                f"physical limit {limit:.0f} m/s — likely numerical blow-up"
            )
        elif verbose:
            print(f"  [vel<{limit:.0f}  ] {name}: max |{name}|={max_abs:.4g} m/s")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    path = Path(args.output)

    if not path.exists():
        print(f"FAIL: output file not found: {path}", file=sys.stderr)
        return 1

    ds = nc.Dataset(path)
    failures: list = []

    if args.verbose:
        print(f"Checking {path.name}")

    check_finite(ds, failures, args.verbose)
    check_density_positive(ds, failures, args.verbose)
    check_theta_positive(ds, failures, args.verbose)
    check_tke(ds, failures, args.tke_floor, args.verbose)
    check_moisture(ds, failures, args.moisture_floor, args.verbose)
    check_velocity_bounds(ds, failures, args.velocity_limit, args.verbose)

    ds.close()

    if failures:
        for msg in failures:
            print(f"FAIL: {msg}", file=sys.stderr)
        print(
            f"\n{len(failures)} physics check(s) failed for {path.name}",
            file=sys.stderr,
        )
        return 1

    print(f"PASS: all physics checks passed for {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
