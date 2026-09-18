"""Smoke tests for Module 1 GeometryEngine (no pytest required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from geo_module import (  # noqa: E402
    GeometryEngine,
    InvalidGeometryInputError,
    is_watertight,
)


def _assert(cond: bool, message: str) -> None:
    if not cond:
        raise AssertionError(message)


def test_naca_closed_2d() -> None:
    out = GeometryEngine().generate_naca("4412", chord=1.0, num_points=80)
    pts = out["points_2d"]
    assert pts is not None
    _assert(out["mode"] == "2D", "NACA must be 2D")
    _assert(out["is_closed"], "NACA contour must be closed")
    _assert(pts.shape[1] == 2, "points_2d must be (N, 2)")
    _assert(np.allclose(pts[0], pts[-1]), "first/last vertex must match")
    _assert(out["num_elements"] == pts.shape[0], "num_elements mismatch")


def test_wing_has_airfoil_thickness() -> None:
    out = GeometryEngine().generate_wing(
        airfoil_code="0012",
        chord=1.0,
        span=4.0,
        sweep_deg=10.0,
        dihedral_deg=5.0,
        num_sections=8,
        airfoil_points=40,
    )
    facets = out["facets_3d"]
    assert facets is not None
    z_span = float(np.ptp(facets[:, :, 2]))
    _assert(z_span > 0.05, "wing must include airfoil thickness in z")
    _assert(out["is_closed"] is False, "uncapped wing skin must be open")


def test_cylinder_watertight() -> None:
    out = GeometryEngine().generate_cylinder(
        radius=0.5,
        length=2.0,
        segments=24,
        rings=6,
    )
    facets = out["facets_3d"]
    assert facets is not None
    _assert(out["is_closed"] is True, "cylinder must report closed")
    _assert(is_watertight(facets) is True, "cylinder must be watertight")


def test_invalid_naca_raises_typed_error() -> None:
    try:
        GeometryEngine().generate_naca("12")
    except InvalidGeometryInputError:
        return
    raise AssertionError("expected InvalidGeometryInputError")


def test_catalog_naca_sample() -> None:
    out = GeometryEngine().generate_sample("NACA 0012 (Symmetric)")
    _assert(out["shape_name"] == "NACA_0012", "catalog title lookup failed")


def main() -> None:
    test_naca_closed_2d()
    test_wing_has_airfoil_thickness()
    test_cylinder_watertight()
    test_invalid_naca_raises_typed_error()
    test_catalog_naca_sample()
    print("geo_module smoke tests passed")


if __name__ == "__main__":
    main()
