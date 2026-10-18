"""Module 4 acceptance checks: reference scaling, copies, and a two-point polar."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from aero_module import AeroEngine, InvalidReferenceQuantityError  # noqa: E402
from geo_module import GeometryEngine  # noqa: E402
from math_module import MeshEngine  # noqa: E402
from physics_module import PhysicsEngine  # noqa: E402


def _assert(cond: bool, message: str) -> None:
    if not cond:
        raise AssertionError(message)


def _solved(alpha: float):
    geometry = GeometryEngine().generate_naca("0012", chord=1.0, num_points=80)
    mesh = MeshEngine().from_geometry(
        geometry,
        {"n_panels": 64, "n_x": 24, "n_y": 16, "clustering": "cosine"},
    )
    physics = PhysicsEngine().solve_from_mesh(
        mesh,
        {"alpha_deg": alpha, "reynolds": 5.0e5, "mach": 0.0, "v_inf": 1.0, "rho": 1.225},
        {"regime": "inviscid"},
    )
    return mesh, physics


def test_default_report_matches_physics_and_does_not_alias_cp() -> None:
    mesh, physics = _solved(4.0)
    before = np.array(physics["inviscid"]["cp_inviscid"], copy=True)
    report = AeroEngine().analyze_from_solution(mesh, physics)
    _assert(
        abs(report["coefficients"]["cl"] - physics["loads"]["cl"]) < 1e-9,
        "unchanged reference must keep Cl",
    )
    _assert(report["coefficients"]["cd_induced"] == 0.0, "a section has no induced drag")
    _assert(np.isfinite(report["l_over_d"]), "L/D should be finite at 4 degrees")
    _assert(np.isfinite(report["coefficients"]["center_of_pressure_x"]), "center of pressure missing")
    _assert(report["spanwise"] is None, "2D report must not invent a spanwise load")
    _assert(not np.shares_memory(report["cp_surface"], physics["inviscid"]["cp_inviscid"]), "Cp was aliased")
    _assert(np.array_equal(before, physics["inviscid"]["cp_inviscid"]), "physics Cp changed")
    _assert(report["dimensional"]["lift"] > 0.0, "dimensional lift should be positive")


def test_reference_area_rescales_lift() -> None:
    mesh, physics = _solved(4.0)
    doubled = AeroEngine().analyze_from_solution(mesh, physics, {"s_ref": 2.0})
    _assert(
        abs(doubled["coefficients"]["cl"] - 0.5 * physics["loads"]["cl"]) < 1e-9,
        "doubling the chord must halve Cl",
    )
    try:
        AeroEngine().analyze_from_solution(mesh, physics, {"s_ref": 1.0, "c_ref": 2.0})
    except InvalidReferenceQuantityError:
        return
    raise AssertionError("2D s_ref and c_ref must be the same chord")


def test_polar_identifies_a_finite_aerodynamic_center() -> None:
    mesh_a, zero = _solved(0.0)
    _mesh_b, four = _solved(4.0)
    polar = AeroEngine().reduce_polar({"mesh": mesh_a, "samples": [four, zero], "parameters": {}})
    _assert(list(np.round(polar["alpha_deg"], 5)) == [0.0, 4.0], "polar must sort by alpha")
    _assert(polar["cl"][1] > polar["cl"][0] + 0.2, "lift must rise between 0 and 4 degrees")
    _assert(np.isfinite(polar["aerodynamic_center_x"]), "two angles should locate an aerodynamic center")
    _assert(abs(polar["cl_alpha_ideal"] - 2.0 * np.pi) < 1e-9, "2D ideal slope is 2 pi")
    _assert(np.isfinite(polar["cl_alpha_ratio"]), "slope ratio missing")
    _assert(polar["static_margin"] != polar["static_margin"], "margin without a center of gravity is NaN")
    with_cg = AeroEngine().reduce_polar(
        {"mesh": mesh_a, "samples": [zero, four], "parameters": {"cg_x_over_c": 0.25}}
    )
    _assert(np.isfinite(with_cg["static_margin"]), "a center of gravity should produce a static margin")


if __name__ == "__main__":
    test_default_report_matches_physics_and_does_not_alias_cp()
    test_reference_area_rescales_lift()
    test_polar_identifies_a_finite_aerodynamic_center()
    print("aero tests passed")
