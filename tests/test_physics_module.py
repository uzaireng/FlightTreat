"""Module 3 acceptance checks: lift, pressure, boundary layer, and fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from geo_module import GeometryEngine  # noqa: E402
from math_module import MeshEngine  # noqa: E402
from physics_module import (  # noqa: E402
    InvalidFlowConditionError,
    PhysicsEngine,
)
from physics_module.physics_skeleton import validate_physics_output  # noqa: E402
from shared.pipeline_runner import run_physics_module_sample  # noqa: E402


def _assert(cond: bool, message: str) -> None:
    if not cond:
        raise AssertionError(message)


def _mesh(spec: dict, parameters: dict):
    geometry = GeometryEngine().generate_from_spec(spec)
    return MeshEngine().from_geometry(geometry, parameters)


def _flow(alpha: float, regime: str = "inviscid") -> tuple[dict, dict]:
    return (
        {"alpha_deg": alpha, "reynolds": 5.0e5, "mach": 0.0, "v_inf": 1.0, "rho": 1.225},
        {"regime": regime},
    )


def test_flat_plate_and_naca_lift() -> None:
    engine = PhysicsEngine()
    plate = _mesh(
        {"type": "flat_plate", "mode": "2D", "chord": 1.0, "num_points": 160, "thickness": 0.01},
        {
            "n_panels": 120,
            "n_x": 32,
            "n_y": 20,
            "clustering": "uniform",
            "le_cluster_factor": 0.0,
            "te_cluster_factor": 0.0,
        },
    )
    plate_out = engine.solve_from_mesh(plate, *_flow(4.0))
    target = 2.0 * np.pi * np.deg2rad(4.0)
    _assert(abs(plate_out["loads"]["cl"] - target) < 0.08, f"plate Cl {plate_out['loads']['cl']}")
    _assert(
        abs(plate_out["loads"]["cl"] - 2.0 * plate_out["loads"]["gamma_total"]) < 0.05,
        "plate lift and circulation disagree",
    )

    naca = _mesh(
        {"type": "naca", "mode": "2D", "code": "0012", "chord": 1.0, "num_points": 80},
        {"n_panels": 80, "n_x": 32, "n_y": 20, "clustering": "cosine"},
    )
    zero = engine.solve_from_mesh(naca, *_flow(0.0))
    four = engine.solve_from_mesh(naca, *_flow(4.0))
    _assert(abs(zero["loads"]["cl"]) < 0.02, f"zero-incidence Cl {zero['loads']['cl']}")
    _assert(abs(four["loads"]["cl"] - 0.48) < 0.08, f"NACA 4 deg Cl {four['loads']['cl']}")
    _assert(
        abs(four["loads"]["cl"] - 2.0 * four["loads"]["gamma_total"]) < 0.05,
        "NACA lift and circulation disagree",
    )
    panel = naca["panel_2d"]
    assert panel is not None
    cp = zero["inviscid"]["cp_inviscid"]
    x = panel["collocation"][:, 0]
    upper = panel["normal"][:, 1] >= 0.0
    order = np.argsort(x[~upper])
    mirrored = np.interp(x[upper], x[~upper][order], cp[~upper][order])
    _assert(float(np.max(np.abs(cp[upper] - mirrored))) < 0.08, "zero-incidence Cp is not symmetric")
    solid = naca["cartesian_2d"]["boundary_mask"] == 6
    field = zero["field_2d"]
    assert field is not None
    _assert(np.all(np.isnan(field["u"][solid])), "solid farfield cells must be nan")
    finite_w = field["w"][~np.isnan(field["w"])]
    _assert(np.all(finite_w == 0.0), "finite field_2d.w must be 0")
    validate_physics_output(zero)


def test_cylinder_nose_pressure() -> None:
    mesh = _mesh(
        {"type": "cylinder", "mode": "3D", "radius": 0.5, "length": 2.0, "segments": 16, "rings": 4},
        {"normal_orientation": "outward", "require_closed_3d": True},
    )
    out = PhysicsEngine().solve_from_mesh(mesh, *_flow(0.0))
    _assert(abs(out["loads"]["cl"]) < 0.05, f"cylinder Cl {out['loads']['cl']}")
    panel = mesh["panel_3d"]
    assert panel is not None
    facing = panel["normals"] @ np.array([1.0, 0.0, 0.0])
    cp = out["inviscid"]["cp_inviscid"]
    cap = float(np.mean(cp[facing < -0.5]))
    side = float(np.mean(cp[np.abs(panel["normals"][:, 1]) > 0.8]))
    _assert(cap > side, f"nose Cp {cap} is not above side-wall Cp {side}")
    _assert(np.all(np.isfinite(cp)), "cylinder Cp must be finite")


def test_separated_march_returns_payload() -> None:
    mesh = _mesh(
        {"type": "naca", "mode": "2D", "code": "0012", "chord": 1.0, "num_points": 80},
        {"n_panels": 64, "n_x": 24, "n_y": 16, "clustering": "cosine"},
    )
    out = PhysicsEngine().solve_from_mesh(
        mesh,
        *_flow(10.0, "viscous_inviscid"),
    )
    assert out["ibl"] is not None
    _assert(out["ibl"]["separated"], "high-incidence march should separate")
    _assert(np.all(np.isfinite(out["inviscid"]["cp_inviscid"])), "separated Cp must stay finite")
    _assert(any("separated" in note.lower() for note in out["warnings"]), "separation needs a warning")


def test_unimplemented_closures_raise() -> None:
    mesh = _mesh(
        {"type": "naca", "mode": "2D", "code": "0012", "chord": 1.0, "num_points": 40},
        {"n_panels": 32, "n_x": 16, "n_y": 12, "clustering": "uniform"},
    )
    flow, _params = _flow(0.0, "viscous_inviscid")
    try:
        PhysicsEngine().solve_from_mesh(mesh, flow, {"ibl_closure": "drela_two_equation"})
    except InvalidFlowConditionError as exc:
        _assert(
            str(exc) == "ibl_closure 'drela_two_equation' is not implemented",
            f"unexpected drela message: {exc}",
        )
    else:
        raise AssertionError("drela closure must be rejected")
    try:
        PhysicsEngine().solve_from_mesh(mesh, flow, {"viscous_coupling": "displacement"})
    except InvalidFlowConditionError as exc:
        _assert("displacement" in str(exc), f"unexpected displacement message: {exc}")
    else:
        raise AssertionError("displacement coupling must be rejected")


def test_mesh_arrays_are_not_mutated() -> None:
    mesh = _mesh(
        {"type": "naca", "mode": "2D", "code": "0012", "chord": 1.0, "num_points": 40},
        {"n_panels": 32, "n_x": 16, "n_y": 12, "clustering": "cosine"},
    )
    panel = mesh["panel_2d"]
    assert panel is not None
    before = np.array(panel["nodes"], copy=True)
    PhysicsEngine().solve_from_mesh(mesh, *_flow(4.0))
    _assert(np.array_equal(panel["nodes"], before), "solver wrote into the mesh nodes")


def test_physics_module_samples() -> None:
    bad = run_physics_module_sample("bad_te_index")
    _assert(bad["error_type"] == "InvalidFlowConditionError", bad["error_type"] or "no error")
    singular = run_physics_module_sample("singular_collocation")
    _assert(
        singular["error_type"] == "SingularInfluenceMatrixError",
        singular["error_type"] or "no error",
    )
    blasius = run_physics_module_sample("thwaites_blasius")
    _assert(blasius["theta_ratio"] is not None and blasius["theta_ratio"] < 0.02, "Thwaites left Blasius")
    attached = run_physics_module_sample("naca0012_alpha0")
    _assert(attached["cl"] is not None and abs(attached["cl"]) < 0.02, f"sample Cl {attached['cl']}")


if __name__ == "__main__":
    test_flat_plate_and_naca_lift()
    test_cylinder_nose_pressure()
    test_separated_march_returns_payload()
    test_unimplemented_closures_raise()
    test_mesh_arrays_are_not_mutated()
    test_physics_module_samples()
    print("physics tests passed")
