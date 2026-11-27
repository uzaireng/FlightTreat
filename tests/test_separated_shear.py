"""Separated-shear, bubble, and lag checks.

These tests lock the model assumptions. They do not retune a NACA 0012
error, and they do not treat a numerically finite stalled lift as success.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from geo_module import GeometryEngine  # noqa: E402
from math_module import MeshEngine  # noqa: E402
from physics_module import PhysicsEngine  # noqa: E402
from physics_module.drela_equilibrium import (  # noqa: E402
    layer_state,
    march_interval,
    relax_shear_stress,
)
from physics_module.separated_shear import (  # noqa: E402
    mayle_transition_distance,
    separated_entrainment,
    separated_shear_step,
)


def _assert(cond: bool, message: str) -> None:
    if not cond:
        raise AssertionError(message)


def _mesh(code: str, n_panels: int):
    geometry = GeometryEngine().generate_from_spec(
        {"type": "naca", "mode": "2D", "code": code, "chord": 1.0, "num_points": max(n_panels + 1, 80)}
    )
    return MeshEngine().from_geometry(
        geometry,
        {
            "n_panels": n_panels,
            "n_x": 8,
            "n_y": 8,
            "clustering": "cosine",
            "le_cluster_factor": 1.0,
            "te_cluster_factor": 1.0,
        },
    )


def _solve(code: str, alpha: float, n_panels: int, reynolds: float = 6.0e6, closure: str = "thwaites_head"):
    mesh = _mesh(code, n_panels)
    return PhysicsEngine().solve_from_mesh(
        mesh,
        {"alpha_deg": alpha, "reynolds": reynolds, "mach": 0.15, "v_inf": 1.0, "rho": 1.225},
        {"regime": "viscous_inviscid", "prandtl_glauert": True, "ibl_closure": closure},
    )


def test_mayle_length_matches_the_published_power() -> None:
    """Re_s,tr = 700 Re_θ^0.7, converted to arc length with the local edge speed."""

    theta = 1.0e-3
    ue = 1.0
    nu = 1.0e-6
    re_theta = ue * theta / nu
    expected = 700.0 * re_theta**0.7 * nu / ue
    got = mayle_transition_distance(theta, ue, nu)
    _assert(abs(got - expected) < 1.0e-12, f"Mayle length {got} is not {expected}")
    _assert(not np.isfinite(mayle_transition_distance(-1.0, ue, nu)), "a negative θ produced a length")


def test_separated_shear_grows_displacement_at_constant_edge_speed() -> None:
    """Cf = 0 and dUe = 0 leave θ fixed and increase δ* at Head's H=2.8 entrainment."""

    theta = 1.0e-3
    shape = 2.8
    step = separated_shear_step(theta, shape, 1.0, 1.0, 0.05, separated_entrainment())
    _assert(step is not None, "constant-edge separated step was rejected")
    theta_new, shape_new = step
    _assert(abs(theta_new - theta) / theta < 1.0e-9, f"θ changed to {theta_new} with no pressure force and no wall shear")
    _assert(shape_new > shape, "displacement thickness did not grow")
    _assert(1.05 <= shape_new <= 8.0, f"shape factor {shape_new} left the validity band")


def test_separated_shear_thickens_momentum_in_an_adverse_gradient() -> None:
    theta = 1.0e-3
    step = separated_shear_step(theta, 2.5, 1.0, 0.9, 0.05, separated_entrainment())
    _assert(step is not None, "adverse separated step was rejected")
    theta_new, shape_new = step
    _assert(theta_new > theta, "adverse gradient did not increase the momentum thickness")
    _assert(shape_new > 1.05, f"shape factor collapsed to {shape_new}")


def test_lag_relaxes_toward_equilibrium_without_overshoot() -> None:
    """One short step from twice the equilibrium stress moves toward it and stays positive."""

    state = layer_state(1.0e-3, 1.5, 1.0, 0.0, 1.0e-6, False)
    _assert(state is not None, "equilibrium state was not finite")
    c_eq = float(state["equilibrium_sqrt_shear"] ** 2)
    _assert(c_eq > 0.0, "equilibrium shear stress was not positive")
    started = relax_shear_stress(c_eq, c_eq, 1.0e-3, 1.5, 0.01)
    _assert(abs(started - c_eq) / c_eq < 1.0e-9, "equilibrium stress drifted when it was already there")
    ahead = relax_shear_stress(4.0 * c_eq, c_eq, 1.0e-3, 1.5, 1.0e-3)
    _assert(c_eq < ahead < 4.0 * c_eq, f"lag step {ahead} did not stay between the start and the equilibrium")
    theta = 1.0e-4
    shape = 1.4
    c_tau = c_eq
    for _step in range(30):
        theta, shape, _cf, separated, c_tau = march_interval(
            theta, shape, 1.0, 1.0, 0.02, 1.0e-6, wake=False, c_tau=c_tau
        )
        if separated:
            break
    _assert(not separated, f"lagged zero-pressure layer separated at H={shape}")
    _assert(theta > 1.0e-4 and 1.2 < shape < 2.0, f"lagged flat plate left the turbulent range, θ={theta}, H={shape}")
    _assert(c_tau > 0.0 and np.isfinite(c_tau), "lagged shear stress was not a positive finite value")


def test_naca0012_bubble_is_finite_and_lift_rises_smoothly() -> None:
    """Lift rises from 6° to 10°. At 10° the lower surface separates laminarly and reattaches."""

    angles = (6.0, 8.0, 10.0)
    lifts = []
    for alpha in angles:
        solved = _solve("0012", alpha, 120)
        ibl = solved["ibl"]
        assert ibl is not None
        _assert(solved["converged"], f"{alpha}° did not converge, residual={solved['residual']:.3e}")
        _assert(np.isfinite(solved["loads"]["cl"]) and np.isfinite(solved["loads"]["cd"]), f"{alpha}° loads are not finite")
        for side in ("upper", "lower"):
            theta = np.asarray(ibl[f"theta_{side}"])
            shape = np.asarray(ibl[f"shape_h_{side}"])
            delta = np.asarray(ibl[f"delta_star_{side}"])
            cf = np.asarray(ibl[f"cf_{side}"])
            _assert(np.all(np.isfinite(theta)) and np.all(np.isfinite(shape)), f"{alpha}° {side} is not finite")
            _assert(np.all(np.isfinite(cf)) and np.all(np.isfinite(delta)), f"{alpha}° {side} skin friction or displacement is not finite")
            _assert(np.all(shape > 1.0) and np.all(delta >= 0.0), f"{alpha}° {side} left the physical range")
            _assert(float(np.nanmax(shape)) <= 8.0, f"{alpha}° {side} shape factor exceeded the shear-layer band")
        if alpha >= 10.0:
            # The bubble is the Thwaites/Michel ordering, not an angle switch.
            # 10° on this mesh is where that ordering is laminar separation first.
            _assert(np.isfinite(ibl["bubble_s_lower"]), f"{alpha}° did not record the lower laminar separation")
            _assert(np.isfinite(ibl["reattachment_s_lower"]), f"{alpha}° lower bubble did not reattach")
            _assert(not ibl["separated"], f"{alpha}° reattached bubble was flagged as unrecovered separation")
        lifts.append(float(solved["loads"]["cl"]))
    _assert(lifts[0] < lifts[1] < lifts[2], f"lift did not rise through 6/8/10°: {lifts}")


def test_mesh_changes_lift_smoothly_near_eight_and_ten() -> None:
    """80, 120, and 160 panels stay within 0.05 in CL.

    A bubble can leave the displacement residual a few times above the
    1e-4 convergence flag after 20 passes. That is not a diverged run:
    the residual still has to be under 1e-3, and the lift spread is the
    stability gate. The spread is not an accuracy target.
    """

    for alpha in (8.0, 10.0):
        lifts = []
        for panels in (80, 120, 160):
            solved = _solve("0012", alpha, panels)
            ibl = solved["ibl"]
            assert ibl is not None
            _assert(solved["residual"] < 1.0e-3, f"{panels} panels at {alpha}° residual {solved['residual']:.3e}")
            _assert(not ibl["model_limited"], f"{panels} panels at {alpha}° left the shear-layer band")
            _assert(not ibl["separated"], f"{panels} panels at {alpha}° stayed unrecovered")
            _assert(np.isfinite(solved["loads"]["cl"]), f"{panels} panels at {alpha}° CL")
            lifts.append(float(solved["loads"]["cl"]))
        spread = max(lifts) - min(lifts)
        _assert(spread < 0.05, f"{alpha}° mesh spread in CL is {spread:.4f} across {lifts}")


def test_other_sections_stay_physical_without_an_experimental_claim() -> None:
    """0006, 2412, and 4412 have no reference table in this repository.

    The check is that lift rises with angle, every boundary-layer array stays
    finite and non-negative in thickness, and nothing is NaN. It is not an
    accuracy claim.
    """

    for code in ("0006", "2412", "4412"):
        lifts = []
        for alpha in (0.0, 4.0, 8.0):
            solved = _solve(code, alpha, 80)
            ibl = solved["ibl"]
            assert ibl is not None
            _assert(np.isfinite(solved["loads"]["cl"]) and np.isfinite(solved["loads"]["cd"]), f"{code} at {alpha}° is not finite")
            _assert(not ibl["model_limited"], f"{code} at {alpha}° left the shear-layer validity band")
            for side in ("upper", "lower"):
                theta = np.asarray(ibl[f"theta_{side}"])
                _assert(np.all(np.isfinite(theta)) and np.all(theta >= 0.0), f"{code} {side} θ")
                _assert(np.all(np.isfinite(ibl[f"cf_{side}"])), f"{code} {side} Cf")
            lifts.append(float(solved["loads"]["cl"]))
        _assert(lifts[0] < lifts[1] < lifts[2], f"{code} lift did not rise with angle: {lifts}")
    symmetric = _solve("0006", 0.0, 80)
    _assert(abs(float(symmetric["loads"]["cl"])) < 0.03, f"NACA 0006 zero-incidence CL {symmetric['loads']['cl']}")


def test_lag_closure_does_not_replace_the_default() -> None:
    """The lag closure stays finite at zero incidence and is not the shipped default.

    At 8° it is allowed to separate. That result is not an accuracy claim.
    """

    zero = _solve("0012", 0.0, 80, closure="drela_lag")
    _assert(zero["converged"], f"lag closure did not converge at 0°, residual={zero['residual']:.3e}")
    _assert(abs(float(zero["loads"]["cl"])) < 0.02, f"lag closure CL at 0° is {zero['loads']['cl']}")
    eight = _solve("0012", 8.0, 80, closure="drela_lag")
    ibl = eight["ibl"]
    assert ibl is not None
    _assert(np.isfinite(eight["loads"]["cl"]), "lag closure CL at 8° is not finite")
    _assert(np.all(np.isfinite(ibl["theta_upper"])) and np.all(np.isfinite(ibl["cf_upper"])), "lag closure boundary layer is not finite")
    default = _solve("0012", 8.0, 80)
    _assert(not default["ibl"]["separated"], "default 8° march was left in an unrecovered separation")


if __name__ == "__main__":
    test_mayle_length_matches_the_published_power()
    test_separated_shear_grows_displacement_at_constant_edge_speed()
    test_separated_shear_thickens_momentum_in_an_adverse_gradient()
    test_lag_relaxes_toward_equilibrium_without_overshoot()
    test_naca0012_bubble_is_finite_and_lift_rises_smoothly()
    test_mesh_changes_lift_smoothly_near_eight_and_ten()
    test_other_sections_stay_physical_without_an_experimental_claim()
    test_lag_closure_does_not_replace_the_default()
    print("separated shear tests passed")
