"""Module 3 acceptance checks: lift, pressure, boundary layer, and fixtures.

The rectangular-wing, swept-planform, and cruise skin-friction tests lock
the corrected trailing-edge jump, planform area, and boundary-layer march.
Each records the calculation and the failure it replaced.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from aero_module import AeroEngine  # noqa: E402
from geo_module import GeometryEngine  # noqa: E402
from math_module import MeshEngine  # noqa: E402
from physics_module import (  # noqa: E402
    InvalidFlowConditionError,
    PhysicsEngine,
)
from physics_module.boundary_layer_1d import (  # noqa: E402
    _march_speed,
    _side_state,
    _thwaites_side,
    last_resolved_index,
)
from physics_module.drela_equilibrium import march_interval, turbulent_skin_friction  # noqa: E402
from physics_module.wake_1d import merge_trailing_edge, momentum_drag_is_usable, squire_young  # noqa: E402
from physics_module.forces_and_moments import _reference_area_3d, integrate_loads  # noqa: E402
from physics_module.influence_2d import unit_vortex_velocity  # noqa: E402
from physics_module.solver_2d import solve_2d  # noqa: E402
from physics_module.solver_3d import _trailing_panels  # noqa: E402
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


# Audited rectangular cruise wing: NACA 0012, chord 0.9, span 6, alpha 4 deg.
# Planform area of a constant-chord wing is chord times span. A spanwise
# shear (sweep) has unit Jacobian, so it does not change that area.
_WING_CHORD = 0.9
_WING_SPAN = 6.0
_WING_ALPHA_DEG = 4.0
# One percent of planform. The outline nodes sit on x in [0, chord] and
# y in [-span/2, span/2], so a true planform integral is exact at this
# geometry. One percent is far above faceting roundoff and far below the
# centroid-box errors this suite is meant to catch: the tip centroids stop
# a strip short of the tip (several percent of span at the station counts
# below), and a swept x-extent adds about (span/2)*tan(sweep).
_PLANFORM_RTOL = 0.01


def _wing_mesh(sweep_deg: float, num_sections: int, airfoil_points: int):
    return _mesh(
        {
            "type": "wing",
            "mode": "3D",
            "airfoil": "0012",
            "chord": _WING_CHORD,
            "span": _WING_SPAN,
            "sweep_deg": sweep_deg,
            "dihedral_deg": 0.0,
            "num_sections": num_sections,
            "airfoil_points": airfoil_points,
        },
        {"normal_orientation": "outward", "require_closed_3d": False},
    )


def _helmbold_cl(aspect_ratio: float, alpha_deg: float) -> float:
    """Unswept finite-wing lift from Helmbold's equation at section slope 2π.

    a = a0 / (sqrt(1 + (a0 / (π AR))^2) + a0 / (π AR)), with a0 = 2π.
    """

    a0 = 2.0 * np.pi
    k = a0 / (np.pi * aspect_ratio)
    slope = a0 / (np.sqrt(1.0 + k * k) + k)
    return float(slope * np.deg2rad(alpha_deg))


def _assert_planform_area(s_ref: float, sweep_deg: float) -> None:
    planform = _WING_CHORD * _WING_SPAN
    sweep_offset = 0.5 * _WING_SPAN * np.tan(np.deg2rad(sweep_deg))
    bounding_box = (_WING_CHORD + sweep_offset) * _WING_SPAN
    if abs(sweep_deg) < 1e-8:
        detail = "A centroid bounding box stops short of the tip and the trailing edge."
    else:
        detail = f"An x-bounding box inflated by the sweep offset would be {bounding_box:.4f}."
    _assert(
        abs(s_ref - planform) <= _PLANFORM_RTOL * planform,
        (
            f"s_ref {s_ref:.4f} is outside {_PLANFORM_RTOL:.0%} of the "
            f"constant-chord planform {planform:.4f}. {detail}"
        ),
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
    # Closed body: projected cross-section from the two largest centroid
    # extents. A wing-station integral must not replace this.
    extents = np.ptp(panel["centroids"], axis=0)
    order = np.argsort(extents)
    cross_section = float(extents[order[-1]] * extents[order[-2]])
    _assert(
        abs(float(out["loads"]["s_ref"]) - cross_section) < 1e-9,
        f"cylinder s_ref {out['loads']['s_ref']} left the projected cross-section {cross_section}",
    )


def test_separated_march_returns_payload() -> None:
    """Higher-angle separation check, kept beside the 4 deg cruise Cf test.

    At 10 deg the march is expected to separate. The payload must still
    return, with finite pressure and a warning. That is a different
    requirement from cruise, where a transition/separation tie must not
    wipe skin friction off the aft surface.
    """

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
        PhysicsEngine().solve_from_mesh(mesh, flow, {"ibl_closure": "not_a_closure"})
    except InvalidFlowConditionError as exc:
        _assert("not_a_closure" in str(exc), f"unexpected closure message: {exc}")
    else:
        raise AssertionError("unknown closure must be rejected")
    try:
        PhysicsEngine().solve_from_mesh(mesh, flow, {"viscous_coupling": "displacement"})
    except InvalidFlowConditionError as exc:
        _assert("displacement" in str(exc), f"unexpected displacement message: {exc}")
    else:
        raise AssertionError("displacement coupling must be rejected")


def test_equilibrium_shear_layer_and_wake_join() -> None:
    """Drela's equilibrium closures on a zero-pressure-gradient layer, and the trailing-edge join.

    The skin friction at H = 1.4 and Re_θ = 5000 is the Coles/CFT value,
    a few thousandths. A flat-plate march must grow θ and keep H near the
    equilibrium value. A zero-pressure-gradient wake must hold θ, because
    there is no wall shear and no pressure force. The join keeps the
    displacement flux and the momentum defect; a stagnant base adds to
    δ* only.
    """

    cf = turbulent_skin_friction(1.4, 5000.0)
    _assert(0.002 < cf < 0.004, f"equilibrium Cf {cf}")
    _assert(
        turbulent_skin_friction(1.4, 20000.0) < cf,
        "skin friction did not fall as Re_θ increased",
    )
    nu = 1.0e-6
    theta = 1.0e-4
    shape = 1.4
    separated = False
    for _step in range(40):
        theta, shape, cf, separated = march_interval(
            theta, shape, 1.0, 1.0, 0.025, nu, wake=False
        )
        if separated:
            break
    _assert(not separated, f"zero-pressure-gradient wall layer separated at H={shape}")
    _assert(theta > 5.0e-4, f"momentum thickness did not grow: {theta}")
    _assert(1.2 < shape < 1.7, f"shape factor left the turbulent range: {shape}")
    target = 0.036 / (1.0e6 ** 0.2)
    _assert(0.5 * target < theta < 2.0 * target, f"θ {theta} is not near the 1/7-power value {target}")

    theta_w = 0.002
    shape_w = 1.45
    for _step in range(20):
        theta_w, shape_w, _cf, separated = march_interval(
            theta_w, shape_w, 1.0, 1.0, 0.05, nu, wake=True
        )
        if separated:
            break
    _assert(not separated, "zero-pressure-gradient wake separated")
    _assert(abs(theta_w - 0.002) / 0.002 < 0.02, f"wake θ changed to {theta_w}")
    _assert(shape_w < 1.45, f"wake shape factor did not fall: {shape_w}")

    ue, th, dst, _h = merge_trailing_edge(1.0, 0.002, 0.001, 0.8, 0.001, 0.0005, 0.0)
    flux = 1.0 * 0.002 + 0.8 * 0.001
    momentum = 1.0 * 0.001 + 0.64 * 0.0005
    _assert(abs(ue * dst - flux) < 1.0e-12, "displacement flux was not kept")
    _assert(abs(ue * ue * th - momentum) < 1.0e-12, "momentum defect was not kept")
    _ue_g, th_g, dst_g, _h_g = merge_trailing_edge(1.0, 0.002, 0.001, 1.0, 0.001, 0.0005, 0.01)
    _assert(abs(th_g - 0.0015) < 1.0e-12, "base height changed the momentum thickness")
    _assert(abs(dst_g - 0.013) < 1.0e-12, "base height missing from the displacement thickness")
    _assert(abs(squire_young(0.002, 1.0, 1.0, 1.0) - 0.004) < 1.0e-12, "Squire–Young at Ue = V∞")
    _assert(momentum_drag_is_usable(0.006, 1.2, 0.99, False), "an attached wake defect was rejected")
    _assert(not momentum_drag_is_usable(0.006, 2.8, 0.99, False), "a separated shape factor was accepted")
    _assert(not momentum_drag_is_usable(0.006, 1.2, 0.99, True), "a separated wake march was accepted")


def test_wake_starts_at_resolved_edge_and_drag_stays_on_the_surface() -> None:
    """The cosine cluster is not the wake's edge speed, and it does not replace the force drag."""

    mesh = _mesh(
        {"type": "naca", "mode": "2D", "code": "0012", "chord": 1.0, "num_points": 160},
        {"n_panels": 80, "n_x": 16, "n_y": 12, "clustering": "cosine", "te_cluster_factor": 1.0},
    )
    flow = {"alpha_deg": 0.0, "reynolds": 6.0e6, "mach": 0.0, "v_inf": 1.0, "rho": 1.225}
    head = PhysicsEngine().solve_from_mesh(mesh, flow, {"regime": "viscous_inviscid", "prandtl_glauert": False})
    ibl = head["ibl"]
    assert ibl is not None
    stations = mesh["ibl_1d"]
    assert stations is not None
    upper = last_resolved_index(stations["s_upper"])
    lower = last_resolved_index(stations["s_lower"])
    expected = 0.5 * (float(ibl["ue_upper"][upper]) + float(ibl["ue_lower"][lower]))
    _assert(abs(float(ibl["wake_ue"][0]) - expected) < 1.0e-9, "wake did not start at the resolved edge speed")
    _assert(
        abs(float(ibl["ue_upper"][-1]) - float(ibl["ue_upper"][upper])) < 1.0e-12,
        "the march kept the unresolved trailing-edge speed",
    )
    loads = head["loads"]
    surface = float(loads["cd_pressure"] + loads["cd_friction"])
    _assert(abs(float(loads["cd"]) - surface) < 1.0e-12, "published drag left the surface integral")
    _assert(np.isfinite(float(ibl["cd_momentum"])) and float(ibl["cd_momentum"]) > 0.0, "wake defect is not finite")
    _assert(abs(float(head["loads"]["cl"])) < 0.02, f"zero-incidence CL {head['loads']['cl']}")

    drela = PhysicsEngine().solve_from_mesh(
        mesh,
        flow,
        {"regime": "viscous_inviscid", "prandtl_glauert": False, "ibl_closure": "drela_two_equation"},
    )
    drela_ibl = drela["ibl"]
    assert drela_ibl is not None
    _assert(drela["converged"], f"equilibrium closure did not converge, residual={drela['residual']:.3e}")
    _assert(abs(float(drela["loads"]["cl"])) < 0.02, f"equilibrium closure CL {drela['loads']['cl']}")
    _assert(float(np.max(drela_ibl["theta_upper"])) > 0.0, "equilibrium march did not grow θ")
    _assert(np.isfinite(float(drela_ibl["cd_momentum"])), "equilibrium wake defect is not finite")


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


def test_rectangular_wing_planform_and_lift() -> None:
    """NACA 0012 rectangular wing, chord 0.9, span 6, alpha 4 deg.

    The skin uses 12 spanwise sections and 32 profile points. That is the
    audited planform with a coarser triangulation than the pipeline case
    (20 sections, 64 points, about 2400 panels). The coarser skin stays
    on a dense solve of a few seconds. Planform area does not depend on
    that count. The lift target is a continuum finite-wing estimate, and
    the tolerance below is wider than normal panel truncation between
    this skin and a finer one.

    Reference area is the geometric planform, 0.9 * 6 = 5.40. See
    ``_PLANFORM_RTOL``.

    Lift. Aspect ratio is span/chord = 6.667. Thin-aerofoil section slope
    a0 = 2π per radian, and Helmbold's unswept-wing equation supplies the
    finite-wing slope (it sits slightly below elliptic lifting line, which
    is the right side for a rectangular planform). The absolute band is
    ±0.08, the same allowance ``test_flat_plate_and_naca_lift`` already
    uses for a NACA 0012 panel solution. On a lift coefficient near 0.33
    that is about a quarter of the estimate: enough for a rectangular
    planform and for this triangulation, and not a fit to a measured run.
    """

    mesh = _wing_mesh(sweep_deg=0.0, num_sections=12, airfoil_points=32)
    solved = PhysicsEngine().solve_from_mesh(mesh, *_flow(_WING_ALPHA_DEG))
    report = AeroEngine().analyze_from_solution(mesh, solved)
    s_ref = float(report["reference"]["s_ref"])
    cl = float(report["coefficients"]["cl"])
    _assert(abs(s_ref - float(solved["loads"]["s_ref"])) < 1e-9, "aero reference area was rescaled")
    _assert(abs(cl - float(solved["loads"]["cl"])) < 1e-9, "aero lift coefficient was rescaled")
    aspect = _WING_SPAN / _WING_CHORD
    expected = _helmbold_cl(aspect, _WING_ALPHA_DEG)
    tolerance = 0.08
    gaps: list[str] = []
    try:
        _assert_planform_area(s_ref, sweep_deg=0.0)
    except AssertionError as exc:
        gaps.append(str(exc))
    try:
        _assert(
            abs(cl - expected) < tolerance,
            (
                f"rectangular-wing Cl {cl:.4f} is outside Helmbold {expected:.4f} +/- {tolerance:.2f} "
                f"(AR={aspect:.3f}, alpha={_WING_ALPHA_DEG:.1f} deg, a0=2*pi)"
            ),
        )
    except AssertionError as exc:
        gaps.append(str(exc))
    panel = mesh["panel_3d"]
    assert panel is not None
    normals = np.asarray(panel["normals"], dtype=np.float64)
    areas = np.asarray(panel["areas"], dtype=np.float64)
    cp = np.asarray(solved["inviscid"]["cp_inviscid"], dtype=np.float64)
    upper = normals[:, 2] >= 0.0
    cp_upper = float(np.sum(cp[upper] * areas[upper]) / np.sum(areas[upper]))
    cp_lower = float(np.sum(cp[~upper] * areas[~upper]) / np.sum(areas[~upper]))
    try:
        # Positive incidence puts suction on the upper surface. The gap is
        # only required to be a real fraction of the lift coefficient, not
        # a measured pressure plot.
        _assert(
            cp_upper < cp_lower - 0.05,
            f"upper Cp {cp_upper:.3f} is not below lower Cp {cp_lower:.3f}",
        )
    except AssertionError as exc:
        gaps.append(str(exc))
    upper_idx, lower_idx = _trailing_panels(panel)
    centroids = np.asarray(panel["centroids"], dtype=np.float64)
    order = np.argsort(centroids[upper_idx, 1])
    mu = np.asarray(solved["inviscid"]["mu"], dtype=np.float64)
    jump = mu[upper_idx[order]] - mu[lower_idx[order]]
    y = centroids[upper_idx[order], 1]
    central = np.abs(y - float(np.mean(y))) <= 0.25 * float(np.ptp(y))
    try:
        _assert(
            float(np.mean(jump[central])) > 0.0,
            f"central trailing-edge doublet jump {float(np.mean(jump[central])):.4f} is not lifting",
        )
    except AssertionError as exc:
        gaps.append(str(exc))
    if gaps:
        raise AssertionError(" ".join(gaps))


def test_wing_direct_and_gmres_lift_agree() -> None:
    """The same wake matrix, factored directly and by GMRES.

    Eight sections keep this on a small dense matrix. The absolute lift
    band is the finer rectangular case. Here the two solvers must return
    the same coefficient: GMRES stops at 1e-6 relative residual, so 0.005
    on Cl is a wide envelope around that residual, not a second theory.
    """

    mesh = _wing_mesh(sweep_deg=0.0, num_sections=8, airfoil_points=24)
    flow, params = _flow(_WING_ALPHA_DEG)
    direct = PhysicsEngine().solve_from_mesh(mesh, flow, params)
    iterative = PhysicsEngine().solve_from_mesh(mesh, flow, {**params, "dense_panel_limit": 1})
    _assert_planform_area(float(direct["loads"]["s_ref"]), sweep_deg=0.0)
    _assert_planform_area(float(iterative["loads"]["s_ref"]), sweep_deg=0.0)
    _assert(
        abs(float(direct["loads"]["cl"]) - float(iterative["loads"]["cl"])) < 0.005,
        f"direct Cl {direct['loads']['cl']:.4f} vs GMRES {iterative['loads']['cl']:.4f}",
    )
    _assert(any("GMRES" in note for note in iterative["warnings"]), "GMRES path was not used")


def test_swept_constant_chord_planform_area() -> None:
    """Constant-chord NACA 0012 at 0 deg and 25 deg sweep, zero dihedral.

    Same chord and span as the rectangular cruise wing, so both planforms
    are 5.40. Sweep shifts each section aft by |y| tan(Λ). The global
    x-extent then grows by (span/2) tan(Λ) ≈ 1.40, and a bounding box of
    that extent times span is about 13.8. Section chord is unchanged
    because the shear Jacobian is 1, so the two reference areas have to
    match each other as well as the planform. Eight sections are enough:
    the difference is geometric. The relative match uses 1e-9, which is
    roundoff on a trapezoid of identical section chords.
    """

    unswept = PhysicsEngine().solve_from_mesh(
        _wing_mesh(sweep_deg=0.0, num_sections=8, airfoil_points=24),
        *_flow(_WING_ALPHA_DEG),
    )
    sweep = 25.0
    swept = PhysicsEngine().solve_from_mesh(
        _wing_mesh(sweep_deg=sweep, num_sections=8, airfoil_points=24),
        *_flow(_WING_ALPHA_DEG),
    )
    area_0 = float(unswept["loads"]["s_ref"])
    area_25 = float(swept["loads"]["s_ref"])
    _assert_planform_area(area_0, sweep_deg=0.0)
    _assert_planform_area(area_25, sweep_deg=sweep)
    _assert(
        abs(area_25 - area_0) <= 1e-9 * max(area_0, 1.0),
        f"sweep changed the planform from {area_0:.6f} to {area_25:.6f}",
    )


def test_open_patch_keeps_centroid_box_area() -> None:
    """An open surface with no spanwise stations is not given a wing chord.

    Every vertex has its own y, so a station chord cannot be formed.
    ``s_ref`` stays the product of the centroid x-extent and y-extent.
    """

    facets = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.3, 0.0], [0.2, 1.0, 0.1]],
            [[1.0, 0.3, 0.0], [1.2, 1.1, 0.0], [0.2, 1.0, 0.1]],
        ],
        dtype=np.float64,
    )
    centroids = np.mean(facets, axis=1)
    mesh = {
        "panel_3d": {
            "facets": facets,
            "centroids": centroids,
            "areas": np.ones(facets.shape[0], dtype=np.float64),
            "is_closed": False,
        }
    }
    s_ref, _length, _origin, _centroid = _reference_area_3d(mesh)  # type: ignore[arg-type]
    expected = float(np.ptp(centroids[:, 0]) * np.ptp(centroids[:, 1]))
    _assert(abs(s_ref - expected) < 1e-12, f"patch s_ref {s_ref} is not the centroid box {expected}")


def test_cruise_transition_does_not_zero_aft_skin_friction() -> None:
    """NACA 0012 at 4 deg and Re_c = 500_000 must keep aft skin friction.

    This angle is well below stall for a 0012 at this Reynolds number
    (stall in the Abbott & von Doenhoff data is near 12–16 deg). On the
    upper surface the suction peak is forward, Michel transition and the
    Thwaites laminar-separation threshold (λ ≤ -0.09) land near the same
    station, and the march has to enter the turbulent closure there. A
    tie that freezes Cf to 0 from that station to the trailing edge is
    the failure. The 10 deg case in ``test_separated_march_returns_payload``
    remains the check that a genuinely high-incidence march may separate.

    Stations: upper surface, 0.40 ≤ x/c ≤ 0.90. Transition on this
    section is expected ahead of 0.40c, and the last 10 percent of chord
    is left out so a short trailing-edge patch is not the whole answer.
    Flat-plate turbulent skin friction, Cf ≈ 0.059 Re_x^{-1/5}, is about
    0.005 at x/c = 0.40 and Re_c = 5e5. The floor 5e-4 is about a tenth
    of that, which still clears an attached turbulent layer in an adverse
    gradient and rejects an exact zero. At least 70 percent of the window
    must clear the floor, so one station inside a bubble does not fail
    the test and a wiped aft surface does.
    """

    mesh = _mesh(
        {"type": "naca", "mode": "2D", "code": "0012", "chord": 1.0, "num_points": 80},
        {"n_panels": 64, "n_x": 24, "n_y": 16, "clustering": "cosine"},
    )
    solved = PhysicsEngine().solve_from_mesh(mesh, *_flow(4.0, "viscous_inviscid"))
    ibl = solved["ibl"]
    stations = mesh["ibl_1d"]
    assert ibl is not None and stations is not None
    x_over_c = np.asarray(stations["xy_upper"], dtype=np.float64)[:, 0]
    cf = np.asarray(ibl["cf_upper"], dtype=np.float64)
    window = (x_over_c >= 0.40) & (x_over_c <= 0.90)
    aft = cf[window]
    _assert(aft.size >= 8, f"aft Cf window has {aft.size} stations")
    _assert(np.all(np.isfinite(aft)), "aft skin friction must be finite")
    floor = 5.0e-4
    fraction = float(np.mean(aft > floor))
    _assert(
        fraction >= 0.70,
        (
            "upper-surface skin friction is zeroed across the aft surface: "
            f"median Cf={float(np.median(aft)):.3e}, fraction above {floor:.1e}={fraction:.2f}, "
            f"transition_s={float(ibl['transition_s_upper']):.4f}, "
            f"separation_s={float(ibl['separation_s_upper']):.4f}"
        ),
    )


def test_tied_transition_stays_attached() -> None:
    """The 4 deg tie must enter Head and must not separate on the trailing cluster.

    Transition on this mesh is near x/c = 0.26. An attached turbulent
    shape factor on a 0012 in a mild adverse gradient sits near 1.4–1.8;
    the laminar separation correlation is about 2.99 and Head's own
    threshold is 2.8. The window 1.2–2.5 is above a laminar Blasius value
    only at the low end and stops short of either separation threshold,
    so it passes only if the turbulent march actually ran and stayed
    attached. Skin friction uses the same floor as the cruise test and a
    ceiling of 0.02, half the march's own Cf cap, which still sits above
    the flat-plate turbulent value near 0.005.
    """

    mesh = _mesh(
        {"type": "naca", "mode": "2D", "code": "0012", "chord": 1.0, "num_points": 80},
        {"n_panels": 64, "n_x": 24, "n_y": 16, "clustering": "cosine"},
    )
    solved = PhysicsEngine().solve_from_mesh(mesh, *_flow(4.0, "viscous_inviscid"))
    ibl = solved["ibl"]
    stations = mesh["ibl_1d"]
    assert ibl is not None and stations is not None
    xy = np.asarray(stations["xy_upper"], dtype=np.float64)
    window = (xy[:, 0] >= 0.40) & (xy[:, 0] <= 0.90)
    shape = np.asarray(ibl["shape_h_upper"], dtype=np.float64)[window]
    cf = np.asarray(ibl["cf_upper"], dtype=np.float64)[window]
    theta = np.asarray(ibl["theta_upper"], dtype=np.float64)[window]
    _assert(np.isfinite(ibl["transition_s_upper"]), "upper transition was not recorded")
    _assert(
        ibl["transition_s_upper"] < 0.40,
        f"transition s={ibl['transition_s_upper']:.4f} is not ahead of the aft window",
    )
    _assert(
        not np.isfinite(ibl["separation_s_upper"]),
        f"upper separation s={ibl['separation_s_upper']} is a tie or a trailing-edge spike",
    )
    _assert(np.all(np.isfinite(shape)) and np.all(np.isfinite(cf)) and np.all(np.isfinite(theta)), "aft boundary layer is not finite")
    _assert(np.all((shape > 1.2) & (shape < 2.5)), f"aft shape factor {float(np.min(shape)):.3f}..{float(np.max(shape)):.3f}")
    _assert(float(np.median(cf)) > 5.0e-4, f"aft median Cf={float(np.median(cf)):.3e}")
    _assert(float(np.median(cf)) < 0.02, f"aft median Cf={float(np.median(cf)):.3e} is above a plausible turbulent value")
    _assert(np.all(theta > 0.0), "momentum thickness vanished in the aft window")


def test_laminar_separation_before_transition_still_freezes() -> None:
    """At 10 deg, Thwaites trips on an earlier station than Michel.

    That ordering is real laminar separation, not the 4 deg tie. Downstream
    skin friction stays at the separated value of zero. The existing 10 deg
    test still requires the warning and a finite pressure field.
    """

    mesh = _mesh(
        {"type": "naca", "mode": "2D", "code": "0012", "chord": 1.0, "num_points": 80},
        {"n_panels": 64, "n_x": 24, "n_y": 16, "clustering": "cosine"},
    )
    solved = PhysicsEngine().solve_from_mesh(mesh, *_flow(10.0, "viscous_inviscid"))
    ibl = solved["ibl"]
    stations = mesh["ibl_1d"]
    assert ibl is not None and stations is not None
    _assert(ibl["separated"], "10 deg march should separate")
    _assert(np.isfinite(ibl["separation_s_upper"]), "upper separation station missing")
    _assert(np.isfinite(ibl["transition_s_upper"]), "upper transition station missing")
    _assert(
        float(ibl["separation_s_upper"]) < float(ibl["transition_s_upper"]),
        (
            f"separation s={ibl['separation_s_upper']:.4f} is not upstream of "
            f"transition s={ibl['transition_s_upper']:.4f}"
        ),
    )
    arc = np.asarray(stations["s_upper"], dtype=np.float64)
    downstream = arc >= float(ibl["separation_s_upper"])
    cf = np.asarray(ibl["cf_upper"], dtype=np.float64)
    _assert(int(np.count_nonzero(downstream)) >= 4, "separation has too few downstream stations")
    _assert(float(np.max(np.abs(cf[downstream]))) == 0.0, "laminar separation did not freeze skin friction")
    _assert(np.all(np.isfinite(cf)), "separated skin friction must stay finite")
    _assert(any("separated" in note.lower() for note in solved["warnings"]), "separation needs a warning")
    _assert(
        any("not a validated drag result" in note.lower() for note in solved["warnings"]),
        "laminar freeze must not be reported as a complete drag solution",
    )
    report = AeroEngine().analyze_from_solution(mesh, solved)
    _assert(
        not np.isfinite(report["l_over_d"]),
        "L/D stayed finite after laminar separation omitted the downstream friction",
    )


def test_trailing_edge_cluster_does_not_force_separation() -> None:
    """A resolved zero pressure gradient plus a clustered trailing-edge drop.

    Median spacing is about 0.04. The last gaps are 0.006 and smaller, under
    a quarter of that median, so they are the cluster. Edge speed is uniform
    on the resolved stations and then falls from 1 to 0.5. That drop would
    separate a march that differentiates into the cluster. The march holds
    the cluster speed and its slope, so the layer stays the zero-pressure
    gradient. At Re = 2e6 Michel still transitions on the flat portion.
    """

    body = np.linspace(0.0, 0.98, 25)
    arc = np.concatenate((body, np.array([0.990, 0.996, 0.999, 1.0], dtype=np.float64)))
    speed = np.ones(arc.shape, dtype=np.float64)
    speed[-4:] = np.array([0.95, 0.80, 0.65, 0.50], dtype=np.float64)
    state = _side_state(arc, speed, nu=1.0 / 2.0e6)
    cf = np.asarray(state["cf"], dtype=np.float64)
    _assert(not bool(state["separated_mask"]), f"cluster drop separated at s={state['separation_s']}")
    _assert(np.isfinite(float(state["transition_s"])), "flat portion did not transition")
    _assert(np.all(np.isfinite(cf)), "synthetic skin friction must be finite")
    _assert(float(np.median(cf[8:20])) > 5.0e-4, f"resolved Cf median={float(np.median(cf[8:20])):.3e}")


def test_skin_friction_drag_uses_freestream_dynamic_pressure() -> None:
    """Local Cf is wall shear over edge dynamic pressure. CD uses freestream q.

    One upper-surface segment, chord 1, tangent along the drag axis, and the
    unit onset V_inf = 1 used by the load integration. Cf = 0.01 at U_e = 2
    contributes 0.04. The same segment at U_e = V_inf contributes 0.01.
    Pressure is zero, so the result is the friction integral alone.
    """

    def loads(edge_speed: float):
        cf_upper = np.array([0.01, 0.01], dtype=np.float64)
        ue_upper = np.array([edge_speed, edge_speed], dtype=np.float64)
        mesh = {
            "mode": "2D",
            "panel_2d": {
                "normal": np.array([[0.0, 1.0]], dtype=np.float64),
                "length": np.array([1.0], dtype=np.float64),
                "collocation": np.array([[0.5, 0.0]], dtype=np.float64),
                "nodes": np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
                "le_index": 0,
                "chord": 1.0,
            },
            "ibl_1d": {
                "tangent_upper": np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64),
                "ds_upper": np.array([1.0], dtype=np.float64),
                "tangent_lower": np.zeros((0, 2), dtype=np.float64),
                "ds_lower": np.zeros(0, dtype=np.float64),
            },
        }
        inviscid = {
            "gamma": None,
            "cp_inviscid": np.zeros(1, dtype=np.float64),
        }
        ibl = {
            "cf_upper": cf_upper,
            "cf_lower": np.zeros(0, dtype=np.float64),
            "ue_upper": ue_upper,
            "ue_lower": np.zeros(0, dtype=np.float64),
        }
        cf_before = cf_upper.copy()
        ue_before = ue_upper.copy()
        solved = integrate_loads(mesh, inviscid, ibl, 0.0, False)  # type: ignore[arg-type]
        _assert(np.array_equal(cf_upper, cf_before), "local Cf was overwritten")
        _assert(np.array_equal(ue_upper, ue_before), "edge speed was overwritten")
        return solved

    doubled = loads(2.0)
    matched = loads(1.0)
    _assert(abs(doubled["cd_pressure"]) < 1e-12, "pressure drag leaked into the friction fixture")
    _assert(abs(matched["cd_pressure"]) < 1e-12, "pressure drag leaked into the unit-speed fixture")
    _assert(
        abs(doubled["cd_friction"] - 0.04) < 1e-12,
        f"Ue=2 freestream friction drag is {doubled['cd_friction']}, expected 0.04",
    )
    _assert(
        abs(matched["cd_friction"] - 0.01) < 1e-12,
        f"Ue=1 freestream friction drag is {matched['cd_friction']}, expected 0.01",
    )
    _assert(abs(doubled["cd"] - doubled["cd_friction"]) < 1e-12, "total drag left the friction integral")
    _assert(abs(matched["cd"] - matched["cd_friction"]) < 1e-12, "unit-speed total drag left the friction integral")


def _lambda_at(arc: np.ndarray, edge_speed: np.ndarray, index: int, nu: float) -> float:
    """Thwaites λ on the marched edge speed, the value the hysteresis test uses."""

    marched = _march_speed(edge_speed)
    _theta, _shape, _cf, lam, _delta_star = _thwaites_side(arc, marched, nu)
    return float(lam[index])


def _aft_deceleration(arc: np.ndarray, drop: float) -> np.ndarray:
    """Attached front, then a linear aft deceleration. The leading edge stays the slowest early station."""

    speed = np.ones(arc.size, dtype=np.float64)
    speed[0] = 0.99
    aft = arc.size // 2
    speed[aft:] = np.linspace(1.05, 1.05 - drop, arc.size - aft)
    return speed


def _drop_for_lambda(arc: np.ndarray, index: int, nu: float, target: float) -> float:
    low = 0.02
    high = 0.2
    for _ in range(40):
        mid = 0.5 * (low + high)
        if _lambda_at(arc, _aft_deceleration(arc, mid), index, nu) < target:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


def test_marginal_rebound_does_not_revoke_laminar_separation() -> None:
    """A prior laminar separation survives λ ≈ -0.089 and releases above -0.08.

    New separation still requires λ ≤ -0.09. The station just above that
    line is not a new separation, but it is inside the retain band, so the
    previous station stays frozen. Past -0.08 the same retained index is
    dropped.
    """

    arc = np.linspace(0.0, 1.0, 60)
    nu = 1.0 / 5.0e5
    station = 50
    marginal_drop = _drop_for_lambda(arc, station, nu, -0.089)
    marginal = _aft_deceleration(arc, marginal_drop)
    marginal_lambda = _lambda_at(arc, marginal, station, nu)
    _assert(
        -0.09 < marginal_lambda <= -0.08 and abs(marginal_lambda + 0.089) < 1.0e-4,
        f"marginal λ at the retained station is {marginal_lambda:.5f}, not about -0.089",
    )
    fresh = _side_state(arc, marginal, nu=nu)
    _assert(
        int(fresh["separated_at"]) != station,
        "the -0.09 rule selected a station that is only inside the retain band",
    )
    held = _side_state(arc, marginal, nu=nu, retained_index=station)
    _assert(int(held["separated_at"]) == station, "marginal rebound revoked the previous separation")
    _assert(float(np.max(np.abs(held["cf"][station:]))) == 0.0, "retained separation did not freeze skin friction")

    recovered_drop = _drop_for_lambda(arc, station, nu, -0.07)
    recovered = _aft_deceleration(arc, recovered_drop)
    recovered_lambda = _lambda_at(arc, recovered, station, nu)
    _assert(recovered_lambda > -0.08, f"recovery λ {recovered_lambda:.5f} did not pass the release level")
    released = _side_state(arc, recovered, nu=nu, retained_index=station)
    _assert(
        int(released["separated_at"]) != station,
        "λ beyond the release level kept the old separation station",
    )


def test_fine_cruise_coupling_converges_without_separation_chatter() -> None:
    """160-panel NACA 0012 at 4 deg must not wipe aft skin friction.

    A graze on the single station before transition is the same two-station
    tie a coarser mesh lands on the transition index. Both meshes stay
    attached, converge inside the iteration cap, and keep the same friction
    drag to within 25 percent. A deep adjacent hit, such as 10 deg, still
    freezes.
    """

    fine = _mesh(
        {"type": "naca", "mode": "2D", "code": "0012", "chord": 1.0, "num_points": 160},
        {
            "n_panels": 160,
            "n_x": 121,
            "n_y": 61,
            "clustering": "cosine",
            "le_cluster_factor": 1.0,
            "te_cluster_factor": 1.0,
        },
    )
    solved = PhysicsEngine().solve_from_mesh(fine, *_flow(4.0, "viscous_inviscid"))
    ibl = solved["ibl"]
    assert ibl is not None
    _assert(solved["converged"], f"fine cruise coupling did not converge, residual={solved['residual']:.3e}")
    _assert(solved["residual"] < 1.0e-4, f"fine cruise residual {solved['residual']:.3e}")
    _assert(int(solved["vic_iterations"]) <= 8, f"fine cruise used {solved['vic_iterations']} iterations")
    _assert(
        not np.isfinite(float(ibl["separation_s_upper"])),
        f"160-panel upper surface froze at {ibl['separation_s_upper']}",
    )
    _assert(float(solved["loads"]["cd_friction"]) > 0.004, "fine-mesh friction was wiped")

    coarse = _mesh(
        {"type": "naca", "mode": "2D", "code": "0012", "chord": 1.0, "num_points": 80},
        {"n_panels": 64, "n_x": 24, "n_y": 16, "clustering": "cosine"},
    )
    coarse_solved = PhysicsEngine().solve_from_mesh(coarse, *_flow(4.0, "viscous_inviscid"))
    coarse_ibl = coarse_solved["ibl"]
    assert coarse_ibl is not None
    _assert(coarse_solved["converged"], f"64-panel control did not converge, residual={coarse_solved['residual']:.3e}")
    _assert(coarse_solved["residual"] < 1.0e-4, f"64-panel residual {coarse_solved['residual']:.3e}")
    _assert(
        not np.isfinite(float(coarse_ibl["separation_s_upper"])),
        f"64-panel upper surface separated at {coarse_ibl['separation_s_upper']}",
    )
    fine_friction = float(solved["loads"]["cd_friction"])
    coarse_friction = float(coarse_solved["loads"]["cd_friction"])
    _assert(
        abs(fine_friction - coarse_friction) / coarse_friction < 0.25,
        f"friction drag jumped from {coarse_friction:.5f} at 64 panels to {fine_friction:.5f} at 160",
    )


def test_transpiration_changes_circulation() -> None:
    """Upper-surface blowing must reduce circulation. Uniform blowing must not.

    The Dirichlet data is the contour integral of the outward transpiration.
    A closed body cannot shed a net volume flux without a wake sheet, so a
    uniform blowing is removed and the vortex strengths stay on the
    solid-wall solution. Swapping which side blows must move the
    circulation in opposite directions. The coupled 0012 and 2412 lifts
    have to fall relative to their own inviscid solutions. The size of
    that fall is not fitted to a measured polar.
    """

    mesh = _mesh(
        {"type": "naca", "mode": "2D", "code": "0012", "chord": 1.0, "num_points": 160},
        {"n_panels": 80, "n_x": 16, "n_y": 12, "clustering": "cosine"},
    )
    panel = mesh["panel_2d"]
    assert panel is not None
    alpha = np.deg2rad(4.0)
    length = np.asarray(panel["length"], dtype=np.float64)
    count = int(length.shape[0])
    solid, _, _ = solve_2d(panel, alpha, None, "vortex", True)
    uniform, _, _ = solve_2d(panel, alpha, np.full(count, 0.03), "vortex", True)
    _assert(
        np.allclose(uniform["gamma"], solid["gamma"], rtol=0.0, atol=1e-8),
        "uniform transpiration changed the vortex strengths",
    )

    # One smooth wavelength around the contour: outward on the upper
    # branch (TE → LE) and inward on the lower branch. A step at the
    # leading edge is finer than this panel spacing can enforce.
    phase = 2.0 * np.pi * (np.arange(count, dtype=np.float64) + 0.5) / count
    differential = 0.02 * np.sin(phase)
    upper_blown, _, _ = solve_2d(panel, alpha, differential, "vortex", True)
    lower_blown, _, _ = solve_2d(panel, alpha, -differential, "vortex", True)
    circulation = lambda solution: float(np.sum(np.asarray(solution["gamma"]) * length))
    solid_gamma = circulation(solid)
    upper_gamma = circulation(upper_blown)
    lower_gamma = circulation(lower_blown)
    _assert(
        upper_gamma < solid_gamma - 0.01,
        f"upper blowing left circulation at {upper_gamma:.4f}; solid wall is {solid_gamma:.4f}",
    )
    _assert(
        lower_gamma > solid_gamma + 0.01,
        f"lower blowing left circulation at {lower_gamma:.4f}; solid wall is {solid_gamma:.4f}",
    )
    _assert(float(np.max(np.abs(upper_blown["gamma"]))) < 5.0, "transpiration spiked the vortex sheet")

    compatible = differential - float(np.sum(differential * length) / np.sum(length))
    nodes = np.asarray(panel["nodes"], dtype=np.float64)
    tangent = np.asarray(panel["tangent"], dtype=np.float64)
    normal = np.asarray(panel["normal"], dtype=np.float64)
    collocation = np.asarray(panel["collocation"], dtype=np.float64)
    vortex_u, vortex_v = unit_vortex_velocity(collocation, nodes, tangent, length)
    vortex_u = np.array(vortex_u, copy=True)
    vortex_v = np.array(vortex_v, copy=True)
    vortex_u[np.diag_indices(count)] = 0.0
    vortex_v[np.diag_indices(count)] = 0.0
    freestream = np.array([np.cos(alpha), np.sin(alpha)], dtype=np.float64)
    gamma = np.asarray(upper_blown["gamma"], dtype=np.float64)
    solid_gamma_panel = np.asarray(solid["gamma"], dtype=np.float64)

    def outward_speed(strength: np.ndarray) -> np.ndarray:
        return (
            (vortex_u @ strength) * normal[:, 0]
            + (vortex_v @ strength) * normal[:, 1]
            + normal @ freestream
        )

    induced = outward_speed(gamma) - outward_speed(solid_gamma_panel)
    correlation = float(np.corrcoef(induced, compatible)[0, 1])
    slope = float(np.dot(induced, compatible) / np.dot(compatible, compatible))
    _assert(
        correlation > 0.95 and 0.8 < slope < 1.2,
        f"blowing did not become outward speed: correlation {correlation:.3f}, slope {slope:.3f}",
    )

    flow = {"alpha_deg": 4.0, "reynolds": 6.0e6, "mach": 0.0, "v_inf": 1.0, "rho": 1.225}
    zero_flow = {"alpha_deg": 0.0, "reynolds": 6.0e6, "mach": 0.0, "v_inf": 1.0, "rho": 1.225}
    for code in ("0012", "2412"):
        section = _mesh(
            {"type": "naca", "mode": "2D", "code": code, "chord": 1.0, "num_points": 160},
            {"n_panels": 80, "n_x": 16, "n_y": 12, "clustering": "cosine"},
        )
        inviscid = PhysicsEngine().solve_from_mesh(section, flow, {"regime": "inviscid", "prandtl_glauert": False})
        viscous = PhysicsEngine().solve_from_mesh(
            section, flow, {"regime": "viscous_inviscid", "prandtl_glauert": False}
        )
        _assert(viscous["converged"], f"NACA {code} coupling did not converge, residual={viscous['residual']:.3e}")
        _assert(
            float(viscous["loads"]["cl"]) < float(inviscid["loads"]["cl"]) - 0.01,
            (
                f"NACA {code} viscous CL {viscous['loads']['cl']:.4f} did not fall below "
                f"inviscid CL {inviscid['loads']['cl']:.4f}"
            ),
        )
        zero = PhysicsEngine().solve_from_mesh(
            section, zero_flow, {"regime": "viscous_inviscid", "prandtl_glauert": False}
        )
        zero_inviscid = PhysicsEngine().solve_from_mesh(
            section, zero_flow, {"regime": "inviscid", "prandtl_glauert": False}
        )
        if code == "0012":
            _assert(abs(float(zero["loads"]["cl"])) < 0.02, f"NACA 0012 zero-incidence CL {zero['loads']['cl']}")
        else:
            _assert(
                float(zero["loads"]["cl"]) > 0.5 * float(zero_inviscid["loads"]["cl"]),
                f"NACA {code} lost its camber lift at zero incidence: {zero['loads']['cl']}",
            )


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
    test_equilibrium_shear_layer_and_wake_join()
    test_wake_starts_at_resolved_edge_and_drag_stays_on_the_surface()
    test_mesh_arrays_are_not_mutated()
    test_rectangular_wing_planform_and_lift()
    test_wing_direct_and_gmres_lift_agree()
    test_swept_constant_chord_planform_area()
    test_open_patch_keeps_centroid_box_area()
    test_cruise_transition_does_not_zero_aft_skin_friction()
    test_tied_transition_stays_attached()
    test_laminar_separation_before_transition_still_freezes()
    test_trailing_edge_cluster_does_not_force_separation()
    test_skin_friction_drag_uses_freestream_dynamic_pressure()
    test_marginal_rebound_does_not_revoke_laminar_separation()
    test_fine_cruise_coupling_converges_without_separation_chatter()
    test_transpiration_changes_circulation()
    test_physics_module_samples()
    print("physics tests passed")
