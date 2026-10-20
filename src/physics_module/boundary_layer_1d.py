"""
1D integral boundary layer: Thwaites laminar, Michel transition, Head turbulent.

Upper and lower sides march from the stagnation station Module 2 already
built. Laminar θ is a cumulative trapezoid. The turbulent step is explicit
and advances both sides together; its width is the mesh spacing.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.spatial import cKDTree

from math_module.math_skeleton import IBLStationMesh1D, PanelMetrics2D

from .physics_skeleton import (
    BoundaryLayerSeparationError,
    IBLClosure,
    IBLSolution,
    InvalidFlowConditionError,
)

_H_FLOOR = 1.15
_H1_FLOOR = 3.05
_LAMINAR_SEPARATION_LAMBDA = -0.09
# A station that already separated stays separated until λ recovers past
# this level. On the 160-panel cruise mesh the coupling itself only
# rebounds from about -0.091 to about -0.089, so a strict -0.09 test
# revokes separation, rebuilds the aft displacement thickness, and crosses
# the threshold again. Release is still below the attached 64-panel value
# near -0.079, so a real recovery is not held.
_LAMINAR_RELEASE_LAMBDA = -0.08


def _march_speed(edge_speed: np.ndarray) -> np.ndarray:
    """Edge speed for a march that starts at the geometric leading edge.

    At angle of attack the stagnation point sits on the lower surface, so the
    geometric leading-edge station is upstream of it and the raw speed falls.
    Thwaites treats that drop as immediate separation. Flatten the speed back
    to the stagnation value; the boundary layer then accelerates away from it.
    """

    speed = np.maximum(np.asarray(edge_speed, dtype=np.float64), 0.0)
    if speed.size < 4:
        return speed
    window = max(3, int(round(0.30 * speed.size)))
    stagnation = int(np.argmin(speed[:window]))
    if stagnation <= 0:
        return speed
    flattened = np.array(speed, dtype=np.float64, copy=True)
    flattened[:stagnation] = flattened[stagnation]
    return flattened


def _trailing_cluster(arc: np.ndarray) -> np.ndarray:
    """Stations whose gap to the trailing edge is below a quarter of the median spacing.

    Cosine clustering makes the last intervals a few thousandths of the chord.
    Dividing a small edge-speed change by that gap produces a slope of tens,
    and one explicit step then drives the shape factor through the separation
    threshold. The leading edge is clustered too, but this mask is only the
    unresolved suffix: it stops at the first gap that is at least the limit.
    """

    mask = np.zeros(arc.shape[0], dtype=bool)
    if arc.size < 4:
        return mask
    gaps = np.diff(arc)
    positive = gaps[gaps > 0.0]
    if positive.size == 0:
        return mask
    short = gaps < 0.25 * float(np.median(positive))
    resolved = np.flatnonzero(~short)
    if resolved.size == 0:
        return mask
    mask[int(resolved[-1]) + 1 :] = True
    return mask


def _edge_slope(arc: np.ndarray, edge_speed: np.ndarray) -> np.ndarray:
    """Edge-speed slope, with the trailing-edge cluster held at the last resolved value."""

    if arc.size < 2:
        return np.zeros(arc.shape, dtype=np.float64)
    slope = np.gradient(np.asarray(edge_speed, dtype=np.float64), arc)
    cluster = _trailing_cluster(arc)
    resolved = np.flatnonzero(~cluster)
    if np.any(cluster) and resolved.size:
        slope = np.array(slope, copy=True)
        slope[cluster] = slope[int(resolved[-1])]
    return slope


def _thwaites_side(
    arc: np.ndarray,
    edge_speed: np.ndarray,
    nu: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized Thwaites solution on one side.

    Returns ``theta``, ``shape_h``, ``cf``, ``lam``, ``delta_star``.
    """

    s = np.asarray(arc, dtype=np.float64)
    ue = np.maximum(np.asarray(edge_speed, dtype=np.float64), 0.0)
    integrand = ue ** 5
    integral = np.zeros(s.shape, dtype=np.float64)
    if s.size > 1:
        integral[1:] = cumulative_trapezoid(integrand, s)
    due = _edge_slope(s, ue)
    ue_safe = np.maximum(ue, 1e-8)
    theta2 = 0.45 * nu * integral / ue_safe ** 6
    stagnating = (ue < 1e-4) & (due > 0.0)
    theta2 = np.where(stagnating, 0.075 * nu / np.maximum(due, 1e-8), theta2)
    theta = np.sqrt(np.maximum(theta2, 0.0))
    lam = theta ** 2 / nu * due
    lam_c = np.clip(lam, -0.09, 0.25)
    shape = 2.61 - 3.75 * lam_c + 5.24 * lam_c ** 2
    ell = 0.22 + 1.57 * lam_c - 1.8 * lam_c ** 2
    re_theta = ue_safe * theta / nu
    # θ vanishes at the stagnation station, so Re_θ hits the floor and the
    # uncapped Cf is non-physical. Local Cf above a few percent does not
    # occur at the Reynolds numbers this march is built for.
    cf = np.clip(2.0 * ell / np.maximum(re_theta, 1.0), 0.0, 0.04)
    delta_star = shape * theta
    return theta, shape, cf, lam, delta_star


def _transition_index(arc: np.ndarray, ue: np.ndarray, theta: np.ndarray, nu: float) -> Optional[int]:
    s = np.asarray(arc, dtype=np.float64)
    speed = np.maximum(np.asarray(ue, dtype=np.float64), 0.0)
    re_s = speed * s / nu
    re_theta = speed * np.asarray(theta, dtype=np.float64) / nu
    valid = re_s > 1.0
    threshold = np.full(s.shape, np.inf, dtype=np.float64)
    threshold[valid] = 1.174 * (1.0 + 22400.0 / re_s[valid]) * np.power(re_s[valid], 0.46)
    hits = np.flatnonzero((re_theta >= threshold) & valid)
    if hits.size == 0:
        return None
    return int(hits[0])


def _invert_h1(h1: float) -> float:
    """Inverse of Head's H1(H) correlation."""

    if h1 <= 3.3:
        return 2.8
    inner = 0.8234 / (h1 - 3.3)
    return float(1.1 + inner ** (1.0 / 1.287))


def _head_from(
    arc: np.ndarray,
    edge_speed: np.ndarray,
    theta: np.ndarray,
    shape: np.ndarray,
    cf: np.ndarray,
    start: int,
    nu: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Explicit Head march from ``start`` toward the trailing edge.

    Returns updated ``theta``, ``shape``, ``cf`` and the first separated
    station index, or ``-1`` when the side stays attached.
    """

    s = np.asarray(arc, dtype=np.float64)
    ue = np.maximum(np.asarray(edge_speed, dtype=np.float64), 0.0)
    due = _edge_slope(s, ue)
    cluster = _trailing_cluster(s)
    theta = np.array(theta, dtype=np.float64, copy=True)
    shape = np.array(shape, dtype=np.float64, copy=True)
    cf = np.array(cf, dtype=np.float64, copy=True)
    n = int(s.size)
    separated_at = -1
    for index in range(start, n - 1):
        h_i = float(np.clip(shape[index], _H_FLOOR, 4.0))
        speed = float(max(ue[index], 1e-8))
        re_theta = max(speed * float(theta[index]) / nu, 1e-8)
        cf_i = float(np.clip(0.246 * 10.0 ** (-0.678 * h_i) * re_theta ** (-0.268), 0.0, 0.04))
        ds = float(s[index + 1] - s[index])
        dtheta = (cf_i / 2.0 - (h_i + 2.0) * float(theta[index]) / speed * float(due[index])) * ds
        theta_new = max(float(theta[index]) + dtheta, 1e-12)
        h1 = 3.3 + 0.8234 * (h_i - 1.1) ** (-1.287)
        h1 = max(h1, _H1_FLOOR)
        entrainment = 0.0306 * (h1 - 3.0) ** (-0.6169)
        momentum_flux = speed * float(theta[index]) * h1 + ds * speed * entrainment
        speed_next = float(max(ue[index + 1], 1e-8))
        h1_next = max(momentum_flux / (speed_next * theta_new), _H1_FLOOR)
        h_new = _invert_h1(h1_next)
        cf[index] = cf_i
        theta[index + 1] = theta_new
        shape[index + 1] = h_new
        # A shape factor that crosses 2.8 only inside the trailing cluster,
        # while the last resolved station is still attached, is the
        # clustered edge-speed drop rather than a resolved separation.
        resolved_attached = False
        if bool(cluster[index + 1]):
            upstream = np.flatnonzero(~cluster[: index + 1])
            incoming = float(shape[int(upstream[-1])]) if upstream.size else h_i
            resolved_attached = incoming < 2.8 and cf_i > 0.0
        if (h_new >= 2.8 or cf_i <= 0.0) and not resolved_attached:
            separated_at = index + 1
            theta[index + 1 :] = theta_new
            shape[index + 1 :] = h_new
            cf[index + 1 :] = min(cf_i, 0.0) if cf_i <= 0.0 else cf_i
            cf[index + 1 :] = 0.0
            break
    if n > start:
        h_last = float(np.clip(shape[-1], _H_FLOOR, 4.0))
        re_last = max(float(ue[-1]) * float(theta[-1]) / nu, 1e-8)
        cf[-1] = float(np.clip(0.246 * 10.0 ** (-0.678 * h_last) * re_last ** (-0.268), 0.0, 0.04))
    return theta, shape, cf, separated_at


def _freeze(values: np.ndarray, index: int) -> np.ndarray:
    frozen = np.array(values, dtype=np.float64, copy=True)
    if 0 <= index < frozen.size:
        frozen[index:] = frozen[index]
    return frozen


def _side_state(
    arc: np.ndarray,
    edge_speed: np.ndarray,
    nu: float,
    retained_index: Optional[int] = None,
) -> dict[str, np.ndarray | float | bool]:
    edge_speed = _march_speed(edge_speed)
    theta, shape, cf, lam, _delta_star = _thwaites_side(arc, edge_speed, nu)
    s = np.asarray(arc, dtype=np.float64)
    laminar_hits = np.flatnonzero(lam <= _LAMINAR_SEPARATION_LAMBDA)
    laminar_sep = int(laminar_hits[0]) if laminar_hits.size else -1
    transition = _transition_index(s, edge_speed, theta, nu)
    if retained_index is not None and 0 <= int(retained_index) < int(lam.size):
        retained = int(retained_index)
        if float(lam[retained]) <= _LAMINAR_RELEASE_LAMBDA:
            moved_upstream = laminar_sep >= 0 and laminar_sep < retained and (
                transition is None or laminar_sep < transition
            )
            if not moved_upstream:
                laminar_sep = retained
    turbulent = np.zeros(s.shape, dtype=bool)
    separated_at = -1

    # A tie is transition. Thwaites λ and Michel's criterion often cross
    # between the same two stations, so both first light on one index.
    # Laminar separation wins only when it is strictly upstream and the
    # turbulent closure was never reached. Starting Head at the tied
    # station lets the turbulent layer carry the adverse gradient.
    # A station retained from the previous coupling pass stays separated
    # until λ there recovers past the release level, so a rebound of about
    # one thousandth does not wipe and restore the aft layer.
    if laminar_sep >= 0 and (transition is None or laminar_sep < transition):
        theta = _freeze(theta, laminar_sep)
        shape = _freeze(shape, laminar_sep)
        cf = _freeze(cf, laminar_sep)
        cf[laminar_sep:] = 0.0
        separated_at = laminar_sep
    elif transition is not None:
        turbulent[transition:] = True
        theta, shape, cf, turbulent_sep = _head_from(s, edge_speed, theta, shape, cf, transition, nu)
        if turbulent_sep >= 0:
            separated_at = turbulent_sep
            turbulent[turbulent_sep:] = True

    delta_star = shape * theta
    ratio = np.maximum(3.0 - lam, 0.2) / 10.0
    delta = delta_star / ratio
    delta = np.where(turbulent, (72.0 / 7.0) * theta, delta)
    return {
        "theta": np.ascontiguousarray(theta, dtype=np.float64),
        "shape": np.ascontiguousarray(shape, dtype=np.float64),
        "cf": np.ascontiguousarray(cf, dtype=np.float64),
        "delta_star": np.ascontiguousarray(delta_star, dtype=np.float64),
        "delta": np.ascontiguousarray(delta, dtype=np.float64),
        "ue": np.ascontiguousarray(np.asarray(edge_speed, dtype=np.float64), dtype=np.float64),
        "transition_s": float("nan") if transition is None else float(s[transition]),
        "separation_s": float("nan") if separated_at < 0 else float(s[separated_at]),
        "separated_mask": separated_at >= 0,
        "separated_at": separated_at,
    }


def _mask_from(size: int, index: int) -> np.ndarray:
    mask = np.zeros(size, dtype=bool)
    if index >= 0:
        mask[index:] = True
    return mask


def transpiration_on_panels(
    panel: PanelMetrics2D,
    ibl: IBLStationMesh1D,
    ue_upper: np.ndarray,
    delta_star_upper: np.ndarray,
    ue_lower: np.ndarray,
    delta_star_lower: np.ndarray,
) -> np.ndarray:
    """Scatter ``d(Ue δ*)/ds`` onto the nearest surface panels."""

    def side_velocity(arc: np.ndarray, xy: np.ndarray, ue: np.ndarray, delta_star: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        flux = np.asarray(ue, dtype=np.float64) * np.asarray(delta_star, dtype=np.float64)
        station_s = np.asarray(arc, dtype=np.float64)
        if station_s.size < 2:
            return xy, np.zeros(station_s.shape, dtype=np.float64)
        return xy, np.gradient(flux, station_s)

    xy_u, vn_u = side_velocity(ibl["s_upper"], ibl["xy_upper"], ue_upper, delta_star_upper)
    xy_l, vn_l = side_velocity(ibl["s_lower"], ibl["xy_lower"], ue_lower, delta_star_lower)
    xy = np.vstack((xy_u, xy_l))
    vn = np.concatenate((vn_u, vn_l))
    collocation = np.asarray(panel["collocation"], dtype=np.float64)
    _distance, nearest = cKDTree(collocation).query(xy)
    accumulated = np.zeros(collocation.shape[0], dtype=np.float64)
    weight = np.zeros(collocation.shape[0], dtype=np.float64)
    np.add.at(accumulated, nearest, vn)
    np.add.at(weight, nearest, 1.0)
    out = np.zeros(collocation.shape[0], dtype=np.float64)
    live = weight > 0.0
    out[live] = accumulated[live] / weight[live]
    # A healthy boundary layer at these Reynolds numbers blows at a few
    # hundredths of V∞. Larger spikes are a separated-station gradient and
    # overturn the source sheet if they are applied raw.
    return np.clip(out, -0.05, 0.05)


def march_ibl(
    ibl: IBLStationMesh1D,
    ue_upper: np.ndarray,
    ue_lower: np.ndarray,
    reynolds: float,
    chord: float,
    panel: PanelMetrics2D,
    *,
    closure: IBLClosure = "thwaites_head",
    fail_on_separation: bool = False,
    retained_upper: Optional[int] = None,
    retained_lower: Optional[int] = None,
) -> tuple[IBLSolution, np.ndarray]:
    """March both sides and return the station solution plus panel transpiration."""

    if closure == "drela_two_equation":
        raise InvalidFlowConditionError("ibl_closure 'drela_two_equation' is not implemented")
    if closure != "thwaites_head":
        raise InvalidFlowConditionError(f"Unknown ibl_closure {closure!r}.")
    if not np.isfinite(reynolds) or reynolds <= 0.0:
        raise InvalidFlowConditionError("Reynolds number must be positive for a viscous march.")
    if not np.isfinite(chord) or chord <= 0.0:
        raise InvalidFlowConditionError("Chord must be positive for a viscous march.")

    nu = float(chord) / float(reynolds)
    upper = _side_state(ibl["s_upper"], ue_upper, nu, retained_index=retained_upper)
    lower = _side_state(ibl["s_lower"], ue_lower, nu, retained_index=retained_lower)
    separated = bool(upper["separated_mask"] or lower["separated_mask"])
    if separated and fail_on_separation:
        side = "upper" if upper["separated_mask"] else "lower"
        station = upper["separation_s"] if upper["separated_mask"] else lower["separation_s"]
        raise BoundaryLayerSeparationError(
            f"Boundary layer separated on the {side} surface at s={float(station):.6g}."
        )

    n_upper = int(np.asarray(ibl["s_upper"]).shape[0])
    n_lower = int(np.asarray(ibl["s_lower"]).shape[0])
    solution = IBLSolution(
        theta_upper=upper["theta"],  # type: ignore[arg-type]
        theta_lower=lower["theta"],  # type: ignore[arg-type]
        delta_star_upper=upper["delta_star"],  # type: ignore[arg-type]
        delta_star_lower=lower["delta_star"],  # type: ignore[arg-type]
        shape_h_upper=upper["shape"],  # type: ignore[arg-type]
        shape_h_lower=lower["shape"],  # type: ignore[arg-type]
        cf_upper=upper["cf"],  # type: ignore[arg-type]
        cf_lower=lower["cf"],  # type: ignore[arg-type]
        ue_upper=upper["ue"],  # type: ignore[arg-type]
        ue_lower=lower["ue"],  # type: ignore[arg-type]
        transition_s_upper=float(upper["transition_s"]),  # type: ignore[arg-type]
        transition_s_lower=float(lower["transition_s"]),  # type: ignore[arg-type]
        separated=separated,
        delta_upper=upper["delta"],  # type: ignore[arg-type]
        delta_lower=lower["delta"],  # type: ignore[arg-type]
        separation_s_upper=float(upper["separation_s"]),  # type: ignore[arg-type]
        separation_s_lower=float(lower["separation_s"]),  # type: ignore[arg-type]
        separated_upper=_mask_from(n_upper, int(upper["separated_at"])),  # type: ignore[arg-type]
        separated_lower=_mask_from(n_lower, int(lower["separated_at"])),  # type: ignore[arg-type]
        stagnation_xy=np.array(ibl["stagnation_xy"], dtype=np.float64, copy=True),
    )
    wall = transpiration_on_panels(
        panel,
        ibl,
        solution["ue_upper"],
        solution["delta_star_upper"],
        solution["ue_lower"],
        solution["delta_star_lower"],
    )
    return solution, wall
