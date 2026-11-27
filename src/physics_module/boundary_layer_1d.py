"""
1D integral boundary layer.

The default closure is Thwaites laminar, Michel transition, and Head
turbulent. ``drela_two_equation`` keeps that laminar march and transition
test, and replaces the turbulent step with the equilibrium two-equation
shear layer. ``drela_lag`` uses the same two equations and integrates
Drela and Giles' shear-stress lag; it is not an XFOIL solution.

A resolved laminar separation upstream of Michel transition is a free
shear layer, not a frozen profile. Transition there follows Mayle's
short-bubble length, then an attached turbulent restart is attempted.
If that restart fails, or a turbulent station reaches H = 2.8 or
Cf ≤ 0, a separated shear layer is marched and its displacement
thickness still feeds transpiration. The trailing-edge cluster is
still not a separation test. Neither closure marches a wake; the wake
is a separate continuation that starts from the trailing-edge state.

Upper and lower sides march from the stagnation station Module 2 already
built. Laminar θ is a cumulative trapezoid. The turbulent step advances
both sides together; its width is the mesh spacing.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.spatial import cKDTree

from math_module.math_skeleton import IBLStationMesh1D, PanelMetrics2D

from .drela_equilibrium import layer_state, march_interval
from .separated_shear import (
    REATTACHMENT_SHAPE,
    laminar_shear_step,
    mayle_transition_distance,
    separated_entrainment,
    separated_shear_step,
    shape_in_range,
)
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
# The station immediately upstream of transition is the same crossing a
# coarser mesh puts on the transition station. One panel can jump λ by
# about 0.02 while its neighbors stay attached, which is not a separated
# run. A deeper adjacent value, and any hit two or more stations upstream,
# still uses -0.09 unchanged. The 10° separation sits two stations upstream.
_ADJACENT_TIE_DEPTH = 0.025


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


def last_resolved_index(arc: np.ndarray) -> int:
    """Last station that is not in the unresolved trailing-edge cluster.

    The cluster is where a cosine gap is too short to support a slope.
    Callers that need a trailing-edge edge speed use this station. The
    geometric trailing-edge sample keeps falling after the march has
    already frozen θ and H.
    """

    stations = np.asarray(arc, dtype=np.float64)
    if stations.size == 0:
        return 0
    cluster = _trailing_cluster(stations)
    resolved = np.flatnonzero(~cluster)
    if resolved.size == 0:
        return int(stations.size - 1)
    return int(resolved[-1])


def _streamwise_derivative(arc: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Streamwise derivative that does not look into the unresolved trailing edge.

    ``np.gradient`` at the last resolved station is a central difference, so
    it uses the next sample. On a cosine mesh that sample sits in the
    cluster, where the panel speed falls after θ and H have been frozen.
    The fall is not an edge-speed gradient. The derivative is formed on the
    resolved stations only, and the cluster is held at that last value.
    """

    stations = np.asarray(arc, dtype=np.float64)
    samples = np.asarray(values, dtype=np.float64)
    if stations.size < 2:
        return np.zeros(stations.shape, dtype=np.float64)
    cluster = _trailing_cluster(stations)
    resolved = np.flatnonzero(~cluster)
    if resolved.size >= 2:
        slope = np.zeros(stations.shape, dtype=np.float64)
        slope[resolved] = np.gradient(samples[resolved], stations[resolved])
        slope[cluster] = slope[int(resolved[-1])]
        return slope
    return np.gradient(samples, stations)


def _hold_cluster_edge(arc: np.ndarray, edge_speed: np.ndarray) -> np.ndarray:
    """Keep the unresolved trailing-edge speed at the last resolved value.

    The panel solution keeps dropping there. A frozen displacement thickness
    times that drop is a wall suction the boundary layer did not march.
    """

    speed = np.array(np.asarray(edge_speed, dtype=np.float64), copy=True)
    cluster = _trailing_cluster(np.asarray(arc, dtype=np.float64))
    resolved = np.flatnonzero(~cluster)
    if resolved.size and np.any(cluster):
        speed[cluster] = speed[int(resolved[-1])]
    return speed


def _edge_slope(arc: np.ndarray, edge_speed: np.ndarray) -> np.ndarray:
    """Edge-speed slope, with the trailing-edge cluster held at the last resolved value."""

    return _streamwise_derivative(arc, edge_speed)


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Explicit Head march from ``start`` toward the trailing edge.

    Returns updated ``theta``, ``shape``, ``cf``, the first unrecovered
    separation index (or ``-1``), and the first station held because the
    separated shear layer left its validity band (or ``-1``).
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
        # The trailing cluster is not a resolved march. Holding its state at
        # the last resolved station keeps a vanishing panel gap from driving
        # the displacement thickness. A deep separation upstream of the
        # cluster still leaves the attached closure through the test below.
        resolved_attached = False
        if bool(cluster[index + 1]):
            upstream = np.flatnonzero(~cluster[: index + 1])
            anchor = int(upstream[-1]) if upstream.size else index
            incoming = float(shape[anchor])
            resolved_attached = incoming < 2.8 and cf_i > 0.0
            if resolved_attached:
                theta[index + 1] = theta[anchor]
                shape[index + 1] = shape[anchor]
                cf[index + 1] = cf[anchor]
                continue
        if (h_new >= 2.8 or cf_i <= 0.0) and not resolved_attached:
            # The attached step that crossed the threshold is not accepted.
            # A separated shear layer starts from the last attached state.
            onset = index + 1
            theta[onset] = theta[index]
            shape[onset] = shape[index]
            hold_at = _apply_separated(s, ue, theta, shape, cf, onset, cluster)
            return theta, shape, cf, onset, hold_at
    if n > start:
        h_last = float(np.clip(shape[-1], _H_FLOOR, 4.0))
        re_last = max(float(ue[-1]) * float(theta[-1]) / nu, 1e-8)
        cf[-1] = float(np.clip(0.246 * 10.0 ** (-0.678 * h_last) * re_last ** (-0.268), 0.0, 0.04))
    return theta, shape, cf, separated_at, -1


def _drela_from(
    arc: np.ndarray,
    edge_speed: np.ndarray,
    theta: np.ndarray,
    shape: np.ndarray,
    cf: np.ndarray,
    start: int,
    nu: float,
    *,
    lag: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Two-equation turbulent march from ``start``.

    ``lag=False`` is the equilibrium closure. ``lag=True`` integrates
    Drela and Giles' shear-stress lag, seeded at the equilibrium value
    of the first turbulent station. A shape factor that reaches 2.8, or
    a non-positive skin friction, starts the separated shear layer from
    the last attached state. The trailing-edge cluster is copied from
    the last resolved station and is not itself a separation test.

    Returns ``theta``, ``shape``, ``cf``, the unrecovered separation
    index or ``-1``, and the validity-hold index or ``-1``.
    """

    s = np.asarray(arc, dtype=np.float64)
    ue = np.maximum(np.asarray(edge_speed, dtype=np.float64), 0.0)
    cluster = _trailing_cluster(s)
    theta = np.array(theta, dtype=np.float64, copy=True)
    shape = np.array(shape, dtype=np.float64, copy=True)
    cf = np.array(cf, dtype=np.float64, copy=True)
    n = int(s.size)
    c_tau: float | None = None
    for index in range(start, n - 1):
        if lag and c_tau is None:
            seeded = _equilibrium_c_tau(
                float(theta[index]),
                float(shape[index]),
                float(max(ue[index], 1.0e-8)),
                nu,
            )
            c_tau = 1.0e-8 if seeded is None else seeded
        state = layer_state(
            float(theta[index]),
            float(shape[index]),
            float(max(ue[index], 1e-8)),
            0.0,
            nu,
            False,
            sqrt_shear=None if c_tau is None else float(np.sqrt(max(c_tau, 0.0))),
        )
        if state is not None:
            cf[index] = state["cf"]
        if bool(cluster[index + 1]):
            upstream = np.flatnonzero(~cluster[: index + 1])
            anchor = int(upstream[-1]) if upstream.size else index
            incoming = float(shape[anchor])
            if incoming < 2.8 and float(cf[index]) > 0.0:
                theta[index + 1] = theta[anchor]
                shape[index + 1] = shape[anchor]
                cf[index + 1] = cf[anchor]
                continue
        ds = float(s[index + 1] - s[index])
        if lag:
            theta_new, shape_new, cf_new, separated, c_next = march_interval(
                float(theta[index]),
                float(shape[index]),
                float(max(ue[index], 1e-8)),
                float(max(ue[index + 1], 1e-8)),
                ds,
                nu,
                wake=False,
                c_tau=c_tau,
            )
        else:
            theta_new, shape_new, cf_new, separated = march_interval(
                float(theta[index]),
                float(shape[index]),
                float(max(ue[index], 1e-8)),
                float(max(ue[index + 1], 1e-8)),
                ds,
                nu,
                wake=False,
            )
            c_next = None
        if separated or shape_new >= 2.8 or cf_new <= 0.0:
            onset = index + 1
            theta[onset] = theta[index]
            shape[onset] = shape[index]
            hold_at = _apply_separated(s, ue, theta, shape, cf, onset, cluster)
            return theta, shape, cf, onset, hold_at
        theta[index + 1] = theta_new
        shape[index + 1] = shape_new
        cf[index + 1] = cf_new
        c_tau = None if c_next is None else float(c_next)
    return theta, shape, cf, -1, -1


def _equilibrium_c_tau(theta: float, shape_h: float, edge_speed: float, nu: float) -> float | None:
    """Equilibrium Cτ at one station. ``None`` when the closure is not finite."""

    state = layer_state(theta, shape_h, edge_speed, 0.0, nu, False)
    if state is None:
        return None
    return float(state["equilibrium_sqrt_shear"] ** 2)


def _apply_separated(
    arc: np.ndarray,
    edge_speed: np.ndarray,
    theta: np.ndarray,
    shape: np.ndarray,
    cf: np.ndarray,
    start: int,
    cluster: np.ndarray,
) -> int:
    """March a separated turbulent shear layer from ``start``.

    Returns the first station that was held because the step left the
    validity band, or ``-1`` when every resolved step stayed inside it.
    Stations in the trailing-edge cluster copy the last resolved state.
    """

    s = np.asarray(arc, dtype=np.float64)
    ue = np.asarray(edge_speed, dtype=np.float64)
    entrainment = separated_entrainment()
    n = int(s.size)
    if start < 0 or start >= n:
        return -1
    cf[start:] = 0.0
    hold_at = -1
    if float(theta[start]) <= 0.0 or not shape_in_range(float(shape[start])):
        hold_at = start
    for index in range(start, n - 1):
        if bool(cluster[index + 1]) or hold_at >= 0:
            theta[index + 1] = theta[index]
            shape[index + 1] = shape[index]
            cf[index + 1] = 0.0
            continue
        ds = float(s[index + 1] - s[index])
        stepped = separated_shear_step(
            float(theta[index]),
            float(shape[index]),
            float(max(ue[index], 1.0e-8)),
            float(max(ue[index + 1], 1.0e-8)),
            ds,
            entrainment,
        )
        if stepped is None:
            hold_at = index + 1
            theta[index + 1] = theta[index]
            shape[index + 1] = shape[index]
            cf[index + 1] = 0.0
            continue
        theta[index + 1] = stepped[0]
        shape[index + 1] = stepped[1]
        cf[index + 1] = 0.0
    return hold_at


def _march_laminar_shear(
    arc: np.ndarray,
    edge_speed: np.ndarray,
    theta: np.ndarray,
    shape: np.ndarray,
    cf: np.ndarray,
    start: int,
    stop: int,
    nu: float,
    cluster: np.ndarray,
) -> int:
    """Laminar free shear on ``[start, stop)``. Returns a hold index or ``-1``."""

    s = np.asarray(arc, dtype=np.float64)
    ue = np.asarray(edge_speed, dtype=np.float64)
    n = int(s.size)
    stop = min(int(stop), n)
    if start < 0 or start >= stop:
        return -1
    cf[start:stop] = 0.0
    hold_at = -1
    for index in range(start, stop - 1):
        if bool(cluster[index + 1]) or hold_at >= 0:
            theta[index + 1] = theta[index]
            shape[index + 1] = shape[index]
            cf[index + 1] = 0.0
            continue
        stepped = laminar_shear_step(
            float(theta[index]),
            float(shape[index]),
            float(max(ue[index], 1.0e-8)),
            float(max(ue[index + 1], 1.0e-8)),
            float(s[index + 1] - s[index]),
            nu,
        )
        if stepped is None:
            hold_at = index + 1
            theta[index + 1] = theta[index]
            shape[index + 1] = shape[index]
            cf[index + 1] = 0.0
            continue
        theta[index + 1] = stepped[0]
        shape[index + 1] = stepped[1]
        cf[index + 1] = 0.0
    return hold_at


def _side_state(
    arc: np.ndarray,
    edge_speed: np.ndarray,
    nu: float,
    retained_index: Optional[int] = None,
    closure: IBLClosure = "thwaites_head",
) -> dict[str, np.ndarray | float | bool]:
    edge_speed = _hold_cluster_edge(arc, _march_speed(edge_speed))
    theta, shape, cf, lam, _delta_star = _thwaites_side(arc, edge_speed, nu)
    s = np.asarray(arc, dtype=np.float64)
    laminar_hits = np.flatnonzero(lam <= _LAMINAR_SEPARATION_LAMBDA)
    laminar_sep = int(laminar_hits[0]) if laminar_hits.size else -1
    transition = _transition_index(s, edge_speed, theta, nu)
    # One index before transition is the same two-station crossing that a
    # coarser mesh lands on the transition index. A single-panel overshoot
    # there starts Head. A value deeper than the tie depth still opens the bubble,
    # as does any hit two or more stations upstream (the 10° case).
    if (
        transition is not None
        and laminar_sep + 1 == transition
        and float(lam[laminar_sep]) > _LAMINAR_SEPARATION_LAMBDA - _ADJACENT_TIE_DEPTH
    ):
        laminar_sep = transition
    if retained_index is not None and 0 <= int(retained_index) < int(lam.size):
        retained = int(retained_index)
        if float(lam[retained]) <= _LAMINAR_RELEASE_LAMBDA:
            moved_upstream = laminar_sep >= 0 and laminar_sep < retained and (
                transition is None or laminar_sep < transition
            )
            adjacent_tie = (
                transition is not None
                and retained + 1 == int(transition)
                and float(lam[retained]) > _LAMINAR_SEPARATION_LAMBDA - _ADJACENT_TIE_DEPTH
            )
            # A one-station graze in front of transition is transition.
            # Retention must not turn that graze back into a laminar bubble.
            if not moved_upstream and not adjacent_tie:
                laminar_sep = retained
    cluster = _trailing_cluster(s)
    # A λ hit that exists only inside the unresolved trailing-edge cluster
    # is the same non-separation as a turbulent crossing there.
    if laminar_sep >= 0 and bool(cluster[laminar_sep]):
        laminar_sep = -1
    turbulent = np.zeros(s.shape, dtype=bool)
    separated_at = -1
    bubble_at = -1
    reattach_at = -1
    hold_at = -1

    # A tie is transition. Thwaites λ and Michel's criterion often cross
    # between the same two stations, so both first light on one index.
    # Laminar separation wins only when it is strictly upstream of Michel.
    # That case is a free shear layer until Mayle transition, then an
    # attached turbulent restart. It is not a frozen profile.
    # A station retained from the previous coupling pass stays the bubble
    # onset until λ there recovers past the release level.
    if laminar_sep >= 0 and (transition is None or laminar_sep < transition):
        bubble_at = laminar_sep
        distance = mayle_transition_distance(
            float(theta[laminar_sep]),
            float(max(edge_speed[laminar_sep], 1.0e-8)),
            nu,
        )
        shear_transition: int | None = None
        if np.isfinite(distance):
            s_tr = float(s[laminar_sep]) + float(distance)
            candidates = np.flatnonzero((s >= s_tr) & (~cluster))
            candidates = candidates[candidates > laminar_sep]
            if candidates.size:
                shear_transition = int(candidates[0])
        stop = int(s.size) if shear_transition is None else shear_transition
        hold_at = _march_laminar_shear(
            s, edge_speed, theta, shape, cf, laminar_sep, stop, nu, cluster
        )
        if shear_transition is None or hold_at >= 0:
            separated_at = laminar_sep
            transition = None
        else:
            transition = shear_transition
            saved_theta = np.array(theta, copy=True)
            saved_shape = np.array(shape, copy=True)
            saved_cf = np.array(cf, copy=True)
            if float(shape[transition]) > REATTACHMENT_SHAPE:
                shape[transition] = REATTACHMENT_SHAPE
            if closure == "thwaites_head":
                theta, shape, cf, turbulent_sep, turb_hold = _head_from(
                    s, edge_speed, theta, shape, cf, transition, nu
                )
            else:
                theta, shape, cf, turbulent_sep, turb_hold = _drela_from(
                    s,
                    edge_speed,
                    theta,
                    shape,
                    cf,
                    transition,
                    nu,
                    lag=closure == "drela_lag",
                )
            immediate = turbulent_sep >= 0 and turbulent_sep <= transition + 1
            if immediate:
                theta = saved_theta
                shape = saved_shape
                cf = saved_cf
                shear_hold = _apply_separated(s, edge_speed, theta, shape, cf, transition, cluster)
                separated_at = transition
                turbulent[transition:] = True
                if shear_hold >= 0:
                    hold_at = shear_hold
            else:
                reattach_at = transition
                turbulent[transition:] = True
                if turbulent_sep >= 0:
                    separated_at = turbulent_sep
                    turbulent[turbulent_sep:] = True
                if turb_hold >= 0:
                    hold_at = turb_hold
    elif transition is not None:
        turbulent[transition:] = True
        if closure == "thwaites_head":
            theta, shape, cf, turbulent_sep, hold_at = _head_from(
                s, edge_speed, theta, shape, cf, transition, nu
            )
        else:
            theta, shape, cf, turbulent_sep, hold_at = _drela_from(
                s,
                edge_speed,
                theta,
                shape,
                cf,
                transition,
                nu,
                lag=closure == "drela_lag",
            )
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
        "bubble_s": float("nan") if bubble_at < 0 else float(s[bubble_at]),
        "reattachment_s": float("nan") if reattach_at < 0 else float(s[reattach_at]),
        "hold_at": int(hold_at),
        "separated_mask": separated_at >= 0,
        "separated_at": separated_at,
    }


def laminar_drag_is_unsupported(ibl: Optional[object]) -> bool:
    """True when laminar separation was not followed by transition.

    The laminar free shear layer sets wall shear to zero. That surface
    integral is not a validated separated-flow drag, so L/D is withheld.
    A reattached bubble has no unrecovered ``separation_s`` and is not
    this case. A turbulent separation at or behind transition is not either.
    """

    if ibl is None:
        return False
    payload = ibl  # type: ignore[index]
    for side in ("upper", "lower"):
        separation = float(payload[f"separation_s_{side}"])
        transition = float(payload[f"transition_s_{side}"])
        if np.isfinite(separation) and (not np.isfinite(transition) or separation < transition):
            return True
    return False


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
    *,
    freeze_upper: int = -1,
    freeze_lower: int = -1,
) -> np.ndarray:
    """Scatter ``d(Ue δ*)/ds`` onto the nearest surface panels.

    A marched separated shear layer keeps its displacement gradient: that
    is the coupling. ``freeze_*`` is only the station where the shear-layer
    model left its validity band and the state was held. Differentiating
    that plateau against a changing edge speed invents a wall-normal
    velocity the march does not contain.
    """

    def side_velocity(
        arc: np.ndarray,
        xy: np.ndarray,
        ue: np.ndarray,
        delta_star: np.ndarray,
        freeze: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        flux = np.asarray(ue, dtype=np.float64) * np.asarray(delta_star, dtype=np.float64)
        station_s = np.asarray(arc, dtype=np.float64)
        if station_s.size < 2:
            return xy, np.zeros(station_s.shape, dtype=np.float64)
        # Same unresolved trailing-edge suffix as the edge-speed slope.
        # Dividing a smooth flux change by a cosine gap of 1e-4 chord
        # otherwise moves δ* by several percent of chord each iteration.
        gradient = _streamwise_derivative(station_s, flux)
        if freeze >= 0:
            gradient = np.array(gradient, copy=True)
            gradient[int(freeze) :] = 0.0
        return xy, gradient

    xy_u, vn_u = side_velocity(
        ibl["s_upper"], ibl["xy_upper"], ue_upper, delta_star_upper, freeze_upper
    )
    xy_l, vn_l = side_velocity(
        ibl["s_lower"], ibl["xy_lower"], ue_lower, delta_star_lower, freeze_lower
    )
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
    # hundredths of V∞. A larger spike is an unresolved streamwise gradient
    # and it overturns the panel solution if it is applied raw.
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

    if closure not in ("thwaites_head", "drela_two_equation", "drela_lag"):
        raise InvalidFlowConditionError(f"Unknown ibl_closure {closure!r}.")
    if not np.isfinite(reynolds) or reynolds <= 0.0:
        raise InvalidFlowConditionError("Reynolds number must be positive for a viscous march.")
    if not np.isfinite(chord) or chord <= 0.0:
        raise InvalidFlowConditionError("Chord must be positive for a viscous march.")

    nu = float(chord) / float(reynolds)
    upper = _side_state(
        ibl["s_upper"], ue_upper, nu, retained_index=retained_upper, closure=closure
    )
    lower = _side_state(
        ibl["s_lower"], ue_lower, nu, retained_index=retained_lower, closure=closure
    )
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
        model_limited=bool(int(upper["hold_at"]) >= 0 or int(lower["hold_at"]) >= 0),
        delta_upper=upper["delta"],  # type: ignore[arg-type]
        delta_lower=lower["delta"],  # type: ignore[arg-type]
        separation_s_upper=float(upper["separation_s"]),  # type: ignore[arg-type]
        separation_s_lower=float(lower["separation_s"]),  # type: ignore[arg-type]
        bubble_s_upper=float(upper["bubble_s"]),  # type: ignore[arg-type]
        bubble_s_lower=float(lower["bubble_s"]),  # type: ignore[arg-type]
        reattachment_s_upper=float(upper["reattachment_s"]),  # type: ignore[arg-type]
        reattachment_s_lower=float(lower["reattachment_s"]),  # type: ignore[arg-type]
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
        freeze_upper=int(upper["hold_at"]),
        freeze_lower=int(lower["hold_at"]),
    )
    return solution, wall
