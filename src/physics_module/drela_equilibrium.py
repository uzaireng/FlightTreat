"""Incompressible equilibrium form of Drela's two-equation shear layer.

The unknowns on a station are momentum thickness and the kinematic shape
factor. They obey the von Kármán momentum integral and the kinetic-energy
shape-parameter equation. Skin friction, dissipation, and the kinetic-energy
shape factor are Drela's incompressible closures (XFOIL 6.99: ``HST``,
``CFT``, ``CFL``, ``DIL``, ``DILW``, and the ``BLVAR`` dissipation split).
Compressibility is absent: the panel solver's edge speed is already the
speed passed to the boundary layer, and the kinematic shape factor is the
shape factor.

What this file does not contain is the rest of XFOIL. There is no shear-stress
lag equation, so the turbulent shear coefficient is the local equilibrium
value rather than a lagged history. There is no envelope e^n transition and
no simultaneous Newton coupling of the boundary layer to the panel system.
A direct step that cannot stay attached stops. It does not march through a
Goldstein singularity.

Wake stations use the same equations with zero skin friction. Dissipation is
doubled because the correlations are for one shear layer and a merged wake
has two.
"""

from __future__ import annotations

import numpy as np

# XFOIL BLPAR defaults. CTCON is 0.5 / (GACON^2 * GBCON), with GACON = 6.70.
_GA = 6.70
_GB = 0.75
_GC = 18.0
_CT = 0.5 / (_GA * _GA * _GB)
_HS_MIN = 1.5
_DHS_INF = 0.015
_LOG10 = np.log(10.0)


def kinetic_shape_factor(shape_h: float, reynolds_theta: float) -> float:
    """Turbulent H*(H, Re_θ), XFOIL ``HST`` at zero Mach."""

    hk = float(shape_h)
    rt = float(max(reynolds_theta, 1.0))
    if hk <= 1.0 or not np.isfinite(hk) or not np.isfinite(rt):
        return float("nan")
    ho = 3.0 + 400.0 / rt if rt > 400.0 else 4.0
    rtz = rt if rt > 200.0 else 200.0
    if hk < ho:
        hr = (ho - hk) / (ho - 1.0)
        return float(
            (2.0 - _HS_MIN - 4.0 / rtz) * hr * hr * 1.5 / (hk + 0.5)
            + _HS_MIN
            + 4.0 / rtz
        )
    grt = float(np.log(rtz))
    hdif = hk - ho
    rtmp = max(hk - ho + 4.0 / grt, 1e-6)
    htmp = 0.007 * grt / (rtmp * rtmp) + _DHS_INF / hk
    return float(hdif * hdif * htmp + _HS_MIN + 4.0 / rtz)


def _laminar_kinetic_shape(shape_h: float) -> float:
    """Laminar H*(H), XFOIL ``HSL``. Used only by the laminar wake floor."""

    hk = float(shape_h)
    if hk < 4.35:
        tmp = hk - 4.35
        return float(
            0.0111 * tmp * tmp / (hk + 1.0)
            - 0.0278 * tmp ** 3 / (hk + 1.0)
            + 1.528
            - 0.0002 * (tmp * hk) ** 2
        )
    return float(0.015 * (hk - 4.35) ** 2 / hk + 1.528)


def laminar_skin_friction(shape_h: float, reynolds_theta: float) -> float:
    """Laminar Cf(H, Re_θ), XFOIL ``CFL``."""

    hk = float(shape_h)
    rt = float(max(reynolds_theta, 1.0))
    if hk < 5.5:
        tmp = (5.5 - hk) ** 3 / (hk + 1.0)
        return float((0.0727 * tmp - 0.07) / rt)
    if hk <= 4.5:
        return float("nan")
    tmp = 1.0 - 1.0 / (hk - 4.5)
    return float((0.015 * tmp * tmp - 0.07) / rt)


def turbulent_skin_friction(shape_h: float, reynolds_theta: float) -> float:
    """Turbulent Cf(H, Re_θ), XFOIL ``CFT`` at zero Mach, CFFAC = 1.

    The laminar value replaces it only when the laminar correlation is
    larger, which is XFOIL's guard for an unreasonably small Re_θ.
    """

    hk = float(shape_h)
    rt = float(max(reynolds_theta, 1.0))
    grt = max(float(np.log(rt)), 3.0)
    gex = -1.74 - 0.31 * hk
    arg = max(-1.33 * hk, -20.0)
    thk = float(np.tanh(4.0 - hk / 0.875))
    cfo = 0.3 * float(np.exp(arg)) * (grt / _LOG10) ** gex
    cf = cfo + 1.1e-4 * (thk - 1.0)
    laminar = laminar_skin_friction(hk, rt)
    if np.isfinite(laminar) and laminar > cf:
        return float(laminar)
    return float(cf)


def _laminar_dissipation(shape_h: float, reynolds_theta: float) -> float:
    """Laminar 2 CD / H*, XFOIL ``DIL``."""

    hk = float(shape_h)
    rt = float(max(reynolds_theta, 1.0))
    if hk < 4.0:
        return float((0.00205 * (4.0 - hk) ** 5.5 + 0.207) / rt)
    hkb = hk - 4.0
    den = 1.0 + 0.02 * hkb * hkb
    return float((-0.0016 * hkb * hkb / den + 0.207) / rt)


def _laminar_wake_dissipation(shape_h: float, reynolds_theta: float) -> float:
    """Laminar wake 2 CD / H*, XFOIL ``DILW``."""

    hk = float(max(shape_h, 1.01))
    rt = float(max(reynolds_theta, 1.0))
    hs = _laminar_kinetic_shape(hk)
    if not np.isfinite(hs) or hs <= 0.0:
        return float("nan")
    rcd = 1.10 * (1.0 - 1.0 / hk) ** 2 / hk
    return float(2.0 * rcd / (hs * rt))


def _slip_velocity(kinetic_shape: float, shape_h: float, wake: bool) -> float:
    """Normalized slip velocity. ``GBCON`` is 0.75, as in XFOIL ``BLVAR``."""

    hs = float(kinetic_shape)
    hk = float(shape_h)
    us = 0.5 * hs * (1.0 - (hk - 1.0) / (_GB * hk))
    ceiling = 0.99995 if wake else 0.98
    return float(np.clip(us, 1.0e-4, ceiling))


def _equilibrium_sqrt_shear(
    kinetic_shape: float,
    shape_h: float,
    reynolds_theta: float,
    slip: float,
    wake: bool,
) -> float:
    """Equilibrium √Cτ. The lag equation would relax toward this value."""

    hk = float(shape_h)
    hs = float(kinetic_shape)
    rt = float(max(reynolds_theta, 1.0))
    hkb = max(hk - 1.0, 1.0e-6)
    if wake:
        hkc = max(hk - 1.0, 1.0e-4)
    else:
        hkc = max(hk - 1.0 - _GC / rt, 0.01)
    usb = max(1.0 - float(slip), 1.0e-4)
    argument = _CT * hs * hkb * hkc * hkc / (usb * hk * hk * hk)
    if not np.isfinite(argument) or argument <= 0.0:
        return 0.0
    return float(np.sqrt(argument))


def dissipation_coefficient(
    shape_h: float,
    reynolds_theta: float,
    kinetic_shape: float,
    slip: float,
    skin_friction: float,
    sqrt_shear: float,
    wake: bool,
) -> float:
    """Return 2 CD / H* for a wall layer or a merged wake.

    The wall piece is the inner shear. The outer piece is the equilibrium
    shear stress plus the small laminar-stress contribution in ``BLVAR``.
    A wake has no wall piece, takes the larger of the turbulent and laminar
    wake values, and is then doubled.
    """

    hk = float(shape_h)
    rt = float(max(reynolds_theta, 1.0))
    hs = float(kinetic_shape)
    us = float(slip)
    if not np.isfinite(hs) or hs <= 0.2:
        return float("nan")
    if wake:
        di = 0.0
    else:
        di = (0.5 * float(skin_friction) * us) * 2.0 / hs
        grt = float(np.log(max(rt, 1.01)))
        hmin = 1.0 + 2.1 / grt
        factor = (hk - 1.0) / max(hmin - 1.0, 1.0e-6)
        di *= 0.5 + 0.5 * float(np.tanh(factor))
    gap = max(0.995 - us, 0.0)
    di += float(sqrt_shear) ** 2 * gap * 2.0 / hs
    di += 0.15 * gap * gap / rt * 2.0 / hs
    if wake:
        laminar = _laminar_wake_dissipation(hk, rt)
        if np.isfinite(laminar):
            di = max(di, laminar)
        di *= 2.0
    else:
        laminar = _laminar_dissipation(hk, rt)
        if np.isfinite(laminar):
            di = max(di, laminar)
    return float(di)


def layer_state(
    theta: float,
    shape_h: float,
    edge_speed: float,
    edge_slope: float,
    nu: float,
    wake: bool,
) -> dict[str, float] | None:
    """Rates at one station.

    ``energy`` is θ dH*/ds, the right-hand side of the kinetic-energy
    equation. ``dtheta`` is dθ/ds from the momentum integral. Both use the
    prescribed edge speed. Returns ``None`` when the closures are not finite.
    """

    hk = float(np.clip(shape_h, 1.00005 if wake else 1.05, 3.99))
    th = float(max(theta, 1.0e-12))
    ue = float(max(edge_speed, 1.0e-6))
    if not np.isfinite(nu) or nu <= 0.0 or not np.isfinite(edge_slope):
        return None
    rt = ue * th / float(nu)
    hs = kinetic_shape_factor(hk, rt)
    if not np.isfinite(hs) or hs <= 1.0:
        return None
    us = _slip_velocity(hs, hk, wake)
    cf = 0.0 if wake else turbulent_skin_friction(hk, rt)
    if not np.isfinite(cf):
        return None
    shear = _equilibrium_sqrt_shear(hs, hk, rt, us, wake)
    di = dissipation_coefficient(hk, rt, hs, us, cf, shear, wake)
    if not np.isfinite(di):
        return None
    due = float(edge_slope)
    dtheta = cf / 2.0 - (hk + 2.0) * th / ue * due
    energy = hs * (di - cf / 2.0 - (1.0 - hk) * (th / ue) * due)
    if not np.isfinite(dtheta) or not np.isfinite(energy):
        return None
    return {
        "theta": th,
        "shape_h": hk,
        "reynolds_theta": float(rt),
        "kinetic_shape": float(hs),
        "cf": float(cf),
        "dtheta": float(dtheta),
        "energy": float(energy),
    }


def _shape_rate(
    theta: float,
    shape_h: float,
    edge_speed: float,
    edge_slope: float,
    nu: float,
    wake: bool,
    state: dict[str, float],
) -> float:
    """dH/ds from dH*/ds and H*(H, Re_θ)."""

    rt = state["reynolds_theta"]
    hs = state["kinetic_shape"]
    hs_h = (kinetic_shape_factor(shape_h + 1.0e-4, rt) - hs) / 1.0e-4
    d_rt = max(rt * 1.0e-4, 1.0e-3)
    hs_rt = (kinetic_shape_factor(shape_h, rt + d_rt) - hs) / d_rt
    if not np.isfinite(hs_h) or not np.isfinite(hs_rt):
        return float("nan")
    drt = (edge_speed / nu) * state["dtheta"] + (theta / nu) * edge_slope
    dhs = state["energy"] / max(theta, 1.0e-12)
    if abs(hs_h) < 1.0e-4:
        hs_h = 1.0e-4 if hs_h >= 0.0 else -1.0e-4
    return float((dhs - hs_rt * drt) / hs_h)


def march_interval(
    theta: float,
    shape_h: float,
    edge_start: float,
    edge_end: float,
    ds: float,
    nu: float,
    *,
    wake: bool,
) -> tuple[float, float, float, bool]:
    """Advance one interval. Returns θ, H, Cf at the downstream end, and a separation flag.

    The step is a predictor-corrector. An interval whose shape factor would
    jump by more than 0.2 is split, down to a minimum width, instead of
    taking the explicit step that drives a direct method through separation.
    """

    if not np.isfinite(ds) or ds <= 0.0:
        state = layer_state(theta, shape_h, edge_start, 0.0, nu, wake)
        cf = 0.0 if state is None else state["cf"]
        return float(theta), float(shape_h), cf, state is None
    return _march_piece(
        float(theta),
        float(shape_h),
        float(edge_start),
        float(edge_end),
        float(ds),
        float(nu),
        wake,
        0,
    )


def _march_piece(
    theta: float,
    shape_h: float,
    edge_start: float,
    edge_end: float,
    ds: float,
    nu: float,
    wake: bool,
    depth: int,
) -> tuple[float, float, float, bool]:
    due = (edge_end - edge_start) / ds
    state = layer_state(theta, shape_h, edge_start, due, nu, wake)
    if state is None:
        return theta, shape_h, 0.0, True
    dtheta = state["dtheta"] * ds
    dshape = _shape_rate(theta, shape_h, edge_start, due, nu, wake, state) * ds
    stiff = abs(dshape) > 0.2 or abs(dtheta) > 0.5 * max(theta, 1.0e-6)
    if stiff and ds > 1.0e-5 and depth < 8 and np.isfinite(dshape):
        mid = 0.5 * (edge_start + edge_end)
        half = 0.5 * ds
        theta, shape_h, cf, separated = _march_piece(
            theta, shape_h, edge_start, mid, half, nu, wake, depth + 1
        )
        if separated:
            return theta, shape_h, cf, True
        return _march_piece(theta, shape_h, mid, edge_end, half, nu, wake, depth + 1)
    if not np.isfinite(dshape) or depth >= 8 and stiff:
        return theta, shape_h, state["cf"], True

    predicted_theta = max(theta + dtheta, 1.0e-12)
    predicted_shape = shape_h + (0.0 if not np.isfinite(dshape) else dshape)
    arrival = layer_state(predicted_theta, predicted_shape, edge_end, due, nu, wake)
    if arrival is None:
        return theta, shape_h, state["cf"], True
    dshape_end = _shape_rate(
        predicted_theta, predicted_shape, edge_end, due, nu, wake, arrival
    ) * ds
    if not np.isfinite(dshape_end):
        return theta, shape_h, state["cf"], True
    theta_new = max(theta + 0.5 * (dtheta + arrival["dtheta"] * ds), 1.0e-12)
    shape_new = shape_h + 0.5 * (dshape + dshape_end)
    cf = arrival["cf"]
    if not np.isfinite(theta_new) or not np.isfinite(shape_new):
        return theta, shape_h, cf, True
    limit = 3.0 if wake else 2.8
    separated = shape_new >= limit or (not wake and cf <= 0.0)
    if shape_new < (1.01 if wake else 1.05):
        shape_new = 1.01 if wake else 1.05
    return float(theta_new), float(shape_new), float(cf), bool(separated)
