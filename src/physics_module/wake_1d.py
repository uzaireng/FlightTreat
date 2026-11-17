"""Direct viscous wake behind a 2D section.

The wake is the same equilibrium two-equation shear layer as the turbulent
wall march, with the wall shear removed. It is not a source sheet, a solid
base, or a displaced panel geometry. Those couplings were tried previously
and did not improve lift without spoiling drag. This wake does not enter
the panel system.

At the trailing edge the two wall layers and, when the edge is open, the
stagnant base are joined so that two integrals are continuous:

- displacement flux ``Ue δ*``, including the base height because fluid at
  rest in the gap contributes its full height to the displacement;
- momentum defect ``Ue² θ``. A stagnant base has ``u = 0``, so it does not
  contribute to θ.

Downstream, θ and H march on the edge speed of the existing panel field,
sampled on the freestream ray that leaves the trailing edge. The initial
edge speed is the last resolved surface station, not the cosine cluster,
because that cluster is where the panel speed falls after θ and H have
already been frozen. One chord downstream, the Squire–Young formula
converts the station into a far-wake momentum thickness. That number is
stored as ``cd_momentum``. It is not written over the surface-force drag:
on the attached cases that were compared, the defect is below the surface
integral and the difference grows with lift.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from math_module.math_skeleton import IBLStationMesh1D, PanelMetrics2D

from .boundary_layer_1d import last_resolved_index
from .drela_equilibrium import march_interval
from .influence_2d import unit_source_velocity, unit_vortex_velocity
from .physics_skeleton import IBLSolution

# Drela applies Squire–Young at the last wake station, typically one chord
# downstream, where the edge speed is close to the freestream and the
# near-wake assumption is no longer required.
_WAKE_CHORDS = 1.0
_WAKE_STATIONS = 41
_SAMPLE_START_CHORDS = 0.02
_BLEND_CHORDS = 0.05


def merge_trailing_edge(
    ue_upper: float,
    delta_star_upper: float,
    theta_upper: float,
    ue_lower: float,
    delta_star_lower: float,
    theta_lower: float,
    base_height: float,
) -> tuple[float, float, float, float]:
    """Join both sides. Returns ``Ue``, ``θ``, ``δ*``, ``H``.

    Displacement flux and momentum defect of the two wall layers are kept.
    The base adds ``Ue * h_TE`` to the displacement flux and nothing to the
    momentum defect.
    """

    ue_u = float(max(ue_upper, 1.0e-8))
    ue_l = float(max(ue_lower, 1.0e-8))
    ue = 0.5 * (ue_u + ue_l)
    gap = float(max(base_height, 0.0))
    flux = ue_u * float(delta_star_upper) + ue_l * float(delta_star_lower) + ue * gap
    momentum = ue_u ** 2 * float(max(theta_upper, 0.0)) + ue_l ** 2 * float(max(theta_lower, 0.0))
    theta = momentum / max(ue * ue, 1.0e-16)
    delta_star = flux / max(ue, 1.0e-8)
    if theta <= 0.0 or not np.isfinite(theta) or not np.isfinite(delta_star):
        return ue, float("nan"), float("nan"), float("nan")
    return ue, float(theta), float(delta_star), float(delta_star / theta)


def squire_young(theta: float, shape_h: float, edge_speed: float, chord: float) -> float:
    """Far-wake drag ``2 θ∞ / c`` from a station that has not yet reached infinity.

    ``θ∞ = θ (Ue / V∞)^((H+5)/2)`` with ``V∞ = 1`` in the panel frame.
    """

    if (
        not np.isfinite(theta)
        or not np.isfinite(shape_h)
        or not np.isfinite(edge_speed)
        or not np.isfinite(chord)
        or theta < 0.0
        or shape_h < 1.0
        or edge_speed <= 0.0
        or chord <= 0.0
    ):
        return float("nan")
    exponent = 0.5 * (float(shape_h) + 5.0)
    return float(2.0 * (float(theta) / float(chord)) * float(edge_speed) ** exponent)


def momentum_drag_is_usable(cd: float, shape_end: float, edge_end: float, wake_separated: bool) -> bool:
    """Reject a march that left the attached-wake range.

    The bounds are failure checks, not a fit to a measured drag. A shape
    factor above 2.4 is still separated, and an edge speed far from the
    freestream means the sample missed the wake.
    """

    if wake_separated:
        return False
    if not np.isfinite(cd) or cd <= 0.0 or cd > 0.2:
        return False
    if not np.isfinite(shape_end) or not 1.01 <= float(shape_end) <= 2.4:
        return False
    if not np.isfinite(edge_end) or not 0.55 <= float(edge_end) <= 1.35:
        return False
    return True


def _panel_edge_speed(
    panel: PanelMetrics2D,
    gamma: np.ndarray,
    sigma: Optional[np.ndarray],
    points: np.ndarray,
    alpha_rad: float,
    beta: float,
) -> np.ndarray:
    """Edge speed on the same definition the wall boundary layer receives."""

    query = np.array(points, dtype=np.float64, copy=True)
    if beta != 1.0:
        query[:, 0] *= beta
    nodes = np.asarray(panel["nodes"], dtype=np.float64)
    tangent = np.asarray(panel["tangent"], dtype=np.float64)
    length = np.asarray(panel["length"], dtype=np.float64)
    vortex_u, vortex_v = unit_vortex_velocity(query, nodes, tangent, length)
    induced_u = vortex_u @ np.asarray(gamma, dtype=np.float64)
    induced_v = vortex_v @ np.asarray(gamma, dtype=np.float64)
    if sigma is not None:
        source_u, source_v = unit_source_velocity(query, nodes, tangent, length)
        induced_u = induced_u + source_u @ np.asarray(sigma, dtype=np.float64)
        induced_v = induced_v + source_v @ np.asarray(sigma, dtype=np.float64)
    freestream = np.array([np.cos(alpha_rad), np.sin(alpha_rad)], dtype=np.float64)
    speed = np.hypot(induced_u + freestream[0], induced_v + freestream[1])
    cp = 1.0 - speed * speed
    if beta != 1.0:
        cp = cp / beta
    return np.sqrt(np.maximum(1.0 - cp, 0.0))


def _empty_wake() -> dict[str, np.ndarray | float | bool]:
    empty = np.zeros(0, dtype=np.float64)
    return {
        "s": empty,
        "theta": empty,
        "delta_star": empty,
        "shape_h": empty,
        "ue": empty,
        "cd_momentum": float("nan"),
        "separated": False,
        "usable": False,
    }


def march_viscous_wake(
    ibl_mesh: IBLStationMesh1D,
    ibl: IBLSolution,
    panel: PanelMetrics2D,
    gamma: np.ndarray,
    chord: float,
    reynolds: float,
    alpha_rad: float,
    *,
    beta: float = 1.0,
    sigma: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray | float | bool]:
    """March the merged wake one chord downstream and evaluate Squire–Young there."""

    if not np.isfinite(chord) or chord <= 0.0 or not np.isfinite(reynolds) or reynolds <= 0.0:
        return _empty_wake()
    ue_u = np.asarray(ibl["ue_upper"], dtype=np.float64)
    ue_l = np.asarray(ibl["ue_lower"], dtype=np.float64)
    if ue_u.size < 2 or ue_l.size < 2:
        return _empty_wake()
    # θ and H are already frozen through the cosine cluster. The edge
    # speed is not: the last panels drive it down. The wake starts from
    # the speed at the last resolved station, which is the speed the
    # boundary layer actually used.
    resolved_u = last_resolved_index(ibl_mesh["s_upper"])
    resolved_l = last_resolved_index(ibl_mesh["s_lower"])
    xy_u = np.asarray(ibl_mesh["xy_upper"], dtype=np.float64)
    xy_l = np.asarray(ibl_mesh["xy_lower"], dtype=np.float64)
    origin = 0.5 * (xy_u[-1] + xy_l[-1])
    gap = float(np.linalg.norm(xy_u[-1] - xy_l[-1]))
    ue0, theta0, _delta0, shape0 = merge_trailing_edge(
        float(ue_u[resolved_u]),
        float(ibl["delta_star_upper"][-1]),
        float(ibl["theta_upper"][-1]),
        float(ue_l[resolved_l]),
        float(ibl["delta_star_lower"][-1]),
        float(ibl["theta_lower"][-1]),
        gap,
    )
    if not np.isfinite(theta0) or not np.isfinite(shape0):
        return _empty_wake()

    direction = np.array([np.cos(alpha_rad), np.sin(alpha_rad)], dtype=np.float64)
    n_sample = _WAKE_STATIONS - 1
    s_sample = np.linspace(
        _SAMPLE_START_CHORDS * chord,
        _WAKE_CHORDS * chord,
        n_sample,
        dtype=np.float64,
    )
    points = origin + s_sample[:, None] * direction
    sampled = _panel_edge_speed(panel, gamma, sigma, points, alpha_rad, beta)
    bad = ~np.isfinite(sampled) | (sampled < 0.2) | (sampled > 1.8)
    sampled = np.array(sampled, copy=True)
    sampled[bad] = ue0
    blend = np.clip(s_sample / (_BLEND_CHORDS * chord), 0.0, 1.0)
    ue_sample = (1.0 - blend) * ue0 + blend * sampled

    s = np.empty(_WAKE_STATIONS, dtype=np.float64)
    ue = np.empty(_WAKE_STATIONS, dtype=np.float64)
    theta = np.empty(_WAKE_STATIONS, dtype=np.float64)
    shape = np.empty(_WAKE_STATIONS, dtype=np.float64)
    s[0] = 0.0
    ue[0] = ue0
    theta[0] = theta0
    shape[0] = shape0
    s[1:] = s_sample
    ue[1:] = ue_sample

    nu = float(chord) / float(reynolds)
    separated = False
    last = 0
    for index in range(0, _WAKE_STATIONS - 1):
        ds = float(s[index + 1] - s[index])
        theta_new, shape_new, _cf, step_separated = march_interval(
            float(theta[index]),
            float(shape[index]),
            float(ue[index]),
            float(ue[index + 1]),
            ds,
            nu,
            wake=True,
        )
        theta[index + 1] = theta_new
        shape[index + 1] = shape_new
        last = index + 1
        if step_separated or not np.isfinite(shape_new) or not np.isfinite(theta_new):
            separated = True
            theta[index + 1 :] = theta_new
            shape[index + 1 :] = shape_new
            last = _WAKE_STATIONS - 1
            break

    delta_star = shape * theta
    cd = squire_young(float(theta[last]), float(shape[last]), float(ue[last]), chord)
    usable = momentum_drag_is_usable(cd, float(shape[last]), float(ue[last]), separated)
    return {
        "s": np.ascontiguousarray(s, dtype=np.float64),
        "theta": np.ascontiguousarray(theta, dtype=np.float64),
        "delta_star": np.ascontiguousarray(delta_star, dtype=np.float64),
        "shape_h": np.ascontiguousarray(shape, dtype=np.float64),
        "ue": np.ascontiguousarray(ue, dtype=np.float64),
        "cd_momentum": float(cd),
        "separated": bool(separated),
        "usable": bool(usable),
    }
