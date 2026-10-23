"""
Constant-strength 2D source and vortex kernels.

Geometric integrals follow Katz & Plotkin: endpoint radii and turning angles
in the panel frame, then a rotation back to global (x, y). Self-influence is
left to the caller; these kernels are the off-body geometric integrals.
"""

from __future__ import annotations

import numpy as np

_TWOPI = 2.0 * np.pi
_EPS = 1e-14


def _wrap_angle(delta: np.ndarray) -> np.ndarray:
    """Fold an angle difference into (-π, π).

    ``atan2`` jumps by 2π when the panel straddles the negative local-x axis.
    The angle subtended by a straight panel is at most π.
    """

    return (delta + np.pi) % _TWOPI - np.pi


def panel_local_coordinates(
    points: np.ndarray,
    origin: np.ndarray,
    tangent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map ``points`` into every panel frame.

    ``points`` is ``(N, 2)``, ``origin`` and ``tangent`` are ``(P, 2)``.
    ``+zl`` lies to the left of ``tangent``. Both outputs are ``(N, P)``.
    """

    relative = points[:, None, :] - origin[None, :, :]
    tx = tangent[:, 0]
    ty = tangent[:, 1]
    xl = relative[..., 0] * tx + relative[..., 1] * ty
    zl = -relative[..., 0] * ty + relative[..., 1] * tx
    return xl, zl


def _endpoint_angles(
    points: np.ndarray,
    nodes: np.ndarray,
    tangent: np.ndarray,
    length: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    origin = np.asarray(nodes[:-1], dtype=np.float64)
    xl, zl = panel_local_coordinates(points, origin, tangent)
    xl2 = xl - np.asarray(length, dtype=np.float64)[None, :]
    r1 = np.hypot(xl, zl)
    r2 = np.hypot(xl2, zl)
    theta1 = np.arctan2(zl, xl)
    theta2 = np.arctan2(zl, xl2)
    return xl, xl2, r1, r2, theta1, theta2


def _to_global(
    u_local: np.ndarray,
    w_local: np.ndarray,
    tangent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate panel-frame ``(u, w)`` with ``w`` along the left normal."""

    tx = tangent[:, 0]
    ty = tangent[:, 1]
    u = u_local * tx[None, :] + w_local * (-ty[None, :])
    v = u_local * ty[None, :] + w_local * (tx[None, :])
    return u, v


def unit_vortex_velocity(
    points: np.ndarray,
    nodes: np.ndarray,
    tangent: np.ndarray,
    length: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Global velocity at ``points`` from a unit vortex on each panel.

    Returns ``u``, ``v`` with shape ``(N, P)``.
    """

    _xl, _xl2, r1, r2, theta1, theta2 = _endpoint_angles(points, nodes, tangent, length)
    # Positive γ follows the panel tangent. The sign is chosen so a
    # lifting solution has Σ γ S > 0 and Cl ≈ 2 Σ γ S / chord.
    u_local = _wrap_angle(theta2 - theta1) / _TWOPI
    w_local = -np.log(np.maximum(r1, _EPS) / np.maximum(r2, _EPS)) / _TWOPI
    return _to_global(u_local, w_local, tangent)


def unit_source_velocity(
    points: np.ndarray,
    nodes: np.ndarray,
    tangent: np.ndarray,
    length: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Global velocity at ``points`` from a unit source on each panel."""

    _xl, _xl2, r1, r2, theta1, theta2 = _endpoint_angles(points, nodes, tangent, length)
    u_local = np.log(np.maximum(r1, _EPS) / np.maximum(r2, _EPS)) / _TWOPI
    w_local = _wrap_angle(theta2 - theta1) / _TWOPI
    return _to_global(u_local, w_local, tangent)


def unit_vortex_potential(
    points: np.ndarray,
    nodes: np.ndarray,
    tangent: np.ndarray,
    length: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Velocity potential and streamfunction of a unit vortex panel.

    The complex potential is the integral of the Katz vortex velocity,
    ``w = -i/2π [(ζ-S) Log(ζ-S) - ζ Log(ζ) + S]``. Branch cuts lie on the
    panel. Farfield samples are diagnostic; surface ``Cp`` does not use them.
    """

    origin = np.asarray(nodes[:-1], dtype=np.float64)
    xl, zl = panel_local_coordinates(points, origin, tangent)
    span = np.asarray(length, dtype=np.float64)[None, :]
    zeta = xl + 1j * zl
    zeta_end = (xl - span) + 1j * zl
    tiny = np.abs(zeta) < _EPS
    tiny_end = np.abs(zeta_end) < _EPS
    zeta = np.where(tiny, _EPS + 0j, zeta)
    zeta_end = np.where(tiny_end, _EPS + 0j, zeta_end)
    potential = -1j / _TWOPI * (zeta_end * np.log(zeta_end) - zeta * np.log(zeta) + span)
    return np.real(potential), np.imag(potential)


def panel_sheet_influence(
    points: np.ndarray,
    nodes: np.ndarray,
    tangent: np.ndarray,
    length: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vortex and source velocity plus vortex potential from one geometry pass.

    Velocity matches ``unit_vortex_velocity`` and ``unit_source_velocity``.
    Potential matches ``unit_vortex_potential`` to roundoff. The surface
    solver keeps the original kernels; this fused pass is for the farfield,
    which was repeating the same endpoint geometry three times.
    """

    origin = np.asarray(nodes[:-1], dtype=np.float64)
    xl, zl = panel_local_coordinates(points, origin, tangent)
    span = np.asarray(length, dtype=np.float64)[None, :]
    xl2 = xl - span
    r1 = np.hypot(xl, zl)
    r2 = np.hypot(xl2, zl)
    theta1 = np.arctan2(zl, xl)
    theta2 = np.arctan2(zl, xl2)
    turned = _wrap_angle(theta2 - theta1)
    safe_r1 = np.maximum(r1, _EPS)
    safe_r2 = np.maximum(r2, _EPS)
    log_ratio = np.log(safe_r1 / safe_r2)
    vortex_u_local = turned / _TWOPI
    vortex_w_local = -log_ratio / _TWOPI
    source_u_local = log_ratio / _TWOPI
    source_w_local = turned / _TWOPI
    vortex_u, vortex_v = _to_global(vortex_u_local, vortex_w_local, tangent)
    source_u, source_v = _to_global(source_u_local, source_w_local, tangent)
    log_r1 = np.log(safe_r1)
    log_r2 = np.log(safe_r2)
    # Real and imaginary parts of (ζ-S) Log(ζ-S) - ζ Log(ζ) + S.
    imag_diff = xl2 * theta2 - xl * theta1 + zl * (log_r2 - log_r1)
    real_diff = xl2 * log_r2 - xl * log_r1 + zl * (theta1 - theta2) + span
    phi = imag_diff / _TWOPI
    psi = -real_diff / _TWOPI
    return vortex_u, vortex_v, source_u, source_v, phi, psi


def left_normal(tangent: np.ndarray) -> np.ndarray:
    """Unit normal to the left of ``tangent`` (the ``+zl`` direction)."""

    return np.column_stack((-tangent[:, 1], tangent[:, 0]))
