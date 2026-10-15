"""
2D constant-strength vortex and source-vortex panel solver.

Unknowns are the panel vortex strength ``γ`` and, for the source-vortex
model, a source strength ``σ``. The Neumann system uses the geometric
integrals in ``influence_2d``. Kutta closes the trailing edge when requested.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import linalg
from scipy.spatial.distance import cdist

from math_module.math_skeleton import IBLStationMesh1D, PanelMetrics2D
from math_module.panel_2d import build_panel_metrics_2d

from .influence_2d import (
    left_normal,
    unit_source_velocity,
    unit_vortex_potential,
    unit_vortex_velocity,
)
from .physics_skeleton import (
    InvalidFlowConditionError,
    InviscidSolution,
    SingularInfluenceMatrixError,
    TwoDPanelModel,
)

_COND_LIMIT = 1.0e12


def trailing_edge_panels(te_index: int, n_panels: int) -> tuple[int, int]:
    """The two panels that meet at ``te_index``.

    Closed contours store the repeated trailing node as index ``n_panels``.
    Any index that does not touch exactly two panels is rejected.
    """

    if n_panels < 3:
        raise InvalidFlowConditionError("2D Kutta condition needs at least 3 panels.")
    te = int(te_index)
    if te == n_panels:
        te = 0
    if te < 0 or te >= n_panels:
        raise InvalidFlowConditionError(
            f"te_index {te_index} does not identify exactly two panels."
        )
    first = te
    second = (te - 1) % n_panels
    if first == second:
        raise InvalidFlowConditionError(
            f"te_index {te_index} does not identify exactly two panels."
        )
    return int(first), int(second)


def _kutta_pair(panel: PanelMetrics2D) -> tuple[int, int]:
    """Upper and lower panels at the trailing edge.

    Sharp sections: the two panels that share ``te_index``. Blunt sections
    (a vertical trailing face): the aft-most panel on each side, which are
    the panels that carry the Kutta jump. Both choices coincide on a NACA
    trailing edge.
    """

    length = np.asarray(panel["length"], dtype=np.float64)
    n_panels = int(length.shape[0])
    touching = trailing_edge_panels(int(panel["te_index"]), n_panels)
    normal = np.asarray(panel["normal"], dtype=np.float64)
    x = np.asarray(panel["collocation"], dtype=np.float64)[:, 0]
    upper = np.flatnonzero(normal[:, 1] > 0.7)
    lower = np.flatnonzero(normal[:, 1] < -0.7)
    if upper.size == 0 or lower.size == 0:
        return touching
    pair = (int(upper[np.argmax(x[upper])]), int(lower[np.argmax(x[lower])]))
    if pair[0] == pair[1]:
        return touching
    # Sharp trailing edges already present the upper and lower panels.
    if touching[0] in (pair[0], pair[1]) and touching[1] in (pair[0], pair[1]):
        return touching
    return pair


def _stretch_panel(panel: PanelMetrics2D, beta: float) -> PanelMetrics2D:
    nodes = np.array(panel["nodes"], dtype=np.float64, copy=True)
    nodes[:, 0] *= beta
    stretched = build_panel_metrics_2d(nodes, clustering=panel["clustering"])
    return stretched


def _solve_dense(matrix: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, float]:
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(rhs)):
        raise SingularInfluenceMatrixError("Influence matrix or right-hand side is non-finite.")
    try:
        condition = float(np.linalg.cond(matrix, p=1))
    except np.linalg.LinAlgError as exc:
        raise SingularInfluenceMatrixError("Influence matrix condition estimate failed.") from exc
    if not np.isfinite(condition) or condition > _COND_LIMIT:
        raise SingularInfluenceMatrixError(
            f"Influence matrix condition number {condition:.3e} exceeds {_COND_LIMIT:.0e}."
        )
    try:
        solution = linalg.solve(matrix, rhs, assume_a="gen", overwrite_a=False, overwrite_b=False)
    except (np.linalg.LinAlgError, ValueError) as exc:
        raise SingularInfluenceMatrixError("Influence matrix is singular.") from exc
    if not np.all(np.isfinite(solution)):
        raise SingularInfluenceMatrixError("Panel solution is non-finite.")
    return np.asarray(solution, dtype=np.float64), condition


def _normal_matrix(normal: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return normal[:, 0:1] * u + normal[:, 1:2] * v


def _tangential_matrix(tangent: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return tangent[:, 0:1] * u + tangent[:, 1:2] * v


def _exterior_vortex_jump(normal: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    """Tangential self-jump coefficient of a unit vortex on the exterior side.

    ``+zl`` (left of the tangent) carries ``+1/2``. The exterior side follows
    the outward normal, so the jump is ``±1/2``.
    """

    sign = np.sign(np.sum(normal * left_normal(tangent), axis=1))
    sign = np.where(sign == 0.0, 1.0, sign)
    # Exterior tangential velocity on the outward side of a unit vortex.
    return 0.5 * sign


def sample_edge_speed(
    collocation: np.ndarray,
    speed: np.ndarray,
    ibl: IBLStationMesh1D,
) -> tuple[np.ndarray, np.ndarray]:
    """Exterior ``|V_t|`` on IBL stations from the panel that owns each segment.

    Each IBL branch is a walk along contour nodes, so a segment midpoint is a
    panel collocation. Station speed is the average of the two adjacent panels.
    """

    return (
        _station_speed(collocation, speed, ibl["xy_upper"]),
        _station_speed(collocation, speed, ibl["xy_lower"]),
    )


def _station_speed(collocation: np.ndarray, speed: np.ndarray, xy: np.ndarray) -> np.ndarray:
    points = np.asarray(xy, dtype=np.float64)
    count = int(points.shape[0])
    if count == 0:
        return np.zeros(0, dtype=np.float64)
    if count == 1:
        nearest = int(np.argmin(np.sum((collocation - points[0]) ** 2, axis=1)))
        return np.array([abs(float(speed[nearest]))], dtype=np.float64)
    midpoints = 0.5 * (points[:-1] + points[1:])
    nearest = np.argmin(cdist(midpoints, collocation), axis=1)
    segment = np.abs(np.asarray(speed, dtype=np.float64)[nearest])
    station = np.empty(count, dtype=np.float64)
    station[0] = segment[0]
    station[-1] = segment[-1]
    if count > 2:
        station[1:-1] = 0.5 * (segment[:-1] + segment[1:])
    return station


def solve_2d(
    panel: PanelMetrics2D,
    alpha_rad: float,
    transpiration: Optional[np.ndarray],
    panel_model: TwoDPanelModel,
    kutta: bool,
    *,
    beta: float = 1.0,
    ibl: Optional[IBLStationMesh1D] = None,
) -> tuple[InviscidSolution, Optional[np.ndarray], Optional[np.ndarray]]:
    """Solve the 2D panel system on a copy of ``panel``.

    Returns the inviscid solution, plus exterior edge speeds on the upper and
    lower IBL stations when ``ibl`` is provided. Speeds are aligned with the
    original (unstretched) stations.
    """

    if panel_model not in ("vortex", "source_vortex"):
        raise InvalidFlowConditionError(f"Unknown 2D panel model {panel_model!r}.")

    original_collocation = np.asarray(panel["collocation"], dtype=np.float64)
    work = panel if beta == 1.0 else _stretch_panel(panel, beta)
    nodes = np.asarray(work["nodes"], dtype=np.float64)
    collocation = np.asarray(work["collocation"], dtype=np.float64)
    tangent = np.asarray(work["tangent"], dtype=np.float64)
    normal = np.asarray(work["normal"], dtype=np.float64)
    length = np.asarray(work["length"], dtype=np.float64)
    n_panels = int(length.shape[0])
    if collocation.shape != (n_panels, 2):
        raise InvalidFlowConditionError("panel_2d collocation does not match the panel count.")

    if transpiration is None:
        wall_normal = np.zeros(n_panels, dtype=np.float64)
    else:
        wall_normal = np.asarray(transpiration, dtype=np.float64).reshape(-1)
        if wall_normal.shape != (n_panels,):
            raise InvalidFlowConditionError(
                f"Transpiration velocity must have shape ({n_panels},)."
            )
        if not np.all(np.isfinite(wall_normal)):
            raise InvalidFlowConditionError("Transpiration velocity must be finite.")

    freestream = np.array([np.cos(alpha_rad), np.sin(alpha_rad)], dtype=np.float64)
    # No-penetration is enforced with the streamfunction. The equivalent
    # discrete Neumann matrix for constant vortices has a checkerboard null
    # space; ψ = const is the integral of that boundary condition and is the
    # system that recovers 2πα. Wall transpiration is a Neumann residual and
    # is carried by the source sheet below — adding it to this RHS excites
    # that near-null mode and spikes the vortex strength.
    _phi, psi = unit_vortex_potential(collocation, nodes, tangent, length)
    freestream_psi = collocation[:, 1] * freestream[0] - collocation[:, 0] * freestream[1]

    vortex_u, vortex_v = unit_vortex_velocity(collocation, nodes, tangent, length)
    tangent_gamma = _tangential_matrix(tangent, vortex_u, vortex_v)
    tangent_gamma[np.diag_indices(n_panels)] = _exterior_vortex_jump(normal, tangent)

    def _vortex_dirichlet() -> tuple[np.ndarray, float]:
        matrix = np.zeros((n_panels + 1, n_panels + 1), dtype=np.float64)
        matrix[:n_panels, :n_panels] = psi
        matrix[:n_panels, n_panels] = -1.0
        rhs = np.zeros(n_panels + 1, dtype=np.float64)
        rhs[:n_panels] = -freestream_psi
        if kutta:
            te_a, te_b = _kutta_pair(work)
            matrix[n_panels, te_a] = 1.0
            matrix[n_panels, te_b] = 1.0
        else:
            matrix[n_panels, :n_panels] = length
        return _solve_dense(matrix, rhs)

    packed, condition = _vortex_dirichlet()
    gamma = packed[:n_panels]
    sigma = None
    tangential_speed = tangent_gamma @ gamma + tangent @ freestream
    if panel_model == "source_vortex" or np.any(wall_normal != 0.0):
        source_u, source_v = unit_source_velocity(collocation, nodes, tangent, length)
        normal_sigma = _normal_matrix(normal, source_u, source_v)
        tangent_sigma = _tangential_matrix(tangent, source_u, source_v)
        normal_sigma[np.diag_indices(n_panels)] = 0.5
        tangent_sigma[np.diag_indices(n_panels)] = 0.0
        normal_gamma = _normal_matrix(normal, vortex_u, vortex_v)
        normal_gamma[np.diag_indices(n_panels)] = 0.0
        # Source strengths cancel the normal velocity the vortex sheet and the
        # freestream still leave, and they carry the viscous transpiration.
        # The source self-term is 1/2.
        source_rhs = -normal @ freestream + wall_normal - normal_gamma @ gamma
        try:
            sigma, _source_condition = _solve_dense(normal_sigma, source_rhs)
        except SingularInfluenceMatrixError:
            sigma = np.linalg.lstsq(normal_sigma, source_rhs, rcond=1e-10)[0]
            if not np.all(np.isfinite(sigma)):
                raise SingularInfluenceMatrixError("Source-vortex system is singular.") from None
        sigma = np.asarray(sigma, dtype=np.float64)
        tangential_speed = (
            tangent_gamma @ gamma + tangent_sigma @ sigma + tangent @ freestream
        )

    speed2 = tangential_speed * tangential_speed
    cp = 1.0 - speed2
    if beta != 1.0:
        cp = cp / beta
        target = np.sqrt(np.maximum(1.0 - cp, 0.0))
        magnitude = np.maximum(np.abs(tangential_speed), 1e-14)
        tangential_speed = tangential_speed * (target / magnitude)

    velocity = tangential_speed[:, None] * np.asarray(panel["tangent"], dtype=np.float64)
    solution = InviscidSolution(
        gamma=np.ascontiguousarray(gamma, dtype=np.float64),
        mu=None,
        sigma=None if sigma is None else np.ascontiguousarray(sigma, dtype=np.float64),
        cp_inviscid=np.ascontiguousarray(cp, dtype=np.float64),
        v_tangential=np.ascontiguousarray(velocity, dtype=np.float64),
        influence_condition_number=float(condition),
    )
    if ibl is None:
        return solution, None, None
    upper, lower = sample_edge_speed(original_collocation, tangential_speed, ibl)
    return solution, upper, lower
