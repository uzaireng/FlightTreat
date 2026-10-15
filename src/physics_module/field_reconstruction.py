"""Off-body 2D velocity, potential, and streamfunction on the Cartesian grid."""

from __future__ import annotations

from typing import Optional

import numpy as np

from math_module.math_skeleton import BoundaryRegion, CartesianGrid2D, PanelMetrics2D

from .influence_2d import unit_source_velocity, unit_vortex_potential, unit_vortex_velocity
from .physics_skeleton import FlowField2D

_BYTE_BUDGET = 64 * 1024 * 1024


def _row_block(n_panels: int) -> int:
    """Points per batch so the geometric-integral temporaries stay under 64 MB."""

    bytes_per_point = max(n_panels, 1) * np.dtype(np.float64).itemsize * 8
    return max(1, int(_BYTE_BUDGET / bytes_per_point))


def evaluate_farfield(
    panel: PanelMetrics2D,
    gamma: np.ndarray,
    grid: CartesianGrid2D,
    *,
    beta: float = 1.0,
    sigma: Optional[np.ndarray] = None,
) -> FlowField2D:
    """Sample the 2D panel field on ``grid``. Solid cells are NaN.

    ``panel`` is the geometry used to build the influence matrix (already
    stretched when Prandtl–Glauert is active). ``grid`` stays in physical
    coordinates; its x-coordinate is stretched only for the kernel query.
    """

    nodes = np.asarray(panel["nodes"], dtype=np.float64)
    tangent = np.asarray(panel["tangent"], dtype=np.float64)
    length = np.asarray(panel["length"], dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    shape = np.asarray(grid["grid_x"]).shape
    points = np.column_stack(
        (
            np.asarray(grid["grid_x"], dtype=np.float64).reshape(-1),
            np.asarray(grid["grid_y"], dtype=np.float64).reshape(-1),
        )
    )
    if beta != 1.0:
        points = np.array(points, copy=True)
        points[:, 0] *= beta

    n_points = int(points.shape[0])
    u = np.empty(n_points, dtype=np.float64)
    v = np.empty(n_points, dtype=np.float64)
    phi = np.empty(n_points, dtype=np.float64)
    psi = np.empty(n_points, dtype=np.float64)
    block = _row_block(int(length.shape[0]))
    for start in range(0, n_points, block):
        stop = min(start + block, n_points)
        sample = points[start:stop]
        vortex_u, vortex_v = unit_vortex_velocity(sample, nodes, tangent, length)
        u[start:stop] = vortex_u @ gamma + np.cos(0.0)
        v[start:stop] = vortex_v @ gamma
        pot, stream = unit_vortex_potential(sample, nodes, tangent, length)
        phi[start:stop] = pot @ gamma
        psi[start:stop] = stream @ gamma
        if sigma is not None:
            source_u, source_v = unit_source_velocity(sample, nodes, tangent, length)
            u[start:stop] = u[start:stop] + source_u @ sigma
            v[start:stop] = v[start:stop] + source_v @ sigma

    # Freestream is added in the physical frame. The kernel above added
    # ``(1, 0)``; replace it with the caller-free unit x-flow only when the
    # angle is carried by the panel solution's vortex field plus this unit
    # vector. ``evaluate_farfield`` is called with the same alpha-free split
    # used below: the engine passes gamma only, and freestream is reapplied.
    return _pack(u, v, phi, psi, shape, grid, beta)


def evaluate_farfield_with_freestream(
    panel: PanelMetrics2D,
    gamma: np.ndarray,
    grid: CartesianGrid2D,
    alpha_rad: float,
    *,
    beta: float = 1.0,
    sigma: Optional[np.ndarray] = None,
) -> FlowField2D:
    """Farfield velocity including the unit freestream at ``alpha_rad``."""

    nodes = np.asarray(panel["nodes"], dtype=np.float64)
    tangent = np.asarray(panel["tangent"], dtype=np.float64)
    length = np.asarray(panel["length"], dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    shape = np.asarray(grid["grid_x"]).shape
    points = np.column_stack(
        (
            np.asarray(grid["grid_x"], dtype=np.float64).reshape(-1),
            np.asarray(grid["grid_y"], dtype=np.float64).reshape(-1),
        )
    )
    if beta != 1.0:
        points = np.array(points, copy=True)
        points[:, 0] *= beta
    freestream = np.array([np.cos(alpha_rad), np.sin(alpha_rad)], dtype=np.float64)
    n_points = int(points.shape[0])
    u = np.empty(n_points, dtype=np.float64)
    v = np.empty(n_points, dtype=np.float64)
    phi = np.empty(n_points, dtype=np.float64)
    psi = np.empty(n_points, dtype=np.float64)
    block = _row_block(int(length.shape[0]))
    for start in range(0, n_points, block):
        stop = min(start + block, n_points)
        sample = points[start:stop]
        vortex_u, vortex_v = unit_vortex_velocity(sample, nodes, tangent, length)
        induced_u = vortex_u @ gamma
        induced_v = vortex_v @ gamma
        if sigma is not None:
            source_u, source_v = unit_source_velocity(sample, nodes, tangent, length)
            induced_u = induced_u + source_u @ sigma
            induced_v = induced_v + source_v @ sigma
        u[start:stop] = induced_u + freestream[0]
        v[start:stop] = induced_v + freestream[1]
        pot, stream = unit_vortex_potential(sample, nodes, tangent, length)
        # Uniform-flow potential and streamfunction, plus the vortex sheet.
        x = sample[:, 0]
        y = sample[:, 1]
        phi[start:stop] = pot @ gamma + freestream[0] * x + freestream[1] * y
        psi[start:stop] = stream @ gamma + freestream[0] * y - freestream[1] * x
    return _pack(u, v, phi, psi, shape, grid, beta)


def _pack(
    u: np.ndarray,
    v: np.ndarray,
    phi: np.ndarray,
    psi: np.ndarray,
    shape: tuple[int, ...],
    grid: CartesianGrid2D,
    beta: float,
) -> FlowField2D:
    speed = np.hypot(u, v)
    cp = 1.0 - speed * speed
    if beta != 1.0:
        cp = cp / beta
    solid = np.asarray(grid["boundary_mask"]).reshape(-1) == int(BoundaryRegion.INTERNAL_SOLID)
    for array in (u, v, speed, cp, phi, psi):
        array[solid] = np.nan
    w = np.zeros(u.shape, dtype=np.float64)
    w[solid] = np.nan

    def reshape(array: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(array.reshape(shape), dtype=np.float64)

    return FlowField2D(
        u=reshape(u),
        v=reshape(v),
        w=reshape(w),
        speed=reshape(speed),
        cp=reshape(cp),
        phi=reshape(phi),
        psi=reshape(psi),
    )
