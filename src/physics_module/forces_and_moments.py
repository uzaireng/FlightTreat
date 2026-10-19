"""Panel pressure and skin-friction integration to surface loads."""

from __future__ import annotations

from typing import Optional

import numpy as np

from math_module.math_skeleton import MeshOutput

from .physics_skeleton import (
    IBLSolution,
    InvalidFlowConditionError,
    InviscidSolution,
    SurfaceLoads,
)

# SurfaceLoads are coefficients at the panel solver's unit onset. Stored
# edge speed is already U_e / V_inf in that frame, so the freestream factor
# below divides by this same unit speed.
_UNIT_FREESTREAM = 1.0


def _segment_traction(
    cf: np.ndarray,
    tangent: np.ndarray,
    ds: np.ndarray,
    edge_speed: np.ndarray,
    v_inf: float,
) -> np.ndarray:
    """Skin-friction force from station ``Cf`` along a marching branch.

    ``Cf`` is wall shear divided by the local edge dynamic pressure.
    Load coefficients divide by freestream dynamic pressure, so each
    station is weighted by ``(U_e / V_inf)**2`` once, before the segment
    average. Force direction and segment length are unchanged.
    """

    if ds.size == 0:
        return np.zeros(tangent.shape[1], dtype=np.float64)
    cf_local = np.asarray(cf, dtype=np.float64)
    speed = np.asarray(edge_speed, dtype=np.float64)
    if speed.shape != cf_local.shape:
        raise InvalidFlowConditionError(
            "Edge speed and skin friction must share the same stations."
        )
    if not np.isfinite(v_inf) or v_inf == 0.0:
        raise InvalidFlowConditionError("Freestream speed must be finite and non-zero.")
    cf_freestream = cf_local * (speed / float(v_inf)) ** 2
    cf_seg = 0.5 * (cf_freestream[:-1] + cf_freestream[1:])
    direction = 0.5 * (tangent[:-1] + tangent[1:])
    norm = np.linalg.norm(direction, axis=1)
    direction = direction / np.maximum(norm[:, None], 1e-14)
    return np.sum(cf_seg[:, None] * direction * ds[:, None], axis=0)


def _loads_2d(
    mesh: MeshOutput,
    inviscid: InviscidSolution,
    ibl: Optional[IBLSolution],
    alpha_rad: float,
    compressibility_applied: bool,
) -> SurfaceLoads:
    panel = mesh["panel_2d"]
    assert panel is not None
    normal = np.asarray(panel["normal"], dtype=np.float64)
    length = np.asarray(panel["length"], dtype=np.float64)
    collocation = np.asarray(panel["collocation"], dtype=np.float64)
    cp = np.asarray(inviscid["cp_inviscid"], dtype=np.float64)
    pressure = np.sum(-cp[:, None] * normal * length[:, None], axis=0)
    friction = np.zeros(2, dtype=np.float64)
    cd_friction = float("nan")
    if ibl is not None and mesh["ibl_1d"] is not None:
        stations = mesh["ibl_1d"]
        friction = _segment_traction(
            ibl["cf_upper"],
            stations["tangent_upper"],
            stations["ds_upper"],
            ibl["ue_upper"],
            _UNIT_FREESTREAM,
        )
        friction = friction + _segment_traction(
            ibl["cf_lower"],
            stations["tangent_lower"],
            stations["ds_lower"],
            ibl["ue_lower"],
            _UNIT_FREESTREAM,
        )
    drag_hat = np.array([np.cos(alpha_rad), np.sin(alpha_rad)], dtype=np.float64)
    lift_hat = np.array([-np.sin(alpha_rad), np.cos(alpha_rad)], dtype=np.float64)
    chord = float(panel["chord"])
    cd_pressure = float(np.dot(pressure, drag_hat) / chord)
    cl = float(np.dot(pressure + friction, lift_hat) / chord)
    if ibl is not None:
        cd_friction = float(np.dot(friction, drag_hat) / chord)
        cd = cd_pressure + cd_friction
    else:
        cd = cd_pressure
    nodes = np.asarray(panel["nodes"], dtype=np.float64)
    le = nodes[int(panel["le_index"])]
    origin_qc = np.array([float(le[0]) + 0.25 * chord, float(le[1])], dtype=np.float64)
    origin_c = np.average(collocation, axis=0, weights=length)
    total = pressure + (friction if ibl is not None else 0.0)

    def moment(origin: np.ndarray) -> float:
        arm = collocation - origin
        panel_moment = np.sum(arm[:, 0] * (-cp * normal[:, 1] * length) - arm[:, 1] * (-cp * normal[:, 0] * length))
        # Friction is already a net force; its moment uses station midpoints.
        return float(panel_moment / chord**2)

    gamma = inviscid["gamma"]
    gamma_total = float(np.sum(np.asarray(gamma) * length)) if gamma is not None else float("nan")
    return SurfaceLoads(
        gamma_total=gamma_total,
        cl=cl,
        cd_pressure=cd_pressure,
        cd_friction=cd_friction,
        cd=float(cd),
        cm_quarter_chord=moment(origin_qc),
        cm_centroid=moment(origin_c),
        moment_origin_qc=np.ascontiguousarray(origin_qc, dtype=np.float64),
        moment_origin_centroid=np.ascontiguousarray(origin_c, dtype=np.float64),
        s_ref=chord,
        compressibility_applied=bool(compressibility_applied),
    )


def _lofted_planform_area(facets: np.ndarray) -> Optional[float]:
    """Projected planform of a spanwise loft, or ``None`` when it is not one.

    Wing sections from the geometry generator share a y coordinate. Sweep
    adds the same ``|y| tan Λ`` to every point on a section, so the
    section's own x-extent is the local chord and does not grow with
    sweep. The planform is the trapezoidal integral of that chord along y.

    A station needs at least four vertices and a positive x-extent. An
    open surface that is not a loft (a patch, a cap, an unstructured
    cloud) returns ``None``. The caller then keeps the centroid-box area
    and does not invent a wing chord.
    """

    verts = np.asarray(facets, dtype=np.float64).reshape(-1, 3)
    if verts.shape[0] < 8:
        return None
    y = verts[:, 1]
    span = float(np.ptp(y))
    if not np.isfinite(span) or span <= 1e-12:
        return None
    tol = max(span * 1e-8, 1e-12)
    order = np.argsort(y, kind="mergesort")
    y_sorted = y[order]
    x_sorted = np.asarray(verts[order, 0], dtype=np.float64)
    jumps = np.diff(y_sorted) > tol
    station = np.cumsum(np.concatenate((np.zeros(1, dtype=np.int64), jumps.astype(np.int64))))
    n_stations = int(station[-1]) + 1
    if n_stations < 2:
        return None
    counts = np.bincount(station)
    if counts.shape[0] != n_stations or np.any(counts < 4):
        return None
    x_min = np.full(n_stations, np.inf, dtype=np.float64)
    x_max = np.full(n_stations, -np.inf, dtype=np.float64)
    np.minimum.at(x_min, station, x_sorted)
    np.maximum.at(x_max, station, x_sorted)
    chord = x_max - x_min
    if not np.all(np.isfinite(chord)) or np.any(chord <= span * 1e-6):
        return None
    y_station = np.bincount(station, weights=y_sorted) / counts
    dy = np.diff(y_station)
    if np.any(dy <= 0.0):
        return None
    area = float(np.sum(0.5 * (chord[:-1] + chord[1:]) * dy))
    if not np.isfinite(area) or area <= 0.0:
        return None
    return area


def _reference_area_3d(mesh: MeshOutput) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Planform (open lofted wing) or projected cross-section (closed body).

    Returns ``s_ref``, reference length, quarter-chord origin, and the
    area-weighted centroid.

    Closed bodies keep the product of the two largest centroid extents
    (projected cross-section). Open lofted wings use
    ``_lofted_planform_area``. Any other open surface keeps the product of
    the centroid x-extent and y-extent. That fallback still includes a
    sweep offset and stops short of the tips; it is used only when the
    vertices do not form chord-bearing stations.
    """

    panel = mesh["panel_3d"]
    assert panel is not None
    centroids = np.asarray(panel["centroids"], dtype=np.float64)
    areas = np.asarray(panel["areas"], dtype=np.float64)
    centroid = np.average(centroids, axis=0, weights=areas)
    if panel["is_closed"]:
        extents = np.ptp(centroids, axis=0)
        order = np.argsort(extents)
        length = float(extents[order[-1]])
        diameter = float(extents[order[-2]])
        return diameter * length, length, centroid.copy(), centroid
    span = float(np.ptp(centroids[:, 1]))
    y_root = 0.5 * (float(np.min(centroids[:, 1])) + float(np.max(centroids[:, 1])))
    window = max(span / 20.0, 1e-8)
    near = np.abs(centroids[:, 1] - y_root) <= window
    if not np.any(near):
        near = np.ones(centroids.shape[0], dtype=bool)
    root = centroids[near]
    x_le = float(np.min(root[:, 0]))
    z_le = float(root[np.argmin(root[:, 0]), 2])
    root_chord = float(np.ptp(root[:, 0]))
    if root_chord <= 0.0:
        root_chord = float(np.ptp(centroids[:, 0]))
    origin = np.array([x_le + 0.25 * root_chord, y_root, z_le], dtype=np.float64)
    planform = _lofted_planform_area(panel["facets"])
    if planform is None:
        planform = float(np.ptp(centroids[:, 0]) * max(span, 1e-12))
    return planform, root_chord, origin, centroid


def _loads_3d(
    mesh: MeshOutput,
    inviscid: InviscidSolution,
    ibl: Optional[IBLSolution],
    alpha_rad: float,
    compressibility_applied: bool,
) -> SurfaceLoads:
    panel = mesh["panel_3d"]
    assert panel is not None
    normal = np.asarray(panel["normals"], dtype=np.float64)
    areas = np.asarray(panel["areas"], dtype=np.float64)
    centroids = np.asarray(panel["centroids"], dtype=np.float64)
    cp = np.asarray(inviscid["cp_inviscid"], dtype=np.float64)
    pressure = np.sum(-cp[:, None] * normal * areas[:, None], axis=0)
    friction = np.zeros(3, dtype=np.float64)
    cd_friction = float("nan")
    s_ref, ref_length, origin_qc, origin_c = _reference_area_3d(mesh)
    drag_hat = np.array([np.cos(alpha_rad), 0.0, np.sin(alpha_rad)], dtype=np.float64)
    lift_hat = np.array([-np.sin(alpha_rad), 0.0, np.cos(alpha_rad)], dtype=np.float64)
    cd_pressure = float(np.dot(pressure, drag_hat) / s_ref)
    if ibl is not None:
        cd_friction = float(np.dot(friction, drag_hat) / s_ref)
        cd = cd_pressure + cd_friction
    else:
        cd = cd_pressure
    cl = float(np.dot(pressure + friction, lift_hat) / s_ref)
    divisor = s_ref * max(ref_length, 1e-12)

    def moment_about(origin: np.ndarray) -> float:
        arm = centroids - origin
        element = -cp[:, None] * normal * areas[:, None]
        torque = np.sum(np.cross(arm, element), axis=0)
        return float(torque[1] / divisor)

    return SurfaceLoads(
        gamma_total=float("nan"),
        cl=cl,
        cd_pressure=cd_pressure,
        cd_friction=cd_friction,
        cd=float(cd),
        cm_quarter_chord=moment_about(origin_qc),
        cm_centroid=moment_about(origin_c),
        moment_origin_qc=np.ascontiguousarray(origin_qc, dtype=np.float64),
        moment_origin_centroid=np.ascontiguousarray(origin_c, dtype=np.float64),
        s_ref=float(s_ref),
        compressibility_applied=bool(compressibility_applied),
    )


def integrate_loads(
    mesh: MeshOutput,
    inviscid: InviscidSolution,
    ibl: Optional[IBLSolution],
    alpha_rad: float,
    compressibility_applied: bool,
) -> SurfaceLoads:
    """Pressure and friction resultant in the freestream frame."""

    if mesh["mode"] == "2D":
        return _loads_2d(mesh, inviscid, ibl, alpha_rad, compressibility_applied)
    return _loads_3d(mesh, inviscid, ibl, alpha_rad, compressibility_applied)
