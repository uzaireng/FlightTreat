"""Panel pressure and skin-friction integration to surface loads."""

from __future__ import annotations

from typing import Optional

import numpy as np

from math_module.math_skeleton import MeshOutput

from .physics_skeleton import IBLSolution, InviscidSolution, SurfaceLoads


def _segment_traction(
    cf: np.ndarray,
    tangent: np.ndarray,
    ds: np.ndarray,
) -> np.ndarray:
    """Skin-friction force from station ``Cf`` along a marching branch."""

    if ds.size == 0:
        return np.zeros(tangent.shape[1], dtype=np.float64)
    cf_seg = 0.5 * (cf[:-1] + cf[1:])
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
            ibl["cf_upper"], stations["tangent_upper"], stations["ds_upper"]
        )
        friction = friction + _segment_traction(
            ibl["cf_lower"], stations["tangent_lower"], stations["ds_lower"]
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


def _reference_area_3d(mesh: MeshOutput) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Planform (open wing) or projected cross-section (closed body).

    Returns ``s_ref``, reference length, quarter-chord origin, and the
    area-weighted centroid.
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
    s_ref = float(np.ptp(centroids[:, 0]) * max(span, 1e-12))
    return s_ref, root_chord, origin, centroid


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
