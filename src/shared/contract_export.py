"""
Contract export for the live pipeline stages.

Turns a ``GeometryOutput`` / ``MeshOutput`` pair into two assessable
artifacts:

* ``contract_summary`` — JSON-safe scalars, array shapes, and range stats.
* ``contract_arrays``  — every raw array, flattened into one ``.npz`` payload.

Downstream modules append their own section here; they must not fork a
second exporter.
"""

from __future__ import annotations

import io
import json
from typing import Any, Dict, Optional

import numpy as np

# Names stay in annotations only. Importing the mesh package here would
# re-enter math_skeleton while it is still loading geo and this file.
# Type names are annotations only so this module can load while the mesh
# package is still initializing.


def _region_names() -> Dict[int, str]:
    from math_module.math_skeleton import BoundaryRegion

    return {int(region): region.name for region in BoundaryRegion}


def _array_stats(array: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(array, dtype=np.float64)
    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _geometry_section(geometry: GeometryOutput) -> Dict[str, Any]:
    payload = geometry["points_2d"] if geometry["mode"] == "2D" else geometry["facets_3d"]
    assert payload is not None
    return {
        "mode": geometry["mode"],
        "shape_name": geometry["shape_name"],
        "num_elements": int(geometry["num_elements"]),
        "is_closed": bool(geometry["is_closed"]),
        "bounding_box": {key: float(value) for key, value in geometry["bounding_box"].items()},
        "payload": {
            "field": "points_2d" if geometry["mode"] == "2D" else "facets_3d",
            **_array_stats(payload),
        },
    }


def _panel_2d_section(panel: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "n_panels": int(panel["length"].size),
        "chord": float(panel["chord"]),
        "clustering": panel["clustering"],
        "le_index": int(panel["le_index"]),
        "te_index": int(panel["te_index"]),
        "te_first": bool(panel["te_first"]),
        "is_closed": bool(panel["is_closed"]),
        "arrays": {
            name: _array_stats(panel[name])
            for name in (
                "nodes",
                "collocation",
                "tangent",
                "normal",
                "length",
                "angle_beta",
                "curvature",
                "x_over_c",
                "arc_length",
            )
        },
    }


def _cartesian_section(grid: Dict[str, Any]) -> Dict[str, Any]:
    mask = np.asarray(grid["boundary_mask"])
    tags, counts = np.unique(mask, return_counts=True)
    names = _region_names()
    return {
        "n_x": int(grid["n_x"]),
        "n_y": int(grid["n_y"]),
        "x_range": [float(grid["x_nodes"][0]), float(grid["x_nodes"][-1])],
        "y_range": [float(grid["y_nodes"][0]), float(grid["y_nodes"][-1])],
        "boundary_cells": {
            names.get(int(tag), str(int(tag))): int(count)
            for tag, count in zip(tags.tolist(), counts.tolist())
        },
    }


def _ibl_section(ibl: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "n_upper_stations": int(ibl["s_upper"].size),
        "n_lower_stations": int(ibl["s_lower"].size),
        "s_upper_max": float(ibl["s_upper"][-1]),
        "s_lower_max": float(ibl["s_lower"][-1]),
        "stagnation_xy": [float(value) for value in ibl["stagnation_xy"]],
        "min_ds": float(min(np.min(ibl["ds_upper"]), np.min(ibl["ds_lower"]))),
    }


def _panel_3d_section(panel: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "n_panels": int(panel["areas"].size),
        "is_closed": bool(panel["is_closed"]),
        "normal_orientation": panel["normal_orientation"],
        "total_area": float(np.sum(panel["areas"])),
        "arrays": {
            name: _array_stats(panel[name])
            for name in (
                "facets",
                "centroids",
                "normals",
                "tangent_1",
                "tangent_2",
                "areas",
                "transforms",
                "inverse_transforms",
            )
        },
    }


def _mesh_section(mesh: MeshOutput) -> Dict[str, Any]:
    section: Dict[str, Any] = {
        "mode": mesh["mode"],
        "shape_name": mesh["shape_name"],
        "source_geometry_closed": bool(mesh["source_geometry_closed"]),
        "quality": {
            key: (float(value) if isinstance(value, float) else int(value))
            for key, value in mesh["quality"].items()
        },
    }
    if mesh["panel_2d"] is not None:
        section["panel_2d"] = _panel_2d_section(mesh["panel_2d"])
    if mesh["cartesian_2d"] is not None:
        section["cartesian_2d"] = _cartesian_section(mesh["cartesian_2d"])
    if mesh["ibl_1d"] is not None:
        section["ibl_1d"] = _ibl_section(mesh["ibl_1d"])
    if mesh["panel_3d"] is not None:
        section["panel_3d"] = _panel_3d_section(mesh["panel_3d"])
    return section


def _physics_section(physics: PhysicsOutput) -> Dict[str, Any]:
    cp = np.asarray(physics["inviscid"]["cp_inviscid"])
    field = physics["field_2d"]
    return {
        "mode": physics["mode"],
        "regime": physics["regime"],
        "converged": bool(physics["converged"]),
        "vic_iterations": int(physics["vic_iterations"]),
        "residual": float(physics["residual"]),
        "cl": float(physics["loads"]["cl"]),
        "cd": float(physics["loads"]["cd"]),
        "cp_shape": [int(cp.shape[0])],
        "field_shape": None if field is None else [int(field["u"].shape[0]), int(field["u"].shape[1])],
        "warnings": list(physics["warnings"]),
    }


def _aero_section(aero: AeroOutput) -> Dict[str, Any]:
    coefficients = aero["coefficients"]
    return {
        "mode": aero["mode"],
        "alpha_deg": aero["alpha_deg"],
        "cl": coefficients["cl"],
        "cd": coefficients["cd"],
        "cd_pressure": coefficients["cd_pressure"],
        "cd_friction": coefficients["cd_friction"],
        "cd_induced": coefficients["cd_induced"],
        "cm": coefficients["cm"],
        "l_over_d": aero["l_over_d"],
        "center_of_pressure_x": coefficients["center_of_pressure_x"],
        "lift": aero["dimensional"]["lift"],
        "drag": aero["dimensional"]["drag"],
        "warnings": list(aero["warnings"]),
    }


def contract_summary(
    geometry: Optional[GeometryOutput],
    mesh: Optional[MeshOutput],
    *,
    label: str = "run",
    notes: Optional[Dict[str, Any]] = None,
    physics: Optional[PhysicsOutput] = None,
    aero: Optional[AeroOutput] = None,
) -> Dict[str, Any]:
    """JSON-safe description of every live stage contract in one run."""

    pending = ["physics", "aero", "ai"]
    if physics is not None:
        pending = ["aero", "ai"]
    if aero is not None:
        pending = ["ai"]
    summary: Dict[str, Any] = {
        "label": label,
        "live_stages": [
            name
            for name, present in (
                ("geo", geometry is not None),
                ("math", mesh is not None),
                ("physics", physics is not None),
                ("aero", aero is not None),
            )
            if present
        ],
        "pending_stages": pending,
    }
    if notes:
        summary["notes"] = json.loads(json.dumps(notes, default=str))
    if geometry is not None:
        summary["geo"] = _geometry_section(geometry)
    if mesh is not None:
        summary["math"] = _mesh_section(mesh)
    if physics is not None:
        summary["physics"] = _physics_section(physics)
    if aero is not None:
        summary["aero"] = _aero_section(aero)
    return summary


def contract_summary_json(
    geometry: Optional[GeometryOutput],
    mesh: Optional[MeshOutput],
    *,
    label: str = "run",
    notes: Optional[Dict[str, Any]] = None,
    physics: Optional[PhysicsOutput] = None,
    aero: Optional[AeroOutput] = None,
) -> str:
    return json.dumps(
        contract_summary(
            geometry,
            mesh,
            label=label,
            notes=notes,
            physics=physics,
            aero=aero,
        ),
        indent=2,
        sort_keys=False,
    )


def contract_arrays(
    geometry: Optional[GeometryOutput],
    mesh: Optional[MeshOutput],
    physics: Optional[PhysicsOutput] = None,
    aero: Optional[AeroOutput] = None,
) -> Dict[str, np.ndarray]:
    """Every raw contract array, namespaced ``<stage>_<block>_<field>``."""

    arrays: Dict[str, np.ndarray] = {}

    if geometry is not None:
        if geometry["points_2d"] is not None:
            arrays["geo_points_2d"] = np.asarray(geometry["points_2d"], dtype=np.float64)
        if geometry["facets_3d"] is not None:
            arrays["geo_facets_3d"] = np.asarray(geometry["facets_3d"], dtype=np.float64)

    if mesh is not None:
        for block in ("panel_2d", "cartesian_2d", "ibl_1d", "panel_3d"):
            payload = mesh[block]  # type: ignore[literal-required]
            if payload is None:
                continue
            for field, value in payload.items():
                if isinstance(value, np.ndarray):
                    arrays[f"math_{block}_{field}"] = value

    if physics is not None:
        inviscid = physics["inviscid"]
        for field in ("gamma", "mu", "sigma", "cp_inviscid", "v_tangential"):
            value = inviscid[field]  # type: ignore[literal-required]
            if isinstance(value, np.ndarray):
                arrays[f"physics_inviscid_{field}"] = value
        if isinstance(physics["surface_velocity"], np.ndarray):
            arrays["physics_surface_velocity"] = physics["surface_velocity"]
        field_2d = physics["field_2d"]
        if field_2d is not None:
            for name in ("u", "v", "w", "speed", "cp", "phi", "psi"):
                arrays[f"physics_field_{name}"] = np.asarray(field_2d[name])  # type: ignore[literal-required]

    if aero is not None:
        arrays["aero_cp_surface"] = np.asarray(aero["cp_surface"], dtype=np.float64)
        if aero["x_over_c"] is not None:
            arrays["aero_x_over_c"] = np.asarray(aero["x_over_c"], dtype=np.float64)
        spanwise = aero["spanwise"]
        if spanwise is not None:
            arrays["aero_spanwise_y"] = np.asarray(spanwise["y"], dtype=np.float64)
            arrays["aero_spanwise_cl"] = np.asarray(spanwise["cl"], dtype=np.float64)

    return arrays


def contract_npz_bytes(
    geometry: Optional[GeometryOutput],
    mesh: Optional[MeshOutput],
    physics: Optional[PhysicsOutput] = None,
    aero: Optional[AeroOutput] = None,
) -> bytes:
    """Compressed ``.npz`` of every contract array, ready for download."""

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **contract_arrays(geometry, mesh, physics, aero))
    return buffer.getvalue()
