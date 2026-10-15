"""Interactive workbench chaining Module 1 (geometry) into Module 2 (mesh)."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pydantic import ValidationError

from geo_module import GeometryEngine, GeometryEngineError, GeometryOutput
from math_module import BoundaryRegion, MeshEngine, MeshEngineError, MeshOutput
from physics_module import PhysicsEngine, PhysicsEngineError
from physics_module.physics_engine import split_flow_payload
from shared.contract_export import contract_npz_bytes, contract_summary_json
from shared.dataset_catalog import (
    DatasetCatalogError,
    list_geo_module_samples,
    list_math_module_samples,
    list_pipeline_cases,
    pending_pipeline_stages,
    spec_as_dict,
)
from shared.pipeline_runner import run_math_module_sample
from streamlit_ui.geo_workbench import (
    PLOT_HEIGHT_2D,
    PLOTLY_2D_CONFIG,
    _bbox_caption,
    apply_2d_plot_layout,
    figure_2d,
    figure_3d,
    section_axis_ranges,
    section_view_id,
)

_REGION_NAME = {int(item): item.name for item in BoundaryRegion}


def _quiver(origin: np.ndarray, direction: np.ndarray, scale: float) -> tuple[np.ndarray, ...]:
    blank = np.full((origin.shape[0], 1), np.nan)
    axes = []
    for column in range(origin.shape[1]):
        segment = np.concatenate(
            (
                origin[:, column : column + 1],
                origin[:, column : column + 1] + scale * direction[:, column : column + 1],
                blank,
            ),
            axis=1,
        )
        axes.append(segment.ravel())
    return tuple(axes)


def _show_2d_chart(fig: go.Figure, key: str) -> None:
    """Stretch the chart and keep its identity across mesh-slider reruns.

    A key that includes panel counts or the farfield box remounts the
    Plotly element on every slider tick and throws away pan and zoom.
    """

    st.plotly_chart(
        fig,
        width="stretch",
        height=PLOT_HEIGHT_2D,
        config=PLOTLY_2D_CONFIG,
        key=key,
    )


def _farfield_caption(mesh: MeshOutput) -> str:
    grid = mesh["cartesian_2d"]
    if grid is None:
        return ""
    x_nodes = grid["x_nodes"]
    y_nodes = grid["y_nodes"]
    return (
        f"farfield {int(grid['n_x'])}×{int(grid['n_y'])} · "
        f"x[{float(x_nodes[0]):.4g}, {float(x_nodes[-1]):.4g}] · "
        f"y[{float(y_nodes[0]):.4g}, {float(y_nodes[-1]):.4g}]"
    )


def _farfield_polylines(x_nodes: np.ndarray, y_nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vertical and horizontal grid lines, separated by NaN breaks."""

    x_line = np.asarray(x_nodes, dtype=np.float64)
    y_line = np.asarray(y_nodes, dtype=np.float64)
    vertical_x = np.repeat(x_line, 3)
    vertical_y = np.tile(np.array([y_line[0], y_line[-1], np.nan], dtype=np.float64), x_line.size)
    horizontal_y = np.repeat(y_line, 3)
    horizontal_x = np.tile(np.array([x_line[0], x_line[-1], np.nan], dtype=np.float64), y_line.size)
    return (
        np.concatenate((vertical_x, horizontal_x)),
        np.concatenate((vertical_y, horizontal_y)),
    )


def figure_panels_2d(geometry: GeometryOutput, mesh: MeshOutput) -> go.Figure:
    panel = mesh["panel_2d"]
    assert panel is not None
    nodes = panel["nodes"]
    length = panel["length"]
    chord = float(panel["chord"])
    step = max(1, int(length.size) // 80)
    qx, qy = _quiver(panel["collocation"][::step], panel["normal"][::step], 0.05 * chord)

    fig = go.Figure()
    grid = mesh["cartesian_2d"]
    if grid is not None:
        x_nodes = grid["x_nodes"]
        y_nodes = grid["y_nodes"]
        grid_x, grid_y = _farfield_polylines(x_nodes, y_nodes)
        fig.add_trace(
            go.Scatter(
                x=grid_x,
                y=grid_y,
                mode="lines",
                name="Farfield grid",
                line={"width": 0.6, "color": "rgba(127, 140, 141, 0.45)"},
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[x_nodes[0], x_nodes[-1], x_nodes[-1], x_nodes[0], x_nodes[0]],
                y=[y_nodes[0], y_nodes[0], y_nodes[-1], y_nodes[-1], y_nodes[0]],
                mode="lines",
                name="Farfield box",
                line={"width": 2, "color": "#1a5276"},
            )
        )
    points = geometry["points_2d"]
    if points is not None:
        fig.add_trace(
            go.Scatter(
                x=points[:, 0],
                y=points[:, 1],
                mode="lines",
                name="Geometry",
                line={"width": 1, "color": "#95a5a6"},
            )
        )
    fig.add_trace(
        go.Scatter(
            x=nodes[:, 0],
            y=nodes[:, 1],
            mode="lines+markers",
            name="Clustered panels",
            marker={
                "size": 5,
                "color": np.concatenate((length, length[:1])),
                "colorscale": "Viridis",
                "showscale": True,
                "colorbar": {"title": "S"},
            },
            line={"width": 2, "color": "#1f4e79"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=qx,
            y=qy,
            mode="lines",
            name="Outward normals",
            line={"width": 1, "color": "#c0392b"},
        )
    )
    fig.update_layout(
        title=(
            f"2D panels — {mesh['shape_name']} ({panel['clustering']})"
            + (f" · {_farfield_caption(mesh)}" if grid is not None else "")
        ),
        xaxis_title="x",
        yaxis_title="y",
        template="plotly_white",
    )
    focus = points if points is not None else nodes
    x_range, y_range = section_axis_ranges(focus[:, 0], focus[:, 1])
    apply_2d_plot_layout(
        fig,
        x_range=x_range,
        y_range=y_range,
        view_id=section_view_id(str(mesh["shape_name"]), focus[:, 0], focus[:, 1]),
    )
    return fig


def figure_farfield(mesh: MeshOutput) -> go.Figure:
    grid = mesh["cartesian_2d"]
    panel = mesh["panel_2d"]
    assert grid is not None and panel is not None
    mask = grid["boundary_mask"]
    fig = go.Figure(
        data=[
            go.Heatmap(
                x=grid["x_nodes"],
                y=grid["y_nodes"],
                z=mask,
                zmin=0,
                zmax=6,
                colorscale=[
                    [0.0, "#d6eaf8"],
                    [0.14, "#d6eaf8"],
                    [0.14, "#1e8449"],
                    [0.28, "#1e8449"],
                    [0.28, "#d35400"],
                    [0.42, "#d35400"],
                    [0.42, "#7f8c8d"],
                    [0.57, "#7f8c8d"],
                    [0.57, "#b0b8bc"],
                    [0.71, "#b0b8bc"],
                    [0.71, "#c0392b"],
                    [0.85, "#c0392b"],
                    [0.85, "#1c2833"],
                    [1.0, "#1c2833"],
                ],
                colorbar={
                    "tickmode": "array",
                    "tickvals": list(range(7)),
                    "ticktext": [_REGION_NAME[i] for i in range(7)],
                },
                hovertemplate="x=%{x:.4f}<br>y=%{y:.4f}<br>tag=%{z}<extra></extra>",
            )
        ]
    )
    nodes = panel["nodes"]
    fig.add_trace(
        go.Scatter(
            x=nodes[:, 0],
            y=nodes[:, 1],
            mode="lines",
            name="Panels",
            line={"width": 2, "color": "#f7dc6f"},
        )
    )
    fig.update_layout(
        title=f"Cartesian farfield mask · {_farfield_caption(mesh)}",
        xaxis_title="x",
        yaxis_title="y",
        template="plotly_white",
    )
    # The mask chart shows the whole tagged box. Its camera id still follows
    # the section, so panel sliders do not snap a zoom the user already set.
    apply_2d_plot_layout(
        fig,
        x_range=[float(grid["x_nodes"][0]), float(grid["x_nodes"][-1])],
        y_range=[float(grid["y_nodes"][0]), float(grid["y_nodes"][-1])],
        view_id=section_view_id(str(mesh["shape_name"]), nodes[:, 0], nodes[:, 1]),
    )
    return fig


def figure_ibl(mesh: MeshOutput) -> go.Figure:
    ibl = mesh["ibl_1d"]
    assert ibl is not None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ibl["xy_upper"][:, 0],
            y=ibl["xy_upper"][:, 1],
            mode="lines+markers",
            name="Upper",
            line={"color": "#1a5276", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ibl["xy_lower"][:, 0],
            y=ibl["xy_lower"][:, 1],
            mode="lines+markers",
            name="Lower",
            line={"color": "#b9770e", "width": 2},
        )
    )
    stag = ibl["stagnation_xy"]
    fig.add_trace(
        go.Scatter(
            x=[float(stag[0])],
            y=[float(stag[1])],
            mode="markers",
            name="LE / stagnation",
            marker={"size": 10, "color": "#c0392b"},
        )
    )
    fig.update_layout(
        title="1D IBL stations",
        xaxis_title="x",
        yaxis_title="y",
        template="plotly_white",
    )
    station_x = np.concatenate((ibl["xy_upper"][:, 0], ibl["xy_lower"][:, 0]))
    station_y = np.concatenate((ibl["xy_upper"][:, 1], ibl["xy_lower"][:, 1]))
    x_range, y_range = section_axis_ranges(station_x, station_y)
    apply_2d_plot_layout(
        fig,
        x_range=x_range,
        y_range=y_range,
        view_id=section_view_id(str(mesh["shape_name"]), station_x, station_y),
    )
    return fig


def figure_normals_3d(mesh: MeshOutput) -> go.Figure:
    panel = mesh["panel_3d"]
    assert panel is not None
    facets = panel["facets"]
    shown = facets if facets.shape[0] <= 25_000 else facets[:25_000]
    vertices = shown.reshape(-1, 3)
    tri = shown.shape[0]
    index = np.arange(0, tri * 3, 3)
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=index,
                j=index + 1,
                k=index + 2,
                opacity=0.55,
                color="#5d6d7e",
                name="Surface",
                showscale=False,
            )
        ]
    )
    step = max(1, int(panel["areas"].shape[0]) // 500)
    scale = 0.08 * float(
        np.linalg.norm(
            np.array(
                [
                    np.ptp(facets[:, :, 0]),
                    np.ptp(facets[:, :, 1]),
                    np.ptp(facets[:, :, 2]),
                ]
            )
        )
    )
    qx, qy, qz = _quiver(
        panel["centroids"][::step],
        panel["normals"][::step],
        max(scale, 1e-6),
    )
    fig.add_trace(
        go.Scatter3d(
            x=qx,
            y=qy,
            z=qz,
            mode="lines",
            name="Unit normals",
            line={"width": 3, "color": "#c0392b"},
        )
    )
    fig.update_layout(
        title=f"3D panels — {mesh['shape_name']}",
        scene={"aspectmode": "data", "xaxis_title": "x", "yaxis_title": "y", "zaxis_title": "z"},
        template="plotly_white",
        height=700,
    )
    return fig


def _mask_table(mesh: MeshOutput) -> Optional[pd.DataFrame]:
    grid = mesh["cartesian_2d"]
    if grid is None:
        return None
    values, counts = np.unique(grid["boundary_mask"], return_counts=True)
    return pd.DataFrame(
        {
            "region": [_REGION_NAME.get(int(value), str(int(value))) for value in values],
            "cells": counts.astype(int),
        }
    )


def _quality_frame(mesh: MeshOutput) -> pd.DataFrame:
    quality = mesh["quality"]
    return pd.DataFrame(
        {
            "metric": list(quality.keys()),
            "value": [quality[key] for key in quality],
        }
    )


_CLUSTERING_CHOICES = ("cosine", "half_cosine", "uniform")
_ORIENTATION_CHOICES = ("outward", "inward")

_MESH_DEFAULTS: Dict[str, Any] = {
    "n_panels": 120,
    "n_x": 97,
    "n_y": 49,
    "clustering": "cosine",
    "le_cluster_factor": 1.0,
    "te_cluster_factor": 1.0,
    "normal_orientation": "outward",
    "require_closed_3d": False,
    "surface_density": 1,
    "domain": {"x_min": -2.0, "x_max": 3.0, "y_min": -2.5, "y_max": 2.5},
}

_DENSITY_CHOICES = (1, 2, 4)


def _clamp(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


def _unless_set(key: str, **initial: Any) -> Dict[str, Any]:
    """Pass a widget default only on its first draw.

    Later reruns must keep the value the user just set. Re-sending ``value``
    snaps farfield counts and the chord-relative box back to the seed.
    """

    if key in st.session_state:
        return {}
    return initial


def _on_step(value: int, low: int, high: int, step: int) -> int:
    snapped = int(low + step * round((int(value) - low) / float(step)))
    return int(min(max(snapped, low), high))


def _seeded_slider(
    label: str,
    low: int,
    high: int,
    seed: int,
    key: str,
    *,
    step: int = 1,
) -> int:
    """Integer slider that keeps the user's value after the first draw."""

    planted = _on_step(seed, low, high, step)
    return int(
        st.sidebar.slider(
            label,
            min_value=int(low),
            max_value=int(high),
            step=int(step),
            key=key,
            **_unless_set(key, value=planted),
        )
    )


def _mesh_knobs(
    *,
    scope: str,
    defaults: Optional[Dict[str, Any]] = None,
    mode: Optional[str] = None,
    geo_spec: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Mesh sliders seeded from ``defaults``, plus geometry resolution overrides.

    ``scope`` namespaces the widget keys. It must change whenever ``defaults``
    change, otherwise Streamlit keeps the previous selection's values.
    Geometry overrides are not mesh-contract fields. The caller rebuilds
    ``GeometryOutput`` with them before ``MeshEngine`` runs.
    """

    seed = _control_seed(defaults, mode)
    show_2d = "n_panels" in seed
    show_3d = "normal_orientation" in seed

    st.sidebar.markdown("**Mesh settings**")
    st.sidebar.caption("Every change rebuilds the mesh on the current shape.")
    knobs: Dict[str, Any] = {}
    geo_overrides: Dict[str, Any] = {}
    family = str((geo_spec or {}).get("type") or "")

    if show_2d:
        knobs["clustering"] = st.sidebar.selectbox(
            "Clustering",
            _CLUSTERING_CHOICES,
            index=_CLUSTERING_CHOICES.index(seed["clustering"]),
            key=f"{scope}_clustering",
        )
        knobs["n_panels"] = int(
            st.sidebar.slider(
                "2D panels",
                16,
                400,
                int(seed["n_panels"]),
                step=1,
                key=f"{scope}_n_panels",
            )
        )
        knobs["le_cluster_factor"] = float(
            st.sidebar.slider(
                "LE cluster factor",
                0.0,
                20.0,
                float(seed["le_cluster_factor"]),
                step=0.5,
                key=f"{scope}_le",
            )
        )
        knobs["te_cluster_factor"] = float(
            st.sidebar.slider(
                "TE cluster factor",
                0.0,
                20.0,
                float(seed["te_cluster_factor"]),
                step=0.5,
                key=f"{scope}_te",
            )
        )
        # step=1 so a stored case seeds these exactly; a coarser grid would
        # snap the seed and silently re-mesh a case the user never touched.
        nx_key = f"{scope}_nx"
        ny_key = f"{scope}_ny"
        knobs["n_x"] = int(
            st.sidebar.slider(
                "Farfield n_x",
                min_value=16,
                max_value=241,
                step=1,
                key=nx_key,
                **_unless_set(nx_key, value=int(seed["n_x"])),
            )
        )
        knobs["n_y"] = int(
            st.sidebar.slider(
                "Farfield n_y",
                min_value=16,
                max_value=161,
                step=1,
                key=ny_key,
                **_unless_set(ny_key, value=int(seed["n_y"])),
            )
        )

    if show_3d:
        spec = geo_spec or {}
        if family == "wing":
            st.sidebar.caption(
                "Spanwise sections and points per section rebuild the wing. "
                "Panel density then splits those triangles."
            )
            geo_overrides["num_sections"] = _seeded_slider(
                "Spanwise sections",
                2,
                80,
                int(spec.get("num_sections", 16)),
                f"{scope}_sections",
            )
            geo_overrides["airfoil_points"] = _seeded_slider(
                "Points per section",
                20,
                160,
                int(spec.get("airfoil_points", 100)),
                f"{scope}_airfoil_points",
                step=4,
            )
        elif family == "cylinder":
            st.sidebar.caption(
                "Segments and rings rebuild the body. Panel density then splits those triangles."
            )
            geo_overrides["segments"] = _seeded_slider(
                "Circumferential segments",
                8,
                128,
                int(spec.get("segments", 32)),
                f"{scope}_segments",
                step=4,
            )
            geo_overrides["rings"] = _seeded_slider(
                "Axial rings",
                1,
                48,
                int(spec.get("rings", 8)),
                f"{scope}_rings",
            )
        else:
            st.sidebar.caption(
                "Panel density splits the loaded triangles. 1 keeps the file or fixture as meshed."
            )
        density_key = f"{scope}_density"
        knobs["surface_density"] = int(
            st.sidebar.select_slider(
                "Surface panel density",
                options=list(_DENSITY_CHOICES),
                key=density_key,
                help="1 keeps each triangle. 2 halves every edge. 4 halves them again.",
                **_unless_set(density_key, value=int(seed["surface_density"])),
            )
        )
        knobs["normal_orientation"] = st.sidebar.selectbox(
            "3D normal orientation",
            _ORIENTATION_CHOICES,
            index=_ORIENTATION_CHOICES.index(seed["normal_orientation"]),
            key=f"{scope}_orientation",
        )
        knobs["require_closed_3d"] = bool(
            st.sidebar.checkbox(
                "Require closed 3D body",
                value=bool(seed["require_closed_3d"]),
                key=f"{scope}_closed",
            )
        )

    if show_2d:
        domain = seed["domain"]
        with st.sidebar.expander("Chord-relative farfield box", expanded=True):
            xmin_key = f"{scope}_xmin"
            xmax_key = f"{scope}_xmax"
            ymin_key = f"{scope}_ymin"
            ymax_key = f"{scope}_ymax"
            knobs["domain"] = {
                "x_min": float(
                    st.number_input(
                        "x_min / chord from LE",
                        step=0.1,
                        format="%.2f",
                        key=xmin_key,
                        **_unless_set(xmin_key, value=float(domain["x_min"])),
                    )
                ),
                "x_max": float(
                    st.number_input(
                        "x_max / chord from LE",
                        step=0.1,
                        format="%.2f",
                        key=xmax_key,
                        **_unless_set(xmax_key, value=float(domain["x_max"])),
                    )
                ),
                "y_min": float(
                    st.number_input(
                        "y_min / chord from mid-height",
                        step=0.1,
                        format="%.2f",
                        key=ymin_key,
                        **_unless_set(ymin_key, value=float(domain["y_min"])),
                    )
                ),
                "y_max": float(
                    st.number_input(
                        "y_max / chord from mid-height",
                        step=0.1,
                        format="%.2f",
                        key=ymax_key,
                        **_unless_set(ymax_key, value=float(domain["y_max"])),
                    )
                ),
            }

    return knobs, geo_overrides


@dataclass
class _ActiveGeometry:
    """Geometry currently on screen, plus the catalog params sliders override."""

    geometry: Optional[GeometryOutput]
    base_params: Dict[str, Any]
    scope: str
    mode: str
    meta: Dict[str, Any]
    geo_spec: Optional[Dict[str, Any]] = None


def _cached_geometry(key: str, build: Callable[[], GeometryOutput]) -> GeometryOutput:
    """Reuse a shape while mesh sliders move. A new key rebuilds it."""

    cache: Dict[str, GeometryOutput] = st.session_state.setdefault("mesh_geo_cache", {})
    cached = cache.get(key)
    if cached is None:
        cached = build()
        cache[key] = cached
    return cached


def _control_seed(defaults: Optional[Dict[str, Any]], mode: Optional[str]) -> Dict[str, Any]:
    """Slider positions before the user touches them, after catalog + clamp."""

    seed = dict(_MESH_DEFAULTS)
    seed.update({key: value for key, value in (defaults or {}).items() if value is not None})
    domain = dict(_MESH_DEFAULTS["domain"])
    domain.update(seed.get("domain") or {})
    show_2d = mode != "3D"
    show_3d = mode != "2D"
    knobs: Dict[str, Any] = {}
    if show_2d:
        clustering = seed["clustering"]
        if clustering not in _CLUSTERING_CHOICES:
            clustering = _CLUSTERING_CHOICES[0]
        knobs["clustering"] = clustering
        knobs["n_panels"] = int(_clamp(seed["n_panels"], 16, 400))
        knobs["le_cluster_factor"] = float(_clamp(seed["le_cluster_factor"], 0.0, 20.0))
        knobs["te_cluster_factor"] = float(_clamp(seed["te_cluster_factor"], 0.0, 20.0))
        knobs["n_x"] = int(_clamp(seed["n_x"], 16, 241))
        knobs["n_y"] = int(_clamp(seed["n_y"], 16, 161))
        knobs["domain"] = {
            "x_min": float(domain["x_min"]),
            "x_max": float(domain["x_max"]),
            "y_min": float(domain["y_min"]),
            "y_max": float(domain["y_max"]),
        }
    if show_3d:
        orientation = seed["normal_orientation"]
        if orientation not in _ORIENTATION_CHOICES:
            orientation = _ORIENTATION_CHOICES[0]
        knobs["normal_orientation"] = orientation
        knobs["require_closed_3d"] = bool(seed["require_closed_3d"])
        density = int(seed.get("surface_density", 1))
        knobs["surface_density"] = density if density in _DENSITY_CHOICES else 1
    return knobs


def _changed_mesh_keys(baseline: Dict[str, Any], knobs: Dict[str, Any]) -> list[str]:
    changed: list[str] = []
    for key, value in knobs.items():
        previous = baseline.get(key)
        if isinstance(value, dict) or isinstance(previous, dict):
            if dict(previous or {}) != dict(value or {}):
                changed.append(key)
        elif previous != value:
            changed.append(key)
    return sorted(changed)


def _geometry_from_upload(engine: GeometryEngine) -> Optional[_ActiveGeometry]:
    st.sidebar.markdown("**Your geometry file**")
    uploaded = st.sidebar.file_uploader(
        "SVG / DXF (2D section) · STL / OBJ (3D body)",
        type=["svg", "dxf", "stl", "obj"],
        key="geometry_upload",
    )
    if uploaded is None:
        st.sidebar.caption(
            "2D sections must be a closed ring (SVG `<polygon>`, SVG path ending "
            "in `Z`, or a DXF polyline with the closed flag). 3D must be watertight. "
            "The file is meshed as soon as it is chosen."
        )
        return None

    payload = uploaded.getvalue()
    token = hashlib.sha256(payload).hexdigest()
    suffix = uploaded.name.rsplit(".", 1)[-1].lower()
    name = uploaded.name

    def _build() -> GeometryOutput:
        if suffix in {"svg", "dxf"}:
            return engine.import_2d(payload, suffix, shape_name=name)
        return engine.import_3d(payload, suffix, shape_name=name)

    geometry = _cached_geometry(f"upload:{token}", _build)
    return _ActiveGeometry(
        geometry=geometry,
        base_params={},
        scope=f"upload_{token[:12]}",
        mode=str(geometry["mode"]),
        meta={"kind": "upload", "title": geometry["shape_name"], "notes": {}},
    )


def _geometry_from_sidebar(engine: GeometryEngine) -> _ActiveGeometry:
    source = st.sidebar.radio(
        "Shape",
        (
            "Geo module sample",
            "NACA 4-digit",
            "Flat plate",
            "Parametric wing",
            "Cylinder",
        ),
        key="shape_kind",
    )
    if source == "Geo module sample":
        samples = list(list_geo_module_samples())
        titles = [sample.title for sample in samples]
        title = st.sidebar.selectbox("Geo unit sample", titles, key="geo_sample_title")
        selected = samples[titles.index(title)]
        spec = spec_as_dict(selected.spec)
        return _ActiveGeometry(
            geometry=None,
            base_params={},
            scope=f"shape_{selected.id}",
            mode=str(spec.get("mode") or "2D"),
            meta={"kind": "interactive", "title": title, "notes": {}},
            geo_spec=spec,
        )
    elif source == "NACA 4-digit":
        code = st.sidebar.text_input("NACA code", "4412", key="naca_code")
        chord = float(st.sidebar.number_input("Chord", min_value=1e-6, value=1.0, key="naca_chord"))
        n_pts = int(st.sidebar.slider("Points", 20, 2000, 160, key="naca_points"))

        def _build() -> GeometryOutput:
            return engine.generate_naca(code, chord=chord, num_points=n_pts)

        geometry = _cached_geometry(f"naca:{code}:{chord}:{n_pts}", _build)
        return _ActiveGeometry(
            geometry=geometry,
            base_params={},
            scope="shape_naca",
            mode=str(geometry["mode"]),
            meta={"kind": "interactive", "title": geometry["shape_name"], "notes": {}},
        )
    if source == "Flat plate":
        chord = float(
            st.sidebar.number_input("Chord", min_value=1e-6, value=1.0, key="plate_chord")
        )
        n_pts = int(st.sidebar.slider("Points", 20, 2000, 120, key="plate_points"))

        def _build() -> GeometryOutput:
            return engine.generate_flat_plate(chord=chord, num_points=n_pts)

        geometry = _cached_geometry(f"plate:{chord}:{n_pts}", _build)
        return _ActiveGeometry(
            geometry=geometry,
            base_params={},
            scope="shape_plate",
            mode=str(geometry["mode"]),
            meta={"kind": "interactive", "title": geometry["shape_name"], "notes": {}},
        )
    if source == "Parametric wing":
        airfoil = st.sidebar.text_input("Section NACA", "0012", key="wing_airfoil")
        chord = float(
            st.sidebar.number_input("Chord", min_value=1e-6, value=1.0, key="wing_chord")
        )
        span = float(st.sidebar.number_input("Span", min_value=1e-6, value=6.0, key="wing_span"))
        sweep = float(st.sidebar.slider("Sweep (deg)", -45.0, 45.0, 15.0, key="wing_sweep"))
        dihedral = float(
            st.sidebar.slider("Dihedral (deg)", -20.0, 20.0, 5.0, key="wing_dihedral")
        )
        return _ActiveGeometry(
            geometry=None,
            base_params={},
            scope="shape_wing",
            mode="3D",
            meta={"kind": "interactive", "title": f"Wing_NACA_{airfoil}", "notes": {}},
            geo_spec={
                "type": "wing",
                "mode": "3D",
                "airfoil": airfoil,
                "chord": chord,
                "span": span,
                "sweep_deg": sweep,
                "dihedral_deg": dihedral,
                "num_sections": 16,
                "airfoil_points": 100,
            },
        )
    radius = float(
        st.sidebar.number_input("Radius", min_value=1e-6, value=0.5, key="cyl_radius")
    )
    length = float(
        st.sidebar.number_input("Length", min_value=1e-6, value=2.0, key="cyl_length")
    )
    return _ActiveGeometry(
        geometry=None,
        base_params={},
        scope="shape_cylinder",
        mode="3D",
        meta={"kind": "interactive", "title": "Cylinder", "notes": {}},
        geo_spec={
            "type": "cylinder",
            "mode": "3D",
            "radius": radius,
            "length": length,
            "segments": 32,
            "rings": 8,
        },
    )


def _store_run(
    *,
    geo: Optional[GeometryOutput],
    mesh: Optional[MeshOutput],
    meta: Optional[Dict[str, Any]],
    error: Optional[str],
    expect_error: Optional[str],
    physics: Any = None,
    physics_error: Optional[str] = None,
) -> None:
    st.session_state["mesh_geo"] = geo
    st.session_state["mesh_out"] = mesh
    st.session_state["mesh_meta"] = meta
    st.session_state["mesh_error"] = error
    st.session_state["mesh_expect"] = expect_error
    st.session_state["physics_out"] = physics
    st.session_state["physics_error"] = physics_error


def _geo_deltas(
    spec: Optional[Dict[str, Any]],
    overrides: Dict[str, Any],
) -> list[str]:
    """Keys the user moved off the spec, treating omitted generator defaults as the seed."""

    if not spec:
        return []
    implied = {"airfoil_points": 100, "num_sections": 10, "segments": 48, "rings": 12}
    changed: list[str] = []
    for key, value in overrides.items():
        baseline = spec[key] if key in spec and spec[key] is not None else implied.get(key, value)
        if baseline != value:
            changed.append(key)
    return sorted(changed)


def _apply_geo_overrides(
    engine: GeometryEngine,
    active: _ActiveGeometry,
    overrides: Dict[str, Any],
) -> GeometryOutput:
    """Rebuild a parametric body when resolution sliders move.

    Uploads and raw facet fixtures have no spec. They keep the geometry
    already parsed, and only the mesh density slider retessellates them.
    """

    if active.geo_spec is None:
        if active.geometry is None:
            raise GeometryEngineError("No geometry is loaded.")
        return active.geometry
    spec = dict(active.geo_spec)
    spec.update(overrides)
    frozen = {key: spec[key] for key in sorted(spec)}
    cache_key = active.scope + "|" + "|".join(f"{name}={frozen[name]}" for name in frozen)

    def _build() -> GeometryOutput:
        return engine.generate_from_spec(dict(frozen))

    return _cached_geometry(cache_key, _build)


def _flow_seed(active: _ActiveGeometry) -> tuple[float, float, float, str]:
    """Catalog freestream when the case has one, otherwise a mode default."""

    stored = dict(active.meta.get("flow") or {})
    closed = False
    if active.geometry is not None:
        closed = bool(active.geometry["is_closed"])
    elif active.geo_spec is not None and active.geo_spec.get("type") == "cylinder":
        closed = True
    alpha = float(stored.get("alpha_deg", 0.0 if closed and active.mode == "3D" else 4.0))
    reynolds = float(stored.get("reynolds", 5.0e5))
    mach = float(stored.get("mach", 0.0))
    regime = stored.get("regime")
    if regime not in ("inviscid", "viscous_inviscid"):
        regime = "inviscid" if active.mode == "3D" else "viscous_inviscid"
    return alpha, reynolds, mach, str(regime)


def _flow_knobs(active: _ActiveGeometry) -> Dict[str, Any]:
    """Sidebar freestream. Keys are scoped so a mesh slider is never reused."""

    st.sidebar.header("Flow conditions")
    alpha, reynolds, mach, regime = _flow_seed(active)
    scope = active.scope
    alpha_key = f"{scope}_alpha"
    log_key = f"{scope}_log_re"
    mach_key = f"{scope}_mach"
    regime_key = f"{scope}_regime"
    alpha_value = float(
        st.sidebar.slider(
            "Angle of attack α (deg)",
            min_value=-12.0,
            max_value=16.0,
            step=0.5,
            key=alpha_key,
            **_unless_set(alpha_key, value=float(np.clip(alpha, -12.0, 16.0))),
        )
    )
    log_seed = float(np.clip(np.log10(max(reynolds, 1.0)), 4.0, 7.0))
    log_re = float(
        st.sidebar.slider(
            "Reynolds number (log10)",
            min_value=4.0,
            max_value=7.0,
            step=0.05,
            key=log_key,
            **_unless_set(log_key, value=log_seed),
        )
    )
    mach_value = float(
        st.sidebar.slider(
            "Mach",
            min_value=0.0,
            max_value=0.65,
            step=0.05,
            key=mach_key,
            **_unless_set(mach_key, value=float(np.clip(mach, 0.0, 0.65))),
        )
    )
    choices = ("inviscid", "viscous_inviscid")
    regime_value = st.sidebar.selectbox(
        "Regime",
        choices,
        key=regime_key,
        **_unless_set(regime_key, index=choices.index(regime)),
    )
    return {
        "alpha_deg": alpha_value,
        "reynolds": float(10.0 ** log_re),
        "mach": mach_value,
        "v_inf": 1.0,
        "rho": 1.225,
        "regime": str(regime_value),
    }


def _solve_flow(mesh: MeshOutput, flow: Dict[str, Any]) -> tuple[Any, Optional[str], float]:
    """Run Module 3. A solver error stays on the page and keeps the mesh."""

    freestream, parameters = split_flow_payload(flow)
    started = time.perf_counter()
    try:
        solved = PhysicsEngine().solve_from_mesh(mesh, freestream, parameters)
    except (PhysicsEngineError, ValueError) as exc:
        return None, str(exc), time.perf_counter() - started
    return solved, None, time.perf_counter() - started


def _publish_mesh(
    mesh_engine: MeshEngine,
    active: _ActiveGeometry,
    knobs: Dict[str, Any],
    geo_tuned: Optional[list[str]] = None,
    flow: Optional[Dict[str, Any]] = None,
) -> None:
    """Layer the live slider values on the catalog params and discretize."""

    if active.geometry is None:
        raise GeometryEngineError("No geometry is loaded.")
    params = dict(active.base_params)
    params.update(knobs)
    started = time.perf_counter()
    mesh = mesh_engine.from_geometry(active.geometry, params)
    seconds = time.perf_counter() - started
    notes = dict(active.meta.get("notes") or {})
    notes["math_seconds"] = seconds
    overrides = _changed_mesh_keys(_control_seed(active.base_params, active.mode), knobs)
    if geo_tuned:
        overrides = sorted(set(overrides).union(geo_tuned))
    if overrides:
        notes["math_overrides"] = overrides
    else:
        notes.pop("math_overrides", None)
    physics = None
    physics_error = None
    if flow is not None:
        physics, physics_error, physics_seconds = _solve_flow(mesh, flow)
        notes["physics_seconds"] = physics_seconds
        seed_alpha, seed_re, seed_mach, seed_regime = _flow_seed(active)
        tuned_flow = []
        if abs(float(flow["alpha_deg"]) - seed_alpha) > 1e-9:
            tuned_flow.append("alpha_deg")
        if abs(np.log10(float(flow["reynolds"])) - np.log10(seed_re)) > 0.02:
            tuned_flow.append("reynolds")
        if abs(float(flow["mach"]) - seed_mach) > 1e-9:
            tuned_flow.append("mach")
        if str(flow["regime"]) != seed_regime:
            tuned_flow.append("regime")
        if tuned_flow:
            notes["physics_overrides"] = tuned_flow
        else:
            notes.pop("physics_overrides", None)
    meta = dict(active.meta)
    meta["notes"] = notes
    if meta.get("kind") == "math_sample":
        meta["seconds"] = seconds
    _store_run(
        geo=active.geometry,
        mesh=mesh,
        meta=meta,
        error=None,
        expect_error=None,
        physics=physics,
        physics_error=physics_error,
    )


def _resolve_pipeline(engine: GeometryEngine) -> _ActiveGeometry:
    cases = list(list_pipeline_cases())
    labels = [f"{case.title}  [{case.id}]" for case in cases]
    default = next(
        (label for label, case in zip(labels, cases) if case.id == "airfoil_naca_0012_cruise"),
        labels[0],
    )
    if "pipeline_case_choice" not in st.session_state:
        st.session_state["pipeline_case_choice"] = default
    choice = st.sidebar.selectbox("End-to-end case", labels, key="pipeline_case_choice")
    selected = cases[labels.index(choice)]
    st.sidebar.caption(selected.description)
    st.sidebar.caption(
        "Live stages: "
        + ", ".join(selected.active_stages)
        + ". Pending stages stay on this same case id."
    )
    spec = spec_as_dict(selected.stages.geo)
    mode = str(spec.get("mode") or selected.stages.geo.mode or "2D")
    return _ActiveGeometry(
        geometry=None,
        base_params=dict(selected.stages.math or {}),
        scope=f"pipe_{selected.id}",
        mode=mode,
        geo_spec=spec,
        meta={
            "kind": "pipeline",
            "case_id": selected.id,
            "title": selected.title,
            "description": selected.description,
            "active_stages": list(selected.active_stages),
            "pending_stages": list(pending_pipeline_stages(selected)),
            "notes": {},
            "flow": dict(selected.stages.physics or {}),
        },
    )


def _resolve_math_sample(engine: GeometryEngine, mesh_engine: MeshEngine) -> Optional[_ActiveGeometry]:
    """Return a tunable sample, or store a guard result and return None."""

    samples = list(list_math_module_samples())
    labels = [f"{sample.title}  [{sample.id}]" for sample in samples]
    choice = st.sidebar.selectbox("Math unit sample", labels, key="math_sample_choice")
    selected = samples[labels.index(choice)]
    st.sidebar.caption(selected.description)
    if selected.spec.expect_error:
        st.sidebar.caption(
            f"Guard sample. Expected: {selected.spec.expect_error}. "
            "It only holds at its stored parameters, so it cannot be tuned."
        )
        result = run_math_module_sample(
            selected.id,
            engine=engine,
            mesh_engine=mesh_engine,
        )
        _store_run(
            geo=result["geo"],
            mesh=result["math"],
            meta={
                "kind": "math_sample",
                "sample_id": result["sample_id"],
                "title": result["title"],
                "description": result["description"],
                "seconds": result["seconds"],
                "notes": {},
            },
            error=result["error_message"],
            expect_error=result["expect_error"],
        )
        return None

    spec = selected.spec
    if spec.facets is not None:
        sample_id = selected.id

        def _build_facets() -> GeometryOutput:
            loaded = run_math_module_sample(
                sample_id,
                engine=engine,
                mesh_engine=mesh_engine,
            )
            if loaded["geo"] is None:
                raise GeometryEngineError(f"Math sample '{sample_id}' has no geometry.")
            return loaded["geo"]

        geometry = _cached_geometry(f"math:{selected.id}:facets", _build_facets)
        mode = "3D"
        geo_spec = None
    else:
        geo_spec = spec_as_dict(spec.geometry) if spec.geometry is not None else {}
        geometry = None
        mode = str(
            (spec.geometry.mode if spec.geometry is not None else None)
            or geo_spec.get("mode")
            or "2D"
        )

    return _ActiveGeometry(
        geometry=geometry,
        base_params=dict(spec.parameters),
        scope=f"sample_{selected.id}",
        mode=mode,
        geo_spec=geo_spec,
        meta={
            "kind": "math_sample",
            "sample_id": selected.id,
            "title": selected.title,
            "description": selected.description,
            "notes": {},
        },
    )


_SOURCES = (
    "Upload my geometry file",
    "Build a shape",
    "Pipeline case",
    "Math module sample",
)

_MATH_SAMPLE_BANNER = (
    "Tunable math samples start from the stored fixture. "
    "Mesh settings override that fixture for this session. "
    "Guard samples stay locked to the parameters that must fail."
)


def _run_from_sidebar(engine: GeometryEngine, mesh_engine: MeshEngine) -> None:
    if "mesh_source" not in st.session_state:
        st.session_state["mesh_source"] = "Pipeline case"
    source = st.sidebar.radio("Source", _SOURCES, key="mesh_source")

    active: Optional[_ActiveGeometry] = None
    try:
        if source == "Pipeline case":
            active = _resolve_pipeline(engine)
        elif source == "Math module sample":
            active = _resolve_math_sample(engine, mesh_engine)
            if active is None:
                return
        elif source == "Upload my geometry file":
            active = _geometry_from_upload(engine)
            if active is None:
                _store_run(
                    geo=None,
                    mesh=None,
                    meta={"kind": "empty", "title": "", "notes": {}},
                    error=None,
                    expect_error=None,
                )
                return
        else:
            active = _geometry_from_sidebar(engine)

        knobs, geo_overrides = _mesh_knobs(
            scope=active.scope,
            defaults=active.base_params,
            mode=active.mode,
            geo_spec=active.geo_spec,
        )
        active.geometry = _apply_geo_overrides(engine, active, geo_overrides)
        _publish_mesh(
            mesh_engine,
            active,
            knobs,
            geo_tuned=_geo_deltas(active.geo_spec, geo_overrides),
            flow=_flow_knobs(active),
        )
    except (GeometryEngineError, MeshEngineError, DatasetCatalogError, ValidationError) as exc:
        st.sidebar.error(str(exc))
        _store_run(
            geo=None if active is None else active.geometry,
            mesh=None,
            meta=None if active is None else active.meta,
            error=str(exc),
            expect_error=None,
        )


def render_contracts(geometry: Optional[GeometryOutput], mesh: MeshOutput) -> None:
    meta = st.session_state.get("mesh_meta") or {}
    label = str(meta.get("case_id") or meta.get("sample_id") or meta.get("title") or "run")
    summary = contract_summary_json(
        geometry,
        mesh,
        label=label,
        notes=meta.get("notes"),
        physics=st.session_state.get("physics_out"),
    )
    stem = label.replace(" ", "_")

    st.caption(
        "Exactly what Module 1 hands to Module 2 and what Module 2 hands to "
        "Module 3. The JSON carries shapes and ranges; the NPZ carries every raw array."
    )
    left, right = st.columns(2)
    left.download_button(
        "Download contract summary (JSON)",
        data=summary,
        file_name=f"{stem}_contracts.json",
        mime="application/json",
        width="stretch",
    )
    right.download_button(
        "Download contract arrays (NPZ)",
        data=contract_npz_bytes(geometry, mesh, st.session_state.get("physics_out")),
        file_name=f"{stem}_contracts.npz",
        mime="application/octet-stream",
        width="stretch",
    )
    st.json(summary, expanded=False)


def render_mesh_views(geometry: Optional[GeometryOutput], mesh: MeshOutput) -> None:
    quality = mesh["quality"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Panels", quality["n_panels"])
    c2.metric("Min length", f"{quality['min_length']:.3e}")
    c3.metric("Max aspect", f"{quality['max_aspect_ratio']:.2f}")
    c4.metric("Normal alignment", f"{quality['mean_normal_alignment']:.3f}")

    if mesh["mode"] == "2D":
        tabs = st.tabs(["Panels", "Farfield mask", "IBL stations", "Quality", "Contracts"])
        with tabs[0]:
            if geometry is not None:
                _show_2d_chart(figure_panels_2d(geometry, mesh), "mesh-panels-2d")
            panel = mesh["panel_2d"]
            assert panel is not None
            st.caption(
                f"chord={panel['chord']:.6g} · LE node={panel['le_index']} · "
                f"TE node={panel['te_index']} · te_first={panel['te_first']} · "
                f"{_farfield_caption(mesh)}"
            )
        with tabs[1]:
            _show_2d_chart(figure_farfield(mesh), "mesh-farfield-2d")
            st.caption(_farfield_caption(mesh))
            table = _mask_table(mesh)
            if table is not None:
                st.dataframe(table, hide_index=True)
        with tabs[2]:
            _show_2d_chart(figure_ibl(mesh), "mesh-ibl-2d")
        with tabs[3]:
            st.dataframe(_quality_frame(mesh), hide_index=True)
        with tabs[4]:
            render_contracts(geometry, mesh)
        _render_flow_tabs_2d(mesh)
        return

    tabs = st.tabs(["Normals", "Quality", "Contracts"])
    with tabs[0]:
        st.plotly_chart(figure_normals_3d(mesh), width="stretch")
        panel = mesh["panel_3d"]
        assert panel is not None
        st.caption(
            f"closed={panel['is_closed']} · orientation={panel['normal_orientation']} · "
            f"edges [{quality['min_length']:.3e}, {quality['max_length']:.3e}] · "
            f"area [{quality['min_area']:.3e}, {quality['max_area']:.3e}]"
        )
    with tabs[1]:
        st.dataframe(_quality_frame(mesh), hide_index=True)
    with tabs[2]:
        render_contracts(geometry, mesh)
    _render_pressure_3d(mesh)


def _cp_figure(mesh: MeshOutput, physics: Dict[str, Any]) -> go.Figure:
    panel = mesh["panel_2d"]
    assert panel is not None
    cp = np.asarray(physics["inviscid"]["cp_inviscid"], dtype=np.float64)
    x = np.asarray(panel["x_over_c"], dtype=np.float64)
    upper = np.asarray(panel["normal"], dtype=np.float64)[:, 1] >= 0.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x[upper], y=cp[upper], mode="lines", name="Upper"))
    fig.add_trace(go.Scatter(x=x[~upper], y=cp[~upper], mode="lines", name="Lower"))
    fig.update_layout(xaxis_title="x/c", yaxis_title="Cp", yaxis_autorange="reversed", height=PLOT_HEIGHT_2D)
    return fig


def _boundary_layer_figure(mesh: MeshOutput, physics: Dict[str, Any]) -> Optional[go.Figure]:
    ibl = physics.get("ibl")
    stations = mesh["ibl_1d"]
    if ibl is None or stations is None:
        return None
    fig = go.Figure()
    for side in ("upper", "lower"):
        s = np.asarray(stations[f"s_{side}"], dtype=np.float64)
        for name, key in (("Cf", "cf"), ("δ*", "delta_star"), ("θ", "theta"), ("δ", "delta")):
            values = np.asarray(ibl[f"{key}_{side}"], dtype=np.float64)
            if values.shape != s.shape:
                continue
            fig.add_trace(go.Scatter(x=s, y=values, mode="lines", name=f"{name} {side}"))
    fig.update_layout(xaxis_title="s", yaxis_title="profile", height=PLOT_HEIGHT_2D)
    return fig


def _farfield_quiver(mesh: MeshOutput, physics: Dict[str, Any]) -> Optional[go.Figure]:
    field = physics.get("field_2d")
    grid = mesh["cartesian_2d"]
    if field is None or grid is None:
        return None
    x_nodes = np.asarray(grid["x_nodes"], dtype=np.float64)
    y_nodes = np.asarray(grid["y_nodes"], dtype=np.float64)
    u = np.asarray(field["u"], dtype=np.float64)
    v = np.asarray(field["v"], dtype=np.float64)
    step = max(1, int(max(u.shape) / 18))
    xs, ys = np.meshgrid(x_nodes, y_nodes)
    xs = xs[::step, ::step]
    ys = ys[::step, ::step]
    us = u[::step, ::step]
    vs = v[::step, ::step]
    live = np.isfinite(us) & np.isfinite(vs)
    origin = np.column_stack((xs[live], ys[live]))
    direction = np.column_stack((us[live], vs[live]))
    speed = np.linalg.norm(direction, axis=1)
    direction = direction / np.maximum(speed[:, None], 1e-8)
    span = float(max(np.ptp(x_nodes), np.ptp(y_nodes), 1e-6))
    qx, qy = _quiver(origin, direction, 0.04 * span)
    fig = go.Figure(go.Scatter(x=qx, y=qy, mode="lines", name="Velocity", line={"width": 1.5}))
    apply_2d_plot_layout(
        fig,
        x_range=[float(x_nodes[0]), float(x_nodes[-1])],
        y_range=[float(y_nodes[0]), float(y_nodes[-1])],
        view_id="physics-farfield",
    )
    return fig


def _render_flow_tabs_2d(mesh: MeshOutput) -> None:
    physics = st.session_state.get("physics_out")
    physics_error = st.session_state.get("physics_error")
    if physics_error:
        st.error(physics_error)
    if physics is None:
        return
    loads = physics["loads"]
    st.caption(
        f"Cl={loads['cl']:.4f} · Cd={loads['cd']:.4f} · "
        f"iterations={physics['vic_iterations']} · residual={physics['residual']:.3e}"
    )
    for note in physics["warnings"]:
        if "separated" in note.lower() or physics["converged"] is False:
            st.warning(note)
        elif "skipped" in note.lower():
            st.info(note)
        else:
            st.warning(note)
    if physics["converged"] is False and not any("residual" in note for note in physics["warnings"]):
        st.warning(
            f"Viscous-inviscid iteration did not converge "
            f"(vic_iterations={physics['vic_iterations']}, residual={physics['residual']:.3e})."
        )
    tabs = st.tabs(["Cp", "Boundary layer", "Farfield vectors"])
    with tabs[0]:
        _show_2d_chart(_cp_figure(mesh, physics), "physics-cp-2d")
    with tabs[1]:
        figure = _boundary_layer_figure(mesh, physics)
        if figure is None:
            st.info("This solve has no boundary-layer stations.")
        else:
            _show_2d_chart(figure, "physics-bl-2d")
            ibl = physics["ibl"]
            if ibl is not None:
                st.caption(
                    f"transition upper s={ibl['transition_s_upper']:.4g} · "
                    f"lower s={ibl['transition_s_lower']:.4g} · "
                    f"separation upper s={ibl['separation_s_upper']:.4g} · "
                    f"lower s={ibl['separation_s_lower']:.4g}"
                )
    with tabs[2]:
        quiver = _farfield_quiver(mesh, physics)
        if quiver is None:
            st.info("No farfield was reconstructed.")
        else:
            _show_2d_chart(quiver, "physics-quiver-2d")


def _render_pressure_3d(mesh: MeshOutput) -> None:
    physics = st.session_state.get("physics_out")
    physics_error = st.session_state.get("physics_error")
    if physics_error:
        st.error(physics_error)
        return
    if physics is None:
        return
    for note in physics["warnings"]:
        if "skipped" in note.lower():
            st.info(note)
        else:
            st.warning(note)
    panel = mesh["panel_3d"]
    if panel is None:
        return
    loads = physics["loads"]
    st.caption(f"Cl={loads['cl']:.4f} · Cd={loads['cd']:.4f}")
    facets = panel["facets"]
    shown = facets if facets.shape[0] <= 25_000 else facets[:25_000]
    cp = np.asarray(physics["inviscid"]["cp_inviscid"], dtype=np.float64)
    if cp.shape[0] != facets.shape[0]:
        return
    cp = cp[: shown.shape[0]]
    vertices = shown.reshape(-1, 3)
    intensity = np.repeat(cp, 3)
    index = np.arange(0, shown.shape[0] * 3, 3)
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=index,
                j=index + 1,
                k=index + 2,
                intensity=intensity,
                colorscale="RdBu_r",
                colorbar={"title": "Cp"},
                name="Cp",
            )
        ]
    )
    step = max(1, int(panel["areas"].shape[0]) // 500)
    scale = 0.08 * float(
        np.linalg.norm(
            np.array([np.ptp(facets[:, :, 0]), np.ptp(facets[:, :, 1]), np.ptp(facets[:, :, 2])])
        )
    )
    qx, qy, qz = _quiver(panel["centroids"][::step], panel["normals"][::step], max(scale, 1e-6))
    fig.add_trace(
        go.Scatter3d(
            x=qx,
            y=qy,
            z=qz,
            mode="lines",
            name="Unit normals",
            line={"width": 3, "color": "#c0392b"},
        )
    )
    st.plotly_chart(fig, width="stretch")


def render_mesh_workbench() -> None:
    st.title("Modules 1–3 — Geometry, Mesh, and Physics")
    st.write(
        "Geometry feeds Module 2, and the mesh feeds the panel solver. Pick a source in the sidebar: "
        "**Upload my geometry file** (SVG/DXF section or STL/OBJ body), "
        "**Build a shape** from parameters, **Pipeline case** for a stored "
        "end-to-end journey, or **Math module sample** for a mesh fixture "
        "including the guard cases that must fail. Mesh sliders rebuild the "
        "panels as soon as you move them."
    )

    engine = GeometryEngine()
    mesh_engine = MeshEngine()
    _run_from_sidebar(engine, mesh_engine)

    meta = st.session_state.get("mesh_meta") or {}
    if meta.get("kind") == "pipeline":
        pending = meta.get("pending_stages") or []
        notes = meta.get("notes") or {}
        seconds = notes.get("math_seconds")
        timing = f" · mesh {seconds * 1000:.1f} ms" if isinstance(seconds, float) else ""
        physics_seconds = notes.get("physics_seconds")
        if isinstance(physics_seconds, float):
            timing += f" · physics {physics_seconds * 1000:.1f} ms"
        tuned = list(notes.get("math_overrides") or [])
        tuned.extend(notes.get("physics_overrides") or [])
        if tuned:
            timing += " · tuned: " + ", ".join(tuned)
        st.success(
            f"Pipeline `{meta.get('case_id')}` · live: "
            + ", ".join(meta.get("active_stages") or [])
            + " · pending: "
            + (", ".join(pending) if pending else "none")
            + timing
        )
        st.caption(str(meta.get("description") or ""))
    elif meta.get("kind") == "math_sample" or st.session_state.get("mesh_source") == "Math module sample":
        st.info(_MATH_SAMPLE_BANNER)
        if meta.get("kind") == "math_sample":
            seconds = meta.get("seconds")
            timing = f" · {float(seconds) * 1000:.1f} ms" if isinstance(seconds, float) else ""
            st.caption(f"Math sample `{meta.get('sample_id')}`{timing}")
            st.caption(str(meta.get("description") or ""))

    expect = st.session_state.get("mesh_expect")
    error = st.session_state.get("mesh_error")
    if expect and error:
        st.success(f"Guard held: {expect}")
        st.caption(error)
    elif error:
        st.error(error)

    geometry = st.session_state.get("mesh_geo")
    mesh = st.session_state.get("mesh_out")
    if geometry is None and mesh is None and not error and not expect:
        st.info(
            "Upload a geometry file, or pick a pipeline case, shape, or math sample. "
            "Mesh settings apply to whatever is loaded."
        )
    if geometry is not None:
        st.subheader("Module 1 — Geometry")
        st.caption(_bbox_caption(geometry))
        if geometry["mode"] == "2D":
            _show_2d_chart(figure_2d(geometry), "mesh-geometry-2d")
        else:
            st.plotly_chart(figure_3d(geometry), width="stretch")

    if mesh is not None:
        st.subheader("Module 2 — Mesh")
        render_mesh_views(geometry, mesh)
