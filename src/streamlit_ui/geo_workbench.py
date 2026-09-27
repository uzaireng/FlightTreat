"""Interactive Module 1 (Geometry Engine) workbench helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from geo_module import (
    GeometryEngine,
    GeometryEngineError,
    GeometryOutput,
    list_builtin_samples,
)
from shared.dataset_catalog import (
    DatasetCatalogError,
    list_geo_module_samples,
    list_pipeline_cases,
)
from shared.pipeline_runner import run_pipeline_case


# Tall enough that a section fills a wide workbench column instead of a short strip.
PLOT_HEIGHT_2D = 800

# Wheel zoom is a Plotly config flag, not a layout attribute.
PLOTLY_2D_CONFIG: Dict[str, Any] = {
    "scrollZoom": True,
    "displaylogo": False,
    "doubleClick": "reset",
}


def _finite_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = np.asarray(x, dtype=np.float64).ravel()
    ys = np.asarray(y, dtype=np.float64).ravel()
    ok = np.isfinite(xs) & np.isfinite(ys)
    if not np.any(ok):
        return (
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([-0.5, 0.5], dtype=np.float64),
        )
    return xs[ok], ys[ok]


def section_axis_ranges(
    x: np.ndarray,
    y: np.ndarray,
    *,
    margin_fraction: float = 0.18,
    min_short_fraction: float = 0.55,
) -> tuple[list[float], list[float]]:
    """Default camera tight on the section outline.

    Farfield nodes stay in the traces so the user can pan out to them.
    The short axis is opened relative to the long axis so a thin airfoil
    is not a hairline. That ratio is only the initial window: box zoom
    is free because the axes are not scale-anchored.
    """

    xs, ys = _finite_xy(x, y)
    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))
    x_span = max(xmax - xmin, 1e-9)
    y_span = max(ymax - ymin, 1e-9)
    long_span = max(x_span, y_span)
    margin = margin_fraction * long_span

    def _pad(span: float) -> float:
        opened = min_short_fraction * long_span
        return max(margin, 0.5 * (opened - span))

    x_pad = _pad(x_span)
    y_pad = _pad(y_span)
    return [xmin - x_pad, xmax + x_pad], [ymin - y_pad, ymax + y_pad]


def section_view_id(name: str, x: np.ndarray, y: np.ndarray) -> str:
    """Identity of the outline, coarse enough that resampling does not move it.

    Mesh sliders rebuild panels on the same section. Keeping this id fixed
    lets Plotly retain pan and zoom across those reruns. A new chord or
    shape changes the id and the camera returns to ``section_axis_ranges``.
    """

    xs, ys = _finite_xy(x, y)
    span_x = float(np.max(xs) - np.min(xs))
    span_y = float(np.max(ys) - np.min(ys))
    return f"{name}|{span_x:.3g}|{span_y:.3g}"


def apply_2d_plot_layout(
    fig: go.Figure,
    *,
    x_range: list[float],
    y_range: list[float],
    view_id: str,
    height: int = PLOT_HEIGHT_2D,
) -> None:
    """Pan by default, wheel-zoom ready, no fixed-aspect box zoom.

    ``scaleanchor`` is intentionally unset. With it set, a horizontal zoom
    box is stretched vertically to keep ``scaleratio``.
    """

    fig.update_layout(
        dragmode="pan",
        height=int(height),
        autosize=True,
        uirevision=view_id,
        margin={"l": 56, "r": 28, "t": 72, "b": 48},
    )
    fig.update_xaxes(
        range=[float(x_range[0]), float(x_range[1])],
        autorange=False,
        fixedrange=False,
        uirevision=view_id,
    )
    fig.update_yaxes(
        range=[float(y_range[0]), float(y_range[1])],
        autorange=False,
        fixedrange=False,
        uirevision=view_id,
    )


def _bbox_caption(output: GeometryOutput) -> str:
    box = output["bounding_box"]
    return (
        f"`{output['shape_name']}` · {output['mode']} · "
        f"{output['num_elements']} elements · "
        f"closed={output['is_closed']} · "
        f"bbox x[{box['xmin']:.4g}, {box['xmax']:.4g}] "
        f"y[{box['ymin']:.4g}, {box['ymax']:.4g}] "
        f"z[{box['zmin']:.4g}, {box['zmax']:.4g}]"
    )


def figure_2d(output: GeometryOutput) -> go.Figure:
    points = output["points_2d"]
    assert points is not None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=points[:, 0].tolist(),
            y=points[:, 1].tolist(),
            mode="lines",
            name=output["shape_name"],
            line={"width": 2, "color": "#1f77b4"},
        )
    )
    fig.update_layout(
        title=f"2D Geometry — {output['shape_name']}",
        xaxis_title="x",
        yaxis_title="y",
        template="plotly_white",
    )
    x_range, y_range = section_axis_ranges(points[:, 0], points[:, 1])
    apply_2d_plot_layout(
        fig,
        x_range=x_range,
        y_range=y_range,
        view_id=section_view_id(output["shape_name"], points[:, 0], points[:, 1]),
    )
    return fig


def figure_3d(output: GeometryOutput, *, max_facets: int = 25_000) -> go.Figure:
    facets = output["facets_3d"]
    assert facets is not None

    shown = facets
    note = ""
    if facets.shape[0] > max_facets:
        shown = facets[:max_facets]
        note = f" (displaying first {max_facets} of {facets.shape[0]} facets)"

    vertices = shown.reshape(-1, 3)
    n_tri = shown.shape[0]
    i = np.arange(0, n_tri * 3, 3)
    j = i + 1
    k = i + 2

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=i,
                j=j,
                k=k,
                name=output["shape_name"],
                intensity=vertices[:, 2],
                colorscale="Viridis",
                showscale=True,
                opacity=0.92,
            )
        ]
    )
    fig.update_layout(
        title=f"3D Geometry — {output['shape_name']}{note}",
        scene={
            "xaxis_title": "x",
            "yaxis_title": "y",
            "zaxis_title": "z",
            "aspectmode": "data",
        },
        template="plotly_white",
        height=680,
    )
    return fig


def render_geometry(output: GeometryOutput) -> None:
    st.caption(_bbox_caption(output))

    if output["mode"] == "2D":
        st.plotly_chart(
            figure_2d(output),
            width="stretch",
            height=PLOT_HEIGHT_2D,
            config=PLOTLY_2D_CONFIG,
            key="geo-section-2d",
        )
        points = output["points_2d"]
        assert points is not None
        st.dataframe(
            pd.DataFrame(
                {
                    "x": points[:, 0],
                    "y": points[:, 1],
                }
            ),
            hide_index=True,
            height=240,
        )
        return

    st.plotly_chart(figure_3d(output), width="stretch")


def _generate_from_sidebar(engine: GeometryEngine) -> Optional[GeometryOutput]:
    source = st.sidebar.radio(
        "Source",
        (
            "Pipeline case",
            "Module sample",
            "NACA 4-digit",
            "Flat plate",
            "Parametric wing",
            "Cylinder",
            "File upload",
        ),
    )

    try:
        if source == "Pipeline case":
            cases = list_pipeline_cases()
            labels = [f"{case.title}  [{case.id}]" for case in cases]
            choice = st.sidebar.selectbox("End-to-end case", labels)
            selected = cases[labels.index(choice)]
            st.sidebar.caption(selected.description)
            st.sidebar.caption(
                "Live stages: "
                + ", ".join(selected.active_stages)
                + ". Pending stages stay on this same case id."
            )
            if st.sidebar.button("Run pipeline", type="primary"):
                result = run_pipeline_case(selected.id, engine=engine)
                st.session_state["pipeline_run"] = {
                    "case_id": result["case_id"],
                    "title": result["title"],
                    "description": result["description"],
                    "active_stages": result["active_stages"],
                    "pending_stages": result["pending_stages"],
                    "notes": result["notes"],
                }
                return result["geo"]
            return None

        if source == "Module sample":
            sample = st.sidebar.selectbox("Geo unit sample", list_builtin_samples())
            meta = next(
                item for item in list_geo_module_samples() if item.title == sample
            )
            st.sidebar.caption(meta.description)
            if st.sidebar.button("Generate", type="primary"):
                st.session_state.pop("pipeline_run", None)
                return engine.generate_sample(sample)
            return None

        if source == "NACA 4-digit":
            code = st.sidebar.text_input("NACA code", "4412")
            chord = st.sidebar.number_input("Chord", min_value=1e-6, value=1.0)
            n_pts = st.sidebar.slider("Points", 20, 2000, 160)
            if st.sidebar.button("Generate", type="primary"):
                st.session_state.pop("pipeline_run", None)
                return engine.generate_naca(code, chord=float(chord), num_points=int(n_pts))
            return None

        if source == "Flat plate":
            chord = st.sidebar.number_input("Chord", min_value=1e-6, value=1.0)
            n_pts = st.sidebar.slider("Points", 20, 2000, 120)
            if st.sidebar.button("Generate", type="primary"):
                st.session_state.pop("pipeline_run", None)
                return engine.generate_flat_plate(chord=float(chord), num_points=int(n_pts))
            return None

        if source == "Parametric wing":
            airfoil = st.sidebar.text_input("Section NACA", "0012")
            chord = st.sidebar.number_input("Chord", min_value=1e-6, value=1.0)
            span = st.sidebar.number_input("Span", min_value=1e-6, value=6.0)
            sweep = st.sidebar.slider("Sweep (deg)", -45.0, 45.0, 15.0)
            dihedral = st.sidebar.slider("Dihedral (deg)", -20.0, 20.0, 5.0)
            sections = st.sidebar.slider("Spanwise sections", 2, 80, 24)
            if st.sidebar.button("Generate", type="primary"):
                st.session_state.pop("pipeline_run", None)
                return engine.generate_wing(
                    airfoil_code=airfoil,
                    chord=float(chord),
                    span=float(span),
                    sweep_deg=float(sweep),
                    dihedral_deg=float(dihedral),
                    num_sections=int(sections),
                )
            return None

        if source == "Cylinder":
            radius = st.sidebar.number_input("Radius", min_value=1e-6, value=0.5)
            length = st.sidebar.number_input("Length", min_value=1e-6, value=2.0)
            segments = st.sidebar.slider("Segments", 8, 128, 48)
            rings = st.sidebar.slider("Rings", 1, 64, 12)
            if st.sidebar.button("Generate", type="primary"):
                st.session_state.pop("pipeline_run", None)
                return engine.generate_cylinder(
                    radius=float(radius),
                    length=float(length),
                    segments=int(segments),
                    rings=int(rings),
                )
            return None

        uploaded = st.sidebar.file_uploader(
            "SVG / DXF / STL / OBJ",
            type=["svg", "dxf", "stl", "obj"],
        )
        if uploaded is None:
            st.sidebar.info("Upload a geometry file, then click Import.")
            return None

        suffix = uploaded.name.rsplit(".", 1)[-1].lower()
        if st.sidebar.button("Import", type="primary"):
            st.session_state.pop("pipeline_run", None)
            payload = uploaded.getvalue()
            if suffix in {"svg", "dxf"}:
                return engine.import_2d(payload, suffix, shape_name=uploaded.name)
            return engine.import_3d(payload, suffix, shape_name=uploaded.name)
        return None

    except (GeometryEngineError, DatasetCatalogError) as exc:
        st.sidebar.error(str(exc))
        return None


def render_geo_workbench() -> None:
    st.title("Module 1 — Geometry Engine")
    st.write(
        "Single-module workbench with two sample tracks: **module samples** "
        "(geo unit fixtures) and **pipeline cases** (the same real-user ids "
        "that later modules will extend)."
    )

    engine = GeometryEngine()
    output = _generate_from_sidebar(engine)

    if output is not None:
        st.session_state["last_geo"] = output
    elif "last_geo" in st.session_state:
        output = st.session_state["last_geo"]
    else:
        result = run_pipeline_case("airfoil_naca_0012_cruise", engine=engine)
        output = result["geo"]
        st.session_state["last_geo"] = output
        st.session_state["pipeline_run"] = {
            "case_id": result["case_id"],
            "title": result["title"],
            "description": result["description"],
            "active_stages": result["active_stages"],
            "pending_stages": result["pending_stages"],
            "notes": result["notes"],
        }
        st.info(
            "Showing default **pipeline case** "
            f"`{result['case_id']}`. Use the sidebar to switch tracks."
        )

    pipeline_meta = st.session_state.get("pipeline_run")
    if pipeline_meta:
        st.success(
            f"Pipeline `{pipeline_meta['case_id']}` · live: "
            + ", ".join(pipeline_meta["active_stages"])
            + " · pending: "
            + (
                ", ".join(pipeline_meta["pending_stages"])
                if pipeline_meta["pending_stages"]
                else "none"
            )
        )
        st.caption(pipeline_meta["description"])

    render_geometry(output)

    with st.expander("Raw GeometryOutput (non-array fields)"):
        summary: Dict[str, Any] = {
            "mode": output["mode"],
            "shape_name": output["shape_name"],
            "num_elements": output["num_elements"],
            "is_closed": output["is_closed"],
            "bounding_box": output["bounding_box"],
            "points_2d_shape": None
            if output["points_2d"] is None
            else list(output["points_2d"].shape),
            "facets_3d_shape": None
            if output["facets_3d"] is None
            else list(output["facets_3d"].shape),
        }
        st.json(summary)
