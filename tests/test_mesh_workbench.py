"""Sidebar mesh controls must rebuild the live mesh without a second click."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

_APP = Path(__file__).resolve().parents[1] / "src" / "streamlit_ui" / "app.py"
_SVG = (
    Path(__file__).resolve().parents[1]
    / "datasets"
    / "modules"
    / "geo"
    / "files"
    / "diamond_section.svg"
)
_PIPE = "pipe_airfoil_naca_0012_cruise"


def _panels(at: AppTest) -> int:
    mesh = at.session_state["mesh_out"]
    assert mesh is not None
    return int(mesh["quality"]["n_panels"])


def _assert_clean(at: AppTest) -> None:
    assert not at.exception, at.exception


def test_sliders_rebuild_mesh_for_upload_shapes_and_pipeline() -> None:
    at = AppTest.from_file(str(_APP), default_timeout=120)
    at.run()
    _assert_clean(at)

    # Seeded pipeline case: catalog n_panels is 160 and n_x is 121.
    assert at.session_state["mesh_meta"]["case_id"] == "airfoil_naca_0012_cruise"
    assert _panels(at) == 160
    assert at.session_state["mesh_out"]["cartesian_2d"]["x_nodes"].shape[0] == 121
    assert not any("tuned:" in str(item.value) for item in at.success)

    at.sidebar.slider(key=f"{_PIPE}_n_panels").set_value(48).run()
    _assert_clean(at)
    assert _panels(at) == 48
    assert at.session_state["mesh_out"]["panel_2d"]["nodes"].shape[0] == 49
    assert any("n_panels" in str(item.value) for item in at.success)

    at.sidebar.slider(key=f"{_PIPE}_nx").set_value(32).run()
    _assert_clean(at)
    grid = at.session_state["mesh_out"]["cartesian_2d"]
    assert grid["x_nodes"].shape[0] == 32
    assert _panels(at) == 48

    at.sidebar.slider(key=f"{_PIPE}_ny").set_value(20).run()
    _assert_clean(at)
    grid = at.session_state["mesh_out"]["cartesian_2d"]
    assert grid["y_nodes"].shape[0] == 20
    assert int(grid["n_x"]) == 32

    at.sidebar.number_input(key=f"{_PIPE}_xmin").set_value(-6.0).run()
    _assert_clean(at)
    grid = at.session_state["mesh_out"]["cartesian_2d"]
    assert float(grid["x_nodes"][0]) < -5.5
    assert any("domain" in str(item.value) for item in at.success)
    _assert_farfield_figures(at.session_state["mesh_geo"], at.session_state["mesh_out"])

    packed = np.array(at.session_state["mesh_out"]["panel_2d"]["nodes"], copy=True)
    at.sidebar.slider(key=f"{_PIPE}_le").set_value(20.0).run()
    _assert_clean(at)
    leading = np.array(at.session_state["mesh_out"]["panel_2d"]["nodes"], copy=True)
    assert _panels(at) == 48
    assert not np.allclose(packed, leading)

    at.sidebar.slider(key=f"{_PIPE}_te").set_value(12.0).run()
    _assert_clean(at)
    trailed = np.array(at.session_state["mesh_out"]["panel_2d"]["nodes"], copy=True)
    assert not np.allclose(leading, trailed)

    # Parametric shape: panel slider and geometry sliders both apply immediately.
    at.sidebar.radio(key="mesh_source").set_value("Build a shape").run()
    _assert_clean(at)
    at.sidebar.radio(key="shape_kind").set_value("NACA 4-digit").run()
    _assert_clean(at)
    points_before = int(at.session_state["mesh_geo"]["points_2d"].shape[0])
    at.sidebar.slider(key="shape_naca_n_panels").set_value(36).run()
    _assert_clean(at)
    assert _panels(at) == 36
    assert at.session_state["mesh_geo"]["points_2d"].shape[0] == points_before

    at.sidebar.slider(key="naca_points").set_value(40).run()
    _assert_clean(at)
    assert int(at.session_state["mesh_geo"]["points_2d"].shape[0]) != points_before
    assert _panels(at) == 36

    at.sidebar.radio(key="shape_kind").set_value("Parametric wing").run()
    _assert_clean(at)
    wing_before = np.array(at.session_state["mesh_geo"]["facets_3d"], copy=True)
    panels_before = _panels(at)
    at.sidebar.slider(key="shape_wing_sections").set_value(4).run()
    _assert_clean(at)
    assert _panels(at) != panels_before
    assert at.session_state["mesh_geo"]["facets_3d"].shape != wing_before.shape

    swept_from = np.array(at.session_state["mesh_geo"]["facets_3d"], copy=True)
    at.sidebar.slider(key="wing_sweep").set_value(40.0).run()
    _assert_clean(at)
    wing_after = np.array(at.session_state["mesh_geo"]["facets_3d"], copy=True)
    assert wing_after.shape == at.session_state["mesh_out"]["panel_3d"]["facets"].shape
    assert wing_after.shape == swept_from.shape
    assert not np.allclose(swept_from, wing_after)
    wing_edge = float(at.session_state["mesh_out"]["quality"]["max_length"])
    at.sidebar.select_slider(key="shape_wing_density").set_value(2).run()
    _assert_clean(at)
    assert at.session_state["mesh_geo"]["num_elements"] == int(wing_after.shape[0])
    assert _panels(at) == int(wing_after.shape[0]) * 4
    assert float(at.session_state["mesh_out"]["quality"]["max_length"]) < wing_edge * 0.6

    # Uploaded section: choosing the file meshes it, and panel count follows the slider.
    payload = _SVG.read_bytes()
    token = hashlib.sha256(payload).hexdigest()[:12]
    at.sidebar.radio(key="mesh_source").set_value("Upload my geometry file").run()
    _assert_clean(at)
    assert at.session_state["mesh_out"] is None

    at.sidebar.file_uploader(key="geometry_upload").set_value(
        ("diamond_section.svg", payload, "image/svg+xml")
    ).run()
    _assert_clean(at)
    assert at.session_state["mesh_geo"]["shape_name"] == "diamond_section.svg"
    at.sidebar.slider(key=f"upload_{token}_n_panels").set_value(24).run()
    _assert_clean(at)
    assert _panels(at) == 24
    assert at.session_state["mesh_out"]["panel_2d"]["nodes"].shape[0] == 25

    at.sidebar.radio(key="mesh_source").set_value("Math module sample").run()
    _assert_clean(at)
    texts = [str(item.value) for item in at.info]
    assert any(
        "Mesh settings override that fixture for this session." in text
        and "Guard samples stay locked" in text
        for text in texts
    )


def _assert_farfield_figures(geometry, mesh) -> None:
    import sys

    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from streamlit_ui.mesh_workbench import figure_farfield, figure_panels_2d

    grid = mesh["cartesian_2d"]
    panels = figure_panels_2d(geometry, mesh)
    box = next(trace for trace in panels.data if trace.name == "Farfield box")
    assert np.isclose(float(min(box.x)), float(grid["x_nodes"][0]))
    assert np.isclose(float(max(box.x)), float(grid["x_nodes"][-1]))
    assert np.isclose(float(min(box.y)), float(grid["y_nodes"][0]))
    assert np.isclose(float(max(box.y)), float(grid["y_nodes"][-1]))
    assert any(trace.name == "Farfield grid" for trace in panels.data)
    assert f"{int(grid['n_x'])}×{int(grid['n_y'])}" in panels.layout.title.text

    mask = figure_farfield(mesh)
    heat = mask.data[0]
    assert len(heat.x) == int(grid["n_x"])
    assert len(heat.y) == int(grid["n_y"])
    assert np.isclose(float(heat.x[0]), float(grid["x_nodes"][0]))
    assert np.isclose(float(heat.y[-1]), float(grid["y_nodes"][-1]))
    assert f"{int(grid['n_x'])}×{int(grid['n_y'])}" in mask.layout.title.text

    # Default camera sits on the section. A farfield that starts near -6 must
    # not become the axis window, and box zoom must not be aspect-locked.
    geo = np.asarray(geometry["points_2d"], dtype=np.float64)
    x_range = [float(value) for value in panels.layout.xaxis.range]
    y_range = [float(value) for value in panels.layout.yaxis.range]
    assert x_range[0] < float(np.min(geo[:, 0]))
    assert x_range[1] > float(np.max(geo[:, 0]))
    assert y_range[0] < float(np.min(geo[:, 1]))
    assert y_range[1] > float(np.max(geo[:, 1]))
    far_span = float(grid["x_nodes"][-1]) - float(grid["x_nodes"][0])
    assert x_range[0] > float(grid["x_nodes"][0]) + 1.0
    assert x_range[1] - x_range[0] < 0.5 * far_span
    assert panels.layout.dragmode == "pan"
    assert panels.layout.yaxis.scaleanchor is None
    assert panels.layout.xaxis.scaleanchor is None
    assert int(panels.layout.height) >= 720
    assert mask.layout.dragmode == "pan"
    assert mask.layout.yaxis.scaleanchor is None


def test_3d_resolution_sliders_cover_every_source() -> None:
    at = AppTest.from_file(str(_APP), default_timeout=120)
    at.run()
    _assert_clean(at)

    at.sidebar.selectbox(key="pipeline_case_choice").set_value(
        "Rectangular UAV wing (3D)  [wing_rect_uav_cruise]"
    ).run()
    _assert_clean(at)
    assert at.session_state["mesh_out"]["mode"] == "3D"
    wing_panels = _panels(at)
    wing_edge = float(at.session_state["mesh_out"]["quality"]["max_length"])
    assert at.session_state["mesh_geo"]["num_elements"] == wing_panels

    at.sidebar.slider(key="pipe_wing_rect_uav_cruise_sections").set_value(8).run()
    _assert_clean(at)
    assert _panels(at) != wing_panels
    assert any("num_sections" in str(item.value) for item in at.success)
    sectioned = _panels(at)

    at.sidebar.slider(key="pipe_wing_rect_uav_cruise_airfoil_points").set_value(32).run()
    _assert_clean(at)
    assert _panels(at) != sectioned
    points_edge = float(at.session_state["mesh_out"]["quality"]["max_length"])
    pointed = _panels(at)

    at.sidebar.select_slider(key="pipe_wing_rect_uav_cruise_density").set_value(2).run()
    _assert_clean(at)
    assert _panels(at) == pointed * 4
    assert float(at.session_state["mesh_out"]["quality"]["max_length"]) < points_edge * 0.6
    assert at.session_state["mesh_out"]["panel_3d"]["facets"].shape[0] == _panels(at)

    at.sidebar.selectbox(key="pipeline_case_choice").set_value(
        "Closed cylinder body (3D)  [body_cylinder_closed]"
    ).run()
    _assert_clean(at)
    cylinder_panels = _panels(at)
    at.sidebar.slider(key="pipe_body_cylinder_closed_rings").set_value(4).run()
    _assert_clean(at)
    assert _panels(at) != cylinder_panels
    assert float(at.session_state["mesh_out"]["quality"]["min_length"]) > 0.0

    at.sidebar.radio(key="mesh_source").set_value("Math module sample").run()
    _assert_clean(at)
    at.sidebar.selectbox(key="math_sample_choice").set_value(
        "Open swept-wing skin  [wing_open_skin]"
    ).run()
    _assert_clean(at)
    sample_panels = _panels(at)
    sample_edge = float(at.session_state["mesh_out"]["quality"]["max_length"])
    at.sidebar.slider(key="sample_wing_open_skin_sections").set_value(4).run()
    _assert_clean(at)
    assert _panels(at) != sample_panels
    at.sidebar.slider(key="sample_wing_open_skin_airfoil_points").set_value(20).run()
    _assert_clean(at)
    sampled = _panels(at)
    sampled_edge = float(at.session_state["mesh_out"]["quality"]["max_length"])
    at.sidebar.select_slider(key="sample_wing_open_skin_density").set_value(2).run()
    _assert_clean(at)
    assert _panels(at) == sampled * 4
    assert float(at.session_state["mesh_out"]["quality"]["max_length"]) < sampled_edge * 0.6
    assert sampled_edge != sample_edge or sampled != sample_panels

    stl = (
        "solid tet\n"
        "facet normal 0 0 -1\n outer loop\n"
        "  vertex 0 0 0\n  vertex 1 0 0\n  vertex 0 1 0\n endloop\nendfacet\n"
        "facet normal 0 -1 0\n outer loop\n"
        "  vertex 0 0 0\n  vertex 0 0 1\n  vertex 1 0 0\n endloop\nendfacet\n"
        "facet normal -1 0 0\n outer loop\n"
        "  vertex 0 0 0\n  vertex 0 1 0\n  vertex 0 0 1\n endloop\nendfacet\n"
        "facet normal 1 1 1\n outer loop\n"
        "  vertex 1 0 0\n  vertex 0 0 1\n  vertex 0 1 0\n endloop\nendfacet\n"
        "endsolid tet\n"
    ).encode("utf-8")
    token = hashlib.sha256(stl).hexdigest()[:12]
    at.sidebar.radio(key="mesh_source").set_value("Upload my geometry file").run()
    _assert_clean(at)
    at.sidebar.file_uploader(key="geometry_upload").set_value(
        ("tet.stl", stl, "model/stl")
    ).run()
    _assert_clean(at)
    assert at.session_state["mesh_geo"]["mode"] == "3D"
    assert _panels(at) == 4
    uploaded_edge = float(at.session_state["mesh_out"]["quality"]["max_length"])
    at.sidebar.select_slider(key=f"upload_{token}_density").set_value(2).run()
    _assert_clean(at)
    assert _panels(at) == 16
    assert float(at.session_state["mesh_out"]["quality"]["max_length"]) < uploaded_edge * 0.6
    assert at.session_state["mesh_out"]["panel_3d"]["facets"].shape[0] == 16


def main() -> None:
    test_sliders_rebuild_mesh_for_upload_shapes_and_pipeline()
    test_3d_resolution_sliders_cover_every_source()
    print("mesh workbench slider tests passed")


if __name__ == "__main__":
    main()
