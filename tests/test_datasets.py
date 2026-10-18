"""Catalog + pipeline-sample smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from geo_module import GeometryEngine  # noqa: E402
from math_module import MeshEngineError  # noqa: E402
from shared.dataset_catalog import (  # noqa: E402
    DatasetCatalogError,
    get_geo_module_sample,
    get_math_module_sample,
    get_pipeline_case,
    list_geo_module_samples,
    list_math_module_samples,
    list_pipeline_cases,
    resolve_dataset_file,
)
from shared.contract_export import (  # noqa: E402
    contract_arrays,
    contract_npz_bytes,
    contract_summary,
)
from shared.pipeline_runner import run_math_module_sample, run_pipeline_case  # noqa: E402


def _assert(cond: bool, message: str) -> None:
    if not cond:
        raise AssertionError(message)


def test_geo_catalog_titles() -> None:
    titles = {sample.title for sample in list_geo_module_samples()}
    _assert("NACA 0012 (Symmetric)" in titles, "canonical NACA 0012 module sample missing")
    sample = get_geo_module_sample("naca_0012")
    _assert(sample.spec.type == "naca", "naca_0012 spec type")


def test_svg_module_file_sample() -> None:
    out = GeometryEngine().generate_sample("svg_diamond_airfoil")
    _assert(out["mode"] == "2D", "SVG sample must be 2D")
    _assert(out["points_2d"] is not None, "SVG sample must produce points_2d")
    _assert(out["num_elements"] >= 3, "SVG polygon too small")
    points = out["points_2d"]
    assert points is not None
    _assert(out["is_closed"], "an SVG <polygon> is closed by definition")
    _assert(
        bool(np.allclose(points[0], points[-1])),
        "closed SVG contour must repeat its first vertex",
    )


def test_dxf_closed_flag_is_honoured() -> None:
    engine = GeometryEngine()
    body = "\n".join(
        ["0", "SECTION", "2", "ENTITIES", "0", "LWPOLYLINE", "70", "{flag}"]
        + ["10", "0.0", "20", "0.0", "10", "1.0", "20", "0.05", "10", "1.0", "20", "-0.05"]
        + ["0", "ENDSEC", "0", "EOF", ""]
    )
    closed = engine.import_2d(body.format(flag=1).encode("utf-8"), "dxf")
    opened = engine.import_2d(body.format(flag=0).encode("utf-8"), "dxf")
    _assert(closed["is_closed"], "DXF group code 70 bit 1 must close the ring")
    _assert(not opened["is_closed"], "an unflagged DXF polyline must stay open")
    _assert(
        closed["num_elements"] == opened["num_elements"] + 1,
        "closing a DXF ring must add exactly one vertex",
    )


def test_pipeline_case_runs_geo_then_math() -> None:
    result = run_pipeline_case("airfoil_naca_0012_cruise")
    _assert(result["geo"] is not None, "pipeline geo stage missing")
    _assert(result["math"] is not None, "pipeline math stage missing")
    _assert(
        result["active_stages"] == ["geo", "math", "physics", "aero"],
        "geo, math, physics, and aero are the live stages",
    )
    _assert(result["pending_stages"] == ["ai"], "only AI stays pending")
    _assert(
        result["notes"]["executed"] == ["geo", "math", "physics", "aero"],
        "runner must record all four stages",
    )
    _assert(result["physics"] is not None, "pipeline physics stage missing")
    _assert(result["aero"] is not None, "pipeline aero stage missing")
    _assert(
        abs(result["aero"]["coefficients"]["cl"] - result["physics"]["loads"]["cl"]) < 1e-9,
        "default aero lift must match the physics coefficient",
    )
    mesh = result["math"]
    assert mesh is not None
    _assert(mesh["panel_2d"] is not None, "2D cruise case must emit vortex panels")
    _assert(mesh["cartesian_2d"] is not None, "2D cruise case must emit a farfield grid")
    _assert(mesh["ibl_1d"] is not None, "2D cruise case must emit IBL stations")
    points = result["geo"]["points_2d"]
    panel = mesh["panel_2d"]
    assert points is not None and panel is not None
    _assert(not np.shares_memory(points, panel["nodes"]), "mesh nodes alias the geometry contour")


def test_all_pipeline_cases_produce_geo_and_math() -> None:
    engine = GeometryEngine()
    for case in list_pipeline_cases():
        result = run_pipeline_case(case.id, engine=engine)
        _assert(result["geo"] is not None, f"{case.id} produced no geometry")
        _assert(result["math"] is not None, f"{case.id} produced no mesh")
        mesh = result["math"]
        assert mesh is not None and result["geo"] is not None
        _assert(mesh["mode"] == result["geo"]["mode"], f"{case.id} mode changed between stages")
        _assert(result["physics"] is not None, f"{case.id} produced no physics")
        _assert(np.isfinite(result["physics"]["loads"]["cl"]), f"{case.id} lift is not finite")
        _assert(result["aero"] is not None, f"{case.id} produced no aero report")
        _assert(np.isfinite(result["aero"]["coefficients"]["cl"]), f"{case.id} aero lift is not finite")


def test_pipeline_covers_every_geometry_source() -> None:
    specs = {case.stages.geo.type for case in list_pipeline_cases()}
    for required in ("naca", "flat_plate", "wing", "cylinder", "file"):
        _assert(required in specs, f"no pipeline case exercises geometry type '{required}'")

    imported = run_pipeline_case("section_svg_import_2d")
    mesh = imported["math"]
    assert mesh is not None
    panel = mesh["panel_2d"]
    assert panel is not None
    _assert(panel["length"].size == 96, "imported SVG case must panel to its requested count")
    _assert(mesh["ibl_1d"] is not None, "imported section must still emit IBL stations")


def test_mesh_overrides_retune_a_stored_case() -> None:
    stored = run_pipeline_case("airfoil_naca_0012_cruise")
    tuned = run_pipeline_case(
        "airfoil_naca_0012_cruise",
        math_overrides={"n_panels": 64, "clustering": "uniform", "n_x": 33, "n_y": 25},
    )
    assert stored["math"] is not None and tuned["math"] is not None
    stored_panel = stored["math"]["panel_2d"]
    tuned_panel = tuned["math"]["panel_2d"]
    tuned_grid = tuned["math"]["cartesian_2d"]
    assert stored_panel is not None and tuned_panel is not None and tuned_grid is not None
    _assert(stored_panel["length"].size == 160, "stored case must keep its catalog panel count")
    _assert(tuned_panel["length"].size == 64, "override must change the panel count")
    _assert(tuned_panel["clustering"] == "uniform", "override must change the clustering scheme")
    _assert((tuned_grid["n_x"], tuned_grid["n_y"]) == (33, 25), "override must resize the farfield")
    _assert(tuned["notes"]["math_overrides"] == ["clustering", "n_panels", "n_x", "n_y"],
            "runner must record which parameters were overridden")

    replay = run_pipeline_case("airfoil_naca_0012_cruise")
    assert replay["math"] is not None and replay["math"]["panel_2d"] is not None
    _assert(replay["math"]["panel_2d"]["length"].size == 160, "overrides leaked into the catalog")

    # The workbench seeds its sliders from stages.math. Resubmitting those exact
    # values must reproduce the stored case, not a silently re-snapped mesh.
    case = get_pipeline_case("airfoil_naca_0012_cruise")
    echoed = run_pipeline_case("airfoil_naca_0012_cruise", math_overrides=dict(case.stages.math))
    assert echoed["math"] is not None and echoed["math"]["cartesian_2d"] is not None
    stored_grid = stored["math"]["cartesian_2d"]
    assert stored_grid is not None
    _assert(
        (echoed["math"]["cartesian_2d"]["n_x"], echoed["math"]["cartesian_2d"]["n_y"])
        == (stored_grid["n_x"], stored_grid["n_y"]),
        "re-submitting the stored parameters changed the farfield size",
    )
    _assert(
        bool(np.allclose(echoed["math"]["panel_2d"]["nodes"], stored_panel["nodes"])),
        "re-submitting the stored parameters changed the panel nodes",
    )

    sample = run_math_module_sample("naca0012_cosine", math_overrides={"n_panels": 32})
    assert sample["math"] is not None and sample["math"]["panel_2d"] is not None
    _assert(sample["math"]["panel_2d"]["length"].size == 32, "math fixture override ignored")

    try:
        run_math_module_sample("domain_too_small", math_overrides={"n_panels": 32})
    except MeshEngineError:
        pass
    else:
        raise AssertionError("a guard fixture must refuse mesh overrides")


def test_contract_export_round_trip() -> None:
    result = run_pipeline_case("airfoil_naca_4412_cruise")
    summary = contract_summary(
        result["geo"], result["math"], label=result["case_id"], notes=result["notes"]
    )
    _assert(summary["live_stages"] == ["geo", "math"], "export must report both live stages")
    _assert(summary["geo"]["payload"]["shape"][1] == 2, "geo payload must be (N, 2)")
    _assert(summary["math"]["panel_2d"]["n_panels"] == 160, "panel count lost in export")
    _assert(
        sum(summary["math"]["cartesian_2d"]["boundary_cells"].values())
        == summary["math"]["cartesian_2d"]["n_x"] * summary["math"]["cartesian_2d"]["n_y"],
        "boundary mask cell counts must cover the whole grid",
    )

    arrays = contract_arrays(result["geo"], result["math"])
    for key in ("geo_points_2d", "math_panel_2d_normal", "math_ibl_1d_s_upper"):
        _assert(key in arrays, f"contract arrays missing '{key}'")
    _assert(len(contract_npz_bytes(result["geo"], result["math"])) > 0, "empty NPZ export")

    wing = run_pipeline_case("wing_rect_uav_cruise")
    wing_summary = contract_summary(wing["geo"], wing["math"], label=wing["case_id"])
    _assert("panel_3d" in wing_summary["math"], "3D export must carry panel_3d")
    _assert(
        wing_summary["math"]["panel_3d"]["total_area"] > 0.0,
        "3D export must report a positive wetted area",
    )


def test_math_module_samples() -> None:
    samples = list_math_module_samples()
    _assert(len(samples) >= 8, "math catalog is missing fixtures")
    cosine = get_math_module_sample("naca0012_cosine")
    _assert(cosine.spec.parameters["clustering"] == "cosine", "cosine sample drifted")

    packed = run_math_module_sample("naca4412_half_cosine")
    ibl = packed["math"]["ibl_1d"] if packed["math"] is not None else None
    _assert(ibl is not None, "cambered sample must emit IBL stations")
    _assert(
        float(np.mean(ibl["xy_upper"][:, 1])) > float(np.mean(ibl["xy_lower"][:, 1])),
        "upper IBL side must sit above the lower side",
    )

    wing = run_math_module_sample("wing_open_skin")
    panel = wing["math"]["panel_3d"] if wing["math"] is not None else None
    _assert(panel is not None and not panel["is_closed"], "wing sample must stay an open skin")

    outward = run_math_module_sample("cylinder_outward")
    inward = run_math_module_sample("cylinder_inward")
    assert outward["math"] is not None and inward["math"] is not None
    out_n = outward["math"]["panel_3d"]["normals"]
    in_n = inward["math"]["panel_3d"]["normals"]
    _assert(float(np.sum(out_n * in_n)) < 0.0, "inward normals must oppose outward normals")

    for sample_id in ("domain_too_small", "degenerate_triangle", "wing_require_closed"):
        guard = run_math_module_sample(sample_id)
        _assert(guard["math"] is None, f"{sample_id} should not emit a mesh")
        _assert(
            guard["error_type"] == guard["expect_error"],
            f"{sample_id} raised {guard['error_type']}",
        )


def test_dataset_path_escape_rejected() -> None:
    try:
        resolve_dataset_file("../.cursorrules")
    except DatasetCatalogError:
        return
    raise AssertionError("expected DatasetCatalogError for path escape")


def main() -> None:
    test_geo_catalog_titles()
    test_svg_module_file_sample()
    test_dxf_closed_flag_is_honoured()
    test_pipeline_case_runs_geo_then_math()
    test_all_pipeline_cases_produce_geo_and_math()
    test_pipeline_covers_every_geometry_source()
    test_mesh_overrides_retune_a_stored_case()
    test_contract_export_round_trip()
    test_math_module_samples()
    test_dataset_path_escape_rejected()
    print("dataset catalog smoke tests passed")


if __name__ == "__main__":
    main()
