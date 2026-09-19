"""Catalog + pipeline-sample smoke tests."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from geo_module import GeometryEngine  # noqa: E402
from shared.dataset_catalog import (  # noqa: E402
    DatasetCatalogError,
    get_geo_module_sample,
    list_geo_module_samples,
    list_pipeline_cases,
    resolve_dataset_file,
)
from shared.pipeline_runner import run_pipeline_case  # noqa: E402


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


def test_pipeline_case_runs_geo() -> None:
    result = run_pipeline_case("airfoil_naca_0012_cruise")
    _assert(result["geo"] is not None, "pipeline geo stage missing")
    _assert(result["active_stages"] == ["geo"], "only geo is live")
    _assert(result["pending_stages"][0] == "math", "next pending stage should be math")
    _assert(result["math"] is None, "math stage must still be empty")


def test_all_pipeline_cases_produce_geo() -> None:
    engine = GeometryEngine()
    for case in list_pipeline_cases():
        result = run_pipeline_case(case.id, engine=engine)
        _assert(result["geo"] is not None, f"{case.id} produced no geometry")


def test_dataset_path_escape_rejected() -> None:
    try:
        resolve_dataset_file("../.cursorrules")
    except DatasetCatalogError:
        return
    raise AssertionError("expected DatasetCatalogError for path escape")


def main() -> None:
    test_geo_catalog_titles()
    test_svg_module_file_sample()
    test_pipeline_case_runs_geo()
    test_all_pipeline_cases_produce_geo()
    test_dataset_path_escape_rejected()
    print("dataset catalog smoke tests passed")


if __name__ == "__main__":
    main()
