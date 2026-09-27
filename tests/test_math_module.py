"""Smoke tests for Module 2 MeshEngine (no pytest required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from geo_module import GeometryEngine  # noqa: E402
from math_module import (  # noqa: E402
    BoundaryRegion,
    DegeneratePanelError,
    InvalidMeshConfigError,
    MeshEngine,
    OpenSurfaceError,
    build_panel_metrics_3d,
    cosine_cluster_airfoil_nodes,
)


def _assert(cond: bool, message: str) -> None:
    if not cond:
        raise AssertionError(message)


def _spacings(nodes: np.ndarray) -> tuple[float, float, float]:
    length = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    le = int(np.argmin(nodes[:, 0]))
    le_len = float(min(length[le - 1], length[le % length.size]))
    te_len = float(min(length[0], length[-1]))
    mid = float(np.max(length))
    return le_len, te_len, mid


def test_clustering_schemes_and_factors() -> None:
    pts = GeometryEngine().generate_naca("0012", num_points=80)["points_2d"]
    assert pts is not None
    common = dict(le_cluster_factor=0.0, te_cluster_factor=0.0)
    cosine = cosine_cluster_airfoil_nodes(pts, 40, scheme="cosine", **common)
    half = cosine_cluster_airfoil_nodes(pts, 40, scheme="half_cosine", **common)
    _assert(not np.allclose(cosine, half), "half-cosine must differ from cosine")

    le_c, te_c, mid_c = _spacings(cosine)
    le_h, te_h, _mid_h = _spacings(half)
    _assert(le_c < 0.5 * mid_c and te_c < 0.5 * mid_c, "cosine must pack both LE and TE")
    _assert(le_h < te_h, "half-cosine must pack the LE more than the TE")

    packed_le = cosine_cluster_airfoil_nodes(
        pts, 40, scheme="cosine", le_cluster_factor=8.0, te_cluster_factor=0.0
    )
    packed_te = cosine_cluster_airfoil_nodes(
        pts, 40, scheme="cosine", le_cluster_factor=0.0, te_cluster_factor=8.0
    )
    le_hi, te_same, _ = _spacings(packed_le)
    le_same, te_hi, _ = _spacings(packed_te)
    _assert(le_hi < 0.5 * le_c, "le_cluster_factor must shorten LE panels")
    _assert(abs(te_same - te_c) / te_c < 0.05, "LE factor must leave TE spacing alone")
    _assert(te_hi < 0.5 * te_c, "te_cluster_factor must shorten TE panels")
    _assert(abs(le_same - le_c) / le_c < 0.05, "TE factor must leave LE spacing alone")

    strong = cosine_cluster_airfoil_nodes(
        pts, 40, scheme="cosine", le_cluster_factor=80.0, te_cluster_factor=80.0
    )
    _assert(float(np.min(np.linalg.norm(np.diff(strong, axis=0), axis=1))) > 1e-12,
            "legal cluster factors must not collapse panels")


def test_ibl_upper_lower_follows_geometry() -> None:
    geo = GeometryEngine().generate_naca("4412", num_points=100)
    params = {
        "n_panels": 48,
        "n_x": 32,
        "n_y": 24,
        "clustering": "half_cosine",
        "le_cluster_factor": 0.0,
        "te_cluster_factor": 0.0,
    }
    fwd = MeshEngine().from_geometry(geo, params)
    pts = geo["points_2d"]
    assert pts is not None
    rev_geo = dict(geo)
    rev_geo["points_2d"] = pts[::-1].copy()
    rev = MeshEngine().from_geometry(rev_geo, params)

    for mesh in (fwd, rev):
        panel = mesh["panel_2d"]
        ibl = mesh["ibl_1d"]
        assert panel is not None and ibl is not None
        _assert(panel["clustering"] == "half_cosine", "clustering scheme was dropped")
        _assert(float(np.mean(ibl["xy_upper"][:, 1])) > float(np.mean(ibl["xy_lower"][:, 1])),
                "upper IBL side must sit above the lower side")
        _assert(mesh["quality"]["max_aspect_ratio"] > 1.0, "aspect ratio must reflect panel grading")
        _assert(mesh["quality"]["mean_normal_alignment"] > 0.2, "normals must point out of the section")
        _assert(np.allclose(geo["points_2d"], pts), "meshing must not mutate geometry")


def test_chord_relative_farfield() -> None:
    geo = GeometryEngine().generate_naca("0012", chord=5.0, num_points=60)
    mesh = MeshEngine().from_geometry(geo, {"n_panels": 24, "n_x": 16, "n_y": 12})
    grid = mesh["cartesian_2d"]
    assert grid is not None
    _assert(float(grid["x_nodes"][0]) < -9.0, "default domain must scale with chord")
    _assert(float(grid["x_nodes"][-1]) > 14.0, "default domain must extend downstream of the TE")


def test_closed_body_normals_match_winding() -> None:
    cyl = GeometryEngine().generate_cylinder(radius=0.5, length=2.0, segments=16, rings=4)
    facets = cyl["facets_3d"]
    assert facets is not None
    original = facets.copy()

    def winding_agreement(panel: dict) -> float:
        f = panel["facets"]
        cross = np.cross(f[:, 1] - f[:, 0], f[:, 2] - f[:, 0])
        cross /= np.linalg.norm(cross, axis=1)[:, None]
        return float(np.min(np.sum(cross * panel["normals"], axis=1)))

    outward = build_panel_metrics_3d(
        facets, normal_orientation="outward", is_closed_hint=True
    )
    inward = build_panel_metrics_3d(
        facets, normal_orientation="inward", is_closed_hint=True
    )
    center = outward["centroids"].mean(axis=0)
    radial = outward["centroids"] - center
    radial /= np.linalg.norm(radial, axis=1)[:, None]
    _assert(outward["is_closed"], "cylinder must be closed")
    _assert(winding_agreement(outward) > 0.999, "outward normals must match facet winding")
    _assert(winding_agreement(inward) > 0.999, "inward normals must match facet winding")
    outward_dots = np.sum(outward["normals"] * radial, axis=1)
    inward_dots = np.sum(inward["normals"] * radial, axis=1)
    _assert(float(np.min(outward_dots)) > 0.0, "every cylinder panel must point outward")
    _assert(float(np.mean(outward_dots)) > 0.5, "outward normals must leave the body")
    _assert(float(np.max(inward_dots)) < 0.0, "every cylinder panel must point inward")
    _assert(np.allclose(facets, original), "3D meshing must not mutate the input facets")


def test_open_wing_is_not_forced_closed() -> None:
    wing = GeometryEngine().generate_wing(
        airfoil_code="0012",
        chord=1.0,
        span=2.0,
        num_sections=6,
        airfoil_points=40,
    )
    facets = wing["facets_3d"]
    assert facets is not None
    hinted = build_panel_metrics_3d(facets, is_closed_hint=True, require_closed=False)
    _assert(not hinted["is_closed"], "an open wing hint must not mark the skin closed")
    mesh = MeshEngine().from_geometry(wing)
    assert mesh["panel_3d"] is not None
    _assert(not mesh["panel_3d"]["is_closed"], "engine must keep the wing skin open")
    _assert(mesh["quality"]["mean_normal_alignment"] > 0.8, "wing normals must be coherent")
    try:
        build_panel_metrics_3d(facets, require_closed=True)
    except OpenSurfaceError:
        return
    raise AssertionError("require_closed must reject an open wing")


def test_degenerate_triangle_and_flat_plate() -> None:
    collapsed = np.zeros((1, 3, 3), dtype=np.float64)
    collapsed[0, 1, 0] = 1.0
    try:
        build_panel_metrics_3d(collapsed)
    except DegeneratePanelError:
        pass
    else:
        raise AssertionError("zero-area triangle must raise DegeneratePanelError")

    plate = GeometryEngine().generate_flat_plate(chord=1.0, num_points=40, thickness=1e-4)
    mesh = MeshEngine().from_geometry(plate, {"n_panels": 32, "n_x": 16, "n_y": 12})
    _assert(mesh["quality"]["min_length"] > 1e-6, "flat plate panels collapsed")

    try:
        cosine_cluster_airfoil_nodes(
            plate["points_2d"], 16, le_cluster_factor=float("nan")
        )
    except InvalidMeshConfigError:
        return
    raise AssertionError("non-finite cluster factor must be rejected")


def test_open_contour_rejected() -> None:
    pts = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 0.1], [0.0, 0.1]],
        dtype=np.float64,
    )
    try:
        cosine_cluster_airfoil_nodes(pts, 16)
    except InvalidMeshConfigError:
        return
    raise AssertionError("an open contour must be rejected")


def test_farfield_mask_tags() -> None:
    geo = GeometryEngine().generate_naca("0012", num_points=80)
    mesh = MeshEngine().from_geometry(geo, {"n_panels": 48, "n_x": 81, "n_y": 81})
    grid = mesh["cartesian_2d"]
    assert grid is not None
    mask = grid["boundary_mask"]
    _assert(int(mask[mask.shape[0] // 2, 0]) == int(BoundaryRegion.INLET), "left boundary must be inlet")
    _assert(int(mask[mask.shape[0] // 2, -1]) == int(BoundaryRegion.OUTLET), "right boundary must be outlet")
    tagged = set(np.unique(mask).tolist())
    _assert(int(BoundaryRegion.AIRFOIL_WALL) in tagged or int(BoundaryRegion.INTERNAL_SOLID) in tagged,
            "airfoil must tag wall or solid cells")


def test_surface_density_splits_edges() -> None:
    wing = GeometryEngine().generate_wing(
        airfoil_code="0012",
        chord=1.0,
        span=2.0,
        num_sections=4,
        airfoil_points=20,
    )
    native = MeshEngine().from_geometry(wing, {"surface_density": 1})
    fine = MeshEngine().from_geometry(wing, {"surface_density": 2})
    _assert(fine["quality"]["n_panels"] == 4 * native["quality"]["n_panels"], "density 2 must quadruple panels")
    _assert(
        fine["quality"]["max_length"] < native["quality"]["max_length"] * 0.6,
        "density 2 must shorten the longest edge",
    )
    _assert(
        fine["quality"]["min_length"] < native["quality"]["min_length"],
        "density 2 must shorten the shortest edge",
    )


def test_wing_orientation_stays_fast() -> None:
    import time

    wing = GeometryEngine().generate_wing(
        airfoil_code="0012",
        chord=1.0,
        span=6.0,
        num_sections=36,
        airfoil_points=64,
    )
    started = time.perf_counter()
    mesh = MeshEngine().from_geometry(wing)
    elapsed = time.perf_counter() - started
    assert mesh["panel_3d"] is not None
    _assert(mesh["panel_3d"]["normals"].shape[0] == wing["num_elements"], "panel count drifted")
    _assert(elapsed < 2.0, f"3D orientation repair took {elapsed:.2f}s")


def main() -> None:
    test_clustering_schemes_and_factors()
    test_ibl_upper_lower_follows_geometry()
    test_chord_relative_farfield()
    test_closed_body_normals_match_winding()
    test_open_wing_is_not_forced_closed()
    test_degenerate_triangle_and_flat_plate()
    test_open_contour_rejected()
    test_farfield_mask_tags()
    test_surface_density_splits_edges()
    test_wing_orientation_stays_fast()
    print("math_module smoke tests passed")


if __name__ == "__main__":
    main()
