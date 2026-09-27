from __future__ import annotations

from typing import Optional

import numpy as np

from .math_skeleton import (
    DomainExtents2D,
    MeshInput,
    MeshOutput,
    MeshParameters,
    GeometryOutput,
    InvalidMeshConfigError,
    validate_mesh_input,
    validate_mesh_output,
)
from .panel_2d import build_panel_metrics_2d, cosine_cluster_airfoil_nodes, build_ibl_stations_1d
from .grid_2d import build_cartesian_grid_2d
from .panel_3d import build_panel_metrics_3d, refine_surface_triangles


def _chord_relative_domain(
    bbox: dict[str, float],
    domain: DomainExtents2D,
    chord: float,
) -> DomainExtents2D:
    """Place the chord-relative farfield on this airfoil.

    ``domain`` stores multiples of chord. The leading edge anchors x and the
    section mid-height anchors y, so a chord other than 1 still gets two
    chords of upstream fetch under the default box.
    """
    x_le = float(bbox["xmin"])
    y_mid = 0.5 * (float(bbox["ymin"]) + float(bbox["ymax"]))
    placed = DomainExtents2D(
        x_min=x_le + float(domain["x_min"]) * chord,
        x_max=x_le + float(domain["x_max"]) * chord,
        y_min=y_mid + float(domain["y_min"]) * chord,
        y_max=y_mid + float(domain["y_max"]) * chord,
    )
    if not (
        placed["x_min"] < float(bbox["xmin"])
        and placed["x_max"] > float(bbox["xmax"])
        and placed["y_min"] < float(bbox["ymin"])
        and placed["y_max"] > float(bbox["ymax"])
    ):
        raise InvalidMeshConfigError(
            "Cartesian domain must strictly contain the geometry. "
            "domain is chord-relative: x = x_le + x_* * chord, "
            "y = y_mid + y_* * chord."
        )
    return placed


def _length_aspect(lengths: np.ndarray) -> tuple[float, float]:
    neighbor = np.minimum(np.roll(lengths, 1), np.roll(lengths, -1))
    aspect = np.maximum(lengths, neighbor) / np.minimum(lengths, neighbor)
    return float(np.min(aspect)), float(np.max(aspect))


def _mean_outward_alignment(
    points: np.ndarray,
    normals: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    if weights is None:
        center = np.mean(points, axis=0)
    else:
        center = np.average(points, axis=0, weights=weights)
    radial = points - center
    dist = np.linalg.norm(radial, axis=1)
    unit = np.divide(
        radial,
        dist[:, None],
        out=np.zeros_like(radial),
        where=dist[:, None] > 0.0,
    )
    dots = np.sum(normals * unit, axis=1)
    ok = dist > 0.0
    if not np.any(ok):
        return 1.0
    return float(np.mean(dots[ok]))


def _mean_open_normal_coherence(normals: np.ndarray, areas: np.ndarray) -> float:
    weight = areas / np.sum(areas)
    moment = (normals * weight[:, None]).T @ normals
    _eigvals, eigvecs = np.linalg.eigh(moment)
    axis = eigvecs[:, int(np.argmax(_eigvals))]
    return float(np.sum(weight * np.abs(normals @ axis)))


class MeshEngine:
    """Public façade for Module 2 geometry discretization and panel metrics."""

    VERSION = "2.0.0"

    def from_geometry(
        self,
        geometry: GeometryOutput,
        parameters: Optional[MeshParameters] = None,
    ) -> MeshOutput:
        payload = validate_mesh_input(
            {"geometry": geometry, "parameters": parameters or {}}
        )
        return self.from_mesh_input(payload)

    def from_mesh_input(self, mesh_input: MeshInput) -> MeshOutput:
        data = validate_mesh_input(mesh_input)
        geometry = data["geometry"]
        params = data["parameters"]
        mode = geometry["mode"]

        if mode == "2D":
            points = np.asarray(geometry["points_2d"], dtype=np.float64).copy()
            if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 4:
                raise InvalidMeshConfigError("2D geometry must contain at least 4 (N,2) points.")
            if not geometry["is_closed"]:
                raise InvalidMeshConfigError("Module 2 2D panelization requires a closed contour.")

            bbox = geometry["bounding_box"]
            chord = float(bbox["xmax"] - bbox["xmin"])
            if not np.isfinite(chord) or chord <= 0.0:
                raise InvalidMeshConfigError("2D geometry requires positive x extent/chord.")

            domain = _chord_relative_domain(bbox, params["domain"], chord)

            n_panels = params["n_panels"]
            if n_panels is None:
                n_panels = max(16, points.shape[0] - 1)

            nodes = cosine_cluster_airfoil_nodes(
                points,
                int(n_panels),
                scheme=params["clustering"],
                le_cluster_factor=float(params["le_cluster_factor"]),
                te_cluster_factor=float(params["te_cluster_factor"]),
            )
            panel = build_panel_metrics_2d(
                nodes,
                min_panel_length=float(params["min_panel_length"]),
                clustering=params["clustering"],
            )
            # Keep the actual geometric chord in the contract.
            panel["chord"] = chord

            grid = build_cartesian_grid_2d(
                points,
                n_x=int(params["n_x"]),
                n_y=int(params["n_y"]),
                domain=domain,
            )
            ibl = build_ibl_stations_1d(panel)

            lengths = panel["length"]
            min_aspect, max_aspect = _length_aspect(lengths)
            quality = {
                "n_panels": int(lengths.size),
                "min_length": float(np.min(lengths)),
                "max_length": float(np.max(lengths)),
                "min_area": 0.0,  # 2D line panels have no surface area metric
                "max_area": 0.0,
                "min_aspect_ratio": min_aspect,
                "max_aspect_ratio": max_aspect,
                "mean_normal_alignment": _mean_outward_alignment(
                    panel["collocation"], panel["normal"]
                ),
                "n_degenerate_rejected": 0,
            }
            out: MeshOutput = {
                "mode": "2D",
                "shape_name": str(geometry["shape_name"]),
                "source_geometry_closed": bool(geometry["is_closed"]),
                "panel_2d": panel,
                "cartesian_2d": grid,
                "ibl_1d": ibl,
                "panel_3d": None,
                "quality": quality,
            }
            return validate_mesh_output(out)

        if mode == "3D":
            facets = refine_surface_triangles(
                np.asarray(geometry["facets_3d"], dtype=np.float64),
                int(params["surface_density"]),
            )
            panel = build_panel_metrics_3d(
                facets,
                normal_orientation=params["normal_orientation"],
                min_panel_area=float(params["min_panel_area"]),
                require_closed=bool(params["require_closed_3d"]),
                is_closed_hint=bool(geometry["is_closed"]),
            )
            areas = panel["areas"]
            edges = np.stack(
                (
                    np.linalg.norm(facets[:, 1] - facets[:, 0], axis=1),
                    np.linalg.norm(facets[:, 2] - facets[:, 1], axis=1),
                    np.linalg.norm(facets[:, 0] - facets[:, 2], axis=1),
                ),
                axis=1,
            )
            shortest = np.min(edges, axis=1)
            longest = np.max(edges, axis=1)
            aspect = longest / np.maximum(shortest, np.finfo(np.float64).tiny)
            if panel["is_closed"]:
                alignment = _mean_outward_alignment(
                    panel["centroids"], panel["normals"], panel["areas"]
                )
            else:
                alignment = _mean_open_normal_coherence(panel["normals"], panel["areas"])
            out = {
                "mode": "3D",
                "shape_name": str(geometry["shape_name"]),
                "source_geometry_closed": bool(geometry["is_closed"]),
                "panel_2d": None,
                "cartesian_2d": None,
                "ibl_1d": None,
                "panel_3d": panel,
                "quality": {
                    "n_panels": int(areas.size),
                    "min_length": float(np.min(shortest)),
                    "max_length": float(np.max(longest)),
                    "min_area": float(np.min(areas)),
                    "max_area": float(np.max(areas)),
                    "min_aspect_ratio": float(np.min(aspect)),
                    "max_aspect_ratio": float(np.max(aspect)),
                    "mean_normal_alignment": alignment,
                    "n_degenerate_rejected": 0,
                },
            }
            return validate_mesh_output(out)

        raise InvalidMeshConfigError(f"Unsupported geometry mode: {mode!r}")


if __name__ == "__main__":
    # Standalone smoke test using the implementation helpers directly.
    theta = np.linspace(0.0, 2.0 * np.pi, 65)
    circle = np.column_stack((0.5 * (1.0 + np.cos(theta)), 0.5 * np.sin(theta)))
    nodes = cosine_cluster_airfoil_nodes(circle, 32)
    p2 = build_panel_metrics_2d(nodes)
    print("2D nodes:", p2["nodes"].shape, "collocation:", p2["collocation"].shape)

    rng = np.random.default_rng(7)
    a = rng.random((6, 3))
    b = a + rng.normal(scale=0.3, size=(6, 3))
    c = a + rng.normal(scale=0.3, size=(6, 3))
    f3 = np.stack((a, b, c), axis=1)
    p3 = build_panel_metrics_3d(f3)
    print("3D facets:", p3["facets"].shape, "normals:", p3["normals"].shape)
