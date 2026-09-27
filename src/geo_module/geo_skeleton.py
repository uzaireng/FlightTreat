"""
geo_skeleton.py

Module 1 (Geometry Engine) production data contract.

This file is the single source of truth for GeometryOutput and the public
generator/parser signatures consumed by math_module, Streamlit, FastAPI,
and tests.

Rules:
    - No CFD / meshing / solver implementation.
    - NumPy payloads are dtype np.float64.
    - Downstream modules MUST copy arrays; never mutate in place.
    - 2D airfoils: continuous boundary contour, shape (N, 2).
    - 3D meshes: triangular surface facets, shape (F, 3, 3)
      indexing [facet, vertex, xyz].
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Protocol, TypedDict

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


GeometryMode = Literal["2D", "3D"]
Supported2DFileType = Literal["svg", "dxf"]
Supported3DFileType = Literal["stl", "obj"]


class BoundingBox(TypedDict):
    """Axis-aligned extents in geometry units (chord-scaled as produced)."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float


class GeometryOutput(TypedDict):
    """
    Strict Module 1 output consumed by math_module and the workbench.

    Field contracts
    ---------------
    mode:
        ``"2D"`` or ``"3D"``. Exclusive payload: 2D fills ``points_2d`` and
        MUST set ``facets_3d`` to None; 3D fills ``facets_3d`` and MUST set
        ``points_2d`` to None.
    shape_name:
        Non-empty human-readable identifier (e.g. ``"NACA_4412"``).
    points_2d:
        Closed or open polyline. Shape ``(N, 2)``, dtype ``np.float64``.
        Column 0 = x, column 1 = y. ``N >= 3``. For closed airfoils the first
        and last vertices coincide (``is_closed is True``).
    facets_3d:
        Triangle soup. Shape ``(F, 3, 3)``, dtype ``np.float64``.
        ``facets_3d[f, v, k]`` is coordinate ``k`` of vertex ``v`` of facet
        ``f``. ``F >= 1``. Degenerate (zero-area) triangles are forbidden.
    num_elements:
        2D: ``N`` (vertex count of ``points_2d``).
        3D: ``F`` (triangle count of ``facets_3d``).
    bounding_box:
        Inclusive AABB. For 2D, ``zmin = zmax = 0.0``.
    is_closed:
        2D: first/last vertex coincide within topology tolerance.
        3D: every undirected edge appears exactly twice (watertight skin).
        Parametric wing skins are intentionally open at root/tip
        (``is_closed is False``). Imported STL/OBJ must be watertight.
    """

    mode: GeometryMode
    shape_name: str
    points_2d: Optional[np.ndarray]
    facets_3d: Optional[np.ndarray]
    num_elements: int
    bounding_box: BoundingBox
    is_closed: bool


class NACAGeneratorParams(TypedDict):
    """Inputs for ``generate_naca_4digit`` / ``GeometryEngine.generate_naca``."""

    code: str
    chord: float
    num_points: int


class FlatPlateGeneratorParams(TypedDict):
    chord: float
    num_points: int
    thickness: Optional[float]


class WingGeneratorParams(TypedDict):
    airfoil_code: str
    chord: float
    span: float
    sweep_deg: float
    dihedral_deg: float
    num_sections: int
    airfoil_points: int


class CylinderGeneratorParams(TypedDict):
    radius: float
    length: float
    segments: int
    rings: int


class GeometryOutputModel(BaseModel):
    """Runtime validator for GeometryOutput (Pydantic + TypedDict pair)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    mode: GeometryMode
    shape_name: str = Field(min_length=1)
    points_2d: Optional[np.ndarray] = None
    facets_3d: Optional[np.ndarray] = None
    num_elements: int = Field(ge=0)
    bounding_box: BoundingBox
    is_closed: bool

    @model_validator(mode="after")
    def _enforce_mode_payload(self) -> GeometryOutputModel:
        if self.mode == "2D":
            if self.points_2d is None:
                raise ValueError("2D GeometryOutput requires points_2d.")
            if self.facets_3d is not None:
                raise ValueError("2D GeometryOutput must set facets_3d to None.")

            points = np.asarray(self.points_2d, dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 2:
                raise ValueError("points_2d must have shape (N, 2).")
            if points.shape[0] < 3:
                raise ValueError("points_2d requires at least three vertices.")
            if points.dtype != np.float64:
                points = np.asarray(points, dtype=np.float64)
            if not np.all(np.isfinite(points)):
                raise ValueError("points_2d contains non-finite coordinates.")
            if int(self.num_elements) != int(points.shape[0]):
                raise ValueError("num_elements must equal len(points_2d).")
            _validate_bounding_box(self.bounding_box, points_2d=points)

            object.__setattr__(
                self,
                "points_2d",
                np.ascontiguousarray(points, dtype=np.float64).copy(),
            )
            return self

        if self.facets_3d is None:
            raise ValueError("3D GeometryOutput requires facets_3d.")
        if self.points_2d is not None:
            raise ValueError("3D GeometryOutput must set points_2d to None.")

        facets = np.asarray(self.facets_3d, dtype=np.float64)
        if facets.ndim != 3 or facets.shape[1:] != (3, 3):
            raise ValueError("facets_3d must have shape (F, 3, 3).")
        if facets.shape[0] < 1:
            raise ValueError("facets_3d requires at least one triangle.")
        if not np.all(np.isfinite(facets)):
            raise ValueError("facets_3d contains non-finite coordinates.")
        if int(self.num_elements) != int(facets.shape[0]):
            raise ValueError("num_elements must equal the facet count.")
        _validate_bounding_box(self.bounding_box, facets_3d=facets)

        object.__setattr__(
            self,
            "facets_3d",
            np.ascontiguousarray(facets, dtype=np.float64).copy(),
        )
        return self

    def to_typed_dict(self) -> GeometryOutput:
        return GeometryOutput(
            mode=self.mode,
            shape_name=self.shape_name,
            points_2d=None if self.points_2d is None else self.points_2d.copy(),
            facets_3d=None if self.facets_3d is None else self.facets_3d.copy(),
            num_elements=int(self.num_elements),
            bounding_box=dict(self.bounding_box),  # type: ignore[arg-type]
            is_closed=bool(self.is_closed),
        )


def _validate_bounding_box(
    box: BoundingBox,
    *,
    points_2d: Optional[np.ndarray] = None,
    facets_3d: Optional[np.ndarray] = None,
) -> None:
    required = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
    for key in required:
        if key not in box:
            raise ValueError(f"bounding_box missing '{key}'.")
        if not np.isfinite(box[key]):  # type: ignore[literal-required]
            raise ValueError(f"bounding_box['{key}'] must be finite.")
    if box["xmax"] < box["xmin"] or box["ymax"] < box["ymin"] or box["zmax"] < box["zmin"]:
        raise ValueError("bounding_box extents are inverted.")

    if points_2d is not None:
        if box["zmin"] != 0.0 or box["zmax"] != 0.0:
            raise ValueError("2D bounding_box must set zmin = zmax = 0.0.")
        mins = np.min(points_2d, axis=0)
        maxs = np.max(points_2d, axis=0)
        if not (
            np.isclose(box["xmin"], mins[0])
            and np.isclose(box["xmax"], maxs[0])
            and np.isclose(box["ymin"], mins[1])
            and np.isclose(box["ymax"], maxs[1])
        ):
            raise ValueError("2D bounding_box does not match points_2d extents.")

    if facets_3d is not None:
        xyz = facets_3d.reshape(-1, 3)
        mins = np.min(xyz, axis=0)
        maxs = np.max(xyz, axis=0)
        if not (
            np.isclose(box["xmin"], mins[0])
            and np.isclose(box["xmax"], maxs[0])
            and np.isclose(box["ymin"], mins[1])
            and np.isclose(box["ymax"], maxs[1])
            and np.isclose(box["zmin"], mins[2])
            and np.isclose(box["zmax"], maxs[2])
        ):
            raise ValueError("3D bounding_box does not match facets_3d extents.")


def validate_geometry_output(payload: GeometryOutput) -> GeometryOutput:
    """Validate and return a defensive copy of a GeometryOutput payload."""

    return GeometryOutputModel.model_validate(payload).to_typed_dict()


# ---------------------------------------------------------------------------
# Public generator / parser signatures (implemented in geo_generators /
# geo_parsers / geometry_module). Bodies live outside this file.
# ---------------------------------------------------------------------------


def generate_naca_4digit(
    code: str,
    num_points: int = 100,
    chord: float = 1.0,
) -> np.ndarray:
    """
    Vectorized NACA 4-digit airfoil contour.

    Ordering: TE → upper surface → LE → lower surface → TE.
    Cosine spacing in x from leading edge to trailing edge.

    Returns
    -------
    np.ndarray
        Shape ``(N, 2)``, dtype ``np.float64``. Closed contour.
    """

    ...


def generate_flat_plate(
    chord: float = 1.0,
    num_points: int = 100,
    thickness: float | None = None,
) -> np.ndarray:
    """
    Closed 2D flat-plate contour (thin polygon, never a zero-area line).

    Returns
    -------
    np.ndarray
        Shape ``(N, 2)``, dtype ``np.float64``.
    """

    ...


def generate_3d_parametric_wing(
    airfoil_pts: np.ndarray,
    span: float,
    sweep_deg: float,
    dihedral_deg: float,
    num_sections: int = 10,
) -> np.ndarray:
    """
    Extrude a 2D airfoil to a 3D triangular wing *skin* (root/tip uncapped).

    Parameters
    ----------
    airfoil_pts:
        Shape ``(N, 2)``, dtype ``np.float64``.

    Returns
    -------
    np.ndarray
        Shape ``(F, 3, 3)``, dtype ``np.float64``. Not watertight.
    """

    ...


def generate_cylinder(
    radius: float = 0.5,
    length: float = 2.0,
    segments: int = 48,
    rings: int = 12,
) -> np.ndarray:
    """
    Closed cylindrical body, axis along X, with end caps.

    Returns
    -------
    np.ndarray
        Shape ``(F, 3, 3)``, dtype ``np.float64``. Watertight.
    """

    ...


def parse_2d_dxf_svg(
    file_content: bytes,
    file_type: str,
) -> np.ndarray:
    """
    Parse SVG or ASCII DXF into a normalized 2D polyline.

    Output is unit-chord in X, centered at the origin.

    Returns
    -------
    np.ndarray
        Shape ``(N, 2)``, dtype ``np.float64``.
    """

    ...


def parse_3d_stl_obj(
    file_content: bytes,
    file_type: str,
) -> np.ndarray:
    """
    Parse STL or OBJ into a normalized watertight triangle mesh.

    Largest AABB dimension scaled to 1; centroid at origin.
    Open surfaces MUST raise ``InvalidGeometryError``.

    Returns
    -------
    np.ndarray
        Shape ``(F, 3, 3)``, dtype ``np.float64``.
    """

    ...


def is_watertight(
    facets: np.ndarray,
    tolerance: float = 1e-9,
) -> bool:
    """
    True iff every undirected quantized edge occurs exactly twice.

    ``facets`` shape ``(F, 3, 3)``, dtype ``np.float64``.
    """

    ...


def mesh_bounding_box(facets: np.ndarray) -> BoundingBox:
    """AABB of a triangle mesh. ``facets`` shape ``(F, 3, 3)``."""

    ...


def validate_file_input(
    file_content: bytes,
    file_type: str,
    *,
    max_size_mb: float = 50.0,
) -> tuple[bytes, str]:
    """Reject empty / oversized / unknown uploads. Returns ``(bytes, ext)``."""

    ...


class GeometryEngineProtocol(Protocol):
    """
    Public GeometryEngine façade contract (implemented in geometry_module.py).

    All methods return a validated ``GeometryOutput`` (defensive array copies).
    """

    VERSION: str
    topology_tolerance: float

    def generate_naca(
        self,
        code: str,
        *,
        chord: float = 1.0,
        num_points: int = 100,
    ) -> GeometryOutput: ...

    def generate_flat_plate(
        self,
        *,
        chord: float = 1.0,
        num_points: int = 100,
        thickness: float | None = None,
    ) -> GeometryOutput: ...

    def generate_wing(
        self,
        *,
        airfoil_code: str = "0012",
        chord: float = 1.0,
        span: float = 5.0,
        sweep_deg: float = 0.0,
        dihedral_deg: float = 0.0,
        num_sections: int = 10,
        airfoil_points: int = 100,
    ) -> GeometryOutput: ...

    def generate_cylinder(
        self,
        *,
        radius: float = 0.5,
        length: float = 2.0,
        segments: int = 48,
        rings: int = 12,
    ) -> GeometryOutput: ...

    def import_2d(
        self,
        file_content: bytes,
        file_type: str,
        *,
        shape_name: str = "Imported_2D",
    ) -> GeometryOutput: ...

    def import_3d(
        self,
        file_content: bytes,
        file_type: str,
        *,
        shape_name: str = "Imported_3D",
    ) -> GeometryOutput: ...

    def generate_from_spec(self, spec: Dict[str, Any]) -> GeometryOutput: ...

    def generate_sample(self, sample_name: str) -> GeometryOutput: ...
