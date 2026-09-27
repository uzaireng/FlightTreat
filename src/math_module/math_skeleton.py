"""
math_skeleton.py

Module 2 (Mathematics & Mesh Engine) data contract.

Designed for the platform solver:
    Viscous-Inviscid Coupled Panel Method
    = 3D surface panels + 2D vortex panels + 1D integral boundary layer.

Module 2 does NOT assemble influence matrices or march the IBL.
It MUST emit every geometric metric Module 3 needs to build
``A · μ = b`` and the 1D IBL stations without re-processing geometry.

All arrays: dtype np.float64 unless noted. No Python grid/panel loops
in the future implementation. Downstream MUST copy arrays.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Literal, Optional, Protocol, TypedDict

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from geo_module.geo_skeleton import GeometryOutput, validate_geometry_output


# ---------------------------------------------------------------------------
# Exceptions (implementation raises these; defined here as the contract)
# ---------------------------------------------------------------------------


class MeshEngineError(Exception):
    """Base exception for Module 2."""


class InvalidMeshConfigError(MeshEngineError, ValueError):
    """Illegal meshing parameters or GeometryOutput that cannot be discretized."""


class DegeneratePanelError(MeshEngineError, ValueError):
    """Zero-area, collapsed, or duplicate-node panel."""


class MeshSingularityError(MeshEngineError, ValueError):
    """Singular panel metric (zero normal, non-invertible local basis)."""


class InvertedNormalError(MeshEngineError, ValueError):
    """Outward-normal orientation failed (inward / inconsistent winding)."""


class OpenSurfaceError(MeshEngineError, ValueError):
    """Closed-body meshing requested on a non-watertight surface."""


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class BoundaryRegion(IntEnum):
    """Integer tags stored in ``boundary_mask`` (shape ``(n_y, n_x)``, int32)."""

    INTERIOR_FLUID = 0
    INLET = 1
    OUTLET = 2
    TOP_FARFIELD = 3
    BOTTOM_FARFIELD = 4
    AIRFOIL_WALL = 5
    INTERNAL_SOLID = 6


ClusteringScheme = Literal["cosine", "half_cosine", "uniform"]
MeshMode = Literal["2D", "3D"]
NormalOrientation = Literal["outward", "inward"]


# ---------------------------------------------------------------------------
# Input contract
# ---------------------------------------------------------------------------


class DomainExtents2D(TypedDict):
    """Farfield box.

    Passed to ``build_cartesian_grid_2d`` these limits are absolute geometry
    units. Inside ``MeshParameters`` the same keys are chord-relative:
    ``MeshEngine`` places ``x = x_le + x_* * chord`` and
    ``y = y_mid + y_* * chord``.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float


class MeshParameters(TypedDict, total=False):
    """
    Optional meshing knobs packed beside GeometryOutput.

    Defaults (implementation MUST apply if omitted)
    ----------------------------------------------
    n_x, n_y: 161, 81          Cartesian farfield visualization grid
    n_panels: None             If set, re-discretize 2D contour to this count
    clustering: "cosine"       LE (x/c → 0) and TE (x/c → 1) clustering
    le_cluster_factor: 1.0     Extra LE refinement weight (>= 0)
    te_cluster_factor: 1.0     Extra TE refinement weight (>= 0)
    domain: chord-relative box, default x ∈ [-2, 3], y ∈ [-2.5, 2.5].
            MeshEngine anchors it on the leading edge and section mid-height
            (2 chords upstream of the LE and 2 chords downstream of the TE).
    require_closed_3d: False   True → reject open wing skins
    normal_orientation: "outward"
    surface_density: 1         1 keeps the surface triangles. 2 or 4 splits
                               each triangle once or twice (edges halve each time).
    min_panel_area: 1e-14
    min_panel_length: 1e-12
    """

    n_x: int
    n_y: int
    n_panels: Optional[int]
    clustering: ClusteringScheme
    le_cluster_factor: float
    te_cluster_factor: float
    domain: DomainExtents2D
    require_closed_3d: bool
    normal_orientation: NormalOrientation
    surface_density: int
    min_panel_area: float
    min_panel_length: float


class MeshInput(TypedDict):
    """Primary Module 2 input: validated geometry plus meshing parameters."""

    geometry: GeometryOutput
    parameters: MeshParameters


# ---------------------------------------------------------------------------
# 2D panel / IBL-ready metrics
# ---------------------------------------------------------------------------


class PanelMetrics2D(TypedDict):
    """
    Boundary-fitted vortex-panel discretization of a 2D contour.

    Node / panel convention
    -----------------------
    ``nodes``: shape ``(N, 2)`` — clustered panel *nodes* (closed: nodes[0]==nodes[-1]).
    ``n_panels`` = N-1 if closed, else N-1 still if polyline; implementation
    uses ``P = nodes.shape[0] - 1`` collocation panels.

    Per-panel arrays have length ``P``:

    collocation     (P, 2)   midpoints (control points)
    tangent         (P, 2)   unit tangential ê_s (TE-ward / contour order)
    normal          (P, 2)   unit normal ê_n, outward from solid
    length          (P,)     panel lengths S_i > 0
    angle_beta      (P,)     panel tangent angle β_i = atan2(ty, tx) [rad]
    curvature       (P,)     discrete curvature κ_i [1/length]
    x_over_c        (P,)     collocation x / chord
    arc_length      (P,)     cumulative s at collocation from TE or LE
                             (documented in ``te_first``)

    LE / TE
    -------
    le_index, te_index index into *nodes* (not panels).
    Used by Module 3 for Kutta condition and IBL stagnation split.
    """

    nodes: np.ndarray
    collocation: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    length: np.ndarray
    angle_beta: np.ndarray
    curvature: np.ndarray
    x_over_c: np.ndarray
    arc_length: np.ndarray
    chord: float
    le_index: int
    te_index: int
    is_closed: bool
    clustering: ClusteringScheme
    te_first: bool


class CartesianGrid2D(TypedDict):
    """
    Structured farfield visualization grid. Not a volume CFD mesh.

    ``grid_x``, ``grid_y``, ``boundary_mask`` share shape ``(n_y, n_x)``.
    Mask dtype ``np.int32`` with ``BoundaryRegion`` values.
    ``x_nodes`` shape ``(n_x,)``, ``y_nodes`` shape ``(n_y,)``.
    """

    x_nodes: np.ndarray
    y_nodes: np.ndarray
    grid_x: np.ndarray
    grid_y: np.ndarray
    boundary_mask: np.ndarray
    n_x: int
    n_y: int


class IBLStationMesh1D(TypedDict):
    """
    1D stations along the surface for the integral boundary layer.

    Split at stagnation / LE into upper and lower (or suction/pressure)
    marching directions. Module 3 consumes these without resampling.

    s_upper, s_lower:     arc length from stagnation, shape (N_u,), (N_l,)
    xy_upper, xy_lower:   surface points, shape (N_u, 2), (N_l, 2)
    tangent_upper/lower:  unit ê_s, shape (N_*, 2)
    ds_upper/lower:       segment lengths, shape (N_*-1,)
    """

    s_upper: np.ndarray
    s_lower: np.ndarray
    xy_upper: np.ndarray
    xy_lower: np.ndarray
    tangent_upper: np.ndarray
    tangent_lower: np.ndarray
    ds_upper: np.ndarray
    ds_lower: np.ndarray
    stagnation_xy: np.ndarray  # shape (2,)


# ---------------------------------------------------------------------------
# 3D surface-panel metrics
# ---------------------------------------------------------------------------


class PanelMetrics3D(TypedDict):
    """
    Triangular surface-panel geometry for 3D constant-strength panels.

    facets              (F, 3, 3)  copied from geometry (never mutated)
    centroids           (F, 3)     collocation C_f = mean of vertices
    normals             (F, 3)     unit n̂ = (v1 × v2) / ||v1 × v2||
    tangent_1           (F, 3)     unit t̂_1 in-plane (edge0 direction)
    tangent_2           (F, 3)     unit t̂_2 = n̂ × t̂_1
    areas               (F,)       S_f = 0.5 ||v1 × v2||
    transforms          (F, 3, 3)  rows [t̂_1, t̂_2, n̂]; p_local = T @ (p - C)
    inverse_transforms  (F, 3, 3)  global = C + T.T @ p_local
    """

    facets: np.ndarray
    centroids: np.ndarray
    normals: np.ndarray
    tangent_1: np.ndarray
    tangent_2: np.ndarray
    areas: np.ndarray
    transforms: np.ndarray
    inverse_transforms: np.ndarray
    is_closed: bool
    normal_orientation: NormalOrientation


# ---------------------------------------------------------------------------
# Quality + primary output
# ---------------------------------------------------------------------------


class MeshQualityMetrics(TypedDict):
    n_panels: int
    min_length: float
    max_length: float
    min_area: float
    max_area: float
    min_aspect_ratio: float
    max_aspect_ratio: float
    mean_normal_alignment: float
    n_degenerate_rejected: int


class MeshOutput(TypedDict):
    """
    Primary Module 2 output.

    2D mode: panel_2d, cartesian_2d, ibl_1d populated; panel_3d is None.
    3D mode: panel_3d populated; 2D farfield/IBL may be None (spanwise IBL
    stations are a future optional extension — leave None until specified).
    """

    mode: MeshMode
    shape_name: str
    source_geometry_closed: bool
    panel_2d: Optional[PanelMetrics2D]
    cartesian_2d: Optional[CartesianGrid2D]
    ibl_1d: Optional[IBLStationMesh1D]
    panel_3d: Optional[PanelMetrics3D]
    quality: MeshQualityMetrics


# ---------------------------------------------------------------------------
# Pydantic runtime validators
# ---------------------------------------------------------------------------


def _require_f64(name: str, array: np.ndarray, ndim: int, tail: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {arr.ndim}.")
    if tail and arr.shape[-len(tail) :] != tail:
        raise ValueError(f"{name} trailing shape must be {tail}, got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return np.ascontiguousarray(arr, dtype=np.float64).copy()


class MeshParametersModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_x: int = Field(default=161, ge=8, le=4096)
    n_y: int = Field(default=81, ge=8, le=4096)
    n_panels: Optional[int] = Field(default=None, ge=16, le=100_000)
    clustering: ClusteringScheme = "cosine"
    le_cluster_factor: float = Field(default=1.0, ge=0.0, le=100.0)
    te_cluster_factor: float = Field(default=1.0, ge=0.0, le=100.0)
    x_min: float = -2.0
    x_max: float = 3.0
    y_min: float = -2.5
    y_max: float = 2.5
    require_closed_3d: bool = False
    normal_orientation: NormalOrientation = "outward"
    surface_density: int = Field(default=1, ge=1, le=4)
    min_panel_area: float = Field(default=1e-14, gt=0.0)
    min_panel_length: float = Field(default=1e-12, gt=0.0)

    @model_validator(mode="after")
    def _box(self) -> MeshParametersModel:
        if not (self.x_max > self.x_min and self.y_max > self.y_min):
            raise ValueError("Farfield domain extents must be strictly ordered.")
        if self.surface_density not in (1, 2, 4):
            raise ValueError("surface_density must be 1, 2, or 4.")
        return self

    def to_parameters(self) -> MeshParameters:
        return MeshParameters(
            n_x=int(self.n_x),
            n_y=int(self.n_y),
            n_panels=None if self.n_panels is None else int(self.n_panels),
            clustering=self.clustering,
            le_cluster_factor=float(self.le_cluster_factor),
            te_cluster_factor=float(self.te_cluster_factor),
            domain=DomainExtents2D(
                x_min=float(self.x_min),
                x_max=float(self.x_max),
                y_min=float(self.y_min),
                y_max=float(self.y_max),
            ),
            require_closed_3d=bool(self.require_closed_3d),
            normal_orientation=self.normal_orientation,
            surface_density=int(self.surface_density),
            min_panel_area=float(self.min_panel_area),
            min_panel_length=float(self.min_panel_length),
        )


class MeshOutputModel(BaseModel):
    """Runtime validator for MeshOutput."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    mode: MeshMode
    shape_name: str = Field(min_length=1)
    source_geometry_closed: bool
    panel_2d: Optional[PanelMetrics2D] = None
    cartesian_2d: Optional[CartesianGrid2D] = None
    ibl_1d: Optional[IBLStationMesh1D] = None
    panel_3d: Optional[PanelMetrics3D] = None
    quality: MeshQualityMetrics

    @model_validator(mode="after")
    def _mode_payload(self) -> MeshOutputModel:
        if self.mode == "2D":
            if self.panel_2d is None or self.cartesian_2d is None:
                raise ValueError("2D MeshOutput requires panel_2d and cartesian_2d.")
            if self.panel_3d is not None:
                raise ValueError("2D MeshOutput must set panel_3d to None.")
            _validate_panel_2d(self.panel_2d)
            _validate_cartesian(self.cartesian_2d)
            if self.ibl_1d is not None:
                _validate_ibl(self.ibl_1d)
            return self

        if self.panel_3d is None:
            raise ValueError("3D MeshOutput requires panel_3d.")
        if self.panel_2d is not None or self.cartesian_2d is not None:
            raise ValueError("3D MeshOutput must not carry 2D panel/grid payloads.")
        _validate_panel_3d(self.panel_3d)
        return self

    def to_typed_dict(self) -> MeshOutput:
        return MeshOutput(
            mode=self.mode,
            shape_name=self.shape_name,
            source_geometry_closed=bool(self.source_geometry_closed),
            panel_2d=self.panel_2d,
            cartesian_2d=self.cartesian_2d,
            ibl_1d=self.ibl_1d,
            panel_3d=self.panel_3d,
            quality=dict(self.quality),  # type: ignore[arg-type]
        )


def _validate_panel_2d(p: PanelMetrics2D) -> None:
    nodes = _require_f64("panel_2d.nodes", p["nodes"], 2, (2,))
    if nodes.shape[0] < 4:
        raise ValueError("panel_2d.nodes requires at least 4 vertices.")
    p_count = nodes.shape[0] - 1
    for name, tail in (
        ("collocation", (2,)),
        ("tangent", (2,)),
        ("normal", (2,)),
        ("length", ()),
        ("angle_beta", ()),
        ("curvature", ()),
        ("x_over_c", ()),
        ("arc_length", ()),
    ):
        arr = _require_f64(f"panel_2d.{name}", p[name], 1 if not tail else 2 if tail == (2,) else 1, tail)  # type: ignore[literal-required]
        if arr.shape[0] != p_count:
            raise ValueError(f"panel_2d.{name} length must equal n_panels={p_count}.")
    if float(p["chord"]) <= 0.0 or not np.isfinite(p["chord"]):
        raise ValueError("panel_2d.chord must be finite and > 0.")
    lengths = np.asarray(p["length"], dtype=np.float64)
    if np.any(lengths <= 0.0):
        raise ValueError("panel_2d.length must be strictly positive.")


def _validate_cartesian(g: CartesianGrid2D) -> None:
    nx, ny = int(g["n_x"]), int(g["n_y"])
    x_nodes = _require_f64("x_nodes", g["x_nodes"], 1, ())
    y_nodes = _require_f64("y_nodes", g["y_nodes"], 1, ())
    if x_nodes.shape != (nx,) or y_nodes.shape != (ny,):
        raise ValueError("x_nodes/y_nodes shapes must match n_x/n_y.")
    gx = _require_f64("grid_x", g["grid_x"], 2, ())
    gy = _require_f64("grid_y", g["grid_y"], 2, ())
    if gx.shape != (ny, nx) or gy.shape != (ny, nx):
        raise ValueError("grid_x/grid_y must have shape (n_y, n_x).")
    mask = np.asarray(g["boundary_mask"])
    if mask.shape != (ny, nx) or mask.dtype != np.int32:
        raise ValueError("boundary_mask must be int32 with shape (n_y, n_x).")
    allowed = {int(v) for v in BoundaryRegion}
    uniq = set(np.unique(mask).tolist())
    if not uniq.issubset(allowed):
        raise ValueError("boundary_mask contains unknown BoundaryRegion tags.")


def _validate_ibl(ibl: IBLStationMesh1D) -> None:
    for key in ("s_upper", "s_lower"):
        arr = _require_f64(key, ibl[key], 1, ())  # type: ignore[literal-required]
        if arr.shape[0] < 2:
            raise ValueError(f"{key} requires at least 2 stations.")
    stag = _require_f64("stagnation_xy", ibl["stagnation_xy"], 1, ())
    if stag.shape != (2,):
        raise ValueError("stagnation_xy must have shape (2,).")


def _validate_panel_3d(p: PanelMetrics3D) -> None:
    facets = _require_f64("facets", p["facets"], 3, (3, 3))
    f = facets.shape[0]
    for name, ndim, tail in (
        ("centroids", 2, (3,)),
        ("normals", 2, (3,)),
        ("tangent_1", 2, (3,)),
        ("tangent_2", 2, (3,)),
        ("areas", 1, ()),
        ("transforms", 3, (3, 3)),
        ("inverse_transforms", 3, (3, 3)),
    ):
        arr = _require_f64(name, p[name], ndim, tail)  # type: ignore[literal-required]
        if arr.shape[0] != f:
            raise ValueError(f"panel_3d.{name} leading dim must equal F={f}.")
    areas = np.asarray(p["areas"], dtype=np.float64)
    if np.any(areas <= 0.0):
        raise ValueError("panel_3d.areas must be strictly positive.")
    nrm = np.asarray(p["normals"], dtype=np.float64)
    nlen = np.linalg.norm(nrm, axis=1)
    if np.any(np.abs(nlen - 1.0) > 1e-6):
        raise ValueError("panel_3d.normals must be unit vectors.")


def validate_mesh_output(payload: MeshOutput) -> MeshOutput:
    """Validate and return MeshOutput (nested arrays copied by field validators)."""

    return MeshOutputModel.model_validate(payload).to_typed_dict()


def validate_mesh_input(payload: MeshInput) -> MeshInput:
    """Reject invalid GeometryOutput / meshing parameters before discretization."""

    geometry = validate_geometry_output(payload["geometry"])
    raw = payload.get("parameters") or {}
    domain = raw.get("domain") or {}
    merged = {
        k: v
        for k, v in raw.items()
        if k != "domain" and v is not None
    }
    merged.update({k: domain[k] for k in ("x_min", "x_max", "y_min", "y_max") if k in domain})
    params = MeshParametersModel.model_validate(merged).to_parameters()
    if geometry["mode"] == "2D" and geometry["points_2d"] is None:
        raise InvalidMeshConfigError("2D geometry missing points_2d.")
    if geometry["mode"] == "3D" and geometry["facets_3d"] is None:
        raise InvalidMeshConfigError("3D geometry missing facets_3d.")
    if params.get("require_closed_3d") and geometry["mode"] == "3D" and not geometry["is_closed"]:
        raise OpenSurfaceError(
            "require_closed_3d=True but GeometryOutput.is_closed is False."
        )
    return MeshInput(geometry=geometry, parameters=params)


# ---------------------------------------------------------------------------
# Engine / helper signatures (implementation in future math_module files)
# ---------------------------------------------------------------------------


def cosine_cluster_airfoil_nodes(
    points_2d: np.ndarray,
    n_panels: int,
    *,
    scheme: ClusteringScheme = "cosine",
    le_cluster_factor: float = 1.0,
    te_cluster_factor: float = 1.0,
) -> np.ndarray:
    """
    Re-space a closed (N, 2) contour with cosine / half-cosine clustering
    at LE (x/c → 0) and TE (x/c → 1).

    Returns clustered nodes shape ``(n_panels + 1, 2)``, dtype np.float64,
    closed (first == last).
    """

    ...


def build_panel_metrics_2d(
    nodes: np.ndarray,
    *,
    min_panel_length: float = 1e-12,
) -> PanelMetrics2D:
    """
    From clustered nodes, compute β_i, S_i, collocation, curvature, normals.
    Vectorized. Raises DegeneratePanelError on S_i <= min_panel_length.
    """

    ...


def build_cartesian_grid_2d(
    points_2d: np.ndarray,
    *,
    n_x: int,
    n_y: int,
    domain: DomainExtents2D,
) -> CartesianGrid2D:
    """
    Structured (n_y, n_x) grid and BoundaryRegion mask
    (INLET, OUTLET, TOP_FARFIELD, BOTTOM_FARFIELD, AIRFOIL_WALL, INTERNAL_SOLID).
    """

    ...


def build_ibl_stations_1d(panel: PanelMetrics2D) -> IBLStationMesh1D:
    """Split surface at LE/stagnation into upper/lower 1D IBL meshes."""

    ...


def build_panel_metrics_3d(
    facets: np.ndarray,
    *,
    normal_orientation: NormalOrientation = "outward",
    min_panel_area: float = 1e-14,
    require_closed: bool = False,
    is_closed_hint: bool = False,
) -> PanelMetrics3D:
    """
    Centroids, n̂ = v1 × v2, t̂_1, t̂_2, S_f, T_f.

    Raises DegeneratePanelError on zero-area facets.
    Raises MeshSingularityError if a local basis is not orthonormal.
    Raises InvertedNormalError if orientation repair fails.
    Raises OpenSurfaceError if require_closed and the skin is open.
    """

    ...


class MeshEngineProtocol(Protocol):
    """Public Module 2 façade. Implementation will live in mesh_engine.py."""

    VERSION: str

    def from_geometry(
        self,
        geometry: GeometryOutput,
        parameters: Optional[MeshParameters] = None,
    ) -> MeshOutput: ...

    def from_mesh_input(self, mesh_input: MeshInput) -> MeshOutput: ...
