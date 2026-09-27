"""Mathematics & Mesh Engine (Module 2)."""

from .grid_2d import build_cartesian_grid_2d
from .math_skeleton import (
    BoundaryRegion,
    CartesianGrid2D,
    DegeneratePanelError,
    DomainExtents2D,
    IBLStationMesh1D,
    InvertedNormalError,
    InvalidMeshConfigError,
    MeshEngineError,
    MeshEngineProtocol,
    MeshInput,
    MeshOutput,
    MeshOutputModel,
    MeshParameters,
    MeshParametersModel,
    MeshQualityMetrics,
    MeshSingularityError,
    OpenSurfaceError,
    PanelMetrics2D,
    PanelMetrics3D,
    validate_mesh_input,
    validate_mesh_output,
)
from .mesh_engine import MeshEngine
from .panel_2d import (
    build_ibl_stations_1d,
    build_panel_metrics_2d,
    cosine_cluster_airfoil_nodes,
)
from .panel_3d import build_panel_metrics_3d

__all__ = [
    "MeshEngine",
    "build_cartesian_grid_2d",
    "build_ibl_stations_1d",
    "build_panel_metrics_2d",
    "build_panel_metrics_3d",
    "cosine_cluster_airfoil_nodes",
    "BoundaryRegion",
    "CartesianGrid2D",
    "DegeneratePanelError",
    "DomainExtents2D",
    "IBLStationMesh1D",
    "InvertedNormalError",
    "InvalidMeshConfigError",
    "MeshEngineError",
    "MeshEngineProtocol",
    "MeshInput",
    "MeshOutput",
    "MeshOutputModel",
    "MeshParameters",
    "MeshParametersModel",
    "MeshQualityMetrics",
    "MeshSingularityError",
    "OpenSurfaceError",
    "PanelMetrics2D",
    "PanelMetrics3D",
    "validate_mesh_input",
    "validate_mesh_output",
]
