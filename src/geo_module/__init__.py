"""
geo_module public API.
"""

from .geo_generators import (
    GeometryGenerationError,
    generate_3d_parametric_wing,
    generate_cylinder,
    generate_flat_plate,
    generate_naca_4digit,
)
from .geo_inputs import (
    GeometryInputError,
    NACAInput,
    WingInput,
    get_builtin_sample,
    list_builtin_samples,
    validate_file_input,
)
from .geo_parsers import (
    GeometryParseError,
    InvalidGeometryError,
    is_watertight,
    parse_2d_dxf_svg,
    parse_3d_stl_obj,
)
from .geo_skeleton import (
    BoundingBox,
    GeometryOutput,
    GeometryOutputModel,
    validate_geometry_output,
)
from .geometry_module import (
    GeometryEngine,
    GeometryEngineError,
    GeometryProcessingError,
    InvalidGeometryInputError,
    UnsupportedGeometryError,
)

__all__ = [
    "BoundingBox",
    "GeometryEngine",
    "GeometryEngineError",
    "GeometryGenerationError",
    "GeometryInputError",
    "GeometryOutput",
    "GeometryOutputModel",
    "GeometryParseError",
    "GeometryProcessingError",
    "InvalidGeometryError",
    "InvalidGeometryInputError",
    "NACAInput",
    "UnsupportedGeometryError",
    "WingInput",
    "generate_3d_parametric_wing",
    "generate_cylinder",
    "generate_flat_plate",
    "generate_naca_4digit",
    "get_builtin_sample",
    "is_watertight",
    "list_builtin_samples",
    "parse_2d_dxf_svg",
    "parse_3d_stl_obj",
    "validate_file_input",
    "validate_geometry_output",
]
