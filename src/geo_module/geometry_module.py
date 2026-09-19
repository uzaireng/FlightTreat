"""
geometry_module.py

Public GeometryEngine façade.

This is the module boundary consumed by FastAPI, Streamlit and downstream
CFD modules.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .geo_generators import (
    generate_3d_parametric_wing,
    generate_cylinder,
    generate_flat_plate,
    generate_naca_4digit,
)
from shared.dataset_catalog import DatasetCatalogError, resolve_dataset_file

from .geo_inputs import (
    GeometryInputError,
    NACAInput,
    WingInput,
    get_builtin_sample,
    validate_file_input,
)
from .geo_parsers import (
    GeometryParseError,
    InvalidGeometryError,
    is_watertight,
    mesh_bounding_box,
    parse_2d_dxf_svg,
    parse_3d_stl_obj,
)
from .geo_skeleton import GeometryOutput, validate_geometry_output


class GeometryEngineError(Exception):
    """Base exception exposed by GeometryEngine."""


class InvalidGeometryInputError(GeometryEngineError):
    """Invalid user geometry input."""


class UnsupportedGeometryError(GeometryEngineError):
    """Unsupported geometry operation or format."""


class GeometryProcessingError(GeometryEngineError):
    """Safe wrapper around geometry processing failures."""


def _empty_bbox() -> Dict[str, float]:
    return {
        "xmin": 0.0,
        "xmax": 0.0,
        "ymin": 0.0,
        "ymax": 0.0,
        "zmin": 0.0,
        "zmax": 0.0,
    }


def _bbox_2d(
    points: np.ndarray,
) -> Dict[str, float]:

    points = np.asarray(
        points,
        dtype=np.float64,
    )

    mins = np.min(
        points,
        axis=0,
    )

    maxs = np.max(
        points,
        axis=0,
    )

    return {
        "xmin": float(mins[0]),
        "xmax": float(maxs[0]),
        "ymin": float(mins[1]),
        "ymax": float(maxs[1]),
        "zmin": 0.0,
        "zmax": 0.0,
    }


def _is_closed_2d(
    points: np.ndarray,
    tolerance: float = 1e-9,
) -> bool:

    if points.shape[0] < 3:
        return False

    return bool(
        np.linalg.norm(
            points[0] - points[-1]
        ) <= tolerance
    )


class GeometryEngine:
    """
    Main public Geometry Engine.

    Typical Streamlit usage:

        engine = GeometryEngine()

        result = engine.generate_naca(
            code="4412",
            chord=1.0,
            num_points=200,
        )

        st.plotly_chart(...)
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        *,
        topology_tolerance: float = 1e-9,
    ) -> None:

        if not np.isfinite(topology_tolerance):
            raise InvalidGeometryInputError(
                "Topology tolerance must be finite."
            )

        if topology_tolerance <= 0.0:
            raise InvalidGeometryInputError(
                "Topology tolerance must be greater than zero."
            )

        self.topology_tolerance = float(
            topology_tolerance
        )

    # ======================================================================
    # NACA
    # ======================================================================

    def generate_naca(
        self,
        code: str,
        *,
        chord: float = 1.0,
        num_points: int = 100,
    ) -> GeometryOutput:

        try:
            params = NACAInput(
                code=code,
                chord=chord,
                num_points=num_points,
            )

            points = generate_naca_4digit(
                params.code,
                params.num_points,
                params.chord,
            )

            return self._build_2d_output(
                shape_name=f"NACA_{params.code}",
                points=points,
            )

        except (
            GeometryInputError,
            ValueError,
            FloatingPointError,
        ) as exc:

            raise InvalidGeometryInputError(
                str(exc)
            ) from exc

    # ======================================================================
    # Flat plate
    # ======================================================================

    def generate_flat_plate(
        self,
        *,
        chord: float = 1.0,
        num_points: int = 100,
        thickness: float | None = None,
    ) -> GeometryOutput:

        try:
            points = generate_flat_plate(
                chord=chord,
                num_points=num_points,
                thickness=thickness,
            )

            return self._build_2d_output(
                shape_name="Flat_Plate",
                points=points,
            )

        except (
            ValueError,
            FloatingPointError,
        ) as exc:

            raise InvalidGeometryInputError(
                str(exc)
            ) from exc

    # ======================================================================
    # Parametric wing
    # ======================================================================

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
    ) -> GeometryOutput:

        try:
            params = WingInput(
                chord=chord,
                span=span,
                sweep_deg=sweep_deg,
                dihedral_deg=dihedral_deg,
                num_sections=num_sections,
            )

            airfoil = generate_naca_4digit(
                airfoil_code,
                num_points=airfoil_points,
                chord=params.chord,
            )

            facets = generate_3d_parametric_wing(
                airfoil_pts=airfoil,
                span=params.span,
                sweep_deg=params.sweep_deg,
                dihedral_deg=params.dihedral_deg,
                num_sections=params.num_sections,
            )

            return self._build_3d_output(
                shape_name=f"Wing_NACA_{airfoil_code}",
                facets=facets,
                closed=False,
            )

        except (
            GeometryInputError,
            ValueError,
            FloatingPointError,
        ) as exc:

            raise InvalidGeometryInputError(
                str(exc)
            ) from exc

    # ======================================================================
    # Cylinder sample
    # ======================================================================

    def generate_cylinder(
        self,
        *,
        radius: float = 0.5,
        length: float = 2.0,
        segments: int = 48,
        rings: int = 12,
    ) -> GeometryOutput:

        try:
            facets = generate_cylinder(
                radius=radius,
                length=length,
                segments=segments,
                rings=rings,
            )

            closed = is_watertight(
                facets,
                tolerance=self.topology_tolerance,
            )

            return self._build_3d_output(
                shape_name="Cylinder",
                facets=facets,
                closed=closed,
            )

        except (
            ValueError,
            FloatingPointError,
        ) as exc:

            raise InvalidGeometryInputError(
                str(exc)
            ) from exc

    # ======================================================================
    # 2D file import
    # ======================================================================

    def import_2d(
        self,
        file_content: bytes,
        file_type: str,
        *,
        shape_name: str = "Imported_2D",
    ) -> GeometryOutput:

        try:
            file_content, file_type = validate_file_input(
                file_content,
                file_type,
            )
            points = parse_2d_dxf_svg(
                file_content=file_content,
                file_type=file_type,
            )

            return self._build_2d_output(
                shape_name=shape_name,
                points=points,
            )

        except (
            GeometryParseError,
            InvalidGeometryError,
            ValueError,
        ) as exc:

            raise InvalidGeometryInputError(
                str(exc)
            ) from exc

    # ======================================================================
    # 3D file import
    # ======================================================================

    def import_3d(
        self,
        file_content: bytes,
        file_type: str,
        *,
        shape_name: str = "Imported_3D",
    ) -> GeometryOutput:

        try:
            file_content, file_type = validate_file_input(
                file_content,
                file_type,
            )
            facets = parse_3d_stl_obj(
                file_content=file_content,
                file_type=file_type,
            )

            closed = is_watertight(
                facets,
                tolerance=self.topology_tolerance,
            )

            return self._build_3d_output(
                shape_name=shape_name,
                facets=facets,
                closed=closed,
            )

        except (
            GeometryParseError,
            InvalidGeometryError,
            ValueError,
        ) as exc:

            raise InvalidGeometryInputError(
                str(exc)
            ) from exc

    # ======================================================================
    # Built-in samples
    # ======================================================================

    def generate_from_spec(
        self,
        spec: Dict[str, Any],
    ) -> GeometryOutput:
        """Materialize a catalog geometry spec (module sample or pipeline geo stage)."""

        if not isinstance(spec, dict) or "type" not in spec:
            raise InvalidGeometryInputError(
                "Geometry spec must be a mapping with a 'type' field."
            )

        sample_type = spec["type"]

        try:
            if sample_type == "naca":
                return self.generate_naca(
                    code=str(spec["code"]),
                    chord=float(spec.get("chord", 1.0)),
                    num_points=int(spec.get("num_points", 100)),
                )

            if sample_type == "flat_plate":
                return self.generate_flat_plate(
                    chord=float(spec.get("chord", 1.0)),
                    num_points=int(spec.get("num_points", 100)),
                    thickness=spec.get("thickness"),
                )

            if sample_type == "wing":
                return self.generate_wing(
                    airfoil_code=str(spec.get("airfoil", "0012")),
                    chord=float(spec.get("chord", 1.0)),
                    span=float(spec.get("span", 5.0)),
                    sweep_deg=float(spec.get("sweep_deg", 0.0)),
                    dihedral_deg=float(spec.get("dihedral_deg", 0.0)),
                    num_sections=int(spec.get("num_sections", 10)),
                    airfoil_points=int(spec.get("airfoil_points", 100)),
                )

            if sample_type == "cylinder":
                return self.generate_cylinder(
                    radius=float(spec.get("radius", 0.5)),
                    length=float(spec.get("length", 2.0)),
                    segments=int(spec.get("segments", 48)),
                    rings=int(spec.get("rings", 12)),
                )

            if sample_type == "file":
                relative = spec.get("path")
                file_type = spec.get("file_type")
                if not relative or not file_type:
                    raise InvalidGeometryInputError(
                        "File geometry spec requires 'path' and 'file_type'."
                    )

                payload = resolve_dataset_file(str(relative)).read_bytes()
                shape_name = str(spec.get("shape_name") or relative)
                mode = str(spec.get("mode") or "").upper()
                file_type_n = str(file_type).strip().lower().lstrip(".")

                if mode == "2D" or file_type_n in {"svg", "dxf"}:
                    return self.import_2d(
                        payload,
                        file_type_n,
                        shape_name=shape_name,
                    )

                if mode == "3D" or file_type_n in {"stl", "obj"}:
                    return self.import_3d(
                        payload,
                        file_type_n,
                        shape_name=shape_name,
                    )

                raise UnsupportedGeometryError(
                    f"Unsupported dataset file type '{file_type}'."
                )

            raise UnsupportedGeometryError(
                f"Unsupported sample type '{sample_type}'."
            )

        except GeometryEngineError:
            raise

        except (GeometryInputError, DatasetCatalogError, KeyError, TypeError, ValueError) as exc:
            raise InvalidGeometryInputError(str(exc)) from exc

    def generate_sample(
        self,
        sample_name: str,
    ) -> GeometryOutput:

        try:
            sample = get_builtin_sample(
                sample_name
            )
            return self.generate_from_spec(sample)

        except GeometryEngineError:
            raise

        except GeometryInputError as exc:
            raise InvalidGeometryInputError(str(exc)) from exc

        except Exception as exc:
            raise GeometryProcessingError(
                "Unable to generate the requested geometry sample."
            ) from exc

    # ======================================================================
    # Internal output builders
    # ======================================================================

    def _build_2d_output(
        self,
        *,
        shape_name: str,
        points: np.ndarray,
    ) -> GeometryOutput:

        points = np.asarray(
            points,
            dtype=np.float64,
        )

        if points.ndim != 2 or points.shape[1] != 2:
            raise GeometryProcessingError(
                "Internal 2D geometry contract violation."
            )

        if not np.all(np.isfinite(points)):
            raise GeometryProcessingError(
                "Internal 2D geometry contains non-finite coordinates."
            )

        return validate_geometry_output(
            GeometryOutput(
                mode="2D",
                shape_name=str(shape_name),
                points_2d=np.ascontiguousarray(
                    points,
                    dtype=np.float64,
                ),
                facets_3d=None,
                num_elements=int(points.shape[0]),
                bounding_box=_bbox_2d(points),
                is_closed=_is_closed_2d(points),
            )
        )

    def _build_3d_output(
        self,
        *,
        shape_name: str,
        facets: np.ndarray,
        closed: Optional[bool] = None,
    ) -> GeometryOutput:

        facets = np.asarray(
            facets,
            dtype=np.float64,
        )

        if facets.ndim != 3 or facets.shape[1:] != (3, 3):
            raise GeometryProcessingError(
                "Internal 3D geometry contract violation."
            )

        if not np.all(np.isfinite(facets)):
            raise GeometryProcessingError(
                "Internal 3D geometry contains non-finite coordinates."
            )

        if closed is None:
            closed = is_watertight(
                facets,
                tolerance=self.topology_tolerance,
            )

        return validate_geometry_output(
            GeometryOutput(
                mode="3D",
                shape_name=str(shape_name),
                points_2d=None,
                facets_3d=np.ascontiguousarray(
                    facets,
                    dtype=np.float64,
                ),
                num_elements=int(facets.shape[0]),
                bounding_box=mesh_bounding_box(facets),
                is_closed=bool(closed),
            )
        )