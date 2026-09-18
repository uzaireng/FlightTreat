"""
geo_skeleton.py

Downstream data contract for Module 1 (Geometry Engine).

GeometryEngine and all public generators/parsers must emit GeometryOutput.
Downstream modules must treat arrays as immutable (fresh copies).
"""

from __future__ import annotations

from typing import Literal, Optional, TypedDict

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class BoundingBox(TypedDict):
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float


class GeometryOutput(TypedDict):
    """Strict Module 1 output consumed by math_module and the workbench."""

    mode: Literal["2D", "3D"]
    shape_name: str
    points_2d: Optional[np.ndarray]
    facets_3d: Optional[np.ndarray]
    num_elements: int
    bounding_box: BoundingBox
    is_closed: bool


class GeometryOutputModel(BaseModel):
    """Runtime validator for GeometryOutput (Pydantic + TypedDict pair)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    mode: Literal["2D", "3D"]
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
            if not np.all(np.isfinite(points)):
                raise ValueError("points_2d contains non-finite coordinates.")
            if int(self.num_elements) != int(points.shape[0]):
                raise ValueError("num_elements must equal len(points_2d).")

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


def validate_geometry_output(payload: GeometryOutput) -> GeometryOutput:
    """Validate and return a defensive copy of a GeometryOutput payload."""

    return GeometryOutputModel.model_validate(payload).to_typed_dict()
