"""
End-to-end pipeline sample runner.

Live stages: geo → math. Physics, aero, and AI contracts exist
(`physics_skeleton`, `aero_skeleton`). When those implementations land,
extend `run_pipeline_case` so the same case ids keep flowing
GeometryOutput → MeshOutput → PhysicsOutput → AeroOutput → AIOutput.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, TypedDict

import numpy as np

from aero_module.aero_skeleton import AeroOutput
from geo_module.geo_skeleton import GeometryOutput, validate_geometry_output
from geo_module.geometry_module import GeometryEngine, GeometryEngineError
from math_module.math_skeleton import InvalidMeshConfigError, MeshEngineError, MeshOutput
from math_module.mesh_engine import MeshEngine
from physics_module.physics_skeleton import PhysicsOutput

from .dataset_catalog import (
    DatasetCatalogError,
    MathModuleSample,
    PipelineCase,
    get_math_module_sample,
    get_pipeline_case,
    pending_pipeline_stages,
    spec_as_dict,
)


class PipelineRunResult(TypedDict):
    case_id: str
    title: str
    description: str
    active_stages: list[str]
    pending_stages: list[str]
    geo: Optional[GeometryOutput]
    math: Optional[MeshOutput]
    physics: Optional[PhysicsOutput]
    aero: Optional[AeroOutput]
    ai: None
    notes: Dict[str, Any]


class MathSampleRun(TypedDict):
    sample_id: str
    title: str
    description: str
    expect_error: Optional[str]
    geo: Optional[GeometryOutput]
    math: Optional[MeshOutput]
    error_type: Optional[str]
    error_message: Optional[str]
    seconds: float


def _immutable_geometry(geometry: GeometryOutput) -> GeometryOutput:
    """Copy arrays so Module 2 cannot alias the geometry stage output."""

    points = geometry["points_2d"]
    facets = geometry["facets_3d"]
    return {
        "mode": geometry["mode"],
        "shape_name": geometry["shape_name"],
        "points_2d": None if points is None else np.array(points, dtype=np.float64, copy=True),
        "facets_3d": None if facets is None else np.array(facets, dtype=np.float64, copy=True),
        "num_elements": int(geometry["num_elements"]),
        "bounding_box": dict(geometry["bounding_box"]),
        "is_closed": bool(geometry["is_closed"]),
    }


def _geometry_from_facets(facets: np.ndarray) -> GeometryOutput:
    """Wrap a raw triangle list in GeometryOutput so it enters MeshEngine."""

    raw = np.asarray(facets, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[1:] != (3, 3) or raw.shape[0] < 1:
        raise InvalidMeshConfigError("Facet fixtures must have shape (F, 3, 3), F>=1.")
    if not np.all(np.isfinite(raw)):
        raise InvalidMeshConfigError("Facet fixture contains non-finite coordinates.")
    xyz = raw.reshape(-1, 3)
    mins = np.min(xyz, axis=0)
    maxs = np.max(xyz, axis=0)
    return validate_geometry_output(
        {
            "mode": "3D",
            "shape_name": "facet_fixture",
            "points_2d": None,
            "facets_3d": np.ascontiguousarray(raw, dtype=np.float64),
            "num_elements": int(raw.shape[0]),
            "bounding_box": {
                "xmin": float(mins[0]),
                "xmax": float(maxs[0]),
                "ymin": float(mins[1]),
                "ymax": float(maxs[1]),
                "zmin": float(mins[2]),
                "zmax": float(maxs[2]),
            },
            "is_closed": False,
        }
    )


def _mesh_from_geometry(
    geometry: GeometryOutput,
    parameters: Optional[Dict[str, Any]],
    mesh_engine: MeshEngine,
) -> MeshOutput:
    params = dict(parameters or {})
    return mesh_engine.from_geometry(_immutable_geometry(geometry), params)


def run_pipeline_case(
    name_or_id: str,
    *,
    engine: Optional[GeometryEngine] = None,
    mesh_engine: Optional[MeshEngine] = None,
    math_overrides: Optional[Dict[str, Any]] = None,
) -> PipelineRunResult:
    """
    Execute every *currently implemented* stage of a pipeline case.

    Downstream modules must append here; they must not fork a second
    catalog of user journeys. Geometry arrays are copied before meshing.

    ``math_overrides`` layers user-tuned meshing parameters on top of the
    stored ``stages.math`` payload. The catalog file is never mutated.
    """

    try:
        case: PipelineCase = get_pipeline_case(name_or_id)
    except DatasetCatalogError as exc:
        raise GeometryEngineError(str(exc)) from exc

    if "geo" not in case.active_stages:
        raise GeometryEngineError(
            f"Pipeline case '{case.id}' does not activate the geo stage."
        )

    geo_engine = engine if engine is not None else GeometryEngine()
    geo_output = geo_engine.generate_from_spec(spec_as_dict(case.stages.geo))

    executed = ["geo"]
    math_output: Optional[MeshOutput] = None
    math_seconds: Optional[float] = None
    if "math" in case.active_stages:
        if case.stages.math is None:
            raise GeometryEngineError(
                f"Pipeline case '{case.id}' activates math without a stages.math payload."
            )
        params = dict(case.stages.math)
        if math_overrides:
            params.update(math_overrides)
        started = time.perf_counter()
        math_output = _mesh_from_geometry(
            geo_output,
            params,
            mesh_engine if mesh_engine is not None else MeshEngine(),
        )
        math_seconds = time.perf_counter() - started
        executed.append("math")

    pending = list(pending_pipeline_stages(case))
    notes: Dict[str, Any] = {
        "executed": executed,
        "deferred": pending,
    }
    if math_seconds is not None:
        notes["math_seconds"] = math_seconds
    if math_overrides:
        notes["math_overrides"] = sorted(math_overrides)
    if case.stages.math is not None and "math" not in executed:
        notes["math_spec_present"] = True
    if pending:
        notes["next_module"] = pending[0]

    return PipelineRunResult(
        case_id=case.id,
        title=case.title,
        description=case.description,
        active_stages=list(case.active_stages),
        pending_stages=pending,
        geo=geo_output,
        math=math_output,
        physics=None,
        aero=None,
        ai=None,
        notes=notes,
    )


def run_math_module_sample(
    name_or_id: str,
    *,
    engine: Optional[GeometryEngine] = None,
    mesh_engine: Optional[MeshEngine] = None,
    math_overrides: Optional[Dict[str, Any]] = None,
) -> MathSampleRun:
    """
    Run one Module 2 fixture.

    Guard samples set ``expect_error``. A matching exception is a successful
    run of that fixture; a different exception or a silent success is not.
    Because a guard only means anything at its stored parameters,
    ``math_overrides`` is rejected for those fixtures.
    """

    try:
        sample: MathModuleSample = get_math_module_sample(name_or_id)
    except DatasetCatalogError as exc:
        raise GeometryEngineError(str(exc)) from exc

    spec = sample.spec
    if math_overrides and spec.expect_error is not None:
        raise InvalidMeshConfigError(
            f"Guard sample '{sample.id}' expects {spec.expect_error} at its stored "
            "parameters and cannot be re-tuned."
        )
    geo_output: Optional[GeometryOutput] = None
    if spec.facets is not None:
        geo_output = _geometry_from_facets(np.asarray(spec.facets, dtype=np.float64))
    elif spec.geometry is not None:
        geo_engine = engine if engine is not None else GeometryEngine()
        geo_output = geo_engine.generate_from_spec(spec_as_dict(spec.geometry))

    started = time.perf_counter()
    mesh_output: Optional[MeshOutput] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    try:
        if geo_output is None:
            raise GeometryEngineError(
                f"Math sample '{sample.id}' has no geometry to discretize."
            )
        params = dict(spec.parameters)
        if math_overrides:
            params.update(math_overrides)
        mesh_output = _mesh_from_geometry(
            geo_output,
            params,
            mesh_engine if mesh_engine is not None else MeshEngine(),
        )
    except MeshEngineError as exc:
        error_type = type(exc).__name__
        error_message = str(exc)
        if spec.expect_error is None:
            raise
        if error_type != spec.expect_error:
            raise MeshEngineError(
                f"Sample '{sample.id}' expected {spec.expect_error}, "
                f"raised {error_type}: {exc}"
            ) from exc
    else:
        if spec.expect_error is not None:
            raise MeshEngineError(
                f"Sample '{sample.id}' expected {spec.expect_error} but succeeded."
            )

    return MathSampleRun(
        sample_id=sample.id,
        title=sample.title,
        description=sample.description,
        expect_error=spec.expect_error,
        geo=geo_output,
        math=mesh_output,
        error_type=error_type,
        error_message=error_message,
        seconds=time.perf_counter() - started,
    )
