"""
End-to-end pipeline sample runner.

Live stages: geo → math → physics → aero. AI stays pending.
`run_pipeline_case` flows
GeometryOutput → MeshOutput → PhysicsOutput → AeroOutput.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, TypedDict

import numpy as np

from aero_module.aero_engine import AeroEngine
from aero_module.aero_skeleton import AeroEngineError, AeroOutput
from geo_module.geo_skeleton import GeometryOutput, validate_geometry_output
from geo_module.geometry_module import GeometryEngine, GeometryEngineError
from math_module.math_skeleton import InvalidMeshConfigError, MeshEngineError, MeshOutput
from math_module.mesh_engine import MeshEngine
from physics_module.physics_engine import PhysicsEngine, split_flow_payload
from physics_module.physics_skeleton import PhysicsEngineError, PhysicsOutput

from .dataset_catalog import (
    DatasetCatalogError,
    MathModuleSample,
    PhysicsModuleSample,
    PipelineCase,
    get_math_module_sample,
    get_physics_module_sample,
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
    physics_engine: Optional[PhysicsEngine] = None,
    aero_engine: Optional[AeroEngine] = None,
    math_overrides: Optional[Dict[str, Any]] = None,
    physics_overrides: Optional[Dict[str, Any]] = None,
    aero_overrides: Optional[Dict[str, Any]] = None,
) -> PipelineRunResult:
    """
    Execute every *currently implemented* stage of a pipeline case.

    Downstream modules must append here; they must not fork a second
    catalog of user journeys. Geometry arrays are copied before meshing.
    The mesh is copied again before the panel solve.

    ``math_overrides`` layers user-tuned meshing parameters on top of the
    stored ``stages.math`` payload. ``physics_overrides`` does the same for
    ``stages.physics``, and ``aero_overrides`` for ``stages.aero``.
    The catalog file is never mutated.
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

    physics_output: Optional[PhysicsOutput] = None
    physics_seconds: Optional[float] = None
    if "physics" in case.active_stages:
        if math_output is None:
            raise GeometryEngineError(
                f"Pipeline case '{case.id}' activates physics without a mesh."
            )
        if case.stages.physics is None:
            raise GeometryEngineError(
                f"Pipeline case '{case.id}' activates physics without a stages.physics payload."
            )
        payload = dict(case.stages.physics)
        if physics_overrides:
            payload.update(physics_overrides)
        freestream, parameters = split_flow_payload(payload)
        started = time.perf_counter()
        physics_output = (
            physics_engine if physics_engine is not None else PhysicsEngine()
        ).solve_from_mesh(math_output, freestream, parameters)
        physics_seconds = time.perf_counter() - started
        executed.append("physics")

    aero_output: Optional[AeroOutput] = None
    aero_seconds: Optional[float] = None
    if "aero" in case.active_stages:
        if physics_output is None or math_output is None:
            raise GeometryEngineError(
                f"Pipeline case '{case.id}' activates aero without a solved flow."
            )
        payload = dict(case.stages.aero or {})
        if aero_overrides:
            payload.update(aero_overrides)
        started = time.perf_counter()
        try:
            aero_output = (
                aero_engine if aero_engine is not None else AeroEngine()
            ).analyze_from_solution(math_output, physics_output, payload)
        except AeroEngineError as exc:
            raise GeometryEngineError(str(exc)) from exc
        aero_seconds = time.perf_counter() - started
        executed.append("aero")

    pending = list(pending_pipeline_stages(case))
    notes: Dict[str, Any] = {
        "executed": executed,
        "deferred": pending,
    }
    if math_seconds is not None:
        notes["math_seconds"] = math_seconds
    if physics_seconds is not None:
        notes["physics_seconds"] = physics_seconds
    if math_overrides:
        notes["math_overrides"] = sorted(math_overrides)
    if physics_overrides:
        notes["physics_overrides"] = sorted(physics_overrides)
    if aero_seconds is not None:
        notes["aero_seconds"] = aero_seconds
    if aero_overrides:
        notes["aero_overrides"] = sorted(str(key) for key in aero_overrides)
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
        physics=physics_output,
        aero=aero_output,
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


class PhysicsSampleRun(TypedDict):
    sample_id: str
    title: str
    description: str
    expect_error: Optional[str]
    error_type: Optional[str]
    error_message: Optional[str]
    cl: Optional[float]
    theta_ratio: Optional[float]
    seconds: float


def _small_airfoil_panel():
    """A short NACA 0012 panel set for solver-only fixtures."""

    from math_module.mesh_engine import MeshEngine

    geometry = GeometryEngine().generate_from_spec(
        {"type": "naca", "mode": "2D", "code": "0012", "chord": 1.0, "num_points": 80}
    )
    mesh = MeshEngine().from_geometry(
        geometry,
        {"n_panels": 48, "n_x": 24, "n_y": 16, "clustering": "cosine"},
    )
    panel = mesh["panel_2d"]
    if panel is None:
        raise GeometryEngineError("NACA fixture did not produce panels.")
    return mesh, panel


def run_physics_module_sample(name_or_id: str) -> PhysicsSampleRun:
    """Run one solver-only physics fixture.

    Guard samples must raise the stored exception. Numeric samples return
    lift or a Blasius thickness ratio.
    """

    from physics_module.boundary_layer_1d import _thwaites_side
    from physics_module.solver_2d import solve_2d

    try:
        sample: PhysicsModuleSample = get_physics_module_sample(name_or_id)
    except DatasetCatalogError as exc:
        raise GeometryEngineError(str(exc)) from exc

    started = time.perf_counter()
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    cl: Optional[float] = None
    theta_ratio: Optional[float] = None
    kind = sample.spec.kind
    try:
        if kind == "bad_te_index":
            _mesh, panel = _small_airfoil_panel()
            broken = dict(panel)
            broken["te_index"] = -3
            solve_2d(broken, 0.0, None, "vortex", True)
        elif kind == "singular_collocation":
            _mesh, panel = _small_airfoil_panel()
            broken = dict(panel)
            collocation = np.array(panel["collocation"], dtype=np.float64, copy=True)
            broken["collocation"] = np.repeat(collocation[:1], collocation.shape[0], axis=0)
            solve_2d(broken, 0.0, None, "vortex", True)
        elif kind == "thwaites_blasius":
            arc = np.linspace(0.0, 1.0, 81)
            speed = np.ones(arc.shape, dtype=np.float64)
            nu = 1.0e-5
            theta, _shape, _cf, _lam, _delta = _thwaites_side(arc, speed, nu)
            target = np.sqrt(0.45 * nu * arc[1:] / speed[1:])
            theta_ratio = float(np.max(np.abs(theta[1:] - target) / target))
        elif kind == "naca_alpha0":
            mesh, _panel = _small_airfoil_panel()
            solved = PhysicsEngine().solve_from_mesh(
                mesh,
                {"alpha_deg": 0.0, "reynolds": 5.0e5, "mach": 0.0, "v_inf": 1.0, "rho": 1.225},
                {"regime": "inviscid"},
            )
            cl = float(solved["loads"]["cl"])
        else:
            raise GeometryEngineError(f"Unknown physics sample kind {kind!r}.")
    except PhysicsEngineError as exc:
        error_type = type(exc).__name__
        error_message = str(exc)
        if sample.spec.expect_error is None:
            raise
        if error_type != sample.spec.expect_error:
            raise PhysicsEngineError(
                f"Sample '{sample.id}' expected {sample.spec.expect_error}, "
                f"raised {error_type}: {exc}"
            ) from exc
    else:
        if sample.spec.expect_error is not None:
            raise PhysicsEngineError(
                f"Sample '{sample.id}' expected {sample.spec.expect_error} but succeeded."
            )

    return PhysicsSampleRun(
        sample_id=sample.id,
        title=sample.title,
        description=sample.description,
        expect_error=sample.spec.expect_error,
        error_type=error_type,
        error_message=error_message,
        cl=cl,
        theta_ratio=theta_ratio,
        seconds=time.perf_counter() - started,
    )
