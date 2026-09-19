"""
End-to-end pipeline sample runner.

Today only the geo stage is live. When math_module (and later engines)
are integrated, extend `run_pipeline_case` so the same case ids keep
flowing GeometryOutput → MeshOutput → PhysicsOutput → AeroOutput → AIOutput.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TypedDict

from geo_module.geo_skeleton import GeometryOutput
from geo_module.geometry_module import GeometryEngine, GeometryEngineError

from .dataset_catalog import (
    DatasetCatalogError,
    PipelineCase,
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
    math: None
    physics: None
    aero: None
    ai: None
    notes: Dict[str, Any]


def run_pipeline_case(
    name_or_id: str,
    *,
    engine: Optional[GeometryEngine] = None,
) -> PipelineRunResult:
    """
    Execute every *currently implemented* stage of a pipeline case.

    Downstream modules must append here; they must not fork a second
    catalog of user journeys.
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

    pending = list(pending_pipeline_stages(case))
    notes: Dict[str, Any] = {
        "executed": ["geo"],
        "deferred": pending,
    }
    if case.stages.math is not None:
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
        math=None,
        physics=None,
        aero=None,
        ai=None,
        notes=notes,
    )
