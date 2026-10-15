"""
Shared dataset catalog loader.

Two professional sample tracks live under /datasets:

1. Module samples  — isolated fixtures for one engine (unit / workbench).
2. Pipeline cases  — end-to-end journeys. Each new physics module must
   extend these cases so a real-user chain stays green.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PIPELINE_STAGE_ORDER: tuple[str, ...] = (
    "geo",
    "math",
    "physics",
    "aero",
    "ai",
)


class DatasetCatalogError(ValueError):
    """Invalid or missing dataset catalog / sample file."""


def project_root() -> Path:
    """MVP repository root (parent of src/)."""

    return Path(__file__).resolve().parents[2]


def datasets_root() -> Path:
    return project_root() / "datasets"


def resolve_dataset_file(relative_path: str) -> Path:
    """
    Resolve a path that must remain inside datasets/.
    """

    if not isinstance(relative_path, str) or not relative_path.strip():
        raise DatasetCatalogError("Dataset file path must be a non-empty string.")

    root = datasets_root().resolve()
    candidate = (root / relative_path.strip()).resolve()

    if candidate != root and root not in candidate.parents:
        raise DatasetCatalogError(
            "Sample files must live under the datasets/ directory."
        )

    if not candidate.is_file():
        raise DatasetCatalogError(f"Dataset file not found: {relative_path}")

    return candidate


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise DatasetCatalogError(f"Catalog not found: {path}")

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetCatalogError(f"Invalid JSON in {path}: {exc}") from exc


class GeometrySpec(BaseModel):
    """Parameters consumed by GeometryEngine.generate_from_spec()."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["naca", "flat_plate", "wing", "cylinder", "file"]
    mode: Optional[Literal["2D", "3D"]] = None
    code: Optional[str] = None
    chord: Optional[float] = None
    num_points: Optional[int] = None
    thickness: Optional[float] = None
    airfoil: Optional[str] = None
    span: Optional[float] = None
    sweep_deg: Optional[float] = None
    dihedral_deg: Optional[float] = None
    num_sections: Optional[int] = None
    airfoil_points: Optional[int] = None
    radius: Optional[float] = None
    length: Optional[float] = None
    segments: Optional[int] = None
    rings: Optional[int] = None
    path: Optional[str] = None
    file_type: Optional[str] = None
    shape_name: Optional[str] = None


class ModuleSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    spec: GeometrySpec


class ModuleCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    module: str
    kind: Literal["module_samples"]
    samples: List[ModuleSample]


class PipelineStages(BaseModel):
    """
    Per-module stage payloads.

    Only `geo` is required today. Later modules add their own models
    without renaming this container.
    """

    model_config = ConfigDict(extra="forbid")

    geo: GeometrySpec
    math: Optional[Dict[str, Any]] = None
    physics: Optional[Dict[str, Any]] = None
    aero: Optional[Dict[str, Any]] = None
    ai: Optional[Dict[str, Any]] = None


class PipelineCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    active_stages: List[Literal["geo", "math", "physics", "aero", "ai"]]
    stages: PipelineStages

    @field_validator("active_stages")
    @classmethod
    def _unique_known_order(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("Pipeline cases must declare at least one active stage.")
        if len(set(value)) != len(value):
            raise ValueError("active_stages must not contain duplicates.")
        order = {name: idx for idx, name in enumerate(PIPELINE_STAGE_ORDER)}
        if value != sorted(value, key=lambda name: order[name]):
            raise ValueError(
                "active_stages must follow geo → math → physics → aero → ai."
            )
        return value


class PipelineCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["pipeline_cases"]
    cases: List[PipelineCase]


@lru_cache(maxsize=8)
def load_geo_module_catalog() -> ModuleCatalog:
    payload = _read_json(datasets_root() / "modules" / "geo" / "catalog.json")
    catalog = ModuleCatalog.model_validate(payload)
    if catalog.module != "geo":
        raise DatasetCatalogError("geo catalog.module must be 'geo'.")
    ids = [sample.id for sample in catalog.samples]
    if len(ids) != len(set(ids)):
        raise DatasetCatalogError("geo module sample ids must be unique.")
    return catalog


@lru_cache(maxsize=8)
def load_pipeline_catalog() -> PipelineCatalog:
    payload = _read_json(datasets_root() / "pipeline" / "catalog.json")
    catalog = PipelineCatalog.model_validate(payload)
    ids = [case.id for case in catalog.cases]
    if len(ids) != len(set(ids)):
        raise DatasetCatalogError("pipeline case ids must be unique.")
    return catalog


def list_geo_module_samples() -> tuple[ModuleSample, ...]:
    return tuple(load_geo_module_catalog().samples)


def get_geo_module_sample(name_or_id: str) -> ModuleSample:
    if not isinstance(name_or_id, str) or not name_or_id.strip():
        raise DatasetCatalogError("Sample name must be a non-empty string.")

    key = name_or_id.strip()
    for sample in list_geo_module_samples():
        if sample.id == key or sample.title == key:
            return sample

    raise DatasetCatalogError(f"Unknown geo module sample '{key}'.")


def list_pipeline_cases() -> tuple[PipelineCase, ...]:
    return tuple(load_pipeline_catalog().cases)


def get_pipeline_case(name_or_id: str) -> PipelineCase:
    if not isinstance(name_or_id, str) or not name_or_id.strip():
        raise DatasetCatalogError("Pipeline case name must be a non-empty string.")

    key = name_or_id.strip()
    for case in list_pipeline_cases():
        if case.id == key or case.title == key:
            return case

    raise DatasetCatalogError(f"Unknown pipeline case '{key}'.")


def pending_pipeline_stages(case: PipelineCase) -> tuple[str, ...]:
    active = set(case.active_stages)
    return tuple(stage for stage in PIPELINE_STAGE_ORDER if stage not in active)


def spec_as_dict(spec: GeometrySpec) -> Dict[str, Any]:
    return spec.model_dump(exclude_none=True)


_MATH_SAMPLE_ERRORS: frozenset[str] = frozenset(
    {
        "InvalidMeshConfigError",
        "DegeneratePanelError",
        "MeshSingularityError",
        "InvertedNormalError",
        "OpenSurfaceError",
    }
)


class MathSampleSpec(BaseModel):
    """
    One Module 2 fixture.

    ``geometry`` is materialized by GeometryEngine. ``facets`` is a raw
    triangle list for traps the geometry engine refuses to emit.
    Exactly one of those sources is set. ``expect_error`` names the
    MeshEngine exception a guard sample must raise.
    """

    model_config = ConfigDict(extra="forbid")

    geometry: Optional[GeometrySpec] = None
    facets: Optional[List[List[List[float]]]] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    expect_error: Optional[str] = None

    @model_validator(mode="after")
    def _one_source(self) -> MathSampleSpec:
        if (self.geometry is None) == (self.facets is None):
            raise ValueError("Math sample spec needs exactly one of geometry or facets.")
        if self.expect_error is not None and self.expect_error not in _MATH_SAMPLE_ERRORS:
            raise ValueError(
                f"expect_error must be one of {sorted(_MATH_SAMPLE_ERRORS)}."
            )
        return self


class MathModuleSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    spec: MathSampleSpec


class MathModuleCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    module: Literal["math"]
    kind: Literal["module_samples"]
    samples: List[MathModuleSample]


@lru_cache(maxsize=8)
def load_math_module_catalog() -> MathModuleCatalog:
    payload = _read_json(datasets_root() / "modules" / "math" / "catalog.json")
    catalog = MathModuleCatalog.model_validate(payload)
    ids = [sample.id for sample in catalog.samples]
    if len(ids) != len(set(ids)):
        raise DatasetCatalogError("math module sample ids must be unique.")
    return catalog


def list_math_module_samples() -> tuple[MathModuleSample, ...]:
    return tuple(load_math_module_catalog().samples)


def get_math_module_sample(name_or_id: str) -> MathModuleSample:
    if not isinstance(name_or_id, str) or not name_or_id.strip():
        raise DatasetCatalogError("Sample name must be a non-empty string.")

    key = name_or_id.strip()
    for sample in list_math_module_samples():
        if sample.id == key or sample.title == key:
            return sample

    raise DatasetCatalogError(f"Unknown math module sample '{key}'.")


class PhysicsSampleSpec(BaseModel):
    """Solver-only fixture. It does not repeat a pipeline journey."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["bad_te_index", "singular_collocation", "thwaites_blasius", "naca_alpha0"]
    expect_error: Optional[str] = None


class PhysicsModuleSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    spec: PhysicsSampleSpec


class PhysicsModuleCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    module: Literal["physics"]
    kind: Literal["module_samples"]
    samples: List[PhysicsModuleSample]


@lru_cache(maxsize=8)
def load_physics_module_catalog() -> PhysicsModuleCatalog:
    payload = _read_json(datasets_root() / "modules" / "physics" / "catalog.json")
    catalog = PhysicsModuleCatalog.model_validate(payload)
    ids = [sample.id for sample in catalog.samples]
    if len(ids) != len(set(ids)):
        raise DatasetCatalogError("physics module sample ids must be unique.")
    return catalog


def list_physics_module_samples() -> tuple[PhysicsModuleSample, ...]:
    return tuple(load_physics_module_catalog().samples)


def get_physics_module_sample(name_or_id: str) -> PhysicsModuleSample:
    if not isinstance(name_or_id, str) or not name_or_id.strip():
        raise DatasetCatalogError("Sample name must be a non-empty string.")

    key = name_or_id.strip()
    for sample in list_physics_module_samples():
        if sample.id == key or sample.title == key:
            return sample

    raise DatasetCatalogError(f"Unknown physics module sample '{key}'.")
