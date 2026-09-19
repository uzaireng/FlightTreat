"""
geo_inputs.py

Input handling, validation models, and built-in sample definitions
for the Geometry Engine.

This module deliberately contains no heavy geometry calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Tuple

from shared.dataset_catalog import (
    DatasetCatalogError,
    get_geo_module_sample,
    list_geo_module_samples,
    spec_as_dict,
)


GeometryMode = Literal["2D", "3D"]


class GeometryInputError(ValueError):
    """Base exception for invalid geometry input."""


@dataclass(frozen=True)
class NACAInput:
    """Validated NACA 4-digit input."""

    code: str
    chord: float = 1.0
    num_points: int = 100

    def __post_init__(self) -> None:
        code = self.code.strip()

        if len(code) != 4 or not code.isdigit():
            raise GeometryInputError(
                "NACA code must be exactly four digits, e.g. '0012' or '4412'."
            )

        object.__setattr__(self, "code", code)

        # NACA 4-digit constraints:
        # m = first digit / 100
        # p = second digit / 10
        # t = last two digits / 100
        m = int(code[0])
        p = int(code[1])
        t = int(code[2:])

        if m > 9 or p > 9 or t > 99:
            raise GeometryInputError("Invalid NACA 4-digit code.")

        if m > 0 and p == 0:
            raise GeometryInputError(
                "Cambered NACA 4-digit codes cannot have p = 0 (second digit)."
            )

        if self.chord <= 0.0:
            raise GeometryInputError("Chord length must be greater than zero.")

        if self.num_points < 20:
            raise GeometryInputError(
                "num_points must be at least 20 for a valid airfoil."
            )

        if self.num_points > 100_000:
            raise GeometryInputError(
                "num_points exceeds the maximum supported resolution of 100000."
            )


@dataclass(frozen=True)
class WingInput:
    """Validated parametric wing configuration."""

    chord: float = 1.0
    span: float = 5.0
    sweep_deg: float = 0.0
    dihedral_deg: float = 0.0
    num_sections: int = 10

    def __post_init__(self) -> None:
        if self.chord <= 0.0:
            raise GeometryInputError("Wing chord must be greater than zero.")

        if self.span <= 0.0:
            raise GeometryInputError("Wing span must be greater than zero.")

        if not -89.0 < self.sweep_deg < 89.0:
            raise GeometryInputError(
                "Sweep angle must be between -89 and +89 degrees."
            )

        if not -89.0 < self.dihedral_deg < 89.0:
            raise GeometryInputError(
                "Dihedral angle must be between -89 and +89 degrees."
            )

        if not 2 <= self.num_sections <= 10_000:
            raise GeometryInputError(
                "num_sections must be between 2 and 10000."
            )


def validate_file_input(
    file_content: bytes,
    file_type: str,
    *,
    max_size_mb: float = 50.0,
) -> Tuple[bytes, str]:
    """
    Validate uploaded file content before sending it to a parser.
    """

    if not isinstance(file_content, (bytes, bytearray)):
        raise GeometryInputError("Uploaded geometry must be provided as bytes.")

    if len(file_content) == 0:
        raise GeometryInputError("Uploaded geometry file is empty.")

    if len(file_content) > max_size_mb * 1024 * 1024:
        raise GeometryInputError(
            f"Geometry file exceeds the {max_size_mb:.0f} MB upload limit."
        )

    normalized_type = file_type.strip().lower().lstrip(".")

    supported = {"svg", "dxf", "stl", "obj"}

    if normalized_type not in supported:
        raise GeometryInputError(
            f"Unsupported geometry format '{file_type}'. "
            f"Supported formats: {', '.join(sorted(supported))}."
        )

    return bytes(file_content), normalized_type


def list_builtin_samples() -> Tuple[str, ...]:
    """Return geo module sample titles (datasets/modules/geo/catalog.json)."""

    try:
        return tuple(sample.title for sample in list_geo_module_samples())
    except DatasetCatalogError as exc:
        raise GeometryInputError(str(exc)) from exc


def get_builtin_sample(name: str) -> Dict[str, Any]:
    """Return a defensive copy of a geo module sample spec."""

    try:
        sample = get_geo_module_sample(name)
    except DatasetCatalogError as exc:
        raise GeometryInputError(str(exc)) from exc

    payload = spec_as_dict(sample.spec)
    payload["id"] = sample.id
    payload["title"] = sample.title
    payload["description"] = sample.description
    return payload