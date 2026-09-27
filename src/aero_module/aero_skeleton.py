"""
aero_skeleton.py

DRAFT incoming contract for Module 4 (Aerodynamics & Analytics).

Consumes PhysicsOutput (+ MeshOutput for moment reference / areas).
Produces force/moment coefficients at wind-tunnel-comparable accuracy
targets (85–95%+ vs experiment for attached 2D/3D cruise cases).

No implementation in this file.
"""

from __future__ import annotations

from typing import Literal, Optional, Protocol, TypedDict

import numpy as np

from math_module.math_skeleton import MeshOutput
from physics_module.physics_skeleton import PhysicsOutput


class AeroEngineError(Exception):
    """Base exception for Module 4."""


class InvalidReferenceQuantityError(AeroEngineError, ValueError):
    """Zero chord, area, or span used for coefficient reduction."""


class AeroParameters(TypedDict, total=False):
    s_ref: Optional[float]
    c_ref: Optional[float]
    b_ref: Optional[float]
    moment_origin: tuple[float, float, float]


class AeroInput(TypedDict):
    mesh: MeshOutput
    physics: PhysicsOutput
    parameters: AeroParameters


class ForceCoefficients(TypedDict):
    """Body/wind axes non-dimensional forces. Total CD = pressure + friction."""

    cl: float
    cd: float
    cd_pressure: float
    cd_friction: float
    cm: float
    cl_inviscid: float
    cd_inviscid: float


class AeroOutput(TypedDict):
    mode: Literal["2D", "3D"]
    coefficients: ForceCoefficients
    cp_surface: np.ndarray
    x_over_c: Optional[np.ndarray]
    aerodynamic_center_x: float
    l_over_d: float
    notes: dict[str, str]


class AeroEngineProtocol(Protocol):
    VERSION: str

    def analyze(self, aero_input: AeroInput) -> AeroOutput: ...
