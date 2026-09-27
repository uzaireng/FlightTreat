"""
physics_skeleton.py

DRAFT incoming contract for Module 3 (Physics & Numerical Solver).

Solver architecture (locked):
    Advanced Viscous-Inviscid Coupled Panel Method
    3D surface panels + 2D vortex panels + 1D integral boundary layer.

This file is a specification stub. Implementation arrives later from ChatGPT.
Module 3 MUST consume MeshOutput as-is (copy arrays; do not remesh).
"""

from __future__ import annotations

from typing import Literal, Optional, Protocol, TypedDict

import numpy as np

from math_module.math_skeleton import MeshOutput


class PhysicsEngineError(Exception):
    """Base exception for Module 3."""


class SingularInfluenceMatrixError(PhysicsEngineError, ValueError):
    """Influence matrix A is singular or ill-conditioned beyond recovery."""


class BoundaryLayerSeparationError(PhysicsEngineError, ValueError):
    """IBL marching failed (separation without prescribed transition/wake model)."""


class InvalidFlowConditionError(PhysicsEngineError, ValueError):
    """Illegal α, Re, Mach, or reference quantities."""


FlowRegime = Literal["inviscid", "viscous_inviscid"]
TwoDPanelModel = Literal["vortex", "source_vortex"]
ThreeDPanelModel = Literal["constant_source_doublet", "constant_doublet"]
IBLClosure = Literal["thwaites_head", "drela_two_equation"]


class FreestreamConditions(TypedDict):
    """
    Reference state. Velocities non-dimensionalized by V_inf unless noted.

    alpha_deg: geometric angle of attack
    reynolds:  chord Reynolds number (IBL). Ignored if regime == inviscid.
    mach:      reserved; incompressible core assumes mach ≈ 0
    v_inf:     dimensional speed [m/s] if forces later need q_inf
    rho:       density [kg/m^3]
    """

    alpha_deg: float
    reynolds: float
    mach: float
    v_inf: float
    rho: float


class PhysicsParameters(TypedDict, total=False):
    regime: FlowRegime
    panel_model_2d: TwoDPanelModel
    panel_model_3d: ThreeDPanelModel
    ibl_closure: IBLClosure
    kutta: bool
    max_vic_iterations: int
    vic_tolerance: float
    viscous_coupling: Literal["displacement", "transpiration"]


class PhysicsInput(TypedDict):
    """Module 3 input: Module 2 mesh + flow + solver knobs."""

    mesh: MeshOutput
    freestream: FreestreamConditions
    parameters: PhysicsParameters


class InviscidSolution(TypedDict):
    """
    Panel strengths and surface inviscid kinematics.

    2D:
        gamma           (P,)     vortex strength / panel
        mu              None
        sigma           optional source (P,) if source-vortex
    3D:
        mu              (F,)     doublet (or equivalent)
        sigma           (F,)     source
        gamma           None

    cp_inviscid         (P,) or (F,)
    v_tangential        (P, 2) or (F, 3)
    collocation_cp      same length as strengths
    """

    gamma: Optional[np.ndarray]
    mu: Optional[np.ndarray]
    sigma: Optional[np.ndarray]
    cp_inviscid: np.ndarray
    v_tangential: np.ndarray
    influence_condition_number: float


class IBLSolution(TypedDict):
    """
    1D integral boundary layer on Module 2 stations.

    theta, delta_star, H, cf, ue along upper/lower.
    transition_s: arc length of transition (nan if fully laminar/turbulent).
    """

    theta_upper: np.ndarray
    theta_lower: np.ndarray
    delta_star_upper: np.ndarray
    delta_star_lower: np.ndarray
    shape_h_upper: np.ndarray
    shape_h_lower: np.ndarray
    cf_upper: np.ndarray
    cf_lower: np.ndarray
    ue_upper: np.ndarray
    ue_lower: np.ndarray
    transition_s_upper: float
    transition_s_lower: float
    separated: bool


class PhysicsOutput(TypedDict):
    """Primary Module 3 output consumed by aero_module."""

    mode: Literal["2D", "3D"]
    regime: FlowRegime
    freestream: FreestreamConditions
    inviscid: InviscidSolution
    ibl: Optional[IBLSolution]
    vic_iterations: int
    residual: float
    converged: bool


class PhysicsEngineProtocol(Protocol):
    VERSION: str

    def solve(self, physics_input: PhysicsInput) -> PhysicsOutput: ...

    def solve_from_mesh(
        self,
        mesh: MeshOutput,
        freestream: FreestreamConditions,
        parameters: Optional[PhysicsParameters] = None,
    ) -> PhysicsOutput: ...
