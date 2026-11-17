"""
physics_skeleton.py

Contract for Module 3 (Physics & Numerical Solver).

Solver architecture (locked):
    Advanced Viscous-Inviscid Coupled Panel Method
    3D surface panels + 2D vortex panels + 1D integral boundary layer.

This file holds types, validators, and exceptions only. Influence kernels
live in the solver modules. Module 3 consumes MeshOutput as-is (copy arrays;
do not remesh).
"""

from __future__ import annotations

from typing import Literal, Optional, Protocol, TypedDict

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

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
    fail_on_separation: bool
    prandtl_glauert: bool
    wake_length_chords: float
    dense_panel_limit: int


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
    The wake arrays are the merged shear layer downstream of the trailing
    edge. ``cd_momentum`` is the Squire–Young value at the last wake
    station. It is not the published drag coefficient.
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
    delta_upper: np.ndarray
    delta_lower: np.ndarray
    separation_s_upper: float
    separation_s_lower: float
    separated_upper: np.ndarray
    separated_lower: np.ndarray
    stagnation_xy: np.ndarray
    wake_s: np.ndarray
    wake_theta: np.ndarray
    wake_delta_star: np.ndarray
    wake_shape_h: np.ndarray
    wake_ue: np.ndarray
    cd_momentum: float


class SurfaceLoads(TypedDict):
    """Integrated pressure and skin-friction loads at unit freestream speed."""

    gamma_total: float
    cl: float
    cd_pressure: float
    cd_friction: float
    cd: float
    cm_quarter_chord: float
    cm_centroid: float
    moment_origin_qc: np.ndarray
    moment_origin_centroid: np.ndarray
    s_ref: float
    compressibility_applied: bool


class FlowField2D(TypedDict):
    """Cartesian farfield sample. Solid cells are NaN. ``w`` is identically zero."""

    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    speed: np.ndarray
    cp: np.ndarray
    phi: np.ndarray
    psi: np.ndarray


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
    loads: SurfaceLoads
    field_2d: Optional[FlowField2D]
    surface_velocity: Optional[np.ndarray]
    warnings: list[str]


class PhysicsEngineProtocol(Protocol):
    VERSION: str

    def solve(self, physics_input: PhysicsInput) -> PhysicsOutput: ...

    def solve_from_mesh(
        self,
        mesh: MeshOutput,
        freestream: FreestreamConditions,
        parameters: Optional[PhysicsParameters] = None,
    ) -> PhysicsOutput: ...


def _f64(name: str, value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float64:
        raise ValueError(f"{name} must be float64, got {array.dtype}.")
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
    return array


def _bool_vec(name: str, value: np.ndarray, size: int) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.bool_:
        raise ValueError(f"{name} must be bool, got {array.dtype}.")
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},).")
    return array


class FreestreamModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alpha_deg: float
    reynolds: float
    mach: float
    v_inf: float
    rho: float

    @model_validator(mode="after")
    def _finite_ranges(self) -> FreestreamModel:
        if not np.isfinite(self.alpha_deg) or not -90.0 <= self.alpha_deg <= 90.0:
            raise ValueError("alpha_deg must be finite and inside [-90, 90].")
        if not np.isfinite(self.mach) or self.mach < 0.0:
            raise ValueError("mach must be finite and >= 0.")
        if self.mach >= 1.0:
            raise ValueError("mach >= 1 is outside the panel solver.")
        if not np.isfinite(self.v_inf) or self.v_inf <= 0.0:
            raise ValueError("v_inf must be finite and > 0.")
        if not np.isfinite(self.rho) or self.rho <= 0.0:
            raise ValueError("rho must be finite and > 0.")
        return self


class PhysicsOutputModel(BaseModel):
    """Runtime shape and dtype checks. No influence formulas."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    mode: Literal["2D", "3D"]
    regime: FlowRegime
    freestream: FreestreamConditions
    inviscid: InviscidSolution
    ibl: Optional[IBLSolution] = None
    vic_iterations: int
    residual: float
    converged: bool
    loads: SurfaceLoads
    field_2d: Optional[FlowField2D] = None
    surface_velocity: Optional[np.ndarray] = None
    warnings: list[str]

    @model_validator(mode="after")
    def _shapes(self) -> PhysicsOutputModel:
        FreestreamModel.model_validate(dict(self.freestream))
        if self.vic_iterations < 1:
            raise ValueError("vic_iterations must be >= 1.")
        if not np.isfinite(self.residual) or self.residual < 0.0:
            raise ValueError("residual must be finite and >= 0.")
        inviscid = self.inviscid
        cp = np.asarray(inviscid["cp_inviscid"])
        vt = np.asarray(inviscid["v_tangential"])
        count = int(cp.shape[0])
        _f64("cp_inviscid", cp, (count,))
        if self.mode == "2D":
            _f64("v_tangential", vt, (count, 2))
            if inviscid["mu"] is not None:
                raise ValueError("2D inviscid solution must set mu to None.")
            gamma = inviscid["gamma"]
            if gamma is None:
                raise ValueError("2D inviscid solution requires gamma.")
            _f64("gamma", gamma, (count,))
            sigma = inviscid["sigma"]
            if sigma is not None:
                _f64("sigma", sigma, (count,))
            if self.surface_velocity is not None:
                raise ValueError("2D physics output must set surface_velocity to None.")
            if self.field_2d is None:
                raise ValueError("2D physics output requires field_2d.")
            _check_field(self.field_2d)
        else:
            _f64("v_tangential", vt, (count, 3))
            if inviscid["gamma"] is not None:
                raise ValueError("3D inviscid solution must set gamma to None.")
            if inviscid["mu"] is None or inviscid["sigma"] is None:
                raise ValueError("3D inviscid solution requires mu and sigma.")
            _f64("mu", inviscid["mu"], (count,))
            _f64("sigma", inviscid["sigma"], (count,))
            if self.field_2d is not None:
                raise ValueError("3D physics output must set field_2d to None.")
            if self.surface_velocity is None:
                raise ValueError("3D physics output requires surface_velocity.")
            _f64("surface_velocity", self.surface_velocity, (count, 3))
        if self.ibl is not None:
            _check_ibl(self.ibl)
        _check_loads(self.loads, self.mode)
        if self.regime == "viscous_inviscid" and not np.isfinite(self.freestream["reynolds"]):
            raise ValueError("viscous_inviscid requires a finite Reynolds number.")
        if self.regime == "viscous_inviscid" and self.freestream["reynolds"] <= 0.0:
            raise ValueError("viscous_inviscid requires Reynolds number > 0.")
        return self

    def to_typed_dict(self) -> PhysicsOutput:
        return PhysicsOutput(
            mode=self.mode,
            regime=self.regime,
            freestream=dict(self.freestream),  # type: ignore[arg-type]
            inviscid=self.inviscid,
            ibl=self.ibl,
            vic_iterations=int(self.vic_iterations),
            residual=float(self.residual),
            converged=bool(self.converged),
            loads=self.loads,
            field_2d=self.field_2d,
            surface_velocity=self.surface_velocity,
            warnings=list(self.warnings),
        )


def _check_field(field: FlowField2D) -> None:
    u = np.asarray(field["u"])
    shape = u.shape
    if u.ndim != 2:
        raise ValueError("field_2d.u must be 2-dimensional.")
    for name in ("u", "v", "w", "speed", "cp", "phi", "psi"):
        _f64(f"field_2d.{name}", field[name], shape)  # type: ignore[literal-required]
    if not np.all(np.asarray(field["w"])[~np.isnan(np.asarray(field["w"]))] == 0.0):
        raise ValueError("field_2d.w must be zero wherever it is finite.")


def _check_ibl(ibl: IBLSolution) -> None:
    n_upper = int(np.asarray(ibl["theta_upper"]).shape[0])
    n_lower = int(np.asarray(ibl["theta_lower"]).shape[0])
    for name in (
        "theta_upper",
        "delta_star_upper",
        "shape_h_upper",
        "cf_upper",
        "ue_upper",
        "delta_upper",
    ):
        _f64(name, ibl[name], (n_upper,))  # type: ignore[literal-required]
    for name in (
        "theta_lower",
        "delta_star_lower",
        "shape_h_lower",
        "cf_lower",
        "ue_lower",
        "delta_lower",
    ):
        _f64(name, ibl[name], (n_lower,))  # type: ignore[literal-required]
    _bool_vec("separated_upper", ibl["separated_upper"], n_upper)
    _bool_vec("separated_lower", ibl["separated_lower"], n_lower)
    _f64("stagnation_xy", ibl["stagnation_xy"], (2,))
    for side in ("upper", "lower"):
        theta = np.asarray(ibl[f"theta_{side}"])  # type: ignore[literal-required]
        shape = np.asarray(ibl[f"shape_h_{side}"])  # type: ignore[literal-required]
        delta_star = np.asarray(ibl[f"delta_star_{side}"])  # type: ignore[literal-required]
        predicted = shape * theta
        scale = np.maximum(np.abs(predicted), 1e-12)
        if np.any(np.abs(delta_star - predicted) > 1e-8 * scale):
            raise ValueError(f"delta_star_{side} must equal shape_h * theta.")
    n_wake = int(np.asarray(ibl["wake_theta"]).shape[0])
    for name in ("wake_s", "wake_theta", "wake_delta_star", "wake_shape_h", "wake_ue"):
        _f64(name, ibl[name], (n_wake,))  # type: ignore[literal-required]
    cd_momentum = float(ibl["cd_momentum"])
    if not np.isnan(cd_momentum) and not np.isfinite(cd_momentum):
        raise ValueError("cd_momentum must be finite or NaN.")
    if n_wake > 0:
        wake_theta = np.asarray(ibl["wake_theta"])
        wake_shape = np.asarray(ibl["wake_shape_h"])
        wake_delta = np.asarray(ibl["wake_delta_star"])
        predicted = wake_shape * wake_theta
        scale = np.maximum(np.abs(predicted), 1e-12)
        if np.any(np.abs(wake_delta - predicted) > 1e-8 * scale):
            raise ValueError("wake_delta_star must equal wake_shape_h * wake_theta.")


def _check_loads(loads: SurfaceLoads, mode: Literal["2D", "3D"]) -> None:
    origin = np.asarray(loads["moment_origin_qc"])
    centroid = np.asarray(loads["moment_origin_centroid"])
    width = 2 if mode == "2D" else 3
    _f64("moment_origin_qc", origin, (width,))
    _f64("moment_origin_centroid", centroid, (width,))
    if not np.isfinite(loads["s_ref"]) or loads["s_ref"] <= 0.0:
        raise ValueError("s_ref must be finite and > 0.")


def validate_physics_output(payload: PhysicsOutput) -> PhysicsOutput:
    """Validate shapes and dtypes and return the same contract object."""

    return PhysicsOutputModel.model_validate(payload).to_typed_dict()
