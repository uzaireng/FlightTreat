"""
aero_skeleton.py

Contract for Module 4 (Aerodynamics & Analytics).

Module 4 does not assemble an influence matrix, march a boundary layer,
remesh, or run a second panel solve. Module 3 already integrated pressure
and skin friction into ``PhysicsOutput.loads`` and wrote surface ``Cp``.
This module reduces that payload into the numbers a user reports:

    coefficients on a chosen reference area and chord
    lift, drag, and moment in newtons
    L/D
    a Cp distribution suitable for a plot
    a polar, and an aerodynamic center, when several angles are supplied

Calculations (implementation must follow these; this file does not run them)
-----------------------------------------------------------------------
Dynamic pressure, from the freestream Module 3 already validated::

    q = 0.5 * rho * v_inf**2

Reference selection. Omitted parameters keep the physics reference.
``S_phys = physics.loads.s_ref``. In 2D that value is the chord, and
``c_phys`` is ``mesh.panel_2d.chord`` (the same length). In 3D ``c_phys``
is not stored on ``PhysicsOutput``; until the caller passes ``c_ref``,
the chord used for moments is the mid-span x-extent of the surface
centroids (the same root-chord estimate Module 3 uses) and the output
must say so in ``warnings``.

    S = parameters.s_ref or S_phys
    c = parameters.c_ref or c_phys
    b = parameters.b_ref or (3D centroid y-extent; omitted in 2D)

In 2D, ``s_ref`` and ``c_ref`` are the same chord. Setting both to
different positive numbers is an error. A 2D coefficient is per unit span.

Rescale. Module 3 non-dimensionalized by its own area and chord. A new
reference multiplies the finished coefficients. It does not re-integrate
``Cp``::

    k_S = S_phys / S
    CL = cl_physics * k_S
    CD_p = cd_pressure_physics * k_S
    CD_f = cd_friction_physics * k_S          # stays NaN when physics has no IBL
    CD = CD_p + CD_f                          # CD_p alone when CD_f is NaN

    2D:  CM_c/4 = cm_quarter_chord_physics * (c_phys / c)**2
    3D:  CM_c/4 = cm_quarter_chord_physics * (S_phys * c_phys) / (S * c)

Moment origin. The default origin is the quarter-chord point already on
``physics.loads.moment_origin_qc``, and ``cm`` equals ``CM_c/4``.
A caller-supplied ``moment_origin`` transfers that moment. It does not
touch the panels. With forces rebuilt from the wind axes
(``lift_hat = (-sin α, cos α)`` in 2D, ``(-sin α, 0, cos α)`` in 3D,
``drag_hat = (cos α, sin α)`` or ``(cos α, 0, sin α)``)::

    C_force = CL * lift_hat + CD * drag_hat
    2D: CM = CM_c/4 + (Δx / c) * C_force_y - (Δy / c) * C_force_x
    3D: CM = CM_c/4 + (Δx / c) * C_force_z - (Δz / c) * C_force_x

``Δ`` is the quarter-chord origin minus the requested origin.
The aerodynamic center always uses ``CM_c/4``, never the shifted moment.

Drag split.

    2D: CD_i = 0. A section coefficient has no induced drag.
        CD_p is pressure drag of the section.
    3D, induced_drag_model "none" (default):
        CD_i is NaN. Integrated CD_p already contains induced drag
        mixed with the rest of the pressure drag. Module 4 must not
        invent a split.
    3D, induced_drag_model "lifting_line":
        AR = b**2 / S
        CD_i = CL**2 / (π * AR * e)
        e = oswald_efficiency, default 0.85, required in (0, 1].
        This estimate is reported beside CD. It is not subtracted
        from CD and it is not a Trefftz-plane result. If b is missing,
        CD_i stays NaN and a warning is attached.

Lift-to-drag::

    L/D = CL / CD    when CD is finite and |CD| >= 1e-6, else NaN

Dimensional loads use the rescaled coefficients::

    2D, per unit span:  L = CL * q * c,  D = CD * q * c,  M = CM * q * c**2
    3D:                 L = CL * q * S,  D = CD * q * S,  M = CM * q * S * c

Inviscid pair. ``cl_inviscid`` and ``cd_inviscid`` equal ``CL`` and ``CD``
when ``physics.regime`` is ``inviscid``. A viscous solve does not store a
second inviscid field, so both are NaN. Module 4 must not re-solve the
inviscid problem to fill them.

Surface pressure is a copy of ``physics.inviscid.cp_inviscid``.
``x_over_c`` is a copy of the 2D panel stations. 3D leaves it as None.
Neither array is written back into the mesh or the physics payload.

Polar. ``analyze`` is one angle. ``reduce_polar`` stacks angles that
Module 3 has already solved on the same mesh. It sorts by ``alpha_deg``
and, when at least two samples have distinct ``CL``, fits

    CL = CL_0 + cl_α * α
    CM_c/4 = CM_0 + cm_α * α

with α in radians, then::

    x_ac / c = 0.25 - (cm_α / cl_α)     # from the leading edge

``analyze`` sets ``aerodynamic_center_x`` to NaN. A flat lift curve
(|cl_α| < 1e-3) also yields NaN. ``(L/D)_max`` is the best sample in
the stack, not an analytic optimum. ``cl_max`` is the largest sampled
lift coefficient, not a stall model.

Stall. ``stall_model`` accepts only ``"none"``. No ``C_L,max`` drop and
no drag spike is applied. Separation warnings already on the physics
payload are copied through. A future model is a new literal and a new
formula; it must not be implied by this contract.

Body axes, from the rescaled wind-axis coefficients. α is the geometric
angle already solved, in radians::

    CN = CL cos α + CD sin α
    CA = CD cos α - CL sin α

Center of pressure, from the quarter-chord moment, as a fraction of
chord from the leading edge. One angle is enough. It is not the
aerodynamic center::

    x_cp / c = 0.25 - CM_c/4 / CL     when |CL| >= 1e-4, else NaN

Pressure markers are a pass over the copied ``Cp``. ``cp_min`` and
``cp_max`` exist in both modes. ``cp_min_x_over_c`` uses the 2D station
at the minimum. 3D has no single chord axis, so that station is NaN.

Viscous card. Module 4 does not march the boundary layer. In 2D it
copies transition and separation stations from ``physics.ibl``. In 3D
Module 3 does not march a boundary layer, so ``available`` is false and
the stations are NaN.

Spanwise lift, 3D only. Panels already carry ``Cp``, area, and a
centroid. Bin the centroids along y. In a bin of width Δy and local
chord c(y) (x-extent of those centroids)::

    ℓ'(y) / q = sum(-Cp * n · lift_hat * area)
    cl(y) = (ℓ'(y) / q) / (c(y) * Δy)

2D sets ``spanwise`` to None. This is not a new lattice and it does not
move the geometry.

Polar extras, same samples as the aerodynamic center, still with no
extra solve.

    α_L0 = -CL_0 / cl_α                         # radians, then stored in degrees
    CM_0 is the quarter-chord moment intercept of the same fit
    CD = CD_0 + K CL²                           # least squares on the stack
    e_fit = 1 / (π AR K)                        # 3D only; NaN in 2D

Ideal slope, reported beside the fitted slope so a bad solve is visible.
It does not replace ``cl_alpha``::

    2D: cl_α,ideal = 2π
    3D: cl_α,ideal = 2π AR / (AR + 2)           # unswept lifting line
    ratio = cl_α / cl_α,ideal

Static margin needs a center of gravity and an aerodynamic center::

    static_margin = x_ac/c - cg_x_over_c

``cg_x_over_c`` is optional. A single angle has no ``x_ac``, so its
static margin is NaN. Omit the center of gravity and the polar margin
is NaN too.

Not this module. Geometry import, the linear solver, standard-atmosphere
formulas, a second panel method, a chart library, and a PDF writer stay
outside the reduction. Total drag remains pressure plus friction.
Induced drag is never substituted for that total.

No implementation in this file.
"""

from __future__ import annotations

from typing import Literal, Optional, Protocol, TypedDict

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from math_module.math_skeleton import MeshOutput
from physics_module.physics_skeleton import FlowRegime, PhysicsOutput


class AeroEngineError(Exception):
    """Base exception for Module 4."""


class InvalidReferenceQuantityError(AeroEngineError, ValueError):
    """Zero chord, area, or span used for coefficient reduction."""


class InvalidAeroConfigError(AeroEngineError, ValueError):
    """A reference, drag, or stall switch this contract does not accept."""


class InconsistentAeroInputError(AeroEngineError, ValueError):
    """Mesh and physics disagree, or a polar mixes operating points."""


InducedDragModel = Literal["none", "lifting_line"]
StallModel = Literal["none"]


class AeroParameters(TypedDict, total=False):
    """
    Knobs for one reduction. Every key is optional.

    Defaults when the key is omitted
    ------------------------------
    s_ref, c_ref, b_ref: physics / mesh reference described above
    moment_origin: quarter-chord origin from ``physics.loads``
    induced_drag_model: "none"
    oswald_efficiency: 0.85, used only by "lifting_line"
    stall_model: "none"
    cg_x_over_c: omitted, so static margin is NaN
    """

    s_ref: Optional[float]
    c_ref: Optional[float]
    b_ref: Optional[float]
    moment_origin: tuple[float, float, float]
    induced_drag_model: InducedDragModel
    oswald_efficiency: float
    stall_model: StallModel
    cg_x_over_c: Optional[float]


class AeroInput(TypedDict):
    """One solved angle. ``parameters`` may be an empty dict."""

    mesh: MeshOutput
    physics: PhysicsOutput
    parameters: AeroParameters


class ReferenceState(TypedDict):
    """Quantities actually used to reduce this operating point."""

    s_ref: float
    c_ref: float
    b_ref: float
    aspect_ratio: float
    q_inf: float
    rho: float
    v_inf: float
    moment_origin: np.ndarray
    moment_about: Literal["quarter_chord", "user"]
    reference_note: str


class ForceCoefficients(TypedDict):
    """
    Wind-axis coefficients after reference rescaling.

    ``cd = cd_pressure + cd_friction`` when friction is finite.
    ``cd_induced`` is 0 in 2D, NaN when the 3D split is off, and the
    lifting-line estimate when that model is requested.
    ``cm`` is about ``reference.moment_origin``.
    ``cm_quarter_chord`` stays about the quarter chord and feeds the
    aerodynamic-center slope.
    ``cl_inviscid`` and ``cd_inviscid`` are NaN on a viscous solve.
    ``cn`` and ``ca`` are the chord-normal and chord-axial pair.
    ``center_of_pressure_x`` is ``x_cp / c`` from the leading edge.
    """

    cl: float
    cd: float
    cd_pressure: float
    cd_friction: float
    cd_induced: float
    cm: float
    cm_quarter_chord: float
    cl_inviscid: float
    cd_inviscid: float
    cn: float
    ca: float
    center_of_pressure_x: float


class PressureMarkers(TypedDict):
    """Extrema of the copied surface pressure. No new flow solution."""

    cp_min: float
    cp_max: float
    cp_min_x_over_c: float


class ViscousSummary(TypedDict):
    """
    Boundary-layer stations copied from Module 3.

    ``available`` is true only when ``physics.ibl`` is present, which is
    the 2D viscous regime. Stations are arc length from stagnation.
    """

    available: bool
    separated: bool
    transition_s_upper: float
    transition_s_lower: float
    separation_s_upper: float
    separation_s_lower: float


class SpanwiseLoad(TypedDict):
    """Section lift along a 3D span. Equal-length arrays. Absent in 2D."""

    y: np.ndarray
    cl: np.ndarray
    chord: np.ndarray


class DimensionalLoads(TypedDict):
    """
    Newtons and newton-metres at the freestream dynamic pressure.

    2D values are per unit span. Any non-finite coefficient produces a
    NaN force of that component rather than a substitute.
    """

    lift: float
    drag: float
    pitching_moment: float


class AeroOutput(TypedDict):
    """Product card for one angle of attack."""

    mode: Literal["2D", "3D"]
    regime: FlowRegime
    alpha_deg: float
    reynolds: float
    mach: float
    coefficients: ForceCoefficients
    dimensional: DimensionalLoads
    reference: ReferenceState
    cp_surface: np.ndarray
    x_over_c: Optional[np.ndarray]
    pressure: PressureMarkers
    viscous: ViscousSummary
    spanwise: Optional[SpanwiseLoad]
    aerodynamic_center_x: float
    static_margin: float
    l_over_d: float
    notes: dict[str, str]
    warnings: list[str]


class AeroPolarInput(TypedDict):
    """
    Several Module 3 solutions of one mesh at different angles.

    The mesh is not re-solved here. Every sample must share ``mode`` and
    ``shape_name`` with ``mesh``.
    """

    mesh: MeshOutput
    samples: list[PhysicsOutput]
    parameters: AeroParameters


class AeroPolarOutput(TypedDict):
    """
    Curves and the two slopes a single angle cannot provide.

    Arrays are sorted by increasing ``alpha_deg`` and share that length.
    ``aerodynamic_center_x`` is ``x_ac / c`` measured from the leading edge.
    ``cl_alpha_ideal`` is the thin-airfoil or unswept lifting-line slope.
    ``cd0`` and ``drag_polar_k`` fit ``CD = CD_0 + K CL²``.
    ``oswald_from_polar`` is ``1 / (π AR K)`` in 3D and NaN in 2D.
    ``cl_max`` is the largest sample, not a ``C_L,max`` model.
    """

    mode: Literal["2D", "3D"]
    alpha_deg: np.ndarray
    cl: np.ndarray
    cd: np.ndarray
    cm: np.ndarray
    l_over_d: np.ndarray
    cl_alpha: float
    cl_alpha_ideal: float
    cl_alpha_ratio: float
    cm_alpha: float
    cm0: float
    alpha_zero_lift_deg: float
    cd0: float
    drag_polar_k: float
    oswald_from_polar: float
    aerodynamic_center_x: float
    static_margin: float
    cl_max: float
    alpha_cl_max_deg: float
    l_over_d_max: float
    alpha_l_over_d_max_deg: float
    samples: list[AeroOutput]


class AeroEngineProtocol(Protocol):
    VERSION: str

    def analyze(self, aero_input: AeroInput) -> AeroOutput: ...

    def reduce_polar(self, polar_input: AeroPolarInput) -> AeroPolarOutput: ...


class AeroParametersModel(BaseModel):
    """Reject reference values and model names this contract cannot reduce."""

    model_config = ConfigDict(extra="forbid")

    s_ref: Optional[float] = None
    c_ref: Optional[float] = None
    b_ref: Optional[float] = None
    moment_origin: Optional[tuple[float, float, float]] = None
    induced_drag_model: InducedDragModel = "none"
    oswald_efficiency: float = Field(default=0.85, gt=0.0, le=1.0)
    stall_model: StallModel = "none"
    cg_x_over_c: Optional[float] = None

    @model_validator(mode="after")
    def _positive_reference(self) -> AeroParametersModel:
        for name in ("s_ref", "c_ref", "b_ref"):
            value = getattr(self, name)
            if value is None:
                continue
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0.")
        if self.moment_origin is not None:
            origin = np.asarray(self.moment_origin, dtype=np.float64)
            if origin.shape != (3,) or not np.all(np.isfinite(origin)):
                raise ValueError("moment_origin must be three finite coordinates.")
        if self.stall_model != "none":
            raise ValueError("stall_model 'none' is the only accepted stall model.")
        if self.cg_x_over_c is not None and not np.isfinite(self.cg_x_over_c):
            raise ValueError("cg_x_over_c must be finite.")
        return self

    def to_typed_dict(self) -> AeroParameters:
        payload: AeroParameters = {
            "induced_drag_model": self.induced_drag_model,
            "oswald_efficiency": float(self.oswald_efficiency),
            "stall_model": self.stall_model,
        }
        if self.cg_x_over_c is not None:
            payload["cg_x_over_c"] = float(self.cg_x_over_c)
        if self.s_ref is not None:
            payload["s_ref"] = float(self.s_ref)
        if self.c_ref is not None:
            payload["c_ref"] = float(self.c_ref)
        if self.b_ref is not None:
            payload["b_ref"] = float(self.b_ref)
        if self.moment_origin is not None:
            payload["moment_origin"] = (
                float(self.moment_origin[0]),
                float(self.moment_origin[1]),
                float(self.moment_origin[2]),
            )
        return payload


def validate_aero_parameters(parameters: AeroParameters | None) -> AeroParameters:
    """Fill defaults and reject an illegal reference or an unknown model."""

    return AeroParametersModel.model_validate(dict(parameters or {})).to_typed_dict()
