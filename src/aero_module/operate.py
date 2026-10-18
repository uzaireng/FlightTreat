"""
One-angle aerodynamic reduction.

Physics ke Cp aur loads ko coefficients, newtons, aur charts mein
badalta hai. Panels, influence matrix, aur boundary layer yahan
dobara nahi bante.
Turns one PhysicsOutput into report coefficients, newtons, and chart
arrays. It does not rebuild panels, the influence matrix, or the
boundary layer.
"""

from __future__ import annotations

import numpy as np

from math_module.math_skeleton import MeshOutput
from physics_module.physics_skeleton import PhysicsOutput

from .aero_skeleton import (
    AeroOutput,
    AeroParameters,
    ForceCoefficients,
    InconsistentAeroInputError,
    PressureMarkers,
    SpanwiseLoad,
    ViscousSummary,
)
from .reference import resolve_reference


_LD_FLOOR = 1.0e-6
_CP_FLOOR = 1.0e-4


def _wind_axes(alpha: float, mode: str) -> tuple[np.ndarray, np.ndarray]:
    # Lift aur drag ki direction, geometry axes mein.
    # Lift and drag directions in geometry axes.
    if mode == "2D":
        lift = np.array([-np.sin(alpha), np.cos(alpha)], dtype=np.float64)
        drag = np.array([np.cos(alpha), np.sin(alpha)], dtype=np.float64)
    else:
        lift = np.array([-np.sin(alpha), 0.0, np.cos(alpha)], dtype=np.float64)
        drag = np.array([np.cos(alpha), 0.0, np.sin(alpha)], dtype=np.float64)
    return lift, drag


def _shift_moment(
    cm_quarter: float,
    cl: float,
    cd: float,
    alpha: float,
    chord: float,
    mode: str,
    quarter: np.ndarray,
    requested: np.ndarray,
) -> float:
    """Move the quarter-chord moment to the caller origin. No panel touch."""

    lift_hat, drag_hat = _wind_axes(alpha, mode)
    force = cl * lift_hat + cd * drag_hat
    delta = np.asarray(quarter, dtype=np.float64) - np.asarray(requested, dtype=np.float64)
    if mode == "2D":
        return float(cm_quarter + (delta[0] / chord) * force[1] - (delta[1] / chord) * force[0])
    return float(cm_quarter + (delta[0] / chord) * force[2] - (delta[2] / chord) * force[0])


def _pressure(cp: np.ndarray, x_over_c: np.ndarray | None) -> PressureMarkers:
    finite = np.isfinite(cp)
    if not np.any(finite):
        return PressureMarkers(cp_min=float("nan"), cp_max=float("nan"), cp_min_x_over_c=float("nan"))
    cp_min = float(np.min(cp[finite]))
    cp_max = float(np.max(cp[finite]))
    station = float("nan")
    if x_over_c is not None and x_over_c.shape == cp.shape:
        index = int(np.argmin(np.where(finite, cp, np.inf)))
        station = float(x_over_c[index])
    return PressureMarkers(cp_min=cp_min, cp_max=cp_max, cp_min_x_over_c=station)


def _viscous(physics: PhysicsOutput) -> ViscousSummary:
    # Boundary layer march nahi hoti. Jo stations physics ne likhe hain,
    # wahi copy hote hain. 3D par ibl maujood nahi, is liye card khali hai.
    # Stations are copied. A 3D solve has no ibl block, so the card is empty.
    ibl = physics["ibl"]
    if ibl is None:
        return ViscousSummary(
            available=False,
            separated=False,
            transition_s_upper=float("nan"),
            transition_s_lower=float("nan"),
            separation_s_upper=float("nan"),
            separation_s_lower=float("nan"),
        )
    return ViscousSummary(
        available=True,
        separated=bool(ibl["separated"]),
        transition_s_upper=float(ibl["transition_s_upper"]),
        transition_s_lower=float(ibl["transition_s_lower"]),
        separation_s_upper=float(ibl["separation_s_upper"]),
        separation_s_lower=float(ibl["separation_s_lower"]),
    )


def spanwise_lift(mesh: MeshOutput, physics: PhysicsOutput, alpha: float) -> SpanwiseLoad | None:
    """
    3D section lift along the span from the existing pressure field.

    Naye panels nahi bante. Centroids ko y-bins mein jama kiya jata hai.
    Existing panel Cp is binned along y. 2D returns None.
    """

    if mesh["mode"] != "3D":
        return None
    panel = mesh["panel_3d"]
    if panel is None:
        return None
    centroids = np.asarray(panel["centroids"], dtype=np.float64)
    normals = np.asarray(panel["normals"], dtype=np.float64)
    areas = np.asarray(panel["areas"], dtype=np.float64)
    cp = np.asarray(physics["inviscid"]["cp_inviscid"], dtype=np.float64)
    count = int(centroids.shape[0])
    if count < 1 or cp.shape != (count,):
        raise InconsistentAeroInputError("Surface Cp does not match the 3D panel count.")
    y = centroids[:, 1]
    span = float(np.ptp(y))
    if not np.isfinite(span) or span <= 1e-12:
        return SpanwiseLoad(
            y=np.zeros(0, dtype=np.float64),
            cl=np.zeros(0, dtype=np.float64),
            chord=np.zeros(0, dtype=np.float64),
        )
    n_bins = int(min(24, max(4, count // 8)))
    edges = np.linspace(float(np.min(y)), float(np.max(y)), n_bins + 1)
    bin_id = np.clip(np.digitize(y, edges) - 1, 0, n_bins - 1)
    lift_hat = np.array([-np.sin(alpha), 0.0, np.cos(alpha)], dtype=np.float64)
    panel_lift = -cp * (normals @ lift_hat) * areas
    lift_sum = np.bincount(bin_id, weights=panel_lift, minlength=n_bins)
    x = centroids[:, 0]
    x_min = np.full(n_bins, np.inf, dtype=np.float64)
    x_max = np.full(n_bins, -np.inf, dtype=np.float64)
    np.minimum.at(x_min, bin_id, x)
    np.maximum.at(x_max, bin_id, x)
    chord = x_max - x_min
    width = np.diff(edges)
    counts = np.bincount(bin_id, minlength=n_bins)
    live = counts > 0
    cl = np.full(n_bins, np.nan, dtype=np.float64)
    denom = chord * width
    ok = live & np.isfinite(denom) & (denom > 1e-12)
    cl[ok] = lift_sum[ok] / denom[ok]
    centers = 0.5 * (edges[:-1] + edges[1:])
    return SpanwiseLoad(
        y=np.ascontiguousarray(centers[live], dtype=np.float64),
        cl=np.ascontiguousarray(cl[live], dtype=np.float64),
        chord=np.ascontiguousarray(chord[live], dtype=np.float64),
    )


def reduce_operating_point(
    mesh: MeshOutput,
    physics: PhysicsOutput,
    parameters: AeroParameters,
) -> AeroOutput:
    """Build one AeroOutput. Inputs are read only."""

    reference, warnings, s_phys, c_phys = resolve_reference(mesh, physics, parameters)
    mode = mesh["mode"]
    loads = physics["loads"]
    cp_source = physics["inviscid"]["cp_inviscid"]
    cp = np.array(cp_source, dtype=np.float64, copy=True)
    if mode == "2D":
        panel = mesh["panel_2d"]
        if panel is None or cp.shape != (int(panel["length"].shape[0]),):
            raise InconsistentAeroInputError("Surface Cp does not match the 2D panel count.")
        x_over_c = np.array(panel["x_over_c"], dtype=np.float64, copy=True)
    else:
        panel3 = mesh["panel_3d"]
        if panel3 is None or cp.shape != (int(panel3["areas"].shape[0]),):
            raise InconsistentAeroInputError("Surface Cp does not match the 3D panel count.")
        x_over_c = None

    area = float(reference["s_ref"])
    chord = float(reference["c_ref"])
    scale = s_phys / area
    cl = float(loads["cl"]) * scale
    cd_pressure = float(loads["cd_pressure"]) * scale
    friction = float(loads["cd_friction"])
    cd_friction = float("nan") if not np.isfinite(friction) else friction * scale
    cd = cd_pressure if not np.isfinite(cd_friction) else cd_pressure + cd_friction

    if mode == "2D":
        cm_quarter = float(loads["cm_quarter_chord"]) * (c_phys / chord) ** 2
    else:
        cm_quarter = float(loads["cm_quarter_chord"]) * (s_phys * c_phys) / (area * chord)

    alpha_deg = float(physics["freestream"]["alpha_deg"])
    alpha = float(np.deg2rad(alpha_deg))
    quarter = np.asarray(loads["moment_origin_qc"], dtype=np.float64)
    if quarter.shape == (2,):
        quarter = np.array([quarter[0], quarter[1], 0.0], dtype=np.float64)
    if reference["moment_about"] == "quarter_chord":
        cm = cm_quarter
    else:
        cm = _shift_moment(
            cm_quarter,
            cl,
            cd_pressure if not np.isfinite(cd) else cd,
            alpha,
            chord,
            mode,
            quarter,
            reference["moment_origin"],
        )

    model = parameters.get("induced_drag_model", "none")
    efficiency = float(parameters.get("oswald_efficiency", 0.85))
    if mode == "2D":
        cd_induced = 0.0
        if model == "lifting_line":
            warnings.append("A 2D section has no induced drag. cd_induced is 0.")
    elif model == "none":
        cd_induced = float("nan")
    else:
        aspect = float(reference["aspect_ratio"])
        if not np.isfinite(aspect) or aspect <= 0.0:
            cd_induced = float("nan")
            warnings.append("Lifting-line induced drag needs a finite aspect ratio.")
        else:
            cd_induced = float(cl * cl / (np.pi * aspect * efficiency))
            warnings.append(
                "cd_induced is a lifting-line estimate and is not subtracted from cd."
            )

    l_over_d = float("nan")
    if np.isfinite(cd) and abs(cd) >= _LD_FLOOR and np.isfinite(cl):
        l_over_d = float(cl / cd)

    cos_a = float(np.cos(alpha))
    sin_a = float(np.sin(alpha))
    cn = float(cl * cos_a + cd * sin_a)
    ca = float(cd * cos_a - cl * sin_a)
    center = float("nan")
    if np.isfinite(cl) and abs(cl) >= _CP_FLOOR and np.isfinite(cm_quarter):
        center = float(0.25 - cm_quarter / cl)

    if physics["regime"] == "inviscid":
        cl_inviscid = cl
        cd_inviscid = cd
    else:
        cl_inviscid = float("nan")
        cd_inviscid = float("nan")

    q_inf = float(reference["q_inf"])
    if mode == "2D":
        lift = cl * q_inf * chord
        drag = cd * q_inf * chord
        moment = cm * q_inf * chord * chord
    else:
        lift = cl * q_inf * area
        drag = cd * q_inf * area
        moment = cm * q_inf * area * chord

    notes = {
        "induced_drag_model": str(model),
        "reference": reference["reference_note"],
        "stall_model": "none",
    }
    warnings = list(physics["warnings"]) + warnings
    return AeroOutput(
        mode=mode,
        regime=physics["regime"],
        alpha_deg=alpha_deg,
        reynolds=float(physics["freestream"]["reynolds"]),
        mach=float(physics["freestream"]["mach"]),
        coefficients=ForceCoefficients(
            cl=float(cl),
            cd=float(cd),
            cd_pressure=float(cd_pressure),
            cd_friction=float(cd_friction),
            cd_induced=float(cd_induced),
            cm=float(cm),
            cm_quarter_chord=float(cm_quarter),
            cl_inviscid=float(cl_inviscid),
            cd_inviscid=float(cd_inviscid),
            cn=cn,
            ca=ca,
            center_of_pressure_x=center,
        ),
        dimensional={
            "lift": float(lift),
            "drag": float(drag),
            "pitching_moment": float(moment),
        },
        reference=reference,
        cp_surface=cp,
        x_over_c=x_over_c,
        pressure=_pressure(cp, x_over_c),
        viscous=_viscous(physics),
        spanwise=spanwise_lift(mesh, physics, alpha),
        aerodynamic_center_x=float("nan"),
        static_margin=float("nan"),
        l_over_d=l_over_d,
        notes=notes,
        warnings=warnings,
    )
