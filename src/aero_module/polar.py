"""
Polar slopes from operating points Module 3 has already solved.

Har angle pehle se solve hai. Yahan sirf Cl, Cd, aur Cm ki lines fit
hoti hain: aerodynamic center, ideal slope, aur drag polar.
Every angle is already solved. This module only fits lift, drag, and
moment: aerodynamic center, the ideal slope, and the drag polar.
"""

from __future__ import annotations

import numpy as np

from math_module.math_skeleton import MeshOutput
from physics_module.physics_skeleton import PhysicsOutput

from .aero_skeleton import (
    AeroParameters,
    AeroPolarOutput,
    InconsistentAeroInputError,
)
from .operate import reduce_operating_point


def _fit_line(x_values: np.ndarray, y_values: np.ndarray) -> tuple[float, float]:
    """Return slope and intercept. NaN when the sample cannot define a line."""

    finite = np.isfinite(x_values) & np.isfinite(y_values)
    if int(np.count_nonzero(finite)) < 2 or float(np.ptp(x_values[finite])) < 1e-10:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x_values[finite], y_values[finite], 1)
    return float(slope), float(intercept)


def _fit_drag(cl: np.ndarray, cd: np.ndarray) -> tuple[float, float]:
    """Least squares of CD = CD0 + K CL^2."""

    finite = np.isfinite(cl) & np.isfinite(cd)
    if int(np.count_nonzero(finite)) < 2 or float(np.ptp(cl[finite])) < 1e-6:
        return float("nan"), float("nan")
    design = np.column_stack(
        (np.ones(int(np.count_nonzero(finite)), dtype=np.float64), cl[finite] ** 2)
    )
    coef, _, _, _ = np.linalg.lstsq(design, cd[finite], rcond=None)
    return float(coef[0]), float(coef[1])


def reduce_polar(
    mesh: MeshOutput,
    samples: list[PhysicsOutput],
    parameters: AeroParameters,
) -> AeroPolarOutput:
    """Stack one mesh at several angles and fit the report curves."""

    if not samples:
        raise InconsistentAeroInputError("A polar needs at least one solved angle.")
    points = [reduce_operating_point(mesh, sample, parameters) for sample in samples]
    order = np.argsort(np.array([point["alpha_deg"] for point in points], dtype=np.float64))
    ordered = [points[int(index)] for index in order]
    alpha = np.array([point["alpha_deg"] for point in ordered], dtype=np.float64)
    cl = np.array([point["coefficients"]["cl"] for point in ordered], dtype=np.float64)
    cd = np.array([point["coefficients"]["cd"] for point in ordered], dtype=np.float64)
    cm_reported = np.array([point["coefficients"]["cm"] for point in ordered], dtype=np.float64)
    cm_quarter = np.array(
        [point["coefficients"]["cm_quarter_chord"] for point in ordered],
        dtype=np.float64,
    )
    glide = np.array([point["l_over_d"] for point in ordered], dtype=np.float64)
    alpha_rad = np.deg2rad(alpha)

    cl_alpha, cl0 = _fit_line(alpha_rad, cl)
    cm_alpha, cm0 = _fit_line(alpha_rad, cm_quarter)
    if not np.isfinite(cl_alpha) or abs(cl_alpha) < 1e-3:
        x_ac = float("nan")
        alpha_zero = float("nan")
    else:
        x_ac = float(0.25 - cm_alpha / cl_alpha) if np.isfinite(cm_alpha) else float("nan")
        alpha_zero = float(np.rad2deg(-cl0 / cl_alpha))

    mode = ordered[0]["mode"]
    aspect = float(ordered[0]["reference"]["aspect_ratio"])
    if mode == "2D":
        ideal = float(2.0 * np.pi)
    elif np.isfinite(aspect) and aspect > 0.0:
        ideal = float(2.0 * np.pi * aspect / (aspect + 2.0))
    else:
        ideal = float("nan")
    ratio = float(cl_alpha / ideal) if np.isfinite(cl_alpha) and np.isfinite(ideal) and ideal != 0.0 else float("nan")

    cd0, drag_k = _fit_drag(cl, cd)
    oswald = float("nan")
    if mode == "3D" and np.isfinite(drag_k) and drag_k > 0.0 and np.isfinite(aspect) and aspect > 0.0:
        oswald = float(1.0 / (np.pi * aspect * drag_k))

    cg = parameters.get("cg_x_over_c")
    if cg is None or not np.isfinite(x_ac) or not np.isfinite(float(cg)):
        margin = float("nan")
    else:
        margin = float(x_ac - float(cg))

    finite_cl = np.isfinite(cl)
    if np.any(finite_cl):
        cl_index = int(np.argmax(np.where(finite_cl, cl, -np.inf)))
        cl_max = float(cl[cl_index])
        alpha_cl_max = float(alpha[cl_index])
    else:
        cl_max = float("nan")
        alpha_cl_max = float("nan")
    finite_ld = np.isfinite(glide)
    if np.any(finite_ld):
        ld_index = int(np.argmax(np.where(finite_ld, glide, -np.inf)))
        ld_max = float(glide[ld_index])
        alpha_ld = float(alpha[ld_index])
    else:
        ld_max = float("nan")
        alpha_ld = float("nan")

    return AeroPolarOutput(
        mode=mode,
        alpha_deg=np.ascontiguousarray(alpha, dtype=np.float64),
        cl=np.ascontiguousarray(cl, dtype=np.float64),
        cd=np.ascontiguousarray(cd, dtype=np.float64),
        cm=np.ascontiguousarray(cm_reported, dtype=np.float64),
        l_over_d=np.ascontiguousarray(glide, dtype=np.float64),
        cl_alpha=float(cl_alpha),
        cl_alpha_ideal=ideal,
        cl_alpha_ratio=ratio,
        cm_alpha=float(cm_alpha),
        cm0=float(cm0),
        alpha_zero_lift_deg=alpha_zero,
        cd0=float(cd0),
        drag_polar_k=float(drag_k),
        oswald_from_polar=oswald,
        aerodynamic_center_x=x_ac,
        static_margin=margin,
        cl_max=cl_max,
        alpha_cl_max_deg=alpha_cl_max,
        l_over_d_max=ld_max,
        alpha_l_over_d_max_deg=alpha_ld,
        samples=ordered,
    )
