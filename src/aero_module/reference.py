"""
Reference area, chord, and dynamic pressure for one aero reduction.

Yeh file flow dobara solve nahi karti. Jo area aur chord Module 3
istamal kar chuka hai, usi ko caller ki override se scale karti hai.
This file does not solve the flow. It only chooses the reference
quantities used to rescale coefficients Module 3 already integrated.
"""

from __future__ import annotations

import numpy as np

from math_module.math_skeleton import MeshOutput
from physics_module.physics_skeleton import PhysicsOutput

from .aero_skeleton import (
    AeroParameters,
    InvalidReferenceQuantityError,
    InconsistentAeroInputError,
    ReferenceState,
)


def _positive(name: str, value: float) -> float:
    if not np.isfinite(value) or value <= 0.0:
        raise InvalidReferenceQuantityError(f"{name} must be finite and > 0.")
    return float(value)


def _chord_3d(mesh: MeshOutput) -> tuple[float, str]:
    """Mid-span chord, matching the length Module 3 used for moments.

    3D ``PhysicsOutput`` chord store nahi karta, is liye wahi mid-span
    x-extent dobara nikala jata hai. The physics payload does not store
    that length, so the same mid-span x-extent is estimated here.
    """

    panel = mesh["panel_3d"]
    if panel is None:
        raise InconsistentAeroInputError("3D mesh is missing panel_3d.")
    centroids = np.asarray(panel["centroids"], dtype=np.float64)
    if panel["is_closed"]:
        extents = np.ptp(centroids, axis=0)
        length = float(extents[int(np.argmax(extents))])
        return _positive("c_phys", length), "closed-body longest centroid extent"
    span = float(np.ptp(centroids[:, 1]))
    y_mid = 0.5 * (float(np.min(centroids[:, 1])) + float(np.max(centroids[:, 1])))
    window = max(span / 20.0, 1e-8)
    near = np.abs(centroids[:, 1] - y_mid) <= window
    if not np.any(near):
        near = np.ones(centroids.shape[0], dtype=bool)
    root_chord = float(np.ptp(centroids[near, 0]))
    if root_chord <= 0.0:
        root_chord = float(np.ptp(centroids[:, 0]))
    return _positive("c_phys", root_chord), "mid-span x-extent"


def _span_3d(mesh: MeshOutput) -> float:
    panel = mesh["panel_3d"]
    if panel is None:
        raise InconsistentAeroInputError("3D mesh is missing panel_3d.")
    centroids = np.asarray(panel["centroids"], dtype=np.float64)
    return float(np.ptp(centroids[:, 1]))


def resolve_reference(
    mesh: MeshOutput,
    physics: PhysicsOutput,
    parameters: AeroParameters,
) -> tuple[ReferenceState, list[str], float, float]:
    """
    Return the reference card, warnings, and ``(S_phys, c_phys)``.

    2D mein area aur chord ek hi lambai hain. Dono alag set hon to error.
    In 2D the area and the chord are the same length.
    """

    if mesh["mode"] != physics["mode"]:
        raise InconsistentAeroInputError(
            f"Mesh mode {mesh['mode']} does not match physics mode {physics['mode']}."
        )
    mode = mesh["mode"]
    loads = physics["loads"]
    s_phys = _positive("S_phys", float(loads["s_ref"]))
    warnings: list[str] = []

    if mode == "2D":
        panel = mesh["panel_2d"]
        if panel is None:
            raise InconsistentAeroInputError("2D mesh is missing panel_2d.")
        c_phys = _positive("c_phys", float(panel["chord"]))
        s_set = parameters.get("s_ref")
        c_set = parameters.get("c_ref")
        if s_set is not None and c_set is not None:
            s_user = _positive("s_ref", float(s_set))
            c_user = _positive("c_ref", float(c_set))
            if abs(s_user - c_user) > 1e-8 * max(s_user, c_user, 1.0):
                raise InvalidReferenceQuantityError(
                    "In 2D, s_ref and c_ref are the same chord."
                )
            area = s_user
            chord = c_user
        elif s_set is not None:
            area = _positive("s_ref", float(s_set))
            chord = area
        elif c_set is not None:
            chord = _positive("c_ref", float(c_set))
            area = chord
        else:
            area = s_phys
            chord = c_phys
        span = float("nan")
        note = "2D coefficients are per unit span."
    else:
        if mesh["panel_3d"] is None:
            raise InconsistentAeroInputError("3D mesh is missing panel_3d.")
        c_phys, chord_note = _chord_3d(mesh)
        area = s_phys if parameters.get("s_ref") is None else _positive("s_ref", float(parameters["s_ref"]))
        if parameters.get("c_ref") is None:
            chord = c_phys
            warnings.append(
                f"3D reference chord is the {chord_note}, not a mean aerodynamic chord."
            )
        else:
            chord = _positive("c_ref", float(parameters["c_ref"]))
        if parameters.get("b_ref") is None:
            span = _span_3d(mesh)
            if not np.isfinite(span) or span <= 0.0:
                span = float("nan")
                warnings.append("3D span is missing, so aspect ratio is undefined.")
        else:
            span = _positive("b_ref", float(parameters["b_ref"]))
        note = "3D coefficients use the selected planform area."

    aspect = float("nan")
    if np.isfinite(span) and span > 0.0 and area > 0.0:
        aspect = float(span * span / area)

    flow = physics["freestream"]
    speed = _positive("v_inf", float(flow["v_inf"]))
    rho = _positive("rho", float(flow["rho"]))
    q_inf = 0.5 * rho * speed * speed

    origin = np.asarray(loads["moment_origin_qc"], dtype=np.float64)
    if parameters.get("moment_origin") is None:
        if origin.shape == (2,):
            stored = np.array([origin[0], origin[1], 0.0], dtype=np.float64)
        else:
            stored = np.array(origin, dtype=np.float64, copy=True)
        about = "quarter_chord"
    else:
        raw = parameters["moment_origin"]
        stored = np.array([raw[0], raw[1], raw[2]], dtype=np.float64)
        about = "user"

    state = ReferenceState(
        s_ref=float(area),
        c_ref=float(chord),
        b_ref=float(span),
        aspect_ratio=float(aspect),
        q_inf=float(q_inf),
        rho=float(rho),
        v_inf=float(speed),
        moment_origin=stored,
        moment_about=about,
        reference_note=note,
    )
    return state, warnings, s_phys, c_phys
