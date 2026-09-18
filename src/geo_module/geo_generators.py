"""
geo_generators.py

Vectorized analytical geometry generators.

No Python loops are used in the numerical NACA/wing calculations.
"""

from __future__ import annotations

from typing import Final

import numpy as np


_EPS: Final[float] = 1e-12


class GeometryGenerationError(ValueError):
    """Raised when analytical geometry cannot be generated safely."""


def _validate_naca_code(code: str) -> str:
    if not isinstance(code, str):
        raise GeometryGenerationError("NACA code must be a string.")

    code = code.strip()

    if len(code) != 4 or not code.isdigit():
        raise GeometryGenerationError(
            "NACA code must contain exactly four digits."
        )

    return code


def generate_naca_4digit(
    code: str,
    num_points: int = 100,
    chord: float = 1.0,
) -> np.ndarray:
    """
    Generate a NACA 4-digit airfoil.

    Output ordering:

        trailing edge
            ↓
        upper surface
            ↓
        leading edge
            ↓
        lower surface
            ↓
        trailing edge

    Returns
    -------
    np.ndarray
        Shape (N, 2), dtype float64.
    """

    code = _validate_naca_code(code)

    if not isinstance(num_points, (int, np.integer)):
        raise GeometryGenerationError("num_points must be an integer.")

    if num_points < 20:
        raise GeometryGenerationError(
            "num_points must be at least 20."
        )

    if num_points > 100_000:
        raise GeometryGenerationError(
            "num_points exceeds the maximum supported resolution."
        )

    if not np.isfinite(chord) or chord <= 0.0:
        raise GeometryGenerationError(
            "Chord must be a finite value greater than zero."
        )

    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0

    # Cosine spacing:
    # x goes from LE -> TE.
    n_half = (num_points + 1) // 2

    beta = np.linspace(
        0.0,
        np.pi,
        n_half,
        dtype=np.float64,
    )

    x = 0.5 * (1.0 - np.cos(beta))

    # Closed trailing-edge NACA thickness equation (-0.1036 so yt(1) = 0).
    yt = 5.0 * t * (
        0.2969 * np.sqrt(np.maximum(x, 0.0))
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1036 * x**4
    )

    if m == 0.0 or p == 0.0:
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
    else:
        forward = x < p

        yc = np.where(
            forward,
            m / p**2 * (2.0 * p * x - x**2),
            m / (1.0 - p) ** 2
            * ((1.0 - 2.0 * p) + 2.0 * p * x - x**2),
        )

        dyc_dx = np.where(
            forward,
            2.0 * m / p**2 * (p - x),
            2.0 * m / (1.0 - p) ** 2 * (p - x),
        )

    theta = np.arctan(dyc_dx)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)

    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # Convert to chord units.
    xu *= chord
    yu *= chord
    xl *= chord
    yl *= chord

    # We currently have LE -> TE.
    # Required output is TE -> upper -> LE -> lower -> TE.
    #
    # Upper:
    #   TE -> LE
    #
    # Lower:
    #   LE -> TE
    #
    upper = np.column_stack(
        (xu[::-1], yu[::-1])
    )

    lower = np.column_stack(
        (xl[1:], yl[1:])
    )

    points = np.vstack((upper, lower))

    if np.linalg.norm(points[0] - points[-1]) > _EPS:
        points = np.vstack((points, points[:1]))

    if not np.all(np.isfinite(points)):
        raise GeometryGenerationError(
            "NACA generation produced non-finite coordinates."
        )

    return np.ascontiguousarray(points, dtype=np.float64)


def generate_flat_plate(
    chord: float = 1.0,
    num_points: int = 100,
    thickness: float | None = None,
) -> np.ndarray:
    """
    Generate a closed 2D flat-plate contour.

    The contour is represented as a very thin closed polygon rather than
    as a zero-area mathematical line, because downstream mesh algorithms
    require a valid boundary.
    """

    if not np.isfinite(chord) or chord <= 0.0:
        raise GeometryGenerationError(
            "Flat-plate chord must be greater than zero."
        )

    if num_points < 20:
        raise GeometryGenerationError(
            "num_points must be at least 20."
        )

    if thickness is None:
        thickness = max(chord * 1e-6, _EPS)
    elif not np.isfinite(thickness) or thickness < 0.0:
        raise GeometryGenerationError(
            "Flat-plate thickness must be finite and non-negative."
        )
    elif thickness == 0.0:
        thickness = max(chord * 1e-6, _EPS)

    x = np.linspace(
        0.0,
        chord,
        num_points // 2,
        dtype=np.float64,
    )

    upper = np.column_stack(
        (x[::-1], np.full_like(x, thickness / 2.0))
    )

    lower = np.column_stack(
        (x[1:], np.full_like(x[1:], -thickness / 2.0))
    )

    points = np.vstack((upper, lower, upper[:1]))

    return np.ascontiguousarray(points, dtype=np.float64)


def generate_3d_parametric_wing(
    airfoil_pts: np.ndarray,
    span: float,
    sweep_deg: float,
    dihedral_deg: float,
    num_sections: int = 10,
) -> np.ndarray:
    """
    Extrude a 2D airfoil along the span to create a 3D wing skin.

    Parameters
    ----------
    airfoil_pts:
        Shape (N, 2), ordered continuously around the airfoil.
        Coordinates are assumed to be normalized to chord = 1.

    span:
        Full wing span.

    sweep_deg:
        Leading-edge sweep angle.

    dihedral_deg:
        Dihedral angle.

    num_sections:
        Number of spanwise airfoil sections.

    Returns
    -------
    np.ndarray
        Shape (F, 3, 3), where each row is one triangular facet.

    Notes
    -----
    The generated wing is a surface skin. Root and tip boundaries are
    intentionally not capped, so the returned skin is not watertight.
    """

    airfoil_pts = np.asarray(airfoil_pts, dtype=np.float64)

    if airfoil_pts.ndim != 2 or airfoil_pts.shape[1] != 2:
        raise GeometryGenerationError(
            "airfoil_pts must have shape (N, 2)."
        )

    if airfoil_pts.shape[0] < 4:
        raise GeometryGenerationError(
            "Airfoil must contain at least four points."
        )

    if not np.all(np.isfinite(airfoil_pts)):
        raise GeometryGenerationError(
            "Airfoil contains NaN or infinite coordinates."
        )

    if not np.isfinite(span) or span <= 0.0:
        raise GeometryGenerationError(
            "Span must be finite and greater than zero."
        )

    if not np.isfinite(sweep_deg):
        raise GeometryGenerationError(
            "Sweep angle must be finite."
        )

    if not np.isfinite(dihedral_deg):
        raise GeometryGenerationError(
            "Dihedral angle must be finite."
        )

    if not isinstance(num_sections, (int, np.integer)):
        raise GeometryGenerationError(
            "num_sections must be an integer."
        )

    if num_sections < 2:
        raise GeometryGenerationError(
            "num_sections must be at least 2."
        )

    if num_sections > 10_000:
        raise GeometryGenerationError(
            "num_sections exceeds the supported maximum."
        )

    # Remove an explicitly repeated final point because the strip
    # connectivity below handles the periodic airfoil boundary itself.
    if np.linalg.norm(airfoil_pts[0] - airfoil_pts[-1]) < _EPS:
        airfoil = airfoil_pts[:-1]
    else:
        airfoil = airfoil_pts

    n_profile = airfoil.shape[0]

    if n_profile < 4:
        raise GeometryGenerationError(
            "Airfoil contains insufficient unique points."
        )

    # Spanwise stations from left tip to right tip.
    y = np.linspace(
        -span / 2.0,
        span / 2.0,
        num_sections,
        dtype=np.float64,
    )

    sweep = np.deg2rad(sweep_deg)
    dihedral = np.deg2rad(dihedral_deg)

    # Sweep displacement is referenced to the airfoil leading edge.
    sweep_x = np.abs(y) * np.tan(sweep)

    # Dihedral rises with distance from centerline.
    z = np.abs(y) * np.tan(dihedral)

    # Shape:
    #   (sections, profile_points, xyz)
    vertices = np.empty(
        (num_sections, n_profile, 3),
        dtype=np.float64,
    )

    vertices[:, :, 0] = (
        airfoil[None, :, 0] + sweep_x[:, None]
    )

    vertices[:, :, 1] = y[:, None]

    vertices[:, :, 2] = (
        airfoil[None, :, 1] + z[:, None]
    )

    # Two triangles per quad.
    #
    # Section i and i+1:
    #
    # p0 ---- p1
    # |     / |
    # |   /   |
    # p3 ---- p2
    #
    # Fully vectorized connectivity.
    section_ids = np.arange(
        num_sections - 1,
        dtype=np.int64,
    )

    profile_ids = np.arange(
        n_profile,
        dtype=np.int64,
    )

    next_profile_ids = (
        (profile_ids + 1) % n_profile
    )

    s, p = np.meshgrid(
        section_ids,
        profile_ids,
        indexing="ij",
    )

    pn = next_profile_ids[p]

    v00 = vertices[s, p]
    v10 = vertices[s + 1, p]
    v11 = vertices[s + 1, pn]

    v01 = vertices[s, pn]

    tri_a = np.stack(
        (v00, v10, v11),
        axis=1,
    )

    tri_b = np.stack(
        (v00, v11, v01),
        axis=1,
    )

    facets = np.concatenate(
        (tri_a, tri_b),
        axis=0,
    )

    facets = facets.reshape(-1, 3, 3)

    if not np.all(np.isfinite(facets)):
        raise GeometryGenerationError(
            "Wing generation produced non-finite coordinates."
        )

    # Zero-area triangle detection.
    edges = facets[:, 1:] - facets[:, :1]

    cross = np.cross(
        edges[:, 0],
        edges[:, 1],
    )

    double_area = np.linalg.norm(
        cross,
        axis=1,
    )

    if np.any(double_area <= _EPS):
        raise GeometryGenerationError(
            "Generated wing contains zero-area triangular facets."
        )

    return np.ascontiguousarray(
        facets,
        dtype=np.float64,
    )


def generate_cylinder(
    radius: float = 0.5,
    length: float = 2.0,
    segments: int = 48,
    rings: int = 12,
) -> np.ndarray:
    """
    Generate a closed cylindrical surface.

    The cylinder axis is X.

    Returns triangular facets with shape (F, 3, 3).
    """

    if not np.isfinite(radius) or radius <= 0.0:
        raise GeometryGenerationError(
            "Cylinder radius must be greater than zero."
        )

    if not np.isfinite(length) or length <= 0.0:
        raise GeometryGenerationError(
            "Cylinder length must be greater than zero."
        )

    if segments < 8:
        raise GeometryGenerationError(
            "Cylinder requires at least 8 circumferential segments."
        )

    if rings < 1:
        raise GeometryGenerationError(
            "Cylinder requires at least one longitudinal ring."
        )

    theta = np.linspace(
        0.0,
        2.0 * np.pi,
        segments,
        endpoint=False,
        dtype=np.float64,
    )

    x = np.linspace(
        -length / 2.0,
        length / 2.0,
        rings + 1,
        dtype=np.float64,
    )

    xx, tt = np.meshgrid(
        x,
        theta,
        indexing="ij",
    )

    yy = radius * np.cos(tt)
    zz = radius * np.sin(tt)

    vertices = np.stack(
        (xx, yy, zz),
        axis=-1,
    )

    # Side surface.
    ring_id = np.arange(
        rings,
        dtype=np.int64,
    )[:, None]

    seg_id = np.arange(
        segments,
        dtype=np.int64,
    )[None, :]

    seg_next = (
        seg_id + 1
    ) % segments

    p00 = vertices[ring_id, seg_id]
    p10 = vertices[ring_id + 1, seg_id]
    p11 = vertices[ring_id + 1, seg_next]
    p01 = vertices[ring_id, seg_next]

    side_a = np.stack(
        (p00, p10, p11),
        axis=2,
    )

    side_b = np.stack(
        (p00, p11, p01),
        axis=2,
    )

    side = np.concatenate(
        (side_a, side_b),
        axis=0,
    ).reshape(-1, 3, 3)

    # End caps.
    cap_angles = theta

    y_cap = radius * np.cos(cap_angles)
    z_cap = radius * np.sin(cap_angles)

    left_center = np.array(
        [-length / 2.0, 0.0, 0.0],
        dtype=np.float64,
    )

    right_center = np.array(
        [length / 2.0, 0.0, 0.0],
        dtype=np.float64,
    )

    left_ring = np.column_stack(
        (
            np.full(segments, -length / 2.0),
            y_cap,
            z_cap,
        )
    )

    right_ring = np.column_stack(
        (
            np.full(segments, length / 2.0),
            y_cap,
            z_cap,
        )
    )

    next_ids = (
        np.arange(segments) + 1
    ) % segments

    left_caps = np.stack(
        (
            np.broadcast_to(
                left_center,
                (segments, 3),
            ),
            left_ring[next_ids],
            left_ring,
        ),
        axis=1,
    )

    right_caps = np.stack(
        (
            np.broadcast_to(
                right_center,
                (segments, 3),
            ),
            right_ring,
            right_ring[next_ids],
        ),
        axis=1,
    )

    facets = np.concatenate(
        (
            side,
            left_caps,
            right_caps,
        ),
        axis=0,
    )

    return np.ascontiguousarray(
        facets,
        dtype=np.float64,
    )