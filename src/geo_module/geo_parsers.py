"""
geo_parsers.py

2D/3D geometry file parsing and geometry validation.
"""

from __future__ import annotations

import re
import struct
import xml.etree.ElementTree as ET
from typing import List, Tuple

import numpy as np


_EPS = 1e-10


class GeometryParseError(ValueError):
    """Raised when a geometry file cannot be parsed safely."""


class InvalidGeometryError(ValueError):
    """Raised when parsed geometry is mathematically invalid."""


# ============================================================================
# Generic validation
# ============================================================================

def _validate_points_2d(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 2:
        raise InvalidGeometryError(
            "2D geometry must have shape (N, 2)."
        )

    if points.shape[0] < 3:
        raise InvalidGeometryError(
            "2D geometry requires at least three points."
        )

    if not np.all(np.isfinite(points)):
        raise InvalidGeometryError(
            "2D geometry contains NaN or infinite coordinates."
        )

    # Remove consecutive duplicates.
    delta = np.diff(
        points,
        axis=0,
    )

    keep = np.concatenate(
        (
            [True],
            np.linalg.norm(delta, axis=1) > _EPS,
        )
    )

    points = points[keep]

    if points.shape[0] < 3:
        raise InvalidGeometryError(
            "2D geometry contains insufficient unique points."
        )

    return np.ascontiguousarray(points)


def _normalize_2d(points: np.ndarray) -> np.ndarray:
    """
    Normalize the geometry to unit chord.

    X/Y are translated so that the bounding-box center is at the origin.
    """

    points = _validate_points_2d(points)

    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)

    span = maxs - mins
    chord = float(span[0])

    if chord <= _EPS:
        raise InvalidGeometryError(
            "2D geometry has zero X extent and cannot be normalized."
        )

    center = 0.5 * (mins + maxs)

    normalized = (
        points - center
    ) / chord

    return np.ascontiguousarray(
        normalized,
        dtype=np.float64,
    )


def _validate_facets_3d(
    facets: np.ndarray,
) -> np.ndarray:

    facets = np.asarray(
        facets,
        dtype=np.float64,
    )

    if facets.ndim != 3 or facets.shape[1:] != (3, 3):
        raise InvalidGeometryError(
            "3D mesh must have shape (F, 3, 3)."
        )

    if facets.shape[0] == 0:
        raise InvalidGeometryError(
            "3D mesh contains no facets."
        )

    if not np.all(np.isfinite(facets)):
        raise InvalidGeometryError(
            "3D mesh contains NaN or infinite coordinates."
        )

    e1 = facets[:, 1] - facets[:, 0]
    e2 = facets[:, 2] - facets[:, 0]

    cross = np.cross(e1, e2)

    double_area = np.linalg.norm(
        cross,
        axis=1,
    )

    if np.any(double_area <= _EPS):
        bad_count = int(
            np.count_nonzero(double_area <= _EPS)
        )

        raise InvalidGeometryError(
            f"Mesh contains {bad_count} zero-area or degenerate triangles."
        )

    return np.ascontiguousarray(
        facets,
        dtype=np.float64,
    )


def _normalize_3d(facets: np.ndarray) -> np.ndarray:
    """
    Translate the mesh bounding-box center to origin and scale the largest
    bounding-box dimension to 1.
    """

    facets = _validate_facets_3d(facets)

    mins = np.min(
        facets.reshape(-1, 3),
        axis=0,
    )

    maxs = np.max(
        facets.reshape(-1, 3),
        axis=0,
    )

    center = 0.5 * (mins + maxs)

    extent = maxs - mins
    scale = float(np.max(extent))

    if scale <= _EPS:
        raise InvalidGeometryError(
            "3D mesh has zero spatial extent."
        )

    normalized = (
        facets - center
    ) / scale

    return np.ascontiguousarray(
        normalized,
        dtype=np.float64,
    )


# ============================================================================
# 2D SVG
# ============================================================================

_NUMBER_RE = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?"
)


def _extract_svg_numbers(text: str) -> np.ndarray:
    values = _NUMBER_RE.findall(text)

    if not values:
        return np.empty(0, dtype=np.float64)

    try:
        return np.asarray(
            [float(v) for v in values],
            dtype=np.float64,
        )
    except (ValueError, OverflowError) as exc:
        raise GeometryParseError(
            "SVG contains invalid numeric coordinates."
        ) from exc


def _parse_svg_path_d(d: str) -> np.ndarray:
    """
    Lightweight SVG path extraction.

    Supports:
        M/m
        L/l
        H/h
        V/v
        Z/z

    Curves are rejected rather than silently approximated.
    """

    if not isinstance(d, str) or not d.strip():
        raise GeometryParseError(
            "SVG path contains no coordinate data."
        )

    if re.search(r"[CcSsQqTtAa]", d):
        raise GeometryParseError(
            "SVG Bézier/arc paths are not supported yet. "
            "Convert the path to line segments before upload."
        )

    tokens = re.findall(
        r"[MmLlHhVvZz]|"
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
        d,
    )

    points: List[Tuple[float, float]] = []

    command = None
    current = np.array([0.0, 0.0], dtype=np.float64)
    start = current.copy()

    i = 0

    while i < len(tokens):
        token = tokens[i]

        if re.fullmatch(r"[MmLlHhVvZz]", token):
            command = token
            i += 1

            if command in ("Z", "z"):
                points.append(
                    (float(start[0]), float(start[1]))
                )

            continue

        if command is None:
            raise GeometryParseError(
                "SVG path begins with coordinates without a command."
            )

        try:
            if command in ("M", "m", "L", "l"):
                if i + 1 >= len(tokens):
                    raise GeometryParseError(
                        "Incomplete SVG coordinate pair."
                    )

                x = float(tokens[i])
                y = float(tokens[i + 1])

                i += 2

                if command.islower():
                    current = current + np.array(
                        [x, y],
                        dtype=np.float64,
                    )
                else:
                    current = np.array(
                        [x, y],
                        dtype=np.float64,
                    )

                points.append(
                    (float(current[0]), float(current[1]))
                )

                # Subsequent coordinate pairs after M behave as L.
                if command in ("M", "m"):
                    command = (
                        "l" if command == "m" else "L"
                    )

            elif command in ("H", "h"):
                x = float(tokens[i])
                i += 1

                if command == "h":
                    current[0] += x
                else:
                    current[0] = x

                points.append(
                    (float(current[0]), float(current[1]))
                )

            elif command in ("V", "v"):
                y = float(tokens[i])
                i += 1

                if command == "v":
                    current[1] += y
                else:
                    current[1] = y

                points.append(
                    (float(current[0]), float(current[1]))
                )

            else:
                raise GeometryParseError(
                    f"Unsupported SVG path command '{command}'."
                )

        except (ValueError, IndexError) as exc:
            raise GeometryParseError(
                "Invalid numeric data inside SVG path."
            ) from exc

    return _validate_points_2d(
        np.asarray(points, dtype=np.float64)
    )


def _parse_svg(content: bytes) -> np.ndarray:
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise GeometryParseError(
            "SVG XML is malformed."
        ) from exc

    points: List[np.ndarray] = []

    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1].lower()

        if tag == "path":
            d = element.attrib.get("d", "")
            if d:
                points.append(
                    _parse_svg_path_d(d)
                )

        elif tag == "polyline":
            raw = element.attrib.get("points", "")
            values = _extract_svg_numbers(raw)

            if values.size % 2 != 0:
                raise GeometryParseError(
                    "SVG polyline contains an incomplete coordinate pair."
                )

            if values.size:
                points.append(
                    values.reshape(-1, 2)
                )

        elif tag == "polygon":
            raw = element.attrib.get("points", "")
            values = _extract_svg_numbers(raw)

            if values.size % 2 != 0:
                raise GeometryParseError(
                    "SVG polygon contains an incomplete coordinate pair."
                )

            if values.size:
                points.append(
                    values.reshape(-1, 2)
                )

        elif tag == "line":
            try:
                x1 = float(element.attrib["x1"])
                y1 = float(element.attrib["y1"])
                x2 = float(element.attrib["x2"])
                y2 = float(element.attrib["y2"])
            except (KeyError, ValueError) as exc:
                raise GeometryParseError(
                    "SVG line contains invalid coordinates."
                ) from exc

            points.append(
                np.array(
                    [[x1, y1], [x2, y2]],
                    dtype=np.float64,
                )
            )

    if not points:
        raise GeometryParseError(
            "No supported SVG geometry was found."
        )

    merged = np.vstack(points)

    return _normalize_2d(merged)


# ============================================================================
# DXF
# ============================================================================

def _parse_dxf(content: bytes) -> np.ndarray:
    """
    Lightweight ASCII DXF parser.

    Supported entities:
        LINE
        LWPOLYLINE
        POLYLINE / VERTEX

    Binary DXF is rejected explicitly because attempting to interpret it as
    ASCII can produce dangerous false geometry.
    """

    if b"\x00" in content[:2048]:
        raise GeometryParseError(
            "Binary DXF is not supported by the lightweight parser. "
            "Export the DXF as ASCII DXF."
        )

    try:
        text = content.decode(
            "utf-8",
            errors="strict",
        )
    except UnicodeDecodeError as exc:
        raise GeometryParseError(
            "DXF is not valid UTF-8 ASCII text."
        ) from exc

    lines = [
        line.strip()
        for line in text.splitlines()
    ]

    if len(lines) < 4:
        raise GeometryParseError(
            "DXF file is too short."
        )

    pairs = []

    for i in range(0, len(lines) - 1, 2):
        code = lines[i]

        try:
            value = lines[i + 1]
        except IndexError:
            break

        pairs.append(
            (code, value)
        )

    points: List[Tuple[float, float]] = []

    entity = None
    x = None
    y = None

    polyline_points: List[Tuple[float, float]] = []

    for code, value in pairs:
        if code == "0":
            if entity == "LINE":
                if x is not None and y is not None:
                    points.append((x, y))

            entity = value.upper()

            x = None
            y = None

            if entity not in {
                "LINE",
                "LWPOLYLINE",
                "POLYLINE",
                "VERTEX",
            }:
                entity = None

        elif entity == "LINE":
            if code == "10":
                x = float(value)
            elif code == "20":
                y = float(value)

        elif entity == "LWPOLYLINE":
            if code == "10":
                x = float(value)
            elif code == "20":
                y = float(value)

                if x is not None:
                    polyline_points.append((x, y))
                    x = None
                    y = None

        elif entity == "VERTEX":
            if code == "10":
                x = float(value)
            elif code == "20":
                y = float(value)

                if x is not None:
                    polyline_points.append((x, y))
                    x = None
                    y = None

    if x is not None and y is not None:
        points.append((x, y))

    points.extend(polyline_points)

    if len(points) < 3:
        raise GeometryParseError(
            "DXF contains insufficient supported 2D geometry."
        )

    return _normalize_2d(
        np.asarray(points, dtype=np.float64)
    )


# ============================================================================
# 2D public parser
# ============================================================================

def parse_2d_dxf_svg(
    file_content: bytes,
    file_type: str,
) -> np.ndarray:
    """
    Parse SVG or ASCII DXF into normalized 2D coordinates.

    Output:
        (N, 2), normalized to unit X chord and centered at origin.
    """

    if not isinstance(file_content, (bytes, bytearray)):
        raise GeometryParseError(
            "file_content must be bytes."
        )

    normalized_type = (
        str(file_type)
        .strip()
        .lower()
        .lstrip(".")
    )

    if normalized_type == "svg":
        return _parse_svg(bytes(file_content))

    if normalized_type == "dxf":
        return _parse_dxf(bytes(file_content))

    raise GeometryParseError(
        "Unsupported 2D file type. Use SVG or DXF."
    )


# ============================================================================
# STL
# ============================================================================

def _looks_like_binary_stl(content: bytes) -> bool:
    if len(content) < 84:
        return False

    try:
        triangle_count = struct.unpack_from(
            "<I",
            content,
            80,
        )[0]
    except struct.error:
        return False

    expected_size = 84 + triangle_count * 50

    return (
        triangle_count > 0
        and expected_size == len(content)
    )


def _parse_binary_stl(content: bytes) -> np.ndarray:
    if len(content) < 84:
        raise GeometryParseError(
            "Binary STL is shorter than the required header."
        )

    triangle_count = struct.unpack_from(
        "<I",
        content,
        80,
    )[0]

    expected_size = 84 + triangle_count * 50

    if expected_size != len(content):
        raise GeometryParseError(
            "Binary STL size does not match its triangle count."
        )

    if triangle_count == 0:
        raise GeometryParseError(
            "Binary STL contains no triangles."
        )

    data = np.frombuffer(
        content,
        dtype=np.uint8,
        offset=84,
    )

    records = data.reshape(
        triangle_count,
        50,
    )

    # Bytes 12:48 contain 9 float32 coordinates:
    # v0 xyz, v1 xyz, v2 xyz
    coords = records[:, 12:48].copy()

    vertices = coords.view(
        "<f4"
    ).reshape(
        triangle_count,
        9,
    )

    facets = vertices.reshape(
        triangle_count,
        3,
        3,
    ).astype(
        np.float64,
        copy=False,
    )

    return _validate_facets_3d(facets)


def _parse_ascii_stl(content: bytes) -> np.ndarray:
    try:
        text = content.decode(
            "utf-8",
            errors="strict",
        )
    except UnicodeDecodeError as exc:
        raise GeometryParseError(
            "ASCII STL is not valid UTF-8."
        ) from exc

    vertices = []

    vertex_pattern = re.compile(
        r"^\s*vertex\s+"
        r"([-+0-9.eE]+)\s+"
        r"([-+0-9.eE]+)\s+"
        r"([-+0-9.eE]+)"
        r"\s*$",
        re.IGNORECASE,
    )

    for line in text.splitlines():
        match = vertex_pattern.match(line)

        if match:
            try:
                vertices.append(
                    (
                        float(match.group(1)),
                        float(match.group(2)),
                        float(match.group(3)),
                    )
                )
            except ValueError as exc:
                raise GeometryParseError(
                    "ASCII STL contains invalid vertex coordinates."
                ) from exc

    if len(vertices) == 0:
        raise GeometryParseError(
            "No STL vertices were found."
        )

    if len(vertices) % 3 != 0:
        raise GeometryParseError(
            "STL vertex count is not divisible by three."
        )

    facets = np.asarray(
        vertices,
        dtype=np.float64,
    ).reshape(-1, 3, 3)

    return _validate_facets_3d(facets)


def _parse_stl(content: bytes) -> np.ndarray:
    if _looks_like_binary_stl(content):
        return _parse_binary_stl(content)

    return _parse_ascii_stl(content)


# ============================================================================
# OBJ
# ============================================================================

def _parse_obj(content: bytes) -> np.ndarray:
    try:
        text = content.decode(
            "utf-8",
            errors="strict",
        )
    except UnicodeDecodeError as exc:
        raise GeometryParseError(
            "OBJ is not valid UTF-8 text."
        ) from exc

    vertices: List[Tuple[float, float, float]] = []
    faces: List[List[int]] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        fields = line.split()

        if not fields:
            continue

        kind = fields[0]

        if kind == "v":
            if len(fields) < 4:
                raise GeometryParseError(
                    "OBJ vertex must contain X, Y and Z."
                )

            try:
                vertices.append(
                    (
                        float(fields[1]),
                        float(fields[2]),
                        float(fields[3]),
                    )
                )
            except ValueError as exc:
                raise GeometryParseError(
                    "OBJ contains invalid vertex coordinates."
                ) from exc

        elif kind == "f":
            if len(fields) < 4:
                raise GeometryParseError(
                    "OBJ face must contain at least three vertices."
                )

            face = []

            for token in fields[1:]:
                # Supports:
                # v
                # v/vt
                # v//vn
                # v/vt/vn
                raw_index = token.split("/")[0]

                try:
                    index = int(raw_index)
                except ValueError as exc:
                    raise GeometryParseError(
                        f"Invalid OBJ face index '{token}'."
                    ) from exc

                if index == 0:
                    raise GeometryParseError(
                        "OBJ indices are 1-based; zero is invalid."
                    )

                if index < 0:
                    index = len(vertices) + index
                else:
                    index -= 1

                if index < 0 or index >= len(vertices):
                    raise GeometryParseError(
                        "OBJ face references a vertex outside the vertex list."
                    )

                face.append(index)

            faces.append(face)

    if not vertices:
        raise GeometryParseError(
            "OBJ contains no vertices."
        )

    if not faces:
        raise GeometryParseError(
            "OBJ contains no faces."
        )

    vertex_array = np.asarray(
        vertices,
        dtype=np.float64,
    )

    triangles: List[np.ndarray] = []

    # Polygon triangulation using a fan.
    # This is intentionally simple and predictable for CFD surface meshes.
    for face in faces:
        if len(face) < 3:
            raise GeometryParseError(
                "OBJ face has fewer than three vertices."
            )

        if len(face) == 3:
            triangles.append(
                vertex_array[np.asarray(face)]
            )
        else:
            base = face[0]

            for i in range(1, len(face) - 1):
                triangles.append(
                    vertex_array[
                        np.asarray(
                            [
                                base,
                                face[i],
                                face[i + 1],
                            ]
                        )
                    ]
                )

    facets = np.asarray(
        triangles,
        dtype=np.float64,
    )

    return _validate_facets_3d(facets)


# ============================================================================
# Mesh topology / watertight validation
# ============================================================================

def _quantized_vertex_keys(
    facets: np.ndarray,
    tolerance: float = 1e-9,
) -> np.ndarray:
    """
    Quantize vertices so numerically equivalent coordinates can be treated
    as identical when determining mesh topology.
    """

    vertices = facets.reshape(-1, 3)

    if tolerance <= 0.0:
        raise ValueError(
            "Topology tolerance must be positive."
        )

    return np.round(
        vertices / tolerance
    ).astype(np.int64)


def is_watertight(
    facets: np.ndarray,
    tolerance: float = 1e-9,
) -> bool:
    """
    Determine whether every undirected mesh edge occurs exactly twice.

    This is a necessary condition for a closed manifold triangle surface.

    It does not attempt to prove every possible manifold/topological property,
    but it reliably detects open boundaries in ordinary CFD surface meshes.
    """

    facets = _validate_facets_3d(facets)

    keys = _quantized_vertex_keys(
        facets,
        tolerance=tolerance,
    )

    _, inverse = np.unique(
        keys,
        axis=0,
        return_inverse=True,
    )

    vertex_ids = inverse.reshape(
        facets.shape[0],
        3,
    )

    edges = np.stack(
        (
            vertex_ids[:, [0, 1]],
            vertex_ids[:, [1, 2]],
            vertex_ids[:, [2, 0]],
        ),
        axis=1,
    )

    edges = edges.reshape(-1, 2)

    edges.sort(axis=1)

    unique_edges, counts = np.unique(
        edges,
        axis=0,
        return_counts=True,
    )

    del unique_edges

    return bool(
        np.all(counts == 2)
    )


def mesh_bounding_box(
    facets: np.ndarray,
) -> dict:
    facets = _validate_facets_3d(facets)

    xyz = facets.reshape(-1, 3)

    mins = np.min(
        xyz,
        axis=0,
    )

    maxs = np.max(
        xyz,
        axis=0,
    )

    return {
        "xmin": float(mins[0]),
        "xmax": float(maxs[0]),
        "ymin": float(mins[1]),
        "ymax": float(maxs[1]),
        "zmin": float(mins[2]),
        "zmax": float(maxs[2]),
    }


# ============================================================================
# 3D public parser
# ============================================================================

def parse_3d_stl_obj(
    file_content: bytes,
    file_type: str,
) -> np.ndarray:
    """
    Parse STL or OBJ into:

        (F, 3, 3)

    and normalize the resulting geometry to unit maximum bounding-box size.

    Raises InvalidGeometryError when topology or geometry is invalid.
    """

    if not isinstance(file_content, (bytes, bytearray)):
        raise GeometryParseError(
            "file_content must be bytes."
        )

    normalized_type = (
        str(file_type)
        .strip()
        .lower()
        .lstrip(".")
    )

    if normalized_type == "stl":
        facets = _parse_stl(bytes(file_content))

    elif normalized_type == "obj":
        facets = _parse_obj(bytes(file_content))

    else:
        raise GeometryParseError(
            "Unsupported 3D file type. Use STL or OBJ."
        )

    facets = _normalize_3d(facets)

    # Imported 3D geometry is expected to be a closed body for the CFD
    # surface-panel pipeline. Do not silently repair an open surface.
    if not is_watertight(facets):
        raise InvalidGeometryError(
            "Imported 3D surface is not watertight/closed. "
            "Please upload a closed surface mesh."
        )

    return facets