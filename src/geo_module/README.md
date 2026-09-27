# Geometry Engine (`geo_module`)

## Purpose

Module 1 of the CFD SaaS pipeline. It generates or imports 2D airfoil/body curves and 3D wing/body surface meshes, then emits a validated `GeometryOutput` contract for Module 2 (`math_module`).

## Input / output schema

**Public façade:** `GeometryEngine`

| Method | Input | Output |
|---|---|---|
| `generate_naca(code, chord, num_points)` | NACA 4-digit string, chord > 0 | 2D closed polyline |
| `generate_flat_plate(chord, num_points, thickness?)` | Chord > 0 | 2D closed thin polygon |
| `generate_wing(...)` | NACA section + span/sweep/dihedral | 3D triangular skin (open at root/tip) |
| `generate_cylinder(...)` | Radius, length, topology counts | 3D closed cylinder |
| `import_2d(bytes, svg\|dxf)` | File bytes | Normalized 2D polyline |
| `import_3d(bytes, stl\|obj)` | File bytes | Normalized watertight mesh |
| `generate_sample(name)` | Module-sample id or title from `datasets/modules/geo/` | `GeometryOutput` |
| `generate_from_spec(spec)` | Catalog geometry spec (module or pipeline geo stage) | `GeometryOutput` |

**Contract** (`geo_skeleton.py` — `GeometryOutput` + generator/parser signatures; auto-updated after Module 1 validation):

- `mode`: `"2D"` or `"3D"`
- `shape_name`: non-empty string
- `points_2d`: `(N, 2)` float64 or `None`
- `facets_3d`: `(F, 3, 3)` float64 or `None`
- `num_elements`: vertex count (2D) or triangle count (3D)
- `bounding_box`: `xmin/xmax/ymin/ymax/zmin/zmax`
- `is_closed`: 2D first/last coincidence, 3D manifold edge pairing

Runtime validation: `GeometryOutputModel` / `validate_geometry_output()`. Arrays are copied before return (immutability).

## Internal file map

| File | Role |
|---|---|
| `geo_skeleton.py` | TypedDict + Pydantic data contract |
| `geo_inputs.py` | `NACAInput`, `WingInput`, upload limits; sample list from datasets |
| `geo_generators.py` | Vectorized NACA / plate / wing / cylinder |
| `geo_parsers.py` | SVG/DXF/STL/OBJ + watertight test |
| `geometry_module.py` | `GeometryEngine` façade and custom exceptions |
| `__init__.py` | Public API |

## Samples

- **Module samples:** `datasets/modules/geo/catalog.json` (unit shapes + `files/diamond_section.svg`).
- **Pipeline cases:** `datasets/pipeline/catalog.json` (stable real-user ids; geo and math are the live stages).
- Runner: `from shared.pipeline_runner import run_pipeline_case`

## Usage example

```python
from geo_module import GeometryEngine

engine = GeometryEngine()
airfoil = engine.generate_naca("4412", chord=1.0, num_points=200)
wing = engine.generate_wing(
    airfoil_code="0012",
    chord=1.0,
    span=6.0,
    sweep_deg=15.0,
    dihedral_deg=5.0,
    num_sections=24,
)
```

Workbench:

```bash
streamlit run src/streamlit_ui/app_geo.py
```

## Edge-case handling

- Chord/span/radius must be finite and `> 0`.
- NACA codes must be four digits; cambered sections cannot have `p = 0`.
- Closed TE thickness term uses `-0.1036 x^4` so `yt(1) = 0`.
- 2D contours are closed by repeating the first vertex when needed.
- Format-declared closed rings stay closed on import: an SVG `<polygon>`, an SVG path ending in `Z`, and a DXF polyline with group code 70 bit 1 all emit the repeating first vertex, so `is_closed=True` and Module 2 can panel them. An SVG `<polyline>` or an unflagged DXF polyline stays open.
- Wing skins are **not** watertight (root/tip uncapped); cylinders **are**.
- Imported 3D meshes are rejected unless watertight.
- SVG Bézier/arc paths and binary DXF are rejected with typed errors.
- Upload size cap: 50 MB (`validate_file_input`).
- Non-finite coordinates and degenerate triangles raise `InvalidGeometryError` / `GeometryProcessingError` — not raw `IndexError` / `ZeroDivisionError`.

## Exceptions

- `InvalidGeometryInputError` — bad user parameters or files
- `UnsupportedGeometryError` — unknown sample type
- `GeometryProcessingError` — internal contract failure
- `GeometryParseError` / `InvalidGeometryError` — parser/topology
- `GeometryInputError` / `GeometryGenerationError` — lower-level helpers
