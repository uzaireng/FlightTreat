# Lightweight Web-Based CFD SaaS

Python pipeline for fast 2D airfoil and 3D wing/body aerodynamic analysis, built as ten independent modules with strict data contracts.

## Integration status

| Module | Status | Contract | Workbench |
|---|---|---|---|
| 1. Geometry (`geo_module`) | Integrated | `GeometryOutput` | `src/streamlit_ui/app_geo.py` |
| 2. Mesh (`math_module`) | Not started | `MeshOutput` | — |
| 3. Physics (`physics_module`) | Not started | `PhysicsOutput` | — |
| 4. Aero (`aero_module`) | Not started | `AeroOutput` | — |
| 5. AI (`ai_module`) | Not started | `AIOutput` | — |
| 6. UI | Geometry workbench only | — | `app.py` = latest stage |
| 7–10. API / DB / Auth / Docker | Not started | — | — |

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

`src/` must be on `PYTHONPATH` (the Streamlit entrypoints add it automatically).

## Run the latest workbench (Module 1)

```bash
streamlit run src/streamlit_ui/app_geo.py
```

Equivalent latest-pipeline alias:

```bash
streamlit run src/streamlit_ui/app.py
```

The workbench generates NACA sections, plates, parametric wings, cylinders, and imports SVG/DXF/STL/OBJ. Plotly shows 2D polylines and 3D triangle skins.

## Samples

Two tracks (see [datasets/README.md](datasets/README.md)):

1. **Module samples** — `datasets/modules/geo/` (and placeholders for math/physics/aero/ai).
2. **Pipeline cases** — `datasets/pipeline/` (same case ids grow as modules are added).

Workbench sidebar: **Pipeline case** vs **Module sample**.

## Layout

```text
datasets/                # Module fixtures + end-to-end pipeline cases
src/geo_module/          # Module 1
src/shared/              # Catalog loader + pipeline runner
src/streamlit_ui/        # Incremental visual test benches
tests/                   # Smoke tests
```

## Module docs

- [Geometry Engine README](src/geo_module/README.md)
- [Datasets README](datasets/README.md)
