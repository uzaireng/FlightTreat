# Lightweight Web-Based CFD SaaS

Python pipeline for fast 2D airfoil and 3D wing/body aerodynamic analysis, built as ten independent modules with strict data contracts.

## Integration status

**Solver (locked):** viscous-inviscid coupled panel method — 3D surface panels + 2D vortex panels + 1D integral boundary layer. Target 85–95%+ \(C_L / C_p / C_D\) accuracy without volume Navier–Stokes. The measured NACA 0012 comparison (conditions, coefficients, and errors) is `src/physics_module/NACA0012_VALIDATION.md`.

| Module | Status | Contract | Workbench |
|---|---|---|---|
| 1. Geometry (`geo_module`) | Integrated & visually validated | `src/geo_module/geo_skeleton.py` | `src/streamlit_ui/app_geo.py` |
| 2. Mesh (`math_module`) | Integrated with geometry (panels, farfield, IBL stations, 3D metrics) | `src/math_module/math_skeleton.py` | `src/streamlit_ui/app_mesh.py` |
| 3. Physics (`physics_module`) | Live (VIC panel + IBL) | `src/physics_module/`, [spec](PHYSICS_MODULE_SPEC.md) | `streamlit run src/streamlit_ui/app.py` |
| 4. Aero (`aero_module`) | Live reduction (coefficients, newtons, \(L/D\), polar fits) | `src/aero_module/` | same workbench as physics |
| 5. AI (`ai_module`) | Not started | `AIOutput` | — |
| 6. UI | Geometry + mesh workbench | — | `app.py` = latest stage |
| 7–10. API / DB / Auth / Docker | Not started | — | — |

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

`src/` must be on `PYTHONPATH` (the Streamlit entrypoints add it automatically).

## Run the latest workbench (Geometry → Aero)

```bash
streamlit run src/streamlit_ui/app.py
```

Same combined stage:

```bash
streamlit run src/streamlit_ui/app_mesh.py
```

Module 1 on its own:

```bash
streamlit run src/streamlit_ui/app_geo.py
```

The combined workbench runs a pipeline case, a math fixture, or a live shape (NACA, plate, wing, cylinder, SVG/DXF/STL/OBJ) through both modules. Plotly shows the contour or skin, clustered panels, the farfield boundary mask, IBL stations, and 3D unit normals. The **Contracts** tab prints and downloads exactly what each stage produced.

## Samples

Two tracks (see [datasets/README.md](datasets/README.md)):

1. **Module samples** — `datasets/modules/geo/`, `datasets/modules/math/`, and `datasets/modules/physics/` (placeholders remain for aero/ai).
2. **Pipeline cases** — `datasets/pipeline/` (same case ids; live stages are geo, math, physics, and aero). Seven cases cover every geometry source: NACA 0012 and 4412 sections, a thin flat plate, an imported SVG section, a rectangular wing, a swept drone wing, and a closed cylinder.

Workbench sidebar **Source**: **Upload my geometry file**, **Build a shape**, **Pipeline case**, or **Math module sample**. The page opens on the NACA 0012 cruise case. Mesh sliders rebuild on every change for all four sources. 2D controls are panel count, LE/TE clustering, farfield `n_x`/`n_y`, and the chord-relative box. 3D controls are spanwise sections and points per section (wings), circumferential segments and axial rings (cylinders), plus surface panel density on every body, including uploads and stored cases. Guard math fixtures stay locked to their stored parameters.

## Inspecting a run

```python
from shared.contract_export import contract_summary_json, contract_npz_bytes
from shared.pipeline_runner import run_pipeline_case

result = run_pipeline_case("airfoil_naca_0012_cruise")
print(contract_summary_json(result["geo"], result["math"], label=result["case_id"]))
```

## Layout

```text
datasets/                # Module fixtures + end-to-end pipeline cases
src/geo_module/          # Module 1 (live)
src/math_module/         # Module 2 mesh engine (live)
src/physics_module/      # Module 3 viscous-inviscid panel solver
src/aero_module/         # Module 4 draft skeleton
src/shared/              # Catalog loader, pipeline runner, contract export
src/streamlit_ui/        # Incremental visual test benches
tests/                   # Smoke tests
```

## Module docs

- [Geometry Engine README](src/geo_module/README.md)
- [Mathematics & Mesh spec](src/math_module/README.md)
- [Physics specification](PHYSICS_MODULE_SPEC.md) (implemented; [module note](src/physics_module/README.md))
- [Aero draft](src/aero_module/README.md)
- [Datasets README](datasets/README.md)
