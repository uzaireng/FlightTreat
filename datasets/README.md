# Datasets

Professional sample library for the CFD SaaS pipeline. There are **two tracks**. Do not mix them.

## 1. Module samples — `datasets/modules/<module>/`

Isolated fixtures for one engine (unit tests + single-module workbench).

| Folder | Status |
|---|---|
| `modules/geo/` | Live catalog + SVG file fixture |
| `modules/math/` | Live catalog: clustering, open wing, closed cylinder, and guard failures |
| `modules/physics/` | Placeholder |
| `modules/aero/` | Placeholder |
| `modules/ai/` | Placeholder |

## 2. Pipeline cases — `datasets/pipeline/`

Real-user journeys. The **same case id** is reused as each downstream module is integrated (`geo` → `math` → `physics` → `aero` → `ai`).

Current live stages: **geo → math**. `active_stages` will grow; do not create a second set of user-journey names per module.

| Case id | Journey | Geometry source |
|---|---|---|
| `airfoil_naca_0012_cruise` | Symmetric 2D section, cosine panels | `naca` |
| `airfoil_naca_4412_cruise` | Cambered 2D section, cosine panels | `naca` |
| `plate_flat_thin_reference` | Thin plate, uniform panels (thin-airfoil reference) | `flat_plate` |
| `section_svg_import_2d` | Customer SVG upload, half-cosine panels | `file` |
| `wing_rect_uav_cruise` | Unswept lifting surface | `wing` |
| `wing_swept_drone_cruise` | Swept wing with dihedral | `wing` |
| `body_cylinder_closed` | Watertight body, outward normals | `cylinder` |

Between them the cases exercise every geometry source Module 1 supports, and each one is panelled by Module 2 in the same run.

## Loader

Python API: `shared.dataset_catalog` and `shared.pipeline_runner.run_pipeline_case`.

## Re-tuning a stored case

`run_pipeline_case` and `run_math_module_sample` accept `math_overrides`, which layers meshing parameters on top of the stored payload for that run only. The catalog file is never rewritten.

```python
run_pipeline_case("airfoil_naca_0012_cruise", math_overrides={"n_panels": 64, "clustering": "uniform"})
```

Guard fixtures (those with `expect_error`) reject overrides, because the expected exception only holds at their stored parameters.

## Inspecting a run

`shared.contract_export` turns the stage outputs of any run into assessable artifacts:

```python
from shared.contract_export import contract_summary_json, contract_npz_bytes
from shared.pipeline_runner import run_pipeline_case

result = run_pipeline_case("airfoil_naca_4412_cruise")
print(contract_summary_json(result["geo"], result["math"], label=result["case_id"]))
open("run.npz", "wb").write(contract_npz_bytes(result["geo"], result["math"]))
```

The workbench exposes the same two artifacts under its **Contracts** tab.
