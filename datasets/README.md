# Datasets

Professional sample library for the CFD SaaS pipeline. There are **two tracks**. Do not mix them.

## 1. Module samples — `datasets/modules/<module>/`

Isolated fixtures for one engine (unit tests + single-module workbench).

| Folder | Status |
|---|---|
| `modules/geo/` | Live catalog + SVG file fixture |
| `modules/math/` | Placeholder until `math_module` lands |
| `modules/physics/` | Placeholder |
| `modules/aero/` | Placeholder |
| `modules/ai/` | Placeholder |

## 2. Pipeline cases — `datasets/pipeline/`

Real-user journeys. The **same case id** is reused as each downstream module is integrated (`geo` → `math` → `physics` → `aero` → `ai`).

Current live stage: **geo only**. `active_stages` will grow; do not create a second set of user-journey names per module.

## Loader

Python API: `shared.dataset_catalog` and `shared.pipeline_runner.run_pipeline_case`.
