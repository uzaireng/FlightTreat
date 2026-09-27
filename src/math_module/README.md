# Mathematics & Mesh Engine (`math_module`)

## Status

**Integrated with Module 1.** `MeshEngine` takes a `GeometryOutput` and returns a `MeshOutput`. Pipeline cases run geometry then mesh. The combined workbench is `src/streamlit_ui/app_mesh.py` (`app.py` is the same stage). Module 1 alone remains `app_geo.py`.

## Purpose

Convert `GeometryOutput` into solver-ready panel metrics so Module 3 can:

1. Assemble influence coefficient matrices \(A \cdot \mu = b\) (2D vortex panels / 3D surface panels).
2. March a **1D integral boundary layer** on pre-built arc-length stations.
3. Visualize a structured Cartesian farfield grid (not a volume RANS mesh).

Target: wind-tunnel-comparable \(C_L\), \(C_p\), and viscous+pressure \(C_D\) (85–95%+) at millisecond-to-second runtimes **without** 3D volume grids.

## Solver chain (locked)

```text
GeometryOutput
    → MeshEngine (this module)
        → MeshOutput {panel_2d | panel_3d, cartesian_2d, ibl_1d}
            → PhysicsEngine  (VIC: inviscid panel + IBL coupling)
                → AeroEngine (CL, CD_p, CD_f, CM, Cp)
```

## A. Discretization requirements

### 2D domain (boundary-fitted vortex panels + Cartesian farfield)

| Item | Contract |
|---|---|
| Input | `points_2d` shape `(N, 2)`, `np.float64` |
| Clustering | `cosine` (default), `half_cosine`, or `uniform`. Extra LE/TE weights `le_cluster_factor`, `te_cluster_factor` refine \(x/c \to 0\) and \(x/c \to 1\) |
| Nodes | Closed contour `(P+1, 2)` with first = last |
| Collocation | Panel midpoints `(P, 2)` |
| \(\beta_i\) | `angle_beta` `(P,)` = `atan2(ty, tx)` |
| \(S_i\) | `length` `(P,)` strictly `> min_panel_length` |
| Curvature | Discrete \(\kappa_i\) `(P,)`, vectorized |
| Tangents / normals | Unit `(P, 2)`; normals **outward** from solid |
| Cartesian grid | `grid_x`, `grid_y`, `boundary_mask` shape `(n_y, n_x)` |
| Mask tags | `INLET`, `OUTLET`, `TOP_FARFIELD`, `BOTTOM_FARFIELD`, `AIRFOIL_WALL`, `INTERNAL_SOLID` (`BoundaryRegion` int32) |
| IBL mesh | Upper/lower stations from LE/stagnation: `s`, `xy`, `tangent`, `ds` |

Default farfield box is chord-relative. `MeshEngine` places it as \(x = x_{LE} + [-2, 3]\,c\) and \(y = y_{mid} + [-2.5, 2.5]\,c\), with `n_x=161`, `n_y=81`. That is two chords upstream of the leading edge and two chords downstream of the trailing edge. `half_cosine` clusters at the leading edge only; `cosine` clusters at both ends. `le_cluster_factor` and `te_cluster_factor` add extra packing at those ends on both surfaces (0 disables the extra packing; the default 1.0 is a mild extra).

### 3D domain (triangular surface panels)

| Item | Contract |
|---|---|
| Input | `facets_3d` `(F, 3, 3)` |
| Centroids | \(C_f = \frac{1}{3}\sum v_i\), shape `(F, 3)` |
| Normals | \(\hat n = (v_1 \times v_2) / \|v_1 \times v_2\|\), `(F, 3)` |
| Tangents | \(\hat t_1\) along first edge; \(\hat t_2 = \hat n \times \hat t_1\) |
| Area | \(S_f = \tfrac12 \|v_1 \times v_2\|\), `(F,)` |
| Transforms | `(F, 3, 3)` with rows \([\hat t_1, \hat t_2, \hat n]\); \(p' = T(p - C)\) |
| Open wings | Allowed (`is_closed=False`). `require_closed_3d=True` raises `OpenSurfaceError` |
| Degenerate \(S_f\) | `DegeneratePanelError` |
| Non-orthonormal basis | `MeshSingularityError` |

### Performance & validation

- Panel metrics, clustering, curvature, and local 3D bases are NumPy array operations.
- 3D winding repair builds a sparse dual graph, takes one SciPy breadth-first tree, and folds orientation with vectorized path-XOR. It does not walk faces in Python.
- The Cartesian mask ray-casts and distance-tests only cells inside the geometry bounding box, in NumPy blocks, so a fine farfield does not allocate a full grid-by-edge temporary.
- Validate via `validate_mesh_input` / `validate_mesh_output`.
- `GeometryOutput` arrays are copied before discretization.

## B. Data contracts

See `src/math_module/math_skeleton.py`.

- **`MeshInput`**: `{geometry: GeometryOutput, parameters: MeshParameters}`
- **`MeshOutput`**: mode, shape name, `panel_2d` / `cartesian_2d` / `ibl_1d` **or** `panel_3d`, plus `MeshQualityMetrics`
- Exceptions: `InvalidMeshConfigError`, `DegeneratePanelError`, `MeshSingularityError`, `InvertedNormalError`, `OpenSurfaceError`

## C. Downstream (Module 3) readiness

Module 3 **must not** recompute normals, areas, or clustering. It uses:

- 2D: collocation, \(\hat t\), \(\hat n\), \(S_i\), \(\beta_i\), LE/TE indices, IBL `s` stations → vortex influence + Kutta + Thwaites/Head (or Drela) IBL
- 3D: centroids, \(\hat n\), \(\hat t_{1,2}\), \(S_f\), \(T_f\) → source/doublet influence + wake/Kutta strip

Draft contracts: `src/physics_module/physics_skeleton.py`, `src/aero_module/aero_skeleton.py`.

## Public façade

```python
from math_module import MeshEngine
from geo_module import GeometryEngine

geo = GeometryEngine().generate_naca("0012", num_points=160)
mesh = MeshEngine().from_geometry(geo, {"clustering": "cosine", "n_panels": 180})
```

`src/` must be on `PYTHONPATH`. Types and validators remain in `math_module.math_skeleton`.

## Workbench

```bash
streamlit run src/streamlit_ui/app_mesh.py
```

`src/streamlit_ui/app.py` launches the same combined stage. The page shows Module 1 geometry and Module 2 discretization together:

- Clustered 2D panels against the original contour, colored by panel length, with outward normals
- Cartesian farfield overlay and `BoundaryRegion` counts
- Upper/lower 1D IBL stations and the leading-edge stagnation point
- 3D surface with unit-normal arrows
- A **Contracts** tab that prints the full geo + mesh contract and downloads it as JSON or `.npz`

### Sidebar inputs

The **Source** radio picks where geometry comes from.

| Source | What you supply | Mesh sliders |
|---|---|---|
| Upload my geometry file | Your own SVG/DXF section or STL/OBJ body. It meshes as soon as the file is chosen | Shown after a file is loaded, and every move rebuilds the mesh |
| Build a shape | NACA code, flat plate, wing planform, cylinder, or a geo module sample. Shape controls are live | Always shown, and every move rebuilds the mesh |
| Pipeline case | One of the seven stored geo → math journeys. The workbench opens on NACA 0012 cruise | Always shown, seeded from `stages.math`, and every move overrides that payload |
| Math module sample | One mesh fixture | Shown for tunable fixtures, seeded from the sample, including 3D section and density overrides. Guard fixtures cannot be tuned |

The sliders cover clustering, panel count, LE/TE cluster factors, farfield `n_x`/`n_y`, the chord-relative box, 3D normal orientation, `require_closed_3d`, and `surface_density` (1, 2, or 4). Wings also expose spanwise sections and points per section; cylinders expose segments and rings. Those geometry knobs rerun `GeometryEngine` before `MeshEngine`. Only the knobs that apply to the selected geometry are rendered, for uploads, built shapes, pipeline cases, and tunable math samples. Moving a slider layers the new values on the catalog payload for that run only — the catalog file is never rewritten, and leaving the sliders on their seeded values reproduces the stored mesh exactly. `surface_density` 1 keeps the generated triangles; 2 and 4 split each triangle so edge lengths halve.

Guard fixtures (`domain_too_small`, `degenerate_triangle`, `wing_require_closed`) refuse overrides: a guard only means something at the parameters that trigger it, so `run_math_module_sample` raises rather than quietly running a different case.

## Datasets

1. `datasets/modules/math/catalog.json` — cosine / half-cosine / uniform fixtures, an imported SVG section, open wing, outward and inward cylinder, plus guard samples for a bad farfield, a zero-area triangle, and `require_closed_3d` on an open skin.
2. Every case in `datasets/pipeline/catalog.json` has `stages.math` and `"math"` in `active_stages`. The seven cases together cover every Module 1 geometry source (`naca`, `flat_plate`, `wing`, `cylinder`, `file`) end to end.
3. `shared.pipeline_runner.run_pipeline_case` runs geo → math on a copy of the geometry arrays. `run_math_module_sample` runs one math fixture.

## Inspecting the output

`shared.contract_export` renders a finished run as a JSON summary (shapes, ranges, quality, boundary-cell counts) and an `.npz` of every raw array. The workbench serves both from its **Contracts** tab.

```python
from shared.contract_export import contract_summary, contract_npz_bytes

summary = contract_summary(geo, mesh, label="airfoil_naca_0012_cruise")
payload = contract_npz_bytes(geo, mesh)   # geo_points_2d, math_panel_2d_normal, ...
```

## Edge cases the implementation must handle

- Open 2D polylines: `InvalidMeshConfigError`. Vortex panels require a closed contour.
- Flat plate (tiny thickness): clustering must not collapse LE/TE to zero length.
- Zero-area wing triangles: already rejected by geo; mesh must re-check.
- Inward-wound imported STL: flip to `normal_orientation="outward"` or raise `InvertedNormalError`.
- Domain smaller than the airfoil AABB: `InvalidMeshConfigError`.
