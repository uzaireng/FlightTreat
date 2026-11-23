# Physics Engine (`physics_module`)

Viscous-inviscid coupled panel solver. Module 3 copies `MeshOutput` and does not remesh.

**Inviscid core.** 2D constant-strength vortex panels use a Dirichlet streamfunction system plus a Kutta condition. The midpoint Neumann matrix for a constant vortex has a checkerboard null space, so it is not the linear system. On a blunt trailing face the Kutta pair is the aft-most upper and lower surface panels; on a sharp trailing edge that pair is the two panels that meet at `te_index`. 3D bodies use constant source and doublet panels with a Morino interior condition. Open wings add a wake sheet whose doublet is the trailing-edge jump. The sheet normal points to the upper side, and both triangles of a spanwise strip carry that strip's jump. The interior Dirichlet condition is φ_μ + φ_σ + V∞·r = 0, with the source integral evaluated exactly on each triangle and the panel self-source kept as the in-plane disk value. Closed-body surface speed is sampled just off the panel. On an open wing the onset is already inside μ, so the tangential perturbation is the chordwise derivative of μ + V∞·r along each spanwise strip. A one-face sample misses the trailing-edge jump, and a difference across a cosine-clustered edge does not.

**Viscous core.** Thwaites laminar, Michel transition, Head turbulent, marched from the geometric leading edge on the stations Module 2 already built. `ibl_closure="drela_two_equation"` keeps that laminar march and replaces the turbulent step with Drela's incompressible equilibrium two-equation shear layer (momentum integral plus kinetic-energy shape parameter, equilibrium shear stress, no lag equation and no simultaneous Newton coupling). A transition/separation tie starts the turbulent step. Laminar separation is only the case where Thwaites trips on an earlier station. The cosine-clustered trailing edge does not supply the slope that decides separation, and its falling panel speed is not used as an edge velocity. Streamwise derivatives are taken on the resolved stations only. Wall transpiration `d(Ue δ*)/ds` enters the vortex system as the contour integral of the outward speed, so the circulation can change. The length-weighted mean of that speed is removed: a closed body has no wake sheet in the panel system, and a raw velocity on the constant-streamfunction right-hand side spikes the vortex strength. The `source_vortex` sheet only corrects the normal residual of that vortex solution. `viscous_coupling="displacement"` raises `InvalidFlowConditionError`.

**Wake.** After the wall march, the two sides are joined so the displacement flux and the momentum defect are continuous. A stagnant trailing-edge gap adds to the displacement flux only. The join uses the last resolved edge speed, not the speed inside the cosine cluster. The wake marches one chord on the panel-field speed with zero skin friction. `IBLSolution.cd_momentum` is the Squire–Young value at the end of that march. The published drag stays the surface pressure integral plus skin friction.

## Input / output

`PhysicsInput` is `mesh` + `freestream` + optional `parameters`. `PhysicsEngine.solve` and `solve_from_mesh` return a validated `PhysicsOutput`: inviscid `Cp`, optional `IBLSolution`, `loads` (`Cl`, pressure and friction drag, moments), `field_2d` on 2D cases, `surface_velocity` on 3D cases, and `warnings`.

## Files

| File | Role |
|---|---|
| `physics_engine.py` | Copy mesh, dispatch 2D/3D, VIC loop, validation |
| `solver_2d.py` | Vortex Dirichlet system, transpiration integral, edge speed |
| `influence_2d.py` | Panel-frame vortex and source kernels |
| `solver_3d.py` | Source/doublet system, wake, exterior velocity |
| `boundary_layer_1d.py` | Thwaites / Head or equilibrium turbulent march, and transpiration |
| `drela_equilibrium.py` | Incompressible equilibrium two-equation closures and the direct step |
| `wake_1d.py` | Trailing-edge join, wake march, Squire–Young defect |
| `forces_and_moments.py` | Pressure and skin-friction loads |
| `field_reconstruction.py` | Cartesian farfield; solid cells are `nan` |
| `physics_skeleton.py` | Contract types and validators only |

## Usage

```python
from physics_module import PhysicsEngine

out = PhysicsEngine().solve_from_mesh(
    mesh,
    {"alpha_deg": 4.0, "reynolds": 5.0e5, "mach": 0.0, "v_inf": 1.0, "rho": 1.225},
    {"regime": "inviscid"},
)
```

Pipeline cases call the same engine through `run_pipeline_case`. `physics_overrides` merges onto `stages.physics` and does not rewrite the catalog.

## Published NACA 0012 check

`NACA0012_VALIDATION.md` compares this solver with Ladson lift (NASA TM 4074, Re = 6×10⁶, M = 0.15), Gregory and O'Reilly upper-surface pressure (R&M 3726, Re = 2.88×10⁶), and the untripped Abbott drag polar at Re = 6×10⁶. The test is `tests/test_naca0012_accuracy.py`. Tripped Ladson drag is not used: this closure transitions on its own.

## Edge cases

- Missing `panel_2d`, `cartesian_2d`, or `ibl_1d` on a 2D mesh raises `InvalidFlowConditionError`.
- A 3D mesh with no `ibl_1d` solves inviscid and warns `3D boundary layer skipped: mesh has no ibl_1d stations`.
- Condition above `1e12`, or a GMRES failure, raises `SingularInfluenceMatrixError`.
- Separation with `fail_on_separation=False` (the default) returns the payload, sets `separated`, and appends a warning.
- Panel counts above `dense_panel_limit` (default 2000) use GMRES on the already-built influence matrix.
- Open lofted wings take `s_ref` from each spanwise station's own chord integrated along y. Sweep does not change that area. Closed bodies keep the product of the two largest centroid extents. An open surface that is not a loft keeps the centroid x-extent times y-extent.
- 2D Prandtl–Glauert stretches the chord, solves, then divides `Cp` by `β`. 3D applies the same `Cp` correction and does not stretch the surface mesh.
