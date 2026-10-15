# Physics Engine (`physics_module`)

Viscous-inviscid coupled panel solver. Module 3 copies `MeshOutput` and does not remesh.

**Inviscid core.** 2D constant-strength vortex panels use a Dirichlet streamfunction system plus a Kutta condition. The midpoint Neumann matrix for a constant vortex has a checkerboard null space, so it is not the linear system. On a blunt trailing face the Kutta pair is the aft-most upper and lower surface panels; on a sharp trailing edge that pair is the two panels that meet at `te_index`. 3D bodies use constant source and doublet panels with a Morino interior condition. Open wings add a wake sheet whose doublet is the trailing-edge jump. Surface speed is the exterior potential gradient, sampled just off the panel, because a centroid-to-centroid gradient mixes the nose cap with the side wall.

**Viscous core.** Thwaites laminar, Michel transition, Head turbulent, marched from the geometric leading edge on the stations Module 2 already built. Wall transpiration `d(Ue δ*)/ds` is carried by a source sheet. Putting that blowing into the vortex streamfunction excites the near-null mode and spikes the vortex strength. `drela_two_equation` and `viscous_coupling="displacement"` raise `InvalidFlowConditionError`.

## Input / output

`PhysicsInput` is `mesh` + `freestream` + optional `parameters`. `PhysicsEngine.solve` and `solve_from_mesh` return a validated `PhysicsOutput`: inviscid `Cp`, optional `IBLSolution`, `loads` (`Cl`, pressure and friction drag, moments), `field_2d` on 2D cases, `surface_velocity` on 3D cases, and `warnings`.

## Files

| File | Role |
|---|---|
| `physics_engine.py` | Copy mesh, dispatch 2D/3D, VIC loop, validation |
| `solver_2d.py` | Vortex Dirichlet system, source transpiration, edge speed |
| `influence_2d.py` | Panel-frame vortex and source kernels |
| `solver_3d.py` | Source/doublet system, wake, exterior velocity |
| `boundary_layer_1d.py` | Thwaites / Head march and transpiration |
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

## Edge cases

- Missing `panel_2d`, `cartesian_2d`, or `ibl_1d` on a 2D mesh raises `InvalidFlowConditionError`.
- A 3D mesh with no `ibl_1d` solves inviscid and warns `3D boundary layer skipped: mesh has no ibl_1d stations`.
- Condition above `1e12`, or a GMRES failure, raises `SingularInfluenceMatrixError`.
- Separation with `fail_on_separation=False` (the default) returns the payload, sets `separated`, and appends a warning.
- Panel counts above `dense_panel_limit` (default 2000) use GMRES on the already-built influence matrix.
- 2D Prandtl–Glauert stretches the chord, solves, then divides `Cp` by `β`. 3D applies the same `Cp` correction and does not stretch the surface mesh.
