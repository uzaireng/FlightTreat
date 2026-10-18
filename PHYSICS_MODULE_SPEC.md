# Module 3 — Physics Engine and Solver Specification

**Status: implemented.** The solver, physics catalog, pipeline `stages.physics`, and workbench flow views follow this contract. Do not rename the fields below.

This document is the build contract for Module 3. The draft types in `src/physics_module/physics_skeleton.py` stay the field-name source of truth. The next implementation turn extends that skeleton only by **adding** fields. It does not rename or remove any name already defined there, in `math_skeleton.py`, or in `aero_skeleton.py`.

Locked solver: viscous-inviscid coupled panel method. 2D vortex panels, 3D constant-strength source/doublet surface panels, and a 1D integral boundary layer. No volume Navier–Stokes mesh.

---

## 1. Non-regression

Module 1 and Module 2 stay intact.

- Geometry importers (SVG, DXF, STL, OBJ), parametric generators (NACA, flat plate, wing, cylinder), catalogs, and workbench sliders stay as they are.
- `MeshEngine` remains the only place that discretizes a contour or a surface. Module 3 copies every incoming array and never writes back into `GeometryOutput` or `MeshOutput`.
- Every live pipeline case and math sample must be solvable by passing its existing `MeshOutput` into `PhysicsEngine`. Guard math samples that are defined to fail in Module 2 still fail there. Module 3 does not catch those errors or relax them.
- `run_pipeline_case` continues to accept `math_overrides`. A future `physics_overrides` argument is additive. It must not change the meaning of `math_overrides`.
- Existing case ids stay the only user journeys. Physics does not invent a second catalog of names.

---

## 2. Data flow

```text
GeometryOutput
    → MeshEngine
        → MeshOutput
            → PhysicsInput
                → PhysicsEngine.solve
                    → PhysicsOutput
                        → AeroInput          (Module 4, later)
                        → Streamlit workbench
```

`PhysicsInput` is exactly the current skeleton:

| Field | Role |
|---|---|
| `mesh` | A `MeshOutput`. Copied on entry. |
| `freestream` | `FreestreamConditions` |
| `parameters` | `PhysicsParameters` (`total=False`; omitted keys take the defaults below) |

Dispatch is `mesh["mode"]`:

| Mode | Required mesh blocks | Solver | Boundary layer | Off-body field |
|---|---|---|---|---|
| `"2D"` | `panel_2d`, `cartesian_2d`, `ibl_1d` | `solver_2d.py` | Marches `ibl_1d` when regime is viscous | Cartesian farfield from `cartesian_2d` |
| `"3D"` | `panel_3d` | `solver_3d.py` | Only if `ibl_1d` is already populated. Open wing skins today have `ibl_1d is None`; the 3D solve is then inviscid and `ibl` is `None` | Surface velocity at centroids. No volume grid |

Missing required blocks raise `InvalidFlowConditionError` with the block name. Module 3 does not rebuild them.

`panel_2d` fields consumed as-is: `nodes`, `collocation`, `tangent`, `normal`, `length`, `angle_beta`, `curvature`, `x_over_c`, `arc_length`, `chord`, `le_index`, `te_index`, `is_closed`, `te_first`.

`ibl_1d` fields consumed as-is: `s_upper`, `s_lower`, `xy_upper`, `xy_lower`, `tangent_upper`, `tangent_lower`, `ds_upper`, `ds_lower`, `stagnation_xy`. Stations are already split at stagnation. Module 3 does not resample arc length.

`panel_3d` fields consumed as-is: `facets`, `centroids`, `normals`, `tangent_1`, `tangent_2`, `areas`, `transforms`, `inverse_transforms`, `is_closed`, `normal_orientation`. Local coordinates stay `p_local = T @ (p - C)` with `T = transforms[f]`.

`cartesian_2d` is an evaluation grid only. It is not a finite-volume mesh and it is not a boundary-condition source.

---

## 3. Contracts

All floating arrays are `np.float64`. Integer masks are `np.int32`. Public entry points return new arrays.

### 3.1 Types that already exist (do not rename)

From `physics_skeleton.py`:

- Exceptions: `PhysicsEngineError`, `SingularInfluenceMatrixError`, `BoundaryLayerSeparationError`, `InvalidFlowConditionError`.
- Literals: `FlowRegime`, `TwoDPanelModel`, `ThreeDPanelModel`, `IBLClosure`.
- `FreestreamConditions`: `alpha_deg`, `reynolds`, `mach`, `v_inf`, `rho`.
- `PhysicsParameters`: `regime`, `panel_model_2d`, `panel_model_3d`, `ibl_closure`, `kutta`, `max_vic_iterations`, `vic_tolerance`, `viscous_coupling`.
- `PhysicsInput`: `mesh`, `freestream`, `parameters`.
- `InviscidSolution`: `gamma`, `mu`, `sigma`, `cp_inviscid`, `v_tangential`, `influence_condition_number`.
- `IBLSolution`: `theta_upper`, `theta_lower`, `delta_star_upper`, `delta_star_lower`, `shape_h_upper`, `shape_h_lower`, `cf_upper`, `cf_lower`, `ue_upper`, `ue_lower`, `transition_s_upper`, `transition_s_lower`, `separated`.
- `PhysicsOutput`: `mode`, `regime`, `freestream`, `inviscid`, `ibl`, `vic_iterations`, `residual`, `converged`.
- `PhysicsEngineProtocol.solve` and `solve_from_mesh`.

`InviscidSolution` array layout stays:

| Mode | `gamma` | `mu` | `sigma` | `cp_inviscid` | `v_tangential` |
|---|---|---|---|---|---|
| 2D vortex | `(P,)` | `None` | `None` | `(P,)` | `(P, 2)` |
| 2D source-vortex | `(P,)` | `None` | `(P,)` | `(P,)` | `(P, 2)` |
| 3D | `None` | `(F,)` | `(F,)` | `(F,)` | `(F, 3)` |

`P` is the panel count from `panel_2d` (`nodes.shape[0] - 1`). `F` is `panel_3d["facets"].shape[0]`.

### 3.2 Defaults

Applied when `parameters` omits the key:

| Key | Default |
|---|---|
| `regime` | `"viscous_inviscid"` in 2D when `ibl_1d` is present; `"inviscid"` in 3D when `ibl_1d` is `None` |
| `panel_model_2d` | `"vortex"` |
| `panel_model_3d` | `"constant_source_doublet"` |
| `ibl_closure` | `"thwaites_head"` |
| `kutta` | `True` for closed 2D sections and open 3D wing skins; `False` for closed 3D bodies |
| `max_vic_iterations` | `8` |
| `vic_tolerance` | `1e-4` (max change of `δ*` divided by chord) |
| `viscous_coupling` | `"transpiration"` |

`drela_two_equation` is a legal literal and must be rejected with `InvalidFlowConditionError("ibl_closure 'drela_two_equation' is not implemented")` until a later spec revision. Do not silently fall back.

### 3.3 Additive parameters (new keys, all optional)

| Key | Type | Default | Meaning |
|---|---|---|---|
| `fail_on_separation` | `bool` | `False` | When `False`, separation is tagged on `IBLSolution.separated` and marching freezes downstream. When `True`, the same event raises `BoundaryLayerSeparationError`. |
| `prandtl_glauert` | `bool` | `True` | Apply the correction in §5.4 when `0 < mach < 0.7`. |
| `wake_length_chords` | `float` | `20.0` | 3D wake sheet length in root-chord units. |
| `dense_panel_limit` | `int` | `2000` | Largest `F` or `P` for a stored dense influence matrix. |

### 3.4 Additive outputs (new keys on the existing TypedDicts)

These are the fields the workbench and Module 4 need that the draft skeleton does not yet carry. Add them on the implementation turn. Do not replace the keys in §3.1.

**`IBLSolution` additions**

| Key | Shape | Meaning |
|---|---|---|
| `delta_upper`, `delta_lower` | `(N_u,)`, `(N_l,)` | Boundary-layer thickness `δ`, estimated from the marched `θ` and `H` by the rule in §5.3. Not an extra differential unknown. |
| `separation_s_upper`, `separation_s_lower` | `float` | Arc length from stagnation of the first separated station. `nan` if that side stays attached. |
| `separated_upper`, `separated_lower` | `(N_u,)`, `(N_l,)` bool | Station mask. `separated` remains the scalar OR of both sides. |
| `stagnation_xy` | `(2,)` | Copy of `ibl_1d["stagnation_xy"]`. |

`theta_*`, `delta_star_*`, `shape_h_*`, `cf_*`, `ue_*` stay the marched quantities. `delta_star = shape_h * theta` within `1e-8` relative tolerance.

**`SurfaceLoads` (new object, attached as `PhysicsOutput["loads"]`)**

Non-dimensional unless noted. Reference dynamic pressure uses the **unit** freestream of the linear system (`|V∞| = 1`). Dimensional `v_inf` and `rho` are echoed so a UI can form `q_inf = 0.5 * rho * v_inf**2` without a second solve.

| Key | Type | Meaning |
|---|---|---|
| `gamma_total` | `float` | `Γ = Σ γ_i S_i` in 2D. `nan` in 3D (doublet solution has no single vortex strength). |
| `cl` | `float` | Lift coefficient. |
| `cd_pressure` | `float` | Pressure drag. |
| `cd_friction` | `float` | Skin-friction drag. `nan` when `ibl` is `None`. |
| `cd` | `float` | `cd_pressure + cd_friction`, or `cd_pressure` when friction is `nan`. |
| `cm_quarter_chord` | `float` | Pitching moment about the quarter-chord (2D) or the root quarter-chord point (3D). |
| `cm_centroid` | `float` | Pitching moment about the area-weighted centroid. |
| `moment_origin_qc` | `(2,)` or `(3,)` | Physical point used for `cm_quarter_chord`. |
| `moment_origin_centroid` | `(2,)` or `(3,)` | Physical point used for `cm_centroid`. |
| `s_ref` | `float` | 2D: chord. 3D: planform area used to non-dimensionalize. |
| `compressibility_applied` | `bool` | `True` when Prandtl–Glauert scaled the system. |

**`FlowField2D` (new object, `PhysicsOutput["field_2d"]`, `None` in 3D)**

Same shape as `cartesian_2d["grid_x"]`, `(n_y, n_x)`, `float64`. Cells whose `boundary_mask` is solid are `nan`.

| Key | Meaning |
|---|---|
| `u`, `v` | Cartesian velocity, freestream magnitude 1. |
| `w` | Zeros in 2D, present so the UI can share a `(u, v, w)` reader. |
| `speed` | `hypot(u, v)`. |
| `cp` | `1 - speed^2` after compressibility correction. |
| `phi` | Velocity potential. |
| `psi` | Streamfunction. |

**`PhysicsOutput` additions**

| Key | Meaning |
|---|---|
| `loads` | `SurfaceLoads` |
| `field_2d` | `FlowField2D` or `None` |
| `surface_velocity` | 3D only, `(F, 3)` freestream-scale velocity at centroids. `None` in 2D (use `inviscid["v_tangential"]`). |
| `warnings` | `list[str]`. Empty on a clean attached solve. Separation, stalled VIC, and skipped 3D boundary layer are warnings, not exceptions, when `fail_on_separation` is false. |

`cp_surface` for Module 4 is `inviscid["cp_inviscid"]`. Module 4 must not recompute it.

### 3.5 Freestream validation

| Field | Rule |
|---|---|
| `alpha_deg` | Finite, and inside `[-90, 90]`. |
| `reynolds` | `> 0` when `regime == "viscous_inviscid"`. Ignored for `"inviscid"`. |
| `mach` | Finite, `>= 0`. `mach >= 1` raises `InvalidFlowConditionError`. `mach >= 0.7` with `prandtl_glauert` raises the same (the correction is outside its range). `mach == 0` is the pure incompressible system. |
| `v_inf` | `> 0`. Does not scale the matrix. |
| `rho` | `> 0`. Does not scale the matrix. |

Angle of attack enters the Neumann data as `α = radians(alpha_deg)` and `V∞ = (cos α, sin α)` in 2D, or `(cos α, 0, sin α)` in 3D (rotation in the x–z plane, span along +y, matching the wing generator).

---

## 4. Sub-modules

All of these files are created on the implementation turn. None of them exist today.

| File | Responsibility |
|---|---|
| `physics_engine.py` | `PhysicsEngine`. Validates input, copies arrays, dispatches 2D or 3D, runs VIC, calls forces and field evaluation, returns `PhysicsOutput`. |
| `solver_2d.py` | Assembles and solves `A · γ = b` (vortex) or the source-vortex system. Returns `InviscidSolution` and the exterior tangential speed sampled on `ibl_1d` stations. |
| `boundary_layer_1d.py` | Thwaites laminar march and Head turbulent march on the stations Module 2 already built. Returns `IBLSolution` and the transpiration distribution. |
| `solver_3d.py` | Constant-strength source/doublet system `A · μ = b`, wake sheet, Kutta condition, local-to-global velocity. |
| `forces_and_moments.py` | Panel-wise pressure and shear integration to `SurfaceLoads`. |
| `field_reconstruction.py` | Block-evaluated `(u, v, φ, ψ, Cp)` on `cartesian_2d`. |

`physics_skeleton.py` gains the additive fields in §3.3–§3.4 and Pydantic checks for shapes and dtypes. It still contains no influence-coefficient formulas.

Internal helpers may exist. Public names above are the ones tests and the workbench import.

### 4.1 Orchestration

```text
copy mesh arrays
validate freestream and parameters
if mode == 2D:
    Ue = 1 on both IBL sides
    repeat up to max_vic_iterations:
        inviscid = solve_2d(panels, alpha, transpiration)
        if regime is inviscid: break
        ibl = march_ibl(stations, Ue from inviscid, reynolds)
        transpiration = d(Ue * delta_star) / ds
        stop if max |Δ delta_star| / chord < vic_tolerance
    field = evaluate_farfield(panels, gamma, cartesian grid)
else:
    inviscid = solve_3d(panels, alpha, wake)
    ibl = march_ibl(...) if ibl_1d is not None else None
    field = None
loads = integrate(inviscid, ibl, mesh)
```

One VIC cycle for `regime == "inviscid"` (`vic_iterations == 1`). `residual` is the max relative `δ*` change on the last cycle, or `0.0` for a pure inviscid solve. `converged` is `True` when that residual is below `vic_tolerance` or the regime is inviscid.

---

## 5. Numerics

Vectorization rule: influence coefficients, surface integrals, boundary-layer updates, and farfield evaluation are NumPy broadcasting plus `scipy.linalg` / `scipy.sparse.linalg`. A Python loop that walks panels, stations, or grid nodes one at a time is not allowed. Batching a matrix-free product over row-blocks is allowed when the batch itself is vectorized.

Do not form `inv(A)`. Solve with `scipy.linalg.solve` (dense LU) or `scipy.sparse.linalg.gmres` (matrix-free). Record `influence_condition_number` from the 1-norm condition estimate of the dense factor when the matrix is stored, and `nan` for GMRES.

### 5.1 Two-dimensional vortex panels (`solver_2d.py`)

Unknown: constant vortex strength `γ_j` on each panel, positive by the right-hand rule of `tangent`.

Exterior Neumann condition at collocation `i`, unit freestream:

```text
Σ_j A_ij γ_j  =  - n_i · V∞  + v_n,i
```

`v_n` is `0` on the first pass and the transpiration velocity after that. `A_ij` is the normal velocity induced at collocation `i` by a unit-strength vortex panel `j`, using the standard constant-strength panel geometric integrals (Katz and Plotkin): relative coordinates of both endpoints in the panel frame, then `atan2` and `log` of the endpoint radii. Self terms use the analytic jump, not a collocated singular sample.

Kutta, when `kutta` is true: replace the equation on one trailing-edge panel with `γ_te_upper + γ_te_lower = 0`. The two trailing-edge panels are the panels that touch the node `te_index`. If `te_index` does not identify exactly two panels, raise `InvalidFlowConditionError`.

Source-vortex (`panel_model_2d == "source_vortex"`) adds a source unknown `σ_j` and a second kinematic condition (exterior tangential velocity equal to the panel-frame value, or a prescribed normal source from thickness). Vortex-only is the required first path. Source-vortex is the same file and the same `InviscidSolution` layout, and it must be implemented before Module 3 is called complete for 2D, because the skeleton already advertises it.

Surface speed at collocation `i` is the exterior tangential velocity (freestream tangent component plus induced tangent plus the panel self-jump `±γ_i / 2`). Then

```text
Cp_i = 1 - |V_t,i|^2
```

before the compressibility step in §5.4.

`Ue` on an IBL station is `|V_t|` interpolated from collocation values onto `xy_upper` / `xy_lower` with a vectorized linear interpolation along `s`. No second panel solve.

### 5.2 Three-dimensional source/doublet panels (`solver_3d.py`)

Unknown: constant doublet `μ_f` per triangle. Source strength for the Neumann formulation:

```text
σ_f = n_f · V∞
```

on a closed body. On an open wing skin, `σ_f = n_f · V∞` still, and a wake doublet sheet carries the trailing-edge jump.

Wake construction is the only auxiliary surface Module 3 may create. It is not written into `MeshOutput`.

1. Classify facet edges with vectorized sorted endpoint keys. A boundary edge appears once.
2. Those boundary edges are the trailing edge for an open skin (`is_closed is False`).
3. Extrude each boundary edge downstream by `wake_length_chords * root_chord` along `V∞` to form wake triangles.
4. Kutta: wake doublet `μ_wake = μ_upper - μ_lower` on the two surface panels that share that trailing edge, which removes the wake unknown.

The interior equation at each collocation point is `φ_μ + φ_σ + V∞·r = 0`, with `σ = n·V∞`. The source potential is the exact triangle integral; the self entry stays the in-plane disk value `-R/2`. Dropping `V∞·r` leaves a right-hand side that shrinks with the panel size, and the trailing-edge jump collapses under chordwise refinement. On an open wing the surface speed is `V∞` plus the chordwise derivative of `μ + V∞·r` on each spanwise strip. Closed bodies still sample the exterior perturbation just off the panel and add `V∞`.

Closed bodies (`is_closed is True`, `body_cylinder_closed` and watertight uploads) use an interior Dirichlet condition at a point under each panel (or the equivalent Morino condition) so the exterior Neumann problem has a unique doublet. `kutta` is false. No wake.

Influence of a unit doublet and a unit source is evaluated in the panel frame given by `transforms`, then the velocity is mapped back with `inverse_transforms`. Self-influence uses the analytic solid-angle and source jump for a constant-strength triangle.

`Cp` and `v_tangential` are evaluated at centroids from the exterior perturbation velocity plus `V∞`.

Panel budget:

| Unknowns | Method | Interactive target |
|---|---|---|
| `P ≤ 400` (2D) | Dense LU | under 100 ms at 160 panels, under 200 ms at 400, excluding the farfield plot |
| `F ≤ dense_panel_limit` (default 2000) | Dense LU | a few seconds |
| `F` above the limit | GMRES, matrix-free matvec, relative tolerance `1e-6`, no stored `A` | still seconds, not a volume solve |

The rectangular-wing cruise mesh is about 2400 panels at density 1. That case is allowed to take the GMRES path. Density 2 and 4 multiply face count by 4 and 16. The workbench must warn when `F` exceeds `dense_panel_limit`, and the solve must still return. Do not refuse a user density setting that Module 2 already accepted.

If GMRES fails to converge, raise `SingularInfluenceMatrixError`. If a dense factor is numerically singular (condition above `1e12`), raise the same exception. Never surface a raw `LinAlgError`.

### 5.3 Integral boundary layer (`boundary_layer_1d.py`)

Closure `thwaites_head`. March upper and lower independently from `s = 0` (stagnation) toward the trailing edge, using `ds_*` from the mesh. Both sides are updated with array operations over stations. The two sides may be stacked and marched together.

Edge velocity `Ue(s)` comes from the latest inviscid solution. Kinematic viscosity is `ν = V_ref * chord / reynolds` with `V_ref = 1` and `chord = panel_2d["chord"]`.

**Laminar — Thwaites**

```text
θ²(s) = 0.45 ν / Ue(s)^6 * ∫_0^s Ue(ξ)^5 dξ
λ = θ² / ν * dUe/ds
```

The integral is a cumulative trapezoid (`scipy.integrate.cumulative_trapezoid` or an equivalent prefix sum). Correlations (Cebeci / Thwaites), evaluated as array expressions of `λ`:

```text
ℓ(λ) = 0.22 + 1.57 λ − 1.8 λ²
H(λ) = 2.61 − 3.75 λ + 5.24 λ²
Cf = 2 ℓ / Re_θ
Re_θ = Ue θ / ν
```

Laminar separation onset: `λ <= -0.09`.

**Transition — Michel**

```text
Re_θ,tr = 1.174 (1 + 22400 / Re_s) Re_s^0.46
Re_s = Ue s / ν
```

The first station with `Re_θ >= Re_θ,tr` is transition. Record it on `transition_s_upper` / `transition_s_lower`. `nan` if that side never transitions. Downstream of transition, switch that side to Head.

**Turbulent — Head entrainment**

```text
H1 = 3.3 + 0.8234 (H − 1.1)^(−1.287)     for H >= 1.1
F(H1) = 0.0306 (H1 − 3)^(−0.6169)
d/ds (Ue θ H1) = Ue F(H1)
```

Integrate that ODE with a vectorized marching step whose width is the mesh `ds` (explicit step is acceptable; the step count is the station count already chosen by Module 2, not an inner Python panel loop). Skin friction, Ludwieg–Tillmann:

```text
Cf = 0.246 * 10^(−0.678 H) * Re_θ^(−0.268)
```

Turbulent separation onset: `H >= 2.8` or `Cf <= 0`.

**Thickness estimate `δ`**

```text
laminar:    δ*/δ = max(3 − λ, 0.2) / 10          →  δ = δ* / (δ*/δ)
turbulent:  δ = (72 / 7) θ                         (1/7-power approximation)
```

`δ* = H θ` is the marched displacement thickness and is what VIC uses. `δ` is diagnostic.

**After separation**

Freeze `θ`, `H`, and `Cf` at the last attached station for every station downstream. Set the separated mask. Set `separation_s_*`. Append a warning string. Raise `BoundaryLayerSeparationError` only when `fail_on_separation` is true.

**Transpiration**

```text
v_n = d(Ue δ*) / ds
```

along each side, then scatter back onto the nearest surface panels with a vectorized bin of station arc length against `panel_2d["arc_length"]`. `viscous_coupling == "displacement"` is accepted as a literal but the first implementation must use transpiration and reject `"displacement"` with `InvalidFlowConditionError` until the displacement-body path is specified. Do not silently alias the two.

### 5.4 Prandtl–Glauert

When `prandtl_glauert` is true and `0 < mach < 0.7`:

```text
β = sqrt(1 − mach²)
```

Stretch the geometry x-coordinates by `β` while building influence coefficients (Göthert), solve the incompressible system in that frame, then scale the reported pressure:

```text
Cp = Cp_incompressible / β
```

`compressibility_applied` is true only on that path. `mach == 0` leaves coordinates and `Cp` unscaled.

### 5.5 Forces and moments (`forces_and_moments.py`)

Pressure coefficient is the value after §5.4.

**2D.** Panel force from pressure (unit chord reference, `|V∞| = 1`):

```text
dF = −Cp * n * length
```

Rotate into lift and drag by `α`. Friction traction on each IBL station is `Cf * tangent * ds`, rotated into the drag direction and summed. `Cd,v` uses both sides. Moment arms are measured from:

- quarter-chord: `x_le + 0.25 * chord`, with `y` at the leading-edge node height
- centroid: arc-length-weighted mean of the collocation points

```text
Γ = Σ γ_i * length_i
Cl from the pressure (and friction) resultant, not from a second formula
```

A check, not a second definition: for attached inviscid flow, `Cl` and `2 Γ / chord` agree within `5e-2` on the NACA 0012 and flat-plate cases. A larger gap fails the unit test. The reported `cl` is always the integrated resultant.

**3D.** Same pressure integral with `areas` and `normals`. Reference area `s_ref` is the projected planform: sum of `|n · e_z| * area` is not the planform, and the wetted-area convention is forbidden for coefficients. An open wing lofted in spanwise stations uses the trapezoidal integral of each station's own x-extent along y. Sweep shifts a whole station together, so it does not enter that integral. A closed body keeps the product of its two largest centroid extents (for the cylinder, the projected cross-section `diameter * length`). An open surface that is not a loft keeps the product of the centroid x-extent and y-extent, which still includes sweep and tip inset; that fallback is not a wing planform. Document the chosen `s_ref` on `loads`.

Friction is included only when an `IBLSolution` exists. Otherwise `cd_friction` is `nan` and `warnings` contains `"3D boundary layer skipped: mesh has no ibl_1d stations"`.

### 5.6 Farfield reconstruction (`field_reconstruction.py`)

Evaluate the same 2D influence kernels at every Cartesian node, in row-blocks if needed so the temporary array stays under roughly 64 MB. Mask solid cells to `nan` using `boundary_mask`. Do not alter the mask.

`phi` and `psi` use the same panel geometric integrals as the velocity kernel (potential and streamfunction of a constant-strength vortex panel). They are diagnostic fields for the workbench, not inputs to the boundary layer.

3D off-body volume sampling is out of scope. `surface_velocity` at the centroids is the 3D vector field.

### 5.7 Speed targets

Measured around `PhysicsEngine.solve` only, warm NumPy, on the stored cruise meshes (density 1):

| Case | Target |
|---|---|
| 2D, 160 panels, inviscid | under 50 ms |
| 2D, 160 panels, viscous-inviscid, 8 cycles worst case | under 400 ms |
| 3D wing, about 2400 panels, inviscid GMRES | under 5 s |
| 3D cylinder, about 1000 panels, inviscid dense or GMRES | under 3 s |

These are engineering targets for the implementation turn, not a reason to drop outputs. If a target is missed, keep the outputs and say so in the module README.

---

## 6. Errors and warnings

| Situation | Result |
|---|---|
| Non-finite or out-of-range freestream | `InvalidFlowConditionError` |
| 2D mesh missing `panel_2d`, `cartesian_2d`, or `ibl_1d` | `InvalidFlowConditionError` |
| 3D mesh missing `panel_3d` | `InvalidFlowConditionError` |
| Unsupported `ibl_closure` or `viscous_coupling` | `InvalidFlowConditionError` |
| Dense matrix condition above `1e12`, or GMRES failure | `SingularInfluenceMatrixError` |
| Separation and `fail_on_separation` | `BoundaryLayerSeparationError` |
| Separation and not `fail_on_separation` | `separated=True`, warning, solve still returns |
| VIC hits `max_vic_iterations` above tolerance | `converged=False`, warning, last cycle returned |
| 3D with `ibl_1d is None` and regime requested viscous | Solve inviscid, `ibl=None`, warning. Do not invent stations. |

Raw `LinAlgError`, `ZeroDivisionError`, and `IndexError` must not leave the module.

---

## 7. Datasets and pipeline (implementation turn, not this turn)

Do not edit `datasets/pipeline/catalog.json` or add `datasets/modules/physics/catalog.json` until implementation starts. When it starts, extend the existing seven case ids. Do not add new journey names.

Each case gains a `stages.physics` object and `"physics"` appended to `active_stages` after `"math"`:

| Case id | `alpha_deg` | `reynolds` | `mach` | `regime` |
|---|---|---|---|---|
| `airfoil_naca_0012_cruise` | `4.0` | `5e5` | `0.0` | `viscous_inviscid` |
| `airfoil_naca_4412_cruise` | `2.0` | `5e5` | `0.0` | `viscous_inviscid` |
| `plate_flat_thin_reference` | `4.0` | `5e5` | `0.0` | `inviscid` |
| `section_svg_import_2d` | `0.0` | `5e5` | `0.0` | `viscous_inviscid` |
| `wing_rect_uav_cruise` | `4.0` | `5e5` | `0.0` | `inviscid` |
| `wing_swept_drone_cruise` | `4.0` | `5e5` | `0.0` | `inviscid` |
| `body_cylinder_closed` | `0.0` | `5e5` | `0.0` | `inviscid` |

Shared freestream defaults for those payloads: `v_inf = 1.0`, `rho = 1.225`. Parameters omitted from the catalog take §3.2.

`run_pipeline_case` then calls `PhysicsEngine.solve_from_mesh` on a copy of the math result. `physics_overrides` merges on top of `stages.physics` the same way `math_overrides` merges onto `stages.math`. The catalog file is not rewritten at runtime.

Module samples under `datasets/modules/physics/` are solver-only fixtures: a known singular panel set, a Kutta edge case, a Thwaites march on a fixed `Ue(s)`, and one attached NACA check. They do not duplicate the seven journeys.

Reference checks to encode as tests, not as catalog truth tables:

- Flat plate, inviscid, `α = 4°`: `Cl` within `0.08` of `2 π α` (radians).
- NACA 0012, inviscid, `α = 0`: `|Cl| < 0.02` and a symmetric `Cp`.
- NACA 0012, inviscid, `α = 4°`: `Cl` within `0.08` of `0.48` (about `2πα` plus a small thickness effect; the pressure integral is the value under test).
- Cylinder, `α = 0`: `|Cl| < 0.05`, and `Cp` near the nose greater than side-wall `Cp`.
- Separated march with `fail_on_separation=False` returns `separated=True` and finite `Cp`.

---

## 8. Workbench

On the implementation turn, extend `src/streamlit_ui/mesh_workbench.py` (and the `app.py` entry that already calls it). Do not remove a source, a mesh slider, a Plotly behavior, or the Contracts tab.

Add a **Flow conditions** block under Mesh settings, shown for every source once the mesh has succeeded:

| Control | Range | Seed |
|---|---|---|
| Angle of attack `α` (deg) | `−12` to `16`, step `0.5` | case payload, else `4.0` (2D and wings) or `0.0` (closed body) |
| Reynolds number | logarithmic slider `1e4` to `1e7` | case payload, else `5e5` |
| Mach | `0` to `0.65`, step `0.05` | case payload, else `0` |
| Regime | inviscid / viscous-inviscid | §3.2 default for that mode |

Changing any of these reruns `PhysicsEngine` on the current `MeshOutput`. Changing a mesh slider still reruns Module 1 (when a spec exists) and Module 2 first, then Module 3. Geometry and mesh figures keep their current keys, height, pan, and scroll-zoom behavior.

New views, in addition to the existing mesh views:

| View | Content |
|---|---|
| `Cp` vs `x/c` | 2D surface pressure. Upper and lower branches distinguished. |
| Boundary layer | `Cf`, `δ*`, `θ`, and `δ` against arc length `s` on upper and lower. Transition and separation stations marked. |
| Farfield vectors | Quiver of `(u, v)` on the Cartesian grid, solid cells omitted. Streamfunction contour optional. |
| 3D pressure | Same mesh as today, colored by `Cp`. A second trace or glyph for surface velocity is optional and must not replace the normal glyphs already available. |

Banners:

- Success line gains `physics <ms>` and, when relevant, `tuned:` names for flow overrides, beside the existing math timing.
- Separation: warning, not a red stack trace, naming `s` on the side that separated.
- `converged is False`: warning with `vic_iterations` and `residual`.
- 3D viscous request without stations: info that the boundary layer was skipped.
- `SingularInfluenceMatrixError` and `InvalidFlowConditionError`: the existing workbench error pattern, message only.

The Contracts tab prints and downloads `PhysicsOutput` next to geometry and mesh through the existing `contract_export` helpers, extended additively.

`app.py` remains the latest-stage entry. After implementation it is Geometry → Mesh → Physics. `app_geo.py` stays Module 1 only.

---

## 9. Handoff to Module 4

Module 4 is still a draft and is not part of the implementation turn.

When it is built, it reads `PhysicsOutput["loads"]`, `inviscid["cp_inviscid"]`, and `MeshOutput` reference geometry. It does not assemble `A`, march the boundary layer, or recompute `Cp`. It is allowed to re-reduce coefficients if the user overrides `s_ref`, `c_ref`, `b_ref`, or `moment_origin` in `AeroParameters`. Default `AeroOutput.coefficients` copy `loads.cl`, `loads.cd`, `loads.cd_pressure`, `loads.cd_friction`, and `loads.cm_quarter_chord`. `aerodynamic_center_x` and `l_over_d` stay Module 4 quantities because they are not required to close the VIC loop.

---

## 10. Acceptance for the implementation turn

The implementation turn is complete when all of the following hold.

1. Modules 1 and 2 tests still pass without edits to their expected panel counts, file parsers, or slider keys.
2. `PhysicsEngine.solve` returns a validated `PhysicsOutput` for all seven pipeline cases and for an uploaded closed STL and an uploaded SVG.
3. Array dtypes and the shapes in §3.1 and §3.4 hold, including `nan` solid cells in `field_2d`.
4. The flat-plate, NACA 0012, and separation checks in §7 pass.
5. A separated 2D case with the default `fail_on_separation=False` returns a payload and a warning.
6. The workbench shows `Cp`, and moving `α` changes `Cl` and the plotted `Cp` without resetting the geometry view to the farfield.
7. No new Python loop over panels, stations, or grid nodes in the solver inner kernels.
8. `physics_skeleton.py` matches the implemented return value, including the additive fields, and still contains no kernel code.

---

## 11. Readiness

Implemented. Module 4 is still a draft.

| Item | State |
|---|---|
| Geometry and mesh engines, importers, sliders, catalogs | Live. This spec does not change them. |
| `PhysicsInput` / `PhysicsOutput` | Live in `physics_skeleton.py`, including the additive loads and field keys. |
| `PipelineRunResult.physics` | Filled by `run_pipeline_case` when `"physics"` is in `active_stages`. |
| Solver, physics catalog, `stages.physics`, workbench flow controls | Live. |
| Module 4 | Unchanged draft. It will consume `loads` and `cp_inviscid`. |
