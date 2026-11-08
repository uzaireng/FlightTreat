# CFD SaaS Audit Report

**Project:** Lightweight web-based panel-method CFD platform (`F:\f\MVP`)
**Audit date:** 23 October 2026
**Scope:** Full tree that exists today — geometry, mesh, viscous-inviscid panel solver, aerodynamic reduction, Streamlit workbench, datasets, and tests. Modules 5 and 7–10 are absent and are scored as such.
**Method:** Contract and source review (`PHYSICS_MODULE_SPEC.md`, skeletons, engines, workbench, catalogs, tests), plus one live 2D probe of `PhysicsEngine` / `AeroEngine` on generated meshes. No 3D solve was timed in this audit; 3D statements use the solver source and the acceptance tests.

This report judges the tree against a commercial launch of the product described in the project rules: a viscous-inviscid coupled panel method (2D vortex panels, 3D constant source/doublet panels, 1D integral boundary layer), aimed at attached-flow \(C_L\), \(C_p\), and total \(C_D\) in the 85–95% band versus Navier–Stokes or a wind tunnel, at millisecond-to-second runtimes, sold as a web SaaS.

---

## 1. Project Destination & Vision

### 1.1 Long-term destination

The destination is a ten-module web platform that lets a user go from a section or a wing skin to forces, pressure, and a short engineering report without building a volume mesh and without waiting on a RANS job.

The locked flow model is an **advanced viscous-inviscid panel method**:

- 2D constant-strength vortex panels, with a source sheet when thickness blowing is active.
- 3D constant-strength source and doublet triangles, with a wake sheet on an open wing and a Morino interior condition on a closed body.
- A 1D integral boundary layer (Thwaites laminar, Michel transition, Head turbulent) coupled back to the inviscid field by wall transpiration.
- Compressibility through a Prandtl–Glauert correction for \(0 < M < 0.7\).
- An analytics layer that rescales coefficients, reports newtons, \(L/D\), a polar, and an aerodynamic center.
- A later AI layer for shape notes and surrogate estimates. That layer has no code.

Volume Navier–Stokes is intentionally out of the default path. Cartesian grids in Module 2 are farfield pictures and boundary tags, not a finite-volume mesh.

The pipeline contract is strict and already real:

```text
GeometryOutput → MeshOutput → PhysicsOutput → AeroOutput → (AIOutput, not built)
```

Each stage copies arrays and is supposed to refuse bad input with a typed exception rather than a raw `IndexError` or `LinAlgError`.

### 1.2 Who it is for

| Audience | Job the product has to do | What they will compare it with |
|---|---|---|
| University design teams | Section and wing polars for a course project or a student UAV, overnight, on a laptop browser | XFOIL, XFLR5, textbook plots |
| Drone and light-aircraft designers | Attached-flow \(C_L\), \(C_D\), \(L/D\), and \(C_m\) for a planform they can loft themselves | VSPAERO, AVL, a consultant RANS run |
| Automotive and HVAC aero groups | External shape pressure and a drag split on a closed or open surface | Panel codes and a coarse RANS check |
| Hobbyists who will pay | A picture of \(C_p\), streamlines, and a number they can quote | Free panel tools |
| Professional subscribers at $45–$60/mo | A result they can put in a design note, with a stated error band, an export, and a project they can reopen | The free tools above, plus the cost of their own time |

The professional row is the one that sets the price. Students and hobbyists already have XFOIL and XFLR5 at no charge. A subscription is justified only when the web product is faster to set up than those tools and honest about where the answer stops being physical.

### 1.3 Core value proposition

The proposition, as the architecture is written, is:

1. **Speed.** A 2D cruise point in well under a second, a few-thousand-panel wing in a few seconds, on CPU, with no volume grid.
2. **Enough physics for early design.** Lift and \(C_p\) from a singularity method; skin friction and a displacement effect from an integral boundary layer; induced drag either integrated on the surface or estimated by lifting line.
3. **A contract a product can cache.** `TypedDict` plus Pydantic validators, NPZ/JSON export of every stage, stable pipeline case ids.
4. **Interactive feedback.** Plotly views of the section, the panels, \(C_p\), the boundary layer, and a 2D farfield quiver.

That proposition is the right product shape for the price band. The tree implements the skeleton of it. It does not yet implement the accuracy, the 3D viscous half, or the commercial shell that the price assumes. Sections 3, 6, and 10 quantify that gap.

---

## 2. Market-Ready Architecture, Inputs, Methods & Equations

### 2.1 What a market-ready run must accept

The solver is linear and incompressible in its matrix, then optionally corrected for subsonic compressibility. Dimensional \(V_\infty\) and \(\rho\) do **not** scale \(A\). They exist so the UI can form \(q_\infty = \tfrac12 \rho V_\infty^2\).

**Geometry (market-ready set)**

| Input | In the tree today | Still required for the audiences in §1 |
|---|---|---|
| NACA 4-digit section, chord, point count | Live. `generate_naca_4digit` | Keep |
| Thin flat plate | Live | Keep as a verification shape, not a customer shape |
| Rectangular wing: one section, span, LE sweep, dihedral | Live. Constant chord. Root and tip are open | Taper, twist, airfoil change along span, control-surface breaks |
| Closed cylinder | Live | Keep as a non-lifting benchmark |
| SVG / DXF section | Live, with stated rejections (no Bézier arcs, no binary DXF) | Customer DXF that contains splines |
| STL / OBJ, watertight only | Live. 50 MB cap | Open STL repair, STEP/IGES, multi-body assemblies |
| Units, scale, moment reference the user picks | Chord-normalized geometry. Aero can override \(S\), \(c\), \(b\), moment origin | A unit system and a named reference the report prints |

**Flow (market-ready set)**

| Input | Contract today | Gap |
|---|---|---|
| Angle of attack \(\alpha \in [-90^\circ, 90^\circ]\) | Enforced. Workbench slider is \(-12^\circ\) to \(16^\circ\) | No sideslip \(\beta\). 3D onset is \((\cos\alpha,\ 0,\ \sin\alpha)\) only |
| Reynolds number | Required and \(> 0\) when the regime is viscous. Workbench \(10^4\)–\(10^7\) | No freestream turbulence intensity. Michel transition has no \(Tu\) knob |
| Mach | \(M \ge 1\) is rejected. \(M \ge 0.7\) with Prandtl–Glauert on is rejected | No Karman–Tsien, no local supersonic pocket, no shock |
| \(V_\infty > 0\), \(\rho > 0\) | Echoed for dynamic pressure | No temperature, viscosity law, or altitude. \(\nu = c / \mathrm{Re}\) with the matrix freestream equal to 1 |
| Regime | `inviscid` or `viscous_inviscid` | 3D viscous is requested and then dropped (see §6) |
| Panel model | 2D `vortex` or `source_vortex`. 3D `constant_source_doublet` or `constant_doublet` | `drela_two_equation` and `viscous_coupling="displacement"` raise `InvalidFlowConditionError` on purpose |

There is no turbulence model beyond Head’s entrainment correlation, no wall roughness, no transition trip the user can place, and no multi-element gap or flap schedule.

### 2.2 Methods that are actually in the code

#### 2.2.1 Two-dimensional inviscid core

The written spec (`PHYSICS_MODULE_SPEC.md` §5.1) asks for a Neumann system

\[
\sum_j A_{ij}\gamma_j = -\mathbf{n}_i\cdot\mathbf{V}_\infty + v_{n,i}
\]

with a Kutta row \(\gamma_{TE,upper} + \gamma_{TE,lower} = 0\).

The implementation does **not** solve that system. `solver_2d.py` documents why: a midpoint Neumann matrix for a constant vortex has a checkerboard null space. The code solves a Dirichlet streamfunction problem instead, which is the formulation that recovers a lift near \(2\pi\alpha\):

\[
\sum_j \psi_{ij}\gamma_j \;-\; \psi_0 \;=\; -\,\psi_\infty(\mathbf{x}_i)
\]

\[
\psi_\infty = y\cos\alpha - x\sin\alpha
\]

Unknowns are the \(N\) vortex strengths plus one streamfunction constant. The last equation is Kutta when `kutta` is true, or a circulation constraint weighted by panel length when it is false. The trailing-edge pair is the two panels that meet at `te_index` on a sharp edge, and the aft-most upper and lower panels whose normal \(y\)-component exceeds 0.7 on a blunt face.

Surface speed is the exterior tangential velocity, including the vortex self-jump \(\pm \gamma/2\):

\[
C_{p,i} = 1 - |V_{t,i}|^2
\]

A source sheet is solved **after** the vortex system, not inside the same matrix, when the model is `source_vortex` or when transpiration is nonzero:

\[
A_\sigma \boldsymbol{\sigma} = -\mathbf{n}\cdot\mathbf{V}_\infty + \mathbf{v}_n - A_{\sigma\leftarrow\gamma}\boldsymbol{\gamma}
\]

The source self-term is \(+1/2\). If that matrix is judged singular, the code catches `SingularInfluenceMatrixError` and replaces the solve with `numpy.linalg.lstsq` at `rcond=1e-10`. That fallback can return a finite source vector from a rank-deficient matrix.

#### 2.2.2 Viscous-inviscid coupling (2D only)

Module 2 builds upper and lower stations by splitting the contour at the **geometric leading-edge node**, and stores that node as `stagnation_xy`. It does not move the split when \(\alpha\) changes.

The march is Thwaites, then Michel, then Head.

Thwaites:

\[
\theta^2(s) = \frac{0.45\,\nu}{U_e^6}\int_0^s U_e^5\,d\xi, \qquad
\lambda = \frac{\theta^2}{\nu}\frac{dU_e}{ds}
\]

\[
\ell(\lambda) = 0.22 + 1.57\lambda - 1.8\lambda^2, \quad
H(\lambda) = 2.61 - 3.75\lambda + 5.24\lambda^2, \quad
C_f = \frac{2\ell}{\mathrm{Re}_\theta}
\]

Laminar separation is marked at \(\lambda \le -0.09\). A hit on the single station immediately upstream of Michel transition is treated as transition if \(\lambda\) is only a shallow graze past \(-0.09\). A separation retained from the previous coupling pass is kept until \(\lambda\) recovers past \(-0.08\). Those two thresholds (`_ADJACENT_TIE_DEPTH`, `_LAMINAR_RELEASE_LAMBDA`) are numerical hysteresis, not part of Thwaites’ correlation.

Michel:

\[
\mathrm{Re}_{\theta,tr} = 1.174\left(1 + \frac{22400}{\mathrm{Re}_s}\right)\mathrm{Re}_s^{0.46}
\]

Head, explicit in arc length, with Ludwieg–Tillmann skin friction and turbulent separation at \(H \ge 2.8\) or \(C_f \le 0\). The step is a Python loop over stations (`boundary_layer_1d.py`, `_head_from`). The spec allows an explicit march whose width is the mesh spacing. Cosine-clustered trailing-edge gaps shorter than a quarter of the median spacing do not supply \(dU_e/ds\); the slope is frozen at the last resolved station.

Transpiration:

\[
v_n = \frac{d}{ds}(U_e\delta^*)
\]

scattered onto the nearest panel collocation with a `cKDTree`, then **clipped to \([-0.05,\ 0.05]\)**. The coupling loop under-relaxes that vector,

\[
\mathbf{v}_n \leftarrow 0.25\,\mathbf{v}_n^{\mathrm{new}} + 0.75\,\mathbf{v}_n^{\mathrm{old}}
\]

for at most 8 iterations, stopping when \(\max|\Delta\delta^*|/c < 10^{-4}\). If a pass produces \(\max|\gamma| > 8\) or \(\min C_p < -20\), the engine throws the blowing away, re-solves the inviscid field, sets `converged = False`, and warns that the pressure was reset.

Edge speed is not the raw panel speed. `_march_speed` finds the minimum of \(|V_t|\) in the first 30% of each side and overwrites every station upstream of that minimum with the minimum. The comment states the reason: at angle of attack the true stagnation sits on the lower surface, and a falling speed at the geometric leading edge makes Thwaites separate immediately.

Kinematic viscosity uses the unit freestream of the matrix: \(\nu = c/\mathrm{Re}\).

#### 2.2.3 Three-dimensional inviscid core

Closed bodies (`solver_3d.py`):

\[
\sigma_f = \mathbf{n}_f\cdot\mathbf{V}_\infty
\]

Interior Dirichlet, collocation shifted inside by \(10^{-4}\sqrt{S_f}\,\mathbf{n}\):

\[
\phi_\mu + \phi_\sigma + \mathbf{V}_\infty\cdot\mathbf{r} = 0
\]

The doublet influence is the solid angle. The source potential is the exact triangle edge integral. The in-plane source self-term is the disk value \(-R/2\), with \(R = \sqrt{S/\pi}\). The interior doublet self-term is set to \(+1/2\) after the body pass.

Open wings: the same source. A wake is shed from aft centroids, not from a classified trailing-edge polyline. In each spanwise bin the code picks the aft-most upper and lower panels (`normals[:, 2] >= 0` versus the rest) and extrudes two triangles downstream along \(\mathbf{V}_\infty\) for `wake_length_chords * root_chord` (default 20 chords). Wake doublet is \(\mu_\mathrm{upper} - \mu_\mathrm{lower}\) and is eliminated as an unknown. The sheet normal is wound toward \(+z\).

Surface speed on a **closed** body is a one-sided difference of potential, sampled a fraction of the panel scale off the wall, with the panel’s own source column dropped. Surface speed on an **open** wing is not that sample. The code fits a polynomial of degree at most 4 to \(\mu + \mathbf{V}_\infty\cdot\mathbf{r}\) along each spanwise strip and adds \(\partial/\partial x\) of that fit to the freestream \(x\)-component. Strips with fewer than 6 panels on a side get slope 0. Spanwise derivatives are not computed. Sweep is therefore carried only insofar as it changed \(\mu\).

\[
C_p = 1 - |\mathbf{V}|^2
\]

Above `dense_panel_limit` (default 2000) the already-built dense matrix is solved with GMRES (`rtol=1e-6`, restart 40, max 80). The spec asks for a matrix-free product and no stored \(A\). The code always stores \(A\).

#### 2.2.4 Compressibility

\[
\beta = \sqrt{1 - M^2}, \qquad 0 < M < 0.7
\]

In 2D the panel \(x\)-coordinates are stretched by \(\beta\) (Göthert), the incompressible system is solved, and reported \(C_p\) is divided by \(\beta\). The edge speed passed to the boundary layer is then rebuilt from that scaled \(C_p\) and clamped so \(1 - C_p\) cannot go negative.

In 3D there is no stretch. `PhysicsEngine._solve_3d` divides the finished \(C_p\) by \(\beta\) only. The module README states this difference explicitly.

#### 2.2.5 Forces

2D pressure force on a panel, unit freestream:

\[
d\mathbf{F} = -C_p\,\mathbf{n}\,S
\]

Lift and drag are the wind-axis projections. Circulation \(\Gamma = \sum \gamma_i S_i\) is reported as a check. The published \(C_L\) is the integral, not \(2\Gamma/c\). The tests require those two to agree within 0.05 on the plate and on NACA 0012.

Skin friction uses local \(C_f\) (shear over **edge** dynamic pressure) multiplied by \((U_e/V_\infty)^2\) before the segment average, so \(C_D\) is referred to freestream \(q\). The pitching moment integrates pressure about the quarter chord and about the arc-length-weighted collocation centroid. The moment function’s comment says friction moment is taken about station midpoints. The return value is the pressure moment only. Viscous \(C_m\) does not include skin friction.

3D reference area for a lofted wing is the trapezoidal integral of each station’s own chord along \(y\). Sweep does not inflate it. A closed body uses the product of the two largest centroid extents. An open surface that is not a loft uses the centroid bounding box in \(x\) and \(y\).

#### 2.2.6 What Module 4 adds, and what it refuses to invent

`AeroEngine` does not re-solve. It rescales Module 3’s coefficients onto a caller-supplied \(S\) and \(c\), forms newtons at the stored \(q_\infty\), copies \(C_p\), bins spanwise \(c_\ell(y)\) on a 3D wing, and, when the caller supplies several already-solved angles, fits

\[
C_L = C_{L0} + c_{\ell_\alpha}\alpha, \qquad
\frac{x_{ac}}{c} = 0.25 - \frac{c_{m_\alpha}}{c_{\ell_\alpha}}
\]

Induced drag is 0 in 2D. In 3D the default is `NaN` (the integrated pressure drag already mixes induced drag with everything else). Optional lifting line,

\[
C_{D,i} = \frac{C_L^2}{\pi\,\mathrm{AR}\,e}, \quad e = 0.85
\]

is reported beside \(C_D\) and is not subtracted. `stall_model` accepts only `"none"`.

If the boundary layer took the laminar-freeze path, `operate.py` forces \(L/D\) to `NaN` even when \(C_D\) is finite.

#### 2.2.7 Farfield

2D only. The same vortex kernel (and the source kernel when \(\sigma\) exists) is evaluated on `cartesian_2d`. Cells tagged internal solid are `NaN`. Outputs are \(u, v, w\equiv 0, |\mathbf{V}|, C_p, \phi, \psi\). There is no 3D off-body field and no extracted streamline polyline. The workbench draws a quiver.

### 2.3 Useful outputs the product can deliver today

| Output | 2D | 3D | Trust level today |
|---|---|---|---|
| Surface \(C_p\) | Yes, upper/lower plot | Yes, mesh colored by \(C_p\) (first 25 000 facets) | Usable for attached inviscid sections. 3D wing \(C_p\) depends on a polynomial strip fit |
| \(C_L\) | Yes | Yes, inviscid | 2D attached lift is in the right band (§6). 3D is checked only against Helmbold within \(\pm 0.08\) |
| \(C_{D,\mathrm{pressure}}\) | Yes | Yes | Inviscid value is numerical leakage, not profile drag |
| \(C_{D,\mathrm{friction}}\) | Yes when an IBL exists | Always `NaN` | Order-of-magnitude on an attached 2D side. Omitted downstream of laminar separation |
| \(C_D\), \(L/D\) | Yes, unless laminar separation voids \(L/D\) | Pressure drag only | Not a design number once any side freezes laminar |
| \(C_{m,c/4}\), \(x_{cp}/c\) | Yes | Yes, about a root quarter-chord estimate | 2D moment sign on a cambered section disagrees with the NACA tables (§8) |
| \(\theta, \delta^*, H, C_f\), transition and separation stations | Yes | No stations are built | Upper-surface cruise behavior is tested. Lower-surface cruise separation is not |
| Farfield vectors, \(\phi\), \(\psi\) | Yes | No | Diagnostic |
| Newtons, polar fit, aerodynamic center, spanwise \(c_\ell\) | Polar and \(x_{ac}\) only if the caller stacks angles. The workbench analyzes one angle | Same | The workbench does not run a polar |
| JSON summary and NPZ of the contracts | Yes | Yes | The right export seed. Not a customer report |

Automated PDF, CSV polars, VTK, and Tecplot are not implemented.

---

## 3. Current Completion Percentage

Percentages are fractions of a **commercial launch**, not fractions of the files that happen to exist. A module can be “integrated” in the README and still be far from something a stranger should pay for.

Weights: the five physics modules are 60% of launch value, the five infrastructure modules are 25%, and the validation-plus-operations evidence a subscriber can audit is 15%.

| Block | Versus a market-ready version of itself | Weight | Contribution |
|---|---:|---:|---:|
| 1. Geometry | 60% | 12% | 7.2 |
| 2. Mesh | 65% | 12% | 7.8 |
| 3. Physics | 42% | 18% | 7.6 |
| 4. Aero reduction | 50% | 10% | 5.0 |
| 5. AI | 0% | 8% | 0.0 |
| 6. Customer UI | 18% | 8% | 1.4 |
| 7. API | 0% | 5% | 0.0 |
| 8. Persistence | 0% | 4% | 0.0 |
| 9. Auth and billing | 0% | 4% | 0.0 |
| 10. Deployment | 0% | 4% | 0.0 |
| Validation, CI, accuracy booklet | 20% | 15% | 3.0 |
| **Launch total** | | **100%** | **32%** |

**About 32% of a commercial launch.**

Read that number as two different facts:

- The **numerical pipeline (modules 1–4)** is roughly **half** of the architecture the project locked. Geometry, panel metrics, a real 2D solve, a real 3D solve, coupling, and a coefficient reducer all run, and they run through one workbench and seven pipeline cases.
- The **product around that pipeline** — accounts, stored projects, a job API, a deployable service, an accuracy statement backed by a public booklet — is only a few percent complete. There is no `Dockerfile`, no FastAPI app, no database, no auth, no CI configuration, and no `pytest.ini`. Tests are scripts with a `__main__` block.

Module-level notes behind the scores:

- **Geometry, 60%.** NACA 4-digit, plate, constant-chord wing, cylinder, SVG, DXF, STL, OBJ, typed errors, watertight check. Missing taper, twist, spanwise airfoil change, multi-element, STEP, and mesh healing.
- **Mesh, 65%.** Cosine / half-cosine / uniform panels, LE/TE clustering, outward normals, curvature, 3D metrics and transforms, Cartesian mask, 2D IBL stations. Every 3D mesh sets `ibl_1d` to `None` (`mesh_engine.py`). No wake mesh, no curvature-based refinement, no boundary-layer stations on a wing.
- **Physics, 42%.** 2D Dirichlet vortex solver, source sheet, Thwaites–Head, transpiration loop, 3D Morino solver, wake, forces, 2D farfield. 3D never marches a boundary layer. Several couplings are clipped, reset, or rejected. The 3D velocity model is a strip polynomial.
- **Aero, 50%.** The reduction contract is implemented and tested for rescale, one polar, and an aerodynamic center. Stall, Trefftz drag, atmosphere, and report files are named in the module README as left open.
- **UI, 18% of a product UI.** As a developer workbench it is much further along: four sources, mesh sliders, flow sliders, \(C_p\), boundary layer, quiver, 3D pressure, contract download. It is still a single Streamlit process with session state, one operating point, and no accounts.

---

## 4. Testing Status

### 4.1 What has been applied

There is no CI. The suite is six scripts under `tests/`, run as Python modules. Counts below are test functions, not assertions.

| Script | Functions | What they actually lock |
|---|---:|---|
| `test_geo_module.py` | 5 | Closed NACA polyline, wing thickness, watertight cylinder, bad NACA code, one catalog sample |
| `test_math_module.py` | 9 | Clustering, IBL side split, farfield box, closed-body normals, open wing allowed, degenerate triangle, open contour rejected, mask tags, surface density, wing orientation timing |
| `test_datasets.py` | 9 | Catalog titles, SVG fixture, DXF closed flag, pipeline geo→math, all seven cases, every geometry source, math overrides, contract round trip, path escape |
| `test_mesh_workbench.py` | 2 | Slider rebuilds for upload, shape, and pipeline; 3D resolution sliders |
| `test_physics_module.py` | 16 | Lift bands, cylinder nose-vs-side \(C_p\), separation returns a payload, rejected closures, mesh immutability, rectangular-wing planform and Helmbold lift, GMRES vs dense, sweep does not change planform, cruise **upper** skin friction, laminar freeze, trailing-edge cluster, friction uses freestream \(q\), separation hysteresis, 160-vs-64 friction within 25%, four physics fixtures |
| `test_aero_module.py` | 3 | Report matches physics and does not alias \(C_p\), area rescale, polar finds a finite aerodynamic center |

**Analytical and semi-analytical checks that exist**

| Check | Tolerance in the test | What a reference panel method would usually be held to |
|---|---|---|
| Flat plate, inviscid, \(\alpha = 4^\circ\), \(C_L\) vs \(2\pi\alpha\) | Absolute 0.08 | \(2\pi\alpha = 0.439\). A band of 0.08 is about \(\pm 18\%\) of the theory. The live probe hit \(C_L = 0.460\) (\(+4.8\%\)) |
| Flat plate, \(C_L\) vs \(2\Gamma/c\) | Absolute 0.05 | A healthy constant-vortex code is usually inside 0.01–0.02 at this panel count |
| NACA 0012, inviscid, \(\alpha = 0\), \(\|C_L\|\) | 0.02 | Live probe: \(C_L = -4\times 10^{-5}\) |
| NACA 0012, inviscid, \(\alpha = 0\), upper/lower \(C_p\) mirror | Max abs 0.08 | Loose for a symmetric section away from the trailing edge |
| NACA 0012, inviscid, \(\alpha = 4^\circ\), \(C_L\) vs 0.48 | Absolute 0.08 | Live probe at 160 cosine panels: \(C_L = 0.480\) |
| Cylinder, \(\alpha = 0\), \(\|C_L\|\) | 0.05 | The classical potential result is also \(C_p = 1 - 4\sin^2\theta\) (stagnation \(+1\), equator \(-3\)). The test only checks that mean nose \(C_p\) exceeds mean side-wall \(C_p\) on a 16×4 cylinder. It does not check those magnitudes |
| Rectangular wing vs Helmbold, \(a_0 = 2\pi\), \(\alpha = 4^\circ\) | Absolute 0.08 on a coefficient near 0.33, i.e. about a quarter of the estimate | Not a wind-tunnel comparison |
| Thwaites vs Blasius on a fixed \(U_e\) | \(\theta\) ratio under 2% | This is the one tight boundary-layer fixture (`thwaites_blasius`) |
| NACA 0012, \(\mathrm{Re}_c = 5\times 10^5\), \(\alpha = 4^\circ\), upper \(C_f\) on \(0.4 \le x/c \le 0.9\) | At least 70% of stations above \(5\times 10^{-4}\); shape factor in \((1.2,\ 2.5)\) | Does not compare \(C_f(x)\) or \(C_D\) with XFOIL or Abbott |
| 64-panel vs 160-panel friction drag | Relative 25% | A 25% move is accepted. That is a convergence check with a wide gate, not a grid-converged drag |

**Live probe from this audit** (unit freestream, \(\rho = 1.225\), \(\mathrm{Re} = 5\times 10^5\) unless noted). These numbers are solver output, not published data.

| Case | \(C_L\) | \(C_{D,p}\) | \(C_{D,f}\) | \(C_m,c/4\) | Notes |
|---|---:|---:|---:|---:|---|
| Plate, 120 uniform panels, \(\alpha=4^\circ\), inviscid | 0.460 | 0.0325 | — | 0.010 | Theory \(C_L = 0.439\), theory \(C_D = 0\) |
| NACA 0012, 160 cosine, \(\alpha=0\), inviscid | \(\approx 0\) | 0.00114 | — | \(\approx 0\) | Leakage drag. Matrix 1-norm condition \(1.64\times 10^7\) |
| NACA 0012, 160 cosine, \(\alpha=4^\circ\), inviscid | 0.480 | 0.00160 | — | 0.0059 | \(L/D \approx 299\) because leakage drag is tiny. That \(L/D\) is not physical |
| NACA 0012, 160 cosine, \(\alpha=4^\circ\), viscous | 0.481 | 0.00096 | 0.00604 | 0.0079 | Lower surface separates at \(s = 0.844\). Upper transition at \(s = 0.251\). **\(L/D\) is NaN.** Coupling took 8 iterations, residual \(9.9\times 10^{-5}\) |
| Same viscous point, 80 panels, \(M = 0.3\) | 0.504 | 0.00263 | 0.00614 | 0.0080 | Lift ratio vs the incompressible 160-panel run is about \(1/\beta\). Lower surface still separates. \(L/D\) is NaN |
| NACA 4412, 120 cosine, \(\alpha=0\), inviscid | 0.512 | 0.00161 | — | **+0.110** | Abbott \(C_{m,c/4}\) is about \(-0.10\) (nose-up positive). See §8 |
| NACA 0012, 64 cosine, \(\alpha=10^\circ\), viscous | 1.186 | **−0.0044** | 0.00306 | 0.009 | Total \(C_D\) is **negative**. \(L/D\) is NaN because laminar separation froze the upper surface at \(s = 0.175\) |

### 4.2 What has not been done

These are the checks that change a research pipeline into a number a subscriber can quote.

1. **XFOIL (or an equivalent 2D VIC code) booklet** on NACA 0012, 4412, 23012, and one laminar section (for example 64-210), at \(\mathrm{Re}_c = 10^5, 5\times 10^5, 3\times 10^6\), \(\alpha\) from \(-4^\circ\) to \(12^\circ\). Report \(C_L\), \(C_D\), \(C_m\), transition \(x/c\), and \(C_p\) at one attached angle. The current cruise case already fails the product rule “print \(L/D\)” on 0012 at \(4^\circ\).
2. **Abbott & von Doenhoff / Ladson** experimental anchors for the same sections, with the comparison labeled attached-flow only.
3. **Cylinder potential solution** at a refined azimuthal count: stagnation \(C_p \to 1\), equator \(C_p \to -3\), \(C_L \to 0\), \(C_D \to 0\), under grid refinement. The present test does not constrain the equator.
4. **Flat-plate skin friction**: Blasius \(C_f = 0.664\,\mathrm{Re}_x^{-1/2}\) and a turbulent \(1/7\)-power or White correlation, marched at zero pressure gradient with the transition station forced. Only \(\theta\) versus Blasius is locked, and only as a physics-module fixture.
5. **Grid convergence** of \(C_L\) and \(C_p\) at 40, 80, 160, 320 panels, with a required asymptotic band (for example \(C_L\) stable to 0.005). The 64-vs-160 gate allows 25% on friction drag.
6. **Camber and moment**: 4412 \(C_L(\alpha=0)\) and the sign of \(C_{m,c/4}\). No test covers 4412 forces at all.
7. **Kutta and trailing-edge leakage** under uniform refinement: \(\gamma_u + \gamma_l \to 0\), and inviscid \(C_D \to 0\) for a closed section (d’Alembert). Plate inviscid \(C_D = 0.032\) shows the leakage is still large on a thick “flat” polygon.
8. **3D wing versus AVL or VSPAERO** on the rectangular and the swept pipeline cases: \(C_L\), spanwise \(c_\ell\), induced drag. Helmbold \(\pm 0.08\) is not that.
9. **Closed-body \(C_p\) magnitude** on the cylinder, including the end caps, at 2× and 4× the current 16×4 mesh.
10. **Compressibility versus a known PG case**, 2D and 3D, checking that 3D does not pretend Göthert while only scaling \(C_p\).
11. **Separation**: a case that is supposed to separate (0012 near \(12^\circ\)–\(16^\circ\) at \(\mathrm{Re}_c = 5\times 10^5\)) and a case that must not (0012 at \(4^\circ\), **both** sides). Today \(\alpha = 10^\circ\) is required to separate, and \(\alpha = 4^\circ\) separates on the lower side anyway.
12. **Stress**: panel counts past the dense limit without allocating \(N^2\) when the spec promised matrix-free GMRES; non-watertight STL; duplicate vertices; \(\alpha = \pm 90^\circ\); \(\mathrm{Re}\) at the slider ends \(10^4\) and \(10^7\); Mach 0.65; a polar of 15 angles in one workbench session.
13. **Regression lock on the seven pipeline cases**: store \(C_L\), \(C_D\), and a \(C_p\) checksum. `test_datasets.py` still describes the pipeline as geo then math in one test name; physics and aero are live in `active_stages` and are not checksummed.
14. **UI**: no browser test that moving \(\alpha\) changes \(C_L\) and leaves the geometry camera alone. The spec lists that as an acceptance item. `test_mesh_workbench.py` exercises slider helpers, not a running Streamlit session.
15. **Performance budgets** in `PHYSICS_MODULE_SPEC.md` §5.7 (50 ms inviscid 2D at 160 panels, 400 ms viscous, 5 s for a ~2400-panel wing). Nothing in `tests/` times them.

---

## 5. Definition of “Fully Market-Ready”

A fully market-ready state for **this** SaaS, at the physics the project already chose, is the following list. Items already true are marked. Everything else is still work.

### 5.1 Solver a stranger can quote

1. **2D attached flow, \(\mathrm{Re}_c\) from \(10^5\) to \(3\times 10^6\), \(|\alpha|\) up to about \(8^\circ\) on a conventional section.** \(C_L\) within 0.02 of XFOIL, \(C_p\) within about 0.05 of XFOIL outside the leading-edge peak, transition location within about \(0.05c\), and \(C_D\) within about 15% of XFOIL. That is the practical content of “85–95% of a viscous solution” for a 2D panel method. **Not met.** The cruise flagship currently returns \(L/D = \mathrm{NaN}\).
2. **2D inviscid closed section.** \(C_D\) under about \(0.001\) at 160 panels (d’Alembert leakage), \(C_L\) within 0.01 of a reference panel code. **Lift is close on 0012. Drag leakage on the plate is 0.032.**
3. **3D wing, attached, inviscid.** \(C_L\) within about 5% of a vortex-lattice or a published panel code on the rectangular and swept cases; spanwise loading the right shape; induced drag from a Trefftz plane or a documented surface integral, not an optional \(e = 0.85\) that is not removed from \(C_D\). **Not met.**
4. **3D wing, viscous.** Strip or fully 3D integral boundary layer, skin friction in the total drag, and a stated limitation at the root and tip. **Absent.** `ibl_1d` is always `None` in 3D.
5. **Closed body.** Inviscid \(C_p\) that refines toward the analytical cylinder, and a clear label that viscous drag is not available until a boundary layer exists. **Only a qualitative nose-versus-side check.**
6. **Compressibility.** One documented correction, the same one in 2D and 3D, switched off automatically when a panel \(C_p\) implies supersonic flow. **2D and 3D do not use the same correction.**
7. **Failure that stays physical.** Separation sets a warning and a drag model that cannot print a negative \(C_D\). The \(\alpha = 10^\circ\) probe printed \(C_D < 0\) and then hid \(L/D\). **Not met.**

### 5.2 Product shell

8. **UI.** A project with a name, a geometry, a polar (not one angle), comparison of two runs, and error banners that say what failed. Stable camera. No developer language (“pipeline case”, “math sample”) on the path a customer uses. **Workbench only.**
9. **Cloud solver.** A job API with a queue, a panel-count budget, a time limit, and a result object keyed by the input hash. Streamlit-in-process is not that. **Absent.**
10. **Errors.** Typed exceptions already exist and the physics façade maps `LinAlgError`, `ZeroDivisionError`, and `IndexError` to `PhysicsEngineError`. Market-ready also means the workbench never shows a traceback, every warning has a next action, and a singular matrix names the geometry feature that caused it (trailing-edge cluster, zero-area triangle, open TE). **Partial.**
11. **Exports.** CSV of \(C_p(x/c)\), CSV of the polar, a one-page PDF or HTML report with the reference quantities and the warnings, and a VTK surface for 3D. JSON/NPZ contract download exists and should stay as the debug export. **Partial.**
12. **Speed, on the stored cruise meshes, warm process.** The spec budgets: 2D inviscid under 50 ms at 160 panels, 2D viscous under 400 ms, ~2400-panel wing under 5 s. Market-ready adds: a 15-angle polar inside a few seconds in 2D, and a density slider that does not allocate an \(N\times N\) matrix once \(N\) exceeds a few thousand. **Budgets are not measured in CI. GMRES still stores \(A\).**
13. **Accounts, stored runs, a subscription check.** Required before a price. **Absent.**
14. **A public validation note** reproduced by a command in the repo. **Absent.**

### 5.3 Explicit non-goals

Market-ready for this product does **not** mean a RANS solver, a transonic shock fit, rotating machinery, or 95% of experimental drag past stall. Those are different products. The honest label past separation is “the boundary layer left the integral model; lift is the inviscid field; drag is not reported.”

---

## 6. Current Flaws & Accuracy Gaps

Two different accuracy targets get mixed together in conversation. They are separated here.

### 6.1 “100% of an advanced panel method”

This means: the discrete singularity solution agrees with a reference panel method to within discretization error. It does **not** mean agreement with a wind tunnel. A correct constant-strength code on a closed section has \(C_D \to 0\) as the mesh is refined, a Kutta condition that is actually enforced, and a matrix whose residual is at the linear-algebra tolerance.

Gaps that keep this tree short of that standard:

1. **The 2D system is not the system the spec writes down.** Dirichlet streamfunction replaced Neumann. That substitution is justified in the source comments and is a known good formulation, but the spec, the skeleton comments, and the code have diverged. A later editor can “fix” the code back to the checkerboard Neumann matrix and destroy lift. The contract should name the Dirichlet system as the method.

2. **Source and vortex are not one matrix.** Lift is fixed before thickness and blowing are applied. A simultaneous source-vortex system (Hess-type) is what `panel_model_2d="source_vortex"` advertises. The sequential solve plus the `lstsq` fallback is a different method.

3. **Inviscid drag is not zero.** Probe: NACA 0012 at \(\alpha = 0\) has \(C_D = 0.00114\); the 1% thick plate at \(4^\circ\) has \(C_D = 0.0325\). The plate figure is large enough to move \(L/D\) to 14 on a shape whose inviscid drag should be a fraction of a count. Causes visible in the method: constant strength (no linear vortex), collocation at panel midpoints, a blunt trailing face on the plate polygon, and a Kutta pair chosen by a normal-component threshold of 0.7.

4. **Condition numbers are high.** The 160-panel 0012 Dirichlet matrix has a 1-norm condition estimate of \(1.64\times 10^7\). The guard is \(10^{12}\). A condition of \(10^7\) in float64 still leaves several digits, so this is not an immediate blow-up, but it is far from a well-scaled influence matrix, and `numpy.linalg.cond` is an extra factorization on every solve. There is no residual check after the dense LU. A solution can be finite and still fail to satisfy Kutta at tight tolerance.

5. **Kutta geometry is a heuristic.** `_kutta_pair` prefers panels with \(|n_y| > 0.7\). A vertical trailing face, a cove, or a highly cambered import can select the wrong pair. The spec’s rule is “the two panels that touch `te_index`.” The code leaves that rule when the normal test finds a different aft pair.

6. **3D wake is not the trailing edge.** Spec §5.2 classifies boundary edges by sorted endpoint keys and extrudes those edges. `solver_3d.py` bins centroids in \(y\), takes the maximum-\(x\) upper and lower panel in the bin, and builds the wake from centroids. On a swept or cosine-clustered skin the wake leaves from interior points, the last chordwise jump is smeared, and a strip with no panel on one side is dropped. That is a singularity-sheet leak: the trailing-edge potential jump is only approximately the Kutta jump.

7. **3D wing \(C_p\) is a curve fit, not the sheet velocity.** `_doublet_gradient_velocity` adds only an \(x\)-derivative of a degree-4 fit. The tangential basis `tangent_1` / `tangent_2` computed in Module 2 is not used on open wings. Strips with fewer than 6 panels contribute freestream only. Constant-strength doublets cannot represent a \(C_p\) peak at the leading edge; the polynomial then smooths whatever \(\mu\) the coarse triangles produced. This is the dominant 3D discretization error for suction peaks and for sectional lift.

8. **Closed-body velocity samples are an engineering offset.** Height \(0.08\sqrt{S}\) and tangential step \(0.25\sqrt{S}\), self-source removed. The spec asks for the exterior perturbation just off the panel. The offsets are untuned against the analytical cylinder, and the test does not hold \(C_p(90^\circ)\) near \(-3\).

9. **GMRES does not change the discretization, and it does not save memory.** The dense matrix is built first. For \(F\) panels the storage is \(8F^2\) bytes: about 46 MB at 2 400 panels, about 3 GB at 20 000, and about 12 GB at the 16× face count the workbench density slider can request (density 4 on a 2 400-panel wing). The spec’s interactive path for “above the dense limit” is not the path in the code. A singular or slow GMRES run raises `SingularInfluenceMatrixError` only after that allocation.

10. **3D Prandtl–Glauert is a \(C_p\) scale, not Göthert.** Dividing \(C_p\) by \(\beta\) without stretching the geometry mis-places the pressure peak and is inconsistent with the 2D path. Local supersonic \(C_p\) is not detected.

11. **Reference area on a non-lofted open surface is a bounding box.** Documented, and tested, and still the wrong area for a cropped STL wing that does not share spanwise stations. Coefficients on an upload can be silently off by the sweep inflation the wing path was written to avoid.

12. **Moment reference in 3D** uses a mid-span \(x\)-extent when the caller omits `c_ref` (`aero_module` contract). A tapered wing, once one exists, would get the wrong \(C_m\) until the user types a mean aerodynamic chord.

### 6.2 “85–95% of Navier–Stokes / a viscous hybrid”

This is the product claim for **attached** lift, pressure, and total drag. Integral methods in the XFOIL class can land in that band on conventional sections when transition and the displacement body are right. They do not land there through stall, in a wing-body junction, or in a separated cove.

What blocks that band here:

1. **The flagship viscous case does not return a usable drag.** NACA 0012, \(\alpha = 4^\circ\), \(\mathrm{Re}_c = 5\times 10^5\), 160 panels: upper transition at \(s \approx 0.25\), lower-surface laminar separation at \(s \approx 0.84\), \(C_{D,f} \approx 0.006\) (a plausible friction magnitude) and \(C_{D,p} \approx 0.001\), then `laminar_drag_is_unsupported` forces \(L/D\) to NaN. A subscriber who opens the default pipeline case does not get a glide ratio. The upper-surface tests pass because they never look at the lower side.

2. **Viscous coupling is deliberately weakened.** Blowing is clipped to 5% of \(V_\infty\), under-relaxed at 0.25, and discarded entirely if \(|\gamma|\) or \(C_p\) spikes. On the 0012 probe the viscous \(C_L\) (0.481) matches the inviscid \(C_L\) (0.480) to three digits. A real displacement interaction moves lift by more than that on a 12% section at this Reynolds number, and it moves pressure drag by more than the 0.001 left in the result. The coupling is stable because most of the blowing never reaches the matrix.

3. **Transition is Michel with no freestream turbulence, and the march does not start at the stagnation point.** Stations are split at the geometric leading edge. `_march_speed` then flattens the first 30% of whichever side slows down. That is a patch for a bad station origin. It shifts \(\mathrm{Re}_s\), \(\theta\), and the station at which \(\lambda = -0.09\). Michel at \(\mathrm{Re}_c = 5\times 10^5\) with zero \(Tu\) is also early or late relative to \(e^N\) depending on the section. XFOIL’s \(e^N\) envelope is the method that actually sits in the 85–95% drag band. It is not implemented; `drela_two_equation` is rejected on purpose.

4. **Separation is a freeze, not a flow.** Downstream \(\theta\), \(H\), and \(C_f\) are held, and \(C_f\) is set to 0. There is no wake, no pressure plateau, and no mass deficit fed back as a thick displacement body. The \(\alpha = 10^\circ\) probe produced \(C_{D,p} = -0.0044\) and a total \(C_D < 0\). Negative drag is the signature of an inviscid pressure field that still nearly satisfies d’Alembert while the friction integral has been zeroed over the separated run. The aero layer hides \(L/D\) and leaves the negative \(C_D\) in the coefficient block.

5. **No 3D boundary layer at all.** Requesting `viscous_inviscid` on a wing warns “3D boundary layer skipped” and solves inviscid. Profile drag of a drone wing, which is most of the drag at low lift, is missing. The 85–95% total-drag claim is not applicable to any 3D case in the catalog.

6. **Induced drag is not a viscous result and is not a Trefftz result.** Default `cd_induced` is NaN. The lifting-line estimate uses \(e = 0.85\) and is not removed from the surface-pressure drag, so a user who turns it on can double-count. Surface-pressure integration on a short wake also leaves a residual streamwise force that is not the induced drag of the lifting line.

7. **No stall model, by contract.** \(C_{L,\max}\) on a polar is the largest sample. At \(10^\circ\) the probe still returned \(C_L = 1.19\), which is an inviscid number. Experimental 0012 at \(\mathrm{Re}_c = 5\times 10^5\) is near stall in that neighborhood and does not carry that lift with that drag. Shipping this as a viscous \(C_L\) would over-claim by a wide margin past roughly \(8^\circ\)–\(10^\circ\).

8. **Reynolds number does not enter the inviscid matrix, and the boundary layer does not see a real gas.** That is correct for an incompressible VIC method and wrong for any UI that lets the user type altitude, speed, and a chord and expect Sutherland viscosity and a Mach number to move together. Today those knobs are independent, and only \(\mathrm{Re}\) and \(M\) do anything.

9. **Head’s march is first-order and serial.** One explicit step per mesh interval, shape factor clipped, \(C_f\) clipped at 0.04, \(H\) floored at 1.15. Clustered trailing-edge stations are pinned to the last resolved state. That pinning was added so a cosine gap cannot trip separation by itself, which is right, and it also means the last few percent of chord never tell the truth about trailing-edge separation.

**Bottom of the accuracy section.** Attached 2D **lift** on NACA 0012 at \(4^\circ\) is already in a respectable panel-method band (\(C_L = 0.480\) against the project’s own 0.48 target). Attached 2D **drag**, cambered **moment**, every **3D viscous** number, and every **separated** number are not in the 85–95% band, and the product currently refuses to print \(L/D\) on its own cruise case.

---

## 7. Remaining Modules

### 7.1 Physics and analytics still inside modules 1–4

These are not new module numbers. They are missing pieces of modules that the README already calls live.

| Piece | Where it should live | Why it blocks release |
|---|---|---|
| IBL stations on a wing (spanwise strips, stagnation per section) | `math_module` | Module 3 is required not to invent stations. 3D viscous cannot start until this exists |
| Simultaneous 2D source-vortex matrix, Dirichlet form written into the spec | `solver_2d.py`, spec | Removes the `lstsq` fallback and the spec/code split |
| Displacement-body coupling, or a transpiration formulation that does not need a 0.05 clip and a spike reset | `physics_engine.py`, `boundary_layer_1d.py` | Current coupling does not move lift |
| \(e^N\) or Drela’s two-equation IBL | `boundary_layer_1d.py` | The closure the spec already reserved a name for |
| Wake drag (Squire–Young or a wake integral) so separation cannot print \(C_D < 0\) | `forces_and_moments.py` | Separated drag |
| True trailing-edge wake from boundary edges | `solver_3d.py` | 3D Kutta |
| Sheet velocity from the solved doublet, in the panel tangent basis | `solver_3d.py` | 3D \(C_p\) |
| Matrix-free 3D matvec | `solver_3d.py` | Density slider and memory |
| The same compressibility correction in 2D and 3D | `physics_engine.py` | Mach knob |
| Strip boundary layer using the new 3D stations | `physics_engine._solve_3d` | 3D profile drag |
| Trefftz induced drag | `aero_module` | A \(C_{D,i}\) that is not a second copy of \(C_D\) |
| Friction contribution to \(C_m\) | `forces_and_moments.py` | The comment claims it; the code omits it |
| Stall policy: stop reporting \(C_L\) as viscous past a documented \(\alpha\), or couple a model that is allowed to change the contract | `aero_skeleton.py` | Prevents a false \(C_{L,\max}\) |
| Polar driver (solve \(N\) angles, then `reduce_polar`) | workbench and `pipeline_runner` | The aero polar code exists and the UI never calls it |
| Taper, twist, spanwise section, sideslip | `geo_module`, freestream | The wing a drone designer actually has |
| STEP, spline DXF, non-watertight STL with a stated repair or a stated refusal that the UI explains | `geo_parsers.py` | Upload path |

### 7.2 Module 5 — AI (`ai_module`)

Not started. `datasets/modules/ai/README.md` is a placeholder. No skeleton file. Market-ready contents, in the order they earn their place:

- A retrieval note over the run that is already computed (“upper transition at 25% of arc length, lower surface frozen, \(L/D\) withheld”), which is a template plus the warnings, not a model.
- A surrogate for \(C_L, C_D\) inside the validated envelope, trained on this solver, with the residual against the solver shown.
- Shape suggestions only after the 2D drag number is trustworthy. An optimizer wrapped around today’s lower-surface separation will tune the hysteresis thresholds.

### 7.3 Module 6 — UI

Keep the Streamlit workbench as the lab. The customer UI still needs:

- One operating point and a polar on the same page.
- A warnings panel that distinguishes “drag withheld”, “3D boundary layer not built”, and “matrix singular”.
- 3D \(C_p\) that does not silently drop facets past 25 000 (`mesh_workbench.py`, `_render_pressure_3d`).
- No second Streamlit entry that can drift (`app.py` and `app_mesh.py` are the same stage today; that is fine only while one of them re-exports the other).
- Session persistence beyond `st.session_state`.

### 7.4 Modules 7–10

| Module | State | Minimum for a public beta |
|---|---|---|
| 7. API | No code | `POST /runs` with the existing `PhysicsInput`, `GET /runs/{id}`, panel and time quotas |
| 8. Database | No code | Users, projects, geometries, run records, the NPZ blob |
| 9. Auth and billing | No code | Login, a plan at the chosen price, a hard stop on panel count |
| 10. Deployment | No code | One container, pinned `requirements.txt` (already pinned), a worker process separate from the web process, health check |

`requirements.txt` today is NumPy, SciPy, Pydantic, typing-extensions, Streamlit, Plotly, and pandas. That list is an honest description of a local workbench.

---

## 8. Exhaustive List of Bugs, Limitations, Uncertainties & Errors

Severity is **blocker** (wrong or missing number on a path the UI offers), **major** (right architecture, wrong or fragile numerics), or **limit** (an honest ceiling, already visible in code or docs).

### 8.1 Blockers

**B1. Default cruise case withholds \(L/D\).**
NACA 0012, \(\alpha = 4^\circ\), \(\mathrm{Re} = 5\times 10^5\), 160 cosine panels, viscous regime. Lower surface separates at arc length 0.844. `laminar_drag_is_unsupported` is true. `AeroEngine` sets `l_over_d` to NaN. The workbench prints `L/D=nan`. The upper-surface unit tests pass on this same angle. This is the pipeline case the workbench opens on.

**B2. Separated 2D drag can be negative.**
Same section, 64 panels, \(\alpha = 10^\circ\), viscous: \(C_{D,p} = -0.00437\), \(C_D = -0.00131\), \(C_L = 1.186\). The aero layer hides \(L/D\) and still returns the negative coefficient. A CSV export of `coefficients.cd` would publish it.

**B3. Cambered pitching moment has the wrong sign relative to the standard tables.**
NACA 4412, \(\alpha = 0\), inviscid, 120 panels: \(C_L = 0.512\), \(C_{m,c/4} = +0.110\). With the code’s own center-of-pressure formula \(x_{cp}/c = 0.25 - C_m/C_L\), that moment places \(x_{cp}\) near \(0.03c\). Abbott & von Doenhoff list \(C_{m,c/4} \approx -0.10\) for this section (nose-up positive), which places the center of pressure aft of the quarter chord. The integral in `_loads_2d` is \(M_z = \sum (r_x F_y - r_y F_x)\) with \(\mathbf{F} = -C_p\mathbf{n}S\), which is nose-up positive for \(x\) aft and \(y\) up. No test asserts 4412 moment. Until a stored convention says otherwise, this is a sign defect, not a documentation preference. Magnitude being near 0.10 suggests the couple is real and the sign is the part that is wrong — or the pressure distribution’s camber loading is reversed. Either one is a blocker for \(C_m\) and for \(x_{ac}\).

**B4. 3D viscous mode is a no-op.**
`MeshEngine` sets `ibl_1d = None` for every 3D mesh. `PhysicsEngine._solve_inner` then rewrites the regime to inviscid and appends “3D boundary layer skipped”. `_solve_3d` assigns `ibl = None` again and sets `cd_friction` to NaN. The workbench still offers the viscous regime on wings. Skin friction, transition, and separation on a wing do not exist.

**B5. Inviscid \(L/D\) is a leakage quotient.**
0012 at \(4^\circ\), inviscid: \(C_D = 0.00160\), \(L/D \approx 299\). The UI will display that if the user selects inviscid. It is \(C_L\) divided by discretization drag.

### 8.2 Major numerical defects

**M1. Transpiration is clipped, under-relaxed, and sometimes deleted.**
`transpiration_on_panels` clips to \(\pm 0.05\). The VIC loop uses \(0.25\) new \(+\) \(0.75\) old. A spike (\(\max|\gamma| > 8\) or \(\min C_p < -20\)) replaces the field with the inviscid solution and clears convergence. Viscous \(C_L\) on the 0012 probe did not move relative to inviscid lift at the third digit.

**M2. Stagnation is the geometric leading edge, then a 30% flatten.**
`build_ibl_stations_1d` sets `stagnation_xy` to `nodes[le]`. `_march_speed` overwrites the edge velocity from the start of the side up to the minimum inside the first 30% of stations. On a cambered section or at high \(\alpha\) that window is not the stagnation point. Transition \(s\) and \(\lambda\) inherit the error. This is the most likely cause of B1.

**M3. Laminar separation hysteresis is tuned to one mesh.**
`_LAMINAR_RELEASE_LAMBDA = -0.08` and `_ADJACENT_TIE_DEPTH` are comments about the 160-panel cruise mesh, where coupling rebounds from about \(-0.091\) to about \(-0.089\). The thresholds are global. Another section can keep a false separation or drop a real one.

**M4. Source-matrix singularity is swallowed.**
`solve_2d` catches `SingularInfluenceMatrixError` on the source system and calls `lstsq`. The vortex condition number is the one stored in `influence_condition_number`. A rank-deficient source sheet does not raise.

**M5. Dense condition is an extra \(O(N^3)\) and is not the LU residual.**
`_solve_dense` calls `numpy.linalg.cond` and then `scipy.linalg.solve`. Spec §5 says to record the condition from the factor that solved the system. There is no \(\|A x - b\|\) check.

**M6. 3D matrix is always dense.**
`solve_3d` allocates `(count, count)` before choosing LU or GMRES. The “GMRES” warning means “iterative solve of the dense matrix.” It does not mean matrix-free. Workbench surface density 2 and 4 multiply face count by 4 and 16.

**M7. Wake built from centroids and a \(y\)-bin.**
`_trailing_panels` uses `span/40` bins and `argmax` of centroid \(x\). Dihedral, sweep, and a blunt tip change which triangles are “upper.” The wake length is applied to centroids, so the sheet does not contain the true trailing edge. Both triangles of a strip share one doublet jump, which is intentional, and the jump is between the wrong panels if the binning is off.

**M8. Open-wing speed ignores the tangent basis and ignores spanwise derivatives.**
`_doublet_gradient_velocity`. Polynomial degree \(\le 4\), minimum 6 panels per side per bin, slope added only to the global \(x\)-component, then the normal component is removed. Coarse tips and the leading-edge suction peak are the casualties.

**M9. Closed-body finite-difference velocity is untuned.**
Offsets `1e-4` inside for the Dirichlet point, `0.08 * sqrt(area)` off the wall, `0.25 * sqrt(area)` tangential step. No analytical \(C_p\) magnitude test.

**M10. 3D compressibility does not match 2D.**
2D stretches \(x\) by \(\beta\) and divides \(C_p\) by \(\beta\). 3D only divides \(C_p\). The 2D path then rebuilds \(|V_t|\) from the scaled \(C_p\) with a `sqrt(max(1 - C_p, 0))` clamp, so the boundary layer at Mach 0.3 sees a hybrid speed, not the incompressible edge speed the IBL correlations were written for. The Mach 0.3 probe still separated on the lower surface.

**M11. Friction is omitted from \(C_m\).**
`_loads_2d.moment` returns the pressure couple only. The comment above the return describes a friction moment that is not computed. \(C_L\) and \(C_D\) do include friction.

**M12. Kutta pair can leave the trailing-edge node.**
`_kutta_pair` switches to aft panels with \(|n_y| > 0.7\) when those are not the two panels that touch `te_index`. The plate uses a finite thickness, so this path is live.

**M13. Head march can leave a station inconsistent.**
The loop writes \(C_f\) at `index` and the updated \(\theta, H\) at `index + 1`, then the trailing-edge cluster may overwrite the update with an upstream anchor, and the last station’s \(C_f\) is recomputed after the loop from the frozen shape factor. Separation sets \(C_f\) to 0 downstream but the last-station recompute runs only if the loop did not return early. The logic is easy to break with the next mesh change. It is also a Python loop over stations, which the project rules forbid for inner kernels and which the physics spec explicitly permits for this march. The spec should stay the authority for this one loop; the project rule and the spec currently disagree.

**M14. Coupling residual is an absolute \(\delta^*\) change on the first pass.**
`_delta_residual` returns \(\max|\delta^*|/c\) when there is no previous solution, and \(\max|\Delta\delta^*|/c\) after that. The first number is the thickness itself, not a convergence rate. A thick laminar layer can look “unconverged” on iteration 1 for a reason that is not a failure, and a thin layer can look converged before the inviscid field has felt the blowing. Combined with M1, “`converged = True`” does not mean the displacement iteration is tight. The 0012 viscous probe did report residual \(9.9\times 10^{-5}\) after 8 iterations, under the \(10^{-4}\) tolerance, while lift had not moved.

**M15. Negative \(C_p\) after the PG division can zero the edge speed.**
In `solve_2d`, if \(C_p/\beta > 1\), the rebuilt tangential speed is driven toward 0 by `sqrt(max(1 - C_p, 0))`. A stagnation panel is fine. A panel that the scaling pushed past \(C_p = 1\) is not stagnation; it is a clamped sensor.

### 8.3 Geometry and mesh limits that become solver errors

**G1. Constant-chord wing only.** No taper, no twist, no spanwise airfoil blend (`generate_3d_parametric_wing`).

**G2. No sideslip.** Freestream \(y\)-component is hard-coded to 0.

**G3. Open wing skins are not capped.** Correct for a panel method, and easy for an upload path to violate: imported STL must be watertight, generated wings must not be. A user who caps a wing in a CAD tool and exports STL gets a closed-body solve (no wake, Morino condition) instead of a lifting wake. Nothing in the UI explains that branch.

**G4. SVG Bézier and arc commands, and binary DXF, are rejected.** The typed error exists. Customer files will hit it.

**G5. 2D IBL requires a closed contour.** An open polyline cannot be solved. That is validated. A section with a trailing-edge gap that does not repeat the first point fails in Module 2 rather than with a trailing-edge wake.

**G6. Farfield extent and mask are visualization.** They do not set boundary conditions. A user who tightens `n_x` and the box is not refining the solver. The workbench presents those sliders next to the panel count, which does refine the solver.

**G7. Surface density multiplies triangles and then the dense matrix.** Documented in the spec as something the solver must still accept. Memory behavior in M6 makes density 4 a process risk.

### 8.4 Product and test gaps that hide the blockers

**T1. No test reads lower-surface separation on the cruise point.** `test_tied_transition_stays_attached` asserts the upper side. B1 is invisible to the suite.

**T2. No test asserts the sign of \(C_m\) on a cambered section.** B3 is invisible.

**T3. No test asserts inviscid \(C_D\) is small.** B5 is invisible. The lift tests can pass while d’Alembert leakage is 0.03.

**T4. Cylinder test does not assert \(C_p\) magnitudes.**

**T5. Dataset copy is stale.** `airfoil_naca_0012_cruise` description still says “Later: potential/panel solve, CL/CD/CM, AI insight” while `active_stages` already contains physics and aero.

**T6. No CI, no timing test, no browser test.** A regression in the workbench camera or in the 5-second wing budget will not fail a check.

**T7. 3D \(C_p\) view truncates at 25 000 facets without a warning** in the figure itself. The solve may have used every facet; the picture does not.

**T8. Workbench 3D caption formats \(L/D\) with `.2f`.** NaN prints as `nan`, which is ugly and, on the viscous wing path, is the number next to a skipped boundary layer. 2D uses an explicit NaN guard; 3D does not.

**T9. `cd_induced` lifting-line warning says the estimate is not subtracted.** If a future UI adds it into a “total drag” tile, it will double-count. The contract forbids the subtraction. The UI does not yet offer the switch, so this is latent.

**T10. Physics samples and pipeline cases are not checksummed.** A 1% \(C_L\) drift does not fail `test_datasets.py`.

### 8.5 Uncertainties (not yet bugs, not yet proven)

**U1. 4412 lift level.** Probe \(C_L(\alpha=0) = 0.512\). Thin-airfoil estimate using \(\alpha_{L0} \approx -4^\circ\) is about 0.44. Experimental values near \(\mathrm{Re}_c = 10^6\) are near 0.4. Some of the gap is inviscid-versus-viscous. The rest may be the same discretization that produced B3. It needs a panel-code reference before it is called a lift bug. It is already too far from experiment to quote.

**U2. Rectangular-wing \(C_L\) versus Helmbold, inside \(\pm 0.08\).** The test can pass with a 20% error. This audit did not re-solve the wing. The true error against AVL is unknown.

**U3. Whether the Dirichlet matrix’s \(10^7\) condition pollutes \(C_p\) in the leading-edge bin more than the constant-strength model already does.** Likely second-order next to constant strength. Worth a residual column in the validation booklet, not a rewrite by itself.

**U4. Memory growth of the farfield evaluator.** `field_reconstruction.py` batches rows. This audit did not measure the 64 MB ceiling the spec names. 2D panel counts in the workbench are small enough that this is secondary to M6.

### 8.6 What was looked for and not found

No unbounded cache that retains every mesh for the life of the process beyond normal Streamlit session state. No `eval` of user geometry. Dataset paths reject traversal (`test_dataset_path_escape_rejected`). Module 3 copies the mesh before the solve (`test_mesh_arrays_are_not_mutated` and `copy_mesh`). Typed errors exist for bad NACA codes, open contours, degenerate triangles, Mach \(\ge 1\), and the two unimplemented closures. Those paths are in good shape and should stay frozen while the blockers above are fixed.

---

## 9. Bulletproof Solutions & Implementation Plan

The order is the order of dependence. Do not start the 3D boundary layer before 2D cruise drag is a number the tests lock. Do not start billing before a validation booklet can be regenerated by a command.

### Phase A — Make the current 2D cruise case tell the truth

**A1. Restart the boundary layer at the aerodynamic stagnation point.**

On the solved inviscid \(|V_t|\), find the station of minimum speed on the whole closed contour, not inside a 30% window of each geometric side. Split upper and lower there. Recompute \(s = 0\) at that point. Delete `_march_speed`’s flatten once the split is in the right place. Keep Module 2’s geometric stations as the mesh; do the split in Module 3 so a change of \(\alpha\) does not require a remesh. If both sides still show \(\lambda \le -0.09\) at \(4^\circ\) on 0012, the edge velocity is wrong (A2) and the split was not the only fault.

Acceptance: NACA 0012, 160 cosine panels, \(\alpha = 4^\circ\), \(\mathrm{Re}_c = 5\times 10^5\), both `separation_s_upper` and `separation_s_lower` are NaN, upper transition \(s/c\) within 0.05 of an XFOIL \(e^9\) run at \(N = 9\), and \(L/D\) is finite.

**A2. Stop clipping the physics to keep the iteration alive.**

Replace the sequential “vortex, then source, then maybe throw it away” loop with one linear system per VIC pass:

- Dirichlet (or a least-squares mix of Dirichlet and Neumann) for the vortex unknowns and the streamfunction constant.
- The source unknown in the **same** matrix, with transpiration as the Neumann data for \(\sigma\), not as a second solve that can be discarded.
- Under-relaxation on \(\delta^*\), not on a clipped \(v_n\). A limit belongs on the update of \(\delta^*\), with the limit recorded in `warnings` when it is active.
- Delete the \(\max|\gamma| > 8\) and \(\min C_p < -20\) reset. If the matrix is singular, raise `SingularInfluenceMatrixError`. If the boundary layer separates, take the Phase A3 path. Do not silently revert to inviscid pressure while leaving `regime = "viscous_inviscid"`.

Remove the `lstsq` fallback in `solve_2d`. A rank-deficient source matrix is a bug, not a least-squares problem.

Acceptance: at the A1 condition, \(|C_{L,\mathrm{viscous}} - C_{L,\mathrm{inviscid}}|\) is greater than 0.005 (the interaction is actually in the system) and smaller than 0.03 (it has not run away). Inviscid 0012 \(C_L\) at \(4^\circ\) stays inside \([0.46,\ 0.50]\).

**A3. Separated drag must be impossible to misread.**

When a side freezes laminar:

- Keep the warning that already exists.
- Set `cd` and `cd_pressure` to NaN, not only `l_over_d`. A negative \(C_D\) must fail a unit test.
- Add a wake contribution before those fields are allowed to be finite again: Squire–Young from the trailing-edge \(\theta\) and \(H\) of the last **attached** station, labeled `cd_wake` in the loads contract (additive field, do not rename `cd_pressure` or `cd_friction`).
- Do not add a stall curve in this phase.

Acceptance: the current \(\alpha = 10^\circ\) fixture returns finite \(C_p\), `separated = True`, and `cd` is NaN until the wake model is on. With the wake model on, \(C_D > 0\).

**A4. Lock moment sign on 4412.**

Compute \(C_m\) about the quarter chord for inviscid NACA 4412 at \(\alpha = 0\) and at \(\alpha = 4^\circ\). Compare with a reference panel code using the same nose-up-positive convention. If this code’s couple is mirrored, the bug is in the normal orientation used by the moment arm (outward normal versus the force \(-\,C_p\mathbf{n}\)) or in which side was called upper when the contour was split. Fix the integral; do not negate the reported coefficient in the UI.

Add the friction moment the comment already promises, using the same station midpoints as `_segment_traction`, so viscous \(C_m\) changes when \(C_f\) changes.

Acceptance: inviscid 4412 \(C_{m,c/4}\) is negative and within 0.02 of the reference panel code. A pure-friction fixture (the one that already locks \(C_D\)) also locks a nonzero \(C_m\) when the shear is applied off the chord line.

**A5. Publish d’Alembert leakage as a test.**

Inviscid closed 0012 at \(\alpha = 0\), 160 panels: \(|C_D| < 5\times 10^{-3}\) as a near-term gate, \(< 2\times 10^{-3}\) once the Kutta pair is the true trailing-edge panels only. Flat plate at \(1\%\) thickness: \(|C_D| < 0.01\) before any claim that inviscid \(L/D\) can be shown. Until that gate passes, the workbench must not display inviscid \(L/D\). Display \(C_L\), \(C_m\), and “inviscid pressure drag is numerical leakage” instead. That is a UI change of a few lines in `_render_flow_tabs_2d` and it removes B5 even before the leakage falls.

**A6. Write the method that is actually solved into the spec.**

Replace §5.1’s Neumann formula with the Dirichlet system, the extra streamfunction unknown, and the simultaneous source column. A spec that contradicts `solver_2d.py` will be “fixed” in the wrong direction later.

### Phase B — Panel-method fidelity (the “100% of the method” work)

**B1. Kutta pair.** Use only the two panels that touch `te_index`, as `trailing_edge_panels` already computes. Keep the blunt-face search behind a test that names a blunt geometry, rather than as a silent override on every airfoil.

**B2. 3D wake from boundary edges.** Implement the spec’s edge-key classification. Extrude the actual trailing-edge segments. Keep \(\mu_w = \mu_u - \mu_l\). Add a test that the wake vertices include the trailing-edge nodes of the rectangular wing, not only centroids.

**B3. 3D velocity from the doublet sheet.** Differentiate \(\mu\) along `tangent_1` (chordwise) and `tangent_2` (spanwise) with a fit that is allowed to see the neighbor panels, including the wake jump on the trailing row. Delete the global-\(x\) polynomial as the production path after the new velocity matches it to a tolerance on an unswept wing and beats it on the swept pipeline case (sweep should change sectional \(C_p\)).

**B4. Cylinder convergence.** Refine segments and rings. Assert stagnation \(C_p > 0.9\), equator \(C_p < -2.5\) on the mid-span ring, \(|C_L| < 0.02\), \(|C_D| < 0.02\), before calling the closed-body path validated. Adjust the off-wall sample only if the analytical values are missed; do not tune it to a single coarse mesh.

**B5. Matrix-free 3D path.** Build the matvec from the same `_panel_influence` batches used to fill \(A\), and do not allocate \(A\) when `count > dense_panel_limit`. Record `influence_condition_number = NaN` as the spec already requires for GMRES. Add a test that a 2 500-panel call’s peak array is the row batch, not a 2 500² matrix. Keep the existing dense-vs-GMRES \(C_L\) agreement test on the small wing, where both paths still form \(A\).

**B6. One compressibility model.** Either stretch 3D in \(x\) by \(\beta\) before influence (consistent Göthert) or stop stretching 2D and document a pure \(C_p/\beta\) Karman-style scale. Do not mix them. Feed the boundary layer the **unscaled** incompressible edge speed, and apply \(C_p/\beta\) only to reported pressure and forces. Refuse the correction on any panel whose scaled \(C_p\) would imply \(M_\mathrm{local} \ge 1\), with a warning that names the panel.

**B7. Condition number from the factor, plus a residual.** Use the LAPACK condition estimate attached to the solve, or `scipy.linalg.lu` once and a triangular condition estimate. Store \(\|Ax-b\|_\infty / \|b\|_\infty\). Fail the solve above \(10^{-8}\) on a dense 2D system rather than only above a condition of \(10^{12}\).

### Phase C — The viscous band (85–95% against XFOIL / a tunnel, attached flow only)

**C1. Replace Michel with an \(e^N\) envelope, or implement the Drela two-equation closure the literal already names.** Keep Thwaites–Head available as the fast mode. The validation booklet quotes the \(e^N\) mode. Default \(N = 9\), user-settable, because that is the knob experimental drag actually depends on.

**C2. Displacement iteration that is allowed to move the panels’ normals in a copied mesh, or the simultaneous transpiration from Phase A2 carried to convergence without a 5% clip.** Target: 0012 and 4412 \(C_D\) within 15% of XFOIL at \(\mathrm{Re}_c = 5\times 10^5\) and \(3\times 10^6\), \(\alpha = 0^\circ, 2^\circ, 4^\circ, 6^\circ\), both sides attached.

**C3. 3D strip boundary layer.** Module 2 emits `ibl_1d` per spanwise station (arc length, local chord, local \(\mathrm{Re}_c\)). Module 3 marches each station with the Phase C closure and couples a strip transpiration back onto the triangles of that station. Root and tip stations are marked approximate. `cd_friction` becomes finite. The warning “3D boundary layer skipped” remains only when the mesh genuinely has no stations.

**C4. Trefftz \(C_{D,i}\).** Integrate the wake jump far downstream. Report it as `cd_induced`. Leave it out of `cd` until the surface-pressure axial force is corrected by the same wake, so the sum is not doubled. Delete the default \(e = 0.85\) from any path the UI can reach, or keep it behind an “estimate” label that cannot be added to \(C_D\).

**C5. Validation booklet as a test.** A script writes `datasets/validation/booklet.json` with the XFOIL and cylinder numbers and fails if a locked case moves by more than the gate. This replaces the \(\pm 0.08\) lift bands as the release criterion. Keep the wide bands only as smoke tests so a broken import still fails in seconds.

### Phase D — Product shell, after Phase A is green and Phase C1–C2 are at least in progress

**D1. Polar in the workbench.** Loop \(\alpha\), call `reduce_polar`, plot \(C_L(\alpha)\), \(C_D(\alpha)\), \(L/D\). This code path exists in `aero_module/polar.py` and is unreached by the UI.

**D2. Exports.** CSV for \(C_p\) and the polar. HTML report that prints reference area, \(\mathrm{Re}\), \(M\), warnings, and “drag withheld” in the same sentence as the coefficients.

**D3. API and worker.** One FastAPI process, one solver worker, the existing `PhysicsEngine.solve_from_mesh` as the job body, a panel-count quota.

**D4. Accounts and a project table.** Not before D3. Billing enforces the quota from D3.

**D5. Geometry a customer will actually upload.** Taper and twist on the wing generator. A clear closed-body versus lifting-surface choice when an STL is both watertight and flat. Spline DXF, or a refusal that shows the unsupported entity.

### What not to do

- Do not add a RANS mode to close the accuracy gap. It abandons the product’s runtime claim and the data contracts.
- Do not negate \(C_m\) in the UI to match Abbott without fixing A4. The aerodynamic center uses the same sign.
- Do not train the AI module on the current cruise output. It would learn that \(L/D\) is NaN at \(4^\circ\).
- Do not raise the laminar-separation threshold until A1 is done. The \(-0.08\) release band is hiding a station-origin problem, not defining a new correlation.

---

## 10. Monetization & Subscription Value Justification ($45–$60/mo)

### 10.1 What the price is being compared with

XFOIL, XFLR5, AVL, and VSPAERO are free. A $45–$60 monthly subscription is a **time and trust** price, not a physics-monopoly price. It is worth paying when all of the following are true:

1. A section polar that matches XFOIL closely enough that the user does not re-run XFOIL to check it.
2. A wing polar, including profile drag, in seconds, without a Windows-only desktop tool and a manual lattice.
3. A project the user can close and reopen, with the geometry and the warnings still attached.
4. A one-page export they can drop into a design review.
5. A written accuracy envelope, and a UI that goes blank on drag rather than printing a negative \(C_D\) or an \(L/D\) of 300.

None of those five are true of this tree today. Charging the price now would buy a local Streamlit workbench that withholds \(L/D\) on NACA 0012 at cruise.

### 10.2 The feature set that earns the price

| Capability | Bar | Why a professional pays |
|---|---|---|
| 2D attached polar | \(C_L\) within 0.02 and \(C_D\) within 15% of XFOIL from \(-2^\circ\) to \(6^\circ\), \(\mathrm{Re}_c\) at \(2\times 10^5\), \(5\times 10^5\), \(3\times 10^6\), for 0012 and 4412 | Replaces a local XFOIL install and a spreadsheet |
| 2D runtime | 15 angles, 160 panels, under 5 s on one worker core | Fast enough to sit inside a design meeting |
| 3D wing | Rectangular and swept pipeline cases, \(C_L\) within 5% of AVL or VSPAERO, \(C_{D,i}\) from the wake, \(C_{D,f}\) from a strip boundary layer | The number XFOIL cannot produce |
| 3D runtime | 2 000–4 000 panels, one angle, under 10 s, matrix-free above 2 000 | Density slider does not kill the worker |
| Honesty | Inviscid \(L/D\) hidden. Separated \(C_D\) is NaN or a labeled wake model. 3D without stations cannot be requested | The subscription is for a number that survives a review |
| Envelope page | The booklet from Phase C5, linked from the report footer | The user can defend the number |
| Project and export | Saved geometry, CSV polar, HTML one-pager | The alternative is screenshots of Streamlit |
| Quota | A stated panel budget and concurrent-job limit at this price | The cost of the worker stays inside the fee |

Optional at this price, not required to justify it: an \(e^N\) slider, taper and twist, STEP import. AI suggestions are a higher tier. They are not what makes $50 honest.

### 10.3 Accuracy sentence the subscription is allowed to use

After Phases A through C, the product can say:

> For a single closed section at \(10^5 \le \mathrm{Re}_c \le 3\times 10^6\) and angle of attack below the validated attach limit, lift is within 0.02 and drag within 15% of XFOIL’s \(e^N\) result. For an isolated wing in that same attached range, lift is within 5% of a vortex-lattice reference and drag includes strip skin friction plus Trefftz induced drag. Past separation, or above the Mach limit of the Prandtl–Glauert correction, the run returns no drag coefficient.

It cannot say “85–95% of Navier–Stokes” with no subject. The defensible version of that slogan is the sentence above, restricted to attached flow, and only after the booklet exists. Today the slogan is ahead of the solver.

### 10.4 Speed sentence the subscription is allowed to use

The spec’s own budgets are the right public copy, once a test enforces them:

- 2D inviscid cruise mesh, under 50 ms.
- 2D viscous cruise mesh, under 400 ms.
- Stored rectangular wing at density 1, under 5 s.

The subscription copy is those three lines plus the polar line in the table. “Real-time CFD” is not an accurate description of a 2 400-panel dense influence matrix, and it will become false the moment a user raises surface density.

### 10.5 What to ship before the first invoice

1. Phase A green, including the cruise case printing a finite positive \(C_D\) and a finite \(L/D\), and the \(10^\circ\) case refusing to print a negative \(C_D\).
2. 4412 moment sign fixed and tested.
3. Inviscid \(L/D\) removed from the UI until leakage is inside the A5 gate.
4. A polar button and a CSV.
5. The validation booklet for 0012 and 4412 against XFOIL, even if 3D viscous is still a warning.

Items 1–5 are a credible **paid beta** for section designers. The 3D viscous strip, Trefftz drag, matrix-free solver, and saved projects are what make the same price feel obvious rather than hopeful for a drone team. Auth and billing come after the beta has one external user who can reproduce the booklet.

### 10.6 Summary judgment

The repository is a serious start on the solver the price requires: contracts, immutability, a real influence solve, a real integral boundary layer, and tests that remember specific past failures (trailing-edge cluster, transition/separation ties, planform area under sweep, freestream dynamic pressure in the friction integral). Those are the habits of a numerical code that can get to a subscription.

The launch is about a third complete. The hole in the middle of the present product is concrete: the default viscous cruise solution of NACA 0012 computes a friction drag near 0.006 and then discards the glide ratio because the lower surface separates at 84% of arc length, while a 10° solution computes a negative drag. Closing that hole, and proving the result against XFOIL, is the engineering that makes $45–$60 a month a fair price. The SaaS shell does not make the price fair before that proof exists.
