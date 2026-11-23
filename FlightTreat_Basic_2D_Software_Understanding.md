# Basic 2D Software Understanding

This report answers the 15 questions from the code that is actually in this workspace. No existing file was changed.

**Name check.** Nothing in the source, tests, datasets, or README is called FlightTreat. The project that is here is a Python aerodynamic pipeline: geometry, surface mesh, a panel solver, force reduction, and a Streamlit workbench. The answers below are about that software. Where a feature is only named in a README and is not in the running code, the report says so.

This is **not** called CFD in this report. The solver does not discretize the Navier–Stokes equations on a volume grid. It is a steady panel method (potential flow on the surface) with a one-dimensional integral boundary layer.

---

## 1. Software purpose

### 1. What is it currently designed to do?

It takes a 2D section (or a 3D wing or body), builds surface panels, and estimates the steady aerodynamic loads on that shape.

For a 2D airfoil the live path is:

1. Build the outline (NACA 4-digit, a thin flat plate, or an imported SVG/DXF section).
2. Cut that outline into straight panels and mark stations for a boundary-layer march.
3. Solve for vortex strength on those panels so air does not pass through the wall, with a Kutta condition at the trailing edge.
4. Optionally march a boundary layer and feed a small wall-blowing velocity back into the panel solve.
5. Turn surface pressure and skin friction into lift, drag, and pitching moment, then plot them.

The same workbench can also run a 3D source/doublet panel solve. That path is inviscid in the code that runs today: the 3D solver never marches the boundary layer.

### 2. What engineering problem does it solve, and who is its intended user?

The engineering problem is: given a section shape, angle of attack, Reynolds number, and a low Mach number, estimate pressure, lift, drag, and moment without building a volume mesh or running a Navier–Stokes code.

The root README names drone startups, university researchers, and aerospace engineers as the audience. That is a product goal written in documentation. The program that exists is a local Streamlit workbench plus Python tests. There is no login, no database, no web API, and no cloud deployment in the repository. A student or developer running it on their own machine is the user the code actually supports.

### 3. Is it a calculator, an aerodynamic analysis tool, 2D simulation software, or something else?

It is a **steady aerodynamic analysis tool** built on a panel method.

- It is more than a pocket calculator: it discretizes a shape, solves a linear system, and draws the result.
- It is not a time-stepping 2D flow simulation. There is no time loop, no animation of unsteady flow, and no volume residual.
- The Cartesian grid around a 2D section is a sampling grid for arrows and a solid/fluid mask. It is not the mesh the equations are solved on.

---

## 2. Engineering calculations

### 4. What aerodynamic and engineering quantities can it calculate?

From the 2D solver and the aero reduction, a single angle produces:

| Quantity | Where it comes from |
|---|---|
| Panel vortex strength γ, and source strength σ when blowing or a source-vortex model is on | `solver_2d.py` |
| Pressure coefficient \(C_p\) on each panel | \(C_p = 1 - V_t^2\) at unit freestream speed |
| Lift coefficient \(C_L\) | Pressure plus skin friction, projected on the lift direction, divided by chord |
| Pressure drag \(C_{D,p}\), friction drag \(C_{D,f}\), and their sum \(C_D\) | Same force split |
| Pitching moment about the quarter-chord and about the panel centroid | Pressure only (see limitations) |
| Total circulation \(\Gamma = \sum \gamma \, S\) | Integrated vortex strength |
| Boundary-layer momentum thickness θ, displacement thickness δ\*, shape factor H, skin friction \(C_f\), boundary-layer thickness δ | Thwaites / Head march, 2D viscous regime only |
| Transition and separation arc-length stations | Michel criterion and separation checks |
| Off-body velocity, potential, and streamfunction on the Cartesian grid | Reconstructed from the same panels; cells inside the body are NaN |
| Lift-to-drag ratio, normal and axial force, center of pressure \(x_{cp}/c\) | `aero_module` |
| Lift, drag, and moment in newtons | \(q_\infty\) times the coefficients. The workbench fixes \(V_\infty = 1\) and \(\rho = 1.225\), so these newtons are for that unit onset unless a caller changes speed and density |
| If several already-solved angles are passed in | Lift-curve slope, zero-lift angle, aerodynamic-center estimate, and a \(C_D = C_{D0} + K C_L^2\) fit |

The workbench itself solves **one** angle at a time. The polar fit exists in `aero_module/polar.py` and is covered by a unit test. The Streamlit page does not sweep angle of attack for the user.

3D runs add surface \(C_p\), inviscid \(C_L\) and pressure \(C_D\), and a spanwise lift binning. Friction drag on 3D is left as NaN because the boundary layer is not marched.

### 5. Which mathematical formulas and physical equations does it use?

**Inviscid 2D panel problem.** Constant-strength vortex panels. The boundary condition is a Dirichlet condition on the streamfunction, not a raw normal-velocity matrix. The code comments say the midpoint Neumann matrix for a constant vortex has a checkerboard null space, so the solver uses the streamfunction instead (`solver_2d.py`).

At each collocation point the vortex streamfunction plus the freestream streamfunction is a constant. The last unknown is that constant. Freestream direction is \((\cos\alpha,\ \sin\alpha)\) at unit speed.

**Kutta condition** when it is turned on (the default for a closed 2D section): the two trailing-edge panel strengths sum to zero,

\[
\gamma_a + \gamma_b = 0.
\]

On a blunt trailing face the pair is the aft-most upper panel and the aft-most lower panel, not whatever two panels merely share the trailing-edge index.

**Pressure.** With freestream speed set to 1,

\[
C_p = 1 - V_t^2.
\]

**Prandtl–Glauert**, used when \(0 < M < 0.7\) and the switch is on. The chord is stretched by

\[
\beta = \sqrt{1 - M^2},
\]

the panel problem is solved on the stretched shape, and \(C_p\) is then divided by \(\beta\). Mach ≥ 1 is rejected. Mach ≥ 0.7 with Prandtl–Glauert on is also rejected.

**Viscous wall blowing.** Displacement thickness and edge speed give a transpiration velocity

\[
v_n = \frac{d}{ds}(U_e \delta^*),
\]

clipped to \(\pm 0.05\), then under-relaxed by 25% each coupling pass. That blowing is carried by a source sheet. The vortex strengths are not asked to enforce it.

**Thwaites laminar boundary layer** (`boundary_layer_1d.py`):

\[
\theta^2 = \frac{0.45\,\nu}{U_e^6}\int U_e^5\,ds,
\quad
\lambda = \frac{\theta^2}{\nu}\frac{dU_e}{ds}.
\]

λ is clipped to \([-0.09,\ 0.25]\). The shape factor and skin-friction parameter are the usual quadratic fits

\[
H = 2.61 - 3.75\lambda + 5.24\lambda^2,
\quad
\ell = 0.22 + 1.57\lambda - 1.8\lambda^2,
\quad
C_f = \frac{2\ell}{Re_\theta},
\]

with \(C_f\) clipped to \([0,\ 0.04]\). Laminar separation is flagged at \(\lambda \le -0.09\).

**Michel transition.** The first station where

\[
Re_\theta \ge 1.174\left(1 + \frac{22400}{Re_s}\right) Re_s^{0.46}.
\]

**Head turbulent closure.** An explicit march of momentum thickness and Head’s entrainment shape factor \(H_1(H)\), with Ludwieg–Tillmann-style skin friction

\[
C_f = 0.246 \cdot 10^{-0.678 H} \, Re_\theta^{-0.268}.
\]

Turbulent separation is flagged when \(H \ge 2.8\) or \(C_f \le 0\). Downstream of a laminar separation the code freezes the profile and sets \(C_f = 0\). A direct wake is marched from the trailing edge and its Squire–Young defect is stored as `cd_momentum`. That defect is not added to the published drag.

**Forces.** Panel pressure force is \(-\,C_p\,\mathbf{n}\,S\). Skin friction is \(C_f (U_e/V_\infty)^2\) along the local tangent. Lift and drag are the projections of those forces onto axes rotated by α, divided by chord. Kutta–Joukowski at unit speed and unit chord is the check \(C_L \approx 2\Gamma\).

**What is named but not the default.** `drela_two_equation` is an equilibrium two-equation turbulent march: the momentum and kinetic-energy equations with Drela's incompressible closures, and no shear-stress lag. The default march remains Thwaites / Michel / Head. Viscous coupling by moving the wall (`displacement`) still raises an error. The aero notes set `stall_model` to `"none"`. There is no stall equation.

### 6. What methods or numerical techniques does it use?

- Straight constant-strength vortex panels, with an optional constant-strength source sheet.
- Influence integrals in the panel frame (endpoint radii and turning angles), the standard Katz & Plotkin geometric integrals (`influence_2d.py`).
- One dense linear solve per pass: `scipy.linalg.solve`. The matrix is rejected if its 1-norm condition number is above \(10^{12}\).
- Cosine or half-cosine clustering of panel nodes comes from the mesh module, before the solver runs. The solver does not remesh.
- Viscous-inviscid iteration: solve panels, march the boundary layer, update blowing, repeat. Default cap is 8 passes. Convergence is the largest change in δ\*, divided by chord, below \(10^{-4}\).
- If \(|\gamma|\) exceeds 8 or \(C_p\) drops below −20 during coupling, the code throws the blowing away and returns the inviscid pressure, with a warning.
- 3D, for completeness: constant source and doublet panels, a Morino interior condition on closed bodies, a wake sheet on open wings, and GMRES only when the panel count exceeds 2000. That path does not run the boundary-layer iteration.

Almost all of the 2D kernels are NumPy array operations. The Head march is an explicit station-by-station loop, because each station depends on the previous one.

### 7. How does the software process input and perform the calculation?

A simple pass for a NACA section:

1. **Geometry.** A 4-digit code, chord, and point count become a closed polyline. Leading edge and trailing edge are identified.
2. **Mesh.** Those points become panel length, midpoint, tangent, and outward normal. The same contour is resampled into upper and lower arc-length stations for the boundary layer. A Cartesian box is tagged so cells inside the airfoil can be blanked later.
3. **Flow inputs.** Angle of attack, Reynolds number, Mach, speed, and density. The workbench sliders cover angle (about −12° to 16°), \(\log_{10} Re\) from 4 to 7, and Mach from 0 to 0.65. Speed and density stay at 1.0 and 1.225 on that page. Regime is `inviscid` or `viscous_inviscid`.
4. **Copy.** The solver copies every mesh array before it works. It does not write back into the mesh.
5. **Inviscid solve.** Build the streamfunction matrix, add the Kutta row, solve for γ. If there is wall blowing, solve a second system for σ.
6. **Boundary layer, if requested.** Kinematic viscosity is \(\nu = c / Re_c\). March both sides from the leading-edge region. Couple by transpiration until the δ\* change is small or the iteration cap is hit.
7. **Loads and field.** Integrate forces. Sample velocity on the Cartesian grid from the same panels and add the freestream. Blank the interior.
8. **Report.** The aero module copies \(C_p\) and the loads, applies the reference chord and area, and computes \(L/D\) and the center of pressure. It does not solve the flow again.

Pipeline cases in `datasets/pipeline/` run this same chain through `run_pipeline_case` for stored shapes such as NACA 0012 and 4412.

---

## 3. Technology

### 8. Which languages, libraries, frameworks, and technologies does it use?

Everything that is installed and imported for the live path is Python.

From `requirements.txt`:

- NumPy
- SciPy
- Pydantic
- Streamlit
- Plotly
- pandas (imported by the workbench; the solver itself does not depend on it for the math)

Tests are plain Python assert scripts under `tests/`. There is no FastAPI app, database, authentication package, or Docker file in this tree, even though the architecture notes list those as later modules.

### 9. What is the role of each major technology?

| Technology | Role in this software |
|---|---|
| Python | All geometry, mesh, solver, report, and UI code |
| NumPy | Panel coordinates, influence matrices, \(C_p\), force sums. Arrays are `float64` |
| SciPy | Dense linear solve, cumulative trapezoid for Thwaites, nearest-panel lookup for blowing, 3D GMRES |
| Pydantic | Checks the data contracts (shapes, finite numbers, allowed regime names) at the module boundaries |
| Streamlit | Local workbench: sidebar inputs, metrics, warnings, download buttons |
| Plotly | 2D curves, panel plots, velocity arrows, and the 3D \(C_p\) coloring |
| pandas | Small tables in the workbench (mesh quality, boundary-mask counts) |

---

## 4. Output and visualization

### 10. What outputs does the software generate?

**Numbers**

- \(C_L\), \(C_D\), \(C_{D,p}\), \(C_{D,f}\), \(C_m\), \(L/D\), \(x_{cp}/c\)
- Iteration count and the δ\* residual
- Transition and separation stations
- Warnings when the coupling did not converge, when the layer separated, or when laminar separation made the drag untrustworthy

**Plots in the workbench**

- Airfoil or wing geometry
- Panel polylines and 3D surface normals
- Farfield boundary-mask plot (not a pressure heatmap)
- Boundary-layer station locations
- \(C_p\) versus \(x/c\), upper and lower, with the vertical axis reversed
- \(C_f\), δ\*, θ, and δ along arc length
- A thinned field of velocity arrows around the section
- On 3D shapes, the surface colored by \(C_p\)

**Files**

- JSON summary of the geometry, mesh, physics, and aero contracts
- A NumPy `.npz` of the same arrays

There is **no** simulation animation, no contour heatmap of the volume pressure, and no automatic PDF report.

### 11. What can a user currently see and interact with?

Run `streamlit run src/streamlit_ui/app.py`.

The sidebar lets the user:

- Upload an SVG/DXF section or an STL/OBJ body, build a NACA section, flat plate, wing, or cylinder, or load a stored pipeline case or mesh sample
- Change panel count, leading-edge and trailing-edge clustering, and the farfield box
- Change angle of attack, Reynolds number, Mach, and inviscid versus viscous-inviscid
- Download the JSON and `.npz` contracts

The page then shows the geometry plot, mesh tabs, the coefficient line, and the flow tabs listed above. Moving a mesh slider rebuilds the mesh and re-solves. Errors from a bad shape or a singular matrix stay on the page as a message instead of a raw crash, when the solver raises its own exception types.

This is a developer workbench. It is not a finished engineering application with projects, accounts, or a written customer report.

### 12. Is it actually performing a 2D aerodynamic simulation, or mainly calculating and visualizing quantities?

It **calculates** a steady 2D potential-flow solution on the surface, optionally corrects it with a 1D boundary layer, and **visualizes** those quantities.

That is a real numerical solve: a linear system for γ (and σ), then a march in arc length. It is not a 2D simulation in the sense of time-evolving Navier–Stokes flow on a grid around the airfoil. The arrows on the Cartesian grid are reconstructed afterward from the panel strengths. They are a picture of the panel solution, not an independent flow solve.

---

## 5. Accuracy and reliability

### 13. What accuracy is expected, and what can actually be verified?

The root README states a target of about 85–95% on \(C_L\), \(C_p\), and \(C_D\) versus a viscous truth. **That number is not measured by the tests.** No test compares a result to a wind-tunnel polar, an XFOIL table, or a published \(C_p\) distribution and then reports a percentage error.

What the tests do check, in `tests/test_physics_module.py`:

- Thin flat plate at 4°: \(C_L\) within **0.08** of thin-airfoil theory \(2\pi\alpha\) (about 0.44). A band of ±0.08 is roughly ±18% of that target. The test proves the lift is in a wide neighborhood of theory, not that it matches to a percent.
- NACA 0012 at 0°: \(|C_L| < 0.02\), and upper/lower \(C_p\) agree within 0.08 after mirroring.
- NACA 0012 at 4°, inviscid, 80 panels: \(C_L\) within **0.08** of 0.48, and \(C_L\) agrees with \(2\Gamma\) within 0.05.
- A viscous 4° cruise case keeps aft upper-surface skin friction from being wiped to zero, and the shape factor stays in a band that means “the turbulent march ran and stayed attached.” It does **not** check \(C_D\) against data.
- A high-incidence case is only required to separate, stay finite, and warn.
- A rectangular NACA 0012 wing is checked against a Helmbold estimate within 0.08. That is a 3D inviscid check.

So: attached, low-angle **lift** is in the right neighborhood of thin-airfoil and a round NACA target, on the meshes the tests use. **Drag, moment, transition location, and separated \(C_L\) are not validated against experiment in this repository.** The code itself refuses to treat laminar-separation drag as a validated number: it warns, and it sets \(L/D\) to NaN in that case.

### 14. What are the major limitations, and what can make results inaccurate?

- **Potential flow at heart.** The inviscid part cannot predict stall, shock waves, or large separation. Mach must stay below 1, and the compressibility correction is refused at Mach ≥ 0.7.
- **No stall model.** `stall_model` is the string `"none"`. Past stall, \(C_L\) will keep following the panel solution rather than a measured maximum lift.
- **Separated drag is incomplete.** Laminar separation sets downstream \(C_f\) to 0 and adds no wake drag. The warning text says \(C_D\) and \(L/D\) are then not a validated drag result.
- **Coupling can give up.** Blowing is clipped and heavily under-relaxed. A non-physical spike resets the pressure to the inviscid field and marks the iteration as not converged. A user can still be shown a number after that reset.
- **Coarse checks.** Test tolerances of 0.08 on \(C_L\) would accept a noticeably wrong lift.
- **Panel resolution and trailing-edge clustering.** Very short panels near the trailing edge are not a resolved edge velocity. The march holds their speed at the last resolved station, and the streamwise derivative is formed only on the resolved stations. A central difference that reached into that cluster was reading the panel-speed fall and feeding it back as wall suction. A coarse or badly clustered mesh still changes \(C_p\) and transition.
- **Moment ignores skin friction.** `forces_and_moments.py` documents a friction moment, but the 2D moment function returns only the pressure moment.
- **3D is not viscous.** Even if the regime says viscous-inviscid, `_solve_3d` never calls the boundary-layer march. Friction drag is NaN and \(C_D\) is pressure drag only.
- **Single operating point in the UI.** A drag polar, aerodynamic center, and maximum \(L/D\) need several angles. The fitter can do that only if some other caller supplies the solves. The workbench does not.
- **Unit freestream in the UI.** Dimensional newtons use \(V = 1\) and \(\rho = 1.225\) unless code outside the sidebar changes them. Coefficients are the meaningful output.
- **Not a closed production tool.** No accounts, saved projects, API, or deployment. A singular matrix or a separated result is a warning or an exception, not a certified engineering deliverable.

---

## 6. Current readiness

### 15. How ready is it as usable 2D engineering software? What are the three most important missing capabilities?

It is a **working student and developer prototype** for steady 2D panel analysis.

A user can generate a NACA section, solve attached flow at a modest angle, read \(C_L\) and \(C_p\), and see a boundary-layer plot. That path runs, and the tests lock the obvious symmetries and a loose lift check. It is not ready to be handed to someone as the tool that sizes a wing or certifies drag. Drag and stall are the weak part, the accuracy target in the README is not evidenced, and the interface is a pipeline workbench rather than an engineering product.

The three most important missing capabilities:

1. **A published-data check of \(C_p\), \(C_L\), and \(C_D\).** Compare NACA 0012 (and at least one cambered section) at known Reynolds numbers with Abbott & von Doenhoff or an accepted panel/IBL code, and record the error. Until that exists, the 85–95% claim cannot be used.
2. **A drag model that stays honest through transition and mild separation.** Wake drag, a real displacement coupling (the `displacement` option is rejected today), and a stall limit. Today laminar separation zeros skin friction and drops \(L/D\).
3. **An angle sweep the user can run.** One button that solves a small α range and plots \(C_L(\alpha)\), \(C_D(C_L)\), and the aerodynamic center. The fitting code is already in `polar.py`. The workbench never calls it.

---

## Final summary

**Software purpose.** This workspace holds a local Python tool that estimates steady airfoil and wing loads with a surface panel method and, in 2D, a simple integral boundary layer. It is not named FlightTreat anywhere in the tree. It is aimed at fast section and wing estimates for students and engineers, but what runs today is a workbench on one machine.

**Current capabilities.** It builds NACA sections, flat plates, simple wings, cylinders, and a few imported file types. It solves 2D vortex panels with a Kutta condition, optionally couples Thwaites–Michel–Head, reconstructs a velocity picture around the section, and reports \(C_L\), \(C_D\), \(C_m\), and \(C_p\). 3D solves are inviscid panel solves only.

**Calculation methods.** Streamfunction panel equation plus \(\gamma_a + \gamma_b = 0\). \(C_p = 1 - V_t^2\), with a Prandtl–Glauert \(\beta\) correction below Mach 0.7. Thwaites laminar, Michel transition, Head turbulent, coupled by \(d(U_e\delta^*)/ds\). Forces are panel pressure and skin friction projected into lift and drag. The linear system is a dense direct solve.

**Technology stack.** Python, NumPy, SciPy, Pydantic, Streamlit, and Plotly.

**Outputs.** Coefficient numbers, \(C_p\) and boundary-layer curves, velocity arrows, a 3D \(C_p\) coloring, and downloadable JSON and `.npz` contracts. No flow animation and no volume heatmap.

**Accuracy.** Low-angle lift sits inside a wide test band around thin-airfoil theory and a 0.48 target for NACA 0012 at 4°. Symmetry at zero incidence is checked. Drag, stall, and a percentage agreement with experiment are not verified. The README’s 85–95% figure is a goal, not a result from this code.

**2D readiness.** Usable as a learning and development prototype for attached, low-speed, steady section flow. Not ready as engineering software whose drag or stall numbers would be used for a design decision.

**Top 3 improvements.**

1. Validate \(C_p\), \(C_L\), and \(C_D\) against published airfoil data and write down the error.
2. Replace the laminar-separation freeze with a drag model that includes the wake, and add a stall limit.
3. Expose an angle-of-attack sweep and polar plot in the workbench, using the fitter that already exists.
