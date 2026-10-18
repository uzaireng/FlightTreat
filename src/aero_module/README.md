# Aero Analytics (`aero_module`)

`AeroEngine` reduces a finished `PhysicsOutput`. The formulas live in `aero_skeleton.py`. The engine does not solve the flow.

Module 4 is the product reduction in front of the panel solver. Module 3 integrates pressure and skin friction. Module 4 turns that one payload into report coefficients, newtons, \(L/D\), and, when several angles are already solved, a polar and an aerodynamic center. It does not build \(A \mu = b\), march the boundary layer, or remesh.

## Input

`analyze` takes one `AeroInput`:

| Field | Source | Role |
|---|---|---|
| `mesh` | Module 2 | Chord, span, panel count, and \(x/c\). Arrays are read and copied. |
| `physics` | Module 3 | `loads`, surface `Cp`, freestream, regime, warnings |
| `parameters` | Caller | Optional `s_ref`, `c_ref`, `b_ref`, `moment_origin`, induced-drag switch |

`reduce_polar` takes the same mesh and parameters plus `samples`: a list of `PhysicsOutput` at different angles. The caller runs Module 3 once per angle. Module 4 only stacks the results.

Defaults when a parameter is omitted: area and chord stay on the physics reference, the moment stays at the quarter chord, induced drag is not split, and no stall correction is applied.

## Output

One angle (`AeroOutput`):

| Field | Meaning |
|---|---|
| `coefficients.cl`, `cd`, `cd_pressure`, `cd_friction` | Wind-axis coefficients on the chosen reference |
| `coefficients.cd_induced` | 0 on a 2D section. NaN in 3D unless the lifting-line switch is on |
| `coefficients.cm` | Pitching moment about the requested origin |
| `coefficients.cm_quarter_chord` | Moment about the quarter chord, used for the aerodynamic center |
| `coefficients.cl_inviscid`, `cd_inviscid` | Filled only when the physics regime is inviscid. NaN on a viscous solve |
| `coefficients.cn`, `ca` | Chord-normal and chord-axial force coefficients |
| `coefficients.center_of_pressure_x` | \(x_{cp}/c\) from one angle. NaN when \(C_L\) is near zero |
| `dimensional` | Lift, drag, and pitching moment at \(q = \tfrac12 \rho V^2\). 2D is per unit span |
| `l_over_d` | \(C_L / C_D\). NaN when drag is below \(10^{-6}\) |
| `cp_surface`, `x_over_c`, `pressure` | Copied pressure, plus minimum and maximum \(C_p\) |
| `viscous` | Transition and separation stations copied from the 2D boundary layer. Empty in 3D |
| `spanwise` | Section \(c_\ell(y)\) on a 3D wing. Absent on a 2D section |
| `aerodynamic_center_x`, `static_margin` | NaN for a single angle. Margin needs `cg_x_over_c` as well |
| `warnings` | Physics warnings, plus any reference or drag-split note |

A polar (`AeroPolarOutput`) adds \(C_L(\alpha)\), \(C_D(\alpha)\), \(C_M(\alpha)\), \(L/D(\alpha)\), \(c_{\ell_\alpha}\), \(c_{m_\alpha}\), \(x_{ac}/c\), the best sampled \(L/D\), and the largest sampled \(C_L\). It also reports zero-lift angle, \(C_{M0}\), a \(C_D = C_{D0} + K C_L^2\) fit, Oswald efficiency from that fit on a 3D wing, and the fitted lift slope divided by the ideal slope (\(2\pi\) in 2D, \(2\pi \mathrm{AR}/(\mathrm{AR}+2)\) in 3D).

## Calculations

Dynamic pressure is \(q = \tfrac12 \rho V_\infty^2\) from the freestream Module 3 stored.

Coefficients are rescaled, not re-integrated. If the caller changes area or chord:

\[
C_L = C_{L,\mathrm{phys}} \frac{S_\mathrm{phys}}{S}, \quad
C_D = C_{D,\mathrm{phys}} \frac{S_\mathrm{phys}}{S}
\]

2D moment scales with \((c_\mathrm{phys}/c)^2\). 3D moment scales with \((S_\mathrm{phys}\, c_\mathrm{phys})/(S\, c)\). In 2D the area and the chord are the same length. A shifted moment origin moves the quarter-chord moment by the force times the offset. The panels are not touched.

2D induced drag is 0. 3D pressure drag is left whole unless `induced_drag_model` is `lifting_line`, in which case

\[
C_{D,i} = \frac{C_L^2}{\pi\, \mathrm{AR}\, e}, \quad \mathrm{AR} = b^2/S
\]

with \(e = 0.85\) unless the caller sets `oswald_efficiency`. That estimate is reported next to \(C_D\). It is not removed from the integrated drag.

The aerodynamic center needs two or more angles:

\[
\frac{x_{ac}}{c} = 0.25 - \frac{c_{m_\alpha}}{c_{\ell_\alpha}}
\]

using moments about the quarter chord. One angle cannot identify it.

\((L/D)_\max\) and \(C_{L,\max}\) on a polar are the best and largest samples. They are not a stall model.

## Left open on purpose

These stay out of the contract until a formula is chosen. `stall_model` accepts only `"none"`, so a stall drop cannot appear by accident.

- An empirical stall model: \(C_{L,\max}\), the lift drop after it, and the drag spike.
- A real induced-drag integral (Trefftz or wake). The lifting-line number is only an estimate, and it is off by default. The polar fit reports a separate Oswald efficiency. It does not replace integrated drag.
- A paired inviscid solution beside a viscous one. That would be a second Module 3 solve, requested by the pipeline, not a hidden solve inside Module 4.
- Mean aerodynamic chord as a replacement for the mid-span x-extent used when 3D `c_ref` is omitted.
- Standard-atmosphere lookup, file import, the linear solver, chart formats, and PDF export. Those are not aerodynamic reductions.

Add those requirements here, then the engine can implement the combined list without changing the field names above.

## Usage

```python
from aero_module import AeroEngine

report = AeroEngine().analyze_from_solution(mesh, physics)
polar = AeroEngine().reduce_polar({"mesh": mesh, "samples": [physics_a, physics_b], "parameters": {}})
```

Pipeline cases call `analyze_from_solution` through `run_pipeline_case`. `aero_overrides` merges onto `stages.aero` for that run only.

## Files

| File | Role |
|---|---|
| `aero_engine.py` | `analyze` and `reduce_polar` |
| `reference.py` | Area, chord, span, dynamic pressure |
| `operate.py` | One-angle coefficients, newtons, Cp copy, spanwise bins |
| `polar.py` | Sorted curves and the slope fits |
| `aero_skeleton.py` | Contract only |

## Runtime

`validate_aero_parameters` checks the knobs. One angle is a pass over the panels already solved. A polar fits a handful of those points. There is no new linear system.
