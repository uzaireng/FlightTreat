# NACA 0012 aerodynamic accuracy validation

This report compares the 2D viscous-inviscid panel solver with published NACA 0012 data. The reference files were not edited for the comparison.

Error is FlightTreat minus the reference. Relative error is that difference divided by the absolute reference value. Relative error is left blank when the reference is within 0.05 of zero for lift, 0.20 for pressure, or 0.002 for drag, because a tiny denominator is not a useful percentage.

## Reference sources and operating conditions

The tables live in `datasets/modules/physics/naca0012_reference/`. They were copied from the NASA Turbulence Modeling Resource NACA 0012 validation page on 6 November 2026.

| Quantity | Source | Conditions used | Why this source |
|---|---|---|---|
| Lift coefficient, CL | Ladson, NASA TM 4074 (1988), file `CLCD_Ladson_expdata.dat` | Re = 6×10⁶, M = 0.15, 120-grit trip. FlightTreat is run at those Re and M with Prandtl–Glauert on. | A force table, not a digitised plot. NASA uses it for forces at this Reynolds number. Lift is taken from it. Drag is not, because the grit trip is not this solver's boundary layer. |
| Lift-curve slope | McCroskey, AGARD CP-429 / NASA TM 100019 (1987), file `0012.mccroskeydata.cl.dat` | The file's fit at Re = 6×10⁶, M = 0.15 | A published fit across several tunnels. It is a slope check, not a second polar. |
| Pressure coefficient, Cp | Gregory and O'Reilly, R&M 3726 (1970), file `CP_Gregory_expdata.dat` | Re = 2.88×10⁶, M = 0.15, upper surface, α = 0°, 10°, 15°. The file does not record Mach. M = 0.15 is inside the 0.08–0.16 range McCroskey lists for this test. | NASA recommends this distribution over the Ladson, Hill, and Johnson pressures because the leading-edge peak is better resolved. |
| Drag coefficient, CD | Abbott and von Doenhoff, *Theory of Wing Sections* (1959), file `0012.abbottdata.cd.dat` | Re = 6×10⁶, natural transition. Compared at the lift FlightTreat actually produced. | Same section, same Reynolds number, and natural transition, which is the class this solver runs. |

Ladson drag is not a validation target. That test fixed transition with grit. This solver transitions on its own with Michel's criterion and cannot place a trip. The two drag conditions do not match, so a Ladson CD error is not reported as an accuracy result.

Ladson, Hill, and Johnson Cp (NASA TM 100526) is also not used. It was measured at Mach 0.3 on a model of aspect ratio 1.333, and the NASA page says the leading-edge suction peak is poorly resolved.

## Solver setup

NACA 0012, chord 1, closed trailing edge (the generator's thickness coefficient is 0.1036 so the trailing edge meets). Maximum thickness on this mesh is 0.1200 chords. 120 cosine-clustered panels, leading-edge and trailing-edge cluster factors 1, which is the workbench default. Regime `viscous_inviscid`. Freestream speed 1 and density 1.225. The 8×8 Cartesian farfield is built because the solver requires it; surface Cp and the forces do not read that grid.

The wind-tunnel models are the ordinary blunt-trailing-edge NACA 0012. NASA's own CFD grids use a slightly thinned sharp-trailing-edge scaling. That geometry difference is small next to the lift differences below and is not offered as the cause of them.

## Lift

120-grit Ladson stations. The grit span is the lowest and highest CL among the 80, 120, and 180 grit rows within 0.2° of that angle. Alpha = 0 has no Ladson row; the nearby tripped measurement at 0.01° is CL = -0.0122.

| α (deg) | CL reference | CL FlightTreat | Absolute error | Relative error | Grit span | Converged | Separated |
|---:|---:|---:|---:|---:|---|---|---|
| -4.01 | -0.4466 | -0.4464 | 0.0002 | 0.1% | -0.4466 to -0.4363 | yes | no |
| -2.12 | -0.2425 | -0.2364 | 0.0061 | 2.5% | -0.2425 to -0.2213 | yes | no |
| 0.00 | — | -0.0008 | — | — | -0.0126 to -0.0115 | yes | no |
| 0.01 | -0.0122 | 0.0004 | 0.0126 | — | -0.0126 to -0.0013 | yes | no |
| 2.15 | 0.2236 | 0.2381 | 0.0145 | 6.5% | 0.2125 to 0.2236 | yes | no |
| 4.11 | 0.4397 | 0.4573 | 0.0176 | 4.0% | 0.4316 to 0.4397 | yes | no |
| 6.01 | 0.6487 | 0.6739 | 0.0252 | 3.9% | 0.6487 to 0.6558 | yes | no |
| 8.08 | 0.8701 | 0.9178 | 0.0477 | 5.5% | 0.8689 to 0.8701 | yes | yes |
| 10.10 | 1.0775 | 1.1543 | 0.0768 | 7.1% | 1.0707 to 1.0809 | yes | yes |
| 12.13 | 1.2720 | 1.3898 | 0.1178 | 9.3% | 1.2605 to 1.2720 | yes | yes |
| 15.27 | 1.5280 | 1.7473 | 0.2193 | 14.4% | 1.5129 to 1.5280 | no | yes |
| 18.18 | 1.1886 | 2.0963 | 0.9077 | 76.4% | 0.9967 to 1.1886 | no | yes |

From 0.01° to 6.01° the FlightTreat lift slope is **0.1122 per degree**. The McCroskey fit at this Reynolds number and Mach number is **0.1075 per degree**. Absolute slope error 0.0048 per degree, relative error 4.4%.

The same 4.11° case shows where the extra lift comes from.

| Case at 4.11° | CL |
|---|---:|
| Ladson 120-grit | 0.4397 |
| Viscous-inviscid, M = 0.15 | 0.4573 |
| Viscous-inviscid, M = 0 | 0.4554 |
| Inviscid, M = 0.15 | 0.4980 |

At 4.11° the viscous-inviscid CL at M = 0.15 is 0.4573. Turning Mach off leaves CL at 0.4554. The inviscid solution at M = 0.15 is 0.4980. Viscous coupling changes CL by -0.0406 from that inviscid solution. Prandtl–Glauert changes the viscous CL by +0.0020.

Through 6° the run is converged and the boundary layer stays attached. CL is high by 0.0145 at 2.15° (6.5% of the measured lift) and by 0.0252 at 6.01° (3.9%). The grit-to-grit CL span on those attached stations is at most 0.0111, so the miss is larger than the trip scatter. The relative errors at -4.01° and -2.12° are 0.1% and 2.5%. From 8° upward the solver reports lower-surface laminar separation, and by 18° the experiment has stalled (CL 1.1886) while the panel solution is still at 2.0963.

## Pressure

Upper-surface Cp from Gregory and O'Reilly. FlightTreat Cp is interpolated in x/c onto each experimental station that lies inside the panel x-range. Stations outside that range are dropped rather than extrapolated. The file says the points were digitised from a photocopy and are only approximate. A mid-chord band, 0.05 ≤ x/c ≤ 0.90, is summarised beside the full set so one leading-edge point does not hide the rest.

### α = 0°

Re = 2.88e+06, M = 0.15. CL = 0.0003, CD = 0.00636. Converged: yes. Separated: no. Stations compared: 23. Dropped: 2.

| | Mean absolute error | Maximum absolute error |
|---|---:|---:|
| All compared stations | 0.032 | 0.208 |
| 0.05 ≤ x/c ≤ 0.90 | 0.010 | 0.025 |

| x/c | Cp reference | Cp FlightTreat | Absolute error | Relative error |
|---:|---:|---:|---:|---:|
| 0.0023 | 0.848 | 0.657 | -0.191 | -22.5% |
| 0.0050 | 0.456 | 0.405 | -0.051 | -11.2% |
| 0.0053 | 0.174 | 0.381 | 0.208 | — |
| 0.0142 | -0.044 | -0.027 | 0.018 | — |
| 0.0209 | -0.175 | -0.159 | 0.016 | — |
| 0.0474 | -0.373 | -0.347 | 0.026 | 6.9% |
| 0.0779 | -0.396 | -0.403 | -0.007 | -1.8% |
| 0.0976 | -0.419 | -0.414 | 0.005 | 1.2% |
| 0.1282 | -0.419 | -0.416 | 0.003 | 0.7% |
| 0.1500 | -0.411 | -0.412 | -0.001 | -0.3% |
| 0.1784 | -0.403 | -0.404 | -0.001 | -0.2% |
| 0.2897 | -0.367 | -0.350 | 0.016 | 4.4% |
| 0.3224 | -0.347 | -0.334 | 0.013 | 3.7% |
| 0.3879 | -0.308 | -0.282 | 0.025 | 8.3% |
| 0.4490 | -0.268 | -0.254 | 0.014 | 5.4% |
| 0.5144 | -0.229 | -0.212 | 0.017 | 7.3% |
| 0.5799 | -0.190 | -0.177 | 0.013 | — |
| 0.6388 | -0.159 | -0.144 | 0.015 | — |
| 0.7043 | -0.115 | -0.105 | 0.009 | — |
| 0.7676 | -0.065 | -0.065 | 0.000 | — |
| 0.8352 | -0.026 | -0.015 | 0.012 | — |
| 0.8963 | 0.035 | 0.045 | 0.010 | — |
| 0.9595 | 0.098 | 0.151 | 0.053 | — |

### α = 10°

Re = 2.88e+06, M = 0.15. CL = 1.1369, CD = 0.01194. Converged: yes. Separated: yes. Stations compared: 22. Dropped: 2.

| | Mean absolute error | Maximum absolute error |
|---|---:|---:|
| All compared stations | 0.076 | 0.560 |
| 0.05 ≤ x/c ≤ 0.90 | 0.043 | 0.092 |

| x/c | Cp reference | Cp FlightTreat | Absolute error | Relative error |
|---:|---:|---:|---:|---:|
| 0.0022 | -5.044 | -5.604 | -0.560 | -11.1% |
| 0.0087 | -5.241 | -5.084 | 0.156 | 3.0% |
| 0.0131 | -4.671 | -4.653 | 0.018 | 0.4% |
| 0.0175 | -4.321 | -4.275 | 0.046 | 1.1% |
| 0.0480 | -2.743 | -2.885 | -0.141 | -5.1% |
| 0.0742 | -2.261 | -2.353 | -0.092 | -4.1% |
| 0.0983 | -1.954 | -2.045 | -0.090 | -4.6% |
| 0.1245 | -1.734 | -1.810 | -0.075 | -4.4% |
| 0.1463 | -1.559 | -1.637 | -0.078 | -5.0% |
| 0.1769 | -1.361 | -1.452 | -0.090 | -6.6% |
| 0.2882 | -1.008 | -1.047 | -0.038 | -3.8% |
| 0.3210 | -0.942 | -0.957 | -0.015 | -1.6% |
| 0.3843 | -0.787 | -0.806 | -0.019 | -2.4% |
| 0.4476 | -0.654 | -0.684 | -0.030 | -4.5% |
| 0.5153 | -0.543 | -0.555 | -0.012 | -2.1% |
| 0.5764 | -0.433 | -0.463 | -0.031 | -7.1% |
| 0.6376 | -0.344 | -0.375 | -0.031 | -9.1% |
| 0.7009 | -0.255 | -0.287 | -0.033 | -12.8% |
| 0.7664 | -0.166 | -0.197 | -0.031 | — |
| 0.8319 | -0.099 | -0.099 | -0.001 | — |
| 0.8930 | -0.010 | 0.007 | 0.017 | — |
| 0.9585 | 0.079 | 0.146 | 0.066 | — |

### α = 15°

Re = 2.88e+06, M = 0.15. CL = 1.7195, CD = 0.02028. Converged: no. Separated: yes. Stations compared: 22. Dropped: 2.

| | Mean absolute error | Maximum absolute error |
|---|---:|---:|
| All compared stations | 0.372 | 1.446 |
| 0.05 ≤ x/c ≤ 0.90 | 0.176 | 0.389 |

| x/c | Cp reference | Cp FlightTreat | Absolute error | Relative error |
|---:|---:|---:|---:|---:|
| 0.0024 | -10.179 | -11.625 | -1.446 | -14.2% |
| 0.0045 | -9.720 | -10.840 | -1.120 | -11.5% |
| 0.0087 | -9.043 | -9.494 | -0.451 | -5.0% |
| 0.0130 | -8.672 | -8.391 | 0.281 | 3.2% |
| 0.0168 | -6.161 | -7.580 | -1.419 | -23.0% |
| 0.0467 | -3.998 | -4.665 | -0.667 | -16.7% |
| 0.0770 | -3.167 | -3.517 | -0.350 | -11.1% |
| 0.0965 | -2.686 | -3.075 | -0.389 | -14.5% |
| 0.1463 | -2.050 | -2.351 | -0.301 | -14.7% |
| 0.1745 | -1.831 | -2.070 | -0.240 | -13.1% |
| 0.2874 | -1.236 | -1.417 | -0.181 | -14.6% |
| 0.3179 | -1.126 | -1.292 | -0.166 | -14.8% |
| 0.3809 | -0.927 | -1.074 | -0.147 | -15.9% |
| 0.4439 | -0.727 | -0.900 | -0.173 | -23.8% |
| 0.5090 | -0.593 | -0.729 | -0.136 | -22.9% |
| 0.5764 | -0.460 | -0.590 | -0.131 | -28.5% |
| 0.6351 | -0.348 | -0.478 | -0.130 | -37.5% |
| 0.6981 | -0.236 | -0.365 | -0.129 | -54.7% |
| 0.7611 | -0.168 | -0.254 | -0.086 | — |
| 0.8285 | -0.099 | -0.131 | -0.032 | — |
| 0.8937 | -0.053 | -0.003 | 0.049 | — |
| 0.9546 | -0.050 | 0.111 | 0.161 | — |

At 0° the mid-chord mean absolute Cp error is 0.010 and the largest mid-chord absolute error is 0.025. The mid-chord band is the size of a rough photocopy digitising, so it does not support a tighter claim. The largest error on the full set is 0.208 at x/c = 0.0053. At 10° the coupling converged flag is yes and separated is yes. The mid-chord mean absolute error is 0.043 (largest 0.092). The mean signed error on the compared stations is -0.048. A separated run is not an attached-flow agreement. At 15° the solver separated flag is yes, the coupling converged flag is no, and the largest absolute Cp error on the compared stations is 1.446. That 15° distribution is recorded so the failure is visible. It is not an accuracy claim.

## Drag

Drag is compared only where two things are both true: the reference is the untripped Abbott polar at Re = 6×10⁶, and this solver did not flag laminar separation. Abbott CD is read at the CL FlightTreat produced, because the file is a drag polar against lift rather than against angle. The NASA file says the digitising is only approximate because the book plot is poor. A difference of a few ten-thousandths can be the digitising.

| α (deg) | CL FlightTreat | CD FlightTreat | CD pressure | CD friction | CD Abbott at that CL | Absolute error | Relative error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | -0.0008 | 0.00610 | 0.00083 | 0.00527 | 0.00591 | 0.00020 | 3.3% |
| 0.01 | 0.0004 | 0.00610 | 0.00083 | 0.00527 | 0.00591 | 0.00019 | 3.3% |
| 2.15 | 0.2381 | 0.00633 | 0.00104 | 0.00529 | 0.00599 | 0.00034 | 5.6% |
| 4.11 | 0.4573 | 0.00696 | 0.00189 | 0.00507 | 0.00659 | 0.00038 | 5.7% |
| 6.01 | 0.6739 | 0.00815 | 0.00339 | 0.00476 | 0.00787 | 0.00028 | 3.6% |

At 0° the absolute CD error against the digitised Abbott value is 0.00020 (3.3%). At 6.01° the absolute CD error is 0.00028 (3.6%). Pressure drag at that station is 0.00339 and friction drag is 0.00476. Stations the solver flags as laminar separation are omitted: skin friction downstream of that flag is dropped. The published drag is this surface sum. The wake momentum defect is a separate number and is not substituted for it.

## Wake momentum defect

The published CD above is the surface pressure integral plus skin friction. After that integral, the two wall layers are joined and marched one chord downstream as a single wake with no wall shear. The march holds the unresolved trailing-edge speed at the last resolved station, and streamwise derivatives are formed only on the resolved stations. At 0° those two upper-surface speeds are both 0.872. `cd_momentum` is `2 θ (Ue)^((H+5)/2)` at the last wake station. It is recorded here and is not the CD in the table above.

| α (deg) | CD surface | CD wake momentum | CD Abbott at the solved CL |
|---:|---:|---:|---:|
| 0.00 | 0.00610 | 0.00559 | 0.00591 |
| 0.01 | 0.00610 | 0.00559 | 0.00591 |
| 2.15 | 0.00633 | 0.00586 | 0.00599 |
| 4.11 | 0.00696 | 0.00617 | 0.00659 |
| 6.01 | 0.00815 | 0.00663 | 0.00787 |

On these attached stations the wake defect lies below the surface sum. It is not substituted for the published drag. The same march on a separated station is not a separated-flow drag result.

## What was validated, and what was not

- **CL** was compared with Ladson at matched Reynolds number and Mach number on the 120-grit table. The section is symmetric at 0°. Through 6° the relative lift error runs from 3.9% to 6.5%. It is a real disagreement. The inviscid thick-section circulation is still above the measured slope, and the boundary-layer coupling removes only the part of that difference that a closed-body transpiration integral can see.
- **Cp** was compared with the digitised Gregory upper-surface distributions at Re = 2.88×10⁶. At 0° the mid-chord mean absolute error is 0.010. At 10° and 15° the tables record whatever the solver produced, including a separated run. A separated distribution is not an attached-flow agreement.
- **CD** was compared with the digitised untripped Abbott polar only at stations the solver still treats as attached. At zero lift and at 6.01° the absolute CD errors are both reported in the table. They are outside a few ten-thousandths of digitising. Tripped Ladson drag was not used, because the trip condition is not what this solver computes.

## Remaining limitations

- The inviscid vortex-panel circulation of this thick section is still steeper than the Ladson and McCroskey slopes. Coupling puts the contour integral of the outward transpiration into that vortex system, so the circulation can change. The length-weighted mean of the transpiration is removed because a closed contour has no wake sheet in the panel system. A displacement body and a wake source were measured separately and were not kept: they did not improve lift without spoiling drag.
- There is no stall model. Past roughly 15° the calculated CL keeps rising after the experiment breaks.
- Lower-surface laminar separation is flagged from about 8° on this 120-panel mesh. The drag warning in the solver applies. The direct wake is not a separated-flow model, and its momentum defect is not the published drag.
- Michel transition is not a grit trip at 5% chord. A tripped-drag comparison would mix two boundary layers.
- Gregory Cp and Abbott CD were digitised from plots. NASA marks both files as only approximate. Differences smaller than a few hundredths in Cp, or a few ten-thousandths in CD, should not be read as a measured improvement.
- One mesh is reported: 120 cosine panels. A finer or coarser mesh will move Cp at the leading edge and can move the separation flag. This is not a grid-converged study.
- The 85–95% accuracy target in the root README is not what this comparison measured.

## Reproduce

```bash
python tests/test_naca0012_accuracy.py
python tests/naca0012_accuracy.py --write
```

The test solves the cases above and checks that this file still matches that run. `--write` regenerates the file after a solver change. Regenerating the file does not make a larger error into an agreement; the tables are the record.
