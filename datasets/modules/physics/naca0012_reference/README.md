# NACA 0012 reference tables

Copied from the NASA Turbulence Modeling Resource page
[2DN00: 2D NACA 0012 Airfoil Validation](https://tmbwg.github.io/turbmodels/naca0012_val.html)
on 6 November 2026. The files are the versions linked from that page. They are inputs to
`tests/naca0012_accuracy.py`. They are not solver fixtures and they are not listed in
`catalog.json`.

| File | What it is | Used for |
|---|---|---|
| `CLCD_Ladson_expdata.dat` | Ladson, NASA TM 4074 (1988). Re = 6×10⁶, M = 0.15, transition tripped (80, 120, and 180 grit). | Lift. The 120-grit zone is the comparison table. |
| `CP_Gregory_expdata.dat` | Gregory and O'Reilly, R&M 3726 (1970). Re = 2.88×10⁶, upper surface only. The file says the points were digitized from a photocopy and are only approximate. | Surface pressure at 0°, 10°, and 15°. |
| `0012.abbottdata.cd.dat` | Abbott and von Doenhoff, *Theory of Wing Sections* (1959). Re = 6×10⁶, not tripped. The file says the plot digitizing is only approximate. | Drag, only as a same-Reynolds untripped polar. |
| `0012.mccroskeydata.cl.dat` | McCroskey fit quoted by NASA, AGARD CP-429 / NASA TM 100019 (1987). Re = 6×10⁶, M = 0.15. | Lift-curve slope. Two endpoints of the published fit, not a tunnel polar. |

Ladson, Hill, and Johnson pressure distributions (NASA TM 100526) are not stored here. That test is at Mach 0.3 on a model of aspect ratio 1.333, and the Turbulence Modeling Resource page says those pressures do not resolve the leading-edge peak well enough for a surface-pressure check.
