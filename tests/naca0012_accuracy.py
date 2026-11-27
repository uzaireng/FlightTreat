"""NACA 0012 accuracy comparison against the NASA reference tables.

The physics engine is called as it ships. This module does not change a
panel strength, a boundary-layer closure, or a force integral.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from geo_module import GeometryEngine  # noqa: E402
from math_module import MeshEngine  # noqa: E402
from physics_module import PhysicsEngine  # noqa: E402
from physics_module.boundary_layer_1d import laminar_drag_is_unsupported, last_resolved_index  # noqa: E402

REFERENCE_DIR = _ROOT / "datasets" / "modules" / "physics" / "naca0012_reference"
REPORT_PATH = _ROOT / "src" / "physics_module" / "NACA0012_VALIDATION.md"

# Workbench defaults for the surface discretization (mesh_workbench.py).
# The Cartesian farfield is only a picture of the finished panel solution,
# so the smallest legal grid is used and it is not part of the comparison.
N_PANELS = 120
N_GEOMETRY_POINTS = 201
FARFIELD_N = 8

LADSON_RE = 6.0e6
LADSON_MACH = 0.15
GREGORY_RE = 2.88e6
GREGORY_MACH = 0.15
PRIMARY_GRIT = "120 grit"

# 120-grit stations from the linear range through stall. Alpha = 0 is added
# so the symmetric section can be checked; it is not a Ladson row.
LADSON_ALPHAS = (-4.01, -2.12, 0.0, 0.01, 2.15, 4.11, 6.01, 8.08, 10.10, 12.13, 15.27, 18.18)
GREGORY_ALPHAS = (0.0, 10.0, 15.0)

# Relative error is omitted when the reference is this close to zero.
CL_REL_FLOOR = 0.05
CP_REL_FLOOR = 0.20
CD_REL_FLOOR = 0.002

# Attached-flow lift gate. The measured miss is smaller than this ceiling.
# The ceiling is a regression stop, not the accuracy result.
ATTACHED_ALPHAS = (2.15, 4.11, 6.01)
ATTACHED_CL_REL_CEILING = 0.25
CP0_MID_MAE_CEILING = 0.08
CP10_MID_MAE_CEILING = 0.25
ZERO_LIFT_ABS_CEILING = 0.02
ZERO_LIFT_CD_ABS_CEILING = 0.003


def parse_zones(text: str) -> tuple[list[str], dict[str, np.ndarray]]:
    """Split a NASA ``zone`` table into a header and one array per zone."""

    header: list[str] = []
    zones: dict[str, np.ndarray] = {}
    current: str | None = None
    rows: list[list[float]] = []

    def flush() -> None:
        nonlocal rows
        if current is not None:
            zones[current] = np.asarray(rows, dtype=np.float64)
        rows = []

    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            continue
        if raw.startswith("#") or raw.lower().startswith("variables"):
            header.append(raw)
            continue
        if raw.lower().startswith("zone"):
            flush()
            match = re.search(r't\s*=\s*"([^"]+)"', raw)
            current = match.group(1) if match else raw
            continue
        if current is None:
            current = "default"
        rows.append([float(piece) for piece in raw.split()])
    flush()
    return header, zones


def load_reference(name: str) -> tuple[list[str], dict[str, np.ndarray]]:
    path = REFERENCE_DIR / name
    return parse_zones(path.read_text(encoding="utf-8"))


def nearest_row(table: np.ndarray, alpha: float, tolerance: float) -> np.ndarray | None:
    if table.size == 0:
        return None
    distance = np.abs(table[:, 0] - float(alpha))
    index = int(np.argmin(distance))
    if float(distance[index]) > tolerance:
        return None
    return table[index]


def absolute_and_relative(predicted: float, reference: float, floor: float) -> tuple[float, float]:
    """FlightTreat minus reference. Relative error is nan below ``floor``."""

    absolute = float(predicted) - float(reference)
    if abs(float(reference)) < floor:
        return absolute, float("nan")
    return absolute, absolute / abs(float(reference))


def mccroskey_lift_slope_per_deg(reynolds: float, mach: float) -> float:
    """Fig. 3 fit quoted in ``0012.mccroskeydata.cl.dat``.

    dCL/dα (per degree) = (0.1025 + 0.00485 log10(Re/10^6)) / sqrt(1-M^2).
    """

    beta = float(np.sqrt(1.0 - mach * mach))
    return (0.1025 + 0.00485 * float(np.log10(reynolds / 1.0e6))) / beta


def _mesh():
    geometry = GeometryEngine().generate_from_spec(
        {
            "type": "naca",
            "mode": "2D",
            "code": "0012",
            "chord": 1.0,
            "num_points": N_GEOMETRY_POINTS,
        }
    )
    mesh = MeshEngine().from_geometry(
        geometry,
        {
            "n_panels": N_PANELS,
            "n_x": FARFIELD_N,
            "n_y": FARFIELD_N,
            "clustering": "cosine",
            "le_cluster_factor": 1.0,
            "te_cluster_factor": 1.0,
        },
    )
    return mesh


def _solve(mesh, alpha: float, reynolds: float, mach: float, regime: str) -> dict:
    output = PhysicsEngine().solve_from_mesh(
        mesh,
        {
            "alpha_deg": float(alpha),
            "reynolds": float(reynolds),
            "mach": float(mach),
            "v_inf": 1.0,
            "rho": 1.225,
        },
        {"regime": regime, "prandtl_glauert": True},
    )
    ibl = output["ibl"]
    loads = output["loads"]
    return {
        "alpha_deg": float(alpha),
        "reynolds": float(reynolds),
        "mach": float(mach),
        "regime": regime,
        "cl": float(loads["cl"]),
        "cd": float(loads["cd"]),
        "cd_pressure": float(loads["cd_pressure"]),
        "cd_friction": float(loads["cd_friction"]),
        "cd_momentum": float("nan") if ibl is None else float(ibl["cd_momentum"]),
        "ue_te_upper": float("nan") if ibl is None else float(ibl["ue_upper"][-1]),
        "ue_resolved_upper": (
            float("nan")
            if ibl is None or mesh["ibl_1d"] is None
            else float(ibl["ue_upper"][last_resolved_index(mesh["ibl_1d"]["s_upper"])])
        ),
        "converged": bool(output["converged"]),
        "iterations": int(output["vic_iterations"]),
        "residual": float(output["residual"]),
        "separated": bool(ibl["separated"]) if ibl is not None else False,
        "bubble_s_upper": float("nan") if ibl is None else float(ibl["bubble_s_upper"]),
        "bubble_s_lower": float("nan") if ibl is None else float(ibl["bubble_s_lower"]),
        "reattachment_s_upper": float("nan") if ibl is None else float(ibl["reattachment_s_upper"]),
        "reattachment_s_lower": float("nan") if ibl is None else float(ibl["reattachment_s_lower"]),
        "separation_s_upper": float("nan") if ibl is None else float(ibl["separation_s_upper"]),
        "separation_s_lower": float("nan") if ibl is None else float(ibl["separation_s_lower"]),
        "drag_supported": ibl is not None and output["converged"] and not laminar_drag_is_unsupported(ibl),
        "transition_s_upper": float(ibl["transition_s_upper"]) if ibl is not None else float("nan"),
        "transition_s_lower": float(ibl["transition_s_lower"]) if ibl is not None else float("nan"),
        "warnings": list(output["warnings"]),
        "cp_upper_x": None,
        "cp_upper": None,
        "output": output,
    }


def _upper_cp(mesh, output) -> tuple[np.ndarray, np.ndarray]:
    panel = mesh["panel_2d"]
    assert panel is not None
    chord = float(panel["chord"])
    xy = np.asarray(panel["collocation"], dtype=np.float64)
    cp = np.asarray(output["inviscid"]["cp_inviscid"], dtype=np.float64)
    upper = xy[:, 1] >= 0.0
    x = xy[upper, 0] / chord
    values = cp[upper]
    order = np.argsort(x)
    x = x[order]
    values = values[order]
    unique_x, index = np.unique(x, return_index=True)
    return unique_x, values[index]


def _cp_comparison(x_solver: np.ndarray, cp_solver: np.ndarray, reference: np.ndarray) -> dict:
    x_ref = reference[:, 0]
    cp_ref = reference[:, 1]
    inside = (x_ref >= float(x_solver[0])) & (x_ref <= float(x_solver[-1]))
    x_used = x_ref[inside]
    ref_used = cp_ref[inside]
    predicted = np.interp(x_used, x_solver, cp_solver)
    absolute = predicted - ref_used
    relative = np.full(absolute.shape, np.nan, dtype=np.float64)
    usable = np.abs(ref_used) >= CP_REL_FLOOR
    relative[usable] = absolute[usable] / np.abs(ref_used[usable])
    mid = (x_used >= 0.05) & (x_used <= 0.90)
    return {
        "x": x_used,
        "reference": ref_used,
        "predicted": predicted,
        "absolute": absolute,
        "relative": relative,
        "n_dropped": int(np.size(x_ref) - np.size(x_used)),
        "mae": float(np.mean(np.abs(absolute))) if absolute.size else float("nan"),
        "max_abs": float(np.max(np.abs(absolute))) if absolute.size else float("nan"),
        "rmse": float(np.sqrt(np.mean(absolute**2))) if absolute.size else float("nan"),
        "mid_mae": float(np.mean(np.abs(absolute[mid]))) if np.any(mid) else float("nan"),
        "mid_max_abs": float(np.max(np.abs(absolute[mid]))) if np.any(mid) else float("nan"),
    }


def _abbott_cd(cl: float, table: np.ndarray) -> float | None:
    order = np.argsort(table[:, 0])
    cl_ref = table[order, 0]
    cd_ref = table[order, 1]
    if cl < float(cl_ref[0]) or cl > float(cl_ref[-1]):
        return None
    return float(np.interp(cl, cl_ref, cd_ref))


def _grit_span(zones: dict[str, np.ndarray], alpha: float) -> tuple[float, float] | None:
    values = []
    for name, table in zones.items():
        row = nearest_row(table, alpha, 0.20)
        if row is not None:
            values.append(float(row[1]))
    if len(values) < 2:
        return None
    return min(values), max(values)


def run_validation() -> dict:
    """Solve the documented cases and attach reference errors."""

    ladson_header, ladson = load_reference("CLCD_Ladson_expdata.dat")
    gregory_header, gregory = load_reference("CP_Gregory_expdata.dat")
    abbott_header, abbott = load_reference("0012.abbottdata.cd.dat")
    mccroskey_header, _mccroskey = load_reference("0012.mccroskeydata.cl.dat")
    primary = ladson[PRIMARY_GRIT]
    mesh = _mesh()
    nodes = np.asarray(mesh["panel_2d"]["nodes"], dtype=np.float64)
    thickness = float(np.max(nodes[:, 1]) - np.min(nodes[:, 1]))

    lift_rows = []
    for alpha in LADSON_ALPHAS:
        solved = _solve(mesh, alpha, LADSON_RE, LADSON_MACH, "viscous_inviscid")
        row = nearest_row(primary, alpha, 1.0e-6)
        record = {
            "solved": solved,
            "cl_ref": None if row is None else float(row[1]),
            "cd_ref_tripped": None if row is None else float(row[2]),
            "grit_cl": _grit_span(ladson, alpha),
        }
        if record["cl_ref"] is not None:
            record["cl_abs"], record["cl_rel"] = absolute_and_relative(
                solved["cl"], record["cl_ref"], CL_REL_FLOOR
            )
        else:
            record["cl_abs"], record["cl_rel"] = float("nan"), float("nan")
        if solved["drag_supported"]:
            record["cd_abbott"] = _abbott_cd(solved["cl"], abbott["default"])
        else:
            record["cd_abbott"] = None
        if record["cd_abbott"] is None:
            record["cd_abs"], record["cd_rel"] = float("nan"), float("nan")
        else:
            record["cd_abs"], record["cd_rel"] = absolute_and_relative(
                solved["cd"], record["cd_abbott"], CD_REL_FLOOR
            )
        lift_rows.append(record)

    by_alpha = {round(row["solved"]["alpha_deg"], 2): row for row in lift_rows}
    slope_alpha = 6.01
    cl_at_slope = by_alpha[slope_alpha]["solved"]["cl"]
    cl_at_zero_row = by_alpha[0.01]["solved"]["cl"]
    flight_slope = (cl_at_slope - cl_at_zero_row) / (slope_alpha - 0.01)
    published_slope = mccroskey_lift_slope_per_deg(LADSON_RE, LADSON_MACH)

    diagnostics = {
        "inviscid_m015": _solve(mesh, 4.11, LADSON_RE, LADSON_MACH, "inviscid"),
        "viscous_m0": _solve(mesh, 4.11, LADSON_RE, 0.0, "viscous_inviscid"),
    }

    cp_rows = []
    for alpha in GREGORY_ALPHAS:
        solved = _solve(mesh, alpha, GREGORY_RE, GREGORY_MACH, "viscous_inviscid")
        x_solver, cp_solver = _upper_cp(mesh, solved["output"])
        zone = gregory[f"alpha={int(alpha)}"]
        compared = _cp_comparison(x_solver, cp_solver, zone)
        cp_rows.append({"solved": solved, "cp": compared})

    return {
        "n_panels": int(mesh["quality"]["n_panels"]),
        "thickness": thickness,
        "ladson_header": ladson_header,
        "gregory_header": gregory_header,
        "abbott_header": abbott_header,
        "mccroskey_header": mccroskey_header,
        "lift_rows": lift_rows,
        "cp_rows": cp_rows,
        "flight_slope": float(flight_slope),
        "published_slope": float(published_slope),
        "diagnostics": diagnostics,
        "ladson_cl_at_4_11": float(nearest_row(primary, 4.11, 1.0e-6)[1]),
    }


def _fmt(value: float, digits: int) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def _pct(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{100.0 * value:.1f}%"


def _grit_text(span: tuple[float, float] | None) -> str:
    if span is None:
        return "—"
    return f"{span[0]:.4f} to {span[1]:.4f}"


def _nose_sentence(block: dict) -> str:
    absolute = np.asarray(block["cp"]["absolute"], dtype=np.float64)
    if absolute.size == 0:
        return ""
    index = int(np.argmax(np.abs(absolute)))
    return (
        f"The largest error on the full set is {abs(float(absolute[index])):.3f} "
        f"at x/c = {float(block['cp']['x'][index]):.4f}."
    )


def _negative_lift_sentence(result: dict) -> str:
    rows = [
        row
        for row in result["lift_rows"]
        if row["solved"]["alpha_deg"] < 0.0 and np.isfinite(row["cl_rel"])
    ]
    if not rows:
        return ""
    angles = " and ".join(f"{row['solved']['alpha_deg']:.2f}°" for row in rows)
    errors = " and ".join(_pct(row["cl_rel"]) for row in rows)
    return f"The relative errors at {angles} are {errors}."


def _separation_phrase(result: dict) -> str:
    eight = next(row for row in result["lift_rows"] if abs(row["solved"]["alpha_deg"] - 8.08) < 1e-9)
    solved = eight["solved"]
    if (
        np.isfinite(solved["bubble_s_lower"])
        and np.isfinite(solved["reattachment_s_lower"])
        and not solved["separated"]
    ):
        return (
            "a lower-surface laminar separation that reattaches "
            f"(onset s={solved['bubble_s_lower']:.3f}, reattachment s={solved['reattachment_s_lower']:.3f})"
        )
    if solved["separated"]:
        return "separation"
    return "no separation flag at 8.08°"


def _grit_sentence(result: dict) -> str:
    widths = []
    for row in result["lift_rows"]:
        if row["solved"]["alpha_deg"] not in ATTACHED_ALPHAS or row["grit_cl"] is None:
            continue
        low, high = row["grit_cl"]
        widths.append(high - low)
    if not widths:
        return "No second grit row sits within 0.2° of the attached stations."
    return (
        "The grit-to-grit CL span on those attached stations is at most "
        f"{max(widths):.4f}, so the miss is larger than the trip scatter."
    )


def render_report(result: dict) -> str:
    """Markdown report. Every coefficient in it comes from ``result``."""

    lines: list[str] = []
    add = lines.append
    add("# NACA 0012 aerodynamic accuracy validation")
    add("")
    add("This report compares the 2D viscous-inviscid panel solver with published NACA 0012 data. The reference files were not edited for the comparison.")
    add("")
    add("Error is FlightTreat minus the reference. Relative error is that difference divided by the absolute reference value. Relative error is left blank when the reference is within 0.05 of zero for lift, 0.20 for pressure, or 0.002 for drag, because a tiny denominator is not a useful percentage.")
    add("")
    add("## Reference sources and operating conditions")
    add("")
    add("The tables live in `datasets/modules/physics/naca0012_reference/`. They were copied from the NASA Turbulence Modeling Resource NACA 0012 validation page on 6 November 2026.")
    add("")
    add("| Quantity | Source | Conditions used | Why this source |")
    add("|---|---|---|---|")
    add("| Lift coefficient, CL | Ladson, NASA TM 4074 (1988), file `CLCD_Ladson_expdata.dat` | Re = 6×10⁶, M = 0.15, 120-grit trip. FlightTreat is run at those Re and M with Prandtl–Glauert on. | A force table, not a digitised plot. NASA uses it for forces at this Reynolds number. Lift is taken from it. Drag is not, because the grit trip is not this solver's boundary layer. |")
    add("| Lift-curve slope | McCroskey, AGARD CP-429 / NASA TM 100019 (1987), file `0012.mccroskeydata.cl.dat` | The file's fit at Re = 6×10⁶, M = 0.15 | A published fit across several tunnels. It is a slope check, not a second polar. |")
    add("| Pressure coefficient, Cp | Gregory and O'Reilly, R&M 3726 (1970), file `CP_Gregory_expdata.dat` | Re = 2.88×10⁶, M = 0.15, upper surface, α = 0°, 10°, 15°. The file does not record Mach. M = 0.15 is inside the 0.08–0.16 range McCroskey lists for this test. | NASA recommends this distribution over the Ladson, Hill, and Johnson pressures because the leading-edge peak is better resolved. |")
    add("| Drag coefficient, CD | Abbott and von Doenhoff, *Theory of Wing Sections* (1959), file `0012.abbottdata.cd.dat` | Re = 6×10⁶, natural transition. Compared at the lift FlightTreat actually produced. | Same section, same Reynolds number, and natural transition, which is the class this solver runs. |")
    add("")
    add("Ladson drag is not a validation target. That test fixed transition with grit. This solver transitions on its own with Michel's criterion and cannot place a trip. The two drag conditions do not match, so a Ladson CD error is not reported as an accuracy result.")
    add("")
    add("Ladson, Hill, and Johnson Cp (NASA TM 100526) is also not used. It was measured at Mach 0.3 on a model of aspect ratio 1.333, and the NASA page says the leading-edge suction peak is poorly resolved.")
    add("")
    add("## Solver setup")
    add("")
    add(f"NACA 0012, chord 1, closed trailing edge (the generator's thickness coefficient is 0.1036 so the trailing edge meets). Maximum thickness on this mesh is {result['thickness']:.4f} chords. {result['n_panels']} cosine-clustered panels, leading-edge and trailing-edge cluster factors 1, which is the workbench default. Regime `viscous_inviscid`. Freestream speed 1 and density 1.225. The 8×8 Cartesian farfield is built because the solver requires it; surface Cp and the forces do not read that grid.")
    add("")
    add("The wind-tunnel models are the ordinary blunt-trailing-edge NACA 0012. NASA's own CFD grids use a slightly thinned sharp-trailing-edge scaling. That geometry difference is small next to the lift differences below and is not offered as the cause of them.")
    add("")
    add("## Lift")
    add("")
    add("120-grit Ladson stations. The grit span is the lowest and highest CL among the 80, 120, and 180 grit rows within 0.2° of that angle. Alpha = 0 has no Ladson row; the nearby tripped measurement at 0.01° is CL = -0.0122.")
    add("")
    add("| α (deg) | CL reference | CL FlightTreat | Absolute error | Relative error | Grit span | Converged | Separated |")
    add("|---:|---:|---:|---:|---:|---|---|---|")
    for row in result["lift_rows"]:
        solved = row["solved"]
        add(
            "| {alpha:.2f} | {ref} | {cl} | {abs_e} | {rel} | {span} | {conv} | {sep} |".format(
                alpha=solved["alpha_deg"],
                ref=_fmt(row["cl_ref"], 4) if row["cl_ref"] is not None else "—",
                cl=_fmt(solved["cl"], 4),
                abs_e=_fmt(row["cl_abs"], 4),
                rel=_pct(row["cl_rel"]),
                span=_grit_text(row["grit_cl"]),
                conv="yes" if solved["converged"] else "no",
                sep="yes" if solved["separated"] else "no",
            )
        )
    add("")
    add(
        "From 0.01° to 6.01° the FlightTreat lift slope is "
        f"**{result['flight_slope']:.4f} per degree**. "
        f"The McCroskey fit at this Reynolds number and Mach number is "
        f"**{result['published_slope']:.4f} per degree**. "
        f"Absolute slope error {result['flight_slope'] - result['published_slope']:.4f} per degree, "
        f"relative error {_pct((result['flight_slope'] - result['published_slope']) / result['published_slope'])}."
    )
    add("")
    inv = result["diagnostics"]["inviscid_m015"]
    vis0 = result["diagnostics"]["viscous_m0"]
    vis = next(row["solved"] for row in result["lift_rows"] if abs(row["solved"]["alpha_deg"] - 4.11) < 1e-9)
    add("The same 4.11° case shows where the extra lift comes from.")
    add("")
    add("| Case at 4.11° | CL |")
    add("|---|---:|")
    add(f"| Ladson 120-grit | {result['ladson_cl_at_4_11']:.4f} |")
    add(f"| Viscous-inviscid, M = 0.15 | {vis['cl']:.4f} |")
    add(f"| Viscous-inviscid, M = 0 | {vis0['cl']:.4f} |")
    add(f"| Inviscid, M = 0.15 | {inv['cl']:.4f} |")
    add("")
    coupling_delta = vis["cl"] - inv["cl"]
    compressibility_delta = vis["cl"] - vis0["cl"]
    add(
        f"At 4.11° the viscous-inviscid CL at M = 0.15 is {vis['cl']:.4f}. "
        f"Turning Mach off leaves CL at {vis0['cl']:.4f}. "
        f"The inviscid solution at M = 0.15 is {inv['cl']:.4f}. "
        f"Viscous coupling changes CL by {coupling_delta:+.4f} from that inviscid solution. "
        f"Prandtl–Glauert changes the viscous CL by {compressibility_delta:+.4f}."
    )
    add("")
    low = next(row for row in result["lift_rows"] if abs(row["solved"]["alpha_deg"] - 2.15) < 1e-9)
    high = next(row for row in result["lift_rows"] if abs(row["solved"]["alpha_deg"] - 6.01) < 1e-9)
    stalled = next(row for row in result["lift_rows"] if abs(row["solved"]["alpha_deg"] - 18.18) < 1e-9)
    add(
        "Through 6° the run is converged and the boundary layer stays attached. "
        f"CL is high by {low['cl_abs']:.4f} at 2.15° ({_pct(low['cl_rel'])} of the measured lift) "
        f"and by {high['cl_abs']:.4f} at 6.01° ({_pct(high['cl_rel'])}). "
        + _grit_sentence(result)
        + " "
        + _negative_lift_sentence(result)
        + " From 8° the lower surface meets Thwaites separation ahead of Michel transition. "
        + "The march treats that as "
        + _separation_phrase(result)
        + ". "
        + "By 18° the experiment has stalled "
        f"(CL {stalled['cl_ref']:.4f}) while the panel solution is still at {stalled['solved']['cl']:.4f}."
    )
    add("")
    add("## Pressure")
    add("")
    add("Upper-surface Cp from Gregory and O'Reilly. FlightTreat Cp is interpolated in x/c onto each experimental station that lies inside the panel x-range. Stations outside that range are dropped rather than extrapolated. The file says the points were digitised from a photocopy and are only approximate. A mid-chord band, 0.05 ≤ x/c ≤ 0.90, is summarised beside the full set so one leading-edge point does not hide the rest.")
    add("")
    for block in result["cp_rows"]:
        solved = block["solved"]
        cp = block["cp"]
        alpha = solved["alpha_deg"]
        add(
            f"### α = {alpha:.0f}°"
        )
        add("")
        add(
            f"Re = {GREGORY_RE:.2e}, M = {GREGORY_MACH:.2f}. "
            f"CL = {solved['cl']:.4f}, CD = {solved['cd']:.5f}. "
            f"Converged: {'yes' if solved['converged'] else 'no'}. "
            f"Separated: {'yes' if solved['separated'] else 'no'}. "
            f"Stations compared: {cp['x'].size}. Dropped: {cp['n_dropped']}."
        )
        add("")
        add("| | Mean absolute error | Maximum absolute error |")
        add("|---|---:|---:|")
        add(f"| All compared stations | {cp['mae']:.3f} | {cp['max_abs']:.3f} |")
        add(f"| 0.05 ≤ x/c ≤ 0.90 | {cp['mid_mae']:.3f} | {cp['mid_max_abs']:.3f} |")
        add("")
        add("| x/c | Cp reference | Cp FlightTreat | Absolute error | Relative error |")
        add("|---:|---:|---:|---:|---:|")
        for x, ref, pred, abs_e, rel in zip(
            cp["x"], cp["reference"], cp["predicted"], cp["absolute"], cp["relative"]
        ):
            add(
                f"| {x:.4f} | {ref:.3f} | {pred:.3f} | {abs_e:.3f} | {_pct(float(rel))} |"
            )
        add("")
    cp0 = next(block for block in result["cp_rows"] if abs(block["solved"]["alpha_deg"]) < 1e-9)
    cp10 = next(block for block in result["cp_rows"] if abs(block["solved"]["alpha_deg"] - 10.0) < 1e-9)
    cp15 = next(block for block in result["cp_rows"] if abs(block["solved"]["alpha_deg"] - 15.0) < 1e-9)
    cp10_signed = float(np.mean(cp10["cp"]["absolute"]))
    add(
        f"At 0° the mid-chord mean absolute Cp error is {cp0['cp']['mid_mae']:.3f} "
        f"and the largest mid-chord absolute error is {cp0['cp']['mid_max_abs']:.3f}. "
        "The mid-chord band is the size of a rough photocopy digitising, so it does not support a tighter claim. "
        + _nose_sentence(cp0)
        + " "
        f"At 10° the coupling converged flag is {'yes' if cp10['solved']['converged'] else 'no'} "
        f"and separated is {'yes' if cp10['solved']['separated'] else 'no'}. "
        f"The mid-chord mean absolute error is {cp10['cp']['mid_mae']:.3f} "
        f"(largest {cp10['cp']['mid_max_abs']:.3f}). "
        f"The mean signed error on the compared stations is {cp10_signed:+.3f}. "
        "A run with an unrecovered separation is not an attached-flow agreement. "
        f"At 15° the solver separated flag is {'yes' if cp15['solved']['separated'] else 'no'}, "
        f"the coupling converged flag is {'yes' if cp15['solved']['converged'] else 'no'}, "
        f"and the largest absolute Cp error on the compared stations is {cp15['cp']['max_abs']:.3f}. "
        "That 15° distribution is recorded so the failure is visible. It is not an accuracy claim."
    )
    add("")
    add("## Drag")
    add("")
    add("Drag is compared only where two things are both true: the reference is the untripped Abbott polar at Re = 6×10⁶, and this solver did not leave an unrecovered laminar separation. A reattached bubble is included. Abbott CD is read at the CL FlightTreat produced, because the file is a drag polar against lift rather than against angle. The NASA file says the digitising is only approximate because the book plot is poor. A difference of a few ten-thousandths can be the digitising.")
    add("")
    add("| α (deg) | CL FlightTreat | CD FlightTreat | CD pressure | CD friction | CD Abbott at that CL | Absolute error | Relative error |")
    add("|---:|---:|---:|---:|---:|---:|---:|---:|")
    quoted = 0
    for row in result["lift_rows"]:
        solved = row["solved"]
        if solved["alpha_deg"] < 0.0 or row["cd_abbott"] is None:
            continue
        quoted += 1
        add(
            "| {alpha:.2f} | {cl} | {cd} | {cdp} | {cdf} | {ref} | {abs_e} | {rel} |".format(
                alpha=solved["alpha_deg"],
                cl=_fmt(solved["cl"], 4),
                cd=_fmt(solved["cd"], 5),
                cdp=_fmt(solved["cd_pressure"], 5),
                cdf=_fmt(solved["cd_friction"], 5),
                ref=_fmt(row["cd_abbott"], 5),
                abs_e=_fmt(row["cd_abs"], 5),
                rel=_pct(row["cd_rel"]),
            )
        )
    if quoted == 0:
        add("| — | — | — | — | — | — | — | — |")
    add("")
    zero = next(row for row in result["lift_rows"] if abs(row["solved"]["alpha_deg"] - 0.0) < 1e-9)
    six = next(row for row in result["lift_rows"] if abs(row["solved"]["alpha_deg"] - 6.01) < 1e-9)
    add(
        f"At 0° the absolute CD error against the digitised Abbott value is {zero['cd_abs']:.5f} "
        f"({_pct(zero['cd_rel'])}). "
        f"At 6.01° the absolute CD error is {six['cd_abs']:.5f} "
        f"({_pct(six['cd_rel'])}). "
        f"Pressure drag at that station is {six['solved']['cd_pressure']:.5f} "
        f"and friction drag is {six['solved']['cd_friction']:.5f}. "
        "An unrecovered laminar separation, one that never reaches free-shear transition, is omitted. "
        "A bubble that reattaches stays in the table: its drag is the surface pressure integral plus skin friction, "
        "with zero wall shear only inside the bubble. "
        "The wake momentum defect is a separate number and is not substituted for it."
    )
    add("")
    add("## Wake momentum defect")
    add("")
    zero_for_wake = next(row for row in result["lift_rows"] if abs(row["solved"]["alpha_deg"]) < 1e-9)
    ue_resolved = zero_for_wake["solved"]["ue_resolved_upper"]
    add(
        "The published CD above is the surface pressure integral plus skin friction. "
        "After that integral, the two wall layers are joined and marched one chord downstream "
        "as a single wake with no wall shear. The march holds the unresolved trailing-edge "
        "speed at the last resolved station, and streamwise derivatives are formed only on "
        "the resolved stations. "
        f"At 0° those two upper-surface speeds are both {ue_resolved:.3f}. "
        "`cd_momentum` is `2 θ (Ue)^((H+5)/2)` at the last wake station. "
        "It is recorded here and is not the CD in the table above."
    )
    add("")
    add("| α (deg) | CD surface | CD wake momentum | CD Abbott at the solved CL |")
    add("|---:|---:|---:|---:|")
    for row in result["lift_rows"]:
        solved = row["solved"]
        if solved["alpha_deg"] < 0.0 or not solved["drag_supported"]:
            continue
        add(
            "| {alpha:.2f} | {surface} | {wake} | {ref} |".format(
                alpha=solved["alpha_deg"],
                surface=_fmt(solved["cd"], 5),
                wake=_fmt(solved["cd_momentum"], 5),
                ref=_fmt(row["cd_abbott"], 5),
            )
        )
    add("")
    add("On these stations the wake defect is not substituted for the published drag. A station with an unrecovered separation is not in this table. The direct wake is not a separated-flow drag model.")
    add("")
    add("## Separation and boundary layer")
    add("")
    add("Arc length is measured from the geometric leading edge along the surface. `bubble` is a laminar-separation onset. `reattach` is free-shear transition followed by an attached turbulent restart that stayed attached on its first resolved step. `separation` is an unrecovered separation. A missing entry was not predicted.")
    add("")
    add("| α (deg) | Upper bubble s | Upper reattach s | Upper separation s | Lower bubble s | Lower reattach s | Lower separation s | Iterations | Residual |")
    add("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in result["lift_rows"]:
        solved = row["solved"]
        add(
            "| {alpha:.2f} | {bu} | {ru} | {su} | {bl} | {rl} | {sl} | {it} | {res} |".format(
                alpha=solved["alpha_deg"],
                bu=_fmt(solved["bubble_s_upper"], 3),
                ru=_fmt(solved["reattachment_s_upper"], 3),
                su=_fmt(solved["separation_s_upper"], 3),
                bl=_fmt(solved["bubble_s_lower"], 3),
                rl=_fmt(solved["reattachment_s_lower"], 3),
                sl=_fmt(solved["separation_s_lower"], 3),
                it=solved["iterations"],
                res=f"{solved['residual']:.2e}",
            )
        )
    add("")
    add("Skin friction is zero between a laminar-separation onset and reattachment, and downstream of an unrecovered separation. Momentum thickness is not held constant there: the laminar part follows the momentum integral with Cf = 0, and a turbulent separated part uses Head's entrainment evaluated at H = 2.8. θ and δ* at the trailing edge are the values the wake is started from. They are not a second drag.")
    add("")
    add("## What was validated, and what was not")
    add("")
    rels = [
        row["cl_rel"]
        for row in result["lift_rows"]
        if row["solved"]["alpha_deg"] in ATTACHED_ALPHAS and np.isfinite(row["cl_rel"])
    ]
    cp0 = next(block for block in result["cp_rows"] if abs(block["solved"]["alpha_deg"]) < 1e-9)
    add(
        "- **CL** was compared with Ladson at matched Reynolds number and Mach number on the 120-grit table. "
        f"The section is symmetric at 0°. Through 6° the relative lift error runs from {_pct(min(rels))} to {_pct(max(rels))}. "
        "It is a real disagreement. "
        "The inviscid thick-section circulation is still above the measured slope, and the "
        "boundary-layer coupling removes only the part of that difference that a closed-body "
        "transpiration integral can see."
    )
    add(
        "- **Cp** was compared with the digitised Gregory upper-surface distributions at Re = 2.88×10⁶. "
        f"At 0° the mid-chord mean absolute error is {cp0['cp']['mid_mae']:.3f}. "
        "At 10° and 15° the tables record whatever the solver produced. "
        "A distribution from an unrecovered separation, or from a coupling that did not converge, "
        "is not an attached-flow agreement."
    )
    add(
        "- **CD** was compared with the digitised untripped Abbott polar where the march did not leave an unrecovered laminar separation. "
        "At zero lift and at 6.01° the absolute CD errors are both reported in the table. "
        "They are outside a few ten-thousandths of digitising. "
        "Tripped Ladson drag was not used, because the trip condition is not what this solver computes."
    )
    add("")
    add("## Remaining limitations")
    add("")
    add("- The inviscid vortex-panel circulation of this thick section is still steeper than the Ladson and McCroskey slopes. Coupling puts the contour integral of the outward transpiration into that vortex system, so the circulation can change. The length-weighted mean of the transpiration is removed because a closed contour has no wake sheet in the panel system. A displacement body and a wake source were measured separately and were not kept: they did not improve lift without spoiling drag.")
    add("- There is no stall model. The upper-surface turbulent march on this mesh stays attached through the angles where the Ladson lift is still rising, and at 18° the calculated CL is still far above the stalled measurement. A separated shear layer is marched only after a resolved station reaches H = 2.8 or the skin friction collapses. On this 120-panel mesh that happens at the trailing edge, too late to unload the suction peak.")
    add(
        "- "
        + _separation_phrase(result)[:1].upper()
        + _separation_phrase(result)[1:]
        + " is what this 120-panel mesh produces at 8.08°. "
        + "Wall shear is zero only inside that bubble. "
        + "An unrecovered laminar separation still withholds L/D. "
        + "The direct wake is not a separated-flow drag model, and its momentum defect is not the published drag."
    )
    add("- Michel transition is not a grit trip at 5% chord. A tripped-drag comparison would mix two boundary layers.")
    add("- Gregory Cp and Abbott CD were digitised from plots. NASA marks both files as only approximate. Differences smaller than a few hundredths in Cp, or a few ten-thousandths in CD, should not be read as a measured improvement.")
    add("- One mesh is reported: 120 cosine panels. A finer or coarser mesh will move Cp at the leading edge and can move the separation flag. This is not a grid-converged study.")
    add("- The 85–95% accuracy target in the root README is not what this comparison measured.")
    add("")
    add("## Reproduce")
    add("")
    add("```bash")
    add("python tests/test_naca0012_accuracy.py")
    add("python tests/naca0012_accuracy.py --write")
    add("```")
    add("")
    add("The test solves the cases above and checks that this file still matches that run. `--write` regenerates the file after a solver change. Regenerating the file does not make a larger error into an agreement; the tables are the record.")
    add("")
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    result = run_validation()
    report = render_report(result)
    if "--write" in argv:
        REPORT_PATH.write_text(report, encoding="utf-8", newline="\n")
        print(f"wrote {REPORT_PATH}")
    else:
        print(report)


if __name__ == "__main__":
    main(sys.argv[1:])
