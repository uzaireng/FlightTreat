"""Published NACA 0012 checks for the existing 2D solver.

Ceilings are regression stops. The measured errors are the tables in
``src/physics_module/NACA0012_VALIDATION.md``, which this test rebuilds
and compares with the committed file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from naca0012_accuracy import (  # noqa: E402
    ATTACHED_ALPHAS,
    ATTACHED_CL_REL_CEILING,
    CP0_MID_MAE_CEILING,
    CP10_MID_MAE_CEILING,
    REFERENCE_DIR,
    REPORT_PATH,
    ZERO_LIFT_ABS_CEILING,
    ZERO_LIFT_CD_ABS_CEILING,
    absolute_and_relative,
    load_reference,
    mccroskey_lift_slope_per_deg,
    nearest_row,
    render_report,
    run_validation,
)


def _assert(cond: bool, message: str) -> None:
    if not cond:
        raise AssertionError(message)


def test_reference_files_state_their_conditions() -> None:
    ladson = (REFERENCE_DIR / "CLCD_Ladson_expdata.dat").read_text(encoding="utf-8")
    _assert("Re=6 million" in ladson, "Ladson file lost the Reynolds number")
    _assert("M=0.15" in ladson, "Ladson file lost the Mach number")
    _assert("transition tripped" in ladson, "Ladson file lost the trip note")
    _header, zones = load_reference("CLCD_Ladson_expdata.dat")
    row = nearest_row(zones["120 grit"], 4.11, 1.0e-6)
    _assert(row is not None, "120-grit table has no 4.11° row")
    _assert(abs(float(row[1]) - 0.4397) < 1.0e-6, f"120-grit CL at 4.11° changed: {row[1]}")
    _assert(abs(float(row[2]) - 0.00879) < 1.0e-6, f"120-grit CD at 4.11° changed: {row[2]}")

    gregory = (REFERENCE_DIR / "CP_Gregory_expdata.dat").read_text(encoding="utf-8")
    _assert("Re=2.88 million" in gregory, "Gregory file lost the Reynolds number")
    _assert("only approximate" in gregory, "Gregory file lost the digitising caveat")
    _header, gzones = load_reference("CP_Gregory_expdata.dat")
    nose = gzones["alpha=0"][0]
    _assert(float(nose[0]) == 0.0 and float(nose[1]) == 1.0, "Gregory α=0 no longer starts at Cp = 1")

    abbott = (REFERENCE_DIR / "0012.abbottdata.cd.dat").read_text(encoding="utf-8")
    _assert("Re=6 million" in abbott, "Abbott file lost the Reynolds number")
    _assert("only approximate" in abbott, "Abbott file lost the digitising caveat")

    mccroskey = (REFERENCE_DIR / "0012.mccroskeydata.cl.dat").read_text(encoding="utf-8")
    _assert("0.1025" in mccroskey and "0.00485" in mccroskey, "McCroskey slope formula is missing")
    slope = mccroskey_lift_slope_per_deg(6.0e6, 0.15)
    _assert(abs(slope - 0.10749) < 1.0e-4, f"McCroskey slope {slope}")


def test_error_uses_flighttreat_minus_reference() -> None:
    absolute, relative = absolute_and_relative(0.4985, 0.4397, 0.05)
    _assert(abs(absolute - 0.0588) < 1.0e-9, absolute)
    _assert(abs(relative - (0.0588 / 0.4397)) < 1.0e-9, relative)
    _absolute, near_zero = absolute_and_relative(0.001, 0.0122, 0.05)
    _assert(not np.isfinite(near_zero), "relative error should be blank near zero lift")


def test_naca0012_matches_published_tables() -> None:
    result = run_validation()
    report = render_report(result)
    stored = REPORT_PATH.read_text(encoding="utf-8")
    _assert(
        stored == report,
        "NACA0012_VALIDATION.md does not match this run. "
        "Regenerate with: python tests/naca0012_accuracy.py --write",
    )
    _assert(result["n_panels"] == 120, result["n_panels"])

    by_alpha = {round(row["solved"]["alpha_deg"], 2): row for row in result["lift_rows"]}
    zero = by_alpha[0.0]["solved"]
    _assert(abs(zero["cl"]) < ZERO_LIFT_ABS_CEILING, f"α=0 CL {zero['cl']}")
    _assert(zero["drag_supported"], "α=0 drag was excluded from the Abbott comparison")
    _assert(abs(by_alpha[0.0]["cd_abs"]) < ZERO_LIFT_CD_ABS_CEILING, by_alpha[0.0]["cd_abs"])

    for alpha in ATTACHED_ALPHAS:
        row = by_alpha[round(alpha, 2)]
        _assert(row["solved"]["converged"], f"{alpha} did not converge")
        _assert(not row["solved"]["separated"], f"{alpha} separated")
        _assert(np.isfinite(row["cl_rel"]), f"{alpha} has no relative lift error")
        _assert(
            abs(row["cl_rel"]) < ATTACHED_CL_REL_CEILING,
            f"{alpha} relative CL error {row['cl_rel']}",
        )

    cp = {block["solved"]["alpha_deg"]: block for block in result["cp_rows"]}
    _assert(cp[0.0]["cp"]["mid_mae"] < CP0_MID_MAE_CEILING, cp[0.0]["cp"]["mid_mae"])
    _assert(cp[0.0]["solved"]["converged"] and not cp[0.0]["solved"]["separated"], "α=0 Cp case separated")
    _assert(cp[10.0]["cp"]["mid_mae"] < CP10_MID_MAE_CEILING, cp[10.0]["cp"]["mid_mae"])
    _assert(np.isfinite(cp[15.0]["cp"]["max_abs"]), "α=15 Cp comparison did not run")
    _assert(cp[15.0]["cp"]["max_abs"] > 0.5, "α=15 Cp error was not recorded")


if __name__ == "__main__":
    test_reference_files_state_their_conditions()
    test_error_uses_flighttreat_minus_reference()
    test_naca0012_matches_published_tables()
    print("naca 0012 accuracy tests passed")
