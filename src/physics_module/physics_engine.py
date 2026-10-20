"""Module 3 façade. Copies the mesh, dispatches 2D or 3D, and couples the boundary layer."""

from __future__ import annotations

import numpy as np

from math_module.math_skeleton import MeshOutput
from math_module.panel_2d import build_panel_metrics_2d

from .boundary_layer_1d import march_ibl
from .field_reconstruction import evaluate_farfield_with_freestream
from .forces_and_moments import integrate_loads
from .physics_skeleton import (
    FreestreamConditions,
    IBLSolution,
    InvalidFlowConditionError,
    InviscidSolution,
    PhysicsEngineError,
    PhysicsInput,
    PhysicsOutput,
    PhysicsParameters,
    validate_physics_output,
)
from .solver_2d import solve_2d
from .solver_3d import solve_3d

_FREESTREAM_KEYS = ("alpha_deg", "reynolds", "mach", "v_inf", "rho")


def _copy_block(block):
    if block is None:
        return None
    copied = {}
    for key, value in block.items():
        if isinstance(value, np.ndarray):
            copied[key] = np.array(value, copy=True)
        else:
            copied[key] = value
    return copied


def copy_mesh(mesh: MeshOutput) -> MeshOutput:
    """Deep-copy every array. Module 3 never writes back into Module 2."""

    return {
        "mode": mesh["mode"],
        "shape_name": mesh["shape_name"],
        "source_geometry_closed": bool(mesh["source_geometry_closed"]),
        "panel_2d": _copy_block(mesh["panel_2d"]),
        "cartesian_2d": _copy_block(mesh["cartesian_2d"]),
        "ibl_1d": _copy_block(mesh["ibl_1d"]),
        "panel_3d": _copy_block(mesh["panel_3d"]),
        "quality": dict(mesh["quality"]),
    }


def _require_finite(name: str, value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidFlowConditionError(f"{name} must be a finite number.") from exc
    if not np.isfinite(number):
        raise InvalidFlowConditionError(f"{name} must be finite.")
    return number


def _validate_freestream(freestream: FreestreamConditions, regime: str, prandtl: bool) -> FreestreamConditions:
    alpha = _require_finite("alpha_deg", freestream["alpha_deg"])
    reynolds = _require_finite("reynolds", freestream["reynolds"])
    mach = _require_finite("mach", freestream["mach"])
    speed = _require_finite("v_inf", freestream["v_inf"])
    rho = _require_finite("rho", freestream["rho"])
    if not -90.0 <= alpha <= 90.0:
        raise InvalidFlowConditionError("alpha_deg must lie inside [-90, 90].")
    if mach < 0.0:
        raise InvalidFlowConditionError("mach must be >= 0.")
    if mach >= 1.0:
        raise InvalidFlowConditionError("mach >= 1 is outside the panel solver.")
    if prandtl and mach >= 0.7:
        raise InvalidFlowConditionError(
            "Prandtl–Glauert is outside its range for mach >= 0.7."
        )
    if speed <= 0.0:
        raise InvalidFlowConditionError("v_inf must be > 0.")
    if rho <= 0.0:
        raise InvalidFlowConditionError("rho must be > 0.")
    if regime == "viscous_inviscid" and reynolds <= 0.0:
        raise InvalidFlowConditionError("Reynolds number must be > 0 for a viscous-inviscid solve.")
    return {
        "alpha_deg": alpha,
        "reynolds": reynolds,
        "mach": mach,
        "v_inf": speed,
        "rho": rho,
    }


def resolve_parameters(mesh: MeshOutput, parameters: PhysicsParameters | None) -> dict:
    """Fill omitted solver keys. Does not mutate ``parameters``."""

    chosen = dict(parameters or {})
    mode = mesh["mode"]
    if chosen.get("ibl_closure", "thwaites_head") == "drela_two_equation":
        raise InvalidFlowConditionError("ibl_closure 'drela_two_equation' is not implemented")
    if chosen.get("viscous_coupling", "transpiration") == "displacement":
        raise InvalidFlowConditionError("viscous_coupling 'displacement' is not implemented")
    if "regime" not in chosen:
        if mode == "2D":
            chosen["regime"] = "viscous_inviscid" if mesh["ibl_1d"] is not None else "inviscid"
        else:
            chosen["regime"] = "inviscid" if mesh["ibl_1d"] is None else "viscous_inviscid"
    if chosen["regime"] not in ("inviscid", "viscous_inviscid"):
        raise InvalidFlowConditionError(f"Unknown regime {chosen['regime']!r}.")
    chosen.setdefault("panel_model_2d", "vortex")
    chosen.setdefault("panel_model_3d", "constant_source_doublet")
    chosen.setdefault("ibl_closure", "thwaites_head")
    if "kutta" not in chosen:
        if mode == "2D":
            panel = mesh["panel_2d"]
            chosen["kutta"] = True if panel is None else bool(panel["is_closed"])
        else:
            panel3 = mesh["panel_3d"]
            chosen["kutta"] = False if panel3 is None else not bool(panel3["is_closed"])
    chosen.setdefault("max_vic_iterations", 8)
    chosen.setdefault("vic_tolerance", 1e-4)
    chosen.setdefault("viscous_coupling", "transpiration")
    chosen.setdefault("fail_on_separation", False)
    chosen.setdefault("prandtl_glauert", True)
    chosen.setdefault("wake_length_chords", 20.0)
    chosen.setdefault("dense_panel_limit", 2000)
    if int(chosen["max_vic_iterations"]) < 1:
        raise InvalidFlowConditionError("max_vic_iterations must be >= 1.")
    if float(chosen["vic_tolerance"]) <= 0.0:
        raise InvalidFlowConditionError("vic_tolerance must be > 0.")
    return chosen


def _beta(mach: float, enabled: bool) -> float:
    if enabled and 0.0 < mach < 0.7:
        return float(np.sqrt(1.0 - mach * mach))
    return 1.0


def _separation_warnings(ibl: IBLSolution) -> list[str]:
    notes: list[str] = []
    if np.isfinite(ibl["separation_s_upper"]):
        notes.append(
            f"Boundary layer separated on the upper surface at s={ibl['separation_s_upper']:.6g}."
        )
    if np.isfinite(ibl["separation_s_lower"]):
        notes.append(
            f"Boundary layer separated on the lower surface at s={ibl['separation_s_lower']:.6g}."
        )
    return notes


def _retained_laminar_index(
    previous: IBLSolution | None,
    arc: np.ndarray,
    separation_key: str,
    transition_key: str,
) -> int | None:
    """Station index of a laminar separation retained from the previous pass.

    Turbulent separation is not retained. A separation at or downstream of
    transition was not the laminar-freeze branch.
    """

    if previous is None:
        return None
    separation = float(previous[separation_key])  # type: ignore[literal-required]
    transition = float(previous[transition_key])  # type: ignore[literal-required]
    if not np.isfinite(separation):
        return None
    if np.isfinite(transition) and separation >= transition:
        return None
    stations = np.asarray(arc, dtype=np.float64)
    index = int(np.argmin(np.abs(stations - separation)))
    if abs(float(stations[index]) - separation) > 1e-6:
        return None
    return index


def _delta_residual(current: IBLSolution, previous: IBLSolution | None, chord: float) -> float:
    if previous is None:
        values = np.concatenate((current["delta_star_upper"], current["delta_star_lower"]))
        return float(np.max(np.abs(values)) / chord)
    delta = np.concatenate(
        (
            current["delta_star_upper"] - previous["delta_star_upper"],
            current["delta_star_lower"] - previous["delta_star_lower"],
        )
    )
    return float(np.max(np.abs(delta)) / chord)


class PhysicsEngine:
    """Viscous-inviscid coupled panel solver."""

    VERSION = "1.0.0"

    def solve(self, physics_input: PhysicsInput) -> PhysicsOutput:
        mesh = copy_mesh(physics_input["mesh"])
        return self._solve_copied(mesh, physics_input["freestream"], physics_input.get("parameters"))

    def solve_from_mesh(
        self,
        mesh: MeshOutput,
        freestream: FreestreamConditions,
        parameters: PhysicsParameters | None = None,
    ) -> PhysicsOutput:
        return self._solve_copied(copy_mesh(mesh), freestream, parameters)

    def _solve_copied(
        self,
        mesh: MeshOutput,
        freestream: FreestreamConditions,
        parameters: PhysicsParameters | None,
    ) -> PhysicsOutput:
        try:
            return self._solve_inner(mesh, freestream, parameters)
        except (InvalidFlowConditionError, PhysicsEngineError):
            raise
        except (FloatingPointError, ZeroDivisionError, IndexError, np.linalg.LinAlgError) as exc:
            raise PhysicsEngineError(str(exc)) from exc

    def _solve_inner(
        self,
        mesh: MeshOutput,
        freestream: FreestreamConditions,
        parameters: PhysicsParameters | None,
    ) -> PhysicsOutput:
        mode = mesh["mode"]
        if mode not in ("2D", "3D"):
            raise InvalidFlowConditionError(f"Mesh mode {mode!r} is not 2D or 3D.")
        chosen = resolve_parameters(mesh, parameters)
        warnings: list[str] = []
        regime = chosen["regime"]
        if mode == "3D" and regime == "viscous_inviscid" and mesh["ibl_1d"] is None:
            warnings.append("3D boundary layer skipped: mesh has no ibl_1d stations")
            regime = "inviscid"
        flow = _validate_freestream(freestream, regime, bool(chosen["prandtl_glauert"]))
        beta = _beta(flow["mach"], bool(chosen["prandtl_glauert"]))
        compressibility = beta != 1.0
        alpha = float(np.deg2rad(flow["alpha_deg"]))

        if mode == "2D":
            self._require_2d(mesh)
            output = self._solve_2d(mesh, flow, chosen, warnings, regime, alpha, beta, compressibility)
        else:
            self._require_3d(mesh)
            output = self._solve_3d(mesh, flow, chosen, warnings, regime, alpha, compressibility)
        return validate_physics_output(output)

    def _solve_2d(
        self,
        mesh: MeshOutput,
        flow: FreestreamConditions,
        chosen: dict,
        warnings: list[str],
        regime: str,
        alpha: float,
        beta: float,
        compressibility: bool,
    ) -> PhysicsOutput:
        panel = mesh["panel_2d"]
        grid = mesh["cartesian_2d"]
        ibl_mesh = mesh["ibl_1d"]
        assert panel is not None and grid is not None and ibl_mesh is not None
        chord = float(panel["chord"])
        transpiration = np.zeros(panel["length"].shape, dtype=np.float64)
        inviscid: InviscidSolution | None = None
        ibl: IBLSolution | None = None
        previous: IBLSolution | None = None
        residual = 0.0
        converged = regime == "inviscid"
        iterations = 0
        limit = int(chosen["max_vic_iterations"]) if regime == "viscous_inviscid" else 1
        influence_panel = panel if beta == 1.0 else None
        for iterations in range(1, limit + 1):
            inviscid, ue_upper, ue_lower = solve_2d(
                panel,
                alpha,
                transpiration,
                chosen["panel_model_2d"],
                bool(chosen["kutta"]),
                beta=beta,
                ibl=ibl_mesh if regime == "viscous_inviscid" else None,
            )
            gamma_now = inviscid["gamma"]
            cp_now = inviscid["cp_inviscid"]
            spiked = gamma_now is not None and float(np.max(np.abs(gamma_now))) > 8.0
            spiked = spiked or float(np.min(cp_now)) < -20.0
            if regime == "viscous_inviscid" and spiked:
                inviscid, ue_upper, ue_lower = solve_2d(
                    panel,
                    alpha,
                    np.zeros(panel["length"].shape, dtype=np.float64),
                    chosen["panel_model_2d"],
                    bool(chosen["kutta"]),
                    beta=beta,
                    ibl=ibl_mesh,
                )
                warnings.append(
                    "Viscous transpiration limited after a non-physical vortex spike; "
                    "surface pressure was reset to the inviscid field."
                )
                converged = False
                break
            if beta != 1.0 and influence_panel is None:
                stretched = np.array(panel["nodes"], copy=True)
                stretched[:, 0] *= beta
                influence_panel = build_panel_metrics_2d(stretched, clustering=panel["clustering"])
            if regime == "inviscid":
                residual = 0.0
                converged = True
                break
            assert ue_upper is not None and ue_lower is not None
            ibl, wall = march_ibl(
                ibl_mesh,
                ue_upper,
                ue_lower,
                flow["reynolds"],
                chord,
                panel,
                closure=chosen["ibl_closure"],
                fail_on_separation=bool(chosen["fail_on_separation"]),
                retained_upper=_retained_laminar_index(
                    previous, ibl_mesh["s_upper"], "separation_s_upper", "transition_s_upper"
                ),
                retained_lower=_retained_laminar_index(
                    previous, ibl_mesh["s_lower"], "separation_s_lower", "transition_s_lower"
                ),
            )
            # Under-relax the wall blowing so a separating station cannot
            # overturn the panel solution on the next iteration.
            transpiration = 0.25 * np.asarray(wall, dtype=np.float64) + 0.75 * transpiration
            residual = _delta_residual(ibl, previous, chord)
            previous = ibl
            converged = residual < float(chosen["vic_tolerance"])
            if converged:
                break
        if inviscid is None:
            raise PhysicsEngineError("2D solve did not produce an inviscid solution.")
        if ibl is not None and ibl["separated"]:
            warnings.extend(_separation_warnings(ibl))
        if regime == "viscous_inviscid" and not converged:
            warnings.append(
                f"Viscous-inviscid iteration stopped at vic_iterations={iterations} "
                f"with residual={residual:.3e}."
            )
        field_panel = panel if influence_panel is None else influence_panel
        gamma = inviscid["gamma"]
        assert gamma is not None
        field = evaluate_farfield_with_freestream(
            field_panel,
            gamma,
            grid,
            alpha,
            beta=beta,
            sigma=inviscid["sigma"],
        )
        loads = integrate_loads(mesh, inviscid, ibl, alpha, compressibility)
        return {
            "mode": "2D",
            "regime": regime,
            "freestream": flow,
            "inviscid": inviscid,
            "ibl": ibl,
            "vic_iterations": int(iterations),
            "residual": float(residual),
            "converged": bool(converged),
            "loads": loads,
            "field_2d": field,
            "surface_velocity": None,
            "warnings": warnings,
        }

    def _solve_3d(
        self,
        mesh: MeshOutput,
        flow: FreestreamConditions,
        chosen: dict,
        warnings: list[str],
        regime: str,
        alpha: float,
        compressibility: bool,
    ) -> PhysicsOutput:
        panel = mesh["panel_3d"]
        assert panel is not None
        count = int(panel["areas"].shape[0])
        if count > int(chosen["dense_panel_limit"]):
            warnings.append(
                f"panel count {count} exceeds dense_panel_limit; using GMRES"
            )
        inviscid, velocity = solve_3d(
            panel,
            alpha,
            kutta=bool(chosen["kutta"]),
            wake_length_chords=float(chosen["wake_length_chords"]),
            dense_panel_limit=int(chosen["dense_panel_limit"]),
            model=chosen["panel_model_3d"],
        )
        if compressibility:
            inviscid["cp_inviscid"] = np.ascontiguousarray(
                inviscid["cp_inviscid"] / _beta(flow["mach"], True),
                dtype=np.float64,
            )
        ibl = None
        loads = integrate_loads(mesh, inviscid, ibl, alpha, compressibility)
        if ibl is None:
            loads["cd_friction"] = float("nan")
            loads["cd"] = loads["cd_pressure"]
        return {
            "mode": "3D",
            "regime": regime,
            "freestream": flow,
            "inviscid": inviscid,
            "ibl": ibl,
            "vic_iterations": 1,
            "residual": 0.0,
            "converged": True,
            "loads": loads,
            "field_2d": None,
            "surface_velocity": velocity,
            "warnings": warnings,
        }

    @staticmethod
    def _require_2d(mesh: MeshOutput) -> None:
        missing = [
            name
            for name in ("panel_2d", "cartesian_2d", "ibl_1d")
            if mesh[name] is None  # type: ignore[literal-required]
        ]
        if missing:
            raise InvalidFlowConditionError(
                "2D physics requires " + ", ".join(missing) + "."
            )

    @staticmethod
    def _require_3d(mesh: MeshOutput) -> None:
        if mesh["panel_3d"] is None:
            raise InvalidFlowConditionError("3D physics requires panel_3d.")


def split_flow_payload(payload: dict) -> tuple[FreestreamConditions, PhysicsParameters]:
    """Split a catalog dict into freestream fields and solver parameters."""

    freestream = {key: payload[key] for key in _FREESTREAM_KEYS if key in payload}
    parameters = {key: value for key, value in payload.items() if key not in _FREESTREAM_KEYS}
    return freestream, parameters  # type: ignore[return-value]
