"""
Aero engine façade.

Module 4 ka darwaza. Geometry, mesh, aur physics apni jagah rehte hain.
Yeh sirf un ke output ko report numbers mein reduce karta hai.
The Module 4 entry point. Geometry, mesh, and physics stay untouched.
This only reduces their output into report numbers.
"""

from __future__ import annotations

from .aero_skeleton import (
    AeroInput,
    AeroOutput,
    AeroParameters,
    AeroPolarInput,
    AeroPolarOutput,
    validate_aero_parameters,
)
from .operate import reduce_operating_point
from .polar import reduce_polar as _reduce_polar


class AeroEngine:
    """Reduce solved flow into coefficients, forces, and polars."""

    VERSION = "1.0.0"

    def analyze(self, aero_input: AeroInput) -> AeroOutput:
        parameters = validate_aero_parameters(aero_input.get("parameters"))
        return reduce_operating_point(aero_input["mesh"], aero_input["physics"], parameters)

    def analyze_from_solution(
        self,
        mesh,
        physics,
        parameters: AeroParameters | None = None,
    ) -> AeroOutput:
        """Same reduction as ``analyze``, with the three pieces passed loose."""

        return reduce_operating_point(mesh, physics, validate_aero_parameters(parameters))

    def reduce_polar(self, polar_input: AeroPolarInput) -> AeroPolarOutput:
        parameters = validate_aero_parameters(polar_input.get("parameters"))
        return _reduce_polar(polar_input["mesh"], list(polar_input["samples"]), parameters)
