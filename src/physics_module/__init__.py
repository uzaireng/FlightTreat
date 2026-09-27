"""Physics solver (Module 3) — draft contract only."""

from .physics_skeleton import (
    BoundaryLayerSeparationError,
    FreestreamConditions,
    InvalidFlowConditionError,
    InviscidSolution,
    IBLSolution,
    PhysicsEngineError,
    PhysicsEngineProtocol,
    PhysicsInput,
    PhysicsOutput,
    PhysicsParameters,
    SingularInfluenceMatrixError,
)

__all__ = [
    "BoundaryLayerSeparationError",
    "FreestreamConditions",
    "InvalidFlowConditionError",
    "InviscidSolution",
    "IBLSolution",
    "PhysicsEngineError",
    "PhysicsEngineProtocol",
    "PhysicsInput",
    "PhysicsOutput",
    "PhysicsParameters",
    "SingularInfluenceMatrixError",
]
