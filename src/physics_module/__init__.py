"""Physics solver (Module 3) — viscous-inviscid panel method."""

from .physics_engine import PhysicsEngine
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
    "PhysicsEngine",
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
