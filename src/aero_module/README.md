# Aero Analytics (`aero_module`)

Draft contract only: `aero_skeleton.py`.

Consumes `MeshOutput` + `PhysicsOutput`. Emits \(C_L\), \(C_D = C_{D,p} + C_{D,f}\), \(C_M\), surface \(C_p\), aerodynamic center.

Module 3 ([PHYSICS_MODULE_SPEC.md](../../PHYSICS_MODULE_SPEC.md)) writes the pressure integral, friction integral, and surface \(C_p\) on `PhysicsOutput`. Module 4 reads those values. It does not assemble the influence matrix or march the boundary layer. Reference-quantity overrides, aerodynamic center, and \(L/D\) stay here.

Implementation not started.
