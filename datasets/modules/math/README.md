# math_module samples

Placeholder. When Module 2 is integrated:

1. Add `catalog.json` here with mesh-only unit fixtures (grids, masks, singular-mesh traps).
2. Extend every case in `datasets/pipeline/catalog.json` with a `stages.math` block and append `"math"` to `active_stages`.
3. Update `shared/pipeline_runner.py` to consume `GeometryOutput` from the geo stage.
