# math_module samples

Live catalog: `catalog.json`.

Fixtures cover cosine, half-cosine, and uniform 2D clustering, an imported SVG section, an open swept wing, a closed cylinder with outward and inward normals, and three guards that must fail:

- farfield box that does not contain the airfoil (`InvalidMeshConfigError`)
- a zero-area triangle (`DegeneratePanelError`)
- `require_closed_3d` on an open wing (`OpenSurfaceError`)

Pipeline cases in `datasets/pipeline/catalog.json` carry a `stages.math` payload and list `math` in `active_stages`. `shared.pipeline_runner.run_pipeline_case` runs geometry into `MeshEngine`. `run_math_module_sample` runs one fixture from this catalog.
