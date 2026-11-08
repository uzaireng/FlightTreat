# physics_module samples

Solver-only fixtures in `catalog.json`. They are not extra user journeys. The seven pipeline cases stay in `datasets/pipeline/` and now include `stages.physics`.

| Sample | Expectation |
|---|---|
| `bad_te_index` | `InvalidFlowConditionError` |
| `singular_collocation` | `SingularInfluenceMatrixError` |
| `thwaites_blasius` | Momentum thickness within 2% of `sqrt(0.45 ν s / U)` |
| `naca0012_alpha0` | Inviscid `|Cl| < 0.02` |

Run them with `run_physics_module_sample` from `src/shared/pipeline_runner.py`.

`naca0012_reference/` holds the Ladson, Gregory, Abbott, and McCroskey tables used by `tests/test_naca0012_accuracy.py`. Those files are published data, not cases in `catalog.json`.
