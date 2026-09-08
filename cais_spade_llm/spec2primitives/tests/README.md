# Tests

Use focused offline suites for the changed boundary. Fixtures validate contracts and failure behavior; they do not measure VLM accuracy or prove absence of bias.

## Grounding and evidence coverage

- `test_pa_production_grounding.py`: native investigation, exact source authority, sensor metadata exclusion, process selection, state values, required arm reachability and no Cartesian requirement.
- `test_pa_completion.py`: the unversioned proposal/evidence/selection/completion chain, both arm identities, complete arm coverage and recovery tampering.
- `test_pa_resource_grounding.py`: configured capability and live MoveIt planning, exact resource choice, rejected/unreachable locations, no substitution, direct current-evidence gate.
- `test_rgbd_segmentation.py` and `test_robot_frame_conversion.py`: opaque model handles with canonical crop provenance and analytic calibrated transforms.
- `test_pa_ui_connection.py`: ontology/images and caveats after reload, desired reference labels, selected arm/validation limits, direct desired-state galleries, contained statement previews, incompatible-record rejection, compact programs with `<unbound>` parameters, responsive validation and duplicate-click exclusion.
- `test_ra_context_handoff.py`: current evidence gate at activation/recovery/composition, exact JID, paired snapshots, direct context pins, preserved RA sequences and supplied parameters, omitted required parameters, invalid supplied references, immutable history and no draft prerequisite.

Existing observation, segmentation, CAD measurement, layout and calibration suites retain analytic geometry/provenance regressions. Internal transforms use canonical frames even though model-facing metadata is opaque. The pose diagnostic consumes the same unranked candidate measurements as production.

## Primitive-input coverage

The nearest RA and UI suites cover minimal composition contracts with unchanged authoritative snapshots, nested helper metadata, missing geometry, invalid supplied references/types, unresolved destination labels, configured frame/EE/TCP evidence, deferred results, conditional insertion outputs, exact RA decisions, immutable history and recovery-hint exclusion on initial and subsequent reads. They also exercise the shared catalog through existing RobotAgent recovery consumers and preserve the legacy extractor fields.

New identifier-separation cases check that `model_name` is absent from composition schemas and reports, including nested declarations and required lists. Supplied identifier arguments and removed helper-output references are rejected without repairing the submission. An older program keeps its request's original catalog for validation and UI rendering, while a subsequent mocked attempt uses the simplified interface. Full backend signatures and history bytes remain unchanged.

The UI callback tests use real worker threads for evidence checks/refresh, duplicate-click exclusion and page detachment. Keep those tests separate from live model or robot evaluation. [COMPOSITION_EVALUATION.md](../COMPOSITION_EVALUATION.md) defines the research experiments; [VALIDATION_AND_REVISION.md](../VALIDATION_AND_REVISION.md) describes the implemented bounded loop and subsequent physical work.

## Focused command

```bash
poetry run pytest -q \
  cais_spade_llm/spec2primitives/tests/test_ra_context_handoff.py \
  cais_spade_llm/spec2primitives/tests/test_pa_ui_connection.py \
  -k 'composition or compose_button or binding or nested_metadata or shared_target or context_only_robot_agent'
```

Run syntax/static checks only on touched files and `git diff --check`. Keep disposable artifacts outside the repository. Live VLM, ROS2/Gazebo and hardware experiments are separate checks and must be reported explicitly when run.

## Experiment boundary

[BIAS_VALIDATION.md](../BIAS_VALIDATION.md) specifies audits and repeated paired counterfactual trials. Score complete goals, current installation claims, task-role ambiguity, coordinate grounding, uncertainty and resource choice separately. Include rejected/incomplete runs in reported denominators. Never let offline fixtures become runtime answers or evaluation evidence.

Historical run files remain unchanged; incompatible records require a fresh interaction and cannot authorize new RA work. `test_primitive_refinement.py` covers selected dependency propagation, strict calculations and recovery parity, modeled movement/custody/outcome checks, source changes, freshness, bounded revisions and cancellation. PA tests verify supplemental retrieval cannot commit Phase 4. UI tests cover refinement and legacy composition across page detachment. Execution identifiers, contact physics and observed outcomes remain future coverage.
