# Tests

Use focused offline suites for the changed boundary. Fixtures validate contracts and failure behavior; they do not measure VLM accuracy or prove absence of bias.

## Grounding and evidence coverage

- `test_pa_production_grounding.py`: native investigation, exact source authority, sensor metadata exclusion, process selection, state values, required arm reachability and no Cartesian requirement.
- `test_pa_completion.py`: the unversioned proposal/evidence/selection/completion chain, both arm identities, complete arm coverage and recovery tampering.
- `test_pa_resource_grounding.py`: configured capability and live MoveIt planning, exact resource choice, rejected/unreachable locations, no substitution, direct current-evidence gate.
- `test_rgbd_segmentation.py` and `test_robot_frame_conversion.py`: opaque model handles with canonical crop provenance and analytic calibrated transforms.
- `test_pa_ui_connection.py`: ontology/images and caveats after reload, desired reference labels, selected arm/validation limits, direct desired-state galleries, contained statement previews and incompatible-record rejection.
- `test_ra_context_handoff.py`: current evidence gate at activation/recovery/draft, historical rejection, exact JID, paired snapshots, catalog symbols and immutable unbound drafts.

Existing observation, segmentation, CAD measurement, layout and calibration suites retain analytic geometry/provenance regressions. Internal transforms use canonical frames even though model-facing metadata is opaque. The pose diagnostic consumes the same unranked candidate measurements as production.

## Focused command

```bash
MPLCONFIGDIR=/tmp/spec2primitives-matplotlib PYTHONDONTWRITEBYTECODE=1 poetry run pytest -q -p no:cacheprovider   cais_spade_llm/spec2primitives/tests/test_pa_production_grounding.py   cais_spade_llm/spec2primitives/tests/test_pa_completion.py   cais_spade_llm/spec2primitives/tests/test_pa_resource_grounding.py   cais_spade_llm/spec2primitives/tests/test_pa_ui_connection.py   cais_spade_llm/spec2primitives/tests/test_ra_context_handoff.py
```

Run syntax/static checks only on touched files and `git diff --check`. Keep disposable artifacts outside the repository. Live VLM, ROS2/Gazebo and hardware experiments are separate checks and must be reported explicitly when run.

## Experiment boundary

[BIAS_VALIDATION.md](../BIAS_VALIDATION.md) specifies audits and repeated paired counterfactual trials. Score complete goals, current installation claims, task-role ambiguity, coordinate grounding, uncertainty and resource choice separately. Include rejected/incomplete runs in reported denominators. Never let offline fixtures become runtime answers or evaluation evidence.

Historical run files remain unchanged; incompatible records require a fresh interaction and cannot authorize new RA work. Future coverage will include binding/context exchange, executable validation and observed outcomes when those phases are implemented.
