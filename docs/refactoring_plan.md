# Gradual refactoring plan

Use this plan alongside the [refactoring guide](refactoring.md). Keep
`SystemBridge` as the public entry point and gradually extract its implementation
in small, independently verified changes.

## Assessment

The following measurements are the initial assessment snapshot from 2026-09-21,
including blank lines and comments. They are not maintained automatically.
Other work was in progress during the assessment; remeasure the affected files
before starting a later step.

| File | Lines | Refactoring concern |
| --- | ---: | --- |
| [bridge.py](../cais_spade_llm/ui/bridge.py) | 39,619 | Many responsibilities and extensive shared state |
| [ur5e_rtde_trajectory_server.py](../ros2/cais_lab_robotics/scripts/ur5e_rtde_trajectory_server.py) | 11,061 | `_execute_insert` alone spans 4,578 lines; physical execution |
| [product_recovery_controller.py](../cais_spade_llm/agents/intelligent_product/product_recovery_controller.py) | 9,030 | Recovery proposals, approval, validation, and execution |
| [control.py](../cais_spade_llm/ui/pages/control.py) | 8,137 | `_predefined_function_record_body` spans 4,529 lines |
| [gazebo_pick_place_controller.py](../cais_spade_llm/resources/robot/gazebo_pick_place_controller.py) | 7,248 | Configuration, target calculations, and execution |
| [multi_turn.py](../cais_spade_llm/agents/intelligent_product/replanner/llm_recovery/modes/multi_turn.py) | 5,900 | Recovery phase coordination and supporting logic |
| [digital_twin_sync.py](../ros2/cais_lab_robotics/scripts/digital_twin_sync.py) | 5,853 | Synchronization, worker lifecycle, and replay |
| [hardware_pick_place_controller.py](../cais_spade_llm/resources/robot/hardware_pick_place_controller.py) | 5,345 | Physical motion and cancellation |
| [multi_turn_outline_generation.py](../cais_spade_llm/agents/intelligent_product/replanner/llm_recovery/modes/multi_turn_outline_generation.py) | 4,693 | Candidate generation and validation |
| [keyboard_teleop.py](../ros2/cais_lab_robotics/scripts/keyboard_teleop.py) | 4,633 | Input handling, command dispatch, and timing |
| [robot_task_runtime.py](../cais_spade_llm/resources/robot/robot_task_runtime.py) | 4,208 | `execute_robot_task` spans 998 lines |
| [perception_manager.py](../cais_spade_llm/ui/perception_manager.py) | 3,478 | Camera, calibration, and process responsibilities |
| [dashboard.py](../cais_spade_llm/ui/pages/dashboard.py) | 3,455 | `render` spans 3,205 lines |

Additional large files include `multi_turn_prompts.py` (3,914),
`spec2primitives_ui.py` (3,975), `production_grounding.py` (3,856), and
`grounding_contracts.py` (3,005). Spec2Primitives refactoring remains a separate
task under its directory instructions.

`test_case3_recovery_dryrun.py` is also large at 6,270 lines. Preserve its
regression coverage; reorganizing tests has lower priority.

At assessment time, `SystemBridge` had 756 direct methods, including 239 public
methods, and initialized 142 instance attributes. Shared state, locks, and
lifecycle dependencies make a broad rewrite risky. File length alone does not
establish faulty behavior.

## First implementation step

- [x] Refresh `git status --short` and the focused test baseline; preserve ongoing edits.
- [x] Extract `_topic_publisher_count_from_output` and
  `_controller_states_from_list_controllers_output` into
  `cais_spade_llm/ui/ros2_processes.py`, retaining their exact names.
- [x] Keep the existing `SystemBridge` methods as thin static wrappers with
  identical signatures and return values.
- [x] Preserve the parsing logic, including malformed input, response-format
  precedence, and unknown controller states.
- [x] Extend `test/test_dual_robot_rviz_startup.py` with explicit expected results
  for publisher counts, missing values, controller response formats, ANSI
  escapes, empty input, and unknown states.
- [x] Complete verification and include this extraction in its own commit.

The 17 new input cases passed against the original `SystemBridge` implementation
before extraction. After extraction, the same cases exercise both the bridge
wrappers and the helpers directly. In particular, unknown states remain present
in `ControllerState(...)` output and remain ignored in plain-text rows.

This step removes relatively little code, but establishes a checked extraction
pattern before touching larger responsibilities. Its commit should contain only
this document, the two parser wrappers, the two helpers, and their focused tests.

## Subsequent backlog, in order

| Order | Work | Boundary |
| --- | --- | --- |
| 1 | Extract the launch section of `control.py` into `ui/components/control_launch.py` | Move `_launch_section`, `_proc_row`, `_hardware_stack_row`, their constants, and required status helpers together. Preserve names and import shared helpers back into `control.py`; avoid circular imports. |
| 2 | Continue extracting stateless `bridge.py` calculations and data transformations | Use existing `digital_twin.py` and `ros2_processes.py` where responsibilities fit. Pass explicit inputs; retain `SystemBridge` wrappers. |
| 3 | Break down `_predefined_function_record_body`, `_teleop_section`, and `dashboard.render` | Separate rendering sections while preserving callback state, selection changes, timers, cleanup, and existing bridge calls. Recheck `dashboard.py` because other work may already have changed it. |
| 4 | Separate recording and bundle operations within `bridge.py` | Preserve paths, JSON fields, hashes, approval results, saved recordings, and error behavior. |
| 5 | Address recovery controllers and generation modules | Preserve exported methods, exact prompt content, state transitions, validation order, and persisted artifacts. |
| 6 | Address hardware controllers, synchronization, and teleop execution | First establish focused coverage of cancellation, contention, feedback freshness, failure cleanup, and execution ordering. |

Items after the first extraction are priorities for individually scoped
follow-up tasks, not one combined refactor.

Keep lifecycle ownership, locks, execution state, and `move_insert`
orchestration in their existing locations during the early steps. Preserve all
public interfaces and fixed symbols. Keep formatting cleanup and behavioral
fixes in separate changes.

## Verification and completion

Run the focused suites before and after this extraction:

```bash
poetry run pytest -q \
  test/test_dual_robot_rviz_startup.py \
  test/test_place_insert_release_only.py \
  test/test_ur5e_latency_contracts.py
```

The initial assessment baseline was 74 passed in 7.65 seconds. The implementation
baseline was refreshed before source changes: 74 passed in 7.62 seconds. These
are selected mocked and source-level checks, not comprehensive bridge coverage
or robot validation.

First-step verification on 2026-09-21:

- Focused suites after extraction: 108 passed in 8.66 seconds in the working tree.
- `poetry check`: passed with existing Poetry metadata deprecation warnings.
- `poetry run python -m compileall -q cais_spade_llm ros2`: passed.
- `git diff --check`: passed.
- An AST comparison confirmed that both extracted parser bodies match the
  originals, both bridge signatures and static-method decorators are unchanged,
  and no other bridge code changed structurally.
- No ROS2 launch/script/RViz or entrypoint changes were made by this step.
  Gazebo and physical-motion validation were not run.

For each extraction:

- Run relevant tests before and after the change.
- Run `poetry check`,
  `poetry run python -m compileall -q cais_spade_llm ros2`, and `git diff --check`.
- For UI component changes, check selected targets, callback arguments, refresh
  behavior, and client cleanup without issuing robot commands.
- Preserve existing source-inspection tests' assertions when updating their
  file locations.
- For entrypoint changes, run
  `poetry run python -m cais_spade_llm.ui_main --help`.
- For ROS2 launch/script/RViz changes, run `make bootstrap-gazebo` before checking
  installed behavior.
- Record verification and commit each completed slice separately, so it can be
  reverted independently. Inspect staged changes to exclude unrelated work,
  including unrelated edits in the same files.

Success means clearer responsibilities with unchanged observable behavior.
No Gazebo or physical-motion validation was performed during the assessment.
