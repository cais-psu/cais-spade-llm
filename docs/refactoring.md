# Refactoring Rules

Use this file for cleanup, extraction, and readability work.

## Before editing

- Run `git status --short`.
- Identify the smallest touched runtime path.
- Do not combine cleanup with unrelated feature or behavior changes.
- Do not rename fixed symbols: identifiers, variable names, actions, resources, states, predicates, or domain-specific wording.

## Extraction order

- Start with high-churn, high-size files only when the user's task touches them.
- For `cais_spade_llm/ui/bridge.py`, preserve `SystemBridge` as the callable public surface first.
- Move cohesive helper code in small slices using existing terms such as `digital_twin`, ROS2 launch/process handling, safety intent preview, bundle handling, and function recording.
- Keep imports lazy where ROS2 availability is optional.
- Keep tests passing after each slice.

## Comments and docstrings

- Do not add comments to every statement.
- Use docstrings for public modules, classes, functions, and methods when touched.
- Use comments to explain why a non-obvious constraint exists, especially hardware safety assumptions, ROS2 timing constraints, `digital twin` authority, `Teach`, `Replay in Twin`, `Preview in Gazebo`, and validation-stage behavior.
- Delete or update stale comments when touching the surrounding code.

## Error handling

- Do not add new silent failure paths.
- When touching `except Exception: pass`, replace it with logging, explicit return status, or a narrow exception if the surrounding behavior allows it.
- Keep fail-closed behavior for safety validation and recovery paths unless the user explicitly asks otherwise.

## Verification

- Use focused tests for extraction-only changes.
- Use `python -m pytest test/test_ur5e_rg2_rtde_gripper.py` for `digital twin`, `Teach`, `Replay in Twin`, or ROS2 launch behavior.
- Use `make bootstrap-gazebo` after ROS2 launch/script/RViz edits.
- State the code-verification versus runtime-verification boundary in the final response.
