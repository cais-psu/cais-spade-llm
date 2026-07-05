# Contributing

This repository is a robotics and multi-agent manufacturing system. Keep changes small, reviewable, and tied to the exact runtime path being changed.

## Before a change

- Read `AGENTS.md`.
- Check `git status --short`.
- Identify the affected path before editing.
- Do not canonicalize, normalize, generalize, rename, or replace terms.
- Preserve identifiers, variable names, actions, resources, states, predicates, and domain-specific wording exactly.

## Making changes

- Keep `SystemBridge` as the public UI-to-runtime surface unless the change explicitly requires a public interface change.
- Keep ROS2 imports lazy when a Python path must run without ROS2.
- Do not mix cleanup with unrelated behavior changes.
- Do not add broad silent exception handling.
- Add docstrings for touched public modules, classes, functions, and methods.
- Use comments for non-obvious intent and constraints, not for obvious statements.

## Verification

- Python-only cleanup: run targeted `python -m pytest ...` commands for the touched area.
- `digital twin`, `Teach`, `Replay in Twin`, or ROS2 launch changes: run `python -m pytest test/test_ur5e_rg2_rtde_gripper.py`.
- ROS2 launch/script/RViz changes: run `make bootstrap-gazebo` before installed workspace checks.
- If live ROS2 or hardware checks are unavailable, report that explicitly.

## Pull requests

- Summarize the behavior changed.
- List the verification commands run.
- Call out any runtime checks that were not possible.
- Keep README changes focused on setup and run instructions.
