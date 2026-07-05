# Code Review Checklist

Use this checklist when reviewing changes in this repository.

## Scope

- Confirm the diff only changes files needed for the request.
- Confirm unrelated dirty worktree changes were not rewritten.
- Confirm no terms, identifiers, variable names, actions, resources, states, predicates, or domain wording were canonicalized, normalized, generalized, renamed, or replaced.
- Confirm `digital twin`, `Teach`, `Replay in Twin`, `Preview in Gazebo`, `Monitor`, and `dual_robots` wording remains exact when touched.

## Behavior

- Check whether a refactor accidentally changes public behavior.
- Check whether UI code still calls `SystemBridge` through the existing public surface.
- Check whether ROS2 imports remain lazy in files that must run without ROS2.
- Check whether validation-stage checks remain separate from runtime gating.
- Check whether broad `except Exception: pass` blocks were added. If an existing block is touched, prefer logging or explicit failure details.

## Readability

- Prefer smaller functions and clear data flow over comments that restate code.
- Add docstrings for public modules, classes, functions, and methods when touching them.
- Add comments only for non-obvious intent, hardware safety assumptions, ROS2 timing constraints, `digital twin` authority, `Teach`, `Replay in Twin`, `Preview in Gazebo`, or validation-stage behavior.
- Do not add line-by-line comments for obvious statements.

## Verification

- For Python-only changes, run the targeted `python -m pytest ...` command for the touched area.
- For `digital twin`, `Teach`, `Replay in Twin`, or ROS2 launch changes, run `python -m pytest test/test_ur5e_rg2_rtde_gripper.py`.
- For ROS2 launch/script/RViz changes, run `make bootstrap-gazebo` before installed workspace checks.
- If a required runtime check cannot run locally, state exactly what did not run and what static or unit checks did run.
