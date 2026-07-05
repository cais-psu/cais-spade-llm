# Code Review Checklist

Use this checklist when reviewing changes in this repository. Lead with bugs,
behavioral regressions, unsafe assumptions, and missing verification.

## Scope

- Confirm the diff only changes files needed for the request.
- Confirm unrelated dirty worktree changes were not rewritten.
- Confirm project terms, identifiers, variable names, actions, resources, states,
  predicates, and domain wording were not canonicalized, normalized,
  generalized, renamed, or replaced.
- Confirm public UI calls still go through `SystemBridge` unless the change
  intentionally changes the UI-to-runtime surface.

## Behavior

- Check whether the change alters runtime behavior that was supposed to stay
  stable.
- Check whether ROS2 imports remain lazy in Python paths that must run without
  ROS2.
- Check whether validation-stage checks remain separate from runtime dispatch
  gating.
- Check whether safety-validation and recovery paths still fail closed.
- Check whether broad silent failures were added. If an existing broad exception
  block is touched, prefer logging, explicit status, or a narrow exception.

## Readability

- Prefer direct, human-readable code that matches the nearby file.
- Prefer small functions and clear data flow over comments that restate code.
- Add docstrings for touched public modules, classes, functions, and methods.
- Add comments only for non-obvious intent, safety assumptions, ROS2 timing
  constraints, validation-stage behavior, or runtime authority.
- Do not introduce generic abstractions unless they remove real duplication or
  match an existing local pattern.

## Verification

- For normal Python changes, expect `poetry check` and
  `poetry run python -m compileall -q cais_spade_llm ros2`.
- For entrypoint changes, expect
  `poetry run python -m cais_spade_llm.ui_main --help`.
- For ROS2 launch/script/RViz changes, expect `make bootstrap-gazebo` before
  installed workspace checks.
- Run focused tests when they are created, restored, or already relevant to the
  touched behavior.
- If live ROS2 or hardware checks cannot run locally, the final report should
  say exactly what did not run and what static, import, compile, CLI, or focused
  checks did run.
