# Refactoring Guide

Use this file only for cleanup, extraction, and readability work. Feature work
and bug fixes should stay focused on the touched runtime path unless a small
refactor is required to make the change safely.

## Before Editing

- Run `git status --short`.
- Read the smallest runtime path involved before changing it.
- Do not combine cleanup with unrelated feature or behavior changes.
- Do not canonicalize, normalize, generalize, rename, or replace fixed project
  terms, identifiers, actions, resources, states, predicates, or domain wording.

## How To Refactor

- Keep behavior unchanged unless the user explicitly asks for behavior changes.
- Prefer direct, human-readable code over generic abstractions.
- Add an abstraction only when it removes real duplication or matches an existing
  local pattern.
- Move code in small slices that can be reviewed and verified independently.
- Preserve `SystemBridge` as the callable UI-to-runtime surface unless the user
  asks for a public interface change.
- Keep ROS2 imports lazy in files that must run without ROS2.
- Keep configuration in JSON manifests under `initialization/` and
  `specification/`; do not replace it with hardcoded robot positions or
  capabilities.

## Comments And Docstrings

- Let clear names, type hints, and small functions carry the simple cases.
- Add Google-style docstrings to public modules, classes, functions, and methods
  when you touch them.
- Use comments only for non-obvious intent, safety assumptions, ROS2 timing
  constraints, validation-stage behavior, or runtime authority.
- Delete or update stale comments when touching nearby code.

## Error Handling

- Do not add silent failure paths.
- When touching a broad exception block, replace it with a narrow exception,
  logging, or an explicit return status when the surrounding behavior allows it.
- Keep safety-validation and recovery paths fail-closed unless the user asks for
  a different behavior.

## Verification

- For Python-only refactors, run `poetry check` and
  `poetry run python -m compileall -q cais_spade_llm ros2`.
- Add or run focused tests in `test/` when the refactor touches behavior that is
  easy to exercise locally.
- For entrypoint or CLI changes, run
  `poetry run python -m cais_spade_llm.ui_main --help`.
- For ROS2 launch/script/RViz changes, run `make bootstrap-gazebo` before
  installed workspace checks.
- State the code-verification versus runtime-verification boundary in the final
  response.
