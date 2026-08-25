# AGENTS.md

## Project

CAIS-SPADE-LLM is a multi-agent manufacturing automation system. It combines
SPADE agents, LLM planning, safety validation, ROS2/Gazebo/MoveIt robot
integration, dual-robot execution, and a NiceGUI operator UI.

## Before editing

- Read the code path you are about to touch before changing it.
- Run `git status --short` and do not overwrite unrelated local changes.
- Keep changes scoped to the user's request and the runtime path involved.
- Preserve `SystemBridge` as the public UI-to-runtime surface unless the user
  explicitly asks for a public interface change.
- Do not mix unrelated refactors into feature work or bug fixes.

## Spec2Primitives / ICRA 2027 isolation

- When a task mentions `Spec2Primitives` or `ICRA 2027`, begin in
  `cais_spade_llm/spec2primitives/` and read its `AGENTS.md` before doing any work.
- Keep searches scoped to `cais_spade_llm/spec2primitives/`; do not run repository-wide
  searches for a Spec2Primitives task unless the user explicitly expands the scope.
- Do not inspect or modify `cais_spade_llm/ui/bridge.py` for Spec2Primitives work unless
  the user explicitly requests work on that file.
- Treat ProductAgent, ResourceAgent, CCA, and RobotAgent as shared, read-only
  runtime authorities. Connect to them only through adapters owned by
  `cais_spade_llm/spec2primitives/` after the user authorizes that integration work.
- Ask before modifying any file outside `cais_spade_llm/spec2primitives/` for a
  Spec2Primitives task.

## Fixed-symbol rule

- Do not canonicalize, normalize, generalize, rename, or replace project terms.
- Keep identifiers, variable names, actions, resources, states, predicates, and
  domain-specific wording exactly as provided.
- If a term is ambiguous or inconsistent, leave it unchanged instead of
  correcting it.
- Treat formal-system tokens as fixed symbols, not labels to standardize.

## How to code here

- Write clean, human-readable code that fits the nearby file.
- Prefer direct code over generic abstractions. Add an abstraction only when it
  removes real duplication or matches an existing local pattern.
- Use clear names, small functions, type hints, and explicit data flow.
- Start Python modules with `from __future__ import annotations` when touching
  them.
- Use `snake_case` for functions and variables, `PascalCase` for classes, and
  `_private_name` for private helpers.
- Use `logging.getLogger(__name__)`; do not add new `print()`.
- Catch narrow exceptions. Do not add bare `except` or broad silent failure
  paths.
- Keep ROS2 imports lazy in Python files that must also run without ROS2.
- Keep robot positions and capabilities configuration-driven through the JSON
  manifests under `initialization/` and `specification/`.
- Never commit `.env` files or credentials.

## Comments and docstrings

- Let readable code carry the simple cases.
- Add Google-style docstrings to public modules, classes, functions, and methods
  when you touch them.
- Use comments only when they explain non-obvious intent, safety assumptions,
  ROS2 timing constraints, validation-stage behavior, or runtime authority.
- Do not add comments that restate the next line of code.

## Verification

- For normal Python changes, run `poetry check` and
  `poetry run python -m compileall -q cais_spade_llm ros2`.
- For entrypoint or CLI changes, run
  `poetry run python -m cais_spade_llm.ui_main --help`.
- For ROS2 launch/script/RViz changes, run `make bootstrap-gazebo` before
  checking installed workspace behavior.
- Extend the nearest existing test suite for bug fixes.
- Create a new test file only for a coherent new subsystem.
- Place throwaway reproduction tests outside the repository and remove any
  temporary repository tests before handoff.
- Never delete regression coverage merely to reduce the test file count.
- Run any focused tests you create or restore for the touched feature.
- If live ROS2 or hardware validation is unavailable, say so and report the
  static, import, compile, CLI, or focused tests that did run.
