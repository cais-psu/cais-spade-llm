# AGENTS.md

## Repository expectations

- Read this file before changing code in this repository.
- Keep changes scoped to the user's request and the touched runtime path.
- Run `git status --short` before editing and do not overwrite unrelated local changes.
- Preserve `SystemBridge` as the public UI-to-runtime surface unless the user explicitly asks for a public interface change.
- Do not refactor behavior while fixing a bug unless the refactor is required for the fix.
- For code review behavior, use `docs/code_review.md`.
- For cleanup and extraction work, use `docs/refactoring.md`.
- For boundary context, use `docs/architecture.md`.
- Follow the `Clean code standards` below; run `make lint` before finishing and do not add new lint findings.

## Fixed-symbol rule

- Do not canonicalize, normalize, generalize, or rename any terms.
- Do not replace specific names with generic labels.
- Do not introduce new abstractions, categories, or standardized forms.
- Keep all identifiers, variable names, and domain-specific wording exactly as provided.
- If a term is ambiguous or inconsistent, leave it unchanged instead of correcting it.
- Treat all tokens (actions, resources, states, predicates) as fixed symbols in a formal system.

## Runtime boundaries

- Keep `digital twin`, `Teach`, `Replay in Twin`, `Preview in Gazebo`, `Monitor`, and `dual_robots` wording exactly as written.
- Treat hardware MoveIt/RViz as the operator surface for `digital twin`; Gazebo is a passive mirror unless the user asks for a different architecture.
- Keep `Teach` as sim/recovery authoring and `Replay in Twin` as the path that commits the saved sim waypoint through hardware MoveIt.
- Keep validation-stage checks separate from downstream runtime gating when the user says `during validation process`.
- Do not import ROS2 modules at top level in Python files that must run without ROS2.

## Clean code standards

These are enforced by `ruff` (config in `pyproject.toml`), the `Makefile` targets,
`.pre-commit-config.yaml`, and the Claude Code hook in `.claude/settings.json`. The
repo carries a known lint backlog being burned down per `docs/refactoring.md` — do
not add to it.

- `make lint-report` shows current per-rule debt counts; `make lint` shows full
  findings; `make lint-fix` applies safe auto-fixes and formats.
- Do not add new `print()`. Use `logging.getLogger(__name__)` per module.
- Do not add new broad `except Exception` or bare `except`. Catch the narrowest
  exception and log or return an explicit status. Keep safety-validation and
  recovery paths fail-closed.
- Keep functions small. Ruff flags excessive complexity, branches, arguments, and
  statements (thresholds are generous now, tightened as files are decomposed).
- Type hints everywhere; start modules with `from __future__ import annotations`.
- `snake_case` for functions/variables, `PascalCase` for classes, private methods
  prefixed with `_`.
- Keep ROS2 imports lazy (never top-level) in files that must run without ROS2.
- Configuration-driven: robot positions and capabilities come from JSON manifests
  under `initialization/` and `specification/` — do not hardcode them.
- Never commit `.env` or credentials.

### Comments and docstrings

- Readability comes from small functions, clear names, and type hints — not from a
  comment on every line. Do not restate code in comments.
- Add Google-style docstrings to public modules, classes, functions, and methods
  when you touch them.
- Use comments only for non-obvious intent: hardware safety assumptions, ROS2
  timing constraints, `digital twin` authority, `Teach`, `Replay in Twin`,
  `Preview in Gazebo`, and validation-stage behavior.

## Verification

- For Python-only cleanup, run targeted `python -m pytest ...` commands for the touched area.
- For `digital twin`, `Teach`, `Replay in Twin`, or ROS2 launch changes, run `python -m pytest test/test_ur5e_rg2_rtde_gripper.py`.
- For ROS2 launch/script/RViz changes, run `make bootstrap-gazebo` before checking installed workspace behavior.
- If live ROS2 validation is unavailable, state that boundary explicitly and report the static or unit checks that did run.
