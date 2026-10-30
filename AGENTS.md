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

## Recovery result retention

- For Recovery UI work, read the current saved-result status and retained record
  identifiers in
  [IMPLEMENTATION_PLAN.md](<writing/Journal Paper 2 (recovery framework)/IMPLEMENTATION_PLAN.md>).
- Keep one accepted recovery outline and one accepted primitive composition for
  each of `Part slippage`, `Machining breakdown during part processing`,
  `Conveyor breakdown`, and `ur5e-1 breakdown`. Retain their required source
  observations, validation evidence, nominal-continuation proof, native models,
  and other dependencies.
- Preserve official failure captures, their required derived observations, and
  regression fixtures. A successful failure capture is not a failed recovery
  attempt, even when its run is `stopped` or has no recovery outline yet.
- After diagnosing a finished failed attempt, record its identifier, concise
  failure reason, and next step in the recovery handoff. Then delete its
  disposable outputs before starting another attempt or ending the task.
- Experiment records are the exception: never delete anything under
  `cais_spade_llm/monitor/environment_runs/recovery/outline_experiments/`
  (Experiments tab batches, their attempt drafts, audits, and manifests),
  whether an attempt finished, failed, stopped, did not finish, or was
  interrupted.
- Protect active work and the current partial attempt still needed for debugging,
  including work owned by another chat or process. Remove abandoned attempts once
  their useful findings are recorded and their required dependencies are retained.
- A failed primitive composition must never cause deletion of its accepted
  recovery outline or shared dependencies. Clean only the failed stage's
  disposable outputs when stages share an attempt directory.
- Verify and load the selected replacement before removing superseded accepted
  results. Do not treat an accepted event prefix, `NEEDS_CONTEXT`, or a successful
  failure capture as a complete accepted recovery outline or primitive composition.
- Before cleanup, build an explicit deletion list, check file ownership and
  recent changes, and exclude active or newly changed records and required
  dependencies. Preserve application code, regression coverage, and unrelated
  working or staged changes.
- After cleanup, confirm that retained results still load with their required
  evidence, that official failure captures remain selectable, and report the
  files removed and space released. Keep the retained identifiers and current
  stage status in the recovery handoff up to date.
- This is a Codex cleanup duty during recovery work. The Recovery UI does not
  automatically delete failed attempts; this rule does not authorize execution
  of recovery or establish completed live recovery.

## Spec2Primitives / ICRA 2027 isolation

- When a task mentions `Spec2Primitives` or `ICRA 2027`, begin in
  `cais_spade_llm/spec2primitives/` and read its `AGENTS.md` before doing any work.
- Keep searches scoped to `cais_spade_llm/spec2primitives/`; do not run repository-wide
  searches for a Spec2Primitives task unless the user explicitly expands the scope.
- Inspect `cais_spade_llm/ui/bridge.py` whenever needed; read-only inspection does
  not require user confirmation.
- Never modify `cais_spade_llm/ui/bridge.py` directly. Keep it read-only because
  it is already large. Put implementation changes in Spec2Primitives adapters
  or other approved modules instead of adding code to the bridge.
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

## No kinds or invented categories

- Do not invent "kinds" or other categorical groupings: no taxonomies of
  fields, lookups, refusal reasons or outcomes. Use the exact names that exist
  (declared state variables, primitive effects, reference paths, check names,
  the audit's outcome names) (user decision, 2026-10-30).
- The user does not want derivation "kinds" (user decision, 2026-10-30):
  labels such as `free`, `unchanged`, `place` or `bookkeeping` that sort
  end-state fields into categories, and above all any category that lets a
  field pass without being established. Do not add new kinds, and do not build
  on the existing ones.
- RA and CCA checks and primitive composition work from the declared state
  variables and the declared primitive effects that exist.
- Both existing uses are gone (2026-10-30): composition's `follows_from_event`,
  and the derivation labels in `ra_feasibility._Check.condition`, removed with
  the user's go. Every stated value is accepted only when it already holds, a
  declared effect of the acting resource's primitives makes it, or one of the
  event's capability checks finds it.

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

## Home and school development

- The school laptop is the main development machine. It holds the
  working code and runs ROS/Gazebo. Keep it plugged in, online, and awake.
- The home Windows laptop should remotely access the school laptop
  to edit the same files and run experiments.
- For recovery-framework work, use the school laptop's Ubuntu-22.04
  WSL checkout at /home/jongh/projects/cais-spade-llm on
  recovery-framework-journal.
- The school laptop's Windows checkout is a separate copy. Verify
  the machine, Linux path, branch, and git status before editing.
- VS Code Remote Tunnels is configured on the school Windows host as
  school-laptop, authenticated with GitHub, and installed as a background
  service. Use the same GitHub account on the home laptop.
- On the home laptop, install Remote - Tunnels in VS Code, run
  Remote Tunnels: Connect to Tunnel, and select school-laptop. Connect
  onward to the school laptop's Ubuntu-22.04 WSL environment and open
  /home/jongh/projects/cais-spade-llm.
- Chrome Remote Desktop Remote Access is enabled on the school Windows
  host as school-laptop. The user confirmed it shows Online, and the
  chromoting service is running with Automatic startup.
  Connect at https://remotedesktop.google.com/access using the same Google
  account and the private PIN chosen during setup. Do not store the PIN
  in this repository.
- Plugged-in sleep and hibernation are disabled; closing the lid while
  plugged in does nothing. Battery power settings are unchanged.
- tmux is installed in WSL. Use a tmux session for long experiments
  that should continue when the remote client disconnects.
- Home-to-school connectivity, remote WSL checkout access, Gazebo
  visibility, and reconnection after reboot still need verification.
- Git is for version history and backups; switching locations should
  not require a commit or push.
- Giving Codex access to the school laptop is a separate task.
  Do not assume the user's remote connection grants Codex access.
