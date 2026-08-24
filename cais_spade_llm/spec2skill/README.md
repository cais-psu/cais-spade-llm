# Spec2Skill

**ICRA 2027 Case Study**

Spec2Skill is the isolated workspace for
**Spec2Skill: A Multi-Agent Framework for Primitive Composition in Robotic Assembly**.
The current scene-only milestone contains the project skeleton, research scope,
future implementation plan, source references, a dedicated no-hardware NIST
Gazebo scene launcher, and a local placeholder chat.

## MUST: Do not leak the answer

Allowed recognition inputs are only the user requirement, approved NIST
documents, approved candidate CAD files, RGB, depth, and camera calibration.
Gazebo model names, Gazebo entity names, world or SDF contents, spawn manifests,
configured spawn poses, `/gazebo/model_states`, `/get_entity_state`, current
detector responses, and evaluator labels are forbidden recognition inputs.

Candidate CAD filenames and document part names are allowed because they belong
to the supplied runtime corpus. The system must still determine which observed
object matches which candidate and where it belongs. Ground truth may be read
only by a separate evaluator after the prediction is finalized. Recognition
code must not import, invoke, or share runtime objects with that evaluator. Any
experiment that violates this boundary is invalid and must not be reported.

## Working boundary

All new Spec2Skill code belongs in this directory. ProductAgent, ResourceAgent,
CCA, and RobotAgent remain shared runtime authorities and will be reached through
future adapters owned here. They are not copied or modified. The dual Gazebo UI
uses a narrow adapter protocol; `SystemBridge` and `cais_spade_llm/ui/bridge.py`
remain outside the Spec2Skill package and are not modified.

The existing application entrypoint remains:

```bash
poetry run python -m cais_spade_llm.ui_main
```

The starting page is available at `/spec2skill` through the `Spec2Skill`
navigation item.

The page can start and stop the no-hardware `gazebo_dual_spec2skill` simulation,
which launches `table_spec2skill.world` with Gazebo, MoveIt, and RViz while
forcing `run_perception:=false`. Its User Interaction chat only displays a local
placeholder response and does not plan or execute the submitted request.

The dedicated scene uses the actual NIST plate, pin, gear fixture, shaft, and
gear STL visuals. `Gear_Plate` and three `Gear_Shaft` instances are pre-installed
as a static fixture while the three loose gears remain on `prusa_mk4_2`. The
plate collisions use their meshes; pin and shaft collisions use exact-size boxes
or cylinders. This scene-only milestone makes no recognition, planning,
insertion-physics, or robot-execution claim.

## Folder guide

- `adapters/`: narrow runtime connections, beginning with
  `gazebo_dual_spec2skill`.
- `cases/`: case-study inputs, beginning with `product requirement: assemble MCP`.
- `references/`: repository paths and provenance for manuals/specification/CAD.
- `schemas/`: future data templates and contracts; none are defined in Phase 0.
- `tests/fixtures/`: future controlled, repeatable inputs.
- `artifacts/`: ignored generated experiment results, traces, logs, and outputs.

## Prompting Codex

Start a new request with:

> I am working on the ICRA 2027 Spec2Skill case study. Restrict searches and
> edits to `cais_spade_llm/spec2skill/`. Do not inspect or modify `bridge.py` or
> existing agent implementations. The `gazebo_dual_spec2skill` adapter is the
> only authorized runtime integration. The MUST do-not-leak boundary applies to
> every recognition experiment. Ask before changing anything outside the
> Spec2Skill folder, except for the dedicated world and exact launch registration
> files already owned by this scene-only milestone.

The repository `AGENTS.md` also directs Spec2Skill and ICRA 2027 tasks into this
directory. Start a new Codex session after changing instruction files so the
instruction chain is loaded again.
