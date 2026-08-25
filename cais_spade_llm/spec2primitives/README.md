# Spec2Primitives

**ICRA 2027 Case Study**

Spec2Primitives is the isolated workspace for
**Spec2Primitives: A Multi-Agent Framework for Dynamic Primitive Composition in Industrial Robotic Assembly**.
The current implementation contains the project skeleton, research scope,
implementation plan, approved source references, demand-driven RGB-D context
tools, a dedicated no-hardware NIST Gazebo scene launcher, a Phase 2 PA
interaction UI connected through Phase 3.3, the Phase 3.1 product-requirement
intake boundary, Phase 3.2 requested-context serving, and configurable Phase 3.3
ProductAgent context reassessment mechanics. The retrieval-first clarification
gate and all Phase 4 context understanding remain unimplemented.

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

All new Spec2Primitives code belongs in this directory. ProductAgent and RobotAgent
remain shared runtime authorities and will be reached through future adapters
under `agents/pa/` and `agents/ra/`. They are not copied or modified. The dual
Gazebo UI uses a narrow adapter protocol; `SystemBridge` and
`cais_spade_llm/ui/bridge.py` remain outside the Spec2Primitives package and are not
modified. ResourceAgent and CCA are outside the current Spec2Primitives roadmap.

The existing application entrypoint remains:

```bash
poetry run python -m cais_spade_llm.ui_main
```

The starting page is available at `/spec2primitives` through the `Spec2Primitives`
navigation item.

The page can start and stop the no-hardware `gazebo_dual_spec2primitives` simulation,
which launches `table_spec2primitives.world` with Gazebo, MoveIt, and RViz while
forcing `run_perception:=false`. Its Phase 2 PA interaction workspace can run
Phase 3.1 through Phase 3.3, display every persisted request and served result,
and stop for completion, clarification, failure, or the selected emergency turn
limit. Current Phase 3.3 behavior can still ask for clarification before all
relevant permitted context is retrieved; that is a documented pending correction,
not completed context understanding. The page does not run document-diagram VLM
interpretation, CAD/RGB-D grounding, planning, RA, CCA, or robot execution.

The dedicated scene uses the actual NIST plate, pin, gear fixture, shaft, and
gear STL visuals. `Gear_Plate` and three `Gear_Shaft` instances are pre-installed
as a static fixture while the three loose gears remain on `prusa_mk4_2`. The
plate collisions use their meshes; pin and shaft collisions use exact-size boxes
or cylinders. This scene-only milestone makes no recognition, planning,
insertion-physics, or robot-execution claim.

## Folder guide

- `agents/pa/`: the Phase 3.1 ProductAgent composition boundary, Phase 3.2 exact
  context serving, Phase 3.3 reassessment loop, and future PA workflow code.
- `agents/ra/`: future Spec2Primitives RobotAgent adapter and RA workflow code.
- `adapters/`: narrow runtime connections, beginning with
  `gazebo_dual_spec2primitives`.
- `cases/`: case-study inputs, beginning with
  `product requirement: assemble Medium Gear`.
- `tools/`: controlled retrieval, observation, and future perception tools.
- `references/products/`: static product documents and approved CAD refs.
- `references/resources/`: future static resource and primitive-catalog refs.
- `contexts/`: ignored per-interaction product and resource runtime context.
- `evaluations/`: isolated post-prediction evaluation and ground truth.
- `schemas/`: future data templates and contracts.
- `tests/fixtures/`: future controlled, repeatable inputs.
- `RESEARCH_POSITIONING.md`: terminology, Manual2Skill comparison, research
  gap, primitive definition, and claim boundaries for the paper.

## Prompting Codex

Start a new request with:

> I am working on the ICRA 2027 Spec2Primitives case study. Restrict searches and
> edits to `cais_spade_llm/spec2primitives/`. Do not inspect or modify `bridge.py` or
> existing agent implementations. The `gazebo_dual_spec2primitives` adapter is the
> only authorized runtime integration. The MUST do-not-leak boundary applies to
> every recognition experiment. Ask before changing anything outside the
> Spec2Primitives folder, except for the dedicated world and exact launch registration
> files already owned by this scene-only milestone.

The repository `AGENTS.md` also directs Spec2Primitives and ICRA 2027 tasks into this
directory. Start a new Codex session after changing instruction files so the
instruction chain is loaded again.
