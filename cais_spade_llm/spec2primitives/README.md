# Spec2Primitives

**ICRA 2027 Case Study**

Spec2Primitives is the isolated workspace for
**Spec2Primitives: A Multi-Agent Framework for Dynamic Primitive Composition in Industrial Robotic Assembly**.
The current implementation contains the project skeleton, research scope,
implementation plan, approved source references, demand-driven RGB-D context
tools, a dedicated no-hardware NIST Gazebo scene launcher, a Phase 2 PA
interaction UI connected through Phase 3.3, the Phase 3.1 product-requirement
intake boundary, Phase 3.2 requested-context serving, and contract-first Phase
3.3 ontology orchestration. The Phase 4.0 RDFLib foundation separates shared
immutable TBox semantics from PA-owned product context, initializes independent
interaction ABoxes before PA's first request, and validates evidence-backed fact
deltas inside the loop. Phase 4.1 adds an injected OpenAI document interpreter
and a separate diagnostic ABox. Phase 4.2A adds assertion-free approved-CAD and
four-camera RGB-D preprocessing. Phase 4.2B1 adds an automatic observation-only
capture, preprocessing, and minimal segmentation path. Phase 4.2B2A adds strict
size-only matching against one requested approved CAD and returns a unique
candidate center in its camera optical frame. The remaining simple Phase 4.2B2
pose increment adds deterministic generalized CAD registration for loose source
candidates and returns an accepted, ambiguous, or rejected camera-frame pose.
The UI remains status-only. Only
persisted Phase 4.3-style assessments can clarify or complete. The main live
loop fails closed until an authoritative TBox, registered complete grounding
producers, and Phase 4.3 grounding runtime are configured.

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
remain shared runtime authorities and are reached only through package-owned
adapters under `agents/pa/` and `agents/ra/`. The narrow PA context adapter is
implemented; the RA adapter and workflow remain future work. Neither shared
agent is copied or modified. The dual
Gazebo UI uses a narrow adapter protocol; `SystemBridge` and
`cais_spade_llm/ui/bridge.py` remain outside the Spec2Primitives package and are not
modified. Unused shared-agent subsystems are outside the current
Spec2Primitives roadmap.

The existing application entrypoint remains:

```bash
poetry run python -m cais_spade_llm.ui_main
```

The starting page is available at `/spec2primitives` through the `Spec2Primitives`
navigation item.

The page can start and stop the no-hardware `gazebo_dual_spec2primitives` simulation,
which launches `table_spec2primitives.world` with Gazebo, MoveIt, and RViz while
forcing `run_perception:=false`. Its Phase 2 PA interaction workspace exposes
the Phase 3.1--3.3 contract and displays `grounding_unavailable` because no
production ontology or complete grounding producers are configured. Controlled tests
inject a schema-only fixture and producer doubles to exercise the complete
orchestration record. A separate Phase 4.1 diagnostic can interpret the approved
NIST PDF when an authoritative TBox and `OPENAI_API_KEY` are configured. It does
not authorize PA completion. RGB-D capture, preprocessing, and minimal
segmentation run automatically only when the observation pipeline is invoked.
The operator card is read-only and displays `idle`, `running`, `ready`, or
`failed`, source and assembly candidate counts, CAD-correspondence and location
states, compact pose state, and robot-frame conversion state. It has no CAD,
coordinate, score, timeout, camera-role, threshold, mask, or artifact controls.
The production card remains
`idle` until an authorized runtime caller invokes the supporting path. CAD
preprocessing stays separate and size association processes only the exact CAD
provided by its future caller. A controlled pose caller can update the same card
with only the compact pose state; coordinates and rotations remain in the typed
record. A separate controlled caller can inject one approved camera-to-robot
calibration and persist a robot-frame transform while the card exposes only its
status. No cross-camera transform, PA integration, planning, RA, or robot
execution occurs.

## Planned dynamic PA/RA workflow

The target workflow is proposed architecture, not current runtime behavior. PA
will dynamically ground only the minimum evidence-backed context needed for a
robot-independent `TaskTransitionContract`. After inherited allocation, the
selected RA will load its complete current primitive-only catalog, whose
cardinality is determined at runtime, and author a structural
`PrimitiveProgramDraft`. Missing robot-owned inputs remain local to RA; missing
product or scene inputs return to PA in a deduplicated `MissingContextBatch`.
PA may invoke several auditable single-source producers before returning a
versioned `CompositionContextBundle`. Batched rounds have no fixed semantic
count, but repeated, ambiguous, unsupported, or non-progressing requests stop
fail-closed. Production Phase 4.3, Phase 5, the RA workflow, validation, and
execution are not implemented.

The dedicated scene uses the actual NIST plate, pin, gear fixture, shaft, and
gear STL visuals. `Gear_Plate` and three `Gear_Shaft` instances are pre-installed
as a static fixture while the three loose gears remain on `prusa_mk4_2`. The
plate collisions use their meshes; pin and shaft collisions use exact-size boxes
or cylinders. This scene-only milestone makes no recognition, planning,
insertion-physics, or robot-execution claim.

## Folder guide

- `agents/pa/`: the Phase 3.1 ProductAgent composition boundary, Phase 3.2 exact
  context serving, Phase 3.3 controlled grounding orchestration contracts,
  PA-owned Phase 4.0 product context, and future PA workflow code.
- `agents/ra/`: future Spec2Primitives RobotAgent adapter and RA workflow code.
- `adapters/`: narrow runtime connections, beginning with
  `gazebo_dual_spec2primitives`.
- `cases/`: case-study inputs, beginning with
  `product requirement: assemble Medium Gear`.
- `ontology/`: shared immutable PPR TBox loading, profile validation,
  fingerprinting, and class-hierarchy queries. It owns no writable ABox.
- `tools/`: controlled retrieval, observation, OpenAI document interpretation,
  Phase 4.2A CAD/RGB-D preprocessing, and Phase 4.2B1 minimal camera-local
  segmentation plus Phase 4.2B2A size-only candidate association tools.
- `references/products/`: static product documents and approved CAD refs.
- `references/resources/`: future static resource and primitive-catalog refs.
- `contexts/`: ignored per-interaction product and resource runtime context;
  PA interaction ABoxes and future separately owned RA resource ABoxes remain
  distinct here.
- `evaluations/`: isolated post-prediction evaluation and ground truth.
- `schemas/`: status and boundaries for current and future data contracts.
- `tests/fixtures/`: controlled, repeatable test-only inputs, beginning with
  ontology fixtures.
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
