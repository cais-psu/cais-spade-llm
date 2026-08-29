# Spec2Primitives

**ICRA 2027 Case Study**

Spec2Primitives is the isolated workspace for
**Spec2Primitives: A Multi-Agent Framework for Dynamic Primitive Composition in Industrial Robotic Assembly**.
The current implementation contains the project skeleton, research scope,
implementation plan, approved source references, demand-driven RGB-D context
tools, a dedicated no-hardware NIST Gazebo scene launcher, a Phase 2 PA
interaction UI connected through Phase 3.5, exact product-requirement intake,
requested-context serving, resumable clarification, and a tamper-checked PA
grounding-completion boundary. Production grounding starts from the exact
requirement, authoritative ontology, available sources, and retrieved typed
records together. PA returns only one `retrieve`, `inspect`, `propose_grounding`,
`ask_user`, or `incomplete` action. `GroundingSession` schema version 2 stores
the host-owned action journal and resolved provider metadata; it contains no
intermediate understanding, statements, information needs, or transitions.
On `propose_grounding`, PA makes one late `OntologyGroundingProposal` containing
TBox-level facts, one cited context summary, and missing information. The
deterministic validator remains the only ABox commit authority. Phase 4.1 supplies cached ontology-neutral
`DocumentOverviewRecord` previews and a question-targeted, full-document
`DocumentEvidenceRecord` fallback. Phase 4.2A adds assertion-free approved-CAD and
four-camera RGB-D preprocessing. Phase 4.2B1 adds an automatic observation-only
capture, preprocessing, and minimal segmentation path. Phase 4.2B2A adds strict
size-only matching against one requested approved CAD and returns a unique
candidate center in its camera optical frame. The remaining simple Phase 4.2B2
pose increment adds deterministic generalized CAD registration for loose source
candidates and returns an accepted, ambiguous, or rejected camera-frame pose.
Phase 4.3 adds the production pre-RA `GroundingSession`, provider capability
descriptors, `TypedContextBinding`, `ProductContextView`, and
`PAContextGroundingCompletion` version 2 contracts. The UI exposes complete,
waiting, incomplete, and ontology-gap states plus the accepted assertions,
typed records, actions, and timeline read-only. The live loop remains
fail-closed unless an authoritative
TBox and `OPENAI_API_KEY` configure that runtime. The project-authoritative
schema-only TBox is `ontology/spec2primitives_ppr_tbox.owl`; paired
`SPEC2PRIMITIVES_PPR_TBOX_PATH` and `SPEC2PRIMITIVES_PPR_NAMESPACE` values may
override it.

The current approved document corpus is one six-page NIST PDF. Targeted
inspection sends those six cached pages together in order; it does not use RAG,
embeddings, or page ranking. Retrieval for larger documents remains future work.

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

### Approved manual lifecycle

1. Register the PDF explicitly in
   `references/products/approved_sources.json`.
2. Prepare its ontology-neutral overview cache with either
   `poetry run python -m cais_spade_llm.spec2primitives.tools.document_evidence.prepare --context-ref <exact-ref>`
   or `--all`.
3. Start the system. F5 validates sources and reports cache state but performs
   no ProductAgent, LLM, or VLM inference request.
4. Run generalized PA grounding. PA receives the requirement, ontology,
   available sources, and retrieved typed records together, then chooses only
   `retrieve`, `inspect`, `propose_grounding`, `ask_user`, or `incomplete`.
5. On `propose_grounding`, create and validate the ontology projection plus one
   cited final context summary. Future RA receives that projection, the typed
   grounding contract, and hash-pinned typed evidence records.

Overview caches are generated under `contexts/source_cache/`, keyed by PDF
SHA-256, document-model settings, and overview-schema version, and are not
committed. A new or changed manual needs no document-purpose prompt or
product-specific code.

The page can start and stop the no-hardware `gazebo_dual_spec2primitives` simulation,
which launches `table_spec2primitives.world` with Gazebo, MoveIt, and RViz while
forcing `run_perception:=false`. Its Phase 2 PA interaction workspace exposes
the Phase 3.1--3.5 workflow with its nested Phase 4 grounding contract. It loads
the project-authoritative `ontology/spec2primitives_ppr_tbox.owl` by default and
uses the production grounding runtime when model configuration and
`OPENAI_API_KEY` are present. Invalid or partial TBox overrides and missing
model authority display `grounding_unavailable`. Pending user-intent questions can be answered
or cancelled in the same interaction; completion is shown only from a verified
`PAContextGroundingCompletion` version 2. Earlier completion formats are not
loaded or migrated. Controlled tests use a schema-only fixture,
deterministic model responses, and producer inputs to exercise the complete
orchestration record. A separate Phase 4.1 diagnostic can select any PDF
registered in `references/products/approved_sources.json` and displays the
overview, PA action journal, targeted evidence, untrusted
ontology proposal, and accepted assertions as separate stages. It does not
authorize PA completion. RGB-D capture, preprocessing, and minimal
segmentation run automatically only when the observation pipeline is invoked.
The operator card is read-only and displays `idle`, `running`, `ready`, or
`failed`, source and assembly candidate counts, CAD-correspondence and location
states, compact pose state, and robot-frame conversion state. It has no CAD,
coordinate, score, timeout, camera-role, threshold, mask, or artifact controls.
The geometry status card remains `idle` until an authorized runtime caller
invokes the supporting path. CAD
preprocessing stays separate and size association processes only the exact CAD
provided by its future caller. A controlled pose caller can update the same card
with only the compact pose state; coordinates and rotations remain in the typed
record. A separate controlled caller can inject one approved camera-to-robot
calibration and persist a robot-frame transform while the card exposes only its
status. Pre-RA PA grounding stops at an accepted camera-frame pose and does not
perform robot-frame conversion. No cross-camera transform, Phase 5 planning,
RA communication, or robot execution occurs.

## Planned dynamic PA/RA workflow

The downstream RA workflow is proposed architecture, not current runtime
behavior. PA now dynamically grounds only the evidence-backed context needed for a
robot-independent `TaskTransitionContract`. After inherited allocation, the
selected RA will load its complete current primitive-only catalog, whose
cardinality is determined at runtime, and author a structural
`PrimitiveProgramDraft`. Missing robot-owned inputs remain local to RA; missing
product or scene inputs return to PA in a deduplicated `MissingContextBatch`.
PA may invoke several auditable single-source producers before returning a
versioned `CompositionContextBundle`. Batched rounds have no fixed semantic
count, but repeated, ambiguous, unsupported, or non-progressing requests stop
fail-closed. Phase 5, the RA workflow, validation, and execution are not
implemented.

The dedicated scene uses the actual NIST plate, pin, gear fixture, shaft, and
gear STL visuals. `Gear_Plate` and three `Gear_Shaft` instances are pre-installed
as a static fixture while the three loose gears remain on `prusa_mk4_2`. The
plate collisions use their meshes; pin and shaft collisions use exact-size boxes
or cylinders. This scene-only milestone makes no recognition, planning,
insertion-physics, or robot-execution claim.

## Folder guide

- `agents/pa/`: the ProductAgent boundary, exact context serving, generalized
  grounding sessions, late ontology mapping, completion v2, PA-owned product
  context, and the production pre-RA runtime.
- `agents/ra/`: future Spec2Primitives RobotAgent adapter and RA workflow code.
- `adapters/`: narrow runtime connections, beginning with
  `gazebo_dual_spec2primitives`.
- `cases/`: case-study inputs, beginning with
  `product requirement: assemble Medium Gear`.
- `ontology/`: supporting infrastructure for semantic interoperability:
  immutable PPR TBox loading, profile validation, fingerprinting, and
  class-hierarchy queries. It does not route evidence, understand tasks,
  compose primitives, or understand documents, and it owns no writable ABox.
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
