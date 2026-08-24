# ICRA 2027 Spec2Skill Scope

## Research title

**Spec2Skill: A Multi-Agent Framework for Primitive Composition in Robotic Assembly**

## Starting case

`product requirement: assemble MCP`

## MUST: Do not leak the answer

Allowed recognition inputs are only the user requirement, approved NIST
documents, approved candidate CAD files, RGB, depth, and camera calibration.
Forbidden recognition inputs are Gazebo model names, Gazebo entity names, world
or SDF contents, spawn manifests, configured spawn poses,
`/gazebo/model_states`, `/get_entity_state`, current detector responses, and
evaluator labels.

Candidate CAD filenames and document part names are allowed because they belong
to the supplied runtime corpus. The system must still determine which observed
object matches which candidate and where it belongs. Ground truth may be read
only by a separate evaluator after the prediction is finalized. Recognition
code must not import, invoke, or share runtime objects with the ground-truth
evaluator. Any experiment that violates this boundary is invalid and must not
be reported.

## Proposed workflow

```text
product requirement: assemble MCP
            ↓
PA retrieves manual/specification/CAD
            ↓
PA grounds target_feature, target pose, insertion axis, tolerances
            ↓
RA retrieves fresh resource state and resource-owned primitive catalog
            ↓
RA authors primitive_steps
            ↓
state checks + CCA + IK/collision/trajectory validation
       ↙ rejected                         accepted ↘
concrete feedback → RA revision       RobotAgent execution
```

## Scene-only milestone boundary

The scene-only milestone provides the isolated structure, the research workflow,
a narrow launcher for the no-hardware `gazebo_dual_spec2skill` simulation, the
dedicated `table_spec2skill.world`, and a local placeholder chat. It does not
define functional schemas, implement recognition, VLM, PA or RA behavior, invoke
CCA or RobotAgent, validate insertion physics, or execute robots.

The starting scene pre-installs the static NIST `Gear_Plate` and three
`Gear_Shaft` fixtures while leaving `gear_small`, `gear_medium`, and
`gear_large` loose on `prusa_mk4_2`. This is scene configuration, not a completed
robot-execution claim.

The case study will use a controlled local corpus.
`NIST_assembly_instructions.pdf` is retained under `references/`; existing NIST
STL files are referenced in place. ProductAgent, ResourceAgent, CCA, and
RobotAgent remain shared runtime authorities outside this package.

## Out of scope for the scene-only milestone

- ProductAgent, ResourceAgent, CCA, RobotAgent, `SystemBridge`, or `bridge.py`
  changes
- functional recognition or agent adapter implementations
- committed schemas or case payloads
- recognition, VLM, agent, detector, automatic attachment, or hardware changes
- insertion-physics accuracy claims
- experiment execution or recognition, planning, or robot-execution claims
