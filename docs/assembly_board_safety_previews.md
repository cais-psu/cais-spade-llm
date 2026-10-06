# Separate assembly-board mutex and precedence previews

Prepared on 2026-10-05. Both requirements now have separate **LLM-generated,
refined Safety-page previews**, explicit AP descriptors, LTLf formulas, and
compiled DFA DOT/PNG artifacts. The framework received the local reference drafts
and review feedback as inputs; this is assisted generation, not an independent
discovery of the specifications. Runtime selection and safety approvals are
unchanged. Neither preview is approved or active.

Earlier attempts returned HTTP 401 `invalid_api_key`. Generation succeeded after
the WSL project's `.env` was updated. Empty failed previews were removed; all
36 pre-existing records, including both local reference drafts, are preserved.

The first generated mutex draft incorrectly conditioned retained resource
location on the next task's destination. Its refinement removes that condition.
The first precedence draft used unrestricted resource wildcards; a later draft
used `assembled` as a resource state. The final refinement binds the configured
actors explicitly and uses the declared product field `part_state=assembled`.
These intermediate generated drafts remain in history with `validation.json`
notes explaining why they needed refinement. No runtime matcher or compiler
behavior was changed during these API retries.

## Requirement files

The Safety page discovers these separate files:

- [safety_assembly_board-v1_mutex.txt](../cais_spade_llm/specification/safety/safety_assembly_board-v1_mutex.txt)
- [safety_assembly_board-v1_gear_small_before_KET4_Square_4mm.txt](../cais_spade_llm/specification/safety/safety_assembly_board-v1_gear_small_before_KET4_Square_4mm.txt)

`safety_none.txt`, the existing mutex text, historical previews, approval records,
verified copies, and `SystemBridge` are preserved. This preparation does not
select or activate both specifications together.

## Scope and source of the APs

The frozen `build_environment_models(scene)` result contains all 12 configured
resources. Its actor-owned `place_approach` transitions with the exact
`destination_location: assembly_board-v1` binding identify `ur5e-3` and `ur5e-4`.
The preview uses those declared native capabilities. Nominal part assignments
do not limit which of these resources may produce the completion witness.

The [saved reference](../cais_spade_llm/specification/safety/assembly_board-v1_preview_reference.json)
contains both reference rules, exact AP meanings, bindings, and evidence sources.
The tables below use the final generated labels. Label numbering can differ from
the reference; checks compare the full descriptors. Labels are local to each
separate preview. An `ap001` in one file is not the same AP as `ap001` in the other.

These previews do **not** establish complete physical coverage of all configured
resources, KMR movement into the board, or unfamiliar recovery event names.
The original requirements remain broader than this native abstraction. Primitive
grounding and the joint CCA admission proof must establish that coverage before
an execution claim. No new physical predicate is introduced by these previews.

The ordinary generation catalog `initialization/tools.json` is currently `[]`.
It is unchanged. Each preview records the path and hash of a separate frozen
catalog derived from the configured models and `robot_task_registry`. Its hash
therefore does not match the current runtime catalog. The previews must not be
treated as ready for runtime approval on that basis.

These drafts use the actual `destination_location` parameter key. A board-valued
origin or another parameter does not override an explicit different destination.
They assume complete authoritative task context. The existing native matcher can
fall back to value matching if a context key is missing; these drafts do not fix
that runtime behavior or turn incomplete task evidence into a valid proof.

## Assembly-board mutex

> At most one resource may occupy the assembly_board-v1 destination area at any time.

| Label | Exact descriptor | Evidence |
| --- | --- | --- |
| `ap001` | `ap_event/assembly/any/ur5e-3/place_approach/destination_location=assembly_board-v1` | The exact task is executing with that destination. |
| `ap003` | `ap_state/assembly/any/ur5e-3/resource_location=assembly_board-v1/any` | Acknowledged resource location remains at the board. |
| `ap002` | `ap_event/assembly/any/ur5e-4/place_approach/destination_location=assembly_board-v1` | The exact task is executing with that destination. |
| `ap004` | `ap_state/assembly/any/ur5e-4/resource_location=assembly_board-v1/any` | Acknowledged resource location remains at the board. |

```text
G (!(((ap001 | ap003) & (ap002 | ap004))))
```

Task execution represents entry; resource location retains the occupied state
between tasks and after release. A `failed` state or empty gripper does not
clear the location AP. An observed retreat is still required to establish
physical clearance; the native location label alone cannot prove that motion.

[Generated mutex rule](../cais_spade_llm/user_verified_safety/previews/20261005T160750Z__safety_assembly_board-v1_mutex__470897e9/cca_safety_logic.json)
and [validation](../cais_spade_llm/user_verified_safety/previews/20261005T160750Z__safety_assembly_board-v1_mutex__470897e9/validation.json).

![Mutex DFA](../cais_spade_llm/user_verified_safety/previews/20261005T160750Z__safety_assembly_board-v1_mutex__470897e9/SAFE_1_dfa.png)

## gear_small before KET4_Square_4mm placement

> Placement of KET4_Square_4mm at assembly_board-v1 may begin only after assembly of gear_small is completed.

| Label | Exact descriptor | Evidence |
| --- | --- | --- |
| `ap001` | `ap_event/assembly/KET4_Square_4mm/ur5e-3/place_approach/destination_location=assembly_board-v1` | KET4 placement task execution; its first activation is the start boundary. |
| `ap002` | `ap_event/assembly/KET4_Square_4mm/ur5e-4/place_approach/destination_location=assembly_board-v1` | The same condition for this resource. |
| `ap003` | `ap_state/assembly/gear_small/ur5e-3/part_state=assembled/destination_location=assembly_board-v1` | Acknowledged gear_small assembly result with its exact part and destination context. |
| `ap004` | `ap_state/assembly/gear_small/ur5e-4/part_state=assembled/destination_location=assembly_board-v1` | The same completion witness for this resource. |

```text
((!((ap001 | ap002)) U ((ap003 | ap004) & !((ap001 | ap002)))) | G (!((ap001 | ap002))))
```

Completion must be witnessed before the first placement observation. The first
simultaneous completion and placement is rejected. The DFA remembers completion
when the corresponding resource state or task context later disappears. A trace
with no placement is permitted without demanding assembly completion.

`part_state=assembled` is a declared model field. It is not an invented
`assembled.gear_small` field or a new `processCompleted` predicate. The native
assembly transition requires its acknowledgement evidence and declares the
assembly product effect. Starting `place_insert`, a slip, and `release_part`
alone cannot establish this witness. Entry using
`ap_event/physical_observation/part_region_entry` is a different condition and
is not substituted for the requested placement-start boundary.

[Generated precedence rule](../cais_spade_llm/user_verified_safety/previews/20261005T161025Z__safety_assembly_board-v1_gear_small_before_ket4_square_4mm__0244b911/cca_safety_logic.json)
and [validation](../cais_spade_llm/user_verified_safety/previews/20261005T161025Z__safety_assembly_board-v1_gear_small_before_ket4_square_4mm__0244b911/validation.json).

![Precedence DFA](../cais_spade_llm/user_verified_safety/previews/20261005T161025Z__safety_assembly_board-v1_gear_small_before_ket4_square_4mm__0244b911/SAFE_1_dfa.png)

## Verification

- Both saved previews are readable through `SystemBridge.get_safety_rule_preview`,
  current against their requirement text, and report one compiled DFA and four
  explicit APs each. This verifies the page's data path; no browser session was run.
- `test/test_safety_previews.py`: **21 passed**, including **8,736 exhaustive
  Boolean trace comparisons** against independent mutex and strict-precedence
  checks, over trace lengths 1 through 3.
- Each final generated DFA was compared with its reference by exploring all
  reachable pairs of DFA states over the complete descriptor alphabet. Mutex
  checked 2 state pairs / 32 transitions; precedence checked 3 / 48. Both accept
  exactly the same finite traces as their references. Independent strict
  recompilation produced equivalent DFAs as well.
- **2,304 native mapping comparisons** passed on the actual final generated
  artifacts, including both board actors, other resources, exact products,
  destination changes, retained location, failure, and unfamiliar recovery names.
- Native AP mapping covers retained occupancy after release/failure, exact part
  and destination identity, unfinished/wrong assembly witnesses, and completion
  memory. Board origin and board-valued part location cannot override a different
  explicit destination. Unknown recovery names do not silently become native task APs.
- The compiler preserves `KET4_Square_4mm` instead of lowercasing it. Both AST and
  raw-AP generation routes have regression coverage.
- All 36 pre-existing preview records and 147 other protected files are
  unchanged. Five new generated preview records preserve the refinement history.
- `poetry check`, compilation of `cais_spade_llm` and `ros2`, focused Ruff checks,
  JSON/local-reference checks, and `git diff --check` pass. Poetry emits its
  existing metadata deprecation warnings.

No robot execution, Gazebo trial, failure injection, approval, or runtime
selection change is included. The legacy
`receiving_region_entry` mismatch remains unchanged.
