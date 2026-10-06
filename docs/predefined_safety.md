# Predefined safety specifications

The journal framework assumes that AP meanings, LTLf formulas, and requirement
scopes are supplied and reviewed. Recovery behavior may be proposed by an LLM;
the selected safety definitions are fixed inputs. Deterministic compilation and
complete grounding do not prove that the reviewed hazard catalog is complete.

The selectable [document](../cais_spade_llm/specification/safety/safety_assembly_board-v1_predefined.txt)
is JSON stored in a `.txt` file for existing Safety-page discovery. Select it and
use **Compile DFA** to prepare its own preview. Review and activation remain
explicit; the existing selection, approvals, and narrower historical previews
are preserved. No safety-generation API call is needed.

| Specification | Supplied LTLf | Concrete bindings |
| --- | --- | --- |
| At most one resource may occupy the assembly_board-v1 destination area at any time. | `G !(ap001 & ap002)` | `shared_area_first_resource` and `shared_area_second_resource`: every unordered pair from `build_environment_models(scene)`, with `region: assembly_board-v1`. Current 12-resource scene gives 66 instances. |
| KET4_Square_4mm may enter assembly_board-v1 only after assembly of gear_small is completed. | `(!ap001 U (ap002 & !ap001)) \| G !ap001` | `ap001`: `part: KET4_Square_4mm`, `region: assembly_board-v1`; `ap002`: `part: gear_small`, `process: assembly`, `target: Gear_Plate/Gear_Shaft_1`. |

Mutex descriptors are
`ap_state/physical_observation/shared_area_first_resource` and
`ap_state/physical_observation/shared_area_second_resource`. Precedence descriptors
are `ap_event/physical_observation/part_region_entry` and
`ap_state/processCompleted/process_target_completed`. Every label stays local to
its concrete `rule_id`; two rules using `ap001` do not share a valuation.

The target-completion AP requires an explicitly complete checkpoint ledger and
the exact assembly record. Its part binding is separate from the entry AP's part.
The existing `process_result_completed` AP is unchanged. The precedence formula
permits a no-entry trace, remembers completion, and rejects first simultaneous
completion and entry. A part already inside at the initial checkpoint creates no
fabricated entry and certifies no missing earlier history.

`SafetyLogic` compiles the exact formulas with the strict LTLfParser/MONA path.
It skips LLM parsing, AP generation, interpretation generation, renumbering, and
formula rewriting. Artifacts retain the full document, source/semantics hashes,
and configured product-geometry hash. Corrupt or changed definitions, bindings,
or DFA artifacts fail closed. The bundle's native-only plan checker cannot certify
these physical requirements; complete physical grounding is still required.

CCA loads these definitions before the legacy generation fallback. It binds all
selected scopes from its own source into the observation/composition inputs.
Physical rules stay on the grounded clock. Existing native monitors retain their
task start/completion clock; both monitor sets must share an accepting continuation.
Recovery plan registration binds behavior to fixed rules instead of generating
replacement safety rules. Outline projections alone cannot certify them.

Nominal starts also require owner-prepared physical evidence. An optional trusted
provider `nominal_request(product_jid, task)` supplies a complete finite registration
for the same coordinator; it cannot supply an authoritative permission flag.
The prepared runtime task and resource-owned primitive program must match exactly.
Without such evidence the nominal request is blocked. No live provider is added.

Optional `product_effect_evidence` records exact task/resource identities, declared
product effects, and completion timestamps. Those times join the joint trace.
Supported acknowledged assembly effects are resolved from configured `place_insert`
declarations; motion, release, and an event name alone create no completion record.
Composition may project a declared effect at a branch completion. Replay requires
a corresponding validated acknowledgement and exact observed effect before actual
history advances. Predicted metadata is compared on a detached copy only; actual
history retains acknowledged provenance. Unchanged prior acknowledged checkpoint
records remain intact. Missing or mismatched acknowledgements block further grants. After this optional
product-effect clock has been used, subsequent horizons must retain a complete
`product_effect_evidence` ledger, even when its `updates` is empty; dropping it
changes the clock identity and invalidates continuation.

The soundness claim is conditional on complete, correct supplied scene geometry,
fixed-orientation piecewise-linear trajectories, durations, stationary coverage,
custody, declared task effects, and observed history. The finite branch model must
include all running work and permitted starts. Under those assumptions, every
consumed valuation is grounded in that trace and CCA grants only starts inside its
nonblocking region. This does not establish arbitrary predicate discovery, physical
feasibility, a complete hazard model, or live execution safety.

The composition restriction on `X`, **20,000 states / 2 seconds** defaults,
`SystemBridge`, PA/RA/CCA authority, original fixtures, pending M1 delivery tasks,
and the documented legacy `receiving_region_entry` mismatch are retained.
No Gazebo trial or new live trajectory provider is included.
