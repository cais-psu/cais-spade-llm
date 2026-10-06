# Generated recovery composition and CCA admission

The `validated` recovery route registers its complete sequence with CCA during
`plan_safety_check`. RA requests `allow` or `block` before invoking
`execute_recovery_macro`. Registration supplies references to an analyzed problem;
it does not grant permission to execute every event in that problem.

The implementation is in
[RecoveryCompositionAdmission](../cais_spade_llm/agents/central_controller/recovery_composition_admission.py),
[the CCA owner adapter](../cais_spade_llm/agents/central_controller/recovery_admission_runtime.py),
and [the composition engine](../cais_spade_llm/agents/central_controller/offline_recovery_composition.py).
`SystemBridge` remains unchanged. PA retains its global-FSA bookkeeping and nominal
planning remains on its existing path.

## Registration and permission

PA preserves `recovery_safety_scope_id`, `outline_id`, `des_event_id`, event names,
task/resource identities, primitive parameters, predecessors, and pending nominal
task IDs. Its `recovery_composition_request` contains the complete task list and
exact dispatch parameters. CCA checks those against owner-supplied composition
inputs and resource preparation, then returns `recovery_composition.task_refs`.
Dispatch adds `params.recovery_composition_ref` to the registered parameters.
An `allowed` flag or DFA state supplied by PA cannot authorize execution.

CCA selects the exact requested event at the current observed time and prefix.
An overall safe continuation cannot authorize a losing start edge. Before a
grant, CCA obtains resource preparation again and rechecks identity, revisions,
monitors, physical state, timing, and the observation ledger under the admission
lock. The graph and detailed `allowed`, `held`, or `inconclusive` result remain
available on the CCA-owned coordinator. The RA reply uses the existing
`safety_decision` message and includes a bound `recovery_composition_grant`.

RA requires the grant's original and resolved primitive parameters to match its
program. It accepts grants only from its configured CCA. A consumed task identity
cannot execute a second time, even if another request arrives before PA's final
acknowledgement. A failed attempt needs a new registered task identity. This
deduplication is resource-lifetime state; a resource restart requires fresh owner
evidence and invalidation of its old proof.
Every permission attempt also has a fresh `recovery_composition_request_id`;
delayed replies for another attempt are ignored. The existing global plan FSA's
enabled-start condition is rechecked under the same lock before the new grant.

The current proof includes every configured resource. Additional starts absent
from that proof are held, including legacy fast-path starts on registered resource
owners. Already-running work is included in the analysis. Atomic admission of
multiple new resource events in one start edge is unavailable through the current
individual-request protocol.
CCA serializes complete-resource registrations across products, including the
analysis interval, so separate ProductAgents cannot obtain overlapping proofs.

## Evidence owners

An application owner may configure `recovery_composition_context_provider` on CCA.
It is a synchronous callable receiving `(product_jid, recovery_id)` while CCA holds
the admission lock. Agent messages cannot install this provider. Its context must
include:

- `composition_inputs`: frozen scene, reviewed catalog and scopes, complete
  geometry and physical evidence, recovery events, already-running work, finite
  start choices, and completion conditions for the existing offline engine.
- `revision`, `time_exact`, and `current_snapshot`: current authoritative identity,
  clock, and complete resource/part evidence.
- `task_monitor_context`: exact native task bindings, resource/product completion
  effects, owned resource/product values, resource JIDs, and state contexts.
  CCA supplies its actual compiled monitors and current states; provider-supplied
  replacements do not become monitor authority.
- `observations`: an append-only validated ledger with exact observation identity,
  time, and evidence. Physical observations and validated task acknowledgements
  have different record kinds.
- `physical_checkpoint`, or an explicit prospective activation checkpoint for
  newly activated physical requirements, plus `execution_mode`.

CCA calls each recovery resource's
`prepare_recovery_composition_evidence(request)`. The default result is
`NEEDS_CONTEXT`. A resource-owned `recovery_composition_evidence_provider.prepare`
must establish the exact program's checkpoint, resolved parameters, trajectories,
durations, helper outputs, custody effects, and execution-observation contract.
Expected endpoints do not establish a trajectory or duration.

KMR's `get_recovery_physical_snapshot()` exports retained `_primitive_state`,
`workflow_custody`, and recorded primitive evidence. Missing fields stay missing.
This export alone cannot establish the preparation contract. No existing live
provider was made complete by this change. Registered `RobotAgent` executor
overrides remain blocked until they support the same bound evidence contract.

Tests explicitly enable `allow_mock_recovery_execution=True`, configure mock
executors, and supply synthetic evidence. The production default rejects this
mode. Inputs marked synthetic cannot authorize live execution by changing only
the provider's `execution_mode` string.

## Safety history and feedback

One branch search carries two distinct histories. Grounded rules consume the
frozen physical observation clock, with per-`rule_id` AP values. Native CCA rules
retain their task acknowledgement clock, exact AP descriptors, actual callable
names, and exact scope identities. Scope `event_ids` are derived by CCA from its
existing scope routing; an event outside a scope adds no DFA tick to that scope.
Starts check admissibility and update running identities without consuming a
native completion tick. Unknown simultaneous acknowledgement orders are retained
as uncontrollable alternatives.

A completion witness must satisfy every included monitor in the same branch.
Separate acceptable task and physical traces are insufficient. An existing
obligation requiring M1 delivery remains unresolved when only the restored
Storage checkpoint is supplied. Boolean/G/F/U formulas are supported; `X` and
other unsupported composition operators return unavailable rather than receiving
an invented branching clock. Standalone reviewed checking retains its own `X`
support.

Physical history requires compatible rules, bindings, geometry, clock, and
replayable observations. Continuation rejects gaps, changed prefixes, changed
bindings, or duplicated consumption. Re-registering a sequence does not reset
either monitor. Observed completion records are replayed into detached state and
checked against actual custody and native state before a history commit.
`EnvironmentAdmission` records an ordered audit of PA-validated nominal
acknowledgements. Recovery matches the task, run, cursor, source/target DFA states,
and transition evidence against that same branch and reuses an already committed
main-monitor tick. Other scopes retain their own clocks. Missing physical records
hold recovery while preserving the committed nominal history; later matching
evidence can resume the check. A raw RA completion can arrive before PA's validated
acknowledgement and does not itself advance native history.

RA reports actual primitive results, physical snapshots, and monotonic execution
times. PA forwards authenticated terminal acknowledgements. The configured owner
must validate these notifications before adding accepted records to the ledger;
the notifications themselves are not physical proof. Registered execution does
not copy projected primitive effects or `out_state` into observed state. Missing
observations, inconsistent outcomes, or failed execution prevent later grants.

## Verification boundaries

The unchanged `storage_interruption` fixture contains four recovery events and
22 KMR primitives. Mock message-path tests exercise PA registration, RA permission
requests, CCA decisions, KMR primitive calls, and feedback. Immediate placement at
time 10 is blocked without executing a primitive; the supplied later start at
time 12 can execute after observed withdrawal and prerequisites. The completed
mock trace retains `KET8_Square_8mm` custody, deposits `KET4_Square_4mm`, preserves
both `processCompleted` ledgers, and leaves all three M1 delivery tasks pending.

The conditional guarantee is limited to the complete, fixed behavior and evidence
supplied by the owners: sound primitive effects and AP evaluators, complete
resource/running-work coverage, correctly compiled formulas, supported timing,
validated observations, and execution conforming to its prepared program. Under
those assumptions, a granted start is in the retained winning region and cannot
escape a modeled conflict by pausing inside an admitted event. This does not
establish physical feasibility, timing robustness, unmodeled predicate support,
Gazebo execution, or completed nominal resumption.

The default budget remains **20,000 states / 2 seconds**. Exhaustion blocks with
an inconclusive result. Tests can explicitly supply a larger time allowance; this
does not change runtime configuration. The legacy `receiving_region_entry`
counterexample, Safety-page selection, approvals, and original fixtures remain
unchanged. Actual test results and the remaining execution gates are recorded in
the [roadmap](<../writing/Journal Paper 2 (recovery framework)/IMPLEMENTATION_PLAN.md>).
