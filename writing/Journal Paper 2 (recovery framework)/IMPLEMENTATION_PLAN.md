# Recovery framework roadmap, including UI dry runs and trustworthy prompt inspection

## Aligned part model — October 23, 2026 — DONE offline, verified (full suite = baseline), committed `d981a3f`; Gazebo/recapture pending after activity model B

**Decision (user, October 23).** Align the part model with the semiconductor framework
(`/home/jongh/projects/SemiconductorSimulation_testing-master`). Do it before the RA rebuild, and batch it
with the RA step 1 constraints (next section) into one recapture.

**Semiconductor reference.**
- `ProductState` (`Agents/sharedInformation/ProductState.java`) is `location` plus `processCompleted`.
  There, `processCompleted` is one value: the current process marker. "startProgramX" means in progress
  (`MachineAgent.java:107`), "X" means done (108), and "startMoving" means being carried
  (`RobotAgent.java:111`).
- The history is separate: `ProductHistory` (`Agents/intelligentProduct/ProductHistory.java`) is the
  graph of the events that occurred and the states visited.
- There is no handling-status field and no interruption.

**The aligned part state.**

| Field | Meaning | Semiconductor |
| --- | --- | --- |
| `part_location` | where the part is: a place, the resource holding it, or none (loose, with a measured pose) | `location` |
| current process (name: D2) | `{process, result, status, machine}` while a process is started and not completed; none otherwise. Status: running, paused, stopped | `processCompleted`, the current marker |
| `processCompleted` | the append-only ledger of completed processes, unchanged (`{process, result}` or `{process, target}`). The CCA's APs read it (`reviewed_primitive_program_safety` 436-476, `_complete_target_ledger` 420-433) | `ProductHistory` |

- **Removed:** `part_state` (the DES variable and the product tracker's `state`), the part-scoped
  `job_state`, and `assembled.<part>`. Each restated location or process.
- **Kept:** the measured `pose` (evidence) and the machine worker's job record (`MachineJobs`, with its own
  per-job `job_state` and `processing_fraction`), which the failure observations and rule A2 read.

**Who writes what.**
- `part_location` is derived from custody only (`held_part`, machine `part_name`/`staging_part`,
  `zone_*_part`, Conveyor `part_location.<part>`, `inventory.*`, `output.*`, Exit `product_location`, and
  the ledger's assembly entry for parts on the product). Robots stop writing part fields.
- Machine `start_job` sets the current process `{trim, square, running, M2}`. `pause_job`, `resume_job`
  and `stop_job` change its status. Observed `complete_job` clears it and appends `{trim, square}` to the
  ledger.
- `place_insert` appends `{assembly, target}`. Printer `complete_job` appends `{print_part}`.
- Guards:
  - `start_job`: the part has no paused or stopped current process.
  - Reload `resume_job`: the part's current process is paused **on this machine**.
  - `place_insert`: no assembly ledger entry for the part (replaces `assembled.<part> == False`).
  - Exit pickup: every part has its assembly entry (replaces `assembled.<p> == True`, `nominal_des`
    759-767).

**Model bugs this fixes.**
1. Cross-machine resume. M2 can resume KET4's job paused on M1, although the job record is per machine,
   and KET4's fresh `start_job` on M2 is blocked.
2. `job_state` is declared part-scoped but behaves as one machine variable. Unload events never clear or
   move it (`nominal_des` 683), so after a paused part is unloaded, `start_job` for the next part is
   blocked. The reload resume also never checks that it is the same part.
3. `place_release` never resets the robot's `part_state` (`nominal_des` 603), so a stale value remains.
4. The v1 plant writes `processCompleted` as strings (`nominal_des` 84-88, `product/state.py` 15, 55,
   96-100), while v2/v3 write `{process, ...}` dicts.

**Decisions for the user.**
- **D1 — keep the ledger and add a current process (recommended).** This is the semiconductor split, and
  the CCA stays untouched. The alternative (a status inside each ledger entry) breaks the CCA ledger checks
  (`_complete_target_ledger` rejects extra keys; the monitor history checks the ledger only grows) and
  touches about six safety modules and their fixtures.
- **D2 — name of the current-process field.** Proposal: `current_process`.
- **D3 — remove the product tracker's `state` too (recommended).** It is the runtime copy of the part
  state, so "aligned everywhere" includes it. Readers: `product/state.py`, `product/environment.py`,
  `environment_runtime` 391/1212, `order.py` 204-210 (v2 `{"state": "assembled"}` requirement),
  `nominal_planner` 20-21, `failure_grounding` 302-311/341-366 (fault identity keys), `slippage_execution`,
  capture preparation (`part_slippage_recovery` 441-457, `outline_gazebo_evidence` 96-107), the products
  UI page.
- **D4 — remove the legacy contract-1 recovery first (recommended; already step 2 of "Validation
  removed").** Most remaining `part_state` readers are legacy: `product_recovery_controller`,
  `process_recovery_planner`, `recovery_session`, contract-1 branches of `multi_turn*`,
  `multi_turn_outline_state`, `recovery_validation_service`, and the validated prompts. Keep what capture
  preparation still calls: `recovery_session.prepare_recovery_request`,
  `_recovery_marked_reentry_context` and `process_recovery_planner._build_resource_search_state`.

**Stages.**
0. **Checkpoint commit: done** (`a33a0dd`, user, October 23 13:12). Baseline on `a33a0dd`: 152 failed,
   4,471 passed (the known list).
   - **D4 skipped (user, October 23, after the mapping below).** The old runtime-recovery engine is not
     dead. Delivery runs (captures, Gazebo deliveries) dispatch through the product agent's
     `_PlanExecutor` → `_next_dispatchable_task_node` → `_select_runtime_event`. That path contains a
     symbolic repair engine (controllable repairs, acquire-entity repair events, runtime plan-validation
     checks, recovery sequences run as macros), which escalates to the old LLM generation
     (`_run_multi_turn_recovery_until` → contract-1 `multi_turn`).
   - Removing it would change what nominal dispatch does on a violation, against the user's condition
     ("legacy removal is ok as long as the main pipeline works").
   - Truly dead, for a later optional cleanup:
     - 29 controller methods (about 1,500 lines: proposal approvals, guidance, preprogrammed scenarios,
       final bundles, DES retry);
     - the Run tab's recovery panel (about 1,800 lines; the tab and its nominal controls stay);
     - 13 panel bridge methods;
     - the CCA `recovery_safety_generated` inbox.
   - Whether nominal dispatch should stop instead of escalating to the old LLM is a behavior decision for
     the user, not a cleanup.
   - The kept engine gets the same mechanical part-field updates as every other reader.
   - Mapping scripts are in the session scratchpad (`trace_drafts.py`, `closure.py`,
     `classify_controller.py`).
   - Offline check: replaying the four October 23 drafts with their recorded replies reproduces the live
     drafts (scratchpad `trace_drafts.py` pattern). Use it after each stage; only the part display should
     differ.
1. **Plant and product model (offline):**
   - `nominal_des`: remove `part_state` and `assembled.*`, convert the guards above, and move the
     ledger strings to dicts.
   - The robot task definitions (`part_in_state`/`part_transition` keep location only) and
     `robot_task_recovery`.
   - `environment_models`: the current process replaces `job_state`, with the new job guards;
     `owners`/`_custody_part`; `product_state_from_custody` returns location only.
   - `product/state.py`, `product/environment.py`, `product/nominal.py`, `nominal_planner`, `order.py`.
   - `machine_jobs.machine_public_state` also maps the worker record onto the part's current process.
2. **Recovery side:**
   - `failure_grounding`: a discrepant part gets a location (the robot when an attachment is measured,
     else none) plus its observed pose, and identity keys use location.
   - `environment_recovery` placeholders; `recovery_observations`/`offline_recovery` copies;
     `slippage_execution`; capture preparation.
   - Draft path:
     - `multi_turn_outline_facts`: Parts lines and the goal "KET4_Square_4mm: trim → square
       completed".
     - `multi_turn_diagnostic`: `_CONDITION_FIELDS`, `_project_machine_jobs`, claims, `goal_reached`,
       `nominal_entry_ids`.
     - `multi_turn_context`: part snapshot and `state_variables`.
     - `multi_turn` symbolic parts, `multi_turn_outline_generation` instance tasks, and the prompt's
       feedback fields.
3. **Safety and UI:** remove `part_state` from `base_safety_checker._fixed_state_fields` and from
   `environment_composition`'s AP params (the CCA ledger APs are unchanged). UI: `nominal_resource_des`
   labels, `products.py` columns, `robot_status_card`.
4. **RA step 1 constraints** (the next section's list).
5. **Tests:**
   - Update about 30 test files. The largest are `test_multi_turn_diagnostic`,
     `test_recovery_context_retrieval`, `test_nominal_resource_des`, `test_environment_capabilities` and
     `test_part_slippage_recovery`.
   - Add regression tests for bugs 1–4.
   - Migrate the checked-in fixtures that carry `part_state`/`assembled.*` (`storage_interruption`,
     `conveyor_b66d_declared_fields.json`, the `part_slippage` turn responses and bundles,
     `legacy_task_ap_preview.json`) in the existing `test/ppr_ap_migration.py` style.
   - Old recorded captures keep loading (extra fields are ignored); the tests pinned to them
     (`test_failure_grounding.OFFICIAL`) get updated expectations.
6. **Verification and Gazebo:**
   - `poetry check`, `compileall`, `ui_main --help`, and the full suite against the known failures.
   - `make bootstrap-gazebo`, then one nominal Gazebo run.
   - Recapture all four failures through the shared queue, with the runbook flags.
   - Switch the UI defaults and archive the superseded captures first (the archive rule).
   - Rerun the four LLM-only drafts.
7. Then RA steps 2–5.

**Progress (October 23, evening; uncommitted).**
- **Done, stage 1a (machine jobs).** `environment_models._declare_machine_jobs` has no `job_state`
  variable. The job events carry `product_effects.current_process`: `start_job` sets
  `{process, result, status: running, machine}`, `pause`/`resume`/`stop` set the status (`_JOB_STATUS`),
  and the observed `complete_job` clears it and appends the ledger entry. `product_guards.current_process`
  is now checked in `project_transition` (`_check_product_guards`):
  - `start_job` is allowed unless this machine holds the part's unfinished job (`not_matches` machine);
  - the reload `resume_job` requires the part's job paused on this machine (`matches`).
  - Helper `current_process_after`. Bugs 1 and 2 are fixed. Test:
    `test_a_reloaded_part_resumes_only_where_its_job_paused_and_starts_fresh_elsewhere`.
- **Done, stage 1b (tracker).** The product tracker is `{location, current_process, last_task,
  processCompleted}` (`product/environment.py` init, `product/state.py` `ProductState`).
  - `product_state_from_custody` returns location only.
  - An assembled part is located on the product (`owners`, `project_transition`, `environment_runtime`
    391).
  - `matches_requirement` reads an order's `{"state": "assembled", "target"}` as the part's `target`.
  - The v1 ledger holds `{"process": name}` dicts (bug 4).
- **Done, stage 1c (robot part fields).** The v1 plant robots declare no `part_state`/`part_location`
  and their events write none. `assembled.<part>` stays in the v1 plant only; the runtime models strip it
  as before.
  - The robot task contracts (`robot_task_definitions` `part_in_state`/`part_transition`) keep their
    vocabulary. Only the old non-delivery bidding planner, the kept engine and the robot recovery
    descriptor read it.
  - The engine's part-state guard became a held-part check only (it was already skipped for unknown
    values: `enforce_unknown=False`).
  - The repair projections and `_recovery_part_entry_mismatch` compare location only.
  - The non-delivery tracker update stores no handling label.
- **Done, other readers:** `environment_composition` AP params, `base_safety_checker._fixed_state_fields`
  (no spec used `part_state`), the `saved_recovery_execution` comparison.
- **Done, grounding:** `failure_grounding` part facts = (location, current_process). A discrepant part is
  at its holder, or at none with its observed pose. `fault_identity`/`belief_key` use location, and
  `part_facts` drop `current_state`.
  - The four current captures' stored groundings (`beliefs.json`) no longer match re-derivation until
    they are recaptured or re-stored with `failure_grounding --store`. Do not rewrite capture files
    without asking the user.
- **Done, stage 2 (first pass, not yet fully tested):**
  - Capture preparation (`recovery_session`):
    - `_contract_part_state_holds` checks a task contract's `part_in_state` by custody
      (`in_gripper`/`in_transit`: the acting resource holds the part; `ready`: it does not);
    - no `resume_entry_part_state` rows (the carried-entity and location rows remain);
    - `part_facts` carry `current_process` instead of `current_state`.
  - `part_slippage_recovery`/`outline_gazebo_evidence` pass location and current process, and an empty
    search-state `part_states`.
  - The robot recovery descriptor declares and writes no `part_state` (it keeps `part_location`, the
    location custody gives).
  - `environment_recovery` placeholders: `part_location`, `current_process`. The machine `job_state`
    description is removed.
  - Draft path:
    - `multi_turn` seeds no `part_state` and merges a stated `current_process`
      (`_stated_current_process`: a bare status merges into the record);
    - `multi_turn_diagnostic`:
      - `condition_holds` translates an older capture's `(part, state)` row through the contract rule;
      - outcomes include ledger effects (`_completed_effects`, `_meets` containment), as does
        `declared_part_effects`;
      - `_project_machine_jobs` follows `current_process`;
      - nominal-connection steps get their model's product guards and effects
        (`_with_product_effects`);
    - context, instance tasks and prompt feedback fields have no `part_state`;
    - the Parts header reads "where it is, who holds it, and its process record";
    - `multi_turn_outline_facts` shows "at M1; processes: trim → square running on M1", and the goal
      reads "trim → square completed".
  - `slippage_execution`: a placement is a changed end `part_location` (assembled = on the product), and
    the robot valuation gets no part fields.
  - Offline prompt preview on the current (old) captures renders correctly. Their goal lines still show
    "job_state completed"/"part_state assembled", which come from the old recorded models, until the
    recapture.
- **Tests updated so far** (fixtures moved to the aligned tracker; the legacy preview fixture migrates
  `part_state=assembled` to `part_location=<destination>` in `test/ppr_ap_migration.py`):
  `test_machine_jobs`, `test_nominal_agents`, `test_recovery_delivery`, `test_failure_grounding`,
  `test_part_slippage_recovery`, `test_safety_previews`.
- **Done, draft path and UI:**
  - The draft-path suites were fixed. Nominal-connection steps carry their model's ledger and
    `current_process` effects, and a stated ledger entry that completes the unfinished process clears it
    (`_finish_completed_process`, a code-derived change).
  - No `job_state` is shown to the LLM (user, October 23). Rule A2's `task_outcome` signal states
    `process, result, status, processing_fraction`, and the jobs line reads
    "KET4_Square_4mm trim → square interrupted". The machine worker's own `MachineJobs` record keeps its
    `job_state` field (controller bookkeeping, not model state); renaming it is a later runtime change.
  - UI: `products.py` shows `current_process` and the unfinished process; `nominal_resource_des` has no
    `part_state`.
- **Verified (October 23, night):** the full suite gives **152 failed / 4,472 passed**, the same 152 as the
  `a33a0dd` baseline, none new (the extra pass is the new job regression test). `poetry check`,
  `compileall` and `ui_main --help` pass. The draft diff has no removed imports (34 files).
- **Remaining:** RA step 1 constraints, then stage 6 Gazebo (bootstrap, nominal run, recapture, draft
  reruns). Both are now ordered after the "activity model B" stage below.

**Activity model B (user, October 23: "b makes sense", "do not implement yet").**
- Every activity is a commanded start (controllable) → a running state → an observed finish
  (uncontrollable). A fault is an observed event from any state (Eqs. 2–3), and custody (so a part's
  location) changes at the finish. This is the Ramadge–Wonham machine (start controllable, finish
  uncontrollable, breakdown uncontrollable); the semiconductor robots and machines already have start →
  in-progress → finish edges.
- Machines (`start_job`/`complete_job`) and the printer already follow it.
- To do:
  - Robots: `pick_approach`, `pick_grasp`, `place_approach`, `place_release`/`place_insert`,
    `move_home`, each with a running state (`approaching_pick`, `grasping`, `approaching_place`,
    `releasing`, `moving_home`).
  - KMR: `travelling`, `grasping`, `positioning`, `releasing`.
  - Conveyor: `resource_state` idle/moving replaces `belt_stopped`, with the move's targets in
    `task_ctx`.
  - Buffer: idle/moving per zone advance.
  - Storage gets `resource_state` (passive).
- Shared events synchronize on the finish events (custody moves); start events carry the "may begin"
  guards. Example: a robot cannot place on a moving belt.
- During an activity a part keeps its last confirmed location, and the owner's running state says it is
  in transit.
- The runtime already has both halves: a dispatch applies the start, an acknowledgement applies the
  finish, and a failure report applies the fault. Verify this when planning.
- Open decisions:
  - event naming;
  - recovery outline events stay outcome-level (x → x′), with completion treated as uncontrollable by
    RA/CCA (recommended);
  - whether the Conveyor gets a controllable mid-move stop (not today).
- Order: plan B as its own stage, then the RA step 1 constraints, then one recapture after both.

**Paper.** Section 3 states the part state as (location, current process, process ledger), citing the
semiconductor `ProductState`/`ProductHistory`. The interrupted-process status is the extension.

## Activity model B — stage plan, October 23, 2026 — APPROVED (B-D1…B-D6 as recommended); B1–B7 and the B8 offline checks done (uncommitted); Gazebo steps await the go-ahead

**Decision (user, October 23).** "b makes sense"; "do not implement yet"; "come up with the detailed plan
for B including the resources that are not aligned correctly right now". Ordered after the aligned part
model (done offline) and before the RA step 1 constraints and the single recapture.

**Formal rule.** Every activity is a controllable start (the resource enters a running state) followed by
an uncontrollable, observed finish (custody, and so the part's location, changes). A fault is an
uncontrollable observed event from any state (Eqs. 2–3). This is the Ramadge–Wonham machine (start
controllable, finish uncontrollable, breakdown uncontrollable).
- The semiconductor robots and machines already have start → in-progress → finish edges. There the
  in-progress state lives in the product node and the edges are timed rather than labeled.
- The CCA's local composition already treats every action as a start plus an uncontrollable "done"
  (`local_composition.py` 129, 237-277).
- Plant composition stays ∥ᵢ Gᵢ with shared events: participants synchronize on the finish, where custody
  moves; the start carries the "may begin" guards.

### Resources that are not aligned today

| Resource | Today | Problem |
| --- | --- | --- |
| UR5e ×4 | `idle, at_pick, picked, positioned, placed`; 5 one-event activities | command and completion merged in one controllable event |
| KMR | `idle, at_pick, carrying, positioned`; `move_to_resource` is `carrying`→`carrying` | travel has no state; activities are one event |
| Conveyor | **no `resource_state`**; `belt_stopped` is always true (nothing sets it false); `loading_reserved_by` is never set to a robot (a binding only) | no state; dead variables; the advance is one event |
| Buffer | **no `resource_state`**; `zone_1..4_part` | no state; `advance_part` is one event |
| Storage | **no `resource_state`**; `inventory.<part>` | no state (passive) |
| M1, M2 | already B (`start_job` → `complete_job`, `fault_states: [interrupted]`) | `controllable_event_alphabet` wrongly lists `complete_job` (`environment_models` 183-184 copies the local alphabet) |
| Printer | already B (`print_part` → `complete_job`) | `local_event_alphabet` lacks `complete_job` (alphabets are computed before `_declare_printer_state`), so `bind_executor` (`product/environment.py` 78) would reject it |
| Exit | `empty, occupied` (participant only) | aligned |

### Target model

| Resource | States | Activities (start → running state; observed finish → state) | At the finish |
| --- | --- | --- | --- |
| UR5e | `idle`, `approaching_pick`, `at_pick`, `grasping`, `picked`, `approaching_place`, `positioned`, `releasing`, `placed`, `moving_home` + fault `failed` | `pick_approach` idle/placed → approaching_pick → at_pick; `pick_grasp` → grasping → picked; `place_approach` → approaching_place → positioned; `place_release`/`place_insert` → releasing → placed; `move_home` → moving_home → idle | grasp: `held_part` = part, the source slot clears. Release: `held_part` = none, the destination gets the part (insert: on the product, ledger += assembly) |
| KMR | `idle`, `approaching`, `at_pick`, `grasping`, `carrying`, `travelling`, `positioning`, `positioned`, `releasing` + `failed` | `pick_approach`, `pick_part`, `move_to_resource` (→ travelling → same state, new `resource_location`), `place_approach`, `place_release` | pick: Storage `inventory` = false. Release: machine `part_name`, machine → `loaded` |
| Conveyor | `idle`, `moving` + `failed` (`resource_state` **replaces `belt_stopped`**) | `advance` idle → moving → idle (`task_ctx` holds the targets); variant with delivery | regions/`part_order` updated; with delivery the leading part goes to Buffer `zone_1_part` |
| Buffer | `idle`, `moving` + `failed` | `advance_part` (zone n → n+1) | zones updated |
| Storage | `idle` (new; passive) | — (a participant in KMR/robot picks) | — |
| M1, M2, Printer | unchanged | unchanged | unchanged (alphabet bugs fixed) |
| Exit | unchanged | — | — |

- **The Conveyor's loading reservation becomes real.** A robot's `place_release` start onto the belt
  sets `loading_reserved_by` = robot (the robot is `releasing`), and its finish clears it. The Conveyor's
  `advance` start requires `idle` and no reservation, so a robot never places onto a moving belt and the
  belt never moves under a releasing robot.
- **A part keeps its last confirmed location during an activity.** The owner's running state says it is
  in transit, so location stays custody-only.

### Decisions (approved by the user on October 23, all as recommended)

- **B-D1 Naming.**
  - The start keeps the function name (`pick_approach`, `advance_conveyor`, …), so programs, executor
    binding and CCA event APs keep working.
  - The finish is `<function>_done` (`pick_approach_done`, …).
  - The machines and printer keep `start_job`/`complete_job` and `print_part`/`complete_job` (fixed
    symbols).
- **B-D2 Runtime commit points (recommended: two commits per execution).** One resource-agent execution
  produces two product commits:
  - **start** when the RA sends `ack("running")` after the CCA grant (`environment_runtime.py` 1593);
  - **finish** at `ack("completed")`;
  - **fault** at `ack("blocked")` or an injected fault.

  Rejected alternative: a separately dispatched finish task, as `complete_job` is today. The robot, KMR
  and transport executors are synchronous (`workflow_execution.py` 497, 578, 616), the worker runs one
  operation at a time, there is no in-flight handle to re-attach to, and each finish would cost a
  negotiation, a CCA preview and reservations (the CCA could even "hold" an uncontrollable event).
  Machines and the printer keep their existing two-task form, which is the same formal model.
- **B-D3 Revisions.** A start commit bumps the participants' revisions, so the pending record is re-bound
  to the post-start revisions before its finish is checked (`product/environment.py` 694-699 and RA checks
  at `environment_runtime.py` 1531, 1583, 1636).
- **B-D4 CCA stepping.**
  - `synchronize` takes typed start/finish/fault records and steps the monitors at each one: start =
    "executing X" event APs plus the running state's state labels; finish = the state labels after.
  - `local_composition` must not add its own "done" to an explicit finish (no double completion).
- **B-D5 Recovery outline events stay outcome-level (x → x′).** The LLM states one event and its end
  state; RA/CCA treat its completion as uncontrollable. The draft's nominal-connection search counts
  activities (start + finish together) so its 6-step budget still means six activities, and it never
  offers fault events as nominal steps (`multi_turn_diagnostic._nominal_successors` iterates every event
  today).
- **B-D6 No mid-motion cancel or stop events now.** Add a controllable `cancel`/`stop` later only if a
  recovery needs one.

### Sub-stages

1. **B1 Model.**
   - `nominal_des._event` takes `controllable`/`observable`.
   - Split builders: `_robot_task_event`, `_robot_pick_events`, `_robot_place_events` (`place_release`
     602), `move_home`, the KMR events in `_machine_and_mobile_events`, `advance_conveyor` (both
     variants) and `advance_part` in `_transport_and_print_events`. Custody updates and product effects
     move to the finish; every participant gets the same `event_id` and `controllable` (the shared-field
     check, `environment_models` 421-441).
   - Running states enter the `resource_state` domains. Conveyor `resource_state` replaces
     `belt_stopped`; Buffer and Storage get one.
   - Alphabets are filtered by the flag and computed after `_declare_printer_state` (fixes both alphabet
     bugs).
   - Invariants: `owners()` (869-881) and `nominal_des._check_valuation` (1048-1059) classify the new
     holding states (`grasping`, `approaching_place`, `releasing`, `travelling`, …) against `held_part`.
   - Program contracts: start `success_state` = the running state (robot registry definitions,
     `KMR_TASKS`, `workflow_task_programs`, `machine_job_functions`); `function_contracts` skips finishes.
     Regenerate the scene `resource_programs` and bump its revision (`gazebo_programs._check_program`
     compares exactly).
2. **B2 Planning.**
   - `event_bindings`/`candidates`/`feasibility`, plus the Conveyor-specific logic moved to the finish:
     `project_transition` load/advance derivations (1090-1119), `_parameterized_event` collection
     guards/effects (529-539), `candidates` (709-730), `project_nominal_event`, and `_task_reservations`
     (`product/environment.py` 614-656).
   - `environment_capabilities`; the path budget (`environment_runtime.py` `max_steps = 64`) roughly
     doubles.
   - `set_plans` `in_state`/`out_state` (718-740) per half.
   - The v1 `nominal_planner` templates.
3. **B3 Runtime (EnvironmentRuntime, and DeliveryRuntime's `_AckInbox` the same way).**
   - The PA commits a start on `"running"` (today only logged at 1369-1370) and a finish on
     `"completed"`.
   - A failure commits the fault (`blocked`/injected), and the run then stops as today.
   - Pending re-bind (B-D3); `environment_admission.synchronize` typed records (B-D4).
   - Finishes are never dispatched as tasks (their executor binding is skipped, like the observed-event
     checks at `environment_runtime.py` 1518-1528).
   - Hard-coded states: `delivery.py` 242-248, `kmr_agent.py` 191-194, `environment_runtime.py` 1169 and
     1263.
4. **B4 Capture and grounding (the failure captures depend on these).**
   - `failure_checkpoints.py` (`picked` 31, last acknowledgement `pick_grasp` 39, `place_approach` 76,
     M1 `completed` 109/131/150, `at_pick` 133, `complete_job`/`pick_approach` 138, `start_job` 184,
     `place_approach` 198) moves to the new states and finish records.
   - `conveyor_fault.py`: `complete_job` 157; `place_approach` step `descend` 247-255, which becomes the
     running state `approaching_place`; `failed` 448.
   - `evaluation_capture.py` checks (183, 194, 244-259, 300).
   - Rule A1: a robot goal cancelled mid-activity is the fault out of the running state, with its finish
     pending (like A2's pending `complete_job`).
5. **B5 Recovery and draft path.**
   - `robot_task_recovery` emits start/finish pairs from the programs (`primitive_support` and
     `composition_contract` on the start only; running states in the domain).
   - `environment_recovery`: `public_evaluation_support` reads finish updates too (137-149); the KMR
     operation names (146-147); `collection_binding` on the Conveyor finish only (316-325); `_owned_event`
     support on the start only.
   - `multi_turn_diagnostic`:
     - activity-level steps and no faults in `_nominal_successors`;
     - `_recorded_task`, `resume_entries` and `_declared_event` bind the start by function name and
       complete with its finish;
     - neighbor pairing (`_neighbor_rows`/`_owner_allows`) by the finish copy;
     - `faulted_resources` with `failed` in the domain.
   - `multi_turn_outline_facts`: state words for Conveyor/Buffer/Storage; `_placement_positions` and
     `declared_record_places` read finish updates.
6. **B6 UI and bridge.**
   - `nominal_resource_des`: function diagrams draw start and finish edges (358-410); the field choice
     prefers occupancy for Conveyor/Buffer/Storage (1068-1076, 716-745).
   - `status_badge.py` colors the running states; `robot_status_card`; bridge robot state maps (28959-28976)
     and `positioned`/`picked` checks (14729-14742, 15770, 19303, 20050, 26496, 27179, 32969).
7. **B7 Tests.**
   - Update the tests that pin activity events and states. Largest: `test_nominal_resource_des` (≈70
     references), `test_multi_turn_diagnostic` (≈50), `test_environment_capabilities` (≈40),
     `test_recovery_feasibility`, `test_part_slippage_recovery`, `test_conveyor_fault`,
     `test_safety_previews`, `test_gazebo_resource_programs`, `test_recovery_case_safety`; plus JSON
     fixtures with `belt_stopped` programs.
   - New regression tests:
     - each resource's start/finish pair and controllability;
     - no place onto a moving belt;
     - a fault from a running state;
     - two commits per execution with re-bound revisions;
     - the CCA steps once per record with no double "done";
     - fixed alphabets for machines and the printer.
8. **B8 Verification.**
   - The full suite against the baseline; `make bootstrap-gazebo`.
   - **One nominal Gazebo order run** (the main pipeline end to end).
   - Then the RA step 1 constraints.
   - Then **one recapture of all four failures** (shared queue, runbook flags), the UI defaults switch,
     archiving first, and four draft reruns.

### Progress (uncommitted; base `d981a3f`)

**B1 model, first pass:**
- `nominal_des`:
  - `_activity(models, name, actor, running, …)` declares a start and its `<name>_done` finish, linked by
    `finish_event_id`/`start_event_id`.
  - The start carries every precondition, the running state and the commanded task-context target.
  - The finish is guarded by the running state and the values it changes, so no other resource's
    unrelated state can block an observed completion.
  - Running states are declared in the model builder (`_ROBOT_RUNNING_STATES`, `_KMR_RUNNING_STATES`),
    not in the saved programs; the robots' programs and real end states are unchanged.
  - KMR travel has loaded and empty variants, told apart at the finish by `held_part`.
  - The robot's release onto the belt sets `loading_reserved_by` at its start.
  - Conveyor `resource_state` idle/moving replaces `belt_stopped`; Buffer idle/moving; Storage idle.
  - `HOLDING_STATES`/`FREE_STATES`/`APPROACH_STATES` replace the held-part invariants.
  - The v1 `project_nominal_event` chooses the variant whose guards hold and applies a start followed by
    its finish, so v1 planning and delivery still see completed activities.
  - The v1 ledger effect is on `place_insert_done`.
- `environment_models`:
  - Finishes get no program.
  - The Conveyor's collection effects are on `place_release_done`/`advance_conveyor_done`.
  - `place_insert` keeps the guards; `place_insert_done` gets the product effects unchanged
    (`"state": "assembled"`, the target, the ledger entry). A first pass dropped `"state"`; it was
    restored because the CCA's product-effect evidence reads it.
  - Event lists are computed after the printer split and filtered by `controllable` (fixes the machine
    list listing `complete_job` and the printer list missing it).
  - `finish_task`/`project_activity`; `candidates` never proposes a finish.
- `function_contracts`: a start's contract has the start's in-state and the finish's out-state, success
  state and product effects.
- Programs: only the Conveyor/Buffer placeholder states changed (`belt_stopped`, "occupied … zone" →
  `idle`), in `workflow_task_programs.py` and the scene `resource_programs` (**revision 6 → 7**).
- `slippage_execution`: belt idle check.

**B2 planning, first pass:** every planning or check projection uses `project_activity`
(`product/environment.py` alternatives, bid check, prepare; `environment_runtime` `set_plan(s)`, the cached
path, the RA pre-check; `environment_composition` local composition). The acknowledgement temporarily
commits the whole activity until B3 splits it.

**B1/B2 second pass (model holes found by the UI diagrams):**
- `move_home` starts from each rest state with an empty gripper (`idle`, `at_pick`, `placed`). Before,
  its only guard was `held_part = None`, so it could start during another running activity. Every start
  now has a rest-state guard (checked over schema 2 and 3).
- A finish observes the reservation its start made (`loading_reserved_by = <robot>`), so the belt can
  no longer record a placement while it moves.
- `_copy_activity` copies a start together with its finish, linked. Used by the printer's last-pickup
  outcome (its finish, not its start, now sets the printer state) and the machines' paused/stopped unload
  variants (before, a copied start stayed linked to the original finish).
- `nominal_des.outcome_events` folds each activity into one outcome-level event: the start's identity,
  guards and program with the finish's effects (B-D5). Used by the recovery descriptor
  (`environment_recovery_descriptor`), the function catalog and graph, and the product/process views.
- v1 nominal acknowledgement takes the finish's product effects.

**B6 UI, first pass:** resource-state diagrams show starts into running states and finishes out of them;
part-level views (function graph, product graph, process rows, function catalog) show activities; the
belt's diagram shows `resource_state` (idle/moving) instead of `belt_stopped`; default fields add
`resource_state` for the Conveyor and Buffer; the Buffer's one-graph diagram keeps zone occupancy.

**B3 runtime, first pass:**
- `EnvironmentProductContext.start(task_id)` commits an activity's start when its resource reports
  `"running"` (`environment_runtime._collect_acknowledgement`). `acknowledge` commits the finish, and
  first the start when none was reported. A reported failure is `record_fault` (no value changes), so a
  failure mid-activity leaves the actor in its running state. Records carry `kind` start/finish/fault.
- B-D3 without changing the task: `started_tasks` keeps the participants' revisions the start left, and
  `relevant_revisions_match` checks a started task against them (PA and RA share the context).
- `_task_participants` includes the finish's participants (none differ today).
- CCA (B-D4): `environment_admission.synchronize` reads every record with `record_cursor`. Start and fault
  records go to `activity_history`, keep the grant and step no monitor; finishes step the monitors as
  before, so `ack_cursor`/`acknowledgement_history` still count completions (the recovery coordinator
  and `gazebo_safety_preparation` audits rely on that). The monitors step only on finishes, as the local
  composition commits DFA history only at "done" (the user chose finish-only, 2026-10-23; see the Paper note).
- `EnvironmentPlant.project` applies only the finish of a running action whose start is committed.
- `verify_environment_run` replays start/finish/fault records.
- Readers: `kmr_primitives` takes custody from finish records; `failure_grounding` takes the later record
  on a revision tie; v1 `nominal_planner` never plans a finish. v1 delivery stays atomic.
- Checked unchanged: `delivery.py` 242-248 and `kmr_agent.py` 191-194 (v1 rest states),
  `environment_runtime` 1169/1263 (rest-state checks), workers read only fields a start never changes.

**B4 started:** `failure_checkpoints.acknowledged_pickup` reads custody from the latest finish and accepts
a robot that has started its placement (`approaching_place`); the robot-breakdown records skip starts.

**CCA and safety evidence (found by the full suite):**
- `nominal_des.outcome_models` folds activities across all models, so a resource that only consults a
  start carries the same participants and product effects as the actor.
- `_product_effect_evidence` and `slippage_execution._declared_consequence` read activities.
- Test fixtures: `ppr_ap_migration` gives transports `idle` in/out states and re-binds archived declared
  tasks to the renumbered events (`migrate_declared_tasks`).

**B4–B7 status (October 23):**
- B4: `failure_checkpoints` done. Checked unchanged: `conveyor_fault` (`complete_job` stays atomic;
  the placement capture already sees `approaching_place` from the committed start), `evaluation_capture`
  (reads finish records), rule A1 (the cancelled goal's task is the started activity; its running state
  comes from the start record).
- B5: covered by the fold. The environment descriptor folds across all models; robot descriptors are
  compiled from the saved programs (activity-level, guarded by `held_part`, not `resource_state`);
  `multi_turn_diagnostic` reads descriptors; `multi_turn_outline_facts` reads bindings and part-field
  updates that the finishes carry.
- B6: `status_badge` shows `RUNNING_STATES` as running. The bridge's robot state maps compare the
  robot agent's own rest states, so the bridge is unchanged.
- B7: tests updated; new regressions: start then finish, completion without a start, a fault leaves the
  activity running, a start keeps the CCA grant without a monitor tick, event alphabets, no `move_home`
  from a running state, belt placement needs its own reservation.

**B8 offline (October 23):** full suite 152 failed / 4,476 passed, the same 152 as the baseline (no new
failures); `poetry check` (deprecation warnings only); `compileall`; `ui_main --help`. **Not run:**
`make bootstrap-gazebo`, the nominal Gazebo run and the recapture (they need the user's go-ahead).

### Risks

- **The main pipeline runtime changes:** two commits per execution, CCA stepping, and capture checkpoints.
  The B8 nominal run and the recapture are the end-to-end proof.
- **Paths double in length:** planning budgets and the draft search are adjusted to count activities.
- **Recorded artifacts go stale:** the scene program revision bump makes old captures non-executable in
  Gazebo (they are recaptured anyway).
- **Size:** larger than the part model change. It touches roughly the model, planning, runtime,
  capture, CCA, recovery, UI and about 25 test files.

### Paper

- Section 3: one resource template (`resource_state` + variables; activities as start/finish with
  controllability; faults from any state).
- Section 5.3: the CCA checks a controllable start before granting it (its running label and projected
  end state) and the monitors step on observed finishes, the same rule in the composition check, the
  live monitor and the recovery monitor histories. "During an activity" is the activity's running label;
  rules do not name running states as state APs. (The user chose finish-only, 2026-10-23. Stepping on starts too, in all four CCA parts together, is a possible later stage.)
- State the outcome-level recovery event convention (B-D5).

## CCA check scope — October 24, 2026 — decided by the user

- **Nominal admission is Eq. 13 (done, uncommitted until the tests finish).**
  `environment_admission.specification_check` checks a start and its own projected finish against every
  specification monitor (an accepting state stays reachable) and still blocks a start that needs a
  reservation held by running work. The plant-wide nonblocking search (`local_composition.analyze`) is no
  longer used for nominal starts: it is not in the paper and timed out on the first full-order start
  even with no rules (2 s / 20,000 states). With no rules the CCA now acts like a bypass. Recovery admission
  (Section 5.3 local composition) is unchanged.
- **Both checks must eventually work for both modes** (user, 2026-10-24: "i dont need to write those on
  paper but i should still have it working"):
  - recovery: symbolic Eq. 13 on outline events, physical Section 5.3 on recovery motions (paper core);
  - nominal: symbolic Eq. 13 (done); **physical nominal check, later stage, not in the paper.** Each
    resource implements `prepare_nominal_safety_program` (its saved nominal program planned into motion
    evidence) and `observe_prepared_nominal_completion`, so `RecoveryCompositionAdmission.check_nominal`
    can admit nominal starts against the predefined (geometric) specs. Today only the URs have an adapter
    and it refuses every saved UR program (see the plan below), so a run with the seven predefined specs
    holds every nominal start. This stage also gives the
    measurement of how much the seven specs delay production.
- **Order (changed by the user 2026-10-24):** video 2 and the four recaptures, then the **physical nominal
  check** (plan first, user approval, then code; it must work with MoveIt collision checking off, so
  `prepare_ur5e_motion` stops requiring it), then the seven-spec nominal run and its delay measurement
  against `nominal_cca_05` (37 min, no rules), then recovery validation.
- Captures run with the CCA bypassed and record no monitor history; recovery validation replays each
  captured trace through the seven monitors offline to get the monitor states at $s_F$.

## Physical nominal check — plan, October 24, 2026 — APPROVED by the user (D1 native + send guard, D2 parallel claims, boundary wait)

**What exists (read from the code, not run).** With a predefined safety file selected, every nominal
start goes to `check_predefined_nominal_start` → `RecoveryCompositionAdmission.check_nominal`, which asks
the owner for `prepare_nominal_safety_program`, plans each step with `prepare_ur5e_motion`, proves the
trace against the physical monitor (`LivePhysicalMonitor.check`), claims mutex regions
(`RegionReservationLedger`), and then *executes the prepared trajectories* through
`execute_recovery_composition_step` (the dispatch guard refuses any other trajectory). The symbolic Eq. 13
check runs inside it first. A robot adapter (`nominal_safety_adapter.NominalRobotAdapter`) exists, but:

1. It accepts only plain Cartesian moves. Every UR program has steps it refuses: `open_gripper`,
   `grasp_part`, `release_part`, routed moves (`move_above_part` with `pick_transit_waypoints`,
   `move_above_destination` with `transit_waypoints`, `retreat_from_source`), and a helper read after a
   motion (`compute_place_targets` in `place_approach`). `prepare_ur5e_motion` also refuses orientation
   changes, mating context, and (since `9c55a0f`) every UR, because `avoid_collisions` is false.
2. KMR, Conveyor, Buffer, M1 and M2 have no nominal adapter at all.
3. It is serial: `holds()` holds every start while any physical program runs, and `_ground` refuses when
   any command is active or any nominal program is running.

So a seven-spec run would block its first start. Recovery uses the same owner preparation, so item 1 also
limits the recovery physical check (a recovery program with a grasp or release cannot be proved today).

**Recommended design: plan, claim, run natively, guard each send.**

- **N1 Plan-only prediction.** At a nominal start the owner plans its whole saved program from the
  current observed state, plan-only, with the same planner calls the native program makes (collision
  checking off, as executed). Gripper steps are stationary intervals; the held part's geometry is added
  from `grasp_part` until `release_part`; routed moves are planned segment by segment; helper reads that
  do not depend on the arm pose (`compute_place_targets`) are read before the motion. KMR, Conveyor and
  Buffer predict their own motion (base and arm; the transported part); M1/M2 are static bodies.
- **N2 Admission.** Eq. 13 symbolic check (unchanged), then the physical check on the predicted trace:
  the swept occupancy (configured continuous geometry plus held part) gives the program's region claim;
  mutex rules hold a start whose claim conflicts with another live claim; part-entry ordering rules
  (`part_region_entry` before `processCompleted`) must allow the entry now if the carried part can enter
  the region. Other running programs are represented by their claims (their resource may be anywhere in
  its claimed regions until it finishes and is observed clear, which the ledger already does), so
  programs in disjoint regions run in parallel.
- **N3 Native execution with a send guard.** Nominal programs keep their native executor (snap-to-slot,
  seating, custody corrections unchanged). For a nominal grant, the dispatch guard checks each
  trajectory actually sent against the claim (its swept occupancy must stay inside the claimed regions and
  respect the ordering rules) instead of requiring the exact prepared trajectory. A send outside the
  claim is refused and the task faults, so the check never relies on the prediction being exact.
- **N4 Finish.** The PA finish commits as today; the claim is released after the stopped-and-clear
  observation (existing ledger behavior); the physical monitor commits the executed trace.
- **N5 Collision flag.** `prepare_ur5e_motion` stops requiring `avoid_collisions`; it plans what is
  executed. This also unblocks the recovery physical check.

**Alternatives not recommended.** (a) Execute nominal programs as exact prepared trajectories, like
recovery: one proof method for both modes, but it replaces the native executor and drops its placement
corrections, risking every insertion. (b) Keep the serial hold: smallest change, but no two programs move
together, so production becomes much slower and the measured delay is mostly serialization.

**Recovery side (for recovery validation, not this stage).** Recovery keeps exact prepared execution
(paper Section 5.3). It needs N5 and the same gripper/routed-step preparation as N1 to prove recovery
programs that grasp or release.

**Tests.** Extend `test/test_nominal_safety_adapter.py` (gripper intervals, held part, routed moves, helper
order, no collision flag), `test/test_recovery_environment_admission.py` (parallel claims, conflicting
claim held, ordering rule held until `processCompleted`, send outside the claim refused).

**Gazebo.** One seven-spec nominal order run (`run_nominal.py --safety-file
safety_assembly_board-v1_plant.txt`), compared with `nominal_cca_05` (37 min): makespan, per-start
admission latency, time each start was held and by which rule, and whether any send was refused.

**Decisions (user, 2026-10-24: "approve").** D1 native execution + send guard; D2 parallel with region
claims. A send outside the claim first tries to extend the claim; if a needed region is busy the robot
waits at that primitive boundary (stopped, holding its part, inside its own claims), never mid-trajectory,
with a timeout after which the task faults and becomes a failure for recovery (prevents two waiting robots
from deadlocking). A task that has not started holds no claims and just waits.

## Nominal order run under B — October 24, 2026

- Captures under B (recapture_v4_20261023, queue item 7): Conveyor `capture_225e2748…`, ur5e-1
  `capture_e3c8151b…`, Machining `capture_695ea223…` (KMR now `travelling`), Part slippage
  `capture_98937559…` (ur5e-4 now `approaching_place`). **Superseded 2026-10-24** by the v5 captures below.
- **Captures after the ur5e-3 route restore and UR collision checks off (recapture_v5_20261024, queue item 8,
  2026-10-24):** Conveyor `capture_3ad495d2bd7845269fa70f1a09963bc9`, Machining
  `capture_37e090a54d0242f4948ae80c95d3cd76` (M1 stops at 6.5 %), Part slippage
  `capture_ad64bf1628564c3fa7f928b7bdebd5d9`, ur5e-1 `capture_b7dea70c1db74e1e9e98849380c09655` (second try;
  the first failed KMR readiness at launch). Same failure points, grounded rules and faulted resources as v4.
  UI defaults (`_DEFAULT_OBSERVATIONS`) and the grounding test use them. Video 2 (ur5e-1 breakdown, 20x) is
  in `monitor/recovery_gazebo_runs/videos-20x-20261024/`.
- Full 11-part order, CCA on with `safety_none.txt` (`run_nominal.py`): with the Eq. 13 admission it ran
  6 minutes with 42 start and 42 finish records (CCA `ack_cursor` 42) and ur5e-4 placed all three gears.
  It then stops deterministically: ur5e-3's `place_approach` of KET4 to the board fails motion validation
  with a self-collision (`ur5e_3_rg2_right_inner_finger` vs `ur5e_3_upper_arm_link`) on the Cartesian move
  from the buffer pickup to above the slot.
- **Resolved 2026-10-24 (user):** ur5e-3's 2026-09-29 home and pick transit waypoints restored, and MoveIt
  collision checking turned off for ur5e-1..4 (`avoid_collisions` false, as on 2026-09-29; "i dont want any
  collision check"); the KMR keeps its own. KMR `arm_motion_wall_timeout_sec` 360 s in the main scene (the
  captures' value). **`nominal_cca_05` passed:** all 11 parts assembled, all final poses within tolerance,
  37 min, 187 start / 203 finish records, CCA `ack_cursor` 203 with no invalid reason. Follow-up for recovery
  validation: `prepare_ur5e_motion` refuses robots without collision checking, so the CCA's physical recovery
  check must stop depending on MoveIt's flag.
- **Was open: ur5e-3's route.** Not B. Since 2026-10-08 (`237c3e4`, "add cca rule") ur5e-3 has
  no pick transit waypoints and a wrapped home configuration (`shoulder_lift` 5.199 rad), and since
  2026-10-02 its Cartesian moves are collision-checked; the last full order passed on 2026-09-28 with
  neither. The failure captures stop before ur5e-3 places anything, so no run exposed it. It also blocks
  any Part slippage recovery that places ur5e-3's KET4. Options: restore the 2026-09-29 ur5e-3 home and
  waypoints, or add a transit waypoint for the board approach; either changes ur5e-3 in the captures.
- Also found and fixed on the way: the late `complete_job` observer, the leftover tracker `state` reads, the
  plan rebuild projecting a started activity from its start, and the protected `cca_safety_logic.json`
  being rewritten by nominal starts (the harness restores it).

## RA feasibility validation — October 23, 2026 — plan approved by the user, not started

**Goal.** Rebuild RA validation as the paper defines it, applied to each event the LLM proposes:
Algorithm 1 (backward derivation) followed by Eq. 12 (forward check). Rebuild rules (October 22): every
constraint is model data, every check maps to one paper element, and the code has no hidden rules.
The LLM-only mode stays as it is.

**Why model data comes first.** Every end value in the four October 23 drafts was matched, value by
value, against the declared transitions of the acting resource:
- These matched: robot `held_part`, `resource_state` and location changes, and every machine job
  change (`paused`, `running`, `selected_program`).
- These did not match. None of them is an LLM error:

  | Mismatch | Count | Cause |
  | --- | --- | --- |
  | `part_state: ready` after a robot places a part on a tray, buffer or machine | 7 | Not a gap. No event sets "ready": the plant derives a part's location and state from custody (`product/state.py` 47-70, `environment_models.product_state_from_custody`), and a part in a staging slot, buffer zone, on the belt or in Storage is "ready" |
  | KMR events' `part_location`/`part_state` | 6 | Not a gap, same custody rule: the KMR's `held_part` makes the part `in_gripper` at the KMR. UR5e events also write the part fields directly, so the two resources state the same fact in two ways |
  | `pose` | 4 | No event sets a pose; only primitives do (`move_cartesian`, bounded by `gripper_reach`), so this is Eq. 12, not transition matching |
  | Conveyor `part_location.KET4: loading_position_2` | 1 | The advance is declared as `collection_effects`, not as plain updates |
  | ur5e-1 `resource_state: failed` | 1 | Unchanged from the start state (the observed fault); needs no derivation |

- Primitive constraints C_ρ and parameter domains Θ are mostly undeclared:
  - UR5e primitives have `capability_constraints: {}` and parameters typed only as numbers. The data
    exists in manifests (`gripper_reach` in `initialization/resources/robot_ur5e.json`, the
    `reachability` list) but is not attached to the primitives.
  - KMR preconditions and effects are prose ("fresh", "observed poses").
  - Machine and Conveyor primitives declare no preconditions or effects. "The Conveyor only runs
    downstream" (Machining's "no route from M1 to M2") is not declared anywhere.

**Steps.**
1. **Complete the model data (A.2).** Declare constraints C_ρ and parameter domains Θ (yes/no
   predicates, no new symbolic effects; see the rule below the list) in machine-readable form, taken
   from the existing declarations:
   UR5e `gripper_reach` and `reachability`, KMR reachable locations and navigation map, machine
   programs, Conveyor direction. The list for review is "Step 1 list" below; show it to the user
   before changing the models.
2. **Backward derivation (Alg. 1).** For each variable the event changes, collect the acting resource's
   declared transitions whose update sets that value (including `collection_effects`), and gather
   their `primitive_support`. Unchanged variables need no derivation. A value nothing produces is
   refused with the fact (for example "no declared transition of ur5e-3 sets part_state to ready"),
   never the fix.
3. **Forward check (Eq. 12).** For each collected primitive, check that some θ ∈ Θ satisfies
   C_ρ(v, v′, θ), with θ drawn from recorded evidence. Examples: a stated pose must lie inside
   `gripper_reach`; a target place must be among the reachable locations.
4. **Hook into the draft loop** behind a `check_ra_feasibility` toggle. A refusal returns to the LLM
   as feedback. An accepted event's stated end state becomes the projected state.
5. **Offline replay** of the four October 23 drafts through the check (no LLM cost) to see what it
   refuses, reviewed with the user before any live run.

**Step 1 list (October 23, for the user's review; nothing changed yet).** Where each declaration
lives today and what to add. Everything stays outside the scene `resource_programs`, whose revision is a
SHA-256 of the whole block (`gazebo_programs.resource_program_revision`), so the existing captures stay
executable.

0. **Part fields: no change.** `part_location`/`part_state` follow from the custody fields of the end
   state (`held_part`, machine `part_name`, `staging_part`, `zone_*_part`, Conveyor `part_location.*`,
   `inventory.*`, `output.*`, `assembled.*`) through the plant's product projection. The RA derives the
   custody fields; the part's condition follows from them.
1. **Resource conditions ν_i: decided by the user (October 23), no data needed.** `resource_state` is the
   DES state's name x′ and is free for every resource. Every other declared variable is a condition v′
   (bookkeeping `task_ctx.*` and private fields are skipped, as today). A machine's job is still checked
   through its part-scoped `job_state`; its `resource_state` only repeats it.
   - An event that changes only the name has no condition to derive, so the RA accepts it.
   - A new name that blocks normal production (nominal guards use the declared names) is the handoff
     check's job (Eq. 6), not the RA's.
2. **UR5e primitives** (YAML frontmatter in the docstrings of
   `resources/robot/gazebo_pick_place_controller.py`, read by `resource_primitives.py` 146-162; every
   `capability_constraints` is `{}` today):
   - `move_cartesian`: the target (x, y, z) lies inside `gripper_reach` (`robot_ur5e.json` 31-42: a
     vertical cylinder, `origin_pose` xy, `max_xy_radius_m` 0.7, `z_min_m` 0.85, `z_max_m` 1.6,
     `tolerance_m` 0.01; the same test as `robot_agent._recovery_pose_gripper_reach_error`). Use the
     value recorded in the capture's `static_capabilities`, which is what the runtime used
     (`workflow_execution` 202-206 computes it when the scene gives none).
   - `move_relative`: the resulting pose lies inside `gripper_reach`.
   - `move_to_named_pose`: `pose_name` is one of the declared `named_positions`.
   - `grasp_part`/`release_part`: the `held_part` preconditions already exist; their targets come from
     recorded evidence and are bounded by the `move_cartesian` constraint.
   - Needs one new constraint operator, "point inside a declared region". The current vocabulary
     (`recovery_transition_derivation._CONDITIONS`) compares state fields only; the region's shape and
     size stay data.
3. **KMR primitives** (`recovery_framework/kmr_primitives.py` docstring frontmatter). Add constraints
   only; the prose contracts in `kmr_tasks.PRIMITIVE_CONTRACTS` (serialized into `resource_programs`)
   stay as they are.
   - `move_base`, Θ: the KMR footprint at `target_pose` is free on the floor map. The data is the
     obstacle bounds `outline_draft._navigation_floor` already reads, plus
     `platform_overall_dimensions_m` [1.19, 0.72] from the scene KMR block. In primitive mode
     `kmr_gazebo.move_base` already allows a free target.
   - Arm moves (`move_cartesian`, `move_relative`, `move_to_named_pose`): the TCP target lies within
     `arm_reach_m` 0.82 of `arm_mount_xyz` (scene KMR block 1400-1411), relative to the base pose.
   - Noticed: the `move_base` effect is `pose_absolute_from_target_pose` in the frontmatter but
     `base_pose: observed` in the JSON, and the effect parsers know only `pose_absolute_from_params`.
     It matters for composition (Eq. 7), not for this check.
4. **Machines** (M1, M2): nothing new. Their primitives map one to one to the job automaton's
   transitions, which backward derivation reads directly. The domains already exist:
   - `program` ∈ `program_options` (the `select_program` parameter domain);
   - `result` ∈ `process_capabilities.trim.supported_results`;
   - `resume_after_reload` and `restart_stopped_jobs` in `job_control`.
5. **Conveyor:** move the region order and "downstream only" from code into the Conveyor's scene entry
   (outside `resource_programs`). Today they are the `nominal_conveyor.CONVEYOR_LOCATIONS` constant and
   a `ValueError` (239-242). `nominal_conveyor` reads the same values from data, so nothing changes at
   runtime. `move_relative` constraints:
   - no resident part's region index decreases;
   - Θ: 0 < `speed_mps` ≤ `simulation_transport.speed_mps`, already checked in `gazebo_programs`
     280-292.
6. **Buffer:** the zone chain is already data (`downstream_zone`, capacity 1). Add the `move_relative`
   speed Θ only.

**Rule: primitives must not be symbolically connectable (user, October 23).** If every primitive had
complete symbolic preconditions and effects over the same variables (location, held part, slot), a
classical planner could chain them from the failure state to the goal without the LLM, and RC1
collapses. So the RA is a verifier, never a generator:
- Backward derivation is a lookup keyed by the LLM's stated values. It reads the DES transitions and
  their `primitive_support`, which exist; it never searches.
- Eq. 12 needs only C_ρ and Θ: yes/no predicates on a given proposal over continuous geometry and
  recorded evidence (reach cylinder, free footprint, belt order, speed bounds). They give no successor
  rule.
- Add no symbolic location, slot or part effects to primitives. The existing UR5e frontmatter effects
  (`held_part`, `gripper_state`, continuous `current_pose`) stay; the composition check (Eq. 7) uses
  them to check a given sequence.
7. **Printer, Storage, Exit:** no primitives (paper Table 2), nothing to add.

Noticed, out of scope: `workflow_gazebo.py` 56-60 indexes `machines[0]`/`[1]` unsorted, while
`nominal_des` sorts them by `conveyor_loading_pose` (the same order today).

**Model bug found October 23: cross-machine resume.**
- The job record and its percentage (`MachineJobs`, `processing_fraction` at `machine_jobs.py` 485-486)
  are per machine: KET4's 7% lives only in M1's record.
- The DES lets the part-scoped `job_state: paused` travel with the part. M2's `resume_job` guard checks
  only "loaded and the part's job paused", not which machine paused it.
- M2's `start_job` is blocked for KET4, because it needs `job_state` none or completed.
- So the model allows a resume M2's worker cannot execute and forbids the fresh start it can. Today's
  Machining draft `draft_b89cc88b…` used exactly that resume.
- The scenario intends `resume_after_reload` for the same machine (M2 resumes RGOCG4-50) and a fresh
  start for KET4 on M2.
- Fix: guard the resume by the machine that holds the job. In the aligned part model, each record entry
  names its machine. Fix it whichever part model is chosen.

**Aligned part model (chosen by the user October 23).** See the section "Aligned part model" at the top:
location, current process and the unchanged `processCompleted` ledger. It fixes this resume bug.

**Design choices (user-approved October 23).**
- Match each changed variable separately, not the whole end state against one transition, so
  combined task-level events (Part slippage's single pick-and-insert event spans two nominal
  transitions) are not refused for their shape.
- With checks on, validate all 1–5 candidates. Take the LLM's selected candidate if it is feasible,
  otherwise the first feasible one in the LLM's order. If none is feasible, refuse with each
  candidate's reason.
- Model data first (step 1), then the checker.

**Resource conditions (decided October 23).** The paper derives from the resource condition,
ν_i(x′) = v′_i, not from the DES state name x′. So `resource_state` is free for every resource, and
backward derivation applies to the other declared variables only (Step 1 list, item 1). Part fields
(`part_location`, `part_state`) follow from the custody fields of the end state by the plant's own
product projection (Step 1 list, item 0).

**Does changing the model data need a recapture?** Not for the outline-stage RA check:
- **Primitive contracts, constraints and domains (C_ρ, Θ):** they are static declarations (A.2) and
  do not change what happened in Gazebo. The captures record the catalogs as they were (robot
  catalogs inside the recorded `recovery_des_model`, other catalogs inside the recorded evidence), so
  the RA check must read the current declarations at validation time, not the recorded catalogs, and
  record their fingerprint in the draft. Do not rewrite capture files: `source_sha256` pins every
  draft to its capture.
- **No event declarations change.** The part-field rows were custody, not gaps (see the table), so the
  recorded DES models stay valid.
- **Later, for Gazebo execution of a recovery:** a change to the scene `resource_programs` bumps its
  revision, and captures with an older revision cannot execute in Gazebo (`delivery.py` 265-266). The
  step 1 list touches nothing inside `resource_programs`.

**Later.** The handoff check (Eq. 6, refuse `recovery_complete` until H holds) as its own toggle, then
CCA safety (Eq. 13). Ablation rows: LLM only, +handoff, +RA, +CCA, all three.

## Resource model correction — October 22, 2026 — stages 0–3 DONE (uncommitted); stage 5 prepared, Gazebo not started; stage 4 dropped

**Why.** The user decided the plant models must be correct before the RA rebuild; the paper
(Table 2, Section 2 models) changes afterwards. Seven items: (1) prompt display of per-part
families and the fake "unknown" state; (2) the Conveyor's `advance_conveyor` missing from its
recovery model; (3) shared events missing from passive resources' recovery models; (4)
Conveyor/Buffer `move_relative` not recovery-selectable; (5) Conveyor/Buffer transport bundled in
one step; (6) coupling through the part's location as in the semiconductor framework
(`SemiconductorSimulation_testing-master`, `BufferAgent`/`RobotAgent`/`MachineAgent`); (7) the two
unlinked M1/M2 state layers.

**Status (October 22, end of day).** Stages 0–3 are implemented and uncommitted; the user asked to stop
right before any Gazebo run.
- Stage 0: `_resource_row` shows `part_location.<part name>` and no state word without a declared
  `resource_state`; `recovery_resource_adapter` leaves `current_state` None for such resources.
- Stage 1: `environment_recovery_descriptor` keeps own events in `events` and other resources'
  events this resource takes part in in `shared_events` (own guards/updates, no program); events with
  `collection_effects` (Conveyor `advance_conveyor`, the Conveyor copy of robot placements) stay unbound
  with `collection_binding: projected_valuation`. `outline_draft._inputs` derives non-robot models
  again from the recorded evidence snapshot (identical to the stored models in all four captures before
  the change, except the KMR catalog in the Part slippage capture). Nominal-connection steps carry the
  other participants' bound updates as `neighbor_end_state` (`by: model`). Replay: three drafts
  unchanged; the ur5e-1 draft's handoff now names the Conveyor's resumed advance as a nominal entry
  instead of an unchecked assumption.
- Stage 2: transport is `detect_parts` → `compute_transport_targets` → `move_relative`
  (`workflow_task_programs.py`, `gazebo_programs._WORKFLOW_EXECUTORS`, `WorkflowPrimitiveRunner`);
  scene `resource_programs` revision 6 with all three recovery-selectable. Not built: executing a single
  Conveyor/Buffer step during recovery composition (it did not exist before; belongs to the composition
  stage).
- Stage 3: `_declare_machine_jobs` builds one automaton (`resource_state` idle, loaded, running, paused,
  stopped, completed), part-scoped `job_state`, uncontrollable `complete_job`, picks from completed,
  paused or stopped, and `fault_states: [interrupted]` (accepted by `owners`). `machine_public_state`
  maps the worker job record onto the machine's `resource_state`; job programs carry entry/success
  states; `validate_function_contracts` and the UI function view skip uncontrollable events.
- Tests: affected tests updated, regression tests added; archived fixtures migrate saved programs in
  `test/ppr_ap_migration.py`. Full suite: 152 failed (the known list), 4,478 passed.
- Stage 5 prepared: queue entry 6 in `recapture_v2_20261015/GAZEBO_QUEUE.md`; captures go to
  `recapture_v3_20261022/`. No capture exercises a belt advance before its failure point, so the split
  transport is covered by unit tests only. User-verified safety preview inputs are stale and need the
  user's re-approval.

**Additions A–C status (October 22, night): implemented, uncommitted, Gazebo not started.**
- A: KMR `move_to_location` removed (nominal_des, kmr_tasks, gazebo_programs, kmr_agent, kmr_gazebo,
  workflow_execution, kmr_recovery, scene `resource_programs`). `KMRPrimitives.move_base` declares its
  effect `base_pose: pose_absolute_from_target_pose: [x, y, yaw]`; `declare_derived_conditions` takes a
  base-pose effect first and marks that pose `locates_resource`, so "moves to any position (x, y, yaw)"
  and the KMR pose condition now come from the primitive.
- B: job-controlled machines lose `machine_part`; `start_job` (no `job_id`; the machine names the job
  after the task) then observed `complete_job` with the trim effect. Worker: `start_job` returns after
  acknowledgement; a `complete_job` request runs the old wait loop (`await_job_completion`), where the
  Machining fault now fires. Runtime: observed events skip the saved-program contract and the CCA
  safety check/running messages, and admission accepts them without a grant. Planner binds `result`
  from the requirement; feasibility checks `start_job`. Failure code moved to `start_job`/`complete_job`
  (fault request, checkpoint hold, ur5e-1 checkpoint, capture check, rule A2 names the pending
  `complete_job`). Draft nominal connection now includes observed events.
- C: printer `print_part` is idle → printing (no output); observed `complete_job` printing → completed
  adds the output and the `print_part` record; custody accepts whichever event makes an output present.
- Open: with the CCA active (not bypassed), an observed event still sends its "completed"
  `resource_event` without a prior grant; untested against a live CCA. Old captures keep their recorded
  models (with `machine_part`, `move_to_location`) until the recapture.

**Approved additions (October 22, evening), before the recapture.**
- A. Remove the KMR `move_to_location` event: it is not nominal, and its program is one `move_base`, so it
  bypasses composition. Keep `move_base` as a primitive.
- B. Nominal machining as controllable `start_job` (loaded → running) then observed `complete_job`
  (running → completed); remove `machine_part`, which duplicated both and ignored the part's job state.
- C. Printer follows the same pattern: a start into a printing state and an observed completion.
- B scope (user chose full B): the runtime had no way to acknowledge an observed (uncontrollable) event, so B adds
  one (a completion observer on the machine worker, dispatch and acknowledgement without a command or CCA
  grant), makes `start_job` plannable, and moves the Machining fault injection, the ur5e-1 checkpoint,
  capture checks and grounding from `machine_part` to `start_job`/`complete_job`.

**Findings that shape the plan.**
- Nominal composition is already $\Vert_i G_i$: `nominal_des._event` (54-94) gives each participant
  its own copy of a shared event holding only its own guards/updates, and
  `project_nominal_event` (1184-1259) applies all copies atomically. What looks like "a robot writes
  the Conveyor" is the Conveyor's own copy. The defect is on the recovery side:
  `environment_recovery_descriptor` keeps only owned events (302-304), and draft nominal-connection
  steps check the owner's guards but never apply its updates (`multi_turn_diagnostic` 1120-1294).
- `advance_conveyor` is enumerated from capture-time belt residents
  (`environment_recovery.py` 305-349), and `conveyor_advance_parameters` raises on an empty belt
  (`nominal_conveyor.py` 219-220), so the Conveyor had no recovery event in the Machining capture.
- Scene `resource_programs` (`initialization/recovery_framework_gazebo.json`, revision 5) marks
  Conveyor/Buffer `move_relative` executable but not recovery-selectable; `gazebo_programs` lets only
  robots/KMR add steps (180-181) and requires primitives = executors (256-257);
  `WorkflowResourceAgent` has no recovery-step methods for transport.
- Machine job state lives only in the worker's `MachineJobs`; DES `job_state`/`active_job` are never
  written during nominal runs. `machine_part` (DES) ignores jobs; its saved program is one
  `start_job` op. The duplicated pick events guard `loaded`+`paused|stopped` but keep a `completed`
  capability source, and no copy exists for `interrupted`. No CCA rule reads job fields.
- Non-robot recovery models are derived views (`capture_files.agent_declarations` keeps
  `recovery_des_model` only for robots), so stages 0–1 can be re-derived offline.

**Stages.**
0. *Prompt display (no model change).* `multi_turn_outline_facts.py` 717 (`part_location.<part
   name>` label) and 738-741 (no state word without a declared `resource_state`);
   `recovery_resource_adapter.py` 127-134 stops writing "unknown" for new captures.
1. *Recovery models (no nominal change).* Keep `advance_conveyor` as a declared event with unbound
   `next_locations`/`delivered_part`; bind at validation against the projected state, never at
   capture. Keep shared events in every participant's recovery model as passive copies (own
   guards/updates, no `primitive_support` or composition; Conveyor load effects through
   `nominal_conveyor`); this affects KMR and M1/M2 too. Apply owners' copies in draft
   nominal-connection steps. Add an offline re-derivation of non-robot recovery models into the four
   captures' `beliefs.json`. `neighbor_end_state` then states the passive participant's copy of a
   shared recovery event.
2. *Conveyor/Buffer primitives (nominal change).* Split transport into `detect_parts`,
   `compute_transport_targets`, `move_relative` (clearance as its precondition, arrival as its
   completion observation); recovery-selectable. Files: `workflow_task_programs.py`,
   `gazebo_programs.py` (step and executor rules), `workflow_gazebo.py` (dispatch 145-151, split
   605-653), scene `resource_programs` (revision bump), `WorkflowResourceAgent` recovery methods,
   `environment_recovery.py` catalog/feasibility (80-107, 440-530), `recovery_case_evidence.py`
   transport contract (563-760). `make bootstrap-gazebo`; one nominal Gazebo run.
3. *Machine model (nominal change; design approved October 22).* One machine automaton
   (approved: `resource_state` idle → loaded → running ⇄ paused, running|paused → stopped,
   running → completed, picks from completed/paused/stopped → idle), with each part's job progress
   as a part condition beside `processCompleted`, so a displaced job stays with its part;
   `interrupted` is an observed fault state (Eqs. 2–3), not a nominal state. Files:
   `environment_models._declare_machine_jobs`, `nominal_des` machine events, `machine_jobs.py`,
   `workflow_gazebo` machine methods, `workflow_execution` validators, fault injection
   (`conveyor_fault.py`, `failure_checkpoints.py`), `failure_grounding` rule A2,
   `recovery_observations` compatibility checks, `evaluation_capture` requirements, prompt/projection
   (`multi_turn_diagnostic` 387-428, outline facts), UI, `product/state.py`; about 40 tests
   (`test_machine_jobs.py`).
4. *Part-location coupling — DROPPED by the user (October 22).* Given the finding above, the nominal
   model is already a correct shared-event composition; stage 1 fixes the recovery side. A semiconductor-style
   redesign would replace slot variables across `nominal_des`, `environment_models`
   (owners/custody/`project_transition`), `product/state.py`, slippage execution, composition, CCA
   handoff evidence, UI and about 35 tests.
5. *Recapture all four scenarios once* after stages 2–3 (shared Gazebo queue; archive superseded
   captures first), then rerun drafts and start the RA rebuild on the corrected models.

**Also stale after stages 2–3.** Program revision (old captures cannot execute in Gazebo,
`delivery.py` 265-266), frozen `user_verified_safety/preview_inputs` (re-approval), saved
proposal fingerprints (`saved_recovery_execution.py` 168-169).

## Validation removed for a clean rebuild — October 22, 2026 — contract-2 removal done, legacy path pending

**Why.** RA feasibility had grown rules the paper does not have (planner custody checks under any
check, shared-area rules at the handoff outside CCA, "part conditions alone" refusals,
availability refusals, KMR-only physical rules), and two feasibility implementations. The user
chose to delete all validation and rebuild RA feasibility (Alg. 1, Eq. 12) and then CCA safety
(Eq. 13) from scratch, with every constraint declared as model data, never as validator code.

**Old code.** Tag `pre-validation-rebuild` = `d389368` (October 21, 22:40). Read a file with
`git show pre-validation-rebuild:<path>`; restore one with `git checkout pre-validation-rebuild -- <path>`.

**Removed (uncommitted until the checkpoint commit).** All contract-2 outline validation: RA
feasibility (`validate_primitive_support`, `validate_recovery_transition`), physical evidence
(outline projections, endpoint evidence, evidence policy, Gazebo outline previews, KMR
calculations), CCA outline safety (`outline_evidence_safety`, `outline_macro_safety`, the CCA
handler), the RA/CCA/ProductAgent validation transport, nominal-continuation proof, the
validated offline workflow (`create_offline_recovery`, saved-result loading, Test in Gazebo), the
validated live outline stage of `PartSlippageRecovery`, the evaluation batch runner and outline
audit, `scenario_runner` and its diagnostic job runner, and the outline check options in the
draft, controller, bridge and UI. Also removed from the draft path: `custody_findings` and
`handoff_conflicts` (handoff records no longer carry `safety_conflicts`).

**Kept.** Validation-off drafts (schema, prompt, LLM selection), failure capture and
`prepared_recovery_request` preparation, DES models and manifests, the CCA runtime monitor,
nominal runtime, and composition and execution code (unreachable until validation returns).
The Recovery page now offers Gazebo Start failure (capture only) and Offline drafts.

**Verified.** A replay harness feeds each draft's recorded LLM responses back through the
pipeline. For the newest draft of each scenario (`draft_3a8544a4`, `draft_3e0bf483`,
`draft_d84e7ef7`, `draft_8a8b775a`, 43 calls) every request and every event is identical to the
pre-removal replay; the only difference is the handoff record (`safety_conflicts` gone, `basis`
text). Every package module imports and `ui_main --help` runs. Tests: at the tag 190 failed and
5,776 passed; after the removal and test cleanup 152 failed and 4,478 passed, with no failure
outside the tag's list (the other 38 known failures were in deleted validation tests). About
1,300 validation test cases were deleted; tests of kept code were updated to the new interfaces.
Until the rebuilt CCA check has tests, nothing checks the meaning of the M1, M2, Conveyor and trim
rules (their catalog and compile tests remain).

**Form checks removed too (October 22, later).** The user chose no checks at all after the LLM
replies. Removed from the draft path: the event form check (`_derive_candidate_outline_task` is no
longer called there), the reply-envelope and selected-index checks, the repeated-state refusal and
its `selection_unresolved` stop, and the `recovery_complete` format check. The draft applies the
chosen event as the LLM states it: `multi_turn_diagnostic.event_record` copies its fields and adds
only `expected_start_state` from the projected state. A reply code cannot read (for example several
candidates and no `selected_candidate_index`) ends the draft as `draft_incomplete` with the error;
`max_turns` still bounds a draft. The response schema is still sent, with `strict: False`, so it is
a hint only. A variable the resource does not declare stays on the event and in `condition_changes`
but is not projected, since the projection writes only declared variables. Replay: `draft_3a8544a4`,
`draft_3e0bf483` and `draft_8a8b775a` are identical to the pre-change replay; `draft_d84e7ef7`
diverges at call 5, where the old check refused `unload_to_conveyor` for undeclared Conveyor
variables (`part_location`, `part_order`) in `neighbor_end_state`. Tests: 152 failed (the same
list) and 4,476 passed. `_derive_candidate_outline_task` stays for the contract-1 code and goes with
step 2 below.

**Next.**
1. Checkpoint commit of the contract-2 removal (user's choice).
2. Remove the legacy contract-1 runtime recovery (`product_recovery_controller`,
   `recovery_session`, `process_recovery_planner`, contract-1 branches of `multi_turn*`, the
   legacy UI panel). `RecoverySessionMixin.prepare_recovery_request` and its helpers build every
   capture's `prepared_recovery_request` and must stay; nominal ProductAgent code shares the
   controller file, and `EXPORTED_METHODS` must be trimmed in step.
3. Rebuild RA feasibility: backward derivation by value over the resource's DES transitions,
   forward `∀ρ ∃θ_ρ: C_ρ` with θ from recorded evidence, the next state from the matched
   transition, and refusals the LLM can act on. Then CCA safety.

## Turn 1 recovery outline, validation off — October 19, 2026 — Part slippage done (uncommitted)

**Status.** Part slippage recovery outline generation with every check off is done (user,
October 19). All code below is uncommitted on `recovery-framework-journal`. The next step is
the user's choice between validation (RA feasibility first) and the next failure scenario.
`.vscode/launch.json` is set to `gpt-5.4` with reasoning effort `medium`. October 20: the first
live Conveyor run with the buffer goal wandered around M1; after three wording fixes and a lookup
fix (see Known issues, Conveyor), the rerun `draft_8bf5b6d05b6642098baedd8e25211852` used the KMR
and handed off. October 20 evening: two live Machining runs; the transfer across the Conveyor is
still open (see Known issues, Machining). October 21 night: neighbor records, `neighbor_end_state`,
the general fixes batch, four runs, the bias review and the open decisions are under Known issues,
"All four scenarios".

**Core contribution (user-confirmed).** The nominal DES models are a closed process bound to
designated regions and parts. The LLM creates new task-level recovery events and new state
values beyond them. Primitive composition grounds those events later; feasibility and safety
validation are added later.

**Prompt principles (user decisions).**
- Generic for any observable failure, including unexpected ones: no failure-type, signal-kind,
  resource-type or scenario wording. Each grounded signal is stated by its name, subjects and
  recorded values.
- No bias toward the solution: no derived answers, no code-chosen relevance, no hints. Every
  resource and part is listed the same way. Facts, not labels ("not held, not at any station").
- Resource names, never JIDs, in the prompt and the UI. Details sit behind lookup references,
  with no overlap between prompt and references. Safety rules are shown once.
- The Recovery Goal is the interrupted work's outcome only. Recovery events may produce it or
  leave it to declared nominal events. October 21: each goal line also names the recorded task that
  normally produces it and the current reported fault of the resources that task needs, a recorded
  fact (`(normally M1's machine_part; M1 unavailable)`), not a hint about what else could do it.
- Process-chain propagation is future work in the paper.

**The prompt now** (`multi_turn_prompts._render_failure_outline_prompt`, facts from
`multi_turn_outline_facts.outline_template`): Task and Role (1–5 candidates, the LLM picks one
with `selected_candidate_index`), Failure, Recovery Goal, Accepted events, Resources (`workspace`
= the declared parameter domain of location-changing events: named places, `declared work envelope`,
`coordinates (x, y, yaw)`, or `own location`; it replaced `serves` on October 19 so the terms fit any
resource as a DES), Parts, Safety Rules, Candidate Rules, Looking Up Facts (`need_context`
with 1–8 references), Retrieved context.

**Prompt terms and their DES meaning (user, October 19).** The prompt uses plain words; each maps to
one paper concept, so the same terms fit any resource and any failure. See also `PAPER_FRAMEWORK.md`.

| Prompt term | DES meaning | Paper |
| --- | --- | --- |
| state (`idle; holds none; at Storage`) | current values of the resource's declared state variables | $\nu_i(x)$ |
| `capabilities:` | what the resource's declarations allow; omitted when it declares none | resource capabilities in $\mathcal{K}$ (Fig. 5) |
| `reachable locations` | places its declared events can take it to (`reachability` / `resource_location` domain) | Fig. 5 "reachable locations" |
| `(normally <resource>'s <function>; <resource> <value>)` on a goal line | the recorded replaced task that normally produces the outcome, and the reported fault of each resource it needs (October 21) | the interrupted work $s_F$'s remaining nominal part |
| `job control: <flag> <value>` | the resource's declared job-control flags, e.g. `resume_after_reload true, restart_stopped_jobs false` (October 20) | resource capabilities in $\mathcal{K}$ (Fig. 5) |
| `places parts on <place> at <position>` | for a reachable place that keeps each part's position (a per-part variable), where its own declared events put a part this resource places there; a separate item, so the place name stays as declared (October 20) | the place's declared effect of the shared placement event in $G=\Vert_i G_i$ |
| `workspace bounds` | a declared reach area (`gripper_reach`), detailed by the pose lookup (`/resources/<name>/pose_relations`); the "(look up)" tag was dropped on October 20 (user) | Fig. 5 "workspace bounds", $C_\rho$ |
| `moves to any position (x, y, yaw)` | a declared event that sets its location from free numeric parameters | free parameters $\theta$ |
| process, e.g. `trim (circle, square)` | declared `process` / `result` values of its events | processes it can run |
| Knowledge Base section | the lookups served from recorded facts | knowledge base $\mathcal{K}$ |
| recovery event, `expected_end_state` | a new event and the state it leads to, $Tr_i(x,e)=x'$ | $e_k^R$, $x_k^R$ |
| "normal production takes over" | handoff to the remaining plan | Eq. 6, handoff condition $H$ |
| `neighbor_end_state` (named `place_end_state` until October 21) | another resource's variables that the event's part move changes; that resource does not act | $\nu_j(x^R_k)$ in $G=\Vert_i G_i$ |

`_capabilities` in `multi_turn_outline_facts.py` documents the rule. It replaced `serves`, `workspace`
and `reachable`/`stationary` on October 19 (the user found motion-only terms robot-centric).

**Step 0, conditions (October 19, uncommitted).** An end state is the paper's $v'_i$: "declared
variable = new value" for what the event changes, and nothing else.
- **Where the variables come from.** `multi_turn_diagnostic.declare_derived_conditions`, called by
  `OutlineDraft`, adds two variables by the same rule for every resource:
  - `pose`: from a primitive's declared `current_pose` effect (UR5e: x, y, z); failing that, from a
    DES event's coordinate inputs (KMR: x, y, yaw);
  - `processCompleted`: a part-scoped record, on resources whose events declare it as a product
    effect. It is seeded from the recorded part history.
- **Lookups.** `/resources/<name>/state_variables` and the new `/parts/<name>/state_variables` serve
  the variables, with a rule: places, parts and positions must be known; other values may be new.
- **Reply format and checks.** October 21 (user decision): the format gives only the event's form,
  the same for every resource. `expected_end_state` is an object with at least one entry, and no
  variable names are listed. The LLM takes the names from the lookups. The event check accepts
  only declared names, and a refusal names the declared variables. `params` is gone; a position
  is `pose`.
- **Projection.** A stated `pose` becomes the resource's position for area checks and the
  shared-area rule. When the event states a `pose` but no `resource_location`, code sets the
  location to the named place whose recorded area contains the pose, or to none.
- **Who set each value (October 19, cleanup).** `apply_tentative_effects` records on each event
  `condition_changes`: one entry per changed variable, with `by` = `LLM` (a recovery event's end
  state), `model` (a nominal step) or `code` plus a reason. Code entries are only consequences:
  the location from a stated pose, a released holder, a place's unknown values, and interrupted
  work whose declared outcome now holds. It replaced `derived_by_code`.
- **Unknown values at a place.** When a part enters or leaves a place that is itself a resource,
  and another resource's event moved it, that place's state and each variable that can hold the
  part become unknown: either the declared values name the part (the buffer's `zone_1_part` to
  `zone_4_part`, a machine's `part_name` and `staging_part`) or the variable records it now. This
  replaced a fixed list (`part_name`, `resource_state`) that named machine fields on every place.
  When the event states that place's change in `place_end_state` (below), nothing of it becomes
  unknown and its unstated variables keep their values. October 21: a part's own entry
  (`part_location.<part>`) counts too, and a place inside a resource (a staging tray) marks only the
  owner variables its declared events set there; the prompt and lookups show these as "not stated".
- **`place_end_state` (user-approved option A, October 20, uncommitted).** An event may also state
  how moving its part changes another resource: `{"<resource name>": {"<variable>": <value>}}`.
  - Why: an event may set only its own resource's and its part's variables, so a KMR setting KET4 in
    the buffer could not say "zone 1"; code then marked zones 1–4 unknown and the handoff rested on
    assumptions (`draft_8bf5b6d0`).
  - Prompt rule (after the `expected_end_state` rule; rule 0 also names the field), shortened on
    October 20 with the other Candidate Rules: "`place_end_state`: when moving the part changes
    another resource (it arrives at or leaves it), that resource's own changed variables, not the
    part's: {...}. That resource does not act; variables you leave out stay as they are."
  - Shorter Candidate Rules (user, October 20; same meaning, about 170 words instead of 280): one
    event per candidate with its fields (`event_name` = what it achieves, new names allowed;
    `rationale` = the fact it relies on); where a resource works and what it handles; what
    `expected_end_state` holds; `place_end_state` (only with every check off); "Use variable names
    exactly as declared. Look up any fact or variable you rely on that is not shown."; completion.
    Built by `_failure_outline_candidate_rules` from `_FAILURE_OUTLINE_EVENT_RULES`,
    `_PLACE_END_STATE_RULE` and `_FAILURE_OUTLINE_CLOSING_RULES`.
  - Offered only when every outline check is off (`llm_selects_next_event`): the rule
    (`_failure_outline_candidate_rules`), the reply format (`schema_with_place_end_state`, in
    `_get_response_schema`'s draft block) and acceptance in `_derive_candidate_outline_task`. No check
    reads the field yet, so drafts with a check, and the validated path, keep the old rules and refuse
    it (found by the code review: otherwise checked drafts would carry unchecked values).
    `response_with_jids` maps the keys to JIDs and `_named_record` back to names.
  - Check (`multi_turn._candidate_place_end_state`), format only: the event names a part; each key is
    a resource other than the acting one; each variable is that resource's public, non-part declared
    variable. Where the part went, and the values, are the LLM's claim. Outside drafts the field is
    still refused as an unknown top-level field.
  - Projection (`multi_turn_diagnostic._apply_place_end_state`): values applied as the LLM's
    (`condition_changes` by `LLM`), removed from the place's unknown list, before holder release and
    the unknown rule. A stated `held_part` on that resource also sets or clears the part's holder, so
    both sides agree; custody notes skip a holder the event states as releasing the part.
    `_event_identity` and `_ARTIFACT_TASK_KEYS` include the field. The Recovery turns view shows a
    "Place changes" column when any event states one.
  - Replay of `draft_8bf5b6d0` with the stated zone (scratch `place_end_state_replay.py`): step 3
    shows `Buffer For Machined parts  zone_1_part = KET4_Square_4mm`, no unknown lines; handoff
    complete with no assumptions; the buffer's row shows the zone.
  - Paper: the other resource's part $\nu_j(x^R_k)$ of the intended end state in $G=\Vert_i G_i$;
    that resource does not act, so it has no composition (Eq. 7 covers $i(k)$ only). The nominal
    models already change other resources in one event (the KMR delivery sets the buffer's
    `zone_1_part`; a staging-tray place sets the machine's `staging_part`).
  - Gaps for step 2: with checks on, `validate_partial_event` runs before the projection, so CCA
    labels (Eq. 13) and RA do not read `place_end_state` yet; nothing ties the acting resource's
    release to the place variable it changes.
  - Tests: `test_an_event_may_state_how_moving_its_part_changes_another_resource`,
    `test_place_end_state_is_checked_for_format_only`,
    `test_a_stated_holding_change_on_another_resource_is_also_the_parts`, plus the prompt, schema
    (offered only with checks off), round-trip, draft-naming and turns-view tests.
- **Draft table.** One row per LLM recovery event: Step, Event, Resource, Part, End state,
  Rationale. The LLM's values read `entity  variable = value` with no label (user, 2026-10-20);
  a value code added ends with `— code (reason)`. Claims of interrupted work ("interrupted <task>
  completed") are no longer listed (user, October 20): they are not values of any resource or part,
  and `_claim_interrupted_work` keeps them only for the handoff record; older drafts hide them. The connecting nominal steps are not rows; the
  "Handoff to nominal work" card shows them with only the public values they change (no
  `task_ctx.*`, no copied values). Drafts saved before this change show their end state as the
  LLM's.
- **Who writes what.** The LLM writes recovery events' end states (new values allowed). Code writes
  nominal steps' end states, start states, unchanged variables and consistency follow-ons.
- **Not done:** `pose` derived from a named place's geometry is deferred (the RA computes targets
  from places in step 2).
- **Data gaps for the manifests:** KMR `move_base` and the M1/M2 job primitives declare no effects;
  the printer declares no `process` value.

**Lookups.** `/parts/<name>/pose_relations` and `/resources/<name>/pose_relations` return
containing regions, served stations that contain the pose, and the pose beside every declared
work envelope (`xy_distance_m`, `max_xy_radius_m`, `z_m`, `z_range_m`, `tolerance_m`), never a
verdict. Resources whose declarations do not bound the pose, including coordinate movers such
as KMR, are listed as `not_evaluated`.

**How a run ends, validation off.**
- Only the LLM ends the outline, with `decision: "recovery_complete"`. Code neither refuses nor
  answers it. Code never ends the outline by itself in this mode.
- Code then searches declared nominal events, at most 6 steps
  (`multi_turn_diagnostic.nominal_connection`), including the interrupted tasks with their
  recorded parameters, for actors not reported faulted. It records the connection, or
  `not_connected_reasons`, in the draft and in the closing turn ("Handoff to nominal work — no
  LLM call" in the UI). A wrong declaration is the user's recorded failure.
- The only replies code sends back are format problems and an exact repeat of an accepted event
  in the same state. The run stops after 3 refused replies in the same state
  (`selection_unresolved`), on repeated lookups with nothing new (`NEEDS_CONTEXT`), or at 20 turns
  (`outline_draft.DRAFT_MAX_TURNS`, a cost limit; lowered from 80 on October 19).
- Custody is recorded in the turn audit, not enforced, when every check is off. Placing a part
  elsewhere releases its holder. A reported fault clears only for a state value the resource's
  model declares.

**What code decides (state in the paper).** The failure facts and reach distances are computed
by code, and the LLM compares them. Reach uses declared cylinder envelopes. The final nominal
steps after the declaration are found by code. Nothing judges feasibility or safety yet.

**Files.** `modes/multi_turn_outline_facts.py` (new), `modes/multi_turn_prompts.py`,
`modes/multi_turn.py`, `modes/multi_turn_diagnostic.py`, `modes/multi_turn_validated_event.py`,
`modes/multi_turn_context.py` (all under `agents/intelligent_product/replanner/llm_recovery/`),
`recovery_framework/outline_draft.py`, `failure_grounding.py`, `slippage_execution.py`,
`agents/intelligent_product/product_recovery_controller.py`, `ui/components/recovery_turns.py`,
`ui/components/part_slippage_recovery.py`, `ui/recovery_evidence.py`. Tests:
`test_multi_turn_diagnostic.py`, `test_outline_draft.py`, `test_failure_grounding.py`,
`test_part_slippage_recovery.py`, `test_project_pages.py`, `test_recovery_request_records.py`.
Full suite on October 19: 190 failed, 5,729 passed; the failures are the known baseline set.

**Runs on `capture_77666affc1b9468ca157cf366d706dc8` (Part slippage, schema 2), under
`environment_runs/recovery/outline_drafts/`.**

| Draft | Model, effort | Calls | Result |
| --- | --- | --- | --- |
| `draft_4579f38026dd4b0b8091b46553d12fef` | gpt-5.4, medium | 8 | Uses the `state_variables` and `pose_relations` lookups and `pose`; KET4 to the buffer, ur5e-3 takes gear_small, clears ur5e-4, assembles; nominal `move_home`, `pick_approach`, `pick_grasp` of KET4. Exposed the unclear table, the fixed-list unknown rule and a stale location after a stated pose (all fixed October 19) |
| `draft_f7c5a94605c54baf83e8a03a21076c9a` | gpt-5.4, medium | 5 | Clears ur5e-4, looks up reach, ur5e-3 sets KET4 down and assembles gear_small, LLM declares complete; nominal connection ur5e-3 `pick_approach`, `pick_grasp` of KET4 |
| `draft_85ee4684d69a43d2b615e9af6aba4b71` | gpt-5.4, low | 5 | Same plan and connection |
| `draft_8bfa737c6a444de3b8668acda31c05c0` | gpt-5.4, medium | 6 | Same plan (before declaration was required); two lookup turns wasted by a since-fixed rejection |
| `draft_f2a9fb151f71496c8333cd3c43971853` | gpt-5.4, medium | 9 | Failed: omitted `held_part`, custody refusals looped; fixed by recording custody and releasing placed parts |
| `draft_9fb1a36de41342cb8a360475da42c667` | gpt-4o | 8 | Failed: no lookups, invented capabilities (ur5e-1 "serves" the board), assembled the gear three times, invalid fields; stopped `selection_unresolved` |

Earlier drafts in that folder are superseded attempts from prompt development. They are
small (22 MB in all) and can be removed once the user agrees.

**Known issues and follow-ups.**
- The template has only been run on Part slippage. Machining
  (`capture_4d1e4cc0fd9b4e07812ce7cbb01d2c87`), Conveyor
  (`capture_3aed616530294a4bbdf353000fc69596`) and ur5e-1
  (`capture_d1ca51654ff7423193d1bd13b146964d`) are untested with it.
- Machining (`capture_4d1e4cc0fd9b4e07812ce7cbb01d2c87`), October 20, gpt-5.4 medium, every check off:
  - `draft_afb0689f4e214ee0b425c411289cb6df` (6 events, 8 calls): ur5e-1 put KET4 "at Conveyor" and
    ur5e-2 took it "from Conveyor"; no event moved it along the belt. M2's pause, the RGOCG4 unload and
    the square trim on M2 match the intended recovery. KET4 stayed in M2 and RGOCG4 on the tray; the
    handoff was not connected ("no sequence of up to 6 declared nominal events reaches its next task").
  - Cause: both robots listed a bare `Conveyor`. The Conveyor's own declared model puts a part ur5e-1
    places at `loading_position_1` and one ur5e-2 places at `loading_position_2` (`place_release`,
    `part_location.{part_name}`); its `advance_conveyor` moves parts downstream only. Its recovery
    descriptor lists `advance_conveyor` only for parts already on the belt, so the empty belt showed no
    capabilities. M1's loading position (x −4.05) is upstream of M2's (x −2.05), so the belt can carry
    a part from M1 to M2; the scenario file's "No conveyor route" line does not match the model.
  - Fix (uncommitted): `multi_turn_outline_facts._placement_positions` states, for a reachable place
    that keeps each part's position, where its own declared events put a part this resource places
    there (`places parts on Conveyor at loading_position_1` / `at loading_position_2`). Runs 2 and 3
    showed it as `Conveyor (loading_position_1)`; it became a separate item after run 3 copied that
    text into its values. `outline_draft._inputs`
    keeps each owner's declared events as `environment_model_events`, because drafts with every check
    off drop `resource_evidence`. Tests: `test_a_reachable_place_states_where_its_own_events_put_a_part_placed_there`,
    `test_draft_states_the_position_a_place_declares_for_a_resource_placing_there`. No re-capture: all
    four captures' `beliefs.json` hold these declarations.
  - Rerun `draft_a48f24ab977e4d07a16cd2bd30ec5b59` (11 events, 16 calls, 537 s): the positions were
    shown, but the same transfer recurred (step 7 places KET4 on the Conveyor, step 9 picks it up, no
    belt move). The Conveyor was looked up only in turn 15, and its snapshot still showed
    `part_location.KET4_Square_4mm: null` after the placement: the unknown-values rule does not cover a
    per-part variable whose values are positions. Also: M1, the faulted machine, unloaded KET4 to its
    tray itself; ur5e-2 picked RGOCG4 up and put it back on the tray; M2's interrupted job was cleared
    (`job_state` None) with no resume. Handoff not connected, same reason.
  - User decisions (October 20): the LLM states a place's position in `place_end_state`; code never
    picks it, so option (a) above (code records the declared position) is dropped. Run 2 is recorded as a
    validation-off failure: the LLM had the positions and still planned an impossible pick. Expected to
    be caught by RA feasibility with physical evidence (Eq. 12, ur5e-2's reach), not by CCA.
  - Fixed (user, October 21; uncommitted), the joint transition of a part move: in $G=\Vert_i G_i$ a part
    move also changes the place's owner, and the LLM states that part ($\nu_j(x^R_k)$, `place_end_state`).
    - Records in view: `_displayed_fields` and `_resource_row` show a resource's part records even when
      empty, namely held_part, variables whose declared values name parts, and per-part entries
      (`zone_1_part none`; `part_location: none (values loading_position_1, …, output_nest)`). They also
      show the place inside the resource that a record belongs to, from the owner's declared events
      (`staging_part none (M2 staging tray)`; `multi_turn_outline_facts.declared_record_places`; only
      events by another resource count).
    - Unstated means "not stated": `_mark_places_unknown`, through `_place_records`, also covers a part's
      own entry (`part_location.<part>`) and, for a place inside a resource, only the owner variables its
      declared events set there (M2's `staging_part`, not the rest of M2). The Resources row, its header
      ("state not stated") and the snapshot lookup show "not stated" (`multi_turn_diagnostic.NOT_STATED`)
      instead of the old value; the Resources header says what it means. Code still never picks a value;
      draft records keep "unknown" with the reason.
    - Tests: `test_a_part_s_own_entry_on_a_place_reads_not_stated_until_the_event_states_it`,
      `test_a_place_inside_a_resource_changes_only_the_owner_record_its_events_declare`.
    - As for a whole place, naming the owner in `place_end_state` states its change; values left out
      stay as they are (the rule's own wording). Only an owner the event does not name is marked.
    - General fixes batch (user, October 21 night; uncommitted). Proposals came from a read-only
      workflow (`wf_b4e15fd3-ed8`: six proposers and one reviewer). They were applied in the
      reviewer's order (A7, A9, A1, A3, A2, A5-snapshot, A8, A6, C18, A4, A5 merged, B10 with fix,
      which includes B11) and pass 1109 tests across nine suites. No resource, part or scenario names
      appear in the added code, and no scenario words in the rendered prompts.
      - A1 `multi_turn._projected_state_value`: the derived `pose` reads `current_pose`, so a first
        stated pose no longer records a false "start condition unknown".
      - A2 `_resource_row`: a pose an earlier event stated reads "last stated pose", not "last measured".
      - A3 `_clear_pose_of_unheld_part`: a part keeps a stated pose only while the stating resource holds it.
      - A4 `faulted_replaced_work`, `_public_pending_tasks`: in drafts, `pending_tasks` leaves out
        replaced work that needs a resource still reported faulted (reviewer's option a; the user may revisit).
      - A5 `current_values`: in drafts, `/resources/<name>/state_variables` adds each variable's current
        value ("not stated" where an event left it unstated), as the parts lookup does (option a).
      - A5-snapshot: the snapshot lookup shows "not stated" also after the first accepted event.
      - A6 `with_named_domains`: a lookup domain names a resource listed by JID and by name once.
      - A7 `declare_derived_conditions`: `processCompleted` only on resources whose declared events add to it.
      - A8: "Recovery Events So Far" uses the reply format's key `resource`.
      - A9 `_inside`: a pose lies in a region only if it states every axis the region bounds; the
        recorded reason reads "no named place's recorded area contains the stated pose".
      - B10 `_neighbor_guards`, `_declaring_neighbors`: the handoff entry check and the connection search
        also check the guards another resource's own declared events put on a joint event (for example
        where a part must be to be picked there). A neighbor that declares such events joins the search
        scope, so its own events (`advance_part`) can connect. Unstated, private or undeclared neighbor
        values are recorded as assumptions rather than blocking (defaults the user may revisit).
      - B11 `_bookkeeping_assumptions`: a connection step's private guards are recorded as assumptions.
      - C18 `_normal_work`, `_part_normal_work`: each Recovery Goal line names the recorded task that
        normally produces it and the current reported fault of the resources it needs, for example
        `KET4_Square_4mm: process trim, result square (normally M1's machine_part; M1 unavailable)`.
    - Runs after the batch (October 21, 02:13; every check off; each part move stated its neighbor's change):
      - Part slippage `draft_32973b0d353b4e5c8e1cb2c7882c8607`: intended recovery; handoff complete. The
        connection now includes the buffer's own three `advance_part` steps (zone 1 to zone 4) before
        ur5e-3's pick (B10), and records ur5e-3's bookkeeping guards as assumptions (B11).
      - Conveyor `draft_ac13c4b5a95f48d1893714444b155be3`: handoff complete. Detour: ur5e-1 put KET4 on
        the M1 tray, took it back and loaded it into M1, so the KMR could pick it at M1. The KMR's
        declared reachable locations are Storage, M1 and M2, not the trays, while the intended recovery
        has the KMR collect from the M1 tray. Setup question: does the KMR reach the trays?
      - Machining `draft_85d421ca5e4e4264a40a7c8434b42518`: paused M2, stated the belt position. Errors:
        ur5e-2 took KET4 from `loading_position_1` (LLM; RA with physical evidence); M2 itself moved
        RGOCG4 to its tray (LLM; no primitive, RA); completion once KET4 was trimmed, leaving RGOCG4
        paused on the tray (LLM, with a missing fact: M2's line then shows only `job_state completed`,
        and RGOCG4's paused job is out of view, item 14). Handoff correctly not connected.
      - ur5e-1 `draft_e86e5ac20cc042d08f23a917a0497041`: handoff complete with the Conveyor assumption;
        the KMR placed KET4 on the M2 tray, which is not among its declared reachable locations (LLM,
        or the same tray-reach question); ur5e-2 then put it on the belt at `loading_position_2`.
      - No clear code bug found, so no extra round. Only the stale `goal_lines` docstring was corrected:
        the handoff status is never told to the LLM, by design.
    - Fixed (user, October 21 morning; uncommitted):
      - Item 14, jobs by part: `multi_turn_diagnostic.machine_jobs` and `_project_machine_jobs` keep
        each resource's recorded jobs. A stated `job_state` changes the job of the part the resource
        holds, or else its last retained job, as the declared variable's description says. A job stays
        recorded when its part leaves. The Resources row shows them
        (`jobs: RGOCG4-50_Round_4mm running (processing_fraction 0.051)`). The connection search
        branches copy the record (`_branch`). Test:
        `test_a_machine_keeps_each_job_by_its_part_so_a_paused_one_stays_in_view`.
      - Bias items 2 and 3, prompt wording: the Task line now reads "only your recovery events change the
        planned state; a job already running keeps running until an event pauses or stops it", and the
        Recovery Goal header "Your events, and then normal production's remaining planned tasks, must
        achieve it" (it was "Your events can achieve it, or leave the rest to normal production").
      - 1110 tests pass across nine suites.
      - Runs (09:57): Conveyor `draft_3e0bf4838fed4f7ababcef0f32c055ed` follows the intended recovery in
        two events (ur5e-1 to the M1 tray, the KMR from the tray to buffer zone 1). Part slippage
        `draft_7b6ae5719a2c4c58b438f95e6fb03d3b` is the intended recovery. ur5e-1
        `draft_8a8b775ad1044d559f64e42d7c7e9159` takes the KMR, M2 tray, ur5e-2, belt route; it also
        paused M2 "for recovery access" and never resumed it, yet the handoff reads complete (M2's next
        entry guards only that M2 is loaded with RGOCG4). Machining `draft_7f6a8c0699f0400c8d721f4b3d81666d`
        showed `jobs: RGOCG4-50_Round_4mm none`: a stated null `job_state` (no active job) was taken as the
        job's state. Fixed: a null `job_state` leaves retained jobs as they are (test extended).
      - Machining rerun `draft_9fff66c30f1745d38fbf7df88bdabdef` (14 calls): paused M2, staged RGOCG4,
        stated every tray and belt change; the last prompt showed `jobs: RGOCG4-50_Round_4mm paused,
        KET4_Square_4mm completed`. It still declared completion once KET4 was trimmed ("satisfies the
        interrupted recovery goal"), with RGOCG4 paused on the tray; handoff not connected. It also took
        KET4 from `loading_position_1` (LLM; RA) and moved RGOCG4 off and back onto the tray. Open (user
        decision): the Recovery Goal covers only the interrupted outcome, and nothing states the paper's
        handoff condition H, that normal production's remaining planned tasks must be able to continue.
      - Decided (user, October 21): the completion rule states H (Eq. 6): "When you judge that normal
        production's remaining planned tasks can continue from the state your events leave, return only
        `thought` and `decision: "recovery_complete"`; normal production then takes over." It was "When you
        judge the failure handled … normal production then takes over where it can", which contradicted
        the paper. The LLM still decides completion; no verdict is given. The next-planned-task fact was
        declined, as a fact surfaced only because a run failed.
      - Prompt frozen (user, October 21). From here the outline prompt changes only where it disagrees with
        the paper's formal definitions or where code states a fact wrongly. Other failures are LLM reasoning,
        left to validation (Algorithm 1) and reported as validation-off results. Next: the validation-only
        fixes (fingerprint and the rest), then the RA plus physical evidence trial.
      - Machining with the frozen prompt, `draft_d84e7ef79e5a4cfd93abf0e2d781b6bf` (20 calls, 1264 s,
        `draft_incomplete`, turn budget). The completion rule worked: it kept going after KET4's trim. It
        reasoned "KET4 is stranded at Conveyor.loading_position_1, where ur5e-2 cannot take it to M2
        directly", and used the KMR from the belt to M2; M2 trimmed KET4. Validation-off result (LLM
        reasoning): step 3 stopped RGOCG4's job to select square despite `restart_stopped_jobs false`, so
        H can never hold (pause-versus-stop trap; RA and CCA accept a declared stop, only H fails); step 1
        used `interrupted` instead of `paused`; KET4 went belt, tray, belt (steps 4-8), spending the
        budget. No code bug: M2's row read `jobs: RGOCG4-50_Round_4mm stopped, KET4_Square_4mm completed`.
    - Bias review (October 21): no scenario words in the prompts, no resource names in the code, the
      same seven safety files for all four scenarios, the same structure for every resource. Sources of
      bias: (1) which facts were surfaced was chosen from these four runs' failures (medium; check on
      held-out failures); (2) the Recovery Goal header "or leave the rest to normal production" invites
      deferring work (medium); (3) the Task line "Until normal production takes over, only your recovery
      events change the state" hides that running jobs progress (medium; the paper's monitors cover
      running tasks); (4) omissions steer too: reach numbers, the Conveyor's movement, retained jobs,
      the KMR's pose (low to medium); (5) a fault is repeated in Failure, Resources and Goal (low).
    - Configuration note (setup data, not a framework bug; user, October 21): ur5e-1's and ur5e-2's
      declared workspace bounds (`gripper_reach`, "computed from the robot kinematics for a top-down grasp
      of a loose part", `workflow_execution._computed_gripper_reach`) span z 0.75–0.90 m from a base at
      z 0.80, but their work places lie higher: `M1.staging`/`M2.staging` z 1.00–1.12,
      `M1.workholding`/`M2.workholding` z 0.94–1.18, `Conveyor.loading_position_1`/`_2` z 1.015–1.065.
      ur5e-3/ur5e-4 (base z 1.021, band 0.971–1.121) cover theirs. With RA on, correct tray, machine or
      belt events of ur5e-1/ur5e-2 may be refused on reach height; fix the configured band or base height.
    - Renamed (user, October 21): `place_end_state` is now `neighbor_end_state` in the prompt rule, the
      reply format, the code (`_apply_neighbor_end_state`, `_candidate_neighbor_end_state`,
      `schema_with_neighbor_end_state`, `_NEIGHBOR_END_STATE_RULE`), the Recovery turns column
      ("Neighbor changes") and the tests. The name follows the semiconductor framework, where resources
      that share a product state are neighbors. Drafts saved earlier keep `place_end_state`, which the
      turns view and `OutlineDraft._named_record` still read. Entries above dated before October 21 keep
      the old name.
    - Runs, October 21 (00:36, all four scenarios, every check off): the LLM stated the place's change
      for every part move (10 of 10; no "not stated" marks). Part slippage
      `draft_c2e626318b0f4be5b2bdd4afbae483e5` and Conveyor `draft_c0fe26c3fc034fb193c467105bb1ca3e`
      match their intended recoveries and hand off. ur5e-1 `draft_63aad79537be495b9764a4bb85ded9ab`
      follows the intended KMR-to-belt route (KMR places KET4 at loading_position_1) and hands off with
      the Conveyor assumption. Machining `draft_c652e2ee2c1047e898ab5c78d11071c9` stated every tray and
      belt change, but ended with KET4 on the belt, saying normal production would carry it to M2 and
      trim it (LLM reasoning; the handoff check reports the goal does not hold).
  - Rerun `draft_2ec9936ee5d94a2d805a21f273bca4b2` (7 events, 11 calls, 518 s), same code: ur5e-1
    unloaded KET4 from M1 to its tray; ur5e-2 unloaded RGOCG4 to M2's tray; the LLM reasoned with belt
    positions and stated `Conveyor: part_location.RGOCG4-50_Round_4mm = loading_position_2` in
    `place_end_state`. Failures: (1) it copied the display text into values (`part_location` and
    `resource_location` "Conveyor (loading_position_1)"), which no check flagged; (2) it declared
    completion with KET4 untrimmed on the belt ("normal production can complete the trim step"); the
    handoff check refused it (Recovery Goal does not hold; M1 is still faulted); (3) it put the untrimmed
    RGOCG4 on the belt, cleared M2's interrupted job, and moved the KMR to M1 and then Storage for no use.
  - Done (user, October 20): the position is a separate item (`places parts on Conveyor at
    loading_position_1`), so a copied place name is always a declared one.
  - Run 4 `draft_1fe8f59879214db6b4b9908b178c1614` (8 events, 10 calls, 437 s): M2 paused; ur5e-1
    unloaded KET4 to M1's tray; the KMR left M2 (rule 3); ur5e-2 unloaded RGOCG4; no copied display text.
    Errors: (1) the same belt transfer (ur5e-1 places KET4 on the Conveyor, ur5e-2 loads it from there,
    no belt move); LLM reasoning, expected to be caught by RA feasibility with physical evidence;
    (2) no `place_end_state` for the Conveyor; LLM, plus the deferred unknown fix; (3) M2 stopped
    RGOCG4's paused job to select square, so RGOCG4 can never resume; a missing fact, not reasoning:
    M2 declares `job_control` (`resume_after_reload: true`, `restart_stopped_jobs: false`), but drafts
    hide `job_control` (`outline_draft._HIDDEN`), and no recorded decision explains why; RA would accept
    a declared stop and CCA has no rule against it, so validation does not supply it; (4) declared
    complete with KET4 in M2 and RGOCG4 not resumed; LLM, refused by the nominal-continuation check
    once it is on.
  - Fixed (user, October 20; uncommitted): `job_control` is no longer in `outline_draft._HIDDEN` (the
    set was added whole with the file in `c40aba2`, with no stated reason for it), and
    `multi_turn_outline_facts._capabilities` states a resource's declared job-control flags by name and
    value: `M2 … job control: resume_after_reload true, restart_stopped_jobs false` (M1 too). The
    Resources header lists "where it places parts" and "job control". Tests:
    `test_a_resource_states_its_declared_job_control_flags`,
    `test_draft_states_a_resource_s_declared_job_control`.
- All four scenarios, October 20 (22:31, user request; gpt-5.4 medium, every check off, settings and
  safety files from each capture's latest draft; script `scenario_run.py` in this session's scratchpad).
  Errors were classified by a read-only workflow (one analyst and one verifier per draft).
  - Part slippage `draft_66d30ad65cc84e959dc726686fd62ae3` (4 events, 8 calls): matches the intended
    recovery; handoff connected. The handoff holds only on ur5e-3's reduced pick model: the buffer's
    own joint pick guard needs KET4 in `zone_4_part`, and the LLM put it in zone_1 (code check gap;
    nothing in the prompt says where nominal work picks from the buffer). gear_small's process record
    never gets the assembly that rule 5a reads.
  - Conveyor breakdown `draft_87419318b57546379276f2f33257d0cb` (3 events, 7 calls): matches intended
    steps 1–2; handoff connected on the LLM's stated `zone_1_part` only. LLM pose errors: the KMR's
    buffer base pose lies inside the Conveyor footprint, and its tray pick pose cannot reach the tray.
  - Machining `draft_0ca75fb5df3f4b298a1f389cafc14071` (7 events, 10 calls): stated
    `Conveyor: part_location.KET4_Square_4mm = loading_position_1`, paused rather than stopped, but
    ur5e-2 again took KET4 from loading_position_1, and step 7 "resumed" RGOCG4's paused job as KET4's
    square trim. RGOCG4 was left paused on the tray; handoff correctly not connected.
  - ur5e-1 breakdown `draft_42e06025bc8242978f07833491afdd25` (5 events, 8 calls): the handoff reads
    complete but is not trustworthy. The check's only required entry is M2's RGOCG4 trim, which held at
    capture; KET4's resumed Conveyor entry is dropped without notice (code check gap). KMR base poses
    are physically impossible; part_state "completed" is a product-tracker value no resource declares.
  - Validation set-up, verified in code: OutlineDraft refuses all three checks at once
    (outline_draft.py:204), and only OutlineDraft uses this prompt (`outline_diagnostic_bypass`), so
    all three checks run the validated workflow with the old prompt. CCA without physical evidence
    cannot evaluate the selected occupancy rules (outline_checks.py:59). With any check on,
    `place_end_state` is not offered (multi_turn.py:2462, 3770), and RA and CCA do not read it.
    Expected from static reading only: with RA on in a draft, RA may answer NEEDS_CONTEXT on every
    robot and KMR event, because drafts add derived `pose`/`processCompleted` to the models.
  - Fixed (user, October 21; uncommitted): the handoff check no longer drops a resumed entry that no
    declared event binds. `multi_turn_diagnostic.unbound_resume_entries` states each one, and `handoff`
    adds it to `assumptions` ("no declared event of <resource> binds the resumed <function>; whether
    nominal work continues there is not checked"). `required_nominal_entry_ids` is unchanged, so no
    obligation changes. The draft message and the Recovery turns "Handoff to nominal work" card show
    "assumed, not checked: …" next to "takes over". Of the four captures, only ur5e-1 has such an entry:
    the Conveyor's `advance_conveyor` `pending_4` (its events are grounded on the empty belt at capture).
    A replay of `draft_42e06025…` now records that assumption; saved drafts keep their old records.
    Tests: `test_a_resumed_entry_no_declared_event_binds_is_stated_as_not_checked`, and the handoff
    card case in `test_closing_turn_shows_the_nominal_handoff_by_resource_name`.
  - Trial, RA feasibility plus physical evidence on (CCA off), Machining,
    `draft_ed7cd24af1e24a24b1f0a6580a7cc419` (3 calls, 178 s): `NEEDS_CONTEXT`. RA refused every
    candidate with "Resource declarations changed after capture" (outline_validation.py:377-379).
    Cause: with checks on, `OutlineDraft` gives each resource its owner's model, then
    `declare_derived_conditions` adds the draft's `pose`/`processCompleted` to that same model;
    `validate_partial_event` fingerprints the extended model (outline_checks.py:146-150), and the owner's
    own model no longer matches. RA also treats `pose` as undeclared (static reading), so the fingerprint
    alone may not be the last blocker. Next step: the user's choice of fix.
- Conveyor: the handoff is refused while the resumed work needs the broken Conveyor; a later
  resume point is not chosen yet. The Conveyor declares no `resource_state` domain, so the
  fault-clearing rule falls back to "state no longer equals the report".
  - First run, `draft_7199f1650ca741a1948360ac0728b14d` (gpt-5.4 low, stopped by the user after
    7 calls): ur5e-1 moved KET4 between the M1 staging tray, M1 and home five times and never
    declared complete. Causes: (1) the Recovery Goal is the broken Conveyor's own fields
    (`next_locations`, `delivered_part`), which no other resource can produce; (2) the capture
    resumes nominal work at ur5e-1 placing KET4 onto the Conveyor, and its recorded handoff
    conditions contradict each other (KET4 in ur5e-1's gripper and at `Buffer For Machined
    parts`); (3) KMR declares only Storage, M1 and M2, and nothing in the prompt shows that it can
    move to coordinates. Next step: resume at the first recorded entry that needs no faulted
    resource (ur5e-3's pick from the buffer), state the goal as KET4's outcome there, and decide
    how a coordinate-move capability is shown.
  - Fixed October 19 (uncommitted): `multi_turn_diagnostic.resume_after_faults`, called by
    `OutlineDraft`, moves resumed tasks that need a faulted resource (as actor or as a place) into
    the replaced work and drops their recorded handoff conditions (`source_task_id`).
    `handover_conditions` then states the goal as what the remaining resumed work needs of those
    parts at their next remaining step. `resume_entries` keeps, per part, only the earliest resumed
    entry in recorded plan order (a task's part is its `part_name`, else a known part another
    non-place parameter names, such as `next_locations`); later steps follow from nominal work, so
    their recorded conditions (`entry_conditions`) and entries are not handoff requirements. The
    Conveyor goal is therefore "Buffer For Machined parts: zone_1_part KET4_Square_4mm" (the buffer
    entry's declared guard; ur5e-3's later pick no longer counts). The ur5e-1 breakdown now restarts at
    M2 and the Conveyor instead of also requiring the buffer and ur5e-3. Part slippage, Machining and
    ur5e-1 breakdown goals are unchanged (checked by diff). An offline replay of ur5e-1 → M1 staging
    tray, KMR → buffer, buffer receives KET4 in `zone_1_part` hands off to the buffer and ur5e-3;
    without the buffer event, the handoff is recorded as not connected. The draft turn budget is now
    20 (D). Coordinate moves (C) are still open; no live Conveyor run yet.
  - Second run, the first with the buffer goal: `draft_6d122269bb53462f9059af8431928557` (gpt-5.4
    medium, 7 calls, 4 events, stopped by the user on October 20). ur5e-1 set KET4 down at M1, moved
    it to the M1 staging tray, took it back home and "stabilized" at M1. No turn mentioned the
    buffer, the goal or the KMR, and the KMR was never looked up. Causes: (1) the Failure section
    kept saying "KET4_Square_4mm: in_gripper, held by ur5e-1" as measured, while Resources said "not
    re-measured", so the LLM undid its own steps ("The freshest failure report says ur5e-1 still has
    KET4"); (2) it planned no route to the goal and waited for others ("pending conveyor recovery",
    "for replan", "manual intervention"). The KMR row already shows it holds parts, works at M1 and
    moves to any position, so no resource-specific change was made.
  - Fixed October 20 (user-approved, uncommitted), wording only, in `_render_failure_outline_prompt`:
    - Task and Role: "Until normal production takes over, only your recovery events change the
      state." Between $s_F$ and $s_N$ only recovery events occur (Eqs. 5–6), with no human steps.
    - "Plan how to get from the current state to the Recovery Goal. Propose 1–5 ways to do the next
      step of that plan, then choose one with `selected_candidate_index`." `thought` gives the plan
      in two short sentences (§5.1). The validated wording keeps "Do not select or rank candidates".
    - Once events exist, the Failure header reads "What was detected and measured at the failure,
      before your events. Resources and Parts below show the state after them." ($x_F$ versus
      $x^R_{k-1}$.)
    - Rejected by the user: "Normal production does not use a resource that has a fault". A faulted
      resource may still act through its primitives (the RA judges), and an event can leave a fault
      by setting another declared state; Part slippage's ur5e-4 reports `resource_state failed`.
      Kept 1–5 candidates: they are $R_{\text{neural}}$, the validated mode checks each, and the
      validation-off ablation keeps the same output.
    - Test: `test_the_prompt_asks_for_a_plan_and_dates_the_failure_facts_once_events_exist`.
    - Next: the user restarts the UI and reruns Conveyor (medium), then Part slippage once. Delete
      the failed attempts `draft_6d122269…` (1.2 MB) and `draft_7199f165…` (1.1 MB) after the rerun
      is compared.
  - Third run, after those fixes: `draft_8bf5b6d05b6642098baedd8e25211852` (gpt-5.4 medium, 7 calls,
    3 events, status `draft`). ur5e-1 stages KET4 on the M1 staging tray; the KMR takes it at M1;
    the KMR places it in the buffer at pose (-0.68, 0.5, yaw 0); the LLM declares complete
    ("ur5e-3 already has pending work to pick it from the buffer"). The handoff is complete with no
    nominal steps; zone_1_part and zone_2_part count as assumptions because code marked the buffer's
    zones unknown (the KMR event cannot state the buffer's variables). The thoughts aim at the goal
    (turn 2: "The Conveyor fault blocks the intended transfer path from M1 toward the machined-parts
    buffer"); it looked up the KMR in turn 1 and ur5e-3's and the KMR's pending tasks in turn 4.
    For validation later: the stated KMR pose is the zone-1 slot itself, so the platform would stand
    inside the buffer (RA); ur5e-1 stays at the M1 staging tray while the KMR works at M1
    (`SAFE_M1_mutex`, CCA).
  - Fourth run, with `place_end_state` and the shorter rules: `draft_0aa3406e820345c8bb72224a2ed9ef14`
    (gpt-5.4 medium, 9 calls, 5 events). ur5e-1 stages KET4 on the M1 staging tray (states M1
    `staging_part` = KET4 in `place_end_state`) and goes home to free M1; the KMR takes KET4 (states
    `staging_part` = none), parks at (-1.35, -0.1, yaw 1.57) off the conveyor and buffer footprints;
    ur5e-3 takes it from the KMR and sets it in the buffer (states `zone_1_part` = KET4 and the KMR's
    `held_part` = none). The LLM declares complete; the handoff is complete with no assumptions and
    no unknown lines. User question: why does ur5e-3 not assemble KET4 right away? The goal is the
    restart condition of KET4's next planned step (the buffer's first advance), and the handoff only
    lets normal production restart there; skipping planned steps whose outcome the recovery already
    reached is not supported (an option for the user, see the October 20 answer).
  - Checked October 20 (offline replay, scratch `kmr_route_replay.py`): a KMR route (ur5e-1 sets
    KET4 down at M1; the KMR fetches it, drives beside the buffer and sets it in zone 1) and a
    handover route (the KMR brings it, ur5e-3 sets it in the buffer) both end, after
    `recovery_complete`, with a complete handoff. The buffer's zones are unknown after the transfer
    and recorded as assumptions; only `goal_reached` stays false, which validation-off drafts do not
    use. So the plan is reachable with the current code.
  - Fixed October 20 (user-approved, uncommitted): `/resources/<name>/snapshot` of a resource an
    event changed served failure-time values the projection never updates (live turn 5: ur5e-1
    "placed" at the M1 staging tray with `gripper_state closed`, `held_part_location ur5e-1`); the
    LLM had asked for it to settle its doubt. In a validation-off draft, `public_resource_snapshot`
    now keeps, for a resource named by an accepted event or its `condition_changes`, only its
    declared variables plus `_PROJECTED_SNAPSHOT_FIELDS` (identity, availability, pose records).
    The rule sits before the version-2 early return that drafts take after the first event; the
    validated path is unchanged. The Knowledge Base line now reads "every recorded value of a
    resource; once your events change it, its expected values after them". Replaying the live run's
    two events, the ur5e-1 snapshot is `{availability, named_poses, resource_jid}` plus the fields
    the rows show. Test (accepts the event through `handle_validated_candidates`):
    `test_a_changed_resource_snapshot_leaves_out_its_failure_time_values`.
  - Open (not fixed, October 20):
    - Fixed October 21: the reply format omitted resources that declare no `resource_state`
      (buffer, Storage, Conveyor), so they could not have their own events, while the event check
      would accept them (`_outline_candidates_response_schema` did not pass `declared_fields_only`).
      It now passes the flag, and the contract-2 format is the same for every resource (see "Reply
      format and checks"). Outlines drafted before this used the per-resource format, so re-run
      them before comparing results. Test: `test_contract_2_outline_event_gives_only_its_form`.
    - `/resources/<name>/pending_tasks` serves the recorded list, so it still shows replaced work
      and a faulted resource's tasks as pending.
    - Physical estimate from the review (unverified): buffer zone 1 is in the KMR arm's reach only if
      the platform parks within about 5 cm; the buffer holds parts lying flat while the KMR places
      upright; a KMR pick at M1 has never run live. This is for RA feasibility later.
- ur5e-1 breakdown: M1 area bounds make ur5e-1 read as outside M1.
- ur5e-1 breakdown goal: its Recovery Goal prints task parameters ("KET4_Square_4mm:
  origin_resource_location M1, destination_location Conveyor, reserved_by ur5e-1, loading_position
  loading_position_1") because ur5e-1's place onto the Conveyor declares no part effect and the
  Conveyor's next step (`advance_conveyor`) declares no guard; the capture records no condition for
  it either. Fix in the model declarations (a part effect on the place, or a Conveyor guard on the
  loading position) when working on that scenario; this may need its models re-derived.
  - First live run, October 20: `draft_16cfbce326e84d348411389ae6a15513` (gpt-5.4 medium, 3 calls,
    1 event). The failed ur5e-1 redoes the failed grasp ("secure_KET4_Square_4mm_for_transfer") and
    the LLM declares complete, leaving the Conveyor placement to ur5e-1; the KMR is never looked at.
    The goal line names ur5e-1 (`reserved_by ur5e-1`), and the fault clears because the event gives
    ur5e-1 a declared state ("picked"). The handoff check is empty for this capture: it reports
    `complete` even before any event (the replaced place declares no effect, "completion assumed",
    and no condition names KET4), so its "handoff complete" means nothing here.
  - Intended recovery (the capture's design, generic KMR abilities): ur5e-1 moves home; the KMR picks
    KET4 at M1, drives carrying it beside the conveyor and releases it on the belt (verified live
    October 15); normal production carries it on. Alternative: the KMR brings it to M2's staging
    tray and ur5e-2 loads the belt.
  - Fixed October 20 (user-approved "fix 1", uncommitted): a replaced final step whose own model
    declares no event for it is now checked by its part reaching the step's destination
    (`multi_turn_diagnostic.destination_outcome`), in the goal text (`_work_outcome`), in
    `_replaced_finals`/`open_replaced_work`/`goal_reached`, and in `_claim_interrupted_work` (only
    the part resting at the destination, with no resource holding it, claims it, reason "<part>
    reached <destination>"; an acting resource's same-named event no longer does). After the code
    review: the step is open whenever the part does not rest at its destination (a later move away
    reopens it), already-claimed work is not claimed again, and a resource's JID and name count as
    the same destination (`_at_destination`). Four-capture diff: only ur5e-1 changes; its goal is
    "KET4_Square_4mm: part_location Conveyor", and the handoff is no longer complete before any event.
    Offline replays (scratch `ur5e1_routes_replay.py`): the intended KMR route claims the step and
    hands off with no assumptions; the live run's retry also hands off, but visibly, through nominal
    steps by ur5e-1 (`place_approach`, `place_insert` to the Conveyor), because its "picked" state
    clears the fault. Test:
    `test_a_replaced_step_its_model_cannot_describe_is_done_when_its_part_reaches_its_destination`.
  - Next: re-run ur5e-1 breakdown once; if the LLM still has ur5e-1 grasp again, record it as a
    validation-off failure that RA feasibility should catch (a grasp by a gripper that reports
    failed).
  - Step 2 note: the captured robot and Conveyor models are reduced (no `place_release` onto the
    Conveyor, no Conveyor events); `nominal_des.py` declares them, and RA needs them.
  - First run on the new capture: `draft_fbc45b44cbcc4fadae005eca67c96e51` (gpt-5.4 medium, 4 calls,
    1 event). With "gripper not working" in the Failure section the LLM says "ur5e-1 cannot finish
    the pick" and has the KMR move KET4 from M1 onto the Conveyor at loading position 1, stating M1
    `part_name` none and the Conveyor slot in `place_end_state`; then `recovery_complete`. Handoff
    complete. But ur5e-1 stays at M1 while the KMR works there ("retreat_failed_robot_from_M1" was a
    candidate, not chosen), which breaks `SAFE_M1_mutex` during the KMR event; with validation off
    nothing checks it (the handoff check only looks at shared areas at handoff). Record as a
    validation-off safety failure; CCA (Eq. 13) should refuse the KMR event and send the LLM back to
    move ur5e-1 first. Also: "interrupted place_approach completed" was credited through the KMR's
    own same-named event (fixed effect `resource_state positioned`), a matching flaw the
    complete-models work fixes.
  - Follow-up (user, October 20): the area lookup told the LLM ur5e-1's position is in no M1 area
    because it tests one measured point (the tool point, z 1.30, above `M1.workholding`), while the
    recorded geometry shows the arm occupying `M1.front_access` and `M1.workholding`. The user judged
    reporting that occupancy in the prompt to be biased (it states the fact that triggers the rule),
    so it is not added. Only the wording changed: the pose_relations Knowledge Base line and the
    lookup's basis now say "(one point, not its full extent)".
  - Second run on the new capture: `draft_c92ace1067984b9096a6c2830229ed06` (same turn-1 facts as
    `draft_fbc45b44`, plus the "(one point, not its full extent)" wording). This time the LLM first
    moves ur5e-1 home ("First clear M1 so another recovery path can move KET4"), then invents
    `reset_gripper_fault` (ur5e-1 back to idle) because "ur5e-1 is the only resource that can reach
    both M1 and Conveyor", and re-picks KET4 with ur5e-1. Code accepts the reset with validation off
    (a declared state leaves the reported fault); RA would refuse it (no primitive repairs a broken
    gripper). Two runs, two different lapses: safety (fbc45b44) and feasibility (c92ace10).
  - Option B (user, October 20): the report now says the gripper fault is a hardware fault that
    needs repair, so a reset is not a valid reading (`failure_checkpoints.ROBOT_BREAKDOWN_GRIPPER_CONDITION`,
    used by the fault injection and checked by `evaluation_capture`). `ur5e1_10` captured
    `capture_b2d4286af67c4567b4bc0d0fd5ee55c7` (exit 0, verified, same starting state); its Failure
    line reads "device_fault_report (ur5e-1): resource_state failed, gripper not working (hardware
    fault, needs repair), reported_by gripper object detection". The default "ur5e-1 breakdown"
    observations point to it. The two earlier ur5e-1 captures (`capture_d1ca5165…`,
    `capture_129c6057…`), their run folders `ur5e1_08/` and `ur5e1_09/`, and their four drafts
    (`draft_16cfbce3…`, `draft_0d6613d3…`, `draft_fbc45b44…`, `draft_c92ace10…`) were archived to
    `/home/jongh/recovery_archives/20261020_ur5e1_superseded_captures/` (sha256 manifest, verified)
    and deleted on October 20 at the user's request: 370 files, 1.16 GB. The list is in
    `recapture_v2_20261015/ur5e1_08_09_deletion_list.json`.
  - Complete-models investigation (October 20, read-only, workflow `wf_a4e9df59-85f`; proposal,
    not implemented, pending the user's decision):
    - Every capture stores the complete nominal plant in `physical_evidence.environment_models`
      (12 resources, shared events by `event_id` with one side per participant = G = ||_i G_i;
      `environment_models.project_transition` replays the recorded steps). All 31 recorded plan
      steps of the four captures are declared there; the reduced `recovery_des_model` misses 6
      (robots come from the robot task registry: no `place_release`; the Conveyor's converted model
      has 0 events because its events are grounded on the empty belt at failure time).
    - Swapping the models in directly breaks the working Part slippage draft and the ur5e-1 goal
      (the matching code reads only the reduced format).
    - Proposed design: keep the reduced model for everything the LLM writes (schema, capabilities,
      place_end_state checks, derived pose/processCompleted, composition); bind each recorded plan
      step to its declared joint event (all participants' sides) when the draft is created and use
      that for the goal, claims, handoff H (guards of every participant) and the nominal redo; a
      completeness check; then remove the destination fallback. Prototype: 31/31 bound, both
      known-good drafts still hand off, prompts unchanged except two Part slippage lines; the
      ur5e-1 handoff becomes stricter (the Conveyor's next step's guards).
    - Two reviews ("revise") found what a complete version must also do: treat spliced recovery
      nodes (a failure during a recovery) as accepted events, not model gaps; decide fault blocking
      by participants, not location parameter names; treat a faulted participant's guards as unmet
      (x_F, no fault value in the complete models); include product effects in claimed outcomes;
      a generic alternatives policy (filter by joint guards); one joint-guard evaluator for H, redo
      and the bounded search; move the s_N search to the complete plant (the prototype's ur5e-1
      retry produced a wrong connection: a free registry approach to the board, then E#33); check
      participants' guards on redo; one name/JID rule for written values; re-check claimed outcomes
      at handoff; leave `outline_checks.validate_partial_handoff` rows unchanged. Estimate 2–3 days.
  - Re-captured October 20 (user-approved): the ur5e-1 and Part slippage fault reports were
    word-for-word the same ("failed", "gripper object detection") although only ur5e-1's gripper is
    broken, so the LLM read ur5e-1's fault as a missed grasp and retried it (`draft_0d6613d3`,
    stopped). Changes: `conveyor_fault._report_gripper_fault` adds `"gripper": "not working"` to
    ur5e-1's report (Part slippage's report is unchanged); `failure_grounding` passes every other
    recorded condition of a device report through as the device's own words (`_REPORT_FIELDS`);
    `evaluation_capture._verify_robot_breakdown_capture` requires the condition. `ur5e1_09` captured
    `capture_129c605761a940ba841b02ab243da444` (exit 0, verified, same starting state as
    `capture_d1ca5165…`). Its Failure line reads "device_fault_report (ur5e-1): resource_state
    failed, gripper not working, reported_by gripper object detection". The default "ur5e-1
    breakdown" observations now point to it (`part_slippage_example._DEFAULT_OBSERVATIONS`). Paper:
    A.1 (relevant resource conditions are observable). That capture was retired on October 20 with the
    other superseded ur5e-1 data (archived first; see the option B entry).
  - Live runs on the option B capture (October 20, gpt-5.4 medium, all checks off, run by Claude at
    the user's request):
    - `draft_df4f2479…` (ur5e-1): ur5e-1 moved out of M1 to the Conveyor (loading_position_1); the
      KMR took KET4 from M1 and placed it on Conveyor loading_position_2 from pose (-2.05, -0.2, 0),
      stating the Conveyor's slot; the LLM declared complete; handoff complete. No gripper reset.
    - Code bug found there and fixed (user-approved): a take whose end state gives the part
      `part_location: null` cleared the part's holder (`multi_turn_diagnostic._release_holders`
      compared the holder only with the new location), so the prompt read "KMR … holds
      KET4_Square_4mm" beside "KET4_Square_4mm completed, not held, not at any station" and the
      table showed "held_by None — code (it was placed elsewhere)". Now code never overrides a
      `held_part` the event states (the acting resource's end state, or `place_end_state` for another
      resource): a stated taker stays the part's holder whatever location the part is given, and only
      holders the event does not mention are released ("… is now held by <resource>"). Tests:
      `test_a_resource_that_takes_a_part_holds_it_whatever_location_it_states` and a hand-over with
      no location in `test_a_stated_holding_change_on_another_resource_is_also_the_parts`.
      Replaying `draft_df4f2479…` gives "held by KMR" and the same handoff as recorded; the fix adds
      no value the LLM did not state.
    - Reruns with the fix, all declared complete with a complete handoff: Part slippage
      `draft_4c0880a7…` (ur5e-4 retreats; ur5e-3 returns KET4 to the buffer, picks gear_small and
      assembles it; nominal work re-picks KET4), Conveyor `draft_1f98b50e…` (ur5e-1 stages KET4 on the
      M1 staging tray; one KMR event takes it to buffer zone 1, stating M1's and the buffer's changes),
      ur5e-1 `draft_a3c0ad20…` (ur5e-1 to the M1 staging tray; the KMR takes KET4 from M1, "held by
      KMR", and places it on the Conveyor).
    - LLM choices left for RA/CCA (coverage check in progress): KMR end poses on or overlapping the
      Conveyor footprint ((-1.35, 0.5) and (-5.0, 0.5)); the faulted ur5e-1 parked next to where the
      KMR then works; one KMR event spanning drive, pick, drive and place.
    - Open (asked by the user): with no `place_end_state` for the Conveyor, its per-part slot
      `part_location.<part>` keeps its old value (null) while the part reads "at Conveyor":
      `_mark_places_unknown` marks unknown only variables whose values name the part or that hold it
      now, and misses a variable named after the part. Nothing reads the slot with checks off (the
      goal uses the part's own location).
- Machining: the capture's KMR lacks a `move_base` primitive (primitive stage).
- Progress is not measured with validation off; per-turn goal metrics for analysis are not
  recorded yet.
- Primitive composition turn headers still show a JID.
- An event that states both a `pose` and a `resource_location` keeps both, even when the pose lies
  outside that place (`draft_4579f380` step 2: ur5e-3 at the buffer with a pose at x 0.25). With
  checks off, code does not judge it; RA feasibility should.
- The validated (checks on) path does not use the new template.
- **Knowledge-base exposure (October 21, decided, not implemented).** Principle (user): one
  knowledge base 𝒦 with the same *facts* in every stage and mode; nominal *procedures* only for the
  RA, which grounds an already chosen event; the remaining plan stays visible to the outline as goals.
  - Facts: resource and part state, capabilities, geometry, regions, reach, pose relations, safety
    rules. Remaining plan as goals: `pending_tasks` and the goal line's "(normally …)". Procedures:
    task contracts, primitive catalogs, contracts and signatures, capability decompositions.
  - Today the outline menu differs by mode (`multi_turn_context.resolve_outline_context` and the
    diagnostic refs at lines 695–710). Validation-off drafts serve `/geometry`, `/regions`,
    `gripper_reach` and `pose_relations`; validated runs keep them private (line 767). This breaks the
    experiment rule that all four ablation configurations get the same lookups.
  - Drafts also serve `/resources/<jid>/primitives` (callable primitive signatures), the one
    procedure leak into the outline. Remove it in both modes (scope rule: primitives appear only at
    composition; user concern that nominal procedures would anchor the outline).
  - Composition (`multi_turn_primitive_generation` resolver) lacks geometry, reach and pose relations.
  - Planned change, to show the user as a diff first because the lookup list is part of the frozen
    outline prompt: one shared resolver for facts; validated runs serve the same facts as drafts;
    remove `primitives` from the outline; give composition geometry, reach and pose relations. Draw
    experiment runs only from drafts made after the change. Check the anchoring concern with the
    new-versus-nominal event counts (Table 7 in the figure mock-ups).

**Next steps (options).**
1. Run turn 1 on the other three captures with validation off, then fix anything the template
   gets wrong for them.
2. Repeat Part slippage 5–10 times per setting (gpt-5.4 medium, gpt-5.4 low, gpt-4o) for
   baseline counts by failure type.
3. Add validation, RA feasibility first, with custody inside it; then CCA safety and physical
   evidence; then measure errors caught against the baseline.

## Schema-2 re-capture — October 15, 2026 — Stage 0 done; four Gazebo stages pending

The user will re-capture all four failures in Gazebo, one scenario per stage. Every new
`observations.json` must hold raw facts only. Plan:
`~/.claude/plans/what-do-you-mean-synchronous-kahn.md`.

**CCA during capture (decided October 15, user questions answered).** Captures run with
the CCA bypassed, as before. Live CCA capture is not possible with the current code:
- the capture session exists only under the bypass;
- under a live CCA, nominal tasks run through `live.execute_nominal`, which never
  attaches the slip trigger (`workflow_execution.py:453`);
- the CCA loads one spec file;
- nominal tasks have no safety contracts for gripper steps or non-arm resources.

Read-only verification, by three investigators plus three skeptics on code and the
October 10 captures, established the following:
- **ur5e-3 waits because the harness holds it.** With the fault armed (the default),
  `failure_checkpoints.holds_task:184-196` holds every ur5e-3 task once its KET4 pickup is
  acknowledged. It is checked at `environment_runtime.py:982` and `:1147`, before any CCA
  branch. ur5e-3 parks above Buffer zone 4, outside the assembly zone: nearest link at
  y = 0.357 against the zone edge at 0.25. It waits there while ur5e-4 lowers the gear and
  the gear slips.
- **The paper must say the checkpoint is staged.** A live CCA would not reproduce the same
  capture. The harness also stages holds the CCA would not make: ur5e-4 waits for
  ur5e-3's pickup, and both pickups are released together. With the mutex alone, ur5e-3
  would enter the board first.
- **Do not disarm the fault in the UI.** Without the harness, nothing keeps ur5e-3 out:
  `place_approach` does not reserve the board workspace.

**Stage 0 changes (code and tests, no Gazebo):**
- **Native preview model readers** use `read_capture`. Before this, the archive step
  rejected every schema-2 capture.
- **Custody after a device fault report** (`gazebo_safety_preparation`): the gripper is
  checked against the robot's own `_held_part`, not the product's belief, which lags
  until grounding. The disagreement is recorded as `custody_after_fault_report`. Without
  a fault report, the old strict check is unchanged.
- **Grounding:**
  - writes the dropped part's `observed_pose` as an `{x,y,z,qx,qy,qz,qw}` record;
  - a robot without an attachment observation has unknown custody, so C1 does not fire;
  - live belief updates are restored (`belief_rows`/`restore_beliefs`) if the retaken
    capture grounds a different fault;
  - a frozen checkpoint's beliefs are never changed.
- **Live `_install_prepared`** also applies `apply_grounding`, so live prompts carry no
  scenario name.
- **`preparation.json` is now `beliefs.json` (2026-10-16, user request).** All 14 files
  under `monitor/environment_runs` were renamed with contents unchanged
  (`recapture_v2_20261015/grounding_store_20261016/beliefs_rename.json`). Earlier entries
  in this plan still say `preparation.json`. The code constant is `BELIEFS_NAME`, and the
  UI button reads "Download beliefs". The record fields `preparation_sha256` and
  `preparation_version` keep their names.
- **Grounding is stored in `beliefs.json` (2026-10-16).** Preparation grounds the
  request once (`_prepare`, and `evaluation_capture` after `_prepare_saved_capture`), and
  a grounded session starts at `outline`. The code-only grounding turn and the per-attempt
  `failure_grounding.json` are gone. Each attempt derives the grounding again and stops with
  NEEDS_CONTEXT if it differs, for example after a rule or `execution_monitoring` change.
  Re-derive with `python -m cais_spade_llm.recovery_framework.failure_grounding --official --store`.
  - The four official beliefs files were re-derived this way; every
    `observations.json` is unchanged. Old and new preparation sha256 values are in
    `recapture_v2_20261015/grounding_store_20261016/store_record.json`. The runs'
    `condition_record.json` and `report.json` still name the old values.
  - `failure_context_raw.failure_context.failure_mode` still holds the scenario name;
    no prompt reads it.
- **`_prepare_saved_capture`** writes with `write_capture`.
- **Evaluation integrity:** `preparation.json` is hashed in `report.json` and in the
  condition record, frozen with evaluation inputs, and checked at registration.
- **`evaluation_capture --seed`** reproduces a randomly timed failure. It is accepted only
  where the setup draws a seed (Machining `during_processing`).

**Hold probe: blocked.** The probe proposes the held task (ur5e-3 places KET4 on the
board) from the captured failure state to the offline checks:
- CCA alone returns NEEDS_CONTEXT: "Selected safety predicates need occupancy or owner
  interval evidence".
- CCA plus physical evidence stops at "Endpoint requires one supported owner
  declaration", the same validator gap as Attempt 17's event 2.

The probe can run on the new captures once that gap is fixed.

**Runbook (the user runs one stage at a time; Claude checks each capture):**
- Stop the Recovery UI and every other Gazebo/ROS process first. Do not edit code
  during a run.

```bash
tmux new -s recapture        # reattach: tmux attach -t recapture
cd /home/jongh/projects/cais-spade-llm
source /opt/ros/humble/setup.bash
source /home/jongh/ros2_ws/install/setup.bash
BASE=cais_spade_llm/monitor/environment_runs/recapture_v2_$(date +%Y%m%d)
mkdir -p "$BASE"
# set SCEN, NAME, DOMAIN, PORT (and SEED for Machining); the --output folder must not exist yet
poetry run python -m cais_spade_llm.recovery_framework.evaluation_capture --run \
  --output "$BASE/$NAME" --scenario "$SCEN" --condition normal --purpose development \
  --ros-domain $DOMAIN --gazebo-port $PORT --timeout 1800 --settle-timeout 300 --scene-image \
  ${SEED:+--seed $SEED} > "$BASE/$NAME.log" 2>&1; echo "exit $?"
```

| Stage | SCEN | NAME | DOMAIN / PORT | SEED | Expected time |
|---|---|---|---|---|---|
| 1 | `Part slippage` | `part_slippage_01` | 89 / 11379 | — | ~15 min |
| 2 | `ur5e-1 breakdown` | `ur5e1_08` (captured 2026-10-16) | 90 / 11380 | — | ~24 min (`--timeout 2400`; trims M1 90 s / M2 60 s in the private scene) |
| 3 | `Conveyor breakdown` | `conveyor_01` | 91 / 11381 | — | ~12–25 min |
| 4 | `Machining breakdown during part processing` | `machining_02` (captured 2026-10-16) | 94 / 11384 | `4000168541` | ~17 min (`--timeout 3600`; trims M1 60 s / M2 90 s in the private scene) |

These are development captures. `recovery_evaluation` registers an evaluation capture
only while every implementation file still matches its capture-time hash
(`verify_frozen`; "Reserved capture was produced by a different implementation"). A
development capture can never be promoted. The open validator gaps will still change code,
so the reserved evaluation captures are re-run with `--purpose evaluation` after the final
code freeze, using the same procedure.

For a retry, use the next NAME (`_02`) and the next domain/port. Known transient
failures:
- Nav2 crash;
- KMR wall timeout at a low real-time factor;
- a black scene image;
- a missed processing overlap.

After each stage, Claude checks:
- `report.json` status is `captured` and the evaluation row is written;
- the capture is schema 2, and `label_audit` is empty;
- `preparation.json` matches the raw file, and offline preparation is ready;
- the expected fault report and grounding;
- an offline code-only turn 1 with no scenario name;
- the capture loads in Recovery Offline.

A failed attempt is recorded here, and only its disposable outputs are deleted.

**Attempts:**
- **`part_slippage_01` (2026-10-15 15:26–15:31; run `ea6c29d8…`): failed safely** with
  "NEEDS_CONTEXT: the retaken physical capture grounds a different fault". No official
  record changed.
  - Cause: the first live physical capture had no observation of ur5e-4 (transient,
    right after the slip), so grounding found A1+B2 and updated only `resource_state`. The
    retake observed ur5e-4's empty gripper (C1, gear_small `misplaced`). The single-retake
    identity check then stopped the run and restored the beliefs.
  - Evidence: the saved retake was at revision 11, with ur5e-4 `failed` but `held_part`
    gear_small.
  - Fix: `_ground_live_capture` regrounds until two consecutive captures agree, with at
    most 3 retakes. The faulted resource and task must stay fixed, beliefs are restored
    otherwise, and every pass is written to `grounding_passes.json`. A replay test covers
    this case.
  - The 46.9 MB run folder was deleted under `part_slippage_01_deletion_list.json`. Its 44
    links into the repository were removed as links only. The log, status and
    `attempt_history.json` are kept in `recapture_v2_20261015/`.
  - Next: `part_slippage_02` on domain 90 / port 11380.
- **`part_slippage_02` (2026-10-15 15:38–15:45; run `c461242a…`): captured as
  `capture_7dda4f66c81949c3b8e50844c6e13e20`** (development).
  - The report status is `captured`, the evaluation row is written, protected files are
    unchanged, and there were no LLM calls. The native model and scene image were saved.
  - The raw file holds no derived views and no labels. `preparation.json` is bound by
    sha256. Offline-ready, and listed in Recovery Offline.
  - The fault report is ur5e-4 `reported_state: failed` from gripper object detection.
  - Grounding from the raw file alone gives A1+B2+C1: ur5e-4, `environment_9`
    `place_approach`, gear_small `misplaced` with a named-pose `observed_pose`.
  - Grounding settled after one retake. The first pass had ur5e-4 unresolved against the
    not-yet-updated belief; the retake had nothing unresolved.
  - The physical state matches the scenario: ur5e-3 holds KET4 at (-0.32, 0.50, 1.50)
    above the Buffer, ur5e-4's gripper is empty, and gear_small rests at (0, 0.30, 1.025).
  - Offline turn 1 is code-only with no LLM call. The first LLM prompt carries
    `detected_signals` and no scenario name or checkpoint.
- **Reach fix and new drop spot (user decisions, 2026-10-15):**
  - The kinematic premise was checked. A top-down grasp by a UR5e with the RG2 reaches about
    0.91 m at table height: closed form verified against the published UR5e zero pose, and
    offline IK on the captured cell model with URDF joint limits. ur5e-4 **could** reach the
    old drop spot (0, 0.30), so "only ur5e-3 can pick up the gear" was false there.
  - The drop spot moved to (0.25, 0.45, 1.04) in `initialization/failure_scenarios/part_slippage.json`.
    Offline IK: ur5e-3 reaches it and ur5e-4 has no solution. Collisions were not checked.
  - `robot_ur5e.json` gained a `kinematics` block (UR5e DH, RG2 TCP 0.218 m, loose-part TCP
    band -0.05…0.10 m above the base).
  - `workflow_execution._robot_configuration` now computes `gripper_reach` when the cell
    declares none: radius 0.886 m, the smaller reach across the band. An explicit cell value
    still wins. The template's 0.7 m printer-cell value is not used.
  - The prompt says the reach bounds picking a part located only by its `observed_pose`;
    stations follow each robot's reachability list.
  - Limitation: the band assumes loose parts rest at about base height. ur5e-1 and ur5e-2 sit
    on pedestals, so a scenario dropping parts near them may need a per-robot band.
  - Tests:
    - new `test_ur5e_reach_is_computed_from_kinematics_for_loose_part_picks`;
    - updated the reach expectation in `test_cell_capability_calculation_…` and the preset in
      `test_recovery_setup`;
    - updated the Phase-C slip-injection tests in `test_recovery_framework_products` to the
      device-report contract. These had been missed in Phase C.
  - Already failing before Phase C, at c40aba2/HEAD:
    - `test_gazebo_recording` (1), `test_recovery_diagnostics` (1);
    - `test_simulation_timing` (6): a missing monitor waypoint file, and `avoid_collisions`
      versus the scene;
    - `test_recovery_framework_products` (4).
- **`part_slippage_03` (2026-10-15 17:54–18:01; run `b96e9650…`): captured as
  `capture_77666affc1b9468ca157cf366d706dc8`** (development). It supersedes `capture_7dda4f66…`.
  - Status `captured`; all per-stage checks pass.
  - Grounding settled after two retakes: the first pass again missed ur5e-4's observation, and
    passes 2 and 3 agreed.
  - The gear rests at (0.25, 0.45, 1.025).
  - The ur5e-3 and ur5e-4 agent declarations carry the computed reach. ur5e-3 is 0.255 m from
    the gear (inside); ur5e-4 is 0.982 m away (outside).
  - The first LLM prompt contains the reach line, and no scenario name or checkpoint.
- **Conveyor breakdown (Stage 3) setup without a held peer (user decisions, 2026-10-16).**
  - **No held peer.** The old setup made the harness hold ur5e-4 with gear_small until the
    fault, which fit the retired rule "KET4 before gear_small". Under rule 5a (gear_small before
    KET4), no rule explains that wait, so the user dropped it. `example_setup` now sets only
    `part_name` = KET4_Square_4mm, so the Conveyor scenario has no holds. On its own schedule
    gear_small is on the board long before ur5e-1 picks KET4 from M1 (the ur5e1_04 log shows
    the snap while the KMR was still bringing KET4 to M1).
  - **Two-part order, M2 and ur5e-2 off**, unchanged.
  - The checkpoint is unchanged: it latches on ur5e-1's acknowledged `pick_grasp` of KET4 from
    the completed M1, before the release onto the Conveyor. The evidence now always records
    `pickups.ur5e-1`.
  - Capture verification (`evaluation_capture._verify_conveyor_capture`) fails closed unless:
    - ur5e-1's pickup and custody are retained, and KET4 is unassembled;
    - the Conveyor is unavailable, with transport disabled and its own fault report
      (`available: false`, schema 2);
    - no task is in progress at the fault;
    - every other ordered part has a completed assembly and is measured as fixed to the board
      (`physics_evidence.attachments`, board → part, `fixed_joint`).

    Captures configured with a held peer, such as the retained `capture_1b3f3d1d…`, keep the
    old checks.
  - The Conveyor private capture scene gets the same 360 s KMR arm wall budget as ur5e-1.
    Conveyor `capture_02` (October 14) failed on the 120 s limit at real-time factor 0.06. The
    trim times are unchanged.
  - `recovery_setup` checks that the Conveyor `part_name` is eligible for ur5e-1 even without
    a peer.
  - Tests:
    - `test_conveyor_fault`: no peer and no holds, `pickups.ur5e-1`, the fault report, and a
      part that is not eligible;
    - `test_recovery_evaluation_capture`: the new verification with 7 fail-closed cases, and
      the Conveyor 360 s scene;
    - `test_conveyor_fault` + `test_recovery_evaluation_capture`: 211 passed.

    The related suites show only the known failures: 27 in `test_recovery_case_safety`, 3 in
    `test_recovery_evaluation` and 4 in `test_recovery_framework_products`.
  - **Capture timing (user):** do not capture until the ur5e-1 capture succeeds. A tmux watcher
    (`conveyor_capture`; scratch script `conveyor_after_ur5e1.sh`; state in
    `recapture_v2_20261015/conveyor_01.watch`) waits for:
    - a `ur5e1_05`-or-later run with exit 0 and report status `captured`;
    - no gzserver, capture or Recovery UI process.

    It then runs `conveyor_01` on domain 91 / port 11381 with `--timeout 2400`.
    The shared order is in `recapture_v2_20261015/GAZEBO_QUEUE.md`. Every peer chat was told
    on 2026-10-16 at 01:02 to check that file before launching Gazebo.
- **`conveyor_01` (2026-10-16 02:33–02:43; run `5a764b64…`): captured as
  `capture_3aed616530294a4bbdf353000fc69596`** (development, first attempt). It started
  automatically after `ur5e1_08` was captured.
  - Report: status `captured`, evaluation row written, `preserved_files_unchanged`, no LLM calls,
    native model saved. The scene image is rendering-verified and shows the red marker on the belt.
  - The raw file has schema 2, no derived keys and an empty `label_audit`. `preparation.json` is
    bound by sha256. Offline preparation reports nothing missing, and the capture is listed in
    Recovery Offline.
  - `verify_capture` passes the new checks:
    - ur5e-1 holds KET4 from the completed M1, with its gripper attachment measured;
    - Conveyor fault report `{available: false, source: "device controller"}`;
    - `transport_enabled` false;
    - no task in progress at the fault;
    - gear_small has a completed assembly and is fixed to `assembly_board_v1`.
  - The timing was natural, with no holds: ur5e-4 placed gear_small before the KMR reached M1.
    The KMR was idle at the Storage dock, 4.7 mm from its docking pose. ur5e-3 and ur5e-4 were
    idle at home.
  - Grounding from the raw file is B2: Conveyor faulted. The failed task is ur5e-1's
    `place_approach` (`_pending_0`), with four blocked tasks, KET4 `in_gripper` at ur5e-1, and no
    belief updates.
  - Offline turn 1 is `turn_00001_grounding` with `llm_called=false`. The first LLM prompt has
    `detected_signals`, `reported_by`, cell geometry, resource primitives and
    `compute_place_targets`. It contains no scenario name, checkpoint or `failure_type`.
  - Check script (scratch, no LLM): `check_conveyor.py`, adapted from the Machining chat's
    `check_machining.py`. The Recovery UI page itself was not opened, because `machining_01`
    started at 02:45.
  - It supersedes `capture_1b3f3d1d…`, which stays protected until all four re-captures are
    verified (retirement decision).
- **Machining timing (user decision, 2026-10-16).** The user wants this case to show the LLM
  producing new states and events, and left the timing to Claude. The scenario logic is unchanged,
  and only the trims changed.
  - The old 5 s + 5 s trims tied on rule 6 in the nominal schedule and left M2 about 3 s at the
    fault.
  - `evaluation_conditions.condition_setup` now writes Machining's private scene with M1 60 s,
    M2 90 s and the KMR 360 s arm budget, for both normal and changed (changed still removes M2
    `square`).
  - The start stays staged (`holds_task`). KET4 finishes first nominally, and M2 has more than 30 s
    left at the fault.
  - The fault point is random over the overlap, not 50%.
  - `test_recovery_evaluation_capture` covers both conditions; 367 focused tests pass.
  - Why the LLM is essential: see `scenarios/03_machining_breakdown.md`.
- **`machining_01` (2026-10-16 02:45–02:49; domain 92): failed** with "Input preparation failed:
  Resource did not complete negotiated work", before any machining.
  - Gazebo answered slowly from startup: 14 read-only `/get_entity_state` retries, against 0 in
    `conveyor_01` and `ur5e1_08`. Then ur5e-4's `pick_approach.detect_parts` hit a `/detect_all`
    timeout.
  - No other Gazebo overlapped. Another chat's 100 MB capture load ran 02:43–02:45, and swap
    was at 1.3 GB, so this is transient memory pressure.
  - Preserved files were unchanged, and there were no LLM calls.
  - Outputs deleted under `machining_01_deletion_list.json` (58 files, 5.2 MB; 46 links removed as
    links only). The log, status and `attempt_history.json` entry are kept.
  - Both peer chats paused heavy work for the retry.
- **`machining_02` (2026-10-16 02:58–03:15; run `4d409713…`; domain 94 / port 11384): captured as
  `capture_4d1e4cc0fd9b4e07812ce7cbb01d2c87`** (development, seed `4000168541`).
  - Report: status `captured`, evaluation row written, `preserved_files_unchanged`, no LLM calls.
    The scene image shows the red marker on M1 and the KMR at the M2 dock.
  - Schema 2: no derived keys in the raw file, empty `label_audit`, `preparation.json` bound.
    Offline preparation reports nothing missing, and the capture is listed in Recovery Offline.
  - `verify_capture` passes; processing compatibility is `supported`.
  - Frozen at sim 90.41 s:
    - **M1:** KET4 `interrupted` at 4.20 of 60 s (7.0%).
    - **M2:** RGOCG4 `running` at 4.61 of 90 s (5.1%), so about 85 s left.
    - The seeded point was sim 89.88 s, in an overlap of 86.0–145.7 s.
  - The private scene records trims of 60/90 s.
  - gear_small is assembled. ur5e-1 to ur5e-4 are idle at home, and the KMR is idle and empty at
    M2.
  - The fault report is M1 `{available: false, source: "device controller"}`.
  - Grounding is A2+B2: M1 faulted on `environment_21` `machine_part`, with KET4 `loaded` at M1.
  - Offline turn 1 is code grounding (`llm_called=false`). The first prompt (30.6k chars) has
    `detected_signals` and no scenario name, checkpoint or `failure_type`.
  - It supersedes `capture_8beafdb8…`, which stays protected until retirement. The run directory
    is kept as the capture's evidence.

## Failure grounding from raw facts — October 15, 2026 — Phases A, B and C applied; re-capture pending

User decisions:
- Captures keep raw facts only (Gazebo measurements, agent self-reports, configuration
  references). Derived views are computed at preparation time.
- Failures are assumed detectable; the paper states this.
- Turn 1 grounding is deterministic code, not the LLM.
- A device fault report (a resource reporting its own fault or unavailability) counts
  as a detection signal.

The injection code writes labels directly (`failure_effects.py:97-102`,
`conveyor_fault.py:406,413,469-473`). Conveyor and ur5e-1 have no physical fault
signal; their only evidence is the injected unavailability or `failed` state.

New `cais_spade_llm/recovery_framework/failure_grounding.py` grounds a capture from an
allowlisted raw view. It never reads stored labels, `part_tracker`, `setup`, the
scenario name or `prepared_recovery_request`. Generic rules:
- A1: controller goal outcome.
- A2: machine job interrupted.
- B1: controller fault reason.
- B2: unavailable resource, or a resource's own reported fault state.
- C1: expected holder vs measured attachment.
- C2: loose part outside its expected region. Fixture attachments place parts first.

Settings are in `execution_monitoring` in `recovery_framework_gazebo.json`.

Results on the four official captures (`monitor/environment_runs/failure_grounding_report.json`):
- **Generic KMR abilities, Stage A — the LLM can express them (2026-10-15).** These are for the
  ur5e-1 gripper-fault redesign. The KMR gets generic abilities only: no new docks, routes or
  destination events, and no prompt hints. Gazebo decides the outcome, and failures are recorded
  as limitations.
  - Outline events may carry `params` (finite JSON values):
    - schema: `multi_turn_prompts._OUTLINE_SYMBOLIC_EVENT_SCHEMA`;
    - checked in `multi_turn._derive_candidate_outline_task`;
    - kept on the accepted task and shown in accepted-prefix rows.
  - No-validation mode (`outline_diagnostic_bypass`):
    - an event carrying `params` counts as progress
      (`multi_turn_outline_generation._classify_candidate_selection_progress`);
    - the repeated-state fingerprint includes `params`
      (`multi_turn_validated_event._event_state_fingerprint`), so an exact repeat is still excluded;
    - validated mode is unchanged.
  - The no-validation outline prompt also shows:
    - **"Cell geometry"** (`outline_draft.scene_geometry`), all from the captured `inputs.scene`
      except the floor map, which comes from
      `ros2/.../generate_recovery_framework_map.py`. It covers:
      - station, machine and robot-base geometry;
      - each mobile base's measured planar pose, platform size, arm mount and reach;
      - the navigation floor extent and its obstacles.

      Routes, docking poses, handling assignments and transfer targets are excluded.
    - **"Resource primitives"**: name and required parameters, rendered identically for every
      resource.
  - Two candidate rules now allow explicit coordinates in `params` where no location token
    applies.
  - The device fault report's own `source` reaches the prompt as `reported_by`
    (`failure_grounding`). No component is derived from it.
  - Tests:
    - new: `test_multi_turn_diagnostic` (params kept, invalid params rejected, repeats excluded,
      new poses accepted), `test_outline_draft` (geometry without hints) and
      `test_failure_grounding` (`reported_by`);
    - four tests left behind by the plant-wide spec change were updated (removed
      `_PART_SLIPPAGE_SAFETY_FILES`, new defaults).
  - Offline check with a mocked LLM on `capture_0a7fde80…`: the prompt shows geometry and the KMR
    `move_base(target_pose)`, and a KMR `move_to_location` event with `params` is accepted into the
    prefix.
  - Already failing without Stage A (checked by temporarily reverting it): three tests in
    `test_case3_recovery_dryrun`.
  - Next: Stage B (driving to free poses in Gazebo), then C (release anywhere) and D (conveyor).
- **Generic KMR abilities, Stages B–D — drive, release and the belt (2026-10-15).** All three are
  checked live in Gazebo. Captures and LLM runs with them are still to do.
  - **Stage B, drive anywhere.**
    - `move_to_location` is bound to `_kmr_executor`.
    - The worker's task mode accepts the runtime's full parameters
      (`resource_id`, `resource_state`, x, y, yaw, `observed_resource_location`).
    - A target with no declared dock skips the configured-route sweep and checks only its
      endpoints; the base controller checks the path it plans. Nominal tasks still refuse
      non-dock targets.
    - Completion requires the measured base within 5 mm and 0.035 rad of the target
      (`_kmr_arrived`).
    - Base travel is bounded in simulation time (240 s) with a 3600 s wall backstop, and the
      KMR worker's outer timeout is 3900 s. A first live run showed the old 240 s wall limit
      cut a drive short at real-time factor 0.067.
    - The generated-recovery executor (`slippage_execution`) has a KMR branch. Custody comes from
      the KMR's measured state; picked parts update the source owner's declared `pick_grasp` or
      `pick_part` consequence, with `{part_name}` bound.
  - **Stage C, release anywhere.**
    - `compute_place_targets(transform, part_pose)` places the held part at an explicit world
      pose with no dock (`kmr_recovery.release_targets`).
    - `release_part` and `detach_part` report where the part came to rest and whether it
      settled.
    - The receiver comes from the settled position (`environment_recovery.observed_receiver`):
      buffer zone, machine workholding, staging tray, the part's Storage slot or a belt location.
      The receiving owner's declared consequence is then committed.
    - Anything else, or a part still moving when observation ends, becomes a misplaced part at
      its observed pose.
  - **Stage D, the conveyor.** A part resting on a stopped belt is recorded at its observed
    location (`nominal_conveyor.conveyor_location_at` and `conveyor_place_parameters`), and the
    existing transport carries it.
  - **Live evidence** (scratch harness; poses chosen by hand the way the LLM would, from the
    cell geometry):
    1. The KMR picked KET4 at Storage.
    2. It drove while carrying to (−3.0, −0.36, −π/2), arriving within 3 mm and 0.025 rad.
    3. It released at part pose (−3.0, 0.40, 1.016). KET4 settled upright 0.3 mm away and was
       recorded on the Conveyor at `after loading_position_1`.
    4. `advance_conveyor` carried it 2.22 m to the output nest, with arrival observed.

    The first base pose tried, facing +x, put the release beyond arm reach (IK failure). Choosing
    a reachable dock is part of what the LLM must reason about.
  - **Review:** an adversarial review found ten confirmed defects, all fixed:
    - the task-mode parameter check;
    - the Storage pickup lookup and template binding;
    - missing machine, staging and Storage receivers;
    - the stale release record;
    - pickup custody checks;
    - re-picking a misplaced part;
    - the worker outer timeout;
    - `params` accepted in validated mode (now rejected, and the guidance moved to the
      no-validation suffix);
    - progress from undeclared arguments, and revisiting earlier states;
    - `reported_by` taken from converted schema-1 paths.
  - **Tests:** 3589 pass. All 175 failures, across 109 test names, also fail without these changes
    (checked against reverted baselines).
  - **Not done yet:**
    - capture wiring for the ur5e-1 gripper fault;
    - a schema-2 capture whose KMR declares the new `compute_place_targets` parameter;
    - the deferred refactor moving the KMR modules to `resources/robot/`.
- **ur5e-1 gripper-fault capture wiring (2026-10-15/16).**
  - Setup: M2 and ur5e-2 stay on, with the three-part order.
  - Trim times in the private capture scene: M1 90 s and M2 60 s.
    - M1's longer trim lets the KMR finish delivering RGOCG4 before ur5e-1 reaches into M1.
    - M2's longer trim keeps KET4's trim finishing first (rule 6) and leaves M2 still trimming at
      the fault.
  - The fault is reported by "gripper object detection" at the existing pre-grasp checkpoint.
  - The capture freezes first (`capture_machine_checkpoint`), so M2 is recorded running.
    `capture_machine_checkpoint` accepts the failed task's own machine with its completed job.
  - The harness verifies, failing closed:
    - M1 finished and M2 still processing;
    - the reporting device;
    - the KMR idle at a dock.
  - **Attempt `ur5e1_02` failed** (outputs deleted).
    - The agents' per-task wall limit (`tool_timeout_s` 600 s) killed M1's `machine_part` inbox
      after 600 wall-s. The 90 sim-s trim needs about 900 wall-s at real-time factor 0.1.
    - RGOCG4 then overtook KET4: M2 finished at sim 145.3, and ur5e-3's nominal RGOCG4
      `place_approach` failed in collision. That stopped the run before the checkpoint.
    - **Fix:** Gazebo worker and agent wall limits are now `_WORKER_WALL_LIMIT_SEC` = 3900 s
      (`workflow_execution.py`). Simulation-time bounds stay authoritative.
    - **Separate observation:** nominally, with the CCA bypassed, nothing orders RGOCG4 after KET4
      on the board, and ur5e-3's RGOCG4 insert collided. Not investigated yet.
  - **`ur5e1_03` and `ur5e1_04` failed on infrastructure** (outputs deleted).
    - `ur5e1_03`: the controller manager was not up within 60 s at startup.
    - `ur5e1_04`: the nominal KMR carrying move Storage→M2 was aborted with "KMR arm left its
      parked configuration". The carrying check needs custody updates within 0.75 s of wall time,
      which can lapse at real-time factor about 0.1.
  - **`ur5e1_05` reached the intended frozen state** (outputs deleted). At sim 129.8, world paused:
    - M1 had KET4 completed;
    - M2 had RGOCG4 at 67%;
    - the KMR was idle at Storage;
    - ur5e-1 was empty in M1, with the fault from gripper object detection.

    Live grounding then refused, because the report's belief update cannot be retaken in a frozen
    checkpoint.
    - **Fix:** a frozen checkpoint keeps its acknowledged beliefs (raw), and offline grounding applies
      the grounded fault before the first LLM turn (`part_slippage_recovery._begin`).
  - **`ur5e1_06` failed at startup** (outputs deleted): `agent_creator` raised ModuleNotFoundError
    because a ruff hook had removed the side-effect import `cais_spade_llm.ui_main` from
    `evaluation_capture.run_capture`. **Fix:** the import is restored with `# noqa: F401` and a
    comment.
  - **`ur5e1_07` captured the intended state but failed its scene-image step** (run directory
    deleted). The after-image check still required a paused world, but a machine checkpoint
    unpauses after its frozen record. **Fix:** both image checks now apply only to a robot freeze.
    Its capture `capture_e3e51a42…` was superseded and deleted on 2026-10-16 at 03:21, after the
    queue went idle: 19 files, 158 MB, no symlinks. The list is in
    `recapture_v2_20261015/ur5e1_07_deletion_list.json`. A running harness hashes every
    `capture_*/observations.json` and fails if one changes, so the deletion had to wait. After it:
    - 14 captures are listed, all offline-ready;
    - the four defaults are unchanged and load;
    - the Part slippage accepted result is still available.
  - **`ur5e1_08` captured `capture_d1ca51654ff7423193d1bd13b146964d`** (exit 0, preserved files
    unchanged; run directory kept). Frozen at sim 131.78:
    - M1: KET4 completed (90 s), no active job;
    - M2: RGOCG4 running at 0.776 of its 60 s trim;
    - the KMR idle and empty at Storage;
    - ur5e-1 empty at its pick pose in M1, with the fault from gripper object detection.

    Offline checks with a mocked LLM:
    - grounding: ur5e-1 faulted, failed task `pick_grasp` for KET4, `reported_by` in the signal;
    - the prompt shows cell geometry, primitives and the seven safety files;
    - the KMR declares `compute_place_targets(transform, part_pose?)`;
    - M2's running job is not in the inline Resources rows (M2 shows only `loaded`). The LLM gets
      it on request: `/resources/<M2>/snapshot` returns `job_state: running`, and
      `static_capabilities` returns the job record. This is the same design Machining uses.
  - **Next:**
    - done 2026-10-16, at the user's request: `_DEFAULT_OBSERVATIONS` now points all four
      scenarios at their schema-2 re-captures (77666aff, 4d1e4cc0, 3aed6165, d1ca5165). The
      schema-1 originals stay selectable as regression fixtures. `test_failure_grounding` covers
      all eight captures, and a new test checks that each default is a schema-2 capture of its own
      scenario;
    - retire `capture_0a7fde80…` once the user confirms;
    - the user runs the LLM.

- **Plant-wide safety spec set (user decisions, 2026-10-15):** all four scenarios now load the same
  seven files in `recovery_case_safety._PLANT_SAFETY_FILES` (`part_slippage_example` takes its
  initial selection from there):
  - rules 1–3, unchanged: board, M1 and M2 mutex.
  - rule 4, new, generalized at the user's request: `…_trim_before_entry.txt`. A single rule
    (`SAFE_trim_before_assembly_board-v1_entry`) has one scope per part whose process plan has a
    trim step: KET4/8/12/16 → square, RGOCG4/8/12/16 → circle. Each part may enter the board only
    after its own trim result is recorded.
    - Gears are excluded: their `print_part` record has no result or target, and printing is never
      at risk.
    - The offline checker runs each scope as its own instance (`rule_id = id:<digest>`).
    - The user kept this rule because the nominal plan does not constrain a recovery. Without it,
      skipping both trims would satisfy rule 6.
    - All four captures track all 8 pegs with complete trim ledgers. Offline outline drafts build
      from all four captures with the set.
  - rule 5, assembly order gear_small → KET4 → RGOCG4: the existing gear_small-before-KET4 rule
    plus new `…_KET4_Square_4mm_before_RGOCG4-50_Round_4mm.txt`. The KET4 target is
    `GMC_Laser_Plate_Virtual/KET4_Square_4mm`.
  - rule 6, unchanged: KET4 trim before RGOCG4 trim. Machining breakdown is the case that tests it.
  - `…_KET4_Square_4mm_before_gear_small.txt` is no longer selected because it contradicts rule 5.
    The file and its tests are kept, because the retained Conveyor capture's records name it.
  - `capture_77666aff…` records KET4's square trim as complete and RGOCG4's ledger as complete
    and empty, so rules 4–6 can be grounded from that capture.
  - Tests:
    - `test_recovery_case_safety`: lookups by file name; defaults equal the plant-wide set.
    - `test_predefined_safety`: the two new files run the compile, automaton and monitor tests.
  - `test_recovery_case_safety` also checks per-peg semantics: another peg's trim never
    satisfies a peg's instance, and a wrong trim result is rejected.
    - `test_machine_jobs`: the default selection.
    - The 27 failures already in `test_recovery_case_safety` are unchanged; they reference the
      retired `safety_Conveyor_breakdown.txt` and `safety_Machining_breakdown…txt`.

- **Part slippage:** A1+B2+C1. ur5e-4, environment_9 `place_approach`, gear_small
  `misplaced`. Agrees with stored labels except resource_state (stored `failed`,
  grounded DES value `picked`, fault listed separately). The transition log conflict
  (its last entry already shows ur5e-4 `failed`) is reported.
- **Machining:** A2+B2. M1, environment_21 `machine_part`. Fully agrees.
- **Conveyor:** B2. Fills the empty interrupted task with `d49d…_pending_2`
  `place_approach`.
- **ur5e-1:** B2. Fills it with `01ee…_pending_0` `pick_grasp`; resource_state stored
  `failed`, grounded `at_pick`.

One false positive was found and fixed during development: the `assembly_board-v1`
region is the shared work area above the board (z 1.05–1.4), not where assembled
parts sit. Measured fixture attachments now take precedence over regions.

Tests: `test/test_failure_grounding.py` (14 passed), plus 1,241 focused tests passed.
`test_environment_capabilities.py` has 113 failures that already exist with the committed
config ("Current configuration differs from the declared selected_program",
`machine_jobs.py:238`). They were not caused by this change and are not investigated
here. `poetry check` reported only existing deprecations; compileall and
`git diff --check` passed.

**Phase B (applied October 15, user-approved):**
- Turn 1 is code-only for any capture that records the agents' state at the
  failure. `multi_turn._code_grounding_turn` runs the existing grounding handler with
  a `grounded` decision. There is no provider call. The turn record stores
  `llm_called=false` and `code_grounding`. The Recovery UI turn view shows it as
  "Code grounding — no LLM call".
- Both offline preparation paths apply grounding:
  - Draft: `outline_draft.OutlineDraft`, which also adds `fault_grounding` to the draft
    record.
  - Full-check: `part_slippage_recovery._install_prepared`, which writes
    `failure_grounding.json`.
  - `failure_grounding.apply_grounding` replaces `llm_input.fault_event` with the grounded
    event, including `detected_signals`. It updates `failure_anchor`, affected part facts
    and the faulted resource row, and drops `failure_type` and `checkpoint`.
  - Captures without raw failure facts keep their stored summaries. A capture with them
    but no detectable fault stops with `NEEDS_CONTEXT`.
- `_slim_fault_event` no longer forwards `failure_type` (the scenario name).
- `environment_models.owners` accepts configured fault states
  (`execution_monitoring.fault_report_values`) for `resource_state`. Recorded captures'
  models omit them. The recorded capture files are unchanged.

Verification (offline, no API calls):
- In both paths, all four captures prepare with turn 1 code-grounded.
- The first LLM request is the outline turn. It carries the grounded fault and no
  scenario name, failure type or checkpoint label.
- Full-check preparation reaches `outline_ready`.
- Tests: 19 grounding tests and 1,219 focused tests passed.

Replaying Attempt 17 under all checks:
- Event 1 (ur5e-4 `move_home`) now passes RA, CCA and transition validation. It is
  still rejected by the comparator as `excluded_no_progress`: clearing a shared area
  neither reduces a goal condition nor enables a new admissible event.
- Event 2 (ur5e-3 places KET4 at the Buffer) has no supporting owner declaration
  ("Endpoint requires one supported owner declaration").
- RA+physical now stops at "Owner static_capabilities were not recorded in this
  attempt's physical model".

These are validator and selection issues, not grounding issues.

**Phase C (applied October 15, user-approved): capture schema 2**
- New `recovery_framework/capture_files.py`. A schema-2 capture is two files:
  - `observations.json` holds raw facts only: metadata, the run's resolved `inputs` plus
    `config_references` (path + sha256 of each source file), agent and device reports
    (`failure`, transitions, acknowledgements, `fault_reports`, `unavailable_resources`,
    job reports), measurements (`physical_evidence` without its derived views,
    `planning_scene_capture`, `robot_poses`), and copies of `agent_reports` and
    `agent_declarations` (the robot DES models, catalogs and static capabilities).
  - `preparation.json` holds the derived views: `prepared_recovery_request`,
    `resource_evidence`, `pending_nodes`, the `part_tracker` and `resource_states`
    beliefs, the physical `snapshot`/`environment_models`/`environment_valuation`/
    `products`, and `failure.*_after`. It is bound to the raw file by sha256.
  - `read_capture` merges the two. Views that do not match the raw file are ignored, so
    preparation reports them missing. Schema-1 files are read and written unchanged.
- The captured `inputs` were kept: the scene file has changed in 49 places since the
  October 10 capture, so a reference alone would silently change old captures.
- Failure injection writes device fault reports to `context.fault_reports` instead of
  labels:
  - Part slippage: the robot reports its lost part (`_current_state="failed"`,
    `_held_part=None`). The product's valuation and part tracker stay unchanged.
  - ur5e-1 breakdown: the robot reports `failed`, and the valuation stays at `at_pick`.
  - Conveyor and M1 report unavailable, alongside `unavailable_resources`.
- Detection design (user question, answered):
  - the failing device detects (gripper object detection, robot or machine controller,
    conveyor PLC);
  - the Gazebo pose stands in for an idealized camera that localizes parts;
  - the product agent's execution monitor (grounding code) derives `misplaced` and
    `failed`.
- Live preparation (`PartSlippageRecovery._begin`) grounds the raw capture.
  `failure_grounding.update_beliefs` then applies `belief_updates` to the live context
  and the belief copies, and records the new `context_revision`. The physical and
  resource evidence is retaken once, so the belief-dependent views agree. A frozen
  checkpoint whose beliefs would change stops with NEEDS_CONTEXT.
- Grounding (schema 2) reads `fault_reports` and builds pending work from the planner's
  report (`capture_files.pending_nodes`, now also used by `_pending_nodes`).
  `verify_capture` checks the ur5e-1 fault report instead of the valuation.
- Conversion check, scratch copies only
  (`python -m cais_spade_llm.recovery_framework.capture_files --official`, report in
  `monitor/environment_runs/capture_v2_preview/conversion_report.json`):
  - All four splits are lossless (same observation fingerprint).
  - Grounding from the raw file alone equals grounding of the original; only the
    evidence path names the new location.
  - Raw sizes: 25–39 MB. Originals: 43–102 MB.
  - The only label left in a raw file is Part slippage `transitions/7/after/ur5e-4`
    `failed`, the schema-1 alias fixed in c40aba2. Re-capture removes it.
  - The official captures were not modified.
- Not changed:
  - The Machining worker still records `interruption=<scenario name>`; grounding never
    reads it.
  - Conveyor injection still sets `inputs.scene.Conveyor.transport_enabled=False`.
  - `PartSlippageExample._use_new_observations` writes into a directory it never
    creates. This was already true before this change.

Tests (offline, no API calls):
- New `test/test_capture_files.py`: 7 passed.
- `test_failure_grounding.py`: 21 passed.
- Injection tests updated to the device-report contract (`test_conveyor_fault.py`):
  93 passed.
- New live-grounding test in `test_part_slippage_recovery.py`: 169 passed.
- Also passed:
  - `test_offline_recovery` (280)
  - `test_recovery_evaluation_capture` (103)
  - `test_machine_jobs` (126)
  - `test_validated_recovery_outline` (264)
  - `test_recovery_outline_evidence` (482)
  - `test_recovery_context_retrieval` (313)
  - `test_multi_turn_diagnostic` (12)
  - `test_outline_draft` (14)
- These failures also occur on unmodified HEAD (checked in a clean worktree):
  - 3 in `test_recovery_evaluation`;
  - 12 in `test_recovery_feasibility`;
  - 1 in `test_environment_recovery`.
- `poetry check` reported only existing deprecations. compileall, `ui_main --help` and
  `git diff --check` passed.
- The preview copies (357 MB) in `monitor/environment_runs/capture_v2_preview/` are
  disposable; the converter regenerates them.

Next:
- Re-capture the four failures in Gazebo with the CCA bypassed. See "Schema-2 re-capture"
  above: live CCA capture is not supported. Live capture of schema 2 has not been run in
  Gazebo.
- Rebuilding `prepared_recovery_request` offline from `agent_declarations` (instead of
  reading `preparation.json`) is possible later; it was not needed for this phase.

Plan: `~/.claude/plans/what-do-you-mean-synchronous-kahn.md`.

## Current diagnostic — October 14, 2026 — three unchanged Machining breakdown attempts

The user requested exactly three sequential fresh `Machining breakdown during part processing`
outline attempts with the current pipeline unchanged, using
`capture_8beafdb8a01e444894cfaf41d29921fd`, `gpt-5.4`, and `medium`.
All three outline checks are disabled; formal nominal continuation, primitive
composition, and execution are not run. Explicit disclosure approval covers
this saved context and its retrieved facts sent to `https://api.openai.com/v1/`.
The baseline records hashes for 277 pipeline/configuration files and 632 protected
capture/result files. Evidence is under
`cais_spade_llm/monitor/environment_runs/machine_outline_three_attempts_20261014_01a13d5b/`.

Attempt 1, `draft_499cb85bf6824b78944b759a83d7ddcd`, finished with 21 provider
calls, 18 selected events, 1,637.09 generation seconds, and 239,310 tokens.
Its status is `draft`, with a proposed handoff; manual inspection finds it
incorrect. `RECOVERY_SEQ4` removes `RGOCG4-50_Round_4mm` without pausing the
running M2 job. `RECOVERY_SEQ9` claims `KET4_Square_4mm` completion from a
start state still naming the running round job and circle program. The sequence
includes repeated handling and nominal processing. Its final handoff assumes
M2 `resource_state` and `part_name`. There were no provider errors or context
retrievals, and all four checks remain `not_checked`; source and pipeline hashes
are unchanged. The sole intercepted subprocess was the SDK's `uname -p`
metadata probe, reproduced locally without a provider call; no validator or
equipment execution was invoked. Exact findings are in `attempt_01/analysis.json`.

Attempt 1 evidence is archived and verified. Its 149 disposable files were
removed, releasing 7,729,152 net allocated bytes after retaining the archive.
All 632 protected files remain unchanged; the four official captures remain
selectable, and the worked Part slippage result still loads.

Attempt 2, `draft_20b60415397741709afc0d5aa1890512`, finished with 12 provider
calls, 11 selected events, 1,116.95 generation seconds, and 138,810 tokens.
It is also incorrect. `RECOVERY_SEQ6` places `KET4_Square_4mm` into occupied M2
and leaves `ur5e-2` holding it. `RECOVERY_SEQ9` claims machining while the part
is in that gripper and M2 still names the running `RGOCG4-50_Round_4mm` job and
circle program. `RECOVERY_SEQ10` performs nominal processing inside recovery.
The final handoff assumes M2 `resource_state` and `part_name`. The first outline
response did propose M2 `pause_job`, but the comparator selected M1 pickup by
`smallest_exact_effect_candidate_id`; no pause or stop was ever selected.
No provider errors or retrievals occurred. All checks remain `not_checked`,
`execution_authorized=false`, and source/pipeline hashes are unchanged. All
11 events are assessed in `attempt_02/analysis.json`.

Attempt 2 evidence is archived and verified. Its 86 disposable files were
removed, releasing 6,004,736 net allocated bytes after retaining the archive.
The 277 pipeline/configuration files and 632 protected files are unchanged.
All four captures, the worked Part slippage result, and the earlier unvalidated
machine draft still load.

Attempt 3, `draft_76486dd8bd604aa8afc96e58011f49af`, finished with 12 provider
calls, 9 selected events, 958.82 generation seconds, and 134,878 tokens.
It is incorrect. `RECOVERY_SEQ6` repeats the occupied-M2 placement and omitted
`held_part` release. `RECOVERY_SEQ7` claims `KET4_Square_4mm` completion from
M2's running `RGOCG4-50_Round_4mm`/circle start state; `RECOVERY_SEQ8` claims
nominal processing inside recovery. `RECOVERY_SEQ9` leaves contradictory
`ur5e-2` custody and `M2 staging tray` location and reaches a handoff by assuming
M2 `resource_state` and `part_name`. Turn 3 was rejected for duplicate candidate
`outline_id` values; turn 11's KMR nominal move was `excluded_no_progress`.
These are generation/selection problems, with no provider errors or retrievals.
All checks remain `not_checked` and `execution_authorized=false`. All nine
events are assessed in `attempt_03/analysis.json`.

All three attempts are finished; none is a logically correct complete recovery
outline. Their initial grounding prompt SHA-256 is identical. Total usage is
45 provider calls and 512,998 tokens: 337,169 prompt tokens and 175,829 completion
tokens, including 159,804 reasoning tokens. The comparison retains both the
application generation timer and wall time per attempt (1,732.18, 1,179.27,
and 1,015.17 seconds). All three attempts stopped at a proposed handoff.

The comparison is saved in `comparison.txt` and `comparison.json` under the
batch directory above. They contain every exact selected event, its manual
assessment, provider/time/token totals, findings, and recommended fixes.
`pipeline_inspection.json` traces the omitted current M2 job fields, candidate
comparison decisions, stale `current_valuation` fallback, and handoff assumptions
to the unchanged source. No factual retrieval occurred in any attempt.
The public outline context omits M2's current `job_state` and `selected_program`;
candidate comparison does not establish operational superiority; tentative
effects leave omitted custody fields unchanged; unknown owner fields can be
refilled from the original capture and can satisfy the proposed handoff by
assumption. These are the recommended areas to repair before another experiment.
No generation fixes have been applied.

All three exact evidence archives are verified. After explicit deletion lists,
ownership checks, and unchanged-file checks, 321 disposable files were removed,
releasing 18,948,096 net allocated bytes after retaining the archives (about
18.1 MiB). Per-attempt `cleanup_manifest.json`, `host_ownership_check.json`, and
`cleanup_receipt.json` record the files, archive hashes, ownership scope, and
released space. Each archived draft reloads through `load_draft`; all 38 selected
events have a saved manual assessment. `final_evidence_verification.json` records
the post-cleanup archive/readback checks and comparison hashes.
All 277 pipeline/configuration files and 632 protected files
match the baseline; the source SHA-256 remains
`c05a39c65313aef89e0eabb09720530996134ecf9adc18ecc828ab0078d29447`.

The four official captures remain selectable. The accepted Part slippage
outline `offline_470087fe7a92421abd624ea721aa2ca1` and composition
`offline_1bb1efac84d7443aa37238521ead3b68` still load from the verified worked
archive with five events and 21 primitives. The earlier unvalidated machine
draft `draft_66b5fc99daab491ea8f25624d1790b5a` also still loads. No new accepted
recovery outline or primitive composition is established. All newly retained
outlines are diagnostic evidence with `execution_authorized=false` and all
four checks `not_checked`; no RA, physical, CCA, formal nominal-continuation,
primitive-composition, Gazebo, or recovery-execution stage was run. The disabled
check dispatch functions only returned `not_checked`; the generic exported
selection summary's `transition_validation=passed` is not a validation result.

Planning verification remains 19 focused tests passed. The interrupted UI
background-task test remains unresolved; it was not rerun during this experiment.
Exactly three fresh attempts were run. Existing application changes and unrelated
working changes were preserved. Further fixes are recommendations only.

## Current setup — October 14, 2026 — simplified Recovery and saved-model preparation

The user requested one Offline outline Generate action, one verified capture per
selected failure, and lightweight checkbox preferences. Generate must always
start fresh, with no previous outline or primitive prefix. The existing
`_DEFAULT_OBSERVATIONS` mapping and the worked Part slippage archive are retained.
The Offline outline card now has one Generate action. Generate outline only,
Load draft, Continue, outline Regenerate, and Start fresh are removed from the
Offline controls. Primitive composition's Generate includes the existing
Continue preparation and preserves its full-validation admission gate.

`SystemBridge.set_failure_example_options` adds `defer_preparation=False`;
the new UI passes True and stores preferences without loading observations,
constructing owners, or writing attempts. `part_slippage_action` forwards the
new `generate_outline_fresh` action. Loading, detached preparation and provider
work run outside the UI event loop. Source summaries are cached; polling does
not rebuild ResourceAgent inputs or scan historical captures. Selections lock
while generating, then edits apply to the next fresh attempt. Previous records
keep their original settings. Source and checkbox revision checks discard stale
poll results. The verified capture can be downloaded before generation.

Blocked attempt `offline_21e13916dd6c4798995a1c341fa0ab99` failed before any
outline event: new preparation required M1 (`recovery-resource-5@localhost`) to
provide `recovery_des_model` and `descriptor_fingerprint`, although its original
Part slippage capture records `{}`. Preparation now preserves unavailable
descriptors and captured evidence for unrelated resources. A dependent enabled
check still requires a supported model and returns `NEEDS_CONTEXT`. No empty
model receives a fabricated fingerprint. Its finished failed stage is included
in the explicit cleanup manifest after replacement preparation verification.

The first actual full-check preparation probe,
`offline_57e18fe8769a4de9a80fbfc4960157b7` under `/tmp/recovery/`, passed the
descriptor step and then correctly stopped at `NEEDS_CONTEXT: conflicting
recorded observed_pose for gear_small`. Its tracker pose and separate physical
measurement differ slightly. This is a distinct physical evidence gap; the
source is not rewritten and the existing strict conflict check is retained.
Next verification covers the default unchecked preparation for all four
captures and records the independent full-check evidence status. The probe
made no LLM or execution request. Its seven disposable files (45,709,084 bytes)
are listed with exact hashes and ownership in `simplification_probe_finding.json`
beside the worked archive; source observations and accepted results are protected.

`ui_simplification_verification.json` records the actual NiceGUI components and
SystemBridge for all four failures: one verified source each, no eager
preparation or attempt creation, the worked-card download, and twelve checkbox
changes per case taking 7.0–8.3 milliseconds total in the component checks.
These are server-side component timings, not remote browser latency measurements.
The historical selector's first full scan took 20.618 seconds; it is no longer
called by the new Recovery view. All three checkboxes start unchecked.

All four real captures prepared as `draft_ready` under the default unchecked
selections. Full preparation reached `outline_ready` for Machining breakdown
during part processing, Conveyor breakdown, and ur5e-1 breakdown. Part slippage
passed the descriptor step and retained the separate conflicting-pose evidence
gap. The new UI's full preparation now reports this as `NEEDS_CONTEXT` without
an error traceback or advancing an outline. The checked source is not claimed
to have sufficient physical evidence merely because its historical worked
outline and composition are retained. Generation and execution were guarded:
`llm_calls=[]`, no outline/primitive output, and execution authorization false.

871 focused regressions passed, covering all eight check combinations, fresh
prefixes, immutable attempt choices, background preparation, strict missing
descriptors, controlled NEEDS_CONTEXT, rapid toggles/stale polling, one capture,
and primitive composition gates. The final UI/context-label recheck passed
113 tests. Poetry, compilation, UI CLI help, and `git diff --check` passed;
Poetry emitted existing metadata deprecations. The UI uses `reload=False`, so
restart it to load these changes. This task starts no UI server, LLM attempt,
ROS/Gazebo recovery, or equipment command.

`simplification_cleanup_manifest.json` lists 42 disposable files: the original
blocked descriptor attempt, private capture-preparation verification outputs,
and the temporary verification script (611,482,710 logical bytes;
611,614,720 allocated bytes). Official inputs, accepted results, the worked
archive, prior retained preparation evidence, regression coverage, and other
chats' work are excluded. Ownership, hashes, recent changes, tracked state, and
open handles were rechecked before deletion. All 42 files were removed; the
cleanup receipt records the retained-result loading check. The earlier seven-file
probe cleanup released another 45,709,084 logical bytes. Both accepted original
results still load, all 287 accepted JSON hashes match the worked archive, and
all four official source hashes remain unchanged after cleanup.

## Part slippage all-unchecked diagnostic — October 14, 2026 — three fresh attempts, no handoff

The user requested three fresh Part slippage outlines with all validation
bypassed. Each ran `OutlineDraft` (all three checks false) on
`capture_f923020bf9ce496eb443826291f8f2c2`, the same path as the UI Generate
action, with `gpt-5.4` / `medium`, in parallel, without a prefix.

| Attempt | Identifier | LLM calls | Selected events | Outcome |
| --- | --- | ---: | ---: | --- |
| 1 | `draft_618ab0ba826d465988c93ae6c2b8eed5` | 16 | 9 | `draft_incomplete`, no progressing candidate |
| 2 | `draft_4039044deb8b4b9c88104e9b79d7242a` | 12 | 6 | `draft_incomplete`, no progressing candidate |
| 3 | `draft_3483bcb3d6b14435bb604a0dd75255ef` | 8 | 4 | `draft_incomplete`, no progressing candidate |

Failure reason: `multi_turn_diagnostic.handoff` requires each `recovery_replaced`
task to be claimed by an event whose end state equals that task's owner updates.
`environment_9` (`place_approach`) needs `positioned`/`in_transit`; the LLM
always wrote `in_gripper`, and once gear_small is assembled the intermediate
state cannot recur. Replaying the claim rule on accepted Attempt 17 claims only
`_pending_2`, so even that outline cannot reach the diagnostic handoff. Attempt 2
reached the intended state after three events (gear_small assembled by ur5e-4,
ur5e-4 retreated, ur5e-3 still holding KET4 at `picked`), but the loop continued.
Secondary: the diagnostic clears `goal_conditions` (`Recovery Goals (none)`);
feedback is only `no_progressing_candidate`; the diagnostic progress override
treats KET4's `ur5e-3` versus JID `part_location` as progress, so ur5e-3
nominal insertion of KET4 was selected and broke the nominal entry in all three;
unchecked guards admitted a failed-robot pick and an unpicked `place_insert`.
Both safety rules were in the prompt text; precedence held, mutex was violated in
attempts 1 and 3 (CCA unchecked).

Next step: make the diagnostic handoff use the recorded `recovery_replaced`
goal conditions (or only the suffix's final task), name open obligations in
feedback, and exclude nominal-suffix work before further API retries. Compact
evidence is under `part_slippage_multiturn_diagnostic_20261014/`; 249 files
(6,588,643 bytes) were removed per its manifest and receipt. Capture and both
accepted Part slippage results keep their hashes.

Follow-up, same day: a generic bypass-mode repair was made in an isolated copy,
not yet in this checkout, because the concurrent machine batch
`machine_outline_three_attempts_20261014_01a13d5b` fails closed on any package
change. The patch is `part_slippage_multiturn_diagnostic_20261014/fixed_pipeline/pipeline_fix.patch`.
Handoff completes when the capture's `marked_reentry_conditions` hold (legacy
task claims only when none are recorded), selected shared-area rules hold for
the resume state, and nominal entries are enabled; obligations are the recorded
`unmet_reentry_conditions`. Custody rejects handling a part held elsewhere or a
second part. The prompt drops the tuned event-count sentence and adds recorded
handoff conditions with `holds`, region bounds, declared `gripper_reach`, and the
interrupted event. Instruction text is identical across all four captures and no
accepted-outline names occur. Offline replay: Attempt 17 completes at event 5;
without ur5e-4 leaving it is held by `SAFE_shared_area_mutex`. Tests: 915 passed
in the copy; 2 copy-only failures (missing `ros2` meshes and `writing/`) pass here.

Three fresh API attempts on the repaired copy (`draft_cdd97f0bfafe4f0f9a5dff423d0cf115`,
`draft_bb69059cedd54941b1717f2d474eddc4`, `draft_2f70891f4e9e4fff8fad626e21814a5c`)
each reached a proposed handoff in 4–5 events, but none is correct: ur5e-4,
outside its declared reach of gear_small, places it in all three. The LLM stated
that only ur5e-3 reaches the gear, yet listed ur5e-4 alternatives; with RA
bypassed the comparator preferred them because they clear obligations sooner,
and once restored the nominal entry over picking the gear.

The user chose option A: with all three checks off, the LLM returns exactly one
chosen event (schema `maxItems=1`); any enabled check keeps the unchanged
five-candidate comparison. Patch: `option_a_llm_selects/pipeline_fix_option_a.patch`;
917 tests passed in the copy, same 2 copy-only failures. Three API attempts:
`draft_6b9893f3c41c45a1a04f65a4c06c8777` and `draft_d901c905e6654d8cbf7e15c663ecb0e2`
chose ur5e-4 (outside declared reach) to re-pick gear_small, then stopped
`draft_incomplete` after `place_approach` was rejected three times as
`no_progressing_candidate`. `draft_1fbc3e148f744dd98e49c0c9cc45b342` used ur5e-3
(KET4 to Buffer, pick gear, insert, re-pick KET4) but cleared ur5e-4 only after
the handoff block, so the insertion violates `SAFE_shared_area_mutex`. None is
correct. Findings: LLM-only selection is unreliable on reach and ordering; the
progress rule also rejects legitimate approach steps. The patch is not applied
here while the machine batch runs; apply it once that batch finishes.

Full-check blocker measured: the gear_small `conflicting recorded observed_pose`
in `capture_f923020bf9ce496eb443826291f8f2c2` is a 1.86e-08 m position and
1.72e-06 rad orientation difference between the tracker pose and the
GETRECOVERYSTATE pose of the same resting part; `restore_recorded_public_facts`
requires exact equality. In the isolated copy, a probe-only 1 mm / 1 mrad
agreement check let full-check preparation reach `outline_ready` with no LLM
call; larger disagreements still return `NEEDS_CONTEXT`.

Applied in this checkout after the machine batch completed (all eight files were
unchanged since the copy; `applied_patch.diff` in the diagnostic folder). The
pose agreement is `safety_preparation.observed_pose_agreement` (0.001 m, 0.001 rad)
in `recovery_framework_gazebo.json`. Verification: 1200 focused tests passed,
`poetry check` (existing deprecations), compileall and `git diff --check`.
A read-only audit found no other two-measurement exact comparison on the offline
path; live re-observation checks (saved_recovery_execution.py:116,
gazebo_safety_preparation.py:378/1323-1340, outline_gazebo_evidence.py:51/58,
kmr_safety_execution.py:83/104, resource_safety_preparation.py:2029-2033) remain
exact, several by documented intent, and matter only for Gazebo work.

Validated full-check attempts from the isolated copy, `offline_49c715da70194d6396a5d7333048b688`
and `offline_a8d51387bae24cf2995e254e3edfdae1`, both stopped `NEEDS_CONTEXT` in
about two minutes; the third was not run because the block is deterministic.
The LLM proposed ur5e-4 to home (RA `FEASIBLE`) and ur5e-3 stowing KET4, but
endpoint evidence calls `environment_models.owners()`, which rejects the whole
recorded plant valuation: ur5e-4 `resource_state="failed"` is outside the
recorded environment-model domain `[idle, at_pick, picked, positioned, placed]`
(robot domains derive from task events; failure injection sets `failed`
directly). Every robot event in this capture is therefore blocked. Accepted
Attempt 17 predates the endpoint policy. Next step (pending user decision):
declare the injected fault state in the plant model, and decide whether
existing captures are re-captured or extended at preparation.

## Previous setup — October 14, 2026 — worked Part slippage and incremental Recovery checks

The worked Part slippage version is preserved separately at
`cais_spade_llm/monitor/environment_runs/recovery_worked/Part slippage/worked.zip`.
Its external `manifest.json` and the manifest inside the ZIP map recorded paths
to archived entries and record hashes, branch, commit, and working changes.
The archive contains 1,273 files, including the accepted outline
`offline_470087fe7a92421abd624ea721aa2ca1` (five events), accepted composition
`offline_1bb1efac84d7443aa37238521ead3b68` (21 ordered primitives), their
observations, native models, provider records, validation evidence, and original
nominal-continuation proof. Original accepted records and official captures
remain retained in their existing locations.

The code snapshot was taken at **2026-10-14T20:31:34.198776+00:00** on
`recovery-framework-journal`, commit
`b120660d35bddb6d9c36678a67ccd660fa8c22d3`, before these Recovery control changes.
It preserves the then-current application and ROS2 source, configuration,
dependency files, relevant tests, and working patch. This snapshot time is
separate from the original generation times retained in the original artifacts
and metadata; it does not claim to reproduce the historical generation code.
Credentials, `.env`, caches, and unrelated generated attempts are excluded.
Four initialization files retain their configuration keys with credential values
omitted, explicitly recorded in the manifest. Original artifact contents and
identifiers are unchanged.

Every archived file was read and hash-verified. Compressed size is 695,069,726
bytes (about 663 MiB); SHA256 is
`4005c518cd9efd8361a01c3f169d7fc5c7e7018c610e745057bb2ddbb3c997e3`.
Recovery displays a read-only **worked — Part slippage** card with the preserved
outline, ordered primitive composition, recorded results, and archive download.
It reports outline safety `passed`, composition RA `FEASIBLE`, CCA composition
`not_checked`, and execution authorization `false`.

New Offline Recovery controls are **RA feasibility validation**, **Physical
evidence**, and **CCA safety validation**. All start unchecked. Select a failure,
use **Start fresh**, choose checks, then use **Generate** to begin an outline.
Selections persist across idle scenario changes, are recorded with the attempt,
and lock when generation starts, including attempts that pause before their
first admitted event. Use **Start fresh** before changing a generated attempt's
checks. A fresh attempt has an empty outline and no inherited primitive program,
worked-result prefix, or earlier rejection history.

| Failure | Default original observations capture | Verified fresh setup record (`draft_ready`, no generation) |
| --- | --- | --- |
| Part slippage | `capture_f923020bf9ce496eb443826291f8f2c2` | `draft_2a63ea11c8674e22b08ebf1296c1c875` |
| Machining breakdown during part processing | `capture_8beafdb8a01e444894cfaf41d29921fd` | `draft_09c80d1e842147d8a0e4005c559052fc` |
| Conveyor breakdown | `capture_1b3f3d1d729a4701b94cff514c783130` | `draft_7047ad5be52b40078ca4dae56e15bd53` |
| ur5e-1 breakdown | `capture_0a7fde8074b6426586b204204f2b3417` | `draft_b460316800b74d3192e3af2c4bdadbdb` |

`SystemBridge.set_failure_example_options` accepts optional
`check_ra_feasibility`, `check_physical_evidence`, and `check_cca_safety` booleans.
Legacy callers retain full validation. The new UI passes choices explicitly.
Generation continues through `execute_multi_turn_recovery`, including factual
retrieval, candidate evaluation, selected-event admission, state propagation,
and nominal handoff. All unchecked uses the existing diagnostic bypass; partial
selections apply individual checks in that same pipeline. Disabled checks record
`not_checked`; structural and identifier checks remain active. No new physical
projection or physical validation runs when Physical evidence is unchecked.
Captured facts remain available and are not rewritten by tentative successors.

CCA evidence depends on the selected predicates. Process-order checking can use
complete, supported owner completion ledgers without spatial geometry. Spatial
mutex checking needs supported occupancy evidence across the proposed event;
a captured stopped pose alone does not establish that interval. If enabled CCA
lacks this evidence, the attempt pauses with `NEEDS_CONTEXT` and retains its
selected specifications. Failed or unresolved enabled checks do not advance the
candidate or monitor history. Partial results remain reviewable with individual
check results and tentative outcomes; they do not become accepted outlines or
worked versions. Existing composition and execution requirements remain in
force. Checking an existing outline without generation is not added here.

`ui_verification.json` beside the archive records actual NiceGUI Recovery page
component and `SystemBridge` verification for all four captures, empty fresh
attempts, unchecked editable controls, the worked card and download, and
`llm_calls=[]`. This is component and bridge verification, not browser JavaScript
verification. `retained_verification.json` records unchanged accepted JSON files,
the five-event/21-primitive result, and all four captures remaining offline-ready.
Archive review and setup make no LLM requests and start no ROS/Gazebo execution.
Recovery outline generation, primitive composition for new attempts, and
execution remain operator-started and pending.

Verification: **863 focused regression tests passed**, covering all eight check
combinations, skipped validator calls, process-order checking without spatial
data, missing mutex evidence, candidate/monitor rollback, nominal handoff gates,
sequential completion-ledger evidence, locked settings after a paused first
proposal, archive integrity, and UI behavior. `poetry check`,
`poetry run python -m compileall -q cais_spade_llm ros2`, the UI CLI help check,
and `git diff --check` passed. Poetry reports existing metadata deprecations.
The final saved-file check found and fixed missing directory creation during
fresh preparation. All **286 affected regression tests passed again**, and
the four-capture UI audit was repeated with exact persisted-record assertions.
All four retained fresh records are `draft_ready`, have empty transition traces,
zero provider calls, and unchecked, unlocked settings.
NiceGUI checks used a normal local event loop after an owned sandbox test
process stalled; only that exact verified test process was stopped. This did
not stop another chat's runtime or start recovery generation or execution.

`setup_cleanup_manifest.json` beside the archive lists four superseded empty
preparation records and two completed temporary verification scripts. Ownership,
hashes, modification times, tracked state, process references, and open handles
were checked before removal: six files, 20,801 bytes released. The four verified
fresh records, original accepted results, captures, native dependencies, and
worked archive are retained. No recovery generation attempt was made or removed.

## Recovery result retention — October 13, 2026 — read before running attempts

**Retirement decision (user, October 15, 2026).** All old data will be retired once
the four schema-2 re-captures are verified and load in Recovery Offline. That covers:
- the schema-1 official captures with their source runs and native checkpoints;
- Attempt 17 and the accepted composition;
- `recovery_worked/`;
- the Conveyor and ur5e-1 capture folders;
- `capture_v2_preview/`.

**Partial retirement (2026-10-16, user request: "it was an old version").** Deleted
`recovery_worked/`, Attempt 17 (`offline_470087fe…`) and the accepted composition
(`offline_1bb1efac…`): 466 files, 8.25 GB. The list is in
`environment_runs/cleanup_20261016_worked_part_slippage/deletion_list.json`.
- Step 3 below (the archive) was not done first, so these files cannot be recovered.
- `monitor/debug` was excluded at the user's instruction.
- `worked_version.py`, its test and `SystemBridge.get_worked_recovery_version` were removed,
  along with the UI card.
- Part slippage has no accepted outline or composition until a new one is accepted.
- Records under `machine_*_20261014*`, `experiments/` and `route_calculations/` still name
  the deleted runs. `experiments/20261010T213423Z/attempt_17_gpt-5.6-sol` keeps Attempt 17's
  session and prompts.

Until then, everything below stays protected. Retirement happens in this order:
1. Point `_DEFAULT_OBSERVATIONS` at the new captures.
2. Move the tests that read retained captures to the new captures or to small fixtures.
3. Write a compressed archive with a sha256 manifest outside the working tree, and
   verify it.
4. Delete the old data under an explicit deletion list.

Keep one accepted recovery outline and one accepted primitive composition for
each of the four failure scenarios. This is the documented Codex cleanup rule
in [AGENTS.md](../../AGENTS.md#recovery-result-retention); the Recovery UI does
not automatically delete failed attempts. Acceptance here refers to saved
planning results and does not establish execution authorization or completed
live recovery.

### Current saved results

| Failure scenario | Recovery outline | Primitive composition |
| --- | --- | --- |
| Part slippage | No accepted (validated) outline. Unvalidated drafts with validation off, October 19: `draft_f7c5a94605c54baf83e8a03a21076c9a` (gpt-5.4 medium) and `draft_85ee4684d69a43d2b615e9af6aba4b71` (gpt-5.4 low) on `capture_77666affc1b9468ca157cf366d706dc8`; see the October 19 section. Attempt 17 was deleted on 2026-10-16 | None. `offline_1bb1efac84d7443aa37238521ead3b68` was deleted on 2026-10-16 at the user's request |
| Machining breakdown during part processing | Successful failure capture retained; incomplete 28-event retry chain retired at the user's request to free disk space; no complete accepted outline | Pending |
| Conveyor breakdown | Verified failure capture `capture_1b3f3d1d729a4701b94cff514c783130` retained and loaded in Recovery Offline; no accepted outline | None accepted |
| ur5e-1 breakdown | Verified failure capture `capture_0a7fde8074b6426586b204204f2b3417` retained and loaded in Recovery Offline; no accepted outline | Pending |

Deleted on 2026-10-16. Attempt 17 was `offline_470087fe7a92421abd624ea721aa2ca1`. Its saved state was
`status=outline_review`, `outline_status=completed`,
`outline_safety_status=passed`, `primitive_status=waiting`, and
`execution_authorized=false`. It contains five outline events and no generated
primitives. The separate accepted primitive composition below uses those same
five events. No further primitive generation is needed for this saved-result goal.

For `Machining breakdown during part processing`, capture
`capture_8beafdb8a01e444894cfaf41d29921fd` records the intended M1 fault with
both machining jobs unfinished and a synchronized frozen checkpoint at simulation
time `84.3455`. Capture compatibility, stationary readiness, native model loading,
and the normal Recovery UI offline preparation all passed. Its setup, seed, and
two safety specifications match the previous scenario. It is a retained failure
capture, not a completed recovery solution. The KMR notes below document partial
planning validation. Their referenced `endpoint_bounds_20261013` replay files
are not currently present in the retained data and are not a loadable accepted
recovery outline.

### Retained record identifiers

The following records are under
`cais_spade_llm/monitor/environment_runs/recovery/`:

| Purpose | Retained record |
| --- | --- |
| Deleted 2026-10-16; was the accepted Part slippage recovery outline | `offline_470087fe7a92421abd624ea721aa2ca1` |
| Deleted 2026-10-16; was the accepted Part slippage primitive composition | `offline_1bb1efac84d7443aa37238521ead3b68` |
| Official Part slippage failure capture | `cf647a0930c64f018d672a27904811b0` |
| Part slippage schema-2 capture used by the October 19 drafts | `capture_77666affc1b9468ca157cf366d706dc8`, archived 2026-10-23 with the other three pre-correction captures (`capture_4d1e4cc0…`, `capture_3aed6165…`, `capture_b2d4286a…`) and their harness folders to `/home/jongh/recovery_archives/20261023_pre_model_correction_captures/` (tar.gz + manifest, 436 files verified), then removed |
| Default failure captures of the corrected resource models (Recovery UI `_DEFAULT_OBSERVATIONS`, 2026-10-23) | Part slippage `capture_82804cb13221467a8b6be4e0eb3e3ac4`, Machining `capture_142d8c7d69d74e39aade558dcbc4b15e`, Conveyor `capture_6d907d9ba40a4bb099cba8804166caf0`, ur5e-1 `capture_41e711f0ce414a1fa0b1b633ec387656` |
| Outline drafts on the corrected-model captures, validation off, gpt-5.4 medium (2026-10-23) | Part slippage `outline_drafts/draft_8149840e8ebc4d5fa4b23d3eef8e061b` (handoff complete, 6 calls); Conveyor `draft_1b233a1b39114d3cb013dffcf73ad0e4` (handoff complete, 7 calls); ur5e-1 `draft_9aea54b0a2fb422da5da018bd8d28cc5` (handoff complete with no assumptions, 9 calls); Machining `draft_b89cc88b33fc4580b2e85aa68369dd23` (LLM declared complete, no handoff, 14 calls: it paused M2's RGOCG4-50 job and left that part on the M2 staging tray, so after M2's `complete_job` for KET4 the goal holds but nominal entry `environment_25` cannot resume) |
| Part slippage outline drafts, validation off (keep for the paper's model comparison) | `outline_drafts/draft_f7c5a94605c54baf83e8a03a21076c9a`, `draft_85ee4684d69a43d2b615e9af6aba4b71`, `draft_8bfa737c6a444de3b8668acda31c05c0`, `draft_f2a9fb151f71496c8333cd3c43971853`, `draft_9fb1a36de41342cb8a360475da42c667` |
| Conveyor breakdown outline drafts, validation off (keep for the paper's model comparison) | `outline_drafts/draft_8bf5b6d05b6642098baedd8e25211852` (works), `draft_0aa3406e820345c8bb72224a2ed9ef14` (works, with `place_end_state`). Failed, user to decide keep or delete: `draft_6d122269bb53462f9059af8431928557`, `draft_7199f1650ca741a1948360ac0728b14d` |
| Official Machining breakdown during part processing failure capture | `capture_8beafdb8a01e444894cfaf41d29921fd` |
| Native checkpoint records for the current Machining breakdown capture | `4a6d23fca53a4d4486ba50f9d005203a` |
| Official ur5e-1 breakdown failure capture | `capture_0a7fde8074b6426586b204204f2b3417` |
| Native checkpoint records for the current ur5e-1 breakdown capture | Source run `01eeb91143e741c69242142690a5a4be` under `../ur5e-1_capture_20261014T182326Z/capture_04/environment_runs/recovery/` |
| Retired incomplete Machining breakdown outline prefix | `offline_6157db87c86841609d5d7895e6b7a0b8`: twenty-eight accepted LLM events through turn 48, nominal continuation `not_ready`. The user explicitly requested removal of this attempt and related outputs. This is no longer a prefix to resume. |
| Retired Machining breakdown retry chain | `attempt_01` through `attempt_18` and `offline_4bed5f73eabd4fb6a6e9ef4ee76984e2`; identifiers and failure findings are recorded in `machine_cleanup_20261014_01a137cc/attempt_history.json`. Official capture and native checkpoint remain protected. |
| Retained source captures required by the saved results | `capture_f923020bf9ce496eb443826291f8f2c2`, `capture_4862656369764af1b78a86063afe6602` |
| Attempt 17 audit | `experiments/20261010T213423Z/attempt_17_gpt-5.6-sol/report.json` |

Keep the required observations, native models, accepted-event validation records,
and other dependencies of these results, plus the small capture/proposal files
required by existing regression tests. Attempt 17's nominal-continuation proof
includes `generation_0001/recovery_outline/turns/turn_00008_validation_0003.json`.
An attempt directory can contain both an accepted stage and a failed stage;
retention is determined by the required evidence, not only the directory's run
status. One accepted result can require multiple supporting files or copies.

## ur5e-1 breakdown — October 14, 2026 — failure captured and verified in Recovery Offline

The user requested removal of `safety_ur5e-1_breakdown.txt`. That 77-rule file
has been deleted. New `ur5e-1 breakdown` selections use only
`safety_M1_mutex.txt`, containing `SAFE_M1_mutex` over `M1.front_access`,
`M1.side_access`, and `M1.workholding` with the existing resource-pair pattern.
The checkpoint name `after_M1_processing_before_pick` is unchanged. It now
requires acknowledged M1 `machine_part` and ur5e-1 `pick_approach`, and latches
before `pick_grasp`. The part remains at M1; the robot records `failed` while
retaining observed recovery motion capability. No home motion is injected.

Retain official capture `capture_0a7fde8074b6426586b204204f2b3417`, source run
`01eeb91143e741c69242142690a5a4be`, captured at
`2026-10-14T19:23:35.853589+00:00`. Its `observations.json` contains the full
acknowledged `machine_part` and `pick_approach` records, twelve remaining nominal
nodes, resource capability evidence, physical M1 occupancy, and the planning
scene. Its `native_preview_model/` archive is self-contained and verified.

The stopped scene was frozen at simulation time `52.434`, native instance
`17891149235950`, checkpoint `94351c8439ec4d27b379384d65d06772`.
`world_paused=true` and `native_observations_unchanged=true`. The robot is
`failed` at M1 with `held_part=None`; M1 is `completed` and retains
`KET4_Square_4mm`. Observed empty attachments and robot occupancy of
`M1.front_access` and `M1.workholding` are verified; `M1.side_access` is empty
of ur5e-1. Supported recovery motion remains available.
The final Gazebo window image is `failure_scene.png`; `failure_scene.json`
binds it to the same frozen instance and simulation time. The owned runtime
was stopped after saving this evidence.

Open **Recovery → Offline → ur5e-1 breakdown** and select the saved observations
from this capture. Actual NiceGUI Recovery page components and `SystemBridge`
loaded and reopened the capture with the sole M1 mutex selected. Offline
preparation reached `outline_ready`, observations remained unchanged, and
**Execute recovery** was disabled. This was component and bridge verification;
no browser JavaScript check was performed. The isolated capture and UI review
both used Manual stages with guards against generation and execution:
`llm_calls=[]`, `recovery_execution_started=false`, and
`execution_authorized=false`. No outline, primitive composition, CCA approval,
recovery execution, or nominal-continuation proof has been produced for this
failure. Recovery generation and execution remain pending.

Keep the successful harness directory
`cais_spade_llm/monitor/environment_runs/ur5e-1_capture_20261014T182326Z/capture_04/`,
including its private `recovery_framework_gazebo.json`, raw observations,
native before/after checkpoints, source-run model files, and report.
`ui_loading.json`, `attempt_history.json`, cleanup manifests, and receipts in
the parent directory record verification and retention decisions.

Attempt `ur5e-1_capture_20261014T182326Z/capture_01`, source run
`b1b222d5c04b43f9a27878802b03559e`, stopped before the requested failure:
KMR `place_approach` at M1 exceeded the wall timeout while waiting for
`/KMR/KMR_iiwa_joint_trajectory_controller/follow_joint_trajectory`.
The trajectory required `9.447576947` simulation seconds; only `7.1`
simulation seconds elapsed during its `120.084358769` wall seconds.
No ur5e-1 fault was injected, no observations record was published, and no
LLM or recovery execution occurred. Its owned launch stopped. The successful
replacement uses `KMR.arm_motion_wall_timeout_sec=360` in its private capture
scene, with unchanged task, controller-completion, and physical-observation
checks. The shared scene and normal controller wall-budget default remain
unchanged.

The finished `capture_01` diagnostics are retained in `attempt_history.json`;
its 58 disposable files were removed (8,949,688 logical bytes). Its remaining
owned embedded XMPP server was identified by its private workspace and stopped
before deletion. The harness now stops its own XMPP server during cleanup.
`capture_02` was refused before launch because this task's CLI help verification
overlapped preflight. No runtime or capture was created; its one log file and
empty directory were removed after recording the finding. Subsequent capture
attempts ran sequentially after the completed CLI check.

`capture_03` reached the intended ur5e-1 fault, source run
`fee200159471412a87bf468093aab35d`. Its real failure observations and native
model archive were protected until the replacement was verified and loaded.
Final capture verification found that `EnvironmentProductContext.acknowledge`
shared its `after` state with current resource valuations. Setting ur5e-1 to `failed`
rewrote the recorded `pick_approach` successor from `at_pick` to `failed`.
The acknowledgement and controller observations still match exactly. The
repair saves an independent `after` snapshot; the strict provenance check
remains unchanged. Frozen raw observations are now persisted before verification.
The replacement passed these checks and loaded in Recovery Offline before
`capture_03` and its log were removed. The stopped UI review duplicate
`offline_b7872fb687b542f99caf31607986ada8` was also removed; it contained no
generated outline or primitives. Ownership, modification times, tracked files,
and open handles were checked against explicit deletion manifests. Retained
observations and native models verified again after cleanup. This task removed
159 disposable regular files across attempts and the UI review: 352,495,198
logical bytes and 353,099,776 allocated bytes (336.74 MiB). Official captures,
accepted results, successful capture dependencies, regression fixtures, and
other chats' work were preserved.

The final focused regression run passed all 299 tests for fault handling,
mutex/default selection, capture, nominal dispatch, and acknowledgement history.
Offline loading and preparation also passed through the actual Recovery page
components and `SystemBridge`. `poetry check`, Python compilation, and the UI
CLI help check passed; Poetry reports existing metadata deprecations.
Mutex coverage rejects simultaneous occupancy across all nine
pairs of the three M1 areas, counts the stopped failed robot, and preserves
`NEEDS_CONTEXT` for missing geometry evidence. These checks validate the saved
failure and supported recovery inputs, not a generated or executed recovery.

The selector assertion now matches the files present in the shared checkout.
Legacy machine/conveyor bundle tests still reference files removed by concurrent
work; those cases were outside this ur5e-1 regression run. Their updates remain
separate from this capture. `retained_result_verification.json` confirms that
all four current failure-capture inputs remain selectable and offline-ready,
and that the accepted Part slippage records remain readable after cleanup.

### Cleanup after recovery work

- After diagnosing a finished failed attempt, record its identifier, concise
  failure reason, and next step in this handoff. Delete its disposable outputs
  before starting another attempt or ending the task.
- Protect active work and the current partial attempt still needed for debugging,
  including work owned by another chat or process. Delete abandoned attempts
  once their useful findings are recorded and required dependencies are retained.
- When primitive composition fails, preserve its accepted recovery outline and
  shared dependencies. Remove only the failed stage's disposable files; do not
  delete the containing attempt wholesale.
- Verify and load a selected replacement before deleting superseded accepted
  results. An accepted event prefix, `NEEDS_CONTEXT`, or a completed failure
  capture does not establish a complete accepted solution.
- Build an explicit deletion list, check file ownership and recent changes,
  and exclude active or newly changed records and required dependencies.
  Preserve application code, regression coverage, and unrelated working or
  staged changes.
- Confirm retained results still load with their accepted events and required
  nominal-continuation evidence, and that official failure captures remain
  selectable. Report the files removed and space released, then update the
  retained identifiers and stage status above.

## Part slippage primitive composition — October 13, 2026 — accepted and verified in Recovery UI

Retain `offline_1bb1efac84d7443aa37238521ead3b68`, with
`generation_0002/composition.json`, `proposal.json`, `outline_request.json`,
`recovery_primitves/`, and the copied `recovery_outline/` nominal-continuation
proof. Its `primitive_status=completed`, `composition_feasibility=FEASIBLE`,
`execution_authorized=false`, and `cca_composition_status=not_checked`.

| Original outline event | Ordered primitives | RA result |
| --- | --- | --- |
| `clear_failed_robot_from_assembly_board` | 1 | `FEASIBLE` |
| `stage_ket4_at_buffer` | 5 | `FEASIBLE` |
| `pick_misplaced_gear_small` | 5 | `FEASIBLE` |
| `place_gear_small_on_assembly_board` | 5 | `FEASIBLE` |
| `reacquire_ket4_for_nominal_continuation` | 5 | `FEASIBLE` |

The final LLM revision removed the unsupported perception step and passed all
unchanged RA checks. Every accepted `primitive_steps` program matches an actual
recorded provider response exactly. All five events retrieved applicable
`/capability_decompositions/` context before authoring. The retained provider
records include the revisions and context requests; they are not additional
accepted compositions. This is one complete composition containing 21 primitives.

The real Recovery UI, using `SystemBridge`, loaded the result through
**Load saved result**, displayed every event's ordered primitive table and RA
evidence, downloaded the composition and original provider input/output, and
reopened the page with the same five programs. Browser checks reported no page
errors. The source outline and nominal-continuation proof passed integrity checks;
185 retained source, capture, and safety files match the original hashes. The
39 missing files from that initial inventory belonged exclusively to unrelated
Machining breakdown captures superseded by concurrent work, as recorded in the
current Machining breakdown handoff; this task did not remove them.

All **265 automated regression tests** passed for composition, Part slippage
lifecycle, and request records. `poetry check`, Python compilation, and the
scoped whitespace check passed. Poetry reports existing metadata deprecations.
These counts describe automated tests, not generated compositions.

Open **Recovery → Offline → Part slippage**, choose the saved Part slippage
entry that enables **Load saved result**, and select that button. Its observation
group is `capture_f923020bf9ce496eb443826291f8f2c2/observations.json`, with capture
time `2026-10-10T17:03:11.848678+00:00`; it resolves to the accepted composition.
Expand
the five event programs under **Primitive composition** and use its Input/Output
and **Download composition evidence** controls. This goal does not include CCA
composition approval, Gazebo execution, hardware execution, or Continue production.

Audit reports, browser evidence, test logs, cleanup manifests, and archived
failure findings are retained under `generation_0002/goal_verification/`.
The verified UI review copy `offline_ea8735865b6c443bb4d51e2d980ef4d2` is
disposable; it contains the same composition and no new generation. The temporary
UI server was stopped before cleanup. The four superseded partial attempts
`offline_02cac3fc894c46dda577ba9cc0f2d7e8`,
`offline_2c0cb51e112d43e7837cccf23ee6198c`,
`offline_e33cbd0bf53e4a43bcd74c2b585a5a7b`, and
`offline_31f19fe0d1a44f459b56a9cf03440fb1`, plus that UI review copy,
were removed after their provider records were checked byte-for-byte against
the accepted bundle and their remaining diagnostic evidence was archived.
The explicit manifest covered 720 files, 20,493,846,660 bytes. Ownership,
modification times, active processes, and open file handles were checked first.
The original accepted outline, official Part slippage capture, regression
fixtures, and unrelated active work were excluded. A fresh Recovery UI load
passed after this cleanup, including provider downloads, all five ordered
programs, and reopening. It created only the disposable review copy
`offline_286a7093ffb84b6499da1c33b96fabef`, with no new generation. Its
temporary UI server is stopped, its browser evidence is archived, and that
verified duplicate was removed after ownership/mtime and open-file checks:
172 files, 3,305,277,648 bytes. Total cleanup during this goal removed
**1,270 disposable files, 41,122,655,001 bytes (38.30 GiB)**. Per-file manifests,
process checks, receipts, and archived findings are in `goal_verification/`.
The accepted composition and Attempt 17 remain retained separately. No temporary
verification UI or generator is running.

The final retained-result audit passed after all cleanup. The saved-result index
contains exactly one complete RA-accepted Part slippage composition and selects
`offline_1bb1efac84d7443aa37238521ead3b68` for its observation group. Both official
failure captures remain selectable and ready for offline preparation; a failure
capture itself is not a completed recovery result. See
`goal_verification/retained_result_selection.json` for the exact paths.

The browser reports contain no JavaScript page errors. NiceGUI also logged a
nonfatal stale-client warning while the test browser reloaded or closed a view;
the completed save, fresh load, all five displayed programs, evidence downloads,
and reopening checks passed. The server logs are retained with the browser audit.

### Earlier attempts and resolved findings

This goal continues Attempt 17 without changing its five accepted events or its
nominal-continuation proof. It uses the legacy `recovery_primitves/1` context
retrieval and event-local LLM authoring pipeline, with ordered ResourceAgent
composition validation and Recovery UI persistence. Execution remains disabled;
CCA composition validation is a separate, unchecked stage.

The user has now explicitly approved sending the saved Part slippage recovery
context to `api.openai.com`: the five outline events, robot/part observations,
primitive contracts, and RA validation feedback, without credentials in prompts.
That approval resolved the earlier permission blocker below. Generation resumed
from the saved replay checkpoint with `execution_authorized=false`.

- `offline_31f19fe0d1a44f459b56a9cf03440fb1` revalidated the first two
  events and accepted `pick_misplaced_gear_small` and
  `place_gear_small_on_assembly_board`. It stopped on three identical fifth-event
  rejections: `detect_parts` cannot use the projected buffer placement as a fresh
  perception observation. All four accepted programs remain retained; this is
  not a complete composition. Local replay reproduced the finding without API
  calls. A diagnostic-only variant omitting the unnecessary `detect_parts`
  passed the unchanged RA validator, while the original response and an
  unrelated-target variant were rejected. This diagnostic is not a generated
  accepted result. The prompt now clarifies that projected placement may support
  `compute_pick_targets` without establishing fresh detection. The legacy pick
  example also computes targets without `detect_parts`. Resume the fifth event
  through the configured LLM, preserving actual provider responses and all checks.
  Retain this four-event checkpoint for the continuation; remove only its
  disposable `primitive_input_preview.json` after ownership/mtime checks.

- `offline_e33cbd0bf53e4a43bcd74c2b585a5a7b` revalidated the saved first
  event and accepted `stage_ket4_at_buffer` on its first new LLM revision.
  It then stopped on three identical `product_geometry` rejections while
  authoring `pick_misplaced_gear_small`. Seven new provider responses are retained
  (one for the accepted buffer revision, two context requests and four pick
  proposals); no complete composition was accepted. The third repeated geometry
  finding produced `primitive_blocked`; the owned process is terminal.
  Diagnosis: these geometric parameters are optional, and the composition owner
  reads recorded geometry internally. The old feedback said only "use recorded
  inputs", so the LLM kept copying recorded geometry into unsupported explicit
  overrides. The repair names all unsupported parameters together and explicitly
  directs the author to omit them, while preserving exactly the same rejection
  checks. The composition prompt now explains this owner input behavior.
  All 38 composition tests pass, including a regression that rejects every
  unsupported override and verifies the omission path. Replay the saved responses
  locally before resuming the third event; retain both accepted events.

- `offline_b1c7e6943b704d71a71c610c7b66d356` is an abandoned diagnostic
  copy of the accepted outline. The sandbox prevented a completed background
  worker from waking the waiting asyncio task. An isolated delayed-worker
  reproduction timed out in the sandbox and passed outside it. The process was
  stopped with exit code 130; no LLM request or composition generation occurred.
  Preparation and asynchronous checks now run outside that sandbox. After
  verifying 224 protected source and safety files unchanged, the explicit
  ownership/mtime-checked deletion list removed this disposable copy: 71 files,
  2,923,398,097 bytes. The original accepted outline and official captures remain.
- The user explicitly authorized use of the OpenAI API after automatic approval
  review rejected the initial generation command. The configured destination is
  `api.openai.com`; current default generation settings are `gpt-5.4` with
  `medium` reasoning effort. Actual requests began in the attempt below.
- `offline_1d4f3ff2df4d4fb8990ef7e51d8b0f9e` completed preparation only;
  no API request was sent. It is a superseded diagnostic copy awaiting final
  replacement verification and cleanup. The original Attempt 17 is retained.
- `offline_8b2b23bb172d424281a88fe5c678fbbb` made two recorded OpenAI
  requests and obtained a FEASIBLE program for
  `clear_failed_robot_from_assembly_board`. Before the second event, the kernel
  killed Python for memory exhaustion (3,638,552 kB anonymous resident memory).
  No second-event request was sent. The active composition was carrying the
  completed outline's large projected geometry through repeated debug copies.
  The scoped fix keeps that geometry and monitor context in the unchanged
  `outline_request.json`, after verifying its nominal proof, and removes them
  from the active composition checkpoint. Recorded owner inputs remain intact.
  Local replay verifies the first event's evidence against both contexts;
  four focused lifecycle/preservation tests pass. The next generation uses this
  fix; raw responses and first-event evidence remain debugging dependencies.
- `offline_02cac3fc894c46dda577ba9cc0f2d7e8` again validated the first
  event with two actual provider responses. It was stopped with exit code 143
  before another request because the prior batch's full debug session was being
  nested inside the next batch. The additional scoped fix passes compact history
  references, excludes historical turns from prompt-input copies, and keeps full
  substream records in their original artifacts. Thirty-four composition tests
  pass, including regressions for this copying path. The next attempt replays
  the saved `need_context` and `primitive_steps_ready` responses through the
  current responsible ResourceAgent before continuing the four remaining events;
  it does not ask the LLM to regenerate the first event.
- `offline_d5ccc1c734e945ae89e1fc0a7af8fc1f` was killed during proof
  verification before primitive generation or any provider request. The unchanged
  nominal-continuation artifact is 937 MiB; reading it while new context copies
  and concurrent tests were resident exceeded available memory. Preparation now
  verifies the source before allocating a composition copy and excludes inactive
  outline fields before copying. Heavy checks and generation run sequentially.
  The replay source remains `offline_02cac3fc894c46dda577ba9cc0f2d7e8`.
- After the transcript-copy fix, all 261 focused composition, lifecycle, and
  request-record tests passed. Poetry check and Python compilation passed;
  Poetry reported existing metadata deprecations.
- `offline_2c0cb51e112d43e7837cccf23ee6198c` reused the first event's
  actual responses and made five new provider requests for `stage_ket4_at_buffer`.
  It was stopped after three repeated `pick_ctx` rejections, before another
  request. Owner preparation enriches its private `_task_ctx` with placement
  outputs; comparing that enriched object with the author's unchanged recorded
  input incorrectly rejected the exact whole-context binding. The validator now
  compares that binding with the recorded input while retaining the owner target
  calculation and geometric checks. Thirty-six composition regression tests pass.
  Replaying the last actual response now reaches a legitimate completion finding:
  the LLM used `approach_pose` for the final retreat instead of `retreat_pose`.
  The next revision resumes the served context and current RA feedback, retaining
  the first validated event and all original responses. The generation loop now
  stops after three identical composition findings even when proposals change.
  The stop-limit regression also passes (37 composition tests total). The
  resumed command was rejected by automatic approval review before execution:
  explicit approval to send the saved recovery context to `api.openai.com` was
  pending at that point. No new attempt or provider request was started by that command.
  The failed stage's disposable input preview was removed with an explicit
  ownership/mtime-checked list: one file, 8,669,011 bytes. Current replay inputs,
  actual provider records, validation evidence, and accepted outline dependencies
  remain protected.
- Cleanup removed superseded diagnostic copies
  `offline_1d4f3ff2df4d4fb8990ef7e51d8b0f9e`,
  `offline_8b2b23bb172d424281a88fe5c678fbbb`, and
  `offline_d5ccc1c734e945ae89e1fc0a7af8fc1f` after dependency and ownership
  checks: 231 files and 11,322,723,101 bytes. All 224 protected file hashes
  matched before deletion. Their failure reasons are recorded above. They are no longer inputs
  to the replay, which retains `offline_2c0cb51e112d43e7837cccf23ee6198c`
  and its referenced first-event records in
  `offline_02cac3fc894c46dda577ba9cc0f2d7e8`.
- Final local verification after the owner-input and stop-limit repairs:
  **264 automated tests passed** across composition, Part slippage lifecycle,
  and request-record coverage. These are tests, not 264 generated compositions.
  `poetry check` and Python compilation passed with existing Poetry metadata
  deprecations. The retained partial attempt is explicitly `blocked`, with
  `primitive_status=failed`; it is not a completed composition. Its
  `generation_0002/local_revalidation/` directory retains the replayed
  one-event prefix, the pending `stage_ket4_at_buffer` batch with eight served
  context entries and current retreat finding, regression logs, cleanup lists,
  and protected-file fingerprints. No request was sent after the repair.
  The same external approval blocker persisted across three consecutive goal
  turns. Goal work was blocked pending the user's answer to the payload-specific
  permission request at that point; all local fixes, tests, and the exact resume checkpoint
  are retained. Completion still requires all five RA-validated event programs
  and verification of the complete saved composition through Recovery UI.
- The real Recovery UI loaded Attempt 17 and reopened its five-event outline
  with `execution_authorized=false`. This verifies the retained outline, not a
  completed primitive composition. The isolated browser's disposable review copy
  is `offline_2b1ced6d396643e6a049be44a24a58bb`; its report and screenshot are
  retained under the partial attempt's `generation_0002/local_revalidation/`.
  The temporary UI server is stopped. The unchanged, owned duplicate was removed
  after verification: 71 files and 2,923,398,097 bytes. Both official failure
  captures remain selectable. Cleanup across this goal removed 376 files and
  17,306,192,671 bytes (about 16.12 GiB); per-file lists and receipts are retained.
- Initial composition regression checks: 32 passed. The broader offline,
  Part slippage, and Recovery UI baseline checks passed all 513 cases in
  72.71 seconds outside the sandbox.

## Current handoff — October 14, 2026 — restore multi_turn diagnostic

The user authorized removal of inactive machine-breakdown retries and the
standalone `draft_9ec5b77f7de846c1ba2043f63b8b7229`, then exactly three fresh
attempts through `execute_multi_turn_recovery`. RA, CCA and physical validation
are bypassed only for the diagnostic; dynamic factual retrieval, candidate
comparison and session progression remain in the existing pipeline. Target
approximately five to six task-level recovery events and stop at proposed
nominal handoff. No formal acceptance or execution is implied.

Cleanup audit: 664 disposable files, 3,321,692,160 allocated bytes, no open-file
or running-Python ownership found. The explicit file/mtime/inode/uid manifest
is retained under `machine_multiturn_diagnostic_20261014/cleanup_manifest.json`.
It covers five inactive machine `offline_` attempts, their
`legacy_forward_backward_20261014` duplicates and the standalone draft.
Protected: official capture `capture_8beafdb8a01e444894cfaf41d29921fd`, its native
model, source run `4a6d23fca53a4d4486ba50f9d005203a`, `capture_04`, both accepted
Part slippage results and their dependencies. The failed attempts established
only one accepted M2 pause prefix; missing part bindings, UR parameter conflicts
and unresolved KMR endpoint search prevented complete machine acceptance.
The supplied robot pickup replay and a before prompt are preserved compactly.
The preceding standalone draft used a different loop and is not a framework
convergence result. `OutlineDraft` now adapts the existing
`execute_multi_turn_recovery`, phase handlers, candidate comparison and factual
retrieval. The diagnostic flag defaults off. Missing start fields are retained
as assumptions only in this diagnostic; normal validation remains unchanged.
Focused and related regression suites passed: 905 tests. `poetry check`, Python
compilation and `git diff --check` passed; Poetry emitted existing metadata
deprecation warnings. All 11 protected hashes match the cleanup manifest, and
the official capture remains selectable with eight native-model files present.

Exactly three independent fresh API attempts finished with the capture's
`gpt-5.4`, `medium` settings, without an earlier accepted prefix or rejection
history. Attempts two and three ran in parallel with attempt one. There were
42 provider calls in total and no provider errors. The genuine attempts made
no context requests; dynamic retrieval was exercised through scripted provider
tests of the same orchestration. No RA, physical, CCA, formal nominal-continuation
or composition call occurred. Every diagnostic check remains `not_checked`.

| Attempt | Identifier | LLM calls | Selected events | Generation seconds | Outcome |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | `draft_a9e49e5d3e7e47b2928cd342ce10ff51` | 12 | 8 | 931.84 | `selection_unresolved` after three no-progress batches; compact evidence retained, full outputs removed |
| 2 | `draft_6e6f4844a4d2422891965f16f65c4e3d` | 20 | 15 | 1582.56 | Proposed handoff with assumptions; compact evidence retained, full outputs removed |
| 3 | `draft_66b5fc99daab491ea8f25624d1790b5a` | 10 | 9 | 687.22 | Selected unvalidated draft; full native prompts, responses, comparisons and draft retained |

Retain the selected draft under
`cais_spade_llm/monitor/environment_runs/recovery/outline_drafts/draft_66b5fc99daab491ea8f25624d1790b5a/`.
`SystemBridge.part_slippage_action("load_draft")` reproduced its exact nine rows
without provider calls. NiceGUI rendering confirmed that `Load draft` is enabled
and `Continue`, `Test in Gazebo`, and `Execute recovery` are disabled. This was a
headless UI verification, not a live browser or Gazebo execution. Verification
records and all three compact summaries are under
`machine_multiturn_diagnostic_20261014/`. The saved request/response timestamps
show 52–89 seconds per outline API response in the selected attempt; the 905
regressions themselves took 199.19 seconds. Codex Fast mode did not alter the
provider payload's saved model or reasoning settings.

The selected sequence is the actual LLM-generated sequence, without regrouping:

| Event | `event_name` | `resource_jid` | `part_name` | Proposed outcome |
| --- | --- | --- | --- | --- |
| `RECOVERY_SEQ1` | `pick_part` | `recovery-resource-1@localhost` | `KET4_Square_4mm` | Held by the resource at M1 |
| `RECOVERY_SEQ2` | `place_part` | `recovery-resource-1@localhost` | `KET4_Square_4mm` | Released on Conveyor |
| `RECOVERY_SEQ3` | `pick_part` | `recovery-resource-2@localhost` | `KET4_Square_4mm` | Held by the resource at Conveyor |
| `RECOVERY_SEQ4` | `place_part` | `recovery-resource-2@localhost` | `KET4_Square_4mm` | Released on M2 staging tray |
| `RECOVERY_SEQ5` | `pick_part` | `recovery-resource-2@localhost` | `KET4_Square_4mm` | Held by the resource at M2 staging tray |
| `RECOVERY_SEQ6` | `place_part` | `recovery-resource-2@localhost` | `KET4_Square_4mm` | `part_location=M2`, `part_state=loaded`; `held_part` omitted |
| `RECOVERY_SEQ7` | `machine_part` | `recovery-resource-6@localhost` | `KET4_Square_4mm` | Square program and completed job |
| `RECOVERY_SEQ8` | `machine_part` | `recovery-resource-6@localhost` | `RGOCG4-50_Round_4mm` | Circle program and completed job |
| `RECOVERY_SEQ9` | `place_part` | `recovery-resource-2@localhost` | `KET4_Square_4mm` | Released on M2 staging tray |

The five-to-six-event target was missed in every attempt. The selected draft
omits pausing/removing the recorded M2 round job before loading KET4, leaves
`held_part` unchanged at event six despite a claimed load, and includes nominal
round processing inside recovery. Its proposed handoff assumes M2's
`resource_state` and `part_name` after the final tentative transfer. Thus it is
not a correct, accepted recovery or a nominal-continuation proof. Attempts one
and two additionally select round completion before square completion, contrary
to the supplied ordering requirement. Attempt one also claims square completion
with KET4 still on Conveyor. These are generation/selection findings with
validators bypassed; they do not establish that RA or CCA is too strict.

Next work: repair task-level granularity, the presentation of captured replacement
objectives and public owner fields, and completed `resume_suffix` feedback before
further unchanged API retries. Keep this diagnostic separate from validation
repair and execution. The first fresh outline prompt contains zero `pick_part`
or `place_part` mentions and excludes `place_release`, `gripper_reach`,
`additional_bindings`, and `active_job`; the former names came from the LLM.
`machine_part` and `move_to_resource` remain exact captured pending-task names.

Cleanup of the other two fresh attempts removed 224 additional files and
9,568,256 allocated bytes after ownership/mtime checks. Total task cleanup:
888 disposable files, 3,331,260,416 allocated bytes (about 3.10 GiB). The official
capture, native model and all protected accepted Part slippage dependencies
retain their original hashes. Separate deletion manifests and receipts record
each removal. The previously documented 12 failures in other AP/KMR suites were
not rerun; none of the current 905 focused regressions failed.

## Previous handoff — standalone generation-only draft (superseded)

The immediate user-requested diagnostic is complete. `Generate outline only`
and `Load draft` are available for saved observations through the existing
`SystemBridge.part_slippage_action` interface. The separate `OutlineDraft`
provider loop uses public factual context and filtered retrieval, requests one
ordered `transition_trace`, and stops at the first well-formed draft. It does
not instantiate resource owners or call RA, physical projection, CCA, candidate
comparison, nominal-continuation checking, or primitive composition. It accepts
missing physical evidence. The six-call budget includes at most one formatting
repair; diagnostic provider calls disable application and SDK retries. Normal
provider callers retain their existing retry policy.

Deleted `draft_9ec5b77f7de846c1ba2043f63b8b7229` at the user's request during the
cleanup above; its concise findings and before prompt remain in the diagnostic
handoff artifacts. It was a genuine `gpt-5.4`, `medium` response from official capture
`capture_8beafdb8a01e444894cfaf41d29921fd`, generated in one API call in
159.28 seconds, with 12 events and no retrieval or formatting-repair turns.
The exact prompt is 15,116 UTF-8 bytes. Provider usage is 7,606 prompt tokens,
10,928 completion tokens (including 8,417 reasoning tokens), 18,534 total.
The draft retains both exact safety requirements, raw provider request and
response, prompt, model settings, source hash, usage, timing, and UI checks.
All four checks are `not_checked`; `execution_authorized=false`. This is an
unvalidated draft, not an accepted recovery outline or nominal-continuation proof.

The proposed order is: pause the M2 round-part job; unload
`RGOCG4-50_Round_4mm` to `M2 staging tray`; record M2 idle; extract
`KET4_Square_4mm` from M1 to Conveyor; record M1 stopped/idle; load the square
part from Conveyor into M2; complete square trim; unload it to Conveyor;
reload the staged round part; complete circle trim; use KMR to transfer the
square part from Conveyor to `Buffer For Machined parts`; use `ur5e-3` to
place it at `assembly_board-v1`. These events are preserved exactly as returned.

Manual reading identifies remaining gaps, without running validators: event 1
keeps `active_job=environment_23` while paused, whereas the owner pause code
clears `active_job`; events 3 and 5 are bookkeeping-style state changes rather
than clearly described physical tasks; no Conveyor transport event connects
the robot handoffs; event 11 assumes KMR can service Conveyor and
`Buffer For Machined parts`; event 12 leaves the part `ready` at
`assembly_board-v1`. Job completion, M2 occupancy, processing continuity and
nominal continuation remain unproven. Do not revise these outputs by hand or
promote them to accepted events. No further API request was made after this draft.

Verification: 322 focused draft, provider-record, Recovery workflow and UI
tests passed. A restricted-sandbox asyncio thread-shutdown hang was reproduced
with a minimal `asyncio.to_thread` program; the suites passed with normal
thread wakeups. `Load draft` reproduced the exact record without a provider
call. NiceGUI Client rendering displayed all 12 rows, enabled both diagnostic
buttons, and disabled Continue, Test in Gazebo and Execute recovery. This was
render/control verification, not a browser-driven or live Gazebo test.
`ui_action_verification.json` records zero forbidden calls and unchanged hashes
for the official capture, the retained machine accepted prefix and both
accepted Part slippage stage records. `ui_render_verification.json` records the
actual rendered controls. `poetry check`, compilation of `cais_spade_llm` and
`ros2`, and `git diff --check` passed; Poetry emitted only the existing metadata
warnings. The retained draft and verification logs comprise 10 files using
143,360 allocated bytes (140 KiB). No failed recovery attempt or duplicated
capture was created.

The validated-generation work below remains incomplete and separate. Its
existing 12 reproduced AP/KMR baseline failures are unchanged; no validation
was relaxed by this diagnostic.

Fresh attempts now use `incremental_candidates_validated`. A later preparation
override that silently restored `incremental_validated` was removed. Saved
results keep their recorded mode and policy. Candidate events use detached
accepted states; only the selected candidate is committed. Comparison ignores
accumulated monitor observations when detecting repeated modeled states.
Unresolved candidate batches remain available for validation-only retry.

Policy version 2 retains matched recovery predecessor guards as composition
prerequisites; exact nominal tasks still enforce them. Machine-owner duration
calculations can start or resume from proven inactive processing without an
observed absolute clock. They retain `calculation_clock_sec` and relative-time
provenance without inventing an observed `simulation_clock_sec`. Program,
access, custody and duration checks remain. Partial peer geometry can stay
unknown when selected safety predicates do not need a more precise value;
stale geometry is invalidated.

The shared `handling_robot_access` calculation and detached correction of
inherited printer-scene capabilities have regression coverage. Explicit limits
remain unchanged. The missing preparation import and later overwrite of
calculated owner evidence were corrected. Public feedback hides operation
menus and numeric reach limits, including repeated nominal feedback filtering.

Local replay accepted the first two retained machine proposals through RA and
both safety specifications. The candidate selector also accepted the retained
pause proposal as progress. These replay checks are not a new complete recovery.
Official capture `capture_8beafdb8a01e444894cfaf41d29921fd` remains the input;
the deleted 28-event attempt has not been restored.

The broad six-suite run recorded 1,626 passes and three failed assertions or
fixtures; their corrections passed focused reruns. The later workflow run
recorded 1,005 passes and one incomplete test-owner fixture; its correction and
the partial-projection cases passed together (six tests). The 12 documented
legacy AP/KMR failures reproduce separately, with their tests and owner code
matching the before-change hashes. `poetry check`, Python compilation and
`git diff --check` passed during this work; repeat final static checks after
any further repair.

Fresh approved API attempt `offline_dd188e10021140579d3c24b4b7e9c29e` stopped
after its first four-candidate batch, with no accepted events. The corrected
robot snapshot was being adapted differently on the validator and preparation
paths (`staging_areas` empty versus omitted). That mismatch is fixed. Current
candidate batches may return unresolved findings for one proposal revision;
repeating the same conditions and evidence retains `NEEDS_CONTEXT` and the
validation-only retry. No missing evidence becomes acceptance.

No-API replay now reaches robot forward checks. The retained pause proposal
conflicts with the owner's `active_job` update; placement lacks current custody
and, for `Conveyor`, destination calculation evidence. The KMR proposal does not
match the actual effects of its matched movement declarations. Preserve exact
provider requests/responses and compact diagnostics; delete this finished
attempt's duplicated disposable state before retrying with these findings.
The next step is a fresh API revision using the corrected owner snapshot and
candidate feedback, without supplying a replacement event or route.

Attempt `offline_bebf26fc74e140cdaa41622dc97a02b0` generated corrected pause and
stop alternatives. Its worker selected the pause with RA `FEASIBLE` and CCA
passed. The run was interrupted to repair repeated calculation cost before
another API request; the selected worker record is retained for local prefix
revalidation, separate from the canceled UI session. Pure joint calculation
reuse is now scoped to one candidate batch, with independent FK/evidence checks
on every use. Changed geometry invalidates it and the cache does not survive
the batch (nine focused tests passed).

No-API attempt `offline_83e0abd9096e4e8da260b34c96ff5ea4` stopped before replay:
the diagnostic runner incorrectly compared a derived `expected_start_state`
against the raw provider response. It now checks that field against the saved
RA primitive-support record and preserves the exact provider-authored fields.
That attempt's disposable preparation was deleted after ownership checks.

Local attempt `offline_59d2957d7039494a9a2d110e954b618b` revalidated the original
LLM-selected M2 pause with RA and CCA, without an API request. Its retained
`attempt_04_revalidation/revalidated_session.json` is the current one-event
prefix. It is not a complete recovery outline.

Robot endpoint projection now uses the recorded robot task declarations,
`handling_robot_access` calculations, joint limits and custody configuration.
It retains unknown intervening occupancy instead of requiring a motion program.
Supplied pickup verification cases for `ur5e-1` at `M1` and `ur5e-2` at `M2`
both pass RA, endpoint projection and the two unchanged safety specifications
from the accepted pause state. These cases are not LLM-generated events and
have not been appended to the accepted prefix. Compact evidence is retained in
`legacy_forward_backward_20261014/supplied_robot_pipeline_replay.json`.

Attempt `offline_0270fc81a3234d1d805831023ba49619` resumed that prefix and made
two genuine provider requests. Its first five-candidate batch required revision;
the following batch reached candidate comparison, which crashed because
`_infer_outline_macro_signature` converted a projected seven-value
`current_pose` list directly with `dict()`. The pose-format compatibility repair
does not change feasibility or safety meanings. Replay the retained second
batch without another API request. Preserve its provider responses and prefix
evidence; remove duplicated preparation output after recording the deletion
list and checking ownership. The original failure capture remains unchanged.

Local attempt `offline_3849da544cb845a88fab41d764054f55` completed that replay
without API calls in 334.22 seconds. The comparison exception is fixed (13
focused tests passed). The prefix remains one accepted M2 pause: the retained
KMR pickup passed independent RA but its endpoint calculation found no joint
witness; this is unresolved search, not demonstrated unreachability. The KMR
reposition passed RA/CCA but did not establish comparison progress. Full
validation of hypothetical follow-up events contributes to comparison cost.
The next retry uses compact factual feedback for recorded parameter conflicts
and bounded-search failures (three focused tests passed), without replacement
events or altered acceptance criteria. Retained rejected outputs are used only
to render that feedback; they do not become accepted state. Attempt 05 cleanup
removed its duplicated preparation, releasing 77,496,320 allocated bytes.
Attempt 06 preparation and bound prefix evidence remain required for reviewed
reuse and debugging. The source capture hash is unchanged.

Attempt `offline_c618ccdc5de94cb2bfeb359ea3994422` stopped before any API call:
the diagnostic evidence-reuse helper still indexed legacy `next_transition`
instead of selecting the recorded `candidate_events` entry. Both formats now
retain the exact original selected fields. Keep its compact report and remove
its failed preparation after ownership checks, then retry from attempt 06.

Attempt `offline_1b6bf9369b314de9bd564eb2c3326617` made three genuine API calls
after strict reuse checks across 51 prepared fields. It returned positioning
proposals, requested public context, then repeated positioning proposals with
the same missing `part_name` bindings. No new event was accepted. The worker
stopped at `NEEDS_CONTEXT`; `report.error=null` does not mean success (the
recorded result is `blocked`). Retain its provider requests/responses, failed
candidate evidence and compact worker session for diagnosis. Remove its
duplicated preparation after ownership checks. Source observations and the
one-event prefix remain unchanged; no execution or Gazebo request occurred.

Current boundary issue: `_kmr_endpoint` still walks the owner's program and
checks intermediate parameter/configuration compatibility after RA has checked
the unordered primitive union independently. A missing bounded-search witness
there does not undo RA `FEASIBLE`, but currently blocks successor evidence.
Audit those checks against required forward parameters and the selected APs;
do not add more endpoint/program requirements to make the outline converge.
The latest API failure itself was earlier, at RA proposal bindings, not CCA or
geometry. The offline error message now preserves that distinction instead of
labeling every `NEEDS_CONTEXT` as endpoint/bounds failure. Hypothetical follow-up
RA/CCA checks in candidate comparison also remain a cost to examine. No further
unchanged API retry is warranted. Completion and saved-result loading remain
outstanding. The latest full outline suite passed 264 tests.

Final retention checks for this continuation: 264 outline tests and three
focused offline pending-evidence tests passed; `poetry check`, compilation and
`git diff --check` passed (existing Poetry metadata warnings only). The 12
previously reproduced AP/KMR baseline failures remain separate. Cleanup of
attempts 05, 07 and 08 removed 24 disposable files and released 374,824,960
allocated bytes (357.46 MiB). Both retained Part slippage results loaded through
`load_offline_result` in temporary review copies, which were removed. The
official machine capture remains selectable, with its original observation and
native manifest hashes. See `retained_results_after_cleanup.json` and the
per-attempt cleanup manifests/receipts. No machine outline has nominal-
continuation proof yet; the retained one-event prefix is still incomplete.

Verification for the robot repair: 529 workflow tests passed; the endpoint and
environment suites recorded 672 passes and two invalid machine-state test
fixtures. Correcting those fixtures produced four passing focused reruns.
`poetry check`, compilation and `git diff --check` passed. The separately
reproduced 12 legacy AP/KMR failures remain recorded above.

Its diagnostics are under
`machine_outline_goal_20261014_01a137cc/legacy_forward_backward_20261014/attempt_01/`.
The first genuine outline request uses `candidate_events` and contains no
`gripper_reach`, `additional_bindings`, `max_xy_radius_m`, or
`execution_primitive_catalog`. It is 28,020 characters; the retained earlier
first outline prompt is 12,312 characters. Expanded factual context makes the
new prompt longer. Both prompts, lengths, replay summaries and test logs are
retained alongside the attempt. A complete accepted outline, nominal-continuation
proof and saved-result loading are still required. Recovery execution remains
unauthorized.

## Historical handoff — October 14, 2026 — user-requested machine-attempt cleanup

The user explicitly approved: “Remove the attempt files; keep the live failure
capture.” This supersedes the earlier instruction to preserve the incomplete
28-event prefix and its resumed-attempt dependencies. No complete accepted
Machining breakdown outline or primitive composition existed in this chain.

Retire `attempt_01` through `attempt_18`, their corresponding offline run
directories, the stopped `offline_4bed5f73eabd4fb6a6e9ef4ee76984e2` preparation,
and disposable machine-attempt replay and diagnostic outputs. The last attempt
`offline_6157db87c86841609d5d7895e6b7a0b8` ended at `NEEDS_CONTEXT` for M2
`machine_part`, with missing projected workholding/access evidence; a subsequent
owner calculation exposed missing `simulation_clock_sec`. The 28 accepted events
included 18 `move_to_location` events and did not establish nominal continuation.
Per-attempt failure findings, exact deletion paths, ownership and dependency
checks, protected hashes, and the cleanup receipt are recorded under
`cais_spade_llm/monitor/environment_runs/machine_cleanup_20261014_01a137cc/`.
Cleanup removed **3,431 files and 778 directories**, releasing
**117,611,106,304 allocated bytes (109.53 GiB)**. All 644 protected file hashes
matched afterwards. The official machine capture remains selectable and
`offline_ready`; its native model loaded with the original manifest hash.
Both retained Part slippage results passed the actual saved-result loader in
temporary review sessions, including the bound nominal-continuation proof.
Those temporary copies were removed. No API, Gazebo, or execution call occurred.

Preserve official capture `capture_8beafdb8a01e444894cfaf41d29921fd`, source
checkpoint `4a6d23fca53a4d4486ba50f9d005203a`, the capture's native model and
capture setup records, both safety specifications, all other official captures,
and accepted Part slippage results. Current code work and its before-change
snapshot remain under `machine_outline_goal_20261014_01a137cc/legacy_forward_backward_20261014/`.
Only compact original provider request/response records, the event audit, and
proposal inputs remain there for the already-authorized local regression and
prompt comparison; they are not a saved recovery result or resumable prefix.

Next: finish the authorized legacy task-level generation and backward/forward
feasibility repair, verify locally from the preserved capture and compact
proposal records, then generate a separate new attempt. The implementation is
in progress and has not yet passed its required regressions. Recovery execution
remains unauthorized. Historical sections below describe the now-retired files;
their old “retain” and “resume” instructions are superseded by this handoff.

Implementation checkpoint after cleanup: candidate generation, public-context
filtering, and version-2 policy plumbing are partially implemented. The current
edits also share `handling_robot_access` target calculations with nominal pickup,
correct the inherited cell capabilities in detached preparation, add a bounded
robot joint calculation when no explicit `gripper_reach` was recorded, and defer
nominal resource guards only for version-2 recovery effect projection. These
changes still need focused regression additions and integration review. Robot
projection, partial evidence handling, machine processing-time semantics,
candidate progress selection, and a successful new LLM outline remain unfinished.
No new API request has been made. `poetry check`, Python compilation and
`git diff --check` pass; Poetry reports its existing metadata deprecations.
The first sandboxed baseline was interrupted after 95 passes because a mocked
threaded observation test stalled; the two corresponding host tests pass.
The current host regression run finished with **1,689 passed and 23 failed**.
Eleven failures concern earlier expectations for public limits/reasons or the
single-event schema; twelve have the previously documented legacy AP/KMR error
signatures, but baseline equivalence has not yet been independently rerun for
this change. The full log and the interrupted baseline log are retained in the
active `legacy_forward_backward_20261014/` directory. Diagnose these failures
and finish the remaining implementation before any generation retry.

## Historical handoff — October 14, 2026 — capability context and additional calculations

The user requested an audit of the twenty-eight accepted events and their
granularity. `nominal_goal_20261014/attempt_18_event_review.json` and
`attempt_18_events.csv` retain the exact accepted records and source hashes.
Eighteen events are `move_to_location`; accepted indices 2–6, 9–13 and 19–25
revisit station locations. These are not all necessary arm lift or retreat
substeps, and accepted-event count must not be reported as recovery progress.
The actual turn-49 provider request explicitly exposes `move_to_location`,
`pick_approach`, `pick_part`, `place_approach` and `place_release` through
`evaluation_support.additional_bindings`; `machine_part` appears in the failure
context. The inspected request does not contain `primitive_support`,
`execution_primitive_catalog` or the checked private primitive names. The
registered event names therefore were not independently invented by the LLM.
`event_name` remains a free string, while owner support is derived from proposed
conditions. Prompt vocabulary and supported transition granularity both matter.

The event path appends a locally validated event before evaluating nominal
continuation; it does not require that event to improve completion or prevent a
return to an earlier state. Historical rationales alternate between clearing M2
and restoring nominal preconditions. The current replay already records
`environment_22` completed at accepted event 9 (`outline_recovery_18`); later
round trips must not be justified by claiming that task is still unfinished.
Task-level generation and repeated-state handling require review separately
from the physical-evidence repairs. Preserve exact identifiers and allow
legitimate revisits with changed custody, processing, configuration, monitor
state or new relevant evidence. Do not insert a reference recovery sequence
into generation. No events were deleted or reclassified by this audit; a
shortened sequence would require independent replay. No new API request or
execution occurred during this review.

The latest finished retry is `attempt_18`,
`offline_6157db87c86841609d5d7895e6b7a0b8`. It revalidated all twenty-seven
retained events unchanged and accepted the original turn 48 `place_release`,
retaining **twenty-eight** accepted LLM events. One new API request followed.
Turn 49 proposes M2 `machine_part` for `KET4_Square_4mm`, with `job_state=completed`
and `resource_state=completed`. RA returns `NEEDS_CONTEXT`; its primitive
capability checks identify missing `workholding and access evidence for the
projected scene at M2`. No physical projection or CCA request occurs. The attempt
finished at `2026-10-14T12:12:30.242785+00:00`, source unchanged and zero
execution-interface calls. Nominal continuation remains `not_ready`.

The recorded successor has `KET4_Square_4mm` supported at the M2 workholding
pose and KMR's release/retreat endpoint. An in-progress machine-owner repair
calculates workholding and access evidence from these projected facts. The local
saved-state calculation reports `correct_part_present=true` and
`robot_access_clear=true`; its conservative KMR and ur5e-2 clearances are
0.3809782632931921 m and 1.142229512050384 m. This is conditional model evidence,
not execution evidence or acceptance of turn 49. The subsequent local
`start_job` calculation returns `NEEDS_CONTEXT: observed simulation_clock_sec`.
Both results are retained under `nominal_goal_20261014/machine_access/attempt_18/`.
Finish review and regression of the calculation and processing-time semantics,
then replay the unchanged provider turn before another API request. Do not treat
missing evidence as a selected safety-specification violation or change either
requirement. Attempt 18's redundant `input_preview.json` was removed after the
recorded deletion and ownership checks, releasing 17,774,295 bytes; retained
capture, session and preparation checks pass.

The preceding retry is `attempt_17`,
`offline_841f1a8e3ad64c5fb5ed7763d7e615eb`. It reused twenty-five accepted events,
revalidated original accepted turn 45 and accepted original pending turn 46,
retaining **twenty-seven** accepted LLM events. Two new API requests followed.
Turn 47 requests `part_state=ready` on KMR `place_release`; the owner correctly
rejects it because the supported effect is `part_state=loaded`. Turn 48 makes
that revision and passes RA, but physical validation returns `NEEDS_CONTEXT`:
`KMR release has no supported deposited-part rest: Static support requires at least three distinct contact points`.
CCA is `not_checked`, with no CCA request. The attempt finished at
`2026-10-14T11:31:44.477822+00:00`, source unchanged and zero execution-interface
calls. Nominal continuation remains `not_ready`; retain the original response.

The earlier supplied support diagnosis now matches this actual LLM proposal.
The initial M2 geometry retirement occurred at accepted event 8, original turn 19
(`outline_recovery_17`); M1 geometry was similarly retired at accepted event 18,
turn 34 (`outline_recovery_33`). The shared `pick_part` updates custody while
the recorded machine body remains static. The policy now preserves that body
only for a static peer whose declaration names the acting resource, has no
collection effects, and updates only `inventory.*`, `resource_state` or
`part_name`. Geometric updates and a different actor still invalidate geometry.
KMR placement also no longer adds an unrecorded 0.003 m offset above
`workholding_pose`: both workholding and slots retain their exact destination
part pose. The separate resting-contact calculation remains required, without
lowering its contact count or margins. This also changes targets used by the
shared KMR implementation; native execution has not been performed.

All **27 focused tests** pass across `Storage`, `M1` and `M2`, including custody,
static-body, changed-actor and geometric-update cases. The full environment,
KMR-evidence and machine regression passes **325 tests**; records are under
`nominal_goal_20261014/machine_placement/attempt_17/`. Poetry, compilation, UI
help, fifteen local runner tests and `git diff --check` also pass. Saved source and accepted
events remain immutable. Replay the original provider sequence and turn 48
locally under the repaired projection before another API request. The local
runner now refuses generation if a retained proposal remains `NEEDS_CONTEXT`.
The earlier seven-event checkpoint has a different logical session identity;
do not relabel it to bypass resource-evidence restoration checks.

Attempt 17's redundant preparation preview was removed after its explicit
deletion list and ownership, mtime, hash, process and open-handle checks,
releasing **17,774,295 bytes**. The accepted session and prepared state are
unchanged, and the official capture still loads with its native model and is
selectable for offline preparation. The completed full replay in `attempt_18`
preserved all twenty-seven accepted events and validated original turn 48 before
the single new generation request. Protect its current partial result.
This is model-based planning validation; no execution is authorized.
Its unchanged original turns 19 and 34 revalidated through RA,
physical projection and CCA with `M2` and `M1` static geometry retained.
`nominal_goal_20261014/machine_placement/attempt_18_projection_comparison.json`
records both before/after validation hashes and confirms identical LLM proposals.
The complete attempt additionally validates placement, but nominal continuation
remains unestablished.

An independent retained-record audit verifies all twenty-seven events in
`attempt_17`: each matches exactly one original provider response, RA is
`FEASIBLE`, physical evidence is `projected`, CCA is `passed`, both selected
specification hashes match, and CCA monitor history is continuous. Its bound
nominal proof loads and remains `not_ready`. This audits the recorded result;
the separate `attempt_18` replay establishes validation under the changed code.
The event table and hashes are in
`nominal_goal_20261014/machine_placement/attempt_17/retained_event_audit/`.
`nominal_goal_20261014/audit_outline.py` can require complete acceptance before
final reporting. The adjacent `recovery_ui.py` and `verify_browser.py` prepare
the real Recovery UI load/reopen check with generation and execution blocked;
they compile but have not yet run for a complete Machining breakdown result.

The preceding retry is `attempt_16`,
`offline_47c47d53a8824405b9ff62fbab018c04`. It retained **twenty-six** accepted
events through turn 45 and made one new API request. Original turn 46 proposes
KMR `place_approach` with `resource_state=positioned`; RA is `FEASIBLE`, physical
validation returns `NEEDS_CONTEXT` (`Projected starting configuration is stale`),
and CCA is `not_checked` with no CCA request. The attempt finished at
`2026-10-14T11:09:02.353459+00:00`; source observations are unchanged and no
execution interface was called. Nominal continuation remains `not_ready`.

Diagnosis: M2 `select_program` changes global projection provenance but does not
move KMR. The zero-duration machine-control projection now preserves a peer
endpoint only when its input binding, state and configuration are valid and its
state, resource model, geometry and all part states remain unchanged. The
successor retains the preceding endpoint fingerprint and binding. Stale evidence
is never renewed; execution readiness remains unestablished. This applies to
unchanged peers without selecting a resource or route. Sixteen machine-control
cases and fifteen local runner dependency tests pass. Original turns 45 and 46
both pass RA, physical projection and CCA locally, producing **twenty-seven**
events without changing source observations or the first twenty-five events.
This local result is not yet a committed UI attempt. Records are under
`nominal_goal_20261014/unchanged_peer_endpoint/attempt_16/`.

The local runner can reuse a reviewed shorter accepted prefix, then revalidate
the affected accepted tail and original pending response. For the next retry,
reuse the twenty-five-event state from attempt 15, revalidate original accepted
turn 45, replay original turn 46, and only then request a new event. The source
attempts remain immutable. All 356 protected source and accepted-result files
matched their retained hashes before this repair.

Attempt 16's single redundant `input_preview.json` was removed after explicit
list, ownership, modification-time, hash, process and open-handle checks:
**17,774,295 bytes** released. Its accepted session and preparation still match
their hashes, and the official capture remains selectable and compatible with
its native model. The dependency review for retry 17 is prepared under
`unchanged_peer_endpoint/attempt_16/accepted_evidence_reuse_review.json`; use
`--resume attempt_16 --reuse-prefix attempt_15` with that review. Retry 17's
finished result is recorded above. The unchanged-tail
replay is local verification only. The full machine/environment regression passes:
**259 passed**, in `unchanged_peer_endpoint/attempt_16/regression_tests.log`.
Focused tests, Poetry, compilation, UI help and `git diff --check` also pass.

The preceding retry is `attempt_15`,
`offline_b9bec2a6d9b5438186772676b47071d1`. It retains **twenty-five** committed
accepted LLM events through turn 43. Checked reuse preserved twenty-three events,
and unchanged provider turn 42 passed before generation continued. Three new
provider responses are retained: turn 43 moves KMR to M2, turn 44 is rejected by
the resource because the recorded part pose differs from the requested
workholding location, and turn 45 requests M2 `select_program` from `circle` to
`square`. Turn 45 initially returned `NEEDS_CONTEXT` for missing
`simulation_clock_sec`. It remains uncommitted in this saved attempt. Nominal
continuation is `not_ready`; no recovery execution occurred. The attempt finished
at `2026-10-14T10:35:09.697966+00:00`, with source observations unchanged and zero
execution-interface calls. Retain its provider records, accepted state, evidence
and dependencies for the next retry.

The clock requirement now permits only `select_program` without a current clock
when every retained job is established inactive. This changes no job time,
progress or completion. Instantaneous simulated machine controls can preserve
bound endpoint occupancy under evidence policy 2 when unrelated complete geometry
is unavailable; stale occupancy is rejected and unknown occupancy remains unknown.
The previous complete-geometry path remains unchanged. Replaying the initial
accepted M2 `interrupt_job` produced exactly the same physical projections before
and after this repair. These changes apply to recorded machine declarations,
without a preferred resource or sequence. Full machine, environment, validated
outline and public-context suites pass: **821 passed**. Poetry, compilation and
`git diff --check` pass, with existing Poetry metadata warnings.

Physical `needs_context` now consistently stops before a CCA request and records
CCA `not_checked`, with the resource owning the finding. A physical guard mismatch
is not a safety-specification rejection. Complete physical evidence still goes
through the unchanged CCA safety checks. Both selected specifications are retained.
Repair records and unchanged-turn replay are in
`nominal_goal_20261014/processing_clock/attempt_15/`. A successful local replay is
not a newly committed accepted event or a completed recovery outline. Resume the
original turn 45 through checked dependency reuse, then continue generation.

Final unchanged-turn-45 replay passes RA `FEASIBLE`, physical `projected` and CCA
`passed`, producing a local twenty-six-event prefix with nominal continuation
still `not_ready`. Its original provider response, source observations and prior
twenty-five events are unchanged. The reviewed retry is `attempt_16`,
`offline_47c47d53a8824405b9ff62fbab018c04`; its later result is recorded above.
The exact accepted-evidence review and dependency hashes are beside the rule-change
record in `processing_clock/attempt_15/`. Attempt 15's redundant
`input_preview.json` was removed after ownership, mtime, hash, process and
open-handle checks, releasing **17,774,295 bytes**. Its accepted session and
preparation are unchanged; the official capture remains selectable, compatible,
and loadable with its native model. The local runner keeps compact turn summaries
while retaining independent full per-turn validation artifacts. Eight local
reuse/compaction tests pass; no production acceptance rule is changed by this
storage adjustment.

The preceding retry is `attempt_14`,
`offline_ac96e80780a149c5b657035473deaf15`. Checked reuse restored twenty-two
accepted events, and original provider turn 41 passed RA, physical projection
and CCA, bringing the committed prefix to **twenty-three**. One new API call,
turn 42, again proposed moving the carried part from M2 to M1. The owned process
was stopped with SIGINT before further API requests; it exited 130 and finished
writing at `2026-10-14T10:18:11.738728+00:00`. Turn 42 passed privately but remains
uncommitted in the authoritative saved session. Retain it for replay. Nominal
continuation is `not_ready`, its bound proof loads, source observations are
unchanged, and zero execution interfaces were called. Preserve this partial
attempt and its source dependencies.

The earlier `attempt_13`,
`offline_7290bd57e4be4b46a9438634b8887ab8`. It retains **twenty-two** committed
accepted actual LLM events through turn 40 and uncommitted provider turn 41.
Two new API calls were made. Turn 40 returns KMR carrying `KET4_Square_4mm`
from M2 to M1; turn 41 proposes returning to M2. The owned process was stopped
with SIGINT to prevent further repeated transport. Its in-flight worker
finished checking turn 41 privately during shutdown; the published saved
session remains at turn 40. The process exited with code 130 and finished
writing its records at `2026-10-14T09:47:53.266365+00:00`. Its stale
`generating_outline` metadata is not evidence of an active process. Nominal
continuation remains `not_ready`. Source observations and the twenty-one-event
source prefix are unchanged, and no execution interface was called. Preserve
this source attempt and its dependencies. Turn 41 was subsequently replayed
in attempt 14 with the precise historical rejection feedback retained.

The exact destination was omitted when deciding whether existing KMR placement
declarations supplied an additional binding. Declarations for M1/M2 must not
suppress evaluation of the proposed `Storage` binding. The corrected selection
also distinguishes a carried part supplied as movement context from pickup.
All 20 focused additional-binding tests pass. The diagnosed missing `Storage`
receiving contract is now implemented as an owner calculation over the exact
recorded slot and current empty inventory. Its detached binding uses the existing
KMR placement program and shared target calculator. Recorded slots use their
exact contact pose; the existing workholding clearance remains unchanged.

Static inventory-only participation no longer discards the peer body's geometry.
Rigid collision shapes and static supports are retained separately from expired
contacts, velocities and attachments. Reuse checks geometry, part identity,
meshes and policy fingerprints; changed support bodies are excluded. This does
not restore expired observations or establish live tracking/path collision safety.
The isolated supplied `Storage` approach and release both pass RA, physical
projection and CCA. Neither is an accepted LLM event or a suggested provider event.

Unchanged provider turn 26 still fails its known guard: `resource_state=carrying`
instead of the owner's `positioned`. A complete declaration search with only
demonstrated guard violations now produces a resource rejection and permits an
LLM revision; incomplete evidence and bounded search retain `NEEDS_CONTEXT`.
Public feedback names exact public conditions and source fingerprints. CCA is
`formal_safety_status=not_checked`, with no safety-rule violation claimed.
Replay, before/after prompts, source copies, supplied verification and the detailed
rule-change record are in `nominal_goal_20261014/storage_placement/`.
The unchanged response returns `need_revision`; prompt lengths are 26,545 and
30,144 characters, including restored production safety context and precise guard
feedback. All 356 protected-file hashes are unchanged. `attempt_12`,
`offline_9142187ce2d24273991edf01a95799ca`, revalidated the thirteen-event prefix
and returned the unchanged turn 26 for revision. Actual new LLM turns 27
(`place_approach`) and 28 (`place_release`) at `Storage` passed RA, physical
evidence and CCA. Turn 30 moves KMR to M1, turn 32 establishes the pickup
approach, and turn 34 picks `KET4_Square_4mm` after exact `part_location`
feedback. Turn 35 moves the carried part to M2; turns 37 and 38 return to M1
and M2, bringing the accepted prefix to twenty-one events. The attempt is now
stopped and nominal continuation remains `not_ready`. Preserve its output. No
complete machine outline or execution has been established.
The final environment-recovery, validated-outline and public-context regression
run passes **712 tests**. The KMR and physical-evidence suites also passed apart
from a subsequently corrected missing-fixture classification test; its focused
rerun and all 15 projected-contact cases pass. Poetry, compilation and whitespace
checks pass. Poetry retains its existing metadata deprecation warnings. This
validation used local calculations; no Gazebo execution was performed.
Four additional guard-classification cases pass: demonstrated guard failure,
missing evidence, incomplete model and bounded-search exhaustion.

The event transaction now avoids sending typed upstream owner/physical failures
to CCA solely to echo them. It retains the same physical verdict and marks the
CCA stage `not_checked`, `request_sent=false`, with no monitor change. Passing
physical evidence still requires actual selected-specification validation.
The unchanged turn 26 replay makes zero CCA requests and preserves accepted
state and source observations. All 243 validated-outline regression cases pass,
including resource attribution, incomplete-evidence handling and actual CCA
calls for valid physical evidence. Poetry, compilation and whitespace checks
pass. The sandbox test process stalled after 19 cases and was terminated; the
complete suite passes with normal loopback access. No baseline test failures
were identified. Evidence is in `nominal_goal_20261014/cca_upstream_guard/`.
The stopped `attempt_12` loaded the earlier code; its records remain protected.

Turn 36 exposed a feedback omission: RA's existing `condition_support` showed
individually supported conditions with an unsupported combination, but the
handler retained only an empty `mismatches` list. The handler now preserves
`condition_support` through the existing filtered prompt renderer. Acceptance
rules and candidate schema are unchanged; no replacement event is supplied.
The retained turn-36 feedback replay and excerpts are in
`nominal_goal_20261014/condition_feedback/`. Its new end-to-end regression passes;
the broader RA/outline run reports **458 passed, 12 failed**, with all 12 failure
identifiers exactly matching the retained pre-change baseline (five legacy AP
fixtures, seven KMR execution-provider cases). Poetry, compilation and whitespace
checks pass. Turn 37 passed all three stages, but was a return to M1 rather
than a corrected placement proposal. Turn 38 returned to M2. The unchanged
retained turn 39 still returns `need_revision` at RA: its conditions are
individually supported but their combination is not. Physical validation and
CCA are not reached; the twenty-one accepted events remain unchanged.

The prompt now shows changed public conditions from the six most recent
accepted events, making their destinations visible, and retains exact public
`domain` and `domain_complete` declarations. Older events remain compact.
Private fields, unchanged conditions and primitive sequences remain excluded;
candidate-schema expressiveness and acceptance rules are unchanged. All
**556** validated-outline and public-context tests pass, including equal
treatment of robot, machine and conveyor declarations. Poetry, compilation and
whitespace checks pass with existing Poetry metadata warnings. Retained proof
loading and all twenty-one per-event RA/physical/CCA records were verified;
the official machine capture remains selectable and compatible.
The exact retained-turn prompt comparison grows from 30,112 to 34,079
characters and contains the shared capability instruction once. Attempt 12's
redundant `input_preview.json` was removed after ownership, hash, mtime,
process and open-handle checks, releasing 17,774,295 bytes. The accepted
prefix, nominal evidence and selectable source capture still load; its
uncommitted provider response is retained.

`attempt_13` revalidated all twenty-one source events, followed by original
turn 39 returning `need_revision` at RA. New provider turn 40 passed all three
stages but repeated transport. Its twenty-two-event prefix and bound nominal
proof still load, and the official machine capture remains selectable and
compatible. PID 151480 has exited. This is not a complete accepted outline.

Further diagnosis found that a rejected combination was being interpreted as
excluding its entire operation, and acceptance cleared the current rejection
feedback. `nominal_goal_20261014/condition_pairs/` retains the repair: report
proven incompatible pairs of proposed public conditions, clarify the rejection
scope, and retain two compact prior rejections tied to their recorded starting
states. No replacement event, route, private primitive sequence or unrestricted
validator prose is supplied. Pair checks are bounded diagnostics; omitted or
incomplete pairs do not prove rejection. Acceptance rules and candidate-schema
expressiveness are unchanged. Replaying original turn 39 still returns
`need_revision`, now identifying incompatible pairs involving `approached_part`;
the source and twenty-one-event prefix remain unchanged. Six focused cases
pass; the broader run reports **772 passed, 12 failed**, with every failure
identifier matching the retained pre-change baseline. Final prompt integration
passes **558** tests after restoring compact rejection summaries at the
prompt-input boundary. Poetry, compilation and whitespace checks pass with
existing Poetry metadata warnings. Attempt 13's redundant preparation preview
was removed after ownership, hash, mtime, process and open-handle checks,
releasing 17,774,295 bytes; accepted state and provider turn 41 are retained.

The local `offline_attempt.py` runner now has an explicit `--reuse-review` path
for the reviewed feedback-only changes. It checks all 344 runtime source hashes,
the exact retained session and preparation, and fifteen preparation dependency
fingerprints against a fresh Recovery UI preparation. It also verifies provider
authorship, every retained RA/physical/CCA acceptance record, the bound nominal
proof, and resource-owned calculation restoration. Any changed dependency stops
reuse. Original validation artifacts are copied and checked; only the retained
uncommitted proposal is revalidated. This does not change an acceptance rule.
Six local positive/negative reuse checks and eighteen existing restoration tests
pass. Poetry, compilation, UI help and whitespace checks pass. The new path is
being exercised in `attempt_14`, `offline_ac96e80780a149c5b657035473deaf15`;
its twenty-two-event prefix was restored and turn 41 passed revalidation.
The earlier full replay took about 23 minutes in attempt 13.
Do not regenerate accepted events or retained uncommitted provider responses. Keep
historical rejection summaries when continuing so an intervening accepted move
does not hide the earlier failure again.
`condition_pairs/accepted_evidence_reuse_review.json` records the exact saved
session/preparation hashes, current runtime source hashes, and the three
runtime files changed after attempt 13 started. Their reviewed diff is retained
as `feedback_only_changes.diff`: successful validation paths are unchanged.
This supports checking reuse; it does not replace fresh preparation, nominal
proof binding, owner-evidence restoration or provider-authorship verification.
The final original-turn-39 replay and prompt lengths are retained in
`condition_pairs/prompt_comparison.json`. Post-cleanup loading again confirms
all twenty-two committed events and the compatible selectable source capture.

The turn-42 prompt already showed M1 as `unavailable`, but its proposal rationale
treated the original resource assignment of `environment_21` as the reason to
return there. The prompt now describes the existing matching product-completion
rule without naming a resource or route: recorded assignment alone is not a
goal, accepted matching completion through another resource may satisfy product
work, and explicit resource goals remain required. Original processing records
are labelled capture-time observations. This changes presentation only; no
validator, schema, model or acceptance rule changed. All **560** public-context
and validated-outline tests pass, along with Poetry, compilation and whitespace
checks. `condition_pairs/nominal_assignment_context/` retains before/after
prompts (37,234 / 37,564 characters), exact source hashes, the one-file review,
and a checked-reuse dependency record for attempt 14. The capability instruction
appears once. Attempt 14's redundant `input_preview.json` was removed after
ownership, mtime, hash, process-exit and open-handle checks, releasing 17,774,295
bytes. Source observations, all accepted events and turn 42 remain retained.

While attempt 13 revalidates the prefix, an isolated supplied placement check
uses attempt 12's retained state without API calls or execution. Its
`place_approach` passes RA, physical projection and CCA. Its `place_release`
passes RA but physical projection returns `needs_context`: no supported resting
contact is established. These cases never enter an accepted prefix or provider
prompt. `nominal_goal_20261014/machine_placement/supplied_summary.json` retains
the stage outcomes. The separate `support_diagnosis.json` varies only two
supplied inputs: retained static M2 geometry and the existing 0.003 m release
clearance. Current geometry fails at both heights; supplying unchanged M2
geometry still fails with the clearance; the exact recorded contact pose with
that geometry establishes support. This diagnoses separate static-evidence
retention and release/settling limitations, not an accepted recovery or an
authorized change to physical limits. No production code was changed for these
cases. One overly large diagnostic load was killed with exit 137; the reduced
calculation completed and the owned recovery retry remained live.
Its complete supplied validation record is retained losslessly as
`machine_placement/supplied_case.json.gz`. The decompressed SHA-256 matches
the original. After ownership, mtime, hash and open-handle checks, only the
redundant uncompressed copy was removed, releasing a net 430,779,586 bytes.
The scripts, compact stage summary and independent geometry diagnostic remain.

Read-only completion-boundary inspection is retained in
`nominal_goal_20261014/completion_boundary_diagnosis.json`. The captured
`goal_part_location` remains `assembly_board-v1` for `KET4_Square_4mm`; completing
machining alone cannot establish the requested proof. Matching supported product
completion by another owner is already recognized. Machine command clock gaps
and the current absence of robot movement endpoints outside KMR are prospective
limitations, not newly rejected LLM proposals. They have not been bypassed.

Attempt 11's redundant preparation preview was removed after ownership, hash,
mtime, process and open-handle checks, releasing 17,774,295 bytes. Its thirteen
events, nominal proof and source capture still load; the original eleven events
are unchanged. The official machine capture remains the sole selectable capture
with the same native manifest. Preserve all accepted-prefix and provider records.

The diagnostic resumer also rebuilt the seed and omitted the existing production
`public_safety_context_version=1`. It now uses the prepared production seed.
The detached prompt replay shows `resource_state=carrying` and
`resource_location=Storage` with their sources, without stale undeclared
`current_state=idle` and `current_location=M2` aliases. Captured declarations,
accepted events and projected state are unchanged. Prompt lengths are 26,545
and 29,008 characters; the increase includes the existing safety context.
Evidence is in `nominal_goal_20261014/public_seed/`.

Machine-owned pickup calculations now support an exact recorded `interrupted`
job with no active/running job, retaining the machine's configuration, job and
source bindings. They do not establish geometric access or completed processing.
The unchanged earlier M1 proposal now reaches its independent contradictory
end-state finding: proposed `resource_state=idle`, owner effect `at_pick`.
A separately labelled supplied verification case with the declared value passes
RA, physical projection and CCA; it is not an accepted LLM event and was not
supplied to the provider. All 264 machine-job, environment-recovery and KMR
evidence regressions pass, as do Poetry, compilation and whitespace checks.
Evidence and source copies are in `nominal_goal_20261014/interrupted_pickup/`.

The preceding retry is `attempt_10`,
`offline_af826168b57547b98c7010ce7939f88f`. Its saved session retains **eleven**
accepted LLM events. Original turn 21 passes after the movement parameter-context
repair. New turn 22 requests release during a supported movement and is correctly
rejected for contradictory owner effects; this is not a safety violation. Turn 23
adds another loaded movement to `Storage`. Three new API responses were received,
with zero forbidden calls and unchanged source observations. The owned runner was
interrupted after diagnosing repeated movement; it exited 130 and is no longer
active. Turn 24 has a retained provider response but was not committed to the
saved session; revalidate that exact proposal after the eleven events. Do not
count it as accepted or ask the provider to regenerate it.

Nominal checking required `environment_22` again even though accepted turn 20
already fulfilled its exact `move_to_resource` declaration. The old replacement
check only considered product completion. Accepted recovery events now retain
exact nominal-task completion evidence after RA, physical projection and CCA.
The existing owner-program witness checks every declared effect and parameter;
endpoint state alone is insufficient. Equivalent validated owner alternatives
and the same detached calculation model remain available to that check. Records
are bound to the attempt, accepted prefix, source, owner context, nominal task
and safety specifications; recorded dependencies and earlier owner tasks must
already be fulfilled. This rule applies across resources without selecting a
recovery route or changing either specification.

The saved turn-20 replay now retires only `environment_22`; `environment_21`
and `environment_23` still block nominal continuation. Source observations,
accepted events, physical state and monitor history are unchanged. Prompt length
falls from 26,855 to 26,347 characters by removing the obsolete finding.
All **571 tests** in the validated-outline, request-record and offline-recovery
suites pass; Poetry, compilation and whitespace checks pass. Full evidence,
before/after prompts, rule-change record and source backups are under
`nominal_goal_20261014/nominal_task_completion/`. Next: resume the retained
uncommitted proposal with this correction, preserving the eleven-event prefix.
No complete machine recovery or execution is established.

Attempt 10's duplicate preparation preview was removed using its explicit
ownership/hash/mtime/open-handle manifest, releasing 17,774,295 bytes. Its eleven
saved events and nominal proof still load, the original nine events are unchanged,
and the sole official machine capture remains selectable with the same native
manifest. All 356 protected-file hashes are unchanged. `attempt_11` resumed this
prefix and the uncommitted provider response as documented above.

The preceding finished retry is `attempt_09`,
`offline_48f1383bd8d743419b225f80339aa656`. It retains **nine** accepted actual
LLM events. The provider corrected turn 18 in turn 19 (`outline_recovery_17`,
KMR `pick_part`, with `part_location=KMR` and `part_state=in_gripper`) and then
authored turn 20 (`outline_recovery_18`, loaded movement to `Storage`). Both
pass RA, physical projection and CCA. Three new API calls were made, with zero
execution calls and unchanged source observations. Nominal continuation remains
`not_ready`; the only saved pending nodes are `environment_21`, `environment_22`
and `environment_23`. The first and third require the unfinished machining work;
moving KMR between previously visited locations alone does not establish it.

Turn 21 proposes loaded movement back to M2 with `part_name` supplied. Both
supported owner alternatives failed the private endpoint parameter lookup:
the proposal's part context was compared as an extra argument to movement
declarations that do not declare that argument. The lookup now distinguishes
that exact contextual field while retaining all actual declaration arguments,
part identities, ambiguity checks and calculation evidence. The unchanged
proposal passes RA, physical projection and CCA locally. The KMR suite passes
49 tests, including five new context/argument-conflict cases. Poetry, compilation
and whitespace checks pass. Evidence and before-source copies are in
`nominal_goal_20261014/movement_part_binding/`. Resume the retained pending
proposal; preserve attempt 09 and its attempt-08 dependencies. No complete
machine outline or execution has been established.

The redundant attempt-09 preparation preview was removed after the explicit
ownership/hash/open-handle checks, releasing 17,774,295 bytes. Its nine-event
prefix and nominal proof still load, its original seven events are unchanged,
and the official machine capture remains the sole selectable capture with the
same native manifest. `attempt_10` is now stopped as documented above. The replay
runner uses the existing compact-history
option while retaining every full per-turn/provider record; RA, physical, CCA
and final-prefix nominal checks are unchanged.

The preceding retry is `attempt_08`,
`offline_74cca5c0ae1c48d286fcce007e8f3e06`. It revalidated all six retained
events and accepted the original turn-17 `pick_approach`, leaving **seven**
accepted LLM events. One new API request proposed turn-18 `pick_part` for
`RGOCG4-50_Round_4mm`. Nominal continuation remains `not_ready`; no complete
machine outline or execution is established. Preserve attempt 08 and its
attempt-07 source dependencies.

Diagnosis and local repair are retained in
`nominal_goal_20261014/custody_endpoint/`. Additional M2 bindings were present
in accepted evidence but the next validation rebuilt only the captured model.
Reuse now validates their attempt/input/model fingerprints and binds the retained
calculation model to the new proposal through RA, physical projection and CCA.
The unchanged turn-18 proposal is correctly rejected: it requests
`part_state=held` and `part_location=recovery-resource-8@localhost`, while the
declared effects are `in_gripper` and `KMR`. Feedback reports those exact public
conditions with `formal_safety_status=not_checked`; this is not a safety-specification
failure. Its replay returns `need_revision` without changing the seven events,
projected state, monitor history or source observations.

The existing KMR custody contract now supports conditional attachment/release
endpoints. It retains exact part identity, transform, gripper joint bounds,
attachment effects and deposited-part support. Interval part occupancy remains
unknown across custody changes. Endpoint evidence does not establish finger
contact dynamics, collision-free motion, continuous support departure or execution.
The numerical endpoint solver uses bounded `dogbox` and recorded seeds, with
independent FK and unchanged acceptance tolerances; a supplied lift case now
converges where the former bounded search was unresolved. A separately labelled
supplied pickup variant passes RA, physical projection and both CCA specifications;
it is not an LLM-generated accepted event. The original LLM response remains
rejected and must be revised by the provider. The focused final suite passes
29 cases; Poetry, compilation and whitespace checks pass. Full regression
results and the precise original/replacement rules are retained with the repair.
The duplicate attempt-08 preparation preview was removed using the explicit
ownership/hash manifest in `attempt_08_preview_deletion_list.json`, releasing
17,774,295 bytes. Its seven accepted events and `not_ready` nominal proof still
load; the sole machine capture and native model remain available. All 356
protected JSON files are unchanged. The unchanged-proposal prompt changes from
25,412 to 25,909 characters. An early regression exposed the legacy singular
`owner_effect_result.field` field, which is preserved alongside the new conflict
list; all eight affected rechecks pass. The sandbox suite stalled at the offline
workflow test and was stopped through its owned session; all **456** tests pass
with local socket access in `regressions_host.log` (255.81 seconds). Attempt 09
subsequently continued this seven-event prefix as recorded above.

The preceding API retry is `attempt_07`,
`offline_1eda276787cb418bb0de6fe0cf04258b`. It retains **six** accepted actual
LLM events: the original four plus turn 15 KMR movement M2 → M1 and turn 16
KMR movement M1 → M2. The two new movements do not complete recovery; nominal
continuation remains `not_ready`. Four new API calls were made, no execution
interface was invoked, and the source observations are unchanged. Turn 14
correctly bound `part_name` but failed RA's recorded ur5e-1 `gripper_reach`.
Turn 17 proposed KMR `pick_approach` for `RGOCG4-50_Round_4mm` at M2; it stopped
at RA `NEEDS_CONTEXT` because location-specific pickup calibration was absent.

The unchanged turn-17 proposal now passes real RA → physical endpoint → CCA
locally. The owner preserves complete recorded calibration and can additionally
calculate targets from exact workholding/docking geometry and its recorded
general pickup settings. Incomplete calibration is not replaced. Bounded inverse
kinematics supplies candidate joint configurations; independent owner FK and
unchanged joint limits verify each witness. Target calculations and an already
open, empty gripper have explicit endpoint effects. This is endpoint evidence,
not collision-path proof or execution. Native pickup calculations now resolve
the same exact origin identifier. The regression run passed 443 tests; three additional gripper-configuration
cases also pass. The unchanged proposal still passes after binding the gripper
endpoint to its recorded joint configuration, independently of its open label.
Poetry, compilation, CLI import and diff checks pass; all 356 protected JSON
files are unchanged. One redundant attempt-07 preview was removed after its
explicit ownership/hash audit, releasing 17,774,295 bytes; the retained six-event
prefix, loadable nominal proof, native model and sole machine capture were
verified afterward. Evidence and the precise rule change are in
`nominal_goal_20261014/additional_pickup/`. The diagnostic supplied endpoint cases
in `nominal_goal_20261014/pickup_geometry_diagnostic.json` are not LLM-generated
recovery, custody, or path proofs. Next, resume attempt 07 and revalidate its
unchanged pending proposal before further API requests. Keep attempt 07 and its
source attempt 06; no complete accepted machine outline exists yet.

The turn-13 missing-binding feedback repair is implemented. RA still returns
`NEEDS_CONTEXT`, but the public finding now names the missing `part_name` and
allows one LLM revision for the same resource, expected conditions, descriptor,
and snapshot. Renaming an event does not reset that bound. Private parameter
references and replacement events remain excluded. The unchanged original
proposal was replayed locally with API and execution interfaces blocked: it
returns `need_revision`, preserves all four accepted events, symbolic states,
monitor context and physical evidence, and leaves source observations unchanged.
The prompt changes from 24,690 to 24,806 characters (1,830 to 1,842 words).
Five focused cases and all 515 validated-outline/context tests pass. Poetry,
compilation, and diff checks pass with existing metadata deprecations.
Evidence, before/after prompts, and the exact rule change are in `nominal_goal_20261014/parameter_feedback/`. This changes generation
feedback only; acceptance requirements and both specifications remain unchanged.

The active nominal-continuation goal repaired the turn-11 endpoint ambiguity.
Policy 2 independently calculates every supported owner declaration and permits
multiple declarations only when their complete calculated consequences agree;
it retains each declaration and endpoint proof. Different or incomplete results
stay `NEEDS_CONTEXT`, and policy 1 remains unchanged. Evidence validation gaps
explicitly report `formal_safety_status=not_checked`. The unchanged saved turn-11
proposal passes RA, physical projection, and both CCA specifications. It is now
retained in `attempt_06`, `offline_bff94739bcd14967835bee4eae750ac8`, together with
the unchanged original two-event prefix and a fourth actual LLM-generated event
at turn 12: KMR `move_to_location` to M2. Both specifications and source
observations remain unchanged. Nominal continuation is still `not_ready`.
Evidence and the precise rule change are under
`machine_outline_goal_20261014_01a137cc/nominal_goal_20261014/`.
The supplied regression case with differing endpoints remains unresolved; the
initial test fixture incorrectly requested prohibited joint motion and was
corrected to explicitly supply a configuration that supports that operation.
Its recheck passed; the other eight focused cases passed on the first run.
The full environment-recovery and validated-outline suites passed **329 tests**;
`poetry check`, Python compilation, and `git diff --check` passed, with the
existing Poetry metadata deprecations. All 356 protected JSON files are unchanged.

Attempt 06 made two new API calls and stopped at turn 13. The LLM proposed
ur5e-1 `move_to_location` with `part_name=null` and the sole end condition
`resource_location=M1`. RA returned `NEEDS_CONTEXT`; physical projection and CCA
were not reached. The successor union is nonempty, despite the generic failure
message: it includes several declared handling operations, whose required
`part_name` and dependent parameter evidence are missing. Do not invent the
part or change this candidate into an accepted result. The next repair should
expose the missing public proposal binding precisely and allow a bounded LLM
revision while retaining the unresolved verdict. Existing M1 `interrupted`
pickup guards, configured UR5e `gripper_reach`, and additional pickup endpoint
evidence remain separate limitations. No complete machine outline or execution
has been established. Preserve attempt 06 as the current partial attempt and
attempt 05 as a source dependency. The generation process exited, with zero
forbidden execution calls. Diagnostic duplicates are listed in
`nominal_goal_20261014/diagnostic_deletion_list.json`; their removal released
28,978,495 bytes while the full independent two-declaration validation remains.
The redundant attempt-06 input preview is separately listed in
`attempt_06_preview_deletion_list.json` and `attempt_06_preview_cleanup.json`.
Together these remove three files and release 46,752,790 bytes (44.6 MiB).
`attempt_06/retained_verification.json` confirms four independently accepted
events, the unchanged original prefix, a loadable `not_ready` nominal proof,
and the sole selectable official machine failure capture with its native model.

The capability-information repair is implemented across resource types. The
complete resource index retains exact recorded identifiers, declarations,
conflicts, limits, sources, and implemented evaluation support. Detailed context
includes every resource whose recorded `reachability` or `reachable_locations`
intersects relevant locations; automatic inclusion is separate from requested
retrieval. Historical execution evidence is scoped to its actual recorded
operation, destination, scene, part, and configuration. It does not validate a
new proposal. Machines, Conveyor, Storage, and robots retain their resource-owned
checks; additional support is advertised only where implemented.

KMR can calculate additional movement, pickup, and placement bindings using
exact recorded geometry and peer access/custody conditions in a detached model.
The same model and bound evidence pass through RA, physical projection, and
CCA, with attempt and dependency checks on reuse/reload. Missing evidence stays
`NEEDS_CONTEXT`; limits, genuinely closed domains, the candidate schema,
`SystemBridge`, and RA → physical evidence → CCA acceptance remain unchanged.
Unsupported public conditions are distinguished from unsupported combinations.

Verification artifacts are retained under
`cais_spade_llm/monitor/environment_runs/machine_outline_goal_20261014_01a137cc/capability_context_20261014/`:

- `before_prompt.txt`, `after_prompt.txt`, and `replay_report.json`: the retained
  machine replay exposes robot context and recorded KMR arrivals at `Storage`,
  `M1`, and `M2`. The retained turn-8 M2 `unload_part` proposal remains
  `INFEASIBLE` at RA because its `staging_part` condition is unsupported; CCA is
  not reached. The shared capability instruction occurs once and has 47 words.
  The total prompt grows from 17,110 characters / 1,511 words to 28,285
  characters / 2,108 words because the missing resource and evidence context is
  now included; facts and capture metadata are deduplicated.
- `supplied_cases.json` and its referenced records demonstrate additional empty
  and loaded movement through RA, physical projection, and CCA. These are
  supplied isolated verification cases, not LLM-generated recovery or Gazebo
  execution. Separate tests cover pickup, placement, missing evidence, occupied
  destinations, attempt isolation, reload, and unchanged unsupported operations.
- `regression_tests_final.log`: **1,147 passed, 12 failed**. All 12 failing test
  identifiers match the recorded pre-change baseline: five legacy AP fixture
  cases and seven KMR prepared-execution-provider cases. `poetry check`, Python
  compilation, and `git diff --check` passed; Poetry reports existing metadata
  deprecations. No API requests, live captures, or execution occurred.
- The replay confirms the candidate schema and accepted prefix are unchanged,
  along with all 356 protected JSON files. `retained_hashes.json` records the
  protected inputs and results. Both machine safety specifications remain
  `SAFE_M2_mutex` and
  `SAFE_KET4_Square_4mm_before_RGOCG4-50_Round_4mm_processing`.

The official machine capture is still
`capture_8beafdb8a01e444894cfaf41d29921fd`, with native checkpoint
`4a6d23fca53a4d4486ba50f9d005203a`. The current partial machine attempt
`offline_bff94739bcd14967835bee4eae750ac8` retains four accepted LLM events:
M2 `interrupt_job` validated against `pause_job`, and KMR `move_to_location`
from M2 to M1, then to Storage, then to M2. Nominal continuation remains
`not_ready`. The earlier two-event `offline_2bcbe6fd3c1b42aab5730aa7ec4d38ee`
remains a source dependency. The earlier one-event
attempt `offline_07d7348ec3054f6a81c83a54a56d33fc` remains unchanged as a source
dependency. The Part slippage outline
`offline_470087fe7a92421abd624ea721aa2ca1`, composition
`offline_1bb1efac84d7443aa37238521ead3b68`, and their dependencies remain protected.
The measured UR5e reach mismatch, current-scene transfer/custody evidence, and a
complete accepted machine recovery remain separate work. The implementation and
supplied calculations do not establish completed recovery or execution authority.

The continuation `attempt_05`, `offline_2bcbe6fd3c1b42aab5730aa7ec4d38ee`,
revalidated both accepted events and replayed the rejected turn-10 pickup using
the corrected session identifier. One new actual OpenAI request proposed KMR
`move_to_location` from M1 to Storage at turn 11. RA returned `FEASIBLE`, physical
projection returned `needs_context`, and CCA returned `unresolved`. Generation
stopped without adding that event; no safety violation was demonstrated.

The detailed cause is **two supported owner declarations**, not an established
infeasible movement: `move_to_resource` (event 1) and `move_to_location` (event
86). `project_bounded_successor` requires exactly one declaration. Its error
also includes `Guard blocked: KMR.resource_location` from a third, rejected
M2-origin alternative, which obscures the actual ambiguity. The exact local
reproduction is `attempt_05/movement_declarations.json`. The next code work
should evaluate the supported alternatives with a justified evidence rule,
retain any differing effects, and distinguish owner/physical evidence failures
from violations of selected safety specifications. Do not simply choose a
favorable declaration, bypass evidence, or presume movement is unrelated to
`SAFE_M2_mutex` from its location names. CCA should require only evidence needed
to establish the selected safety predicates; state/evidence requirements for
later recovery events and nominal continuation remain separate obligations.

This retry made three new actual API requests across attempts 04 and 05. The
session-identity repair passed both focused cases and all **622** context,
environment-recovery, and validated-outline regression tests. `poetry check`,
Python compilation, and `git diff --check` passed. Both owned generation
processes exited; forbidden execution calls were zero. `retry_summary.json`
indexes the result and preservation checks. Attempt 05 retains its partial
prefix, original provider records, and validation dependencies. Its redundant
preview cleanup is recorded in `attempt_05_preview_deletion_list.json` and
`attempt_05_preview_cleanup.json`. No complete machine outline, primitive
composition, live recovery, or nominal execution has been established.

Retry `attempt_04`, `offline_27c13d69f16b432f94f13f1586e9416c`, made two actual
OpenAI requests. It independently revalidated the retained M2 pause event and
accepted the new LLM turn-9 KMR `move_to_location` from M2 to M1 through RA,
physical evidence, and CCA. The accepted prefix is now two events; nominal
continuation remains `not_ready`. Turn 10 proposed KMR `pick_approach` for
`KET4_Square_4mm` at M1 and exposed an integration omission: the production
multi-turn seed did not carry its existing `recovery_session.session_id` into
validation. The seed now preserves that identifier; the additional-binding test
uses the production seed and both empty/loaded cases pass. No identifier is
invented and missing attempt identity still blocks additional evidence.

`attempt_04/pending_revalidation.json` replays the unchanged turn-10 proposal
locally after the fix. RA now reports `INFEASIBLE` before physical projection or
CCA. `pickup_peer_conditions.json` identifies the actual M1 mismatch: nominal
pickup requires `resource_state=completed`, while recovery pickup requires a
loaded part with `job_state=paused` or `job_state=stopped`. The captured M1 has
`resource_state=loaded`, `job_state=interrupted`, and `active_job=null`. None of
the recorded peer custody contracts matches. Source observations, declarations,
and the two accepted events remain unchanged. The pending replay made no API
requests and issued no commands. The current partial result and its evidence
remain retained; only the redundant `input_preview.json` is listed for cleanup
in `attempt_04_preview_deletion_list.json` and `attempt_04_preview_cleanup.json`.
The continuation uses this actual validator feedback without supplying a route
or replacement event.

The requested retry first created `offline_2acf319c7166404e97d5aa2f0296a9da`
(`attempt_03`) and stopped before any API request: the diagnostic resume helper
matched only `outline_id`, which both a rejected turn-2 proposal and the accepted
turn-3 revision used. The helper now matches all original provider proposal
fields against the accepted event before its existing independent revalidation.
The failed preparation has no accepted events or provider records. Its report
is retained; redundant preparation outputs are listed in
`attempt_03_deletion_list.json` and `attempt_03_cleanup.json`. The next retry
continues from the unchanged accepted turn-3 response with the improved context.

## Previous handoff — October 13, 2026 — machine outline goal in progress

The goal is a complete accepted `Machining breakdown during part processing`
outline with retained nominal-continuation proof, loadable through the Recovery
UI. Primitive composition and recovery execution remain excluded. The user
subsequently authorized OpenAI API requests and another live Gazebo failure
capture if needed. The user also requested deleting obsolete machine failure
captures and retaining the correct replacement observations. Verify and load
the replacement and check dependencies before removing the original machine
capture or its derived copy; other scenarios and regression coverage remain
protected.

Initial inspection found only the original `4a81d7fc7ccd46d1a18b79740d16add7`
machine capture and its retained derived observations. Their
machine records are at simulation times 84.0/84.2, while physical observations
are at 85.65–85.8665, without `capture_checkpoint`. The current compatibility
validator reports `NEEDS_CONTEXT`. The later synchronized source and two-event
replay mentioned below are absent; their documented acceptance is not a
loadable session. No new provider request was sent against that known gap.

Work and preservation evidence are under
`cais_spade_llm/monitor/environment_runs/machine_outline_goal_20261014_01a137cc/`.
The recorded setup, timing seed `4000168541`, and both selected safety files
remain unchanged. The current Gazebo workspace was built successfully.

- `capture_01`: failed before nominal startup because the runner replaced the
  sourced ROS `PYTHONPATH`, producing `ros2 imports failed`. Its owned launch
  stopped, protected file hashes matched, and no failure capture was produced.
  The corrected invocation preserves the sourced path; controller and native
  preview imports passed. Its explicit deletion inventory and cleanup receipt
  record 87 removed entries and 396,081 regular-file bytes released.
- `capture_02`: the 900-second wall-clock deadline interrupted nominal
  `RGOCG4-50_Round_4mm` placement at M2 before the failure checkpoint. KMR
  controllers were progressing; no failure observations or provider requests
  were produced. All owned processes exited and retained hashes matched.
  The diagnosis and explicit deletion inventory are in
  `capture_02_failure.json` and `capture_02_deletion_list.json`. The next
  capture uses the unchanged setup and seed with an 1800-second window on
  ROS domain 89 and Gazebo port 11379. LLM, primitive generation, and recovery
  execution remain disabled during capture.
  Its cleanup receipt records 90 removed entries and 512,016 regular-file
  bytes released, without following workspace symlinks.
- `capture_03` / `3df0234a84214934a027f109b7fe7e22`: reached the intended M1
  `during_processing` fault, then failed checkpoint preparation with
  `Handling root has not stopped: ur5e-1/linear_velocity`. The exact-zero
  check rejected native integration jitter on the fixed UR5e root. Its fallback
  observations have no `capture_checkpoint` and cannot replace the official
  source. All owned processes exited. The failure report and original native
  observation before cancellation are retained separately under the goal
  directory. Disposable output removal is listed in
  `capture_03_deletion_list.json`; it also includes the metadata-only
  `18d84a91b2804d3c9d17658b2d70d497` record left by `capture_02`.
  The completed cleanup removed 98 entries and 102,251,808 regular-file bytes.
  All 14 protected source and specification hashes still match.
- Checkpoint repair records the physics robot description and proves a fixed
  joint chain to `world` before using the existing declared stationary speed
  limits. KMR uses its physics description, keeps the zero native velocity
  requirement, and all controllers still require authoritative idle evidence.
  Native observations are retained unchanged. Focused checkpoint checks:
  29 passed. Capture, machine-job, and conveyor-fault regressions: 181 passed.
  `poetry check`, compileall, and `git diff --check` passed.
- `capture_04` succeeded, retaining
  `capture_8beafdb8a01e444894cfaf41d29921fd` and its source checkpoint records
  `4a6d23fca53a4d4486ba50f9d005203a`. Both machine observations and geometry
  share simulation time `84.3455`; M2 remains running and unfinished.
  `replacement_capture_verification.json` records supported compatibility,
  native-model verification, unchanged setup, and successful saved selection.
  All owned Gazebo processes exited, protected hashes matched, and capture
  made no LLM calls or recovery execution.
  The exported observations retain identical checkpoint, machine, physical,
  resource, part, failure, and pending-task facts from the source run. Its
  redundant `observations.json` was removed after verification, releasing
  73,765,395 bytes. All source `checkpoint_*.json` evidence remains retained.
  `machine_capture_selection_after_cleanup.json` confirms exactly one machine
  capture choice, with successful observation and native-archive loading.
  The file inventory and receipt are
  `duplicate_capture_observation_deletion_list.json` and
  `duplicate_capture_observation_cleanup.json`.
- Offline attempt `offline_510e5db485a647b989c5c8a9f5546a33` ended blocked
  after 12 actual `gpt-5.4` / `medium` requests. Its owned process exited.
  `attempt_01/source_preparation.json` verifies normal Recovery UI preparation
  and both selected specifications. Provider responses, validation history, and
  the one-event prefix remain required inputs for local revalidation and resume.
  No complete accepted outline or nominal proof has been established yet.
- The user-authorized replacement makes `4a81d7fc7ccd46d1a18b79740d16add7` and
  `derived_4a81d7fc7ccd46d1a18b79740d16add7_20261012` obsolete. The explicit
  removal inventory is `old_machine_capture_deletion_list.proposed.json`.
  Neither code nor regression fixtures reference them. The only references
  outside those directories are historical preservation/selection inventories
  from the unrelated Part slippage task, which remain untouched. No process
  held either directory at the pre-deletion ownership check. The cleanup is
  complete: 46 entries and 255,151,941 regular-file bytes removed. Replacement
  observation/native-model loading and Recovery UI selection passed again;
  the other 12 protected source and specification hashes remain unchanged.
  The receipt is `old_machine_capture_cleanup.json`.
- The fresh LLM attempt accepted M2 `stop_job`
  (`outline_stop_m2_preserve_trim_order`, turn 4) through RA, physical evidence,
  and CCA. Nominal continuation is still `not_ready`. Turns 5 and 6 attempted
  commands on unavailable M1 and were rejected; this is model feedback, not a
  reason to change the captured machine availability. The accepted prefix is
  still a partial result and execution remains disabled.
- The finished attempt repeated unsupported M2 `unload_part` proposals at turns
  7, 9, and 11. Turn 12 supplied only a part condition and was reported as
  `NEEDS_CONTEXT`. The diagnostic runner now counts repeated findings across
  intervening proposals while the accepted prefix is unchanged, stopping before
  a fourth request against the same unchanged blocker.
- Saved-input diagnosis found that context selection omitted KMR's unfinished
  `M2` to `Storage` task because its exact `source_resource` and
  `target_resource` fields were not considered, and the affected part's location
  was not included in dependency closure. The scoped repair exposes KMR's
  recorded state and declarations without selecting a recovery action.
  A second repair classifies a part-only `expected_end_state` as `INFEASIBLE`
  when the recorded declarations are complete; genuinely missing declarations
  remain `NEEDS_CONTEXT`. The saved turn-12 proposal is now rejected locally.
  Evidence: `recorded_context_proposal_recheck.json`.
- Five new focused checks pass. The combined context/feasibility suites report
  491 passed and 12 failures; all 12 failures reproduce with the pre-repair
  functions. They concern legacy AP descriptors and unprepared KMR execution
  fixtures. Their unchanged baseline is recorded in
  `context_proposal_preexisting_failures.log`. Poetry check, compileall, and
  diff checks pass. The next attempt revalidates the actual saved grounding and
  accepted provider response before continuing from the corrected context;
  no new request was sent before establishing these changes.
- The continuation launch was rejected by automatic approval review before
  execution because sending the saved internal recovery context to
  `api.openai.com` requires payload-specific user approval. The user has been
  asked to authorize the machine observations, robot/part states and geometry,
  public declarations, both specifications, accepted events, and validation
  feedback for the remaining outline requests. No request was sent after this
  rejection. The exact unsent prompt and parameters are retained in
  `local_prefix_revalidation/next_request_prompt.txt` and
  `local_prefix_revalidation/next_request_parameters.json` under the goal
  directory; `api_payload_approval_pending.json` records the external blocker.
- Local-only revalidation created
  `offline_4bed5f73eabd4fb6a6e9ef4ee76984e2`. The unchanged actual grounding
  response, accepted turn-4 `stop_job` response, and rejected turn-12 response
  were replayed through the existing phase handlers. The prefix remains exactly
  one LLM-authored event; the old part-only proposal now returns `need_revision`.
  Twelve original requests/responses were retained byte-for-byte, and zero new
  requests were sent. `local_prefix_revalidation/prefix_revalidation.json`
  records response hashes and validation lineage. The saved `prepared.json`
  and `session.json` in that directory preserve the revalidated state.
- After successful replay, an explicit ownership/mtime/hash-checked cleanup
  removed two redundant previews: the failed attempt's `input_preview.json`
  and the large duplicate preview containing local-only prompt inputs.
  Actual provider records, original inputs, exact unsent prompt/parameters, and
  current debugging dependencies remain. `failed_attempt_preview_cleanup.json`
  records 19,524,527 bytes released. `retained_load_after_prefix_cleanup.json`
  confirms one machine capture choice, successful native-model and observation
  loading, an unchanged prefix, a bound loadable nominal proof with status
  `not_ready`, and all 12 remaining protected files unchanged.
- An additional local reproduction isolated the KMR nominal witness gap:
  recorded `environment_22` (`move_to_resource`, `M2` to `Storage`) passes RA
  and CCA, but `_exact_nominal_witness` expected the registered robot-task
  witness format instead of its recorded environment-event contract.
  `nominal_environment_22_revalidation.json` retains the complete reproduction.
  The scoped endpoint-policy-2 repair checks the exact owner/model, parameter
  binding, selected empty/carrying program, every supported program step, and
  final declared effects. It preserves conditional planning endpoints and all
  execution obligations. Rechecking the original evidence now passes this
  witness check (`nominal_environment_22_exact_witness_after_fix.json`).
  Nineteen focused checks and all 208 outline regression tests pass. The first
  sandboxed full run stalled at the known asynchronous-worker boundary and was
  stopped; the host run completed in 11.79 seconds. Required Python checks pass.
  The complete existing nominal-continuation validator was then rerun locally:
  it advances through `environment_22`, while `environment_21` remains blocked
  by unavailable M1 and `environment_23` lacks supported `start_job` capability.
  Its detailed owner finding is `NEEDS_CONTEXT: workholding and access evidence
  for the projected scene at M2`; no such evidence was fabricated.
  `nominal_continuation_after_kmr_witness_fix.json` references the retained
  `recovery_outline/turns/turn_00012_validation_0001.json` proof in the local
  revalidation run. Overall status remains `not_ready`, with one accepted
  recovery event and zero generated primitives. Continue by resolving the
  recorded M2 access-evidence obligation and obtaining revised LLM proposals
  once the pending payload approval is answered. All local diagnostic and
  test processes have exited; no new API request or recovery execution occurred.
- Local nominal-continuation repair: endpoint policy 2 previously rejected
  every `machine_part` before reaching the existing owner dwell/completion
  calculation. It now routes that exact owner declaration to
  `_project_machine_dwell`, retaining its remaining-job duration, geometry,
  stationary evidence, and declared-effect checks. Endpoint policy 1 and
  unmarked behavior are unchanged. The new modern-job regression also verifies
  both selected specifications: premature `RGOCG4-50_Round_4mm` completion
  remains unsafe. Focused projection checks: 8 passed; machine jobs: 56 passed.
  The complete environment projection run had 98 passes and two overly narrow
  new test assertions about missing-peer wording; the focused rerun corrected
  those assertions and passed all eight affected cases. No provider-generated
  complete outline or nominal proof has been obtained yet. The focused nominal
  regression selection also passed 64 tests; required Python checks passed.
  The final complete environment projection suite passed all 100 tests.
- Payload approval was received: "Approve this payload and subsequent revisions
  for this goal." `api_payload_approval.json` records authorization for the
  remaining OpenAI outline-generation requests and their revised context.
  Local owner checks then established that the earlier `stop_job` prefix is
  unsuitable: `resume_job` requires a paused job, `start_job` rejects restarting
  this unfinished part, and `pause_job` requires the active running job.
  `stopped_prefix_control_recheck.json` retains the exact unchanged-state checks.
  This does not prove that the failure scenario lacks a solution. Attempt 02
  starts a fresh LLM outline from `capture_8beafdb8a01e444894cfaf41d29921fd`,
  with factual prior-branch feedback in `fresh_attempt_feedback.json`; it does
  not import the stopped prefix or prescribe replacement recovery events.
  The existing provider configuration and three-repeat guard remain in use.
  The original provider responses and local reproduction inputs remain needed
  for debugging; redundant failed-attempt previews have already been removed.
- Attempt 02, `offline_07d7348ec3054f6a81c83a54a56d33fc`, finished after
  eight actual OpenAI requests/responses in 481 seconds. One grounding turn and
  seven proposed events produced one accepted event: the exact LLM-authored
  `interrupt_job` / `outline_recovery_1` at turn 3, validated against the owner's
  `pause_job` declaration, with `active_job=null` and `job_state=paused`.
  Its nominal-continuation proof remains `not_ready` at
  `generation_0001/recovery_outline/turns/turn_00003_validation_0001.json`
  (SHA-256 `061991ecb9616a3c57cb30aee6e2d97ecbbdc03b0e2de442eb98aabbfabc10ff`).
  Repeated unsupported M2 end conditions reached the three-repeat guard;
  turn 9's five provider-wrapper retries were blocked locally before sending.
  They are request/error records, not five additional API calls. The owned
  process exited, observations are unchanged, and forbidden execution calls
  remain zero. Attempt 02 retains its provider records and current debugging
  inputs under the goal directory's `attempt_02/`.
  The next step is local diagnosis, not another unchanged API attempt:
  `recorded_handling_declarations_audit.json` records all four UR5e owner reach
  checks against both unfinished parts. The nearest robot for each part reports
  `xy_radius=0.7244 m, max=0.7000 m` (recorded tolerance `0.01 m`), and KMR's
  recorded pickup declarations bind `origin_resource_location=Storage`.
  Check the captured robot model and existing handling implementation before
  deciding whether a scoped correction is supported; do not relax reach limits
  or add unsupported handling claims merely to accept a proposal.
  The exact targets from `calculate_recovery_pick_target`, including the
  recorded TCP offset, now have bounded joint solutions verified by owner FK
  for both nearest UR5e robots (position residual below `1.5e-14 m`).
  `recorded_ur5e_pick_targets.json` and
  `recorded_ur5e_owner_pick_kinematic_diagnosis.json` retain the inputs/results;
  the accompanying diagnostic script sends no command. The preliminary
  `recorded_ur5e_kinematic_diagnosis.json` used a raw part-centre tool target
  without the owner TCP offset and is not a pickup-capability result.
  Kinematics alone does not establish collision-free pickup or justify changing
  the declared workspace. Check exact owner endpoints and the source of the
  scenario's inherited `gripper_reach` before changing its configuration.
  Further saved-model diagnosis now passes all four native calculations:
  `ur5e-1` and `ur5e-2`, each from the recorded start to its exact owner pickup
  joint solution, and each with a separate explicitly hypothetical endpoint
  check. Results are in `recorded_ur5e_pick_collision_diagnosis.json`; original
  Goal/Result files and provenance are under `pickup_collision_native/`.
  All return `PREVIEW_COMPLETE`; the native checker verified model bounds and
  sampled collisions while retaining the measured stationary joints and declared
  mounting contacts. These calculations do not prove continuous collision freedom,
  custody transfer, CCA authorization, or execution readiness. They do not enter
  the accepted LLM prefix and do not change the recorded `gripper_reach`.
  The source observation hash remains unchanged and the owned process exited 0.
  Next, establish a supported configuration correction and the actual M1-to-M2
  transfer route before another accepted-outline attempt. The Conveyor's recorded
  normal output is `Buffer For Machined parts`; no M1-to-M2 Conveyor route has
  been established. The current reach rejection and the missing M2 workholding/
  access finding are both reported at RA, before separate physical projection
  and CCA. A full unvalidated LLM draft may aid diagnosis, but must remain distinct
  from an accepted outline and cannot bypass those acceptance gates.
  Cleanup removed only this finished attempt's redundant `input_preview.json`
  (13,387,804 bytes), after ownership and metadata/hash checks. The explicit
  inventory and receipt are `attempt_02_preview_deletion_list.json` and
  `attempt_02_preview_cleanup.json`. `retained_load_after_attempt_02_cleanup.json`
  confirms all 12 protected hashes, one selectable supported machine capture,
  native model loading, the unchanged paused prefix, and its loadable
  `not_ready` nominal proof. No additional API calls or execution occurred.
- Checkpoint verification: `168 passed` across the existing capture, machine
  jobs, and conveyor fault suites. The first sandboxed test process stalled
  and was terminated; the completed run used host access. The unrelated
  Part slippage composition task was left untouched.

## Previous October 13 handoff — exact saved KMR planning validation passes

The unchanged attempt-07 M2 pause_job and KMR move_to_location from M2 to M1
now both pass the actual validated-event → RA → CCA path and enter the accepted
prefix. The saved KMR proposal carries no part. This completes the scoped
planning-validation goal; it does not establish placement into M2, complete
nominal continuation, or permission to execute the robot.

### Corrections

- New preparations use endpoint_evidence_policy_version=2 while retaining
  outline_evidence_policy_version=1, validation_contract_version=2, and
  primitive_support_semantics="successor_primitive_union". Endpoint policy 1
  and unmarked sessions retain their recorded behavior. The endpoint policy
  participates in requirement, scene/cache, occupancy, and monitor identities.
- A declared base endpoint conditions its geometry on successful owner action
  completion. Missing starting base readiness is retained with scope=execution,
  blocks_planning=false, and execution_authorized=false. It no longer prevents
  CCA from checking independently supported predicate bounds. An action that
  does not establish a base endpoint still requires its existing base evidence;
  owner capability, arm hold, timing, and selected-predicate checks remain.
- RA forwards the existing owner state declarations to CCA for complete mutex
  participant grounding, using the same declaration expansion as the existing
  owner-ledger path. This does not manufacture task/state ledgers. CCA
  independently reproduces the declarations with the rest of the projection.
- The accepted KMR endpoint retains its owner JID, checked against the task,
  owner model, and physical resource binding. Physical capture rows can omit
  this identity; dropping it previously broke the next capability validation.
  Conflicting identities remain unresolved.
- Full trajectory generation, sampling, and replay remain disabled. Endpoint
  calculations and conservative Boolean occupancy alternatives provide the
  selected evidence; supported peer exclusion permits unknown KMR occupancy.
  No event name, destination label, or safety-rule-specific bypass was added.

### Exact replay and successor verification

Artifacts are under
cais_spade_llm/monitor/environment_runs/endpoint_bounds_20261013/.

| Check | RA | Physical evidence | CCA / accepted prefix |
| --- | --- | --- | --- |
| Exact M2 pause_job | FEASIBLE | Owner job-control effects and selected predicate coverage | Passed; prefix length 1 |
| Exact KMR M2 → M1, empty | FEASIBLE | Supported endpoint; base (-4.7, 2.2), tool approximately (-5.2, 1.95, 1.16); all 62 required APs covered | Both selected specifications passed; prefix length 2 |
| Separate M1 → M2 verification branch from the saved accepted prefix | FEASIBLE | Consumed the accepted M1 base pose, not the captured pose; recalculated M2 endpoint (-2.7, 2.2) | Passed; branch prefix length 3; both monitor histories independently rechecked |

replay_goal_final/report.json and turn_1.json/turn_2.json contain the exact
saved replay. Its source_responses.json matches the earlier saved replay's
proposal file byte for byte. saved_successor_final/report.json and turn.json
record the separate successor-consumption check. That third proposal is
explicitly a verification-only branch; it does not replace or extend the
original LLM-authored proposal record. Both diagnostics report zero forbidden
API, live-interface, execution, and trajectory calls. No new capture was made.

The original base-readiness gap remains visible as an execution obligation.
Projected arrival is not an observation of arrival, and a passed CCA planning
result does not authorize dispatch. SystemBridge and the execution guards were
not changed. Earlier unresolved diagnostic artifacts remain unchanged.

### Verification

Final focused regression run: **1,550 passed in 264.02 seconds**, across
test_environment_recovery.py, test_validated_recovery_outline.py,
test_recovery_outline_evidence.py, test_recovery_context_retrieval.py,
test_kmr_outline_evidence.py, test_machine_jobs.py, test_offline_recovery.py,
and test_recovery_case_safety.py. The new cases cover absent/active base
readiness, historical policy 1, owner-declared participant populations without
duplicate geometry flags, identity-free capture rows, conflicting identities,
and tampered readiness/declarations/policy. Existing negative checks still
block missing peer exclusion, possible overlap, relevant running jobs, stale
evidence, and unsupported predicates, while preserving accepted state/history.

poetry check passed with existing metadata deprecation warnings;
poetry run python -m compileall -q cais_spade_llm ros2 and git diff --check
passed. See goal_regression_final.log, goal_poetry_check.log, and
goal_compileall_final.log. goal_immutable_after.json verifies both original
captures and all eight safety files against immutable_before.json: all 10
hashes are unchanged. Public bridge, safety files, configuration files, and
the unrelated concurrent composition/UI work were not edited by this repair.

## Previous October 13 handoff — endpoint policy 1 before readiness separation

The KMR physical-evidence milestone uses owner endpoint calculations and
conservative occupancy alternatives. New planning validation never requests,
generates, samples, or replays a full physical trajectory. Robot execution is
unchanged and remains a separate, unauthorized stage in these checks.

New preparations retain outline_evidence_policy_version=1 and add
endpoint_evidence_policy_version=1 to their internal physical evidence. The
new marker, owner facts, exact proposal, starting state, models, and selected
specifications participate in evidence and cache identities. Sessions without
the endpoint marker retain their historical validation behavior. Contract
version 2, successor_primitive_union, SystemBridge, fixed symbols, safety files,
original captures, and LLM-authored proposals remain unchanged.

### Implemented calculation and validation

- RA resolves a unique supported owner declaration and its actual program
  variant, then reuses owner parameter calculations and FK. KMR base targets
  update the model root; arm, tool, component bounds, and attached part poses
  are calculated in the same world frame. Both declared docks and explicit
  x/y/yaw targets are supported. This is a conditional result after successful
  owner action completion, not an observed arrival or execution authorization.
- The interval carries Boolean occupancy alternatives. KMR can remain
  [false, true] throughout a movement. Independently checked peer holds and
  their robot/tool/carried-part bounds can exclude every peer from the bound
  regions. CCA's existing monitor checks every represented possibility. A
  possible violation requests more precise owner bounds; no trajectory is the
  fallback. A captured peer position alone cannot establish a future hold.
- Requirements still derive from exact AP dependencies. Processing uses exact
  ledgers and relevant progression; spatial checks retain the complete relevant
  population. Unknown elapsed time retires unrelated running-job facts and
  clocks. Relevant unknown progression blocks acceptance.
- Successful endpoint evidence is returned separately from readiness, timing,
  and predicate gaps. Accepted successors retain recalculated geometry and
  supported holds, retire old scene caches and aliases, and preserve original
  observations as history. CCA independently recalculates the owner result,
  requirements, and coverage. Failed candidates retain accepted state/history.
- Offline automatic previews, explicit owner preview entry points, and old
  rejected/incomplete-motion fallback paths are disabled for marked attempts.
  The carried move_to_location capability witness now uses the existing
  carrying variant's attachment argument, without changing the primitive union.
- The new bounds policy encloses passive-joint observations within eight
  floating-point steps of a declared limit instead of clipping them or treating
  rounding as physical overshoot. Larger discrepancies remain unresolved;
  legacy holds keep their strict comparison. This covers the saved
  KMR_rg2_finger_width value 0.11000000000000007 against its 0.11 limit.

Implementation starts at recovery_framework/outline_endpoint_evidence.py and
outline_evidence_policy.py. The existing RA/CCA and shared validated-event path
use these calculations; this is not only a standalone diagnostic.

### KMR milestone and exact saved replay

Artifacts are under
cais_spade_llm/monitor/environment_runs/endpoint_bounds_20261013/.

supported_kmr.json records integrated **RA → CCA → accepted successor → next
geometry-dependent event** checks, empty and carrying, from explicitly supported
synthetic starting states. They use real SavedRecoveryResource capability
checks, real owner declarations, CCA monitors, and the normal commit function.
Both movements pass; the second consumes the first endpoint and resolves M2's
configured dock. Tests check unchanged arm joints, tool FK, transformed carried
parts, and coordinate-only destinations without a named dock. The synthetic
fixtures are separate from the unchanged saved proposals.

replay_verified/report.json records the exact attempt-07 provider responses
from turns 5 and 16, with API, live interfaces, full trajectory generation,
sampling, and replay guarded against use. The selected specifications request
60 resource-region APs plus two processing-completion APs.

| Exact saved proposal | RA capability | Required evidence and projection | CCA / accepted state |
| --- | --- | --- | --- |
| M2 pause_job | FEASIBLE | Existing instantaneous owner job control supplies the effect and selected predicate coverage | Passed; one event accepted |
| KMR move_to_location to M1 | FEASIBLE | Endpoint SUPPORTED; base target (-4.7, 2.2), tool approximately (-5.2, 1.95, 1.16), seven arm joints retained; peer bounds available. Starting base readiness remains unsupported | Unresolved on readiness; accepted prefix and monitor history unchanged |

The exact saved KMR proposal targets M1, not M2. Its missing fact is the
captured base readiness/command-gate evidence, not a trajectory. No capture was
edited to mark its active controller idle. No new capture, API generation, or
robot execution was used. Both replay and supported fixtures have zero
forbidden calls.

### Verification and remaining scope

Final focused regression run: **1,538 passed in 135.14 seconds**, across
test_environment_recovery.py, test_validated_recovery_outline.py,
test_recovery_outline_evidence.py, test_recovery_context_retrieval.py,
test_kmr_outline_evidence.py, test_machine_jobs.py, test_offline_recovery.py,
and test_recovery_case_safety.py. poetry check passed with existing metadata
deprecation warnings. poetry run python -m compileall -q cais_spade_llm ros2
and git diff --check passed. Live ROS2/hardware testing was excluded.

The focused suites cover processing-only and mixed specifications, held parts,
missing holds, possible overlapping occupancy, running jobs, unsupported
predicates, changed bindings, stale aliases/caches, tampered evidence, rollback,
historical sessions, and exact nominal obligations. Trajectory entry points are
guarded to fail even for specifications that remain unresolved. See
focused_tests_final.log, final_policy_tests.log, poetry_check.log, compileall.log,
and immutable_after.json for the final results.

The immutable check includes both original captures and all eight safety text
files. Their hashes remain equal to immutable_before.json. The derived capture
still hashes to 870b2ff445fbc600c8d73fbac5c4c94ad0c66e5107002d73d783c24634676dfd;
safety_M2_mutex.txt remains
5f282ecbbfeb83862b6b1958304d4930005163935042c6649145ef905ee1d28f.

Complete recovery and nominal continuation are not established by this milestone.
The new endpoint path currently calculates supported KMR movement programs and
instantaneous machine commands. Other owner programs, output-dependent effects,
missing final joint witnesses, or predicates requiring unavailable timing/state
facts remain explicit evidence gaps. Preserve exact nominal task/parameter/effect
obligations; CCA acceptance cannot manufacture task completion. Arbitrary
specifications need not be decidable from available bounds. Do not restore a
full-trajectory fallback to address these gaps.

Other composition modules and multi_turn_primitive_generation.py appeared as
concurrent work during final status inspection; this task did not edit them.
Keep those changes and the pre-existing worktree changes separate from this
endpoint-policy milestone.

## Previous October 13 handoff — selective evidence before endpoint bounds

### Implemented: evidence selected by safety predicates

The approved selective-evidence repair is implemented in the school Ubuntu
checkout on recovery-framework-journal. The scoped repair is complete;
the machine-breakdown recovery outline and nominal continuation remain incomplete.
No new KMR movement validator, API request, Gazebo capture, or execution was added.

The current pipeline is:

**Captured facts → authored proposal → RA capability support → owner successor
effects and selected evidence requirements → CCA coverage and monitor replay →
commit accepted state → shared nominal-continuation validation.**

New attempts carry outline_evidence_policy_version=1. They retain
validation_contract_version=2, primitive_support_semantics="successor_primitive_union",
SystemBridge, exact safety files, and LLM authorship. Sessions without the new
marker keep the legacy strict projection/monitor path; loading does not upgrade them.

outline_evidence_requirements now has an event-specific form. It uses the
existing exact AP grounding, complete resource population, owner declarations,
current facts and interacting jobs. Requirements are not chosen by a proposed
event name or destination label. Requirements and coverage bind the exact
proposal, starting snapshot, physical facts, owner model/evidence, and selected
specification contents/hashes. CCA independently recomputes these records and
reproduces the selective owner successor before checking its monitors.

For known processing predicates, an unambiguous supported environment movement
program can preserve exact completion ledgers for any elapsed duration without
requesting native geometry preview. The implementation reuses
declared_environment_successors and the existing parameter calculations.
Neither the proposed end state nor the unordered primitive union establishes a
physical result. Unknown dependencies, ambiguous effects, other owner programs,
and independent physical or exact nominal obligations retain the conservative
owner path.

Spatial predicates still require bound regions, complete relevant resources,
tool geometry and carried parts. Supported continuous occupancy envelopes can
cover an AP without another detailed trajectory. A clear endpoint alone cannot
cover a changing spatial predicate. New CCA checks use per-AP coverage instead of
an unconditional interval_complete gate. Unsupported predicates do not become
safe through selective checking.

Supported symbolic successor effects are committed even when geometry is
unnecessary. Uncalculated poses, configurations, bounds, holds and cached scene
evidence are invalidated; their original observations remain historical.
Validation input, public retrieval and subsequent nominal checks cannot recover
those current facts from the original capture. Relevant running job progress
blocks a processing verdict when unknown. Unrelated unknown progress and elapsed
clocks are retired explicitly instead of remaining frozen successor facts.

Monitor contexts retain replayable predicate observations and coverage history.
Changed bindings, region definitions, tampered coverage or replay states fail
validation. Rejected/unresolved candidates leave accepted state and history
unchanged. Policy and requirements are included in projection cache identities.
Projection/request and reuse reasons appear in the existing expandable records.

Nominal continuation uses the same event-validation path. Its registered task,
parameter, step and effect obligations remain enforced. Selective CCA evidence
does not create a nominal completion witness or authorize execution.

### Exact saved-proposal replay and remaining blockers

Final evidence is in
cais_spade_llm/monitor/environment_runs/selective_evidence_20261013/.
replay_handoff/report.json records RA, required evidence, projection, CCA and
accepted-state preservation separately. The replay used the unchanged exact
provider responses from attempt-07 turns 5 and 16. Both requests generated
62 grounded AP requirements: 60 resource-region bindings for SAFE_M2_mutex
and two exact processing-completion predicates.

| Saved proposal | RA | Required evidence / projection | CCA | Accepted prefix |
| --- | --- | --- | --- | --- |
| M2 pause_job | FEASIBLE | Owner job-control effect and required AP coverage established | Passed both selected specifications | One event |
| KMR move_to_location | FEASIBLE | Required owner program / occupancy interval evidence missing | Unresolved | Still one event; accepted state/history unchanged |

Final replay: **0 API calls, 0 live-interface calls, 0 forbidden calls**.
KMR acceptance is not a requirement for this scoped repair. The saved moving
checkpoint does not establish stopped-start or future route/arrival coverage.
Another capture alone would not establish the proposed motion. No recapture
is needed to demonstrate this repair, and none was started.

The existing KMR move_base owner-preview connection still lacks its supported
continuous base-controller/carried-part motion evidence. Resolving that capability,
choosing an additional capture, or resuming API generation is separate follow-up
work. Primitive composition, recovery execution, concurrent runtime recovery
composition, and observed assembly completion remain unverified. Do not claim
that this repair establishes the saved attempt's nominal continuation.

### Verification and files for continuation

Final focused regression run: **1,333 passed in 54.61 seconds** across
test_environment_recovery.py, test_validated_recovery_outline.py,
test_recovery_outline_evidence.py, test_recovery_context_retrieval.py,
test_kmr_outline_evidence.py, test_machine_jobs.py, and test_offline_recovery.py.
See focused_tests_final.log. Added cases cover processing-only movement without
preview, relevant and unrelated running work, carried parts, mixed/unsupported
specifications, interval coverage, changed bindings, invalidated geometry, stale
caches, tampered coverage, history replay, rollback, legacy sessions and nominal
obligations.

poetry check passed with existing metadata deprecation warnings.
poetry run python -m compileall -q cais_spade_llm ros2 passed.
git diff --check passed. Live ROS2/hardware validation was deliberately excluded.

before.json and after.json confirm unchanged SHA-256 values:

- Original derived_dfa4017c0119403088d1d4f353a407d5_20261013_public/observations.json:
  870b2ff445fbc600c8d73fbac5c4c94ad0c66e5107002d73d783c24634676dfd.
- safety_M2_mutex.txt:
  5f282ecbbfeb83862b6b1958304d4930005163935042c6649145ef905ee1d28f.
- safety_KET4_Square_4mm_before_RGOCG4-50_Round_4mm_processing.txt:
  b3a34b5a8958dad3bec8411f598f1ac35e1395e6025628d2c1b213e0eeee4c76.

Start future inspection with recovery_framework/outline_evidence_policy.py,
outline_validation.py, resources/environment_recovery.py,
agents/central_controller/outline_evidence_safety.py, and the shared
multi_turn_validated_event.py / multi_turn_nominal_evidence.py path.
replay_selected_policy.py reproduces the final offline replay; it omits the
historical runner's optional additional preview-preparation action. Keep one
agent, preserve captures and exact provider records, and diagnose missing owner
evidence locally before expanding the scope.

## Previous October 13 handoff — historical context

The following handoff predates the selective-evidence repair above. Its proposed
capture/API next steps are historical and do not authorize new runs in this task.

This section is the starting point for the next chat. Older entries below are
historical evidence, not the current next-step list. Preserve them. The user
requested this handoff because the previous chat became too long and costly.

### Objective and current result

Focus on **Machining breakdown during part processing**:

**Capture → stop Gazebo → LLM proposal → RA → physical projection → CCA →
append/project → repeat until nominal continuation is established → UI review.**

The latest machine-breakdown attempt is **incomplete**, but it is no longer
zero accepted events. In `attempt_07`, the provider's corrected M2 `pause_job`
proposal passed RA, physical projection, and both CCA specifications. The
accepted prefix contains that one event. The later KMR `move_to_location`
proposal passed RA but could not establish physical projection. There is no
complete machine-breakdown outline, nominal-continuation proof, primitive
composition, recovery execution, or observed assembly completion to report.
Earlier Part slippage success is recorded separately below; it does not prove
this machine-breakdown case works.

Seven development API attempts are retained in the October 13 result bundle
(957,517 total API tokens). Do not repeat another API attempt against the same
deterministic KMR implementation blocker. Reproduce and repair it locally first.

### User constraints and authorization to carry forward

- Use one agent and the school Ubuntu checkout
  `/home/jongh/projects/cais-spade-llm`, branch `recovery-framework-journal`.
  Read `AGENTS.md` and inspect the dirty worktree before editing. Many changes
  predate this handoff; do not reset, discard, or rename them.
- Preserve `SystemBridge`, exact project identifiers, Offline Manual, expandable
  turns, original captures, historical provider responses, and the current UI run.
- Shared compact instructions and dynamic public `need_context` retrieval remain.
  Do not supply a preferred resource, recovery sequence, worked answer, or event
  count. Test fixtures and private validator programs must stay out of prompts
  and public retrieval. Every accepted recovery event must originate in an exact
  captured provider response.
- Preserve `primitive_support_semantics="successor_primitive_union"` and
  `validation_contract_version=2`. RA checks unordered primitive support against
  one starting snapshot. It does not establish an execution order or trajectory.
- Preserve `safety_M2_mutex.txt` and
  `safety_KET4_Square_4mm_before_RGOCG4-50_Round_4mm_processing.txt` unchanged.
  Keep unknown evidence unresolved; do not weaken CCA to obtain acceptance.
- The user already authorized OpenAI API use for the repair loop and an isolated
  Gazebo recapture when necessary. Use `gpt-5.6-sol`, reasoning effort `low`,
  service tier `default`, 80 turns, and no overall ten-minute deadline. Preserve
  the existing scenario conditions, durations, and seed. Do not reintroduce the
  rejected changed-condition evaluation or disable KMR, `side_access`, or M2.
- Ordinary offline preparation, retrieval, and validation must not call Gazebo,
  perception, controllers, or SPADE transport. Capture is a separate owned run.
- Primitive composition, recovery execution, tool changes, new KMR movement
  validators, and concurrent runtime CCA integration remain outside this repair.
  Reuse existing owner/projection components and repair demonstrated integration
  defects. Explain any unavailable capability before expanding the scope.

### Why another capture was discussed

KMR **was captured**. Its base was moving toward Storage in both saved frozen
checkpoints (`885361b9a7244e2fa4c6a8daa47ca278` and
`dfa4017c0119403088d1d4f353a407d5`). Freezing Gazebo time preserves that moving
state; it does not prove the base has stopped or will remain at that pose.
The saved records do not establish the stopped starting configuration required
by the current projection approach.

One new capture is therefore the planned way to obtain that starting evidence.
It is **not needed for each LLM proposal**, and it will not, by itself, repair
the missing KMR `move_base` projection connection. Code repair and local replay
can proceed without Gazebo. Do not promise that recapture alone completes the
outline.

The checkpoint repair now implemented in `recovery_observations.py` does this:

1. Block new nominal dispatch and save the original fault observations and
   pending tasks before stopping handling motion.
2. Cancel owned handling motions through existing owner interfaces. Wait for
   controller inactivity and native observations confirming stopped roots.
   A cancellation acknowledgement or frozen simulation alone is insufficient.
3. Leave machine job commands untouched: do not issue M2 `pause_job`, `stop_job`,
   or a recovery event. M2 must still be processing at the captured checkpoint.
4. Freeze simulation time, obtain owner job observations, physical observations,
   planning scene, and model evidence for that checkpoint; preserve fault time
   separately from checkpoint time. Record later shutdown separately.
5. Report an ended M1/M2 processing overlap or missing evidence without silently
   altering timing or automatically repeating the trial. Preserve the unfinished
   KMR Storage task; cancelling its motion does not complete that task.

Relevant code: `_cancel_checkpoint_motion`, `capture_machine_checkpoint`, and
`checkpoint_fact_fingerprint` in
`cais_spade_llm/recovery_framework/recovery_observations.py`.
The cancellation evidence is included in new checkpoint fingerprints without
changing the fingerprint format of historical records lacking that field.
**This repair has local tests; it has not had a new live capture trial.**

### Physical projection and local parallel composition

Physical projection answers which configurations, custody, geometry, and process
facts follow from the proposed event. CCA uses those facts for the selected APs.
An M2 mutex verdict requires supported robot/tool/carried-part occupancy through
the relevant interval; a station name or clear TCP is insufficient. Processing
precedence uses exact supported part/process/result completion evidence.

Local parallel composition already exists in nominal CCA admission through
`environment_admission.py`, `environment_composition.py`, and
`agents/central_controller/local_composition.py`. `EnvironmentAdmission` calls
`analyze` before granting nominal dispatch and includes interacting work and
the selected monitors. Do not rebuild this nominal functionality.

The current offline outline instead checks each proposed transition with RA,
physical projection, and `validate_outline_evidence_safety`; it then uses
`validate_nominal_continuation` to check the recorded unfinished tasks from the
accepted projected state. It does **not** establish concurrent recovery × recovery
or recovery × nominal execution. That later runtime integration must not be
described as already complete, or confused with the separate primitive
composition stage. In a combined CCA check, supported physical facts supply the
AP valuations; composition cannot manufacture missing geometry.

### Remaining demonstrated blocker and next steps

1. **Complete the existing KMR owner projection connection locally.**
   `kmr_outline_evidence.py::_motion_request` still explicitly raises
   `KMR base movement needs recorded continuous base-controller and carried-part motion evidence`
   for `move_base`. RA success cannot bypass it. Earlier released-part collision
   identity and destination-parameter preparation defects were repaired locally.
   Reuse `kmr_motion.py`, the owner declarations, and the existing motion/geometry
   calculations; keep supported parameters and provenance separate from proposals.
2. **Use the native preview work already present, without overclaiming it.**
   `PreviewRecoveryMotion.action`, `preview_recovery_motion.cpp`, and
   `recovery_motion_preview.py` contain `base_trajectory` support. This is not yet
   connected into a complete KMR owner/CCA proof. A sampled native collision
   check alone is not continuous-motion evidence. Do not invent a controller
   trajectory, use arbitrary interpolation as recorded Nav2 behavior, or present
   a moving capture as stationary. Separate endpoint and interval coverage.
3. **Carry the whole supported scene forward.** Retain peer configurations,
   carried-part geometry, attachments, deposited parts, retained jobs, processing
   time, and CCA history. Inspect `_finish_environment_projection` in
   `resources/environment_recovery.py` when connecting KMR: propagation of changed
   models/geometry into `successor_evidence` needs verification, not an assumed
   pass. Running peer jobs must not silently freeze during projected intervals.
4. **Take one isolated capture after local preparation is ready.** Use the fixed
   checkpoint ordering and unchanged saved setup. Verify KMR stopped evidence,
   M2 still processing, scene/model compatibility, complete relevant observations,
   and source hashes before stopping the owned simulation. Keep a new directory;
   do not overwrite or relaunch the existing UI run.
5. **Replay saved proposals before paying for another API attempt.** Use exact
   captured M2 and KMR proposals as local replay input only. Verify separate RA,
   projection, and CCA results and unchanged accepted state/history after rejection.
   If facts remain unavailable, report their exact sources and gaps.
6. **Then run a fresh offline API attempt.** Freeze implementation, models,
   specifications, and capture hashes during each attempt. Allow successive LLM
   events and dynamic retrieval until nominal continuation passes or a specific
   blocker is recorded. Diagnose each new blocker locally before another attempt.
   Never insert recovery events or relax evidence requirements to finish.
7. **Accept only a complete independently audited outline.** Every accepted event
   must match a provider proposal and pass RA, physical projection, and CCA; the
   unfinished nominal obligations must have supported continuation. Save exact
   requests/responses, retrievals, timings, tokens, and all failures for UI review.

### Evidence and commands for the next chat

All paths below are relative to `/home/jongh/projects/cais-spade-llm`.

- Current immutable source:
  `cais_spade_llm/monitor/environment_runs/recovery/derived_dfa4017c0119403088d1d4f353a407d5_20261013_public/observations.json`
  — SHA-256 `870b2ff445fbc600c8d73fbac5c4c94ad0c66e5107002d73d783c24634676dfd`.
- API/capture history:
  `cais_spade_llm/monitor/environment_runs/machine_synchronized_capture_20261013T040945Z/`.
  Read `RESULT.md`, `RESULT.json`, and `attempt_07/report.json` selectively.
  Exact attempt-07 provider records are in
  `cais_spade_llm/monitor/environment_runs/recovery/offline_b593cb4f3d1b4ad0b9a1a6026c41efcd/generation_0001/recovery_outline/requests/`.
  Use provider `raw_response.choices[0].message.content`, not an enriched session
  response, when checking exact LLM authorship.
- Latest completed local KMR replay:
  `cais_spade_llm/monitor/environment_runs/kmr_offline_goal_20261013T120213Z/local_replay_parameters/report.json`.
  This replay makes zero API/live-interface calls; M2 passes, KMR stops at the
  missing continuous `move_base` projection. Source observations remain unchanged.
- Current checkpoint repair evidence:
  `cais_spade_llm/monitor/environment_runs/kmr_owner_base_20261013T124342Z/`.
  Read `checkpoint_repair_report.json`, `tests_checkpoint_motion.log`,
  `poetry_check.log`, and `compileall.log`. `before.json` retains source and safety
  fingerprints. No capture or API attempt was started in this handoff work.
- Existing runners in `machine_synchronized_capture_20261013T040945Z/`:
  `replay_attempt_07.py` and `offline_runner.py` take source and new output paths;
  inspect their arguments before reuse. `capture_runner_03.py` is a template with
  fixed `capture_03` output: copy it into a new attempt directory and change only
  its output destination. It checks the prior setup and restores its recorded seed.
  The isolated harness used ROS domain 89 and Gazebo port 11379; check availability
  and process ownership before reuse. Source `/opt/ros/humble/setup.bash` and
  `/home/jongh/ros2_ws/install/setup.bash` before the live harness. The earlier first
  capture failed when ROS imports were unavailable.

Latest local verification: **168 passed in 40.24 seconds** across
`test/test_recovery_evaluation_capture.py`, `test/test_machine_jobs.py`, and
`test/test_conveyor_fault.py`. The checkpoint integration fixture was corrected
to include its recorded run/resource identities and participant list; production
compatibility checks were not weakened. Poetry check passed with existing metadata
deprecation warnings; `poetry run python -m compileall -q cais_spade_llm ros2`
passed. Both selected safety files and the source capture still match their hashes.
The earlier 432-test owner-evidence result is separate and overlaps other suites;
do not add test counts together.

Local pytest previously stalled inside the restricted sandbox; the completed
suite ran outside it with mocked live/API interfaces. All owned test commands
from this handoff finished. No new Gazebo or API process was launched. Verify
host processes before any future launch; do not terminate unrelated processes.

**Suggested next-chat instruction:** Read this current handoff first, inspect the
existing KMR projection blocker and saved local replay, and continue the scoped
repair toward a complete machine-breakdown offline outline. Keep one agent and
factual prompts. Do not start with another unchanged API call or read every large
historical capture. Finish the missing code/evidence connection locally, obtain
the one required stopped checkpoint, then resume offline validation and generation.

### October 9: restore RA primitive support and independent capability validation

New explicit version-2 UI/offline attempts pin
`primitive_support_semantics="successor_primitive_union"`. RA matches declared
public resource successor conditions, derives the complete unordered primitive
union, and requires an independent parameter/capability witness for every
primitive against the unchanged starting snapshot. Resource declarations need
not contain `resource_state`; `held_part` has no special matching rule. Declared
bindings, domains and supported conditions retain their exact identifiers.
Primitive preconditions/effects and product end conditions remain composition
obligations. Outline validation no longer searches for an ordered primitive
program. The shared parameter-assignment limit is 4,096; incomplete evidence
or search remains `NEEDS_CONTEXT`.

Scoped, sourced parameter evidence is passed to resource-owned evaluators.
Validator retrieval updates the next validation inputs and terminates when
recorded facts do not change. Detached offline resources perform no equipment
or SPADE calls. Private validator evidence remains outside PA's public prompt.
Historical unmarked attempts retain their validation path; saved-result loading
preserves the recorded marker and evidence. New marked requests reject stale
replies using different primitive support semantics.

RA `FEASIBLE` is recorded separately from missing geometry and CCA outcomes.
An unordered primitive union supplies no ordered trace, physical successor or
motion interval. The exact selected specifications and CCA history requirements
remain enforced. Unresolved events do not append, project, advance nominal
continuation or authorize execution. Supplying independent physical projection
evidence to CCA remains subsequent work; this RA repair does not implement it.

The saved buffer proposal replay now matches `place_insert` and derives
`move_cartesian`, `release_part`, `move_relative`. `part_state="ready"` and
`part_location` are deferred. Its remaining `NEEDS_CONTEXT` concerns scoped
parameter/capability evidence, including `_task_ctx.insert_pose`, `model_name`,
`assembly_slot` and the values needed for `dz`. The original observations hash
is unchanged. This replay is recorded-data verification, not new LLM generation.

**Automated verification:** 502 recovery/context/offline/UI/request/evidence
tests passed. The full feasibility suite recorded 158 passes and 12 failures
in retired AP descriptor fixtures and KMR prepared-execution provider setup;
the recorded stacks do not enter the changed union code, and a pre-change
baseline replay was not performed. Poetry checks, Python compilation and diff
checks passed, with existing Poetry metadata warnings.

**Requested model comparison:** three fresh attempts each for `gpt-4o` and
`gpt-5.6-sol low` completed in
[comparisons/20261009T173155Z/REPORT.md](../../cais_spade_llm/monitor/environment_runs/recovery/comparisons/20261009T173155Z/REPORT.md).
The user authorized all six after the explicit disclosure question; automatic
approval review accepted the retry. These are six new attempts, separate from
the preceding comparison. All six ended `NEEDS_CONTEXT`, with zero accepted
events. Twenty-one exact provider requests and responses were captured; no API
errors or timeouts occurred. Initial prompts, schemas, the two selected safety
specifications, original capture and 343 frozen files remained unchanged.

`gpt-5.6-sol low` requested saved resource context and proposed `place` for
`KET4_Square_4mm` at `Buffer For Machined parts` in all three attempts. RA now
derives `move_cartesian`, `release_part`, `move_relative` from `place_insert`;
the remaining blocker is scoped parameter/capability evidence. Retrieval of
`/resources/recovery-resource-3@localhost/parameter_witnesses` returned no facts.
The capture contains no resource `parameter_witnesses` fields. Recorded parameter
references still depend on `_task_ctx.insert_pose`, `model_name`, `assembly_slot`
and `dz`; the offline capability path does not calculate or establish this
evidence. `gpt-4o` also reached missing parameter/capability evidence, after two
structural errors and four unsupported resource successor proposals across its
three trials. Complete model quality remains unmeasured because validation stops
before any accepted event.

No CCA call, projection, nominal continuation, primitive generation or execution
occurred. Mean elapsed time to the blocker was 11.419 seconds for `gpt-4o` and
17.074 seconds for `gpt-5.6-sol low`; mean total tokens were 13,306 and 10,171.
The next work remains connecting declared parameter references and owner
capability calculations to saved facts, plus independent event physical projection
for CCA. Parameter applicability also needs correction: the saved `place_insert`
support includes `assembly_slot`, but the owning `release_part` accepts that
optional parameter only for `assembly_board-v1`; buffer placement must not demand
assembly-only slot values. This is a code-inspection finding, not a controller
test. This comparison does not establish those missing checks. The offline
guards recorded zero live calls, and the current UI run was not interrupted.
**User-observed Gazebo recovery and remaining assembly completion remain pending.**

### October 9: three actual offline attempts per model

`comparisons/20261009T164725Z/REPORT.md` and `combined_comparison.json` record
**three attempts each for `gpt-4o` and `gpt-5.6-sol low`**, including the earlier
pilot pair. The user requested two further attempts per model despite the shared
pilot evidence blocker. Only the separately saved comparison runner's trial range
and experiment stop condition changed. Application code, initial messages and
schemas, selected safety specifications, source observations and validation
limits remained unchanged. No recovery sequence or event-count target was added.

| Model | Attempts | Proposed / accepted events | Public retrieval turns | Mean generation time | Total tokens | Complete validated outlines |
|---|---:|---:|---:|---:|---:|---:|
| `gpt-4o` | 3 | 3 / 0 | 0 | 6.975 s | 15,870 | 0 / 3 |
| `gpt-5.6-sol low` | 3 | 3 / 0 | 5 | 24.983 s | 39,935 | 0 / 3 |

All six attempts proposed one event on `recovery-resource-3@localhost` and
stopped at RA with `NEEDS_CONTEXT`. GPT-4o proposed
`recovery_pick_place_execution`, `align_gear_small` and
`transfer_misplaced_part_to_assembly` for `gear_small`. GPT-5.6 Sol proposed
`place` for `KET4_Square_4mm` at `Buffer For Machined parts` in attempts 1 and 3;
attempt 2 proposed `place_approach` toward `Exit` while retaining custody.
These are exact LLM proposals, not accepted recovery events or CCA verdicts.

The unresolved RA evidence set was identical: incomplete task programs and
primitive support, required owner evaluators, and incomplete parameter evidence.
No search exhausted the 4,096 limit and no primitive witness was derived.
Validator-driven retrieval returned saved model/snapshot declarations without
new evidence. Geometry projection, CCA, append/project and nominal continuation
were not reached. Outline safety is unresolved; execution safety is skipped.
Times and tokens describe progress to this blocker, with cached inputs in later
attempts. These results do not establish a winner for complete recovery.

All **17 provider requests and responses** were captured, without provider/parse
errors, timeouts or pending requests. The source hash and frozen application,
configuration and safety fingerprints remained unchanged. Guarded live-interface
and later-stage calls were zero; the current UI run was untouched.

**Automated verification:** continuation record checks passed for distinct runs,
input and source fidelity, exact proposals, provider parameters, complete request
outcomes and unresolved RA evidence. The prior 406 focused regression results
remain recorded separately; those suites were not rerun for unchanged application
code. The continuation runner compilation, Poetry checks, Python compilation and
diff checks passed; existing Poetry metadata warnings remain.
**User-observed Gazebo recovery and remaining assembly completion remain
pending.** The next evidence work remains exposing complete recorded resource-owned
task declarations to RA with provenance, followed by sufficient geometry,
attachment/motion and process-completion evidence for CCA and nominal continuation.

### October 9: actual gpt-4o / gpt-5.6-sol low comparison

`comparisons/20261009T162226Z/REPORT.md` records actual configured OpenAI API
attempts on the unchanged `1c3bec8faca6495cb1b0e22046277d5c` capture. GPT-4o omits
`reasoning_effort`; GPT-5.6 Sol sends `low`. Structured requests and summaries now
preserve that distinction, while saved historical settings remain unchanged.
Both models received identical initial messages and schemas with the two selected
safety specifications. Dynamic public retrieval remained enabled. No recovery
sequence or event-count target was supplied.

| Model | API calls | Public retrieval turns | Proposed / accepted events | Generation time | Total tokens |
|---|---:|---:|---:|---:|---:|
| `gpt-4o` | 2 | 0 | 1 / 0 | 6.317 s | 5,286 |
| `gpt-5.6-sol low` | 4 | 2 | 1 / 0 | 27.758 s | 15,015 |

GPT-4o proposed `recovery_pick_place_execution` for `gear_small` on
`recovery-resource-3@localhost`. GPT-5.6 Sol retrieved each robot's recorded
`static_capabilities` and proposed `place` for `KET4_Square_4mm` at
`Buffer For Machined parts`. Both stopped at RA abstract derivation with
`NEEDS_CONTEXT` and the same incomplete task/primitive declarations. The shared
missing complete task programs for `pick_approach`, `pick_grasp`, `place_approach`
and `place_insert` stopped further repetitions after the pilot pair. No supporting
primitive witness was derived. Geometry projection and CCA were not reached; no
event was appended and nominal continuation was not established. This is not a
complete validated recovery outline or a definitive model-quality ranking.

All six API requests and responses were captured, with zero provider/parse errors.
Source and code/configuration fingerprints remained unchanged; guarded live and
later-stage calls were zero. The actual initial grounding and outline messages
contained 4,570 and 11,426 characters respectively, excluding schemas. Outline
safety is unresolved, execution safety is skipped, and execution remains
unauthorized. The current UI run was untouched.

**Automated verification:** 165 model-parameter/request-record/offline tests and
241 context/validated-outline/geometry-evidence/workflow tests passed, totaling
406 focused tests. Comparison-summary checks, Poetry checks, compilation and
diff checks passed. Existing Poetry metadata warnings remain.

**Next evidence work:** expose complete recorded task declarations to the RA
validation path with preserved provenance, then obtain sufficient geometry,
attachment/motion and process-completion evidence for later CCA and nominal
continuation checks. Those missing facts were not fabricated. **User-observed
Gazebo acceptance and remaining assembly completion remain pending.**

### October 9: current offline comparison preparation and retrieval audit

The requested three-attempt `gpt-5.4 low` / `gpt-5.6-sol low` comparison is prepared
at `comparisons/20261009T153403Z/REPORT.md`, using the unchanged saved
`1c3bec8faca6495cb1b0e22046277d5c` capture and the current RA/CCA outline contract.
No new provider request has been sent: automatic approval review rejected the
external API disclosure, and explicit approval is pending. These preparation and
local replay records are not another completed model comparison.

PA still supplies the main facts, public capabilities and supported context
references; `need_context` retrieves recorded facts. The audit fixed projected
unknown part facts falling back to saved facts, duplicated/unsanitized retrieval
text, and omitted exact resource/custom part goals. The focused context,
validated-outline and request-record suites passed **231 tests**; Poetry,
compilation and diff checks passed. Original previews and observations remain
preserved.

Local replay of a previous exact LLM buffer-placement proposal, with its starting
state derived through the current runtime path, stops at RA with `NEEDS_CONTEXT`
for incomplete primitive/task declarations. CCA is not reached and no event is
appended. The capture also lacks the complete projected robot/tool/carried-part
and motion evidence required for later geometry and CCA checks. This is an
evidence limit, not a new model-comparison result. API generation, user-observed
Gazebo acceptance and remaining assembly completion remain unverified.

## 2026-10-09: RA-derived recovery outline evidence and CCA validation

New generation attempts use `validation_contract_version=2` and follow
**Capture facts → one LLM `next_transition` → ResourceAgent backward derivation
and forward validation → CCA validation → append and project → nominal
continuation validation**. Historical attempts retain their original contract
and status. The ProductAgent no longer supplies custody effects or ranks/inserts
recovery events on this path. The proposed event name remains an identifier;
the resource derives support from declared primitives and, when fully available,
complete task composition contracts. Search is bounded to 4,096 state/binding
combinations. Unsupported semantics, missing facts and incomplete searches remain
`NEEDS_CONTEXT`; a complete negative result or established contradiction is
`INFEASIBLE`.

The shared live/detached RA path records ordered prerequisites, bindings,
intermediate effects, model/source fingerprints and projected resource/part state.
Non-executing owner previews use an isolated recorded scene. Captures include
configured physical observations, joint configurations, tool/part geometry and
attachments when available. A held part is supported by actual attachment evidence.
Missing resources cannot disappear from the CCA population. FK configurations,
carried parts and deposited-part poses carry forward between events. A named-pose
endpoint does not prove an entire movement safe. Current native interfaces without
an isolated planning scene are not used as a substitute for future motion evidence.

CCA uses unchanged `SAFE_shared_area_mutex` and
`SAFE_gear_small_before_KET4_Square_4mm` specifications. Each rule retains its AP
namespace and monitor history. Robot/tool/carried-part geometry determines
occupancy; `processCompleted` requires its complete recorded ledger or a supported
declared completion effect. Unknown occupancy is retained. Existing active CCA
history without a compatible recorded outline context blocks prospective reset.
Only a successful RA and CCA result commits the projected state and monitor state.
Nominal continuation must use the exact saved nominal task contract, respect
dependencies and discharge monitor obligations; a missing task index is not
completion.

Public `need_context` retrieval remains available to the LLM. Validator requests
are recorded separately and retry the same proposal only when new evidence is
available. Offline uses saved facts and local calculations, with no live interfaces.
Fresh initial observations cannot replace a predicted state after accepted events.
Per-turn records retain derivation, geometry coverage, retrieval exchanges, CCA
results and nominal-continuation evidence. The UI keeps the proposal separate from
these records and labels outline safety separately from execution validation.
Offline remains Manual. New outlines/proposals cannot enter the diagnostic
execution or saved-result Gazebo dispatch path.

**Automated results:** the combined outline integration, geometry, capture,
resource preparation, context, offline and workflow suites passed 579 tests. The
UI/request-record/setup suites passed 201 tests. RA focused checks passed 89 tests,
including 52 new derivation cases. The separate geometry/continuous-motion suites
passed 68 tests. `poetry check` and Python
compilation passed; Poetry reports its existing metadata deprecation warnings.
After the final public-feedback, retrieval-progress and stale-model fixes, the
five outline/context/offline/workflow suites passed 339 tests; the context/request
record suites passed 198 tests. These overlap the combined suites above and are
not an additional distinct-test total. Rejection prompts show exact violated rule
identifiers and AP values while omitting private models and primitive details.
Restricted-sandbox asyncio callbacks stalled even in a minimal standard-library
reproduction; mocked asynchronous suites ran successfully outside that sandbox.
The broader RA suite also reported 12 failures before/in legacy AP and KMR provider
fixture paths, separately from the new derivation tests: five slash-AP preparation
failures and seven unavailable KMR prepared-execution provider failures.

The original `1c3bec8faca6495cb1b0e22046277d5c/observations.json` SHA256 remains
`d92fe0228978431eaf2343d26f7c7fbe47322b9a81dc4cad3f713850d04110b2`.
That older capture is not automatically sufficient for this milestone: absent
joint/attachment/motion evidence and incomplete native completion contracts remain
explicit blockers. **User-observed acceptance:** pending through the UI. No live
LLM request, Gazebo launch, recovery execution or current-run restart was performed
for this implementation.

### Subsequent runtime work — not implemented by this milestone

- Validate complete primitive compositions against the accepted outline evidence.
- Check recovery × recovery and recovery × nominal before dispatch and against
  observed execution; retain shared CCA history and resource/region ownership.
- Bind execution permission to exact commands, models and observations through
  recovery and resumed nominal work.
- Handle stale evidence, deviations, failure and Stop with blocked admission,
  controller cancellation, and observed completion or clearance.
- Add concurrent execution only with adequate combined motion and timing evidence.

**Historical implementation, 2026-10-08:** the Recovery page follows **Example →
Safety → Recovery outline → Primitive composition →
Execution**. Details and its legacy controls are no longer rendered; existing
records remain. Example offers Gazebo / Offline, the four unchanged scenario
names, and Start failure with Part slippage initially selected. All four examples
use their existing failure checkpoints and retain observations; recovery
generation and execution currently support Part slippage. One Recovery mode
Auto/Manual control in Example applies to both Recovery outline and Primitive
composition. Gazebo defaults to Auto and stops before execution; Offline always
uses Manual, with the shared control disabled. Manual retains the Generate/review
steps in each stage. SystemBridge remains the public interface; background
workflow ownership survives page navigation. The Current observations card is
removed. Example retains the capture status, source run, capture time and
Download observations; Retry input preparation appears with the outline error.

Offline follows **Select saved observations → Generate → Review**, without a
Start offline test button. Captures appear newest first by capture time. Selecting
one prepares an input preview marked **not sent** without calling the LLM. Outline
Generate is available after preparation; disabled controls show the actual reason.
Offline pauses for review and Continue after the outline, then permits primitive
Generate. Each explicit Generate completes that stage's retrieval turns. Switching
back to Gazebo restores its previous mode selection. Stop followed by Generate
creates a separate attempt from the selected capture; obsolete preparations are
cancelled when the selection changes. Historical run status does not own work.

This first functional experiment explicitly uses `execution_mode="simulation"`,
`diagnostic_cca_bypass=true`, `runtime_recovery_validation_policy="no_validation"`
and no active safety requirement. Part slippage automatically selects
`safety_assembly_board-v1_mutex.txt` and
`safety_assembly_board-v1_gear_small_before_KET4_Square_4mm.txt` in the Safety
panel. These selections remain editable metadata for later integration without
activating checks or adding them to prompts. Formal safety
and CCA admission are **skipped**. Offline validation retains schema, grounding,
declared transitions, capabilities and primitive state projections. Fresh
controller and trajectory checks are **deferred**, not passed. Test in Gazebo
checks source identity, configuration, unfinished tasks, custody, fresh poses,
controller readiness and primitive feasibility before enabling Execute recovery;
the explicit execution click rechecks imported proposals. Mismatches require
regeneration and are not silently rewritten. Simulation/controller guards reject
hardware execution. Confirmed primitive completion precedes nominal resumption.

Observations are saved before preparation or LLM calls. Preparation uses
registered `executables` and saved task definitions; errors leave the capture
available for download and retryable. Complete captures contain the prepared context, resource
models, primitive catalogs, configuration, geometry, task definitions and goals
needed by both stages. Offline attempts use detached saved resources, with no
Gazebo, controller, perception or SPADE transport calls; LLM generation still
uses the configured API and existing templates. Each attempt has separate
artifacts under `cais_spade_llm/monitor/environment_runs/recovery/<run_id>/`.
Exact request/tool/response captures, prepared input marked not sent and original
downloads appear in chronological turns. Each Input and Output expands independently
and loads its content on demand. Expansion choices survive polling, navigation
and reconnection, scoped to the run, generation, stage and turn. Retries and
provider tool rounds remain grouped; primitive turns from different resources
remain separate even when their local numbers repeat. Legacy summaries are
identified explicitly when exact captures are unavailable. Load saved result restores a
completed composition after application restart without another LLM call.

New outline sessions use `decision="need_context"` without committing candidate
events. The initial prompt selects observed failure, custody, unfinished goals
and interacting work from recorded relationships, and keeps alternative resources
discoverable through a compact capability index. Detailed public declarations,
poses, target geometry, pending tasks and primitive contracts are retrieved by
supported references. Saved facts are used first; offline retrieval never calls
live interfaces. Missing facts and repeated requests without new information stop
with `NEEDS_CONTEXT`. Full captures and validator context remain intact. Each turn
records context requests, returned facts, missing facts, inclusion reasons and
provenance separately from the authored proposal. Live revision changes invalidate
dependent generation; accepted transitions recompute projected prompt facts.

**Automated verification for dynamic retrieval and expandable turns:** 143 focused
tests passed, plus 11 existing prompt/schema/context/retrieval regressions. Poetry
and Python compilation passed. Focused Ruff checks passed for the new context
module and affected UI modules; broader checks of the existing test files retain
seven preexisting style findings. No live model call or equipment action was used.
On the same saved capture, outline prompt text decreased from 10,193 to 6,409
characters and grounding text from 6,358 to 3,367. These are offline rendering
measurements, not model-quality or live recovery results.

**Previous offline verification:** 118 focused workflow, offline, page, request-record
and diagnostic tests passed. Another 359 setup, saved-program, mocked Gazebo and
environment-admission tests passed. The separate feasibility suite completed
with 46 passed and 12 failed: five legacy AP fixture failures and seven KMR
prepared-execution fixture failures, outside the new offline path. Poetry,
compilation, UI CLI help, focused Ruff checks and diff checking passed.

**User-observed Gazebo acceptance: pending.** No live LLM generation, Gazebo
launch or recovery execution was initiated for this update. Earlier trials
observed both pickups and `gear_small` slippage and generated some outlines, but
did not execute generated recovery primitives or establish final assembly.
Older `environment_run.json` files remain viewable. In particular,
`aa8e27d245584008b593c84111daa47e` lacks a prepared recovery request, so it reports
`NEEDS_CONTEXT` for offline generation and requires a fresh complete capture.
The next gate belongs to the UI operator: capture failure → stop Gazebo →
generate both stages offline → expand compact inputs and retrieval outputs →
Test in Gazebo → explicitly Execute recovery →
observe recovery and remaining assembly completion, with safety checks deferred.

**Earlier status reviewed on 2026-10-08 (CCA integration):** nominal and recovery physical admission now
share `RecoveryCompositionAdmission`, its admission lock, command ledger,
reservations and physical execution history. `EnvironmentAdmission` retains
nominal eligibility and joint-completion authority through `commit_prepared`.
Exact prepared command ownership and authenticated feedback plumbing are
implemented. The current target is the user's four-step model-based Gazebo
method: fixed AP definitions, recovery events mapped to known primitives,
intermediate motion and modeled failure AP evaluation, and local composition with
relevant nominal operations. Stronger physical guarantees and automatic nominal
resumption are explicitly outside this implementation. Background Gazebo checks captured 11 controllers
and all 12 configured resources with both predefined specifications active and
`diagnostic_cca_bypass=false`. The supplied single-motion path now reaches an
allowed common composition graph with 67 instantiated rules and a committed
CCA grant. In `acceptance-11`, the native controller executed the exact authorized
trajectory and CCA committed authenticated observed completion once. This passes
the first empty-custody supplied `move_cartesian` gate under explicit models;
`physical_execution_verified` remains false. Earlier failed feedback and its
reservation remain recorded without retroactive completion. Both dedicated Gazebo
launches are stopped. Live occupied-region rejection, ongoing nominal composition,
modeled failure branches and retained history across successive executions remain
unfinished. Full `part_slippage` recovery and the four failure scenarios have not
been established.

**Roadmap maintenance requirement:** after every implementation or verification
update, synchronize the current status and dependencies in this file and append
a dated entry containing changes, checks performed, evidence paths, remaining
blockers and the next acceptance gate. This is the single current roadmap; retain
historical results as dated evidence rather than treating them as current results.

The earlier CCA implementation grounds the existing CCA path for nominal and
recovery execution under explicit motion and failure-model assumptions, preserving
the Gazebo plant and actuator behavior. CCA must still evaluate intermediate motion,
include relevant nominal work, bind grants to exact prepared commands, and retain
observed monitor history. Missing model inputs remain `NEEDS_CONTEXT`. Model-based
admission does not assert proven bounds on every physical deviation, contact or
stopping motion. Use background Gazebo on school WSL. Actual recovery-event
generation for the four failure scenarios follows validation of this CCA path;
automatic nominal resumption and full physical guarantees were later milestones
for that CCA work. The functional Part slippage workflow above is the current
implementation scope.

**Historical status reviewed on 2026-10-05 (superseded where stated above):
paper-aligned RA derivation, CCA outline checks,
and diagnostic evidence are implemented; local parallel composition is implemented
for nominal execution through `EnvironmentRuntime`. The new offline slice implements
reusable reviewed LTLf checking over primitive observations, automatic grounding
from the complete frozen scene, and synthetic KMR recovery evidence. Existing
structured `ap_event` / `ap_state` descriptors connect through explicit task/state
evidence without changing their identities or substituting occupancy. Bounded offline
local parallel composition now connects these observations to generated recovery
events and included already-running work. The `validated` route now registers complete
recovery sequences with CCA and requires an event-specific composition grant before
RA invokes primitives. Predefined AP meanings, formulas, and scopes are now a
reviewed input assumption. A selectable document compiles deterministically and
CCA uses the same fixed definitions for grounded nominal/recovery admission;
predefined safety never calls the LLM to replace them. This connection is verified
with explicit mock executors. Resource-owned non-dispatching preparation and the
recovery page's Prepare and check action now save a detached CCA result. Registered
`scene.resource_models` and owner-provided pure models support additional resources
without changes to AP evaluation, LTLf compilation, or local composition. Existing
UR5e controller preparation and KMR offline evidence remain supported. The supplied-motion
slice adds a read-only Gazebo attachment/state reader, registered preparation assembly,
and continuous joint-motion bounds with uncertainty in the same composition search.
Its focused software regressions pass; the dedicated Gazebo attempts remain
`NEEDS_CONTEXT`, so safe/conflicting live-motion acceptance is explicitly incomplete.
Active-rule Gazebo execution, nominal resumption, and the legacy
`receiving_region_entry` counterexample remain open.**

The current focused experiment uses three supplied mock `part_slippage` plans
and synthetic primitive observations with the unchanged predefined specifications.
The [computed report](../../test/fixtures/part_slippage/predefined_safety/REPORT.md)
records a mutex-only rejection, a precedence-only rejection, and satisfaction of
both rules for the correctly ordered trace. This offline experiment does not
depend on Gazebo preparation; it tests checking newly named recovery behavior.
The mock inputs use the existing offline observation contract and are not native
UR5e controller dispatch programs or newly collected LLM outputs.
The subsequent [Gazebo companions](../../test/fixtures/part_slippage/gazebo/README.md)
preserve those identities. Their live preflight captured all 12 configured resources
and resolved the static board envelope, but did not stage or execute any trial.
The requested output is now **one silent, captioned 20× video**, showing the
separately staged mutex, precedence and safe trials in sequence. Both predefined
specifications must be active together in every trial. The combined exporter is
implemented and tested with encoding fixtures; the live video remains unproduced.
Missing live evidence is not a mutex or precedence counterexample.
The 2026-09-29 source baseline was `c485e51`; the 2026-10-03 review inspected
`e64d694` and the new, untracked offline primitive safety files. This is the sole current
roadmap for Journal Paper 2. Software checks do not establish live recovery or
hardware safety.

**Prepare and check** now implements capture from registered resource owners,
preparation through their native contracts, and checking supported evidence using
a detached CCA snapshot. Synthetic machine, Conveyor, Buffer, seven-joint and
additional-resource providers exercise the common path. The first supplied candidates
use the existing UR5e controllers; configured fixed equipment and stationary KMR
now expose stamped physics observations through registered owners. Continuous
bounds preserve prepared joint interpolation and do not infer straight TCP paths.
The earlier live preparation attempts were blocked before guarded execution:
observed link motion, missing controller idle records and incomplete part geometry
blocked those sessions; a later retry also timed out reading the service. Those
single-motion blockers are superseded by the 2026-10-08 model-based acceptance
recorded above. Missing geometry, custody evidence, stationary
coverage, or supported motion interpolation remains `NEEDS_CONTEXT`. It cannot
install an execution proof, grant permission, or advance either monitor history.
The later part_slippage captures obtain the board envelope from its explicitly
configured static fixture models. An opt-in read-only controller query now reports
holding, active/pending goals, controller incarnation and command revision. A
dedicated launch with navigation disabled returned all 11 controller observations
covering the 17 configured action endpoints, alongside the complete physics
snapshot. This establishes current idle evidence, not future stationary coverage.
Fresh checkpoint validation and owner model evidence are exercised by the first
supplied-motion acceptance; prepared native custody execution remains open.
Pure continuous owner models now support explicit grasp,
carried-part geometry and release effects in synthetic checking; the live controller
does not yet supply or execute that complete contract.

## Summary

Verify newly generated recovery behavior while preserving existing specifications,
ongoing work, and valid nominal continuation. Assume that safety specifications,
AP meanings, and formulas are supplied and reviewed, as in the journal paper.
The LLM may propose recovery behavior; it does not discover or reinterpret these
selected specifications. Establish declared trace semantics,
resource-owned primitive effects, reusable formula evaluation, and deterministic
applicability, then bounded offline recovery composition using the existing local
parallel composition machinery and authoritative CCA admission. Evaluate supplied
mock recovery plans with synthetic evidence independently of live preparation.
Complete live preparation and execution-observation contracts before executing
four recovery scenarios. Use Gazebo for execution
validation; hardware experiments remain later work.

A recorded prompt in the UI must be the complete application-controlled request
actually submitted to the model. Expected recovery answers remain outside model
inputs. Prompts are read-only in the UI. Dry runs support reviewed fixtures and
saved Gazebo failure snapshots.

Use the [formal reference](JOURNAL_VALIDATION_AND_SELECTOR_ACTIONS.md) for model
semantics and claim boundaries, [the document index](README.md) for supporting
references, and [implementation history](IMPLEMENTATION_HISTORY.md) for prior
results. The supplied manuscript excerpt is authoritative for feasibility and
outline safety; its exact text is preserved in the
[formal reference](JOURNAL_VALIDATION_AND_SELECTOR_ACTIONS.md#supplied-manuscript-excerpt).
Selector policy, deterministic safety construction, and local composition are
separate implementation decisions. PA schema checks remain preparation, rather
than a new paper-level validation contribution.

## Current status and dependencies

| Area | Observed baseline | Remaining milestone |
| --- | --- | --- |
| Nominal Gazebo execution | Retained 11-part run with 195 matching transitions; machining uses simulated processing and gears start preprinted. | Establish a fresh baseline with active rules and CCA bypass disabled before recovery experiments. |
| Feasibility and DFA validation | Union of all successor-support primitives; witnesses for every distinct primitive; shared strict DFA transitions, prefix continuation checks, and detailed findings. | Extend physical witnesses and resource contracts to all four larger-setup scenarios; selector correction remains a separate work item. |
| Recovery safety generation | Legacy offline `receiving_region_entry` and `shared_area_mutex` use fixed DFAs and 1D observations. Automatic offline grounding now derives the complete population from `build_environment_models(scene)`: 12 resources and 66 mutex pairs, without participant lists or exclusions. Reviewed scopes and explicit physical/task/state evidence preserve AP meanings. Predefined mode compiles the supplied definitions unchanged and uses them throughout CCA; legacy natural-language mode retains LLM rule/event selection. | 2: resolve the legacy counterexample; validate reviewed hazard coverage and trusted evidence for live admission. |
| CCA composition | As of 2026-10-08, `RecoveryCompositionAdmission` shares nominal/recovery physical admission, command authorization, reservations and history; `EnvironmentAdmission.commit_prepared` retains nominal commitment. Local parallel composition is implemented for nominal `EnvironmentRuntime` execution, including dependency closure, joint completion analysis, caching, snapshot validation, and atomic CCA admission. A separate offline path composes generated recovery events with fixed running work and reviewed primitive observations using finite supplied start alternatives. The `validated` PA → RA → CCA route now registers the sequence and gates each requested start against the retained composition and observed history. Global plan FSA and active-window runtime paths remain. The first supplied empty-custody `move_cartesian` now passed live common composition, exact authorization, execution and one observed completion with both selected specifications active and bypass disabled. | 3: validate occupied-region rejection in Gazebo, then connect relevant moving nominal work, remaining recovery events, modeled failure AP observations and retained continuous history across live programs. Full physical guarantees and automatic nominal resumption are later milestones. `X` remains unsupported. |
| KMR recovery preparation | The synthetic `storage_interruption` fixture has four events and 22 primitives: stage `KET8_Square_8mm`, recover `KET4_Square_4mm`, then reacquire `KET8_Square_8mm`. Companions supply complete 12-resource evidence and two explicit timing alternatives with already-running `ur5e-3` withdrawal. Original fixture JSONs are retained. | Establish physical feasibility and later pending M1 delivery separately; the offline endpoint restores the Storage checkpoint. |
| Mock part_slippage safety experiment | Three supplied plans each contain seven newly identified events and 19 modeled primitives. The unchanged predefined mutex and strict precedence produce the expected separate violations and a complete safe trace. All 12 configured resources and 66 mutex pairs are included; event renaming preserves physical AP values. Separate Gazebo companions retain these identifiers; the combined 20× exporter preserves distinct trial evidence. | Complete native preparation, staged checkpoints and observed CCA-granted execution before publishing one video of the three trials with both specifications active. No live violation or successful recovery has been established. Actual LLM generation remains a separate evaluation. |
| Diagnostic UI | Recovery page has Gazebo / Offline, four failure scenarios, recorded Safety selections, visible observations, independent Auto/Manual generation stages and exact prompt inspection. Complete saved captures support generation without equipment or SPADE transport; each attempt has independent records and completed results can be loaded after restart. Test in Gazebo checks fresh compatibility before explicit execution. Details is removed; existing `scenario_runner` records and backend diagnostic tools remain. | Operator acceptance: capture failure, stop Gazebo, generate both stages offline, return to Gazebo, explicitly execute and observe remaining assembly completion. Generation/execution for the other three scenarios and active Safety integration remain later work. |
| Physical failure scenarios | Simulation failure injection support is implemented; the [2026-10-02 validation notes](../../docs/conveyor_breakdown.md#validation-on-2026-10-02) record blocked checkpoint attempts. | 5: establish checkpoint acceptance with CCA enabled, then recovery execution, observation, and resumption. |
| Resource-owned preparation | Recovery-page Prepare and check uses common owner hooks and a registered entity reader. `scene.resource_models` appends trusted declarations and preserves existing owner identities; native programs stay in `resource_programs.resources`. Pure models provide supported physical effects independently of AP meanings. A synthetic additional resource produces 78 mutex pairs; equipment and seven-joint mocks use the same checking path. Existing UR5e `move_cartesian` planning remains in its controller. The read-only Gazebo service and configured owner readers capture the complete population; continuous polynomial/FK bounds feed the same composition path. Static constituent geometry resolves the empty board carrier envelope. The opt-in controller query establishes current idle ownership; synthetic continuous custody effects and explicit joint-error envelopes extend detached checking. | Exact prepared command ownership and authenticated feedback passed the first supplied live motion. Physical tracking, future stationary containment and stopping guarantees remain unavailable and outside this scope; the accepted path uses explicit models. Complete conflicting supplied-motion acceptance, then native grasp/release, observed carried-part transforms and required KMR/machine/Conveyor/Buffer effects. Registration and idle observations alone supply no physical execution guarantee. |
| Paper evaluation | Protocol specified here; no new trials collected by this change. | Experimental design after the relevant acceptance gates. |

The 2026-09-29 baseline settings selected `safety_none.txt`, `diagnostic_cca_bypass=true`,
`neurosymbolic`, `action_horizon=1`, `candidate_count="adaptive"`, and
`candidate_proposal_budget=5`. Historical bypass runs establish execution evidence,
not active-rule CCA safety. See [the retained run](../../cais_spade_llm/monitor/recovery_gazebo_runs/attempt-f2090c8976d84acd90cf35353c5acc34/README.md).

The RA → CCA → UI implementation slice is available for saved-context inspection.
Milestone 2 treats the reviewed AP/formula catalog as an input assumption, then
proceeds through trace semantics, resource-owned effects, deterministic compilation,
and automatic applicability. Safety-page compilation, saved artifacts, bundles,
and CCA preserve this same predefined source and its fingerprints. Automatic population and binding
construction now covers every configured resource, including additional registered
`scene.resource_models`, with no exclusion pruning. New supported actions enter
through trusted owner contracts. Missing declarations/providers/evidence remain
unavailable; no shared AP or composition branch dispatches on a new resource name.
Task/state evidence adds boundaries to the same frozen physical trace; pending
requirements and explicit finite-trace completion remain visible. The legacy receiving-region
counterexample and full reviewed hazard coverage remain open. Nominal local parallel
composition already exists. The offline milestone 3 path now uses grounded primitive
evidence and replayed monitor history for finite generated recovery alternatives;
CCA now adds the authoritative generated-event admission gate on the existing
`validated` route. Owner-supplied live geometry, trajectories, timing, acknowledgements,
and observation support remain required; missing evidence blocks admission. The
runtime recovery path still builds a global FSA after inserting recovery macros.

Supply reviewed staging/grasp/transport contracts and failure snapshots alongside
this work, including physical evidence that the current validators report as
`NEEDS_CONTEXT`. Milestone 4's remaining scenario fixtures can proceed alongside
2–3; its current `safety` stage still records the existing LLM-selected grounding
method. Milestone 5 depends on the relevant capabilities and validated safety
paths. The supplied mock `part_slippage` comparison uses the existing offline
checker and does not require live observation, failure injection, or an LLM call.
For later live trials, first verify non-dispatching preparation through resource-owned contracts,
including all configured resources and unchanged CCA histories. The reusable path
is tested with synthetic providers. Supplied recovery events and programs are the test
inputs; neither failure injection nor recovery generation is needed for this next gate.
The dedicated live session collected observations but did not establish either a safe
moving candidate or a definite mutex counterexample. Complete those two checks and
the missing-evidence negative case before guarded execution.
For the requested three-case part_slippage recording, use separately staged,
observed post-slippage checkpoints and preserve the original mock identities.
Keep both specifications in the same composition for all three cases. Publish
one 20× video with clearly labeled trial boundaries; keep each run's original
CCA decisions and observations separately. Encoding tests do not establish any
of those live outcomes.
The staging does not demonstrate the drop. Missing live preparation/admission
support must leave acceptance incomplete; publish no substitute safety-rejection
video for `NEEDS_CONTEXT`.
Establish a fresh nominal Gazebo baseline with active rules and
`diagnostic_cca_bypass=false` before recovery/resumption experiments. Collect
confirmatory trials only after all gates relevant to their study pass.

Preserve `SystemBridge`, exact project symbols, current evidence, and the
PA/RA/CCA authority split. Positions, capabilities, and resource bindings remain
configuration-driven. Do not change the separate Spec2Primitives subsystem for
this roadmap.

## Milestone 1: Formal validation contract

Retain `G = G_P || (||_{r in R} G_r)` and the
[stage contract](JOURNAL_VALIDATION_AND_SELECTOR_ACTIONS.md#formal-model-and-authority):

| Stage | Authority | Required behavior |
| --- | --- | --- |
| Symbolic feasibility | PA | Exact declared-start agreement, valid successor effects, `part_traceability`, and fresh state. |
| Primitive feasibility | Responsible RA | Establish primitive support and parameter/capability witnesses; distinguish `FEASIBLE`, `INFEASIBLE`, and `NEEDS_CONTEXT`. |
| Primitive composition | Responsible RA | Check sequence ordering, compatible parameters, and achievement of the complete intended successor. |
| Safety | CCA | Evaluate applicable LTLf DFAs using projected history, running events, persistent state, and pending obligations. |
| Execution and commit | RA, PA, CCA | Revalidate relevant evidence and commit only acknowledged, observed effects. |

### Implemented paper-aligned validation slice

PA proposes new recovery events from projected conditions and public capabilities,
without receiving the complete private RA transition model. RA derives support
from its own successor valuations; neither the proposed event nor its starting
condition needs to be a predefined task transition.

- Collect the union of primitives from **all** matching successor transitions.
  Deduplicate by primitive identity and retain every source transition, binding,
  and covered valuation field. Require a parameter/capability witness for every
  collected primitive. A known empty union is rejected; missing descriptors,
  observations, evaluators, or incompletely searched domains remain `NEEDS_CONTEXT`.
- Return the union and evidence to PA and composition. PA verifies the descriptor,
  exact proposal, sources, and primitive identities. Feasible support does not
  establish compatible ordering or complete product effects; those remain explicit
  composition obligations. Private primitive preconditions and conditional steps
  remain in the contract evidence.
- A `release_part` effect can clear `held_part`; it cannot establish staging by
  itself. Composition needs a bound target and motion to that same target. Motion
  away invalidates the earlier landing fact. Actual support/contact and final
  part location still require physical evidence and observed execution.
- CCA deterministically projects each successor label, including relevant running
  tasks and persistent state. Every applicable DFA must take one unambiguous
  transition and retain a path to acceptance; an unfinished prefix need not already
  be accepting. Empty steps remain in the trace. Missing rules, mappings, required
  fields, or malformed transitions cannot authorize admission.
- Rejected candidates leave the accepted prefix, projected state, and monitor
  state unchanged. Records retain each rule, requirement, label, transition,
  acceptance/reachability finding, and rejection reason. Outline checks do not
  certify the intermediates of an ungenerated primitive composition.

**Verified scope:** parameterized/novel successor matching, primitive-union
semantics, missing witnesses, airborne-release rejection, DFA continuation and
malformed transitions, and evidence inspection. The
[contract audit](JOURNAL_VALIDATION_AND_SELECTOR_ACTIONS.md#resource-coverage-and-remaining-limitations)
identifies physical and resource-specific gaps. Existing runtime interfaces are
retained. PA schema checks remain implementation preparation.

### Separate selector correction (planned)

Correct strict-expansion selection: event swaps cannot count as progress.
Incorporate CCA admissibility within the documented enabled-event sets rather
than an additional undocumented progression rule. Preserve arbitrary but stable
exact-effect tie resolution and the one-action horizon; make no cost-optimality
claim. This correction is not part of the manuscript's feasibility algorithm.

**Acceptance:** strict expansion, event swaps, event loss, unchanged sets,
obligation changes, and nominal reentry have regression coverage.

## Milestone 2: Deterministic safety construction

### Implemented offline primitive checking (2026-10-03)

The [specifications and evidence contract](../../cais_spade_llm/specification/safety/primitive_observation_safety.md)
define `receiving_region_entry` and `shared_area_mutex` with explicit resource,
part, and region bindings. The
[offline CCA checker](../../cais_spade_llm/agents/central_controller/primitive_program_safety.py)
instantiates concrete bindings and the two fixed specification DFAs deterministically,
without an LLM call. It does not provide arbitrary LTLf formula compilation.

- [Resource-owned primitive observations](../../cais_spade_llm/resources/primitive_observations.py)
  model supplied `compute_place_targets`, `move_cartesian`, and `release_part`
  evidence under configurable 1D geometry. Joint trajectories share one time axis;
  checking includes the initial state, boundary crossings, and intervals between
  crossings. Closed intervals count contact as occupancy.
- `BaseSafetyChecker.transition_evidence` checks AP valuations separately for each
  `rule_id`. Missing evidence remains `NEEDS_CONTEXT`; rejected or unresolved
  candidates preserve the supplied DFA history. Accepted modeled snapshots retain
  occupancy, custody, and deposited-part containment between programs.
- Applicable bindings and resource-model evidence are supplied by the caller.
  Omitted requirements are not discovered or checked. Capacity checking and
  runtime admission integration remain deferred. This legacy checker can
  inspect supported nominal or generated recovery primitive programs offline;
  passing does not authorize execution.

The implemented 2026-10-04 offline extension separates these four stages:

1. Freeze one joint observation trace: initial state, primitive/trajectory/custody
   boundaries, region contact, and representative observations between boundaries.
   `X` means the next observation in this declared clock. Version and fingerprint
   the clock, scope, geometry, AP meanings, and compiled formulas. Continue only
   through contiguous slices of that same trace; completion adds no observation.
2. Add opt-in `geometry.dimension: 3` through
   [resource-owned 3D effects](../../cais_spade_llm/resources/primitive_observations_3d.py):
   fixed-orientation axis-aligned envelopes, piecewise-linear XYZ trajectories,
   separate KMR base/tool motion, release, stationary deposition, and reacquisition.
   Preserve `processCompleted`; modeled placement never establishes `assembly`.
3. Add [reviewed formula evaluation](../../cais_spade_llm/agents/central_controller/reviewed_primitive_program_safety.py)
   using `LTLfParser`/MONA without LLM calls or preview writes. Fixed AP evaluators
   bind exact identifiers; formulas may use `G`, `F`, `X`, and `U`. Results distinguish
   `prefix_checked`, `satisfied`, `violated`, and `unavailable`, with pending rule IDs.
   Satisfaction requires explicit completion at the actual trace end. A checked
   prefix does not establish a physically feasible joint continuation.
4. The first reviewed slice derived all 10 `shared_area_mutex` pairs from the four
   `ur5e` resources and `KMR` in its caller-declared study population. It required every selected catalog specification,
   participant, and motion/stationary witness. Apply the separate requirement
   “KET4_Square_4mm may enter assembly_board-v1 only after its trim with result: square is completed.”
   Ground entry from part geometry and completion from the exact process/result
   record in an explicitly complete checkpoint ledger. Missing evidence is unresolved.

The subsequent automatic grounding slice adds
[`validate_grounded_primitive_program_safety`](../../cais_spade_llm/agents/central_controller/offline_safety_grounding.py).
It freezes the supplied scene and uses every `build_environment_models(scene)` key:
all 12 resources generate 66 mutex instances. Reviewed `requirement_scopes` supply
the applicable region, part, process, and result; callers supply no resource pairs,
participant allowlists, or exclusion declarations. Every configured resource needs
geometry, a snapshot, and complete motion/stationary evidence. Fixed equipment uses
`stationary_only: true` without invented gripper states; resources declaring
`held_part` cannot use that flag to omit custody evidence. A declaration-only builder
extension permits an additional configured robot without inventing a nominal handling
assignment; 13 resources generate 78 mutex instances with the same catalog/scopes.

Existing structured APs retain their exact `full` and `label`. Explicit associations
ground task APs from complete task execution intervals and state APs from resource-owned
state evidence. Physical APs continue to use primitive observations. `place_approach`
never becomes an occupancy predicate. The combined clock includes physical contacts,
task start/end boundaries, and completed state updates jointly; task intervals are
start-inclusive and end-exclusive. `X` advances one observation on this declared clock.
Its separate version and fingerprint reject incompatible continuation records.
Per-AP results expose descriptors, bindings, source evidence, observation indices,
and Boolean values; physical counterexamples preserve recovery event and primitive-step
references. Formula evaluation delegates to the existing reviewed monitor. Both
earlier public checking entrypoints retain their contracts.

Trace preparation is also available internally without a verdict: composition
receives complete observations and per-`rule_id` valuations, including unsafe
branches. Its optional supplied decision times join the boundary set before
midpoint construction. The standalone checker keeps its existing `X`, continuation,
and explicit-completion meanings; the new composition path restricts formulas to
Boolean combinations of `G`, `F`, and `U` through parsed-formula inspection.

The [synthetic companion](../../test/fixtures/KMR_assembly_board-v1_recovery/storage_interruption/README.md)
retains all four original recovery events and 22 primitive parameters/references
over `[0, 22]`. Recovery restores KMR carrying `KET8_Square_8mm` at `Storage`;
`move_to_resource(M1) → place_approach → place_release` remains pending. This is
preparation for offline checking, not physical recovery or nominal-resumption evidence.
The legacy entrypoint and DFA-state meanings remain compatible. The new part-entry
predicate does not resolve or reinterpret the legacy mismatch below. Safety-page
requirements, selection, and approvals remain unchanged.

### Unresolved receiving-region counterexample

The specification's requirement is:

> An incoming part must not begin entering an applicable resource's receiving region while that resource contains another part.

Its `ap_event/physical_observation/receiving_region_entry` meaning is:

> The bound resource's incoming tool/part begins occupying the bound receiving region while carrying the bound part. Boundary contact counts as occupancy; initial occupancy does not create an entry.

These wordings are retained exactly. The current predicate detects entry from
combined robot/tool-or-carried-part occupancy. When the tool initially overlaps
the receiving region but the carried part starts outside, the part can enter
without a new occupancy edge, leaving `ap001=False` and returning `is_safe=True`.

The 2026-10-03 review reproduced this with `KMR` moving from `x=1.1` to `x=1.3`
over `[0.0, 1.0]`, resource footprint `[-0.2, 0.2]`, `attachment_offset=-0.3`,
and part `p` initially at `x=0.8` with footprint `[-0.05, 0.05]`. `M2` contains
`q`, and its receiving region is `[1.0, 1.2]`. At `time=0.75`, `p` reaches
`x=0.95` and touches the occupied region, but the checker reports no entry event.

Resolve this mismatch against the written incoming-part requirement and add
regression coverage before accepting this rule as complete. Preserve the exact
identifiers and make the requirement/AP decision explicit; do not silently
change either wording. The existing 98 passing tests do not cover this
counterexample. The separate part-specific AP in the new path does not fix it.

### Remaining deterministic construction and acceptance

Use `supplied reviewed AP meanings + LTLf + requirement scopes → deterministic DFA compilation → grounded bindings`.
The [predefined document](../../cais_spade_llm/specification/safety/safety_assembly_board-v1_predefined.txt)
contains separate assembly-board mutex and gear_small-before-KET4 entry specifications.
The Safety page displays their exact descriptors, bindings, supplied LTLf, and
compiled DFAs under **Compile DFA**. Its complete text hash covers definitions and
bindings; product geometry has a separate validated fingerprint. New review and
explicit activation remain required; narrower historical previews confer no approval.

`G !(ap001 & ap002)` expands to all 66 resource pairs in the current scene without
an allowlist. Labels remain local to each instance. Precedence uses
`(!ap001 U (ap002 & !ap001)) | G !ap001`, with
`ap_event/physical_observation/part_region_entry` bound to `KET4_Square_4mm` and
`assembly_board-v1`, and `ap_state/processCompleted/process_target_completed`
bound to `gear_small`, `assembly`, and `Gear_Plate/Gear_Shaft_1`. The latter target
is checked against configured product geometry. Completion must precede entry;
first simultaneous completion and entry fail. Initial occupancy creates no entry.
`process_result_completed` retains its existing meaning.

Keep LLM recovery-action and primitive proposals. In predefined mode CCA associates
recovery scopes with these fixed definitions, never with replacement generated rules.
Invalid source/artifacts/compiler support block admission without a generation
fallback. The [legacy generation path](../../cais_spade_llm/agents/central_controller/recovery_safety_generation.py)
still uses `ask_llm_structured` for legacy natural-language selections. Automatic
grounding covers the complete configured population for every supplied requirement;
it does not discover missing hazards. See [the predefined contract](../../docs/predefined_safety.md).

The [resource-owned preparation contract](../../docs/gazebo_safety_preparation.md) retains
UR5e's exact `x`, `y`, `z`, `speed`, and quaternion parameters separately from
the physical observation evidence. Its configured joint identities, trajectory points, derivatives,
nanosecond timing, custody, checkpoint/configuration fingerprints and original
event/step references are saved. A joint plan alone never establishes linear TCP
motion. Legacy evidence retains its separate fixed-orientation Cartesian and resource-envelope
coverage. The new optional `continuous_motion` owner effect preserves spline
positions, derivatives and nanosecond timing; rational polynomial bounds and
outward-rounded interval FK enclose configured link envelopes. Physical occupancy
is definite or possible, with unresolved alternatives retained rather than treated
as false. Resource names and AP labels do not select this model. Version 3 owner
effects add explicit grasp/release boundaries and local carried-part envelopes.
Custody, geometry, inventories and deposited stationary coverage must agree;
these effects preserve `processCompleted`. The continuous custody clock is
`continuous_physical_boundaries_v2`. Supplied joint-error envelopes enlarge the
checked motion but do not prove that a live controller stays within those bounds.
Unresolved live helpers, grasp/release preparation, named poses, changing commanded
tool orientation and unsupported interpolation remain `NEEDS_CONTEXT`.
The common [resource_safety_preparation.py](../../cais_spade_llm/resources/resource_safety_preparation.py)
holds registration, pose/timing validation, model descriptors and evidence helpers.
The old `ur5e_safety_preparation.py` module is removed; controller-dependent
operations live in the existing controller. `RobotAgent` delegates under its motion
lock; fixed equipment needs no joints or gripper fields. `ResourceAgent` defaults
remain `NEEDS_CONTEXT`.

Optional keyword-only `primitive_models` flows through observation, reviewed
checking, automatic grounding and composition. Trusted owner callbacks validate
native parameters and supply complete fixed-orientation body/part paths, timed
containment transfers and native state effects. They return no APs or verdicts.
Containment and gripper custody remain distinct; both transfer participants and
inventories must agree. New owner-modeled fixed equipment can work while its body
remains stationary, preserving legacy `stationary_only` behavior for existing callers.
Declared task effects, guards, exact supported programs and acknowledgement evidence
justify process/state changes; `dwell`, motion and task names cannot establish a
completion by themselves. Model identity/version/configuration and resulting effects
are fingerprinted with the trace; changed contracts invalidate continuation.
These effects remain detached projections and do not advance actual history.
The KMR fixtures and legacy `receiving_region_entry` meanings remain unchanged.

The full milestone's reviewed rule catalog must cover machine access/interlocks,
exclusive M1/M2 `workholding` access through front access and `side_access`, staging/buffer
capacity, Conveyor handoffs, KMR docking/arm/base conditions, assembly-region
exclusion, persistent occupancy, and recovery/resumption obligations. Bind from
resource declarations and observations, never from the expected recovery answer.

Continue preserving event/state APs, predicted successor facts, empty trace steps, and
finite-trace completion semantics when integrating recovery. Nonaccepting prefixes with a valid accepting
continuation remain pending obligations. Missing required evidence or unsupported
rules block admission with an explicit reason. Version the templates and record
AP/rule coverage with each bundle.

**Acceptance:** the receiving-region counterexample is resolved with regression
coverage; identical inputs produce identical safety artifacts; all required rules
have deterministic applicability and grounded coverage; and authoritative safety
construction makes no LLM calls. Connect the validated primitive safety evidence
to the existing local composition backend before claiming runtime integration.
Treat construction reproducibility separately from the completeness of the
reviewed hazard model. Capacity checking, arbitrary physical predicates, changing
orientations, controller feasibility, and complete live evidence remain deferred.
The opt-in 3D slice is conditional on complete, accurate configured envelopes and
trajectories; it does not establish physical feasibility or collision-free motion.

## Milestone 3: Local parallel composition

### Implemented nominal execution

Local parallel composition is implemented and connected to nominal execution
through [EnvironmentRuntime](../../cais_spade_llm/recovery_framework/environment_runtime.py).
PA uses `set_plans(..., compile_fsa=False)` and submits
`composition_backend="local_parallel"`. CCA checks candidates through
`EnvironmentAdmission.check(..., commit=False)` and performs atomic admission
with `commit=True` when RA requests permission before execution.

The [local composition backend](../../cais_spade_llm/agents/central_controller/local_composition.py),
[EnvironmentPlant](../../cais_spade_llm/recovery_framework/environment_composition.py),
and [EnvironmentAdmission](../../cais_spade_llm/recovery_framework/environment_admission.py)
implement dependency closure, running-task and joint completion analysis,
composition caching, monitor history, snapshot validation, and atomic grants.
Existing regression coverage exercises these behaviors. The 2026-10-03 review
did not complete fresh local-composition regression runs, so no new pass result
is attributed to that review. Fresh 2026-10-04 compatibility results are recorded
under Verification and handoff below.

### Remaining generated recovery integration and acceptance

The pure offline
[`analyze_grounded_recovery_composition`](../../cais_spade_llm/agents/central_controller/offline_recovery_composition.py)
now reuses `Scope`, `Analysis`, `Budget`, and `nonblocking_region`. It accepts frozen
grounding inputs, exact recovery event/primitive references, included running work,
finite fully supplied timing alternatives, completion conditions, and an optional
replayable prefix. It builds reachable shared prefixes directly, without compiling
a global FSA. All configured resources and applicable bindings remain in scope;
there is no exclusion pruning or scaling claim.

Predefined physical rules are evaluated on the grounded observation path, with
CCA-owned source bindings; the native task monitor does not interpret physical
descriptors. Interacting nominal starts also require a complete owner-supplied
joint proof. Missing nominal or recovery evidence blocks before a grant commits.
A trusted owner can register finite nominal behavior through `nominal_request`;
there is no default live trajectory provider. Plan/bundle native validation alone
cannot certify physical rules. `SystemBridge` and its public interface are unchanged.

Optional acknowledged product-effect evidence contributes timestamps to the joint
clock. Supported declared task effects may append exact process records; the
existing `place_insert` assembly record and synthetic `M1` `machine_part` trim
record use this path while preserving other process records and custody.
Owner-provided body/part trajectories, joint transfer boundaries and native state
effects join the same branch search through `primitive_models`. Registration adds
neither an AP meaning nor permission. Branches may project
these effects, but actual history requires validated acknowledgements and exact
observed effects. Task and physical monitors retain distinct histories within one
shared continuation. Unsupported effects or incompatible histories remain unavailable.

Whole recovery-event starts are controllable. Internal observations, primitive
progress, and completion are uncontrollable; unsafe forced successors remain in
the graph. Waiting is possible only where an explicit alternative supplies its
timing and complete stationary evidence. Every alternative uses fixed trajectories
and durations. Different mutex instances consume their own `rule_id` valuations,
even though their AP labels remain `ap001` and `ap002`.

The separate
[composition companion](../../test/fixtures/KMR_assembly_board-v1_recovery/storage_interruption/composition_evidence.json)
uses `[0, 26]`: immediate placement starts the four events at `0, 5, 10, 14`;
delayed placement uses `0, 5, 12, 16`. Included `ur5e-3` work withdraws by `12`.
The composition observes simultaneous effects jointly, consumes each observation
once, and adds no empty terminal observation. Replay retains pending obligations
and preceding observations within the same fingerprinted problem. `X` is unavailable
in this path until its branching clock is defined; standalone reviewed checking
continues to support it.

Completion requires the restored Storage checkpoint, completed included running
work, and accepting selected DFAs. Missing behavior needed to discharge a pending
`F` or `U` requirement is inconclusive. `KET8_Square_8mm` custody,
deposited `KET4_Square_4mm`, both process ledgers, and all three pending M1 delivery
task IDs survive. Restoring this checkpoint does not complete delivery. These
results establish bounded offline recovery composition, conditional on the supplied
evidence; they do not grant runtime admission or establish physical feasibility.

The inspected `ProductRecoveryController._dispatch_runtime_plan_validation_check`
in [product_recovery_controller.py](../../cais_spade_llm/agents/intelligent_product/product_recovery_controller.py)
still calls `compile_global_fsa()` after inserting recovery macros. The nominal
`EnvironmentPlant` resolves and enumerates declared resource-model events.
The new
[`RecoveryCompositionAdmission`](../../cais_spade_llm/agents/central_controller/recovery_composition_admission.py)
registers the complete sequence during `plan_safety_check`, binds exact dispatch
identities and primitive programs, and grants only the specific next winning start.
CCA recognizes registered generated tasks before nominal `EnvironmentRuntime`
matching. PA carries `recovery_composition_ref`; only the CCA-owned grant permits
RA's macro executor to proceed. An overall `allowed` analysis does not authorize
an unsafe immediate start. Full-resource proofs hold additional unmodeled starts.

The same branch search retains the main and active recovery-scope native monitors
alongside the grounded physical monitors, with separate states and clocks. Existing
scope routing supplies exact native event eligibility; out-of-scope completions add
no DFA tick. Pending obligations, replayed physical checkpoints, simultaneous
completion-order alternatives, and incompatible histories remain explicit. Only
validated acknowledgements and observations advance retained histories; predicted
endpoints cannot do so. Identity, state, revision, timing, and program changes are
rechecked under the admission lock. Re-registration and duplicate requests cannot
reset monitor history or execute a consumed task twice.

The new [Prepare and check adapter](../../cais_spade_llm/recovery_framework/gazebo_safety_preparation.py)
inspects a detached CCA snapshot through the same pure composition entrypoint.
It captures all configured owners, running work, revisions, ledgers, reservations
and monitor states; stale capture windows and changed programs or histories are
unavailable. It does not register an admission session, issue `allow`, synchronize
actual monitor history, or activate the selected specifications. Live preparation
acceptance remains a prerequisite for a later guarded-execution connection.
The CCA-owned `recovery_safety_preparation_provider` is now registered from trusted
`scene.safety_preparation` configuration. The short `GAZEBO_MOTION_SAFE` and
`GAZEBO_MOTION_CONFLICT` inputs retain their mock origin and native event/step
references; observations and planner outputs remain separate. Every saved report
has `dispatch_authorized=False`. The idle slice retains the initialized ledger,
rejects unprovided history and checks resource/controller activity before planning.

For `continuous_physical_boundaries_v1`, each uncertain interval represents possible
finite AP words. Numerical refinement points are not semantic clock ticks. DFA
state sets and uncontrollable unsafe alternatives remain in the same branch search;
only a continuation safe across all retained alternatives may be `allowed`.
Definite conflicts produce `held`; unresolved bounds or budget exhaustion produce
`inconclusive`. Rule-local AP labels, native monitor history and pending obligations
remain separate and require a common completion witness. `X` is still unsupported.
The existing deterministic 1D and piecewise-linear paths retain their clocks.

The resource-owned `prepare_recovery_composition_evidence` hook defaults to
`NEEDS_CONTEXT`. KMR exports retained physical pose and custody fields, but complete
live motion, timing, helper-output, and observation contracts are not supplied by
that export. Synthetic providers are confined to tests with explicitly enabled mock
executors. Registered executor overrides without the required evidence contract
remain blocked. See the [admission contract](../../docs/recovery_composition_admission.md)
for owner interfaces, supported clocks, execution feedback, and conditional claims.
CCA now resolves pure primitive models from registered owners, passes those models
to the same composition search, and fingerprints their descriptors. Executable
callbacks cannot arrive in serialized context; changed contracts invalidate a
registered proof. `RobotAgent` can delegate a registered macro to the common
grant-consuming executor only when its owner and initialized controller both
declare prepared execution support. As of 2026-10-08, the native controller and
owner adapters implement exact prepared command ownership and dispatch plumbing,
but do not provide justified tracking, stationary containment or stopping bounds;
physical execution coverage therefore still blocks live dispatch. Continuous observed-history
replay across recovery boundaries remains required; exact synthetic graph states
cannot stand in for measured Gazebo states.
Individual permission messages cannot authorize an atomic multi-resource start
edge; those choices remain inconclusive. Active-rule Gazebo execution and observed
recovery/resumption remain separate acceptance gates.

Keep CCA as the decision authority. Extend the existing direct factored-model
analysis; do not build a global FSA and then slice it. Determine scope from shared
events, resource participation, guard/update dependencies, product custody,
task prerequisites, and safety APs. Expand scope when dependencies cross its
boundary; omitted resource behavior cannot be assumed safe or fixed without
validated evidence.

Select dependencies through the primitive AP bindings as well as declared events,
and verify which generated-event boundaries are actually controllable. A controller
that cannot stop inside a primitive cannot use every observation as a dispatch
decision. Reconcile the frozen primitive observation clock with concurrent nominal
behavior explicitly, especially for `X`; do not drop intermediate observations.

Preserve shared-event synchronization, atomic custody, running tasks, persistent
occupancy, and pending obligations. Check immediate safety and joint safe
completion. Separate successful rule checks do not establish a compatible joint
continuation. Preserve the event clock and empty/end-step semantics; dropping
an event is not automatically equivalent to stuttering.

Cache by model, rule, relevant-state, and scope fingerprints and invalidate on
change. Current local `Budget` defaults are **20,000 states / 2 seconds**;
`EnvironmentAdmission` uses the run's `composition_budget` when supplied.
The earlier roadmap proposed **200,000 states / 10 seconds** for local validation;
those are not the current local defaults. This update does not change runtime
configuration. Exhaustion is inconclusive and holds affected dispatch, not proof
of safety or infeasibility. Strong coupling may still require a large product.
Completion and failure observations are consumed; only controllable starts are
gated. Recheck the snapshot and reservations before committing concurrent starts.

**Interfaces:** retain `SystemBridge` and existing PA/RA/CCA protocol identities.
Reuse the internal factored-plan backend and retain result evidence for checked
scope, dependencies, fingerprints, explored states, timing, and inconclusive reasons.
Keep global and existing active-window backends as comparisons. Do not equate a
local accepting continuation with a general global nonblocking guarantee.

**Acceptance:** extend local/global small-model agreement to generated recovery
behavior for safety and completion; detect coupled conflicts; and avoid enlarging
products unnecessarily for unrelated resource growth. Test open obligations across
scope changes, concurrent starts, shared handoffs, and stale cached results.
Retain active-rule nominal Gazebo execution and observed recovery/resumption as
separate acceptance gates, with CCA bypass disabled. Implemented nominal local
composition does not establish those execution outcomes.

For `storage_interruption`, preserve both parts' custody and `processCompleted`,
running work, all safety-derived obligations, and the original pending M1 delivery
task IDs. Verify a compatible continuation from the restored carrying checkpoint
and later observed delivery; recovery completion alone must not complete those tasks.

## Milestone 4: UI dry runs and exact prompt evidence

Use the existing **recovery → Test a failure scenario** panel. Extend its
[diagnostic service](../../cais_spade_llm/recovery_framework/diagnostics.py)
now launches the reusable
[scenario runner](../../cais_spade_llm/recovery_framework/scenario_runner.py).
The former Case 3 test file retains its regression cases and imports that runner.
The page reads the authoritative roadmap/formal sections; README compatibility
headings remain short links.

The following is the full milestone contract. Its request recording, input
freezing, cancellation, explicit replay, and validation-evidence foundations are
implemented. Currently selectable saved fixtures are the legacy `lg_slippage`
and `move_home_failure` contexts. They do not constitute reviewed fixtures for
all four larger-setup scenarios; that acceptance gate remains open.

### Available inspection now

Open **recovery-framework → recovery → Test a failure scenario**. The existing
`lg_slippage` scenario pairs with `test/fixtures/case3_recovery/runtime_context.json`;
select `outline` first. Inspect input provenance and dependencies before running.
Live calls are the default. Missing physical readiness or saved CCA history is an
unresolved validation result, not an automatic fixture approval. Saved active-rule
contexts need `recovery_safety_context.safety_dfa_states` and an explicit
`running_aps` list. Explicit replay requires a JSON array of response objects and
is labeled as mocked. The larger ur5e-3/ur5e-4 scenario still needs its own matching
snapshot and contracts.

After a run, select its record to inspect exact requests/responses and each
candidate's detailed RA/CCA evidence. Requests from cancelled or failed attempts
remain available even when there is no completed stage result.

### Workflow

1. Select one of the four failure scenarios and a variant.
2. Select a reviewed fixture or saved Gazebo failure snapshot.
3. Inspect the starting state, observations, goals, capabilities, safety
   configuration, model/settings, and provenance.
4. Select `outline`, `primitive`, `safety`, or `full`; default to `outline`.
5. Run or cancel the diagnostic.
6. Inspect every turn's request, response, validation findings, selected
   transition, and projected successor.

Use live model calls by default. Fixture-response replay requires explicit
selection and visible labeling. API errors never silently substitute scripted
answers. Deterministic safety mode shows construction inputs and generated
artifacts without presenting them as an LLM conversation.

### Shared behavior and isolation

Dry runs and runtime use the same prompt builders, schemas, selection logic, and
validators. Supply observations through snapshot-backed adapters and label
observed, synthetic, and mocked evidence. Missing physical evidence remains
`NEEDS_CONTEXT`; fixtures cannot silently make every action feasible.

Dry runs do not dispatch robot actions, change live plant state, or complete
production tasks. Snapshot tools expose only declared non-executing context.
Freeze inputs and all referenced dependencies when the run is created, and
actually read those copies throughout execution. A copied manifest that still
reads mutable live files is insufficient.

Permit accepted-outline reuse only when inputs, prompt/schema, settings, and
validator fingerprints match. Preserve one active diagnostic job, cancellation,
partial results, and progress across page navigation. Viewing or refreshing a
record must not send a new model request.

### Exact request and response records

Capture every application-level attempt immediately at the provider-call
boundary, including all message roles and exact contents, tool definitions,
response schemas, model settings, and explicit generation parameters. Capture
all tool rounds, assistant tool calls, tool results, and application retries.
Record raw returned responses before parsing, plus errors and available response
metadata. Exclude sensitive transport credentials.

Persist immutable records keyed by run, stage, turn, and attempt. Record a
prepared request separately from invocation and response outcomes. The UI reads
these artifacts directly and presents raw records separately from parsed
outputs, selected transitions, and validation summaries. Provide the complete
record for download; visual summaries must not replace it.

A prepared preview says **not sent**. Missing capture says **not captured**.
Reconstructed prompts, filename guesses, and `latest` files cannot stand in for
an exact historical request. Existing legacy records retain their provenance
and capture limitations. Capture scope and current limitations are in the
[formal reference](JOURNAL_VALIDATION_AND_SELECTOR_ACTIONS.md#prompt-evidence-and-information-boundaries).

### Answer-leakage controls

Keep expected recovery sequences, golden responses, and scoring criteria in
evaluator-only data. Prompt builders and context tools must not expose that data.
Explicit response replay is loaded separately from operational inputs and is
labeled as scripted; replay responses are never an outline-prompt ingredient.
Outline inputs contain operational facts, legitimate goals, capabilities,
safety constraints, and applicable validation feedback, not a prescribed route
or assignment from the evaluator's answer.

Primitive generation may receive the current run's validated outline and the
responsible resource's primitive catalog, not a golden primitive sequence.
Keep private RA models and selector internals outside model-facing fields unless
explicitly part of the documented method. Exact constraint-derived feedback is
legitimate guidance and must remain visible in the record.

Record operator guidance explicitly and distinguish assisted runs from
unassisted evaluation. Audit prompts and tool results together. Use the term
**controlled information exposure and answer-leakage prevention**; do not claim
universal absence of bias. Preserve formal tokens when varying input order.

**Acceptance:** all four scenarios can be tested before live recovery exists;
transport-spy tests prove displayed payloads equal actual call arguments;
evaluator-only canaries never reach messages or tool results; no diagnostic
moves equipment. Cover malformed responses, retries, cancellation, missing
capture, frozen-dependency changes, and checkpoint mismatches.

## Milestone 5: Four physical failure scenarios

Use once-per-run, state-based injections reached through acknowledged execution.
Record the actual failure state and distinguish configured drop targets from
observations. These recoveries are evaluation hypotheses, not prompt material
or scripted runtime answers. Alternative validated sequences may succeed.

| Failure and checkpoint | Recovery behavior | Negative variants |
| --- | --- | --- |
| **Conveyor breakdown:** after `ur5e-1` acquires the completed part, before Conveyor release. | `ur5e-1` stages the completed part in the M1 staging tray. KMR collects it and transports it directly to Buffer For Machined parts for `ur5e-3` to retrieve. | Unavailable KMR, blocked route, occupied staging/buffer. |
| **Machining station handling robot breakdown (`ur5e-1`):** completed part remains in M1 before acquisition. | KMR docks at M1, unloads the completed part through `side_access`, and places it on the operational Conveyor for delivery to Buffer For Machined parts. | Blocked access, missing interlock evidence, unavailable loading space. |
| **Machining breakdown during part processing (M1):** at 50% of configured simulated processing duration. | `ur5e-1` or KMR extracts recoverable WIP from M1. KMR transports it to the M2 staging tray, and `ur5e-2` loads M2 to complete the remaining operation after any required configuration change. | Unsafe extraction, incompatible configuration, unavailable M2, unrecoverable WIP. |
| **Part slippage:** `ur5e-4` drops `gear_small` into `ur5e-3`'s region. `KET4_Square_4mm` in `Buffer For Machined parts` is the desired starting condition, not an observed checkpoint. | Prepare recovery using the observed custody of `ur5e-3`, preserve its pending work, and recover `gear_small` under the separate mutex and precedence specifications. | Unreachable part, unavailable staging, unsuitable part condition. |

Preserve the failed robot's actual pose, WIP identity/progress, custody, and both
interrupted-task obligations. Implement missing extraction, staging,
configuration-change, and route support through resource-owned capabilities.
Validate new KMR recovery routes without preloading them as hidden nominal
alternatives. Simulated machining progress does not establish material-removal
physics or hardware resumability.

**Acceptance:** observe recovery and nominal resumption, or a justified
rejection/hold without false completion or custody loss. A blocked reference
route is not proof that every alternative is infeasible. Preserve unsuccessful
attempts and incomplete observations.

## Predefined task-level DES audit

For the claim that a failure needs synthesis, separately establish **no recovery
path within the predefined task-level DES** from the actual failure valuation to
all remaining goals, including both interrupted tasks. An RA's private
successor/effect model is used to derive primitive support; it is not evidence
that a complete task-level recovery route already exists.

The diagnostic runner writes `task_des_audit.json` using a bounded exhaustive
search of a supplied explicit task model. Bind the model's exact failure valuation
to `failure_snapshot_fingerprint(runtime_context)` in
`predefined_task_des_audit.snapshot_fingerprint`. Include `model` (`states`,
`transitions`, `events`), `failure_valuation`, `goal_conditions`, and an explicit
`complete` model-construction assertion. Results are:

- `path_found`: report the legitimate modeled continuation without suppressing it.
- `no_path`: complete exploration of a model explicitly declared complete.
- `inconclusive`: absent/incomplete model, unmatched/stale failure snapshot, or
  exhausted 200,000-state/10-second search budget.

Current larger-setup audit (2026-09-29):

| Scenario | Result | Missing evidence |
| --- | --- | --- |
| Conveyor breakdown | `inconclusive` | Exact post-acquisition failure snapshot and complete task graph with staging/buffer capacity. |
| ur5e-1 breakdown | `inconclusive` | Exact in-M1 failure valuation and complete task graph including access/interlock facts. |
| M1 processing breakdown | `inconclusive` | 50% WIP snapshot and complete task graph with progress, configuration and resumption obligations. |
| ur5e-4 part slippage | `inconclusive` | Observe `gear_small` in `ur5e-3`'s region and the desired `KET4_Square_4mm` buffer condition; retain both resources' pending work. Legacy LG/MCP fixtures and earlier opposite-direction experiments remain historical evidence. |

No symbolic-unsolvability claim is established for these four scenarios yet.
Keep baseline audits outside model prompts and do not remove valid nominal
alternatives to obtain a desired result.

## Experimental design

Separate four studies so their effects can be assessed independently:

| Study | Comparison and measurements |
| --- | --- |
| Recovery effectiveness | No recovery, DES-only, `pure_llm`, `neurosymbolic`; recovery success, resumed work, validation-stage rejection, revisions, latency, primitive/execution failures, custody errors. |
| Composition scalability | Global, existing active-window, local; construction time, explored states, peak memory, decision latency, cache reuse, disagreements, timeouts. |
| Safety construction | Existing LLM-selected grounding versus deterministic grounding on identical recorded inputs; required-rule coverage, AP correctness, repeatability, false acceptance/rejection against reviewed cases, construction time. |
| Prompt integrity | Dry-run/runtime payload parity, every-round capture, evaluator-data leakage, and sensitivity to input ordering with exact symbols retained. |

### Recovery trials

Use four scenarios × one feasible and one deliberately blocked condition × four
methods. Run three pilot repetitions, followed by ten confirmatory repetitions:
**96 pilot trials and 320 recorded trials**. Pilots are not confirmatory data.

Use `KET4_Square_4mm` for initial machining/transport. For the planned slippage
case, observe `ur5e-4` holding `gear_small` before a drop into `ur5e-3`'s region;
`KET4_Square_4mm` in `Buffer For Machined parts` is the desired starting condition
and has not been established by preparation tests. Retain earlier slippage
experiments as historical evidence. Initial blocked conditions are unavailable
KMR, blocked `side_access`, incompatible M2 configuration, and a slipped part
unreachable by permitted resources, respectively. Freeze exact configurations
and reference labels before collection; include other negative variants in
regression coverage first. Score validated alternative recoveries on their
observed outcome rather than requiring the reference sequence.

For controlled LLM comparisons, add exactly-three-proposals support in both
modes, keeping adaptive settings unchanged outside experiments. Use matched
models, settings, seeds where supported, and budgets. Compare selectors
separately on identical recorded candidate pools; disclose candidate-generation
and actual-count differences in end-to-end runs.

Use shared limits of 30 proposal turns and 30 minutes after injection. Budget
exhaustion is recorded separately from infeasibility. Freeze configurations
after pilot revisions and before collecting confirmatory trials. Establish a
fresh nominal Gazebo baseline with active rules and CCA bypass disabled first.

### Scaling, safety construction, and prompt integrity

Benchmark 2, 4, 8, 12, 24, and 48 resource factors using documented subsets and
synthetic extensions. Include sparse and tightly coupled dependencies, ten
repetitions per configuration, and identical 60-second/2-GiB benchmark limits.
Count global construction cost as well as checking cost. These offline limits
are separate from the configurable local runtime validation budget, whose current
default is 2 seconds. Milestone 3 retains the earlier 10-second proposal separately.

Use identical recorded contexts/outlines for safety-construction comparisons.
Include valid, invalid, and missing-context cases, unmapped APs, stale evidence,
open obligations, and boundary conflicts. Repeat identical deterministic inputs
to establish artifact reproducibility. Prompt-integrity tests cover initial
requests, tools, retries, error paths, and fixture/Gazebo provenance.

All executed methods retain deterministic safety gating. No recovery retains
normal stop behavior; DES-only uses the modeled recovery available to it. Run
removed-validation treatments only through offline replay. Keep dry-run
validation, Gazebo execution, and hardware outcomes distinct.

### Reporting and evidence

Record baseline/code versions, scenario and variant, exact inputs and snapshot
origin, seeds, model and prompt/schema settings, safety bundle, composition
backend, budgets, per-attempt requests/responses, stage findings, selected
actions, controller acknowledgements, observed outcomes, and wall/simulation
time. Preserve failed, cancelled, and incomplete runs with their true status.

Randomize method order within matched conditions. Report denominators, 95%
uncertainty intervals, variability, and all failures/timeouts. Use binary
success intervals and resampled latency/difference intervals; distinguish
successful-run latency from time-limited runs. Exclude pilots from confirmatory
results and identify every exclusion rather than silently dropping runs.
No recovery-success or throughput claim follows from a validated projection alone.

## Verification and handoff

Documentation acceptance requires one roadmap, working index/reference links,
preserved history, and the exact README headings `Working Thesis`, `Experiment
Plan`, and `Neurosymbolic Selection Method` retained as compatibility links.
The UI now reads the authoritative sections directly.

Extend the nearest existing diagnostics, UI, feasibility, and recovery suites.
Tests must cover every scenario, request/tool round, malformed response,
cancellation, snapshot change, missing capture, checkpoint mismatch, and
absence of robot dispatch. Add transport-spy and evaluator-data-canary checks.
Test strict expansion, local/global agreement, coupled obligations, concurrent
starts, stale evidence, and deterministic rule coverage.

For Python changes run `poetry check` and
`poetry run python -m compileall -q cais_spade_llm ros2`. Run
`poetry run python -m cais_spade_llm.ui_main --help` for entrypoint changes.
Run `make bootstrap-gazebo` before installed-workspace checks for ROS2
launch/script/RViz changes. Always run focused changed-feature tests and
`git diff --check`.

The historical planning baseline retained in the 2026-09-29 roadmap passed
36 feasibility/DFA and 14 diagnostics/UI checks. Verification on **2026-09-29**
recorded **244 focused tests** for the implemented validation/inspection slice,
plus the final artifact-label checks, Poetry, compilation, and CLI checks;
[verification details](JOURNAL_VALIDATION_AND_SELECTOR_ACTIONS.md#verification-and-claim-limits)
are recorded in the formal reference.
These historical results are software validation; no live Gazebo or hardware
recovery was performed in that verification.

The **2026-10-03** source review ran
`poetry run python -m pytest -q test/test_primitive_program_safety.py`:
**98 passed in 2.37 seconds**, using offline modeled evidence. A separate
reproduction exposed the receiving-region counterexample recorded in milestone 2.
The `test/test_local_composition.py` rerun and selected admission/composition cases
from `test/test_environment_capabilities.py` were interrupted; no fresh pass result
is claimed for them. No live Gazebo or hardware validation was performed in that
review. The two offline safety verification slices immediately below did not
rerun local composition.

Verification on **2026-10-04** ran
`poetry run python -m pytest -q test/test_primitive_program_safety.py`:
**186 passed in 6.95 seconds**, including the retained 98 legacy tests and 88 new
reviewed-checker cases. Sixteen formula cases include **5,440 comparisons** against
an independent finite-trace evaluator. The synthetic baseline produces **107 joint
observations** for **22 primitives** and satisfies all **10 mutex pairs plus KET4
trim precedence** at explicit trace completion. Negative cases cover concurrent
and internal conflicts, simultaneous contact, part-specific entry, missing trim,
incomplete evidence, unsupported effects, and invalid continuation. All 22 primitive
endpoints agree with the original saved physical expectations; M1 delivery remains
pending. Passing does not establish a feasible joint continuation or physical execution.

`poetry check` passed with existing metadata deprecation warnings; compilation of
`cais_spade_llm` and `ros2`, focused Ruff checks, JSON/local-reference checks, and
`git diff --check` passed. Protected-artifact hashes and the unrelated tracked diff
were preserved. The
[evidence contract](../../cais_spade_llm/specification/safety/primitive_observation_safety.md#conditional-soundness-and-evidence-limits)
states the clock-specific conditional soundness argument and model assumptions.
No LLM, runtime admission, local-composition integration, Gazebo execution, or
nominal resumption was performed in this verification.

Subsequent verification on **2026-10-04** for automatic offline grounding ran
the same safety suite: **259 passed in 26.23 seconds**, retaining the preceding
186 cases and adding 73 regressions. The unchanged 22-primitive behavior produces
107 observations for **66 mutex instances plus KET4 trim precedence**, now covering
every configured resource. Adding `ur5e-5` produces **78 mutex instances** and
detects its conflict without changing the requirement catalog or scopes. Tests
also cover fixed-resource conflicts and missing coverage, unchanged physical APs
under new recovery event identities, preserved `ap002` / `ap008` task/state
descriptors, complete ledgers, exact Boolean/string state semantics, joint
completion boundaries, `X`, pending requirements, and rejected continuation.

**53 relevant nominal declaration/projection tests passed**, including 10 new
resource/handling regressions. A broader nominal/UI run was interrupted in
unrelated UI tests and is not reported as passing. No local-composition rerun
was performed. `poetry check`, compilation, focused Ruff checks on changed source
and the safety suite, JSON/source-reference checks, and `git diff --check` passed.
Poetry retains its metadata deprecation warnings; the nominal test file's nine
pre-existing Ruff findings were verified unchanged. Original recovery JSONs,
Safety-page artifacts and selection, scene configuration, historical evidence,
and unrelated tracked changes remain preserved. All added physical/task/state
evidence is offline and synthetic; these results establish neither dispatch nor
actual recovery/resumption.

Subsequent verification on **2026-10-04** for bounded offline recovery composition
ran `poetry run python -m pytest -q test/test_offline_recovery_composition.py test/test_primitive_program_safety.py`:
**305 passed in 66.48 seconds**, including **46 new composition cases** and all
**259 existing safety cases**. Four formula cases compare every supplied timing
alternative with an independent finite-word evaluator. Tests cover the shared
initial starts, held immediate placement, explicitly permitted waiting, unavoidable
internal conflicts, per-`rule_id` AP values, pending `F`/`U`, exact observation
consumption, same-problem prefix replay, unavailable `X`, and budget exhaustion.
They also reject choice-dependent trajectories or unrelated task/state effects,
missing coverage, inconsistent custody, and completion claims based on copied
`current_state` or `resource_location` metadata. Distinct `des_event_id` and
`outline_id` symbols remain distinct.

All **13 existing nominal local-composition tests passed** in separate runs:
**12 passed in 1.87 seconds** in the restricted environment; the asynchronous
arrival/admission test hung there with an idle worker and a waiting event loop,
then **1 passed in 1.57 seconds** outside that sandbox. The restricted attempts
were interrupted and are not counted as successful runs. The historical
2026-10-03 interrupted results remain unchanged.

`poetry check` passed with its existing metadata deprecation warnings. Compilation
of `cais_spade_llm` and `ros2`, focused Ruff checks, JSON/source-hash checks,
local-link checks, and `git diff --check` passed. All **26 roadmap headings** and
unrelated tracked changes are preserved. Protected fixture JSONs, initialization,
Safety-page artifacts, and nominal runtime source retain their prior hashes;
only the authorized companion documentation changed among the protected files.

The synthetic immediate schedule's first mutex conflict is at
`time=11.407407407407407`; the supplied delayed schedule provides a completed
offline witness for the restored Storage checkpoint and included `ur5e-3`
withdrawal. All 22 KMR primitives retain their parameters and provenance. The
three original M1 delivery tasks remain pending. These results establish bounded
offline composition under supplied geometry, trajectories, start alternatives,
and evidence, with unchanged **20,000 states / 2 seconds** defaults. No LLM,
runtime admission, carrying-base extension, Gazebo execution, or M1 delivery
execution was performed.

Verification on **2026-10-05** for generated recovery admission ran the primitive
safety, offline composition, coordinator, PA → RA → CCA admission, nominal
acknowledgement, nominal local-composition, recovery feasibility, and recovery
delivery suites together: **566 passed, 8 deselected in 138.43 seconds**. The eight
deselected cases are the existing
`test_complete_run_page_uses_existing_start_stop_without_render_dispatch` UI group.
A preceding broader feasibility/delivery run recorded **178 passed and 5 failed**;
all five failures raised NiceGUI's `RuntimeError: Request is not set` at the
unchanged `ui/recovery_run.py:777`. These UI failures remain unresolved and are not
counted as passing. Restricted asynchronous runs that stalled were interrupted;
the reported combined run completed outside that sandbox with mock executors.

The combined run includes all **259 primitive safety cases**, **69 offline
composition cases**, and **13 nominal local-composition cases**. Tests establish
one shared native/physical completion witness, retained pending obligations,
exact scope clocks, replayed physical history, and rejection of incompatible
checkpoints. Admission tests cover the specific requested start, stale evidence,
program changes, unavailable live preparation, synthetic evidence on the live
route, duplicate requests and acknowledgements, and unmodeled concurrent starts.
The real nominal acknowledgement path contributes one committed main-monitor tick;
the recovery coordinator verifies and reuses its audit instead of consuming it twice.

The complete mock message-path test invokes all **22 KMR primitives** through
PA registration, RA permission requests, CCA decisions, and validated feedback.
Placement at time 10 is blocked without another primitive call; the supplied start
at time 12 receives permission after the required observed progress. The final
Storage checkpoint retains `KET8_Square_8mm` custody, deposited `KET4_Square_4mm`,
both process ledgers, and all three pending M1 delivery tasks. Fresh permission
request identities reject delayed replies, and CCA serializes complete-resource
proofs across products. Global plan FSA start conditions remain an additional gate.

After a final coordinator readability cleanup, **39 coordinator/nominal
acknowledgement tests passed in 30.07 seconds**, and **6 PA → RA → CCA integration
tests passed in 7.39 seconds**. `poetry check` passed with the existing metadata
deprecation warnings. Compilation of `cais_spade_llm` and `ros2`, focused Ruff
checks on the new composition/admission modules and affected test suites,
JSON/local-reference checks, and `git diff --check` passed. All **46 protected
files** retain their starting hashes, including original fixtures, Safety-page
artifacts/selection, `SystemBridge`, and unrelated changes; all **26 roadmap
headings** remain unchanged. Existing complexity findings in large agent/admission
handlers are outside the focused Ruff result.

These results establish the actual blocking gate with mock execution. The default
resource preparation hook returns `NEEDS_CONTEXT`; complete live motion, timing,
custody, and execution-observation support remain unavailable. No LLM call, Gazebo
trial, hardware trial, carrying-base extension, or M1 delivery execution was
performed. Active-rule Gazebo recovery and observed nominal resumption remain
separate acceptance gates. The default budget remains **20,000 states / 2 seconds**.

Nominal local parallel composition and bounded offline generated recovery composition
are implemented, with generated-event admission connected to the `validated` route
and verified using mock executors. The legacy two-specification
checker retains the recorded counterexample. The automatic grounding path establishes
reusable offline checks across the complete frozen configured population for supported
AP meanings and supplied reviewed scopes. Complete reviewed hazard coverage, trusted
live physical/task/state evidence, complete resource preparation and observation
contracts, active-rule Gazebo execution, observed recovery/resumption, and
confirmatory experiments remain separate acceptance gates.

Verification on **2026-10-05** for predefined specifications ran the compiler,
Safety-page preview, primitive grounding, offline composition, recovery coordinator,
nominal acknowledgement, PA → RA → CCA admission, predefined admission, and nominal
local-composition suites together: **520 passed in 150.38 seconds**. This includes
**290 primitive safety cases**, **98 composition cases**, **52 compiler/preview cases**,
and **13 nominal local-composition cases**. After adding the guard against restarting
physical history through another ProductAgent, **35 related admission tests passed
in 16.79 seconds**. The final startup-metadata/history checks passed **2 cases**.
Restricted asynchronous attempts stalled with an idle worker and waiting event
loop and were interrupted or timed out; the combined and related admission runs
completed outside that sandbox using mock executors.

The tests establish exact predefined formula/AP preservation with zero safety
LLM calls, deterministic compilation and fail-closed loading, all 66 configured
mutex instances plus the separate precedence instance, and a common native/physical
continuation. Prior gear_small completion permits KET4 entry; entry before or at
first completion fails. Wrong target, part, unacknowledged effects, changed source,
missing participant evidence, and incompatible history cannot authorize a start.
Nominal mock starts use the same coordinator, exact prepared task/program matching,
atomic grants, duplicate handling, and the existing acknowledgement audit. Actual
product effects must match declared effects and validated acknowledgements;
predictions never commit completion history. Native-only bundle validation explicitly
reports unavailable physical grounding instead of a safety pass or safety-free replan.

`poetry check` passed with existing metadata deprecation warnings. Compilation of
`cais_spade_llm` and `ros2`, focused Ruff, JSON/geometry-reference checks, local-link
checks, and `git diff --check` passed. All **206 protected files** retain their
starting hashes, including `SystemBridge`, selection, approvals, all saved preview
artifacts, and original recovery fixtures. All **26 roadmap headings** remain.
The new predefined document is selectable but has not been approved or activated.
No safety-generation LLM call, browser trial, Gazebo execution, new live trajectory
provider, carrying-base extension, or M1 delivery execution was performed. The
legacy `receiving_region_entry` mismatch and historical interrupted results remain.

Verification on **2026-10-05** for UR5e Prepare and check ran the new preparation
tests together with predefined compilation, previews, primitive grounding, offline
composition, recovery admission and nominal composition: **551 passed in 156.57s**.
After final trajectory-header, provenance, unresolved route-context, explicit-later-start,
and concurrent execution-feedback checks, the dedicated preparation suite passed
**33 tests in 6.13s**. These use mock ROS services
and registered test owners. They establish no-dispatch preparation, six-joint and
nanosecond preservation, all 12 resources / 66 mutex pairs, independent AP labels,
mutex conflicts, strict gear_small-before-KET4 precedence, missing/stale/changed
evidence rejection, explicit later starts, and unchanged histories/ledgers/pending
work. The NiceGUI component requires an explicit click; no browser or live execution
claim is made. The first sandbox UI run stalled in the asyncio selector; the
mock-only runs completed outside that restriction.

The broader controller suite reported **250 passed / 6 failed**. All six failures
were reproduced using the unchanged `HEAD` Cartesian execution method: five expect
collision checking to be disabled despite the current configured `true`, and one
requires the absent historical `recorded_two_part_validation/saved_waypoints_scene.json`.
The combined preparation/UI-setup run reported **91 passed / 10 failed**; the ten
failures occur in the untouched Run-page `client.ip` path with a test client lacking
an HTTP request (`RuntimeError: Request is not set`). These broader suites are not
claimed clean, and their fixtures or runtime settings were not changed to obtain
a pass.

`poetry check` and compilation of `cais_spade_llm` and `ros2` passed; Poetry retained
its existing metadata warnings. Focused Ruff, JSON/reference checks, local links,
and `git diff --check` passed. The 48-file preservation capture for this slice and
the earlier **206-file protected set** remain unchanged, including `SystemBridge`,
selection, approvals, previews and original fixtures. All **26 headings** remain.
No local `gzserver` was running, so no live capture/planning result was collected.
The current scene has no complete `safety_geometry` contract; fresh attachment
observations, full-resource Cartesian/envelope coverage and stationary/running-work
coverage also remain blocking gaps. No proof activation, grant, LLM call, failure
injection, Gazebo motion, or recovery/resumption trial was performed.

Verification on **2026-10-05** for extensible resource-owned preparation ran the
combined preparation, predefined safety, primitive grounding, offline composition,
CCA admission and nominal local-composition suites: **579 passed in 164.03s**.
Final affected product/native-history checks passed **111 tests** (53 deselected),
and registration/model checks passed **51 tests in 9.11s**. After preserving the
historical UR5e evidence contract, the complete owner/preparation suites passed
**81 tests in 11.62s**, including the explicit NiceGUI action. These counts are
overlapping runs, not additional independent trials. Existing program-contract
tests passed **21 tests in 2.43s**; focused environment capability/composition tests
passed **14 tests** (123 deselected).

The new tests establish a registered additional resource and its declared action,
78 mutex pairs, physical conflict detection without catalog changes, a seven-joint
mock through the same preparation adapter, and fixed-equipment evidence for `M1`
`machine_part` / `dwell`, `Conveyor` `advance_conveyor` / `move_relative`, and
`Buffer For Machined parts` `advance_part` / `move_relative`. They check matching
containment transfers, exact inventories and native state effects, declared trim
completion, pending `F` satisfaction in one native/physical composition witness,
parameterized `inventory.{part_name}` declarations, immutable actual histories,
and incompatible contract/history rejection. Compatibility fixes retain the
historical `assembly_board-v1` containment destination and disallow unsupported
legacy native completion fields. No command, grant, active proof, or live resource
provider was introduced.

The broader program/Resources-page run reported **111 passed / 2 failed** before
stopping. Both failures were reproduced with the pre-change environment-model
and program modules: `ReadOnlyBridge` lacks `get_conveyor_fault` in the untouched
Resources-page status reader. That suite is not claimed clean. Broader environment
inbox runs were interrupted without a completed result; no whole-suite pass is
claimed. Restricted asyncio runs also stalled; the focused mock-only acceptance
runs completed outside that restriction.

`poetry check` passed with the existing metadata warnings. Compilation of
`cais_spade_llm` and `ros2`, focused evidence/preparation Ruff, fixture JSON and
local-reference checks, changed-file whitespace checks, and `git diff --check`
passed. The **58-file protected capture** is unchanged, including `SystemBridge`,
Safety-page artifacts, initialization files and original fixtures. All **26 roadmap
headings** and earlier dated results remain. This establishes reusable preparation
and conditional offline checking. Complete live observation and motion contracts,
guarded execution, failure injection, and recovery/resumption remain later gates;
the legacy `receiving_region_entry` mismatch, composition's restriction on `X`,
and **20,000 states / 2 seconds** defaults are unchanged.


Verification on **2026-10-05** for supplied recovery motion ran the continuous-motion,
preparation/UI, resource-owner, primitive safety, predefined safety, offline recovery
composition, CCA admission and nominal local-composition suites: **620 passed in
174.27s**. These include between-point conflicts, outward-bound uncertainty, pending
`F`/`U`, rejected `X`, preserved event provenance and AP labels, unsupported evidence,
unchanged custody and histories, and no-dispatch preparation. Existing before/after/
simultaneous-completion precedence cases remain in the passing regressions. Counts
from earlier runs remain historical and overlap this run.

`make bootstrap-gazebo` built **16 packages** after adding `/GETRECOVERYSTATE` to
the attachment owner. The first build exposed a missing collision header; the first
live read exposed infinite ground-plane bounds in JSON. Both were corrected, and
installed observations subsequently reported **36 models and zero attachments**.
The dedicated headless session explicitly selected the predefined document and used
real registered owners and CCA without starting PA/RA dispatch behaviors. Both
supplied candidates saved **12-resource** checkpoints and returned `NEEDS_CONTEXT`:
UR5e link motion exceeded the declared idle limits, controller action servers had
not supplied idle-status records, and the `assembly_board-v1` carrier had no finite
collision envelope. Removing a required owner declaration also returned
`NEEDS_CONTEXT`. A later retry timed out on the read-only service. See the
[recorded evidence index](../../docs/gazebo_safety_preparation_2026-10-05.json)
for the immutable report paths and hashes.

**Live acceptance remains incomplete:** no prepared moving candidate reached a
live `allowed` result or a definite live mutex counterexample. No recovery
primitives, controller motion, attachment mutations, entity repositioning, failure
injection, grants or LLM calls were dispatched by Prepare and check. Precedence
has no new non-vacuous live result; empty-custody motion with no
`KET4_Square_4mm` entry would only check the no-entry case. Guarded execution and
recovery/resumption remain subsequent milestones. Do not substitute an offline
pass or a captured checkpoint for either missing live acceptance result.

Final affected continuous-motion, owner/preparation and UI checks passed **125 tests
in 16.53s** after adding stationary-component coverage, explicit quaternion validation,
owner hold-contract rejection and controller-configuration revalidation. This run
overlaps the 620-test run. `poetry check` passed with existing metadata warnings;
compilation of `cais_spade_llm`, `ros2` and the test runner, CLI help, focused Ruff,
JSON/report-fingerprint and local-link checks, and `git diff --check` passed.
The **57 existing protected files** checked against this turn's initial capture
are unchanged, including `SystemBridge`, Safety-page files, saved safety artifacts,
`cca.json` and original fixtures. All **26 roadmap headings** remain. The scene's
new preparation configuration changes no runtime budget or safety selection.

Verification on **2026-10-05** for the focused mock `part_slippage` experiment:
the existing primitive safety and predefined compiler suites passed **338 tests
in 39.48s**, including **17 new fixture/report regressions**. The standalone
runner checked all three complete traces against the same supplied document.
`PART_SLIPPAGE_MUTEX` violates only mutex at **349/66 s**;
`PART_SLIPPAGE_PRECEDENCE` violates only precedence at **415/66 s**;
`PART_SLIPPAGE_SAFE` satisfies both, with synthetic gear assembly completion at
**9 s** preceding actual modeled KET4 entry. The report retains the exact
counterexample events, primitive indices, AP bindings, and DFA transitions.
Each candidate has seven events and 19 primitives with complete 12-resource
evidence. Renaming every recovery event preserves physical AP values and
verdicts; missing evidence and forged completion cannot pass. Removing the
separate synthetic acknowledgement causes the previously safe trace to violate
precedence. Both parts are deposited, both robots end empty, KET4's trim ledger
is preserved, and its assembly task remains pending.

`poetry check` passed with the existing metadata warnings; `compileall` for
`cais_spade_llm`, `ros2`, and the runner, CLI help, and focused runner Ruff passed.
JSON, source/report hashes, local links, and `git diff --check` passed. Comparison
with the initial file capture confirms that existing runtime code, Safety-page
artifacts, and historical fixtures remain unchanged; only the existing safety
test file and this roadmap were extended, alongside the new demo files.
The evidence is explicitly synthetic and uses the existing offline primitive
observation contract, not native UR5e dispatch programs. This experiment calls
the pure grounded checker; it does not add an admission grant or a new
composition result. No LLM, Gazebo, or primitive execution was invoked. The
earlier live-preparation gaps, historical results, and legacy
`receiving_region_entry` counterexample remain unchanged.

Verification on **2026-10-05** for the requested three-case Gazebo recording:
the [live report](../../cais_spade_llm/monitor/recovery_gazebo_runs/part-slippage-20261006T021725Z/REPORT.md)
retains the separate mutex, precedence and safe prerequisite captures. The ROS
bootstrap completed **16 packages**; the native controllers reported ready and
the attachment owner returned **36 models with zero attachments**. Each final
capture includes all **12 configured resources**, 12 resource envelopes and
12 part envelopes. Explicit static constituent geometry resolves the empty
`assembly_board_v1` carrier envelope without guessing dimensions. Fixed equipment's
configured `static_body_and_idle_containment` contract now requires observed
static and idle evidence without robot fields.

All three attempts are **`NEEDS_CONTEXT`**: observed link motion and missing
controller goal-status records prevent a usable idle checkpoint; the first final
capture also exceeded the freshness window. CCA and the recovery resources have
no registered live execution-evidence providers. Continuous grasp/release,
carried-part effects, bound prepared execution and validated feedback are still
required. The new companions preserve seven events and 19 primitive references
per case, but contain no resolved native programs or prepared trajectories.
They never substitute the original synthetic parameters or one-second durations.
No post-slippage checkpoint was staged, no PA → RA → CCA execution trial occurred,
and no mutex/precedence counterexample or safe recovery was established. No
requested MP4 was produced. The owned Gazebo session was stopped after capture.

**611 tests passed**: 65 preparation checks, the isolated UI check, 504 safety/
composition/admission checks and 41 recording checks. The first preparation
suite was interrupted with 63 checks passed while its UI test hung; that test
also timed out alone in the restricted sandbox, then passed outside it in 1.51s.
Those partial/repeated runs are not added to the total. `poetry check` passed
with existing metadata warnings; compilation, both CLI help checks, focused
Ruff, JSON/report fingerprints, local links and `git diff --check` passed.
All 26 roadmap headings remain. Safety-page selection/approvals, `SystemBridge`,
`cca.json`, original fixtures, historical recordings and unrelated working-tree
changes are preserved. The legacy `receiving_region_entry` mismatch, `X`
restriction and **20,000 states / 2 seconds** defaults remain unchanged.

The **2026-10-05 live-connection follow-up** changes the requested output to one
silent, captioned 20× video containing the three separately staged trials, with
both fixed specifications active together. The combined exporter passed complete
decoding, timing and source-retention tests using synthetic encoding frames.
No requested live video exists yet. The [follow-up report](../../cais_spade_llm/monitor/recovery_gazebo_runs/part-slippage-live-20261006T030859Z/REPORT.md)
retains actual read-only Gazebo observations: all 11 controller rows covering
17 action endpoints, a complete physics snapshot, and empty observed attachments.
The configured population remains 12 resources. The successful read window was
0.234 seconds after cold-discovery failures; a diagnostic 15-second service timeout
permitted discovery without changing the observation freshness requirement.
The dedicated launch disabled Nav2 and enabled the opt-in controller reader;
default launch/controller behavior is preserved.

Trusted pure models now reach CCA's composition search without serialized callbacks,
and changed descriptors invalidate registration. Synthetic continuous custody
effects preserve part geometry, release/reacquisition ordering and process ledgers;
explicit joint-error envelopes widen motion bounds. Registered RobotAgent execution
requires both owner preparation/feedback and initialized controller support. That
live execution contract and compatible continuous observed-history replay are still
missing. No staged checkpoint, live safety rejection, recovery execution or
observed resumption is claimed, and no missing-evidence capture is published as a
successful trial.

Final selected suites passed **954 tests**: **212** preparation/owner/admission,
**478** safety/composition/nominal-admission, **221** Gazebo launch/resource, and
**43** recording tests. Earlier overlapping runs and corrected fixture/default
failures are excluded from that total. `make bootstrap-gazebo` completed **16
packages**. `poetry check` passed with existing metadata warnings; compilation,
UI/preflight CLI checks, focused evidence/admission Ruff, JSON/reference checks,
local links and `git diff --check` passed. The recorder module retains pre-existing
lint findings outside the new exporter. All **26 headings** and **77 protected
files** are preserved. The unchanged specifications, approvals, fixtures, runtime
authorities, legacy counterexample, `X` restriction and budget defaults remain.


**2026-10-08 consolidation and grounded CCA implementation:** the
[retained consolidation verification](../../cais_spade_llm/monitor/recovery_gazebo_runs/cca-consolidation-20261008T033602Z/verification.json)
records the shared coordinator, removal of five superseded production `live_*.py`
modules, exact nominal grant/command fixes and checks performed. Test groups in
that report overlap and must not be summed. Three pre-existing UI fixture failures
remain explicitly reported. Its background Gazebo run captured 11 controllers and
12 resources with `SAFE_shared_area_mutex` and
`SAFE_gear_small_before_KET4_Square_4mm` active and bypass disabled. Observed UR
link motion prevented a complete idle checkpoint; no command or physical-history
entry was committed. Shutdown required forced termination, recorded separately.

Implementation now follows the accepted grounding sequence above. The first
blocking gate is a justified native tracking, stationary-containment and stopping
contract under unchanged Gazebo behavior. Existing position commands, sampled
stationarity and observed error maxima do not establish it. Record precise missing
evidence and retain `NEEDS_CONTEXT` wherever a guarantee cannot be established.
Continuous history and a recoverable hold are supporting implementation work;
neither is successful recovery or permission to resume. Full acceptance requires
actual grants, unchanged prepared execution, authenticated effects and valid nominal
continuation with both selected specifications active. Recovery-event generation
for the four failure scenarios remains downstream of that gate.

**2026-10-08 scope clarification (supersedes the full physical-guarantee gate
above):** the user confirmed that the required implementation is Safety Monitor
Construction and Validation (Modified), Steps 1–4, working in Gazebo under explicit
motion and failure models. Proving universal physical tracking/contact/stopping
bounds and implementing automatic nominal resumption are not prerequisites for
this scope. Keep model-based CCA decisions and observed execution distinct from
physical guarantees; never mark `physical_execution_verified` true for modeled
evidence. Retain exact command authorization, selected specifications and
`diagnostic_cca_bypass=false`. Optional runtime hold changes from the broader plan
are being removed; continuous-history replay and evidence-identity checks remain
relevant. The next acceptance gate is an actual CCA-checked modeled motion in
Gazebo, plus a rejected modeled conflict and retained observations.

**2026-10-08 model-based integration update:** continuous checkpoint version 2
now replays possible DFA states across contiguous modeled programs while retaining
the scalar checkpoint contract. Prepared-motion admission now requires an allowed
immediate start from the existing common composition engine; missing graphs,
wrong events and unfulfilled temporal continuations cannot authorize dispatch.
The optional runtime hold changes were removed after the scope clarification.
Owner model preparation separates declared interpolation/stationary assumptions
from `physical_execution_verified`, which remains false. Native provenance checks
remain available as diagnostics without making perfect sampled stationarity a
prerequisite for the model-based method.

The [current background Gazebo evidence](../../cais_spade_llm/monitor/recovery_gazebo_runs/grounded-cca-20261008T040700Z/acceptance-2/result.json)
captures all 12 resources without an unresolved modeled checkpoint. It has both
selected specifications active and bypass disabled. Native preparation then rejected
the first trajectory timestamp, 1 ns rather than 0 ns. The raw rejected points are
retained in the owner preparation; no command was authorized or sent. The next
gate is complete native preparation and common-composition evaluation before any
execution. These results do not establish successful recovery.


**2026-10-08 native preparation and evidence update:** owner preparation now
retains the raw native trajectory and explicitly inserts its observed initial hold
at time zero for the specific 1 ns, identical-position, zero-derivative initial
waypoint. Every native waypoint and the final duration remain unchanged; CCA and
execution use the same resulting prepared trajectory. Other invalid start times
remain unsupported. The resource-preparation suite passed 104 tests. Fresh
admission validation now binds controller identity and command revisions as well
as the declared motion model; its 70 admission tests passed before the subsequent
stationary-observation refinement.

The next [Gazebo attempt](../../cais_spade_llm/monitor/recovery_gazebo_runs/grounded-cca-20261008T040700Z/acceptance-3.log)
reached report writing but could not serialize a native unsigned numeric value.
Its CCA decision is unavailable and cannot count as a completed acceptance run.
[Read-only reconciliation](../../cais_spade_llm/monitor/recovery_gazebo_runs/grounded-cca-20261008T040700Z/acceptance-3/post-failure-observations.json)
found unchanged controller instances and command revisions, with no active or
pending goals. The harness now preserves native integral values as JSON integers;
11 focused harness and Boolean-mode tests passed. Stronger physical guarantees
remain excluded. The next gate is a fully retained common-composition decision,
exact grant, dispatched trajectory and fresh observed completion under the
explicit models.


**2026-10-08 modeled admission and validation update:** fresh observed geometry
is re-grounded through every selected AP and the complete composition calculation;
changed measurements can proceed only when the AP trace, temporal boundaries and
winning immediate start agree. Controller incarnation, command revision, custody
and model configuration remain bound. Sampled `observed_stationary` is retained
as evidence but is not a physical-guarantee prerequisite in explicit model mode.
Source `resource_jid` remains optional as in the existing primitive schema; when
provided it must agree, and prepared owner bindings are checked independently.
Post-execution capture now refreshes Gazebo observations and uses the retained
per-joint model allowance when checking the final joint state.

The [fifth live attempt](../../cais_spade_llm/monitor/recovery_gazebo_runs/grounded-cca-20261008T040700Z/acceptance-5/result.json)
passed owner model preparation with `model_execution_verified=true` and
`physical_execution_verified=false`. It exposed missing configured assembly-target
metadata in measured part geometry before the common graph could run; no command
was sent. The geometry adapter now preserves the exact `assembly_target_map`
symbol alongside measured envelopes, without asserting an assembly completion.
The next gate remains a fully recorded permitted live motion and observed outcome.

Current checks include 81 admission, 60 live-safety/nominal-adapter, 45 continuous
motion, 140 KMR and 23 nominal-admission tests passing. All 90 preparation cases
passed across the core run and the separately run UI case; the sandbox UI stall
was resolved by running that case outside the restricted sandbox. The combined
resource/simulation-timing run had 367 passes (including all 115 resource tests)
and six failures in existing collision-policy expectations and a missing archived
recording fixture. The broader coordinator run exceeded its 120-second limit;
its partial progress is not a passing suite. Poetry, compilation and UI CLI help
passed. Repository-wide diff checking reports an unrelated trailing blank line in
`test/test_environment_capabilities.py`; touched-file checks are clean. Logs remain
in the [run checks directory](../../cais_spade_llm/monitor/recovery_gazebo_runs/grounded-cca-20261008T040700Z/checks).


**2026-10-08 first live-grounded common composition:** the
[sixth background Gazebo attempt](../../cais_spade_llm/monitor/recovery_gazebo_runs/grounded-cca-20261008T040700Z/acceptance-6/result.json)
returned an allowed common composition for the supplied first `move_cartesian`.
The graph includes all 12 configured resources, both selected specifications as
67 instantiated rules, an allowed start and a completion witness. This is a
prediction using live observations and native preparation, not observed execution.
Fresh admission then rejected a missing modeled-start binding in the installed
nominal adapter, which overrides the controller preparation method. No grant or
command was emitted. The next gate is preserving the same model binding through
that existing shared adapter and observing the authorized command's outcome.

All 93 current preparation tests passed after preserving assembly-target metadata.
Final retained replay tests passed 22 cases, and source-provenance tests passed two.
The implementation uses supplied recovery motion to test CCA integration; no new
LLM recovery generation has occurred. The single-motion harness assumes other
resources remain in commanded holds. Incorporation of moving nominal work and
modeled failure branches is still a remaining implementation milestone.


**2026-10-08 granted native motion and retained failure evidence:** fixed-link
validation now uses literal native containing links and exact URDF fixed ancestry.
Captured, fresh and modeled root-fixed groups must have identical definite
occupancy for every bound region under the declared stationary model. Moving-body
enclosures remain checked separately, so arm occupancy cannot hide a changed base
AP. This adds no distance tolerance or physical guarantee. Observed initial joints
may differ from the planned initial joints only within named, declared model
bounds; their values and the prepared trajectory are not rewritten. The three
owner/adapter/preparation suites passed 239 tests; continuous motion passed 54,
continuous replay passed 17, and admission passed 81 after these changes.

The [ninth live attempt](../../cais_spade_llm/monitor/recovery_gazebo_runs/grounded-cca-20261008T040700Z/acceptance-9/result.json)
received a committed CCA grant with both specifications active and bypass disabled.
Its native goal identity is retained. The observation executor then failed because
installed ROS Humble calls subscription callbacks with one argument, while the
reader required message metadata as a second argument. This caused a read-only
service timeout and prevented observed completion from being committed.
[Direct read-only reconciliation](../../cais_spade_llm/monitor/recovery_gazebo_runs/grounded-cca-20261008T040700Z/acceptance-9/post-run-direct-observations.json)
found the same controller incarnation, no active/pending goal, and final joints
within 2.93e-5 rad of the exact prepared endpoint. That observation is not a
retroactive CCA completion. The failed command and reservation remain recorded.

The callback now accepts the installed one-argument interface. Missing publisher
metadata remains unauthenticated for the legacy topic-only idle path; configured
native controller services remain the owner evidence source. Thirteen focused
callback/owner-query tests passed. A fresh identical Gazebo world will test the
complete corrected path without migrating the old grant or erasing its evidence.
Successful exactly-once completion and live occupied-region rejection remain the
next acceptance gates; moving nominal work, modeled failure branches and the four
full recovery scenarios remain unfinished.


**2026-10-08 first supplied live motion completed:**
[acceptance-11](../../cais_spade_llm/monitor/recovery_gazebo_runs/grounded-cca-20261008T040700Z/acceptance-11/result.json)
passed the first empty-custody `move_cartesian` gate in a fresh background Gazebo
world on school WSL. Both `SAFE_shared_area_mutex` and
`SAFE_gear_small_before_KET4_Square_4mm` were active, with
`diagnostic_cca_bypass=false`. The common composition allowed the supplied event;
CCA committed a grant; the prepared, authorized and executed trajectories are
identical; authenticated observed completion was committed once, advancing
physical history to revision 1. The
[verification summary](../../cais_spade_llm/monitor/recovery_gazebo_runs/grounded-cca-20261008T040700Z/acceptance-11/verification-summary.json)
retains the exact command and native goal identities. This is a manually supplied
engineering candidate, not an LLM-generated recovery or completed `part_slippage`
scenario. `physical_execution_verified=false` remains explicit.

AP occupancy comes from the resource geometry along the prepared trajectory,
not the `move_cartesian` symbol or only its TCP target. In this run the configured
`assembly_board-v1` region has world bounds x/y [-0.25, 0.25] m and z [1.05, 1.4] m.
The final observed TCP is approximately (-0.100, -0.080, 1.500) m, while an observed
`ur5e-4` gripper finger envelope has x [-0.079, -0.048], y [-0.144, -0.113], and
z [1.299, 1.357] m, inside that region. Intermediate AP possibilities were evaluated
from continuous modeled geometry before dispatch; final geometry is separately
retained as observation evidence. These envelopes do not assert universal contact
or tracking guarantees.

Final checks include 239 owner/nominal-adapter/preparation tests, the updated full
95-case preparation suite after the callback correction, 81 admission tests,
54 continuous-motion tests, 17 continuous-checkpoint tests, 22 replay tests,
60 live-safety/nominal-adapter tests, 140 KMR tests and 23 nominal-admission tests.
Groups overlap and must not be summed. Poetry, compilation, ROS-free imports,
UI/harness CLI help, `make bootstrap-gazebo` (16 packages), and scoped diff checking
passed. The six unrelated simulation-timing failures and timed-out broader
coordinator run described above remain limitations; the global diff check still
reports the unrelated trailing blank line. No new Python modules were added.
The [aggregate verification record](../../cais_spade_llm/monitor/recovery_gazebo_runs/grounded-cca-20261008T040700Z/verification.json)
indexes checks, all attempts and cleanup. Both owned Gazebo launches and their
descendants stopped; the earlier failed grant/history was not migrated or rewritten.

The next acceptance gate is a candidate rejected before dispatch because observed
geometry places another resource in the same region. Subsequent work connects
moving nominal operations, remaining recovery events, supported modeled failure
outcomes and authenticated continuous history across successive live programs.
Grasp/release and required KMR/machine/Conveyor/Buffer effects remain unfinished.
Actual recovery-event generation for the four scenarios remains downstream of
those model-based CCA gates. Automatic nominal resumption and universal physical
bounds remain outside this implementation scope.


**2026-10-08 simple Part slippage UI and operator handoff:**

The Recovery page exposes Start part_slippage, Stop, Reset example, independent
Auto/Manual outline and primitive composition modes, Input/Output inspection,
feedback/Regenerate, and explicit Execute recovery. Existing diagnostic tools
and saved evidence are under Details. Runtime ownership survives page navigation;
Auto ends at operator execution review. Exact request/tool/response captures and
original downloads remain separated by run and generation. Manual mode exposes
the captured observations and prepared context before generation.

The selected example scopes formal-safety deferral to simulation, no selected
safety requirement, diagnostic_cca_bypass=true and no_validation. Both nominal
and recovery paths record safety as skipped. Controller feasibility, program
ownership, cancellation, custody and observed completion checks remain. The
original two specifications and their later Safety integration remain separate.

Live trials observed both pickups and gear_small slippage during ur5e-4 lowering.
They exposed missing resource dispatch, pose-planning and primitive-parameter
evidence, which were corrected with resource/controller-owned evidence. The
interrupted place_approach name, reconciled part observations and saved task
contracts now enter the prompt. Retained part assembly precedes cached resource
cleanup in the unfinished-task context. No recovery sequence or default operator
guidance is supplied to the LLM. Existing prompt templates remain in use.

At the user's request, agent-driven live trials stopped so the next test can be
performed visibly through the UI. No generated recovery primitives were executed.
The last run is 74f28bc26fb74c09a28c472b8c164b8e; earlier outline captures include
0a5711b46c3b4c1e85b19a08a7f2e4c9 and 04c97010cc70411aadf5635171d87171 under
cais_spade_llm/monitor/environment_runs/recovery/. Stopped and incomplete runs
remain retained. These runs establish the failure checkpoint and parts of outline
generation, not successful full recovery or remaining assembly completion.

Checks: 238 workflow/UI/fault/setup/saved-program tests passed before the final
context/UI changes. The updated workflow/UI group passed 63 tests; the subsequent
factual-context preparation test passed separately. Seven focused feasibility
regressions passed, including inbox survival after controller errors. Poetry,
Python compilation, UI CLI help and diff checking passed. Existing unrelated
legacy Safety fixture incompatibilities remain outside this deferred-safety task.

Next acceptance gate: the operator inspects captured observations in Manual mode,
generates/reviews the outline and primitive composition, executes once, and
observes both primitive completion and the remaining assembly. This is functional
Gazebo recovery with safety checks deferred. Existing offline diagnostics can use
compatible saved runtime_context bundles; export of the new live capture format
into that bundle is still required for a direct offline rerun.

**2026-10-08 visible observations, offline generation and later Gazebo execution:**

This update supersedes the preceding entry's Details layout and offline-export
limitation. The page order is Example, Safety, Current observations, Recovery
outline, Primitive composition, Execution. Gazebo and Offline are explicit
sources. The scenario names remain Part slippage, Conveyor breakdown, ur5e-1
breakdown and Machining breakdown during part processing. The Part slippage
preset retains KET4_Square_4mm upright in Buffer For Machined parts zone 4, both
observed pickups, and gear_small slippage during ur5e-4 placement lowering. The
other scenarios use their existing M1 checkpoints.

Reconciled failure facts are written to `observations.json` before prompt
preparation. Registered `executables` replace the invalid `function_names`
access. Errors preserve the observations, display their actual cause and allow
preparation retry. A successful capture adds the saved preparation context;
offline sessions never query live agents, controllers, perception or Gazebo.
Unknown required facts remain NEEDS_CONTEXT. Existing templates and request
writers are reused without a suggested recovery sequence or default guidance.
Prepared input is marked not sent; exact requests, tool exchanges, raw responses
and resulting outputs remain inspectable and downloadable. Recorded observations
retain their provenance and timestamps through grounding.

Every offline attempt has its own directory linked to immutable source facts.
Outline regeneration invalidates primitive composition. Primitive regeneration
preserves the outline checkpoint, and Stop/Reset preserve completed saved
results. Load saved result restores them after application restart. Test in
Gazebo can reuse the owned paused failure or rerun the example; it checks the
saved proposal against current identities, configuration, task state, custody,
fresh poses, controller readiness and primitive feasibility. A mismatch blocks
execution without rewriting the proposal. Execute recovery remains an explicit
action and repeats imported-proposal checks. Safety selections are metadata
only; formal safety is skipped, and offline physical checks are deferred.

Automated results are separate from live acceptance:

- 118 tests passed in `test_offline_recovery.py`,
  `test_part_slippage_recovery.py`, `test_project_pages.py`,
  `test_recovery_request_records.py` and `test_recovery_diagnostics.py`.
- 359 tests passed in the setup, saved-program, mocked Gazebo and
  environment-admission suites. These use test fixtures, not a live trial.
- `test_recovery_feasibility.py` completed with 46 passed and 12 failed. Five
  fail because legacy slash AP descriptors are rejected during fixture
  preparation. Seven KMR tests lack the current prepared execution provider.
  Those paths and their fixed symbols were not changed by this update.
- `poetry check`, `poetry run python -m compileall -q cais_spade_llm ros2`,
  `poetry run python -m cais_spade_llm.ui_main --help`, focused Ruff checks and
  `git diff --check` passed. Poetry reports existing metadata deprecations.
  Async regression tests ran outside the filesystem sandbox because its
  asyncio executor shutdown hung; LLM and equipment calls remained mocked.

Evidence remains under `cais_spade_llm/monitor/environment_runs/recovery/`.
The existing `aa8e27d245584008b593c84111daa47e/environment_run.json` retains
failure facts but lacks the prepared recovery request; it is viewable and is
explicitly incomplete for offline generation. A fresh capture is needed.

No live LLM request, Gazebo launch, generated recovery execution or assembly
completion was performed for this update. The user-observed acceptance gate
remains: Start failure → inspect/save complete observations → stop Gazebo →
Start offline test → inspect both stages → Test in Gazebo → Execute recovery
once → observe primitive completion and remaining assembly. Record that later
result as functional recovery with safety checks deferred.

**2026-10-08 dynamic context retrieval and expandable recovery turns:**

New sessions preserve complete saved observations and prepared recovery context
while selecting prompt facts separately. Selection follows failure, goal, custody,
shared pending-work destinations and explicit predecessor/successor relationships;
it does not use scenario-specific resource lists or insert a recovery sequence.
Alternative resource identities and declared capabilities remain visible even
when their detailed state is outside the initial context. Public field declarations
are available through `/resources/<jid>/state_variables`. The same allowlist
provides resource/part discovery, snapshots, recorded poses, target geometry,
pending tasks and declared primitive catalogs/contracts. Private models and
evaluator data are excluded.

Outline `need_context` stays in outline generation, commits no event and returns
recorded facts with source, capture time and revision. Missing live part poses use
the existing grounding path; missing resource poses use the owning
`get_recovery_snapshot` interface and require a feedback capture time. Offline
retrieval performs no live calls. Missing required context remains explicit;
repeated unchanged requests stop with `NEEDS_CONTEXT`. Candidate generation still
uses the existing validation and selection path with complete context and the
current diagnostic policy. Previously saved sessions and the current live process
were not migrated, restarted or interrupted.

The page removes Current observations and replaces stage tabs/artifact dropdowns
with independently expandable Input/Output for each turn. It preserves exact
messages and original-file downloads, keeps runtime retrieval/validation separate
from LLM output, and marks prepared previews as not sent. Indexing reads bounded
headers/tails; full payloads load only on expansion. Browser expansion choices are
scoped to run/generation/stage/turn. Provider retries/tool rounds are grouped with
their call; legacy primitive streams with repeated local turn numbers remain
distinct. Capture download and Retry input preparation retain evidence access.

Automated evidence for this update:

- 143 tests passed across `test_recovery_context_retrieval.py`,
  `test_recovery_request_records.py`, `test_project_pages.py`,
  `test_part_slippage_recovery.py`, `test_offline_recovery.py` and
  `test_recovery_diagnostics.py`. Coverage includes independent Auto/Manual modes,
  regeneration, Stop/Reset, offline reload, observation ownership, stale revisions,
  missing/private references, exact captures and independent lazy expansions.
- 11 existing tests passed in `test_case3_recovery_dryrun.py` selected by
  `prompt or schema or context or retrieval` (54 deselected). Tests used mocks;
  no provider requests, Gazebo launches or controller executions were initiated.
- `poetry check` and `poetry run python -m compileall -q cais_spade_llm ros2`
  passed. Poetry reports existing metadata deprecations. Focused Ruff checks
  passed for the new context module, affected UI modules and new test module.
  Broader lint inspection found four existing import-order findings in
  `test_project_pages.py` and three existing nested-with findings in
  `test_recovery_request_records.py`; these are unrelated to this feature.
  Previously recorded feasibility failures were not rerun or claimed fixed.
- Using the unchanged `e2392568bec34aabb32fb29552ada692/generation_0001/inputs.json`
  under `monitor/environment_runs/recovery/`, toggling context selection on the
  same initial session reduced grounding text **6,358 → 3,367** and outline text
  **10,193 → 6,409** characters. The source file was not modified. The legacy
  `recovery_outline/worked_previous/multi_turn_turn03_outline_request_20260802T145141.txt`
  has **4,682** characters of prompt text (trimmed message body) and **11,210**
  characters in its complete request summary including schema and metadata.
  This legacy example has two resources; the saved current capture keeps twelve
  resource identities discoverable. These comparisons exclude response schemas
  from prompt-text counts and do not establish recovery quality.

**User-observed Gazebo results for this change: pending.** The user retains control
of live generation and execution. After loading the updated UI for a new session,
inspect the compact input, observe any `need_context` turns, expand their outputs,
review the outline and primitive composition, then use the explicit Gazebo testing
and execution buttons. Record observed primitive completion and remaining assembly
separately from these automated results; formal safety remains skipped.

**2026-10-08 Part slippage safety selection defaults:** the existing mutex and
`gear_small` before `KET4_Square_4mm` specification files are preselected for a new
Part slippage example and added when switching back to Part slippage. Explicit
manual deselection is retained across polling and source changes. The UI reflects
the coordinator's selections without starting work. `selected_safety_file` remains
`__NONE__`; the diagnostic validation policy and skipped formal safety status are
unchanged. Saved records and the current live run were not modified.

Automated verification: 95 tests passed across `test_part_slippage_recovery.py`,
`test_project_pages.py` and `test_offline_recovery.py`, including scenario-switch
selection, manual override, UI synchronization and deferred-policy checks.
`poetry check`, Python compilation, focused Ruff checks and `git diff --check`
passed; Poetry retains its existing metadata deprecation warnings. No live
Gazebo or LLM action was initiated. User-observed verification remains pending:
load the updated UI for a new example and inspect the two Safety selections.

**2026-10-08 Safety panel selection visibility:** fixed the mismatch between
relative default selection paths and absolute paths returned by SystemBridge.
The dropdown keeps the recorded selection strings as option keys for matching
files and displays complete filenames as selected chips, including after
scenario changes. Adding defaults retains an already selected absolute path
without duplicating its relative counterpart. Manual deselection remains valid.

Automated verification: 74 tests passed in `test_part_slippage_recovery.py` and
`test_project_pages.py`; the UI suite was rerun after extending synchronization
to scenario changes. Poetry, compilation, focused Ruff and diff checks passed.
These are mocked checks; live UI verification remains pending. The app uses
`reload=False`, so changes require the next UI process restart. The ongoing run
was not restarted or modified, and formal safety remains skipped.

**2026-10-08 combined recovery mode:** replaced the separate outline and primitive
composition Auto/Manual toggles with one Recovery mode control in Example. Each
explicit change uses the existing SystemBridge operation to set both stage modes
to the selected value. Auto remains the default; Execute recovery remains an
explicit action. The per-stage runtime settings and saved records stay compatible.
Opening or reconnecting the page does not change modes or start work. Older runs
with different stage modes show their recorded values without rewriting them.

Automated verification: 99 tests passed across `test_project_pages.py`,
`test_part_slippage_recovery.py` and `test_offline_recovery.py`, including the
shared control for Gazebo/Offline, both mode selections, reconnection and existing
mixed-mode records. Poetry, compilation, focused Ruff and diff checks passed.
Live UI verification is pending the next UI restart; no live runtime was restarted
and no model or equipment action was initiated.

**2026-10-08 direct generation from saved observations:** SystemBridge now exposes
`prepare_offline_recovery`, retaining the existing start and saved-result entrypoints.
Preparation and generation remain owned by the coordinator/session. Capture
readiness is derived from recorded preparation inputs, with actionable missing
context, rather than trusting the persisted `offline_ready` flag. All recorded
resource identities and capabilities remain available for discovery. A recovery
model and primitive support are required when validation uses that resource;
missing evidence remains `NEEDS_CONTEXT`. Empty prepared pose fields no longer
hide measured poses in the same capture. Recorded poses seed the existing
grounding store with saved provenance, separately from projected states. The
requested drop pose is never substituted for a measured pose.

Automated verification: **147 tests passed** across `test_offline_recovery.py`,
`test_part_slippage_recovery.py`, `test_project_pages.py`,
`test_recovery_context_retrieval.py` and `test_recovery_request_records.py`.
Coverage includes direct Generate in both shared modes, all four retained stage
mode combinations, explicit execution, twelve resources including missing recovery
models, null prepared poses, chronological ordering, missing-context explanations,
no-call preparation, duplicate clicks, cancelled/late preparation, Stop/new attempt,
Reset, reconnection, saved-result reload and exact request captures. Offline tests
reject live interface calls and verify immutable sources and separate artifacts.
Poetry and Python compilation passed; Poetry retains its metadata deprecation
warnings. Focused Ruff checks passed for the affected runtime/UI/context modules
and the offline/context tests. Existing unrelated test-file lint findings remain
separate from those checks.

A preparation-only check used the actual capture
`1c3bec8faca6495cb1b0e22046277d5c/observations.json`, captured
`2026-10-08T20:46:46.864548+00:00` (October 8, 4:46:46 PM EDT).
It reached `outline_ready` with twelve resources, retrieved the exact saved
`gear_small` measured pose and produced a **4,030-character grounding preview**
(prompt text only, excluding schema/metadata). Artifacts were isolated in a
temporary directory; the source file hash remained unchanged. The LLM entrypoint
was configured to fail if called. No Gazebo, controller or SPADE operation was
requested. This verifies preparation and retrieval, not recovery quality or motion.

**User-observed results for direct offline generation: pending.** Load the updated
UI after the current run is finished; the app uses `reload=False`. Select Offline,
select the October 8 capture, inspect the not-sent input, and click Generate.
Inspect dynamic retrieval turns and review both generated stages before using
Test in Gazebo and the explicit Execute recovery button. Observe primitive
completion and remaining assembly completion separately. Formal safety remains
skipped. The current live run was not restarted or interrupted by this change.

**2026-10-08 Offline Manual policy, capture cleanup, and outline schema fix:**
Offline mode is enforced as Manual in the coordinator, offline runtime factory,
session progression and UI. Legacy entrypoints prepare an input without starting
generation; explicit generation, regeneration and saved-result loading retain
Manual review boundaries. Attempts to enable Auto through the mode operation
are rejected. Gazebo mode preferences survive source changes; existing request
artifacts retain their original settings.

The user-observed attempt `offline_4d5bd6209de040f5a5b594901a66c3c3` completed
grounding and then failed at Turn 2 with HTTP 400. The retrieval extension had
introduced a top-level `anyOf` in `multi_turn_outline_candidates_response`,
which the provider rejected. The provider schema now retains a root object,
the existing fields and `strict=false`. Conditional retrieval/candidate requirements
are checked locally before dispatch, retaining existing candidate validation.
Mixed responses, missing required candidate fields and malformed context requests
cannot commit events or advance to primitive composition. API failures mark the
affected stage failed and retain `stage_error.json`; explicit retries create a
separate generation directory.

The authorized one-time cleanup retained only
`1c3bec8faca6495cb1b0e22046277d5c/observations.json` and removed `observations.json`
from `e2392568bec34aabb32fb29552ada692`,
`offline_5fd94b310bba42dca8518a6003570e00` and
`offline_4d5bd6209de040f5a5b594901a66c3c3`. All 31 other files in those directories
were unchanged, including the failed attempt's provider captures. The retained
capture SHA-256 is `d92fe0228978431eaf2343d26f7c7fbe47322b9a81dc4cad3f713850d04110b2`.
The dropdown lists actual `observations.json` files; legacy `environment_run.json`
reading remains available through the existing reader. No automatic retention
policy was added; future records are retained.

Automated verification: **175 focused tests passed** across the offline recovery,
Part slippage runtime, project pages, dynamic context retrieval and provider
request-record suites. Added cases cover the provider schema boundary for four
outline modes and three action horizons, local conditional requirements, forced
Manual behavior, Gazebo mode restoration, saved-result compatibility and preserved
errors across outline/primitive retries. Poetry checks and Python compilation
passed. Focused Ruff checks passed for the recovery/UI/context helpers and
offline/context tests. Separately, Poetry retains metadata deprecation warnings,
and `test_recovery_request_records.py` retains three pre-existing `SIM117` findings.
The sandbox test run stalled in thread handling; the completed suites ran outside
the sandbox with mocked provider/equipment interfaces.

A separate replay used the retained full capture and the exact recorded grounding
response, with all outputs isolated in a temporary directory. A mocked provider
accepted one grounding and two outline request schemas; outline retrieval returned
the exact measured `gear_small` pose with saved provenance. The deliberately
repeated context request terminated at `NEEDS_CONTEXT`, confirming the no-progress
boundary. This verifies schema construction, request capture and saved-fact
retrieval, not a newly generated recovery proposal. The source hash was unchanged
and no live API, Gazebo, controller or SPADE operation ran.

**User-observed verification after the fix: pending.** At the next UI restart
(`reload=False`), select Offline and the retained capture, confirm Manual, then
Generate the outline. Review it, Continue, and explicitly Generate primitive
composition. Live API acceptance, proposal quality and any subsequent Gazebo
execution remain to be observed using the user's buttons. Formal safety remains
skipped; no live run was restarted for these changes.

### 2026-10-08: Saved observations selection and repeated entries

Reproduced a delayed page poll restoring the previous Gazebo/Offline selection
after the operator selected Offline and a capture. Poll completion now reads the
current bridge state and discards files from an obsolete run or generation.
Saved-observation options restore the current runtime selection, including a
previously selected offline copy, without starting preparation during refresh.
Unchanged options are no longer rebuilt by background list reads.

Offline attempts retained copies of the same input, so one captured failure
appeared three times. The list now groups equal observation content using the
existing observation fingerprint, prefers the original run's file, and retains
links to completed offline results. Different recorded facts remain separate.
No observation or diagnostic file was deleted or rewritten for this fix.

**Automated verification:** 123 tests passed across `test_project_pages.py`,
`test_offline_recovery.py`, and `test_part_slippage_recovery.py`. Coverage includes
an in-flight stale poll, selection restoration, unchanged-option refresh,
duplicate copies, changed facts, an unavailable original, and saved-result loading
through the original capture. `poetry check`, Python compilation for
`cais_spade_llm` and `ros2`, and focused Ruff checks for changed runtime/UI modules
and offline tests passed. Existing Poetry deprecation warnings and four existing
import-order findings elsewhere in `test_project_pages.py` remain separate.
A read-only check of the actual retained files returned one dropdown entry from
three copies of the October 8 capture; all three file hashes were unchanged.

**User-observed verification: pending.** Apply at the next UI restart
(`reload=False`), then select Offline and the capture once. No live generation,
Gazebo execution, or running-UI restart was performed for this fix.

### 2026-10-08: Shorter prompts and LLM proposed events

Each turn's Output now begins with the events from the captured LLM response,
in their original order, including rejected candidates. The table shows exact
event, resource and part identifiers. The full response follows with readable
JSON formatting. Runtime selections remain in a separate, initially collapsed
Validation records section; Returned context is independently expandable.
Retries, original downloads, missing captures and legacy parsed-response labels
are retained. Expansion choices survive refresh and reconnection and are scoped
to the run, generation and turn. Runtime-selected events never fill a missing
LLM proposal table.

The saved re-pick rejection was traced to `held_part.domain=[null]`, which reflects
captured empty custody rather than all future part identities. Successor matching
now permits preserved `held_part` when its declared `equals_from_param="part_name"`
guard matches the grounded task `part_name`. Conflicting parameters, other guards,
closed location/field domains and missing primitives remain checked. Progress
rules are unchanged: a custody change alone still does not qualify.

The exact recorded Turn 3 from `offline_5a7481bb4f4c4724b9f8a4a4979f6f35` was
replayed with live calls disabled and temporary output. It now selects the
recorded `recover_pick_gear_small` proposal because the saved `place_approach`
continuation is supported. The other two proposals remain rejected. This is a
saved-candidate test, not a new LLM proposal or executed recovery. Physical checks
remain deferred, safety skipped and execution unauthorized.

Prompts retain essential facts, all resource alternatives and previously retrieved
context while removing repeated display content. They distinguish symbolic
locations, measured poses and projected states. Feedback reports recorded reasons
instead of encouraging a larger claimed `expected_end_state`. Brief `thought`
and `rationale` instructions do not truncate responses or alter response schemas.

Measured message-body lengths from the same saved run, excluding response schemas:

| Turn | Before | After |
|---|---:|---:|
| 1, grounding | 4,030 | 4,079 |
| 2, outline | 6,645 | 5,628 |
| 3, outline | 9,822 | 7,474 |
| 4, outline | 11,478 | 9,332 |

Turn 1 adds 49 characters for concise-response guidance. All before sizes match
the original captured messages. All 32 archived generation files remained
hash-identical during the comparison, and the original observations retain
SHA-256 `d92fe0228978431eaf2343d26f7c7fbe47322b9a81dc4cad3f713850d04110b2`.

**Automated verification:** the combined project-page, offline recovery,
Part slippage runtime, context retrieval, request capture and feasibility suites
finished with **253 passed and 12 existing failures**. The same 12 failures were
reproduced with the pre-change successor function: five involve legacy safety AP
test data and seven involve the unavailable KMR prepared-execution test provider.
They are not reported as passing. Additional focused legacy prompt/replay checks
passed; these counts overlap the context suite and are not added to the total.
Poetry checks and Python compilation passed. Focused Ruff checks passed for the
UI, context and feasibility helpers and offline/context tests; existing prompt
complexity/import-style findings and Poetry deprecation warnings remain separate.

**User-observed verification: pending.** At the next UI restart (`reload=False`),
start a new Offline generation. Confirm the proposed-events table, shorter inputs,
collapsed validation/context records and Manual review boundaries. No live LLM
call, Gazebo execution, capture rewrite or running-UI restart was performed.


### 2026-10-08: Offline model comparison with selected safety requirements

Offline factories accept per-attempt `model` and `reasoning_effort` overrides.
Each attempt records the effective settings and their sources without changing
the captured settings, original observations, or UI defaults. Offline remains
Manual. Selected safety files are captured once for the attempt, with exact
requirement identifiers, text, source paths and hashes; both grounding and outline
prompts include those requirements. This supplies planning constraints only:
formal safety remains skipped, physical validation deferred, and execution is
not authorized by prompt compliance.

An initial diagnostic comparison exposed a retrieval interface gap: the models
requested `place_approach` and `place_insert` through primitive contract references,
while their saved declarations were task contracts. Public task discovery and
`/task_contracts/<jid>/<event_name>` now return recorded guards, effects, parameter
bindings and primitive support. Unsupported/private portions are omitted and
incomplete results explicitly marked `partial`; the complete saved context remains
available to the validators. Missing declarations do not trigger live observations.

The first factual prompt revision states the supported references, the 1-to-8 request
limit, and the existing rule that `need_context` excludes candidate fields even
when empty. It distinguishes task contracts from primitive contracts. No event
sequence or target event count was added. Two redundant whole-request copies were
removed from validation; detached output facts and all checks remain intact.
The second factual prompt revision explains the existing successor-matching rule:
a generated event need not execute the registered task next or require its nominal
predecessor now. Guard, domain, binding and primitive-support checks are unchanged.
Repeated retrieval now compares facts and errors by reference and revision, so
reordering already-answered requests cannot create a no-information loop. New facts
and refreshes after accepted transitions remain allowed.
A replay of the saved input produced identical projections, with ten projection
calls taking 0.3352 seconds before and 0.0031 seconds afterward. This local
microbenchmark is separate from API or complete-stage timing.

**Automated verification:** 253 focused recovery, offline, UI, context-retrieval
and provider-capture tests passed. The suite includes per-attempt override
precedence, safety snapshots, exact identifiers, private-field exclusion, explicit
missing declarations, offline calls blocked, and validation output independence.
`poetry check`, Python compilation of `cais_spade_llm` and `ros2`, and
`git diff --check` passed. Existing Poetry metadata deprecation warnings remain.

**Completed API comparison (prompt revision 2):** three attempts each used
`gpt-5.4 low` and `gpt-5.6-sol low`. The configured account did not expose the
`gpt-5.6` alias, so its documented target `gpt-5.6-sol` was recorded explicitly.
Results are under `comparisons/20261009T031500Z/REPORT.md`. `gpt-5.4` produced one
runtime-accepted outline in three attempts; `gpt-5.6-sol` produced zero in three.
All six initial inputs were identical. Original observations remained hash-identical,
and guarded live-interface calls remained zero.

The accepted outline has two LLM events, `move_home` and `recover_place_insert`.
The latter claims the combined recovery and assembly effect. Its recorded support
covers the `place_insert` successor; pickup, prerequisite establishment and the
complete part effects still require primitive composition. Clearance of the shared
assembly area before the other robot's remaining work is also unestablished.
This is recorded as a runtime-accepted outline, not demonstrated recovery.

The blocked attempts exposed misunderstanding of the recorded
`held_part.domain=[null]` despite permitted grounded parameter bindings, repeated
attempts to bind the nominal pickup's null symbolic source, and malformed mixed
candidate/context responses. The prior 12 documented unrelated feasibility-suite
failures were not covered or claimed fixed by the focused test result above.

**User-requested legacy inspection:** the saved August run used `gpt-5.4 low`;
its first outline message was 4,683 characters, versus 6,702 in revision 2.
Its five-event outline contains three LLM events and two `robot_task_program`
continuations. RA/CCA interfaces are marked mocked; actual model-based workspace
rejection is recorded. `primitive_program_complete=false` prevents treating that
result as physical recovery. The recorded scenario, field declarations and safety
texts differ from the October capture. Detailed evidence is in
`comparisons/20261009T031500Z/legacy_comparison.json`.

After the user explicitly clarified that further factual prompt changes are
allowed, revision 3 clarifies recorded custody versus permitted later bindings,
provides compact affected-resource field declarations initially, and distinguishes
candidate responses with null/empty retrieval markers. No preferred event,
resource choice, recovery sequence or event-count target is supplied.
Revision 3 automated verification: **284 focused tests passed**, including
exact neutral candidate markers, continued rejection of mixed retrieval/candidate
responses, source immutability, privacy, and agreement between initial declarations
and requested declarations. Poetry checks, Python compilation and diff checks
passed. The first outline message is 8,364 characters: essential declarations and
binding semantics add 1,662 characters; this is not claimed as a prompt-size
reduction. The matched API run is recorded separately under
`comparisons/20261009T033800Z`.

**Completed API comparison (prompt revision 3):** all six attempts used identical
initial messages and response schemas, the two selected safety requirements, and
unchanged validators and generation limits. Results and exact provider artifacts
are linked from `comparisons/20261009T033800Z/REPORT.md`.

**October 9 correction:** the two runtime-accepted outlines below conflict with
recorded workspace evidence. `gear_small` is 0.799839 m from the recorded
`ur5e-4` reach origin, beyond its 0.71 m bound including tolerance. Both outlines
assigned acquisition to that resource. Retrieval omitted `gripper_reach`, and
offline validation deferred a check that could be performed from saved facts.
These historical acceptance counts do not establish supported recovery outlines.
The independent replay is saved as
`comparisons/20261009T033800Z/recorded_workspace_review.json`; original comparison
and provider records are preserved. The square-peg proposals rejected in other
attempts retained custody during `place_approach`; no complete put-down was
proposed and rejected by those records.

| Model, effort | Runtime-accepted outlines | Mean generation time | Mean total tokens |
|---|---:|---:|---:|
| `gpt-5.4 low` | 0/3 | 123.3 s | 72,203 |
| `gpt-5.6-sol low` | 2/3 | 91.3 s | 65,237 |

The two accepted outlines contain three LLM events: `move_home`, `recover_pick`,
then `recover_place_insert` or `recover_place`. The prompt supplies neither those
events nor an event-count target. Projected custody changes from empty to
`gear_small` to empty. These remain outline proposals: primitive composition,
physical feasibility, and shared-area clearance before remaining assembly are
unproved. All four blocked attempts accepted only `move_home`, then treated the
missing symbolic pickup location as requiring more context despite the measured
pose and generated-successor support. Response-format failures did not recur.
All failures remain recorded. Three attempts on one capture do not establish
general reliability or demonstrate recovery for other failure scenarios.

The user also requires applicability to resources beyond robots. Context selection
and public fact discovery depend on saved relationships/declarations, not failure
names or robot types. Thirty additional tests cover `Conveyor`, `M1`, and
`3D Printing Station`, custom public fields, parameter bindings, closed domains,
missing declarations, private fields, and offline retrieval without live calls.
They exposed two retrieval defects: the supported `not_equals_from_param` guard
was omitted, and snapshots could expose private fields from the standard field
list. Both are fixed. The exact `held_part` interpretation is included only when
its declared guards/updates apply; no general domain exception was introduced.

**Final automated verification:** 314 focused tests passed after those two fixes;
Poetry checks, Python compilation and diff checks passed. Replay verified identical
Part slippage prompts and 56 public facts before and after the final fixes, with
the original observation hash unchanged. The final fixes and replay are recorded
separately from the frozen API comparison. Its guarded live-interface call count
was zero. The existing UI/offline factory still enables generation only for
`Part slippage`; these shared prompt tests do not enable or verify non-robot
execution. Existing unrelated feasibility-suite failures remain separate.
The interrupted initial diagnostic batch is retained under
`cais_spade_llm/monitor/environment_runs/recovery/comparisons/20261009T024700Z`;
it is excluded from the complete matched comparison. Its two completed attempts
and interrupted third attempt remain available with exact requests/responses.
The second diagnostic batch at `comparisons/20261009T030800Z` retains one completed
blocked attempt and one interrupted attempt. It exposed the repeat-request loop
and the unclear successor explanation and is also excluded from model rankings.

**User-observed Gazebo verification: pending.** No Gazebo trial, primitive
composition, controller call, execution, or running-UI restart was performed for
this comparison. Saved outline results still require fresh Gazebo checks and an
explicit Execute recovery action before claiming functional recovery.

### October 9: public outline facts and recorded offline constraints

The outline prompt and its retrieval interface now exclude predefined task
contracts, primitive catalogs, private guards, implementation paths and evaluator
records. Actual interrupted/pending task names and LLM-authored event names remain
visible. Internal successor matching, primitive support, progress comparison and
runtime continuation are retained. Removing public catalogs does not authorize
arbitrary generated effects.

The compact resource index advertises available public capability fields. Requested
facts include the recorded `gripper_reach`, `staging_areas`, `reachability` and
`reachable_locations`, with their exact source fields and conflicting declarations.
Feedback preserves public rejection evidence without exposing a suggested action
or recovery sequence. Both selected safety requirements remain prompt inputs;
formal safety validation remains **skipped**.

Offline validation now checks recorded custody, destination declarations and reach
bounds before deferring fresh controller/trajectory evidence. Acquisition is derived
from declared effects and custody, including combined acquisition/release, rather
than event names. Robot checks do not run on nonrobot resources. A robot's named
`home` pose is kept separate from part-station reachability. Preserved part fields
do not falsely require already-held custody; actual changes and acquisition still
retain their custody checks.

Part and resource locations are separate, including explicit null values. Predicted
part movement retains historical measurements as `last_observed_pose`, invalidates
their use as a current source, and keeps proposed coordinates in `projected_pose`.
This also covers `resource_only` movement of a recorded carried part. Neither saved
fallbacks nor observation-store overlays may restore an obsolete measured pose.
Required missing source evidence remains `NEEDS_CONTEXT`.

**Final automated verification:** 546 tests passed in the combined offline, recovery,
context-retrieval, UI, request-record, setup and case3 suites, including all 105
offline tests. The accepted-turn regression now exercises the real candidate handler
through pickup and release, then verifies that reacquisition cannot use the old
measurement. Nine focused physical custody tests and 18 successor/custody/support
tests also passed. The broader feasibility suite has 66 passes and the same 12
previously recorded unrelated failures: five legacy AP descriptor fixtures and seven
KMR prepared-execution-provider cases. One NiceGUI `AwaitableResponse` warning remains
in the combined suite. `poetry check`, Python compilation and diff checks passed;
Poetry reports its existing metadata deprecation warnings.

**Completed API comparison:** `comparisons/20261009T123200Z/REPORT.md` records
three attempts each for `gpt-5.4 low` and `gpt-5.6-sol low`. Source hashes and the
observation hash stayed unchanged across all six trials; all initial requests and
schemas match. Guarded live-interface calls: **0**. Previous attempts and their
corrections remain preserved. No event-count target or example recovery sequence
was supplied.

| Model, effort | Reached outline review | Accepted event counts | Mean generation time | Mean total tokens |
|---|---:|---|---:|---:|
| `gpt-5.4 low` | 3/3 | 5, 3, 4 | 119.2 s | 28,769 |
| `gpt-5.6-sol low` | 3/3 | 3, 3, 3 | 55.0 s | 22,840 |

The first outline message is **7,123 characters**, down from revision 3's 8,364
(14.8%); the August legacy message was 4,683 characters and had different recorded
facts. Every accepted event matches an exact LLM proposal. All six outlines stage
`KET4_Square_4mm` at `Buffer For Machined parts` and project `gear_small` assembled.
Invalid `ur5e-4` acquisitions, including a combined acquisition/place proposal,
are rejected by the recorded reach check. These are six diagnostic planning
outcomes on one capture, not six executed recoveries or proof of general reliability.

**Remaining evidence limits:** formal safety is skipped and shared-area clearance
is unresolved. The saved `ur5e-4` symbolic location is `3D Printing Station`, while
its measured TCP and failure checkpoint record stopped placement lowering; neither
establishes a validated assembly-area occupancy or clearance result. Proposed home
events do not establish observed clearance. `outline_review` also does not prove
exact nominal resumption or completed remaining assembly. See the separate
`outcome_review.md` and `prompt_review.md` for source pointers and captured evidence.

**Corrections after the fixed API batch:** final artifact review found that the
accepted-turn path uses `_apply_task_effects_to_symbolic_state`, while the earlier
pose fix covered `_apply_outline_task_effects`. The actual commit path is now fixed
and covered by the accepted-turn regression above. The batch never reacquired a
relocated part, so its first-acquisition reach evidence remains applicable. A separate
metadata correction labels unchanged saved `pending_tasks` as `saved_observations`
after projected transitions. The original API records and frozen code hashes are
retained; final-code saved-response replay is recorded separately in
`comparisons/20261009T123200Z/post_review_replay.md`, without new model calls or
new model performance claims. All six replayed response streams reached
`outline_review` with matching accepted events and candidate decisions. All 40
recorded responses were consumed unchanged; initial messages remained identical,
and changed later messages were recorded separately. API/equipment calls: **0**.
The original API manifest and source capture hashes are unchanged. The first replay
harness attempt failed to create its output directory; its records and explanation
are retained separately from the successful replay and do not count as model trials.

**User-observed Gazebo verification: pending.** This implementation and comparison
do not generate primitives, execute recovery, restart the running UI, or establish
remaining assembly completion. The source capture remains unchanged. Offline stays
Manual; later execution requires fresh checks and an explicit operator action.

### October 9: RA evidence preparation and isolated CCA preview repair

New outline attempts retain `validation_contract_version=2` and
`primitive_support_semantics="successor_primitive_union"`. Resource-owned parameter
preparation now connects recorded facts, declared bindings, and supported local
calculations to independent primitive capability checks. Every check uses the
same starting snapshot. This does not order primitives or establish composition.
The recorded buffer proposal can validate `release_part` without an inapplicable
`assembly_slot`; product `part_state="ready"` remains a separate obligation.
Missing movement inputs remain explicit rather than being replaced with prior
motion parameters, placeholder TCP offsets, or inferred coordinates.

New captures retain owner configuration, actual attachment evidence, full poses,
declared pick geometry and policy, finite gripper joint bounds, and TCP observation
provenance. Projected facts remain separate from recorded facts. A captured TCP
height offset cannot be reused after a different projected orientation. Private
validator retrieval uses precise source references and retries the same proposal.
Saved supplements are restored only against their source capture and checkpoint;
new attempts do not silently inherit old supplements.

`Prepare Gazebo evidence` is available for unresolved outline validation through
`SystemBridge.part_slippage_action(...)`. It verifies the owned paused checkpoint,
including configuration, custody, resource/task state, revision, acknowledgements,
transitions, and monitor history. A mismatch produces a separate capture and
`Use new observations`; it never modifies the original capture. Stop drains the
owned preview before another action is allowed. Evidence retries call validation
without another LLM request and retain separate records under the owning turn.

`PreviewRecoveryMotion` plans against a complete private MoveIt scene with exact
resource, frame, link, model, scene, capture, and proposal bindings. It supports
Cartesian waypoints and declared named-pose joint targets. Its server has no
execution, gripper, attachment-mutation, live-scene-update, or command-ledger
clients. Full paths, partial paths, cancellation, and stale evidence are recorded
separately. Typed ROS field hashing avoids CDR padding differences between Python
and C++. A complete preview reports `command_sent=false`,
`execution_authorized=false`, and `continuous_collision_verified=false`; the
existing continuous geometry and CCA checks remain required.

RA, physical projection, and CCA results are separate in the expandable UI records.
Missing physical evidence does not overwrite RA `FEASIBLE`. Only supported event
projections and CCA results can advance the accepted prefix and monitor state.
Peer configurations and carried parts remain in the projected scene. Nominal
continuation requires exact unfinished-task evidence; primitive support alone
does not prove it. Both selected safety specifications retain their AP meanings.

**Remaining evidence limits:** the October 8 capture lacks some joint, frame, and
TCP/attachment information. New captures do not automatically establish every
future motion. A loose dynamic part appearing at rest is not stationary interval
evidence. Capture records static geometry or the existing owner's supported idle
containment contract where available; unresolved parts remain explicit. Stable
release placement remains unresolved when `release_settling_modeled=false`.
Product `ready` and completion effects cannot be fabricated from a release.

**Automated verification:** the final combined recovery/offline/UI/request/evidence
suites passed **935 tests**, with the same **12 existing failures** recorded
separately: five retired AP descriptor fixtures and seven KMR prepared-execution-
provider fixtures. One existing NiceGUI `AwaitableResponse` warning remains.
`poetry check`, Python compilation, and diff checks passed; Poetry retains its
metadata deprecation warnings. `make bootstrap-gazebo` built 16 packages, and all
**8 installed ROS preview integration tests** passed in an isolated localhost
domain with synthetic scenes and no Gazebo or execution interfaces.

**Model comparison:** the fresh runner and verification records are in
`cais_spade_llm/monitor/environment_runs/recovery/comparisons/20261010T013058Z/`.
After explicit disclosure approval, all **six new API attempts completed** with
identical initial prompts/schemas, safety specifications, and the unchanged source
capture. Automatic approval review had rejected the earlier command before process
creation; that rejection and the subsequent approval are retained separately.
Implementation and safety fingerprints stayed unchanged throughout all six attempts;
guarded live-interface calls were **0**. See `REPORT.md`, `report.json`, and
`outcome_explanation.md` in that comparison directory.

| Model | Attempts | Complete outlines | Accepted events | Mean seconds | Mean total tokens |
|---|---:|---:|---|---:|---:|
| `gpt-4o` (reasoning parameter omitted) | 3 | 0 | 0, 0, 0 | 5.688 | 6,723.667 |
| `gpt-5.6-sol low` | 3 | 0 | 0, 0, 0 | 21.124 | 11,575.667 |

`gpt-4o` attempt 1 stopped during grounding because the saved observed pose of
`KET4_Square_4mm` is unavailable. Attempt 2 passed RA for `move_to_named_pose`,
then reached unresolved physical projection and CCA because the saved physical
scene is missing. Attempt 3 proposed unsupported successor conditions and then
lacked required capability evidence. `gpt-5.6-sol` attempt 1 proposed returning
`KET4_Square_4mm` to `Buffer For Machined parts`: `release_part` passed, while
movement inputs remained missing. Its attempts 2 and 3 proposed `gear_small`
pickup and lacked grasp/movement evidence. PA public context retrieval was used
in its attempts 1 and 3; private validator retrieval remained separate.

Five attempts ended `NEEDS_CONTEXT`; the grounding failure retains its original
runner classification `incomplete` with the explicit missing-observation error.
None established nominal continuation. Outline safety remained unresolved and
execution safety skipped. A separate exact saved-proposal replay confirms the
corrected release capability and remaining movement inputs; it is not an extra
model attempt. No missing evidence was bypassed and no prompt solution hints,
preferred event sequence, or event-count target were supplied.

**User-observed Gazebo verification: pending.** No live capture, Gazebo preview,
recovery execution, current-run interruption, or UI restart was performed. Fresh
capture and preview acceptance remain available through the UI controls. Complete
primitive composition validation, recovery × recovery and recovery × nominal
monitoring, shared CCA history/ownership, dispatch authorization, deviation and
Stop handling, and concurrent execution remain the subsequent runtime phase.

### Authorized Gazebo capture and `gpt-5.6-sol low` repair trial — October 9–10, 2026

The user subsequently authorized agent-operated Gazebo capture and model trials.
These results are separate from user-observed UI acceptance and from the six
fixed-capture comparison attempts above. Recovery execution remains disabled.

The trials identified and repaired omitted native mesh scale, attachment and
controller evidence, ROS octet decoding, stale MoveIt part geometry, and native
preview handling of captured passive mimic joints. Private previews now retain
those measured joints instead of replacing them with ideal URDF mimic values.
Eight native preview tests passed twice after that change. New attempts replay
recorded unfinished nominal tasks; historical attempts retain their prior
semantics. Nominal completion requires supported effects and CCA evidence.

The fresh `gpt-5.6-sol low` attempts under `experiments/20261010T0234Z` and
`experiments/20261010T0250Z` made real API requests and reached RA `FEASIBLE` for
their proposed events. Both remained incomplete at physical projection; neither
is counted as a passed outline. Exact requests, responses, and preview failures
are retained in their linked generation directories.

Isolated Gazebo contact tests then reproduced persistent gear motion from dense
mesh contact sampling. The selected correction preserves every original binary
triangle record, normal, and attribute in deterministic order. Separate collision
assets leave the visual originals unchanged. Global, gear, and table contact
limits are 64; the table is resolved explicitly to avoid Gazebo's cached model.
The isolated test observed all three gears naturally disabled with zero velocity
through its remaining samples; no body was forcibly disabled and no validation
threshold was relaxed. Proofs and raw samples are retained under
`experiments/20261010T0250Z/physics_diagnostics`. Eight focused asset checks passed.
The corresponding ROS builds each passed all 16 packages.

`experiments/20261010T0343Z` captured the intended drop but exposed the cached
table override. Its later capture had complete resource observations and all
unheld parts supported as stationary except `gear_small`. No LLM request was
made from that capture. The corrected full run is recorded separately under
`experiments/20261010T0348Z`. It retained three fresh captures, including
`capture_047a4eff436844198dfc21f05629b14f`, and stopped Gazebo before the
subsequent offline generation. All unheld parts except `gear_small` had
supported stationary evidence; the remaining motion is recorded, not hidden.

Two real `gpt-5.6-sol low` offline attempts then used that unchanged capture.
`experiments/20261010T040213Z` made four API calls and proposed returning
`KET4_Square_4mm` to `Buffer For Machined parts`, after two dynamic retrieval
turns. RA exposed a loader bug: legacy quaternion restoration compared a
complete owner snapshot with a separately sampled `robot_poses` entry. The
repair restores missing quaternion fields only, retaining the explicitly
recorded owner snapshot and all supplied-evidence integrity checks. The exact
proposal replay reached RA `FEASIBLE`; all 156 offline tests passed.

`experiments/20261010T040634Z` independently made four new API calls, again
retrieved context and proposed the buffer event. RA passed. Physical projection
and CCA remain unresolved because no matching owner event preview is recorded.
The attempt took 45.620 seconds and 16,090 tokens, with zero live-interface calls,
unchanged source and implementation hashes, and zero accepted events. Neither
attempt is counted as a complete outline. An isolated native preview of the
saved planning scene is being recorded separately, without Gazebo, controller
commands, additional LLM calls, or recovery execution.

**User-observed UI acceptance remains pending.** Agent-operated simulation,
model-based outline validation, and recovery execution are distinct results.

The subsequent live run `experiments/20261010T041055Z` captured the intended
drop and all required resource observations. Its separate capture
`capture_4862656369764af1b78a86063afe6602` records supported stationary evidence
for every unheld part, including `gear_small`, under the unchanged velocity
limits. Gazebo was stopped before generation. The observations hash remains
`a6c4c34f4f3953b11067684c5a1b4c903277f665204da2f3949083ee327ce787`.

`experiments/20261010T041521Z` then made three real `gpt-5.6-sol low` API calls:
grounding, dynamic retrieval, and a proposal to place `KET4_Square_4mm` in
`Buffer For Machined parts`. RA returned `FEASIBLE`. Generation took 32.885
seconds and 10,756 tokens. The original capture and implementation hashes were
unchanged, no live interfaces were called, and no event was accepted without
the required physical and CCA evidence.

The native preview now also accepts archived Goal/Result files. This uses the
same scene, model, joint, collision, path, timing, and provenance checks as the
ROS action, without an action server or controller connection. Eight native
file-mode regression tests passed, and the corresponding ROS build passed all
16 packages. The actual saved proposal reached the native calculation and
reported three starting contacts: `KMR_base_link` with `iiwa_link_0`, and each
of `ur5e_1_base_link_inertia` and `ur5e_2_base_link_inertia` with its recorded
pedestal `mounting_plate`. Those mounting declarations are being verified
separately; the original proposal, capture, and rejected preview remain intact.
This is still an unresolved outline, not a passed CCA or execution result.

Subsequent detached replays retained that same LLM proposal. The three mounting
contacts now have exact fixed-joint or recorded pedestal evidence; only those
verified contact pairs are permitted in the private scene. Projected support
records also retain JSON-compatible captured numeric poses instead of leaking
internal rational numbers into artifacts. The resource-owned placement
calculation derives a horizontal supporting face and stable contact from the
recorded rigid box, destination surface, gravity, and attachment transform. It
does not alter observations or claim a settling outcome. RA capability checks
and the independent preview now receive the same recorded physical inputs.

`experiments/20261010T045750Z` passed native Cartesian endpoint refinement but
remained unresolved in physical projection. Investigation found that the native
MoveIt model used the configured fixed robot base transform while Python used
the measured Gazebo base transform. The resulting micrometre-scale endpoint
difference prevented supported contact. This is a model-input inconsistency,
not a new LLM failure; that replay made no provider call and accepted no event.
The repair must preserve measured roots in the private model and retain both
original and calculated model fingerprints.

Automated results for this repair: 467 focused outline/evidence/offline/context
tests passed; Poetry, compilation, and whitespace checks passed. The native
file-mode suite passed 11 tests and ROS build 15 completed all 16 packages.
The broader feasibility suite recorded 426 passes and the same 12 previously
recorded failures in retired AP-descriptor and KMR prepared-provider paths;
those paths remain separate from the offline outline repair. Source observation
hashes for both the original October 8 capture and the fresh capture are
unchanged. User-observed UI acceptance and recovery execution remain pending.

The subsequent private-model repair preserves the measured Gazebo root
transforms while retaining the original URDF and source/model fingerprints.
It also retains the source joint limits and solver configuration. Native
Cartesian calculation precision is recorded separately from physical tracking
tolerance; Python independently checks the recorded endpoint against that
numerical contract. No measured pose, contact threshold, safety region, or
safety specification was changed to obtain acceptance.

The pending-proposal handoff now uses the retained task with its observed
starting conditions and checks that every LLM-authored field is unchanged.
CCA reuses certified interval refinement when a bound for the entire movement
is inconclusive. For this proposal, two bounded intervals establish clearance
while retaining the original seven semantic observations and all peer geometry.
Unknown occupancy remains unresolved if refinement cannot establish it.

**First accepted LLM event, full outline still incomplete:**
`experiments/20261010T052904Z/report.json` records a detached validation retry
of the exact `gpt-5.6-sol low` proposal from `20261010T041521Z`. The event is
`place`, owned by `recovery-resource-3@localhost`, placing `KET4_Square_4mm`
in `Buffer For Machined parts`. RA, independent physical projection, and both
`SAFE_shared_area_mutex` and `SAFE_gear_small_before_KET4_Square_4mm` passed;
the event was appended. The lineage audit links the accepted event to the
original provider response and the successful validation artifact. The retry
took 44.295 seconds, made zero API or forbidden live-interface calls, and
preserved the source, Python, and native implementation hashes. It did not
execute recovery or grant execution permission.

Nominal continuation is still `not_ready`. `environment_8` requires
`resource_state="picked"` for `recovery-resource-4@localhost`, whose supported
state remains `failed`. `ed9a5f9c8c4b463496402821945cfe72_pending_3` requires
`resource_state="picked"` for `recovery-resource-3@localhost`, whose supported
state is now `placed`. Further LLM-proposed events are required. This result
is one accepted event, not a complete successful outline or completed assembly.

Final automated checks for the current repair: 640 focused recovery evidence,
preview, validated-outline, continuous-motion, offline, context-retrieval, and
request-record tests passed in 45.93 seconds. Poetry, Python compilation, and
whitespace checks passed. The native file-mode suite passed 14 tests, and the
latest `make bootstrap-gazebo` build completed all 16 packages. The previously
recorded 12 broader-suite failures remain recorded separately above.

Further real generation is waiting for explicit disclosure approval. Automatic
approval review rejected a new API request because it would send saved internal
observations and recovery context to the configured external OpenAI API. That
request was not run or bypassed; subsequent retries above used the saved
provider response locally. User-observed UI acceptance, a complete accepted
outline, primitive composition, and recovery execution remain pending.

### October 10, 2026: six fresh offline comparison attempts

The user explicitly approved disclosure of the saved observations, internal
resource/task context, both specifications, and dynamically retrieved facts to
the configured external OpenAI API. All six requested attempts then ran under
`comparisons/20261010T102919Z`; `RESULTS.txt` and `results_summary.json` retain
their proposals, acceptance evidence, retrieval requests, timing, token usage,
and links to the complete original provider and validation records.

Each attempt started from scratch using `capture_4862656369764af1b78a86063afe6602`.
No earlier proposal or accepted prefix was loaded. The prepared grounding and
outline prompts were identical across all attempts. `gpt-4o` omitted
`reasoning_effort`; `gpt-5.6-sol` used `low`. Neither a preferred recovery sequence
nor a 3–5 event target was added to the prompts. Runtime generation limits and
the 600-second attempt deadline were retained; all three attempts per model ran
despite repeated blockers. Native physical previews used recorded models and
private scenes with live interfaces and execution blocked.

| Model | Attempt | LLM-proposed events | Accepted events | Elapsed seconds | Total tokens | Complete outline |
|---|---:|---:|---:|---:|---:|---|
| `gpt-4o` | 1 | 1 | 0 | 24.9 | 5,549 | No |
| `gpt-4o` | 2 | 1 | 0 | 22.2 | 5,562 | No |
| `gpt-4o` | 3 | 1 | 0 | 23.2 | 5,518 | No |
| `gpt-5.6-sol low` | 1 | 2 | 2 | 192.8 | 15,723 | No |
| `gpt-5.6-sol low` | 2 | 1 | 0 | 48.2 | 10,926 | No |
| `gpt-5.6-sol low` | 3 | 1 | 1 | 136.9 | 16,066 | No |

`gpt-4o` proposed `recover_gear_small`, `Fix_and_Reposition_gear_small`, and
`reallocate_failure_task`, respectively. Each assigned `gear_small` to
`recovery-resource-3@localhost`, whose recorded `held_part` and
`_task_ctx.part_name` are `KET4_Square_4mm`. RA returned `NEEDS_CONTEXT` with
`Recorded task context belongs to another part`; no CCA acceptance occurred.
The independent owner event preview also lacked a supporting declaration for
the requested combined end conditions. This classification stops for missing
evidence rather than allowing an LLM revision from infeasibility feedback, so
these attempts do not test the model's ability to revise that proposal.

`gpt-5.6-sol low` attempt 1 proposed `place` for `KET4_Square_4mm` into
`Buffer For Machined parts`, followed by `pick` for `gear_small` using
`recovery-resource-4@localhost`. Both events passed RA, physical projection,
and both CCA checks. Attempt 3 proposed and passed the same `gear_small` pickup
as its first event. Both attempts then stopped while validating the recorded
unfinished nominal task `environment_8` / `place_approach`: the private preview
reported `gear_small/link/collision <> table_xarm6/link/surface` in the starting
configuration. The supported pickup preview ends at grasp without a post-grasp
lift; the next preview treats that same part as attached and rejects its
remaining support contact. The runtime pauses at this nominal validation and
does not ask the LLM for another proposal. This is an integration gap between
pickup and loaded motion; it does not establish that the LLM failed to propose
a needed subsequent event. A repair needs supported contact separation or
loaded retreat evidence, with collision checking, rather than an assumed lift
or a general collision exemption.

`gpt-5.6-sol low` attempt 2 proposed `move_home` for
`recovery-resource-4@localhost` with explicit `part_name: null`. RA passed.
The runtime task constructor omitted that field, and `pending_outline_event`
rejected the changed record before physical preview. This is an implementation
error: preserve the authored explicit null through task preparation and retain
the exact-proposal check. No physical infeasibility or CCA rejection was
established for that event.

None of the six attempts reached 3–5 proposed events or a complete accepted
outline. The blocking implementation remained frozen throughout the batch;
these results must not be presented as a conclusive model-quality comparison.
There were 17 provider requests and responses, 59,344 tokens, zero provider
errors, and zero timeouts. Source, implementation, native preview, and selected
safety fingerprints remained unchanged; no forbidden live-interface calls or
recovery execution occurred. The focused request-record suite passed all 52
tests in the host environment after sandboxed collection stalled and was
cancelled. Poetry and compilation checks passed. The earlier 640-test repair
result remains separately recorded above. User-observed UI acceptance and
complete outline/assembly acceptance remain pending.

### October 10, 2026: declared departure and validation feedback repair

New attempts retain `validation_contract_version=2` and
`primitive_support_semantics="successor_primitive_union"`, and pin
`physical_preview_contract_version=3`. Explicit `part_name: null` survives task
construction and exact-proposal comparison. Recorded parameters remain tied to
their originating part and task; another part's `_task_ctx` cannot supply them.
Established contradictions return factual rejection feedback, and implementation
exceptions retain their original error details.

The independent owner preview now includes the departure declared after grasp
and the movement declared after release. These are resolved from owner
declarations and supported parameters, including `travel_z`; display examples
are not motion evidence. The private scene advances through each verified
interval and custody change, retaining peer configurations and carried-part
geometry. A validated pickup calculation remains tied to its owner, part,
contract, and binding for a subsequent placement. New attempts cannot reuse a
historical preview that omitted these movements. Historical artifacts are
preserved with their original contract versions.

Supporting-contact departure retains the exact recorded collision pair and
penetration bound. Native contact checks and independent continuous bounds
reject deeper penetration, unrelated contacts, incomplete contact inventory,
missing separation, or re-entry. `contact_surface_layer=0.001` is not used as
an additional penetration allowance. Neither preview nor outline validation
grants execution permission.

The sequential square-placement/gear-pickup replay exposed a separate incomplete
Cartesian search for `ur5e-3`. Its native IK query timed out under the recorded
0.05-second solver budget. No callback collision or joint-bound rejection was
reported; the last accepted wrist joint value was near its upper bound, which
does not establish the cause or exhaustive infeasibility. Finer sampling also
failed. The exact native goal, solver diagnostics, and refinement results are
retained in
`experiments/20261010T113634Z/diagnostics/cartesian_pickup/report.json`.
The declared motion, coordinates, and limits were not changed to force success.

For a complete-input bounded IK or joint-jump failure, physical validation stays
`NEEDS_CONTEXT`, CCA remains `not_checked`, and the accepted state and monitor
history remain unchanged. The PA can receive concise solver facts and propose
a revision. Repeating the same public conditions with unchanged scene, models,
owner evidence, and motion policy stops; renaming an event does not reset this
guard. Missing input facts, unavailable solvers, incomplete contact evidence,
global deadlines, and cancellation remain blocked. The same distinction applies
to pending nominal validation. No preferred event, sequence, or event-count
target was added to generation prompts.

Automated verification before the next fresh API attempt is recorded in
`experiments/20261010T113634Z/automated_verification.json`. The strict evidence
suite passed 190 tests; offline/workflow/context/outline checks passed 484 tests,
with a final nearest-suite run of 78 tests after the eager-copy regression.
Native-client checks passed 84 tests, native file-mode checks passed 21, and
native action checks passed 11. These are separate runs, not additive totals.
`make bootstrap-gazebo` completed all 16 packages; Poetry, compilation, and
whitespace checks passed. The broader earlier repair run passed 1,145 tests and
reported the same 12 previously recorded older-path failures. The original
capture hash remains
`a6c4c34f4f3953b11067684c5a1b4c903277f665204da2f3949083ee327ce787`.

The saved `move_home`, square-placement, and gear-pickup proposals each completed
their version-3 physical preview in `experiments/20261010T113634Z/replay_v3`.
These independent local replays are not a generated complete outline. They made
no API or live-equipment calls and executed no recovery. Fresh generation
results and user-observed Gazebo acceptance are recorded separately below.

### October 10, 2026: owner reach declarations and artifact memory use

New attempts pin `physical_preview_contract_version=4`. The independent physical
preview now checks its calculated Cartesian targets against the captured owner's
`static_capabilities`, including `gripper_reach`. This closes a gap in which a
native kinematic preview could pass while violating the owner's public declared
reach. The unordered RA union checks each supporting primitive against the same
starting snapshot; it does not prove the approach to a new pickup target. For
the saved `gear_small` pose, the calculated radius for `ur5e-4` is approximately
0.8000 m, exceeding its declared 0.7000 m radius and 0.0100 m tolerance. The
version-3 pickup replay above remains a historical native geometry result and
cannot satisfy the new physical contract. The original capture, motion limits,
selected safety specifications, and generation prompts were not changed to
force success. Exact owner declarations are bound before preview calculation;
incompatible historical cached previews cannot be reused.

The version-4 local preparation replay in
`experiments/20261010T120821Z/replay_v4_prepare/summary.json` prepares the recorded
`move_home` and square-placement proposals and rejects the recorded `ur5e-4`
gear pickup using that owner declaration. It preserves explicit `part_name:
null`, makes no API or live-interface calls, and leaves the source hash unchanged.

Artifact preparation also copied the same large physical evidence graph several
times before writing each turn, and built an unused transcript when transcript
writing was disabled. These copies were removed while preserving exact provider
captures and independent mutable audit records. A recorded turn benchmark
changed from 10.087 seconds / 1,230,432 KiB peak memory to 0.00198 seconds /
476,032 KiB, with equal request, context, and retrieval hashes. Loading that
fixture alone used 476,032 KiB. A complete local workflow replay with one saved
response used 1,405,036 KiB peak memory and preserved the accepted prefix,
projected resources, monitor history, and context. Evidence is in
`experiments/20261010T1215Z-memory-repair/report.json`. This is an implementation
repair, not an alteration of the LLM's proposal or validation requirements.

The immediately preceding third API attempt ended with `SIGKILL`; its parent
record confirms cancellation and no hard timeout. The saved process PID is not
the PID in the available kernel memory-kill entry, so the termination cause is
not asserted as proven. Its preserved provider records contain five responses,
22,546 tokens, two proposed events, and one accepted square-placement event.
Its gear pickup has no completed retained validation and nominal continuation
was not established. Original partial records were kept; reconstruction is in
`experiments/20261010T113634Z/attempt_3_gpt-5.6-sol/recovered_report.json`.

Pre-attempt verification is recorded in
`experiments/20261010T120821Z/automated_verification.json`: 223 owner/evidence
tests, 84 preview tests, 252 validated-outline/offline tests, 496
workflow/validated/offline/context tests, and 20 focused artifact/version tests
passed in their respective runs. These overlapping suites are not additive.
Two experiment-report tests also passed. Poetry, compilation, and whitespace
checks passed. Native code is unchanged since the successful 16-package
bootstrap, 21 native file-mode tests, and 11 native action tests. The previously
recorded 12 older-path failures remain separate from these results.

The fourth fresh `gpt-5.6-sol low` attempt is retained in
`experiments/20261010T120821Z/attempt_4_gpt-5.6-sol/report.json`. It made eight
provider calls (39,890 tokens), independently proposed five events, and accepted
two: square placement in `Buffer For Machined parts`, then `ur5e-3` movement to
`home` with explicit `part_name: null`. In between, the `ur5e-4` pickup was
rejected by its declared reach and the `ur5e-3` pickup remained unresolved after
an incomplete Cartesian search. The LLM requested additional resource
capabilities and proposed the home movement itself. Its final pickup proposal
was not fully validated before the 600-second generation deadline. The parent
confirmed process exit and complete cancellation; report finalization finished
after the generation deadline. This is a timed-out partial outline, not complete
success. Source, implementation, native, and safety fingerprints remained
unchanged, with zero forbidden-interface calls. Peak child memory was
5,352,720 KiB. Further local replay and memory profiling precede another API
attempt; no event count or sequence was supplied to the model.

The remaining turn-memory defect was traced to `prompt_input.session_state`:
the saved Turn 8 embedded 541,018,203 formatted bytes of internal preparation
state, including earlier turn inputs. Those bytes were not the provider prompt.
The version-2 builder now excludes unused internal history and validation
records from its detached prompt presentation; full validation state, captures,
and prepared inputs remain available separately. Rebuilding the actual Turn 8
against its complete public resource/part view yields an identical 17,357-character
provider prompt (SHA256
`1fcf38db55811d67ac9c6b8ca5293c60fbc8c31deef38178aaccd5b11ec6c182`),
while the preparation artifact becomes 2,019,487 bytes. The atomic diagnostic
writer also streams JSON with the same formatting instead of allocating another
complete encoded string. Evidence is in
`experiments/20261010T120821Z/memory_diagnostics/turn8_prompt_comparison.json`.

The rejection-feedback path now retains the owner's supported calculated target,
recorded `gripper_reach`, source/revision, and exact numeric rejection instead of
replacing it with generic wording. Private validation models remain excluded.
Recovery and nominal obstruction paths preserve the same verdicts and solver
evidence. The validated-outline/context suite passed 249 tests after this repair;
Poetry and compilation passed.

Local replay of the last proposal after the accepted `home` event still returned
`NEEDS_CONTEXT`, with native Cartesian fraction 0.2894736842. The search timed
out near the recorded wrist-3 bound without a collision rejection. Home retains
the same joint branch; this does not establish exhaustive physical infeasibility.
The exact replay is in `experiments/20261010T120821Z/replay_turn8_after_home`.
A separate diagnostic tried the existing declared
`ignore_current_height_for_travel_z=True` option with a recorded approach height;
it also remained unresolved (fraction 0.2857142857). It did not change production
policy, source observations, orientation, joint bounds, or solver settings and
does not count as accepted outline evidence. See
`experiments/20261010T120821Z/declared_pick_parameter_diagnostic_v4/report.json`.

The fifth fresh `gpt-5.6-sol low` attempt is retained in
`experiments/20261010T123123Z/attempt_5_gpt-5.6-sol/report.json`. It completed six
provider calls (28,326 tokens), including two context-request turns, proposed
three events, and accepted one square-placement event. The model independently
recognized that `gear_small` was outside `ur5e-4`'s declared reach and retrieved
`ur5e-3`'s capabilities. Its subsequent `ur5e-3` pickup remained `NEEDS_CONTEXT`
after incomplete native IK search. The exact next provider input included path
fraction 0.2894736842, query 22/error -6/timeout 0.05 seconds, and the last accepted
`ur5e_3_wrist_3_joint` value and declared bounds. The model repeated the same
resource, part, end conditions and absent parameters; the unchanged-evidence
guard stopped generation without appending it. The model was not given a
preferred recovery sequence. The parent recorded completed cancellation after
242.91 seconds, with 2,125,740 KiB peak child memory and no deadline expiry.

The independent `prefix_audit.json` and `independent_replay_report.json` beside
that report confirm the accepted event's exact provider provenance, saved RA
evidence, projected state chain, and replayed CCA history. Both selected rules
remain unchanged. Nominal continuation remains `not_ready`; no complete outline,
recovery execution or assembly completion is claimed. The audit made no API,
native-preview or live-equipment calls. All five repair attempts are consolidated
in `experiments/20261010T123123Z/repair_attempts_1_to_5.json` and the accompanying
text report. Implementation changed between those attempts, so these results
are not a controlled model comparison. Attempt 3's missing total duration and
unconfirmed termination cause are retained as unknown.

The latest automated checks are recorded in
`experiments/20261010T123123Z/automated_verification.json`: 249 validated-outline
and context tests, 22 diagnostic tests, and four targeted explicit-null/prompt/
artifact regressions passed. The broader artifact/request/UI suite passed 184
tests and retained one older `ra_admissible_event_ids` audit assertion failure.
These overlapping suites are separate runs, not additive totals. Poetry,
compilation and whitespace checks passed. Fresh route-calculation work, if
implemented, must supply separately supported owner evidence; incomplete IK
search is not exhaustive infeasibility. User-observed Gazebo recovery acceptance
remains unrecorded for these offline attempts.


The subsequent nominal-continuation repair recognizes supported completion of
an original product obligation by another declared owner. It compares exact
product effects and parameters, checks the projected completion record against
an accepted recovery event and its owner declaration, and preserves original
resource-state goals and cleanup tasks. The saved `ur5e-4`/event65 and
`ur5e-3`/event59 `place_insert` declarations exercise this distinction without
changing their identifiers. An LLM-authored `part_state="assembled"`, a missing
task entry, an unsupported effect, or an unresolved external dependency cannot
retire the original task. Historical same-owner behavior is retained. Separate
pending product groups are not combined to infer an external dependency;
uncertain cases remain pending.

The offline/workflow/context/validated suites passed 515 tests after this repair,
and the final nominal/completion selection passed 42 tests with the exact saved
declaration structure. The actual-declaration fulfillment check is explicitly
synthetic: it tests bookkeeping with independently supplied completion records,
not a newly validated physical assembly. The source observation hash is unchanged.
Records are in
`experiments/20261010T123123Z/nominal_owner_completion_repair.json` and
`nominal_owner_fulfillment_after_repair.json`. Poetry, compilation, and whitespace
checks passed. No API, native preview, or execution call was made for that check.

The older artifact assertion failure is now repaired: compact audit records
retain explicitly present `ra_admissible_event_ids`, including empty lists and
`null`, while absent fields stay absent. The case3/request-record/project-page
suites passed 186 tests in 20.47 seconds; the preceding failure record is
preserved. This changes audit fidelity, not recovery planning.

A sixth fresh attempt is staged in `experiments/20261010T131151Z`, with the same
source observations, selected safety specifications, initial prompt hashes,
`gpt-5.6-sol low`, Manual stages, 80-turn limit, and 600-second generation deadline.
It requires preview contract version 5 and records native route calculation
separately from strict trajectory replay. Route candidates alone are not
accepted event evidence. The staged runner also fingerprints the installed
action interfaces and native libraries, blocks equipment and execution calls,
and requires a separate saved-prefix audit before reporting complete success.
The staged runner has not called the API; generation waits for integration,
focused verification, and a frozen implementation.


The version 5 local replay of the exact attempt 5 pickup now establishes RA
`FEASIBLE` and complete physical projection, but CCA rejects
`SAFE_shared_area_mutex`. The derived continuous cells establish both `ur5e-3`
and `ur5e-4` occupying `assembly_board-v1`: observation 1 cells 9 and 10, and
observation 6 cell 0, each records both occupancy values as `[true]`. This is a
modeled safety violation, not an execution collision. The original first
counterexample displayed a coarser uncertain cell; the repaired diagnostic now
shows definite violating cell 9 with both AP values `[true]`. Exact saved CCA
replay took 8.39 seconds, retained monitor history, and separately found
`SAFE_gear_small_before_KET4_Square_4mm` feasible for this proposed event. The pickup was not appended,
and the accepted square-placement prefix, projected scene, and monitor history
were retained unchanged.

This replay reused four exact native results only after comparing every typed
Goal field; it made no API, new native, or equipment calls. Source observations,
original provider responses, and native records are unchanged. It took 170.84
seconds: 86.74 seconds for owner evidence preparation/finalization and 69.32
seconds for the validation retry. Records are in
`experiments/20261010T131151Z/local_attempt5_pickup_v5_retry4`; the complete exact
CCA replay input is retained separately as `cca_replay_input.json.gz`, and the
cell-level occupancy facts are in `cca_mutex_cells.json`.

The preceding preparation repairs are also verified: scene-object provenance
now constructs JSON lists instead of tuples, preserving exact equality after
saving without weakening conflicting-mapping checks (4 focused tests passed).
The continuous support-departure bound repair passed 39 focused tests and the
exact saved trajectory over 119 certified cells. Two synthetic audit tests
verify accepted-prefix order and no append after rejection. Native verification
passed 24 file-mode and 11 action tests, with 26 subtests; bootstrap completed
16 packages. Version 5 factory/historical restore tests passed 252 tests. The
full recovery outline and nominal continuation are still unproven; attempt 6
remains staged pending the CCA diagnostic check and implementation freeze.

The CCA factual feedback formatter also retains the recorded `cell_index`, exact
`interval`, and `definite_violation` in the PA prompt, alongside the rule-local
AP values. It adds no missing values or suggested recovery events. Eight focused
prompt and nominal-feedback tests passed; Poetry, compilation, and whitespace
checks passed. Initial prompt instructions are unchanged.


Attempt 6 ran the unchanged initial prompts with `gpt-5.6-sol low` and produced
six actual provider responses: three proposed events and two context-retrieval
turns, totaling 28,029 tokens. The owner rejected the first `gear_small` pickup
by `recovery-resource-4@localhost` because the calculated target is outside its
declared `gripper_reach`. The square-buffer placement by
`recovery-resource-3@localhost` passed RA, physical projection, and CCA and was
appended. The later pickup by `recovery-resource-3@localhost` reached physical
preparation, where the Python process unexpectedly exited with SIGSEGV (-11).
The run lasted 222.46 seconds, peaked at 1,888,304 KiB child RSS, and did not reach
the generation deadline. Kernel records confirm SIGSEGV; neither OOM nor a
specific Python function is established as the cause by the periodic stack.

The original partial report, provider captures, and native records are retained.
`experiments/20261010T131151Z/attempt_6_recovered` separately reconstructs the last
durable validated state and independently verifies its one accepted event;
nominal continuation and a complete outline remain false. Final in-memory guard
counters were unavailable after the crash and are recorded as unavailable.
Original observations and frozen implementation/native fingerprints match.

The exact pending pickup was then replayed locally from that saved prefix with
fatal tracing and phase instrumentation. It completed without crashing in
173.54 seconds, with four non-executing native calculations and zero API or
equipment calls. RA and physical projection passed; CCA rejected the established
`SAFE_shared_area_mutex` violation, leaving the accepted prefix unchanged.
Projected scene dictionaries contained plain Python values, no cycles or aliases,
and preserved exact JSON roundtrip equality. The installed generated action and
native library hashes matched the frozen run. This local replay does not explain
the intermittent crash or establish a complete recovery outline. Its records are
in `local_attempt6_pickup_fatal2`; the earlier wrapper import failure is retained
separately in `local_attempt6_pickup_fatal`. A full six-response local replay is
staged but has not run. No production workaround or validation relaxation was
introduced for the unexplained crash.

The combined CCA, outline, context, and offline regression run finished with
650 passing tests. A new actual attempt is staged separately in
`experiments/20261010T140822Z`, with fatal tracing enabled and the same generation
limits; its generated outcome and independent audit are recorded separately.


Attempt 7 produced seven actual provider responses, four proposed events, two
context-retrieval turns, and 35,723 tokens. The exact square-buffer placement
(`place_part`, `recovery-resource-3@localhost`) and subsequent `move_home` by
`recovery-resource-4@localhost` with `part_name=null` passed RA, physical
projection, and CCA. The first `gear_small` pickup was rejected by the definite
`SAFE_shared_area_mutex` counterexample; its exact cell, interval, and AP values
were included in the following prompt. The model then independently proposed
`move_home`. During the next pickup's route-evidence preparation, the interpreter
exited with SIGSEGV (-11), after 463.02 seconds and 2,049,708 KiB peak child RSS.
This was not the generation deadline. No complete outline or nominal continuation
was established.

`experiments/20261010T140822Z/attempt_7_recovered` separately preserves the last
durable state. Its independent audit verifies both accepted events and their
state/history chain, with no audit errors or equipment calls. Original
observations and frozen source/native fingerprints match. The crashed run's
final in-memory guard counters remain unavailable and are not reported as zero.

A full fresh-state replay then consumed six exact saved responses under gdb and
Python's debug allocator, preserving the original structured request adapter,
provider worker, parser, and cancellation path while blocking HTTP. It reproduced
SIGSEGV after `move_home` physical preparation, during the following validation.
The debugger caught a dereference of `0xdddddddddddddddd`, the debug allocator's
freed-memory fill, in Python function/frame handling. This narrows the failure
to memory lifetime corruption but does not yet identify its cause. The replay
is explicitly labelled as replayed, not a new provider capture; gdb's exit code
0 does not imply success. Records are in `local_attempt7_full_debug`. Periodic
stack sampling is being investigated separately from fatal tracing; no
production workaround or evidence requirement has been changed.

A controlled standard-library-only reproduction identifies the periodic stack
sampler as a concrete corruption source in the installed Python binary. With
`faulthandler.dump_traceback_later` enabled, two runs failed after 0.11 and 0.67
seconds (the second with SIGSEGV); matching fatal-only controls both completed
20 seconds. Disassembly shows the background sampler changing code-object
reference counts without holding the GIL, consistent with the freed
`co_zombieframe` pointer caught in the full replay. No ROS, project, API, or
third-party code was imported by the minimal probe. Detailed evidence is in
`attempt_7_gpt-5.6-sol/crash_diagnostics/faulthandler_reproduction`. A separate
full saved-response replay retains fatal tracing and gdb while removing both
periodic timer entrypoints. Its result remains pending; validation requirements
and source observations are unchanged.

The full fatal-only saved-response replay completed normally in 617.02 seconds
under gdb and the debug allocator. It replayed exactly the seven retained
responses with zero API/equipment calls and unchanged source/code/native hashes.
Square placement, `move_home` by `recovery-resource-4@localhost`, and the final
`gear_small` pickup by `recovery-resource-3@localhost` passed the runtime RA,
physical projection, and CCA checks. The earlier pickup still received the same
modeled mutual-exclusion rejection. No validation requirement was weakened.
The full reproduction with periodic sampling had crashed; the same lifecycle
without periodic sampling reached this result.

Nominal continuation remains correctly `not_ready`: `environment_8` requires
`recovery-resource-4@localhost.resource_state= picked`, but the projected value is
`idle`; `ed9a5f9c8c4b463496402821945cfe72_pending_3` requires
`held_part=KET4_Square_4mm` on `recovery-resource-3@localhost`, which currently
holds `gear_small`. These are explicit conditions for further proposal feedback.
The replay stopped before requesting any eighth response. It is not a new model
attempt, complete recovery outline, or execution. Records are in
`experiments/20261010T140822Z/local_attempt7_fatal_only`, including a separate
verification summary and the unchanged-source report. The real API generation
deadline remains 600 seconds; the 1,200-second cap was solely for this instrumented
local diagnostic.

The geometry calculation now reuses repeated FK results only within a fixed-input
motion checkpoint. On the exact saved real `gear_small` pickup fixture, complete
projection time decreased from 28.21 to 18.41 seconds (34.7%). The complete
projection, CCA result, counterexample, and monitor-history hashes are identical;
the same mutual-exclusion rejection remains. This measured result is separate
from the earlier synthetic approximately threefold improvement. No evidence
bound or validation requirement changed, and no cache crosses validation calls.
The comparison made no API, native, or equipment calls. Its source hashes, exact
outputs, fixture hash, and timing are retained in
`experiments/20261010T140822Z/geometry_performance/report.json`. Focused regression
results are recorded there separately when complete.

The geometry optimization regression run completed with 310 passing tests in
43.02 seconds; Poetry, Python compilation, and whitespace checks passed. The
implementation is frozen before the next actual model attempt.

Attempt 8 used actual `gpt-5.6-sol low` responses with fatal-only tracing and the
verified geometry optimization. It did not crash. Nine provider responses
contained five proposals and three context-retrieval turns (45,956 tokens). The
accepted square placement, `move_home` by `recovery-resource-4@localhost`, and
`gear_small` pickup by `recovery-resource-3@localhost` independently passed the
saved RA/state/CCA-history audit. The source capture and implementation/native
hashes were unchanged during the attempt; equipment/execution guard calls were
zero. Nominal continuation and the complete outline remain unproven.

The first pickup received the established mutual-exclusion rejection. The model
then retrieved the failed resource's snapshot and measured pose and proposed
`move_home`. Its later gear-placement proposal reached a checked native collision
between `Gear_Plate/Gear_Shaft_1/collision` and `gear_small/link/collision`; this
placement was rejected and did not alter the accepted prefix. The fixed
600-second generation deadline was exceeded; process shutdown and cancellation
completed at 639.90 seconds, with 2,148,912 KiB peak child RSS. The runner failed
to catch Python 3.10's distinct `asyncio.TimeoutError`, so its original report's
`error=null` is preserved and corrected only in a separate
`attempt_8_gpt-5.6-sol/outcome_annotation.json`. Exact provider captures, native
requests/results, and `prefix_audit.json` are retained under
`experiments/20261010T144129Z`. This is a partial validated outline, not a complete
recovery result; the gear-placement geometry is being investigated locally.


A separate saved-proposal diagnostic now validates Attempt 8's exact `gear_small`
placement after its three previously accepted events. It uses the unchanged
capture and historical version-5 prefix with version-6 calculations for this
pending event only. The recorded attachment link and the motion control link are
accepted as equivalent only through their verified fixed URDF connection; a
movable connection remains rejected. The three isolated native previews,
continuous mating evidence, projected fixture attachment, RA, and both CCA rules
passed. The event was appended, leaving four accepted events. Nominal continuation
remains `not_ready`: the recorded Square `place_approach` requires `resource_state`
`picked`, while its current projected value is `placed`. This is a local diagnostic,
not a fresh successful model attempt or executed recovery.

The diagnostic completed in 604.57 seconds: physical preparation took 528.65
seconds and the runtime validation retry took 44.65 seconds. A private one-entry
owner projection cache reused only its own successful calculation with identical
complete inputs and projector identity; the real projection took 177.94 seconds
initially and 2.28 seconds on the matching retry. CCA was checked independently.
Fifteen focused cache tests cover mutation isolation, changed inputs, missing
versus null, external route-file changes, untrusted supplied results, and the
single-entry limit. Source observations, historical records, provider response,
and code/native fingerprints stayed unchanged during this diagnostic; API and
forbidden equipment calls were zero. Original failed fixed-link diagnostics remain
separate. Records are under `experiments/20261010T151845Z`, including
`local_attempt8_placement_v6_retry/report.json` and
`owner_projection_cache_verification.json`. Repeated continuous mating proof
calculations remain a measured performance bottleneck before the next fresh
600-second model attempt.


The exact placement diagnostic was repeated after reusing fixed FK origins and
one locally produced successful continuous mating proof. It again accepted the
same fourth event and retained the same nominal obstruction. Total time decreased
from 604.57 to 179.47 seconds; physical preparation decreased from 528.65 to 98.97
seconds. All six mating checks returned the same complete 2,617-cell proof as the
baseline. The first calculation took 43.67 seconds; subsequent exact-input checks
took 0.139–0.169 seconds. Accepted events, derived and symbolic state, and CCA
monitor history match exactly. Independent native reruns have different measured
execution durations and artifact paths/hashes; only their three corresponding
response fingerprints differ in the projected evidence. All other native request,
trajectory, collision, and geometry fields match. Full differences and the exact
comparison are retained in `mating_optimization_comparison_final.json`; no API or
equipment call occurred. The mating/preview/outline regression run passed 490
tests, and the broader validation/UI/retrieval integration run passed 875 tests.

A separate audit of the recorded Square task found a configuration contradiction:
its declared origin was at z=1.0239916, above the actual supporting table at
z=1.015. Its 4 mm collision box fits inside the recorded 4.11 mm through-hole;
the complete plate mesh has no positive overlap with that footprint. There is
no declared retained fixture attachment for this part. The geometry declaration
was corrected to the measured-model table support height, with the original
capture unchanged. `square_recorded_release_support_audit.json` and
`square_geometry_repair.json` retain the exact source geometry and calculation;
311 surrounding geometry tests passed. A new capture is required before a fresh
model attempt because the configuration fingerprint changed. Attempt 9 remains
staged with explicit source/hash arguments, fatal-only tracing, and the unchanged
600-second/80-turn limits. Eleven mocked runner tests pass, including source
immutability and cancellation. Neither this configuration repair nor the saved
placement replay establishes a complete recovery outline or executed assembly.


A new isolated Gazebo failure capture was recorded on October 10 at 16:18:26 UTC
(`capture_ab5fbd8696e24a5284470ec127fa5334`, SHA256
`f6cb7a09f7a49e05fb1ae4987d4261d19ec962389914b1fe5b5bdcb799d1fd31`).
Both pickups and the intentional lowering failure were observed, and the Square
geometry uses the corrected z=1.015 declaration. No LLM request or recovery
execution occurred during capture. The previous source and UI records remain
unchanged. Detached Manual input preparation succeeds, but this capture is not
sufficient for whole-scene physical validation: `RGOCG4-50_Round_4mm` has recorded
angular speed 0.0028648084 rad/s, above the unchanged 0.001 rad/s rest criterion.
The top-level `physical_evidence.unresolved` list omitted this child
`stationary_part_evidence` status. The first capture audit checked only that
coarse list; its original reports are preserved and corrected separately in
`experiments/20261010T160847Z/fresh_capture_audit_correction.json`. The next audit
requires supported evidence for every unheld part.

New attempts now use physical preview contract 7 while RA remains contract 2.
Exact nominal task coverage accounts for each recorded step, guard, parameter,
calculated output, custody change, and simulation `snap_part_to_slot`. The snap
has separate scene-endpoint evidence; it is not called a continuous movement.
Owner task context is carried from independently replayed Cartesian endpoints
using the existing runtime bookkeeping. Changed declared product geometry,
unsupported steps, untrusted owner facts, and conflicting declared state effects
remain unresolved. Historical contracts retain their recorded behavior. Focused
owner/witness/workflow tests passed 559 cases; owner-authority checks passed 476;
effect-consistency checks passed 354. Endpoint verification separately passed
735 Python, 33 native file, and 11 native action cases, with bootstrap, Poetry
checks and compilation. These overlapping suite totals are separate runs.

A no-API replay of retained Attempt 8 responses against the new capture first
stopped at an exact native model-fingerprint mismatch (28.34 seconds). Its file
adapter used the previous capture's SRDF bytes. A separate parameter archive now
uses the exact new captured URDF/SRDF and archived local MoveIt declarations;
the native fingerprint matches `2c8664d56332c8406155a1121b3610e6d93054300f260f2079d8b5f5c2636602`
exactly, and the same saved native Goal completes. URDF, kinematics and planning
parameters are equal to the old archive; SRDF bytes differ. The matching proof
is retained in `model_preview/provenance.json`, without changing either capture.
The fresh capture does not itself archive every typed native startup parameter;
that future capture gap is recorded separately and is not claimed fixed.
A second no-API replay correctly stopped at the unresolved Round part's rest
evidence (31.19 seconds). Neither replay appended an event; both preserve source,
implementation and native hashes and report zero API/equipment/execution calls.
Attempt 9 remains unstarted pending a new capture with sufficient recorded rest
evidence; no thresholds or observations are changed to obtain acceptance.


A replacement isolated capture was saved at 17:03:11 UTC on October 10 as
`capture_f923020bf9ce496eb443826291f8f2c2` (SHA256
`e771f6b9d0f43e0f9efb2d51d779cdb1c92835313f14b28bd7dfde0897cb6e5e`).
The capture waited for recorded stationary evidence for every unheld part: the
first two samples retained unresolved moving parts, and the third supported all
of them under the unchanged criteria. All owned Gazebo processes were then
stopped. The exact native startup parameter files were archived while alive;
the file-mode native fingerprint matches the saved model
`cf377822f8e5148dc56343121aa67edc5f58064fd1894b683be1fb662b56c2a5`.
Independent detached preparation passed the stricter child-evidence checks,
actual pickups/drop, corrected geometry, exact selected safety specifications,
initial AP source fingerprints, retained history, and Offline Manual status.
It made zero API/live/execution calls and preserved both previous captures.
Evidence is in `experiments/20261010T160847Z/fresh_capture_f923_audit/report.json`
and `experiments/20261010T160611Z/fresh_capture_stationary_2/`. This confirms
initial saved evidence and preparation, not acceptance of future motions.
A fresh-source replay of the retained Attempt 8 responses is staged separately
in `experiments/20261010T170458Z`; actual Attempt 9 remains unsent until its
local validation outcome is known.


Fresh-source local validation then replayed all nine retained Attempt 8 responses
without API calls. It accepted four events after RA, physical, and CCA checks;
the earlier pickup remained rejected for a definite mutual-exclusion violation.
The remaining nominal resource3 task required `resource_state="picked"` while
the accepted projected state was `placed`. The replay stopped before a new
response, preserving the capture and implementation; elapsed time was 624.60s,
including 474.08s of physical preparation and validation. This is a local replay,
not a fresh complete model result. Exact records remain in
`experiments/20261010T170458Z/local_attempt8_fresh_v7/`.

Actual GPT-5.6-sol low Attempt 9 ran afterward with the unchanged 600-second and
80-turn limits. It independently proposed `recover_resource_4_home` first;
that event passed RA, physical projection, and CCA and was independently audited
against its exact provider capture. The attempt stopped cleanly after 179.34s
(183.50s process time), three provider responses and 13,375 tokens, with one
accepted event. It did not complete or time out. The next unfinished nominal
`place_approach` had a successful physical preview but RA could not bind the
calculated `place_targets.approach_pose` / `target_pose` x/y/z values to its
independent `move_cartesian` capability checks. No nominal state was committed.
This concrete parameter-preparation gap is under repair; missing evidence was
not bypassed. Peak child RSS was 2,339,428KiB; no forbidden live/execution calls
occurred, and source/Python/native fingerprints stayed unchanged. Report,
independent prefix audit, and exact local RA replay inputs are retained under
`experiments/20261010T170458Z/attempt_9_gpt-5.6-sol/`.


The exact Attempt 9 nominal task was replayed locally after repairing RA's
calculated `place_targets` parameter witnesses; RA became FEASIBLE. Focused RA
and outline verification passed 483 cases, with the same 12 unrelated legacy
AP/KMR fixture failures recorded separately. The next observed blocker was CCA:
it stopped at an early uncertain occupancy boundary, hiding a later supported
Square region entry. CCA now retains possible safe monitor states through later
observations while preserving the earliest uncertainty unless a definite
violation is established. The unchanged saved trajectory proves precedence
failure at observation 5 (`ap001=[true]`, `ap002=false`); mutual exclusion remains
FEASIBLE. No occupancy threshold, AP meaning, path, or geometry was changed.
The CCA repair passed 468 focused tests, Poetry checks and compilation.
A full retained-evidence retry then returned that exact factual rejection to PA
and restored `outline_ready` for another proposal; it did not mark the outline
complete or commit nominal state/history. This check took 85.02 seconds with
zero API/native/live/execution calls, and preserved source/implementation
hashes. Its feedback is recorded in
`experiments/20261010T170458Z/local_attempt9_nominal_cca_retry/feedback_summary.json`.
Fresh actual Attempt 10 is kept separately in `experiments/20261010T173709Z`,
with the same capture, 600-second/80-turn limits, and fresh provider responses.
The production capture provider still omits a complete typed native startup
parameter archive; the experiment capture explicitly archived those actual
startup files and verified its exact native model fingerprint separately.


Actual Attempt 10 reached the unchanged 600-second generation deadline while
preparing its fourth proposed event. GPT-5.6-sol low returned seven real responses,
four proposed events and two retrieval turns (four exact context references),
using 41,682 tokens. Three events were accepted in the model's original order:
`move_home` for `recovery-resource-4@localhost` with `part_name=null`, `place` of
`KET4_Square_4mm` at `Buffer For Machined parts` by
`recovery-resource-3@localhost`, and `pick` of `gear_small` by that resource.
The fourth proposal was `place` of `gear_small` at `assembly_board-v1`; its
physical preparation was interrupted by the deadline, so it was not appended.
The nominal Square `place_approach` was rejected for its supported precedence
violation and returned factual feedback before the second recovery proposal.

Independent audit verified all three accepted events against exact provider
captures, RA candidates, physical bindings, CCA verdicts, projected states and
monitor history. Nominal continuation and the complete outline remain
unestablished. Runtime was 623.29 seconds and total process time 627.59 seconds,
including cancellation and artifact cleanup; cancellation completed. Peak child
RSS was 4,006,100KiB. Source, parent capture, Python and native fingerprints
stayed unchanged, and no forbidden live or execution call occurred. Original
report, precise timings, token usage and independent audit remain under
`experiments/20261010T173709Z/attempt_10_gpt-5.6-sol/`. This is a valid partial
outline, not a complete success. The next work is a measured repair of repeated
archive copying and JSON writing; generation limits and validation requirements
are unchanged, and another API attempt has not started.


Before Attempt 11, narrow read-only-copy repairs preserved exact saved nominal
evidence, projected state and CCA history while reducing one full retained retry
from 70.2226s to 57.6137s (17.96%). The complete saved session bytes were
identical. Verification passed 783 focused cases, 11 runner guards, Poetry checks
and compilation. The writer retained its streamed memory behavior.

Actual GPT-5.6-sol low Attempt 11 used exactly the same initial prompt hashes as
Attempt 10. It stopped after 348.04s (353.04s process time), not at the deadline.
Four real responses used 20,644 tokens: grounding, `move_home` for
`recovery-resource-4@localhost`, one context request for that resource's
capabilities and `gear_small`'s target, and `place_approach` for
`recovery-resource-3@localhost` carrying `KET4_Square_4mm` at
`Buffer For Machined parts`. The first event passed the independently audited
provider/RA/physical/CCA chain. The second proposed `resource_state="positioned"`
and `part_state="in_gripper"`; it was not appended.

The RA successor union had a matching `place_approach` contributor and a feasible
`compute_place_targets`; its `move_cartesian` checks lacked scoped witnesses for
`place_targets.approach_pose` and `target_pose`. Independently, the owner projection
reported a conflict between its declared effect and the proposed `part_state`.
These are retained as separate findings; the broad RA summary must not be read
as an empty union. The runtime left the accepted prefix and monitor history
unchanged. Nominal continuation remains incomplete. Peak child RSS was
3,858,100KiB; source/Python/native fingerprints stayed unchanged and forbidden
calls were zero. Exact report, findings and audit are retained in
`experiments/20261010T175921Z/attempt_11_gpt-5.6-sol/`. Another API attempt remains
held until the concrete parameter and feedback paths are repaired and checked
locally. The stronger audit is staged separately so historical audit files are
not rewritten.

The Attempt 11 buffer parameter repair made its exact RA proposal `FEASIBLE`.
The separate owner-effect check now returns factual revision feedback for the
established `part_state` conflict (`in_gripper` requested, `in_transit` declared),
without a native preview, append, projected-state change, or monitor commit.
The exact local retry took 25.2097s without profiling; its complete saved session
bytes matched the profiled replay. The profile exposed streamed artifact writing
and copied historical evidence as costs; the small isolated RA-copy prototype
was not adopted. Verification included 1,091 focused tests, 23 producer tests,
27 runner/audit guards, Poetry and compilation. The full feasibility suite
retained 12 unrelated legacy AP/KMR failures. Separate records are under
`experiments/20261010T181625Z/`.

Actual GPT-5.6-sol low Attempt 12 retained the same initial prompts and immutable
`capture_f923020bf9ce496eb443826291f8f2c2` observations. Seven real responses used
44,563 tokens. Two public retrieval turns requested five references. Four events
were proposed; three were appended after RA, physical projection and both CCA
checks: `recovery_move_r4_home_1`, `recovery_place_ket4_buffer_2`, and
`recovery_pick_gear_small_r3_3`. The independently replayed original Square
nominal approach remained rejected by its precedence rule. The final proposal,
`recovery_place_gear_small_r3_4`, was still in physical preparation when the
600s generation deadline fired. Cancellation completed; process time was
611.02s and peak child RSS was 3,931,156KiB. This attempt is incomplete, not a
model or execution success.

The separate prefix audit verified exact provider authorship, all three accepted
RA/physical/CCA chains, complete scene propagation and monitor history. Source,
parent capture, Python and native hashes were unchanged; forbidden calls were
zero. Reports and audit are preserved in
`experiments/20261010T181625Z/attempt_12_gpt-5.6-sol/`. The completed pickup's
physical preparation took 78.15s and validation retry 32.77s. The next step is a
local, exact-input positive preparation/validation profile with retained native
responses; no unchanged API repetition or increased generation limit is planned.

The positive Attempt 12 pickup was then replayed locally with its independently
reconstructed two-event prefix and exact provider proposal. Each of four saved
native responses was reused only after every typed Goal field matched. The
baseline unprofiled Python preparation took 69.624s and retry 32.951s. Selective
profiling identified repeated ROS message subtree copies, repeated static joint
transforms in the support-departure proof, and detached copies of historical
validation data. The streamed writer was retained: its actual unprofiled runtime
writes cost 6.49s, substantially less than their profiler-inflated timings.

Narrow copy and static-calculation repairs reduced the same preparation to
57.535s and retry to 31.241s: combined 102.575s to 88.776s (13.45%). The complete
prepared evidence and complete resumed runtime session retained identical
semantic fingerprints, including all projected states and CCA monitor history.
The replay's archive scope used the original turn 6 as its one full turn; this
is a controlled before/after local comparison, not another real model attempt.
No new API, native planning, or equipment call was made. Evidence and profiles
are retained under `experiments/20261010T184631Z/`. Verification passed 494 nearest
outline cases, 11 ROS conversion cases, 45 support-departure cases, and 30
conservative owner-effect expression cases. The latter guard leaves the current
capture's 48 supported resource updates unchanged. Attempt 13 is staged in
`experiments/20261010T190600Z/`, with the same source, initial prompt hashes,
model/effort and 600s/80-turn limits; it has not started at this update.

After the final entry-copy regression checks and 1,120 combined tests, actual
GPT-5.6-sol low Attempt 13 ran with the same capture and initial prompt hashes.
It stopped at `NEEDS_CONTEXT` after 314.85s of generation (329.43s process time),
before the deadline. Five actual responses used 28,628 tokens. Two retrieval
turns requested four public references. The newly named `recovery-step-1`
(`move_to_named_pose`, `recovery-resource-4@localhost`, explicit `part_name:null`)
passed and was independently audited. The original nominal Square approach was
again rejected by its precedence rule without a state or monitor commit.

The next actual proposal was `recovery-step-2`, a loaded `place_approach` by
`recovery-resource-3@localhost` to `Exit`, retaining `KET4_Square_4mm` and proposing
`part_state="in_transit"`. RA found the matching successor union; it did not fail
because the event name was new or the union empty. The owner calculation lacked
a declared support surface height for `Exit`, leaving the approach/target xyz
parameters unresolved. Physical preparation independently reported that the
recorded geometry could not establish the requested loaded destination. CCA was
not checked for this unresolved proposal; nothing was appended. This is neither
a geometric rejection nor complete outline success. All source, parent capture,
Python and native integrity gates passed; no forbidden call occurred. Exact
findings, usage and the independent one-event audit are in
`experiments/20261010T190600Z/attempt_13_gpt-5.6-sol/`. Another API call remains
held while the recorded `Exit` evidence boundary is assessed.


### 2026-10-10: measured retrieval-generation overhead and staged Attempt 14

- Replayed only the exact saved API13 retrieval responses for turns 3 and 4, through the original provider worker with HTTP, native planning and execution blocked. Both reconstructed prompts matched their saved provider requests exactly; accepted prefix, projected state and monitor history remained unchanged.
- The scoped profile measured 223.67 s in five full debug publications out of 239.17 s under cProfile. Prompt construction was 3.01 s and artifact writing 0.012 s; profiled durations are diagnostic overhead, not API latency.
- Removed only the redundant end publication for contract-2 successful `need_context` turns that immediately continue. The synchronous artifact writer receives the current session; the next loop still publishes independent snapshots before its first await. Terminal, failed, paused, legacy and exhausted paths retain their publications. Exact request/audit/resume bytes and mutation isolation are covered by regressions.
- Unprofiled exact retrieval replay improved from 87.738 s to 51.018 s (41.85%). Original prompts/responses and retrieval artifact facts match; only local replay provenance paths differ. Capture and original session hashes are unchanged. Evidence: `experiments/20261010T190600Z/retrieval_optimization.json`.
- Verification: 551 focused tests passed, 11 targeted publication/artifact tests passed, `poetry check` and full Python compilation passed. No archive pruning, validator weakening, changed generation limits or suggested recovery events were introduced.
- Attempt 14 is staged under `experiments/20261010T193410Z`, using the same `capture_f923020bf9ce496eb443826291f8f2c2`, model/effort, selected specifications and 600-second/80-turn controls. Its API is not started; the original API13 `Exit` proposal must first complete the separate local validation replay. The pure geometry calculation result is recorded separately from full RA/physical/CCA acceptance.


### 2026-10-10: Attempt 14 retained four accepted events and timed out

The separate API13 `Exit` replay completed before the fresh attempt. The repaired
owner calculation established the receiver height from recorded geometry; RA
was `FEASIBLE`. Its native direct path collided with `ur5e_3_upper_arm_link`, and
the alternative route lacked the required fixed starting orientation. The
workflow returned factual revision feedback without appending or changing its
accepted state/history. That local result is not another model attempt.

Actual GPT-5.6-sol low Attempt 14 used eight provider responses and 50,220 tokens,
with two retrieval turns and five proposed events. Four passed: the resource-4
home event, Square placement at `Buffer For Machined parts`, resource-3
`gear_small` pickup, and its placement at `assembly_board-v1`. The exact fifth
provider proposal was `pick_ket4_for_nominal_continuation`, a resource-3 pickup
of `KET4_Square_4mm`. Its physical preparation was interrupted by the unchanged
600-second deadline. Workflow elapsed time was 606.109 s, including delayed
cancellation; process elapsed time was 613.055 s. This is a timeout, never a
within-deadline completion. Completed preparation/retry stages totalled
410.770 s; nominal continuation remained `not_ready`.

The separate audit verified all four exact provider/state chains, saved
RA/physical evidence bindings, and recomputed CCA histories. It did not perform
new RA derivation or native preview. Capture, parent, Python and native hashes
were unchanged and forbidden calls were zero. Reports, the independent audit,
and a separate deadline attestation are retained under
`experiments/20261010T193410Z/attempt_14_gpt-5.6-sol/`. No recovery execution was
performed. Attempt 15 is only staged: the next local work validates the actual
fifth proposal and recorded nominal tasks, and measures archival of full nominal
proofs without pruning runtime validation inputs. New audit guards require an
explicit terminal timestamp within the deadline, a clean reaped process, and
manifest membership for every selected provider/validation proof artifact.


### 2026-10-10: complete nominal proof archival and local fifth-proposal check

New attempts save the full nominal validation record before replacing repeated
in-memory diagnostic copies with its exact relative path, SHA256 and accepted
state/source/model/safety binding. The complete proof remains downloadable and
expandable; missing, changed or incompatible files cannot establish completion.
Pending nominal tasks/scenes, accepted state, monitor history and validation
inputs remain intact. Historical embedded proofs retain their prior behavior.

Verification passed 1,335 cases across nine focused suites (53.35 s), 72 UI
cases, 111 targeted archive cases and 48 runner/audit guards. `poetry check`, full
Python compilation and diff checks passed. An earlier sandbox selector stall
was interrupted/reaped and is not recorded as a complete test pass. Evidence is
under `experiments/20261010T195300Z/automated_verification.json`.

The exact API14 fifth provider proposal is now undergoing a separate local
1200-second diagnostic from its four-event independently audited prefix, with
new private native file calculations and subsequent recorded nominal tasks.
There are no new provider, equipment or execution calls. A separate retrieval
benchmark will include archival cost and compare the complete hydrated proof,
public prompts and accepted state/history before any actual Attempt 15 launch.


The retained-prefix diagnostic completed in 356.326 s with no API/forbidden call
and unchanged source, session-source, Python and native hashes. The actual fifth
Square pickup passed and appended locally. The recorded nominal `place_approach`
also passed. The next recorded `place_insert` produced two complete native motion
results and a separate complete snap endpoint, but exact nominal program coverage
correctly remained unresolved. Its `lift` step resolves `dz=0.052500000000002656`
from the current nominal task context (target z=1.3384999752316755); the preview
instead used the prior pickup travel height (target z=1.500773097154814). Insert,
release and snap step coverage had passed. This is a parameter-source mismatch,
not floating-point noise or a missing native result. The five-event recovery
prefix remained retained; no execution or complete nominal result is claimed.
The exact witness and compact replay input are saved under
`experiments/20261010T195300Z/local_api14_pending_replay/`. Attempt 15 remains held
for this narrow repair and the separate archival performance comparison.

The nominal departure repair now retains the exact authoritative `_task_ctx` for
`place_insert`, including its explicit null fields, and binds cached coverage to
that recorded context. The corrected lift target is z=1.3384999752316755. A new
local native retry passed insertion, the separate snap endpoint, lift, RA and
CCA, then exposed a final-goal reader defect: supported `part_state=assembled`
and `part_location=assembly_board-v1` were being shadowed by the retained capture
fields `current_state=misplaced` and `current_location=null`. Version-2 completion
now reads its supported fields first, distinguishes absent from null, and keeps
historical behavior. The source capture was not modified.

The subsequent exact retained-evidence retry completed the nominal suffix in
258.731 s, including persistence, with no new API, native or live calls. The
independent audit matched all five accepted events to their original API14
provider responses, replayed both CCA histories and derived state transitions,
validated exact nominal task ownership/program coverage and replacement evidence,
and checked final goals. `accepted_chain_valid`, `nominal_chain_valid` and
`nominal_complete` are true, with no unfinished task or audit error. This is a
local saved-provider replay; `fresh_generation_valid` and fresh `outline_complete`
remain false. It establishes no recovery execution. Evidence is in
`experiments/20261010T195300Z/local_api14_nominal_goal_retry_2/independent_audit.json`.

Final focused verification passed 1,377 cases across nine suites, 75 runner/audit
guards and the earlier 72 UI cases. `poetry check` and full Python compilation
passed. The staged fresh-attempt gate additionally binds every provider request
and response to its current process, saved source, empty generation seed and
artifact root. The exact archival-cost benchmark precedes actual Attempt 15;
its 600-second deadline and 80-turn bound remain unchanged.


### 2026-10-10: archival benchmark and fresh Attempt 15 deadline result

The exact saved retrieval benchmark includes full archival creation, writing and
hashing. Generation time fell from 48.720 s to 16.878 s (65.36%); the latter
includes 6.280 s to archive the proof. Total benchmark time fell from 72.241 s to
36.977 s and peak RSS from 3,177,780 to 1,593,672 KiB. Both public prompts, the
complete hydrated nominal proof, accepted state/history and retrieved context
matched exactly. Original artifacts were unchanged; API, native and forbidden
calls were zero. Results are in
`experiments/20261010T195300Z/nominal_archive_comparison.json`.

Fresh Attempt 15 used the same immutable `capture_f923020bf9ce496eb443826291f8f2c2`,
`gpt-5.6-sol`, low reasoning, 80 turns and a 600-second deadline. Eight actual
provider responses contained two retrieval requests and five recovery proposals
(51,135 tokens). Four events were accepted before the deadline. The fifth
Square pickup also validated, but its synchronous retry ended at 640.147 s; the
explicit stage boundary raised `TimeoutError`. This is not within-deadline
completion. The process was fully reaped, with no recovery execution.

The independent audit validates all five exact provider/state chains, saved
RA/physical bindings and replayed CCA histories. Fresh-generation identity,
manifest membership, source, parent, Python and native integrity all pass;
forbidden calls and audit errors are zero. Square `place_approach` and
`place_insert` remain unfinished, so `nominal_complete`,
`completed_within_deadline` and `outline_complete` are false. This audit performs
no new RA derivation or native preview. Records are retained under
`experiments/20261010T195300Z/attempt_15_gpt-5.6-sol/`.

Exact stage boundaries attribute 5.463 s to input preparation, 113.516 s to
generation (37.723 s provider calls and 75.793 s other generation work),
304.170 s to physical preparation plus evidence persistence, and 216.998 s to
validation retry. The final physical stage used 44.394 s and its retry 50.921 s.
The next work is measured performance repair and exact saved-result parity;
an unchanged fresh Attempt 16 has not been launched.

### 2026-10-10: Attempt 16 and exact named-pose endpoint repair

The user removed the ten-minute comparison deadline. Fresh Attempt 16 used the
same immutable `capture_f923020bf9ce496eb443826291f8f2c2`, unchanged initial
prompts, `gpt-5.6-sol` with low reasoning, and the existing 80-turn bound. It
finished normally in 584.12 s; no timeout occurred. Nine actual provider
responses used 56,709 tokens and proposed five events. Four appended after RA,
physical projection and both CCA requirements. The independent audit confirmed
all four provider/state/evidence chains, fresh-generation identity and unchanged
source, code and native inputs, with no forbidden call or audit error. Nominal
continuation remained incomplete.

The fifth proposal, `clear_assembly_board_resource_3` / `move_home`, exposed a
native interpolation defect: `ur5e_3_wrist_3_joint` ended at 5.142049823384552
instead of its exact named target 5.142049823384553. The final waypoint now uses
the exact target before the existing bounds, contact and trajectory checks.
Post-timing endpoint equality is checked; no tolerance or contact rule changed.
`make bootstrap-gazebo` completed 16 packages and the native suite passed 46
tests plus 42 subtests, including collision rejection at interior and final
states. Replaying the exact saved Goal produced exact first/final configurations
and retained all peer joints.

A separate 69.63 s local replay then accepted this same fifth provider proposal
through RA, physical projection and CCA. It made one private native calculation,
zero API/live/execution calls, and preserved all recorded source/code hashes.
The nominal result correctly remains `not_ready`: the next recorded task requires
`resource_state=picked`, while the supported projected state is `idle`. This is
factual feedback for another LLM proposal, not an inserted recovery event.
Evidence is under `experiments/20261010T211008Z/`, including
`home3_endpoint_diagnosis/` and `local_api16_home3_replay/`.

The next fresh attempt will exercise the production Offline Generate operation.
The implementation connects saved-model native calculations and same-proposal
validation retries to that operation, retains immutable companion model files,
and allows completed outline review to reload separately from primitive
composition. Focused integration verification is in progress. No user-observed
Gazebo acceptance or recovery execution is claimed for this work.

### 2026-10-10: fresh Attempt 17 completes the offline outline

Fresh `gpt-5.6-sol` with reasoning effort `low` completed the production Offline
Generate operation from unchanged
`capture_f923020bf9ce496eb443826291f8f2c2/observations.json` and its separately
stored, verified `native_preview_model`. Generation took 1,029.50 s, including
33.27 s in eight actual provider requests. It used 51,759 tokens, two public
context retrieval turns and 21 private file-native calculations. Gazebo,
perception, controller, SPADE transport, primitive generation and execution
calls were blocked; no prohibited call occurred. No event count or recovery
sequence was added to the prompts. Initial prompts matched Attempt 16 exactly.

The five independently proposed and accepted events, in order, were:

| `outline_id` | `event_name` | `resource_jid` | `part_name` |
| --- | --- | --- | --- |
| `clear_failed_robot_from_assembly_board` | `move_home` | `recovery-resource-4@localhost` | `null` |
| `stage_ket4_at_buffer` | `place` | `recovery-resource-3@localhost` | `KET4_Square_4mm` |
| `pick_misplaced_gear_small` | `pick` | `recovery-resource-3@localhost` | `gear_small` |
| `place_gear_small_on_assembly_board` | `place` | `recovery-resource-3@localhost` | `gear_small` |
| `reacquire_ket4_for_nominal_continuation` | `pick` | `recovery-resource-3@localhost` | `KET4_Square_4mm` |

The independent saved-evidence audit passed all five provider/RA/physical/state
chains and replayed both exact CCA specifications, `SAFE_shared_area_mutex` and
`SAFE_gear_small_before_KET4_Square_4mm`, with retained monitor history. It
verified nominal tasks `pending_3` and `pending_4` and their final goals:
`fresh_generation_valid`, `accepted_chain_valid`, `nominal_chain_valid`,
`nominal_complete` and `outline_complete` are true; `unfinished_task_ids` and
`errors` are empty. Source, parent capture, frozen implementation, native
inputs, provider records and manifest membership verified unchanged. The audit
checked saved RA and physical evidence and replayed CCA; it did not rerun RA
derivation or native previews. The original attempt report remains unchanged;
the completed assessment is recorded separately in
`experiments/20261010T213423Z/audited_result.json` and `independent_audit.json`.

Production Load saved result also passed against this real result, restoring
all five events and the complete bound nominal proof into a separate Manual
review run. Primitive composition remains waiting and execution is disabled.
All 40 protected source files and 62 copied provider/validation records retained
their hashes; the reload sent no provider, native or live call. See
`saved_result_restore_verification.json` in the same experiment directory.

After the audit, the progress display was corrected to show `generating_outline`
while automatic offline physical calculations are active instead of retaining
an intermediate `blocked` label. This changes no validation rule or frozen
Attempt 17 record. The original observation file remains unchanged.

Automated preparation verification passed 414 backend/offline cases, 347
workflow/offline cases, 543 outline/request fixture cases and 90 runner/audit
guards before Attempt 17; these suites overlap and are not summed. The earlier
broader run's seven fixture failures were corrected and both affected suites
passed; its existing NiceGUI warning is retained separately. Native verification
passed 46 tests plus 42 subtests, and `make bootstrap-gazebo` completed 16
packages. Previously recorded unrelated feasibility failures are not claimed
fixed by this work.

After the progress-label change, both focused workflow/offline suites passed
again: 347 tests in 35.28 s. `poetry check`,
`poetry run python -m compileall -q cais_spade_llm ros2` and `git diff --check`
passed. Poetry reported its existing metadata deprecation warnings.

This is a complete offline outline and model-based nominal-continuation result.
User-observed UI/Gazebo acceptance, separate primitive composition, execution
authorization, recovery execution and observed assembly completion remain
separate subsequent milestones. No new recovery execution was performed.

### 2026-10-12: four cases and reserved evaluation (in progress)

The same public prompt and one-event response path now supports `Conveyor
breakdown`, `ur5e-1 breakdown`, `Machining breakdown during part processing`,
and `Part slippage`. Context selection follows recorded relationships and
custody; all eligible resources remain discoverable. Public prompt and retrieval
tests exclude evaluator records, reference recoveries, route hints, private
primitive sequences, and operator metadata. Operator-guided runs remain outside
the unassisted evaluation. Large selected safety sets list every requirement
with exact identifiers and provide the full formula and AP bindings through
the existing public retrieval references.

Resource-owned calculations now prepare `KMR`, `M1`, `M2`, and `Conveyor`
capability evidence against the same starting snapshot. Unordered
`successor_primitive_union`, `validation_contract_version=2`, and the 4,096
assignment bound remain unchanged. Private physical projection, CCA validation,
and nominal continuation remain separate obligations. Missing continuous base
motion, clearance, or controller evidence remains unresolved rather than being
converted from a route or a station label into an approved motion.

The Recovery UI retains the four exact scenario names, scenario-matched saved
captures, Offline Manual, expandable evidence, and original downloads. New
cases stop at outline review. Source capture, configuration, selected
specifications, and implementation provenance are recorded separately from the
LLM proposal. A failure checkpoint no longer acts as operator Stop.

Automated verification completed so far: 465 owner/offline/declared-program/
validated-outline tests, 276 prompt/retrieval/request/evaluation tests, and a
later 213 KMR/retrieval tests passed. These overlapping suites are not summed.
Poetry checks, Python compilation, evaluation CLI help, and Gazebo bootstrap
passed. Poetry retains its metadata deprecation warnings. A broader earlier
safety run was interrupted to release simulation resources after 857 passes;
it is not recorded as a completed suite.

Development capture evidence is retained in
`monitor/environment_runs/recovery/evaluation_development/`. The fifth
Conveyor development run reached its real paused failure checkpoint and
prepared recovery context without an LLM or recovery-execution call. Earlier
runs exposed a missing system-Python dependency boundary, a disabled Nav2
launch, and lost lifecycle-transition replies. The repaired readiness path
requires an observed map and active managed nodes; the lifecycle proxy checks
observed state after a lost reply without repeating physical goals. The fifth
run then exposed an unset optional KMR parameter during offline model archival;
the original observations remain retained unchanged. The native companion was
subsequently archived offline after preserving unset ROS parameters as typed
records and omitting only their inapplicable YAML overrides. The companion's
native model fingerprint matches the captured model.

New-attempt preparation now preserves all twelve resources' declared fields,
explicit null values, and capability catalogs from that capture. A detached
replay reported `outline_ready` with zero API, controller, observation, or native
calls and an unchanged observation SHA256. The completed `ur5e-1` pickup at the
Conveyor failure checkpoint remains recorded as completed; it is not relabelled
as a failed task. The recorded `gear_small` assembly is complete, leaving eleven
unfinished tasks for `KET4_Square_4mm`.

Further verification: 385 recovery/context/capture tests passed, 312 shared
prompt/request/evaluation/KMR tests passed, and 13 UI/attempt-version tests
passed. These suites overlap. The installed native preview suite passed 46
tests and 42 subtests in an isolated synthetic planning scene without Gazebo or
motion execution. Poetry, full Python compilation, and UI CLI help passed.
New attempts use verified native motion durations for UR projection clocks;
historical ordinal records retain their original interpretation and status.

KMR base travel still requires independently supported controller-motion
evidence. Saved routes, docking targets, and map samples alone do not establish
the timed closed-loop Nav2 motion or carried-part clearance. The development
record `evaluation_development/KMR_base_motion_evidence_gap_20261012.json`
distinguishes missing capture data from the unimplemented complete base
projection. Such results remain `NEEDS_CONTEXT` and cannot be counted as LLM
planning failures or complete recovery outlines.

The 24-attempt evaluation has not started. It requires eight separately
reserved Gazebo captures, each produced by the same unchanged implementation,
with actual normal/changed-condition facts and verified native companions.
Development captures cannot be promoted into this batch. The runner uses
`gpt-5.6-sol`, `low`, 80 turns, and no attempt deadline; all 24 results must be
retained, including unresolved evidence and API errors. Complete success
requires provider authorship, RA, physical evidence, CCA, and validated nominal
continuation. No recovery execution or user-observed UI acceptance is claimed.

The first reserved capture batch, `evaluation_captures/20261012_frozen_01`,
was stopped before any model attempt. Its Conveyor capture exposed an old
Part slippage observation contract requiring Nav2 to be disabled. The new
capture records KMR's source-bound command gate and sole output publisher,
retaining direct Nav2 activity as observed or unknown. It does not equate a
closed actuator gate with all Nav2 goals being idle. KMR base hold additionally
binds the physical Gazebo description to the same links and joints in the
planning description; the MoveIt description alone omits the required Gazebo
declarations. Offline hold checks retain both evidence obligations.

The changed Part slippage condition now binds its reach bound to KMR's
captured map extent, controller, complete planning base chain, and physical
base evidence. Planning proxy joint limits alone cannot establish mobile-base
reach. The bound assumes the captured map and model configuration remain
unchanged and grants no trajectory approval. The capture harness enables
Nav2 for all four capture cases; existing UI launch defaults are unchanged.

Further automated verification: 977 combined capture/safety/outline/Gazebo
Python tests passed, followed by eight real-validator map/base integration
tests and 33 capture tests. These suites overlap. The evaluation runner also
passed 30 tests covering complete eight-condition manifest verification,
manifest changes before API calls, and SIGTERM cancellation with owned-child
cleanup. Poetry and Python compilation passed. A new reserved batch is still
required after the final capture checks; the stopped batch remains retained.

The second reserved batch, `evaluation_captures/20261012_frozen_02`, also
stopped before any model attempt. Its Conveyor capture exposed a different
observation gap: the already assembled `gear_small` was attached by a native
fixed joint to static `assembly_board_v1::link`, but the old attachment record
did not establish the live joint or its endpoint transforms. Zero velocity and
missing contact samples cannot supply that evidence. `GETRECOVERYSTATE` now
records the observed joint, endpoints, relative transform, and attachment
revision. A detached calculation checks those observations before recording
conditional stationary geometry. This does not grant execution permission or
infer any future placement outcome.

The corresponding combined regression run passed 796 tests and exposed three
stale error-message expectations from this milestone's preservation of original
exception types. Those expectations were corrected; all three focused retry
tests passed. They are recorded as corrected expectations, not unrelated
failures. The native observation plugin rebuilt successfully. A new development
Conveyor capture is checking the complete observation path before another
reserved batch. No reserved model attempts or recovery execution have started.

### October 12: M1 breakdown machine job control

New simulation machine programs use `select_program`, `start_job`, `pause_job`,
`resume_job`, and `stop_job`; `machine_part` remains the machining task. Retained
job identities, active processing time, paused time, configuration, and completion
history survive pauses and unloading. Paused jobs can resume only with the same
part, machine, program, tool and workholding. Stopped jobs cannot resume. Program
selection cannot change tools or workholding. This is simulated active-time
accounting, not measured material removal or physical machining resumability.
Historical `dwell` programs retain their replay path.

The new machine-breakdown order includes `RGOCG4-50_Round_4mm` alongside the
existing square and gear. M2/ur5e-2 are enabled for nominal preparation. The
`during_processing` checkpoint waits for observed overlapping jobs and selects
a seeded nonterminal time; `during_processing_halfway` remains available.
`processing_observations` records failure-time job facts before cancellation;
shutdown records are separate. No recovery events or preferred sequence were
added to prompts. Both exact machine safety specifications remain selected.

RA uses the same pure owner calculations offline and live, with unordered
`successor_primitive_union`, contract version 2 and the 4,096-assignment bound.
Job controls have separate physical projections. Missing evidence remains
`NEEDS_CONTEXT`; an active peer job cannot silently freeze during another
positive-duration event. Current access/workholding evidence is required after
projected custody changes. Primitive composition and recovery execution remain
outside this milestone.

Automated results: 225 machine/setup/legacy tests passed; 227 overlapping
machine/safety tests passed; final focused runs passed 24 machine/Gazebo checks,
28 outline checks and 19 retrieval checks. Poetry, Python compilation, UI help,
and the 16-package Gazebo bootstrap passed. One unrelated robot-control test
stalled and was excluded from the 225-test run; two broader runs timed out and
are recorded separately, not counted as passed.

Live acceptance remains blocked. One isolated Gazebo capture failed before
machining at the ROS readiness query timeout. It produced no new observations,
made no LLM calls, left the preserved files unchanged, and stopped all owned
launch groups. No offline GPT-5.6 attempt was started, because the existing
empty-M2 capture does not represent the requested case. No recovery execution,
validated nominal continuation, or user-observed acceptance is claimed.

Evidence: `cais_spade_llm/monitor/environment_runs/machine_job_control_capture_20261012/IMPLEMENTATION_RESULT.md`
and its `implementation_verification.json`, capture report and retained logs.
Next: resolve the observed readiness blocker, capture overlapping M1/M2 jobs,
then perform the single authorized `gpt-5.6-sol`/`low` offline attempt with the
80-turn limit and no ten-minute generation deadline. Do not alter requirements,
insert recovery events, or repeat trials to obtain a favorable outcome.

### October 12 follow-up: readiness diagnostics and observed M1 failure

Readiness commands now retain startup phase, launch identity, ROS domain,
elapsed time, timeout, exit status, stdout and stderr. The actual failure reason
reaches SystemBridge. Owned probe processes are terminated and reaped on timeout
or cancellation. Required service checks, the 10-second probe timeout, serialized
discovery, and stale-launch rejection remain in place.

Automated verification: 112 startup/capture/machine-job tests passed, followed by
33 overlapping capture tests after adding the diagnostic-path assertion. Poetry,
Python compilation, UI/capture CLI help and diff checks passed. No ROS2 source
changed in this repair. The original readiness failure did not recur and its
original underlying cause remains unconfirmed.

The first instrumented capture passed readiness but exhausted its 600-second
capture allowance during circular-part transfer to M2. A retained retry used the
identical setup and failure seed with a 2400-second capture allowance. It observed
the requested M1 `during_processing` failure while both machine jobs were running:
M1 square processing was at 24% and M2 circular processing at 12% of their configured
active simulation durations; neither process was complete. The original capture is
`recovery/4a81d7fc7ccd46d1a18b79740d16add7/observations.json`. All owned Gazebo and
agent processes stopped, and existing preserved-file hashes remained unchanged.

Offline acceptance remains blocked by evidence reconciliation. A later physical
export reported `KMR.current_pose` conflicting with owner state. The original saved
capture's local offline preparation found `M1.active_job=environment_21` and
`M2.active_job=environment_23` in prepared snapshots/environment valuations while
both owner snapshots recorded explicit `null`; `job_state` also disagrees. The
check made zero forbidden/live/LLM calls and left the capture hash unchanged.
No values were overwritten and no validator was bypassed.

API disclosure approval for one machine-breakdown GPT-5.6-sol/low attempt is now
recorded, but generation was not started because preparation is blocked. No
recovery proposals, accepted recovery events, RA/CCA recovery verdicts, validated
nominal continuation, recovery execution, or user-observed UI acceptance are
claimed. The next repair is consistent capture revisions and owner snapshots for
machine job records, plus retention/reconciliation of the exact KMR pose evidence.

Evidence: `cais_spade_llm/monitor/environment_runs/machine_startup_repair_20261012_02/IMPLEMENTATION_RESULT.md`,
`implementation_verification.json`, `local_preparation.json`, and
`recorded_state_comparison.json`; both capture attempts retain their diagnostics.

### October 12 follow-up: machine/KMR snapshot repair and one offline attempt

Capture now preserves validated machine-worker job fields through physical
snapshot refresh. KMR preparation uses frozen measured observations and retains
earlier primitive endpoints as history, without rereading owner state during
preparation. Conflicts, changed revisions, and frame mismatches remain explicit.
Raw observations and preparation errors persist even when export fails. Readiness
checks saved-state consistency before generation.

The original `4a81d7fc7ccd46d1a18b79740d16add7/observations.json` remains unchanged
(SHA-256 `a6070287e3a90ae30ea5493b45118c88b3460667a4a907c9ffff0d4dad4682be`).
A separately derived capture records corrections and provenance. Local replay
reached `outline_ready` with all LLM and live interfaces forbidden. Failure-time
processing and later physical observations retain their different timestamps;
their compatibility is not assumed for physical validation.

Automated checks: 489 capture/offline/machine/resource tests and 511
validation/retrieval/UI tests passed, with overlapping counts. Poetry, Python
compilation, UI/capture CLI help, and diff checks passed. Sandboxed asyncio thread
cleanup hung in a minimal reproduction; the checks completed outside that sandbox.
An obsolete machine-scenario test was updated to expect M2 and ur5e-2 enabled.
A final 307-test subset passed after ensuring missing machine-worker records
remain unknown in the detached loader. The attempt's recorded M1/M2 job fields
were rechecked unchanged; no additional API attempt was made.

One unchanged-settings live recapture passed readiness but stopped before the M1
failure checkpoint: `/KMR/move_base` aborted because arm/payload clearance timed
out after 0/1 states. Owned Gazebo and agents stopped, with preserved-file hashes
unchanged. This additional clearance-service blocker was not expanded into new
KMR motion work.

Exactly one approved GPT-5.6-sol / low offline attempt then used the repaired
original capture, both selected machine safety specifications, shared factual
prompts, and dynamic retrieval. It made 9 provider requests/responses, 4 retrieval
turns, and 4 proposals for the first event: M2 `pause_job`, M1 `interrupt_job`, M2
`pause_job`, and KMR `move_to`. Accepted events: 0. The pause proposals retained
an active job contrary to the recorded pause successor; M1 was unavailable; KMR
`move_base` lacked an accepted parameter witness. Physical preview preparation
then stopped at missing `ur5e-4` attachment-link evidence. CCA was not reached and
nominal continuation was not established. Outcome: `NEEDS_CONTEXT` with the
original stage exception retained; no evidence was bypassed.

The attempt took 136.89 seconds including preparation and used 101,964 tokens
(99,086 prompt, 2,878 completion), with no provider errors or forbidden live calls.
Implementation and source hashes remained unchanged. No recovery execution or
primitive composition occurred. User-observed UI acceptance remains pending.

Evidence: `cais_spade_llm/monitor/environment_runs/machine_observation_repair_20261012/IMPLEMENTATION_RESULT.md`
and `RESULT.json`, `machine_observation_repair_20261012_capture/report.json`, and
`machine_observation_repair_20261012_offline_01/report.json`. Remaining work is
separate: obtain compatible complete physical observations, diagnose the live
KMR clearance-service timeout, and repair the demonstrated KMR parameter-contract
gaps before claiming a fully validated machine-breakdown outline.

### October 12 follow-up: local offline validation repair without recovery hints

Completed the optional/nested primitive schema correction, KMR empty-custody
parameter preparation, precise public successor-conflict feedback, and structured
missing-scene evidence handling. New catalogs and descriptors are rebuilt; original
captures and historical results remain unchanged. The prompt implementation is
byte-identical to its pre-repair version. No API calls or Gazebo launch occurred.

Local replay of the four saved proposals now gives RA `FEASIBLE` for KMR `move_to`.
Both M2 `pause_job` proposals still reject the conflicting `active_job` value with
specific feedback, and unavailable M1 still rejects `interrupt_job`. A supported
paused-state regression fixture passed separately; it was not provider-authored.

Physical evidence remains unresolved: the recorded
`KET4_Square_4mm/link/collision` identity is absent from the saved MoveIt object list,
which instead contains `KET4_Square_4mm`. Its scene acknowledgement is false, so no
identifier substitution or clearance evidence is invented. The existing stale
`ur5e-4` observation and processing/physical timing distinction remain recorded.
Unresolved preparation now stops with source references, without an unproductive
validation retry. No outline append, CCA approval, or nominal advancement occurred.

Automated checks: 1,071 focused tests and 438 outline physical-evidence tests
passed; the broader RA suite had 198 passes and 12 failures, all reproduced with
the pre-repair sources (legacy AP descriptors and KMR prepared-execution-provider
tests). Poetry, compilation, and UI command-line help passed.
Source capture, safety-file, and prompt hashes were preserved. User-observed UI
acceptance, a fresh capture, and another model attempt remain separate work. The
live KMR clearance-service timeout was not changed.

Evidence: `cais_spade_llm/monitor/environment_runs/offline_validation_repair_20261012/IMPLEMENTATION_RESULT.md`,
`local_replay_after.json`, and `verification.json`.

### October 12 follow-up: one fresh offline GPT-5.6 retest

The requested `gpt-5.6-sol` / `low` attempt used the unchanged derived machine
capture, both selected machine safety specifications, shared prompts, and dynamic
retrieval. All live equipment and later-stage interfaces remained blocked. There
were 17 successful provider calls: one grounding turn, three retrieval turns, and
13 proposals for the first recovery event. Accepted events: zero. Physical
projection, CCA, and nominal continuation were not reached in this attempt.

The test exposed a separate outline response-schema defect. M2's `active_job`
declaration permits `["string", "null"]`, but its incomplete observed domain
contains only `"environment_23"`. `_outline_state_field_schema` uses that domain
to generate `["string"]`, excluding the `null` value required by the declared
paused state. Local replay verified that this value passes RA while the provider
schema rejects it. The local fixture was not sent to the model or accepted as an
LLM event. No implementation or prompt changes were made during this retest.

The owned attempt was stopped after confirming the contradictory constraints,
with all 17 started provider calls completed. Duration: 230.54 seconds; token
usage: 187,128 prompt and 7,020 completion, totaling 194,148. Sources and
implementation fingerprints remained unchanged. The raw cancellation snapshot
retains `generating_outline`; the separate test result records the stopped
process and terminal outcome. No second attempt or recovery execution occurred.

Next repair: respect declared nullable types when constructing public outline
response schemas. Saved physical-scene evidence remains a separate unresolved
requirement; this schema repair alone cannot establish CCA acceptance.

Evidence: `cais_spade_llm/monitor/environment_runs/offline_validation_retest_20261012220045Z/TEST_RESULT.md`,
`TEST_RESULT.json`, and `schema_reproduction.json`.

### October 12 follow-up: declared outline response types repaired locally

The existing outline field-schema helper now preserves an explicit declared
type before considering observed domain values. M2's nullable active_job remains
nullable in both resource-only and part-bound response schemas. Legacy fallback
is unchanged; malformed explicit types use the existing preparation-error path.
No RA, physical, or CCA validation logic or public interface was added.

Local replay regenerated the actual response schema from the saved machine
session. The original pause_job proposal remains rejected by RA. A separately
labelled supported paused-state fixture passes the corrected schema and RA; it
was not sent to a provider, included in prompt content, or appended as an LLM
event. Unsupported M1 operations remain rejected. Saved KMR move_to support
remains FEASIBLE, with its physical evidence unresolved.

All 900 focused context, request-record, validated-outline, workflow, and offline
tests passed. Poetry checks and Python compilation passed. Prompt instructions,
source observations, and historical request/response records were preserved.
The changed source is covered by the existing implementation fingerprint.
The earlier 12 unrelated broad RA-suite failures remain separately documented
and were not rerun in this scope.

No API calls, Gazebo launch, recovery execution, or live UI acceptance occurred.
The remaining saved physical-scene evidence gap is unchanged; complete CCA
acceptance and nominal continuation are not established.

Evidence: `cais_spade_llm/monitor/environment_runs/outline_schema_repair_20261012221658Z/RESULT.md`,
`verification.json`, `schema_replay.json`, and `response_schema_not_sent.json`.

### October 12 follow-up: real offline retest after the nullable-type repair

One fresh `gpt-5.6-sol` / `low` attempt used the unchanged derived machine
capture and both machine safety specifications. The exact provider requests
now permit `active_job=null`. After two public retrieval turns, the model first
proposed M2 `pause_job` retaining `active_job="environment_23"`; RA rejected that
conflict. Its next proposal independently changed that field to null and passed
RA as `FEASIBLE`. No proposal was rewritten or inserted by the runtime.

Physical projection and CCA remained unresolved because failure-time processing
records at simulation times 84.0/84.2 lack recorded compatibility with physical
measurements at 85.65/85.8665. Zero events were appended and nominal continuation
was not established. This is missing evidence, not a proved safety violation.

The subsequent offline evidence calculation also exposed a preview problem.
The standalone test initially lacked the installed ROS message-library environment
and failed importing `moveit_msgs`. Sourcing ROS fixed that import locally without
starting Gazebo. Replaying the unchanged proposal with its saved prepared inputs,
with API and live calls forbidden, then failed an SRDF named-pose lookup for the
M2 job-control proposal. That fallback needs the appropriate machine-owner
projection path. It cannot substitute robot motion evidence for job control.

The single API attempt completed five successful provider calls in 69.28 seconds:
one grounding, two retrievals, and two proposals for the first event. Usage was
42,537 prompt and 1,186 completion tokens (43,723 total). Zero provider errors
and zero forbidden live calls were recorded. Original/derived captures, safety
files, and frozen implementation hashes remained unchanged. No additional API
attempt, primitive composition, recovery execution, or user-observed UI acceptance
occurred. The prior 900-test schema-repair result remains recorded separately;
this retest changed no runtime source.

Evidence: `cais_spade_llm/monitor/environment_runs/offline_schema_retest_20261012T222911Z/TEST_RESULT.md`,
`TEST_RESULT.json`, `attempt_01/report.json`, and `local_prepared_evidence_replay.json`.

### October 12 follow-up: resource-owned outline evidence routing repaired

Offline evidence preparation now uses the existing resource-owned projection
for machine and other nonrobot proposals. Robot and KMR paths are selected from
the recorded resource type; unsupported owners return missing evidence. Machine
job control no longer requires a native robot archive or falls through to SRDF
named-pose lookup. Source-hash checks and unresolved checkpoint compatibility
remain enforced. The change adds no prompt instructions, primitives, safety
specifications, recovery events, or execution permission.

Exact local replay preserves the previous M2 proposals: Turn 4 remains RA
INFEASIBLE for its active_job conflict; Turn 5 remains RA FEASIBLE. Turn 5 now
returns the recorded processing/physical checkpoint-compatibility gap directly,
without a ROS import or robot named-pose exception. No event is appended, and
CCA approval and nominal continuation remain unresolved. Failure-time processing
and later physical observations cannot be combined without supporting evidence.

All 1,108 focused tests passed: 675 machine/environment/workflow/physical-evidence
tests and 433 offline/validated-outline/KMR tests. The tests include live and
detached M1/M2 owner calculations, Conveyor and Buffer For Machined parts,
unsupported owners, stale source hashes, retained peer geometry, and forbidden
robot preparation for nonrobot events. Poetry checks and Python compilation
passed. The initial sandboxed mocked stage-mode test stalled; the same test and
both complete suites passed outside the sandbox. That interrupted diagnostic run
is recorded separately. Earlier unrelated broad-suite failures were not rerun.

All 44 protected hashes remained unchanged, including captures, the prompt
builder, both safety files, and prior request/response records. No API call,
Gazebo launch, recovery execution, or user-observed UI acceptance occurred.

Evidence: `cais_spade_llm/monitor/environment_runs/owner_evidence_repair_20261012T224635Z/RESULT.md`,
`verification.json`, `saved_proposal_replay.json`, and `source_changes.patch`.

### October 12 follow-up: one offline API retest after owner-evidence routing repair

One fresh `gpt-5.6-sol` / `low` attempt used the unchanged derived machine
capture, both machine safety specifications, shared prompts, dynamic retrieval,
Offline Manual, 80 turns, and no overall generation deadline. The runner was
reused unchanged; no runtime, model, prompt, or specification edits were made.

All five provider calls succeeded. The model grounded the capture, made two
public retrieval turns, and proposed two versions of the first M2 pause_job
event. Turn 4 retained active_job="environment_23" with resource_state="loaded"
and job_state="paused"; RA rejected the full end state. This attempt's feedback
identified resource_state="completed" with the other fields fixed. Turn 5 made
that change while retaining the active job. RA derived start_job support from
machine_part, but the owner rejected its capability checks because a machining
job was already active. RA returned NEEDS_CONTEXT, not FEASIBLE. Its private
parameter_witnesses retrieval returned no saved evidence.

Subsequent physical evidence preparation returned the recorded checkpoint
compatibility gap directly, before robot/native preparation. The earlier ROS
import/SRDF exceptions did not recur. Successful machine projection and CCA were
not reached; zero events were appended and nominal continuation was not
established. This result differs from the earlier local replay that passed RA
with active_job=null. Two proposal revisions are not two accepted events.

The attempt took 65.43 seconds and used 43,696 prompt plus 1,264 completion tokens
(44,960 total). All 523 frozen files and 44 protected historical files passed
hash verification. Zero forbidden live or execution calls occurred. Exact
requests/responses, retrievals, RA, and physical-preparation records are saved.
The preceding 1,108-test result remains separately recorded; no new regression
suite or user-observed Gazebo/UI acceptance is claimed for this retest.

No additional API attempt, capture repair, Gazebo launch, primitive composition,
or recovery execution occurred. Further repairs remain separate work.

Evidence: `cais_spade_llm/monitor/environment_runs/offline_owner_evidence_retest_20261012T235206Z/TEST_RESULT.md`,
`TEST_RESULT.json`, `attempt_01/report.json`, and `attempt_01/session.json`.

### October 12 follow-up: repair derivation and shorten outline prompts (local replay)

**Implemented and locally verified.** Outline derivation now compares intended
conditions with the recorded or accepted projected start. Only declarations
that explicitly support a requested resource change contribute primitives;
preserving a requested value cannot add unrelated support. Owners pass their
existing snapshots into preparation. Contributors, bindings, preserved
conditions, and additional effects remain recorded as obligations. Unordered
`successor_primitive_union`, independent capability checks,
`validation_contract_version=2`, and separate physical/CCA validation remain.
The internal parameter-trial limit accepts a positive configured value; its
default remains 4096. No extra validation layer or movement validator was added.

Local replay of the saved M2 evidence verifies that the minimal
`job_state="paused"` fixture derives only `pause_job` and passes RA. The original
provider proposal with `active_job="environment_23"` still contradicts the
relevant declaration. Its feedback is independent of JSON key order and does
not suggest `resource_state="completed"`. Supported paused-state fixtures remain
explicitly separate from provider responses and prompt content. The nominal
checker still validates each exact recorded unfinished task; unchanged public
values alone do not establish task completion.

Initial outline message size fell from **20,202 to 11,127 characters (44.9%)**.
The provider `response_format`, including its unchanged schema, remains
**16,653 characters**. One resource index and retrieval section replace
duplicate listings. Initial processing facts and public field meanings are
compact; full declarations, job records, programs, and safety bindings remain
available through existing public references. Both exact machine-breakdown
requirements remain included. The saved four context requests were replayed
successfully without live calls; their prepared message is 17,079 characters.
All new previews are marked not sent. Full observations and audit files remain
available, and no action, resource preference, sequence, or event-count target
was added to the prompt.

Automated verification: **1,979 passed; 12 pre-existing failures**, reproduced
against saved pre-edit source. Those failures are five retired AP descriptor
cases and seven KMR execution-provider cases. `poetry check` and
`poetry run python -m compileall -q cais_spade_llm ros2` passed. The guarded local
replay made zero API/live-interface calls and verified all 53 protected hashes.
Original/derived observations and historical requests/results were preserved.

**Remaining evidence gap:** failure-time processing and later physical
observations still lack recorded checkpoint compatibility. RA success does not
establish a physical projection, CCA approval, or nominal continuation. No new
API attempt, Gazebo capture, primitive composition, or recovery execution was
performed. User-observed UI/Gazebo acceptance remains separate.

Evidence: `cais_spade_llm/monitor/environment_runs/derivation_prompt_repair_20261013T002221Z/RESULT.md`,
`local_replay.json`, `verification.json`, `initial_outline_after_not_sent.txt`,
`retrieval_replay.json`, `source_changes.patch`, and the baseline/final test logs.

### October 12 follow-up: one offline API retest after derivation and prompt repair

One fresh `gpt-5.6-sol` / `low` attempt used the unchanged derived machine
capture, both exact machine safety specifications, Offline Manual, 80 turns,
and no overall generation deadline. The runner was reused unchanged with an
independent session; no runtime implementation or global settings were changed.

All four provider calls succeeded. Turn 1 grounded the facts; Turn 2 retrieved
M2 and M1 `static_capabilities`. Turn 3 proposed M2 `pause_job` while retaining
active_job="environment_23". RA returned the precise active_job contradiction,
without the former unrelated completed-state correction. Turn 4 proposed
active_job=null with job_state="paused" and resource_state="loaded"; **RA passed
and derived only pause_job support**. Part conditions remain deferred obligations.

Physical projection returned the known failure-time processing/later physical
observation compatibility gap. CCA was invoked and returned unresolved with no
completed safety checks; this is not a demonstrated safety violation. Owner
evidence preparation reported the same gap without robot named-pose fallback.
Zero events were appended, the symbolic state and monitor history were
unchanged, and nominal continuation was not established. The two proposals are
revisions of one event, not two accepted recovery events.

The actual initial outline message was 11,127 characters; its response-format
schema remained 16,653 characters. The complete provider payloads before the
first proposal contained neither pause_job nor start_job. No operator guidance,
preferred sequence, or event-count target was added. Dynamic retrieval served
both requests from saved facts without errors.

The attempt took 55.34 seconds and used 24,355 prompt plus 1,020 completion
tokens (25,375 total). All 523 frozen files and 53 protected historical files
passed hash verification; zero forbidden live or execution calls occurred.
The preceding repair's regression results remain separately recorded and were
not rerun here. No additional API attempt, fresh capture, primitive composition,
recovery execution, or user-observed UI/Gazebo acceptance occurred.

Evidence: `cais_spade_llm/monitor/environment_runs/offline_derivation_retest_20261013T005942Z/TEST_RESULT.md`,
`TEST_RESULT.json`, `attempt_01/report.json`, and `attempt_01/session.json`.

### October 12 follow-up: frozen machine checkpoint implementation; live trial blocked

Implemented machine checkpoint collection before worker cancellation for
`during_processing`. New nominal dispatch stops first; the owned Gazebo world
is frozen without issuing `pause_job`. Read-only machine requests account for
processing through the observed checkpoint and preserve completion races.
GETRECOVERYSTATE can service paused-world observations under the physics lock.
Raw native observations, machine replies, physical evidence and the planning
scene are retained before preparation; shutdown remains separate.

`processing_capture_compatibility` now requires recorded checkpoint identity,
revision, source and fact bindings. Equal timestamps alone cannot certify a
capture. Preparation and the capture harness reuse the frozen checkpoint and
report incomplete evidence before generation. Stop retains the observation
connection until it can unfreeze the owned world. Historical records, both
machine safety specifications and recovery prompts remain unchanged.

Automated results: **1,813 passed; 34 skipped** in the focused suites;
**36 passed; 1 skipped** in the final cleanup checks; **384 passed** in the
final capture/offline checks after binding unfinished obligations. These suites
overlap. Poetry checks, Python compilation, UI/capture CLI help,
`make bootstrap-gazebo` and the native preview suite passed. Local fixtures
do not establish a complete accepted sequence for the live case. Existing
checks still require supported processing progress across event intervals;
unaccounted running jobs remain unresolved rather than silently frozen.

Observed by the isolated harness: the first startup failed before jobs ran
because its invocation lacked the sourced ROS environment. The corrected
invocation retained the same scenario setup, durations and seed `4000168541`.
It reached nominal work, but KMR `pick_approach` for `KET4_Square_4mm` failed
with `KMR fresh joint feedback timed out` at 21:34:18 EDT. Logs also retain
read-only ROS query retries and lost Nav2 map-server heartbeats. The underlying
timing/transport cause is not yet established. M1's failure and the new frozen
checkpoint were never reached. All owned simulation processes stopped.

**Milestone remains incomplete.** No consistent new capture was available,
so the one offline `gpt-5.6-sol` / `low` attempt was not started: zero API
calls/tokens and no new proposed or accepted events. RA, physical projection,
CCA and nominal continuation were not evaluated for a new generated outline.
The historical capture was not substituted and no evidence checks were
bypassed. KMR fresh-feedback diagnosis and a successful checkpoint remain the
prerequisites for the complete offline attempt. No primitive composition,
recovery execution or user-observed UI acceptance occurred. All 61 protected
file hashes and both safety specification hashes are unchanged.

Evidence: `cais_spade_llm/monitor/environment_runs/machine_checkpoint_repair_20261013T011337Z/RESULT.md`,
`verification.json`, `preservation.json`, `capture_sourced/report.json`,
`capture_sourced.log`, `capture_sourced/gazebo.log` and the test/build logs.

### October 12 follow-up: requested offline test from the previous capture

The user requested reuse of the recorded machine-breakdown observations instead
of another Gazebo trial. Offline preparation now permits proposal generation
when physical checkpoint compatibility is missing, displays that limitation,
and retains the compatibility result for physical/CCA validation. Conflicting
resource facts still block preparation; live capture checks remain unchanged.
This supersedes the preceding requirement for a new capture before any offline
generation, without changing the evidence required to accept an event.

One real `gpt-5.6-sol` / `low` attempt used the unchanged
`derived_4a81d7fc7ccd46d1a18b79740d16add7_20261012` capture, both selected machine
safety specifications, Offline Manual, 80 turns and no overall deadline. Shared
prompts and public retrieval were unchanged. Seven requests succeeded: one
grounding turn, four retrieval turns, and two revisions of M2 `pause_job`.
The first retained a conflicting `active_job="environment_23"`; the second
requested only `active_job=null` and `job_state="paused"` and passed RA.

Physical projection remained `NEEDS_CONTEXT` for the saved processing/physical
checkpoint compatibility gap. CCA returned `unresolved`; no event was appended,
projected state and monitor history were unchanged, and nominal continuation
was not established. The attempt stopped on that evidence finding, not an
event-count limit. Elapsed time was 70.88 s; usage was 48,531 prompt and 1,464
completion tokens (49,995 total). No API errors or forbidden live-interface
calls occurred. No Gazebo launch, new capture, primitive composition, or
recovery execution occurred.

Automated checks: **958 passed**, Poetry checks passed with existing metadata
warnings, and Python compilation passed. Original and derived captures, safety
files and prompt sources passed hash verification; implementation/model/safety
fingerprints stayed fixed during generation. This is an automated offline API
result; user-observed UI acceptance and complete outline acceptance remain
separate and pending.

Evidence: `cais_spade_llm/monitor/environment_runs/offline_saved_machine_retest_20261013T020100Z/RESULT.md`,
`attempt_01/report.json`, `attempt_01/session.json`, `preservation.json`, and
`verification.json`. Exact turns are saved under
`recovery/offline_77022cb8e9dd48b7bfa6bd208614baa0/generation_0001/`.

### October 12: required evidence checks and instantaneous machine controls

Local replay of the unchanged saved M2 `pause_job` now reaches its owner
calculation. Preparation recovers the omitted `ur5e-4` collision bounds from
the saved native `dual_robot` record, without inventing a tool transform or
motion coverage. Machine fields remain sourced from job records. Capture-wide
compatibility no longer vetoes unrelated proposals: checks use the event,
selected AP dependencies and projected duration. Resource discovery retains
declarations with missing geometry; CCA still requires relevant participant
evidence. Private retrieval preserves projected facts and terminates when no
new evidence is returned.

CCA owner ledgers now represent declared instantaneous simulation job controls
with separate, fingerprinted before/after checkpoints. Complete local fixtures
pass both unchanged machine safety specifications; missing peer geometry still
returns `NEEDS_CONTEXT`. No positive motion duration is invented.

**Saved-capture result:** RA is `FEASIBLE`, but physical/CCA validation remains
unresolved and zero events are accepted. M2's job record is at 84.0 simulation
seconds; physical measurements are at 85.65–85.8665. The owner model bounds
processing to at most 2.4665 of 5.0 seconds, excluding completion, but cannot
establish `active_job` and `job_state` alongside that physical checkpoint.
The later owner export copies the earlier record. Saved measurements and
shutdown records were not rewritten or presented as simultaneous. Compatible
job/control history or a synchronized capture is still required; nominal
continuation is not established. No new API request was made against this
known evidence gap, and no Gazebo or recovery execution was started.

Automated results: **1,620 focused tests passed**, followed by **102 final
targeted checks**; Poetry checks and Python compilation passed. Protected
capture/specification/prompt hashes remained unchanged. The complete saved
response replay retained zero accepted events and unchanged state/history.
User-observed acceptance and full-outline acceptance remain pending.

Evidence: `cais_spade_llm/monitor/environment_runs/offline_nominal_repair_20261013T022658Z/RESULT.md`,
`local_replay/transaction_turn.json`, `verification.json`, `preservation.json`
and `implementation_changes.patch`.

### October 13: synchronized capture and successive offline repair attempts

The authorized isolated harness reached M1's fault while M2 was still
processing. Capture 03 retained native and machine observations at frozen
simulation time 84.8675, instance `11407416760602`. Export failed after a pure
local calculation exceeded the wall-age check; raw checkpoint evidence was
already saved. A separate derived capture restored omitted ur5e-2 fields from
those frozen measurements and matching recorded models. Original observations
and shutdown records remain separate. KMR was moving at the checkpoint;
freezing simulation does not establish a stationary controller. All owned
Gazebo processes stopped.

Repaired nominal expanded-task lookup, saved public facts and exact resource
identities, declared robot approach preparation, optional callable defaults,
recorded detection provenance, frame/TCP evidence recovery, persistent factual
rejection feedback, and public M2 mutex binding retrieval. The shared recovery
instructions contain no suggested sequence, preferred resource or event-count
target. RA retains unordered `successor_primitive_union`; physical projection
and CCA remain separate. Both machine safety files are unchanged.

Seven independent `gpt-5.6-sol` / `low` development attempts used Standard
service (`default`), Offline Manual, 80 turns and no overall generation
deadline. They used successive repaired implementations, so these are not a
frozen comparative evaluation. All seven accepted one M2 job-control event;
**none established nominal continuation**. Total reported API usage was
957,517 tokens. Exact proposals, retrievals, validation records, timings and
per-attempt usage are retained. Source and implementation fingerprints stayed
fixed within every attempt; no forbidden live or execution calls occurred.

In attempt 07, the exact turn-5 M2 `pause_job` proposal passed RA, physical
projection and CCA. Its next movement proposals did not pass all stages. The
last KMR proposal passed RA but scene preparation could not bind the recorded
`KET4_Square_4mm/link/collision` identity to the captured model-name collision
object. Local replay reproduced this result with API and live interfaces
forbidden and the source hash unchanged. Separately, the existing KMR owner
projection explicitly lacks continuous `move_base` evidence; repairing an
object binding alone cannot fill that gap. Recorded robot pickup targets also
exceeded their declared `gripper_reach` limits. Those limits and measurements
were not changed to obtain acceptance. No new KMR movement validator was added.

Automated checks (overlapping suites): 1,323 recovery tests passed; 921
capture/offline/outline tests passed after physical-field restoration; 648
RA/projection tests passed with 14 deselected; 278 context-retrieval tests
passed after mutex binding repair. Installed native preview checks: 46 passed.
Gazebo bootstrap built 16 packages. Poetry checks, Python compilation, UI CLI
help and the final scoped whitespace check passed. A broader run recorded
904 passed and 12 failures in legacy AP fixtures and KMR prepared-execution
mocks. These are reported separately; no restored baseline comparison was
performed. The older independent outline-audit runner also requires metadata
not emitted by this runner, so no audit pass is claimed.

**Milestone remains incomplete.** Remaining scene-binding and movement-evidence
gaps must be resolved before another unchanged API attempt is useful.
Automated capture and generation results do not establish user-observed UI
acceptance, a complete CCA-approved outline or assembly completion. Primitive
composition and recovery execution were not started.

Evidence: `cais_spade_llm/monitor/environment_runs/machine_synchronized_capture_20261013T040945Z/RESULT.md`,
`RESULT.json`, `preservation_final.json`, `attempt_07/report.json`,
`local_replay_attempt_07/report.json`, and `verification/`.

### October 13: KMR saved-evidence preparation repairs

Local replay of the unchanged attempt-07 provider proposals now passes the
released-part scene binding and missing coordinate preparation. Captured KMR
`release_part` acknowledgements bind the installed model-name collision IDs to
their exact SDF/native declarations, retaining both identities. The requested
`move_to_location` destination supplies its declared docking coordinates without
selecting a destination, route, or recovery event. Captured empty/carrying program
variants are respected; new physical preview records use version 2.

For future captures, an unsuccessful KMR base-hold check retains the raw
controller observation and physical fields already collected, together with the
original error. Moving or incomplete controller evidence remains unresolved.
Historical observations and both machine-breakdown specifications are unchanged.

Automated verification: **432 focused tests passed**, Poetry check passed with
existing metadata deprecation warnings, and Python compilation passed. Local
replay made zero API or forbidden live-interface calls. M2 `pause_job` still
passes RA, physical projection and CCA. KMR movement passes RA but reaches the
explicit missing continuous `move_base` projection. The saved KMR was moving;
stationary geometry cannot be assumed. **Nominal continuation remains incomplete.**

No Gazebo launch, primitive composition or recovery execution occurred. No new
API attempt was spent on this remaining deterministic implementation gap.
User-observed UI acceptance remains separate.

Evidence: `cais_spade_llm/monitor/environment_runs/kmr_offline_goal_20261013T120213Z/RESULT.md`,
`RESULT.json`, `local_replay_parameters/report.json`, and `tests_owner_evidence.log`.

## Conveyor breakdown specifications and capture — October 14, 2026 — capture in progress

The Conveyor Recovery example now selects exactly `SAFE_shared_area_mutex`
at `assembly_board-v1` and `SAFE_KET4_Square_4mm_before_gear_small`. The latter
requires `gear_small` entry after completed `assembly` of `KET4_Square_4mm`
at `GMC_Laser_Plate_Virtual/KET4_Square_4mm`; its weak-until formula retains
strict earlier completion. The opposite Part slippage ordering, both
Machining breakdown specifications, old broad documents, and existing
saved results remain unchanged. `SystemBridge` is unchanged.

The Conveyor example retains `after_M1_pick_before_release`, with
`additional_condition` requiring `ur5e-4` custody of `gear_small` from
`3D Printing Station`. Either pickup may arrive first. Both robots retain
their acknowledged parts while waiting; the fault leaves completed
`KET4_Square_4mm` held by `ur5e-1`, before Conveyor release.

Verification: 381 safety/checkpoint/capture tests and 165 Recovery UI
lifecycle tests passed. Poetry, compilation, both CLI help checks, and
scoped whitespace checks passed; Poetry retains existing metadata warnings.
All 18 protected configuration, safety, source capture, and bridge files
matched their pre-change hashes.

Capture-only run: `cais_spade_llm/monitor/environment_runs/conveyor_capture_20261014T170217Z/capture_01/`.
Its ROS domain is 89, Gazebo port is 11379, stages are Manual, and the
existing example records `diagnostic_cca_bypass=true`. `--scene-image`
saves only the owned Gazebo window. The user stopped the VS Code UI after
preflight refused the existing process; the isolated capture then started.
No LLM or recovery-execution call is permitted. An official Conveyor
failure capture is not yet retained; finish observation/native-model
verification and image saving before recording a retained identifier.

Conveyor capture_01 ended before nominal execution: Nav2 controller_server exited with SIGSEGV during configure; controller discovery failed and map activation did not complete. No failure checkpoint or observation capture was produced.
The failed stage has no retained failure capture and is disposable after
ownership and active-process checks. Diagnostic reports and logs are archived
under cais_spade_llm/monitor/environment_runs/conveyor_capture_20261014T170217Z/capture_01_diagnosis.
Next: a fresh launch on unused endpoints; retain the configured scene,
two-part order, both pickup conditions, and actual readiness requirements.

Conveyor `capture_01` disposable outputs were removed after the ownership
audit: 81 files, 894,328 regular-file bytes; archived diagnosis and original
failure captures remain retained.

Conveyor `capture_02` passed Nav2, scene, and joint-state readiness, then
failed while KMR loaded M1: `place_approach` exceeded the 120-second
wall-clock arm-action timeout. The observed Gazebo GUI real-time factor
was approximately 0.06. No failure checkpoint or observations were produced;
protected files matched and the owned ROS launch stopped. Its separately
owned embedded XMPP group was verified and stopped.

Diagnosis is retained under
`cais_spade_llm/monitor/environment_runs/conveyor_capture_20261014T170217Z/capture_02_diagnosis/`;
the explicit cleanup inventory is `capture_02_deletion_list.json` in its
parent. Next: retain headless nominal preparation and open the owned GUI
only at the stopped failure checkpoint. Six focused GUI tests cover
checkpoint gating, image provenance, ownership rejection, and cleanup.

The stopped nominal runtime record from Conveyor `capture_02` is
`113eb048c6694ed9b53f3c5956e91741`. It contains the same KMR timeout,
no failure, no observations, and no outline or primitive composition.
Its two records are archived losslessly under `capture_02_diagnosis/`;
`capture_02_runtime_deletion_list.json` records their ownership, times,
sizes, and hashes before removal. Conveyor `capture_03` remains active
and excluded from cleanup, as are all other chats' records.

Conveyor `capture_03` reached and verified the failure with both pickups,
unfinished assemblies, unavailable Conveyor, stationary unheld parts, and
`gear_small` outside `assembly_board-v1`. Its valid observations and verified
native companion are retained as `capture_2219b78844c34a2c9664aced794f2883`,
source run `53c715b9e11245b3be7f5260b312691c`; no LLM or recovery execution
calls occurred. Owned ROS and embedded XMPP processes stopped and protected
files matched. This successful failure data is excluded from cleanup.

The image failed visual acceptance: its owned Gazebo window was still black
five seconds after mapping. `capture_03/image_check.json` records this result;
the image is diagnostic, not the requested final scene. Next: repeat headless
capture, require nonblank rendered content stable for five seconds before
saving the image, then verify Offline UI loading. The harness now also directs
nominal reports to its private attempt directory and fingerprints the shared
latest reports, so the replacement cannot overwrite another chat's status.

## Conveyor breakdown handoff — October 14, 2026 — capture verified

Selected capture: `capture_1b3f3d1d729a4701b94cff514c783130` under
`cais_spade_llm/monitor/environment_runs/recovery/`; source run
`d49d4e9e71354325b11328a1cee4a5ce`. `observations.json` retains the two
acknowledged pickups, completed M1 before pickup, unfinished assemblies,
unavailable Conveyor transport, `gear_small` outside `assembly_board-v1`,
13 unfinished planner nodes, monitor history, complete physical evidence,
and planning-scene data. `native_preview_model/manifest.json` verifies the
bound native models. Recovery stages are Manual and
`diagnostic_cca_bypass=true`; LLM and recovery execution calls are zero.

`failure_scene.png` is the visually checked 1908×999 owned Gazebo window,
showing the red `Conveyor breakdown` marker, M1, the printer, and the
assembly area. `failure_scene.json` binds its SHA256, source observations,
run, window PID, physical fingerprint, and acquisition recovery. The
original parent exited on an X11 `BadMatch` while the window was mapping;
the same owned Gazebo instance remained alive. Read-only native observations
before and after image acquisition confirmed idle controllers, both fixed
gripper attachments, and all captured part positions within 0.002 m.
Those independent native records and the image recovery script are retained
under `conveyor_capture_20261014T170217Z/capture_04/`. The GUI group was
stopped separately after the process registry discarded its entry. All
recorded owned launch, GUI, and XMPP groups are stopped. The harness now
confines X11 reads to a child process and retries mapping failures.

The real Recovery page loaded this capture through Offline → Conveyor
breakdown → Saved observations, with exactly `SAFE_shared_area_mutex` and
`SAFE_KET4_Square_4mm_before_gear_small`. Browser reload passed, custody was
retained, all 12 resources were detached `SavedRecoveryResource` instances,
and generation/execution remained zero. Browser report and screenshot are
under `conveyor_capture_20261014T170217Z/ui_verification/capture_04/`.
The temporary verification session `offline_aa7126b2b0c942648a092c8595bf586e`
has no outline or primitive composition; its useful metadata is archived
there and its explicit cleanup inventory is `ui_review_deletion_list.json`.
The selected capture, its native companion, and all official earlier failure
captures are excluded from this cleanup.

Final focused suites: 414 passed, including concurrent `ur5e-1 breakdown`
coverage; the earlier 165 Recovery UI lifecycle tests also passed. All
protected files matched at capture completion. A later concurrent change
replaced the `ur5e-1 breakdown` broad document/default with
`safety_M1_mutex.txt`; this Conveyor work preserves that change. The other
17 protected source, configuration, safety, and bridge files still match.
There is no accepted Conveyor recovery outline or primitive composition.
Next stages remain recovery generation, CCA authorization, and execution.

Final cleanup removed 172 explicitly inventoried disposable files totaling
175,212,797 regular-file bytes, including the temporary UI review copy.
No official failure capture was removed. `retained_captures_after_cleanup.json`
confirms that both Conveyor captures, the original Machining breakdown capture,
and the original Part slippage capture remain selectable and offline-ready.
Image/source hashes match and no owned runtime process remains. Final Poetry,
compilation, capture CLI help, and scoped whitespace checks passed; Poetry
reported only the existing metadata deprecations. The task-level report is
`conveyor_capture_20261014T170217Z/implementation_verification.json`.
