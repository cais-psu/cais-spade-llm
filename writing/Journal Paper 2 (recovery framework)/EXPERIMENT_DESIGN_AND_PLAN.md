# Journal 2: experiment design, results, and figures

Date: 2026-10-13

Last updated: 2026-10-14

Status: Phase 1 documentation complete, revised on 2026-10-14 to make the
four-rung component ladder the first study. Phase 2 partially delivered: the
Experiments placeholder, navigation, and design download are available.
The independent outline auditor (phase 3) is the next implementation step.
No new experimental results are reported in this document.

## 1. Purpose and study boundary

Build an Experiments page for Journal Paper 2 around three questions, in
this order:

1. **Component ladder.** Starting from an LLM alone and adding one framework
   component at a time, what does each component change in the feasibility,
   safety, and completeness of recovery outlines, and what does it cost?
   (Study 1)
2. **End-to-end effectiveness.** Does the full framework produce complete
   validated recovery plans for each failure, and at what effort? (Study 2)
3. **Enforcement contribution.** On fixed inputs, which infeasible or unsafe
   proposals does each feasibility and safety gate prevent? (Study 3)

Study 1 comes first. Study 2 alone shows that the framework can complete a
recovery; it does not show what its components contribute. Other comparisons
are listed under Possible studies and are deferred. Implement one phase at a
time, recording verification and the next phase here after each delivery.

This is the dedicated experiment design and implementation plan requested by
the user. The broader [implementation roadmap](IMPLEMENTATION_PLAN.md) retains
runtime development and historical handoffs. The experiment choices here
replace its earlier four-method matrix for this work; earlier trial counts
and recorded evaluations are historical and must not be relabeled.

The study targets unforeseen disruptions outside predefined recovery
capability. `DES-only` and `No recovery` are excluded from the experiment
matrix. Their exclusion is a scope decision, not a measured zero-success result.

Use these existing scenario names exactly:

1. `Conveyor breakdown`
2. `ur5e-1 breakdown`
3. `Machining breakdown during part processing`
4. `Part slippage`

The initial implementation makes fresh LLM calls against saved observations
and performs offline validation. It does not inject new physical failures or
execute recovery. A validated outline, a validated primitive composition,
and observed recovery of production are distinct outcomes. Actual Gazebo or
hardware recovery measurements require a later execution study.

### Component ladder

Study 1 compares four rungs. Each rung adds one thing to the rung before it.

| Rung | Generation | Checks during generation |
| --- | --- | --- |
| 1. LLM, single-turn | Whole outline in one response | None |
| 2. LLM, multi-turn | One event per turn | None |
| 3. LLM + RA | Multi-turn | RA feasibility + physical evidence |
| 4. LLM + RA + CCA (full framework) | Multi-turn | RA + physical evidence + CCA safety |

**1. LLM, single-turn.** One request returns a complete recovery outline.
The LLM decides every event and when the outline is finished. Code only parses
the response and runs PA syntax and grounding checks. There is no retrieval
during generation, so the prompt supplies up front the same public facts that
the multi-turn rungs can request. This is the only rung in which the LLM alone
makes every decision, and it is the paper's pure-LLM baseline.

**2. LLM, multi-turn.** The LLM proposes one event per turn. Between turns,
code applies the proposed effects tentatively and shows the updated state in
the next prompt; public context retrieval is available. Generation stops when
the tentative state satisfies the nominal handoff conditions or the budget
ends. Nothing checks feasibility, physical support, or safety. This rung is
not LLM-only: code tracks state and decides when to stop.

**3. LLM + RA.** Rung 2 plus RA feasibility and RA physical evidence on each
proposed event. A rejected event is not admitted, the state does not advance,
and the findings are returned to the LLM.

**4. LLM + RA + CCA (full framework).** Rung 3 plus CCA safety for the
selected specifications, using the projected state and monitor history.
An event is admitted only when RA feasibility, physical evidence, and CCA
safety all pass.

#### Why RA comes before CCA

The order is set by the architecture. CCA checks of spatial predicates, such
as `SAFE_shared_area_mutex`, use RA physical evidence. Without it, CCA falls
back to `state_only_projection` in
[outline_checks.py](../../cais_spade_llm/recovery_framework/outline_checks.py)
and pauses with `NEEDS_CONTEXT` when the selected predicates need occupancy
evidence. A CCA-only rung would therefore not test safety for the selected
specifications. Physical evidence stays with RA because RA owns it.

#### Held constant across rungs

- The same frozen capture, selected safety specifications, model profile,
  request policy, and public facts. Rung 1 uses a single-turn prompt; rungs
  2–4 share the multi-turn prompts and schema versions.
- PA syntax and grounding checks in every rung. They are preparation, not a
  paper-level validation contribution.
- One proposed event per turn in rungs 2–4. No candidate selection takes
  place, so the symbolic selector is not part of Study 1. Record
  `recovery_selection_mode` anyway; with a single proposal it has no choice
  to make.
- The turn budget and stopping rule in rungs 2–4. Each rung stops on its own
  state: tentative in rung 2, RA-validated in rung 3, and RA- and
  CCA-validated in rung 4. The independent audit, not the stopping rule,
  decides whether an outline is complete.
- Rung 1 gets one generation request. A malformed or incomplete response is
  recorded as did not finish; only recorded transport retries are allowed.

#### Mapping to existing code identifiers

Rung names are labels for configurations. They do not rename code
identifiers. Preserve `pure_llm`, `neurosymbolic`, `outline_diagnostic_bypass`,
`incremental_validated`, `incremental_candidates_validated`, and
`action_horizon` values exactly.

| Rung | Generation setting | `check_ra_feasibility`, `check_physical_evidence`, `check_cca_safety` | Current availability |
| --- | --- | --- | --- |
| 1 | Complete outline in one response | all `false` | Not available. There is no standalone single-turn mode. `action_horizon="full"` prompt text asks for "a complete recovery outline trace" per candidate, but it is unverified on the `validation_contract_version` 2 path. Phase 4. |
| 2 | One proposal per turn | all `false` | Not available as specified. **Start fresh** with all checks off proposes up to five candidates and selects with the symbolic comparator. Phase 4. |
| 3 | One proposal per turn | `true`, `true`, `false` | Check selections exist. That the one-proposal path honors them needs verification. Phase 4. |
| 4 | One proposal per turn | all `true` | Ran as Attempt 17 with `pure_llm` and `incremental_validated` on the legacy validated route. Phase 4 moves it onto the shared pipeline. |

Distinctions found in the current code on 2026-10-14:

- The code identifier `pure_llm` names LLM candidate selection and retains
  RA/CCA validation. It is not rung 1 and does not mean "no validation".
- The Recovery page's **Start fresh** path
  ([outline_draft.py](../../cais_spade_llm/recovery_framework/outline_draft.py))
  fixes `outline_mode=incremental_candidates_validated`,
  `recovery_selection_mode=neurosymbolic`, `candidate_proposal_budget=5`, and
  `outline_diagnostic_bypass=true`. It applies the selected checks one at a
  time in
  [multi_turn_diagnostic.py](../../cais_spade_llm/agents/intelligent_product/replanner/llm_recovery/modes/multi_turn_diagnostic.py).
  With all checks unchecked, the symbolic comparator still chooses among
  candidates from their unvalidated tentative effects. That setting is not
  rung 2.
- Attempt 17, the only accepted outline, matches the rung 4 configuration on
  a different pipeline, with its production nominal-continuation proof. It
  remains a historical record and the auditor's positive control; it is not
  relabeled as a Study 1 trial.
- The selector's progression predicate still differs from the formal
  strict-expansion rule. Study 1 does not use the selector; the comparison
  belongs to possible study A.

Rungs 2–4 must run on one shared generation pipeline. Comparing the legacy
validated route with the Start fresh path would mix pipeline differences into
the rung effect.

## 2. Studies

### Study 1: component ladder (first study)

**Question:** Starting from an LLM alone, what does each added framework
component change in the feasibility, safety, and completeness of recovery
outlines, and what does it cost?

| Step | What it adds |
| --- | --- |
| Rung 1 → rung 2 | Multi-turn generation: turn structure, tentative state tracking, retrieval, and the symbolic stopping rule together |
| Rung 2 → rung 3 | RA feasibility and physical evidence, with rejection feedback |
| Rung 3 → rung 4 | CCA safety, with rejection feedback |
| Rung 1 → rung 4 | The whole framework |

The rung 1 → 2 step changes several things at once. Report it as single-turn
versus the multi-turn harness, not as the effect of turn count alone.

**Hypotheses, stated before collection.** These are measurements to collect,
not promised results.

- **H1:** Rungs 1 and 2 finish outlines that the audit marks infeasible,
  physically unsupported, unsafe, or without nominal continuation.
- **H2:** Rung 3 finishes outlines that pass RA but fail CCA safety, showing
  that feasibility checking does not imply safety.
- **H3:** Rung 4 finishes audited outlines, at a measured cost in turns,
  tokens, and time relative to rungs 2 and 3.

No directional hypothesis is set for rung 1 versus rung 2.

**Endpoint: the outline stage.** Primitive composition is excluded from
Study 1 because the retained composition still has
`cca_composition_status=not_checked`. Extending Study 1 to primitive
composition requires an equivalent composition audit.

#### What the ladder can and cannot show

- **Eliminations by construction.** Rung 3 cannot finish an outline that
  fails RA during generation, and rung 4 cannot finish one that fails RA or
  CCA. The audit uses the same checkers. The disappearance of the infeasible
  class at rung 3 and of the unsafe class at rung 4 therefore follows from
  construction; do not present it as evidence. The ladder's evidence is:
  - the violation rates of rungs 1 and 2;
  - unsafe outlines that pass RA at rung 3;
  - completion rates and cost of rungs 3 and 4;
  - how often the LLM proposes events that rungs 3 and 4 reject.
- **Practical baseline missing.** The ladder does not include an LLM whose
  complete outlines are validated after generation and regenerated with the
  findings under the same budget. That comparison justifies validating each
  event during generation. It is possible study B and a likely reviewer
  request.
- **Completion dependency.** Rung 4 has completed an audited outline only for
  `Part slippage` (Attempt 17). The retired `Machining breakdown during part
  processing` attempt admitted 28 events without nominal continuation, and
  `Conveyor breakdown` and `ur5e-1 breakdown` have no accepted outline. A
  failure enters the Study 1 matrix only after rung 4 completes there; that
  runtime work belongs to the broader roadmap.
- **One capture per failure.** Results are conditional on that starting
  state. Additional captures per failure would support broader claims.

#### Independent outline audit

Every finished outline from every rung goes through one audit that sends no
feedback to any model:

1. Start from the frozen capture.
2. For each outline event in order, run RA feasibility, RA physical evidence,
   and CCA safety for the selected specifications. Propagate state only
   through the validated path.
3. After the last event, check nominal continuation with the production
   proof, not the tentative handoff.
4. Stop at the first failing event. Mark later events as not audited after
   failure; they are neither passes nor violations.

Record every failing check of the first failing event, its index, and its
original findings. Classify the attempt by the first failing check in the
fixed order RA feasibility, physical evidence, CCA safety, nominal
continuation, so outcome classes do not overlap. Missing evidence is
`unresolved`, never a pass or a violation.

The audit reuses the
[outline check path](../../cais_spade_llm/recovery_framework/outline_checks.py),
but it is new code. The roadmap records that checking an existing outline
without generation has not been added, and Attempt 17's audit replays saved
evidence that rungs 1 and 2 never produce. Study 3's fixed-input policy
replay and possible study B use the same component.

#### Independent support for the audit

The paper treats the reviewed safety specifications and owner models as the
definition of correct recovery, which is the usual basis of formal-methods
evaluation. Strengthen this in two ways:

- Review every violation the audit reports for rungs 1–3, and a sample of
  their passes, against the specification text and observations. Record
  whether each is a real violation, a checker artifact, or missing evidence.
  Keep review notes outside model prompts and retrieval.
- Later, replay selected outlines that the audit marks unsafe in Gazebo and
  record the observed consequence. Do not substitute a rendering or a
  projected result for that observation.

#### Measurements

Definitions are in section 4.

- Started attempts, generator-reported completion, and the audited outcome
  class of each attempt.
- First failing event and the specification or owner check involved.
- Rungs 3 and 4: proposals rejected during generation, by check.
- Non-progress admissions for every rung, computed after the run.
- Turns, provider requests, retrieval turns, revisions, tokens, generation
  time, and audit time.

#### Pilot and staging

1. **Pilot**, after phases 3 and 4: `Part slippage`, all four rungs, three
   repetitions each (12 attempts). It is the only failure where rung 4 has
   completed. For scale, Attempt 17 took about 1,030 s, eight provider
   requests, and about 52k tokens with `gpt-5.6-sol` at reasoning effort `low`.
2. **Other failures**: add each failure as a frozen four-rung batch after
   rung 4 completes there. Do not pool batches that ran under different code
   manifests.
3. **Confirmatory batch**: ten repetitions per rung, frozen after the pilot.

**Primary figures:** audited outcome by rung, and effort by rung (see
section 5). Add one worked example: a rung 2 outline event that the audit
marks unsafe, with its CCA finding, beside the rung 4 outline for the same
capture.

### Study 2: end-to-end effectiveness of the full framework

**Question:** Can the full framework (rung 4) produce a complete validated
recovery from each of the four unforeseen disruptions, and what effort does
that require?

Use the rung 4 configuration. Record the actual selection mode and generation
strategy, and hold them fixed within each reported configuration. Start with
one model profile; additional models repeat this same study rather than
creating another study.

Run fresh attempts from matched saved captures through outline generation,
primitive generation, and the required final checks. Study 2 may continue
Study 1's rung 4 attempts through primitive composition; those remain the same
attempts. Record these endpoints from the same attempt:

- **Outline completion:** a complete validated outline with required
  nominal-continuation proof and independently checked evidence.
- **Primitive composition completion:** complete primitive programs passing the
  required resource, safety, and composition checks. Report the conditional
  completion count among attempts that reached an accepted outline.
- **Overall planning completion:** both stages and required final audits complete
  in the same attempt, divided by all started attempts.
- **Effort and failure location:** generation and validation time, revisions,
  provider calls, token usage, and the stage at which each attempt stopped.

The separate stage results reveal bottlenecks without requiring independent
outline and primitive strategy experiments. Do not multiply separately
collected stage rates to report an observed overall completion rate.

Use the same captures, selected requirements, budgets, and evidence policy
when comparing model profiles. Report results for each failure and condition,
with unavailable captures and incomplete support visible.

Initially this measures validated planning from saved observations. A claim
of recovered manufacturing operation also needs observed execution and resumed
production, for example in Gazebo. Add that measured endpoint to this study
when the execution path is available; do not infer it from an accepted plan.

**Primary figure:** completion by failure, with common-denominator panels for
outline and overall completion. Put conditional primitive completion and
latency/revision breakdowns in the accompanying table.

Use one recorded recovery timeline or ordered resource-event diagram to explain
the workflow. Choose and disclose the example-selection rule, and retain
representative failed or unresolved cases. Generated schedules must be labeled.
Use a seconds axis only when durations have recorded or supported evidence;
otherwise show ordinal steps. This is a reporting example within Study 2.

### Study 3: contribution of feasibility and safety enforcement

**Question:** Which infeasible or unsafe proposals are prevented from being
accepted by the RA and CCA checks, and how often are valid proposals rejected?

Use the same frozen proposals, primitive programs, starting states, and
reviewed reference labels under four acceptance policies:

1. Full validation.
2. Feasibility enforcement removed.
3. Safety enforcement removed.
4. Both enforcement gates removed.

Evaluate the policy decisions offline on identical inputs with the phase 3
auditor. Collect relevant records from Studies 1 and 2 and supplement them
with reviewed valid, infeasible, unsafe, interacting-composition, and
missing-evidence regression cases. Rung 1 and rung 2 events that the Study 1
audit flagged are naturally occurring infeasible and unsafe proposals. Keep
saved-run and synthetic origins visible. Study 3 reuses evidence rather than
requiring four new complete generation runs for every earlier attempt.

Apply all necessary checks for independent scoring, while recording which
results each acceptance policy uses. A removed gate does not become a recorded
`FEASIBLE` or safe result. Preserve PA binding/schema checks and trustworthy
state/effect derivation across policies. If required evidence cannot be
derived, record an unresolved decision rather than manufacture a successor.

Reference labels require reviewed justification independent of the output
being scored. Keep labels and evaluation notes outside model prompts and
retrieval. A check's output copied into its reference label is not an
independent correctness assessment.

Record infeasible acceptance, unsafe acceptance, rejection of valid proposals,
unresolved decisions, and checker time. Report separate feasibility and safety
labels because an input can violate both. Score candidate/event-level proposals
and complete programs separately; their counts are not additional recovery trials.

A unique reviewed case contributes one correctness observation per policy.
Repeated replay can measure timing variability but does not create new
independent reference cases. This study measures enforcement contribution on
fixed inputs. Its results do not establish how removing feedback would change
an LLM's subsequent generation; Study 1 covers that cumulatively.

**Primary figure:** grouped proportion plots for the four policies, showing
infeasible acceptance, unsafe acceptance, and valid-case rejection with counts
and uncertainty. Keep unresolved outcomes visible. Detailed confusion matrices
can accompany the figure or appear in supplementary material.

### How the studies relate

Study 1 adds components one at a time during generation on the same inputs.
Study 2 asks whether the full framework completes the whole planning pipeline
for each failure. Study 3 explains, on fixed inputs, which gate prevents
which error.

Study 1 changes what the LLM sees during generation, because each added gate
also returns feedback. Study 3 removes gates on fixed inputs and says nothing
about how generation would change. Studies can share inputs, attempts, and
evidence, but their endpoints differ: rung-level audited outcomes,
attempt-level completion, and case-level validation errors. Do not count
reused cases as new recovery attempts or present the studies as independent
replications of the same result.

### Possible studies

These are follow-up options. They are not prerequisites for the core studies
and do not need workers or figures in the first functional Experiments page.

| Possible study | Additional question and controlled comparison | Useful measurements and figure |
| --- | --- | --- |
| A. LLM and symbolic candidate selection | With rung 4 checks, compare `pure_llm` and `neurosymbolic` selection over the same candidate pool size. Requires the selector correction in the [formal selection reference](JOURNAL_VALIDATION_AND_SELECTOR_ACTIONS.md#current-mismatch-and-required-correction). A second analysis replays identical frozen candidate pools. | Valid selection, formal progress, non-progress admissions, completion within budget, and selection cost; paired dot plot or candidate decision graph. |
| B. Generate-and-test versus incremental validation | A single-turn complete outline is checked by the same validators after generation and regenerated with the findings until an equal token budget ends, compared with rung 4. Rung 1 outputs are its first drafts. Tests whether validating each event during generation beats validating complete drafts; a likely reviewer request. | Audited completion within budget, cost to first valid outline, and both by recovery length; time-to-valid-outline curves. |
| C. Primitive generation granularity | From identical frozen accepted outlines, compare event-by-event programs with complete-program generation before feedback. Preserve owner models, catalogs, and specifications. | Complete-program acceptance, composition outcomes, revisions, time, and tokens; paired comparison panels. |
| D. Separate RA feasibility and physical evidence rungs | Split rung 3 into RA feasibility only and RA feasibility with physical evidence. A CCA-only rung stays excluded because CCA spatial checks depend on physical evidence. | Audited outcome classes and cost per added check. |
| E. Broader model/provider benchmarking | Expand Studies 1 and 2 to more model families, reasoning settings, and hosted or local providers when adapters are available. | Completion versus latency/cost tradeoffs and model-by-failure heatmaps. Ordinary additional-model runs remain part of Studies 1 and 2. |

For possible study A, preserve actual candidate counts and malformed responses;
do not silently pad or replace candidates. Verify the selection implementation
against the formal contract before making a claim about that selector.

For B and C, report first-draft and budgeted outcomes separately and apply
equivalent final validation. A fixed-outline primitive trial cannot silently
regenerate its outline. Possible study D remains offline with execution blocked.

## 3. Shared collection protocol

### Captures, controls, and repetitions

- Select scenario-matched saved captures with the evidence needed by the study.
  Preserve recorded condition names such as `normal` and `changed`; a name is
  not a reference feasibility label. Missing captures appear as unavailable.
- Keep fresh-provider trials, response replay, synthetic fixtures, operator-assisted
  runs, and historical runs separately labeled. Do not pool them as fresh
  unassisted trials.
- Freeze the source capture and companions, code and uncommitted-change
  fingerprints, safety specifications, prompt/schema versions, owner evidence
  policy, and model/study settings before a batch. Changed inputs invalidate
  continuation of that batch; retain completed attempts under the old manifest.
- Use a fresh isolated session for each attempt. Prior trial transcripts and
  accepted prefixes do not become hidden input to the next trial.
- Randomize rung order within matched capture blocks using a recorded
  scheduling seed. Use a provider seed only where supported and record whether
  it was actually sent; a seed is not a guarantee of cross-model equivalence.
- Default to three pilot repetitions. A separate confirmatory configuration
  offers ten repetitions, frozen after pilot revisions. Pilots remain excluded
  from confirmatory summaries. Ten repetitions are a configurable starting
  point, not a claim of adequate statistical power for every comparison.
- Retain the existing session-turn budget of 80 and no overall deadline as
  initial configurable defaults for multi-turn rungs. Record proposal, context,
  tool, provider, and transport-retry counts separately. Freeze request
  timeouts, retry policy, and any additional token or time limits, including
  explicitly unset limits.
- Retried experimental attempts receive new identifiers. An API retry within
  one attempt is separately recorded and contributes to that attempt's
  measured cost and elapsed time.

### Retention of experiment records

Experiment attempt records are results, including attempts that fail or do
not finish. They are not disposable failed recovery attempts. Keep each
attempt's manifest, provider requests and responses, per-event check results,
audit result, timings, and stop reason. Large regenerable intermediates, such
as native preview outputs and caches, may be removed after the audit has read
them and recorded their hashes. Record the retained size of each attempt and
project the batch size before collection; the worked `Part slippage` archive
alone is about 663 MiB.

The [AGENTS.md recovery cleanup rule](../../AGENTS.md#recovery-result-retention)
directs deletion of disposable outputs from finished failed attempts. It needs
an explicit exception for experiment records before collection starts. Until
then, do not run experiment batches whose failed attempts that rule would
remove. Non-experiment recovery attempts keep the existing rule.

### Planned counts

For Study 1, planned attempts equal captures × rungs × model profiles ×
repetitions. The pilot is 1 × 4 × 1 × 3 = 12 attempts. One capture for each
of four failures gives 48 pilot or 160 confirmatory attempts per model profile.

For Study 2, planned attempts equal captures × model profiles × repetitions
for the fixed rung 4 configuration. One capture for each of four failures
gives 12 pilot or 40 confirmatory attempts per model profile; two conditions
per failure give 24 or 80. Study 2 attempts that continue Study 1 rung 4
attempts are not additional attempts.

Study 3 evaluates its unique reviewed cases under all four acceptance policies.
Report case and policy-decision counts separately from attempt counts.
These are planning counts, not collected results.

### Model profiles and future connections

Initially use the existing GPT connection with editable exact model IDs. Store
provider, requested model, returned model identity, supported request parameters,
response format, and usage separately from the experimental design.

Introduce a small model-client boundary around the existing request/response
capture path. A later hosted or local provider adapter must preserve the same
trial records, public input boundary, tool interactions, and validation path.
The provider adapter supplies its supported settings and usage fields. Do not
build additional providers in the document-only or initial UI phase.

Different reasoning settings are separate profiles. Do not silently map an
unsupported setting, fall back to a different model, or assume all model
families support the same structured-output or tool interface. Surface an
unsupported profile before collection where possible; retain provider errors
as real failed attempts when discovered after a request.

Credentials stay in the existing credential mechanism and out of experiment
manifests. Missing usage stays missing. Record provider-reported token counts;
different tokenizers mean those counts are not identical units across providers.
Report monetary cost only when the applicable dated pricing and usage are
available, with estimation and missing components clearly identified.

## 4. Required results and their interpretation

The required results are measurements to collect, not target percentages or
promised improvements. Report failures and inconclusive comparisons as well as
successes. Keep the rung-level and attempt-level outcomes of Studies 1 and 2
distinct from the reviewed-case policy outcomes of Study 3.

### Measurement definitions

| Measurement | Definition and reporting rule |
| --- | --- |
| Planned and started trials | Report both counts. A started trial has entered its isolated worker and opened its attempt record; zero-provider-call failures can still be started trials. Unavailable inputs and profiles remain visible as preflight exclusions. |
| Generator-reported completion | The rung's own stopping rule fired, or rung 1 returned a complete outline. Never reported as completion without the independent audit. |
| Audited outcome class | Study 1. One class per started attempt: passed, infeasible, physically unsupported, unsafe, no nominal continuation, unresolved, or did not finish (with its stop reason). Finished attempts are classified by the first failing check of the first failing event, in the fixed order RA feasibility, physical evidence, CCA safety, nominal continuation. |
| First failing event | Study 1. One-based index of the first event that fails the audit, with every failing check, the specification or owner identifiers, and the original findings. Later events are not audited after failure. |
| Rejected proposals during generation | Rungs 3 and 4: proposed events rejected by RA feasibility, physical evidence, or CCA safety during generation, counted per check and once in the proposal total. A context request is not a rejected proposal. |
| Non-progress admissions | Admitted events that satisfy none of the three formal progress cases in the [formal selection reference](JOURNAL_VALIDATION_AND_SELECTOR_ACTIONS.md#neurosymbolic-selection-method), computed after the run for every rung. Descriptive only; no rung in Study 1 selects by progress. |
| Audited outline completion | Complete accepted outline with independently checked per-transition evidence and the required nominal-continuation result. An LLM completion claim or `outline_review` alone is insufficient. |
| Primitive composition completion | Complete programs for the attempt's accepted outline with required resource feasibility, safety, and composition evidence. Report the conditional count among attempts that reached an accepted outline. Partial programs do not count as complete. |
| Overall planning completion | Outline, primitive composition, and required final audits complete in the same attempt, divided by all started attempts for that fixed workflow configuration. This is an offline planning endpoint until execution is observed. |
| Completion proportion | Audited completions divided by all started trials in the displayed input/treatment group. Break out infrastructure failures, cancellations, budget exhaustion, and unresolved results. Report recoverable and independently blocked reference conditions separately. |
| First-draft validity | Possible study B metric. For complete generation, score the first entire draft before revision; for incremental generation, score first proposals per event and sequence completion without rejections. These are different output units. |
| Valid selection | Possible study A metric: the selected candidate passes the common required validators for its frozen state. Record symbolic progress separately using the formal definition. |
| Invalid-case acceptance | Accepted results divided by all scored cases whose independently reviewed reference label is invalid for that checker. Include numerator and denominator and report unresolved checker decisions separately. |
| Valid-case rejection | Rejected results divided by all scored cases whose reference label is valid for that checker. Unknown reference cases remain a separate category. |
| Ablation unsafe/infeasible acceptance | For Study 3, policy-accepted cases with a reviewed unsafe/infeasible reference label. Show errors among all scored cases in that reference category, accepted-case counts, and unresolved outcomes. Safety and feasibility categories can overlap; never use attempt counts as this denominator. |
| Latency | Wall time from worker trial start to terminal status, plus recorded generation, retrieval, validation, and audit timings. Also report time to first independently valid complete output. |
| Generation effort | Proposal responses, candidate counts, retrievals, revisions, accepted events, provider requests, transport retries, and provider-reported usage. A context request is not a rejected proposal. |
| Evidence completeness | Required evidence present, missing fields, stale bindings, audit status, and links to original records. Absence of a rejection record does not establish acceptance. |

Keep original runtime statuses, including `NEEDS_CONTEXT`, and original
validation findings. A separate reporting classification may group them for
plots but must retain their original values and source references. Missing
measurements display as `not recorded`, never zero.

Record generation terminal time and audit duration separately. Final figures
show audited outcomes, with their timing definition in the caption. Do not
double-count overlapping stage timings in stacked bars; when measurements
overlap, plot them as separate durations instead of a total decomposition.

### Statistical reporting

- Report each model profile, rung, strategy, scenario, capture/condition,
  validation treatment, and pilot/confirmatory cohort separately before pooling.
- Include counts and 95% Wilson intervals for within-input binary proportions.
  Repeated calls on one saved capture measure variability conditional on that
  capture; they do not constitute new physical failure observations.
- For adjacent-rung contrasts on the same capture, report the difference in
  proportions with its interval. With three pilot repetitions, report counts
  and intervals without significance claims; the confirmatory batch carries
  the comparison.
- Use matched comparisons and capture-level resampling for aggregate differences
  when multiple independent captures are available. Record resampling settings
  and the analysis seed. With one capture per condition, describe uncertainty
  as conditional and avoid claims about unseen starting states.
- Show latency distributions or individual points with median and spread.
  Label successful-run latency with its own sample count and retain terminal
  times for failures and cancellations separately. Do not turn incomplete
  trials into zero-latency observations.
- Keep all started trials in the denominator of time-to-valid-output curves.
  Trials without a valid output contribute no completion; do not silently
  remove them or treat every failure as harmless statistical censoring.
- List exclusions with their reasons and counts. Freeze confirmatory settings
  before collection. Neither small pilots nor zero observed violations prove
  universal safety or recovery capability.

### Tables and exports required for the paper

1. **Protocol table:** model profiles, rungs and their exact settings,
   captures/conditions, fixed workflow and generation settings, budgets,
   repetitions, evidence versions, and exclusions.
2. **Study 1 results:** audited outcome classes `n/N` per rung and failure,
   first failing checks, rejected proposals during generation, non-progress
   admissions, turns, tokens, and time.
3. **Study 2 results:** outline and overall completion `n/N` per failure and model,
   conditional primitive completion, latency, revisions, tokens, and stop stages.
4. **Study 3 results:** unique reference cases, coverage, policy decisions, false
   acceptance/rejection, unresolved outcomes, and checker time.
5. **Raw exports:** one row per recovery attempt, plus separate linked tables for
   audited events, reviewed cases, proposals, programs, and checks. Cases and
   proposals must not become extra attempts in the summary.

Export JSON records and CSV tables with identifiers, source hashes, actual
settings, code/evidence-policy versions, outcome definitions, and artifact links.
Preserve the evidence required by accepted results and reference cases, and
keep experiment records under the retention rule in section 3.

## 5. Figures

Use measured data for paper figures. Any preliminary illustrative layout must
be labeled as an example and excluded from result exports. Start with the two
Study 1 figures; add the others with their studies.

| Figure | Study and purpose | Axes and evidence requirements |
| --- | --- | --- |
| Audited outcome by rung | Study 1: show what each added component changes | One stacked proportion bar per rung, faceted by failure. Classes are mutually exclusive by the first-failing-check rule: passed, infeasible, physically unsupported, unsafe, no nominal continuation, unresolved, did not finish. Label `n/N`; all started attempts are the denominator. The caption states which eliminations follow from construction at rungs 3 and 4. |
| Effort by rung | Study 1: show what each added component costs | Individual points with medians for turns, tokens, and time to audited completion, per rung. Show attempts that did not complete separately with their terminal times. |
| Recovery completion by failure | Study 2: show whether the full workflow completes | Horizontal: the four exact failures. Vertical: completion proportion, with `n/N` and 95% intervals. Show outline and overall completion in separate panels with all started attempts as denominator. Put conditional primitive completion in its own table/panel with its denominator stated. Distinguish model profiles when more than one is evaluated. |
| Validation-policy error plot | Study 3: show what feasibility and safety enforcement prevent | Horizontal: the four acceptance policies. Separate panels for infeasible acceptance, unsafe acceptance, and valid-case rejection; include counts, intervals, and unresolved outcomes. Preserve separate proposal/program units and avoid stacking overlapping error categories. |
| One resource timeline or ordered event diagram | Study 2: explain the composed recovery | Lanes: exact resource identifiers. Use supported schedule time or explicitly ordinal steps. Show outline events, primitive programs, dependencies, and represented nominal work. Label offline schedules and all timing assumptions. |

Detailed feasibility/safety confusion matrices can accompany Study 3 in
supplementary material. With additional model profiles, a model-by-failure
completion heatmap or completion-versus-cost scatter plot can summarize
Studies 1 and 2. These are alternative views of the same results, not
additional studies.

Selection plots, candidate decision graphs, generate-and-test curves, and
fixed-outline comparison panels belong to the relevant Possible studies.
Do not make them requirements of the core implementation.

Produce publication figures with standard plotting tools from exported tables.
Save plotting configuration/source, selected-data identifiers, SVG or PDF, and
PNG previews. Use accessible colors plus shapes or line styles, common scales
for compared panels, and gray unavailable cells. Caption every figure with its
stage, input origin, cohort, denominator, and evidence limits.

## 6. UI, records, and integration plan

The `experiments` placeholder at `/recovery-framework?tab=experiments` is
delivered. Build its first functional slice for Study 1: rung selection (all
four rungs, each unavailable rung showing its reason), capture readiness,
repetitions and budgets, the planned attempt matrix, explicit start/cancel
controls, progress, audited outcome results, and exports. Add Study 2 and
Study 3 controls afterwards. List Possible studies as deferred options without
active controls.

Only applicable combinations are selectable. An unsupported rung, stage, or
missing input has a visible reason and cannot launch a misleading comparison.
A failure where rung 4 has not completed is shown as not yet comparable.

Keep experiment records in a dedicated versioned directory under the existing
recovery monitoring area. Each batch has a frozen configuration/manifest and
each attempt has its own state, result, audit, and evidence references.
Persist queued, running, terminal, and interrupted status so page navigation
or reconnect does not launch duplicate work or discard completed results.

Use recovery-framework-owned orchestration modules and isolated sequential
workers. Extend the reserved evaluator in
[recovery_evaluation.py](../../cais_spade_llm/recovery_framework/recovery_evaluation.py)
with a versioned rung field rather than writing a second runner; it already
freezes inputs, isolates attempts, and blocks live calls. Preserve
`SystemBridge` as the public boundary for existing live-runtime access; keep
`cais_spade_llm/ui/bridge.py` read-only. The experiment worker uses saved
inputs and cannot dispatch live robot work. Do not change global model
configuration or the active recovery session to select a trial's settings.

Reuse the existing request writer, saved-observation construction, validators,
and evidence viewers where their contracts match the study. Keep legacy
evaluation manifests readable and preserve the reserved protocol's original
semantics. Historical records missing study metadata remain historical;
do not infer that they belong to a new controlled matrix.

### Integration notes for the focused plan

The existing recovery preparation, candidate-selection code, Start fresh
path, and reserved evaluator are distinct paths. Record the settings actually
used by a worker; do not infer that a `pure_llm` value disables validation or
that a `neurosymbolic` value proves every intended stage is integrated. The
Start fresh path sets `outline_diagnostic_bypass=true` even when every check
is selected, so its all-checks-selected behavior must be verified against the
legacy validated route before either is chosen as the shared pipeline for
rungs 2–4.

Rung 1 needs a single-turn prompt that supplies the same public facts the
multi-turn rungs can retrieve. Build or verify it before the pilot; a prompt
with less information would measure missing facts rather than turn structure.

The core pipeline needs verified primitive support, CCA checks, and final
audits for each enabled scenario. The retained recovery handoff currently
records an accepted Part slippage outline and an RA-accepted primitive
composition whose `cca_composition_status=not_checked`. That record must not
be promoted to the full Study 2 endpoint without the remaining required checks.
Retained failure captures and incomplete prefixes are not complete results.

Preserve current owner endpoint/predicate evidence policy. Missing evidence
cannot trigger fabricated timing or unsolicited trajectory generation.
Re-read the evolving runtime code before implementing workers, and keep the
reserved evaluator's original manifest semantics and historical reports intact.

## 7. Phased implementation and acceptance

Implement one numbered phase at a time, run its checks, and record a handoff
here before advancing. A completed software phase does not mean its
experiments have run. Recommended order: phase 3, phase 4, then complete
phase 2 using the record format fixed in phase 4, then phases 5 onward.

| Phase | Deliverable | Acceptance gate | Status |
| --- | --- | --- | --- |
| 1 | Separate experiment design document | Exact scenarios, studies, rung definitions, model extensibility, measurement definitions, figure specifications, and phase checklist; valid local references | Complete: 2026-10-13; revised 2026-10-14 |
| 2 | Experiments tab and persistent configuration/results | Navigation preserves form state; capture readiness and trial counts are correct; Study 1 rung selection shows unavailable rungs with reasons; records and exports round-trip; no automatic model calls | Partial: placeholder, navigation, and design download delivered 2026-10-14; configuration and persistent records pending |
| 3 | Independent outline auditor | Attempt 17's outline passes, including nominal continuation. Reviewed negative controls fail at the expected event and check: one each for RA feasibility, physical support, `SAFE_shared_area_mutex`, `SAFE_gear_small_before_KET4_Square_4mm`, and nominal continuation (the mock `part_slippage` plans may supply the two safety cases if their events convert to the outline format). Missing evidence yields `unresolved`. No provider, Gazebo, controller, or feedback path. | Next |
| 4 | Study 1 rungs and the `Part slippage` pilot | Rung 1 single-turn prompt with the same public facts. Rungs 2–4 on one shared pipeline with one proposal per turn and exact recorded settings. Rungs 1 and 2 make zero RA/CCA calls and receive no validation findings; rung 3 makes zero CCA calls. Experiment-record retention exception agreed. The pilot runs 12 attempts and every finished outline is audited. | Planned |
| 5 | Study 1 on the remaining failures | Each added failure enters only after rung 4 completes an audited outline there (runtime roadmap dependency); each runs as a frozen four-rung batch | Planned |
| 6 | Study 2 end-to-end workers | Fresh isolated attempts; actual settings captured; outline, primitive, and overall endpoints verified; required RA/CCA/nominal evidence; cancellation and retained history | Planned |
| 7 | Study 3 reviewed cases and policy replay | Identical inputs across four policies using the phase 3 auditor; independent labels; accurate acceptance and unresolved outcomes; zero live dispatch | Planned |
| 8 | Reports and figures | Correct rung, attempt, and case denominators; measured outcome, completion, and policy-error plots; one evidence-backed recovery example; reproducible exports | Planned |
| 9 | Possible studies | Implement only a later selected comparison with its own frozen protocol; no changes to core interpretation. The selector correction belongs to possible study A. | Deferred |
| 10 | Additional model adapters | Reuse core studies and records; preserve request/evidence contracts; show unsupported settings and missing usage | Later extension |

### Tests to add or extend during implementation

- **Outline auditor:** positive and negative controls, first-failure
  classification, events not audited after failure, unresolved missing
  evidence, unchanged source hashes, and no model feedback path.
- **Rung configuration:** each rung maps to exact recorded settings; rungs 1
  and 2 make zero validator calls and receive no validation findings; rung 3
  makes zero CCA calls; rung 1 receives the same public facts the multi-turn
  rungs can retrieve; rungs 2–4 propose one event per turn and share the
  stopping rule.
- **Configuration and model transport:** exact settings and identities, supported
  and unsupported parameters, raw capture for tools/errors/retries, unavailable
  usage, and adapter parity with the initial connection.
- **Input and worker isolation:** source or code changes, fresh sessions, conflicting
  starts, cancellation, process cleanup, restart/interruption, retained results,
  and absence of live agents, controllers, and observation calls.
- **Study 2:** correct successive state and stage handoff, complete outlines,
  primitive compositions and required audits, partial or unsupported stages,
  budget exhaustion, and same-attempt overall completion.
- **Study 3:** identical reviewed inputs, four policy decisions, independent
  safety/feasibility labels, overlapping violations, stale/missing evidence,
  unresolved outcomes, and case counts distinct from recovery attempts.
- **Possible studies:** add selector or generate-and-test verification when
  that comparison is selected. These are not core prerequisites.
- **Information boundary:** reference labels, review notes, and private
  evaluation notes stay outside all model prompts and retrieval.
- **Results and figures:** duplicate artifacts do not become extra trials;
  missing fields stay missing; filters retain denominators; all outcomes export;
  plot values match tables; ordinal diagrams do not imply measured duration.

Extend nearby suites. A new test file is appropriate for the coherent experiment
subsystem; retain existing regression coverage. Use mocked transports and saved
fixtures for automated checks without launching paid trials or robot execution.

For Python changes run `poetry check`,
`poetry run python -m compileall -q cais_spade_llm ros2`, the focused tests,
and `git diff --check`. Run
`poetry run python -m cais_spade_llm.ui_main --help` when entrypoints change.
If a later phase changes ROS2 launch/script/RViz files, run
`make bootstrap-gazebo` before installed-workspace checks. Report separately
whether live-model, Gazebo, or hardware validation actually ran.

### Phase handoff record

After each phase record its date, completed behavior, changed modules, checks
and outcomes, outstanding limitations, and next phase. List experiment batch
identifiers only for trials that actually ran.

| Date | Phase | Delivery and verification | Next step |
| --- | --- | --- | --- |
| 2026-10-13 | 1 | Separate document created. All 15 local link targets exist; required scenario/study sections, final newline, and whitespace checks passed. Python/runtime checks do not apply to this Markdown-only delivery. No UI/runtime edits or new trials. | Phase 2: Experiments tab and persistent records |
| 2026-10-14 | 2, partial | Added the `experiments` tab and Journal 2 sidebar link at `/recovery-framework?tab=experiments`, a Coming soon placeholder, and the experiment-design download. Existing browser-history handling includes the new tab. All 84 project-page tests passed, including direct links, one retained placeholder, download target, and no bridge access. `poetry check` passed with existing metadata deprecation warnings; Python compilation and scoped whitespace checks passed. No experiment trials ran. | Complete phase 2 configuration, capture readiness, and persistent records before adding workers |
| 2026-10-14 | Design focus | Reduced the required evaluation to full-workflow effectiveness and fixed-input feasibility/safety policy replay. Moved selector and generation-strategy comparisons to Possible studies. Clarified that both existing selector modes retain RA/CCA validation. Documentation changes only; no new experimental evidence. | Implement configuration and records for the two core studies first |
| 2026-10-14 | Study reorder and method definitions | Made a method comparison Study 1 and defined three methods by decision authority. Former Studies 1 and 2 became Studies 2 and 3. Added the experiment-record retention rule pending an AGENTS.md exception. Superseded the same day by the four-rung ladder. Documentation only; no code changes or trials. | Superseded |
| 2026-10-14 | Four-rung ladder | Replaced the three methods with the user-selected ladder: LLM single-turn, LLM multi-turn, LLM + RA, LLM + RA + CCA. Removed the separate LLM-with-validation method; one proposal per turn in rungs 2–4, so the symbolic selector moved to possible study A. Recorded construction-guaranteed eliminations, the missing generate-and-test baseline (possible study B), the rung 4 completion dependency, and current code availability for each rung. Documentation only; no code changes or trials. | Phase 3: independent outline auditor |

## 8. Repository references

- [Broader runtime roadmap and historical experimental design](IMPLEMENTATION_PLAN.md)
- [Formal validation and selection reference](JOURNAL_VALIDATION_AND_SELECTOR_ACTIONS.md)
- [Recovery page and tab construction](../../cais_spade_llm/ui/pages/recovery_framework.py)
- [Saved recovery result browsing](../../cais_spade_llm/ui/recovery_results.py)
- [Reserved outline evaluator and evidence reporting](../../cais_spade_llm/recovery_framework/recovery_evaluation.py)
- [Saved-observation recovery construction](../../cais_spade_llm/recovery_framework/offline_recovery.py)
- [Start fresh outline setup](../../cais_spade_llm/recovery_framework/outline_draft.py)
- [Selected outline checks](../../cais_spade_llm/recovery_framework/outline_checks.py)
- [Recovery outline preparation and stage controls](../../cais_spade_llm/recovery_framework/part_slippage_recovery.py)
- [Candidate selection and outline strategies](../../cais_spade_llm/agents/intelligent_product/replanner/llm_recovery/modes/multi_turn_outline_generation.py)
- [Candidate validation and comparison](../../cais_spade_llm/agents/intelligent_product/replanner/llm_recovery/modes/multi_turn_validated_event.py)
- [Diagnostic bypass bookkeeping](../../cais_spade_llm/agents/intelligent_product/replanner/llm_recovery/modes/multi_turn_diagnostic.py)
- [Isolated diagnostic jobs and checkpoint reuse](../../cais_spade_llm/recovery_framework/diagnostics.py)
- [Offline grounded composition checks](../../cais_spade_llm/agents/central_controller/offline_recovery_composition.py)
- [Diagnostic regression tests](../../test/test_recovery_diagnostics.py)
- [Validated outline regression tests](../../test/test_validated_recovery_outline.py)
- [Offline composition regression tests](../../test/test_offline_recovery_composition.py)
