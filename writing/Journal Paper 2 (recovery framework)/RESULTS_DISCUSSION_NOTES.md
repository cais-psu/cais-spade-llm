# Section 6.3 results: findings, draft text, discussion and limitations

Working notes from the 2026-11-10 review of the Study 1 results, for drafting Section 6.3 (Evaluation and
Analysis) and the Discussion and Limitations subsection.

**Source of every number:** the 720 Study 1 attempt records behind the RCIM Recovery Results page
(generated 2026-11-09 18:09 UTC; Study 1 outcomes regenerated 2026-11-12 with the audit review below; 18 batches in
`cais_spade_llm/monitor/environment_runs/recovery/outline_experiments/`), and the Study 2 records in
`.../recovery/study2_compositions/`. The informed baseline and framework batches ran on `9776af2`; the baseline
batches ran on `4ffcc2c` (see "Things to disclose").

**Audit review (2026-11-12).** The Study 1 outcomes in these notes follow the post-collection audit review. Every
attempt that ended with an outline (697 of 720) was re-audited with six checker rules corrected, the same rules for
every method; 118 outcomes changed, all in the two baselines (20 of them to passed), and every framework outcome
stayed the same. The review is a separate file beside the records
(`.../outline_experiments/audit_review_20261112.json`); `audit.json` and `batch.json` are unchanged, and the
Experiments tab and the results page (regenerated 2026-11-12) show the reviewed outcome with the recorded one
beside it. Evidence for each of the 720 attempts: `evidence/study1_audit_720.xlsx`. Effort numbers (tokens, time,
requests, lookups) belong to the attempt and do not change. Details: "Things to disclose" below and
EXPERIMENT_DESIGN_AND_PLAN.md, Deviations, 2026-11-12.

Contents:

1. [Figure captions](#1-figure-captions)
2. [Study 1 numbers](#2-study-1-numbers)
3. [Study 1 findings, major to minor](#3-study-1-findings-major-to-minor)
4. [Study 1 draft text](#4-study-1-draft-text)
5. [OL vs. EL: what can and cannot be inferred](#5-ol-vs-el-what-can-and-cannot-be-inferred)
6. [The 40-turn limit](#6-the-40-turn-limit)
7. [Discussion points](#7-discussion-points)
8. [Limitations](#8-limitations)
9. [Things to disclose](#9-things-to-disclose)
10. [Statistics](#10-statistics)
11. [Not yet discussed](#11-not-yet-discussed)
12. [Model comparison points](#12-model-comparison-points)
13. [6.3.3 Discussion and Limitations: insights and draft](#13-633-discussion-and-limitations-insights-and-draft)

**Section structure (decided 2026-11-10).** Organise by stage, with the LLM comparison inside each stage:

```
6.3 Evaluation and Analysis            setup (Table 4)
    6.3.1 Recovery Task Synthesis Results
          methods (Fig. 11) → OL vs. EL → LLMs (Fig. 12)
    6.3.2 Primitive Composition Results
          by scenario and by LLM (Table 5) → execution in Gazebo
          → closing point: the LLM ranking changes between stages
    6.3.3 Discussion and Limitations
```

Name the stage "recovery task synthesis", as in the framework section, and its output a "recovery outline". Keep
"outline-level (OL)". Drop the "Study 1:" and "Study 2:" paragraph labels unless the paper refers to them
elsewhere. Table 5 is full width (`table*`), with (a) by failure scenario and (b) by LLM.

---

## 1. Figure captions

Wording preferences settled in the review: no "audited", no "pool"/"pooled", no "LLM models", plain
"95% confidence intervals" in captions (name the Wilson method once in the text).

**Fig. 11** (panels as laid out in `figures/paper_figures/figure11_performance_results.png`):

```latex
\caption{Outcome of each attempt by method and failure scenario: (a) Conveyor breakdown, (b) Part slippage, (c) UR5e-1 robot breakdown, (d) M1 machine breakdown. Each bar shows 30 attempts (three~\glspl{llm} × 10 runs). Only the framework is checked during generation; after generation, every outline is checked with~\gls{ra} and~\gls{cca}. The median tokens, time and lookups per attempt are shown on the right.}
```

- Panel letters run down the columns ((a) top left, (b) below it). Most readers expect them to run across rows
  ((b) top right). If the panels are swapped, change the caption order to Conveyor, UR5e-1, Part slippage, M1.
- The effort columns are whole-attempt totals (all turns and events), and each shows the median over the row's
  30 attempts. Tokens = input (including cached) + output (including reasoning), summed over every request.
  Time = wall-clock generation time, including the RA and CCA checks during generation, excluding the final
  check. Lookups = facts retrieved from 𝒦. The medians include attempts that ended without an outline.

**Fig. 12:**

```latex
\caption{Comparison of the three~\glspl{llm}. (a) Share of attempts that passed, by method, across all four failure scenarios (40 attempts per marker), with 95\% confidence intervals; grey bars show the three~\glspl{llm} together. (b) Share of candidate events that each check rejected in the~\gls{el} neuro-symbolic framework.}
```

- Keep the grey-bar clause while the figure has grey bars.
- The legend says "All models pooled"; "All LLMs" would match the caption.
- `figure12_model_comparison.png` has a transparent background; export it on white before submission.
- Re-export both figures from the results page regenerated on 2026-11-12: the Baseline and Informed baseline bars
  in Fig. 11 and markers in Fig. 12a changed with the audit review. Fig. 12b does not change: it counts the
  framework's rejections during generation, under the checks as they ran. Neither caption needs to change; the
  review is disclosed in the text (Section 9 of these notes).
- Blocking (CCA) is not drawn in (b) because it runs once per outline, when the LLM declares the outline
  complete, not on each candidate event. Add "Blocking (CCA) is not shown because it checks the whole outline
  once, not each event." if a reader would miss it.
- 95% is a convention fixed before the runs (STUDY1_MODELS_AND_REPETITIONS.md: "Pass rates are reported with
  95% Wilson score intervals"); the code uses z = 1.96 (`experimental_results/fig12_model_comparison.py:71`).

## 2. Study 1 numbers

All three LLMs together, 120 attempts per row (3 LLMs × 4 scenarios × 10 runs).

| Method | Level | Passed | Unsafe (CCA) | Infeasible: symbolic (RA) | Infeasible: physical (RA) | Blocking (CCA) | No outline established | Median tokens | Median time | Median lookups |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | OL | 9 | 70 | 14 | 3 | 16 | 8 | 30k | 1.0 min | 15 |
| Baseline | EL | 1 | 74 | 14 | 5 | 18 | 8 | 90k | 2.1 min | 13 |
| Informed baseline | OL | 41 | 15 | 24 | 7 | 26 | 7 | 40k | 1.1 min | 16 |
| Informed baseline | EL | 29 | 15 | 27 | 5 | 32 | 12 | 109k | 2.8 min | 16 |
| Framework | OL | 112 | 0 | 0 | 0 | 0 | 8 | 78k | 3.7 min | 17 |
| Framework | EL | 106 | 0 | 0 | 0 | 0 | 14 | 210k | 6.2 min | 16 |

No outline established = `unresolved` (a finished outline the final check could not establish) plus
`did_not_finish` (no outline). In the baselines it is all `unresolved` except one Informed baseline attempt that
used its 40 requests; in the framework it is all `did_not_finish`. The baselines' 34 `unresolved` outlines: an
event leaves out a value it changes (28), the bounded arm search found no pose to judge (5), and one drive the
route check could not judge at the stated heading.

Recorded before the audit review (for reference only): Baseline 1 / 0 passed, Informed baseline 34 / 25 passed
(OL / EL); Infeasible: physical (RA) 92 in the two baselines. The 118 changed outcomes: physical to Blocking (35),
physical to passed (15), physical to symbolic (15), symbolic to Blocking (15), unresolved to symbolic (17, the
KMR's location named after a tray or the Buffer, which its model does not have), physical to unsafe (8), symbolic
to passed (5), and 8 others. An outline that passed a corrected check went on to its next check, so most former
physical refusals now end at a later real refusal, usually Blocking.

Passed per scenario (30 attempts each):

| Method, level | Conveyor | UR5e-1 | M1 | Part slippage |
|---|---|---|---|---|
| Baseline, OL | 4 | 5 | 0 | 0 |
| Baseline, EL | 0 | 1 | 0 | 0 |
| Informed baseline, OL | 4 | 8 | 9 | 20 |
| Informed baseline, EL | 0 | 4 | 1 | 24 |
| Framework, OL | 23 | 29 | 30 | 30 |
| Framework, EL | 29 | 29 | 30 | 18 |

Framework passes per LLM (10 per cell), OL / EL:

| LLM | Conveyor | UR5e-1 | M1 | Part slippage | Total |
|---|---|---|---|---|---|
| Claude Opus 5.5 | 10 / 10 | 10 / 10 | 10 / 10 | 10 / 8 | 40 / 38 |
| GPT-6.1 Sol | 9 / 10 | 10 / 9 | 10 / 10 | 10 / 5 | 39 / 34 |
| GPT-6.1 Terra | 4 / 9 | 9 / 10 | 10 / 10 | 10 / 5 | 33 / 34 |

## 3. Study 1 findings, major to minor

**Major**

1. **Validation, not knowledge of the rules, makes outlines meet the recovery requirements.** The framework
   passed 218/240 (91%) without the rules in its prompt, against 70/240 (29%) for the Informed baseline and
   10/240 (4%) for the Baseline (p < 0.001). It held for all three LLMs (83–100%) and in every scenario (at least
   18/30).
2. **Giving the rules in the prompt fixes safety only partly and moves the failures elsewhere.** Unsafe
   outlines fell from 144/240 (60%) to 30/240 (12.5%), but RA failures rose from 36/240 (15%) to 63/240 (26%)
   and Blocking from 34/240 (14%) to 58/240 (24%). The Informed baseline's passes are mostly part slippage (44/60,
   against 26/180 in the other three scenarios).
3. **Unchecked LLM proposals break every requirement.** The Baseline failed safety in 60% of attempts (144/240),
   the RA checks in 15% (36/240: 17 at OL, 19 at EL) and task continuation in 14% (34/240). Even inside the
   framework, about 40% of candidate events
   were rejected (Fig. 12b: 41%, 45%, 39%). The checks find the errors and the feedback gets them fixed; the
   LLM rarely gets them right the first time.

**Moderate**

4. **The physical check (Eq. 12) catches real errors, fewer than first recorded.** After the audit review, all 20
   Infeasible: physical (RA) outcomes in the two baselines are in conveyor breakdown (5) and UR5e-1 breakdown
   (15), the scenarios that need reach or a base pose at a new place: the KMR has no drive route even with one
   turn (10), the arm meets a station (5), or the stated base overlaps a station with the stated 0.03 m margin
   (5). The recorded audit had 92; most of the difference came from the checker's own strictness (Section 9).
   In the event-level framework the physical check rejected 1–22% of candidate events per LLM (Fig. 12b).
5. **The task-continuation check (Eq. 6) catches what checking each event cannot.** 92 baseline outlines are
   Blocking (58 Informed baseline, 34 Baseline): every event passed the safety and feasibility checks, yet nominal
   production could not resume. In 89 of the 92 the outline leaves the KMR at no named place (resource_location
   null, placed or idle), from which it cannot carry on its nominal tasks; the other 3 leave the square peg unable
   to reach its trim and assembly. They occur in conveyor (37), UR5e-1 (28) and M1 (27) breakdown, none in part
   slippage. (An earlier version of these notes put the M1 cases down to the trim order φ7; the refusals name the
   KMR.)
6. **The framework never finished an outline that fails a requirement.** All 22 of its unsuccessful attempts
   ended without an outline (16 at the turn limit, 6 at the no-progress stop). This holds by construction;
   state it as such.
7. **OL and EL are equally reliable, and OL is cheaper** (Section 5 of these notes).

**Minor**

8. **Validation costs about twice the tokens and two to three times the time** of the Informed baseline at
   the same level, but still minutes per attempt (median 3.7 min at OL, 6.2 at EL). Lookups stay at 13–17, so
   the extra cost reflects revisions after refused proposals, not more context retrieval.
9. **The turn limit rarely constrained passing attempts** (Section 6 of these notes).
10. **The baselines leave state values out:** 28 of their 480 outlines (6%; Baseline 15, Informed baseline 13)
    cannot be checked because an event omits a value it changes. The framework refuses such events and the LLM
    then states the value.
11. **The weakest model with the rules alone reaches the same level with the framework:** Terra rose from 18% in
    the Informed baseline to 83–85% with the framework, a gain of 65–68 points, about the same as Opus (30% to
    95–100%). Before the audit review Terra's gain looked the largest (from 10%); it no longer is. Belongs in
    6.3.2.

## 4. Study 1 draft text

Plain text as agreed in the review; convert names to `\gls{}` and references to `\Cref{}`.

- Equation numbers come from the 2026-11-07 notes (Eq. 6 task continuation, Eq. 7 feasibility, Eq. 8 safety).
  Check them against the current manuscript; `\Cref{subsubsec:recovery-requirements}` is a safe substitute.
- The scenario-dependence sentences in the second paragraph (findings 4 and 5) were not in the version read in
  the chat.

**Study 1: Outline Generation**

An LLM can propose recovery events outside the predefined plant model (RC1), but nothing in its output ensures
that these events satisfy the feasibility, safety, and task-continuation requirements in Eqs. 6–8 (RC2). Study 1
therefore examines whether these requirements can be met by informing the LLM of the safety rules, or whether
the symbolic validation by RA and CCA is needed. After generation, every recovery outline was checked against all
three requirements, whatever the method. Fig. 11 shows the outcomes of all 720 attempts (three LLMs, three
methods, two generation levels, four scenarios, and 10 runs each).

Without safety rules or validation, the Baseline passed only 10 of 240 attempts (4%). This result was expected, since
the LLM never sees the safety specifications. The Baseline is still needed because it receives the same prompt
as the framework, which also omits the rules. The two methods therefore differ only in RA and CCA validation, and
the Baseline shows what the LLM proposes when nothing checks it, which is what the framework's checks must catch.
Its most common failure was Unsafe (CCA), in 144 attempts (60%); a further 36 attempts (15%) failed the RA checks,
and 34 (14%) were Blocking. Supplying the rules in the prompt reduced unsafe outlines to 30 of 240 (12.5%) and
raised the pass rate to 70 of 240 (29%) in the Informed baseline. However, most of these passes came from part
slippage (44 of 60 attempts), compared with 26 of 180 in the other three scenarios. The remaining failures shifted
to the RA checks (63 attempts, 26%) and to Blocking (58 attempts, 24%), and they depended on the scenario. All 20
physical failures in the two baselines occurred in conveyor and UR5e-1 breakdown, where a robot or the KMR must
reach a new place. Blocking occurred only in conveyor, UR5e-1 and M1 breakdown: in 89 of the 92 Blocking outcomes,
every event was safe and feasible, but the outline left the KMR at no named place, from which it could not resume
its nominal tasks. Knowing the rules therefore helps the LLM avoid unsafe events, but
it does not let the LLM confirm that resources can perform the proposed events or that nominal production can
resume. In both baselines, the No outline established segments are finished outlines that the checks could not
establish, for example because an event left out a value it changes.

The neuro-symbolic framework passed 218 of 240 attempts (91%) without the safety rules in its prompt: 112 of 120
at OL and 106 of 120 at EL. It passed at least 18 of 30 attempts in every scenario, including the three scenarios
where the Informed baseline passed at most 9. The difference from the Informed baseline was significant at both
generation levels (Cochran–Mantel–Haenszel test stratified by scenario and LLM, p < 0.001 after Holm correction).
Because the framework accepts an event only after it passes RA and CCA, all 22 of its unsuccessful attempts
ended without an outline: 16 at the 40-turn limit and 6 after repeated proposals made no progress. These
attempts were concentrated in two cases: GPT-6.1 Terra on conveyor breakdown at OL (6 attempts) and part
slippage at EL (12 attempts). Validation feedback, not knowledge of the rules, is therefore what makes
LLM-generated outlines meet the recovery requirements (RC2).

*Audit review (one or two sentences, in the setup or here; numbers in Section 9 of these notes):* After the runs, a
review of the final check found that some of its geometric searches were stricter than the rules stated to the LLM;
for example, the KMR could turn only at the start of a drive, and its clearance was tested on a 5 cm grid rather
than the stated 3 cm margin. All 697 finished outlines were re-checked with these rules corrected, the same for
every method: 118 baseline outcomes changed, 20 of them to passed, and every framework outcome stayed the same. The
same rules were active while the framework generated its outlines, and they kept refusing 9 of its 22 unfinished
attempts until the turn limit, so the correction does not favour the framework.

Validation feedback increased cost. Per attempt, the framework used a median of 78k tokens and 3.7 min at OL,
and 210k tokens and 6.2 min at EL. This is about twice the tokens of the Informed baseline at the same level and
two to three times its time. The median number of lookups stayed similar across all methods (13 to 17), so the
added cost reflects revisions after refused proposals rather than additional context retrieval.

EL was included because generating a plan one step at a time, with feedback at each step, is a common way to
support LLM planning [cite]. In this study, it did not improve the pass rate: OL passed as many attempts (112
against 106 of 120, p = 0.25) at lower cost, using fewer tokens in all four scenarios. In part slippage, EL
lowered the pass rate for all three LLMs (18 of 30 against 30 of 30); its only gain was for GPT-6.1 Terra on
conveyor breakdown (9 of 10 against 4 of 10). In M1 machine breakdown, where recoveries had a median of 18
events, both levels passed all 30 attempts, and OL used less than a third of the tokens. One possible
explanation is that current LLMs can keep the complete recovery in context and reason over it in a single
answer, which removes the need to break it into steps. Since all three LLMs evaluated here are recent models,
this study cannot confirm that explanation, and EL may still benefit smaller or older models, a pattern also
suggested by GPT-6.1 Terra on conveyor breakdown.

All 22 unsuccessful framework attempts ended at the 40-turn limit or the no-progress stop. A larger limit might
allow some of them to finish, but the limit was kept for three reasons. First, it was set before the runs and
applied equally to all methods, so raising it after seeing the results would favour the method that uses the
most turns. Second, it rarely constrained successful attempts: passing attempts needed a median of 6 requests at
OL and 16 at EL, and 95% finished within 22 and 30 requests, respectively. Third, the unsuccessful attempts were
not converging: in their last turns they repeated proposals that the checks had already refused, and each used
between 0.28 and 0.73 million tokens, because every request resends the growing conversation. A larger limit was
therefore not tested.

## 5. OL vs. EL: what can and cannot be inferred

**Can be inferred**

- **OL is cheaper.** Median tokens are lower at OL in all four scenarios (Conveyor 193k vs. 218k, UR5e-1 96k vs.
  158k, M1 82k vs. 270k, Part slippage 50k vs. 97k). Time is lower in three of four; Conveyor is about equal.
  Passing attempts needed a median of 6 requests at OL against 16 at EL.
- **Neither level is more reliable overall.** 112 vs. 106 of 120 (p = 0.25). Per LLM: Opus 40 vs. 38, Sol 39
  vs. 34, Terra 33 vs. 34.
- **EL produced detours that OL did not:** 11 of the 29 EL passes in conveyor breakdown move the square peg back
  to a place it had already left (for example, staging tray → UR5e-1 → M1 → UR5e-1 → M1 → KMR); none of the 23
  OL passes do.

**Report only as observations**

- Part slippage favours OL for all three LLMs (30/30 vs. 18/30, p < 0.001); consistent across models, but one
  scenario.
- Conveyor favours EL only for GPT-6.1 Terra (4/10 vs. 9/10, p ≈ 0.06); one cell of 10 runs.

**Cannot be inferred**

- **Why each level fails where it does.** From the refusals of the 22 failed attempts:
  - OL on conveyor rewrote the whole outline 33–37 times, and every version still had an event the RA refused.
  - EL on part slippage got 5–16 events accepted, then kept proposing events that could not be accepted. In 9 of
    the 12, these were events at Exit, whose state no primitive can change. In the other 3, UR5e-3 tried to grasp
    the gear while still holding the peg.

  These are hypotheses, not results of the study.
- **"OL for short recoveries, EL for long ones."** The data points the other way. On the shortest recovery (part
  slippage, median 5 events) OL was better; on the longest (M1, median 18–19 events) both passed 30/30 and OL used
  under a third of EL's tokens. Recovery length also cannot be separated from the scenario, since each scenario
  differs in resources, failure and constraints. A length claim would need recoveries of different lengths within
  one scenario.
- **"We expected EL to be better."** H4 in EXPERIMENT_DESIGN_AND_PLAN.md (line 239) set no direction: "the paper
  reports whichever mode does better and does not assume incremental wins". Motivate EL by common practice instead
  (draft text above).

## 6. The 40-turn limit

| | Median requests (passing attempts) | 95% finished within | Max |
|---|---|---|---|
| Framework, OL | 6 | 22 | 35 |
| Framework, EL | 16 | 30 | 35 |

- Of the 22 failed framework attempts, 6 were stopped by the no-progress rule (repeating the same refusal). The
  other 16 were repeating refused proposals in their last turns.
- Failed attempts used 0.28–0.73 million tokens each, more than 90% of it input, because each request resends
  the conversation.
- Reasons for keeping the limit:
  1. It was set before the runs and is the same for every method.
  2. Passing attempts rarely came near it.
  3. The failed attempts were not converging.
  4. Token use and planning time grow with each turn, and a recovery plan is needed while the cell is stopped.
- Give tokens and time as the reason, never money. Say plainly that a larger limit was not tested.

## 7. Discussion points

Open the Discussion with the strengths, then non-optimality as an honest observation with future work, and
close with the limitations.

**Strengths, each backed by data**

- **Robust across LLMs and failure types.** The effect holds for three LLMs from two providers and four failures
  that differ in what failed: a part, a machine, the conveyor and a robot.
- **Repeatable recovery beyond the plant model (RC1).** The behaviours described in Section 6.2, such as the KMR
  delivering to the buffer from a new base pose, are not one-off examples: 91% of attempts produced a valid
  recovery.
- **The LLM proposes and the checks decide.** The LLM is good at proposing behaviour outside the plant model but
  unreliable at confirming it is valid. The framework splits those jobs and works without showing the LLM the
  rules, so the rules stay in the formal model. Changing a specification means changing the model, not the
  prompt.
- **Knowing the rules is not the same as checking them.** Even with the rules, the LLM could not work out reach
  and free space, or whether the remaining production can still finish. Those need calculation on the models, not
  reading. This is the argument for a neuro-symbolic approach over prompt engineering.
- **Every check is needed.** Physical failures occur only where reach matters (conveyor, UR5e-1), and
  continuation failures where the recovery leaves a shared resource, almost always the KMR, unable to resume its
  nominal tasks (conveyor, UR5e-1, M1). Each check catches errors the others would miss.
- **Safe failure.** The framework never accepted an outline that breaks a requirement; when it failed, it
  produced none, leaving the disruption to an operator instead of the robots.
- **Modest overhead:** minutes per attempt, before execution, with no extra context lookups.
- **Practical default:** OL, with the same reliability as EL at lower cost.
- **A weaker model reaches the same level** (Terra: 18% → 83–85%, a gain about equal to Opus's 30% → 95–100%).
- **Rigorous protocol:** models chosen by criteria set in advance (model screen), equal runs per cell, a fixed
  turn limit, the statistical test chosen before the data, and every started attempt reported.

**Valid, but not optimal**

- Passed outlines for the same failure varied widely in length:

  | Scenario | Events in passed outlines |
  |---|---|
  | Conveyor | 7–22 |
  | UR5e-1 | 4–21 |
  | M1 | 14–27 |
  | Part slippage | 4–7 |

  Some valid recoveries are three to five times longer than the shortest found for the same failure.
- Circular moves: 11 of 29 EL passes in conveyor breakdown move the square peg back to a place it had left; none
  at OL. Moving the round peg out of M2 and back in M1 machine breakdown is intended (unload, then reload to
  resume) and is not counted.
- Why: the requirements (Eqs. 6–8) ask only whether a recovery is valid; Eq. 6 asks only that a continuation
  exists. Nothing scores length or time, so the framework accepts any valid outline.
- Future work: a cost criterion for choosing among valid outlines. Under the no-hidden-rules policy, it must not
  be code that edits the LLM's events.

**Generality and solution bias**

The case for "not biased to the solution":

- The prompt code names no scenario, part or solution. Scenario names appear only in the code that injects
  failures in simulation, and all four scenarios get the same seven safety specifications.
- The prompt states facts (state, models, capabilities), never steps; refusals say what is wrong, not how to fix
  it.
- The Baseline gets the same facts as the framework and passed 10/240 (4%). If the prompt gave the answer away,
  the baselines would pass.
- Passed outlines for one failure differ in resource assignment and length, which rules out a hidden recipe.

What still limits the claim:

- The prompt wording and choice of facts were refined while running these four failures.
- The cell was made recoverable (staging trays moved 0.30 m north on 2026-11-05 so the KMR reaches them from its
  dock).
- Part slippage carries a CCA assumption (UR5e-3 clear of the board) that corrects the capture, which ran with
  the CCA bypassed.
- Four failure types is not "many".

Claim: "failure-agnostic by design, demonstrated on four failure types". Do not claim "generalizes to many
failures". Suggested wording:

> The framework contains no failure-specific logic: all four scenarios used the same prompt, specifications, and
> checks, and the prompt states the cell's facts and models without describing recovery steps. The Baseline
> received the same facts and passed 10 of 240 attempts, so the facts alone do not reveal a recovery, and the
> passed outlines differed in resource assignment and length across attempts. However, the prompt was developed
> using these four scenarios, and the evaluation covers four failure types in one cell. Generality to other
> failures and cells therefore follows from the design but remains to be shown.

## 8. Limitations

Put 1–4 first; reviewers are most likely to raise them. Give each a concrete next step.

1. **Real cells.** The checks are only as good as the models: recorded geometry, primitive declarations, and
   MoveIt's collision model. Real cells add contact and friction, grasp slip, uncertain part poses, sensor noise,
   calibration error and timing. The guarantees hold relative to the models; real execution needs hardware
   validation. *Next step:* run compositions on hardware.
2. **Other cells.** One cell, one layout, one set of resources, and four scenarios designed by the authors.
   Resources, positions and primitives come from configuration files, so another cell needs new models rather
   than new code, but this is untested. *Next step:* a second cell with different resources, and scenarios not
   used during development.
3. **The framework's pass is judged by its own checks.** The study shows that outlines satisfy the paper's formal
   requirements, not that they are correct in the real cell. Study 2's motion-planning checks and the Gazebo runs
   give partial evidence beyond the checks.
4. **No feedback-only comparison.** The study cannot separate the value of the symbolic feedback from simply
   having more turns. *Next step:* a baseline that hears only "invalid, try again".
5. **The checks are sound but not complete.** Eq. 12's physical evidence comes from bounded searches: arm poses
   from a fixed set of seeds, one arm transfer path, KMR routes with a limited number of turns, and the base
   controller's footprint test. An outline they pass is feasible in the models, but a feasible outline can be
   refused. The audit review found such refusals in the baselines and corrected the rules for the final check
   (Section 9); the same rules ran during the framework's generation. Two conventions the checks used were not in
   the prompt: the stated 0.03 m base margin was tested on a 5 cm grid (about 0.055–0.08 m in effect), and an
   event's actions were checked against the state before the event, so "release, then go home" was refused.
   *Next step:* fix these rules in the code, state each convention in the prompt, and widen the searches.
6. **The prompt was developed on these scenarios** (first on gpt-5.4, later fixes tried on Claude Opus 5.5), which
   may favour them and that model.
7. **Recovery quality is not measured.** "Passed" means valid, not efficient (Section 7, "Valid, but not
   optimal").
8. **Failures are assumed detected and observed correctly.** Each attempt starts from a recorded capture;
   detection and perception errors are out of scope by assumption.
9. **Fixed settings:** one reasoning effort (medium) and a 40-turn limit; larger limits not tested.
10. **Model versions:** GPT-6.1 Sol and Terra are aliases with no dated snapshot; give the access dates.
11. **Small per-cell samples:** 10 runs per cell cannot resolve differences within one scenario (such as Terra on
    conveyor). The explanations of the OL and EL failures are hypotheses.

## 9. Things to disclose

- **Commits:** the Baseline batches ran on `4ffcc2c`; the Informed baseline and framework batches on `9776af2`.
  The only change is a one-line fix to a crash in the task-continuation check when an outline states a part
  location as an object, which an earlier Baseline batch hit (`batch_…79f045`, 3 errors; it and the unfinished
  `batch_…195a39` are not used). No Informed baseline or framework run reached that code.
- **Part slippage CCA assumption** (UR5e-3 clear of the board), because its capture ran with the CCA bypassed
  (`recovery_framework/recovery_case_safety.py`).
- **The cell was adjusted so recoveries are physically possible** (staging trays moved north on 2026-11-05).
- **Prompt development:** first developed on gpt-5.4 (frozen 2026-10-21); later code fixes tried on Claude Opus 5.5.
  Put this in the Section 6.3 text, not a caption.
- **Model IDs and access dates:** `claude-opus-5-5`, `gpt-6.1-sol`, `gpt-6.1-terra`; Sol and Terra are aliases.
- **No dollar costs**; effort is tokens, turns and time.
- **Audit review (2026-11-12), after collection.** The review asked whether the final check refused outlines for
  reasons the outline was not at fault for. It re-audited every finished outline (697; the 23 that did not finish
  have no outline) with six rules corrected, the same for every method, and changed 118 outcomes, all baseline
  (Baseline passed 1 → 10, Informed baseline 59 → 70); the framework's 218 passes and 22 unfinished attempts stay.
  The corrected rules:
  1. the KMR drive may turn in place once anywhere on the way (it could turn only at its start; the real base
     controller turns along the way);
  2. when an event leaves the KMR at no named place with no pose, the base search may pick a free spot but never a
     dock (it tried docks first, then refused the event for not naming the dock it picked);
  3. a release followed by a move home in one event is checked in that order;
  4. a KMR base is judged by the stated rule, its footprint plus 0.03 m against each obstacle outline, not the
     controller's 5 cm grid test;
  5. the arm search tries 24 more seeds when it finds no pose;
  6. where the KMR arm's fixed transfer path fails, the arm may move joint by joint to a pose at the same approach
     point and then take the RA's straight descent.

  Two relabels: a KMR location named after a tray or the Buffer (no such dock in its model) is Infeasible:
  symbolic (RA), not unresolved; a stated base overlapping a station is Infeasible: physical (RA). Kept as
  recorded after review: the stated-pose rule, schema errors (undeclared variables, labels no action sets, omitted
  values), Blocking for a KMR left away from a dock, every Unsafe outcome (including the trim-order rule and the M2
  mutex at docking), and `complete_job` stated together with its start (uncontrollable). The recorded `audit.json`
  and `batch.json` are unchanged; the review is `outline_experiments/audit_review_20261112.json`, read by the
  Experiments tab and the results page, and each of the 720 attempts is listed with its evidence in
  `evidence/study1_audit_720.xlsx`.
- **The same rules stopped framework attempts.** 9 of the framework's 22 unfinished attempts (Terra conveyor OL 6,
  Sol conveyor OL 1, Terra conveyor EL 1, Terra UR5e-1 OL 1) kept being refused by rules 1 and 2 until the
  40-request limit. They have no outline to re-audit and cannot be re-scored without new generation, so they stay
  as did not finish. Fig. 12b likewise counts rejections under the rules as they ran.
- **Eq. 12 wording.** The paper's revised Eq. 12 text says an event's primitives are checked with no order between
  them and no chaining through a failed precondition. The review passes two Sol outlines whose single event
  releases a part and then moves home (rule 3), because the prompt never told the LLM this. Either keep the
  wording and disclose these two, or say that an event's actions are checked in their stated order.

## 10. Statistics

- **Confidence intervals:** 95% Wilson score intervals, stated in the plan before the runs.
- **Comparison test:** Cochran–Mantel–Haenszel, stratified by scenario × model as the plan states, continuity
  corrected, Holm correction over the four confirmatory comparisons. Recomputed 2026-11-12 on the reviewed
  outcomes:

  | Comparison | Level | χ² | p | Holm-adjusted p |
  |---|---|---|---|---|
  | Framework vs. Baseline | EL | 179.8 | 5.4×10⁻⁴¹ | 2.2×10⁻⁴⁰ |
  | Framework vs. Baseline | OL | 168.8 | 1.4×10⁻³⁸ | 4.0×10⁻³⁸ |
  | Framework vs. Informed baseline | OL | 102.0 | 5.6×10⁻²⁴ | 1.1×10⁻²³ |
  | Framework vs. Informed baseline | EL | 99.0 | 2.6×10⁻²³ | 2.6×10⁻²³ |

  All p < 10⁻²². Informed baseline vs. Baseline (descriptive in the plan): OL p = 1.9×10⁻⁷, EL p = 1.6×10⁻⁹.
  OL vs. EL within the framework: p = 0.25 (framework outcomes unchanged).
- **Pass rates with 95% Wilson intervals** (all levels and LLMs): Baseline 10/240, 4.2% (2.3–7.5); Informed
  baseline 70/240, 29.2% (23.8–35.2); framework 218/240, 90.8% (86.5–93.9).
- These p-values were computed in the review with a hand-written script, since no statistics package is
  installed. Recompute them with a package before submission; the conclusions do not depend on the exact method.
- The plan pre-stated OL vs. EL and Baseline vs. Informed baseline as descriptive comparisons. If the paper follows
  the plan exactly, drop their p-values.
- If p-values appear in the paper, add one sentence naming the test to the setup.

## 11. Not yet discussed

### 11.1 LLM comparison (Section 6.3.2, Fig. 12)

Numbers available:

| Setting | Opus 5.5 | GPT-6.1 Sol | GPT-6.1 Terra |
|---|---|---|---|
| Baseline, OL | 5% | 5% | 13% |
| Baseline, EL | 0% | 0% | 3% |
| Informed baseline, OL | 30% | 55% | 18% |
| Informed baseline, EL | 30% | 25% | 18% |
| Framework, OL | 100% | 98% | 83% |
| Framework, EL | 95% | 85% | 85% |

Baseline and Informed baseline rows follow the audit review (40 attempts per cell: Baseline 2, 2, 5 at OL and 0, 0,
1 at EL; Informed baseline 12, 22, 7 at OL and 12, 10, 7 at EL).

Candidate events rejected in the EL framework (Fig. 12b, rejections during generation, unchanged by the review),
Opus / Sol / Terra:

| Check | Opus | Sol | Terra |
|---|---|---|---|
| Infeasible: symbolic (RA) | 27% | 31% | 12% |
| Infeasible: physical (RA) | 1% | 8% | 22% |
| Unsafe (CCA) | 13% | 6% | 5% |
| All checks | 41% | 45% | 39% |

What to say: Section 12.1. The model screen (Sonnet 5.5 and Opus 4.6 excluded by pre-stated criteria) goes in the
setup.

### 11.2 Study 2: primitive composition

Numbers available (results page, Table 7):

- 36 compositions, 3 per LLM and scenario.
- Passed: Opus 12/12, Sol 11/12, Terra 4/12, 27/36 in all. Share of events composed: 100%, 97%, 56%.
- Terra's Machining counts are its Conveyor-fix re-runs, reported in place of the first compositions (Option B,
  decided 2026-11-12; Deviations). The first compositions were 0/3 (5, 5 and 6 of 18 events), all stalled at the
  Conveyor event served without descriptions or outputs; the re-runs gave 18/18, 2/18 and 4/18.
- Sol's 11/12 counts its Part slippage second attempts. On first attempts Sol is 9/12 (all three Part slippage
  first attempts stalled on lookups of `product_geometry` and `/geometry/Gear_Plate`), so 25/36 overall (24/36
  with Terra's first Machining compositions too). Opus
  got no second attempt (it did not need one); Terra got none by decision. Decided 2026-11-10: the table keeps
  Sol's second attempts as they are, with no footnote.
- By scenario: Conveyor 8/9, UR5e-1 6/9, M1 7/9, Part slippage 6/9 (4/9 on first attempts).
- Medians per composition: 7 events, 26 primitives, 2 refused programs, 23 calls, 61.5 lookups, 313k tokens,
  14.8 min. M1 is the heaviest: 446k tokens, 38.6 min.
- The 9 failed compositions (8 Terra, 1 Sol second attempt) stopped after 1–8 accepted events. 6 of the 9 stalled
  (6 turns without progress); CONV_terra_3 ended on a MoveIt refusal (KMR joint a6 at its limit), PS_terra_3 when
  the LLM reported that no primitive can reach the stated state, and the MACH_terra_1 re-run on a refusal that the
  `pick_ctx` its program passes differs from the pick context the calculation reads. Seven stopped at an event
  outside the resource's nominal functions: KMR placing in the buffer, KMR picking from M1, UR5e-3 handling the peg
  or the dropped gear. The two failed Machining re-runs stopped at ur5e-1 placing the square part on the conveyor,
  an event their first compositions had composed.
- All 9 composed conveyor outlines use the KMR to deliver the peg to the buffer, a task outside its nominal model;
  8 of the 9 were fully composed.
- Code: `386b8cd` / `7f6f712`, the same Python; the C++ change reached only a gear insertion.

Issues to fix in the text:

- **Selection wording.** The LaTeX says "we selected the three passed event-level framework outlines, generated by
  each LLM". The compositions used the three with the **fewest events** per LLM and scenario (all 36 queue items
  match `selection_shortest.json`, not `selection.json`). Say so, and note that composing the shortest outlines
  may favour composition success.
- **Sample size.** 3 compositions per cell is enough for counts, not for a significance test between models.
- **Evidence beyond the checks.** Composition adds motion planning (MoveIt) on the recorded cell. This is the
  first evidence beyond the outline checks (limitation 3).

### 11.3 Execution in Gazebo (Table 8)

- Table 8 is empty on the results page (`execution` is null), but four runs exist, all completed: Part slippage
  4/4, Conveyor 7/7, UR5e-1 4/4, M1 14/14 (`.../recovery/gazebo_runs/runs/final_*`).
- Disclose:
  - one run per scenario, chosen by fewest events across models;
  - the live CCA was bypassed, as for the captures;
  - M1 is the second take, after a base-controller fix.
- These runs support "the composed programs run in simulation" as a demonstration. They do not support an
  execution success rate. The logs checked show only the recovery events, not production resuming afterwards.
- Claiming more ("N of 26 compositions executed", "production resumed") would need more Gazebo runs: all 26, or a
  sample chosen in advance.

### 11.4 Other

- **Study 3** (removing one check at a time on fixed inputs) from the October plan was not run. If the paper text
  still mentions it, cut that text; Table O1 already shows which check rejected what.
- **Optional figures:** Fig. O1 (context retrieval) and Table O1 (what validation rejected) on the results page are
  not discussed yet.
- **Machine Breakdown paragraph:** in Section 6.2 it starts with a lowercase "machine M1 fails".

## 12. Model comparison points

Framing sentence for both sections: "The comparison shows whether the framework's effect holds across LLMs, not
which LLM is best."

### 12.1 In 6.3.1 (recovery task synthesis)

**Major**

1. **Good: the framework's effect holds for every LLM.** In the framework the three LLMs passed 83–100% of
   attempts, against 18–55% in the Informed baseline and 0–13% in the Baseline, with two providers. The method
   does not depend on one model.
2. **Good: validation narrows the gap between LLMs.** The Informed baseline ranges from 18% to 55% across LLMs;
   the framework ranges from 83% to 100%. Terra, the lower-priced model and the weakest with the rules alone
   (18%), gains about as much as Opus (Terra 18% → 83–85%, Opus 30% → 95–100%).

**Moderate**

3. **Good for the framework: every LLM needs the checks.** Each LLM had 39–45% of its candidate events rejected,
   so even the strongest model does not propose valid events reliably on its own.
4. **Good: prompting alone is model-dependent, the framework is consistent.** Informed baseline results swing by
   model and scenario. Sol's 55% at OL comes mostly from M1 (9/10) and Part slippage (10/10), and Opus reaches
   10/10 on Part slippage at EL. The framework is uniformly high.
5. **Observation: each LLM makes different errors.**
   - Terra's rejections are mostly physical (22%, against 1% for Opus and 8% for Sol): poses, bases or routes the
     resources cannot reach.
   - Opus's and Sol's rejections are mostly symbolic (27% and 31%).
   - Opus has the most unsafe rejections (13%).

   Describe this; do not rank on it.

**Minor**

6. **OL vs. EL per LLM:** Opus 40 vs. 38, Sol 39 vs. 34, Terra 33 vs. 34. Terra is the only LLM that did not do
   better at OL, which comes from conveyor breakdown (4/10 at OL against 9/10 at EL).
7. **Caveats (one or two sentences, so the comparison is not read as a ranking):**
   - The LLMs differ in vendor, size and release date.
   - The prompt was first developed on gpt-5.4, and later fixes were tried on Opus.
   - Sol and Terra are aliases with no dated snapshot.
   - Only medium reasoning effort was tested.
   - The model screen excluded weaker candidates (Sonnet 5.5 recovered 1 of 4 scenarios).

### 12.2 In 6.3.2 (primitive composition)

**Major**

1. **Important: composition separates the models, and the ranking changes between stages.**
   - Opus composed 12/12 outlines (100% of events), Sol 11/12 (97%) and Terra 4/12 (56%).
   - Terra passed 83–85% in synthesis, so synthesis success does not predict composition success.
   - The checks make up for a weaker model at the task level, but not yet at the primitive level, where every
     number must refer to 𝒦 and every motion must pass motion planning.
2. **Good: strong LLMs compose reliably.** Opus and Sol composed 23 of 24 outlines, so the stage works with a
   capable model.

**Moderate**

3. **Bad: Terra fails by stalling, not by being refused.** Refused programs are similar for all three LLMs
   (median 2, 1 and 2). 6 of the 9 failed compositions ended after six turns without progress, so Terra's problem
   is producing a program at all, not producing wrong ones.
4. **Bad: more effort did not help.** Terra used the most lookups (76.5) and tokens (417k) per composition and
   still failed 8 of 12.
5. **Observation: the failures cluster on new uses of resources.** Seven of the nine failures stopped at an event
   outside the resource's nominal functions:
   - the KMR placing in the buffer;
   - the KMR picking from M1;
   - UR5e-3 handling the peg or the dropped gear.

   Opus and Sol composed the same events in their own outlines, so these events are harder, not impossible.
   Correction (audit, 2026-11-12): the Conveyor advance to loading position 2 appears only in Terra's three Machining
   outlines, so no other LLM composed it, and the first compositions were served the Conveyor's primitives without
   descriptions or outputs. In the re-runs with them served (Table 7 reports these), MACH_terra_3 composed the
   Conveyor event and the whole outline; the other two stopped earlier, at ur5e-1 placing the square part on the
   conveyor, which their first compositions had composed. The same outline can therefore stop at different events
   from one composition to the next.

**Minor**

6. **Cost differs by LLM:** Opus needed the fewest tokens and least time per composition (176k tokens and 13.2 min,
   against 296k and 14.4 min for Sol and 417k and 17.3 min for Terra).
7. **Caveats:**
   - 3 compositions per LLM and scenario is enough for counts, not for a statistical test between LLMs.
   - Point back to the 6.3.1 caveats rather than repeating them.

Close 6.3.2 with major point 1, since it ties the two sections together. It also motivates future work: making
composition reliable for weaker models.

## 13. 6.3.3 Discussion and Limitations: insights and draft

6.3.1 and 6.3.2 report what happened, with the numbers. 6.3.3 says what it means: why it happened, what it implies
for building recovery systems, and what the study cannot show. Cite results by section or figure instead of
repeating numbers. Sections 7 and 8 of these notes hold the underlying points.

### 13.1 From result to insight

| Result (stays in 6.3.1 / 6.3.2) | Insight (goes in 6.3.3) |
|---|---|
| Framework 91% vs. Informed baseline 29% | **Verification cannot be handed to the LLM, even when it knows the rules.** Reach, free space and whether production can still finish need calculation, not reading. The LLM's value is proposing behaviour outside the model; checking it belongs to symbolic methods. For practitioners: invest in formal models and checks rather than prompt engineering. |
| The framework passes without rules in the prompt | **Specifications belong in the model, not the prompt.** Changing a rule means changing the model; the prompt does not grow with the rule set, and the LLM cannot overlook a rule it never needed to read. |
| Physical failures in conveyor and UR5e-1; Blocking where the KMR is left off its nominal places | **The requirements complement each other.** Different disruptions put pressure on different requirements, so a recovery framework needs feasibility (symbolic and physical), safety and continuation together. Continuation is a property of the remaining production as a whole, invisible when checking event by event. |
| Unsuccessful framework attempts end with no outline | **Validation turns LLM unreliability into "no plan" rather than "a bad plan".** That makes the framework suitable alongside an operator: an unsolved disruption goes to a person, not to the robots. |
| OL as reliable as EL, and cheaper | **Breaking the task into steps was not needed with current LLMs:** whole-outline reasoning with feedback was enough. Hedged: smaller models may still benefit. |
| Terra 83–85% in synthesis, 33% in composition | **The hard part moves down to composition.** With checks, task-level reasoning is largely handled; turning new uses of a resource into primitive programs is where models now struggle. Future effort goes to primitive knowledge, such as examples of checked programs. A cheaper model may be enough for synthesis while composition still needs a stronger one (a possibility only; not tested). |
| Outline lengths 4–27 events; detours | **Valid is not optimal.** The formulation only asks that a continuation exists (Eq. 6), so nothing prefers a shorter or faster recovery. Adding a cost criterion while leaving every event value to the LLM is an open question. |
| No failure-specific logic; Baseline 10/240 on the same facts | **Failure-agnostic by design, demonstrated on four failure types;** generality beyond this cell remains to be shown. |

### 13.2 Paragraph plan

1. Verification belongs to symbolic methods; specifications belong in the model (rows 1–2).
2. The requirements complement each other; the framework fails safe (rows 3–4).
3. Where the difficulty lies now: composition, generation level, model choice (rows 5–6).
4. Valid is not optimal; generality (rows 7–8).
5. Limitations, without numbers, ending with future work.

### 13.3 Draft text

Plain text; convert names to `\gls{}` and references to `\Cref{}`. Equation numbers follow the 2026-11-07 notes.

**Discussion and Limitations**

The results suggest a clear division of labour between the LLM and the symbolic layer. The LLM's contribution is
proposing recovery behaviour outside the predefined plant model (RC1): reassigning tasks, using resources for tasks
they never perform in nominal production, and finding new handoff points. Confirming that this behaviour is valid
could not be left to the LLM, even when it was given the rules (Section 6.3.1). Whether a robot can reach a pose,
whether the mobile base fits along a route, and whether the remaining production can still finish are calculations
on the cell's models, not facts that can be read from a rule. The framework therefore keeps the safety
specifications in the formal model and out of the prompt. Changing a rule means changing the model the CCA checks,
the prompt does not grow with the rule set, and the LLM cannot overlook a rule it never had to read. For
practitioners, this suggests investing in formal resource models and checks rather than in prompts that describe
the rules.

The recovery requirements also proved complementary. Different disruptions put pressure on different requirements:
with the rules in its prompt, the LLM failed part slippage mostly on safety, while in the three failures that
move parts with the KMR or a robot to new places it failed mostly on feasibility and task continuation (Fig. 11). Task continuation (Eq. 6) concerns the
remaining production as a whole and cannot be seen by checking events one at a time. A recovery framework therefore
needs all three requirements, not a single check. Validation also changes how the system fails. Because an outline
is accepted only when every event passes, an unreliable LLM proposal leads to no plan rather than a wrong plan. A
disruption the framework cannot resolve is handed to an operator instead of reaching the robots, which makes the
framework suitable for operation alongside people.

Within the framework, the difficulty moved from deciding what to do to deciding how to do it. With the checks, all
three LLMs produced valid recovery outlines in most attempts, and generating one event per turn brought no benefit
over proposing the whole outline at once. This suggests that current LLMs can reason over a complete recovery when
they receive feedback, although smaller or older models may still benefit from step-by-step generation. Turning
outlines into primitive programs was harder and separated the LLMs (Section 6.3.2). The events that failed were
those in which a resource performs a task outside its nominal functions, where the decompositions of nominal
functions give the least direct guidance. This points future work toward richer primitive knowledge in 𝒦, such as
programs that have already passed the checks. It also suggests that a less capable model may be sufficient for task
synthesis while composition still requires a stronger one, although this study did not test such a combination.

The framework guarantees validity, not efficiency. The recovery requirements ask only that a continuation exists
(Eq. 6), so any valid recovery is accepted. Recoveries for the same failure varied widely in length, and some moved
a part back to a place it had already left. Preferring shorter or faster recoveries would require a cost criterion
for choosing among valid outlines while leaving every event value to the LLM, which remains open. Regarding
generality, the framework contains no failure-specific logic: all four scenarios used the same prompt,
specifications and checks, and the prompt states the cell's facts and models without describing recovery steps. The
Baseline received the same facts and rarely produced a valid recovery, so the facts alone do not reveal a recovery.
Generality across failures therefore follows from the design, but the evaluation demonstrates it on four failure
types in one cell.

The study has several limitations. First, all results come from simulation. The checks are only as accurate as the
models they use: the recorded geometry, the declared primitives, and the collision model used in motion planning. A
real cell adds contact and friction, grasp slip, uncertain part poses, sensor noise and calibration error, so
validation on hardware is still needed. Second, the evaluation covers one cell and four failures designed by the
authors, and the prompt was developed using these failures; other cells, resources and unseen failures remain to be
tested. Third, the framework's outlines are judged by the same checks that guide their generation, so Section 6.3.1
shows that outlines satisfy the formal requirements, not that they are correct in the physical cell. Primitive
composition with motion planning and the executions in Gazebo provide partial evidence beyond these checks, but only
for the shortest outlines, three per LLM and scenario, and for one execution per scenario. Fourth, the study does not
separate the value of the symbolic feedback from the extra turns it allows; a baseline that receives only a generic
rejection would answer this. Finally, failures are assumed to be detected and observed correctly, recovery quality
beyond validity is not measured, and the results reflect one reasoning effort, one turn limit, and the model versions
available at the time of the runs.

Future work will therefore validate recoveries on hardware, apply the framework to other cells and unseen failures,
add a cost criterion for choosing among valid recoveries, and make composition reliable for new uses of resources.

### 13.4 Added: predefined safety specifications and industrial deployment

Place both after the generality paragraph and before the limitations.

**Points: predefined safety specifications**

- The safety guarantee is relative to Φ. The CCA can only catch hazards that some specification describes.
- Recovery makes gaps more likely than nominal operation, because it uses resources in places and roles the
  specifications were not written for (for example, a mobile manipulator handing a part to a robot at a staging
  tray).
- Partly covered elsewhere: the physical RA check and composition's motion planning catch collisions with the
  recorded cell regardless of Φ.
- Not covered: hazards that are not geometric, such as a new shared area that needs mutual exclusion, a new ordering
  constraint, or people.
- Completeness of Φ is an assumption. Directions to reduce it:
  - derive Φ from the cell's hazard analysis;
  - write general specifications that cover every shared area, not only those nominal production uses;
  - let the LLM propose candidate specifications for a new interaction, enforced only after an engineer approves
    them (the LLM proposes, a person decides, as with recovery events).

**Points: industrial deployment**

- Verdict: not ready for autonomous use; well suited to operator-supported recovery.
- Strengths for industry:
  - readable task-level outlines with the checks each event passed;
  - every proposal and refusal recorded, giving traceability;
  - safe failure: no plan, handed to the operator.
- Open issues:
  1. It plans above the safety-rated layer and does not replace it. Certified safety functions (ISO 10218,
     ISO 13849) still bound any accepted plan; the models and checks would need verification.
  2. Model fidelity and upkeep: resource models, geometry, primitives and Φ must track the physical cell, which
     takes engineering effort.
  3. Latency: minutes for synthesis and longer for composition. This suits disruptions that would otherwise stop
     the cell for longer, not responses needed within seconds.
  4. LLM service:
     - plant data go to an external provider;
     - model versions change without notice;
     - an on-premise model may be too weak for composition, the harder stage (Section 6.3.2).
  5. Failure detection and diagnosis are assumed; a plant needs integration with monitoring, PLCs and the MES.

**Draft text**

The framework's safety guarantee is relative to the specifications it is given. The CCA checks recovery events
against the predefined LTLf specifications Φ (Table 3), so a hazard that no specification describes cannot be
detected. Recovery makes such gaps more likely than nominal operation, because it deliberately uses resources in
places and roles the specifications were not written for, such as a mobile manipulator handing a part to a robot at
a staging tray. Part of this gap is covered elsewhere: the physical RA check and motion planning during composition
detect collisions with the recorded cell regardless of Φ. Hazards that are not geometric, such as a new shared area
that needs mutual exclusion, a new ordering constraint between parts, or the presence of people, still depend on the
specifications being complete. Completeness of Φ is therefore an assumption of this work. It could be reduced by
deriving the specifications systematically from the cell's hazard analysis, by writing general specifications that
cover every shared area rather than only those used in nominal production, and by letting the LLM propose candidate
specifications when a recovery creates a new interaction, to be enforced only after an engineer approves them.

The framework is not yet ready to run autonomously in an industrial cell, but its structure suits
operator-supported recovery. Each recovery outline is a readable sequence of task-level events with the checks each
event passed, and every proposal and refusal is recorded, so an operator can review a recovery before it runs and
trace afterwards why it was accepted. When the framework finds no valid recovery, it produces no plan and hands the
disruption to the operator. Several issues remain before wider use. First, the framework plans above the cell's
safety-rated layer and does not replace it: safety functions certified under standards such as ISO 10218 and
ISO 13849 must still bound what any accepted plan can do, and the models and checks themselves would need
verification. Second, the checks are only as good as the models, so the resource models, geometry, primitive
declarations and specifications must be built and kept consistent with the physical cell as it changes. Third,
planning takes minutes for task synthesis and longer for primitive composition, which suits disruptions that would
otherwise stop the cell for longer, but not those that need a response within seconds. Fourth, the framework depends
on an LLM service: plant data are sent to an external provider, and model versions can change without notice. An
on-premise model would avoid both, but Section 6.3.2 suggests that a less capable model may not compose primitives
reliably. Finally, the framework assumes that failures are detected and diagnosed, which in a plant requires
integration with monitoring and with existing controllers such as PLCs and manufacturing execution systems.

### 13.5 Added: feasibility is bounded by the declared primitives

Place right after the safety-specification paragraph. The two form a pair: safety holds only within Φ, and
feasibility only within the primitive sets 𝒫_i (assumption A.2: finite parameterized primitives with known
preconditions, constraints and effects).

**Points**

- The framework finds behaviour outside the plant model, not outside the primitive model. A recovery that needs a
  skill no resource has (for example, pushing a part, or emptying a station that declares no primitives) cannot be
  produced.
- Seen in the results:
  - In 9 of the 12 part slippage EL attempts that ended without an outline, the refusals were about Exit, which
    declares no primitives ("no Exit primitive sets Exit's resource_state to empty", "Exit declares no
    primitives").
  - PS_terra_3's composition stopped when the LLM reported that no visible primitive could set the stated
    resource_location.
- The cell's primitive set is small, which keeps choices, prompts and lookups manageable. Scaling to resources with
  many primitives is untested. Lookups per composition are already 55–132, mostly primitive contracts and
  decompositions.
- Future work, in order of difficulty:
  1. **Scaling study:** resources with many primitives (for example, dual-arm robots or tool changers), measuring
     how synthesis and composition behave as the library grows.
  2. **Composite functions:** store compositions that passed the checks as new reusable functions. Their contracts
     follow from the primitives they use, so the RA can still check them, and reuse would also shorten composition
     time. This is the natural next step.
  3. **New primitives:** learned from demonstration or generated as code, to cover skills the library lacks. Each
     needs a verified contract before the RA can check it, which is an open problem.

**Draft text**

Feasibility is bounded in the same way by the declared primitives. Each resource offers a finite set of
parameterized primitives with known preconditions and effects (Table 2), and an event is feasible only if some
composition of them realizes it (Eq. 7). The framework can therefore find behaviour outside the plant model, but not
outside the primitive model: a recovery that needs a skill no resource has, such as pushing a part or emptying a
station that declares no primitives, cannot be produced. This bound appeared in the results. In part slippage, most
event-level attempts that ended without an outline kept proposing events that needed Exit to be empty, although Exit
declares no primitive that could empty it, and one composition stopped when the LLM reported that no primitive could
reach the state it had to produce. The cell's primitive sets are also small, which keeps the choices, the prompt and
the lookups manageable; how synthesis and composition scale to resources with many primitives remains to be tested.
Two extensions could widen the bound. Compositions that pass the checks could be stored as new composite functions,
whose contracts follow from the primitives they use, so that later recoveries can reuse them and compose faster.
Genuinely new primitives, learned from demonstration or generated as code, could cover skills the library lacks, but
each would need a verified contract before the RA could check it, which remains an open problem.

### 13.6 Framing the limitations, and the final order of 6.3.3

**Is the paper worth it? Yes.** A long limitations list mostly reflects how carefully the paper was reviewed.

- **The main claim is strongly supported.** Validation, not knowledge of the rules, makes LLM recovery outlines
  valid: 91% against 29% against 4%, across 720 attempts. It holds for three LLMs from two providers, in all four
  failure types and at both generation levels, with p < 10⁻²².
- **The design is rigorous:**
  - models chosen by criteria set in advance;
  - equal runs per cell;
  - a baseline that gets the same prompt as the framework;
  - every started attempt reported;
  - a fixed turn limit.
- **It goes beyond planning on paper.** Outlines were turned into primitive programs checked by motion planning, and
  four ran in Gazebo.
- **The results contain insights, not just a success rate:**
  - knowing the rules is not the same as checking them;
  - the requirements complement each other;
  - the framework fails safe;
  - the hard part moves to composition.

**Most limitations are scope, not flaws.**

- *Scope limits shared by almost every paper in this area:* simulation only, one cell, safety relative to Φ,
  feasibility relative to the primitives. Any formal-verification approach is only as good as its models; reviewers
  expect authors to say so, not to have solved it.
- *Limitations that touch the main claim (only two):*
  - The framework is judged by its own checks. The baselines, composition and Gazebo already answer this partly.
  - No feedback-only baseline. One future experiment, not a hole in the results.

**How to present them.**

- Keep 4–5 limitations with substance:
  1. simulation only;
  2. one cell and four authored scenarios, with the prompt developed on them;
  3. judged by its own checks (partly answered by composition and Gazebo);
  4. no feedback-only baseline.

  Merge the rest into one sentence: failure detection assumed, recovery quality not measured, one reasoning effort,
  one turn limit, the model versions of the time, and the composition sample (shortest outlines, 3 per cell).
- Turn each limitation into a direction: hardware validation, other cells and unseen failures, a feedback-only
  baseline, composite functions, and specifications approved by an engineer.
- Keep the safety-specification (13.4) and primitive (13.5) paragraphs in the Discussion, not the Limitations. They
  are insights about the approach and read as understanding, not weakness.
- State limitations plainly and briefly, without apologising. Reviewers reject overclaiming far more often than a
  clearly stated scope.

**Final order of 6.3.3**

1. Verification belongs to symbolic methods; specifications belong in the model (13.3, paragraph 1).
2. The requirements complement each other; the framework fails safe (13.3, paragraph 2).
3. Where the difficulty lies now: composition, generation level, model choice (13.3, paragraph 3).
4. Valid is not optimal; generality (13.3, paragraph 4; compress if 6.3.3 is too long).
5. Safety is bounded by Φ (13.4, first paragraph).
6. Feasibility is bounded by the declared primitives (13.5).
7. Industrial deployment (13.4, second paragraph).
8. Limitations: the 4–5 above, the rest in one sentence. Drop "failures are assumed to be detected and observed
   correctly" here, since paragraph 7 already says it.
9. Future work, one sentence:

> Future work will therefore validate recoveries on hardware, apply the framework to other cells and unseen
> failures, add a cost criterion for choosing among valid recoveries, make composition reliable for new uses of
> resources, and extend the primitive library with composite functions and new primitives.

Expected length: about 1,500 words. If that is too long, compress paragraph 4 and the deployment paragraph first.
