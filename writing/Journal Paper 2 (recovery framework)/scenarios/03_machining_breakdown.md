# Machining breakdown during part processing

**Scenario key:** `Machining breakdown during part processing` · **Failed resource:** `M1` ·
**Driving specifications:** 6 (KET4 trim before RGOCG4 trim) and 4 (trim before board entry),
supported by 2 (M1 mutex) and 3 (M2 mutex)

## Setup

- **Order:** `assembly_board-v1-machine-breakdown.json`.

  | Part | Process plan |
  | --- | --- |
  | `KET4_Square_4mm` | `trim → square`, then `assembly` |
  | `gear_small` | `print_part`, then `assembly` |
  | `RGOCG4-50_Round_4mm` | `trim → circle`, then `assembly` |

- **Machines run at the same time:**
  - M1, served by `ur5e-1`, runs the square program on KET4 (60 s trim).
  - M2, served by `ur5e-2`, runs the circle program on RGOCG4 (90 s trim).
  - gear_small is already assembled.
  - The start is staged: the capture harness starts both trims together once both machines are
    loaded. Because M1's trim is shorter, the nominal schedule trims KET4 first, as rule 6 requires.
- **Programs:** both machines support `trim` with the results `square` and `circle`.

## Failure

- M1 fails at a random point while both machines are processing. The point is drawn uniformly over
  the overlap from a recorded seed (`4000168541`); it is not fixed at 50%.
  - With this seed, M1 fails about 4 s into the trims. KET4 is about 7% trimmed and RGOCG4 about 5%.
- KET4 is left partly trimmed in M1. RGOCG4 is partway through its trim in M2, with more than 30 s
  of it left.

## Who can act

- **M2** is the only working machine. It supports the square program, and its job control decides
  how its running job can be interrupted:

  | Action | Effect |
  | --- | --- |
  | `pause_job` | A paused job resumes after the part is reloaded (`resume_after_reload: true`). |
  | `stop_job` | A stopped job cannot be restarted (`restart_stopped_jobs: false`). |

  `resume_job` is also available.
- **Robots:**
  - `ur5e-1` serves M1, and `ur5e-2` serves M2.
  - The KMR has docking poses at M1 (−4.7, 2.2) and M2 (−2.7, 2.2), and declared access to both
    workholdings.
- **M2 staging tray:** at (−2.6, 1.1, 1.0) m, capacity 1.
- **No conveyor route:** the conveyor runs only toward the assembly cell, so there is none from M1
  to M2.

## Specifications the recovery must respect

```
(6) a = trimmed(RGOCG4-50_Round_4mm, circle)
    b = trimmed(KET4_Square_4mm, square)
    (¬a U (b ∧ ¬a)) ∨ G ¬a
    → M2, the only working machine, must not finish RGOCG4 first. It must interrupt RGOCG4,
      trim KET4, and only then complete RGOCG4.

(4) a = enters(KET4_Square_4mm, assembly_board-v1)
    b = trimmed(KET4_Square_4mm, square)
    (¬a U (b ∧ ¬a)) ∨ G ¬a
    → the partly trimmed KET4 cannot skip its remaining trim.
```

- **The two rules together:**
  - Without rule 4, a recovery that never completes either trim would satisfy rule 6 trivially.
  - Rule 4 rules that out, so KET4 must be trimmed on M2.
  - Rule 6 then forces M2 to stop RGOCG4 partway, using the resumable pause and not the
    irreversible stop.
- **Supporting:**
  - 2: `ur5e-1` and the KMR must not both be in M1 while KET4 is extracted.
  - 3: `ur5e-2` and the KMR must not both be in M2 while the parts are swapped.
- **In force but satisfied:**
  - 1: the board is not contended at the failure.
  - 5a: gear_small is already assembled.
  - 5b: it follows from rule 6 in the intended order.

## Intended recovery

This is an evaluation hypothesis; other validated sequences may succeed.

1. M2 pauses the RGOCG4 job (`pause_job`, not `stop_job`).
2. `ur5e-2` unloads RGOCG4 to the M2 staging tray.
3. `ur5e-1` or the KMR extracts KET4 from M1, and the KMR carries it to M2. The staging tray
   already holds RGOCG4, so KET4 must be handed over directly or sequenced.
4. `ur5e-2` loads KET4. M2 selects the square program and completes the trim.
5. `ur5e-2` unloads KET4, and the KMR carries it to `Buffer For Machined parts`.
6. `ur5e-2` reloads RGOCG4. M2 selects the circle program, resumes and completes the trim.
7. Assembly continues in the order KET4, then RGOCG4. Nominal handoff.

## What this scenario tests

- Temporal reasoning over machine schedules: interrupting a healthy machine because of an
  ordering rule.
- The difference between an operation that can be resumed (pause) and one that cannot (stop).
- Spatial exclusion while parts are swapped between two machines and a mobile manipulator.

This is the scenario with the most driving specifications.

## Where the LLM is essential

Rules 4 and 6 together leave no recovery inside the nominal task graph:
- Letting M2 finish RGOCG4 and then trimming KET4 trims RGOCG4 first, which breaks rule 6.
- Abandoning KET4 means it can never enter the board (rule 4). Rule 6 then forbids ever trimming
  RGOCG4, so the order cannot finish.

Only interrupting a healthy machine works, and that needs events the nominal plan never contains:
1. M2 pauses its RGOCG4 job. A healthy job is interrupted because of a temporal rule.
2. `ur5e-2` unloads the partly trimmed RGOCG4 to the M2 staging tray.
3. `ur5e-1` or the KMR takes KET4 out of the failed M1 (a pickup from an interrupted job).
4. The KMR carries KET4 from M1 to M2. No nominal route exists, because the conveyor only goes to
   assembly.
5. M2 switches to the square program and trims KET4. Nominally, M2 runs only circle.
6. M2 switches back to circle, RGOCG4 is reloaded, and its job resumes.

These events create states the nominal plan never reaches:
- M2 holds a paused job while another part is loaded;
- RGOCG4 sits partly trimmed on a tray;
- KET4 is at M2.

**Pause versus stop is a reasoning trap.** A stopped job cannot be restarted. Stopping RGOCG4 would
leave it untrimmable, and rule 4 would then keep it off the board for good. The machine's declared
job control allows exactly the intended sequence: pause, change the program, trim KET4, change the
program back, then resume after reloading.

## Open items (internal — not for the paper)

- **Capture:** the official failure capture `capture_8beafdb8a01e444894cfaf41d29921fd` (schema 1)
  is retained. It used 5 s trims, which tie on rule 6 nominally and leave M2 only about 3 s at the
  fault.
- **Schema-2 capture:** `capture_142d8c7d69d74e39aade558dcbc4b15e` (run `machining_02` of
  `recapture_v3_20261022`, 2026-10-23, seed `4000168541`, on the corrected resource models). It is
  verified offline and is the Recovery UI default.
  - M1 fails with KET4 at 7% (4.2 of 60 s) and RGOCG4 at 4% (4.0 of 90 s). At the frozen
    checkpoint, KET4's job is interrupted in M1 and RGOCG4 is running in M2 at 5% (4.7 of 90 s).
  - M1's state is the observed fault state `interrupted`. The failed task is the pending observed
    `complete_job`; `machine_part` no longer exists.
  - The KMR is idle at M2, and ur5e-1 to ur5e-4 are at home.
  - It supersedes `capture_4d1e4cc0…` (which superseded `capture_8beafdb8…`), archived on
    2026-10-23 to `/home/jongh/recovery_archives/20261023_pre_model_correction_captures/`.
  - M1's recorded `job_state` field still reads `running` (its value before the fault), while its
    job record says `interrupted`. The turn-1 grounding states the interruption correctly.
  - The 60/90 s trims are set only in the capture's private scene
    (`evaluation_conditions.condition_setup`, user decision 2026-10-16).
- **Earlier LLM draft:** it proposed essentially this order with gaps, and was never validated.
  See IMPLEMENTATION_PLAN.md.
- **Left to the LLM:**
  - whether `ur5e-1` or the KMR extracts from the failed M1;
  - how KET4 is handed over at M2 while its staging tray is occupied.
