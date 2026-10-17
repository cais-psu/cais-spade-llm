# ur5e-1 gripper fault while unloading M1

**Scenario key:** `ur5e-1 breakdown` · **Failed resource:** `ur5e-1` (gripper only) · **Driving
specifications:** 2 (M1 mutex), 5b (KET4 before RGOCG4)

## Setup

- **Order:** the same three-part order as the Machining breakdown
  (`assembly_board-v1-machine-breakdown.json`): `KET4_Square_4mm`, `gear_small` and
  `RGOCG4-50_Round_4mm`.
- **Resources:** M1, M2, `ur5e-1`, `ur5e-2` and the KMR all take part.
- **Machines:** M1 trims KET4 (square) and M2 trims RGOCG4 (circle). gear_small is already
  assembled.

## Failure

- M1 completes KET4's square trim. `ur5e-1` reaches into M1 to unload it, approaching through the
  front of the machine to the workholding at (−4.13, 1.82, 1.085) m.
- The gripper then fails: it can neither close on the part nor detect it, and `ur5e-1` reports the
  fault itself.
- The arm remains operational; only the gripper has failed.
- At the same time, M2 is partway through trimming RGOCG4.

## Who can act

| Resource | Status |
| --- | --- |
| `ur5e-1` | Can move but cannot grasp. |
| `ur5e-2` | Its base at (−2.05, 1.1) m is 2.2 m from M1's workholding, out of reach. |
| KMR | Has a docking pose at M1 (−4.7, 2.2) from which it picks top-down from M1's workholding. It is the only possible substitute. |

Beyond its predefined routes (Storage↔M1, Storage↔M2), the KMR offers only generic abilities:
- driving its base to any planar pose (x, y, yaw);
- releasing a held part at any world pose its arm reaches.

No dock exists at the conveyor or the assembly cell. To deliver KET4 there, the LLM must choose a
base pose and a release pose itself, from the cell geometry and the KMR's footprint and arm reach.
Gazebo execution decides whether the choice works.

## Specifications the recovery must respect

```
(2)  G ¬( in(ur5e-1, M1) ∧ in(KMR, M1) )
     → ur5e-1's arm is still inside M1. It must retract before the KMR can dock and reach in.

(5b) a = enters(RGOCG4-50_Round_4mm, assembly_board-v1)
     b = assembled(KET4_Square_4mm @ GMC_Laser_Plate_Virtual/KET4_Square_4mm)
     (¬a U (b ∧ ¬a)) ∨ G ¬a
     → RGOCG4 keeps flowing from M2 and reaches the assembly cell before the delayed KET4.
       Inserting RGOCG4 first would keep the line moving but violates 5b. RGOCG4 must wait in
       the buffer.
```

**In force but satisfied:**
- 6: KET4's trim completed before the failure.
- 4: both parts are trimmed before they enter the board.
- 5a: gear_small is already assembled.
- 1 and 3: the intended recovery never puts two resources in the board area or in M2 at the same
  time.

## Intended recovery

This is an evaluation hypothesis; other validated sequences may succeed.

1. `ur5e-1` retracts out of M1 to its home pose.
2. The KMR docks at M1 and picks KET4 top-down from the workholding.
3. The KMR drives to a base pose it chooses beside the conveyor and releases KET4 onto the
   stopped belt. The conveyor then carries KET4 toward `Buffer For Machined parts`.
4. RGOCG4, trimmed on M2, does not go to the board first. The buffer is first-in-first-out by
   convention, but every zone lies within `ur5e-3`'s reach, so the LLM may reason that `ur5e-3`
   can pick KET4 out of order.
5. `ur5e-3` inserts KET4, then RGOCG4. Nominal handoff.

## What this scenario tests

- **The fault must be partial.** If `ur5e-1` were completely disabled inside M1, specification 2
  would block the KMR indefinitely and no recovery would exist. The recovery must use the impaired
  robot's remaining capability (retracting) before the substitute can act.
- **A different failure class.** It differs from the other three scenarios:

  | Scenario | What fails |
  | --- | --- |
  | Part slippage | a part is lost |
  | Machining breakdown | a machine fails |
  | Conveyor breakdown | transport fails |
  | ur5e-1 gripper fault | a robot loses one capability, and another resource must substitute for it |

- **Concurrency with nominal flow.** The recovery must hold back a part that is still flowing
  nominally (RGOCG4). This is the only scenario driven by 5b.

## Open items (internal — not for the paper)

- **Scenario key:** stays `ur5e-1 breakdown`.
- **New capture (2026-10-16):** `capture_d1ca51654ff7423193d1bd13b146964d` (development, schema 2).
  It is frozen at sim 131.78:
  - M1 has KET4's 90 s square trim completed;
  - M2 has RGOCG4 at 78% of its 60 s circle trim;
  - the KMR is idle at Storage;
  - `ur5e-1` is empty in M1, with the fault from "gripper object detection".

  The fault comes before the grasp, at the existing pre-grasp checkpoint, so the "unloading" in
  this file means `ur5e-1` has reached in but has not closed its gripper. The trim times are set
  only in the capture's private scene.
- **Old capture:** `capture_0a7fde8074b6426586b204204f2b3417` records the earlier design (fails
  before the pick, with M2 off). It stays selectable as a regression fixture; since 2026-10-16 the
  UI default is the new capture.
- **KMR status (2026-10-15):**
  - Generic driving and release are verified live (see IMPLEMENTATION_PLAN.md, Stages B–D): pick
    at Storage, carrying drive to (−3.0, −0.36, −π/2), release onto the belt at (−3.0, 0.40), and
    belt transport to the output nest.
  - A KMR pick at M1 has not yet run live.
  - The declared `KMR_access` side poses are unused by code.
  - The arm mount is marked `mount_verified: false`.
