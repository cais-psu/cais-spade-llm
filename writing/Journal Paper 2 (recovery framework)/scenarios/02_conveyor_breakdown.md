# Conveyor breakdown

**Scenario key:** `Conveyor breakdown` · **Failed resource:** `Conveyor` · **Specifications:** 1
(board mutex) in force, none contended

## Setup

- **Parts in play:** `KET4_Square_4mm` (trimmed on M1) and `gear_small`. M2 and `ur5e-2` are not
  permitted in this scenario.
- **Conveyor:** a 4.2 m conveyor along y = 0.5 m. It runs from M1's loading pose (−4.05, 0.5) to an
  output nest at (−0.78, 0.5) in the assembly cell, which `ur5e-3` serves. Its normal output goes
  to `Buffer For Machined parts`.
- **At the assembly cell:** `ur5e-4` has already inserted `gear_small` at
  `Gear_Plate/Gear_Shaft_1`. On its own schedule the gear reaches the board long before `ur5e-1`
  picks KET4 from M1. `ur5e-3` and `ur5e-4` are at home.

## Failure

After `ur5e-1` has picked the trimmed KET4 from M1, and before it releases KET4 onto the conveyor,
the conveyor stops.

## Who can act

- **KMR:** the KUKA KMR iiwa (mobile base, LBR iiwa 14 R820 arm, RG2 gripper, 0.82 m arm reach). It
  is the only remaining transporter between the machining area and the assembly cell.
- **M1 staging tray:** at (−4.6, 1.1, 1.0) m, capacity 1. It is the handoff point between `ur5e-1`
  and the KMR.
- **Assembly robots:** `ur5e-3` and `ur5e-4` are both free; gear_small is already on the board.

## Specifications the recovery must respect

```
(1) G ¬( in(r, assembly_board-v1) ∧ in(r', assembly_board-v1) )
    → in force, but only ur5e-3's KET4 insertion uses the board, so the intended recovery
      never contends for it.
```

**Why no specification drives this scenario:**
- **5a (gear_small before KET4):** gear_small is already assembled before the failure.
- **2 (M1 mutex):** the staging tray is about 1 m from M1 (0.86 m from its workholding), so the
  handoff between `ur5e-1` and the KMR takes place outside M1's protected areas.
- **4 (trim before board entry):** KET4 is already trimmed.

Conveyor breakdown therefore exercises capability substitution, not safety reasoning. It is the
scenario with the fewest specifications in play.

## Intended recovery

This is an evaluation hypothesis; other validated sequences may succeed.

1. `ur5e-1` places KET4 on the M1 staging tray.
2. The KMR collects KET4 and carries it to `Buffer For Machined parts`.
3. `ur5e-3` retrieves KET4 from the buffer and inserts it.
4. Nominal handoff.

## What this scenario tests

- Substituting a failed transport capability (conveyor → mobile manipulator).
- A multi-robot handoff through a capacity-1 staging tray.

## Open items (internal — not for the paper)

- **Capture:** `capture_6d907d9ba40a4bb099cba8804166caf0` (development, schema 2, 2026-10-22, on
  the corrected resource models) is the Recovery UI default. At the failure, `ur5e-1` holds the
  trimmed KET4 at M1, gear_small is assembled, `ur5e-3`/`ur5e-4` are home, and the KMR is idle at
  Storage. It supersedes `capture_3aed6165…`, archived on 2026-10-23 to
  `/home/jongh/recovery_archives/20261023_pre_model_correction_captures/`. The schema-1
  `capture_1b3f3d1d729a4701b94cff514c783130`, recorded with the old gear hold, stays as a
  regression fixture.
- **Design decisions (user):**
  - 2026-10-15: keep the scenario as it is and document its specifications. It previously used
    `KET4_Square_4mm_before_gear_small`, which made `ur5e-4` wait while holding the gear; the
    plant-wide gear-first order removed that tension.
  - 2026-10-16: drop the harness hold that kept `ur5e-4` holding gear_small until the fault. Under
    rule 5a no rule explains that wait. Without it the gear is assembled before the failure, so
    the board mutex is no longer contended (see IMPLEMENTATION_PLAN.md, "Conveyor breakdown
    (Stage 3) setup without a held peer").
- **Staging-tray distance:** taken from the configured poses; still to be confirmed against M1's
  region geometry.
