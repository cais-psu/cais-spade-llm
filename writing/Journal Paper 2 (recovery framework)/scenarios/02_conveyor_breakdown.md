# Conveyor breakdown

**Scenario key:** `Conveyor breakdown` · **Failed resource:** `Conveyor` · **Driving
specification:** 1 (board mutex)

## Setup

- **Parts in play:** `KET4_Square_4mm` (trimmed on M1) and `gear_small`. M2 and `ur5e-2` are not
  permitted in this scenario.
- **Conveyor:** a 4.2 m conveyor along y = 0.5 m. It runs from M1's loading pose (−4.05, 0.5) to an
  output nest at (−0.78, 0.5) in the assembly cell, which `ur5e-3` serves. Its normal output goes
  to `Buffer For Machined parts`.
- **At the assembly cell:** `ur5e-4` holds `gear_small`.

## Failure

After `ur5e-1` has picked the trimmed KET4 from M1, and before it releases KET4 onto the conveyor,
the conveyor stops.

## Who can act

- **KMR:** the KUKA KMR iiwa (mobile base, LBR iiwa 14 R820 arm, RG2 gripper, 0.82 m arm reach). It
  is the only remaining transporter between the machining area and the assembly cell.
- **M1 staging tray:** at (−4.6, 1.1, 1.0) m, capacity 1. It is the handoff point between `ur5e-1`
  and the KMR.
- **Assembly robots:** `ur5e-4` holds gear_small, and `ur5e-3` is free.

## Specifications the recovery must respect

```
(1) G ¬( in(r, assembly_board-v1) ∧ in(r', assembly_board-v1) )
    → ur5e-4 inserting gear_small and ur5e-3 inserting KET4 must not overlap at the board.
```

**Why no other specification drives this scenario:**
- **5a (gear_small before KET4):** gear_small is already in `ur5e-4`'s gripper, and KET4 arrives
  late by KMR, so gear-first holds by timing alone.
- **2 (M1 mutex):** the staging tray is about 1 m from M1 (0.86 m from its workholding), so the
  handoff between `ur5e-1` and the KMR takes place outside M1's protected areas.
- **4 (trim before board entry):** KET4 is already trimmed.

Conveyor breakdown therefore exercises capability substitution more than safety reasoning. It is
the scenario with the fewest driving specifications.

## Intended recovery

This is an evaluation hypothesis; other validated sequences may succeed.

1. `ur5e-1` places KET4 on the M1 staging tray.
2. The KMR collects KET4 and carries it to `Buffer For Machined parts`.
3. `ur5e-4` inserts gear_small at `Gear_Plate/Gear_Shaft_1` and retracts.
4. `ur5e-3` retrieves KET4 from the buffer and inserts it.
5. Nominal handoff.

## What this scenario tests

- Substituting a failed transport capability (conveyor → mobile manipulator).
- A multi-robot handoff through a capacity-1 staging tray.
- Sequencing two robots at a shared region.

## Open items (internal — not for the paper)

- **Capture:** `capture_1b3f3d1d729a4701b94cff514c783130` (schema 1) is retained. A schema-2
  re-capture is pending.
- **Design decision (user, 2026-10-15):** keep the scenario as it is and document that it drives
  only the board mutex.
  - It previously used `KET4_Square_4mm_before_gear_small`, which made `ur5e-4` wait while holding
    the gear.
  - The plant-wide gear-first order removed that tension.
- **Staging-tray distance:** taken from the configured poses; still to be confirmed against M1's
  region geometry.
