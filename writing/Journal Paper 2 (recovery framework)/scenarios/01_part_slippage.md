# Part slippage

**Scenario key:** `Part slippage` · **Failed resource:** `ur5e-4` · **Driving specifications:**
1 (board mutex), 5a (gear_small before KET4)

## Setup

- **Assembly cell:** `ur5e-3` (base at (0, 0.5, 1.021) m) and `ur5e-4` (base at (0, −0.5, 1.021) m)
  share the assembly table and `assembly_board-v1`.
- **Initial condition:** `KET4_Square_4mm`, already trimmed (`trim → square`), stands upright in
  zone 4 of `Buffer For Machined parts`.
- **Nominal step:**
  - `ur5e-3` picks KET4 and `ur5e-4` picks `gear_small`; both pickups are observed.
  - `ur5e-4` then inserts gear_small at `Gear_Plate/Gear_Shaft_1` (`place_insert`).

## Failure

- At 50% of the place lowering, gear_small slips from `ur5e-4`'s gripper. It comes to rest upright
  on the table in `ur5e-3`'s region, observed at (0.25, 0.45, 1.025) m.
- `ur5e-4` reports the fault itself through its gripper's object detection, and stops with its tool
  at the board.
- `ur5e-3` still holds KET4.

The recovery locates the gear from its observed pose, not from a configured target.

## Who can act

Each UR5e declares a top-down gripper reach for picking a loose part:
- It is computed from the UR5e kinematic parameters and the RG2 tool offset of 0.218 m.
- The radius is 0.886 m around the robot base, the smallest reach over the height band in which
  loose parts rest.

| Robot | Distance to the gear | Within reach |
| --- | --- | --- |
| `ur5e-3` | 0.255 m | yes |
| `ur5e-4` | 0.982 m | no |

Offline inverse kinematics on the captured robot model agrees: a top-down grasp has a solution for
`ur5e-3` and none for `ur5e-4`.

Only `ur5e-3` can retrieve the gear, and its gripper is occupied by KET4.

## Specifications the recovery must respect

```
(1)  G ¬( in(r, assembly_board-v1) ∧ in(r', assembly_board-v1) )
     At the failure, in(ur5e-4, assembly_board-v1) holds.
     → ur5e-3 cannot work at the board until ur5e-4 has retracted.

(5a) a = enters(KET4_Square_4mm, assembly_board-v1)
     b = assembled(gear_small @ Gear_Plate/Gear_Shaft_1)
     (¬a U (b ∧ ¬a)) ∨ G ¬a
     → ur5e-3 must not insert the KET4 it holds. It has to set KET4 down before it can pick up
       the gear.
```

Both specifications constrain the one robot able to recover the gear:
- the board mutex makes it wait for `ur5e-4`;
- 5a forbids the shortcut of inserting KET4 first.

**In force but satisfied:** 4 (KET4 is already trimmed) and 5b. Specifications 2, 3 and 6 concern
the machining area, which this recovery does not touch.

## Intended recovery

This is an evaluation hypothesis; other validated sequences may succeed.

1. `ur5e-4` moves home, leaving the board.
2. `ur5e-3` places KET4 in `Buffer For Machined parts`.
3. `ur5e-3` picks gear_small from its observed pose and inserts it at `Gear_Plate/Gear_Shaft_1`.
4. Nominal handoff: `ur5e-3` inserts KET4.

## What this scenario tests

- Grounding a lost part from observation rather than from the plan.
- Choosing a robot by reachability: the nearby robot can reach the part, and the robot that dropped
  it cannot.
- Two specifications acting on the same robot, which forces a detour (setting KET4 down) before the
  direct action.

## Open items (internal — not for the paper)

- **Capture:** `capture_77666affc1b9468ca157cf366d706dc8` (development, schema 2, 2026-10-15). It
  supersedes `capture_7dda4f66…`.
- **Drop spot:** moved to (0.25, 0.45, 1.04) on 2026-10-15. At the earlier spot (0, 0.30),
  `ur5e-4` could still reach the gear.
- **Height band in the robot's own pose check:** the band is −0.05 to +0.10 m above the robot
  base. The robot's runtime pose check applies the same band to approach and retreat poses, so a
  pose above about 1.13 m may be rejected. Unverified.
- **Validators:** bypassed for now.
