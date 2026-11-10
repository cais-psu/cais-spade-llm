# Gazebo videos of the four failure scenarios

One folder per scenario, each with the failure and the recovery from the same failure capture:

| File | Content |
|---|---|
| `failure.mp4`, `failure_20x.mp4` | Nominal production until the failure, recorded from the capture's setup |
| `recovery.mp4`, `recovery_20x.mp4` | The recovery composition executed from the failure capture |

Only the 20x copies and this file are in git. The full-length `failure.mp4` and `recovery.mp4` stay on disk
(`.gitignore` in this folder); the originals are also in `cais_spade_llm/monitor/environment_runs/recovery/gazebo_runs/runs/`.

## Scenarios

The recoveries are the Study 2 video picks: per scenario, the passed Study 1 Framework outline with the fewest
events across the three LLMs, composed in Study 2 round 9 (the same set as `appendix/appendix_set.json` and the
Recovery UI).

| Folder | Scenario | LLM | Outline | Composition | Events run |
|---|---|---|---|---|---|
| `part_slippage` | Part slippage | Claude Opus 5.5 | `draft_fa0c3793…` | `composition_096a6822…` | 4/4 |
| `ur5e1_breakdown` | ur5e-1 breakdown | GPT-6.1 Sol | `draft_6f500c9f…` | `composition_3075cbc4…` | 4/4 |
| `conveyor_breakdown` | Conveyor breakdown | Claude Opus 5.5 | `draft_71aa4a94…` | `composition_9e6b4263…` | 7/7 |
| `machining_breakdown` | Machining breakdown during part processing | GPT-6.1 Sol | `draft_91d5a2aa…` | `composition_040388ef…` | 14/14 |

| Folder | Failure capture | Camera | Failure (full / 20x) | Recovery (full / 20x) | Recorded (failure / recovery) | Source runs |
|---|---|---|---|---|---|---|
| `part_slippage` | `capture_6a4195fe…` | assembly close-up | 5:00 / 0:15 | 7:02 / 0:21 | 2026-11-07 / 2026-11-09 | `fail_v8b_part_slippage`, `final_part_slippage` |
| `ur5e1_breakdown` | `capture_4f9097aa…` | whole cell | 39:19 / 1:58 | 52:15 / 2:37 | 2026-11-06 / 2026-11-09 | `fail_v7b_ur5e1`, `final_ur5e1` |
| `conveyor_breakdown` | `capture_460d78e9…` | whole cell | 14:05 / 0:42 | 30:37 / 1:32 | 2026-11-06 / 2026-11-09 | `fail_v7b_conveyor`, `final_conveyor` |
| `machining_breakdown` | `capture_6cd8cd69…` | whole cell | 30:01 / 1:30 | 77:15 / 3:52 | 2026-11-06 / 2026-11-09 | `fail_v7b_machining`, `final_machining_r2` |

## Notes

- The outlines and compositions were checked by RA and CCA before execution. In Gazebo the composition is replayed
  with the CCA bypassed, as for the failure captures, so these videos show execution, not live CCA gating.
- Machining's recovery is the second take. The first stopped at event 3 when Nav2's path follower could not drive the
  KMR from M2's stop to M1's; the KMR base controller was fixed in `dab1f1e` (IMPLEMENTATION_PLAN.md, November 9).
- Part slippage's failure run report says `failed`: the runner's check after recording found that the dropped gear
  had not yet settled. The video shows the slip and was kept (GAZEBO_QUEUE item 119).
- Gazebo ran far below real time (Machining's recovery at about 7%), so the full-length videos are slow; the 20x
  copies are the ones to show.
- `nominal_production.mp4` (1:04, 20x): nominal production with all four UR robots and the KMR, recorded
  2026-09-29, before the staging trays moved 0.30 m north on 2026-11-05. It is a copy of
  `cais_spade_llm/monitor/recovery_gazebo_runs/attempt-f2090c8976d84acd90cf35353c5acc34/assembly-20x.mp4`; the
  full-length `assembly.mp4` and a 10x copy are in that folder. The opening minutes of each `failure.mp4` show
  nominal production in the current cell. `evidence/2026-09-28-gazebo-11-parts/` holds an older run in which only
  ur5e-1 and the KMR are present.
