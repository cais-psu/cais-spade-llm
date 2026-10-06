# part_slippage Gazebo companions

These companions preserve the seven event identifiers, predecessor references,
and 19 primitive references from each existing mock plan:

- [mutex.json](mutex.json): planned overlapping board occupancy; CCA must block.
- [precedence.json](precedence.json): KET4 entry before gear assembly; CCA must block.
- [safe.json](safe.json): gear assembly, board clearance, then KET4 placement;
  CCA must grant each event and observations must confirm completion.

They retain the unchanged predefined document hash and require
`diagnostic_cca_bypass=False`. The original synthetic parameters, trajectories,
one-second durations, and acknowledgement records are **not live inputs**.
`native_preparation.programs` is empty and `acceptance_complete` is false until
the native programs and their execution evidence are available. These files are
not executable recovery programs.

Each trial requires a separately staged and observed post-slippage checkpoint:
`gear_small` displaced near `ur5e-3`, `KET4_Square_4mm` in
`Buffer For Machined parts`, and both grippers empty. Staging does not demonstrate
the original drop. The KET4 trim record is a declared initial condition; gear
assembly completion must come from validated task acknowledgement and observed
effects. Position and release alone cannot establish that completion.

The read-only prerequisite runner is:

```bash
poetry run python scripts/check_part_slippage_gazebo.py --output-directory /tmp/part-slippage-live-preflight
```

Run it in a dedicated launched simulation with the ROS workspace sourced. It
constructs the configured owners and CCA with the companions' predefined document,
captures fresh physics/checkpoint evidence, and reports unavailable preparation or
admission support. It does not start SPADE dispatch behaviors, stage the scene,
execute recovery, or record. Exit code **2** means acceptance remains incomplete.
It leaves the Safety-page selection, approvals and monitor history unchanged.

The [2026-10-05 live report](../../../../cais_spade_llm/monitor/recovery_gazebo_runs/part-slippage-20261006T021725Z/REPORT.md)
records actual Gazebo captures. Those attempts are prerequisites, not accepted
mutex/precedence trials: `NEEDS_CONTEXT` is not a safety-specification violation.
No requested MP4 is published until its trial outcome and the existing exporter's
complete decoding, silent stream and 20× duration checks pass. Keep logs/AP evidence
outside the run-specific `videos-20x/` folder and preserve historical recordings.
