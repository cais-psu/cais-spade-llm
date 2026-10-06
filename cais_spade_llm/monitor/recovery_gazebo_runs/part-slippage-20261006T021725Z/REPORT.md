# Three mock part_slippage cases: Gazebo preflight

**Live acceptance is incomplete. No recovery was executed and no videos were published.**

On October 5, 2026 (America/New_York), a dedicated Gazebo session launched with
the installed observation service after `make bootstrap-gazebo` completed. All
four native UR5e controllers reported their services ready. The explicitly
selected predefined document was compiled without changing the Safety-page
selection or approvals. `diagnostic_cca_bypass` remained false.

The initial stamped physics snapshot contained **36 models and zero attachments**.
The final captures each include **12 configured resource observations, 12 resource
envelopes, and 12 part envelopes**. The empty `assembly_board_v1` carrier obtains
its envelope from the explicitly configured, observed static
`GMC_Laser_Plate_Virtual` and `Gear_Plate` models. This resolves that geometry gap;
it does not establish a complete admissible checkpoint or stationary future motion.

| Requested case | Actual result | Accepted trial |
| --- | --- | --- |
| [Mutex](preflight-final/mutex.json) | `NEEDS_CONTEXT` during live preflight | No |
| [Precedence](preflight-final/precedence.json) | `NEEDS_CONTEXT` during live preflight | No |
| [Safe](preflight-final/safe.json) | `NEEDS_CONTEXT` during live preflight | No |

The reported conditions are missing evidence, **not specification violations**.
There is no CCA allow/block decision or physical AP counterexample in these reports.
No PA → RA → CCA execution trial was performed. No supplied event or primitive
was dispatched, no assembly acknowledgement was fabricated, and no ledger or
pending manufacturing task was completed. The requested post-slippage checkpoint
was not staged: the observed scene remains the initialization scene.

Actual capture blockers:

- The four UR5e resources report link motion above their configured idle limits.
- Controller goal-status observations are unavailable; an unpublished initial
  status is not treated as proof of no active goals.
- The first final capture also exceeded its two-second capture window and
  reported a stale resource observation. The other two remained within the window.
- CCA has no registered `recovery_composition_context_provider`, and the two
  recovery resources have no registered execution-evidence providers.

The current live preparation/model path also supports only empty-custody motion:
complete native grasp/release preparation, carried-part continuous observations,
prepared-command execution binding and validated execution feedback remain required.
The separate companions retain the original seven event identities and 19 primitive
references but do not contain resolved native programs or prepared trajectories.
The original synthetic trajectories and one-second durations were not substituted
for live evidence. Gear assembly completion still needs actual acknowledgement
and observed effects before KET4 entry can be permitted.

The WSL launch shell initially had no `DISPLAY`, so the launched viewer exited.
Opening the viewer on the verified WSLg display `:0` succeeded; the existing
Gazebo-window capture produced [this diagnostic frame](gazebo-preflight.png).
No frame sequence or MP4 was recorded. This task's Gazebo launch and viewer were
stopped after capture; historical recordings were preserved.

[summary.json](summary.json) contains run/launch/checkpoint identities and final
report hashes. [startup-physics.json](startup-physics.json),
[gazebo-launch.log](gazebo-launch.log), [preflight-final.log](preflight-final.log),
and [bootstrap.log](bootstrap.log) retain the observations and execution logs.
The earlier [preflight](preflight/summary.json) is retained separately.

The required `part_slippage-mutex-20x.mp4`, `part_slippage-precedence-20x.mp4`, and
`part_slippage-safe-20x.mp4` remain unproduced. Publish them together only after the
actual trial outcomes and the existing silent 20× export/decoding checks pass.

Verification: **611 tests passed** across the preparation/UI (65 + 1), safety,
composition and admission (504), and recording (41) suites. The initial combined
preparation run was interrupted at its UI test; the isolated UI test also timed
out inside the restricted sandbox and passed outside it. Those partial and
repeat runs are not added to the total. [verification.json](verification.json)
retains commands, timings and logs.

`poetry check` passed with existing metadata warnings. Compilation, both CLI help
checks, focused Ruff, JSON/fingerprint checks, 64 local references and
`git diff --check` passed. All 26 roadmap headings remain. The protected Safety-page
files, `cca.json`, `SystemBridge`, original fixtures, historical recordings and
unrelated pre-existing changes remain unchanged. Runtime budgets remain
20,000 states / 2 seconds. No LLM call was made.
