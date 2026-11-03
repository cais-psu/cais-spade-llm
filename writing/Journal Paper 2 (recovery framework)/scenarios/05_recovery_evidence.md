# Evidence for recovery beyond predefined behavior (Section 6.2)

Draft text for the subsection after the failure scenarios and Fig. 10. The LaTeX blocks are paper text.
Everything outside them is for the authors.

## What this evidence is

- **Source.** One accepted run per scenario: Framework, incremental, Claude Opus 5.5 medium, November 2, 2026.
  Records are listed under "Source records". These are single runs. Rates belong to Section 6.3.
- **Stage.** Offline, from the failure captures. No recovery program ran. Composition `FEASIBLE` means the declared
  composition checks only, `cca_composition_status` is `not_checked`, and a composed record cannot be dispatched yet.
- **Conveyor breakdown has no composition.** It stopped at event 3 of 9, so its evidence is at outline level only.
- **Safety table.** These four runs used the earlier board-entry form of $\varphi_5$ and $\varphi_6$. The code and the
  table now use assembly completion (user, November 2, 2026), with board entry left to the mutex $\varphi_1$. So every
  number here needs a re-run before it is reported.
- **Part slippage needs a new run.** `draft_22cb7fac…` no longer replays: event 4 inserts the gear and states no
  `processCompleted`. `composition_9275980f…` stops loading with it. With that one value stated, the outline passes RA,
  CCA and Eq. 6 and hand programs compose 5 of 5. The sentence below about the refused first candidate describes the old
  rule. In that state ur5e-4 is still on the board, so under the new rule the mutex refuses it ($\varphi_1$), not
  $\varphi_5$. Replace the paragraph from the new run.
- **The other three outlines** replay under the new rule, and the saved ur5e-1 and Machining compositions still load.
  None of them places a part on the board.
- **Part slippage's safety check** starts from a declared assumption, stated in the text below.
- **"Not predefined"** means proposed by the LLM and absent from the remaining nominal plan. The primitives
  themselves are predefined. Each event's composition is what is new.
- **Empty-recovery check.** It is not a stored record. It is `cca_nonblocking.check_completion` on the session seed
  with no events. Script and output: `/home/jongh/recovery_review_1030/precedence_trial_20261102/evidence.py`,
  `evidence.out`.

## Paper text

```latex
\subsection{Evidence of recovery beyond predefined behavior}

For each scenario we report what nominal work the failure blocks, which behavior lies outside the
predefined plan, and how feasibility, safety, and continuation were established
(Table~\ref{tab:recovery-evidence}). An event counts as not predefined when the LLM proposed it and no
task of the remaining nominal plan performs it. The primitives $\mathcal{P}_i$ are declared; the
composition $PC_k$ of each event is generated for that event.

To show that a failure lies outside predefined behavior, we applied the nonblocking check of Eq.~6 with
an empty recovery: from $x_F$, using the nominal resource models alone, can production still finish? In
all four scenarios the answer is no. Feasibility was then established per event by the RA (Eqs.~7 and
12), safety per event by the CCA (Eq.~13, including uncontrollable completions the event enables), and
continuation by the CCA when the LLM declared the recovery complete (Eq.~6).

\paragraph{Part slippage} The small gear slips from ur5e-4 during insertion and rests on the table at a
position that is no declared place. ur5e-4's approach and insertion are blocked, and with no recovery
the gear cannot reach its assembly target. Only ur5e-3 can reach the gear, but it holds the 4\,mm
square peg and has no task on the gear. The five accepted events are not in the nominal plan: ur5e-3
sets the peg down in the buffer, ur5e-4 retracts from the board, ur5e-3 picks the gear at its measured
position and inserts it, then picks the peg up again. The RA refused four candidates, among them
picks whose stated pose drove the gripper through the table. The CCA refused the LLM's first candidate,
placing the peg on the board, which breaks $\varphi_5$, and its next candidate, setting the peg aside,
was applied. The completed recovery is nonblocking over 52 states and returns ur5e-3 to its own place
approach. The check assumes ur5e-3
was clear of the board at the failure, since the capture was taken with the CCA bypassed.

\paragraph{Conveyor breakdown} The conveyor reports itself unavailable after ur5e-1 has picked the
trimmed square peg from M1. ur5e-1's place approach and release and the two conveyor advances are
blocked, and with no recovery the peg cannot leave ur5e-1's gripper toward assembly. In the nine
accepted events ur5e-1 returns the peg to M1's workholding and retreats to the M1 staging tray, and
the KMR, which has no nominal task on this peg, picks it, drives to base poses it states itself, since
no dock serves the buffer, and places it in buffer zone 1. The RA refused 17 candidates, among them a
direct hand-off, because another resource's gripper is not a recorded place, and a base pose whose
footprint overlaps the assembly station. The CCA refused the KMR's pick while ur5e-1 was inside M1
($\varphi_2$). The first declared completion was refused under Eq.~6 because the KMR stood at no
place normal production has; after it returns to Storage, the recovery is nonblocking over 616 states
and the buffer's nominal advances resume. Composition stopped at the third event, so this case is
established at outline level only.

\paragraph{ur5e-1 breakdown} ur5e-1's gripper fails while unloading the square peg from M1. Its pick,
place approach, and release are blocked, and with no recovery the peg cannot leave M1 and M1 cannot
carry on. In the eight accepted events the KMR replaces ur5e-1 and picks the peg from M1, carries it
to M2, places it on the M2 staging tray from a base pose it states between the work areas, and
ur5e-2 moves it to the conveyor. The CCA refused three KMR picks while the failed arm was inside M1
($\varphi_2$), so the arm retracts first, which its fault allows. The RA refused placing from the M2
dock, from which the tray is out of reach, and a base pose overlapping the conveyor. The first
declared completion was refused under Eq.~6 because the KMR stood at no place normal production has;
after it returns to M2, the recovery is nonblocking over 13,792 states. The conveyor's nominal advance
then takes over. The eight events compose into 26 primitives.

\paragraph{Machining breakdown} M1 fails at 6.5\% of the square peg's trim. Its job is blocked, and
with no recovery the peg cannot reach its remaining trim and assembly. M2 is healthy and is trimming the
4\,mm round peg. Letting it finish would trim the round peg first, which breaks $\varphi_7$: the CCA
refused four candidates that unloaded the square peg while M2 ran, because M2's trim could then complete
on its own. The LLM therefore interrupts a healthy machine. In the 14 accepted events M2 pauses the
round peg's job, ur5e-2 moves that peg to the staging tray, the square peg travels from M1 to M2 by
ur5e-1, the conveyor, and ur5e-2, M2 selects the square program and finishes the trim, the round peg is
reloaded, and M2 selects the circle program and resumes. No nominal task pauses a job or trims this peg
on M2. The RA refused five candidates, among them a pause that left the part's process unstated. The
first declared completion was refused under Eq.~6 for the KMR's location; the accepted recovery is
nonblocking over 52,536 states. The 14 events compose into 60 primitives.

\paragraph{Limits of this evidence} These are single runs, validated offline from failure captures.
Safety is established over $s_F s^R$; the remaining nominal events are left to the CCA's normal
admission, and Eq.~8 is not checked over $s_N$. Normal admission checks each step, with no completion
search: if a robot carrying a later part enters a free assembly board first, its assembly waits under
$\varphi_5$ or $\varphi_6$ while $\varphi_1$ keeps the robot with the earlier part out, and neither can
proceed. Compositions satisfy the declared primitive contracts, and no composed program has been
checked by the CCA or executed in simulation.
```

```latex
\begin{table}[t]
\centering
\footnotesize
\caption{Evidence per scenario, one accepted run each. Blocked: nominal tasks the failure stops.
Eq.~6 at $x_F$: the check with no recovery. Refused: candidates the RA and the CCA refused, and
declared completions refused under Eq.~6, before acceptance. States: explored by the accepted
nonblocking check.}
\label{tab:recovery-evidence}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lccccccc@{}}
\toprule
\textbf{Scenario} & \textbf{Blocked} & \textbf{Eq.~6 at $x_F$} & \textbf{Events} &
\textbf{RA} & \textbf{CCA} & \textbf{Eq.~6} & \textbf{States} \\
\midrule
Part slippage & 2 & blocking & 5 & 4 & 1 & 0 & 52 \\
Conveyor breakdown & 4 & blocking & 9 & 17 & 1 & 1 & 616 \\
ur5e-1 breakdown & 3 & blocking & 8 & 2 & 3 & 1 & 13,792 \\
Machining breakdown & 1 & blocking & 14 & 5 & 4 & 1 & 52,536 \\
\bottomrule
\end{tabular}
\end{table}
```

## Source records

| Scenario | Capture | Outline | Composition | Composed |
|---|---|---|---|---|
| Part slippage | `capture_ad64bf16…` | `draft_22cb7fac…` | `composition_9275980f…` | 5 of 5, 21 primitives |
| Conveyor breakdown | `capture_3ad495d2…` | `draft_b997e2f9…` | removed after diagnosis | stopped, 2 of 9 |
| ur5e-1 breakdown | `capture_b7dea70c…` | `draft_95f9d084…` | `composition_86b3b566…` | 8 of 8, 26 primitives |
| Machining breakdown during part processing | `capture_37e090a5…` | `draft_1a2947d9…` | `composition_02365fa8…` | 14 of 14, 60 primitives |

Empty-recovery results, from the check's own reasons:

| Scenario | Why production cannot finish from $x_F$ | States |
|---|---|---|
| Part slippage | `gear_small` cannot reach assembly at `Gear_Plate/Gear_Shaft_1` from no named place | 52 |
| Conveyor breakdown | `KET4_Square_4mm` cannot reach its assembly from `ur5e-1` | 97 |
| ur5e-1 breakdown | `KET4_Square_4mm` cannot reach its assembly from `M1`; `M1` cannot carry on | 1,436 |
| Machining breakdown | `KET4_Square_4mm` cannot reach trim → square and assembly from `M1`; `KMR` cannot carry on | 444 |
