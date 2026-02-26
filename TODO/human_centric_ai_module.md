# TODO: Human-Centric AI — Human Interaction Module

## Motivation

The current system is fully automated with no structured human-in-the-loop capability. When replanning fails (DES BFS + LLM bridge exhaust retries), the system calls `os._exit(1)`. The `User` agent (`user.py`) exists but is a passive observer — it cannot intervene, approve, or override. There is no way for a human operator to:

- Define or modify product requirements at runtime
- Review or override safety constraints before execution
- Intervene when automated recovery fails
- Monitor system status in real time through a structured interface

This module fills those gaps by building structured human interaction points into the existing SPADE multi-agent architecture.

---

## 1. Product Requirements Interface

**Goal:** Allow a human operator to define, review, and modify product assembly requirements before and during execution.

### 1.1 Structured Requirements Editor
- [ ] Define a JSON schema for product requirements (currently free-text in `specification/products/requirements/assembly_board-v1.txt`)
- [ ] Build a requirements input form/CLI that validates against the schema
- [ ] Support both natural-language and structured input modes (NL gets parsed by LLM into structured format for human review)
- [ ] Store validated requirements in `initialization/products/<product>.json` alongside existing fields

### 1.2 Requirements Review Before Planning
- [ ] After `ProcessPlanner.build_high_level()` parses NL → requirement nodes, present the parsed structure to the operator for confirmation before proceeding to task expansion
- [ ] Display: parsed requirement nodes, inferred part dependencies, assigned resources
- [ ] Allow the operator to accept, modify, or reject individual requirement nodes
- [ ] Log all human edits to `monitor/human/requirements_review.jsonl`

### 1.3 Runtime Requirements Modification
- [ ] Allow the operator to add/remove/reorder assembly steps mid-execution (triggers replan)
- [ ] Route modifications through the existing `replan_request` message flow (PA → ProcessPlanner)
- [ ] Validate modifications against safety constraints before applying

**Files to create/modify:**
- `agents/shared_information/user.py` — add `submit_requirements()`, `review_plan()` behaviours
- `agents/intelligent_product/process_planner.py` — add human-review checkpoint after `build_high_level()`
- New: `specification/schemas/product_requirements_schema.json`

---

## 2. Safety Constraints Interface

**Goal:** Allow a human operator to define, review, and manage safety constraints with full traceability.

### 2.1 Safety Rule Authoring
- [ ] Provide a guided interface for defining safety rules (currently free-text in `specification/safety/safety_requirements.txt`)
- [ ] Support rule templates for common patterns:
  - Ordering: "X must happen before Y"
  - Mutex: "A and B must not run simultaneously"
  - Exclusion: "Never do X on resource Y"
  - Conditional: "If X fails, do not attempt Y"
- [ ] Auto-generate LTLf formula preview from NL input (reuse `safety_logic.py` pipeline)
- [ ] Allow the operator to inspect the compiled DFA (visual `.dot` rendering)

### 2.2 Safety Constraint Review at Plan Validation
- [ ] When `OfflineSafetyValidator` detects a violation, present the violation trace to the operator before auto-rejecting
- [ ] Display: which safety rule was violated, the offending task sequence, the DFA state path
- [ ] Allow the operator to:
  - Confirm rejection (default)
  - Override with justification (logged and flagged)
  - Temporarily relax a constraint (with scope and expiry)
- [ ] Log all safety overrides to `monitor/human/safety_overrides.jsonl` with timestamp, operator ID, justification

### 2.3 Runtime Safety Override Protocol
- [ ] When `OnlineSafetyMonitor` blocks a task at runtime, optionally route to the operator instead of immediate block
- [ ] Configurable per-rule: `auto_block` (default) vs `human_review` escalation mode
- [ ] Timeout: if operator does not respond within configurable window, default to `block`

**Files to create/modify:**
- `agents/central_controller/central_controller_agent.py` — add human-review path in `_Monitor` behaviour for `block` decisions
- `agents/shared_information/user.py` — add `review_safety_violation()` behaviour
- `specification/safety/` — add structured safety rule format alongside existing NL text
- New: `monitor/human/safety_overrides.jsonl`

---

## 3. Human Intervention on Replanning Failure (Layer 3)

**Goal:** Replace `os._exit(1)` with a structured human escalation protocol when all automated recovery layers fail.

### 3.1 Escalation Trigger
- [ ] In `process_planner.py`, replace `os._exit(1)` (line 527) with a structured `escalation_request` message to the `User` agent
- [ ] In `product_agent.py`, when `_runtime_repair_fail_streak` exceeds `_runtime_repair_max_attempts`, send escalation instead of just logging
- [ ] Include in escalation payload:
  - Current system state snapshot (all resource states, part tracker, FSA position)
  - What was attempted (DES BFS result, LLM bridge attempts with rejection reasons)
  - The stuck state and why no path was found
  - Suggested actions (if any partial plans were found)

### 3.2 Human Decision Interface
- [ ] Present the escalation context to the operator in a structured format
- [ ] Operator action options:
  - **Manual plan**: Provide a sequence of steps to execute (validated against safety DFAs)
  - **Override and retry**: Relax a specific constraint and re-trigger automated replanning
  - **Partial recovery**: Accept a partial plan that achieves a safe intermediate state
  - **Abort**: Gracefully stop execution, move all robots to home positions
  - **Teleop**: Take direct control of a specific resource (send individual commands)
- [ ] Validate human-provided plans through the same safety pipeline (Tier 1-2 from `safety_constraints_in_replanning.md`)

### 3.3 Resumption Protocol
- [ ] After human intervention resolves the stuck state, re-enter the automated pipeline
- [ ] Update the environment model M_e with any state changes from human actions
- [ ] Re-sync `OnlineFsaMonitor` and `OnlineSafetyMonitor` with the new state
- [ ] Log the full intervention episode to `monitor/human/interventions.jsonl`

**Files to create/modify:**
- `agents/intelligent_product/process_planner.py` — replace `os._exit(1)` with escalation message
- `agents/intelligent_product/product_agent.py` — add `_EscalationDispatch` behaviour
- `agents/shared_information/user.py` — add `_EscalationInbox` behaviour with decision handling
- `agents/central_controller/central_controller_agent.py` — add state re-sync after human intervention

---

## 4. System Status Monitoring Module

**Goal:** Provide the human operator with real-time and historical visibility into system state.

### 4.1 Live Status Dashboard Data
- [ ] Aggregate existing monitor outputs into a unified status API/stream:
  - **Plan progress**: Current FSA state, completed/remaining tasks (from `OnlineFsaMonitor`)
  - **Resource states**: Per-robot position, held part, gripper state, current task (from RA state snapshots)
  - **Part tracker**: Per-part location, state, observation status (from PA `_part_tracker`)
  - **Safety monitor**: Per-rule DFA state, any active violations (from `OnlineSafetyMonitor`)
  - **Replan history**: Number of replans triggered, success/failure, method used (DES/LLM/human)
- [ ] Publish status updates via XMPP messages to the `User` agent on every state change
- [ ] Write consolidated snapshots to `monitor/status/system_status.json` (overwritten each update)

### 4.2 Event Timeline
- [ ] Extend existing `online_fsa_trace.jsonl` with richer event types:
  - `task_dispatched`, `safety_check_requested`, `safety_check_result`, `task_started`, `task_completed`, `task_failed`
  - `replan_triggered`, `replan_succeeded`, `replan_failed`, `escalation_sent`, `human_intervention`
- [ ] Include duration metrics per task (dispatch-to-completion latency)
- [ ] Support filtering by resource, part, or time window

### 4.3 Alert System
- [ ] Define alert severity levels: `INFO`, `WARNING`, `CRITICAL`
- [ ] Alert triggers:
  - `INFO`: Task completed, plan progress milestone
  - `WARNING`: Safety check blocked a task, replan triggered, camera observation failed
  - `CRITICAL`: All replan attempts failed, safety override used, system entering escalation
- [ ] Route alerts to `User` agent via dedicated `alert` message type
- [ ] Support configurable alert thresholds (e.g., suppress INFO during normal operation)

### 4.4 Historical Analysis
- [ ] Archive monitor data per session (extend existing `spade_main.py` archival to include human interaction logs)
- [ ] Provide summary statistics per session: total tasks, replans, interventions, safety violations, completion time

**Files to create/modify:**
- New: `agents/shared_information/status_monitor.py` — aggregates all monitor data, publishes to User agent
- `agents/central_controller/online_fsa_monitor.py` — extend event types
- `agents/central_controller/online_safety_monitor.py` — publish DFA state changes
- `agents/shared_information/user.py` — add `_StatusInbox`, `_AlertInbox` behaviours
- `spade_main.py` — initialize status monitor, wire to existing agents

---

## 5. Operator Authentication and Audit Trail

**Goal:** Ensure all human interactions are attributed and traceable.

### 5.1 Operator Identity
- [ ] Add operator ID to the `User` agent configuration (from `initialization/user.json`)
- [ ] Tag all human-originated messages with operator ID and timestamp
- [ ] Support multiple simultaneous operators (e.g., safety officer vs line operator) with role-based permissions

### 5.2 Audit Log
- [ ] Maintain an append-only audit log at `monitor/human/audit.jsonl`
- [ ] Record every human action: requirement edits, safety overrides, intervention decisions, manual commands
- [ ] Include: timestamp, operator ID, action type, payload, system state at time of action
- [ ] Never delete or modify audit entries (append-only)

**Files to create/modify:**
- `agents/shared_information/user.py` — add audit logging to all human action handlers
- New: `monitor/human/audit.jsonl`
- `initialization/` — add `user.json` config with operator roles

---

## 6. Communication Protocol Extensions

**Goal:** Define the XMPP message types needed for human interaction, integrating with existing message protocols.

### 6.1 New Message Types
- [ ] `escalation_request` (PA/CCA → User): Automated recovery failed, human needed
- [ ] `escalation_response` (User → PA): Human decision (manual plan / override / abort)
- [ ] `status_update` (StatusMonitor → User): Periodic system state snapshot
- [ ] `alert` (CCA/PA → User): Severity-tagged event notification
- [ ] `safety_review_request` (CCA → User): Safety violation needs human judgment
- [ ] `safety_review_response` (User → CCA): Override / confirm block
- [ ] `requirement_update` (User → PA): Runtime requirement modification
- [ ] `manual_command` (User → RA): Direct resource control during teleop

### 6.2 Message Routing
- [ ] Register new message templates in each agent's behaviour setup
- [ ] Ensure all new message types go through the existing XMPP bus (no separate channels)
- [ ] Add message type validation in receiver behaviours

**Files to create/modify:**
- All agent files — register new message templates
- `agents/shared_information/user.py` — handle all new outbound/inbound message types

---

## Implementation Priority

| Phase | Items | Effort | Impact |
|-------|-------|--------|--------|
| **Phase 1** | 3.1 Escalation trigger (replace `os._exit`) | Low | Eliminates hard crashes |
| **Phase 2** | 3.2-3.3 Human decision + resumption | Medium | Enables Layer 3 recovery |
| **Phase 3** | 4.1-4.3 Live status + alerts | Medium | Operator situational awareness |
| **Phase 4** | 2.2-2.3 Safety review interface | Medium | Human-gated safety decisions |
| **Phase 5** | 1.1-1.2 Requirements editor + review | Medium | Structured input pipeline |
| **Phase 6** | 5.1-5.2 Auth + audit trail | Low | Traceability and compliance |
| **Phase 7** | 1.3 Runtime requirements modification | High | Full dynamic reconfiguration |
| **Phase 8** | 6.1-6.2 Protocol extensions | Low | Wraps all above into clean message protocol |

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Human Operator                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────────┐  │
│  │ Req.     │ │ Safety   │ │ Status   │ │ Intervention  │  │
│  │ Editor   │ │ Reviewer │ │ Monitor  │ │ Console       │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────┬────────┘  │
└───────┼────────────┼────────────┼───────────────┼───────────┘
        │            │            │               │
   ┌────▼────────────▼────────────▼───────────────▼────┐
   │              User Agent (XMPP)                     │
   │  _RequirementInbox  _SafetyReviewInbox             │
   │  _StatusInbox       _EscalationInbox               │
   │  _AlertInbox        AuditLogger                    │
   └────┬────────────┬────────────┬───────────────┬────┘
        │            │            │               │
   requirement_   safety_     status_        escalation_
   update        review_     update         response
                 response
        │            │            │               │
   ┌────▼──┐    ┌────▼──┐   ┌────▼──────┐   ┌────▼──┐
   │  PA   │    │  CCA  │   │  Status   │   │  PA   │
   │       │    │       │   │  Monitor  │   │       │
   └───────┘    └───────┘   └───────────┘   └───────┘
```

---

## Relation to Existing TODOs

- **`safety_constraints_in_replanning.md`** — Tier 3 (re-query loop) and its "escalate to Layer 3" directly feeds into Section 3 of this document
- **`validate_llm_bridge_output.md`** — LLM bridge validation tiers produce the rejection context that gets included in escalation payloads (Section 3.1)
- **`connect_ros2_moveit_to_agent.md`** — Teleop capability (Section 3.2) requires the ROS2 bridge to be in place for direct robot commands
