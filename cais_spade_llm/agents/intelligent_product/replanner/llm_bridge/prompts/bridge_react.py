"""Legacy bridge ReAct prompt builders for LLM-guided recovery."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from textwrap import dedent
from typing import Any

from cais_spade_llm.resources.resource_profile import (
    ResourceProfile,
    get_resource_profile,
)

# ----------------------------------------------------------------------
# State exploration prompt
# ----------------------------------------------------------------------
def build_state_exploration_prompt(
    stuck_state: dict,
    part_tracker: dict | None,
    P_id: list[str],
    goal_state: str,
    ra_jid: str,
    tools_catalog: list[dict],
    resource_infos: list[dict],
    obligation_targets: list[dict] | None = None,
    operator_feedback: str = "",
    primitive_catalog: list[dict] | None = None,
    bridge_snapshot: dict | None = None,
    grounding_context: dict | None = None,
    bridge_resources: dict | None = None,
) -> str:
    """
    Prompt to generate a bridge recovery macro proposal when DES finds no modeled path.

    When primitive_catalog is provided, the bridge LLM is asked to compose
    recovery macros from controller primitives (the new path).  When absent,
    falls back to the legacy behavior of composing from catalog task functions.
    """
    from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.primitive_semantics import (
        filter_synthesis_primitive_catalog,
    )

    part_info = json.dumps(part_tracker, indent=2) if part_tracker else "unavailable"
    resource_info = json.dumps(resource_infos, indent=2)
    obligation_info = json.dumps(obligation_targets or [], indent=2)
    feedback_text = str(operator_feedback or "").strip() or "(none)"
    bridge_snapshot_info = json.dumps(bridge_snapshot or {}, indent=2)
    grounding_context_info = json.dumps(grounding_context or {}, indent=2)

    # Decide whether to use primitive-based or legacy catalog-based prompt.
    if primitive_catalog or bridge_resources:
        resource_overview = {}
        resource_catalogs = {}
        for resource_jid, raw_entry in (bridge_resources or {}).items():
            if not isinstance(raw_entry, dict):
                continue
            resource_overview[str(resource_jid)] = {
                "primitive_snapshot": raw_entry.get("bridge_snapshot") or raw_entry.get("primitive_snapshot") or {},
                "modeled_state": raw_entry.get("modeled_state") or {},
                "pending_tasks": raw_entry.get("pending_tasks") or [],
                "static_capabilities": raw_entry.get("static_capabilities") or {},
            }
            resource_catalogs[str(resource_jid)] = filter_synthesis_primitive_catalog(
                raw_entry.get("primitive_catalog")
                or raw_entry.get("execution_primitive_catalog")
                or []
            )
        if not resource_catalogs and primitive_catalog:
            resource_catalogs[str(ra_jid)] = filter_synthesis_primitive_catalog(
                primitive_catalog
            )
        if not resource_overview and bridge_snapshot:
            resource_overview[str(ra_jid)] = {
                "primitive_snapshot": bridge_snapshot or {},
                "modeled_state": {},
                "pending_tasks": [],
                "static_capabilities": {},
            }
        resources_info = json.dumps(resource_overview, indent=2)
        primitives_info = json.dumps(resource_catalogs, indent=2)
        tools_info = json.dumps(tools_catalog, indent=2)
        return (
            f"The system is disrupted. Current focused/stuck resource: {ra_jid}\n"
            f"CURRENT DISRUPTED SEARCH STATE:\n{json.dumps(stuck_state, indent=2)}\n\n"
            f"Current part states and locations (including camera coordinates for lost parts):\n{part_info}\n\n"
            f"Parts that still need to reach {goal_state}: {P_id}\n\n"
            f"ACTIVE SAFETY OBLIGATION TARGETS:\n{obligation_info}\n\n"
            f"OPERATOR REFINEMENT FEEDBACK:\n{feedback_text}\n\n"
            f"WHOLE-SYSTEM BRIDGE RESOURCES (snapshots, modeled states, pending tasks):\n{resources_info}\n\n"
            f"GROUNDING CONTEXT (use context_ref paths into this structure for grounded values; "
            "resource is the focused resource and resources/<resource_jid>/... exposes all bridge resources):\n"
            f"{grounding_context_info}\n\n"
            f"TASK-LEVEL TOOLS CATALOG (for reference on normal task semantics):\n{tools_info}\n\n"
            f"PER-RESOURCE BRIDGE PRIMITIVES (LLM-facing synthesis catalog; use grasp_part/release_part instead of raw gripper attach/detach pairs):\n{primitives_info}\n\n"
            f"RESOURCE CAPABILITIES: (Check reachability and staging areas before assigning coordinates)\n{resource_info}\n\n"
            "There is no catalog-valid modeled continuation for the active recovery situation. "
            "Propose exactly one SAFETY-DRIVEN BRIDGE PLAN as JSON.\n"
            "Rules:\n"
            "1. The bridge plan MUST target exactly one primary active safety obligation from ACTIVE SAFETY OBLIGATION TARGETS.\n"
            "2. The bridge plan MUST use ordered macro_tasks[] and the order is serial in v1.\n"
            "3. The sequence must be state-connected: each macro_task must be valid from the projected post-state of earlier macro_tasks.\n"
            "4. Each macro_task may choose any resource_jid shown in WHOLE-SYSTEM BRIDGE RESOURCES if that is needed to satisfy the primary obligation.\n"
            "5. Do not optimize unrelated work; only add the bridge tasks needed to satisfy the primary obligation and restore modeled DES continuation.\n"
            "6. primitive_steps MUST use only the controller primitives listed above for that macro_task.resource_jid.\n"
            "7. Each primitive_steps entry needs the exact primitive name and params.\n"
            "8. When a param value should come from current observed or known context, use "
            '{"context_ref": "/..."} pointing into the GROUNDING CONTEXT.\n'
            "9. Do not invent absolute Cartesian coordinates or pose names when the grounding context already provides them.\n"
            "10. Small literal relative offsets for move_relative are allowed when they are part of the recovery motion itself.\n"
            "11. Observational primitives may include store_as to save one output for later steps.\n"
            '12. Later steps may reference stored outputs via {"context_ref": "/step_outputs/<alias>/..."}.\n'
            "13. In v1, use store_as only with detect_parts(part_name=...) or get_current_pose().\n"
            "14. Respect each primitive's preconditions and effects over the projected snapshot.\n"
            "15. Do not use a primitive whose preconditions are false after earlier steps.\n"
            "16. Specify expected_start_state matching the resource's projected current state for each macro_task.\n"
            "17. If a macro_task manipulates exactly one part, set part_name to that canonical part name.\n"
            "18. Put any macro-level task context needed for tracking/safety in task_params "
            "(for example destination_location or last_known_location).\n"
            "19. If task_metadata.required_context_keys or task_metadata.part_transition "
            "refer to params such as destination_location, include them in task_params.\n"
            "20. Specify task_metadata with in_state, out_state for safety validation.\n"
            "21. If the macro_task manipulates parts, include part_transition in task_metadata.\n"
            "22. Prefer the smallest serial macro_tasks[] sequence that satisfies the primary safety obligation and reconnects DES.\n"
            "23. The final projected post-state after the last macro_task must discharge the primary obligation and restore a state where normal DES planning can continue.\n"
            'Example grounded param:\n{"primitive":"move_cartesian","params":{"x":{"context_ref":"/resource/current_pose/x"}}}\n'
            'Example observation binding:\n{"primitive":"detect_parts","params":{"part_name":"SG"},"store_as":"detected_sg"}\n'
            'Then later use {"context_ref":"/step_outputs/detected_sg/pose/x"}\n'
            "{\n"
            '  "primary_obligation": {\n'
            '    "rule_id": "<one rule_id from ACTIVE SAFETY OBLIGATION TARGETS>",\n'
            '    "resource_jid": "<matching obligation target resource_jid>"\n'
            "  },\n"
            '  "macro_tasks": [\n'
            "    {\n"
            f'      "resource_jid": "{ra_jid}",\n'
            '      "macro_name": "<descriptive recovery macro name>",\n'
            '      "description": "<what this macro_task accomplishes>",\n'
            '      "rationale": "<why this step helps satisfy the primary obligation>",\n'
            '      "expected_start_state": "<projected current state for this resource>",\n'
            '      "part_name": "<optional canonical part name for tracking>",\n'
            '      "task_params": {"<tracking_or_context_key>": "<literal or context_ref object>"},\n'
            '      "task_metadata": {\n'
            '        "in_state": "<resource state before macro>",\n'
            '        "out_state": "<resource state after macro>",\n'
            '        "required_context_keys": [],\n'
            '        "context_mapping": {},\n'
            '        "part_transition": null\n'
            "      },\n"
            '      "primitive_steps": [\n'
            "        {\n"
            '          "primitive": "<controller primitive name>",\n'
            '          "params": {"<param_name>": "<literal or context_ref object>"},\n'
            '          "store_as": "<optional alias for detect_parts/get_current_pose output>"\n'
            "        }\n"
            "      ]\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Return ONLY the JSON object, no explanation."
        )

    # Legacy fallback: compose from existing catalog functions.
    tools_info = json.dumps(tools_catalog, indent=2)
    return (
        f"A resource ({ra_jid}) is stuck in state:\n{json.dumps(stuck_state, indent=2)}\n\n"
        f"Current part states and locations (including camera coordinates for lost parts):\n{part_info}\n\n"
        f"Parts that still need to reach {goal_state}: {P_id}\n\n"
        f"ACTIVE SAFETY OBLIGATION TARGETS:\n{obligation_info}\n\n"
        f"OPERATOR REFINEMENT FEEDBACK:\n{feedback_text}\n\n"
        f"TOOLS CATALOG: (Reference this for available capabilities)\n{tools_info}\n\n"
        f"RESOURCE CAPABILITIES: (Check reachability and staging areas before assigning coordinates)\n{resource_info}\n\n"
        "The resource has no catalog-valid modeled path to satisfy the active recovery target. "
        "Propose exactly one HIGH-LEVEL recovery macro as JSON.\n"
        "Rules:\n"
        "1. The outer proposal function_name MAY be new.\n"
        "2. macro_steps MUST compile to EXISTING exact catalog function names for the same resource.\n"
        "3. Do NOT invent low-level controller capabilities.\n"
        "4. Prefer the smallest macro that satisfies the active safety obligation target.\n"
        "5. Every macro_steps entry must include exact params needed for execution.\n"
        "{\n"
        '  "function_name": "<new high-level recovery macro name>",\n'
        f'  "resource_jid": "{ra_jid}",\n'
        '  "description": "<what this recovery macro accomplishes>",\n'
        '  "rationale": "<why this satisfies the obligation or unsticks the resource>",\n'
        '  "macro_steps": [\n'
        "    {\n"
        '      "function_name": "<EXISTING catalog function name>",\n'
        '      "params": {"<param_name>": "<value>"}\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Return ONLY the JSON object, no explanation."
    )


def _bridge_observe_domain_context() -> str:
    """Domain context injected only in the observe_required phase."""
    return dedent(
        """\
        OBSERVATION OUTPUT SHAPES (stored under /step_outputs/<store_as>/...):
        - detect_parts(part_name) -> {part_name, pose: {x, y, z, qx?, qy?, qz?, qw?}, model_name?}
          Reference: /step_outputs/<alias>/pose/x, .../pose/y, .../pose/z
        - get_current_pose() -> {pose: {x, y, z, qx, qy, qz, qw}}
          Reference: /step_outputs/<alias>/pose/qx, .../pose/qy, etc.

        CONTEXT_REF SYNTAX FOR store_as:
        - Use {"context_ref": "/step_outputs/<alias>/..."} in later primitives to reference stored values.
        - Example: {"context_ref": "/step_outputs/detected_part/pose/x"} resolves to the x coordinate.
        """
    ).strip()


def _bridge_operation_kind(catalog_entry: dict[str, Any]) -> str:
    semantics = dict(catalog_entry.get("bridge_semantics") or {})
    return str(semantics.get("operation_kind", "") or "").strip().lower()


def _bridge_resource_has_operation(
    resource_entry: dict[str, Any],
    operation_kinds: set[str],
) -> bool:
    return any(
        _bridge_operation_kind(catalog_entry) in operation_kinds
        for catalog_entry in (resource_entry.get("primitive_catalog") or [])
        if isinstance(catalog_entry, dict)
    )


def _bridge_resource_core(resource_entry: dict[str, Any]) -> dict[str, Any]:
    return dict(resource_entry.get("resource_core") or {})


def _bridge_resource_facets(resource_entry: dict[str, Any]) -> dict[str, Any]:
    return dict(resource_entry.get("resource_facets") or {})


def _bridge_manipulator_facet(resource_entry: dict[str, Any]) -> dict[str, Any]:
    return dict(_bridge_resource_facets(resource_entry).get("manipulator") or {})


def _check_pose_in_bounds(
    pose: dict[str, Any],
    bounds: dict[str, Any],
) -> tuple[bool, str]:
    """Check if a Cartesian pose falls within workspace bounds.

    Pure-function equivalent of ``RobotAgent._is_pose_in_workspace``.
    Returns ``(is_inside, reason)``.
    """
    violations: list[str] = []
    for axis in ("x", "y", "z"):
        val = pose.get(axis)
        if val is None:
            continue
        try:
            val = float(val)
        except (TypeError, ValueError):
            continue
        lo_key, hi_key = f"{axis}_min_m", f"{axis}_max_m"
        lo = bounds.get(lo_key)
        hi = bounds.get(hi_key)
        if lo is not None and val < float(lo):
            violations.append(f"{axis}={val:.4f} < {lo_key}={float(lo):.4f}")
        if hi is not None and val > float(hi):
            violations.append(f"{axis}={val:.4f} > {hi_key}={float(hi):.4f}")
    if violations:
        return False, "pose outside workspace: " + ", ".join(violations)
    return True, "pose within workspace"


def _bridge_blocked_states(resource_entry: dict[str, Any]) -> set[str]:
    """Return states explicitly declared as blocked by any primitive's preconditions."""
    blocked: set[str] = set()
    for entry in (resource_entry.get("primitive_catalog") or []):
        if not isinstance(entry, dict):
            continue
        preconditions = dict(entry.get("preconditions") or {})
        state_rule = dict(preconditions.get("current_state") or {})
        not_equals = state_rule.get("not_equals")
        if isinstance(not_equals, str) and not_equals.strip():
            blocked.add(not_equals.strip())
    return blocked


def _bridge_unblocking_primitive(resource_entry: dict[str, Any]) -> str:
    catalog = [
        entry
        for entry in (resource_entry.get("primitive_catalog") or [])
        if isinstance(entry, dict)
    ]
    for preferred_name in ("move_to_named_pose", "move_home", "move_to_safe_pose"):
        for entry in catalog:
            if str(entry.get("name", "") or "").strip() == preferred_name:
                return preferred_name
    for operation_kind in ("home", "clear", "motion"):
        for entry in catalog:
            if _bridge_operation_kind(entry) == operation_kind:
                return str(entry.get("name", "") or "").strip()
    return ""


def _format_bridge_condition_hint(condition: dict[str, Any]) -> str:
    entity = str(condition.get("entity", "") or "").strip()
    field = str(condition.get("field", "") or "").strip()
    expected = condition.get("expected")
    if entity and field:
        return f"{entity}.{field}={expected!r}"
    if entity:
        return entity
    return str(expected)


def _relevant_bridge_safety_constraint(
    constraint: dict[str, Any],
    *,
    relevant_parts: set[str],
    relevant_resources: set[str],
    held_parts: set[str],
) -> bool:
    part_name = str(constraint.get("part_name", "") or "").strip()
    resource_jid = str(constraint.get("resource_jid", "") or "").strip()
    if part_name and part_name in (relevant_parts | held_parts):
        return True
    if resource_jid and resource_jid in relevant_resources:
        return True
    for until_condition in (constraint.get("until_conditions") or []):
        if not isinstance(until_condition, dict):
            continue
        entity = str(until_condition.get("entity", "") or "").strip()
        entity_kind = str(until_condition.get("entity_kind", "") or "").strip().lower()
        if entity_kind == "part" and entity in relevant_parts:
            return True
        if entity_kind == "resource" and entity in relevant_resources:
            return True
    return False


def _summarize_bridge_safety_constraint(constraint: dict[str, Any]) -> str:
    part_name = str(constraint.get("part_name", "") or "").strip()
    resource_jid = str(constraint.get("resource_jid", "") or "").strip()
    forbidden_location = str(constraint.get("forbidden_location", "") or "").strip()
    location = str(constraint.get("location", "") or "").strip()
    resource_jids = [
        str(item).strip()
        for item in (constraint.get("resource_jids") or [])
        if str(item).strip()
    ]
    after_event_kind = str(constraint.get("after_event_kind", "") or "").strip()
    after_part_name = str(constraint.get("after_part_name", "") or "").strip()
    constraint_type = str(constraint.get("constraint_type", "") or "").strip()
    rule_id = str(constraint.get("rule_id", "") or "").strip()
    until_conditions = [
        _format_bridge_condition_hint(condition)
        for condition in (constraint.get("until_conditions") or [])
        if isinstance(condition, dict)
    ]
    until_text = ""
    if until_conditions:
        until_text = f" until {' and '.join(until_conditions[:2])}"
    if part_name and forbidden_location:
        return (
            f"Safety constraint: part '{part_name}' must not be placed at "
            f"'{forbidden_location}'{until_text}."
        )
    if location and resource_jids:
        return (
            f"Safety constraint: resources {', '.join(resource_jids)} "
            f"must not occupy '{location}' at the same time."
        )
    before_conditions = [
        _format_bridge_condition_hint(condition)
        for condition in (constraint.get("before_conditions") or [])
        if isinstance(condition, dict)
    ]
    if not before_conditions:
        before_condition = constraint.get("before_condition")
        if isinstance(before_condition, dict):
            before_conditions = [_format_bridge_condition_hint(before_condition)]
    if after_event_kind and before_conditions:
        target = after_part_name or part_name or "the targeted entity"
        return (
            f"Safety constraint: before {after_event_kind} of {target}, "
            + " and ".join(before_conditions[:2])
            + " must hold."
        )
    details: list[str] = []
    if part_name:
        details.append(f"part='{part_name}'")
    if resource_jid:
        details.append(f"resource='{resource_jid}'")
    if forbidden_location:
        details.append(f"forbidden_location='{forbidden_location}'")
    if location:
        details.append(f"location='{location}'")
    if resource_jids:
        details.append(f"resources={resource_jids}")
    if constraint_type:
        details.append(f"constraint_type='{constraint_type}'")
    if rule_id:
        details.append(f"rule_id='{rule_id}'")
    if not details:
        reason = str(constraint.get("reason", "") or "").strip()
        if reason:
            details.append(reason)
    if until_text:
        details.append(until_text.strip())
    return f"Safety constraint: {', '.join(details)}."


def _generate_bridge_event_hints(
    *,
    bridge_resources: dict[str, Any] | None = None,
    grounding_context: dict[str, Any] | None = None,
    unmet_reentry_conditions: list[dict[str, Any]] | None = None,
    validation_feedback: list[dict[str, Any]] | None = None,
    bridge_safety_context: dict[str, Any] | None = None,
) -> list[str]:
    return []


def _append_feasibility_feedback_hints(
    hints: list[str],
    validation_feedback: list[dict[str, Any]] | None,
) -> None:
    """Append feasibility feedback hints (genuine runtime feedback, not bias)."""
    emitted = 0
    for feedback in reversed(list(validation_feedback or [])):
        if not isinstance(feedback, dict):
            continue
        message = str(feedback.get("message", "") or "").strip()
        if not message:
            continue
        lowered = message.lower()
        if not any(
            token in lowered
            for token in ("infeasible", "workspace", "pose outside", "feasibility")
        ):
            continue
        hints.append(
            f"Prior proposal rejected: '{message}'. Do not re-propose the same resource "
            "for that operation unless the state or target pose has changed."
        )
        emitted += 1
        if emitted >= 2:
            break


def _bridge_events_domain_context(
    *,
    bridge_resources: dict[str, Any] | None = None,
    grounding_context: dict[str, Any] | None = None,
    unmet_reentry_conditions: list[dict[str, Any]] | None = None,
    validation_feedback: list[dict[str, Any]] | None = None,
    bridge_safety_context: dict[str, Any] | None = None,
) -> str:
    """Domain context injected only in the bridge_events phase."""
    # Determine which resource types participate in this bridge.
    participating_types: set[str] = set()
    for _jid, entry in (bridge_resources or {}).items():
        rtype = str(
            (entry if isinstance(entry, dict) else {}).get("resource_type", "")
        ).strip().lower()
        if rtype:
            participating_types.add(rtype)

    part_states_section = dedent(
        """\
        PART STATES:
        - Parts track states: unknown, ready, in_gripper, assembled, misplaced.
        - Picking a part transitions it to in_gripper.
        - Placing a part at its goal destination transitions it to assembled.
        - Releasing a part at a non-goal location transitions it to ready.
        - Use expected_part_delta on each event to declare intended part state changes.
        """
    ).strip() if (not participating_types or "robot" in participating_types) else dedent(
        """\
        PART STATES:
        - Parts track states: unknown, ready, assembled, misplaced.
        - Use expected_part_delta on each event to declare intended part state changes.
        """
    ).strip()
    return part_states_section


def _bridge_catalog_names(entries: list[dict[str, Any]] | None) -> set[str]:
    names: set[str] = set()
    for entry in (entries or []):
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "") or "").strip()
        if name:
            names.add(name)
    return names


def _catalog_supports_manipulator_pick_place(catalog_names: set[str]) -> bool:
    base_required = {
        "detect_parts",
        "get_current_pose",
        "compute_pick_targets",
        "compute_place_targets",
        "move_cartesian",
        "move_pose",
    }
    names = set(catalog_names or set())
    if not base_required <= names:
        return False
    return (
        {"grasp_part", "release_part"} <= names
        or {"close_gripper", "open_gripper", "attach_part", "detach_part"} <= names
    )


def _bridge_prompt_profiles(
    *,
    bridge_resources: dict[str, Any] | None = None,
    primitive_catalog: list[dict[str, Any]] | None = None,
) -> list[ResourceProfile]:
    profiles: list[ResourceProfile] = []
    seen_types: set[str] = set()

    def _add_profile(resource_type: Any) -> None:
        profile = get_resource_profile(str(resource_type or "").strip().lower() or "resource")
        profile_type = str(profile.resource_type or "resource").strip().lower() or "resource"
        if profile_type in seen_types:
            return
        seen_types.add(profile_type)
        profiles.append(profile)

    for raw_entry in (bridge_resources or {}).values():
        if not isinstance(raw_entry, dict):
            continue
        adapter = dict(raw_entry.get("bridge_adapter") or {})
        catalog = list(raw_entry.get("primitive_catalog") or [])
        if not catalog and not adapter.get("supports_executable_bridge"):
            continue
        resource_type = str(
            raw_entry.get("resource_type")
            or dict(raw_entry.get("bridge_snapshot") or {}).get("resource_type")
            or dict(dict(raw_entry.get("bridge_snapshot") or {}).get("resource_core") or {}).get("resource_type")
            or "resource"
        ).strip().lower() or "resource"
        _add_profile(resource_type)

    if not bridge_resources:
        for entry in (primitive_catalog or []):
            if not isinstance(entry, dict):
                continue
            _add_profile(entry.get("resource_type"))

    return profiles


def _bridge_final_plan_domain_context(
    primitive_card: str,
    *,
    profiles: list[ResourceProfile] | None = None,
) -> str:
    """Domain context injected only in the final_plan phase."""
    sections = [
        dedent(
            """\
            GENERIC FINAL-PLAN COMPOSITION RULES:
            - Realize APPROVED BRIDGE EVENTS in the same order unless the planner draft is
              explicitly wrong about order.
            - Use only primitives that appear in the PRIMITIVE REFERENCE CARD for the
              chosen resource.
            - Chain primitives so that each step's effects satisfy the next step's
              preconditions.
            - Use task_metadata to express resource-state transitions, context
              requirements, and part transitions that the macro is intended to close.
            - Reuse validated values from GROUNDING CONTEXT via context_ref instead of
              inventing new coordinates, destinations, or identifiers.
            - When clearing or relocating a resource, move it only to a validated safe
              destination that is already present in context or primitive semantics.

            CONTEXT_REF SYNTAX:
            - Grounding context paths: {"context_ref": "/parts/<PART>/observed_pose/x"}, {"context_ref": "/parts/<PART>/target/model_name"}, {"context_ref": "/resources/<JID>/resource_core/current_location"}.
            - Step output paths: {"context_ref": "/step_outputs/<alias>/pose/x"}, {"context_ref": "/step_outputs/<alias>/target_pose/x"}.
            """
        ).strip(),
    ]
    for profile in (profiles or []):
        addendum = str(getattr(profile, "prompt_addendum", "") or "").strip()
        if addendum:
            sections.append(addendum)
    if primitive_card:
        sections.append(f"PRIMITIVE REFERENCE CARD:\n{primitive_card}")
    return "\n\n".join(sections)


def _bridge_final_plan_generic_shape_example() -> str:
    return dedent(
        """\
        GENERIC FINAL-PLAN SHAPE EXAMPLE:
        - Use this only as a schema guide. Replace placeholder primitive names with real
          primitives from the chosen resource's PRIMITIVE REFERENCE CARD.
        - Keep bridge_event_summary and macro order aligned with APPROVED BRIDGE EVENTS.

        {
          "type": "final_plan",
          "plan": {
            "primary_obligation": {
              "rule_id": "<RULE_ID>",
              "resource_jid": "<RESOURCE_JID>"
            },
            "bridge_event_summary": [
              {
                "event_name": "<APPROVED_EVENT_NAME>",
                "resource_jid": "<RESOURCE_JID>",
                "part_name": "<optional PART>",
                "closes_conditions": [
                  {
                    "entity_kind": "<resource|part>",
                    "entity": "<ENTITY_ID>",
                    "field": "<FIELD>",
                    "expected": "<VALUE>"
                  }
                ],
                "rationale": "<why this event is needed>"
              }
            ],
            "macro_tasks": [
              {
                "resource_jid": "<RESOURCE_JID>",
                "macro_name": "realize_<approved_event>",
                "description": "Realize one approved bridge event.",
                "expected_start_state": "<STATE_BEFORE>",
                "part_name": "<optional PART>",
                "task_params": {},
                "task_metadata": {
                  "in_state": "<STATE_BEFORE>",
                  "out_state": "<STATE_AFTER>",
                  "required_context_keys": [],
                  "context_mapping": {},
                  "part_transition": null
                },
                "primitive_steps": [
                  {
                    "primitive": "<primitive_from_reference_card>",
                    "params": {}
                  }
                ]
              }
            ]
          }
        }
        """
    ).strip()


def _bridge_final_plan_manipulator_example() -> str:
    return dedent(
        """\
        MANIPULATOR PICK/PLACE REPAIR EXAMPLE:
        - Use this only when the chosen resource exposes manipulator pick/place primitives.
        - Adapt resource JIDs, part names, geometry, and states to the current bridge.
        - Keep macro order aligned with APPROVED BRIDGE EVENTS.

        {
          "type": "final_plan",
          "plan": {
            "macro_tasks": [
              {
                "resource_jid": "<RESOURCE_JID>",
                "macro_name": "pick_<part>",
                "description": "Acquire <PART> from its observed location.",
                "expected_start_state": "idle",
                "part_name": "<PART>",
                "task_params": {},
                "task_metadata": {
                  "in_state": "idle",
                  "out_state": "picked",
                  "required_context_keys": [],
                  "context_mapping": {},
                  "part_transition": {
                    "completed": {
                      "state": "in_gripper",
                      "location_template": "{resource_jid}_gripper"
                    }
                  }
                },
                "primitive_steps": [
                  {"primitive": "detect_parts", "params": {"part_name": "<PART>"}, "store_as": "detected_part"},
                  {"primitive": "get_current_pose", "params": {}, "store_as": "pre_pick_pose"},
                  {"primitive": "compute_pick_targets", "params": {"part_name": "<PART>", "product_geometry": {"board_center": {"x": 0.0, "y": 0.0, "z": 1.0}}}, "store_as": "part_pick_targets"},
                  {"primitive": "move_cartesian", "params": {"x": {"context_ref": "/step_outputs/detected_part/pose/x"}, "y": {"context_ref": "/step_outputs/detected_part/pose/y"}, "z": {"context_ref": "/step_outputs/part_pick_targets/travel_z"}, "speed": 1.2}},
                  {"primitive": "move_pose", "params": {"x": {"context_ref": "/step_outputs/detected_part/pose/x"}, "y": {"context_ref": "/step_outputs/detected_part/pose/y"}, "z": {"context_ref": "/step_outputs/part_pick_targets/pick_z"}, "qx": {"context_ref": "/step_outputs/pre_pick_pose/pose/qx"}, "qy": {"context_ref": "/step_outputs/pre_pick_pose/pose/qy"}, "qz": {"context_ref": "/step_outputs/pre_pick_pose/pose/qz"}, "qw": {"context_ref": "/step_outputs/pre_pick_pose/pose/qw"}, "speed": 0.8}},
                  {"primitive": "grasp_part", "params": {"model_name": {"context_ref": "/parts/<PART>/target/model_name"}, "part_name": "<PART>"}},
                  {"primitive": "move_relative", "params": {"dx": 0.0, "dy": 0.0, "dz": 0.05, "speed": 0.8}}
                ]
              },
              {
                "resource_jid": "<RESOURCE_JID>",
                "macro_name": "place_<part>",
                "description": "Place <PART> at its destination.",
                "expected_start_state": "picked",
                "part_name": "<PART>",
                "task_params": {"destination_location": "<DESTINATION>"},
                "task_metadata": {
                  "in_state": "picked",
                  "out_state": "idle",
                  "required_context_keys": ["destination_location"],
                  "context_mapping": {"location_param": "destination_location"},
                  "part_transition": {
                    "completed": {
                      "state": "assembled",
                      "location_param": "destination_location"
                    }
                  }
                },
                "primitive_steps": [
                  {"primitive": "get_current_pose", "params": {}, "store_as": "pre_place_pose"},
                  {"primitive": "compute_place_targets", "params": {"part_name": "<PART>", "product_geometry": {"board_center": {"x": 0.0, "y": 0.0, "z": 1.0}}}, "store_as": "part_place_targets"},
                  {"primitive": "move_cartesian", "params": {"x": {"context_ref": "/step_outputs/part_place_targets/slot_x"}, "y": {"context_ref": "/step_outputs/part_place_targets/slot_y"}, "z": {"context_ref": "/step_outputs/pre_place_pose/pose/z"}, "speed": 1.2}},
                  {"primitive": "move_pose", "params": {"x": {"context_ref": "/step_outputs/part_place_targets/slot_x"}, "y": {"context_ref": "/step_outputs/part_place_targets/slot_y"}, "z": {"context_ref": "/step_outputs/part_place_targets/place_z"}, "qx": {"context_ref": "/step_outputs/pre_place_pose/pose/qx"}, "qy": {"context_ref": "/step_outputs/pre_place_pose/pose/qy"}, "qz": {"context_ref": "/step_outputs/pre_place_pose/pose/qz"}, "qw": {"context_ref": "/step_outputs/pre_place_pose/pose/qw"}, "speed": 0.8}},
                  {"primitive": "release_part", "params": {"model_name": {"context_ref": "/parts/<PART>/target/model_name"}, "assume_released_if_open": true}},
                  {"primitive": "move_relative", "params": {"dx": 0.0, "dy": 0.0, "dz": 0.08, "speed": 1.0}}
                ]
              }
            ]
          }
        }

        REPAIR RULES:
        - Every macro_task MUST have a non-empty primitive_steps array.
        - compute_pick_targets and compute_place_targets MUST include "part_name" in params.
        - store_as aliases MUST be lowercase_snake_case and refs must use the exact same alias.
        - step_outputs are scoped to the current macro_task; do not reference aliases from another macro.
        - Keep macro count and macro order aligned with APPROVED BRIDGE EVENTS unless the planner draft is explicitly wrong about order.
        """
    ).strip()


def _bridge_final_plan_printer_job_control_example() -> str:
    return dedent(
        """\
        PRINTER JOB CONTROL EXAMPLE:
        A printer bridge event that cancels the active job:
        {
          "type": "final_plan",
          "plan": {
            "bridge_events": [
              {
                "event_name": "cancel_active_print",
                "resource_jid": "printer@localhost",
                "expected_resource_delta": {"from": "printing", "to": "idle"},
                "closes_conditions": [],
                "rationale": "Cancel the print job to free the printer for recovery."
              }
            ],
            "macro_tasks": [
              {
                "resource_jid": "printer@localhost",
                "macro_name": "cancel_active_print",
                "description": "Cancel the active print job.",
                "expected_start_state": "printing",
                "part_name": "",
                "task_params": {},
                "task_metadata": {
                  "in_state": "printing",
                  "out_state": "idle",
                  "required_context_keys": [],
                  "context_mapping": {},
                  "part_transition": null
                },
                "primitive_steps": [
                  {"primitive": "cancel_job", "params": {}}
                ]
              }
            ]
          }
        }
        """
    ).strip()


def _bridge_final_plan_repair_few_shot(
    *,
    profiles: list[ResourceProfile] | None = None,
) -> str:
    sections = [_bridge_final_plan_generic_shape_example()]
    for profile in (profiles or []):
        repair_example = str(getattr(profile, "repair_example", "") or "").strip()
        if repair_example:
            sections.append(repair_example)
    return "\n\n".join(section for section in sections if section)


def _bridge_generalize_location_summary(value: Any) -> Any:
    token = str(value or "").strip()
    if not token:
        return deepcopy(value)
    lowered = token.lower()
    if lowered.endswith("_gripper") or "@localhost_gripper" in lowered:
        return "resource_gripper"
    if "assembly_board" in lowered or "assembly station" in lowered:
        return "protected_assembly_region"
    if "fixture" in lowered or "recovery" in lowered:
        return "known_non_goal_workspace_region"
    if "prusa" in lowered or "printer" in lowered or "stash" in lowered:
        return "validated_staging_region"
    if "buffer" in lowered:
        return "shared_buffer_region"
    if "@localhost" in lowered:
        return "resource_workspace_region"
    return "known_workspace_region"


def _bridge_none_prompt_text_redact(text: str) -> str:
    normalized = str(text or "")
    patterns = (
        (r"\bassembly_board-[A-Za-z0-9_-]+\b", "protected_assembly_region"),
        (r"\bassembly board station\b", "protected_assembly_region"),
        (r"\bassembly board\b", "protected_assembly_region"),
        (r"\bprusa-[A-Za-z0-9_-]+\b", "validated_staging_region"),
        (r"\bfixture_[A-Za-z0-9_-]+\b", "known_non_goal_workspace_region"),
        (r"\bAssembly Station\b", "protected_assembly_region"),
    )
    for pattern, replacement in patterns:
        normalized = re.sub(pattern, replacement, normalized)
    return normalized


def _bridge_none_validation_feedback(
    validation_feedback: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for feedback in (validation_feedback or []):
        if not isinstance(feedback, dict):
            continue
        entry = deepcopy(feedback)
        message = str(entry.get("message", "") or "").strip()
        if message:
            lowered = message.lower()
            if "pose outside workspace" in lowered or "infeasible" in lowered:
                entry["message"] = (
                    "a prior proposal was infeasible for the selected resource under "
                    "current workspace limits"
                )
            elif (
                "did not close all current gamma" in lowered
                or "did not restore" in lowered
                or "marked re-entry" in lowered
                or "re-entry" in lowered
            ):
                unresolved_fields: list[str] = []
                for raw_line in message.splitlines():
                    line = str(raw_line or "").strip()
                    if not line.startswith("-"):
                        continue
                    field_ref = line[1:].strip().split("->", 1)[0].strip()
                    if field_ref:
                        unresolved_fields.append(_bridge_none_prompt_text_redact(field_ref))
                if not unresolved_fields:
                    unresolved_fields.extend(
                        _bridge_none_prompt_text_redact(match.strip())
                        for match in re.findall(
                            r"([A-Za-z0-9_@.-]+\.[A-Za-z0-9_]+)\s+expected",
                            message,
                        )
                        if str(match).strip()
                    )
                if unresolved_fields:
                    entry["message"] = (
                        "a prior proposal left required continuation conditions "
                        "unresolved for: " + ", ".join(unresolved_fields[:4])
                    )
                else:
                    entry["message"] = (
                        "a prior proposal left required continuation conditions unresolved"
                    )
            elif "inconsistent for robot resources" in lowered:
                entry["message"] = (
                    "a prior proposal used a resource-state transition inconsistent "
                    "with primitive semantics"
                )
            elif "modeled continuation" in lowered:
                entry["message"] = (
                    "a prior proposal did not restore a resumable modeled continuation"
                )
            elif "missing task_params" in lowered:
                entry["message"] = (
                    "a prior final plan omitted required task parameters"
                )
            else:
                message = re.sub(
                    r"bridge event '[^']+'",
                    "prior proposal",
                    message,
                    count=1,
                )
                entry["message"] = _bridge_none_prompt_text_redact(message)
        sanitized.append(entry)
    return sanitized


def _bridge_none_last_failure_summary(
    last_plan_failure: dict[str, Any] | None,
) -> dict[str, Any]:
    raw = last_plan_failure if isinstance(last_plan_failure, dict) else {}
    if not raw:
        return {}
    summary: dict[str, Any] = {}
    kind = str(raw.get("kind", "") or "").strip()
    if kind:
        summary["kind"] = kind
    message = str(raw.get("message", "") or raw.get("error", "") or "").strip()
    if message:
        summary["feedback"] = (
            _bridge_none_validation_feedback([{"message": message}])[0].get("message", "")
        )
    approved_events = raw.get("approved_bridge_events") or []
    if isinstance(approved_events, list) and approved_events:
        summary["approved_event_count"] = len(approved_events)
    compile_path = str(raw.get("compile_path", "") or "").strip()
    if compile_path:
        summary["compile_path"] = compile_path
    modeled_gap = raw.get("modeled_continuation_gap")
    if isinstance(modeled_gap, dict) and modeled_gap:
        gap_summary: dict[str, Any] = {}
        goal_state = str(modeled_gap.get("goal_state", "") or "").strip()
        if goal_state:
            gap_summary["goal_state"] = goal_state
        remaining_parts: list[dict[str, Any]] = []
        for raw_entry in (modeled_gap.get("remaining_parts") or []):
            if not isinstance(raw_entry, dict):
                continue
            part_name = str(raw_entry.get("part_name") or "").strip()
            if not part_name:
                continue
            entry = {
                "part_name": part_name,
                "current_state": raw_entry.get("current_state"),
                "current_location_summary": _bridge_generalize_location_summary(
                    raw_entry.get("current_location")
                ),
            }
            remaining_parts.append(entry)
        candidate_resources: list[dict[str, Any]] = []
        for raw_entry in (modeled_gap.get("candidate_resources") or []):
            if not isinstance(raw_entry, dict):
                continue
            resource_jid = str(raw_entry.get("resource_jid") or "").strip()
            if not resource_jid:
                continue
            entry = {
                "resource_jid": resource_jid,
                "resource_state": str(raw_entry.get("resource_state") or "").strip(),
                "current_location_summary": _bridge_generalize_location_summary(
                    raw_entry.get("current_location")
                ),
                "has_bid": bool(raw_entry.get("has_bid", False)),
            }
            current_part = str(raw_entry.get("current_part") or "").strip()
            if current_part:
                entry["current_part"] = current_part
            next_function_name = str(raw_entry.get("next_function_name") or "").strip()
            if next_function_name:
                entry["next_function_name"] = next_function_name
            candidate_resources.append(entry)
        pending_suffix_summary: list[dict[str, Any]] = []
        for raw_entry in (raw.get("pending_suffix_summary") or []):
            if not isinstance(raw_entry, dict):
                continue
            if str(raw_entry.get("role") or "").strip() != "resume_suffix":
                continue
            resource_jid = str(raw_entry.get("resource_jid") or "").strip()
            if not resource_jid:
                continue
            entry = {
                "resource_jid": resource_jid,
                "entry_task_id": str(raw_entry.get("entry_task_id") or "").strip(),
                "entry_function_name": str(raw_entry.get("entry_function_name") or "").strip(),
                "entry_part_name": str(raw_entry.get("entry_part_name") or "").strip(),
                "required_resource_state": str(raw_entry.get("required_resource_state") or "").strip(),
                "required_part_state": str(raw_entry.get("required_part_state") or "").strip(),
                "required_location_summary": _bridge_generalize_location_summary(
                    raw_entry.get("required_location")
                ),
            }
            pending_suffix_summary.append(entry)
        gap_summary["remaining_parts"] = remaining_parts
        gap_summary["candidate_resources"] = candidate_resources
        if pending_suffix_summary:
            gap_summary["resume_suffix_entries"] = pending_suffix_summary
        summary["modeled_continuation_gap"] = gap_summary
    return summary


def _bridge_none_parts_requiring_observation(
    *,
    stuck_state: dict[str, Any] | None,
    unmet_reentry_conditions: list[dict[str, Any]] | None,
    observation_history: list[dict[str, Any]] | None,
) -> list[str]:
    required: set[str] = set()
    state_payload = dict(stuck_state or {})
    for part_name, raw_state in (state_payload.get("part_states") or {}).items():
        name = str(part_name or "").strip()
        state_token = str(raw_state or "").strip().lower()
        if name and state_token in {"misplaced", "unknown"}:
            required.add(name)
    for raw_condition in (unmet_reentry_conditions or []):
        if not isinstance(raw_condition, dict):
            continue
        if str(raw_condition.get("entity_kind", "") or "").strip() != "part":
            continue
        if str(raw_condition.get("role", "") or "").strip() != "bridge_replaced":
            continue
        entity = str(raw_condition.get("entity", "") or "").strip()
        if entity:
            required.add(entity)

    observed: set[str] = set()
    for row in (observation_history or []):
        if not isinstance(row, dict):
            continue
        observation = row.get("observation") or {}
        observed_part_name = str(
            (observation.get("part_name") if isinstance(observation, dict) else "")
            or (row.get("params") or {}).get("part_name")
            or ""
        ).strip()
        if observed_part_name:
            observed.add(observed_part_name)
    return sorted(part_name for part_name in required if part_name not in observed)


def _bridge_none_requirement_summary(
    *,
    unmet_reentry_conditions: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for raw_condition in (unmet_reentry_conditions or []):
        if not isinstance(raw_condition, dict):
            continue
        entity_kind = str(raw_condition.get("entity_kind", "") or "").strip()
        entity = str(raw_condition.get("entity", "") or "").strip()
        field = str(raw_condition.get("field", "") or "").strip()
        role = str(raw_condition.get("role", "") or "").strip()
        kind = str(raw_condition.get("kind", "") or "").strip()
        if not entity_kind or not entity or not field:
            continue
        item = {
            "entity_kind": entity_kind,
            "entity": entity,
            "field": field,
        }
        if role:
            item["role"] = role
        if kind == "focused_resource_terminal_state" or (
            entity_kind == "resource" and field == "current_state"
        ):
            item["expected_summary"] = (
                "restore_resume_entry_state"
                if role == "resume_suffix"
                else "restore_modeled_terminal_state"
            )
        elif field == "location":
            item["expected_summary"] = _bridge_generalize_location_summary(
                raw_condition.get("expected")
            )
        elif entity_kind == "part" and field == "state":
            expected = str(raw_condition.get("expected", "") or "").strip()
            if role == "bridge_replaced" and expected:
                item["expected"] = expected
            else:
                item["expected_summary"] = "restore_resume_entry_part_state"
        elif entity_kind == "resource" and field == "held_part":
            item["expected_summary"] = "restore_resume_carried_part_requirement"
        else:
            item["expected_summary"] = "restore_required_condition"
        summary.append(item)
    return summary


def _bridge_none_resource_role_summary(
    *,
    focused_resource_jid: str,
    bridge_resources: dict[str, Any] | None,
    infeasible_assignments: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for resource_jid, raw_entry in (bridge_resources or {}).items():
        jid = str(resource_jid or "").strip()
        if not jid or not isinstance(raw_entry, dict):
            continue
        adapter = dict(raw_entry.get("bridge_adapter") or {})
        snapshot = dict(raw_entry.get("bridge_snapshot") or {})
        manipulator = dict((snapshot.get("resource_facets") or {}).get("manipulator") or {})
        supports_pick_place = bool(adapter.get("supports_manipulator_pick_place"))
        if not supports_pick_place and jid != focused_resource_jid:
            continue
        entry: dict[str, Any] = {
            "resource_jid": jid,
            "role": (
                "resource_to_restore"
                if jid == str(focused_resource_jid or "").strip()
                else "candidate_recovery_executor"
            ),
            "current_state": str(snapshot.get("current_state") or "").strip(),
            "current_location_summary": _bridge_generalize_location_summary(
                snapshot.get("current_location")
            ),
        }
        held_part = str(
            manipulator.get("held_part") or snapshot.get("held_part") or ""
        ).strip()
        if held_part:
            entry["held_part"] = held_part
        gripper_state = str(
            manipulator.get("gripper_state") or snapshot.get("gripper_state") or ""
        ).strip()
        if gripper_state:
            entry["gripper_state"] = gripper_state
        candidates.append(entry)

    ruled_out: list[dict[str, Any]] = []
    for raw_entry in (infeasible_assignments or []):
        if not isinstance(raw_entry, dict):
            continue
        entry = {
            "resource_jid": str(raw_entry.get("resource_jid") or "").strip(),
            "part_name": str(raw_entry.get("part_name") or "").strip(),
            "scope": str(raw_entry.get("scope") or "").strip(),
            "reason_summary": _bridge_none_prompt_text_redact(
                str(raw_entry.get("reason") or "").strip()
            ),
        }
        if entry["resource_jid"]:
            ruled_out.append(entry)

    return {
        "resource_to_restore": str(focused_resource_jid or "").strip(),
        "restoration_role_is_distinct_from_recovery_executor_choice": True,
        "candidate_recovery_executors": candidates,
        "ruled_out_assignments": ruled_out[:6],
    }


def _bridge_low_bias_prompt_value(
    value: Any,
    *,
    strict_none: bool = False,
) -> Any:
    if strict_none and isinstance(value, str):
        return _bridge_none_prompt_text_redact(value)
    if isinstance(value, list):
        return [
            _bridge_low_bias_prompt_value(item, strict_none=strict_none)
            for item in value
        ]
    if not isinstance(value, dict):
        return deepcopy(value)

    sanitized: dict[str, Any] = {}
    field_name = str(value.get("field", "") or "").strip()
    for raw_key, raw_child in value.items():
        key = str(raw_key)
        if key == "pending_tasks":
            continue
        if strict_none and key in {
            "source_task_id",
            "source_task_ids",
            "source_function_name",
            "entry_task_id",
            "entry_function_name",
            "terminal_task_id",
            "terminal_function_name",
            "safety_rules",
            "reason",
            "description",
            "source",
            "until_conditions",
        }:
            continue
        if strict_none and key == "message":
            sanitized[key] = _bridge_none_prompt_text_redact(str(raw_child or ""))
            continue
        if strict_none and key == "reachability":
            if isinstance(raw_child, list):
                summaries: list[str] = []
                for item in raw_child:
                    summary = str(_bridge_generalize_location_summary(item) or "").strip()
                    if summary and summary not in summaries:
                        summaries.append(summary)
                sanitized["reachability_summary"] = summaries
            continue
        if strict_none and key == "staging_areas":
            sanitized["staging_area_count"] = (
                len(raw_child) if isinstance(raw_child, dict) else 0
            )
            continue
        if strict_none and key == "goal":
            goal_payload = raw_child if isinstance(raw_child, dict) else {}
            pending_parts = [
                part_name
                for part_name in (goal_payload.get("pending_parts") or [])
                if str(part_name).strip()
            ]
            sanitized["goal_summary"] = {
                "pending_part_count": len(pending_parts),
                "goal_defined": bool(str(goal_payload.get("goal_state") or "").strip()),
            }
            continue
        if key == "part_locations":
            if isinstance(raw_child, dict):
                sanitized["part_location_summary"] = {
                    str(part_name): _bridge_generalize_location_summary(location_value)
                    for part_name, location_value in raw_child.items()
                    if str(part_name).strip()
                }
            continue
        if key in {"location", "current_location", "last_known_location", "required_location"}:
            sanitized[f"{key}_summary"] = _bridge_generalize_location_summary(raw_child)
            continue
        if key in {"destination", "protected_destination", "forbidden_location", "location_to"}:
            sanitized[f"{key}_summary"] = _bridge_generalize_location_summary(raw_child)
            continue
        if strict_none and key == "pending_parts":
            sanitized["pending_part_count"] = (
                len(raw_child) if isinstance(raw_child, list) else 0
            )
            continue
        if strict_none and key == "goal_state":
            sanitized["goal_defined"] = bool(str(raw_child or "").strip())
            continue
        if key == "terminal_resource_state":
            sanitized["terminal_state_requirement"] = "restore_modeled_terminal_state"
            continue
        if key == "terminal_function_name":
            continue
        if strict_none and key == "expected" and field_name == "location":
            sanitized["expected_summary"] = _bridge_generalize_location_summary(raw_child)
            continue
        sanitized[key] = _bridge_low_bias_prompt_value(raw_child, strict_none=strict_none)
    return sanitized


def _bridge_low_bias_prompt_conditions(
    conditions: list[dict[str, Any]] | None,
    *,
    strict_none: bool = False,
) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for raw_condition in (conditions or []):
        if not isinstance(raw_condition, dict):
            continue
        kind = str(raw_condition.get("kind", "") or "").strip()
        field = str(raw_condition.get("field", "") or "").strip()
        if kind == "focused_resource_terminal_state":
            continue
        if field == "location" or kind in {
            "bridge_part_goal_location",
            "resume_entry_resource_location",
        }:
            continue
        if strict_none:
            cleaned = {
                "entity_kind": str(raw_condition.get("entity_kind", "") or "").strip(),
                "entity": str(raw_condition.get("entity", "") or "").strip(),
                "field": field,
            }
            if not all(cleaned.values()):
                continue
        else:
            cleaned = _bridge_low_bias_prompt_value(
                raw_condition,
                strict_none=strict_none,
            )
        if isinstance(cleaned, dict) and cleaned:
            sanitized.append(cleaned)
    return sanitized


def _bridge_none_continuation_summary(
    *,
    pending_suffix_summary: list[dict[str, Any]] | None,
    unmet_reentry_conditions: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    resources: list[dict[str, Any]] = []
    affected_parts: set[str] = set()
    for raw_summary in (pending_suffix_summary or []):
        if not isinstance(raw_summary, dict):
            continue
        normalized = {
            "resource_jid": str(raw_summary.get("resource_jid", "") or "").strip(),
            "pending_task_count": len(raw_summary.get("pending_task_ids") or []),
        }
        parts = [
            str(part_name)
            for part_name in (raw_summary.get("parts") or [])
            if str(part_name).strip()
        ]
        affected_parts.update(parts)
        if not normalized["resource_jid"]:
            continue
        resources.append(normalized)

    unmet_entities: set[tuple[str, str]] = set()
    for raw_condition in (unmet_reentry_conditions or []):
        if not isinstance(raw_condition, dict):
            continue
        entity_kind = str(raw_condition.get("entity_kind", "") or "").strip()
        entity = str(raw_condition.get("entity", "") or "").strip()
        field = str(raw_condition.get("field", "") or "").strip()
        kind = str(raw_condition.get("kind", "") or "").strip()
        if not entity_kind or not entity or not field:
            continue
        if kind == "focused_resource_terminal_state" or field == "location":
            continue
        unmet_entities.add((entity_kind, entity))

    return {
        "affected_resources": resources,
        "affected_parts": sorted(affected_parts),
        "continuation_required": bool(resources),
        "unmet_entity_count": len(unmet_entities),
        "required_outcomes": _bridge_none_requirement_summary(
            unmet_reentry_conditions=unmet_reentry_conditions,
        ),
    }


def _bridge_none_contract_targets(
    *,
    unmet_reentry_conditions: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for raw_condition in (unmet_reentry_conditions or []):
        if not isinstance(raw_condition, dict):
            continue
        entity_kind = str(raw_condition.get("entity_kind", "") or "").strip()
        entity = str(raw_condition.get("entity", "") or "").strip()
        field = str(raw_condition.get("field", "") or "").strip()
        expected = raw_condition.get("expected")
        if not entity_kind or not entity or not field or expected in (None, ""):
            continue
        item: dict[str, Any] = {
            "entity_kind": entity_kind,
            "entity": entity,
            "field": field,
            "expected": deepcopy(expected),
        }
        role = str(raw_condition.get("role", "") or "").strip()
        if role:
            item["role"] = role
        actual = raw_condition.get("actual")
        if field == "location":
            item["actual_summary"] = _bridge_generalize_location_summary(actual)
        elif actual not in (None, ""):
            item["actual"] = deepcopy(actual)
        targets.append(item)
    return targets


def _bridge_none_active_executor_summary(
    *,
    executor_bindings: list[dict[str, Any]] | None,
    handoff_requirements: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    bindings: list[dict[str, Any]] = []
    for raw_entry in (executor_bindings or []):
        if not isinstance(raw_entry, dict):
            continue
        part_name = str(raw_entry.get("part_name") or "").strip()
        resource_jid = str(raw_entry.get("resource_jid") or "").strip()
        if not part_name or not resource_jid:
            continue
        entry = {
            "part_name": part_name,
            "resource_jid": resource_jid,
            "scope": str(raw_entry.get("scope") or "").strip() or "recover_part_to_goal",
            "status": str(raw_entry.get("status") or "").strip() or "active",
        }
        bindings.append(entry)

    handoffs: list[dict[str, Any]] = []
    for raw_entry in (handoff_requirements or []):
        if not isinstance(raw_entry, dict):
            continue
        part_name = str(raw_entry.get("part_name") or "").strip()
        bound_resource_jid = str(raw_entry.get("bound_resource_jid") or "").strip()
        if not part_name or not bound_resource_jid:
            continue
        entry = {
            "part_name": part_name,
            "bound_resource_jid": bound_resource_jid,
            "grounded_destination_available": bool(
                raw_entry.get("grounded_destination_available", False)
            ),
            "required_for_switch": bool(
                raw_entry.get("required_for_switch", False)
            ),
        }
        pose_status = str(raw_entry.get("pose_status") or "").strip()
        if pose_status:
            entry["pose_status"] = pose_status
        current_location = raw_entry.get("current_location")
        if current_location not in (None, ""):
            entry["current_location_summary"] = _bridge_generalize_location_summary(
                current_location
            )
        reason = str(raw_entry.get("reason") or "").strip()
        if reason:
            entry["reason_summary"] = _bridge_none_prompt_text_redact(reason)
        handoffs.append(entry)

    return {
        "executor_bindings": bindings,
        "handoff_requirements": handoffs,
    }


def build_bridge_turn_prompt(
    *,
    session_id: str,
    turn_index: int,
    max_turns: int,
    phase: str,
    focused_resource_jid: str,
    stuck_state: dict[str, Any],
    goal_state: str,
    pending_parts: list[str],
    obligation_targets: list[dict[str, Any]] | None,
    bridge_resources: dict[str, Any] | None,
    grounding_context: dict[str, Any] | None,
    observation_history: list[dict[str, Any]] | None,
    operator_feedback_history: list[str] | None,
    validation_feedback: list[dict[str, Any]] | None,
    marked_reentry_conditions: list[dict[str, Any]] | None,
    unmet_reentry_conditions: list[dict[str, Any]] | None,
    pending_suffix_summary: list[dict[str, Any]] | None,
    last_plan_failure: dict[str, Any] | None,
    allowed_observation_primitives: list[str] | None,
    bridge_outline: list[dict[str, Any]] | None = None,
    approved_bridge_events: list[dict[str, Any]] | None = None,
    infeasible_assignments: list[dict[str, Any]] | None = None,
    executor_bindings: list[dict[str, Any]] | None = None,
    handoff_requirements: list[dict[str, Any]] | None = None,
    bridge_safety_context: dict[str, Any] | None = None,
    draft_final_plan: dict[str, Any] | None = None,
    draft_final_plan_status: dict[str, Any] | None = None,
) -> str:
    """Build one compact ReAct turn prompt for the bridge session."""
    from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.primitive_semantics import (
        build_primitive_reference_card,
        filter_synthesis_primitive_catalog,
    )

    # --- strip primitive catalogs from bridge_resources for the prompt copy ---
    prompt_resources: dict[str, Any] = {}
    all_catalog_entries: list[dict[str, Any]] = []
    for resource_jid, raw_entry in (bridge_resources or {}).items():
        if not isinstance(raw_entry, dict):
            prompt_resources[resource_jid] = raw_entry
            continue
        slimmed = {
            k: v
            for k, v in raw_entry.items()
            if k not in {"primitive_catalog", "execution_primitive_catalog"}
        }
        prompt_resources[resource_jid] = slimmed
        all_catalog_entries.extend(
            filter_synthesis_primitive_catalog(
                raw_entry.get("primitive_catalog")
                or raw_entry.get("execution_primitive_catalog")
                or []
            )
        )

    # deduplicate catalog by primitive name for the reference card
    seen_names: set[str] = set()
    deduped_catalog: list[dict[str, Any]] = []
    for entry in all_catalog_entries:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        if name and name not in seen_names:
            seen_names.add(name)
            deduped_catalog.append(entry)

    primitive_card = build_primitive_reference_card(
        filter_synthesis_primitive_catalog(deduped_catalog)
    )
    prompt_profiles = _bridge_prompt_profiles(
        bridge_resources=bridge_resources,
        primitive_catalog=deduped_catalog,
    )
    phase_token = str(phase or "").strip().lower() or "observe_required"

    prompt_stuck_state: dict[str, Any] = deepcopy(stuck_state or {})
    prompt_obligations = deepcopy(obligation_targets or [])
    prompt_grounding_context = deepcopy(grounding_context or {})
    prompt_observation_history = deepcopy(observation_history or [])
    prompt_operator_feedback = deepcopy(operator_feedback_history or [])
    prompt_validation_feedback = deepcopy(validation_feedback or [])
    prompt_marked_reentry = deepcopy(marked_reentry_conditions or [])
    prompt_unmet_reentry = deepcopy(unmet_reentry_conditions or [])
    prompt_suffix_summary = deepcopy(pending_suffix_summary or [])
    prompt_last_failure = deepcopy(last_plan_failure or {})
    prompt_allowed = list(allowed_observation_primitives or [])
    prompt_bridge_outline = deepcopy(bridge_outline or [])
    prompt_approved_events = deepcopy(approved_bridge_events or [])
    prompt_infeasible_assignments = deepcopy(infeasible_assignments or [])
    prompt_executor_bindings = deepcopy(executor_bindings or [])
    prompt_handoff_requirements = deepcopy(handoff_requirements or [])
    prompt_bridge_safety_context = deepcopy(bridge_safety_context or {})
    prompt_draft_final_plan = deepcopy(draft_final_plan or {})
    prompt_draft_status = deepcopy(draft_final_plan_status or {})
    none_observation_targets = _bridge_none_parts_requiring_observation(
        stuck_state=stuck_state,
        unmet_reentry_conditions=unmet_reentry_conditions,
        observation_history=observation_history,
    )

    prompt_obligations = _bridge_low_bias_prompt_value(
        prompt_obligations,
        strict_none=True,
    )
    prompt_resources = _bridge_low_bias_prompt_value(
        prompt_resources,
        strict_none=True,
    )
    prompt_stuck_state = _bridge_low_bias_prompt_value(
        prompt_stuck_state,
        strict_none=True,
    )
    prompt_grounding_context = _bridge_low_bias_prompt_value(
        prompt_grounding_context,
        strict_none=True,
    )
    prompt_observation_history = _bridge_low_bias_prompt_value(
        prompt_observation_history,
        strict_none=True,
    )
    prompt_marked_reentry = _bridge_low_bias_prompt_conditions(
        prompt_marked_reentry,
        strict_none=True,
    )
    prompt_unmet_reentry = _bridge_low_bias_prompt_conditions(
        prompt_unmet_reentry,
        strict_none=True,
    )
    prompt_suffix_summary = _bridge_low_bias_prompt_value(
        prompt_suffix_summary,
        strict_none=True,
    )
    prompt_infeasible_assignments = _bridge_low_bias_prompt_value(
        prompt_infeasible_assignments,
        strict_none=True,
    )
    prompt_executor_bindings = _bridge_low_bias_prompt_value(
        prompt_executor_bindings,
        strict_none=True,
    )
    prompt_handoff_requirements = _bridge_low_bias_prompt_value(
        prompt_handoff_requirements,
        strict_none=True,
    )
    prompt_bridge_safety_context = _bridge_low_bias_prompt_value(
        prompt_bridge_safety_context,
        strict_none=True,
    )
    prompt_operator_feedback = []
    prompt_validation_feedback = _bridge_none_validation_feedback(prompt_validation_feedback)
    prompt_last_failure = _bridge_none_last_failure_summary(prompt_last_failure)
    prompt_draft_final_plan = {}
    prompt_draft_status = {}
    if phase_token == "observe_required" and none_observation_targets:
        parts_payload = dict(prompt_grounding_context.get("parts") or {})
        for part_name in none_observation_targets:
            part_entry = dict(parts_payload.get(part_name) or {})
            if not part_entry:
                continue
            part_entry["observed_pose"] = None
            parts_payload[part_name] = part_entry
        prompt_grounding_context["parts"] = parts_payload

    obligations_json = json.dumps(prompt_obligations, indent=2)
    resources_json = json.dumps(prompt_resources, indent=2)
    grounding_json = json.dumps(prompt_grounding_context, indent=2)
    observations_json = json.dumps(prompt_observation_history, indent=2)
    feedback_json = json.dumps(prompt_operator_feedback, indent=2)
    validation_json = json.dumps(prompt_validation_feedback, indent=2)
    marked_reentry_json = json.dumps(prompt_marked_reentry, indent=2)
    unmet_json = json.dumps(prompt_unmet_reentry, indent=2)
    suffix_json = json.dumps(prompt_suffix_summary, indent=2)
    last_failure_json = json.dumps(prompt_last_failure, indent=2)
    allowed_json = json.dumps(prompt_allowed, indent=2)
    bridge_outline_json = json.dumps(prompt_bridge_outline, indent=2)
    approved_events_json = json.dumps(prompt_approved_events, indent=2)
    bridge_safety_json = json.dumps(prompt_bridge_safety_context, indent=2)
    draft_final_plan_json = json.dumps(prompt_draft_final_plan, indent=2)
    draft_status_json = json.dumps(prompt_draft_status, indent=2)
    repair_mode = (
        phase_token == "final_plan"
        and isinstance(draft_final_plan_status, dict)
        and str(draft_final_plan_status.get("compile_path", "")).strip() == "llm_repair"
    )
    show_repair_context = False
    repair_draft_section = (
        f"PLANNER-GENERATED FINAL PLAN DRAFT:\n{draft_final_plan_json}"
        if show_repair_context
        else ""
    )
    repair_status_section = (
        f"DRAFT FINAL PLAN STATUS:\n{draft_status_json}"
        if show_repair_context
        else ""
    )
    repair_few_shot_section = (
        _bridge_final_plan_repair_few_shot() if show_repair_context else ""
    )
    if show_repair_context:
        repair_few_shot_section = _bridge_final_plan_repair_few_shot(
            profiles=prompt_profiles
        )

    # --- phase-specific domain context ---
    if phase_token == "observe_required":
        domain_context = _bridge_observe_domain_context()
    elif phase_token == "bridge_outline":
        domain_context = _bridge_events_domain_context(
            bridge_resources=bridge_resources,
            grounding_context=grounding_context,
            unmet_reentry_conditions=unmet_reentry_conditions,
            validation_feedback=validation_feedback,
            bridge_safety_context=bridge_safety_context,
        )
    elif phase_token == "bridge_events":
        domain_context = _bridge_events_domain_context(
            bridge_resources=bridge_resources,
            grounding_context=grounding_context,
            unmet_reentry_conditions=unmet_reentry_conditions,
            validation_feedback=validation_feedback,
            bridge_safety_context=bridge_safety_context,
        )
    else:
        domain_context = _bridge_final_plan_domain_context(
            primitive_card,
            profiles=prompt_profiles,
        )

    if phase_token == "observe_required":
        phase_rules = dedent(
            """\
            - Current planner phase: observe_required.
            - You must return ONE observe request.
            - Choose the observation that most reduces uncertainty using the current state, workspace bounds, and observation history.
            - If any displaced or unknown part still lacks a live part observation, prioritize localizing that part before requesting a resource-pose query.
            - Do not return bridge_events or final_plan in this phase.
            """
        ).strip()
        response_contract = dedent(
            """\
            Observation request:
            {
              "type": "observe",
              "resource_jid": "<jid from AVAILABLE BRIDGE RESOURCES>",
              "primitive": "<one primitive from ALLOWED OBSERVATION PRIMITIVES>",
              "params": {},
              "store_as": "<optional alias for storing the normalized result>",
              "reason_summary": "<optional short rationale>",
              "react_trace": {
                "observed_facts": ["<brief factual observations>"],
                "gap_to_close": ["<what remains uncertain or unresolved>"],
                "decision_basis": ["<why this observation was chosen>"],
                "expected_progress": ["<what this observation should clarify>"]
              }
            }
            """
        ).strip()
    elif phase_token == "bridge_outline":
        phase_rules = dedent(
            """\
            - Current planner phase: bridge_outline.
            - Observation history is available in this prompt.
            - You must return ONE bridge_outline response describing the recovery subproblems and goal milestones that still need to be resolved.
            - Keep the outline at problem/goal level. Do not describe primitive commands, named poses, target-computation helpers, or low-level motion choices here.
            - Prefer abstract milestones such as restoring a blocked resource, satisfying a safety precondition, freeing a capable manipulator, recovering a misplaced part, or restoring resumable state.
            - Do not commit to a specific resource_jid or operation_family unless the current state already makes that choice unavoidable.
            - Distinguish the resource that must be restored from the resource that should execute part recovery; they may be different.
            - If a resource/part recovery assignment has already been ruled infeasible, do not anchor the outline around reusing it.
            - Use BRIDGE CONTRACT TARGETS as the exact closure criteria, but keep the outline abstract and subproblem-oriented.
            - Use the provided state, safety, workspace, and continuation context to infer the needed sequence.
            - Do not return observe, bridge_events, or final_plan in this phase.
            """
        ).strip()
        response_contract = dedent(
            """\
            Bridge outline:
            {
              "type": "bridge_outline",
              "steps": [
                {
                  "step_name": "<short abstract milestone name>",
                  "objective": "<which subproblem or goal this milestone resolves>",
                  "success_signal": "<what should become true after this milestone>",
                  "rationale": "<optional short rationale>"
                }
              ],
              "reason_summary": "<optional short rationale>",
              "react_trace": {
                "observed_facts": ["<brief factual observations>"],
                "gap_to_close": ["<what the outline must eventually resolve>"],
                "decision_basis": ["<why this high-level sequence was chosen>"],
                "expected_progress": ["<what the next bridge-event slices should accomplish>"]
              }
            }

            Notes:
            - Outline unresolved problems/goals first; do not pre-commit to exact resource assignments unless they are already forced.
            - resource_jid, part_name, and operation_family are optional in this phase and may be omitted when still undecided.
            """
        ).strip()
    elif phase_token == "bridge_events":
        phase_rules = dedent(
            """\
            - Current planner phase: bridge_events.
            - Observation history is available in this prompt.
            - Observation history and the current projected bridge state are available in this prompt.
            - Extend the approved bridge prefix from the current projected state instead of rewriting it from scratch.
            - Return only the next bridge-event slice needed to make progress; do not repeat already approved events unless you are explicitly correcting an earlier mistake.
            - The next slice does not need to close the full bridge in one turn, but it must advance the recovery without regressing previously closed conditions.
            - Use only literal symbolic values in expected_resource_delta, expected_part_delta, and closes_conditions. Never place context_ref objects in those fields.
            - Stay at task/event level in this phase. Do not describe primitive commands or primitive-step sequences.
            - Distinguish the resource that must be restored from the resource that should execute part recovery; they may be different.
            - If a resource/part recovery assignment is listed under ruled_out_assignments, do not propose it again unless new observation changes feasibility.
            - Use BRIDGE CONTRACT TARGETS as the exact symbolic closure criteria for this phase.
            - If ACTIVE EXECUTOR SUMMARY shows that a part is already assigned to an executor, do not switch executors for that part unless a grounded handoff or new observation makes the reassignment valid.
            - If BRIDGE CONTRACT TARGETS are already closed but MODELED CONTINUATION GAP is still present, do not restart the bridge from scratch; add only the extra bridge events needed to restore a resumable modeled continuation.
            - If you intend to relocate a resource into or out of a protected/shared region, include projected_effects.occupancy.location so the projected bridge state reflects that move.
            - You must return ONE bridge_events response that resolves the disruption using only the provided state, safety, workspace, and continuation context.
            - Each event MUST include explicit operation_family.
            - Each event MUST include expected_resource_delta with from/to states.
            - Include expected_part_delta when the event changes a part's state.
            - Do not return observe, bridge_outline, or final_plan in this phase.
            """
        ).strip()
        response_contract = dedent(
            """\
            Bridge event proposal:
            {
              "type": "bridge_events",
              "events": [
                {
                  "event_name": "<DES-style bridge controllable event name>",
                  "resource_jid": "<resource that realizes this bridge event>",
                  "operation_family": "<clear|home|pick|stage|place|assemble|pick_place or other supported family>",
                  "part_name": "<optional canonical part name>",
                  "expected_resource_delta": {
                    "from": "<resource state before this event>",
                    "to": "<resource state after this event>"
                  },
                  "expected_part_delta": {
                    "part_name": "<canonical part name>",
                    "from": "<part state before>",
                    "to": "<part state after>",
                    "location_to": "<optional destination location>"
                  },
                  "projected_effects": {
                    "occupancy": {
                      "location": "<optional explicit projected resource location after this event>"
                    }
                  },
                  "closes_conditions": [
                    {
                      "entity_kind": "<resource|part>",
                      "entity": "<entity id>",
                      "field": "<field name>",
                      "expected": "<expected value>"
                    }
                  ],
                  "rationale": "<optional short rationale>"
                }
              ],
              "reason_summary": "<optional short rationale>",
              "react_trace": {
                "observed_facts": ["<brief factual observations>"],
                "gap_to_close": ["<abstract requirements this event set should close>"],
                "decision_basis": ["<why these resources/events were chosen>"],
                "expected_progress": ["<what should be true after these events>"]
              }
            }

            Notes:
            - operation_family is required on every event; do not rely on name inference.
            - expected_resource_delta is required on every event.
            - expected_part_delta is null for events that do not touch a part.
            - closes_conditions remains the authoritative bridge-event meaning.
            - In incremental bridge mode, return only new events that extend the approved prefix.
            """
        ).strip()
    else:
        phase_rules = dedent(
            """\
            - Current planner phase: final_plan.
            - APPROVED BRIDGE EVENTS are authoritative and must be realized in order.
            - You must return ONE final_plan.
            - Do not return observe or bridge_events in this phase.
            """
        ).strip()
        if show_repair_context:
            phase_rules += "\n- The planner already produced a draft final_plan. Repair the draft instead of rewriting the bridge from scratch."
        response_contract = dedent(
            f"""\
            Final plan:
            {{
              "type": "final_plan",
              "plan": {{
                "primary_obligation": {{
                  "rule_id": "<one rule_id from ACTIVE OBLIGATION TARGETS>",
                  "resource_jid": "<matching resource_jid>"
                }},
                "bridge_event_summary": [
                  {{
                    "event_name": "<DES-style bridge controllable event name>",
                    "resource_jid": "<resource that realizes this bridge event>",
                    "part_name": "<optional canonical part name>",
                    "closes_conditions": [
                      {{
                        "entity_kind": "<resource|part>",
                        "entity": "<entity id>",
                        "field": "<field name>",
                        "expected": "<expected value>"
                      }}
                    ],
                    "rationale": "<optional short rationale>"
                  }}
                ],
                "macro_tasks": [
                  {{
                    "resource_jid": "{focused_resource_jid}",
                    "macro_name": "<descriptive recovery macro name>",
                    "description": "<short description>",
                    "rationale": "<why this helps restore continuation>",
                    "expected_start_state": "<projected current state for this resource>",
                    "part_name": "<optional canonical part name>",
                    "task_params": {{}},
                    "task_metadata": {{
                      "in_state": "<resource state before macro>",
                      "out_state": "<resource state after macro>",
                      "required_context_keys": [],
                      "context_mapping": {{}},
                      "part_transition": null
                    }},
                    "primitive_steps": [
                      {{
                        "primitive": "<controller primitive name>",
                        "params": {{}},
                        "store_as": "<optional alias; observation primitives only>"
                      }}
                    ]
                  }}
                ]
              }},
              "reason_summary": "<optional short rationale>",
              "react_trace": {{
                "observed_facts": ["<brief factual observations>"],
                "gap_to_close": ["<what the final plan still had to realize>"],
                "decision_basis": ["<why this final plan structure was chosen>"],
                "expected_progress": ["<what should hold after execution>"]
              }}
            }}
            """
        ).strip()

    role_summary_json = json.dumps(
        _bridge_none_resource_role_summary(
            focused_resource_jid=focused_resource_jid,
            bridge_resources=prompt_resources,
            infeasible_assignments=prompt_infeasible_assignments,
        ),
        indent=2,
    )
    contract_targets_json = json.dumps(
        _bridge_none_contract_targets(
            unmet_reentry_conditions=unmet_reentry_conditions,
        ),
        indent=2,
    )
    active_executor_json = json.dumps(
        _bridge_none_active_executor_summary(
            executor_bindings=prompt_executor_bindings,
            handoff_requirements=prompt_handoff_requirements,
        ),
        indent=2,
    )
    session_rules = dedent(
        """\
        Session rules:
        - DES could not find a modeled continuation.
        - Do not output explanations outside JSON.
        - Do not invent new primitives, resources, context keys, or coordinates.
        - Use context_ref objects into GROUNDING CONTEXT when values are already available there.
        - Prefer string context_ref paths such as "/step_outputs/<alias>/approach_pose/x" or "parts.<PART>.observed_pose.x".
        - Mid-loop actuation is not allowed in this phase. Only the listed observation/generation primitives may be requested.
        - Symbolic station names, exact continuation targets, and modeled suffix details may be abstracted unless surfaced explicitly in the contract/context sections below.
        - Infer the needed bridge from current state, observation history, workspace limits, resource snapshots, and the continuation context summary.
        - The bridge must leave the system safe and able to resume continuation.
        - If current information is insufficient, request an observation before committing to bridge_events.
        - In final_plan primitive_steps, use "store_as" ONLY on these primitives: detect_parts, get_current_pose, compute_pick_targets, compute_place_targets.
        - Never include "store_as" on action primitives such as move_to_named_pose, move_relative, move_cartesian, move_pose, grasp_part, release_part, pause_job, resume_job, or cancel_job.
        """
    ).strip()
    session_metadata = dedent(
        f"""\
        SESSION:
        - session_id: {session_id}
        - turn: {turn_index}/{max_turns}
        - phase: {phase_token}
        - focused_resource_jid: {focused_resource_jid}
        """
    ).strip()
    continuation_context_json = json.dumps(
        _bridge_none_continuation_summary(
            pending_suffix_summary=pending_suffix_summary,
            unmet_reentry_conditions=unmet_reentry_conditions,
        ),
        indent=2,
    )
    observation_need_section = ""
    if phase_token == "observe_required":
        observation_need_section = dedent(
            f"""\
            OBSERVATION NEED SUMMARY:
            {json.dumps({
                "critical_parts_requiring_live_observation": none_observation_targets,
                "observation_goal": "localize displaced bridge-relevant parts before proposing bridge events",
            }, indent=2)}
            """
        ).strip()
    continuation_sections = dedent(
        f"""\
        {observation_need_section}

        CONTINUATION CONTEXT SUMMARY:
        {continuation_context_json}

        BRIDGE CONTRACT TARGETS:
        {contract_targets_json}

        RESOURCE ROLE SUMMARY:
        {role_summary_json}

        ACTIVE EXECUTOR SUMMARY:
        {active_executor_json}
        """
    ).strip()

    operator_guidance_section = dedent(
        f"""\
        OPERATOR GUIDANCE HISTORY:
        {feedback_json}
        """
    ).strip()
    operator_guidance_section = ""

    last_failure_heading = "LAST FAILURE SUMMARY"
    last_failure_section = dedent(
        f"""\
        {last_failure_heading}:
        {last_failure_json}
        """
    ).strip()
    if prompt_last_failure == {}:
        last_failure_section = ""
    modeled_continuation_gap_section = ""
    modeled_gap = (
        prompt_last_failure.get("modeled_continuation_gap")
        if isinstance(prompt_last_failure, dict)
        else None
    )
    if isinstance(modeled_gap, dict) and modeled_gap:
        modeled_continuation_gap_section = dedent(
            f"""\
            MODELED CONTINUATION GAP:
            {json.dumps(modeled_gap, indent=2)}
            """
        ).strip()

    return dedent(
        f"""\
        You are the bridge recovery planner for a bounded multi-turn ReAct session.

        {session_rules}
        {phase_rules}

        {domain_context}

        {session_metadata}

        CURRENT DISRUPTED SEARCH STATE:
        {json.dumps(prompt_stuck_state, indent=2)}

        ACTIVE OBLIGATION TARGETS:
        {obligations_json}

        AVAILABLE BRIDGE RESOURCES:
        {resources_json}

        GROUNDING CONTEXT:
        {grounding_json}

        PRIOR OBSERVATIONS:
        {observations_json}

        {operator_guidance_section}

        VALIDATION FEEDBACK FROM PRIOR FINAL PLAN ATTEMPTS:
        {validation_json}

        {continuation_sections}

        {last_failure_section}

        {modeled_continuation_gap_section}

        CURRENT BRIDGE OUTLINE:
        {bridge_outline_json}

        APPROVED BRIDGE EVENTS:
        {approved_events_json}

        BRIDGE SAFETY CONTEXT:
        {bridge_safety_json}

        ALLOWED OBSERVATION PRIMITIVES:
        {allowed_json}

        {repair_draft_section}

        {repair_status_section}

        {repair_few_shot_section}

        Return exactly one JSON object in the allowed shape for the current phase.

        {response_contract}

        Return ONLY the JSON object.
        """
    )
