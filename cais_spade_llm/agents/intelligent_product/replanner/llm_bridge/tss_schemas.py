"""TSS v3 bridge schemas: structured response format and projection tool definition.

These schemas enforce the v3 repair turn response structure at LLM generation time
(OpenAI ``response_format`` with ``type: "json_schema"``) and define the projection
tool that the LLM can call mid-turn to simulate primitive sequences.

Note: ``strict`` is set to ``False`` because the ``function_defs`` and ``steps``
fields contain LLM-synthesized free-form objects whose schemas cannot be fully
enumerated at definition time.  The ``parse_structured_response`` function
performs runtime validation instead.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Structured response schema (OpenAI response_format)
# ---------------------------------------------------------------------------

REPAIR_TURN_RESPONSE_SCHEMA: dict[str, Any] = {
    "name": "repair_turn_response",
    "strict": False,
    "schema": {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": ["observe", "repair_outline", "repair_program"],
            },
            "reasoning": {
                "type": "object",
                "description": (
                    "Mandatory structured planning analysis before proposing "
                    "actions"
                ),
                "properties": {
                    "current_state_analysis": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "One entry per resource: current state, held parts, "
                            "location"
                        ),
                    },
                    "goal_gap_analysis": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "One entry per obligation: what must change"
                        ),
                    },
                    "blocked_transitions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "transition": {"type": "string"},
                                "affected_entities": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "why_state_is_insufficient": {"type": "string"},
                                "requires_observation": {"type": "boolean"},
                                "smallest_observation_batch": {"type": "integer"},
                            },
                            "required": [
                                "transition",
                                "affected_entities",
                                "why_state_is_insufficient",
                            ],
                        },
                        "description": (
                            "One entry per currently blocked executable transition."
                        ),
                    },
                    "abstract_repair_order": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "phase_type": {
                                    "type": "string",
                                    "enum": [
                                        "resolve_safety",
                                        "restore_capability",
                                        "free_executor",
                                        "recover_entities",
                                        "restore_resume_entry",
                                        "adapt_goals",
                                        "replace_suffix",
                                        "resume_modeled_suffix",
                                    ],
                                },
                                "objective": {"type": "string"},
                                "target_entities": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "advances_obligations": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": [
                                "phase_type",
                                "objective",
                                "target_entities",
                                "advances_obligations",
                            ],
                        },
                        "description": (
                            "Task/state-level repair outline before primitive refinement."
                        ),
                    },
                    "outline_actions": {
                        "anyOf": [
                            {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "action_id": {"type": "string"},
                                        "phase_type": {"type": "string"},
                                        "objective": {"type": "string"},
                                        "target_entities": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "advances_obligations": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                        "must_complete_before": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                    },
                                },
                            },
                            {"type": "null"},
                        ],
                        "description": (
                            "Named task-level repair actions. Prefer explicit action_id values in repair_outline; "
                            "the runtime can derive fallback names if omitted."
                        ),
                    },
                    "transition_plan": {
                        "anyOf": [
                            {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "step": {"type": "string"},
                                        "resource": {"type": "string"},
                                        "from_state": {"type": "string"},
                                        "to_state": {"type": "string"},
                                        "primitive": {"type": "string"},
                                        "obligation_advanced": {"type": "string"},
                                        "safety_note": {"type": "string"},
                                    },
                                    "required": [
                                        "step",
                                        "resource",
                                        "from_state",
                                        "to_state",
                                        "primitive",
                                        "obligation_advanced",
                                        "safety_note",
                                    ],
                                },
                            },
                            {"type": "null"},
                        ],
                        "description": (
                            "Legacy optional primitive narration. Accepted for "
                            "backward compatibility but ignored during canonical "
                            "repair_program normalization."
                        ),
                    },
                    "safety_check": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "One entry per safety rule: why plan does not "
                            "violate it"
                        ),
                    },
                },
                "required": [
                    "safety_check",
                ],
            },
            "function_defs": {
                "anyOf": [
                    {"type": "array", "items": {}},
                    {"type": "null"},
                ],
                "description": (
                    "SynthesizedTaskFn definitions (null for observe/repair_outline)"
                ),
            },
            "steps": {
                "anyOf": [
                    {"type": "array", "items": {}},
                    {"type": "null"},
                ],
                "description": (
                    "RepairStep sequence (null for observe/repair_outline)"
                ),
            },
            "success_conditions": {
                "anyOf": [
                    {"type": "array", "items": {}},
                    {"type": "null"},
                ],
                "description": (
                    "Reachability conditions (null for observe/repair_outline)"
                ),
            },
            "rationale": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "null"},
                ],
                "description": "One-line summary",
            },
            "observe_request": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {
                            "semantic_operation": {"type": "string"},
                            "target_entity": {"type": "string"},
                            "resource_jid": {"type": "string"},
                            "primitive": {"type": "string"},
                            "params": {"type": "object"},
                            "store_as": {"type": "string"},
                        },
                    },
                    {"type": "null"},
                ],
                "description": (
                    "For observe type: {resource_jid, primitive, params, "
                    "store_as}"
                ),
            },
            "observe_requests": {
                "anyOf": [
                    {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 3,
                        "items": {
                            "type": "object",
                            "properties": {
                                "semantic_operation": {"type": "string"},
                                "target_entity": {"type": "string"},
                                "resource_jid": {"type": "string"},
                                "primitive": {"type": "string"},
                                "params": {"type": "object"},
                                "store_as": {"type": "string"},
                            },
                        },
                    },
                    {"type": "null"},
                ],
                "description": (
                    "For observe type: an ordered list of 1-3 observation "
                    "requests. Legacy observe_request is still accepted "
                    "during migration."
                ),
            },
        },
        "required": ["type", "reasoning"],
    },
}


# ---------------------------------------------------------------------------
# Projection tool definition (OpenAI tools parameter)
# ---------------------------------------------------------------------------

PROJECT_PRIMITIVE_SEQUENCE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "project_primitive_sequence",
        "description": (
            "Simulate a sequence of primitives on a resource and return the "
            "projected state snapshot after all steps. Use this to verify your "
            "plan before committing. Returns {is_valid, projected_snapshot, "
            "error}."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "resource_jid": {
                    "type": "string",
                    "description": (
                        "The resource JID to simulate on "
                        "(e.g., 'ur5e@localhost')"
                    ),
                },
                "steps": {
                    "type": "array",
                    "description": "Ordered list of primitives to simulate",
                    "items": {
                        "type": "object",
                        "properties": {
                            "primitive": {
                                "type": "string",
                                "description": "Primitive name",
                            },
                            "params": {
                                "type": "object",
                                "description": "Primitive parameters",
                            },
                        },
                        "required": ["primitive", "params"],
                    },
                },
            },
            "required": ["resource_jid", "steps"],
        },
    },
}


# ---------------------------------------------------------------------------
# Repair-program normalization helpers
# ---------------------------------------------------------------------------

def _normalize_function_def(raw: dict[str, Any]) -> dict[str, Any]:
    resource_constraints = dict(raw.get("resource_constraints") or {})
    resource_jid = str(raw.get("resource_jid") or "").strip()
    primitive_program = list(
        raw.get("primitive_program") or raw.get("primitives") or []
    )
    if not resource_jid:
        primitive_resources = {
            str(step.get("resource_jid") or "").strip()
            for step in primitive_program
            if isinstance(step, dict) and str(step.get("resource_jid") or "").strip()
        }
        if len(primitive_resources) == 1:
            resource_jid = next(iter(primitive_resources))
    if resource_jid and "resource_jid" not in resource_constraints:
        resource_constraints["resource_jid"] = resource_jid
    return {
        "name": raw.get("name") or raw.get("function_name") or "",
        "intent": raw.get("intent") or raw.get("description") or "",
        "resource_constraints": resource_constraints,
        "inputs": dict(raw.get("inputs") or {}),
        "preconditions": dict(raw.get("preconditions") or {}),
        "effects": dict(raw.get("effects") or {}),
        "primitive_program": primitive_program,
        "expected_post_state": dict(raw.get("expected_post_state") or {}),
    }


def _normalize_success_condition(raw: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(raw)
    if "expected" not in normalized and "value" in normalized:
        normalized["expected"] = normalized.get("value")
    entity = str(normalized.get("entity") or "").strip()
    resource_alias = str(
        normalized.get("resource_jid")
        or normalized.get("resource")
        or ""
    ).strip()
    part_alias = str(normalized.get("part") or "").strip()
    if not entity and resource_alias:
        normalized["entity"] = resource_alias
        normalized["entity_kind"] = "resource"
        entity = resource_alias
    elif not entity and part_alias:
        normalized["entity"] = part_alias
        normalized["entity_kind"] = "part"
        entity = part_alias
    elif entity and "entity_kind" not in normalized:
        normalized["entity_kind"] = "part"
    return normalized


def _normalize_repair_step(
    raw: dict[str, Any],
    *,
    fn_resource_lookup: dict[str, str],
) -> dict[str, Any]:
    if raw.get("resume_suffix") is True:
        return {"kind": "resume_suffix", "payload": {}}

    fn_name = str(raw.get("fn") or "").strip()
    if fn_name:
        resource_jid = str(raw.get("resource_jid") or "").strip()
        if not resource_jid:
            resource_jid = fn_resource_lookup.get(fn_name, "")
        return {
            "kind": "call_function",
            "payload": {
                "function_name": fn_name,
                "resource_jid": resource_jid,
                "args": dict(raw.get("args") or {}),
            },
        }

    kind = str(raw.get("kind") or "").strip()
    payload = dict(raw.get("payload") or {})
    if kind in ("execute_function", "execute"):
        kind = "call_function"
    if kind == "call_function":
        if "function_name" not in payload and raw.get("function_name"):
            payload["function_name"] = raw.get("function_name")
        fn_name = str(payload.get("function_name") or "").strip()
        if "resource_jid" not in payload:
            resource_jid = str(raw.get("resource_jid") or "").strip()
            if not resource_jid and fn_name:
                resource_jid = fn_resource_lookup.get(fn_name, "")
            if resource_jid:
                payload["resource_jid"] = resource_jid
        if "args" not in payload:
            payload["args"] = dict(raw.get("args") or {})
        return {"kind": kind, "payload": payload}

    if kind == "wait":
        if "until" not in payload and isinstance(raw.get("until"), dict):
            payload["until"] = dict(raw.get("until") or {})
        return {"kind": kind, "payload": payload}

    if kind == "task_mutation":
        if not payload:
            payload = {
                key: raw.get(key)
                for key in ("mutation_type", "target_task_ids", "payload")
                if key in raw
            }
        return {"kind": kind, "payload": payload}

    if kind == "resume_suffix":
        return {"kind": kind, "payload": {}}

    return dict(raw)


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_structured_response(
    raw: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    """Parse and lightly validate a v3 structured response from the LLM.

    Parameters
    ----------
    raw:
        The dict returned by ``json.loads(choice.message.content)`` after a
        structured-output LLM call.

    Returns
    -------
    tuple:
        ``(parsed_dict, error_or_None)``.  On success the parsed dict contains
        at minimum ``type`` and ``reasoning``.  On failure the dict is the
        original ``raw`` and the error string explains what went wrong.
    """
    if not isinstance(raw, dict):
        return {}, f"expected dict, got {type(raw).__name__}"

    response_type = str(raw.get("type", "")).strip().lower()
    if response_type not in ("observe", "repair_outline", "repair_program"):
        return raw, f"invalid response type: {raw.get('type')!r}"

    reasoning = raw.get("reasoning")
    if not isinstance(reasoning, dict):
        return raw, "missing or invalid 'reasoning' object"
    required_reasoning_fields = (
        "safety_check",
    )
    if response_type == "observe":
        required_reasoning_fields = (
            "blocked_transitions",
            *required_reasoning_fields,
        )
    if response_type == "observe":
        extra_fields = [
            "abstract_repair_order",
            "transition_plan",
        ]
        extra_fields.insert(0, "goal_gap_analysis")
        required_reasoning_fields = (
            *required_reasoning_fields,
            *extra_fields,
        )
    elif response_type == "repair_program":
        required_reasoning_fields = required_reasoning_fields
    missing_reasoning_fields = [
        field for field in required_reasoning_fields
        if field not in reasoning
    ]
    if missing_reasoning_fields:
        return raw, (
            "reasoning object missing required fields: "
            + ", ".join(missing_reasoning_fields)
        )

    if response_type == "observe":
        normalized = dict(raw)
        observe_requests = normalized.get("observe_requests")
        if observe_requests is None:
            legacy_request = normalized.get("observe_request")
            if isinstance(legacy_request, dict):
                observe_requests = [legacy_request]
            else:
                observe_requests = None
        if not isinstance(observe_requests, list) or not observe_requests:
            return raw, (
                "observe response missing 'observe_requests' array "
                "(or legacy 'observe_request' object)"
            )
        if len(observe_requests) > 3:
            return raw, "observe_requests may contain at most 3 requests"

        normalized_requests: list[dict[str, Any]] = []
        for index, observe_request in enumerate(observe_requests, start=1):
            if not isinstance(observe_request, dict):
                return raw, f"observe_requests[{index}] must be an object"
            semantic_operation = str(
                observe_request.get("semantic_operation") or ""
            ).strip()
            target_entity = str(observe_request.get("target_entity") or "").strip()
            params = observe_request.get("params") or {}
            if not isinstance(params, dict):
                params = {}
            if not target_entity:
                target_entity = str(
                    params.get("part_name")
                    or params.get("target_entity")
                    or params.get("resource_jid")
                    or ""
                ).strip()
            has_concrete_binding = bool(
                observe_request.get("resource_jid")
                and observe_request.get("primitive")
            )
            if not has_concrete_binding and not semantic_operation:
                return raw, (
                    f"observe_requests[{index}] must provide either "
                    "'resource_jid'+'primitive' or 'semantic_operation'"
                )
            normalized_requests.append({
                "semantic_operation": semantic_operation,
                "target_entity": target_entity,
                "resource_jid": observe_request.get("resource_jid"),
                "primitive": observe_request.get("primitive"),
                "params": params,
                "store_as": observe_request.get("store_as", ""),
            })
        normalized["observe_requests"] = normalized_requests
        if normalized_requests:
            normalized["observe_request"] = dict(normalized_requests[0])
        return normalized, None

    elif response_type == "repair_outline":
        transition_plan = reasoning.get("transition_plan")
        if transition_plan not in (None, []) and not (
            isinstance(transition_plan, list) and len(transition_plan) == 0
        ):
            return raw, (
                "repair_outline must not emit primitive-level reasoning.transition_plan "
                "(use null or [])"
            )
        if raw.get("function_defs") not in (None, []):
            return raw, "repair_outline must not include function_defs"
        if raw.get("steps") not in (None, []):
            return raw, "repair_outline must not include steps"
        if raw.get("success_conditions") not in (None, []):
            return raw, "repair_outline must not include success_conditions"
        normalized = dict(raw)
        normalized_reasoning = dict(reasoning)
        if "blocked_transitions" not in normalized_reasoning or not isinstance(
            normalized_reasoning.get("blocked_transitions"), list
        ):
            normalized_reasoning["blocked_transitions"] = []
        if "current_state_analysis" not in normalized_reasoning or not isinstance(
            normalized_reasoning.get("current_state_analysis"), list
        ):
            normalized_reasoning["current_state_analysis"] = []
        if "goal_gap_analysis" not in normalized_reasoning or not isinstance(
            normalized_reasoning.get("goal_gap_analysis"), list
        ):
            normalized_reasoning["goal_gap_analysis"] = []
        if "abstract_repair_order" not in normalized_reasoning or not isinstance(
            normalized_reasoning.get("abstract_repair_order"), list
        ):
            normalized_reasoning["abstract_repair_order"] = []
        if "outline_actions" not in normalized_reasoning or not isinstance(
            normalized_reasoning.get("outline_actions"), list
        ):
            normalized_reasoning["outline_actions"] = []
        normalized["reasoning"] = normalized_reasoning
        normalized["function_defs"] = []
        normalized["steps"] = []
        normalized["success_conditions"] = []
        normalized["rationale"] = str(raw.get("rationale") or "")
        return normalized, None

    elif response_type == "repair_program":
        if not isinstance(raw.get("function_defs"), list):
            return raw, "repair_program missing 'function_defs' array"
        if not isinstance(raw.get("steps"), list):
            return raw, "repair_program missing 'steps' array"
        normalized = dict(raw)
        normalized_reasoning = dict(reasoning)
        normalized_reasoning.pop("transition_plan", None)
        normalized_fn_defs: list[dict[str, Any]] = []
        fn_resource_lookup: dict[str, str] = {}
        for index, fn_def in enumerate(raw.get("function_defs") or [], start=1):
            if not isinstance(fn_def, dict):
                return raw, f"function_defs[{index}] must be an object"
            normalized_fn = _normalize_function_def(fn_def)
            if not str(normalized_fn.get("name") or "").strip():
                return raw, f"function_defs[{index}] missing 'name'"
            primitive_program = normalized_fn.get("primitive_program")
            if not isinstance(primitive_program, list) or not primitive_program:
                return raw, (
                    f"function_defs[{index}] missing 'primitive_program' "
                    "(legacy alias 'primitives' is accepted)"
                )
            resource_constraints = dict(normalized_fn.get("resource_constraints") or {})
            resource_jid = str(resource_constraints.get("resource_jid") or "").strip()
            if resource_jid:
                fn_resource_lookup[str(normalized_fn["name"]).strip()] = resource_jid
            normalized_fn_defs.append(normalized_fn)

        normalized_steps: list[dict[str, Any]] = []
        for index, step in enumerate(raw.get("steps") or [], start=1):
            if not isinstance(step, dict):
                return raw, f"steps[{index}] must be an object"
            normalized_step = _normalize_repair_step(
                step,
                fn_resource_lookup=fn_resource_lookup,
            )
            kind = str(normalized_step.get("kind") or "").strip()
            if not kind:
                return raw, f"steps[{index}] missing 'kind'"
            if kind == "call_function":
                payload = normalized_step.get("payload") or {}
                if not str(payload.get("function_name") or "").strip():
                    return raw, f"steps[{index}] missing call_function payload.function_name"
                if not str(payload.get("resource_jid") or "").strip():
                    return raw, f"steps[{index}] missing call_function payload.resource_jid"
            normalized_steps.append(normalized_step)

        normalized["function_defs"] = normalized_fn_defs
        normalized["steps"] = normalized_steps
        normalized["reasoning"] = normalized_reasoning
        normalized["success_conditions"] = [
            _normalize_success_condition(condition)
            for condition in (raw.get("success_conditions") or [])
            if isinstance(condition, dict)
        ]
        normalized["rationale"] = str(raw.get("rationale") or "")
        return normalized, None

    return raw, None
