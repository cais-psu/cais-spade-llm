from __future__ import annotations

"""Prompt builders for the active multi-turn recovery mode."""

import json
import re
from copy import deepcopy
from typing import Any

from cais_spade_llm.resources.resource_primitives import (
    filter_synthesis_primitive_catalog,
)
from cais_spade_llm.resources.resource_profile import get_resource_profile

_DEFAULT_CANDIDATE_BOUND = 5
_ONE_STEP_CANDIDATE_COUNT = 3


# ---------------------------------------------------------------------------
# Phase titles
# ---------------------------------------------------------------------------

_PHASE_TITLES: dict[str, str] = {
    "grounding": "Observation / State Estimation",
    "outline": "Recovery Event Synthesis",
    "primitive_generation": "Event-to-Primitive Grounding",
    "finalize": "Executable Recovery Trace Finalization",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _task_resource_jid(task: dict[str, Any]) -> str:
    return str(task.get("resource_jid") or task.get("resource_binding") or "").strip()


def _task_part_name(task: dict[str, Any]) -> str:
    return str(task.get("part_name") or "").strip()


def _task_target_ref(task: dict[str, Any]) -> str:
    action_target = dict(task.get("action_target") or {})
    end_state = dict(task.get("expected_end_state") or {})
    return str(
        task.get("target_ref")
        or action_target.get("target_ref")
        or action_target.get("target_location")
        or end_state.get("part_location")
        or ""
    ).strip()


def _task_source_ref(task: dict[str, Any]) -> str:
    action_target = dict(task.get("action_target") or {})
    start_state = dict(task.get("expected_start_state") or {})
    return str(
        task.get("source_ref")
        or action_target.get("source_ref")
        or action_target.get("source_location")
        or start_state.get("part_location")
        or ""
    ).strip()


def _task_description(task: dict[str, Any]) -> str:
    return str(task.get("description") or task.get("rationale") or "").strip()


def _outline_task_view(task: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(task, dict):
        return {}
    view = deepcopy(task)
    if "resource_jid" not in view or not str(view.get("resource_jid") or "").strip():
        resource_jid = _task_resource_jid(task)
        if resource_jid:
            view["resource_jid"] = resource_jid
    if "part_name" not in view or not str(view.get("part_name") or "").strip():
        part_name = _task_part_name(task)
        if part_name:
            view["part_name"] = part_name
    if "target_ref" not in view or not str(view.get("target_ref") or "").strip():
        target_ref = _task_target_ref(task)
        if target_ref:
            view["target_ref"] = target_ref
    if "description" not in view or not str(view.get("description") or "").strip():
        description = _task_description(task)
        if description:
            view["description"] = description
    return view


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

_OUTLINE_STATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "resource_state": {"type": "string", "minLength": 1},
        "resource_location": {"type": ["string", "null"]},
        "held_part": {"type": ["string", "null"], "minLength": 1},
        "part_state": {"type": "string", "minLength": 1},
        "part_location": {"type": ["string", "null"]},
    },
    "required": ["resource_state"],
}


_OUTLINE_SYMBOLIC_EVENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "outline_id": {"type": "string"},
        "event_name": {"type": "string", "minLength": 1},
        "resource_jid": {"type": "string"},
        "part_name": {"type": "string", "minLength": 1},
        "expected_end_state": deepcopy(_OUTLINE_STATE_SCHEMA),
        "rationale": {"type": "string"},
    },
    "required": [
        "outline_id",
        "event_name",
        "resource_jid",
        "expected_end_state",
        "rationale",
    ],
    "allOf": [
        {
            "if": {"required": ["part_name"]},
            "then": {},
            "else": {
                "properties": {
                    "expected_end_state": {
                        "allOf": [
                            {"not": {"required": ["held_part"]}},
                            {"not": {"required": ["part_state"]}},
                            {"not": {"required": ["part_location"]}},
                        ]
                    }
                }
            },
        },
        {
            "if": {
                "properties": {
                    "expected_end_state": {"required": ["held_part"]}
                }
            },
            "then": {
                "properties": {
                    "expected_end_state": {
                        "required": [
                            "resource_state",
                            "held_part",
                            "part_state",
                            "part_location",
                        ]
                    }
                }
            },
        },
    ],
}


def _json_schema_types_for_domain(domain: list[Any]) -> list[str]:
    types: set[str] = set()
    for value in domain:
        if value is None:
            types.add("null")
        elif isinstance(value, bool):
            types.add("boolean")
        elif isinstance(value, int):
            types.add("integer")
        elif isinstance(value, float):
            types.add("number")
        elif isinstance(value, dict):
            types.add("object")
        elif isinstance(value, list):
            types.add("array")
        else:
            types.add("string")
    return sorted(types) or ["string", "number", "boolean", "object", "array", "null"]


def _outline_state_field_schema(
    field_name: str,
    declaration: dict[str, Any],
) -> dict[str, Any]:
    """Return the response-schema type for one public RA state variable."""
    declaration = dict(declaration or {})
    if "type" in declaration:
        declared_type = declaration["type"]
        type_names = [declared_type] if isinstance(declared_type, str) else declared_type
        if (
            not isinstance(type_names, list)
            or not type_names
            or any(
                not isinstance(name, str)
                or name not in {"string", "number", "integer", "boolean", "array", "object", "null"}
                for name in type_names
            )
            or len(set(type_names)) != len(type_names)
        ):
            raise ValueError(f"Invalid declared type for outline field {field_name!r}: {declared_type!r}")
        # A recorded domain can be incomplete; it cannot narrow a declared type.
        schema = {"type": deepcopy(declared_type)}
        if field_name in {"resource_state", "part_state", "held_part"} and "string" in type_names:
            schema["minLength"] = 1
        return schema
    domain = list(declaration.get("domain") or [])
    if field_name in {"resource_state", "part_state"}:
        return {"type": "string", "minLength": 1}
    if field_name == "held_part":
        return {"type": ["string", "null"], "minLength": 1}
    return {"type": _json_schema_types_for_domain(domain)}


def _outline_symbolic_event_schema(
    declared_state_variables: dict[str, dict[str, Any]] | None,
) -> dict[str, Any]:
    if not declared_state_variables:
        return deepcopy(_OUTLINE_SYMBOLIC_EVENT_SCHEMA)
    state_schema = deepcopy(_OUTLINE_STATE_SCHEMA)
    for field_name, declaration in sorted(declared_state_variables.items()):
        token = str(field_name or "").strip()
        if not token:
            continue
        state_schema["properties"][token] = _outline_state_field_schema(
            token,
            dict(declaration or {}),
        )
    event_schema = deepcopy(_OUTLINE_SYMBOLIC_EVENT_SCHEMA)
    event_schema["properties"]["expected_end_state"] = deepcopy(state_schema)
    part_scoped_fields = {
        str(field_name)
        for field_name, declaration in declared_state_variables.items()
        if str(dict(declaration or {}).get("scope") or "resource") == "part"
    }
    forbidden_without_part = sorted({"held_part", *part_scoped_fields})
    event_schema["allOf"] = [
        {
            "if": {"required": ["part_name"]},
            "then": {},
            "else": {
                "properties": {
                    "expected_end_state": {
                        "allOf": [
                            {"not": {"required": [field_name]}}
                            for field_name in forbidden_without_part
                        ]
                    }
                }
            },
        },
        {
            "if": {
                "properties": {
                    "expected_end_state": {"required": ["held_part"]}
                }
            },
            "then": {
                "properties": {
                    "expected_end_state": {
                        "required": [
                            "resource_state",
                            "held_part",
                            "part_state",
                            "part_location",
                        ]
                    }
                }
            },
        },
    ]
    return event_schema


def _outline_candidate_schema_definitions(  # noqa: C901 - retain historical schema branches beside declared-field validation
    declared_state_variables: dict[str, dict[str, Any]] | None,
    declared_state_variables_by_resource_jid: (
        dict[str, dict[str, dict[str, Any]]] | None
    ) = None,
    *, declared_fields_only: bool = False,
) -> dict[str, dict[str, Any]]:
    """Return mutually exclusive schemas for recovery candidate shapes."""
    if declared_state_variables_by_resource_jid:
        resource_event_branches: list[dict[str, Any]] = []
        state_definitions: dict[str, dict[str, Any]] = {}
        state_definition_names: dict[tuple[str, str], str] = {}
        state_definition_counts: dict[str, int] = {}
        for resource_jid, raw_declarations in sorted(
            declared_state_variables_by_resource_jid.items()
        ):
            declarations = {
                str(field_name): dict(declaration or {})
                for field_name, declaration in dict(raw_declarations or {}).items()
                if str(field_name).strip()
                and dict(declaration or {}).get("private") is not True
                and not str(field_name).startswith("task_ctx.")
                and (not declared_fields_only or not str(field_name).startswith("_"))
            }
            if not declarations or not declared_fields_only and "resource_state" not in declarations:
                continue
            candidate_branches = _resource_candidate_event_schemas(
                resource_jid=str(resource_jid),
                declared_state_variables=declarations,
                declared_fields_only=declared_fields_only,
            )
            for branch_kind, branch in zip(
                ("resource_only", "part", "custody")[
                    : len(candidate_branches)
                ],
                candidate_branches,
                strict=True,
            ):
                state_schema = deepcopy(
                    branch["properties"]["expected_end_state"]
                )
                state_signature = json.dumps(
                    state_schema,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                definition_key = (branch_kind, state_signature)
                definition_name = state_definition_names.get(definition_key)
                if definition_name is None:
                    count = state_definition_counts.get(branch_kind, 0) + 1
                    state_definition_counts[branch_kind] = count
                    definition_name = f"{branch_kind}_outline_state"
                    if count > 1:
                        definition_name = f"{definition_name}_{count}"
                    state_definition_names[definition_key] = definition_name
                    state_definitions[definition_name] = state_schema
                branch["properties"]["expected_end_state"] = {
                    "$ref": f"#/$defs/{definition_name}"
                }
                resource_event_branches.append(branch)
        if resource_event_branches:
            event_properties = deepcopy(
                _OUTLINE_SYMBOLIC_EVENT_SCHEMA["properties"]
            )
            event_properties["part_name"] = {
                "type": ["string", "null"],
                "minLength": 1,
            }
            event_properties["expected_end_state"] = {"type": "object"}
            return {
                **state_definitions,
                "outline_event": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": event_properties,
                    "required": deepcopy(
                        _OUTLINE_SYMBOLIC_EVENT_SCHEMA["required"]
                    ),
                    "anyOf": resource_event_branches,
                }
            }

    if declared_fields_only:
        declarations = {name: row for name, row in (declared_state_variables or {}).items()
                        if not row.get("private") and not name.startswith(("task_ctx.", "_"))}
        event = deepcopy(_OUTLINE_SYMBOLIC_EVENT_SCHEMA)
        event.pop("allOf", None)
        event["properties"]["part_name"] = {"type": ["string", "null"], "minLength": 1}
        event["properties"]["expected_end_state"] = {
            "type": "object", "additionalProperties": False, "minProperties": 1,
            "properties": {name: _outline_state_field_schema(name, row) for name, row in declarations.items()},
        }
        return {"outline_event": event}
    expanded_event = _outline_symbolic_event_schema(declared_state_variables)
    state_properties = deepcopy(
        expanded_event["properties"]["expected_end_state"]["properties"]
    )
    declarations = dict(declared_state_variables or {})

    def field_scope(field_name: str) -> str:
        if field_name in {"part_state", "part_location"}:
            return "part"
        declaration = dict(declarations.get(field_name) or {})
        declared_scope = str(declaration.get("scope") or "").strip()
        if declared_scope:
            return declared_scope
        return "resource"

    resource_state_properties = {
        field_name: deepcopy(field_schema)
        for field_name, field_schema in state_properties.items()
        if field_name != "held_part" and field_scope(field_name) != "part"
    }
    part_state_properties = {
        field_name: deepcopy(field_schema)
        for field_name, field_schema in state_properties.items()
        if field_name != "held_part"
    }

    def state_schema(
        properties: dict[str, Any],
        *,
        required: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": deepcopy(properties),
            "required": list(required or ["resource_state"]),
        }

    common_event_properties = {
        field_name: deepcopy(field_schema)
        for field_name, field_schema in expanded_event["properties"].items()
        if field_name not in {"part_name", "expected_end_state"}
    }
    common_event_required = [
        "outline_id",
        "event_name",
        "resource_jid",
        "expected_end_state",
        "rationale",
    ]

    def event_schema(*, state_ref: str, include_part_name: bool) -> dict[str, Any]:
        properties = deepcopy(common_event_properties)
        if include_part_name:
            properties["part_name"] = deepcopy(
                expanded_event["properties"]["part_name"]
            )
        properties["expected_end_state"] = {"$ref": state_ref}
        required = list(common_event_required)
        if include_part_name:
            required.append("part_name")
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
            "required": required,
        }

    return {
        "resource_only_outline_state": state_schema(resource_state_properties),
        "part_outline_state": state_schema(part_state_properties),
        "custody_outline_state": state_schema(
            state_properties,
            required=[
                "resource_state",
                "held_part",
                "part_state",
                "part_location",
            ],
        ),
        "resource_only_outline_event": event_schema(
            state_ref="#/$defs/resource_only_outline_state",
            include_part_name=False,
        ),
        "part_outline_event": event_schema(
            state_ref="#/$defs/part_outline_state",
            include_part_name=True,
        ),
        "custody_outline_event": event_schema(
            state_ref="#/$defs/custody_outline_state",
            include_part_name=True,
        ),
        "outline_event": {
            "anyOf": [
                {"$ref": "#/$defs/resource_only_outline_event"},
                {"$ref": "#/$defs/part_outline_event"},
                {"$ref": "#/$defs/custody_outline_event"},
            ]
        },
    }


def _resource_candidate_event_schemas(
    *,
    resource_jid: str,
    declared_state_variables: dict[str, dict[str, Any]],
    declared_fields_only: bool = False,
) -> list[dict[str, Any]]:
    """Return compact exact-JID constraints for one RA's candidate branches."""
    state_properties = {
        field_name: _outline_state_field_schema(field_name, declaration)
        for field_name, declaration in sorted(declared_state_variables.items())
    }

    def field_scope(field_name: str) -> str:
        return str(
            dict(declared_state_variables.get(field_name) or {}).get("scope")
            or "resource"
        )

    resource_state_properties = {
        field_name: deepcopy(field_schema)
        for field_name, field_schema in state_properties.items()
        if (declared_fields_only or field_name != "held_part") and field_scope(field_name) != "part"
    }
    part_state_properties = {
        field_name: deepcopy(field_schema)
        for field_name, field_schema in state_properties.items()
        if declared_fields_only or field_name != "held_part"
    }

    def state_schema(
        properties: dict[str, Any],
        *,
        required: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": deepcopy(properties),
            "required": list(required or ([] if declared_fields_only else ["resource_state"])),
            **({"minProperties": 1} if declared_fields_only else {}),
        }

    def branch_schema(
        *,
        expected_end_state_schema: dict[str, Any],
        require_part_name: bool,
    ) -> dict[str, Any]:
        branch: dict[str, Any] = {
            "properties": {
                "resource_jid": {"enum": [resource_jid]},
                "expected_end_state": deepcopy(expected_end_state_schema),
            }
        }
        if require_part_name:
            branch["properties"]["part_name"] = {
                "type": "string",
                "minLength": 1,
            }
            branch["required"] = ["part_name"]
        else:
            # Structured responses can encode an omitted resource-only binding as
            # null, while part-bound branches still require an exact string token.
            branch["properties"]["part_name"] = {"type": "null"}
        return branch

    branches = [
        branch_schema(
            expected_end_state_schema=state_schema(resource_state_properties),
            require_part_name=False,
        ),
        branch_schema(
            expected_end_state_schema=state_schema(part_state_properties),
            require_part_name=True,
        ),
    ]
    custody_fields = {"resource_state", "held_part", "part_state", "part_location"}
    if not declared_fields_only and custody_fields.issubset(state_properties):
        branches.append(
            branch_schema(
                expected_end_state_schema=state_schema(
                    state_properties,
                    required=[
                        "resource_state",
                        "held_part",
                        "part_state",
                        "part_location",
                    ],
                ),
                require_part_name=True,
            )
        )
    return branches


def _outline_incremental_response_schema(
    *,
    declared_state_variables: dict[str, dict[str, Any]] | None = None,
    declared_state_variables_by_resource_jid: (
        dict[str, dict[str, dict[str, Any]]] | None
    ) = None,
    declared_fields_only: bool = False,
) -> dict[str, Any]:
    schema_definitions = _outline_candidate_schema_definitions(
        declared_state_variables,
        declared_state_variables_by_resource_jid,
        declared_fields_only=declared_fields_only,
    )
    outline_event = {"$ref": "#/$defs/outline_event"}
    return {
        "name": "multi_turn_outline_response",
        "strict": False,
        "schema": {
            "type": "object",
            "$defs": schema_definitions,
            "properties": {
                "thought": {"type": "string"},
                "next_transition": deepcopy(outline_event),
                "transition_suffix": {
                    "type": "array",
                    "items": deepcopy(outline_event),
                },
            },
            "required": ["thought", "next_transition"],
        },
    }


def _outline_candidates_response_schema(
    *,
    candidate_bound: int = _DEFAULT_CANDIDATE_BOUND,
    candidate_count: int | str = "auto",
    recovery_selection_mode: str = "pure_llm",
    action_horizon: str = "1",
    action_horizon_steps: int | str | None = None,
    action_horizon_k: int = 3,
    declared_state_variables: dict[str, dict[str, Any]] | None = None,
    declared_state_variables_by_resource_jid: (
        dict[str, dict[str, dict[str, Any]]] | None
    ) = None,
) -> dict[str, Any]:
    normalized_bound = max(1, int(candidate_bound or 1))
    normalized_candidate_count: int | str
    if isinstance(candidate_count, str):
        candidate_count_token = candidate_count.strip().lower()
        if candidate_count_token in {"adaptive", "auto", "n"}:
            normalized_candidate_count = "auto"
        else:
            try:
                normalized_candidate_count = max(1, int(candidate_count_token))
            except ValueError:
                normalized_candidate_count = "auto"
    else:
        try:
            normalized_candidate_count = max(1, int(candidate_count))
        except (TypeError, ValueError):
            normalized_candidate_count = "auto"
    normalized_horizon = str(action_horizon or "1").strip().lower()
    if normalized_horizon not in {"1", "k", "full"}:
        normalized_horizon = "1"
    normalized_selection_mode = str(
        recovery_selection_mode or "pure_llm"
    ).strip().lower()
    if normalized_selection_mode not in {"pure_llm", "neurosymbolic"}:
        normalized_selection_mode = "pure_llm"
    if normalized_selection_mode == "neurosymbolic":
        normalized_horizon = "1"
        candidate_min_items = 1
        candidate_max_items = _DEFAULT_CANDIDATE_BOUND
    elif normalized_horizon == "1":
        candidate_min_items = _ONE_STEP_CANDIDATE_COUNT
        candidate_max_items = _ONE_STEP_CANDIDATE_COUNT
    elif normalized_candidate_count == "auto":
        candidate_min_items = 1
        candidate_max_items = normalized_bound
    else:
        candidate_min_items = int(normalized_candidate_count)
        candidate_max_items = int(normalized_candidate_count)
    if normalized_horizon == "k" and action_horizon_steps not in (None, "", "full"):
        normalized_horizon_k = max(1, int(action_horizon_steps or action_horizon_k or 3))
    else:
        normalized_horizon_k = max(1, int(action_horizon_k or 3))
    schema_definitions = _outline_candidate_schema_definitions(
        declared_state_variables,
        declared_state_variables_by_resource_jid,
    )
    candidate_item_schema = {"$ref": "#/$defs/outline_event"}
    required = ["thought"]
    if normalized_selection_mode == "pure_llm":
        required.append("selected_candidate_index")
    if normalized_horizon == "1":
        candidate_property_name = "candidate_events"
        required.append(candidate_property_name)
        candidate_property = {
            "type": "array",
            "minItems": candidate_min_items,
            "maxItems": candidate_max_items,
            "items": deepcopy(candidate_item_schema),
        }
    else:
        candidate_property_name = "candidate_traces"
        required.append(candidate_property_name)
        event_array_schema: dict[str, Any] = {
            "type": "array",
            "minItems": 1,
            "items": deepcopy(candidate_item_schema),
        }
        if normalized_horizon == "k":
            event_array_schema["maxItems"] = normalized_horizon_k
        candidate_property = {
            "type": "array",
            "minItems": candidate_min_items,
            "maxItems": candidate_max_items,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "events": event_array_schema,
                    "rationale": {"type": "string"},
                },
                "required": ["events"],
            },
        }
    response_properties: dict[str, Any] = {
        "thought": {"type": "string"},
        candidate_property_name: candidate_property,
    }
    if normalized_selection_mode == "pure_llm":
        response_properties["selected_candidate_index"] = {
            "type": "integer",
            "minimum": 0,
            "maximum": max(0, candidate_max_items - 1),
        }
    return {
        "name": "multi_turn_outline_candidates_response",
        "strict": False,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "$defs": schema_definitions,
            "properties": response_properties,
            "required": required,
        },
    }


def _outline_single_pass_response_schema(
    *,
    declared_state_variables: dict[str, dict[str, Any]] | None = None,
    declared_state_variables_by_resource_jid: (
        dict[str, dict[str, dict[str, Any]]] | None
    ) = None,
) -> dict[str, Any]:
    schema_definitions = _outline_candidate_schema_definitions(
        declared_state_variables,
        declared_state_variables_by_resource_jid,
    )
    return {
        "name": "multi_turn_outline_single_pass_response",
        "strict": False,
        "schema": {
            "type": "object",
            "$defs": schema_definitions,
            "properties": {
                "thought": {"type": "string"},
                "transition_trace": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/outline_event"},
                },
            },
            "required": ["thought", "transition_trace"],
        },
    }


def _grounding_response_schema() -> dict[str, Any]:
    return {
        "name": "multi_turn_grounding_response",
        "strict": False,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "thought": {"type": "string"},
                "decision": {"type": "string", "enum": ["observe", "grounded"]},
                "observe_requests": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "fact_type": {"type": "string"},
                            "entity": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["fact_type", "entity"],
                    },
                },
            },
            "required": ["thought", "decision", "observe_requests"],
        },
    }


def _primitive_generation_response_schema() -> dict[str, Any]:
    return {
        "name": "multi_turn_primitive_generation_response",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "decision": {
                    "type": "string",
                    "enum": [
                        "primitive_steps_ready",
                        "need_context",
                        "need_primitive_revision",
                        "primitive_blocked",
                    ],
                },
                "outline_id": {"type": "string"},
                "resource_jid": {"type": "string"},
                "context_requests": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "rationale": {"type": "string"},
                "notes": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "primitive_steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "primitive": {"type": "string"},
                            "params": {"type": "object"},
                        },
                        "required": ["primitive", "params"],
                    },
                },
            },
            "required": [
                "thought",
                "decision",
                "outline_id",
                "resource_jid",
            ],
        },
    }


def _finalize_response_schema() -> dict[str, Any]:
    return {
        "name": "multi_turn_finalize_response",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "decision": {"type": "string"},
            },
            "required": ["thought"],
        },
    }


def _with_outline_context(schema: dict[str, Any]) -> dict[str, Any]:
    """Advertise both response shapes; enforce their requirements in the handler."""
    body = schema["schema"]
    body["properties"].update(
        decision={"type": ["string", "null"], "enum": [None, "need_context"],
                  "description": "Use null for proposed events; use need_context only to request facts without proposing events."},
        context_requests={"type": "array", "minItems": 0, "maxItems": 8,
                          "description": "Use [] for proposed events; need_context requires 1 to 8 references and no candidate fields.",
                          "items": {"type": "string", "pattern": "^/"}},
    )
    body["required"] = ["thought"]
    return schema


def multi_turn_phase_response_schema(  # noqa: PLR0913
    phase: str,
    *,
    outline_mode: str = "incremental",
    outline_context_retrieval: bool = False,
    validation_contract_version: int | None = None,
    candidate_bound: int | None = None,
    candidate_count: int | str = "auto",
    recovery_selection_mode: str = "pure_llm",
    action_horizon: str = "1",
    action_horizon_steps: int | str | None = None,
    action_horizon_k: int = 3,
    declared_state_variables: dict[str, dict[str, Any]] | None = None,
    declared_state_variables_by_resource_jid: (
        dict[str, dict[str, dict[str, Any]]] | None
    ) = None,
) -> dict[str, Any]:
    """Return the JSON response schema for the given phase."""
    normalized = phase.strip().lower()
    if normalized == "outline":
        if validation_contract_version == 2 and outline_mode == "incremental_candidates_validated":
            schema = _outline_candidates_response_schema(
                recovery_selection_mode="neurosymbolic", action_horizon="1",
                declared_state_variables=declared_state_variables,
                declared_state_variables_by_resource_jid=declared_state_variables_by_resource_jid)
        elif validation_contract_version == 2:
            schema = _outline_incremental_response_schema(
                declared_state_variables=declared_state_variables,
                declared_state_variables_by_resource_jid=declared_state_variables_by_resource_jid,
                declared_fields_only=True)
        elif outline_mode == "single_pass":
            schema = _outline_single_pass_response_schema(
                declared_state_variables=declared_state_variables,
                declared_state_variables_by_resource_jid=(
                    declared_state_variables_by_resource_jid
                ),
            )
        elif outline_mode == "incremental_candidates_validated":
            schema = _outline_candidates_response_schema(
                candidate_bound=max(1, int(candidate_bound or _DEFAULT_CANDIDATE_BOUND)),
                candidate_count=candidate_count,
                recovery_selection_mode=recovery_selection_mode,
                action_horizon=action_horizon,
                action_horizon_steps=action_horizon_steps,
                action_horizon_k=action_horizon_k,
                declared_state_variables=declared_state_variables,
                declared_state_variables_by_resource_jid=(
                    declared_state_variables_by_resource_jid
                ),
            )
        else:
            schema = _outline_incremental_response_schema(
                declared_state_variables=declared_state_variables,
                declared_state_variables_by_resource_jid=declared_state_variables_by_resource_jid,
            )
        if validation_contract_version == 2:
            schema["schema"]["properties"].pop("transition_suffix", None)
            schema["schema"]["additionalProperties"] = False
        return _with_outline_context(schema) if outline_context_retrieval else schema
    if normalized == "grounding":
        return _grounding_response_schema()
    if normalized == "primitive_generation":
        return _primitive_generation_response_schema()
    if normalized == "finalize":
        return _finalize_response_schema()
    raise ValueError(f"Unknown phase: {phase!r}")


# ---------------------------------------------------------------------------
# Prompt input builder
# ---------------------------------------------------------------------------


def build_multi_turn_phase_prompt_input(
    *,
    phase: str,
    llm_input: dict[str, Any],
    session_state: dict[str, Any],
    recovery_resources: dict[str, Any] | None = None,
    world_observation_surface: dict[str, Any] | None = None,
    recovery_gap_state: dict[str, Any] | None = None,
    pruned_actions: list[dict[str, Any]] | None = None,
    current_recovery_blockers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the structured prompt input payload for a given phase."""
    return {
        "phase": phase,
        "llm_input": deepcopy(llm_input),
        "session_state": deepcopy(session_state),
        "recovery_resources": deepcopy(recovery_resources or {}),
        "world_observation_surface": deepcopy(world_observation_surface or {}),
        "recovery_gap_state": deepcopy(recovery_gap_state or {}),
        "pruned_actions": deepcopy(pruned_actions or []),
        "current_recovery_blockers": deepcopy(current_recovery_blockers or []),
    }


# ---------------------------------------------------------------------------
# Grounding prompt
# ---------------------------------------------------------------------------


def _compact_json(obj: Any) -> str:
    """JSON with 2-space indent, no markdown fences."""
    return json.dumps(obj, indent=2, default=str, ensure_ascii=False)


def _inline_json(obj: Any) -> str:
    """Single-line JSON fragment for prompt diagnostics."""
    return json.dumps(
        obj,
        default=str,
        ensure_ascii=False,
        sort_keys=True,
        separators=(", ", ": "),
    )


def _recovery_safety_rule_text(rule: dict[str, Any]) -> str:
    """Render recovery semantics while preserving each supplied formal token."""
    constraint_type = str(rule.get("constraint_type") or "").strip()
    if constraint_type == "ordering_place_approach_priority":
        products = [
            str(item).strip()
            for item in (rule.get("product") or [])
            if str(item).strip()
        ]
        event = str(rule.get("event") or "").strip()
        destination = str((rule.get("context") or {}).get("destination") or "").strip()
        if len(products) >= 2 and event:
            destination_text = f" to {destination}" if destination else ""
            return (
                f"{products[0]} {event} must occur before {products[1]} "
                f"{event}{destination_text}."
            )
    return str(rule.get("raw_text") or rule.get("summary") or "").strip()


def _compact_safety_rules(llm_input: dict[str, Any]) -> str:
    """Render safety rules using their recovery-facing semantics."""
    if "prompt_safety_rules" in llm_input:
        lines = list(dict.fromkeys(
            f"- {rule['id']}: {rule['requirement']}"
            for rule in llm_input["prompt_safety_rules"]
        ))
        return "\n".join(lines) if lines else "(none)"
    rules = llm_input.get("loaded_safety_rules") or []
    lines: list[str] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        rule_id = str(rule.get("id") or rule.get("rule_id") or "").strip()
        rule_text = _recovery_safety_rule_text(rule)
        if rule_id and rule_text:
            lines.append(f"- {rule_id}: {rule_text}")
    return "\n".join(lines) if lines else "(none)"


def _compact_assembly_requirements(llm_input: dict[str, Any]) -> str:
    """Extract summary lines from assembly requirements."""
    reqs = llm_input.get("relevant_assembly_requirements") or []
    lines: list[str] = []
    for req in reqs:
        if not isinstance(req, dict):
            continue
        req_id = str(req.get("requirement_id") or "").strip()
        summary = str(req.get("summary") or "").strip()
        status = str(req.get("status") or "").strip()
        if req_id and summary:
            lines.append(f"- {req_id} [{status}]: {summary}")
    return "\n".join(lines) if lines else "(none)"


def _compact_recovery_goals(
    llm_input: dict[str, Any],
    *,
    projected_parts: list[dict[str, Any]],
    current_recovery_blockers: list[dict[str, Any]],
    exact_conditions: bool = False,
) -> str:
    del current_recovery_blockers
    if exact_conditions:
        rows = [row for row in llm_input.get("goal_conditions", []) if isinstance(row, dict)]
        rows.extend({"part_name": name, "requirements": deepcopy(requirements)}
                    for name, requirements in llm_input.get("recorded_product_requirements", {}).items())
        return _fact_rows(rows)
    parts_by_name: dict[str, dict[str, Any]] = {}
    for row in projected_parts:
        if not isinstance(row, dict):
            continue
        part_name = str(row.get("part_name") or "").strip()
        if part_name and part_name not in parts_by_name:
            parts_by_name[part_name] = dict(row)

    goal_conditions_by_part: dict[str, dict[str, Any]] = {}
    for raw_condition in llm_input.get("goal_conditions") or []:
        if not isinstance(raw_condition, dict):
            continue
        condition = dict(raw_condition)
        if str(condition.get("entity_kind") or "").strip() != "part":
            continue
        part_name = str(condition.get("entity") or "").strip()
        field = str(condition.get("field") or "").strip()
        if not part_name or not field:
            continue
        goal_conditions_by_part.setdefault(part_name, {})[field] = deepcopy(
            condition.get("expected")
        )

    lines: list[str] = []
    for part_name in sorted(goal_conditions_by_part):
        part_row = dict(parts_by_name.get(part_name) or {})
        conditions = goal_conditions_by_part[part_name]
        goal_location = str(
            conditions.get("location") or part_row.get("goal_location") or ""
        ).strip()
        goal_state = str(conditions.get("state") or "").strip()
        if goal_location and goal_state:
            lines.append(
                f"- restore {part_name} to {goal_location} with part_state={goal_state}"
            )
        elif goal_location:
            lines.append(f"- restore {part_name} to {goal_location}")
        elif goal_state:
            lines.append(f"- restore {part_name} with part_state={goal_state}")
        else:
            lines.append(f"- restore {part_name}")
    return "\n".join(lines) if lines else "(none)"


def _outline_rejection_history(session_state: dict[str, Any]) -> list[dict[str, Any]]:
    """Return prior rejected outline attempts with their validation findings."""
    history: list[dict[str, Any]] = []
    for turn in session_state.get("turns") or []:
        if not isinstance(turn, dict):
            continue
        if str(turn.get("phase") or "").strip().lower() != "outline":
            continue
        findings = [
            deepcopy(row)
            for row in (turn.get("validation_findings") or [])
            if isinstance(row, dict)
        ]
        if not findings:
            continue
        row: dict[str, Any] = {
            "turn_index": int(turn.get("turn_index") or 0),
            "decision": str(turn.get("decision") or "").strip() or None,
            "proposed_next_transition": deepcopy(dict(turn.get("next_transition") or {})),
            "validation_findings": findings,
        }
        transition_suffix = [
            deepcopy(item)
            for item in (turn.get("transition_suffix") or [])
            if isinstance(item, dict)
        ]
        if transition_suffix:
            row["transition_suffix"] = transition_suffix
        history.append(row)
    return history


_PROMPT_FEEDBACK_RENDER_STYLES = {"des_event_diagnostic", "raw_code"}

_PROMPT_PERSISTENT_FINDING_CODES = {
    "source_reference_unavailable",
    "unsatisfied_guard_predicate",
    "unknown_location_binding",
    "unknown_product_binding",
    "unknown_resource_binding",
}

def _feedback_render_style_token(session_state: dict[str, Any]) -> str:
    return "des_event_diagnostic"


def _finding_constraint_code_token(finding: dict[str, Any]) -> str:
    return str(finding.get("constraint_code") or "").strip().lower()


def _finding_durable(finding: dict[str, Any]) -> bool:
    durable = finding.get("durable")
    if durable is not None:
        return bool(durable)
    return _finding_constraint_code_token(finding) in _PROMPT_PERSISTENT_FINDING_CODES


def _task_target_ref(task: dict[str, Any], finding: dict[str, Any] | None = None) -> str:
    direct_target = str(task.get("target_ref") or "").strip()
    if direct_target:
        return direct_target
    action_target = task.get("action_target")
    if isinstance(action_target, dict):
        target = str(
            action_target.get("target_ref")
            or action_target.get("target_location")
            or action_target.get("location")
            or ""
        ).strip()
        if target:
            return target
    end_state = dict(task.get("expected_end_state") or {})
    direct_target = str(end_state.get("part_location") or "").strip()
    if direct_target:
        return direct_target
    if isinstance(finding, dict):
        evidence = dict(finding.get("evidence") or {})
        return str(evidence.get("target_ref") or evidence.get("target_location") or "").strip()
    return ""


def _candidate_event_label(
    *,
    task: dict[str, Any],
    finding: dict[str, Any] | None = None,
) -> str:
    outline_id = str(
        task.get("outline_id") or task.get("task_id") or (finding or {}).get("task_id") or ""
    ).strip()
    resource_jid = str(
        _task_resource_jid(task) or (finding or {}).get("resource_jid") or ""
    ).strip()
    part_name = str(_task_part_name(task) or (finding or {}).get("part_name") or "").strip()
    label_parts = [item for item in (outline_id, resource_jid, part_name) if item]
    label = "/".join(label_parts) if label_parts else "anonymous_event"
    target_ref = _task_target_ref(task, finding=finding)
    if target_ref:
        label += f"->{target_ref}"
    return label


def _finding_resources_by_jid(
    projected_resources: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("resource_jid") or "").strip(): dict(row)
        for row in projected_resources
        if isinstance(row, dict) and str(row.get("resource_jid") or "").strip()
    }


def _finding_parts_by_name(
    projected_parts: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("part_name") or "").strip(): dict(row)
        for row in projected_parts
        if isinstance(row, dict) and str(row.get("part_name") or "").strip()
    }


def _finding_state_evidence_text(
    *,
    task: dict[str, Any],
    finding: dict[str, Any],
    resources_by_jid: dict[str, dict[str, Any]] | None = None,
    parts_by_name: dict[str, dict[str, Any]] | None = None,
) -> str:
    evidence = dict(finding.get("evidence") or {})
    resource_jid = str(_task_resource_jid(task) or finding.get("resource_jid") or "").strip()
    part_name = str(_task_part_name(task) or finding.get("part_name") or "").strip()
    target_ref = _task_target_ref(task, finding=finding)
    resource_row = dict((resources_by_jid or {}).get(resource_jid) or {})
    part_row = dict((parts_by_name or {}).get(part_name) or {})

    facts: list[str] = []
    if resource_jid:
        facts.append(f"resource_jid={resource_jid}")
    if part_name:
        facts.append(f"part_name={part_name}")
    if target_ref:
        facts.append(f"target_ref={target_ref}")

    checked_pose = dict(evidence.get("checked_pose") or finding.get("pose") or {})
    if checked_pose:
        facts.append(f"checked_pose=({_pose_xyz_text(checked_pose)})")

    changed_fields = [
        str(field).strip() for field in (evidence.get("changed_fields") or []) if str(field).strip()
    ]
    if changed_fields:
        facts.append("changed_fields=" + ",".join(changed_fields))

    condition_ids = [
        str(item).strip()
        for item in (evidence.get("condition_ids") or finding.get("condition_ids") or [])
        if str(item).strip()
    ]
    if condition_ids:
        facts.append("condition_ids=" + ",".join(condition_ids))

    rule_id = str(finding.get("rule_id") or evidence.get("rule_id") or "").strip()
    if rule_id:
        facts.append(f"rule_id={rule_id}")

    mismatches = [
        dict(item) for item in (evidence.get("mismatches") or []) if isinstance(item, dict)
    ]
    if mismatches:
        mismatch_parts: list[str] = []
        for item in mismatches:
            field_name = str(item.get("field") or "").strip()
            if not field_name:
                continue
            expected_text = _inline_json(item.get("expected"))
            available = bool(item.get("available"))
            actual_text = _inline_json(item.get("actual")) if available else "<unavailable>"
            mismatch_parts.append(f"{field_name}(expected={expected_text}, actual={actual_text})")
        if mismatch_parts:
            facts.append("state_mismatches=" + ", ".join(mismatch_parts))

    if "resource_state" in resource_row:
        facts.append(f"resource_state={_inline_json(resource_row.get('resource_state'))}")

    held_part = str(resource_row.get("held_part") or "").strip()
    if held_part:
        facts.append(f"held_part={held_part}")

    if "part_state" in part_row:
        facts.append(f"part_state={_inline_json(part_row.get('part_state'))}")

    part_holder = str(part_row.get("part_holder_resource_jid") or "").strip()
    if part_holder:
        facts.append(f"part_holder_resource_jid={part_holder}")

    part_location = str(part_row.get("part_location") or "").strip()
    if part_location:
        facts.append(f"part_location={part_location}")

    unsatisfied_predicates = [
        str(item).strip()
        for item in (finding.get("unsatisfied_predicates") or [])
        if str(item).strip()
    ]
    if unsatisfied_predicates:
        facts.append("unsatisfied_predicates=" + ",".join(unsatisfied_predicates))

    return "; ".join(facts) if facts else "projected_state_facts=not_available"


def _des_diagnostic_fields(finding: dict[str, Any]) -> dict[str, str]:
    stage = str(finding.get("stage") or "").strip().lower()
    constraint_code = _finding_constraint_code_token(finding)
    constraint_family = str(finding.get("constraint_family") or "").strip().lower()

    if stage in {"ontology_binding", "schema_grounding"}:
        return {
            "event_status": "disabled",
            "diagnosis": "ontology_or_schema_binding_failed",
            "guard_or_condition": (
                "candidate event instance does not bind cleanly to the PPR entity/schema model"
            ),
            "re_enablement": (
                "bind the event to known resources/products/locations and a registered event schema"
            ),
        }

    if stage in {"event_enabledness", "plant_enabledness"}:
        unsatisfied = [
            str(item).strip()
            for item in (finding.get("unsatisfied_predicates") or [])
            if str(item).strip()
        ]
        predicate_text = ", ".join(unsatisfied) if unsatisfied else "event guard predicates"
        return {
            "event_status": "disabled",
            "diagnosis": "event_not_enabled",
            "guard_or_condition": f"event guard predicates are false: {predicate_text}",
            "re_enablement": (
                str(finding.get("retry_hint") or "").strip()
                or "first satisfy the missing event guard predicates"
            ),
        }

    if stage == "supervisor_admissibility":
        return {
            "event_status": "blocked_by_supervisor",
            "diagnosis": "supervisor_admissibility_block",
            "guard_or_condition": (
                "candidate event is not admissible under the current ordering or safety supervisor"
            ),
            "re_enablement": (
                "choose an enabled event whose projected successor remains supervisor-admissible"
            ),
        }

    if stage == "marked_progress":
        return {
            "event_status": "disabled",
            "diagnosis": "recovery_goal_progress_failure",
            "guard_or_condition": (
                "candidate event is enabled but does not advance a recovery goal"
            ),
            "re_enablement": (
                "choose an enabled event that strictly reduces the active continuation blockers"
            ),
        }

    if constraint_code in {
        "unknown_location_token",
        "unknown_named_pose",
        "unknown_state_token",
        "source_reference_unavailable",
        "part_ambiguous",
    }:
        return {
            "event_status": "disabled",
            "diagnosis": "state_estimate_not_grounded",
            "guard_or_condition": (
                "candidate symbolic bindings are not grounded in the current state estimate"
            ),
            "re_enablement": (
                "ground the referenced entity, pose, or location in the current state estimate"
            ),
        }

    if constraint_code in {
        "candidate_schema_violation",
        "disallowed_outline_state_field",
    }:
        return {
            "event_status": "disabled",
            "diagnosis": "candidate_schema_violation",
            "guard_or_condition": (
                "candidate event uses fields outside the recovery candidate rules"
            ),
            "re_enablement": (
                "author only the allowed state fields shown in Recovery Candidate Rules"
            ),
        }

    if constraint_code in {
        "invalid_dependency_reference",
    }:
        return {
            "event_status": "disabled",
            "diagnosis": "dependency_reference_invalid",
            "guard_or_condition": (
                "candidate event uses predecessors outside the current candidate response"
            ),
            "re_enablement": (
                "omit predecessors in candidate mode; if you must use them elsewhere, "
                "reference only ids declared in the same response"
            ),
        }

    if (
        constraint_code
        in {
            "blocker_open",
            "dependency_unsatisfied",
            "order_violation",
            "claimed_condition_not_currently_unmet",
        }
        or constraint_family == "sequence"
    ):
        return {
            "event_status": "blocked_by_supervisor",
            "diagnosis": "supervisor_ordering_block",
            "guard_or_condition": (
                "projected successor would advance before open continuation conditions are cleared"
            ),
            "re_enablement": (
                "first execute an event that clears the open prerequisite or continuation condition"
            ),
        }

    if constraint_code == "safety_rule_violation" or constraint_family == "safety":
        return {
            "event_status": "blocked_by_supervisor",
            "diagnosis": "supervisor_safety_block",
            "guard_or_condition": (
                "projected successor is outside the safety-admissible region of the "
                "product automaton"
            ),
            "re_enablement": ("choose an event whose projected APs remain supervisor-admissible"),
        }

    if constraint_code == "no_state_change":
        return {
            "event_status": "disabled",
            "diagnosis": "no_state_change",
            "guard_or_condition": "candidate event leaves the authored symbolic state unchanged",
            "re_enablement": (
                "revise the event so the authored end state differs from the current "
                "projected authored state"
            ),
        }

    return {
        "event_status": "disabled",
        "diagnosis": "runtime_validation_block",
        "guard_or_condition": ("candidate event is blocked by a runtime validation constraint"),
        "re_enablement": ("revise the event to satisfy the reported validation condition"),
    }


def _render_des_event_diagnostic(
    *,
    task: dict[str, Any],
    finding: dict[str, Any],
    resources_by_jid: dict[str, dict[str, Any]] | None = None,
    parts_by_name: dict[str, dict[str, Any]] | None = None,
) -> str:
    constraint_code = str(
        finding.get("constraint_code") or "validation_rejected"
    ).strip()
    reason = str(finding.get("reason") or "Candidate was rejected.").strip()
    line = (
        f"- candidate_event={_candidate_event_label(task=task, finding=finding)}"
        f" | constraint_code={constraint_code}"
        f" | reason={reason}"
    )
    evidence = dict(finding.get("evidence") or {})
    if evidence:
        line += f" | evidence={_inline_json(evidence)}"
    else:
        line += (
            " | state_evidence="
            + _finding_state_evidence_text(
                task=task,
                finding=finding,
                resources_by_jid=resources_by_jid,
                parts_by_name=parts_by_name,
            )
        )
    if _finding_durable(finding):
        line += " | persistence=diagnosis persists until the relevant projected state facts change"
    return line


def _candidate_diagnostic_signature(
    *,
    task: dict[str, Any],
    finding: dict[str, Any],
    resources_by_jid: dict[str, dict[str, Any]] | None = None,
    parts_by_name: dict[str, dict[str, Any]] | None = None,
) -> tuple[str, str, str, str, str, str]:
    """Stable semantic key for feedback de-duplication.

    Candidate ids are deliberately excluded so repeated RECOVERY_SEQ*_N rows
    collapse when they describe the same event/failure under the same state.
    """
    diagnostic = _des_diagnostic_fields(finding)
    resource_jid = str(task.get("resource_jid") or finding.get("resource_jid") or "").strip()
    part_name = str(task.get("part_name") or finding.get("part_name") or "").strip()
    target_ref = _task_target_ref(task, finding=finding)
    finding_evidence = dict(finding.get("evidence") or {})
    evidence_text = (
        _inline_json(finding_evidence)
        if finding_evidence
        else _finding_state_evidence_text(
            task=task,
            finding=finding,
            resources_by_jid=resources_by_jid,
            parts_by_name=parts_by_name,
        )
    )
    return (
        resource_jid,
        part_name,
        target_ref,
        diagnostic["event_status"],
        diagnostic["diagnosis"],
        evidence_text,
    )


def _outline_validation_summary(
    findings: list[dict[str, Any]],
    *,
    feedback_render_style: str = "raw_code",
    resources_by_jid: dict[str, dict[str, Any]] | None = None,
    parts_by_name: dict[str, dict[str, Any]] | None = None,
) -> str:
    if feedback_render_style == "des_event_diagnostic":
        lines: list[str] = []
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            task = {
                "outline_id": str(finding.get("task_id") or "").strip(),
                "resource_jid": str(finding.get("resource_jid") or "").strip(),
                "part_name": str(finding.get("part_name") or "").strip(),
            }
            lines.append(
                _render_des_event_diagnostic(
                    task=task,
                    finding=finding,
                    resources_by_jid=resources_by_jid,
                    parts_by_name=parts_by_name,
                )
            )
        return "\n".join(lines) if lines else "(none)"

    lines: list[str] = []
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        task_id = str(finding.get("task_id") or "").strip()
        resource_jid = str(finding.get("resource_jid") or "").strip()
        part_name = str(finding.get("part_name") or "").strip()
        stage = str(finding.get("stage") or "").strip()
        validation_category = str(finding.get("validation_category") or "").strip()
        constraint_code = str(finding.get("constraint_code") or "").strip()
        reason = str(finding.get("reason") or "").strip()

        label_parts = [item for item in (task_id, resource_jid, part_name) if item]
        label = " / ".join(label_parts) if label_parts else "validation finding"
        code_label = constraint_code
        if stage:
            code_label = f"{stage}:{constraint_code}" if constraint_code else stage
        if validation_category:
            code_label = (
                f"{validation_category}/{code_label}" if code_label else validation_category
            )
        if code_label and reason:
            lines.append(f"- {label} [{code_label}]: {reason}")
        elif reason:
            lines.append(f"- {label}: {reason}")
        elif code_label:
            lines.append(f"- {label} [{code_label}]")
    return "\n".join(lines) if lines else "(none)"


def _effective_task_part_holder(
    state: dict[str, Any],
    *,
    part_name: str,
    resource_jid: str,
) -> str:
    holder = str(
        state.get("part_holder_resource_jid") or state.get("current_holder_resource_jid") or ""
    ).strip()
    if holder:
        return holder
    held_part = str(state.get("held_part") or "").strip()
    if part_name and held_part == part_name:
        return resource_jid
    return ""


def _structured_task_action_summary(task: dict[str, Any]) -> str:
    view = _outline_task_view(task)
    event_name = str(view.get("event_name") or "").strip()
    surface_event_name = str(view.get("surface_event_name") or "").strip()
    surface_description = str(view.get("surface_description") or "").strip()
    part_name = str(view.get("part_name") or "").strip()
    target_ref = str(view.get("target_ref") or "").strip()
    if event_name:
        summary = event_name
        lower_summary = event_name.lower()
        if part_name and part_name.lower() not in lower_summary:
            summary = f"{summary} {part_name}"
            lower_summary = summary.lower()
        if target_ref and target_ref.lower() not in lower_summary:
            summary = f"{summary} to {target_ref}"
        return summary
    if surface_event_name or surface_description:
        return surface_event_name or surface_description

    resource_jid = str(view.get("resource_jid") or "").strip()
    start_state = dict(view.get("expected_start_state") or {})
    end_state = dict(view.get("expected_end_state") or {})

    start_resource_state = str(start_state.get("resource_state") or "").strip()
    end_resource_state = str(end_state.get("resource_state") or "").strip()
    start_part_state = str(start_state.get("part_state") or "").strip()
    end_part_state = str(end_state.get("part_state") or "").strip()
    start_part_location = str(start_state.get("part_location") or "").strip()
    end_part_location = str(end_state.get("part_location") or "").strip()
    start_holder = _effective_task_part_holder(
        start_state,
        part_name=part_name,
        resource_jid=resource_jid,
    )
    end_holder = _effective_task_part_holder(
        end_state,
        part_name=part_name,
        resource_jid=resource_jid,
    )

    summary_parts: list[str] = []
    if part_name:
        if end_holder == resource_jid and start_holder != resource_jid:
            summary_parts.append(f"acquire {part_name}")
        elif start_holder == resource_jid and end_holder != resource_jid:
            if end_part_location:
                summary_parts.append(f"place {part_name} at {end_part_location}")
            else:
                summary_parts.append(f"release {part_name}")
        elif start_part_location != end_part_location and end_part_location:
            summary_parts.append(f"move {part_name} to {end_part_location}")
        elif start_part_state != end_part_state and end_part_state:
            summary_parts.append(f"{part_name} {start_part_state or 'state'} -> {end_part_state}")

    if start_resource_state != end_resource_state and end_resource_state:
        summary_parts.append(f"{start_resource_state or 'resource'} -> {end_resource_state}")

    if summary_parts:
        return "; ".join(summary_parts)
    return str(view.get("description") or "").strip() or "state transition"


def _outline_task_sequence_summary(
    tasks: list[dict[str, Any]],
    *,
    include_descriptions: bool = True,
) -> str:
    lines: list[str] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        view = _outline_task_view(task)
        outline_id = str(view.get("outline_id") or "").strip()
        resource_jid = str(view.get("resource_jid") or "").strip()
        part_name = str(view.get("part_name") or "").strip()
        label_parts = [item for item in (outline_id, resource_jid, part_name) if item]
        label = " / ".join(label_parts) if label_parts else "accepted event"
        lines.append(f"- {label} -> {_structured_task_action_summary(view)}")
        description = str(view.get("description") or "").strip()
        if include_descriptions and description:
            lines.append(f"  ({description})")
    return "\n".join(lines) if lines else "(none)"


def _accepted_transition_prefix_for_prompt(
    tasks: list[dict[str, Any]],
    *,
    hide_registered_functions: bool = False,
    public_fields: dict[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    """Return identities and optional recent accepted public changes."""
    from cais_spade_llm.agents.shared_information.recovery_validation_protocol import recovery_validation_fingerprint

    prompt_rows: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            continue
        row: dict[str, Any] = {}
        for field_name in ("outline_id", "event_name", "resource_jid", "part_name"):
            if hide_registered_functions and task.get("candidate_source") == "robot_task_program" and field_name == "event_name":
                continue
            if field_name not in task:
                continue
            field_value = task.get(field_name)
            if field_name == "part_name" and not str(field_value or "").strip():
                continue
            row[field_name] = deepcopy(field_value)
        if hide_registered_functions and task.get("candidate_source") == "robot_task_program":
            row["candidate_source"] = task["candidate_source"]
            row["expected_end_state"] = deepcopy(task.get("expected_end_state") or {})
        if public_fields is not None:
            row.pop("expected_end_state", None)
            fields = public_fields.get(task.get("resource_jid"), set())
            start = task.get("expected_start_state") or {}
            changes = {field: deepcopy(value) for field, value in (task.get("expected_end_state") or {}).items()
                if field in fields and (field not in start or recovery_validation_fingerprint(value)
                                        != recovery_validation_fingerprint(start[field]))}
            if index >= len(tasks) - 6 and changes:
                row["expected_end_state"] = changes
        if row:
            prompt_rows.append(row)
    return prompt_rows


def _outline_rejection_history_summary(history: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for row in history:
        if not isinstance(row, dict):
            continue
        turn_index = int(row.get("turn_index") or 0)
        task = _outline_task_view(dict(row.get("proposed_next_transition") or {}))
        outline_id = str(task.get("outline_id") or "").strip()
        resource_jid = str(task.get("resource_jid") or "").strip()
        part_name = str(task.get("part_name") or "").strip()
        finding_summary = _outline_validation_summary(
            [item for item in (row.get("validation_findings") or []) if isinstance(item, dict)]
        ).replace("\n- ", "; ")
        finding_summary = (
            finding_summary[2:] if finding_summary.startswith("- ") else finding_summary
        )

        label_parts = [item for item in (outline_id, resource_jid, part_name) if item]
        label = " / ".join(label_parts) if label_parts else "rejected event"
        sentence = f"- Turn {turn_index}: {label}"
        action_summary = _structured_task_action_summary(task)
        if action_summary:
            sentence += f" -> {action_summary}"
        if finding_summary and finding_summary != "(none)":
            sentence += f". Feedback: {finding_summary}"
        lines.append(sentence)
    return "\n".join(lines) if lines else "(none)"


def _candidate_rejection_history(session_state: dict[str, Any]) -> list[dict[str, Any]]:
    """Return rejected candidate batches since the last accepted recovery event."""
    history: list[dict[str, Any]] = []
    for turn in session_state.get("turns") or []:
        if not isinstance(turn, dict):
            continue
        if str(turn.get("phase") or "").strip().lower() != "outline":
            continue
        if isinstance(turn.get("selected_transition"), dict) or isinstance(
            turn.get("next_transition"), dict
        ):
            history = []
            continue
        candidate_evaluations = [
            deepcopy(row)
            for row in (turn.get("candidate_evaluations") or [])
            if isinstance(row, dict) and not bool(row.get("valid"))
        ]
        if not candidate_evaluations:
            continue
        history.append(
            {
                "turn_index": int(turn.get("turn_index") or 0),
                "candidate_evaluations": candidate_evaluations,
            }
        )
    return history


def _compact_selection_feedback_evidence(row: dict[str, Any]) -> dict[str, Any]:
    """Keep only comparison evidence that can change the next LLM proposal."""
    evidence = dict(row.get("evidence") or {})
    is_no_progress = (
        str(row.get("constraint_code") or "").strip()
        == "no_progressing_candidate"
    )
    compact: dict[str, Any] = {}
    for field_name in (
        "nondominated_candidate_ids",
        "selection_revision_count",
        "selection_revision_limit",
    ):
        if field_name in evidence:
            compact[field_name] = deepcopy(evidence.get(field_name))
    comparison = []
    for candidate in evidence.get("candidate_comparison") or []:
        if not isinstance(candidate, dict):
            continue
        if is_no_progress:
            part_name = str(candidate.get("part_name") or "").strip()
            comparison_row = {
                "event_name": str(candidate.get("event_name") or "").strip(),
                "resource_jid": str(candidate.get("resource_jid") or "").strip(),
                **({"part_name": part_name} if part_name else {}),
                "selection_status": str(candidate.get("selection_status") or ""),
                "selection_constraint_codes": deepcopy(
                    candidate.get("selection_constraint_codes") or []
                ),
                "expected_end_state": deepcopy(candidate.get("expected_end_state") or {}),
            }
        else:
            selection_evidence = dict(candidate.get("selection_evidence") or {})
            comparison_row = {
                "candidate_id": str(candidate.get("candidate_id") or ""),
                "selection_status": str(candidate.get("selection_status") or ""),
                "cleared_recovery_obligation_ids": deepcopy(
                    selection_evidence.get("cleared_recovery_obligation_ids") or []
                ),
                "open_recovery_obligation_ids_after": deepcopy(
                    selection_evidence.get("open_recovery_obligation_ids_after") or []
                ),
            }
        comparison.append(comparison_row)
    if comparison:
        compact["candidate_comparison"] = comparison
    return compact


def _no_progress_selection_feedback_summary(row: dict[str, Any]) -> str:
    """Render the recorded rejection reason without prescribing new effects."""
    reason = str(row.get("reason") or "Recorded checks did not establish recovery progress.").strip()
    lines = [
        "- model_based_selection [no_progressing_candidate]: " + reason
    ]
    evidence = dict(row.get("evidence") or {})
    for candidate in evidence.get("candidate_comparison") or []:
        if not isinstance(candidate, dict):
            continue
        event_name = str(
            candidate.get("event_name") or candidate.get("outline_id") or "anonymous_event"
        ).strip()
        resource_jid = str(candidate.get("resource_jid") or "").strip()
        part_name = str(candidate.get("part_name") or "").strip()
        selection_status = str(candidate.get("selection_status") or "").strip()
        constraint_codes = [
            str(code).strip()
            for code in (candidate.get("selection_constraint_codes") or [])
            if str(code).strip()
        ]
        fields = [f"candidate_event={event_name}"]
        if resource_jid:
            fields.append(f"resource_jid={resource_jid}")
        if part_name:
            fields.append(f"part_name={part_name}")
        if selection_status:
            fields.append(f"selection_status={selection_status}")
        if constraint_codes:
            fields.append("constraint_codes=" + ",".join(constraint_codes))
        fields.append(
            "expected_end_state="
            + _inline_json(dict(candidate.get("expected_end_state") or {}))
        )
        lines.append("  - " + " | ".join(fields))
    return "\n".join(lines)


def _candidate_rejection_learning_summary(  # noqa: C901
    *,
    history: list[dict[str, Any]],
    feedback_rows: list[dict[str, Any]],
    feedback_render_style: str = "raw_code",
    resources_by_jid: dict[str, dict[str, Any]] | None = None,
    parts_by_name: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Summarize rejected candidate patterns without repeating action labels."""
    if feedback_render_style == "des_event_diagnostic":
        line_by_key: dict[tuple[str, str, str, str, str, str], str] = {}

        def _add_des_rows(*, task: dict[str, Any], findings: list[dict[str, Any]]) -> None:
            for finding in findings:
                if not isinstance(finding, dict):
                    continue
                key = _candidate_diagnostic_signature(
                    task=task,
                    finding=finding,
                    resources_by_jid=resources_by_jid,
                    parts_by_name=parts_by_name,
                )
                if key in line_by_key:
                    line_by_key.pop(key, None)
                line_by_key[key] = _render_des_event_diagnostic(
                    task=task,
                    finding=finding,
                    resources_by_jid=resources_by_jid,
                    parts_by_name=parts_by_name,
                )

        for row in history:
            if not isinstance(row, dict):
                continue
            for evaluation in row.get("candidate_evaluations") or []:
                if not isinstance(evaluation, dict):
                    continue
                _add_des_rows(
                    task=dict(evaluation.get("task") or {}),
                    findings=[
                        item
                        for item in (evaluation.get("validation_findings") or [])
                        if isinstance(item, dict)
                    ],
                )

        for row in feedback_rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("constraint_code") or "").strip():
                constraint_code = str(row.get("constraint_code") or "").strip()
                if constraint_code == "no_progressing_candidate":
                    line_by_key[("", "", "", constraint_code, "", "")] = (
                        _no_progress_selection_feedback_summary(row)
                    )
                    continue
                if constraint_code in {"selection_ambiguous", "selection_unresolved"}:
                    line_by_key[("", "", "", constraint_code, "", "")] = (
                        f"- model_based_selection [{constraint_code}]: "
                        f"{str(row.get('reason') or '').strip()} | evidence="
                        f"{_compact_json(_compact_selection_feedback_evidence(row))}"
                    )
                    continue
                _add_des_rows(task={}, findings=[row])
                continue
            _add_des_rows(
                task=dict(row.get("task") or {}),
                findings=[
                    item
                    for item in (row.get("validation_findings") or [])
                    if isinstance(item, dict)
                ],
            )

        lines = list(line_by_key.values())[-8:]
        return "\n".join(lines) if lines else "(none)"

    line_by_key: dict[tuple[str, str, str, str, str], str] = {}

    def _target_ref(task: dict[str, Any]) -> str:
        direct_target = str(task.get("target_ref") or "").strip()
        if direct_target:
            return direct_target
        action_target = task.get("action_target")
        if isinstance(action_target, dict):
            return str(
                action_target.get("target_ref") or action_target.get("location") or ""
            ).strip()
        return ""

    def _add_evaluation(
        *,
        task: dict[str, Any],
        findings: list[dict[str, Any]],
        turn_index: int | None = None,
    ) -> None:
        resource_jid = str(task.get("resource_jid") or "").strip()
        part_name = str(task.get("part_name") or "").strip()
        target_ref = _target_ref(task)
        label_parts = [item for item in (resource_jid, part_name) if item]
        label = " / ".join(label_parts) if label_parts else "candidate"
        if target_ref:
            label += f" to {target_ref}"
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            constraint_code = str(finding.get("constraint_code") or "").strip()
            reason = str(finding.get("reason") or "").strip()
            if not constraint_code and not reason:
                continue
            key = (resource_jid, part_name, target_ref, constraint_code, reason)
            sentence = f"- {label}"
            if turn_index is not None:
                sentence += f" (turn {turn_index})"
            if constraint_code and reason:
                sentence += f" [{constraint_code}]: {reason}"
            elif constraint_code:
                sentence += f" [{constraint_code}]"
            else:
                sentence += f": {reason}"
            if constraint_code == "workspace_unreachable":
                sentence += " (requires fresh robot-model validation)"
            if key in line_by_key:
                line_by_key.pop(key, None)
            line_by_key[key] = sentence

    for row in history:
        if not isinstance(row, dict):
            continue
        turn_index = int(row.get("turn_index") or 0)
        for evaluation in row.get("candidate_evaluations") or []:
            if not isinstance(evaluation, dict):
                continue
            _add_evaluation(
                task=dict(evaluation.get("task") or {}),
                findings=[
                    item
                    for item in (evaluation.get("validation_findings") or [])
                    if isinstance(item, dict)
                ],
                turn_index=turn_index or None,
            )

    for row in feedback_rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("constraint_code") or "").strip():
            constraint_code = str(row.get("constraint_code") or "").strip()
            if constraint_code == "no_progressing_candidate":
                key = ("", "", "", constraint_code, "")
                line_by_key[key] = _no_progress_selection_feedback_summary(row)
                continue
            if constraint_code in {"selection_ambiguous", "selection_unresolved"}:
                key = ("", "", "", constraint_code, str(row.get("reason") or ""))
                line_by_key[key] = (
                    f"- candidate [{constraint_code}]: "
                    f"{str(row.get('reason') or '').strip()} | evidence="
                    f"{_compact_json(_compact_selection_feedback_evidence(row))}"
                )
                continue
            _add_evaluation(task={}, findings=[row])
            continue
        _add_evaluation(
            task=dict(row.get("task") or {}),
            findings=[
                item for item in (row.get("validation_findings") or []) if isinstance(item, dict)
            ],
        )

    lines = list(line_by_key.values())[-8:]
    return "\n".join(lines) if lines else "(none)"


def _candidate_rejection_diagnostic_signatures(
    *,
    history: list[dict[str, Any]],
    feedback_rows: list[dict[str, Any]],
    resources_by_jid: dict[str, dict[str, Any]] | None = None,
    parts_by_name: dict[str, dict[str, Any]] | None = None,
) -> set[tuple[str, str, str, str, str, str]]:
    signatures: set[tuple[str, str, str, str, str, str]] = set()

    def _add_rows(*, task: dict[str, Any], findings: list[dict[str, Any]]) -> None:
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            signatures.add(
                _candidate_diagnostic_signature(
                    task=task,
                    finding=finding,
                    resources_by_jid=resources_by_jid,
                    parts_by_name=parts_by_name,
                )
            )

    for row in history:
        if not isinstance(row, dict):
            continue
        for evaluation in row.get("candidate_evaluations") or []:
            if not isinstance(evaluation, dict):
                continue
            _add_rows(
                task=dict(evaluation.get("task") or {}),
                findings=[
                    item
                    for item in (evaluation.get("validation_findings") or [])
                    if isinstance(item, dict)
                ],
            )

    for row in feedback_rows:
        if not isinstance(row, dict):
            continue
        _add_rows(
            task=dict(row.get("task") or {}),
            findings=[
                item for item in (row.get("validation_findings") or []) if isinstance(item, dict)
            ],
        )

    return signatures


def _current_recovery_blockers_summary(blockers: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for row in blockers:
        if not isinstance(row, dict):
            continue
        summary = str(row.get("summary") or "").strip()
        if not summary or summary in seen:
            continue
        seen.add(summary)
        lines.append(f"- {summary}")
    return "\n".join(lines) if lines else "(none)"


def _named_pose_tokens(value: Any) -> list[str]:
    tokens: list[str] = []
    if isinstance(value, dict):
        tokens.extend(str(token).strip() for token in value.keys() if str(token).strip())
    else:
        tokens.extend(str(token).strip() for token in (value or []) if str(token).strip())
    deduped: list[str] = []
    for token in tokens:
        if token not in deduped:
            deduped.append(token)
    return deduped




def _pose_xyz_text(pose: dict[str, Any]) -> str:
    if not isinstance(pose, dict) or not pose:
        return "unknown"
    parts: list[str] = []
    for axis in ("x", "y", "z"):
        coord = _float_or_none(pose.get(axis))
        parts.append(f"{axis}=?" if coord is None else f"{axis}={coord:.2f}")
    return ",".join(parts)


def _resource_capabilities_summary(recovery_resources: dict[str, Any]) -> str:
    if not isinstance(recovery_resources, dict) or not recovery_resources:
        return "(none advertised)"

    rows: list[dict[str, Any]] = []
    for resource_jid in sorted(recovery_resources):
        entry = dict(recovery_resources.get(resource_jid) or {})
        recovery_snapshot = dict(entry.get("recovery_snapshot") or {})
        static_capabilities = dict(entry.get("static_capabilities") or {})
        recovery_adapter = dict(
            entry.get("recovery_adapter") or recovery_snapshot.get("recovery_adapter") or {}
        )

        row: dict[str, Any] = {"resource_jid": resource_jid}
        resource_type = (
            entry.get("resource_type")
            or recovery_adapter.get("resource_type")
            or recovery_snapshot.get("resource_type")
            or static_capabilities.get("resource_type")
        )
        if resource_type not in (None, "", [], {}):
            row["resource_type"] = deepcopy(resource_type)
        for field_name, value in recovery_adapter.items():
            if str(field_name).startswith("supports_") and value is True:
                row[str(field_name)] = True
        example_families = recovery_adapter.get("example_families")
        if example_families not in (None, "", [], {}):
            row["example_families"] = deepcopy(example_families)
        for field_name in (
            "named_poses",
            "available_named_poses",
            "reachability",
            "reachable_locations",
        ):
            for source in (recovery_snapshot, static_capabilities):
                value = source.get(field_name)
                if value in (None, "", [], {}):
                    continue
                row[field_name] = deepcopy(value)
                break
        rows.append(row)
    if not rows:
        return "(none advertised)"
    return "\n".join(
        "- "
        + " | ".join(
            f"{field_name}={_inline_json(value)}"
            for field_name, value in row.items()
        )
        for row in rows
    )


def _resource_state_vocabulary_summary(
    *,
    recovery_resources: dict[str, Any],
    recovery_des_models: dict[str, Any],
) -> str:
    """Render public resource-state labels grouped by exact resource type."""
    resource_jids = sorted(set(recovery_resources) | set(recovery_des_models))
    rows_by_resource_type: dict[str, dict[str, Any]] = {}
    for resource_jid in resource_jids:
        entry = dict(recovery_resources.get(resource_jid) or {})
        recovery_adapter = dict(entry.get("recovery_adapter") or {})
        recovery_snapshot = dict(entry.get("recovery_snapshot") or {})
        static_capabilities = dict(entry.get("static_capabilities") or {})
        resource_type = str(
            entry.get("resource_type")
            or recovery_adapter.get("resource_type")
            or recovery_snapshot.get("resource_type")
            or static_capabilities.get("resource_type")
            or resource_jid
        ).strip()
        descriptor = dict(
            recovery_des_models.get(resource_jid)
            or entry.get("recovery_des_model")
            or {}
        )
        values = _modeled_state_values(
            descriptor=descriptor,
            field_name="resource_state",
        )
        if not values:
            continue
        row = rows_by_resource_type.setdefault(
            resource_type,
            {"resource_jids": [], "resource_state": []},
        )
        row["resource_jids"].append(resource_jid)
        for value in values:
            if value not in row["resource_state"]:
                row["resource_state"].append(value)
    if not rows_by_resource_type:
        return "(none advertised)"
    lines: list[str] = []
    for resource_type in sorted(rows_by_resource_type):
        row = rows_by_resource_type[resource_type]
        row["resource_state"].sort()
        lines.append(
            f"- resource_type={_inline_json(resource_type)} | "
            f"resource_jids={_inline_json(row['resource_jids'])}"
        )
        lines.append(
            f"  - resource_state: {_inline_json(row['resource_state'])}"
        )
    return "\n".join(lines)


def _part_state_vocabulary_summary(
    *,
    llm_input: dict[str, Any],
    recovery_resources: dict[str, Any],
    recovery_des_models: dict[str, Any],
) -> str:
    """Render known part-state labels once, independent of validating RAs."""
    values: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str) and value and value not in values:
            values.append(value)

    for condition in llm_input.get("goal_conditions") or []:
        if not isinstance(condition, dict):
            continue
        if str(condition.get("entity_kind") or "").strip() != "part":
            continue
        if str(condition.get("field") or "").strip() in {
            "state",
            "current_state",
            "part_state",
        }:
            add(condition.get("expected"))
    for resource_jid in sorted(set(recovery_resources) | set(recovery_des_models)):
        entry = dict(recovery_resources.get(resource_jid) or {})
        descriptor = dict(
            recovery_des_models.get(resource_jid)
            or entry.get("recovery_des_model")
            or {}
        )
        for value in _modeled_state_values(
            descriptor=descriptor,
            field_name="part_state",
        ):
            add(value)
    values.sort()
    return f"- part_state: {_inline_json(values)}" if values else "(none advertised)"


def _modeled_state_values(
    *,
    descriptor: dict[str, Any],
    field_name: str,
) -> list[str]:
    """Return exact literal state labels used by public modeled transitions."""
    declaration = dict(
        dict(descriptor.get("state_variables") or {}).get(field_name) or {}
    )
    if declaration.get("private") is True:
        return []
    values: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str) and value and value not in values:
            values.append(value)

    for raw_event in descriptor.get("events") or []:
        if not isinstance(raw_event, dict) or raw_event.get("private") is True:
            continue
        guard = dict(dict(raw_event.get("guards") or {}).get(field_name) or {})
        for operator in ("equals", "not_equals"):
            if operator in guard:
                add(guard.get(operator))
        update = dict(dict(raw_event.get("updates") or {}).get(field_name) or {})
        if "set" in update:
            add(update.get("set"))
    values.sort()
    return values


def _resource_exact_fields_summary(
    *,
    recovery_resources: dict[str, Any],
    recovery_des_models: dict[str, Any],
) -> str:
    """Render public resource-scoped exact fields under their validating RA."""
    lines: list[str] = []
    for resource_jid in sorted(set(recovery_resources) | set(recovery_des_models)):
        entry = dict(recovery_resources.get(resource_jid) or {})
        descriptor = dict(
            recovery_des_models.get(resource_jid)
            or entry.get("recovery_des_model")
            or {}
        )
        fields: list[tuple[str, dict[str, Any]]] = []
        for field_name, raw_declaration in dict(
            descriptor.get("state_variables") or {}
        ).items():
            declaration = dict(raw_declaration or {})
            field_token = str(field_name)
            if (
                declaration.get("private") is True
                or field_token.startswith("task_ctx.")
                or field_token in {"resource_state", "part_state", "held_part"}
                or str(declaration.get("scope") or "resource") == "part"
            ):
                continue
            public_declaration = {
                key: deepcopy(declaration.get(key))
                for key in ("scope", "domain")
                if key in declaration
            }
            if field_token in {"resource_location", "part_location"} and isinstance(
                public_declaration.get("domain"), list
            ):
                public_declaration["domain"] = [
                    deepcopy(value)
                    for value in public_declaration["domain"]
                    if value is not None
                ]
            fields.append((field_token, public_declaration))
        if not fields:
            continue
        lines.append(f"- {resource_jid}")
        for field_name, declaration in fields:
            lines.append(f"  - {field_name}: {_inline_json(declaration)}")
    return "\n".join(lines) if lines else "(none advertised)"


def _public_custody_state_is_declared(
    *,
    recovery_resources: dict[str, Any],
    recovery_des_models: dict[str, Any],
) -> bool:
    """Return whether any current RA publicly declares ``held_part``."""
    for resource_jid in sorted(set(recovery_resources) | set(recovery_des_models)):
        entry = dict(recovery_resources.get(resource_jid) or {})
        descriptor = dict(
            recovery_des_models.get(resource_jid)
            or entry.get("recovery_des_model")
            or {}
        )
        declaration = dict(
            dict(descriptor.get("state_variables") or {}).get("held_part") or {}
        )
        if (
            declaration
            and declaration.get("private") is not True
            and str(declaration.get("scope") or "resource") == "resource"
        ):
            return True
    return False


def _outline_resource_fields_text(payload: dict[str, Any]) -> str:
    """Show essential declarations separately from runtime binding semantics."""
    facts = payload.get("outline_resource_fields") or {}
    if not facts:
        return "Request /resources/<jid>/state_variables for declarations before using unknown fields or values."
    lines = ["Saved public declarations; current values are in Current DES State."]
    interpretations: dict[str, list[str]] = {}
    for jid, fact in facts.items():
        if fact.get("missing_context"):
            lines.append(f"- {jid}: {fact['missing_context']}")
            continue
        fields = {name: declaration for name, declaration in fact.get("value", {}).items()
                  if name not in {"resource_state", "part_state"}}
        if fields:
            lines.append(f"- {jid}: {_inline_json(fields)}")
        if fact.get("interpretation"):
            text = _inline_json(fact["interpretation"])
            interpretations.setdefault(text, []).append(jid)
    if interpretations:
        lines.append("Runtime binding interpretation (not an observation or proposed action):")
        for text, jids in interpretations.items():
            lines.append(f"- resource_jids={_inline_json(jids)} | {text}")
    return "\n".join(lines)


def _active_primitive_outline_event(
    session_state: dict[str, Any],
) -> tuple[int, dict[str, Any] | None]:
    accepted_prefix = [
        dict(row)
        for row in (session_state.get("accepted_outline_prefix") or [])
        if isinstance(row, dict)
    ]
    cursor = int(session_state.get("primitive_generation_cursor") or 0)
    if cursor < 0:
        cursor = 0
    if cursor >= len(accepted_prefix):
        return cursor, None
    return cursor, deepcopy(accepted_prefix[cursor])


def _primitive_catalog_description_for_prompt(raw_entry: dict[str, Any]) -> str:
    primitive_kind = str(raw_entry.get("primitive_kind") or "").strip()
    preconditions = dict(raw_entry.get("preconditions") or {})
    effects = dict(raw_entry.get("effects") or {})
    summary_parts: list[str] = []
    if primitive_kind:
        summary_parts.append(f"{primitive_kind} primitive")
    if preconditions:
        summary_parts.append(
            "preconditions: " + ", ".join(sorted(str(key) for key in preconditions))
        )
    if effects:
        summary_parts.append("effects: " + ", ".join(sorted(str(key) for key in effects)))
    if summary_parts:
        return "; ".join(summary_parts)
    return str(raw_entry.get("semantic_summary") or raw_entry.get("description") or "").strip()


def _slim_primitive_catalog_for_prompt(catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    slim: list[dict[str, Any]] = []
    for raw_entry in filter_synthesis_primitive_catalog(catalog or []):
        if not isinstance(raw_entry, dict):
            continue
        name = str(raw_entry.get("name") or "").strip()
        if not name:
            continue
        description = _primitive_catalog_description_for_prompt(raw_entry)
        entry: dict[str, Any] = {
            "name": name,
            "primitive_kind": str(raw_entry.get("primitive_kind") or "").strip(),
            "description": description,
            "params": deepcopy(raw_entry.get("params") or {}),
            "required_params": deepcopy(raw_entry.get("required_params") or []),
        }
        preconditions = deepcopy(raw_entry.get("preconditions") or {})
        effects = deepcopy(raw_entry.get("effects") or {})
        if preconditions:
            entry["preconditions"] = preconditions
        if effects:
            entry["effects"] = effects
        resource_type = str(raw_entry.get("resource_type") or "").strip()
        trace_fact_contract = deepcopy(
            raw_entry.get("trace_fact_contract") or raw_entry.get("trace_facts") or {}
        )
        if not trace_fact_contract and resource_type:
            profile = get_resource_profile(resource_type)
            trace_fact_contract = deepcopy(
                dict(profile.primitive_trace_fact_map or {}).get(name) or {}
            )
        if trace_fact_contract:
            entry["trace_fact_contract"] = trace_fact_contract
        slim.append(entry)
    return slim


def _primitive_signature_card(
    primitive_catalogs_by_resource: dict[str, list[dict[str, Any]]],
) -> str:
    lines: list[str] = []
    for resource_jid in sorted(primitive_catalogs_by_resource.keys()):
        catalog = primitive_catalogs_by_resource.get(resource_jid) or []
        signatures: list[str] = []
        for entry in catalog:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").strip()
            if not name:
                continue
            required = [
                str(item).strip()
                for item in (entry.get("required_params") or [])
                if str(item).strip()
            ]
            if not required:
                params = dict(entry.get("params") or {})
                required = [str(key).strip() for key in params if str(key).strip()]
            signatures.append(f"{name}({','.join(required)})")
        if signatures:
            lines.append(f"- {resource_jid}: " + "; ".join(signatures))
    return "\n".join(lines).strip() or "(none)"


def _trace_fact_spec_text(spec: Any) -> str:
    if isinstance(spec, dict):
        fact_name = str(spec.get("fact") or "").strip()
        when = str(spec.get("when") or "").strip()
    else:
        fact_name = str(spec or "").strip()
        when = ""
    if not fact_name:
        return ""
    text = f"{fact_name}(part)"
    if when:
        text += f" when {when}"
    return text


def _primitive_trace_fact_contract_card(
    primitive_catalogs_by_resource: dict[str, list[dict[str, Any]]],
) -> str:
    lines: list[str] = []
    for resource_jid in sorted(primitive_catalogs_by_resource.keys()):
        fragments: list[str] = []
        for entry in primitive_catalogs_by_resource.get(resource_jid) or []:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").strip()
            contract = dict(entry.get("trace_fact_contract") or {})
            if not name or not contract:
                continue
            establishes = [
                _trace_fact_spec_text(spec) for spec in (contract.get("establishes") or [])
            ]
            requires = [_trace_fact_spec_text(spec) for spec in (contract.get("requires") or [])]
            details: list[str] = []
            if any(establishes):
                details.append("establishes " + ", ".join(item for item in establishes if item))
            if any(requires):
                details.append("requires " + ", ".join(item for item in requires if item))
            if details:
                fragments.append(f"{name}: " + "; ".join(details))
        if fragments:
            lines.append(f"- {resource_jid}: " + " | ".join(fragments))
    return "\n".join(lines).strip() or "(none)"


def _active_event_start_facts_for_prompt(
    *,
    resources: list[dict[str, Any]],
    parts: list[dict[str, Any]],
    active_event: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(active_event, dict) or not active_event:
        return deepcopy(resources), deepcopy(parts)

    resource_jid = str(active_event.get("resource_jid") or "").strip()
    part_name = str(active_event.get("part_name") or "").strip()
    expected_start = dict(active_event.get("expected_start_state") or {})
    projected_resources = [deepcopy(row) for row in resources if isinstance(row, dict)]
    projected_parts = [deepcopy(row) for row in parts if isinstance(row, dict)]

    for resource in projected_resources:
        if str(resource.get("resource_jid") or "").strip() != resource_jid:
            continue
        if "resource_state" in expected_start:
            resource["current_state"] = deepcopy(expected_start.get("resource_state"))
        if "resource_location" in expected_start:
            resource["current_location"] = deepcopy(expected_start.get("resource_location"))
        if "held_part" in expected_start:
            resource["held_part"] = deepcopy(expected_start.get("held_part"))

    for part in projected_parts:
        if str(part.get("part_name") or "").strip() != part_name:
            continue
        if "part_state" in expected_start:
            part["current_state"] = deepcopy(expected_start.get("part_state"))
        if "part_location" in expected_start:
            part["current_location"] = deepcopy(expected_start.get("part_location"))
        if "held_part" in expected_start:
            held_part = str(expected_start.get("held_part") or "").strip()
            part["current_holder_resource_jid"] = (
                resource_jid if part_name and held_part == part_name else None
            )
    return projected_resources, projected_parts


def _primitive_rejection_feedback_summary(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        outline_id = str(row.get("outline_id") or "").strip()
        resource_jid = str(row.get("resource_jid") or "").strip()
        reason = str(row.get("reason") or row.get("error") or "").strip()
        constraint_code = str(row.get("constraint_code") or "").strip()
        label_parts = [item for item in (outline_id, resource_jid) if item]
        label = " / ".join(label_parts) if label_parts else "primitive candidate"
        if constraint_code and reason:
            lines.append(f"- {label} [{constraint_code}]: {reason}")
        elif constraint_code:
            lines.append(f"- {label} [{constraint_code}]")
        elif reason:
            lines.append(f"- {label}: {reason}")
    return "\n".join(lines) if lines else "(none)"


def _primitive_escalation_diagnostics_summary(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        outline_id = str(row.get("outline_id") or "").strip()
        resource_jid = str(row.get("resource_jid") or "").strip()
        part_name = str(row.get("part_name") or "").strip()
        constraint_code = str(row.get("trigger") or "primitive_authoring_stalled").strip()
        reason = str(row.get("reason") or "").strip()
        label_parts = [item for item in (outline_id, resource_jid, part_name) if item]
        label = " / ".join(label_parts) if label_parts else "primitive escalation"
        if constraint_code and reason:
            lines.append(f"- {label} [{constraint_code}]: {reason}")
        elif constraint_code:
            lines.append(f"- {label} [{constraint_code}]")
        elif reason:
            lines.append(f"- {label}: {reason}")
    return "\n".join(lines) if lines else "(none)"


def _top_validation_feedback_section(
    *,
    summary: str,
    label: str = "Previous Validation Feedback (read first)",
) -> list[str]:
    normalized = str(summary or "").strip()
    if not normalized or normalized == "(none)":
        return []
    return [
        "",
        label,
        normalized,
    ]


def _no_progress_revision_instruction(summary: str) -> str:
    """Explain the recorded selection result without prescribing new effects."""
    if "no_progressing_candidate" not in str(summary or ""):
        return ""
    return (
        "\nexcluded_no_progress means the recorded checks did not establish progress. "
        "Use the reported reasons; request missing facts when needed. "
        "Do not claim additional effects merely to satisfy this check."
    )


def _outline_immediate_validation_feedback_summary(
    *,
    outline_validation_findings: list[dict[str, Any]],
    candidate_rejection_feedback: list[dict[str, Any]],
    primitive_escalation_diagnostics: list[dict[str, Any]],
    feedback_render_style: str = "raw_code",
    resources_by_jid: dict[str, dict[str, Any]] | None = None,
    parts_by_name: dict[str, dict[str, Any]] | None = None,
) -> str:
    if outline_validation_findings:
        return _outline_validation_summary(
            outline_validation_findings,
            feedback_render_style=feedback_render_style,
            resources_by_jid=resources_by_jid,
            parts_by_name=parts_by_name,
        )
    if candidate_rejection_feedback:
        return _candidate_rejection_learning_summary(
            history=[],
            feedback_rows=candidate_rejection_feedback,
            feedback_render_style=feedback_render_style,
            resources_by_jid=resources_by_jid,
            parts_by_name=parts_by_name,
        )
    if primitive_escalation_diagnostics and (
        outline_validation_findings or candidate_rejection_feedback
    ):
        return _primitive_escalation_diagnostics_summary(primitive_escalation_diagnostics)
    return "(none)"


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None






def _candidate_grounding_target_refs(
    *,
    recovery_resources: dict[str, Any],
    projected_parts: list[dict[str, Any]],
) -> list[str]:
    tokens: list[str] = []

    for resource_jid in sorted(recovery_resources):
        entry = dict(recovery_resources.get(resource_jid) or {})
        recovery_snapshot = dict(entry.get("recovery_snapshot") or {})
        static_capabilities = dict(entry.get("static_capabilities") or {})
        reachable_locations = [
            str(token).strip()
            for token in (
                static_capabilities.get("reachability")
                or static_capabilities.get("reachable_locations")
                or recovery_snapshot.get("reachability")
                or recovery_snapshot.get("reachable_locations")
                or []
            )
            if str(token).strip()
        ]
        for token in reachable_locations:
            if token not in tokens:
                tokens.append(token)

    for part in projected_parts:
        if not isinstance(part, dict):
            continue
        held_location = ""
        if str(
            part.get("part_holder_resource_jid")
            or part.get("current_holder_resource_jid")
            or ""
        ).strip():
            held_location = str(
                part.get("part_location") or part.get("current_location") or ""
            ).strip()
        for raw_token in (
            part.get("part_location"),
            part.get("current_location"),
        ):
            token = str(raw_token or "").strip()
            if not token or token == held_location:
                continue
            if token not in tokens:
                tokens.append(token)

    return tokens


def _candidate_grounding_facts_summary(
    *,
    recovery_resources: dict[str, Any],
    projected_resources: list[dict[str, Any]],
    projected_parts: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    target_refs = _candidate_grounding_target_refs(
        recovery_resources=recovery_resources,
        projected_parts=projected_parts,
    )
    if target_refs:
        lines.append("- grounded_target_refs=" + ", ".join(target_refs))

    resources_by_jid = {
        str(row.get("resource_jid") or "").strip(): dict(row)
        for row in projected_resources
        if isinstance(row, dict) and str(row.get("resource_jid") or "").strip()
    }

    for resource_jid in sorted(recovery_resources):
        entry = dict(recovery_resources.get(resource_jid) or {})
        recovery_snapshot = dict(entry.get("recovery_snapshot") or {})
        static_capabilities = dict(entry.get("static_capabilities") or {})
        named_poses = _named_pose_tokens(
            recovery_snapshot.get("named_poses")
            or static_capabilities.get("named_poses")
            or recovery_snapshot.get("available_named_poses")
            or static_capabilities.get("available_named_poses")
            or []
        )
        reachable_refs = [
            str(token).strip()
            for token in (
                static_capabilities.get("reachability")
                or static_capabilities.get("reachable_locations")
                or recovery_snapshot.get("reachability")
                or recovery_snapshot.get("reachable_locations")
                or []
            )
            if str(token).strip()
        ]
        resource_row = dict(resources_by_jid.get(resource_jid) or {})
        named_pose_text = ", ".join(named_poses) if named_poses else "null"
        reachable_text = ", ".join(dict.fromkeys(reachable_refs)) if reachable_refs else "null"
        resource_state = str(resource_row.get("resource_state") or "").strip() or "null"
        held_part = str(resource_row.get("held_part") or "").strip() or "null"
        lines.append(
            f"- resource_jid={resource_jid} | resource_state={resource_state} | "
            f"named_poses={named_pose_text} | "
            f"grounded_target_refs={reachable_text} | held_part={held_part}"
        )

    for part in projected_parts:
        if not isinstance(part, dict):
            continue
        part_name = str(part.get("part_name") or "").strip()
        if not part_name:
            continue
        part_state = str(part.get("part_state") or "").strip() or "null"
        part_holder = str(part.get("part_holder_resource_jid") or "").strip() or "null"
        part_location = str(part.get("part_location") or "").strip() or "null"
        source_ref = "observed_pose" if dict(part.get("observed_pose") or {}) else "null"
        lines.append(
            f"- part_name={part_name} | part_state={part_state} | "
            f"part_holder_resource_jid={part_holder} | part_location={part_location} | "
            f"source_ref={source_ref}"
        )

    return "\n".join(lines) if lines else "(none)"


_PRUNED_ACTION_DURABLE_CONSTRAINT_CODES = {
    "source_reference_unavailable",
    "safety_rule_violation",
    "blocker_open",
    "dependency_unsatisfied",
    "order_violation",
}


def _finding_currently_applicable_for_pruned_actions(
    finding: dict[str, Any],
    *,
    task: dict[str, Any],
    resources_by_jid: dict[str, dict[str, Any]],
    parts_by_name: dict[str, dict[str, Any]],
) -> bool:
    if not _finding_durable(finding):
        return False
    stage = str(finding.get("stage") or "").strip().lower()
    constraint_code = str(finding.get("constraint_code") or "").strip().lower()
    if not constraint_code and not stage:
        return False

    resource_jid = str(task.get("resource_jid") or finding.get("resource_jid") or "").strip()
    part_name = str(task.get("part_name") or finding.get("part_name") or "").strip()
    resource_row = dict(resources_by_jid.get(resource_jid) or {})
    part_row = dict(parts_by_name.get(part_name) or {})

    if stage in {"event_enabledness", "plant_enabledness"}:
        unsatisfied_predicates = {
            str(item).strip()
            for item in (finding.get("unsatisfied_predicates") or [])
            if str(item).strip()
        }
        if any(predicate.startswith("holds(") for predicate in unsatisfied_predicates):
            actual_held = str(resource_row.get("held_part") or "").strip()
            current_holder = str(part_row.get("current_holder_resource_jid") or "").strip()
            return bool(
                resource_jid
                and part_name
                and (actual_held != part_name or current_holder != resource_jid)
            )
        if any(
            predicate.startswith("observed_pose(") or predicate.startswith("available_source(")
            for predicate in unsatisfied_predicates
        ):
            observed_pose = dict(part_row.get("observed_pose") or {})
            current_holder = str(part_row.get("current_holder_resource_jid") or "").strip()
            current_location = str(part_row.get("current_location") or "").strip()
            return bool(
                part_name and not current_holder and not current_location and not observed_pose
            )
        return True

    if stage == "supervisor_admissibility":
        return True

    if constraint_code == "part_relocation_without_carrier":
        actual_held = str(resource_row.get("held_part") or "").strip()
        current_holder = str(part_row.get("current_holder_resource_jid") or "").strip()
        return bool(
            resource_jid
            and part_name
            and current_holder != resource_jid
            and actual_held != part_name
        )

    if constraint_code == "source_reference_unavailable":
        observed_pose = dict(part_row.get("observed_pose") or {})
        current_holder = str(part_row.get("current_holder_resource_jid") or "").strip()
        current_location = str(part_row.get("current_location") or "").strip()
        return bool(part_name and not current_holder and not current_location and not observed_pose)

    return True


def _history_derived_pruned_actions_summary(
    *,
    pruned_actions: list[dict[str, Any]],
    rejection_history: list[dict[str, Any]],
    candidate_rejection_history: list[dict[str, Any]],
    candidate_rejection_feedback: list[dict[str, Any]],
    projected_resources: list[dict[str, Any]],
    projected_parts: list[dict[str, Any]],
    feedback_render_style: str = "raw_code",
    excluded_diagnostic_signatures: set[tuple[str, str, str, str, str, str]] | None = None,
) -> str:
    line_by_key: dict[tuple[Any, ...], str] = {}
    excluded_diagnostic_signatures = set(excluded_diagnostic_signatures or set())
    resources_by_jid = {
        str(row.get("resource_jid") or "").strip(): dict(row)
        for row in projected_resources
        if isinstance(row, dict) and str(row.get("resource_jid") or "").strip()
    }
    parts_by_name = {
        str(row.get("part_name") or "").strip(): dict(row)
        for row in projected_parts
        if isinstance(row, dict) and str(row.get("part_name") or "").strip()
    }

    def _add_row(
        *,
        task: dict[str, Any],
        findings: list[dict[str, Any]],
        explicit_summary: str = "",
        explicit_reason: str = "",
    ) -> None:
        applicable_findings = [
            dict(item)
            for item in findings
            if isinstance(item, dict)
            and _finding_currently_applicable_for_pruned_actions(
                dict(item),
                task=task,
                resources_by_jid=resources_by_jid,
                parts_by_name=parts_by_name,
            )
        ]
        if not applicable_findings:
            return

        if feedback_render_style == "des_event_diagnostic":
            for finding in applicable_findings:
                signature = _candidate_diagnostic_signature(
                    task=task,
                    finding=finding,
                    resources_by_jid=resources_by_jid,
                    parts_by_name=parts_by_name,
                )
                if signature in excluded_diagnostic_signatures:
                    continue
                key = signature
                if key in line_by_key:
                    line_by_key.pop(key, None)
                line_by_key[key] = _render_des_event_diagnostic(
                    task=task,
                    finding=finding,
                    resources_by_jid=resources_by_jid,
                    parts_by_name=parts_by_name,
                )
            return

        summary = str(explicit_summary or "").strip()
        reason = str(explicit_reason or "").strip()
        if not summary:
            summary = _structured_task_action_summary(task)
        if not reason:
            codes = [
                str(item.get("constraint_code") or "").strip()
                for item in applicable_findings
                if isinstance(item, dict) and str(item.get("constraint_code") or "").strip()
            ]
            reason = "/".join(dict.fromkeys(codes))

        resource_jid = str(dict(task or {}).get("resource_jid") or "").strip()
        part_name = str(dict(task or {}).get("part_name") or "").strip()
        key = (resource_jid, part_name, summary, reason)

        label_parts = [item for item in (resource_jid, part_name) if item]
        label = " / ".join(label_parts) if label_parts else "rejected action"
        sentence = f"- {label}"
        if summary:
            sentence += f" -> {summary}"
        # Show only durable constraint codes (supervisor facts, not hints).
        if reason:
            durable_codes = {
                code.strip()
                for code in reason.split("/")
                if code.strip() in _PRUNED_ACTION_DURABLE_CONSTRAINT_CODES
            }
            if durable_codes:
                sentence += f" [{'/'.join(sorted(durable_codes))}]"
        if key in line_by_key:
            line_by_key.pop(key, None)
        line_by_key[key] = sentence

    for row in pruned_actions or []:
        if not isinstance(row, dict):
            continue
        _add_row(
            task=dict(row.get("task") or row.get("action") or {}),
            findings=[dict(row.get("guard") or row)] if isinstance(row, dict) else [],
            explicit_summary=str(row.get("summary") or "").strip(),
            explicit_reason=str(row.get("reason") or "").strip(),
        )

    for row in rejection_history or []:
        if not isinstance(row, dict):
            continue
        _add_row(
            task=dict(row.get("proposed_next_transition") or {}),
            findings=[
                item for item in (row.get("validation_findings") or []) if isinstance(item, dict)
            ],
        )

    for row in candidate_rejection_history or []:
        if not isinstance(row, dict):
            continue
        for evaluation in row.get("candidate_evaluations") or []:
            if not isinstance(evaluation, dict):
                continue
            _add_row(
                task=dict(evaluation.get("task") or {}),
                findings=[
                    item
                    for item in (evaluation.get("validation_findings") or [])
                    if isinstance(item, dict)
                ],
            )

    for row in candidate_rejection_feedback or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("constraint_code") or "").strip():
            _add_row(task={}, findings=[row])
            continue
        _add_row(
            task=dict(row.get("task") or {}),
            findings=[
                item for item in (row.get("validation_findings") or []) if isinstance(item, dict)
            ],
        )

    lines = list(line_by_key.values())[-8:]
    return "\n".join(lines) if lines else "(none)"


def _projected_outline_resources(
    *,
    llm_input: dict[str, Any],
    session_state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Render projected resource state from symbolic state when available."""
    symbolic_resources = dict(session_state.get("symbolic_resources") or {})
    if symbolic_resources:
        return [
            deepcopy(symbolic_resources[resource_jid])
            for resource_jid in sorted(symbolic_resources)
            if isinstance(symbolic_resources[resource_jid], dict)
        ]
    observed_runtime_state = dict(llm_input.get("observed_runtime_state") or {})
    rows = [
        deepcopy(row)
        for row in (observed_runtime_state.get("resources") or [])
        if isinstance(row, dict)
    ]
    return sorted(rows, key=lambda row: str(row.get("resource_jid") or ""))


def _projected_outline_parts(
    *,
    llm_input: dict[str, Any],
    session_state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Render projected part state from symbolic state when available."""
    symbolic_parts = dict(session_state.get("symbolic_parts") or {})
    base_parts = [
        deepcopy(row)
        for row in (
            symbolic_parts.values() if symbolic_parts else (llm_input.get("part_facts") or [])
        )
        if isinstance(row, dict)
    ]

    parts_by_name: dict[str, dict[str, Any]] = {}
    for row in base_parts:
        part_name = str(row.get("part_name") or "").strip()
        if not part_name:
            continue
        parts_by_name[part_name] = deepcopy(row)

    for entry in dict(session_state.get("observation_store") or {}).values():
        if not isinstance(entry, dict):
            continue
        part_name = str(entry.get("part_name") or "").strip()
        if not part_name:
            continue
        is_new_part = part_name not in parts_by_name
        if part_name not in parts_by_name:
            parts_by_name[part_name] = {"part_name": part_name}
        part_row = parts_by_name[part_name]
        pose = dict(entry.get("pose") or {})
        if not pose and entry.get("x") is not None:
            pose = {"x": entry.get("x"), "y": entry.get("y"), "z": entry.get("z")}
        if pose and part_row.get("observed_pose_current") is not False:
            part_row["observed_pose"] = deepcopy(pose)
        if (
            is_new_part
            and part_row.get("current_location") in (None, "")
            and entry.get("current_location") not in (None, "")
        ):
            part_row["current_location"] = deepcopy(entry.get("current_location"))
        holder = str(entry.get("current_holder_resource_jid") or "").strip()
        if (
            is_new_part
            and not str(part_row.get("current_holder_resource_jid") or "").strip()
            and holder
        ):
            part_row["current_holder_resource_jid"] = holder

    return [deepcopy(parts_by_name[name]) for name in sorted(parts_by_name)]


def _slim_fault_event(fault_event: dict[str, Any]) -> dict[str, Any]:
    """Expose recorded failure fields without forwarding capture metadata."""
    fields = {
        "focused_resource_jid", "blocked_at_task_id", "blocked_at_function",
        "resource_state", "current_location", "failure_type", "affected_part_names",
    }
    result = {key: deepcopy(value) for key, value in fault_event.items() if key in fields}
    checkpoint = fault_event.get("checkpoint")
    if isinstance(checkpoint, dict):
        public_checkpoint = {}
        if isinstance(checkpoint.get("name"), str):
            public_checkpoint["name"] = checkpoint["name"]
        for field, types in {
            "simulation_started_sec": (int, float), "simulation_elapsed_sec": (int, float),
            "processing_time_sec": (int, float), "processing_fraction": (int, float),
            "process": (str,), "result": (str,), "process_completed": (bool,), "observation_status": (str,),
        }.items():
            if field in checkpoint and (checkpoint[field] is None or type(checkpoint[field]) in types):
                public_checkpoint[field] = checkpoint[field]
        completed = checkpoint.get("completed_task")
        processing = checkpoint.get("processing_observations")
        if isinstance(processing, dict):
            from cais_spade_llm.resources.machine.machine_jobs import public_processing_observations
            public_checkpoint["processing_observations"] = {
                rid: public_processing_observations(record) for rid, record in processing.items()}
        if isinstance(completed, dict) and completed.get("status") == "completed":
            public_checkpoint["completed_task"] = {key: value for key, value in completed.items()
                if key in {"task_id", "resource_id", "event_name", "status"} and isinstance(value, str)}
        if public_checkpoint:
            result["checkpoint"] = public_checkpoint
    return result


def _slim_resource_facts(resources: list[Any]) -> list[dict[str, Any]]:
    """Keep only fields needed for outline reasoning."""
    _keep = {
        "resource_jid",
        "current_state",
        "current_location",
        "held_part",
        "held_part_location",
        "current_pose",
        "current_pose_ref",
    }
    rows = [
        {k: v for k, v in dict(row).items() if k in _keep}
        for row in resources
        if isinstance(row, dict)
    ]
    for row in rows:
        if row.get("held_part") in (None, ""):
            row.pop("held_part_location", None)
    return sorted(rows, key=lambda row: str(row.get("resource_jid") or ""))


def _slim_part_facts(parts: list[Any]) -> list[dict[str, Any]]:
    """Keep only fields relevant to grounding decisions."""
    _keep = {
        "part_name",
        "current_state",
        "current_location",
        "observed_pose",
        "current_holder_resource_jid",
        "origin_location",
    }
    rows = [
        {k: v for k, v in dict(row).items() if k in _keep} for row in parts if isinstance(row, dict)
    ]
    return sorted(rows, key=lambda row: str(row.get("part_name") or ""))


def _prompt_outline_resource_view(resource: dict[str, Any]) -> dict[str, Any]:
    """Render one resource row in outline-contract vocabulary."""
    if not isinstance(resource, dict):
        return {}
    view: dict[str, Any] = {}
    resource_jid = str(resource.get("resource_jid") or "").strip()
    if resource_jid:
        view["resource_jid"] = resource_jid
    if "resource_state" in resource:
        view["resource_state"] = deepcopy(resource.get("resource_state"))
    elif "current_state" in resource:
        view["resource_state"] = deepcopy(resource.get("current_state"))
    if "held_part" in resource:
        view["held_part"] = deepcopy(resource.get("held_part"))
    if resource.get("held_part") not in (None, "") and "held_part_location" in resource:
        view["held_part_location"] = deepcopy(resource.get("held_part_location"))
    if "resource_location" in resource:
        view["resource_location"] = deepcopy(resource.get("resource_location"))
    elif "current_location" in resource:
        view["resource_location"] = deepcopy(resource.get("current_location"))
    if "current_pose" in resource:
        view["current_pose"] = deepcopy(resource.get("current_pose"))
    if "current_pose_captured_at" in resource:
        view["current_pose_captured_at"] = deepcopy(resource.get("current_pose_captured_at"))
    return view


def _prompt_outline_part_view(part: dict[str, Any]) -> dict[str, Any]:
    """Render one part row in outline-contract vocabulary."""
    if not isinstance(part, dict):
        return {}
    view: dict[str, Any] = {}
    part_name = str(part.get("part_name") or "").strip()
    if part_name:
        view["part_name"] = part_name
    if "part_state" in part:
        view["part_state"] = deepcopy(part.get("part_state"))
    elif "current_state" in part:
        view["part_state"] = deepcopy(part.get("current_state"))
    if "part_location" in part:
        view["part_location"] = deepcopy(part.get("part_location"))
    elif "current_location" in part:
        view["part_location"] = deepcopy(part.get("current_location"))
    if "part_holder_resource_jid" in part:
        view["part_holder_resource_jid"] = deepcopy(part.get("part_holder_resource_jid"))
    elif "current_holder_resource_jid" in part:
        view["part_holder_resource_jid"] = deepcopy(part.get("current_holder_resource_jid"))
    if "observed_pose" in part:
        view["observed_pose"] = deepcopy(part.get("observed_pose"))
    for field in ("observed_pose_current", "last_observed_pose", "projected_pose"):
        if field in part:
            view[field] = deepcopy(part[field])
    if "location_basis" in part:
        view["location_basis"] = deepcopy(part.get("location_basis"))
    return view


def _outline_prompt_resource_facts(resources: list[Any]) -> list[dict[str, Any]]:
    return [_prompt_outline_resource_view(dict(row)) for row in resources if isinstance(row, dict)]


def _outline_prompt_part_facts(parts: list[Any]) -> list[dict[str, Any]]:
    return [_prompt_outline_part_view(dict(row)) for row in parts if isinstance(row, dict)]


def _fact_rows(rows: list[dict[str, Any]]) -> str:
    """Keep each entity on its own readable line without repeating JSON indentation."""
    return "\n".join(_inline_json(row) for row in rows) or "(none recorded)"


def _resource_index_text(rows: list[dict[str, Any]]) -> str:
    """Show all recorded resource alternatives with shared column names."""
    if not rows:
        return "(none recorded)"
    return _fact_rows(rows)


def _render_grounding_prompt(payload: dict[str, Any]) -> str:
    llm_input = dict(payload.get("llm_input") or {})
    session_state = dict(payload.get("session_state") or {})
    world_observation_surface = dict(payload.get("world_observation_surface") or {})
    observed_runtime_state = dict(llm_input.get("observed_runtime_state") or {})
    observation_store = dict(session_state.get("observation_store") or {})

    sections: list[str] = [
        "Task and Role",
        (
            "You are the active replanner for a DES fallback recovery session.\n"
            "Current phase: Observation / State Estimation.\n"
            "Assess the current world state and decide whether you have enough "
            "grounded state predicates to synthesize a recovery event sequence, "
            "or whether you need additional observations first."
        ),
        "",
        "Fault Event",
        _compact_json(_slim_fault_event(llm_input.get("fault_event") or {})),
        "",
        "Current Resource Facts",
        _compact_json(_slim_resource_facts(observed_runtime_state.get("resources") or [])),
        "",
        "Current Part Facts",
        _compact_json(_slim_part_facts(llm_input.get("part_facts") or [])),
        "",
        "Safety Rules",
        _compact_safety_rules(llm_input),
        "",
        "Recovery Goals",
        _compact_recovery_goals(
            llm_input,
            projected_parts=[
                deepcopy(row)
                for row in (llm_input.get("part_facts") or [])
                if isinstance(row, dict)
            ],
            current_recovery_blockers=[],
        ),
    ]

    if world_observation_surface:
        sections.extend(
            [
                "",
                "World Observation Surface",
                _compact_json(world_observation_surface),
            ]
        )

    if observation_store:
        sections.extend(
            [
                "",
                "Session Observation Store",
                _compact_json(observation_store),
            ]
        )

    phase_feedback = [
        deepcopy(row)
        for row in (session_state.get("phase_feedback") or [])
        if isinstance(row, dict) and str(row.get("phase") or "").strip() == "grounding"
    ]
    if phase_feedback:
        sections.extend(
            [
                "",
                "Prior Grounding Feedback",
                _compact_json(phase_feedback),
            ]
        )

    sections.extend(
        [
            "",
            "Decision Rules",
            "- Keep `thought` to at most two short sentences.",
            "- If the current resource facts and part facts provide enough grounded "
            'information to plan recovery, set decision to "grounded".',
            '- When decision is "grounded", leave observe_requests empty.',
            "- Do not request facts already present in the current session observations.",
            '- To request an observation, set decision to "observe" and include '
            "observe_requests with fact_type and entity.",
        ]
    )

    return "\n".join(sections).strip() + "\n"


# ---------------------------------------------------------------------------
# Primitive-generation prompt
# ---------------------------------------------------------------------------


def _render_primitive_generation_prompt(payload: dict[str, Any]) -> str:
    session_state = dict(payload.get("session_state") or {})
    active_event_token = dict(payload.get("primitive_active_outline_event_token") or {})
    active_event_full = dict(payload.get("primitive_active_outline_event") or {})
    if not active_event_token or not active_event_full:
        _cursor, fallback_active_event = _active_primitive_outline_event(session_state)
        if fallback_active_event:
            from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes.multi_turn_primitive_generation import (
                _compact_active_event_token,
                _primitive_authoring_event_context,
            )

            if not active_event_token:
                active_event_token = _compact_active_event_token(fallback_active_event)
            if not active_event_full:
                active_event_full = _primitive_authoring_event_context(fallback_active_event)
    accepted_outline_prefix = [
        dict(row)
        for row in (session_state.get("accepted_outline_prefix") or [])
        if isinstance(row, dict)
    ]
    cursor_state = dict(payload.get("primitive_generation_cursor_state") or {})
    if not cursor_state:
        accepted_prefix_len = len(session_state.get("accepted_outline_prefix") or [])
        cursor_idx = int(session_state.get("primitive_generation_cursor") or 0)
        cursor_state = {
            "active_index": cursor_idx,
            "accepted_outline_count": accepted_prefix_len,
            "remaining_events_count": max(accepted_prefix_len - cursor_idx, 0),
        }

    visible_catalog = [
        dict(row)
        for row in (payload.get("primitive_visible_catalog") or [])
        if isinstance(row, dict)
    ]
    published_ref_schema = [
        dict(row)
        for row in (payload.get("primitive_published_ref_schema") or [])
        if isinstance(row, dict)
    ]
    served_context = dict(payload.get("primitive_served_context") or {})
    context_errors = [
        dict(row)
        for row in (payload.get("primitive_context_errors") or [])
        if isinstance(row, dict)
    ]
    input_diagnostics = [
        dict(row)
        for row in (payload.get("primitive_input_diagnostics") or [])
        if isinstance(row, dict)
    ]
    capability_names = [
        str(name).strip()
        for name in (payload.get("primitive_capability_decomposition_names") or [])
        if str(name).strip()
    ]
    active_resource_named_poses = [
        str(name).strip()
        for name in (payload.get("primitive_active_resource_named_poses") or [])
        if str(name).strip()
    ]
    suggested_capability_names = [
        str(name).strip()
        for name in (payload.get("primitive_suggested_capability_decompositions") or [])
        if str(name).strip()
    ]
    memo_summary = [
        dict(row)
        for row in (payload.get("primitive_authoring_memo_summary") or [])
        if isinstance(row, dict)
    ]

    primitive_feedback = _primitive_rejection_feedback_summary(
        [
            dict(row)
            for row in (session_state.get("primitive_rejection_feedback") or [])
            if isinstance(row, dict)
        ]
    )

    sections: list[str] = [
        "Task and Role",
        (
            "You are the active replanner authoring primitive_steps for a DES "
            "fallback recovery session.\n"
            "Current phase: Event-to-Primitive Authoring (event-local).\n"
            "Author primitive_steps for ONLY the active DES transition, using only "
            "the visible primitive catalog for the active resource. There is no "
            "mechanical composer behind you: if you emit no primitive_steps, no "
            "trace is produced for this event.\n"
            "RA primitive_support records independent capability witnesses. Primitive Composition "
            "must still establish ordering, compatible parameter bindings, and achievement of "
            "the complete expected_end_state. An accepted outline is not an executable program.\n"
            "The active event body and the surrounding accepted outline are "
            "provided inline below so you can reason about intent immediately. "
            "For peripheral facts (poses, contract cards, capability decompositions, "
            "projected snapshots, prior decompositions), emit decision=need_context "
            "with refs in context_requests and the recovery will resolve them in the "
            "next turn under 'Served Context'. Do not guess values. Treat "
            "/capability_decompositions/<function_name> as the primary modeled "
            "source for how this resource composes valid low-level actions. When "
            "capability decompositions are available for the active resource, "
            "retrieve the closest applicable one before authoring primitive_steps. "
            "Adapt the retrieved decomposition to the served context instead of "
            "inventing a fresh sequence. The <function_name> token must be one "
            "of the names listed under 'Available Capability Decompositions'; "
            "primitive names such as grasp_part or release_part belong under "
            "/primitive_contracts/<name>, not /capability_decompositions/<function_name>. "
            "Examples of refs worth pulling: "
            "/primitive_contracts/<name> (full contract card), "
            "/resources/<jid>/current_pose, /parts/<name>/observed_pose, "
            "/projected_snapshot/<jid>, /capability_decompositions/<function_name> "
            "(primary modeled task-function decompositions for this resource), /memo/primitive_authoring "
            "(prior accepted decompositions)."
        ),
    ]
    sections.extend(_top_validation_feedback_section(summary=primitive_feedback))
    sections.extend(
        [
            "",
            "Active DES Transition (token)",
            _compact_json(active_event_token),
            "",
            "Active DES Transition (full event)",
            _compact_json(active_event_full) if active_event_full else "(unavailable)",
            "",
            "Transition Authoring Cursor",
            _compact_json(cursor_state),
        ]
    )

    if accepted_outline_prefix:
        sections.extend(
            [
                "",
                "Accepted Outline Summary (context for active event)",
                _outline_task_sequence_summary(
                    accepted_outline_prefix,
                    include_descriptions=True,
                ),
            ]
        )

    sections.extend(
        [
            "",
            "Visible Primitive Catalog (names only; full contract via /primitive_contracts/<name>)",
            _compact_json(visible_catalog),
            "",
            "Published Ref Schema (for context_requests)",
            _compact_json(published_ref_schema),
        ]
    )

    if capability_names:
        sections.extend(
            [
                "",
                "Available Capability Decompositions (names only; retrieve the closest applicable one before authoring)",
                _compact_json(capability_names),
            ]
        )
    if suggested_capability_names:
        sections.extend(
            [
                "",
                "Suggested Capability Decompositions For This Event",
                _compact_json(suggested_capability_names),
            ]
        )

    if memo_summary:
        sections.extend(
            [
                "",
                "Prior Accepted Decompositions (session-scoped memo; full via /memo/primitive_authoring)",
                _compact_json(memo_summary),
            ]
        )

    if served_context:
        sections.extend(
            [
                "",
                "Served Context (retrieved earlier in this event)",
                _compact_json(served_context),
            ]
        )

    if context_errors:
        sections.extend(
            [
                "",
                "Context Request Errors (previous turn)",
                _compact_json(context_errors),
            ]
        )

    if input_diagnostics:
        sections.extend(
            [
                "",
                "Input Diagnostics",
                _compact_json(input_diagnostics),
            ]
        )

    active_resource_jid = str(active_event_token.get("resource_jid") or "").strip()
    sections.extend(
        [
            "",
            "Named Pose Rules",
            _compact_json(active_resource_named_poses),
            "- move_to_named_pose may only use named poses advertised for the active resource.",
            (
                "- Full named-pose details remain retrievable via "
                f"/resources/{active_resource_jid}/static_capabilities."
                if active_resource_jid
                else "- Full named-pose details remain retrievable via /resources/<jid>/static_capabilities."
            ),
        ]
    )

    sections.extend(
        [
            "",
            "Output Contract",
            "- Keep `thought` to at most two short sentences and `rationale` to one short sentence.",
            "- Author primitive_steps for ONLY the active DES transition shown above.",
            "- outline_id and resource_jid MUST exactly match the active transition.",
            "- primitive_steps[*].primitive MUST be a name in Visible Primitive Catalog; hidden primitives (e.g. move_pose, get_current_pose) are rejected.",
            '- Each step MUST include params; bind params to grounded values via {"context_ref": "event_facts.<path>"} referencing deterministic event-local facts from prior data-producing primitives, or via literal values derived from retrieved poses. Do NOT hardcode pose/offset numeric constants.',
            "- Data-producing primitives publish event_facts automatically: get_current_pose -> event_facts.current_pose; detect_parts(part_name=P) -> event_facts.detected_part.P; compute_pick_targets(part_name=P) -> event_facts.pick_targets.P; compute_place_targets(part_name=P) -> event_facts.place_targets.P.",
            "- Do not include store_as or any per-step alias field. Action primitives do not publish custom event_facts.",
            "- If you need a grounded value (current_pose, observed_pose, contract card, projected snapshot, full event body), emit decision=need_context with refs in context_requests and leave primitive_steps empty; do not guess.",
            "- When capability decompositions are available for the active resource, retrieve the closest applicable /capability_decompositions/<function_name> before authoring primitive_steps.",
            "- /capability_decompositions/<function_name> accepts only names from Available Capability Decompositions. Primitive names like grasp_part and release_part must be retrieved via /primitive_contracts/<name> instead.",
            "- Prefer adapting a retrieved capability decomposition to served context over authoring a novel low-level sequence from scratch.",
            "- Author from scratch only when no retrieved capability decomposition fits; name that gap explicitly in rationale.",
            "- If retrieved context or input diagnostics still contradict the active event, emit decision=primitive_blocked and explain the blocker; do not send the event back to outline from this phase.",
            "- Use decision=primitive_steps_ready only when primitive_steps is non-empty AND every param is either a literal known-safe value, a context_ref to a prior step output, or a value derived from previously served context.",
            "- Use decision=need_primitive_revision with a non-empty rationale when the visible catalog cannot safely satisfy the active event (name the contract gap, e.g. 'no visible primitive establishes part orientation for flipped-SG insert').",
            "- Use decision=primitive_blocked when the active event is blocked by contradictory or insufficient primitive-side context that should pause for operator inspection.",
            "- Do not invent observations, resources, parts, or grounded locations. Do not produce a fragile-but-valid trace.",
        ]
    )
    return "\n".join(sections).strip() + "\n"


# ---------------------------------------------------------------------------
# Outline prompt
# ---------------------------------------------------------------------------


def _outline_safety_checks_for_prompt(checks: list[Any]) -> list[dict[str, Any]]:
    """Keep rule-local Boolean counterexamples without private monitor models."""
    rows = []
    for check in checks:
        if not isinstance(check, dict) or check.get("is_safe") is True:
            continue
        counterexample = check.get("counterexample")
        if not isinstance(counterexample, dict) or not isinstance(counterexample.get("rule_id"), str):
            continue
        values = counterexample.get("ap_values")
        if not isinstance(values, dict) or not values or any(
            not isinstance(label, str) or not (type(value) is bool or isinstance(value, list)
                and value and all(type(item) is bool for item in value)) for label, value in values.items()
        ):
            continue
        public = {"rule_id": counterexample["rule_id"], "ap_values": deepcopy(values)}
        for key in ("possible_violation", "definite_violation"):
            if type(counterexample.get(key)) is bool:
                public[key] = counterexample[key]
        if type(counterexample.get("cell_index")) is int and counterexample["cell_index"] >= 0:
            public["cell_index"] = counterexample["cell_index"]
        interval = counterexample.get("interval")
        if isinstance(interval, list) and len(interval) == 2 and all(
            type(value) in {int, float}
            or isinstance(value, str) and re.fullmatch(r"-?\d+(?:/[1-9]\d*)?", value)
            for value in interval
        ):
            public["interval"] = deepcopy(interval)
        matching = [row for row in check.get("rule_checks", []) if isinstance(row, dict)
                    and row.get("rule_id") == counterexample["rule_id"] and row.get("ap_values") == values]
        observed = matching[-1] if matching else counterexample
        for key in ("observation_index", "phase", "time"):
            if type(observed.get(key)) in {int, float, str}:
                public[key] = observed[key]
        row = {"counterexample": public}
        if check.get("validation_status") in {"INFEASIBLE", "NEEDS_CONTEXT", "rejected", "unresolved"}:
            row["validation_status"] = check["validation_status"]
        if isinstance(check.get("source_path"), str):
            row["source_path"] = check["source_path"]
        rows.append(row)
    return rows


def _outline_public_evidence(evidence: dict[str, Any], public_fields: set[str]) -> dict[str, Any]:  # noqa: C901
    """Retain typed public facts rather than opaque validator payloads."""
    def pose(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        return {key: item for key, item in value.items() if key in {"x", "y", "z", "qx", "qy", "qz", "qw"}
                and type(item) in {int, float}}

    def reach(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        result = {key: item for key, item in value.items()
                  if key in {"max_xy_radius_m", "z_min_m", "z_max_m", "tolerance_m"} and type(item) in {int, float}}
        if isinstance(value.get("frame"), str):
            result["frame"] = value["frame"]
        if isinstance(value.get("origin_pose"), dict):
            result["origin_pose"] = pose(value["origin_pose"])
        return result

    result = {}
    conflicts = [{key: deepcopy(row[key]) for key in ("field", "actual", "expected")}
                 for row in evidence.get("parameter_conflicts", []) if isinstance(row, dict)
                 and row.get("field") in public_fields | {"part_name"}
                 and all(key in row and (row[key] is None or type(row[key]) in (str, bool, int, float))
                         for key in ("actual", "expected"))]
    if conflicts:
        result["parameter_conflicts"] = conflicts
    missing_fields = evidence.get("missing_candidate_fields")
    if isinstance(missing_fields, list):
        fields = sorted({field for field in missing_fields if field == "part_name"})
        if fields:
            result["missing_candidate_fields"] = fields
    conditions = evidence.get("condition_support")
    if isinstance(conditions, dict):
        rows = [{"field": row["field"], "actual": deepcopy(row.get("actual")), "supported": row.get("supported")}
                for row in conditions.get("conditions", []) if isinstance(row, dict)
                and row.get("field") in public_fields
                and (row.get("actual") is None or type(row.get("actual")) in (str, int, float, bool))
                and (row.get("supported") is None or type(row.get("supported")) is bool)]
        if rows:
            result["condition_support"] = {"conditions": rows,
                "combination_supported": conditions.get("combination_supported") is True,
                "individual_conditions_supported": all(row["supported"] is True for row in rows),
                "source": {"descriptor_fingerprint": (conditions.get("source") or {}).get("descriptor_fingerprint")}}
            combinations = []
            for combination in conditions.get("unsupported_combinations", []):
                if (isinstance(combination, list) and len(combination) == 2
                        and all(isinstance(item, dict) and item.get("field") in public_fields
                            and (item.get("actual") is None or type(item.get("actual")) in (str, int, float, bool))
                            for item in combination)):
                    combinations.append([{key: deepcopy(item[key]) for key in ("field", "actual") if key in item}
                                         for item in combination])
            if combinations:
                result["condition_support"]["unsupported_combinations"] = combinations
    capability = evidence.get("capability_result")
    if isinstance(capability, dict) and capability.get("feasibility_status") == "INFEASIBLE":
        checks = []
        guards = [{key: deepcopy(row[key]) for key in
                   ('resource_id', 'field', 'operator', 'expected', 'actual', 'declaration_fingerprint')}
            for row in capability.get('guard_failures', []) if isinstance(row, dict)
            and row.get('field') in public_fields and isinstance(row.get('resource_id'), str)
            and isinstance(row.get('declaration_fingerprint'), str)
            and row.get('operator') in {'equals', 'not_equals', 'equals_from_param', 'not_equals_from_param'}
            and all(key in row and (row[key] is None or type(row[key]) in (str, bool, int, float))
                    for key in ('actual', 'expected'))]
        if guards:
            result['capability_result'] = {'feasibility_status': 'INFEASIBLE', 'guard_failures': guards}
        for check in capability.get("checks", []):
            if (not isinstance(check, dict) or check.get("feasibility_status") != "INFEASIBLE"
                    or not (_outline_recorded_reach_reason(check.get("reason"))
                        or check.get("reason") == "The calculated target is outside the recorded owner motion capability."
                        and check.get("pose_source") == "calculated owner motion target")):
                continue
            checks.append({"feasibility_status": check["feasibility_status"],
                "reason": "The calculated target is outside the recorded owner motion capability.",
                "checked_pose": pose(check.get("checked_pose")),
                "pose_source": "calculated owner motion target",
                **{key: check[key] for key in ("motion_index", "target_index") if type(check.get(key)) is int},
                **({"frame": check["frame"]} if isinstance(check.get("frame"), str) else {})})
        if checks:
            result["capability_result"] = {"feasibility_status": "INFEASIBLE", "checks": checks}
            if _outline_recorded_reach_reason(capability.get("reason")):
                result["capability_result"]["reason"] = "The calculated target is outside the recorded owner motion capability."
            source = evidence.get("capability_source")
            if isinstance(source, dict) and source.get("kind") == "recorded":
                result["capability_source"] = {"kind": "recorded"}
                if isinstance(source.get("ref"), str):
                    result["capability_source"]["ref"] = source["ref"]
                capture = source.get("capture")
                if isinstance(capture, dict):
                    result["capability_source"]["capture"] = {key: capture[key]
                        for key in ("kind", "run_id", "captured_at", "revision")
                        if key in capture and type(capture[key]) in (str, int, float)}
    summary = evidence.get("solver_summary")
    if (isinstance(summary, dict) and summary.get("source") == "resource-owned endpoint calculation"
            and summary.get("failure_kind") == "bounded_search_failed"):
        result["solver_summary"] = {key: summary[key] for key in ("source", "failure_kind", "source_ref")
                                    if isinstance(summary.get(key), str)}
    if (isinstance(summary, dict) and summary.get("source") == "PreviewRecoveryMotion"
            and summary.get("failure_kind") in {"ik_search_failed", "joint_jump"}):
        public = {key: summary[key] for key in ("source", "source_ref", "failure_kind", "fraction")
                  if key in summary and type(summary[key]) in (str, int, float)}
        for key, fields in (("failed_query", ("index", "error_code", "timeout_seconds")),
                            ("last_accepted_nearest_bound", ("position",))):
            value = summary.get(key)
            if isinstance(value, dict):
                public[key] = {field: value[field] for field in fields
                               if field in value and type(value[field]) in (str, int, float)}
        result["solver_summary"] = public
    if isinstance(evidence.get("safety_checks"), list):
        result["safety_checks"] = _outline_safety_checks_for_prompt(evidence["safety_checks"])
    owner_effect = evidence.get("owner_effect_result")
    if isinstance(owner_effect, dict) and owner_effect.get("feasibility_status") == "INFEASIBLE":
        conflicts = [{key: deepcopy(row[key]) for key in ("field", "requested", "derived")}
            for row in owner_effect.get("conflicts", [])
            if isinstance(row, dict) and row.get("field") in public_fields
            and all(key in row and (row[key] is None or type(row[key]) in (str, int, float, bool))
                    for key in ("requested", "derived"))]
        if conflicts:
            result["owner_effect_result"] = {"feasibility_status": "INFEASIBLE",
                "source": "declared owner effects", "conflicts": conflicts}
    for key in ("checked_pose",):
        if isinstance(evidence.get(key), dict):
            result[key] = pose(evidence[key]) if key == "checked_pose" else reach(evidence[key])
    for key in ("source_location", "target_location", "location_token", "proposed_part_location",
                "expected_carried_part_location", "evidence_source", "source_run_id",
                "captured_at", "pose_frame", "pose_source", "role", "held_part", "current_holder_resource_jid", "destination"):
        if key in evidence and (evidence[key] is None or type(evidence[key]) in {str, int, float, bool}):
            result[key] = evidence[key]
    for key in ("reachability", "reachable_locations", "changed_fields", "condition_ids",
                "missing_task_ids", "pending_rule_ids", "projected_after_task_ids"):
        if isinstance(evidence.get(key), list) and all(isinstance(item, str) for item in evidence[key]):
            result[key] = deepcopy(evidence[key])
    field = str(evidence.get("field") or "")
    state_field = field.removeprefix("expected_start_state.").removeprefix("expected_end_state.")
    known_field = state_field in (public_fields | {"reachability", "reachable_locations"}) - {"gripper_reach"}
    if known_field:
        result["field"] = field
        for key in ("actual", "expected"):
            value = evidence.get(key)
            if key in evidence and (value is None or type(value) in {str, int, float, bool}):
                result[key] = value
        if type(evidence.get("available")) is bool:
            result["available"] = evidence["available"]
        if evidence.get("condition_operator") in {"equals", "not_equals", "exists", "in", "not_in"}:
            result["condition_operator"] = evidence["condition_operator"]
            if isinstance(evidence.get("expected"), list) and all(
                    value is None or type(value) in {str, int, float, bool} for value in evidence["expected"]):
                result["expected"] = deepcopy(evidence["expected"])
        if isinstance(evidence.get("declarations"), list):
            declarations = []
            for item in evidence["declarations"]:
                if not isinstance(item, (list, tuple)) or len(item) != 2 or not isinstance(item[0], str):
                    continue
                source, value = item
                if state_field in {"reachability", "reachable_locations"} and isinstance(value, list) and all(isinstance(token, str) for token in value):
                    declarations.append([source, deepcopy(value)])
            if declarations:
                result["declarations"] = declarations
    if isinstance(evidence.get("mismatches"), list):
        result["mismatches"] = [_outline_public_evidence(item, public_fields)
                                for item in evidence["mismatches"] if isinstance(item, dict)]
    return result


def _outline_recorded_reach_reason(reason: Any) -> bool:
    """Recognize the owner's numeric reach result without exposing other prose."""
    if not isinstance(reason, str):
        return False
    number = r"-?\d+\.\d{4}"
    return re.fullmatch(r"pose is outside gripper_reach: (?:xy_radius=" + number
        + r" m, max=" + number + r" m|z=" + number + r" m, range=\["
        + number + r", " + number + r"\] m)", reason) is not None


def _outline_feedback_for_prompt(
    row: dict[str, Any], public_fields: set[str], *, resource_fields: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    """Expose recorded outcomes without private validator or execution internals."""
    public_fields = (resource_fields or {}).get(str(row.get("resource_jid") or ""), public_fields)
    result = {key: deepcopy(value) for key, value in row.items() if key in {
        "phase", "turn_index", "decision", "constraint_code", "validation_category", "selection_status",
        "resource_jid", "part_name", "task_id", "outline_id", "candidate_id", "candidate_index",
        "candidate_source", "valid", "event_name", "selection_constraint_codes",
        "cleared_recovery_obligation_ids", "open_recovery_obligation_ids_after",
        "nondominated_candidate_ids", "selection_revision_count", "selection_revision_limit",
        "feasibility_status", "validation_status",
    }}
    if row.get("candidate_source") == "robot_task_program":
        result.pop("event_name", None)
    for key in ("expected_start_state", "expected_end_state"):
        if isinstance(row.get(key), dict):
            result[key] = {name: deepcopy(value) for name, value in row[key].items() if name in public_fields}
    for key in ("task", "proposed_next_transition", "next_transition", "selected_transition", "selection_evidence"):
        if isinstance(row.get(key), dict):
            result[key] = _outline_feedback_for_prompt(row[key], public_fields, resource_fields=resource_fields)
    for key in ("candidate_comparison", "validation_findings", "candidate_evaluations", "transition_suffix"):
        if isinstance(row.get(key), list):
            result[key] = [_outline_feedback_for_prompt(item, public_fields, resource_fields=resource_fields)
                           for item in row[key] if isinstance(item, dict)]
    if isinstance(row.get("evidence"), dict):
        evidence = row["evidence"]
        result["evidence"] = _outline_feedback_for_prompt(evidence, public_fields, resource_fields=resource_fields)
        result["evidence"].update(_outline_public_evidence(evidence, public_fields))
        counterexample = evidence.get("mating_counterexample")
        if isinstance(counterexample, dict):
            summary = {key: deepcopy(counterexample[key]) for key in ("segment", "time_from_start", "fraction")
                       if key in counterexample}
            summary["violated_conditions"] = [{key: deepcopy(condition[key]) for key in
                ("condition", "relation", "value_bounds", "limit_bounds") if key in condition}
                for condition in counterexample.get("violated_conditions", []) if isinstance(condition, dict)]
            result["evidence"]["mating_counterexample"] = summary
        endpoint = evidence.get("endpoint_counterexample")
        if isinstance(endpoint, dict):
            summary = {"changed_object_ids": deepcopy(endpoint.get("changed_object_ids", [])),
                "robot_contacts": [{key: deepcopy(contact[key]) for key in ("body_1", "body_2", "depth") if key in contact}
                    for contact in endpoint.get("robot_contacts", []) if isinstance(contact, dict)],
                "world_contacts": [], "support_pairs": []}
            for pair in endpoint.get("checked_pairs", endpoint.get("world_contacts", [])):
                if isinstance(pair, dict) and pair.get("contacts"):
                    summary["world_contacts"].append({**{key: deepcopy(pair[key]) for key in
                        ("body_1", "body_2", "allowed_support_tangency") if key in pair},
                        "contacts": [{key: deepcopy(contact[key]) for key in ("depth", "position", "normal") if key in contact}
                            for contact in pair["contacts"] if isinstance(contact, dict)]})
            for pair in endpoint.get("support_pairs", []):
                if isinstance(pair, dict) and type(pair.get("minimum_signed_distance")) in (int, float) and pair["minimum_signed_distance"] < 0:
                    summary["support_pairs"].append({key: deepcopy(pair[key]) for key in
                        ("part_object_id", "support_object_id", "minimum_signed_distance", "plane_point", "plane_normal") if key in pair})
            result["evidence"]["endpoint_counterexample"] = summary
    explanations = {
        "no_progressing_candidate": "Recorded checks did not establish recovery progress.",
        "resource_validation_unavailable": "NEEDS_CONTEXT: resource validation evidence is unavailable.",
        "unsupported_successor_condition": "The resource validator does not support the proposed end conditions together. "
            "This does not rule out the operation with different end conditions.",
        "workspace_unreachable": "The proposed pose is outside the recorded resource workspace.",
        "unknown_location_token": "The proposed location is not declared for this resource.",
        "validation_state_stale": "The recorded start state differs from the candidate start state.",
        "held_part_location_mismatch": "held_part and part_location do not describe the same ResourceAgent transition",
        "custody_discontinuity": "A resource cannot handle a part held by another resource, "
            "and a resource holds at most one part.",
        "safety_rule_violation": "The proposed transition violates a selected safety requirement; the recorded rule and AP values are included.",
        "safety_validation_unavailable": "NEEDS_CONTEXT: selected safety requirements could not be established from the recorded evidence.",
        "physical_motion_rejected": "The checked owner motion intersects recorded geometry. This result applies to that motion; alternative motions have not been exhausted.",
        "continuous_mating_rejected": "The supplied interpolated part leaves the recorded mating corridor. This result applies to that motion; alternative motions have not been exhausted.",
        "physical_endpoint_rejected": "The checked deposited-part endpoint has recorded contact or penetration that its declared support does not permit. This result applies to that endpoint; alternative endpoints have not been exhausted.",
        "physical_motion_unresolved": "The supplied owner motion did not yield a complete path. Its physical validation remains NEEDS_CONTEXT; the event was not appended and other motions have not been ruled out.",
        "physical_capability_rejected": "The calculated owner motion target is outside the recorded resource capability limits.",
        "owner_effect_rejected": "The proposed end conditions conflict with the declared effects of every supporting owner alternative. The returned values are declaration effects, not observations.",
        "transition_feasibility_rejected": "ResourceAgent did not establish the proposed transition from its declared effects and recorded facts.",
        "nominal_task_condition_not_met": "A recorded unfinished task condition is not satisfied in the checked nominal continuation.",
        "nominal_task_condition_unavailable": "NEEDS_CONTEXT: a recorded unfinished task condition has no supported value.",
        "nominal_continuation_not_established": "Recorded unfinished tasks did not establish nominal continuation from the accepted projected state.",
        "nominal_continuation_unavailable": "NEEDS_CONTEXT: evidence required to establish the recorded nominal continuation is unavailable.",
        "nominal_goal_not_met": "A recorded goal condition remains unmet after the checked nominal continuation.",
        "nominal_goal_fact_unavailable": "NEEDS_CONTEXT: a recorded goal condition has no supported value.",
        "nominal_product_completion_unavailable": "NEEDS_CONTEXT: recorded product completion evidence is unavailable.",
        "nominal_product_requirements_not_met": "Recorded product requirements remain unfinished after the checked nominal continuation.",
    }
    if row.get("constraint_code") in explanations:
        result["reason"] = explanations[row["constraint_code"]]
        missing_fields = result.get("evidence", {}).get("missing_candidate_fields")
        if row["constraint_code"] == "resource_validation_unavailable" and missing_fields:
            result["reason"] = "NEEDS_CONTEXT: required proposal fields are unbound: " + ", ".join(missing_fields) + ". No event was accepted."
        elif row["constraint_code"] == "resource_validation_unavailable" and result.get("evidence", {}).get("parameter_conflicts"):
            result["reason"] = "NEEDS_CONTEXT: recorded parameter values conflict with the declared bindings shown below."
        elif row["constraint_code"] == "resource_validation_unavailable" and result.get("evidence", {}).get("solver_summary", {}).get("failure_kind") == "bounded_search_failed":
            result["reason"] = "NEEDS_CONTEXT: the bounded owner calculation did not establish endpoint reachability. This is not a demonstrated constraint violation."
        if row["constraint_code"] == "physical_capability_rejected":
            capability = result.get("evidence", {}).get("capability_result", {})
            if capability.get('guard_failures'):
                result['reason'] = 'Recorded public conditions block the supported owner declarations; safety specifications were not checked.'
            if _outline_recorded_reach_reason(row.get("reason")) and capability.get("checks"):
                result["reason"] = "The calculated target is outside the recorded owner motion capability."
    elif row.get("constraint_code"):
        result["reason"] = "Recorded validation or comparison result; see the constraint code and public evidence."
    return result


def _render_validated_event_prompt(payload: dict[str, Any], session: dict[str, Any]) -> str:
    """Render public facts for one proposed event without private validator models."""
    from .multi_turn_context import public_part_snapshot, public_resource_snapshot

    llm_input = payload.get("llm_input") or {}
    prepared = {"recovery_resources": payload.get("recovery_resources") or {}, "llm_input": llm_input}
    scope = payload.get("outline_context") or {}
    selected_resources = set(scope.get("resource_jids", prepared["recovery_resources"]))
    resources = [public_resource_snapshot(prepared, session, jid) for jid in sorted(selected_resources)]
    all_parts = [public_part_snapshot(prepared, session, row["part_name"])
                 for row in _projected_outline_parts(llm_input=llm_input, session_state=session) if row.get("part_name")]
    selected_parts = set(scope.get("part_names", [row["part_name"] for row in all_parts]))
    parts = [row for row in all_parts if row["part_name"] in selected_parts]
    declarations = {}
    prefix_fields = {jid: {name for name, row in model.get("state_variables", {}).items()
                          if not row.get("private") and not name.startswith(("task_ctx.", "_"))}
                     for jid, model in session.get("recovery_des_models", {}).items()}
    for jid, fields in (payload.get("outline_resource_fields") or {}).items():
        if jid not in selected_resources or f"/resources/{jid}/state_variables" in payload.get("outline_served_context", {}):
            continue
        values = fields.get("value", fields)
        declarations[jid] = {name: {
            "type": _outline_state_field_schema(name, row)["type"],
            **{key: deepcopy(row[key]) for key in ("scope", "description", "domain", "domain_complete") if key in row},
        } for name, row in values.items()}
    fault = _slim_fault_event(llm_input.get("fault_event") or {})
    processing = fault.get("checkpoint", {}).pop("processing_observations", {})
    processing_rows = []
    feedback = deepcopy(session.get("candidate_rejection_feedback") or [])
    for finding in session.get("outline_validation_findings") or []:
        if finding not in feedback:
            feedback.append(deepcopy(finding))
    rejection_history = [_outline_feedback_for_prompt(row,
        {"resource_state", "part_state", "held_part", "resource_location", "part_location"},
        resource_fields=prefix_fields) for row in _outline_rejection_history(session)
        if row.get("decision") in {"need_revision", "need_context"}
        and not all(finding in feedback for finding in row["validation_findings"])][-2:]
    for rid, record in processing.items():
        if any(ref.endswith("/static_capabilities") and isinstance(fact.get("value"), dict)
               and fact["value"].get("processing_observations") == record
               for ref, fact in payload.get("outline_served_context", {}).items()):
            continue
        for job in record.get("jobs", {}).values():
            processing_rows.append({"resource_id": rid, **{key: deepcopy(job[key]) for key in (
                "job_id", "task_id", "part_name", "process", "result", "job_state",
                "simulation_clock_sec", "processing_active_sec", "processing_time_sec",
                "processing_fraction", "process_completed", "simulation_completed_sec",
            ) if key in job}})
    candidates = session.get("outline_mode") == "incremental_candidates_validated"
    proposal_field = "candidate_events" if candidates else "next_transition"
    sections = [
        "Task and Role",
        (("Propose every materially distinct supported next recovery event you can ground, from one up to 5; do not pad the list. "
          "Return thought and candidate_events. Do not select or rank the candidates; validated projected DES product states are compared after generation. ")
         if candidates else "Propose one next recovery event from the supplied facts. Return thought and next_transition. ")
        + "Each event describes a task-level outcome for one resource. ResourceAgent checks backward and forward feasibility; "
        "Central Controller Agent checks the selected safety specifications. The runtime checks nominal continuation.",
        *(["Unfinished recorded tasks remain available for nominal continuation. Final goals must hold after "
           "the recovery and a validated remaining nominal suffix; recovery need not itself complete every goal. "
           "A recorded task's resource assignment is not itself a goal; accepted matching product-completion evidence "
           "may satisfy its product work through another resource. Explicit resource goals remain required."]
          if session.get("nominal_continuation_semantics") == "recorded_unfinished_tasks" else []),
        "Keep thought to at most two short sentences and rationale to one sentence.",
        "Choose a condition change that enables unfinished work. Use rejection feedback to revise unsupported conditions; "
        "avoid repeating accepted state changes without advancing recovery. Respect recorded resource availability.",
        "Observation source", _inline_json(payload.get("outline_observation_source") or {}),
        "Fault Event", _inline_json(fault),
        "Interrupted task", _inline_json(payload.get("outline_interrupted_task") or {}),
        *(["Processing observations", _fact_rows(processing_rows),
           "These are capture-time job records; use Current resource states after accepted events.",
           "processing_fraction is active simulated processing time / configured duration, not measured material removal."]
          if processing_rows else []),
        "Current resource states", _fact_rows(resources),
        "Current part states", _fact_rows(parts),
        "Recovery goals", _compact_recovery_goals(llm_input, projected_parts=all_parts, current_recovery_blockers=[], exact_conditions=True),
        "Resource Capabilities",
        "Use recorded facts and request missing public context. Declarations describe recorded capabilities; they are not a complete list of possible recovery events. "
        "RA checks backward and forward feasibility. Physical calculations support those checks and the selected safety predicates. "
        "Incomplete required evidence remains NEEDS_CONTEXT.",
        _resource_index_text(scope.get("resource_index") or []),
        "Public state declarations", _inline_json(declarations),
        "Safety Rules", (_outline_selected_safety_text(payload["outline_safety_context"])
                         if "outline_safety_context" in payload else _compact_safety_rules(llm_input)),
        *(["Current geometric occupancy", _inline_json({
           **{key: value for key, value in payload["outline_safety_context"].items()
              if key in {"current_occupancy", "missing_context"}},
           "source": {key: value for key, value in payload["outline_safety_context"].get("source", {}).items()
                      if key in {"source", "basis", "captured_at", "accepted_trace_length", "coverage"}}}),
           "Occupancy is calculated from the saved or accepted projected endpoint. It does not establish a future movement's safety. "
           "[false, true] is unresolved; null means evidence is unavailable. AP labels are scoped to each specification. "
           "Pairwise resource bindings cover each unordered pair in resource_bindings; concrete bindings are retrievable."]
          if "outline_safety_context" in payload else []),
        *(["Safety rule bindings in validation feedback", _inline_json(payload["outline_safety_feedback_bindings"])]
          if payload.get("outline_safety_feedback_bindings") else []),
        "Accepted Transition Prefix (projected; not executed)",
        _compact_json(_accepted_transition_prefix_for_prompt(session.get("accepted_outline_prefix") or [],
            hide_registered_functions=True, public_fields=prefix_fields)),
        "Validation feedback", _compact_json(feedback),
        *(["Recent rejected proposals (checked in their recorded starting states)", _compact_json(rejection_history)]
          if rejection_history else []),
        "Output Constraints",
        "Use exact supplied identifiers and declared public state fields. "
        f"Each event in {proposal_field} contains outline_id, event_name, resource_jid, expected_end_state, rationale, "
        "and part_name when a part is involved. Its starting state is the current projected state. "
        "expected_end_state may contain only the intended conditions; unchanged fields need not be copied. "
        + ("Do not include selection indices or a transition_suffix. " if candidates else
           "Do not include candidate lists, selection indices, or a transition_suffix. ") +
        "Measured poses, recorded symbolic locations and projected states have distinct sources; "
        "do not treat a resource station as a measured part pose.",
    ]
    return "\n\n".join(sections) + "\n"


def _render_outline_prompt(payload: dict[str, Any]) -> str:
    llm_input = dict(payload.get("llm_input") or {})
    session_state = dict(payload.get("session_state") or {})
    recovery_resources = dict(payload.get("recovery_resources") or {})
    recovery_des_models = dict(session_state.get("recovery_des_models") or {})
    scope = payload.get("outline_context")
    if scope or session_state.get("validation_contract_version") == 2:
        public_fields = {"resource_state", "part_state", "held_part", "resource_location", "part_location"}
        resource_fields = {}
        for jid, model in recovery_des_models.items():
            fields = model.get("state_variables", {})
            private_fields = {name for name, declaration in fields.items()
                              if declaration.get("private") or name.startswith(("task_ctx.", "_"))}
            resource_fields[jid] = (public_fields | set(fields)) - private_fields
        for key in ("outline_validation_findings", "candidate_rejection_feedback", "primitive_escalation_diagnostics", "pruned_actions", "turns"):
            session_state[key] = [_outline_feedback_for_prompt(row, public_fields, resource_fields=resource_fields)
                                  for row in session_state.get(key, []) if isinstance(row, dict)]
    if session_state.get("validation_contract_version") == 2 and not session_state.get("outline_diagnostic_bypass"):
        return _render_validated_event_prompt(payload, session_state)
    custody_state_declared = _public_custody_state_is_declared(
        recovery_resources=recovery_resources,
        recovery_des_models=recovery_des_models,
    )
    feedback_render_style = _feedback_render_style_token(session_state)
    outline_mode = str(session_state.get("outline_mode") or "incremental").strip().lower()
    accepted_prefix = list(session_state.get("accepted_outline_prefix") or [])
    previous_lookahead = list(session_state.get("outline_lookahead") or [])
    pruned_actions = list(session_state.get("pruned_actions") or [])
    current_recovery_blockers = [
        deepcopy(row)
        for row in (payload.get("current_recovery_blockers") or [])
        if isinstance(row, dict)
    ]
    outline_validation_findings = list(session_state.get("outline_validation_findings") or [])
    rejection_history = _outline_rejection_history(session_state)
    projected_resources = _projected_outline_resources(
        llm_input=llm_input,
        session_state=session_state,
    )
    projected_parts = _projected_outline_parts(
        llm_input=llm_input,
        session_state=session_state,
    )
    prompt_projected_resources = _outline_prompt_resource_facts(projected_resources)
    prompt_projected_parts = _outline_prompt_part_facts(projected_parts)
    resources_by_jid = _finding_resources_by_jid(prompt_projected_resources)
    parts_by_name = _finding_parts_by_name(prompt_projected_parts)
    candidate_rejection_feedback = [
        deepcopy(row)
        for row in (session_state.get("candidate_rejection_feedback") or [])
        if isinstance(row, dict)
    ]
    primitive_escalation_diagnostics = [
        deepcopy(row)
        for row in (session_state.get("primitive_escalation_diagnostics") or [])
        if isinstance(row, dict)
    ]
    is_single_pass = outline_mode == "single_pass"
    is_candidate_mode = outline_mode == "incremental_candidates_validated"
    recovery_selection_mode = str(
        session_state.get("recovery_selection_mode") or "pure_llm"
    ).strip().lower()
    if recovery_selection_mode not in {"pure_llm", "neurosymbolic"}:
        recovery_selection_mode = "pure_llm"
    candidate_rejection_history = (
        [] if is_candidate_mode else _candidate_rejection_history(session_state)
    )
    action_horizon = str(session_state.get("action_horizon") or "1").strip().lower()
    if action_horizon not in {"1", "k", "full"}:
        action_horizon = "1"
    try:
        action_horizon_k = max(1, int(session_state.get("action_horizon_k") or 3))
    except (TypeError, ValueError):
        action_horizon_k = 3
    action_horizon_steps = session_state.get("action_horizon_steps")
    if action_horizon_steps in (None, ""):
        if action_horizon == "full":
            action_horizon_steps = "full"
        elif action_horizon == "k":
            action_horizon_steps = action_horizon_k
        else:
            action_horizon_steps = 1
    recent_candidate_diagnostic_signatures = (
        _candidate_rejection_diagnostic_signatures(
            history=candidate_rejection_history,
            feedback_rows=candidate_rejection_feedback,
            resources_by_jid=resources_by_jid,
            parts_by_name=parts_by_name,
        )
        if is_candidate_mode and feedback_render_style == "des_event_diagnostic"
        else set()
    )
    history_pruned_actions = _history_derived_pruned_actions_summary(
        pruned_actions=pruned_actions,
        rejection_history=[] if is_candidate_mode else rejection_history,
        candidate_rejection_history=[] if is_candidate_mode else candidate_rejection_history,
        candidate_rejection_feedback=[] if is_candidate_mode else candidate_rejection_feedback,
        projected_resources=prompt_projected_resources,
        projected_parts=prompt_projected_parts,
        feedback_render_style=feedback_render_style,
        excluded_diagnostic_signatures=recent_candidate_diagnostic_signatures,
    )
    immediate_validation_feedback = _outline_immediate_validation_feedback_summary(
        outline_validation_findings=outline_validation_findings,
        candidate_rejection_feedback=candidate_rejection_feedback,
        primitive_escalation_diagnostics=primitive_escalation_diagnostics,
        feedback_render_style=feedback_render_style,
        resources_by_jid=resources_by_jid,
        parts_by_name=parts_by_name,
    )
    if is_single_pass:
        role_text = (
            "You are the active replanner for a recovery session.\n"
            "Current phase: Recovery Event Synthesis (single pass).\n"
            "Propose ALL recovery events as symbolic outline rows, "
            "serialized in JSON field `transition_trace`.\n"
            "Each row must be one enabled recovery transition for one resource. "
            "Author the expected symbolic start and end states directly."
        )
    elif is_candidate_mode:
        if recovery_selection_mode == "neurosymbolic":
            candidate_budget = _DEFAULT_CANDIDATE_BOUND
            selection_owner_text = (
                "Do not select or rank the candidates; validated projected DES product "
                "states are compared after generation."
            )
            horizon_text = (
                "Propose every materially distinct supported next recovery event you can "
                f"ground, from one up to {candidate_budget}; do not pad the list."
            )
        else:
            selection_owner_text = (
                "You must choose the best candidate by setting `selected_candidate_index`."
            )
            horizon_text = {
                "1": (
                    "Propose exactly three candidates; each contains exactly one next "
                    "recovery event."
                ),
                "k": (
                    f"Each candidate contains a sequence of up to {action_horizon_steps} next recovery events."
                ),
                "full": "Each candidate contains a complete recovery outline trace.",
            }[action_horizon]
        role_text = (
            "You are the active replanner for a recovery session.\n"
            "Current phase: Recovery Event Candidate Proposal.\n"
            "Propose grounded symbolic recovery transitions enabled by the current symbolic state.\n"
            "Accepted events extend the recovery trace toward safe nominal resumption.\n"
            f"{horizon_text}\n"
            f"{selection_owner_text}\n"
            "Return authored symbolic rows only; Product validates the stated transition."
        )
        if scope:
            from .multi_turn_diagnostic import llm_selects_next_event

            chooses = llm_selects_next_event(session_state)
            if recovery_selection_mode == "neurosymbolic":
                selection_owner_text = ("No validator checks or compares events; you choose the next one."
                                        if chooses else "Do not select or rank candidates; runtime compares validated projected states.")
            role_text = (
                "You are the active replanner for a recovery session.\n"
                "Current phase: Recovery Event Candidate Proposal.\n"
                + ("Choose the next event" if chooses else "Propose supported next events")
                + " from the supplied facts and declared capabilities.\n"
                f"{selection_owner_text}"
            )
            if recovery_selection_mode != "neurosymbolic":
                role_text += "\n" + horizon_text
        if candidate_rejection_feedback:
            role_text += (
                "\nUse listed diagnostics as projected-state evidence and avoid "
                "repeating candidate events that remain disabled or blocked_by_supervisor "
                "under unchanged facts."
            )
            if any(
                str(row.get("constraint_code") or "").strip()
                == "selection_ambiguous"
                for row in candidate_rejection_feedback
                if isinstance(row, dict)
            ):
                role_text += (
                    "\nReturn at most one representative for each exact equivalent "
                    "successor class listed in the comparison evidence."
                )
            role_text += _no_progress_revision_instruction(
                immediate_validation_feedback
            )
    else:
        role_text = (
            "You are the active replanner for a DES fallback recovery session.\n"
            "Current phase: Recovery Event Synthesis (one transition at a time).\n"
            "Propose exactly ONE next symbolic recovery row to append after the accepted "
            "transition prefix, serialized in JSON field `next_transition`.\n"
            "You may include an optional transition suffix, serialized in JSON "
            "field `transition_suffix`, to show your intended remaining trace; "
            "only `next_transition` will be validated and accepted.\n"
            "When `next_transition` is your LAST recovery event, leave "
            "`transition_suffix` empty to signal that the recovery event sequence "
            "is complete."
        )

    sections: list[str] = [
        "Task and Role",
        role_text,
        "Limit `thought` to two short sentences; each `rationale` to one." if scope
        else "Keep `thought` to at most two short sentences and each `rationale` to one short sentence.",
    ]
    if payload.get("outline_context"):
        sections += ["", "Fault Event", _inline_json(_slim_fault_event(llm_input.get("fault_event") or {}))]

    if not is_single_pass and accepted_prefix:
        if is_candidate_mode:
            sections.extend(
                [
                    "",
                    "Accepted Transition Prefix (projected; not executed)" if scope
                    else "Accepted Transition Prefix (already applied; do not repeat)",
                    _compact_json(
                        _accepted_transition_prefix_for_prompt(
                            accepted_prefix,
                            hide_registered_functions=bool(scope),
                        )
                    ),
                    "- These transitions were applied only to the projected state. Do not propose them again." if scope
                    else "- These transitions have already been applied. Do not propose them again.",
                    "- Use Current DES State below as the exact start point for the next candidate.",
                ]
            )
        else:
            sections.extend(
                [
                    "",
                    "Accepted Transition Prefix (projected; not executed)" if scope
                    else "Accepted Transition Prefix (already applied; do not repeat)",
                    _compact_json(
                        _accepted_transition_prefix_for_prompt(accepted_prefix, hide_registered_functions=bool(scope))
                    ),
                    "- These transitions were applied only to the projected state. Do not propose them again." if scope
                    else "- These transitions have already been applied. Do not propose them again.",
                    "- Use Current DES State below as the exact start point for the next transition.",
                ]
            )

    sections.extend(
        _top_validation_feedback_section(
            summary=immediate_validation_feedback,
            label="Validation Feedback",
        )
    )

    if primitive_escalation_diagnostics:
        sections.extend(
            [
                "",
                "Primitive Escalation Diagnostics",
                _compact_json(primitive_escalation_diagnostics),
            ]
        )

    if is_candidate_mode:
        if history_pruned_actions != "(none)":
            sections.extend(
                [
                    "",
                    "Persistently Disabled Candidate Events",
                    history_pruned_actions,
                ]
            )
        sections.extend(
            [
                "",
                "Resource Capabilities",
                _resource_index_text(scope["resource_index"]) if scope
                else _resource_capabilities_summary(recovery_resources),
                "",
                "Resource States",
                *([] if scope else ["These are modeled `resource_state` labels grouped by `resource_type`. Preserve exact observed start conditions; an intended resource condition must have a predefined successor with RA-validated primitive support."]),
                _resource_state_vocabulary_summary(
                    recovery_resources=recovery_resources,
                    recovery_des_models=recovery_des_models,
                ),
                "",
                "Part States",
                *([] if scope else ["These are modeled or goal-declared product/part `part_state` labels. Preserve exact observations; proposed effects still require RA feasibility and later primitive-composition validation."]),
                _part_state_vocabulary_summary(
                    llm_input=llm_input,
                    recovery_resources=recovery_resources,
                    recovery_des_models=recovery_des_models,
                ),
                "",
                "Resource Fields",
                *([] if scope else ["These public resource-scoped fields use values declared by the responsible Resource Agent."]),
                _outline_resource_fields_text(payload)
                if payload.get("outline_context") else _resource_exact_fields_summary(
                    recovery_resources=recovery_resources,
                    recovery_des_models=recovery_des_models,
                ),
            ]
        )
    elif outline_validation_findings:
        sections.extend(
            [
                "",
                "Active Transition Diagnostics (still unresolved)",
                _outline_validation_summary(
                    outline_validation_findings,
                    feedback_render_style=feedback_render_style,
                    resources_by_jid=resources_by_jid,
                    parts_by_name=parts_by_name,
                ),
            ]
        )

    if not is_candidate_mode and history_pruned_actions != "(none)":
        sections.extend(
            [
                "",
                "Persistently Disabled Candidate Events",
                history_pruned_actions,
            ]
        )

    if rejection_history and not is_candidate_mode:
        sections.extend(
            [
                "",
                "Rejected Transition Attempts And Validation Feedback",
                _outline_rejection_history_summary(rejection_history),
            ]
        )

    sections.extend(
        [
            "",
            "Current DES State",
            (
                "States below are projected after the accepted transitions, not newly observed. "
                "A saved pose is not a new observation after these transitions."
                if accepted_prefix else "States below are recorded before recovery; expected_end_state describes a proposed effect."
            ) if scope else "`Current DES State` is the authoritative propagated state after applying the accepted transition prefix. Code binds each candidate to this state.",
            *(["resource_location and part_location are symbolic; use observed_pose for measured part positions.",
               "Capture (null: not recorded): " + _inline_json(payload.get("outline_observation_source", {}))]
              if scope else []),
            "Resources",
            _fact_rows(prompt_projected_resources) if scope else _compact_json(prompt_projected_resources),
            "",
            "Parts",
            *(["last_observed_pose is historical; projected_pose is unexecuted. Current observed_pose is unavailable after a projected move."]
              if scope and any(part.get("observed_pose_current") is False for part in prompt_projected_parts) else []),
            _fact_rows(prompt_projected_parts) if scope else _compact_json(prompt_projected_parts),
            "Recovery Goals",
            _compact_recovery_goals(
                llm_input,
                projected_parts=projected_parts,
                current_recovery_blockers=current_recovery_blockers,
            ),
        ]
    )

    sections.extend(
        [
            "",
            "Safety Rules",
            _compact_safety_rules(llm_input),
        ]
    )

    if is_candidate_mode:
        candidate_property_name = (
            "`candidate_events`" if action_horizon == "1" else "`candidate_traces`"
        )
        if action_horizon == "1":
            candidate_shape_rule = (
                "- Each candidate is one transition for one listed resource and includes "
                "`outline_id`, `event_name`, `resource_jid`, `expected_end_state`, and "
                "`rationale`."
            )
            if scope:
                candidate_shape_rule = "- Each candidate: one resource transition with `outline_id`, `event_name`, `resource_jid`, `expected_end_state`, `rationale`."
        else:
            candidate_shape_rule = (
                "- Each event in a candidate trace is one transition for one listed "
                "resource and includes `outline_id`, `event_name`, `resource_jid`, "
                "`expected_end_state`, and `rationale`."
            )
        response_action = "- When required facts are available, return" if scope else "- Return"
        from .multi_turn_diagnostic import llm_selects_next_event

        if recovery_selection_mode == "neurosymbolic" and llm_selects_next_event(session_state):
            response_rules = [
                f"{response_action} one JSON object with `thought` and exactly one `candidate_events` "
                "item, the next event you choose; do not return `selected_candidate_index`."
            ]
        elif recovery_selection_mode == "neurosymbolic":
            response_rules = [
                f"{response_action} one JSON object with `thought` and one to 5 materially "
                "distinct `candidate_events`; do not return `selected_candidate_index` "
                "or padding."
            ]
        else:
            response_rules = [
                f"{response_action} one JSON object with `thought`, {candidate_property_name}, and "
                "an integer `selected_candidate_index` pointing at your chosen candidate.",
                *(
                    ["- `candidate_events` must contain exactly three candidates."]
                    if action_horizon == "1"
                    else []
                ),
            ]
        candidate_part_rules = [
            "- Without `part_name`, omit every part-scoped state field."
        ]
        if custody_state_declared:
            candidate_part_rules.append(
                "- Use `held_part` only in a candidate branch whose Resource Agent "
                "schema includes it. Without `part_name`, omit `held_part`. Any custody "
                "change requires `part_name`, `held_part`, `part_state`, and "
                "`part_location`. While held, `part_location` equals `resource_jid`; "
                "release requires an exact non-null `part_location`."
            )
            if scope:
                candidate_part_rules[-1] = (
                    "- Use `held_part` only where declared, with `part_name`. Custody changes require "
                    "`part_name`, `held_part`, `part_state`, `part_location`. While held, `part_location`="
                    "`resource_jid`; release needs an exact non-null `part_location`."
                )
        sections.extend(
            [
                "",
                "Recovery Candidate Rules",
                *response_rules,
                candidate_shape_rule,
                *candidate_part_rules,
                *([] if scope else ["- A non-null `observed_pose` is grounded source evidence for its part."]),
                (
                    "- Author `event_name` and effects from recorded facts and public capabilities. "
                    "Resource validators check whether the proposed condition is supported; primitive composition checks feasibility."
                    if scope else
                    "- `event_name` may be new, and the proposed transition need not be predefined. "
                    "Its intended resource condition must match a predefined successor. "
                    "The RA derives alternative primitive supports and requires feasible parameter "
                    "assignments for every primitive in the union of all matching successor transitions. Missing evidence requires revision."
                ),
                (
                    "- Use only supplied `resource_jid` and `part_name` values. Location "
                    "tokens must belong to the same RA state field, except that the exact "
                    "current `part_location` may be preserved."
                ),
            ]
        )

    if not is_single_pass and not is_candidate_mode and previous_lookahead:
        sections.extend(
            [
                "",
                "Your Previous Lookahead (non-binding, for context)",
                _compact_json(previous_lookahead),
            ]
        )

    constraints: list[str] = []
    if not is_candidate_mode:
        part_constraints = [
            "- If part_name is present without a custody change, use only state fields "
            "declared by that Resource Agent.",
            "- If part_name is absent, do not emit part-scoped state keys.",
        ]
        if custody_state_declared:
            part_constraints.insert(
                1,
                "- Use held_part only for a Resource Agent that publicly declares it; "
                "a custody change requires part_name, held_part, part_state, and "
                "part_location, with an exact non-null release location.",
            )
        constraints.extend(
            [
                "",
                "Output Constraints",
                "- Use only the listed resources.",
                "- Each transition is one symbolic recovery row by one resource.",
                "- Each transition must include outline_id, event_name, resource_jid, expected_end_state, and rationale.",
                "- Do not emit ppr_ontology, source_ref, target_ref, event_schema_id, resource_binding, object_bindings, parameters, surface_event_name, surface_description, or predecessors in outline mode.",
                "- Author expected_end_state in outline mode; code binds the transition to Current DES State.",
                *part_constraints,
                "- Bind only entities and location tokens grounded in the current plant state.",
                "- Rationale should explain enabledness or which recovery goal is advanced.",
            ]
        )

    if is_single_pass:
        constraints.append(
            "- Propose all recovery events as an ordered transition trace serialized in JSON field `transition_trace`."
        )
    elif not is_candidate_mode:
        constraints.extend(
            [
                "- Propose exactly one next transition serialized in JSON field `next_transition`.",
                "- Optional remaining transitions are serialized in JSON field `transition_suffix`.",
            ]
        )

    sections.extend(constraints)
    return "\n".join(sections).strip() + "\n"


# ---------------------------------------------------------------------------
# Prompt renderer
# ---------------------------------------------------------------------------


def render_multi_turn_phase_prompt(prompt_input: dict[str, Any]) -> str:
    """Render a human-readable prompt string for the current phase."""
    payload = deepcopy(prompt_input or {})
    phase = str(payload.get("phase") or "").strip().lower()

    if phase == "grounding":
        return _render_grounding_prompt(payload) + _render_retrieved_context(payload, outline=False)
    if phase == "outline":
        return _render_outline_prompt(payload) + _render_retrieved_context(payload, outline=True)
    if phase == "primitive_generation":
        return _render_primitive_generation_prompt(payload)

    # Other phases: placeholder until implemented
    phase_title = _PHASE_TITLES.get(phase, phase)
    return f"Phase: {phase_title}\n\n[Prompt content for {phase} — not yet implemented]\n"


def _render_retrieved_context(payload: dict[str, Any], *, outline: bool) -> str:
    scope = payload.get("outline_context")
    if not scope:
        return ""
    alternatives = [row for row in scope["resource_index"] if row["resource_jid"] not in scope["resource_jids"]]
    sections = []
    session = payload.get("session_state", {})
    if alternatives and not (outline and (session.get("validation_contract_version") == 2
            or session.get("outline_mode") == "incremental_candidates_validated")):
        sections = ["", "Available Resource Capabilities", _resource_index_text(alternatives)]
    fault = payload.get("llm_input", {}).get("fault_event", {})
    if outline and session.get("validation_contract_version") != 2 and payload.get("outline_interrupted_task") and not all(
        fault.get(key) for key in ("focused_resource_jid", "blocked_at_task_id", "blocked_at_function")
    ):
        sections += ["", "Interrupted task", _inline_json(payload["outline_interrupted_task"])]
    if outline:
        sections += ["", "Context retrieval",
            "Proposing events: omit `decision`/`context_requests` or use null/[]; never combine with need_context.",
            'Missing facts: return only `thought`, `decision`="need_context", `context_requests` (1 to 8 refs). Omit all candidate fields, even null/empty.',
            "Refs: /resources; /parts; /resources/<jid>/{snapshot,current_pose,held_part,static_capabilities,capability_evidence,state_variables,pending_tasks}; /parts/<name>/{snapshot,observed_pose,target}. Expand braces to one field.",
            *(["/safety_rules; /safety_rules/{specification}; /safety_rules/{specification}/instances; /regions; /regions/{region}/occupancy."]
              if "outline_safety_context" in payload else []),
            "static_capabilities supplies public declarations and job records; state_variables supplies public declarations. "
            "Missing values are unknown. Retrieve necessary facts; do not invent facts or effects. Replies distinguish observations, declarations and projected states."]
    if payload.get("outline_served_context"):
        sections += ["", "Retrieved context", _retrieved_outline_context_text(payload)]
    if payload.get("outline_context_errors"):
        errors = []
        for error in payload["outline_context_errors"]:
            ref = str(error.get("ref") or "")
            segments = ref.split("/")
            public = (ref in {"/resources", "/parts"} or len(segments) == 4 and (
                segments[1] == "resources" and segments[3] in {"snapshot", "current_pose", "held_part", "static_capabilities", "capability_evidence", "state_variables", "pending_tasks"}
                or segments[1] == "parts" and segments[3] in {"snapshot", "observed_pose", "target"}))
            if "outline_safety_context" in payload:
                public = public or ref in {"/safety_rules", "/regions"} or (
                    len(segments) == 3 and segments[1] == "safety_rules") or (
                    len(segments) == 4 and (segments[1] == "safety_rules" and segments[3] == "instances"
                                          or segments[1] == "regions" and segments[3] == "occupancy"))
            errors.append({"ref": ref, "error": "NEEDS_CONTEXT: requested public fact is unavailable"} if public
                          else {"error": "Unsupported context reference; use the listed public resource and part references."})
        sections += ["", "Missing context", _compact_json(errors)]
    return "\n".join(sections) + "\n"


def _retrieved_outline_context_text(payload: dict[str, Any]) -> str:
    """Render all retrieved facts once while retaining their source metadata."""
    facts = payload["outline_served_context"]
    metadata_keys = ("captured_at", "source_run_id", "revision", "accepted_trace_length")
    first = next(iter(facts.values()))
    shared = {key: first[key] for key in metadata_keys
              if key in first and all(key in fact and fact[key] == first[key] for fact in facts.values())}
    observation = payload.get('outline_observation_source') or {}
    displayed_shared = {key: value for key, value in shared.items() if key not in observation or observation[key] != value}
    lines = ["Shared source metadata: " + _inline_json(displayed_shared)] if displayed_shared else []
    resources = _outline_prompt_resource_facts(_projected_outline_resources(
        llm_input=payload.get("llm_input", {}), session_state=payload.get("session_state", {})))
    parts = _outline_prompt_part_facts(_projected_outline_parts(
        llm_input=payload.get("llm_input", {}), session_state=payload.get("session_state", {})))
    shown = {"resources": {row["resource_jid"]: row for row in resources},
             "parts": {row["part_name"]: row for row in parts}}
    for ref, fact in facts.items():
        row = {key: deepcopy(value) for key, value in fact.items() if key not in shared}
        already_shown = ""
        if (ref == "/resources" and payload.get("phase") == "outline"
                and {item['resource_jid'] for item in row.get('value', [])}
                == {item['resource_jid'] for item in payload.get("outline_context", {}).get("resource_index", [])}):
            row.pop("value", None)
            already_shown = "Value already shown in Resource Capabilities."
        segments = ref.split("/")
        if len(segments) == 4 and segments[1] == "resources" and segments[3] == "static_capabilities":
            index = next((item for item in payload.get("outline_context", {}).get("resource_index", [])
                          if item["resource_jid"] == segments[2]), {})
            value = row.get("value")
            if isinstance(value, dict):
                row["value"] = {key: item for key, item in value.items() if key not in index or index[key] != item}
                if 'field_sources' in row['value']:
                    row['value']['field_sources'] = {key: item for key, item in row['value']['field_sources'].items()
                        if item != index.get('field_sources', {}).get(key)}
                if row["value"] != value:
                    already_shown = "Repeated fields are in Resource Capabilities."
        current = shown.get(segments[1], {}).get(segments[2], {}) if len(segments) == 4 else {}
        if (len(segments) == 4 and segments[1] == "resources" and segments[3] == "state_variables"
                and payload.get("phase") == "outline" and payload.get("session_state", {}).get("validation_contract_version") != 2):
            declarations = payload.get("outline_resource_fields", {}).get(segments[2], {})
            if declarations.get("value") == row.get("value"):
                row.pop("value", None)
                if declarations.get("interpretation") == row.get("interpretation"):
                    row.pop("interpretation", None)
                already_shown = "Declarations and runtime interpretation already shown in Resource Fields."
        if current and payload.get("phase") == "outline":
            field = segments[3]
            value = row.get("value")
            if field == "snapshot" and isinstance(value, dict):
                repeated = [key for key, item in value.items() if key in current and current[key] == item]
                if repeated:
                    row["value"] = {key: item for key, item in value.items() if key not in repeated}
                    already_shown = "Fields already shown in Current DES State: " + ", ".join(repeated)
            elif field in current and current[field] == value:
                row.pop("value", None)
                already_shown = "Value already shown in Current DES State: " + field
        lines.append(ref + ": " + _inline_json(row))
        if already_shown:
            lines.append(already_shown)
    return "\n".join(lines)


__all__ = [
    "build_multi_turn_phase_prompt_input",
    "multi_turn_phase_response_schema",
    "render_multi_turn_phase_prompt",
]


def _outline_selected_safety_text(context: dict) -> str:
    """Render selected declarations once without compiled monitor internals."""
    lines = []
    for row in context["specifications"]:
        lines.append(f"- {row['id']}: {row['requirement']}")
        lines.append("  Formula: " + row["formula"])
        scopes = [scope for scope in row["requirement_scopes"] if scope.get("specification", row["id"]) == row["id"]]
        lines.append("  Scope bindings: " + _inline_json(scopes))
    return "\n".join(lines) if lines else "(none)"
