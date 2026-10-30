"""Generic resource primitive catalog and recovery snapshot helpers."""

from __future__ import annotations

import inspect
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from typing import Any

from cais_spade_llm.agents.shared_information.recovery_validation_protocol import (
    recovery_validation_fingerprint,
)
from cais_spade_llm.function_analyzer import FunctionAnalyzer
from cais_spade_llm.resources.resource_profile import (
    get_resource_profile_for_agent,
    resource_event_fact_contract,
    resource_snapshot_availability,
    resource_snapshot_field_value,
)


INTERNAL_GRIPPER_PRIMITIVES = frozenset({"open_gripper", "close_gripper", "attach_part", "detach_part"})


def filter_recovery_primitive_catalog(
    primitive_catalog: list[dict[str, Any]], *, resource_type: str = "",
) -> list[dict[str, Any]]:
    """Keep gripper operations inside grasp_part and release_part for recovery.

    Args:
        primitive_catalog: Live declarations or an immutable saved catalog.
        resource_type: Owner type when saved rows do not repeat it.

    Returns:
        Independent catalog rows excluding internal robot and KMR gripper controls.
    """
    return [deepcopy(row) for row in primitive_catalog
            if not ((resource_type or row.get("resource_type")) in {"robot", "kmr"}
                    and row.get("name") in INTERNAL_GRIPPER_PRIMITIVES)]


def primitive_summary(
    *,
    description: str,
    preconditions: dict[str, Any],
    effects: dict[str, Any],
) -> str:
    base = description or "Recovery primitive"
    pre_keys = ", ".join(sorted(str(key) for key in preconditions)) if preconditions else ""
    effect_keys = ", ".join(sorted(str(key) for key in effects)) if effects else ""
    detail_parts: list[str] = []
    if pre_keys:
        detail_parts.append(f"pre: {pre_keys}")
    if effect_keys:
        detail_parts.append(f"effects: {effect_keys}")
    if not detail_parts:
        return base
    return f"{base} ({'; '.join(detail_parts)})"


def _build_catalog_owner(resource_agent: Any) -> tuple[Any, Any]:
    profile = get_resource_profile_for_agent(resource_agent)
    owner = resource_agent
    if profile.primitive_owner_resolver is not None:
        try:
            owner = profile.primitive_owner_resolver(resource_agent) or resource_agent
        except Exception:
            owner = resource_agent
    return owner, profile


def _raw_recovery_primitive_names(resource_agent: Any) -> list[str]:
    raw = getattr(resource_agent, "_RECOVERY_PRIMITIVES", ()) or ()
    if isinstance(raw, (list, tuple)):
        names = [str(name or "").strip() for name in raw if str(name or "").strip()]
    else:
        names = sorted({str(name or "").strip() for name in raw if str(name or "").strip()})
    return [name for name in names if name]


def _callable_for_primitive(resource_agent: Any, owner: Any, primitive_name: str) -> Any:
    fn = getattr(resource_agent, primitive_name, None)
    if callable(fn):
        return fn
    fn = getattr(owner, primitive_name, None)
    if callable(fn):
        return fn
    return None


def _schema_properties_for_function(fn: Any) -> tuple[dict[str, Any], list[str], str]:
    analyzer = FunctionAnalyzer()
    analyzed = analyzer.analyze_function(fn)
    parameters = dict(analyzed.get("parameters") or {})
    properties = deepcopy(parameters.get("properties") or {})
    signature = inspect.signature(fn)
    required: list[str] = []
    for param_name, parameter in signature.parameters.items():
        if str(param_name) == "self":
            continue
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if parameter.default is inspect._empty:
            required.append(str(param_name))
        elif parameter.default is None and param_name in properties:
            schema = properties[param_name]
            if isinstance(schema.get("type"), str):
                schema["type"] = [schema["type"], "null"]
            elif isinstance(schema.get("type"), list) and "null" not in schema["type"]:
                schema["type"].append("null")
    description = str(analyzed.get("description") or "").strip()
    return properties, required, description


def _effect_parameters(effects: dict[str, Any], properties: dict[str, Any]) -> list[str]:
    """Name the parameters a declared effect cannot do without, so the catalog asks for them.

    An effect that reads one parameter needs it, even when the call signature leaves it optional;
    an effect with several alternatives needs only one of them, so none becomes required.
    """
    needed: list[str] = []
    for rule in effects.values():
        if not isinstance(rule, dict):
            continue
        for operator, binding in rule.items():
            if operator == "set_from_param" and isinstance(binding, str):
                needed.append(binding)
            elif operator == "set_from_param_any_of" and isinstance(binding, list) and len(binding) == 1:
                needed.append(str(binding[0]))
            elif operator in {"pose_absolute_from_params", "pose_relative_from_params"} and isinstance(binding, list):
                needed.extend(str(name) for name in binding)
    return [name for name in needed if name in properties]


def _effective_catalog_required_params(
    required: list[str],
    *,
    profile: Any,
    primitive_name: str,
) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    contract = resource_event_fact_contract(profile, primitive_name)
    for raw_name in list(required or []) + list(contract.get("required_params") or []):
        token = str(raw_name or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        merged.append(token)
    return merged


def _resource_type_for_agent(
    resource_agent: Any,
    *,
    snapshot: dict[str, Any] | None = None,
) -> str:
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery import recovery_resource_adapter

    static_capabilities = deepcopy(getattr(resource_agent, "static_capabilities", {}) or {})
    return recovery_resource_adapter.resolve_recovery_resource_type(
        resource=resource_agent,
        snapshot=snapshot or {},
        modeled_state={},
        static_capabilities=static_capabilities,
    )


def build_execution_primitive_catalog(resource_agent: Any) -> list[dict[str, Any]]:
    """Build the execution primitive catalog from a resource recovery surface."""
    if resource_agent is None:
        return []
    owner, profile = _build_catalog_owner(resource_agent)
    resource_type = _resource_type_for_agent(resource_agent)
    entries: list[dict[str, Any]] = []
    for primitive_name in _raw_recovery_primitive_names(resource_agent):
        fn = _callable_for_primitive(resource_agent, owner, primitive_name)
        if not callable(fn):
            continue
        properties, required, description = _schema_properties_for_function(fn)
        frontmatter = FunctionAnalyzer._extract_yaml_frontmatter(fn) or {}
        preconditions = deepcopy(frontmatter.get("preconditions") or {})
        effects = deepcopy(frontmatter.get("effects") or {})
        required = _effective_catalog_required_params(
            required + _effect_parameters(effects, properties),
            profile=profile,
            primitive_name=primitive_name,
        )
        observation_schema = deepcopy(
            dict(profile.observation_output_schema_map or {}).get(primitive_name) or {}
        )
        entry = {
            "name": primitive_name,
            "resource_type": resource_type,
            "description": str(frontmatter.get("description") or description or "").strip(),
            "params": properties,
            "required_params": required,
            "preconditions": preconditions,
            "capability_validator": "check_recovery_primitive_feasibility",
            "capability_constraints": deepcopy(frontmatter.get("capability_constraints") or {}),
            "required_evidence": deepcopy(dict(profile.primitive_trace_fact_map or {}).get(primitive_name) or {}),
            "effects": effects,
            "primitive_kind": str(
                dict(profile.primitive_kind_map or {}).get(primitive_name) or ""
            ).strip(),
            "output_schema": observation_schema,
            "synthesis_hidden": bool(frontmatter.get("synthesis_hidden", False)),
        }
        entry["semantic_summary"] = primitive_summary(
            description=str(entry.get("description") or ""),
            preconditions=preconditions,
            effects=effects,
        )
        entries.append(entry)
    return filter_recovery_primitive_catalog(entries, resource_type=resource_type)


def current_primitive_declarations(scene: dict[str, Any]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Read each scene resource's primitive declarations from the code beside its executors.

    The recovery checker reads these at validation time rather than the catalogs a capture recorded
    (paper assumption A.2). Robots and the KMR declare their primitives in docstring frontmatter;
    machines and transport, whose primitives are dispatched by name, declare them as data beside
    their commands.

    Args:
        scene: The Gazebo scene. Its ``resource_programs`` names each resource's primitives, and a
            machine's entry supplies its job control.

    Returns:
        Per resource id and primitive name, the alternative contracts its executor declares. A
        frontmatter declaration is one contract; a primitive its executor leaves undeclared has none.
        An uncontrollable outcome the executor declares, such as a machine's ``complete_job``, is kept
        beside the primitives with ``controllable`` false.
    """
    from cais_spade_llm.recovery_framework.kmr_primitives import KMRPrimitives
    from cais_spade_llm.recovery_framework.workflow_gazebo import TRANSPORT_PRIMITIVE_CONTRACTS
    from cais_spade_llm.resources.machine.machine_jobs import machine_primitive_contracts
    from cais_spade_llm.resources.robot.robot_profile import robot_primitive_owner_class

    machines = {row["resource_id"]: machine_primitive_contracts(row) for row in scene.get("machines", [])}
    robots = {row["resource_id"] for row in scene.get("robots", [])}
    declarations: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for rid, bundle in scene["resource_programs"]["resources"].items():
        if rid in machines:
            contracts = machines[rid]
        elif isinstance(scene.get(rid), dict) and "simulation_transport" in scene[rid]:
            contracts = {name: [deepcopy(contract)] for name, contract in TRANSPORT_PRIMITIVE_CONTRACTS.items()}
        else:
            owner = KMRPrimitives if rid == "KMR" else robot_primitive_owner_class(rid) if rid in robots else None
            contracts = {name: [_frontmatter_contract(getattr(owner, name))]
                         for name in bundle["primitives"] if callable(getattr(owner, name, None))}
        declarations[rid] = {name: contracts.get(name, []) for name in bundle["primitives"]
                             if not ((rid in robots or rid == "KMR") and name in INTERNAL_GRIPPER_PRIMITIVES)}
        declarations[rid].update({name: variants for name, variants in contracts.items()
                                  if any(variant.get("controllable") is False for variant in variants)})
    # A device that reports itself unavailable runs none of its primitives until it is repaired by hand,
    # so every primitive needs its resource available, as the resource's recorded availability states.
    for contracts in declarations.values():
        for name, variants in contracts.items():
            contracts[name] = [{**variant, "preconditions": {"availability": {"equals": "available"},
                                                             **(variant.get("preconditions") or {})}}
                               for variant in variants]
    return declarations


def _frontmatter_contract(fn: Any) -> dict[str, Any]:
    frontmatter = FunctionAnalyzer._extract_yaml_frontmatter(fn) or {}
    contract = {"preconditions": deepcopy(frontmatter.get("preconditions") or {}),
                "effects": deepcopy(frontmatter.get("effects") or {})}
    if frontmatter.get("capability_constraints"):
        contract["capability_constraints"] = deepcopy(frontmatter["capability_constraints"])
    return contract


def filter_synthesis_primitive_catalog(
    primitive_catalog: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Filter a generic LLM-facing primitive surface."""
    filtered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_entry in primitive_catalog or []:
        if not isinstance(raw_entry, dict):
            continue
        name = str(raw_entry.get("name") or "").strip()
        if not name or name in seen:
            continue
        if bool(raw_entry.get("synthesis_hidden")):
            continue
        entry = deepcopy(raw_entry)
        entry.pop("synthesis_hidden", None)
        filtered.append(entry)
        seen.add(name)
    return filtered


def build_synthesis_primitive_catalog(
    resource_agent: Any | None = None,
    *,
    primitive_catalog: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build a generic LLM-facing synthesis catalog from execution primitives."""
    source_catalog = primitive_catalog
    if source_catalog is None and isinstance(resource_agent, list):
        source_catalog = resource_agent
        resource_agent = None
    if source_catalog is None:
        source_catalog = build_execution_primitive_catalog(resource_agent)
    return filter_synthesis_primitive_catalog(source_catalog or [])


def build_primitive_catalog(resource_agent: Any) -> list[dict[str, Any]]:
    """Compatibility alias for callers expecting the older name."""
    return build_execution_primitive_catalog(resource_agent)


def _descriptor_domain_values(*values: Any) -> list[Any]:
    """Return stable exact values for one finite EFA variable declaration."""
    unique: dict[str, Any] = {}
    for value in values:
        key = recovery_validation_fingerprint(value)
        unique.setdefault(key, deepcopy(value))
    return [unique[key] for key in sorted(unique)]


def build_recovery_des_model(
    resource_agent: Any,
    *,
    snapshot: dict[str, Any] | None = None,
    descriptor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and bind an explicit RA-owned recovery-level DES descriptor.

    The base resource never derives a DES alphabet from execution primitives.
    Resource subclasses may pass a task-level descriptor, while simpler
    resources may declare one under ``static_capabilities.recovery_des_model``.
    Explicit state and primitive declarations may have no predefined events.
    Missing or malformed state declarations fail closed.
    """
    if resource_agent is None:
        return {}
    live_snapshot = (
        deepcopy(snapshot)
        if isinstance(snapshot, dict)
        else get_resource_recovery_snapshot(resource_agent)
    )
    static_capabilities = deepcopy(
        getattr(resource_agent, "static_capabilities", {}) or {}
    )
    raw_descriptor = deepcopy(
        descriptor
        if isinstance(descriptor, dict)
        else static_capabilities.get("recovery_des_model") or {}
    )
    raw_state_variables = raw_descriptor.get("state_variables")
    raw_events = raw_descriptor.get("events", [])
    if not isinstance(raw_state_variables, dict) or not raw_state_variables or not isinstance(raw_events, list):
        return {}

    state_variables: dict[str, dict[str, Any]] = {}
    current_valuation: dict[str, Any] = {}
    configured_valuation = dict(raw_descriptor.get("current_valuation") or {})
    for raw_field_name in sorted(raw_state_variables):
        field_name = str(raw_field_name or "").strip()
        declaration = raw_state_variables.get(raw_field_name)
        if not field_name or not isinstance(declaration, dict):
            return {}
        scope = str(declaration.get("scope") or "resource").strip()
        domain = declaration.get("domain")
        if scope not in {"resource", "part"} or not isinstance(domain, list):
            return {}
        snapshot_field = {
            "resource_state": "current_state",
            "resource_location": "current_location",
        }.get(field_name, field_name)
        if field_name in live_snapshot:
            current_value = deepcopy(live_snapshot.get(field_name))
        elif snapshot_field in live_snapshot:
            current_value = deepcopy(live_snapshot.get(snapshot_field))
        else:
            current_value = deepcopy(configured_valuation.get(field_name))
        state_variables[field_name] = {
            "scope": scope,
            "domain": _descriptor_domain_values(
                current_value,
                *domain,
            ),
        }
        if declaration.get("private") is True:
            state_variables[field_name]["private"] = True
        if scope == "resource":
            current_valuation[field_name] = current_value

    events: list[dict[str, Any]] = []
    for raw_event in raw_events:
        if not isinstance(raw_event, dict):
            return {}
        event_name = str(raw_event.get("event_name") or "").strip()
        guards = raw_event.get("guards") or {}
        updates = raw_event.get("updates") or {}
        if (
            not event_name
            or not isinstance(guards, dict)
            or not isinstance(updates, dict)
            or any(field_name not in state_variables for field_name in guards)
            or any(field_name not in state_variables for field_name in updates)
            or any(not isinstance(value, dict) for value in guards.values())
            or any(not isinstance(value, dict) for value in updates.values())
        ):
            return {}
        event = {
            "event_name": event_name,
            "controllable": bool(raw_event.get("controllable", True)),
            "observable": bool(raw_event.get("observable", True)),
            "guards": deepcopy(guards),
            "updates": deepcopy(updates),
        }
        for key in (
            "parameter_bindings",
            "requires_part_binding",
            "recovery_visible_steps",
            "primitive_support",
            "composition_contract",
            "source",
        ):
            if key in raw_event:
                event[key] = deepcopy(raw_event.get(key))
        task_definition = getattr(getattr(resource_agent, event_name, None), "__tool_spec__", None)
        program = getattr(task_definition, "program", None)
        if program is not None and is_dataclass(program):
            contract = deepcopy(event.get("composition_contract") or {})
            contract.update({
                "steps": [asdict(step) for step in program.steps],
                "entry_state": program.entry_state,
                "part_in_state": program.part_in_state,
                "entry_guards": [asdict(guard) for guard in program.entry_guards],
                "required_context_keys": list(program.required_context_keys),
                "effects": [asdict(effect) for effect in program.effects],
                "part_transition": deepcopy(program.part_transition),
                "params": task_definition.argument_properties(),
                "required_params": task_definition.required_argument_names(),
            })
            event["composition_contract"] = contract
        events.append(event)
    resource_jid = str(
        getattr(resource_agent, "jid", "")
        or live_snapshot.get("resource_jid")
        or live_snapshot.get("agent_name")
        or ""
    ).strip()
    bound_descriptor = {
        "model_type": "extended_finite_automaton",
        "resource_jid": resource_jid,
        "state_variables": state_variables,
        "current_valuation": current_valuation,
        "local_event_alphabet": [row["event_name"] for row in events],
        "controllable_event_alphabet": [
            row["event_name"] for row in events if row["controllable"]
        ],
        "observable_event_alphabet": [
            row["event_name"] for row in events if row["observable"]
        ],
        "events": events,
        "primitive_catalog": _execution_catalog_for_snapshot(resource_agent),
        "marked_state_conditions": deepcopy(
            raw_descriptor.get("marked_state_conditions")
            or static_capabilities.get("recovery_marked_state_conditions")
            or []
        ),
    }
    bound_descriptor["descriptor_fingerprint"] = recovery_validation_fingerprint(
        bound_descriptor
    )
    return bound_descriptor


def recovery_des_declared_state_fields(descriptor: dict[str, Any] | None) -> set[str]:
    """Return exact transition-state fields declared by one RA descriptor."""
    return {
        str(field_name)
        for field_name in dict((descriptor or {}).get("state_variables") or {})
        if str(field_name)
    }


def build_primitive_reference_card(primitive_catalog: list[dict[str, Any]] | None) -> str:
    """Render a compact human-readable reference card for prompt builders."""
    lines: list[str] = []
    for entry in filter_synthesis_primitive_catalog(primitive_catalog or []):
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        description = str(entry.get("description") or entry.get("semantic_summary") or "").strip()
        params = dict(entry.get("params") or {})
        required = {
            str(item).strip() for item in (entry.get("required_params") or []) if str(item).strip()
        }
        param_tokens: list[str] = []
        for param_name, schema in params.items():
            if not isinstance(schema, dict):
                continue
            type_name = str(schema.get("type") or "string").strip()
            suffix = " required" if param_name in required else ""
            param_tokens.append(f"{param_name}:{type_name}{suffix}")
        param_text = ", ".join(param_tokens) if param_tokens else "(no params)"
        lines.append(f"- {name}: {description}")
        lines.append(f"  params: {param_text}")
    return "\n".join(lines).strip()


def _snapshot_builder_payload(resource_agent: Any, profile: Any) -> dict[str, Any]:
    if profile.snapshot_builder is not None:
        built = profile.snapshot_builder(resource_agent)
        if isinstance(built, dict):
            return deepcopy(built)
    if hasattr(resource_agent, "_snapshot_state"):
        try:
            built = resource_agent._snapshot_state()
        except Exception:
            built = None
        if isinstance(built, dict):
            return deepcopy(built)
    return {}


def _declared_snapshot_fields(profile: Any, catalog: list[dict[str, Any]]) -> list[str]:
    fields = [str(field) for field in profile.snapshot_fields or ()]
    for entry in catalog:
        for key in ("preconditions", "effects"):
            fields.extend(str(field) for field in (entry.get(key) or {}))
    return list(dict.fromkeys(fields))


def _execution_catalog_for_snapshot(resource_agent: Any) -> list[dict[str, Any]]:
    method = getattr(resource_agent, "recovery_execution_primitive_catalog", None)
    if callable(method):
        try:
            catalog = method()
            if isinstance(catalog, list):
                return deepcopy(catalog)
        except Exception:
            pass
    return build_execution_primitive_catalog(resource_agent)


def get_resource_recovery_snapshot(resource_agent: Any) -> dict[str, Any]:
    """Build the canonical recovery snapshot for a resource without recursion."""
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery import recovery_resource_adapter

    if resource_agent is None:
        return {}
    profile = get_resource_profile_for_agent(resource_agent)
    raw_snapshot = _snapshot_builder_payload(resource_agent, profile)
    resource_jid = str(
        getattr(resource_agent, "jid", "") or getattr(resource_agent, "agent_name", "") or ""
    ).strip()
    resource_type = _resource_type_for_agent(resource_agent, snapshot=raw_snapshot)
    adapted_resource = recovery_resource_adapter.adapt_recovery_resource_snapshot(
        resource_jid=resource_jid,
        resource_type=resource_type,
        snapshot=raw_snapshot,
        modeled_state={},
    )
    execution_catalog = _execution_catalog_for_snapshot(resource_agent)
    # Every resource keeps the conditions it reports or its primitives declare, not only those a profile's
    # facet lists, so a macro checks each primitive against the resource's actual facts.
    for field in _declared_snapshot_fields(profile, execution_catalog):
        if field not in adapted_resource and field in raw_snapshot:
            adapted_resource[field] = deepcopy(raw_snapshot[field])
    adapted_resource["recovery_adapter"] = (
        recovery_resource_adapter.recovery_resource_capabilities(
            resource_type,
            primitive_catalog=execution_catalog,
        )
    )
    return adapted_resource


def sync_agent_from_recovery_snapshot(resource_agent: Any, snapshot: dict[str, Any]) -> None:
    """Apply canonical snapshot fields back onto the live resource agent."""
    if resource_agent is None:
        return
    profile = get_resource_profile_for_agent(resource_agent)

    current_state = resource_snapshot_field_value(snapshot, "current_state", profile=profile)
    resource_agent._current_state = current_state

    current_location = resource_snapshot_field_value(snapshot, "current_location", profile=profile)
    if hasattr(resource_agent, "_current_location") or current_location is not None:
        resource_agent._current_location = current_location

    availability = resource_snapshot_availability(snapshot, profile=profile)
    if hasattr(resource_agent, "_availability") or availability:
        resource_agent._availability = availability

    for field, target in dict(profile.sync_map or {}).items():
        value = resource_snapshot_field_value(snapshot, str(field), profile=profile)
        if callable(target):
            target(resource_agent, deepcopy(value))
            continue
        attr_name = str(target or "").strip()
        if attr_name:
            setattr(resource_agent, attr_name, deepcopy(value))


__all__ = [
    "INTERNAL_GRIPPER_PRIMITIVES",
    "filter_recovery_primitive_catalog",
    "build_recovery_des_model",
    "build_execution_primitive_catalog",
    "build_primitive_catalog",
    "build_primitive_reference_card",
    "build_synthesis_primitive_catalog",
    "filter_synthesis_primitive_catalog",
    "get_resource_recovery_snapshot",
    "primitive_summary",
    "recovery_des_declared_state_fields",
    "sync_agent_from_recovery_snapshot",
]
