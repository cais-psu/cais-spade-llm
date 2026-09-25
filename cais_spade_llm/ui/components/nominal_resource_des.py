"""Read-only nominal capabilities, configured inventory, and DES details."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from collections.abc import Awaitable, Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

from nicegui import context, ui

from cais_spade_llm.resources.environment_models import build_environment_models, process_json
from cais_spade_llm.ui.bridge import SystemBridge
from cais_spade_llm.ui.components.resource_function_catalog import (
    render_resource_function_rows,
    resource_function_rows,
)

logger = logging.getLogger(__name__)
SCENE_PATH = Path(__file__).resolve().parents[2] / "initialization/recovery_framework_gazebo.json"


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _diagram_label(value: Any) -> str:
    return (
        _text(value)
        .replace("&", "#38;")
        .replace('"', "#34;")
        .replace("<", "#60;")
        .replace(">", "#62;")
    )


def _event_signature(event: dict) -> str:
    parameters = []
    for name in (
        "part_name",
        "origin_resource_location",
        "destination_location",
        "source_resource",
        "target_resource",
        "loading_position",
        "zone",
        "downstream_zone",
        "delivered_part",
        "next_locations",
    ):
        binding = event["parameter_bindings"].get(name)
        if binding is not None:
            parameters.append(f"{name}={_text(binding['equals'])}" if "equals" in binding else name)
    return f"{event['event_name']}({', '.join(parameters)})"


def _state_text(state: dict) -> str:
    return "\n".join(f"{field} = {_text(value)}" for field, value in state.items())


def _capability_events(model: dict, models: dict) -> list[dict]:
    events = [
        event
        for event in model["events"]
        if event["updates"]
        or event.get("collection_effects")
        or event["parameter_bindings"]["resource_id"]["equals"] == model["resource_id"]
    ]
    return sorted(events, key=lambda event: event["event_id"])


def _graph_parameter(event: dict, name: str) -> Any:
    binding = event["parameter_bindings"].get(name, {})
    if "equals" in binding:
        return binding["equals"]
    return {"reference": name}


def _graph_field(resource: str, field: str) -> str:
    return f"{resource}.{field}"


def _graph_value(event: dict, value: Any) -> Any:
    if isinstance(value, dict):
        if "set_from_param" in value:
            return _graph_parameter(event, value["set_from_param"])
        return {key: _graph_value(event, item) for key, item in value.items()}
    if isinstance(value, list):
        return [_graph_value(event, item) for item in value]
    return value


def _graph_transition(event: dict, models: dict) -> dict:
    actor = event["parameter_bindings"]["resource_id"]["equals"]
    peers = {
        rid: peer
        for rid, model in models.items()
        for peer in model["events"]
        if peer["event_id"] == event["event_id"]
    }
    updates, guards = {}, {}
    for rid, peer in peers.items():
        for field, guard in peer["guards"].items():
            operator, value = next(iter(guard.items()))
            if operator.endswith("_from_param"):
                operator = operator.removesuffix("_from_param")
                value = _graph_parameter(event, value)
            guards[_graph_field(rid, field)] = {operator: value}
        for field, update in peer["updates"].items():
            operator, value = next(iter(update.items()))
            if operator == "set_from_param":
                value = _graph_value(event, update)
            updates[_graph_field(rid, field)] = value
        for field, effect in peer.get("collection_effects", {}).items():
            if field in models[rid]["state_variables"]:
                updates.setdefault(_graph_field(rid, field), {"collection_effect": effect})
    return {
        "event": event,
        "actor": actor,
        "source": {field: guard["equals"] for field, guard in guards.items() if "equals" in guard},
        "guards": guards,
        "updates": updates,
        "peers": peers,
    }


def _graph_equal(actual: Any, expected: Any) -> bool | None:
    # Collection outcomes and unbound parameters impose edge conditions; they
    # are not concrete observations or proof that two task bindings are equal.
    if isinstance(actual, dict) and "collection_effect" in actual:
        return None
    if type(actual) is type(expected) and actual == expected:
        return True
    if any(isinstance(value, dict) and "reference" in value for value in (actual, expected)):
        return False if actual is None or expected is None else None
    return False


def _graph_apply(state: dict, transition: dict) -> dict | None:
    after = deepcopy(state)
    for field, guard in transition["guards"].items():
        operator, expected = next(iter(guard.items()))
        if field in after:
            equal = _graph_equal(after[field], expected)
            if (operator == "equals" and equal is False) or (
                operator == "not_equals" and equal is True
            ):
                return None
        if operator == "equals":
            after[field] = deepcopy(expected)
    for field, value in transition["updates"].items():
        if isinstance(value, dict) and "collection_effect" in value:
            # A shared collection update invalidates earlier indexed facts
            # without renaming parameters such as delivered_part to part_name.
            prefix = field.split("{", 1)[0]
            for indexed in after:
                if "{" in field and indexed.startswith(prefix):
                    after[indexed] = deepcopy(value)
    after.update(deepcopy(transition["updates"]))
    return after


def nominal_capability_graph(model: dict, models: dict | None = None) -> dict:
    """Build conditional local states and task/handoff edges for one resource.

    Args:
        model: The resource whose local guards and updates define the states.
        models: Other participants' descriptors, used only for edge details.

    Returns:
        Local parameterized states and existing event edges. Unknown entry facts
        and peer guards remain conditions, not evidence of executable tasks.
    """
    models = {**(models or {}), model["resource_id"]: model}
    local = {model["resource_id"]: model}
    transitions = [_graph_transition(event, local) for event in _capability_events(model, models)]
    for row in transitions:
        row["peers"] = {
            rid: peer
            for rid, descriptor in models.items()
            for peer in descriptor["events"]
            if peer["event_id"] == row["event"]["event_id"]
        }
    nodes, edges, indices, queue = [], [], {}, deque()

    def intern(state: dict) -> int:
        key = json.dumps(state, sort_keys=True, separators=(",", ":"))
        if key not in indices:
            indices[key] = len(nodes)
            nodes.append({"id": len(nodes), "state": state})
            queue.append(len(nodes) - 1)
        return indices[key]

    represented = set()
    for entry in transitions:
        if entry["event"]["event_id"] in represented:
            continue
        # Capabilities can start under different conditions; current inventory
        # must neither seed this graph nor hide an otherwise configured task.
        intern(entry["source"])
        while queue:
            source = queue.popleft()
            for row in transitions:
                after = _graph_apply(nodes[source]["state"], row)
                if after is None:
                    continue
                target = intern(after)
                event = row["event"]
                represented.add(event["event_id"])
                edges.append(
                    {
                        "source": source,
                        "target": target,
                        "event_id": event["event_id"],
                        "resource_id": row["actor"],
                        "event": event,
                        "guards": {rid: peer["guards"] for rid, peer in row["peers"].items()},
                        "updates": {rid: peer["updates"] for rid, peer in row["peers"].items()},
                        **{
                            field: {
                                rid: peer[field] for rid, peer in row["peers"].items() if field in peer
                            }
                            for field in ("collection_guards", "collection_effects")
                        },
                    }
                )
    return {"nodes": nodes, "edges": edges}


def nominal_capability_rows(
    model: dict[str, Any], models: dict | None = None
) -> list[dict[str, Any]]:
    """List local task and handoff conditions without expanding part bindings.

    Args:
        model: One resource descriptor with local guards and updates.
        models: Shared descriptors; neighbors do not expand this projection.

    Returns:
        Tasks and shared handoffs with only the selected resource's effects.
    """
    rows = []
    for event in _capability_events(model, models or {}):
        transition = _graph_transition(event, {model["resource_id"]: model})
        rows.append(
            {
                "id": event["event_id"],
                "resource_id": event["parameter_bindings"]["resource_id"]["equals"],
                "event": event["event_name"],
                "signature": _event_signature(event),
                "source": _state_text(dict(sorted(transition["guards"].items()))),
                "target": _state_text(dict(sorted(transition["updates"].items()))),
            }
        )
    return rows


def nominal_capability_mermaid(
    model: dict[str, Any], event_id: int | None = None, *, models: dict | None = None
) -> str:
    """Render the selected resource's local states, tasks, and shared handoffs.

    Args:
        model: One nominal resource descriptor.
        event_id: An existing event to highlight, if supplied.
        models: Other participants' descriptors for edge details only.

    Returns:
        A graph retaining exact resource, location, event, and parameter names.
        It describes available tasks, not a live state or a scheduling policy.
    """
    return _capability_graph_mermaid(nominal_capability_graph(model, models), event_id)


def _capability_graph_mermaid(graph: dict, event_id: int | None = None) -> str:
    lines = ["flowchart TB"]
    for node in graph["nodes"]:
        state = dict(sorted(node["state"].items()))
        label = "<br/>".join(_diagram_label(line) for line in _state_text(state).splitlines())
        lines.append(f'    s{node["id"]}["{label}"]')
    for index, edge in enumerate(graph["edges"]):
        event = edge["event"]
        label = _diagram_label(
            f"{edge['resource_id']}: {_event_signature(event)} [event_id={event['event_id']}]"
        )
        label = label.replace(", ", ",<br/>")
        lines.append(f'    s{edge["source"]} -->|"{label}"| s{edge["target"]}')
        if edge["event_id"] == event_id:
            lines.append(f"    linkStyle {index} stroke:#d97706,stroke-width:4px")
    return "\n".join(lines)


def nominal_resource_capability_diagram(model: dict) -> dict:
    """Draw one resource's declared capability transitions and event names.

    Args:
        model: One resource descriptor containing the existing event endpoints.

    Returns:
        One compact graph, its exact endpoint table, and original event IDs.
        Resource valuations and shared guards remain separate conditions.
    """
    nodes: list[dict] = []
    edges: list[dict] = []
    node_ids: dict[str, str] = {}
    edge_ids: dict[tuple[str, str, str], int] = {}

    def intern(value: dict) -> str:
        key = json.dumps(value, sort_keys=True, ensure_ascii=False)
        if key not in node_ids:
            node_id = f"s{len(nodes)}"
            node_ids[key] = node_id
            nodes.append({"id": node_id, "value": deepcopy(value)})
        return node_ids[key]

    for event in _capability_events(model, {}):
        transition = event["capability_transition"]
        source = intern(transition["source"])
        target = intern(transition["target"])
        key = (source, target, event["event_name"])
        if key not in edge_ids:
            edge_ids[key] = len(edges)
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "event_name": event["event_name"],
                    "event_ids": [],
                }
            )
        edges[edge_ids[key]]["event_ids"].append(event["event_id"])

    lines = ["flowchart LR"]
    state_rows = []
    for node in nodes:
        value = node["value"]
        short: list[str] | None = None
        if len(value) == 1:
            field, item = next(iter(value.items()))
            candidate = _text(item)
            if len(field) <= 20 and len(candidate) <= 24:
                short = [field, candidate]
        elif set(value) == {"part_location", "zone"}:
            short = [f"zone = {_text(value['zone'])}"]
        label = node["id"] if short is None else "<br/>".join(
            [node["id"], *(_diagram_label(part) for part in short)]
        )
        lines.append(f'    {node["id"]}(("{label}"))')
        state_rows.append({"id": node["id"], "value": _text(value)})
    for edge in edges:
        lines.append(
            f'    {edge["source"]} -->|"{_diagram_label(edge["event_name"])}"| {edge["target"]}'
        )
    return {"nodes": nodes, "edges": edges, "state_rows": state_rows, "mermaid": "\n".join(lines)}


def nominal_resource_default_fields(model: dict) -> list[str]:
    """Choose the exact local state variables shown for a resource by default.

    Args:
        model: One configured resource descriptor.

    Returns:
        Existing state-variable names in diagram order.
    """
    resource_id = model["resource_id"]
    if resource_id in {"ur5e-1", "ur5e-2", "ur5e-3", "ur5e-4"}:
        requested = ["resource_state"]
    elif resource_id in {"M1", "M2"}:
        requested = ["resource_state", "staging_part"]
    elif resource_id == "KMR":
        requested = ["resource_state", "resource_location"]
    elif resource_id == "Conveyor":
        requested = [
            "part_location.{part_name}", "part_order.{part_name}", "loading_reserved_by"
        ]
    elif resource_id == "Buffer For Machined parts":
        requested = [f"zone_{zone}_part" for zone in (1, 2, 3, 4)]
    elif resource_id == "Storage":
        requested = [field for field in model["state_variables"] if field.startswith("inventory.")]
    elif resource_id == "3D Printing Station":
        requested = [field for field in model["state_variables"] if field.startswith("output.")]
    elif resource_id == "Exit":
        requested = ["resource_state", "product_location"]
    else:
        requested = list(model["state_variables"])[:1]
    return [field for field in requested if field in model["state_variables"]]


def _process_requirement_label(requirement: dict) -> str:
    return " ".join(f"{key}={_text(value)}" for key, value in requirement.items())


def nominal_product_process_plan_diagram(
    requirements: dict, initial_product_states: dict, part_name: str
) -> dict:
    """Draw one part's ordered processPlan specification and configured start.

    Args:
        requirements: Resolved, ordered processesToComplete for selected parts.
        initial_product_states: Product states before any acknowledged task.
        part_name: Exact selected part identifier.

    Returns:
        A per-part process-stage graph. Transport events stutter in this view.
    """
    steps = requirements[part_name]
    effects = []
    groups = []
    for step in steps:
        group = []
        for requirement in step["processesToComplete"]:
            group.append(len(effects))
            effects.append(requirement)
        groups.append(group)

    nodes: list[dict] = []
    edges: list[dict] = []
    node_ids: dict[frozenset[int], str] = {}
    queue: deque[frozenset[int]] = deque()

    def intern(completed: frozenset[int]) -> str:
        if completed not in node_ids:
            node_id = f"p{len(nodes)}"
            node_ids[completed] = node_id
            nodes.append({
                "id": node_id,
                "completed": [deepcopy(effects[index]) for index in sorted(completed)],
                "indices": sorted(completed),
            })
            queue.append(completed)
        return node_ids[completed]

    intern(frozenset())
    initial_effects = initial_product_states.get(part_name, {}).get("processCompleted", [])
    initial = frozenset(
        index for index, effect in enumerate(effects) if effect in initial_effects
    )
    initial_id = intern(initial)
    while queue:
        completed = queue.popleft()
        group = next(
            (group for group in groups if any(index not in completed for index in group)),
            None,
        )
        if group is None:
            continue
        for index in group:
            if index in completed:
                continue
            target = frozenset((*completed, index))
            edges.append({
                "source": node_ids[completed],
                "target": intern(target),
                "requirement": deepcopy(effects[index]),
            })

    lines = ["flowchart LR"]
    state_rows = []
    for node in nodes:
        completed = frozenset(node["indices"])
        next_step = next(
            (number for number, group in enumerate(groups, 1)
             if any(index not in completed for index in group)),
            None,
        )
        label = node["id"] if next_step is None else f"{node['id']}<br/>step {next_step}"
        if next_step is None:
            lines.append(f'    {node["id"]}((("{label}")))')
        else:
            lines.append(f'    {node["id"]}(("{label}"))')
        state_rows.append({
            "id": node["id"],
            "completed": deepcopy(node["completed"]),
            "next_step": next_step,
        })
    lines.extend(["    start(( ))", f"    start --> {initial_id}"])
    for edge in edges:
        label = _diagram_label(_process_requirement_label(edge["requirement"]))
        lines.append(f'    {edge["source"]} -->|"{label}"| {edge["target"]}')
    return {
        "part_name": part_name,
        "nodes": nodes,
        "edges": edges,
        "state_rows": state_rows,
        "initial_id": initial_id,
        "mermaid": "\n".join(lines),
    }


def _declares_process_effect(event: dict, requirement: dict) -> bool:
    for effect in event["product_effects"].get("processCompleted", []):
        if set(effect) != set(requirement):
            continue
        for field, value in effect.items():
            if isinstance(value, dict) and "set_from_param" in value:
                binding = event["parameter_bindings"].get(value["set_from_param"], {})
                if "equals" in binding and binding["equals"] != requirement[field]:
                    break
            elif value != requirement[field]:
                break
        else:
            return True
    return False


def nominal_product_process_event_rows(models: dict, requirements: dict, part_name: str) -> list[dict]:
    """List declared event variants that can establish one part's process steps.

    Args:
        models: Resource descriptors from one configured model snapshot.
        requirements: Resolved processesToComplete for selected parts.
        part_name: Exact selected part identifier.

    Returns:
        Potential event variants with their unchanged participant conditions.
    """
    rows = []
    for step_number, step in enumerate(requirements[part_name], 1):
        for requirement in step["processesToComplete"]:
            for resource_id, model in models.items():
                for event in model["events"]:
                    actor = event["parameter_bindings"]["resource_id"]["equals"]
                    part_binding = event["parameter_bindings"].get("part_name", {})
                    if (
                        actor != resource_id
                        or ("equals" in part_binding and part_binding["equals"] != part_name)
                        or not _declares_process_effect(event, requirement)
                    ):
                        continue
                    peers = {
                        peer_id: next(
                            item for item in models[peer_id]["events"]
                            if item["event_id"] == event["event_id"]
                        )
                        for peer_id in event["participants"]
                    }
                    rows.append({
                        "step": step_number,
                        "requirement": deepcopy(requirement),
                        "event_id": event["event_id"],
                        "event_name": event["event_name"],
                        "actor": actor,
                        "participants": list(event["participants"]),
                        "parameter_bindings": deepcopy(event["parameter_bindings"]),
                        "guards": {rid: deepcopy(peer["guards"]) for rid, peer in peers.items()},
                        "updates": {rid: deepcopy(peer["updates"]) for rid, peer in peers.items()},
                        "product_guards": deepcopy(event.get("product_guards", {})),
                        "product_effects": deepcopy(event["product_effects"]),
                        "collection_guards": {
                            rid: deepcopy(peer["collection_guards"])
                            for rid, peer in peers.items() if "collection_guards" in peer
                        },
                        "collection_effects": {
                            rid: deepcopy(peer["collection_effects"])
                            for rid, peer in peers.items() if "collection_effects" in peer
                        },
                    })
    return rows


def environment_capability_mermaid(environment: dict) -> str:
    """Render the selected returned path, or the discovered product-state edges."""
    transitions = environment.get("edges", [])
    selected = environment.get("selected_path", [])
    if selected:
        path = []
        source_id = environment.get("initial_state_id")
        for task in selected:
            edge = next(
                (
                    edge
                    for edge in transitions
                    if edge["task"] == task and edge["source_id"] == source_id
                ),
                None,
            )
            if edge is None:
                break
            path.append(edge)
            source_id = edge["target_id"]
        transitions = path
    lines = ["flowchart LR"]
    states: dict[str, str] = {}
    for edge in transitions:
        for endpoint in ("source", "target"):
            state, key = edge[endpoint], edge[f"{endpoint}_id"]
            if key not in states:
                node = f"e{len(states)}"
                states[key] = node
                lines.append(f'    {node}["{_diagram_label(state)}"]')
        source = states[edge["source_id"]]
        target = states[edge["target_id"]]
        task = edge["task"]
        label = _diagram_label(f"{task['resource_id']}: {task['event_name']}")
        lines.append(f'    {source} -->|"{label}"| {target}')
    if not transitions:
        lines.append('    pending["Awaiting capability replies"]')
    return "\n".join(lines)


def nominal_inventory_rows(model: dict[str, Any]) -> list[dict[str, Any]]:
    """List only the configured identities relevant to the selected resource.

    Args:
        model: One nominal resource descriptor.

    Returns:
        Exact part names and their configured/assumed initial facts, if any.
    """
    if model.get("schema_version") in {2, 3}:
        facts: dict[str, dict] = {}
        for field, value in model["current_valuation"].items():
            part = None
            if field.startswith(("inventory.", "output.")) and value is True:
                part = field.split(".", 1)[1]
            elif field.startswith("part_location.") and value is not None:
                part = field.split(".", 1)[1]
            elif field in {"held_part", "part_name", "staging_part"} or field.startswith("zone_"):
                part = value
            if isinstance(part, str):
                facts.setdefault(part, {})[field] = value
        return [
            {"part_name": part, "initial": json.dumps(values, ensure_ascii=False)}
            for part, values in facts.items()
        ]
    assignments = model["assignments"]
    parts = []
    for field in ("held_part", "part_name"):
        parts.extend(
            value
            for value in model["state_variables"].get(field, {}).get("domain", [])
            if value is not None
        )
    for field in ("nominal_parts", "slots", "supported_products"):
        parts.extend(assignments.get(field, []))
    rows = []
    for part in dict.fromkeys(parts):
        initial = {
            field: value
            for field, value in model["current_valuation"].items()
            if field
            in {
                f"{prefix}.{part}"
                for prefix in ("inventory", "output", "part_location", "part_order", "assembled")
            }
            or value == part
        }
        rows.append(
            {
                "part_name": part,
                "initial": json.dumps(initial, ensure_ascii=False, indent=2) if initial else "—",
            }
        )
    return rows


def nominal_resource_state_diagram(
    model: dict[str, Any], field: str, *, initial_valuation: dict | None = None
) -> dict:
    """Project guarded local events onto one declared resource state variable.

    Args:
        model: One nominal resource descriptor.
        field: An exact state-variable name from the descriptor.
        initial_valuation: Initial values; the UI supplies configured values.

    Returns:
        A compact DES diagram with exact values, event names, and original IDs.
        Other participant conditions remain in the full event details.
    """
    declaration = model["state_variables"][field]
    if initial_valuation is None:
        initial_valuation = model["current_valuation"]
    local_graph = nominal_capability_graph(model)
    state_key = _graph_field(model["resource_id"], field)
    nodes: list[dict] = []
    edges: list[dict] = []
    node_ids: dict[str, str] = {}
    edge_ids: dict[tuple[str, str, str], int] = {}

    def intern(value: Any) -> str:
        key = json.dumps(value, sort_keys=True, ensure_ascii=False)
        if key not in node_ids:
            node_id = f"s{len(nodes)}"
            node_ids[key] = node_id
            nodes.append({"id": node_id, "value": deepcopy(value)})
        return node_ids[key]

    if "{part_name}" not in field:
        for value in declaration.get("domain", []):
            intern(value)
    initial_id = intern(initial_valuation[field]) if field in initial_valuation else None

    for edge in local_graph["edges"]:
        event = edge["event"]
        if field not in event["updates"] and field not in event.get("collection_effects", {}):
            continue
        before = local_graph["nodes"][edge["source"]]["state"]
        after = local_graph["nodes"][edge["target"]]["state"]
        source_value = before.get(state_key, {})
        guard = event["guards"].get(field, {})
        if "equals" in guard:
            source_value = guard["equals"]
        elif "equals_from_param" in guard:
            source_value = _graph_parameter(event, guard["equals_from_param"])
        target_value = after.get(state_key, {})
        source = intern(source_value)
        target = intern(target_value)
        key = (source, target, event["event_name"])
        if key not in edge_ids:
            edge_ids[key] = len(edges)
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "event_name": event["event_name"],
                    "event_ids": [],
                }
            )
        if event["event_id"] not in edges[edge_ids[key]]["event_ids"]:
            edges[edge_ids[key]]["event_ids"].append(event["event_id"])

    marked = {
        json.dumps(condition[field]["equals"], sort_keys=True, ensure_ascii=False)
        for condition in model["marked_state_conditions"]
        if set(condition) == {field} and "equals" in condition[field]
    }
    lines = ["flowchart LR"]
    state_rows = []
    for node in nodes:
        value = node["value"]
        value_text = _text(value)
        label = node["id"] if isinstance(value, dict) or len(value_text) > 24 else (
            f"{node['id']}<br/>{_diagram_label(value_text)}"
        )
        if json.dumps(value, sort_keys=True, ensure_ascii=False) in marked:
            lines.append(f'    {node["id"]}((("{label}")))')
        else:
            lines.append(f'    {node["id"]}(("{label}"))')
        state_rows.append({"id": node["id"], "value": value_text})
    if initial_id is not None:
        lines.extend(["    start(( ))", f"    start --> {initial_id}"])
    for edge in edges:
        lines.append(
            f'    {edge["source"]} -->|"{_diagram_label(edge["event_name"])}"| {edge["target"]}'
        )
    return {
        "field": field,
        "nodes": nodes,
        "edges": edges,
        "state_rows": state_rows,
        "initial_id": initial_id,
        "mermaid": "\n".join(lines),
    }


def nominal_des_mermaid(
    model: dict[str, Any], field: str, *, initial_valuation: dict | None = None
) -> str:
    """Render one declared state-variable projection as a guarded DES diagram."""
    return nominal_resource_state_diagram(
        model, field, initial_valuation=initial_valuation
    )["mermaid"]


def nominal_state_rows(model: dict[str, Any]) -> list[dict[str, Any]]:
    """Return exact declarations and explicitly labeled initial values.

    Args:
        model: One nominal resource descriptor.

    Returns:
        Rows for the state-variable table.
    """
    return [
        {
            "field": field,
            "scope": declaration["scope"],
            "domain": _text(
                declaration.get(
                    "domain", {key: value for key, value in declaration.items() if key != "scope"}
                )
            ),
            "initial": _text(
                model["current_valuation"].get(
                    field,
                    {
                        name: value
                        for name, value in model["current_valuation"].items()
                        if "{part_name}" in field and name.startswith(field.split("{", 1)[0])
                    },
                )
            ),
        }
        for field, declaration in model["state_variables"].items()
    ]


def nominal_event_rows(models: dict[str, dict], resource_id: str, event_name: str) -> list[dict]:
    """Include every participant's guards and updates in the event table.

    Args:
        models: All nominal descriptors.
        resource_id: Exact selected resource.
        event_name: Exact selected task event.

    Returns:
        Parameterized task rows with matching shared handoff definitions.
    """
    rows = []
    for event in models[resource_id]["events"]:
        if event["event_name"] != event_name:
            continue
        guards, updates = {}, {}
        for participant in event["participants"]:
            peer = next(
                row for row in models[participant]["events"] if row["event_id"] == event["event_id"]
            )
            if peer["guards"]:
                guards[participant] = peer["guards"]
            if peer["updates"]:
                updates[participant] = peer["updates"]
        rows.append(
            {
                "id": event["event_id"],
                "event": event_name,
                "bindings": json.dumps(event["parameter_bindings"], ensure_ascii=False, indent=2),
                "guards": json.dumps(guards, ensure_ascii=False, indent=2),
                "updates": json.dumps(updates, ensure_ascii=False, indent=2),
                "notes": event["notes"],
                "controllable": event["controllable"],
                "observable": event["observable"],
            }
        )
    return rows


def nominal_composition_event_rows(models: dict[str, dict], resource_id: str) -> list[dict]:
    """Collect exact shared variants shown by one resource's local diagrams.

    Args:
        models: Resource descriptors from one configured model snapshot.
        resource_id: Exact selected resource identifier.

    Returns:
        One row per event_id with every participant's unchanged conditions.
    """
    rows = []
    for event in sorted(models[resource_id]["events"], key=lambda item: item["event_id"]):
        peers = {
            rid: next(
                peer for peer in models[rid]["events"]
                if peer["event_id"] == event["event_id"]
            )
            for rid in event["participants"]
        }
        rows.append({
            "event_id": event["event_id"],
            "event_name": event["event_name"],
            "actor": event["parameter_bindings"]["resource_id"]["equals"],
            "participants": list(event["participants"]),
            "parameter_bindings": deepcopy(event["parameter_bindings"]),
            "guards": {rid: deepcopy(peer["guards"]) for rid, peer in peers.items()},
            "updates": {rid: deepcopy(peer["updates"]) for rid, peer in peers.items()},
            "product_guards": deepcopy(event.get("product_guards", {})),
            "product_effects": deepcopy(event["product_effects"]),
            "collection_guards": {
                rid: deepcopy(peer["collection_guards"])
                for rid, peer in peers.items() if "collection_guards" in peer
            },
            "collection_effects": {
                rid: deepcopy(peer["collection_effects"])
                for rid, peer in peers.items() if "collection_effects" in peer
            },
        })
    return rows


def _table(
    columns: list[tuple[str, str]], rows: list[dict], row_key: str, page_size: int = 8
) -> None:
    ui.table(
        columns=[
            {
                "name": key,
                "field": key,
                "label": label,
                "align": "left",
                "style": "white-space: pre-wrap; overflow-wrap: anywhere; max-width: 32rem",
            }
            for key, label in columns
        ],
        rows=rows,
        row_key=row_key,
        pagination=page_size,
    ).classes("w-full").props("dense flat bordered wrap-cells")


def render_nominal_resource_des(bridge: SystemBridge) -> Callable[[], Awaitable[None]]:
    """Build stable resource controls and local capability graphs.

    Args:
        bridge: Existing public UI-to-runtime surface for read-only snapshots.

    Returns:
        The page's awaited snapshot refresh callback.
    """
    client = context.client
    configured: dict = {}
    configured_stamp: tuple | None = None
    models: dict = {}
    live: dict = {}
    revision: Any = object()
    busy = False
    initialized = False
    refresh_sections: list[Callable] = []
    selected_resource_id = ""
    product_part_choice: str | None = None
    expanded_sections: dict[tuple[str, ...], bool] = {}
    snapshot_reader = getattr(bridge, "get_environment_capabilities", lambda: {})
    revision_reader = getattr(bridge, "get_environment_capabilities_revision", None)

    with ui.card().classes("w-full"):
        ui.label("Resource capabilities").classes("text-lg font-semibold")
        ui.label("Configured capability graphs and completed task events.").classes(
            "text-sm text-slate-500"
        )
        loading = ui.label("Loading resource capabilities...")
        body = ui.column().classes("w-full")

    def lazy_section(
        label: str, builder: Callable, value: Callable, *,
        icon: str = "data_object", expanded: bool = False,
    ):
        section_key = (selected_resource_id, label)
        expansion = ui.expansion(
            label, icon=icon, value=expanded_sections.get(section_key, expanded)
        ).classes("w-full")
        content = ui.column().classes("w-full")
        content.move(expansion)
        previous: Any = object()

        def refresh() -> None:
            nonlocal previous
            if not expansion.value:
                return
            if any(
                isinstance(parent, ui.expansion) and not parent.value
                for parent in expansion.ancestors()
            ):
                return
            current = value()
            if current == previous:
                return
            previous = deepcopy(current)
            content.clear()
            with content:
                builder(current)

        def on_expansion_change() -> None:
            expanded_sections[section_key] = bool(expansion.value)
            refresh()

        expansion.on_value_change(on_expansion_change)
        refresh_sections.append(refresh)
        if expansion.value:
            refresh()
        return expansion

    def json_section(label: str, value: Callable) -> None:
        lazy_section(
            label,
            lambda data: ui.code(
                json.dumps(data, indent=2, ensure_ascii=False), language="json"
            ).classes("w-full"),
            value,
        )

    def build() -> None:
        with body:
            with ui.row().classes("w-full gap-2"):
                for resource_id in models:
                    ui.badge(resource_id, color="grey-7")
            selected = (
                ui.select(list(models), value="Conveyor", label="Resource")
                .classes("w-full")
                .props("outlined use-input")
            )

            @ui.refreshable
            def details() -> None:
                nonlocal selected_resource_id
                selected_resource_id = selected.value
                refresh_sections.clear()

                def model() -> dict:
                    return models[selected.value]

                ui.label(selected.value).classes("text-lg font-semibold mt-2")
                occupancy = ui.label().classes("text-sm")
                neighbors = ui.label().classes("text-sm")
                ui.label("Resource capability graph").classes("font-semibold")
                local_diagrams = ui.column().classes("w-full")

                def draw_local_diagrams() -> None:
                    local_diagrams.clear()
                    with local_diagrams:
                        for field in nominal_resource_default_fields(model()):
                            diagram = nominal_resource_state_diagram(
                                model(), field,
                                initial_valuation=configured[selected.value]["current_valuation"],
                            )
                            ui.label(field).classes("font-medium mt-2")
                            ui.mermaid(diagram["mermaid"]).classes("w-full overflow-auto")
                            _table(
                                [("id", "State"), ("value", "Exact value")],
                                diagram["state_rows"], "id", 10,
                            )

                draw_local_diagrams()
                marked_conditions = ui.label().classes("text-sm text-slate-600")
                ui.label(
                    "These are projections of local state variables. The event name labels "
                    "each arrow; the exact event_id and full parameter binding identify a "
                    "shared event variant. All participant and product conditions still apply."
                ).classes("text-sm text-slate-500")
                lazy_section(
                    "Configured initial values and marked state conditions",
                    lambda data: ui.code(
                        json.dumps(data, indent=2, ensure_ascii=False), language="json"
                    ).classes("w-full"),
                    lambda: {
                        "current_valuation": configured[selected.value]["current_valuation"],
                        "marked_state_conditions": model()["marked_state_conditions"],
                    },
                )
                ui.label("Product processPlan stages").classes("font-semibold")
                product_part_select = ui.select([], label="Part").classes("w-full")
                product_holder = ui.column().classes("w-full")
                product_current = ui.label().classes("text-sm text-slate-600")
                ui.label(
                    "This per-part processPlan projection shows exact ordered requirements. "
                    "Transport events leave these stages unchanged, while their resource and "
                    "custody conditions still apply. A final circle means this part's "
                    "processPlan is complete; run completion also requires the robots at home."
                ).classes("text-sm text-slate-500")
                if selected.value == "Conveyor":
                    ui.label(
                        "advance_conveyor updates all resident parts together and preserves "
                        "downstream order. These diagrams do not claim independent part "
                        "movement or validated physical spacing."
                    ).classes("text-sm text-slate-600")
                lazy_section(
                    "Functions and composed primitives",
                    render_resource_function_rows,
                    lambda: resource_function_rows(model()),
                    icon="precision_manufacturing",
                    expanded=True,
                )
                last_local_definition = deepcopy((
                    model()["events"], model()["state_variables"],
                    model()["marked_state_conditions"],
                    configured[selected.value]["current_valuation"],
                ))
                last_product_definition: Any = object()
                product_refreshing = False
                refresh_product_variants: Callable[[], None] | None = None

                def refresh_local_graph() -> None:
                    nonlocal last_local_definition
                    current = model()
                    definition = (
                        current["events"], current["state_variables"],
                        current["marked_state_conditions"],
                        configured[selected.value]["current_valuation"],
                    )
                    if definition != last_local_definition:
                        last_local_definition = deepcopy(definition)
                        draw_local_diagrams()

                def refresh_product_graph() -> None:
                    nonlocal last_product_definition, product_refreshing, product_part_choice
                    nonlocal refresh_product_variants
                    if product_refreshing:
                        return
                    product_refreshing = True
                    try:
                        requirements = live.get("requirements", {}) if "processPlan" in live else {}
                        parts = list(requirements)
                        preferred = product_part_choice if product_part_choice in parts else None
                        if preferred is None and parts:
                            current_part = live.get("environment_model", {}).get("part_name")
                            preferred = current_part if current_part in parts else parts[0]
                        if product_part_select.options != parts:
                            product_part_select.set_options(parts, value=preferred)
                        elif product_part_select.value != preferred:
                            product_part_select.set_value(preferred)
                        product_part_choice = preferred
                        product_part_select.set_visibility(bool(parts))
                        definition = (
                            live.get("run_id"), preferred,
                            deepcopy(requirements.get(preferred)) if preferred else None,
                            deepcopy(live.get("initial_product_states", {}).get(preferred))
                            if preferred else None,
                        )
                        if definition != last_product_definition:
                            last_product_definition = deepcopy(definition)
                            product_holder.clear()
                            refresh_product_variants = None
                            with product_holder:
                                if preferred is None:
                                    ui.label("No active processPlan.").classes("text-sm text-slate-500")
                                else:
                                    diagram = nominal_product_process_plan_diagram(
                                        requirements, live.get("initial_product_states", {}),
                                        preferred,
                                    )
                                    ui.mermaid(diagram["mermaid"]).classes("w-full overflow-auto")
                                    _table(
                                        [("id", "State"), ("completed", "processCompleted"),
                                         ("next_step", "Next step")],
                                        [
                                            {**row, "completed": _text(row["completed"]),
                                             "next_step": row["next_step"] or "complete"}
                                            for row in diagram["state_rows"]
                                        ],
                                        "id", 10,
                                    )
                                    initial = live.get("initial_product_states", {}).get(preferred, {})
                                    ui.label(
                                        "Configured initial processCompleted: "
                                        + _text(initial.get("processCompleted", []))
                                    ).classes("text-sm text-slate-600")
                                    variant_key = ("product", preferred, "event variants")
                                    variant_expansion = ui.expansion(
                                        "Declared event variants for these process steps",
                                        icon="data_object",
                                        value=expanded_sections.get(variant_key, False),
                                    ).classes("w-full")
                                    variant_content = ui.column().classes("w-full")
                                    variant_content.move(variant_expansion)
                                    previous_variants: list[dict] | None = None

                                    def show_variants() -> None:
                                        nonlocal previous_variants
                                        expanded_sections[variant_key] = bool(variant_expansion.value)
                                        if not variant_expansion.value:
                                            return
                                        rows = nominal_product_process_event_rows(
                                            models, requirements, preferred
                                        )
                                        if rows == previous_variants:
                                            return
                                        previous_variants = deepcopy(rows)
                                        variant_content.clear()
                                        with variant_content:
                                            ui.label(
                                                "Declared variants are potential matches; "
                                                "current guards and execution evidence still apply."
                                            ).classes("text-sm text-slate-500")
                                            ui.code(
                                                json.dumps(rows, indent=2, ensure_ascii=False),
                                                language="json",
                                            ).classes("w-full")

                                    variant_expansion.on_value_change(show_variants)
                                    refresh_product_variants = show_variants
                                    show_variants()
                        if refresh_product_variants is not None:
                            refresh_product_variants()
                        if preferred is None:
                            product_current.text = ""
                        else:
                            current = live.get("product_states", {}).get(preferred, {})
                            product_current.text = (
                                "Current processCompleted: "
                                + _text(current.get("processCompleted", []))
                            )
                    finally:
                        product_refreshing = False

                def choose_product_part() -> None:
                    nonlocal product_part_choice
                    product_part_choice = product_part_select.value
                    refresh_product_graph()

                product_part_select.on_value_change(choose_product_part)

                def refresh_summary() -> None:
                    current = model()
                    occupancy.text = "Occupancy: " + current.get(
                        "state_evidence", "configured initial assumptions"
                    )
                    neighbors.text = "Connected neighbors: " + ", ".join(
                        current.get("neighbors", [])
                    )
                    marked_conditions.text = "Marked state conditions: " + json.dumps(
                        current["marked_state_conditions"], ensure_ascii=False
                    )
                    refresh_local_graph()
                    refresh_product_graph()

                refresh_summary()
                refresh_sections.append(refresh_summary)
                json_section(
                    "Current configuration and process capabilities",
                    lambda: {
                        "current_configuration": model().get("current_configuration", {}),
                        "process_capabilities": model().get("process_capabilities", {}),
                        "executable_tasks": model().get("executable_tasks", []),
                        "execution_support": model()["execution_support"],
                    },
                )
                exploration_status = ui.label().classes("text-sm")
                activity_status = ui.label().classes("text-sm")
                component_status = ui.label().classes("text-sm")
                timing_status = ui.label().classes("text-sm text-slate-600")

                def refresh_exploration_status() -> None:
                    outcome = live.get("outcome", {})
                    exploration_status.text = str(
                        outcome.get("status", "No active environmental exploration")
                    ) + (": " + outcome["reason"] if outcome.get("reason") else "")
                    activity = live.get("resource_activity", {}).get(selected.value, {})
                    progress = activity.get("progress") or {}
                    countdown = ""
                    if "simulation_remaining_sec" in progress:
                        countdown = (
                            f" · {float(progress['simulation_remaining_sec']):.1f} simulation s remaining"
                        )
                    activity_status.text = (
                        f"Activity: {activity.get('status', 'idle')}{countdown}"
                    )
                    completed = live.get("component_progress", {})
                    component_status.text = (
                        f"Completed components: {completed.get('completed', 0)}/"
                        f"{completed.get('total', 0)}"
                    )
                    timings = live.get("timings", {})
                    timing_status.text = " · ".join(
                        f"{label} {float(timings.get(field, 0.0)):.2f}s"
                        for field, label in (
                            ("planning_sec", "planning"),
                            ("motion_sec", "motion"),
                            ("waiting_sec", "waiting"),
                            ("simulation_sec", "simulation"),
                            ("wall_clock_sec", "wall"),
                        )
                    )

                refresh_sections.append(refresh_exploration_status)
                refresh_exploration_status()

                def exploration(data: dict) -> None:
                    environment = data.get("environment_model", {})
                    outcome = data.get("outcome", {})
                    ui.label(
                        f"{environment.get('part_name', '')}: {_text(environment.get('desired_property', {}))}"
                    )
                    ui.mermaid(environment_capability_mermaid(environment)).classes(
                        "w-full overflow-auto"
                    )
                    conditions = [
                        *environment.get("rejections", []),
                        *outcome.get("unresolved", []),
                    ]
                    conditions.extend(
                        {
                            "resource_id": row["resource_id"],
                            "reason": f"{row['event_name']}: controller and completion validator unavailable",
                        }
                        for row in outcome.get("execution_unavailable", [])
                    )
                    reasons = sorted({(row["resource_id"], row["reason"]) for row in conditions})
                    _table(
                        [("resource_id", "Resource"), ("reason", "Offer conditions")],
                        [
                            {"id": index, "resource_id": rid, "reason": reason}
                            for index, (rid, reason) in enumerate(reasons)
                        ],
                        "id",
                    )

                lazy_section(
                    "Current environmental exploration",
                    exploration,
                    lambda: {
                        "environment_model": live.get("environment_model", {}),
                        "outcome": live.get("outcome", {}),
                    },
                    icon="route",
                )
                json_section(
                    "Environmental replies and selected path JSON",
                    lambda: live.get("environment_model", {}),
                )
                lazy_section(
                    "Capability tasks",
                    lambda rows: _table(
                        [
                            ("resource_id", "Resource performing task"),
                            ("signature", "Task"),
                            ("source", "Local conditions"),
                            ("target", "Local effects"),
                        ],
                        rows,
                        "id",
                    ),
                    lambda: nominal_capability_rows(model(), models),
                    icon="route",
                )

                def detailed_capability_graph(data: dict) -> None:
                    endpoints = nominal_resource_capability_diagram(model())
                    ui.label("Declared capability endpoints").classes("font-medium")
                    ui.mermaid(endpoints["mermaid"]).classes("w-full overflow-auto")
                    _table(
                        [("id", "State"), ("value", "Exact endpoint")],
                        endpoints["state_rows"], "id", 10,
                    )
                    ui.label("Guarded local graph").classes("font-medium")
                    ui.mermaid(_capability_graph_mermaid(data)).classes(
                        "w-full overflow-auto"
                    )
                    ui.code(
                        json.dumps(data, indent=2, ensure_ascii=False), language="json"
                    ).classes("w-full")
                    ui.label("Shared event variants").classes("font-medium")
                    ui.code(
                        json.dumps(
                            nominal_composition_event_rows(models, selected.value),
                            indent=2, ensure_ascii=False,
                        ),
                        language="json",
                    ).classes("w-full")

                lazy_section(
                    "Local capability graph and event details",
                    detailed_capability_graph,
                    lambda: nominal_capability_graph(model(), models),
                )
                lazy_section(
                    "Inventory and occupancy",
                    lambda rows: _table(
                        [("part_name", "part_name"), ("initial", "Current contents")],
                        rows,
                        "part_name",
                    ),
                    lambda: nominal_inventory_rows(model()),
                    icon="inventory_2",
                )
                ui.label(
                    "These identities identify current contents. Resource eligibility is evaluated from capabilities and conditions."
                ).classes("text-sm text-slate-500")

                des_key = (selected.value, "DES details")
                with ui.expansion(
                    "DES details", icon="account_tree",
                    value=expanded_sections.get(des_key, False),
                ).classes("w-full") as des:
                    ui.label(model()["execution_support"]).classes("text-sm text-amber-800")
                    for note in model()["notes"]:
                        ui.label(note).classes("text-sm text-slate-600")
                    json_section(
                        "Assignments and scene configuration",
                        lambda: {
                            key: value
                            for key, value in model()["assignments"].items()
                            if key
                            not in {
                                "nominal_parts",
                                "slots",
                                "supported_products",
                                "initial_products",
                            }
                        },
                    )
                    fields = list(model()["state_variables"])
                    default = "resource_state" if "resource_state" in fields else fields[0]
                    diagram_field = ui.select(
                        fields, value=default, label="State variable for diagram"
                    ).classes("w-full")
                    diagram_holder = ui.column().classes("w-full")
                    diagram = None

                    def refresh_diagram() -> None:
                        nonlocal diagram
                        if not des.value:
                            return
                        source = nominal_des_mermaid(
                            model(), diagram_field.value,
                            initial_valuation=configured[selected.value]["current_valuation"],
                        )
                        if diagram is None:
                            with diagram_holder:
                                diagram = ui.mermaid(source).classes("w-full overflow-auto")
                        elif diagram.content != source:
                            diagram.set_content(source)

                    refresh_sections.append(refresh_diagram)

                    def on_des_change() -> None:
                        expanded_sections[des_key] = bool(des.value)
                        refresh_diagram()

                    des.on_value_change(on_des_change)
                    diagram_field.on_value_change(refresh_diagram)
                    ui.label(
                        "This diagram projects one state variable. All event guards still "
                        "apply. Parameter and collection outcomes remain symbolic."
                    ).classes("text-xs text-slate-500")
                    evidence = ui.label().classes("text-sm text-slate-500")

                    def refresh_evidence() -> None:
                        evidence.text = (
                            "Configured / assumed initial values — not live observations."
                            if not live
                            else "Acknowledged resource state; evidence: "
                            + model().get("state_evidence", "configured initial assumptions")
                        )

                    refresh_evidence()
                    refresh_sections.append(refresh_evidence)
                    lazy_section(
                        "State variables and values",
                        lambda rows: _table(
                            [
                                ("field", "State variable"),
                                ("scope", "Scope"),
                                ("domain", "Domain"),
                                ("initial", "Current value"),
                            ],
                            rows,
                            "field",
                        ),
                        lambda: nominal_state_rows(model()),
                    )
                    json_section(
                        "Marked state conditions", lambda: model()["marked_state_conditions"]
                    )
                    event_names = model()["local_event_alphabet"]
                    event_select = ui.select(
                        event_names, value=event_names[0], label="Nominal event"
                    ).classes("w-full")
                    json_section(
                        "Complete process JSON", lambda: process_json(models, event_select.value)
                    )
                    lazy_section(
                        "Event conditions",
                        lambda rows: _table(
                            [
                                ("event", "Event"),
                                ("bindings", "Parameters and completion inputs"),
                                ("guards", "Guards for all participants"),
                                ("updates", "Updates for all participants"),
                                ("notes", "Conditions"),
                                ("controllable", "Controllable"),
                                ("observable", "Observable"),
                            ],
                            rows,
                            "id",
                            4,
                        ),
                        lambda: nominal_event_rows(models, selected.value, event_select.value),
                    )
                    event_select.on_value_change(
                        lambda: [refresh() for refresh in refresh_sections]
                    )

            details()
            selected.on_value_change(details.refresh)

    async def refresh_runtime() -> None:
        nonlocal configured, configured_stamp, models, live, revision, busy, initialized
        if busy or getattr(client, "_deleted", False):
            return
        busy = True
        try:
            stat = await asyncio.to_thread(SCENE_PATH.stat)
            stamp = (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
            configuration_changed = stamp != configured_stamp
            if not configured or configuration_changed:
                configured = await asyncio.to_thread(
                    lambda: build_environment_models(bridge.load_config(str(SCENE_PATH)))
                )
                configured_stamp = stamp
            current = await asyncio.to_thread(revision_reader) if revision_reader else None
            if (
                initialized
                and not configuration_changed
                and revision_reader
                and current == revision
            ):
                return
            updated = await asyncio.to_thread(snapshot_reader)
            if getattr(client, "_deleted", False):
                return
            revision = current
            if initialized and not configuration_changed and updated == live:
                return
            live = updated
            models = live.get("models") or configured
            loading.set_visibility(False)
            if not initialized:
                build()
                initialized = True
            else:
                for refresh in tuple(refresh_sections):
                    refresh()
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Nominal resource DES unavailable: %s", exc)
            if not initialized:
                body.clear()
                refresh_sections.clear()
            loading.text = f"Nominal resource DES unavailable: {exc}"
            loading.set_visibility(True)
        finally:
            busy = False

    return refresh_runtime
