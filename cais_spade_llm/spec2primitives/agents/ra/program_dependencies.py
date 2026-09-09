from __future__ import annotations

"""Trace selected data dependencies and route evidence needs without repairing steps."""

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any

from .parameter_binding import assess_parameter_bindings


def selected_references(value: Any, path: str = "") -> list[tuple[str, str, Mapping[str, Any]]]:
    """Return only references explicitly present in a submitted parameter tree."""
    if isinstance(value, Mapping):
        if set(value) in ({"result_ref"}, {"value_ref"}):
            kind = next(iter(value))
            return [(path, kind, value[kind])]
        return [
            ref
            for key, item in value.items()
            for ref in selected_references(
                item, path + "/" + key.replace("~", "~0").replace("/", "~1")
            )
        ]
    if isinstance(value, list):
        return [
            ref
            for index, item in enumerate(value)
            for ref in selected_references(item, path + "/" + str(index))
        ]
    return []


def assess_program_dependencies(
    steps: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Mapping[str, Any]],
    robot_state: Mapping[str, Any],
    *,
    read_evidence: Callable[[str, str], Any],
    result_schema: Callable[[dict[str, Any]], Any],
) -> dict[str, Any]:
    """Route unresolved selected measurements and trace their dependent outputs."""
    issues = assess_parameter_bindings(
        steps, catalog, robot_state, read_evidence=read_evidence, result_schema=result_schema
    )
    # A numeric intermediate control target is an RA proposal for motion checks;
    # it is not a claim that a product measurement was observed at that location.
    issues = [
        item
        for item in issues
        if not (
            steps[item["step_index"] - 1]["primitive_symbol"] == "move_cartesian"
            and item["status"] == "unverified"
            and item["parameter_path"] in {"/x", "/y", "/z", "/qx", "/qy", "/qz", "/qw"}
        )
    ]
    blocked: dict[int, set[int]] = {}
    needs = []
    dependencies = []
    for index, step in enumerate(steps, start=1):
        direct = [
            item for item in issues if item["step_index"] == index and item["status"] != "deferred"
        ]
        causes = {index} if direct else set()
        refs = selected_references(step["params"])
        for path, kind, reference in refs:
            if kind != "result_ref":
                continue
            producer = reference["step_index"]
            dependencies.append(
                {"step_index": index, "parameter_path": path, "producer_step": producer}
            )
            upstream = blocked.get(producer, set())
            causes.update(upstream)
            if upstream:
                issues.append(
                    {
                        "step_index": index,
                        "parameter_path": path,
                        "status": "blocked",
                        "blocked_by": sorted(upstream),
                        "message": f"Selected output from step {producer} is blocked by unresolved inputs at steps {sorted(upstream)}.",
                    }
                )
        blocked[index] = causes
        for issue in direct:
            path = issue["parameter_path"]
            fields = [
                name.replace("~1", "/").replace("~0", "~") for name in path.split("/")[1:]
            ]
            product_input = fields[0] in {"product_geometry", "target_pose", "detected_parts"}
            if issue["status"] != "missing" and not (
                product_input and issue["status"] in {"incompatible", "unverified"}
            ):
                continue
            schema = catalog[step["primitive_symbol"]]["parameter_schemas"].get(fields[0], {})
            for name in fields[1:]:
                schema = (
                    schema.get("items", {})
                    if schema.get("type") == "array"
                    else schema.get("properties", {}).get(name, {})
                )
            route = "PA" if product_input else "RA"
            # A populated but rejected measurement is still an unresolved input.
            # PA can investigate evidence; only RA can replace the rejected binding.
            needs.append(
                {
                    "step_index": index,
                    "parameter_path": issue["parameter_path"],
                    "authority": route,
                    "quantity": issue["parameter_path"],
                    "primitive_symbol": step["primitive_symbol"],
                    "quantity_schema": deepcopy(schema),
                    "reason": issue["message"],
                    "evidence_refs": [
                        deepcopy(ref) for _, kind, ref in refs if kind == "value_ref"
                    ],
                }
            )
    return {"issues": issues, "dependencies": dependencies, "context_requests": needs}
