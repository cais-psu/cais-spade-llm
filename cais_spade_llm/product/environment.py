"""Product-owned requirements, environmental bids, and acknowledged state."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any
from uuid import uuid4

from cais_spade_llm.product.order import validate_product_order
from cais_spade_llm.resources.environment_models import (
    build_environment_models,
    candidates,
    feasibility,
    matches_requirement,
    owners,
    project_transition,
)


def fingerprint(value: Any) -> str:
    """Fingerprint exact JSON values for state, configuration, and correlation."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


class EnvironmentResourceContext:
    """Own a resource's model, observed valuation, and execution adapters."""

    def __init__(self, model: dict) -> None:
        self.resource_id = model["resource_id"]
        self.model = deepcopy(model)
        self.valuation = deepcopy(model["current_valuation"])
        self.executors: dict = {}
        self.completion_validators: dict = {}
        self.start_validators: dict = {}
        self.validated_completions: dict = {}
        self.completion_observations: dict = {}
        self.revision = 0
        self.evidence = "configured initial assumptions"
        self.visited: dict[tuple[str, str], tuple] = {}

    def snapshot(self) -> dict:
        """Return acknowledged values without exposing mutable state."""
        return deepcopy(self.valuation)

    def descriptor(self) -> dict:
        """Return the actual runtime model and separately identified occupancy."""
        return {
            **deepcopy(self.model),
            "current_valuation": self.snapshot(),
            "revision": self.revision,
            "state_evidence": self.evidence,
            "executable_tasks": sorted(self.executors),
        }

    def bind_executor(
        self, event_name: str, execute, validate_completion, *, validate_start
    ) -> None:
        """Bind a controller and its evidence validator, never just an availability flag."""
        if event_name not in self.model["local_event_alphabet"]:
            raise ValueError("Executor does not correspond to a declared task")
        if not all(
            callable(callback) for callback in (execute, validate_start, validate_completion)
        ):
            raise ValueError(
                "Execution requires a controller, a start validator, and a completion validator"
            )
        self.executors[event_name] = execute
        self.completion_validators[event_name] = validate_completion
        self.start_validators[event_name] = validate_start

    def alternatives(self, context, valuation: dict, products: dict, part: str, desired: dict):
        """Evaluate locally owned transitions against a candidate path's state."""
        for task in candidates(
            self.model, valuation, part, desired, context.requirements.get(part, [])
        ):
            try:
                after, predicted = project_transition(
                    context.models,
                    valuation,
                    products,
                    task,
                    context.product_name,
                    context.requirements,
                )
            except ValueError:
                continue
            status, reasons = feasibility(self.model, task, context.geometry.get(part, {}))
            if status == "INFEASIBLE":
                yield {"task": task, "status": status, "reasons": reasons}
                continue
            yield {
                "task": task,
                "status": status,
                "reasons": reasons,
                "valuation": after,
                "products": predicted,
                "executable": task["event_name"] in self.executors,
            }


class EnvironmentProductContext:
    """Coordinate one order without storing a preassigned production route."""

    def __init__(
        self,
        scene: dict,
        product_order: dict,
        geometry: dict,
        permitted_resources: list[str] | None = None,
    ) -> None:
        validated = validate_product_order(
            product_order, geometry, require_process_requirements=True
        )
        if validated.payload["quantity"] != 1:
            raise ValueError("Registered workpiece identities currently support quantity: 1")
        self.product_order = deepcopy(validated.payload)
        self.schema_version = 3 if "processPlan" in self.product_order else 2
        self.selected_parts = validated.selected_parts
        self.product_name = product_order["product"]
        if self.product_name != scene["Exit"]["completed_product"]:
            raise ValueError("Product does not match the configured assembly endpoint")
        field = "processPlan" if self.schema_version == 3 else "requirements"
        self.requirements = {
            part: deepcopy(self.product_order[field][part]) for part in self.selected_parts
        }
        if self.schema_version == 3:
            for part, steps in self.requirements.items():
                for step in steps:
                    for requirement in step["processesToComplete"]:
                        if requirement["process"] == "assembly":
                            requirement["target"] = geometry["parts"]["assembly_target_map"][part]
        self.inputs = deepcopy(
            {"scene": scene, "product_order": product_order, "geometry": geometry}
        )
        names = list(
            dict.fromkeys(
                [
                    *geometry["assembly_board"]["slots"],
                    *scene["Storage"]["slots"],
                    *scene["3D Printing Station"]["initial_products"],
                    self.product_name,
                ]
            )
        )
        self.models = build_environment_models(scene, names, schema_version=self.schema_version)
        self.resources = {
            rid: EnvironmentResourceContext(model) for rid, model in self.models.items()
        }
        self.models = {rid: resource.model for rid, resource in self.resources.items()}
        self.permitted_resources = (
            list(self.models) if permitted_resources is None else list(permitted_resources)
        )
        if len(set(self.permitted_resources)) != len(self.permitted_resources) or any(
            rid not in self.models for rid in self.permitted_resources
        ):
            raise ValueError("Unknown or repeated permitted resource")
        part_data = geometry["parts"]
        self.geometry = {
            name: {
                "dimensions_m": deepcopy(part_data.get("dimensions_m", {}).get(name)),
                "target": part_data.get("assembly_target_map", {}).get(name),
                "mass_kg": part_data.get("mass_kg", {}).get(name),
                "model_name": part_data.get("model_map", {}).get(name),
            }
            for name in names
        }
        self.geometry[self.product_name]["dimensions_m"] = deepcopy(
            geometry["assembly_board"].get("dimensions_m")
        )
        self.part_tracker = {
            name: {"location": None, "state": "unknown", "last_task": None, "processCompleted": []}
            for name in names
        }
        for name in scene["Storage"]["slots"]:
            self.part_tracker[name].update(location="Storage", state="ready")
        for name in scene["3D Printing Station"]["initial_products"]:
            self.part_tracker[name].update(
                location="3D Printing Station",
                state="ready",
                processCompleted=[{"process": "print_part"}],
            )
        self.part_tracker[self.product_name].update(location=self.product_name, state="ready")
        initial_owners = owners(self.models, self.snapshot(), self.part_tracker, self.product_name)
        for part, owner in initial_owners.items():
            if self.part_tracker[part]["location"] is None:
                self.part_tracker[part].update(location=owner, state="ready")
        self.initial_product_states = deepcopy(self.part_tracker)
        self.run_id = uuid4().hex
        self.revision = 0
        self.pending: dict | None = None
        self.acknowledgements: dict[str, dict] = {}
        self.transitions: list[dict] = []
        self.explorations: list[dict] = []
        self.environment_model: dict = {"nodes": [], "edges": [], "bids": [], "rejections": []}

    def snapshot(self) -> dict:
        """Read one coherent resource valuation for discovery or commit."""
        return {rid: resource.snapshot() for rid, resource in self.resources.items()}

    def revisions(self) -> dict:
        """Include configuration changes in offer and acknowledgement validity."""
        return {
            rid: {
                "revision": resource.revision,
                "model": fingerprint(
                    [resource.model, resource.valuation, sorted(resource.executors)]
                ),
            }
            for rid, resource in self.resources.items()
        }

    def outstanding(self) -> tuple[str, dict] | None:
        """Return the next unmet ordered property, including final product delivery."""
        for part in self.selected_parts:
            for desired in self.requirements[part]:
                if not matches_requirement(self.part_tracker[part], desired):
                    return part, deepcopy(desired)
        desired = {"location": "Exit", "state": "completed"}
        if not matches_requirement(self.part_tracker[self.product_name], desired):
            return self.product_name, desired
        return None

    def contact_resource(self, part: str) -> str:
        """Identify the present custodian, including assembly and staging endpoints."""
        location = self.part_tracker[part]["location"]
        if location in self.resources:
            return location
        if location == self.product_name:
            return "Exit"
        for rid in self.resources:
            if location == f"{rid} staging tray":
                return rid
        if location is None:
            return (
                self.inputs["geometry"]["parts"]
                .get("initial_source_resource_map", {})
                .get(part, "Storage")
            )
        raise ValueError("Current product location has no configured ResourceAgent")

    def request(self, deadline: float) -> dict | None:
        """Start a correlated exploration from acknowledged state, without effects."""
        outstanding = self.outstanding()
        if outstanding is None:
            return None
        part, desired = outstanding
        request_id = uuid4().hex
        self.environment_model = {
            "request_id": request_id,
            "part_name": part,
            "desired_property": desired,
            "initial_state_id": fingerprint([self.snapshot(), self.part_tracker]),
            "nodes": [],
            "edges": [],
            "bids": [],
            "rejections": [],
            "status": "exploring",
        }
        for resource in self.resources.values():
            resource.visited.clear()
        return {
            "run_id": self.run_id,
            "request_id": request_id,
            "branch_id": uuid4().hex,
            "revision": self.revision,
            "resource_revisions": self.revisions(),
            "part_name": part,
            "desired_property": desired,
            "geometry": deepcopy(self.geometry[part]),
            "deadline": deadline,
            "valuation": self.snapshot(),
            "products": deepcopy(self.part_tracker),
            "path": [],
            "unresolved": [],
        }

    def current_request(self, request: dict) -> bool:
        """Reject changed product or resource contexts before using an offer."""
        return (
            request.get("run_id") == self.run_id
            and request.get("revision") == self.revision
            and request.get("resource_revisions") == self.revisions()
            and request.get("request_id") == self.environment_model.get("request_id")
            and not self.environment_model.get("closed", False)
        )

    def merge_reply(self, reply: dict) -> None:
        """Merge resource-owned transitions and bids into the current environment."""
        for transition in reply.get("transitions", []):
            for endpoint in ("source", "target"):
                state = {"id": transition[f"{endpoint}_id"], "product_state": transition[endpoint]}
                if state not in self.environment_model["nodes"]:
                    self.environment_model["nodes"].append(state)
            if transition not in self.environment_model["edges"]:
                self.environment_model["edges"].append(deepcopy(transition))
        for bid in reply.get("bids", []):
            try:
                checked = self._check_bid(bid)
            except (ValueError, KeyError, TypeError) as exc:
                self.environment_model["rejections"].append(
                    {"resource_id": reply["resource_id"], "reason": str(exc)}
                )
            else:
                self.environment_model["bids"].append(checked)
        self.environment_model["rejections"].extend(deepcopy(reply.get("rejections", [])))

    def _check_bid(self, bid: dict) -> dict:
        valuation, products = self.snapshot(), deepcopy(self.part_tracker)
        unresolved = []
        execution_unavailable = []
        part = self.environment_model["part_name"]
        for task in bid["path"]:
            if task["resource_id"] not in self.permitted_resources:
                raise ValueError("Bid uses an excluded resource")
            if task["parameters"].get("part_name", part) != part:
                raise ValueError("Bid contains another workpiece")
            actor = self.resources[task["resource_id"]]
            if task["event_name"] not in actor.executors:
                execution_unavailable.append(
                    {"resource_id": actor.resource_id, "event_name": task["event_name"]}
                )
            status, reasons = feasibility(actor.model, task, self.geometry[part])
            if status == "INFEASIBLE":
                raise ValueError("; ".join(reasons))
            unresolved.extend(
                {"resource_id": actor.resource_id, "reason": reason} for reason in reasons
            )
            valuation, products = project_transition(
                self.models, valuation, products, task, self.product_name, self.requirements
            )
        if not matches_requirement(products[part], self.environment_model["desired_property"]):
            raise ValueError("Returned path does not establish the requested property")
        return {
            "path": deepcopy(bid["path"]),
            "unresolved": unresolved,
            "execution_unavailable": execution_unavailable,
        }

    def prepare(self, task: dict, *, simulated: bool = False) -> dict:
        """Recheck a selected transition and bind it to a single pending execution."""
        if self.pending is not None:
            raise ValueError("A task is already awaiting acknowledgement")
        rid = task["resource_id"]
        if rid not in self.permitted_resources:
            raise ValueError("Resource is excluded from this order")
        actor = self.resources[rid]
        part = task["parameters"].get("part_name") or task.get("part_name")
        if part is None:
            outstanding = self.outstanding()
            part = outstanding[0] if outstanding else self.product_name
        status, reasons = feasibility(actor.model, task, self.geometry.get(part, {}))
        if status != "FEASIBLE":
            raise ValueError("; ".join(reasons))
        if not simulated and task["event_name"] not in actor.executors:
            raise ValueError("No execution adapter and completion evidence validator for this task")
        project_transition(
            self.models,
            self.snapshot(),
            self.part_tracker,
            task,
            self.product_name,
            self.requirements,
        )
        self.pending = {
            **deepcopy(task),
            "task_id": f"environment_{self.revision + 1}",
            "run_id": self.run_id,
            "revision": self.revision,
            "resource_revisions": self.revisions(),
            "evidence": "simulated" if simulated else "resource",
        }
        return deepcopy(self.pending)

    def acknowledge(self, acknowledgement: dict) -> bool:
        """Commit exactly matching acknowledged effects once, atomically."""
        task_id = acknowledgement.get("task_id")
        if task_id in self.acknowledgements:
            if fingerprint(acknowledgement) != fingerprint(self.acknowledgements[task_id]):
                raise ValueError("Conflicting repeated acknowledgement")
            return False
        if self.pending is None:
            raise ValueError("No pending task")
        expected = {**self.pending, "status": "completed"}
        if (
            fingerprint(expected) != fingerprint(acknowledgement)
            or self.pending["resource_revisions"] != self.revisions()
        ):
            raise ValueError("Acknowledgement does not match pending task or current configuration")
        actor = self.resources[self.pending["resource_id"]]
        if (
            self.pending["evidence"] == "resource"
            and actor.validated_completions.get(task_id) != expected
        ):
            raise ValueError("Resource has not validated completion evidence")
        before = self.snapshot()
        after, products = project_transition(
            self.models,
            before,
            self.part_tracker,
            self.pending,
            self.product_name,
            self.requirements,
        )
        for part, state in products.items():
            if state != self.part_tracker[part]:
                state["last_task"] = task_id
        record = {
            "acknowledgement": deepcopy(acknowledgement),
            "before": before,
            "after": after,
            "product_before": deepcopy(self.part_tracker),
            "product_after": deepcopy(products),
        }
        if self.pending["evidence"] == "resource":
            record["observations"] = deepcopy(actor.completion_observations[task_id])
        for rid, values in after.items():
            if values != self.resources[rid].valuation:
                self.resources[rid].revision += 1
                self.resources[rid].valuation = values
                self.resources[rid].evidence = self.pending["evidence"]
        self.part_tracker = products
        self.revision += 1
        self.transitions.append(record)
        self.acknowledgements[task_id] = deepcopy(acknowledgement)
        self.pending = None
        return True

    def report(self) -> dict:
        """Save the order's semantics without reinterpreting historical process facts."""
        return deepcopy(
            {
                "schema_version": self.schema_version,
                "run_id": self.run_id,
                "inputs": self.inputs,
                "permitted_resources": self.permitted_resources,
                "models": self.models,
                "initial_product_states": self.initial_product_states,
                "explorations": self.explorations,
                "environment_model": self.environment_model,
                "transitions": self.transitions,
                "final_valuation": self.snapshot(),
                "final_product_states": self.part_tracker,
                "revision": self.revision,
            }
        )


def select_environment_path(bids: list[dict]) -> dict:
    """Prefer executable paths, then fewest tasks and exact resource/event ordering."""
    if not bids:
        return {"status": "blocked", "tasks": []}
    best = min(
        bids,
        key=lambda bid: (
            bool(bid["unresolved"]),
            bool(bid.get("execution_unavailable")),
            len(bid["path"]),
            tuple((t["resource_id"], t["event_id"]) for t in bid["path"]),
        ),
    )
    return {
        "status": "needs_context" if best["unresolved"] else "planned",
        "tasks": deepcopy(best["path"]),
        "unresolved": deepcopy(best["unresolved"]),
        "executable": not best["unresolved"] and not best.get("execution_unavailable"),
        "execution_unavailable": deepcopy(best.get("execution_unavailable", [])),
    }


def verify_environment_run(report: dict) -> None:
    """Replay versioned simulated acknowledgements and reject altered saved state."""
    if report.get("schema_version") not in {2, 3}:
        raise ValueError("Environmental runs require schema_version 2 or 3; v1 remains historical")
    context = EnvironmentProductContext(
        **report["inputs"], permitted_resources=report["permitted_resources"]
    )
    context.run_id = report["run_id"]
    if context.schema_version != report["schema_version"]:
        raise ValueError("Saved schema_version differs from the order's process semantics")
    if (
        context.models != report["models"]
        or context.initial_product_states != report["initial_product_states"]
    ):
        raise ValueError("Environment model or initial state differs from saved inputs")
    for record in report["transitions"]:
        ack = record["acknowledgement"]
        if ack["evidence"] != "simulated":
            raise ValueError(
                "Controller evidence requires its resource validator; cannot replay as simulated"
            )
        task = {key: ack[key] for key in ("resource_id", "event_id", "event_name", "parameters")}
        context.prepare(task, simulated=True)
        context.acknowledge(ack)
        if context.transitions[-1] != record:
            raise ValueError("Saved transition differs from acknowledged effects")
    if (
        context.snapshot() != report["final_valuation"]
        or context.part_tracker != report["final_product_states"]
        or context.revision != report["revision"]
    ):
        raise ValueError("Saved final state differs from acknowledgement history")
