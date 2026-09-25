"""Product-owned requirements, environmental bids, and acknowledged state."""

from __future__ import annotations

import hashlib
import json
from copy import copy, deepcopy
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
            if not context.allows_task(task):
                continue
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
    """Coordinate an order through resource capabilities and optional machine binding."""

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
        self.machine_resource = self.product_order.get("machine_resource")
        self.machine_ids = {row["resource_id"] for row in scene["machines"]}
        if self.machine_resource is not None and self.machine_resource not in self.machine_ids:
            raise ValueError("machine_resource is not a configured machine")
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
        configured_scene = deepcopy(scene)
        if self.machine_resource is not None:
            part = self.selected_parts[0]
            trim_results = [
                requirement["result"]
                for step in self.requirements[part]
                for requirement in step["processesToComplete"]
                if requirement["process"] == "trim"
            ]
            if len(trim_results) != 1:
                raise ValueError("machine_resource requires one trim result")
            machine = next(
                row for row in configured_scene["machines"]
                if row["resource_id"] == self.machine_resource
            )
            effect = {"process": "trim", "result": trim_results[0]}
            current = machine["current_configuration"]
            if effect not in (current.get("program") or {}).get("effects", []):
                option = machine.get("program_options", {}).get(trim_results[0])
                if not isinstance(option, dict):
                    raise ValueError("Bound machine has no program for the required trim result")
                machine["current_configuration"] = deepcopy(option)
            selected = machine["current_configuration"]
            program = selected.get("program") or {}
            if (
                trim_results[0] not in machine["process_capabilities"]["trim"]["supported_results"]
                or not selected.get("tool")
                or not selected.get("workholding")
                or program.get("validated") is not True
                or effect not in program.get("effects", [])
                or any(
                    selected.get(field) != value
                    for field, value in program.get("required_configuration", {}).items()
                )
            ):
                raise ValueError("Bound machine lacks a validated program for the required trim result")
        self.inputs = deepcopy(
            {"scene": configured_scene, "product_order": product_order, "geometry": geometry}
        )
        names = list(
            dict.fromkeys(
                [
                    *geometry["assembly_board"]["slots"],
                    *configured_scene["Storage"]["slots"],
                    *configured_scene["3D Printing Station"]["initial_products"],
                    self.product_name,
                ]
            )
        )
        self.models = build_environment_models(
            configured_scene, names, schema_version=self.schema_version
        )
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
        self.geometry[self.product_name]["model_name"] = geometry["assembly_board"].get(
            "model_name"
        )
        self.geometry[self.product_name]["mass_kg"] = geometry["assembly_board"].get(
            "mass_kg"
        )
        self.geometry[self.product_name]["grasp_pose"] = deepcopy(
            geometry["assembly_board"].get("grasp_pose")
        )
        self.part_tracker = {
            name: {"location": None, "state": "unknown", "last_task": None, "processCompleted": []}
            for name in names
        }
        for name in configured_scene["Storage"]["slots"]:
            self.part_tracker[name].update(location="Storage", state="ready")
        for name in configured_scene["3D Printing Station"]["initial_products"]:
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
        self._task_sequence = 0
        self.pending_tasks: dict[str, dict] = {}
        self.reservations: dict[str, str] = {}
        self.acknowledgements: dict[str, dict] = {}
        self.transitions: list[dict] = []
        self.explorations: list[dict] = []
        self.environment_model: dict = {"nodes": [], "edges": [], "bids": [], "rejections": []}
        self.exploration_models: dict[str, dict] = {}
        self.negotiations: list[dict] = []

    @property
    def pending(self) -> dict | None:
        """Return the sole pending task for compatibility with serial callers."""
        if not self.pending_tasks:
            return None
        if len(self.pending_tasks) == 1:
            return deepcopy(next(iter(self.pending_tasks.values())))
        return {task_id: deepcopy(task) for task_id, task in self.pending_tasks.items()}

    def pending_for(self, task_id: str | None) -> dict | None:
        """Return one prepared task by identity."""
        task = self.pending_tasks.get(str(task_id or ""))
        return deepcopy(task) if task is not None else None

    def snapshot(self) -> dict:
        """Read one coherent resource valuation for discovery or commit."""
        return {rid: resource.snapshot() for rid, resource in self.resources.items()}

    def calculation_snapshot(self) -> EnvironmentProductContext:
        """Detach calculation inputs while reusing immutable revision snapshots."""
        self.revisions()
        snapshot = copy(self)
        snapshot.resources = {}
        for rid, resource in self.resources.items():
            frozen = copy(resource)
            # revisions() replaces this detached copy whenever inputs change;
            # calculations only read its model and valuation.
            frozen.model, frozen.valuation, executors = resource._revision_inputs
            frozen.executors = frozenset(executors)
            frozen.visited = dict(resource.visited)
            snapshot.resources[rid] = frozen
        snapshot.models = {rid: resource.model for rid, resource in snapshot.resources.items()}
        snapshot.part_tracker = deepcopy(self.part_tracker)
        snapshot.geometry = deepcopy(self.geometry)
        snapshot.requirements = deepcopy(self.requirements)
        snapshot.permitted_resources = list(self.permitted_resources)
        snapshot.reservations = dict(self.reservations)
        snapshot.exploration_models = {}
        return snapshot

    def allows_task(self, task: dict) -> bool:
        """Keep a bound one-part order on its declared machine lane."""
        if self.machine_resource is None:
            return True
        references = (
            task["resource_id"],
            *(task["parameters"].get(field) for field in (
                "origin_resource_location", "destination_location",
                "source_resource", "target_resource",
            )),
        )
        return all(
            reference == self.machine_resource
            for reference in references if reference in self.machine_ids
        )

    def revisions(self, resources=None) -> dict:
        """Detect changed offer inputs without repeatedly serializing unchanged models."""
        result = {}
        for rid in self.resources if resources is None else resources:
            resource = self.resources[rid]
            inputs = [resource.model, resource.valuation, sorted(resource.executors)]
            if inputs != getattr(resource, "_revision_inputs", None):
                resource._revision_inputs = deepcopy(inputs)
                resource._revision_fingerprint = fingerprint(inputs)
            result[rid] = {"revision": resource.revision, "model": resource._revision_fingerprint}
        return result

    def outstanding(self) -> tuple[str, dict] | None:
        """Return the next unmet ordered component process property."""
        for part in self.selected_parts:
            for desired in self.requirements[part]:
                if not matches_requirement(self.part_tracker[part], desired):
                    return part, deepcopy(desired)
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

    def request(
        self, deadline: float, *, part: str | None = None, desired: dict | None = None,
        resource_goal: dict | None = None, candidate: dict | None = None,
    ) -> dict | None:
        """Start a correlated exploration from acknowledged state, without effects."""
        outstanding = (part, desired) if part is not None else self.outstanding()
        if outstanding is None:
            return None
        part, desired = outstanding
        if part not in self.part_tracker or not isinstance(desired, dict):
            raise ValueError("Capability request requires a registered part and requirement")
        request_id = uuid4().hex
        self.environment_model = {
            "request_id": request_id,
            "part_name": part,
            "desired_property": deepcopy(desired),
            "resource_goal": deepcopy(resource_goal),
            "candidate": deepcopy(candidate),
            "initial_state_id": fingerprint([self.snapshot(), self.part_tracker]),
            "nodes": [],
            "edges": [],
            "bids": [],
            "rejections": [],
            "status": "exploring",
            "dependencies": {},
            "observed_resources": {},
            "part_state": deepcopy(self.part_tracker[part]),
            "permitted_resources": list(self.permitted_resources),
        }
        self.exploration_models[request_id] = self.environment_model
        request = {
            "run_id": self.run_id,
            "request_id": request_id,
            "branch_id": uuid4().hex,
            "revision": self.revision,
            "resource_revisions": self.revisions(),
            "part_name": part,
            "desired_property": deepcopy(desired),
            "resource_goal": deepcopy(resource_goal),
            "candidate": deepcopy(candidate),
            "geometry": deepcopy(self.geometry[part]),
            "deadline": deadline,
            "valuation": self.snapshot(),
            "products": deepcopy(self.part_tracker),
            "path": [],
            "unresolved": [],
        }

        contact = (candidate["resource_id"] if candidate else
                   resource_goal["resource_id"] if resource_goal else self.contact_resource(part))
        self.request_dependencies(request, [contact])
        return request

    def request_dependencies(self, request: dict, resources: list[str]) -> None:
        """Track the acknowledged revisions read by one correlated discovery."""
        model = self.exploration_models.get(request.get("request_id"))
        if model is None:
            return
        for rid in resources:
            model["dependencies"][rid] = deepcopy(request["resource_revisions"][rid])

    def current_request(self, request: dict) -> bool:
        """Invalidate only discoveries whose observed inputs have changed."""
        model = self.exploration_models.get(request.get("request_id"), {})
        if (request.get("run_id") != self.run_id or not model or model.get("closed")
                or model["permitted_resources"] != self.permitted_resources):
            return False
        if not model.get("resource_goal") and model["part_state"] != self.part_tracker[model["part_name"]]:
            return False
        current = self.revisions(model["dependencies"])
        return all(current.get(rid) == revision for rid, revision in model["dependencies"].items())

    def merge_reply(self, reply: dict) -> None:
        """Merge resource-owned transitions and bids into the current environment."""
        model = self.exploration_models.get(reply.get("request_id"), self.environment_model)
        for transition in reply.get("transitions", []):
            for endpoint in ("source", "target"):
                state = {"id": transition[f"{endpoint}_id"], "product_state": transition[endpoint]}
                if state not in model["nodes"]:
                    model["nodes"].append(state)
            if transition not in model["edges"]:
                model["edges"].append(deepcopy(transition))
        for bid in reply.get("bids", []):
            try:
                checked = self._check_bid(bid, model=model)
            except (ValueError, KeyError, TypeError) as exc:
                model["rejections"].append(
                    {"resource_id": reply["resource_id"], "reason": str(exc)}
                )
            else:
                model["bids"].append(checked)
        model["rejections"].extend(deepcopy(reply.get("rejections", [])))

    def _check_bid(self, bid: dict, *, model: dict | None = None) -> dict:
        model = self.environment_model if model is None else model
        valuation, products = self.snapshot(), deepcopy(self.part_tracker)
        unresolved = []
        execution_unavailable = []
        part = model["part_name"]
        for task in bid["path"]:
            if task["resource_id"] not in self.permitted_resources:
                raise ValueError("Bid uses an excluded resource")
            if not self.allows_task(task):
                raise ValueError("Bid leaves the bound machine lane")
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
        goal = model.get("resource_goal")
        candidate = model.get("candidate")
        established = (
            all(valuation[goal["resource_id"]].get(key) == value
                for key, value in goal["values"].items())
            if goal else matches_requirement(products[part], model["desired_property"])
        )
        if candidate is not None:
            established = bool(bid["path"]) and all(
                bid["path"][0].get(key) == candidate.get(key)
                for key in ("resource_id", "event_id", "event_name", "parameters")
            )
        if not established:
            raise ValueError("Returned path does not establish the requested property")
        return {
            "path": deepcopy(bid["path"]),
            "unresolved": unresolved,
            "execution_unavailable": execution_unavailable,
        }

    def prepare(self, task: dict, *, simulated: bool = False) -> dict:
        """Recheck a transition and reserve its exact execution dependencies."""
        rid = task["resource_id"]
        if rid not in self.permitted_resources:
            raise ValueError("Resource is excluded from this order")
        if not self.allows_task(task):
            raise ValueError("Task leaves the bound machine lane")
        actor = self.resources[rid]
        offered_revisions = task.get("offer_resource_revisions")
        if offered_revisions is not None:
            current_revisions = self.revisions()
            if any(offered_revisions.get(peer) != current_revisions[peer]
                   for peer in self._task_participants(task)):
                raise ValueError("Stale capability offer requires renewed PA-RA negotiation")
        part = task["parameters"].get("part_name") or task.get("part_name")
        if part is None:
            outstanding = self.outstanding()
            part = outstanding[0] if outstanding else self.product_name
        if ("offer_product_state" in task
                and task["offer_product_state"] != self.part_tracker.get(part)):
            raise ValueError("Stale product observation requires renewed PA-RA negotiation")
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
        reservations = self._task_reservations(task, part)
        conflicts = sorted(key for key in reservations if key in self.reservations)
        if conflicts:
            raise ValueError("Task conflicts with active reservations: " + ", ".join(conflicts))
        self._task_sequence += 1
        task_id = f"environment_{self._task_sequence}"
        resource_revisions = self.revisions()
        pending = {
            **deepcopy(task),
            "task_id": task_id,
            "run_id": self.run_id,
            "revision": self.revision,
            "resource_revisions": resource_revisions,
            "participant_revisions": {
                rid: resource_revisions[rid] for rid in self._task_participants(task)
            },
            "reservations": sorted(reservations),
            "evidence": "simulated" if simulated else "resource",
        }
        self.pending_tasks[task_id] = pending
        self.reservations.update({key: task_id for key in reservations})
        return deepcopy(pending)

    def _task_participants(self, task: dict) -> list[str]:
        event = next(
            event
            for event in self.models[task["resource_id"]]["events"]
            if event["event_id"] == task["event_id"]
        )
        return list(event["participants"])

    def _task_reservations(self, task: dict, part: str) -> set[str]:
        """Name controller, custody, and shared-workspace reservations."""
        reservations = {f"resource:{rid}" for rid in self._task_participants(task)}
        reservations.add(f"part:{part}")
        params = task["parameters"]
        machine_ids = {row["resource_id"] for row in self.inputs["scene"]["machines"]}
        machine_refs = {
            value
            for value in (
                task["resource_id"],
                params.get("origin_resource_location"),
                params.get("destination_location"),
                params.get("source_resource"),
                params.get("target_resource"),
            )
            if value in machine_ids
        }
        if task["event_name"] == "move_home":
            reservations.discard(f"part:{part}")
            model = self.models[task["resource_id"]]
            home_origin = self.resources[task["resource_id"]].valuation["resource_location"]
            if home_origin in machine_ids:
                machine_refs.add(home_origin)
            machine = model["assignments"].get("machine")
            if machine in machine_ids:
                machine_refs.add(machine)
            if self.product_name in model["state_variables"]["resource_location"]["domain"]:
                reservations.add(f"workspace:{self.product_name}")
        empty_kmr_return = (
            task["resource_id"] == "KMR"
            and task["event_name"] == "move_to_resource"
            and params.get("source_resource") in machine_ids
            and params.get("target_resource") == "Storage"
        )
        if empty_kmr_return:
            reservations.discard(f"part:{part}")
            machine_refs.discard(params.get("source_resource"))
        for machine_id in machine_refs:
            machine = next(
                row
                for row in self.inputs["scene"]["machines"]
                if row["resource_id"] == machine_id
            )
            reservations.add(f"access:{machine_id}")
            reservations.add(f"resource:{machine['handling_robot']}")
        if task["event_name"] == "place_insert" or params.get("part_name") == self.product_name:
            reservations.add(f"workspace:{self.product_name}")
        if task["resource_id"] == "Conveyor":
            reservations.add("transport:Conveyor")
            reservations.update(
                f"part:{name}" for name, state in self.part_tracker.items()
                if state.get("location") == "Conveyor"
            )
        if params.get("loading_position"):
            reservations.add(f"loading:{params['loading_position']}")
        if task["resource_id"] == "Buffer For Machined parts":
            reservations.add("transport:Buffer For Machined parts")
            reservations.add(f"buffer-zone:{params.get('zone')}")
            reservations.add(f"buffer-zone:{params.get('downstream_zone')}")
        return reservations

    def relevant_revisions_match(self, task: dict) -> bool:
        """Check only resources whose state or controller the task can change."""
        expected = task.get("participant_revisions", {})
        current = self.revisions(expected)
        return bool(expected) and all(current.get(rid) == value for rid, value in expected.items())

    def cancel_pending(self, task_id: str) -> dict | None:
        """Release reservations while retaining acknowledged state."""
        pending = self.pending_tasks.pop(task_id, None)
        if pending is None:
            return None
        for key in pending.get("reservations", []):
            if self.reservations.get(key) == task_id:
                self.reservations.pop(key, None)
        return deepcopy(pending)

    def acknowledge(self, acknowledgement: dict) -> bool:
        """Commit exactly matching acknowledged effects once, atomically."""
        task_id = acknowledgement.get("task_id")
        if task_id in self.acknowledgements:
            if fingerprint(acknowledgement) != fingerprint(self.acknowledgements[task_id]):
                raise ValueError("Conflicting repeated acknowledgement")
            return False
        pending = self.pending_tasks.get(str(task_id or ""))
        if pending is None:
            raise ValueError("No pending task")
        expected = {**pending, "status": "completed"}
        if (
            fingerprint(expected) != fingerprint(acknowledgement)
            or not self.relevant_revisions_match(pending)
        ):
            raise ValueError("Acknowledgement does not match pending task or current configuration")
        actor = self.resources[pending["resource_id"]]
        if (
            pending["evidence"] == "resource"
            and actor.validated_completions.get(task_id) != expected
        ):
            raise ValueError("Resource has not validated completion evidence")
        before = self.snapshot()
        after, products = project_transition(
            self.models,
            before,
            self.part_tracker,
            pending,
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
        if pending["evidence"] == "resource":
            record["observations"] = deepcopy(actor.completion_observations[task_id])
        for rid, values in after.items():
            if values != self.resources[rid].valuation:
                self.resources[rid].revision += 1
                self.resources[rid].valuation = values
                self.resources[rid].evidence = pending["evidence"]
        self.part_tracker = products
        self.revision += 1
        self.transitions.append(record)
        self.acknowledgements[task_id] = deepcopy(acknowledgement)
        self.cancel_pending(task_id)
        return True

    def report(self, *, _memo: dict | None = None) -> dict:
        """Save the order's semantics without reinterpreting historical process facts."""
        return deepcopy(
            {
                "schema_version": self.schema_version,
                "run_id": self.run_id,
                "inputs": self.inputs,
                "permitted_resources": self.permitted_resources,
                "models": self.models,
                "executable_tasks": {
                    rid: sorted(resource.executors)
                    for rid, resource in self.resources.items()
                },
                "initial_product_states": self.initial_product_states,
                "explorations": self.explorations,
                "negotiations": self.negotiations,
                "environment_model": self.environment_model,
                "transitions": self.transitions,
                "pending_tasks": self.pending_tasks,
                "reservations": self.reservations,
                "final_valuation": self.snapshot(),
                "final_product_states": self.part_tracker,
                "revision": self.revision,
            }, _memo
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
    for rid, event_names in report.get("executable_tasks", {}).items():
        if rid not in context.resources:
            raise ValueError("Saved executable task belongs to an unknown resource")
        resource = context.resources[rid]
        for event_name in event_names:
            resource.bind_executor(
                event_name,
                lambda _task: None,
                lambda _task, _evidence: True,
                validate_start=lambda _task, _state, _geometry: True,
            )
    for record in report["transitions"]:
        ack = record["acknowledgement"]
        if ack["evidence"] != "simulated":
            raise ValueError(
                "Controller evidence requires its resource validator; cannot replay as simulated"
            )
        task = {key: ack[key] for key in ("resource_id", "event_id", "event_name", "parameters")}
        task.update({key: deepcopy(ack[key]) for key in ('offer_request_id', 'offer_resource_revisions', 'offer_product_state') if key in ack})
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
