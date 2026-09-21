"""ProductAgent's nominal state, plans, and atomic acknowledgement handling."""

from __future__ import annotations

import json
from copy import deepcopy

from cais_spade_llm.product.nominal_planner import nominal_requirement, plan_nominal_order
from cais_spade_llm.product.order import validate_completion_conditions, validate_product_order
from cais_spade_llm.product.state import nominal_product_states, project_product_states
from cais_spade_llm.resources.nominal_context import NominalResourceContext
from cais_spade_llm.resources.nominal_des import build_nominal_resource_des_models


class NominalProductContext:
    """Own one product/order and coordinate acknowledged shared handoffs."""

    def __init__(self, scene: dict, product_order: dict, geometry: dict) -> None:
        validated = validate_product_order(product_order, geometry)
        if validated.payload["quantity"] != 1:
            raise ValueError("The bounded nominal inventory supports quantity: 1")
        self.models = build_nominal_resource_des_models(scene)
        self.product_name = self.models["Exit"]["assignments"]["completed_product"]
        if validated.payload["product"] != self.product_name:
            raise ValueError("Product order does not match the configured completed_product")
        self.selected_parts = list(validated.selected_parts)
        if len(self.selected_parts) != len(set(self.selected_parts)):
            raise ValueError("The bounded nominal inventory cannot repeat a part identity")
        self.component_parts = [
            *self.models["Conveyor"]["assignments"]["nominal_parts"],
            *self.models["3D Printing Station"]["assignments"]["supported_products"],
        ]
        if not set(self.selected_parts) <= set(self.component_parts):
            raise ValueError("Product order includes parts outside the nominal inventory")
        self.product_order = deepcopy(validated.payload)
        conditions = self.product_order.get("completion_conditions")
        if conditions is not None:
            validate_completion_conditions(conditions, self.models)
        self.requirements = {
            part: deepcopy(conditions) if conditions is not None else nominal_requirement(self.models, part)
            for part in self.selected_parts
        }
        if conditions is None and set(self.selected_parts) == set(self.component_parts):
            self.requirements[self.product_name] = nominal_requirement(
                self.models, self.product_name
            )
        self.inputs = {
            "scene": deepcopy(scene),
            "product_order": deepcopy(product_order),
            "geometry": deepcopy(geometry),
        }
        self.resources = {rid: NominalResourceContext(model) for rid, model in self.models.items()}
        self.revision = 0
        self.part_tracker = nominal_product_states(self.models, self.snapshot())
        self.initial_product_states = deepcopy(self.part_tracker)
        self.history: dict[str, list[dict]] = {part: [] for part in self.part_tracker}
        self.plans: list[dict] = []
        self.capability_requests: list[dict] = []
        self.transitions: list[dict] = []
        self.pending: dict | None = None
        self._acknowledgements: dict[str, dict] = {}
        self._next_task = 1

    def snapshot(self) -> dict:
        """Read a coherent copy of every resource's acknowledged valuation."""
        return {rid: resource.snapshot() for rid, resource in self.resources.items()}

    def plan(self, *, max_search_states: int = 50_000) -> dict:
        """Request and record a nominal plan without committing its predictions."""
        result = plan_nominal_order(self, max_search_states=max_search_states)
        self.capability_requests.extend(deepcopy(result["requests"]))
        self.plans.append({"revision": self.revision, **deepcopy(result)})
        return result

    def prepare(self, task: dict, *, task_id: str | None = None) -> dict:
        """Revalidate and bind one pending task to the current valuation revision."""
        if self.pending is not None:
            raise ValueError("A nominal task is already awaiting acknowledgement")
        if task_id is not None and (not isinstance(task_id, str) or not task_id
                                    or task_id in self._acknowledgements):
            raise ValueError("Expected a new task identity")
        rid = task.get("resource_id") if isinstance(task, dict) else None
        if not isinstance(rid, str) or rid not in self.resources:
            raise ValueError("Unknown nominal ResourceAgent")
        self.resources[rid].validate_nominal_event(self.models, self.snapshot(), task)
        self.pending = {
            **deepcopy(task),
            "task_id": task_id if task_id is not None else f"nominal_{self._next_task}",
            "revision": self.revision,
        }
        self._next_task += 1
        return deepcopy(self.pending)

    def acknowledge(self, acknowledgement: dict) -> bool:
        """Commit a matching simulated completion exactly once, or change nothing."""
        return self._acknowledge(acknowledgement, evidence="simulated")

    def acknowledge_gazebo(self, acknowledgement: dict) -> bool:
        """Commit delivery effects verified by the responsible Gazebo resource.

        The transport adapter must also authenticate the ResourceAgent sender.
        """
        from cais_spade_llm.recovery_framework.delivery import validate_execution_evidence

        validate_execution_evidence(acknowledgement)
        return self._acknowledge(acknowledgement, evidence="gazebo")

    def _acknowledge(self, acknowledgement: dict, *, evidence: str) -> bool:
        task_id = acknowledgement.get("task_id") if isinstance(acknowledgement, dict) else None
        if not isinstance(task_id, str):
            raise ValueError("Nominal acknowledgement task_id must be a string")
        try:
            encoded = json.dumps(acknowledgement, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("Nominal acknowledgement must contain exact JSON bindings") from exc
        if task_id in self._acknowledgements:
            if encoded != json.dumps(self._acknowledgements[task_id], sort_keys=True):
                raise ValueError("Conflicting duplicate nominal acknowledgement")
            return False
        if self.pending is None:
            raise ValueError("No matching pending nominal task")
        expected = {**self.pending, "status": "completed", "evidence": evidence}
        if evidence == "gazebo":
            expected["observations"] = acknowledgement["observations"]
        if (
            encoded != json.dumps(expected, sort_keys=True)
            or self.pending["revision"] != self.revision
        ):
            raise ValueError("Nominal acknowledgement does not match pending task and revision")
        task = self.pending
        before = self.snapshot()
        actor = self.resources[task["resource_id"]]
        after = actor.validate_nominal_event(self.models, before, task)
        event = next(
            row
            for row in self.models[task["resource_id"]]["events"]
            if row["event_id"] == task["event_id"]
        )
        products, affected = project_product_states(
            self.models, before, after, self.part_tracker, task, event
        )
        record = {
            "task": deepcopy(task),
            "acknowledgement": deepcopy(acknowledgement),
            "revision": self.revision + 1,
            "before": before,
            "after": after,
            "product_before": deepcopy(self.part_tracker),
            "product_after": deepcopy(products),
            "affected_parts": affected,
        }
        history = deepcopy(self.history)
        for part in affected:
            history[part].append(
                {
                    "task_id": task_id,
                    "revision": record["revision"],
                    "resource_id": task["resource_id"],
                    "event_name": task["event_name"],
                    "parameters": deepcopy(task["parameters"]),
                    "state": deepcopy(products[part]),
                }
            )
        # No await or external callback may split this in-process transaction.
        for rid, values in after.items():
            self.resources[rid]._commit(deepcopy(values))
        self.part_tracker = products
        self.history = history
        self.revision += 1
        self.transitions.append(record)
        self._acknowledgements[task_id] = deepcopy(acknowledgement)
        self.pending = None
        return True
