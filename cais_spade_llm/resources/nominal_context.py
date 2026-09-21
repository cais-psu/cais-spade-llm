"""Transport-independent ResourceAgent ownership of a nominal resource model."""

from __future__ import annotations

from copy import deepcopy

from cais_spade_llm.resources.nominal_des import project_nominal_event


class NominalResourceContext:
    """Own one resource's descriptor and acknowledged valuation."""

    def __init__(self, model: dict) -> None:
        self.resource_id = model["resource_id"]
        self._model = deepcopy(model)
        self._valuation = deepcopy(model["current_valuation"])

    def nominal_des_model(self) -> dict:
        """Return independent capabilities with the current valuation."""
        model = deepcopy(self._model)
        model["current_valuation"] = self.snapshot()
        return model

    def snapshot(self) -> dict:
        """Return this resource's acknowledged state."""
        return deepcopy(self._valuation)

    def validate_nominal_event(self, models: dict, valuation: dict, task: dict) -> dict:
        """Validate a task performed by this resource against every participant."""
        if (
            not isinstance(task, dict)
            or not isinstance(task.get("resource_id"), str)
            or not isinstance(task.get("event_name"), str)
            or type(task.get("event_id")) is not int
            or not isinstance(task.get("parameters"), dict)
        ):
            raise ValueError(
                "Nominal task requires resource_id, event_id, event_name, and parameters"
            )
        if task["resource_id"] != self.resource_id:
            raise ValueError("Nominal task resource_id does not match ResourceAgent")
        event = next(
            (event for event in self._model["events"] if event["event_id"] == task["event_id"]),
            None,
        )
        if event is None or event["event_name"] != task["event_name"]:
            raise ValueError("Nominal task does not identify a declared capability")
        expected = event["parameter_bindings"]
        for field, binding in expected.items():
            if "equals" in binding and task["parameters"].get(field) != binding["equals"]:
                raise ValueError("Nominal task bindings do not match event_id")
        return project_nominal_event(
            models, valuation, self.resource_id, task["event_name"], task["parameters"]
        )

    def _commit(self, valuation: dict) -> None:
        self._valuation = valuation
