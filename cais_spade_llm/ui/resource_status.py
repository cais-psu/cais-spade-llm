"""Read resource status through SystemBridge without dispatching runtime work."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from cais_spade_llm.resources.environment_models import build_environment_models
from cais_spade_llm.ui import recovery_setup

if TYPE_CHECKING:
    from cais_spade_llm.ui.bridge import SystemBridge

logger = logging.getLogger(__name__)

SNAPSHOT_EVIDENCE = "Agent snapshot; observation evidence unavailable."
TELEMETRY_FIELDS = ("execution_mode", "controller_ready", "gripper_state", "position")


class ResourceStatusReader:
    """Cache descriptors while reading current values for active resources."""

    def __init__(self, bridge: SystemBridge) -> None:
        """Initialize a page-owned reader using the existing bridge surface.

        Args:
            bridge: Public UI-to-runtime surface.
        """
        self.bridge = bridge
        self._configuration_signature: tuple | None = None
        self._configured_models: dict[str, dict] = {}
        self._runtime_revision: Any = object()
        self._runtime_models: dict[str, dict] = {}
        self._outcome: dict = {}

    def _read_environment(self) -> None:
        revision_reader = getattr(self.bridge, "get_environment_capabilities_revision", None)
        snapshot_reader = getattr(self.bridge, "get_environment_capabilities", None)
        if snapshot_reader is None:
            return
        revision = revision_reader() if revision_reader is not None else None
        if revision_reader is not None and revision == self._runtime_revision:
            return
        snapshot = snapshot_reader()
        self._runtime_models = {
            name: {
                "state_variables": deepcopy(model["state_variables"]),
                "current_valuation": deepcopy(model.get("current_valuation", {})),
                "state_evidence": model.get("state_evidence"),
            }
            for name, model in snapshot.get("models", {}).items()
        }
        self._outcome = deepcopy(snapshot.get("outcome", {}))
        if self._outcome.get("status") == "blocked" and not self._outcome.get("tasks"):
            self._outcome["tasks"] = deepcopy(
                snapshot.get("environment_model", {}).get("selected_path", [])[:1]
            )
        self._runtime_revision = revision

    def _read_configuration(self) -> None:
        setup = recovery_setup.load_setup()
        path = recovery_setup.reference_path(setup["scene_file"])
        stat = path.stat()
        signature = (str(path), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
        if signature == self._configuration_signature:
            return
        models = build_environment_models(self.bridge.load_config(str(path)))
        # Configured values describe assumptions, never observations for this panel.
        self._configured_models = {
            name: {"state_variables": deepcopy(model["state_variables"])}
            for name, model in models.items()
        }
        self._configuration_signature = signature

    def read(self) -> dict[str, Any]:
        """Return active resource values, descriptors, evidence, and execution outcome.

        Returns:
            A display snapshot. Missing descriptors leave the supplied agent
            values available and include an explicit descriptor error.
        """
        states = self.bridge.get_robot_states()
        self._read_environment()
        error = ""
        if any(name not in self._runtime_models for name in states):
            try:
                self._read_configuration()
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self._configuration_signature = None
                self._configured_models = {}
                error = f"Resource descriptors unavailable: {exc}"
                logger.warning("%s", error)

        resources = {}
        outcome = deepcopy(self._outcome)
        for name, state in states.items():
            live = self._runtime_models.get(name)
            model = live if live is not None else self._configured_models.get(name)
            values = deepcopy(state)
            evidence = state.get("state_evidence") or SNAPSHOT_EVIDENCE
            if not outcome and isinstance(state.get("execution_outcome"), dict):
                outcome = deepcopy(state["execution_outcome"])
                details = outcome.get("details", {})
                if details.get("content"):
                    outcome.setdefault("reason", details["content"])
            if live is not None:
                values = deepcopy(live.get("current_valuation", {}))
                values.update(
                    {key: deepcopy(state[key]) for key in TELEMETRY_FIELDS if key in state}
                )
                evidence = live.get("state_evidence") or SNAPSHOT_EVIDENCE
            resources[name] = {
                "state": values,
                "model": (
                    {"state_variables": deepcopy(model["state_variables"])}
                    if model is not None
                    else None
                ),
                "evidence": evidence,
            }
        return {
            "resources": resources,
            "outcome": outcome if states else {},
            "error": error,
        }
