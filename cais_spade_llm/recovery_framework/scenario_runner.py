"""Snapshot-backed recovery diagnostics using the runtime planners and validators."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("CAIS_RECOVERY_INPUT_ROOT", ROOT)).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _bootstrap_repo_site_packages(root: Path) -> None:
    venv_lib = root / ".venv" / "lib"
    if not venv_lib.exists():
        return
    for site_packages in sorted(venv_lib.glob("python*/site-packages")):
        site_path = str(site_packages.resolve())
        if site_path not in sys.path:
            sys.path.insert(0, site_path)


_bootstrap_repo_site_packages(ROOT)


from cais_spade_llm.agents.central_controller import (  # noqa: E402
    outline_macro_safety as outline_macro_safety_module,
)
from cais_spade_llm.agents.central_controller.outline_macro_safety import (  # noqa: E402
    validate_outline_macro_recovery_safety,
)
from cais_spade_llm.agents.central_controller.recovery_safety_generation import (
    generate_recovery_safety_bundle,
)  # noqa: E402
from cais_spade_llm.agents.intelligent_product.process_planner import ProcessPlanner  # noqa: E402
from cais_spade_llm.agents.intelligent_product.product_recovery_controller import (  # noqa: E402
    _private_pending_nominal_tasks,
)
from cais_spade_llm.agents.intelligent_product.replanner.failure_context import (  # noqa: E402
    build_failure_event,
    failure_context_from_scenario_config,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (  # noqa: E402
    multi_turn as multi_turn_mode,
)
from cais_spade_llm.agents.resource_agent.resource_agent import (  # noqa: E402
    ResourceAgent,
)
from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent  # noqa: E402
from cais_spade_llm.agents.shared_information import llm_agent as llm_agent_module  # noqa: E402
from cais_spade_llm.agents.shared_information.recovery_validation_protocol import (  # noqa: E402
    recovery_validation_fingerprint,
)
from cais_spade_llm.resources.resource_primitives import (  # noqa: E402
    get_resource_recovery_snapshot,
)

DEBUG_ROOT = ROOT / "cais_spade_llm" / "monitor" / "debug"
CASE3_RESPONSE_FIXTURES = ROOT / "test" / "fixtures" / "part_slippage"
CASE3_RUNTIME_CONTEXT = CASE3_RESPONSE_FIXTURES / "runtime_context.json"
AUTO_CANDIDATE_COUNT_CAP = 8


def _load_local_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def _env_default(*env_names: str, fallback: str) -> str:
    for env_name in env_names:
        token = str(os.environ.get(env_name) or "").strip()
        if token:
            return token
    return fallback


_load_local_env(ROOT / ".env")
DEFAULT_LIVE_MODEL = _env_default(
    "CASE3_RECOVERY_MODEL",
    "CAIS_SPADE_LLM_MODEL",
    "OPENAI_MODEL",
    fallback="gpt-5.4",
)
DEFAULT_REASONING_EFFORT = _env_default(
    "CAIS_SPADE_REASONING_EFFORT",
    "OPENAI_REASONING_EFFORT",
    fallback="medium",
)


def _normalize_reasoning_effort_for_model(model_name: str, effort: str) -> str:
    normalized_model = str(model_name or "").strip().lower()
    normalized_effort = str(effort or "").strip().lower()
    if normalized_model.startswith("gpt-5.4") and normalized_effort == "minimal":
        return "none"
    return normalized_effort


def _parse_structured_json_text(raw_text: str) -> Any:
    text = str(raw_text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        decoder = json.JSONDecoder()
        for start_idx, ch in enumerate(text):
            if ch not in "{[":
                continue
            try:
                parsed, _end_idx = decoder.raw_decode(text[start_idx:])
            except json.JSONDecodeError:
                continue
            return parsed
        raise exc


def _load_json(path: Path) -> dict[str, Any] | list[Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if DATA_ROOT == ROOT:
        return value

    def frozen(item: Any) -> Any:
        if isinstance(item, dict):
            return {key: frozen(child) for key, child in item.items()}
        if isinstance(item, list):
            return [frozen(child) for child in item]
        if isinstance(item, str) and Path(item).suffix:
            reference = Path(item)
            if reference.is_absolute() and reference.is_relative_to(ROOT):
                reference = reference.relative_to(ROOT)
            for base in (DATA_ROOT, path.parent):
                candidate = (base / reference).resolve()
                if candidate.is_relative_to(DATA_ROOT) and candidate.is_file():
                    return str(candidate)
        return item

    return frozen(value)


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return str(path)


def _utc_token() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def _debug_root(debug_root: Path | None = None) -> Path:
    root = debug_root or DEBUG_ROOT
    return root if root.is_absolute() else (ROOT / root)


def _repo_path(raw_path: Any) -> Path:
    raw_token = str(raw_path or "").strip()
    if not raw_token:
        raise ValueError("runtime context path is empty")
    path = Path(raw_token)
    if path.is_absolute():
        if DATA_ROOT != ROOT and not path.is_relative_to(DATA_ROOT):
            path = DATA_ROOT / path.relative_to(ROOT)
        return path
    return DATA_ROOT / path


def _load_case3_runtime_context(runtime_context_path: Path | None = None) -> dict[str, Any]:
    payload = _load_json(runtime_context_path or CASE3_RUNTIME_CONTEXT)
    if not isinstance(payload, dict):
        raise TypeError(f"runtime context {CASE3_RUNTIME_CONTEXT} did not decode to an object")
    return payload


def _bundle_artifact_path(bundle_root: Path, artifacts: dict[str, Any], key: str) -> Path:
    raw_path = str(artifacts.get(key) or "").strip()
    if not raw_path:
        raise KeyError(f"bundle manifest is missing artifacts.{key}")
    path = Path(raw_path)
    return _repo_path(path) if path.is_absolute() else bundle_root / path


def _case3_paths(runtime_context: dict[str, Any] | None = None) -> dict[str, Path]:
    context = dict(runtime_context or _load_case3_runtime_context())
    bundle_root = _repo_path(context.get("bundle_root"))
    bundle_manifest = bundle_root / "bundle_manifest.json"
    manifest_payload = _load_json(bundle_manifest)
    if not isinstance(manifest_payload, dict):
        raise TypeError(f"bundle manifest {bundle_manifest} did not decode to an object")
    artifacts = dict(manifest_payload.get("artifacts") or {})
    failure_scenario_id = str(context.get("failure_scenario_id") or "").strip()
    if not failure_scenario_id:
        raise ValueError("runtime context failure_scenario_id is empty")
    return {
        "bundle_manifest": bundle_manifest,
        "tools": _bundle_artifact_path(bundle_root, artifacts, "tools_json"),
        "plan": _bundle_artifact_path(bundle_root, artifacts, "plan_json"),
        "requirements": _bundle_artifact_path(bundle_root, artifacts, "requirements_json"),
        "safety_logic": _bundle_artifact_path(bundle_root, artifacts, "safety_logic_json"),
        "geometry": _repo_path(context.get("product_geometry")),
        "failure_scenario": (
            DATA_ROOT
            / "cais_spade_llm"
            / "initialization"
            / "failure_scenarios"
            / f"{failure_scenario_id}.json"
        ),
        "recovery_outline_experiment_settings": (
            DATA_ROOT
            / "cais_spade_llm"
            / "initialization"
            / "recovery_outline_experiment_settings.json"
        ),
    }


def _case3_bundle_context(paths: dict[str, Path]) -> dict[str, Any]:
    manifest_payload = _load_json(paths["bundle_manifest"])
    if not isinstance(manifest_payload, dict):
        manifest_payload = {}
    manifest_payload["artifacts"] = {
        **dict(manifest_payload.get("artifacts") or {}),
        "requirements_json": str(paths["requirements"]),
        "plan_json": str(paths["plan"]),
        "tools_json": str(paths["tools"]),
        "safety_logic_json": str(paths["safety_logic"]),
    }
    manifest_payload["manifest_path"] = str(paths["bundle_manifest"])
    return manifest_payload


def _normalize_recovery_outline_experiment_settings(raw_settings: Any) -> dict[str, Any]:
    raw = dict(raw_settings or {}) if isinstance(raw_settings, dict) else {}
    enabled = bool(raw.get("enabled", True))
    recovery_selection_mode = str(raw.get("recovery_selection_mode") or "pure_llm").strip().lower()
    if recovery_selection_mode not in {"pure_llm", "neurosymbolic"}:
        recovery_selection_mode = "pure_llm"

    raw_action_horizon = raw.get("action_horizon", 1)
    action_horizon: int | str
    if isinstance(raw_action_horizon, str):
        horizon_token = raw_action_horizon.strip().lower()
        if horizon_token == "full":
            action_horizon = "full"
        else:
            try:
                action_horizon = max(1, int(horizon_token))
            except ValueError:
                action_horizon = 1
    else:
        try:
            action_horizon = max(1, int(raw_action_horizon))
        except (TypeError, ValueError):
            action_horizon = 1

    raw_candidate_count = raw.get("candidate_count", "auto")
    candidate_count: int | str
    if isinstance(raw_candidate_count, str):
        candidate_token = raw_candidate_count.strip().lower()
        if candidate_token in {"adaptive", "auto", "n"}:
            candidate_count = "auto"
        else:
            try:
                candidate_count = max(1, int(candidate_token))
            except ValueError:
                candidate_count = "auto"
    else:
        try:
            candidate_count = max(1, int(raw_candidate_count))
        except (TypeError, ValueError):
            candidate_count = "auto"

    candidate_proposal_budget = max(
        1,
        int(raw.get("candidate_proposal_budget") or 5),
    )
    if recovery_selection_mode == "neurosymbolic":
        action_horizon = 1
        candidate_count = "auto"

    return {
        "enabled": enabled,
        "recovery_selection_mode": recovery_selection_mode,
        "action_horizon": action_horizon,
        "candidate_count": candidate_count,
        "candidate_proposal_budget": candidate_proposal_budget,
    }


def _recovery_action_horizon_fields(action_horizon: int | str) -> dict[str, Any]:
    if action_horizon == "full":
        return {
            "action_horizon": "full",
            "action_horizon_steps": "full",
            "action_horizon_k": 1,
        }
    steps = max(1, int(action_horizon))
    return {
        "action_horizon": "1" if steps == 1 else "k",
        "action_horizon_steps": steps,
        "action_horizon_k": steps,
    }


def _recovery_candidate_count_fields(candidate_count: int | str) -> dict[str, Any]:
    if candidate_count == "auto":
        return {
            "candidate_count": "auto",
            "candidate_bound": AUTO_CANDIDATE_COUNT_CAP,
            "candidate_bound_cap": AUTO_CANDIDATE_COUNT_CAP,
        }
    count = max(1, int(candidate_count))
    return {
        "candidate_count": count,
        "candidate_bound": count,
        "candidate_bound_cap": count,
    }


def _load_recovery_outline_experiment_settings(
    settings_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(settings_override, dict):
        return _normalize_recovery_outline_experiment_settings(settings_override)
    settings_path = (
        DATA_ROOT / "cais_spade_llm/initialization/recovery_outline_experiment_settings.json"
    )
    if not settings_path.exists():
        return _normalize_recovery_outline_experiment_settings({})
    payload = _load_json(settings_path)
    return _normalize_recovery_outline_experiment_settings(payload)


def _load_robot_config(path: Path, key: str) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise TypeError(f"robot config {path} did not decode to an object")
    config = payload.get(key)
    if not isinstance(config, dict):
        raise KeyError(f"robot config {path} is missing top-level key {key!r}")
    return config


class ProcessPlannerPrepareTrace(ProcessPlanner):
    """ProcessPlanner with the active recovery-session wiring."""


class FakeProductAgent:
    """Small product-agent surface used by the Case 3 dry-run harness."""

    def __init__(
        self,
        *,
        tools_catalog: list[dict[str, Any]],
        product_geometry: dict[str, Any],
        llm_model: str | None = None,
        llm_reasoning_effort: str | None = None,
        precomputed_bundle: dict[str, Any] | None = None,
        scripted_responses: list[dict[str, Any]] | None = None,
    ) -> None:
        self.jid = "assembly_board-v1@localhost"
        self.cca_jid = "central_controller@localhost"
        self.agent_name = "assembly_board-v1"
        self.instructions = "Part slippage saved-context recovery diagnostic product agent."
        self.logger = logging.getLogger("case3_recovery_dryrun")
        self.tools_catalog = deepcopy(tools_catalog)
        self.product_geometry = deepcopy(product_geometry)
        self.llm_model = str(llm_model or DEFAULT_LIVE_MODEL).strip()
        self.llm_reasoning_effort = _normalize_reasoning_effort_for_model(
            self.llm_model,
            str(llm_reasoning_effort or DEFAULT_REASONING_EFFORT).strip(),
        )
        self.precomputed_bundle = deepcopy(precomputed_bundle or {})
        self.structured_requirements_path = Path(
            str((self.precomputed_bundle.get("artifacts") or {}).get("requirements_json") or "")
        )
        self._recovery_reasoning_mode = "multi_turn"
        self.turn_log: list[dict[str, Any]] = []
        self._turn_index = 0
        self._uses_scripted_responses = scripted_responses is not None
        self._scripted_responses = [deepcopy(row) for row in scripted_responses or []]
        self._last_structured_request: dict[str, Any] = {}
        self._mock_recovery_validation_resources: dict[str, Any] = {}
        self._saved_safety_context: dict[str, Any] = {}

    def _saved_monitor_states(self, monitor: Any) -> dict[str, str]:
        """Use saved CCA history; only explicit replay may mock an initial state."""
        saved = self._saved_safety_context
        if saved.get("history_error"):
            raise ValueError("Saved CCA monitor history is unavailable")
        states = saved.get("safety_dfa_states")
        if states is None and (self._uses_scripted_responses or not monitor.current_states):
            return deepcopy(monitor.current_states)
        if not isinstance(states, dict) or set(states) != set(monitor.current_states):
            raise ValueError(
                "Saved CCA safety_dfa_states are missing or do not match the active rules"
            )
        if not isinstance(saved.get("running_aps"), list):
            raise ValueError("Saved CCA running_aps evidence is missing")
        for rule_id, state in states.items():
            dfa = monitor.dfas[rule_id]
            known = set(dfa["transitions"]) | set(dfa["accepting_states"])
            known.update(target for edges in dfa["transitions"].values() for _, target in edges)
            if state not in known:
                raise ValueError(f"Saved CCA state for {rule_id!r} is invalid")
        return deepcopy(states)

    async def request_recovery_outline_physical_validation(
        self,
        *,
        resource_jid: str,
        payload: dict[str, Any],
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        """Invoke the explicitly mocked owning RA transport for dry-run tests."""
        del timeout_s
        resource = self._mock_recovery_validation_resources.get(resource_jid)
        if resource is None:
            raise RuntimeError(f"mocked RA '{resource_jid}' is unavailable")
        validation = ResourceAgent.validate_recovery_outline_physical_candidates(
            resource,
            payload,
        )
        return {
            "request_id": str(payload.get("request_id") or "mocked-ra-request"),
            "recovery_session_id": str(payload.get("recovery_session_id") or ""),
            "turn_index": int(payload.get("turn_index") or 0),
            "state_fingerprint": str(payload.get("state_fingerprint") or ""),
            **deepcopy(validation),
            "latency_ms": 0.0,
            "mocked": True,
        }

    async def request_recovery_outline_safety_validation(
        self,
        *,
        payload: dict[str, Any],
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        """Invoke the explicitly mocked CCA owner for dry-run tests."""
        del timeout_s
        results: list[dict[str, Any]] = []
        all_rules: list[dict[str, Any]] = []
        live_safety_dfa_states: dict[str, str] = {}
        for row in payload.get("candidates") or []:
            candidate = dict(row or {})
            safety_input = deepcopy(dict(candidate.get("safety_input") or {}))
            llm_input = deepcopy(dict(safety_input.get("llm_input") or {}))
            rules = [
                deepcopy(rule)
                for rule in (llm_input.get("loaded_safety_rules") or [])
                if isinstance(rule, dict)
            ]
            all_rules.extend(rules)
            live_monitor = outline_macro_safety_module._build_recovery_safety_monitor(
                rules=rules,
                state_aps=[],
                llm_input=llm_input,
            )
            live_safety_dfa_states = self._saved_monitor_states(live_monitor)
            safety_rule_fingerprint = recovery_validation_fingerprint(rules)
            live_safety_dfa_state_fingerprint = recovery_validation_fingerprint(
                live_safety_dfa_states
            )
            expected_rule_fingerprint = str(candidate.get("safety_rule_fingerprint") or "").strip()
            if expected_rule_fingerprint and expected_rule_fingerprint != safety_rule_fingerprint:
                raise RuntimeError("CCA safety-rule fingerprint changed")
            expected_live_state_fingerprint = str(
                candidate.get("live_safety_dfa_state_fingerprint") or ""
            ).strip()
            if (
                expected_live_state_fingerprint
                and expected_live_state_fingerprint != live_safety_dfa_state_fingerprint
            ):
                raise RuntimeError("CCA live safety DFA state changed")
            projected_dfa_states = candidate.get("safety_dfa_states_before")
            if not isinstance(projected_dfa_states, dict):
                projected_dfa_states = live_safety_dfa_states
            result = validate_outline_macro_recovery_safety(
                task=deepcopy(safety_input.get("task") or {}),
                signature=deepcopy(safety_input.get("signature") or {}),
                pre_resources=deepcopy(safety_input.get("pre_resources") or {}),
                pre_parts=deepcopy(safety_input.get("pre_parts") or {}),
                projected_resources=deepcopy(safety_input.get("projected_resources") or {}),
                projected_parts=deepcopy(safety_input.get("projected_parts") or {}),
                llm_input=llm_input,
                safety_dfa_states_before=deepcopy(projected_dfa_states),
            )
            results.append(
                {
                    "candidate_index": int(candidate.get("candidate_index") or 0),
                    "event_id": str(candidate.get("event_id") or "").strip(),
                    "is_safe": bool(result.get("is_safe")),
                    "findings": deepcopy(result.get("findings") or []),
                    "active_rule_identifiers": [
                        str(rule.get("id") or "").strip()
                        for rule in rules
                        if str(rule.get("id") or "").strip()
                    ],
                    "safety_context": deepcopy(result.get("safety_ctx") or {}),
                    "safety_dfa_states_before": deepcopy(
                        result.get("safety_dfa_states_before") or {}
                    ),
                    "safety_dfa_states_after": deepcopy(
                        result.get("safety_dfa_states_after") or {}
                    ),
                    "cca_admissible_goal_recovery_event_ids": deepcopy(
                        result.get("cca_admissible_goal_recovery_event_ids") or []
                    ),
                    "cca_admissible_goal_recovery_event_ids_before": deepcopy(
                        result.get("cca_admissible_goal_recovery_event_ids_before") or []
                    ),
                    "cca_admissible_goal_recovery_event_ids_after": deepcopy(
                        result.get("cca_admissible_goal_recovery_event_ids_after") or []
                    ),
                    "admissible_nominal_reentry_event_ids": deepcopy(
                        result.get("admissible_nominal_reentry_event_ids") or []
                    ),
                    "admissible_nominal_reentry_event_ids_before": deepcopy(
                        result.get("admissible_nominal_reentry_event_ids_before") or []
                    ),
                    "admissible_nominal_reentry_event_ids_after": deepcopy(
                        result.get("admissible_nominal_reentry_event_ids_after") or []
                    ),
                }
            )
        return {
            "request_id": str(payload.get("request_id") or "mocked-cca-request"),
            "recovery_session_id": str(payload.get("recovery_session_id") or ""),
            "turn_index": int(payload.get("turn_index") or 0),
            "state_fingerprint": str(payload.get("state_fingerprint") or ""),
            "validator_jid": self.cca_jid,
            "results": results,
            "admissible_recovery_event_ids": sorted(
                {
                    str(row.get("event_id") or "").strip()
                    for row in results
                    if bool(row.get("is_safe")) and str(row.get("event_id") or "").strip()
                }
            ),
            "active_rule_identifiers": sorted(
                {
                    str(rule.get("id") or "").strip()
                    for rule in all_rules
                    if str(rule.get("id") or "").strip()
                }
            ),
            "safety_rule_fingerprint": recovery_validation_fingerprint(all_rules),
            "live_safety_dfa_states": deepcopy(live_safety_dfa_states),
            "live_safety_dfa_state_fingerprint": recovery_validation_fingerprint(
                live_safety_dfa_states
            ),
            "latency_ms": 0.0,
            "mocked": True,
        }

    def _geometry_for_part(self, part_name: str) -> dict[str, Any]:
        board = dict(self.product_geometry.get("assembly_board") or {})
        parts = dict(self.product_geometry.get("parts") or {})
        slot_xy = dict(board.get("slots") or {}).get(part_name)
        if slot_xy is None:
            return {}
        return {
            "slot_xy": slot_xy,
            "part_height_m": dict(parts.get("heights_m") or {}).get(part_name),
            "model_name": dict(parts.get("model_map") or {}).get(part_name),
            "slot_floor_z_m": board.get("slot_floor_z_m"),
            "board_center": deepcopy(board.get("center") or {}),
        }

    async def ask_llm_structured(
        self,
        prompt: str,
        *,
        response_format: dict[str, Any],
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Callable[[str, dict[str, Any]], Any] | None = None,
        max_tool_rounds: int = 3,
        include_agent_instructions: bool = True,
    ) -> dict[str, Any]:
        assert include_agent_instructions is False
        response_source = "mocked_scripted_fixture" if self._uses_scripted_responses else "live"
        self._last_structured_request = {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "reasoning_effort": self.llm_reasoning_effort,
            "response_format": {
                "type": "json_schema",
                "json_schema": deepcopy(response_format),
            },
            "response_source": response_source,
            "request_sent": not self._uses_scripted_responses,
        }
        if tools:
            self._last_structured_request["tools"] = deepcopy(tools)
        if self._scripted_responses:
            parsed = deepcopy(self._scripted_responses.pop(0))
            self._record_turn(prompt, parsed)
            return parsed
        if self._uses_scripted_responses:
            raise RuntimeError("mocked scripted response fixtures were exhausted")

        self.model = self.llm_model
        self.reasoning_effort = self.llm_reasoning_effort
        parsed = await llm_agent_module.LlmAgent.ask_llm_structured(
            self,
            prompt,
            response_format=response_format,
            tools=tools,
            tool_executor=tool_executor,
            max_tool_rounds=max_tool_rounds,
            include_agent_instructions=False,
        )
        self._record_turn(prompt, parsed)
        return parsed

    def _record_turn(self, prompt: str, response: dict[str, Any]) -> None:
        self._turn_index += 1
        self.turn_log.append(
            {
                "turn_index": self._turn_index,
                "model": self.llm_model,
                "prompt": prompt,
                "response": deepcopy(response),
            }
        )


class FakeRecoveryRobot:
    """Small robot-agent surface used by the recovery dry-run."""

    _RECOVERY_PRIMITIVES = (
        "move_cartesian",
        "move_pose",
        "move_relative",
        "move_to_named_pose",
        "delay",
        "grasp_part",
        "release_part",
        "open_gripper",
        "close_gripper",
        "detect_parts",
        "compute_pick_targets",
        "compute_place_targets",
        "attach_part",
        "detach_part",
        "get_current_pose",
    )
    recovery_des_model = RobotAgent.recovery_des_model
    resolve_registered_function_names = RobotAgent.resolve_registered_function_names
    recovery_validation_resource_matches = ResourceAgent.recovery_validation_resource_matches

    def __init__(  # noqa: PLR0913 - snapshot adapter mirrors the configured robot state
        self,
        *,
        config: dict[str, Any],
        execution_env: str,
        current_state: str,
        held_part: str | None,
        gripper_state: str,
        pose_ref: str | None,
        position: dict[str, float],
        current_location: str | None = None,
        occupancy: dict[str, Any] | None = None,
        observations: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        env_block = dict(config.get(execution_env) or {})
        self.agent_name = str(config.get("jid") or "").strip()
        self.jid = self.agent_name
        self.execution_mode = "dry_run"
        self.static_capabilities = deepcopy(env_block.get("static_capabilities") or {})
        self.static_capabilities.setdefault("resource_type", "robot")
        self.named_positions = deepcopy(env_block.get("named_positions") or {})
        self.controller_config = deepcopy(env_block.get("controller") or {})
        self._current_state = str(current_state)
        self._held_part = held_part
        self._gripper_state = str(gripper_state)
        self._recovery_pose_ref = pose_ref
        self._position = deepcopy(position)
        self._current_location = current_location
        self._occupancy = deepcopy(occupancy or {})
        self._observations = deepcopy(observations or {})
        self._shared_observations: dict[str, dict[str, Any]] = {}
        self.logger = logging.getLogger(f"FakeRecoveryRobot.{self.agent_name}")

    def set_shared_observations(self, observations: dict[str, dict[str, Any]] | None) -> None:
        self._shared_observations = deepcopy(observations or {})

    def _observation_catalog(self) -> dict[str, dict[str, Any]]:
        catalog = deepcopy(self._shared_observations)
        catalog.update(deepcopy(self._observations))
        return catalog

    def move_cartesian(
        self,
        x: float,
        y: float,
        z: float,
        speed: float | None = None,
    ) -> dict[str, Any]:
        """
        ---
        description: Move end-effector to an absolute Cartesian position.
        params:
          x: {type: number}
          y: {type: number}
          z: {type: number}
          speed: {type: number}
        preconditions: {}
        effects:
          current_pose:
            pose_absolute_from_params: [x, y, z]
          current_pose_ref:
            set_unknown: true
        ---
        """
        del speed
        self._position = {"x": float(x), "y": float(y), "z": float(z)}
        self._recovery_pose_ref = None
        return {"success": True, "message": "fake move_cartesian ok"}

    def move_pose(
        self,
        x: float,
        y: float,
        z: float,
        qx: float,
        qy: float,
        qz: float,
        qw: float,
        speed: float | None = None,
    ) -> dict[str, Any]:
        """
        ---
        description: Move end-effector to an absolute pose with orientation.
        params:
          x: {type: number}
          y: {type: number}
          z: {type: number}
          qx: {type: number}
          qy: {type: number}
          qz: {type: number}
          qw: {type: number}
          speed: {type: number}
        preconditions: {}
        effects:
          current_pose:
            pose_absolute_from_params: [x, y, z]
          current_pose_ref:
            set_unknown: true
        ---
        """
        del qx, qy, qz, qw, speed
        self._position = {"x": float(x), "y": float(y), "z": float(z)}
        self._recovery_pose_ref = None
        return {"success": True, "message": "fake move_pose ok"}

    def move_relative(
        self,
        dx: float,
        dy: float,
        dz: float,
        speed: float | None = None,
    ) -> dict[str, Any]:
        """
        ---
        description: Move end-effector relative to its current position.
        params:
          dx: {type: number}
          dy: {type: number}
          dz: {type: number}
          speed: {type: number}
        preconditions:
          current_pose:
            exists: true
        effects:
          current_pose:
            pose_relative_from_params: [dx, dy, dz]
          current_pose_ref:
            set_unknown: true
        ---
        """
        del speed
        self._position = {
            "x": float(self._position.get("x", 0.0)) + float(dx),
            "y": float(self._position.get("y", 0.0)) + float(dy),
            "z": float(self._position.get("z", 0.0)) + float(dz),
        }
        self._recovery_pose_ref = None
        return {"success": True, "message": "fake move_relative ok"}

    def move_to_named_pose(self, pose_name: str, speed: float | None = None) -> dict[str, Any]:
        """
        ---
        description: Move to a named joint configuration.
        params:
          pose_name: {type: string}
          speed: {type: number}
        preconditions:
          held_part:
            equals: null
        effects:
          current_state:
            set: idle
          current_pose_ref:
            set_from_param: pose_name
          current_pose:
            set_unknown: true
          occupancy.location:
            set_from_param: pose_name
        ---
        """
        del speed
        self._current_state = "idle"
        self._recovery_pose_ref = str(pose_name or "").strip() or None
        return {"success": True, "message": "fake move_to_named_pose ok"}

    def delay(self, duration_sec: float) -> dict[str, Any]:
        """
        ---
        description: Wait intentionally between robot task steps.
        params:
          duration_sec: {type: number}
        preconditions: {}
        effects: {}
        synthesis_hidden: true
        ---
        """
        return {"success": True, "message": f"fake delay ok {float(duration_sec):.3f}"}

    def open_gripper(self) -> bool:
        """
        ---
        description: Open the robot gripper.
        params: {}
        preconditions: {}
        effects:
          gripper_state:
            set: open
        ---
        """
        self._gripper_state = "open"
        return True

    def close_gripper(self, position: float | None = None) -> bool:
        """
        ---
        description: Close the robot gripper.
        params:
          position: {type: number}
        preconditions: {}
        effects:
          gripper_state:
            set: closed
        ---
        """
        del position
        self._gripper_state = "closed"
        return True

    def attach_part(
        self,
        model_name: str,
        link: str | None = None,
        part_name: str = "",
    ) -> dict[str, Any]:
        """
        ---
        description: Attach a part model to the robot gripper.
        params:
          model_name: {type: string}
          link: {type: string}
          part_name: {type: string}
        preconditions:
          held_part:
            equals: null
          gripper_state:
            equals: closed
        effects:
          held_part:
            set_from_param_any_of: [part_name, model_name]
        ---
        """
        del link
        self._held_part = str(part_name or model_name or "").strip() or None
        return {"success": True, "message": "fake attach_part ok"}

    def detach_part(
        self,
        model_name: str = "",
        link: str | None = None,
        assume_released_if_open: bool = False,
    ) -> dict[str, Any]:
        """
        ---
        description: Detach a part model from the robot gripper.
        params:
          model_name: {type: string}
          link: {type: string}
          assume_released_if_open: {type: boolean}
        preconditions:
          held_part:
            not_equals: null
          gripper_state:
            equals: open
        effects:
          held_part:
            set: null
        ---
        """
        del model_name, link, assume_released_if_open
        self._held_part = None
        return {"success": True, "message": "fake detach_part ok"}

    def grasp_part(
        self,
        model_name: str,
        part_name: str = "",
        position: float | None = None,
    ) -> dict[str, Any]:
        """
        ---
        description: Close the gripper and attach the target part.
        params:
          model_name: {type: string}
          part_name: {type: string}
          position: {type: number}
        preconditions:
          held_part:
            equals: null
        effects:
          current_state:
            set: picked
          gripper_state:
            set: closed
          held_part:
            set_from_param_any_of: [part_name, model_name]
        ---
        """
        self.close_gripper(position=position)
        self.attach_part(model_name=model_name, part_name=part_name)
        self._current_state = "picked"
        return {"success": True, "message": "fake grasp_part ok"}

    def release_part(
        self,
        model_name: str = "",
        part_name: str = "",
        assume_released_if_open: bool = False,
    ) -> dict[str, Any]:
        """
        ---
        description: Open the gripper and detach the currently held part.
        params:
          model_name: {type: string}
          part_name: {type: string}
          assume_released_if_open: {type: boolean}
        preconditions:
          held_part:
            not_equals: null
        effects:
          current_state:
            set: idle
          gripper_state:
            set: open
          held_part:
            set: null
        ---
        """
        del assume_released_if_open
        self.open_gripper()
        self.detach_part(model_name=model_name)
        self._current_state = "idle"
        return {"success": True, "message": f"fake release_part ok {part_name or model_name}"}

    def get_current_pose(self) -> dict[str, Any]:
        """
        ---
        description: Return the current end-effector pose.
        params: {}
        preconditions: {}
        effects:
          current_pose_ref:
            set_unknown: true
        ---
        """
        return {
            "success": True,
            "message": "fake get_current_pose ok",
            "pose": {
                "x": float(self._position.get("x", 0.0)),
                "y": float(self._position.get("y", 0.0)),
                "z": float(self._position.get("z", 0.0)),
                "qx": 0.0,
                "qy": 0.0,
                "qz": 0.0,
                "qw": 1.0,
            },
        }

    def compute_pick_targets(
        self,
        part_name: str = "",
        target_pose: dict[str, Any] | None = None,
        target_pose_source: str = "",
        approach_height_override_m: float | None = None,
        surface_clearance_override_m: float | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        """
        ---
        description: Compute pick target positions from perception and geometry.
        params:
          part_name: {type: string}
          target_pose: {type: object}
          target_pose_source: {type: string}
          approach_height_override_m: {type: number}
          surface_clearance_override_m: {type: number}
        preconditions: {}
        effects: {}
        ---
        """
        target = dict(target_pose or {})
        pose = {
            "x": float(target.get("x", 0.0) or 0.0),
            "y": float(target.get("y", 0.0) or 0.0),
            "z": float(target.get("z", 1.0) or 1.0),
        }
        surface_clearance = float(surface_clearance_override_m or 0.0)
        approach_height = float(approach_height_override_m or 0.2)
        return {
            "success": True,
            "part_name": str(part_name or target.get("part_name") or ""),
            "model_name": "fake_model",
            "approach_pose": {"x": pose["x"], "y": pose["y"], "z": pose["z"] + approach_height},
            "target_pose": {
                "x": pose["x"],
                "y": pose["y"],
                "z": pose["z"] + 0.02 + surface_clearance,
            },
            "target_pose_source": target_pose_source,
        }

    def compute_place_targets(
        self,
        pick_ctx: dict[str, Any] | None = None,
        part_name: str = "",
        z_adjustment_m: float = 0.0,
        destination_location: str = "",
        **_kwargs: Any,
    ) -> dict[str, Any]:
        """
        ---
        description: Compute placement target positions from context and geometry.
        params:
          pick_ctx: {type: object}
          part_name: {type: string}
          z_adjustment_m: {type: number}
          destination_location: {type: string}
        preconditions: {}
        effects: {}
        ---
        """
        del destination_location
        pick = dict(pick_ctx or {})
        x = float((pick.get("target_pose") or {}).get("x", 0.0) or 0.0)
        y = float((pick.get("target_pose") or {}).get("y", 0.0) or 0.0)
        z = 1.05 + float(z_adjustment_m or 0.0)
        return {
            "success": True,
            "part_name": str(part_name or pick.get("part_name") or ""),
            "model_name": "fake_model",
            "approach_pose": {"x": x, "y": y, "z": z + 0.1},
            "target_pose": {"x": x, "y": y, "z": z},
        }

    def detect_parts(self, part_name: str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        """
        ---
        description: Detect parts via perception service.
        params:
          part_name: {type: string}
        preconditions: {}
        effects: {}
        ---
        """
        if not part_name:
            part_name = kwargs.get("filter_part_name")
        catalog = self._observation_catalog()
        if part_name:
            observation = deepcopy(catalog.get(str(part_name).strip()) or {})
            return [observation] if observation else []
        return [deepcopy(item) for item in catalog.values()]

    def get_recovery_snapshot(self) -> dict[str, Any]:
        snapshot = get_resource_recovery_snapshot(self)
        if self._current_location not in (None, ""):
            snapshot["current_location"] = self._current_location
            snapshot["resource_location"] = self._current_location
        if self._occupancy:
            snapshot["occupancy"] = deepcopy(self._occupancy)
        return snapshot

    def _snapshot_state(self) -> dict[str, Any]:
        return {
            "resource_jid": self.jid,
            "resource_type": "robot",
            "current_state": self._current_state,
            "current_pose": deepcopy(self._position),
            "current_pose_ref": self._recovery_pose_ref,
            "current_location": self._current_location,
            "occupancy": deepcopy(self._occupancy),
            "held_part": self._held_part,
            "gripper_state": self._gripper_state,
        }

    async def _ensure_controller_prewarmed(self) -> None:
        return None

    recovery_physical_validation_snapshot = staticmethod(
        RobotAgent.recovery_physical_validation_snapshot
    )
    check_recovery_physical_feasibility = RobotAgent.check_recovery_physical_feasibility
    check_recovery_primitive_feasibility = RobotAgent.check_recovery_primitive_feasibility

    async def execute_recovery_observation(
        self,
        primitive: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = deepcopy(params or {})
        primitive_name = str(primitive or "").strip()
        if primitive_name == "get_current_pose":
            return {
                "success": True,
                "message": "fake get_current_pose succeeded",
                "primitive": primitive_name,
                "params": payload,
                "observation": {"pose": deepcopy(self._position), "resource_jid": self.jid},
                "snapshot": self.get_recovery_snapshot(),
            }
        if primitive_name != "detect_parts":
            return {
                "success": False,
                "message": f"unsupported observation primitive {primitive_name!r}",
                "snapshot": self.get_recovery_snapshot(),
            }
        part_name = str(payload.get("part_name") or "").strip()
        if not part_name:
            for alt_key in ("part_names", "targets"):
                alt = payload.get(alt_key)
                if isinstance(alt, list) and len(alt) == 1 and str(alt[0] or "").strip():
                    part_name = str(alt[0]).strip()
                    break
        catalog = self._observation_catalog()
        if not part_name and len(catalog) == 1:
            part_name = next(iter(catalog.keys()))
        observation = deepcopy(catalog.get(part_name) or {})
        if not observation:
            return {
                "success": False,
                "message": f"no observation configured for part {part_name!r}",
                "snapshot": self.get_recovery_snapshot(),
            }
        return {
            "success": True,
            "message": f"fake detect_parts succeeded for {part_name}",
            "primitive": primitive_name,
            "params": payload,
            "observation": observation,
            "snapshot": self.get_recovery_snapshot(),
        }


class SnapshotRecoveryRobot(FakeRecoveryRobot):
    """Read saved observations and canonical robot contracts without dispatch."""

    _RECOVERY_PRIMITIVES = RobotAgent._RECOVERY_PRIMITIVES
    _RESOURCE_PROFILE = RobotAgent._RESOURCE_PROFILE

    def recovery_execution_primitive_catalog(self) -> list[dict[str, Any]]:
        """Use controller-class metadata exactly as the runtime catalog does."""
        from cais_spade_llm.resources.resource_primitives import build_execution_primitive_catalog

        metadata = SimpleNamespace(
            agent_name=self.agent_name,
            static_capabilities=self.static_capabilities,
            _RECOVERY_PRIMITIVES=self._RECOVERY_PRIMITIVES,
            _RESOURCE_PROFILE=self._RESOURCE_PROFILE,
        )
        return build_execution_primitive_catalog(metadata)

    def get_recovery_snapshot(self) -> dict[str, Any]:
        """Overlay explicitly saved physical evidence on the resource view."""
        snapshot = super().get_recovery_snapshot()
        saved = deepcopy(getattr(self, "_saved_snapshot", {}))
        for key in (
            "controller_ready",
            "tf_ready",
            "tcp_ready",
            "perception_ready",
            "destination_localization_ready",
            "current_pose_captured_at",
            "physical_evidence",
        ):
            snapshot[key] = saved.get(key)
        snapshot["evidence_source"] = saved.get("evidence_source", "synthetic")
        return snapshot

    def __getattribute__(self, name: str) -> Any:
        value = super().__getattribute__(name)
        if name in RobotAgent._RECOVERY_PRIMITIVES:

            @wraps(value)
            def no_dispatch(*args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("Equipment dispatch is disabled in recovery diagnostics")

            return no_dispatch
        return value


def _task_node_by_id(plan_payload: dict[str, Any], task_id: str) -> dict[str, Any]:
    for node in plan_payload.get("nodes") or []:
        if isinstance(node, dict) and str(node.get("id") or "").strip() == task_id:
            return deepcopy(node)
    return {}


def _apply_runtime_status_snapshot(
    plan_nodes: list[dict[str, Any]],
    task_statuses: dict[str, Any],
) -> None:
    for node in plan_nodes:
        if not isinstance(node, dict) or str(node.get("type") or "").strip() != "task":
            continue
        node_id = str(node.get("id") or "").strip()
        configured_status = str(task_statuses.get(node_id) or "").strip()
        if configured_status:
            node["status"] = configured_status


def _merge_part_tracker(
    base_part_tracker: dict[str, Any],
    derived_part_tracker: dict[str, Any],
) -> dict[str, Any]:
    """Fill missing supplied tracker facts from the derived failure facts."""
    merged_part_tracker = deepcopy(base_part_tracker)
    for part_name, derived_entry in (derived_part_tracker or {}).items():
        if not isinstance(derived_entry, dict):
            continue
        current_entry = dict(merged_part_tracker.get(part_name) or {})
        for key, value in derived_entry.items():
            if key == "_force_keys":
                continue
            if value in (None, "", [], {}):
                continue
            if current_entry.get(key) not in (None, "", [], {}, "unknown"):
                continue
            current_entry[key] = deepcopy(value)
        merged_part_tracker[str(part_name)] = current_entry
    return merged_part_tracker


def _build_live_style_failure_payload(
    plan_payload: dict[str, Any],
    *,
    runtime_context: dict[str, Any],
) -> dict[str, Any]:
    failure_event = dict(runtime_context.get("failure_event") or {})
    failed_task_id = str(failure_event.get("failed_task_id") or "").strip()
    if not failed_task_id:
        raise ValueError("runtime context failure_event.failed_task_id is empty")
    failed_task = _task_node_by_id(plan_payload, failed_task_id)
    if not failed_task:
        raise KeyError(f"runtime context failed task {failed_task_id!r} is not in the plan")
    failed_resource_jid = str(failed_task.get("resource_jid") or "").strip()
    failed_function_name = str(failed_task.get("function_name") or "").strip()
    scenario_id = str(runtime_context.get("failure_scenario_id") or "").strip()
    scenario_config = _load_json(
        DATA_ROOT / "cais_spade_llm/initialization/failure_scenarios" / f"{scenario_id}.json"
    )
    base_context = failure_context_from_scenario_config(scenario_config)
    # Injection targets/templates describe the configured fault, not observations.
    base_context.pop("observations", None)
    observations = {
        "last_commanded_location": str(
            dict(failed_task.get("params") or {}).get("destination_location") or ""
        ).strip(),
    }
    observations.update(deepcopy(dict(failure_event.get("observations") or {})))
    return build_failure_event(
        failed_task_id=failed_task_id,
        failed_resource_jid=failed_resource_jid,
        failed_function_name=failed_function_name,
        final_status=str(failure_event.get("final_status") or "failed").strip(),
        part_name=str(failure_event.get("part_name") or "").strip(),
        base_failure_context=base_context,
        observations=observations,
        state_before=deepcopy(dict(failure_event.get("state_before") or {})),
        state_after=deepcopy(dict(failure_event.get("state_after") or {})),
    )


def _build_live_style_slippage_fixture(
    *,
    planner: ProcessPlanner,
    plan_payload: dict[str, Any],
    runtime_context: dict[str, Any],
) -> dict[str, Any]:
    failure_payload = _build_live_style_failure_payload(
        plan_payload,
        runtime_context=runtime_context,
    )
    base_part_tracker = deepcopy(dict(runtime_context.get("part_tracker") or {}))
    if not base_part_tracker:
        raise ValueError("runtime context part_tracker is empty")
    derived_part_tracker = planner._derive_part_tracker_from_violations([failure_payload])
    part_tracker = _merge_part_tracker(base_part_tracker, derived_part_tracker)
    part_tracker = {name: part_tracker[name] for name in sorted(part_tracker)}
    part_states = {
        str(name): info.get("state")
        for name, info in part_tracker.items()
        if isinstance(info, dict)
    }
    part_locations = {
        str(name): info.get("location")
        for name, info in part_tracker.items()
        if isinstance(info, dict)
    }
    resource_states: dict[str, dict[str, Any]] = {}
    for raw_snapshot in sorted(
        (row for row in (runtime_context.get("resource_snapshots") or []) if isinstance(row, dict)),
        key=lambda row: str(row.get("resource_jid") or ""),
    ):
        resource_jid = str(raw_snapshot.get("resource_jid") or "").strip()
        if not resource_jid:
            continue
        resource_states[resource_jid] = {
            "current_state": deepcopy(raw_snapshot.get("current_state")),
            "held_part": deepcopy(raw_snapshot.get("held_part")),
            "current_location": deepcopy(raw_snapshot.get("current_location")),
        }
    failed_resource_jid = str(failure_payload.get("failed_resource_jid") or "").strip()
    if failed_resource_jid not in resource_states:
        raise KeyError(
            f"runtime context has no resource snapshot for failed resource {failed_resource_jid!r}"
        )
    default_resource_state = str(runtime_context.get("default_resource_state") or "idle").strip()
    goal_state = str(runtime_context.get("goal_state") or "").strip()
    if not goal_state:
        raise ValueError("runtime context goal_state is empty")
    stuck_state = planner._build_resource_search_state(
        resource_jid=failed_resource_jid,
        resource_states=resource_states,
        default_resource_state=default_resource_state,
        part_states=part_states,
        part_locations=part_locations,
    )
    return {
        "failed_task_id": str(failure_payload.get("failed_task_id") or "").strip(),
        "failed_resource_jid": failed_resource_jid,
        "goal_state": goal_state,
        "P_id": [name for name in sorted(part_states) if part_states[name] != goal_state],
        "obligation_targets": deepcopy(runtime_context.get("obligation_targets") or []),
        "recovery_feedback": str(runtime_context.get("recovery_feedback") or ""),
        "default_resource_state": default_resource_state,
        "part_tracker": part_tracker,
        "part_states": part_states,
        "part_locations": part_locations,
        "resource_states": resource_states,
        "stuck_state": stuck_state,
        "recovery_safety_context": deepcopy(
            dict(runtime_context.get("recovery_safety_context") or {})
        ),
        "failure_context": failure_payload,
    }


def _normalized_observation_pose(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    pose: dict[str, Any] = {}
    nested_pose = payload.get("pose")
    if isinstance(nested_pose, dict):
        pose.update(deepcopy(nested_pose))
    for axis in ("x", "y", "z", "qx", "qy", "qz", "qw"):
        if payload.get(axis) is not None:
            pose[axis] = deepcopy(payload.get(axis))
    if any(axis not in pose for axis in ("x", "y", "z")):
        return None
    return {axis: float(pose[axis]) for axis in ("x", "y", "z")}


def _holder_resource_jid_for_part_row(
    part_row: dict[str, Any],
    *,
    resource_jids: set[str],
) -> str:
    holder_resource_jid = str(part_row.get("current_holder_resource_jid") or "").strip()
    if holder_resource_jid:
        return holder_resource_jid
    current_location = str(part_row.get("current_location") or "").strip()
    if current_location in resource_jids:
        return current_location
    return ""


def _build_shared_grounding_observation_catalog(
    *,
    prepared_recovery_request: dict[str, Any],
    robots: list[FakeRecoveryRobot],
) -> dict[str, dict[str, Any]]:
    llm_input = dict(prepared_recovery_request.get("llm_input") or {})
    robots_by_jid = {robot.jid: robot for robot in robots}
    explicit_observations_by_part: dict[str, dict[str, Any]] = {}
    for robot in robots:
        for part_name, observation in robot._observations.items():
            if part_name not in explicit_observations_by_part:
                explicit_observations_by_part[part_name] = deepcopy(observation)

    shared_catalog: dict[str, dict[str, Any]] = {}
    for part_row in llm_input.get("part_facts") or []:
        if not isinstance(part_row, dict):
            continue
        part_name = str(part_row.get("part_name") or "").strip()
        if not part_name:
            continue
        observation = deepcopy(explicit_observations_by_part.get(part_name) or {})
        normalized_pose = _normalized_observation_pose(observation)
        if normalized_pose is None:
            normalized_pose = _normalized_observation_pose(part_row.get("observed_pose"))
        holder_resource_jid = _holder_resource_jid_for_part_row(
            part_row,
            resource_jids=set(robots_by_jid),
        )
        if normalized_pose is None:
            continue
        observation.setdefault("evidence_source", "synthetic")
        observation["part_name"] = part_name
        observation["x"] = normalized_pose["x"]
        observation["y"] = normalized_pose["y"]
        observation["z"] = normalized_pose["z"]
        observation["pose"] = deepcopy(normalized_pose)
        if part_row.get("current_location"):
            observation["current_location"] = deepcopy(part_row.get("current_location"))
        if holder_resource_jid:
            observation["current_holder_resource_jid"] = holder_resource_jid
        shared_catalog[part_name] = observation
    return shared_catalog


def _configure_live_recovery_session(
    prepared_recovery_request: dict[str, Any],
    *,
    experiment_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    recovery_session = dict(prepared_recovery_request.get("recovery_session") or {})
    settings = _load_recovery_outline_experiment_settings(experiment_settings)
    recovery_session["reasoning_mode"] = "multi_turn"
    recovery_session["max_turns"] = max(int(recovery_session.get("max_turns", 6) or 6), 1000)
    recovery_session["repair_mode"] = "recover"
    recovery_session["observation_backend"] = "mock_detect_parts_harness"
    recovery_session["outline_mode"] = "incremental_candidates_validated"
    if settings.get("enabled", True):
        recovery_session["recovery_selection_mode"] = settings["recovery_selection_mode"]
        recovery_session.update(_recovery_action_horizon_fields(settings["action_horizon"]))
        recovery_session.update(_recovery_candidate_count_fields(settings["candidate_count"]))
        recovery_session["candidate_proposal_budget"] = int(settings["candidate_proposal_budget"])
        if settings["recovery_selection_mode"] == "neurosymbolic":
            recovery_session["candidate_bound"] = int(settings["candidate_proposal_budget"])
    prepared_recovery_request["recovery_session"] = recovery_session
    prepared_recovery_request["recovery_outline_experiment_settings"] = deepcopy(settings)
    return settings


async def _prepare_recovery_dryrun_harness(
    *,
    debug_root: Path | None = None,
    scripted_responses: list[dict[str, Any]] | None = None,
    experiment_settings: dict[str, Any] | None = None,
    runtime_context_path: Path | None = None,
) -> tuple[dict[str, Any], FakeProductAgent, ProcessPlanner, dict[str, Any]]:
    runtime_context = _load_case3_runtime_context(runtime_context_path)
    paths = _case3_paths(runtime_context)
    tools_catalog = _load_json(paths["tools"])
    plan_payload = _load_json(paths["plan"])
    geometry_payload = _load_json(paths["geometry"])
    if not isinstance(tools_catalog, list):
        raise TypeError("tools catalog did not decode to a list")
    if not isinstance(plan_payload, dict):
        raise TypeError("case3 plan did not decode to an object")
    if not isinstance(geometry_payload, dict):
        raise TypeError("case3 geometry did not decode to an object")

    product_agent = FakeProductAgent(
        tools_catalog=tools_catalog,
        product_geometry=deepcopy(geometry_payload.get("gazebo") or {}),
        precomputed_bundle=_case3_bundle_context(paths),
        scripted_responses=scripted_responses,
    )
    product_agent._saved_safety_context = deepcopy(
        runtime_context.get("recovery_safety_context") or {}
    )
    robots: list[FakeRecoveryRobot] = []
    for raw_snapshot in sorted(
        (row for row in (runtime_context.get("resource_snapshots") or []) if isinstance(row, dict)),
        key=lambda row: str(row.get("resource_jid") or ""),
    ):
        resource_jid = str(raw_snapshot.get("resource_jid") or "").strip()
        config_path = _repo_path(raw_snapshot.get("resource_config"))
        config_key = str(raw_snapshot.get("resource_config_key") or "").strip()
        execution_env = str(raw_snapshot.get("execution_env") or "").strip()
        if not resource_jid or not config_key or not execution_env:
            raise ValueError("runtime context resource snapshot is missing identity fields")
        config = _load_robot_config(config_path, config_key)
        robot_class = FakeRecoveryRobot if scripted_responses is not None else SnapshotRecoveryRobot
        robot = robot_class(
            config=config,
            execution_env=execution_env,
            current_state=str(raw_snapshot.get("current_state") or "").strip(),
            held_part=(
                str(raw_snapshot.get("held_part") or "").strip()
                if raw_snapshot.get("held_part") not in (None, "")
                else None
            ),
            gripper_state=str(raw_snapshot.get("gripper_state") or "").strip(),
            pose_ref=(
                str(raw_snapshot.get("current_pose_ref") or "").strip()
                if raw_snapshot.get("current_pose_ref") not in (None, "")
                else None
            ),
            position=deepcopy(dict(raw_snapshot.get("current_pose") or {})),
            current_location=(
                str(raw_snapshot.get("current_location") or "").strip()
                if raw_snapshot.get("current_location") not in (None, "")
                else None
            ),
            occupancy=deepcopy(dict(raw_snapshot.get("occupancy") or {})),
            observations=deepcopy(dict(raw_snapshot.get("observations") or {})),
        )
        if robot.jid != resource_jid:
            raise ValueError(
                f"resource config jid {robot.jid!r} does not match runtime context {resource_jid!r}"
            )
        robot._saved_snapshot = deepcopy(raw_snapshot)
        robots.append(robot)
    if not robots:
        raise ValueError("runtime context resource_snapshots is empty")

    planner = ProcessPlannerPrepareTrace(product_agent, robots)
    product_agent._mock_recovery_validation_resources = {robot.jid: robot for robot in robots}
    planner.nodes = deepcopy(plan_payload.get("nodes") or [])
    _apply_runtime_status_snapshot(
        planner.nodes,
        dict(runtime_context.get("task_statuses") or {}),
    )

    fixture = _build_live_style_slippage_fixture(
        planner=planner,
        plan_payload=plan_payload,
        runtime_context=runtime_context,
    )

    prepared_recovery_request = await planner.prepare_recovery_request(
        stuck_state=deepcopy(fixture["stuck_state"]),
        P_id=deepcopy(fixture["P_id"]),
        ra_jid=str(fixture["failed_resource_jid"]),
        goal_state=str(fixture["goal_state"]),
        tools_catalog=deepcopy(tools_catalog),
        part_tracker=deepcopy(fixture["part_tracker"]),
        obligation_targets=deepcopy(fixture["obligation_targets"]),
        recovery_feedback=str(fixture["recovery_feedback"]),
        resource_states=deepcopy(fixture["resource_states"]),
        default_resource_state=str(fixture["default_resource_state"]),
        part_states=deepcopy(fixture["part_states"]),
        part_locations=deepcopy(fixture["part_locations"]),
        recovery_safety_context=deepcopy(fixture.get("recovery_safety_context") or {}),
        failure_context=deepcopy(fixture.get("failure_context") or {}),
    )

    shared_observations = _build_shared_grounding_observation_catalog(
        prepared_recovery_request=prepared_recovery_request,
        robots=robots,
    )
    for robot in robots:
        robot.set_shared_observations(shared_observations)
    _configure_live_recovery_session(
        prepared_recovery_request,
        experiment_settings=experiment_settings,
    )
    prepared_recovery_request["multi_turn_session_seed"] = (
        multi_turn_mode.build_multi_turn_session_seed(prepared_recovery_request)
    )

    artifact_root = _debug_root(debug_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    recovery_debug = dict(prepared_recovery_request.get("recovery_debug") or {})
    recovery_debug["artifact_directory"] = str(artifact_root)
    recovery_debug["per_turn_debug_dir"] = str(artifact_root)
    prepared_recovery_request["recovery_debug"] = recovery_debug
    return fixture, product_agent, planner, prepared_recovery_request


def _outline_trace_from_session_state(session_state: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(session_state, dict):
        return []
    for rows in (
        session_state.get("accepted_outline_prefix"),
        session_state.get("transition_trace"),
        dict(session_state.get("final_output") or {}).get("transition_trace"),
    ):
        trace = [deepcopy(row) for row in (rows or []) if isinstance(row, dict)]
        if trace:
            return trace
    return []


def _latest_session(
    prepared_recovery_request: dict[str, Any], planner: ProcessPlanner
) -> dict[str, Any]:
    recovery_debug = (
        deepcopy(planner.get_last_recovery_debug())
        if callable(getattr(planner, "get_last_recovery_debug", None))
        else {}
    )
    for candidate in (
        recovery_debug.get("multi_turn_session"),
        dict(prepared_recovery_request.get("recovery_debug") or {}).get("multi_turn_session"),
        prepared_recovery_request.get("multi_turn_session_state"),
        prepared_recovery_request.get("multi_turn_session_seed"),
    ):
        if isinstance(candidate, dict):
            return deepcopy(candidate)
    return {}


def _primitive_program_from_session(session_state: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(session_state, dict):
        return []
    for rows in (
        session_state.get("accepted_primitive_program"),
        dict(session_state.get("final_output") or {}).get("accepted_primitive_program"),
    ):
        program = [deepcopy(row) for row in (rows or []) if isinstance(row, dict)]
        if program:
            return program
    return []


async def _execute_recovery_until(
    planner: ProcessPlanner,
    prepared_recovery_request: dict[str, Any],
    *,
    stop_after: str,
) -> dict[str, Any] | None:
    target = str(stop_after or "").strip().lower()
    prepared_recovery_request["_stop_after_multi_turn_phase"] = (
        target if target in {"outline", "primitive"} else ""
    )
    session_state = dict(prepared_recovery_request.get("multi_turn_session_state") or {})
    if session_state:
        proposal = await multi_turn_mode.execute_multi_turn_recovery(
            planner,
            prepared_recovery_request,
            session_state=session_state,
        )
    else:
        proposal = await planner.execute_prepared_recovery_request(prepared_recovery_request)
    max_resume = max(
        10,
        int(
            dict(prepared_recovery_request.get("recovery_session") or {}).get("max_turns")
            or dict(prepared_recovery_request.get("multi_turn_session_seed") or {}).get("max_turns")
            or 0
        ),
    )
    for _resume_index in range(max_resume):
        session_state = _latest_session(prepared_recovery_request, planner)
        pause_status = str(session_state.get("status") or "").strip().lower()
        current_phase = str(session_state.get("current_phase") or "").strip().lower()
        if target == "outline":
            if _outline_trace_from_session_state(session_state) and current_phase in {
                "primitive_generation",
                "finalize",
            }:
                break
            resume_needed = pause_status == "paused_after_outline_turn"
        elif target == "primitive":
            if _primitive_ready_payload(prepared_recovery_request, session_state):
                break
            resume_needed = pause_status in {
                "paused_after_outline_turn",
                "ready_for_primitive_generation",
                "paused_after_primitive_turn",
            }
        else:
            resume_needed = pause_status in {
                "paused_after_outline_turn",
                "ready_for_primitive_generation",
                "paused_after_primitive_turn",
            }
        if not resume_needed:
            break
        proposal = await multi_turn_mode.execute_multi_turn_recovery(
            planner,
            prepared_recovery_request,
            session_state=session_state,
        )
    prepared_recovery_request["_stop_after_multi_turn_phase"] = ""
    return proposal


def _build_recovery_safety_payload(
    *,
    prepared_recovery_request: dict[str, Any],
    session_state: dict[str, Any],
    debug_root: Path | None = None,
) -> dict[str, Any]:
    def _nominal_candidate_tasks(
        *,
        pending_nominal_tasks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        pending_by_id = {
            str(row.get("id") or row.get("task_id") or "").strip(): deepcopy(row)
            for row in pending_nominal_tasks
            if str(row.get("id") or row.get("task_id") or "").strip()
        }
        requirement_task_index = dict(prepared_recovery_request.get("requirement_task_index") or {})
        task_requirement_map = dict(prepared_recovery_request.get("task_requirement_map") or {})
        recovery_resources = dict(prepared_recovery_request.get("recovery_resources") or {})
        pending_task_rows_by_id: dict[str, dict[str, Any]] = {}
        active_requirement_ids: set[str] = set()

        for resource_entry in recovery_resources.values():
            if not isinstance(resource_entry, dict):
                continue
            for task in resource_entry.get("pending_tasks") or []:
                if not isinstance(task, dict):
                    continue
                task_id = str(task.get("id") or "").strip()
                if not task_id:
                    continue
                pending_task_rows_by_id[task_id] = deepcopy(task)
                requirement_id = str(
                    task.get("requirement_id") or task_requirement_map.get(task_id) or ""
                ).strip()
                if requirement_id:
                    active_requirement_ids.add(requirement_id)

        for task_id in pending_by_id:
            requirement_id = str(task_requirement_map.get(task_id) or "").strip()
            if requirement_id:
                active_requirement_ids.add(requirement_id)

        candidate_rows: list[dict[str, Any]] = []
        seen_task_ids: set[str] = set()
        for requirement_id in sorted(active_requirement_ids):
            for raw_task in requirement_task_index.get(requirement_id) or []:
                if not isinstance(raw_task, dict):
                    continue
                task_id = str(raw_task.get("task_id") or raw_task.get("id") or "").strip()
                if not task_id or task_id in seen_task_ids:
                    continue
                seen_task_ids.add(task_id)
                enriched = dict(pending_task_rows_by_id.get(task_id) or {})
                pending_row = dict(pending_by_id.get(task_id) or {})
                params = dict(enriched.get("params") or raw_task.get("params") or {})
                candidate_rows.append(
                    {
                        "id": task_id,
                        "function": str(
                            raw_task.get("function_name")
                            or enriched.get("function_name")
                            or pending_row.get("function")
                            or ""
                        ).strip(),
                        "resource": str(
                            raw_task.get("resource_jid")
                            or enriched.get("resource_jid")
                            or pending_row.get("resource")
                            or ""
                        ).strip(),
                        "part": str(
                            raw_task.get("part_name")
                            or params.get("part_name")
                            or pending_row.get("part")
                            or ""
                        ).strip(),
                        "status": str(
                            raw_task.get("status")
                            or enriched.get("status")
                            or pending_row.get("status")
                            or ""
                        ).strip(),
                        "in_state": str(
                            raw_task.get("in_state")
                            or enriched.get("in_state")
                            or pending_row.get("in_state")
                            or ""
                        ).strip(),
                        "out_state": str(
                            raw_task.get("out_state")
                            or enriched.get("out_state")
                            or pending_row.get("out_state")
                            or ""
                        ).strip(),
                        "requirement_id": requirement_id,
                        "sequence_index": int(raw_task.get("sequence_index") or 0),
                        "product_jid": str(params.get("product_jid") or "").strip(),
                        "destination_location": str(
                            params.get("destination_location") or ""
                        ).strip(),
                        "blocked_by_condition_ids": [
                            str(token).strip()
                            for token in (pending_row.get("blocked_by_condition_ids") or [])
                            if str(token).strip()
                        ],
                        "expected_start_state": deepcopy(
                            pending_row.get("expected_start_state") or {}
                        ),
                        "expected_end_state": deepcopy(pending_row.get("expected_end_state") or {}),
                        "projected_outline_state": deepcopy(
                            pending_row.get("projected_outline_state") or {}
                        ),
                    }
                )
        return candidate_rows

    accepted_outline_prefix = _outline_trace_from_session_state(session_state)
    if not accepted_outline_prefix:
        return {}
    projected_outline_state = deepcopy(
        accepted_outline_prefix[-1].get("projected_outline_state")
        or session_state.get("projected_outline_state")
        or {}
    )
    pending_nominal_tasks = _private_pending_nominal_tasks(prepared_recovery_request)
    pending_nominal_task_ids = [
        str(row.get("id") or "").strip()
        for row in pending_nominal_tasks
        if str(row.get("id") or "").strip()
    ]
    nominal_candidate_tasks = _nominal_candidate_tasks(
        pending_nominal_tasks=pending_nominal_tasks,
    )
    recovery_safety_dir = _debug_root(debug_root) / "recovery_safety"
    return {
        "product_jid": "assembly_board-v1@localhost",
        "recovery_safety_scope_id": str(
            (prepared_recovery_request.get("recovery_session") or {}).get("session_id")
            or "dryrun_recovery_scope"
        ),
        "accepted_outline_prefix": accepted_outline_prefix,
        "projected_outline_state": projected_outline_state,
        "pending_nominal_tasks": pending_nominal_tasks,
        "pending_nominal_task_ids": pending_nominal_task_ids,
        "nominal_candidate_tasks": nominal_candidate_tasks,
        "nominal_candidate_task_ids": [
            str(row.get("id") or "").strip()
            for row in nominal_candidate_tasks
            if str(row.get("id") or "").strip()
        ],
        "loaded_safety_rules": [
            deepcopy(rule)
            for rule in (prepared_recovery_request.get("loaded_safety_rules") or [])
            if isinstance(rule, dict)
        ],
        "recovery_safety_context": deepcopy(
            prepared_recovery_request.get("recovery_safety_context") or {}
        ),
        "recovery_safety_dir": str(recovery_safety_dir),
        "recovery_plan_dir": str(recovery_safety_dir),
        "recovery_safery_dir": str(recovery_safety_dir),
        "tools_catalog": deepcopy(prepared_recovery_request.get("tools_catalog") or []),
    }


async def _run_safety_from_outline(
    *,
    product_agent: FakeProductAgent,
    prepared_recovery_request: dict[str, Any],
    session_state: dict[str, Any],
    debug_root: Path | None = None,
) -> dict[str, Any]:
    payload = _build_recovery_safety_payload(
        prepared_recovery_request=prepared_recovery_request,
        session_state=session_state,
        debug_root=debug_root,
    )
    if not payload:
        raise AssertionError("safety synthsis requires an accepted recovery outline")
    return await generate_recovery_safety_bundle(product_agent, payload)


def _primitive_ready_payload(
    prepared_recovery_request: dict[str, Any],
    session_state: dict[str, Any],
) -> dict[str, Any]:
    for candidate in (
        dict(session_state.get("final_output") or {}),
        dict(prepared_recovery_request.get("recovery_debug") or {}).get("final_output"),
    ):
        if not isinstance(candidate, dict):
            continue
        if str(candidate.get("final_output_stage") or "").strip() == "primitive_program_ready":
            return deepcopy(candidate)
    return {}


def _primitive_ready_source_path(session_state: dict[str, Any]) -> str:
    for turn in reversed(
        [row for row in (session_state.get("turns") or []) if isinstance(row, dict)]
    ):
        if str(turn.get("phase") or "").strip().lower() != "final_output":
            continue
        if str(turn.get("final_output_stage") or "").strip() != "primitive_program_ready":
            continue
        source = str(turn.get("response_artifact_path") or "").strip()
        if source:
            return source
    return ""


def _write_recovery_final_bundle(
    *,
    prepared_recovery_request: dict[str, Any],
    session_state: dict[str, Any],
    recovery_safety_generation: dict[str, Any],
    debug_root: Path | None = None,
) -> dict[str, str]:
    if not recovery_safety_generation.get("ok"):
        return {}
    primitive_payload = _primitive_ready_payload(prepared_recovery_request, session_state)
    if not primitive_payload:
        return {}
    recovery_safety_logic_json = str(
        recovery_safety_generation.get("recovery_safety_logic_json") or ""
    ).strip()
    if not recovery_safety_logic_json:
        return {}
    recovery_safety_logic_path = Path(recovery_safety_logic_json)
    if not recovery_safety_logic_path.exists():
        return {}

    recovery_final_dir = _debug_root(debug_root) / "recovery_final"
    recovery_final_dir.mkdir(parents=True, exist_ok=True)
    source_final_output_path = _primitive_ready_source_path(session_state)
    if source_final_output_path and Path(source_final_output_path).exists():
        source_path = Path(source_final_output_path)
        target_final_output_path = recovery_final_dir / source_path.name
        if source_path.resolve() != target_final_output_path.resolve():
            shutil.copy2(source_path, target_final_output_path)
    else:
        target_final_output_path = (
            recovery_final_dir / f"multi_turn_final_output_response_{_utc_token()}.txt"
        )
        _write_json(target_final_output_path, primitive_payload)

    target_logic_path = recovery_final_dir / "cca_safety_logic.json"
    shutil.copy2(recovery_safety_logic_path, target_logic_path)
    for source_dfa_path in sorted(recovery_safety_logic_path.parent.glob("*_dfa.dot")):
        shutil.copy2(source_dfa_path, recovery_final_dir / source_dfa_path.name)
    return {
        "recovery_safety_logic_json": str(target_logic_path.resolve()),
        "recovery_final_dir": str(recovery_final_dir),
        "recovery_final_output_path": str(target_final_output_path.resolve()),
    }


async def _run_actual_recovery(
    *,
    mode: str,
    debug_root: Path | None = None,
    scripted_responses: list[dict[str, Any]] | None = None,
    experiment_settings: dict[str, Any] | None = None,
    runtime_context_path: Path | None = None,
    outline_checkpoint: Path | None = None,
) -> dict[str, Any]:
    from cais_spade_llm.recovery_framework.diagnostics import (
        inspect_inputs,
        load_outline_checkpoint,
        validator_fingerprint,
        write_record,
    )

    if mode not in {"outline", "primitive", "safety", "full"}:
        raise ValueError("Unsupported recovery test mode")
    if outline_checkpoint and mode == "outline":
        raise ValueError("Select primitive, safety, or full to reuse an outline")
    runtime_context = _load_case3_runtime_context(runtime_context_path)
    inputs = inspect_inputs(
        runtime_context_path or CASE3_RUNTIME_CONTEXT,
        root=DATA_ROOT,
        response_source="fixture_response_replay" if scripted_responses is not None else "live",
    )
    if debug_root is not None and (debug_root / "inputs.json").is_file():
        recorded_inputs = _load_json(debug_root / "inputs.json")
        if recorded_inputs.get("fingerprint") != inputs["fingerprint"]:
            raise ValueError("Saved inputs changed after test selection; select the inputs again")
    checkpoint = load_outline_checkpoint(outline_checkpoint, inputs) if outline_checkpoint else None
    (
        fixture,
        product_agent,
        planner,
        prepared_recovery_request,
    ) = await _prepare_recovery_dryrun_harness(
        debug_root=debug_root,
        scripted_responses=scripted_responses,
        experiment_settings=experiment_settings,
        runtime_context_path=runtime_context_path,
    )
    paths = _case3_paths(runtime_context)
    from cais_spade_llm.recovery_framework.task_des_audit import audit_saved_failure

    baseline_audit = audit_saved_failure(inputs["runtime_context"])
    if debug_root is not None:
        write_record(debug_root / "task_des_audit.json", baseline_audit)
    proposal: dict[str, Any] | None = None
    recovery_safety_generation: dict[str, Any] = {}
    recovery_final: dict[str, str] = {}

    if checkpoint is not None:
        if checkpoint.get("experiment_settings") != prepared_recovery_request.get(
            "recovery_outline_experiment_settings"
        ):
            raise ValueError("Outline experiment settings differ; generate a new outline")
        outline_session = deepcopy(checkpoint["multi_turn_session"])
        prepared_recovery_request["multi_turn_session_state"] = deepcopy(outline_session)
    else:
        proposal = await _execute_recovery_until(
            planner,
            prepared_recovery_request,
            stop_after="outline",
        )
        outline_session = _latest_session(prepared_recovery_request, planner)
    outline_trace = _outline_trace_from_session_state(outline_session)
    _validate_outline_trace(outline_trace, label="transition_trace")
    if outline_session.get("current_phase") not in {"primitive_generation", "finalize"}:
        raise ValueError("Outline generation did not reach an accepted outline checkpoint")
    if debug_root is not None:
        write_record(
            debug_root / "outline_checkpoint.json",
            {
                "input_fingerprint": inputs["fingerprint"],
                "validator_fingerprint": validator_fingerprint(),
                "experiment_settings": prepared_recovery_request.get(
                    "recovery_outline_experiment_settings"
                ),
                "multi_turn_session": outline_session,
                "outline_source": str(outline_checkpoint) if outline_checkpoint else None,
            },
        )

    primitive_session: dict[str, Any] = {}
    primitive_program: list[dict[str, Any]] = []
    if mode in {"primitive", "full"}:
        proposal = await _execute_recovery_until(
            planner,
            prepared_recovery_request,
            stop_after="primitive",
        )
        primitive_session = _latest_session(prepared_recovery_request, planner)
        primitive_program = _primitive_program_from_session(primitive_session)
        _validate_primitive_program(
            primitive_program,
            outline_ids=_outline_ids(outline_trace),
        )

    if mode in {"safety", "full"}:
        recovery_safety_generation = await _run_safety_from_outline(
            product_agent=product_agent,
            prepared_recovery_request=prepared_recovery_request,
            session_state=outline_session,
            debug_root=debug_root,
        )
        _validate_safety_result(
            recovery_safety_generation,
            _outline_trace_from_safety(recovery_safety_generation),
        )

    if mode == "full":
        recovery_final = _write_recovery_final_bundle(
            prepared_recovery_request=prepared_recovery_request,
            session_state=primitive_session,
            recovery_safety_generation=recovery_safety_generation,
            debug_root=debug_root,
        )

    recovery_debug = (
        deepcopy(planner.get_last_recovery_debug())
        if callable(getattr(planner, "get_last_recovery_debug", None))
        else {}
    )
    final_session = primitive_session or outline_session
    return {
        "mode": mode,
        "scenario": str(runtime_context.get("failure_scenario_id") or "").strip(),
        "runtime_context_source": str(runtime_context_path or CASE3_RUNTIME_CONTEXT),
        "llm_response_source": (
            "mocked_scripted_fixture" if scripted_responses is not None else "live"
        ),
        "failure_scenario_source": str(paths["failure_scenario"]),
        "failure_context": deepcopy(fixture.get("failure_context") or {}),
        "proposal": deepcopy(proposal),
        "prepared_recovery_request": prepared_recovery_request,
        "context_summary": deepcopy(prepared_recovery_request.get("context_summary") or {}),
        "llm_input": deepcopy(prepared_recovery_request.get("llm_input") or {}),
        "recovery_debug": recovery_debug,
        "experiment_settings": deepcopy(
            prepared_recovery_request.get("recovery_outline_experiment_settings") or {}
        ),
        "multi_turn_session": deepcopy(final_session),
        "turns": [
            deepcopy(row) for row in (final_session.get("turns") or []) if isinstance(row, dict)
        ],
        "turn_log": deepcopy(product_agent.turn_log),
        "transition_trace": outline_trace,
        "accepted_primitive_program": primitive_program,
        "recovery_safety_generation": recovery_safety_generation,
        "recovery_final": recovery_final,
    }


def run_recovery_outline_only(
    *,
    debug_root: Path | None = None,
    scripted_responses: list[dict[str, Any]] | None = None,
    experiment_settings: dict[str, Any] | None = None,
    runtime_context_path: Path | None = None,
    outline_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Run Case 3 runtime failure context through accepted recovery outline."""
    return asyncio.run(
        _run_actual_recovery(
            mode="outline",
            debug_root=debug_root,
            scripted_responses=scripted_responses,
            experiment_settings=experiment_settings,
            runtime_context_path=runtime_context_path,
            outline_checkpoint=outline_checkpoint,
        )
    )


def run_primitive_composition(
    *,
    debug_root: Path | None = None,
    scripted_responses: list[dict[str, Any]] | None = None,
    experiment_settings: dict[str, Any] | None = None,
    runtime_context_path: Path | None = None,
    outline_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Run Case 3 outline first, then actual primitive composition."""
    return asyncio.run(
        _run_actual_recovery(
            mode="primitive",
            debug_root=debug_root,
            scripted_responses=scripted_responses,
            experiment_settings=experiment_settings,
            runtime_context_path=runtime_context_path,
            outline_checkpoint=outline_checkpoint,
        )
    )


def run_safety_synthsis(
    *,
    debug_root: Path | None = None,
    scripted_responses: list[dict[str, Any]] | None = None,
    experiment_settings: dict[str, Any] | None = None,
    runtime_context_path: Path | None = None,
    outline_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Run Case 3 outline first, then actual recovery safety synthsis."""
    return asyncio.run(
        _run_actual_recovery(
            mode="safety",
            debug_root=debug_root,
            scripted_responses=scripted_responses,
            experiment_settings=experiment_settings,
            runtime_context_path=runtime_context_path,
            outline_checkpoint=outline_checkpoint,
        )
    )


def run_full(
    *,
    debug_root: Path | None = None,
    scripted_responses: list[dict[str, Any]] | None = None,
    experiment_settings: dict[str, Any] | None = None,
    runtime_context_path: Path | None = None,
    outline_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Run Case 3 outline, safety synthsis, primitive composition, and final bundle."""
    return asyncio.run(
        _run_actual_recovery(
            mode="full",
            debug_root=debug_root,
            scripted_responses=scripted_responses,
            experiment_settings=experiment_settings,
            runtime_context_path=runtime_context_path,
            outline_checkpoint=outline_checkpoint,
        )
    )


def _outline_ids(trace: list[dict[str, Any]]) -> list[str]:
    return [str(row.get("outline_id") or "").strip() for row in trace]


def _outline_trace_from_safety(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        deepcopy(row)
        for row in (payload.get("accepted_outline_prefix") or [])
        if isinstance(row, dict)
    ]


def _validate_outline_trace(trace: list[dict[str, Any]], *, label: str) -> None:
    if not trace:
        raise AssertionError(f"{label} is empty")
    seen_outline_ids: set[str] = set()
    for index, row in enumerate(trace):
        if not isinstance(row, dict):
            raise AssertionError(f"{label}[{index}] is not an object")
        for key in ("outline_id", "event_name", "resource_jid"):
            if not str(row.get(key) or "").strip():
                raise AssertionError(f"{label}[{index}] is missing {key}")
        for key in ("expected_start_state", "expected_end_state"):
            value = row.get(key)
            if value is not None and not isinstance(value, dict):
                raise AssertionError(f"{label}[{index}].{key} is not an object")
        outline_id = str(row.get("outline_id") or "").strip()
        if outline_id in seen_outline_ids:
            raise AssertionError(f"{label} has duplicate outline_id {outline_id!r}")
        seen_outline_ids.add(outline_id)


def _validate_primitive_program(
    primitive_program: list[dict[str, Any]],
    *,
    outline_ids: list[str],
) -> None:
    if not primitive_program:
        raise AssertionError("accepted_primitive_program is empty")
    known_outline_ids = set(outline_ids)
    primitive_outline_ids: set[str] = set()
    for index, row in enumerate(primitive_program):
        if not isinstance(row, dict):
            raise AssertionError(f"accepted_primitive_program[{index}] is not an object")
        outline_id = str(row.get("outline_id") or "").strip()
        if not outline_id:
            raise AssertionError(f"accepted_primitive_program[{index}] is missing outline_id")
        if known_outline_ids and outline_id not in known_outline_ids:
            raise AssertionError(
                f"accepted_primitive_program[{index}] references unknown outline_id {outline_id!r}"
            )
        primitive_outline_ids.add(outline_id)
        primitive_steps = row.get("primitive_steps")
        if not isinstance(primitive_steps, list) or not primitive_steps:
            raise AssertionError(f"accepted_primitive_program[{index}] has no primitive_steps")
    missing_outline_ids = [
        outline_id for outline_id in outline_ids if outline_id not in primitive_outline_ids
    ]
    if missing_outline_ids:
        raise AssertionError(
            f"accepted_primitive_program is missing produced outline_id(s): {missing_outline_ids}"
        )


def _validate_safety_result(
    payload: dict[str, Any],
    accepted_outline_prefix: list[dict[str, Any]],
) -> None:
    _validate_outline_trace(accepted_outline_prefix, label="accepted_outline_prefix")
    if "ok" not in payload:
        raise AssertionError("safety result is missing ok")
    if not str(payload.get("recovery_safety_status") or "").strip():
        raise AssertionError("safety result is missing recovery_safety_status")
    rule_ids = [
        str(rule_id).strip() for rule_id in (payload.get("rule_ids") or []) if str(rule_id).strip()
    ]
    if not rule_ids:
        raise AssertionError("safety result has no rule_ids")


def _compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, ensure_ascii=True)


def _emit(line: str = "") -> None:
    sys.stdout.write(f"{line}\n")


def _print_result(result: dict[str, Any]) -> None:
    _emit(f"Case 3 actual recovery dry-run mode: {result.get('mode') or '-'}")
    _emit(f"LLM response source: {result.get('llm_response_source') or '-'}")
    _emit(f"Runtime context source: {result.get('runtime_context_source') or '-'}")
    _emit(f"Failure scenario source: {result.get('failure_scenario_source') or '-'}")
    _print_experiment_settings(result.get("experiment_settings") or {})
    _emit("Failure context payload:")
    _emit(_compact_json(result.get("failure_context") or {}))
    _print_fault_event(result.get("context_summary") or {})
    _print_turns(result.get("turns") or [])
    _print_artifact_paths(result)
    _emit()
    _print_outline(result.get("transition_trace") or [])
    if result.get("accepted_primitive_program"):
        _emit()
        _print_primitives(result.get("accepted_primitive_program") or [])
    if result.get("recovery_safety_generation"):
        _emit()
        _print_safety(result.get("recovery_safety_generation") or {})
    if result.get("recovery_final"):
        _emit()
        _emit("Recovery final")
        for key, value in (result.get("recovery_final") or {}).items():
            _emit(f"  {key}: {value}")


def _print_experiment_settings(settings: dict[str, Any]) -> None:
    if not settings:
        return
    _emit("Recovery outline experiment settings:")
    for key in (
        "enabled",
        "recovery_selection_mode",
        "action_horizon",
        "candidate_count",
    ):
        _emit(f"  {key}: {settings.get(key)}")


def _print_fault_event(context_summary: dict[str, Any]) -> None:
    fault_event = dict(context_summary.get("fault_event") or {})
    _emit()
    _emit("Fault event used in prompt")
    _emit(f"  focused_resource_jid: {fault_event.get('focused_resource_jid') or '-'}")
    _emit(f"  blocked_at_task_id: {fault_event.get('blocked_at_task_id') or '-'}")
    _emit(f"  blocked_at_function: {fault_event.get('blocked_at_function') or '-'}")
    _emit(f"  resource_state: {fault_event.get('resource_state') or '-'}")


def _print_turns(turns: list[dict[str, Any]]) -> None:
    _emit()
    _emit("Multi-turn trace")
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        _emit(
            "  turn={turn} phase={phase} decision={decision}".format(
                turn=turn.get("turn_index") or "-",
                phase=turn.get("phase") or "-",
                decision=turn.get("decision") or turn.get("final_output_stage") or "-",
            )
        )
        phase = str(turn.get("phase") or "").strip().lower()
        request_path = str(turn.get("request_artifact_path") or "").strip()
        grounding_result_path = str(turn.get("grounding_result_artifact_path") or "").strip()
        outline_result_path = str(turn.get("outline_result_artifact_path") or "").strip()
        prompt_path = str(turn.get("prompt_artifact_path") or "").strip()
        llm_response_path = str(turn.get("llm_response_artifact_path") or "").strip()
        response_path = str(turn.get("response_artifact_path") or "").strip()
        stack_path = str(turn.get("outline_stack_artifact_path") or "").strip()
        index_path = str(turn.get("turn_index_artifact_path") or "").strip()
        if phase == "grounding":
            if request_path or prompt_path:
                _emit(f"    request: {request_path or prompt_path}")
            if grounding_result_path or response_path:
                _emit(f"    grounding result: {grounding_result_path or response_path}")
            continue
        if phase == "outline":
            if request_path or prompt_path:
                _emit(f"    outline request: {request_path or prompt_path}")
            if outline_result_path or response_path:
                _emit(f"    outline result: {outline_result_path or response_path}")
            if stack_path:
                _emit(f"    outline stack: {stack_path}")
            continue
        if prompt_path:
            _emit(f"    prompt: {prompt_path}")
        if llm_response_path:
            _emit(f"    raw LLM response: {llm_response_path}")
        if response_path:
            _emit(f"    enriched response: {response_path}")
        if stack_path:
            _emit(f"    outline stack: {stack_path}")
        if index_path:
            _emit(f"    turn index: {index_path}")


def _print_artifact_paths(result: dict[str, Any]) -> None:
    safety = dict(result.get("recovery_safety_generation") or {})
    final = dict(result.get("recovery_final") or {})
    paths = [
        safety.get("snapshot_artifact_path"),
        safety.get("grounding_prompt_artifact_path"),
        safety.get("grounding_llm_response_artifact_path"),
        safety.get("grounding_response_artifact_path"),
        safety.get("recovery_safety_logic_json"),
        final.get("recovery_final_dir"),
        final.get("recovery_final_output_path"),
    ]
    paths = [str(path).strip() for path in paths if str(path or "").strip()]
    if not paths:
        return
    _emit()
    _emit("Stage artifact paths")
    for path in paths:
        _emit(f"  {path}")


def _print_outline(trace: list[dict[str, Any]]) -> None:
    _emit("Recovery outline")
    for row in trace:
        outline_id = str(row.get("outline_id") or "-")
        event_name = str(row.get("event_name") or "-")
        resource_jid = str(row.get("resource_jid") or "-")
        part_name = str(row.get("part_name") or "-")
        _emit(f"  {outline_id}: {event_name}")
        _emit(f"    resource_jid: {resource_jid}")
        _emit(f"    part_name: {part_name}")
        _emit(f"    expected_start_state: {_compact_json(row.get('expected_start_state') or {})}")
        _emit(f"    expected_end_state:   {_compact_json(row.get('expected_end_state') or {})}")


def _print_primitives(primitive_program: list[dict[str, Any]]) -> None:
    _emit("Primitive composition")
    for row in primitive_program:
        outline_id = str(row.get("outline_id") or "-")
        event_name = str(row.get("event_name") or "-")
        primitive_steps = [
            step for step in (row.get("primitive_steps") or []) if isinstance(step, dict)
        ]
        _emit(f"  {outline_id}: {event_name} ({len(primitive_steps)} step(s))")
        for index, step in enumerate(primitive_steps, start=1):
            primitive = str(step.get("primitive") or "-")
            params = step.get("params") if isinstance(step.get("params"), dict) else {}
            _emit(f"    {index}. {primitive} {_compact_json(params)}")


def _print_safety(result: dict[str, Any]) -> None:
    _emit("Safety synthsis")
    _emit(f"  ok: {result.get('ok')}")
    _emit(f"  recovery_safety_status: {result.get('recovery_safety_status') or '-'}")
    _emit(f"  recovery_safety_scope_id: {result.get('recovery_safety_scope_id') or '-'}")
    _emit(f"  recovery_safety_logic_json: {result.get('recovery_safety_logic_json') or '-'}")
    _emit(f"  rule_ids: {result.get('rule_ids') or []}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Saved-context recovery generation and validation debugger"
    )
    parser.add_argument(
        "--mode",
        choices=("outline", "primitive", "safety", "full"),
        default="outline",
        help="Recovery stage to run",
    )
    parser.add_argument(
        "--runtime-context", type=Path, required=True, help="Saved runtime_context JSON"
    )
    parser.add_argument("--debug-root", type=Path, help="Directory for this test's artifacts")
    parser.add_argument(
        "--outline-checkpoint", type=Path, help="Accepted outline to reuse with matching inputs"
    )
    parser.add_argument(
        "--replay-responses",
        type=Path,
        help="Explicit evaluator response replay JSON (never sent to the model)",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = _parse_args()
    run_by_mode = {
        "outline": run_recovery_outline_only,
        "primitive": run_primitive_composition,
        "safety": run_safety_synthsis,
        "full": run_full,
    }
    result = run_by_mode[args.mode](
        runtime_context_path=args.runtime_context,
        debug_root=args.debug_root,
        outline_checkpoint=args.outline_checkpoint,
        scripted_responses=_load_json(args.replay_responses) if args.replay_responses else None,
    )
    if args.debug_root is not None:
        from cais_spade_llm.recovery_framework.diagnostics import write_record

        write_record(
            args.debug_root / "result.json",
            {
                "mode": args.mode,
                "scenario": result["scenario"],
                "llm_response_source": result["llm_response_source"],
                "transition_trace": result["transition_trace"],
                "accepted_primitive_program": result["accepted_primitive_program"],
                "recovery_safety_generation": result["recovery_safety_generation"],
                "experiment_settings": result["experiment_settings"],
            },
        )
    _print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
