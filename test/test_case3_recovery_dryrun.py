"""Case 3 runtime failure-context dry-run for recovery stages.

Run directly:
    python test/test_case3_recovery_dryrun.py --mode outline
    python test/test_case3_recovery_dryrun.py --mode primitive
    python test/test_case3_recovery_dryrun.py --mode safety
    python test/test_case3_recovery_dryrun.py --mode full

Each mode loads runtime_context.json and derives the Case 3 failure facts from
the configured bundle, failure event, part tracker, and resource snapshots.
The Case 3 response fixtures are only used by tests as mocked LLM responses.
"""

# ruff: noqa: E402, I001

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
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
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

from cais_spade_llm.agents.central_controller.recovery_safety_generation import (
    generate_recovery_safety_bundle,
)  # noqa: E402
from cais_spade_llm.agents.central_controller import (  # noqa: E402
    outline_macro_safety as outline_macro_safety_module,
)
from cais_spade_llm.agents.central_controller.outline_macro_safety import (  # noqa: E402
    validate_outline_macro_recovery_safety,
)
from cais_spade_llm.agents.intelligent_product.process_planner import ProcessPlanner  # noqa: E402
from cais_spade_llm.agents.intelligent_product.replanner.failure_context import (  # noqa: E402
    build_failure_event,
    failure_context_from_scenario_config,
    load_failure_scenario_config,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (  # noqa: E402
    multi_turn as multi_turn_mode,
    multi_turn_outline_state,
    multi_turn_prompts,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery import (  # noqa: E402
    recovery_validation_service,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_artifacts import (  # noqa: E402
    write_recovery_artifacts,
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
CASE3_RESPONSE_FIXTURES = ROOT / "test" / "fixtures" / "case3_recovery"
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
    return json.loads(path.read_text(encoding="utf-8"))


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
    return path if path.is_absolute() else ROOT / path


def _load_case3_runtime_context() -> dict[str, Any]:
    payload = _load_json(CASE3_RUNTIME_CONTEXT)
    if not isinstance(payload, dict):
        raise TypeError(f"runtime context {CASE3_RUNTIME_CONTEXT} did not decode to an object")
    return payload


def _bundle_artifact_path(bundle_root: Path, artifacts: dict[str, Any], key: str) -> Path:
    raw_path = str(artifacts.get(key) or "").strip()
    if not raw_path:
        raise KeyError(f"bundle manifest is missing artifacts.{key}")
    path = Path(raw_path)
    return path if path.is_absolute() else bundle_root / path


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
            ROOT
            / "cais_spade_llm"
            / "initialization"
            / "failure_scenarios"
            / f"{failure_scenario_id}.json"
        ),
        "recovery_outline_experiment_settings": (
            ROOT / "cais_spade_llm" / "initialization" / "recovery_outline_experiment_settings.json"
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
    paths = _case3_paths()
    settings_path = paths["recovery_outline_experiment_settings"]
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
        self.instructions = "Case 3 runtime recovery dry-run product agent."
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
        snapshot = resource.get_recovery_snapshot()
        recovery_des_model = resource.recovery_des_model(snapshot=snapshot)
        results: list[dict[str, Any]] = []
        for row in payload.get("candidates") or []:
            candidate = dict(row or {})
            task = dict(candidate.get("task") or {})
            physical_input = dict(candidate.get("physical_input") or {})
            validation_snapshot = resource.recovery_physical_validation_snapshot(
                live_snapshot=snapshot,
                physical_input=physical_input,
                recovery_des_model=recovery_des_model,
            )
            result = resource.check_recovery_physical_feasibility(
                part_context=deepcopy(physical_input.get("part_context") or {}),
                recovery_snapshot=validation_snapshot,
                grounded_action=deepcopy(physical_input.get("grounded_action") or {}),
                operation_kind=str(physical_input.get("operation_kind") or ""),
                part_name=physical_input.get("part_name"),
            )
            findings = []
            if not bool(result.get("allowed")):
                findings.append(
                    {
                        "validation_category": "physical_feasibility",
                        "constraint_owner": "resource",
                        "constraint_family": "resource_feasibility",
                        "constraint_code": str(
                            result.get("constraint_code") or "resource_blocked"
                        ),
                        "reason": str(result.get("reason") or "mocked RA rejected"),
                        "resource_jid": str(task.get("resource_jid") or resource_jid),
                        "part_name": task.get("part_name"),
                        "evidence": deepcopy(result.get("evidence") or {}),
                    }
                )
            results.append(
                {
                    "candidate_index": int(candidate.get("candidate_index") or 0),
                    "allowed": bool(result.get("allowed")),
                    "findings": findings,
                    "resource_result": deepcopy(result),
                }
            )
        return {
            "request_id": str(payload.get("request_id") or "mocked-ra-request"),
            "recovery_session_id": str(payload.get("recovery_session_id") or ""),
            "turn_index": int(payload.get("turn_index") or 0),
            "state_fingerprint": str(payload.get("state_fingerprint") or ""),
            "validator_jid": resource_jid,
            "snapshot": deepcopy(snapshot),
            "snapshot_fingerprint": recovery_validation_fingerprint(snapshot),
            "recovery_des_model": deepcopy(recovery_des_model),
            "recovery_des_model_fingerprint": str(
                recovery_des_model.get("descriptor_fingerprint") or ""
            ),
            "results": results,
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
            live_safety_dfa_states = {
                str(rule_id): str(state)
                for rule_id, state in sorted(live_monitor.current_states.items())
            }
            safety_rule_fingerprint = recovery_validation_fingerprint(rules)
            live_safety_dfa_state_fingerprint = recovery_validation_fingerprint(
                live_safety_dfa_states
            )
            expected_rule_fingerprint = str(
                candidate.get("safety_rule_fingerprint") or ""
            ).strip()
            if (
                expected_rule_fingerprint
                and expected_rule_fingerprint != safety_rule_fingerprint
            ):
                raise RuntimeError("CCA safety-rule fingerprint changed")
            expected_live_state_fingerprint = str(
                candidate.get("live_safety_dfa_state_fingerprint") or ""
            ).strip()
            if (
                expected_live_state_fingerprint
                and expected_live_state_fingerprint
                != live_safety_dfa_state_fingerprint
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
                projected_resources=deepcopy(
                    safety_input.get("projected_resources") or {}
                ),
                projected_parts=deepcopy(safety_input.get("projected_parts") or {}),
                llm_input=llm_input,
                safety_dfa_states_before=deepcopy(projected_dfa_states),
            )
            cleared = [
                str(item).strip()
                for item in (result.get("cleared_condition_ids") or [])
                if str(item).strip()
            ]
            active = [
                str(item).strip()
                for item in (candidate.get("active_safety_condition_ids") or [])
                if str(item).strip()
            ]
            results.append(
                {
                    "candidate_index": int(candidate.get("candidate_index") or 0),
                    "is_safe": bool(result.get("is_safe")),
                    "findings": deepcopy(result.get("findings") or []),
                    "active_rule_identifiers": [
                        str(rule.get("id") or "").strip()
                        for rule in rules
                        if str(rule.get("id") or "").strip()
                    ],
                    "cleared_safety_condition_identifiers": cleared,
                    "remaining_safety_condition_identifiers": [
                        condition_id
                        for condition_id in active
                        if condition_id not in set(cleared)
                    ],
                    "safety_context": deepcopy(result.get("safety_ctx") or {}),
                    "safety_dfa_states_before": deepcopy(
                        result.get("safety_dfa_states_before") or {}
                    ),
                    "safety_dfa_states_after": deepcopy(
                        result.get("safety_dfa_states_after") or {}
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
    ) -> dict[str, Any]:
        response_source = (
            "mocked_scripted_fixture" if self._uses_scripted_responses else "live"
        )
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

        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - depends on local env
            raise RuntimeError("openai package required for live recovery") from exc
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        client = OpenAI()

        def _call() -> dict[str, Any]:
            messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
            for _round_index in range(max_tool_rounds + 1):
                kwargs: dict[str, Any] = {
                    "model": self.llm_model,
                    "messages": messages,
                    "reasoning_effort": self.llm_reasoning_effort,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": response_format,
                    },
                }
                if tools:
                    kwargs["tools"] = tools
                response = client.chat.completions.create(**kwargs)
                choice = response.choices[0].message
                if getattr(choice, "tool_calls", None) and tool_executor:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": choice.content or "",
                            "tool_calls": [
                                {
                                    "id": tool_call.id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_call.function.name,
                                        "arguments": tool_call.function.arguments,
                                    },
                                }
                                for tool_call in choice.tool_calls
                            ],
                        }
                    )
                    for tool_call in choice.tool_calls:
                        result = tool_executor(
                            tool_call.function.name,
                            json.loads(tool_call.function.arguments),
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(result, default=str),
                            }
                        )
                    continue
                return _parse_structured_json_text(choice.content or "{}")
            raise RuntimeError("exceeded max tool rounds")

        parsed = await asyncio.to_thread(_call)
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

    def __init__(
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

    _is_pose_in_workspace = RobotAgent._is_pose_in_workspace
    recovery_physical_validation_snapshot = staticmethod(
        RobotAgent.recovery_physical_validation_snapshot
    )
    check_recovery_physical_feasibility = (
        RobotAgent.check_recovery_physical_feasibility
    )

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
    scenario_config = load_failure_scenario_config(scenario_id)
    drop_pose = deepcopy(dict(scenario_config.get("injection") or {}).get("drop_pose") or {})
    observations = {
        "last_commanded_location": str(
            dict(failed_task.get("params") or {}).get("destination_location") or ""
        ).strip(),
    }
    if drop_pose:
        observations["dropped_location"] = drop_pose
    observations.update(deepcopy(dict(failure_event.get("observations") or {})))
    return build_failure_event(
        failed_task_id=failed_task_id,
        failed_resource_jid=failed_resource_jid,
        failed_function_name=failed_function_name,
        final_status=str(failure_event.get("final_status") or "failed").strip(),
        part_name=str(failure_event.get("part_name") or "").strip(),
        base_failure_context=failure_context_from_scenario_config(scenario_config),
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
        (
            row
            for row in (runtime_context.get("resource_snapshots") or [])
            if isinstance(row, dict)
        ),
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


def _holder_resource_jid_for_part_row(part_row: dict[str, Any]) -> str:
    holder_resource_jid = str(part_row.get("current_holder_resource_jid") or "").strip()
    if holder_resource_jid:
        return holder_resource_jid
    current_location = str(part_row.get("current_location") or "").strip()
    if current_location.endswith("_gripper"):
        return current_location.rsplit("_gripper", 1)[0]
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
        holder_resource_jid = _holder_resource_jid_for_part_row(part_row)
        if normalized_pose is None and holder_resource_jid:
            holder_robot = robots_by_jid.get(holder_resource_jid)
            if holder_robot is not None:
                normalized_pose = _normalized_observation_pose(holder_robot._position)
        if normalized_pose is None:
            continue
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
        recovery_session["candidate_proposal_budget"] = int(
            settings["candidate_proposal_budget"]
        )
        if settings["recovery_selection_mode"] == "neurosymbolic":
            recovery_session["candidate_bound"] = int(
                settings["candidate_proposal_budget"]
            )
    prepared_recovery_request["recovery_session"] = recovery_session
    prepared_recovery_request["recovery_outline_experiment_settings"] = deepcopy(settings)
    return settings


async def _prepare_recovery_dryrun_harness(
    *,
    debug_root: Path | None = None,
    scripted_responses: list[dict[str, Any]] | None = None,
    experiment_settings: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], FakeProductAgent, ProcessPlanner, dict[str, Any]]:
    runtime_context = _load_case3_runtime_context()
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
    robots: list[FakeRecoveryRobot] = []
    for raw_snapshot in sorted(
        (
            row
            for row in (runtime_context.get("resource_snapshots") or [])
            if isinstance(row, dict)
        ),
        key=lambda row: str(row.get("resource_jid") or ""),
    ):
        resource_jid = str(raw_snapshot.get("resource_jid") or "").strip()
        config_path = _repo_path(raw_snapshot.get("resource_config"))
        config_key = str(raw_snapshot.get("resource_config_key") or "").strip()
        execution_env = str(raw_snapshot.get("execution_env") or "").strip()
        if not resource_jid or not config_key or not execution_env:
            raise ValueError("runtime context resource snapshot is missing identity fields")
        config = _load_robot_config(config_path, config_key)
        robot = FakeRecoveryRobot(
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
        robots.append(robot)
    if not robots:
        raise ValueError("runtime context resource_snapshots is empty")

    planner = ProcessPlannerPrepareTrace(product_agent, robots)
    product_agent._mock_recovery_validation_resources = {
        robot.jid: robot for robot in robots
    }
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

    async def _direct_to_thread(func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    with patch(
        "cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.recovery_session.asyncio.to_thread",
        new=_direct_to_thread,
    ):
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
    llm_input = dict(prepared_recovery_request.get("llm_input") or {})
    modeled_continuation_gap = dict(llm_input.get("modeled_continuation_gap") or {})
    pending_nominal_tasks = [
        deepcopy(row)
        for row in (modeled_continuation_gap.get("pending_nominal_tasks") or [])
        if isinstance(row, dict)
    ]
    pending_nominal_task_ids = [
        str(task_id).strip()
        for task_id in (
            modeled_continuation_gap.get("pending_nominal_task_ids")
            or [row.get("task_id") or row.get("id") for row in pending_nominal_tasks]
        )
        if str(task_id).strip()
    ]
    nominal_candidate_tasks = _nominal_candidate_tasks(
        pending_nominal_tasks=pending_nominal_tasks,
    )
    recovery_safety_dir = _debug_root(debug_root) / "recovery_safety"
    return {
        "product_jid": "assembly_board-v1@localhost",
        "recovery_safety_scope_id": "dryrun_recovery_scope",
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
) -> dict[str, Any]:
    runtime_context = _load_case3_runtime_context()
    fixture, product_agent, planner, prepared_recovery_request = await _prepare_recovery_dryrun_harness(
        debug_root=debug_root,
        scripted_responses=scripted_responses,
        experiment_settings=experiment_settings,
    )
    paths = _case3_paths()
    proposal: dict[str, Any] | None = None
    recovery_safety_generation: dict[str, Any] = {}
    recovery_final: dict[str, str] = {}

    proposal = await _execute_recovery_until(
        planner,
        prepared_recovery_request,
        stop_after="outline",
    )
    outline_session = _latest_session(prepared_recovery_request, planner)
    outline_trace = _outline_trace_from_session_state(outline_session)
    _validate_outline_trace(outline_trace, label="transition_trace")

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
        "runtime_context_source": str(CASE3_RUNTIME_CONTEXT),
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
) -> dict[str, Any]:
    """Run Case 3 runtime failure context through accepted recovery outline."""
    return asyncio.run(
        _run_actual_recovery(
            mode="outline",
            debug_root=debug_root,
            scripted_responses=scripted_responses,
            experiment_settings=experiment_settings,
        )
    )


def run_primitive_composition(
    *,
    debug_root: Path | None = None,
    scripted_responses: list[dict[str, Any]] | None = None,
    experiment_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run Case 3 outline first, then actual primitive composition."""
    return asyncio.run(
        _run_actual_recovery(
            mode="primitive",
            debug_root=debug_root,
            scripted_responses=scripted_responses,
            experiment_settings=experiment_settings,
        )
    )


def run_safety_synthsis(
    *,
    debug_root: Path | None = None,
    scripted_responses: list[dict[str, Any]] | None = None,
    experiment_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run Case 3 outline first, then actual recovery safety synthsis."""
    return asyncio.run(
        _run_actual_recovery(
            mode="safety",
            debug_root=debug_root,
            scripted_responses=scripted_responses,
            experiment_settings=experiment_settings,
        )
    )


def run_full(
    *,
    debug_root: Path | None = None,
    scripted_responses: list[dict[str, Any]] | None = None,
    experiment_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run Case 3 outline, safety synthsis, primitive composition, and final bundle."""
    return asyncio.run(
        _run_actual_recovery(
            mode="full",
            debug_root=debug_root,
            scripted_responses=scripted_responses,
            experiment_settings=experiment_settings,
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
        grounding_result_path = str(
            turn.get("grounding_result_artifact_path") or ""
        ).strip()
        outline_result_path = str(
            turn.get("outline_result_artifact_path") or ""
        ).strip()
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
    parser = argparse.ArgumentParser(description="Case 3 actual runtime recovery dry-run debugger")
    parser.add_argument(
        "--mode",
        choices=("outline", "primitive", "safety", "full"),
        default="outline",
        help="Recovery stage to run",
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
    result = run_by_mode[args.mode]()
    _print_result(result)
    return 0


def _load_response_fixture(filename: str) -> dict[str, Any]:
    payload = _load_json(CASE3_RESPONSE_FIXTURES / filename)
    if not isinstance(payload, dict):
        raise TypeError(f"response fixture {filename} did not decode to an object")
    return payload


def _without_description_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_description_fields(item)
            for key, item in value.items()
            if key != "description"
        }
    if isinstance(value, list):
        return [_without_description_fields(item) for item in value]
    return deepcopy(value)


def _fixture_outline_responses() -> list[dict[str, Any]]:
    fixture_specs = (
        (
            "turn01_grounding_response.json",
            ("thought", "decision", "observe_requests"),
        ),
        (
            "turn02_grounding_response.json",
            ("thought", "decision", "observe_requests"),
        ),
        (
            "turn03_outline_response.json",
            ("thought", "selected_candidate_index", "candidate_events"),
        ),
        (
            "turn04_outline_response.json",
            ("thought", "selected_candidate_index", "candidate_events"),
        ),
        (
            "turn05_outline_response.json",
            ("thought", "selected_candidate_index", "candidate_events"),
        ),
        (
            "turn06_outline_response.json",
            ("thought", "selected_candidate_index", "candidate_events"),
        ),
    )
    responses: list[dict[str, Any]] = []
    for filename, response_keys in fixture_specs:
        payload = _load_response_fixture(filename)
        model_response = {
            key: deepcopy(payload[key]) for key in response_keys if key in payload
        }
        if (
            "candidate_events" in model_response
            and _load_recovery_outline_experiment_settings()[
                "recovery_selection_mode"
            ]
            == "neurosymbolic"
        ):
            selected_index = int(model_response.pop("selected_candidate_index", 0) or 0)
            candidate_events = list(model_response.get("candidate_events") or [])
            model_response["candidate_events"] = [
                deepcopy(candidate_events[selected_index])
            ]
        responses.append(_without_description_fields(model_response))
    return responses


def _with_three_mocked_candidate_events(response: dict[str, Any]) -> dict[str, Any]:
    """Expand a test-only response to the three-candidate runtime contract."""
    expanded = deepcopy(response)
    candidate_events = [
        deepcopy(row)
        for row in (expanded.get("candidate_events") or [])
        if isinstance(row, dict)
    ]
    selected_index = int(expanded.get("selected_candidate_index") or 0)
    selected_event = deepcopy(candidate_events[selected_index])
    while len(candidate_events) < 3:
        alternative_number = len(candidate_events) + 1
        alternative = deepcopy(selected_event)
        alternative["outline_id"] = (
            f"{str(selected_event.get('outline_id') or '').strip()}_alternative_"
            f"{alternative_number}"
        )
        alternative["event_name"] = (
            f"{str(selected_event.get('event_name') or '').strip()}_alternative_"
            f"{alternative_number}"
        )
        alternative["rationale"] = (
            f"Mocked alternative {alternative_number} with the same declared physical effect."
        )
        candidate_events.append(alternative)
    expanded["candidate_events"] = candidate_events
    return expanded


def _novel_symbol_outline_responses() -> list[dict[str, Any]]:
    responses = _fixture_outline_responses()[:2]
    novel_responses = [
        {
            "thought": "Use explicit occupancy and a grounded named pose.",
            "selected_candidate_index": 0,
            "candidate_events": [
                {
                    "outline_id": "novel_clear_xarm6",
                    "event_name": "evt_q7",
                    "resource_jid": "xarm6@localhost",
                    "expected_start_state": {
                        "resource_state": "failed",
                        "resource_location": "assembly_board-v1",
                    },
                    "expected_end_state": {
                        "resource_state": "xarm6_clear_state",
                        "resource_location": "home",
                    },
                    "rationale": "Leave the explicitly occupied protected destination.",
                }
            ],
        },
        {
            "thought": "Free the reachable manipulator without entering the destination.",
            "selected_candidate_index": 0,
            "candidate_events": [
                {
                    "outline_id": "novel_stage_mcp",
                    "event_name": "evt_z9",
                    "resource_jid": "ur5e@localhost",
                    "part_name": "MCP",
                    "expected_start_state": {
                        "resource_state": "picked",
                        "held_part": "MCP",
                        "part_state": "in_gripper",
                        "part_location": "ur5e@localhost_gripper",
                    },
                    "expected_end_state": {
                        "resource_state": "mcp_buffer_clear",
                        "held_part": None,
                        "part_state": "mcp_waiting_recovery",
                        "part_location": "prusa-mk4-2",
                    },
                    "rationale": "Release MCP at a supplied reachable location.",
                }
            ],
        },
        {
            "thought": "Acquire the affected part from the grounded observation.",
            "selected_candidate_index": 0,
            "candidate_events": [
                {
                    "outline_id": "novel_acquire_lg",
                    "event_name": "evt_n4",
                    "resource_jid": "ur5e@localhost",
                    "part_name": "LG",
                    "expected_start_state": {
                        "resource_state": "mcp_buffer_clear",
                        "held_part": None,
                        "part_state": "misplaced",
                    },
                    "expected_end_state": {
                        "resource_state": "lg_secured",
                        "held_part": "LG",
                        "part_state": "lg_under_recovery_control",
                        "part_location": "ur5e@localhost_gripper",
                    },
                    "rationale": "Acquire LG using its observed pose.",
                }
            ],
        },
        {
            "thought": "Restore the held affected part to its supplied goal location.",
            "selected_candidate_index": 0,
            "candidate_events": [
                {
                    "outline_id": "novel_restore_lg",
                    "event_name": "evt_v2",
                    "resource_jid": "ur5e@localhost",
                    "part_name": "LG",
                    "expected_start_state": {
                        "resource_state": "lg_secured",
                        "held_part": "LG",
                        "part_state": "lg_under_recovery_control",
                        "part_location": "ur5e@localhost_gripper",
                    },
                    "expected_end_state": {
                        "resource_state": "lg_recovery_complete",
                        "resource_location": "assembly_board-v1",
                        "held_part": None,
                        "part_state": "lg_restored_state",
                        "part_location": "assembly_board-v1",
                    },
                    "rationale": "Release LG at assembly_board-v1 after occupancy clears.",
                }
            ],
        },
    ]
    if (
        _load_recovery_outline_experiment_settings()["recovery_selection_mode"]
        == "neurosymbolic"
    ):
        for response in novel_responses:
            response.pop("selected_candidate_index", None)
        responses.extend(deepcopy(novel_responses))
    else:
        responses.extend(
            _with_three_mocked_candidate_events(response) for response in novel_responses
        )
    return responses


def _render_non_case3_candidate_prompt(
    *,
    reverse_order: bool = False,
    candidate_rejection_feedback: list[dict[str, Any]] | None = None,
    accepted_outline_prefix: list[dict[str, Any]] | None = None,
) -> str:
    resources = [
        {
            "resource_jid": "ur5e@localhost",
            "current_state": "idle",
            "current_location": "Assembly Station",
            "held_part": None,
            "gripper_state": "open",
            "current_pose": {"x": -0.25, "y": 0.22, "z": 1.18},
        },
        {
            "resource_jid": "xarm6@localhost",
            "current_state": "failed",
            "current_location": "station",
            "held_part": None,
            "gripper_state": "open",
            "current_pose": {"x": 0.1, "y": 0.08, "z": 1.2},
        },
    ]
    parts = [
        {
            "part_name": "LCP",
            "current_state": "misplaced",
            "current_location": "station",
            "current_holder_resource_jid": None,
            "origin_location": "prusa-mk4-1",
            "goal_location": "station",
            "goal_requirement_id": "REQ_1",
            "observed_pose": {"x": 0.0, "y": 0.2, "z": 1.035},
        },
        {
            "part_name": "MG",
            "current_state": "assembled",
            "current_location": "Assembly Station",
            "current_holder_resource_jid": None,
            "origin_location": "prusa-mk4-2",
            "goal_location": "Assembly Station",
            "goal_requirement_id": "REQ_3",
            "observed_pose": {"x": 0.0, "y": -0.08, "z": 1.025},
        },
    ]
    if reverse_order:
        resources.reverse()
        parts.reverse()

    recovery_resources = {
        row["resource_jid"]: {
            "static_capabilities": {
                "named_poses": ["home", row["current_location"]],
                "reachable_locations": ["prusa-mk4-1", "prusa-mk4-2", "station"],
            },
            "recovery_adapter": {
                "supports_executable_recovery": True,
                "supports_manipulator_pick_place": True,
            },
            "recovery_snapshot": deepcopy(row),
        }
        for row in resources
    }
    llm_input = {
        "fault_event": {
            "focused_resource_jid": "xarm6@localhost",
            "blocked_at_task_id": "REQ_1_T3",
            "blocked_at_function": "place_approach",
            "affected_part_names": ["LCP"],
        },
        "observed_runtime_state": {"resources": deepcopy(resources)},
        "part_facts": deepcopy(parts),
        "relevant_assembly_requirements": [
            {
                "requirement_id": "REQ_1",
                "status": "failed",
                "summary": "xarm6@localhost place_approach LCP to station",
            }
        ],
        "loaded_safety_rules": [
            {
                "id": "SAFE_1",
                "raw_text": "LCP by xarm6@localhost must place_approach before MG by ur5e@localhost.",
                "constraint_type": "ordering_place_approach_priority",
                "product": ["LCP", "MG"],
                "event": "place_approach",
                "context": {"destination": "station"},
            }
        ],
    }
    session_state = {
        "outline_mode": "incremental_candidates_validated",
        "recovery_selection_mode": "pure_llm",
        "action_horizon": "1",
        "action_horizon_steps": 1,
        "action_horizon_k": 3,
        "candidate_count": "auto",
        "accepted_outline_prefix": deepcopy(accepted_outline_prefix or []),
        "observation_store": {},
        "outline_lookahead": [],
        "pruned_actions": [],
        "outline_validation_findings": [],
        "candidate_rejection_feedback": deepcopy(candidate_rejection_feedback or []),
        "primitive_escalation_diagnostics": [],
        "symbolic_resources": {row["resource_jid"]: deepcopy(row) for row in resources},
        "symbolic_parts": {row["part_name"]: deepcopy(row) for row in parts},
    }
    prompt_input = multi_turn_prompts.build_multi_turn_phase_prompt_input(
        phase="outline",
        llm_input=llm_input,
        session_state=session_state,
        recovery_resources=recovery_resources,
        current_recovery_blockers=[
            {"summary": "REQ_1_T3 is blocked for LCP at station"}
        ],
    )
    return multi_turn_prompts.render_multi_turn_phase_prompt(prompt_input)


def _candidate_event(event_name: str, *, outline_id: str | None = None) -> dict[str, Any]:
    return {
        "outline_id": outline_id or event_name,
        "event_name": event_name,
        "resource_jid": "xarm6@localhost",
        "part_name": "LG",
        "expected_start_state": {"resource_state": "idle", "held_part": None},
        "expected_end_state": {"resource_state": "picked", "held_part": "LG"},
        "rationale": f"{event_name} rationale",
    }


def _candidate_session(
    *,
    recovery_selection_mode: str,
    action_horizon: str,
    candidate_count: int | str = "auto",
) -> dict[str, Any]:
    action_horizon_setting: int | str = 3 if action_horizon == "k" else action_horizon
    session = {
        "accepted_outline_prefix": [],
        "candidate_rejection_feedback": [],
        "candidate_prune_history": {},
        "recovery_selection_mode": recovery_selection_mode,
        "outline_validation_findings": [],
        "pruned_actions": [],
        "symbolic_resources": {},
        "symbolic_parts": {},
    }
    session.update(_recovery_action_horizon_fields(action_horizon_setting))
    session.update(_recovery_candidate_count_fields(candidate_count))
    return session


async def _run_mocked_candidate_handler(
    *,
    session_state: dict[str, Any],
    parsed_response: dict[str, Any],
    invalid_candidate_indexes: set[int] | None = None,
) -> tuple[str, dict[str, Any]]:
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (
        multi_turn_outline_generation,
    )

    def _mock_remaining_counts(
        *,
        session_state: dict[str, Any],
        prepared_recovery_request: dict[str, Any],
    ) -> tuple[int, int]:
        del prepared_recovery_request
        accepted_count = len(session_state.get("accepted_outline_prefix") or [])
        return (0, 0) if accepted_count >= 2 else (1, 0)

    async def _mock_validate_candidate_sequence(
        *,
        candidate: dict[str, Any],
        sequence_index: int,
        session_state: dict[str, Any],
        prepared_recovery_request: dict[str, Any],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        events = [
            deepcopy(row)
            for row in (candidate.get("surface_events") or [])
            if isinstance(row, dict)
        ]
        committed = [
            multi_turn_mode._commit_selected_candidate_task(
                task=event,
                sequence_index=sequence_index + index,
            )
            for index, event in enumerate(events)
        ]
        candidate_index = int(candidate.get("candidate_index") or 0)
        invalid = candidate_index in set(invalid_candidate_indexes or set())
        findings = (
            [
                multi_turn_mode._candidate_schema_finding(
                    task=events[0] if events else {},
                    reason="selected candidate test rejection",
                    evidence={"candidate_index": candidate_index},
                )
            ]
            if invalid
            else []
        )
        return {
            "candidate_index": candidate_index,
            "valid": not invalid,
            "task": deepcopy(events[0] if events else {}),
            "surface_events": events,
            "validated_events": deepcopy(events),
            "committed_events": committed,
            "grounded_actions": [],
            "validation_findings": findings,
            "validation_stages": [],
            "remaining_blocked_issues": 0,
            "resource_switch_count": 0,
            "pa_state_fingerprint": multi_turn_outline_generation._pa_state_fingerprint(
                session_state=session_state,
                prepared_recovery_request=prepared_recovery_request,
            ),
        }

    with (
        patch.object(multi_turn_mode, "_active_pruned_actions", return_value=[]),
        patch.object(
            multi_turn_outline_generation,
            "_validate_candidate_sequence",
            side_effect=_mock_validate_candidate_sequence,
        ),
        patch.object(
            multi_turn_mode,
            "_remaining_blocked_issue_counts",
            side_effect=_mock_remaining_counts,
        ),
        patch.object(multi_turn_mode, "_apply_task_effects_to_symbolic_state", return_value=None),
        patch.object(multi_turn_mode, "_promote_durable_candidate_rejections", return_value=None),
    ):
        return await multi_turn_outline_generation._handle_outline_incremental_candidates_validated(
            session_state=session_state,
            parsed_response=parsed_response,
            prepared_recovery_request={},
            planner=object(),
        )


def test_case3_actual_outline_uses_runtime_failure_context(tmp_path: Path) -> None:
    runtime_context = _load_case3_runtime_context()
    expected_failed_task_id = str(
        dict(runtime_context.get("failure_event") or {}).get("failed_task_id") or ""
    ).strip()
    result = run_recovery_outline_only(
        debug_root=tmp_path,
        scripted_responses=_fixture_outline_responses(),
    )
    trace = result["transition_trace"]
    prepared_recovery_request = result["prepared_recovery_request"]
    failure_context = prepared_recovery_request["failure_context_raw"]
    plan_payload = _load_json(_case3_paths(runtime_context)["plan"])
    assert isinstance(plan_payload, dict)
    failed_task = _task_node_by_id(plan_payload, expected_failed_task_id)
    expected_p_id = sorted(
        part_name
        for part_name, part_row in dict(runtime_context.get("part_tracker") or {}).items()
        if isinstance(part_row, dict)
        and str(part_row.get("state") or "").strip()
        != str(runtime_context.get("goal_state") or "").strip()
    )

    assert trace
    assert failure_context["failed_task_id"] == expected_failed_task_id
    assert failure_context["failed_function_name"] == "place_insert"
    assert prepared_recovery_request["ra_jid"] == failed_task["resource_jid"]
    assert prepared_recovery_request["P_id"] == expected_p_id
    assert result["llm_response_source"] == "mocked_scripted_fixture"
    assert (tmp_path / "recovery_outline").is_dir()
    grounding_request_paths = sorted(
        (tmp_path / "recovery_outline").glob(
            "multi_turn_turn*_grounding_request_*.txt"
        )
    )
    grounding_result_paths = sorted(
        (tmp_path / "recovery_outline").glob(
            "multi_turn_turn*_grounding_result_*.json"
        )
    )
    assert len(grounding_request_paths) == 2
    assert len(grounding_result_paths) == 2
    assert not list(
        (tmp_path / "recovery_outline").glob(
            "multi_turn_turn*_grounding_llm_response_*.txt"
        )
    )
    assert not list(
        (tmp_path / "recovery_outline").glob(
            "multi_turn_turn*_grounding_response_*.txt"
        )
    )

    expected_lg_observation = deepcopy(
        next(
            row
            for row in (runtime_context.get("resource_snapshots") or [])
            if isinstance(row, dict)
            and str(row.get("resource_jid") or "").strip() == "xarm6@localhost"
        )["observations"]["LG"]
    )
    first_grounding_result = json.loads(
        grounding_result_paths[0].read_text(encoding="utf-8")
    )
    second_grounding_result = json.loads(
        grounding_result_paths[1].read_text(encoding="utf-8")
    )
    assert first_grounding_result["effective_decision"] == "observe"
    assert first_grounding_result["next_phase"] == "grounding"
    assert first_grounding_result["llm_response"]["decision"] == "observe"
    assert first_grounding_result["observation_results"][0]["observation_key"] == (
        "observed_pose_LG"
    )
    assert first_grounding_result["observation_results"][0]["output"] == (
        expected_lg_observation
    )
    assert first_grounding_result["observation_store_after_turn"]["observed_pose_LG"] == (
        expected_lg_observation
    )
    assert second_grounding_result["effective_decision"] == "grounded"
    assert second_grounding_result["next_phase"] == "outline"
    assert second_grounding_result["observation_results"] == []
    assert second_grounding_result["observation_store_after_turn"]["observed_pose_LG"] == (
        expected_lg_observation
    )

    first_request_text = grounding_request_paths[0].read_text(encoding="utf-8")
    request_message_text = first_request_text.split("Response Format", maxsplit=1)[0]
    assert "role=system" not in first_request_text
    assert "role=user" in first_request_text
    assert '"name": "multi_turn_grounding_response"' in first_request_text
    assert "multi_turn_grounding_response" not in request_message_text

    grounding_turns = [
        turn
        for turn in result["turns"]
        if isinstance(turn, dict)
        and str(turn.get("phase") or "").strip().lower() == "grounding"
    ]
    assert len(grounding_turns) == 2
    for turn in grounding_turns:
        assert turn["request_artifact_path"] == turn["prompt_artifact_path"]
        assert turn["grounding_result_artifact_path"] == turn["response_artifact_path"]
        assert "llm_response_artifact_path" not in turn

    outline_request_paths = sorted(
        (tmp_path / "recovery_outline").glob(
            "multi_turn_turn*_outline_request_*.txt"
        )
    )
    outline_result_paths = sorted(
        (tmp_path / "recovery_outline").glob(
            "multi_turn_turn*_outline_result_*.json"
        )
    )
    outline_stack_paths = sorted(
        (tmp_path / "recovery_outline").glob(
            "multi_turn_turn*_outline_stack_*.json"
        )
    )
    outline_turns = [
        turn
        for turn in result["turns"]
        if isinstance(turn, dict)
        and str(turn.get("phase") or "").strip().lower() == "outline"
    ]
    assert len(outline_request_paths) == len(outline_turns)
    assert len(outline_result_paths) == len(outline_turns)
    assert len(outline_stack_paths) == len(outline_turns)
    assert not list(
        (tmp_path / "recovery_outline").glob(
            "multi_turn_turn*_outline_llm_response_*.txt"
        )
    )
    assert not list(
        (tmp_path / "recovery_outline").glob(
            "multi_turn_turn*_outline_response_*.txt"
        )
    )
    assert not list(
        (tmp_path / "recovery_outline").glob("multi_turn_turn*_outline_index_*.json")
    )

    first_outline_request = outline_request_paths[0].read_text(encoding="utf-8")
    outline_message_text = first_outline_request.split(
        "Response Format", maxsplit=1
    )[0]
    assert len(outline_message_text) < 6_000
    assert len(first_outline_request) < 12_000
    for private_model_token in (
        "RA-Owned Recovery DES Models",
        "local_event_alphabet",
        "descriptor_fingerprint",
        '"guards"',
        '"updates"',
        "detect_parts",
        "compute_pick_targets",
        "get_current_pose",
        "move_relative",
    ):
        assert private_model_token not in outline_message_text
    expected_pose = dict(expected_lg_observation.get("pose") or {})
    assert '"observed_pose": {' in first_outline_request
    for axis in ("x", "y", "z"):
        assert f'"{axis}": {expected_pose[axis]}' in first_outline_request

    first_outline_result = json.loads(
        outline_result_paths[0].read_text(encoding="utf-8")
    )
    llm_response = first_outline_result["llm_response"]
    assert set(llm_response) == {"thought", "candidate_events"}
    assert len(llm_response["candidate_events"]) == 1
    assert "candidate_evaluation_summary" not in llm_response
    assert len(first_outline_result["candidate_evaluation_summary"]) == 1
    private_models = first_outline_result["private_recovery_des_models"]
    assert set(private_models) == {"ur5e@localhost", "xarm6@localhost"}
    for descriptor in private_models.values():
        assert descriptor["descriptor_fingerprint"]
        assert "events" in descriptor
    for evaluation in first_outline_result["candidate_evaluation_summary"]:
        stages = list(evaluation.get("validation_stages") or [])
        assert [stage["validator_role"] for stage in stages] == [
            "PA",
            "PA",
            "RA",
            "CCA",
        ]
        assert stages[0]["validation_category"] == "syntax_and_grounding_validation"
        assert stages[1]["validation_category"] == "transition_feasibility"
        assert stages[2]["validation_category"] == "physical_feasibility"
        assert stages[3]["validation_category"] == "safety"
        assert stages[0]["mocked"] is False
        assert stages[1]["mocked"] is False
        assert stages[2]["mocked"] is True
        assert stages[3]["mocked"] is True
    assert "selected_transition" in first_outline_result
    assert "transition_trace" in first_outline_result
    for result_path in outline_result_paths:
        outline_result = json.loads(result_path.read_text(encoding="utf-8"))
        result_artifact_paths = outline_result["artifact_paths"]
        assert result_artifact_paths["outline_result_artifact_path"] == str(result_path)
        assert result_artifact_paths["response_artifact_path"] == str(result_path)
        assert result_artifact_paths["request_artifact_path"] == (
            result_artifact_paths["prompt_artifact_path"]
        )
        assert Path(result_artifact_paths["outline_stack_artifact_path"]).exists()
    for stack_path in outline_stack_paths:
        stack_payload = json.loads(stack_path.read_text(encoding="utf-8"))
        assert isinstance(stack_payload, list)
        assert stack_payload
    latest_outline_turn = outline_turns[-1]
    assert latest_outline_turn["request_artifact_path"] == (
        latest_outline_turn["prompt_artifact_path"]
    )
    assert latest_outline_turn["outline_result_artifact_path"] == (
        latest_outline_turn["response_artifact_path"]
    )
    assert "llm_response_artifact_path" not in latest_outline_turn
    assert Path(latest_outline_turn["outline_stack_artifact_path"]).exists()
    assert "turn_index_artifact_path" not in latest_outline_turn


def test_case3_mocked_novel_symbol_sequence_converges_without_semantic_cycles(
    tmp_path: Path,
) -> None:
    result = run_recovery_outline_only(
        debug_root=tmp_path,
        scripted_responses=_novel_symbol_outline_responses(),
    )
    trace = result["transition_trace"]
    session_state = result["multi_turn_session"]

    assert [row["event_name"] for row in trace] == ["evt_q7", "evt_z9", "evt_n4", "evt_v2"]
    assert trace[0]["expected_end_state"]["resource_state"] == "xarm6_clear_state"
    assert trace[1]["expected_end_state"]["part_state"] == "mcp_waiting_recovery"
    assert trace[2]["expected_end_state"]["resource_state"] == "lg_secured"
    assert trace[3]["expected_end_state"]["part_state"] == "lg_restored_state"
    assert "semantic_state_history" not in session_state
    assert session_state["status"] == "ready_for_primitive_generation"
    assert result["llm_response_source"] == "mocked_scripted_fixture"


def test_case3_runtime_context_has_no_expected_recovery_answers() -> None:
    runtime_context = _load_case3_runtime_context()
    paths = _case3_paths(runtime_context)
    manifest = _load_json(paths["bundle_manifest"])

    assert isinstance(manifest, dict)
    artifacts = dict(manifest.get("artifacts") or {})
    bundle_root = paths["bundle_manifest"].parent
    assert paths["plan"] == bundle_root / str(artifacts["plan_json"])
    assert paths["requirements"] == bundle_root / str(artifacts["requirements_json"])
    assert paths["tools"] == bundle_root / str(artifacts["tools_json"])
    assert paths["safety_logic"] == bundle_root / str(artifacts["safety_logic_json"])

    serialized_context = json.dumps(runtime_context, sort_keys=True)
    for forbidden_token in (
        "candidate_events",
        "candidate_traces",
        "selected_candidate_index",
        "recover_to_home_idle",
        "place_mcp_to_prusa_mk4_2_temp",
        "pick_lg_from_observed_pose",
        "place_lg_to_assembly_board_v1",
    ):
        assert forbidden_token not in serialized_context


def test_case3_dryrun_uses_production_ra_and_cca_validator_functions() -> None:
    assert (
        FakeRecoveryRobot.check_recovery_physical_feasibility
        is RobotAgent.check_recovery_physical_feasibility
    )
    assert (
        FakeRecoveryRobot._is_pose_in_workspace
        is RobotAgent._is_pose_in_workspace
    )
    assert (
        FakeRecoveryRobot.recovery_physical_validation_snapshot
        is RobotAgent.recovery_physical_validation_snapshot
    )
    assert (
        validate_outline_macro_recovery_safety
        is outline_macro_safety_module.validate_outline_macro_recovery_safety
    )


def test_case3_recovery_context_has_no_nominal_terminal_or_stale_resource_location() -> None:
    _fixture, _product_agent, planner, prepared_recovery_request = asyncio.run(
        _prepare_recovery_dryrun_harness(scripted_responses=_fixture_outline_responses())
    )

    marked_reentry_context = dict(
        prepared_recovery_request.get("marked_reentry_context") or {}
    )
    assert all(
        str(condition.get("kind") or "") != "focused_resource_terminal_state"
        for condition in (marked_reentry_context.get("marked_reentry_conditions") or [])
        if isinstance(condition, dict)
    )
    continuation_conditions = list(
        dict(prepared_recovery_request.get("llm_input") or {})
        .get("modeled_continuation_gap", {})
        .get("unmet_continuation_conditions", [])
    )
    assert all(
        str(condition.get("kind") or "") != "focused_resource_terminal_state"
        for condition in continuation_conditions
        if isinstance(condition, dict)
    )

    context_resources = {
        str(row.get("resource_jid") or ""): row
        for row in (
            dict(prepared_recovery_request.get("context_summary") or {})
            .get("current_product_state", {})
            .get("resources", [])
        )
        if isinstance(row, dict)
    }
    assert context_resources["ur5e@localhost"]["current_location"] is None
    assert context_resources["ur5e@localhost"]["current_location_basis"] == "current_pose_only"
    assert context_resources["xarm6@localhost"]["current_location"] == "assembly_board-v1"
    assert context_resources["xarm6@localhost"]["current_location_basis"] == "recovery_snapshot"

    explicit_location, explicit_basis = planner._resolve_resource_location(
        resource_jid="ur5e@localhost",
        snapshot={"current_location": "prusa-mk3"},
        profile=object(),
        prepared_recovery_request=prepared_recovery_request,
    )
    assert explicit_location == "prusa-mk3"
    assert explicit_basis == "recovery_snapshot"

    loaded_rules = {
        str(rule.get("id") or rule.get("rule_id") or ""): rule
        for rule in (prepared_recovery_request.get("loaded_safety_rules") or [])
        if isinstance(rule, dict)
    }
    safe_1_resources = {
        str(dict(ap.get("selector") or {}).get("resource") or "")
        for ap in (dict(loaded_rules["SAFE_1"]).get("recovery_aps") or [])
        if isinstance(ap, dict)
    }
    safe_2_resources = {
        str(dict(ap.get("selector") or {}).get("resource") or "")
        for ap in (dict(loaded_rules["SAFE_2"]).get("recovery_aps") or [])
        if isinstance(ap, dict)
    }
    assert safe_1_resources == {"any"}
    assert {"ur5e", "xarm6"}.issubset(safe_2_resources)

    session_state = deepcopy(prepared_recovery_request["multi_turn_session_seed"])
    grounding_prompt = multi_turn_prompts.render_multi_turn_phase_prompt(
        multi_turn_prompts.build_multi_turn_phase_prompt_input(
            phase="grounding",
            llm_input=prepared_recovery_request["llm_input"],
            session_state=session_state,
            recovery_resources=prepared_recovery_request["recovery_resources"],
        )
    )
    assert "Assembly Requirements" not in grounding_prompt
    assert "pending_nominal_tasks" not in grounding_prompt
    assert "LG by xarm6" not in grounding_prompt
    assert "MCP by ur5e" not in grounding_prompt
    assert (
        "LG place_approach must occur before MCP place_approach to assembly_board-v1."
        in grounding_prompt
    )

    session_state["current_phase"] = "outline"
    outline_prompt_input, outline_prompt = multi_turn_mode._build_phase_prompt(
        prepared_recovery_request,
        session_state,
    )
    occupancy_blockers = [
        blocker
        for blocker in (outline_prompt_input.get("current_recovery_blockers") or [])
        if str(blocker.get("kind") or "") == "safety_destination_occupancy"
    ]
    assert occupancy_blockers
    assert any(
        str(blocker.get("summary") or "")
        == "xarm6@localhost currently occupies assembly_board-v1 under SAFE_2"
        for blocker in occupancy_blockers
    )
    assert (
        "xarm6@localhost currently occupies assembly_board-v1 under SAFE_2"
        not in outline_prompt
    )
    assert '"resource_location": "assembly_board-v1"' in outline_prompt
    assert (
        "SAFE_2: ur5e and xarm6 must not both be in the assembly board destination "
        "area at the same time."
        in outline_prompt
    )
    assert "restore LG to assembly_board-v1" in outline_prompt

    session_state["symbolic_parts"]["LG"].update(
        {
            "current_state": "assembled",
            "part_state": "assembled",
            "current_location": "assembly_board-v1",
            "part_location": "assembly_board-v1",
            "current_holder_resource_jid": "ur5e@localhost",
            "part_holder_resource_jid": "ur5e@localhost",
        }
    )
    remaining_findings, remaining_conditions = multi_turn_mode._remaining_blocked_issue_counts(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert remaining_findings == 0
    assert remaining_conditions == 1
    assert session_state["symbolic_resources"]["xarm6@localhost"]["current_state"] == "failed"

    multi_turn_mode._apply_task_effects_to_symbolic_state(
        {
            "outline_id": "novel_xarm6_clear",
            "event_name": "evt_q7",
            "resource_jid": "xarm6@localhost",
            "expected_end_state": {
                "resource_state": "xarm6_clear_state",
                "resource_location": "home",
            },
        },
        session_state,
    )
    remaining_findings, remaining_conditions = multi_turn_mode._remaining_blocked_issue_counts(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert remaining_findings == 0
    assert remaining_conditions == 0
    assert session_state["symbolic_resources"]["xarm6@localhost"]["current_state"] == (
        "xarm6_clear_state"
    )


def test_non_case3_prompt_contains_only_supplied_runtime_facts() -> None:
    prompt = _render_non_case3_candidate_prompt()

    for supplied_token in (
        "REQ_1_T3",
        "ur5e@localhost",
        "xarm6@localhost",
        "LCP",
        "MG",
        "station",
        "Assembly Station",
        "prusa-mk4-1",
        "prusa-mk4-2",
        "SAFE_1",
        "LCP place_approach must occur before MG place_approach to station.",
    ):
        assert supplied_token in prompt
    for leaked_token in (
        "LG",
        "MCP",
        "REQ_2_T4",
        "recover_to_home_idle",
        "place_mcp_to_prusa_mk4_2_temp",
        "pick_lg_from_observed_pose",
        "place_lg_to_assembly_board_v1",
    ):
        assert leaked_token not in prompt

    assert "LCP by xarm6@localhost" not in prompt
    assert "MG by ur5e@localhost" not in prompt
    assert "Open Guard / Marking Conditions" not in prompt
    assert "Grounded Event Facts" not in prompt
    assert "Marked-State Conditions" not in prompt
    assert "Assembly Requirements" not in prompt
    assert "Candidate Count:" not in prompt
    assert "Action Horizon:" not in prompt
    assert "Selection Mode:" not in prompt
    assert "current_pose(x=" not in prompt
    assert prompt.count('"x": -0.25') == 1
    assert "Recovery Goals" in prompt
    assert "restore LCP to station" in prompt
    assert "restore MG" not in prompt


def test_candidate_prompt_resource_and_part_order_is_stable() -> None:
    assert _render_non_case3_candidate_prompt() == _render_non_case3_candidate_prompt(
        reverse_order=True
    )


def test_acquire_and_release_propagate_part_holder_and_location() -> None:
    acquire = {
        "resource_jid": "xarm6@localhost",
        "part_name": "LCP",
        "expected_end_state": {
            "resource_state": "picked",
            "held_part": "LCP",
            "part_state": "in_gripper",
            "part_location": "xarm6@localhost_gripper",
        },
    }
    release = {
        "resource_jid": "xarm6@localhost",
        "part_name": "LCP",
        "expected_end_state": {
            "resource_state": "positioned",
            "held_part": None,
            "part_state": "assembled",
            "part_location": "station",
        },
    }
    session_state = {
        "symbolic_resources": {
            "xarm6@localhost": {
                "resource_jid": "xarm6@localhost",
                "current_state": "idle",
                "held_part": None,
            }
        },
        "symbolic_parts": {
            "LCP": {
                "part_name": "LCP",
                "current_state": "misplaced",
                "current_location": "prusa-mk4-1",
                "current_holder_resource_jid": None,
            }
        },
    }

    multi_turn_mode._apply_task_effects_to_symbolic_state(acquire, session_state)
    acquired_part = session_state["symbolic_parts"]["LCP"]
    assert acquired_part["part_holder_resource_jid"] == "xarm6@localhost"
    assert acquired_part["current_holder_resource_jid"] == "xarm6@localhost"
    assert acquired_part["part_location"] == "xarm6@localhost_gripper"
    assert acquired_part["current_location"] == "xarm6@localhost_gripper"

    multi_turn_mode._apply_task_effects_to_symbolic_state(release, session_state)
    released_part = session_state["symbolic_parts"]["LCP"]
    assert released_part["part_holder_resource_jid"] is None
    assert released_part["current_holder_resource_jid"] is None
    assert released_part["part_location"] == "station"
    assert released_part["current_location"] == "station"

    projected_resources = deepcopy(session_state["symbolic_resources"])
    projected_parts = {
        "LCP": {
            "part_name": "LCP",
            "current_state": "misplaced",
            "current_location": "prusa-mk4-1",
            "current_holder_resource_jid": None,
        }
    }
    multi_turn_outline_state._apply_outline_task_effects(
        acquire,
        resources_by_jid=projected_resources,
        parts_by_name=projected_parts,
        task_type="part_handling",
    )
    assert projected_parts["LCP"]["current_location"] == "xarm6@localhost_gripper"
    multi_turn_outline_state._apply_outline_task_effects(
        release,
        resources_by_jid=projected_resources,
        parts_by_name=projected_parts,
        task_type="part_handling",
    )
    assert projected_parts["LCP"]["current_holder_resource_jid"] is None
    assert projected_parts["LCP"]["current_location"] == "station"


def test_pa_custody_propagation_does_not_infer_resource_mechanism_or_location() -> None:
    acquire = {
        "resource_jid": "handler@localhost",
        "part_name": "LCP",
        "expected_end_state": {
            "resource_state": "loaded",
            "held_part": "LCP",
            "part_state": "controlled",
        },
    }
    session_state = {
        "recovery_des_models": {
            "handler@localhost": {
                "state_variables": {
                    "resource_state": {"scope": "resource"},
                    "held_part": {"scope": "resource"},
                    "part_state": {"scope": "part"},
                    "part_location": {"scope": "part"},
                }
            }
        },
        "symbolic_resources": {
            "handler@localhost": {
                "resource_jid": "handler@localhost",
                "current_state": "idle",
                "held_part": None,
                "gripper_state": "open",
            }
        },
        "symbolic_parts": {
            "LCP": {
                "part_name": "LCP",
                "current_state": "available",
                "current_location": "station",
                "part_location": "station",
                "current_holder_resource_jid": None,
            }
        },
    }

    multi_turn_mode._apply_task_effects_to_symbolic_state(acquire, session_state)

    resource = session_state["symbolic_resources"]["handler@localhost"]
    part = session_state["symbolic_parts"]["LCP"]
    assert resource["held_part"] == "LCP"
    assert resource["gripper_state"] == "open"
    assert part["current_holder_resource_jid"] == "handler@localhost"
    assert part["part_location"] == "station"
    assert part["current_location"] == "station"

    projected_resources = {
        "handler@localhost": {
            "resource_jid": "handler@localhost",
            "current_state": "idle",
            "held_part": None,
            "gripper_state": "open",
        }
    }
    projected_parts = {
        "LCP": {
            "part_name": "LCP",
            "current_state": "available",
            "current_location": "station",
            "part_location": "station",
            "current_holder_resource_jid": None,
        }
    }
    multi_turn_outline_state._apply_outline_task_effects(
        acquire,
        resources_by_jid=projected_resources,
        parts_by_name=projected_parts,
        task_type="part_handling",
        state_field_scopes={
            "resource_state": "resource",
            "held_part": "resource",
            "part_state": "part",
            "part_location": "part",
        },
    )
    assert projected_resources["handler@localhost"]["gripper_state"] == "open"
    assert projected_parts["LCP"]["current_holder_resource_jid"] == (
        "handler@localhost"
    )
    assert projected_parts["LCP"]["part_location"] == "station"


def test_candidate_completeness_uses_responsible_ra_state_variables() -> None:
    common = {
        "outline_id": "resource_specific_fields",
        "event_name": "authored_event",
        "resource_jid": "resource@localhost",
        "part_name": "LCP",
    }
    resource_only_states = {
        "resource_state": {"scope": "resource", "domain": ["idle", "ready"]}
    }
    findings = multi_turn_mode._candidate_state_completeness_findings(
        candidate_task=common,
        part_name="LCP",
        start_state={"resource_state": "idle"},
        end_state={"resource_state": "ready"},
        state_variables=resource_only_states,
    )
    assert findings == []

    custody_states = {
        **resource_only_states,
        "held_part": {"scope": "resource", "domain": [None, "LCP"]},
        "part_state": {"scope": "part", "domain": ["available", "controlled"]},
        "part_location": {"scope": "part", "domain": ["station"]},
    }
    findings = multi_turn_mode._candidate_state_completeness_findings(
        candidate_task=common,
        part_name="LCP",
        start_state={
            "resource_state": "idle",
            "held_part": None,
            "part_state": "available",
        },
        end_state={
            "resource_state": "ready",
            "held_part": "LCP",
            "part_state": "controlled",
        },
        state_variables=custody_states,
    )
    assert findings[0]["constraint_code"] == "candidate_schema_violation"
    assert findings[0]["evidence"]["missing"] == ["part_location"]

    response_schema = multi_turn_prompts.multi_turn_phase_response_schema(
        "outline",
        outline_mode="incremental_candidates_validated",
        recovery_selection_mode="neurosymbolic",
        declared_state_variables=custody_states,
    )
    outline_event_schema = response_schema["schema"]["$defs"]["outline_event"]
    assert "allOf" not in outline_event_schema


def test_candidate_prompt_has_no_numeric_selected_index_example() -> None:
    prompt = _render_non_case3_candidate_prompt()
    schema = multi_turn_prompts.multi_turn_phase_response_schema(
        "outline",
        outline_mode="incremental_candidates_validated",
        candidate_bound=8,
        recovery_selection_mode="pure_llm",
        action_horizon="1",
    )["schema"]

    assert '"selected_candidate_index": 0' not in prompt
    assert "integer `selected_candidate_index`" in prompt
    assert "Outline Candidate Contract" not in prompt
    assert "Recovery Candidate Rules" in prompt
    assert "`candidate_events` must contain exactly three candidates" in prompt
    assert (
        "Resource-only actions may include `resource_state`, `resource_location`, and `held_part`"
        in prompt
    )
    assert "`part_state` and `part_location` require `part_name`" in prompt
    assert "selected_candidate_index" in schema["required"]
    assert schema["properties"]["selected_candidate_index"]["type"] == "integer"
    assert schema["properties"]["selected_candidate_index"]["maximum"] == 2
    assert schema["properties"]["candidate_events"]["minItems"] == 3
    assert schema["properties"]["candidate_events"]["maxItems"] == 3


def test_candidate_prompt_allows_new_event_and_state_symbols_without_prefix_anchoring() -> None:
    accepted_event_name = "previously_authored_event"
    prompt = _render_non_case3_candidate_prompt(
        accepted_outline_prefix=[
            {
                "outline_id": "RECOVERY_SEQ1",
                "event_name": accepted_event_name,
                "resource_jid": "xarm6@localhost",
                "expected_start_state": {"resource_state": "failed"},
                "expected_end_state": {
                    "resource_state": "new_clear_state",
                    "resource_location": "home",
                },
            }
        ]
    )

    assert "Accepted Transition Prefix" not in prompt
    assert accepted_event_name not in prompt
    assert "You may author a new `event_name`" in prompt
    assert "optional new `resource_state` or `part_state` values" in prompt
    assert "A new state name has no meaning by itself" in prompt
    assert "must accompany a concrete state effect" in prompt
    assert '"resource_location": "station"' in prompt
    assert "Use only supplied resources, parts, locations, named poses, predicates" in prompt
    assert "Outline Candidate Contract" not in prompt


def test_production_cca_rejects_safe1_out_of_order_mcp_placement() -> None:
    _fixture, _product_agent, _planner, prepared_recovery_request = asyncio.run(
        _prepare_recovery_dryrun_harness(scripted_responses=_fixture_outline_responses())
    )
    session_state = deepcopy(prepared_recovery_request["multi_turn_session_seed"])
    session_state["symbolic_resources"]["xarm6@localhost"].update(
        {
            "current_state": "idle",
            "resource_state": "idle",
            "current_location": "home",
            "resource_location": "home",
            "occupancy": {"location": "home"},
        }
    )
    task = {
        "outline_id": "mcp_early",
        "event_name": "evt_mcp_early",
        "resource_jid": "ur5e@localhost",
        "part_name": "MCP",
        "expected_start_state": {
            "resource_state": "picked",
            "held_part": "MCP",
            "part_state": "in_gripper",
            "part_location": "ur5e@localhost_gripper",
        },
        "expected_end_state": {
            "resource_state": "idle",
            "resource_location": "assembly_board-v1",
            "held_part": None,
            "part_state": "assembled",
            "part_location": "assembly_board-v1",
        },
    }
    safety_input = recovery_validation_service.build_recovery_safety_validation_input(
        task=task,
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    safety_result = validate_outline_macro_recovery_safety(**safety_input)

    assert safety_result["is_safe"] is False
    assert any(
        str(finding.get("rule_id") or "") == "SAFE_1"
        and str(finding.get("constraint_code") or "") == "safety_rule_violation"
        for finding in (safety_result.get("findings") or [])
    )


def test_novel_event_and_state_symbols_use_declared_effects_and_clear_safe2() -> None:
    from cais_spade_llm.resources.robot import robot_primitives, robot_profile

    _fixture, _product_agent, planner, prepared_recovery_request = asyncio.run(
        _prepare_recovery_dryrun_harness(scripted_responses=_fixture_outline_responses())
    )
    session_state = deepcopy(prepared_recovery_request["multi_turn_session_seed"])

    def _validate(candidate: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        validated, schema_findings = multi_turn_mode._derive_candidate_outline_task(
            candidate_task=deepcopy(candidate),
            session_state=session_state,
            prepared_recovery_request=prepared_recovery_request,
        )
        assert schema_findings == []
        assert validated is not None
        findings, _grounded_action = multi_turn_mode._validate_single_outline_task(
            planner=planner,
            task=validated,
            session_state=session_state,
            prepared_recovery_request=prepared_recovery_request,
        )
        return validated, findings

    stage_mcp = {
        "outline_id": "novel_stage_mcp",
        "event_name": "evt_z9",
        "resource_jid": "ur5e@localhost",
        "part_name": "MCP",
        "expected_start_state": {
            "resource_state": "picked",
            "held_part": "MCP",
            "part_state": "in_gripper",
            "part_location": "ur5e@localhost_gripper",
        },
        "expected_end_state": {
            "resource_state": "mcp_buffer_clear",
            "held_part": None,
            "part_state": "mcp_waiting_recovery",
            "part_location": "prusa-mk4-2",
        },
        "rationale": "Declare a reachable release that frees the gripper.",
    }
    validated_stage, findings = _validate(stage_mcp)
    assert findings == []
    assert robot_profile._robot_event_family(validated_stage) == "place"
    assert robot_primitives._robot_event_family(validated_stage) == "place"
    multi_turn_mode._apply_task_effects_to_symbolic_state(validated_stage, session_state)
    assert session_state["symbolic_resources"]["ur5e@localhost"]["current_state"] == (
        "mcp_buffer_clear"
    )
    assert session_state["symbolic_parts"]["MCP"]["current_state"] == (
        "mcp_waiting_recovery"
    )
    session_state["symbolic_parts"]["LG"]["observed_pose"] = {
        "x": 0.0,
        "y": 0.2,
        "z": 1.035,
    }

    acquire_lg = {
        "outline_id": "novel_acquire_lg",
        "event_name": "evt_n4",
        "resource_jid": "ur5e@localhost",
        "part_name": "LG",
        "expected_start_state": {
            "resource_state": "mcp_buffer_clear",
            "held_part": None,
            "part_state": "misplaced",
        },
        "expected_end_state": {
            "resource_state": "lg_secured",
            "held_part": "LG",
            "part_state": "lg_under_recovery_control",
            "part_location": "ur5e@localhost_gripper",
        },
        "rationale": "Acquire the observed part using its grounded pose.",
    }
    validated_acquire, findings = _validate(acquire_lg)
    assert findings == []
    assert robot_profile._robot_event_family(validated_acquire) == "pick"
    assert robot_primitives._robot_event_family(validated_acquire) == "pick"
    multi_turn_mode._apply_task_effects_to_symbolic_state(validated_acquire, session_state)

    restore_lg = {
        "outline_id": "novel_restore_lg",
        "event_name": "evt_v2",
        "resource_jid": "ur5e@localhost",
        "part_name": "LG",
        "expected_start_state": {
            "resource_state": "lg_secured",
            "held_part": "LG",
            "part_state": "lg_under_recovery_control",
            "part_location": "ur5e@localhost_gripper",
        },
        "expected_end_state": {
            "resource_state": "lg_recovery_complete",
            "resource_location": "assembly_board-v1",
            "held_part": None,
            "part_state": "lg_restored_state",
            "part_location": "assembly_board-v1",
        },
        "rationale": "Restore the held part to its supplied goal location.",
    }
    validated_restore, blocked_findings = _validate(restore_lg)
    assert blocked_findings == []
    safety_input = recovery_validation_service.build_recovery_safety_validation_input(
        task=validated_restore,
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    safety_result = validate_outline_macro_recovery_safety(**safety_input)
    assert any(
        str(finding.get("constraint_code") or "") == "safety_rule_violation"
        and str(finding.get("rule_id") or "") == "SAFE_2"
        for finding in (safety_result.get("findings") or [])
    )

    clear_xarm6 = {
        "outline_id": "novel_clear_xarm6",
        "event_name": "evt_q7",
        "resource_jid": "xarm6@localhost",
        "expected_start_state": {
            "resource_state": "failed",
            "held_part": None,
            "resource_location": "assembly_board-v1",
        },
        "expected_end_state": {
            "resource_state": "xarm6_clear_state",
            "held_part": None,
            "resource_location": "home",
        },
        "rationale": "Leave the explicitly occupied protected destination.",
    }
    validated_clear, findings = _validate(clear_xarm6)
    assert findings == []
    assert robot_profile._robot_event_family(validated_clear) == "home"
    assert robot_primitives._robot_event_family(validated_clear) == "home"
    multi_turn_mode._apply_task_effects_to_symbolic_state(validated_clear, session_state)

    validated_restore, findings = _validate(restore_lg)
    assert findings == []
    safety_input = recovery_validation_service.build_recovery_safety_validation_input(
        task=validated_restore,
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert validate_outline_macro_recovery_safety(**safety_input)["is_safe"] is True
    multi_turn_mode._apply_task_effects_to_symbolic_state(validated_restore, session_state)
    remaining_findings, remaining_conditions = multi_turn_mode._remaining_blocked_issue_counts(
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert remaining_findings == 0
    assert remaining_conditions == 0
    assert session_state["symbolic_resources"]["xarm6@localhost"]["current_state"] == (
        "xarm6_clear_state"
    )
    assert session_state["symbolic_parts"]["LG"]["current_state"] == "lg_restored_state"

    assert "semantic_state_history" not in session_state


def test_ra_declared_gripper_state_survives_ambiguity_revision() -> None:
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (
        multi_turn_outline_generation,
    )

    _fixture, _product_agent, planner, prepared_recovery_request = asyncio.run(
        _prepare_recovery_dryrun_harness(scripted_responses=_fixture_outline_responses())
    )
    session_state = deepcopy(prepared_recovery_request["multi_turn_session_seed"])
    session_state["recovery_selection_mode"] = "neurosymbolic"
    session_state["candidate_count"] = "auto"

    clear_xarm6 = {
        "outline_id": "clear_xarm6_with_declared_gripper_state",
        "event_name": "clear_board_home",
        "resource_jid": "xarm6@localhost",
        "expected_start_state": {
            "resource_state": "failed",
            "resource_location": "assembly_board-v1",
            "held_part": None,
            "gripper_state": "open",
        },
        "expected_end_state": {
            "resource_state": "home",
            "resource_location": "home",
            "held_part": None,
            "gripper_state": "open",
        },
        "rationale": "Clear the explicitly occupied destination.",
    }
    decision, clear_turn = asyncio.run(
        multi_turn_outline_generation._handle_outline_incremental_candidates_validated(
            session_state=session_state,
            parsed_response={
                "thought": "clear occupancy",
                "candidate_events": [clear_xarm6],
            },
            prepared_recovery_request=prepared_recovery_request,
            planner=planner,
        )
    )
    assert decision == "need_next_task"
    assert [
        stage["status"]
        for stage in clear_turn["candidate_evaluations"][0]["validation_stages"]
    ] == ["passed", "passed", "passed", "passed"]

    def _stage_candidate(location: str) -> dict[str, Any]:
        return {
            "outline_id": f"stage_mcp_{location}",
            "event_name": f"stage_mcp_at_{location}",
            "resource_jid": "ur5e@localhost",
            "part_name": "MCP",
            "expected_start_state": {
                "resource_state": "picked",
                "resource_location": None,
                "held_part": "MCP",
                "part_state": "in_gripper",
                "part_location": "ur5e@localhost_gripper",
                "gripper_state": "closed",
            },
            "expected_end_state": {
                "resource_state": "placed",
                "resource_location": location,
                "held_part": None,
                "part_state": "placed",
                "part_location": location,
                "gripper_state": "open",
            },
            "rationale": "Release MCP at one exact reachable staging location.",
        }

    stage_prusa_mk3 = _stage_candidate("prusa-mk3")
    stage_prusa_mk4_1 = _stage_candidate("prusa-mk4-1")
    decision, _ambiguous_turn = asyncio.run(
        multi_turn_outline_generation._handle_outline_incremental_candidates_validated(
            session_state=session_state,
            parsed_response={
                "thought": "compare staging successors",
                "candidate_events": [stage_prusa_mk3, stage_prusa_mk4_1],
            },
            prepared_recovery_request=prepared_recovery_request,
            planner=planner,
        )
    )
    assert decision == "selection_ambiguous"
    assert len(session_state["accepted_outline_prefix"]) == 1
    assert session_state["active_selection_ambiguity_feedback"][
        "constraint_code"
    ] == "selection_ambiguous"

    decision, selected_turn = asyncio.run(
        multi_turn_outline_generation._handle_outline_incremental_candidates_validated(
            session_state=session_state,
            parsed_response={
                "thought": "one representative",
                "candidate_events": [stage_prusa_mk3],
            },
            prepared_recovery_request=prepared_recovery_request,
            planner=planner,
        )
    )
    assert decision == "need_next_task"
    assert selected_turn["selected_transition"]["event_name"] == (
        "stage_mcp_at_prusa-mk3"
    )
    assert session_state["symbolic_resources"]["ur5e@localhost"][
        "gripper_state"
    ] == "open"
    assert session_state["selection_revision_count"] == 0
    assert session_state["active_selection_ambiguity_feedback"] == {}

    projection_seed = deepcopy(prepared_recovery_request["multi_turn_session_seed"])
    safety_input = recovery_validation_service.build_recovery_safety_validation_input(
        task=stage_prusa_mk3,
        session_state=projection_seed,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert safety_input["projected_resources"]["ur5e@localhost"][
        "gripper_state"
    ] == "open"


def test_case3_workspace_rejection_and_nonprogressing_home_remain_authoritative() -> None:
    from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery.modes import (
        multi_turn_outline_generation,
    )

    _fixture, _product_agent, planner, prepared_recovery_request = asyncio.run(
        _prepare_recovery_dryrun_harness(scripted_responses=_fixture_outline_responses())
    )
    session_state = deepcopy(prepared_recovery_request["multi_turn_session_seed"])
    session_state["recovery_selection_mode"] = "neurosymbolic"
    session_state["candidate_count"] = "auto"
    clear_xarm6 = {
        "outline_id": "RECOVERY_SEQ1",
        "event_name": "clear_board_home",
        "resource_jid": "xarm6@localhost",
        "expected_start_state": {
            "resource_state": "failed",
            "resource_location": "assembly_board-v1",
            "held_part": None,
        },
        "expected_end_state": {
            "resource_state": "home",
            "resource_location": "home",
            "held_part": None,
        },
        "rationale": "Clear the occupied board.",
    }
    session_state["accepted_outline_prefix"] = [deepcopy(clear_xarm6)]
    multi_turn_mode._apply_task_effects_to_symbolic_state(clear_xarm6, session_state)
    session_state["symbolic_parts"]["LG"]["observed_pose"] = {
        "x": 0.0,
        "y": 0.2,
        "z": 1.035,
    }

    acquire_lg_with_xarm6 = {
        "outline_id": "xarm6_acquire_lg",
        "event_name": "acquire_lg",
        "resource_jid": "xarm6@localhost",
        "part_name": "LG",
        "expected_start_state": {
            "resource_state": "home",
            "resource_location": "home",
            "held_part": None,
            "part_state": "misplaced",
            "part_location": None,
        },
        "expected_end_state": {
            "resource_state": "picked",
            "resource_location": None,
            "held_part": "LG",
            "part_state": "in_gripper",
            "part_location": "xarm6@localhost_gripper",
        },
        "rationale": "Attempt the observed LG pose.",
    }
    ur5e_home_with_mcp = {
        "outline_id": "ur5e_home_with_mcp",
        "event_name": "move_home_with_mcp",
        "resource_jid": "ur5e@localhost",
        "part_name": "MCP",
        "expected_start_state": {
            "resource_state": "picked",
            "resource_location": None,
            "held_part": "MCP",
            "part_state": "in_gripper",
            "part_location": "ur5e@localhost_gripper",
        },
        "expected_end_state": {
            "resource_state": "home",
            "resource_location": "home",
            "held_part": "MCP",
            "part_state": "in_gripper",
            "part_location": "ur5e@localhost_gripper",
        },
        "rationale": "Retain MCP while moving to home.",
    }
    decision, turn_entry = asyncio.run(
        multi_turn_outline_generation._handle_outline_incremental_candidates_validated(
            session_state=session_state,
            parsed_response={
                "thought": "validate physical and symbolic alternatives",
                "candidate_events": [acquire_lg_with_xarm6, ur5e_home_with_mcp],
            },
            prepared_recovery_request=prepared_recovery_request,
            planner=planner,
        )
    )
    evaluations = turn_entry["candidate_evaluations"]
    assert evaluations[0]["validation_findings"][0]["constraint_code"] == (
        "workspace_unreachable"
    )
    assert evaluations[1]["valid"] is True
    assert evaluations[1]["selection_status"] == "excluded_no_progress"
    assert decision == "need_revision"


def test_pa_rejects_label_only_change_but_allows_prior_physical_arrangement() -> None:
    _fixture, _product_agent, planner, prepared_recovery_request = asyncio.run(
        _prepare_recovery_dryrun_harness(scripted_responses=_fixture_outline_responses())
    )
    session_state = deepcopy(prepared_recovery_request["multi_turn_session_seed"])

    label_only = {
        "outline_id": "label_only",
        "event_name": "evt_l1",
        "resource_jid": "xarm6@localhost",
        "expected_start_state": {
            "resource_state": "failed",
            "resource_location": "assembly_board-v1",
        },
        "expected_end_state": {
            "resource_state": "new_label_only",
            "resource_location": "assembly_board-v1",
        },
        "rationale": "Only relabel the state.",
    }
    validated, schema_findings = multi_turn_mode._derive_candidate_outline_task(
        candidate_task=label_only,
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert schema_findings == []
    findings, _grounded_action = multi_turn_mode._validate_single_outline_task(
        planner=planner,
        task=validated,
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert findings[0]["constraint_code"] == "label_only_state_change"

    clear = {
        **label_only,
        "outline_id": "clear",
        "event_name": "evt_clear",
        "expected_end_state": {
            "resource_state": "clear_state",
            "resource_location": "home",
        },
    }
    validated_clear, schema_findings = multi_turn_mode._derive_candidate_outline_task(
        candidate_task=clear,
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert schema_findings == []
    findings, _grounded_action = multi_turn_mode._validate_single_outline_task(
        planner=planner,
        task=validated_clear,
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert findings == []
    multi_turn_mode._apply_task_effects_to_symbolic_state(validated_clear, session_state)

    return_to_prior = {
        **label_only,
        "outline_id": "return_to_prior",
        "event_name": "evt_return",
        "expected_start_state": {
            "resource_state": "clear_state",
            "resource_location": "home",
        },
        "expected_end_state": {
            "resource_state": "returned_state",
            "resource_location": "assembly_board-v1",
        },
    }
    validated_return, schema_findings = multi_turn_mode._derive_candidate_outline_task(
        candidate_task=return_to_prior,
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert schema_findings == []
    findings, _grounded_action = multi_turn_mode._validate_single_outline_task(
        planner=planner,
        task=validated_return,
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert findings == []
    assert "semantic_state_history" not in session_state


def test_novel_states_do_not_allow_invented_resources_parts_or_locations() -> None:
    _fixture, _product_agent, planner, prepared_recovery_request = asyncio.run(
        _prepare_recovery_dryrun_harness(scripted_responses=_fixture_outline_responses())
    )
    session_state = deepcopy(prepared_recovery_request["multi_turn_session_seed"])

    unknown_resource = {
        "outline_id": "unknown_resource",
        "event_name": "evt_u1",
        "resource_jid": "new_robot@localhost",
        "expected_start_state": {"resource_state": "failed"},
        "expected_end_state": {
            "resource_state": "new_state",
            "resource_location": "home",
        },
        "rationale": "Invalid resource binding.",
    }
    _validated, findings = multi_turn_mode._derive_candidate_outline_task(
        candidate_task=unknown_resource,
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert "unknown resource" in findings[0]["reason"]

    unknown_part = {
        "outline_id": "unknown_part",
        "event_name": "evt_u2",
        "resource_jid": "ur5e@localhost",
        "part_name": "NEW_PART",
        "expected_start_state": {
            "resource_state": "picked",
            "held_part": "MCP",
            "part_state": "in_gripper",
        },
        "expected_end_state": {
            "resource_state": "new_state",
            "held_part": None,
            "part_state": "new_part_state",
            "part_location": "prusa-mk4-2",
        },
        "rationale": "Invalid part binding.",
    }
    _validated, findings = multi_turn_mode._derive_candidate_outline_task(
        candidate_task=unknown_part,
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert "unknown part" in findings[0]["reason"]

    unknown_location = {
        "outline_id": "unknown_location",
        "event_name": "evt_u3",
        "resource_jid": "ur5e@localhost",
        "part_name": "MCP",
        "expected_start_state": {
            "resource_state": "picked",
            "held_part": "MCP",
            "part_state": "in_gripper",
            "part_location": "ur5e@localhost_gripper",
        },
        "expected_end_state": {
            "resource_state": "new_state",
            "held_part": None,
            "part_state": "new_part_state",
            "part_location": "invented_location",
        },
        "rationale": "Invalid location binding.",
    }
    validated, findings = multi_turn_mode._derive_candidate_outline_task(
        candidate_task=unknown_location,
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert findings == []
    validation_findings, _grounded_action = multi_turn_mode._validate_single_outline_task(
        planner=planner,
        task=dict(validated or {}),
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert validation_findings[0]["constraint_code"] == "unknown_location_token"


def test_resource_only_held_part_is_allowed_but_part_fields_require_part_name() -> None:
    _fixture, _product_agent, _planner, prepared_recovery_request = asyncio.run(
        _prepare_recovery_dryrun_harness(scripted_responses=_fixture_outline_responses())
    )
    session_state = deepcopy(prepared_recovery_request["multi_turn_session_seed"])

    resource_only = {
        "outline_id": "resource_only_clear_xarm6",
        "event_name": "evt_resource_only_clear",
        "resource_jid": "xarm6@localhost",
        "expected_start_state": {
            "resource_state": "failed",
            "held_part": None,
            "resource_location": "assembly_board-v1",
        },
        "expected_end_state": {
            "resource_state": "xarm6_clear_state",
            "held_part": None,
            "resource_location": "home",
        },
        "rationale": "Clear the occupied destination without changing any part state.",
    }
    validated, findings = multi_turn_mode._derive_candidate_outline_task(
        candidate_task=resource_only,
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert findings == []
    assert validated is not None

    ambiguous_named_pose = deepcopy(resource_only)
    ambiguous_named_pose["outline_id"] = "resource_only_ambiguous_home"
    ambiguous_named_pose["expected_end_state"] = {
        "resource_state": "home",
        "held_part": None,
        "resource_location": None,
    }
    _validated, findings = multi_turn_mode._derive_candidate_outline_task(
        candidate_task=ambiguous_named_pose,
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert findings
    assert findings[0]["constraint_code"] == "candidate_schema_violation"
    assert "resource_location" in str(findings[0].get("reason") or "")
    assert "home" in str(findings[0].get("reason") or "")

    missing_part_name = {
        "outline_id": "missing_part_name",
        "event_name": "evt_missing_part_name",
        "resource_jid": "ur5e@localhost",
        "expected_start_state": {
            "resource_state": "picked",
            "held_part": "MCP",
            "part_state": "in_gripper",
            "part_location": "ur5e@localhost_gripper",
        },
        "expected_end_state": {
            "resource_state": "mcp_buffer_clear",
            "held_part": None,
            "part_state": "mcp_waiting_recovery",
            "part_location": "prusa-mk4-2",
        },
        "rationale": "Invalid because part fields require part_name.",
    }
    _validated, findings = multi_turn_mode._derive_candidate_outline_task(
        candidate_task=missing_part_name,
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert findings
    reason = str(findings[0].get("reason") or "")
    assert "part_name" in reason
    assert "part_state" in reason
    assert "part_location" in reason
    assert "held_part" not in reason

    inconsistent_held_part = {
        "outline_id": "inconsistent_held_part",
        "event_name": "evt_inconsistent_held_part",
        "resource_jid": "ur5e@localhost",
        "part_name": "LG",
        "expected_start_state": {
            "resource_state": "idle",
            "held_part": None,
            "part_state": "misplaced",
        },
        "expected_end_state": {
            "resource_state": "invalid_hold",
            "held_part": "MCP",
            "part_state": "lg_under_recovery_control",
        },
        "rationale": "Invalid because held_part contradicts part_name.",
    }
    _validated, findings = multi_turn_mode._derive_candidate_outline_task(
        candidate_task=inconsistent_held_part,
        session_state=session_state,
        prepared_recovery_request=prepared_recovery_request,
    )
    assert findings
    assert "contradicts part_name slot" in str(findings[0].get("reason") or "")


def test_grounding_schema_contains_only_structural_decision_fields() -> None:
    schema = multi_turn_prompts.multi_turn_phase_response_schema("grounding")["schema"]

    assert schema["required"] == ["thought", "decision", "observe_requests"]
    assert set(schema["properties"]) == {"thought", "decision", "observe_requests"}
    assert schema["additionalProperties"] is False
    request_schema = schema["properties"]["observe_requests"]["items"]
    assert set(request_schema["properties"]) == {"fact_type", "entity", "reason"}
    assert request_schema["required"] == ["fact_type", "entity"]


def test_production_structured_request_records_actual_system_message() -> None:
    calls: list[dict[str, Any]] = []

    def _create(**kwargs: Any) -> Any:
        calls.append(deepcopy(kwargs))
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "thought": "enough context",
                                "decision": "grounded",
                                "observe_requests": [],
                            }
                        ),
                        tool_calls=None,
                    )
                )
            ]
        )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create))
    )

    async def _direct_to_thread(
        func: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Any:
        return func(*args, **kwargs)

    agent = object.__new__(llm_agent_module.LlmAgent)
    agent.model = "test-model"
    agent.reasoning_effort = "medium"
    agent.instructions = "generic product system instructions"
    response_format = multi_turn_prompts.multi_turn_phase_response_schema("grounding")

    with (
        patch.object(llm_agent_module, "_client", fake_client),
        patch.object(llm_agent_module.asyncio, "to_thread", new=_direct_to_thread),
    ):
        parsed = asyncio.run(
            llm_agent_module.LlmAgent.ask_llm_structured(
                agent,
                "grounding user prompt",
                response_format=response_format,
            )
        )

    assert parsed["decision"] == "grounded"
    assert calls
    recorded_request = agent._last_structured_request
    assert recorded_request["model"] == "test-model"
    assert recorded_request["messages"] == [
        {"role": "system", "content": "generic product system instructions"},
        {"role": "user", "content": "grounding user prompt"},
    ]
    assert calls[0]["messages"] == recorded_request["messages"]
    assert calls[0]["response_format"] == recorded_request["response_format"]


def test_candidate_validation_feedback_is_rendered_once() -> None:
    prompt = _render_non_case3_candidate_prompt(
        candidate_rejection_feedback=[
            {
                "candidate_index": 0,
                "task": {
                    "outline_id": "candidate_1",
                    "resource_jid": "xarm6@localhost",
                },
                "validation_findings": [
                    {
                        "stage": "syntax_and_grounding_validation",
                        "constraint_code": "candidate_schema_violation",
                        "reason": "candidate contains an unsupported field",
                    }
                ],
            }
        ]
    )

    assert prompt.count("candidate_schema_violation") == 1
    assert prompt.count("Validation Feedback") == 1
    assert "Disabled And Blocked Candidate Events" not in prompt


def test_incomplete_release_feedback_preserves_the_exact_current_finding() -> None:
    reason = "Task releases 'LCP' without specifying a concrete grounded destination."
    prompt = _render_non_case3_candidate_prompt(
        candidate_rejection_feedback=[
            {
                "candidate_index": 0,
                "task": {
                    "outline_id": "candidate_release",
                    "event_name": "authored_release",
                    "resource_jid": "xarm6@localhost",
                    "part_name": "LCP",
                },
                "validation_findings": [
                    {
                        "validation_category": "transition_feasibility",
                        "constraint_code": "missing_release_destination",
                        "reason": reason,
                        "resource_jid": "xarm6@localhost",
                        "part_name": "LCP",
                    }
                ],
            }
        ]
    )

    assert prompt.count("missing_release_destination") == 1
    assert prompt.count(reason) == 1
    assert "no_progressing_candidate" not in prompt


def test_outline_request_result_and_stack_artifacts_are_complete(tmp_path: Path) -> None:
    transition_trace = [_candidate_event("better_score", outline_id="RECOVERY_SEQ1")]
    candidate_evaluations = [
        {
            "candidate_index": candidate_index,
            "valid": True,
            "remaining_blocked_issues": 0,
            "validated_task": _candidate_event(event_name),
            "validation_findings": [],
        }
        for candidate_index, event_name in enumerate(
            ("low_score", "better_score", "third_choice")
        )
    ]
    llm_raw_response = {
        "thought": "select the second candidate",
        "selected_candidate_index": 1,
        "candidate_events": [
            _candidate_event("low_score"),
            _candidate_event("better_score"),
            _candidate_event("third_choice"),
        ],
    }
    enriched_response = {
        **deepcopy(llm_raw_response),
        "candidate_evaluations": deepcopy(candidate_evaluations),
        "selected_transition": deepcopy(transition_trace[0]),
        "selected_candidate_index": 1,
        "transition_trace": deepcopy(transition_trace),
    }
    turn = {
        "turn_index": 3,
        "phase": "outline",
        "decision": "need_next_task",
        "prompt_text": "candidate prompt",
        "llm_request": {
            "model": "test-model",
            "messages": [{"role": "user", "content": "candidate prompt"}],
            "response_format": {"type": "json_schema", "json_schema": {}},
            "response_source": "mocked_scripted_fixture",
            "request_sent": False,
        },
        "llm_raw_response": deepcopy(llm_raw_response),
        "raw_response": deepcopy(enriched_response),
        "candidate_evaluations": deepcopy(candidate_evaluations),
        "selected_candidate_index": 1,
        "selected_transition": deepcopy(transition_trace[0]),
        "transition_trace": deepcopy(transition_trace),
    }
    artifact_paths = write_recovery_artifacts(
        {
            "reasoning_mode": "multi_turn",
            "multi_turn_current_turn": deepcopy(turn),
            "multi_turn_llm_raw_response": deepcopy(llm_raw_response),
            "recovery_debug": {
                "multi_turn_session": {
                    "session_id": "artifact_test",
                    "turn_index": 3,
                    "turns": [deepcopy(turn)],
                }
            },
        },
        phase_label="multi_turn",
        debug_dir=tmp_path,
    )

    request_artifact = Path(artifact_paths["request_artifact_path"])
    result_artifact = Path(artifact_paths["outline_result_artifact_path"])
    stack_artifact = Path(artifact_paths["outline_stack_artifact_path"])
    assert request_artifact.name.startswith("multi_turn_turn03_outline_request_")
    assert result_artifact.name.startswith("multi_turn_turn03_outline_result_")
    assert stack_artifact.name.startswith("multi_turn_turn03_outline_stack_")
    assert artifact_paths["prompt_artifact_path"] == str(request_artifact)
    assert artifact_paths["response_artifact_path"] == str(result_artifact)
    assert "llm_response_artifact_path" not in artifact_paths
    assert "turn_index_artifact_path" not in artifact_paths
    assert {path.name for path in (tmp_path / "recovery_outline").iterdir()} == {
        request_artifact.name,
        result_artifact.name,
        stack_artifact.name,
    }

    request_text = request_artifact.read_text(encoding="utf-8")
    assert "Model: test-model" in request_text
    assert "role=user" in request_text
    assert "candidate prompt" in request_text

    result_payload = json.loads(result_artifact.read_text(encoding="utf-8"))
    assert result_payload["llm_response"] == llm_raw_response
    assert len(result_payload["candidate_evaluation_summary"]) == 3
    assert result_payload["selected_transition"] == transition_trace[0]
    assert result_payload["transition_trace"] == transition_trace
    assert result_payload["turn_index"] == 3
    assert result_payload["phase"] == "outline"
    assert result_payload["decision"] == "need_next_task"
    assert result_payload["selected_candidate_index"] == 1
    assert result_payload["selected_transition_outline_id"] == "RECOVERY_SEQ1"
    assert result_payload["accepted_trace_length"] == 1
    assert result_payload["remaining_blocked_issue_count"] == 0
    assert result_payload["artifact_paths"] == {
        "request_artifact_path": str(request_artifact),
        "outline_result_artifact_path": str(result_artifact),
        "outline_stack_artifact_path": str(stack_artifact),
        "prompt_artifact_path": str(request_artifact),
        "response_artifact_path": str(result_artifact),
    }
    stack_payload = json.loads(stack_artifact.read_text(encoding="utf-8"))
    assert stack_payload == transition_trace


def test_rejected_outline_stack_preserves_only_the_accepted_trace(
    tmp_path: Path,
) -> None:
    transition_trace = [_candidate_event("accepted_before", outline_id="RECOVERY_SEQ1")]
    turn = {
        "turn_index": 4,
        "phase": "outline",
        "decision": "need_revision",
        "prompt_text": "candidate prompt",
        "llm_raw_response": {
            "thought": "invalid selected candidate",
            "selected_candidate_index": 0,
            "candidate_events": [_candidate_event("rejected_candidate")],
        },
        "raw_response": {
            "decision": "need_revision",
            "selected_candidate_index": 0,
            "transition_trace": deepcopy(transition_trace),
            "transition_validation": {"status": "rejected"},
        },
    }
    artifact_paths = write_recovery_artifacts(
        {
            "reasoning_mode": "multi_turn",
            "multi_turn_current_turn": deepcopy(turn),
            "recovery_debug": {
                "multi_turn_session": {
                    "session_id": "rejected_stack_test",
                    "turn_index": 4,
                    "turns": [deepcopy(turn)],
                }
            },
        },
        phase_label="multi_turn",
        debug_dir=tmp_path,
    )

    stack_payload = json.loads(
        Path(artifact_paths["outline_stack_artifact_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert stack_payload == transition_trace


def test_case3_actual_outline_validation_is_structural() -> None:
    trace = [
        {
            "outline_id": "A",
            "event_name": "event_a",
            "resource_jid": "resource@localhost",
            "expected_start_state": {},
            "expected_end_state": {},
        }
    ]
    _validate_outline_trace(trace, label="transition_trace")


def test_case3_experiment_settings_are_loaded_from_file() -> None:
    settings = _load_recovery_outline_experiment_settings()

    assert settings["recovery_selection_mode"] == "neurosymbolic"
    assert settings["action_horizon"] == 1
    assert settings["candidate_count"] == "auto"
    assert settings["candidate_proposal_budget"] == 5


def test_case3_configured_neurosymbolic_uses_adaptive_symbolic_selection() -> None:
    session_state = _candidate_session(
        recovery_selection_mode="neurosymbolic",
        action_horizon="1",
    )
    schema = multi_turn_mode._get_response_schema(
        "outline",
        {
            **session_state,
            "outline_mode": "incremental_candidates_validated",
            "candidate_bound": 5,
        },
    )

    assert session_state["recovery_selection_mode"] == "neurosymbolic"
    assert "selected_candidate_index" not in schema["schema"]["properties"]
    candidate_schema = schema["schema"]["properties"]["candidate_events"]
    assert candidate_schema["minItems"] == 1
    assert candidate_schema["maxItems"] == 5


def test_case3_pure_llm_keeps_llm_selected_one_step_candidate() -> None:
    session_state = _candidate_session(
        recovery_selection_mode="pure_llm",
        action_horizon="1",
    )
    _decision, turn_entry = asyncio.run(
        _run_mocked_candidate_handler(
            session_state=session_state,
            parsed_response={
                "thought": "select my preferred candidate",
                "selected_candidate_index": 1,
                "candidate_events": [
                    _candidate_event("low_score"),
                    _candidate_event("better_score"),
                    _candidate_event("third_choice"),
                ],
            },
        )
    )

    assert turn_entry["selected_by"] == "pure_llm"
    assert turn_entry["selected_candidate_index"] == 1
    assert [row["valid"] for row in turn_entry["candidate_evaluations"]] == [
        True,
        True,
        True,
    ]
    assert session_state["accepted_outline_prefix"][0]["event_name"] == "better_score"


def test_case3_invalid_llm_selected_candidate_is_not_substituted() -> None:
    session_state = _candidate_session(
        recovery_selection_mode="pure_llm",
        action_horizon="1",
    )
    decision, turn_entry = asyncio.run(
        _run_mocked_candidate_handler(
            session_state=session_state,
            invalid_candidate_indexes={0},
            parsed_response={
                "thought": "select the first candidate",
                "selected_candidate_index": 0,
                "candidate_events": [
                    _candidate_event("invalid_selected"),
                    _candidate_event("valid_alternative"),
                    _candidate_event("valid_third"),
                ],
            },
        )
    )

    assert decision == "need_revision"
    assert session_state["accepted_outline_prefix"] == []
    assert turn_entry["transition_validation"]["selected_candidate_index"] == 0
    assert turn_entry["candidate_evaluations"][0]["valid"] is False
    assert turn_entry["candidate_evaluations"][1]["valid"] is True
    assert [
        row["candidate_index"]
        for row in turn_entry["candidate_rejection_feedback"]
    ] == [0]


def test_case3_missing_selected_candidate_index_is_rejected() -> None:
    session_state = _candidate_session(
        recovery_selection_mode="pure_llm",
        action_horizon="1",
    )
    decision, turn_entry = asyncio.run(
        _run_mocked_candidate_handler(
            session_state=session_state,
            parsed_response={
                "thought": "forgot to select",
                "candidate_events": [
                    _candidate_event("low_score"),
                    _candidate_event("better_score"),
                    _candidate_event("third_choice"),
                ],
            },
        )
    )

    assert decision == "need_revision"
    assert turn_entry["transition_validation"]["status"] == "rejected"
    assert "selected_candidate_index" in turn_entry["validation_findings"][0]["reason"]


def test_case3_out_of_range_selected_candidate_index_is_rejected() -> None:
    session_state = _candidate_session(
        recovery_selection_mode="pure_llm",
        action_horizon="1",
    )
    decision, turn_entry = asyncio.run(
        _run_mocked_candidate_handler(
            session_state=session_state,
            parsed_response={
                "thought": "bad selection",
                "selected_candidate_index": 3,
                "candidate_events": [
                    _candidate_event("low_score"),
                    _candidate_event("better_score"),
                    _candidate_event("third_choice"),
                ],
            },
        )
    )

    assert decision == "need_revision"
    assert turn_entry["transition_validation"]["status"] == "rejected"
    assert "selected_candidate_index" in turn_entry["validation_findings"][0]["reason"]


def test_case3_pure_llm_k_horizon_commits_selected_sequence() -> None:
    session_state = _candidate_session(
        recovery_selection_mode="pure_llm",
        action_horizon="k",
    )
    _decision, turn_entry = asyncio.run(
        _run_mocked_candidate_handler(
            session_state=session_state,
            parsed_response={
                "thought": "propose short traces",
                "selected_candidate_index": 1,
                "candidate_traces": [
                    {"events": [_candidate_event("low_score")]},
                    {
                        "events": [
                            _candidate_event("step_a"),
                            _candidate_event("step_b"),
                        ]
                    },
                ],
            },
        )
    )

    assert turn_entry["selected_by"] == "pure_llm"
    assert turn_entry["selected_candidate_index"] == 1
    assert len(turn_entry["selected_transition_sequence"]) == 2
    assert [row["event_name"] for row in session_state["accepted_outline_prefix"]] == [
        "step_a",
        "step_b",
    ]


def test_case3_pure_llm_full_horizon_requires_complete_trace() -> None:
    session_state = _candidate_session(
        recovery_selection_mode="pure_llm",
        action_horizon="full",
    )
    decision, turn_entry = asyncio.run(
        _run_mocked_candidate_handler(
            session_state=session_state,
            parsed_response={
                "thought": "propose full traces",
                "selected_candidate_index": 1,
                "candidate_traces": [
                    {"events": [_candidate_event("better_score")]},
                    {
                        "events": [
                            _candidate_event("step_a"),
                            _candidate_event("step_b"),
                        ]
                    },
                ],
            },
        )
    )

    assert decision == "outline_ready"
    assert turn_entry["selected_candidate_index"] == 1
    assert len(session_state["accepted_outline_prefix"]) == 2


def test_case3_one_step_candidate_count_rejects_fewer_than_three() -> None:
    session_state = _candidate_session(
        recovery_selection_mode="pure_llm",
        action_horizon="1",
        candidate_count="auto",
    )
    decision, turn_entry = asyncio.run(
        _run_mocked_candidate_handler(
            session_state=session_state,
            parsed_response={
                "thought": "too few alternatives",
                "candidate_events": [
                    _candidate_event("low_score"),
                    _candidate_event("better_score"),
                ],
            },
        )
    )

    assert decision == "need_revision"
    assert "exactly 3 candidates" in turn_entry["error"]


def test_case3_one_step_candidate_count_rejects_more_than_three() -> None:
    session_state = _candidate_session(
        recovery_selection_mode="pure_llm",
        action_horizon="1",
        candidate_count="auto",
    )
    decision, turn_entry = asyncio.run(
        _run_mocked_candidate_handler(
            session_state=session_state,
            parsed_response={
                "thought": "too many alternatives",
                "selected_candidate_index": 0,
                "candidate_events": [
                    _candidate_event("first_choice"),
                    _candidate_event("second_choice"),
                    _candidate_event("third_choice"),
                    _candidate_event("fourth_choice"),
                ],
            },
        )
    )

    assert decision == "need_revision"
    assert "exactly 3 candidates" in turn_entry["error"]


if __name__ == "__main__":
    raise SystemExit(main())
