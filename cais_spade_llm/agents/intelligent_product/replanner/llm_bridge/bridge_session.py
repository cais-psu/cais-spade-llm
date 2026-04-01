"""Bridge session — prepare-trace slice."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha1
from math import sqrt
from pathlib import Path
from typing import Any
from uuid import uuid4

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.modes import (
    build_multi_turn_session_seed,
    build_single_shot_prompt_artifacts,
    execute_multi_turn_bridge,
    execute_single_shot_bridge,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_resource_normalization import (
    bridge_resource_capabilities,
    normalize_bridge_resource,
    resolve_bridge_resource_type,
)
from cais_spade_llm.resources.resource_profile import (
    get_resource_profile,
    resource_snapshot_carried_entity,
    resource_snapshot_carried_entity_location,
    resource_snapshot_fields_map,
)


class BridgeSessionMixin:
    @staticmethod
    def _normalize_bridge_reasoning_mode(value: Any) -> str:
        mode = str(value or "single_shot").strip().lower()
        if mode not in {"single_shot", "multi_turn"}:
            return "single_shot"
        return mode

    def _resolve_bridge_reasoning_mode(self) -> str:
        product_agent = getattr(self, "product_agent", None)
        raw_mode = getattr(product_agent, "_bridge_reasoning_mode", None)
        if raw_mode not in (None, ""):
            return self._normalize_bridge_reasoning_mode(raw_mode)
        bundle = dict(getattr(product_agent, "precomputed_bundle", {}) or {})
        precomputed_policy = (
            bundle.get("replan_policy", {})
            if isinstance(bundle.get("replan_policy"), dict)
            else {}
        )
        return self._normalize_bridge_reasoning_mode(
            precomputed_policy.get("bridge_reasoning_mode", "single_shot")
        )

    def _bridge_artifact_path(self, artifact_key: str) -> Path | None:
        product_agent = getattr(self, "product_agent", None)
        candidates: list[Path] = []
        if product_agent is not None:
            bundle = dict(getattr(product_agent, "precomputed_bundle", {}) or {})
            artifacts = dict(bundle.get("artifacts") or {})
            raw_path = artifacts.get(artifact_key)
            if raw_path:
                candidates.append(Path(str(raw_path)))
            if artifact_key == "requirements_json":
                structured_path = getattr(product_agent, "structured_requirements_path", None)
                if structured_path:
                    candidates.append(Path(str(structured_path)))

        for candidate in candidates:
            try:
                path = candidate if candidate.is_absolute() else (Path.cwd() / candidate)
                resolved = path.resolve()
            except Exception:
                resolved = candidate
            if resolved.exists():
                return resolved
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _load_json_artifact(path: Path | None) -> dict[str, Any] | list[Any]:
        if path is None or not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _bridge_requirement_inventory(self) -> dict[str, Any]:
        payload = self._load_json_artifact(self._bridge_artifact_path("requirements_json"))
        if isinstance(payload, dict):
            raw_nodes = payload.get("nodes") or []
        elif isinstance(payload, list):
            raw_nodes = payload
        else:
            raw_nodes = []

        requirement_nodes: list[dict[str, Any]] = []
        requirement_ids_seen: set[str] = set()
        for raw_node in raw_nodes:
            if not isinstance(raw_node, dict):
                continue
            req_id = str(raw_node.get("id") or "").strip()
            if not req_id or req_id in requirement_ids_seen:
                continue
            requirement_ids_seen.add(req_id)
            requirement_nodes.append(
                {
                    "requirement_id": req_id,
                    "summary": str(
                        raw_node.get("description")
                        or raw_node.get("raw_text")
                        or ""
                    ).strip(),
                    "raw_text": str(raw_node.get("raw_text") or "").strip(),
                    "phase": deepcopy(raw_node.get("phase")),
                    "process_type": deepcopy(raw_node.get("process_type")),
                    "product": deepcopy(raw_node.get("product")),
                    "context": deepcopy(raw_node.get("context") or {}),
                }
            )

        task_nodes = [
            node
            for node in getattr(self, "nodes", [])
            if isinstance(node, dict) and node.get("type") == "task"
        ]
        requirement_task_index: dict[str, list[dict[str, Any]]] = {}
        task_requirement_map: dict[str, str] = {}

        def _task_sort_key(node: dict[str, Any]) -> tuple[int, str]:
            raw_seq = node.get("sequence_index")
            try:
                sequence_index = int(raw_seq)
            except (TypeError, ValueError):
                sequence_index = 0
            return sequence_index, str(node.get("id") or "").strip()

        for task in sorted(task_nodes, key=_task_sort_key):
            req_id = str(task.get("requirement_id") or "").strip()
            task_id = str(task.get("id") or "").strip()
            if not req_id or not task_id:
                continue
            task_requirement_map[task_id] = req_id
            requirement_task_index.setdefault(req_id, []).append(
                {
                    "task_id": task_id,
                    "function_name": str(task.get("function_name") or "").strip(),
                    "resource_jid": str(task.get("resource_jid") or "").strip(),
                    "part_name": str((task.get("params") or {}).get("part_name") or "").strip(),
                    "status": deepcopy(task.get("status")),
                    "sequence_index": deepcopy(task.get("sequence_index")),
                }
            )

        requirements_status: dict[str, dict[str, Any]] = {
            str(node.get("requirement_id") or ""): {
                "goal": str(node.get("summary") or node.get("raw_text") or "").strip(),
                "status": "unknown",
                "completion": 0,
            }
            for node in requirement_nodes
            if str(node.get("requirement_id") or "").strip()
        }

        req_ids_from_tasks = set(requirement_task_index)
        for req_id in req_ids_from_tasks:
            requirements_status.setdefault(
                req_id,
                {"goal": "", "status": "unknown", "completion": 0},
            )

        for req_id in req_ids_from_tasks:
            req_tasks = requirement_task_index.get(req_id) or []
            total_tasks = len(req_tasks)
            completed_tasks = sum(1 for task in req_tasks if task.get("status") == "completed")
            has_failed = any(
                isinstance(task.get("status"), str) and str(task.get("status")).startswith("failed")
                for task in req_tasks
            )
            all_completed = bool(req_tasks) and all(task.get("status") == "completed" for task in req_tasks)

            if total_tasks > 0:
                requirements_status[req_id]["completion"] = int((completed_tasks / total_tasks) * 100)

            if has_failed:
                requirements_status[req_id]["status"] = "failed"
            elif all_completed:
                requirements_status[req_id]["status"] = "completed"
            elif completed_tasks > 0:
                requirements_status[req_id]["status"] = "in_progress"
            else:
                requirements_status[req_id]["status"] = "pending"

        return {
            "requirement_nodes": requirement_nodes,
            "requirements_status": requirements_status,
            "requirement_task_index": requirement_task_index,
            "task_requirement_map": task_requirement_map,
        }

    def _bridge_loaded_safety_rules(self) -> list[dict[str, Any]]:
        payload = self._load_json_artifact(self._bridge_artifact_path("safety_logic_json"))
        if isinstance(payload, dict):
            raw_rules = payload.get("rules") or []
        elif isinstance(payload, list):
            raw_rules = payload
        else:
            raw_rules = []
        return [deepcopy(rule) for rule in raw_rules if isinstance(rule, dict)]

    @staticmethod
    def _brief_safety_rule(rule: dict[str, Any]) -> dict[str, Any]:
        return {
            "rule_id": str(rule.get("id") or rule.get("rule_id") or "").strip(),
            "constraint_type": str(rule.get("constraint_type") or "").strip(),
            "summary": str(
                rule.get("generated_interpretation")
                or rule.get("text")
                or rule.get("raw_text")
                or ""
            ).strip(),
        }

    @staticmethod
    def _normalize_resource_token(value: Any) -> str:
        token = str(value or "").strip().lower()
        if "@" in token:
            token = token.split("@", 1)[0]
        return token

    def _filter_relevant_loaded_safety_rules(
        self,
        *,
        raw_rules: list[dict[str, Any]],
        active_rule_ids: list[str],
        relevant_parts: list[str],
        relevant_resource_jids: list[str],
        relevant_locations: list[str],
    ) -> list[dict[str, Any]]:
        if not raw_rules:
            return []

        active_rule_id_set = {
            str(rule_id or "").strip()
            for rule_id in active_rule_ids
            if str(rule_id or "").strip()
        }
        resource_tokens = {
            self._normalize_resource_token(resource_jid)
            for resource_jid in relevant_resource_jids
            if self._normalize_resource_token(resource_jid)
        }
        location_tokens = {
            str(location or "").strip().lower()
            for location in relevant_locations
            if str(location or "").strip()
        }
        part_tokens = {
            str(part_name or "").strip().lower()
            for part_name in relevant_parts
            if str(part_name or "").strip()
        }

        selected: list[dict[str, Any]] = []
        seen_signatures: set[str] = set()
        for rule in raw_rules:
            if not isinstance(rule, dict):
                continue
            rule_id = str(rule.get("id") or rule.get("rule_id") or "").strip()
            rule_resource_tokens = {
                self._normalize_resource_token(item)
                for item in (rule.get("resources") or [])
                if self._normalize_resource_token(item)
            }
            context = dict(rule.get("context") or {})
            destination = str(context.get("destination") or "").strip().lower()
            search_blob = " ".join(
                [
                    str(rule.get("generated_interpretation") or ""),
                    str(rule.get("text") or ""),
                    str(rule.get("raw_text") or ""),
                    json.dumps(context, sort_keys=True, default=str),
                ]
            ).lower()

            matches = False
            if rule_id and rule_id in active_rule_id_set:
                matches = True
            elif rule_resource_tokens and resource_tokens.intersection(rule_resource_tokens):
                matches = True
            elif destination and destination in location_tokens:
                matches = True
            elif any(part_token in search_blob for part_token in part_tokens):
                matches = True

            if not matches:
                continue
            signature = self._bridge_safety_rule_signature(rule)
            if not signature or signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            selected.append(self._brief_safety_rule(rule))

        if selected:
            return selected

        fallback: list[dict[str, Any]] = []
        seen_fallback: set[str] = set()
        for rule in raw_rules:
            if not isinstance(rule, dict):
                continue
            signature = self._bridge_safety_rule_signature(rule)
            if not signature or signature in seen_fallback:
                continue
            seen_fallback.add(signature)
            fallback.append(self._brief_safety_rule(rule))
        return fallback

    def _build_relevant_assembly_requirements(
        self,
        *,
        prepared_bridge_request: dict[str, Any],
        bridge_resources: dict[str, Any],
        failure_anchor: dict[str, Any],
    ) -> list[dict[str, Any]]:
        requirement_nodes = [
            dict(node)
            for node in (prepared_bridge_request.get("requirement_nodes") or [])
            if isinstance(node, dict)
        ]
        requirements_status = dict(prepared_bridge_request.get("requirements_status") or {})
        requirement_task_index = dict(prepared_bridge_request.get("requirement_task_index") or {})
        task_requirement_map = dict(prepared_bridge_request.get("task_requirement_map") or {})

        requirement_nodes_by_id = {
            str(node.get("requirement_id") or "").strip(): node
            for node in requirement_nodes
            if str(node.get("requirement_id") or "").strip()
        }
        relevant_requirement_ids: list[str] = []
        seen_requirement_ids: set[str] = set()

        def _add_requirement_id(raw_req_id: Any) -> None:
            req_id = str(raw_req_id or "").strip()
            if not req_id or req_id in seen_requirement_ids:
                return
            seen_requirement_ids.add(req_id)
            relevant_requirement_ids.append(req_id)

        blocked_task_id = str(failure_anchor.get("failed_task_id") or "").strip()
        if blocked_task_id:
            _add_requirement_id(task_requirement_map.get(blocked_task_id))

        relevant_parts = {
            str(part_name or "").strip()
            for part_name in (prepared_bridge_request.get("P_id") or [])
            if str(part_name or "").strip()
        }
        relevant_parts.update(
            str(part.get("entity_id") or "").strip()
            for part in (failure_anchor.get("affected_parts") or [])
            if isinstance(part, dict) and str(part.get("entity_id") or "").strip()
        )

        pending_task_ids = {
            str(task.get("id") or "").strip()
            for raw_entry in (bridge_resources or {}).values()
            if isinstance(raw_entry, dict)
            for task in (raw_entry.get("pending_tasks") or [])
            if isinstance(task, dict) and str(task.get("id") or "").strip()
        }

        for req_id, node in requirement_nodes_by_id.items():
            product = str(node.get("product") or "").strip()
            if product and product in relevant_parts:
                _add_requirement_id(req_id)

        for req_id, task_entries in requirement_task_index.items():
            if not isinstance(task_entries, list):
                continue
            for task_entry in task_entries:
                if not isinstance(task_entry, dict):
                    continue
                task_id = str(task_entry.get("task_id") or "").strip()
                part_name = str(task_entry.get("part_name") or "").strip()
                if task_id in pending_task_ids or (part_name and part_name in relevant_parts):
                    _add_requirement_id(req_id)
                    break

        relevant_requirements: list[dict[str, Any]] = []
        for req_id in relevant_requirement_ids:
            node = dict(requirement_nodes_by_id.get(req_id) or {})
            status_entry = dict(requirements_status.get(req_id) or {})
            related_tasks = [
                dict(task)
                for task in (requirement_task_index.get(req_id) or [])
                if isinstance(task, dict)
            ]
            resource_jids = sorted(
                {
                    str(task.get("resource_jid") or "").strip()
                    for task in related_tasks
                    if str(task.get("resource_jid") or "").strip()
                }
            )
            related_task_ids = [
                str(task.get("task_id") or "").strip()
                for task in related_tasks
                if str(task.get("task_id") or "").strip()
            ]
            pending_related_task_ids = [
                task_id
                for task_id in related_task_ids
                if task_id in pending_task_ids
            ]
            relevant_requirements.append(
                {
                    "requirement_id": req_id,
                    "summary": str(
                        status_entry.get("goal")
                        or node.get("summary")
                        or node.get("raw_text")
                        or ""
                    ).strip(),
                    "status": str(status_entry.get("status") or "unknown").strip(),
                    "product": deepcopy(node.get("product")),
                    "resource_jid": resource_jids[0] if len(resource_jids) == 1 else None,
                    "pending_task_ids": pending_related_task_ids,
                }
            )

        return relevant_requirements

    @staticmethod
    def _coerce_xyz_pose(value: Any) -> dict[str, float] | None:
        if not isinstance(value, dict):
            return None
        pose: dict[str, float] = {}
        for axis in ("x", "y", "z"):
            raw_value = value.get(axis)
            if raw_value is None:
                return None
            try:
                pose[axis] = float(raw_value)
            except (TypeError, ValueError):
                return None
        return pose

    @staticmethod
    def _pose_distance(left: dict[str, float] | None, right: dict[str, float] | None) -> float | None:
        if left is None or right is None:
            return None
        return sqrt(
            ((left.get("x", 0.0) - right.get("x", 0.0)) ** 2)
            + ((left.get("y", 0.0) - right.get("y", 0.0)) ** 2)
            + ((left.get("z", 0.0) - right.get("z", 0.0)) ** 2)
        )

    def _resolve_part_location(
        self,
        *,
        part_name: str,
        tracker_entry: dict[str, Any],
        prepared_bridge_request: dict[str, Any],
    ) -> tuple[Any, str]:
        del part_name, prepared_bridge_request
        explicit_location = tracker_entry.get("location")
        if explicit_location not in (None, ""):
            return deepcopy(explicit_location), "part_tracker"

        observed_pose = self._coerce_xyz_pose(tracker_entry.get("observed_pose"))
        if observed_pose is None:
            return None, "unavailable"
        return None, "observed_pose_only"

    def _resolve_resource_location(
        self,
        *,
        resource_jid: str,
        snapshot: dict[str, Any],
        profile: Any,
        prepared_bridge_request: dict[str, Any],
    ) -> tuple[Any, str]:
        explicit_location = snapshot.get("current_location")
        if explicit_location not in (None, ""):
            return deepcopy(explicit_location), "bridge_snapshot"

        current_pose_ref = snapshot.get("current_pose_ref")
        if current_pose_ref not in (None, ""):
            return deepcopy(current_pose_ref), "current_pose_ref"

        held_part = str(resource_snapshot_carried_entity(snapshot, profile=profile) or "").strip()
        current_pose = self._coerce_xyz_pose(snapshot.get("current_pose"))

        if held_part:
            part_tracker = dict(prepared_bridge_request.get("part_tracker") or {})
            held_entry = dict(part_tracker.get(held_part) or {})
            origin_location = str(held_entry.get("origin_resource_location") or "").strip()
            current_state = str(snapshot.get("current_state") or "").strip().lower()
            if origin_location and current_state == "picked":
                return origin_location, "held_part_origin"

        if current_pose is not None:
            return None, "current_pose_only"
        return None, "unavailable"

    def _trim_pending_tasks(
        self,
        *,
        bridge_resources: dict[str, dict[str, Any]],
        part_tracker: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        node_by_id = {
            str(node.get("id") or "").strip(): node
            for node in getattr(self, "nodes", [])
            if isinstance(node, dict) and str(node.get("id") or "").strip()
        }
        completed_by_resource: dict[str, int] = {}
        for raw_entry in (part_tracker or {}).values():
            entry = raw_entry if isinstance(raw_entry, dict) else {}
            task_id = str(entry.get("last_successful_task") or "").strip()
            node = node_by_id.get(task_id) or {}
            resource_jid = str(node.get("resource_jid") or "").strip()
            if not resource_jid:
                continue
            try:
                sequence_index = int(node.get("sequence_index"))
            except (TypeError, ValueError):
                continue
            completed_by_resource[resource_jid] = max(
                sequence_index,
                completed_by_resource.get(resource_jid, sequence_index),
            )

        refined: dict[str, dict[str, Any]] = {}
        for resource_jid, raw_entry in (bridge_resources or {}).items():
            entry = deepcopy(raw_entry if isinstance(raw_entry, dict) else {})
            cutoff = completed_by_resource.get(str(resource_jid or "").strip())
            pending_tasks = [
                task for task in (entry.get("pending_tasks") or [])
                if isinstance(task, dict)
            ]
            if cutoff is None:
                refined[str(resource_jid)] = entry
                continue
            filtered_tasks: list[dict[str, Any]] = []
            for task in pending_tasks:
                try:
                    sequence_index = int(task.get("sequence_index"))
                except (TypeError, ValueError):
                    filtered_tasks.append(task)
                    continue
                if sequence_index > cutoff:
                    filtered_tasks.append(task)
            entry["pending_tasks"] = filtered_tasks
            refined[str(resource_jid)] = entry
        return refined

    @staticmethod
    def _brief_condition(
        entry: dict[str, Any],
        *,
        condition_family: str = "",
    ) -> dict[str, Any]:
        return {
            "kind": str(entry.get("kind") or "").strip(),
            "condition_family": (
                str(entry.get("condition_family") or "").strip()
                or str(condition_family or "").strip()
            ),
            "entity_kind": str(entry.get("entity_kind") or "").strip(),
            "entity": str(entry.get("entity") or "").strip(),
            "field": str(entry.get("field") or "").strip(),
            "expected": deepcopy(entry.get("expected")),
            "actual": deepcopy(entry.get("actual")),
            "source_task_id": str(entry.get("source_task_id") or "").strip(),
            "source_task_ids": [
                str(task_id or "").strip()
                for task_id in (entry.get("source_task_ids") or [])
                if str(task_id or "").strip()
            ],
            "source_function_name": str(entry.get("source_function_name") or "").strip(),
            "role": str(entry.get("role") or "").strip(),
        }

    @staticmethod
    def _condition_signature(entry: dict[str, Any]) -> tuple[str, str, str, str, str]:
        return (
            str(entry.get("condition_family") or "").strip(),
            str(entry.get("entity_kind") or "").strip(),
            str(entry.get("entity") or "").strip(),
            str(entry.get("field") or "").strip(),
            json.dumps(entry.get("expected"), sort_keys=True, default=str),
        )

    @staticmethod
    def _condition_id(entry: dict[str, Any]) -> str:
        payload = {
            "kind": str(entry.get("kind") or "").strip(),
            "condition_family": str(entry.get("condition_family") or "").strip(),
            "entity_kind": str(entry.get("entity_kind") or "").strip(),
            "entity": str(entry.get("entity") or "").strip(),
            "field": str(entry.get("field") or "").strip(),
            "expected": deepcopy(entry.get("expected")),
            "actual": deepcopy(entry.get("actual")),
            "source_task_id": str(entry.get("source_task_id") or "").strip(),
            "source_task_ids": [
                str(task_id or "").strip()
                for task_id in (entry.get("source_task_ids") or [])
                if str(task_id or "").strip()
            ],
            "source_function_name": str(entry.get("source_function_name") or "").strip(),
            "blocking_rule_id": str(entry.get("blocking_rule_id") or "").strip(),
        }
        digest = sha1(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:12]
        return f"cond_{digest}"

    @staticmethod
    def _ordered_resource_jids(
        *,
        focused_resource_jid: str,
        observed_resources: list[dict[str, Any]],
        bridge_resources: dict[str, dict[str, Any]],
    ) -> list[str]:
        ordered_resource_jids: list[str] = []
        seen_resource_jids: set[str] = set()

        def _append_resource_jid(raw_value: Any) -> None:
            resource_jid = str(raw_value or "").strip()
            if not resource_jid or resource_jid in seen_resource_jids:
                return
            seen_resource_jids.add(resource_jid)
            ordered_resource_jids.append(resource_jid)

        _append_resource_jid(focused_resource_jid)
        for row in observed_resources:
            _append_resource_jid(row.get("resource_jid"))
        for resource_jid in sorted(bridge_resources):
            _append_resource_jid(resource_jid)
        return ordered_resource_jids

    def _bridge_grounding_context(
        self,
        *,
        focused_resource_jid: str,
        bridge_snapshot: dict[str, Any] | None,
        bridge_resources: dict[str, dict[str, Any]] | None,
        part_tracker: dict[str, Any],
        goal_state: str,
        P_id: list[str],
        obligation_targets: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        def _flat_facet_fields(normalized_snapshot: dict[str, Any]) -> dict[str, Any]:
            flat: dict[str, Any] = {}
            for facet in (normalized_snapshot.get("resource_facets") or {}).values():
                if not isinstance(facet, dict):
                    continue
                for key, value in facet.items():
                    flat[str(key)] = deepcopy(value)
            return flat

        snapshot = dict(bridge_snapshot or {})
        focused_snapshot = normalize_bridge_resource(
            resource_jid=str(focused_resource_jid or "").strip(),
            resource_type=resolve_bridge_resource_type(
                snapshot=snapshot,
                modeled_state={},
                static_capabilities={},
            ),
            snapshot=snapshot,
            modeled_state={},
        )
        context: dict[str, Any] = {
            "resource": {
                "jid": str(focused_resource_jid or "").strip(),
                "resource_core": deepcopy(focused_snapshot.get("resource_core") or {}),
                "resource_facets": deepcopy(focused_snapshot.get("resource_facets") or {}),
                "resource_type": focused_snapshot.get("resource_type"),
                "current_state": focused_snapshot.get("current_state"),
                "current_location": deepcopy(focused_snapshot.get("current_location")),
            },
            "resources": {},
            "parts": {},
            "goal": {
                "goal_state": str(goal_state or "").strip(),
                "pending_parts": [str(part_name) for part_name in (P_id or []) if str(part_name).strip()],
            },
            "focused_resource_jid": str(focused_resource_jid or "").strip(),
            "obligation_targets": deepcopy(obligation_targets or []),
        }
        focused_profile = get_resource_profile(str(focused_snapshot.get("resource_type") or "resource"))
        context["resource"].update(
            resource_snapshot_fields_map(
                focused_snapshot,
                focused_profile.snapshot_fields,
                profile=focused_profile,
            )
        )
        context["resource"].update(_flat_facet_fields(focused_snapshot))

        for resource_jid, raw_entry in (bridge_resources or {}).items():
            if not isinstance(raw_entry, dict):
                continue
            resource_snapshot = dict(raw_entry.get("bridge_snapshot") or raw_entry.get("primitive_snapshot") or {})
            resource_type = resolve_bridge_resource_type(
                snapshot=resource_snapshot,
                modeled_state=dict(raw_entry.get("modeled_state") or {}),
                static_capabilities=dict(raw_entry.get("static_capabilities") or {}),
            )
            normalized_entry = normalize_bridge_resource(
                resource_jid=str(resource_jid),
                resource_type=resource_type,
                snapshot=resource_snapshot,
                modeled_state=dict(raw_entry.get("modeled_state") or {}),
            )
            profile = get_resource_profile(str(normalized_entry.get("resource_type") or "resource"))
            context["resources"][str(resource_jid)] = {
                "jid": str(resource_jid),
                "resource_core": deepcopy(normalized_entry.get("resource_core") or {}),
                "resource_facets": deepcopy(normalized_entry.get("resource_facets") or {}),
                "resource_type": normalized_entry.get("resource_type"),
                "current_state": normalized_entry.get("current_state"),
                "current_location": deepcopy(normalized_entry.get("current_location")),
                "modeled_state": deepcopy(raw_entry.get("modeled_state") or {}),
                "pending_tasks": deepcopy(raw_entry.get("pending_tasks") or []),
                "static_capabilities": deepcopy(raw_entry.get("static_capabilities") or {}),
                "bridge_adapter": deepcopy(raw_entry.get("bridge_adapter") or {}),
            }
            context["resources"][str(resource_jid)].update(
                resource_snapshot_fields_map(
                    normalized_entry,
                    profile.snapshot_fields,
                    profile=profile,
                )
            )
            context["resources"][str(resource_jid)].update(_flat_facet_fields(normalized_entry))

        geometry_lookup = getattr(self.product_agent, "_geometry_for_part", None)
        for part_name, raw_info in (part_tracker or {}).items():
            name = str(part_name or "").strip()
            if not name:
                continue

            info = raw_info if isinstance(raw_info, dict) else {}
            observed_pose = None
            for candidate_key in ("observed_pose", "position", "pose", "location"):
                observed_pose = self._coerce_xyz_pose(info.get(candidate_key))
                if observed_pose is not None:
                    break

            target: dict[str, Any] | None = None
            if callable(geometry_lookup):
                try:
                    geometry = geometry_lookup(name) or {}
                except Exception:
                    geometry = {}
                if isinstance(geometry, dict):
                    slot_xy = geometry.get("slot_xy")
                    board_center = geometry.get("board_center") or {}
                    if isinstance(slot_xy, (list, tuple)) and len(slot_xy) >= 2:
                        try:
                            board_top_z = float(
                                geometry.get("slot_floor_z_m", board_center.get("z", 0.0))
                            )
                            target = {
                                "slot_pose": {
                                    "x": float(board_center.get("x", 0.0)) + float(slot_xy[0]),
                                    "y": float(board_center.get("y", 0.0)) + float(slot_xy[1]),
                                    "z": board_top_z,
                                },
                                "board_top_z": board_top_z,
                            }
                            if geometry.get("part_height_m") is not None:
                                target["part_height"] = float(geometry["part_height_m"])
                            if geometry.get("model_name"):
                                target["model_name"] = str(geometry["model_name"])
                            product_location = str(getattr(self.product_agent, "jid", "") or "").strip()
                            if product_location:
                                target["location"] = product_location.split("@", 1)[0]
                        except (TypeError, ValueError):
                            target = None

            context["parts"][name] = {
                "state": info.get("state"),
                "location": deepcopy(info.get("location")),
                "last_known_location": deepcopy(info.get("last_known_location")),
                "observed_pose": observed_pose,
                "target": target,
            }

        return context

    async def _bridge_primitive_context(
        self,
        *,
        target_jid: str,
        resource_states: dict[str, dict[str, Any]],
        default_resource_state: str,
        part_states: dict[str, Any],
        part_locations: dict[str, Any],
    ) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None, dict[str, dict[str, Any]]]:
        focused_resource = self._resource_by_jid(target_jid)
        if focused_resource is None:
            return None, None, {}

        from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.primitive_semantics import (
            build_execution_primitive_catalog,
            build_synthesis_primitive_catalog,
        )

        focused_primitive_catalog: list[dict[str, Any]] | None = None
        focused_bridge_snapshot: dict[str, Any] | None = None
        bridge_resources: dict[str, dict[str, Any]] = {}

        try:
            for resource in self.resource_agents:
                resource_jid = str(getattr(resource, "jid", "")).strip()
                if not resource_jid:
                    continue
                raw_bridge_snapshot: dict[str, Any] = {}
                if hasattr(resource, "get_bridge_snapshot"):
                    bridge_snapshot_value = await asyncio.to_thread(resource.get_bridge_snapshot)
                    if isinstance(bridge_snapshot_value, dict):
                        raw_bridge_snapshot = deepcopy(bridge_snapshot_value)
                elif hasattr(resource, "_snapshot_state"):
                    snapshot_value = await asyncio.to_thread(resource._snapshot_state)
                    if isinstance(snapshot_value, dict):
                        raw_bridge_snapshot = deepcopy(snapshot_value)

                modeled_state = self._build_resource_search_state(
                    resource_jid=resource_jid,
                    resource_states=resource_states,
                    default_resource_state=default_resource_state,
                    part_states=part_states,
                    part_locations=part_locations,
                )
                static_capabilities = deepcopy(getattr(resource, "static_capabilities", {}) or {})
                resource_type = resolve_bridge_resource_type(
                    resource=resource,
                    snapshot=raw_bridge_snapshot,
                    modeled_state=modeled_state,
                    static_capabilities=static_capabilities,
                )
                bridge_snapshot = normalize_bridge_resource(
                    resource_jid=resource_jid,
                    resource_type=resource_type,
                    snapshot=raw_bridge_snapshot,
                    modeled_state=modeled_state,
                )
                execution_primitive_catalog = build_execution_primitive_catalog(resource) or []
                primitive_catalog = build_synthesis_primitive_catalog(
                    primitive_catalog=execution_primitive_catalog,
                ) or []
                adapter_capabilities = bridge_resource_capabilities(
                    resource_type,
                    primitive_catalog=execution_primitive_catalog,
                )
                bridge_resources[resource_jid] = {
                    "resource_jid": resource_jid,
                    "resource_type": resource_type,
                    "bridge_adapter": adapter_capabilities,
                    "primitive_catalog": primitive_catalog,
                    "execution_primitive_catalog": execution_primitive_catalog,
                    "bridge_snapshot": bridge_snapshot or {},
                    "resource_core": deepcopy(bridge_snapshot.get("resource_core") or {}),
                    "resource_facets": deepcopy(bridge_snapshot.get("resource_facets") or {}),
                    "modeled_state": modeled_state,
                    "pending_tasks": deepcopy(self._pending_resource_tasks(resource_jid)),
                    "static_capabilities": static_capabilities,
                }
                if resource_jid == target_jid:
                    focused_primitive_catalog = (
                        primitive_catalog if adapter_capabilities.get("supports_executable_bridge") else None
                    )
                    focused_bridge_snapshot = bridge_snapshot or None

            return focused_primitive_catalog, focused_bridge_snapshot, bridge_resources
        except Exception:
            self.logger.exception(
                "[Planner] Failed to build whole-system primitive bridge context for %s",
                target_jid,
            )
            return None, None, {}

    def _bridge_effective_part_facts(
        self,
        *,
        fallback_part_tracker: dict[str, Any],
        projected_parts: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
        part_states: dict[str, Any] = {}
        part_locations: dict[str, Any] = {}
        part_entries: dict[str, dict[str, Any]] = {}

        for part_name, raw_info in (fallback_part_tracker or {}).items():
            name = str(part_name or "").strip()
            if not name:
                continue
            info = raw_info if isinstance(raw_info, dict) else {}
            part_states[name] = info.get("state")
            part_locations[name] = info.get("location")
            part_entries[name] = deepcopy(info)

        for part_name, raw_info in (projected_parts or {}).items():
            name = str(part_name or "").strip()
            if not name:
                continue
            info = raw_info if isinstance(raw_info, dict) else {}
            entry = dict(part_entries.get(name) or {})
            entry.update(deepcopy(info))
            part_entries[name] = entry
            if "state" in info:
                part_states[name] = info.get("state")
            if "location" in info:
                part_locations[name] = info.get("location")

        return part_states, part_locations, part_entries

    def _bridge_effective_resource_facts(
        self,
        *,
        bridge_resources: dict[str, dict[str, Any]],
        projected_resource_snapshots: dict[str, dict[str, Any]] | None,
    ) -> dict[str, dict[str, Any]]:
        effective: dict[str, dict[str, Any]] = {}
        for resource_jid, raw_entry in (bridge_resources or {}).items():
            jid = str(resource_jid or "").strip()
            if not jid:
                continue
            entry = raw_entry if isinstance(raw_entry, dict) else {}
            modeled_state = dict(entry.get("modeled_state") or {})
            snapshot = dict(
                (projected_resource_snapshots or {}).get(jid)
                or entry.get("bridge_snapshot")
                or {}
            )
            normalized_snapshot = normalize_bridge_resource(
                resource_jid=jid,
                resource_type=str(
                    snapshot.get("resource_type")
                    or dict(snapshot.get("resource_core") or {}).get("resource_type")
                    or entry.get("resource_type")
                    or dict(entry.get("resource_core") or {}).get("resource_type")
                    or dict((entry.get("bridge_snapshot") or {})).get("resource_type")
                    or "resource"
                ),
                snapshot=snapshot,
                modeled_state=modeled_state,
            )
            resource_core = dict(
                normalized_snapshot.get("resource_core")
                or entry.get("resource_core")
                or {}
            )
            resource_facets = dict(
                normalized_snapshot.get("resource_facets")
                or entry.get("resource_facets")
                or {}
            )
            profile = get_resource_profile(
                str(
                    resource_core.get("resource_type")
                    or normalized_snapshot.get("resource_type")
                    or snapshot.get("resource_type")
                    or modeled_state.get("resource_type")
                    or "resource"
                ).strip().lower()
                or "resource"
            )
            effective[jid] = {
                "current_state": (
                    resource_core.get("current_state")
                    if resource_core.get("current_state") is not None
                    else normalized_snapshot.get("current_state")
                    if normalized_snapshot.get("current_state") is not None
                    else snapshot.get("current_state")
                    if snapshot.get("current_state") is not None
                    else modeled_state.get("resource_state")
                ),
                "current_location": (
                    resource_core.get("current_location")
                    if resource_core.get("current_location") is not None
                    else normalized_snapshot.get("current_location")
                    if normalized_snapshot.get("current_location") is not None
                    else snapshot.get("current_location")
                    if snapshot.get("current_location") is not None
                    else modeled_state.get("current_location")
                ),
                "resource_type": (
                    resource_core.get("resource_type")
                    or normalized_snapshot.get("resource_type")
                    or snapshot.get("resource_type")
                    or modeled_state.get("resource_type")
                ),
                "resource_core": deepcopy(resource_core),
                "resource_facets": deepcopy(resource_facets),
                "occupancy": deepcopy(
                    resource_core.get("occupancy")
                    or normalized_snapshot.get("occupancy")
                    or snapshot.get("occupancy")
                    or {}
                ),
                **resource_snapshot_fields_map(
                    normalized_snapshot,
                    profile.snapshot_fields,
                    profile=profile,
                ),
            }
        return effective

    def _bridge_select_resumable_entry_task(
        self,
        *,
        resource_jid: str,
        pending_tasks: list[dict[str, Any]],
        tools_catalog: list[dict[str, Any]],
        effective_resources: dict[str, dict[str, Any]],
        part_states: dict[str, Any],
        part_locations: dict[str, Any],
    ) -> dict[str, Any] | None:
        effective_resource = dict(effective_resources.get(resource_jid) or {})
        profile = get_resource_profile(str(effective_resource.get("resource_type") or "resource"))
        current_state = str(effective_resource.get("current_state") or "").strip()
        carried_entity = str(
            resource_snapshot_carried_entity(
                effective_resource,
                profile=profile,
            )
            or ""
        ).strip()
        current_location = effective_resource.get("current_location")
        carried_location = resource_snapshot_carried_entity_location(
            resource_jid=resource_jid,
            snapshot=effective_resource,
            profile=profile,
        )

        for task in pending_tasks:
            if not isinstance(task, dict):
                continue
            function_name = str(task.get("function_name", "")).strip()
            if not function_name:
                continue
            tool_row = self._tool_row_for_task(
                resource_jid=resource_jid,
                function_name=function_name,
                tools_catalog=tools_catalog,
            )
            params = dict(task.get("params") or {})
            part_name = str(params.get("part_name") or "").strip()
            in_state = str(tool_row.get("in_state") or "").strip()
            if in_state and in_state.lower() != "any" and in_state != current_state:
                continue

            part_in_state = str(tool_row.get("part_in_state") or "").strip()
            if part_name and part_in_state and part_states.get(part_name) != part_in_state:
                continue

            if part_name:
                if carried_entity and carried_entity != part_name:
                    continue
                if part_locations.get(part_name) == carried_location and carried_entity != part_name:
                    continue

            ctx_map = dict(tool_row.get("context_mapping") or {})
            location_param = str(ctx_map.get("location_param") or "").strip()
            location_type = str(ctx_map.get("location_type") or "").strip()
            location_value = params.get(location_param) if location_param else None
            if location_param and location_value not in (None, ""):
                if location_type == "current_location" and current_location != location_value:
                    continue
                if location_type == "part_location" and part_name:
                    if part_locations.get(part_name) != location_value:
                        continue

            return task

        return pending_tasks[0] if pending_tasks else None

    def _bridge_marked_reentry_context(
        self,
        prepared_bridge_request: dict[str, Any],
        *,
        projected_resource_snapshots: dict[str, dict[str, Any]] | None = None,
        projected_parts: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        focused_resource_jid = str(prepared_bridge_request.get("ra_jid", "") or "").strip()
        goal_state = str(prepared_bridge_request.get("goal_state", "") or "").strip()
        bridge_resources = deepcopy(prepared_bridge_request.get("bridge_resources") or {})
        tools_catalog = deepcopy(list(prepared_bridge_request.get("tools_catalog") or []))
        part_states, part_locations, _ = self._bridge_effective_part_facts(
            fallback_part_tracker=deepcopy(prepared_bridge_request.get("part_tracker") or {}),
            projected_parts=deepcopy(projected_parts or {}),
        )
        effective_resources = self._bridge_effective_resource_facts(
            bridge_resources=bridge_resources,
            projected_resource_snapshots=projected_resource_snapshots,
        )

        pending_suffix_summary: list[dict[str, Any]] = []
        requirement_entries: list[dict[str, Any]] = []

        def _add_requirement(entry: dict[str, Any]) -> None:
            entity_kind = str(entry.get("entity_kind", "")).strip()
            entity = str(entry.get("entity", "")).strip()
            field = str(entry.get("field", "")).strip()
            expected = entry.get("expected")
            if not entity_kind or not entity or not field or expected in (None, ""):
                return
            entry["_key"] = (entity_kind, entity, field, json.dumps(expected, sort_keys=True, default=str))
            requirement_entries.append(entry)

        for resource_jid, raw_entry in (bridge_resources or {}).items():
            jid = str(resource_jid or "").strip()
            if not jid:
                continue
            entry = raw_entry if isinstance(raw_entry, dict) else {}
            pending_tasks = [
                task for task in (entry.get("pending_tasks") or [])
                if isinstance(task, dict)
            ]
            if not pending_tasks:
                continue

            role = "bridge_replaced" if jid == focused_resource_jid else "resume_suffix"
            task_ids = [
                str(task.get("id", "")).strip()
                for task in pending_tasks
                if str(task.get("id", "")).strip()
            ]
            task_parts = sorted(
                {
                    str((task.get("params") or {}).get("part_name") or "").strip()
                    for task in pending_tasks
                    if str((task.get("params") or {}).get("part_name") or "").strip()
                }
            )
            suffix_summary = {
                "resource_jid": jid,
                "role": role,
                "pending_task_ids": task_ids,
                "parts": task_parts,
            }

            if jid == focused_resource_jid:
                last_task = pending_tasks[-1]
                last_fn = str(last_task.get("function_name", "")).strip()
                last_tool = self._tool_row_for_task(
                    resource_jid=jid,
                    function_name=last_fn,
                    tools_catalog=tools_catalog,
                )
                terminal_state = str(last_tool.get("out_state") or "").strip()
                suffix_summary["terminal_task_id"] = str(last_task.get("id", "")).strip()
                suffix_summary["terminal_function_name"] = last_fn
                if terminal_state:
                    suffix_summary["terminal_resource_state"] = terminal_state
                    _add_requirement(
                        {
                            "kind": "focused_resource_terminal_state",
                            "entity_kind": "resource",
                            "entity": jid,
                            "field": "current_state",
                            "expected": terminal_state,
                            "source_task_id": str(last_task.get("id", "")).strip(),
                            "source_function_name": last_fn,
                            "role": role,
                        }
                    )
                if goal_state:
                    for part_name in task_parts:
                        _add_requirement(
                            {
                                "kind": "bridge_part_goal",
                                "entity_kind": "part",
                                "entity": part_name,
                                "field": "state",
                                "expected": goal_state,
                                "source_task_ids": task_ids,
                                "role": role,
                            }
                        )
                        parts_ctx = (
                            (prepared_bridge_request.get("grounding_context") or {})
                            .get("parts", {})
                        )
                        part_ctx = parts_ctx.get(part_name) if isinstance(parts_ctx, dict) else {}
                        if not isinstance(part_ctx, dict):
                            part_ctx = {}
                        target_ctx = part_ctx.get("target") or {}
                        if not isinstance(target_ctx, dict):
                            target_ctx = {}
                        target_location = str(target_ctx.get("location") or "").strip()
                        if target_location:
                            _add_requirement(
                                {
                                    "kind": "bridge_part_goal_location",
                                    "entity_kind": "part",
                                    "entity": part_name,
                                    "field": "location",
                                    "expected": target_location,
                                    "source_task_ids": task_ids,
                                    "role": role,
                                }
                            )
            else:
                first_task = self._bridge_select_resumable_entry_task(
                    resource_jid=jid,
                    pending_tasks=pending_tasks,
                    tools_catalog=tools_catalog,
                    effective_resources=effective_resources,
                    part_states=part_states,
                    part_locations=part_locations,
                )
                if first_task is None:
                    pending_suffix_summary.append(suffix_summary)
                    continue
                first_fn = str(first_task.get("function_name", "")).strip()
                first_tool = self._tool_row_for_task(
                    resource_jid=jid,
                    function_name=first_fn,
                    tools_catalog=tools_catalog,
                )
                entry_task_id = str(first_task.get("id", "")).strip()
                entry_part_name = str((first_task.get("params") or {}).get("part_name") or "").strip()
                entry_in_state = str(first_tool.get("in_state") or "").strip()
                entry_part_state = str(first_tool.get("part_in_state") or "").strip()
                suffix_summary.update(
                    {
                        "entry_task_id": entry_task_id,
                        "entry_function_name": first_fn,
                        "entry_part_name": entry_part_name,
                    }
                )
                if entry_in_state:
                    suffix_summary["required_resource_state"] = entry_in_state
                if entry_part_state and entry_part_name:
                    suffix_summary["required_part_state"] = entry_part_state
                ctx_map = dict(first_tool.get("context_mapping") or {})
                location_param = str(ctx_map.get("location_param") or "").strip()
                location_type = str(ctx_map.get("location_type") or "").strip()
                location_value = first_task.get("params", {}).get(location_param) if location_param else None
                if location_param and location_value not in (None, ""):
                    suffix_summary["required_location"] = location_value
                    suffix_summary["required_location_type"] = location_type

                if entry_in_state and entry_in_state.lower() != "any":
                    _add_requirement(
                        {
                            "kind": "resume_entry_resource_state",
                            "entity_kind": "resource",
                            "entity": jid,
                            "field": "current_state",
                            "expected": entry_in_state,
                            "source_task_id": entry_task_id,
                            "source_function_name": first_fn,
                            "role": role,
                        }
                    )
                if location_param and location_value not in (None, ""):
                    if location_type == "current_location":
                        _add_requirement(
                            {
                                "kind": "resume_entry_resource_location",
                                "entity_kind": "resource",
                                "entity": jid,
                                "field": "current_location",
                                "expected": location_value,
                                "source_task_id": entry_task_id,
                                "source_function_name": first_fn,
                                "role": role,
                            }
                        )
                    elif location_type == "part_location" and entry_part_name:
                        _add_requirement(
                            {
                                "kind": "resume_entry_part_location",
                                "entity_kind": "part",
                                "entity": entry_part_name,
                                "field": "location",
                                "expected": location_value,
                                "source_task_id": entry_task_id,
                                "source_function_name": first_fn,
                                "role": role,
                            }
                        )
                if entry_part_name and entry_part_state:
                    profile = get_resource_profile(
                        str((effective_resources.get(jid) or {}).get("resource_type") or "resource")
                    )
                    _add_requirement(
                        {
                            "kind": "resume_entry_part_state",
                            "entity_kind": "part",
                            "entity": entry_part_name,
                            "field": "state",
                            "expected": entry_part_state,
                            "source_task_id": entry_task_id,
                            "source_function_name": first_fn,
                            "role": role,
                        }
                    )
                    carried_entity_field = str(profile.carried_entity_field or "").strip()
                    if carried_entity_field:
                        _add_requirement(
                            {
                                "kind": "resume_entry_carried_entity",
                                "entity_kind": "resource",
                                "entity": jid,
                                "field": carried_entity_field,
                                "expected": entry_part_name,
                                "source_task_id": entry_task_id,
                                "source_function_name": first_fn,
                                "role": role,
                            }
                        )

            pending_suffix_summary.append(suffix_summary)

        deduped_requirements: list[dict[str, Any]] = []
        seen_requirement_keys: set[tuple[str, str, str, str]] = set()
        for entry in requirement_entries:
            key = entry.pop("_key", None)
            if key in seen_requirement_keys:
                continue
            seen_requirement_keys.add(key)
            deduped_requirements.append(entry)

        unmet_reentry_conditions: list[dict[str, Any]] = []
        for requirement in deduped_requirements:
            entity_kind = str(requirement.get("entity_kind", "")).strip()
            entity = str(requirement.get("entity", "")).strip()
            field = str(requirement.get("field", "")).strip()
            expected = requirement.get("expected")
            actual = None
            if entity_kind == "resource":
                actual = (effective_resources.get(entity) or {}).get(field)
            elif entity_kind == "part":
                if field == "state":
                    actual = part_states.get(entity)
                elif field == "location":
                    actual = part_locations.get(entity)
            if actual != expected:
                unmet = deepcopy(requirement)
                unmet["actual"] = actual
                unmet_reentry_conditions.append(unmet)

        return {
            "goal_state": goal_state,
            "focused_resource_jid": focused_resource_jid,
            "pending_suffix_summary": pending_suffix_summary,
            "marked_reentry_conditions": deduped_requirements,
            "unmet_reentry_conditions": unmet_reentry_conditions,
            "pending_suffixes": pending_suffix_summary,
            "continuation_requirements": deduped_requirements,
            "unmet_requirements": unmet_reentry_conditions,
        }

    def _derive_bridge_safety_constraints(
        self,
        prepared_bridge_request: dict[str, Any],
        *,
        marked_reentry_context: dict[str, Any],
    ) -> dict[str, Any]:
        del marked_reentry_context
        raw = deepcopy(prepared_bridge_request.get("bridge_safety_context") or {})
        if not isinstance(raw, dict):
            raw = {}
        raw["constraints"] = [
            deepcopy(item)
            for item in (raw.get("constraints") or [])
            if isinstance(item, dict)
        ]
        raw["safety_rules"] = [
            deepcopy(item)
            for item in (raw.get("safety_rules") or [])
            if isinstance(item, dict)
        ]
        raw["rule_ids"] = [
            str(rule_id).strip()
            for rule_id in (raw.get("rule_ids") or [])
            if str(rule_id).strip()
        ]
        return raw

    def _collect_bridge_safety_context(
        self,
        violations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        rule_ids: list[str] = []
        safe_next_task_ids: list[str] = []
        running_aps: list[Any] = []
        candidate_aps: list[Any] = []
        predicted_state_aps: list[Any] = []
        constraints: list[dict[str, Any]] = []
        safety_rules: list[dict[str, Any]] = []
        statuses: list[str] = []
        reasons: list[str] = []
        seen_rule_ids: set[str] = set()
        seen_next_task_ids: set[str] = set()
        seen_constraint_signatures: set[str] = set()
        seen_safety_rule_signatures: set[str] = set()

        for violation in violations or []:
            if not isinstance(violation, dict):
                continue
            safety_ctx = violation.get("safety_ctx") or {}
            if not isinstance(safety_ctx, dict):
                continue
            raw_bridge = safety_ctx.get("bridge_safety_context") or {}
            if not isinstance(raw_bridge, dict):
                raw_bridge = {}

            raw_rule_ids = (
                raw_bridge.get("rule_ids")
                or safety_ctx.get("rule_ids")
                or safety_ctx.get("violated_rule")
                or safety_ctx.get("violated_rule_id")
            )
            if isinstance(raw_rule_ids, (list, tuple, set)):
                values = raw_rule_ids
            else:
                values = [raw_rule_ids]
            for value in values:
                token = str(value or "").strip()
                if token and token not in seen_rule_ids:
                    seen_rule_ids.add(token)
                    rule_ids.append(token)

            for task_id in (
                raw_bridge.get("safe_next_task_ids")
                or safety_ctx.get("safe_next_task_ids")
                or []
            ):
                token = str(task_id or "").strip()
                if token and token not in seen_next_task_ids:
                    seen_next_task_ids.add(token)
                    safe_next_task_ids.append(token)

            running_aps.extend(list(raw_bridge.get("running_aps") or safety_ctx.get("running_aps") or []))
            candidate_aps.extend(list(raw_bridge.get("candidate_aps") or safety_ctx.get("candidate_aps") or []))
            predicted_state_aps.extend(
                list(raw_bridge.get("predicted_state_aps") or safety_ctx.get("predicted_state_aps") or [])
            )
            status = str(raw_bridge.get("status") or safety_ctx.get("status") or "").strip()
            if status:
                statuses.append(status)
            reason = str(raw_bridge.get("reason") or safety_ctx.get("reason") or "").strip()
            if reason:
                reasons.append(reason)

            for constraint in (raw_bridge.get("constraints") or safety_ctx.get("constraints") or []):
                if not isinstance(constraint, dict):
                    continue
                signature = self._bridge_constraint_signature(constraint)
                if not signature or signature in seen_constraint_signatures:
                    continue
                seen_constraint_signatures.add(signature)
                constraints.append(deepcopy(constraint))

            for rule in (raw_bridge.get("safety_rules") or safety_ctx.get("safety_rules") or []):
                if not isinstance(rule, dict):
                    continue
                signature = self._bridge_safety_rule_signature(rule)
                if not signature or signature in seen_safety_rule_signatures:
                    continue
                seen_safety_rule_signatures.add(signature)
                safety_rules.append(deepcopy(rule))

        return {
            "rule_ids": rule_ids,
            "safe_next_task_ids": safe_next_task_ids,
            "running_aps": running_aps,
            "candidate_aps": candidate_aps,
            "predicted_state_aps": predicted_state_aps,
            "status": statuses[-1] if statuses else "",
            "reason": reasons[-1] if reasons else "",
            "constraints": constraints,
            "safety_rules": safety_rules,
        }

    @staticmethod
    def _bridge_constraint_signature(constraint: dict[str, Any]) -> str:
        if not isinstance(constraint, dict):
            return ""
        return json.dumps(constraint, sort_keys=True, default=str)

    @staticmethod
    def _bridge_safety_rule_signature(rule: dict[str, Any]) -> str:
        if not isinstance(rule, dict):
            return ""
        rule_id = str(rule.get("id", "") or "").strip()
        if rule_id:
            return rule_id
        return json.dumps(rule, sort_keys=True, default=str)

    @staticmethod
    def _format_pose_brief(value: Any) -> str:
        if not isinstance(value, dict):
            return "-"
        try:
            return "({x:.3f}, {y:.3f}, {z:.3f})".format(
                x=float(value.get("x", 0.0)),
                y=float(value.get("y", 0.0)),
                z=float(value.get("z", 0.0)),
            )
        except (TypeError, ValueError):
            return str(value)

    def _prepare_trace_lines(self, context_summary: dict[str, Any]) -> list[str]:
        fault_event = dict(context_summary.get("fault_event") or {})
        current_product_state = dict(context_summary.get("current_product_state") or {})
        relevant_assembly_requirements = list(context_summary.get("relevant_assembly_requirements") or [])
        modeled_continuation_gap = dict(context_summary.get("modeled_continuation_gap") or {})
        resources = list(current_product_state.get("resources") or [])
        parts = list(current_product_state.get("parts") or [])
        active_safety_diagnosis = dict(current_product_state.get("active_safety_diagnosis") or {})
        loaded_safety_rules = list(current_product_state.get("loaded_safety_rules") or [])
        has_active_safety_diagnosis = bool(
            (active_safety_diagnosis.get("obligation_targets") or [])
            or (active_safety_diagnosis.get("rule_ids") or [])
            or (active_safety_diagnosis.get("active_rules") or [])
            or str(active_safety_diagnosis.get("status") or "").strip()
            or str(active_safety_diagnosis.get("reason") or "").strip()
        )

        def _derived_from(entry: dict[str, Any]) -> str:
            task_id = str(entry.get("source_task_id") or "").strip()
            function_name = str(entry.get("source_function_name") or "").strip()
            if not task_id and not function_name:
                return "-"
            return f"{task_id or '-'}" + (f"/{function_name}" if function_name else "")

        lines = [
            "",
            "Prepare-trace context build completed.",
            "",
            "Fault Event",
            f"  focused_resource_jid: {fault_event.get('focused_resource_jid')}",
            f"  blocked_at_task_id:   {fault_event.get('blocked_at_task_id') or '-'}",
            f"  blocked_at_function:  {fault_event.get('blocked_at_function') or '-'}",
            f"  resource_state:       {fault_event.get('resource_state') or '-'}",
            "",
            "Current Product State",
            "Resources",
        ]
        for row in resources:
            if not isinstance(row, dict):
                continue
            lines.append(
                "  {jid}: state={state}, held_part={held}, location={location} [{basis}], "
                "availability={availability}, pending={pending}".format(
                    jid=row.get("resource_jid") or "-",
                    state=row.get("current_state") or "-",
                    held=row.get("held_part") or "-",
                    location=row.get("current_location") or "-",
                    basis=row.get("current_location_basis") or "-",
                    availability=row.get("availability") or "-",
                    pending=row.get("pending_task_ids") or [],
                )
            )

        lines.extend(["", "Parts"])
        for row in parts:
            if not isinstance(row, dict):
                continue
            lines.append(
                "  {part}: state={state}, location={location} [{basis}], observed_pose={pose}, "
                "target_location={target}".format(
                    part=row.get("part_name") or "-",
                    state=row.get("state") or "-",
                    location=row.get("location") or "-",
                    basis=row.get("location_basis") or "-",
                    pose=self._format_pose_brief(row.get("observed_pose")),
                    target=row.get("target_location") or "-",
                )
            )

        if has_active_safety_diagnosis:
            lines.extend(
                [
                    "",
                    "Active Safety Diagnosis",
                    f"  obligation_targets: {active_safety_diagnosis.get('obligation_targets') or []}",
                    f"  active_rule_ids:    {active_safety_diagnosis.get('rule_ids') or []}",
                ]
            )
            if str(active_safety_diagnosis.get("status") or "").strip():
                lines.append(f"  status:             {active_safety_diagnosis.get('status')}")
            if str(active_safety_diagnosis.get("reason") or "").strip():
                lines.append(f"  reason:             {active_safety_diagnosis.get('reason')}")
            for rule in (active_safety_diagnosis.get("active_rules") or [])[:4]:
                if not isinstance(rule, dict):
                    continue
                lines.append(
                    "  active_rule: {rule_id} [{constraint_type}] {summary}".format(
                        rule_id=rule.get("rule_id") or "-",
                        constraint_type=rule.get("constraint_type") or "-",
                        summary=rule.get("summary") or "-",
                    )
                )
        lines.extend(
            [
                "",
                "Loaded Safety Rules",
            ]
        )
        if not loaded_safety_rules:
            lines.append("  rules: []")
        for rule in loaded_safety_rules[:6]:
            if not isinstance(rule, dict):
                continue
            lines.append(
                "  loaded_rule: {rule_id} [{constraint_type}] {summary}".format(
                    rule_id=rule.get("rule_id") or "-",
                    constraint_type=rule.get("constraint_type") or "-",
                    summary=rule.get("summary") or "-",
                )
            )

        lines.extend(["", "Relevant Assembly Requirements"])
        if not relevant_assembly_requirements:
            lines.append("  requirements: []")
        for requirement in relevant_assembly_requirements:
            if not isinstance(requirement, dict):
                continue
            lines.append(
                "  {requirement_id} [{status}] {summary}".format(
                    requirement_id=requirement.get("requirement_id") or "-",
                    status=requirement.get("status") or "unknown",
                    summary=requirement.get("summary") or "-",
                )
            )

        lines.extend(
            [
                "",
                "Modeled Continuation Gap",
                f"  goal_state:               {modeled_continuation_gap.get('goal_state') or '-'}",
                f"  pending_nominal_task_ids: {modeled_continuation_gap.get('pending_nominal_task_ids') or []}",
                f"  resume_ready:             {modeled_continuation_gap.get('resume_ready')}",
            ]
        )
        for entry in (modeled_continuation_gap.get("unsatisfied_conditions") or [])[:8]:
            if not isinstance(entry, dict):
                continue
            condition_family = str(entry.get("condition_family") or "").strip()
            if condition_family == "goal":
                lines.append(
                    "  unmet goal: {entity}.{field} expected={expected!r} actual={actual!r}".format(
                        entity=entry.get("entity") or "?",
                        field=entry.get("field") or "?",
                        expected=entry.get("expected"),
                        actual=entry.get("actual"),
                    )
                )
            else:
                lines.append(
                    "  unmet continuation: {entity}.{field} expected={expected!r} actual={actual!r} "
                    "derived_from={derived_from}".format(
                        entity=entry.get("entity") or "?",
                        field=entry.get("field") or "?",
                        expected=entry.get("expected"),
                        actual=entry.get("actual"),
                        derived_from=_derived_from(entry),
                    )
                )
        return lines

    def emit_prepare_trace_summary(self, prepared_bridge_request: dict[str, Any]) -> None:
        context_summary = dict((prepared_bridge_request or {}).get("context_summary") or {})
        if not context_summary:
            return
        for line in self._prepare_trace_lines(context_summary):
            print(line, flush=True)

    def _bridge_summary(self, proposal: dict[str, Any] | None) -> list[str]:
        if not isinstance(proposal, dict):
            return []
        summary: list[str] = []
        primary_obligation = proposal.get("primary_obligation") or {}
        rule_id = str(primary_obligation.get("rule_id") or "").strip() if isinstance(primary_obligation, dict) else ""
        if rule_id:
            summary.append(f"rule:{rule_id}")
        for task in proposal.get("macro_tasks") or []:
            if not isinstance(task, dict):
                continue
            macro_name = str(task.get("macro_name", "")).strip()
            if macro_name:
                summary.append(macro_name)
            for step in task.get("primitive_steps") or []:
                if not isinstance(step, dict):
                    continue
                primitive = str(step.get("primitive", "")).strip()
                if primitive:
                    summary.append(primitive)
        macro_name = str(proposal.get("macro_name", "")).strip()
        if macro_name:
            summary.append(macro_name)
        function_name = str(proposal.get("function_name", "")).strip()
        if function_name:
            summary.append(function_name)
        for step in proposal.get("primitive_steps") or []:
            if not isinstance(step, dict):
                continue
            primitive = str(step.get("primitive", "")).strip()
            if primitive:
                summary.append(primitive)
        for step in proposal.get("macro_steps") or []:
            if not isinstance(step, dict):
                continue
            fn = str(step.get("function_name", "")).strip()
            if fn:
                summary.append(fn)
        return summary

    def validate_preprogrammed_bridge_proposal(
        self,
        *,
        proposal: dict[str, Any],
        prepared_bridge_request: dict[str, Any],
        source: str = "preprogrammed_scenario",
        scenario_id: str = "",
    ) -> dict[str, Any]:
        del proposal, prepared_bridge_request, source, scenario_id
        raise NotImplementedError(
            "preprogrammed bridge proposal validation is not wired in the active bridge mode"
        )

    @staticmethod
    def _failure_anchor_from_payload(
        failure_context_raw: dict[str, Any] | None,
        *,
        fallback_resource_jid: str = "",
    ) -> dict[str, Any]:
        raw = dict(failure_context_raw or {})
        details = raw.get("failure_context")
        details = details if isinstance(details, dict) else raw
        observations = details.get("observations")
        observations = observations if isinstance(observations, dict) else {}
        affected_parts: list[dict[str, Any]] = []
        for entity in (details.get("affected_entities") or []):
            if not isinstance(entity, dict):
                continue
            if str(entity.get("entity_type") or "").strip().lower() != "part":
                continue
            part_name = str(entity.get("entity_id") or "").strip()
            if not part_name:
                continue
            affected_parts.append(
                {
                    "part_name": part_name,
                    "state": deepcopy(entity.get("state")),
                    "location": deepcopy(entity.get("location")),
                }
            )

        state_before = observations.get("state_before")
        state_before = deepcopy(state_before) if isinstance(state_before, dict) else {}
        state_after = observations.get("state_after")
        state_after = deepcopy(state_after) if isinstance(state_after, dict) else {}

        return {
            "failed_task_id": str(raw.get("failed_task_id") or raw.get("task_id") or "").strip(),
            "failed_function_name": str(
                raw.get("failed_function_name") or observations.get("function_name") or ""
            ).strip(),
            "failed_resource_jid": str(
                raw.get("failed_resource_jid")
                or raw.get("resource_jid")
                or fallback_resource_jid
                or ""
            ).strip(),
            "resource_state_before": state_before,
            "resource_state_after": state_after,
            "affected_parts": affected_parts,
        }

    def _build_context_summary(
        self,
        prepared_bridge_request: dict[str, Any],
        *,
        input_bridge_safety_context: dict[str, Any],
    ) -> dict[str, Any]:
        failure_anchor = dict(prepared_bridge_request.get("failure_anchor") or {})
        focused_resource_jid = str(
            failure_anchor.get("failed_resource_jid")
            or prepared_bridge_request.get("ra_jid")
            or ""
        ).strip()
        bridge_resources = dict(prepared_bridge_request.get("bridge_resources") or {})
        grounding_context = dict(prepared_bridge_request.get("grounding_context") or {})
        marked_reentry_context = dict(prepared_bridge_request.get("marked_reentry_context") or {})
        bridge_safety_context = dict(prepared_bridge_request.get("bridge_safety_context") or {})
        blocked_task_id = str(failure_anchor.get("failed_task_id") or "").strip()
        blocked_function_name = str(failure_anchor.get("failed_function_name") or "").strip()

        resources: list[dict[str, Any]] = []
        for resource_jid in sorted(bridge_resources):
            entry = dict(bridge_resources.get(resource_jid) or {})
            snapshot = dict(entry.get("bridge_snapshot") or {})
            resource_type = str(
                entry.get("resource_type")
                or snapshot.get("resource_type")
                or dict(snapshot.get("resource_core") or {}).get("resource_type")
                or "resource"
            ).strip()
            profile = get_resource_profile(resource_type)
            pending_tasks = [
                task for task in (entry.get("pending_tasks") or [])
                if isinstance(task, dict)
            ]
            current_location, current_location_basis = self._resolve_resource_location(
                resource_jid=resource_jid,
                snapshot=snapshot,
                profile=profile,
                prepared_bridge_request=prepared_bridge_request,
            )
            resources.append(
                {
                    "resource_jid": resource_jid,
                    "role": "focused" if resource_jid == focused_resource_jid else "supporting",
                    "current_state": deepcopy(snapshot.get("current_state")),
                    "current_location": deepcopy(current_location),
                    "current_location_basis": current_location_basis,
                    "availability": deepcopy(snapshot.get("availability")),
                    "held_part": deepcopy(resource_snapshot_carried_entity(snapshot, profile=profile)),
                    "held_part_location": resource_snapshot_carried_entity_location(
                        resource_jid=resource_jid,
                        snapshot=snapshot,
                        profile=profile,
                    ),
                    "pending_task_ids": [
                        str(task.get("id") or "").strip()
                        for task in pending_tasks
                        if str(task.get("id") or "").strip()
                    ],
                }
            )

        parts_ctx = dict(grounding_context.get("parts") or {})
        part_names = sorted({
            str(name).strip()
            for name in (list(prepared_bridge_request.get("part_tracker") or {}) + list(parts_ctx))
            if str(name).strip()
        })
        parts: list[dict[str, Any]] = []
        for part_name in part_names:
            tracker_entry = dict((prepared_bridge_request.get("part_tracker") or {}).get(part_name) or {})
            ctx_entry = dict(parts_ctx.get(part_name) or {})
            target = dict(ctx_entry.get("target") or {})
            part_location = (
                tracker_entry.get("location")
                if tracker_entry.get("location") is not None
                else ctx_entry.get("location")
                if ctx_entry.get("location") is not None
                else None
            )
            location_basis = str(
                tracker_entry.get("location_basis")
                or ctx_entry.get("location_basis")
                or (
                    "part_tracker"
                    if part_location not in (None, "")
                    else "unavailable"
                )
            ).strip()
            parts.append(
                {
                    "part_name": part_name,
                    "state": deepcopy(
                        tracker_entry.get("state")
                        if tracker_entry.get("state") is not None
                        else ctx_entry.get("state")
                        if ctx_entry.get("state") is not None
                        else None
                    ),
                    "location": deepcopy(
                        part_location
                    ),
                    "location_basis": location_basis,
                    "observed_pose": deepcopy(
                        tracker_entry.get("observed_pose")
                        if tracker_entry.get("observed_pose") is not None
                        else ctx_entry.get("observed_pose")
                    ),
                    "target_location": deepcopy(target.get("location")),
                    "target_model_name": deepcopy(target.get("model_name")),
                }
            )

        parts_by_name = {
            str(row.get("part_name") or "").strip(): row
            for row in parts
            if isinstance(row, dict) and str(row.get("part_name") or "").strip()
        }
        continuation_requirements = [
            self._brief_condition(entry, condition_family="continuation")
            for entry in (marked_reentry_context.get("marked_reentry_conditions") or [])
            if isinstance(entry, dict)
        ]
        unmet_continuation_conditions = [
            self._brief_condition(entry, condition_family="continuation")
            for entry in (marked_reentry_context.get("unmet_reentry_conditions") or [])
            if isinstance(entry, dict)
        ]
        goal_state = str(prepared_bridge_request.get("goal_state") or "").strip()
        goal_conditions: list[dict[str, Any]] = []
        unsatisfied_goal_conditions: list[dict[str, Any]] = []
        for part_name in [
            str(name or "").strip()
            for name in (prepared_bridge_request.get("P_id") or [])
            if str(name or "").strip()
        ]:
            part_row = dict(parts_by_name.get(part_name) or {})
            if goal_state:
                state_condition = {
                    "kind": "goal_part_state",
                    "condition_family": "goal",
                    "entity_kind": "part",
                    "entity": part_name,
                    "field": "state",
                    "expected": goal_state,
                }
                goal_conditions.append(state_condition)
                actual_state = part_row.get("state")
                if actual_state != goal_state:
                    unsatisfied_entry = deepcopy(state_condition)
                    unsatisfied_entry["actual"] = deepcopy(actual_state)
                    unsatisfied_goal_conditions.append(unsatisfied_entry)
            target_location = part_row.get("target_location")
            if target_location not in (None, ""):
                location_condition = {
                    "kind": "goal_part_location",
                    "condition_family": "goal",
                    "entity_kind": "part",
                    "entity": part_name,
                    "field": "location",
                    "expected": deepcopy(target_location),
                }
                goal_conditions.append(location_condition)
                actual_location = part_row.get("location")
                if actual_location != target_location:
                    unsatisfied_entry = deepcopy(location_condition)
                    unsatisfied_entry["actual"] = deepcopy(actual_location)
                    unsatisfied_goal_conditions.append(unsatisfied_entry)

        # Detect pending suffix tasks blocked by safety ordering rules.
        # For each "ordering_place_approach_priority" rule, if the gateway part
        # has not yet been assembled, find pending place_approach tasks for the
        # blocked part and inject them as continuation conditions.  This ensures
        # the LLM sees the full scope of the deadlock, not just the xarm6 state gap.
        for _rule in (prepared_bridge_request.get("loaded_safety_rules") or []):
            if not isinstance(_rule, dict):
                continue
            if str(_rule.get("constraint_type") or "").strip() != "ordering_place_approach_priority":
                continue
            _rule_id = str(_rule.get("id") or _rule.get("rule_id") or "").strip()
            _products = [str(p or "").lower().strip() for p in (_rule.get("product") or [])]
            if len(_products) < 2:
                continue
            _gateway_lower = _products[0]
            _blocked_lower = _products[1]
            _dest = str((_rule.get("context") or {}).get("destination") or "").strip()
            _gw_row = next(
                (row for name, row in parts_by_name.items() if name.lower() == _gateway_lower),
                {},
            )
            _gw_state = _gw_row.get("state")
            _gw_location = _gw_row.get("location")
            _gateway_satisfied = (
                bool(goal_state)
                and _gw_state == goal_state
                and (not _dest or _gw_location == _dest)
            )
            if _gateway_satisfied:
                continue
            # The blocked event type comes from the rule itself (e.g. "place_approach"),
            # so detection generalises to any ordering constraint the rule specifies.
            _blocked_event = str(_rule.get("event") or "").strip()
            for _raw_entry in bridge_resources.values():
                if not isinstance(_raw_entry, dict):
                    continue
                for _task in (_raw_entry.get("pending_tasks") or []):
                    if not isinstance(_task, dict):
                        continue
                    _fn = str(_task.get("function_name") or "").strip()
                    _task_part = str((_task.get("params") or {}).get("part_name") or "").lower().strip()
                    _task_id = str(_task.get("id") or "").strip()
                    if (
                        _task_id
                        and _task_part == _blocked_lower
                        and (_blocked_event and _fn == _blocked_event)
                    ):
                        unmet_continuation_conditions.append({
                            "kind": "safety_blocked_suffix_task",
                            "condition_family": "continuation",
                            "entity_kind": "task",
                            "entity": _task_id,
                            "field": "safety_gate",
                            "expected": "unblocked",
                            "actual": f"blocked_by_{_rule_id}",
                            "blocking_rule_id": _rule_id,
                            "blocking_reason": (
                                f"{_gateway_lower.upper()} must be placed at {_dest!r} "
                                f"before {_blocked_lower.upper()} place_approach"
                            ),
                        })

        unsatisfied_conditions: list[dict[str, Any]] = []
        seen_unsatisfied_signatures: set[tuple[str, str, str, str, str]] = set()
        for entry in unsatisfied_goal_conditions + unmet_continuation_conditions:
            signature = self._condition_signature(entry)
            if signature in seen_unsatisfied_signatures:
                continue
            seen_unsatisfied_signatures.add(signature)
            unsatisfied_conditions.append(deepcopy(entry))

        active_rule_ids = [
            str(rule_id or "").strip()
            for rule_id in (bridge_safety_context.get("rule_ids") or [])
            if str(rule_id or "").strip()
        ]
        active_safety_diagnosis = {
            "obligation_targets": [
                deepcopy(target)
                for target in (prepared_bridge_request.get("obligation_targets") or [])
                if isinstance(target, dict)
            ],
            "rule_ids": active_rule_ids,
            "active_rules": [
                self._brief_safety_rule(rule)
                for rule in (bridge_safety_context.get("safety_rules") or [])
                if isinstance(rule, dict)
            ],
            "status": deepcopy(bridge_safety_context.get("status")),
            "reason": deepcopy(bridge_safety_context.get("reason")),
            "safe_next_task_ids": deepcopy(bridge_safety_context.get("safe_next_task_ids") or []),
        }
        relevant_loaded_safety_rules = self._filter_relevant_loaded_safety_rules(
            raw_rules=[
                deepcopy(rule)
                for rule in (prepared_bridge_request.get("loaded_safety_rules") or [])
                if isinstance(rule, dict)
            ],
            active_rule_ids=active_rule_ids,
            relevant_parts=[
                str(name or "").strip()
                for name in (prepared_bridge_request.get("P_id") or [])
                if str(name or "").strip()
            ],
            relevant_resource_jids=list(bridge_resources),
            relevant_locations=[
                str(row.get("target_location") or "").strip()
                for row in parts
                if isinstance(row, dict) and str(row.get("target_location") or "").strip()
            ],
        )
        relevant_assembly_requirements = self._build_relevant_assembly_requirements(
            prepared_bridge_request=prepared_bridge_request,
            bridge_resources=bridge_resources,
            failure_anchor=failure_anchor,
        )
        modeled_continuation_gap = {
            "basis": "remaining modeled suffix requirements",
            "goal_state": deepcopy(prepared_bridge_request.get("goal_state")),
            "remaining_suffixes": deepcopy(marked_reentry_context.get("pending_suffix_summary") or []),
            "pending_nominal_task_ids": [
                str(task.get("id") or "").strip()
                for raw_entry in bridge_resources.values()
                for task in (raw_entry.get("pending_tasks") or [])
                if isinstance(raw_entry, dict)
                and isinstance(task, dict)
                and str(task.get("id") or "").strip()
            ],
            "pending_resource_jids": [
                str(resource_jid)
                for resource_jid, raw_entry in bridge_resources.items()
                if isinstance(raw_entry, dict) and list(raw_entry.get("pending_tasks") or [])
            ],
            "goal_conditions": deepcopy(goal_conditions),
            "continuation_requirements": deepcopy(continuation_requirements),
            "unsatisfied_goal_conditions": deepcopy(unsatisfied_goal_conditions),
            "unsatisfied_conditions": deepcopy(unsatisfied_conditions),
            "resume_ready": not unsatisfied_conditions,
            "blocking_reasons": deepcopy(unsatisfied_conditions),
        }

        return {
            "fault_event": {
                "focused_resource_jid": focused_resource_jid,
                "blocked_at_task_id": blocked_task_id,
                "blocked_at_function": blocked_function_name,
                "resource_state": deepcopy(
                    (failure_anchor.get("resource_state_after") or {}).get("current_state")
                    if isinstance(failure_anchor.get("resource_state_after"), dict)
                    else None
                )
                or deepcopy((prepared_bridge_request.get("stuck_state") or {}).get("resource_state")),
                "current_location": deepcopy((prepared_bridge_request.get("stuck_state") or {}).get("current_location")),
                "resource_state_before": deepcopy(failure_anchor.get("resource_state_before") or {}),
                "resource_state_after": deepcopy(failure_anchor.get("resource_state_after") or {}),
                "affected_parts": deepcopy(failure_anchor.get("affected_parts") or []),
            },
            "current_product_state": {
                "resources": resources,
                "parts": parts,
                "active_safety_diagnosis": active_safety_diagnosis,
                "loaded_safety_rules": relevant_loaded_safety_rules,
                "formal_state": deepcopy(prepared_bridge_request.get("des_monitor") or {}),
            },
            "relevant_assembly_requirements": relevant_assembly_requirements,
            "modeled_continuation_gap": modeled_continuation_gap,
            "data_flow_trace": {
                "fault_event": ["runtime failure inputs", "failure_context_raw", "stuck_state"],
                "current_product_state": [
                    "resource agent snapshot",
                    "resource_states",
                    "part_tracker",
                    "grounding_context",
                    "des_monitor",
                ],
                "active_safety_diagnosis": [
                    "bridge_safety_context",
                    "runtime safety diagnosis",
                ],
                "loaded_safety_rules": [
                    "verified bundle safety_logic_json",
                    "prepared_bridge_request.loaded_safety_rules",
                ],
                "relevant_assembly_requirements": [
                    "requirements_json",
                    "planner task nodes",
                    "requirements_status",
                    "remaining pending tasks",
                ],
                "modeled_continuation_gap": [
                    "goal_state",
                    "remaining goal parts",
                    "pending tasks",
                    "tool metadata",
                    "derived reentry context",
                ],
            },
        }

    def _build_llm_input(
        self,
        prepared_bridge_request: dict[str, Any],
    ) -> dict[str, Any]:
        context_summary = dict(prepared_bridge_request.get("context_summary") or {})
        fault_event = dict(context_summary.get("fault_event") or {})
        current_product_state = dict(context_summary.get("current_product_state") or {})
        active_safety_diagnosis = dict(current_product_state.get("active_safety_diagnosis") or {})
        relevant_assembly_requirements = [
            deepcopy(entry)
            for entry in (context_summary.get("relevant_assembly_requirements") or [])
            if isinstance(entry, dict)
        ]
        modeled_continuation_gap = dict(context_summary.get("modeled_continuation_gap") or {})
        bridge_resources = dict(prepared_bridge_request.get("bridge_resources") or {})
        current_resources = [
            dict(row)
            for row in (current_product_state.get("resources") or [])
            if isinstance(row, dict)
        ]
        current_resource_by_jid = {
            str(row.get("resource_jid") or "").strip(): row
            for row in current_resources
            if str(row.get("resource_jid") or "").strip()
        }
        ordered_resource_jids = self._ordered_resource_jids(
            focused_resource_jid=str(fault_event.get("focused_resource_jid") or "").strip(),
            observed_resources=current_resources,
            bridge_resources=bridge_resources,
        )

        observed_resources: list[dict[str, Any]] = []
        held_part_to_resource: dict[str, str] = {}
        for resource_jid in ordered_resource_jids:
            current_row = dict(current_resource_by_jid.get(resource_jid) or {})
            bridge_entry = dict(bridge_resources.get(resource_jid) or {})
            bridge_snapshot = self._build_prompt_bridge_snapshot(bridge_entry)
            observed_resource = {
                "resource_jid": resource_jid,
                "current_state": deepcopy(
                    current_row.get("current_state")
                    if "current_state" in current_row
                    else bridge_snapshot.get("current_state")
                ),
                "current_location": deepcopy(
                    current_row.get("current_location")
                    if "current_location" in current_row
                    else bridge_snapshot.get("current_location")
                ),
                "availability": deepcopy(
                    current_row.get("availability")
                    if "availability" in current_row
                    else bridge_snapshot.get("availability")
                ),
                "held_part": deepcopy(
                    current_row.get("held_part")
                    if "held_part" in current_row
                    else bridge_snapshot.get("held_part")
                ),
            }
            for optional_field in (
                "gripper_state",
                "current_pose",
                "current_pose_ref",
                "named_poses",
                "workspace_bounds",
            ):
                optional_value = deepcopy(bridge_snapshot.get(optional_field))
                if optional_value not in (None, "", [], {}):
                    observed_resource[optional_field] = optional_value
            observed_resources.append(observed_resource)
            held_part = str(observed_resource.get("held_part") or "").strip()
            if held_part:
                held_part_to_resource.setdefault(held_part, resource_jid)

        raw_unmet_continuation_conditions = [
            deepcopy(entry)
            for entry in (modeled_continuation_gap.get("unsatisfied_conditions") or [])
            if isinstance(entry, dict)
            and str(entry.get("condition_family") or "").strip() == "continuation"
        ]
        unmet_continuation_conditions: list[dict[str, Any]] = []
        condition_ids_by_task_id: dict[str, list[str]] = {}
        for entry in raw_unmet_continuation_conditions:
            enriched_entry = deepcopy(entry)
            condition_id = self._condition_id(enriched_entry)
            enriched_entry["condition_id"] = condition_id
            unmet_continuation_conditions.append(enriched_entry)

            task_refs: set[str] = set()
            source_task_id = str(enriched_entry.get("source_task_id") or "").strip()
            if source_task_id:
                task_refs.add(source_task_id)
            if str(enriched_entry.get("entity_kind") or "").strip() == "task":
                entity_task_id = str(enriched_entry.get("entity") or "").strip()
                if entity_task_id:
                    task_refs.add(entity_task_id)
            task_refs.update(
                str(task_id or "").strip()
                for task_id in (enriched_entry.get("source_task_ids") or [])
                if str(task_id or "").strip()
            )
            for task_id in task_refs:
                bucket = condition_ids_by_task_id.setdefault(task_id, [])
                if condition_id not in bucket:
                    bucket.append(condition_id)

        pending_tasks_detail: list[dict[str, Any]] = []
        pending_task_ids_by_part: dict[str, list[str]] = {}
        for resource_jid in sorted(bridge_resources):
            raw_entry = dict(bridge_resources.get(resource_jid) or {})
            for task in (raw_entry.get("pending_tasks") or []):
                if not isinstance(task, dict):
                    continue
                task_id = str(task.get("id") or "").strip()
                if not task_id:
                    continue
                task_part = str((task.get("params") or {}).get("part_name") or "").strip() or None
                pending_tasks_detail.append(
                    {
                        "id": task_id,
                        "function": str(task.get("function_name") or "").strip(),
                        "part": task_part,
                        "resource": resource_jid,
                        "blocked_by_condition_ids": deepcopy(
                            condition_ids_by_task_id.get(task_id) or []
                        ),
                    }
                )
                if task_part:
                    pending_task_ids_by_part.setdefault(task_part, []).append(task_id)

        requirements_by_product: dict[str, list[dict[str, Any]]] = {}
        for requirement_entry in relevant_assembly_requirements:
            product = str(requirement_entry.get("product") or "").strip()
            if product:
                requirements_by_product.setdefault(product, []).append(requirement_entry)

        current_parts = [
            dict(row)
            for row in (current_product_state.get("parts") or [])
            if isinstance(row, dict)
        ]
        current_part_by_name = {
            str(row.get("part_name") or "").strip(): row
            for row in current_parts
            if str(row.get("part_name") or "").strip()
        }
        tracker_by_part = {
            str(part_name or "").strip(): dict(raw_entry or {})
            for part_name, raw_entry in dict(prepared_bridge_request.get("part_tracker") or {}).items()
            if str(part_name or "").strip() and isinstance(raw_entry, dict)
        }
        ordered_part_names: list[str] = []
        seen_observed_part_names: set[str] = set()
        for row in current_parts:
            part_name = str(row.get("part_name") or "").strip()
            if part_name and part_name not in seen_observed_part_names:
                seen_observed_part_names.add(part_name)
                ordered_part_names.append(part_name)
        for part_name in sorted(tracker_by_part):
            if part_name not in seen_observed_part_names:
                seen_observed_part_names.add(part_name)
                ordered_part_names.append(part_name)

        part_facts: list[dict[str, Any]] = []
        for part_name in ordered_part_names:
            current_part_row = dict(current_part_by_name.get(part_name) or {})
            tracker_entry = dict(tracker_by_part.get(part_name) or {})
            requirement_entry = dict((requirements_by_product.get(part_name) or [None])[0] or {})
            current_location = deepcopy(current_part_row.get("location"))
            holder_resource_jid = str(held_part_to_resource.get(part_name) or "").strip()
            if (
                not holder_resource_jid
                and isinstance(current_location, str)
                and current_location.endswith("_gripper")
            ):
                holder_resource_jid = current_location.rsplit("_gripper", 1)[0]
            part_facts.append(
                {
                    "part_name": part_name,
                    "current_state": deepcopy(current_part_row.get("state")),
                    "current_location": current_location,
                    "location_basis": deepcopy(current_part_row.get("location_basis")),
                    "observed_pose": deepcopy(current_part_row.get("observed_pose")),
                    "current_holder_resource_jid": holder_resource_jid or None,
                    "origin_location": deepcopy(tracker_entry.get("origin_resource_location")),
                    "goal_location": deepcopy(current_part_row.get("target_location")),
                    "goal_requirement_id": (
                        str(requirement_entry.get("requirement_id") or "").strip() or None
                    ),
                    "nominal_requirement_resource_jid": (
                        str(requirement_entry.get("resource_jid") or "").strip() or None
                    ),
                    "pending_nominal_task_ids": deepcopy(
                        pending_task_ids_by_part.get(part_name) or []
                    ),
                }
            )

        affected_part_names: list[str] = []
        seen_part_names: set[str] = set()
        for raw_entry in (fault_event.get("affected_parts") or []):
            if isinstance(raw_entry, dict):
                part_name = str(raw_entry.get("part_name") or "").strip()
            else:
                part_name = str(raw_entry or "").strip()
            if not part_name or part_name in seen_part_names:
                continue
            seen_part_names.add(part_name)
            affected_part_names.append(part_name)

        return {
            "fault_event": {
                "focused_resource_jid": deepcopy(fault_event.get("focused_resource_jid")),
                "blocked_at_task_id": deepcopy(fault_event.get("blocked_at_task_id")),
                "blocked_at_function": deepcopy(fault_event.get("blocked_at_function")),
                "resource_state": deepcopy(fault_event.get("resource_state")),
                "resource_state_after": {
                    k: v
                    for k, v in deepcopy(fault_event.get("resource_state_after") or {}).items()
                    if k not in ("execution_mode", "controller_ready")
                },
                "affected_part_names": affected_part_names,
            },
            "observed_runtime_state": {
                "resources": observed_resources,
            },
            "part_facts": part_facts,
            "loaded_safety_rules": [
                deepcopy(rule)
                for rule in (current_product_state.get("loaded_safety_rules") or [])
                if isinstance(rule, dict)
            ],
            "obligation_targets": [
                deepcopy(target)
                for target in (active_safety_diagnosis.get("obligation_targets") or [])
                if isinstance(target, dict)
            ],
            "relevant_assembly_requirements": relevant_assembly_requirements,
            "modeled_continuation_gap": {
                "pending_nominal_tasks": pending_tasks_detail,
                "unmet_continuation_conditions": unmet_continuation_conditions,
                "resume_ready": deepcopy(modeled_continuation_gap.get("resume_ready")),
            },
            "allowed_execution_surface": self._build_allowed_execution_surface(
                prepared_bridge_request
            ),
        }

    @staticmethod
    def _build_prompt_bridge_snapshot(
        resource_entry: dict[str, Any],
    ) -> dict[str, Any]:
        bridge_snapshot = dict(resource_entry.get("bridge_snapshot") or {})
        resource_core = dict(bridge_snapshot.get("resource_core") or {})
        resource_facets = dict(bridge_snapshot.get("resource_facets") or {})
        manipulator = dict(resource_facets.get("manipulator") or {})

        prompt_snapshot = {
            "resource_jid": str(
                resource_entry.get("resource_jid")
                or resource_core.get("resource_jid")
                or ""
            ).strip(),
            "resource_type": deepcopy(
                bridge_snapshot.get("resource_type")
                or resource_core.get("resource_type")
            ),
            "current_state": deepcopy(
                bridge_snapshot.get("current_state")
                if "current_state" in bridge_snapshot
                else resource_core.get("current_state")
            ),
            "current_location": deepcopy(
                bridge_snapshot.get("current_location")
                if "current_location" in bridge_snapshot
                else resource_core.get("current_location")
            ),
            "availability": deepcopy(
                bridge_snapshot.get("availability")
                if "availability" in bridge_snapshot
                else resource_core.get("availability")
            ),
            "active_work": deepcopy(
                bridge_snapshot.get("active_work")
                if "active_work" in bridge_snapshot
                else resource_core.get("active_work")
            ),
            "occupancy": deepcopy(
                bridge_snapshot.get("occupancy")
                or resource_core.get("occupancy")
                or {}
            ),
            "held_part": deepcopy(
                bridge_snapshot.get("held_part")
                if "held_part" in bridge_snapshot
                else manipulator.get("held_part")
            ),
            "gripper_state": deepcopy(
                bridge_snapshot.get("gripper_state")
                if "gripper_state" in bridge_snapshot
                else manipulator.get("gripper_state")
            ),
            "current_pose": deepcopy(
                bridge_snapshot.get("current_pose")
                if "current_pose" in bridge_snapshot
                else manipulator.get("current_pose")
            ),
            "current_pose_ref": deepcopy(
                bridge_snapshot.get("current_pose_ref")
                if "current_pose_ref" in bridge_snapshot
                else manipulator.get("current_pose_ref")
            ),
            "named_poses": deepcopy(
                bridge_snapshot.get("named_poses")
                if "named_poses" in bridge_snapshot
                else manipulator.get("named_poses")
            ),
            "workspace_bounds": deepcopy(
                dict(resource_entry.get("static_capabilities") or {}).get("workspace_bounds")
            ),
        }
        return {
            key: value
            for key, value in prompt_snapshot.items()
            if value not in (None, "", [], {})
        }

    @staticmethod
    def _build_prompt_allowed_primitives(
        resource_entry: dict[str, Any],
    ) -> list[dict[str, Any]]:
        resource_type = str(resource_entry.get("resource_type") or "resource").strip() or "resource"
        profile = get_resource_profile(resource_type)
        preview_output_map = dict(profile.preview_output_map or {})
        extract_output_map = dict(profile.extract_output_map or {})
        prompt_catalog: list[dict[str, Any]] = []
        for raw_item in (resource_entry.get("primitive_catalog") or []):
            if not isinstance(raw_item, dict):
                continue
            primitive_name = str(raw_item.get("name") or "").strip()
            if not primitive_name:
                continue
            prompt_item: dict[str, Any] = {"name": primitive_name}
            description = str(
                raw_item.get("description") or raw_item.get("semantic_summary") or ""
            ).strip()
            if description:
                prompt_item["description"] = description
            required_params = [
                str(item).strip()
                for item in (raw_item.get("required_params") or [])
                if str(item).strip()
            ]
            prompt_item["required_params"] = required_params
            primitive_kind = str(raw_item.get("primitive_kind") or "").strip()
            if primitive_kind:
                prompt_item["primitive_kind"] = primitive_kind
            output_schema = dict(raw_item.get("output_schema") or {})
            output_fields = [
                str(field).strip()
                for field in output_schema.keys()
                if str(field).strip()
            ]
            if output_fields:
                prompt_item["output_fields"] = output_fields
            hard_preconditions = deepcopy(raw_item.get("preconditions") or {})
            if hard_preconditions:
                prompt_item["hard_preconditions"] = hard_preconditions
            prompt_item["supports_store_as"] = (
                primitive_name in preview_output_map or primitive_name in extract_output_map
            )
            prompt_catalog.append(prompt_item)
        return prompt_catalog

    def _build_allowed_execution_surface(
        self,
        prepared_bridge_request: dict[str, Any],
    ) -> dict[str, Any]:
        context_summary = dict(prepared_bridge_request.get("context_summary") or {})
        fault_event = dict(context_summary.get("fault_event") or {})
        focused_resource_jid = str(fault_event.get("focused_resource_jid") or "").strip()
        current_product_state = dict(context_summary.get("current_product_state") or {})
        bridge_resources = dict(prepared_bridge_request.get("bridge_resources") or {})
        observed_resources = [
            dict(row)
            for row in (current_product_state.get("resources") or [])
            if isinstance(row, dict)
        ]
        observed_by_jid = {
            str(row.get("resource_jid") or "").strip(): row
            for row in observed_resources
            if str(row.get("resource_jid") or "").strip()
        }
        ordered_resource_jids = self._ordered_resource_jids(
            focused_resource_jid=focused_resource_jid,
            observed_resources=observed_resources,
            bridge_resources=bridge_resources,
        )

        prompt_resources: list[dict[str, Any]] = []
        for resource_jid in ordered_resource_jids:
            bridge_entry = dict(bridge_resources.get(resource_jid) or {})
            adapter_capabilities = dict(bridge_entry.get("bridge_adapter") or {})
            if not adapter_capabilities.get("supports_executable_bridge"):
                continue
            observed_row = dict(observed_by_jid.get(resource_jid) or {})
            resource_type = str(
                bridge_entry.get("resource_type")
                or dict((bridge_entry.get("bridge_snapshot") or {}).get("resource_core") or {}).get("resource_type")
                or "unknown"
            ).strip()
            prompt_resources.append(
                {
                    "resource_jid": resource_jid,
                    "resource_type": resource_type,
                    "role": deepcopy(
                        observed_row.get("role")
                        or ("focused" if resource_jid == focused_resource_jid else "supporting")
                    ),
                    "allowed_primitives": self._build_prompt_allowed_primitives(
                        bridge_entry
                    ),
                }
            )
        return {
            "focused_resource_jid": focused_resource_jid,
            "resources": prompt_resources,
        }

    def _build_single_shot_prompt_artifacts(
        self,
        prepared_bridge_request: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        return build_single_shot_prompt_artifacts(self, prepared_bridge_request)

    async def prepare_bridge_request(
        self,
        *,
        stuck_state: dict[str, Any],
        P_id: list[str],
        ra_jid: str,
        goal_state: str,
        tools_catalog: list[dict[str, Any]],
        part_tracker: dict[str, Any],
        obligation_targets: list[dict[str, Any]],
        bridge_feedback: str,
        resource_states: dict[str, dict[str, Any]],
        default_resource_state: str,
        part_states: dict[str, Any],
        part_locations: dict[str, Any],
        bridge_safety_context: dict[str, Any] | None = None,
        failure_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        input_bridge_safety_context = deepcopy(bridge_safety_context or {})
        input_failure_context = deepcopy(failure_context or {})
        primitive_catalog, bridge_snapshot, bridge_resources = await self._bridge_primitive_context(
            target_jid=ra_jid,
            resource_states=deepcopy(resource_states),
            default_resource_state=str(default_resource_state or "").strip(),
            part_states=deepcopy(part_states),
            part_locations=deepcopy(part_locations),
        )
        bridge_resources = self._trim_pending_tasks(
            bridge_resources=deepcopy(bridge_resources or {}),
            part_tracker=deepcopy(part_tracker),
        )
        session_id = f"prepare_{uuid4().hex[:8]}"
        reasoning_mode = self._resolve_bridge_reasoning_mode()

        prepared_bridge_request = {
            "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            "stuck_state": deepcopy(stuck_state),
            "P_id": deepcopy(P_id),
            "ra_jid": str(ra_jid or "").strip(),
            "goal_state": str(goal_state or "").strip(),
            "tools_catalog": deepcopy(tools_catalog),
            "primitive_catalog": deepcopy(primitive_catalog or []),
            "bridge_snapshot": deepcopy(bridge_snapshot or {}),
            "part_tracker": deepcopy(part_tracker),
            "obligation_targets": deepcopy(obligation_targets),
            "bridge_safety_context": deepcopy(input_bridge_safety_context),
            "loaded_safety_rules": self._bridge_loaded_safety_rules(),
            "failure_context_raw": deepcopy(input_failure_context),
            "failure_anchor": self._failure_anchor_from_payload(
                input_failure_context,
                fallback_resource_jid=str(ra_jid or "").strip(),
            ),
            "bridge_resources": deepcopy(bridge_resources),
            "grounding_context": {},
            "marked_reentry_context": {},
            "des_monitor": {
                "execution_state_x": {
                    "available": False,
                    "value": None,
                    "reason": "DES execution state x is not wired yet",
                },
                "monitor_state_q": {
                    "available": False,
                    "value": None,
                    "reason": "Safety monitor state q is not wired yet",
                },
            },
            "bridge_session": {
                "phase": "prepare_trace",
                "llm_enabled": False,
                "reasoning_mode": reasoning_mode,
                "session_id": session_id,
            },
        }
        requirement_inventory = self._bridge_requirement_inventory()
        prepared_bridge_request["requirement_nodes"] = deepcopy(
            requirement_inventory.get("requirement_nodes") or []
        )
        prepared_bridge_request["requirements_status"] = deepcopy(
            requirement_inventory.get("requirements_status") or {}
        )
        prepared_bridge_request["requirement_task_index"] = deepcopy(
            requirement_inventory.get("requirement_task_index") or {}
        )
        prepared_bridge_request["task_requirement_map"] = deepcopy(
            requirement_inventory.get("task_requirement_map") or {}
        )

        grounding_context = self._bridge_grounding_context(
            focused_resource_jid=str(ra_jid or "").strip(),
            bridge_snapshot=deepcopy(bridge_snapshot or {}),
            bridge_resources=deepcopy(bridge_resources),
            part_tracker=deepcopy(part_tracker),
            goal_state=str(goal_state or "").strip(),
            P_id=deepcopy(P_id),
            obligation_targets=deepcopy(obligation_targets),
        )
        prepared_bridge_request["grounding_context"] = deepcopy(grounding_context)

        normalized_part_tracker = deepcopy(part_tracker)
        normalized_parts_ctx = dict((grounding_context.get("parts") or {}))
        for part_name in sorted(
            {
                str(name).strip()
                for name in (list(normalized_part_tracker) + list(normalized_parts_ctx))
                if str(name).strip()
            }
        ):
            tracker_entry = dict(normalized_part_tracker.get(part_name) or {})
            location, location_basis = self._resolve_part_location(
                part_name=part_name,
                tracker_entry=tracker_entry,
                prepared_bridge_request=prepared_bridge_request,
            )
            tracker_entry["location_basis"] = location_basis
            if location is not None:
                tracker_entry["location"] = deepcopy(location)
                if tracker_entry.get("last_known_location") in (None, ""):
                    tracker_entry["last_known_location"] = deepcopy(location)
            normalized_part_tracker[part_name] = tracker_entry

            ctx_entry = dict(normalized_parts_ctx.get(part_name) or {})
            ctx_entry["location_basis"] = location_basis
            if location is not None:
                ctx_entry["location"] = deepcopy(location)
                if ctx_entry.get("last_known_location") in (None, ""):
                    ctx_entry["last_known_location"] = deepcopy(location)
            normalized_parts_ctx[part_name] = ctx_entry

        grounding_context["parts"] = normalized_parts_ctx
        prepared_bridge_request["part_tracker"] = normalized_part_tracker
        prepared_bridge_request["grounding_context"] = deepcopy(grounding_context)
        marked_reentry_context = self._bridge_marked_reentry_context(prepared_bridge_request)
        prepared_bridge_request["marked_reentry_context"] = deepcopy(marked_reentry_context)
        prepared_bridge_request["bridge_safety_context"] = self._derive_bridge_safety_constraints(
            prepared_bridge_request,
            marked_reentry_context=marked_reentry_context,
        )
        prepared_bridge_request["context_summary"] = self._build_context_summary(
            prepared_bridge_request,
            input_bridge_safety_context=input_bridge_safety_context,
        )
        prepared_bridge_request["llm_input"] = self._build_llm_input(
            prepared_bridge_request,
        )
        if reasoning_mode == "single_shot":
            single_shot_prompt_input, single_shot_prompt_text = (
                self._build_single_shot_prompt_artifacts(prepared_bridge_request)
            )
            prepared_bridge_request["single_shot_prompt_input"] = deepcopy(
                single_shot_prompt_input
            )
            prepared_bridge_request["single_shot_prompt_text"] = str(
                single_shot_prompt_text or ""
            )
        elif reasoning_mode == "multi_turn":
            prepared_bridge_request["multi_turn_session_seed"] = deepcopy(
                build_multi_turn_session_seed(prepared_bridge_request)
            )

        bridge_debug = {
            "builder": "prepare_trace",
            "status": "prepared",
            "reasoning_mode": reasoning_mode,
            "context_summary": deepcopy(prepared_bridge_request.get("context_summary") or {}),
        }
        if reasoning_mode == "single_shot":
            bridge_debug["single_shot_turn"] = {
                "reasoning_mode": reasoning_mode,
                "status": "prepared_for_llm",
                "prompt_input": deepcopy(
                    prepared_bridge_request.get("single_shot_prompt_input") or {}
                ),
                "prompt_text": str(
                    prepared_bridge_request.get("single_shot_prompt_text") or ""
                ),
            }
        elif reasoning_mode == "multi_turn":
            bridge_debug["multi_turn_session"] = deepcopy(
                prepared_bridge_request.get("multi_turn_session_seed") or {}
            )
        prepared_bridge_request["bridge_debug"] = bridge_debug
        if hasattr(self, "_set_last_bridge_debug"):
            self._set_last_bridge_debug(bridge_debug)
        self.logger.info(
            "[Bridge] Prepared context trace for %s.",
            str(ra_jid or "").strip(),
        )
        return prepared_bridge_request

    async def execute_prepared_bridge_request(
        self,
        prepared_bridge_request: dict[str, Any],
    ) -> dict[str, Any] | None:
        bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
        reasoning_mode = self._normalize_bridge_reasoning_mode(
            bridge_session.get("reasoning_mode") or self._resolve_bridge_reasoning_mode()
        )
        bridge_session["reasoning_mode"] = reasoning_mode
        prepared_bridge_request["bridge_session"] = bridge_session

        bridge_debug = deepcopy(prepared_bridge_request.get("bridge_debug") or {})
        bridge_debug.setdefault("builder", "prepare_trace")
        bridge_debug["source"] = "llm_bridge_v4"
        bridge_debug["reasoning_mode"] = reasoning_mode

        prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)
        if hasattr(self, "_set_last_bridge_debug"):
            self._set_last_bridge_debug(bridge_debug)

        if reasoning_mode == "single_shot":
            return await execute_single_shot_bridge(self, prepared_bridge_request)
        if reasoning_mode == "multi_turn":
            return await execute_multi_turn_bridge(self, prepared_bridge_request)

        bridge_debug["status"] = "unsupported_reasoning_mode"
        bridge_debug["message"] = (
            "The selected bridge reasoning_mode is not implemented in the active bridge yet."
        )
        prepared_bridge_request["bridge_debug"] = deepcopy(bridge_debug)
        if hasattr(self, "_set_last_bridge_debug"):
            self._set_last_bridge_debug(bridge_debug)
        return None
