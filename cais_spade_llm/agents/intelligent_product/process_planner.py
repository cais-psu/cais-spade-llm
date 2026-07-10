"""Process planner that turns requirements into tasks and plan automata."""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any

from cais_spade_llm.agents.intelligent_product.process_recovery_planner import (
    ProcessRecoveryPlanner,
)
from cais_spade_llm.agents.intelligent_product.replanner.des_search.resource_bidding import (
    compute_bid,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_recovery import (
    LlmRecoveryReplannerMixin,
)
from cais_spade_llm.product.order import (
    derive_ordering_constraints_from_safety,
    part_place_geometry,
    validate_product_order,
)
from cais_spade_llm.product.profile import ProductProfile
from cais_spade_llm.prompts import (
    build_requirement_parse_prompt,
    build_task_expansion_prompt,
)


class ProcessPlanner(LlmRecoveryReplannerMixin):
    """
    1. NL → structured requirement nodes   (build_high_level)
    2. requirement nodes → executable task DAG   (expand_requirements_to_tasks)
    """

    def __init__(self, product_agent, resource_agents: Iterable[Any]):
        """Initialize planner state with agent references and empty node graphs."""
        self.product_agent = product_agent
        self.resource_agents = list(resource_agents)
        self.logger = product_agent.logger
        self.nodes: list[dict[str, Any]] = []
        self.phase_to_node: dict[str, dict[str, Any]] = {}
        self.global_fsa: dict[str, Any] | None = None
        self.last_recovery_debug: dict[str, Any] = {}
        self.product_order_runtime: dict[str, Any] = {}
        self.last_product_order_artifact: dict[str, Any] = {}
        self.recovery_planner = ProcessRecoveryPlanner(self)
        self.recovery_planner.bind_methods()

    @staticmethod
    def _primitive_recovery_macro_tasks(proposal: dict[str, Any]) -> list[dict[str, Any]]:
        raw_tasks = proposal.get("macro_tasks")
        if isinstance(raw_tasks, list) and raw_tasks:
            return [dict(task) for task in raw_tasks if isinstance(task, dict)]
        if proposal.get("primitive_steps"):
            return [dict(proposal)]
        return []

    # ------------------------------------------------------------------ #
    # Product order → executable task nodes
    # ------------------------------------------------------------------ #
    def build_product_order_runtime_skeleton(
        self,
        product_order: dict[str, Any],
        *,
        safety_text: str = "",
    ) -> dict[str, Any]:
        """Validate product-order work without assigning resources or creating task nodes."""
        self.nodes.clear()
        self.phase_to_node.clear()
        self.global_fsa = None

        geometry = dict(getattr(self.product_agent, "product_geometry", {}) or {})
        validated = validate_product_order(product_order, geometry)
        order_payload = dict(validated.payload)
        selected_parts = list(validated.selected_parts)
        destination_location = str(order_payload.get("product") or "").strip()
        product_jid = str(
            order_payload.get("product_jid") or getattr(self.product_agent, "jid", "")
        ).strip()
        ordering_constraints = derive_ordering_constraints_from_safety(
            safety_text,
            selected_parts,
        )

        self.product_order_runtime = {
            "enabled": True,
            "product_order": deepcopy(order_payload),
            "selected_parts": selected_parts,
            "destination_location": destination_location,
            "product_jid": product_jid,
            "ordering_constraints": ordering_constraints,
            "pending_product_order_parts": list(selected_parts),
            "committed_product_order_parts": [],
            "completed_product_order_parts": [],
            "bid_evidence_by_part": {},
            "system_plan": [],
            "task_by_part_fn": {},
            "part_task_ids": {},
            "part_requirement_ids": {},
            "next_requirement_index": 1,
        }
        self._sync_product_order_runtime_artifact()
        self.logger.info(
            "[Planner] Built product-order runtime skeleton for %d pending part(s), %d task node(s).",
            len(selected_parts),
            0,
        )
        return deepcopy(self.last_product_order_artifact)

    def ready_product_order_parts(self) -> list[str]:
        """Return pending product-order parts whose product-order predecessors are completed."""
        runtime = self.product_order_runtime if isinstance(self.product_order_runtime, dict) else {}
        if not runtime.get("enabled"):
            return []
        self._refresh_completed_product_order_parts_from_nodes()
        pending = [
            str(part or "").strip()
            for part in (runtime.get("pending_product_order_parts") or [])
            if str(part or "").strip()
        ]
        completed = {
            str(part or "").strip()
            for part in (runtime.get("completed_product_order_parts") or [])
            if str(part or "").strip()
        }
        ready: list[str] = []
        for part_name in pending:
            if not self._product_order_unmet_place_before_parts(
                runtime,
                part_name,
                completed_parts=completed,
            ):
                ready.append(part_name)
        return ready

    def _product_order_unmet_place_before_parts(
        self,
        runtime: dict[str, Any],
        part_name: str,
        *,
        completed_parts: set[str] | None = None,
    ) -> list[str]:
        """Return unfinished place_before predecessor parts for one selected part."""
        part_name = str(part_name or "").strip()
        if not part_name:
            return []
        completed = (
            set(completed_parts)
            if completed_parts is not None
            else {
                str(part or "").strip()
                for part in (runtime.get("completed_product_order_parts") or [])
                if str(part or "").strip()
            }
        )
        unmet: list[str] = []
        for constraint in runtime.get("ordering_constraints") or []:
            if not isinstance(constraint, dict) or constraint.get("type") != "place_before":
                continue
            if str(constraint.get("after") or "").strip() != part_name:
                continue
            before = str(constraint.get("before") or "").strip()
            if before and before not in completed:
                unmet.append(before)
        return list(dict.fromkeys(unmet))

    def commit_product_order_part(
        self,
        part_name: str,
        *,
        unavailable_resource_jids: Iterable[str] | None = None,
        status: str = "pending_validation",
    ) -> dict[str, Any]:
        """Assign and materialize one ready product-order part chain."""
        runtime = self.product_order_runtime if isinstance(self.product_order_runtime, dict) else {}
        if not runtime.get("enabled"):
            raise ValueError("product-order runtime skeleton is not initialized")

        part_name = str(part_name or "").strip()
        if not part_name:
            raise ValueError("product-order part name is required")
        self._refresh_completed_product_order_parts_from_nodes()
        runtime = self.product_order_runtime if isinstance(self.product_order_runtime, dict) else {}
        if part_name not in set(runtime.get("pending_product_order_parts") or []):
            raise ValueError(f"product-order part {part_name} is not pending")
        if part_name in set(runtime.get("committed_product_order_parts") or []):
            raise ValueError(f"product-order part {part_name} is already committed")
        unmet_place_before = self._product_order_unmet_place_before_parts(runtime, part_name)
        if unmet_place_before:
            raise ValueError(
                f"product-order part {part_name} is not ready; unmet place_before predecessor(s): "
                f"{', '.join(unmet_place_before)}"
            )

        destination_location = str(runtime.get("destination_location") or "").strip()
        product_jid = str(runtime.get("product_jid") or "").strip()
        geometry = dict(getattr(self.product_agent, "product_geometry", {}) or {})
        resource_options = self._product_order_resource_options(destination_location)
        unavailable = {
            str(jid or "").strip()
            for jid in (unavailable_resource_jids or [])
            if str(jid or "").strip()
        }
        if unavailable:
            resource_options = [
                option
                for option in resource_options
                if str(option.get("resource_jid") or "").strip() not in unavailable
            ]
        if not resource_options:
            raise ValueError(
                f"no available product bid resources for part {part_name} to {destination_location}"
            )

        tools_catalog = list(getattr(self.product_agent, "tools_catalog", []) or [])
        if not tools_catalog:
            raise ValueError("product bidding requires tools_catalog")
        goal_state = self._product_order_goal_state(tools_catalog)
        if not goal_state:
            raise ValueError("product bidding could not resolve goal part state from tools_catalog")

        load_counts = self._product_order_runtime_load_counts(resource_options)
        selected_bid = self._select_product_order_bid(
            part_name=part_name,
            resource_options=resource_options,
            destination_location=destination_location,
            tools_catalog=tools_catalog,
            goal_state=goal_state,
            load_counts=load_counts,
        )
        resource_jid = str(selected_bid["resource_jid"])
        source_location = str(selected_bid["source_location"])
        selected_parts = [
            str(part or "").strip()
            for part in (runtime.get("selected_parts") or [])
            if str(part or "").strip()
        ]
        if part_name in selected_parts:
            requirement_index = selected_parts.index(part_name) + 1
        else:
            requirement_index = int(runtime.get("next_requirement_index") or 1)
            runtime["next_requirement_index"] = requirement_index + 1
        requirement_id = f"REQ_{requirement_index}"
        place_geometry = part_place_geometry(part_name, geometry)
        task_specs = self._task_specs_from_product_order_bid(
            bid_events=list(selected_bid["events"]),
            part_name=part_name,
            product_jid=product_jid,
            place_geometry=place_geometry,
        )

        task_by_part_fn = dict(runtime.get("task_by_part_fn") or {})
        new_nodes: list[dict[str, Any]] = []
        part_task_ids: list[str] = []
        for task_index, (function_name, params) in enumerate(task_specs, start=1):
            task_id = f"{requirement_id}_T{task_index}"
            params = dict(params)
            params["task_id"] = task_id
            predecessors = [part_task_ids[-1]] if part_task_ids else []
            node = {
                "id": task_id,
                "type": "task",
                "requirement_id": requirement_id,
                "function_name": function_name,
                "params": params,
                "resource_jid": resource_jid,
                "sequence_index": len(self.nodes) + len(new_nodes),
                "status": str(status or "pending_validation"),
                "predecessors": predecessors,
                "successors": [],
                "product_order_part": part_name,
                "product_order_file": str(
                    getattr(self.product_agent, "product_order_file", "") or ""
                ),
                "product_order_commit_status": str(status or "pending_validation"),
            }
            if predecessors:
                new_nodes[-1].setdefault("successors", []).append(task_id)
            new_nodes.append(node)
            part_task_ids.append(task_id)
            task_by_part_fn[(part_name, function_name)] = task_id

        for constraint in runtime.get("ordering_constraints") or []:
            if not isinstance(constraint, dict) or constraint.get("type") != "place_before":
                continue
            if str(constraint.get("after") or "").strip() != part_name:
                continue
            before_tid = task_by_part_fn.get(
                (str(constraint.get("before") or "").strip(), "place_insert")
            )
            after_tid = task_by_part_fn.get((part_name, "place_insert"))
            if not before_tid or not after_tid or before_tid == after_tid:
                continue
            before_node = next((node for node in self.nodes if node.get("id") == before_tid), None)
            after_node = next((node for node in new_nodes if node.get("id") == after_tid), None)
            if not before_node or not after_node:
                continue
            if before_tid not in after_node.setdefault("predecessors", []):
                after_node["predecessors"].append(before_tid)
            if after_tid not in before_node.setdefault("successors", []):
                before_node["successors"].append(after_tid)

        self.nodes.extend(new_nodes)
        self._normalize_same_resource_chains(self.nodes)

        runtime["pending_product_order_parts"] = [
            part
            for part in (runtime.get("pending_product_order_parts") or [])
            if str(part or "").strip() != part_name
        ]
        committed = list(runtime.get("committed_product_order_parts") or [])
        if part_name not in committed:
            committed.append(part_name)
        runtime["committed_product_order_parts"] = committed
        runtime["task_by_part_fn"] = task_by_part_fn
        part_task_id_map = dict(runtime.get("part_task_ids") or {})
        part_task_id_map[part_name] = list(part_task_ids)
        runtime["part_task_ids"] = part_task_id_map
        part_requirement_ids = dict(runtime.get("part_requirement_ids") or {})
        part_requirement_ids[part_name] = requirement_id
        runtime["part_requirement_ids"] = part_requirement_ids

        row = {
            "part": part_name,
            "source_location": source_location,
            "target_slot": part_name,
            "destination_location": destination_location,
            "resource_jid": resource_jid,
            "feasible_resources": [
                {
                    "resource_jid": str(opt["resource_jid"]),
                    "source_locations": list(opt["source_locations"]),
                    "destination_location": destination_location,
                }
                for opt in resource_options
            ],
            "product_bidding": {
                "selected_bid": {
                    "resource_jid": resource_jid,
                    "source_location": source_location,
                    "destination_location": destination_location,
                    "event_count": len(selected_bid["events"]),
                    "score": dict(selected_bid.get("score") or {}),
                    "events": deepcopy(selected_bid["events"]),
                },
                "candidates": deepcopy(selected_bid.get("candidates") or []),
            },
            "operations": list(part_task_ids),
            "commit_status": str(status or "pending_validation"),
        }
        system_plan = list(runtime.get("system_plan") or [])
        system_plan.append(row)
        runtime["system_plan"] = system_plan
        bid_evidence = dict(runtime.get("bid_evidence_by_part") or {})
        bid_evidence[part_name] = deepcopy(row["product_bidding"])
        runtime["bid_evidence_by_part"] = bid_evidence
        self.product_order_runtime = runtime
        self._sync_product_order_runtime_artifact()
        self.logger.info(
            "[Planner] Committed product-order part %s -> %s from %s (%d task node(s)).",
            part_name,
            resource_jid,
            source_location,
            len(part_task_ids),
        )
        return {
            "part": part_name,
            "resource_jid": resource_jid,
            "source_location": source_location,
            "task_ids": list(part_task_ids),
            "requirement_id": requirement_id,
            "product_bidding": deepcopy(row["product_bidding"]),
        }

    def recompile_committed_product_order_fsa(self) -> dict[str, Any] | None:
        """Compile the active-window FSA for currently executable product-order task nodes."""
        active_nodes = self.active_product_order_fsa_nodes()
        if not active_nodes:
            self.global_fsa = None
            self._sync_product_order_runtime_artifact()
            return None

        saved_nodes = self.nodes
        try:
            self.nodes = active_nodes
            self._normalize_same_resource_chains(self.nodes)
            fsa = self.compile_global_fsa()
        finally:
            self.nodes = saved_nodes

        self.global_fsa = fsa
        self._sync_product_order_runtime_artifact()
        return fsa

    def active_product_order_fsa_nodes(self) -> list[dict[str, Any]]:
        """Return active executable Product-order nodes, excluding completed history/backlog."""
        active_statuses = {
            "pending_validation",
            "pending",
            "dispatched",
            "accepted",
            "running",
            "blocked",
            "human_required",
        }
        active_ids: set[str] = set()
        active_nodes: list[dict[str, Any]] = []
        for node in self.nodes:
            if not isinstance(node, dict) or node.get("type") != "task":
                continue
            status = str(node.get("status") or "").strip().lower()
            if status in active_statuses:
                node_id = str(node.get("id") or "").strip()
                if node_id:
                    active_ids.add(node_id)
                    active_nodes.append(deepcopy(node))

        for node in active_nodes:
            node["predecessors"] = [
                pred
                for pred in (node.get("predecessors") or [])
                if str(pred or "").strip() in active_ids
            ]
            node["successors"] = [
                succ
                for succ in (node.get("successors") or [])
                if str(succ or "").strip() in active_ids
            ]
        return active_nodes

    def mark_product_order_part_completed(self, part_name: str) -> None:
        """Record that one committed product-order part chain completed."""
        runtime = self.product_order_runtime if isinstance(self.product_order_runtime, dict) else {}
        if not runtime.get("enabled"):
            return
        part_name = str(part_name or "").strip()
        if not part_name:
            return
        completed = list(runtime.get("completed_product_order_parts") or [])
        if part_name not in completed:
            completed.append(part_name)
        runtime["completed_product_order_parts"] = completed
        runtime["committed_product_order_parts"] = [
            part
            for part in (runtime.get("committed_product_order_parts") or [])
            if str(part or "").strip() != part_name
        ]
        self.product_order_runtime = runtime
        self._sync_product_order_runtime_artifact()

    def mark_product_order_commit_validated(self, part_names: Iterable[str]) -> list[str]:
        """Make validated committed Product-order nodes dispatchable."""
        runtime = self.product_order_runtime if isinstance(self.product_order_runtime, dict) else {}
        if not runtime.get("enabled"):
            return []
        targets = {
            str(part or "").strip() for part in (part_names or []) if str(part or "").strip()
        }
        validated_parts: list[str] = []
        for node in self.nodes:
            part_name = str(node.get("product_order_part") or "").strip()
            if part_name not in targets:
                continue
            if str(node.get("status") or "").strip() == "pending_validation":
                node["status"] = "pending"
            node["product_order_commit_status"] = "validated"
            if part_name not in validated_parts:
                validated_parts.append(part_name)
        for row in runtime.get("system_plan") or []:
            if str(row.get("part") or "").strip() in targets:
                row["commit_status"] = "validated"
        self.product_order_runtime = runtime
        self._sync_product_order_runtime_artifact()
        return validated_parts

    def rollback_product_order_committed_parts(self, part_names: Iterable[str]) -> list[str]:
        """Remove not-yet-dispatched committed product-order parts so they can be rebid."""
        runtime = self.product_order_runtime if isinstance(self.product_order_runtime, dict) else {}
        if not runtime.get("enabled"):
            return []
        targets = {
            str(part or "").strip() for part in (part_names or []) if str(part or "").strip()
        }
        if not targets:
            return []

        active_statuses = {
            "dispatched",
            "accepted",
            "running",
            "completed",
            "finished",
            "blocked",
            "human_required",
        }
        removable: set[str] = set()
        for part_name in sorted(targets):
            part_nodes = [
                node
                for node in self.nodes
                if str(node.get("product_order_part") or "").strip() == part_name
            ]
            if not part_nodes:
                continue
            if any(
                str(node.get("status") or "").strip().lower().startswith("failed")
                or str(node.get("status") or "").strip().lower() in active_statuses
                for node in part_nodes
            ):
                continue
            removable.add(part_name)

        if not removable:
            return []

        removed_ids = {
            str(node.get("id") or "").strip()
            for node in self.nodes
            if str(node.get("product_order_part") or "").strip() in removable
            and str(node.get("id") or "").strip()
        }
        self.nodes = [
            node
            for node in self.nodes
            if str(node.get("product_order_part") or "").strip() not in removable
        ]
        for node in self.nodes:
            node["predecessors"] = [
                pred
                for pred in (node.get("predecessors") or [])
                if str(pred or "").strip() not in removed_ids
            ]
            node["successors"] = [
                succ
                for succ in (node.get("successors") or [])
                if str(succ or "").strip() not in removed_ids
            ]

        pending = list(runtime.get("pending_product_order_parts") or [])
        for part_name in sorted(removable):
            if part_name not in pending:
                pending.append(part_name)
        selected_order = {
            str(part or "").strip(): index
            for index, part in enumerate(runtime.get("selected_parts") or [])
        }
        pending.sort(key=lambda part: selected_order.get(str(part or "").strip(), 10**9))
        runtime["pending_product_order_parts"] = pending
        runtime["committed_product_order_parts"] = [
            part
            for part in (runtime.get("committed_product_order_parts") or [])
            if str(part or "").strip() not in removable
        ]
        runtime["system_plan"] = [
            row
            for row in (runtime.get("system_plan") or [])
            if str(row.get("part") or "").strip() not in removable
        ]
        for key in ("part_task_ids", "part_requirement_ids", "bid_evidence_by_part"):
            mapping = dict(runtime.get(key) or {})
            for part_name in removable:
                mapping.pop(part_name, None)
            runtime[key] = mapping
        task_by_part_fn = dict(runtime.get("task_by_part_fn") or {})
        runtime["task_by_part_fn"] = {
            key: value
            for key, value in task_by_part_fn.items()
            if not (isinstance(key, tuple) and key and str(key[0]) in removable)
        }
        self.product_order_runtime = runtime
        self._sync_product_order_runtime_artifact()
        if self.nodes:
            self._normalize_same_resource_chains(self.nodes)
        else:
            self.global_fsa = None
        return sorted(removable)

    def _refresh_completed_product_order_parts_from_nodes(self) -> None:
        runtime = self.product_order_runtime if isinstance(self.product_order_runtime, dict) else {}
        if not runtime.get("enabled"):
            return
        completed = {
            str(part or "").strip()
            for part in (runtime.get("completed_product_order_parts") or [])
            if str(part or "").strip()
        }
        for part_name, task_ids in dict(runtime.get("part_task_ids") or {}).items():
            part = str(part_name or "").strip()
            if not part:
                continue
            part_nodes = [
                self._find_node(str(task_id or "").strip())
                for task_id in (task_ids or [])
                if str(task_id or "").strip()
            ]
            part_nodes = [node for node in part_nodes if isinstance(node, dict)]
            if part_nodes and all(
                str(node.get("status") or "").strip().lower() in {"completed", "finished"}
                for node in part_nodes
            ):
                completed.add(part)
        runtime["completed_product_order_parts"] = [
            part
            for part in (runtime.get("selected_parts") or [])
            if str(part or "").strip() in completed
        ]
        runtime["committed_product_order_parts"] = [
            part
            for part in (runtime.get("committed_product_order_parts") or [])
            if str(part or "").strip() not in completed
        ]
        self.product_order_runtime = runtime
        self._sync_product_order_runtime_artifact()

    def _product_order_runtime_load_counts(
        self,
        resource_options: list[dict[str, Any]],
    ) -> dict[str, int]:
        runtime = self.product_order_runtime if isinstance(self.product_order_runtime, dict) else {}
        load_counts: dict[str, int] = {
            str(opt.get("resource_jid") or "").strip(): 0
            for opt in resource_options
            if str(opt.get("resource_jid") or "").strip()
        }
        completed = {
            str(part or "").strip()
            for part in (runtime.get("completed_product_order_parts") or [])
            if str(part or "").strip()
        }
        for row in runtime.get("system_plan") or []:
            if not isinstance(row, dict):
                continue
            part_name = str(row.get("part") or "").strip()
            resource_jid = str(row.get("resource_jid") or "").strip()
            if resource_jid in load_counts and part_name not in completed:
                load_counts[resource_jid] += 1
        return load_counts

    def _sync_product_order_runtime_artifact(self) -> None:
        runtime = self.product_order_runtime if isinstance(self.product_order_runtime, dict) else {}
        if not runtime.get("enabled"):
            return
        self.last_product_order_artifact = {
            "product_order": deepcopy(runtime.get("product_order") or {}),
            "selected_parts": list(runtime.get("selected_parts") or []),
            "system_plan": deepcopy(runtime.get("system_plan") or []),
            "ordering_constraints": deepcopy(runtime.get("ordering_constraints") or []),
            "monitor_rules": [
                {
                    "type": "ordering_constraint",
                    "raw_text": str(item.get("raw_text", "")),
                    "before_event": str(item.get("before_event", "")),
                    "after_event": str(item.get("after_event", "")),
                }
                for item in (runtime.get("ordering_constraints") or [])
                if isinstance(item, dict)
            ],
            "derived_nodes": deepcopy(self.nodes),
            "active_window_nodes": self.active_product_order_fsa_nodes(),
            "pending_product_order_parts": list(runtime.get("pending_product_order_parts") or []),
            "committed_product_order_parts": list(
                runtime.get("committed_product_order_parts") or []
            ),
            "completed_product_order_parts": list(
                runtime.get("completed_product_order_parts") or []
            ),
            "bid_evidence_by_part": deepcopy(runtime.get("bid_evidence_by_part") or {}),
            "rolling_runtime_product_bidding": True,
        }

    def build_from_product_order(
        self,
        product_order: dict[str, Any],
        *,
        safety_text: str = "",
    ) -> dict[str, Any]:
        """Build deterministic executable task nodes from product-order JSON."""
        self.nodes.clear()
        self.phase_to_node.clear()
        self.global_fsa = None

        geometry = dict(getattr(self.product_agent, "product_geometry", {}) or {})
        validated = validate_product_order(product_order, geometry)
        order_payload = dict(validated.payload)
        selected_parts = list(validated.selected_parts)
        destination_location = str(order_payload.get("product") or "").strip()
        product_jid = str(
            order_payload.get("product_jid") or getattr(self.product_agent, "jid", "")
        ).strip()

        resource_options = self._product_order_resource_options(destination_location)
        if not resource_options:
            raise ValueError(
                f"no feasible resources can reach {destination_location} with a source staging area"
            )
        tools_catalog = list(getattr(self.product_agent, "tools_catalog", []) or [])
        if not tools_catalog:
            raise ValueError("product bidding requires tools_catalog")
        goal_state = self._product_order_goal_state(tools_catalog)
        if not goal_state:
            raise ValueError("product bidding could not resolve goal part state from tools_catalog")

        ordering_constraints = derive_ordering_constraints_from_safety(
            safety_text,
            selected_parts,
        )

        nodes: list[dict[str, Any]] = []
        system_plan: list[dict[str, Any]] = []
        task_by_part_fn: dict[tuple[str, str], str] = {}
        load_counts: dict[str, int] = {str(opt["resource_jid"]): 0 for opt in resource_options}

        for req_index, part_name in enumerate(selected_parts, start=1):
            feasible_resources = [
                {
                    "resource_jid": str(opt["resource_jid"]),
                    "source_locations": list(opt["source_locations"]),
                    "destination_location": destination_location,
                }
                for opt in resource_options
            ]
            selected_bid = self._select_product_order_bid(
                part_name=part_name,
                resource_options=resource_options,
                destination_location=destination_location,
                tools_catalog=tools_catalog,
                goal_state=goal_state,
                load_counts=load_counts,
            )
            resource_jid = str(selected_bid["resource_jid"])
            load_counts[resource_jid] = load_counts.get(resource_jid, 0) + 1
            source_location = str(selected_bid["source_location"])
            requirement_id = f"REQ_{req_index}"
            place_geometry = part_place_geometry(part_name, geometry)
            task_specs = self._task_specs_from_product_order_bid(
                bid_events=list(selected_bid["events"]),
                part_name=part_name,
                product_jid=product_jid,
                place_geometry=place_geometry,
            )

            part_task_ids: list[str] = []
            for task_index, (function_name, params) in enumerate(task_specs, start=1):
                task_id = f"{requirement_id}_T{task_index}"
                params = dict(params)
                params["task_id"] = task_id
                predecessors = [part_task_ids[-1]] if part_task_ids else []
                node = {
                    "id": task_id,
                    "type": "task",
                    "requirement_id": requirement_id,
                    "function_name": function_name,
                    "params": params,
                    "resource_jid": resource_jid,
                    "sequence_index": len(nodes),
                    "status": "pending",
                    "predecessors": predecessors,
                    "successors": [],
                    "product_order_part": part_name,
                    "product_order_file": str(
                        getattr(self.product_agent, "product_order_file", "") or ""
                    ),
                }
                if predecessors:
                    prev = nodes[-1]
                    prev.setdefault("successors", []).append(task_id)
                nodes.append(node)
                part_task_ids.append(task_id)
                task_by_part_fn[(part_name, function_name)] = task_id

            system_plan.append(
                {
                    "part": part_name,
                    "source_location": source_location,
                    "target_slot": part_name,
                    "destination_location": destination_location,
                    "resource_jid": resource_jid,
                    "feasible_resources": feasible_resources,
                    "product_bidding": {
                        "selected_bid": {
                            "resource_jid": resource_jid,
                            "source_location": source_location,
                            "destination_location": destination_location,
                            "event_count": len(selected_bid["events"]),
                            "score": dict(selected_bid.get("score") or {}),
                            "events": deepcopy(selected_bid["events"]),
                        },
                        "candidates": deepcopy(selected_bid.get("candidates") or []),
                    },
                    "operations": list(part_task_ids),
                }
            )

        for constraint in ordering_constraints:
            if constraint.get("type") != "place_before":
                continue
            before_tid = task_by_part_fn.get((str(constraint.get("before")), "place_insert"))
            after_tid = task_by_part_fn.get((str(constraint.get("after")), "place_insert"))
            if not before_tid or not after_tid or before_tid == after_tid:
                continue
            before_node = next((node for node in nodes if node.get("id") == before_tid), None)
            after_node = next((node for node in nodes if node.get("id") == after_tid), None)
            if not before_node or not after_node:
                continue
            if before_tid not in after_node.setdefault("predecessors", []):
                after_node["predecessors"].append(before_tid)
            if after_tid not in before_node.setdefault("successors", []):
                before_node["successors"].append(after_tid)

        self._normalize_same_resource_chains(nodes)
        self.nodes = nodes
        artifact = {
            "product_order": deepcopy(order_payload),
            "selected_parts": selected_parts,
            "system_plan": system_plan,
            "ordering_constraints": ordering_constraints,
            "monitor_rules": [
                {
                    "type": "ordering_constraint",
                    "raw_text": str(item.get("raw_text", "")),
                    "before_event": str(item.get("before_event", "")),
                    "after_event": str(item.get("after_event", "")),
                }
                for item in ordering_constraints
            ],
            "derived_nodes": deepcopy(nodes),
        }
        self.last_product_order_artifact = artifact
        self.logger.info(
            "[Planner] Built product-order plan for %d part(s), %d task node(s).",
            len(selected_parts),
            len(nodes),
        )
        return artifact

    @staticmethod
    def _product_order_goal_state(tools_catalog: list[dict[str, Any]]) -> str:
        completed_states: list[str] = []
        intermediate_states: set[str] = set()
        for tool in tools_catalog or []:
            if not isinstance(tool, dict):
                continue
            part_in_state = str(tool.get("part_in_state") or "").strip()
            if part_in_state:
                intermediate_states.add(part_in_state)
            completed_state = str(
                (tool.get("part_transition") or {}).get("completed", {}).get("state", "")
            ).strip()
            if completed_state:
                completed_states.append(completed_state)
        for state in completed_states:
            if state not in intermediate_states:
                return state
        return completed_states[-1] if completed_states else ""

    def _product_order_source_pose(
        self,
        *,
        part_name: str,
        source_location: str,
    ) -> tuple[dict[str, float], str]:
        robot_env = str(getattr(self.product_agent, "robot_env", "") or "").strip().lower()
        execution_mode = "physical" if robot_env == "real" else "simulation"
        geometry = ProductProfile.resolve_place_geometry(
            part_name=str(part_name or "").strip(),
            destination_location=str(source_location or "").strip(),
            product_geometry={},
            execution_mode=execution_mode,
        )

        target_origin_pose = dict(geometry.get("target_origin_pose") or {})
        if {"x", "y", "z"} <= set(target_origin_pose.keys()):
            x = self._product_order_float(target_origin_pose.get("x"))
            y = self._product_order_float(target_origin_pose.get("y"))
            z = self._product_order_float(target_origin_pose.get("z"))
            if x is not None and y is not None and z is not None:
                return {"x": x, "y": y, "z": z}, ""

        board_center = dict(geometry.get("board_center") or {})
        slot_xy = geometry.get("slot_xy")
        if isinstance(slot_xy, (list, tuple)) and len(slot_xy) >= 2:
            center_x = self._product_order_float(board_center.get("x"))
            center_y = self._product_order_float(board_center.get("y"))
            slot_x = self._product_order_float(slot_xy[0])
            slot_y = self._product_order_float(slot_xy[1])
            slot_floor_z = self._product_order_float(geometry.get("slot_floor_z_m"))
            part_height = self._product_order_float(geometry.get("part_height_m")) or 0.0
            if (
                center_x is not None
                and center_y is not None
                and slot_x is not None
                and slot_y is not None
                and slot_floor_z is not None
            ):
                return {
                    "x": center_x + slot_x,
                    "y": center_y + slot_y,
                    "z": slot_floor_z + (max(0.0, part_height) * 0.5),
                }, ""

        return {}, f"source pose unavailable for {part_name} at {source_location}"

    @staticmethod
    def _product_order_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _product_order_source_anchor_violations(
        self,
        *,
        source_location: str,
        source_pose: dict[str, Any],
        staging_areas: dict[str, Any],
    ) -> list[str]:
        pose_x = self._product_order_float(source_pose.get("x"))
        pose_y = self._product_order_float(source_pose.get("y"))
        if pose_x is None or pose_y is None or not isinstance(staging_areas, dict):
            return []

        distances: dict[str, float] = {}
        for location, metadata in staging_areas.items():
            token = str(location or "").strip()
            if not token or not isinstance(metadata, dict):
                continue
            anchor = dict(metadata.get("anchor_pose") or metadata.get("board_center") or {})
            anchor_x = self._product_order_float(anchor.get("x"))
            anchor_y = self._product_order_float(anchor.get("y"))
            if anchor_x is None or anchor_y is None:
                continue
            distances[token] = math.hypot(pose_x - anchor_x, pose_y - anchor_y)

        if len(distances) < 2:
            return []
        source_token = str(source_location or "").strip()
        source_distance = distances.get(source_token)
        if source_distance is None:
            return []
        nearest_distance = min(distances.values())
        tolerance = 1e-6
        if source_distance <= nearest_distance + tolerance:
            return []
        nearest_locations = [
            token
            for token, distance in distances.items()
            if distance <= nearest_distance + tolerance
        ]
        nearest_text = ",".join(sorted(nearest_locations))
        return [
            f"source_location={source_token} distance_m={source_distance:.4f} "
            f"is not nearest staging_area={nearest_text} distance_m={nearest_distance:.4f}"
        ]

    def _product_order_pose_in_gripper_reach(
        self,
        *,
        pose: dict[str, Any],
        gripper_reach: dict[str, Any],
    ) -> tuple[bool, list[str], dict[str, Any]]:
        if not isinstance(gripper_reach, dict) or not gripper_reach:
            return False, ["gripper_reach metadata unavailable"], {}

        frame = str(gripper_reach.get("frame") or "world").strip()
        if frame != "world":
            return False, [f"unsupported gripper_reach frame={frame}"], {"frame": frame}

        origin_pose = dict(gripper_reach.get("origin_pose") or {})
        origin_x = self._product_order_float(origin_pose.get("x"))
        origin_y = self._product_order_float(origin_pose.get("y"))
        max_xy_radius = self._product_order_float(gripper_reach.get("max_xy_radius_m"))
        pose_x = self._product_order_float(pose.get("x"))
        pose_y = self._product_order_float(pose.get("y"))
        if origin_x is None or origin_y is None:
            return False, ["gripper_reach.origin_pose x/y unavailable"], {}
        if max_xy_radius is None:
            return False, ["gripper_reach.max_xy_radius_m unavailable"], {}
        if pose_x is None or pose_y is None:
            return False, ["source pose x/y unavailable"], {}

        distance_xy = math.hypot(pose_x - origin_x, pose_y - origin_y)
        evidence = {
            "origin_pose": {"x": origin_x, "y": origin_y},
            "distance_xy_m": distance_xy,
            "max_xy_radius_m": max_xy_radius,
        }
        violations: list[str] = []
        tolerance = self._product_order_float(gripper_reach.get("tolerance_m")) or 0.0
        if distance_xy > max_xy_radius + tolerance:
            violations.append(
                f"distance_xy_m={distance_xy:.4f} > max_xy_radius_m={max_xy_radius:.4f}"
            )

        pose_z = self._product_order_float(pose.get("z"))
        z_min = self._product_order_float(gripper_reach.get("z_min_m"))
        z_max = self._product_order_float(gripper_reach.get("z_max_m"))
        if pose_z is not None:
            evidence["source_z_m"] = pose_z
            if z_min is not None and pose_z < z_min - tolerance:
                violations.append(f"z={pose_z:.4f} < z_min_m={z_min:.4f}")
            if z_max is not None and pose_z > z_max + tolerance:
                violations.append(f"z={pose_z:.4f} > z_max_m={z_max:.4f}")
        return len(violations) == 0, violations, evidence

    def _select_product_order_bid(
        self,
        *,
        part_name: str,
        resource_options: list[dict[str, Any]],
        destination_location: str,
        tools_catalog: list[dict[str, Any]],
        goal_state: str,
        load_counts: dict[str, int],
    ) -> dict[str, Any]:
        candidates: list[dict[str, Any]] = []
        complete_candidates: list[dict[str, Any]] = []

        for option in resource_options:
            resource_jid = str(option.get("resource_jid") or "").strip()
            source_locations: list[str] = []
            seen_sources: set[str] = set()
            for raw_source_location in option.get("source_locations") or []:
                source_location = str(raw_source_location or "").strip()
                if not source_location or source_location in seen_sources:
                    continue
                seen_sources.add(source_location)
                source_locations.append(source_location)
            for source_order, source_location in enumerate(source_locations):
                candidate = self._product_order_bid_candidate(
                    part_name=part_name,
                    resource_jid=resource_jid,
                    source_location=source_location,
                    source_order=source_order,
                    destination_location=destination_location,
                    tools_catalog=tools_catalog,
                    goal_state=goal_state,
                    load_count=load_counts.get(resource_jid, 0),
                    gripper_reach=dict(option.get("gripper_reach") or {}),
                    staging_areas=dict(option.get("staging_areas") or {}),
                )
                candidates.append(candidate)
                if candidate.get("status") == "complete":
                    complete_candidates.append(candidate)

        if not complete_candidates:
            only_reach_rejections = bool(candidates) and all(
                str(candidate.get("status") or "").strip() == "incomplete"
                and (
                    "source pose outside gripper_reach" in str(candidate.get("reason") or "")
                    or "source pose not nearest staging_area" in str(candidate.get("reason") or "")
                    or "gripper_reach metadata unavailable" in str(candidate.get("reason") or "")
                    or "gripper_reach." in str(candidate.get("reason") or "")
                    or "source pose unavailable" in str(candidate.get("reason") or "")
                )
                for candidate in candidates
            )
            if only_reach_rejections:
                raise ValueError(
                    f"no available product bid resources for part {part_name} to {destination_location}"
                )
            raise ValueError(
                f"no complete product bid for part {part_name} to {destination_location}"
            )

        selected = min(
            complete_candidates,
            key=lambda item: (
                int(item.get("event_count", 0)),
                int(item.get("load_count", 0)),
                str(item.get("resource_jid") or ""),
                int(item.get("source_order", 0)),
                str(item.get("source_location") or ""),
            ),
        )
        for candidate in candidates:
            same_candidate = str(candidate.get("resource_jid") or "") == str(
                selected.get("resource_jid") or ""
            ) and str(candidate.get("source_location") or "") == str(
                selected.get("source_location") or ""
            )
            if same_candidate:
                candidate["status"] = "selected"
            elif candidate.get("status") == "complete":
                candidate["status"] = "rejected"
        selected = dict(selected)
        selected["status"] = "selected"
        selected["candidates"] = candidates
        return selected

    def _product_order_bid_candidate(
        self,
        *,
        part_name: str,
        resource_jid: str,
        source_location: str,
        source_order: int,
        destination_location: str,
        tools_catalog: list[dict[str, Any]],
        goal_state: str,
        load_count: int,
        gripper_reach: dict[str, Any] | None = None,
        staging_areas: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        candidate: dict[str, Any] = {
            "resource_jid": resource_jid,
            "source_location": source_location,
            "source_order": int(source_order),
            "destination_location": destination_location,
            "load_count": int(load_count),
        }
        source_pose, source_pose_error = self._product_order_source_pose(
            part_name=part_name,
            source_location=source_location,
        )
        if source_pose:
            candidate["source_pose"] = deepcopy(source_pose)
        if source_pose_error:
            candidate["source_pose_error"] = source_pose_error

        if not source_pose:
            candidate.update(
                {
                    "status": "incomplete",
                    "reason": source_pose_error or "source pose unavailable",
                }
            )
            return candidate

        source_anchor_violations = self._product_order_source_anchor_violations(
            source_location=source_location,
            source_pose=source_pose,
            staging_areas=dict(staging_areas or {}),
        )
        if source_anchor_violations:
            candidate.update(
                {
                    "status": "incomplete",
                    "reason": "source pose not nearest staging_area: "
                    + "; ".join(source_anchor_violations),
                    "source_anchor_violations": source_anchor_violations,
                }
            )
            return candidate

        reach = dict(gripper_reach or {})
        candidate["gripper_reach"] = deepcopy(reach)
        reachable, violations, evidence = self._product_order_pose_in_gripper_reach(
            pose=source_pose,
            gripper_reach=reach,
        )
        if evidence:
            candidate["gripper_reach_evidence"] = deepcopy(evidence)
        if not reachable:
            candidate.update(
                {
                    "status": "incomplete",
                    "reason": "source pose outside gripper_reach: " + "; ".join(violations),
                    "gripper_reach_violations": violations,
                }
            )
            return candidate

        x_c = {
            "resource_state": "idle",
            "current_part": None,
            "current_location": None,
            "part_states": {part_name: "ready"},
            "part_locations": {part_name: source_location},
        }
        reachability = [
            location
            for location in (source_location, destination_location)
            if str(location or "").strip()
        ]
        staging_areas = {source_location: {}}
        bid = compute_bid(
            x_c=x_c,
            P_id=[part_name],
            goal_state=goal_state,
            tools=tools_catalog,
            reachability=reachability,
            staging_areas=staging_areas,
            resource_jid=resource_jid,
            goal_resource_state="idle",
        )
        if not bid:
            candidate.update({"status": "no_bid", "reason": "compute_bid returned no bid"})
            return candidate

        events = [dict(event) for event in (bid.str_e or []) if isinstance(event, dict)]
        states = [dict(state) for state in (bid.str_x or []) if isinstance(state, dict)]
        candidate.update(
            {
                "complete": bool(bid.complete),
                "event_count": len(events),
                "events": deepcopy(events),
            }
        )
        valid, reason = self._product_order_bid_is_complete(
            bid_complete=bool(bid.complete),
            events=events,
            states=states,
            part_name=part_name,
            source_location=source_location,
            destination_location=destination_location,
            goal_state=goal_state,
        )
        if not valid:
            candidate.update({"status": "incomplete", "reason": reason})
            return candidate

        candidate.update(
            {
                "status": "complete",
                "score": {
                    "event_count": len(events),
                    "load_count": int(load_count),
                    "resource_jid": resource_jid,
                    "source_order": int(source_order),
                    "source_location": source_location,
                },
            }
        )
        return candidate

    @staticmethod
    def _product_order_bid_is_complete(
        *,
        bid_complete: bool,
        events: list[dict[str, Any]],
        states: list[dict[str, Any]],
        part_name: str,
        source_location: str,
        destination_location: str,
        goal_state: str,
    ) -> tuple[bool, str]:
        if not bid_complete:
            return False, "bid did not report complete"
        if not events:
            return False, "bid has no events"
        if str(events[-1].get("function_name") or "").strip() != "move_home":
            return False, "bid does not return resource to idle through move_home"
        first_params = dict(events[0].get("params") or {})
        if str(first_params.get("origin_resource_location") or "").strip() != source_location:
            return False, "bid does not start from evaluated source_location"
        has_destination_approach = any(
            str(event.get("function_name") or "").strip() == "place_approach"
            and str((event.get("params") or {}).get("destination_location") or "").strip()
            == destination_location
            for event in events
        )
        if not has_destination_approach:
            return False, "bid does not approach destination_location"
        has_destination_insert = any(
            str(event.get("function_name") or "").strip() == "place_insert"
            and str((event.get("params") or {}).get("destination_location") or "").strip()
            == destination_location
            for event in events
        )
        if not has_destination_insert:
            return False, "bid does not insert at destination_location"
        final_state = states[-1] if states else {}
        part_states = dict(final_state.get("part_states") or {})
        part_locations = dict(final_state.get("part_locations") or {})
        if str(final_state.get("resource_state") or "").strip() != "idle":
            return False, "bid final resource_state is not idle"
        if str(part_states.get(part_name) or "").strip() != goal_state:
            return False, "bid final part state is not goal_state"
        if str(part_locations.get(part_name) or "").strip() != destination_location:
            return False, "bid final part location is not destination_location"
        return True, ""

    @staticmethod
    def _task_specs_from_product_order_bid(
        *,
        bid_events: list[dict[str, Any]],
        part_name: str,
        product_jid: str,
        place_geometry: dict[str, Any],
    ) -> list[tuple[str, dict[str, Any]]]:
        task_specs: list[tuple[str, dict[str, Any]]] = []
        for event in bid_events:
            function_name = str(event.get("function_name") or "").strip()
            params = dict(event.get("params") or {})
            if function_name != "move_home":
                params.setdefault("part_name", part_name)
                params["product_geometry"] = deepcopy(place_geometry)
            params["product_jid"] = product_jid
            if function_name == "pick_approach":
                params.setdefault("speed", None)
            elif function_name == "pick_grasp":
                params.setdefault("gripper", None)
            elif function_name == "place_approach":
                params.setdefault("speed", None)
            elif function_name == "place_insert":
                params.setdefault("orientation", None)
            task_specs.append((function_name, params))
        return task_specs

    def _product_order_resource_options(self, destination_location: str) -> list[dict[str, Any]]:
        options: list[dict[str, Any]] = []
        destination = str(destination_location or "").strip()
        for resource in self.resource_agents:
            resource_jid = str(getattr(resource, "jid", "") or "").strip()
            caps = getattr(resource, "static_capabilities", {}) or {}
            if not isinstance(caps, dict):
                continue
            reachability = [
                str(item) for item in (caps.get("reachability") or []) if str(item or "").strip()
            ]
            if destination and destination not in reachability:
                continue
            staging = caps.get("staging_areas") or {}
            staging_locations = (
                [str(name) for name in staging.keys() if str(name or "").strip()]
                if isinstance(staging, dict)
                else []
            )
            staging_set = set(staging_locations)
            source_locations = [
                item
                for item in reachability
                if item and item in staging_set and item != destination
            ]
            for location in staging_locations:
                if location not in source_locations:
                    source_locations.append(location)
            if not source_locations:
                source_locations = [item for item in reachability if item and item != destination]
            if not source_locations:
                continue
            options.append(
                {
                    "resource_jid": resource_jid,
                    "source_locations": source_locations,
                    "destination_location": destination,
                    "reachability": reachability,
                    "staging_areas": dict(staging) if isinstance(staging, dict) else {},
                    "gripper_reach": dict(caps.get("gripper_reach") or {}),
                    "workspace_bounds": dict(caps.get("workspace_bounds") or {}),
                }
            )
        return sorted(options, key=lambda item: str(item.get("resource_jid", "")))

    # ------------------------------------------------------------------ #
    # 1. NL → High-level requirements
    # ------------------------------------------------------------------ #
    async def build_high_level(
        self,
        requirement_text: str,
        *,
        refinement_feedback: str = "",
        previous_preview_requirements: list[dict[str, Any]] | None = None,
    ) -> str:
        """
        Parse natural-language manufacturing requirements into internal requirement nodes.

        No normalization is applied:
        - phase, process_type, product, context are kept exactly as returned by the LLM.
        """
        # Reset state
        self.nodes.clear()
        self.phase_to_node.clear()

        try:
            structured = await self._llm_parse_requirements(
                requirement_text,
                refinement_feedback=refinement_feedback,
                previous_preview_requirements=previous_preview_requirements,
            )
        except Exception as exc:
            self.logger.exception("[Planner] LLM requirement parsing failed: %s", exc)
            structured = []

        for idx, req in enumerate(structured, start=1):
            node_id = f"REQ_{idx}"

            # Use values exactly as returned by _llm_parse_requirements
            raw_text = req.get("raw_text", "")
            phase = req.get("phase")  # e.g. "ASSEMBLY", "printing", or None
            process_type = req.get("process_type")  # e.g. "PICK_PLACE", "FDM_PRINT", or None
            product = req.get("product")  # e.g. "SG", "MCP", or None
            context = (
                req.get("context") or {}
            )  # e.g. {"origin": "prusa-mk4-2", "destination": "assembly board"}

            node = {
                "id": node_id,
                "type": "requirement",
                "raw_text": raw_text,
                "phase": phase,
                "process_type": process_type,
                "product": product,
                "context": context,
            }

            self.nodes.append(node)

        msg = f"[Planner] Parsed {len(structured)} requirement(s) via LLM."
        self.logger.info(msg)
        return msg

    async def _llm_parse_requirements(
        self,
        requirement_text: str,
        *,
        refinement_feedback: str = "",
        previous_preview_requirements: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Call the LLM to parse requirements into a structured list."""
        tools_catalog = getattr(self.product_agent, "tools_catalog", [])

        prompt = build_requirement_parse_prompt(
            requirement_text,
            tools_catalog,
            refinement_feedback=refinement_feedback,
            previous_preview_requirements=previous_preview_requirements,
        )

        raw = await self.product_agent.ask_llm(
            prompt=prompt,
            with_functions=False,
            temperature=0.0,
        )

        if isinstance(raw, dict):
            self.logger.error("[Planner] ask_llm returned a dict, expected JSON string.")
            raise RuntimeError("ask_llm returned dict; expected JSON string.")

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            self.logger.error("[Planner] LLM did not return valid JSON: %s\nRaw: %s", exc, raw)
            raise

        reqs = parsed.get("requirements", [])
        cleaned: list[dict[str, Any]] = []

        for r in reqs:
            if not isinstance(r, dict):
                continue
            ctx = r.get("context") or {}
            if not isinstance(ctx, dict):
                ctx = {}
            cleaned.append(
                {
                    "raw_text": r.get("raw_text", ""),
                    "phase": r.get("phase"),
                    "process_type": r.get("process_type"),
                    "product": r.get("product"),
                    "context": ctx,
                }
            )

        return cleaned

    # ------------------------------------------------------------------ #
    # 2. REQUIREMENT → LLM TASK EXPANSION (replaces hard-coded version)
    # ------------------------------------------------------------------ #
    async def expand_requirements_to_tasks(
        self,
        safety_text: str = "",
        *,
        refinement_feedback: str = "",
        previous_preview_requirements: list[dict[str, Any]] | None = None,
        previous_preview_tasks: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Replace requirement nodes with task nodes produced by LLM.

        The LLM returns a DIRECTED ACYCLIC GRAPH (DAG) of tasks where
        dependencies are expressed via `predecessors` and `successors`.
        """
        req_nodes = [n for n in self.nodes if n.get("type") == "requirement"]
        if not req_nodes:
            self.logger.warning("[Planner] No requirement nodes to expand.")
            self.nodes = []
            return

        # Prepare payload for LLM
        req_payload = [
            {
                "id": n["id"],
                "product": n.get("product"),
                "context": n.get("context"),
                "raw_text": n.get("raw_text"),
            }
            for n in req_nodes
        ]

        tools_catalog = getattr(self.product_agent, "tools_catalog", [])

        # capabilities overview from ProductAgent
        caps_overview = ""
        if hasattr(self.product_agent, "_static_caps_overview"):
            try:
                caps_overview = self.product_agent._static_caps_overview()
            except Exception:
                caps_overview = ""

        resource_infos = [
            {
                "jid": str(getattr(ra, "jid", "")),
                "static_capabilities": getattr(ra, "static_capabilities", {}),
            }
            for ra in self.resource_agents
        ]

        # Build LLM prompt (from prompts.py)
        prompt = build_task_expansion_prompt(
            requirements=req_payload,
            tools_catalog=tools_catalog,
            resource_infos=resource_infos,
            caps_overview=caps_overview,
            safety_text=safety_text,
            refinement_feedback=refinement_feedback,
            previous_preview_requirements=previous_preview_requirements,
            previous_preview_tasks=previous_preview_tasks,
        )

        raw = await self.product_agent.ask_llm(
            prompt=prompt,
            with_functions=False,
            temperature=0.0,
        )

        parsed = json.loads(raw)
        task_specs = parsed.get("tasks", [])

        new_nodes: list[dict[str, Any]] = []

        # Create task nodes from LLM output
        for t in task_specs:
            params = t.get("params") or {}
            # Ensure product_jid is always present
            params["product_jid"] = str(self.product_agent.jid)

            node = {
                "id": t.get("id"),
                "type": "task",
                "requirement_id": t.get("requirement_id"),
                "function_name": t.get("function_name"),
                "params": params,
                "resource_jid": t.get("resource_jid"),
                "sequence_index": t.get("sequence_index", 0),
                "status": "pending",
                # Preserve graph structure from LLM if provided
                "predecessors": t.get("predecessors", []),
                "successors": t.get("successors", []),
            }
            new_nodes.append(node)

        # OPTIONAL: fill in trivial per-requirement chains
        # for tasks that have no predecessors/successors at all.
        # This keeps things backwards-compatible if the LLM omits edges.
        by_req: dict[str, list[dict[str, Any]]] = {}
        for n in new_nodes:
            rid = n.get("requirement_id")
            by_req.setdefault(rid, []).append(n)

        for rid, seq in by_req.items():
            # Check if ALL tasks under this requirement have completely empty edges
            all_edges_empty = all(
                not n.get("predecessors") and not n.get("successors") for n in seq
            )
            if not all_edges_empty:
                # LLM already defined some graph structure for this requirement;
                # do NOT overwrite it.
                continue

            # Otherwise, fall back to a simple linear chain by sequence_index
            seq.sort(key=lambda n: n.get("sequence_index", 0))
            for prev, nxt in zip(seq, seq[1:]):
                prev["successors"].append(nxt["id"])
                nxt["predecessors"].append(prev["id"])

        self._normalize_same_resource_chains(new_nodes)
        self.nodes = new_nodes
        self.logger.info(
            "[Planner] Expanded to %d LLM-generated task node(s).",
            len(self.nodes),
        )

    # ------------------------------------------------------------------ #
    # 3. REPLAN WHEN SAFETY VIOLATION OR ONLINE FAILURE OCCURS
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Prompt helpers
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Debug helpers
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Scheduling helpers
    # ------------------------------------------------------------------ #
    def _extract_conflict_task_ids(self, violations: list[dict], *, source: str) -> set[str]:
        """Normalize offline/online feedback into a conflict task ID set."""
        conflict_task_ids: set[str] = set()

        def _add_ids(values: Any) -> None:
            if isinstance(values, (list, tuple, set)):
                for v in values:
                    if v:
                        conflict_task_ids.add(str(v))
            elif isinstance(values, str) and values:
                conflict_task_ids.add(values)

        for v in violations:
            if not isinstance(v, dict):
                continue

            if source == "offline":
                _add_ids(v.get("witness_trace"))
                _add_ids(v.get("witness_task_ids"))
                _add_ids(v.get("conflict_task_ids"))

                witness_transitions = v.get("witness_transitions") or []
                for t in witness_transitions:
                    tid = t.get("task_id") if isinstance(t, dict) else None
                    if tid:
                        conflict_task_ids.add(str(tid))

                rel_tasks = v.get("affected_tasks", [])
                if isinstance(rel_tasks, list):
                    for rt in rel_tasks:
                        if isinstance(rt, dict) and "id" in rt:
                            conflict_task_ids.add(str(rt["id"]))
            else:
                _add_ids(v.get("failed_task_id"))
                _add_ids(v.get("task_id"))
                _add_ids(v.get("blocked_task_ids"))
                _add_ids(v.get("unreachable_task_ids"))
                _add_ids(v.get("impact_task_ids"))
                _add_ids(v.get("affected_task_ids"))

                rel_tasks = v.get("affected_tasks", [])
                if isinstance(rel_tasks, list):
                    for rt in rel_tasks:
                        if isinstance(rt, dict) and "id" in rt:
                            conflict_task_ids.add(str(rt["id"]))
                dep_tasks = v.get("dependent_tasks", [])
                if isinstance(dep_tasks, list):
                    for rt in dep_tasks:
                        if isinstance(rt, dict) and "id" in rt:
                            conflict_task_ids.add(str(rt["id"]))

        return conflict_task_ids

    def _ensure_graph_consistency(self, nodes: list[dict[str, Any]] | None = None) -> None:
        """
        Helper to ensure that if A lists B as a predecessor,
        B lists A as a successor (and vice versa).
        This fixes 'one-sided' edits from the LLM.
        """
        working_nodes = nodes if nodes is not None else self.nodes
        node_map = {n["id"]: n for n in working_nodes}

        for nid, node in node_map.items():
            # 1. Sync Predecessors -> Successors
            # If 'node' thinks 'pid' is a predecessor, make sure 'pid' knows 'node' is a successor.
            preds = list(dict.fromkeys(node.get("predecessors", []) or []))
            node["predecessors"] = preds
            for pid in preds:
                if pid in node_map:
                    p_node = node_map[pid]
                    if nid not in p_node.get("successors", []):
                        p_node.setdefault("successors", []).append(nid)

            # 2. Sync Successors -> Predecessors
            # If 'node' thinks 'sid' is a successor, make sure 'sid' knows 'node' is a predecessor.
            succs = list(dict.fromkeys(node.get("successors", []) or []))
            node["successors"] = succs
            for sid in succs:
                if sid in node_map:
                    s_node = node_map[sid]
                    if nid not in s_node.get("predecessors", []):
                        s_node.setdefault("predecessors", []).append(nid)

    @staticmethod
    def _requirement_order_key(requirement_id: Any) -> tuple[int, str]:
        token = str(requirement_id or "").strip()
        if token.startswith("REQ_"):
            suffix = token[4:]
            if suffix.isdigit():
                return (int(suffix), token)
        return (10**9, token)

    @staticmethod
    def _task_sequence_index_key(value: Any) -> int:
        if value is None:
            return 10**9
        try:
            return int(value)
        except (TypeError, ValueError):
            return 10**9

    def _task_order_key_within_requirement(self, node: dict[str, Any]) -> tuple[int, str]:
        return (
            self._task_sequence_index_key(node.get("sequence_index")),
            str(node.get("id", "")),
        )

    def _ordered_requirement_resource_tasks(
        self,
        tasks: list[dict[str, Any]],
        node_map: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if len(tasks) <= 1:
            return list(tasks)

        task_ids = {str(task.get("id", "")) for task in tasks if task.get("id")}
        local_successors: dict[str, list[str]] = {tid: [] for tid in task_ids}
        indegree: dict[str, int] = {tid: 0 for tid in task_ids}

        for task in tasks:
            tid = str(task.get("id", ""))
            for sid in task.get("successors", []) or []:
                if sid not in task_ids or sid == tid:
                    continue
                if sid not in local_successors[tid]:
                    local_successors[tid].append(sid)
                    indegree[sid] += 1
            for pid in task.get("predecessors", []) or []:
                if pid not in task_ids or pid == tid:
                    continue
                if tid not in local_successors[pid]:
                    local_successors[pid].append(tid)
                    indegree[tid] += 1

        ready = sorted(
            [tid for tid, degree in indegree.items() if degree == 0],
            key=lambda tid: self._task_order_key_within_requirement(node_map[tid]),
        )
        ordered_ids: list[str] = []

        while ready:
            tid = ready.pop(0)
            ordered_ids.append(tid)
            for sid in sorted(
                local_successors.get(tid, []),
                key=lambda task_id: self._task_order_key_within_requirement(node_map[task_id]),
            ):
                indegree[sid] -= 1
                if indegree[sid] == 0:
                    ready.append(sid)
                    ready.sort(
                        key=lambda task_id: self._task_order_key_within_requirement(
                            node_map[task_id]
                        )
                    )

        if len(ordered_ids) != len(task_ids):
            return sorted(tasks, key=self._task_order_key_within_requirement)
        return [node_map[tid] for tid in ordered_ids]

    def _normalize_same_resource_chains(
        self,
        nodes: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        Rewrite tasks sharing the same resource_jid into one strict local chain.

        Cross-resource predecessors/successors are preserved exactly as-is.
        """
        working_nodes = nodes if nodes is not None else self.nodes
        task_nodes = [n for n in working_nodes if n.get("type") == "task"]
        if not task_nodes:
            return

        self._ensure_graph_consistency(working_nodes)

        node_map = {
            str(node.get("id", "")): node for node in task_nodes if str(node.get("id", "")).strip()
        }

        if not node_map:
            return

        resource_to_requirements: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
        for task in task_nodes:
            resource_jid = str(task.get("resource_jid", "")).strip()
            if not resource_jid:
                continue
            requirement_id = str(task.get("requirement_id", "")).strip()
            requirement_tasks = resource_to_requirements[resource_jid].setdefault(
                requirement_id, []
            )
            requirement_tasks.append(task)

        for resource_jid, requirement_groups in resource_to_requirements.items():
            ordered_requirements = sorted(
                requirement_groups.items(),
                key=lambda item: self._requirement_order_key(item[0]),
            )
            local_chain: list[dict[str, Any]] = []
            local_task_ids: set[str] = set()

            for _requirement_id, requirement_tasks in ordered_requirements:
                ordered_tasks = self._ordered_requirement_resource_tasks(
                    requirement_tasks,
                    node_map,
                )
                local_chain.extend(ordered_tasks)
                local_task_ids.update(
                    str(task.get("id", "")) for task in ordered_tasks if task.get("id")
                )

            if not local_chain:
                continue

            for task in local_chain:
                tid = str(task.get("id", ""))
                preserved_predecessors = [
                    pid
                    for pid in task.get("predecessors", []) or []
                    if pid not in local_task_ids
                    or node_map.get(pid, {}).get("resource_jid") != resource_jid
                ]
                preserved_successors = [
                    sid
                    for sid in task.get("successors", []) or []
                    if sid not in local_task_ids
                    or node_map.get(sid, {}).get("resource_jid") != resource_jid
                ]
                task["predecessors"] = list(dict.fromkeys(preserved_predecessors))
                task["successors"] = list(dict.fromkeys(preserved_successors))

            for index, task in enumerate(local_chain):
                if index > 0:
                    prev_id = str(local_chain[index - 1].get("id", ""))
                    task.setdefault("predecessors", []).append(prev_id)
                if index + 1 < len(local_chain):
                    next_id = str(local_chain[index + 1].get("id", ""))
                    task.setdefault("successors", []).append(next_id)
                task["predecessors"] = list(dict.fromkeys(task.get("predecessors", []) or []))
                task["successors"] = list(dict.fromkeys(task.get("successors", []) or []))
                task["sequence_index"] = index

        self._ensure_graph_consistency(working_nodes)

    def _validate_task_graph(self, nodes: list[dict[str, Any]] | None = None) -> None:
        """Reject invalid task graphs before FSA compilation or execution."""
        working_nodes = nodes if nodes is not None else self.nodes
        tasks = [n for n in working_nodes if n.get("type") == "task"]
        node_map = {n["id"]: n for n in tasks}
        successors: dict[str, list[str]] = {nid: [] for nid in node_map}
        indegree: dict[str, int] = {nid: 0 for nid in node_map}

        for nid, node in node_map.items():
            for pid in node.get("predecessors", []) or []:
                if pid not in node_map:
                    raise ValueError(
                        f"Task graph references unknown predecessor '{pid}' from task '{nid}'."
                    )
                if pid == nid:
                    raise ValueError(f"Task graph contains a self-dependency on '{nid}'.")
                if nid not in successors[pid]:
                    successors[pid].append(nid)
                    indegree[nid] += 1

        q = deque(sorted(nid for nid, degree in indegree.items() if degree == 0))
        visited: list[str] = []

        while q:
            nid = q.popleft()
            visited.append(nid)
            for sid in successors[nid]:
                indegree[sid] -= 1
                if indegree[sid] == 0:
                    q.append(sid)

        if len(visited) == len(node_map):
            return

        cycle = self._extract_task_cycle(node_map, successors)
        if cycle:
            cycle_text = " -> ".join(cycle)
            raise ValueError(f"Task graph must remain a DAG; cycle detected: {cycle_text}")

        remaining = sorted(nid for nid, degree in indegree.items() if degree > 0)
        raise ValueError(
            "Task graph must remain a DAG; unresolved cyclic dependency among tasks: "
            + ", ".join(remaining)
        )

    def _extract_task_cycle(
        self,
        node_map: dict[str, dict[str, Any]],
        successors: dict[str, list[str]],
    ) -> list[str]:
        """Return one cycle path for diagnostics, e.g. A -> B -> A."""
        color: dict[str, int] = {nid: 0 for nid in node_map}
        stack: list[str] = []
        stack_index: dict[str, int] = {}

        def _dfs(nid: str) -> list[str]:
            color[nid] = 1
            stack_index[nid] = len(stack)
            stack.append(nid)

            for sid in successors.get(nid, []):
                if color[sid] == 0:
                    cycle = _dfs(sid)
                    if cycle:
                        return cycle
                elif color[sid] == 1:
                    start = stack_index[sid]
                    return stack[start:] + [sid]

            stack.pop()
            stack_index.pop(nid, None)
            color[nid] = 2
            return []

        for nid in node_map:
            if color[nid] == 0:
                cycle = _dfs(nid)
                if cycle:
                    return cycle
        return []

    def _find_node(self, node_id: str) -> dict[str, Any] | None:
        """Return the first node with the matching id (or None)."""
        for n in self.nodes:
            if n.get("id") == node_id:
                return n
        return None

    def next_ready_task(self) -> dict[str, Any] | None:
        """Select the next task whose predecessors are all satisfied."""
        ready_nodes = self.graph_ready_task_nodes()
        return ready_nodes[0] if ready_nodes else None

    def graph_ready_task_nodes(self) -> list[dict[str, Any]]:
        """Return pending task nodes whose DAG predecessors are completed."""

        def _pred_satisfied(status: Any) -> bool:
            s = str(status) if status else ""
            return s == "completed"

        def _pred_ready(pred_id: str) -> bool:
            pred = self._find_node(pred_id)
            if pred is None:
                return False
            return _pred_satisfied(pred.get("status"))

        ready_nodes: list[dict[str, Any]] = []
        for node in self.nodes:
            if node.get("type") != "task":
                continue
            if node.get("status") != "pending":
                continue

            preds = node.get("predecessors", [])
            if not preds:
                ready_nodes.append(node)
                continue

            if all(_pred_ready(pid) for pid in preds):
                ready_nodes.append(node)

        return ready_nodes

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {"nodes": self.nodes}
        with p.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self.logger.debug(f"[Planner] Saved plan to {p.resolve()}")

    def load(self, path: Path | str) -> None:
        p = Path(path)
        if not p.exists():
            self.logger.warning(f"[Planner] Plan file missing: {p}")
            return
        with p.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        self.nodes = payload.get("nodes", [])
        self.logger.debug(f"[Planner] Loaded plan from {p.resolve()}")

    # ------------------------------------------------------------------ #
    # Build Global FSA
    # ------------------------------------------------------------------ #
    def compile_global_fsa(self) -> dict[str, Any]:
        """Compile the task DAG into a global plan FSA for offline/online monitoring."""

        # ---- extract task nodes ----
        tasks = [n for n in self.nodes if n.get("type") == "task"]
        if not tasks:
            raise ValueError("No task nodes exist in self.nodes (cannot compile FSA).")

        self._validate_task_graph(self.nodes)

        by_id = {t["id"]: t for t in tasks}
        tools_catalog = list(getattr(self.product_agent, "tools_catalog", []) or [])

        def _resource_short_name(value: str) -> str:
            token = str(value or "").strip()
            if "@" in token:
                token = token.split("@", 1)[0]
            return token.lower()

        tool_meta_by_resource_fn: dict[tuple[str, str], dict[str, Any]] = {}
        fallback_tool_meta_by_fn: dict[str, dict[str, Any]] = {}
        for row in tools_catalog:
            if not isinstance(row, dict):
                continue
            fn = str(row.get("function", "")).strip()
            if not fn:
                continue
            owner = _resource_short_name(str(row.get("function_owner_agent", "")).strip())
            if owner:
                tool_meta_by_resource_fn.setdefault((owner, fn), row)
            fallback_tool_meta_by_fn.setdefault(fn, row)

        def _tool_meta_for_task(task: dict[str, Any]) -> dict[str, Any]:
            """Look up tool metadata, preferring node-level in_state/out_state for recovery macros."""
            resource = _resource_short_name(str(task.get("resource_jid", "")).strip())
            fn = str(task.get("function_name", "")).strip()
            catalog_meta = (
                tool_meta_by_resource_fn.get((resource, fn))
                or fallback_tool_meta_by_fn.get(fn)
                or {}
            )
            # Prefer node-level metadata (set by recovery macro proposals)
            # over shared-catalog metadata.
            if task.get("in_state") or task.get("out_state"):
                merged = dict(catalog_meta)
                if task.get("in_state"):
                    merged["in_state"] = task["in_state"]
                if task.get("out_state"):
                    merged["out_state"] = task["out_state"]
                return merged
            return catalog_meta

        # ---- deterministic per-resource local order ----
        res_to_tasks: dict[str, list[str]] = defaultdict(list)
        for t in tasks:
            res = t.get("resource_jid")
            if not res:
                raise ValueError(f"Task {t.get('id')} missing resource_jid.")
            res_to_tasks[res].append(t["id"])

        def sort_key(tid: str):
            si = by_id[tid].get("sequence_index", None)
            return (10**9 if si is None else int(si), tid)

        for res in res_to_tasks:
            res_to_tasks[res].sort(key=sort_key)

        resources = sorted(res_to_tasks.keys())

        pos: dict[str, dict[str, int]] = {
            res: {tid: i for i, tid in enumerate(res_to_tasks[res])} for res in resources
        }

        # ---- labeling helpers (NEW) ----
        def task_sig(tid: str) -> str:
            """
            Human-readable task signature.
            Example: 'pick_grasp(SG)' or 'place_approach(MCP)'.
            """
            t = by_id[tid]
            fn = t.get("function_name") or "unknown_fn"
            params = t.get("params") or {}

            # Prefer showing the part if present (often the most informative)
            part = params.get("part_name")
            if part:
                return f"{fn}({part})"
            return fn

        def readable_event(tid: str, phase: str) -> str:
            """
            Example: 'pick_grasp(SG).start' / 'pick_grasp(SG).done'
            """
            return f"{task_sig(tid)}.{phase}"

        # ---- helpers ----
        def _idx(res: str) -> int:
            return resources.index(res)

        def _get_local(x, res):
            return x[_idx(res)]

        def _set_local(x, res, new_local):
            xl = list(x)
            xl[_idx(res)] = new_local
            return tuple(xl)

        def is_completed(x, task_id):
            t = by_id[task_id]
            res = t["resource_jid"]
            k, run = _get_local(x, res)

            # task position in that resource's local list
            task_pos = pos[res][task_id]

            # done iff k has advanced past task_pos AND we aren't currently running it
            return (k > task_pos) and not (run == task_pos)

        def next_local_task_if_any(x, res):
            k, run = _get_local(x, res)
            local = res_to_tasks[res]
            return None if k >= len(local) else local[k]

        def is_next_task_enabled(x, tid):
            preds = by_id[tid].get("predecessors", []) or []
            for p in preds:
                if p not in by_id:
                    return False

                # If predecessor is still running, block start
                t = by_id[p]
                res = t["resource_jid"]
                k, run = _get_local(x, res)
                if run is not None and res_to_tasks[res][run] == p:
                    return False

                # If predecessor not completed yet, block start
                if not is_completed(x, p):
                    return False

            return True

        # ---- initial / marked ----
        x0 = tuple((0, None) for _ in resources)
        x_marked = tuple((len(res_to_tasks[r]), None) for r in resources)

        def state_name(x):
            """
            Make states readable by showing which task is running (by function_name + part)
            instead of run index.
            """
            parts = []
            for i, res in enumerate(resources):
                k, run = x[i]
                if run is None:
                    parts.append(f"{res}=(k={k},idle)")
                else:
                    tid = res_to_tasks[res][run]
                    fn = by_id[tid].get("function_name") or "unknown_fn"
                    parts.append(f"{res}=(k={k},run={tid}:{fn})")
            return "(" + ",".join(parts) + ")"

        # ---- BFS ----
        visited = {x0}
        q = deque([x0])
        name_map = {x0: state_name(x0)}
        transitions = []

        while q:
            x = q.popleft()
            sx = name_map[x]

            for res in resources:
                k, run = _get_local(x, res)

                # IDLE → start
                if run is None:
                    tid = next_local_task_if_any(x, res)
                    if tid and is_next_task_enabled(x, tid):
                        e = f"{tid}.start"
                        x_next = _set_local(x, res, (k, pos[res][tid]))

                        if x_next not in visited:
                            visited.add(x_next)
                            name_map[x_next] = state_name(x_next)
                            q.append(x_next)

                        transitions.append(
                            {
                                "from": sx,
                                "event": e,  # canonical
                                "readable_event": readable_event(tid, "start"),
                                "to": name_map[x_next],
                                "resource_jid": res,
                                "task_id": tid,
                                "function_name": by_id[tid].get("function_name"),
                                "params": dict(by_id[tid].get("params") or {}),
                                "in_state": _tool_meta_for_task(by_id[tid]).get("in_state"),
                                "out_state": _tool_meta_for_task(by_id[tid]).get("out_state"),
                            }
                        )

                # RUNNING → done
                else:
                    tid = res_to_tasks[res][run]
                    e = f"{tid}.done"
                    x_done = _set_local(x, res, (k + 1, None))

                    if x_done not in visited:
                        visited.add(x_done)
                        name_map[x_done] = state_name(x_done)
                        q.append(x_done)

                    transitions.append(
                        {
                            "from": sx,
                            "event": e,  # canonical
                            "readable_event": readable_event(tid, "done"),
                            "to": name_map[x_done],
                            "resource_jid": res,
                            "task_id": tid,
                            "function_name": by_id[tid].get("function_name"),
                            "params": dict(by_id[tid].get("params") or {}),
                            "in_state": _tool_meta_for_task(by_id[tid]).get("in_state"),
                            "out_state": _tool_meta_for_task(by_id[tid]).get("out_state"),
                        }
                    )

        if x_marked not in name_map:
            raise ValueError(
                "Global FSA has no reachable marked state. "
                "The current task graph likely contains a cycle or an unreachable dependency."
            )

        fsa = {
            "A": {
                "X": sorted(name_map.values()),
                "E": sorted({t["event"] for t in transitions}),
                "Tr": transitions,
                "x0": name_map[x0],
                "Xm": [name_map[x_marked]],
            },
            "meta": {
                "resources": resources,
                "num_reachable_states": len(visited),
                "num_transitions": len(transitions),
                "has_failure_states": False,
            },
        }

        self.global_fsa = fsa  # ← STORE IT
        return fsa

    def save_global_fsa(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        if self.global_fsa is None:
            self.compile_global_fsa()

        with p.open("w", encoding="utf-8") as f:
            json.dump(self.global_fsa, f, indent=2)

        self.logger.debug(f"[Planner] Saved global FSA to {p.resolve()}")

    def load_global_fsa(self, path: Path | str) -> None:
        p = Path(path)
        if not p.exists():
            self.logger.warning(f"[Planner] Global FSA file missing: {p}")
            return
        with p.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            self.logger.warning(f"[Planner] Invalid global FSA payload in {p}")
            return
        self.global_fsa = payload
        self.logger.debug(f"[Planner] Loaded global FSA from {p.resolve()}")

    # ------------------------------------------------------------------ #
    # v2 Universal Repair — apply validated program to live graph
    # ------------------------------------------------------------------ #
