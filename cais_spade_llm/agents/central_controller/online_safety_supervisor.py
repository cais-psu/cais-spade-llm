"""Online supervisor over live plan/safety product states."""

from __future__ import annotations

import logging
from collections import deque
from copy import deepcopy
from typing import Any

from cais_spade_llm.agents.central_controller.online_fsa_monitor import OnlineFsaMonitor
from cais_spade_llm.agents.central_controller.online_safety_monitor import OnlineSafetyMonitor

ProductState = tuple[str, tuple[str, ...], tuple[tuple[str, str, str], ...]]


class OnlineSafetySupervisor:
    """
    Supervisory layer over the live plan FSA state and per-rule safety DFA states.

    `preventive` and `reactive` use a precomputed winning set over the product
    automaton. `truly_reactive` skips winning-set data entirely and instead
    performs exact on-the-fly BFS from the current live state.
    """

    def __init__(
        self,
        *,
        winning_set_data: dict[str, Any] | None = None,
        fsa_monitor: OnlineFsaMonitor,
        safety_monitor: OnlineSafetyMonitor,
        enforcement_mode: str = "preventive",
        plan: dict[str, Any] | None = None,
    ) -> None:
        self.logger = logging.getLogger("OnlineSafetySupervisor")
        self.fsa = fsa_monitor
        self.safety = safety_monitor
        normalized_mode = str(enforcement_mode or "preventive").strip().lower()
        if normalized_mode not in {"preventive", "reactive", "truly_reactive"}:
            normalized_mode = "preventive"
        self.enforcement_mode = normalized_mode
        self.plan = dict(plan or {})

        winning_payload = dict(winning_set_data or {})
        self.W: set[ProductState] = set(winning_payload.get("W") or set())
        self.graph: dict[ProductState, list[dict[str, Any]]] = dict(
            winning_payload.get("product_graph") or {}
        )
        self.accepting_states: set[ProductState] = set(
            winning_payload.get("accepting_states") or set()
        )
        self.rule_ids: list[str] = list(
            winning_payload.get("rule_ids")
            or sorted(str(rule_id) for rule_id in self.safety.dfas.keys())
        )

        fsa_payload = (self.fsa.fsa or {}).get("A") or {}
        self._enabled = self._index_enabled(list(fsa_payload.get("Tr") or []))
        self._marked_states: set[str] = set(
            str(x) for x in (fsa_payload.get("Xm") or []) if str(x).strip()
        )
        self._task_lookup = self._build_task_lookup(self.plan)
        self._task_meta_lookup = self._build_transition_task_lookup(
            self._enabled,
            self._task_lookup,
        )
        self._rule_ap_sets: dict[str, set[str]] = {
            rule_id: set(self.safety.dfas.get(rule_id, {}).get("ap_symbols", []))
            for rule_id in self.rule_ids
        }

        base_resource_states = winning_payload.get("initial_resource_states")
        if not isinstance(base_resource_states, dict):
            base_resource_states = self._initial_resource_states(
                self._enabled,
                str(fsa_payload.get("x0") or ""),
                self._task_meta_lookup,
            )
        self.base_resource_states: dict[str, dict[str, Any]] = deepcopy(base_resource_states)

    @staticmethod
    def _index_enabled(transitions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        enabled: dict[str, list[dict[str, Any]]] = {}
        for transition in transitions:
            from_state = str(transition.get("from") or "").strip()
            if not from_state:
                continue
            enabled.setdefault(from_state, []).append(transition)
        return enabled

    def _build_task_lookup(self, plan: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for node in (plan or {}).get("nodes") or []:
            task_id = str(node.get("id") or "").strip()
            if task_id:
                out[task_id] = dict(node)
        return out

    @staticmethod
    def _merge_task_metadata(dest: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
        for key in ("task_id", "resource_jid", "function_name", "in_state", "out_state"):
            value = src.get(key)
            if value is None:
                continue
            token = str(value).strip()
            if token:
                dest[key] = value
        params = src.get("params")
        if isinstance(params, dict):
            if dest.get("params") is None:
                dest["params"] = {}
            if params:
                dest["params"] = dict(params)
        return dest

    def _build_transition_task_lookup(
        self,
        enabled: dict[str, list[dict[str, Any]]],
        task_lookup: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}

        for task_id, node in task_lookup.items():
            out[str(task_id)] = self._merge_task_metadata({}, dict(node))

        for transitions in enabled.values():
            for transition in transitions:
                task_id = str(transition.get("task_id") or "").strip()
                if not task_id:
                    continue
                entry = out.setdefault(task_id, {})
                self._merge_task_metadata(entry, transition)
                if task_lookup.get(task_id):
                    self._merge_task_metadata(entry, task_lookup[task_id])
                entry["task_id"] = task_id
                if "params" not in entry:
                    entry["params"] = {}

        return out

    def _transition_task_meta(self, transition: dict[str, Any]) -> dict[str, Any]:
        task_id = str(transition.get("task_id") or "").strip()
        meta: dict[str, Any] = {}
        if task_id and task_id in self._task_meta_lookup:
            self._merge_task_metadata(meta, self._task_meta_lookup[task_id])
        if task_id and task_id in self._task_lookup:
            self._merge_task_metadata(meta, self._task_lookup[task_id])
        self._merge_task_metadata(meta, transition)
        if task_id:
            meta["task_id"] = task_id
        if not isinstance(meta.get("params"), dict):
            meta["params"] = {}
        return meta

    def _initial_resource_states(
        self,
        enabled: dict[str, list[dict[str, Any]]],
        x0: str,
        task_meta_lookup: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        resource_states: dict[str, dict[str, Any]] = {}
        owner_to_jid: dict[str, str] = {}
        default_domain = "localhost"

        for meta in task_meta_lookup.values():
            resource_jid = str(meta.get("resource_jid") or "").strip()
            if resource_jid and resource_jid not in resource_states:
                resource_states[resource_jid] = {"current_state": "idle", "params": {}}
            if resource_jid:
                owner_to_jid[self.safety._resource_short_name(resource_jid)] = resource_jid
                if "@" in resource_jid:
                    default_domain = resource_jid.split("@", 1)[1] or default_domain

        for row in self.safety.tools_catalog:
            owner = self.safety._resource_short_name(
                str(row.get("function_owner_agent") or "").strip()
            )
            if not owner:
                continue
            resource_jid = owner_to_jid.get(owner) or f"{owner}@{default_domain}"
            resource_states.setdefault(resource_jid, {"current_state": "idle", "params": {}})

        for transition in enabled.get(str(x0), []):
            task_id = str(transition.get("task_id") or "").strip()
            meta = task_meta_lookup.get(task_id, {})
            resource_jid = str(
                meta.get("resource_jid") or transition.get("resource_jid") or ""
            ).strip()
            in_state = str(meta.get("in_state") or transition.get("in_state") or "").strip()
            if not resource_jid:
                continue
            resource_states.setdefault(resource_jid, {"current_state": "idle", "params": {}})
            if in_state and in_state.lower() != "any":
                resource_states[resource_jid]["current_state"] = in_state

        return resource_states

    def _resource_state_signature(
        self,
        resource_states: dict[str, dict[str, Any]],
    ) -> tuple[tuple[str, str, str], ...]:
        items: list[tuple[str, str, str]] = []
        for resource_jid, payload in sorted((resource_states or {}).items()):
            current_state = str((payload or {}).get("current_state") or "").strip()
            params = dict((payload or {}).get("params") or {})
            state_token = self.safety._resource_state_signature_token(
                str(resource_jid),
                current_state,
                params,
            )
            items.append((str(resource_jid), current_state, state_token))
        return tuple(items)

    def _merged_resource_states(self) -> dict[str, dict[str, Any]]:
        merged = deepcopy(self.base_resource_states)
        for resource_jid, payload in (getattr(self.safety, "resource_states", {}) or {}).items():
            merged[str(resource_jid)] = {
                "current_state": str((payload or {}).get("current_state") or ""),
                "params": dict((payload or {}).get("params") or {}),
            }
        return merged

    def _current_q_vector(self) -> tuple[str, ...]:
        return tuple(str(self.safety.current_states.get(rule_id, "1")) for rule_id in self.rule_ids)

    def _current_safety_state_map(self) -> dict[str, str]:
        return {
            rule_id: str(self.safety.current_states.get(rule_id, "1")) for rule_id in self.rule_ids
        }

    def current_product_state(self) -> ProductState:
        x = str(self.fsa.current_state or "")
        q_vec = self._current_q_vector()
        resource_sig = self._resource_state_signature(self._merged_resource_states())
        return (x, q_vec, resource_sig)

    def _pending_rule_ids(self, q_vec: tuple[str, ...]) -> list[str]:
        pending: list[str] = []
        for idx, rule_id in enumerate(self.rule_ids):
            accepting_states = set(
                str(s) for s in (self.safety.dfas.get(rule_id, {}).get("accepting_states") or [])
            )
            violation_state = str(
                self.safety.dfas.get(rule_id, {}).get("violation_state") or ""
            ).strip()
            if violation_state and q_vec[idx] == violation_state:
                continue
            if accepting_states and q_vec[idx] not in accepting_states:
                pending.append(rule_id)
        return pending

    def _violated_rule_ids(self, q_vec: tuple[str, ...]) -> list[str]:
        violated: list[str] = []
        for idx, rule_id in enumerate(self.rule_ids):
            violation_state = str(
                self.safety.dfas.get(rule_id, {}).get("violation_state") or ""
            ).strip()
            if violation_state and q_vec[idx] == violation_state:
                violated.append(rule_id)
        return violated

    def _running_tasks_from_state(self, x: str) -> list[dict[str, str]]:
        running: list[dict[str, str]] = []
        parsed = self.fsa._parse_state(x or "")
        for resource_jid, info in parsed.items():
            if str(info.get("status") or "") != "running":
                continue
            task_id = str(info.get("run_task_id") or "").strip()
            function_name = str(info.get("run_function") or "").strip()
            if not task_id or not function_name:
                continue
            running.append(
                {
                    "resource_jid": str(resource_jid),
                    "task_id": task_id,
                    "function_name": function_name,
                }
            )
        return running

    def _running_event_aps_from_state_all(
        self,
        x: str,
    ) -> set[str]:
        sigma: set[str] = set()
        for item in self._running_tasks_from_state(x):
            meta = dict(self._task_meta_lookup.get(item["task_id"]) or {})
            resource_jid = str(meta.get("resource_jid") or item["resource_jid"] or "").strip()
            function_name = str(meta.get("function_name") or item["function_name"] or "").strip()
            params = dict(meta.get("params") or {})
            if not resource_jid or not function_name:
                continue
            sigma.update(self.safety._map_task_to_aps(resource_jid, function_name, params))
        return sigma

    def _state_aps_for_resources_all(
        self,
        resource_states: dict[str, dict[str, Any]],
    ) -> set[str]:
        sigma: set[str] = set()
        for resource_jid, payload in (resource_states or {}).items():
            current_state = str((payload or {}).get("current_state") or "").strip()
            if not current_state:
                continue
            params = dict((payload or {}).get("params") or {})
            sigma.update(self.safety._map_state_to_aps(resource_jid, current_state, params))
        return sigma

    def _local_edges_from(
        self,
        product_state: ProductState,
        *,
        resource_states: dict[str, dict[str, Any]],
        include_violating: bool = False,
    ) -> list[dict[str, Any]]:
        x, q_vec, _ = product_state
        edges: list[dict[str, Any]] = []

        for transition in self._enabled.get(x, []):
            x2 = str(transition.get("to") or "").strip()
            if not x2:
                continue

            event_name = str(transition.get("event") or "").strip()
            meta = self._transition_task_meta(transition)
            resource_jid = str(meta.get("resource_jid") or "").strip()
            function_name = str(meta.get("function_name") or "").strip()
            params = dict(meta.get("params") or {})

            running_before_all = set(self._running_event_aps_from_state_all(x))
            persistent_before_all = set(self._state_aps_for_resources_all(resource_states))

            if resource_jid and function_name:
                candidate_event_aps_all = set(
                    self.safety._map_task_to_aps(resource_jid, function_name, params)
                )
                predicted_state_aps_all = set(
                    self.safety._predict_state_aps(resource_jid, function_name, params)
                )
            else:
                candidate_event_aps_all = set()
                predicted_state_aps_all = set()

            next_resource_states = deepcopy(resource_states)

            if event_name.endswith(".done") or event_name.endswith(".finish"):
                running_after_all = set(running_before_all)
                for ap in candidate_event_aps_all:
                    running_after_all.discard(ap)

                if resource_jid:
                    next_payload = next_resource_states.setdefault(
                        resource_jid,
                        {
                            "current_state": next_resource_states.get(resource_jid, {}).get(
                                "current_state",
                                "idle",
                            ),
                            "params": {},
                        },
                    )
                    out_state = str(meta.get("out_state") or "").strip()
                    if out_state and out_state.lower() != "any":
                        next_payload["current_state"] = out_state
                        next_payload["params"] = dict(params)
                    elif "params" not in next_payload:
                        next_payload["params"] = dict(params)

                persistent_after_all = set(self._state_aps_for_resources_all(next_resource_states))
                sigma_all = frozenset(
                    running_after_all | persistent_after_all | candidate_event_aps_all
                )
            elif event_name.endswith(".start"):
                sigma_all = frozenset(
                    running_before_all
                    | persistent_before_all
                    | candidate_event_aps_all
                    | predicted_state_aps_all
                )
            else:
                sigma_all = frozenset(
                    running_before_all | persistent_before_all | candidate_event_aps_all
                )

            checked_vec: list[str] = []
            committed_vec: list[str] = []
            violated_rule_ids: list[str] = []
            for idx, rule_id in enumerate(self.rule_ids):
                sigma_rule = frozenset(
                    ap for ap in sigma_all if ap in self._rule_ap_sets.get(rule_id, set())
                )
                current_q = q_vec[idx]
                checked_q = self.safety._delta(rule_id, current_q, sigma_rule)
                checked_vec.append(checked_q)
                committed_q = current_q if event_name.endswith(".start") else checked_q
                committed_vec.append(committed_q)

                violation_state = str(
                    self.safety.dfas.get(rule_id, {}).get("violation_state") or ""
                ).strip()
                if violation_state and checked_q == violation_state:
                    violated_rule_ids.append(rule_id)

            successor = (
                x2,
                tuple(committed_vec),
                self._resource_state_signature(next_resource_states),
            )
            edge_meta = {
                "event": transition.get("event"),
                "task_id": transition.get("task_id"),
                "resource_jid": transition.get("resource_jid"),
                "function_name": transition.get("function_name"),
                "params": dict(transition.get("params") or {}),
                "from": x,
                "to": x2,
                "successor": successor,
                "_checked_q_vec": tuple(checked_vec),
                "_violated_rule_ids": violated_rule_ids,
                "_resource_states": deepcopy(next_resource_states),
            }
            if violated_rule_ids and not include_violating:
                continue
            edges.append(edge_meta)

        return edges

    def _online_bfs_analysis(
        self,
        *,
        start: ProductState | None = None,
        resource_states: dict[str, dict[str, Any]] | None = None,
        root_pending_rule_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        start_state = start or self.current_product_state()
        start_resource_states = deepcopy(resource_states or self._merged_resource_states())
        start_pending = set(root_pending_rule_ids or self._pending_rule_ids(start_state[1]))
        immediate_edges = self._local_edges_from(start_state, resource_states=start_resource_states)
        immediate_start_task_ids = sorted(
            {
                str(edge.get("task_id"))
                for edge in immediate_edges
                if str(edge.get("event") or "").endswith(".start") and edge.get("task_id")
            }
        )

        analysis: dict[str, Any] = {
            "start_pending_rule_ids": sorted(start_pending),
            "immediate_edges": immediate_edges,
            "immediate_start_task_ids": immediate_start_task_ids,
            "has_discharge": False,
            "discharge_trace": [],
        }
        if not start_pending:
            return analysis

        queue = deque([(start_state, deepcopy(start_resource_states))])
        seen: set[ProductState] = {start_state}
        parent: dict[ProductState, ProductState | None] = {start_state: None}
        parent_edge: dict[ProductState, dict[str, Any] | None] = {start_state: None}
        target: ProductState | None = None

        while queue:
            node, node_resource_states = queue.popleft()
            for edge in self._local_edges_from(node, resource_states=node_resource_states):
                successor = edge.get("successor")
                if not isinstance(successor, tuple):
                    continue

                if successor in seen:
                    continue

                seen.add(successor)
                parent[successor] = node
                parent_edge[successor] = edge
                successor_pending = set(self._pending_rule_ids(successor[1]))
                if start_pending - successor_pending:
                    target = successor
                    continue

                queue.append(
                    (
                        successor,
                        deepcopy(edge.get("_resource_states") or {}),
                    )
                )
                if target is not None:
                    break
            if target is not None:
                break

        if target is None:
            return analysis

        trace: list[dict[str, Any]] = []
        cur = target
        while cur != start_state:
            edge = parent_edge.get(cur)
            if edge is not None:
                trace.append(edge)
            prev = parent.get(cur)
            if prev is None:
                break
            cur = prev
        trace.reverse()

        analysis["has_discharge"] = True
        analysis["discharge_trace"] = trace
        return analysis

    def _reachability_basis(self) -> str:
        return "on_the_fly_bfs" if self.enforcement_mode == "truly_reactive" else "winning_set"

    def _safe_edges_from(self, product_state: ProductState) -> list[dict[str, Any]]:
        edges = list(self.graph.get(product_state, []))
        return [edge for edge in edges if edge.get("successor") in self.W]

    def _truly_reactive_safe_next_task_ids(self, analysis: dict[str, Any]) -> list[str]:
        start_pending = set(analysis.get("start_pending_rule_ids") or [])
        if not start_pending:
            return list(analysis.get("immediate_start_task_ids") or [])

        discharge_task_ids: set[str] = set()
        for edge in analysis.get("immediate_edges") or []:
            event_name = str(edge.get("event") or "").strip()
            task_id = str(edge.get("task_id") or "").strip()
            successor = edge.get("successor")
            if not event_name.endswith(".start") or not task_id or not isinstance(successor, tuple):
                continue

            successor_pending = set(self._pending_rule_ids(successor[1]))
            if start_pending - successor_pending:
                discharge_task_ids.add(task_id)
                continue

            successor_analysis = self._online_bfs_analysis(
                start=successor,
                resource_states=deepcopy(edge.get("_resource_states") or {}),
                root_pending_rule_ids=start_pending,
            )
            if successor_analysis["has_discharge"]:
                discharge_task_ids.add(task_id)

        return sorted(discharge_task_ids)

    def safe_next_task_ids(self, *, analysis: dict[str, Any] | None = None) -> list[str]:
        current = self.current_product_state()
        if self.enforcement_mode == "truly_reactive":
            if self._violated_rule_ids(current[1]):
                return []
            resolved = analysis or self._online_bfs_analysis()
            return self._truly_reactive_safe_next_task_ids(resolved)

        task_ids = {
            str(edge.get("task_id"))
            for edge in self._safe_edges_from(current)
            if str(edge.get("event") or "").endswith(".start") and edge.get("task_id")
        }
        return sorted(task_ids)

    def _serialize_hint_trace(
        self,
        trace: list[dict[str, Any]],
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        serialized = [
            {
                "task_id": edge.get("task_id"),
                "event": edge.get("event"),
                "resource_jid": edge.get("resource_jid"),
                "function_name": edge.get("function_name"),
                "params": dict(edge.get("params") or {}),
            }
            for edge in trace
        ]
        if limit is not None:
            return serialized[:limit]
        return serialized

    def safe_suffix_hint(
        self,
        *,
        limit: int | None = None,
        analysis: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        start = self.current_product_state()
        if self.enforcement_mode == "truly_reactive":
            if self._violated_rule_ids(start[1]):
                return []
            resolved = analysis or self._online_bfs_analysis()
            trace = list(resolved.get("discharge_trace") or [])
            if not trace:
                return []
            return self._serialize_hint_trace(trace, limit=limit)

        if start not in self.W:
            return []
        if start in self.accepting_states:
            return []

        queue = deque([start])
        parent: dict[ProductState, ProductState | None] = {start: None}
        parent_edge: dict[ProductState, dict[str, Any] | None] = {start: None}
        seen: set[ProductState] = {start}

        target: ProductState | None = None
        while queue:
            node = queue.popleft()
            if node in self.accepting_states:
                target = node
                break
            for edge in self._safe_edges_from(node):
                successor = edge.get("successor")
                if successor in seen:
                    continue
                seen.add(successor)
                parent[successor] = node
                parent_edge[successor] = edge
                queue.append(successor)

        if target is None:
            return []

        trace: list[dict[str, Any]] = []
        cur = target
        while cur != start:
            edge = parent_edge.get(cur)
            if edge is not None:
                trace.append(edge)
            prev = parent.get(cur)
            if prev is None:
                break
            cur = prev
        trace.reverse()
        return self._serialize_hint_trace(trace, limit=limit)

    def classify(self, *, event_kind: str | None = None) -> dict[str, Any]:
        current = self.current_product_state()
        x, q_vec, _ = current
        violated_rule_ids = self._violated_rule_ids(q_vec)
        pending_rule_ids = self._pending_rule_ids(q_vec)
        truly_reactive_analysis: dict[str, Any] | None = None
        if self.enforcement_mode == "truly_reactive" and not violated_rule_ids:
            truly_reactive_analysis = self._online_bfs_analysis()
        safe_next_task_ids = self.safe_next_task_ids(analysis=truly_reactive_analysis)
        event_token = str(event_kind or "").strip().lower()

        if violated_rule_ids:
            status = "violated"
            reason = "Current safety DFA state is already in a violation state."
        elif self.enforcement_mode == "truly_reactive":
            analysis = truly_reactive_analysis or self._online_bfs_analysis()
            if pending_rule_ids and analysis["has_discharge"]:
                status = "pending_obligation"
                reason = (
                    "Safety obligations remain open, but exact on-the-fly BFS found a "
                    "reachable non-violating discharge path."
                )
            elif pending_rule_ids:
                status = "inevitable_violation"
                reason = (
                    "A safety obligation is now open, but exact on-the-fly BFS found no "
                    "reachable modeled discharge path from the current live state."
                )
            else:
                status = "safe"
                reason = (
                    "Truly reactive runtime supervision is active; no current safety "
                    "obligation or violation is open."
                )
        else:
            if self.enforcement_mode == "reactive":
                if event_token == "fail" and current not in self.W:
                    status = "inevitable_violation"
                    reason = "No remaining accepting continuation exists after the runtime failure."
                elif pending_rule_ids and current not in self.W:
                    status = "inevitable_violation"
                    reason = (
                        "A safety obligation is now open, but no remaining accepting continuation "
                        "exists in the current modeled plan."
                    )
                elif pending_rule_ids:
                    status = "pending_obligation"
                    reason = "Safety obligations remain open, but a safe continuation still exists."
                else:
                    status = "safe"
                    reason = (
                        "Reactive runtime supervision is active; no current safety obligation or "
                        "violation has been triggered."
                    )
            else:
                if current not in self.W:
                    status = "inevitable_violation"
                    reason = (
                        "No remaining accepting continuation exists in the current modeled plan."
                    )
                elif pending_rule_ids:
                    status = "pending_obligation"
                    reason = "Safety obligations remain open, but a safe continuation still exists."
                else:
                    status = "safe"
                    reason = "Current execution state is recoverable and all tracked safety DFAs are accepting."

        diagnosis = {
            "status": status,
            "rule_ids": violated_rule_ids or pending_rule_ids,
            "current_plan_state": x,
            "current_safety_states": self._current_safety_state_map(),
            "safe_next_task_ids": safe_next_task_ids,
            "safe_suffix_hint": (
                self.safe_suffix_hint(limit=3, analysis=truly_reactive_analysis)
                if self.enforcement_mode == "truly_reactive" or current in self.W
                else []
            ),
            "reason": reason,
            "enforcement_mode": self.enforcement_mode,
            "reachability_basis": self._reachability_basis(),
        }
        return diagnosis

    def check_candidate(self, event: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        current = self.current_product_state()
        diagnosis = self.classify()
        if diagnosis["status"] == "violated":
            return False, diagnosis

        task_id = str(event.get("task_id") or "").strip()
        event_label = f"{task_id}.start" if task_id else ""

        if self.enforcement_mode == "truly_reactive":
            resource_states = self._merged_resource_states()
            matching_edges = [
                edge
                for edge in self._local_edges_from(
                    current,
                    resource_states=resource_states,
                    include_violating=True,
                )
                if str(edge.get("event") or "").strip() == event_label
            ]
            if not matching_edges:
                return False, {
                    **diagnosis,
                    "status": "blocked_candidate",
                    "candidate_event": event_label,
                    "reason": "Candidate task is not modeled from the current plan state.",
                }

            matching_safe_edges = [
                edge for edge in matching_edges if not edge.get("_violated_rule_ids")
            ]
            if not matching_safe_edges:
                return False, {
                    **diagnosis,
                    "status": "blocked_candidate",
                    "candidate_event": event_label,
                    "reason": "Candidate task would immediately violate safety.",
                }

            root_pending_rule_ids = set(self._pending_rule_ids(current[1]))
            if root_pending_rule_ids:
                discharge_reaching_edges = []
                for edge in matching_safe_edges:
                    successor = edge.get("successor")
                    if not isinstance(successor, tuple):
                        continue

                    successor_pending = set(self._pending_rule_ids(successor[1]))
                    if root_pending_rule_ids - successor_pending:
                        discharge_reaching_edges.append(edge)
                        continue

                    successor_analysis = self._online_bfs_analysis(
                        start=successor,
                        resource_states=deepcopy(edge.get("_resource_states") or {}),
                        root_pending_rule_ids=root_pending_rule_ids,
                    )
                    if successor_analysis["has_discharge"]:
                        discharge_reaching_edges.append(edge)

                if not discharge_reaching_edges:
                    return False, {
                        **diagnosis,
                        "status": "blocked_candidate",
                        "candidate_event": event_label,
                        "reason": (
                            "Candidate task would move the system into a reachable subtree "
                            "with no discharge path for the current open obligation."
                        ),
                    }

            return True, {
                **diagnosis,
                "status": "safe",
                "candidate_event": event_label,
            }

        if self.enforcement_mode == "reactive":
            matching_edges = [
                edge
                for edge in self._local_edges_from(
                    current,
                    resource_states=self._merged_resource_states(),
                    include_violating=True,
                )
                if str(edge.get("event") or "").strip() == event_label
            ]
            if not matching_edges:
                return False, {
                    **diagnosis,
                    "status": "blocked_candidate",
                    "candidate_event": event_label,
                    "reason": "Reactive runtime supervision could not match a safe product edge for the candidate event.",
                }

            matching_safe_edges = [
                edge for edge in matching_edges if not edge.get("_violated_rule_ids")
            ]
            if not matching_safe_edges:
                return False, {
                    **diagnosis,
                    "status": "blocked_candidate",
                    "candidate_event": event_label,
                    "reason": "Candidate task would immediately violate safety.",
                }

            root_pending_rule_ids = set(self._pending_rule_ids(current[1]))
            if root_pending_rule_ids:
                discharge_reaching_edges = []
                for edge in matching_safe_edges:
                    successor = edge.get("successor")
                    if not isinstance(successor, tuple):
                        continue

                    successor_pending = set(self._pending_rule_ids(successor[1]))
                    if root_pending_rule_ids - successor_pending:
                        discharge_reaching_edges.append(edge)
                        continue

                    successor_analysis = self._online_bfs_analysis(
                        start=successor,
                        resource_states=deepcopy(edge.get("_resource_states") or {}),
                        root_pending_rule_ids=root_pending_rule_ids,
                    )
                    if successor_analysis["has_discharge"]:
                        discharge_reaching_edges.append(edge)

                if not discharge_reaching_edges:
                    return False, {
                        **diagnosis,
                        "status": "blocked_candidate",
                        "candidate_event": event_label,
                        "reason": (
                            "Candidate task would move the system into a reachable subtree "
                            "with no discharge path for the current open obligation."
                        ),
                    }

            return True, {
                **diagnosis,
                "status": "safe",
                "candidate_event": event_label,
            }

        if diagnosis["status"] == "inevitable_violation":
            return False, diagnosis

        matching_safe_edges = [
            edge
            for edge in self._safe_edges_from(current)
            if str(edge.get("event") or "").strip() == event_label
        ]
        if matching_safe_edges:
            return True, {
                **diagnosis,
                "status": "safe",
                "candidate_event": event_label,
            }

        matching_edges = [
            edge
            for edge in (self.graph.get(current, []) or [])
            if str(edge.get("event") or "").strip() == event_label
        ]
        reason = (
            "Candidate task would leave the recoverable winning set."
            if matching_edges
            else "Candidate task is not modeled as a safe successor from the current product state."
        )
        return False, {
            **diagnosis,
            "status": "blocked_candidate",
            "candidate_event": event_label,
            "reason": reason,
        }
