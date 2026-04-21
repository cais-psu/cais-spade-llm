"""Process planner that turns requirements into tasks and plan automata."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from cais_spade_llm.agents.intelligent_product.process_recovery_planner import (
    ProcessRecoveryPlanner,
)
from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge import (
    LlmBridgeReplannerMixin,
)
from cais_spade_llm.prompts import (
    build_requirement_parse_prompt,
    build_task_expansion_prompt,
)

class ProcessPlanner(LlmBridgeReplannerMixin):
    """
    1. NL → structured requirement nodes   (build_high_level)
    2. requirement nodes → executable task DAG   (expand_requirements_to_tasks)
    """

    def __init__(self, product_agent, resource_agents: Iterable[Any]):
        """Initialize planner state with agent references and empty node graphs."""
        self.product_agent = product_agent
        self.resource_agents = list(resource_agents)
        self.logger = product_agent.logger
        self.nodes: List[Dict[str, Any]] = []
        self.phase_to_node: Dict[str, Dict[str, Any]] = {}
        self.global_fsa: Optional[Dict[str, Any]] = None
        self.last_bridge_debug: Dict[str, Any] = {}
        self.recovery_planner = ProcessRecoveryPlanner(self)
        self.recovery_planner.bind_methods()

    @staticmethod
    def _primitive_bridge_macro_tasks(proposal: dict[str, Any]) -> list[dict[str, Any]]:
        raw_tasks = proposal.get("macro_tasks")
        if isinstance(raw_tasks, list) and raw_tasks:
            return [dict(task) for task in raw_tasks if isinstance(task, dict)]
        if proposal.get("primitive_steps"):
            return [dict(proposal)]
        return []

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
            phase = req.get("phase")              # e.g. "ASSEMBLY", "printing", or None
            process_type = req.get("process_type")  # e.g. "PICK_PLACE", "FDM_PRINT", or None
            product = req.get("product")          # e.g. "SG", "MCP", or None
            context = req.get("context") or {}    # e.g. {"origin": "prusa-mk4-2", "destination": "assembly board"}

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
            self.logger.error(
                "[Planner] LLM did not return valid JSON: %s\nRaw: %s", exc, raw
            )
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
        by_req: Dict[str, List[Dict[str, Any]]] = {}
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
    def _extract_conflict_task_ids(self, violations: list[dict], *, source: str) -> Set[str]:
        """Normalize offline/online feedback into a conflict task ID set."""
        conflict_task_ids: Set[str] = set()

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
    def _ensure_graph_consistency(self, nodes: Optional[List[Dict[str, Any]]] = None) -> None:
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

    def _validate_task_graph(self, nodes: Optional[List[Dict[str, Any]]] = None) -> None:
        """Reject invalid task graphs before FSA compilation or execution."""
        working_nodes = nodes if nodes is not None else self.nodes
        tasks = [n for n in working_nodes if n.get("type") == "task"]
        node_map = {n["id"]: n for n in tasks}
        successors: Dict[str, List[str]] = {nid: [] for nid in node_map}
        indegree: Dict[str, int] = {nid: 0 for nid in node_map}

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
        visited: List[str] = []

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
        node_map: Dict[str, Dict[str, Any]],
        successors: Dict[str, List[str]],
    ) -> List[str]:
        """Return one cycle path for diagnostics, e.g. A -> B -> A."""
        color: Dict[str, int] = {nid: 0 for nid in node_map}
        stack: List[str] = []
        stack_index: Dict[str, int] = {}

        def _dfs(nid: str) -> List[str]:
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

    def _find_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Return the first node with the matching id (or None)."""
        for n in self.nodes:
            if n.get("id") == node_id:
                return n
        return None

    def next_ready_task(self) -> Optional[Dict[str, Any]]:
        """Select the next task whose predecessors are all satisfied."""
        ready_nodes = self.graph_ready_task_nodes()
        return ready_nodes[0] if ready_nodes else None

    def graph_ready_task_nodes(self) -> list[Dict[str, Any]]:
        """Return pending task nodes whose DAG predecessors are completed."""
        def _pred_satisfied(status: Any) -> bool:
            s = str(status) if status else ""
            return s == "completed"

        def _pred_ready(pred_id: str) -> bool:
            pred = self._find_node(pred_id)
            if pred is None:
                return False
            return _pred_satisfied(pred.get("status"))

        ready_nodes: list[Dict[str, Any]] = []
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
    def compile_global_fsa(self) -> Dict[str, Any]:
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

        tool_meta_by_resource_fn: Dict[tuple[str, str], dict[str, Any]] = {}
        fallback_tool_meta_by_fn: Dict[str, dict[str, Any]] = {}
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
            """Look up tool metadata, preferring node-level in_state/out_state for bridge macros."""
            resource = _resource_short_name(str(task.get("resource_jid", "")).strip())
            fn = str(task.get("function_name", "")).strip()
            catalog_meta = (
                tool_meta_by_resource_fn.get((resource, fn))
                or fallback_tool_meta_by_fn.get(fn)
                or {}
            )
            # Prefer node-level metadata (set by bridge macro proposals)
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
        res_to_tasks: Dict[str, List[str]] = defaultdict(list)
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

        pos: Dict[str, Dict[str, int]] = {
            res: {tid: i for i, tid in enumerate(res_to_tasks[res])}
            for res in resources
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

                        transitions.append({
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
                        })

                # RUNNING → done
                else:
                    tid = res_to_tasks[res][run]
                    e = f"{tid}.done"
                    x_done = _set_local(x, res, (k + 1, None))

                    if x_done not in visited:
                        visited.add(x_done)
                        name_map[x_done] = state_name(x_done)
                        q.append(x_done)

                    transitions.append({
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
                    })

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

        self.global_fsa = fsa        # ← STORE IT
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

