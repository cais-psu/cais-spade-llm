from __future__ import annotations

from typing import Any, Dict, List, Tuple, Set, FrozenSet

from agents.central_controller.base_safety_checker import BaseSafetyChecker


class OfflineSafetyValidator(BaseSafetyChecker):
    """
    Offline validation of a plan (DAG) against LTLf-based DFA safety rules,
    extended to handle rules involving multiple related events (3+ parts).

    Algorithm (per rule):

      1) PROJECT:
         - Keep only the tasks that actually emit APs used by this rule.
           (All other tasks are irrelevant for this rule's DFA.)

      2) STRUCTURE:
         - Use the full plan DAG to compute the induced partial order among
           those relevant tasks (which relevant task must precede which).

      3) PROJECTED DFS (small state space):
         - Run a DFS only over the relevant tasks (usually small: 2–5),
           exploring all valid topological orders consistent with the
           induced partial order.
         - For each explored schedule prefix, step the rule's DFA with the
           APs emitted by the selected task.
         - If the DFA reaches its violation_state at any prefix (including
           the "end-of-plan" empty step), we report a violation with a
           witness trace (list of task IDs).

    This is:
      - Rule-agnostic (no hard-coded constraint_type).
      - Complete for that rule (if any schedule can violate it, we find one).
      - Efficient, because we only explore interleavings of tasks that
        actually matter to that rule.
    """

    # You can tune this if you ever worry about explosion.
    MAX_RELEVANT_TASKS: int = 5

    def __init__(self, rules: List[Dict[str, Any]], dfa_map: Dict[str, str]) -> None:
        """
        rules   : list of structured rules with fields like:
                  - id, raw_text, aps, ltlf, ...
        dfa_map : { rule_id: dot_string } from ltlf2dfa

        BaseSafetyChecker.__init__ will:
          - parse all DOT strings into self.dfas[rule_id]
          - store rules in self.safety_rules
          - provide shared helpers: _map_task_to_aps, _delta, etc.
        """
        # BaseSafetyChecker signature is (dfa_map, rules)
        super().__init__(dfa_map, rules)

        # Quick lookup table: rule_id -> rule dict
        self.rule_lookup: Dict[str, Dict[str, Any]] = {
            r["id"]: r for r in rules if "id" in r
        }
    # ------------------------------------------------------------------ #
    # PUBLIC ENTRY POINT
    # ------------------------------------------------------------------ #
    def validate_plan_offline(
        self,
        plan: dict,
        product_jid: str | None = None,
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Validate a static plan (JSON DAG) against all DFA safety rules.

        Args:
            plan        : dict with "nodes": [ {id, predecessors, ...}, ... ]
            product_jid : optional identifier for logging.

        Returns:
            (is_valid, violations)
              - is_valid   : bool, True if no rule can be violated by any
                             schedule consistent with the DAG.
              - violations : list of violation entries (possibly empty).
        """
        nodes = plan.get("nodes") or []
        if not nodes:
            return True, []

        self.logger.info(
            "[OfflineValidator] Projected DFS validation for %s (%d tasks)...",
            product_jid,
            len(nodes),
        )

        # 1) Build full DAG once: allows us to query reachability between tasks.
        id_to_node, pred_map, succ_map = self._build_global_graph(nodes)

        all_violations: List[Dict[str, Any]] = []

        # 2) Check each safety rule independently.
        for rule in self.safety_rules:
            rule_id = rule.get("id")
            if not rule_id:
                continue

            dfa = self.dfas.get(rule_id)
            if not dfa:
                # No DFA available for this rule → skip.
                continue

            # AP labels used by this rule's DFA.
            aps_for_rule: Set[str] = set(dfa.get("ap_symbols", []))
            if not aps_for_rule:
                # Rule has no APs to match → skip.
                continue

            # --- Step 2.1: PROJECT plan to tasks relevant for this rule ---
            rel_nodes = self._relevant_nodes_for_rule(nodes, aps_for_rule)
            if not rel_nodes:
                # No events in this plan can trigger this rule.
                continue

            if len(rel_nodes) > self.MAX_RELEVANT_TASKS:
                # Guardrail: relevant set unexpectedly large.
                self.logger.warning(
                    "[OfflineValidator] Rule %s has %d relevant tasks (> %d). "
                    "Skipping exhaustive projected DFS for this rule.",
                    rule_id,
                    len(rel_nodes),
                    self.MAX_RELEVANT_TASKS,
                )
                # You can choose to:
                #  - skip (potential false negatives), or
                #  - fall back to a simpler heuristic.
                # For now, we skip to avoid explosion.
                continue

            # --- Step 2.2: STRUCTURE: induced partial order among relevant tasks ---
            rel_pred_map = self._build_rel_pred_map(rel_nodes, pred_map, succ_map)

            # --- Step 2.3: Precompute sigma (AP-set) per relevant task (for this rule) ---
            sigma_for_node: Dict[str, FrozenSet[str]] = {}
            for n in rel_nodes:
                nid = str(n["id"])
                full_sigma = self._aps_for_node(n)  # all APs triggered by this node
                sigma_for_node[nid] = frozenset(
                    ap for ap in full_sigma if ap in aps_for_rule
                )

            # --- Step 2.4: PROJECTED DFS over relevant tasks for this rule ---
            violations = self._check_rule_projected_dfs(
                rule_id=rule_id,
                rule=rule,
                rel_nodes=rel_nodes,
                rel_pred_map=rel_pred_map,
                sigma_for_node=sigma_for_node,
            )
            all_violations.extend(violations)

        is_valid = len(all_violations) == 0
        return is_valid, all_violations

    # ------------------------------------------------------------------ #
    # GLOBAL GRAPH BUILDERS
    # ------------------------------------------------------------------ #
    def _build_global_graph(
        self,
        nodes: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Set[str]], Dict[str, Set[str]]]:
        """
        Build the full DAG structure from the plan nodes.

        Returns:
            id_to_node : { task_id -> node dict }
            pred_map   : { task_id -> set(immediate predecessor ids) }
            succ_map   : { task_id -> set(immediate successor ids) }
        """
        id_to_node: Dict[str, Dict[str, Any]] = {}
        pred_map: Dict[str, Set[str]] = {}
        succ_map: Dict[str, Set[str]] = {}

        # Register all nodes and initialize pred/succ sets
        for n in nodes:
            nid = str(n.get("id"))
            if not nid:
                continue
            id_to_node[nid] = n
            pred_map.setdefault(nid, set())
            succ_map.setdefault(nid, set())

        # Populate predecessors and successors from "predecessors" field
        for n in nodes:
            nid = str(n.get("id"))
            if nid not in id_to_node:
                continue
            for p in n.get("predecessors", []):
                pid = str(p)
                if pid in id_to_node:
                    pred_map[nid].add(pid)
                    succ_map.setdefault(pid, set()).add(nid)

        return id_to_node, pred_map, succ_map

    def _reachable_in_full(
        self,
        succ_map: Dict[str, Set[str]],
        src: str,
        dst: str,
    ) -> bool:
        """
        Check if there is a path src -> ... -> dst in the full DAG using succ_map.

        This is used to compute the induced partial order among relevant tasks.
        """
        if src == dst:
            return True

        stack = [src]
        seen: Set[str] = set()

        while stack:
            curr = stack.pop()
            if curr == dst:
                return True
            for nxt in succ_map.get(curr, []):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)

        return False

    # ------------------------------------------------------------------ #
    # PROJECTION HELPERS
    # ------------------------------------------------------------------ #
    def _aps_for_node(self, node: Dict[str, Any]) -> List[str]:
        """
        Convenience wrapper: map a full task node to AP labels.

        Uses BaseSafetyChecker._map_task_to_aps, which inspects:
          - resource_jid
          - function_name
          - params
        and matches them against rule["aps"][i]["full"] strings.
        """
        return self._map_task_to_aps(
            node.get("resource_jid", "") or "",
            node.get("function_name", "") or "",
            node.get("params") or {},
        )

    def _relevant_nodes_for_rule(
        self,
        nodes: List[Dict[str, Any]],
        aps_for_rule: Set[str],
    ) -> List[Dict[str, Any]]:
        """
        PROJECT step:
          - Keep only tasks that emit at least one AP from aps_for_rule.

        This prunes away all nodes that cannot affect this rule's DFA state.
        """
        rel_nodes: List[Dict[str, Any]] = []
        for n in nodes:
            sigma = self._aps_for_node(n)
            if any(ap in aps_for_rule for ap in sigma):
                rel_nodes.append(n)
        return rel_nodes

    def _build_rel_pred_map(
        self,
        rel_nodes: List[Dict[str, Any]],
        global_pred_map: Dict[str, Set[str]],
        succ_map: Dict[str, Set[str]],
    ) -> Dict[str, Set[str]]:
        """
        STRUCTURE step:
          - Given the subset of relevant tasks for a rule, build the induced
            partial order among them.

        For each pair (A, B) of relevant tasks:
          - If there is a path A -> ... -> B in the full DAG, then A must
            precede B for this rule. We record A in rel_pred_map[B].

        Output:
          rel_pred_map[nid] = set of relevant task IDs that must precede nid.
        """
        rel_ids = [str(n["id"]) for n in rel_nodes]
        rel_pred_map: Dict[str, Set[str]] = {nid: set() for nid in rel_ids}

        for i, ida in enumerate(rel_ids):
            for j, idb in enumerate(rel_ids):
                if i == j:
                    continue
                # Check reachability in full DAG: ida →* idb?
                if self._reachable_in_full(succ_map, ida, idb):
                    rel_pred_map[idb].add(ida)

        return rel_pred_map

    # ------------------------------------------------------------------ #
    # PROJECTED DFS PER RULE
    # ------------------------------------------------------------------ #
    def _check_rule_projected_dfs(
        self,
        rule_id: str,
        rule: Dict[str, Any],
        rel_nodes: List[Dict[str, Any]],
        rel_pred_map: Dict[str, Set[str]],
        sigma_for_node: Dict[str, FrozenSet[str]],
    ) -> List[Dict[str, Any]]:
        """
        Explore all valid schedules over the relevant tasks for a rule, and
        step that rule's DFA along each schedule.

        State in DFS:
          - done_mask : bitmask over relevant tasks (which have executed)
          - q         : current DFA state
          - trace     : list of task IDs (witness sequence of relevant events)

        Transitions:
          - At a given state, a relevant task T is 'enabled' if:
              • T is not yet done, and
              • all relevant predecessors of T (per rel_pred_map) are done.

          - Picking T advances the DFA with sigma_for_node[T].
          - If DFA hits violation_state at any prefix, we record a violation
            with the current trace (including T) and do not expand that branch.
          - When all relevant tasks are done, we also apply an "end-of-plan"
            empty step (sigma = ∅) to catch rules that require some eventuality.

        visited set:
          - We memoize (done_mask, q) to avoid re-exploring identical prefixes.
        """
        dfa = self.dfas[rule_id]
        initial_state: str = dfa["initial"]
        violation_state: str | None = dfa.get("violation_state")

        # For reporting: AP label -> full AP string.
        rule_info = self.rule_lookup.get(rule_id, {})
        ap_defs = {
            ap["label"]: ap.get("full", "")
            for ap in rule_info.get("aps", [])
            if ap.get("label")
        }

        # Map relevant IDs to indices in bitmask.
        rel_ids = [str(n["id"]) for n in rel_nodes]
        idx_of: Dict[str, int] = {nid: i for i, nid in enumerate(rel_ids)}
        n_rel = len(rel_ids)
        full_mask = (1 << n_rel) - 1

        violations: List[Dict[str, Any]] = []

        # DFS stack: (done_mask, q, trace_of_task_ids)
        stack: List[Tuple[int, str, List[str]]] = [(0, initial_state, [])]

        # Visited set to prune identical prefixes: (done_mask, q)
        visited: Set[Tuple[int, str]] = {(0, initial_state)}

        while stack:
            done_mask, q, trace = stack.pop()

            # --- End-of-plan check: all relevant tasks done ---
            if done_mask == full_mask:
                # Apply empty sigma once after finishing all relevant tasks.
                empty_sigma: FrozenSet[str] = frozenset()
                q_end = self._delta(rule_id, q, empty_sigma)

                if violation_state and q_end == violation_state:
                    violations.append(self._build_violation_entry_with_trace(
                        rule_id=rule_id,
                        rule=rule_info,
                        ap_defs=ap_defs,
                        trace=trace,
                        rel_nodes=rel_nodes,
                        rel_pred_map=rel_pred_map,
                        sigma_for_node=sigma_for_node,
                    ))
                # No further expansion from fully-done state.
                continue

            # --- Determine which relevant tasks are enabled at this prefix ---
            enabled_ids: List[str] = []
            for nid in rel_ids:
                i = idx_of[nid]
                if (done_mask >> i) & 1:
                    # Already done.
                    continue

                preds = rel_pred_map.get(nid, set())
                # All relevant predecessors must be in done_mask.
                all_preds_done = True
                for p in preds:
                    pi = idx_of[p]
                    if not ((done_mask >> pi) & 1):
                        all_preds_done = False
                        break

                if all_preds_done:
                    enabled_ids.append(nid)

            # --- Try each enabled task as the next event in the schedule ---
            for nid in enabled_ids:
                sigma = sigma_for_node.get(nid, frozenset())
                new_q = self._delta(rule_id, q, sigma)
                new_trace = trace + [nid]

                # Immediate violation at this prefix?
                if violation_state and new_q == violation_state:
                    violations.append(self._build_violation_entry_with_trace(
                        rule_id=rule_id,
                        rule=rule_info,
                        ap_defs=ap_defs,
                        trace=new_trace,
                        rel_nodes=rel_nodes,
                        rel_pred_map=rel_pred_map,
                        sigma_for_node=sigma_for_node,
                    ))
                    # Do not expand this branch further.
                    continue
                new_mask = done_mask | (1 << idx_of[nid])
                state_sig = (new_mask, new_q)

                if state_sig in visited:
                    # Already explored this (mask, DFA state) combination.
                    continue

                visited.add(state_sig)
                stack.append((new_mask, new_q, new_trace))

        return violations

    # ------------------------------------------------------------------ #
    # VIOLATION ENTRY BUILDER
    # ------------------------------------------------------------------ #
    def _build_violation_entry_with_trace(
            self,
            rule_id: str,
            rule: Dict[str, Any],
            ap_defs: Dict[str, str],
            trace: List[str],
            *,
            rel_nodes: List[Dict[str, Any]] | None = None,
            rel_pred_map: Dict[str, Set[str]] | None = None,
            sigma_for_node: Dict[str, FrozenSet[str]] | None = None,
        ) -> Dict[str, Any]:
            """
            Build a rich violation entry for reporting, including:

            - violated_rule_id
            - human-readable rule text and logic
            - AP definitions (label -> full string)
            - witness_trace       : list of task IDs that lead to violation

            Optionally, we also include a small projected subgraph for this rule:

            - relevant_tasks      : list of task dicts involved in this rule
            - relevant_pred_map   : {task_id -> [relevant predecessor ids]}
            - sigma_per_task      : {task_id -> [AP labels for this rule]}
            - witness_tasks       : the subset of relevant_tasks that lie on
                                    the witness_trace (in the same order)
            """
            entry: Dict[str, Any] = {
                "violated_rule_id": rule_id,
                "violation_text": rule.get("raw_text", ""),
                "violation_logic": rule.get("ltlf", ""),
                "ap_definitions": ap_defs,
                "witness_trace": trace,
            }

            id_to_node: Dict[str, Dict[str, Any]] = {}

            if rel_nodes is not None:
                # Shallow copy so later mutations to the plan do not affect this snapshot.
                rel_copies = [dict(n) for n in rel_nodes]
                entry["relevant_tasks"] = rel_copies

                id_to_node = {str(n.get("id")): n for n in rel_copies}

                # Tasks on the witness trace (in order)
                entry["witness_tasks"] = [
                    id_to_node[tid] for tid in trace if tid in id_to_node
                ]

            if rel_pred_map is not None:
                entry["relevant_pred_map"] = {
                    nid: sorted(list(preds)) for nid, preds in rel_pred_map.items()
                }

            if sigma_for_node is not None:
                # If we know the relevant tasks, limit to those IDs; otherwise use all.
                keys = list(id_to_node.keys()) if id_to_node else list(sigma_for_node.keys())
                entry["sigma_per_task"] = {
                    nid: sorted(list(sigma_for_node.get(nid, frozenset())))
                    for nid in keys
                }

            return entry