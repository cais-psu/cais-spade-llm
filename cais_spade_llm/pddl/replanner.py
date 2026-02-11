"""
Generic LLM-powered PDDL replanner.

Pipeline:
    1. Build prompt with failure context, system state, tools catalog, safety rules
    2. LLM generates PDDL domain + problem (two ```pddl code blocks)
    3. pyperplan BFS solves the problem
    4. Deterministic translator converts PDDL actions to task DAG patches

Returns None on any failure so the caller can fall back to pure-LLM replan.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Awaitable, Callable

from cais_spade_llm.pddl.solver import PlannerNoSolutionError, solve

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def replan(
    *,
    ask_llm: Callable[..., Awaitable[str]],
    violations: list[dict],
    system_state: dict[str, Any],
    tools_catalog: list[dict],
    resource_infos: list[dict],
    safety_text: str,
    failed_plan_nodes: list[dict],
    product_jid: str,
    existing_task_ids: set[str],
) -> dict[str, Any] | None:
    """
    Attempt PDDL-based replanning for manufacturing recovery.

    Returns {"tasks": [...]} matching the format expected by
    process_planner._apply_replan_patch(), or None if PDDL cannot solve it.
    """
    # 1. Build prompt
    prompt = _build_pddl_prompt(
        violations=violations,
        system_state=system_state,
        tools_catalog=tools_catalog,
        resource_infos=resource_infos,
        safety_text=safety_text,
        failed_plan_nodes=failed_plan_nodes,
    )

    # 2. LLM call
    try:
        raw_response = await ask_llm(
            prompt=prompt,
            with_functions=False,
            temperature=0.0,
        )
    except Exception:
        logger.exception("[PDDL] LLM call failed")
        return None

    # 3. Extract PDDL blocks
    domain_pddl, problem_pddl = _parse_pddl_blocks(raw_response)
    if domain_pddl is None or problem_pddl is None:
        logger.warning("[PDDL] Could not extract domain+problem from LLM response")
        return None

    # 4. Solve
    try:
        plan = solve(domain_pddl, problem_pddl)
    except PlannerNoSolutionError:
        logger.warning("[PDDL] Planner found no solution")
        return None
    except Exception:
        logger.exception("[PDDL] Planner crashed")
        return None

    if not plan:
        logger.warning("[PDDL] Empty plan returned")
        return None

    # 5. Translate
    try:
        patch = _translate_plan(
            pddl_plan=plan,
            tools_catalog=tools_catalog,
            resource_infos=resource_infos,
            product_jid=product_jid,
            existing_task_ids=existing_task_ids,
            failed_plan_nodes=failed_plan_nodes,
        )
    except Exception:
        logger.exception("[PDDL] Translation failed")
        return None

    if not patch.get("tasks"):
        logger.warning("[PDDL] Translation produced no tasks")
        return None

    logger.info(
        "[PDDL] Recovery plan: %d tasks",
        len(patch["tasks"]),
    )
    return patch


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_PDDL_SYSTEM_PROMPT = """\
You are a PDDL planning expert for manufacturing recovery.

You receive:
1. A failure description (what went wrong)
2. The current system state (robot/printer/CNC states, part locations)
3. A tools catalog (available functions with in_state/out_state transitions)
4. Resource capabilities (workspace boundaries, reachability, staging areas)
5. Safety constraints (ordering rules, etc.)
6. The current plan (failed/pending tasks)

YOUR JOB:
Generate a PDDL domain and problem that, when solved, produces a recovery
plan to complete the remaining manufacturing goals despite the failure.

CRITICAL PDDL CONSTRAINTS (pyperplan compatibility):
- Requirements: (:requirements :strips :typing :negative-preconditions)
- NO conditional effects (when ...)
- NO action costs or :functions
- NO equality predicates (= ...)
- NO existential/universal quantification (exists/forall)
- Types: resource, part, location (all subtypes of object)

NAMING CONVENTIONS:
- Action names use hyphens matching tools catalog function names:
  move_to_pick_location -> move-to-pick-location
- Object names: lowercase, hyphens for separators
  ur5e@localhost -> ur5e-localhost
  assembly_board-v1 -> assembly-board-v1
- Each PDDL action corresponds to exactly ONE function in the tools catalog

ACTION DESIGN:
- Parameters: (?r - resource ?p - part ?l - location) or a subset
- Preconditions encode: resource availability, current state, reachability
- Effects encode: state transitions matching in_state -> out_state from tools catalog
- Include safety ordering predicates where needed (e.g. placement-allowed)
- Mark unavailable/broken resources by omitting their (resource-available ?r) fact

PROBLEM DESIGN:
- Include ONLY objects relevant to the recovery (not the entire system)
- Init: current robot states, part locations, reachability facts, safety flags
- Goal: the manufacturing goals that still need to be achieved
- Only include reachability for locations a resource can actually reach

OUTPUT FORMAT:
Return exactly TWO fenced code blocks with ```pddl:
1. First block: the PDDL domain
2. Second block: the PDDL problem

Do NOT include any text outside the two ```pddl blocks.
Do NOT include comments inside the PDDL."""


def _build_pddl_prompt(
    *,
    violations: list[dict],
    system_state: dict[str, Any],
    tools_catalog: list[dict],
    resource_infos: list[dict],
    safety_text: str,
    failed_plan_nodes: list[dict],
) -> str:
    """Assemble the full prompt for LLM-based PDDL generation."""

    # Format violations
    violation_lines: list[str] = []
    for v in violations:
        fc = v.get("failure_context") or {}
        violation_lines.append(
            f"- Failed task: {v.get('failed_task_id')}\n"
            f"  Type: {v.get('type')}\n"
            f"  Failure mode: {fc.get('failure_mode', 'unknown')}\n"
            f"  Severity: {fc.get('severity', 'unknown')}\n"
            f"  Retryable: {fc.get('retryable', False)}\n"
            f"  Affected: {json.dumps(fc.get('affected_entities', []))}\n"
            f"  Observations: {json.dumps(fc.get('observations', {}), indent=2)}"
        )

    # Format robot/resource states
    robots = system_state.get("robots") or {}
    robot_lines: list[str] = []
    for jid, rs in robots.items():
        held = rs.get("held_part") or "nothing"
        state = rs.get("current_state", "unknown")
        pos = rs.get("position")
        line = f"  {jid}: state={state}, holding={held}"
        if pos:
            line += f", position=({pos['x']},{pos['y']},{pos['z']})"
        robot_lines.append(line)

    # Format part states
    parts = system_state.get("parts") or {}
    part_lines: list[str] = []
    for pname, ps in parts.items():
        pstate = ps.get("state", "unknown")
        pos = ps.get("position")
        loc = ps.get("location") or ps.get("last_known_location")
        line = f"  {pname}: state={pstate}"
        if pos:
            line += f", position=({pos['x']},{pos['y']},{pos['z']})"
        if loc:
            line += f", location={loc}"
        part_lines.append(line)

    return f"""{_PDDL_SYSTEM_PROMPT}

=== FAILURE CONTEXT ===
{chr(10).join(violation_lines) if violation_lines else "(none)"}

=== CURRENT SYSTEM STATE ===
Resources:
{chr(10).join(robot_lines) if robot_lines else "  (none)"}

Parts:
{chr(10).join(part_lines) if part_lines else "  (none)"}

=== TOOLS CATALOG ===
{json.dumps(tools_catalog, indent=2)}

=== RESOURCE CAPABILITIES ===
{json.dumps(resource_infos, indent=2)}

=== SAFETY CONSTRAINTS ===
{safety_text.strip() if safety_text and safety_text.strip() else "(none)"}

=== FAILED/PENDING PLAN NODES ===
{json.dumps(failed_plan_nodes, indent=2)}
"""


# ---------------------------------------------------------------------------
# PDDL block parser
# ---------------------------------------------------------------------------

def _parse_pddl_blocks(
    response: str,
) -> tuple[str | None, str | None]:
    """
    Extract domain and problem PDDL strings from an LLM response.

    Looks for ```pddl fenced code blocks. Falls back to bare (define ...)
    blocks if fences are missing.
    """
    # Try fenced blocks first
    pattern = r"```(?:pddl|PDDL|lisp)?\s*\n(.*?)```"
    blocks = re.findall(pattern, response, re.DOTALL)

    if len(blocks) < 2:
        # Fallback: bare (define ...) blocks
        define_pattern = r"(\(define\s+\((?:domain|problem)\b.*?\n\))"
        blocks = re.findall(define_pattern, response, re.DOTALL)

    domain_str: str | None = None
    problem_str: str | None = None

    for block in blocks:
        text = block.strip()
        if "(define (domain" in text and domain_str is None:
            domain_str = text
        elif "(define (problem" in text and problem_str is None:
            problem_str = text

    if domain_str is None or problem_str is None:
        logger.warning(
            "[PDDL] Extracted %d blocks; domain=%s, problem=%s",
            len(blocks),
            domain_str is not None,
            problem_str is not None,
        )

    return domain_str, problem_str


# ---------------------------------------------------------------------------
# Plan translator
# ---------------------------------------------------------------------------

def _short_id() -> str:
    return uuid.uuid4().hex[:6].upper()


def _unique_id(prefix: str, existing: set[str]) -> str:
    candidate = f"{prefix}_{_short_id()}"
    while candidate in existing:
        candidate = f"{prefix}_{_short_id()}"
    return candidate


def _pddl_name(s: str) -> str:
    """Convert a real name to its PDDL equivalent (lowercase, hyphens)."""
    return s.lower().replace("_", "-").replace("@", "-").replace(".", "-")


def _translate_plan(
    *,
    pddl_plan: list[tuple[str, list[str]]],
    tools_catalog: list[dict],
    resource_infos: list[dict],
    product_jid: str,
    existing_task_ids: set[str],
    failed_plan_nodes: list[dict],
) -> dict[str, Any]:
    """
    Convert a PDDL plan into task DAG patches.

    Translation is driven entirely by tools_catalog:
    - PDDL action name (hyphens) → function_name (underscores) via catalog lookup
    - PDDL parameters → params dict using catalog's param schema
    - PDDL object names → real JIDs/names via reverse name maps

    Returns {"tasks": [...]} matching process_planner merge format.
    """
    # Build function lookup: hyphenated-name → catalog entry
    func_lookup: dict[str, dict] = {}
    for entry in tools_catalog:
        fn = entry.get("function")
        if fn:
            func_lookup[fn.replace("_", "-")] = entry

    # Build reverse name maps
    resource_jid_map: dict[str, str] = {}
    for ri in resource_infos:
        jid = ri.get("jid", "")
        resource_jid_map[_pddl_name(jid)] = jid

    part_name_map: dict[str, str] = {}
    for node in failed_plan_nodes:
        pn = (node.get("params") or {}).get("part_name")
        if pn:
            part_name_map[_pddl_name(pn)] = pn
    # Also pull from system state parts if available in plan nodes
    parts_state = {}
    for node in failed_plan_nodes:
        params = node.get("params") or {}
        for key in ("part_name",):
            val = params.get(key)
            if val:
                part_name_map[_pddl_name(val)] = val

    location_map: dict[str, str] = {}
    for ri in resource_infos:
        caps = ri.get("static_capabilities") or {}
        for loc in caps.get("reachability", []):
            location_map[_pddl_name(loc)] = loc
        for zone_name in (caps.get("staging_areas") or {}).keys():
            location_map[_pddl_name(zone_name)] = zone_name

    # Translate each PDDL action to a task
    tasks: list[dict] = []
    prev_id: str | None = None
    ids_used = set(existing_task_ids)

    for action_name, params in pddl_plan:
        catalog_entry = func_lookup.get(action_name)
        if catalog_entry is None:
            logger.warning("[PDDL] Unknown action '%s' — skipping", action_name)
            continue

        function_name = catalog_entry["function"]

        # Standard PDDL parameter positions: (?r, ?p, ?l)
        robot_pddl = params[0] if len(params) > 0 else ""
        part_pddl = params[1] if len(params) > 1 else ""
        loc_pddl = params[2] if len(params) > 2 else ""

        robot_jid = resource_jid_map.get(robot_pddl, robot_pddl)
        part_name = part_name_map.get(part_pddl, part_pddl.upper())
        location = location_map.get(loc_pddl, loc_pddl.replace("-", "_"))

        task_id = _unique_id(f"RECOVERY_{action_name.upper()}", ids_used)
        ids_used.add(task_id)

        # Build params from catalog's param schema
        task_params: dict[str, Any] = {
            "product_jid": product_jid,
            "task_id": task_id,
            "part_name": part_name,
        }

        param_schema = catalog_entry.get("params") or {}
        if "origin_resource_location" in param_schema:
            task_params["origin_resource_location"] = location
        if "destination_location" in param_schema:
            task_params["destination_location"] = location

        task: dict[str, Any] = {
            "id": task_id,
            "function_name": function_name,
            "params": task_params,
            "resource_jid": robot_jid,
            "predecessors": [prev_id] if prev_id else [],
            "successors": [],
            "change_reason": (
                f"INSERTION: PDDL recovery — {function_name}("
                f"{part_name}, {location}) on {robot_jid}"
            ),
        }
        tasks.append(task)
        prev_id = task_id

    # Wire predecessor chain for successors
    for i in range(len(tasks) - 1):
        tasks[i]["successors"] = [tasks[i + 1]["id"]]

    # Rewire blocked downstream tasks: if any pending task had a failed
    # predecessor, point it at the recovery tail instead
    if tasks:
        failed_ids = {
            n["id"]
            for n in failed_plan_nodes
            if isinstance(n.get("status"), str)
            and (n["status"].startswith("failed") or n["status"] == "blocked")
        }
        recovery_tail_id = tasks[-1]["id"]

        for node in failed_plan_nodes:
            if node.get("status") != "pending":
                continue
            preds = node.get("predecessors") or []
            if any(pid in failed_ids for pid in preds):
                new_preds = [
                    p for p in preds if p not in failed_ids
                ] + [recovery_tail_id]
                tasks.append({
                    "id": node["id"],
                    "predecessors": new_preds,
                    "change_reason": (
                        f"Updated predecessors: replaced failed dependency "
                        f"with recovery tail {recovery_tail_id}"
                    ),
                })

    return {"tasks": tasks}
