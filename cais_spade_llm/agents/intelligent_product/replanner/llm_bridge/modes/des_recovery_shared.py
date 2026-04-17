"""Active shared helpers for the DES recovery engine."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

from cais_spade_llm.agents.central_controller.base_safety_checker import (
    BaseSafetyChecker,
)

_logger = logging.getLogger(__name__)


def extract_safety_dfas(
    planner: Any,
    prepared_bridge_request: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Extract loaded safety DFA rules from planner or llm_input context."""
    product_agent = getattr(planner, "product_agent", None)
    cca = None
    if product_agent is not None:
        cca = getattr(product_agent, "cca_agent", None) or getattr(product_agent, "_cca", None)
    for owner in (planner, product_agent, cca):
        if owner is None:
            continue
        for attr_name in (
            "plan_safety_validator",
            "safety_checker",
            "online_safety_monitor",
            "online_safety_supervisor",
            "safety_monitor",
            "safety",
        ):
            safety_checker = getattr(owner, attr_name, None)
            if safety_checker is None:
                continue
            dfas = getattr(safety_checker, "dfas", None)
            if isinstance(dfas, dict) and dfas:
                return deepcopy(dfas)

    llm_input = dict(prepared_bridge_request.get("llm_input") or {})
    rules = llm_input.get("loaded_safety_rules") or []
    dfas: dict[str, dict[str, Any]] = {}
    dfa_dots: dict[str, str] = {}
    dot_rules: list[dict[str, Any]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        rule_id = str(rule.get("id") or rule.get("rule_id") or "").strip()
        dfa_data = rule.get("dfa")
        if rule_id and isinstance(dfa_data, dict):
            dfas[rule_id] = deepcopy(dfa_data)
            continue
        if rule_id and isinstance(dfa_data, str) and dfa_data.strip():
            dfa_dots[rule_id] = dfa_data
            dot_rules.append(rule)
            continue
        dfa_dot = str(rule.get("dfa_dot") or rule.get("dot") or "").strip()
        if rule_id and dfa_dot:
            dfa_dots[rule_id] = dfa_dot
            dot_rules.append(rule)
    if dfa_dots:
        try:
            parsed = BaseSafetyChecker(dfa_dots, dot_rules).dfas
            for rule_id, dfa in parsed.items():
                if rule_id and isinstance(dfa, dict):
                    dfas.setdefault(rule_id, deepcopy(dfa))
        except Exception as exc:
            _logger.warning("[DESRecovery] Failed to parse safety DFA DOT sources: %s", exc)
    return dfas


def _parse_ap_full(full: Any) -> dict[str, str]:
    token = str(full or "").strip()
    segments = [segment.strip() for segment in token.split("/") if segment.strip()]
    if len(segments) < 5:
        return {}
    prefix = segments[0]
    if prefix not in {"ap", "ap_event"}:
        return {}
    return {
        "prefix": prefix,
        "process": segments[1] if len(segments) > 1 else "",
        "product": segments[2] if len(segments) > 2 else "",
        "resource": segments[3] if len(segments) > 3 else "",
        "function_name": segments[4] if len(segments) > 4 else "",
        "context": segments[5] if len(segments) > 5 else "",
    }


def _normalize_ap_descriptor(raw_ap: dict[str, Any], *, label: str | None = None) -> dict[str, Any] | None:
    if not isinstance(raw_ap, dict):
        return None
    entry = deepcopy(raw_ap)
    ap_label = str(label or entry.get("label") or entry.get("ap_label") or "").strip()
    if not ap_label:
        return None
    entry["label"] = ap_label

    parsed = _parse_ap_full(entry.get("full"))
    selector = dict(entry.get("selector") or {})
    for target_key, source_key in (
        ("resource", "resource"),
        ("product", "product"),
        ("part", "product"),
        ("function_name", "function_name"),
        ("function", "function_name"),
        ("context", "context"),
    ):
        if str(entry.get(target_key) or "").strip():
            continue
        value = str(parsed.get(source_key) or selector.get(target_key) or "").strip()
        if value:
            entry[target_key] = value
    if not str(entry.get("context") or "").strip():
        destination = str(selector.get("destination") or "").strip()
        if destination:
            entry["context"] = destination
    if not str(entry.get("full") or "").strip() and not any(
        str(entry.get(key) or "").strip()
        for key in ("resource", "product", "part", "function_name", "function")
    ):
        return None
    return entry


def extract_ap_descriptors(
    planner: Any,
    prepared_bridge_request: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract AP descriptor list from the safety-rule context."""
    del planner
    llm_input = dict(prepared_bridge_request.get("llm_input") or {})
    rules = llm_input.get("loaded_safety_rules") or []
    descriptors: list[dict[str, Any]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        ap_defs = rule.get("bridge_aps") or rule.get("ap_definitions") or rule.get("aps") or []
        if isinstance(ap_defs, list):
            for raw_ap in ap_defs:
                normalized = _normalize_ap_descriptor(raw_ap)
                if normalized:
                    descriptors.append(normalized)
        elif isinstance(ap_defs, dict):
            for label, desc in ap_defs.items():
                if not isinstance(desc, dict):
                    continue
                normalized = _normalize_ap_descriptor(desc, label=str(label))
                if normalized:
                    descriptors.append(normalized)
    return descriptors


__all__ = [
    "extract_ap_descriptors",
    "extract_safety_dfas",
]
