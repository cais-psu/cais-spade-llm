# agent_creator.py  (SPADE-ready, concise)
from __future__ import annotations
import os, json
from pathlib import Path
from typing import Iterable, List
from collections import defaultdict

import utils

from agents.shared_information.user import User
from agents.intelligent_product.product_agent import ProductAgent
from agents.resource_agent.printing_agent import PrintingAgent
from agents.resource_agent.robot_agent import RobotAgent

ALLOWED_FUNCS: dict[str, set[str]] = defaultdict(set)

# ---------- helpers ----------
def _as_list(x): return x if isinstance(x, (list, tuple)) else ([x] if x else [])

def _flat_or_nested_config(obj: dict) -> list[dict]:
    # Accept {"name": {...}} OR flat {"name": "...", ...}
    if "name" in obj: return [obj]
    # nested → expand to list with injected name
    return [{**v, "name": k} for k, v in obj.items()]

def _jid_pw(meta: dict, default_domain="localhost", default_pw="none"):
    jid = meta.get("jid") or f'{meta["name"]}@{meta.get("domain", default_domain)}'
    pw  = meta.get("password", default_pw)
    return jid, pw

def _fn_names(meta: dict) -> list[str]:
    f = meta.get("function_names", meta.get("functions", []))
    return list(dict.fromkeys(f))  # dedupe, keep order

# ---------- public API ----------
def create_user():
    # Return a SPADE user agent if you have one; else None is fine.
    try:
        return User("user@localhost", "none")
    except Exception:
        return None

def create_resource_agents(resource_init_list: Iterable[str]):
    agents = []
    for init_file in resource_init_list:
        raw = utils.load_json_data(init_file)
        for meta in _flat_or_nested_config(raw):
            name = meta["name"]
            kind = (meta.get("type") or "").lower()
            jid, pw = _jid_pw(meta)

            ALLOWED_FUNCS[name].update(_fn_names(meta))

            common = dict(
                name=name,
                annotation=meta.get("annotation"),
                instructions=meta.get("instructions"),
                function_names=_fn_names(meta),
                static_capabilities=meta.get("static_capabilities")
            )

            if kind == "printing":
                agent = PrintingAgent(jid, pw, **common)
            elif kind == "robot":
                agent = RobotAgent(jid, pw, **common)
            else:
                print(f"[WARN] Unknown resource type '{kind}' for {name} in {init_file}; skipped.")
                continue

            agents.append(agent)
    return agents


def create_product_agents(product_init_list: Iterable[str], resource_agents: list) -> List[ProductAgent]:
    """
    Build ProductAgent instances.
    If `targets` is omitted in JSON, auto-use *all* provided resource agents.
    """
    # Map names→JIDs for optional explicit target resolution
    res_lookup = {
        getattr(r, "agent_name", getattr(r, "name", str(r.jid))).lower(): str(r.jid)
        for r in resource_agents
    }
    # JIDs for all RAs (default when no targets specified)
    all_ra_jids = [str(r.jid) for r in resource_agents]

    agents: List[ProductAgent] = []
    for init_file in product_init_list:
        raw = utils.load_json_data(init_file)
        for meta in _flat_or_nested_config(raw):
            name = meta["name"]
            jid, pw = _jid_pw(meta)

            fn_names = _fn_names(meta)
            ALLOWED_FUNCS[name].update(fn_names)

            # If JSON has targets, resolve; else use ALL resource agents
            targets_conf = _as_list(meta.get("targets"))
            if targets_conf:
                resource_jids = [res_lookup.get(t.lower(), t) for t in targets_conf]
            else:
                resource_jids = all_ra_jids[:]   # ← no hard-coding; just use live RAs

            agent = ProductAgent(
                jid, pw,
                name=name,
                annotation=meta.get("annotation"),
                instructions=meta.get("instructions"),
                product_specification_file=meta.get("product_specification_file"),
                function_names=fn_names,
                resource_jids=resource_jids,
                broadcast=bool(meta.get("broadcast", False)),
            )

            if hasattr(agent, "inbox"):
                agent.inbox.append([("user", (meta.get("inbox") or "") + f" Your product name is `{name}`.")])

            agents.append(agent)
    return agents
