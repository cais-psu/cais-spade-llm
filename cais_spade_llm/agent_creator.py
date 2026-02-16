# agent_creator.py  (SPADE-ready, concise)
"""Factory helpers that ingest JSON manifests and spawn SPADE agents with the right wiring."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, List, Optional

import utils

from agents.shared_information.user import User
from agents.intelligent_product.product_agent import ProductAgent
from agents.resource_agent.printing_agent import PrintingAgent
from agents.resource_agent.robot_agent import RobotAgent
from agents.central_controller.central_controller_agent import CentralControllerAgent
from sensors.camera_module import CameraModule

# Mock camera observations for simulation.
# SG: slipped into ur5e-only territory (x=750 > xarm6 upper bound of 650).
# MCP: correctly placed at assembly board position.
# Workspace boundaries (mm): xarm6 x=[-150,650], ur5e x=[100,900].
_MOCK_CAMERA = CameraModule(mock_observations={
    "SG":  {"x": 750.0, "y": -200.0, "z": 50.0},
    "MCP": {"x": 400.0, "y": -100.0, "z": 50.0},
})

# FunctionAnalyzer consults this registry to decide which methods each agent is allowed to expose.
ALLOWED_FUNCS: dict[str, set[str]] = defaultdict(set)


# ---------- helpers ----------
def _as_list(x):
    """Normalize optional values to a list so manifests can specify scalars or arrays."""
    return x if isinstance(x, (list, tuple)) else ([x] if x else [])


def _flat_or_nested_config(obj: dict) -> list[dict]:
    """Support both flat entries and nested {name: {...}} sections by yielding a homogenous list."""
    if "name" in obj:
        return [obj]
    return [{**meta, "name": name} for name, meta in obj.items()]


def _jid_pw(meta: dict, default_domain="localhost", default_pw="none"):
    """Derive SPADE credentials, falling back to name@domain plus a default password."""
    jid = meta.get("jid") or f'{meta["name"]}@{meta.get("domain", default_domain)}'
    pw = meta.get("password", default_pw)
    return jid, pw


def _fn_names(meta: dict) -> list[str]:
    """Extract and deduplicate function names while preserving the author-defined order."""
    declared = meta.get("function_names", meta.get("functions", []))
    return list(dict.fromkeys(declared))


# ---------- public API ----------
def create_user():
    """Instantiate the optional User agent (if credentials are configured)."""
    try:
        return User("user@localhost", "none")
    except Exception:
        return None

def create_resource_agents(resource_init_list: Iterable[str], cca_init_file: str):
    """Build resource-oriented agents (printing, robot, etc.) from their manifest files."""
    cca_config = utils.load_json_data(cca_init_file) or {}
    cca_meta = cca_config.get("cca", cca_config)  # handle {"cca": {...}} or flat
    cca_jid = cca_meta.get("jid")

    agents = []
    for init_file in resource_init_list:
        raw = utils.load_json_data(init_file)
        for meta in _flat_or_nested_config(raw):
            name = meta["name"]
            kind = (meta.get("type") or "").lower()
            jid, pw = _jid_pw(meta)

            fn_names = _fn_names(meta)
            ALLOWED_FUNCS[name].update(fn_names)

            common = dict(
                name=name,
                instructions=meta.get("instructions"),
                function_names=fn_names,
                static_capabilities=meta.get("static_capabilities"),
                cca_jid=cca_jid,  # <-- pass CCA JID into every resource agent
            )

            if kind == "printing":
                agent = PrintingAgent(jid, pw, **common)
            elif kind == "robot":
                if "sg_slippage_mode" in meta:
                    common["sg_slippage_mode"] = meta.get("sg_slippage_mode")
                if "sg_slippage_scope" in meta:
                    common["sg_slippage_scope"] = meta.get("sg_slippage_scope")
                agent = RobotAgent(jid, pw, **common)
            else:
                print(
                    f"[WARN] Unknown resource type '{kind}' for {name} in {init_file}; skipped."
                )
                continue

            agents.append(agent)
    return agents

def create_product_agents(
    product_init_list: Iterable[str], resource_agents: list, cca_init_file: str,
) -> List[ProductAgent]:
    """
    Build ProductAgent instances from JSON manifests.

    If a product omits `targets`, it automatically receives every available resource agent.
    """
    cca_config = utils.load_json_data(cca_init_file) or {}
    cca_meta = cca_config.get("cca", cca_config)  # handle {"cca": {...}} or flat
    cca_jid = cca_meta.get("jid")

    # Map agent-name -> JID so manifests can reference either friendly names or raw addresses.
    res_lookup = {
        getattr(r, "agent_name", getattr(r, "name", str(r.jid))).lower(): str(r.jid)
        for r in resource_agents
    }
    # JIDs for all resource agents; used when a product doesn't specify explicit targets.
    all_ra_jids = [str(r.jid) for r in resource_agents]

    agents: List[ProductAgent] = []
    for init_file in product_init_list:
        raw = utils.load_json_data(init_file)
        for meta in _flat_or_nested_config(raw):
            name = meta["name"]
            jid, pw = _jid_pw(meta)

            fn_names = _fn_names(meta)
            ALLOWED_FUNCS[name].update(fn_names)

            targets_conf = _as_list(meta.get("targets"))
            if targets_conf:
                # Resolve manifest targets through the lookup, falling back to the raw string.
                resource_jids = [res_lookup.get(t.lower(), t) for t in targets_conf]
            else:
                resource_jids = all_ra_jids[:]  # Use a copy so later mutation is safe.

            agent = ProductAgent(
                jid,
                pw,
                name=name,
                instructions=meta.get("instructions"),
                product_specification_file=meta.get("product_specification_file"),
                safety_file=meta.get("safety_file"),
                function_names=fn_names,
                resource_jids=resource_jids,
                resource_agents=resource_agents,
                cca_jid=cca_jid,
                camera=_MOCK_CAMERA,
                replan_mode=meta.get("replan_mode", "llm"),
            )

            # Seed the inbox with optional canned messages so the user agent can demo interactions.
            if hasattr(agent, "inbox"):
                agent.inbox.append(
                    [
                        (
                            "user",
                            (meta.get("inbox") or "")
                            + f" Your product name is `{name}`.",
                        )
                    ]
                )

            agents.append(agent)
    return agents


def create_central_controller(
    cca_init_file: str,
    resource_agents: list | None = None,
) -> CentralControllerAgent:
    """
    Build exactly ONE CentralControllerAgent from a JSON manifest.

    Expected JSON format:

    {
      "cca": {
        "type": "cca",
        "jid": "cca@localhost",
        "password": "none",
        "domain": "localhost",
        "instructions": " ... ",
        "safety_file": "cais_spade_llm/specification/cca/safety_requirements.txt"
      }
    }
    """

    data = utils.load_json_data(cca_init_file)

    if "cca" not in data:
        raise ValueError(f"No 'cca' section found in {cca_init_file}.")

    meta = data["cca"]

    # Minimal validation
    if (meta.get("type") or "").lower() != "cca":
        raise ValueError("The 'cca' object must have type='cca'")

    name = meta.get("name", "cca")   # default agent name = "cca"
    jid = meta.get("jid")
    password = meta.get("password")

    if not jid or not password:
        raise ValueError("CCA must have 'jid' and 'password' fields.")

    safety_file = meta.get("safety_file")

    controller = CentralControllerAgent(
        jid,
        password,
        name=name,
        instructions=meta.get("instructions", None),
        safety_file=safety_file,
        resource_agents=resource_agents,
    )

    return controller
