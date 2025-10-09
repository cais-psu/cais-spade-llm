# agent_creator.py  (SPADE-ready, concise)
from __future__ import annotations
import os, json
from pathlib import Path
from typing import Any, Iterable
from collections import defaultdict

import utils
from rag_handler import build_agent_vector_store
from function_analyzer import FunctionAnalyzer  # only if you use it here

from agents.shared_information.user import User
from agents.intelligent_product.product_agent import ProductAgent
from agents.resource_agent.printing_agent import PrintingAgent
from agents.resource_agent.robot_agent import RobotAgent

ALLOWED_FUNCS: dict[str, set[str]] = defaultdict(set)

# ---------- helpers ----------
def _as_list(x): return x if isinstance(x, (list, tuple)) else ([x] if x else [])

def _vec_if(paths: Any, dest: str):
    paths = _as_list(paths)
    return build_agent_vector_store(doc_paths=paths, persist_dir=dest) if paths else None

def _cad_vecs_if(paths: Any, dest_root: str):
    paths = _as_list(paths)
    if not paths: return None
    out = {}
    for p in paths:
        stem = Path(p).stem
        out[stem] = build_agent_vector_store(doc_paths=[p], persist_dir=f"{dest_root}/{stem}")
    return out

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

            # vector stores (optional)
            vecs = {}
            v = _vec_if(meta.get("capabilities_file"), f"chromadb/{name}/manual")
            if v: vecs["manual"] = v
            v = _vec_if(meta.get("material_file"), f"chromadb/{name}/material")
            if v: vecs["material"] = v
            v = _vec_if(meta.get("slicer_file"), f"chromadb/{name}/slicer")
            if v: vecs["slicer"] = v
            v = _vec_if(meta.get("gripper_file"), f"chromadb/{name}/gripper")
            if v: vecs["gripper"] = v

            common = dict(
                name=name,
                annotation=meta.get("annotation"),
                instructions=meta.get("instructions"),
                function_names=_fn_names(meta),
                static_capabilities=meta.get("static_capabilities"),
                vector_dbs=vecs,
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

def create_product_agents(product_init_list: Iterable[str], resource_agents: list):
    # Build lookup so targets can be names OR full JIDs
    res_lookup = {getattr(r, "agent_name", getattr(r, "name", str(r.jid))).lower(): str(r.jid) for r in resource_agents}

    agents = []
    for init_file in product_init_list:
        raw = utils.load_json_data(init_file)
        for meta in _flat_or_nested_config(raw):
            name = meta["name"]
            jid, pw = _jid_pw(meta)

            # path simplification: base_path/spec_path supported
            base = meta.get("base_path", "")
            cad_files = [str(Path(base) / f) for f in _as_list(meta.get("cad_files"))]

            # vector stores (optional)
            vecs = {}
            if cad_files:
                v = _cad_vecs_if(cad_files, f"chromadb/{name}/cad")
                if v: vecs["cad"] = v
            v = _vec_if(meta.get("product_specification_file"), f"chromadb/{name}/spec")
            if v: vecs["spec"] = v

            ALLOWED_FUNCS[name].update(_fn_names(meta))

            # targets: list of names or JIDs
            targets_conf = _as_list(meta.get("targets"))
            resource_jids = [res_lookup.get(t.lower(), t) for t in targets_conf] if targets_conf else []

            agent = ProductAgent(
                jid, pw,
                name=name,
                annotation=meta.get("annotation"),
                instructions=meta.get("instructions"),
                product_specification_file=meta.get("product_specification_file"),
                cad_files=cad_files or None,
                function_names=_fn_names(meta),
                resource_jids=resource_jids,
                vector_dbs=vecs,
            )

            # optional initial message
            inbox_msg = (meta.get("inbox") or "") + f" Your product name is `{name}`."
            if hasattr(agent, "inbox"):
                agent.inbox.append([("user", inbox_msg)])

            agents.append(agent)
    return agents
