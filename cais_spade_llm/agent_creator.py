# agent_creator.py  (SPADE-ready, concise)
"""Factory helpers that ingest JSON manifests and spawn SPADE agents with the right wiring."""

from __future__ import annotations

import logging
import os
import threading
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from cais_spade_llm import utils

_log = logging.getLogger(__name__)
_REQ_DIR = Path("cais_spade_llm/specification/products/requirements")
_ORDER_DIR = Path("cais_spade_llm/specification/products/orders")

from cais_spade_llm.agents.central_controller.central_controller_agent import CentralControllerAgent
from cais_spade_llm.agents.intelligent_product.product_agent import ProductAgent
from cais_spade_llm.agents.resource_agent.printing_agent import PrintingAgent
from cais_spade_llm.agents.resource_agent.robot_agent import RobotAgent
from cais_spade_llm.agents.shared_information.user import User
from cais_spade_llm.resources.sensor.camera_module import CameraModule

_CAMERA_LOCK = threading.Lock()
_UNSET_OVERRIDE = object()

# Environment mode: "gazebo" (default) or "real".
# Controls which sub-config block is read from robot JSON manifests.
ROBOT_ENV = os.environ.get("ROBOT_ENV", "gazebo").strip().lower()

# Execution mode override: when set, takes precedence over per-robot JSON values.
# Values: "dry_run", "simulation", "physical".
_EXECUTION_MODE_OVERRIDE = os.environ.get("EXECUTION_MODE", "").strip().lower() or None


def _normalize_camera_backend(raw: str | None) -> str:
    explicit = str(raw or "").strip().lower()
    if explicit in {"none", "mock", "gazebo_gt", "yolo"}:
        return explicit
    return "none"


def _resolve_camera_backend() -> str:
    """Choose camera backend from PERCEPTION_BACKEND env only."""
    return _normalize_camera_backend(os.environ.get("PERCEPTION_BACKEND", ""))


def _build_camera(backend: str) -> CameraModule:
    if backend == "mock":
        # Optional deterministic mock map for offline tests.
        return CameraModule(
            backend="mock",
            mock_observations={
                "SG": {"x": 0.750, "y": -0.200, "z": 0.050, "frame_id": "world"},
                "MCP": {"x": 0.400, "y": -0.100, "z": 0.050, "frame_id": "world"},
            },
        )
    return CameraModule(backend=backend)


_CAMERA_BACKEND = _resolve_camera_backend()
_CAMERA = _build_camera(_CAMERA_BACKEND)


def configure_runtime(
    *,
    robot_env: str | None = None,
    execution_mode: str | None = None,
    perception_backend: str | None = None,
) -> None:
    """Refresh runtime globals without requiring module reload."""
    global ROBOT_ENV, _EXECUTION_MODE_OVERRIDE, _CAMERA_BACKEND, _CAMERA

    if robot_env is not None:
        ROBOT_ENV = str(robot_env).strip().lower() or "gazebo"
    if execution_mode is not None:
        _EXECUTION_MODE_OVERRIDE = str(execution_mode).strip().lower() or None

    backend = (
        _normalize_camera_backend(perception_backend)
        if perception_backend is not None
        else _resolve_camera_backend()
    )
    with _CAMERA_LOCK:
        if backend == _CAMERA_BACKEND:
            return
        old_camera = _CAMERA
        _CAMERA = _build_camera(backend)
        _CAMERA_BACKEND = backend
    try:
        if old_camera is not None:
            old_camera.destroy()
    except Exception:
        pass


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
    jid = meta.get("jid") or f"{meta['name']}@{meta.get('domain', default_domain)}"
    pw = meta.get("password", default_pw)
    return jid, pw


def _fn_names(meta: dict) -> list[str]:
    """Extract and deduplicate function names while preserving the author-defined order."""
    declared = meta.get("function_names", meta.get("functions", []))
    if isinstance(declared, str):
        token = declared.strip()
        return [token] if token else []
    return list(dict.fromkeys(declared))


# ---------- public API ----------
def create_user():
    """Instantiate the optional User agent (if credentials are configured)."""
    try:
        return User("user@localhost", "none")
    except Exception:
        return None


def create_resource_agents(
    resource_init_list: Iterable[str],
    cca_init_file: str,
    prewarmed_controllers: dict | None = None,
):
    """Build resource-oriented agents (printing, robot, etc.) from their manifest files.

    Args:
        prewarmed_controllers: Optional dict mapping robot key (e.g. "ur5e", "xarm6")
            to a pre-initialized GazeboPickPlaceController that the RobotAgent can
            adopt instead of creating a new one.
    """
    prewarmed = prewarmed_controllers or {}
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

            # Resolve environment-specific config (gazebo / real).
            env_block = meta.get(ROBOT_ENV, {})
            fn_names = _fn_names(meta)
            raw_declared = meta.get("function_names", meta.get("functions", None))
            if kind == "robot" and (
                raw_declared is None
                or (isinstance(raw_declared, str) and raw_declared.strip().lower() == "auto")
            ):
                fn_names = RobotAgent.resolve_registered_function_names(
                    static_capabilities=env_block.get(
                        "static_capabilities", meta.get("static_capabilities")
                    )
                    or {},
                    named_positions=env_block.get("named_positions", {}) or {},
                    controller_config=env_block.get("controller", {}) or {},
                )
            ALLOWED_FUNCS[name].update(fn_names)

            common = dict(
                name=name,
                instructions=meta.get("instructions"),
                function_names=fn_names,
                static_capabilities=env_block.get(
                    "static_capabilities", meta.get("static_capabilities")
                ),
                cca_jid=cca_jid,  # <-- pass CCA JID into every resource agent
            )

            if kind == "printing":
                agent = PrintingAgent(jid, pw, **common)
            elif kind == "robot":
                if isinstance(meta.get("failure_scenarios"), list):
                    common["failure_scenarios"] = list(meta.get("failure_scenarios") or [])
                common["execution_mode"] = (
                    _EXECUTION_MODE_OVERRIDE
                    or str(env_block.get("execution_mode", meta.get("execution_mode", "dry_run")))
                    .strip()
                    .lower()
                )
                common["controller_config"] = env_block.get("controller", {})
                common["named_positions"] = env_block.get("named_positions", {})
                # Inject prewarmed controller if available for this robot.
                robot_key = str(name or "").split("@", 1)[0].lower()
                pw_ctrl = prewarmed.pop(robot_key, None)
                if pw_ctrl is not None:
                    common["prewarmed_controller"] = pw_ctrl
                agent = RobotAgent(jid, pw, **common)
            else:
                print(f"[WARN] Unknown resource type '{kind}' for {name} in {init_file}; skipped.")
                continue

            agents.append(agent)

    # Shut down any prewarmed controllers that were not claimed by an agent.
    for leftover_key, leftover_ctrl in prewarmed.items():
        try:
            leftover_ctrl.shutdown()
        except Exception:
            pass
    return agents


def _default_requirement_file(product_name: str | None = None) -> str | None:
    """Return the default requirement file path for a product, falling back to the first .txt file."""
    name = str(product_name or "").strip()
    if name:
        candidate = _REQ_DIR / f"{name}.txt"
        _log.info("[agent_creator] No product_specification_file set; defaulting to %s", candidate)
        return str(candidate)
    if _REQ_DIR.is_dir():
        files = sorted(_REQ_DIR.glob("*.txt"))
        if files:
            _log.info("[agent_creator] No product_specification_file set; using %s", files[0])
            return str(files[0])
    return None


def _default_product_order_file(product_name: str | None = None) -> str | None:
    """Return the default product-order JSON path for a product."""
    name = str(product_name or "").strip()
    if name:
        candidate = _ORDER_DIR / f"{name}.json"
        _log.info("[agent_creator] No product_order_file set; defaulting to %s", candidate)
        return str(candidate)
    if _ORDER_DIR.is_dir():
        files = sorted(_ORDER_DIR.glob("*.json"))
        if files:
            _log.info("[agent_creator] No product_order_file set; using %s", files[0])
            return str(files[0])
    return None


def create_product_agents(
    product_init_list: Iterable[str],
    resource_agents: list,
    cca_init_file: str,
    bundle_context: dict | None = None,
    product_requirement_file: str | None = None,
    product_order_file: str | None = None,
    safety_file_override: object = _UNSET_OVERRIDE,
) -> list[ProductAgent]:
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

    agents: list[ProductAgent] = []
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

            product_bundle = None
            if bundle_context:
                b_product_name = str(bundle_context.get("product_name", "")).strip()
                b_spec_file = str(bundle_context.get("product_spec_file", "")).strip()
                resolved_spec_file = str(
                    product_requirement_file or meta.get("product_specification_file", "")
                ).strip()
                m_spec_file = resolved_spec_file

                name_match = not b_product_name or b_product_name == str(name)
                spec_match = True
                if b_spec_file and m_spec_file:
                    spec_match = (
                        Path(b_spec_file).resolve().as_posix()
                        == Path(m_spec_file).resolve().as_posix()
                    )

                if name_match and spec_match:
                    product_bundle = bundle_context

            agent = ProductAgent(
                jid,
                pw,
                name=name,
                instructions=meta.get("instructions"),
                product_order_file=(
                    product_order_file
                    or meta.get("product_order_file")
                    or _default_product_order_file(name)
                ),
                product_specification_file=(
                    product_requirement_file or meta.get("product_specification_file") or None
                ),
                product_geometry_file=meta.get("product_geometry_file"),
                safety_file=(
                    (meta.get("safety_file") or cca_meta.get("safety_file"))
                    if safety_file_override is _UNSET_OVERRIDE
                    else safety_file_override
                ),
                function_names=fn_names,
                resource_jids=resource_jids,
                resource_agents=resource_agents,
                cca_jid=cca_jid,
                camera=_CAMERA,
                precomputed_bundle=product_bundle,
            )

            # Seed the inbox with optional canned messages so the user agent can demo interactions.
            if hasattr(agent, "inbox"):
                agent.inbox.append(
                    [
                        (
                            "user",
                            (meta.get("inbox") or "") + f" Your product name is `{name}`.",
                        )
                    ]
                )

            agents.append(agent)
    return agents


def create_central_controller(
    cca_init_file: str,
    resource_agents: list | None = None,
    bundle_context: dict | None = None,
    safety_file_override: object = _UNSET_OVERRIDE,
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

    name = meta.get("name", "cca")  # default agent name = "cca"
    jid = meta.get("jid")
    password = meta.get("password")

    if not jid or not password:
        raise ValueError("CCA must have 'jid' and 'password' fields.")

    safety_file = (
        meta.get("safety_file") if safety_file_override is _UNSET_OVERRIDE else safety_file_override
    )

    controller = CentralControllerAgent(
        jid,
        password,
        name=name,
        instructions=meta.get("instructions", None),
        safety_file=safety_file,
        resource_agents=resource_agents,
        precomputed_bundle=bundle_context,
    )

    return controller
