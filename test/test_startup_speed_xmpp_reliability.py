import asyncio
import base64
import importlib
import json
import os
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from spade.message import Message

os.environ.setdefault("OPENAI_API_KEY", "test")

from cais_spade_llm.ui import bridge as bridge_module
from cais_spade_llm.ui.bridge import SystemBridge
from cais_spade_llm.agents.central_controller.central_controller_agent import CentralControllerAgent
from cais_spade_llm.agents.intelligent_product.product_agent import ProductAgent
from cais_spade_llm.agents.intelligent_product.process_planner import ProcessPlanner
from cais_spade_llm.agents.resource_agent.resource_agent import ResourceAgent
from cais_spade_llm.agents.shared_information.local_dispatch import send_agent_message


def test_ensure_xmpp_server_serializes_concurrent_prewarm(monkeypatch) -> None:
    bridge = SystemBridge.__new__(SystemBridge)
    bridge._xmpp_proc = None
    bridge._xmpp_host = "127.0.0.1"
    bridge._xmpp_port = 5222
    bridge._xmpp_start_lock = None
    bridge._xmpp_start_lock_loop = None
    bridge._diag_emit = lambda message: None
    bridge._tcp_port_open = lambda host, port, timeout_sec=0.25: False

    calls = {"spawn": 0, "wait": 0}

    class FakeProc:
        pid = 4242

        def poll(self):
            return None

    def fake_popen(*args, **kwargs):
        calls["spawn"] += 1
        return FakeProc()

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    async def fake_wait_for_xmpp_ready(timeout_sec=45.0):
        calls["wait"] += 1
        await asyncio.sleep(0.01)

    monkeypatch.setattr(bridge_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(bridge_module.asyncio, "to_thread", fake_to_thread)
    bridge._wait_for_xmpp_ready = fake_wait_for_xmpp_ready

    async def run_concurrent() -> None:
        await asyncio.gather(
            bridge._ensure_xmpp_server(),
            bridge._ensure_xmpp_server(),
        )

    asyncio.run(run_concurrent())

    assert calls["spawn"] == 1
    assert calls["wait"] == 2


def test_startup_product_reuses_matching_selected_product() -> None:
    bridge = SystemBridge.__new__(SystemBridge)
    bridge.selected_product = "/tmp/product_a.json"
    bridge._selected_product_matches_requirement = lambda product, requirement: True

    def fail_resolve(requirement):
        raise AssertionError("full requirement scan should not run")

    bridge.resolve_product_init_for_requirement = fail_resolve

    resolved = bridge._resolve_startup_product_file(
        "/tmp/req_a.txt",
        ["/tmp/product_fallback.json"],
    )

    assert resolved == "/tmp/product_a.json"


def test_startup_product_falls_back_when_selected_product_mismatches() -> None:
    bridge = SystemBridge.__new__(SystemBridge)
    bridge.selected_product = "/tmp/stale_product.json"
    bridge._selected_product_matches_requirement = lambda product, requirement: False
    bridge.resolve_product_init_for_requirement = lambda requirement: "/tmp/product_b.json"

    resolved = bridge._resolve_startup_product_file(
        "/tmp/req_b.txt",
        ["/tmp/product_fallback.json"],
    )

    assert resolved == "/tmp/product_b.json"


def test_resource_agent_start_timing_logs_all_agents_and_aggregates_failure() -> None:
    class FakeAgent:
        def __init__(self, name: str, *, fail: bool = False) -> None:
            self.agent_name = name
            self.fail = fail
            self.started_with_auto_register = None

        async def start(self, auto_register: bool = True) -> None:
            self.started_with_auto_register = auto_register
            await asyncio.sleep(0.01)
            if self.fail:
                raise RuntimeError(f"{self.agent_name} boom")

    bridge = SystemBridge.__new__(SystemBridge)
    bridge.resource_agents = [
        FakeAgent("xarm6"),
        FakeAgent("ur5e", fail=True),
        FakeAgent("printer"),
    ]
    diagnostics: list[str] = []
    bridge._diag_emit = diagnostics.append

    with pytest.raises(RuntimeError, match="Failed to start 1/3 resource agents"):
        asyncio.run(bridge._start_resource_agents_for_startup(startup_id=7))

    assert all(agent.started_with_auto_register is True for agent in bridge.resource_agents)
    assert any("resource agent xarm6 started" in line for line in diagnostics)
    assert any("resource agent ur5e FAILED" in line for line in diagnostics)
    assert any("resource agent printer started" in line for line in diagnostics)


def test_startup_agent_group_timing_logs_each_agent() -> None:
    class FakeAgent:
        def __init__(self, name: str) -> None:
            self.agent_name = name

        async def start(self, auto_register: bool = True) -> None:
            await asyncio.sleep(0)

    bridge = SystemBridge.__new__(SystemBridge)
    diagnostics: list[str] = []
    bridge._diag_emit = diagnostics.append

    asyncio.run(
        bridge._start_agent_group_for_startup(
            [FakeAgent("cca"), FakeAgent("operator")],
            startup_id=8,
            group_label="control",
        )
    )

    assert any("control agent cca started" in line for line in diagnostics)
    assert any("control agent operator started" in line for line in diagnostics)


def test_product_task_message_includes_dispatch_trace() -> None:
    agent = ProductAgent.__new__(ProductAgent)

    msg = agent._compose_task_msg(
        to="ur5e@localhost",
        task_id="REQ_1_T1",
        instruction={"function_name": "pick_approach", "params": {}},
    )
    payload = json.loads(msg.body)

    assert msg.metadata["type"] == "task"
    assert payload["trace"]["product_dispatch_sent_at"]


def test_local_dispatch_delivers_same_container_message_without_xmpp() -> None:
    class FakeTraces:
        def __init__(self) -> None:
            self.rows = []

        def append(self, msg, category: str) -> None:
            self.rows.append((msg, category))

    class FakeTarget:
        def __init__(self) -> None:
            self.received = []

        def dispatch(self, msg) -> None:
            self.received.append(msg)

    class FakeContainer:
        def __init__(self, target: FakeTarget) -> None:
            self.target = target

        def has_agent(self, jid: str) -> bool:
            return jid == "resource@localhost"

        def get_agent(self, jid: str):
            assert jid == "resource@localhost"
            return self.target

    target = FakeTarget()
    sender = types.SimpleNamespace(
        jid="product@localhost",
        container=FakeContainer(target),
        traces=FakeTraces(),
        logger=types.SimpleNamespace(info=lambda *args, **kwargs: None, debug=lambda *args, **kwargs: None),
    )
    fallback_sends = []

    class FakeBehaviour:
        agent = sender

        async def send(self, msg):
            fallback_sends.append(msg)

    msg = Message(to="resource@localhost")
    msg.set_metadata("type", "task")
    msg.body = json.dumps({"trace": {"product_dispatch_sent_at": "2026-04-18T00:00:00+00:00"}})

    transport = asyncio.run(
        send_agent_message(
            FakeBehaviour(),
            msg,
            trace_category="test/local",
            transport_label="product_dispatch",
        )
    )

    assert transport == "local"
    assert fallback_sends == []
    assert target.received == [msg]
    assert str(msg.sender) == "product@localhost"
    assert msg.sent is True
    assert sender.traces.rows == [(msg, "test/local")]
    payload = json.loads(msg.body)
    assert payload["trace"]["product_dispatch_transport"] == "local"


def test_local_dispatch_falls_back_to_spade_send_for_remote_agent() -> None:
    sender = types.SimpleNamespace(
        jid="product@localhost",
        container=types.SimpleNamespace(
            has_agent=lambda _jid: False,
            get_agent=lambda _jid: None,
        ),
        logger=types.SimpleNamespace(info=lambda *args, **kwargs: None, debug=lambda *args, **kwargs: None),
    )
    sent = []

    class FakeBehaviour:
        agent = sender

        async def send(self, msg):
            sent.append(json.loads(msg.body))
            msg.sent = True

    msg = Message(to="resource@remote")
    msg.set_metadata("type", "task")
    msg.body = json.dumps({"trace": {"product_dispatch_sent_at": "2026-04-18T00:00:00+00:00"}})

    transport = asyncio.run(
        send_agent_message(
            FakeBehaviour(),
            msg,
            transport_label="product_dispatch",
        )
    )

    assert transport == "xmpp"
    assert sent[0]["trace"]["product_dispatch_transport"] == "xmpp"
    assert msg.sent is True


def test_local_dispatch_schedules_delivery_on_recipient_loop() -> None:
    class FakeLoop:
        def __init__(self) -> None:
            self.calls = []

        def is_running(self) -> bool:
            return True

        def call_soon_threadsafe(self, callback, *args) -> None:
            self.calls.append((callback, args))
            callback(*args)

    class FakeTarget:
        def __init__(self) -> None:
            self.loop = FakeLoop()
            self.received = []

        def dispatch(self, msg) -> None:
            self.received.append(msg)

    target = FakeTarget()

    class FakeContainer:
        def has_agent(self, jid: str) -> bool:
            return jid == "resource@localhost"

        def get_agent(self, jid: str):
            assert jid == "resource@localhost"
            return target

    sender = types.SimpleNamespace(
        jid="product@localhost",
        container=FakeContainer(),
        traces=types.SimpleNamespace(append=lambda *args, **kwargs: None),
        logger=types.SimpleNamespace(info=lambda *args, **kwargs: None, debug=lambda *args, **kwargs: None),
    )

    class FakeBehaviour:
        agent = sender

        async def send(self, msg):
            raise AssertionError("same-container delivery should not use SPADE/XMPP fallback")

    msg = Message(to="resource@localhost")
    msg.set_metadata("type", "task")
    msg.body = json.dumps({"trace": {"product_dispatch_sent_at": "2026-04-18T00:00:00+00:00"}})

    transport = asyncio.run(
        send_agent_message(
            FakeBehaviour(),
            msg,
            transport_label="product_dispatch",
        )
    )

    assert transport == "local"
    assert target.received == [msg]
    assert target.loop.calls


def test_planner_treats_safety_check_as_active_not_ready() -> None:
    logger = types.SimpleNamespace(
        warning=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
        exception=lambda *args, **kwargs: None,
    )
    planner = ProcessPlanner(types.SimpleNamespace(logger=logger), [])
    planner.nodes = [
        {
            "id": "REQ_1_T1",
            "type": "task",
            "status": "safety_check",
            "resource_jid": "ur5e@localhost",
            "function_name": "pick_approach",
            "predecessors": [],
            "successors": ["REQ_1_T2"],
        },
        {
            "id": "REQ_1_T2",
            "type": "task",
            "status": "pending",
            "resource_jid": "ur5e@localhost",
            "function_name": "pick_grasp",
            "predecessors": ["REQ_1_T1"],
            "successors": [],
        },
    ]

    assert planner.next_ready_task() is None


def test_planner_treats_safety_timeout_as_failed_not_active() -> None:
    logger = types.SimpleNamespace(
        warning=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
        exception=lambda *args, **kwargs: None,
    )
    planner = ProcessPlanner(types.SimpleNamespace(logger=logger), [])
    planner.nodes = [
        {
            "id": "REQ_1_T1",
            "type": "task",
            "status": "failed:safety_decision_timeout",
            "resource_jid": "ur5e@localhost",
            "function_name": "pick_approach",
            "predecessors": [],
            "successors": ["REQ_1_T2"],
        },
        {
            "id": "REQ_1_T2",
            "type": "task",
            "status": "pending",
            "resource_jid": "ur5e@localhost",
            "function_name": "pick_grasp",
            "predecessors": ["REQ_1_T1"],
            "successors": [],
        },
    ]

    assert planner.next_ready_task() is None


def test_resource_safety_decision_wait_times_out() -> None:
    agent = ResourceAgent.__new__(ResourceAgent)
    agent._safety_decisions = {}
    agent.safety_decision_timeout_s = 0.01

    decision = asyncio.run(agent._wait_for_safety_decision("missing-task"))

    assert decision is None


def test_cca_safety_decision_payload_includes_reason_and_trace() -> None:
    logger = types.SimpleNamespace(info=lambda *args, **kwargs: None)
    sent = []

    class FakeBehaviour:
        agent = types.SimpleNamespace(logger=logger)

        async def send(self, msg):
            sent.append(msg)

    async def run() -> None:
        await CentralControllerAgent._Monitor._send_decision(
            FakeBehaviour(),
            "ur5e@localhost",
            "REQ_1_T1",
            "block",
            trace={"resource_safety_sent_at": "2026-04-18T00:00:00+00:00"},
            reason="plan_block:not_enabled",
        )

    asyncio.run(run())

    assert sent
    assert sent[0].metadata["type"] == "safety_decision"
    payload = json.loads(sent[0].body)
    assert payload["task_id"] == "REQ_1_T1"
    assert payload["decision"] == "block"
    assert payload["reason"] == "plan_block:not_enabled"
    assert payload["trace"]["cca_decision_sent_at"]


class _DummyTransport:
    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.paused = False

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    def pause_reading(self) -> None:
        self.paused = True

    def get_extra_info(self, name: str, default=None):
        if name == "peername":
            return ("127.0.0.1", 5222)
        return default


def _install_xmpp_runtime(monkeypatch, *, local_only: bool):
    monkeypatch.setenv("CAIS_XMPP_LOCAL_ONLY", "1" if local_only else "0")
    runtime = importlib.import_module("cais_spade_llm.xmpp_runtime")
    runtime._INSTALLED = False
    runtime.install_xmpp_runtime_patches()
    return runtime


def _inspect_xmpp_client_flags(jid: str = "agent@localhost") -> tuple[bool, bool, bool, bool, bool]:
    from spade.xmpp_client import XMPPClient

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        client = XMPPClient(jid, "pw", False, True)
        return (
            bool(client.enable_starttls),
            bool(client.enable_direct_tls),
            bool(client.enable_plaintext),
            bool(client["feature_mechanisms"].unencrypted_plain),
            bool(client.plugin.registered("xep_0077")),
        )
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def test_xmpp_local_only_omits_starttls_and_enables_plaintext_client(monkeypatch) -> None:
    _install_xmpp_runtime(monkeypatch, local_only=True)
    stream_handler_module = importlib.import_module("pyjabber.stream.StreamHandler")
    transport = _DummyTransport()

    handler = stream_handler_module.StreamHandler(transport, lambda: None)
    handler.handle_open_stream()
    payload = b"".join(transport.writes)

    assert b"starttls" not in payload
    assert b"mechanisms" in payload
    assert b"register" in payload

    enable_starttls, enable_direct_tls, enable_plaintext, unencrypted_plain, has_register = _inspect_xmpp_client_flags()
    assert enable_starttls is False
    assert enable_direct_tls is False
    assert enable_plaintext is True
    assert unencrypted_plain is True
    assert has_register is False

    external_starttls, external_direct_tls, external_plaintext, external_unencrypted_plain, external_has_register = (
        _inspect_xmpp_client_flags("agent@example.com")
    )
    assert external_starttls is True
    assert external_direct_tls is True
    assert external_plaintext is False
    assert external_unencrypted_plain is False
    assert external_has_register is True


def test_xmpp_tls_mode_preserves_starttls_and_default_client(monkeypatch) -> None:
    _install_xmpp_runtime(monkeypatch, local_only=False)
    stream_handler_module = importlib.import_module("pyjabber.stream.StreamHandler")
    transport = _DummyTransport()

    handler = stream_handler_module.StreamHandler(transport, lambda: None)
    handler.handle_open_stream()
    payload = b"".join(transport.writes)

    assert b"starttls" in payload

    enable_starttls, enable_direct_tls, _enable_plaintext, unencrypted_plain, has_register = _inspect_xmpp_client_flags()
    assert enable_starttls is True
    assert enable_direct_tls is True
    assert unencrypted_plain is False
    assert has_register is True


def test_xmpp_local_only_autocreates_loopback_credentials(monkeypatch) -> None:
    _install_xmpp_runtime(monkeypatch, local_only=True)
    sasl_module = importlib.import_module("pyjabber.features.SASLFeature")

    stored: dict[str, tuple[bytes]] = {}

    class FakeResult:
        def __init__(self, row):
            self.row = row

        def fetchone(self):
            return self.row

    class FakeConnection:
        def execute(self, query):
            if isinstance(query, FakeInsert):
                stored[query.payload["jid"]] = (query.payload["hash_pwd"],)
                return FakeResult(None)
            return FakeResult(stored.get("agent"))

        def commit(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    class FakeDB:
        @staticmethod
        def connection():
            return FakeConnection()

    class FakeColumn:
        def __eq__(self, other):
            return ("eq", other)

    class FakeQuery:
        def where(self, _condition):
            return self

    class FakeInsert:
        def __init__(self):
            self.payload = {}

        def values(self, payload):
            self.payload = dict(payload)
            return self

    monkeypatch.setattr(sasl_module, "DB", FakeDB)
    monkeypatch.setattr(
        sasl_module,
        "Model",
        types.SimpleNamespace(
            Credentials=types.SimpleNamespace(
                c=types.SimpleNamespace(jid=FakeColumn(), hash_pwd=FakeColumn())
            )
        ),
    )
    monkeypatch.setattr(sasl_module, "select", lambda _column: FakeQuery())
    monkeypatch.setattr(sasl_module, "insert", lambda _table: FakeInsert())
    monkeypatch.setattr(sasl_module.metadata, "HOST", "localhost")
    monkeypatch.setattr(
        sasl_module,
        "bcrypt",
        types.SimpleNamespace(
            gensalt=lambda: b"salt",
            hashpw=lambda pwd, _salt: b"hashed:" + pwd,
            checkpw=lambda pwd, hashed: hashed == b"hashed:" + pwd,
        ),
    )

    class FakeConnectionManager:
        def __init__(self):
            self.bound = None

        def set_jid(self, peername, jid):
            self.bound = (peername, str(jid))

    sasl = sasl_module.SASL()
    sasl._peername = ("127.0.0.1", 50123)
    sasl._connection_manager = FakeConnectionManager()

    element = ET.Element("auth")
    element.text = base64.b64encode(b"\x00agent\x00pw").decode()

    result = sasl.handleAuth(element)

    assert result[0] == sasl_module.Signal.RESET
    assert stored["agent"] == (b"hashed:pw",)
    assert sasl._connection_manager.bound[0] == ("127.0.0.1", 50123)


class _FakeBundleStore:
    def __init__(self, active_id: str | None, *, root: Path | None = None, manifest: dict | None = None) -> None:
        self.active_id = active_id
        self.set_calls: list[str | None] = []
        self.root = root
        self.manifest = manifest

    def get_active_bundle_id(self) -> str | None:
        return self.active_id

    def set_active_bundle_id(self, bundle_id: str | None) -> None:
        self.active_id = bundle_id
        self.set_calls.append(bundle_id)

    def load_manifest(self, _bundle_id: str) -> dict | None:
        return self.manifest

    def bundle_dir(self, _bundle_id: str) -> Path:
        if self.root is None:
            return Path("/tmp/nonexistent-bundle")
        return self.root

    def manifest_path(self, _bundle_id: str) -> Path:
        if self.root is None:
            return Path("/tmp/nonexistent-bundle/bundle_manifest.json")
        return self.root / "bundle_manifest.json"


def test_startup_bundle_compatibility_uses_snapshot_without_tools_refresh(tmp_path) -> None:
    req = tmp_path / "req.txt"
    safety = tmp_path / "safety.txt"
    bundle_root = tmp_path / "bundle"
    catalog = bundle_root / "catalog"
    tools = catalog / "tools.json"
    req.write_text("assemble product", encoding="utf-8")
    safety.write_text("always safe", encoding="utf-8")
    catalog.mkdir(parents=True)
    tools.write_text('{"tools": []}', encoding="utf-8")

    manifest = {
        "status": bridge_module.BUNDLE_STATUS_VERIFIED,
        "product_spec_file": str(req),
        "execution_mode": "simulation",
        "robot_env": "gazebo",
        "safety_file": str(safety),
        "source_hashes": {
            "requirements_sha256": bridge_module.sha256_text("assemble product"),
            "safety_sha256": bridge_module.sha256_text("always safe"),
            "tools_sha256": bridge_module.sha256_file(tools),
        },
        "artifacts": {"tools_json": "catalog/tools.json"},
    }
    (bundle_root / "bundle_manifest.json").write_text("{}", encoding="utf-8")

    bridge = SystemBridge.__new__(SystemBridge)
    bridge.bundle_store = _FakeBundleStore("bundle-ok", root=bundle_root, manifest=manifest)
    bridge._startup_bundle_compatibility_cache = {}
    bridge._active_tools_signature_payload = lambda: ("resources",)
    bridge._diag_emit = lambda _message: None
    bridge._refresh_active_tools_catalogue = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("startup bundle compatibility must not refresh tools")
    )

    ok, reasons = bridge._check_startup_bundle_compatibility_cached(
        "bundle-ok",
        str(req),
        "simulation",
        "gazebo",
        str(safety),
    )

    assert ok is True
    assert reasons == []


def test_startup_bundle_compatibility_cache_reuses_result() -> None:
    bridge = SystemBridge.__new__(SystemBridge)
    bridge._startup_bundle_compatibility_cache = {}
    bridge._startup_bundle_compatibility_signature = lambda *args, **kwargs: ("same",)
    calls = {"count": 0}

    def fake_check(*args, **kwargs):
        calls["count"] += 1
        return True, []

    bridge.check_bundle_compatibility = fake_check

    assert bridge._check_startup_bundle_compatibility_cached("b", "req", "sim", "gazebo") == (True, [])
    assert bridge._check_startup_bundle_compatibility_cached("b", "req", "sim", "gazebo") == (True, [])
    assert calls["count"] == 1


def test_startup_bundle_context_diagnostics_no_active_bundle() -> None:
    bridge = SystemBridge.__new__(SystemBridge)
    bridge.bundle_store = _FakeBundleStore(None)
    diagnostics: list[str] = []
    bridge._diag_emit = diagnostics.append

    context, notice = bridge._resolve_startup_bundle_context(
        product_spec_file="/tmp/req.txt",
        execution_mode="simulation",
        robot_env="gazebo",
        safety_requirement_file="/tmp/safety.txt",
    )

    assert context is None
    assert notice is None
    assert any("No active verified plan set selected" in line for line in diagnostics)


def test_startup_bundle_context_diagnostics_accepted_bundle() -> None:
    bridge = SystemBridge.__new__(SystemBridge)
    bridge.bundle_store = _FakeBundleStore("bundle-ok")
    bridge.check_bundle_compatibility = lambda *args, **kwargs: (True, [])
    bridge._resolve_active_bundle_context = lambda **kwargs: {"bundle_id": "bundle-ok"}
    diagnostics: list[str] = []
    bridge._diag_emit = diagnostics.append

    context, notice = bridge._resolve_startup_bundle_context(
        product_spec_file="/tmp/req.txt",
        execution_mode="simulation",
        robot_env="gazebo",
        safety_requirement_file="/tmp/safety.txt",
    )

    assert context == {"bundle_id": "bundle-ok"}
    assert notice is None
    assert bridge.bundle_store.set_calls == []
    assert any("Checking active verified plan set bundle-ok" in line for line in diagnostics)
    assert any("Accepted active verified plan set bundle-ok" in line for line in diagnostics)


def test_startup_bundle_context_diagnostics_deactivates_incompatible_bundle() -> None:
    bridge = SystemBridge.__new__(SystemBridge)
    bridge.bundle_store = _FakeBundleStore("bundle-stale")
    bridge.check_bundle_compatibility = lambda *args, **kwargs: (False, ["product_spec_file"])
    bridge._describe_bundle_incompatibility_reasons = lambda reasons: "product requirement changed"
    diagnostics: list[str] = []
    bridge._diag_emit = diagnostics.append

    context, notice = bridge._resolve_startup_bundle_context(
        product_spec_file="/tmp/req.txt",
        execution_mode="simulation",
        robot_env="gazebo",
        safety_requirement_file="/tmp/safety.txt",
    )

    assert context is None
    assert "deactivated" in str(notice)
    assert bridge.bundle_store.active_id is None
    assert bridge.bundle_store.set_calls == [None]
    assert any("Auto-deactivated incompatible active bundle bundle-stale" in line for line in diagnostics)


def test_startup_consistency_warns_for_plan_tool_live_resource_mismatch(tmp_path) -> None:
    req = tmp_path / "req.txt"
    safety = tmp_path / "safety.txt"
    plan = tmp_path / "plan.json"
    tools = tmp_path / "tools.json"
    req.write_text("assemble case3", encoding="utf-8")
    safety.write_text("stay safe", encoding="utf-8")
    plan.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "REQ_2_T1",
                        "type": "task",
                        "resource_jid": "xarm6@localhost",
                        "function_name": "pick_approach",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    tools.write_text(
        json.dumps([{"function": "pick_approach", "function_owner_agent": "ur5e"}]),
        encoding="utf-8",
    )

    bridge = SystemBridge.__new__(SystemBridge)
    bridge.bundle_store = _FakeBundleStore("bundle-ok")
    bridge.execution_mode = "simulation"
    bridge.robot_env = "gazebo"
    bridge.resource_agents = [types.SimpleNamespace(jid="ur5e@localhost")]
    bridge.list_selected_resource_entries = lambda: [
        {"key": "ur5e", "jid": "ur5e@localhost"}
    ]
    diagnostics: list[str] = []
    bridge._diag_emit = diagnostics.append
    bundle_context = {
        "bundle_id": "bundle-ok",
        "product_spec_file": str(req),
        "safety_file": str(safety),
        "artifacts": {"plan_json": str(plan)},
    }

    warnings = bridge._emit_startup_consistency_diagnostics(
        startup_id=9,
        selected_requirement_file=str(req),
        selected_safety_file=str(safety),
        selected_resource_files=[str(tmp_path / "robot_ur5e.json")],
        bundle_context=bundle_context,
        tools_catalogue_path=str(tools),
        tools_meta={"using_bundle_tools": False},
        perception_backend="none",
    )

    assert any("not selected for startup" in warning for warning in warnings)
    assert any("missing from active tools catalogue: xarm6" in warning for warning in warnings)
    assert any("without live resource agents: xarm6" in warning for warning in warnings)
    assert any("tool_owners=ur5e" in line for line in diagnostics)


def test_startup_consistency_allows_selected_resource_superset(tmp_path) -> None:
    req = tmp_path / "req.txt"
    safety = tmp_path / "safety.txt"
    plan = tmp_path / "plan.json"
    tools = tmp_path / "tools.json"
    req.write_text("assemble case1", encoding="utf-8")
    safety.write_text("stay safe", encoding="utf-8")
    plan.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "REQ_1_T1",
                        "type": "task",
                        "resource_jid": "xarm6@localhost",
                        "function_name": "pick_approach",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    tools.write_text(
        json.dumps([{"function": "pick_approach", "function_owner_agent": "xarm6"}]),
        encoding="utf-8",
    )

    bridge = SystemBridge.__new__(SystemBridge)
    bridge.bundle_store = _FakeBundleStore("bundle-ok")
    bridge.execution_mode = "simulation"
    bridge.robot_env = "gazebo"
    bridge.resource_agents = [
        types.SimpleNamespace(jid="ur5e@localhost"),
        types.SimpleNamespace(jid="xarm6@localhost"),
    ]
    bridge.list_selected_resource_entries = lambda: [
        {"key": "ur5e", "jid": "ur5e@localhost"},
        {"key": "xarm6", "jid": "xarm6@localhost"},
    ]
    diagnostics: list[str] = []
    bridge._diag_emit = diagnostics.append
    bundle_context = {
        "bundle_id": "bundle-ok",
        "product_spec_file": str(req),
        "safety_file": str(safety),
        "artifacts": {"plan_json": str(plan)},
    }

    warnings = bridge._emit_startup_consistency_diagnostics(
        startup_id=10,
        selected_requirement_file=str(req),
        selected_safety_file=str(safety),
        selected_resource_files=[
            str(tmp_path / "robot_ur5e.json"),
            str(tmp_path / "robot_xarm6.json"),
        ],
        bundle_context=bundle_context,
        tools_catalogue_path=str(tools),
        tools_meta={"using_bundle_tools": False},
        perception_backend="none",
    )

    assert not any("not selected for startup" in warning for warning in warnings)
    assert any("include extra robots" in line for line in diagnostics)
