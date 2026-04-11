from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cais_spade_llm.ui import bridge as bridge_mod


def _copy_runtime_resources(tmp_path: Path) -> Path:
    resource_dir = tmp_path / "resources"
    resource_dir.mkdir(parents=True, exist_ok=True)
    source_dir = ROOT / "cais_spade_llm" / "initialization" / "resources"
    for name in (
        "robot_xarm6.json",
        "robot_ur5e.json",
        "robot_xarm6_2.json",
        "robot_ur5e_2.json",
    ):
        (resource_dir / name).write_text(
            (source_dir / name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    return resource_dir


def _make_bridge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> bridge_mod.SystemBridge:
    resource_dir = _copy_runtime_resources(tmp_path)
    tools_path = tmp_path / "tools.json"
    monkeypatch.setattr(bridge_mod, "_RESOURCE_DIR", resource_dir)
    monkeypatch.setattr(bridge_mod, "_TOOLS_OUT", tools_path)
    monkeypatch.setattr(bridge_mod, "_XARM6_RESOURCE", resource_dir / "robot_xarm6.json")
    monkeypatch.setattr(bridge_mod, "_UR5E_RESOURCE", resource_dir / "robot_ur5e.json")
    monkeypatch.setattr(bridge_mod, "_XARM6_2_RESOURCE", resource_dir / "robot_xarm6_2.json")
    monkeypatch.setattr(bridge_mod, "_UR5E_2_RESOURCE", resource_dir / "robot_ur5e_2.json")
    monkeypatch.setattr(
        bridge_mod.SystemBridge,
        "_maybe_start_agent_creator_prefetch",
        lambda self: None,
    )
    fake_agent_creator = types.ModuleType("agent_creator")
    fake_agent_creator.ALLOWED_FUNCS = {}

    def _configure_runtime(*, robot_env=None, execution_mode=None, perception_backend=None) -> None:
        return None

    def _create_resource_agents(resource_init_list, cca_init_file, prewarmed_controllers=None):
        agents = []
        allowed: dict[str, set[str]] = {}
        for raw_path in resource_init_list:
            payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
            name = next(iter(payload.keys()))

            def _pick_approach():
                """stub"""

            _pick_approach.__name__ = "pick_approach"
            agent = types.SimpleNamespace(
                agent_name=name,
                pick_approach=_pick_approach,
            )
            agents.append(agent)
            allowed[name] = {"pick_approach"}
        fake_agent_creator.ALLOWED_FUNCS = allowed
        return agents

    fake_agent_creator.configure_runtime = _configure_runtime
    fake_agent_creator.create_resource_agents = _create_resource_agents

    fake_function_analyzer = types.ModuleType("function_analyzer")

    class _FakeFunctionAnalyzer:
        @staticmethod
        def build_tools_catalogue(agents, allowed=None, outfile=Path("tools.json")) -> None:
            rows = []
            for agent in agents:
                owner = getattr(agent, "agent_name", "")
                for fn_name in sorted((allowed or {}).get(owner, {"pick_approach"})):
                    rows.append(
                        {
                            "function_owner_agent": owner,
                            "function": fn_name,
                        }
                    )
            Path(outfile).write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fake_function_analyzer.FunctionAnalyzer = _FakeFunctionAnalyzer
    fake_llm_agent_mod = types.ModuleType("cais_spade_llm.agents.shared_information.llm_agent")

    class _FakeLlmAgent:
        _TOOLS_CATALOG_PATH = None

        @classmethod
        def configure_shared_tools_catalogue(cls, path):
            cls._TOOLS_CATALOG_PATH = Path(path)
            return str(cls._TOOLS_CATALOG_PATH)

    fake_llm_agent_mod.LlmAgent = _FakeLlmAgent
    monkeypatch.setitem(sys.modules, "agent_creator", fake_agent_creator)
    monkeypatch.setitem(sys.modules, "function_analyzer", fake_function_analyzer)
    monkeypatch.setitem(sys.modules, "cais_spade_llm.agents.shared_information.llm_agent", fake_llm_agent_mod)
    return bridge_mod.SystemBridge()


def test_bridge_defaults_to_primary_active_resources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bridge = _make_bridge(tmp_path, monkeypatch)

    assert [row["key"] for row in bridge.list_available_resource_entries()] == [
        "xarm6",
        "ur5e",
        "xarm6-2",
        "ur5e-2",
    ]
    assert bridge.get_selected_resource_keys() == ["xarm6", "ur5e"]
    assert bridge.selected_resource_summary() == "xArm6, UR5e"


def test_bridge_rebuilds_tools_catalogue_from_active_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = _make_bridge(tmp_path, monkeypatch)

    bridge._refresh_active_tools_catalogue(force=True)
    tools_payload = json.loads((tmp_path / "tools.json").read_text(encoding="utf-8"))
    owners = {str(row.get("function_owner_agent") or "") for row in tools_payload}
    assert owners == {"xarm6", "ur5e"}

    bridge.set_selected_resource_keys(["xarm6", "ur5e", "xarm6-2"])
    bridge._refresh_active_tools_catalogue(force=True)
    tools_payload = json.loads((tmp_path / "tools.json").read_text(encoding="utf-8"))
    owners = {str(row.get("function_owner_agent") or "") for row in tools_payload}
    assert owners == {"xarm6", "ur5e", "xarm6-2"}


def test_safety_text_auto_activates_explicit_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = _make_bridge(tmp_path, monkeypatch)

    selected = bridge._ensure_selected_resources_cover_safety_text(
        "xarm6, ur5e, and xarm6-2 should not be place_approach at the same time to the assembly station.",
        purpose="safety preview generation",
    )

    assert selected == ["xarm6", "ur5e", "xarm6-2"]
    assert bridge.get_selected_resource_keys() == ["xarm6", "ur5e", "xarm6-2"]
    assert "xArm6-2" in str(bridge.last_notice or "")

    selected = bridge._ensure_selected_resources_cover_safety_text(
        "xarm6, ur5e, and xarm6-2 should not be place_approach at the same time to the assembly station."
    )
    assert selected == ["xarm6", "ur5e", "xarm6-2"]
