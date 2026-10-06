"""Given safety definitions reach existing artifacts without an LLM call."""

from __future__ import annotations

import asyncio
import json
import logging
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cais_spade_llm.agents.central_controller.predefined_safety import (
    compile_predefined_safety,
    parse_predefined_safety,
    validate_predefined_safety_artifact,
)
from cais_spade_llm.agents.central_controller.safety_logic import SafetyLogic
from cais_spade_llm.bundles.bundle_compiler import BundleCompiler

_ROOT = Path(__file__).resolve().parents[1]
_SOURCE = _ROOT / "cais_spade_llm/specification/safety/safety_assembly_board-v1_predefined.txt"


def _document() -> dict:
    return json.loads(_SOURCE.read_text(encoding="utf-8"))


def _logic(path: Path) -> SafetyLogic:
    return SafetyLogic(SimpleNamespace(
        logger=logging.getLogger(__name__), tools_catalog=[],
        ask_llm=AsyncMock(side_effect=AssertionError("Predefined input must not call an LLM")),
    ), path)


def _saved(tmp_path: Path, document: dict | None = None) -> tuple[SafetyLogic, Path, str]:
    source = tmp_path / "given.txt"
    text = json.dumps(document if document is not None else _document(), indent=2)
    source.write_text(text, encoding="utf-8")
    logic = _logic(source)
    asyncio.run(logic.build_safety_rules_and_logic(text))
    asyncio.run(logic.build_preview_interpretations())
    artifact = tmp_path / "preview" / "cca_safety_logic.json"
    logic.save(artifact)
    logic.build_dfas_per_rule(artifact.parent)
    return logic, artifact, text


def test_predefined_compile_save_load_preserves_rule_local_labels_without_llm(tmp_path: Path) -> None:
    logic, artifact, text = _saved(tmp_path)
    document = _document()
    payload = json.loads(artifact.read_text())
    assert validate_predefined_safety_artifact(payload, source_text=text) == document
    for supplied, rule in zip(document["catalog"]["specifications"], logic.rules, strict=True):
        assert (rule["id"], rule["ltlf"], rule["aps"]) == (
            supplied["id"], supplied["formula"], supplied["aps"])
        assert (artifact.parent / f"{rule['id']}_dfa.dot").is_file()
        assert (artifact.parent / f"{rule['id']}_dfa.png").is_file()
    # The unchanged Safety page discovers SAFE_* artifacts; these are new IDs,
    # not replacements for the legacy shared_area_mutex catalog identifier.
    assert len(list(artifact.parent.glob("SAFE_*_dfa.dot"))) == 2
    assert [ap["label"] for ap in logic.rules[0]["aps"]] == [ap["label"] for ap in logic.rules[1]["aps"]]
    assert logic.global_safety_spec == {}
    restored = _logic(logic.safety_file)
    restored.load(artifact)
    assert restored.rules == logic.rules
    assert restored.predefined_metadata == logic.predefined_metadata
    assert restored.build_dfas_per_rule(tmp_path / "restored") == logic.rule_dfas
    with pytest.raises(ValueError, match="separate rule-local AP namespaces"):
        restored.build_global_dfa()
    logic.controller_agent.ask_llm.assert_not_called()
    restored.controller_agent.ask_llm.assert_not_called()


@pytest.mark.parametrize("change", ["version", "scope", "target", "predicate", "formula"])
def test_invalid_given_definitions_never_fall_back_to_llm(tmp_path: Path, change: str) -> None:
    document = _document()
    if change == "version":
        document["version"] = 2
    elif change == "scope":
        document["requirement_scopes"].pop()
    elif change == "target":
        document["requirement_scopes"][1]["physical_ap_bindings"]["ap002"]["target"] = "Gear_Plate/Gear_Shaft_2"
    elif change == "predicate":
        document["catalog"]["specifications"][0]["aps"][0]["full"] = "ap_state/physical_observation/unknown"
    else:
        document["catalog"]["specifications"][0]["formula"] = "G ap999"
    logic = _logic(tmp_path / "given.txt")
    with pytest.raises(ValueError):
        asyncio.run(logic.build_safety_rules_and_logic(json.dumps(document)))
    logic.controller_agent.ask_llm.assert_not_called()


@pytest.mark.parametrize("text", ['{"mode":"predefined",', '{"mode":"predefined","mode":"predefined"}'])
def test_malformed_or_duplicate_json_cannot_be_reinterpreted(tmp_path: Path, text: str) -> None:
    logic = _logic(tmp_path / "given.txt")
    with pytest.raises(ValueError):
        asyncio.run(logic.build_safety_rules_and_logic(text))
    logic.controller_agent.ask_llm.assert_not_called()


def test_missing_compiler_support_fails_even_after_cached_compilation(monkeypatch) -> None:
    from cais_spade_llm.agents.central_controller import reviewed_primitive_program_safety

    compile_predefined_safety(_document())
    monkeypatch.setattr(reviewed_primitive_program_safety.shutil, "which", lambda _: None)
    with pytest.raises(ValueError, match="MONA or LTLfParser"):
        compile_predefined_safety(_document())


@pytest.mark.parametrize("change", ["formula", "label", "scope", "mode", "source"])
def test_changed_predefined_artifact_is_rejected(tmp_path: Path, change: str) -> None:
    _, artifact, text = _saved(tmp_path)
    payload = json.loads(artifact.read_text())
    if change == "formula":
        payload["rules"][0]["ltlf"] = "G ap001"
    elif change == "label":
        payload["rules"][0]["aps"][0]["label"] = "ap009"
    elif change == "scope":
        payload["predefined_safety"]["requirement_scopes"][1]["physical_ap_bindings"]["ap001"]["part"] = "KET8_Square_8mm"
    elif change == "mode":
        payload.pop("mode")
    else:
        text = text.replace("may enter", "may touch")
    with pytest.raises(ValueError):
        validate_predefined_safety_artifact(payload, source_text=text)


def test_changed_geometry_owner_invalidates_saved_artifact(tmp_path: Path) -> None:
    document = _document()
    owner = tmp_path / "geometry.json"
    owner.write_bytes((_ROOT / document["product_geometry"]["path"]).read_bytes())
    document["product_geometry"]["path"] = str(owner)
    _, artifact, text = _saved(tmp_path, document)
    payload = json.loads(artifact.read_text())
    owner.write_text(owner.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="geometry changed"):
        validate_predefined_safety_artifact(payload, source_text=text)


def test_bundle_preserves_predefined_metadata_and_rejects_changed_dfa(tmp_path: Path) -> None:
    logic, artifact, text = _saved(tmp_path)
    descriptor = {
        "safety_logic_json": str(artifact),
        "safety_sha256": logic.safety_text_sha256,
        "dfa_dot_files": [str(path) for path in artifact.parent.glob("*_dfa.dot")],
        "dfa_png_files": [str(path) for path in artifact.parent.glob("*_dfa.png")],
    }
    copied, dfas = BundleCompiler._copy_precomputed_safety_artifacts(
        descriptor, tmp_path / "bundle", source_text=text)
    assert json.loads(copied.read_text()) == json.loads(artifact.read_text())
    assert dfas == logic.rule_dfas
    Path(descriptor["dfa_dot_files"][0]).write_text("digraph { broken; }", encoding="utf-8")
    with pytest.raises(ValueError, match="DFA artifacts differ"):
        BundleCompiler._copy_precomputed_safety_artifacts(
            descriptor, tmp_path / "changed", source_text=text)


def test_predefined_source_rejects_legacy_artifact_and_source_fingerprint(tmp_path: Path) -> None:
    logic, artifact, text = _saved(tmp_path)
    payload = json.loads(artifact.read_text())
    legacy = {key: deepcopy(payload[key]) for key in ("rules", "safety_text_sha256")}
    with pytest.raises(ValueError, match="unmarked legacy"):
        validate_predefined_safety_artifact(legacy, source_text=text)
    with pytest.raises(ValueError, match="source fingerprint"):
        BundleCompiler._copy_precomputed_safety_artifacts({
            "safety_logic_json": str(artifact), "safety_sha256": "stale",
        }, tmp_path / "wrong_source", source_text=text)
    logic.controller_agent.ask_llm.assert_not_called()
    assert parse_predefined_safety("[Safety Requirements]\n- Keep the existing requirement.") is None


def _startup_agent(source: Path, artifact: Path | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        safety_file=source, logger=logging.getLogger(__name__), safety_logic=_logic(source),
        safety_monitor=object(), _tools_catalog_for_safety=lambda: [],
        precomputed_bundle={} if artifact is None else {
            "artifacts": {"safety_logic_json": str(artifact)},
        },
    )


def test_cca_loads_given_definitions_without_native_physical_ap_mapping(tmp_path: Path) -> None:
    from cais_spade_llm.agents.central_controller.predefined_safety_runtime import (
        initialize_predefined_safety,
    )

    logic, artifact, _ = _saved(tmp_path)
    agent = _startup_agent(logic.safety_file, artifact)
    assert initialize_predefined_safety(agent)
    assert not agent.predefined_safety_error
    assert agent.predefined_safety == _document()
    assert agent.safety_logic.rule_dfas == logic.rule_dfas
    assert agent.safety_monitor.safety_rules == []
    assert agent.predefined_geometry_sha256 == logic.predefined_metadata["predefined_geometry_sha256"]
    agent.safety_logic.controller_agent.ask_llm.assert_not_called()


@pytest.mark.parametrize("change", ["malformed_json", "array", "missing_metadata", "wrong_source_hash",
                                     "changed_dot", "missing_dot", "missing_source", "changed_source",
                                     "removed_source_and_mode", "removed_all_mode_metadata"])
def test_cca_invalid_predefined_artifacts_never_fall_back(tmp_path: Path, change: str) -> None:
    from cais_spade_llm.agents.central_controller.predefined_safety_runtime import (
        initialize_predefined_safety,
        predefined_required,
    )

    logic, artifact, _ = _saved(tmp_path)
    agent = _startup_agent(logic.safety_file, artifact)
    payload = json.loads(artifact.read_text())
    if change == "malformed_json":
        artifact.write_text("{broken", encoding="utf-8")
    elif change == "array":
        artifact.write_text("[]", encoding="utf-8")
    elif change == "missing_metadata":
        payload.pop("predefined_safety")
        artifact.write_text(json.dumps(payload), encoding="utf-8")
    elif change == "wrong_source_hash":
        agent.precomputed_bundle["safety_source"] = {"safety_sha256": "stale", "mode": "predefined"}
    elif change in {"changed_dot", "missing_dot"}:
        dot = artifact.parent / f"{logic.rules[0]['id']}_dfa.dot"
        if change == "missing_dot":
            dot.unlink()
        else:
            dot.write_text(logic.rule_dfas[logic.rules[1]["id"]], encoding="utf-8")
    elif change == "missing_source":
        logic.safety_file.unlink()
    elif change == "changed_source":
        logic.safety_file.write_text("[Safety Requirements]\n- any requirement\n", encoding="utf-8")
    else:
        logic.safety_file.write_text("[Safety Requirements]\n", encoding="utf-8")
        payload = {key: value for key, value in payload.items()
                   if key != "mode" and not key.startswith("predefined_")}
        artifact.write_text(json.dumps(payload), encoding="utf-8")
        if change == "removed_source_and_mode":
            agent.precomputed_bundle["safety_source"] = {"definition_mode": "predefined"}
    assert predefined_required(agent)
    assert initialize_predefined_safety(agent)
    assert agent.predefined_safety_error
    assert agent.safety_monitor is None
    agent.safety_logic.controller_agent.ask_llm.assert_not_called()


def test_bundle_native_validator_cannot_certify_predefined_physical_rules() -> None:
    from unittest.mock import Mock

    from cais_spade_llm.agents.central_controller.plan_safety_validator import PlanSafetyValidator

    rules, dfas = compile_predefined_safety(_document())
    validator = PlanSafetyValidator(rules, dfas)
    validator.validate_plan_fsa = Mock(side_effect=AssertionError("Native AP mapping must not run"))
    planner = SimpleNamespace(
        replan_with_feedback_offline=AsyncMock(side_effect=AssertionError("Missing evidence is not an LLM repair")),
        compile_global_fsa=Mock(side_effect=AssertionError("No repair compilation")),
    )
    result = asyncio.run(BundleCompiler.run_offline_repair_loop(
        product_agent=SimpleNamespace(process_planner=planner), validator=validator,
        product_jid="product@localhost", auto_replan_max_attempts=3,
        seed_replan_violations=[{"reason": "missing physical evidence"}],
    ))
    assert result["ok"] is False
    assert result["status"] == "inconclusive"
    assert result["stop_reason"] == "predefined_physical_grounding_required"
    assert result["pending_rule_ids"] == [rule["id"] for rule in rules]
    assert result["auto_replans_used"] == 0
    validator.validate_plan_fsa.assert_not_called()
    planner.replan_with_feedback_offline.assert_not_called()
    planner.compile_global_fsa.assert_not_called()


@pytest.mark.parametrize("plant_compile_fails", [False, True])
def test_predefined_bundle_cannot_be_native_verified_or_replanned_without_safety(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plant_compile_fails: bool,
) -> None:
    from unittest.mock import Mock

    from cais_spade_llm.agents.central_controller.plan_safety_validator import PlanSafetyValidator
    from cais_spade_llm.bundles.bundle_store import BundleStore

    async def inline_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", inline_thread)
    for key in ("ROBOT_ENV", "EXECUTION_MODE", "PERCEPTION_BACKEND"):
        monkeypatch.setenv(key, "test")
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("Mock product requirements", encoding="utf-8")
    product = tmp_path / "product.json"
    product.write_text(json.dumps({"name": "product", "product_specification_file": str(requirements)}))
    cca = tmp_path / "cca.json"
    cca.write_text(json.dumps({"cca": {"name": "cca", "safety_file": str(_SOURCE)}}))
    tools, prompts = tmp_path / "tools.json", tmp_path / "prompts.json"
    tools.write_text("{}")
    prompts.write_text("{}")

    def save_global_fsa(path: Path) -> None:
        if plant_compile_fails:
            raise ValueError("Synthetic plant cannot compile")
        path.write_text(json.dumps({"A": {"x0": "ready", "Xm": ["done"], "Tr": []}}))

    planner = SimpleNamespace(
        nodes=[], global_fsa={"A": {}}, build_high_level=AsyncMock(),
        expand_requirements_to_tasks=AsyncMock(),
        save=lambda path: path.write_text('{"nodes":[]}'), save_global_fsa=save_global_fsa,
        replan_with_feedback_offline=AsyncMock(side_effect=AssertionError("No evidence repair")),
    )
    compiler = BundleCompiler(
        store=BundleStore(tmp_path / "bundles"), project_root=_ROOT,
        product_init_dir=tmp_path, resource_init_dir=tmp_path / "resources",
        cca_init_path=cca, tools_path=tools, prompts_path=prompts,
    )
    monkeypatch.setattr(compiler, "_import_runtime_classes", lambda: (
        lambda *args, **kwargs: SimpleNamespace(
            jid="product@localhost", process_planner=planner, tools_catalog=[], camera=None),
        lambda *args, **kwargs: SimpleNamespace(safety_logic=_logic(Path(kwargs["safety_file"]))),
        PlanSafetyValidator, lambda **kwargs: None,
        SimpleNamespace(configure_shared_tools_catalogue=Mock()),
    ))
    call = compiler.compile_bundle(product_init_file=str(product), execution_mode="mock", robot_env="gazebo")
    if plant_compile_fails:
        with pytest.raises(ValueError, match="predefined_physical_grounding_required"):
            asyncio.run(call)
    else:
        result = asyncio.run(call)
        assert result["manifest"]["verified"] is False
        assert result["manifest"]["status"] == "invalid"
        validation = result["manifest"]["validation_summary"]
        assert validation["ok"] is False
        assert validation["stop_reason"] == "predefined_physical_grounding_required"
        artifact = Path(result["bundle_dir"]) / result["manifest"]["artifacts"]["safety_logic_json"]
        assert validate_predefined_safety_artifact(json.loads(artifact.read_text())) == _document()
    planner.build_high_level.assert_awaited_once()
    planner.expand_requirements_to_tasks.assert_awaited_once()
    planner.replan_with_feedback_offline.assert_not_called()
