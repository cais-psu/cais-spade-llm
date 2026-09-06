from __future__ import annotations

"""Tests for package-local PA, document, and observation model configuration."""


import json
from pathlib import Path

import pytest

from cais_spade_llm.spec2primitives.config import (
    DEFAULT_MODEL_RUNTIME_CONFIG_PATH,
    GroundingLimits,
    load_model_runtime_config,
)


def test_default_model_configuration_uses_compatible_gpt_5_6_efforts() -> None:
    config = load_model_runtime_config()

    assert not hasattr(config, "schema_version")
    assert config.product_agent_llm.model == "gpt-5.6"
    assert config.product_agent_llm.reasoning_effort == "none"
    assert config.robot_agent_llm.model == "gpt-5.6"
    assert config.robot_agent_llm.reasoning_effort == "medium"
    assert config.grounding_limits == GroundingLimits(max_evidence_operations=24, max_proposals=6)
    for vision_config in (config.document_vlm, config.observation_vlm):
        assert vision_config.provider == "openai"
        assert vision_config.model == "gpt-5.6"
        assert vision_config.reasoning_effort == "medium"
        assert vision_config.image_detail == "high"
        assert vision_config.max_output_tokens == 4096
        assert vision_config.timeout_seconds == 90.0
    assert "API_KEY" not in DEFAULT_MODEL_RUNTIME_CONFIG_PATH.read_text(encoding="utf-8")


@pytest.mark.parametrize("model", ["gpt-5.4", "gpt-5.6", "gpt-5.6-sol"])
def test_chat_completions_tool_models_require_none_reasoning_effort(
    tmp_path: Path,
    model: str,
) -> None:
    value = _default_value()
    value["product_agent_llm"]["model"] = model
    value["product_agent_llm"]["reasoning_effort"] = "medium"
    path = tmp_path / "incompatible.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="reasoning_effort must be none",
    ):
        load_model_runtime_config(path)


def test_models_can_be_changed_only_through_the_validated_config_file(
    tmp_path: Path,
) -> None:
    value = _default_value()
    value["product_agent_llm"]["model"] = "controlled-pa-model"
    value["robot_agent_llm"]["model"] = "controlled-ra-model"
    value["document_vlm"]["model"] = "controlled-vision-model"
    value["observation_vlm"]["model"] = "controlled-observation-model"
    path = tmp_path / "model_runtime.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    config = load_model_runtime_config(path)

    assert config.product_agent_llm.model == "controlled-pa-model"
    assert config.robot_agent_llm.model == "controlled-ra-model"
    assert config.document_vlm.model == "controlled-vision-model"
    assert config.observation_vlm.model == "controlled-observation-model"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"extra": True}),
        lambda value: value.__setitem__("schema_version", 2),
        lambda value: value["product_agent_llm"].__setitem__("unknown", True),
        lambda value: value["product_agent_llm"].__setitem__("reasoning_effort", "maximum"),
        lambda value: value["robot_agent_llm"].__setitem__("unknown", True),
        lambda value: value["robot_agent_llm"].__setitem__("reasoning_effort", "maximum"),
        lambda value: value["document_vlm"].__setitem__("provider", "other"),
        lambda value: value["document_vlm"].__setitem__("reasoning_effort", "maximum"),
        lambda value: value["document_vlm"].__setitem__("image_detail", "original"),
        lambda value: value["document_vlm"].__setitem__("max_output_tokens", 0),
        lambda value: value["document_vlm"].__setitem__("timeout_seconds", True),
        lambda value: value["observation_vlm"].__setitem__("provider", "other"),
        lambda value: value["grounding_limits"].__setitem__("unknown", 1),
        lambda value: value["grounding_limits"].__setitem__("max_evidence_operations", 0),
        lambda value: value["grounding_limits"].__setitem__("max_proposals", True),
    ],
)
def test_invalid_or_unknown_model_configuration_is_rejected(
    tmp_path: Path,
    mutation: object,
) -> None:
    value = _default_value()
    mutation(value)  # type: ignore[operator]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError):
        load_model_runtime_config(path)


def _default_value() -> dict[str, object]:
    return json.loads(DEFAULT_MODEL_RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "6", None])
@pytest.mark.parametrize("field", ["max_evidence_operations", "max_proposals"])
def test_grounding_limits_require_positive_integers(field, value) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        GroundingLimits(**{field: value})


def test_grounding_limits_are_loaded_without_changing_model_settings(tmp_path: Path) -> None:
    value = _default_value()
    value["grounding_limits"] = {"max_evidence_operations": 9, "max_proposals": 4}
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    loaded = load_model_runtime_config(path)
    assert loaded.grounding_limits == GroundingLimits(9, 4)
    assert loaded.product_agent_llm == load_model_runtime_config().product_agent_llm
