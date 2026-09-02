"""Tests for package-local PA, document, and observation model configuration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cais_spade_llm.spec2primitives.config import (
    DEFAULT_MODEL_RUNTIME_CONFIG_PATH,
    load_model_runtime_config,
)


def test_default_model_configuration_uses_gpt_5_6_medium() -> None:
    config = load_model_runtime_config()

    assert config.schema_version == 3
    assert config.product_agent_llm.model == "gpt-5.6"
    assert config.product_agent_llm.reasoning_effort == "medium"
    for vision_config in (config.document_vlm, config.observation_vlm):
        assert vision_config.provider == "openai"
        assert vision_config.model == "gpt-5.6"
        assert vision_config.reasoning_effort == "medium"
        assert vision_config.image_detail == "high"
        assert vision_config.max_output_tokens == 4096
        assert vision_config.timeout_seconds == 90.0
    assert "API_KEY" not in DEFAULT_MODEL_RUNTIME_CONFIG_PATH.read_text(encoding="utf-8")


def test_models_can_be_changed_only_through_the_validated_config_file(
    tmp_path: Path,
) -> None:
    value = _default_value()
    value["product_agent_llm"]["model"] = "controlled-pa-model"
    value["document_vlm"]["model"] = "controlled-vision-model"
    value["observation_vlm"]["model"] = "controlled-observation-model"
    path = tmp_path / "model_runtime.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    config = load_model_runtime_config(path)

    assert config.product_agent_llm.model == "controlled-pa-model"
    assert config.document_vlm.model == "controlled-vision-model"
    assert config.observation_vlm.model == "controlled-observation-model"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"extra": True}),
        lambda value: value.__setitem__("schema_version", 2),
        lambda value: value["product_agent_llm"].__setitem__("unknown", True),
        lambda value: value["product_agent_llm"].__setitem__("reasoning_effort", "maximum"),
        lambda value: value["document_vlm"].__setitem__("provider", "other"),
        lambda value: value["document_vlm"].__setitem__("reasoning_effort", "maximum"),
        lambda value: value["document_vlm"].__setitem__("image_detail", "original"),
        lambda value: value["document_vlm"].__setitem__("max_output_tokens", 0),
        lambda value: value["document_vlm"].__setitem__("timeout_seconds", True),
        lambda value: value["observation_vlm"].__setitem__("provider", "other"),
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
