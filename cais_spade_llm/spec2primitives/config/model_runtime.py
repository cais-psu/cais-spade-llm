"""Load the model choices used by the isolated Spec2Primitives runtime."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL_RUNTIME_CONFIG_PATH = Path(__file__).with_name("model_runtime.json")

_ROOT_KEYS = {
    "schema_version",
    "product_agent_llm",
    "robot_agent_llm",
    "document_vlm",
    "observation_vlm",
}
_PRODUCT_AGENT_KEYS = {"model", "reasoning_effort"}
_ROBOT_AGENT_KEYS = {"model", "reasoning_effort"}
_DOCUMENT_VLM_KEYS = {
    "provider",
    "model",
    "reasoning_effort",
    "image_detail",
    "max_output_tokens",
    "timeout_seconds",
}


@dataclass(frozen=True)
class ProductAgentModelConfig:
    """Select the shared ProductAgent LLM without changing its interface."""

    model: str
    reasoning_effort: str


@dataclass(frozen=True)
class RobotAgentModelConfig:
    """Select the shared RobotAgent LLM without changing its interface."""

    model: str
    reasoning_effort: str


@dataclass(frozen=True)
class DocumentVLMConfig:
    """Configure the OpenAI document interpretation request."""

    provider: str
    model: str
    reasoning_effort: str
    image_detail: str
    max_output_tokens: int
    timeout_seconds: float


@dataclass(frozen=True)
class ObservationVLMConfig:
    """Configure the OpenAI observation-candidate review request."""

    provider: str
    model: str
    reasoning_effort: str
    image_detail: str
    max_output_tokens: int
    timeout_seconds: float


@dataclass(frozen=True)
class ModelRuntimeConfig:
    """Hold all package-local model configuration."""

    schema_version: int
    product_agent_llm: ProductAgentModelConfig
    robot_agent_llm: RobotAgentModelConfig
    document_vlm: DocumentVLMConfig
    observation_vlm: ObservationVLMConfig


def load_model_runtime_config(
    config_path: Path = DEFAULT_MODEL_RUNTIME_CONFIG_PATH,
) -> ModelRuntimeConfig:
    """Read and strictly validate a model runtime JSON file.

    Args:
        config_path: JSON file containing only the supported model settings.

    Returns:
        Immutable validated model configuration.

    Raises:
        OSError: If the file cannot be read.
        ValueError: If the JSON or any field is invalid.
    """
    path = Path(config_path)
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict) or set(value) != _ROOT_KEYS:
        raise ValueError("Model config fields are invalid.")
    if value["schema_version"] != 4:
        raise ValueError("Model config schema_version must be 4.")

    product_agent = _required_mapping(
        value["product_agent_llm"],
        _PRODUCT_AGENT_KEYS,
        "product_agent_llm",
    )
    robot_agent = _required_mapping(
        value["robot_agent_llm"],
        _ROBOT_AGENT_KEYS,
        "robot_agent_llm",
    )
    document_vlm = _required_mapping(
        value["document_vlm"],
        _DOCUMENT_VLM_KEYS,
        "document_vlm",
    )
    observation_vlm = _required_mapping(
        value["observation_vlm"],
        _DOCUMENT_VLM_KEYS,
        "observation_vlm",
    )
    product_agent_model = _nonempty_string(product_agent["model"], "product_agent_llm.model")
    product_agent_reasoning_effort = _nonempty_string(
        product_agent["reasoning_effort"],
        "product_agent_llm.reasoning_effort",
    )
    if product_agent_reasoning_effort not in {
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
    }:
        raise ValueError("product_agent_llm.reasoning_effort is invalid.")
    if product_agent_model.startswith(
        ("gpt-5.4", "gpt-5.6")
    ) and product_agent_reasoning_effort != "none":
        raise ValueError(
            "product_agent_llm.reasoning_effort must be none for GPT-5.4 or GPT-5.6 "
            "function tools through Chat Completions."
        )
    robot_agent_model = _nonempty_string(
        robot_agent["model"],
        "robot_agent_llm.model",
    )
    robot_agent_reasoning_effort = _nonempty_string(
        robot_agent["reasoning_effort"],
        "robot_agent_llm.reasoning_effort",
    )
    if robot_agent_reasoning_effort not in {
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
    }:
        raise ValueError("robot_agent_llm.reasoning_effort is invalid.")
    document_values = _validated_vision_config(document_vlm, "document_vlm")
    observation_values = _validated_vision_config(
        observation_vlm,
        "observation_vlm",
    )
    return ModelRuntimeConfig(
        schema_version=4,
        product_agent_llm=ProductAgentModelConfig(
            model=product_agent_model,
            reasoning_effort=product_agent_reasoning_effort,
        ),
        robot_agent_llm=RobotAgentModelConfig(
            model=robot_agent_model,
            reasoning_effort=robot_agent_reasoning_effort,
        ),
        document_vlm=DocumentVLMConfig(
            **document_values,
        ),
        observation_vlm=ObservationVLMConfig(
            **observation_values,
        ),
    )


def _validated_vision_config(
    value: Mapping[str, object],
    field_name: str,
) -> dict[str, Any]:
    """Validate one package-owned OpenAI vision configuration."""
    provider = _nonempty_string(value["provider"], f"{field_name}.provider")
    if provider != "openai":
        raise ValueError(f"{field_name}.provider must be openai.")
    model = _nonempty_string(value["model"], f"{field_name}.model")
    reasoning_effort = _nonempty_string(
        value["reasoning_effort"],
        f"{field_name}.reasoning_effort",
    )
    if reasoning_effort not in {"none", "low", "medium", "high", "xhigh"}:
        raise ValueError(f"{field_name}.reasoning_effort is invalid.")
    image_detail = _nonempty_string(
        value["image_detail"],
        f"{field_name}.image_detail",
    )
    if image_detail not in {"auto", "low", "high"}:
        raise ValueError(f"{field_name}.image_detail is invalid.")
    return {
        "provider": provider,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "image_detail": image_detail,
        "max_output_tokens": _positive_int(
            value["max_output_tokens"],
            f"{field_name}.max_output_tokens",
        ),
        "timeout_seconds": _positive_number(
            value["timeout_seconds"],
            f"{field_name}.timeout_seconds",
        ),
    }


def _required_mapping(
    value: object,
    expected_keys: set[str],
    field_name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError(f"{field_name} fields are invalid.")
    return value


def _nonempty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty exact string.")
    return value


def _positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value


def _positive_number(value: object, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"{field_name} must be a positive finite number.")
    return float(value)
