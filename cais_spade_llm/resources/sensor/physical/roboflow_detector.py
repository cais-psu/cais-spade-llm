"""Roboflow model client and exact gear-label parsing."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

LABEL_TO_PART = {
    "small_gear": ("SG", "gear_small"),
    "medium_gear": ("MG", "gear_medium"),
}
UNSUPPORTED_PARTS = {
    "LG": "LG unavailable: model has no large_gear class",
}


class RoboflowConfigurationError(RuntimeError):
    """Raised when the Roboflow model client is not configured."""


class RoboflowResponseError(RuntimeError):
    """Raised when the Roboflow response does not contain usable predictions."""


class DuplicateDetectionError(RoboflowResponseError):
    """Raised when more than one detection exists for a supported gear class."""


@dataclass(frozen=True)
class RoboflowSettings:
    """Configuration required to invoke the published gear model."""

    api_key: str
    api_url: str = "https://detect.roboflow.com"
    model_id: str = "hrc-assembly-gph6m/5"

    @classmethod
    def from_environment(cls) -> RoboflowSettings:
        """Load settings without logging or exposing the API key."""
        api_key = str(os.environ.get("ROBOFLOW_API_KEY", "")).strip()
        if not api_key:
            raise RoboflowConfigurationError("ROBOFLOW_API_KEY is not configured")
        model_id = str(
            os.environ.get("ROBOFLOW_MODEL_ID", "hrc-assembly-gph6m/5")
        ).strip()
        if not model_id:
            raise RoboflowConfigurationError("ROBOFLOW_MODEL_ID is not configured")
        return cls(
            api_key=api_key,
            api_url=str(
                os.environ.get("ROBOFLOW_API_URL", "https://detect.roboflow.com")
            ).strip(),
            model_id=model_id,
        )


@dataclass(frozen=True)
class GearBoundingBox:
    """One accepted Roboflow gear bounding box in color-image pixels."""

    part_name: str
    model_name: str
    label: str
    confidence: float
    center_x: float
    center_y: float
    width: float
    height: float


def _prediction_rows(result: Any) -> list[dict[str, Any]]:
    """Extract the documented model response field named ``predictions``."""
    outputs = result if isinstance(result, list) else [result]
    for output in outputs:
        if not isinstance(output, dict) or "predictions" not in output:
            continue
        predictions = output["predictions"]
        if isinstance(predictions, dict):
            predictions = predictions.get("predictions")
        if isinstance(predictions, list):
            return [row for row in predictions if isinstance(row, dict)]
        raise RoboflowResponseError("Roboflow response 'predictions' is not a list")
    raise RoboflowResponseError(
        "Roboflow response is missing the 'predictions' field"
    )


def parse_gear_predictions(
    result: Any,
    *,
    minimum_confidence: float = 0.70,
) -> list[GearBoundingBox]:
    """Parse exact SG/MG labels and reject duplicate supported classes."""
    accepted: list[GearBoundingBox] = []
    seen_labels: set[str] = set()
    for row in _prediction_rows(result):
        label = str(row.get("class") or row.get("class_name") or "").strip()
        mapping = LABEL_TO_PART.get(label)
        if mapping is None:
            continue
        confidence = float(row.get("confidence", 0.0))
        if confidence < minimum_confidence:
            continue
        if label in seen_labels:
            raise DuplicateDetectionError(f"duplicate same-class detection rejected: {label}")
        seen_labels.add(label)
        try:
            center_x = float(row["x"])
            center_y = float(row["y"])
            width = float(row["width"])
            height = float(row["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RoboflowResponseError(f"invalid bounding box for {label}") from exc
        if width <= 0.0 or height <= 0.0:
            raise RoboflowResponseError(f"invalid bounding box size for {label}")
        accepted.append(
            GearBoundingBox(
                part_name=mapping[0],
                model_name=mapping[1],
                label=label,
                confidence=confidence,
                center_x=center_x,
                center_y=center_y,
                width=width,
                height=height,
            )
        )
    return accepted


class RoboflowGearDetector:
    """Run the published Roboflow model directly on a NumPy color image."""

    def __init__(
        self,
        settings: RoboflowSettings,
        *,
        minimum_confidence: float = 0.70,
        client: Any | None = None,
    ) -> None:
        self.settings = settings
        self.minimum_confidence = float(minimum_confidence)
        if client is None:
            try:
                from inference_sdk import InferenceHTTPClient
            except ImportError as exc:
                raise RoboflowConfigurationError(
                    "inference-sdk is not installed; run poetry install"
                ) from exc
            client = InferenceHTTPClient(
                api_url=settings.api_url,
                api_key=settings.api_key,
            )
        self._client = client

    def detect(self, color_image: Any) -> list[GearBoundingBox]:
        """Invoke the model and return validated supported detections."""
        try:
            from inference_sdk.http.errors import HTTPClientError
        except ImportError:
            HTTPClientError = OSError
        try:
            result = self._client.infer(
                color_image,
                model_id=self.settings.model_id,
            )
        except (HTTPClientError, OSError, TimeoutError) as exc:
            detail = str(getattr(exc, "description", "") or exc)
            detail = detail.replace(self.settings.api_key, "[redacted]")
            if len(detail) > 600:
                detail = f"{detail[:600].rstrip()}..."
            raise RoboflowResponseError(f"Roboflow model request failed: {detail}") from exc
        return parse_gear_predictions(result, minimum_confidence=self.minimum_confidence)
