"""Describe neutral RGB-D candidates without assigning product-state roles."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from openai import AsyncOpenAI, OpenAIError
from PIL import Image

if TYPE_CHECKING:
    from cais_spade_llm.spec2primitives.config.model_runtime import ObservationVLMConfig

_GROUNDING_ROOT = Path("products/grounding/rgb_d_cad_grounding")
_PRODUCER = "rgb_d_cad_grounding"
_OUTPUT_NAME = "spec2primitives_observation_candidate_review"
_OUTPUT_KEYS = {"candidates"}
_CANDIDATE_OUTPUT_KEYS = {
    "observation_handle",
    "candidate_handle",
    "description",
    "uncertainty",
}
_PART_IDENTITY_TERM = re.compile(r"\b(?:shaft|pin|peg|gear)s?\b", re.IGNORECASE)


class ObservationCandidateReviewError(ValueError):
    """Raised when candidate review input, output, or persistence is invalid."""


@dataclass(frozen=True)
class ObservationCandidateImage:
    """Hold one opaque candidate and its source and crop image inputs."""

    observation_handle: str
    candidate_handle: str
    source_data_url: str
    crop_data_url: str


@dataclass(frozen=True)
class ObservationCandidateReviewRequest:
    """Contain only opaque candidate identities and visual evidence."""

    candidates: tuple[ObservationCandidateImage, ...]


@dataclass(frozen=True)
class ObservationCandidateReviewResponse:
    """Return one structured neutral review and provider audit fields."""

    response_id: str
    model: str
    output: Mapping[str, object]


@dataclass(frozen=True)
class ObservationCandidateReviewResult:
    """Return one persisted, hash-pinned candidate review."""

    record_path: Path
    crop_paths: tuple[Path, ...]
    record: Mapping[str, object]


class ObservationVisionRuntime(Protocol):
    """Narrow injected boundary for one ontology-neutral observation review."""

    async def review_candidates(
        self,
        request: ObservationCandidateReviewRequest,
    ) -> ObservationCandidateReviewResponse:
        """Describe candidate images without choosing their roles."""
        ...


class OpenAIObservationVisionRuntime:
    """Call the OpenAI Responses API for neutral candidate descriptions."""

    def __init__(
        self,
        config: ObservationVLMConfig,
        *,
        client: Any | None = None,
    ) -> None:
        """Create the adapter with an optional offline-test client."""
        self._config = config
        self._client = client or AsyncOpenAI(
            max_retries=0,
            timeout=config.timeout_seconds,
        )

    async def review_candidates(
        self,
        request: ObservationCandidateReviewRequest,
    ) -> ObservationCandidateReviewResponse:
        """Submit all neutral candidate images in one strict structured request."""
        content: list[dict[str, object]] = [
            {"type": "input_text", "text": _request_text(request.candidates)}
        ]
        previous_observation: str | None = None
        for candidate in request.candidates:
            if candidate.observation_handle != previous_observation:
                content.extend(
                    (
                        {
                            "type": "input_text",
                            "text": f"Source view {candidate.observation_handle}",
                        },
                        {
                            "type": "input_image",
                            "image_url": candidate.source_data_url,
                            "detail": self._config.image_detail,
                        },
                    )
                )
                previous_observation = candidate.observation_handle
            content.extend(
                (
                    {
                        "type": "input_text",
                        "text": (
                            f"Candidate {candidate.observation_handle} / "
                            f"{candidate.candidate_handle}"
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": candidate.crop_data_url,
                        "detail": self._config.image_detail,
                    },
                )
            )
        try:
            response = await self._client.responses.create(
                model=self._config.model,
                instructions=_OPENAI_INSTRUCTIONS,
                input=[{"role": "user", "content": content}],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": _OUTPUT_NAME,
                        "strict": True,
                        "schema": observation_candidate_review_schema(),
                    }
                },
                reasoning={"effort": self._config.reasoning_effort},
                max_output_tokens=self._config.max_output_tokens,
                store=False,
            )
        except OpenAIError as exc:
            raise ObservationCandidateReviewError(
                f"OpenAI observation candidate review failed: {type(exc).__name__}."
            ) from exc
        return _review_response(response)


async def review_observation_candidates(
    *,
    interaction_root: Path,
    segmentation_record_path: Path,
    review_number: int,
    config: ObservationVLMConfig,
    vision_runtime: ObservationVisionRuntime,
) -> ObservationCandidateReviewResult:
    """Create one immutable review for every candidate in a segmentation record."""
    if isinstance(review_number, bool) or not isinstance(review_number, int) or review_number < 1:
        raise ObservationCandidateReviewError("review_number must be a positive integer.")
    root = Path(interaction_root).resolve()
    segmentation_path, segmentation = _load_segmentation(root, segmentation_record_path)
    destination = _GROUNDING_ROOT / f"observation_review_{review_number:04d}"
    destination_path = root / destination
    if destination_path.exists():
        raise ObservationCandidateReviewError(
            f"Observation candidate review {review_number:04d} already exists."
        )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=".observation-review-", dir=destination_path.parent)
    )
    try:
        candidates, candidate_records = _prepare_candidate_images(
            root,
            segmentation,
            temporary_root,
            destination,
        )
        if not candidates:
            raise ObservationCandidateReviewError(
                "Observation candidate review requires at least one segmented candidate."
            )
        response = await vision_runtime.review_candidates(
            ObservationCandidateReviewRequest(candidates=tuple(candidates))
        )
        if not _response_model_matches_config(response.model, config.model):
            raise ObservationCandidateReviewError(
                "Observation review response model does not match configured model."
            )
        reviewed = _validated_output(response.output, candidates)
        for candidate_record, output in zip(candidate_records, reviewed, strict=True):
            candidate_record["description"] = output["description"]
            candidate_record["uncertainty"] = output["uncertainty"]
        segmentation_ref = segmentation_path.relative_to(root).as_posix()
        record: dict[str, object] = {
            "schema_version": 1,
            "record_type": "ObservationCandidateReview",
            "producer": _PRODUCER,
            "review_number": review_number,
            "observation_ref": segmentation["observation_ref"],
            "source_segmentation": {
                "ref": segmentation_ref,
                "sha256": _sha256_path(segmentation_path),
            },
            "provider": config.provider,
            "configured_model": config.model,
            "response_model": response.model,
            "response_id": response.response_id,
            "store": False,
            "status": "accepted",
            "evidence_refs": [segmentation["observation_ref"], segmentation_ref],
            "candidates": candidate_records,
        }
        record["fingerprint"] = _fingerprint(record)
        record_path = temporary_root / "observation_candidate_review.json"
        _write_json(record_path, record)
        temporary_root.rename(destination_path)
    except ObservationCandidateReviewError:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    except (OSError, TypeError, ValueError) as exc:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise ObservationCandidateReviewError(
            f"Observation candidate review failed: {type(exc).__name__}: {exc}"
        ) from exc
    final_record_path = destination_path / "observation_candidate_review.json"
    return ObservationCandidateReviewResult(
        record_path=final_record_path,
        crop_paths=tuple(
            destination_path / str(candidate["crop_artifact"]["ref"]).split("/")[-1]
            for candidate in candidate_records
        ),
        record=record,
    )


def observation_candidate_review_schema() -> dict[str, object]:
    """Return the strict output schema for neutral candidate descriptions."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["candidates"],
        "properties": {
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": sorted(_CANDIDATE_OUTPUT_KEYS),
                    "properties": {
                        "observation_handle": {"type": "string", "minLength": 1},
                        "candidate_handle": {"type": "string", "minLength": 1},
                        "description": {"type": "string", "minLength": 1},
                        "uncertainty": {"type": "string", "minLength": 1},
                    },
                },
            }
        },
    }


def _prepare_candidate_images(
    root: Path,
    segmentation: Mapping[str, object],
    temporary_root: Path,
    destination: Path,
) -> tuple[list[ObservationCandidateImage], list[dict[str, object]]]:
    cameras = segmentation.get("cameras")
    if not isinstance(cameras, list):
        raise ObservationCandidateReviewError("Segmentation cameras are invalid.")
    inputs: list[ObservationCandidateImage] = []
    records: list[dict[str, object]] = []
    for camera_index, camera in enumerate(cameras):
        if not isinstance(camera, Mapping):
            raise ObservationCandidateReviewError("Segmentation camera is invalid.")
        observation_handle = _nonempty(camera.get("observation_handle"), "observation_handle")
        candidates = camera.get("candidates")
        artifacts = camera.get("source_artifacts")
        rgb = artifacts.get("rgb") if isinstance(artifacts, Mapping) else None
        if not isinstance(candidates, list) or not isinstance(rgb, Mapping):
            raise ObservationCandidateReviewError("Candidate source RGB is invalid.")
        rgb_ref = _nonempty(rgb.get("ref"), "source RGB ref")
        rgb_sha256 = _nonempty(rgb.get("sha256"), "source RGB sha256")
        rgb_path = _resolved_ref(root, rgb_ref)
        if not rgb_path.is_file() or _sha256_path(rgb_path) != rgb_sha256:
            raise ObservationCandidateReviewError("Candidate source RGB hash is invalid.")
        source_data_url = _image_data_url(rgb_path.read_bytes())
        with Image.open(rgb_path) as source_image:
            source = source_image.convert("RGB")
            for candidate_index, candidate in enumerate(candidates):
                if not isinstance(candidate, Mapping):
                    raise ObservationCandidateReviewError("Segmentation candidate is invalid.")
                candidate_handle = _nonempty(
                    candidate.get("candidate_handle"),
                    "candidate_handle",
                )
                bounds = _pixel_bounds(candidate.get("pixel_bounds_uv"), source.size)
                crop = source.crop(_expanded_bounds(bounds, source.size))
                buffer = io.BytesIO()
                crop.save(buffer, format="PNG")
                crop_bytes = buffer.getvalue()
                crop_name = f"{observation_handle}_{candidate_handle}.png"
                crop_path = temporary_root / crop_name
                crop_path.write_bytes(crop_bytes)
                crop_ref = (destination / crop_name).as_posix()
                inputs.append(
                    ObservationCandidateImage(
                        observation_handle=observation_handle,
                        candidate_handle=candidate_handle,
                        source_data_url=source_data_url,
                        crop_data_url=_image_data_url(crop_bytes),
                    )
                )
                records.append(
                    {
                        "observation_handle": observation_handle,
                        "candidate_handle": candidate_handle,
                        "source_field_path": (
                            f"/cameras/{camera_index}/candidates/{candidate_index}"
                        ),
                        "pixel_bounds_uv": candidate["pixel_bounds_uv"],
                        "source_rgb": {"ref": rgb_ref, "sha256": rgb_sha256},
                        "crop_artifact": {
                            "ref": crop_ref,
                            "sha256": hashlib.sha256(crop_bytes).hexdigest(),
                        },
                    }
                )
    return inputs, records


def _load_segmentation(root: Path, record_path: Path) -> tuple[Path, Mapping[str, object]]:
    path = Path(record_path).resolve()
    try:
        relative = path.relative_to(root)
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ObservationCandidateReviewError(
            "RGBDSegmentationRecord could not be read from this interaction."
        ) from exc
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != 2
        or value.get("record_type") != "RGBDSegmentationRecord"
        or value.get("producer") != _PRODUCER
        or relative.parts[:3] != _GROUNDING_ROOT.parts
    ):
        raise ObservationCandidateReviewError("RGBDSegmentationRecord is invalid.")
    return path, value


def _validated_output(
    value: Mapping[str, object],
    expected: Sequence[ObservationCandidateImage],
) -> list[dict[str, str]]:
    if not isinstance(value, Mapping) or set(value) != _OUTPUT_KEYS:
        raise ObservationCandidateReviewError("Observation review output fields are invalid.")
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(expected):
        raise ObservationCandidateReviewError(
            "Observation review must cover every candidate exactly once."
        )
    reviewed: list[dict[str, str]] = []
    for index, (item, expected_item) in enumerate(zip(candidates, expected, strict=True)):
        if not isinstance(item, Mapping) or set(item) != _CANDIDATE_OUTPUT_KEYS:
            raise ObservationCandidateReviewError(
                f"Observation review candidate {index} fields are invalid."
            )
        observation_handle = _nonempty(item["observation_handle"], "observation_handle")
        candidate_handle = _nonempty(item["candidate_handle"], "candidate_handle")
        description = _nonempty(item["description"], "description")
        uncertainty = _nonempty(item["uncertainty"], "uncertainty")
        if _PART_IDENTITY_TERM.search(description) or _PART_IDENTITY_TERM.search(uncertainty):
            raise ObservationCandidateReviewError(
                "Observation review must describe morphology without inferred part identity."
            )
        if (
            observation_handle != expected_item.observation_handle
            or candidate_handle != expected_item.candidate_handle
        ):
            raise ObservationCandidateReviewError(
                "Observation review candidate order or identity is invalid."
            )
        reviewed.append(
            {
                "observation_handle": observation_handle,
                "candidate_handle": candidate_handle,
                "description": description,
                "uncertainty": uncertainty,
            }
        )
    return reviewed


def _review_response(response: object) -> ObservationCandidateReviewResponse:
    output_text = getattr(response, "output_text", None)
    response_id = getattr(response, "id", None)
    model = getattr(response, "model", None)
    if not all(isinstance(value, str) and value for value in (output_text, response_id, model)):
        raise ObservationCandidateReviewError("Observation review response metadata is invalid.")
    try:
        output = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise ObservationCandidateReviewError("Observation review response is not JSON.") from exc
    if not isinstance(output, Mapping):
        raise ObservationCandidateReviewError("Observation review response must be an object.")
    return ObservationCandidateReviewResponse(
        response_id=response_id,
        model=model,
        output=output,
    )


def _request_text(candidates: Sequence[ObservationCandidateImage]) -> str:
    handles = [
        {
            "observation_handle": candidate.observation_handle,
            "candidate_handle": candidate.candidate_handle,
        }
        for candidate in candidates
    ]
    return (
        "Describe every supplied candidate in this exact order. The full source view "
        "provides spatial context and the following image is the candidate crop. Report "
        "only visible shape, relative size, color, openings, cylindrical forms, surfaces, "
        "and support contacts or relationships. Describe morphology without naming a "
        "candidate as a shaft, pin, peg, gear, or any other inferred part identity. "
        "Preserve the opaque handles exactly. Do not assign current_state, desired_state, "
        "a process, a CAD identity, or a resource. State uncertainty rather than guessing. "
        f"Expected handles:\n{json.dumps(handles, indent=2)}"
    )


_OPENAI_INSTRUCTIONS = (
    "You are the ontology-neutral observation reviewer for Spec2Primitives. Use only "
    "the supplied images and opaque handles. Describe only morphology, dimensions, "
    "surfaces, and support contacts. Do not infer a part identity such as shaft or pin, "
    "and do not choose a product state, CAD file, process, robot, or execution action."
)


def _pixel_bounds(value: object, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    if not isinstance(value, Mapping) or set(value) != {"minimum", "maximum"}:
        raise ObservationCandidateReviewError("Candidate pixel bounds are invalid.")
    minimum = value["minimum"]
    maximum = value["maximum"]
    if (
        not isinstance(minimum, list)
        or not isinstance(maximum, list)
        or len(minimum) != 2
        or len(maximum) != 2
        or not all(
            isinstance(item, int) and not isinstance(item, bool) for item in (*minimum, *maximum)
        )
    ):
        raise ObservationCandidateReviewError("Candidate pixel bounds are invalid.")
    left, top = minimum
    right, bottom = maximum
    width, height = image_size
    if not (0 <= left <= right < width and 0 <= top <= bottom < height):
        raise ObservationCandidateReviewError("Candidate pixel bounds leave the source image.")
    return left, top, right + 1, bottom + 1


def _expanded_bounds(
    bounds: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bounds
    padding = max(12, round(max(right - left, bottom - top) * 0.20))
    width, height = image_size
    return (
        max(0, left - padding),
        max(0, top - padding),
        min(width, right + padding),
        min(height, bottom + padding),
    )


def _response_model_matches_config(response_model: str, configured_model: str) -> bool:
    return response_model == configured_model or (
        configured_model == "gpt-5.6" and response_model == "gpt-5.6-sol"
    )


def _resolved_ref(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ObservationCandidateReviewError("Observation artifact ref is invalid.")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ObservationCandidateReviewError(
            "Observation artifact ref leaves interaction."
        ) from exc
    return path


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ObservationCandidateReviewError(f"{field} is invalid.")
    return value


def _image_data_url(source: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(source).decode("ascii")


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(value: Mapping[str, object]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
