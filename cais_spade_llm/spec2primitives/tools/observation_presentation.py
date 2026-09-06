from __future__ import annotations

"""Present observations without exposing sensor order or segmentation rank."""

import hashlib
import json
import re
import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PRESENTATION_REF = "products/grounding/presentation/observation_presentation_record.json"
_TOKEN = re.compile(
    r"/cameras/[0-9]+/candidates/[0-9]+|\bview_[0-9]{4}\b|\bcandidate_[0-9]{4}_[0-9]{4}\b"
)
_MODEL_TOKEN = re.compile(
    r"/candidates/observation_[0-9a-f]{24}|\b(?:view|candidate)_[0-9a-f]{24}\b"
)


def _fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class ObservationPresentation:
    """Pin a random interaction-local mapping; resolve only observed exact identities.

    The immutable key and mapping algorithm define the mapping for every pinned
    segmentation. No camera order or candidate rank is recoverable from a handle.
    Canonical pointers, calibration frames and source artifacts stay on the host.
    """

    def __init__(self, root: Path, *, create: bool = False) -> None:
        """Load the mapping, creating it only at an authorized observation boundary."""
        self.root = Path(root).resolve()
        self.path = self.root / PRESENTATION_REF
        if create and not self.path.exists():
            record = {
                "record_type": "ObservationPresentationRecord",
                "mapping_algorithm": "sha256_exact_reference_v1",
                "key": secrets.token_hex(32),
            }
            record["fingerprint"] = _fingerprint(record)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("x") as stream:
                json.dump(record, stream, indent=2)
                stream.write("\n")
        self.record = json.loads(self.path.read_text()) if self.path.exists() else None
        self.assert_unchanged()

    def assert_unchanged(self) -> None:
        """Reject altered mapping authority without silently generating a replacement."""
        if self.record is None:
            return
        payload = dict(self.record)
        fingerprint = payload.pop("fingerprint", None)
        if (
            (self.record != json.loads(self.path.read_text()))
            or (fingerprint != _fingerprint(payload))
            or (payload.get("record_type") != "ObservationPresentationRecord")
            or (payload.get("mapping_algorithm") != "sha256_exact_reference_v1")
            or (not re.fullmatch(r"[0-9a-f]{64}", str(payload.get("key", ""))))
        ):
            raise ValueError("Observation presentation changed or is invalid.")

    def handle(self, value: str) -> str:
        """Return one opaque handle for an exact host identity or candidate pointer."""
        if self.record is None:
            return value
        prefix = "/candidates/observation" if value.startswith("/cameras/") else value.split("_")[0]
        digest = hashlib.sha256(f"{self.record['key']}\0{value}".encode()).hexdigest()[:24]
        return f"{prefix}_{digest}"

    def project(self, value: object) -> Any:
        """Map metadata and randomize candidate/view lists while retaining source content."""
        self.assert_unchanged()

        def visit(item: object, key: str = "") -> Any:
            if key == "extracted_text":
                return item
            if isinstance(item, str):
                return _TOKEN.sub(lambda match: self.handle(match.group()), item)
            if isinstance(item, Mapping):
                return {name: visit(nested, str(name)) for name, nested in item.items()}
            if isinstance(item, (list, tuple)):
                values = [visit(nested, key) for nested in item]
                if self.record is not None and key in {
                    "views",
                    "candidates",
                    "candidate_measurements",
                }:
                    values.sort(
                        key=lambda nested: str(
                            nested.get("candidate_handle", nested.get("observation_handle", ""))
                        )
                        if isinstance(nested, Mapping)
                        else _fingerprint(nested)
                    )
                return values
            return item

        return visit(value)

    def resolve(self, value: object) -> Any:
        """Resolve issued model handles against canonical segmentation records only."""
        self.assert_unchanged()
        if self.record is None:
            return value
        reverse: dict[str, str] = {}
        for path in (self.root / "products/grounding/rgb_d_cad_grounding").glob(
            "segmentation_*/segmentation_record.json"
        ):
            segmentation = json.loads(path.read_text())
            for camera_index, camera in enumerate(segmentation["cameras"]):
                view = camera["observation_handle"]
                reverse[self.handle(view)] = view
                for candidate_index, candidate in enumerate(camera["candidates"]):
                    handle = candidate["candidate_handle"]
                    pointer = f"/cameras/{camera_index}/candidates/{candidate_index}"
                    reverse[self.handle(handle)] = handle
                    reverse[self.handle(pointer)] = pointer

        def replace(match: re.Match[str]) -> str:
            try:
                return reverse[match.group()]
            except KeyError as exc:
                raise ValueError("Observation reference was not presented.") from exc

        def visit(item: object, key: str = "") -> Any:
            if key == "extracted_text":
                return item
            if isinstance(item, str):
                if key in {
                    "observation_handle",
                    "candidate_handle",
                    "field_path",
                    "candidate_field_paths",
                } and _TOKEN.search(item):
                    raise ValueError("Canonical observation metadata was not presented.")
                return _MODEL_TOKEN.sub(replace, item)
            if isinstance(item, Mapping):
                return {name: visit(nested, str(name)) for name, nested in item.items()}
            if isinstance(item, (list, tuple)):
                return [visit(nested, key) for nested in item]
            return item

        return visit(value)
