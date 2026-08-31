"""Load the validated Spec2Primitives workcell profile."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_WORKCELL_PROFILE_PATH = Path(__file__).with_name("workcell_profile.json")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class WorkcellProfileError(ValueError):
    """Raised when the configured workcell profile cannot be trusted."""


@dataclass(frozen=True)
class WorkcellResourceProfile:
    """Identify one ordered resource and its authoritative manifest."""

    symbol: str
    iri: str
    manifest_ref: str
    manifest_path: Path


@dataclass(frozen=True)
class WorkcellProfile:
    """Hold the validated process and ordered resource authorities."""

    process_symbol: str
    process_iri: str
    resources: tuple[WorkcellResourceProfile, ...]
    source_path: Path
    source_sha256: str

    def assert_unchanged(self) -> None:
        """Raise if the profile bytes changed after validation."""
        try:
            digest = hashlib.sha256(self.source_path.read_bytes()).hexdigest()
        except OSError as exc:
            raise WorkcellProfileError("Workcell profile is unavailable.") from exc
        if digest != self.source_sha256:
            raise WorkcellProfileError("Workcell profile changed after validation.")


def load_workcell_profile(
    path: Path = DEFAULT_WORKCELL_PROFILE_PATH,
    *,
    repository_root: Path = _REPOSITORY_ROOT,
) -> WorkcellProfile:
    """Load one exact, ordered workcell profile without product semantics."""
    source_path = Path(path).resolve()
    root = Path(repository_root).resolve()
    try:
        source = source_path.read_bytes()
        value = json.loads(source.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkcellProfileError("Workcell profile is not readable JSON.") from exc
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "process",
        "resources",
    }:
        raise WorkcellProfileError("Workcell profile fields are invalid.")
    if value["schema_version"] != 1:
        raise WorkcellProfileError("Workcell profile schema version is unsupported.")
    process = value["process"]
    if not isinstance(process, dict) or set(process) != {"symbol", "iri"}:
        raise WorkcellProfileError("Workcell process identity is invalid.")
    process_symbol = _exact_text(process["symbol"], "process.symbol")
    process_iri = _absolute_iri(process["iri"], "process.iri")
    raw_resources = value["resources"]
    if not isinstance(raw_resources, list) or not raw_resources:
        raise WorkcellProfileError("Workcell resources must be a non-empty list.")
    resources: list[WorkcellResourceProfile] = []
    symbols: set[str] = set()
    iris: set[str] = set()
    for index, item in enumerate(raw_resources):
        if not isinstance(item, dict) or set(item) != {
            "symbol",
            "iri",
            "manifest_ref",
        }:
            raise WorkcellProfileError(f"Workcell resource {index} fields are invalid.")
        symbol = _exact_text(item["symbol"], f"resources[{index}].symbol")
        iri = _absolute_iri(item["iri"], f"resources[{index}].iri")
        manifest_ref = _exact_text(
            item["manifest_ref"], f"resources[{index}].manifest_ref"
        )
        manifest_path = (root / manifest_ref).resolve()
        try:
            manifest_path.relative_to(root)
        except ValueError as exc:
            raise WorkcellProfileError(
                f"Workcell resource {index} manifest leaves the repository."
            ) from exc
        if symbol in symbols or iri in iris:
            raise WorkcellProfileError("Workcell resource identities must be unique.")
        if not manifest_path.is_file():
            raise WorkcellProfileError(
                f"Workcell resource manifest is unavailable: {manifest_ref}"
            )
        symbols.add(symbol)
        iris.add(iri)
        resources.append(
            WorkcellResourceProfile(
                symbol=symbol,
                iri=iri,
                manifest_ref=manifest_ref,
                manifest_path=manifest_path,
            )
        )
    return WorkcellProfile(
        process_symbol=process_symbol,
        process_iri=process_iri,
        resources=tuple(resources),
        source_path=source_path,
        source_sha256=hashlib.sha256(source).hexdigest(),
    )


def _exact_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise WorkcellProfileError(f"{label} must be exact non-empty text.")
    return value


def _absolute_iri(value: object, label: str) -> str:
    text = _exact_text(value, label)
    if "://" not in text:
        raise WorkcellProfileError(f"{label} must be an absolute IRI.")
    return text
