from __future__ import annotations

"""Load the validated Spec2Primitives workcell profile."""


import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_WORKCELL_PROFILE_PATH = Path(__file__).with_name("workcell_profile.json")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class WorkcellProfileError(ValueError):
    """Raised when the configured workcell profile cannot be trusted."""


@dataclass(frozen=True)
class WorkcellProcessProfile:
    """Identify one exact process authorized for feature grounding."""

    symbol: str
    iri: str


@dataclass(frozen=True)
class WorkcellResourceProfile:
    """Identify one resource, its manifest, and broad process capabilities."""

    symbol: str
    iri: str
    manifest_ref: str
    manifest_path: Path
    capable_process_iris: tuple[str, ...]


@dataclass(frozen=True)
class WorkcellProfile:
    """Hold validated process and resource-capability authorities."""

    processes: tuple[WorkcellProcessProfile, ...]
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
    if not isinstance(value, dict) or set(value) != {"processes", "resources"}:
        raise WorkcellProfileError("Workcell profile fields are invalid.")
    processes = _process_profiles(value["processes"])
    process_iris = {process.iri for process in processes}
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
            "capable_process_iris",
        }:
            raise WorkcellProfileError(f"Workcell resource {index} fields are invalid.")
        symbol = _exact_text(item["symbol"], f"resources[{index}].symbol")
        iri = _absolute_iri(item["iri"], f"resources[{index}].iri")
        manifest_ref = _exact_text(item["manifest_ref"], f"resources[{index}].manifest_ref")
        capable_process_iris = _capable_process_iris(
            item["capable_process_iris"],
            process_iris=process_iris,
            label=f"resources[{index}].capable_process_iris",
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
            raise WorkcellProfileError(f"Workcell resource manifest is unavailable: {manifest_ref}")
        symbols.add(symbol)
        iris.add(iri)
        resources.append(
            WorkcellResourceProfile(
                symbol=symbol,
                iri=iri,
                manifest_ref=manifest_ref,
                manifest_path=manifest_path,
                capable_process_iris=capable_process_iris,
            )
        )
    return WorkcellProfile(
        processes=processes,
        resources=tuple(resources),
        source_path=source_path,
        source_sha256=hashlib.sha256(source).hexdigest(),
    )


def _process_profiles(value: object) -> tuple[WorkcellProcessProfile, ...]:
    if not isinstance(value, list) or not value:
        raise WorkcellProfileError("Workcell processes must be a non-empty list.")
    processes: list[WorkcellProcessProfile] = []
    symbols: set[str] = set()
    iris: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {"symbol", "iri"}:
            raise WorkcellProfileError(f"Workcell process {index} fields are invalid.")
        symbol = _exact_text(item["symbol"], f"processes[{index}].symbol")
        iri = _absolute_iri(item["iri"], f"processes[{index}].iri")
        if symbol in symbols or iri in iris:
            raise WorkcellProfileError("Workcell process identities must be unique.")
        symbols.add(symbol)
        iris.add(iri)
        processes.append(WorkcellProcessProfile(symbol=symbol, iri=iri))
    return tuple(processes)


def _capable_process_iris(
    value: object,
    *,
    process_iris: set[str],
    label: str,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise WorkcellProfileError(f"{label} must be a list.")
    result: list[str] = []
    for index, item in enumerate(value):
        iri = _absolute_iri(item, f"{label}[{index}]")
        if iri not in process_iris:
            raise WorkcellProfileError(f"{label}[{index}] is not an authorized process.")
        if iri in result:
            raise WorkcellProfileError(f"{label} values must be unique.")
        result.append(iri)
    return tuple(result)


def _exact_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise WorkcellProfileError(f"{label} must be exact non-empty text.")
    return value


def _absolute_iri(value: object, label: str) -> str:
    text = _exact_text(value, label)
    if "://" not in text:
        raise WorkcellProfileError(f"{label} must be an absolute IRI.")
    return text
