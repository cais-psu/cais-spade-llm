"""Experiment runner for offline planning and task-allocation studies."""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

from cais_spade_llm.bundles import BundleCompiler, BundleStore
from cais_spade_llm.bundles.models import (
    atomic_json_write,
    sha256_file,
    sha256_text,
    utc_now_compact,
)

_PKG_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _PKG_ROOT.parent
_BASE = _PROJECT_ROOT / "cais_spade_llm"
_DEFAULT_PRODUCT_INIT = _BASE / "initialization" / "products" / "assembly_board-v1.json"
_DEFAULT_CCA_INIT = _BASE / "initialization" / "cca.json"
_DEFAULT_PROMPTS = _BASE / "prompts.py"
_DEFAULT_RESULTS_ROOT = _PROJECT_ROOT / "writing" / "experiments" / "results"
_PRODUCT_DIR = _BASE / "initialization" / "products"
_RESOURCE_DIR = _BASE / "initialization" / "resources"
_PRODUCT_REQUIREMENTS_DIR = _BASE / "specification" / "products" / "requirements"
_SAFETY_REQUIREMENTS_DIR = _BASE / "specification" / "safety"
_USER_VERIFIED_SAFETY = _BASE / "user_verified_safety"
_SAFETY_INTENT_APPROVALS = _USER_VERIFIED_SAFETY / "intent_approvals.json"
_SAFETY_INTENT_PREVIEWS = _USER_VERIFIED_SAFETY / "intent_previews.json"
_EXPERIMENT_RESOURCES_DIR = _PROJECT_ROOT / "writing" / "experiments" / "resources"
_DEFAULT_AUTO_REPLAN_MAX_ATTEMPTS = 5
_DEFAULT_METHODS = ["llm_nl_safety", "verified"]
_METHOD_ALIAS_MAP = {
    "pure_llm": "llm_nl_safety",
    "llm_nl_safety": "llm_nl_safety",
    "verified": "verified",
}
_SOURCE_LOCATION_ORDER = [
    "prusa-mk4-1",
    "prusa-mk4-2",
    "prusa-mk3",
    "prusa-mk3-2",
    "assembly_board-v1",
]
_REQUIREMENT_ASSEMBLY_LINE_RE = re.compile(
    r"^\s*-\s*(?:(?P<resource>[A-Za-z0-9_-]+)\s+)?assemble\s+"
    r"(?P<part>[A-Za-z0-9_-]+)\s+from\s+(?P<source>[A-Za-z0-9_.@-]+)\s+to\b",
    flags=re.IGNORECASE,
)


def _make_experiment_logger() -> logging.Logger:
    logger = logging.getLogger("agent:experiment")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    try:
        log_dir = _BASE / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_dir / "experiment_actions.log",
            mode="a",
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception:
        pass
    logger.propagate = False
    return logger


log = _make_experiment_logger()


@dataclass(frozen=True)
class ScenarioWorkspace:
    """Resolved workspace paths for a single study scenario."""

    scenario_id: str
    scenario_dir: Path
    product_init_file: Path
    cca_init_file: Path
    resources_dir: Path
    requirements_file: Path
    safety_file: Path
    tools_path: Path
    active_resource_names: list[str]
    parts: list[str]
    precomputed_safety_artifacts: dict[str, Any] | None = None


class OfflineStudyRunner:
    """Run offline planning study scenarios from a JSON manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        project_root: str | Path | None = None,
        results_root: str | Path | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.project_root = Path(project_root).resolve() if project_root else _PROJECT_ROOT
        self.results_root = Path(results_root).resolve() if results_root else _DEFAULT_RESULTS_ROOT

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"expected JSON object at {path}")
        return payload

    @staticmethod
    def _first_named_entry(raw: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if "name" in raw:
            name = str(raw.get("name", "resource")).strip() or "resource"
            return name, raw
        if not raw:
            raise ValueError("empty manifest")
        first_key = next(iter(raw.keys()))
        meta = raw[first_key]
        if not isinstance(meta, dict):
            raise ValueError("invalid manifest shape")
        return str(first_key), meta

    @staticmethod
    def _as_str_list(raw: Any) -> list[str]:
        out: list[str] = []
        for item in raw if isinstance(raw, list) else []:
            token = str(item or "").strip()
            if token:
                out.append(token)
        return out

    @staticmethod
    def _mean_number(values: list[float | int]) -> float:
        if not values:
            return 0.0
        return float(mean(values))

    @staticmethod
    def _mean_optional_number(values: list[float | int | None]) -> float | None:
        filtered = [float(value) for value in values if value is not None]
        if not filtered:
            return None
        return float(mean(filtered))

    @staticmethod
    def _rule_satisfaction_was_evaluated(row: dict[str, Any]) -> bool:
        if row.get("rule_satisfaction_evaluated") is False:
            return False
        stop_reason = str(row.get("stop_reason", "") or "")
        validation_call_count = int(row.get("validation_call_count", 0) or 0)
        return not (stop_reason == "grounding_invalid" and validation_call_count == 0)

    @staticmethod
    def _trial_rule_satisfaction_rate(row: dict[str, Any]) -> float | None:
        if not OfflineStudyRunner._rule_satisfaction_was_evaluated(row):
            return None
        raw_rate = row.get("rule_satisfaction_rate")
        if raw_rate is not None:
            return float(raw_rate)
        if "violated_rules" not in row and "violated_rule_count" not in row:
            return None
        safety_rules = int(row.get("safety_rules", 0) or 0)
        if not safety_rules:
            return None
        violated = row.get("violated_rules", [])
        violated_count = len(
            {str(rule_id) for rule_id in violated if str(rule_id or "").strip()}
        ) if isinstance(violated, list) else int(row.get("violated_rule_count", 0) or 0)
        return max(0, safety_rules - min(safety_rules, violated_count)) / safety_rules

    @staticmethod
    def _repair_history_violated_count(entry: dict[str, Any]) -> int | None:
        if not isinstance(entry, dict):
            return None
        raw_count = entry.get("violated_rule_count")
        if raw_count not in (None, ""):
            try:
                return max(0, int(raw_count))
            except (TypeError, ValueError):
                pass
        violated_rules = entry.get("violated_rules")
        if isinstance(violated_rules, list):
            return len({str(rule_id) for rule_id in violated_rules if str(rule_id or "").strip()})
        safety_rule_count = entry.get("safety_rule_count")
        satisfied_rule_count = entry.get("satisfied_rule_count")
        if safety_rule_count not in (None, "") and satisfied_rule_count not in (None, ""):
            try:
                return max(0, int(safety_rule_count) - int(satisfied_rule_count))
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _repair_corrections_dict(row: dict[str, Any]) -> dict[int, float]:
        raw = row.get("repair_corrections_by_attempt")
        if not isinstance(raw, dict):
            return {}
        corrections: dict[int, float] = {}
        for raw_attempt, raw_value in raw.items():
            try:
                attempt = int(raw_attempt)
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if attempt > 0:
                corrections[attempt] = value
        return dict(sorted(corrections.items()))

    @staticmethod
    def _format_repair_correction_summary(corrections: dict[int, float]) -> str:
        if not corrections:
            return "none"
        return ", ".join(
            f"R{attempt}: +{float(value):.2f}"
            for attempt, value in sorted(corrections.items())
        )

    @classmethod
    def _repair_progress_from_history(
        cls,
        repair_history: list[dict[str, Any]],
        *,
        fallback_violated_rule_count: int | None = None,
    ) -> dict[str, Any]:
        initial_violated: int | None = None
        final_violated: int | None = fallback_violated_rule_count
        previous_validated_violated: int | None = None
        corrections_by_attempt: dict[int, float] = {}

        for entry in repair_history:
            if not isinstance(entry, dict):
                continue
            phase = str(entry.get("phase", "") or "").strip()
            try:
                attempt = int(entry.get("attempt_index", 0) or 0)
            except (TypeError, ValueError):
                attempt = 0
            if phase == "repair" and attempt > 0:
                corrections_by_attempt.setdefault(attempt, 0.0)
                continue
            if phase != "validation":
                continue
            violated = cls._repair_history_violated_count(entry)
            if violated is None:
                continue
            if initial_violated is None:
                initial_violated = violated
            elif attempt > 0 and previous_validated_violated is not None:
                corrections_by_attempt[attempt] = max(
                    corrections_by_attempt.get(attempt, 0.0),
                    float(max(0, previous_validated_violated - violated)),
                )
            previous_validated_violated = violated
            final_violated = violated

        if initial_violated is None:
            initial_violated = fallback_violated_rule_count
        if final_violated is None:
            final_violated = fallback_violated_rule_count

        return {
            "initial_violated_rule_count": initial_violated,
            "final_violated_rule_count": final_violated,
            "repair_corrections_by_attempt": dict(sorted(corrections_by_attempt.items())),
            "repair_correction_summary": cls._format_repair_correction_summary(corrections_by_attempt),
        }

    @staticmethod
    def _read_text_file(path: Path) -> str:
        return path.read_text(encoding="utf-8").strip()

    @classmethod
    def _parse_requirement_text_layout(
        cls,
        requirement_text: str,
        *,
        valid_parts: set[str] | None = None,
    ) -> dict[str, Any]:
        ordered_parts: list[str] = []
        part_sources: dict[str, str] = {}
        seen_parts: set[str] = set()
        valid = set(valid_parts or set())
        for raw_line in str(requirement_text or "").splitlines():
            match = _REQUIREMENT_ASSEMBLY_LINE_RE.match(str(raw_line or "").strip())
            if not match:
                continue
            part = str(match.group("part") or "").strip()
            source = str(match.group("source") or "").strip()
            if not part:
                continue
            if valid and part not in valid:
                continue
            if part not in seen_parts:
                seen_parts.add(part)
                ordered_parts.append(part)
            if source and part not in part_sources:
                part_sources[part] = source
        return {
            "derived": bool(ordered_parts),
            "parts": ordered_parts,
            "part_order": list(ordered_parts),
            "part_sources": {part: str(part_sources.get(part, "") or "") for part in ordered_parts},
        }

    @classmethod
    def parse_requirement_file_layout(
        cls,
        product_requirement_file: str | Path,
        *,
        project_root: str | Path | None = None,
        valid_parts: set[str] | None = None,
    ) -> dict[str, Any]:
        raw = str(product_requirement_file or "").strip()
        if not raw:
            return {"derived": False, "parts": [], "part_order": [], "part_sources": {}}
        base = Path(project_root).resolve() if project_root else _PROJECT_ROOT
        path = Path(raw)
        if not path.is_absolute():
            path = (base / path).resolve()
        if not path.exists() or not path.is_file():
            return {"derived": False, "parts": [], "part_order": [], "part_sources": {}}
        try:
            requirement_text = cls._read_text_file(path)
        except Exception:
            return {"derived": False, "parts": [], "part_order": [], "part_sources": {}}
        return cls._parse_requirement_text_layout(requirement_text, valid_parts=valid_parts)

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._json_safe(item) for item in value]
        return value

    def _resolve_repo_path(self, raw_value: str | Path) -> Path:
        raw = str(raw_value or "").strip()
        if not raw:
            raise ValueError("path value is required")
        p = Path(raw)
        if p.is_absolute():
            return p.resolve()
        return (self.project_root / p).resolve()

    @staticmethod
    def _norm_abs_path(raw_value: str | Path) -> str:
        raw = str(raw_value or "").strip()
        if not raw:
            return ""
        return str(Path(raw).resolve())

    @staticmethod
    def _repo_ref(raw_value: str | Path, *, project_root: Path = _PROJECT_ROOT) -> str:
        raw = str(raw_value or "").strip()
        if not raw:
            return ""
        candidate = Path(raw)
        if not candidate.is_absolute():
            return candidate.as_posix()
        try:
            return candidate.resolve().relative_to(project_root.resolve()).as_posix()
        except Exception:
            return str(candidate.resolve())

    @classmethod
    def _default_resource_manifest_map(cls) -> dict[str, str]:
        out: dict[str, str] = {}
        for directory in (_RESOURCE_DIR, _EXPERIMENT_RESOURCES_DIR):
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.json")):
                try:
                    payload = cls._load_json(path)
                    resource_key, _ = cls._first_named_entry(payload)
                except Exception:
                    continue
                token = str(resource_key or "").strip()
                if token:
                    out[token] = cls._repo_ref(path)
        return out

    def _list_requirement_files(self) -> list[str]:
        if not _PRODUCT_REQUIREMENTS_DIR.exists():
            return []
        return [
            self._path_ref(path)
            for path in sorted(_PRODUCT_REQUIREMENTS_DIR.glob("*.txt"))
            if path.is_file()
        ]

    def _path_ref(self, raw_value: str | Path) -> str:
        return self._repo_ref(self._resolve_repo_path(raw_value), project_root=self.project_root)

    def _product_catalog_assets(
        self,
    ) -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, list[str]]]:
        requirement_files = self._list_requirement_files()
        product_catalog: list[dict[str, Any]] = []
        requirement_files_by_product: dict[str, list[str]] = {}
        part_catalog_by_product: dict[str, list[str]] = {}

        if not _PRODUCT_DIR.exists():
            return product_catalog, requirement_files_by_product, part_catalog_by_product

        for path in sorted(_PRODUCT_DIR.glob("*.json")):
            try:
                payload = self._load_json(path)
                product_name, meta = self._first_named_entry(payload)
                product_ref = self._path_ref(path)
                default_req_raw = str(meta.get("product_specification_file", "") or "").strip()
                default_req_ref = self._path_ref(default_req_raw) if default_req_raw else ""
                geometry_raw = str(meta.get("product_geometry_file", "") or "").strip()
                geometry_ref = self._path_ref(geometry_raw) if geometry_raw else ""
                part_catalog = sorted(self._load_product_parts(path))
            except Exception:
                continue

            ordered_requirements = list(requirement_files)
            if default_req_ref and default_req_ref in ordered_requirements:
                ordered_requirements = [default_req_ref] + [
                    item for item in ordered_requirements if item != default_req_ref
                ]

            product_catalog.append(
                {
                    "path": product_ref,
                    "name": str(product_name or Path(path).stem),
                    "default_requirement_file": default_req_ref,
                    "geometry_file": geometry_ref,
                    "part_count": len(part_catalog),
                }
            )
            requirement_files_by_product[product_ref] = ordered_requirements
            part_catalog_by_product[product_ref] = part_catalog

        return product_catalog, requirement_files_by_product, part_catalog_by_product

    def _load_verified_safety_assets(
        self,
    ) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
        approvals_payload = self._load_json(_SAFETY_INTENT_APPROVALS) if _SAFETY_INTENT_APPROVALS.exists() else {}
        previews_payload = self._load_json(_SAFETY_INTENT_PREVIEWS) if _SAFETY_INTENT_PREVIEWS.exists() else {}
        approvals = approvals_payload.get("approvals", {}) if isinstance(approvals_payload, dict) else {}
        previews = previews_payload.get("previews", {}) if isinstance(previews_payload, dict) else {}
        approvals = approvals if isinstance(approvals, dict) else {}
        previews = previews if isinstance(previews, dict) else {}

        catalog: list[dict[str, Any]] = []
        rules_by_file: dict[str, list[dict[str, Any]]] = {}
        preview_payloads_by_file: dict[str, dict[str, Any]] = {}

        if not _SAFETY_REQUIREMENTS_DIR.exists():
            return catalog, rules_by_file, preview_payloads_by_file

        for safety_path in sorted(_SAFETY_REQUIREMENTS_DIR.glob("*.txt")):
            try:
                safety_text = safety_path.read_text(encoding="utf-8").strip()
            except Exception:
                continue
            if not safety_text:
                continue

            safety_abs = self._norm_abs_path(safety_path)
            approval = approvals.get(safety_abs, {}) if isinstance(approvals.get(safety_abs), dict) else {}
            if not approval or not bool(approval.get("approved", False)):
                continue
            if str(approval.get("safety_sha256", "")).strip() != sha256_text(safety_text):
                continue

            preview_id = str(approval.get("preview_id", "") or "").strip()
            history = previews.get(safety_abs, [])
            if not isinstance(history, list) or not history:
                continue
            preview_record = {}
            for item in history:
                if not isinstance(item, dict):
                    continue
                if preview_id and str(item.get("preview_id", "")).strip() == preview_id:
                    preview_record = dict(item)
                    break
            if not preview_record:
                first_item = history[0]
                preview_record = dict(first_item) if isinstance(first_item, dict) else {}
            if not preview_record:
                continue

            logic_path = Path(str(preview_record.get("safety_logic_json", "") or "").strip())
            if not logic_path.exists():
                continue
            try:
                logic_payload = self._load_json(logic_path)
            except Exception:
                continue
            raw_rules = logic_payload.get("rules", []) if isinstance(logic_payload, dict) else []
            rules = [dict(rule) for rule in raw_rules if isinstance(rule, dict)]
            safety_ref = self._path_ref(safety_path)
            verified_file_raw = str(approval.get("verified_file", "") or "").strip()
            catalog.append(
                {
                    "path": safety_ref,
                    "name": safety_path.stem,
                    "approved_at_utc": str(approval.get("approved_at_utc", "") or ""),
                    "preview_id": str(preview_record.get("preview_id", "") or ""),
                    "rule_count": len(rules),
                    "verified_file": self._path_ref(verified_file_raw) if verified_file_raw else "",
                }
            )
            rules_by_file[safety_ref] = rules
            preview_payloads_by_file[safety_ref] = {
                "approval_record": approval,
                "preview_record": preview_record,
                "logic_path": str(logic_path.resolve()),
            }

        return catalog, rules_by_file, preview_payloads_by_file

    def _load_product_parts(self, product_init_path: Path) -> set[str]:
        product_raw = self._load_json(product_init_path)
        _, product_meta = self._first_named_entry(product_raw)
        geometry_ref = str(product_meta.get("product_geometry_file", "")).strip()
        if not geometry_ref:
            raise ValueError(f"product init file {product_init_path} is missing product_geometry_file")
        geometry_path = self._resolve_repo_path(geometry_ref)
        geometry_payload = self._load_json(geometry_path)
        gazebo = geometry_payload.get("gazebo", {})
        parts = (((gazebo if isinstance(gazebo, dict) else {}).get("parts") or {}).get("model_map") or {})
        if not isinstance(parts, dict):
            raise ValueError(f"product geometry missing parts.model_map: {geometry_path}")
        return {str(name).strip() for name in parts.keys() if str(name).strip()}

    def _load_resource_definitions(
        self,
        resource_manifest_map: dict[str, Any],
        *,
        robot_env: str,
    ) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for key, raw_path in resource_manifest_map.items():
            resource_key = str(key or "").strip()
            if not resource_key:
                continue
            manifest_path = self._resolve_repo_path(str(raw_path or "").strip())
            payload = self._load_json(manifest_path)
            resource_name, meta = self._first_named_entry(payload)
            env_block = meta.get(robot_env, {}) if isinstance(meta.get(robot_env), dict) else {}
            static_caps = env_block.get("static_capabilities", meta.get("static_capabilities", {}))
            if not isinstance(static_caps, dict):
                static_caps = {}
            reachability = self._as_str_list(static_caps.get("reachability", []))
            out[resource_key] = {
                "key": resource_key,
                "name": resource_name,
                "path": manifest_path,
                "jid": str(meta.get("jid") or f"{resource_name}@{meta.get('domain', 'localhost')}").strip(),
                "kind": str(meta.get("type") or "").strip().lower(),
                "reachability": reachability,
                "static_capabilities": static_caps,
            }
        return out

    @staticmethod
    def normalize_method_name(method: str) -> str:
        token = str(method or "").strip().lower()
        if not token:
            return ""
        return _METHOD_ALIAS_MAP.get(token, token)

    @classmethod
    def normalize_methods(cls, raw_methods: Any) -> list[str]:
        methods: list[str] = []
        seen: set[str] = set()
        for raw_method in cls._as_str_list(raw_methods or _DEFAULT_METHODS):
            method = cls.normalize_method_name(raw_method)
            if not method or method in seen:
                continue
            seen.add(method)
            methods.append(method)
        return methods or list(_DEFAULT_METHODS)

    @staticmethod
    def method_label(method: str) -> str:
        labels = {
            "llm_nl_safety": "Pure LLM",
            "verified": "LLM + Formal Verification",
        }
        return labels.get(str(method or "").strip(), str(method or "").strip())

    def _resource_catalog(
        self,
        resource_defs: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        preferred_key_order = ["xarm6", "ur5e", "xarm6-2", "ur5e-2"]
        order_index = {key: idx for idx, key in enumerate(preferred_key_order)}
        locations_in_catalog = {
            location
            for resource in resource_defs.values()
            for location in self._as_str_list(resource.get("reachability", []))
        }
        ordered_locations = [
            location for location in _SOURCE_LOCATION_ORDER if location in locations_in_catalog
        ]
        extras = sorted(locations_in_catalog - set(ordered_locations))
        all_locations = ordered_locations + extras

        rows: list[dict[str, Any]] = []
        for key, resource in sorted(
            resource_defs.items(),
            key=lambda item: (
                order_index.get(str(item[0]), len(order_index)),
                str(item[0]),
            ),
        ):
            reachability = self._as_str_list(resource.get("reachability", []))
            rows.append(
                {
                    "key": str(key),
                    "name": str(resource.get("name") or key),
                    "jid": str(resource.get("jid") or ""),
                    "kind": str(resource.get("kind") or ""),
                    "reachability": reachability,
                    "source_locations": [loc for loc in all_locations if loc in set(reachability)],
                }
            )
        return rows

    @staticmethod
    def _ordered_source_locations(resource_defs: dict[str, dict[str, Any]]) -> list[str]:
        locations = {
            location
            for resource in resource_defs.values()
            for location in OfflineStudyRunner._as_str_list(resource.get("reachability", []))
        }
        ordered = [loc for loc in _SOURCE_LOCATION_ORDER if loc in locations]
        extras = sorted(locations - set(ordered))
        return ordered + extras

    @staticmethod
    def _scenario_resource_keys(
        raw_scenario: dict[str, Any],
        resource_sets: dict[str, list[str]],
    ) -> list[str]:
        explicit = OfflineStudyRunner._as_str_list(
            raw_scenario.get("resource_keys", raw_scenario.get("selected_robots", []))
        )
        if explicit:
            return explicit
        resource_set_name = str(raw_scenario.get("resource_set", "")).strip()
        return list(resource_sets.get(resource_set_name, []))

    @staticmethod
    def _resource_set_token(resource_keys: list[str]) -> tuple[str, ...]:
        return tuple(str(key).strip() for key in resource_keys if str(key).strip())

    @classmethod
    def canonicalize_manifest(cls, payload: dict[str, Any]) -> dict[str, Any]:
        manifest = cls._json_safe(dict(payload or {}))
        defaults = dict(manifest.get("defaults", {}) or {})
        defaults["execution_mode"] = str(defaults.get("execution_mode", "dry_run")).strip().lower() or "dry_run"
        defaults["robot_env"] = str(defaults.get("robot_env", "gazebo")).strip().lower() or "gazebo"
        defaults["trials_per_method"] = max(1, int(defaults.get("trials_per_method", 10) or 10))
        defaults["auto_replan_max_attempts"] = max(
            0,
            min(
                int(
                    defaults.get(
                        "auto_replan_max_attempts",
                        _DEFAULT_AUTO_REPLAN_MAX_ATTEMPTS,
                    )
                    or _DEFAULT_AUTO_REPLAN_MAX_ATTEMPTS
                ),
                10,
            ),
        )
        defaults["methods"] = cls.normalize_methods(defaults.get("methods", _DEFAULT_METHODS))
        manifest["defaults"] = defaults

        resource_manifests = cls._default_resource_manifest_map()
        resource_manifests.update(
            {
                str(key).strip(): cls._repo_ref(value)
                for key, value in dict(manifest.get("resource_manifests", {}) or {}).items()
                if str(key).strip() and str(value or "").strip()
            }
        )
        manifest["resource_manifests"] = resource_manifests

        raw_resource_sets = dict(manifest.get("resource_sets", {}) or {})
        tuple_to_name: dict[tuple[str, ...], str] = {}
        used_names: set[str] = set()
        for name, values in raw_resource_sets.items():
            token = cls._resource_set_token(cls._as_str_list(values))
            set_name = str(name or "").strip()
            if not token or not set_name:
                continue
            tuple_to_name[token] = set_name
            used_names.add(set_name)

        def _next_set_name(resource_count: int) -> str:
            prefix = f"r{int(resource_count)}"
            if prefix not in used_names:
                used_names.add(prefix)
                return prefix
            suffix = 2
            while f"{prefix}_{suffix}" in used_names:
                suffix += 1
            candidate = f"{prefix}_{suffix}"
            used_names.add(candidate)
            return candidate

        scenarios_out: list[dict[str, Any]] = []
        canonical_resource_sets: dict[str, list[str]] = {}
        for raw_scenario in manifest.get("scenarios", []) if isinstance(manifest.get("scenarios", []), list) else []:
            if not isinstance(raw_scenario, dict):
                continue
            scenario = dict(raw_scenario)
            resource_keys = cls._scenario_resource_keys(scenario, raw_resource_sets)
            token = cls._resource_set_token(resource_keys)
            set_name = tuple_to_name.get(token)
            if token and not set_name:
                set_name = _next_set_name(len(token))
                tuple_to_name[token] = set_name
            if set_name and token:
                canonical_resource_sets[set_name] = list(token)
                scenario["resource_set"] = set_name
            scenario.pop("selected_robots", None)
            scenario["trials"] = int(defaults["trials_per_method"])
            scenario["resource_keys"] = list(token)
            scenario["product_init_file"] = cls._repo_ref(
                scenario.get("product_init_file", _DEFAULT_PRODUCT_INIT)
            )
            scenario["product_requirement_file"] = cls._repo_ref(
                scenario.get("product_requirement_file", "")
            )
            scenario["safety_requirement_file"] = cls._repo_ref(
                scenario.get("safety_requirement_file", "")
            )
            scenario["enabled_safety_rule_ids"] = cls._as_str_list(
                scenario.get("enabled_safety_rule_ids", [])
            )
            scenario["parts"] = cls._as_str_list(scenario.get("parts", []))
            part_order = cls._as_str_list(scenario.get("part_order", scenario.get("parts", [])))
            scenario["part_order"] = [
                part for part in part_order if part in set(scenario["parts"])
            ] or list(scenario["parts"])
            scenario["part_sources"] = {
                str(key).strip(): str(value).strip()
                for key, value in dict(scenario.get("part_sources", {})).items()
                if str(key).strip()
            }
            scenario["notes"] = str(scenario.get("notes", "") or "")
            scenario.pop("constraints", None)
            scenarios_out.append(scenario)

        manifest["resource_sets"] = canonical_resource_sets or {
            name: cls._as_str_list(values) for name, values in raw_resource_sets.items()
        }
        manifest["scenarios"] = scenarios_out
        return manifest

    def load_editor_context(self) -> dict[str, Any]:
        manifest = self.canonicalize_manifest(self._load_json(self.manifest_path))
        product_catalog, requirement_files_by_product, part_catalog_by_product = self._product_catalog_assets()
        verified_safety_catalog, verified_rules_by_safety_file, _ = self._load_verified_safety_assets()
        robot_env = str(dict(manifest.get("defaults", {}) or {}).get("robot_env", "gazebo") or "gazebo")
        resource_defs = self._load_resource_definitions(
            dict(manifest.get("resource_manifests", {}) or {}),
            robot_env=robot_env,
        )
        editor_manifest = self._json_safe(dict(manifest))
        default_product_init = product_catalog[0]["path"] if product_catalog else self._path_ref(_DEFAULT_PRODUCT_INIT)
        default_product_requirements = (
            requirement_files_by_product.get(default_product_init, [])
            if default_product_init
            else []
        )
        default_safety = verified_safety_catalog[0]["path"] if verified_safety_catalog else ""
        default_rule_ids = [
            str(rule.get("id") or "")
            for rule in verified_rules_by_safety_file.get(default_safety, [])
            if str(rule.get("id") or "").strip()
        ]
        editor_scenarios: list[dict[str, Any]] = []
        for raw_scenario in manifest.get("scenarios", []) if isinstance(manifest.get("scenarios", []), list) else []:
            if not isinstance(raw_scenario, dict):
                continue
            scenario = dict(raw_scenario)
            product_init_file = str(scenario.get("product_init_file", "") or "").strip() or default_product_init
            if product_init_file not in part_catalog_by_product and part_catalog_by_product:
                product_init_file = default_product_init
            scenario["product_init_file"] = product_init_file
            requirement_options = requirement_files_by_product.get(product_init_file, [])
            scenario["product_requirement_file"] = (
                str(scenario.get("product_requirement_file", "") or "").strip()
                or (requirement_options[0] if requirement_options else "")
            )
            safety_requirement_file = str(scenario.get("safety_requirement_file", "") or "").strip() or default_safety
            if safety_requirement_file not in verified_rules_by_safety_file and verified_rules_by_safety_file:
                safety_requirement_file = default_safety
            scenario["safety_requirement_file"] = safety_requirement_file
            available_rule_ids = [
                str(rule.get("id") or "")
                for rule in verified_rules_by_safety_file.get(safety_requirement_file, [])
                if str(rule.get("id") or "").strip()
            ]
            scenario["enabled_safety_rule_ids"] = [
                rid for rid in self._as_str_list(scenario.get("enabled_safety_rule_ids", []))
                if rid in set(available_rule_ids)
            ] or list(available_rule_ids if safety_requirement_file else default_rule_ids)
            available_parts = set(part_catalog_by_product.get(product_init_file, []))
            derived_requirement_layout = self.parse_requirement_file_layout(
                scenario["product_requirement_file"],
                project_root=self.project_root,
                valid_parts=available_parts,
            )
            if bool(derived_requirement_layout.get("derived", False)):
                scenario["parts"] = list(derived_requirement_layout.get("parts", []))
                scenario["part_order"] = list(derived_requirement_layout.get("part_order", []))
                scenario["part_sources"] = dict(derived_requirement_layout.get("part_sources", {}))
            else:
                scenario["parts"] = [
                    part for part in self._as_str_list(scenario.get("parts", []))
                    if part in available_parts
                ]
                scenario["part_order"] = [
                    part
                    for part in self._as_str_list(scenario.get("part_order", scenario.get("parts", [])))
                    if part in set(scenario["parts"])
                ] or list(scenario["parts"])
                scenario["part_sources"] = {
                    part: source
                    for part, source in dict(scenario.get("part_sources", {}) or {}).items()
                    if str(part).strip() in set(scenario["parts"])
                }
            editor_scenarios.append(scenario)
        editor_manifest["scenarios"] = editor_scenarios
        return {
            "manifest": editor_manifest,
            "normalized_study": None,
            "robot_catalog": self._resource_catalog(resource_defs),
            "product_catalog": product_catalog,
            "requirement_files_by_product": requirement_files_by_product,
            "verified_safety_catalog": verified_safety_catalog,
            "verified_rules_by_safety_file": verified_rules_by_safety_file,
            "part_catalog_by_product": part_catalog_by_product,
            "source_locations": self._ordered_source_locations(resource_defs),
        }

    def load_study_definition(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"study manifest missing: {self.manifest_path}")

        payload = self.canonicalize_manifest(self._load_json(self.manifest_path))
        study_id = str(payload.get("study_id", "")).strip()
        if not study_id:
            raise ValueError("study manifest missing study_id")

        defaults = payload.get("defaults", {})
        if not isinstance(defaults, dict):
            raise ValueError("study manifest defaults must be an object")

        execution_mode = str(defaults.get("execution_mode", "dry_run")).strip().lower() or "dry_run"
        robot_env = str(defaults.get("robot_env", "gazebo")).strip().lower() or "gazebo"
        trials_per_method = max(1, int(defaults.get("trials_per_method", 10) or 10))
        auto_replan_max_attempts = max(
            0,
            min(
                int(
                    defaults.get(
                        "auto_replan_max_attempts",
                        _DEFAULT_AUTO_REPLAN_MAX_ATTEMPTS,
                    )
                    or _DEFAULT_AUTO_REPLAN_MAX_ATTEMPTS
                ),
                10,
            ),
        )
        methods = self.normalize_methods(defaults.get("methods", _DEFAULT_METHODS))
        if not methods:
            raise ValueError("study manifest defaults.methods must include at least one method")
        unknown_methods = sorted(set(methods) - {"llm_nl_safety", "verified"})
        if unknown_methods:
            raise ValueError(f"unsupported study methods: {', '.join(unknown_methods)}")

        resource_manifest_map = payload.get("resource_manifests", {})
        if not isinstance(resource_manifest_map, dict) or not resource_manifest_map:
            raise ValueError("study manifest missing resource_manifests")
        resource_defs = self._load_resource_definitions(resource_manifest_map, robot_env=robot_env)

        resource_sets_raw = payload.get("resource_sets", {})
        if not isinstance(resource_sets_raw, dict) or not resource_sets_raw:
            raise ValueError("study manifest missing resource_sets")
        resource_sets: dict[str, list[str]] = {}
        for set_name, raw_items in resource_sets_raw.items():
            items = self._as_str_list(raw_items)
            if not items:
                raise ValueError(f"resource_set {set_name} is empty")
            missing = [name for name in items if name not in resource_defs]
            if missing:
                raise ValueError(
                    f"resource_set {set_name} references unknown resources: {', '.join(missing)}"
                )
            resource_sets[str(set_name)] = items

        product_catalog, requirement_files_by_product, part_catalog_by_product = self._product_catalog_assets()
        product_catalog_by_path = {
            str(item.get("path") or ""): dict(item)
            for item in product_catalog
            if isinstance(item, dict) and str(item.get("path") or "").strip()
        }
        verified_safety_catalog, verified_rules_by_safety_file, preview_payloads_by_safety_file = (
            self._load_verified_safety_assets()
        )
        verified_safety_paths = {
            str(item.get("path") or "")
            for item in verified_safety_catalog
            if isinstance(item, dict) and str(item.get("path") or "").strip()
        }

        scenarios_raw = payload.get("scenarios", [])
        if not isinstance(scenarios_raw, list) or not scenarios_raw:
            raise ValueError("study manifest missing scenarios")

        scenarios: list[dict[str, Any]] = []
        for raw_scenario in scenarios_raw:
            if not isinstance(raw_scenario, dict):
                raise ValueError("scenario entries must be objects")
            scenario_id = str(raw_scenario.get("id", "")).strip()
            if not scenario_id:
                raise ValueError("scenario missing id")
            resource_keys_from_scenario = self._scenario_resource_keys(raw_scenario, resource_sets)
            resource_set_name = str(raw_scenario.get("resource_set", "")).strip()
            if not resource_set_name and resource_keys_from_scenario:
                resource_set_name = "__inline__"
                resource_sets[resource_set_name] = resource_keys_from_scenario
            if resource_set_name not in resource_sets:
                raise ValueError(
                    f"scenario {scenario_id} references unknown resource_set {resource_set_name}"
                )
            active_resource_keys = list(resource_keys_from_scenario or resource_sets[resource_set_name])
            if not active_resource_keys:
                raise ValueError(f"scenario {scenario_id} has no active resources")
            active_resources = [resource_defs[key] for key in active_resource_keys]
            active_resource_names = [str(item.get("name") or "").strip() for item in active_resources]

            product_init_file = str(raw_scenario.get("product_init_file", "") or "").strip()
            if not product_init_file:
                raise ValueError(f"scenario {scenario_id} is missing product_init_file")
            product_init_file = self._path_ref(product_init_file)
            if product_init_file not in product_catalog_by_path:
                raise ValueError(
                    f"scenario {scenario_id} references unknown product_init_file {product_init_file}"
                )
            product_meta = product_catalog_by_path[product_init_file]
            valid_parts = set(part_catalog_by_product.get(product_init_file, []))

            product_requirement_file = str(raw_scenario.get("product_requirement_file", "") or "").strip()
            if not product_requirement_file:
                product_requirement_file = str(product_meta.get("default_requirement_file", "") or "")
            if not product_requirement_file:
                raise ValueError(
                    f"scenario {scenario_id} is missing product_requirement_file and the selected product has no default"
                )
            product_requirement_file = self._path_ref(product_requirement_file)
            if not self._resolve_repo_path(product_requirement_file).exists():
                raise ValueError(
                    f"scenario {scenario_id} references missing product_requirement_file {product_requirement_file}"
                )

            safety_requirement_file = str(raw_scenario.get("safety_requirement_file", "") or "").strip()
            if not safety_requirement_file:
                raise ValueError(f"scenario {scenario_id} is missing safety_requirement_file")
            safety_requirement_file = self._path_ref(safety_requirement_file)
            if safety_requirement_file not in verified_safety_paths:
                raise ValueError(
                    f"scenario {scenario_id} references a safety file that is not currently verified: "
                    f"{safety_requirement_file}"
                )
            preview_payload = dict(preview_payloads_by_safety_file.get(safety_requirement_file, {}) or {})
            available_rules = [
                dict(rule) for rule in verified_rules_by_safety_file.get(safety_requirement_file, [])
                if isinstance(rule, dict)
            ]
            available_rule_ids = [
                str(rule.get("id") or "") for rule in available_rules if str(rule.get("id") or "").strip()
            ]
            enabled_safety_rule_ids = self._as_str_list(
                raw_scenario.get("enabled_safety_rule_ids", available_rule_ids)
            )
            missing_rule_ids = [
                rid for rid in enabled_safety_rule_ids if rid not in set(available_rule_ids)
            ]
            if missing_rule_ids:
                raise ValueError(
                    f"scenario {scenario_id} references unknown verified safety rule ids: "
                    + ", ".join(missing_rule_ids)
                )
            available_rules_by_id = {
                str(rule.get("id") or ""): dict(rule)
                for rule in available_rules
                if str(rule.get("id") or "").strip()
            }
            selected_safety_rules = [
                dict(available_rules_by_id[rid])
                for rid in enabled_safety_rule_ids
                if rid in available_rules_by_id
            ]

            derived_requirement_layout = self.parse_requirement_file_layout(
                product_requirement_file,
                project_root=self.project_root,
                valid_parts=valid_parts,
            )
            if bool(derived_requirement_layout.get("derived", False)):
                parts = list(derived_requirement_layout.get("parts", []))
                ordered_parts = list(derived_requirement_layout.get("part_order", []))
                part_sources = dict(derived_requirement_layout.get("part_sources", {}))
            else:
                parts = self._as_str_list(raw_scenario.get("parts", []))
                if not parts:
                    raise ValueError(f"scenario {scenario_id} has no parts")
                unknown_parts = sorted(set(parts) - valid_parts)
                if unknown_parts:
                    raise ValueError(
                        f"scenario {scenario_id} references unknown parts: {', '.join(unknown_parts)}"
                    )
                part_order = self._as_str_list(raw_scenario.get("part_order", parts))
                ordered_parts = [part for part in part_order if part in set(parts)]
                for part in parts:
                    if part not in ordered_parts:
                        ordered_parts.append(part)

                part_sources_raw = raw_scenario.get("part_sources", {})
                if not isinstance(part_sources_raw, dict):
                    raise ValueError(f"scenario {scenario_id} part_sources must be an object")
                missing_part_sources = [part for part in parts if part not in part_sources_raw]
                if missing_part_sources:
                    raise ValueError(
                        f"scenario {scenario_id} is missing part_sources for: "
                        + ", ".join(missing_part_sources)
                    )
                part_sources = {
                    str(part).strip(): str(part_sources_raw.get(part) or "").strip()
                    for part in parts
                }

            reachable_locations = {
                location
                for resource in active_resources
                for location in self._as_str_list(resource.get("reachability", []))
            }
            invalid_sources = sorted(
                {
                    source
                    for source in part_sources.values()
                    if source and source not in reachable_locations
                }
            )
            if invalid_sources:
                raise ValueError(
                    f"scenario {scenario_id} uses unreachable source locations: {', '.join(invalid_sources)}"
                )

            scenarios.append(
                {
                    "id": scenario_id,
                    "resource_set": resource_set_name,
                    "resource_keys": active_resource_keys,
                    "resource_names": active_resource_names,
                    "resource_count": len(active_resources),
                    "product_init_file": product_init_file,
                    "product_requirement_file": product_requirement_file,
                    "safety_requirement_file": safety_requirement_file,
                    "enabled_safety_rule_ids": enabled_safety_rule_ids,
                    "selected_safety_rules": selected_safety_rules,
                    "safety_preview": preview_payload,
                    "parts": parts,
                    "part_order": ordered_parts,
                    "part_sources": part_sources,
                    "part_count": len(parts),
                    "safety_rule_count": len(selected_safety_rules),
                    "trials": int(trials_per_method),
                    "notes": str(raw_scenario.get("notes", "") or ""),
                }
            )

        return {
            "study_id": study_id,
            "defaults": {
                "execution_mode": execution_mode,
                "robot_env": robot_env,
                "trials_per_method": trials_per_method,
                "auto_replan_max_attempts": auto_replan_max_attempts,
                "methods": methods,
            },
            "resource_manifests": {
                key: str(value.get("path"))
                for key, value in resource_defs.items()
            },
            "resource_definitions": resource_defs,
            "resource_sets": resource_sets,
            "product_catalog": product_catalog,
            "requirement_files_by_product": requirement_files_by_product,
            "verified_safety_catalog": verified_safety_catalog,
            "verified_rules_by_safety_file": verified_rules_by_safety_file,
            "part_catalog_by_product": part_catalog_by_product,
            "scenarios": scenarios,
        }

    def render_requirements_text(self, scenario: dict[str, Any]) -> str:
        requirement_file = str(scenario.get("product_requirement_file", "") or "").strip()
        if requirement_file:
            path = self._resolve_repo_path(requirement_file)
            if path.exists() and path.is_file():
                try:
                    text = self._read_text_file(path)
                except Exception:
                    text = ""
                if text:
                    return text + "\n"
        lines = ["[Product Requirements]"]
        ordered_parts = self._as_str_list(scenario.get("part_order", scenario.get("parts", [])))
        for part in ordered_parts:
            source = str((scenario.get("part_sources") or {}).get(part) or "").strip()
            lines.append(f"- assemble {part} from {source} to the Assembly Station.")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _dedupe_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
        seen: set[tuple[str, str]] = set()
        out: list[tuple[str, str]] = []
        for before, after in pairs:
            key = (str(before).strip(), str(after).strip())
            if not key[0] or not key[1] or key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out

    def render_safety_text(
        self,
        scenario: dict[str, Any],
        *,
        verified_rules_by_safety_file: dict[str, list[dict[str, Any]]] | None = None,
        resource_defs: dict[str, dict[str, Any]] | None = None,
    ) -> str:
        lines = ["[Safety Requirements]"]
        selected_rules = [
            dict(rule)
            for rule in (scenario.get("selected_safety_rules", []) if isinstance(scenario.get("selected_safety_rules", []), list) else [])
            if isinstance(rule, dict)
        ]
        if not selected_rules and verified_rules_by_safety_file is not None:
            safety_file = str(scenario.get("safety_requirement_file", "") or "").strip()
            enabled_ids = self._as_str_list(scenario.get("enabled_safety_rule_ids", []))
            available_rules = [
                dict(rule)
                for rule in verified_rules_by_safety_file.get(safety_file, [])
                if isinstance(rule, dict)
            ]
            lookup = {
                str(rule.get("id") or ""): dict(rule)
                for rule in available_rules
                if str(rule.get("id") or "").strip()
            }
            selected_rules = [lookup[rid] for rid in enabled_ids if rid in lookup]

        if selected_rules:
            for rule in selected_rules:
                raw_text = str(rule.get("raw_text", "") or "").strip()
                if raw_text:
                    lines.append(f"- {raw_text}")
            return "\n".join(lines) + "\n"

        constraints = dict(scenario.get("constraints", {}) or {})
        if not constraints or resource_defs is None:
            return "\n".join(lines) + "\n"

        resource_keys = list(scenario.get("resource_keys", []))
        active_resources = [resource_defs[key] for key in resource_keys if key in resource_defs]
        active_resource_names = [str(item.get("name") or "").strip() for item in active_resources]
        ordering_pairs = [
            (str(before).strip(), str(after).strip())
            for before, after in (
                [tuple(pair) for pair in constraints.get("ordering_pairs", [])]
                if isinstance(constraints.get("ordering_pairs", []), list)
                else []
            )
        ]
        if constraints.get("ordering_chain", False):
            parts = self._as_str_list(scenario.get("parts", []))
            ordering_pairs.extend(zip(parts, parts[1:]))
        for before, after in self._dedupe_pairs(ordering_pairs):
            lines.append(
                f"- {before} must be placed into the Assembly Station before {after} is placed into the Assembly Station."
            )
        if constraints.get("shared_board_mutex", False):
            for left, right in combinations(active_resource_names, 2):
                lines.append(
                    f"- {left} and {right} must not both be in the assembly board destination area at the same time."
                )
        if constraints.get("move_home_response", False):
            for resource_name in active_resource_names:
                lines.append(f"- {resource_name} should move back home after placing its part is done.")
        source_locations = self._as_str_list(constraints.get("source_area_mutex_locations", []))
        for location in source_locations:
            reachable_names = [
                str(resource.get("name") or "").strip()
                for resource in active_resources
                if location in self._as_str_list(resource.get("reachability", []))
            ]
            for left, right in combinations(reachable_names, 2):
                lines.append(
                    f"- {left} and {right} must not both be in the {location} source area at the same time."
                )

        return "\n".join(lines) + "\n"

    def _copy_file(self, src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)

    @staticmethod
    def _safety_interpretation_summary(rules: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for rule in rules:
            rid = str(rule.get("id", "") or "").strip()
            explanation = str(
                rule.get("generated_interpretation", "")
                or rule.get("ltlf_plain_feedback", "")
                or ""
            ).strip()
            if rid and explanation:
                lines.append(f"- {rid}: {explanation}")
        return "\n".join(lines)

    @staticmethod
    def _relabeled_ltlf_from_full_aps(
        full_ltlf: str,
        labeled_aps: list[dict[str, Any]],
    ) -> str:
        formula = str(full_ltlf or "").strip()
        replacements = [
            (str(ap.get("full", "")).strip(), str(ap.get("label", "")).strip())
            for ap in labeled_aps
            if isinstance(ap, dict)
            and str(ap.get("full", "")).strip()
            and str(ap.get("label", "")).strip()
        ]
        for full_ap, label in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
            formula = formula.replace(full_ap, label)
        return formula

    @classmethod
    def _normalize_selected_safety_rules(
        cls,
        rules: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        from cais_spade_llm.agents.central_controller.safety_logic import SafetyLogic

        normalized_rules: list[dict[str, Any]] = []
        for raw_rule in rules:
            rule = dict(raw_rule)
            labeled_aps = [
                dict(ap)
                for ap in (rule.get("aps", []) if isinstance(rule.get("aps"), list) else [])
                if isinstance(ap, dict)
            ]
            full_aps = [
                str(ap.get("full", "")).strip()
                for ap in labeled_aps
                if str(ap.get("full", "")).strip()
            ]
            family = SafetyLogic._infer_ltlf_family(rule, full_aps)
            deterministic_full_ltlf = SafetyLogic._compile_ltlf_for_rule(rule, full_aps)
            if family == "precedence" and deterministic_full_ltlf:
                rule["ltlf"] = cls._relabeled_ltlf_from_full_aps(
                    deterministic_full_ltlf,
                    labeled_aps,
                )
                rule["generated_interpretation"] = SafetyLogic._fallback_rule_interpretation(rule)
            normalized_rules.append(rule)
        return normalized_rules

    @staticmethod
    def _label_precedence_pair(ltlf: str) -> tuple[str, str] | None:
        normalized = " ".join(str(ltlf or "").strip().split())
        match = re.fullmatch(r"\(\(!?(ap\d+)\) U (ap\d+)\)", normalized)
        if not match:
            match = re.fullmatch(r"\(!?(ap\d+)\) U (ap\d+)", normalized)
        if not match:
            return None
        later_label = str(match.group(1)).strip()
        earlier_label = str(match.group(2)).strip()
        if not later_label or not earlier_label:
            return None
        return earlier_label, later_label

    @staticmethod
    def _precedence_dfa_dot(earlier_label: str, later_label: str) -> str:
        return "\n".join(
            [
                "digraph MONA_DFA {",
                " rankdir = LR;",
                " center = true;",
                ' size = "7.5,10.5";',
                " edge [fontname = Courier];",
                " node [height = .5, width = .5];",
                " node [shape = doublecircle]; 2;",
                " node [shape = circle]; 1;",
                " node [shape = circle]; 2;",
                " node [shape = circle]; 3;",
                ' init [shape = plaintext, label = ""];',
                " init -> 1;",
                f' 1 -> 2 [label="{earlier_label}"];',
                f' 1 -> 1 [label="~{earlier_label} & ~{later_label}"];',
                f' 1 -> 3 [label="~{earlier_label} & {later_label}"];',
                ' 2 -> 2 [label="true"];',
                ' 3 -> 3 [label="true"];',
                "}",
                "",
            ]
        )

    def _build_filtered_precomputed_safety_artifacts(
        self,
        *,
        scenario: dict[str, Any],
        scenario_dir: Path,
        safety_file: Path,
        tools_path: Path,
    ) -> dict[str, Any]:
        preview_payload = dict(scenario.get("safety_preview", {}) or {})
        preview_record = dict(preview_payload.get("preview_record", {}) or {})
        original_selected_rules = [
            dict(rule)
            for rule in scenario.get("selected_safety_rules", [])
            if isinstance(rule, dict)
        ]
        selected_rules = self._normalize_selected_safety_rules(original_selected_rules)

        subset_dir = scenario_dir / "approved_safety_subset"
        subset_dir.mkdir(parents=True, exist_ok=True)
        logic_payload = {
            "preview_interpretation_summary": self._safety_interpretation_summary(selected_rules),
            "rules": selected_rules,
        }
        logic_path = subset_dir / "cca_safety_logic.json"
        atomic_json_write(logic_path, logic_payload)

        dot_files: list[str] = []
        png_files: list[str] = []
        for key, target_list in (("dfa_dot_files", dot_files), ("dfa_png_files", png_files)):
            for raw_path in (
                preview_record.get(key, [])
                if isinstance(preview_record.get(key, []), list)
                else []
            ):
                src = Path(str(raw_path or "").strip())
                if not src.exists():
                    continue
                dst = subset_dir / src.name
                shutil.copyfile(src, dst)
                target_list.append(str(dst.resolve()))

        original_by_id = {
            str(rule.get("id") or "").strip(): dict(rule)
            for rule in original_selected_rules
            if str(rule.get("id") or "").strip()
        }
        for rule in selected_rules:
            rid = str(rule.get("id") or "").strip()
            if not rid:
                continue
            original_rule = original_by_id.get(rid, {})
            if str(original_rule.get("ltlf", "") or "").strip() == str(rule.get("ltlf", "") or "").strip():
                continue
            precedence_pair = self._label_precedence_pair(str(rule.get("ltlf", "") or ""))
            if not precedence_pair:
                continue
            earlier_label, later_label = precedence_pair
            dot_path = subset_dir / f"{rid}_dfa.dot"
            dot_path.write_text(
                self._precedence_dfa_dot(earlier_label, later_label),
                encoding="utf-8",
            )
            if str(dot_path.resolve()) not in dot_files:
                dot_files.append(str(dot_path.resolve()))
            png_path = subset_dir / f"{rid}_dfa.png"
            png_path.unlink(missing_ok=True)
            png_files = [path for path in png_files if Path(path).name != png_path.name]

        safety_text = safety_file.read_text(encoding="utf-8").strip()
        return {
            "mode": "approved_preview_subset",
            "preview_id": str(preview_record.get("preview_id", "") or "").strip(),
            "preview_generated_at_utc": str(
                preview_record.get("generated_at_utc", "") or ""
            ).strip(),
            "safety_logic_json": str(logic_path.resolve()),
            "dfa_dot_files": sorted(dot_files),
            "dfa_png_files": sorted(png_files),
            "safety_sha256": sha256_text(safety_text),
            "tools_sha256": sha256_file(tools_path),
            "prompts_sha256": sha256_file(_DEFAULT_PROMPTS),
        }

    def _build_tools_catalogue(
        self,
        *,
        resources_dir: Path,
        cca_init_file: Path,
        robot_env: str,
        execution_mode: str,
        tools_path: Path,
    ) -> None:
        from cais_spade_llm import agent_creator
        from cais_spade_llm.function_analyzer import FunctionAnalyzer

        agent_creator.configure_runtime(
            robot_env=robot_env,
            execution_mode=execution_mode,
            perception_backend="none",
        )
        agent_creator.ALLOWED_FUNCS.clear()
        resource_files = sorted(str(p.resolve()) for p in resources_dir.glob("*.json"))
        resource_agents = agent_creator.create_resource_agents(
            resource_files,
            str(cca_init_file.resolve()),
        )
        allowed = {
            key: set(value)
            for key, value in dict(agent_creator.ALLOWED_FUNCS).items()
        }
        FunctionAnalyzer.build_tools_catalogue(
            resource_agents,
            allowed=allowed,
            outfile=tools_path,
        )

    def prepare_scenario_workspace(
        self,
        study: dict[str, Any],
        scenario: dict[str, Any],
        *,
        run_root: Path,
    ) -> ScenarioWorkspace:
        scenario_dir = run_root / "workspaces" / str(scenario.get("id") or "").strip()
        resources_dir = scenario_dir / "init" / "resources"
        product_dir = scenario_dir / "init" / "products"
        catalog_dir = scenario_dir / "catalog"
        spec_dir = scenario_dir / "spec"
        requirements_file = spec_dir / "products" / "requirements" / f"{scenario['id']}_requirements.txt"
        safety_file = spec_dir / "safety" / f"{scenario['id']}_safety.txt"
        tools_path = catalog_dir / "tools.json"
        cca_init_file = scenario_dir / "init" / "cca.json"
        source_product_init_file = self._resolve_repo_path(str(scenario.get("product_init_file") or ""))
        product_init_file = product_dir / source_product_init_file.name

        resources_dir.mkdir(parents=True, exist_ok=True)
        product_dir.mkdir(parents=True, exist_ok=True)
        catalog_dir.mkdir(parents=True, exist_ok=True)
        (spec_dir / "products" / "requirements").mkdir(parents=True, exist_ok=True)
        (spec_dir / "safety").mkdir(parents=True, exist_ok=True)

        self._copy_file(source_product_init_file, product_init_file)
        self._copy_file(_DEFAULT_CCA_INIT, cca_init_file)

        resource_defs = study.get("resource_definitions", {})
        for resource_key in scenario.get("resource_keys", []):
            resource_def = resource_defs.get(resource_key)
            if not isinstance(resource_def, dict):
                raise ValueError(f"missing resource definition for {resource_key}")
            self._copy_file(
                Path(str(resource_def.get("path"))),
                resources_dir / Path(str(resource_def.get("path"))).name,
            )

        requirements_file.write_text(self.render_requirements_text(scenario), encoding="utf-8")
        safety_file.write_text(
            self.render_safety_text(
                scenario,
                verified_rules_by_safety_file=dict(study.get("verified_rules_by_safety_file", {}) or {}),
                resource_defs=resource_defs,
            ),
            encoding="utf-8",
        )

        product_payload = self._load_json(product_init_file)
        product_name, product_meta = self._first_named_entry(product_payload)
        updated_product_meta = dict(product_meta)
        updated_product_meta["product_specification_file"] = str(requirements_file.resolve())
        if "safety_file" in updated_product_meta:
            updated_product_meta["safety_file"] = str(safety_file.resolve())
        if "name" in product_payload:
            product_payload.update(updated_product_meta)
        else:
            product_payload = {product_name: updated_product_meta}
        atomic_json_write(product_init_file, product_payload)

        self._build_tools_catalogue(
            resources_dir=resources_dir,
            cca_init_file=cca_init_file,
            robot_env=str(study["defaults"]["robot_env"]),
            execution_mode=str(study["defaults"]["execution_mode"]),
            tools_path=tools_path,
        )
        precomputed_safety_artifacts = self._build_filtered_precomputed_safety_artifacts(
            scenario=scenario,
            scenario_dir=scenario_dir,
            safety_file=safety_file,
            tools_path=tools_path,
        )

        atomic_json_write(
            scenario_dir / "scenario_manifest.json",
            {
                "scenario": scenario,
                "requirements_file": str(requirements_file),
                "safety_file": str(safety_file),
                "tools_path": str(tools_path),
                "precomputed_safety_artifacts": precomputed_safety_artifacts,
            },
        )

        return ScenarioWorkspace(
            scenario_id=str(scenario["id"]),
            scenario_dir=scenario_dir,
            product_init_file=product_init_file,
            cca_init_file=cca_init_file,
            resources_dir=resources_dir,
            requirements_file=requirements_file,
            safety_file=safety_file,
            tools_path=tools_path,
            active_resource_names=list(scenario.get("resource_names", [])),
            parts=list(scenario.get("parts", [])),
            precomputed_safety_artifacts=precomputed_safety_artifacts,
        )

    def _extract_trial_record(
        self,
        *,
        scenario: dict[str, Any],
        method: str,
        trial_index: int,
        compile_result: dict[str, Any],
        workspace: ScenarioWorkspace,
    ) -> dict[str, Any]:
        manifest = (
            dict(compile_result.get("manifest", {}))
            if isinstance(compile_result.get("manifest"), dict)
            else {}
        )
        validation = (
            dict(manifest.get("validation_summary", {}))
            if isinstance(manifest.get("validation_summary"), dict)
            else {}
        )
        validator_stats = (
            dict(validation.get("validator_stats", {}))
            if isinstance(validation.get("validator_stats"), dict)
            else {}
        )
        repair_history = (
            list(validation.get("repair_history", []))
            if isinstance(validation.get("repair_history"), list)
            else []
        )
        violated_rules = (
            list(validation.get("violated_rules", []))
            if isinstance(validation.get("violated_rules"), list)
            else []
        )
        safety_rule_count = int(scenario.get("safety_rule_count", 0) or 0)
        violated_rule_count = min(
            safety_rule_count,
            len({str(rule_id) for rule_id in violated_rules if str(rule_id or "").strip()}),
        )
        repair_progress = self._repair_progress_from_history(
            repair_history,
            fallback_violated_rule_count=violated_rule_count,
        )
        stop_reason = str(validation.get("stop_reason", "") or "")
        validation_call_count = int(
            validation.get(
                "validation_call_count",
                1 if "verification_time_ms" in validator_stats else 0,
            )
            or 0
        )
        rule_satisfaction_evaluated = self._rule_satisfaction_was_evaluated(
            {
                "stop_reason": stop_reason,
                "validation_call_count": validation_call_count,
            }
        )
        satisfied_rule_count = (
            max(0, safety_rule_count - violated_rule_count)
            if rule_satisfaction_evaluated
            else None
        )
        rule_satisfaction_rate = (
            satisfied_rule_count / safety_rule_count
            if rule_satisfaction_evaluated and safety_rule_count and satisfied_rule_count is not None
            else None
        )
        return {
            "scenario_id": str(scenario.get("id") or ""),
            "method": str(method),
            "trial_index": int(trial_index),
            "robots": int(scenario.get("resource_count", 0) or 0),
            "parts": int(scenario.get("part_count", 0) or 0),
            "safety_rules": safety_rule_count,
            "resource_names": list(scenario.get("resource_names", [])),
            "part_ids": list(scenario.get("parts", [])),
            "product_init_file": str(scenario.get("product_init_file", "") or ""),
            "product_requirement_file": str(scenario.get("product_requirement_file", "") or ""),
            "safety_requirement_file": str(scenario.get("safety_requirement_file", "") or ""),
            "enabled_safety_rule_ids": list(scenario.get("enabled_safety_rule_ids", [])),
            "ok": bool(validation.get("ok", False)),
            "stop_reason": stop_reason,
            "violated_rules": violated_rules,
            "satisfied_rule_count": satisfied_rule_count,
            "violated_rule_count": violated_rule_count,
            "rule_satisfaction_rate": rule_satisfaction_rate,
            "rule_satisfaction_evaluated": rule_satisfaction_evaluated,
            "witness_count": int(validation.get("witness_count", 0) or 0),
            "auto_replans_used": int(validation.get("auto_replans_used", 0) or 0),
            "cumulative_validation_time_ms": float(
                validation.get(
                    "cumulative_validation_time_ms",
                    validator_stats.get("verification_time_ms", 0.0),
                )
                or 0.0
            ),
            "validation_call_count": validation_call_count,
            "grounding_summary": validation.get("grounding_summary", {}),
            "repair_history": repair_history,
            "initial_violated_rule_count": repair_progress["initial_violated_rule_count"],
            "final_violated_rule_count": repair_progress["final_violated_rule_count"],
            "repair_corrections_by_attempt": repair_progress["repair_corrections_by_attempt"],
            "repair_correction_summary": repair_progress["repair_correction_summary"],
            "validator_stats": validator_stats,
            "fsa_reachable_states": int(validator_stats.get("fsa_reachable_states", 0) or 0),
            "fsa_transitions": int(validator_stats.get("fsa_transitions", 0) or 0),
            "per_rule_product_states_explored": dict(
                validator_stats.get("per_rule_product_states_explored", {})
            )
            if isinstance(validator_stats.get("per_rule_product_states_explored"), dict)
            else {},
            "total_product_states_explored": int(
                validator_stats.get("total_product_states_explored", 0) or 0
            ),
            "verification_time_ms": float(validator_stats.get("verification_time_ms", 0.0) or 0.0),
            "product_state_limit_hit": bool(validator_stats.get("product_state_limit_hit", False)),
            "first_pass_valid": bool(validation.get("ok", False)) if method == "llm_nl_safety" else None,
            "final_verified_valid": bool(validation.get("ok", False)) if method == "verified" else None,
            "requirements_file": str(workspace.requirements_file),
            "safety_file": str(workspace.safety_file),
            "bundle_id": str(manifest.get("bundle_id", "") or ""),
            "bundle_dir": str(compile_result.get("bundle_dir", "") or ""),
        }

    def _run_trial(
        self,
        *,
        study: dict[str, Any],
        scenario: dict[str, Any],
        workspace: ScenarioWorkspace,
        method: str,
        trial_index: int,
        run_root: Path,
    ) -> dict[str, Any]:
        trial_dir = run_root / "trials" / scenario["id"] / method / f"trial_{trial_index:02d}"
        store = BundleStore(trial_dir / "bundle_store")
        compiler = BundleCompiler(
            store=store,
            project_root=self.project_root,
            product_init_dir=workspace.product_init_file.parent,
            resource_init_dir=workspace.resources_dir,
            cca_init_path=workspace.cca_init_file,
            tools_path=workspace.tools_path,
            prompts_path=_DEFAULT_PROMPTS,
        )
        auto_replan_max_attempts = 0 if method == "llm_nl_safety" else int(
            study["defaults"]["auto_replan_max_attempts"]
        )
        # Both methods are evaluated against the same approved formal rule subset.
        # The baseline still differs because it receives no verifier-driven replanning
        # feedback and is limited to a single LLM planning attempt.
        precomputed_safety_artifacts = (
            dict(workspace.precomputed_safety_artifacts or {})
            if workspace.precomputed_safety_artifacts
            else None
        )
        resource_files = []
        resource_defs = dict(study.get("resource_definitions", {}) or {})
        for resource_key in scenario.get("resource_keys", []):
            resource_def = resource_defs.get(resource_key)
            if not isinstance(resource_def, dict):
                continue
            source_path = Path(str(resource_def.get("path") or ""))
            candidate = workspace.resources_dir / source_path.name
            if candidate.exists():
                resource_files.append(str(candidate.resolve()))
        if not resource_files:
            resource_files = sorted(str(path.resolve()) for path in workspace.resources_dir.glob("*.json"))
        return asyncio.run(
            compiler.compile_bundle(
                product_init_file=str(workspace.product_init_file),
                execution_mode=str(study["defaults"]["execution_mode"]),
                robot_env=str(study["defaults"]["robot_env"]),
                product_requirement_file=str(workspace.requirements_file),
                safety_requirement_file=str(workspace.safety_file),
                precomputed_safety_artifacts=precomputed_safety_artifacts,
                resource_files=resource_files,
                selected_resource_keys=list(scenario.get("resource_keys", [])),
                auto_replan_max_attempts=auto_replan_max_attempts,
            )
        )

    def _trial_exception_record(
        self,
        *,
        scenario: dict[str, Any],
        method: str,
        trial_index: int,
        workspace: ScenarioWorkspace,
        exc: Exception,
    ) -> dict[str, Any]:
        safety_rule_count = int(scenario.get("safety_rule_count", 0) or 0)
        return {
            "scenario_id": str(scenario.get("id") or ""),
            "method": str(method),
            "trial_index": int(trial_index),
            "robots": int(scenario.get("resource_count", 0) or 0),
            "parts": int(scenario.get("part_count", 0) or 0),
            "safety_rules": safety_rule_count,
            "resource_names": list(scenario.get("resource_names", [])),
            "part_ids": list(scenario.get("parts", [])),
            "product_init_file": str(scenario.get("product_init_file", "") or ""),
            "product_requirement_file": str(scenario.get("product_requirement_file", "") or ""),
            "safety_requirement_file": str(scenario.get("safety_requirement_file", "") or ""),
            "enabled_safety_rule_ids": list(scenario.get("enabled_safety_rule_ids", [])),
            "ok": False,
            "stop_reason": "trial_exception",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "violated_rules": [],
            "satisfied_rule_count": None,
            "violated_rule_count": 0,
            "rule_satisfaction_rate": None,
            "rule_satisfaction_evaluated": False,
            "witness_count": 0,
            "auto_replans_used": 0,
            "cumulative_validation_time_ms": 0.0,
            "validation_call_count": 0,
            "grounding_summary": {},
            "repair_history": [],
            "initial_violated_rule_count": 0,
            "final_violated_rule_count": 0,
            "repair_corrections_by_attempt": {},
            "repair_correction_summary": "none",
            "validator_stats": {},
            "fsa_reachable_states": 0,
            "fsa_transitions": 0,
            "per_rule_product_states_explored": {},
            "total_product_states_explored": 0,
            "verification_time_ms": 0.0,
            "product_state_limit_hit": False,
            "first_pass_valid": False if method == "llm_nl_safety" else None,
            "final_verified_valid": False if method == "verified" else None,
            "requirements_file": str(workspace.requirements_file),
            "safety_file": str(workspace.safety_file),
            "bundle_id": "",
            "bundle_dir": "",
        }

    def _aggregate_rows(
        self,
        *,
        study: dict[str, Any],
        trial_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        scenario_lookup = {
            str(item.get("id") or ""): item
            for item in study.get("scenarios", [])
            if isinstance(item, dict)
        }
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in trial_rows:
            grouped[(str(row.get("scenario_id") or ""), str(row.get("method") or ""))].append(row)

        aggregate_rows: list[dict[str, Any]] = []
        for (scenario_id, method), rows in sorted(grouped.items()):
            scenario = scenario_lookup.get(scenario_id, {})
            validator_rule_totals: dict[str, list[float]] = defaultdict(list)
            for row in rows:
                for rule_id, count in dict(row.get("per_rule_product_states_explored", {}) or {}).items():
                    validator_rule_totals[str(rule_id)].append(float(count or 0))

            total_trials = len(rows)
            ok_trials = sum(1 for row in rows if bool(row.get("ok", False)))
            invalid_trials = total_trials - ok_trials
            rule_satisfaction_evaluated = any(
                self._rule_satisfaction_was_evaluated(row)
                for row in rows
            )
            rule_satisfaction_rate = self._mean_optional_number(
                [self._trial_rule_satisfaction_rate(row) for row in rows]
            )
            repair_attempts = sorted(
                {
                    attempt
                    for row in rows
                    for attempt in self._repair_corrections_dict(row)
                }
            )
            repair_corrections_by_attempt = {
                str(attempt): self._mean_optional_number(
                    [
                        self._repair_corrections_dict(row).get(attempt)
                        for row in rows
                    ]
                )
                for attempt in repair_attempts
            }
            repair_corrections_for_summary = {
                int(attempt): float(value)
                for attempt, value in repair_corrections_by_attempt.items()
                if value is not None
            }
            aggregate_rows.append(
                {
                    "scenario_id": scenario_id,
                    "method": method,
                    "robots": int(scenario.get("resource_count", rows[0].get("robots", 0)) if rows else 0),
                    "parts": int(scenario.get("part_count", rows[0].get("parts", 0)) if rows else 0),
                    "safety_rules": int(
                        scenario.get("safety_rule_count", rows[0].get("safety_rules", 0)) if rows else 0
                    ),
                    "trials": total_trials,
                    "valid_trials": ok_trials,
                    "invalid_trials": invalid_trials,
                    "valid_rate": ok_trials / total_trials if total_trials else 0.0,
                    "rule_satisfaction_rate": rule_satisfaction_rate,
                    "rule_satisfaction_evaluated": rule_satisfaction_evaluated,
                    "avg_initial_violated_rules": self._mean_optional_number(
                        [
                            row.get("initial_violated_rule_count")
                            for row in rows
                        ]
                    ),
                    "avg_final_violated_rules": self._mean_optional_number(
                        [
                            row.get("final_violated_rule_count")
                            for row in rows
                        ]
                    ),
                    "avg_repair_corrections_by_attempt": repair_corrections_by_attempt,
                    "repair_correction_summary": self._format_repair_correction_summary(
                        repair_corrections_for_summary
                    ),
                    "unsafe_or_invalid_rate": invalid_trials / total_trials if total_trials else 0.0,
                    "first_pass_validity_rate": (
                        ok_trials / total_trials if total_trials and method == "llm_nl_safety" else None
                    ),
                    "final_verified_validity_rate": (
                        ok_trials / total_trials if total_trials and method == "verified" else None
                    ),
                    "first_pass_rule_satisfaction_rate": (
                        rule_satisfaction_rate if method == "llm_nl_safety" else None
                    ),
                    "final_verified_rule_satisfaction_rate": (
                        rule_satisfaction_rate if method == "verified" else None
                    ),
                    "avg_verification_time_ms": self._mean_number(
                        [float(row.get("verification_time_ms", 0.0) or 0.0) for row in rows]
                    ),
                    "avg_cumulative_validation_time_ms": self._mean_number(
                        [float(row.get("cumulative_validation_time_ms", 0.0) or 0.0) for row in rows]
                    ),
                    "avg_validation_call_count": self._mean_number(
                        [float(row.get("validation_call_count", 0) or 0) for row in rows]
                    ),
                    "avg_fsa_reachable_states": self._mean_number(
                        [float(row.get("fsa_reachable_states", 0) or 0) for row in rows]
                    ),
                    "avg_fsa_transitions": self._mean_number(
                        [float(row.get("fsa_transitions", 0) or 0) for row in rows]
                    ),
                    "avg_total_product_states_explored": self._mean_number(
                        [float(row.get("total_product_states_explored", 0) or 0) for row in rows]
                    ),
                    "avg_auto_replans_used": self._mean_number(
                        [float(row.get("auto_replans_used", 0) or 0) for row in rows]
                    ),
                    "product_state_limit_hits": sum(
                        1 for row in rows if bool(row.get("product_state_limit_hit", False))
                    ),
                    "avg_per_rule_product_states_explored": {
                        rule_id: self._mean_number(values)
                        for rule_id, values in sorted(validator_rule_totals.items())
                    },
                }
            )
        return aggregate_rows

    @staticmethod
    def _paper_rule_satisfaction_rate(row: dict[str, Any]) -> float | None:
        if not isinstance(row, dict):
            return None
        if not OfflineStudyRunner._rule_satisfaction_was_evaluated(row):
            return None
        method = OfflineStudyRunner.normalize_method_name(str(row.get("method") or ""))
        if method == "llm_nl_safety":
            value = row.get("first_pass_rule_satisfaction_rate")
            resolved = (
                float(value)
                if value is not None
                else OfflineStudyRunner._trial_rule_satisfaction_rate(row)
            )
            return resolved if resolved is not None else float(row.get("valid_rate", 0.0) or 0.0)
        if method == "verified":
            value = row.get("final_verified_rule_satisfaction_rate")
            resolved = (
                float(value)
                if value is not None
                else OfflineStudyRunner._trial_rule_satisfaction_rate(row)
            )
            return resolved if resolved is not None else float(row.get("valid_rate", 0.0) or 0.0)
        value = row.get("rule_satisfaction_rate")
        return float(value) if value is not None else (
            float(row.get("valid_rate", 0.0) or 0.0) if "valid_rate" in row else None
        )

    @classmethod
    def build_paired_comparison_rows(
        cls,
        aggregate_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        scenario_order: list[str] = []
        grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in aggregate_rows:
            if not isinstance(row, dict):
                continue
            scenario_id = str(row.get("scenario_id") or "").strip()
            method = cls.normalize_method_name(str(row.get("method") or ""))
            if not scenario_id or method not in {"llm_nl_safety", "verified"}:
                continue
            if scenario_id not in grouped:
                scenario_order.append(scenario_id)
            grouped[scenario_id][method] = dict(row)

        paired_rows: list[dict[str, Any]] = []
        for scenario_id in scenario_order:
            by_method = grouped.get(scenario_id, {})
            pure_row = by_method.get("llm_nl_safety")
            verified_row = by_method.get("verified")
            reference_row = pure_row or verified_row or {}
            pure_trials = int((pure_row or {}).get("trials", 0) or 0)
            verified_trials = int((verified_row or {}).get("trials", 0) or 0)
            pure_rate = cls._paper_rule_satisfaction_rate(pure_row or {})
            verified_rate = cls._paper_rule_satisfaction_rate(verified_row or {})
            paired_rows.append(
                {
                    "scenario_id": scenario_id,
                    "robots": int(reference_row.get("robots", 0) or 0),
                    "parts": int(reference_row.get("parts", 0) or 0),
                    "safety_rules": int(reference_row.get("safety_rules", 0) or 0),
                    "pure_llm_trials": pure_trials,
                    "verified_trials": verified_trials,
                    "paired_trials": min(pure_trials, verified_trials) if pure_trials and verified_trials else 0,
                    "pure_llm_rule_satisfaction_rate": pure_rate,
                    "verified_rule_satisfaction_rate": verified_rate,
                    "mean_verified_repair_attempts": (
                        float((verified_row or {}).get("avg_auto_replans_used", 0.0) or 0.0)
                        if verified_trials
                        else None
                    ),
                    "mean_cumulative_verified_time_ms": (
                        float((verified_row or {}).get("avg_cumulative_validation_time_ms", 0.0) or 0.0)
                        if verified_trials
                        else None
                    ),
                    "delta_rule_satisfaction_rate": (
                        round(float(verified_rate - pure_rate), 12)
                        if pure_rate is not None and verified_rate is not None
                        else None
                    ),
                }
            )
        return paired_rows

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "scenario_id",
            "method",
            "robots",
            "parts",
            "safety_rules",
            "trials",
            "valid_trials",
            "invalid_trials",
            "valid_rate",
            "rule_satisfaction_rate",
            "avg_initial_violated_rules",
            "avg_final_violated_rules",
            "repair_correction_summary",
            "unsafe_or_invalid_rate",
            "first_pass_validity_rate",
            "final_verified_validity_rate",
            "first_pass_rule_satisfaction_rate",
            "final_verified_rule_satisfaction_rate",
            "avg_verification_time_ms",
            "avg_cumulative_validation_time_ms",
            "avg_validation_call_count",
            "avg_fsa_reachable_states",
            "avg_fsa_transitions",
            "avg_total_product_states_explored",
            "avg_auto_replans_used",
            "product_state_limit_hits",
        ]
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})

    @staticmethod
    def _format_markdown_rate(value: float | None) -> str:
        if value is None:
            return "N/A"
        return f"{float(value):.1%}"

    @staticmethod
    def _format_markdown_delta(value: float | None) -> str:
        if value is None:
            return "N/A"
        return f"{float(value) * 100:+.1f} pp"

    @staticmethod
    def _format_markdown_float(value: float | None) -> str:
        if value is None:
            return "N/A"
        return f"{float(value):.1f}"

    @classmethod
    def build_markdown_table(cls, rows: list[dict[str, Any]]) -> str:
        candidate_rows = (
            rows
            if rows and "pure_llm_rule_satisfaction_rate" in rows[0]
            else cls.build_paired_comparison_rows(rows)
        )
        comparison_rows = list(candidate_rows)
        header = (
            "| Scenario | Scale (Robots / Parts / Rules) | Trials | "
            "Pure LLM Rule Satisfaction % (single-shot) | "
            "LLM + Formal Verification Rule Satisfaction % (<=5 repairs) | "
            "Mean Verified Repair Attempts | Mean Cumulative Verified Time (ms) | Delta |\n"
            "|---|---|---:|---:|---:|---:|---:|---:|\n"
        )
        body_lines = []
        for row in comparison_rows:
            body_lines.append(
                "| {scenario_id} | {scale} | {paired_trials} | {pure_llm_valid_rate} | "
                "{verified_valid_rate} | {mean_verified_repair_attempts} | "
                "{mean_cumulative_verified_time_ms} | {delta_valid_rate} |".format(
                    scenario_id=row.get("scenario_id", ""),
                    scale=(
                        f"{int(row.get('robots', 0) or 0)} / "
                        f"{int(row.get('parts', 0) or 0)} / "
                        f"{int(row.get('safety_rules', 0) or 0)}"
                    ),
                    paired_trials=(
                        int(row.get("paired_trials", 0) or 0)
                        if int(row.get("paired_trials", 0) or 0) > 0
                        else max(
                            int(row.get("pure_llm_trials", 0) or 0),
                            int(row.get("verified_trials", 0) or 0),
                        )
                    ),
                    pure_llm_valid_rate=cls._format_markdown_rate(
                        row.get("pure_llm_rule_satisfaction_rate")
                    ),
                    verified_valid_rate=cls._format_markdown_rate(
                        row.get("verified_rule_satisfaction_rate")
                    ),
                    mean_verified_repair_attempts=cls._format_markdown_float(
                        row.get("mean_verified_repair_attempts")
                    ),
                    mean_cumulative_verified_time_ms=cls._format_markdown_float(
                        row.get("mean_cumulative_verified_time_ms")
                    ),
                    delta_valid_rate=cls._format_markdown_delta(
                        row.get("delta_rule_satisfaction_rate")
                    ),
                )
            )
        return header + "\n".join(body_lines) + ("\n" if body_lines else "")

    @staticmethod
    def _filter_study_scenarios(
        study: dict[str, Any],
        *,
        scenario_id: str = "",
    ) -> dict[str, Any]:
        selected = str(scenario_id or "").strip()
        if not selected:
            return study
        scenarios = [dict(item) for item in study.get("scenarios", []) if isinstance(item, dict)]
        filtered = [scenario for scenario in scenarios if str(scenario.get("id") or "") == selected]
        if not filtered:
            raise ValueError(f"selected scenario not found in study: {selected}")
        scoped_study = dict(study)
        scoped_study["scenarios"] = filtered
        return scoped_study

    def run(self, *, scenario_id: str = "") -> dict[str, Any]:
        study = self._filter_study_scenarios(
            self.load_study_definition(),
            scenario_id=scenario_id,
        )
        run_id = utc_now_compact()
        run_root = (self.results_root / str(study["study_id"]) / run_id).resolve()
        run_root.mkdir(parents=True, exist_ok=True)

        self._copy_file(self.manifest_path, run_root / "study_manifest.json")
        atomic_json_write(run_root / "normalized_study.json", self._json_safe(study))

        trial_rows: list[dict[str, Any]] = []
        methods = [str(method) for method in study["defaults"]["methods"]]
        total_trials = sum(
            int(scenario.get("trials", 0) or 0) * len(methods)
            for scenario in study.get("scenarios", [])
            if isinstance(scenario, dict)
        )
        global_trial_index = 0
        log.info(
            "[Experiment] Run %s started: study=%s scenarios=%d methods=%s total_trials=%d results=%s",
            run_id,
            study["study_id"],
            len(study.get("scenarios", []) or []),
            ",".join(methods),
            total_trials,
            run_root,
        )
        for scenario in study.get("scenarios", []):
            workspace = self.prepare_scenario_workspace(study, scenario, run_root=run_root)
            scenario_id = str(scenario.get("id") or "")
            scenario_trials = int(scenario["trials"])
            log.info(
                "[Experiment] Run %s scenario=%s prepared: scale=%d/%d/%d trials_per_method=%d resources=%s parts=%s",
                run_id,
                scenario_id,
                int(scenario.get("resource_count", 0) or 0),
                int(scenario.get("part_count", 0) or 0),
                int(scenario.get("safety_rule_count", 0) or 0),
                scenario_trials,
                ",".join(str(item) for item in scenario.get("resource_names", []) or []),
                ",".join(str(item) for item in scenario.get("parts", []) or []),
            )
            for method in methods:
                for trial_index in range(1, scenario_trials + 1):
                    global_trial_index += 1
                    log.info(
                        "[Experiment] Run %s scenario=%s method=%s trial=%d/%d global=%d/%d START",
                        run_id,
                        scenario_id,
                        method,
                        trial_index,
                        scenario_trials,
                        global_trial_index,
                        total_trials,
                    )
                    try:
                        compile_result = self._run_trial(
                            study=study,
                            scenario=scenario,
                            workspace=workspace,
                            method=method,
                            trial_index=trial_index,
                            run_root=run_root,
                        )
                    except Exception as exc:
                        log.exception(
                            "[Experiment] Run %s scenario=%s method=%s trial=%d/%d global=%d/%d FAILED before raw record",
                            run_id,
                            scenario_id,
                            method,
                            trial_index,
                            scenario_trials,
                            global_trial_index,
                            total_trials,
                        )
                        trial_record = self._trial_exception_record(
                            scenario=scenario,
                            method=method,
                            trial_index=trial_index,
                            workspace=workspace,
                            exc=exc,
                        )
                        compile_result = {
                            "summary": {
                                "status": "invalid",
                                "verified": False,
                                "stop_reason": "trial_exception",
                            },
                            "manifest": {},
                            "validation_summary": {
                                "ok": False,
                                "stop_reason": "trial_exception",
                                "exception_type": type(exc).__name__,
                                "exception_message": str(exc),
                            },
                            "bundle_dir": "",
                        }
                    else:
                        trial_record = self._extract_trial_record(
                            scenario=scenario,
                            method=method,
                            trial_index=trial_index,
                            compile_result=compile_result,
                            workspace=workspace,
                        )
                    rule_sat = trial_record.get("rule_satisfaction_rate")
                    rule_sat_text = (
                        "N/A"
                        if rule_sat is None
                        else f"{float(rule_sat) * 100:.1f}%"
                    )
                    log.info(
                        "[Experiment] Run %s scenario=%s method=%s trial=%d/%d global=%d/%d DONE ok=%s stop=%s rule_sat=%s violated=%s repairs=%d validation_calls=%d",
                        run_id,
                        scenario_id,
                        method,
                        trial_index,
                        scenario_trials,
                        global_trial_index,
                        total_trials,
                        bool(trial_record.get("ok", False)),
                        str(trial_record.get("stop_reason", "") or ""),
                        rule_sat_text,
                        ",".join(str(item) for item in trial_record.get("violated_rules", []) or []) or "none",
                        int(trial_record.get("auto_replans_used", 0) or 0),
                        int(trial_record.get("validation_call_count", 0) or 0),
                    )
                    raw_payload = {
                        "study_id": study["study_id"],
                        "run_id": run_id,
                        "scenario": scenario,
                        "method": method,
                        "trial_index": trial_index,
                        "workspace": {
                            "requirements_file": str(workspace.requirements_file),
                            "safety_file": str(workspace.safety_file),
                            "tools_path": str(workspace.tools_path),
                            "resources_dir": str(workspace.resources_dir),
                        },
                        "trial_record": trial_record,
                        "compile_result": compile_result,
                    }
                    raw_path = run_root / "raw" / scenario["id"] / str(method) / f"trial_{trial_index:02d}.json"
                    atomic_json_write(raw_path, raw_payload)
                    trial_rows.append(trial_record)

        aggregate_rows = self._aggregate_rows(study=study, trial_rows=trial_rows)
        paired_comparison_rows = self.build_paired_comparison_rows(aggregate_rows)
        summary_json_path = run_root / "aggregate_summary.json"
        summary_csv_path = run_root / "aggregate_summary.csv"
        markdown_path = run_root / "paper_table.md"
        markdown_text = self.build_markdown_table(paired_comparison_rows)

        atomic_json_write(
            summary_json_path,
            {
                "study_id": study["study_id"],
                "run_id": run_id,
                "run_root": str(run_root),
                "aggregate_rows": aggregate_rows,
                "paired_comparison_rows": paired_comparison_rows,
            },
        )
        self._write_csv(summary_csv_path, aggregate_rows)
        markdown_path.write_text(markdown_text, encoding="utf-8")
        log.info(
            "[Experiment] Run %s completed: trials=%d summary=%s table=%s",
            run_id,
            len(trial_rows),
            summary_json_path,
            markdown_path,
        )

        return {
            "study_id": study["study_id"],
            "run_id": run_id,
            "run_root": str(run_root),
            "executed_scenario_ids": [str(item.get("id") or "") for item in study.get("scenarios", [])],
            "aggregate_summary_json": str(summary_json_path),
            "aggregate_summary_csv": str(summary_csv_path),
            "paper_table_markdown": str(markdown_path),
            "aggregate_rows": aggregate_rows,
            "paired_comparison_rows": paired_comparison_rows,
            "markdown_preview": markdown_text,
        }

    @classmethod
    def list_runs(
        cls,
        manifest_path: str | Path,
        *,
        project_root: str | Path | None = None,
        results_root: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        runner = cls(manifest_path, project_root=project_root, results_root=results_root)
        study = runner.load_study_definition()
        study_root = (runner.results_root / str(study["study_id"])).resolve()
        if not study_root.exists():
            return []
        rows: list[dict[str, Any]] = []
        for run_dir in sorted([p for p in study_root.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True):
            summary_path = run_dir / "aggregate_summary.json"
            row = {
                "run_id": run_dir.name,
                "run_root": str(run_dir),
                "summary_path": str(summary_path),
                "has_summary": summary_path.exists(),
            }
            rows.append(row)
        return rows

    @classmethod
    def load_run_summary(
        cls,
        manifest_path: str | Path,
        *,
        run_root: str | Path | None = None,
        project_root: str | Path | None = None,
        results_root: str | Path | None = None,
    ) -> dict[str, Any] | None:
        runner = cls(manifest_path, project_root=project_root, results_root=results_root)
        study = runner.load_study_definition()
        if run_root:
            target_root = Path(run_root).resolve()
        else:
            runs = cls.list_runs(
                manifest_path,
                project_root=project_root,
                results_root=results_root,
            )
            if not runs:
                return None
            target_root = Path(str(runs[0]["run_root"])).resolve()

        summary_json = target_root / "aggregate_summary.json"
        if not summary_json.exists():
            return None
        payload = runner._load_json(summary_json)
        aggregate_rows = [
            {
                **dict(row),
                "method": cls.normalize_method_name(str(dict(row).get("method") or "")),
            }
            for row in (payload.get("aggregate_rows", []) if isinstance(payload.get("aggregate_rows"), list) else [])
            if isinstance(row, dict)
        ]
        groups: list[dict[str, Any]] = []
        grouped_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        raw_trial_records: list[dict[str, Any]] = []
        raw_dir = target_root / "raw"
        for raw_path in sorted(raw_dir.glob("*/*/trial_*.json")):
            try:
                raw_payload = runner._load_json(raw_path)
            except Exception:
                continue
            trial_record = raw_payload.get("trial_record", {})
            if not isinstance(trial_record, dict):
                continue
            raw_trial_records.append(dict(trial_record))
            key = (
                str(trial_record.get("scenario_id") or ""),
                cls.normalize_method_name(str(trial_record.get("method") or "")),
            )
            grouped_rows[key].append(
                {
                    "trial_index": int(trial_record.get("trial_index", 0) or 0),
                    "ok": bool(trial_record.get("ok", False)),
                    "witness_count": int(trial_record.get("witness_count", 0) or 0),
                    "verification_time_ms": float(trial_record.get("verification_time_ms", 0.0) or 0.0),
                    "raw_path": str(raw_path),
                }
            )
        recomputed_aggregate_rows = False
        if raw_trial_records and (
            any("rule_satisfaction_rate" not in row for row in aggregate_rows)
            or any("avg_initial_violated_rules" not in row for row in aggregate_rows)
            or any(not cls._rule_satisfaction_was_evaluated(row) for row in raw_trial_records)
        ):
            aggregate_rows = runner._aggregate_rows(study=study, trial_rows=raw_trial_records)
            recomputed_aggregate_rows = True
        raw_paired_rows = [
            dict(row)
            for row in (
                payload.get("paired_comparison_rows", [])
                if isinstance(payload.get("paired_comparison_rows"), list)
                else []
            )
            if isinstance(row, dict)
        ]
        paired_comparison_rows = (
            raw_paired_rows
            if (
                not recomputed_aggregate_rows
                and raw_paired_rows
                and "pure_llm_rule_satisfaction_rate" in raw_paired_rows[0]
            )
            else cls.build_paired_comparison_rows(aggregate_rows)
        )
        for row in aggregate_rows:
            scenario_id = str(row.get("scenario_id") or "")
            method = cls.normalize_method_name(str(row.get("method") or ""))
            groups.append(
                {
                    **dict(row),
                    "method": method,
                    "method_label": cls.method_label(method),
                    "trials_list": sorted(
                        grouped_rows.get((scenario_id, method), []),
                        key=lambda item: int(item.get("trial_index", 0) or 0),
                    ),
                }
            )
        markdown_path = target_root / "paper_table.md"
        markdown_preview = markdown_path.read_text(encoding="utf-8") if markdown_path.exists() else ""
        if "Rule Satisfaction" not in markdown_preview:
            markdown_preview = cls.build_markdown_table(paired_comparison_rows)
        return {
            "study_id": study["study_id"],
            "run_root": str(target_root),
            "aggregate_summary_json": str(summary_json),
            "aggregate_summary_csv": str(target_root / "aggregate_summary.csv"),
            "paper_table_markdown": str(markdown_path),
            "aggregate_rows": aggregate_rows,
            "paired_comparison_rows": paired_comparison_rows,
            "groups": groups,
            "markdown_preview": markdown_preview,
        }

    @classmethod
    def load_trial_details(
        cls,
        manifest_path: str | Path,
        *,
        run_root: str | Path,
        scenario_id: str,
        method: str,
        trial_index: int,
        project_root: str | Path | None = None,
        results_root: str | Path | None = None,
    ) -> dict[str, Any]:
        runner = cls(manifest_path, project_root=project_root, results_root=results_root)
        method_name = cls.normalize_method_name(method)
        raw_path = (
            Path(run_root).resolve()
            / "raw"
            / str(scenario_id).strip()
            / method_name
            / f"trial_{int(trial_index):02d}.json"
        )
        if not raw_path.exists():
            legacy_path = (
                Path(run_root).resolve()
                / "raw"
                / str(scenario_id).strip()
                / str(method).strip()
                / f"trial_{int(trial_index):02d}.json"
            )
            raw_path = legacy_path
        if not raw_path.exists():
            raise FileNotFoundError(f"trial payload missing: {raw_path}")

        raw_payload = runner._load_json(raw_path)
        compile_result = dict(raw_payload.get("compile_result", {})) if isinstance(raw_payload.get("compile_result"), dict) else {}
        manifest = dict(compile_result.get("manifest", {})) if isinstance(compile_result.get("manifest"), dict) else {}
        artifacts = dict(manifest.get("artifacts", {})) if isinstance(manifest.get("artifacts"), dict) else {}
        bundle_dir = Path(str(compile_result.get("bundle_dir") or "")).resolve() if str(compile_result.get("bundle_dir") or "").strip() else None

        def _load_artifact(rel_path: str) -> Any:
            if not bundle_dir or not rel_path:
                return None
            path = (bundle_dir / rel_path).resolve()
            if not path.exists():
                return None
            if path.suffix.lower() == ".json":
                try:
                    return cls._load_json(path)
                except Exception:
                    return None
            return path.read_text(encoding="utf-8")

        plan_json = _load_artifact(str(artifacts.get("plan_json", "") or ""))
        global_fsa_json = _load_artifact(str(artifacts.get("global_fsa_json", "") or ""))
        requirements_json = _load_artifact(str(artifacts.get("requirements_json", "") or ""))
        validation_json = _load_artifact(
            str(artifacts.get("plan_validation_json", artifacts.get("offline_validation_json", "")) or "")
        )

        workspace = dict(raw_payload.get("workspace", {})) if isinstance(raw_payload.get("workspace"), dict) else {}
        requirements_text = ""
        safety_text = ""
        requirements_file = Path(str(workspace.get("requirements_file") or "")).resolve() if str(workspace.get("requirements_file") or "").strip() else None
        safety_file = Path(str(workspace.get("safety_file") or "")).resolve() if str(workspace.get("safety_file") or "").strip() else None
        if requirements_file and requirements_file.exists():
            requirements_text = requirements_file.read_text(encoding="utf-8")
        if safety_file and safety_file.exists():
            safety_text = safety_file.read_text(encoding="utf-8")

        return {
            "run_root": str(Path(run_root).resolve()),
            "scenario_id": str(scenario_id).strip(),
            "method": method_name,
            "method_label": cls.method_label(method_name),
            "trial_index": int(trial_index),
            "raw_path": str(raw_path),
            "raw_payload": raw_payload,
            "trial_record": raw_payload.get("trial_record", {}),
            "validation_summary": manifest.get("validation_summary", {}),
            "validation_json": validation_json,
            "requirements_text": requirements_text,
            "safety_text": safety_text,
            "requirements_json": requirements_json,
            "plan_json": plan_json,
            "global_fsa_json": global_fsa_json,
        }


def latest_study_run_summary(
    manifest_path: str | Path,
    *,
    project_root: str | Path | None = None,
    results_root: str | Path | None = None,
) -> dict[str, Any] | None:
    """Load the latest completed run summary for a study manifest."""
    return OfflineStudyRunner.load_run_summary(
        manifest_path,
        project_root=project_root,
        results_root=results_root,
    )


def run_hybrid_vs_procedural_comparison(
    scenarios: list[dict[str, Any]],
    *,
    results_root: str | Path | None = None,
) -> dict[str, Any]:
    """Write a lightweight hybrid-vs-procedural comparison CSV.

    Each scenario entry should already include the paired outcome fields for the
    two DES modes, for example::

        {
            "scenario_id": "case3",
            "disruption_class": "cross_robot_handoff",
            "baseline_found": False,
            "hybrid_found": True,
            "baseline_trace_length": None,
            "hybrid_trace_length": 4,
            "baseline_solver_states_explored": 12,
            "hybrid_solver_states_explored": 21,
            "baseline_safety_violations": 1,
            "hybrid_safety_violations": 0,
            "derived_labels_used": 4,
            "revisions_used": 1,
        }
    """
    target_root = Path(results_root).resolve() if results_root else _DEFAULT_RESULTS_ROOT
    target_root.mkdir(parents=True, exist_ok=True)
    csv_path = target_root / "hybrid_vs_procedural_comparison.csv"

    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            continue
        row = {
            "scenario_id": str(scenario.get("scenario_id") or scenario.get("id") or "").strip(),
            "disruption_class": str(scenario.get("disruption_class") or "").strip(),
            "baseline_found": bool(scenario.get("baseline_found", False)),
            "hybrid_found": bool(scenario.get("hybrid_found", False)),
            "trace_length_delta": (
                (int(scenario.get("hybrid_trace_length") or 0) - int(scenario.get("baseline_trace_length") or 0))
                if scenario.get("hybrid_trace_length") not in (None, "")
                and scenario.get("baseline_trace_length") not in (None, "")
                else None
            ),
            "solver_states_explored": int(scenario.get("hybrid_solver_states_explored") or 0),
            "safety_violations": int(scenario.get("hybrid_safety_violations") or 0),
            "derived_labels_used": int(scenario.get("derived_labels_used") or 0),
            "revisions_used": int(scenario.get("revisions_used") or 0),
        }
        rows.append(row)
        grouped[row["disruption_class"]].append(row)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario_id",
                "disruption_class",
                "baseline_found",
                "hybrid_found",
                "trace_length_delta",
                "solver_states_explored",
                "safety_violations",
                "derived_labels_used",
                "revisions_used",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    by_disruption_class: list[dict[str, Any]] = []
    for disruption_class, disruption_rows in sorted(grouped.items()):
        by_disruption_class.append({
            "disruption_class": disruption_class,
            "scenario_count": len(disruption_rows),
            "baseline_success_rate": sum(1 for row in disruption_rows if row["baseline_found"]) / len(disruption_rows),
            "hybrid_success_rate": sum(1 for row in disruption_rows if row["hybrid_found"]) / len(disruption_rows),
        })

    return {
        "csv_path": str(csv_path),
        "rows": rows,
        "by_disruption_class": by_disruption_class,
    }
