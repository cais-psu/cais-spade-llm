"""Offline compiler for verified (safety + plan + validation) bundles."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .bundle_store import BundleStore
from .models import (
    BUNDLE_STATUS_DRAFT,
    BUNDLE_STATUS_INVALID,
    sha256_file,
    sha256_text,
    slug,
    utc_now_compact,
    utc_now_iso,
)

log = logging.getLogger("bundles.compiler")
DEFAULT_AUTO_REPLAN_MAX_ATTEMPTS = 5


class BundleCompiler:
    """Build and persist plan+safety bundle artifacts for later startup reuse."""

    def __init__(
        self,
        *,
        store: BundleStore,
        project_root: Path | str,
        product_init_dir: Path | str,
        resource_init_dir: Path | str,
        cca_init_path: Path | str,
        tools_path: Path | str,
        prompts_path: Path | str,
    ) -> None:
        self.store = store
        self.project_root = Path(project_root)
        self.product_init_dir = Path(product_init_dir)
        self.resource_init_dir = Path(resource_init_dir)
        self.cca_init_path = Path(cca_init_path)
        self.tools_path = Path(tools_path)
        self.prompts_path = Path(prompts_path)

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"expected JSON object at {path}")
        return data

    @staticmethod
    def _first_named_entry(raw: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if "name" in raw:
            name = str(raw.get("name", "product")).strip() or "product"
            return name, raw
        if not raw:
            raise ValueError("empty manifest")
        first_key = next(iter(raw.keys()))
        meta = raw[first_key]
        if not isinstance(meta, dict):
            raise ValueError("invalid manifest shape")
        return str(first_key), meta

    @staticmethod
    def _resource_key_from_jid(jid: str) -> str:
        token = str(jid or "").strip()
        if "@" in token:
            token = token.split("@", 1)[0]
        return token.strip()

    def _infer_selected_resource_keys(
        self,
        *,
        resource_files: list[str] | None,
        resource_jids: list[str],
    ) -> list[str]:
        keys: list[str] = []
        seen: set[str] = set()

        def _append(value: str) -> None:
            key = self._resource_key_from_jid(value)
            if key and key not in seen:
                seen.add(key)
                keys.append(key)

        if resource_files is not None:
            for raw_path in resource_files:
                path = Path(str(raw_path or "")).resolve()
                try:
                    name, meta = self._first_named_entry(self._load_json(path))
                except Exception:
                    continue
                _append(str(meta.get("jid") or f"{name}@{meta.get('domain', 'localhost')}"))
        if not keys:
            for jid in resource_jids:
                _append(jid)
        return keys

    def _load_product_meta(self, product_init_file: Path) -> tuple[str, dict[str, Any]]:
        raw = self._load_json(product_init_file)
        return self._first_named_entry(raw)

    def _load_cca_meta(self) -> tuple[str, dict[str, Any]]:
        raw = self._load_json(self.cca_init_path)
        if "cca" in raw and isinstance(raw["cca"], dict):
            meta = raw["cca"]
            name = str(meta.get("name", "cca")).strip() or "cca"
            return name, meta
        return self._first_named_entry(raw)

    def _collect_resource_refs(
        self,
        robot_env: str,
        resource_files: list[str] | None = None,
    ) -> list[Any]:
        refs: list[Any] = []
        if resource_files is None:
            paths = sorted(self.resource_init_dir.glob("*.json"))
        else:
            paths = [Path(str(raw_path or "")).resolve() for raw_path in resource_files]
        for path in paths:
            try:
                raw = self._load_json(path)
            except Exception:
                continue
            entries: list[tuple[str, dict[str, Any]]] = []
            if "name" in raw:
                entries.append((str(raw.get("name", "")), raw))
            else:
                for key, value in raw.items():
                    if isinstance(value, dict):
                        entries.append((str(key), value))
            for key, meta in entries:
                jid = str(meta.get("jid") or f"{key}@{meta.get('domain', 'localhost')}")
                env_block = meta.get(robot_env, {}) if isinstance(meta.get(robot_env), dict) else {}
                static_caps = env_block.get("static_capabilities", meta.get("static_capabilities", {}))
                refs.append(
                    SimpleNamespace(
                        jid=jid,
                        static_capabilities=static_caps if isinstance(static_caps, dict) else {},
                    )
                )
        return refs

    def _abs_path(self, path_str: str | Path) -> Path:
        p = Path(path_str)
        if p.is_absolute():
            return p
        return self.project_root / p

    def _load_known_part_tokens(self, product_meta: dict[str, Any]) -> list[str]:
        """Return part tokens from the product geometry's gazebo.parts.model_map."""
        geometry_ref = str(product_meta.get("product_geometry_file", "") or "").strip()
        if not geometry_ref:
            return []
        try:
            geometry_path = self._abs_path(geometry_ref)
            if not geometry_path.exists():
                return []
            payload = self._load_json(geometry_path)
        except Exception:
            return []
        gazebo = payload.get("gazebo", {})
        if not isinstance(gazebo, dict):
            return []
        parts = gazebo.get("parts", {})
        if not isinstance(parts, dict):
            return []
        model_map = parts.get("model_map", {})
        if not isinstance(model_map, dict):
            return []
        tokens: list[str] = []
        seen: set[str] = set()
        for name in model_map.keys():
            token = str(name or "").strip()
            if not token:
                continue
            lower = token.lower()
            if lower in seen:
                continue
            seen.add(lower)
            tokens.append(token)
        return tokens

    def _load_parent_plan_context(
        self,
        parent_bundle_id: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        bid = str(parent_bundle_id or "").strip()
        if not bid:
            return [], []

        manifest = self.store.load_manifest(bid)
        if not isinstance(manifest, dict):
            raise ValueError(f"parent plan set not found: {bid}")

        root = self.store.bundle_dir(bid)
        artifacts = manifest.get("artifacts", {}) if isinstance(manifest.get("artifacts"), dict) else {}

        requirements_nodes: list[dict[str, Any]] = []
        requirements_rel = str(artifacts.get("requirements_json", "")).strip()
        if requirements_rel:
            requirements_path = (root / requirements_rel).resolve()
            if requirements_path.exists():
                requirements_payload = self._load_json(requirements_path)
                raw_nodes = requirements_payload.get("nodes", [])
                if isinstance(raw_nodes, list):
                    requirements_nodes = [node for node in raw_nodes if isinstance(node, dict)]

        task_nodes: list[dict[str, Any]] = []
        plan_rel = str(artifacts.get("plan_json", "")).strip()
        if plan_rel:
            plan_path = (root / plan_rel).resolve()
            if plan_path.exists():
                plan_payload = self._load_json(plan_path)
                raw_nodes = plan_payload.get("nodes", [])
                if isinstance(raw_nodes, list):
                    task_nodes = [node for node in raw_nodes if isinstance(node, dict)]

        return requirements_nodes, task_nodes

    @staticmethod
    def _safety_rule_ids(rules: list[dict[str, Any]]) -> list[str]:
        out: list[str] = []
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            rid = str(rule.get("id", "")).strip()
            if rid:
                out.append(rid)
        return out

    @staticmethod
    def _dfa_rule_id_from_filename(path: Path) -> str:
        rid = str(path.stem or "").strip()
        if rid.endswith("_dfa"):
            rid = rid[: -len("_dfa")]
        return rid

    @classmethod
    def _copy_precomputed_safety_artifacts(
        cls,
        descriptor: dict[str, Any],
        safety_dir: Path,
    ) -> tuple[Path, dict[str, str]]:
        logic_src_raw = str(descriptor.get("safety_logic_json", "") or "").strip()
        if not logic_src_raw:
            raise ValueError("approved safety preview is missing safety_logic_json")
        logic_src = Path(logic_src_raw).resolve()
        if not logic_src.exists():
            raise FileNotFoundError(f"approved safety logic missing: {logic_src}")

        safety_dir.mkdir(parents=True, exist_ok=True)
        logic_dst = (safety_dir / "cca_safety_logic.json").resolve()
        shutil.copyfile(logic_src, logic_dst)

        logic_payload = cls._load_json(logic_dst)
        raw_rules = logic_payload.get("rules", [])
        rules = raw_rules if isinstance(raw_rules, list) else []
        rule_ids = cls._safety_rule_ids([rule for rule in rules if isinstance(rule, dict)])

        dfa_map: dict[str, str] = {}
        for raw_dot in descriptor.get("dfa_dot_files", []) if isinstance(descriptor.get("dfa_dot_files", []), list) else []:
            dot_src = Path(str(raw_dot or "").strip()).resolve()
            if not dot_src.exists():
                raise FileNotFoundError(f"approved safety DFA missing: {dot_src}")
            dot_dst = (safety_dir / dot_src.name).resolve()
            shutil.copyfile(dot_src, dot_dst)
            rid = cls._dfa_rule_id_from_filename(dot_dst)
            if rid:
                dfa_map[rid] = dot_dst.read_text(encoding="utf-8")

        missing_rule_ids = [rid for rid in rule_ids if rid not in dfa_map]
        if missing_rule_ids:
            raise ValueError(
                "approved safety preview is missing DFA DOT artifacts for rules: "
                + ", ".join(missing_rule_ids)
            )

        for raw_png in descriptor.get("dfa_png_files", []) if isinstance(descriptor.get("dfa_png_files", []), list) else []:
            png_src = Path(str(raw_png or "").strip()).resolve()
            if not png_src.exists():
                raise FileNotFoundError(f"approved safety DFA image missing: {png_src}")
            png_dst = (safety_dir / png_src.name).resolve()
            shutil.copyfile(png_src, png_dst)

        return logic_dst, dfa_map

    @staticmethod
    def _import_runtime_classes():
        from cais_spade_llm.agents.central_controller.central_controller_agent import (
            CentralControllerAgent,
        )
        from cais_spade_llm.agents.central_controller.plan_safety_validator import (
            PlanSafetyValidator,
        )
        from cais_spade_llm.agents.intelligent_product.product_agent import ProductAgent
        from cais_spade_llm.agents.shared_information.llm_agent import LlmAgent
        from cais_spade_llm.resources.sensor.camera_module import CameraModule

        return ProductAgent, CentralControllerAgent, PlanSafetyValidator, CameraModule, LlmAgent

    @staticmethod
    def _task_nodes_hash(nodes: list[dict[str, Any]]) -> str:
        task_nodes = [node for node in nodes if isinstance(node, dict) and node.get("type") == "task"]
        canonical = json.dumps(task_nodes, sort_keys=True, separators=(",", ":"), default=str)
        return sha256_text(canonical)

    @staticmethod
    def _violated_rules(violations: list[dict[str, Any]]) -> list[str]:
        return sorted(
            {
                str(v.get("violated_rule_id"))
                for v in violations
                if isinstance(v, dict) and v.get("violated_rule_id")
            }
        )

    @staticmethod
    def _validator_safety_rule_count(validator: Any) -> int:
        rules = getattr(validator, "rules", [])
        return len(rules) if isinstance(rules, list) else 0

    @classmethod
    def _safety_rule_counts(
        cls,
        violations: list[dict[str, Any]],
        safety_rule_count: int,
    ) -> tuple[int, int]:
        violated_rule_count = len(cls._violated_rules(violations))
        if safety_rule_count > 0:
            violated_rule_count = min(safety_rule_count, violated_rule_count)
            return violated_rule_count, max(0, safety_rule_count - violated_rule_count)
        return violated_rule_count, 0

    @staticmethod
    def _task_nodes_by_id(nodes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {
            str(node.get("id")): node
            for node in nodes
            if isinstance(node, dict)
            and node.get("type") == "task"
            and str(node.get("id") or "").strip()
        }

    @classmethod
    def _changed_task_ids(
        cls,
        before_nodes: list[dict[str, Any]],
        after_nodes: list[dict[str, Any]],
    ) -> list[str]:
        before = cls._task_nodes_by_id(before_nodes)
        after = cls._task_nodes_by_id(after_nodes)
        changed: list[str] = []
        for task_id in sorted(set(before) | set(after)):
            if json.dumps(before.get(task_id), sort_keys=True, default=str) != json.dumps(
                after.get(task_id),
                sort_keys=True,
                default=str,
            ):
                changed.append(task_id)
        return changed

    @classmethod
    def _predecessor_map_for_tasks(
        cls,
        nodes: list[dict[str, Any]],
        task_ids: list[str],
    ) -> dict[str, list[str]]:
        node_map = cls._task_nodes_by_id(nodes)
        return {
            task_id: list(node_map.get(task_id, {}).get("predecessors", []) or [])
            for task_id in task_ids
            if task_id in node_map
        }

    @classmethod
    def _repair_history_validation_entry(
        cls,
        *,
        ok: bool,
        violations: list[dict[str, Any]],
        safety_rule_count: int,
        validation_call_index: int,
        auto_replans_used: int,
        stop_reason: str,
    ) -> dict[str, Any]:
        violated_rule_count, satisfied_rule_count = cls._safety_rule_counts(
            violations,
            safety_rule_count,
        )
        return {
            "phase": "validation",
            "attempt_index": int(auto_replans_used),
            "validation_call_index": int(validation_call_index),
            "ok": bool(ok),
            "violated_rules": cls._violated_rules(violations),
            "violated_rule_count": int(violated_rule_count),
            "satisfied_rule_count": int(satisfied_rule_count),
            "safety_rule_count": int(safety_rule_count),
            "witness_count": len(violations),
            "stop_reason": str(stop_reason),
        }

    @classmethod
    def _repair_history_repair_entry(
        cls,
        *,
        attempt_index: int,
        before_hash: str,
        after_hash: str,
        changed_task_ids: list[str],
        compile_ok: bool,
        error_message: str = "",
        retry_planned: bool = False,
        restored_last_compileable: bool = False,
    ) -> dict[str, Any]:
        return {
            "phase": "repair",
            "attempt_index": int(attempt_index),
            "before_hash": str(before_hash),
            "after_hash": str(after_hash),
            "changed_task_ids": list(changed_task_ids),
            "compile_ok": bool(compile_ok),
            "error_message": str(error_message or ""),
            "retry_planned": bool(retry_planned),
            "restored_last_compileable": bool(restored_last_compileable),
        }

    @classmethod
    def _synthetic_repair_feedback(
        cls,
        *,
        error_message: str,
        changed_task_ids: list[str],
        planner_nodes: list[dict[str, Any]],
        previous_violations: list[dict[str, Any]],
        restored_last_compileable: bool,
    ) -> list[dict[str, Any]]:
        prior_task_ids: list[str] = []
        for violation in previous_violations:
            if not isinstance(violation, dict):
                continue
            for key in ("witness_task_ids", "witness_trace", "conflict_task_ids"):
                values = violation.get(key)
                if isinstance(values, (list, tuple, set)):
                    prior_task_ids.extend(str(value) for value in values if value)
                elif isinstance(values, str) and values.strip():
                    prior_task_ids.append(values.strip())
        focus_task_ids = list(dict.fromkeys([*changed_task_ids, *prior_task_ids]))
        relevant_tasks = [
            deepcopy(node)
            for node in planner_nodes
            if isinstance(node, dict) and str(node.get("id") or "") in set(focus_task_ids)
        ]
        prior_rules = cls._violated_rules(previous_violations)
        restore_text = (
            " The planner restored the last compileable graph before asking for this repair."
            if restored_last_compileable
            else " The current task graph is still available for editing, but it did not compile."
        )
        return [
            {
                "violated_rule_id": "REPAIR_GRAPH_INVALID",
                "violation_text": (
                    "The previous LLM repair produced an invalid task graph or "
                    f"same-resource requirement block order: {error_message}.{restore_text} "
                    "Return a new patch that removes the cyclic/block-order dependency while "
                    "still satisfying the original safety violation(s): "
                    + (", ".join(prior_rules) if prior_rules else "unknown")
                    + "."
                ),
                "violation_logic": "planner_compile_error",
                "witness_trace": focus_task_ids,
                "witness_task_ids": focus_task_ids,
                "conflict_task_ids": focus_task_ids,
                "relevant_tasks": relevant_tasks,
                "relevant_pred_map": cls._predecessor_map_for_tasks(
                    planner_nodes,
                    focus_task_ids,
                ),
                "error_message": str(error_message or ""),
                "previous_violated_rules": prior_rules,
            }
        ]

    @classmethod
    def build_validation_payload(
        cls,
        *,
        ok: bool,
        violations: list[dict[str, Any]],
        auto_replans_used: int,
        stop_reason: str,
        grounding_summary: dict[str, Any] | None = None,
        validator_stats: dict[str, Any] | None = None,
        cumulative_validation_time_ms: float = 0.0,
        validation_call_count: int = 0,
        repair_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        violated_rules = cls._violated_rules(violations)
        return {
            "ok": bool(ok),
            "violations": violations,
            "violated_rules": violated_rules,
            "witness_count": len(violations),
            "auto_replans_used": int(auto_replans_used),
            "stop_reason": str(stop_reason),
            "cumulative_validation_time_ms": float(cumulative_validation_time_ms or 0.0),
            "validation_call_count": int(validation_call_count or 0),
            "grounding_summary": (
                dict(grounding_summary)
                if isinstance(grounding_summary, dict)
                else {
                    "corrected_task_count": 0,
                    "invalid_task_count": 0,
                    "unresolved_task_count": 0,
                    "findings": [],
                }
            ),
            "validator_stats": dict(validator_stats or {}),
            "repair_history": list(repair_history or []),
        }

    @classmethod
    async def run_offline_repair_loop(
        cls,
        *,
        product_agent: Any,
        validator: Any,
        product_jid: str,
        auto_replan_max_attempts: int,
        seed_replan_violations: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        planner = product_agent.process_planner

        auto_replans_used = 0
        cumulative_validation_time_ms = 0.0
        validation_call_count = 0
        safety_rule_count = cls._validator_safety_rule_count(validator)
        repair_history: list[dict[str, Any]] = []
        last_validation_violations: list[dict[str, Any]] = []
        pending_repair_feedback: list[dict[str, Any]] | None = (
            list(seed_replan_violations)
            if seed_replan_violations
            else None
        )
        last_compileable_nodes = deepcopy(getattr(planner, "nodes", []) or [])
        last_compileable_fsa = deepcopy(getattr(planner, "global_fsa", None))

        while True:
            grounding_summary = (
                planner.get_last_grounding_summary()
                if hasattr(planner, "get_last_grounding_summary")
                else dict(getattr(planner, "last_grounding_summary", {}) or {})
            )
            if pending_repair_feedback is None:
                if int(grounding_summary.get("invalid_task_count", 0) or 0) > 0:
                    repair_history.append(
                        {
                            "phase": "validation",
                            "attempt_index": int(auto_replans_used),
                            "validation_call_index": int(validation_call_count),
                            "ok": False,
                            "violated_rules": [],
                            "violated_rule_count": 0,
                            "satisfied_rule_count": 0,
                            "safety_rule_count": int(safety_rule_count),
                            "witness_count": 0,
                            "stop_reason": "grounding_invalid",
                        }
                    )
                    return cls.build_validation_payload(
                        ok=False,
                        violations=[],
                        auto_replans_used=auto_replans_used,
                        stop_reason="grounding_invalid",
                        grounding_summary=grounding_summary,
                        validator_stats=dict(getattr(validator, "last_run_stats", {}) or {}),
                        cumulative_validation_time_ms=cumulative_validation_time_ms,
                        validation_call_count=validation_call_count,
                        repair_history=repair_history,
                    )

                ok, violations = validator.validate_plan_fsa(
                    fsa=planner.global_fsa or {},
                    plan={"nodes": planner.nodes},
                    product_jid=product_jid,
                )
                latest_validator_stats = dict(getattr(validator, "last_run_stats", {}) or {})
                cumulative_validation_time_ms += float(
                    latest_validator_stats.get("verification_time_ms", 0.0) or 0.0
                )
                validation_call_count += 1
                stop_reason = "valid" if ok else "violations_found"
                last_validation_violations = list(violations or [])
                repair_history.append(
                    cls._repair_history_validation_entry(
                        ok=ok,
                        violations=last_validation_violations,
                        safety_rule_count=safety_rule_count,
                        validation_call_index=validation_call_count,
                        auto_replans_used=auto_replans_used,
                        stop_reason=stop_reason,
                    )
                )
                if ok:
                    final_stop_reason = "initial_valid" if auto_replans_used == 0 else "repaired_valid"
                    repair_history[-1]["stop_reason"] = final_stop_reason
                    return cls.build_validation_payload(
                        ok=True,
                        violations=violations,
                        auto_replans_used=auto_replans_used,
                        stop_reason=final_stop_reason,
                        grounding_summary=grounding_summary,
                        validator_stats=latest_validator_stats,
                        cumulative_validation_time_ms=cumulative_validation_time_ms,
                        validation_call_count=validation_call_count,
                        repair_history=repair_history,
                    )

                if auto_replans_used >= auto_replan_max_attempts:
                    repair_history[-1]["stop_reason"] = "max_attempts_reached"
                    return cls.build_validation_payload(
                        ok=False,
                        violations=violations,
                        auto_replans_used=auto_replans_used,
                        stop_reason="max_attempts_reached",
                        grounding_summary=grounding_summary,
                        validator_stats=latest_validator_stats,
                        cumulative_validation_time_ms=cumulative_validation_time_ms,
                        validation_call_count=validation_call_count,
                        repair_history=repair_history,
                    )
                repair_feedback = last_validation_violations
            else:
                latest_validator_stats = dict(getattr(validator, "last_run_stats", {}) or {})
                if auto_replans_used >= auto_replan_max_attempts:
                    return cls.build_validation_payload(
                        ok=False,
                        violations=last_validation_violations,
                        auto_replans_used=auto_replans_used,
                        stop_reason="repair_exception",
                        grounding_summary=grounding_summary,
                        validator_stats=latest_validator_stats,
                        cumulative_validation_time_ms=cumulative_validation_time_ms,
                        validation_call_count=validation_call_count,
                        repair_history=repair_history,
                    )
                repair_feedback = pending_repair_feedback
                pending_repair_feedback = None

            before_hash = cls._task_nodes_hash(planner.nodes)
            before_nodes = deepcopy(getattr(planner, "nodes", []) or [])
            auto_replans_used += 1
            try:
                await planner.replan_with_feedback_offline(repair_feedback)
                after_hash = cls._task_nodes_hash(planner.nodes)
                changed_task_ids = cls._changed_task_ids(before_nodes, planner.nodes)
                if before_hash == after_hash:
                    error_message = "LLM repair did not change any task nodes."
                    retry_planned = auto_replans_used < auto_replan_max_attempts
                    repair_history.append(
                        cls._repair_history_repair_entry(
                            attempt_index=auto_replans_used,
                            before_hash=before_hash,
                            after_hash=after_hash,
                            changed_task_ids=changed_task_ids,
                            compile_ok=False,
                            error_message=error_message,
                            retry_planned=retry_planned,
                            restored_last_compileable=False,
                        )
                    )
                    if not retry_planned:
                        return cls.build_validation_payload(
                            ok=False,
                            violations=last_validation_violations,
                            auto_replans_used=auto_replans_used,
                            stop_reason="repair_no_change",
                            grounding_summary=(
                                planner.get_last_grounding_summary()
                                if hasattr(planner, "get_last_grounding_summary")
                                else dict(getattr(planner, "last_grounding_summary", {}) or {})
                            ),
                            validator_stats=dict(getattr(validator, "last_run_stats", {}) or {}),
                            cumulative_validation_time_ms=cumulative_validation_time_ms,
                            validation_call_count=validation_call_count,
                            repair_history=repair_history,
                        )
                    pending_repair_feedback = cls._synthetic_repair_feedback(
                        error_message=error_message,
                        changed_task_ids=changed_task_ids,
                        planner_nodes=planner.nodes,
                        previous_violations=last_validation_violations,
                        restored_last_compileable=False,
                    )
                    continue
                planner.compile_global_fsa()
                repair_history.append(
                    cls._repair_history_repair_entry(
                        attempt_index=auto_replans_used,
                        before_hash=before_hash,
                        after_hash=after_hash,
                        changed_task_ids=changed_task_ids,
                        compile_ok=True,
                        retry_planned=False,
                    )
                )
                last_compileable_nodes = deepcopy(planner.nodes)
                last_compileable_fsa = deepcopy(getattr(planner, "global_fsa", None))
            except Exception as exc:
                error_message = str(exc or "")
                log.exception("Offline replan attempt %d failed.", auto_replans_used)
                after_hash = cls._task_nodes_hash(getattr(planner, "nodes", []) or [])
                changed_task_ids = cls._changed_task_ids(
                    before_nodes,
                    getattr(planner, "nodes", []) or [],
                )
                # A repair can leave the DAG acyclic but still uncompilable for the
                # resource-level FSA layout, for example by creating conflicting
                # same-resource requirement block order. In both cases, continue
                # from the last graph that actually compiled.
                planner.nodes = deepcopy(last_compileable_nodes)
                planner.global_fsa = deepcopy(last_compileable_fsa)
                restored_last_compileable = True
                retry_planned = auto_replans_used < auto_replan_max_attempts
                repair_history.append(
                    cls._repair_history_repair_entry(
                        attempt_index=auto_replans_used,
                        before_hash=before_hash,
                        after_hash=after_hash,
                        changed_task_ids=changed_task_ids,
                        compile_ok=False,
                        error_message=error_message,
                        retry_planned=retry_planned,
                        restored_last_compileable=restored_last_compileable,
                    )
                )
                if not retry_planned:
                    return cls.build_validation_payload(
                        ok=False,
                        violations=last_validation_violations,
                        auto_replans_used=auto_replans_used,
                        stop_reason="repair_exception",
                        grounding_summary=(
                            planner.get_last_grounding_summary()
                            if hasattr(planner, "get_last_grounding_summary")
                            else dict(getattr(planner, "last_grounding_summary", {}) or {})
                        ),
                        validator_stats=dict(getattr(validator, "last_run_stats", {}) or {}),
                        cumulative_validation_time_ms=cumulative_validation_time_ms,
                        validation_call_count=validation_call_count,
                        repair_history=repair_history,
                    )
                pending_repair_feedback = cls._synthetic_repair_feedback(
                    error_message=error_message,
                    changed_task_ids=changed_task_ids,
                    planner_nodes=planner.nodes,
                    previous_violations=last_validation_violations,
                    restored_last_compileable=restored_last_compileable,
                )
                continue

    async def compile_bundle(
        self,
        *,
        product_init_file: str,
        execution_mode: str,
        robot_env: str,
        product_requirement_file: str | None = None,
        safety_requirement_file: str | None = None,
        precomputed_safety_artifacts: dict[str, Any] | None = None,
        resource_files: list[str] | None = None,
        selected_resource_keys: list[str] | None = None,
        auto_replan_max_attempts: int = DEFAULT_AUTO_REPLAN_MAX_ATTEMPTS,
        refinement_feedback: str = "",
        parent_bundle_id: str = "",
    ) -> dict[str, Any]:
        if not self.tools_path.exists():
            raise FileNotFoundError(
                f"tools catalogue missing: {self.tools_path}. Build tools.json first."
            )
        if not self.prompts_path.exists():
            raise FileNotFoundError(f"prompts file missing: {self.prompts_path}")

        product_init_path = Path(product_init_file)
        if not product_init_path.exists():
            raise FileNotFoundError(f"product init file missing: {product_init_path}")

        product_name, product_meta = self._load_product_meta(product_init_path)
        cca_name, cca_meta = self._load_cca_meta()

        default_product_spec_file = str(product_meta.get("product_specification_file", "")).strip()
        product_spec_file = str(product_requirement_file or default_product_spec_file).strip()
        if not product_spec_file:
            raise ValueError(
                "No product requirement file selected and product manifest has no product_specification_file default"
            )
        product_spec_path = self._abs_path(product_spec_file)
        if not product_spec_path.exists():
            raise FileNotFoundError(f"requirements file missing: {product_spec_path}")

        default_safety_file = str(cca_meta.get("safety_file", "")).strip()
        if not default_safety_file:
            default_safety_file = str(product_meta.get("safety_file", "")).strip()
        safety_file = str(safety_requirement_file or default_safety_file).strip()
        if not safety_file:
            raise ValueError("cca manifest missing safety_file")
        safety_path = self._abs_path(safety_file)
        if not safety_path.exists():
            raise FileNotFoundError(f"safety file missing: {safety_path}")

        requirement_text = product_spec_path.read_text(encoding="utf-8").strip()
        safety_text = safety_path.read_text(encoding="utf-8").strip()
        if not requirement_text:
            raise ValueError(f"requirements file is empty: {product_spec_path}")
        if not safety_text:
            raise ValueError(f"safety file is empty: {safety_path}")

        source_hashes = {
            "requirements_sha256": sha256_text(requirement_text),
            "safety_sha256": sha256_text(safety_text),
            "tools_sha256": sha256_file(self.tools_path),
            "prompts_sha256": sha256_file(self.prompts_path),
        }

        product_stem = slug(Path(product_spec_file).stem)
        short_hash = sha256_text(
            "|".join(
                [
                    source_hashes["requirements_sha256"],
                    source_hashes["safety_sha256"],
                    source_hashes["tools_sha256"],
                    source_hashes["prompts_sha256"],
                ]
            )
        )[:8]
        stamp = utc_now_compact()
        bundle_id = f"{stamp}__{product_stem}__{execution_mode}__{robot_env}__{short_hash}"

        ProductAgent, CentralControllerAgent, PlanSafetyValidator, CameraModule, LlmAgent = (
            self._import_runtime_classes()
        )
        LlmAgent.configure_shared_tools_catalogue(self.tools_path)

        resources = self._collect_resource_refs(
            robot_env=robot_env,
            resource_files=resource_files,
        )
        resource_jids = [str(r.jid) for r in resources]
        selected_resource_keys_for_manifest = [
            str(key).strip()
            for key in (selected_resource_keys or [])
            if str(key).strip()
        ]
        if not selected_resource_keys_for_manifest:
            selected_resource_keys_for_manifest = self._infer_selected_resource_keys(
                resource_files=resource_files,
                resource_jids=resource_jids,
            )

        os.environ["ROBOT_ENV"] = str(robot_env)
        os.environ["EXECUTION_MODE"] = str(execution_mode)
        os.environ.setdefault("PERCEPTION_BACKEND", "none")

        auto_replan_max_attempts = max(0, min(int(auto_replan_max_attempts), 10))
        refinement_feedback = str(refinement_feedback or "").strip()
        parent_bundle_id = str(parent_bundle_id or "").strip()
        previous_preview_requirements, previous_preview_tasks = self._load_parent_plan_context(parent_bundle_id)

        with self.store.generation_lock(timeout_sec=0.0):
            tmp_dir = self.store.create_temp_bundle_dir(bundle_id)
            plan_dir = tmp_dir / "plan"
            safety_dir = tmp_dir / "safety"
            validation_dir = tmp_dir / "validation"
            catalog_dir = tmp_dir / "catalog"
            plan_dir.mkdir(parents=True, exist_ok=True)
            safety_dir.mkdir(parents=True, exist_ok=True)
            validation_dir.mkdir(parents=True, exist_ok=True)
            catalog_dir.mkdir(parents=True, exist_ok=True)

            tools_snapshot_path = catalog_dir / "tools.json"
            shutil.copyfile(self.tools_path, tools_snapshot_path)

            product_pw = str(product_meta.get("password", "none"))
            product_jid = str(
                product_meta.get("jid")
                or f"{product_name}@{product_meta.get('domain', 'localhost')}"
            )
            cca_pw = str(cca_meta.get("password", "none"))
            cca_jid = str(cca_meta.get("jid") or f"{cca_name}@{cca_meta.get('domain', 'localhost')}")

            product_agent = ProductAgent(
                product_jid,
                product_pw,
                name=product_name,
                resource_jids=resource_jids,
                resource_agents=resources,
                product_specification_file=str(product_spec_path),
                product_geometry_file=product_meta.get("product_geometry_file"),
                safety_file=str(safety_path),
                instruction_override=None,
                cca_jid=cca_jid,
                camera=CameraModule(backend="none"),
            )
            known_parts = self._load_known_part_tokens(product_meta)
            cca_agent = CentralControllerAgent(
                cca_jid,
                cca_pw,
                name=cca_name,
                resource_agents=resources,
                safety_file=str(safety_path),
                safety_known_parts=known_parts,
            )

            try:
                safety_logic_path = safety_dir / "cca_safety_logic.json"
                safety_source = {
                    "mode": "live_regeneration",
                    "safety_sha256": source_hashes["safety_sha256"],
                    "tools_sha256": source_hashes["tools_sha256"],
                    "prompts_sha256": source_hashes["prompts_sha256"],
                }
                if precomputed_safety_artifacts:
                    for key in ("safety_sha256", "tools_sha256", "prompts_sha256"):
                        expected = str(precomputed_safety_artifacts.get(key, "") or "").strip()
                        actual = str(source_hashes.get(key, "") or "").strip()
                        if expected != actual:
                            raise ValueError(
                                f"approved safety preview hash mismatch for {key}: expected {expected}, current {actual}"
                            )
                    safety_logic_path, dfa_map = self._copy_precomputed_safety_artifacts(
                        precomputed_safety_artifacts,
                        safety_dir,
                    )
                    safety_payload = self._load_json(safety_logic_path)
                    raw_rules = safety_payload.get("rules", [])
                    safety_rules = [rule for rule in raw_rules if isinstance(rule, dict)] if isinstance(raw_rules, list) else []
                    safety_source = {
                        "mode": "approved_preview",
                        "preview_id": str(precomputed_safety_artifacts.get("preview_id", "")).strip(),
                        "preview_generated_at_utc": str(
                            precomputed_safety_artifacts.get("preview_generated_at_utc", "") or ""
                        ).strip(),
                        "safety_sha256": source_hashes["safety_sha256"],
                        "tools_sha256": source_hashes["tools_sha256"],
                        "prompts_sha256": source_hashes["prompts_sha256"],
                    }
                    log.info(
                        "Using approved safety preview %s for bundle %s.",
                        safety_source["preview_id"] or "<unknown>",
                        bundle_id,
                    )
                else:
                    safety_logic = getattr(cca_agent, "safety_logic", None)
                    if safety_logic is None:
                        raise RuntimeError("failed to initialize SafetyLogic")
                    await safety_logic.build_safety_rules_and_logic(safety_text)
                    await safety_logic.build_preview_interpretations()
                    await asyncio.to_thread(safety_logic.save, safety_logic_path)
                    dfa_map = await asyncio.to_thread(
                        safety_logic.build_dfas_per_rule,
                        safety_dir,
                    )
                    safety_rules = list(safety_logic.rules or [])

                await product_agent.process_planner.build_high_level(
                    requirement_text,
                    refinement_feedback=refinement_feedback,
                    previous_preview_requirements=previous_preview_requirements,
                )
                requirements_path = plan_dir / f"{product_stem}_requirements.json"
                await asyncio.to_thread(product_agent.process_planner.save, requirements_path)

                await product_agent.process_planner.expand_requirements_to_tasks(
                    safety_text=safety_text,
                    refinement_feedback=refinement_feedback,
                    previous_preview_requirements=previous_preview_requirements,
                    previous_preview_tasks=previous_preview_tasks,
                )
                plan_path = plan_dir / f"{product_stem}_plan.json"
                await asyncio.to_thread(product_agent.process_planner.save, plan_path)

                global_fsa_path = plan_dir / f"{product_stem}_global_fsa.json"
                await asyncio.to_thread(
                    product_agent.process_planner.save_global_fsa,
                    global_fsa_path,
                )

                validator = PlanSafetyValidator(
                    rules=safety_rules,
                    dfa_map=dfa_map,
                    tools_catalog=getattr(product_agent, "tools_catalog", []),
                )
                validation_payload = await self.run_offline_repair_loop(
                    product_agent=product_agent,
                    validator=validator,
                    product_jid=str(product_agent.jid),
                    auto_replan_max_attempts=auto_replan_max_attempts,
                )
                await asyncio.to_thread(product_agent.process_planner.save, plan_path)
                await asyncio.to_thread(
                    product_agent.process_planner.save_global_fsa,
                    global_fsa_path,
                )
                validation_path = validation_dir / "plan_validation.json"
                with validation_path.open("w", encoding="utf-8") as f:
                    json.dump(validation_payload, f, indent=2)

                dot_files = sorted(p.name for p in safety_dir.glob("SAFE_*_dfa.dot"))
                png_files = sorted(p.name for p in safety_dir.glob("SAFE_*_dfa.png"))
                ok = bool(validation_payload.get("ok", False))
                violated_rules = list(validation_payload.get("violated_rules", []))
                witness_count = int(validation_payload.get("witness_count", 0))
                auto_replans_used = int(validation_payload.get("auto_replans_used", 0))
                stop_reason = str(validation_payload.get("stop_reason", "max_attempts_reached"))
                cumulative_validation_time_ms = float(
                    validation_payload.get("cumulative_validation_time_ms", 0.0) or 0.0
                )
                validation_call_count = int(
                    validation_payload.get("validation_call_count", 0) or 0
                )
                grounding_summary = (
                    dict(validation_payload.get("grounding_summary", {}))
                    if isinstance(validation_payload.get("grounding_summary"), dict)
                    else {}
                )
                validator_stats = (
                    dict(validation_payload.get("validator_stats", {}))
                    if isinstance(validation_payload.get("validator_stats"), dict)
                    else {}
                )
                repair_history = (
                    list(validation_payload.get("repair_history", []))
                    if isinstance(validation_payload.get("repair_history"), list)
                    else []
                )
                status = BUNDLE_STATUS_DRAFT if ok else BUNDLE_STATUS_INVALID
                manifest = {
                    "bundle_id": bundle_id,
                    "display_name": (
                        f"{product_name} | {execution_mode}/{robot_env} | "
                        f"{'draft' if ok else 'invalid'} | {stamp}"
                    ),
                    "created_at_utc": utc_now_iso(),
                    "status": status,
                    "verified": False,
                    "product_spec_file": product_spec_file,
                    "safety_file": safety_file,
                    "product_name": product_name,
                    "execution_mode": execution_mode,
                    "robot_env": robot_env,
                    "selected_resource_keys": list(selected_resource_keys_for_manifest),
                    "llm_models": {
                        "planner_model": getattr(product_agent, "non_function_model", ""),
                        "safety_model": getattr(cca_agent, "non_function_model", ""),
                    },
                    "source_hashes": source_hashes,
                    "replan_policy": {
                        "auto_replan_max_attempts": auto_replan_max_attempts,
                    },
                    "safety_source": safety_source,
                    "parent_bundle_id": parent_bundle_id,
                    "refinement_feedback": refinement_feedback,
                    "artifacts": {
                        "tools_json": str(tools_snapshot_path.relative_to(tmp_dir)),
                        "requirements_json": str(requirements_path.relative_to(tmp_dir)),
                        "plan_json": str(plan_path.relative_to(tmp_dir)),
                        "global_fsa_json": str(global_fsa_path.relative_to(tmp_dir)),
                        "safety_logic_json": str(safety_logic_path.relative_to(tmp_dir)),
                        "safety_dfa_dot_files": [f"safety/{name}" for name in dot_files],
                        "safety_dfa_png_files": [f"safety/{name}" for name in png_files],
                        "plan_validation_json": str(validation_path.relative_to(tmp_dir)),
                        "offline_validation_json": str(validation_path.relative_to(tmp_dir)),
                    },
                    "validation_summary": {
                        "ok": bool(ok),
                        "violated_rules": violated_rules,
                        "witness_count": witness_count,
                        "auto_replans_used": auto_replans_used,
                        "stop_reason": stop_reason,
                        "cumulative_validation_time_ms": cumulative_validation_time_ms,
                        "validation_call_count": validation_call_count,
                        "grounding_summary": grounding_summary,
                        "validator_stats": validator_stats,
                        "repair_history": repair_history,
                    },
                }

                self.store.save_manifest(tmp_dir, manifest)
                final_dir = self.store.finalize_bundle_dir(tmp_dir, bundle_id)
                summary = {
                    "bundle_id": bundle_id,
                    "display_name": manifest["display_name"],
                    "created_at_utc": manifest["created_at_utc"],
                    "status": status,
                    "verified": False,
                    "product_name": product_name,
                    "product_spec_file": product_spec_file,
                    "safety_file": safety_file,
                    "execution_mode": execution_mode,
                    "robot_env": robot_env,
                    "parent_bundle_id": parent_bundle_id,
                    "refinement_feedback": refinement_feedback,
                    "manifest_path": str(final_dir / "bundle_manifest.json"),
                }
                self.store.upsert_bundle_summary(summary)
                return {
                    "bundle_id": bundle_id,
                    "summary": summary,
                    "manifest": manifest,
                    "bundle_dir": str(final_dir),
                }
            except Exception:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise
            finally:
                try:
                    if hasattr(product_agent, "camera") and product_agent.camera:
                        product_agent.camera.destroy()
                except Exception:
                    pass
