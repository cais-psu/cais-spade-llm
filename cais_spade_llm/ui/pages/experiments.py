"""Experiments page: orchestrate offline studies from Products, Safety, and Resources."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from nicegui import ui

from cais_spade_llm.experiments.offline_study import OfflineStudyRunner
from cais_spade_llm.ui.bridge import SystemBridge
from cais_spade_llm.ui.components.dag_graph import build_fsa_dag_overlay, nodes_to_mermaid

_METHOD_OPTIONS = [
    ("llm_nl_safety", "Pure LLM"),
    ("verified", "LLM + Formal Verification"),
]
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_AUTO_REPLAN_MAX_ATTEMPTS = 5


def _asset_catalog_payload(context: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(context or {})
    return {
        "product_catalog": payload.get("product_catalog", []),
        "requirement_files_by_product": payload.get("requirement_files_by_product", {}),
        "verified_safety_catalog": payload.get("verified_safety_catalog", []),
        "verified_rules_by_safety_file": payload.get("verified_rules_by_safety_file", {}),
    }


def _asset_catalog_signature(context: dict[str, Any] | None) -> str:
    return json.dumps(_asset_catalog_payload(context), sort_keys=True)


def _selected_file_name(path_ref: str | Path) -> str:
    name = Path(str(path_ref or "").strip()).name
    return name or "(none selected)"


def _read_preview_text(
    path_ref: str | Path,
    *,
    fallback: str = "",
    project_root: Path = _PROJECT_ROOT,
) -> str:
    raw = str(path_ref or "").strip()
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = project_root / path
        try:
            if path.exists() and path.is_file():
                text = path.read_text(encoding="utf-8").strip()
                if text:
                    return text
        except Exception:
            pass
    return str(fallback or "")


def _requirement_layout(
    product_requirement_file: str | Path,
    *,
    valid_parts: set[str] | None = None,
) -> dict[str, Any]:
    return OfflineStudyRunner.parse_requirement_file_layout(
        product_requirement_file,
        project_root=_PROJECT_ROOT,
        valid_parts=valid_parts,
    )


def _repair_history_rows(
    validation_summary: dict[str, Any] | None,
    trial_record: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    summary = validation_summary if isinstance(validation_summary, dict) else {}
    record = trial_record if isinstance(trial_record, dict) else {}
    raw_history = summary.get("repair_history")
    if not isinstance(raw_history, list):
        raw_history = record.get("repair_history")
    if not isinstance(raw_history, list):
        return []

    rows: list[dict[str, Any]] = []
    for idx, entry in enumerate(raw_history):
        if not isinstance(entry, dict):
            continue
        phase = str(entry.get("phase", "") or "").strip()
        attempt = int(entry.get("attempt_index", 0) or 0)
        if phase == "validation":
            step = "Initial validation" if attempt == 0 else f"Validation after replan {attempt}"
            result = "valid" if bool(entry.get("ok", False)) else str(entry.get("stop_reason") or "invalid")
        elif phase == "repair":
            step = f"Replan {attempt}"
            result = "compiled" if bool(entry.get("compile_ok", False)) else "repair failed"
        else:
            step = f"Step {idx + 1}"
            result = str(entry.get("stop_reason") or phase or "unknown")

        safety_rule_count = entry.get("safety_rule_count", "")
        satisfied_rule_count = entry.get("satisfied_rule_count", "")
        if safety_rule_count not in ("", None) and satisfied_rule_count not in ("", None):
            safety = f"{int(satisfied_rule_count or 0)}/{int(safety_rule_count or 0)}"
        else:
            safety = ""
        violated_rules = entry.get("violated_rules", [])
        changed_task_ids = entry.get("changed_task_ids", [])
        rows.append(
            {
                "row_id": f"repair_{idx}",
                "step": step,
                "result": result,
                "safety_rules": safety,
                "violated_rules": ", ".join(str(item) for item in violated_rules)
                if isinstance(violated_rules, list)
                else str(violated_rules or ""),
                "witnesses": int(entry.get("witness_count", 0) or 0),
                "changed_tasks": ", ".join(str(item) for item in changed_task_ids)
                if isinstance(changed_task_ids, list)
                else str(changed_task_ids or ""),
                "error": str(entry.get("error_message") or ""),
            }
        )
    return rows


def _format_optional_count(value: Any) -> str:
    if value in (None, ""):
        return "N/A"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def render(bridge: SystemBridge) -> None:
    ui.label("Experiments").classes("text-2xl font-bold px-6 pt-6")

    manifest_files = bridge.list_experiment_manifest_files()
    selected_manifest = manifest_files[0] if manifest_files else ""
    initial_context = (
        bridge.load_experiment_editor_context(selected_manifest) if selected_manifest else {"manifest": {}}
    )
    initial_runs = bridge.list_experiment_runs(selected_manifest) if selected_manifest else []

    state: dict[str, Any] = {
        "manifest_file": selected_manifest,
        "context": initial_context,
        "runs": initial_runs,
        "selected_run_root": str(initial_runs[0]["run_root"]) if initial_runs else "",
        "selected_scenario_index": 0,
        "analysis": None,
        "trial_details_cache": {},
        "dirty": False,
        "advanced_json": json.dumps(initial_context.get("manifest", {}), indent=2),
        "asset_catalog_signature": _asset_catalog_signature(initial_context),
        "pending_asset_catalog_signature": "",
        "asset_refresh_notice": "",
        "asset_refresh_inflight": False,
    }
    run_busy = {"value": False}
    analyze_busy = {"value": False}
    advanced_editor: Any = None
    dirty_label: Any = None
    asset_notice_label: Any = None

    def _current_context() -> dict[str, Any]:
        return dict(state.get("context") or {})

    def _current_manifest() -> dict[str, Any]:
        context = _current_context()
        manifest = context.get("manifest", {})
        if not isinstance(manifest, dict):
            manifest = {}
            context["manifest"] = manifest
            state["context"] = context
        return manifest

    def _scenario_list() -> list[dict[str, Any]]:
        manifest = _current_manifest()
        scenarios = manifest.get("scenarios", [])
        if not isinstance(scenarios, list):
            scenarios = []
            manifest["scenarios"] = scenarios
        cleaned = [dict(item) for item in scenarios if isinstance(item, dict)]
        manifest["scenarios"] = cleaned
        return cleaned

    def _current_runs() -> list[dict[str, Any]]:
        return list(state.get("runs") or [])

    def _selected_scenario_index() -> int:
        scenarios = _scenario_list()
        if not scenarios:
            state["selected_scenario_index"] = 0
            return 0
        try:
            index = int(state.get("selected_scenario_index", 0) or 0)
        except Exception:
            index = 0
        index = max(0, min(index, len(scenarios) - 1))
        state["selected_scenario_index"] = index
        return index

    def _selected_scenario() -> dict[str, Any] | None:
        scenarios = _scenario_list()
        if not scenarios:
            return None
        return scenarios[_selected_scenario_index()]

    def _trial_detail_key(run_root: str, scenario_id: str, method: str, trial_index: int) -> str:
        return "|".join(
            [
                str(run_root or "").strip(),
                str(scenario_id or "").strip(),
                OfflineStudyRunner.normalize_method_name(method),
                str(int(trial_index or 0)),
            ]
        )

    def _trial_detail_cache() -> dict[str, Any]:
        cache = state.get("trial_details_cache")
        if not isinstance(cache, dict):
            cache = {}
            state["trial_details_cache"] = cache
        return cache

    def _set_selected_scenario_index(value: int) -> None:
        state["selected_scenario_index"] = max(0, int(value or 0))
        quick_editor_card.refresh()
        scenarios_card.refresh()

    def _robot_catalog() -> list[dict[str, Any]]:
        rows = _current_context().get("robot_catalog", [])
        return [dict(row) for row in rows] if isinstance(rows, list) else []

    def _product_catalog() -> list[dict[str, Any]]:
        rows = _current_context().get("product_catalog", [])
        return [dict(row) for row in rows] if isinstance(rows, list) else []

    def _product_lookup() -> dict[str, dict[str, Any]]:
        return {
            str(row.get("path") or ""): row
            for row in _product_catalog()
            if str(row.get("path") or "").strip()
        }

    def _requirement_files_by_product() -> dict[str, list[str]]:
        payload = _current_context().get("requirement_files_by_product", {})
        if not isinstance(payload, dict):
            return {}
        return {
            str(key): [str(item) for item in value if str(item).strip()]
            for key, value in payload.items()
            if isinstance(value, list)
        }

    def _verified_safety_catalog() -> list[dict[str, Any]]:
        rows = _current_context().get("verified_safety_catalog", [])
        return [dict(row) for row in rows] if isinstance(rows, list) else []

    def _verified_rules_by_safety_file() -> dict[str, list[dict[str, Any]]]:
        payload = _current_context().get("verified_rules_by_safety_file", {})
        if not isinstance(payload, dict):
            return {}
        return {
            str(key): [dict(rule) for rule in value if isinstance(rule, dict)]
            for key, value in payload.items()
            if isinstance(value, list)
        }

    def _part_catalog_by_product() -> dict[str, list[str]]:
        payload = _current_context().get("part_catalog_by_product", {})
        if not isinstance(payload, dict):
            return {}
        return {
            str(key): [str(item) for item in value if str(item).strip()]
            for key, value in payload.items()
            if isinstance(value, list)
        }

    def _source_locations() -> list[str]:
        rows = _current_context().get("source_locations", [])
        return [str(item) for item in rows if str(item).strip()] if isinstance(rows, list) else []

    def _method_display(method: str) -> str:
        normalized = OfflineStudyRunner.normalize_method_name(method)
        return dict(_METHOD_OPTIONS).get(str(normalized), str(normalized))

    def _method_enabled(method: str) -> bool:
        defaults = dict(_current_manifest().get("defaults", {}) or {})
        methods = defaults.get("methods", [])
        return str(method) in methods if isinstance(methods, list) else False

    def _default_product_path() -> str:
        catalog = _product_catalog()
        return str(catalog[0].get("path") or "") if catalog else ""

    def _default_requirement_file(product_init_file: str) -> str:
        options = _requirement_files_by_product().get(product_init_file, [])
        return options[0] if options else ""

    def _default_safety_file() -> str:
        catalog = _verified_safety_catalog()
        return str(catalog[0].get("path") or "") if catalog else ""

    def _default_rule_ids(safety_requirement_file: str) -> list[str]:
        return [
            str(rule.get("id") or "")
            for rule in _verified_rules_by_safety_file().get(safety_requirement_file, [])
            if str(rule.get("id") or "").strip()
        ]

    def _default_robot_keys() -> list[str]:
        return [
            str(row.get("key") or "")
            for row in _robot_catalog()[:2]
            if str(row.get("key") or "").strip()
        ]

    def _available_parts(product_init_file: str) -> list[str]:
        return list(_part_catalog_by_product().get(product_init_file, []))

    def _available_requirement_files(product_init_file: str) -> list[str]:
        return list(_requirement_files_by_product().get(product_init_file, []))

    def _available_rules(safety_requirement_file: str) -> list[dict[str, Any]]:
        return [dict(rule) for rule in _verified_rules_by_safety_file().get(safety_requirement_file, [])]

    def _available_sources_for(resource_keys: list[str]) -> list[str]:
        lookup = {
            str(row.get("key") or ""): row
            for row in _robot_catalog()
            if str(row.get("key") or "").strip()
        }
        union: set[str] = set()
        for key in resource_keys:
            union.update(str(loc) for loc in lookup.get(str(key), {}).get("source_locations", []))
        ordered = [source for source in _source_locations() if source in union]
        return ordered or _source_locations()

    def _next_scenario_id() -> str:
        existing = {str(scenario.get("id") or "").strip() for scenario in _scenario_list()}
        idx = 1
        while True:
            candidate = f"S{idx}"
            if candidate not in existing:
                return candidate
            idx += 1

    def _sync_advanced_json() -> None:
        state["advanced_json"] = json.dumps(_current_manifest(), indent=2)
        if advanced_editor is not None:
            advanced_editor.value = state["advanced_json"]

    def _set_asset_refresh_notice(message: str = "") -> None:
        text = str(message or "").strip()
        state["asset_refresh_notice"] = text
        if asset_notice_label is not None:
            asset_notice_label.text = text
            asset_notice_label.classes(
                replace="text-sm text-amber-700" if text else "text-sm text-slate-500"
            )

    def _mark_dirty() -> None:
        state["dirty"] = True
        dirty_label.text = "Unsaved experiment changes."
        _sync_advanced_json()
        quick_editor_card.refresh()

    def _clear_dirty() -> None:
        state["dirty"] = False
        dirty_label.text = "All experiment changes saved."
        _sync_advanced_json()
        quick_editor_card.refresh()

    def _normalize_scenario_in_place(scenario: dict[str, Any]) -> None:
        defaults = dict(_current_manifest().get("defaults", {}) or {})
        product_init_file = str(scenario.get("product_init_file", "") or "").strip() or _default_product_path()
        if product_init_file not in _product_lookup() and _product_lookup():
            product_init_file = _default_product_path()
        scenario["product_init_file"] = product_init_file
        available_parts = set(_available_parts(product_init_file))

        requirement_options = _available_requirement_files(product_init_file)
        requirement_file = str(scenario.get("product_requirement_file", "") or "").strip()
        if requirement_file not in requirement_options:
            requirement_file = requirement_options[0] if requirement_options else ""
        scenario["product_requirement_file"] = requirement_file
        derived_requirement_layout = _requirement_layout(
            requirement_file,
            valid_parts=available_parts,
        )

        safety_requirement_file = (
            str(scenario.get("safety_requirement_file", "") or "").strip() or _default_safety_file()
        )
        if safety_requirement_file not in _verified_rules_by_safety_file() and _verified_rules_by_safety_file():
            safety_requirement_file = _default_safety_file()
        scenario["safety_requirement_file"] = safety_requirement_file

        available_rule_ids = [
            str(rule.get("id") or "")
            for rule in _available_rules(safety_requirement_file)
            if str(rule.get("id") or "").strip()
        ]
        enabled_rule_ids = [
            rid
            for rid in OfflineStudyRunner._as_str_list(scenario.get("enabled_safety_rule_ids", []))
            if rid in set(available_rule_ids)
        ]
        scenario["enabled_safety_rule_ids"] = enabled_rule_ids or list(available_rule_ids)

        scenario["id"] = str(scenario.get("id", "")).strip() or _next_scenario_id()
        scenario["trials"] = max(1, int(defaults.get("trials_per_method", 10) or 10))
        scenario["resource_keys"] = [
            str(key).strip()
            for key in scenario.get("resource_keys", [])
            if str(key).strip()
        ]
        if bool(derived_requirement_layout.get("derived", False)):
            scenario["parts"] = list(derived_requirement_layout.get("parts", []))
            scenario["part_order"] = list(derived_requirement_layout.get("part_order", []))
            scenario["part_sources"] = dict(derived_requirement_layout.get("part_sources", {}))
        else:
            scenario["parts"] = [
                part for part in OfflineStudyRunner._as_str_list(scenario.get("parts", []))
                if part in available_parts
            ]
            scenario["part_order"] = [
                part
                for part in OfflineStudyRunner._as_str_list(scenario.get("part_order", scenario.get("parts", [])))
                if part in set(scenario["parts"])
            ]
            for part in scenario["parts"]:
                if part not in scenario["part_order"]:
                    scenario["part_order"].append(part)

            scenario["part_sources"] = {
                str(key).strip(): str(value).strip()
                for key, value in dict(scenario.get("part_sources", {}) or {}).items()
                if str(key).strip() in set(scenario["parts"])
            }
            available_sources = _available_sources_for(list(scenario.get("resource_keys", [])))
            preferred_source = next((item for item in available_sources if item != "assembly_board-v1"), "")
            default_source = preferred_source or (available_sources[0] if available_sources else "")
            for part in scenario["parts"]:
                source = str(scenario["part_sources"].get(part, "") or "").strip()
                if source not in available_sources:
                    source = default_source
                scenario["part_sources"][part] = source

        scenario["notes"] = str(scenario.get("notes", "") or "")

    def _ensure_defaults() -> None:
        manifest = _current_manifest()
        defaults = dict(manifest.get("defaults", {}) or {})
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
        defaults["methods"] = OfflineStudyRunner.normalize_methods(
            defaults.get("methods", [method for method, _ in _METHOD_OPTIONS])
        )
        manifest["defaults"] = defaults
        if not _scenario_list() and _product_catalog():
            _scenario_list().append(_new_scenario_payload())
            state["selected_scenario_index"] = 0
        for scenario in _scenario_list():
            _normalize_scenario_in_place(scenario)

    def _new_scenario_payload() -> dict[str, Any]:
        product_init_file = _default_product_path()
        safety_requirement_file = _default_safety_file()
        scenario = {
            "id": _next_scenario_id(),
            "product_init_file": product_init_file,
            "product_requirement_file": _default_requirement_file(product_init_file),
            "safety_requirement_file": safety_requirement_file,
            "enabled_safety_rule_ids": _default_rule_ids(safety_requirement_file),
            "resource_keys": _default_robot_keys(),
            "parts": [],
            "part_order": [],
            "part_sources": {},
            "notes": "",
        }
        _normalize_scenario_in_place(scenario)
        return scenario

    def _scenario_preview(scenario: dict[str, Any]) -> tuple[str, str]:
        runner = OfflineStudyRunner(
            state["manifest_file"] or "writing/experiments/offline_planning_experiment_pack.json"
        )
        preview = {
            "product_requirement_file": str(scenario.get("product_requirement_file", "") or ""),
            "part_order": list(scenario.get("part_order", [])),
            "parts": list(scenario.get("parts", [])),
            "part_sources": dict(scenario.get("part_sources", {})),
            "safety_requirement_file": str(scenario.get("safety_requirement_file", "") or ""),
            "enabled_safety_rule_ids": list(scenario.get("enabled_safety_rule_ids", [])),
        }
        generated_requirements = runner.render_requirements_text(preview)
        generated_safety = runner.render_safety_text(
            preview,
            verified_rules_by_safety_file=_verified_rules_by_safety_file(),
        )
        return (
            _read_preview_text(
                str(scenario.get("product_requirement_file", "") or ""),
                fallback=generated_requirements,
            ),
            _read_preview_text(
                str(scenario.get("safety_requirement_file", "") or ""),
                fallback=generated_safety,
            ),
        )

    def _render_trial_detail(detail: dict[str, Any]) -> None:
        plan_json = detail.get("plan_json", {})
        global_fsa_json = detail.get("global_fsa_json", {})
        nodes = list(plan_json.get("nodes", [])) if isinstance(plan_json, dict) else []
        validation_summary = detail.get("validation_summary", {})
        task_states: dict[str, str] = {}
        ordered_task_ids: list[str] = []
        node_lookup: dict[str, dict[str, Any]] = {}
        for node in nodes:
            if not isinstance(node, dict):
                continue
            task_id = str(node.get("id") or node.get("task_id") or "").strip()
            if not task_id:
                continue
            ordered_task_ids.append(task_id)
            node_lookup[task_id] = node
            task_states[task_id] = str(node.get("status", "pending") or "pending")
        if not isinstance(validation_summary, dict):
            validation_summary = {}
        ui.label("Task DAG").classes("text-base font-semibold mt-4")
        if nodes:
            overlay = build_fsa_dag_overlay(
                nodes,
                global_fsa_json if isinstance(global_fsa_json, dict) else {},
            )
            ui.mermaid(nodes_to_mermaid(nodes, task_states, overlay=overlay)).classes("w-full")
            startable = sorted(
                task_id
                for task_id, entry in overlay.items()
                if str(entry.get("label_suffix", "")).strip() == "FSA-startable"
            )
            resource_blocked = sorted(
                f"{task_id} ({str(entry.get('label_suffix', '')).strip()})"
                for task_id, entry in overlay.items()
                if str(entry.get("label_suffix", "")).strip().startswith("resource-blocked:")
            )
            if startable:
                ui.label(
                    "FSA-startable from x0: " + ", ".join(startable)
                ).classes("text-sm text-emerald-700 mt-2")
            if resource_blocked:
                ui.label(
                    "DAG roots blocked by same-resource execution order: "
                    + ", ".join(resource_blocked)
                ).classes("text-sm text-amber-700 mt-1")
        else:
            ui.label("No task DAG available for this trial.").classes("text-sm text-slate-500 italic mt-1")
        with ui.row().classes("w-full gap-4 items-start no-wrap mt-4"):
            with ui.column().classes("w-1/2 gap-2"):
                ui.label("Task States").classes("text-base font-semibold")
                task_table = ui.table(
                    columns=[
                        {"name": "task_id", "label": "Task ID", "field": "task_id", "sortable": True},
                        {"name": "status", "label": "Status", "field": "status", "sortable": True},
                    ],
                    rows=[
                        {"task_id": task_id, "status": task_states.get(task_id, "pending")}
                        for task_id in ordered_task_ids
                    ],
                    row_key="task_id",
                    selection="multiple",
                ).classes("w-full")
            with ui.column().classes("w-1/2 gap-2"):
                detail_label = ui.label("Node Detail").classes("text-base font-semibold")
                detail_table = ui.table(
                    columns=[
                        {"name": "key", "label": "Field", "field": "key"},
                        {"name": "value", "label": "Value", "field": "value"},
                    ],
                    rows=[],
                    row_key="row_id",
                ).classes("w-full")

                def _set_node_details(task_ids: list[str]) -> None:
                    if not task_ids:
                        detail_label.text = "Node Detail"
                        detail_table.rows = []
                        return
                    all_rows: list[dict[str, Any]] = []
                    row_idx = 0
                    for task_id in task_ids:
                        node = node_lookup.get(task_id)
                        if not isinstance(node, dict):
                            continue
                        if all_rows:
                            all_rows.append({"row_id": f"sep_{row_idx}", "key": "-----", "value": "-----"})
                            row_idx += 1
                        for key, value in node.items():
                            display = json.dumps(value, default=str) if isinstance(value, (dict, list)) else str(value)
                            all_rows.append({"row_id": f"{task_id}_{key}", "key": key, "value": display})
                            row_idx += 1
                    detail_label.text = "Node Detail - " + ", ".join(task_ids)
                    detail_table.rows = all_rows

                def _on_task_select(event: Any) -> None:
                    selected_ids: list[str] = []
                    for row in (event.selection or []):
                        task_id = str(dict(row).get("task_id", "") or "").strip()
                        if task_id:
                            selected_ids.append(task_id)
                    _set_node_details(selected_ids)

                task_table.on_select(_on_task_select)

                if ordered_task_ids:
                    preferred_task = next(
                        (
                            task_id
                            for task_id in ordered_task_ids
                            if any(
                                marker in task_states.get(task_id, "").lower()
                                for marker in ("running", "dispatched", "accepted", "completed", "failed")
                            )
                        ),
                        ordered_task_ids[0],
                    )
                    _set_node_details([preferred_task])
        with ui.expansion("Requirement and Safety Text", icon="description", value=False).classes("w-full mt-4"):
            with ui.row().classes("w-full gap-4 items-start no-wrap"):
                ui.textarea(
                    label="Requirement Text",
                    value=str(detail.get("requirements_text", "") or ""),
                ).classes("w-1/2 font-mono").props("outlined readonly autogrow")
                ui.textarea(
                    label="Safety Text",
                    value=str(detail.get("safety_text", "") or ""),
                ).classes("w-1/2 font-mono").props("outlined readonly autogrow")
        repair_rows = _repair_history_rows(
            validation_summary,
            detail.get("trial_record", {}) if isinstance(detail.get("trial_record"), dict) else {},
        )
        with ui.expansion("Repair Timeline", icon="timeline", value=bool(repair_rows)).classes("w-full mt-3"):
            if repair_rows:
                ui.table(
                    columns=[
                        {"name": "step", "label": "Step", "field": "step"},
                        {"name": "result", "label": "Result", "field": "result"},
                        {"name": "safety_rules", "label": "Safety Rules", "field": "safety_rules"},
                        {"name": "violated_rules", "label": "Violated Rules", "field": "violated_rules"},
                        {"name": "witnesses", "label": "Witnesses", "field": "witnesses"},
                        {"name": "changed_tasks", "label": "Changed Tasks", "field": "changed_tasks"},
                        {"name": "error", "label": "Error", "field": "error"},
                    ],
                    rows=repair_rows,
                    row_key="row_id",
                ).classes("w-full")
            else:
                ui.label("No repair timeline recorded for this run.").classes(
                    "text-sm text-slate-500 italic"
                )
        with ui.expansion("Validation Summary", icon="rule", value=False).classes("w-full mt-3"):
            ui.code(json.dumps(validation_summary, indent=2), language="json").classes("w-full")
        with ui.expansion("Trial Metadata", icon="info", value=False).classes("w-full mt-3"):
            ui.label(f"Raw trial payload: {detail.get('raw_path', '')}").classes("text-xs text-slate-500")

    def _set_context(payload: dict[str, Any]) -> None:
        state["context"] = payload
        state["advanced_json"] = json.dumps(payload.get("manifest", {}), indent=2)
        state["asset_catalog_signature"] = _asset_catalog_signature(payload)
        state["pending_asset_catalog_signature"] = ""
        _set_asset_refresh_notice("")
        runs = bridge.list_experiment_runs(state["manifest_file"]) if state["manifest_file"] else []
        state["runs"] = runs
        available_roots = {str(row.get("run_root") or "") for row in runs}
        if not state["selected_run_root"] or state["selected_run_root"] not in available_roots:
            state["selected_run_root"] = str(runs[0]["run_root"]) if runs else ""
        _ensure_defaults()
        _selected_scenario_index()

    async def _reload_context() -> None:
        manifest_path = str(state["manifest_file"] or "").strip()
        if not manifest_path:
            return
        payload = await asyncio.to_thread(bridge.load_experiment_editor_context, manifest_path)
        _set_context(payload)
        state["analysis"] = None
        state["trial_details_cache"] = {}
        _clear_dirty()
        settings_card.refresh()
        quick_editor_card.refresh()
        scenarios_card.refresh()
        analysis_card.refresh()

    async def _poll_asset_catalogs() -> None:
        manifest_path = str(state["manifest_file"] or "").strip()
        if (
            not manifest_path
            or run_busy["value"]
            or analyze_busy["value"]
            or bool(state.get("asset_refresh_inflight", False))
        ):
            return
        state["asset_refresh_inflight"] = True
        try:
            payload = await asyncio.to_thread(bridge.load_experiment_editor_context, manifest_path)
        except Exception:
            return
        finally:
            state["asset_refresh_inflight"] = False

        current_signature = str(state.get("asset_catalog_signature", "") or "")
        next_signature = _asset_catalog_signature(payload)
        if next_signature == current_signature:
            return
        if state.get("dirty", False):
            if str(state.get("pending_asset_catalog_signature", "") or "") != next_signature:
                state["pending_asset_catalog_signature"] = next_signature
                _set_asset_refresh_notice(
                    "New product or verified safety assets are available. Save or reload study to pick them up."
                )
            return

        _set_context(payload)
        settings_card.refresh()
        quick_editor_card.refresh()
        scenarios_card.refresh()

    async def _save_study(*, notify: bool = True) -> None:
        manifest_path = str(state["manifest_file"] or "").strip()
        if not manifest_path:
            ui.notify("Select a study manifest first.", type="warning")
            return
        try:
            payload = await asyncio.to_thread(
                bridge.save_experiment_study,
                manifest_path,
                _current_manifest(),
            )
            _set_context(payload)
            _clear_dirty()
            settings_card.refresh()
            quick_editor_card.refresh()
            scenarios_card.refresh()
            analysis_card.refresh()
            if notify:
                ui.notify("Experiment study saved.", type="positive")
        except Exception as exc:
            ui.notify(f"Failed to save study: {exc}", type="negative")

    async def _save_advanced_json() -> None:
        manifest_path = str(state["manifest_file"] or "").strip()
        if not manifest_path:
            ui.notify("Select a study manifest first.", type="warning")
            return
        try:
            payload = json.loads(advanced_editor.value or "{}")
        except json.JSONDecodeError as exc:
            ui.notify(f"Invalid JSON: {exc}", type="negative")
            return
        try:
            saved = await asyncio.to_thread(bridge.save_experiment_study, manifest_path, payload)
            _set_context(saved)
            _clear_dirty()
            settings_card.refresh()
            quick_editor_card.refresh()
            scenarios_card.refresh()
            analysis_card.refresh()
            ui.notify("Advanced JSON saved.", type="positive")
        except Exception as exc:
            ui.notify(f"Failed to save advanced JSON: {exc}", type="negative")

    def _set_defaults_trials(value: int) -> None:
        manifest = _current_manifest()
        defaults = dict(manifest.get("defaults", {}) or {})
        defaults["trials_per_method"] = max(1, int(value or 1))
        manifest["defaults"] = defaults
        for scenario in _scenario_list():
            scenario["trials"] = defaults["trials_per_method"]
        _mark_dirty()
        settings_card.refresh()
        quick_editor_card.refresh()
        scenarios_card.refresh()

    def _set_defaults_auto_replans(value: int) -> None:
        manifest = _current_manifest()
        defaults = dict(manifest.get("defaults", {}) or {})
        defaults["auto_replan_max_attempts"] = max(0, min(int(value or 0), 10))
        manifest["defaults"] = defaults
        _mark_dirty()
        settings_card.refresh()

    def _toggle_method(method: str, enabled: bool) -> None:
        manifest = _current_manifest()
        defaults = dict(manifest.get("defaults", {}) or {})
        methods = OfflineStudyRunner.normalize_methods(
            defaults.get("methods", [method for method, _ in _METHOD_OPTIONS])
        )
        if enabled and method not in methods:
            methods.append(method)
        if not enabled and method in methods:
            methods = [item for item in methods if item != method]
        defaults["methods"] = methods or ["llm_nl_safety"]
        manifest["defaults"] = defaults
        _mark_dirty()
        settings_card.refresh()

    def _add_scenario() -> None:
        _scenario_list().append(_new_scenario_payload())
        state["selected_scenario_index"] = len(_scenario_list()) - 1
        _mark_dirty()
        scenarios_card.refresh()

    def _delete_scenario(index: int) -> None:
        scenarios = _scenario_list()
        if 0 <= index < len(scenarios):
            scenarios.pop(index)
            _selected_scenario_index()
            _mark_dirty()
            scenarios_card.refresh()

    def _duplicate_scenario(index: int) -> None:
        scenarios = _scenario_list()
        if 0 <= index < len(scenarios):
            duplicate = json.loads(json.dumps(scenarios[index]))
            duplicate["id"] = _next_scenario_id()
            _normalize_scenario_in_place(duplicate)
            scenarios.insert(index + 1, duplicate)
            state["selected_scenario_index"] = index + 1
            _mark_dirty()
            scenarios_card.refresh()

    def _move_scenario(index: int, delta: int) -> None:
        scenarios = _scenario_list()
        other = index + delta
        if 0 <= index < len(scenarios) and 0 <= other < len(scenarios):
            scenarios[index], scenarios[other] = scenarios[other], scenarios[index]
            if _selected_scenario_index() == index:
                state["selected_scenario_index"] = other
            elif _selected_scenario_index() == other:
                state["selected_scenario_index"] = index
            _mark_dirty()
            scenarios_card.refresh()

    def _set_scenario_id(index: int, value: str) -> None:
        scenarios = _scenario_list()
        if 0 <= index < len(scenarios):
            scenarios[index]["id"] = str(value or "").strip() or _next_scenario_id()
            _normalize_scenario_in_place(scenarios[index])
            _mark_dirty()
            scenarios_card.refresh()

    def _set_scenario_notes(index: int, value: str) -> None:
        scenarios = _scenario_list()
        if 0 <= index < len(scenarios):
            scenarios[index]["notes"] = str(value or "")
            _mark_dirty()

    def _set_scenario_product(index: int, value: str) -> None:
        scenarios = _scenario_list()
        if 0 <= index < len(scenarios):
            scenario = scenarios[index]
            scenario["product_init_file"] = str(value or "")
            scenario["parts"] = []
            scenario["part_order"] = []
            scenario["part_sources"] = {}
            _normalize_scenario_in_place(scenario)
            _mark_dirty()
            scenarios_card.refresh()

    def _set_scenario_requirement_file(index: int, value: str) -> None:
        scenarios = _scenario_list()
        if 0 <= index < len(scenarios):
            scenarios[index]["product_requirement_file"] = str(value or "")
            _normalize_scenario_in_place(scenarios[index])
            _mark_dirty()
            scenarios_card.refresh()

    def _set_scenario_safety_file(index: int, value: str) -> None:
        scenarios = _scenario_list()
        if 0 <= index < len(scenarios):
            scenario = scenarios[index]
            scenario["safety_requirement_file"] = str(value or "")
            scenario["enabled_safety_rule_ids"] = []
            _normalize_scenario_in_place(scenario)
            _mark_dirty()
            scenarios_card.refresh()

    def _toggle_scenario_rule(index: int, rule_id: str, enabled: bool) -> None:
        scenarios = _scenario_list()
        if not (0 <= index < len(scenarios)):
            return
        scenario = scenarios[index]
        ids = [
            rid
            for rid in OfflineStudyRunner._as_str_list(scenario.get("enabled_safety_rule_ids", []))
            if str(rid).strip()
        ]
        if enabled and rule_id not in ids:
            ids.append(rule_id)
        if not enabled and rule_id in ids:
            ids = [rid for rid in ids if rid != rule_id]
        scenario["enabled_safety_rule_ids"] = ids
        _normalize_scenario_in_place(scenario)
        _mark_dirty()
        scenarios_card.refresh()

    def _toggle_robot(index: int, robot_key: str, enabled: bool) -> None:
        scenarios = _scenario_list()
        if not (0 <= index < len(scenarios)):
            return
        scenario = scenarios[index]
        keys = [str(key) for key in scenario.get("resource_keys", []) if str(key).strip()]
        if enabled and robot_key not in keys:
            keys.append(robot_key)
        if not enabled and robot_key in keys:
            keys = [key for key in keys if key != robot_key]
        scenario["resource_keys"] = keys
        _normalize_scenario_in_place(scenario)
        _mark_dirty()
        scenarios_card.refresh()

    def _add_part(index: int, part: str) -> None:
        scenarios = _scenario_list()
        if not (0 <= index < len(scenarios)):
            return
        scenario = scenarios[index]
        token = str(part or "").strip()
        if token and token not in scenario.get("parts", []):
            scenario.setdefault("parts", []).append(token)
            scenario.setdefault("part_order", []).append(token)
            _normalize_scenario_in_place(scenario)
            _mark_dirty()
            scenarios_card.refresh()

    def _remove_part(index: int, part_index: int) -> None:
        scenarios = _scenario_list()
        if not (0 <= index < len(scenarios)):
            return
        scenario = scenarios[index]
        parts = list(scenario.get("part_order", scenario.get("parts", [])))
        if 0 <= part_index < len(parts):
            part = parts.pop(part_index)
            scenario["parts"] = [item for item in scenario.get("parts", []) if item != part]
            scenario["part_order"] = [item for item in parts if item in set(scenario["parts"])]
            scenario.setdefault("part_sources", {}).pop(part, None)
            _normalize_scenario_in_place(scenario)
            _mark_dirty()
            scenarios_card.refresh()

    def _move_part(index: int, part_index: int, delta: int) -> None:
        scenarios = _scenario_list()
        if not (0 <= index < len(scenarios)):
            return
        scenario = scenarios[index]
        parts = list(scenario.get("part_order", scenario.get("parts", [])))
        other = part_index + delta
        if 0 <= part_index < len(parts) and 0 <= other < len(parts):
            parts[part_index], parts[other] = parts[other], parts[part_index]
            scenario["part_order"] = parts
            _normalize_scenario_in_place(scenario)
            _mark_dirty()
            scenarios_card.refresh()

    def _set_part_source(index: int, part: str, source: str) -> None:
        scenarios = _scenario_list()
        if not (0 <= index < len(scenarios)):
            return
        scenario = scenarios[index]
        scenario.setdefault("part_sources", {})[str(part)] = str(source or "")
        _normalize_scenario_in_place(scenario)
        _mark_dirty()
        scenarios_card.refresh()

    async def _run_study() -> None:
        manifest_path = str(state["manifest_file"] or "").strip()
        if not manifest_path:
            ui.notify("Select a study manifest first.", type="warning")
            return
        selected_scenario = _selected_scenario()
        selected_scenario_id = str((selected_scenario or {}).get("id") or "").strip()
        if not selected_scenario_id:
            ui.notify("Select a scenario first.", type="warning")
            return
        run_busy["value"] = True
        run_btn.set_enabled(False)
        analyze_btn.set_enabled(False)
        run_status.text = f"Running scenario {selected_scenario_id}..."
        analysis_card.refresh()
        try:
            if state["dirty"]:
                await _save_study(notify=False)
            result = await asyncio.to_thread(
                bridge.run_experiment_study,
                manifest_path,
                scenario_id=selected_scenario_id,
            )
            state["runs"] = bridge.list_experiment_runs(manifest_path)
            state["selected_run_root"] = str(result.get("run_root") or "")
            run_status.text = (
                f"Completed scenario {selected_scenario_id}: {state['selected_run_root']}"
            )
            state["analysis"] = await asyncio.to_thread(
                bridge.analyze_experiment_run,
                manifest_path,
                run_root=state["selected_run_root"],
            )
            state["trial_details_cache"] = {}
            analysis_card.refresh()
            ui.notify("Experiment study finished.", type="positive")
        except Exception as exc:
            run_status.text = f"Experiment study failed: {exc}"
            ui.notify(f"Experiment study failed: {exc}", type="negative")
        finally:
            run_busy["value"] = False
            run_btn.set_enabled(True)
            analyze_btn.set_enabled(True)
            analysis_card.refresh()

    async def _analyze_selected_run() -> None:
        manifest_path = str(state["manifest_file"] or "").strip()
        run_root = str(state["selected_run_root"] or "").strip()
        if not manifest_path:
            ui.notify("Select a study manifest first.", type="warning")
            return
        analyze_busy["value"] = True
        run_btn.set_enabled(False)
        analyze_btn.set_enabled(False)
        analysis_card.refresh()
        try:
            state["analysis"] = await asyncio.to_thread(
                bridge.analyze_experiment_run,
                manifest_path,
                run_root=run_root,
            )
            state["trial_details_cache"] = {}
            analysis_card.refresh()
            ui.notify("Experiment analysis loaded.", type="positive")
        except Exception as exc:
            ui.notify(f"Failed to analyze run: {exc}", type="negative")
        finally:
            analyze_busy["value"] = False
            run_btn.set_enabled(True)
            analyze_btn.set_enabled(True)
            analysis_card.refresh()

    async def _load_trial_detail(scenario_id: str, method: str, trial_index: int) -> None:
        manifest_path = str(state["manifest_file"] or "").strip()
        run_root = str(state["selected_run_root"] or "").strip()
        if not manifest_path or not run_root:
            return
        try:
            detail = await asyncio.to_thread(
                bridge.get_experiment_trial_details,
                manifest_path,
                run_root=run_root,
                scenario_id=scenario_id,
                method=method,
                trial_index=trial_index,
            )
            _trial_detail_cache()[
                _trial_detail_key(run_root, scenario_id, method, trial_index)
            ] = detail
            analysis_card.refresh()
        except Exception as exc:
            ui.notify(f"Failed to load trial: {exc}", type="negative")

    with ui.row().classes("w-full px-6 gap-6 items-start no-wrap"):
        with ui.column().classes("flex-1 gap-6 min-w-0"):
            with ui.card().classes("w-full"):
                ui.label("Study Settings").classes("text-lg font-semibold mb-2")
                dirty_label = ui.label("").classes("text-sm text-slate-700")
                asset_notice_label = ui.label("").classes("text-sm text-slate-500")
                run_status = ui.label("No experiment run yet.").classes("text-sm text-slate-700")
                with ui.row().classes("w-full gap-3 items-end flex-wrap"):
                    manifest_select = ui.select(
                        {path: Path(path).name for path in manifest_files},
                        value=selected_manifest if selected_manifest else None,
                        label="Study Manifest",
                    ).classes("w-96")
                    run_btn = ui.button("Run Selected Scenario", icon="science", on_click=_run_study).props(
                        "color=primary"
                    )
                    analyze_btn = ui.button("Analyze Results", icon="analytics", on_click=_analyze_selected_run).props(
                        "flat color=secondary"
                    )
                    save_btn = ui.button("Save Study", icon="save").props("flat color=green")
                    reload_btn = ui.button("Reload Study", icon="refresh").props("flat")
                ui.label(
                    "Run Selected Scenario executes only the scenario currently chosen in Scenario Controls."
                ).classes("text-xs text-slate-500 mt-1")

                @ui.refreshable
                def settings_card() -> None:
                    manifest = _current_manifest()
                    defaults = dict(manifest.get("defaults", {}) or {})
                    ui.label(
                        "Experiments orchestrates existing Product, Safety, and Resource assets. "
                        "Author the assets on those pages, then select them here."
                    ).classes("text-sm text-slate-600")
                    with ui.row().classes("w-full gap-4 items-start flex-wrap mt-2"):
                        ui.number(
                            "Trials per Method (All Scenarios)",
                            value=int(defaults.get("trials_per_method", 10) or 10),
                            min=1,
                            step=1,
                            on_change=lambda e: _set_defaults_trials(int(e.value or 1)),
                        ).classes("w-48")
                        ui.number(
                            "Auto-Replan Max Attempts",
                            value=int(
                                defaults.get(
                                    "auto_replan_max_attempts",
                                    _DEFAULT_AUTO_REPLAN_MAX_ATTEMPTS,
                                )
                                or _DEFAULT_AUTO_REPLAN_MAX_ATTEMPTS
                            ),
                            min=0,
                            max=10,
                            step=1,
                            on_change=lambda e: _set_defaults_auto_replans(int(e.value or 0)),
                        ).classes("w-48")
                        with ui.column().classes("gap-2"):
                            ui.label("Methods").classes("text-sm font-medium")
                            for method, label in _METHOD_OPTIONS:
                                ui.checkbox(
                                    label,
                                    value=_method_enabled(method),
                                    on_change=lambda e, method_name=method: _toggle_method(
                                        method_name,
                                        bool(e.value),
                                    ),
                                )

                @ui.refreshable
                def quick_editor_card() -> None:
                    with ui.card().classes("w-full"):
                        ui.label("Scenario Controls").classes("text-lg font-semibold mb-2")
                        ui.label(
                            "Use these dropdowns to edit the study manifest through the UI. "
                            "No direct JSON editing is required."
                        ).classes("text-sm text-slate-600")

                        product_catalog = _product_catalog()
                        safety_catalog = _verified_safety_catalog()
                        scenarios = _scenario_list()

                        if not product_catalog:
                            ui.label(
                                "No product manifests found. Configure a product on the Products page first."
                            ).classes("text-red-700 mt-2")
                            return
                        if not scenarios:
                            ui.label(
                                "No scenarios exist yet. Click 'Add Scenario' below and the editor will populate."
                            ).classes("text-amber-700 mt-2")
                            return

                        selected_index = _selected_scenario_index()
                        selected_scenario = _selected_scenario()
                        if selected_scenario is None:
                            ui.label("No scenario selected.").classes("text-amber-700 mt-2")
                            return

                        _normalize_scenario_in_place(selected_scenario)
                        scenario_options = {
                            str(idx): str(item.get("id", f"S{idx + 1}"))
                            for idx, item in enumerate(scenarios)
                        }

                        with ui.row().classes("w-full gap-4 items-start flex-wrap mt-2"):
                            ui.select(
                                scenario_options,
                                value=str(selected_index),
                                label="Scenario",
                                on_change=lambda e: _set_selected_scenario_index(int(str(e.value or "0"))),
                            ).classes("w-40")
                            ui.input(
                                label="Scenario ID",
                                value=str(selected_scenario.get("id", "")),
                                on_change=lambda e, idx=selected_index: _set_scenario_id(idx, str(e.value or "")),
                            ).classes("w-40")

                        with ui.row().classes("w-full gap-4 items-start flex-wrap mt-2"):
                            ui.select(
                                {row["path"]: row["name"] for row in product_catalog},
                                value=str(selected_scenario.get("product_init_file", "")),
                                label="Product",
                                on_change=lambda e, idx=selected_index: _set_scenario_product(
                                    idx,
                                    str(e.value or ""),
                                ),
                            ).classes("w-72")
                            ui.select(
                                {
                                    path: Path(path).name
                                    for path in _available_requirement_files(
                                        str(selected_scenario.get("product_init_file", ""))
                                    )
                                },
                                value=str(selected_scenario.get("product_requirement_file", "")),
                                label="Requirement File",
                                on_change=lambda e, idx=selected_index: _set_scenario_requirement_file(
                                    idx,
                                    str(e.value or ""),
                                ),
                            ).classes("w-80")
                            ui.select(
                                {row["path"]: row["name"] for row in safety_catalog},
                                value=str(selected_scenario.get("safety_requirement_file", "")),
                                label="Verified Safety File",
                                on_change=lambda e, idx=selected_index: _set_scenario_safety_file(
                                    idx,
                                    str(e.value or ""),
                                ),
                            ).classes("w-80")
                        ui.label(
                            "Requirement files update automatically from Products. Safety dropdowns include only approved Safety-page files."
                        ).classes("text-xs text-slate-500 mt-1")

                        with ui.row().classes("w-full gap-4 items-start flex-wrap mt-2"):
                            ui.label(
                                f"Selected safety rules: {len(selected_scenario.get('enabled_safety_rule_ids', []))}"
                            ).classes("text-sm text-slate-700")
                            ui.label(
                                f"Selected robots: {len(selected_scenario.get('resource_keys', []))}"
                            ).classes("text-sm text-slate-700")
                            ui.label(
                                f"Selected parts: {len(selected_scenario.get('parts', []))}"
                            ).classes("text-sm text-slate-700")

                @ui.refreshable
                def scenarios_card() -> None:
                    ui.label("Scenario Builder").classes("text-lg font-semibold mb-2")
                    ui.label(
                        "Choose an existing product, requirement file, approved safety file and rule subset, "
                        "then select robots and parts for each scenario."
                    ).classes("text-sm text-slate-600")
                    ui.label(
                        "Scenario details stay collapsed until you choose one to edit."
                    ).classes("text-xs text-slate-500")
                    ui.button("Add Scenario", icon="add", on_click=_add_scenario).props(
                        "flat color=primary"
                    ).classes("mt-2")

                    scenarios = _scenario_list()
                    if not scenarios:
                        ui.label("No scenarios configured yet.").classes("text-slate-400 italic mt-2")
                        return

                    robot_catalog = _robot_catalog()
                    product_catalog = _product_catalog()
                    safety_catalog = _verified_safety_catalog()
                    if not product_catalog:
                        ui.label("No product manifests found. Configure a product on the Products page first.").classes(
                            "text-red-700 mt-2"
                        )
                        return
                    if not safety_catalog:
                        ui.label(
                            "No verified safety files are available yet. Generate and verify a safety preview on the Safety page first."
                        ).classes("text-amber-700 mt-2")
                    ui.label("Detailed Scenario Editors").classes("text-base font-semibold mt-4")

                    for scenario_index, scenario in enumerate(scenarios):
                        _normalize_scenario_in_place(scenario)
                        preview_requirements, preview_safety = _scenario_preview(scenario)
                        title = (
                            f"{scenario.get('id', 'Scenario')} | "
                            f"{len(scenario.get('resource_keys', []))} robots | "
                            f"{len(scenario.get('parts', []))} parts | "
                            f"{len(scenario.get('enabled_safety_rule_ids', []))} safety rules"
                        )
                        with ui.expansion(
                            title,
                            icon="tune",
                            value=False,
                        ).classes("w-full mt-3"):
                            with ui.row().classes("w-full gap-3 items-end flex-wrap"):
                                ui.input(
                                    label="Scenario ID",
                                    value=str(scenario.get("id", "")),
                                    on_change=lambda e, idx=scenario_index: _set_scenario_id(idx, str(e.value or "")),
                                ).classes("w-40")
                                ui.button(
                                    "Duplicate",
                                    icon="content_copy",
                                    on_click=lambda idx=scenario_index: _duplicate_scenario(idx),
                                ).props("flat")
                                ui.button(
                                    "Up",
                                    icon="arrow_upward",
                                    on_click=lambda idx=scenario_index: _move_scenario(idx, -1),
                                ).props("flat")
                                ui.button(
                                    "Down",
                                    icon="arrow_downward",
                                    on_click=lambda idx=scenario_index: _move_scenario(idx, 1),
                                ).props("flat")
                                ui.button(
                                    "Delete",
                                    icon="delete",
                                    on_click=lambda idx=scenario_index: _delete_scenario(idx),
                                ).props("flat color=red")

                            with ui.row().classes("w-full gap-4 items-start flex-wrap mt-3"):
                                ui.select(
                                    {row["path"]: row["name"] for row in product_catalog},
                                    value=str(scenario.get("product_init_file", "")),
                                    label="Product",
                                    on_change=lambda e, idx=scenario_index: _set_scenario_product(
                                        idx,
                                        str(e.value or ""),
                                    ),
                                ).classes("w-72")
                                requirement_options = _available_requirement_files(
                                    str(scenario.get("product_init_file", ""))
                                )
                                ui.select(
                                    {path: Path(path).name for path in requirement_options},
                                    value=str(scenario.get("product_requirement_file", "")),
                                    label="Requirement File",
                                    on_change=lambda e, idx=scenario_index: _set_scenario_requirement_file(
                                        idx,
                                        str(e.value or ""),
                                    ),
                                ).classes("w-80")
                                ui.select(
                                    {row["path"]: row["name"] for row in safety_catalog},
                                    value=str(scenario.get("safety_requirement_file", "")),
                                    label="Verified Safety File",
                                    on_change=lambda e, idx=scenario_index: _set_scenario_safety_file(
                                        idx,
                                        str(e.value or ""),
                                    ),
                                ).classes("w-80")
                            ui.label(
                                "New requirement uploads appear automatically. Safety dropdowns show only approved Safety-page files."
                            ).classes("text-xs text-slate-500 mt-1")

                            ui.textarea(
                                label="Scenario Notes",
                                value=str(scenario.get("notes", "")),
                                on_change=lambda e, idx=scenario_index: _set_scenario_notes(idx, str(e.value or "")),
                            ).classes("w-full mt-3").props("outlined autogrow")

                            ui.label("Verified Safety Rules").classes("text-sm font-medium mt-3")
                            rules = _available_rules(str(scenario.get("safety_requirement_file", "")))
                            if not rules:
                                ui.label(
                                    "No verified rules are available for the selected safety file."
                                ).classes("text-slate-400 italic mt-1")
                            else:
                                for rule in rules:
                                    rule_id = str(rule.get("id") or "")
                                    with ui.card().classes("w-full mt-2 border border-slate-200"):
                                        ui.checkbox(
                                            f"{rule_id} | {str(rule.get('constraint_type', '') or 'rule')}",
                                            value=rule_id in scenario.get("enabled_safety_rule_ids", []),
                                            on_change=lambda e, idx=scenario_index, rid=rule_id: _toggle_scenario_rule(
                                                idx,
                                                rid,
                                                bool(e.value),
                                            ),
                                        )
                                        ui.label(str(rule.get("raw_text", "") or "(empty rule text)")).classes(
                                            "text-sm text-slate-700"
                                        )

                            ui.label("Robots").classes("text-sm font-medium mt-3")
                            with ui.row().classes("w-full gap-3 flex-wrap"):
                                for robot in robot_catalog:
                                    robot_key = str(robot.get("key") or "")
                                    selected = robot_key in scenario.get("resource_keys", [])
                                    card_classes = "w-56 border-2 "
                                    card_classes += "border-green-500" if selected else "border-slate-200"
                                    with ui.card().classes(card_classes):
                                        ui.checkbox(
                                            str(robot.get("name") or robot_key),
                                            value=selected,
                                            on_change=lambda e, idx=scenario_index, key=robot_key: _toggle_robot(
                                                idx,
                                                key,
                                                bool(e.value),
                                            ),
                                        )
                                        ui.label(
                                            "Reachability: " + ", ".join(robot.get("source_locations", []))
                                        ).classes("text-xs text-slate-600")

                            ui.label("Ordered Parts").classes("text-sm font-medium mt-3")
                            requirement_layout = _requirement_layout(
                                str(scenario.get("product_requirement_file", "")),
                                valid_parts=set(_available_parts(str(scenario.get("product_init_file", "")))),
                            )
                            if bool(requirement_layout.get("derived", False)):
                                ui.label(
                                    "Auto-derived from the selected requirement file. Edit the requirement file to change the parts or sources."
                                ).classes("text-xs text-slate-500 mt-1")
                            else:
                                available_parts = [
                                    part
                                    for part in _available_parts(str(scenario.get("product_init_file", "")))
                                    if part not in scenario.get("parts", [])
                                ]
                                selected_part_to_add = {"value": available_parts[0] if available_parts else ""}
                                with ui.row().classes("w-full gap-2 items-end flex-wrap"):
                                    ui.select(
                                        {part: part for part in available_parts},
                                        value=selected_part_to_add["value"] if available_parts else None,
                                        label="Add Part",
                                        on_change=lambda e, holder=selected_part_to_add: holder.__setitem__(
                                            "value",
                                            str(e.value or ""),
                                        ),
                                    ).classes("w-40")
                                    ui.button(
                                        "Add Part",
                                        icon="add",
                                        on_click=lambda idx=scenario_index, holder=selected_part_to_add: _add_part(
                                            idx,
                                            holder["value"],
                                        ),
                                    ).props("flat color=primary")

                            ordered_parts = list(scenario.get("part_order", scenario.get("parts", [])))
                            if not ordered_parts:
                                ui.label("No parts selected yet.").classes("text-slate-400 italic mt-2")
                            for part_index, part in enumerate(ordered_parts):
                                sources = _available_sources_for(list(scenario.get("resource_keys", [])))
                                if bool(requirement_layout.get("derived", False)):
                                    with ui.row().classes("w-full gap-2 items-end flex-wrap mt-2"):
                                        ui.label(f"{part_index + 1}. {part}").classes("w-28 text-sm font-medium")
                                        ui.label(
                                            "Source: "
                                            + str(scenario.get("part_sources", {}).get(part, "") or "(missing)")
                                        ).classes("text-sm text-slate-700")
                                else:
                                    with ui.row().classes("w-full gap-2 items-end flex-wrap mt-2"):
                                        ui.label(f"{part_index + 1}. {part}").classes("w-28 text-sm font-medium")
                                        ui.select(
                                            {source: source for source in sources},
                                            value=str(scenario.get("part_sources", {}).get(part, "")),
                                            label="Source",
                                            on_change=lambda e, idx=scenario_index, part_name=part: _set_part_source(
                                                idx,
                                                part_name,
                                                str(e.value or ""),
                                            ),
                                        ).classes("w-56")
                                        ui.button(
                                            "Up",
                                            icon="arrow_upward",
                                            on_click=lambda idx=scenario_index, part_idx=part_index: _move_part(
                                                idx,
                                                part_idx,
                                                -1,
                                            ),
                                        ).props("flat")
                                        ui.button(
                                            "Down",
                                            icon="arrow_downward",
                                            on_click=lambda idx=scenario_index, part_idx=part_index: _move_part(
                                                idx,
                                                part_idx,
                                                1,
                                            ),
                                        ).props("flat")
                                        ui.button(
                                            "Remove",
                                            icon="remove_circle",
                                            on_click=lambda idx=scenario_index, part_idx=part_index: _remove_part(
                                                idx,
                                                part_idx,
                                            ),
                                        ).props("flat color=red")

                            with ui.row().classes("w-full gap-4 items-start no-wrap mt-3"):
                                with ui.column().classes("w-1/2 gap-1"):
                                    ui.label(
                                        f"Requirement File: {_selected_file_name(scenario.get('product_requirement_file', ''))}"
                                    ).classes("text-xs text-slate-600")
                                    ui.textarea(
                                        label="Requirement File Preview",
                                        value=preview_requirements,
                                    ).classes("w-full font-mono").props("outlined readonly autogrow")
                                with ui.column().classes("w-1/2 gap-1"):
                                    ui.label(
                                        f"Verified Safety File: {_selected_file_name(scenario.get('safety_requirement_file', ''))}"
                                    ).classes("text-xs text-slate-600")
                                    ui.textarea(
                                        label="Verified Safety File Preview",
                                        value=preview_safety,
                                    ).classes("w-full font-mono").props("outlined readonly autogrow")

                @ui.refreshable
                def analysis_card() -> None:
                    ui.label("Analysis").classes("text-lg font-semibold mb-2")
                    runs = _current_runs()
                    run_options = {str(run.get("run_root") or ""): str(run.get("run_id") or "") for run in runs}
                    with ui.row().classes("w-full gap-3 items-end flex-wrap"):
                        ui.select(
                            run_options,
                            value=state["selected_run_root"] if state["selected_run_root"] in run_options else None,
                            label="Study Run",
                            on_change=lambda e: state.__setitem__("selected_run_root", str(e.value or "")),
                        ).classes("w-[32rem]")
                        if run_busy["value"]:
                            with ui.row().classes("items-center gap-2 text-primary"):
                                ui.spinner(size="sm")
                                ui.label("Running experiments...")
                        elif analyze_busy["value"]:
                            with ui.row().classes("items-center gap-2 text-primary"):
                                ui.spinner(size="sm")
                                ui.label("Analyzing results...")

                    analysis = state.get("analysis")
                    if not isinstance(analysis, dict):
                        ui.label("No analyzed run loaded yet.").classes("text-slate-400 italic mt-2")
                        return

                    aggregate_rows = analysis.get("aggregate_rows", [])
                    if isinstance(aggregate_rows, list) and aggregate_rows:
                        summary_rows = []
                        for row in aggregate_rows:
                            item = dict(row)
                            item["method"] = _method_display(str(item.get("method") or ""))
                            rule_rate = item.get("rule_satisfaction_rate")
                            item["rule_satisfaction_rate"] = (
                                "N/A" if rule_rate is None else f"{float(rule_rate or 0.0):.2%}"
                            )
                            item["avg_initial_violated_rules"] = _format_optional_count(
                                item.get("avg_initial_violated_rules")
                            )
                            item["repair_correction_summary"] = str(
                                item.get("repair_correction_summary") or "none"
                            )
                            item["avg_final_violated_rules"] = _format_optional_count(
                                item.get("avg_final_violated_rules")
                            )
                            item["valid_rate"] = f"{float(item.get('valid_rate', 0.0) or 0.0):.2%}"
                            item["unsafe_or_invalid_rate"] = f"{float(item.get('unsafe_or_invalid_rate', 0.0) or 0.0):.2%}"
                            item["avg_verification_time_ms"] = f"{float(item.get('avg_verification_time_ms', 0.0) or 0.0):.2f}"
                            item["avg_total_product_states_explored"] = (
                                f"{float(item.get('avg_total_product_states_explored', 0.0) or 0.0):.2f}"
                            )
                            item["avg_auto_replans_used"] = f"{float(item.get('avg_auto_replans_used', 0.0) or 0.0):.2f}"
                            item["summary_key"] = f"{item.get('scenario_id', '')}:{item.get('method', '')}"
                            summary_rows.append(item)
                        ui.table(
                            columns=[
                                {"name": "scenario_id", "label": "Scenario", "field": "scenario_id"},
                                {"name": "method", "label": "Method", "field": "method"},
                                {"name": "robots", "label": "Robots", "field": "robots"},
                                {"name": "parts", "label": "Parts", "field": "parts"},
                                {"name": "trials", "label": "Trials", "field": "trials"},
                                {"name": "rule_satisfaction_rate", "label": "Rule Satisfaction", "field": "rule_satisfaction_rate"},
                                {"name": "avg_initial_violated_rules", "label": "Initial Wrong", "field": "avg_initial_violated_rules"},
                                {"name": "repair_correction_summary", "label": "Corrected by Repair", "field": "repair_correction_summary"},
                                {"name": "avg_final_violated_rules", "label": "Final Wrong", "field": "avg_final_violated_rules"},
                                {"name": "valid_rate", "label": "Plan Valid Rate", "field": "valid_rate"},
                                {"name": "unsafe_or_invalid_rate", "label": "Plan Invalid Rate", "field": "unsafe_or_invalid_rate"},
                                {"name": "avg_verification_time_ms", "label": "Avg Verify (ms)", "field": "avg_verification_time_ms"},
                                {"name": "avg_total_product_states_explored", "label": "Avg Product States", "field": "avg_total_product_states_explored"},
                                {"name": "avg_auto_replans_used", "label": "Avg Auto-Replans", "field": "avg_auto_replans_used"},
                            ],
                            rows=summary_rows,
                            row_key="summary_key",
                        ).classes("w-full mt-2")

                    ui.textarea(
                        label="Paper Table Markdown",
                        value=str(analysis.get("markdown_preview", "") or ""),
                    ).classes("w-full font-mono mt-3").props("outlined readonly autogrow")

                    groups = analysis.get("groups", [])
                    if isinstance(groups, list):
                        trial_cache = _trial_detail_cache()
                        for group in groups:
                            scenario_id = str(group.get("scenario_id") or "")
                            method = str(group.get("method") or "")
                            trials_list = list(group.get("trials_list", []))
                            has_loaded_trial = any(
                                _trial_detail_key(
                                    str(state.get("selected_run_root") or ""),
                                    scenario_id,
                                    method,
                                    int(trial.get("trial_index", 0) or 0),
                                )
                                in trial_cache
                                for trial in trials_list
                            )
                            group_rule_rate = group.get("rule_satisfaction_rate")
                            group_rule_label = (
                                "N/A"
                                if group_rule_rate is None
                                else f"{float(group_rule_rate or 0.0):.2%}"
                            )
                            title = (
                                f"{scenario_id} | {_method_display(method)} | "
                                f"{int(group.get('trials', 0) or 0)} trials | "
                                f"{group_rule_label} rule satisfaction"
                            )
                            with ui.expansion(title, icon="analytics", value=has_loaded_trial).classes("w-full mt-3"):
                                if not trials_list:
                                    ui.label("No trial records found for this scenario/method.").classes(
                                        "text-slate-400 italic"
                                    )
                                else:
                                    for trial in trials_list:
                                        trial_index = int(trial.get("trial_index", 0) or 0)
                                        cache_key = _trial_detail_key(
                                            str(state.get("selected_run_root") or ""),
                                            scenario_id,
                                            method,
                                            trial_index,
                                        )
                                        trial_detail = trial_cache.get(cache_key)
                                        trial_title = (
                                            f"Trial {trial_index} | "
                                            f"{'valid' if bool(trial.get('ok', False)) else 'invalid'} | "
                                            f"{float(trial.get('verification_time_ms', 0.0) or 0.0):.2f} ms"
                                        )
                                        with ui.expansion(
                                            trial_title,
                                            icon="account_tree",
                                            value=bool(trial_detail),
                                        ).classes("w-full mt-2"):
                                            with ui.row().classes("w-full items-center gap-3 flex-wrap"):
                                                ui.badge(
                                                    "Valid" if bool(trial.get("ok", False)) else "Invalid"
                                                ).props(
                                                    f"color={'green' if bool(trial.get('ok', False)) else 'red'}"
                                                )
                                                ui.label(
                                                    f"Witnesses: {int(trial.get('witness_count', 0) or 0)}"
                                                ).classes("text-xs text-slate-600")
                                                ui.label(
                                                    "Initial wrong: "
                                                    + _format_optional_count(
                                                        trial.get("initial_violated_rule_count")
                                                    )
                                                ).classes("text-xs text-slate-600")
                                                ui.label(
                                                    "Corrected: "
                                                    + str(trial.get("repair_correction_summary") or "none")
                                                ).classes("text-xs text-slate-600")
                                                ui.label(
                                                    "Final wrong: "
                                                    + _format_optional_count(
                                                        trial.get("final_violated_rule_count")
                                                    )
                                                ).classes("text-xs text-slate-600")
                                                ui.label(
                                                    f"Verification: {float(trial.get('verification_time_ms', 0.0) or 0.0):.2f} ms"
                                                ).classes("text-xs text-slate-600")
                                                ui.button(
                                                    "Load DAG Details",
                                                    icon="visibility",
                                                    on_click=lambda scenario_name=scenario_id, method_name=method, idx=trial_index: asyncio.create_task(
                                                        _load_trial_detail(scenario_name, method_name, idx)
                                                    ),
                                                ).props("flat color=primary")
                                            if isinstance(trial_detail, dict):
                                                _render_trial_detail(trial_detail)
                                            else:
                                                ui.label(
                                                    "Load this trial to inspect the generated task DAG and node-detail tables."
                                                ).classes("text-xs text-slate-500 mt-2")

                with ui.expansion("Advanced JSON", icon="code", value=False).classes("w-full"):
                    ui.label(
                        "Debug and import/export view of the persisted study manifest. Hidden by default."
                    ).classes("text-xs text-slate-600 mb-2")
                    advanced_editor = ui.textarea(label="Study Manifest JSON").classes("w-full font-mono").props(
                        "outlined autogrow"
                    )
                    with ui.row().classes("gap-2 mt-2"):
                        ui.button("Save Advanced JSON", icon="save", on_click=_save_advanced_json).props(
                            "flat color=primary"
                        )
                        ui.button("Reload Advanced JSON", icon="refresh", on_click=lambda: _sync_advanced_json()).props(
                            "flat"
                        )

        with ui.column().classes("w-[24rem] shrink-0 sticky top-20 self-start"):
            with ui.card().classes("w-full"):
                ui.label("Workflow").classes("text-lg font-semibold")
                ui.label("1. Author products and requirements on Products.").classes("text-sm text-slate-600")
                ui.label("2. Author and verify safety files on Safety.").classes("text-sm text-slate-600")
                ui.label("3. Use Resources for runtime robots; Experiments also adds the offline-only robots.").classes(
                    "text-sm text-slate-600"
                )
                ui.label("4. Select assets here, run trials, then analyze aggregate results and single-trial details.").classes(
                    "text-sm text-slate-600 mt-2"
                )
                ui.label(
                    "Pure LLM uses only the safety text prompt. LLM + Formal Verification reuses the approved Safety-page rule preview as the formal layer."
                ).classes("text-sm text-slate-600 mt-2")

    def _handle_manifest_change(value: Any) -> None:
        manifest_path = str(value or "").strip()
        if not manifest_path:
            return
        state["manifest_file"] = manifest_path
        asyncio.create_task(_reload_context())

    manifest_select.on_value_change(lambda e: _handle_manifest_change(e.value))
    save_btn.on_click(lambda: asyncio.create_task(_save_study()))
    reload_btn.on_click(lambda: asyncio.create_task(_reload_context()))

    _ensure_defaults()
    _sync_advanced_json()
    _clear_dirty()
    settings_card()
    quick_editor_card()
    scenarios_card()
    analysis_card()
    ui.timer(2.0, lambda: asyncio.create_task(_poll_asset_catalogs()))
