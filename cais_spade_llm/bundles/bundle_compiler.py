"""Offline compiler for verified (safety + plan + validation) bundles."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
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

    def _collect_resource_refs(self, robot_env: str) -> list[Any]:
        refs: list[Any] = []
        for path in sorted(self.resource_init_dir.glob("*.json")):
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

    @classmethod
    def build_validation_payload(
        cls,
        *,
        ok: bool,
        violations: list[dict[str, Any]],
        auto_replans_used: int,
        stop_reason: str,
    ) -> dict[str, Any]:
        violated_rules = cls._violated_rules(violations)
        return {
            "ok": bool(ok),
            "violations": violations,
            "violated_rules": violated_rules,
            "witness_count": len(violations),
            "auto_replans_used": int(auto_replans_used),
            "stop_reason": str(stop_reason),
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
        if seed_replan_violations:
            await planner.replan_with_feedback_offline(seed_replan_violations)
            planner.compile_global_fsa()

        auto_replans_used = 0
        while True:
            ok, violations = validator.validate_plan_fsa(
                fsa=planner.global_fsa or {},
                plan={"nodes": planner.nodes},
                product_jid=product_jid,
            )
            if ok:
                stop_reason = "initial_valid" if auto_replans_used == 0 else "repaired_valid"
                return cls.build_validation_payload(
                    ok=True,
                    violations=violations,
                    auto_replans_used=auto_replans_used,
                    stop_reason=stop_reason,
                )

            if auto_replans_used >= auto_replan_max_attempts:
                return cls.build_validation_payload(
                    ok=False,
                    violations=violations,
                    auto_replans_used=auto_replans_used,
                    stop_reason="max_attempts_reached",
                )

            before_hash = cls._task_nodes_hash(planner.nodes)
            auto_replans_used += 1
            try:
                await planner.replan_with_feedback_offline(violations)
                after_hash = cls._task_nodes_hash(planner.nodes)
                if before_hash == after_hash:
                    return cls.build_validation_payload(
                        ok=False,
                        violations=violations,
                        auto_replans_used=auto_replans_used,
                        stop_reason="repair_no_change",
                    )
                planner.compile_global_fsa()
            except Exception:
                log.exception("Offline replan attempt %d failed.", auto_replans_used)
                return cls.build_validation_payload(
                    ok=False,
                    violations=violations,
                    auto_replans_used=auto_replans_used,
                    stop_reason="repair_exception",
                )

    async def compile_bundle(
        self,
        *,
        product_init_file: str,
        execution_mode: str,
        robot_env: str,
        product_requirement_file: str | None = None,
        safety_requirement_file: str | None = None,
        precomputed_safety_artifacts: dict[str, Any] | None = None,
        auto_replan_max_attempts: int = 3,
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

        resources = self._collect_resource_refs(robot_env=robot_env)
        resource_jids = [str(r.jid) for r in resources]

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
            cca_agent = CentralControllerAgent(
                cca_jid,
                cca_pw,
                name=cca_name,
                resource_agents=resources,
                safety_file=str(safety_path),
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
