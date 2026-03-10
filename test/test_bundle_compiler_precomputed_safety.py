from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "cais_spade_llm"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from cais_spade_llm.bundles.bundle_compiler import BundleCompiler
from cais_spade_llm.bundles.bundle_store import BundleStore
from cais_spade_llm.bundles.models import sha256_file, sha256_text


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class _FakePlanner:
    def __init__(self) -> None:
        self.nodes: list[dict] = []
        self.global_fsa: dict = {}

    async def build_high_level(self, requirement_text: str, **_: object) -> None:
        assert requirement_text
        self.nodes = [{"id": "REQ_1", "type": "requirement"}]

    async def expand_requirements_to_tasks(self, *, safety_text: str = "", **_: object) -> None:
        assert safety_text
        self.nodes = [{"id": "TASK_1", "type": "task", "predecessors": []}]
        self.global_fsa = {
            "A": {"X": ["S0"], "E": [], "Tr": [], "x0": "S0", "Xm": ["S0"]},
            "meta": {},
        }

    def save(self, path: Path) -> None:
        _write_json(path, {"nodes": self.nodes})

    def save_global_fsa(self, path: Path) -> None:
        _write_json(path, self.global_fsa)


class _FakeCamera:
    def __init__(self, backend: str = "none") -> None:
        self.backend = backend

    def destroy(self) -> None:
        return None


class _FakeProductAgent:
    def __init__(self, jid: str, password: str, **kwargs: object) -> None:
        del password, kwargs
        self.jid = jid
        self.process_planner = _FakePlanner()
        self.non_function_model = "planner-model"
        self.tools_catalog: list[dict] = []
        self.camera = _FakeCamera()


class _ExplodingSafetyLogic:
    async def build_safety_rules_and_logic(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("live safety logic regeneration should not run")

    async def build_preview_interpretations(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("preview interpretation generation should not run")

    def save(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("live safety save should not run")

    def build_dfas_per_rule(self, *args: object, **kwargs: object) -> dict[str, str]:
        del args, kwargs
        raise AssertionError("live DFA generation should not run")


class _FakeCCAAgent:
    def __init__(self, jid: str, password: str, **kwargs: object) -> None:
        del jid, password, kwargs
        self.non_function_model = "safety-model"
        self.safety_logic = _ExplodingSafetyLogic()


class _FakePlanSafetyValidator:
    last_init: dict[str, object] = {}

    def __init__(self, rules: list[dict], dfa_map: dict[str, str], tools_catalog: list[dict] | None = None) -> None:
        self.rules = rules
        self.dfa_map = dfa_map
        self.tools_catalog = tools_catalog or []
        type(self).last_init = {
            "rules": rules,
            "dfa_map": dfa_map,
            "tools_catalog": self.tools_catalog,
        }

    def validate_plan_fsa(self, **_: object) -> tuple[bool, list[dict]]:
        return True, []


class _FakeLlmAgent:
    configured_tools_path: str = ""

    @classmethod
    def configure_shared_tools_catalogue(cls, path: Path) -> None:
        cls.configured_tools_path = str(path)


def test_bundle_compiler_uses_precomputed_approved_safety_artifacts(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir(parents=True, exist_ok=True)

    requirement_file = project_root / "requirements.txt"
    requirement_file.write_text("[Product Requirements]\n- assemble demo\n", encoding="utf-8")
    safety_file = project_root / "safety.txt"
    safety_file.write_text("[Safety Requirements]\n- keep robots separated\n", encoding="utf-8")

    product_init_dir = project_root / "products"
    resource_init_dir = project_root / "resources"
    product_init_dir.mkdir(parents=True, exist_ok=True)
    resource_init_dir.mkdir(parents=True, exist_ok=True)
    product_init_file = product_init_dir / "demo_product.json"
    _write_json(
        product_init_file,
        {
            "demo_product": {
                "jid": "demo_product@localhost",
                "password": "none",
                "domain": "localhost",
                "product_specification_file": str(requirement_file),
            }
        },
    )

    cca_init_path = project_root / "cca.json"
    _write_json(
        cca_init_path,
        {
            "cca": {
                "jid": "cca@localhost",
                "password": "none",
                "domain": "localhost",
                "safety_file": str(safety_file),
            }
        },
    )

    tools_path = project_root / "tools.json"
    tools_path.write_text(json.dumps([{"function": "place_approach"}], indent=2), encoding="utf-8")
    prompts_path = project_root / "prompts.py"
    prompts_path.write_text("PROMPTS_VERSION = 'test'\n", encoding="utf-8")

    preview_dir = project_root / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    source_logic_path = preview_dir / "cca_safety_logic.json"
    _write_json(
        source_logic_path,
        {
            "preview_interpretation_summary": "- SAFE_1: keep robots separated",
            "rules": [{"id": "SAFE_1", "ltlf": "G(true)", "aps": []}],
        },
    )
    source_dot_path = preview_dir / "SAFE_1_dfa.dot"
    source_dot_path.write_text("digraph { 1 -> 1 [label=\"true\"]; }", encoding="utf-8")
    source_png_path = preview_dir / "SAFE_1_dfa.png"
    source_png_path.write_bytes(b"PNG")

    store = BundleStore(project_root / "user_verified_plan")
    compiler = BundleCompiler(
        store=store,
        project_root=project_root,
        product_init_dir=product_init_dir,
        resource_init_dir=resource_init_dir,
        cca_init_path=cca_init_path,
        tools_path=tools_path,
        prompts_path=prompts_path,
    )

    original_import_runtime = BundleCompiler._import_runtime_classes
    original_collect_resources = BundleCompiler._collect_resource_refs
    original_to_thread = asyncio.to_thread
    try:
        BundleCompiler._import_runtime_classes = staticmethod(
            lambda: (
                _FakeProductAgent,
                _FakeCCAAgent,
                _FakePlanSafetyValidator,
                _FakeCamera,
                _FakeLlmAgent,
            )
        )
        BundleCompiler._collect_resource_refs = lambda self, robot_env: []
        async def _direct_to_thread(func, /, *args, **kwargs):
            return func(*args, **kwargs)
        asyncio.to_thread = _direct_to_thread

        result = asyncio.run(
            compiler.compile_bundle(
                product_init_file=str(product_init_file),
                execution_mode="simulation",
                robot_env="gazebo",
                product_requirement_file=str(requirement_file),
                safety_requirement_file=str(safety_file),
                precomputed_safety_artifacts={
                    "mode": "approved_preview",
                    "preview_id": "preview_approved",
                    "preview_generated_at_utc": "2026-03-10T15:00:00+00:00",
                    "safety_logic_json": str(source_logic_path.resolve()),
                    "dfa_dot_files": [str(source_dot_path.resolve())],
                    "dfa_png_files": [str(source_png_path.resolve())],
                    "safety_sha256": sha256_text(safety_file.read_text(encoding="utf-8").strip()),
                    "tools_sha256": sha256_file(tools_path),
                    "prompts_sha256": sha256_file(prompts_path),
                },
                auto_replan_max_attempts=1,
            )
        )
    finally:
        BundleCompiler._import_runtime_classes = original_import_runtime
        BundleCompiler._collect_resource_refs = original_collect_resources
        asyncio.to_thread = original_to_thread

    bundle_dir = Path(str(result["bundle_dir"]))
    assert bundle_dir.exists()
    copied_logic_path = bundle_dir / "safety" / "cca_safety_logic.json"
    copied_dot_path = bundle_dir / "safety" / "SAFE_1_dfa.dot"
    copied_png_path = bundle_dir / "safety" / "SAFE_1_dfa.png"
    assert copied_logic_path.read_text(encoding="utf-8") == source_logic_path.read_text(encoding="utf-8")
    assert copied_dot_path.read_text(encoding="utf-8") == source_dot_path.read_text(encoding="utf-8")
    assert copied_png_path.read_bytes() == source_png_path.read_bytes()

    manifest = result["manifest"]
    assert manifest["safety_source"]["mode"] == "approved_preview"
    assert manifest["safety_source"]["preview_id"] == "preview_approved"
    assert manifest["artifacts"]["safety_logic_json"] == "safety/cca_safety_logic.json"
    assert "safety/SAFE_1_dfa.dot" in manifest["artifacts"]["safety_dfa_dot_files"]
    assert "safety/SAFE_1_dfa.png" in manifest["artifacts"]["safety_dfa_png_files"]

    validator_init = _FakePlanSafetyValidator.last_init
    assert validator_init["rules"] == [{"id": "SAFE_1", "ltlf": "G(true)", "aps": []}]
    assert "SAFE_1" in validator_init["dfa_map"]
