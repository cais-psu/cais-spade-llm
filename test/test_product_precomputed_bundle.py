from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "sk-local-test")

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / "cais_spade_llm"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from cais_spade_llm.agents.intelligent_product.product_agent import ProductAgent
from cais_spade_llm.resources.sensor.camera_module import CameraModule


def test_product_agent_loads_precomputed_plan_bundle(tmp_path):
    plan_src = tmp_path / "bundle" / "plan" / "product_plan.json"
    fsa_src = tmp_path / "bundle" / "plan" / "product_global_fsa.json"
    req_src = tmp_path / "bundle" / "plan" / "product_requirements.json"
    plan_src.parent.mkdir(parents=True, exist_ok=True)

    plan_src.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "REQ_1_T1",
                        "type": "task",
                        "status": "pending",
                        "function_name": "pick_approach",
                        "params": {},
                        "resource_jid": "xarm6@localhost",
                        "predecessors": [],
                        "successors": [],
                    }
                ]
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    fsa_src.write_text(
        json.dumps(
            {
                "A": {
                    "X": ["S0"],
                    "E": [],
                    "Tr": [],
                    "x0": "S0",
                    "Xm": ["S0"],
                },
                "meta": {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    req_src.write_text(json.dumps({"nodes": []}, indent=2), encoding="utf-8")

    resource = SimpleNamespace(jid="xarm6@localhost", static_capabilities={})
    agent = ProductAgent(
        "assembly_board-v1@localhost",
        "none",
        name="assembly_board-v1",
        resource_jids=["xarm6@localhost"],
        resource_agents=[resource],
        product_specification_file="cais_spade_llm/specification/products/requirements/assembly_board-v1.txt",
        safety_file="cais_spade_llm/specification/safety/safety_requirements.txt",
        camera=CameraModule(backend="none"),
        precomputed_bundle={
            "bundle_id": "B1",
            "artifacts": {
                "requirements_json": str(req_src),
                "plan_json": str(plan_src),
                "global_fsa_json": str(fsa_src),
            },
        },
    )
    agent.structured_requirements_path = tmp_path / "runtime" / "requirements.json"
    agent.plan_path = tmp_path / "runtime" / "plan.json"
    agent.global_fsa_path = tmp_path / "runtime" / "fsa.json"

    ok = agent._load_precomputed_plan_bundle()
    assert ok is True
    assert agent.process_planner.nodes and agent.process_planner.nodes[0]["id"] == "REQ_1_T1"
    assert isinstance(agent.process_planner.global_fsa, dict)
    assert agent.plan_path.exists()
    assert agent.global_fsa_path.exists()
    assert agent.structured_requirements_path.exists()
