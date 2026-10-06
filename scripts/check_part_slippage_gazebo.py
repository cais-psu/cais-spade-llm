"""Capture live prerequisites for the three supplied part_slippage Gazebo companions.

This is a preflight, not a recovery executor or a safety-violation demonstration.
It preserves missing evidence and never substitutes synthetic trajectories,
stages entities, installs a proof, or records/publishes an unsuccessful trial.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from cais_spade_llm.recovery_framework import PRODUCT_PATH, ROOT, SCENE_PATH, fingerprint, read_json

logger = logging.getLogger(__name__)
COMPANIONS = ROOT / "test/fixtures/part_slippage/gazebo"
CASES = ("mutex", "precedence", "safe")


def load_companion(name: str, directory: Path = COMPANIONS) -> dict:
    """Check exact fixture/specification identities without loading synthetic motion."""
    if name not in CASES:
        raise ValueError("Unknown part_slippage case: " + name)
    companion = read_json(directory / (name + ".json"))
    source = companion["source_fixture"]
    raw = (directory / source["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != source["sha256"]:
        raise ValueError("Original mock fixture changed")
    original = json.loads(raw)
    safety = companion["predefined_safety"]
    if hashlib.sha256((ROOT / safety["path"]).read_bytes()).hexdigest() != safety["sha256"]:
        raise ValueError("Predefined safety definitions changed")
    if (companion["case_id"] != original["case_id"] or companion["candidate_origin"] != "mock"
            or companion["diagnostic_cca_bypass"] is not False):
        raise ValueError("Companion identity or CCA authority changed")
    expected = []
    for event in original["outline_events"]:
        program = next(row for row in original["inputs"]["programs"] if row["resource_id"] == event["resource_id"])
        row = {key: deepcopy(event[key]) for key in (
            "outline_id", "des_event_id", "event_name", "resource_id", "resource_jid", "part_name",
            "predecessors", "primitive_step_indices",
        )}
        row["primitive_steps"] = [
            {key: deepcopy(program["primitive_steps"][index][key]) for key in ("primitive", "source")}
            for index in event["primitive_step_indices"]
        ]
        expected.append(row)
    if companion["outline_events"] != expected:
        raise ValueError("Companion changed event ordering or primitive provenance")
    return companion


def _save(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


async def capture_case(companion: dict, runtime, cca) -> dict:
    """Capture a fresh checkpoint and report actual owner admission support."""
    from cais_spade_llm.recovery_framework.gazebo_safety_preparation import capture_checkpoint
    from cais_spade_llm.recovery_framework.live_safety_preparation import install_live_preparation

    report = {
        "version": 1, "case_id": companion["case_id"], "run_id": runtime.context.run_id,
        "phase": "live_preflight", "status": "NEEDS_CONTEXT", "acceptance_complete": False,
        "diagnostic_cca_bypass": False, "dispatch_authorized": False,
        "staging_performed": False, "recovery_execution_started": False,
        "recording_started": False, "published_videos": [], "cca_decision": None,
        "safety_verdict": None, "ap_evidence": [], "prepared_programs": [],
        "companion": deepcopy(companion), "owner_admission_support": [], "unresolved": [],
    }
    actors = {row["resource_id"] for row in companion["outline_events"]}
    for owner in runtime.resource_agents:
        if owner.agent_name not in actors:
            continue
        registered = callable(getattr(getattr(owner, "recovery_composition_evidence_provider", None), "prepare", None))
        report["owner_admission_support"].append({
            "resource_id": owner.agent_name, "resource_jid": str(owner.jid),
            "status": "registered" if registered else "NEEDS_CONTEXT",
            "reason": ("Registered provider still requires resolved native programs and execution evidence"
                       if registered else "Resource-owned execution evidence provider is unavailable"),
        })
    if not callable(getattr(cca, "recovery_composition_context_provider", None)):
        report["unresolved"].append({"reason": "live_recovery_evidence_unavailable", "owner": "CCA"})
    try:
        install_live_preparation(runtime, cca)
        provider = cca.recovery_safety_preparation_provider
        provider.initialize()
        checkpoint = capture_checkpoint(runtime, cca)
        report["checkpoint"] = checkpoint
        report["unresolved"].extend(deepcopy(checkpoint["unresolved"]))
        provider.validate_idle(checkpoint)
    except (ValueError, KeyError, RuntimeError, OSError, TypeError) as exc:
        report["unresolved"].append({"reason": str(exc)})
    report["unresolved"].extend(
        {"resource_id": row["resource_id"], "reason": row.get("reason", "Incomplete admission evidence")}
        for row in report["owner_admission_support"] if row["status"] != "prepared"
    )
    # Preflight captures the initial scene, never claims it is the requested
    # post-slippage checkpoint or turns the fixture's ledgers into observations.
    report["unresolved"].append({
        "reason": "Post-slippage staging, native preparation and CCA-granted execution remain required",
    })
    report["fingerprint"] = fingerprint(report)
    return report


async def check_cases(output_directory: Path) -> list[dict]:
    """Inspect all three companions in a dedicated simulation, with no agent dispatch."""
    from cais_spade_llm.agents.central_controller.central_controller_agent import (
        CentralControllerAgent,
    )
    from cais_spade_llm.agents.central_controller.predefined_safety_runtime import (
        initialize_predefined_safety,
    )
    from cais_spade_llm.product.environment import EnvironmentProductContext
    from cais_spade_llm.recovery_framework.environment_runtime import EnvironmentRuntime
    from cais_spade_llm.recovery_framework.workflow_execution import (
        create_environment_resource_agents,
    )

    companions = [load_companion(name) for name in CASES]
    definitions = companions[0]["predefined_safety"]
    if any(row["predefined_safety"] != definitions for row in companions):
        raise ValueError("All cases must use the same unchanged specifications")
    meta = next(iter(read_json(PRODUCT_PATH).values()))
    inputs = {"scene": read_json(SCENE_PATH), "product_order": read_json(ROOT / meta["product_order_file"]),
              "geometry": read_json(ROOT / meta["product_geometry_file"])["gazebo"]}
    context = EnvironmentProductContext(**inputs)
    owners = create_environment_resource_agents(inputs["scene"], context.models, "cca@localhost")
    runtime = EnvironmentRuntime(SimpleNamespace(jid="assembly_board-v1@localhost"), {
        "inputs": inputs, "setup": {"execution_mode": "simulation", "permitted_resources": list(context.models),
                                     "diagnostic_cca_bypass": False},
    }, owners)
    cca = CentralControllerAgent("cca@localhost", "none", name="cca", resource_agents=owners,
                                 safety_file=str(ROOT / definitions["path"]))
    results = []
    try:
        initialize_predefined_safety(cca)
        if cca.predefined_safety_error:
            raise ValueError(cca.predefined_safety_error)
        for owner in owners:
            controller = getattr(owner, "_controller", None)
            if controller is not None and not controller.wait_for_services(timeout_sec=30):
                raise ValueError("Controller not ready: " + owner.agent_name)
        for name, companion in zip(CASES, companions, strict=True):
            report = await capture_case(companion, runtime, cca)
            path = output_directory / (name + ".json")
            _save(path, report)
            results.append({"case_id": companion["case_id"], "phase": report["phase"],
                            "status": report["status"], "acceptance_complete": False, "report": str(path)})
            logger.info("%s: live acceptance incomplete; %s", companion["case_id"], path)
    finally:
        provider = getattr(cca, "recovery_safety_preparation_provider", None)
        if provider is not None and provider.reader is not None:
            provider.reader.close()
        for owner in owners:
            controller = getattr(owner, "_controller", None)
            if controller is not None:
                controller.shutdown()
    return results


def main() -> int:
    """Save a new prerequisite report; preflight alone never reports acceptance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args()
    directory = args.output_directory.resolve()
    if directory.exists():
        parser.error("Use a new directory to preserve previous evidence")
    directory.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO)
    try:
        summary = {"cases": asyncio.run(check_cases(directory)), "acceptance_complete": False,
                   "dispatch_authorized": False, "published_videos": []}
    except (ValueError, KeyError, RuntimeError, OSError, TypeError) as exc:
        summary = {"status": "NEEDS_CONTEXT", "acceptance_complete": False,
                   "dispatch_authorized": False, "published_videos": [], "reason": str(exc)}
        logger.error("Live prerequisites unavailable: %s", exc)
    _save(directory / "summary.json", summary)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
