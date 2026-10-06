"""Inspect supplied candidates in an already launched, dedicated Gazebo test session.

This creates registered owners and a CCA, but starts no SPADE behaviors, discovery,
admission or execution. Initialization ledgers remain configured assumptions until
the preparation gate establishes its checkpoint evidence. Saved results never
authorize dispatch. Use a fresh test launch, not a production run with prior work.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from cais_spade_llm.recovery_framework import PRODUCT_PATH, ROOT, SCENE_PATH, read_json

logger = logging.getLogger(__name__)


async def check_candidates(safety_file: Path, output_directory: Path) -> list[dict]:
    """Run the existing adapter with real registered owners and no agent behaviors."""
    from cais_spade_llm.agents.central_controller.central_controller_agent import (
        CentralControllerAgent,
    )
    from cais_spade_llm.agents.central_controller.predefined_safety_runtime import (
        initialize_predefined_safety,
    )
    from cais_spade_llm.product.environment import EnvironmentProductContext
    from cais_spade_llm.recovery_framework.environment_runtime import EnvironmentRuntime
    from cais_spade_llm.recovery_framework.gazebo_safety_preparation import (
        capture_checkpoint,
        prepare_and_check,
    )
    from cais_spade_llm.recovery_framework.live_safety_preparation import (
        build_supplied_candidate,
        install_live_preparation,
    )
    from cais_spade_llm.recovery_framework.workflow_execution import (
        create_environment_resource_agents,
    )

    meta = next(iter(read_json(PRODUCT_PATH).values()))
    inputs = {
        "scene": read_json(SCENE_PATH),
        "product_order": read_json(ROOT / meta["product_order_file"]),
        "geometry": read_json(ROOT / meta["product_geometry_file"])["gazebo"],
    }
    context = EnvironmentProductContext(**inputs)
    owners = create_environment_resource_agents(inputs["scene"], context.models, "cca@localhost")
    # EnvironmentRuntime receives an inert product identity. No PA is started.
    runtime = EnvironmentRuntime(
        SimpleNamespace(jid="assembly_board-v1@localhost"),
        {
            "inputs": inputs,
            "setup": {"execution_mode": "simulation", "permitted_resources": list(context.models)},
        },
        owners,
    )
    cca = CentralControllerAgent(
        "cca@localhost", "none", name="cca", resource_agents=owners, safety_file=str(safety_file)
    )
    bridge = SimpleNamespace(resource_agents=owners, cca=cca, selected_safety_file=str(safety_file))
    results = []
    request = None
    try:
        initialize_predefined_safety(cca)
        if cca.predefined_safety_error:
            raise ValueError(cca.predefined_safety_error)
        for owner in owners:
            controller = getattr(owner, "_controller", None)
            if controller is not None and not controller.wait_for_services(timeout_sec=30):
                raise ValueError("Controller not ready: " + owner.agent_name)
        for candidate_id in inputs["scene"]["safety_preparation"]["candidates"]:
            report = {
                "version": 1,
                "candidate_id": candidate_id,
                "origin": "mock",
                "dispatch_authorized": False,
                "status": "NEEDS_CONTEXT",
                "analysis": None,
                "prepared_programs": [],
                "unresolved": [],
            }
            try:
                install_live_preparation(runtime, cca)
                provider = cca.recovery_safety_preparation_provider
                provider.initialize()
                report["checkpoint"] = capture_checkpoint(runtime, cca)
                request = build_supplied_candidate(bridge, candidate_id)
                report = prepare_and_check(bridge, request, output_root=output_directory)
            except (ValueError, KeyError, RuntimeError, OSError, TypeError) as exc:
                report["unresolved"].append({"reason": str(exc)})
            if 'artifact_path' in report:
                path = Path(report['artifact_path'])
            else:
                path = output_directory / uuid4().hex / "result.json"
                path.parent.mkdir(parents=True, exist_ok=False)
                path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            results.append(
                {
                    "candidate_id": candidate_id,
                    "status": report["status"],
                    "artifact_path": str(path),
                    "dispatch_authorized": False,
                }
            )
            logger.info("%s: %s (%s)", candidate_id, report["status"], path)
        # Invalidate one required owner declaration in this isolated test provider.
        # The normal adapter must reject it before planning; restore it afterward.
        provider = cca.recovery_safety_preparation_provider
        declarations = deepcopy(provider.configuration["resources"])
        try:
            provider.configuration["resources"].pop(next(iter(declarations)))
            if request is not None:
                negative = prepare_and_check(bridge, request, output_root=output_directory)
                results.append(
                    {
                        "candidate_id": "missing_resource_evidence",
                        "status": negative["status"],
                        "artifact_path": negative["artifact_path"],
                        "dispatch_authorized": False,
                    }
                )
        finally:
            provider.configuration["resources"] = declarations
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
    """Require an explicit predefined document and save the test-session outcome."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predefined-safety", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args()
    if args.output_directory.exists():
        parser.error('Use a new output directory so previous test evidence is preserved')
    logging.basicConfig(level=logging.INFO)
    results = asyncio.run(
        check_candidates(args.predefined_safety.resolve(), args.output_directory.resolve())
    )
    args.output_directory.mkdir(parents=True, exist_ok=True)
    (args.output_directory / "summary.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    statuses = {row["candidate_id"]: row["status"] for row in results}
    return (
        0
        if statuses
        == {
            "GAZEBO_MOTION_SAFE": "allowed",
            "GAZEBO_MOTION_CONFLICT": "held",
            "missing_resource_evidence": "NEEDS_CONTEXT",
        }
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
