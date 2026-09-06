from __future__ import annotations

"""Shared ontology configuration for ProductAgent grounding tests."""


from pathlib import Path

from cais_spade_llm.spec2primitives.agents.pa.context_grounding import (
    PAOntologyConfig,
)

PPR_NAMESPACE = "http://PAonto.com#"
MINIMAL_TBOX_PATH = Path(__file__).parent / "fixtures/ontology/minimal_ppr_tbox.owl"


def ontology_config() -> PAOntologyConfig:
    """Return the schema-only test fixture configuration."""
    return PAOntologyConfig(
        tbox_path=MINIMAL_TBOX_PATH,
        ppr_namespace=PPR_NAMESPACE,
    )


class OfflineMoveItPlanning:
    """Return concrete offline joint-plan fixtures without contacting ROS or a robot."""

    def __init__(self, statuses=None):
        self.statuses = statuses or {}

    async def validate_plan_only_allocation(self, request):
        status = self.statuses.get(request["resource_symbol"], "accepted")
        result = {
            "status": "accepted",
            "state_locations": {
                state: [
                    {
                        "evidence_handle": location["evidence_handle"],
                        "status": "accepted",
                        "message": "Offline MoveIt planning fixture.",
                        "error_code": 1,
                        "plan": {
                            "joint_names": ["fixture_joint"],
                            "points": [[0.0], [0.1]],
                            "start_joint_names": ["fixture_joint"],
                            "start_joint_positions": [0.0],
                        },
                    }
                    for location in locations
                ]
                for state, locations in request["state_locations"].items()
            },
            "feedback": None,
        }
        if status != "accepted":
            result["status"] = status
            result["feedback"] = (
                "Offline planning unavailable." if status == "needs_context" else None
            )
            for locations in result["state_locations"].values():
                for location in locations:
                    location.update(
                        status=status,
                        error_code=None if status == "needs_context" else -1,
                        plan=None,
                    )
        return result
