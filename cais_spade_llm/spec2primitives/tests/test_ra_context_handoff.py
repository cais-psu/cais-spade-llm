"""Tests for the Phase 5.1 selected-RA assignment and context snapshots."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from cais_spade_llm.spec2primitives.agents.ra.context_handoff import (
    RAContextHandoffError,
    SelectedRAAssignmentEnvelope,
    activate_selected_ra_context,
)
from cais_spade_llm.spec2primitives.tests.test_pa_completion import (
    persist_native_completion_fixture,
)


class _AssignedContextRuntime:
    def __init__(
        self,
        *,
        self_jid: str = "xarm6@localhost",
        transform: Callable[[dict[str, object]], None] | None = None,
        failure: Exception | None = None,
    ) -> None:
        self.self_jid = self_jid
        self.transform = transform
        self.failure = failure
        self.assignments: list[SelectedRAAssignmentEnvelope] = []

    async def request_assigned_context(
        self,
        assignment: SelectedRAAssignmentEnvelope,
    ) -> dict[str, object]:
        assignment.assert_addressed_to(self.self_jid)
        self.assignments.append(assignment)
        if self.failure is not None:
            raise self.failure
        response: dict[str, object] = {
            "resource_jid": self.self_jid,
            "assignment_fingerprint": assignment.fingerprint,
            "robot_state": {
                "execution_mode": "simulation",
                "controller_ready": True,
                "held_part": None,
                "current_state": "idle",
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "gripper_state": "open",
            },
            "primitive_catalog": _valid_catalog(),
        }
        if self.transform is not None:
            self.transform(response)
        return response


def test_phase_5_1_dispatches_assignment_and_appends_paired_snapshots(
    tmp_path: Path,
) -> None:
    persist_native_completion_fixture(tmp_path)
    runtime = _AssignedContextRuntime()

    first = asyncio.run(activate_selected_ra_context(runtime, tmp_path))
    first_state_bytes = first.robot_state_path.read_bytes()
    first_catalog_bytes = first.primitive_catalog_path.read_bytes()
    second = asyncio.run(activate_selected_ra_context(runtime, tmp_path))

    assert len(runtime.assignments) == 2
    assert runtime.assignments[0] == runtime.assignments[1]
    assert first.assignment.product_requirement == "assemble medium gear"
    assert first.assignment.selected_resource_jid == "xarm6@localhost"
    assert runtime.assignments[0].fingerprint == first.assignment.fingerprint
    assert first.assignment_path.relative_to(tmp_path).as_posix() == (
        "composition/selected_ra_assignments/assignment_0001.json"
    )
    assert first.robot_state_path.name == "snapshot_0001.json"
    assert first.primitive_catalog_path.name == "snapshot_0001.json"
    assert second.robot_state_path.name == "snapshot_0002.json"
    assert second.primitive_catalog_path.name == "snapshot_0002.json"
    assert first.robot_state_path.read_bytes() == first_state_bytes
    assert first.primitive_catalog_path.read_bytes() == first_catalog_bytes
    assert [entry["primitive_symbol"] for entry in first.primitive_catalog.primitive_catalog] == [
        "detect_parts",
        "move_pose",
    ]

    assignment_record = _read_json(first.assignment_path)
    catalog_record = _read_json(first.primitive_catalog_path)
    assert "typed_context_refs" not in assignment_record
    assert "ontology_projection_ref" not in assignment_record
    assert "assertions" not in assignment_record
    assert catalog_record["assignment_fingerprint"] == first.assignment.fingerprint
    assert (
        catalog_record["robot_state_ref"] == first.robot_state_path.relative_to(tmp_path).as_posix()
    )
    assert catalog_record["robot_state_fingerprint"] == first.robot_state.fingerprint
    assert not (tmp_path / "composition/context_bundles").exists()
    assert not (tmp_path / "resources/xarm6@localhost/primitive_program_drafts").exists()
    assert not (tmp_path / "composition/missing_context_batches").exists()
    assert not (tmp_path / "resources/xarm6@localhost/primitive_steps").exists()
    assert not (tmp_path / "resources/xarm6@localhost/validation").exists()


def test_changed_phase_4_selection_prevents_ra_dispatch(tmp_path: Path) -> None:
    completion = persist_native_completion_fixture(tmp_path).to_record()
    selection_path = tmp_path / str(completion["resource_selection_ref"])
    selection = _read_json(selection_path)
    selection["selected_resource_jid"] = "ur5e@localhost"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    runtime = _AssignedContextRuntime()

    with pytest.raises(
        RAContextHandoffError,
        match="unchanged PAContextGroundingCompletion",
    ):
        asyncio.run(activate_selected_ra_context(runtime, tmp_path))

    assert runtime.assignments == []
    assert not (tmp_path / "composition/selected_ra_assignments").exists()


def test_assignment_addressed_to_another_ra_is_rejected(tmp_path: Path) -> None:
    persist_native_completion_fixture(tmp_path)
    runtime = _AssignedContextRuntime(self_jid="ur5e@localhost")

    with pytest.raises(RAContextHandoffError, match="addressed to a different RA"):
        asyncio.run(activate_selected_ra_context(runtime, tmp_path))

    assert runtime.assignments == []
    assert (tmp_path / "composition/selected_ra_assignments/assignment_0001.json").is_file()
    assert not (tmp_path / "resources/xarm6@localhost").exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("resource_jid", "ur5e@localhost", "response JID"),
        ("assignment_fingerprint", "0" * 64, "assignment fingerprint"),
    ],
)
def test_response_must_match_assignment(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    persist_native_completion_fixture(tmp_path)

    def transform(response: dict[str, object]) -> None:
        response[field] = value

    runtime = _AssignedContextRuntime(transform=transform)
    with pytest.raises(RAContextHandoffError, match=message):
        asyncio.run(activate_selected_ra_context(runtime, tmp_path))

    assert len(runtime.assignments) == 1
    assert not (tmp_path / "resources/xarm6@localhost").exists()


@pytest.mark.parametrize(
    "variant",
    ["empty", "duplicate", "composite", "non_finite"],
)
def test_invalid_ra_context_is_not_persisted(tmp_path: Path, variant: str) -> None:
    persist_native_completion_fixture(tmp_path)

    def transform(response: dict[str, object]) -> None:
        if variant == "empty":
            response["primitive_catalog"] = []
        elif variant == "duplicate":
            catalog = _valid_catalog()
            catalog[1]["primitive_symbol"] = catalog[0]["primitive_symbol"]
            response["primitive_catalog"] = catalog
        elif variant == "composite":
            catalog = _valid_catalog()
            catalog[0]["primitive_steps"] = []
            response["primitive_catalog"] = catalog
        else:
            state = deepcopy(response["robot_state"])
            state["position"]["x"] = float("nan")
            response["robot_state"] = state

    runtime = _AssignedContextRuntime(transform=transform)
    with pytest.raises(RAContextHandoffError):
        asyncio.run(activate_selected_ra_context(runtime, tmp_path))

    assert len(runtime.assignments) == 1
    assert (tmp_path / "composition/selected_ra_assignments/assignment_0001.json").is_file()
    assert not (tmp_path / "resources/xarm6@localhost").exists()


def test_unpaired_snapshot_prevents_ra_dispatch(tmp_path: Path) -> None:
    persist_native_completion_fixture(tmp_path)
    state_path = tmp_path / "resources/xarm6@localhost/robot_state/snapshot_0001.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text("{}\n", encoding="utf-8")
    runtime = _AssignedContextRuntime()

    with pytest.raises(RAContextHandoffError, match="revisions are unpaired"):
        asyncio.run(activate_selected_ra_context(runtime, tmp_path))

    assert runtime.assignments == []


def test_runtime_failure_leaves_only_assignment_audit(tmp_path: Path) -> None:
    persist_native_completion_fixture(tmp_path)
    runtime = _AssignedContextRuntime(failure=RuntimeError("RA unavailable"))

    with pytest.raises(RuntimeError, match="RA unavailable"):
        asyncio.run(activate_selected_ra_context(runtime, tmp_path))

    assert len(runtime.assignments) == 1
    assert (tmp_path / "composition/selected_ra_assignments/assignment_0001.json").is_file()
    assert not (tmp_path / "resources/xarm6@localhost").exists()


def _valid_catalog() -> list[dict[str, Any]]:
    return [
        {
            "primitive_symbol": "detect_parts",
            "operation_description": (
                "Detect parts via perception service. Optionally filter by part name."
            ),
            "typed_parameters": [{"name": "part_name", "type": "string", "required": False}],
            "typed_results": [{"name": "parts", "type": "array"}],
            "invocation_binding": "detect_parts",
            "truthful_limits": ["requires the configured perception service"],
            "direct_evidence": [],
            "evaluator_endpoints": [],
            "conditions": {},
            "effects": {},
        },
        {
            "primitive_symbol": "move_pose",
            "operation_description": (
                "Move end-effector to an absolute pose with explicit quaternion orientation."
            ),
            "typed_parameters": [
                {"name": "x", "type": "number", "required": True},
                {"name": "y", "type": "number", "required": True},
                {"name": "z", "type": "number", "required": True},
                {"name": "qx", "type": "number", "required": True},
                {"name": "qy", "type": "number", "required": True},
                {"name": "qz", "type": "number", "required": True},
                {"name": "qw", "type": "number", "required": True},
                {"name": "speed", "type": "number", "required": False},
            ],
            "typed_results": [
                {"name": "success", "type": "boolean"},
                {"name": "message", "type": "string"},
            ],
            "invocation_binding": "move_pose",
            "truthful_limits": [
                "requires a usable controller",
                "does not prove IK or collision feasibility before validation",
            ],
            "direct_evidence": [],
            "evaluator_endpoints": [],
            "conditions": {},
            "effects": {"current_pose": {"pose_absolute_from_params": ["x", "y", "z"]}},
        },
    ]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value
