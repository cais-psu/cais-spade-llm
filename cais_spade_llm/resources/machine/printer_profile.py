"""Built-in printer resource profile and printer-specific bridge helpers."""

from __future__ import annotations

from copy import deepcopy
from textwrap import dedent
from typing import Any

from cais_spade_llm.resources.resource_profile import (
    ResourceProfile,
    register_resource_profile,
    resource_snapshot_field_value,
)


def _availability_from_state(
    raw_snapshot: dict[str, Any],
    current_state: str,
    *,
    busy_states: set[str],
) -> str:
    availability = str(raw_snapshot.get("availability", "") or "").strip().lower()
    if availability:
        return availability
    normalized_state = str(current_state or "").strip().lower()
    if normalized_state in {"faulted", "down", "offline", "error"}:
        return "unavailable"
    if normalized_state in busy_states:
        return "busy"
    return "available"


def _printer_availability(raw_snapshot: dict[str, Any], current_state: str) -> str:
    return _availability_from_state(
        raw_snapshot,
        current_state,
        busy_states={"busy", "printing", "running", "paused"},
    )


def _printer_facet(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "active_job": resource_snapshot_field_value(snapshot, "active_job"),
        "job_state": (
            resource_snapshot_field_value(snapshot, "job_state")
            or resource_snapshot_field_value(snapshot, "current_state")
        ),
        "material_state": resource_snapshot_field_value(snapshot, "material_state"),
        "bed_state": resource_snapshot_field_value(snapshot, "bed_state"),
    }


def _printer_occupancy(current_location: Any, snapshot: dict[str, Any]) -> dict[str, Any]:
    occupancy: dict[str, Any] = {}
    if current_location not in (None, ""):
        occupancy["location"] = deepcopy(current_location)
    active_job = resource_snapshot_field_value(snapshot, "active_job")
    if active_job not in (None, ""):
        occupancy["active_job"] = deepcopy(active_job)
    return occupancy


def _printer_event_family(event: dict[str, Any]) -> str:
    event_name = str(event.get("event_name", "") or "").strip().lower()
    if event_name in {"pause_job", "resume_job", "cancel_job"}:
        return event_name
    if event_name.startswith("pause_job") or event_name.startswith("pause_"):
        return "pause_job"
    if event_name.startswith("resume_job") or event_name.startswith("resume_"):
        return "resume_job"
    if event_name.startswith("cancel_job") or event_name.startswith("cancel_"):
        return "cancel_job"
    return ""


def _printer_event_contract_validator(
    *,
    event: dict[str, Any],
    resource_jid: str,
    **_: Any,
) -> str | None:
    operation_family = str(event.get("operation_family", "") or "").strip().lower()
    event_name = str(event.get("event_name", "") or "").strip() or resource_jid or operation_family
    part_name = str(event.get("part_name", "") or "").strip()
    part_delta = dict(event.get("expected_part_delta") or {})

    if operation_family in {"pause_job", "resume_job", "cancel_job"}:
        if part_name:
            return (
                f"bridge event '{event_name}' is inconsistent for printer resources: "
                f"'{operation_family}' must not declare part_name"
            )
        if part_delta:
            return (
                f"bridge event '{event_name}' is inconsistent for printer resources: "
                f"'{operation_family}' must not declare expected_part_delta"
            )
        return None

    if part_name or part_delta:
        return (
            f"bridge event '{event_name}' is inconsistent for printer resources: "
            "manipulator-style part transitions are not supported by this profile"
        )
    return None


_PRINTER_PROMPT_ADDENDUM = dedent(
    """\
    JOB CONTROL ADDENDUM:
    - This addendum applies only when the chosen resource exposes job
      control primitives such as pause_job, resume_job, cancel_job.
    - Each job control primitive is a single-step macro.
    - Use expected_resource_delta to declare the state transition.
    """
).strip()


_PRINTER_REPAIR_EXAMPLE = dedent(
    """\
    JOB CONTROL EXAMPLE:
    - A job-control bridge event typically compiles to a single primitive step.
    """
).strip()


PRINTER_PROFILE = ResourceProfile(
    resource_type="printer",
    snapshot_fields=("current_state", "active_job", "job_state"),
    facet_key="printer",
    facet_builder=_printer_facet,
    occupancy_builder=_printer_occupancy,
    availability_resolver=_printer_availability,
    primitive_owner_resolver=lambda agent: agent if getattr(agent, "_BRIDGE_PRIMITIVES", None) else None,
    sync_map={
        "active_job": "_active_job",
        "job_state": "_job_state",
        "material_state": "_material_state",
        "bed_state": "_bed_state",
    },
    primitive_kind_map={
        "pause_job": "job_control",
        "resume_job": "job_control",
        "cancel_job": "job_control",
    },
    event_family_resolver=_printer_event_family,
    event_contract_validator=_printer_event_contract_validator,
    family_to_primitive={
        "pause_job": "pause_job",
        "resume_job": "resume_job",
        "cancel_job": "cancel_job",
    },
    capability_flags={"supports_printer_job_control": True},
    example_families=("generic_bridge", "printer_job_control"),
    prompt_addendum=_PRINTER_PROMPT_ADDENDUM,
    repair_example=_PRINTER_REPAIR_EXAMPLE,
)


register_resource_profile(PRINTER_PROFILE)
