"""Shared helpers for the contract-driven hybrid DES event model.

These helpers normalize legacy bridge event payloads and derive internal
runtime semantics from symbolic state deltas instead of action labels.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


RESOURCE_ID_FIELDS = {"resource_jid", "id", "name"}
PART_ID_FIELDS = {"part_name", "id", "name"}
LEGACY_EVENT_FIELDS = {"action_type", "target_ref", "derived_state_labels", "effect_kind"}
_OBSERVED_POSE_TOKEN = "observed_pose"


def _string_token(value: Any) -> str:
    return str(value or "").strip()


def _lower_token(value: Any) -> str:
    return _string_token(value).lower()


def _is_observed_pose_ref(location_ref: Any, *, part_name: str = "") -> bool:
    token = _lower_token(location_ref)
    if not token:
        return False
    if token == _OBSERVED_POSE_TOKEN:
        return True
    part_token = _lower_token(part_name)
    return bool(part_token) and token == f"{part_token}_observed_pose"


def event_location_ref(event: dict[str, Any]) -> str:
    """Return the normalized direction-neutral grounded anchor for an event."""
    return _string_token(event.get("location_ref") or event.get("target_ref"))


def normalize_event_contract(
    raw_event: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Normalize an event payload to the current bridge contract.

    Returns the normalized event and a list of legacy field names that were
    accepted only for transitional compatibility.
    """
    event = deepcopy(dict(raw_event or {}))
    legacy_fields: list[str] = []

    location_ref = event_location_ref(event)
    if location_ref:
        event["location_ref"] = location_ref
    if "target_ref" in event:
        legacy_fields.append("target_ref")
        event.pop("target_ref", None)

    for field_name in ("action_type", "derived_state_labels", "effect_kind"):
        if field_name in event:
            legacy_fields.append(field_name)
            event.pop(field_name, None)

    if not isinstance(event.get("projected_effect"), dict):
        event.pop("projected_effect", None)

    return event, sorted(set(legacy_fields))


def normalize_state_metadata_entry(
    raw_entry: Any,
) -> tuple[dict[str, Any], list[str]]:
    """Normalize state audit metadata and drop legacy semantic labels."""
    if not isinstance(raw_entry, dict):
        return {}, []

    entry = deepcopy(raw_entry)
    legacy_fields: list[str] = []

    atomic_bindings = dict(entry.get("atomic_bindings") or {})
    normalized: dict[str, Any] = {
        "atomic_bindings": {
            _string_token(label): bool(value)
            for label, value in atomic_bindings.items()
            if _string_token(label)
        }
    }
    marking_predicate = _string_token(entry.get("marking_predicate"))
    if marking_predicate:
        normalized["marking_predicate"] = marking_predicate

    if "derived_state_labels" in entry:
        legacy_fields.append("derived_state_labels")

    return normalized, legacy_fields


def _filtered_updates(raw_updates: Any, *, id_field: str) -> dict[str, Any]:
    if not isinstance(raw_updates, dict):
        return {}
    return {
        _string_token(field): deepcopy(value)
        for field, value in raw_updates.items()
        if _string_token(field) and _string_token(field) not in {id_field, "id", "name"}
    }


def _iter_effect_rows(raw_rows: Any, *, id_field: str) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    if isinstance(raw_rows, dict):
        for raw_id, raw_updates in raw_rows.items():
            entity_id = _string_token(raw_id)
            updates = _filtered_updates(raw_updates, id_field=id_field)
            if entity_id and updates:
                rows.append((entity_id, updates))
    elif isinstance(raw_rows, list):
        for raw_row in raw_rows:
            if not isinstance(raw_row, dict):
                continue
            entity_id = _string_token(raw_row.get(id_field) or raw_row.get("id"))
            updates = _filtered_updates(raw_row.get("updates") or raw_row, id_field=id_field)
            if entity_id and updates:
                rows.append((entity_id, updates))
    return rows


def explicit_projected_effect(event: dict[str, Any]) -> dict[str, Any]:
    effect = event.get("projected_effect")
    return dict(effect) if isinstance(effect, dict) else {}


def apply_explicit_projected_effect(
    event: dict[str, Any],
    *,
    resources: dict[str, dict[str, Any]],
    parts: dict[str, dict[str, Any]],
) -> bool:
    """Apply declarative projected_effect deltas to resource/part snapshots."""
    effect = explicit_projected_effect(event)
    if not effect:
        return False

    resource_jid = _string_token(event.get("resource_jid"))
    part_name = _string_token(event.get("part_name"))
    applied = False

    resource_updates = _filtered_updates(effect.get("resource"), id_field="resource_jid")
    if resource_jid and resource_updates:
        resource_row = resources.setdefault("{}".format(resource_jid), {"resource_jid": resource_jid})
        resource_row.update(resource_updates)
        applied = True

    part_updates = _filtered_updates(effect.get("part"), id_field="part_name")
    if part_name and part_updates:
        part_row = parts.setdefault(part_name, {"part_name": part_name})
        part_row.update(part_updates)
        applied = True

    for entity_id, updates in (
        _iter_effect_rows(effect.get("resources"), id_field="resource_jid")
        + _iter_effect_rows(effect.get("other_resources"), id_field="resource_jid")
    ):
        resource_row = resources.setdefault(entity_id, {"resource_jid": entity_id})
        resource_row.update(updates)
        applied = True

    for entity_id, updates in (
        _iter_effect_rows(effect.get("parts"), id_field="part_name")
        + _iter_effect_rows(effect.get("other_parts"), id_field="part_name")
    ):
        part_row = parts.setdefault(entity_id, {"part_name": entity_id})
        part_row.update(updates)
        applied = True

    return applied


def apply_event_contract_effects(
    event: dict[str, Any],
    *,
    resources: dict[str, dict[str, Any]],
    parts: dict[str, dict[str, Any]],
) -> bool:
    """Project successor symbolic state from predecessor state plus event contract."""
    if apply_explicit_projected_effect(event, resources=resources, parts=parts):
        return True

    resource_jid = _string_token(event.get("resource_jid"))
    part_name = _string_token(event.get("part_name"))
    location_ref = event_location_ref(event)
    pose = event.get("pose")

    resource_row = resources.get(resource_jid)
    if resource_jid and resource_row is None:
        resource_row = {"resource_jid": resource_jid}
        resources[resource_jid] = resource_row

    if not part_name:
        if resource_row and location_ref:
            resource_row["current_location"] = location_ref
            held_part = _string_token(resource_row.get("held_part"))
            if held_part and held_part in parts:
                parts[held_part]["current_location"] = location_ref
            return True
        return isinstance(pose, dict) and bool(pose)

    part_row = parts.get(part_name)
    if part_row is None:
        part_row = {"part_name": part_name}
        parts[part_name] = part_row

    current_holder = _string_token(part_row.get("current_holder_resource_jid"))
    current_part_location = _string_token(part_row.get("current_location"))
    gripper_location = f"{resource_jid}_gripper" if resource_jid else ""
    observed_ref = _is_observed_pose_ref(location_ref, part_name=part_name)

    if current_holder and current_holder != resource_jid and resource_jid:
        previous_holder_row = resources.get(current_holder)
        if isinstance(previous_holder_row, dict) and _string_token(previous_holder_row.get("held_part")) == part_name:
            previous_holder_row["held_part"] = None
            if "gripper_state" in previous_holder_row:
                previous_holder_row["gripper_state"] = "open"
        resource_row["held_part"] = part_name
        resource_row["gripper_state"] = "closed"
        part_row["current_holder_resource_jid"] = resource_jid
        part_row["current_location"] = gripper_location or current_part_location
        return True

    if current_holder == resource_jid and resource_jid:
        if location_ref and location_ref not in {gripper_location, current_part_location}:
            resource_row["held_part"] = None
            resource_row["gripper_state"] = "open"
            part_row["current_holder_resource_jid"] = None
            part_row["current_location"] = location_ref
            return True
        if location_ref:
            resource_row["current_location"] = location_ref
            part_row["current_location"] = gripper_location or location_ref
            return True
        return False

    if resource_jid and (location_ref or current_part_location or isinstance(pose, dict)):
        resource_row["held_part"] = part_name
        resource_row["gripper_state"] = "closed"
        part_row["current_holder_resource_jid"] = resource_jid
        if observed_ref or current_part_location or location_ref:
            part_row["current_location"] = gripper_location or current_part_location or location_ref
        return True

    if isinstance(pose, dict):
        part_row["observed_pose"] = deepcopy(pose)
        return True
    return False


def _row_delta(before: dict[str, Any], after: dict[str, Any], *, ignored_fields: set[str]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for field_name in sorted(set(before) | set(after)):
        if field_name in ignored_fields:
            continue
        if before.get(field_name) != after.get(field_name):
            delta[field_name] = {"before": deepcopy(before.get(field_name)), "after": deepcopy(after.get(field_name))}
    return delta


def infer_event_semantics(
    event: dict[str, Any],
    *,
    pre_resources: dict[str, dict[str, Any]],
    pre_parts: dict[str, dict[str, Any]],
    post_resources: dict[str, dict[str, Any]],
    post_parts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Infer internal runtime semantics from predecessor/successor symbolic state."""
    resource_jid = _string_token(event.get("resource_jid"))
    part_name = _string_token(event.get("part_name"))
    location_ref = event_location_ref(event)

    pre_resource = dict(pre_resources.get(resource_jid) or {})
    post_resource = dict(post_resources.get(resource_jid) or {})
    pre_part = dict(pre_parts.get(part_name) or {}) if part_name else {}
    post_part = dict(post_parts.get(part_name) or {}) if part_name else {}

    before_holder = _string_token(pre_part.get("current_holder_resource_jid"))
    after_holder = _string_token(post_part.get("current_holder_resource_jid"))
    before_part_location = _string_token(pre_part.get("current_location"))
    after_part_location = _string_token(post_part.get("current_location"))
    before_resource_location = _string_token(pre_resource.get("current_location"))
    after_resource_location = _string_token(post_resource.get("current_location"))

    resource_delta = _row_delta(pre_resource, post_resource, ignored_fields=RESOURCE_ID_FIELDS)
    part_delta = _row_delta(pre_part, post_part, ignored_fields=PART_ID_FIELDS)

    category = "observation"
    if part_name:
        if after_holder == resource_jid and before_holder and before_holder != resource_jid:
            category = "transfer"
        elif after_holder == resource_jid and before_holder != resource_jid:
            category = "acquisition"
        elif before_holder == resource_jid and after_holder != resource_jid:
            category = "release"
        elif (
            before_holder == resource_jid
            and after_holder == resource_jid
            and (
                before_resource_location != after_resource_location
                or before_part_location != after_part_location
            )
        ):
            category = "carry"
        elif resource_delta and not part_delta:
            category = "resource_only_transition"
        elif part_delta:
            category = "observation"
    elif resource_delta:
        category = "resource_only_transition"

    effect_scope = "resource_and_part"
    if resource_delta and not part_delta:
        effect_scope = "resource_only"
    elif part_delta and not resource_delta:
        effect_scope = "part_only"
    elif not resource_delta and not part_delta:
        effect_scope = "observation"

    return {
        "category": category,
        "effect_scope": effect_scope,
        "resource_jid": resource_jid or None,
        "part_name": part_name or None,
        "location_ref": location_ref or None,
        "resource_before": deepcopy(pre_resource),
        "resource_after": deepcopy(post_resource),
        "part_before": deepcopy(pre_part),
        "part_after": deepcopy(post_part),
        "resource_delta": resource_delta,
        "part_delta": part_delta,
    }


def snapshot_signature(
    *,
    resources: dict[str, dict[str, Any]],
    parts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "resources": deepcopy(resources),
        "parts": deepcopy(parts),
    }


__all__ = [
    "apply_event_contract_effects",
    "apply_explicit_projected_effect",
    "event_location_ref",
    "explicit_projected_effect",
    "infer_event_semantics",
    "normalize_event_contract",
    "normalize_state_metadata_entry",
    "snapshot_signature",
]
