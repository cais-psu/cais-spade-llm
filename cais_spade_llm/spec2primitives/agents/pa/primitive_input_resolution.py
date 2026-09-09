from __future__ import annotations

"""Resolve declared primitive needs from accepted associations, without authoring steps."""

import json
from collections.abc import Mapping
from typing import Any

from ..ra.composition_context import _resolve_json_pointer
from ..ra.refinement_records import fingerprint
from ..ra.validation_scope import GAZEBO_OBSERVED_SCOPE


class _GroundingPrerequisite(ValueError):
    """An accepted task association is missing or contradictory."""


def _feature_source(
    target: Mapping[str, Any], records: Mapping[str, Any], part_name: str, *, destination: bool,
) -> dict[str, Any]:
    associations = [association for association in target.get("assembly_feature_association", [])
                    if any(feature.get("owner", {}).get("name") == part_name
                           and feature.get("state_name") == "current_state"
                           for feature in association.get("assembly_features", []))]
    if len(associations) != 1:
        raise _GroundingPrerequisite(f"Grounding must establish one accepted assembly relationship for {part_name!r}.")
    features = [feature for feature in associations[0]["assembly_features"]
                if (feature.get("state_name") == "desired_state" if destination else
                    feature.get("state_name") == "current_state" and feature.get("owner", {}).get("name") == part_name)]
    role = "destination" if destination else "part"
    if len(features) != 1:
        raise _GroundingPrerequisite(f"Grounding must establish one accepted {role} feature for {part_name!r}.")
    feature = features[0]
    states = [state for state in target.get("resolved_state_values", [])
              if state.get("state") == feature["state_name"] and state.get("name") == feature["state_value_name"]]
    if len(states) != 1 or not isinstance(states[0].get("resolved_value"), dict):
        raise _GroundingPrerequisite(f"Grounding must bind {feature['name']!r} to one observed candidate.")
    state = states[0]
    ref = state.get("value_ref", {}).get("record_ref")
    segmentation = records.get(ref, {})
    handle = state["resolved_value"].get("candidate_handle")
    matches = [(camera, candidate) for camera in segmentation.get("cameras", [])
               for candidate in camera.get("candidates", []) if candidate.get("candidate_handle") == handle]
    if segmentation.get("record_type") != "RGBDSegmentationRecord" or len(matches) != 1:
        raise _GroundingPrerequisite(f"The accepted {role} candidate is absent or duplicated in its issued observation.")
    camera, candidate = matches[0]
    if candidate != state["resolved_value"]:
        raise _GroundingPrerequisite(f"The issued {role} observation contradicts its accepted candidate association.")
    return {"feature": feature, "segmentation_ref": ref, "candidate_handle": handle,
            "observation_handle": camera["observation_handle"], "camera": camera}


def _measurement_request(root: Any, producer: Any, records: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    from .primitive_context import _compatible_calibrations
    from ...tools.observation_presentation import ObservationPresentation

    ref, handle = source["segmentation_ref"], source["observation_handle"]
    camera, stamp = producer.observation(ref, handle)
    if camera.get("support_plane", {}).get("status") != "detected":
        raise _GroundingPrerequisite("The selected observation needs a measured support plane before observed bounds can be calculated.")
    locations = [record for record in records.values()
                 if record.get("record_type") == "RobotFrameLocationRecord"
                 and record.get("location") == "available" and record.get("robot_frame_conversion") == "accepted"
                 and record.get("target_frame") == "world"
                 and record.get("source_segmentation", {}).get("ref") == ref
                 and record.get("candidate_reference", {}).get("candidate_handle") == source["candidate_handle"]
                 and record.get("candidate_reference", {}).get("observation_handle") == handle]
    compatible = _compatible_calibrations(root, records, camera, stamp)
    selected = set(record.get("source_calibration", {}).get("ref") for record in locations)
    if selected and (len(selected) != 1 or not selected.issubset(compatible)):
        raise _GroundingPrerequisite("Accepted location evidence has conflicting or incompatible calibration links.")
    if not selected:
        selected = set(compatible)
    if len(selected) != 1:
        raise _GroundingPrerequisite(
            "The selected observation requires one approved compatible calibration; "
            "surface_calibration is available when an approved calibration can be supplied."
        )
    return {"tool_name": "observed_geometry", "arguments": json.dumps(ObservationPresentation(root).project({
        "segmentation_ref": ref, "observation_handle": handle, "calibration_ref": next(iter(selected)),
    }))}


def _observed_records(records: Mapping[str, Any], source: Mapping[str, Any]) -> list[tuple[str, Any]]:
    return [(ref, record) for ref, record in records.items()
            if record.get("record_type") == "ObservedGeometryEvidence"
            and record.get("status") == "accepted" and record.get("frame_id") == "world"
            and record.get("units") == "m"
            and record.get("segmentation_ref") == source["segmentation_ref"]
            and record.get("candidate_reference", {}).get("candidate_handle") == source["candidate_handle"]
            and record.get("candidate_reference", {}).get("observation_handle") == source["observation_handle"]]


def _one_value(records: list[tuple[str, Any]], pointer: str) -> str | None:
    candidates = []
    for ref, record in records:
        try:
            value = _resolve_json_pointer(record, pointer)
        except (KeyError, ValueError):
            continue
        if value is not None:
            candidates.append((ref, value))
    if len({fingerprint(value) for _, value in candidates}) > 1:
        raise _GroundingPrerequisite(f"Issued measurements conflict for {pointer}; the approved source must be resolved.")
    return min(ref for ref, _ in candidates) if candidates else None


def _scene_inputs(
    target: Mapping[str, Any], producer: Any, records: Mapping[str, Any], names: set[str],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Cover every candidate in the accepted part and destination observations."""
    if not names:
        raise _GroundingPrerequisite("Scene coverage requires an accepted part association.")
    views = {}
    for name in sorted(names):
        for destination in (False, True):
            source = _feature_source(target, records, name, destination=destination)
            views[(source["segmentation_ref"], source["observation_handle"])] = source
    geometry_refs, surface_refs, operations = [], [], {}
    for (segmentation_ref, handle), source in sorted(views.items()):
        selected = []
        for candidate in source["camera"]["candidates"]:
            matches = _observed_records(records, {**source, "candidate_handle": candidate["candidate_handle"]})
            for pointer in ("/reference_pose", "/size_m", "/product_geometry", "/placement_surface_point"):
                _one_value(matches, pointer)
            if not matches:
                break
            # Custody binding does not change a box; keep its original source stable.
            matches.sort(key=lambda item: item[0])
            selected.append(matches[0][0])
        if len(selected) != len(source["camera"]["candidates"]):
            operation = _measurement_request(producer.root, producer, records, source)
            operations[fingerprint(operation)] = operation
            continue
        ancestors, pending = set(), list(selected)
        while pending:
            ref = pending.pop()
            if ref in ancestors:
                continue
            ancestors.add(ref)
            pending.extend(item["ref"] for item in records.get(ref, {}).get("source_refs", []))
        planes = {ref for ref in ancestors if records.get(ref, {}).get("record_type") == "AssemblySurfaceEvidence"}
        patches = [(ref, record) for ref, record in records.items()
                   if record.get("record_type") == "AssemblySurfaceEvidence"
                   and record.get("status") == "accepted" and record.get("coverage") == "observed_plane_patch"
                   and record.get("frame_id") == "world" and record.get("units") == "m"
                   and any(item["ref"] in planes for item in record.get("source_refs", []))]
        if not patches:
            raise _GroundingPrerequisite(f"Observation {handle!r} has no measured finite support patch.")
        signatures = {fingerprint({key: record.get(key) for key in (
            "normal", "offset_m", "observation_timestamp_ns",
        )} | {"mesh_sha256": record.get("mesh", {}).get("sha256")}) for _, record in patches}
        if len(signatures) != 1:
            raise _GroundingPrerequisite(f"Observation {handle!r} has conflicting support patches.")
        geometry_refs.extend(selected)
        surface_refs.append(min(ref for ref, _ in patches))
    if operations:
        return None, list(operations.values())
    arguments = {"geometry_refs": sorted(set(geometry_refs)), "surface_refs": sorted(set(surface_refs)),
                 "segmentation_refs": sorted({ref for ref, _ in views})}
    return {"tool_name": "scene_geometry", "arguments": json.dumps(arguments, sort_keys=True)}, []


def resolve_primitive_inputs(
    request: Mapping[str, Any], producer: Any, records: Mapping[str, Any],
    checked: Mapping[str, Any], attempted: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return checked-source selections and uniquely determined measurement requests.

    The caller validates and persists answers and executes requests through the same
    bounded tool runner as PA. No primitive or parameter is added or changed here.
    """
    if request.get("validation_scope") != GAZEBO_OBSERVED_SCOPE or not request.get("primitive_steps"):
        return [], []
    steps, target = request["primitive_steps"], request["target_feature"]
    answers, operations = [], {}
    names = {step.get("params", {}).get("part_name") for step in steps
             if step.get("primitive_symbol") in {"compute_pick_targets", "grasp_part"}
             and isinstance(step.get("params", {}).get("part_name"), str)}
    for need in request["needs"]:
        if need["need_id"] in checked:
            continue
        index, quantity = need.get("step_index"), need["quantity"]
        if index is None and quantity == "scene":
            selected = request.get("validation_refs", {}).get("scene", {}).get("ref")
            scene = records.get(selected, {})
            if (scene.get("record_type") == "AssemblySceneEvidence" and scene.get("status") == "accepted"
                    and scene.get("geometry_model") == "observed_bounds" and scene.get("declared_observations")
                    and scene.get("coverage") == "all_observed_candidates" and not scene.get("unresolved_candidates")):
                answers.append({"need_id": need["need_id"], "record_ref": selected})
                continue
            try:
                operation, measurements = _scene_inputs(target, producer, records, names)
                if operation:
                    arguments = json.loads(operation["arguments"])
                    matching = [ref for ref, record in records.items()
                                if record.get("record_type") == "AssemblySceneEvidence"
                                and record.get("status") == "accepted"
                                and record.get("geometry_model") == "observed_bounds"
                                and record.get("coverage") == "all_observed_candidates"
                                and not record.get("unresolved_candidates")
                                and {item["ref"] for item in record.get("source_refs", [])}
                                == {ref for refs in arguments.values() for ref in refs}]
                    if matching:
                        answers.append({"need_id": need["need_id"], "record_ref": min(matching)})
                    else:
                        measurements.append(operation)
                for operation in measurements:
                    key = fingerprint(operation)
                    if key not in attempted:
                        operations[key] = operation
            except _GroundingPrerequisite as exc:
                answers.append({"need_id": need["need_id"], "blocked": str(exc)})
            continue
        role = index is None and quantity == "part"
        if index is None and not role:
            continue
        step = steps[index - 1] if index is not None and 0 < index <= len(steps) else {}
        symbol = step.get("primitive_symbol")
        pointer, destination = None, False
        if symbol == "compute_pick_targets":
            if quantity in {"/product_geometry", "/product_geometry/board_center/z", "/product_geometry/part_height_m"}:
                pointer = quantity
            elif quantity in {"/target_pose", "/target_pose/x", "/target_pose/y", "/target_pose/z"}:
                pointer = quantity.replace("/target_pose", "/reference_pose", 1)
        elif symbol == "compute_place_targets":
            if quantity.startswith("/product_geometry/placement_surface_point"):
                pointer, destination = quantity.removeprefix("/product_geometry"), True
            elif quantity == "/product_geometry/part_height_m":
                pointer = quantity
        if not role and pointer is None:
            continue
        name = step.get("params", {}).get("part_name") if not role else next(iter(names)) if len(names) == 1 else None
        if not isinstance(name, str):
            continue
        try:
            source = _feature_source(target, records, name, destination=destination)
            candidates = _observed_records(records, source)
            selected = _one_value(candidates, pointer or "/product_geometry")
            if selected is None:
                operation = _measurement_request(producer.root, producer, records, source)
            elif role:
                bound = [(ref, value) for ref, value in candidates if value.get("part_name") == name
                         and value.get("accepted_feature_name") == source["feature"]["name"]]
                if bound:
                    answers.append({"need_id": need["need_id"], "record_ref": _one_value(bound, "/product_geometry")})
                    continue
                owner_refs = source["feature"]["owner"]["evidence_refs"]
                cad_refs = {record["CAD"]["record"]["ref"] for ref, record in records.items()
                            if ref in owner_refs and record.get("record_type") == "CADSizeCorrespondenceRecord"
                            and record.get("CAD", {}).get("context_ref") in owner_refs}
                if len(cad_refs) != 1 or not cad_refs.issubset(records):
                    raise _GroundingPrerequisite("The accepted part feature needs one issued CAD identity association.")
                operation = {"tool_name": "bind_observed_part", "arguments": json.dumps({
                    "part_ref": selected, "cad_ref": next(iter(cad_refs)), "feature_name": source["feature"]["name"],
                })}
            else:
                answers.append({"need_id": need["need_id"], "value_ref": {"record_ref": selected, "field_path": pointer}})
                continue
            key = fingerprint(operation)
            if key not in attempted:
                operations[key] = operation
        except _GroundingPrerequisite as exc:
            answers.append({"need_id": need["need_id"], "blocked": str(exc)})
    return answers, list(operations.values())
