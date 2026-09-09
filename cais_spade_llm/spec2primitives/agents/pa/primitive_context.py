from __future__ import annotations

"""Investigate Phase 5 product needs without reassigning or rewriting Phase 4."""

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..ra.composition_context import _composition_state_view, _without_model_name
from ..ra.refinement_records import append_record, owned_path, pin, read_pin
from ..ra.validation_scope import (
    GAZEBO_OBSERVED_SCOPE,
    VALIDATION_SCOPE,
    read_validation_scope,
    required_validation_roles,
    validation_scope_instruction,
)
from .context_interaction import ProductAgentContextRuntime
from .presentation_records import load_or_create_evidence_presentation
from .product_context import load_interaction_abox
from .production_grounding import (
    ProductionProductContextGroundingRuntime,
    _NativeEvidenceInvestigation,
    _analyze_candidate_layout_tool,
    _approved_evidence_handles,
    _approved_evidence_sources,
    _compare_cad_size_tool,
    _current_evidence_catalog,
    _next_number,
    _query_document_tool,
    _retrieve_tool,
)
from ...tools.assembly_geometry import AssemblyGeometryProducer
from ...tools.observation_presentation import ObservationPresentation

_GEOMETRY_TOOLS = {
    "observed_geometry": {
        "arguments": {
            "segmentation_ref": "Issued RGBDSegmentationRecord.",
            "observation_handle": "PA-selected observation handle; all candidates in this view are measured.",
            "calibration_ref": "Compatible issued world-frame calibration for this observation.",
        },
        "description": (
            "For the observed-geometry Gazebo scope, measure candidate boxes from calibrated RGB-D "
            "points extended to this view's measured support plane. Returns geometry_refs for every "
            "candidate and surface_ref for its finite observed support patch. Each ObservedGeometryEvidence "
            "contains reference_pose (observed_bounds_center), part_height_m, product_geometry.board_center.z "
            "and placement_surface_point at its observed top. Metres in world; CAD orientation remains "
            "unknown and occluded geometry unmeasured. PA selects which measured point/surface has a task "
            "role. A unique CAD pose is not needed. One operation measures the selected view."
        ),
    },
    "bind_observed_part": {
        "arguments": {
            "part_ref": "PA-selected ObservedGeometryEvidence for the accepted candidate.",
            "cad_ref": "Issued CADMeshRecord for its already accepted CAD identity.",
            "feature_name": "Exact accepted assembly feature name for the selected part.",
        },
        "description": (
            "Bind observed bounds to the accepted part identity for the part validation role. "
            "Checks candidate identity and CAD owner without claiming a CAD pose or physical grasp fit. "
            "Use the bound record in the scene geometry list in place of its unbound duplicate."
        ),
    },
    "inspect_features": {
        "arguments": {
            "cad_ref": "Issued CADMeshRecord reference.",
            "pose_ref": "Issued RobotFramePoseRecord for the same CAD and observed candidate.",
        },
        "description": (
            "Measure neutral planar/circular CAD features using the observed pose. Returns "
            "AssemblyGeometryEvidence in world metres with origin_pose, part_height_m, bounds_m, "
            "mesh and features when the pose is accepted. An ambiguous pose may supply only a "
            "part_height_m estimate from its highest-ranked qualified hypothesis, with source "
            "and warning, plus uncertainty.qualified_height_estimates_m and "
            "qualified_height_range_m measured across every qualified hypothesis. "
            "These partial measurements cannot establish full part, goal or scene evidence. "
            "Camera-frame poses and correspondence records are not valid pose_ref inputs."
        ),
    },
    "bind_part": {
        "arguments": {
            "part_ref": "Accepted observed AssemblyGeometryEvidence reference.",
            "feature_name": "Exact feature name from target_feature.assembly_feature_association.",
        },
        "description": (
            "Associate measured geometry with one accepted assembly feature, checking its "
            "observed candidate and CAD owner. Returns bound AssemblyGeometryEvidence for "
            "the part validation role or a selected assembly_geometry input. It does not "
            "measure a missing pose or choose a feature association."
        ),
    },
    "observed_surface": {
        "arguments": {
            "segmentation_ref": "Issued RGBDSegmentationRecord reference.",
            "observation_handle": "Selected cameras entry's exact observation_handle.",
            "calibration_ref": "Issued CameraToRobotCalibrationRecord for that camera and observation time.",
        },
        "description": (
            "Transform a detected cameras entry's support_plane into the calibrated frame. "
            "Returns AssemblySurfaceEvidence with a unit normal, offset_m and fitting "
            "uncertainty; distances are in metres. "
            "observation_catalog lists every view and compatible issued calibrations; "
            "read_observation can inspect a selected support_plane. A support plane can be measured even when no "
            "board_center or accepted object pose record exists. The plane's task role "
            "remains PA's decision; an undetected plane cannot supply a measurement."
        ),
    },
    "surface_height": {
        "arguments": {
            "surface_ref": "Accepted world-frame AssemblySurfaceEvidence with normal and offset_m.",
            "location_ref": "Issued calibrated RobotFrameLocationRecord at which to evaluate height.",
        },
        "description": (
            "Measure the selected support plane's world Z at the selected location's XY, "
            "without needing a unique CAD-origin pose. Returns AssemblySurfaceEvidence with "
            "product_geometry.board_center.z in metres and the exact source pins. A plane's "
            "offset_m is not generally its world height. PA chooses whether the plane and "
            "location support the requested primitive input; this does not assign a task role "
            "or establish full part, mating, scene or tolerance evidence."
        ),
    },
    "select_grasp_point": {
        "arguments": {
            "part_ref": "Accepted registered AssemblyGeometryEvidence for the selected part.",
            "plane_id": "PA-selected measured features entry's plane_id.",
            "circle_id": "PA-selected circle_id within that plane.",
        },
        "description": (
            "Bind the centre of a selected measured circular CAD feature as a grasp reference. "
            "Returns the same registered part geometry and CAD-origin pose with grasp_reference "
            "containing point_CAD_m and point_world_m in metres. The CAD file origin can be "
            "outside the physical part and is not automatically a grasp point. PA chooses the "
            "feature; this operation does not prove grasp success or resolve an ambiguous pose. "
            "Use this selected part record consistently for pick and mating derivations so "
            "placement can retain the measured grasp offset from the CAD origin."
        ),
    },
    "pick_geometry": {
        "arguments": {
            "part_ref": "Accepted AssemblyGeometryEvidence with CAD_origin, origin_pose and part_height_m.",
            "surface_ref": "Accepted AssemblySurfaceEvidence selected as the object's support surface.",
        },
        "description": (
            "Derive pick inputs from object and support evidence in the same frame and metres. "
            "Returns AssemblyGeometryEvidence with target_pose at the selected grasp reference "
            "(or CAD origin when it lies within the physical part) and "
            "product_geometry.board_center.z from the measured support plane, plus "
            "product_geometry.part_height_m. Requires a vertical-compatible support plane. "
            "An observed candidate centre or raw CAD height cannot replace these inputs."
        ),
    },
    "assembly_geometry": {
        "arguments": {
            "part_ref": "Accepted bound AssemblyGeometryEvidence for the part.",
            "part_plane_id": "Selected part features entry's plane_id.",
            "part_circle_id": "Selected circle_id within that part plane.",
            "target_ref": "Accepted bound AssemblyGeometryEvidence for the mating target.",
            "target_plane_id": "Selected target features entry's plane_id.",
            "target_circle_id": "Selected circle_id within that target plane.",
        },
        "description": (
            "Derive the final CAD origin by aligning PA-selected mating circles and opposing "
            "seating planes from the same accepted assembly relationship. Returns goal "
            "AssemblyGeometryEvidence with target_origin_pose and product_geometry containing "
            "board_center.x/y, slot_xy, slot_floor_z_m, part_height_m, target_reference and "
            "target_origin_pose, in world metres. Requires compatible vertical geometry; "
            "destination labels and observed shaft centres do not establish seating geometry. "
            "When the part has a selected grasp_reference, product_geometry also contains "
            "grasp_point_offset_world_m and target_reference.grasp_point; RA must retain "
            "these fields when binding placement geometry. This tool supplies no acceptance tolerances."
        ),
    },
    "estimate_pose": {
        "arguments": {"correspondence_ref": "Issued CADSizeCorrespondenceRecord reference."},
        "description": (
            "Register the selected CAD against its observed RGB-D candidate. Returns a "
            "camera-frame CADPoseEstimationRecord with pose hypotheses and uncertainty. "
            "A correspondence alone does not establish pose; registration may remain ambiguous."
        ),
    },
    "convert_pose": {
        "arguments": {"pose_ref": "Issued camera-frame CADPoseEstimationRecord reference."},
        "description": (
            "Use approved camera calibration to return a RobotFramePoseRecord in world metres. "
            "Preserves pose ambiguity and provenance; conversion is not performed automatically "
            "by inspect_features. Fails when approved calibration is unavailable."
        ),
    },
    "surface_calibration": {
        "arguments": {
            "segmentation_ref": "Issued RGBDSegmentationRecord reference.",
            "observation_handle": "Selected cameras entry's exact observation_handle.",
        },
        "description": (
            "Materialize approved calibration for the selected camera, returning a "
            "CameraToRobotCalibrationRecord from its frame to world for observed_surface. "
            "Requires configured approved calibration; it does not estimate calibration "
            "or obtain simulator geometry."
        ),
    },
    "read_record": {
        "arguments": {
            "record_ref": "Exact issued record reference from evidence_catalog.",
            "field_path": "RFC 6901 JSON pointer; empty selects the root.",
        },
        "description": (
            "Read hash-verified existing evidence without acquiring measurements. Returns "
            "the selected value, preserving its units, frame and status. Results over 12000 "
            "characters require a narrower field_path. Use read_observation for a selected "
            "view; camera array positions and internal sensor identities are not exposed. "
            "An absent derived field does not prove source measurements "
            "are unavailable."
        ),
    },
    "read_observation": {
        "arguments": {
            "segmentation_ref": "Exact issued RGBDSegmentationRecord reference.",
            "observation_handle": "Exact opaque observation_handle from observation_catalog.",
            "field_path": "JSON pointer relative to the selected view; empty selects its permitted metadata.",
        },
        "description": (
            "Look up exactly the selected observation without enumerating camera positions. "
            "Returns its selected value and compatible issued calibration references. "
            "For example /support_plane reads the existing measured plane; it does not "
            "assign a support role or produce world geometry. The 12000-character value "
            "limit and source checks apply. This is one evidence read, not a new measurement."
        ),
    },
    "scene_geometry": {
        "arguments": {
            "geometry_refs": "List of accepted registered AssemblyGeometryEvidence, or ObservedGeometryEvidence in the observed scope.",
            "surface_refs": "List of accepted AssemblySurfaceEvidence references.",
            "segmentation_refs": "List of issued RGBDSegmentationRecord references defining observed coverage.",
        },
        "description": (
            "Build AssemblySceneEvidence from selected meshes, poses and support planes in "
            "the same frame and metres. Complete coverage requires geometry for every "
            "candidate in the selected segmentation records (or each declared view for observed boxes). "
            "Observed boxes need no CAD pose; preserve all candidates in each selected view. Missing candidates remain "
            "unresolved and status incomplete; unobserved space is unmodeled."
        ),
    },
    "document_quantity": {
        "arguments": {
            "record_ref": "Issued document record reference.",
            "field_path": "RFC 6901 pointer to a document text string.",
            "start": "Integer start character offset of the selected numeric span.",
            "end": "Integer exclusive end character offset of that span.",
            "quantity": "Exact quantity whose meaning is supported by that document span.",
        },
        "description": (
            "Extract one explicitly unit-bearing number as AssemblyQuantityEvidence. "
            "Supported units are mm, m, rad, deg and °; outputs use metres or radians. "
            "PA selects the quantity's meaning. A part dimension is not an assembly "
            "acceptance tolerance; missing documented tolerances cannot be invented."
        ),
    },
    "validation_specification": {
        "arguments": {
            "position_quantity_ref": "AssemblyQuantityEvidence for positive position_tolerance_m in metres.",
            "axis_quantity_ref": "AssemblyQuantityEvidence for positive axis_tolerance_rad in radians.",
        },
        "description": (
            "Build AssemblyValidationSpecification for vertical_gear_assembly using the "
            "selected documented acceptance quantities. It supplies no default tolerances "
            "and does not cover yaw-specific assembly, threading or force control."
        ),
    },
}


_INTERNAL_OBSERVATION_FIELDS = {
    "camera_id", "camera_name", "candidate_id", "calibration_id",
    "camera_order", "camera_index", "candidate_order", "candidate_index",
    "source_artifacts", "source_point_cloud", "label_mask_artifact",
}
_CAMERA_POSITION = re.compile(r"/cameras/[0-9]+(?:/|$)")
_GEOMETRY_OPERATIONS = {
    "observed_geometry",
    "estimate_pose", "inspect_features", "observed_surface", "surface_height", "select_grasp_point", "pick_geometry",
    "assembly_geometry", "scene_geometry",
}


def _issued_records(producer: AssemblyGeometryProducer) -> dict[str, Any]:
    return {
        ref: read_pin(producer.root, {"ref": ref, "sha256": sha})
        for ref, sha in producer.authorized.items()
    }


def _pa_evidence_projection(root: Path, value: Any, records: Mapping[str, Any]) -> Any:
    """Keep physical measurements while removing internal camera identity cues."""
    identities: set[str] = set()

    def collect(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if key == "extracted_text":
                    continue
                if key in {"camera_id", "camera_name", "calibration_id", "frame", "source_frame", "coordinate_frame"}:
                    if isinstance(nested, str) and nested and nested not in {"world", "CAD_local"}:
                        identities.add(nested)
                collect(nested)
        elif isinstance(item, list):
            for nested in item:
                collect(nested)

    collect(records)
    collect(value)
    presentation = ObservationPresentation(root)
    value = presentation.project(_without_model_name(_composition_state_view(value)))

    def visit(item: Any, field: str = "") -> Any:
        if field == "extracted_text":
            return item
        if isinstance(item, str):
            for identity in sorted(identities, key=len, reverse=True):
                item = item.replace(identity, "[internal camera metadata]")
            return re.sub(r"/cameras/[0-9]+", "/cameras/[select observation_handle]", item)
        if isinstance(item, Mapping):
            return {
                key: visit(nested, str(key))
                for key, nested in item.items()
                if key not in _INTERNAL_OBSERVATION_FIELDS
                and not (
                    key in {"frame", "source_frame", "coordinate_frame"}
                    and nested not in ("world", "CAD_local")
                )
            }
        if isinstance(item, list):
            values = [visit(nested, field) for nested in item]
            if field == "candidate_handles":
                values.sort()
            if field in {"cameras", "observation_catalog"}:
                values.sort(key=lambda nested: (
                    str(nested.get("observation_handle", "")), str(nested.get("segmentation_ref", ""))
                ) if isinstance(nested, Mapping) else str(nested))
            return values
        return item

    return visit(value)


def _compatible_calibrations(
    root: Path, records: Mapping[str, Any], camera: Mapping[str, Any], stamp: int
) -> list[str]:
    from ...tools.rgb_d_cad_grounding.frame_conversion import (
        RobotFrameConversionError, _load_calibration,
    )

    matches = []
    for ref, record in records.items():
        if (
            record.get("record_type") != "CameraToRobotCalibrationRecord"
            or record.get("source_frame") != camera.get("frame")
            or record.get("target_frame") != "world"
        ):
            continue
        try:
            _load_calibration(root, owned_path(root, ref), observation_timestamp_ns=stamp)
        except RobotFrameConversionError:
            # A pinned calibration can still be inapplicable at this observation time.
            continue
        matches.append(ref)
    return matches


def _observation_catalog(root: Path, records: Mapping[str, Any]) -> list[dict[str, Any]]:
    catalog = []
    for ref, record in records.items():
        if record.get("record_type") != "RGBDSegmentationRecord":
            continue
        source = read_pin(root, record["source_record"])
        stamps = {camera["camera_id"]: camera["depth_timestamp_ns"] for camera in source["cameras"]}
        seen = set()
        for camera in record["cameras"]:
            handle = camera["observation_handle"]
            if handle in seen:
                raise ValueError("An observation handle is duplicated in the issued segmentation.")
            seen.add(handle)
            catalog.append({
                "segmentation_ref": ref,
                "observation_handle": handle,
                "support_plane": deepcopy(camera["support_plane"]),
                "candidate_handles": [candidate["candidate_handle"] for candidate in camera["candidates"]],
                "compatible_calibration_refs": _compatible_calibrations(
                    root, records, camera, stamps[camera["camera_id"]]
                ),
            })
    return catalog


def _read_pa_value(root: Path, record: Any, pointer: str, records: Mapping[str, Any]) -> Any:
    from ..ra.composition_context import _resolve_json_pointer

    # Positional reads defeat the opaque observation interface even if the value is filtered.
    if _CAMERA_POSITION.search(pointer):
        raise ValueError("Select an observation_handle with read_observation; camera positions are internal.")
    fields = pointer.split("/")[1:]
    if any(field in _INTERNAL_OBSERVATION_FIELDS or field in {"frame", "source_frame"} for field in fields):
        raise ValueError("Internal camera metadata is not a measurement; use the observation catalog.")
    if pointer.startswith("/candidates/observation_"):
        canonical = ObservationPresentation(root).resolve({"field_path": pointer})["field_path"]
        value = _pa_evidence_projection(root, _resolve_json_pointer(record, canonical), records)
    else:
        projected = _pa_evidence_projection(root, record, records)
        value = _resolve_json_pointer(projected, pointer) if pointer else projected
    if len(json.dumps(value)) > 12000:
        return {"error": "Read a narrower field_path.", "fields": list(value)[:32] if isinstance(value, dict) else []}
    return {"value": value}


class ProductPrimitiveContextRuntime:
    """Use the existing ProductAgent boundary for a bounded supplemental investigation."""

    def __init__(
        self,
        runtime: ProductionProductContextGroundingRuntime,
        product_agent: ProductAgentContextRuntime,
    ) -> None:
        """Retain the already-authorized evidence producers and selected PA model."""
        self.runtime, self.product_agent = runtime, product_agent

    async def investigate(
        self,
        *,
        interaction_root: Path,
        directory: Path,
        request: Mapping[str, Any],
        max_operations: int,
        progress: Callable[[Mapping[str, Any]], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """Investigate selected needs and return evidence without authoring a program.

        Args:
            interaction_root: Completed interaction containing the issued sources.
            directory: Append-only directory for this investigation's trace and evidence.
            request: Unresolved inputs, PA validation findings, and authorized evidence pins.
            max_operations: Maximum PA-selected evidence operations for this batch.
            progress: Optional asynchronous consumer of model and operation progress.

        Returns:
            Selected evidence pins, unresolved PA findings, and consumed operations.
        """
        root = interaction_root.resolve()
        scope = read_validation_scope(request)

        async def emit(message: str) -> None:
            if progress is not None:
                await progress({
                    "message": message, "operations_used": operations,
                    "geometry_operations": geometry_operations,
                })

        investigation = await asyncio.to_thread(self._investigation, root, directory / "native")
        authorized = {
            reference["ref"]: reference["sha256"] for reference in request["evidence_refs"]
        }
        from ..ra.refinement_records import verify_evidence_tree

        for ref, sha in authorized.items():
            await asyncio.to_thread(verify_evidence_tree, root, {"ref": ref, "sha256": sha})
        producer = AssemblyGeometryProducer(
            root, directory / "geometry", authorized, request["target_feature"]
        )
        tools = [
            _retrieve_tool(tuple(investigation.handles.values())),
            _query_document_tool(tuple(investigation.handles.values())),
            _compare_cad_size_tool(tuple(investigation.handles.values())),
            _analyze_candidate_layout_tool(),
        ]
        tool_descriptions = [
            *tools,
            *[
                {"name": name, **contract}
                for name, contract in _GEOMETRY_TOOLS.items()
                if scope == GAZEBO_OBSERVED_SCOPE or name not in {"observed_geometry", "bind_observed_part"}
            ],
        ]
        exchanges = []
        operations = 0
        geometry_operations = 0
        finish_corrections = 0
        partial_result = None
        for turn in range(max_operations + 1 + (scope == GAZEBO_OBSERVED_SCOPE)):
            records = await asyncio.to_thread(_issued_records, producer)
            observations = await asyncio.to_thread(_observation_catalog, root, records)
            evidence_catalog = _current_evidence_catalog(
                tuple(investigation.handles.values()), investigation
            )
            for ref, record in records.items():
                record = _pa_evidence_projection(root, record, records)
                # Discovery metadata includes earlier partial records without selecting
                # a measurement or promoting its status to accepted geometry.
                evidence_catalog.append(
                    {
                        "record_ref": ref,
                        **{
                            key: record[key]
                            for key in ("record_type", "status", "pose", "conversion", "CAD_correspondence", "location")
                            if key in record
                        },
                        "fields": list(record),
                        **({"cad_context_ref": record["CAD"]["context_ref"]}
                           if isinstance(record.get("CAD"), dict) and "context_ref" in record["CAD"] else {}),
                        **({"candidate_reference": record["candidate_reference"]}
                           if isinstance(record.get("candidate_reference"), dict) else {}),
                        **({"selected_candidate": {
                            key: record["selected_candidate"][key]
                            for key in ("observation_handle", "candidate_handle")
                            if key in record["selected_candidate"]
                        }} if isinstance(record.get("selected_candidate"), dict) else {}),
                    }
                )
            prompt = (
                validation_scope_instruction(scope) +
                "You are ProductAgent investigating missing product/scene facts for an already accepted target_feature. "
                "Use only approved documents, CAD, RGB-D and calibration. Task roles and mating-feature associations "
                "remain your decision; geometry producers return measurements, not task answers. Do not author "
                "primitive sequences, robot coordinates, simulator identifiers, or invented measurements. "
                "The accepted identities, desired relationship and resource assignment cannot be changed here. "
                "Return authority_conflict if new evidence invalidates them. Use an issued record reference exactly. "
                "Use finish with only selected issued records and unresolved needs. Raw CAD and a destination "
                "label do not establish final placement geometry. Missing facts remain unresolved. "
                "Review every request in needs, including missing or rejected measurement inputs derived "
                "from RA-selected primitive contracts, PA-owned validation findings and RA-authored "
                "supplemental requests. A null step_index identifies a program-level validation need; "
                "its quantity names the failed check. A rejected binding remains unchanged until RA "
                "revises it; you may investigate evidence for that quantity without replacing the binding. "
                "Reuse applicable issued records, including partial "
                "evidence from earlier batches; read_record can inspect the fields listed in evidence_catalog. "
                "observation_catalog contains every issued view, its measured support_plane and "
                "compatible_calibration_refs checked against its frame and measurement time. "
                "Use an exact observation_handle with read_observation when more detail is needed. "
                "No compatible issued calibration does not prove approved calibration is unavailable; "
                "surface_calibration can request one for your selected observation. "
                "An absent derived record is not an unavailable source measurement. Consider the listed "
                "measurement tools and their prerequisites before declaring a requested quantity unavailable. "
                "Reading locations and raw CAD dimensions alone does not investigate an observed support "
                "plane, mating geometry or scene coverage. If a measurement cannot be attempted, identify "
                "the specific missing prerequisite rather than only the absent output record. "
                "When one requested measurement is blocked, consider useful supported investigations for the "
                "remaining requests. Choose the tools and their order yourself. Before finish, explain each "
                "unresolved request using its exact supplied quantity and step_index. Distinguish a request "
                "you did not investigate from a measurement you attempted that remains ambiguous or unavailable. "
                + ("For finish in this scope, provide exactly one outcome per requested (step_index, quantity). "
                   "Use provided with an issued record_ref and field_path, blocked with the specific missing "
                   "prerequisite, or not_investigated. An early unresolved finish with remaining budget receives "
                   "one continuation with the actual operation count. Continue useful investigation or confirm "
                   "the missing source prerequisite; the host does not choose a tool for you. "
                   "A missing derived output alone is not a missing source prerequisite. "
                   "Observed bounds satisfy this scope's part/scene representation without unique CAD poses. "
                   if scope == GAZEBO_OBSERVED_SCOPE else "") +
                "Finish early when no useful supported operation remains; you need not spend every operation "
                "or repeat a blocked measurement with unchanged inputs. "
                "Geometry tool arguments are JSON objects with exactly the listed argument fields. "
                "Tool descriptions state required records, outputs, units and limitations; they prescribe no order. "
                "Select and invoke the required tools explicitly; no conversion is performed automatically. "
                "RA alone binds returned evidence and revises primitive_steps; do not supply replacement steps. "
                + ("finish retains four validation_refs keys: part (identity-bound observed box for simulated attachment and "
                   if scope == GAZEBO_OBSERVED_SCOPE else
                   "finish retains four validation_refs keys: part (bound observed CAD geometry for grasp and ") +
                "carried-part checks), goal (derived mating geometry), scene (observed collision coverage), "
                "and specification (documented assembly tolerances). Supply only the roles required by "
                "the recorded scope; use null for other or unavailable roles. These validation roles are "
                "distinct from individual helper inputs. "
                "Retain useful issued partial records in evidence_refs even when a validation role remains "
                "incomplete. Ambiguous poses and height estimates alone do not establish complete part, "
                "goal or scene evidence; keep those validation_refs null and explain the unresolved facts. "
                "No sequence, recovery example, or evaluator answer is available.\n\n"
                + json.dumps(
                    _pa_evidence_projection(
                        root,
                        {
                            "validation_scope": scope,
                            "target_feature": request["target_feature"],
                            "needs": request["needs"],
                            "evidence_catalog": evidence_catalog,
                            "observation_catalog": observations,
                            "issued_records": list(authorized),
                            "tools": tool_descriptions,
                            "operations_remaining": max_operations - operations,
                            "exchanges": exchanges,
                        },
                        records,
                    ),
                    ensure_ascii=False,
                )
            )
            await emit(
                f"PA request {turn + 1}: waiting for model response; "
                f"{operations}/{max_operations} operations completed. Prompt: {len(prompt)} characters."
            )
            model_started = time.monotonic()
            response = await self.product_agent.ask_llm_structured(
                prompt, response_format=_response_format(scope), tools=None, max_tool_rounds=0
            )
            await emit(f"PA request {turn + 1}: model response received in {time.monotonic() - model_started:.2f} s.")
            action = response.get("action")
            result: dict[str, Any]
            if not isinstance(action, Mapping):
                raise ValueError("PA context response requires one action.")
            kind = action.get("kind")
            if kind == "finish":
                refs = action.get("evidence_refs")
                if isinstance(refs, list):
                    refs = [investigation._canonical_by_pa_ref.get(ref, ref) for ref in refs]
                if not isinstance(refs, list) or any(
                    not isinstance(ref, str) or ref not in authorized for ref in refs
                ):
                    raise ValueError("PA supplied a context record it was not issued.")
                result = {
                    "status": "provided" if refs else "unavailable",
                    "operations_used": operations,
                    "evidence_refs": [
                        {"ref": ref, "sha256": authorized[ref]} for ref in dict.fromkeys(refs)
                    ],
                    "unresolved": action.get("unresolved", []),
                    "validation_refs": {},
                }
                roles = action["validation_refs"]
                if not isinstance(roles, dict) or set(roles) != {
                    "part",
                    "goal",
                    "scene",
                    "specification",
                }:
                    raise ValueError("PA validation evidence roles are invalid.")
                for role, ref in roles.items():
                    if ref is not None:
                        if role not in required_validation_roles(scope):
                            raise ValueError(f"Validation role {role} must remain null in scope {scope}.")
                        ref = investigation._canonical_by_pa_ref.get(ref, ref)
                        if ref not in refs:
                            raise ValueError(
                                "Validation evidence must be explicitly selected in evidence_refs."
                            )
                        result["validation_refs"][role] = {"ref": ref, "sha256": authorized[ref]}
                if scope == GAZEBO_OBSERVED_SCOPE:
                    expected = {(need["step_index"], need["quantity"]) for need in request["needs"]}
                    seen = set()
                    unresolved = []
                    for outcome in action["outcomes"]:
                        key = (outcome["step_index"], outcome["quantity"])
                        if key not in expected or key in seen:
                            raise ValueError("PA outcome must identify exactly one supplied input need.")
                        seen.add(key)
                        state = outcome["status"]
                        if state == "provided":
                            ref, pointer = outcome["record_ref"], outcome["field_path"]
                            ref = investigation._canonical_by_pa_ref.get(ref, ref)
                            if ref not in refs or not isinstance(pointer, str):
                                raise ValueError("Provided PA quantities require a selected issued record and pointer.")
                            value = _read_pa_value(root, producer.read(ref), pointer, records)
                            if "value" not in value:
                                raise ValueError("The selected PA quantity is unavailable or exceeds the read limit.")
                        else:
                            if state not in {"blocked", "not_investigated"} or not outcome["reason"].strip():
                                raise ValueError("An unresolved PA outcome needs a status and specific prerequisite.")
                            unresolved.append(f"step_index {key[0]}, quantity {key[1]}: {state}: {outcome['reason']}")
                    if seen != expected:
                        raise ValueError("PA must account for every requested input quantity.")
                    result["unresolved"] = unresolved
                    partial_result = deepcopy(result)
                    # A blocked label can mistakenly report an exhausted budget or only
                    # a missing derived record. Return the real allowance once; PA still
                    # decides whether evidence is obtainable and may confirm its blocker.
                    if unresolved and operations < max_operations and finish_corrections == 0:
                        finish_corrections += 1
                        result = {"status": "continue", "operations_remaining": max_operations - operations,
                                  "message": (
                                      f"{operations}/{max_operations} evidence operations used; "
                                      f"{max_operations - operations} remain. The operation budget is not exhausted. "
                                      "Review unresolved needs, including those labeled blocked. Continue useful "
                                      "investigation or finish again with a specific unavailable source prerequisite. "
                                      "An absent derived record alone does not establish unavailable source evidence."
                                  ),
                                  "unresolved": unresolved}
                        await emit(result["message"])
            elif kind == "authority_conflict":
                result = {
                    "status": "authority_conflict",
                    "reason": action["reason"],
                    "operations_used": operations,
                    "evidence_refs": [],
                    "unresolved": [],
                }
            elif kind == "investigate" and operations < max_operations:
                operations += 1
                name = str(action.get("tool_name", ""))
                operation_kind = (
                    "read" if name in {"read_record", "read_observation"}
                    else "measurement/geometry" if name in _GEOMETRY_OPERATIONS else "evidence"
                )
                operation_started = time.monotonic()
                await emit(f"PA operation {operations}/{max_operations}: starting {operation_kind} {name}.")
                try:
                    arguments = json.loads(action["arguments"])
                    if not isinstance(arguments, dict):
                        raise ValueError("Tool arguments must be an object.")
                    name = action["tool_name"]
                    if name in _GEOMETRY_TOOLS:
                        if set(arguments) != set(_GEOMETRY_TOOLS[name]["arguments"]):
                            raise ValueError("Geometry tool argument fields are invalid.")
                        canonical = ObservationPresentation(root).resolve(arguments)
                        if name == "read_record":
                            # Preserve proof that a candidate pointer was opaque, not a guessed camera index.
                            canonical["field_path"] = arguments["field_path"]

                        def resolve_alias(value: Any) -> Any:
                            if isinstance(value, str):
                                return investigation._canonical_by_pa_ref.get(value, value)
                            if isinstance(value, list):
                                return [resolve_alias(item) for item in value]
                            if isinstance(value, dict):
                                return {key: resolve_alias(item) for key, item in value.items()}
                            return value

                        canonical = resolve_alias(canonical)
                        geometry_operations += name in _GEOMETRY_OPERATIONS
                        result = await asyncio.to_thread(self._geometry, producer, name, canonical)
                    else:
                        if name not in {tool["function"]["name"] for tool in tools}:
                            raise ValueError("The requested evidence producer is not available.")
                        result = dict(await investigation.execute(name, arguments))
                        # Only records actually issued by the native investigation
                        # are admitted; no filesystem crawl expands model authority.
                        authorized.update(investigation._issued_record_hashes)
                except (
                    KeyError,
                    StopIteration,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as exc:
                    result = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
                result = _pa_evidence_projection(root, result, records)
                outcome_status = (
                    "unavailable" if "error" in result else result.get("status")
                    or result.get("record", {}).get("status")
                    or result.get("record", {}).get("pose") or "completed"
                )
                await emit(
                    f"PA operation {operations}/{max_operations}: {operation_kind} {name} "
                    f"{outcome_status} in {time.monotonic() - operation_started:.2f} s."
                )
            else:
                result = {
                    "status": "budget_exhausted",
                    "operations_used": operations,
                    "evidence_refs": [],
                    "unresolved": ["PA evidence operation budget exhausted."],
                }
            append_record(
                root,
                directory,
                f"exchange_{turn + 1:04d}.json",
                {
                    "record_type": "PrimitiveContextExchange",
                    "prompt": prompt,
                    "response": deepcopy(response),
                    "result": deepcopy(result),
                    "created_at_ns": time.time_ns(),
                },
            )
            exchanges.append({"response": response, "result": result})
            if (
                (kind in {"finish", "authority_conflict"} and result.get("status") != "continue")
                or operations >= max_operations
                and kind != "investigate"
            ):
                await emit(
                    f"PA investigation ended: {operations}/{max_operations} operations used; "
                    + (f"{geometry_operations} measurement/geometry operations attempted."
                       if geometry_operations else "No measurement/geometry operations were attempted.")
                )
                return result
        if partial_result is not None:
            return {**partial_result, "operations_used": operations}
        return {
            "status": "budget_exhausted",
            "operations_used": operations,
            "evidence_refs": [],
            "unresolved": ["PA investigation did not select evidence before its budget ended."],
        }

    def _investigation(self, root: Path, directory: Path) -> _NativeEvidenceInvestigation:
        tbox = self.runtime._tbox
        abox = load_interaction_abox(root, tbox)
        presentation = load_or_create_evidence_presentation(
            root,
            sources=_approved_evidence_sources(),
            explicit_order=self.runtime._evidence_presentation_order,
        )
        return _NativeEvidenceInvestigation(
            runtime=self.runtime,
            interaction_root=root,
            tbox=tbox,
            abox=abox,
            requirement=abox.product_requirement,
            handles=_approved_evidence_handles(presentation),
            presentation=presentation,
            supplemental_directory=directory,
        )

    def _geometry(
        self, producer: AssemblyGeometryProducer, name: str, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        from ...tools.rgb_d_cad_grounding.frame_conversion import (
            transform_camera_pose_to_robot_frame,
        )
        from ...tools.rgb_d_cad_grounding.pose_estimation import estimate_camera_frame_pose

        root = producer.root
        if name == "read_record":
            value = producer.read(arguments["record_ref"])
            return _read_pa_value(root, value, arguments["field_path"], _issued_records(producer))
        if name in {"read_observation", "observed_surface", "observed_geometry"}:
            camera, stamp = producer.observation(
                arguments["segmentation_ref"], arguments["observation_handle"]
            )
            records = _issued_records(producer)
            compatible = _compatible_calibrations(root, records, camera, stamp)
            if name == "read_observation":
                return {
                    **_read_pa_value(root, camera, arguments["field_path"], records),
                    "compatible_calibration_refs": compatible,
                }
            producer.read(arguments["calibration_ref"])
            if arguments["calibration_ref"] not in compatible:
                return {
                    "status": "unavailable",
                    "observation_handle": arguments["observation_handle"],
                    "compatible_calibration_refs": compatible,
                    "reason": (
                        "The selected calibration is incompatible with this observation's frame, "
                        "world target, or measurement timestamp. "
                        + ("Compatible issued calibration references are listed." if compatible else
                           "No compatible issued calibration was found. surface_calibration can request "
                           "approved calibration for this selected observation; approval availability "
                           "has not been established by this lookup.")
                    ),
                }
        if name == "estimate_pose":
            producer.read(arguments["correspondence_ref"])
            result = estimate_camera_frame_pose(
                interaction_root=root,
                correspondence_record_path=owned_path(root, arguments["correspondence_ref"]),
                pose_number=_next_number(
                    root, "products/grounding/rgb_d_cad_grounding/pose_*/pose_record.json"
                ),
            )
            reference = pin(root, result.record_path)
            producer.authorized[reference["ref"]] = reference["sha256"]
            return {"record_ref": reference["ref"], "record": result.record}
        if name in {"convert_pose", "surface_calibration"}:
            source_ref = (
                arguments["pose_ref"] if name == "convert_pose" else arguments["segmentation_ref"]
            )
            if name == "convert_pose":
                frame = producer.read(source_ref)["coordinate_frame"]
            else:
                camera, _ = producer.observation(source_ref, arguments["observation_handle"])
                frame = camera["frame"]
            calibration_runtime = self.runtime._camera_to_world_calibration_runtime
            if calibration_runtime is None:
                raise ValueError("Approved camera calibration is unavailable.")
            calibration = calibration_runtime.materialize_camera_to_world_calibration(
                interaction_root=root,
                grounding_record_path=owned_path(root, source_ref),
                source_frame=frame,
                target_frame="world",
                calibration_number=_next_number(
                    root,
                    "products/grounding/rgb_d_cad_grounding/calibration_*/calibration_record.json",
                ),
            )
            reference = pin(root, calibration.record_path)
            producer.authorized[reference["ref"]] = reference["sha256"]
            if name == "surface_calibration":
                return {"record_ref": reference["ref"], "record": calibration.record}
            pose = transform_camera_pose_to_robot_frame(
                interaction_root=root,
                pose_record_path=owned_path(root, source_ref),
                calibration_record_path=calibration.record_path,
                target_frame="world",
                conversion_number=_next_number(
                    root,
                    "products/grounding/rgb_d_cad_grounding/robot_pose_*/robot_frame_pose_record.json",
                ),
            )
            reference = pin(root, pose.record_path)
            producer.authorized[reference["ref"]] = reference["sha256"]
            return {"record_ref": reference["ref"], "record": pose.record}
        return getattr(producer, name)(**arguments)


def _response_format(scope: str = VALIDATION_SCOPE) -> dict[str, Any]:
    text = {"type": "string"}
    variants = []
    for kind, fields in {
        "investigate": {"tool_name": text, "arguments": text},
        "finish": {
            "evidence_refs": {"type": "array", "items": text},
            "unresolved": {"type": "array", "items": text},
            "validation_refs": {
                "type": "object",
                "additionalProperties": False,
                "required": ["part", "goal", "scene", "specification"],
                "properties": {
                    role: {"type": ["string", "null"]}
                    for role in ("part", "goal", "scene", "specification")
                },
            },
        },
        "authority_conflict": {"reason": text},
    }.items():
        if kind == "finish" and scope == GAZEBO_OBSERVED_SCOPE:
            fields.pop("unresolved")
            fields["outcomes"] = {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["step_index", "quantity", "status", "record_ref", "field_path", "reason"],
                "properties": {
                    "step_index": {"type": ["integer", "null"]}, "quantity": text,
                    "status": {"type": "string", "enum": ["provided", "blocked", "not_investigated"]},
                    "record_ref": {"type": ["string", "null"]},
                    "field_path": {"type": ["string", "null"]}, "reason": text,
                },
            }}
        properties = {"kind": {"type": "string", "enum": [kind]}, **fields}
        variants.append(
            {
                "type": "object",
                "additionalProperties": False,
                "required": list(properties),
                "properties": properties,
            }
        )
    return {
        "name": "spec2primitives_primitive_context",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["action"],
            "properties": {"action": {"anyOf": variants}},
        },
    }
