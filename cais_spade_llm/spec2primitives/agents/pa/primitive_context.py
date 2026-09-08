from __future__ import annotations

"""Investigate Phase 5 product needs without reassigning or rewriting Phase 4."""

import asyncio
import json
import time
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..ra.composition_context import _composition_state_view, _without_model_name
from ..ra.refinement_records import append_record, owned_path, pin, read_pin
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
    "inspect_features": ["cad_ref", "pose_ref"],
    "bind_part": ["part_ref", "feature_name"],
    "observed_surface": ["segmentation_ref", "observation_handle", "calibration_ref"],
    "pick_geometry": ["part_ref", "surface_ref"],
    "assembly_geometry": [
        "part_ref",
        "part_plane_id",
        "part_circle_id",
        "target_ref",
        "target_plane_id",
        "target_circle_id",
    ],
    "estimate_pose": ["correspondence_ref"],
    "convert_pose": ["pose_ref"],
    "surface_calibration": ["segmentation_ref", "observation_handle"],
    "read_record": ["record_ref", "field_path"],
    "scene_geometry": ["geometry_refs", "surface_refs", "segmentation_refs"],
    "document_quantity": ["record_ref", "field_path", "start", "end", "quantity"],
    "validation_specification": ["position_quantity_ref", "axis_quantity_ref"],
}


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
    ) -> dict[str, Any]:
        """Return PA-selected evidence pins and unresolved needs; never author a program."""
        root = interaction_root.resolve()
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
                {"name": name, "arguments": arguments}
                for name, arguments in _GEOMETRY_TOOLS.items()
            ],
        ]
        exchanges = []
        operations = 0
        for turn in range(max_operations + 1):
            evidence_catalog = _current_evidence_catalog(
                tuple(investigation.handles.values()), investigation
            )
            for ref, sha in authorized.items():
                record = await asyncio.to_thread(read_pin, root, {"ref": ref, "sha256": sha})
                record = _without_model_name(_composition_state_view(record))
                # Discovery metadata includes earlier partial records without selecting
                # a measurement or promoting its status to accepted geometry.
                evidence_catalog.append(
                    {
                        "record_ref": ref,
                        **{
                            key: record[key]
                            for key in ("record_type", "status", "pose", "conversion")
                            if key in record
                        },
                        "fields": list(record),
                    }
                )
            prompt = (
                "You are ProductAgent investigating missing product/scene facts for an already accepted target_feature. "
                "Use only approved documents, CAD, RGB-D and calibration. Task roles and mating-feature associations "
                "remain your decision; geometry producers return measurements, not task answers. Do not author "
                "primitive sequences, robot coordinates, simulator identifiers, or invented measurements. "
                "The accepted identities, desired relationship and resource assignment cannot be changed here. "
                "Return authority_conflict if new evidence invalidates them. Use an issued record reference exactly. "
                "Use finish with only selected issued records and unresolved needs. Raw CAD and a destination "
                "label do not establish final placement geometry. Seating tolerances require approved evidence "
                "or an explicitly supplied experiment specification. Missing facts remain unresolved. "
                "Review every request in needs, including exact missing inputs derived from RA-selected "
                "primitive contracts and RA-authored supplemental requests. Reuse applicable issued records, including partial "
                "evidence from earlier batches; read_record can inspect the fields listed in evidence_catalog. "
                "When one requested measurement is blocked, consider useful supported investigations for the "
                "remaining requests. Choose the tools and their order yourself. Before finish, explain each "
                "unresolved request using its exact supplied quantity and step_index. Distinguish a request "
                "you did not investigate from a measurement you attempted that remains ambiguous or unavailable. "
                "Finish early when no useful supported operation remains; you need not spend every operation "
                "or repeat a blocked measurement with unchanged inputs. "
                "Geometry tool arguments are JSON objects with the listed fields. estimate_pose consumes an "
                "issued CADSizeCorrespondenceRecord and returns a camera-frame CADPoseEstimationRecord. "
                "convert_pose consumes that CADPoseEstimationRecord and returns a RobotFramePoseRecord in world "
                "using approved calibration; conversion preserves any pose ambiguity. "
                "inspect_features requires a CADMeshRecord cad_ref and a RobotFramePoseRecord pose_ref for "
                "that same CAD record. A correspondence record or camera-frame pose is not a valid pose_ref. "
                "Select and invoke the required tools explicitly; no conversion is performed automatically. "
                "inspect_features returns neutral planar/circular CAD feature candidates, or an ambiguous "
                "record that may contain part_height_m estimated from the highest-ranked qualified "
                "pose for the same observed candidate, with its source and warning. The height estimate "
                "is usable as that scalar input; it does not establish an oriented feature or full pose. "
                "assembly_geometry consumes your selected mating circles and opposing seating planes. "
                "bind_part associates a measured candidate with one exact accepted assembly feature name. "
                "finish includes validation_refs: part (bound observed geometry), goal (derived assembly geometry), "
                "scene (observed collision scene) and specification (documented tolerances); use null for missing roles. "
                "Retain useful issued partial records in evidence_refs even when a validation role remains "
                "incomplete. Ambiguous poses and height estimates alone do not establish complete part, "
                "goal or scene evidence; keep those validation_refs null and explain the unresolved facts. "
                "No sequence, recovery example, or evaluator answer is available.\n\n"
                + json.dumps(
                    ObservationPresentation(root).project(
                        {
                            "target_feature": request["target_feature"],
                            "needs": request["needs"],
                            "evidence_catalog": evidence_catalog,
                            "issued_records": list(authorized),
                            "tools": tool_descriptions,
                            "operations_remaining": max_operations - operations,
                            "exchanges": exchanges,
                        }
                    ),
                    ensure_ascii=False,
                )
            )
            response = await self.product_agent.ask_llm_structured(
                prompt, response_format=_response_format(), tools=None, max_tool_rounds=0
            )
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
                    "unresolved": action["unresolved"],
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
                        ref = investigation._canonical_by_pa_ref.get(ref, ref)
                        if ref not in refs:
                            raise ValueError(
                                "Validation evidence must be explicitly selected in evidence_refs."
                            )
                        result["validation_refs"][role] = {"ref": ref, "sha256": authorized[ref]}
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
                try:
                    arguments = json.loads(action["arguments"])
                    if not isinstance(arguments, dict):
                        raise ValueError("Tool arguments must be an object.")
                    name = action["tool_name"]
                    if name in _GEOMETRY_TOOLS:
                        if set(arguments) != set(_GEOMETRY_TOOLS[name]):
                            raise ValueError("Geometry tool argument fields are invalid.")
                        canonical = ObservationPresentation(root).resolve(arguments)

                        def resolve_alias(value: Any) -> Any:
                            if isinstance(value, str):
                                return investigation._canonical_by_pa_ref.get(value, value)
                            if isinstance(value, list):
                                return [resolve_alias(item) for item in value]
                            if isinstance(value, dict):
                                return {key: resolve_alias(item) for key, item in value.items()}
                            return value

                        canonical = resolve_alias(canonical)
                        result = await asyncio.to_thread(self._geometry, producer, name, canonical)
                    else:
                        if name not in {tool["function"]["name"] for tool in tools}:
                            raise ValueError("The requested evidence producer is not available.")
                        result = dict(await investigation.execute(name, arguments))
                        # Only records actually issued by the native investigation
                        # are admitted; no filesystem crawl expands model authority.
                        authorized.update(investigation._issued_record_hashes)
                    result = ObservationPresentation(root).project(
                        _without_model_name(_composition_state_view(result))
                    )
                except (
                    KeyError,
                    StopIteration,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as exc:
                    result = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
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
                kind in {"finish", "authority_conflict"}
                or operations >= max_operations
                and kind != "investigate"
            ):
                return result
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
        from ..ra.composition_context import _resolve_json_pointer
        from ...tools.rgb_d_cad_grounding.frame_conversion import (
            transform_camera_pose_to_robot_frame,
        )
        from ...tools.rgb_d_cad_grounding.pose_estimation import estimate_camera_frame_pose

        root = producer.root
        if name == "read_record":
            value = producer.read(arguments["record_ref"])
            pointer = arguments["field_path"]
            value = _resolve_json_pointer(value, pointer) if pointer else value
            if len(json.dumps(value)) > 12000:
                return {
                    "error": "Read a narrower field_path.",
                    "fields": list(value)[:32] if isinstance(value, dict) else [],
                }
            return {"value": value}
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
            record = producer.read(source_ref)
            if name == "convert_pose":
                frame = record["coordinate_frame"]
            else:
                frame = next(
                    camera["frame"]
                    for camera in record["cameras"]
                    if camera["observation_handle"] == arguments["observation_handle"]
                )
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


def _response_format() -> dict[str, Any]:
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
