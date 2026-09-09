from __future__ import annotations

"""Investigate Phase 5 product needs without reassigning or rewriting Phase 4."""

import asyncio
import json
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..ra.composition_context import _composition_state_view, _without_model_name
from ..ra.refinement_records import _verify_evidence_tree, append_record, fingerprint, owned_path, pin, read_pin, verify_evidence_tree, verify_record
from ..ra.validation_scope import (
    GAZEBO_OBSERVED_SCOPE,
    read_validation_scope,
    required_validation_roles,
)
from .context_interaction import ProductAgentContextRuntime
from .production_grounding import ProductionProductContextGroundingRuntime
from ...tools.assembly_geometry import AssemblyGeometryProducer, _ObservedGeometryBatch
from ...tools.observation_presentation import ObservationPresentation

_GEOMETRY_ARGUMENTS = {
    "observed_geometry": {"segmentation_ref", "observation_handle", "calibration_ref"},
    "bind_observed_part": {"part_ref", "cad_ref", "feature_name"},
    "scene_geometry": {"geometry_refs", "surface_refs", "segmentation_refs"},
}
_INTERNAL_OBSERVATION_FIELDS = {
    "camera_id", "camera_name", "candidate_id", "calibration_id",
    "camera_order", "camera_index", "candidate_order", "candidate_index",
    "source_artifacts", "source_point_cloud", "label_mask_artifact",
}
_CAMERA_POSITION = re.compile(r"/cameras/[0-9]+(?:/|$)")
_GEOMETRY_OPERATIONS = {"observed_geometry", "scene_geometry"}


def _issued_records(producer: AssemblyGeometryProducer) -> dict[str, Any]:
    return {
        ref: read_pin(producer.root, {"ref": ref, "sha256": sha})
        for ref, sha in producer.authorized.items()
    }


def _verified_issued_records(producer: AssemblyGeometryProducer) -> dict[str, Any]:
    checked: dict[str, tuple[str, Any]] = {}
    for ref, sha in list(producer.authorized.items()):
        _verify_evidence_tree(producer.root, {"ref": ref, "sha256": sha}, checked)
    records = {}
    for ref, (sha, record) in checked.items():
        if isinstance(record, dict):
            # Reusable scene/part pins also authorize their hash-checked measurements.
            producer.authorized[ref] = sha
            records[ref] = record
    return records


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


def _number_needs(needs: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{**deepcopy(need), "need_id": f"need_{index:04d}"} for index, need in enumerate(needs, 1)]



def _checked_selection(
    entry: dict[str, Any], need: dict[str, Any], producer: AssemblyGeometryProducer,
    records: Mapping[str, Any], aliases: Mapping[str, str], scope: str,
) -> dict[str, Any]:
    fields = set(entry) - {"need_id"}
    if fields not in ({"value_ref"}, {"record_ref"}, {"blocked"}):
        raise ValueError("Select exactly one of value_ref, record_ref, or blocked.")
    answer = {"need_id": need["need_id"], "need": deepcopy(need)}
    if "blocked" in entry:
        reason = entry["blocked"]
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("blocked requires a specific missing prerequisite.")
        return {**answer, "blocked": reason}
    role = need["quantity"] if need.get("step_index") is None else None
    if "value_ref" in entry:
        selection = entry["value_ref"]
        if not isinstance(selection, dict) or set(selection) != {"record_ref", "field_path"}:
            raise ValueError("value_ref requires record_ref and a string field_path JSON pointer.")
        ref, pointer = selection["record_ref"], selection["field_path"]
        if not isinstance(pointer, str) or (pointer and not pointer.startswith("/")):
            raise ValueError("A measured value requires a string field_path JSON pointer; use '' for its root.")
        if role in required_validation_roles(scope):
            raise ValueError(f"The {role} validation need requires a whole record_ref without a pointer.")
    else:
        ref = entry["record_ref"]
        if role not in required_validation_roles(scope):
            raise ValueError("Whole record_ref is for a requested validation role; use value_ref for an input.")
    if not isinstance(ref, str):
        raise ValueError("record_ref must be an issued string reference.")
    ref = aliases.get(ref, ref)
    if ref not in producer.authorized:
        raise ValueError(f"Record {ref!r} was not issued; select a reference from issued_records.")
    source = {"ref": ref, "sha256": producer.authorized[ref]}
    record = verify_evidence_tree(producer.root, source)
    if "value_ref" in entry:
        selected = _read_pa_value(producer.root, record, pointer, records)
        if "value" not in selected:
            raise ValueError("Selected value exceeds the 12000-character read limit; select a narrower field_path.")
        if selected["value"] is None:
            raise ValueError("The selected field is null, not an available measured value.")
        answer.update(value_ref={"record_ref": ref, "field_path": pointer}, value=selected["value"])
    else:
        expected = {
            "part": "ObservedGeometryEvidence" if scope == GAZEBO_OBSERVED_SCOPE else "AssemblyGeometryEvidence",
            "goal": "AssemblyGeometryEvidence", "scene": "AssemblySceneEvidence",
            "specification": "AssemblyValidationSpecification",
        }[role]
        if not isinstance(record, dict) or record.get("record_type") != expected:
            raise ValueError(f"The {role} role requires a typed {expected} record.")
        answer["record_ref"] = ref
    return {**answer, "source_ref": source}


def _check_answers(
    entries: Any, needs: list[dict[str, Any]], checked: dict[str, Any],
    producer: AssemblyGeometryProducer, records: Mapping[str, Any],
    aliases: Mapping[str, str], scope: str,
) -> list[dict[str, Any]]:
    """Accept independent selections without discarding an earlier checked answer."""
    if not isinstance(entries, list):
        return [{"status": "rejected", "reason": "answers must be an array of new or revised selections."}]
    by_id = {need["need_id"]: need for need in needs}
    grouped: dict[str, list[Any]] = {}
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("need_id"), str):
            grouped.setdefault(entry["need_id"], []).append(entry)
    conflicts = {key for key, items in grouped.items() if any(item != items[0] for item in items[1:])}
    diagnostics, handled = [], set()
    for index, entry in enumerate(entries):
        diagnostic = {"entry_index": index, "status": "rejected"}
        try:
            if not isinstance(entry, dict) or not isinstance(entry.get("need_id"), str):
                raise ValueError("Each answer requires a string need_id.")
            need_id = entry["need_id"]
            diagnostic["need_id"] = need_id
            if need_id not in by_id:
                raise ValueError(f"Unknown need_id {need_id}; use an ID from needs.")
            if need_id in conflicts:
                raise ValueError("Conflicting duplicate updates for this need in one reply; send one selection.")
            if need_id in handled:
                continue
            handled.add(need_id)
            checked[need_id] = _checked_selection(entry, by_id[need_id], producer, records, aliases, scope)
            diagnostic["status"] = "accepted"
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            diagnostic["reason"] = str(exc)
        diagnostics.append(diagnostic)
    return diagnostics


def _answer_result(needs: list[dict[str, Any]], checked: Mapping[str, Any], operations: int) -> dict[str, Any]:
    refs, roles, unresolved = {}, {}, []
    for need in needs:
        answer = checked.get(need["need_id"], {})
        if "source_ref" in answer:
            source = answer["source_ref"]
            refs[source["ref"]] = deepcopy(source)
            if "record_ref" in answer:
                roles[need["quantity"]] = deepcopy(source)
        else:
            state = f"blocked: {answer['blocked']}" if "blocked" in answer else "not_investigated: no checked answer was selected."
            unresolved.append(f"step_index {need['step_index']}, quantity {need['quantity']}: {state}")
    return {"status": "provided" if refs else "unavailable", "operations_used": operations,
            "evidence_refs": list(refs.values()), "validation_refs": roles, "unresolved": unresolved}


def _read_answer_checkpoint(root: Path, reference: Mapping[str, str]) -> dict[str, Any]:
    """Verify saved answer ownership while leaving physical checks to validation."""
    checkpoint = verify_record(root, reference)
    request_ref = checkpoint["request_ref"]
    automatic = "resolution_ref" in checkpoint
    reply_ref = checkpoint["resolution_ref"] if automatic else checkpoint["reply_ref"]
    directory = Path(reference["ref"]).parent
    request = verify_record(root, request_ref)
    reply = verify_record(root, reply_ref)
    if (checkpoint.get("record_type") != "PrimitiveContextAnswers"
            or Path(request_ref["ref"]) != directory / "request.json"
            or Path(reply_ref["ref"]).parent != directory
            or reply.get("record_type") != ("PrimitiveInputResolution" if automatic else "PrimitiveContextReply")
            or reply.get("request_ref") != request_ref):
        raise ValueError("PA answers do not belong to their saved request and reply.")
    needs = {need["need_id"]: need for need in request["needs"]}
    seen = set()
    for answer in checkpoint["answers"]:
        need_id = answer["need_id"]
        if need_id in seen or needs.get(need_id) != answer["need"]:
            raise ValueError("Checked PA answer does not match its requested need.")
        seen.add(need_id)
    return checkpoint


def _batch_reference_check(arguments: Any, authorized: Mapping[str, str], field: str = "") -> None:
    # Native producers also resolve aliases. Freeze reference eligibility before any
    # item starts so a later item cannot guess an output from an earlier batch item.
    if isinstance(arguments, dict):
        for key, value in arguments.items():
            _batch_reference_check(value, authorized, key)
    elif isinstance(arguments, list):
        for value in arguments:
            _batch_reference_check(value, authorized, field)
    elif field.endswith(("_ref", "_refs")) and isinstance(arguments, str) and arguments not in authorized:
        raise ValueError(f"Record {arguments!r} was not issued when this batch started; dependent calls need a later reply.")


class ProductPrimitiveContextRuntime:
    """Serve deterministic measurements through the owned ProductAgent boundary."""

    def __init__(
        self,
        runtime: ProductionProductContextGroundingRuntime,
        product_agent: ProductAgentContextRuntime,
    ) -> None:
        """Retain the owned grounding and ProductAgent message boundaries."""
        self.runtime, self.product_agent = runtime, product_agent

    async def request_primitive_context(
        self, *, robot_runtime: Any, assignment: Any, interaction_root: Path,
        directory: Path, request: Mapping[str, Any], max_operations: int,
        deadline: float, progress: Callable[[Mapping[str, Any]], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """Serve this request only after its correlated SPADE delivery from RA.

        Args:
            robot_runtime: Owned adapter addressing the selected RA.
            assignment: Validated RA assignment.
            interaction_root: Root containing the immutable request.
            directory: This run's PA batch directory.
            request: Exact persisted input needs and authority pins.
            max_operations: Deterministic measurement operation budget.
            deadline: Absolute monotonic deadline.
            progress: Optional evidence progress consumer on PA's owning loop.

        Returns:
            The verified SPADE response and its response_ref.
        """
        root = interaction_root.resolve()
        request_ref = pin(root, directory / "request.json")
        saved = verify_record(root, request_ref)
        if {key: value for key, value in saved.items() if key != "fingerprint"} != request:
            raise ValueError("PA message request differs from its saved input needs.")
        run = verify_record(root, saved["run_request_ref"])
        if (Path(saved["run_request_ref"]["ref"]).parent != directory.relative_to(root).parent
                or run["base_context_refs"] != saved["base_context_refs"]
                or saved["assignment_fingerprint"] != assignment.fingerprint):
            raise ValueError("PA request has different run or assignment authority.")
        candidate = read_pin(root, saved["candidate_ref"])
        authoring = read_pin(root, {"ref": candidate["request_ref"], "sha256": candidate["request_sha256"]})
        if (candidate.get("record_type") != "PrimitiveProgramCandidate" or candidate.get("status") != "proposed"
                or candidate["primitive_steps"] != saved["primitive_steps"]
                or authoring["context_refs"] != saved["base_context_refs"]):
            raise ValueError("PA request has different proposal authority.")

        async def handle() -> dict[str, str]:
            outcome = await self.investigate(interaction_root=root, directory=directory,
                                             request=request, max_operations=max_operations, progress=progress)
            return await asyncio.to_thread(append_record, root, directory, "response.json", {
                "record_type": "PrimitiveContextResponse", "request_ref": request_ref, **outcome,
            })

        thread = uuid.uuid4().hex
        inbox_factory = getattr(self.product_agent, "primitive_context_inbox", None)
        if not callable(inbox_factory):
            raise RuntimeError("The ProductAgent runtime does not expose a scoped SPADE context inbox.")
        async with inbox_factory(
            root=root, request_ref=request_ref, sender=assignment.selected_resource_jid,
            thread=thread, deadline=deadline, handler=handle,
        ) as (recipient, service):
            reply = asyncio.create_task(robot_runtime.request_primitive_context(
                assignment, root=root, recipient=recipient, request_ref=request_ref,
                thread=thread, deadline=deadline,
            ))
            try:
                completed, _ = await asyncio.wait({service, reply}, return_when=asyncio.FIRST_COMPLETED)
                if service in completed:
                    service.result()
                return dict(await reply)
            finally:
                if not reply.done():
                    reply.cancel()
                await asyncio.gather(reply, return_exceptions=True)

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
            max_operations: Maximum deterministic evidence operations for this batch.
            progress: Optional asynchronous consumer of measurement progress.

        Returns:
            Selected evidence pins, unresolved PA findings, and consumed operations.
        """
        root = interaction_root.resolve()
        scope = read_validation_scope(request)
        needs = _number_needs(request["needs"])
        request = {**deepcopy(request), "needs": needs}
        request_path = directory / "request.json"
        if request_path.exists():
            request_ref = pin(root, request_path)
            if {key: value for key, value in read_pin(root, request_ref).items() if key != "fingerprint"} != request:
                raise ValueError("The saved PA request differs from the investigation request.")
        else:
            request_ref = append_record(root, directory, "request.json", request)
        operations = geometry_operations = 0
        checked: dict[str, Any] = {}
        answers_ref = None

        async def emit(message: str, **fields: Any) -> None:
            if progress is not None:
                await progress({"message": message, "operations_used": operations,
                                "geometry_operations": geometry_operations, "model_responses": 0,
                                "operations_remaining": max_operations - operations, **fields})

        authorized = {reference["ref"]: reference["sha256"] for reference in request["evidence_refs"]}
        producer = AssemblyGeometryProducer(root, directory / "geometry", authorized, request["target_feature"])
        records = await asyncio.to_thread(_verified_issued_records, producer)
        async def operate(
            item: Any, batch_authorized: dict[str, str],
            prepared: _ObservedGeometryBatch | None = None, preparation_error: str | None = None,
        ) -> dict[str, Any]:
            nonlocal operations, geometry_operations
            # Cancellation delivered here prevents queued operations from consuming budget.
            await asyncio.sleep(0)
            operations += 1
            number = operations
            name = str(item.get("tool_name", "")) if isinstance(item, dict) else ""
            operation_kind = ("read" if name in {"read_record", "read_observation"}
                              else "measurement/geometry" if name in _GEOMETRY_OPERATIONS else "evidence")
            child = AssemblyGeometryProducer(root, directory / f"operation_{number:04d}" / "geometry",
                                             dict(batch_authorized), request["target_feature"], _batch=prepared)
            started = time.monotonic()
            append_record(root, child.directory.parent, "request.json", {
                "record_type": "PrimitiveContextOperation", "request": deepcopy(item),
                "authorized_refs": [{"ref": ref, "sha256": sha} for ref, sha in batch_authorized.items()],
                "created_at_ns": time.time_ns(),
            })
            await emit(f"PA operation {number}/{max_operations}: starting {operation_kind} {name}.")
            cancelled = False
            try:
                if not isinstance(item, dict) or set(item) != {"tool_name", "arguments"}:
                    raise ValueError("Each request requires tool_name and JSON-encoded arguments.")
                arguments = json.loads(item["arguments"])
                if not isinstance(arguments, dict):
                    raise ValueError("Tool arguments must be an object.")
                canonical = ObservationPresentation(root).resolve(arguments)
                _batch_reference_check(canonical, batch_authorized)
                if preparation_error:
                    raise ValueError(preparation_error)
                if name in _GEOMETRY_ARGUMENTS:
                    if set(arguments) != _GEOMETRY_ARGUMENTS[name]:
                        raise ValueError("Geometry tool argument fields are invalid.")
                    if scope != GAZEBO_OBSERVED_SCOPE and name in {"observed_geometry", "bind_observed_part"}:
                        raise ValueError("The requested tool is unavailable in this scope.")
                    geometry_operations += name in _GEOMETRY_OPERATIONS
                    def measure() -> dict[str, Any]:
                        measured = self._geometry(child, name, canonical)
                        child.verify_outputs()
                        return measured

                    worker = asyncio.create_task(asyncio.to_thread(measure))
                    try:
                        result = await asyncio.shield(worker)
                    except asyncio.CancelledError:
                        # A thread cannot be killed. Retain its verified completed files
                        # before propagating cancellation; never launch its next item.
                        cancelled = True
                        result = await worker
                else:
                    raise ValueError("No deterministic measurement operation supports this request.")
            except asyncio.CancelledError:
                cancelled = True
                result = {"status": "cancelled", "reason": "The operation was cancelled."}
            except (KeyError, StopIteration, OSError, RuntimeError, TypeError, ValueError) as exc:
                result = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
            issued = []
            verified: dict[str, tuple[str, Any]] = {}
            for ref, sha in child.authorized.items():
                if ref in batch_authorized:
                    continue
                reference = {"ref": ref, "sha256": sha}
                _verify_evidence_tree(root, reference, verified)
                if ref in authorized and authorized[ref] != sha:
                    raise ValueError("Concurrent evidence producers issued conflicting hashes.")
                authorized[ref] = sha
                issued.append(reference)
            result = _pa_evidence_projection(root, result, records)
            operation_ref = append_record(root, child.directory.parent, "result.json", {
                "record_type": "PrimitiveContextOperationResult", "result": result, "issued_refs": issued,
                "cancelled": cancelled, "elapsed_sec": time.monotonic() - started, "created_at_ns": time.time_ns(),
                "timings": {**child.timings, **({"batch_inputs": prepared.timings} if prepared else {})},
            })
            status = ("unavailable" if "error" in result else result.get("status")
                      or result.get("record", {}).get("status") or result.get("record", {}).get("pose") or "completed")
            await emit(f"PA operation {number}/{max_operations}: {operation_kind} {name} {status} "
                       f"in {time.monotonic() - started:.2f} s; {max_operations - operations} operations remain.",
                       pa_operation_ref=operation_ref)
            if cancelled:
                raise asyncio.CancelledError
            return {"request": deepcopy(item), "result": result, "operation_ref": operation_ref}

        async def run_batch(requests: list[Any]) -> list[Any]:
            batch_authorized = dict(authorized)
            if all(isinstance(item, dict) and item.get("tool_name") == "observed_geometry" for item in requests):
                prepared, preparation_error = None, None
                try:
                    arguments = [ObservationPresentation(root).resolve(json.loads(item["arguments"])) for item in requests]
                    for argument in arguments:
                        _batch_reference_check(argument, batch_authorized)
                    prepared = await asyncio.to_thread(_ObservedGeometryBatch.prepare, root, batch_authorized, arguments)
                    await emit("Measurement batch inputs prepared: " + ", ".join(
                        f"{key}={value:.2f}" for key, value in prepared.timings.items()))
                except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
                    preparation_error = str(exc)
                ordered: list[Any] = [None] * len(requests)
                next_index = 0
                async def work() -> None:
                    nonlocal next_index
                    while next_index < len(requests):
                        index = next_index
                        next_index += 1
                        ordered[index] = await operate(requests[index], batch_authorized,
                                                       prepared, preparation_error)
                workers = [asyncio.create_task(work()) for _ in range(min(2, len(requests)))]
                try:
                    await asyncio.gather(*workers)
                except (asyncio.CancelledError, KeyError, OSError, RuntimeError, TypeError, ValueError):
                    for worker in workers:
                        worker.cancel()
                    await asyncio.gather(*workers, return_exceptions=True)
                    raise
            else:
                ordered = [await operate(item, batch_authorized) for item in requests]
            return ordered

        from .primitive_input_resolution import resolve_primitive_inputs

        attempted: set[str] = set()
        for resolution_number in range(1, max_operations + 2):
            records = await asyncio.to_thread(_verified_issued_records, producer)
            entries, requests = await asyncio.to_thread(
                resolve_primitive_inputs, request, producer, records, checked, attempted,
            )
            if not entries and not requests:
                break
            resolution_ref = append_record(root, directory, f"resolution_{resolution_number:04d}.json", {
                "record_type": "PrimitiveInputResolution", "request_ref": request_ref,
                "answers": entries, "requests": requests, "created_at_ns": time.time_ns(),
            })
            diagnostics = _check_answers(entries, needs, checked, producer, records, {}, scope)
            answers_ref = append_record(root, directory, f"answers_auto_{resolution_number:04d}.json", {
                "record_type": "PrimitiveContextAnswers", "request_ref": request_ref,
                "resolution_ref": resolution_ref, "answers": list(checked.values()), "diagnostics": diagnostics,
                "operations_used": operations, "model_responses": 0, "created_at_ns": time.time_ns(),
            })
            await emit(f"Direct input lookup: {len(checked)}/{len(needs)} needs answered; "
                       f"{len(requests)} measurement operations selected from accepted associations.",
                       pa_answers_ref=answers_ref, input_resolution_ref=resolution_ref)
            if not requests:
                break
            if len(requests) > max_operations - operations:
                await emit("Direct measurement batch exceeds the remaining operation budget; no item was started.")
                break
            attempted.update(fingerprint(item) for item in requests)
            await run_batch(requests)

        outcome = _answer_result(needs, checked, operations)
        outcome.update(answers_ref=answers_ref, model_responses=0)
        await emit(f"PA input resolution completed: {len(checked)}/{len(needs)} needs answered; "
                   f"{operations}/{max_operations} deterministic operations used.")
        return outcome

    def _geometry(
        self, producer: AssemblyGeometryProducer, name: str, arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        if name not in _GEOMETRY_ARGUMENTS:
            raise ValueError("Unsupported deterministic measurement operation.")
        return getattr(producer, name)(**arguments)
