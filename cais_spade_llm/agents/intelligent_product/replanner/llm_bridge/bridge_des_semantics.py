"""Typed Hybrid-DES bridge semantics for multi-turn recovery outline validation."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any

BridgeProjector = Callable[..., tuple["BridgeEventProjection", list[str]]]
BridgeProgressPolicy = Callable[..., tuple[int, dict[str, Any]]]


@dataclass
class ProductEntity:
    entity_id: str
    entity_kind: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceEntity:
    resource_jid: str
    resource_type: str
    snapshot: dict[str, Any] = field(default_factory=dict)
    capabilities: dict[str, Any] = field(default_factory=dict)


@dataclass
class BridgePlantState:
    resources_by_jid: dict[str, dict[str, Any]] = field(default_factory=dict)
    parts_by_name: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class ProcessSchema:
    schema_id: str
    display_name: str
    action_type: str
    operation_kind: str
    object_roles: dict[str, str] = field(default_factory=dict)
    parameter_roles: tuple[str, ...] = ()
    projector: BridgeProjector | None = None
    progress_policy: BridgeProgressPolicy | None = None
    retry_hint: str = ""


@dataclass
class BridgeEventInstance:
    outline_id: str
    event_schema_id: str
    resource_binding: str
    object_bindings: dict[str, str] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    predecessors: list[str] = field(default_factory=list)
    rationale: str = ""


CanonicalBridgeEventInstance = BridgeEventInstance


@dataclass
class BridgeValidationFinding:
    stage: str
    code: str
    reason: str
    task_id: str = ""
    resource_jid: str = ""
    part_name: str = ""
    owner: str = ""
    durable: bool | None = None
    retriable: bool | None = None
    unsatisfied_predicates: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    retry_hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        traits = _finding_traits(stage=self.stage, code=self.code)
        payload["owner"] = self.owner or traits["owner"]
        payload["durable"] = traits["durable"] if self.durable is None else bool(self.durable)
        payload["retriable"] = (
            traits["retriable"] if self.retriable is None else bool(self.retriable)
        )
        payload["constraint_owner"] = payload["owner"]
        payload["constraint_family"] = self.stage
        payload["constraint_code"] = self.code
        payload["resource_jid"] = self.resource_jid or None
        payload["part_name"] = self.part_name or None
        return payload


@dataclass
class BridgeValidationContext:
    resources_by_jid: dict[str, dict[str, Any]]
    parts_by_name: dict[str, dict[str, Any]]
    product_entities: dict[str, ProductEntity]
    resource_entities: dict[str, ResourceEntity]
    process_schemas: dict[str, ProcessSchema]
    known_locations: set[str]
    marked_conditions: list[dict[str, Any]] = field(default_factory=list)
    prepared_bridge_request: dict[str, Any] = field(default_factory=dict)
    plant_state: BridgePlantState = field(default_factory=BridgePlantState)


@dataclass
class BridgeEventProjection:
    start_state: dict[str, Any]
    end_state: dict[str, Any]
    action_target: dict[str, Any]
    part_name: str
    target_ref: str
    description: str


@dataclass
class BridgeValidationResult:
    ok: bool
    event_instance: BridgeEventInstance
    findings: list[BridgeValidationFinding] = field(default_factory=list)
    schema: ProcessSchema | None = None
    projection: BridgeEventProjection | None = None
    projected_resources: dict[str, dict[str, Any]] = field(default_factory=dict)
    projected_parts: dict[str, dict[str, Any]] = field(default_factory=dict)

    def finding_dicts(self) -> list[dict[str, Any]]:
        return [finding.to_dict() for finding in self.findings]


def _finding_traits(*, stage: str, code: str) -> dict[str, Any]:
    stage_token = str(stage or "").strip().lower()
    code_token = str(code or "").strip().lower()
    if stage_token in {"ontology_binding", "schema_grounding"}:
        return {"owner": "binding", "durable": True, "retriable": True}
    if stage_token == "plant_enabledness":
        return {"owner": "plant", "durable": True, "retriable": True}
    if stage_token == "supervisor_admissibility":
        return {"owner": "supervisor", "durable": True, "retriable": True}
    if stage_token == "resource_realizability":
        return {"owner": "resource", "durable": True, "retriable": True}
    if stage_token == "marked_progress":
        return {"owner": "progress", "durable": False, "retriable": True}
    if code_token in {"workspace_unreachable", "resource_validation_unavailable"}:
        return {"owner": "resource", "durable": True, "retriable": True}
    return {"owner": "validation", "durable": False, "retriable": True}


def bridge_process_schema_registry() -> dict[str, ProcessSchema]:
    schemas = [
        ProcessSchema(
            schema_id="recover_resource_idle",
            display_name="Recover Resource To Idle",
            action_type="recover_resource",
            operation_kind="recover_resource",
            projector=_project_recover_resource_idle,
            progress_policy=_progress_recover_resource_idle,
            retry_hint="choose recovery only when the resource is not already idle",
        ),
        ProcessSchema(
            schema_id="pick_part",
            display_name="Pick Part",
            action_type="acquire_part",
            operation_kind="pick_part",
            object_roles={"part": "product", "source_location": "location"},
            projector=_project_pick_part,
            progress_policy=_progress_pick_part,
            retry_hint="bind a visible source location or observation before proposing pick_part",
        ),
        ProcessSchema(
            schema_id="place_part",
            display_name="Place Part",
            action_type="release_part",
            operation_kind="place_part",
            object_roles={"part": "product", "target_location": "location"},
            projector=_project_place_part,
            retry_hint="establish control of the part before proposing place_part",
        ),
        ProcessSchema(
            schema_id="stage_part",
            display_name="Stage Part",
            action_type="stage_part",
            operation_kind="stage_part",
            object_roles={"part": "product", "target_location": "location"},
            projector=_project_stage_part,
            retry_hint="establish control of the part before proposing stage_part",
        ),
        ProcessSchema(
            schema_id="resume_nominal_task",
            display_name="Resume Nominal Task",
            action_type="resume_nominal_task",
            operation_kind="resume_nominal_task",
            parameter_roles=("nominal_task_id",),
            projector=_project_resume_nominal_task,
            retry_hint="bind a concrete nominal_task_id and ensure the resource is idle",
        ),
    ]
    return {schema.schema_id: schema for schema in schemas}


def build_bridge_validation_context(
    *,
    resources_by_jid: dict[str, dict[str, Any]],
    parts_by_name: dict[str, dict[str, Any]],
    prepared_bridge_request: dict[str, Any],
) -> BridgeValidationContext:
    bridge_resources = dict(prepared_bridge_request.get("bridge_resources") or {})
    product_entities: dict[str, ProductEntity] = {}
    resource_entities: dict[str, ResourceEntity] = {}
    known_locations: set[str] = {"observed_pose"}

    for part_name, raw_row in (parts_by_name or {}).items():
        token = str(part_name or "").strip()
        if not token or not isinstance(raw_row, dict):
            continue
        row = deepcopy(raw_row)
        product_entities[token] = ProductEntity(
            entity_id=token,
            entity_kind="part",
            attributes=row,
        )
        for field_name in (
            "current_location",
            "goal_location",
            "origin_location",
            "target_location",
        ):
            value = str(row.get(field_name) or "").strip()
            if value:
                known_locations.add(value)
        if dict(row.get("observed_pose") or {}):
            known_locations.add("observed_pose")

    for resource_jid, raw_row in (resources_by_jid or {}).items():
        token = str(resource_jid or "").strip()
        if not token or not isinstance(raw_row, dict):
            continue
        bridge_entry = dict(bridge_resources.get(token) or {})
        bridge_snapshot = dict(bridge_entry.get("bridge_snapshot") or {})
        static_capabilities = dict(bridge_entry.get("static_capabilities") or {})
        resource_type = str(
            raw_row.get("resource_type")
            or bridge_entry.get("resource_type")
            or bridge_snapshot.get("resource_type")
            or static_capabilities.get("resource_type")
            or "resource"
        ).strip()
        capability_block = deepcopy(static_capabilities)
        capability_block.update(
            {
                key: deepcopy(value)
                for key, value in bridge_snapshot.items()
                if key not in capability_block
            }
        )
        resource_entities[token] = ResourceEntity(
            resource_jid=token,
            resource_type=resource_type,
            snapshot=deepcopy(raw_row),
            capabilities=capability_block,
        )
        for source in (raw_row, bridge_snapshot, static_capabilities):
            if not isinstance(source, dict):
                continue
            for key in (
                "current_location",
                "reachable_locations",
                "reachability",
                "known_locations",
                "staging_areas",
                "available_named_poses",
                "named_poses",
            ):
                value = source.get(key)
                if isinstance(value, dict):
                    known_locations.update(
                        str(item).strip() for item in value.keys() if str(item).strip()
                    )
                elif isinstance(value, (list, tuple, set)):
                    known_locations.update(str(item).strip() for item in value if str(item).strip())
                else:
                    token_value = str(value or "").strip()
                    if token_value:
                        known_locations.add(token_value)

    marked_conditions: list[dict[str, Any]] = []
    grounding_parts = dict(
        dict(prepared_bridge_request.get("grounding_context") or {}).get("parts") or {}
    )
    for part_name, part_entry in grounding_parts.items():
        token = str(part_name or "").strip()
        target_location = str(dict(part_entry.get("target") or {}).get("location") or "").strip()
        if token and target_location:
            marked_conditions.append(
                {
                    "entity_kind": "part",
                    "entity": token,
                    "field": "current_location",
                    "expected": target_location,
                }
            )
            known_locations.add(target_location)

    return BridgeValidationContext(
        resources_by_jid=deepcopy(resources_by_jid or {}),
        parts_by_name=deepcopy(parts_by_name or {}),
        product_entities=product_entities,
        resource_entities=resource_entities,
        process_schemas=bridge_process_schema_registry(),
        known_locations=known_locations,
        marked_conditions=marked_conditions,
        prepared_bridge_request=deepcopy(prepared_bridge_request or {}),
        plant_state=BridgePlantState(
            resources_by_jid=deepcopy(resources_by_jid or {}),
            parts_by_name=deepcopy(parts_by_name or {}),
        ),
    )


def parse_bridge_event_instance(
    raw_event: dict[str, Any],
    *,
    outline_id: str,
) -> BridgeEventInstance:
    raw = dict(raw_event or {})
    return BridgeEventInstance(
        outline_id=str(raw.get("outline_id") or outline_id or "").strip(),
        event_schema_id=str(raw.get("event_schema_id") or "").strip(),
        resource_binding=str(raw.get("resource_binding") or "").strip(),
        object_bindings={
            str(key).strip(): str(value).strip()
            for key, value in dict(raw.get("object_bindings") or {}).items()
            if str(key).strip() and str(value).strip()
        },
        parameters=deepcopy(dict(raw.get("parameters") or {})),
        predecessors=[
            str(item).strip() for item in (raw.get("predecessors") or []) if str(item).strip()
        ],
        rationale=str(raw.get("rationale") or "").strip(),
    )


def validate_bridge_event_instance(
    instance: BridgeEventInstance,
    *,
    context: BridgeValidationContext,
) -> BridgeValidationResult:
    findings: list[BridgeValidationFinding] = []
    resource_jid = str(instance.resource_binding or "").strip()
    schema_id = str(instance.event_schema_id or "").strip()
    schema = context.process_schemas.get(schema_id)

    if not resource_jid:
        findings.append(
            BridgeValidationFinding(
                stage="ontology_binding",
                code="missing_resource_binding",
                reason="candidate must include resource_binding",
                task_id=instance.outline_id,
                evidence={"field": "resource_binding"},
            )
        )
        return BridgeValidationResult(ok=False, event_instance=instance, findings=findings)
    if resource_jid not in context.resource_entities:
        findings.append(
            BridgeValidationFinding(
                stage="ontology_binding",
                code="unknown_resource_binding",
                reason=f"resource_binding '{resource_jid}' does not resolve to a known resource",
                task_id=instance.outline_id,
                resource_jid=resource_jid,
                evidence={"field": "resource_binding", "token": resource_jid},
            )
        )
        return BridgeValidationResult(ok=False, event_instance=instance, findings=findings)

    if not schema_id:
        findings.append(
            BridgeValidationFinding(
                stage="schema_grounding",
                code="missing_event_schema_id",
                reason="candidate must include a canonical event_schema_id",
                task_id=instance.outline_id,
                resource_jid=resource_jid,
                evidence={"field": "event_schema_id"},
            )
        )
        return BridgeValidationResult(ok=False, event_instance=instance, findings=findings)
    if schema is None:
        findings.append(
            BridgeValidationFinding(
                stage="schema_grounding",
                code="unknown_event_schema_id",
                reason=f"event_schema_id '{schema_id}' is not registered",
                task_id=instance.outline_id,
                resource_jid=resource_jid,
                evidence={"field": "event_schema_id", "token": schema_id},
            )
        )
        return BridgeValidationResult(ok=False, event_instance=instance, findings=findings)

    role_findings = _validate_schema_roles(instance, schema=schema, context=context)
    if role_findings:
        return BridgeValidationResult(
            ok=False,
            event_instance=instance,
            findings=role_findings,
            schema=schema,
        )

    projector = schema.projector
    if projector is None:
        findings.append(
            BridgeValidationFinding(
                stage="schema_grounding",
                code="schema_semantics_unavailable",
                reason=f"schema '{schema.schema_id}' does not expose plant transition semantics",
                task_id=instance.outline_id,
                resource_jid=resource_jid,
                evidence={"event_schema_id": schema.schema_id},
            )
        )
        return BridgeValidationResult(
            ok=False,
            event_instance=instance,
            findings=findings,
            schema=schema,
        )

    projection, unsatisfied_predicates = projector(instance, context=context)
    if unsatisfied_predicates:
        part_name = str(instance.object_bindings.get("part") or "").strip()
        findings.append(
            BridgeValidationFinding(
                stage="plant_enabledness",
                code="unsatisfied_guard_predicate",
                reason=(
                    f"event '{schema_id}' is not enabled because one or more guard predicates are false"
                ),
                task_id=instance.outline_id,
                resource_jid=resource_jid,
                part_name=part_name,
                unsatisfied_predicates=deepcopy(unsatisfied_predicates),
                evidence={
                    "event_schema_id": schema_id,
                    "object_bindings": deepcopy(instance.object_bindings),
                },
                retry_hint=schema.retry_hint
                or "propose an event whose required guard predicates are satisfied",
            )
        )
        return BridgeValidationResult(
            ok=False,
            event_instance=instance,
            findings=findings,
            schema=schema,
        )

    projected_resources, projected_parts = _project_symbolic_state(
        context=context,
        resource_jid=resource_jid,
        projection=projection,
    )
    return BridgeValidationResult(
        ok=True,
        event_instance=instance,
        findings=[],
        schema=schema,
        projection=projection,
        projected_resources=projected_resources,
        projected_parts=projected_parts,
    )


def _validate_schema_roles(
    instance: BridgeEventInstance,
    *,
    schema: ProcessSchema,
    context: BridgeValidationContext,
) -> list[BridgeValidationFinding]:
    findings: list[BridgeValidationFinding] = []
    resource_jid = str(instance.resource_binding or "").strip()
    part_name = str(instance.object_bindings.get("part") or "").strip()

    unexpected_roles = sorted(
        role
        for role in instance.object_bindings
        if str(role or "").strip() and role not in schema.object_roles
    )
    if unexpected_roles:
        findings.append(
            BridgeValidationFinding(
                stage="schema_grounding",
                code="unexpected_object_binding",
                reason=(
                    f"event schema '{schema.schema_id}' does not allow object_bindings "
                    f"keys: {', '.join(unexpected_roles)}"
                ),
                task_id=instance.outline_id,
                resource_jid=resource_jid,
                part_name=part_name,
                evidence={
                    "unexpected_roles": unexpected_roles,
                    "allowed_roles": sorted(schema.object_roles),
                },
            )
        )
        return findings

    required_roles = {
        role: role_kind
        for role, role_kind in schema.object_roles.items()
        if not (schema.schema_id == "pick_part" and role == "source_location")
    }
    for role, role_kind in required_roles.items():
        if str(instance.object_bindings.get(role) or "").strip():
            continue
        findings.append(
            BridgeValidationFinding(
                stage="schema_grounding",
                code="missing_object_binding",
                reason=f"event schema '{schema.schema_id}' requires object_bindings['{role}']",
                task_id=instance.outline_id,
                resource_jid=resource_jid,
                part_name=part_name,
                evidence={"role": role, "role_kind": role_kind},
            )
        )
        return findings

    for role, role_kind in schema.object_roles.items():
        token = str(instance.object_bindings.get(role) or "").strip()
        if not token:
            continue
        if role_kind == "product" and token not in context.product_entities:
            findings.append(
                BridgeValidationFinding(
                    stage="ontology_binding",
                    code="unknown_product_binding",
                    reason=f"object_bindings['{role}']='{token}' does not resolve to a known product entity",
                    task_id=instance.outline_id,
                    resource_jid=resource_jid,
                    part_name=token,
                    evidence={"role": role, "token": token},
                )
            )
            return findings
        if role_kind == "location" and token not in context.known_locations:
            findings.append(
                BridgeValidationFinding(
                    stage="ontology_binding",
                    code="unknown_location_binding",
                    reason=f"object_bindings['{role}']='{token}' does not resolve to a known location token",
                    task_id=instance.outline_id,
                    resource_jid=resource_jid,
                    part_name=part_name,
                    evidence={
                        "role": role,
                        "token": token,
                        "known_locations": sorted(context.known_locations),
                    },
                )
            )
            return findings

    for parameter_name in schema.parameter_roles:
        if parameter_name in instance.parameters:
            continue
        findings.append(
            BridgeValidationFinding(
                stage="schema_grounding",
                code="missing_parameter_binding",
                reason=f"event schema '{schema.schema_id}' requires parameters['{parameter_name}']",
                task_id=instance.outline_id,
                resource_jid=resource_jid,
                part_name=part_name,
                evidence={"parameter": parameter_name},
            )
        )
        return findings

    unexpected_parameters = sorted(
        key
        for key in instance.parameters
        if str(key or "").strip() and key not in schema.parameter_roles
    )
    if unexpected_parameters:
        findings.append(
            BridgeValidationFinding(
                stage="schema_grounding",
                code="unexpected_parameter_binding",
                reason=(
                    f"event schema '{schema.schema_id}' does not allow parameters "
                    f"keys: {', '.join(unexpected_parameters)}"
                ),
                task_id=instance.outline_id,
                resource_jid=resource_jid,
                part_name=part_name,
                evidence={
                    "unexpected_parameters": unexpected_parameters,
                    "allowed_parameters": list(schema.parameter_roles),
                },
            )
        )
        return findings
    return findings


def _projection_inputs(
    instance: BridgeEventInstance,
    *,
    context: BridgeValidationContext,
) -> dict[str, Any]:
    resource_jid = str(instance.resource_binding or "").strip()
    resource_row = deepcopy(context.resources_by_jid.get(resource_jid) or {})
    part_name = str(instance.object_bindings.get("part") or "").strip()
    part_row = deepcopy(context.parts_by_name.get(part_name) or {})
    current_resource_state = _resource_state(resource_row)
    current_held_part = _resource_held_part(resource_row)
    current_part_state = _part_state(part_row)
    current_part_holder = _part_holder(part_row)
    current_part_location = _part_location(part_row)
    source_location = str(instance.object_bindings.get("source_location") or "").strip()
    target_location = str(instance.object_bindings.get("target_location") or "").strip()
    start_state: dict[str, Any] = {
        "resource_state": current_resource_state or None,
        "held_part": current_held_part or None,
    }
    if part_name:
        start_state.update(
            {
                "part_state": current_part_state or None,
                "part_location": current_part_location or None,
                "part_holder_resource_jid": current_part_holder or None,
            }
        )
    return {
        "resource_jid": resource_jid,
        "resource_row": resource_row,
        "part_name": part_name,
        "part_row": part_row,
        "current_resource_state": current_resource_state,
        "current_held_part": current_held_part,
        "current_part_state": current_part_state,
        "current_part_holder": current_part_holder,
        "current_part_location": current_part_location,
        "source_location": source_location,
        "target_location": target_location,
        "start_state": start_state,
    }


def _project_recover_resource_idle(
    instance: BridgeEventInstance,
    *,
    context: BridgeValidationContext,
) -> tuple[BridgeEventProjection, list[str]]:
    inputs = _projection_inputs(instance, context=context)
    resource_jid = inputs["resource_jid"]
    unsatisfied: list[str] = []
    end_state = deepcopy(inputs["start_state"])
    if inputs["current_resource_state"] == "idle":
        unsatisfied.append(f"needs_recovery({resource_jid})")
    end_state["resource_state"] = "idle"
    return BridgeEventProjection(
        start_state=inputs["start_state"],
        end_state=end_state,
        action_target={},
        part_name="",
        target_ref="",
        description=(
            instance.rationale
            or f"Recover {resource_jid} to idle so recovery execution can continue."
        ),
    ), unsatisfied


def _project_pick_part(
    instance: BridgeEventInstance,
    *,
    context: BridgeValidationContext,
) -> tuple[BridgeEventProjection, list[str]]:
    inputs = _projection_inputs(instance, context=context)
    part_name = inputs["part_name"]
    resource_jid = inputs["resource_jid"]
    source_location = inputs["source_location"]
    part_row = inputs["part_row"]
    actual_source = (
        source_location
        or inputs["current_part_location"]
        or ("observed_pose" if dict(part_row.get("observed_pose") or {}) else "")
    )
    unsatisfied: list[str] = []
    if inputs["current_resource_state"] in {"failed", "recovery_required"}:
        unsatisfied.append(f"idle({resource_jid})")
    if inputs["current_held_part"]:
        unsatisfied.append(f"free_gripper({resource_jid})")
    if inputs["current_part_holder"] and inputs["current_part_holder"] != resource_jid:
        unsatisfied.append(f"unheld({part_name})")
    if source_location:
        if source_location == "observed_pose":
            if not dict(part_row.get("observed_pose") or {}):
                unsatisfied.append(f"observed_pose({part_name})")
        elif inputs["current_part_location"] != source_location:
            unsatisfied.append(f"at({part_name}, {source_location})")
    elif not actual_source:
        unsatisfied.append(f"available_source({part_name})")
    action_target = {"source_location": actual_source or source_location or "observed_pose"}
    end_state = deepcopy(inputs["start_state"])
    end_state["resource_state"] = "picked"
    end_state["held_part"] = part_name or None
    end_state["part_state"] = "in_gripper" if part_name else None
    end_state["part_location"] = f"{resource_jid}_gripper" if part_name else None
    end_state["part_holder_resource_jid"] = resource_jid if part_name else None
    return BridgeEventProjection(
        start_state=inputs["start_state"],
        end_state=end_state,
        action_target=action_target,
        part_name=part_name,
        target_ref="",
        description=(
            instance.rationale
            or f"Pick {part_name} with {resource_jid} from {action_target['source_location']}."
        ),
    ), unsatisfied


def _project_place_part(
    instance: BridgeEventInstance,
    *,
    context: BridgeValidationContext,
) -> tuple[BridgeEventProjection, list[str]]:
    return _project_release_like_part(
        instance,
        context=context,
        release_schema_id="place_part",
    )


def _project_stage_part(
    instance: BridgeEventInstance,
    *,
    context: BridgeValidationContext,
) -> tuple[BridgeEventProjection, list[str]]:
    return _project_release_like_part(
        instance,
        context=context,
        release_schema_id="stage_part",
    )


def _project_release_like_part(
    instance: BridgeEventInstance,
    *,
    context: BridgeValidationContext,
    release_schema_id: str,
) -> tuple[BridgeEventProjection, list[str]]:
    inputs = _projection_inputs(instance, context=context)
    part_name = inputs["part_name"]
    resource_jid = inputs["resource_jid"]
    target_location = inputs["target_location"]
    unsatisfied: list[str] = []
    if inputs["current_resource_state"] in {"failed", "recovery_required"}:
        unsatisfied.append(f"idle({resource_jid})")
    if inputs["current_held_part"] != part_name:
        unsatisfied.append(f"holds({resource_jid}, {part_name})")
    if not target_location:
        unsatisfied.append("target_location_bound(event)")
    action_target = {"target_location": target_location}
    end_state = deepcopy(inputs["start_state"])
    end_state["resource_state"] = "idle"
    end_state["held_part"] = None
    if release_schema_id == "stage_part":
        part_end_state = "staged"
    else:
        target_goal = _goal_location_for_part(context, part_name=part_name)
        part_end_state = "assembled" if target_goal and target_location == target_goal else "ready"
    end_state["part_state"] = part_end_state if part_name else None
    end_state["part_location"] = target_location or None
    end_state["part_holder_resource_jid"] = None
    verb = "Stage" if release_schema_id == "stage_part" else "Place"
    return BridgeEventProjection(
        start_state=inputs["start_state"],
        end_state=end_state,
        action_target=action_target,
        part_name=part_name,
        target_ref=target_location,
        description=instance.rationale
        or f"{verb} {part_name} with {resource_jid} to {target_location}.",
    ), unsatisfied


def _project_resume_nominal_task(
    instance: BridgeEventInstance,
    *,
    context: BridgeValidationContext,
) -> tuple[BridgeEventProjection, list[str]]:
    inputs = _projection_inputs(instance, context=context)
    resource_jid = inputs["resource_jid"]
    nominal_task_id = str(instance.parameters.get("nominal_task_id") or "").strip()
    unsatisfied: list[str] = []
    if inputs["current_resource_state"] not in {"idle", "ready", ""}:
        unsatisfied.append(f"idle({resource_jid})")
    if not nominal_task_id:
        unsatisfied.append("nominal_task_bound(event)")
    return BridgeEventProjection(
        start_state=inputs["start_state"],
        end_state=deepcopy(inputs["start_state"]),
        action_target={},
        part_name="",
        target_ref="",
        description=instance.rationale
        or f"Resume nominal task {nominal_task_id or 'unknown'} on {resource_jid}.",
    ), unsatisfied


def build_outline_task_row(
    *,
    instance: BridgeEventInstance,
    schema: ProcessSchema,
    projection: BridgeEventProjection,
) -> dict[str, Any]:
    resource_jid = str(instance.resource_binding or "").strip()
    part_name = str(projection.part_name or "").strip()
    event_name = _event_name_for_instance(instance, schema=schema, projection=projection)
    outline_task: dict[str, Any] = {
        "outline_id": instance.outline_id,
        "event_schema_id": schema.schema_id,
        "resource_jid": resource_jid,
        "event_name": event_name,
        "description": projection.description,
        "predecessors": [
            str(item).strip() for item in (instance.predecessors or []) if str(item).strip()
        ],
        "expected_start_state": deepcopy(projection.start_state),
        "expected_end_state": deepcopy(projection.end_state),
        "action_type": schema.action_type,
        "task_kind": schema.action_type,
        "action_target": deepcopy(projection.action_target),
        "bridge_event_instance": {
            "event_schema_id": schema.schema_id,
            "resource_binding": resource_jid,
            "object_bindings": deepcopy(instance.object_bindings),
            "parameters": deepcopy(instance.parameters),
            "predecessors": deepcopy(instance.predecessors),
            "rationale": instance.rationale,
        },
    }
    if part_name:
        outline_task["part_name"] = part_name
    target_ref = str(projection.target_ref or "").strip()
    if target_ref:
        outline_task["target_ref"] = target_ref
    return outline_task


def _project_symbolic_state(
    *,
    context: BridgeValidationContext,
    resource_jid: str,
    projection: BridgeEventProjection,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    projected_resources = deepcopy(context.resources_by_jid or {})
    projected_parts = deepcopy(context.parts_by_name or {})

    resource_row = projected_resources.setdefault(resource_jid, {"resource_jid": resource_jid})
    resource_row["current_state"] = deepcopy(projection.end_state.get("resource_state"))
    if "held_part" in projection.end_state:
        resource_row["held_part"] = deepcopy(projection.end_state.get("held_part"))
        resource_row["gripper_state"] = (
            "closed" if projection.end_state.get("held_part") not in (None, "") else "open"
        )

    if projection.part_name:
        part_row = projected_parts.setdefault(
            projection.part_name, {"part_name": projection.part_name}
        )
        part_row["current_state"] = deepcopy(projection.end_state.get("part_state"))
        part_row["current_location"] = deepcopy(projection.end_state.get("part_location"))
        part_row["current_holder_resource_jid"] = deepcopy(
            projection.end_state.get("part_holder_resource_jid")
        )
    return projected_resources, projected_parts


def _event_name_for_instance(
    instance: BridgeEventInstance,
    *,
    schema: ProcessSchema,
    projection: BridgeEventProjection,
) -> str:
    part_name = str(projection.part_name or "").strip()
    resource_jid = str(instance.resource_binding or "").strip()
    target_ref = str(projection.target_ref or "").strip()
    if schema.schema_id == "pick_part":
        source = str(projection.action_target.get("source_location") or "").strip()
        return f"pick {part_name} from {source}".strip()
    if schema.schema_id == "place_part":
        return f"place {part_name} to {target_ref}".strip()
    if schema.schema_id == "stage_part":
        return f"stage {part_name} to {target_ref}".strip()
    if schema.schema_id == "recover_resource_idle":
        return f"recover {resource_jid} to idle".strip()
    if schema.schema_id == "resume_nominal_task":
        nominal_task_id = str(instance.parameters.get("nominal_task_id") or "").strip()
        return f"resume nominal task {nominal_task_id}".strip()
    return schema.display_name


def _progress_pick_part(
    *,
    event_instance: BridgeEventInstance,
    projection: BridgeEventProjection,
    context: BridgeValidationContext,
    semantic_result: BridgeValidationResult,
    projected_satisfied: int,
) -> tuple[int, dict[str, Any]]:
    del event_instance, semantic_result
    part_name = str(projection.part_name or "").strip()
    marked_parts = {
        str(row.get("entity") or "").strip()
        for row in (context.marked_conditions or [])
        if str(row.get("entity_kind") or "").strip() == "part"
    }
    if part_name and part_name in marked_parts:
        return (
            1,
            {
                "resolved_marked_conditions": 0,
                "blocker_part_acquired": 1,
                "remaining_marked_conditions": max(
                    0, len(context.marked_conditions) - projected_satisfied
                ),
            },
        )
    return (
        0,
        {
            "resolved_marked_conditions": 0,
            "remaining_marked_conditions": max(
                0, len(context.marked_conditions) - projected_satisfied
            ),
        },
    )


def _progress_recover_resource_idle(
    *,
    event_instance: BridgeEventInstance,
    projection: BridgeEventProjection,
    context: BridgeValidationContext,
    semantic_result: BridgeValidationResult,
    projected_satisfied: int,
) -> tuple[int, dict[str, Any]]:
    del event_instance, semantic_result
    start_state = dict(projection.start_state or {})
    if str(start_state.get("resource_state") or "").strip() not in {"", "idle"}:
        return (
            1,
            {
                "resolved_marked_conditions": 0,
                "recovered_resource_idle": 1,
                "remaining_marked_conditions": max(
                    0, len(context.marked_conditions) - projected_satisfied
                ),
            },
        )
    return (
        0,
        {
            "resolved_marked_conditions": 0,
            "remaining_marked_conditions": max(
                0, len(context.marked_conditions) - projected_satisfied
            ),
        },
    )


def _goal_location_for_part(context: BridgeValidationContext, *, part_name: str) -> str:
    entity = context.product_entities.get(part_name)
    if entity is not None:
        attributes = dict(entity.attributes or {})
        target_location = str(
            attributes.get("goal_location") or attributes.get("target_location") or ""
        ).strip()
        if target_location:
            return target_location
    grounding_parts = dict(
        dict(context.prepared_bridge_request.get("grounding_context") or {}).get("parts") or {}
    )
    return str(
        dict(dict(grounding_parts.get(part_name) or {}).get("target") or {}).get("location") or ""
    ).strip()


def _resource_state(resource_row: dict[str, Any]) -> str:
    return str(resource_row.get("current_state") or resource_row.get("state") or "").strip()


def _resource_held_part(resource_row: dict[str, Any]) -> str:
    return str(resource_row.get("held_part") or "").strip()


def _part_state(part_row: dict[str, Any]) -> str:
    token = str(part_row.get("current_state") or part_row.get("state") or "").strip()
    if token:
        return token
    if str(part_row.get("current_holder_resource_jid") or "").strip():
        return "in_gripper"
    return ""


def _part_location(part_row: dict[str, Any]) -> str:
    token = str(part_row.get("current_location") or part_row.get("location") or "").strip()
    if token:
        return token
    if dict(part_row.get("observed_pose") or {}):
        return "observed_pose"
    return ""


def _part_holder(part_row: dict[str, Any]) -> str:
    return str(
        part_row.get("current_holder_resource_jid") or part_row.get("holder_resource_jid") or ""
    ).strip()


__all__ = [
    "BridgeEventInstance",
    "BridgePlantState",
    "BridgeEventProjection",
    "BridgeValidationContext",
    "BridgeValidationFinding",
    "BridgeValidationResult",
    "CanonicalBridgeEventInstance",
    "ProcessSchema",
    "ProductEntity",
    "ResourceEntity",
    "build_outline_task_row",
    "bridge_process_schema_registry",
    "build_bridge_validation_context",
    "parse_bridge_event_instance",
    "validate_bridge_event_instance",
]
