"""Bridge safety and event validation helpers for DES-guided bridge replanning."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from typing import Any

from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.bridge_adapters import (
    bridge_adapter_capabilities,
    bridge_resource_type,
    canonical_bridge_constraint_from_rule,
    canonical_bridge_event,
    canonical_bridge_resource,
)
from cais_spade_llm.resources.resource_profile import (
    all_registered_operation_kinds,
    get_resource_profile,
    resource_snapshot_carried_entity,
    resource_snapshot_fields_map,
)
from cais_spade_llm.prompts import BridgeHintLevel


class BridgeSafetyMixin:
    @staticmethod
    def _bridge_resource_core_view(entry: dict[str, Any]) -> dict[str, Any]:
        return deepcopy(dict(entry.get("resource_core") or {}))

    @staticmethod
    def _bridge_resource_facets_view(entry: dict[str, Any]) -> dict[str, Any]:
        return deepcopy(dict(entry.get("resource_facets") or {}))

    def _bridge_grounding_context(
        self,
        *,
        focused_resource_jid: str,
        bridge_snapshot: dict[str, Any] | None,
        bridge_resources: dict[str, dict[str, Any]] | None,
        part_tracker: dict[str, Any],
        goal_state: str,
        P_id: list[str],
        obligation_targets: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build the generic grounding context exposed to primitive bridge prompts."""
        def _flat_facet_fields(canonical_snapshot: dict[str, Any]) -> dict[str, Any]:
            flat: dict[str, Any] = {}
            for facet in (canonical_snapshot.get("resource_facets") or {}).values():
                if not isinstance(facet, dict):
                    continue
                for key, value in facet.items():
                    flat[str(key)] = deepcopy(value)
            return flat

        snapshot = dict(bridge_snapshot or {})
        focused_canonical = canonical_bridge_resource(
            resource_jid=str(focused_resource_jid or "").strip(),
            resource_type=bridge_resource_type(
                snapshot=snapshot,
                modeled_state={},
                static_capabilities={},
            ),
            snapshot=snapshot,
            modeled_state={},
        )
        context: dict[str, Any] = {
            "resource": {
                "jid": str(focused_resource_jid or "").strip(),
                "resource_core": deepcopy(focused_canonical.get("resource_core") or {}),
                "resource_facets": deepcopy(focused_canonical.get("resource_facets") or {}),
                "resource_type": focused_canonical.get("resource_type"),
                "current_state": focused_canonical.get("current_state"),
                "current_location": deepcopy(focused_canonical.get("current_location")),
            },
            "resources": {},
            "parts": {},
            "goal": {
                "goal_state": str(goal_state or "").strip(),
                "pending_parts": [str(part_name) for part_name in (P_id or []) if str(part_name).strip()],
            },
            "focused_resource_jid": str(focused_resource_jid or "").strip(),
            "obligation_targets": deepcopy(obligation_targets or []),
        }
        focused_profile = get_resource_profile(str(focused_canonical.get("resource_type") or "resource"))
        context["resource"].update(
            resource_snapshot_fields_map(
                focused_canonical,
                focused_profile.snapshot_fields,
                profile=focused_profile,
            )
        )
        context["resource"].update(_flat_facet_fields(focused_canonical))

        for resource_jid, raw_entry in (bridge_resources or {}).items():
            if not isinstance(raw_entry, dict):
                continue
            resource_snapshot = dict(raw_entry.get("bridge_snapshot") or raw_entry.get("primitive_snapshot") or {})
            resource_type = bridge_resource_type(
                snapshot=resource_snapshot,
                modeled_state=dict(raw_entry.get("modeled_state") or {}),
                static_capabilities=dict(raw_entry.get("static_capabilities") or {}),
            )
            canonical_entry = canonical_bridge_resource(
                resource_jid=str(resource_jid),
                resource_type=resource_type,
                snapshot=resource_snapshot,
                modeled_state=dict(raw_entry.get("modeled_state") or {}),
            )
            profile = get_resource_profile(str(canonical_entry.get("resource_type") or "resource"))
            context["resources"][str(resource_jid)] = {
                "jid": str(resource_jid),
                "resource_core": deepcopy(canonical_entry.get("resource_core") or {}),
                "resource_facets": deepcopy(canonical_entry.get("resource_facets") or {}),
                "resource_type": canonical_entry.get("resource_type"),
                "current_state": canonical_entry.get("current_state"),
                "current_location": deepcopy(canonical_entry.get("current_location")),
                "modeled_state": deepcopy(raw_entry.get("modeled_state") or {}),
                "pending_tasks": deepcopy(raw_entry.get("pending_tasks") or []),
                "static_capabilities": deepcopy(raw_entry.get("static_capabilities") or {}),
                "bridge_adapter": deepcopy(raw_entry.get("bridge_adapter") or {}),
            }
            context["resources"][str(resource_jid)].update(
                resource_snapshot_fields_map(
                    canonical_entry,
                    profile.snapshot_fields,
                    profile=profile,
                )
            )
            context["resources"][str(resource_jid)].update(_flat_facet_fields(canonical_entry))

        geometry_lookup = getattr(self.product_agent, "_geometry_for_part", None)
        for part_name, raw_info in (part_tracker or {}).items():
            name = str(part_name or "").strip()
            if not name:
                continue

            info = raw_info if isinstance(raw_info, dict) else {}
            observed_pose = None
            for candidate_key in ("observed_pose", "position", "pose", "location"):
                observed_pose = self._coerce_xyz_pose(info.get(candidate_key))
                if observed_pose is not None:
                    break

            target: dict[str, Any] | None = None
            if callable(geometry_lookup):
                try:
                    geometry = geometry_lookup(name) or {}
                except Exception:
                    geometry = {}
                if isinstance(geometry, dict):
                    slot_xy = geometry.get("slot_xy")
                    board_center = geometry.get("board_center") or {}
                    if isinstance(slot_xy, (list, tuple)) and len(slot_xy) >= 2:
                        try:
                            board_top_z = float(
                                geometry.get("slot_floor_z_m", board_center.get("z", 0.0))
                            )
                            target = {
                                "slot_pose": {
                                    "x": float(board_center.get("x", 0.0)) + float(slot_xy[0]),
                                    "y": float(board_center.get("y", 0.0)) + float(slot_xy[1]),
                                    "z": board_top_z,
                                },
                                "board_top_z": board_top_z,
                            }
                            if geometry.get("part_height_m") is not None:
                                target["part_height"] = float(geometry["part_height_m"])
                            if geometry.get("model_name"):
                                target["model_name"] = str(geometry["model_name"])
                            product_location = str(getattr(self.product_agent, "jid", "") or "").strip()
                            if product_location:
                                target["location"] = product_location.split("@", 1)[0]
                        except (TypeError, ValueError):
                            target = None

            context["parts"][name] = {
                "state": info.get("state"),
                "location": deepcopy(info.get("location")),
                "last_known_location": deepcopy(info.get("last_known_location")),
                "observed_pose": observed_pose,
                "target": target,
            }

        return context

    async def _bridge_primitive_context(
        self,
        *,
        target_jid: str,
        resource_states: dict[str, dict[str, Any]],
        default_resource_state: str,
        part_states: dict[str, Any],
        part_locations: dict[str, Any],
    ) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None, dict[str, dict[str, Any]]]:
        focused_resource = self._resource_by_jid(target_jid)
        if focused_resource is None:
            return None, None, {}

        from cais_spade_llm.agents.intelligent_product.replanner.llm_bridge.primitive_semantics import (
            build_primitive_catalog,
        )

        focused_primitive_catalog: list[dict[str, Any]] | None = None
        focused_bridge_snapshot: dict[str, Any] | None = None
        bridge_resources: dict[str, dict[str, Any]] = {}

        try:
            for resource in self.resource_agents:
                resource_jid = str(getattr(resource, "jid", "")).strip()
                if not resource_jid:
                    continue
                raw_bridge_snapshot: dict[str, Any] = {}
                if hasattr(resource, "get_bridge_snapshot"):
                    bridge_snapshot_value = await asyncio.to_thread(resource.get_bridge_snapshot)
                    if isinstance(bridge_snapshot_value, dict):
                        raw_bridge_snapshot = deepcopy(bridge_snapshot_value)
                elif hasattr(resource, "_snapshot_state"):
                    snapshot_value = await asyncio.to_thread(resource._snapshot_state)
                    if isinstance(snapshot_value, dict):
                        raw_bridge_snapshot = deepcopy(snapshot_value)

                modeled_state = self._build_resource_search_state(
                    resource_jid=resource_jid,
                    resource_states=resource_states,
                    default_resource_state=default_resource_state,
                    part_states=part_states,
                    part_locations=part_locations,
                )
                static_capabilities = deepcopy(getattr(resource, "static_capabilities", {}) or {})
                resource_type = bridge_resource_type(
                    resource=resource,
                    snapshot=raw_bridge_snapshot,
                    modeled_state=modeled_state,
                    static_capabilities=static_capabilities,
                )
                bridge_snapshot = canonical_bridge_resource(
                    resource_jid=resource_jid,
                    resource_type=resource_type,
                    snapshot=raw_bridge_snapshot,
                    modeled_state=modeled_state,
                )
                primitive_catalog = build_primitive_catalog(resource) or []
                adapter_capabilities = bridge_adapter_capabilities(
                    resource_type, primitive_catalog=primitive_catalog,
                )
                bridge_resources[resource_jid] = {
                    "resource_jid": resource_jid,
                    "resource_type": resource_type,
                    "bridge_adapter": adapter_capabilities,
                    "primitive_catalog": primitive_catalog,
                    "bridge_snapshot": bridge_snapshot or {},
                    "resource_core": deepcopy(bridge_snapshot.get("resource_core") or {}),
                    "resource_facets": deepcopy(bridge_snapshot.get("resource_facets") or {}),
                    "modeled_state": modeled_state,
                    "pending_tasks": deepcopy(self._pending_resource_tasks(resource_jid)),
                    "static_capabilities": static_capabilities,
                }
                if resource_jid == target_jid:
                    focused_primitive_catalog = (
                        primitive_catalog if adapter_capabilities.get("supports_executable_bridge") else None
                    )
                    focused_bridge_snapshot = bridge_snapshot or None

            return focused_primitive_catalog, focused_bridge_snapshot, bridge_resources
        except Exception:
            self.logger.exception(
                "[Planner] Failed to build whole-system primitive bridge context for %s",
                target_jid,
            )
            return None, None, {}

    def _collect_bridge_safety_context(
        self,
        violations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        rule_ids: list[str] = []
        safe_next_task_ids: list[str] = []
        running_aps: list[Any] = []
        candidate_aps: list[Any] = []
        predicted_state_aps: list[Any] = []
        constraints: list[dict[str, Any]] = []
        safety_rules: list[dict[str, Any]] = []
        statuses: list[str] = []
        reasons: list[str] = []
        seen_rule_ids: set[str] = set()
        seen_next_task_ids: set[str] = set()
        seen_constraint_signatures: set[str] = set()
        seen_safety_rule_signatures: set[str] = set()

        for violation in violations or []:
            if not isinstance(violation, dict):
                continue
            safety_ctx = violation.get("safety_ctx") or {}
            if not isinstance(safety_ctx, dict):
                continue
            raw_bridge = safety_ctx.get("bridge_safety_context") or {}
            if not isinstance(raw_bridge, dict):
                raw_bridge = {}

            raw_rule_ids = (
                raw_bridge.get("rule_ids")
                or safety_ctx.get("rule_ids")
                or safety_ctx.get("violated_rule")
                or safety_ctx.get("violated_rule_id")
            )
            if isinstance(raw_rule_ids, (list, tuple, set)):
                values = raw_rule_ids
            else:
                values = [raw_rule_ids]
            for value in values:
                token = str(value or "").strip()
                if token and token not in seen_rule_ids:
                    seen_rule_ids.add(token)
                    rule_ids.append(token)

            for task_id in (
                raw_bridge.get("safe_next_task_ids")
                or safety_ctx.get("safe_next_task_ids")
                or []
            ):
                token = str(task_id or "").strip()
                if token and token not in seen_next_task_ids:
                    seen_next_task_ids.add(token)
                    safe_next_task_ids.append(token)

            running_aps.extend(list(raw_bridge.get("running_aps") or safety_ctx.get("running_aps") or []))
            candidate_aps.extend(
                list(raw_bridge.get("candidate_aps") or safety_ctx.get("candidate_aps") or [])
            )
            predicted_state_aps.extend(
                list(
                    raw_bridge.get("predicted_state_aps")
                    or safety_ctx.get("predicted_state_aps")
                    or []
                )
            )
            status = str(raw_bridge.get("status") or safety_ctx.get("status") or "").strip()
            if status:
                statuses.append(status)
            reason = str(raw_bridge.get("reason") or safety_ctx.get("reason") or "").strip()
            if reason:
                reasons.append(reason)

            for constraint in (
                raw_bridge.get("constraints")
                or safety_ctx.get("constraints")
                or []
            ):
                if not isinstance(constraint, dict):
                    continue
                signature = self._bridge_constraint_signature(constraint)
                if not signature or signature in seen_constraint_signatures:
                    continue
                seen_constraint_signatures.add(signature)
                constraints.append(deepcopy(constraint))

            for rule in (
                raw_bridge.get("safety_rules")
                or safety_ctx.get("safety_rules")
                or []
            ):
                if not isinstance(rule, dict):
                    continue
                signature = self._bridge_safety_rule_signature(rule)
                if not signature or signature in seen_safety_rule_signatures:
                    continue
                seen_safety_rule_signatures.add(signature)
                safety_rules.append(deepcopy(rule))

        return {
            "rule_ids": rule_ids,
            "safe_next_task_ids": safe_next_task_ids,
            "running_aps": running_aps,
            "candidate_aps": candidate_aps,
            "predicted_state_aps": predicted_state_aps,
            "status": statuses[-1] if statuses else "",
            "reason": reasons[-1] if reasons else "",
            "constraints": constraints,
            "safety_rules": safety_rules,
        }

    @staticmethod
    def _bridge_constraint_signature(constraint: dict[str, Any]) -> str:
        if not isinstance(constraint, dict):
            return ""
        return json.dumps(constraint, sort_keys=True, default=str)

    @staticmethod
    def _bridge_safety_rule_signature(rule: dict[str, Any]) -> str:
        if not isinstance(rule, dict):
            return ""
        rule_id = str(rule.get("id", "") or "").strip()
        if rule_id:
            return rule_id
        return json.dumps(rule, sort_keys=True, default=str)

    @staticmethod
    def _bridge_resource_token(value: Any) -> str:
        token = str(value or "").strip().lower()
        if "@" in token:
            token = token.split("@", 1)[0]
        return token

    def _bridge_rule_resource_jids(
        self,
        *,
        rule: dict[str, Any],
        bridge_resources: dict[str, dict[str, Any]],
    ) -> list[str]:
        resource_lookup: dict[str, str] = {}
        for resource_jid in (bridge_resources or {}).keys():
            jid = str(resource_jid or "").strip()
            if not jid:
                continue
            resource_lookup[jid.lower()] = jid
            resource_lookup[self._bridge_resource_token(jid)] = jid

        resolved: list[str] = []
        for raw_resource in (rule.get("resources") or []):
            token = self._bridge_resource_token(raw_resource)
            if not token:
                continue
            resource_jid = resource_lookup.get(token) or resource_lookup.get(str(raw_resource).strip().lower())
            if resource_jid and resource_jid not in resolved:
                resolved.append(resource_jid)
        return resolved

    @staticmethod
    def _bridge_turn_observation_primitives() -> list[str]:
        return [
            "detect_parts",
            "get_current_pose",
        ]

    @staticmethod
    def _bridge_phase_allowed_types(phase: str) -> set[str]:
        token = str(phase or "").strip().lower()
        if token == "observe_required":
            return {"observe"}
        if token == "bridge_outline":
            return {"bridge_outline"}
        if token == "bridge_events":
            return {"bridge_events"}
        if token == "review":
            return set()
        return {"final_plan"}

    @staticmethod
    def _bridge_incremental_event_mode(
        prepared_bridge_request: dict[str, Any],
    ) -> bool:
        raw_hint_level = str(
            prepared_bridge_request.get("hint_level", "") or ""
        ).strip().lower()
        return raw_hint_level == "none"

    def _bridge_critical_parts(
        self,
        prepared_bridge_request: dict[str, Any],
    ) -> list[str]:
        marked_reentry_context = deepcopy(
            prepared_bridge_request.get("marked_reentry_context")
            or prepared_bridge_request.get("continuation_context")
            or {}
        )
        critical: set[str] = set()
        for unmet in (marked_reentry_context.get("unmet_reentry_conditions") or []):
            if not isinstance(unmet, dict):
                continue
            if str(unmet.get("entity_kind", "")).strip() != "part":
                continue
            if str(unmet.get("role", "")).strip() != "bridge_replaced":
                continue
            part_name = str(unmet.get("entity", "")).strip()
            if part_name:
                critical.add(part_name)
        if critical:
            return sorted(critical)

        for part_name, raw_info in (prepared_bridge_request.get("part_tracker") or {}).items():
            name = str(part_name or "").strip()
            info = raw_info if isinstance(raw_info, dict) else {}
            state = str(info.get("state", "") or "").strip().lower()
            if name and state in {"misplaced", "unknown"}:
                critical.add(name)
        return sorted(critical)

    def _bridge_has_fresh_part_observation(
        self,
        prepared_bridge_request: dict[str, Any],
        *,
        part_name: str,
    ) -> bool:
        part_token = str(part_name or "").strip()
        if not part_token:
            return False

        bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
        for row in (bridge_session.get("observation_history") or []):
            if not isinstance(row, dict):
                continue
            observation = row.get("observation") or {}
            observed_part_name = str(
                (observation.get("part_name") if isinstance(observation, dict) else "")
                or (row.get("params") or {}).get("part_name")
                or ""
            ).strip()
            if observed_part_name == part_token:
                return True

        raw_hint_level = str(
            prepared_bridge_request.get("hint_level", "") or ""
        ).strip().lower()
        if raw_hint_level == "none":
            return False

        def _has_usable_pose(value: Any) -> bool:
            pose = value if isinstance(value, dict) else {}
            return all(pose.get(axis) is not None for axis in ("x", "y", "z"))

        part_tracker = dict(prepared_bridge_request.get("part_tracker") or {})
        tracked_part = dict(part_tracker.get(part_token) or {})
        if _has_usable_pose(tracked_part.get("observed_pose")):
            return True

        grounding_parts = dict(
            (prepared_bridge_request.get("grounding_context") or {}).get("parts") or {}
        )
        grounded_part = dict(grounding_parts.get(part_token) or {})
        if _has_usable_pose(grounded_part.get("observed_pose")):
            return True

        return False

    def _bridge_requires_live_observation(
        self,
        prepared_bridge_request: dict[str, Any],
    ) -> bool:
        for part_name in self._bridge_critical_parts(prepared_bridge_request):
            if not self._bridge_has_fresh_part_observation(
                prepared_bridge_request,
                part_name=part_name,
            ):
                return True
        return False

    def _bridge_current_phase(
        self,
        prepared_bridge_request: dict[str, Any],
    ) -> str:
        bridge_session = dict(prepared_bridge_request.get("bridge_session") or {})
        phase = str(bridge_session.get("phase", "") or "").strip().lower()
        if phase == "review":
            return "review"
        if self._bridge_requires_live_observation(prepared_bridge_request):
            return "observe_required"
        if self._bridge_incremental_event_mode(prepared_bridge_request):
            if bridge_session.get("bridge_events_complete"):
                return "final_plan"
            if not list(bridge_session.get("bridge_outline") or []):
                return "bridge_outline"
            return "bridge_events"
        if bridge_session.get("approved_bridge_events"):
            return "final_plan"
        return "bridge_events"

    def _bridge_event_condition_keys(self, events: list[dict[str, Any]]) -> set[tuple[str, str, str, str]]:
        keys: set[tuple[str, str, str, str]] = set()
        for event in events:
            if not isinstance(event, dict):
                continue
            for condition in (event.get("closes_conditions") or []):
                if not isinstance(condition, dict):
                    continue
                key = self._marked_reentry_condition_key(condition)
                if all(key):
                    keys.add(key)
        return keys

    def _bridge_event_operation_kind(self, event: dict[str, Any]) -> str:
        operation_family = str(canonical_bridge_event(event).get("operation_family", "") or "").strip()
        if operation_family:
            return operation_family
        return "bridge"

    @staticmethod
    def _bridge_feasibility_operation_kind(operation_kind: str) -> str:
        return str(operation_kind or "").strip().lower()

    def _bridge_part_context(
        self,
        prepared_bridge_request: dict[str, Any],
        *,
        part_name: str,
        projected_part_entries: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        grounding_parts = dict(
            (prepared_bridge_request.get("grounding_context") or {}).get("parts") or {}
        )
        part_context = dict(grounding_parts.get(str(part_name or "").strip()) or {})
        projected_entry = dict(
            (projected_part_entries or {}).get(str(part_name or "").strip()) or {}
        )
        if projected_entry:
            part_context.update(deepcopy(projected_entry))
        target = dict(part_context.get("target") or {})
        if target:
            part_context["target"] = target
        if part_context.get("pose") is None and part_context.get("observed_pose") is not None:
            part_context["pose"] = deepcopy(part_context.get("observed_pose"))
        return part_context

    def _bridge_feasibility_decision(
        self,
        prepared_bridge_request: dict[str, Any],
        *,
        resource_jid: str,
        operation_kind: str,
        part_name: str | None = None,
        projected_part_entries: dict[str, dict[str, Any]] | None = None,
        projected_resource_snapshots: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        decision = {
            "allowed": True,
            "resource_jid": str(resource_jid or "").strip(),
            "operation_kind": str(operation_kind or "").strip(),
            "part_name": str(part_name or "").strip() or None,
            "reason": "",
            "evidence": {},
        }
        oracle_operation_kind = self._bridge_feasibility_operation_kind(operation_kind)
        resource = self._resource_by_jid(resource_jid)
        part_context = self._bridge_part_context(
            prepared_bridge_request,
            part_name=str(part_name or "").strip(),
            projected_part_entries=projected_part_entries,
        )
        snapshot = deepcopy(
            (projected_resource_snapshots or {}).get(str(resource_jid or "").strip())
            or (
                (prepared_bridge_request.get("bridge_resources") or {})
                .get(str(resource_jid or "").strip(), {})
                .get("bridge_snapshot")
            )
            or {}
        )
        adapter_meta = deepcopy(
            (
                (prepared_bridge_request.get("bridge_resources") or {})
                .get(str(resource_jid or "").strip(), {})
                .get("bridge_adapter")
            )
            or {}
        )
        evidence = {
            "part_context": deepcopy(part_context),
            "resource_snapshot": snapshot,
            "static_capabilities": deepcopy(getattr(resource, "static_capabilities", {}) or {}),
            "bridge_adapter": adapter_meta,
            "oracle_operation_kind": oracle_operation_kind,
        }
        if not adapter_meta.get("supports_executable_bridge", True):
            decision["allowed"] = False
            decision["reason"] = (
                f"resource type '{adapter_meta.get('resource_type') or 'resource'}' "
                "does not advertise executable bridge primitives"
            )
            decision["evidence"] = evidence
            return decision
        enforce_low_bias_executor_rules = (
            self._bridge_hint_level(prepared_bridge_request) is BridgeHintLevel.NONE
        )
        if (
            enforce_low_bias_executor_rules
            and oracle_operation_kind in {"pick", "pick_place"}
            and str(part_name or "").strip()
        ):
            pose_status = str(part_context.get("pose_status") or "").strip().lower()
            target_pose = part_context.get("observed_pose") or part_context.get("pose")
            if target_pose is None and pose_status in {"unknown", "carried", ""}:
                decision["allowed"] = False
                decision["reason"] = (
                    "pick target pose is ungrounded after prior bridge events; "
                    "request a new observation or use a grounded destination before reassigning pickup"
                )
                decision["evidence"] = evidence
                return decision
        oracle = getattr(resource, "bridge_feasibility_oracle", None) if resource is not None else None
        if callable(oracle):
            try:
                raw = oracle(
                    operation_kind=oracle_operation_kind,
                    part_name=str(part_name or "").strip() or None,
                    part_context=deepcopy(part_context),
                    bridge_snapshot=deepcopy(snapshot),
                )
            except Exception as exc:
                decision["allowed"] = False
                decision["reason"] = f"bridge feasibility oracle raised {exc!r}"
                decision["evidence"] = evidence
                return decision
            if isinstance(raw, dict):
                decision["allowed"] = bool(raw.get("allowed", False))
                decision["reason"] = str(raw.get("reason", "") or "").strip()
                decision["evidence"] = deepcopy(raw.get("evidence") or evidence)
                return decision

        decision["reason"] = "no bridge feasibility oracle available; planner fallback allowed"
        decision["evidence"] = evidence
        return decision

    def _bridge_validate_executor_assignments(
        self,
        prepared_bridge_request: dict[str, Any],
        *,
        events: list[dict[str, Any]],
    ) -> tuple[bool, str | None]:
        projected_resources = self._bridge_effective_resource_facts(
            bridge_resources=deepcopy(prepared_bridge_request.get("bridge_resources") or {}),
            projected_resource_snapshots=None,
        )
        part_states, part_locations, part_entries = self._bridge_effective_part_facts(
            fallback_part_tracker=deepcopy(prepared_bridge_request.get("part_tracker") or {}),
            projected_parts=None,
        )
        grounding_parts = dict(
            (prepared_bridge_request.get("grounding_context") or {}).get("parts") or {}
        )
        executor_bindings: dict[str, dict[str, Any]] = {}
        handoff_requirements: dict[str, dict[str, Any]] = {}

        for event in events:
            if not isinstance(event, dict):
                continue
            normalized_event = canonical_bridge_event(
                event,
                bridge_resources=deepcopy(prepared_bridge_request.get("bridge_resources") or {}),
            )
            resource_jid = str(normalized_event.get("resource_jid", "") or "").strip()
            part_name = str(normalized_event.get("part_name", "") or "").strip()
            semantic_kind = self._bridge_event_semantic_kind(normalized_event)
            event_name = str(normalized_event.get("event_name", "") or resource_jid).strip()

            if part_name and semantic_kind == "pick":
                current_binding = dict(executor_bindings.get(part_name) or {})
                if current_binding and str(current_binding.get("resource_jid") or "").strip() != resource_jid:
                    handoff = dict(handoff_requirements.get(part_name) or {})
                    if not bool(handoff.get("grounded_destination_available", False)):
                        return False, (
                            f"bridge event '{event_name}' switches executor for '{part_name}' "
                            f"from {current_binding.get('resource_jid')} to {resource_jid} "
                            "without a grounded handoff or new observation"
                        )

            self._bridge_apply_event_projection(
                normalized_event,
                projected_resources=projected_resources,
                projected_part_states=part_states,
                projected_part_locations=part_locations,
                projected_part_entries=part_entries,
                grounding_parts=grounding_parts,
            )
            self._bridge_update_executor_memory_after_event(
                normalized_event=normalized_event,
                projected_part_entries=part_entries,
                grounding_parts=grounding_parts,
                executor_bindings=executor_bindings,
                handoff_requirements=handoff_requirements,
            )
        return True, None

    def _bridge_validate_bridge_events(
        self,
        prepared_bridge_request: dict[str, Any],
        *,
        events: list[dict[str, Any]],
        require_full_gamma_closure: bool = True,
        require_primitive_preview: bool = True,
    ) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]], str | None]:
        normalized_events = [
            canonical_bridge_event(
                event,
                bridge_resources=prepared_bridge_request.get("bridge_resources") or {},
            )
            for event in (events or [])
            if isinstance(event, dict)
        ]
        marked_reentry_context = deepcopy(
            prepared_bridge_request.get("marked_reentry_context")
            or prepared_bridge_request.get("continuation_context")
            or {}
        )
        unmet_conditions = [
            condition
            for condition in (marked_reentry_context.get("unmet_reentry_conditions") or [])
            if isinstance(condition, dict)
        ]
        unmet_keys = {
            self._marked_reentry_condition_key(condition)
            for condition in unmet_conditions
        }
        covered_keys = self._bridge_event_condition_keys(normalized_events)
        missing_keys = [condition for condition in unmet_conditions if self._marked_reentry_condition_key(condition) not in covered_keys]
        feasibility_decisions: list[dict[str, Any]] = []
        allowed_operation_kinds = all_registered_operation_kinds()
        projected_resources = self._bridge_effective_resource_facts(
            bridge_resources=deepcopy(prepared_bridge_request.get("bridge_resources") or {}),
            projected_resource_snapshots=None,
        )
        part_states, part_locations, part_entries = self._bridge_effective_part_facts(
            fallback_part_tracker=deepcopy(prepared_bridge_request.get("part_tracker") or {}),
            projected_parts=None,
        )
        grounding_parts = dict(
            (prepared_bridge_request.get("grounding_context") or {}).get("parts") or {}
        )

        for event in normalized_events:
            if not isinstance(event, dict):
                continue
            resource_jid = str(event.get("resource_jid", "") or "").strip()
            part_name = str(event.get("part_name", "") or "").strip()
            operation_kind = self._bridge_event_operation_kind(event)
            if resource_jid and operation_kind in allowed_operation_kinds:
                decision = self._bridge_feasibility_decision(
                    prepared_bridge_request,
                    resource_jid=resource_jid,
                    operation_kind=operation_kind,
                    part_name=part_name or None,
                    projected_part_entries=part_entries,
                    projected_resource_snapshots=projected_resources,
                )
                feasibility_decisions.append(decision)
                if not decision.get("allowed", False):
                    return None, feasibility_decisions, (
                        f"bridge event '{str(event.get('event_name', '')).strip() or resource_jid}' "
                        f"is infeasible on {resource_jid}: {decision.get('reason') or 'unknown reason'}"
                    )
            self._bridge_apply_event_projection(
                event,
                projected_resources=projected_resources,
                projected_part_states=part_states,
                projected_part_locations=part_locations,
                projected_part_entries=part_entries,
                grounding_parts=grounding_parts,
            )

        # --- state-delta consistency check ---
        resource_projected_state: dict[str, str] = {}
        stuck_state = dict(prepared_bridge_request.get("stuck_state") or {})
        for jid, raw_entry in (
            prepared_bridge_request.get("bridge_resources") or {}
        ).items():
            if isinstance(raw_entry, dict):
                snapshot = raw_entry.get("bridge_snapshot") or {}
                state = str(snapshot.get("current_state") or "").strip()
                if state:
                    resource_projected_state[str(jid)] = state

        for event in normalized_events:
            if not isinstance(event, dict):
                continue
            resource_delta = event.get("expected_resource_delta")
            if not isinstance(resource_delta, dict):
                continue
            resource_jid = str(event.get("resource_jid", "")).strip()
            delta_from = str(resource_delta.get("from", "")).strip()
            delta_to = str(resource_delta.get("to", "")).strip()
            if not resource_jid or not delta_from or not delta_to:
                continue
            current = resource_projected_state.get(resource_jid, "")
            if current and current != delta_from:
                event_name = str(event.get("event_name", "")).strip() or resource_jid
                return None, feasibility_decisions, (
                    f"bridge event '{event_name}' declares expected_resource_delta.from='{delta_from}' "
                    f"but {resource_jid} is projected to be in state '{current}' at that point"
                )
            resource_projected_state[resource_jid] = delta_to

        continuity_ok, continuity_error = self._bridge_validate_event_state_continuity(
            prepared_bridge_request,
            events=normalized_events,
        )
        if not continuity_ok:
            return None, feasibility_decisions, continuity_error

        if self._bridge_hint_level(prepared_bridge_request) is BridgeHintLevel.NONE:
            executor_ok, executor_error = self._bridge_validate_executor_assignments(
                prepared_bridge_request,
                events=normalized_events,
            )
            if not executor_ok:
                return None, feasibility_decisions, executor_error

        if require_full_gamma_closure and missing_keys:
            missing_lines = [
                f"- {condition.get('entity')}.{condition.get('field')} -> {condition.get('expected')!r}"
                for condition in missing_keys[:6]
            ]
            return None, feasibility_decisions, (
                "bridge_events did not close all current Gamma(x_d, M_bridge) conditions:\n"
                + "\n".join(missing_lines)
            )

        safety_ok, safety_error = self._bridge_validate_safety_constraints(
            prepared_bridge_request,
            events=normalized_events,
        )
        if not safety_ok:
            return None, feasibility_decisions, safety_error

        preview_bridge_events = getattr(self, "_bridge_preview_approved_events", None)
        if require_primitive_preview and callable(preview_bridge_events):
            _preview_plan, _preview_normalized, preview_error = preview_bridge_events(
                prepared_bridge_request,
                approved_events=normalized_events,
            )
            if preview_error:
                return None, feasibility_decisions, preview_error

        return deepcopy(normalized_events), feasibility_decisions, None

    def _bridge_validate_event_state_continuity(
        self,
        prepared_bridge_request: dict[str, Any],
        *,
        events: list[dict[str, Any]],
    ) -> tuple[bool, str | None]:
        projected_resources = self._bridge_effective_resource_facts(
            bridge_resources=deepcopy(prepared_bridge_request.get("bridge_resources") or {}),
            projected_resource_snapshots=None,
        )
        projected_part_states, projected_part_locations, _ = self._bridge_effective_part_facts(
            fallback_part_tracker=deepcopy(prepared_bridge_request.get("part_tracker") or {}),
        )

        for event in events:
            if not isinstance(event, dict):
                continue
            normalized_event = canonical_bridge_event(
                event,
                bridge_resources=deepcopy(prepared_bridge_request.get("bridge_resources") or {}),
            )
            resource_jid = str(normalized_event.get("resource_jid", "") or "").strip()
            part_name = str(normalized_event.get("part_name", "") or "").strip()
            event_name = (
                str(normalized_event.get("event_name", "") or "").strip()
                or resource_jid
                or "bridge_event"
            )
            resource_entry = dict(projected_resources.get(resource_jid) or {})
            resource_type = (
                str(resource_entry.get("resource_type") or "resource").strip().lower()
                or "resource"
            )
            profile = get_resource_profile(resource_type)

            next_projected_resources = deepcopy(projected_resources)
            next_projected_part_states = deepcopy(projected_part_states)
            next_projected_part_locations = deepcopy(projected_part_locations)
            self._bridge_apply_event_projection(
                normalized_event,
                projected_resources=next_projected_resources,
                projected_part_states=next_projected_part_states,
                projected_part_locations=next_projected_part_locations,
            )

            if profile.event_state_validator is not None:
                validator_error = profile.event_state_validator(
                    event=normalized_event,
                    resource_jid=resource_jid,
                    before_resource=resource_entry,
                    after_resource=dict(next_projected_resources.get(resource_jid) or {}),
                    before_part_states=deepcopy(projected_part_states),
                    before_part_locations=deepcopy(projected_part_locations),
                    after_part_states=deepcopy(next_projected_part_states),
                    after_part_locations=deepcopy(next_projected_part_locations),
                    profile=profile,
                )
                if validator_error:
                    return False, validator_error

            projected_resources = next_projected_resources
            projected_part_states = next_projected_part_states
            projected_part_locations = next_projected_part_locations

        return True, None

    @staticmethod
    def _bridge_constraint_before_conditions(constraint: dict[str, Any]) -> list[dict[str, Any]]:
        before_conditions = [
            item
            for item in (constraint.get("before_conditions") or [])
            if isinstance(item, dict)
        ]
        if before_conditions:
            return before_conditions
        before_condition = constraint.get("before_condition")
        if isinstance(before_condition, dict):
            return [before_condition]
        return []

    @classmethod
    def _bridge_constraint_has_precedence_shape(cls, constraint: dict[str, Any]) -> bool:
        return bool(
            str(constraint.get("after_event_kind", "")).strip()
            and cls._bridge_constraint_before_conditions(constraint)
        )

    @staticmethod
    def _bridge_constraint_has_shared_location_shape(constraint: dict[str, Any]) -> bool:
        resource_jids = [
            str(item).strip()
            for item in (constraint.get("resource_jids") or [])
            if str(item).strip()
        ]
        return bool(
            str(constraint.get("location", "")).strip()
            and len(resource_jids) >= 2
        )

    @staticmethod
    def _bridge_constraint_has_guarded_target_shape(constraint: dict[str, Any]) -> bool:
        until_conditions = [
            item
            for item in (constraint.get("until_conditions") or [])
            if isinstance(item, dict)
        ]
        return bool(
            str(constraint.get("forbidden_location", "")).strip()
            and until_conditions
        )

    def _bridge_validate_event_precedence_constraint(
        self,
        *,
        constraint: dict[str, Any],
        normalized_event: dict[str, Any],
        event_name: str,
        semantic_kind: str,
        part_name: str,
        projected_resources: dict[str, dict[str, Any]],
        projected_part_states: dict[str, Any],
        projected_part_locations: dict[str, Any],
    ) -> str | None:
        if str(constraint.get("after_event_kind", "")).strip() != semantic_kind:
            return None
        after_part_name = str(constraint.get("after_part_name", "")).strip()
        if after_part_name and after_part_name != part_name:
            return None
        before_conditions = [
            item
            for item in (constraint.get("before_conditions") or [])
            if isinstance(item, dict)
        ]
        if not before_conditions:
            before_condition = constraint.get("before_condition")
            if isinstance(before_condition, dict):
                before_conditions = [before_condition]
        unsatisfied = [
            condition
            for condition in before_conditions
            if not self._bridge_condition_is_satisfied(
                condition,
                projected_resources=projected_resources,
                projected_part_states=projected_part_states,
                projected_part_locations=projected_part_locations,
            )
        ]
        if not unsatisfied:
            return None
        condition = unsatisfied[0]
        return (
            f"bridge event '{event_name}' violates bridge safety ordering: "
            f"{condition.get('entity')}.{condition.get('field')} must be "
            f"{condition.get('expected')!r} before {semantic_kind} of "
            f"{after_part_name or part_name or 'the targeted entity'}"
        )

    def _bridge_validate_shared_location_constraint(
        self,
        *,
        constraint: dict[str, Any],
        normalized_event: dict[str, Any],
        event_name: str,
        resource_jid: str,
        projected_resources: dict[str, dict[str, Any]],
        **_: Any,
    ) -> str | None:
        protected_location = str(constraint.get("location", "")).strip()
        protected_resources = [
            str(item).strip()
            for item in (constraint.get("resource_jids") or [])
            if str(item).strip()
        ]
        event_target_location = self._bridge_event_target_location(normalized_event)
        enters_protected_location = (
            protected_location
            and event_target_location == protected_location
        )
        if (
            not resource_jid
            or resource_jid not in protected_resources
            or not enters_protected_location
        ):
            return None
        blocking_resource = next(
            (
                other_resource_jid
                for other_resource_jid in protected_resources
                if other_resource_jid != resource_jid
                and str(
                    (projected_resources.get(other_resource_jid) or {}).get(
                        "current_location"
                    )
                    or ""
                ).strip()
                == protected_location
            ),
            "",
        )
        if not blocking_resource:
            return None
        rule_id = str(constraint.get("rule_id", "")).strip()
        rule_label = f" under {rule_id}" if rule_id else ""
        return (
            f"bridge event '{event_name}' violates bridge safety{rule_label}: "
            f"{resource_jid} cannot enter {protected_location!r} while "
            f"{blocking_resource} is still projected there"
        )

    def _bridge_validate_guarded_target_constraint(
        self,
        *,
        constraint: dict[str, Any],
        normalized_event: dict[str, Any],
        event_name: str,
        part_name: str,
        projected_resources: dict[str, dict[str, Any]],
        projected_part_states: dict[str, Any],
        projected_part_locations: dict[str, Any],
        **_: Any,
    ) -> str | None:
        constrained_part = str(constraint.get("part_name", "")).strip()
        forbidden_location = str(constraint.get("forbidden_location", "")).strip()
        if constrained_part and constrained_part != part_name:
            return None
        event_location = self._bridge_event_target_location(normalized_event)
        if not forbidden_location or event_location != forbidden_location:
            return None
        until_conditions = [
            item
            for item in (constraint.get("until_conditions") or [])
            if isinstance(item, dict)
        ]
        unsatisfied = [
            condition
            for condition in until_conditions
            if not self._bridge_condition_is_satisfied(
                condition,
                projected_resources=projected_resources,
                projected_part_states=projected_part_states,
                projected_part_locations=projected_part_locations,
            )
        ]
        if not unsatisfied:
            return None
        return (
            f"bridge event '{event_name}' violates bridge safety: "
            f"{constrained_part or part_name or 'part'} cannot be moved to "
            f"{forbidden_location!r} before marked re-entry conditions are satisfied"
        )

    @staticmethod
    def _bridge_event_expected_part_delta(event: dict[str, Any]) -> dict[str, Any]:
        normalized = canonical_bridge_event(event)
        delta = normalized.get("expected_part_delta") or {}
        return dict(delta) if isinstance(delta, dict) else {}

    @staticmethod
    def _bridge_event_target_location(event: dict[str, Any]) -> str:
        normalized = canonical_bridge_event(event)
        projected_effects = dict(normalized.get("projected_effects") or {})
        occupancy = dict(projected_effects.get("occupancy") or {})
        if occupancy.get("location") not in (None, ""):
            return str(occupancy.get("location") or "").strip()
        targets = dict(normalized.get("targets") or {})
        return str(targets.get("location", "") or "").strip()

    def _bridge_event_semantic_kind(self, event: dict[str, Any]) -> str:
        operation_family = str(event.get("operation_family", "") or "").strip()
        if operation_family:
            return operation_family
        return str(canonical_bridge_event(event).get("operation_family", "") or "bridge").strip()

    @staticmethod
    def _bridge_part_target_info(
        prepared_bridge_request: dict[str, Any],
        *,
        part_name: str,
    ) -> dict[str, Any]:
        parts = dict((prepared_bridge_request.get("grounding_context") or {}).get("parts") or {})
        return dict((parts.get(str(part_name or "").strip()) or {}).get("target") or {})

    @staticmethod
    def _bridge_part_goal_requirements(
        marked_reentry_context: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        goals: dict[str, dict[str, Any]] = {}
        for requirement in (marked_reentry_context.get("marked_reentry_conditions") or []):
            if not isinstance(requirement, dict):
                continue
            if str(requirement.get("role", "")).strip() != "bridge_replaced":
                continue
            if str(requirement.get("entity_kind", "")).strip() != "part":
                continue
            part_name = str(requirement.get("entity", "")).strip()
            field = str(requirement.get("field", "")).strip()
            if not part_name or field not in {"state", "location"}:
                continue
            goals.setdefault(part_name, {})[field] = deepcopy(requirement.get("expected"))
        return goals

    def _derive_bridge_safety_constraints(
        self,
        prepared_bridge_request: dict[str, Any],
        *,
        marked_reentry_context: dict[str, Any],
    ) -> dict[str, Any]:
        raw = deepcopy(prepared_bridge_request.get("bridge_safety_context") or {})
        if not isinstance(raw, dict):
            raw = {}

        constraints: list[dict[str, Any]] = [
            deepcopy(item)
            for item in (raw.get("constraints") or [])
            if isinstance(item, dict)
        ]
        safety_rules: list[dict[str, Any]] = [
            deepcopy(item)
            for item in (raw.get("safety_rules") or [])
            if isinstance(item, dict)
        ]
        rule_ids = [
            str(rule_id).strip()
            for rule_id in (raw.get("rule_ids") or [])
            if str(rule_id).strip()
        ]
        seen_constraints: set[str] = {
            self._bridge_constraint_signature(constraint)
            for constraint in constraints
            if self._bridge_constraint_signature(constraint)
        }
        seen_safety_rule_signatures: set[str] = set()
        deduped_safety_rules: list[dict[str, Any]] = []
        for rule in safety_rules:
            signature = self._bridge_safety_rule_signature(rule)
            if not signature or signature in seen_safety_rule_signatures:
                continue
            seen_safety_rule_signatures.add(signature)
            deduped_safety_rules.append(rule)
            rule_id = str(rule.get("id", "") or "").strip()
            if rule_id and rule_id not in rule_ids:
                rule_ids.append(rule_id)

        part_states, part_locations, _ = self._bridge_effective_part_facts(
            fallback_part_tracker=deepcopy(prepared_bridge_request.get("part_tracker") or {}),
        )
        effective_resources = self._bridge_effective_resource_facts(
            bridge_resources=deepcopy(prepared_bridge_request.get("bridge_resources") or {}),
            projected_resource_snapshots=None,
        )
        bridge_part_goals = self._bridge_part_goal_requirements(marked_reentry_context)
        for rule in deduped_safety_rules:
            resource_jids = self._bridge_rule_resource_jids(
                rule=rule,
                bridge_resources=deepcopy(prepared_bridge_request.get("bridge_resources") or {}),
            )
            constraint = canonical_bridge_constraint_from_rule(
                rule=rule,
                resource_jids=resource_jids,
            )
            if not isinstance(constraint, dict):
                continue
            signature = self._bridge_constraint_signature(constraint)
            if signature and signature not in seen_constraints:
                seen_constraints.add(signature)
                constraints.append(constraint)

        for suffix in (marked_reentry_context.get("pending_suffix_summary") or []):
            if not isinstance(suffix, dict):
                continue
            if str(suffix.get("role", "")).strip() != "resume_suffix":
                continue
            resource_jid = str(suffix.get("resource_jid", "")).strip()
            entry_part_name = str(suffix.get("entry_part_name", "")).strip()
            if not entry_part_name:
                suffix_parts = [
                    str(part_name).strip()
                    for part_name in (suffix.get("parts") or [])
                    if str(part_name).strip()
                ]
                if len(suffix_parts) == 1:
                    entry_part_name = suffix_parts[0]
            if not resource_jid or not entry_part_name:
                continue
            resource_entry = dict(effective_resources.get(resource_jid) or {})
            profile = get_resource_profile(str(resource_entry.get("resource_type") or "resource"))
            carried_entity = str(
                resource_snapshot_carried_entity(
                    resource_entry,
                    profile=profile,
                )
                or ""
            ).strip()
            carried_location = ""
            if profile.carried_entity_location_builder is not None:
                carried_location = str(
                    profile.carried_entity_location_builder(resource_jid, resource_entry) or ""
                ).strip()
            part_location = str(part_locations.get(entry_part_name) or "").strip()
            if carried_entity != entry_part_name and part_location != carried_location:
                continue
            for bridge_part_name, goal_info in bridge_part_goals.items():
                if not bridge_part_name or bridge_part_name == entry_part_name:
                    continue
                goal_state = goal_info.get("state")
                goal_location = goal_info.get("location")
                if goal_state is None:
                    continue
                if (
                    part_states.get(bridge_part_name) == goal_state
                    and (goal_location is None or part_locations.get(bridge_part_name) == goal_location)
                ):
                    continue
                before_conditions = [
                    {
                        "entity_kind": "part",
                        "entity": bridge_part_name,
                        "field": "state",
                        "expected": deepcopy(goal_state),
                    }
                ]
                if goal_location not in (None, ""):
                    before_conditions.append(
                        {
                            "entity_kind": "part",
                            "entity": bridge_part_name,
                            "field": "location",
                            "expected": deepcopy(goal_location),
                        }
                    )
                resume_target_location = str(
                    (
                        self._bridge_part_target_info(
                            prepared_bridge_request,
                            part_name=entry_part_name,
                        ).get("location")
                    )
                    or ""
                ).strip()
                if resume_target_location:
                    constraint = {
                        "part_name": entry_part_name,
                        "forbidden_location": resume_target_location,
                        "until_conditions": before_conditions,
                        "reason": (
                            "resume-suffix parts may not be staged at their protected goal "
                            "location before bridge-replaced parts satisfy marked re-entry"
                        ),
                        "rule_id": rule_ids[0] if rule_ids else "",
                    }
                    signature = self._bridge_constraint_signature(constraint)
                    if signature in seen_constraints:
                        continue
                    seen_constraints.add(signature)
                    constraints.append(constraint)

        raw["constraints"] = constraints
        raw["rule_ids"] = rule_ids
        raw["safety_rules"] = deduped_safety_rules
        return raw

    @staticmethod
    def _bridge_condition_is_satisfied(
        condition: dict[str, Any],
        *,
        projected_resources: dict[str, dict[str, Any]],
        projected_part_states: dict[str, Any],
        projected_part_locations: dict[str, Any],
    ) -> bool:
        entity_kind = str(condition.get("entity_kind", "")).strip()
        entity = str(condition.get("entity", "")).strip()
        field = str(condition.get("field", "")).strip()
        expected = condition.get("expected")
        actual = None
        if entity_kind == "resource":
            actual = (projected_resources.get(entity) or {}).get(field)
        elif entity_kind == "part":
            if field == "state":
                actual = projected_part_states.get(entity)
            elif field == "location":
                actual = projected_part_locations.get(entity)
        return actual == expected

    def _bridge_apply_event_projection(
        self,
        event: dict[str, Any],
        *,
        projected_resources: dict[str, dict[str, Any]],
        projected_part_states: dict[str, Any],
        projected_part_locations: dict[str, Any],
        projected_part_entries: dict[str, dict[str, Any]] | None = None,
        grounding_parts: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        raw_resource_jid = str((event or {}).get("resource_jid", "")).strip()
        raw_resource_entry = dict(projected_resources.get(raw_resource_jid) or {})
        raw_resource_type = str(raw_resource_entry.get("resource_type") or "resource").strip().lower() or "resource"
        normalized_event = canonical_bridge_event(event, resource_type=raw_resource_type)
        resource_jid = str(normalized_event.get("resource_jid", "")).strip() or raw_resource_jid
        part_name = str(normalized_event.get("part_name", "")).strip()
        semantic_kind = self._bridge_event_semantic_kind(normalized_event)
        resource_delta = dict(normalized_event.get("expected_resource_delta") or {})
        part_delta = self._bridge_event_expected_part_delta(normalized_event)
        projected_effects = dict(normalized_event.get("projected_effects") or {})
        occupancy_effects = dict(projected_effects.get("occupancy") or {})

        resource_entry = dict(projected_resources.get(resource_jid) or {})
        resource_type = str(resource_entry.get("resource_type") or "resource").strip().lower() or "resource"
        profile = get_resource_profile(resource_type)
        delta_to = str(resource_delta.get("to", "")).strip()
        if delta_to:
            resource_entry["current_state"] = delta_to
        if "location" in occupancy_effects:
            resource_entry["current_location"] = deepcopy(occupancy_effects.get("location"))
        elif "location" in resource_delta and resource_delta.get("location") in (None, ""):
            resource_entry["current_location"] = None
        elif semantic_kind in {"clear", "home"}:
            # A clear/home bridge event semantically exits the currently occupied protected area
            # even when the high-level event omits an explicit destination location.
            resource_entry["current_location"] = None

        if part_name and part_delta:
            part_to = str(part_delta.get("to", "")).strip()
            if part_to:
                projected_part_states[part_name] = part_to
            location_to = part_delta.get("location_to")
            if location_to not in (None, ""):
                projected_part_locations[part_name] = location_to
        part_entry = dict((projected_part_entries or {}).get(part_name) or {})
        grounding_part = dict((grounding_parts or {}).get(part_name) or {})

        if profile.state_projector and part_name:
            resource_entry["_target_location"] = self._bridge_event_target_location(normalized_event)
            profile.state_projector(
                resource_entry=resource_entry,
                semantic_kind=semantic_kind,
                part_name=part_name,
                part_delta=part_delta,
                projected_part_states=projected_part_states,
                projected_part_locations=projected_part_locations,
                resource_jid=resource_jid,
            )
            resource_entry.pop("_target_location", None)

        if part_name and projected_part_entries is not None:
            if part_name in projected_part_states:
                part_entry["state"] = deepcopy(projected_part_states.get(part_name))
            if part_name in projected_part_locations:
                part_entry["location"] = deepcopy(projected_part_locations.get(part_name))
                part_entry["last_known_location"] = deepcopy(projected_part_locations.get(part_name))

            target = dict(grounding_part.get("target") or {})
            semantic_kind = self._bridge_event_semantic_kind(normalized_event)
            if semantic_kind == "pick":
                part_entry["observed_pose"] = None
                part_entry["pose"] = None
                part_entry["pose_status"] = "carried"
            elif semantic_kind in {"assemble", "place", "pick_place"}:
                target_location = str(target.get("location") or "").strip()
                current_location = str(part_entry.get("location") or "").strip()
                projected_target_pose = self._coerce_xyz_pose(
                    target.get("slot_pose") or target.get("pose")
                )
                if projected_target_pose is not None and current_location and current_location == target_location:
                    part_entry["observed_pose"] = deepcopy(projected_target_pose)
                    part_entry["pose"] = deepcopy(projected_target_pose)
                    part_entry["pose_status"] = "projected_target"
                elif semantic_kind != "pick_place":
                    part_entry["observed_pose"] = None
                    part_entry["pose"] = None
                    part_entry["pose_status"] = "unknown"
            elif semantic_kind == "stage":
                part_entry["observed_pose"] = None
                part_entry["pose"] = None
                part_entry["pose_status"] = "unknown"

            projected_part_entries[part_name] = part_entry
        projected_resources[resource_jid] = resource_entry

    def _bridge_validate_safety_constraints(
        self,
        prepared_bridge_request: dict[str, Any],
        *,
        events: list[dict[str, Any]],
    ) -> tuple[bool, str | None]:
        safety_context = deepcopy(prepared_bridge_request.get("bridge_safety_context") or {})
        constraints = [
            dict(item)
            for item in (safety_context.get("constraints") or [])
            if isinstance(item, dict)
        ]
        if not constraints:
            return True, None

        part_states, part_locations, _ = self._bridge_effective_part_facts(
            fallback_part_tracker=deepcopy(prepared_bridge_request.get("part_tracker") or {}),
        )
        projected_resources = self._bridge_effective_resource_facts(
            bridge_resources=deepcopy(prepared_bridge_request.get("bridge_resources") or {}),
            projected_resource_snapshots=None,
        )

        for event in events:
            if not isinstance(event, dict):
                continue
            normalized_event = canonical_bridge_event(
                event,
                bridge_resources=deepcopy(prepared_bridge_request.get("bridge_resources") or {}),
            )
            resource_jid = str(normalized_event.get("resource_jid", "")).strip()
            semantic_kind = self._bridge_event_semantic_kind(normalized_event)
            part_name = str(normalized_event.get("part_name", "")).strip()
            event_name = str(event.get("event_name", "")).strip() or semantic_kind or "bridge_event"

            for constraint in constraints:
                constraint_error = None
                if self._bridge_constraint_has_precedence_shape(constraint):
                    constraint_error = self._bridge_validate_event_precedence_constraint(
                        constraint=constraint,
                        normalized_event=normalized_event,
                        event_name=event_name,
                        resource_jid=resource_jid,
                        semantic_kind=semantic_kind,
                        part_name=part_name,
                        projected_resources=projected_resources,
                        projected_part_states=part_states,
                        projected_part_locations=part_locations,
                    )
                elif self._bridge_constraint_has_shared_location_shape(constraint):
                    constraint_error = self._bridge_validate_shared_location_constraint(
                        constraint=constraint,
                        normalized_event=normalized_event,
                        event_name=event_name,
                        resource_jid=resource_jid,
                        semantic_kind=semantic_kind,
                        part_name=part_name,
                        projected_resources=projected_resources,
                        projected_part_states=part_states,
                        projected_part_locations=part_locations,
                    )
                elif self._bridge_constraint_has_guarded_target_shape(constraint):
                    constraint_error = self._bridge_validate_guarded_target_constraint(
                        constraint=constraint,
                        normalized_event=normalized_event,
                        event_name=event_name,
                        resource_jid=resource_jid,
                        semantic_kind=semantic_kind,
                        part_name=part_name,
                        projected_resources=projected_resources,
                        projected_part_states=part_states,
                        projected_part_locations=part_locations,
                    )
                if constraint_error:
                    return False, constraint_error

            self._bridge_apply_event_projection(
                normalized_event,
                projected_resources=projected_resources,
                projected_part_states=part_states,
                projected_part_locations=part_locations,
            )
        return True, None
