from __future__ import annotations

"""CCA ownership of predefined specifications and their physical admission gate."""

import json
from copy import deepcopy
from pathlib import Path

from cais_spade_llm.agents.central_controller.base_safety_checker import BaseSafetyChecker
from cais_spade_llm.agents.central_controller.online_safety_monitor import OnlineSafetyMonitor
from cais_spade_llm.agents.central_controller.predefined_safety import (
    compile_predefined_safety,
    parse_predefined_safety,
    predefined_safety_metadata,
    validate_predefined_safety_artifact,
)
from cais_spade_llm.agents.central_controller.reviewed_primitive_program_safety import _digest
from cais_spade_llm.agents.central_controller.ppr_ap import physical_ap_kind


def predefined_required(agent) -> bool:
    """Recognize fixed input before startup can publish any permissive monitor."""
    if getattr(agent, "predefined_safety_required", False):
        return True
    try:
        bundle = getattr(agent, "precomputed_bundle", {}) or {}
        if not isinstance(bundle, dict):
            raise ValueError("invalid_safety_bundle")
        source = bundle.get("safety_source", {})
        if not isinstance(source, dict):
            raise ValueError("invalid_safety_source")
        marked = source.get("mode") == "predefined" or source.get("definition_mode") == "predefined"
        marked = marked or any(key.startswith("predefined_") for key in source)
        path = getattr(agent, "safety_file", None)
        if path is not None:
            marked = Path(path).read_text(encoding="utf-8").lstrip().startswith("{") or marked
        artifacts = bundle.get("artifacts", {})
        if not isinstance(artifacts, dict):
            raise ValueError("invalid_safety_artifacts")
        logic_path = artifacts.get("safety_logic_json")
        if logic_path:
            payload = json.loads(Path(logic_path).read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("invalid_safety_artifact")
            marked = marked or payload.get("mode") == "predefined"
            marked = marked or any(key.startswith("predefined_") for key in payload)
            marked = marked or any(
                physical_ap_kind(ap) is not None for rule in payload.get("rules", [])
                if isinstance(rule, dict) for ap in rule.get("aps", []) if isinstance(ap, dict)
            )
    except (ValueError, TypeError, OSError):
        # A damaged selected source/artifact must not trigger runtime generation.
        marked = True
    if marked:
        agent.predefined_safety_required = True
    return marked


def initialize_predefined_safety(agent) -> bool:
    """Load fixed definitions or leave admission unavailable, without an LLM fallback.

    Returns:
        Whether this startup belongs to the predefined path, including failures.
    """
    if not predefined_required(agent):
        return False
    agent.predefined_safety_required = True
    agent.predefined_safety_error = "predefined_safety_not_ready"
    agent.safety_monitor = None
    try:
        from cais_spade_llm.agents.central_controller.predefined_safety import (
            predefined_safety_metadata,
        )

        bundle = getattr(agent, "precomputed_bundle", {}) or {}
        if not isinstance(bundle, dict) or not isinstance(bundle.get("artifacts", {}), dict):
            raise ValueError("invalid_safety_bundle")
        logic_path = bundle.get("artifacts", {}).get("safety_logic_json")
        source = Path(agent.safety_file).read_text(encoding="utf-8")
        document = parse_predefined_safety(source)
        if document is None:
            raise ValueError("predefined_source_unavailable")
        metadata = predefined_safety_metadata(document, source)
        bundle_source = bundle.get("safety_source", {})
        if not isinstance(bundle_source, dict):
            raise ValueError("invalid_safety_source")
        expected_source = bundle_source.get("safety_sha256")
        if expected_source is not None and expected_source != metadata["predefined_source_sha256"]:
            raise ValueError("predefined_bundle_source_changed")
        expected_semantics = bundle_source.get("predefined_semantics_sha256")
        if expected_semantics is not None and expected_semantics != metadata["predefined_semantics_sha256"]:
            raise ValueError("predefined_bundle_definitions_changed")
        if logic_path:
            payload = json.loads(Path(logic_path).read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("invalid_safety_artifact")
            if validate_predefined_safety_artifact(payload, source_text=source) != document:
                raise ValueError("predefined_artifact_disagrees_with_source")
        elif bundle_source:
            raise ValueError("predefined_artifact_missing")
        rules, compiled = compile_predefined_safety(document)
        if logic_path:
            for rule in rules:
                path = Path(logic_path).parent / f"{rule['id']}_dfa.dot"
                saved = BaseSafetyChecker({rule['id']: path.read_text()}, [rule])
                expected = BaseSafetyChecker({rule['id']: compiled[rule['id']]}, [rule])
                if saved.dfas != expected.dfas:
                    raise ValueError("predefined_dfa_changed")
        agent.predefined_safety = deepcopy(document)
        agent.predefined_safety_fingerprint = _digest(document)
        agent.predefined_geometry_sha256 = metadata["predefined_geometry_sha256"]
        agent.predefined_source_text = source
        agent.safety_rules = deepcopy(rules)
        agent.safety_logic.rules = deepcopy(rules)
        agent.safety_logic.rule_dfas = compiled
        agent.safety_logic.predefined_metadata = deepcopy(metadata)
        agent.safety_logic.safety_text_sha256 = metadata["predefined_source_sha256"]
        agent.safety_logic.global_safety_spec = {}
        # Physical APs must be consumed by the grounded gate. An empty native
        # monitor keeps its task clock available without interpreting them as false.
        agent.safety_monitor = OnlineSafetyMonitor({}, [], tools_catalog=agent._tools_catalog_for_safety())
        agent.predefined_safety_error = ""
        return True
    except (ValueError, KeyError, TypeError, OSError) as exc:
        agent.predefined_safety_error = str(exc)
        agent.logger.error("Predefined safety unavailable: %s", exc)
        return True


def bind_predefined_context(agent, context: dict) -> None:
    """Bind all selected rules from CCA, never from a proposed recovery plan."""
    if not predefined_required(agent):
        return
    error = getattr(agent, "predefined_safety_error", "predefined_safety_not_ready")
    if error:
        raise ValueError(error)
    source = Path(agent.safety_file).read_text(encoding="utf-8")
    current = parse_predefined_safety(source)
    if current != agent.predefined_safety or source != agent.predefined_source_text:
        raise ValueError("predefined_specifications_changed")
    # Compilation validates the configured assembly target and its geometry owner.
    compile_predefined_safety(current)
    if predefined_safety_metadata(current, source)["predefined_geometry_sha256"] != agent.predefined_geometry_sha256:
        raise ValueError("predefined_product_geometry_changed")
    grounding = context["composition_inputs"]["grounding_inputs"]
    for key in ("catalog", "requirement_scopes"):
        expected = current[key]
        if key in grounding and grounding[key] != expected:
            raise ValueError("owner_context_disagrees_with_predefined_" + key)
        grounding[key] = deepcopy(expected)
    context["predefined_safety_fingerprint"] = agent.predefined_safety_fingerprint


def predefined_scope(agent, scope_id: str) -> dict:
    """Associate a recovery scope with fixed rules without resetting their history."""
    if not scope_id or not isinstance(scope_id, str):
        raise ValueError("recovery_safety_scope_id_unavailable")
    if getattr(agent, "predefined_safety_error", "predefined_safety_not_ready"):
        raise ValueError(agent.predefined_safety_error)
    existing = agent.recovery_safety_scopes.get(scope_id)
    fingerprint = agent.predefined_safety_fingerprint
    if existing is not None:
        if existing.get("predefined_safety_fingerprint") != fingerprint:
            raise ValueError("recovery_scope_specifications_changed")
    else:
        agent.recovery_safety_scopes[scope_id] = {
            "status": "ready", "rules": [], "rule_dfas": {},
            "predefined_safety_fingerprint": fingerprint,
            "monitor": OnlineSafetyMonitor({}, []),
        }
    return {
        "ok": True, "recovery_safety_scope_id": scope_id,
        "recovery_safety_status": "ready", "rules": [], "rule_dfas": {},
        "rule_ids": [rule["id"] for rule in agent.safety_rules],
        "predefined_safety_fingerprint": fingerprint,
        "requires_grounded_composition": True,
        "recovery_safety_logic_json": "", "dfa_dot_files": [],
    }


def nominal_unavailable(agent) -> dict:
    """Explain why native-only admission cannot authorize physical requirements."""
    return {
        "status": "inconclusive",
        "reason": getattr(agent, "predefined_safety_error", "")
        or "predefined_physical_evidence_required",
    }


async def check_predefined_nominal_start(agent, task: dict, product_jid: str, *, commit: bool) -> dict:
    """Require an owner-registered joint proof before a nominal start.

    A provider may describe finite permitted nominal behavior through its
    ``nominal_request(product_jid, task)`` hook. It supplies evidence, never a
    permission result. CCA checks the same graph and exact start as recovery.
    """
    from cais_spade_llm.agents.central_controller.recovery_admission_runtime import (
        physical_admission,
        register_recovery_composition,
    )

    if getattr(agent, "predefined_safety_error", ""):
        return nominal_unavailable(agent)
    runtime = agent._environment_runtime_for_sender(product_jid)
    if runtime is not None:
        install = getattr(agent, "_environment_admission", None)
        if callable(install):
            install(runtime)
        live = getattr(runtime, "live_safety_runtime", None)
        if live is not None:
            if runtime.context.pending_for(task["task_id"]) != task:
                return {"status": "inconclusive", "reason": "nominal_task_identity_changed"}
            return await live.check_nominal(deepcopy(task), product_jid, commit=commit)
    provider = getattr(agent, "recovery_composition_context_provider", None)
    prepare = getattr(provider, "nominal_request", None)
    if prepare is None or getattr(agent, "predefined_safety_error", ""):
        return nominal_unavailable(agent)
    try:
        request = prepare(product_jid, deepcopy(task))
        if not isinstance(request, dict):
            raise ValueError("nominal_physical_registration_unavailable")
        runtime = agent._environment_runtime_for_sender(product_jid)
        if runtime is None or runtime.context.pending_for(task["task_id"]) != task:
            raise ValueError("nominal_task_identity_changed")
        requested = next(row for row in request["tasks"] if row["task_id"] == task["task_id"])
        if (requested["resource_id"] != task["resource_id"]
                or requested["resource_jid"] != runtime.jids[task["resource_id"]]
                or requested["function_name"] != task["event_name"]
                or requested["params"] != task["parameters"]):
            raise ValueError("nominal_dispatch_identity_mismatch")
        registration = await register_recovery_composition(agent, request, product_jid)
        if registration["status"] != "allowed":
            return registration
        event = {
            "task_id": task["task_id"], "resource_jid": requested["resource_jid"],
            "function_name": requested["function_name"], "status": "safety_check",
            "params": {**deepcopy(requested["params"]),
                       "recovery_composition_ref": registration["task_refs"][task["task_id"]]},
        }
        return await physical_admission(agent, product_jid).check(
            event, sender=requested["resource_jid"], commit=commit)
    except (ValueError, KeyError, TypeError, StopIteration, OSError) as exc:
        return {"status": "inconclusive", "reason": str(exc)}
