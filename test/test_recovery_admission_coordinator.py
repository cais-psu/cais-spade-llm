"""Authoritative recovery admission with explicitly configured mock evidence owners."""

from __future__ import annotations

import asyncio
import hashlib
import json
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

import pytest

from cais_spade_llm.agents.central_controller.base_safety_checker import BaseSafetyChecker
from cais_spade_llm.agents.central_controller.local_composition import Budget
from cais_spade_llm.agents.central_controller.recovery_composition_admission import (
    RecoveryCompositionAdmission,
)

_FIXTURE = Path(__file__).parent / "fixtures/KMR_assembly_board-v1_recovery/storage_interruption"
_PRODUCT = "recovery-product@localhost"
_SCOPE = "recovery_scope_KMR_storage_interruption"


def _case() -> dict:
    document = json.loads((_FIXTURE / "composition_evidence.json").read_text())
    reference = document["base_evidence"]
    path = _FIXTURE / reference["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == reference["sha256"]
    inputs = json.loads(path.read_text())["inputs"]
    inputs.update(document["grounding_input_overrides"])
    return {"grounding_inputs": inputs, **document["inputs"]}


class _Harness:
    def __init__(self) -> None:
        self.case = _case()
        snapshot = self.case["grounding_inputs"]["snapshot"]
        self.jids = {key: value.get("resource_jid") or key + "@localhost"
                     for key, value in snapshot["resources"].items()}
        tasks, bindings = [], {}
        for event in self.case["recovery_events"]:
            program = next(row for row in self.case["event_start_choices"][0]["programs"]
                           if row["resource_id"] == event["resource_id"])
            steps = [deepcopy(program["primitive_steps"][index]) for index in event["primitive_step_indices"]]
            task_id = event["outline_id"] + "_TASK"
            params = {"outline_id": event["outline_id"], "primitive_steps": steps,
                      "event_name": event["event_name"],
                      "product_jid": _PRODUCT, "task_id": task_id, "start_safety_mode": "cca_check",
                      "recovery_safety_scope_id": _SCOPE}
            tasks.append({"task_id": task_id, "outline_id": event["outline_id"],
                          "resource_id": event["resource_id"], "resource_jid": self.jids[event["resource_id"]],
                          "function_name": "execute_recovery_macro", "params": params})
            bindings[event["outline_id"]] = {
                "task": {"task_id": task_id, "resource_id": event["resource_id"],
                         "event_name": event["event_name"], "function_name": "execute_recovery_macro",
                         "parameters": deepcopy(params)},
                "participants": [event["resource_id"]], "resource_updates": {}, "product_updates": {},
            }
        for work in self.case["running_work"]:
            bindings[work["task_id"]] = {
                "task": {"task_id": work["task_id"], "resource_id": work["resource_id"],
                         "event_name": work["event_name"], "parameters": {}},
                "participants": [work["resource_id"]], "resource_updates": {}, "product_updates": {},
            }
        native = {
            "monitors": [{"scope_id": scope, "monitor": BaseSafetyChecker({}, []), "current_states": {}}
                         for scope in (None, _SCOPE)],
            "resources": {key: {} for key in self.jids}, "products": {}, "contexts": {},
            "jids": self.jids, "tasks": bindings,
        }
        self.context = {
            "revision": 0, "time_exact": "0", "current_snapshot": deepcopy(snapshot),
            "composition_inputs": self.case, "task_monitor_context": native, "observations": [],
            "execution_mode": "mock", "physical_rule_activation": "prospective",
        }
        self.request = {
            "recovery_id": "KMR_storage_interruption", "recovery_safety_scope_id": _SCOPE,
            "tasks": tasks, "pending_nominal_task_ids": [key for key, status in snapshot["task_statuses"].items()
                                                         if status == "pending"],
        }
        self.preparations, self.commits = [], []
        self.coordinator = RecoveryCompositionAdmission(
            context_provider=self.provide_context, resource_evidence_provider=self.prepare,
            native_history_commit=self.commit, allow_mock_execution=True,
            budget_factory=lambda: Budget(seconds=30),
        )
        self.registration = None

    def provide_context(self, product_jid, recovery_id):
        assert product_jid == _PRODUCT
        assert recovery_id == self.request["recovery_id"]
        return self.context

    def prepare(self, request):
        self.preparations.append(deepcopy(request))
        return {"status": "prepared", "resource_jid": request["resource_jid"],
                "program_hash": request["program_hash"], "execution_mode": "mock", "mock_executor": True,
                "preparation_id": "mock:" + request["task_id"]}

    def commit(self, state, record):
        self.commits.append(deepcopy(record))
        native = self.context["task_monitor_context"]
        for current, observed in zip(native["monitors"], state["monitors"], strict=True):
            assert current["scope_id"] == observed["scope_id"]
            current["current_states"] = deepcopy(observed["state"]["states"])
        self.context["revision"] += 1

    async def register(self):
        self.registration = await self.coordinator.register(self.request, product_jid=_PRODUCT)
        assert self.registration["status"] == "allowed", self.registration
        return self.registration

    def event(self, index):
        task = self.request["tasks"][index]
        return {"task_id": task["task_id"], "resource_jid": task["resource_jid"],
                "function_name": task["function_name"], "status": "safety_check",
                "params": {**deepcopy(task["params"]), "recovery_composition_ref":
                           deepcopy(self.registration["task_refs"][task["task_id"]])}}

    async def check(self, index, **kwargs):
        event = self.event(index)
        return await self.coordinator.check(event, sender=event["resource_jid"], **kwargs)

    @property
    def session(self):
        return self.coordinator.sessions[self.request["recovery_id"]]

    def feed(self, edge):
        """Mock evidence owner acknowledges one actual fixture transition."""
        target = self.session["nodes"][edge["target"]]
        self.context["time_exact"] = edge["time_exact"]
        self.context["current_snapshot"] = {key: deepcopy(target["state"][key]) for key in ("resources", "parts")}
        native_state = target["task_monitor_state"]["monitors"][0]["state"]
        for key in ("resources", "products", "contexts"):
            self.context["task_monitor_context"][key] = deepcopy(native_state[key])
        self.context["revision"] += 1
        if edge["kind"] in {"observation", "task_completion"}:
            record = {"record_id": edge["edge_id"], "kind": edge["kind"], "time_exact": edge["time_exact"]}
            if edge["kind"] == "observation":
                record["observation"] = deepcopy(edge["observation"])
            else:
                record.update(task_id=edge["task_id"], status="completed")
            self.context["observations"].append(record)
        result = self.coordinator.observe({"status": "owner_observation"}, sender=_PRODUCT)
        assert not self.session["invalid_reason"], result

    def advance_to(self, time):
        for _ in range(1000):
            edges = self.session["edges"].get(self.session["node"], [])
            if not edges:
                return
            forced = [edge for edge in edges if not edge["controllable"]]
            edge = forced[0] if forced else edges[0] if len(edges) == 1 and edges[0]["kind"] == "wait" else None
            if edge is None or Fraction(edge["time_exact"]) > Fraction(str(time)):
                return
            self.feed(edge)
        raise AssertionError("Mock graph walk did not terminate")


def _harness() -> _Harness:
    return _Harness()


def test_recovery_admission_requires_owner_context() -> None:
    async def scenario():
        harness = _harness()
        result = await RecoveryCompositionAdmission().register(harness.request, product_jid=_PRODUCT)
        assert result["status"] == "inconclusive"
        assert result["reason"] == "live_recovery_evidence_unavailable"
    asyncio.run(scenario())


def test_registered_primitive_models_reach_composition_and_invalidate_changed_contracts(monkeypatch):
    from test_continuous_motion import composition_case

    from cais_spade_llm.agents.central_controller import recovery_composition_admission as admission
    from cais_spade_llm.resources.resource_safety_preparation import PrimitiveModel

    async def scenario():
        case, models = composition_case()
        for rid, state in case["grounding_inputs"]["snapshot"]["resources"].items():
            state.setdefault("resource_jid", rid + "@localhost")
        for programs in (case["grounding_inputs"]["programs"], case["event_start_choices"][0]["programs"]):
            for program in programs:
                jid = case["grounding_inputs"]["snapshot"]["resources"][program["resource_id"]]["resource_jid"]
                for step in (*program["primitive_steps"], *program["step_results"]):
                    step["source"]["resource_jid"] = jid
        monkeypatch.setitem(globals(), "_case", lambda: case)
        harness = _harness()
        model = models["ur5e-4"]
        harness.coordinator.primitive_models_provider = lambda: models
        original, seen = admission.analyze_grounded_recovery_composition, []

        def analyze(**kwargs):
            seen.append(kwargs["primitive_models"])
            return original(**kwargs)

        monkeypatch.setattr(admission, "analyze_grounded_recovery_composition", analyze)
        await harness.register()
        assert seen == [models]
        assert harness.session["primitive_model_descriptors"] == {"ur5e-4": model.descriptor()}
        assert "primitive_model_descriptors" not in harness.context
        # Serialized inputs and grants retain metadata, never executable models.
        json.dumps(harness.registration)
        models["ur5e-4"] = PrimitiveModel(model.id, model.version + 1, model.configuration, model.evaluate)
        result = await harness.check(0)
        assert result["status"] == "inconclusive"
        assert "resource_primitive_model_changed" in result["reason"]
        assert not harness.session["grants"]

    asyncio.run(scenario())


@pytest.mark.parametrize("location", ["context", "composition_inputs"])
def test_executable_models_cannot_arrive_through_serialized_context(location):
    async def scenario():
        harness = _harness()
        target = harness.context if location == "context" else harness.context["composition_inputs"]
        target["primitive_models"] = {"M1": object()}
        result = await harness.coordinator.register(harness.request, product_jid=_PRODUCT)
        assert result["status"] == "inconclusive"
        assert result["reason"] == "executable_primitive_models_require_owner_registration"
        assert not harness.coordinator.sessions
        assert not harness.preparations

    asyncio.run(scenario())


@pytest.mark.parametrize("change,reason", [
    ("mock_disabled", "synthetic_evidence_cannot_authorize_live_execution"),
    ("physical_history", "physical_monitor_history_unavailable"),
    ("unprepared", "resource_preparation_unavailable_or_mismatched"),
    ("mock_executor", "synthetic_evidence_requires_mock_executor"),
    ("stale", "stale_registration_snapshot"),
    ("live_mode", "synthetic_evidence_cannot_authorize_live_execution"),
])
def test_recovery_registration_rejects_untrusted_or_stale_evidence(change, reason) -> None:
    async def scenario():
        harness = _harness()
        if change == "mock_disabled":
            harness.coordinator.allow_mock_execution = False
        elif change == "physical_history":
            harness.context.pop("physical_rule_activation")
        elif change == "live_mode":
            harness.context["execution_mode"] = "live"
        else:
            def prepare(request):
                row = harness.prepare(request)
                if change == "unprepared":
                    row["status"] = "NEEDS_CONTEXT"
                elif change == "mock_executor":
                    row.pop("mock_executor")
                else:
                    harness.context["revision"] += 1
                return row
            harness.coordinator.resource_evidence_provider = prepare
        result = await harness.coordinator.register(harness.request, product_jid=_PRODUCT)
        assert result["status"] == "inconclusive", result
        assert result["reason"] == reason
        assert not harness.coordinator.sessions
    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["program", "scope", "missing_event", "pending"])
def test_complete_recovery_registration_preserves_exact_bindings(change) -> None:
    async def scenario():
        harness = _harness()
        if change == "program":
            harness.request["tasks"][0]["params"]["primitive_steps"][0]["primitive"] = "other_primitive"
        elif change == "scope":
            harness.request["recovery_safety_scope_id"] = "other_scope"
        elif change == "missing_event":
            harness.request["tasks"].pop()
        else:
            harness.request["pending_nominal_task_ids"].pop()
        result = await harness.coordinator.register(harness.request, product_jid=_PRODUCT)
        assert result["status"] == "inconclusive", result
        assert not harness.coordinator.sessions
    asyncio.run(scenario())


def test_recovery_admission_grants_exact_first_event_and_duplicate_is_idempotent() -> None:
    async def scenario():
        harness = _harness()
        await harness.register()
        denied = await harness.check(1)
        assert denied["status"] == "held"
        allowed = await harness.check(0)
        assert allowed["status"] == "allowed", allowed
        grant = allowed["recovery_composition_grant"]
        assert grant["primitive_steps"] == harness.request["tasks"][0]["params"]["primitive_steps"]
        assert len(grant["resolved_primitive_steps"]) == 5
        epoch, commits = harness.coordinator.epoch, len(harness.commits)
        duplicate = await harness.check(0)
        assert duplicate["reason"] == "already_admitted"
        assert duplicate["recovery_composition_grant"] == grant
        assert harness.coordinator.epoch == epoch
        assert len(harness.commits) == commits
        assert harness.coordinator.holds()
    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["sender", "reference", "params"])
def test_recovery_admission_rejects_changed_dispatch(change) -> None:
    async def scenario():
        harness = _harness()
        await harness.register()
        event = harness.event(0)
        sender = event["resource_jid"]
        if change == "sender":
            sender = "other@localhost"
        elif change == "reference":
            event["params"]["recovery_composition_ref"]["program_hash"] = "other"
        else:
            event["params"]["primitive_steps"][0]["primitive"] = "other_primitive"
        result = await harness.coordinator.check(event, sender=sender)
        assert result["status"] == "inconclusive"
        assert result["reason"] == "recovery_dispatch_reference_mismatch"
        assert not harness.session["grants"]
    asyncio.run(scenario())


def test_recovery_admission_rechecks_revision_after_resource_preparation() -> None:
    async def scenario():
        harness = _harness()
        await harness.register()
        def prepare(request):
            harness.context["revision"] += 1
            return harness.prepare(request)
        harness.coordinator.resource_evidence_provider = prepare
        result = await harness.check(0)
        assert result["reason"] == "stale_admission_snapshot"
        assert not harness.session["grants"]
    asyncio.run(scenario())


def test_raw_completion_cannot_advance_predicted_observations() -> None:
    async def scenario():
        harness = _harness()
        await harness.register()
        assert (await harness.check(0))["status"] == "allowed"
        path = deepcopy(harness.session["path"])
        event = harness.event(0)
        event["status"] = "completed"
        harness.coordinator.observe(event, sender=event["resource_jid"])
        assert harness.session["path"] == path
        assert not harness.session["invalid_reason"]
        assert (await harness.check(1))["status"] == "held"
        event.update(status="recovery_acknowledgement", observations={"status": "completed"})
        harness.coordinator.observe(event, sender=_PRODUCT)
        assert harness.session["invalid_reason"] == "completion_not_in_authoritative_history"
        assert (await harness.check(0))["status"] == "inconclusive"
    asyncio.run(scenario())


@pytest.mark.parametrize("field", ["task_id", "function_name", "parameters"])
def test_native_task_binding_must_match_real_dispatch(field) -> None:
    async def scenario():
        harness = _harness()
        task = next(iter(harness.context["task_monitor_context"]["tasks"].values()))["task"]
        task[field] = {} if field == "parameters" else "other"
        result = await harness.coordinator.register(harness.request, product_jid=_PRODUCT)
        assert result["reason"] == "native_task_binding_disagrees_with_dispatch"
        assert not harness.coordinator.sessions
    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["revision", "bindings", "completion_effects", "owner_incarnation"])
def test_changed_authority_between_grants_invalidates_retained_proof(change) -> None:
    async def scenario():
        harness = _harness()
        await harness.register()
        if change == "revision":
            harness.context["revision"] += 1
        elif change == "bindings":
            harness.context["task_monitor_context"]["monitors"][0]["monitor"].resource_bindings = {"other": "binding"}
        elif change == "owner_incarnation":
            harness.context["owner_incarnations"] = [(harness.jids["KMR"], "replacement")]
        else:
            first = next(iter(harness.context["task_monitor_context"]["tasks"].values()))
            first["resource_updates"] = {"KMR": {"resource_state": "invented"}}
        result = await harness.check(0)
        assert result["status"] == "inconclusive"
        assert harness.session["invalid_reason"]
        assert not harness.session["grants"]
    asyncio.run(scenario())


def test_wrong_actual_custody_on_completion_never_commits_native_history() -> None:
    async def scenario():
        harness = _harness()
        await harness.register()
        assert (await harness.check(0))["status"] == "allowed"
        while True:
            edge = harness.session["edges"][harness.session["node"]][0]
            if edge["kind"] == "task_completion":
                break
            harness.feed(edge)
        path = deepcopy(harness.session["path"])
        commits = deepcopy(harness.commits)
        target = harness.session["nodes"][edge["target"]]
        harness.context["time_exact"] = edge["time_exact"]
        harness.context["revision"] += 1
        harness.context["current_snapshot"] = {key: deepcopy(target["state"][key]) for key in ("resources", "parts")}
        harness.context["current_snapshot"]["resources"]["KMR"]["held_part"] = "wrong_part"
        state = target["task_monitor_state"]["monitors"][0]["state"]
        for key in ("resources", "products", "contexts"):
            harness.context["task_monitor_context"][key] = deepcopy(state[key])
        harness.context["observations"].append({"kind": "task_completion", "time_exact": edge["time_exact"],
                                                "task_id": edge["task_id"], "status": "completed"})
        harness.coordinator.observe({"status": "owner_observation"}, sender=_PRODUCT)
        assert harness.session["invalid_reason"] == "authoritative_physical_snapshot_mismatch"
        assert harness.session["path"] == path
        assert harness.commits == commits
        assert not harness.session["completed_tasks"]
    asyncio.run(scenario())


@pytest.mark.parametrize("record", [
    {"status": "failed:tool_timeout"},
    {"status": "recovery_acknowledgement", "observations": {"status": "failed:primitive"}},
])
def test_actual_failure_invalidates_existing_grant(record) -> None:
    async def scenario():
        harness = _harness()
        await harness.register()
        assert (await harness.check(0))["status"] == "allowed"
        event = {**harness.event(0), **record}
        harness.coordinator.observe(event, sender=_PRODUCT)
        assert harness.session["invalid_reason"] == "recovery_execution_failed"
        assert (await harness.check(0))["status"] == "inconclusive"
    asyncio.run(scenario())


def test_delayed_failure_for_other_recovery_does_not_invalidate_active_proof() -> None:
    async def scenario():
        harness = _harness()
        await harness.register()
        assert (await harness.check(0))["status"] == "allowed"
        event = harness.event(0)
        event["task_id"] = "completed_older_recovery_task"
        event["params"]["recovery_composition_ref"]["recovery_id"] = "completed_older_recovery"
        event["status"] = "failed:tool_timeout"
        harness.coordinator.observe(event, sender=event["resource_jid"])
        assert not harness.session["invalid_reason"]
        assert (await harness.check(0))["reason"] == "already_admitted"
    asyncio.run(scenario())


def test_actual_mock_observations_hold_early_start_then_complete_all_22_steps() -> None:
    async def scenario():
        harness = _harness()
        pending = deepcopy(harness.request["pending_nominal_task_ids"])
        await harness.register()
        grants = []
        grants.append(await harness.check(0))
        harness.advance_to(5)
        grants.append(await harness.check(1))
        harness.advance_to(10)
        held = await harness.check(2)
        assert held["status"] == "held", held
        assert held["reason"] == "specific_start_has_no_joint_completion"
        harness.advance_to(12)
        grants.append(await harness.check(2))
        harness.advance_to(16)
        grants.append(await harness.check(3))
        harness.advance_to(26)
        assert all(row["status"] == "allowed" for row in grants), grants
        assert sum(len(row["recovery_composition_grant"]["resolved_primitive_steps"]) for row in grants) == 22
        assert harness.session["complete"]
        assert not harness.coordinator.holds()
        assert harness.request["pending_nominal_task_ids"] == pending
        assert harness.context["composition_inputs"]["grounding_inputs"]["snapshot"]["task_statuses"] == _case()["grounding_inputs"]["snapshot"]["task_statuses"]
        next_request = deepcopy(harness.request)
        next_request["recovery_id"] = "KMR_storage_interruption_after_completion"
        harness.coordinator.context_provider = lambda *_: harness.context
        restarted = await harness.coordinator.register(next_request, product_jid=_PRODUCT)
        assert restarted["reason"] == "existing_physical_history_requires_checkpoint"
    asyncio.run(scenario())


@pytest.mark.parametrize("missing", [None, "acknowledged_effect", "native_ack", "completion_record"])
def test_assembly_effect_replay_commits_only_with_actual_effect_and_native_ack(monkeypatch, missing) -> None:
    from test_offline_recovery_composition import _assembly_completion_case

    async def scenario():
        case, source_native = _assembly_completion_case(12.5)
        monkeypatch.setitem(globals(), "_case", lambda: deepcopy(case))
        harness = _harness()
        native = harness.context["task_monitor_context"]
        identity = case["running_work"][0]["task_id"]
        native["products"] = {"gear_small": deepcopy(source_native["products"]["gear_small"])}
        native["tasks"][identity] = deepcopy(source_native["tasks"][identity])
        harness.context.update(nominal_run_id="assembly_effect_run", nominal_acknowledgements=[])
        await harness.register()
        assert (await harness.check(0))["status"] == "allowed"
        harness.advance_to(5)
        assert (await harness.check(1))["status"] == "allowed"
        harness.advance_to(12)
        assert (await harness.check(2))["status"] == "allowed"
        harness.advance_to(12.49)
        observation = harness.session["edges"][harness.session["node"]][0]
        assert observation["kind"] == "observation" and observation["time_exact"] == "25/2"
        completion = next(row for row in harness.session["edges"][observation["target"]]
                          if row["kind"] == "task_completion" and row["task_id"] == identity)
        target = harness.session["nodes"][completion["target"]]
        observed = deepcopy(observation["observation"])
        snapshot = {key: deepcopy(target["state"][key]) for key in ("resources", "parts")}
        if missing != "acknowledged_effect":
            for value in (observed, snapshot):
                value["parts"]["gear_small"]["product_effect_evidence"][0]["kind"] = "acknowledged"
        record = {"record_id": observation["edge_id"], "kind": "observation",
                  "time_exact": observation["time_exact"], "observation": observed}
        harness.context["observations"].append(record)
        if missing != "completion_record":
            harness.context["observations"].append({
                "record_id": completion["edge_id"], "kind": "task_completion",
                "time_exact": completion["time_exact"], "task_id": identity, "status": "completed",
            })
        if missing != "native_ack":
            harness.context["nominal_acknowledgements"].append({
                "cursor": 1, "run_id": "assembly_effect_run", "admitted": True,
                "task_id": identity, "task": deepcopy(native["tasks"][identity]["task"]),
                "before_states": {}, "after_states": {}, "rule_checks": [],
            })
        harness.context.update(time_exact=completion["time_exact"], current_snapshot=snapshot)
        harness.context["revision"] += 1
        for key in ("resources", "products", "contexts"):
            native[key] = deepcopy(target["task_monitor_state"]["monitors"][0]["state"][key])
        path, commits = deepcopy(harness.session["path"]), deepcopy(harness.commits)
        acknowledged = set() if missing == "native_ack" else {identity}
        if missing in {"acknowledged_effect", "native_ack"}:
            with pytest.raises(ValueError):
                harness.coordinator._observation_matches(observation, record, acknowledged)
        else:
            assert harness.coordinator._observation_matches(observation, record, acknowledged)
        if missing:
            with pytest.raises(ValueError):
                harness.coordinator._synchronize(harness.session, harness.context)
            assert harness.session["path"] == path
            assert harness.commits == commits
        else:
            harness.coordinator._synchronize(harness.session, harness.context)
            assert harness.session["node"] == completion["target"]
            assert harness.session["path"] == [*path, observation["edge_id"], completion["edge_id"]]
            assert len(harness.commits) == len(commits) + 1
            committed = harness.commits[-1]["acknowledged_transitions"]
            assert [row["task_id"] for row in committed] == [identity]
            assert harness.context["current_snapshot"]["parts"]["gear_small"]["product_effect_evidence"][0]["kind"] == "acknowledged"
            assert not harness.session["invalid_reason"]

    asyncio.run(scenario())
