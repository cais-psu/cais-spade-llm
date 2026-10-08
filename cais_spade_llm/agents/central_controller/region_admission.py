from __future__ import annotations

"""Conservative physical relevance and shared region admission claims."""

from copy import deepcopy
from fractions import Fraction
from threading import RLock
from typing import Any


def event_region_relevance(prepared: dict, resource_id: str,
                           interval: tuple[Fraction, Fraction]) -> dict:
    """Collect every possibly affected region over one complete event execution.

    Args:
        prepared: Validated full-scene physical trace, including swept-motion cells.
        resource_id: Exact configured resource identifier.
        interval: Closed execution interval, including initial and final custody.

    Returns:
        Possible region interactions and unresolved regions. These describe the
        whole execution and must never be used as instantaneous AP valuations.
    """
    regions = sorted(prepared["frozen"]["geometry"]["regions"])
    relevant, uncertain = set(), set()
    evidence = []
    covered = False
    for observation in prepared["observations"]:
        bounds = observation.get("continuous_interval")
        start, end = (tuple(Fraction(str(value)) for value in bounds) if bounds
                      else (Fraction(observation["time_exact"]),) * 2)
        if end < interval[0] or start > interval[1]:
            continue
        covered = True
        cells = observation.get("continuous_cells") or [observation]
        for cell in cells:
            cell_bounds = cell.get("interval", [str(start), str(end)])
            if (Fraction(str(cell_bounds[1])) < interval[0]
                    or Fraction(str(cell_bounds[0])) > interval[1]):
                continue
            possibilities = cell.get("occupancy_possibilities")
            for region in regions:
                if possibilities is not None:
                    values = possibilities.get(region, {}).get(resource_id)
                else:
                    value = observation.get("region_occupancy", {}).get(region, {}).get(resource_id)
                    values = [value] if type(value) is bool else None
                if (not isinstance(values, list) or not values
                        or any(type(value) is not bool for value in values)):
                    values = [False, True]
                if True in values:
                    relevant.add(region)
                    evidence.append({"region": region, "interval": list(cell_bounds),
                                     "possible_values": sorted(set(values))})
                if len(set(values)) > 1:
                    uncertain.add(region)
    if not covered:
        relevant.update(regions)
        uncertain.update(regions)
    return {"resource_id": resource_id,
            "interval": [str(value) for value in interval],
            "affected_regions": sorted(relevant), "unverified_regions": sorted(uncertain),
            "evidence": evidence,
            "semantics": "whole_program_possible_interaction_not_instantaneous_AP"}


def selected_region_mutexes(rules: list[dict]) -> list[dict]:
    """Resolve configured always-not-both rules to exact resource-region pairs.

    Non-mutex requirements impose no additional exclusion. AP identifiers are
    resolved through grounded predicate semantics rather than their spelling.
    """
    from ltlf2dfa.parser.ltlf import LTLfParser

    from cais_spade_llm.agents.central_controller.offline_safety_grounding import _ap_binding
    from cais_spade_llm.agents.central_controller.ppr_ap import physical_ap_kind

    result = []
    for rule in rules:
        expression = LTLfParser()(rule["formula"])
        if type(expression).__name__ != "LTLfAlways":
            continue
        body = expression.f
        if (type(body).__name__ != "LTLfNot"
                or type(body.f).__name__ != "LTLfAnd"
                or len(body.f.formulas) != 2
                or any(type(item).__name__ != "LTLfAtomic" for item in body.f.formulas)):
            continue
        labels = {str(item) for item in body.f.formulas}
        aps = [ap for ap in rule["aps"] if ap["label"] in labels]
        if len(aps) != 2 or any(physical_ap_kind(ap) != "resource_region" for ap in aps):
            continue
        bindings = [_ap_binding(rule, ap) for ap in aps]
        if bindings[0]["region"] != bindings[1]["region"]:
            continue
        resources = sorted({binding["resource"] for binding in bindings})
        if len(resources) != 2:
            continue
        result.append({"rule_id": rule["rule_id"], "region": bindings[0]["region"],
                       "resources": resources})
    return sorted(result, key=lambda row: (row["region"], row["resources"], row["rule_id"]))


class RegionReservationLedger:
    """Atomically retain complete-program claims across nominal and recovery work.

    Completion and failure only retire the command. A claim is released after a
    later authoritative observation proves its owner stopped and the region clear.
    Pending and running programs retain claims even while temporarily outside.
    """

    def __init__(self, *, lock=None) -> None:
        """Create a ledger using the shared runtime admission lock."""
        self.lock = lock or RLock()
        self.revision = 0
        self.claims: dict[str, dict] = {}
        self.observation_revision = -1
        self.observation_time: Fraction | None = None
        self.observation: dict[str, Any] | None = None

    def snapshot(self) -> dict:
        """Return detached pending, active and retained claims for cache identity."""
        with self.lock:
            return {"revision": self.revision, "claims": deepcopy(self.claims),
                    "observation_revision": self.observation_revision}

    def observe(self, observation: dict) -> None:
        """Accept a monotonic owner observation and release observed clear claims.

        Args:
            observation: Exact owner revision, time_exact, complete region_occupancy
                and stationary_resources. Unknown occupancy is retained as None.
        """
        with self.lock:
            revision = observation.get("revision")
            occupancy = observation.get("region_occupancy")
            stationary = observation.get("stationary_resources")
            if (type(revision) is not int or revision < 0 or not isinstance(occupancy, dict)
                    or not isinstance(stationary, list)
                    or any(not isinstance(value, str) for value in stationary)
                    or any(not isinstance(region, str) or not isinstance(values, dict)
                           for region, values in occupancy.items())):
                raise ValueError("authoritative_region_observation_unavailable")
            time = Fraction(str(observation["time_exact"]))
            if revision < self.observation_revision or (
                    self.observation_time is not None and time < self.observation_time):
                raise ValueError("stale_region_observation")
            if revision == self.observation_revision:
                if observation != self.observation:
                    raise ValueError("region_observation_revision_reused")
                return
            self.observation = deepcopy(observation)
            self.observation_revision, self.observation_time = revision, time
            for token, claim in list(self.claims.items()):
                if (claim["status"] not in {"completed", "failed"}
                        or revision <= claim["retired_at_revision"]
                        or claim["resource_id"] not in stationary):
                    continue
                remaining = [region for region in claim["regions"]
                             if occupancy.get(region, {}).get(claim["resource_id"]) is not False]
                if remaining:
                    claim["regions"] = remaining
                else:
                    del self.claims[token]
            self.revision += 1

    def reserve(self, *, token: str, task_id: str, resource_id: str, regions: list[str],
                expected_revision: int, observation_revision: int,
                conflicting_resources: dict[str, list[str]] | None = None) -> dict:
        """Reserve every affected region atomically before any command can dispatch.

        Args:
            token: Unique immutable identity shared with the exact executor.
            task_id: Exact task identifier.
            resource_id: Resource authorized to execute the complete program.
            regions: Complete set of selected mutex regions possibly affected.
            expected_revision: Ledger snapshot used for this admission decision.
            observation_revision: Fresh owner observation used by the caller.
            conflicting_resources: Exact resource peers from selected mutex rules.

        Returns:
            The detached claim. No partial claim is stored on a conflict.
        """
        with self.lock:
            regions = sorted(set(regions))
            if not token or not task_id or not resource_id:
                raise ValueError("exact_region_claim_identity_required")
            peers = ({region: sorted(set(conflicting_resources[region])) for region in regions}
                     if conflicting_resources is not None else None)
            existing = self.claims.get(token)
            identity = {"task_id": task_id, "resource_id": resource_id, "regions": regions,
                        "conflicting_resources": peers}
            if existing is not None:
                if any(existing[key] != value for key, value in identity.items()):
                    raise ValueError("region_claim_identity_changed")
                return deepcopy(existing)
            if expected_revision != self.revision:
                raise ValueError("stale_region_reservation_snapshot")
            if self.observation is None or observation_revision != self.observation_revision:
                raise ValueError("stale_region_observation")
            occupancy = self.observation["region_occupancy"]
            for region in regions:
                current = occupancy.get(region)
                needed = set(peers[region]) | {resource_id} if peers is not None else set(current or {})
                if (not isinstance(current, dict) or not needed or not needed <= set(current)
                        or any(type(current[owner]) is not bool for owner in needed)):
                    raise ValueError("region_clearance_unverified")
                if any(current[owner] for owner in needed if owner != resource_id):
                    raise ValueError("region_occupied_by_another_resource")
                if any(
                    region in claim["regions"] and claim["resource_id"] != resource_id
                    and (claim["resource_id"] in needed
                         or claim["conflicting_resources"] is None
                         or resource_id in claim["conflicting_resources"].get(region, []))
                    for claim in self.claims.values()
                ):
                    raise ValueError("region_reserved_by_another_resource")
            claim = {**identity, "token": token, "status": "pending",
                     "observed_at_revision": observation_revision, "retired_at_revision": None}
            self.claims[token] = claim
            self.revision += 1
            return deepcopy(claim)

    def activate(self, token: str) -> None:
        """Record dispatch without releasing any complete-program claim."""
        with self.lock:
            claim = self.claims[token]
            if claim["status"] not in {"pending", "active"}:
                raise ValueError("retired_region_claim_cannot_dispatch")
            if claim["status"] != "active":
                claim["status"] = "active"
                self.revision += 1

    def finish(self, token: str, *, success: bool) -> None:
        """Retain claims after success or failure until new stopped-clear evidence."""
        with self.lock:
            claim = self.claims.get(token)
            if claim is None:
                return
            status = "completed" if success else "failed"
            if claim["status"] in {"completed", "failed"}:
                return
            claim.update(status=status, retired_at_revision=self.observation_revision)
            self.revision += 1


def reserve_prepared_program(*, ledger: RegionReservationLedger, prepared: dict,
                             resource_id: str, task_id: str, token: str,
                             observation: dict, expected_revision: int | None = None) -> dict:
    """Reserve selected mutex regions from a complete owner-certified program.

    Args:
        ledger: The shared nominal/recovery admission ledger.
        prepared: Fully grounded trace returned by physical trace preparation.
        resource_id: Exact configured executor identifier.
        task_id: Exact runtime task identity.
        token: Immutable identity of this prepared program and execution.
        observation: Fresh trusted whole-scene observed clearance and stopped state.
        expected_revision: Optional earlier claim revision for stale-check rejection.

    Returns:
        A complete-program claim and its relevance evidence. This grants exclusion
        only; the caller must also validate every selected temporal requirement,
        native DES enablement, and exact command authorization before dispatch.
    """
    interval = tuple(Fraction(str(value)) for value in prepared["frozen"]["horizon"])
    relevance = event_region_relevance(prepared, resource_id, interval)
    mutexes = selected_region_mutexes(prepared["rules"])
    population = set(prepared["models"])
    if resource_id not in population:
        raise ValueError("region_executor_outside_scene")
    if any(set(values) != population for values in observation["region_occupancy"].values()):
        raise ValueError("region_observation_scene_incomplete")
    peers = {}
    for mutex in mutexes:
        if resource_id in mutex["resources"] and mutex["region"] in relevance["affected_regions"]:
            peers.setdefault(mutex["region"], set()).update(mutex["resources"])
    with ledger.lock:
        if expected_revision is not None and ledger.revision != expected_revision:
            raise ValueError("stale_region_reservation_snapshot")
        ledger.observe(observation)
        snapshot = ledger.snapshot()
        claim = ledger.reserve(
            token=token, task_id=task_id, resource_id=resource_id, regions=sorted(peers),
            expected_revision=snapshot["revision"],
            observation_revision=snapshot["observation_revision"],
            conflicting_resources={region: sorted(resources) for region, resources in peers.items()},
        )
        return {"claim": claim, "relevance": relevance, "mutexes": mutexes}
