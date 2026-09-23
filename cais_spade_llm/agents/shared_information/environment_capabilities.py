"""SPADE environmental-capability requests, neighbor propagation, and bids."""

from __future__ import annotations

import asyncio
import json
import time
from copy import deepcopy
from uuid import uuid4

from spade.behaviour import CyclicBehaviour
from spade.message import Message

from cais_spade_llm.agents.shared_information.local_dispatch import send_agent_message
from cais_spade_llm.product.environment import fingerprint
from cais_spade_llm.resources.environment_models import candidates, feasibility, matches_requirement


def message(to: str, kind: str, payload: dict) -> Message:
    """Create a correlated JSON message on the existing agent transport."""
    result = Message(to=to)
    result.set_metadata("type", kind)
    result.set_metadata("protocol", "environment")
    result.body = json.dumps(payload, allow_nan=False)
    return result


def handoff_matches(model: dict, location: str | None) -> bool:
    """Check shared product endpoints, rather than physical proximity or part names."""
    return any(
        location
        in (
            event["capability_transition"]["source"].get("part_location"),
            event["capability_transition"]["source"].get("resource_location"),
        )
        for event in model["events"]
    )


def _transition(context, request: dict, offer: dict, resource_id: str) -> dict:
    task, part = offer["task"], request["part_name"]
    participants = {
        rid: event
        for rid, model in context.models.items()
        for event in model["events"]
        if event["event_id"] == task["event_id"]
    }
    return {
        "source_id": fingerprint([request["valuation"], request["products"]]),
        "target_id": fingerprint([offer["valuation"], offer["products"]]),
        "source": deepcopy(request["products"][part]),
        "target": deepcopy(offer["products"][part]),
        "task": task,
        "prerequisites": {rid: deepcopy(event["guards"]) for rid, event in participants.items()},
        "predicted_effects": {
            rid: {
                field: value
                for field, value in values.items()
                if request["valuation"][rid][field] != value
            }
            for rid, values in offer["valuation"].items()
            if values != request["valuation"][rid]
        },
        "product_effects": deepcopy(participants[resource_id]["product_effects"]),
        "resource_revision": request["resource_revisions"][resource_id],
        "status": offer["status"],
        "unresolved": deepcopy(offer["reasons"]),
        "executable": offer["executable"],
    }


class CapabilityRequestInbox(CyclicBehaviour):
    """Evaluate one local graph step and forward reachable neighboring searches."""

    async def run(self) -> None:
        msg = await self.receive(timeout=1)
        if msg is None:
            return
        runtime = getattr(self.agent, "environment_runtime", None)
        if runtime is None:
            return
        try:
            request = json.loads(msg.body)
            await self.handle(runtime, str(msg.sender).split("/", 1)[0], request)
        except (ValueError, TypeError, KeyError) as exc:
            self.agent.logger.warning("Rejected capability request: %s", exc)

    async def handle(self, runtime, sender: str, request: dict) -> None:
        """Run the same handler for local SPADE delivery and XMPP delivery."""
        context = runtime.context
        rid = self.agent.agent_name
        resource = context.resources[rid]
        model = context.exploration_models.get(request.get("request_id"), {})
        allowed = {
            runtime.product_jid,
            *(runtime.jids[peer] for peer in resource.model["neighbors"]),
            runtime.jids[rid],
        }
        if sender not in allowed or rid not in context.permitted_resources:
            raise ValueError("Capability request sender or resource is not permitted")
        reply = {
            key: request[key]
            for key in ("run_id", "request_id", "branch_id", "revision", "resource_revisions")
        }
        reply.update(resource_id=rid, children=[], transitions=[], bids=[], rejections=[])
        children = []
        if model:
            model["observed_resources"][rid] = deepcopy(request["resource_revisions"][rid])
        if request.get("scope") == "intake":
            context.request_dependencies(request, sorted({
                rid, *(peer for event in resource.model["events"] for peer in event["participants"])
            }))
        if (
            not context.current_request(request)
            or request["resource_revisions"][rid] != context.revisions([rid])[rid]
            or request.get("part_name") != model.get("part_name")
            or request.get("desired_property") != model.get("desired_property")
            or request.get("resource_goal") != model.get("resource_goal")
            or request.get("candidate") != model.get("candidate")
            or request.get("geometry") != context.geometry.get(request.get("part_name"))
            or time.time() > request["deadline"]
            or runtime.stopped
        ):
            reply["rejections"].append(
                {"resource_id": rid, "reason": "Expired or changed exploration context"}
            )
        elif request.get("scope") == "intake":
            reply["intake_offers"] = []
            if request.get("intake_parts") != model.get("intake_parts"):
                raise ValueError("Intake request differs from its correlated part queue")
            for part in request["intake_parts"]:
                desired = next(step for step in context.requirements[part]
                               if not matches_requirement(context.part_tracker[part], step))
                for task in candidates(resource.model, request["valuation"], part, desired, context.requirements[part]):
                    if (task["parameters"].get("process") not in resource.model["process_capabilities"]
                            or not context.allows_task(task)):
                        continue
                    status, reasons = feasibility(resource.model, task, context.geometry[part])
                    if status != "FEASIBLE":
                        reply["rejections"].append({"resource_id": rid, "part_name": part,
                                                    "reason": "; ".join(reasons)})
                        continue
                    event = next(event for event in resource.model["events"]
                                 if event["event_id"] == task["event_id"])
                    reply["intake_offers"].append({
                        "resource_id": rid, "part_name": part, "task": task,
                        "prerequisites": deepcopy(event["guards"]),
                        "effects": deepcopy(event["product_effects"]),
                        "resource_revision": deepcopy(request["resource_revisions"][rid]),
                        "executable": task["event_name"] in resource.executors,
                        "available": resource.valuation.get("part_name") is None
                                     and f"resource:{rid}" not in context.reservations
                                     and f"access:{rid}" not in context.reservations,
                    })
        else:
            state_key = fingerprint([request["valuation"], request["products"]])
            execution_unavailable = any(
                task["event_name"] not in context.resources[task["resource_id"]].executors
                for task in request["path"]
            )
            cost = (
                bool(request["unresolved"]),
                execution_unavailable,
                len(request["path"]),
                tuple((task["resource_id"], task["event_id"]) for task in request["path"]),
            )
            key = (request["request_id"], state_key)
            previous = resource.visited.get(key)
            best_length = model.get("best_length", runtime.max_steps)
            if execution_unavailable:
                best_length = min(
                    best_length,
                    model.get("model_best_length", runtime.max_steps),
                )
            if (previous is None or cost < previous) and len(request["path"]) <= min(
                best_length, runtime.max_steps
            ):
                resource.visited[key] = cost
                part, desired = request["part_name"], request["desired_property"]
                goal = request.get("resource_goal")
                candidate = request.get("candidate")
                achieved = (
                    all(request["valuation"][goal["resource_id"]].get(key) == value
                        for key, value in goal["values"].items())
                    if goal else matches_requirement(request["products"][part], desired)
                )
                if achieved and candidate is None:
                    reply["bids"].append(
                        {"path": request["path"], "unresolved": request["unresolved"]}
                    )
                else:
                    for offer in resource.alternatives(
                        context, request["valuation"], request["products"], part, desired
                    ):
                        if offer["status"] == "INFEASIBLE":
                            reply["rejections"].append(
                                {
                                    "resource_id": rid,
                                    "event_name": offer["task"]["event_name"],
                                    "reason": "; ".join(offer["reasons"]),
                                }
                            )
                            continue
                        task = offer["task"]
                        if candidate is not None:
                            context.request_dependencies(request, context._task_participants(task))
                            if all(task.get(key) == candidate.get(key)
                                   for key in ("resource_id", "event_id", "event_name", "parameters")):
                                reply["bids"].append({"path": [task], "unresolved": [
                                    {"resource_id": rid, "reason": reason}
                                    for reason in offer["reasons"]
                                ]})
                                reply["transitions"].append(_transition(context, request, offer, rid))
                            continue
                        if (
                            goal is None
                            and
                            task["event_name"] == "move_to_resource"
                            and request["valuation"][rid].get("held_part") != part
                            and request["products"][part]["location"] != "Storage"
                        ):
                            continue
                        context.request_dependencies(request, context._task_participants(task))
                        unresolved = [
                            *request["unresolved"],
                            *(
                                {"resource_id": rid, "reason": reason}
                                for reason in offer["reasons"]
                            ),
                        ]
                        following = {
                            **request,
                            "branch_id": uuid4().hex,
                            "valuation": offer["valuation"],
                            "products": offer["products"],
                            "path": [*request["path"], task],
                            "unresolved": unresolved,
                        }
                        children.append((rid, following))
                        reply["transitions"].append(_transition(context, request, offer, rid))
                    location = request["products"][part]["location"]
                    for neighbor in (() if candidate is not None or goal else resource.model["neighbors"]):
                        if neighbor in context.permitted_resources and handoff_matches(
                            context.models[neighbor], location
                        ):
                            children.append((neighbor, {**request, "branch_id": uuid4().hex}))
                    if not children:
                        reply["rejections"].append(
                            {
                                "resource_id": rid,
                                "reason": "No enabled transition or compatible neighboring handoff",
                            }
                        )
        reply["children"] = [child["branch_id"] for _, child in children]
        # Announce child searches before dispatch. PA tracks branch identifiers,
        # so replies arriving on different agent loops may safely interleave.
        await send_agent_message(self, message(runtime.product_jid, "capability_reply", reply))
        for peer, child in children:
            await send_agent_message(self, message(runtime.jids[peer], "capability_request", child))


class CapabilityReplyInbox(CyclicBehaviour):
    """Authenticate ResourceAgent replies and wake the requesting ProductAgent."""

    async def run(self) -> None:
        msg = await self.receive(timeout=1)
        runtime = getattr(self.agent, "environment_runtime", None)
        if msg is None or runtime is None:
            return
        try:
            payload = json.loads(msg.body)
            sender = str(msg.sender).split("/", 1)[0]
            if runtime.jids.get(payload.get("resource_id")) != sender:
                raise ValueError("Capability reply sender differs from its resource")
            if runtime.context.current_request(payload):
                queue = getattr(runtime, "request_queues", {}).get(payload["request_id"])
                if queue is not None:
                    queue.put_nowait(payload)
                else:
                    runtime.replies.put_nowait(payload)
        except (ValueError, KeyError, TypeError) as exc:
            self.agent.logger.warning("Rejected capability reply: %s", exc)


async def match_intake(runtime, behaviour, parts: list[str]) -> dict:
    """Ask process owners for local matches, without exploring queued part routes."""
    context = runtime.context
    part = parts[0]
    desired = next(step for step in context.requirements[part]
                   if not matches_requirement(context.part_tracker[part], step))
    request = context.request(time.time() + 15, part=part, desired=desired)
    model = context.exploration_models[request["request_id"]]
    model["intake_parts"] = list(parts)
    request.update(scope="intake", intake_parts=list(parts))
    peers = [rid for rid, actor in context.resources.items()
             if actor.model["process_capabilities"] and rid in context.permitted_resources]
    context.request_dependencies(request, peers)
    if not hasattr(runtime, "request_queues"):
        runtime.request_queues = {}
    queue = runtime.request_queues[request["request_id"]] = asyncio.Queue()
    offers = []
    started = time.monotonic()
    context.negotiations.append({
        "kind": "capability_request", "scope": "intake", "request_id": request["request_id"],
        "parts": list(parts), "resource_ids": peers, "timestamp": time.time(),
    })
    try:
        pending = set(peers)
        for rid in peers:
            await send_agent_message(behaviour, message(runtime.jids[rid], "capability_request", request))
        status = "matched"
        while pending and not runtime.stopped:
            if not context.current_request(request):
                status = "stale"
                break
            remaining = request["deadline"] - time.time()
            if remaining <= 0:
                status = "timeout"
                break
            try:
                reply = await asyncio.wait_for(queue.get(), min(.1, remaining))
            except asyncio.TimeoutError:
                continue
            rid = reply["resource_id"]
            if rid not in pending:
                continue
            pending.remove(rid)
            offers.extend(reply.get("intake_offers", []))
            context.negotiations.append({
                "kind": "capability_reply", "scope": "intake", "request_id": request["request_id"],
                "resource_id": rid, "offers": deepcopy(reply.get("intake_offers", [])),
                "rejections": deepcopy(reply.get("rejections", [])), "timestamp": time.time(),
            })
        return {"status": status, "offers": offers if status == "matched" else [],
                "dependencies": deepcopy(model["dependencies"]), "request_id": request["request_id"]}
    finally:
        model.update(closed=True, status="intake_matched", offers=deepcopy(offers))
        runtime.request_queues.pop(request["request_id"], None)
        runtime.planning_wall_time_sec += time.monotonic() - started


async def explore(
    runtime, behaviour, *, timeout: float | None = None, part: str | None = None,
    desired: dict | None = None, resource_goal: dict | None = None, candidate: dict | None = None,
) -> dict:
    """Collect a bounded distributed search without changing acknowledged state."""
    context = runtime.context
    if timeout is None:
        outstanding = context.outstanding()
        if outstanding is None:
            timeout = 15.0
        else:
            timeout_part, _ = outstanding
            timeout = min(
                120.0,
                max(15.0, 5.0 * len(context.resources) + 10.0 * len(context.requirements[timeout_part])),
            )
    request = context.request(
        time.time() + timeout, part=part, desired=desired,
        resource_goal=resource_goal, candidate=candidate,
    )
    if request is None:
        context.environment_model.update(status="completed", closed=True)
        return {"status": "completed", "tasks": []}
    model = context.exploration_models[request["request_id"]]
    started = time.monotonic()
    if not hasattr(runtime, "request_queues"):
        runtime.request_queues = {}
    queue = runtime.request_queues[request["request_id"]] = asyncio.Queue()
    try:
        contact = (candidate["resource_id"] if candidate else
                   resource_goal["resource_id"] if resource_goal else
                   context.contact_resource(request["part_name"]))
        if contact not in context.permitted_resources:
            model.update(
                status="blocked",
                closed=True,
                rejections=[{"resource_id": contact, "reason": "Current custodian is excluded"}],
            )
            return {"status": "blocked", "tasks": [], "reason": "Current custodian is excluded"}
        context.negotiations.append({
            "kind": "capability_request", "request_id": request["request_id"],
            "part_name": request["part_name"], "desired_property": deepcopy(request["desired_property"]),
            "resource_goal": deepcopy(resource_goal), "candidate": deepcopy(candidate),
            "resource_revisions": deepcopy(request["resource_revisions"]),
        })
        pending, completed = {request["branch_id"]}, set()
        await send_agent_message(
            behaviour, message(runtime.jids[contact], "capability_request", request)
        )
        status = None
        while pending and not runtime.stopped:
            if not context.current_request(request):
                status = "stale"
                break
            remaining = request["deadline"] - time.time()
            if remaining <= 0:
                status = "timeout"
                break
            try:
                reply = await asyncio.wait_for(queue.get(), min(.1, remaining))
            except asyncio.TimeoutError:
                # Acknowledgements can invalidate the request while its replies
                # are in flight. Recheck revisions without waiting for its deadline.
                continue
            if time.time() > request["deadline"]:
                status = "timeout"
                break
            if not context.current_request(reply) or reply["branch_id"] in completed:
                continue
            completed.add(reply["branch_id"])
            pending.discard(reply["branch_id"])
            pending.update(branch for branch in reply["children"] if branch not in completed)
            context.merge_reply(reply)
            context.negotiations.append({
                "kind": "capability_reply", "request_id": request["request_id"],
                "resource_id": reply["resource_id"],
                "transitions": deepcopy(reply.get("transitions", [])),
                "rejections": deepcopy(reply.get("rejections", [])),
                "bid_count": len(reply.get("bids", [])),
            })
            resolved = [bid for bid in model["bids"] if not bid["unresolved"]]
            if resolved:
                model["model_best_length"] = min(
                    len(bid["path"]) for bid in resolved
                )
                executable = [bid for bid in resolved if not bid["execution_unavailable"]]
                if executable:
                    model["best_length"] = min(
                        len(bid["path"]) for bid in executable
                    )
                    break
            if len(completed) + len(pending) > runtime.max_search_states:
                status = "budget_exhausted"
                break
        result = runtime.agent.process_planner.select_environment_path(
            model["bids"]
        )
        if result.get("tasks"):
            result["tasks"][0]["offer_request_id"] = request["request_id"]
            result["tasks"][0]["offer_resource_revisions"] = deepcopy(request["resource_revisions"])
            if resource_goal is None:
                result["tasks"][0]["offer_product_state"] = deepcopy(model["part_state"])
        dependencies = model["dependencies"] if result.get("tasks") else model["observed_resources"]
        result["dependencies"] = deepcopy(dependencies)
        if not result.get("tasks"):
            current = context.revisions(dependencies)
            if any(current[rid] != revision for rid, revision in dependencies.items()):
                status = "stale"
        if runtime.stopped:
            status = "stopped"
        if status:
            result = {"status": status, "tasks": [], "candidate": result}
        model.update(
            status=result["status"], selected_path=[
                {key: deepcopy(value) for key, value in task.items()
                 if key not in {"offer_request_id", "offer_resource_revisions", "offer_product_state"}}
                for task in result.get("tasks", [])
            ],
        )
        context.explorations.append(
            {
                "request": {
                    key: value
                    for key, value in request.items()
                    if key not in {"valuation", "products", "path"}
                },
                "result": deepcopy(result),
                "visited": len(completed),
                "wall_time_sec": time.monotonic() - started,
            }
        )
        # Close the correlation window even if queued resource work finishes later.
        context.environment_model = model
        context.negotiations.append({
            "kind": "selection", "request_id": request["request_id"],
            "result": deepcopy(result), "wall_time_sec": time.monotonic() - started,
        })
        return result
    finally:
        model["closed"] = True
        runtime.request_queues.pop(request["request_id"], None)
        for resource in context.resources.values():
            for key in tuple(resource.visited):
                if key[0] == request["request_id"]:
                    resource.visited.pop(key, None)
