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
from cais_spade_llm.resources.environment_models import matches_requirement


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
        if (
            not context.current_request(request)
            or request.get("part_name") != context.environment_model["part_name"]
            or request.get("desired_property") != context.environment_model["desired_property"]
            or request.get("geometry") != context.geometry.get(request.get("part_name"))
            or time.time() > request["deadline"]
            or runtime.stopped
        ):
            reply["rejections"].append(
                {"resource_id": rid, "reason": "Expired or changed exploration context"}
            )
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
            best_length = context.environment_model.get("best_length", runtime.max_steps)
            if execution_unavailable:
                best_length = min(
                    best_length,
                    context.environment_model.get("model_best_length", runtime.max_steps),
                )
            if (previous is None or cost < previous) and len(request["path"]) <= min(
                best_length, runtime.max_steps
            ):
                resource.visited[key] = cost
                part, desired = request["part_name"], request["desired_property"]
                if matches_requirement(request["products"][part], desired):
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
                        if (
                            task["event_name"] == "move_to_resource"
                            and request["valuation"][rid].get("held_part") != part
                            and request["products"][part]["location"] != "Storage"
                        ):
                            continue
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
                    for neighbor in resource.model["neighbors"]:
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
                runtime.replies.put_nowait(payload)
        except (ValueError, KeyError, TypeError) as exc:
            self.agent.logger.warning("Rejected capability reply: %s", exc)


async def explore(runtime, behaviour, *, timeout: float = 15.0) -> dict:
    """Collect a bounded distributed search without changing acknowledged state."""
    context = runtime.context
    request = context.request(time.time() + timeout)
    if request is None:
        context.environment_model.update(status="completed", closed=True)
        return {"status": "completed", "tasks": []}
    contact = context.contact_resource(request["part_name"])
    if contact not in context.permitted_resources:
        context.environment_model.update(
            status="blocked",
            closed=True,
            rejections=[{"resource_id": contact, "reason": "Current custodian is excluded"}],
        )
        return {"status": "blocked", "tasks": [], "reason": "Current custodian is excluded"}
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
            reply = await asyncio.wait_for(runtime.replies.get(), remaining)
        except asyncio.TimeoutError:
            status = "timeout"
            break
        if time.time() > request["deadline"]:
            status = "timeout"
            break
        if not context.current_request(reply) or reply["branch_id"] in completed:
            continue
        completed.add(reply["branch_id"])
        pending.discard(reply["branch_id"])
        pending.update(branch for branch in reply["children"] if branch not in completed)
        context.merge_reply(reply)
        resolved = [bid for bid in context.environment_model["bids"] if not bid["unresolved"]]
        if resolved:
            context.environment_model["model_best_length"] = min(
                len(bid["path"]) for bid in resolved
            )
            executable = [bid for bid in resolved if not bid["execution_unavailable"]]
            if executable:
                context.environment_model["best_length"] = min(
                    len(bid["path"]) for bid in executable
                )
        if len(completed) + len(pending) > runtime.max_search_states:
            status = "budget_exhausted"
            break
    result = runtime.agent.process_planner.select_environment_path(
        context.environment_model["bids"]
    )
    if runtime.stopped:
        status = "stopped"
    if status:
        result = {"status": status, "tasks": [], "candidate": result}
    context.environment_model.update(
        status=result["status"], selected_path=deepcopy(result.get("tasks", []))
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
        }
    )
    # Close the correlation window even if queued resource work finishes later.
    context.environment_model["closed"] = True
    return result
