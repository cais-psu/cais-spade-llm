from __future__ import annotations

"""Exchange pinned primitive context through scoped SPADE inboxes."""

import asyncio
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from spade.behaviour import OneShotBehaviour
from spade.message import Message
from spade.template import Template

from cais_spade_llm.agents.shared_information.local_dispatch import send_agent_message

from ..ra.refinement_records import append_record, owned_path, verify_record


class _ScopedInbox(OneShotBehaviour):
    """Consume one conversation without starting PA's production lifecycle."""

    def __init__(self, operation: Callable[[Any], Awaitable[Any]]) -> None:
        super().__init__()
        self.operation = operation
        self.task: asyncio.Task[Any] | None = None

    def start(self) -> None:
        # The adapter owns this task. SPADE's normal _start waits for Agent.start,
        # which would also run ProductAgent's unrelated production kickoff.
        if self.task is None:
            self.task = asyncio.create_task(self.run())

    async def run(self) -> Any:
        return await self.operation(self)


def _template(kind: str, thread: str) -> Template:
    result = Template(thread=thread)
    result.set_metadata("type", kind)
    return result


def _check_message(
    message: Message | None, *, sender: str, recipient: str, thread: str, kind: str,
    request_ref: Mapping[str, str],
) -> dict[str, Any]:
    if message is None:
        raise TimeoutError("The primitive context SPADE reply deadline was reached.")
    if (str(message.sender) != sender or str(message.to) != recipient or message.thread != thread
            or message.get_metadata("type") != kind):
        raise ValueError("Primitive context message has the wrong sender, recipient, or conversation.")
    body = json.loads(message.body or "null")
    expected = {"request_ref"} if kind == "PrimitiveContextRequest" else {"request_ref", "response_ref"}
    if not isinstance(body, dict) or set(body) != expected or body["request_ref"] != request_ref:
        raise ValueError("Primitive context message does not match its pinned request.")
    return body


def _message_record(message: Message) -> dict[str, Any]:
    return {"record_type": "PrimitiveContextMessage", "sender": str(message.sender),
            "recipient": str(message.to), "thread": message.thread,
            "type": message.get_metadata("type"), "body": json.loads(message.body),
            "created_at_ns": time.time_ns()}


@asynccontextmanager
async def _registered(agent: Any, inbox: _ScopedInbox, template: Template) -> AsyncIterator[_ScopedInbox]:
    original_loop = agent.loop
    # Both local_dispatch and Agent.dispatch must enqueue on the inbox's owner loop.
    agent.set_loop(asyncio.get_running_loop())
    agent.add_behaviour(inbox, template)
    inbox.start()
    try:
        yield inbox
    finally:
        if inbox.task is not None:
            if not inbox.task.done():
                inbox.task.cancel()
            await asyncio.gather(inbox.task, return_exceptions=True)
        if agent.has_behaviour(inbox):
            agent.remove_behaviour(inbox)
        agent.set_loop(original_loop)


@asynccontextmanager
async def primitive_context_inbox(
    product_agent: Any, *, root: Path, request_ref: Mapping[str, str], sender: str,
    thread: str, deadline: float, handler: Callable[[], Awaitable[dict[str, str]]],
) -> AsyncIterator[tuple[str, asyncio.Task[Any]]]:
    """Install one PA-owned receiver for one already-authorized context request.

    Args:
        product_agent: Existing shared PA, accessed only by its owned adapter.
        root: Interaction containing the pinned request.
        request_ref: Expected immutable request reference.
        sender: Exact assigned RobotAgent JID.
        thread: Host-issued SPADE conversation identifier.
        deadline: Absolute monotonic deadline for this request.
        handler: Deterministic evidence service returning a pinned response.

    Yields:
        The PA recipient JID and the service task, so handler errors propagate promptly.
    """
    recipient = str(product_agent.jid)
    request = verify_record(root, request_ref)
    if request.get("record_type") != "PrimitiveContextRequest":
        raise ValueError("SPADE context service requires a PrimitiveContextRequest.")

    async def serve(inbox: _ScopedInbox) -> None:
        message = await inbox.receive(timeout=max(0.001, deadline - time.monotonic()))
        _check_message(message, sender=sender, recipient=recipient, thread=thread,
                       kind="PrimitiveContextRequest", request_ref=request_ref)
        if time.monotonic() >= deadline:
            raise TimeoutError("Primitive context request arrived after its deadline.")
        if verify_record(root, request_ref) != request:
            raise ValueError("Primitive context request changed before PA handling.")
        response_ref = await handler()
        if time.monotonic() >= deadline:
            raise TimeoutError("Primitive context response completed after its deadline.")
        message = Message(to=sender, sender=recipient, thread=thread)
        message.set_metadata("type", "PrimitiveContextResponse")
        message.body = json.dumps({"request_ref": dict(request_ref), "response_ref": response_ref})
        await send_agent_message(inbox, message, transport_label="primitive_context_response")

    inbox = _ScopedInbox(serve)
    async with _registered(product_agent, inbox, _template("PrimitiveContextRequest", thread)):
        yield recipient, inbox.task
        if inbox.task is not None and inbox.task.done():
            inbox.task.result()


async def request_context_message(
    robot_agent: Any, *, root: Path, recipient: str, request_ref: Mapping[str, str],
    thread: str, deadline: float,
) -> dict[str, Any]:
    """Send from the exact RA and accept only PA's matching, pinned response.

    Args:
        robot_agent: Selected live RA supplied by its owned adapter.
        root: Authorized interaction root.
        recipient: PA JID with a registered scoped inbox.
        request_ref: Pinned request to send.
        thread: Matching conversation identifier.
        deadline: Absolute monotonic request deadline.

    Returns:
        The verified PA response with its response_ref for coordinator audit.
    """
    sender = str(robot_agent.jid)
    directory = owned_path(root, request_ref["ref"]).parent
    request = verify_record(root, request_ref)

    async def exchange(inbox: _ScopedInbox) -> dict[str, Any]:
        message = Message(to=recipient, sender=sender, thread=thread)
        message.set_metadata("type", "PrimitiveContextRequest")
        message.body = json.dumps({"request_ref": dict(request_ref)})
        await asyncio.to_thread(append_record, root, directory, "request_message.json", _message_record(message))
        await send_agent_message(inbox, message, transport_label="primitive_context_request")
        reply = await inbox.receive(timeout=max(0.001, deadline - time.monotonic()))
        body = _check_message(reply, sender=recipient, recipient=sender, thread=thread,
                              kind="PrimitiveContextResponse", request_ref=request_ref)
        if time.monotonic() >= deadline or verify_record(root, request_ref) != request:
            raise ValueError("Primitive context reply is late or its request changed.")
        if owned_path(root, body["response_ref"]["ref"]) != directory / "response.json":
            raise ValueError("Primitive context response belongs to another request directory.")
        response = verify_record(root, body["response_ref"])
        if response.get("record_type") != "PrimitiveContextResponse" or response.get("request_ref") != request_ref:
            raise ValueError("Primitive context response has different request authority.")
        await asyncio.to_thread(append_record, root, directory, "response_message.json", _message_record(reply))
        return {**response, "response_ref": body["response_ref"]}

    inbox = _ScopedInbox(exchange)
    async with _registered(robot_agent, inbox, _template("PrimitiveContextResponse", thread)):
        try:
            return await asyncio.wait_for(inbox.task, timeout=max(0.001, deadline - time.monotonic()))
        except asyncio.TimeoutError as exc:
            raise TimeoutError("The primitive context SPADE reply deadline was reached.") from exc
