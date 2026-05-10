"""Local-first SPADE message delivery for agents in the same process."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

from spade.message import Message

_log = logging.getLogger(__name__)


def _trace_label_key(label: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(label or "").strip().lower()).strip("_")


def _message_type(msg: Message) -> str:
    try:
        return str((msg.metadata or {}).get("type") or "").strip()
    except Exception:
        return ""


def _message_to(msg: Message) -> str:
    return str(getattr(msg, "to", "") or "").strip()


def _agent_jid(agent: Any) -> str:
    return str(getattr(agent, "jid", "") or "").strip()


def _set_sender_if_missing(agent: Any, msg: Message) -> None:
    try:
        empty_sender = msg.empty_sender()
    except Exception:
        empty_sender = not bool(str(getattr(msg, "sender", "") or "").strip())
    if empty_sender:
        msg.sender = _agent_jid(agent)


def _append_trace(agent: Any, msg: Message, *, category: str) -> None:
    traces = getattr(agent, "traces", None)
    append = getattr(traces, "append", None)
    if not callable(append):
        return
    try:
        append(msg, category=category)
    except Exception:
        _log.debug("[Transport] failed to append SPADE trace", exc_info=True)


def _stamp_trace_transport(msg: Message, *, label: str, transport: str) -> None:
    key = _trace_label_key(label)
    if not key:
        return

    try:
        payload = json.loads(msg.body or "{}")
    except Exception:
        return
    if not isinstance(payload, dict):
        return

    trace = payload.get("trace")
    if not isinstance(trace, dict):
        return

    trace[f"{key}_transport"] = transport
    payload["trace"] = trace
    msg.body = json.dumps(payload)


def _resolve_local_target(agent: Any, msg: Message) -> Any | None:
    container = getattr(agent, "container", None)
    if container is None:
        return None

    to_jid = _message_to(msg)
    if not to_jid:
        return None

    has_agent = getattr(container, "has_agent", None)
    get_agent = getattr(container, "get_agent", None)
    if not callable(has_agent) or not callable(get_agent):
        return None

    try:
        if not has_agent(to_jid):
            return None
        return get_agent(to_jid)
    except Exception:
        _log.debug(
            "[Transport] failed to resolve local target to=%s type=%s",
            to_jid,
            _message_type(msg) or "-",
            exc_info=True,
        )
        return None


def _dispatch_on_recipient_loop(target: Any, msg: Message) -> None:
    target_loop = getattr(target, "loop", None)
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    if (
        target_loop is not None
        and target_loop is not current_loop
        and callable(getattr(target_loop, "is_running", None))
        and target_loop.is_running()
    ):
        target_loop.call_soon_threadsafe(target.dispatch, msg)
        return

    target.dispatch(msg)


def _log_transport(
    agent: Any,
    *,
    label: str,
    msg: Message,
    transport: str,
    elapsed_ms: float,
) -> None:
    logger = getattr(agent, "logger", None)
    if logger is None:
        return

    label_text = _trace_label_key(label) or "message"
    try:
        logger.debug(
            "[Transport] %s type=%s to=%s transport=%s dispatch_ms=%.1f",
            label_text,
            _message_type(msg) or "-",
            _message_to(msg) or "-",
            transport,
            elapsed_ms,
        )
    except Exception:
        pass


def dispatch_local_agent_message(
    agent: Any,
    msg: Message,
    *,
    trace_category: str = "agent",
    transport_label: str = "",
) -> bool:
    """Dispatch *msg* directly when the target agent lives in the same SPADE container."""
    _set_sender_if_missing(agent, msg)
    target = _resolve_local_target(agent, msg)
    if target is None:
        return False

    start = time.perf_counter()
    _stamp_trace_transport(msg, label=transport_label, transport="local")
    _dispatch_on_recipient_loop(target, msg)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    msg.sent = True
    _append_trace(agent, msg, category=trace_category)
    _log_transport(
        agent,
        label=transport_label,
        msg=msg,
        transport="local",
        elapsed_ms=elapsed_ms,
    )
    return True


async def send_agent_message(
    behaviour: Any,
    msg: Message,
    *,
    trace_category: str = "agent",
    transport_label: str = "",
) -> str:
    """Send from a SPADE Behaviour, preferring in-process delivery for local agents."""
    agent = getattr(behaviour, "agent", None)
    if agent is not None and dispatch_local_agent_message(
        agent,
        msg,
        trace_category=trace_category,
        transport_label=transport_label,
    ):
        return "local"

    _stamp_trace_transport(msg, label=transport_label, transport="xmpp")
    start = time.perf_counter()
    await behaviour.send(msg)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if agent is not None:
        _log_transport(
            agent,
            label=transport_label,
            msg=msg,
            transport="xmpp",
            elapsed_ms=elapsed_ms,
        )
    return "xmpp"


def send_agent_message_sync(
    agent: Any,
    msg: Message,
    *,
    trace_category: str = "agent",
    transport_label: str = "",
) -> str:
    """Send from agent helper code outside a Behaviour, keeping XMPP as fallback."""
    if dispatch_local_agent_message(
        agent,
        msg,
        trace_category=trace_category,
        transport_label=transport_label,
    ):
        return "local"

    _set_sender_if_missing(agent, msg)
    client = getattr(agent, "client", None)
    if client is None:
        raise RuntimeError("agent client is not connected")

    start = time.perf_counter()
    _stamp_trace_transport(msg, label=transport_label, transport="xmpp")
    slixmpp_msg = msg.prepare(client)
    slixmpp_msg.send()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    msg.sent = True
    _append_trace(agent, msg, category=trace_category)
    _log_transport(
        agent,
        label=transport_label,
        msg=msg,
        transport="xmpp",
        elapsed_ms=elapsed_ms,
    )
    return "xmpp"
