"""Immutable application-level provider requests and unparsed responses."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

_REQUEST_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar(
    "llm_request_context", default=None
)
_DEFAULT_DIRECTORY = Path(__file__).resolve().parents[2] / "monitor/debug/llm_requests"


@contextmanager
def capture_requests(
    *, directory: str | Path | None, run_id: str, stage: str, turn: int
) -> Iterator[None]:
    """Bind request records to a recovery run without changing model inputs."""
    token = _REQUEST_CONTEXT.set(
        {
            "directory": str(directory) if directory else str(_DEFAULT_DIRECTORY),
            "run_id": run_id,
            "stage": stage,
            "turn": turn,
        }
    )
    try:
        yield
    finally:
        _REQUEST_CONTEXT.reset(token)


async def recorded_structured_call(
    call: Callable[..., Any],
    *,
    directory: str | Path | None,
    run_id: str,
    stage: str,
    turn: int,
    **kwargs: Any,
) -> Any:
    """Call the shared model adapter within an evidence-only capture context."""
    with capture_requests(directory=directory, run_id=run_id, stage=stage, turn=turn):
        return await call(**kwargs)


def recovery_request_directory(prepared_request: dict[str, Any], stage: str) -> Path:
    """Locate a recovery stage's records using its existing artifact directory."""
    debug = prepared_request.get("recovery_debug") or {}
    root = debug.get("per_turn_debug_dir") or debug.get("artifact_directory")
    base = Path(root) if root else _DEFAULT_DIRECTORY.parent
    folder = {
        "outline": "recovery_outline",
        "grounding": "recovery_outline",
        "primitive_generation": "recovery_primitves",
        "safety": "recovery_safety",
    }.get(stage, stage)
    return base / folder


def _response_payload(response: Any) -> Any:
    if callable(getattr(response, "model_dump", None)):
        return response.model_dump(mode="json")
    if isinstance(response, dict):
        return {key: _response_payload(value) for key, value in response.items()}
    if isinstance(response, (list, tuple)):
        return [_response_payload(value) for value in response]
    if hasattr(response, "__dict__"):
        return _response_payload(vars(response))
    return response


class RequestCaptureError(RuntimeError):
    """Stop a call when its evidence cannot be persisted without retrying it."""


class RequestRecords:
    """Write each request attempt and outcome once, at the provider boundary."""

    def __init__(self, request_summary: dict[str, Any]) -> None:
        context = dict(_REQUEST_CONTEXT.get() or {})
        self.call_id = uuid4().hex
        self.directory = (
            Path(context.pop("directory", _DEFAULT_DIRECTORY)) / "requests" / self.call_id
        )
        self.directory.mkdir(parents=True, exist_ok=False)
        self.identity = {
            "run_id": context.get("run_id") or self.call_id,
            "stage": context.get("stage") or "structured",
            "turn": context.get("turn", 0),
            "call_id": self.call_id,
        }
        self.summary = request_summary
        self.summary.update(capture_status="not captured", request_record_paths=[])

    def write(self, name: str, payload: dict[str, Any]) -> str:
        """Persist an immutable event; never overwrite an earlier attempt."""
        path = self.directory / f"{name}.json"
        record = {
            **self.identity,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            **deepcopy(payload),
        }
        try:
            serialized = json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False)
            with path.open("x", encoding="utf-8") as stream:
                stream.write(serialized)
        except (OSError, TypeError, ValueError) as exc:
            self.summary["capture_status"] = "not captured"
            raise RequestCaptureError(f"Unable to persist request evidence: {path.name}") from exc
        self.summary["request_record_paths"].append(str(path))
        return str(path)

    def submit(
        self,
        create: Callable[..., Any],
        payload: dict[str, Any],
        *,
        tool_round: int,
        attempt: int,
    ) -> Any:
        """Record exact call arguments and raw returned data before parsing.

        Transport credentials and SDK-private requests are outside this record.
        A request without an outcome remains an interrupted or pending attempt.
        """
        identity = {"tool_round": tool_round, "attempt": attempt}
        prefix = f"round{tool_round:02d}_attempt{attempt:02d}"
        self.write(
            f"{prefix}_request",
            {
                **identity,
                "kind": "provider_request",
                "capture_status": "captured",
                "payload": deepcopy(payload),
            },
        )
        self.summary.update(capture_status="captured", request_sent=True)
        try:
            response = create(**payload)
        except Exception as exc:  # noqa: BLE001 - capture and re-raise provider errors
            self.write(
                f"{prefix}_error",
                {
                    **identity,
                    "kind": "provider_error",
                    "error_type": type(exc).__name__,
                    "status_code": getattr(exc, "status_code", None),
                    "request_id": getattr(exc, "request_id", None),
                    "code": getattr(exc, "code", None),
                },
            )
            raise
        self.write(
            f"{prefix}_response",
            {
                **identity,
                "kind": "provider_response",
                "raw_response": _response_payload(response),
                "request_id": getattr(response, "_request_id", None),
            },
        )
        return response
