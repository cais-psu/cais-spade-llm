"""Scenario gates around ordinary negotiated robot execution."""

from __future__ import annotations

import asyncio
import time


class SlippagePickupGate:
    """Release the preset's two admitted pickup approaches together."""

    def __init__(self, runtime, configuration: dict) -> None:
        self.runtime = runtime
        other = configuration["additional_condition"]
        self.parts = {configuration["resource_id"]: configuration["part_name"],
                      other["resource_id"]: other["part_name"]}
        self.arrived: dict[str, str] = {}
        self.released = asyncio.Event()

    async def wait(self, task: dict) -> None:
        """Wait for both owning tasks after their normal safety approvals."""
        rid = task["resource_id"]
        if (task["event_name"] != "pick_approach" or rid not in self.parts
                or task["parameters"].get("part_name") != self.parts[rid] or self.released.is_set()):
            return
        context = self.runtime.context
        if task.get("run_id") != context.run_id or context.pending_for(task["task_id"]) != task:
            raise ValueError("Pickup gate requires the current admitted resource task")
        if self.runtime.stopped:
            raise asyncio.CancelledError()
        previous = self.arrived.get(rid)
        if previous is not None and previous != task["task_id"]:
            raise ValueError("Pickup gate received another task for the same resource")
        self.arrived[rid] = task["task_id"]
        context.negotiations.append({
            "kind": "slippage_pickup_ready", "run_id": context.run_id,
            "task_id": task["task_id"], "resource_id": rid, "timestamp": time.time(),
        })
        if set(self.arrived) == set(self.parts):
            context.negotiations.append({
                "kind": "slippage_pickups_released", "run_id": context.run_id,
                "tasks": dict(self.arrived), "timestamp": time.time(),
            })
            self.released.set()
        await self.released.wait()
        recording_ready = getattr(self.runtime, "slippage_recording_ready", None)
        if recording_ready is not None:
            await recording_ready.wait()
        if self.runtime.stopped or context.pending_for(task["task_id"]) != task:
            raise asyncio.CancelledError()
