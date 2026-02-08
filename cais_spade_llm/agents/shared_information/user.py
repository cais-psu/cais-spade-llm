"""User-facing SPADE agent that sends commands and collects replies."""

# agents/shared_information/user.py
from __future__ import annotations
import json, logging
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
from spade.template import Template

class User(Agent):
    """Minimal operator agent that tracks inbox messages and task statuses."""
    agent_role = "user"

    def __init__(self, jid: str, password: str, *, name: str = "user"):
        """Initialize the user agent and set up its logging/inbox state."""
        super().__init__(jid, password)
        self.name = name
        self.annotation = "Entry point for a human operator."
        # Store raw incoming messages for inspection/debugging.
        self.inbox: list[tuple[str, str | None, str]] = []
        # Track task_id -> status for quick operator feedback.
        self.task_states: dict[str, str] = {}

        # Dedicated per-agent logger (file + console) for operator-visible trails.
        self.logger = logging.getLogger(f"agent_{self.name}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh = logging.FileHandler(f"cais_spade_llm/log/{self.name}_actions.log", mode="a"); fh.setFormatter(fmt)
        ch = logging.StreamHandler(); ch.setFormatter(fmt)
        if not self.logger.handlers:
            self.logger.addHandler(fh); self.logger.addHandler(ch)

    async def setup(self):
        """Register inbox behaviours for common operator-facing message types."""
        # Only capture common operator-facing types; avoid stealing replies awaited elsewhere
        for t in ("chat", "ack", "task"):
            tpl = Template(); tpl.set_metadata("type", t)
            self.add_behaviour(self._Inbox(), tpl)
        self.logger.info(f"[ready] {self.jid} (User agent)")

    class _Inbox(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=0.5)
            if not msg: return
            mtype = (msg.metadata or {}).get("type")
            body = msg.body or ""
            sender = str(msg.sender)
            # Keep a raw record of every message for later review.
            self.agent.inbox.append((sender, mtype, body))
            try:
                data = json.loads(body)
                tid, status = data.get("task_id"), data.get("status")
                if tid and status:
                    # Store latest status per task so UIs can quickly render progress.
                    self.agent.task_states[tid] = status
            except Exception:
                pass
            self.agent.logger.info(f"[User.inbox] from={sender} type={mtype} body={body[:200]}")

    async def say(self, to_jid: str, text: str, *, mtype: str = "chat") -> None:
        """Send a one-way message to another agent."""
        msg = Message(to=to_jid); msg.set_metadata("type", mtype); msg.body = text
        await self.send(msg)
        self.logger.info(f"[User.say] → {to_jid} type={mtype} body={text[:200]}")

    async def request_reply(self, to_jid: str, *, mtype: str, payload: dict, timeout: float = 10.0):
        """Send a request and wait (briefly) for a matching reply message."""
        msg = Message(to=to_jid); msg.set_metadata("type", mtype); msg.body = json.dumps(payload)
        await self.send(msg)

        deadline = self.loop.time() + timeout
        while self.loop.time() < deadline:
            reply = await self.receive(timeout=0.5)
            if not reply: 
                continue
            if str(reply.sender) == to_jid and reply.metadata.get("type") in ("llm.reply", "tool.reply"):
                return reply
            # log any other unsolicited messages too
            self.inbox.append((str(reply.sender), reply.metadata.get("type"), reply.body or ""))
        return None
