from __future__ import annotations
import asyncio
import json
from datetime import datetime
from spade import run as spade_run
from spade.agent import Agent
from spade.behaviour import OneShotBehaviour, CyclicBehaviour
from spade.message import Message
from spade.template import Template
import sys, asyncio
import logging

for name in ("pyjabber", "winloop", "asyncio"):
    logging.getLogger(name).setLevel(logging.CRITICAL)

#if sys.platform.startswith("win"):
    #asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

DOMAIN = "localhost"   # <- single source of truth for JIDs

# --- Minimal LLM stub ---
class LLMAgent:
    def __init__(self, model: str = "stub-llm"):
        self.model = model

    def generate(self, prompt: str) -> str:
    # Return a deterministic, short instruction for the demo
        return f"INSTRUCTION: pick MCP from P1 and place at AZ. (via {self.model})"

# --- Robot / Resource Agent ---
class RobotAgent(Agent):
    class Inbox(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=1)
            if not msg:
                return
            if msg.metadata.get("type") != "task":
                return
            data = json.loads(msg.body)
            task_id = data.get("task_id")
            instruction = data.get("instruction")
            print(f"[Robot] got task {task_id} :: {instruction}")

            # Simulate doing the work
            await asyncio.sleep(1.0)

            # Send ACK
            ack = Message(to=str(msg.sender))
            ack.set_metadata("type", "ack")
            ack.body = json.dumps({
            "task_id": task_id,
            "status": "completed",
            "finished_at": datetime.utcnow().isoformat()+"Z",
            })
            await self.send(ack)
            print(f"[Robot] completed {task_id}")

    async def setup(self):
        t = Template()
        t.set_metadata("type", "task")
        self.add_behaviour(self.Inbox(), t)
        print(f"[Robot] {self.jid} ready")

# --- Product Agent ---
class ProductAgent(Agent):
    def __init__(self, *args, robot_jid: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.robot_jid = robot_jid
        self.llm = LLMAgent()

    class Kickoff(OneShotBehaviour):
        async def run(self):
            # Use the LLM to create a simple NL instruction
            instruction = self.agent.llm.generate("Create a pick/place instruction for MCP")
            payload = {
            "task_id": "T-001",
            "instruction": instruction,
            }
            msg = Message(to=self.agent.robot_jid)
            msg.set_metadata("type", "task")
            msg.body = json.dumps(payload)
            await self.send(msg)
            print(f"[Product] sent T-001 → {self.agent.robot_jid}")


    class WaitAck(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=1)
            if not msg:
                return
            if msg.metadata.get("type") != "ack":
                return
            data = json.loads(msg.body)
            print(f"[Product] ACK: {data}")


    async def setup(self):
        self.add_behaviour(self.Kickoff())
        t = Template()
        t.set_metadata("type", "ack")
        self.add_behaviour(self.WaitAck(), t)
        print(f"[Product] {self.jid} ready")


async def spade_main():
    robot_jid   = f"robot@{DOMAIN}"
    product_jid = f"product@{DOMAIN}"

    print("[Boot] creating agents…", robot_jid, product_jid)
    robot = RobotAgent(robot_jid, "none")
    product = ProductAgent(product_jid, "none", robot_jid=robot_jid)

    print("[Boot] starting robot…")
    await robot.start(auto_register=True)

    await asyncio.sleep(0.1)

    print("[Boot] starting product…")
    await product.start(auto_register=True)

    await asyncio.sleep(0.1)  # let behaviours attach
    print("Agents started. Press Ctrl+C to stop.")

    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        print("Stopping…")
    finally:
        await product.stop()
        await robot.stop()

if __name__ == "__main__":
    # Only pass embedded_xmpp_server=True; no other kwargs
    spade_run(spade_main(), embedded_xmpp_server=True)