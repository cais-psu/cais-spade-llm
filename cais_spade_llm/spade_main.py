"""SPADe entry point that collects initialization data, spins up agents, and keeps them running."""

from __future__ import annotations
import os, asyncio, logging
from spade import run as spade_run
import utils, agent_creator
from function_analyzer import FunctionAnalyzer
from agent_creator import ALLOWED_FUNCS

# Silence third‑party loggers that otherwise spam the console during development.
for n in ("pyjabber", "winloop", "asyncio"):
    logging.getLogger(n).setLevel(logging.CRITICAL)

# Static filesystem locations for initialization payloads and generated tool catalogues.
PRODUCT_DIR  = "cais_spade_llm/initialization/products/"
RESOURCE_DIR = "cais_spade_llm/initialization/resources/"
TOOLS_OUT    = "cais_spade_llm/initialization/tools.json"
CCA_INIT     = "cais_spade_llm/initialization/cca.json"

async def spade_main():
    """Orchestrate the entire SPADE session: load configs, spawn agents, register tools, and keep the loop alive."""
    # Collect initialization payloads describing products and hardware resources.
    prod_files = utils.get_init_files(PRODUCT_DIR)
    res_files  = utils.get_init_files(RESOURCE_DIR)

    # Instantiate the user agent plus every resource/product agent defined in the JSON payloads.
    user       = agent_creator.create_user()
    resources  = agent_creator.create_resource_agents(res_files, CCA_INIT)
    products   = agent_creator.create_product_agents(prod_files, resources, CCA_INIT)
    cca        = agent_creator.create_central_controller(CCA_INIT, resources)  # <-- pass resources for capability overview

    # Build the single tools catalogue consumed by the LLM so it knows which agent functions are callable.
    FunctionAnalyzer.build_tools_catalogue(
        agents=products + resources,
        allowed=ALLOWED_FUNCS,
        outfile=TOOLS_OUT,
    )

    # Start agents: resources -> CCA -> user -> products
    for ra in resources:
        await ra.start(auto_register=True)

    if cca:
        await cca.start(auto_register=True)

    if user:
        await user.start(auto_register=True)

    for i, pa in enumerate(products):
        await asyncio.sleep(0.2 * i)
        await pa.start(auto_register=True)

    print("Agents running. Press Ctrl+C to stop.")
    try:
        # Keep the event loop alive so agents remain connected until the operator interrupts the process.
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        for a in products + resources + ([cca] if cca else []) + ([user] if user else []):
            try:
                await a.stop()
            except:
                pass

if __name__ == "__main__":
    # Launch the SPADE runtime with an embedded XMPP server so the agents can communicate locally.
    spade_run(spade_main(), embedded_xmpp_server=True)
