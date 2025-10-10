# main_spade.py
from __future__ import annotations
import os, asyncio, logging
from spade import run as spade_run
import utils, agent_creator
from function_analyzer import FunctionAnalyzer
from agent_creator import ALLOWED_FUNCS

# Silence noisy logs
for n in ("pyjabber", "winloop", "asyncio"):
    logging.getLogger(n).setLevel(logging.CRITICAL)

# --- Use old-style static paths ---
PRODUCT_DIR  = "cais_spade_llm/initialization/products/"
RESOURCE_DIR = "cais_spade_llm/initialization/resources/"
TOOLS_OUT    = "cais_spade_llm/initialization/tools.json"

async def spade_main():
    # Load initialization files
    prod_files = utils.get_init_files(PRODUCT_DIR)
    res_files  = utils.get_init_files(RESOURCE_DIR)

    # Create agents
    user      = agent_creator.create_user()
    resources = agent_creator.create_resource_agents(res_files)
    products  = agent_creator.create_product_agents(prod_files, resources)

    # Build tools catalogue
    FunctionAnalyzer.build_tools_catalogue(
        agents=products + resources,
        allowed=ALLOWED_FUNCS,
        outfile=TOOLS_OUT,
    )
    
    # Start agents
    for ra in resources:
        await ra.start(auto_register=True)
    if user:
        await user.start(auto_register=True)
    for i, pa in enumerate(products):
        await asyncio.sleep(0.2 * i)
        await pa.start(auto_register=True)

    print("Agents running. Press Ctrl+C to stop.")
    try:
        while True:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        for a in products + resources + ([user] if user else []):
            try:
                await a.stop()
            except:
                pass

if __name__ == "__main__":
    spade_run(spade_main(), embedded_xmpp_server=True)
