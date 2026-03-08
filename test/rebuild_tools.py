import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from agents.resource_agent.robot_agent import RobotAgent
from function_analyzer import FunctionAnalyzer

def rebuild():
    # We create two instances just like the main initialization to seed both xarm6 and ur5e names
    xarm = RobotAgent(jid="dummy@localhost", password="", name="xarm6")
    ur5e = RobotAgent(jid="dummy@localhost", password="", name="ur5e")
    
    FunctionAnalyzer.build_tools_catalogue(
        agents=[xarm, ur5e], 
        outfile="cais_spade_llm/initialization/tools.json"
    )

if __name__ == "__main__":
    rebuild()
