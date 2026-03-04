
from agents.resource_agent.robot_agent import RobotAgent

if __name__ == "__main__":
    print("Starting RobotAgent...")
    ra = RobotAgent("robot1@localhost", "password", name="robot1", robotIP="192.168.1.172", robotType="ur5e")
