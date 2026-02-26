import os

filepath = '/home/jongh/projects/cais-spade-llm/ros2/xarm_gazebo/launch/xarm6_ur5e_gazebo.launch.py'
with open(filepath, 'r') as f:
    content = f.read()

target = "        f'simulation_controllers:={combined_controllers_yaml}',\n    ]).decode('utf-8')"
replacement = "        f'simulation_controllers:={combined_controllers_yaml}',\n        f'initial_positions_file:={os.path.join(get_package_share_directory(\"xarm_gazebo\"), \"config\", \"ur5e_initial_positions.yaml\")}',\n    ]).decode('utf-8')"

if target in content:
    content = content.replace(target, replacement)
    with open(filepath, 'w') as f:
        f.write(content)
    print("Success")
else:
    print("Failed to find target")
