import os
import xml.etree.ElementTree as ET

filepath = '/home/jongh/projects/cais-spade-llm/ros2/xarm_gazebo/launch/xarm6_ur5e_gazebo.launch.py'
with open(filepath, 'r') as f:
    content = f.read()

injection = """
    # Inject initial values for xArm joints
    xarm_initial_positions = {
        f'{xarm_prefix}joint1': '0.0',
        f'{xarm_prefix}joint2': '-1.047',
        f'{xarm_prefix}joint3': '-1.047',
        f'{xarm_prefix}joint4': '0.0',
        f'{xarm_prefix}joint5': '-1.57',
        f'{xarm_prefix}joint6': '0.0',
    }
    for hw_joint in xarm_root.findall('.//ros2_control/joint'):
        j_name = hw_joint.get('name')
        if j_name in xarm_initial_positions:
            pos_intf = hw_joint.find('state_interface[@name="position"]')
            if pos_intf is not None:
                param = ET.Element('param', {'name': 'initial_value'})
                param.text = xarm_initial_positions[j_name]
                pos_intf.append(param)
"""

if "xarm_initial_positions = {" not in content:
    content = content.replace("    _strip_world_links_and_joints(xarm_root)", injection + "\n    _strip_world_links_and_joints(xarm_root)")
    with open(filepath, 'w') as f:
        f.write(content)
    print("Patched xArm launch")
else:
    print("Already patched")
