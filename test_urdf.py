import xml.etree.ElementTree as ET
import subprocess

# we just want to run the generation logic manually
import subprocess

try:
    ur5e_prefix = 'ur5e_'
    onrobot_prefix = f'ur5e_rg2_'
    onrobot_raw = subprocess.check_output([
        'xacro',
        '/home/jongh/ros2_ws/src/OnRobot_ROS2_Description/urdf/onrobot.urdf.xacro',
        'onrobot_type:=rg2',
        'name:=rg2',
        f'prefix:={onrobot_prefix}',
        'sim_gazebo:=true',
    ]).decode('utf-8')
    onrobot_root = ET.fromstring(onrobot_raw)

    print("--- ROS2 CONTROL TAGS IN ONROBOT ---")
    for ros2_ctl in onrobot_root.findall('ros2_control'):
        for j in ros2_ctl.findall('joint'):
            print(f"Joint: {j.get('name')}")
            
    print("--- GAZEBO PLUGIN TAGS ---")
    for gz in onrobot_root.findall('gazebo'):
        for pl in gz.findall('plugin'):
            print(f"Plugin: {pl.get('filename')} - {pl.get('name')}")

    print("--- MIMIC JOINTS IN ONROBOT ---")
    for j in onrobot_root.findall('joint'):
        mimic = j.find('mimic')
        if mimic is not None:
            print(f"Mimic: {j.get('name')} -> {mimic.get('joint')}")

except Exception as e:
    import traceback
    traceback.print_exc()
