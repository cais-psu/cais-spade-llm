import re

filepath = '/home/jongh/projects/cais-spade-llm/ros2/xarm_gazebo/worlds/table.world'
with open(filepath, 'r') as f:
    content = f.read()

# Models
content = content.replace('<pose>-0.6 0 1.04 0 0 0</pose>', '<pose>-0.3 -0.5 1.04 0 0 0</pose>')
content = content.replace('<pose>0.6 0.4 1.04 0 0 0</pose>', '<pose>0.3 0.6 1.04 0 0 0</pose>')
content = content.replace('<pose>0.6 -0.4 1.04 0 0 0</pose>', '<pose>0.3 0.2 1.04 0 0 0</pose>')

# Small set
content = content.replace('<pose>-0.6 0.1 1.15 0 0 0</pose>', '<pose>-0.3 -0.4 1.15 0 0 0</pose>')
content = content.replace('<pose>-0.6 0.0 1.15 0 0 0</pose>', '<pose>-0.3 -0.5 1.15 0 0 0</pose>')
content = content.replace('<pose>-0.6 -0.1 1.15 0 0 0</pose>', '<pose>-0.3 -0.6 1.15 0 0 0</pose>')

# Medium set
content = content.replace('<pose>0.6 0.5 1.15 0 0 0</pose>', '<pose>0.3 0.7 1.15 0 0 0</pose>')
content = content.replace('<pose>0.6 0.4 1.15 0 0 0</pose>', '<pose>0.3 0.6 1.15 0 0 0</pose>')
content = content.replace('<pose>0.6 0.3 1.15 0 0 0</pose>', '<pose>0.3 0.5 1.15 0 0 0</pose>')

# Large set
content = content.replace('<pose>0.6 -0.3 1.15 0 0 0</pose>', '<pose>0.3 0.3 1.15 0 0 0</pose>')
content = content.replace('<pose>0.6 -0.4 1.15 0 0 0</pose>', '<pose>0.3 0.2 1.15 0 0 0</pose>')
content = content.replace('<pose>0.6 -0.5 1.15 0 0 0</pose>', '<pose>0.3 0.1 1.15 0 0 0</pose>')

# Physics
old_physics = '<surface><friction><ode><mu>1.5</mu><mu2>1.5</mu2></ode></friction><contact><ode><kp>1e6</kp><kd>1.0</kd><min_depth>0.001</min_depth></ode></contact></surface>'
new_physics = '<surface><friction><ode><mu>100.0</mu><mu2>100.0</mu2></ode></friction><contact><ode><kp>1e5</kp><kd>1.0</kd><min_depth>0.005</min_depth></ode></contact></surface>'
content = content.replace(old_physics, new_physics)

with open(filepath, 'w') as f:
    f.write(content)
print("World patched successfully")
