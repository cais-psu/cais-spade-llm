from robot_UR_patch import apply_urx_patches
apply_urx_patches()

import urx

rob = urx.Robot("192.168.0.10")   # example IP
print(rob.getl())                 # now returns a plain python list
rob.movel([0.3, -0.1, 0.28, -3.14, 0, 0], acc=0.2, vel=0.2)  # movex is used internally