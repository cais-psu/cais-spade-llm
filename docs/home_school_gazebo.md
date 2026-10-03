# Home and school Gazebo

The updated code and Gazebo simulation run in school WSL at
`/home/jongh/projects/cais-spade-llm` on `recovery-framework-journal`.
The home PC runs `gzclient` to display that same simulation.

## One installation at home

1. In school-connected VS Code, select `school wsl` under Run and Debug.
   Stop the current debug session and press F5 to load the updated code.
2. Close the manually started home Gazebo window.
3. In home Ubuntu (`home-pc`), run:

   ```bash
   curl -fsS http://100.111.96.116:8080/home-gazebo-viewer/launcher.py -o /tmp/home_gazebo_viewer.py
   python3 /tmp/home_gazebo_viewer.py --install
   ```

The installer copies the launcher to
`~/.local/bin/cais_home_gazebo_viewer.py`, starts it in the background, and
creates `CAIS Gazebo viewer.lnk` in the home Windows Startup folder.
It runs again at Windows login. No Administrator access is needed.
The launcher polls school over Tailscale; it opens no inbound home ports.

The existing home checkout supplies the KMR meshes and parts under
`~/projects/cais-spade-llm/ros2/cais_lab_robotics/`. The viewer uses the
installed home `libcais_recovery_render_rate.so` and software rendering.

## At home

1. Keep the school PC online and the school UI running with `school wsl`.
2. Open <http://100.111.96.116:8080/> in the home browser.
3. Click **Start Simulation**. The home Gazebo window opens automatically.

If the simulation is already running, the same button opens the home viewer
without restarting the school simulation. Repeated clicks preserve an existing
viewer. The launcher closes only its own viewer when the school simulation stops.

Use the direct Tailscale UI address: a VS Code `localhost:8080` tunnel hides the
home browser address that selects the home viewer. The `school wsl` profile sets
`CAIS_HOME_GAZEBO_IP=100.108.30.10` for that selection.

## At school

Run the same code in school WSL and use the school Gazebo window.
A click from the school browser does not request a home window.
The existing Gazebo launch still starts the school viewer when its WSL display
is available.

## If the home window does not open

In home Ubuntu, read:

```bash
tail -n 40 ~/.cache/cais-spade-llm/home-gazebo-viewer/launcher.log
tail -n 40 ~/.cache/cais-spade-llm/home-gazebo-viewer/gzclient.log
```

After `wsl --shutdown`, start the launcher again or sign out and back in:

```bash
python3 ~/.local/bin/cais_home_gazebo_viewer.py
```

To remove automatic startup, delete `CAIS Gazebo viewer.lnk` from the home
Windows Startup folder (`shell:startup`). This does not modify school code or
stop the school simulation.
