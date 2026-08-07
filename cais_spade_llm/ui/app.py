"""NiceGUI application factory: layout shell, sidebar navigation, page routing."""

from __future__ import annotations

import asyncio
import base64
from contextlib import nullcontext, suppress
from pathlib import Path

from nicegui import app, ui
from nicegui.elements.drawer import Drawer as NiceGUIDrawer
from nicegui.elements.timer import Timer as NiceGUITimer
from starlette.responses import Response, StreamingResponse

from cais_spade_llm.ui.bridge import SystemBridge
from cais_spade_llm.ui.gazebo_cleanup import keep_gazebo_on_exit

_STATIC_DIR = Path(__file__).parent / "static"
_NO_PERCEPTION_FRAME_JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQ"
    "FxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMK"
    "ChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgo"
    "KCgoKCj/wAARCAAMABADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBg"
    "cICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0"
    "KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZW"
    "ZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxM"
    "XGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQ"
    "AAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQ"
    "dhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSE"
    "lKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqK"
    "mqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADA"
    "MBAAIRAxEAPwD6DooooA//2Q=="
)


# Colour palette.
_SIDEBAR_BG = "bg-slate-800"
_HEADER_BG = "bg-slate-900"
_NAV_ITEMS = [
    ("Dashboard", "/", "dashboard"),
    ("Perception", "/perception", "photo_camera"),
    ("Control", "/control", "gamepad"),
    ("Safety", "/safety", "shield"),
    ("Products", "/products", "inventory_2"),
    ("Resources", "/resources", "precision_manufacturing"),
]


def _patch_nicegui_lifecycle() -> None:
    """Harden NiceGUI against known client-lifecycle timer/drawer issues."""
    if getattr(_patch_nicegui_lifecycle, "_patched", False):
        return

    # 1) Prevent timer crashes when page slots are deleted during reconnect/navigation.
    original_timer_get_context = NiceGUITimer._get_context

    def _safe_timer_get_context(self):
        try:
            return original_timer_get_context(self)
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "parent slot of the element has been deleted" in msg:
                self.cancel()
                return nullcontext()
            raise

    NiceGUITimer._get_context = _safe_timer_get_context

    # 2) Avoid drawer JS probing timeout by forcing explicit initial value.
    original_drawer_init = NiceGUIDrawer.__init__

    def _safe_drawer_init(self, side, **kwargs):
        if kwargs.get("value") is None:
            kwargs["value"] = True
        return original_drawer_init(self, side, **kwargs)

    NiceGUIDrawer.__init__ = _safe_drawer_init
    _patch_nicegui_lifecycle._patched = True


def _header(bridge: SystemBridge) -> None:
    with ui.header().classes(f"{_HEADER_BG} text-white items-center gap-4 px-6"):
        with ui.link(target="/").classes("no-underline flex items-center gap-3"):
            ui.image("/static/favicon.ico").classes("w-8 h-8")
            ui.label("Penn State CAIS Lab Multi-Agent Manufacturing System").classes(
                "text-lg font-bold text-white"
            )
        ui.space()

        # System status indicator.
        status_icon = ui.icon("circle").classes("text-sm")
        status_label = ui.label("Stopped").classes("text-sm")

        def _update_status():
            if bridge.system_running:
                status_icon.props("color=green")
                status_label.text = "Running"
            elif bridge._starting:
                status_icon.props("color=yellow")
                status_label.text = "Starting..."
            else:
                status_icon.props("color=red")
                status_label.text = "Stopped"

        ui.timer(1.0, _update_status)

        # Execution mode badge.
        _MODE_DISPLAY = {"dry_run": "Dry Run", "simulation": "Simulation", "physical": "Physical"}
        mode_badge = ui.badge(
            _MODE_DISPLAY.get(bridge.execution_mode, bridge.execution_mode)
        ).props("color=blue outline")

        def _update_badge():
            mode_badge.text = _MODE_DISPLAY.get(bridge.execution_mode, bridge.execution_mode)

        ui.timer(2.0, _update_badge)


def _sidebar() -> None:
    # Use an explicit initial drawer state to avoid JS value probing timeout on slow/disconnecting clients.
    with (
        ui.left_drawer(value=True).classes(f"{_SIDEBAR_BG} text-white").props("width=240 bordered")
    ):
        ui.label("Navigation").classes(
            "text-xs text-slate-400 uppercase tracking-wider px-4 pt-4 pb-2"
        )
        for label, path, icon in _NAV_ITEMS:
            with ui.link(target=path).classes("no-underline"):
                with ui.row().classes(
                    "items-center gap-3 px-4 py-2 hover:bg-slate-700 rounded cursor-pointer w-full"
                ):
                    ui.icon(icon).classes("text-slate-300")
                    ui.label(label).classes("text-slate-200 text-sm leading-5")


def _page_wrapper(bridge: SystemBridge):
    """Standard page wrapper: header + sidebar."""
    _header(bridge)
    _sidebar()


def create_app() -> None:
    """Register all NiceGUI pages and configure the app."""
    _patch_nicegui_lifecycle()
    bridge = SystemBridge.instance()
    watchdog_task: asyncio.Task | None = None
    perception_recovery_task: asyncio.Task | None = None
    ros2_process_watchdog_task: asyncio.Task | None = None

    # Serve static assets and set Penn State favicon.
    app.add_static_files("/static", str(_STATIC_DIR))

    # Serve safety preview artifacts (DFA PNGs) so browsers that reject
    # large base64 data URLs (e.g. Microsoft Edge) can load images via URL.
    _safety_previews_dir = (
        Path(__file__).resolve().parent.parent / "user_verified_safety" / "previews"
    )
    _safety_previews_dir.mkdir(parents=True, exist_ok=True)
    app.add_static_files("/safety-previews", str(_safety_previews_dir))

    # Import page renderers.
    from cais_spade_llm.ui.pages import control, dashboard, perception, products, resources, safety

    @app.get("/perception/stream/{camera_role}/{stream_name}")
    async def perception_stream(camera_role: str, stream_name: str) -> StreamingResponse:
        """Stream throttled preview JPEGs without invoking Roboflow inference."""
        from cais_spade_llm.ui.perception_manager import CAMERA_ROLES, PREVIEW_ROOT

        if camera_role not in CAMERA_ROLES or stream_name not in {
            "color",
            "depth",
            "detection",
        }:
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="unknown perception preview")
        image_path = PREVIEW_ROOT / camera_role / f"{stream_name}.jpg"
        fallback_path = (
            PREVIEW_ROOT / camera_role / "color.jpg"
            if stream_name == "detection"
            else image_path
        )

        async def frames():
            last_image: tuple[str, int] | None = None
            while True:
                try:
                    current_path = image_path if image_path.is_file() else fallback_path
                    if current_path.is_file():
                        modified_ns = current_path.stat().st_mtime_ns
                        image_identity = (str(current_path), modified_ns)
                        payload = current_path.read_bytes()
                    else:
                        image_identity = ("no-perception-frame", 0)
                        payload = _NO_PERCEPTION_FRAME_JPEG
                except OSError:
                    image_identity = ("no-perception-frame", 0)
                    payload = _NO_PERCEPTION_FRAME_JPEG
                if image_identity != last_image:
                    last_image = image_identity
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(payload)}\r\n\r\n".encode()
                        + payload
                        + b"\r\n"
                    )
                await asyncio.sleep(0.15)

        return StreamingResponse(
            frames(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/perception/frame/{camera_role}/{stream_name}")
    async def perception_frame(camera_role: str, stream_name: str) -> Response:
        """Return the latest preview JPEG without invoking Roboflow inference."""
        from cais_spade_llm.ui.perception_manager import CAMERA_ROLES, PREVIEW_ROOT

        if camera_role not in CAMERA_ROLES or stream_name not in {
            "color",
            "depth",
            "detection",
        }:
            from fastapi import HTTPException

            raise HTTPException(status_code=404, detail="unknown perception preview")
        image_path = PREVIEW_ROOT / camera_role / f"{stream_name}.jpg"
        fallback_path = (
            PREVIEW_ROOT / camera_role / "color.jpg"
            if stream_name == "detection"
            else image_path
        )
        try:
            current_path = image_path if image_path.is_file() else fallback_path
            payload = (
                current_path.read_bytes()
                if current_path.is_file()
                else _NO_PERCEPTION_FRAME_JPEG
            )
        except OSError:
            payload = _NO_PERCEPTION_FRAME_JPEG
        return Response(
            payload,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )

    @ui.page("/")
    def index_page():
        _page_wrapper(bridge)
        dashboard.render(bridge)

    @ui.page("/control")
    def control_page():
        _page_wrapper(bridge)
        control.render(bridge)

    @ui.page("/perception")
    def perception_page():
        _page_wrapper(bridge)
        perception.render(bridge)

    @ui.page("/resources")
    def resources_page():
        _page_wrapper(bridge)
        resources.render(bridge)

    @ui.page("/safety")
    def safety_page():
        _page_wrapper(bridge)
        safety.render(bridge)

    @ui.page("/products")
    def products_page():
        _page_wrapper(bridge)
        products.render(bridge)

    async def _ui_watchdog() -> None:
        """Detect event-loop stalls that can trigger websocket reconnects."""
        loop = asyncio.get_running_loop()
        interval_s = 0.5
        target = loop.time() + interval_s
        last_emit_ts = 0.0
        while True:
            await asyncio.sleep(interval_s)
            now = loop.time()
            lag = now - target
            target = now + interval_s

            # Keep idle mode less noisy: focus diagnostics on active startup/running phases.
            active = bridge._starting or bridge.system_running
            warn_threshold_s = 1.2 if active else 3.5
            min_emit_interval_s = 5.0 if active else 30.0

            if lag >= warn_threshold_s and (now - last_emit_ts) >= min_emit_interval_s:
                try:
                    bridge.log_event_loop_lag(lag)
                    last_emit_ts = now
                except Exception:
                    pass

    async def _perception_recovery_watchdog() -> None:
        """Recover only camera roles explicitly connected by the operator."""
        while True:
            await asyncio.sleep(1.5)
            with suppress(OSError, RuntimeError, ValueError):
                await asyncio.to_thread(bridge.perception_reconcile_connections)

    async def _ros2_process_watchdog() -> None:
        """Reap failed launchers and their child groups without ROS discovery."""
        while True:
            await asyncio.sleep(3.0)
            with suppress(OSError, RuntimeError, ValueError):
                await asyncio.to_thread(bridge.ros2_all_statuses)

    async def _on_startup() -> None:
        nonlocal perception_recovery_task, ros2_process_watchdog_task, watchdog_task
        if keep_gazebo_on_exit():
            import logging

            logging.getLogger("ui.app").info(
                "App startup: CAIS_KEEP_GAZEBO_ON_EXIT=1; preserving prior UI processes."
            )
        else:
            await asyncio.to_thread(bridge.cleanup_previous_ui_processes)
        watchdog_task = asyncio.create_task(_ui_watchdog())
        perception_recovery_task = asyncio.create_task(_perception_recovery_watchdog())
        ros2_process_watchdog_task = asyncio.create_task(_ros2_process_watchdog())

    async def _on_shutdown() -> None:
        """Clean up all background resources when the app exits."""
        nonlocal perception_recovery_task, ros2_process_watchdog_task, watchdog_task
        import logging

        log = logging.getLogger("ui.app")
        log.info("App shutdown: cleaning up resources...")

        # Stop watchdogs first.
        background_tasks = (
            watchdog_task,
            perception_recovery_task,
            ros2_process_watchdog_task,
        )
        for task in background_tasks:
            if task is None:
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                log.debug("App shutdown: watchdog cleanup skipped")
        watchdog_task = None
        perception_recovery_task = None
        ros2_process_watchdog_task = None

        # 1) Stop SPADE agents (which also shuts down robot controllers).
        if bridge.system_running:
            try:
                await bridge.stop_system()
            except Exception:
                log.exception("App shutdown: stop_system failed")

        # A manual Robot Functions session owns a physical UR5e controller even
        # when the SPADE system was never started.
        try:
            await bridge.shutdown_ur5e_robot_function_agent()
        except Exception:
            log.exception("App shutdown: manual ur5e Function Execution cleanup failed")

        # 1b) Stop tracked ROS2 launch processes unless the operator is
        # preserving Gazebo for a debug session.
        if keep_gazebo_on_exit():
            log.info(
                "App shutdown: CAIS_KEEP_GAZEBO_ON_EXIT=1; preserving Gazebo/MoveIt processes."
            )
            bridge.release_ui_process_ownership()
        else:
            try:
                bridge.ros2_stop_all(reason="app_shutdown")
            except Exception:
                log.debug("App shutdown: ROS2 process cleanup skipped")

        # 2) Stop XMPP server.
        try:
            await bridge._stop_xmpp_server()
        except Exception:
            log.debug("App shutdown: XMPP cleanup skipped")

        # 3) Shut down any remaining prewarm controllers.
        try:
            bridge._shutdown_gazebo_prewarm_controllers()
        except Exception:
            log.debug("App shutdown: prewarm cleanup skipped")

        try:
            bridge._shutdown_agent_runtime_loop()
        except Exception:
            log.debug("App shutdown: agent runtime cleanup skipped")

        log.info("App shutdown: cleanup complete.")

    app.on_startup(_on_startup)
    app.on_shutdown(_on_shutdown)

    ui.run(
        title="Penn State CAIS Lab Multi-Agent Manufacturing System",
        favicon=str(_STATIC_DIR / "favicon.ico"),
        port=8080,
        reload=False,
        show=False,
        reconnect_timeout=30.0,
    )
