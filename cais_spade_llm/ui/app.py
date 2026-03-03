"""NiceGUI application factory: layout shell, sidebar navigation, page routing."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

from nicegui import ui, app
from nicegui.elements.drawer import Drawer as NiceGUIDrawer
from nicegui.elements.timer import Timer as NiceGUITimer

from cais_spade_llm.ui.bridge import SystemBridge

_STATIC_DIR = Path(__file__).parent / "static"


# Colour palette.
_SIDEBAR_BG = "bg-slate-800"
_HEADER_BG = "bg-slate-900"
_NAV_ITEMS = [
    ("Dashboard", "/", "dashboard"),
    ("Control", "/control", "gamepad"),
    ("Central Controller Agent", "/safety", "shield"),
    ("Product Agent", "/products", "inventory_2"),
    ("Resource Agent", "/resources", "precision_manufacturing"),
    ("Logs", "/logs", "terminal"),
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
        if kwargs.get("value", None) is None:
            kwargs["value"] = True
        return original_drawer_init(self, side, **kwargs)

    NiceGUIDrawer.__init__ = _safe_drawer_init
    _patch_nicegui_lifecycle._patched = True


def _header(bridge: SystemBridge) -> None:
    with ui.header().classes(f"{_HEADER_BG} text-white items-center gap-4 px-6"):
        with ui.link(target="/").classes("no-underline flex items-center gap-3"):
            ui.image("/static/favicon.ico").classes("w-8 h-8")
            ui.label("Penn State CAIS Lab Multi-Agent Manufacturing System").classes("text-lg font-bold text-white")
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
        _MODE_DISPLAY = {"simulate": "Dry Run", "ros2": "Simulation", "real": "Physical"}
        mode_badge = ui.badge(_MODE_DISPLAY.get(bridge.execution_mode, bridge.execution_mode)).props("color=blue outline")

        def _update_badge():
            mode_badge.text = _MODE_DISPLAY.get(bridge.execution_mode, bridge.execution_mode)

        ui.timer(2.0, _update_badge)


def _sidebar() -> None:
    # Use an explicit initial drawer state to avoid JS value probing timeout on slow/disconnecting clients.
    with ui.left_drawer(value=True).classes(f"{_SIDEBAR_BG} text-white").props("width=240 bordered"):
        ui.label("Navigation").classes("text-xs text-slate-400 uppercase tracking-wider px-4 pt-4 pb-2")
        for label, path, icon in _NAV_ITEMS:
            with ui.link(target=path).classes("no-underline"):
                with ui.row().classes("items-center gap-3 px-4 py-2 hover:bg-slate-700 rounded cursor-pointer w-full"):
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

    # Serve static assets and set Penn State favicon.
    app.add_static_files("/static", str(_STATIC_DIR))

    # Import page renderers.
    from cais_spade_llm.ui.pages import dashboard, control, logs, safety, resources, products

    @ui.page("/")
    def index_page():
        _page_wrapper(bridge)
        dashboard.render(bridge)

    @ui.page("/control")
    def control_page():
        _page_wrapper(bridge)
        control.render(bridge)

    @ui.page("/resources")
    def resources_page():
        _page_wrapper(bridge)
        resources.render(bridge)

    @ui.page("/safety")
    def safety_page():
        _page_wrapper(bridge)
        safety.render(bridge)

    @ui.page("/logs")
    def logs_page():
        _page_wrapper(bridge)
        logs.render(bridge)

    @ui.page("/products")
    def products_page():
        _page_wrapper(bridge)
        products.render(bridge)

    ui.run(
        title="Penn State CAIS Lab Multi-Agent Manufacturing System",
        favicon=str(_STATIC_DIR / "favicon.ico"),
        port=8080,
        reload=False,
        show=False,
        reconnect_timeout=30.0,
    )
