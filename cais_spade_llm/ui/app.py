"""NiceGUI application factory: layout shell, sidebar navigation, page routing."""

from __future__ import annotations

from nicegui import ui, app

from cais_spade_llm.ui.bridge import SystemBridge


# Colour palette.
_SIDEBAR_BG = "bg-slate-800"
_HEADER_BG = "bg-slate-900"
_NAV_ITEMS = [
    ("Dashboard", "/", "dashboard"),
    ("Plan", "/plan", "account_tree"),
    ("Robots", "/robots", "precision_manufacturing"),
    ("Safety", "/safety", "shield"),
    ("Logs", "/logs", "terminal"),
    ("Products", "/products", "inventory_2"),
]


def _header(bridge: SystemBridge) -> None:
    with ui.header().classes(f"{_HEADER_BG} text-white items-center gap-4 px-6"):
        ui.label("CAIS-SPADE Operator Console").classes("text-lg font-bold")
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
        mode_badge = ui.badge(bridge.execution_mode).props("color=blue outline")

        def _update_badge():
            mode_badge.text = bridge.execution_mode

        ui.timer(2.0, _update_badge)


def _sidebar() -> None:
    with ui.left_drawer().classes(f"{_SIDEBAR_BG} text-white").props("width=220 bordered"):
        ui.label("Navigation").classes("text-xs text-slate-400 uppercase tracking-wider px-4 pt-4 pb-2")
        for label, path, icon in _NAV_ITEMS:
            with ui.link(target=path).classes("no-underline"):
                with ui.row().classes("items-center gap-3 px-4 py-2 hover:bg-slate-700 rounded cursor-pointer w-full"):
                    ui.icon(icon).classes("text-slate-300")
                    ui.label(label).classes("text-slate-200 text-sm")


def _page_wrapper(bridge: SystemBridge):
    """Standard page wrapper: header + sidebar."""
    _header(bridge)
    _sidebar()


def create_app() -> None:
    """Register all NiceGUI pages and configure the app."""
    bridge = SystemBridge.instance()

    # Import page renderers.
    from cais_spade_llm.ui.pages import dashboard, plan, logs, safety, robots, products

    @ui.page("/")
    def index_page():
        _page_wrapper(bridge)
        dashboard.render(bridge)

    @ui.page("/plan")
    def plan_page():
        _page_wrapper(bridge)
        plan.render(bridge)

    @ui.page("/robots")
    def robots_page():
        _page_wrapper(bridge)
        robots.render(bridge)

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
        title="CAIS-SPADE Operator Console",
        port=8080,
        reload=False,
        show=False,
    )
