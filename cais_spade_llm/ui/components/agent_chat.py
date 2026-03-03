"""Reusable chat widget for querying SPADE agents in natural language."""

from __future__ import annotations

from nicegui import ui

from cais_spade_llm.ui.bridge import SystemBridge


def render_chat(
    bridge: SystemBridge,
    *,
    agent_jid: str | None = None,
    agent_options: dict[str, str] | None = None,
    title: str = "Agent Chat",
) -> None:
    """Render a chat panel inside the current NiceGUI context.

    Args:
        bridge: The SystemBridge singleton.
        agent_jid: Fixed target agent JID. When set the selector is hidden.
        agent_options: ``{jid: display_name}`` for the dropdown.
            Ignored when *agent_jid* is provided.  When *None* and
            *agent_jid* is also *None*, a default set with Auto-route
            is shown.
        title: Card title.
    """
    with ui.card().classes("w-full h-full flex flex-col"):
        ui.label(title).classes("text-lg font-semibold mb-2")

        # ── Agent selector (only in auto-route mode) ──────────
        show_selector = agent_jid is None
        if show_selector:
            default_options = {"auto": "Auto-route"}
            if agent_options:
                default_options.update(agent_options)
            agent_select = ui.select(
                default_options,
                value="auto",
                label="Target Agent",
            ).classes("w-full")
        else:
            agent_select = None

        # ── Chat history ──────────────────────────────────────
        chat_log = ui.scroll_area().classes("w-full border rounded bg-slate-50").style("min-height: 600px")
        chat_column = None
        with chat_log:
            chat_column = ui.column().classes("w-full gap-2 p-3")

        # Message store (closure-local).
        messages: list[dict] = []

        def _render_messages():
            chat_column.clear()
            with chat_column:
                if not messages:
                    ui.label("No messages yet. Type a question below.").classes(
                        "text-slate-400 italic text-sm"
                    )
                    return
                for msg in messages[-50:]:
                    if msg["role"] == "user":
                        with ui.row().classes("w-full justify-end"):
                            ui.chat_message(
                                msg["text"],
                                name="You",
                                sent=True,
                            ).classes("max-w-xs")
                    else:
                        with ui.row().classes("w-full"):
                            ui.chat_message(
                                msg["text"],
                                name=msg.get("agent", "Agent"),
                                sent=False,
                            ).classes("max-w-xs")

        _render_messages()

        # ── Input row ─────────────────────────────────────────
        with ui.row().classes("w-full gap-2 mt-2 items-center"):
            chat_input = ui.input(placeholder="Ask a question...").classes(
                "flex-grow"
            ).props("outlined dense")
            send_btn = ui.button(icon="send").props("color=primary round dense")

        async def _on_send():
            text = chat_input.value.strip()
            if not text:
                return

            # Record user message.
            messages.append({"role": "user", "text": text})
            chat_input.value = ""
            _render_messages()

            # Determine target label.
            if agent_jid:
                target_label = agent_jid.split("@")[0]
            elif agent_select and agent_select.value != "auto":
                target_label = agent_select.value.split("@")[0]
            else:
                target_label = "system"

            # Placeholder response (no backend wired yet).
            messages.append({
                "role": "agent",
                "agent": target_label,
                "text": f"[Chat backend not connected yet. Your message: \"{text}\"]",
            })
            _render_messages()

        send_btn.on_click(_on_send)
        chat_input.on("keydown.enter", _on_send)
