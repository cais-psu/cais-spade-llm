"""Multi-camera connection, viewing, calibration, and detection operator page."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from nicegui import ui

from cais_spade_llm.ui.bridge import SystemBridge
from cais_spade_llm.ui.perception_manager import CAMERA_ROLES

_ROLE_LABELS = {"ur5e": "UR5e", "xarm6": "xArm6", "stationary": "Stationary"}


def _safe_age(value: Any) -> str:
    try:
        return f"{float(value):.2f} s"
    except (TypeError, ValueError):
        return "n/a"


class _PerceptionPage:
    """Keep UI element state separate from camera/process implementation."""

    def __init__(self, bridge: SystemBridge) -> None:
        self.bridge = bridge
        self.preflight_labels: dict[str, Any] = {}
        self.role_labels: dict[str, dict[str, Any]] = {}
        self.calibration_labels: dict[str, Any] = {}
        self.detection_view_labels: dict[str, Any] = {}
        self.detection_codes: dict[str, Any] = {}
        self.serial_selects: dict[str, Any] = {}
        self.assigned_serials: dict[str, str] = {}
        self.control_buttons: dict[str, Any] = {}
        self.board_inputs: dict[str, Any] = {}
        self.wsl_rows_by_busid: dict[str, dict[str, str]] = {}
        self.stream_images: list[tuple[Any, str]] = []
        self.image_refresh_sequence = 0
        self.board_initialized = False
        self.refreshing = False

    @staticmethod
    def _notify_result(error: str | None, success: str) -> None:
        ui.notify(
            error or success,
            type="warning" if error else "positive",
            timeout=6000 if error else 2500,
        )

    def render(self) -> None:
        """Render all Perception page sections."""
        ui.label("Camera & Perception").classes("text-2xl font-bold text-slate-800")
        ui.label(
            "Start detection and calibrate ur5e, xarm6, and stationary RealSense cameras. "
            "Viewing, calibration solving, and Test Detection never initiate robot motion."
        ).classes("text-sm text-slate-600 mb-3")
        self._render_preflight()
        self._render_assignments()
        self._render_camera_cards()
        self._render_live_view()
        self._render_calibration()
        self._render_detection()
        self._render_digital_twin()
        self._render_diagnostics()
        ui.timer(0.5, self._refresh_images)
        ui.timer(1.5, self._refresh, immediate=True)

    def _render_preflight(self) -> None:
        with ui.card().classes("w-full"):
            with ui.row().classes("w-full items-center"):
                ui.label("System Preflight").classes("text-lg font-semibold")
                ui.space()
                self.preflight_summary = ui.label("Checking host prerequisites...").classes(
                    "text-sm text-slate-600"
                )
            self.camera_inventory_summary = ui.label(
                "Windows D435: checking | WSL attached: checking | assigned roles: checking"
            ).classes("text-sm text-slate-600")
            with ui.row().classes("gap-2 flex-wrap"):
                for key, label in (
                    ("wsl", "WSL"),
                    ("video_group", "video group"),
                    ("video_permissions", "device permissions"),
                    ("realsense_cli", "librealsense CLI"),
                    ("realsense2_camera_package", "realsense2_camera"),
                    ("realsense2_description_package", "realsense2_description"),
                    ("realsense_udev_rules", "RealSense udev rules"),
                    ("ros2", "ROS2"),
                    ("roboflow_api_key", "Roboflow"),
                ):
                    self.preflight_labels[key] = ui.badge(f"{label}: checking").props("outline")
            ui.label(
                "One-time setup may request sudo outside this page. A sudo password is never "
                "stored. Under WSL, Windows binds each camera once; attachment repeats after "
                "unplug or restart."
            ).classes("text-xs text-slate-500")
            with ui.expansion("Advanced USB", icon="usb").classes("w-full"):
                ui.label(
                    "Normal operation attaches Shared cameras automatically. Use these controls "
                    "only for provisioning diagnostics."
                ).classes("text-xs text-slate-500")
                with ui.row().classes("items-end gap-2 flex-wrap"):
                    self.wsl_busid = ui.select(
                        {},
                        label="Windows RealSense BUSID",
                    ).classes("w-96")
                    self.wsl_busid.on_value_change(self._update_wsl_guidance)
                    ui.button(
                        "Refresh WSL USB",
                        on_click=self._refresh_wsl_devices,
                        icon="refresh",
                    ).props("dense outline")
                    ui.button(
                        "Attach to WSL",
                        on_click=self._attach_wsl_device,
                        icon="usb",
                    ).props("dense outline")
                self.wsl_guidance = ui.label(
                    "Refresh WSL USB to see each camera's Attached, Shared, or Not shared state."
                ).classes("text-xs text-amber-700")

    def _render_assignments(self) -> None:
        with ui.card().classes("w-full"):
            with ui.row().classes("w-full items-center"):
                ui.label("Camera Assignments").classes("text-lg font-semibold")
                ui.space()
                ui.label("Stored in ~/.config/cais-spade-llm/perception_cameras.yaml").classes(
                    "text-xs text-slate-500"
                )
            with ui.row().classes("w-full gap-4 items-end flex-wrap"):
                for role in CAMERA_ROLES:
                    self.serial_selects[role] = ui.select(
                        {"": "Unassigned"},
                        label=_ROLE_LABELS[role],
                        value="",
                    ).classes("w-64")
                ui.button("Discover", on_click=self._discover, icon="usb").props("outline")
                ui.button(
                    "Save Assignments",
                    on_click=self._save_assignments,
                    icon="save",
                ).props("color=primary")
            with ui.row().classes("gap-2"):
                ui.button(
                    "Start All Detection",
                    on_click=self._start_all_detection,
                    icon="play_arrow",
                ).props("color=green")
                ui.button(
                    "Stop All Detection",
                    on_click=self._stop_all_detection,
                    icon="stop",
                ).props("outline color=red")

    def _render_camera_cards(self) -> None:
        ui.label("Camera Cards").classes("text-lg font-semibold mt-3")
        with ui.grid(columns=3).classes("w-full gap-4"):
            for role in CAMERA_ROLES:
                self._render_camera_card(role)

    def _render_camera_card(self, role: str) -> None:
        authority = (
            "executable pose authority"
            if role == "ur5e"
            else "diagnostic" if role == "xarm6" else "observes and verifies"
        )
        with ui.card().classes("w-full"):
            with ui.row().classes("w-full items-center"):
                ui.label(_ROLE_LABELS[role]).classes("text-lg font-semibold")
                ui.space()
                ui.badge(authority).props("outline")
            self.role_labels[role] = {
                "health": ui.label("Not assigned").classes("text-sm text-slate-600"),
                "identity": ui.label("").classes("text-xs text-slate-500"),
                "streams": ui.label("").classes("text-xs text-slate-500"),
                "recovery": ui.label("").classes("text-xs text-blue-700"),
                "error": ui.label("").classes("text-xs text-red-700"),
            }
            with ui.row().classes("gap-1 flex-wrap"):
                ui.button(
                    "Start Detection",
                    on_click=lambda camera_role=role: self._start_detection(camera_role),
                    icon="visibility",
                ).props("dense outline")
                ui.button(
                    "Stop Detection",
                    on_click=lambda camera_role=role: self._stop_detection(camera_role),
                    icon="stop",
                ).props("dense flat")
                ui.button(
                    "Clean Up Camera",
                    on_click=lambda camera_role=role: self._clean_up_camera(camera_role),
                    icon="restart_alt",
                ).props("dense flat color=amber").tooltip(
                    "Stop duplicate or orphaned processes, then restart this camera and detection"
                )
                if role == "ur5e":
                    self.control_buttons[role] = ui.button(
                        "Open Control for world pose",
                        on_click=lambda: ui.navigate.to("/control"),
                        icon="open_in_new",
                    ).props("dense flat color=amber")

    def _render_live_view(self) -> None:
        ui.label("Live View").classes("text-lg font-semibold mt-3")
        ui.label(
            "The large image is the exact annotated inference frame. Raw color and depth remain "
            "live diagnostics; viewing cannot call Roboflow or move a robot."
        ).classes("text-xs text-slate-500")
        with ui.grid(columns=3).classes("w-full gap-4"):
            for role in CAMERA_ROLES:
                self._render_live_card(role)

    def _stream_image(self, source: str, label: str, classes: str) -> Any:
        """Register an ordinary JPEG image for periodic source refresh."""
        image = (
            ui.element("img")
            .props(f'src="{source}" alt="{label}" draggable=false')
            .classes(classes)
        )
        self.stream_images.append((image, source))
        return image

    def _refresh_images(self) -> None:
        """Request fresh preview JPEGs without invoking perception or motion."""
        if getattr(self.preflight_summary, "is_deleted", True):
            return
        self.image_refresh_sequence += 1
        for image, source in self.stream_images:
            if not image.is_deleted:
                image.props["src"] = f"{source}?v={self.image_refresh_sequence}"

    def _render_live_card(self, role: str) -> None:
        color_url = f"/perception/frame/{role}/color"
        depth_url = f"/perception/frame/{role}/depth"
        detection_url = f"/perception/frame/{role}/detection"
        with ui.card().classes("w-full"):
            ui.label(_ROLE_LABELS[role]).classes("font-semibold")
            ui.label("Detection view").classes("text-xs text-slate-500")
            self._stream_image(
                detection_url,
                f"{_ROLE_LABELS[role]} detection view",
                "w-full aspect-[4/3] bg-slate-200 object-contain",
            )
            with ui.row().classes("w-full gap-2 items-start no-wrap"):
                with ui.column().classes("w-1/2 gap-1"):
                    ui.label("Raw color").classes("text-xs text-slate-500")
                    self._stream_image(
                        color_url,
                        f"{_ROLE_LABELS[role]} raw color",
                        "w-full aspect-[4/3] bg-slate-200 object-contain",
                    )
                with ui.column().classes("w-1/2 gap-1"):
                    ui.label("Depth").classes("text-xs text-slate-500")
                    self._stream_image(
                        depth_url,
                        f"{_ROLE_LABELS[role]} depth",
                        "w-full aspect-[4/3] bg-slate-200 object-contain",
                    )
            self.detection_view_labels[role] = ui.label(
                "Waiting for first detection."
            ).classes("text-xs text-slate-600")
            with ui.row().classes("w-full gap-1 flex-wrap"):
                ui.button(
                    "Enlarge",
                    on_click=lambda source=detection_url, label=(
                        f"{_ROLE_LABELS[role]} Detection"
                    ): self._enlarge(
                        source,
                        label,
                    ),
                    icon="center_focus_strong",
                ).props("dense flat")
                ui.button(
                    "Save",
                    on_click=lambda camera_role=role: self._snapshot(
                        camera_role,
                        "detection",
                    ),
                    icon="save",
                ).props("dense flat")
                ui.button(
                    "rqt_image_view",
                    on_click=lambda camera_role=role: self._viewer(camera_role),
                    icon="open_in_new",
                ).props("dense flat")

    def _render_calibration(self) -> None:
        with ui.card().classes("w-full mt-3"):
            ui.label("Calibration").classes("text-lg font-semibold")
            ui.label(
                "For wrist cameras, teach 25 varied poses with Save Pose + Capture. Automatic "
                "replay plans each reviewed pose first and starts only after explicit "
                "confirmation. "
                "UR5e Save Pose + Capture supports teach-pendant Local Control through read-only "
                "RTDE receive and never requests robot motion. It starts its read-only state/TF "
                "monitor automatically; MoveIt and RG2 control remain stopped."
            ).classes("text-xs text-slate-500")
            for role in CAMERA_ROLES:
                self._render_calibration_role(role)
            ui.separator()
            self._render_stationary_board_pose()

    def _render_calibration_role(self, role: str) -> None:
        with ui.expansion(_ROLE_LABELS[role], icon="tune").classes("w-full"):
            self.calibration_labels[role] = ui.label("No calibration status yet.").classes(
                "text-sm text-slate-600"
            )
            color_url = f"/perception/frame/{role}/color"
            ui.label("Calibration live view — raw color").classes(
                "text-xs font-medium text-slate-600"
            )
            self._stream_image(
                color_url,
                f"{_ROLE_LABELS[role]} calibration raw color",
                "w-full max-w-3xl aspect-[4/3] bg-slate-200 object-contain",
            )
            ui.button(
                "Enlarge Calibration View",
                on_click=lambda source=color_url, label=(
                    f"{_ROLE_LABELS[role]} Calibration"
                ): self._enlarge(source, label),
                icon="center_focus_strong",
            ).props("dense flat")
            with ui.row().classes("gap-2 flex-wrap"):
                capture_text = "Capture Sample" if role == "stationary" else "Save Pose + Capture"
                ui.button(
                    capture_text,
                    on_click=lambda camera_role=role: self._capture_sample(camera_role),
                    icon="add_a_photo",
                ).props("dense")
                if role in {"ur5e", "xarm6"}:
                    self._render_replay_buttons(role)
                ui.button(
                    "Solve Candidate",
                    on_click=lambda camera_role=role: self._solve(camera_role),
                    icon="calculate",
                ).props("dense outline")
                ui.button(
                    "Activate",
                    on_click=lambda camera_role=role: self._activate(camera_role),
                    icon="check_circle",
                ).props("dense color=green outline")
                ui.button(
                    "Rollback",
                    on_click=lambda camera_role=role: self._rollback(camera_role),
                    icon="undo",
                ).props("dense flat")
            if role == "ur5e":
                ui.button(
                    "Calibrate Table Plane (10 frames)",
                    on_click=self._table_plane,
                    icon="horizontal_rule",
                ).props("dense outline")

    def _render_replay_buttons(self, role: str) -> None:
        ui.button(
            "Preview Automatic Calibration",
            on_click=lambda camera_role=role: self._preview_replay(camera_role),
            icon="route",
        ).props("dense outline")
        ui.button(
            "Run Automatic Calibration",
            on_click=lambda camera_role=role: self._automatic_confirmation(camera_role),
            icon="smart_toy",
        ).props("dense color=red outline")
        for label, action, icon in (
            ("Pause", "pause", "pause"),
            ("Resume", "resume", "play_arrow"),
            ("Skip", "skip", "skip_next"),
            ("Abort", "abort", "stop"),
        ):
            ui.button(
                label,
                on_click=lambda camera_role=role, replay_action=action: self._replay_control(
                    camera_role,
                    replay_action,
                ),
                icon=icon,
            ).props("dense flat")

    def _render_stationary_board_pose(self) -> None:
        ui.label("Stationary surveyed ChArUco mount pose in world (metres/radians)").classes(
            "text-sm font-medium"
        )
        with ui.row().classes("gap-2 flex-wrap"):
            for field in ("x", "y", "z", "roll", "pitch", "yaw"):
                self.board_inputs[field] = ui.number(
                    label=field,
                    value=0.0,
                    format="%.6f",
                ).classes("w-28")
            ui.button(
                "Save Surveyed Pose",
                on_click=self._save_board_pose,
                icon="save",
            ).props("outline")

    def _render_detection(self) -> None:
        with ui.card().classes("w-full mt-3"):
            ui.label("Detection").classes("text-lg font-semibold")
            ui.label(
                "UR5e also owns canonical /detect_all and /detect_part. xArm6 and stationary use "
                "/perception/<role>/detect_all and remain diagnostic."
            ).classes("text-xs text-slate-500")
            ui.label("LG unavailable: model has no large_gear class").classes(
                "text-xs text-amber-700"
            )
            for role in CAMERA_ROLES:
                with ui.expansion(
                    f"{_ROLE_LABELS[role]} Test Detection",
                    icon="search",
                ).classes("w-full"):
                    self.detection_codes[role] = ui.code("[]", language="json").classes("w-full")
                    ui.button(
                        "Test Detection",
                        on_click=lambda camera_role=role: self._test_detection(camera_role),
                        icon="search",
                    ).props("dense color=secondary")
            self.disagreement = ui.label(
                "No UR5e/stationary comparison is available."
            ).classes("text-sm text-slate-600")

    def _render_digital_twin(self) -> None:
        with ui.card().classes("w-full mt-3"):
            ui.label("Digital Twin").classes("text-lg font-semibold")
            self.twin_status = ui.label("Mirror not running.").classes("text-sm text-slate-600")
            ui.label(
                "Only the authoritative UR5e world pose can spawn/update the passive Gazebo mirror."
            ).classes("text-xs text-slate-500")

    def _render_diagnostics(self) -> None:
        with ui.card().classes("w-full mt-3"):
            ui.label("Diagnostics").classes("text-lg font-semibold")
            ui.button(
                "Record 5 Seconds",
                on_click=self._record_diagnostics,
                icon="fiber_manual_record",
            ).props("outline color=red")
            self.diagnostics = ui.code("{}", language="json").classes(
                "w-full max-h-80 overflow-auto"
            )

    async def _refresh_wsl_devices(self) -> None:
        rows = await asyncio.to_thread(
            self.bridge.perception_discover_wsl_attachments,
            True,
        )
        devices = await asyncio.to_thread(
            self.bridge.perception_discover_devices,
            True,
        )
        self._set_wsl_device_options(rows)
        self._set_serial_options(devices)
        ui.notify(f"Found {len(rows)} Windows RealSense USB row(s)", type="info")

    async def _attach_wsl_device(self) -> None:
        error = await asyncio.to_thread(
            self.bridge.perception_attach_wsl_camera,
            str(self.wsl_busid.value or ""),
        )
        self._notify_result(error, "RealSense attached to WSL")
        rows = await asyncio.to_thread(self.bridge.perception_discover_wsl_attachments)
        self._set_wsl_device_options(rows)
        if error is None:
            devices = await asyncio.to_thread(self.bridge.perception_discover_devices)
            self._set_serial_options(devices)

    def _set_wsl_device_options(self, rows: list[dict[str, str]]) -> None:
        self.wsl_rows_by_busid = {
            str(row.get("busid") or ""): dict(row)
            for row in rows
            if str(row.get("busid") or "")
        }
        current = str(self.wsl_busid.value or "")
        self.wsl_busid.options = {
            busid: (
                f"{busid} — {row.get('description', 'RealSense')} — "
                f"{row.get('state', 'Unknown')}"
            )
            for busid, row in self.wsl_rows_by_busid.items()
        }
        self.wsl_busid.value = current if current in self.wsl_rows_by_busid else None
        self.wsl_busid.update()
        self._update_wsl_guidance()

    def _update_wsl_guidance(self, _event: Any = None) -> None:
        if not hasattr(self, "wsl_guidance"):
            return
        busid = str(self.wsl_busid.value or "")
        row = self.wsl_rows_by_busid.get(busid, {})
        state = str(row.get("state") or "")
        if state == "Not shared":
            text = (
                f"Not shared. In Administrator Windows PowerShell run: "
                f"usbipd bind --busid {busid}. Then refresh and Attach to WSL."
            )
        elif state == "Shared":
            text = "Shared. Click Attach to WSL; administrator credentials are not requested."
        elif state == "Attached":
            text = "Attached to WSL. Click Discover, then assign its serial explicitly."
        else:
            text = "Select a Windows RealSense BUSID to see its attachment guidance."
        self.wsl_guidance.set_text(text)

    def _set_serial_options(self, devices: list[dict[str, str]]) -> None:
        discovered_options = {
            str(row.get("serial") or ""): (
                f"{row.get('serial')} — {row.get('model', 'RealSense')}"
            )
            for row in devices
            if str(row.get("serial") or "")
        }
        saved_options = {
            serial: (
                f"{serial} — saved {_ROLE_LABELS[role]} assignment (not discovered)"
            )
            for role, serial in self.assigned_serials.items()
            if serial and serial not in discovered_options
        }
        for role, select in self.serial_selects.items():
            current = str(select.value or "")
            options = {"": "Unassigned", **saved_options, **discovered_options}
            if current and current not in options:
                options[current] = f"{current} — saved assignment (not discovered)"
            assigned = self.assigned_serials.get(role, "")
            select.options = options
            select.value = current or assigned
            select.update()

    async def _discover(self) -> None:
        devices = await asyncio.to_thread(
            self.bridge.perception_discover_devices,
            True,
        )
        self._set_serial_options(devices)
        if devices:
            ui.notify(f"Discovered {len(devices)} RealSense camera(s)", type="info")
        else:
            ui.notify(
                "No live RealSense cameras were discovered; saved assignments were retained.",
                type="warning",
                timeout=5000,
            )

    async def _save_assignments(self) -> None:
        assignments = {
            role: str(select.value or "")
            for role, select in self.serial_selects.items()
        }
        try:
            await asyncio.to_thread(
                self.bridge.perception_save_assignments,
                assignments,
            )
        except (RuntimeError, ValueError) as exc:
            ui.notify(str(exc), type="warning", timeout=5000)
            return
        self.assigned_serials = assignments
        ui.notify("Camera assignments saved", type="positive")

    async def _start_all_detection(self) -> None:
        outcomes = await asyncio.to_thread(self.bridge.perception_start_all)
        warnings = [f"{role}: {result}" for role, result in outcomes.items() if result != "started"]
        ui.notify(
            "; ".join(warnings) if warnings else "Detection started for all assigned cameras",
            type="warning" if warnings else "positive",
            timeout=6000,
        )

    async def _stop_all_detection(self) -> None:
        await asyncio.to_thread(self.bridge.perception_stop_all)
        ui.notify("Detection and camera stacks stopped for all roles", type="info")

    async def _start_detection(self, role: str) -> None:
        error = await asyncio.to_thread(self.bridge.perception_start_detection, role)
        self._notify_result(error, f"{role} detection started")

    async def _stop_detection(self, role: str) -> None:
        await asyncio.to_thread(self.bridge.perception_stop_detection, role)
        ui.notify(f"{role} detection and camera stack stopped", type="info")

    async def _clean_up_camera(self, role: str) -> None:
        error = await asyncio.to_thread(self.bridge.perception_reset_camera, role)
        self._notify_result(error, f"{role} camera processes cleaned and detection restarted")

    def _enlarge(self, source: str, label: str) -> None:
        with ui.dialog() as dialog, ui.card().classes("w-[90vw] max-w-none"):
            ui.label(f"{label} Live View").classes("text-lg font-semibold")
            self._stream_image(
                source,
                f"{label} enlarged view",
                "w-full max-h-[80vh] object-contain",
            )
            ui.button("Close", on_click=dialog.close)
        dialog.open()

    async def _snapshot(self, role: str, stream: str = "color") -> None:
        try:
            path = await asyncio.to_thread(
                self.bridge.perception_save_snapshot,
                role,
                stream,
            )
        except RuntimeError as exc:
            ui.notify(str(exc), type="warning")
            return
        ui.notify(f"Snapshot saved: {path}", type="positive", timeout=5000)

    async def _viewer(self, role: str) -> None:
        error = await asyncio.to_thread(self.bridge.perception_open_viewer, role)
        self._notify_result(error, f"Opened {role} rqt_image_view")

    async def _capture_sample(self, role: str) -> None:
        try:
            result = await asyncio.to_thread(self.bridge.perception_save_pose_and_capture, role)
        except RuntimeError as exc:
            ui.notify(str(exc), type="warning", timeout=7000)
            return
        ui.notify(str(result["message"]), type="positive", timeout=5000)

    async def _solve(self, role: str) -> None:
        try:
            candidate = await asyncio.to_thread(self.bridge.perception_solve_calibration, role)
        except RuntimeError as exc:
            ui.notify(str(exc), type="warning", timeout=7000)
            return
        ui.notify(f"Accepted candidate: {candidate}", type="positive", timeout=6000)

    async def _activate(self, role: str) -> None:
        try:
            path = await asyncio.to_thread(self.bridge.perception_activate_calibration, role)
        except RuntimeError as exc:
            ui.notify(str(exc), type="warning")
            return
        ui.notify(f"Activated {path}; restart that perception instance", type="positive")

    async def _rollback(self, role: str) -> None:
        try:
            path = await asyncio.to_thread(self.bridge.perception_rollback_calibration, role)
        except RuntimeError as exc:
            ui.notify(str(exc), type="warning")
            return
        ui.notify(f"Restored {path}; restart that perception instance", type="positive")

    async def _table_plane(self) -> None:
        error = await asyncio.to_thread(self.bridge.perception_start_table_plane_calibration)
        self._notify_result(error, "UR5e table-plane calibration started")

    async def _preview_replay(self, role: str) -> None:
        error = await asyncio.to_thread(
            self.bridge.perception_preview_calibration_replay,
            role,
        )
        self._notify_result(error, f"{role} no-motion calibration preview started")

    def _automatic_confirmation(self, role: str) -> None:
        with ui.dialog() as dialog, ui.card().classes("max-w-2xl"):
            ui.label(f"Run Automatic Calibration — {_ROLE_LABELS[role]}").classes(
                "text-lg font-semibold"
            )
            ui.label(
                "This is the only calibration action on this page that moves a robot. It is "
                "available only after the no-motion preview plans every reviewed pose. MoveIt "
                "rechecks each plan, then executes at 10% speed. Keep the workcell clear and "
                "keep an emergency stop available."
            ).classes("text-sm text-red-700")

            async def confirmed_start() -> None:
                dialog.close()
                error = await asyncio.to_thread(
                    self.bridge.perception_start_calibration_replay,
                    role,
                    confirmed=True,
                )
                self._notify_result(error, f"{role} automatic calibration started")

            with ui.row().classes("justify-end w-full"):
                ui.button("Cancel", on_click=dialog.close).props("flat")
                ui.button(
                    "I Confirm — Plan and Replay",
                    on_click=confirmed_start,
                    icon="warning",
                ).props("color=red")
        dialog.open()

    async def _replay_control(self, role: str, action: str) -> None:
        try:
            await asyncio.to_thread(
                self.bridge.perception_calibration_replay_control,
                role,
                action,
            )
        except (RuntimeError, ValueError) as exc:
            ui.notify(str(exc), type="warning")
            return
        ui.notify(f"{role} calibration: {action}", type="info")

    async def _save_board_pose(self) -> None:
        await asyncio.to_thread(
            self.bridge.perception_save_stationary_board_pose,
            {field: float(element.value or 0.0) for field, element in self.board_inputs.items()},
        )
        ui.notify("Stationary surveyed board pose saved", type="positive")

    async def _test_detection(self, role: str) -> None:
        result = await asyncio.to_thread(self.bridge.perception_test_detection, role)
        rows = (
            result.get("detections", [])
            if result.get("world_pose_ready")
            else result.get("visual_detections", [])
        )
        self.detection_codes[role].content = json.dumps(rows, indent=2)
        ui.notify(
            str(result.get("message") or ""),
            type="positive" if result.get("success") else "warning",
            timeout=6000,
        )

    async def _record_diagnostics(self) -> None:
        ui.notify("Recording 5 seconds of camera diagnostics...", type="info")
        path = await asyncio.to_thread(
            self.bridge.perception_record_diagnostics,
            duration_sec=5.0,
        )
        ui.notify(f"Diagnostic recording saved: {path}", type="positive", timeout=6000)

    async def _refresh(self) -> None:
        if getattr(self.preflight_summary, "is_deleted", True):
            return
        if self.refreshing:
            return
        self.refreshing = True
        try:
            status = await asyncio.to_thread(self.bridge.perception_status)
            self._refresh_preflight(status.get("preflight", {}))
            cameras = status.get("cameras", {})
            assignments = {
                role: str((cameras.get(role, {}) or {}).get("serial") or "")
                for role in CAMERA_ROLES
            }
            if assignments != self.assigned_serials:
                self.assigned_serials = assignments
                self._set_serial_options(status.get("devices", []))
            for role in CAMERA_ROLES:
                self._refresh_camera(role, cameras.get(role, {}))
            self._refresh_comparison(status.get("cross_camera_comparisons", []))
            self._refresh_twin(status.get("digital_twin", {}))
            self._refresh_diagnostics(cameras)
            if not self.board_initialized:
                board = status.get("stationary_board_world_pose", {})
                for field, element in self.board_inputs.items():
                    element.value = float(board.get(field, 0.0) or 0.0)
                self.board_initialized = True
        finally:
            self.refreshing = False

    def _refresh_preflight(self, preflight: dict[str, Any]) -> None:
        checks = preflight.get("checks", {})
        setup_message = str(preflight.get("setup_message") or "")
        discovery_error = str(preflight.get("device_discovery_error") or "").strip()
        self.preflight_summary.text = (
            f"{setup_message} RealSense discovery: {discovery_error}"
            if discovery_error
            else setup_message
        )
        inventory = dict(preflight.get("camera_inventory") or {})
        self.camera_inventory_summary.set_text(
            f"Windows D435: {int(inventory.get('windows_d435_devices', 0) or 0)} | "
            f"WSL attached: {int(inventory.get('wsl_attached_devices', 0) or 0)} | "
            f"WSL discovered: {int(inventory.get('wsl_discovered_devices', 0) or 0)} | "
            f"assigned roles: {int(inventory.get('assigned_roles', 0) or 0)}/"
            f"{int(inventory.get('total_roles', len(CAMERA_ROLES)) or len(CAMERA_ROLES))}"
        )
        for key, badge in self.preflight_labels.items():
            ready = bool(checks.get(key, False))
            label = badge.text.split(":", 1)[0]
            badge.text = f"{label}: {'ready' if ready else 'not ready'}"
            badge.props(f"color={'green' if ready else 'orange'} outline")

    @staticmethod
    def _detection_states(camera: dict[str, Any]) -> tuple[str, str]:
        if camera.get("visual_detection_ready"):
            detection_state = "ready"
        elif camera.get("perception_process") == "running":
            detection_state = "waiting"
        else:
            detection_state = "stopped"
        if camera.get("world_pose_ready"):
            world_pose_state = "ready"
        elif camera.get("perception_process") == "running":
            world_pose_state = f"waiting for world → {camera.get('parent_frame', 'tool0')}"
        else:
            world_pose_state = "not evaluated"
        return detection_state, world_pose_state

    def _refresh_camera(self, role: str, camera: dict[str, Any]) -> None:
        assigned_serial = str(camera.get("serial") or "")
        select = self.serial_selects[role]
        if assigned_serial and assigned_serial not in select.options:
            select.options = {**dict(select.options), assigned_serial: assigned_serial}
            select.update()
        if not select.value:
            select.value = assigned_serial
        device = camera.get("device") or {}
        preview = camera.get("preview") or {}
        perception = camera.get("perception") or {}
        detection_preview = camera.get("detection_preview") or {}
        labels = self.role_labels[role]
        detection_state, world_pose_state = self._detection_states(camera)
        labels["health"].text = (
            f"camera={camera.get('camera_process', 'stopped')} | "
            f"preview={camera.get('preview_process', 'stopped')} | "
            f"perception={camera.get('perception_process', 'stopped')} | "
            f"topics={'ready' if camera.get('ros_topic_ready') else 'not ready'} | "
            f"2D detection={detection_state} | world pose={world_pose_state} | "
            f"frame age={_safe_age(camera.get('frame_age_sec'))}"
        )
        labels["identity"].text = (
            f"serial={camera.get('serial') or 'unassigned'} | "
            f"{device.get('model', 'not discovered')} | firmware={device.get('firmware', 'n/a')} | "
            f"USB={device.get('usb_type', 'n/a')}"
            + (
                " — USB2 allowed; move to USB3.x if frames become stale or drop"
                if str(device.get("usb_type") or "").startswith("2")
                else ""
            )
        )
        charuco_marker_count = int(preview.get("charuco_marker_count", 0) or 0)
        charuco_visible = bool(preview.get("charuco_visible", False))
        labels["streams"].text = (
            f"color={float(preview.get('color_average_hz', 0) or 0):.1f} Hz | "
            f"depth={float(preview.get('depth_average_hz', 0) or 0):.1f} Hz | "
            f"ChArUco board={'visible' if charuco_visible else 'NOT VISIBLE'} "
            f"(markers={charuco_marker_count}) | "
            f"{camera.get('topics', {}).get('color', '')}"
        )
        labels["streams"].classes(
            replace=(
                "text-xs text-amber-700"
                if camera.get("ros_topic_ready") and not charuco_visible
                else "text-xs text-slate-500"
            )
        )
        recovery = camera.get("recovery") or {}
        retry_at = recovery.get("next_retry_at")
        retry_text = "n/a"
        if retry_at:
            retry_text = f"{max(0.0, float(retry_at) - time.time()):.1f} s"
        labels["recovery"].text = (
            f"USB attachment={camera.get('attachment_state', 'Unknown')} | "
            f"desired={'yes' if camera.get('desired_connected') else 'no'} | "
            f"recovery={recovery.get('state', 'disconnected')} | "
            f"attempts={int(recovery.get('attempt_count', 0) or 0)}/3 | next={retry_text}"
        )
        perception_error = str(perception.get("last_error") or "").strip()
        pose_error = str(detection_preview.get("pose_error") or "").strip()
        if pose_error and perception_error == pose_error:
            perception_error = ""
        labels["error"].text = str(
            camera.get("camera_last_error")
            or recovery.get("last_error")
            or perception_error
            or preview.get("last_error")
            or ""
        )
        detection_rows = detection_preview.get("detections") or []
        detection_summary = " | ".join(
            f"{row.get('part_name', '')} {row.get('label', '')} "
            f"{float(row.get('confidence', 0.0) or 0.0) * 100.0:.1f}%"
            for row in detection_rows
            if isinstance(row, dict)
        )
        if not detection_preview.get("captured_at"):
            detection_summary = "Waiting for first detection."
        elif not detection_summary:
            detection_summary = "No accepted SG/MG detection"
        latency = detection_preview.get("roboflow_latency_ms")
        detection_summary += f" | age={_safe_age(detection_preview.get('frame_age_sec'))}"
        if latency is not None:
            detection_summary += f" | latency={float(latency):.0f} ms"
        if detection_preview.get("world_pose_ready"):
            detection_summary += " | world pose=ready"
        elif detection_preview.get("captured_at"):
            detection_summary += " | 2D detection only — world pose unavailable"
            if pose_error:
                detection_summary += f": {pose_error}"
        self.detection_view_labels[role].text = detection_summary
        control_button = self.control_buttons.get(role)
        if control_button is not None:
            control_button.visible = bool(
                camera.get("perception_process") == "running"
                and not camera.get("world_pose_ready")
            )
        self._refresh_calibration(role, camera)

    def _refresh_calibration(self, role: str, camera: dict[str, Any]) -> None:
        calibration = camera.get("calibration", {})
        validation = calibration.get("validation", {})
        replay = camera.get("replay", {})
        preview = camera.get("preview", {})
        charuco_visible = bool(preview.get("charuco_visible", False))
        charuco_marker_count = int(preview.get("charuco_marker_count", 0) or 0)
        self.calibration_labels[role].text = (
            f"samples={camera.get('sample_count', 0)} | reviewed poses={camera.get('pose_count', 0)} | "
            f"ChArUco={'visible' if charuco_visible else 'NOT VISIBLE'} "
            f"({charuco_marker_count} markers) | "
            f"active={'accepted' if calibration.get('ready') else 'missing/not accepted'} | "
            f"candidate={'ready' if calibration.get('candidate_ready') else 'none'} | "
            f"reprojection={validation.get('median_reprojection_error_px', 'n/a')} px | "
            f"translation RMS={validation.get('fixed_board_translation_rms_m', 'n/a')} m | "
            f"rotation RMS={validation.get('fixed_board_rotation_rms_deg', 'n/a')} deg | "
            f"replay={replay.get('state', 'idle')}"
        )
        self.calibration_labels[role].classes(
            replace=(
                "text-sm text-amber-700"
                if camera.get("ros_topic_ready") and not charuco_visible
                else "text-sm text-slate-600"
            )
        )

    def _refresh_comparison(self, comparisons: list[dict[str, Any]]) -> None:
        self.disagreement.text = (
            " | ".join(
                f"{row['part_name']}: {float(row['disagreement_m']) * 1000.0:.1f} mm"
                + (" WARNING" if row.get("warning") else "")
                for row in comparisons
            )
            or "No UR5e/stationary comparison is available."
        )
        self.disagreement.classes(
            replace=(
                "text-sm text-amber-700"
                if any(row.get("warning") for row in comparisons)
                else "text-sm text-slate-600"
            )
        )

    def _refresh_twin(self, twin: dict[str, Any]) -> None:
        state = str(twin.get("state") or "not running")
        mirrored = twin.get("mirrored_models") or []
        mirrored_text = ", ".join(str(model) for model in mirrored) or "none"
        waiting_reason = str(twin.get("waiting_reason") or "").strip()
        degraded_reason = str(twin.get("degraded_reason") or "").strip()
        details = [f"state={state}", f"mirrored gears={mirrored_text}"]
        if waiting_reason:
            details.append(f"world pose waiting: {waiting_reason}")
        if degraded_reason:
            details.append(f"degraded: {degraded_reason}")
        if "table plane" in waiting_reason.lower() or "table-plane" in waiting_reason.lower():
            details.append(
                "revalidate hand-eye across varied wrist orientations; if the error remains above "
                "5 mm, recapture hand-eye calibration and then calibrate the table plane"
            )
        self.twin_status.text = " | ".join(details)

    def _refresh_diagnostics(self, cameras: dict[str, Any]) -> None:
        self.diagnostics.content = json.dumps(
            {
                role: {
                    "camera_process": cameras.get(role, {}).get("camera_process"),
                    "preview_process": cameras.get(role, {}).get("preview_process"),
                    "perception_process": cameras.get(role, {}).get("perception_process"),
                    "frame_age_sec": cameras.get(role, {}).get("frame_age_sec"),
                    "attachment_state": cameras.get(role, {}).get("attachment_state"),
                    "recovery": cameras.get(role, {}).get("recovery"),
                    "detection_preview": cameras.get(role, {}).get("detection_preview"),
                    "roboflow_latency_ms": cameras.get(role, {})
                    .get("perception", {})
                    .get("roboflow_latency_ms"),
                    "color_dropped_estimate": cameras.get(role, {})
                    .get("preview", {})
                    .get("color_dropped_estimate"),
                    "depth_dropped_estimate": cameras.get(role, {})
                    .get("preview", {})
                    .get("depth_dropped_estimate"),
                    "last_error": (
                        cameras.get(role, {}).get("camera_last_error")
                        or cameras.get(role, {}).get("perception", {}).get("last_error")
                        or cameras.get(role, {}).get("preview", {}).get("last_error")
                    ),
                    "logs": cameras.get(role, {}).get("logs", {}),
                }
                for role in CAMERA_ROLES
            },
            indent=2,
        )


def render(bridge: SystemBridge) -> None:
    """Render the Camera & Perception page through `SystemBridge`."""
    _PerceptionPage(bridge).render()
