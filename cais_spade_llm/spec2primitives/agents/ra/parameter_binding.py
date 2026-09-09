from __future__ import annotations

"""Describe unresolved primitive inputs without calculating targets or repairing steps."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .composition_context import _resolve_json_pointer

_MISSING = object()
_CONFIGURED_FRAME = "controller_config.move_group.frame_id"


def assess_parameter_bindings(
    steps: Sequence[Mapping[str, Any]],
    catalog: Mapping[str, Mapping[str, Any]],
    robot_state: Mapping[str, Any],
    *,
    read_evidence: Callable[[str, str], Any],
    result_schema: Callable[[dict[str, Any]], Any],
) -> list[dict[str, Any]]:
    """Report missing, incompatible, unverified and deferred inputs of validated steps.

    The caller first validates symbols, parameter types and authorized references.
    This assessment reads only those references. It never selects a source,
    evaluates a primitive, or changes the submitted program.
    """
    report = _BindingReport(robot_state, read_evidence, result_schema)
    for index, step in enumerate(steps, start=1):
        entry = catalog[step["primitive_symbol"]]
        params = step["params"]
        required = {p["name"] for p in entry["typed_parameters"] if p["required"]}
        schemas = entry["parameter_schemas"]
        for name, declaration in schemas.items():
            schema = _schema(declaration)
            needed = name in required or schema.get("x-grounding-required") is True
            if name in params or needed:
                report.visit(index, f"/{name}", params.get(name, _MISSING), schema)
        if entry["result_schemas"]:
            report.add(index, "/results", "deferred", "Primitive outputs have not been calculated.")
        if step["primitive_symbol"] == "compute_pick_targets" and "target_pose" in schemas:
            if not params.get("target_pose") and not params.get("detected_parts"):
                report.add(
                    index,
                    "/target_pose",
                    "missing",
                    "Supply an observed target_pose or detected_parts; live detection is not run here.",
                )
        if step["primitive_symbol"] in {"compute_pick_targets", "compute_place_targets"}:
            report.check_geometry_record(index, params.get("product_geometry"))
        if step["primitive_symbol"] == "move_cartesian":
            for field in ("frame_id", "ee_link"):
                if not report.motion_context.get(field):
                    report.add(
                        index,
                        "/robot_state/motion_context/" + field,
                        "missing",
                        f"The selected robot's configured {field} is unavailable; capture fresh context.",
                    )
        if step["primitive_symbol"] in {"compute_pick_targets", "compute_place_targets"}:
            # These vertical helpers mix world product geometry with configured
            # EE feedback. A non-world controller context needs a separate fix.
            report.check_frame(
                index,
                "/robot_state/motion_context/frame_id",
                "world",
                report.motion_context.get("frame_id"),
                True,
            )
            if not report.motion_context.get("tcp_link"):
                report.add(
                    index,
                    "/robot_state/motion_context/tcp_link",
                    "missing",
                    "The configured TCP link is unavailable; a robot target is not just an object location.",
                )
    return report.issues


def _schema(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {"type": value}
    if "type" in value:
        return dict(value)
    return {"type": "object", "properties": dict(value)}


def _field_path(path: str, name: str) -> str:
    return path + "/" + name.replace("~", "~0").replace("/", "~1")


def required_geometry_fields(
    value: Mapping[str, Any], schema: Mapping[str, Any], *, resolve_value: Callable[[Any], Any],
) -> set[str]:
    """Read declared grounding conditions from selected values without choosing evidence."""
    required = set(schema.get("required", [])) | set(schema.get("x-grounding-fields", []))
    for name, declaration in schema.get("properties", {}).items():
        if not isinstance(declaration, Mapping):
            continue
        condition = declaration.get("x-grounding-when", {})
        if not condition:
            continue
        selected: Any = value
        for field in condition["field_path"].split("/")[1:]:
            selected = resolve_value(selected)
            selected = (
                selected.get(field.replace("~1", "/").replace("~0", "~"), _MISSING)
                if isinstance(selected, Mapping) else _MISSING
            )
        if selected is not _MISSING and resolve_value(selected) == condition["equals"]:
            required.add(name)
    return required


class _BindingReport:
    """Inspect one immutable proposal and cache its already-pinned evidence reads."""

    def __init__(
        self,
        robot_state: Mapping[str, Any],
        read_evidence: Callable[[str, str], Any],
        result_schema: Callable[[dict[str, Any]], Any],
    ) -> None:
        context = robot_state.get("motion_context")
        self.motion_context = dict(context) if isinstance(context, Mapping) else {}
        self._read_evidence = read_evidence
        self._result_schema = result_schema
        self._records: dict[str, Any] = {}
        self.issues: list[dict[str, Any]] = []

    def add(self, index: int, path: str, status: str, message: str) -> None:
        issue = {"step_index": index, "parameter_path": path, "status": status, "message": message}
        if issue not in self.issues:
            self.issues.append(issue)

    def record(self, ref: str) -> Any:
        if ref not in self._records:
            self._records[ref] = self._read_evidence(ref, "")
        return self._records[ref]

    def selected_value(self, value: Any) -> Any:
        """Resolve only an explicit selected reference through this check's verified reader."""
        if isinstance(value, dict) and set(value) == {"value_ref"}:
            ref = value["value_ref"]
            document = self.record(ref["record_ref"])
            return _resolve_json_pointer(document, ref["field_path"]) if ref["field_path"] else document
        return value

    def visit(
        self,
        index: int,
        path: str,
        value: Any,
        schema: dict[str, Any],
        source: Mapping[str, Any] | None = None,
    ) -> None:
        if value is _MISSING:
            required = set(schema.get("required", [])) | set(schema.get("x-grounding-fields", []))
            if required:
                for name in sorted(required):
                    self.visit(
                        index,
                        _field_path(path, name),
                        _MISSING,
                        _schema(schema.get("properties", {}).get(name, {})),
                    )
                return
            message = "Required input is unbound."
            if schema.get("x-binding-role") == "controller_identifier" or path == "/model_name":
                message = (
                    "Controller identifier is unbound; part_name does not establish model_name."
                )
            self.add(index, path, "missing", message)
            return
        if isinstance(value, dict) and set(value) == {"value_ref"}:
            ref = value["value_ref"]
            document = self.record(ref["record_ref"])
            if (path == "/target_pose" or path.startswith("/target_pose/")) and document.get("record_type") == "RobotFrameLocationRecord":
                self.add(index, path, "incompatible",
                         "The observed candidate location does not establish the required geometry reference_pose.")
            elif path.split("/")[1] in {"target_pose", "product_geometry"}:
                self.check_measurement_source(index, path, ref, document)
            resolved = self.selected_value(value)
            self.visit(index, path, resolved, schema, document)
            return
        if isinstance(value, dict) and set(value) == {"result_ref"}:
            ref = value["result_ref"]
            actual = _schema(self._result_schema(ref))
            self.add(
                index,
                path,
                "deferred",
                f"Uses an unexecuted result from step {ref['step_index']}; no value is available yet.",
            )
            self.check_result_fields(index, path, schema, actual)
            return

        role = schema.get("x-binding-role")
        if role == "controller_identifier":
            self.add(
                index,
                path,
                "unverified",
                "No controller-identifier binding is established by this value; labels and filenames are insufficient.",
            )
        elif role == "destination_identifier" and value:
            self.add(
                index,
                path,
                "unverified",
                "No approved destination resolver establishes this token's placement geometry.",
            )
        elif role == "end_effector_coordinate" and source is not None:
            if source.get("record_type") in {"RobotFrameLocationRecord", "RobotFramePoseRecord"}:
                self.add(
                    index,
                    path,
                    "unverified",
                    "An observed product reference does not establish a controlled end-effector target or grasp offset.",
                )
        elif role == "vertical_part_height":
            if source is not None and source.get("record_type") == "CADMeshRecord":
                self.add(
                    index,
                    path,
                    "incompatible",
                    "A CAD-local dimension needs an established object orientation to represent vertical part height.",
                )
            elif source is None:
                self.add(
                    index,
                    path,
                    "unverified",
                    "Part height has no measurement or geometry reference.",
                )
        expected_frame = schema.get("x-frame-source")
        if expected_frame:
            self.check_frame(
                index, path, expected_frame, self.source_frame(source), source is not None
            )

        if isinstance(value, dict):
            properties = schema.get("properties", {})
            required = required_geometry_fields(value, schema, resolve_value=self.selected_value)
            for name in sorted(required - value.keys()):
                self.visit(
                    index, _field_path(path, name), _MISSING, _schema(properties.get(name, {}))
                )
            for name, item in value.items():
                if name in properties:
                    self.visit(
                        index, _field_path(path, name), item, _schema(properties[name]), source
                    )
        elif isinstance(value, list) and "items" in schema:
            for position, item in enumerate(value):
                self.visit(
                    index, _field_path(path, str(position)), item, _schema(schema["items"]), source
                )

    def check_result_fields(
        self, index: int, path: str, expected: dict[str, Any], actual: dict[str, Any]
    ) -> None:
        frame = expected.get("x-frame-source")
        if frame:
            self.check_frame(index, path, frame, actual.get("x-frame-source"), True)
        role = expected.get("x-binding-role")
        if (
            role in {"controller_identifier", "destination_identifier"}
            and actual.get("x-binding-role") != role
        ):
            self.add(
                index,
                path,
                "unverified",
                "The selected result does not declare this identifier's meaning.",
            )
        if expected["type"] != "object":
            return
        properties = actual.get("properties", {})
        required = set(expected.get("required", [])) | set(expected.get("x-grounding-fields", []))
        for name in sorted(required - properties.keys()):
            self.add(
                index,
                _field_path(path, name),
                "unverified",
                "The referenced output does not declare this required geometry field.",
            )
        for name, declaration in expected.get("properties", {}).items():
            if name in properties:
                self.check_result_fields(
                    index, _field_path(path, name), _schema(declaration), _schema(properties[name])
                )

    def check_frame(
        self, index: int, path: str, expected: str, actual: str | None, has_source: bool
    ) -> None:
        expected = (
            self.motion_context.get("frame_id") if expected == _CONFIGURED_FRAME else expected
        )
        actual = self.motion_context.get("frame_id") if actual == _CONFIGURED_FRAME else actual
        if not expected:
            self.add(index, path, "missing", "The consumer coordinate frame is unavailable.")
        elif not has_source:
            self.add(
                index,
                path,
                "unverified",
                "Literal coordinate is a proposal without a measurement reference.",
            )
        elif not actual:
            self.add(
                index, path, "unverified", "The coordinate source does not establish its frame."
            )
        elif expected != actual:
            self.add(
                index,
                path,
                "incompatible",
                f"Source frame {actual} differs from required frame {expected}; no conversion was applied.",
            )

    def source_frame(self, source: Mapping[str, Any] | None) -> str | None:
        if source is None:
            return None
        kind = source.get("record_type")
        if kind in {"RobotFrameLocationRecord", "RobotFramePoseRecord"}:
            return source.get("target_frame")
        if kind == "CADMeshRecord":
            return source.get("coordinate_frame")
        if kind == "RobotStateSnapshot":
            state = source.get("robot_state")
            context = state.get("motion_context") if isinstance(state, Mapping) else None
            return context.get("frame_id") if isinstance(context, Mapping) else None
        if kind in {"ObservedGeometryEvidence", "AssemblyGeometryEvidence", "AssemblySurfaceEvidence", "AssemblySceneEvidence", "AssemblyGoalEvidence", "RobotValidationContext", "PrimitiveCalculationRecord"}:
            return source.get("frame_id")
        return None

    def check_measurement_source(
        self, index: int, path: str, ref: Mapping[str, Any], source: Mapping[str, Any],
    ) -> None:
        kind = source.get("record_type")
        if kind == "CADMeshRecord":
            return  # Existing dimension/frame checks give the precise CAD-local finding.
        uncertainty = source.get("uncertainty", {})
        height_estimate = (
            path == "/product_geometry/part_height_m" and ref["field_path"] == "/part_height_m"
            and kind == "AssemblyGeometryEvidence" and source.get("status") == "ambiguous"
            and uncertainty.get("method") == "highest_ranked_qualified_pose_hypothesis"
            and uncertainty.get("hypothesis_index") == 0
            and uncertainty.get("complete_pose_established") is False
            and isinstance(uncertainty.get("source_pose"), dict) and isinstance(source.get("warning"), str)
        )
        if (kind not in {"ObservedGeometryEvidence", "AssemblyGeometryEvidence", "AssemblySurfaceEvidence", "AssemblyGoalEvidence"}
                or (source.get("status") != "accepted" and not height_estimate)
                or source.get("frame_id") != "world" or source.get("units") != "m"):
            self.add(index, path, "incompatible", "The selected input requires accepted geometry in world metres with an established semantic reference.")
        elif path == "/product_geometry/board_center/z" and kind == "AssemblySurfaceEvidence" and ref["field_path"] != path:
            self.add(index, path, "incompatible", "A plane coefficient is not a measured world support height.")
        elif path == "/target_pose" or path.startswith("/target_pose/"):
            if kind == "ObservedGeometryEvidence" and (
                source.get("reference_point") != "observed_bounds_center"
                or ref["field_path"] != path.replace("/target_pose", "/reference_pose", 1)
            ):
                self.add(index, path, "incompatible", "The target must select the observed bounds reference_pose for this parameter.")

    def check_geometry_record(self, index: int, value: Any) -> None:
        value = self.selected_value(value)
        if isinstance(value, dict) and value.get("record_type") == "CADMeshRecord":
            self.add(
                index,
                "/product_geometry",
                "incompatible",
                "CADMeshRecord describes a CAD-local mesh; it is not the helper's grounded geometry input.",
            )
