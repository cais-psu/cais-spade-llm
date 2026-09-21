"""Saved recovery-framework experiment settings, without runtime side effects."""

from __future__ import annotations

import json
import math
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

from cais_spade_llm.product.order import validate_completion_conditions, validate_product_order
from cais_spade_llm.recovery_framework import PRODUCT_PATH, ROOT, SCENE_PATH, read_json
from cais_spade_llm.resources.nominal_des import build_nominal_resource_des_models

SETUP_PATH = ROOT / "cais_spade_llm/initialization/recovery_framework_setup.json"
SETUP_RELATIVE = SETUP_PATH.relative_to(ROOT)
EXPERIMENT_PATH = "cais_spade_llm/initialization/recovery_outline_experiment_settings.json"
MODE_OPTIONS = {"simulation": "Simulation", "physical": "Physical"}
RECOVERY_MODE_OPTIONS = {"auto": "Auto", "manual": "Manual", "pre_ran": "Pre-ran"}
RECOVERY_VALIDATION_OPTIONS = {
    "validated": "Recovery Safety Check",
    "no_validation": "No Recovery Safety Check",
}
FAILURE_SCENARIOS = (
    "Conveyor breakdown",
    "ur5e-1 breakdown",
    "Machining breakdown during part processing",
    "Part slippage",
)
CHECKPOINTS = {
    "before_execute": "Before executing the selected task",
    "after_execute_before_commit": "After execution, before successful completion is recorded",
}


def reference_path(value: str, root: Path = ROOT) -> Path:
    """Resolve a configuration reference without changing its stored spelling."""
    if not isinstance(value, str) or not value:
        raise ValueError("A configuration file reference is required")
    path = Path(value)
    return path if path.is_absolute() else root / path


def product_inputs(product_file: str, order_file: str, root: Path = ROOT) -> dict[str, Any]:
    """Read the exact selected product, order, and linked NIST geometry."""
    manifest = read_json(reference_path(product_file, root))
    if len(manifest) != 1:
        raise ValueError("Select a manifest containing one product")
    product, meta = next(iter(manifest.items()))
    if not isinstance(meta, dict):
        raise ValueError("Invalid product manifest")
    geometry_file = meta.get("product_geometry_file")
    geometry = read_json(reference_path(geometry_file, root))["gazebo"]
    order = read_json(reference_path(order_file, root))
    if order.get("product") != product or order.get("product_jid") != meta.get("jid"):
        raise ValueError("Product Order does not match the selected product exactly")
    if type(order.get("quantity")) is not int or order["quantity"] != 1:
        raise ValueError("recovery-framework settings support quantity: 1")
    parts = order.get("parts", "all")
    slots = geometry["assembly_board"]["slots"]
    if parts != "all" and (
        not isinstance(parts, list)
        or not parts
        or any(not isinstance(part, str) or part not in slots for part in parts)
        or len(set(parts)) != len(parts)
    ):
        raise ValueError("Product Order must use exact, distinct configured NIST part names")
    checked = validate_product_order(order, geometry)
    model_map = geometry["parts"]["model_map"]
    if any(part not in model_map for part in checked.selected_parts):
        raise ValueError("Selected NIST components need exact Gazebo model bindings")
    return {
        "product": product,
        "product_geometry_file": geometry_file,
        "geometry": geometry,
        "product_order": order,
        "selected_parts": checked.selected_parts,
    }


def default_setup(root: Path = ROOT) -> dict[str, Any]:
    """Build unsaved defaults from the current files, without writing them."""
    product_file = str(PRODUCT_PATH.relative_to(ROOT))
    scene_file = str(SCENE_PATH.relative_to(ROOT))
    manifest = read_json(root / product_file)
    meta = next(iter(manifest.values()))
    from cais_spade_llm.resources.environment_models import build_environment_models

    models = build_environment_models(read_json(root / scene_file))
    return {
        "schema_version": 1,
        "selected_product": product_file,
        "selected_product_order_file": meta["product_order_file"],
        "product_geometry_file": meta["product_geometry_file"],
        "scene_file": scene_file,
        "permitted_resources": list(models),
        "selected_safety_file": "cais_spade_llm/specification/safety/safety_none.txt",
        "execution_mode": "simulation",
        "runtime_recovery_mode": "pre_ran",
        "runtime_recovery_validation_policy": "validated",
        "runtime_recovery_archive_path": "",
        "runtime_recovery_archive_label": "",
        "recovery_experiment_settings_file": EXPERIMENT_PATH,
        "recovery_experiment_settings": read_json(root / EXPERIMENT_PATH),
        "failure_scenario": None,
    }


def load_setup(path: Path | None = None, *, root: Path = ROOT) -> dict[str, Any]:
    """Read saved settings, or return defaults only when no setup file exists."""
    path = path if path is not None else SETUP_PATH
    setup = read_json(path) if path.exists() else default_setup(root)
    string_fields = (
        "selected_product",
        "selected_product_order_file",
        "product_geometry_file",
        "scene_file",
        "selected_safety_file",
        "execution_mode",
        "runtime_recovery_mode",
        "runtime_recovery_validation_policy",
        "runtime_recovery_archive_path",
        "runtime_recovery_archive_label",
        "recovery_experiment_settings_file",
    )
    if setup.get("schema_version") != 1 or any(
        not isinstance(setup.get(field), str) for field in string_fields
    ):
        raise ValueError("Invalid recovery-framework setup fields or schema_version")
    if not isinstance(setup.get("permitted_resources"), list) or any(
        not isinstance(rid, str) for rid in setup["permitted_resources"]
    ):
        raise ValueError("Invalid permitted_resources")
    if (
        not isinstance(setup.get("recovery_experiment_settings"), dict)
        or "failure_scenario" not in setup
    ):
        raise ValueError("Invalid experiment or failure scenario settings")
    failure = setup["failure_scenario"]
    if failure is not None and (
        not isinstance(failure, dict) or failure.get("scenario") not in FAILURE_SCENARIOS
    ):
        raise ValueError("Unknown failure scenario")
    if failure is not None:
        _validate_failure_fields(failure)
    return setup


def _validate_failure_fields(failure: dict) -> None:
    if any(not isinstance(failure.get(key), str) for key in ("resource_id", "checkpoint", "mode")):
        raise ValueError("Invalid failure resource, checkpoint, or occurrence")
    if failure["scenario"] != "Part slippage":
        return
    if (
        any(not isinstance(failure.get(key), str) for key in ("part_name", "event_name"))
        or type(failure.get("event_id")) is not int
    ):
        raise ValueError("Invalid Part slippage part or task fields")
    if not isinstance(failure.get("parameter_bindings"), dict):
        raise ValueError("Invalid Part slippage parameter_bindings")
    _validate_drop_pose(failure)
    condition = failure.get("additional_condition")
    if condition is not None and (
        not isinstance(condition, dict)
        or any(not isinstance(condition.get(key), str) for key in ("resource_id", "part_name"))
    ):
        raise ValueError("Invalid additional condition")


def eligible_parts(model: dict, selected_parts: list[str]) -> list[str]:
    """Return selected components in this manipulator's exact held_part domain."""
    declaration = model["state_variables"].get("held_part", {})
    if declaration.get("reference") == "part_name":
        return list(selected_parts)
    domain = declaration.get("domain", [])
    return [part for part in selected_parts if part in domain]


def part_task_events(models: dict, resource_id: str, part_name: str) -> list[dict]:
    """Read applicable actor-owned capability alternatives for one exact part."""
    result = []
    for event in models[resource_id]["events"]:
        bindings = event["parameter_bindings"]
        if bindings["resource_id"].get("equals") != resource_id:
            continue
        rule = bindings.get("part_name", {})
        if "equals" in rule:
            allowed = rule["equals"] == part_name
        elif "from_assignment" in rule:
            owner = rule["from_assignment"]
            allowed = part_name in models[owner["resource_id"]]["assignments"][owner["field"]]
        elif rule.get("reference") == "part_name":
            allowed = True
        else:
            allowed = False
        if allowed:
            result.append(event)
    return result


def task_label(event: dict) -> str:
    """Describe an exact task alternative using its configured parameters."""
    parameters = [
        f"{name}={rule['equals']}"
        for name, rule in event["parameter_bindings"].items()
        if "equals" in rule
        and name
        in {
            "origin_resource_location",
            "destination_location",
            "target_resource",
            "loading_position",
            "source_resource",
        }
    ]
    return f"{event['event_name']}({', '.join(parameters)})"


def _validate_drop_pose(failure: dict) -> None:
    for field, keys in (
        ("drop_pose", ("x", "y", "z")),
        ("orientation_quat", ("qx", "qy", "qz", "qw")),
    ):
        values = failure.get(field)
        if not isinstance(values, dict) or set(values) != set(keys):
            raise ValueError(f"Part slippage requires {field}: {', '.join(keys)}")
        if any(
            type(value) not in {int, float} or not math.isfinite(value) for value in values.values()
        ):
            raise ValueError(f"{field} must contain finite numbers")
    norm = sum(value * value for value in failure["orientation_quat"].values())
    if not math.isclose(norm, 1.0, abs_tol=1e-6):
        raise ValueError(
            "orientation_quat must have unit length; values are not adjusted automatically"
        )


def _validate_failure(failure: Any, models: dict, parts: list[str], permitted: list[str]) -> None:
    if failure is None:
        return
    if not isinstance(failure, dict) or failure.get("scenario") not in FAILURE_SCENARIOS:
        raise ValueError("Unknown failure scenario")
    rid = failure.get("resource_id")
    if rid not in permitted:
        raise ValueError("Failure resource must be permitted for this experiment")
    if failure.get("mode") != "once":
        raise ValueError("Failure occurrence must be once per run")
    if failure.get("checkpoint") not in CHECKPOINTS:
        raise ValueError("Unknown failure checkpoint")
    scenario = failure["scenario"]
    if scenario == "Conveyor breakdown" and rid != "Conveyor":
        raise ValueError("Conveyor breakdown requires Conveyor")
    if scenario == "ur5e-1 breakdown" and rid != "ur5e-1":
        raise ValueError("ur5e-1 breakdown requires ur5e-1")
    if scenario == "Machining breakdown during part processing" and not any(
        event["event_name"] == "machine_part" for event in models[rid]["events"]
    ):
        raise ValueError("Machining breakdown requires a configured machining resource")
    if scenario != "Part slippage":
        return
    part = failure.get("part_name")
    if part not in eligible_parts(models[rid], parts):
        raise ValueError(
            "Part slippage requires an exact selected NIST part eligible for this resource"
        )
    events = part_task_events(models, rid, part)
    if not any(
        type(failure.get("event_id")) is int
        and event["event_id"] == failure["event_id"]
        and event["event_name"] == failure.get("event_name")
        and event["parameter_bindings"] == failure.get("parameter_bindings")
        for event in events
    ):
        raise ValueError("Part slippage task bindings do not match the configured capability")
    _validate_drop_pose(failure)
    _validate_additional_condition(failure, models, parts, permitted)


def _validate_additional_condition(
    failure: dict, models: dict, parts: list[str], permitted: list[str]
) -> None:
    condition = failure.get("additional_condition")
    if condition is not None:
        if not isinstance(condition, dict):
            raise ValueError("Invalid additional condition")
        other = condition.get("resource_id")
        other_part = condition.get("part_name")
        if other not in permitted or other == failure["resource_id"]:
            raise ValueError("Additional condition requires another permitted resource")
        if other_part == failure["part_name"] or other_part not in eligible_parts(
            models[other], parts
        ):
            raise ValueError(
                "Additional condition requires another exact selected part eligible for that resource"
            )


def validate_setup(setup: dict, *, root: Path = ROOT) -> dict[str, Any]:
    """Validate settings against their source definitions without projecting events."""
    if setup.get("schema_version") != 1:
        raise ValueError("Unsupported recovery-framework setup schema_version")
    inputs = product_inputs(setup["selected_product"], setup["selected_product_order_file"], root)
    if setup.get("product_geometry_file") != inputs["product_geometry_file"]:
        raise ValueError("Geometry reference does not match the selected product manifest")
    scene = read_json(reference_path(setup["scene_file"], root))
    if "completion_conditions" in inputs["product_order"]:
        models = build_nominal_resource_des_models(scene)
    else:
        from cais_spade_llm.product.environment import EnvironmentProductContext

        models = EnvironmentProductContext(scene, inputs["product_order"], inputs["geometry"],
                                           setup.get("permitted_resources")).models
    if "completion_conditions" in inputs["product_order"]:
        validate_completion_conditions(inputs["product_order"]["completion_conditions"], models)
    parts = inputs["selected_parts"]
    known_parts = list(inputs["geometry"]["assembly_board"]["slots"])
    if any(part not in known_parts for part in parts):
        raise ValueError("Product Order contains parts outside the configured NIST inventory")
    permitted = setup.get("permitted_resources")
    if (
        not isinstance(permitted, list)
        or any(not isinstance(rid, str) or rid not in models for rid in permitted)
        or len(set(permitted)) != len(permitted)
    ):
        raise ValueError("Permitted resources must be distinct exact configured resource IDs")
    safety = setup.get("selected_safety_file")
    if safety != "__NONE__":
        reference_path(safety, root).read_text(encoding="utf-8")
    for field, options in (
        ("execution_mode", MODE_OPTIONS),
        ("runtime_recovery_mode", RECOVERY_MODE_OPTIONS),
        ("runtime_recovery_validation_policy", RECOVERY_VALIDATION_OPTIONS),
    ):
        if setup.get(field) not in options:
            raise ValueError(f"Unsupported {field}")
    archive = setup.get("runtime_recovery_archive_path")
    if not isinstance(archive, str) or not isinstance(
        setup.get("runtime_recovery_archive_label"), str
    ):
        raise ValueError("Invalid Archived Recovery Run reference")
    if archive and not reference_path(archive, root).is_file():
        raise ValueError("Archived Recovery Run file does not exist")
    experiment = read_json(reference_path(setup["recovery_experiment_settings_file"], root))
    if setup.get("recovery_experiment_settings") != experiment:
        raise ValueError(
            "Recovery experiment settings changed; reload their current values before saving"
        )
    _validate_failure(setup.get("failure_scenario"), models, parts, permitted)
    return {**inputs, "scene": scene, "models": models}


def save_setup(setup: dict, path: Path | None = None, *, root: Path = ROOT) -> None:
    """Validate and atomically save only the separate experiment setup file."""
    validate_setup(setup, root=root)
    path = path if path is not None else SETUP_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(setup, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def startup_block_reason(setup: dict, *, root: Path = ROOT) -> str:
    """Reject settings the existing runtime cannot honor, before calling Start System."""
    try:
        inputs = validate_setup(setup, root=root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return f"Invalid experiment setup: {exc}. Open setup to correct it."
    reasons = []
    if "completion_conditions" in inputs["product_order"]:
        from cais_spade_llm.recovery_framework.delivery import is_delivery_order

        if setup["execution_mode"] != "simulation" or not is_delivery_order(inputs["product_order"]):
            reasons.append("Selected completion_conditions have no integrated execution path")
    if setup["failure_scenario"] is not None:
        reasons.append(f"{setup['failure_scenario']['scenario']}: execution not integrated")
    excluded = [rid for rid in inputs["models"] if rid not in setup["permitted_resources"]]
    if excluded and "completion_conditions" in inputs["product_order"]:
        reasons.append("Resource restrictions are not integrated: " + ", ".join(excluded))
    if (
        reference_path(setup["scene_file"], root).resolve()
        != (root / SCENE_PATH.relative_to(ROOT)).resolve()
    ):
        reasons.append("Alternative plant configuration execution is not integrated")
    return "; ".join(reasons)


class StartupValidation:
    """Cache display validation while retaining an unconditional check on Start."""

    def __init__(self, *, root: Path = ROOT) -> None:
        self.root = root
        self._signature: tuple | None = None
        self._result: tuple[dict, str, bool] = ({}, "Checking saved setup...", False)

    def _dependencies(self, setup: dict) -> tuple:
        references = [SETUP_PATH]
        for field in (
            "selected_product",
            "selected_product_order_file",
            "product_geometry_file",
            "scene_file",
            "selected_safety_file",
            "recovery_experiment_settings_file",
            "runtime_recovery_archive_path",
        ):
            value = setup.get(field)
            if value and value != "__NONE__":
                references.append(reference_path(value, self.root))
        # Follow the manifest too: a changed link must invalidate the display cache.
        product = setup.get("selected_product")
        if product:
            manifest = read_json(reference_path(product, self.root))
            for meta in manifest.values():
                if isinstance(meta, dict) and meta.get("product_geometry_file"):
                    references.append(reference_path(meta["product_geometry_file"], self.root))
        stamps = []
        for path in references:
            try:
                stat = path.stat()
                stamps.append((str(path), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size))
            except FileNotFoundError:
                stamps.append((str(path), None))
        return json.dumps(setup, sort_keys=True), tuple(stamps)

    def read(self, *, force: bool = False) -> tuple[dict, str, bool]:
        """Read the saved setup and validate changed dependencies on a worker thread."""
        from cais_spade_llm.recovery_framework.delivery import is_delivery_order

        try:
            setup = load_setup(root=self.root)
            signature = self._dependencies(setup)
            if force or signature != self._signature:
                reason = startup_block_reason(setup, root=self.root)
                order = read_json(reference_path(setup["selected_product_order_file"], self.root))
                delivery = setup["execution_mode"] == "simulation" and is_delivery_order(order)
                if signature != self._dependencies(setup):
                    return (
                        setup,
                        "Configuration changed during validation. Checking again...",
                        False,
                    )
                self._result = (setup, reason, delivery)
                self._signature = signature
            return deepcopy(self._result)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self._signature = None
            return {}, f"Experiment setup could not be loaded: {exc}", False


def setup_summary(setup: dict) -> list[dict[str, str]]:
    """Describe saved inputs without mixing them with live state or run evidence."""
    rows = [
        ("Product", setup.get("selected_product")),
        ("Product Order", setup.get("selected_product_order_file")),
        ("Product Geometry", setup.get("product_geometry_file")),
        ("Plant configuration", setup.get("scene_file")),
        ("Permitted resources", ", ".join(setup.get("permitted_resources", []))),
        ("Safety", setup.get("selected_safety_file")),
        ("Mode", MODE_OPTIONS.get(setup.get("execution_mode"), setup.get("execution_mode"))),
        ("Recovery Handoff Mode", setup.get("runtime_recovery_mode")),
        ("Recovery Safety Mode", setup.get("runtime_recovery_validation_policy")),
        ("Archived Recovery Run", setup.get("runtime_recovery_archive_path") or "None"),
        ("Failure scenario", (setup.get("failure_scenario") or {}).get("scenario", "None")),
    ]
    return [{"setting": key, "value": str(value)} for key, value in rows]


def slippage_example(models: dict, resource_id: str, part_name: str) -> dict:
    """Build a draft example with no fabricated drop pose or runtime observations."""
    event = next(
        event
        for event in part_task_events(models, resource_id, part_name)
        if event["event_name"] == "place_insert"
    )
    return {
        "scenario": "Part slippage",
        "resource_id": resource_id,
        "part_name": part_name,
        "event_id": event["event_id"],
        "event_name": event["event_name"],
        "parameter_bindings": deepcopy(event["parameter_bindings"]),
        "checkpoint": "after_execute_before_commit",
        "mode": "once",
        "drop_pose": {axis: None for axis in ("x", "y", "z")},
        "orientation_quat": {"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0},
        "additional_condition": None,
    }
