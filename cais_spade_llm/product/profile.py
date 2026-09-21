"""Read-only product-domain profile helpers and place-target resolution."""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import InitVar, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from cais_spade_llm.product.order import load_product_order_file
from cais_spade_llm.product.stl_geometry import actual_mg_stl_geometry_for_part

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_GAZEBO_WORLD_PATH = _REPO_ROOT / "ros2" / "cais_lab_robotics" / "worlds" / "table_recovery_framework.world"


@dataclass(frozen=True)
class ProductProfile:
    """Static product configuration and read-only product-domain helpers."""

    name: str
    product_specification_file: str | None = None
    product_order_file: str | None = None
    product_geometry_file: str | None = None
    safety_file: str | None = None
    instruction_override: str | None = None
    precomputed_bundle: dict[str, Any] | None = None
    robot_env: str = ""
    product_geometry: dict[str, Any] = field(default_factory=dict)
    logger: InitVar[Any | None] = None

    def __post_init__(self, logger: Any | None) -> None:
        robot_env = str(self.robot_env or os.environ.get("ROBOT_ENV", "gazebo")).strip().lower()
        if not robot_env:
            robot_env = "gazebo"
        object.__setattr__(self, "robot_env", robot_env)
        object.__setattr__(self, "precomputed_bundle", dict(self.precomputed_bundle or {}))

        geometry = dict(self.product_geometry or {})
        if not geometry and self.product_geometry_file:
            geometry = self.load_product_geometry(
                self.product_geometry_file,
                robot_env=robot_env,
                logger=logger,
            )
        object.__setattr__(self, "product_geometry", geometry)

    def read_safety_text(self, *, logger: Any | None = None) -> str:
        """Read the NL safety file if provided. Returns empty string if missing."""
        return self.read_safety_file(self.safety_file, logger=logger)

    @staticmethod
    def read_safety_file(
        safety_file: str | None,
        *,
        logger: Any | None = None,
    ) -> str:
        if not safety_file:
            return ""

        try:
            path = Path(safety_file)
            if path.exists():
                text = path.read_text(encoding="utf-8").strip()
                if logger is not None:
                    logger.info(f"[Product] Loaded safety constraints from {path}")
                return text
            if logger is not None:
                logger.warning(f"[Product] Safety file path provided but not found: {path}")
        except Exception as exc:
            if logger is not None:
                logger.exception(f"[Product] Failed to read safety file: {exc}")
        return ""

    def read_spec_text(self, *, logger: Any | None = None) -> str | None:
        """Return instruction text: prefer explicit override, then spec file."""
        return self.read_spec_file(
            self.product_specification_file,
            instruction_override=self.instruction_override,
            logger=logger,
        )

    @staticmethod
    def read_spec_file(
        product_specification_file: str | None,
        *,
        instruction_override: str | None = None,
        logger: Any | None = None,
    ) -> str | None:
        if instruction_override:
            text = instruction_override.strip()
            if text:
                return text

        if product_specification_file:
            try:
                if logger is not None:
                    logger.debug(f"[Product] Current working directory: {os.getcwd()}")
                path = Path(product_specification_file)
                if logger is not None:
                    logger.debug(f"[Product] Attempting to open: {path.resolve()}")
                text = path.read_text(encoding="utf-8").strip()
                if text:
                    return text
                if logger is not None:
                    logger.warning(f"[Product] Spec file is empty: {path}")
            except Exception as exc:
                if logger is not None:
                    logger.exception(f"[Product] Failed to read spec: {exc}")
        return None

    def extract_requirement_text(self, *, logger: Any | None = None) -> str | None:
        """Load requirement text from the configured product specification file."""
        return self.extract_requirement_file(self.product_specification_file, logger=logger)

    def read_product_order(self, *, logger: Any | None = None) -> dict[str, Any] | None:
        """Load product-order JSON if configured."""
        return self.read_product_order_file(self.product_order_file, logger=logger)

    @staticmethod
    def read_product_order_file(
        product_order_file: str | None,
        *,
        logger: Any | None = None,
    ) -> dict[str, Any] | None:
        if not product_order_file:
            return None
        try:
            path = Path(product_order_file)
            if not path.exists():
                if logger is not None:
                    logger.warning("[Product] Product order file not found: %s", path)
                return None
            payload = load_product_order_file(path)
            if logger is not None:
                logger.info("[Product] Using product order file: %s", path.resolve())
            return payload
        except Exception as exc:
            if logger is not None:
                logger.exception(
                    "[Product] Failed to read product order from %s: %s",
                    product_order_file,
                    exc,
                )
            raise

    @staticmethod
    def extract_requirement_file(
        product_specification_file: str | None,
        *,
        logger: Any | None = None,
    ) -> str | None:
        if not product_specification_file:
            return None

        try:
            path = Path(product_specification_file)
            if not path.exists():
                return None
            text = path.read_text(encoding="utf-8").strip()
            if text:
                if logger is not None:
                    logger.info(f"[Product] Using requirement file: {path.resolve()}")
                return text
            if logger is not None:
                logger.warning(f"[Product] Requirement file empty: {path}")
        except Exception as exc:
            if logger is not None:
                logger.exception(
                    "[Product] Failed to read requirements from %s: %s",
                    product_specification_file,
                    exc,
                )
        return None

    @staticmethod
    def load_product_geometry(
        geometry_file: str | None,
        *,
        robot_env: str | None = None,
        logger: Any | None = None,
    ) -> dict[str, Any]:
        """Load product geometry JSON, selecting the configured environment block."""
        if not geometry_file:
            return {}

        env = str(robot_env or os.environ.get("ROBOT_ENV", "gazebo")).strip().lower()
        if not env:
            env = "gazebo"
        try:
            path = Path(geometry_file)
            if not path.exists():
                if logger is not None:
                    logger.warning("[Product] Geometry file not found: %s", path)
                return {}
            raw = json.loads(path.read_text(encoding="utf-8"))
            geometry = raw.get(env, {})
            if geometry:
                if logger is not None:
                    logger.info("[Product] Loaded geometry for env='%s' from %s", env, path)
                return dict(geometry)

            if logger is not None:
                logger.warning("[Product] Geometry file %s has no usable '%s' block.", path, env)
        except Exception:
            if logger is not None:
                logger.exception("[Product] Failed to load geometry from %s", geometry_file)
        return {}

    def geometry_for_part(
        self,
        part_name: str,
        *,
        product_geometry: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Extract placement geometry for a single part from loaded product geometry."""
        geometry = self.product_geometry if product_geometry is None else product_geometry
        return self.geometry_for_part_from_geometry(part_name, geometry)

    @staticmethod
    def geometry_for_part_from_geometry(
        part_name: str,
        product_geometry: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not product_geometry:
            return {}
        board = product_geometry.get("assembly_board", {})
        parts = product_geometry.get("parts", {})
        slot_xy = board.get("slots", {}).get(part_name)
        if slot_xy is None:
            return {}
        geometry = {
            "part_name": part_name,
            "slot_xy": deepcopy(slot_xy),
            "part_height_m": parts.get("heights_m", {}).get(part_name),
            "model_name": parts.get("model_map", {}).get(part_name),
            "slot_floor_z_m": board.get("slot_floor_z_m"),
            "board_center": deepcopy(board.get("center", {})),
            "assembly_board-v1_aruco_to_assembly_board-v1": deepcopy(
                board.get("assembly_board-v1_aruco_to_assembly_board-v1", {})
            ),
            "target_reference": deepcopy(product_geometry.get("target_reference", {})),
        }
        geometry.update(actual_mg_stl_geometry_for_part(part_name, parts))
        return geometry

    @staticmethod
    def has_place_geometry_fields(geometry: Mapping[str, Any] | None) -> bool:
        """Return True when the dict already contains enough place-target geometry."""
        if not isinstance(geometry, Mapping):
            return False
        slot_xy = geometry.get("slot_xy")
        board_center = geometry.get("board_center")
        slot_floor_z = geometry.get("slot_floor_z_m")
        return (
            isinstance(slot_xy, (list, tuple))
            and len(slot_xy) >= 2
            and isinstance(board_center, Mapping)
            and slot_floor_z is not None
        )

    @staticmethod
    def destination_token_from_place_inputs(
        *,
        destination_location: str = "",
        product_geometry: Mapping[str, Any] | None = None,
    ) -> str:
        """Extract a symbolic destination token from primitive params."""
        direct_token = _normalize_destination_token(destination_location)
        if direct_token:
            return direct_token

        geometry = dict(product_geometry) if isinstance(product_geometry, Mapping) else {}
        for key in ("destination_location", "destination", "location", "fixture_name"):
            token = _normalize_destination_token(geometry.get(key))
            if token:
                return token
        return ""

    @classmethod
    def resolve_place_geometry(
        cls,
        *,
        part_name: str,
        destination_location: str = "",
        product_geometry: Mapping[str, Any] | None = None,
        execution_mode: str = "simulation",
    ) -> dict[str, Any]:
        """Return explicit place geometry, resolving symbolic destinations when possible."""
        geometry = dict(product_geometry) if isinstance(product_geometry, Mapping) else {}
        if cls.has_place_geometry_fields(geometry):
            return geometry

        token = cls.destination_token_from_place_inputs(
            destination_location=destination_location,
            product_geometry=geometry,
        )
        if not token or not str(part_name or "").strip():
            return geometry

        resolved = _load_geometry_for_destination_part(
            token=token,
            part_name=str(part_name or "").strip(),
            execution_mode=execution_mode,
        )
        if not resolved:
            return geometry

        merged = dict(resolved)
        merged.update(geometry)
        return _apply_target_reference_pose(
            geometry=merged,
            execution_mode=execution_mode,
        )


def _load_geometry_for_destination_part(
    *,
    token: str,
    part_name: str,
    execution_mode: str,
) -> dict[str, Any]:
    geometry_doc = _load_geometry_doc_for_destination(
        token=token,
        execution_mode=execution_mode,
    )
    if not geometry_doc:
        return {}

    board = dict(geometry_doc.get("assembly_board") or {})
    parts = dict(geometry_doc.get("parts") or {})
    slot_xy = dict(board.get("slots") or {}).get(part_name)
    if not isinstance(slot_xy, (list, tuple)) or len(slot_xy) < 2:
        return {}

    return {
        "slot_xy": list(slot_xy[:2]),
        "part_height_m": dict(parts.get("heights_m") or {}).get(part_name),
        "model_name": dict(parts.get("model_map") or {}).get(part_name),
        "slot_floor_z_m": board.get("slot_floor_z_m"),
        "board_center": board.get("center") or {},
        "assembly_board-v1_aruco_to_assembly_board-v1": dict(
            board.get("assembly_board-v1_aruco_to_assembly_board-v1") or {}
        ),
        "target_reference": dict(geometry_doc.get("target_reference") or {}),
    }


def _apply_target_reference_pose(
    *,
    geometry: Mapping[str, Any],
    execution_mode: str,
) -> dict[str, Any]:
    """Apply environment-backed target reference metadata when available."""
    resolved = dict(geometry)
    if str(execution_mode or "").strip().lower() == "physical":
        return resolved

    target_reference = dict(resolved.get("target_reference") or {})
    if str(target_reference.get("source") or "").strip() != "gazebo_model_spawn_pose":
        return resolved

    model_name = str(resolved.get("model_name") or "").strip()
    if not model_name:
        return resolved

    spawn_pose = _load_gazebo_model_spawn_pose(
        model_name=model_name,
        world_path=str(_gazebo_world_path()),
    )
    if not spawn_pose:
        return resolved

    board_center = dict(resolved.get("board_center") or {})
    try:
        center_x = float(board_center.get("x", 0.0))
        center_y = float(board_center.get("y", 0.0))
        spawn_x = float(spawn_pose["x"])
        spawn_y = float(spawn_pose["y"])
    except (KeyError, TypeError, ValueError):
        return resolved

    resolved["board_center"] = board_center
    resolved["slot_xy"] = [spawn_x - center_x, spawn_y - center_y]
    resolved["target_origin_pose"] = dict(spawn_pose)
    return resolved


def _gazebo_world_path() -> Path:
    override = os.environ.get("CAIS_GAZEBO_WORLD_PATH", "").strip()
    if override:
        return Path(override).expanduser()
    return _DEFAULT_GAZEBO_WORLD_PATH


@lru_cache(maxsize=64)
def _load_gazebo_model_spawn_pose(
    *,
    model_name: str,
    world_path: str,
) -> dict[str, float]:
    path = Path(world_path)
    if not model_name or not path.exists():
        return {}
    try:
        root = ET.parse(str(path)).getroot()
    except Exception:
        return {}

    for model in root.iter():
        if _xml_local_name(model.tag) != "model":
            continue
        if str(model.attrib.get("name") or "").strip() != model_name:
            continue
        pose_text = ""
        for child in list(model):
            if _xml_local_name(child.tag) == "pose":
                pose_text = child.text or ""
                break
        values = pose_text.split()
        if len(values) < 3:
            return {}
        try:
            return {
                "x": float(values[0]),
                "y": float(values[1]),
                "z": float(values[2]),
            }
        except ValueError:
            return {}
    return {}


def _xml_local_name(tag: str) -> str:
    return str(tag or "").rsplit("}", 1)[-1]


@lru_cache(maxsize=32)
def _load_geometry_doc_for_destination(
    *,
    token: str,
    execution_mode: str,
) -> dict[str, Any]:
    for geometry_path in (
        _product_geometry_path_for_token(token),
        _direct_geometry_path_for_token(token),
    ):
        if geometry_path is None or not geometry_path.exists():
            continue
        try:
            payload = json.loads(geometry_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        env_key = "real" if str(execution_mode or "").strip().lower() == "physical" else "gazebo"
        geometry_doc = payload.get(env_key)
        if not isinstance(geometry_doc, dict):
            geometry_doc = payload
        if isinstance(geometry_doc, dict):
            return dict(geometry_doc)
    return {}


def _product_geometry_path_for_token(token: str) -> Path | None:
    product_meta = _load_product_meta(token)
    if not product_meta:
        return None
    geometry_relpath = str(product_meta.get("product_geometry_file") or "").strip()
    if not geometry_relpath:
        return None
    return _REPO_ROOT / geometry_relpath


def _direct_geometry_path_for_token(token: str) -> Path:
    # Printer/support destinations may have simulation/real geometry fallbacks
    # without being registered as full product manifests.
    return _PACKAGE_ROOT / "specification" / "products" / "geometry" / f"{token}.json"


@lru_cache(maxsize=16)
def _load_product_meta(token: str) -> dict[str, Any]:
    """Resolve one exact product identifier independently of its manifest filename."""
    init_dir = _PACKAGE_ROOT / "initialization" / "products"
    matches = []
    for init_path in sorted(init_dir.glob("*.json")):
        try:
            payload = json.loads(init_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get(token), dict):
            matches.append(payload[token])
    return dict(matches[0]) if len(matches) == 1 else {}


def _normalize_destination_token(value: Any) -> str:
    token = str(value or "").strip()
    if not token:
        return ""
    if "@" in token:
        token = token.split("@", 1)[0].strip()
    return token
