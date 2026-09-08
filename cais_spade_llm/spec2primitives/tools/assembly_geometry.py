from __future__ import annotations

"""Measure candidate assembly geometry without choosing task roles or robot steps."""

import math
import re
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..adapters.robot_validation_context import matrix_pose, pose_matrix
from ..agents.ra.refinement_records import append_record, fingerprint, owned_path, read_pin
from .rgb_d_cad_grounding.size_correspondence import _load_cad_input


def planar_features(triangles: np.ndarray) -> list[dict[str, Any]]:
    """Find planar face groups and circular boundary candidates in CAD-local metres."""
    edges = triangles[:, 1:] - triangles[:, :1]
    cross = np.cross(edges[:, 0], edges[:, 1])
    lengths = np.linalg.norm(cross, axis=1)
    groups: dict[tuple[float, ...], list[int]] = defaultdict(list)
    for index in np.flatnonzero(lengths > 1e-12):
        normal = cross[index] / lengths[index]
        key = (*np.round(normal, 5), round(float(normal @ triangles[index, 0]), 6))
        groups[key].append(int(index))
    features = []
    for key, indices in sorted(groups.items()):
        area = float(lengths[indices].sum() * 0.5)
        if area < 1e-6:
            continue
        faces = triangles[indices]
        normal = np.asarray(key[:3], dtype=float)
        normal /= np.linalg.norm(normal)
        point = faces.reshape(-1, 3).mean(axis=0)
        counts: Counter[Any] = Counter()
        for face in faces:
            vertices = [tuple(np.round(vertex, 7)) for vertex in face]
            for i in range(3):
                counts[tuple(sorted((vertices[i], vertices[(i + 1) % 3])))] += 1
        adjacency: dict[Any, set[Any]] = defaultdict(set)
        for (first, second), count in counts.items():
            if count == 1:
                adjacency[first].add(second)
                adjacency[second].add(first)
        circles = []
        visited = set()
        for vertex in sorted(adjacency):
            if vertex in visited:
                continue
            pending, component = [vertex], set()
            while pending:
                item = pending.pop()
                if item not in component:
                    component.add(item)
                    pending.extend(adjacency[item] - component)
            visited.update(component)
            if len(component) < 8 or any(len(adjacency[item]) != 2 for item in component):
                continue
            points = np.asarray(sorted(component))
            u = points[0] - points.mean(axis=0)
            u -= normal * float(u @ normal)
            if np.linalg.norm(u) < 1e-12:
                continue
            u /= np.linalg.norm(u)
            v = np.cross(normal, u)
            xy = np.column_stack((points @ u, points @ v))
            solution, _, rank, _ = np.linalg.lstsq(
                np.column_stack((2 * xy, np.ones(len(xy)))), np.sum(xy**2, axis=1), rcond=None
            )
            if rank != 3:
                continue
            center = solution[:2]
            radius_squared = float(solution[2] + center @ center)
            if radius_squared <= 0:
                continue
            radius = math.sqrt(radius_squared)
            residual = float(np.max(np.abs(np.linalg.norm(xy - center, axis=1) - radius)))
            if residual > min(0.0001, radius * 0.01):
                continue
            center3 = u * center[0] + v * center[1] + normal * float(point @ normal)
            circles.append(
                {
                    "circle_id": f"circle_{len(circles) + 1:04d}",
                    "center_m": center3.tolist(),
                    "radius_m": radius,
                    "fit_residual_m": residual,
                }
            )
        features.append(
            {
                "plane_id": f"plane_{len(features) + 1:04d}",
                "normal": normal.tolist(),
                "point_m": point.tolist(),
                "area_m2": area,
                "circles": circles,
            }
        )
    return features


class AssemblyGeometryProducer:
    """Consume exact approved records; persist neutral geometry or selected derivations."""

    def __init__(
        self,
        root: Path,
        directory: Path,
        authorized: dict[str, str],
        target_feature: Mapping[str, Any] | None = None,
    ) -> None:
        """Pin the producer to an interaction and the investigation's issued evidence."""
        self.root, self.directory, self.authorized = root, directory, authorized
        self.target_feature = dict(target_feature or {})

    def read(self, ref: str) -> dict[str, Any]:
        """Read only an issued record with its originally accepted bytes."""
        if ref not in self.authorized:
            raise ValueError("Geometry input was not issued to this investigation.")
        from ..agents.ra.refinement_records import verify_evidence_tree

        return verify_evidence_tree(self.root, {"ref": ref, "sha256": self.authorized[ref]})

    def save(self, payload: Mapping[str, Any], sources: list[str]) -> dict[str, Any]:
        """Append a measurement with complete direct source pins."""
        for ref in sources:
            self.read(ref)
        number = len(list(self.directory.glob("geometry_*.json"))) + 1
        reference = append_record(
            self.root,
            self.directory,
            f"geometry_{number:04d}.json",
            {
                **payload,
                "source_refs": [
                    {"ref": ref, "sha256": self.authorized[ref]} for ref in dict.fromkeys(sources)
                ],
                "created_at_ns": time.time_ns(),
            },
        )
        self.authorized[reference["ref"]] = reference["sha256"]
        return {
            "record_ref": reference["ref"],
            "record_sha256": reference["sha256"],
            "record": self.read(reference["ref"]),
        }

    def inspect_features(self, cad_ref: str, pose_ref: str) -> dict[str, Any]:
        """Measure oriented features or a height estimate from approved observed poses."""
        self.read(cad_ref)
        pose = self.read(pose_ref)
        if pose.get("record_type") != "RobotFramePoseRecord":
            # Report the interface dependency without selecting or invoking a tool for PA.
            raise ValueError(
                "inspect_features requires pose_ref to identify a RobotFramePoseRecord; "
                f"received {pose.get('record_type')!r}. estimate_pose returns a "
                "CADPoseEstimationRecord; invoke convert_pose on that record before inspect_features."
            )
        if pose.get("CAD", {}).get("record", {}).get("ref") != cad_ref:
            raise ValueError(
                "The selected RobotFramePoseRecord refers to a different CAD record than cad_ref."
            )
        cad = _load_cad_input(self.root, owned_path(self.root, cad_ref))
        frame_pose = pose.get("robot_frame_pose") or {}
        if pose.get("pose") != "accepted" or "robot_from_CAD_transform" not in frame_pose:
            record = {
                "record_type": "AssemblyGeometryEvidence",
                "status": "ambiguous",
                "pose_status": "ambiguous",
                "frame_id": pose.get("target_frame"),
                "units": "m",
                "message": "A unique CAD-origin pose is not established; no final pose or oriented feature is fabricated.",
            }
            from .rgb_d_cad_grounding.frame_conversion import _load_calibration

            source = read_pin(self.root, pose["source_pose"])
            calibration = _load_calibration(
                self.root,
                owned_path(self.root, pose["source_calibration"]["ref"]),
                observation_timestamp_ns=pose["observation_timestamp_ns"],
            )
            hypotheses = source.get("qualified_pose_hypotheses", [])
            selected = source.get("selected_candidate") or {}
            if selected.get("candidate_handle") and hypotheses and all(
                item["candidate_handle"] == selected.get("candidate_handle")
                and item["frame"] == calibration.source_frame
                and item["camera_id"] == selected.get("camera_id")
                for item in hypotheses
            ):
                # Pose estimation already ranks these by registration quality. Its
                # first qualified fit supplies an estimate, without resolving pose.
                hypothesis = hypotheses[0]
                transform = calibration.target_from_camera @ np.asarray(
                    hypothesis["camera_from_CAD_transform"]
                )
                matrix_pose(transform)
                height = float(np.ptp(cad.triangles_m.reshape(-1, 3) @ transform[2, :3]))
                record.update(
                    part_height_m=height,
                    cad_context_ref=cad.context_ref,
                    candidate_reference=pose["candidate_reference"],
                    observation_timestamp_ns=pose["observation_timestamp_ns"],
                    uncertainty={
                        "method": "highest_ranked_qualified_pose_hypothesis",
                        "source_pose": pose["source_pose"],
                        "hypothesis_index": 0,
                        "registration": hypothesis["registration"],
                        "complete_pose_established": False,
                    },
                    warning=(
                        "part_height_m is a world-vertical CAD extent estimate from the "
                        "highest-ranked qualified observed pose. Full pose and mating geometry "
                        "remain ambiguous; this estimate does not establish sensor accuracy."
                    ),
                )
            return self.save(record, [cad_ref, pose_ref])
        transform = np.asarray(frame_pose["robot_from_CAD_transform"], dtype=float)
        origin = matrix_pose(transform)
        vertices = cad.triangles_m.reshape(-1, 3) @ transform[:3, :3].T + transform[:3, 3]
        features = planar_features(cad.triangles_m)
        for feature in features:
            feature["world_normal"] = (transform[:3, :3] @ feature["normal"]).tolist()
            feature["world_point_m"] = (
                transform[:3, :3] @ feature["point_m"] + transform[:3, 3]
            ).tolist()
            for circle in feature["circles"]:
                circle["world_center_m"] = (
                    transform[:3, :3] @ circle["center_m"] + transform[:3, 3]
                ).tolist()
        return self.save(
            {
                "record_type": "AssemblyGeometryEvidence",
                "status": "accepted",
                "frame_id": pose["target_frame"],
                "units": "m",
                "reference_point": "CAD_origin",
                "origin_pose": origin,
                "world_from_CAD": transform.tolist(),
                "part_height_m": float(np.ptp(vertices[:, 2])),
                "bounds_m": {
                    "minimum": vertices.min(axis=0).tolist(),
                    "maximum": vertices.max(axis=0).tolist(),
                },
                "mesh": {"ref": cad.mesh_ref, "sha256": cad.mesh_sha256},
                "features": features,
                "cad_context_ref": cad.context_ref,
                "object_id": "observed_"
                + fingerprint(
                    {
                        "candidate": pose["candidate_reference"],
                        "segmentation": pose["segmentation"],
                        "CAD": cad.context_ref,
                    }
                )[:20],
                "segmentation_ref": pose["segmentation"]["record"]["ref"],
                "candidate_reference": pose["candidate_reference"],
                "observation_timestamp_ns": pose["observation_timestamp_ns"],
                "uncertainty": {"pose_record_ref": pose_ref, "numerical_feature_fit_only": True},
            },
            [cad_ref, pose_ref],
        )

    def bind_part(self, part_ref: str, feature_name: str) -> dict[str, Any]:
        """Associate measured geometry with an exact already-accepted assembly feature."""
        part = self.read(part_ref)
        matches = [
            (association, feature)
            for association in self.target_feature.get("assembly_feature_association", [])
            for feature in association["assembly_features"]
            if feature["name"] == feature_name
        ]
        if len(matches) != 1:
            raise ValueError(
                "Select one exact accepted assembly feature; a new identity needs the PA authority gate."
            )
        association, feature = matches[0]
        state = next(
            value
            for value in self.target_feature["resolved_state_values"]
            if value["state"] == feature["state_name"]
            and value["name"] == feature["state_value_name"]
        )
        if (
            part.get("status") != "accepted"
            or part.get("segmentation_ref") != state["value_ref"]["record_ref"]
            or part.get("candidate_reference", {}).get("candidate_handle")
            != state["resolved_value"].get("candidate_handle")
        ):
            raise ValueError(
                "The measured candidate differs from the accepted physical instance; return to the PA authority gate."
            )
        if part["cad_context_ref"] not in feature["owner"]["evidence_refs"]:
            raise ValueError(
                "The selected CAD geometry is not bound to the accepted feature owner."
            )
        payload = {
            key: value
            for key, value in part.items()
            if key not in {"fingerprint", "source_refs", "created_at_ns"}
        }
        payload.update(
            part_name=feature["owner"]["name"],
            accepted_feature_name=feature_name,
            assembly_association_sha256=fingerprint(association),
        )
        return self.save(payload, [part_ref])

    def observed_surface(
        self, segmentation_ref: str, observation_handle: str, calibration_ref: str
    ) -> dict[str, Any]:
        """Transform one observed support-plane candidate without assigning its role."""
        from .rgb_d_cad_grounding.frame_conversion import _load_calibration
        from .rgb_d_cad_grounding.size_correspondence import _load_candidates

        self.read(segmentation_ref)
        self.read(calibration_ref)
        _, segmentation, _ = _load_candidates(self.root, owned_path(self.root, segmentation_ref))
        camera = next(
            item
            for item in segmentation["cameras"]
            if item["observation_handle"] == observation_handle
        )
        plane = camera["support_plane"]
        if plane.get("status") != "detected":
            raise ValueError("No support-plane candidate was measured in this view.")
        preprocessing = read_pin(self.root, segmentation["source_record"])
        source_camera = next(
            item for item in preprocessing["cameras"] if item["camera_id"] == camera["camera_id"]
        )
        stamp = source_camera["depth_timestamp_ns"]
        calibration = _load_calibration(
            self.root, owned_path(self.root, calibration_ref), observation_timestamp_ns=stamp
        )
        if calibration.source_frame != camera["frame"]:
            raise ValueError("Surface and calibration frames disagree.")
        transform = calibration.target_from_camera
        normal = transform[:3, :3] @ np.asarray(plane["normal"])
        offset = float(plane["offset_m"] - normal @ transform[:3, 3])
        return self.save(
            {
                "record_type": "AssemblySurfaceEvidence",
                "status": "accepted",
                "frame_id": calibration.target_frame,
                "units": "m",
                "normal": normal.tolist(),
                "offset_m": offset,
                "rms_distance_m": plane["rms_distance_m"],
                "observation_timestamp_ns": stamp,
                "reference_point": "observed_plane",
            },
            [segmentation_ref, calibration_ref],
        )

    def pick_geometry(self, part_ref: str, surface_ref: str) -> dict[str, Any]:
        """Build helper geometry from PA-selected object and support evidence."""
        part, surface = self.read(part_ref), self.read(surface_ref)
        _same_frame(part, surface)
        if part.get("reference_point") != "CAD_origin":
            raise ValueError("An observed candidate centre is not an established CAD origin.")
        normal = np.asarray(surface["normal"])
        if abs(normal[2]) < 1 - 1e-5:
            raise ValueError(
                "The vertical helper cannot represent the selected inclined support plane."
            )
        origin = part["origin_pose"]
        support_z = (
            -(surface["offset_m"] + normal[0] * origin["x"] + normal[1] * origin["y"]) / normal[2]
        )
        return self.save(
            {
                "record_type": "AssemblyGeometryEvidence",
                "status": "accepted",
                "frame_id": part["frame_id"],
                "units": "m",
                "reference_point": "CAD_origin",
                "target_pose": {key: origin[key] for key in ("x", "y", "z")},
                "product_geometry": {
                    "board_center": {"z": float(support_z)},
                    "part_height_m": part["part_height_m"],
                },
                "observation_timestamp_ns": part["observation_timestamp_ns"],
                "uncertainty": {
                    "support_rms_m": surface["rms_distance_m"],
                    "pose_record_ref": part_ref,
                },
            },
            [part_ref, surface_ref],
        )

    def assembly_geometry(
        self,
        part_ref: str,
        part_plane_id: str,
        part_circle_id: str,
        target_ref: str,
        target_plane_id: str,
        target_circle_id: str,
    ) -> dict[str, Any]:
        """Align selected mating axes and seat the selected part plane on a target plane."""
        part, target = self.read(part_ref), self.read(target_ref)
        _same_frame(part, target)
        if not part.get("assembly_association_sha256") or part.get(
            "assembly_association_sha256"
        ) != target.get("assembly_association_sha256"):
            raise ValueError(
                "Both mating geometries must be bound to the same accepted assembly relationship."
            )
        part_plane = next(item for item in part["features"] if item["plane_id"] == part_plane_id)
        target_plane = next(
            item for item in target["features"] if item["plane_id"] == target_plane_id
        )
        part_circle = next(
            item for item in part_plane["circles"] if item["circle_id"] == part_circle_id
        )
        target_circle = next(
            item for item in target_plane["circles"] if item["circle_id"] == target_circle_id
        )
        if any(abs(plane["world_normal"][2]) < 1 - 1e-5 for plane in (part_plane, target_plane)):
            raise ValueError(
                "Selected mating features require an orientation operation outside this vertical derivation."
            )
        if np.dot(part_plane["world_normal"], target_plane["world_normal"]) > -1 + 1e-5:
            raise ValueError("Selected contact planes do not face one another.")
        matrix = pose_matrix(part["origin_pose"])
        part_axis_offset = matrix[:3, :3] @ part_circle["center_m"]
        part_seat_offset = matrix[:3, :3] @ part_plane["point_m"]
        matrix[0, 3] = target_circle["world_center_m"][0] - part_axis_offset[0]
        matrix[1, 3] = target_circle["world_center_m"][1] - part_axis_offset[1]
        matrix[2, 3] = target_plane["world_point_m"][2] - part_seat_offset[2]
        origin = matrix_pose(matrix)
        return self.save(
            {
                "record_type": "AssemblyGeometryEvidence",
                "status": "accepted",
                "frame_id": part["frame_id"],
                "units": "m",
                "reference_point": "final_CAD_origin",
                "target_origin_pose": origin,
                "insertion_axis": [0.0, 0.0, -1.0],
                "part_name": part["part_name"],
                "part_object_id": part["object_id"],
                "assembly_association_sha256": part["assembly_association_sha256"],
                "part_axis_local": part_plane["normal"],
                "product_geometry": {
                    "board_center": {"x": origin["x"], "y": origin["y"]},
                    "slot_xy": [0.0, 0.0],
                    "slot_floor_z_m": target_plane["world_point_m"][2],
                    "part_height_m": part["part_height_m"],
                    "target_reference": {
                        "target_point": "inserted_part_origin",
                        "surface_role": "assembly_slot",
                    },
                    "target_origin_pose": {key: origin[key] for key in ("x", "y", "z")},
                },
                "selected_features": {
                    "part_plane_id": part_plane_id,
                    "part_circle_id": part_circle_id,
                    "target_plane_id": target_plane_id,
                    "target_circle_id": target_circle_id,
                },
                "nominal_radial_clearance_m": part_circle["radius_m"] - target_circle["radius_m"],
                "tolerances": None,
                "uncertainty": {"part_pose_ref": part_ref, "target_pose_ref": target_ref},
                "observation_timestamp_ns": min(
                    part["observation_timestamp_ns"], target["observation_timestamp_ns"]
                ),
            },
            [part_ref, target_ref],
        )

    def scene_geometry(
        self, geometry_refs: list[str], surface_refs: list[str], segmentation_refs: list[str]
    ) -> dict[str, Any]:
        """Cover observed candidates with selected registered meshes and measured planes."""
        geometry = [self.read(ref) for ref in geometry_refs]
        surfaces = [self.read(ref) for ref in surface_refs]
        if not geometry or not surfaces or not segmentation_refs:
            raise ValueError(
                "Scene coverage needs observed object geometry, support surfaces and segmentation records."
            )
        _same_frame(*geometry, *surfaces)
        covered = {
            (item["segmentation_ref"], item["candidate_reference"]["candidate_handle"])
            for item in geometry
        }
        unresolved = []
        for ref in segmentation_refs:
            segmentation = self.read(ref)
            if segmentation.get("record_type") != "RGBDSegmentationRecord":
                raise ValueError("Scene coverage must cite actual segmentation evidence.")
            for camera in segmentation["cameras"]:
                for candidate in camera["candidates"]:
                    if (ref, candidate["candidate_handle"]) not in covered:
                        unresolved.append(
                            {
                                "segmentation_ref": ref,
                                "candidate_handle": candidate["candidate_handle"],
                            }
                        )
        objects = [
            {"object_id": item["object_id"], "mesh": item["mesh"], "pose": item["origin_pose"]}
            for item in geometry
        ]
        if len({item["object_id"] for item in objects}) != len(objects):
            raise ValueError(
                "Scene contains duplicate object geometry; no implicit instance fusion is performed."
            )
        identity_pose = {"x": 0.0, "y": 0.0, "z": 0.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}
        objects.extend(
            {
                "object_id": "surface_" + str(index),
                "plane": [*item["normal"], item["offset_m"]],
                "pose": identity_pose,
            }
            for index, item in enumerate(surfaces)
        )
        return self.save(
            {
                "record_type": "AssemblySceneEvidence",
                "status": "accepted" if not unresolved else "incomplete",
                "frame_id": geometry[0]["frame_id"],
                "units": "m",
                "objects": objects,
                "coverage": "all_observed_candidates" if not unresolved else "incomplete",
                "unresolved_candidates": unresolved,
                "unobserved_space": "unmodeled",
                "observation_timestamp_ns": min(
                    item["observation_timestamp_ns"] for item in [*geometry, *surfaces]
                ),
            },
            [*geometry_refs, *surface_refs, *segmentation_refs],
        )

    def document_quantity(
        self, record_ref: str, field_path: str, start: int, end: int, quantity: str
    ) -> dict[str, Any]:
        """Extract one explicitly unit-bearing numeric span from issued document evidence."""
        from ..agents.ra.composition_context import _resolve_json_pointer

        record = self.read(record_ref)
        if not str(record.get("record_type", "")).startswith("Document"):
            raise ValueError("A document quantity must cite issued document evidence.")
        text = _resolve_json_pointer(record, field_path)
        if (
            not isinstance(text, str)
            or type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= len(text)
        ):
            raise ValueError("Select an exact numeric span from a document text field.")
        quote = text[start:end]
        match = re.fullmatch(
            r"\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)\s*(mm|m|rad|deg|°)\s*", quote
        )
        if match is None:
            raise ValueError("The cited span must contain one explicit number and supported unit.")
        value, unit = float(match[1]), match[2]
        value *= {"m": 1.0, "mm": 0.001, "rad": 1.0, "deg": math.pi / 180, "°": math.pi / 180}[unit]
        return self.save(
            {
                "record_type": "AssemblyQuantityEvidence",
                "status": "accepted",
                "quantity": quantity,
                "value": value,
                "unit": "m" if unit in {"m", "mm"} else "rad",
                "source_quote": quote,
                "source_field_path": field_path,
                "span": [start, end],
                "semantic_assignment": "PA_selected_document_quantity",
            },
            [record_ref],
        )

    def validation_specification(
        self, position_quantity_ref: str, axis_quantity_ref: str
    ) -> dict[str, Any]:
        """Use selected documented acceptance quantities; never choose numeric tolerances."""
        position, axis = self.read(position_quantity_ref), self.read(axis_quantity_ref)
        for record, quantity, unit in (
            (position, "position_tolerance_m", "m"),
            (axis, "axis_tolerance_rad", "rad"),
        ):
            if (
                record.get("record_type") != "AssemblyQuantityEvidence"
                or record.get("quantity") != quantity
                or record.get("unit") != unit
                or record.get("value", 0) <= 0
            ):
                raise ValueError(
                    "Select documented positive assembly acceptance tolerances with the required meaning and units."
                )
        return self.save(
            {
                "record_type": "AssemblyValidationSpecification",
                "status": "accepted",
                "family": "vertical_gear_assembly",
                "position_tolerance_m": position["value"],
                "axis_tolerance_rad": axis["value"],
                "yaw_required": False,
                "requires_threading": False,
                "requires_force_control": False,
                "unmodeled": ["tooth engagement", "yaw-specific assembly", "contact mechanics"],
            },
            [position_quantity_ref, axis_quantity_ref],
        )


def _same_frame(*records: Mapping[str, Any]) -> None:
    if any(record.get("status") != "accepted" or record.get("units") != "m" for record in records):
        raise ValueError("Geometry inputs must be accepted measurements in metres.")
    if len({record.get("frame_id") for record in records}) != 1 or not records[0].get("frame_id"):
        raise ValueError("Geometry frames disagree; no implicit conversion is allowed.")
