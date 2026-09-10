from __future__ import annotations

"""Measure candidate assembly geometry without choosing task roles or robot steps."""

import math
import re
import time
from copy import deepcopy
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..adapters.robot_validation_context import matrix_pose, pose_matrix
from ..agents.ra.refinement_records import _verify_evidence_tree, append_record, fingerprint, owned_path, read_pin
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


def circular_profile(triangles: np.ndarray) -> dict[str, Any]:
    """Measure one circular axis and its conservative CAD envelope.

    Args:
        triangles: Approved CAD triangles in metres and their original local frame.

    Returns:
        The circular axis, axial extent, and radii in CAD metres; zero inner radius
        indicates a solid or obstructed envelope, not a through opening.
    """
    triangles = np.asarray(triangles)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not len(triangles) or not np.isfinite(triangles).all():
        raise ValueError("CAD triangles must be finite three-dimensional measurements.")
    features = planar_features(triangles)
    pairs = []
    for index, first in enumerate(features):
        for second in features[index + 1:]:
            normal = np.asarray(first["normal"])
            if float(normal @ second["normal"]) > -1 + 1e-5:
                continue
            for a in first["circles"]:
                for b in second["circles"]:
                    delta = np.asarray(b["center_m"]) - a["center_m"]
                    axial = float(delta @ normal)
                    if (abs(axial) > 1e-5 and np.linalg.norm(delta - axial * normal) < 1e-5
                            and abs(a["radius_m"] - b["radius_m"]) < 1e-5):
                        pairs.append((abs(axial), (np.asarray(a["center_m"]) + b["center_m"]) / 2, normal, a["radius_m"]))
    if not pairs:
        raise NotImplementedError("The approved CAD has no supported pair of coaxial circular end faces.")
    extent = max(item[0] for item in pairs)
    selected = [item for item in pairs if math.isclose(item[0], extent, abs_tol=1e-5)]
    _, center, axis, _ = selected[0]
    if any(abs(float(axis @ item[2])) < 1 - 1e-5 or np.linalg.norm(center - item[1]) > 1e-5
           for item in selected):
        raise ValueError("CAD has ambiguous circular mounting axes.")
    u = np.eye(3)[int(np.argmin(np.abs(axis)))]
    u -= axis * float(u @ axis)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    local = np.asarray(triangles) - center
    axial = local @ axis
    if float(np.ptp(axial)) > extent + 1e-5:
        raise NotImplementedError("The circular end faces do not span the complete CAD part; this mating shape is unsupported.")
    xy = np.stack((local @ u, local @ v), axis=-1)
    first, second = xy, np.roll(xy, -1, axis=1)
    edges = second - first
    square = np.sum(edges ** 2, axis=-1)
    amount = np.divide(-np.sum(first * edges, axis=-1), square,
                       out=np.zeros_like(square), where=square > 0)
    closest = first + np.clip(amount, 0, 1)[..., None] * edges
    radii = np.linalg.norm(closest, axis=-1)
    cross = first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]
    inside = (np.all(cross > 1e-14, axis=1) | np.all(cross < -1e-14, axis=1))
    inner = 0.0 if inside.any() else float(radii.min())
    outer = float(np.linalg.norm(xy, axis=-1).max())
    if inner > 0 and not any(item[3] < outer - 1e-5 for item in selected):
        raise NotImplementedError("The CAD opening does not have supported coaxial circular boundaries at both end faces.")
    return {"center_m": center.tolist(), "axis": axis.tolist(), "height_m": float(np.ptp(axial)),
            "inner_radius_m": inner, "outer_radius_m": outer}


def annular_collision_segments(inner_radius_m: float, outer_radius_m: float, height_m: float) -> np.ndarray:
    """Build 64 closed convex sectors that preserve a conservative circular opening.

    Args:
        inner_radius_m: CAD opening radius, or zero for a filled shaft envelope.
        outer_radius_m: Maximum CAD radius to enclose.
        height_m: Measured axial extent.

    Returns:
        A sector-by-triangle array in metres, centered on the circular axis.
    """
    if not all(math.isfinite(value) for value in (inner_radius_m, outer_radius_m, height_m)) or not (
        0 <= inner_radius_m < outer_radius_m and height_m > 0
    ):
        raise ValueError("Collision sectors require finite positive dimensions and an open radial interval.")
    count = 64
    outer = outer_radius_m / math.cos(math.pi / count)
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                      [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                      [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    sectors = []
    for index in range(count):
        a, b = 2 * math.pi * index / count, 2 * math.pi * (index + 1) / count
        corners = [(radius * math.cos(angle), radius * math.sin(angle))
                   for radius, angle in ((inner_radius_m, a), (outer, a), (outer, b), (inner_radius_m, b))]
        vertices = np.array([[x, y, z] for z in (-height_m / 2, height_m / 2) for x, y in corners])
        triangles = vertices[faces]
        area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
        sectors.append(triangles[area > 1e-16])
    return np.asarray(sectors)


def observed_circular_face(points: np.ndarray, profile: Mapping[str, Any]) -> dict[str, Any]:
    """Measure a circular face axis and center while leaving CAD yaw unresolved.

    Args:
        points: Calibrated observed candidate points in world metres.
        profile: Circular dimensions checked against the approved CAD triangles.

    Returns:
        The observed face center, normal, and fit diagnostics.
    """
    from scipy.spatial import ConvexHull
    from .rgb_d_cad_grounding.pose_estimation import measured_planar_feature

    tolerance = min(0.0001, profile["outer_radius_m"] / 100)
    plane = measured_planar_feature(points, tolerance)
    if plane is None:
        raise ValueError("The observed candidate does not establish a planar circular mounting face.")
    axis = np.asarray(plane["normal"])
    if axis[2] < 0:
        axis = -axis
    if axis[2] < 1 - 1e-5:
        raise ValueError("The measured mounting axis is tilted; rigid vertical fitting remains unresolved.")
    u = np.array([1.0, 0.0, 0.0])
    u -= axis * float(u @ axis)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    points = plane["inlier_points_m"]
    center = points.mean(axis=0)
    xy = np.column_stack(((points - center) @ u, (points - center) @ v))
    hull = xy[ConvexHull(xy).vertices]
    fit, _, rank, _ = np.linalg.lstsq(np.column_stack((2 * hull, np.ones(len(hull)))),
                                     np.sum(hull ** 2, axis=1), rcond=None)
    if rank != 3 or fit[2] + fit[:2] @ fit[:2] <= 0:
        raise ValueError("The observed circular outline has an ambiguous center.")
    radius = math.sqrt(float(fit[2] + fit[:2] @ fit[:2]))
    residual = float(np.max(np.abs(np.linalg.norm(hull - fit[:2], axis=1) - radius)))
    # Pixel centers undersample the silhouette. Preserve the measured residual;
    # CAD supplies the envelope radius, never an invented observed orientation.
    if residual > profile["outer_radius_m"] * 0.1 or not 0.7 < radius / profile["outer_radius_m"] < 1.1:
        raise ValueError("The observed circular outline contradicts the selected CAD envelope.")
    center += u * fit[0] + v * fit[1]
    return {"center_m": center.tolist(), "axis": axis.tolist(), "radius_m": radius,
            "outline_residual_m": residual, "plane_rms_distance_m": plane["rms_distance_m"],
            "measurement_resolution_m": tolerance, "inlier_count": plane["inlier_count"]}


class _ObservedGeometryBatch:
    """Share checked, immutable observation inputs for one measurement batch only."""

    def __init__(self, root: Path, authorized: Mapping[str, str]) -> None:
        self.root = root
        self.authorized = dict(authorized)
        self.records: dict[str, tuple[str, Any]] = {}
        self.candidates: dict[str, list[dict[str, Any]]] = {}
        self.points: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        self.timings: dict[str, float] = {}

    @classmethod
    def prepare(cls, root: Path, authorized: Mapping[str, str], requests: list[Mapping[str, Any]]) -> _ObservedGeometryBatch:
        from .rgb_d_cad_grounding.size_correspondence import _load_candidates

        batch = cls(root, authorized)
        started = time.monotonic()
        for request in requests:
            for field in ("segmentation_ref", "calibration_ref"):
                ref = request[field]
                if ref not in authorized:
                    raise ValueError("Measurement input was not issued when the batch started.")
                _verify_evidence_tree(root, {"ref": ref, "sha256": authorized[ref]}, batch.records)
        batch.timings["preparation_sec"] = time.monotonic() - started
        started = time.monotonic()
        for ref in dict.fromkeys(request["segmentation_ref"] for request in requests):
            points: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            _, _, candidates = _load_candidates(root, owned_path(root, ref), _decoded_points=points)
            for candidate in candidates:
                for value in candidate.values():
                    if isinstance(value, np.ndarray):
                        value.flags.writeable = False
            for arrays in points.values():
                for value in arrays:
                    value.flags.writeable = False
            batch.points[ref], batch.candidates[ref] = points, candidates
        batch.timings["decoding_sec"] = time.monotonic() - started
        started = time.monotonic()
        batch.verify_sources()
        batch.timings["verification_sec"] = time.monotonic() - started
        return batch

    def verify_sources(self) -> None:
        # Rehash prepared dependencies after decoding/processing. Reuse ends with
        # this batch; it cannot refresh sensor time or survive a model response.
        from ..agents.ra.refinement_records import pin

        for ref, (sha, _) in self.records.items():
            if pin(self.root, owned_path(self.root, ref))["sha256"] != sha:
                raise ValueError("Measurement source changed during processing: " + ref)


class AssemblyGeometryProducer:
    """Consume exact approved records; persist neutral geometry or selected derivations."""

    def __init__(
        self,
        root: Path,
        directory: Path,
        authorized: dict[str, str],
        target_feature: Mapping[str, Any] | None = None,
        *,
        _batch: _ObservedGeometryBatch | None = None,
    ) -> None:
        """Pin the producer to an interaction and the investigation's issued evidence."""
        self.root, self.directory, self.authorized = root, directory, authorized
        self.target_feature = dict(target_feature or {})
        self._batch = _batch
        self._checked_records = dict(_batch.records) if _batch else None
        self._issued: list[dict[str, str]] = []
        self.timings = {"geometry_sec": 0.0, "verification_sec": 0.0, "persistence_sec": 0.0}

    def read(self, ref: str) -> dict[str, Any]:
        """Read only an issued record with its originally accepted bytes."""
        if ref not in self.authorized:
            raise ValueError("Geometry input was not issued to this investigation.")
        from ..agents.ra.refinement_records import verify_evidence_tree

        source = {"ref": ref, "sha256": self.authorized[ref]}
        if self._checked_records is not None:
            return deepcopy(_verify_evidence_tree(self.root, source, self._checked_records))
        return verify_evidence_tree(self.root, source)

    def verify_outputs(self) -> None:
        """Recheck sources and issued output bytes before the coordinator admits them."""
        started = time.monotonic()
        if self._batch is not None:
            self._batch.verify_sources()
        checked: dict[str, tuple[str, Any]] = {}
        for source in self._issued:
            _verify_evidence_tree(self.root, source, checked)
        self.timings["verification_sec"] += time.monotonic() - started

    def save(self, payload: Mapping[str, Any], sources: list[str]) -> dict[str, Any]:
        """Append a measurement with complete direct source pins."""
        started = time.monotonic()
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
        self._issued.append(reference)
        self.timings["persistence_sec"] += time.monotonic() - started
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
                vertices = cad.triangles_m.reshape(-1, 3)
                heights = []
                for qualified in hypotheses:
                    qualified_transform = calibration.target_from_camera @ np.asarray(
                        qualified["camera_from_CAD_transform"]
                    )
                    matrix_pose(qualified_transform)
                    heights.append(float(np.ptp(vertices @ qualified_transform[2, :3])))
                height = heights[0]
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
                        "qualified_height_estimates_m": heights,
                        "qualified_height_range_m": [min(heights), max(heights)],
                    },
                    warning=(
                        "part_height_m is a world-vertical CAD extent estimate from the "
                        "highest-ranked qualified observed pose. Full pose and mating geometry "
                        "remain ambiguous; the range across all qualified hypotheses is recorded "
                        "in uncertainty. These estimates do not establish sensor accuracy."
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

    def observed_geometry(
        self, segmentation_ref: str, observation_handle: str, calibration_ref: str,
    ) -> dict[str, Any]:
        """Measure all candidate boxes and the detected support in a PA-selected view.

        Boxes describe the visible points extended to the observed support plane.
        That support assumption and missing occluded geometry remain explicit.
        Identity, task role and CAD orientation are not inferred by this operation.
        """
        standalone = self._batch is None
        if standalone:
            self._batch = _ObservedGeometryBatch.prepare(self.root, self.authorized, [{
                "segmentation_ref": segmentation_ref, "calibration_ref": calibration_ref,
                "observation_handle": observation_handle,
            }])
            self._checked_records = dict(self._batch.records)
        try:
            result = self._observed_geometry(segmentation_ref, observation_handle, calibration_ref)
            if standalone:
                self.verify_outputs()
            return result
        finally:
            if standalone:
                self._batch = self._checked_records = None

    def _observed_geometry(self, segmentation_ref: str, observation_handle: str, calibration_ref: str) -> dict[str, Any]:
        from .rgb_d_cad_grounding.frame_conversion import _load_calibration

        started, persisted = time.monotonic(), self.timings["persistence_sec"]
        camera, stamp = self.observation(segmentation_ref, observation_handle)
        surface_result = self.observed_surface(segmentation_ref, observation_handle, calibration_ref)
        surface = surface_result["record"]
        if surface["frame_id"] != "world" or abs(surface["normal"][2]) < 1e-9:
            raise ValueError("Observed geometry requires a world-frame support plane with finite vertical height.")
        calibration = _load_calibration(
            self.root, owned_path(self.root, calibration_ref), observation_timestamp_ns=stamp,
        )
        candidates = self._batch.candidates[segmentation_ref]
        selected = [item for item in candidates if item["observation_handle"] == observation_handle]
        if not selected:
            raise ValueError("The selected observation contains no segmented candidates.")
        normal, offset = np.asarray(surface["normal"]), float(surface["offset_m"])
        measured = []
        for candidate in selected:
            points = np.asarray(candidate["points_m"], dtype=float)
            transform = calibration.target_from_camera
            points = points @ transform[:3, :3].T + transform[:3, 3]
            minimum, maximum = points.min(axis=0), points.max(axis=0)
            center_xy = (minimum[:2] + maximum[:2]) / 2
            support_z = -(offset + normal[:2] @ center_xy) / normal[2]
            # Extend visible object points to the selected view's measured support;
            # never substitute a CAD-local dimension for observed world height.
            corners = np.array([[x, y] for x in (minimum[0], maximum[0])
                                for y in (minimum[1], maximum[1])])
            minimum[2] = min(minimum[2], float(np.min(-(offset + corners @ normal[:2]) / normal[2])))
            size = maximum - minimum
            if not np.isfinite(size).all() or np.any(size <= 0):
                raise ValueError("Candidate points do not establish finite positive observed bounds.")
            center = (minimum + maximum) / 2
            reference_pose = dict(zip(("x", "y", "z"), map(float, center)))
            reference_pose.update(qx=0.0, qy=0.0, qz=0.0, qw=1.0)
            measured.append(self.save({
                "record_type": "ObservedGeometryEvidence", "status": "accepted",
                "frame_id": "world", "units": "m", "reference_point": "observed_bounds_center",
                "reference_pose": reference_pose,
                "bounds_m": {"minimum": minimum.tolist(), "maximum": maximum.tolist()},
                "size_m": size.tolist(), "part_height_m": float(size[2]),
                "product_geometry": {"part_height_m": float(size[2]), "board_center": {"z": float(support_z)}},
                "placement_surface_point": {"x": float(center[0]), "y": float(center[1]), "z": float(maximum[2])},
                "object_id": "observed_" + fingerprint({"segmentation_ref": segmentation_ref,
                    "candidate_handle": candidate["candidate_handle"]})[:20],
                "segmentation_ref": segmentation_ref,
                "candidate_reference": {"observation_handle": observation_handle,
                                        "candidate_handle": candidate["candidate_handle"]},
                "observation_timestamp_ns": stamp,
                "uncertainty": {"geometry_model": "observed_bounds",
                    "support_assumption": "visible_points_extended_to_selected_observed_plane",
                    "partial_visibility": candidate["partial_visibility"],
                    "CAD_orientation": "not_established", "occluded_geometry": "unmeasured",
                    "support_rms_distance_m": surface["rms_distance_m"]},
            }, [segmentation_ref, calibration_ref, surface_result["record_ref"]]))
        # A finite measured patch avoids treating an observed tabletop as an
        # infinite obstacle through unrelated parts of the robot's workspace.
        camera_points, _ = self._batch.points[segmentation_ref][observation_handle]
        distance = np.abs(camera_points @ np.asarray(camera["support_plane"]["normal"]) + camera["support_plane"]["offset_m"])
        threshold = max(float(surface["rms_distance_m"]) * 3, 1e-6)
        patch = camera_points[distance <= threshold]
        if len(patch) < 3:
            raise ValueError("The selected plane has insufficient observed support coverage.")
        patch = patch @ calibration.target_from_camera[:3, :3].T + calibration.target_from_camera[:3, 3]
        low, high = patch.min(axis=0), patch.max(axis=0)
        vertices = np.array([[x, y, -(offset + normal[0] * x + normal[1] * y) / normal[2]]
                             for x, y in ((low[0], low[1]), (high[0], low[1]),
                                          (high[0], high[1]), (low[0], high[1]))])
        mesh_path = self.directory / (Path(surface_result["record_ref"]).stem + "_surface.npz")
        with mesh_path.open("xb") as stream:
            np.savez_compressed(stream, triangles_m=vertices[[[0, 1, 2], [0, 2, 3]]])
        from ..agents.ra.refinement_records import pin
        patch_result = self.save({**{key: value for key, value in surface.items()
                                    if key not in {"fingerprint", "source_refs", "created_at_ns"}},
            "mesh": pin(self.root, mesh_path),
            "coverage": "observed_plane_patch",
        }, [surface_result["record_ref"], segmentation_ref])
        self.timings["geometry_sec"] += time.monotonic() - started - (self.timings["persistence_sec"] - persisted)
        return {"status": "accepted", "geometry_refs": [item["record_ref"] for item in measured],
                "surface_ref": patch_result["record_ref"], "measurements": measured}

    def bind_observed_part(self, part_ref: str, cad_ref: str, feature_name: str) -> dict[str, Any]:
        """Bind PA-selected observed bounds and CAD to an already accepted feature."""
        part, cad = self.read(part_ref), self.read(cad_ref)
        if part.get("record_type") != "ObservedGeometryEvidence" or cad.get("record_type") != "CADMeshRecord":
            raise ValueError("Select ObservedGeometryEvidence and an issued CADMeshRecord.")
        cad_input = _load_cad_input(self.root, owned_path(self.root, cad_ref))
        selected = self.save({**{key: value for key, value in part.items()
                                if key not in {"fingerprint", "source_refs", "created_at_ns"}},
            "cad_context_ref": cad_input.context_ref,
            "CAD_mesh": {"ref": cad_input.mesh_ref, "sha256": cad_input.mesh_sha256},
        }, [part_ref, cad_ref])
        return self.bind_part(selected["record_ref"], feature_name)

    def _candidate_points(self, record: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        from .rgb_d_cad_grounding.frame_conversion import _load_calibration

        pending = [source["ref"] for source in record["source_refs"]]
        lineage = {}
        while pending:
            ref = pending.pop()
            if ref in lineage:
                continue
            ancestor = self.read(ref)
            lineage[ref] = ancestor
            pending.extend(source["ref"] for source in ancestor.get("source_refs", []))
        handle = record["candidate_reference"]["observation_handle"]
        camera, stamp = self.observation(record["segmentation_ref"], handle)
        calibrations = [ref for ref, value in lineage.items()
                        if value.get("record_type") == "CameraToRobotCalibrationRecord"
                        and value.get("source_frame") == camera["frame"] and value.get("target_frame") == "world"]
        if len(calibrations) != 1:
            raise ValueError("Mating geometry requires one original observation calibration.")
        ref = calibrations[0]
        if self._batch is None:
            self._batch = _ObservedGeometryBatch.prepare(self.root, self.authorized, [{
                "segmentation_ref": record["segmentation_ref"], "calibration_ref": ref,
            }])
            self._checked_records = dict(self._batch.records)
        if record["segmentation_ref"] not in self._batch.candidates:
            raise ValueError("Mating observations must belong to the same declared segmentation.")
        candidates = [item for item in self._batch.candidates[record["segmentation_ref"]]
                      if item["candidate_handle"] == record["candidate_reference"]["candidate_handle"]
                      and item["observation_handle"] == handle]
        if len(candidates) != 1 or candidates[0]["partial_visibility"]:
            raise ValueError("The circular mounting candidate is missing, ambiguous or clipped.")
        transform = _load_calibration(self.root, owned_path(self.root, ref), observation_timestamp_ns=stamp).target_from_camera
        arrays = (candidates[0]["points_m"], self._batch.points[record["segmentation_ref"]][handle][0])
        return tuple(points @ transform[:3, :3].T + transform[:3, 3] for points in arrays)

    def observed_mating_geometry(self, part_ref: str, target_ref: str) -> dict[str, Any]:
        """Measure a bore, shaft and seating surface for the accepted observed pair.

        Args:
            part_ref: ObservedGeometryEvidence bound to the accepted moving feature and CAD.
            target_ref: ObservedGeometryEvidence bound to the accepted destination and CAD.

        Returns:
            Pinned measured goal geometry, including clearance, for fitting validation.
        """
        from scipy.spatial import ConvexHull
        from scipy.spatial.transform import Rotation
        from .rgb_d_cad_grounding.pose_estimation import measured_planar_feature
        from ..agents.ra.refinement_records import pin

        part, target = self.read(part_ref), self.read(target_ref)
        _same_frame(part, target)
        if (not part.get("assembly_association_sha256")
                or part["assembly_association_sha256"] != target.get("assembly_association_sha256")):
            raise ValueError("Mating geometry requires the same accepted assembly relationship at both endpoints.")
        profiles = []
        for record in (part, target):
            if record.get("record_type") != "ObservedGeometryEvidence" or "CAD_mesh" not in record:
                raise ValueError("Mating geometry requires observed instances with approved CAD bindings.")
            try:
                with np.load(owned_path(self.root, record["CAD_mesh"]["ref"]), allow_pickle=False) as mesh:
                    profiles.append(circular_profile(mesh["triangles_m"]))
            except NotImplementedError as exc:
                return self.save({
                    "record_type": "AssemblyGeometryEvidence", "status": "unsupported",
                    "reason": str(exc), "frame_id": part["frame_id"], "units": "m",
                    "reference_point": "observed_bounds_center", "part_name": part["part_name"],
                    "part_object_id": part["object_id"],
                    "assembly_association_sha256": part["assembly_association_sha256"],
                    "part_geometry_ref": part_ref, "target_geometry_ref": target_ref, "product_geometry": {},
                    "observation_timestamp_ns": min(part["observation_timestamp_ns"], target["observation_timestamp_ns"]),
                }, [part_ref, target_ref])
        if profiles[0]["inner_radius_m"] <= 0:
            return self.save({
                "record_type": "AssemblyGeometryEvidence", "status": "failed",
                "reason": "CAD triangles close or obstruct the moving component's through opening.",
                "frame_id": part["frame_id"], "units": "m", "reference_point": "observed_bounds_center",
                "part_name": part["part_name"], "part_object_id": part["object_id"],
                "assembly_association_sha256": part["assembly_association_sha256"],
                "part_geometry_ref": part_ref, "target_geometry_ref": target_ref, "product_geometry": {},
                "observation_timestamp_ns": min(part["observation_timestamp_ns"], target["observation_timestamp_ns"]),
            }, [part_ref, target_ref])
        shaped = []
        for (ref, record, moving), profile in zip(((part_ref, part, True), (target_ref, target, False)), profiles):
            points, all_points = self._candidate_points(record)
            face = observed_circular_face(points, profile)
            axis = np.asarray(face["axis"])
            center = np.asarray(face["center_m"]) - axis * profile["height_m"] / 2
            rotation = Rotation.align_vectors([axis], [[0.0, 0.0, 1.0]])[0].as_matrix()
            sectors = annular_collision_segments(profile["inner_radius_m"],
                                                 profile["outer_radius_m"], profile["height_m"])
            triangles = sectors.reshape(-1, 3, 3) @ rotation.T + center
            reference = pose_matrix(record["reference_pose"])
            local = (triangles - reference[:3, 3]) @ reference[:3, :3]
            path = self.directory / ("part_collision.npz" if moving else "target_collision.npz")
            self.directory.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                np.savez_compressed(stream, triangles_m=local)
            payload = {key: value for key, value in record.items()
                       if key not in {"fingerprint", "source_refs", "created_at_ns"}}
            payload.update(mesh=pin(self.root, path), circular_profile=profile, observed_circular_face=face,
                           axis_local=(reference[:3, :3].T @ axis).tolist(),
                           center_local_m=(reference[:3, :3].T @ (center - reference[:3, 3])).tolist(),
                           uncertainty={**record.get("uncertainty", {}),
                                        "collision_geometry": "CAD circular envelope with measured axis",
                                        "CAD_orientation": "not_established"})
            shaped.append(self.save(payload, [ref]))
        moving, destination = (item["record"] for item in shaped)
        radius = moving["circular_profile"]["inner_radius_m"] * math.cos(math.pi / 64)
        shaft_radius = destination["circular_profile"]["outer_radius_m"] / math.cos(math.pi / 64)
        face = destination["observed_circular_face"]
        cap, axis = np.asarray(face["center_m"]), np.asarray(face["axis"])
        radial = all_points - cap
        heights = radial @ axis
        distances = np.linalg.norm(radial - heights[:, None] * axis, axis=1)
        ring = all_points[(distances > shaft_radius) & (distances < moving["circular_profile"]["outer_radius_m"])
                          & (heights < -face["measurement_resolution_m"])
                          & (heights > -destination["circular_profile"]["height_m"] * 2)]
        surfaces = []
        for _ in range(4):
            plane = measured_planar_feature(ring, face["measurement_resolution_m"])
            if plane is None:
                break
            normal = np.asarray(plane["normal"])
            if normal @ axis < 0:
                normal, plane["offset_m"] = -normal, -plane["offset_m"]
            if float(normal @ axis) >= 1 - 1e-5:
                points = plane["inlier_points_m"]
                hull = ConvexHull(points[:, :2])
                position = cap - axis * (float(normal @ cap) + plane["offset_m"]) / float(normal @ axis)
                if np.all(hull.equations[:, :2] @ position[:2] + hull.equations[:, 2] <= 1e-9):
                    surfaces.append((float(position @ axis), plane, normal, position, points[hull.vertices]))
            residual = np.abs(ring @ np.asarray(plane["normal"]) + (
                plane["offset_m"] if np.dot(normal, plane["normal"]) > 0 else -plane["offset_m"]))
            ring = ring[residual > plane["tolerance_m"]]
        if not surfaces:
            raise ValueError("The destination needs a measured seating surface around the shaft; its top is not a seating height.")
        _, plane, normal, position, boundary = max(surfaces, key=lambda item: item[0])
        if float(axis @ (cap - position)) <= plane["tolerance_m"]:
            raise ValueError("The measured seating surface does not establish shaft engagement.")
        projected = boundary - (boundary @ normal + plane["offset_m"])[:, None] * normal
        triangles = np.array([[projected[0], projected[index], projected[index + 1]]
                              for index in range(1, len(projected) - 1)])
        path = self.directory / "seating_surface.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, triangles_m=triangles)
        surface = self.save({
            "record_type": "AssemblySurfaceEvidence", "status": "accepted", "frame_id": "world", "units": "m",
            "normal": normal.tolist(), "offset_m": plane["offset_m"], "rms_distance_m": plane["rms_distance_m"],
            "coverage": "observed_plane_patch", "mesh": pin(self.root, path),
            "observation_timestamp_ns": target["observation_timestamp_ns"],
        }, [target_ref])
        # A measured interference is valid evidence, not a missing measurement.
        # Preserve it for target calculation and the separate fitting check.
        reference = pose_matrix(moving["reference_pose"])
        current_axis = reference[:3, :3] @ moving["axis_local"]
        rotation = Rotation.align_vectors([axis], [current_axis])[0].as_matrix() @ reference[:3, :3]
        height = moving["circular_profile"]["height_m"]
        resolution = max(plane["tolerance_m"], moving["observed_circular_face"]["measurement_resolution_m"])
        # Release within the measured surface resolution, with a positive gap so
        # the collision model never relies on penetrating the supporting surface.
        center = position + axis * (height / 2 + resolution / 2)
        final = np.eye(4)
        final[:3, :3] = rotation
        final[:3, 3] = center - rotation @ moving["center_local_m"]
        origin = matrix_pose(final)
        goal = {
            "record_type": "AssemblyGeometryEvidence", "status": "accepted", "frame_id": "world", "units": "m",
            "reference_point": "observed_bounds_center", "target_origin_pose": origin,
            "part_name": moving["part_name"], "part_object_id": moving["object_id"],
            "assembly_association_sha256": moving["assembly_association_sha256"],
            "part_axis_local": moving["axis_local"], "part_center_local_m": moving["center_local_m"],
            "part_outer_radius_m": moving["circular_profile"]["outer_radius_m"],
            "part_height_m": height, "bore_radius_m": radius, "shaft_radius_m": shaft_radius,
            "shaft_axis": axis.tolist(), "shaft_top_m": cap.tolist(),
            "shaft_height_m": destination["circular_profile"]["height_m"],
            "seating_point_m": position.tolist(), "seating_normal": normal.tolist(),
            "seating_resolution_m": resolution, "nominal_radial_clearance_m": radius - shaft_radius,
            "part_geometry_ref": shaped[0]["record_ref"], "target_geometry_ref": shaped[1]["record_ref"],
            "seating_surface_ref": surface["record_ref"],
            "insertion_axis": (-axis).tolist(),
            "product_geometry": {"placement_surface_point": dict(zip(("x", "y", "z"), position.tolist())),
                                 "target_origin_pose": origin, "insertion_axis": (-axis).tolist(),
                                 "insertion_distance_m": float(axis @ (cap - position)), "part_height_m": height},
            "observation_timestamp_ns": min(part["observation_timestamp_ns"], target["observation_timestamp_ns"]),
            "uncertainty": {"CAD_orientation": "not_established", "yaw_specific_assembly": "unmodeled",
                            "seating": "nominal contact within measured plane resolution"},
        }
        return self.save(goal, [item["record_ref"] for item in shaped] + [surface["record_ref"]])

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

    def observation(
        self, segmentation_ref: str, observation_handle: str
    ) -> tuple[dict[str, Any], int]:
        """Resolve one exact observed view and its verified depth timestamp.

        Args:
            segmentation_ref: Issued segmentation record reference.
            observation_handle: Exact canonical handle resolved by the PA adapter.

        Returns:
            The selected camera entry and its source depth timestamp.
        """
        from .rgb_d_cad_grounding.size_correspondence import (
            CAMERA_IDS, _load_segmentation_source, _validate_camera_metadata,
        )

        segmentation = self.read(segmentation_ref)
        if self._batch is None:
            _, segmentation, preprocessing = _load_segmentation_source(self.root, owned_path(self.root, segmentation_ref))
        else:
            preprocessing = self._checked_records[segmentation["source_record"]["ref"]][1]
        matches = [
            item
            for item in segmentation["cameras"]
            if item["observation_handle"] == observation_handle
        ]
        if len(matches) != 1:
            raise ValueError(
                "The selected observation handle is absent or duplicated in this segmentation."
            )
        camera = matches[0]
        source_camera = next(
            item for item in preprocessing["cameras"] if item["camera_id"] == camera["camera_id"]
        )
        if self._batch is None:
            _validate_camera_metadata(self.root, camera, camera_id=camera["camera_id"],
                                      camera_index=CAMERA_IDS.index(camera["camera_id"]), source_camera=source_camera)
        return camera, source_camera["depth_timestamp_ns"]

    def observed_surface(
        self, segmentation_ref: str, observation_handle: str, calibration_ref: str
    ) -> dict[str, Any]:
        """Transform one observed support-plane candidate without assigning its role."""
        from .rgb_d_cad_grounding.frame_conversion import _load_calibration

        camera, stamp = self.observation(segmentation_ref, observation_handle)
        self.read(calibration_ref)
        plane = camera["support_plane"]
        if plane.get("status") != "detected":
            raise ValueError("No support-plane candidate was measured in this view.")
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

    def surface_height(self, surface_ref: str, location_ref: str) -> dict[str, Any]:
        """Measure an issued plane's height at a PA-selected observed world location."""
        surface, location = self.read(surface_ref), self.read(location_ref)
        _same_frame(surface)
        if surface.get("record_type") != "AssemblySurfaceEvidence":
            raise ValueError("Select accepted observed plane evidence for surface_ref.")
        if (
            location.get("record_type") != "RobotFrameLocationRecord"
            or location.get("location") != "available"
            or location.get("robot_frame_conversion") != "accepted"
            or location.get("target_frame") != surface["frame_id"]
            or surface["frame_id"] != "world"
        ):
            raise ValueError("Select a calibrated RobotFrameLocationRecord in the plane's world frame.")
        point = np.asarray(location["translated_location_m"], dtype=float)
        normal = np.asarray(surface["normal"], dtype=float)
        offset = float(surface["offset_m"])
        if (
            point.shape != (3,) or normal.shape != (3,)
            or not np.isfinite(point).all() or not np.isfinite(normal).all()
            or not math.isfinite(offset) or abs(normal[2]) < 1e-9
        ):
            raise ValueError("The selected plane and point do not establish a finite vertical height.")
        height = -(offset + normal[0] * point[0] + normal[1] * point[1]) / normal[2]
        return self.save({
            "record_type": "AssemblySurfaceEvidence", "status": "accepted",
            "frame_id": "world", "units": "m", "normal": normal.tolist(),
            "offset_m": offset, "rms_distance_m": surface["rms_distance_m"],
            "reference_point": "observed_plane_at_selected_location",
            "evaluation_point_m": point.tolist(),
            "product_geometry": {"board_center": {"z": float(height)}},
            "observation_timestamp_ns": min(
                surface["observation_timestamp_ns"], location["observation_timestamp_ns"]
            ),
        }, [surface_ref, location_ref])

    def select_grasp_point(
        self, part_ref: str, plane_id: str, circle_id: str
    ) -> dict[str, Any]:
        """Bind a PA-selected circular feature as the grasp reference, retaining the CAD frame."""
        part = self.read(part_ref)
        _same_frame(part)
        if part.get("reference_point") != "CAD_origin":
            raise ValueError("A grasp feature requires registered CAD geometry.")
        plane = next(item for item in part["features"] if item["plane_id"] == plane_id)
        circle = next(item for item in plane["circles"] if item["circle_id"] == circle_id)
        point = np.asarray(circle["center_m"], dtype=float)
        transform = pose_matrix(part["origin_pose"])
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("The selected grasp feature has no finite CAD-local centre.")
        payload = {
            key: value for key, value in part.items()
            if key not in {"fingerprint", "source_refs", "created_at_ns"}
        }
        payload["grasp_reference"] = {
            "reference_point": "selected_CAD_feature",
            "plane_id": plane_id,
            "circle_id": circle_id,
            "point_CAD_m": point.tolist(),
            "point_world_m": (transform[:3, :3] @ point + transform[:3, 3]).tolist(),
        }
        return self.save(payload, [part_ref])

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
        grasp = part.get("grasp_reference")
        point = (
            np.asarray(grasp["point_world_m"], dtype=float) if grasp
            else np.asarray([origin[key] for key in ("x", "y", "z")])
        )
        bounds = part["bounds_m"]
        if np.any(point < np.asarray(bounds["minimum"]) - 1e-6) or np.any(
            point > np.asarray(bounds["maximum"]) + 1e-6
        ):
            raise ValueError(
                "The selected grasp reference lies outside the measured part bounds. "
                "A CAD file origin is not automatically a physical grasp point; "
                "select_grasp_point can bind a PA-selected measured feature."
            )
        support_z = (
            -(surface["offset_m"] + normal[0] * point[0] + normal[1] * point[1]) / normal[2]
        )
        return self.save(
            {
                "record_type": "AssemblyGeometryEvidence",
                "status": "accepted",
                "frame_id": part["frame_id"],
                "units": "m",
                "reference_point": "selected_CAD_feature" if grasp else "CAD_origin",
                "target_pose": dict(zip(("x", "y", "z"), point.tolist(), strict=True)),
                "product_geometry": {
                    "board_center": {"z": float(support_z)},
                    "part_height_m": part["part_height_m"],
                },
                "observation_timestamp_ns": part["observation_timestamp_ns"],
                "uncertainty": {
                    "support_rms_m": surface["rms_distance_m"],
                    "pose_record_ref": part_ref,
                },
                **({"grasp_reference": grasp} if grasp else {}),
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
        grasp = part.get("grasp_reference")
        grasp_offset = (
            (matrix[:3, :3] @ np.asarray(grasp["point_CAD_m"], dtype=float)).tolist()
            if grasp else None
        )
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
                        **({"grasp_point": "selected_CAD_feature"} if grasp else {}),
                    },
                    "target_origin_pose": {key: origin[key] for key in ("x", "y", "z")},
                    **({"grasp_point_offset_world_m": grasp_offset} if grasp else {}),
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
        observed = any(item.get("record_type") == "ObservedGeometryEvidence" for item in geometry)
        declared_views = {(item["segmentation_ref"], item["candidate_reference"]["observation_handle"])
                          for item in geometry}
        if observed and {ref for ref, _ in declared_views} != set(segmentation_refs):
            raise ValueError("Declared observed geometry and segmentation coverage must reference the same observations.")
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
                if observed and (ref, camera["observation_handle"]) not in declared_views:
                    continue
                for candidate in camera["candidates"]:
                    if (ref, candidate["candidate_handle"]) not in covered:
                        unresolved.append(
                            {
                                "segmentation_ref": ref,
                                "candidate_handle": candidate["candidate_handle"],
                            }
                        )
        objects = [
            {"object_id": item["object_id"],
             **({"mesh": item["mesh"]} if "mesh" in item else {"size_m": item["size_m"]}),
             "pose": item["reference_pose"] if item.get("record_type") == "ObservedGeometryEvidence" else item["origin_pose"]}
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
                **({"mesh": item["mesh"]} if observed and "mesh" in item else
                   {"plane": [*item["normal"], item["offset_m"]]}),
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
                **({"geometry_model": "observed_bounds", "declared_observations": [
                    {"segmentation_ref": ref, "observation_handle": handle}
                    for ref, handle in sorted(declared_views)]} if observed else {}),
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
