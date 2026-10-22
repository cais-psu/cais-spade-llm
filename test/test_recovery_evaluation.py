from __future__ import annotations

"""Frozen implementation manifests and evaluation condition coverage."""

from copy import deepcopy

import pytest

from cais_spade_llm.recovery_framework import recovery_evaluation as evaluation
from cais_spade_llm.recovery_framework import evaluation_conditions as conditions


def test_freeze_detects_input_change_and_missing_files(tmp_path):
    path = tmp_path / "declaration.json"
    path.write_text('{"part_name": null}')
    manifest = {"files": {str(path): evaluation.file_hash(path)}}
    evaluation.verify_frozen(manifest)
    path.write_text('{"part_name": "KET4_Square_4mm"}')
    with pytest.raises(ValueError, match="Frozen evaluation input changed"):
        evaluation.verify_frozen(manifest)
    path.unlink()
    with pytest.raises(ValueError, match="Frozen evaluation input changed"):
        evaluation.verify_frozen(manifest)


def test_freeze_rejects_added_implementation_files(tmp_path):
    package = tmp_path / "cais_spade_llm"
    package.mkdir()
    (tmp_path / "pyproject.toml").write_text("configuration")
    (tmp_path / "poetry.lock").write_text("lock")
    (package / "owner.py").write_text("value = 1\n")
    paths = evaluation._implementation_files(tmp_path)
    manifest = {"files": {str(path.relative_to(tmp_path)): evaluation.file_hash(path) for path in paths},
                "implementation_files": sorted(str(path.relative_to(tmp_path)) for path in paths)}
    evaluation.verify_frozen(manifest, root=tmp_path)
    (package / "new_validator.py").write_text("value = 2\n")
    with pytest.raises(ValueError, match="inventory changed"):
        evaluation.verify_frozen(manifest, root=tmp_path)


@pytest.mark.parametrize("change", ["content", "added", "removed"])
def test_freeze_retains_nav2_map_pixels_and_inventory(tmp_path, change):
    (tmp_path / "pyproject.toml").write_text("configuration")
    (tmp_path / "poetry.lock").write_text("lock")
    directory = tmp_path / "ros2/cais_lab_robotics/config"
    directory.mkdir(parents=True)
    map_path = directory / "recovery_framework_map.pgm"
    map_path.write_bytes(b"P5\n2 1\n255\n\xff\x00")
    (directory / "recovery_framework_map.yaml").write_text("image: recovery_framework_map.pgm\n")
    paths = evaluation._implementation_files(tmp_path)
    assert map_path in paths
    manifest = {"files": evaluation.frozen_files([], root=tmp_path),
                "implementation_files": sorted(str(path.relative_to(tmp_path)) for path in paths)}
    evaluation.verify_frozen(manifest, root=tmp_path)
    if change == "content":
        map_path.write_bytes(b"P5\n2 1\n255\n\x00\x00")
    elif change == "added":
        (directory / "additional_map.pgm").write_bytes(map_path.read_bytes())
    else:
        map_path.unlink()
    expected = "Frozen evaluation input changed" if change == "content" else "inventory changed"
    with pytest.raises(ValueError, match=expected):
        evaluation.verify_frozen(manifest, root=tmp_path)


def test_changed_machine_condition_cannot_be_claimed_from_missing_requirements():
    capture = {"scenario": "Machining breakdown during part processing", "evidence_source": "Gazebo",
               "setup": {"execution_mode": "simulation"},
               "failure": {"injection_status": "completed", "part_name": "KET4_Square_4mm"},
               "inputs": {"scene": {"machines": [{"resource_id": "M2", "current_configuration": {}}]}},
               "requirements": {"KET4_Square_4mm": []}}
    with pytest.raises(ValueError, match="NEEDS_CONTEXT"):
        conditions.verify_capture_condition(capture, capture["scenario"], "changed")


def test_recorded_kinematic_bound_includes_mobile_base_limits():
    robot = '''<robot name="recorded"><joint name="KMR_base_x_joint" type="prismatic">
      <parent link="world"/><child link="base"/><limit lower="-20" upper="20"/>
      </joint><joint name="arm" type="fixed"><parent link="base"/><child link="tcp"/>
      <origin xyz="1 0 0"/></joint></robot>'''
    assert conditions._kinematic_radius(robot) == 21.
    with pytest.raises(ValueError, match="unsupported mobile joint"):
        conditions._kinematic_radius(robot.replace('type="prismatic"', 'type="planar"'))


def test_isolated_configuration_allows_only_removing_declared_programs():
    from cais_spade_llm.ui.recovery_setup import _declared_program_subset

    original = {"machines": [{"resource_id": "M2", "current_configuration": {"program": "circle"},
                             "program_options": {"square": {"program": "square"}}}], "robots": ["ur5e-2"]}
    selected = deepcopy(original)
    selected["machines"][0]["program_options"] = {}
    assert _declared_program_subset(original, selected)
    selected["robots"] = ["ur5e-1"]
    assert not _declared_program_subset(original, selected)
    selected = deepcopy(original)
    selected["machines"][0]["program_options"]["square"] = {"program": "unknown"}
    assert not _declared_program_subset(original, selected)
