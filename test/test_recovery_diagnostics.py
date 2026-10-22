"""Diagnostic records are read and published without changing their JSON values."""

from __future__ import annotations

import json

import pytest

from cais_spade_llm.recovery_framework import diagnostics


def test_write_record_streams_exact_original_json_bytes(tmp_path, monkeypatch):
    value = {"part_name": None, "identifier": "KET4_Square_4mm", "text": "é\n\"quoted\"",
             "values": [False, 0, 1.25, {"missing_part_name": True}]}
    expected = json.dumps(value, indent=2, ensure_ascii=False).encode("utf-8")
    path = tmp_path / "record.json"

    def forbid_complete_encoded_copy(*args, **kwargs):
        raise AssertionError("The diagnostic writer must stream its encoded record")

    monkeypatch.setattr(diagnostics.json, "dumps", forbid_complete_encoded_copy)
    diagnostics.write_record(path, value)

    assert path.read_bytes() == expected
    assert not path.with_suffix(".json.tmp").exists()


def test_write_record_keeps_published_record_when_serialization_fails(tmp_path):
    path = tmp_path / "record.json"
    original = {"part_name": None, "status": "NEEDS_CONTEXT"}
    diagnostics.write_record(path, original)
    before = path.read_bytes()

    with pytest.raises(TypeError, match="not JSON serializable"):
        diagnostics.write_record(path, {"status": "incomplete", "unsupported": object()})

    assert path.read_bytes() == before
    assert diagnostics.read_object(path) == original
    diagnostics.write_record(path, {"status": "ready", "part_name": None})
    assert diagnostics.read_object(path) == {"status": "ready", "part_name": None}
    assert not path.with_suffix(".json.tmp").exists()
