"""Tests for #3 US1/US3: rejection echoes received arg keys + forensics log on dropped args."""

from unittest.mock import patch

import pytest

from tools.tool_search_validation import (
    normalize_tool_call_entries,
    validate_deferred_call_args,
)

REQUIRED_SCHEMA = {
    "type": "function",
    "function": {
        "name": "mcp__srv__worker",
        "parameters": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "agent_id": {"type": "string"},
            },
            "required": ["task_id", "agent_id"],
        },
    },
}

NO_REQUIRED_SCHEMA = {
    "type": "function",
    "function": {
        "name": "mcp__srv__ping",
        "parameters": {"type": "object", "properties": {}},
    },
}


def _fake_registry():
    class FakeRegistry:
        def get_schema(self, name):
            return {"mcp__srv__worker": REQUIRED_SCHEMA,
                    "mcp__srv__ping": NO_REQUIRED_SCHEMA}.get(name)

    return FakeRegistry()


@pytest.fixture()
def fake_registry():
    with patch("tools.registry.registry", _fake_registry()):
        yield


# --- US1: rejection message echoes received argument keys ---

def test_valid_call_passes(fake_registry):
    assert validate_deferred_call_args(
        "mcp__srv__worker", {"task_id": "t1", "agent_id": "a1"}) is None


def test_empty_args_echoes_empty_key_list(fake_registry):
    msg = validate_deferred_call_args("mcp__srv__worker", {})
    assert msg is not None
    assert "missing required argument(s): task_id, agent_id" in msg
    assert "received argument keys: []" in msg


def test_partial_args_echoes_received_keys(fake_registry):
    msg = validate_deferred_call_args("mcp__srv__worker", {"task_id": "t1"})
    assert msg is not None
    assert "received argument keys: [task_id]" in msg


# --- US3: forensic log line on the drop path ---

def test_forensics_fires_once_per_absent_args(fake_registry, caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="hermes.tools.args_forensics"):
        entries, err = normalize_tool_call_entries(
            {"calls": [{"name": "mcp__srv__worker"}]})
    assert err is None and entries[0]["arguments"] == {}
    records = [r for r in caplog.records
               if r.name == "hermes.tools.args_forensics"]
    assert len(records) == 1
    rec = records[0].getMessage()
    assert "mcp__srv__worker" in rec
    assert "outcome=absent" in rec
    assert "task_id" in rec  # names the required params that were dropped


def test_forensics_fires_on_unparseable_args(fake_registry, caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="hermes.tools.args_forensics"):
        entries, err = normalize_tool_call_entries(
            {"calls": [{"name": "mcp__srv__worker", "arguments": "{not json"}]})
    assert err is not None and "not valid JSON" in err
    records = [r for r in caplog.records
               if r.name == "hermes.tools.args_forensics"]
    assert len(records) == 1
    assert "outcome=unparseable" in records[0].getMessage()


def test_forensics_silent_for_tool_without_required_params(fake_registry, caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="hermes.tools.args_forensics"):
        entries, err = normalize_tool_call_entries(
            {"calls": [{"name": "mcp__srv__ping"}]})
    assert err is None and entries
    assert not [r for r in caplog.records
                if r.name == "hermes.tools.args_forensics"]


def test_forensics_truncates_raw_repr(fake_registry, caplog):
    import logging
    big = "x" * 5000
    with caplog.at_level(logging.WARNING, logger="hermes.tools.args_forensics"):
        normalize_tool_call_entries(
            {"calls": [{"name": "mcp__srv__worker", "arguments": " " + big}]})
    records = [r for r in caplog.records
               if r.name == "hermes.tools.args_forensics"]
    assert len(records) == 1
    assert len(records[0].getMessage()) < 700  # single short line, no stack trace
