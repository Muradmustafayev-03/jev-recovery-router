"""Linux cloud descendant ownership and bounded cleanup, without real signals."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def supervisor():
    path = Path(__file__).resolve().parents[1] / "scripts/cloud_supervisor.py"
    spec = importlib.util.spec_from_file_location("cloud_supervisor_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup_tracker(supervisor, monkeypatch, snapshot):
    calls = []
    monkeypatch.setattr(supervisor, "os", SimpleNamespace(
        getpid=lambda: 100, getpgrp=lambda: 100, WNOHANG=1,
        waitpid=lambda pid, flags: calls.append(("reap", pid)),
        killpg=lambda pgid, sig: calls.append(("group", pgid, sig)),
        kill=lambda pid, sig: calls.append(("process", pid, sig))))
    monkeypatch.setattr(supervisor, "process_snapshot", lambda: dict(snapshot))
    return supervisor.Descendants(200), calls


def test_proc_stat_parses_spaces_and_parentheses(supervisor, tmp_path):
    folder = tmp_path / "42"
    folder.mkdir()
    fields = ["S", "12", "42"] + ["0"] * 16 + ["9876"]
    (folder / "stat").write_text("42 (name with ) spaces) " + " ".join(fields))
    assert supervisor.process_snapshot(tmp_path)[42] == supervisor.Process(42, 12, 42, 9876, "S")


def test_adopted_stage_survives_launcher_and_unrelated_groups_are_untouched(supervisor, monkeypatch):
    P = supervisor.Process
    snapshot = {100: P(100, 1, 100, 1, "S"), 200: P(200, 100, 200, 2, "S"),
                300: P(300, 200, 300, 3, "S"), 999: P(999, 1, 999, 9, "S")}
    tracker, calls = setup_tracker(supervisor, monkeypatch, snapshot)
    tracker.refresh()
    del snapshot[200]
    snapshot[300] = P(300, 100, 300, 3, "S")
    snapshot[400] = P(400, 300, 400, 4, "S")
    tracker.signal_owned(15)
    assert calls == [("group", 400, 15), ("group", 300, 15)] or calls == [("group", 300, 15), ("group", 400, 15)]
    # A reused PID/PGID with a different start time must never inherit ownership.
    snapshot[300] = P(300, 1, 300, 30, "S")
    del snapshot[400]
    calls.clear()
    assert tracker.signal_owned(9) == {}
    assert calls == []


def test_shared_supervisor_group_signals_only_owned_pid_and_reaps_orphans(supervisor, monkeypatch):
    P = supervisor.Process
    snapshot = {100: P(100, 1, 100, 1, "S"), 200: P(200, 100, 200, 2, "Z"),
                300: P(300, 100, 100, 3, "S"), 400: P(400, 100, 400, 4, "Z")}
    tracker, calls = setup_tracker(supervisor, monkeypatch, snapshot)
    tracker.signal_owned(15)
    assert ("process", 300, 15) in calls
    assert ("reap", 400) in calls
    assert ("reap", 200) not in calls
    assert not any(call[0] == "group" for call in calls)


def test_force_kill_tracks_separate_stage_group_after_shell_has_exited(supervisor, monkeypatch):
    P = supervisor.Process
    snapshot = {100: P(100, 1, 100, 1, "S"), 300: P(300, 100, 300, 3, "S")}
    tracker, calls = setup_tracker(supervisor, monkeypatch, snapshot)
    monkeypatch.setattr(supervisor, "signal", SimpleNamespace(SIGTERM=15, SIGKILL=9))
    def killpg(group, sig):
        calls.append(("group", group, sig))
        if sig == 9:
            del snapshot[300]
    supervisor.os.killpg = killpg
    result = tracker.stop(SimpleNamespace(poll=lambda: 0), grace_seconds=0, kill_seconds=0)
    assert calls == [("group", 300, 15), ("group", 300, 9)]
    assert result == {"status": "complete", "surviving_processes": []}


def test_force_kill_reports_verified_survivors(supervisor, monkeypatch):
    P = supervisor.Process
    snapshot = {100: P(100, 1, 100, 1, "S"), 300: P(300, 100, 300, 3, "D")}
    tracker, calls = setup_tracker(supervisor, monkeypatch, snapshot)
    monkeypatch.setattr(supervisor, "signal", SimpleNamespace(SIGTERM=15, SIGKILL=9))
    result = tracker.stop(SimpleNamespace(poll=lambda: 0), grace_seconds=0, kill_seconds=0)
    assert result["status"] == "incomplete"
    assert result["surviving_processes"] == [snapshot[300]._asdict()]
