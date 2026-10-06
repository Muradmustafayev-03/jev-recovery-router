"""Bounded cleanup and incomplete reporting for interrupted GPU runs."""
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / "scripts/run_experiment.py"
    spec = importlib.util.spec_from_file_location("run_experiment_cleanup", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("interrupt", [False, True])
def test_failure_stops_child_and_preserves_original_error(runner, tmp_path, monkeypatch, interrupt):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"budget": {"max_elapsed_hours": 12, "max_usd": 100,
                                            "assumed_tax_multiplier": 1.2, "incidentals_usd": 5,
                                            "gpu_usd_per_hour": 3.18}}))
    output = tmp_path / "run"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_experiment.py", "--config", str(config), "--output", str(output)])
    monkeypatch.setattr(runner, "run_signature", lambda *args: {})
    monkeypatch.setattr(runner, "build_stages", lambda *args: [("fixture_stage", ["fixture"])])
    calls = []

    class Child:
        returncode = 3

        def poll(self):
            if interrupt:
                signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return self.returncode

    child = Child()
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: child)
    monkeypatch.setattr(runner, "stop_child", lambda value: calls.append(("stop", value)))

    def aggregate(command, **kwargs):
        calls.append(("aggregate", command))
        assert kwargs["timeout"] == 30
        assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(runner.subprocess, "run", aggregate)
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    expected = KeyboardInterrupt if interrupt else RuntimeError
    with pytest.raises(expected, match="SIGTERM" if interrupt else "fixture_stage failed"):
        runner.main()
    ledger = json.loads((output / "run_ledger.json").read_text())
    assert ledger["status"] == "incomplete"
    assert ledger["error_type"] == expected.__name__
    assert ledger["failure_aggregation"]["error_type"] == "TimeoutExpired"
    assert [call[0] for call in calls] == ["stop", "aggregate"]
    assert calls[0][1] is child
    assert all(signal.getsignal(sig) == handler for sig, handler in previous.items())


def test_stop_child_kills_and_reaps_unresponsive_process_group(runner, monkeypatch):
    signals, waits = [], []
    monkeypatch.setattr(runner, "os", SimpleNamespace(name="posix", killpg=lambda pid, sig: signals.append((pid, sig))))
    monkeypatch.setattr(runner, "signal", SimpleNamespace(SIGTERM=signal.SIGTERM, SIGKILL=9))

    def wait(timeout):
        waits.append(timeout)
        if len(waits) == 1:
            raise subprocess.TimeoutExpired("fixture", timeout)
        return -9

    child = SimpleNamespace(pid=123, poll=lambda: None, wait=wait)
    runner.stop_child(child)
    assert signals == [(123, signal.SIGTERM), (123, 9)]
    assert waits == [20, 5]


def test_stop_child_accepts_process_exit_race(runner, monkeypatch):
    def vanished(*args):
        raise ProcessLookupError("already exited")

    monkeypatch.setattr(runner, "os", SimpleNamespace(name="posix", killpg=vanished))
    runner.stop_child(SimpleNamespace(pid=123, poll=lambda: None))
