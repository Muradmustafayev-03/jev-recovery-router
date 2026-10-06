"""Offline checks for real-online measurement and optional paid-run planning."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from recovery_router.policy import Costs


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def benchmark():
    return load_script("benchmark_online")


@pytest.fixture
def jev_plan(tmp_path):
    script = load_script("run_jev_comparison")
    data = tmp_path / "data"
    data.mkdir()
    candidates = [{"id": f"intent:{i}", "label": f"intent_{i}"} for i in range(5)]
    for split in script.FIT + script.TEST:
        row = {"id": split, "text": f"Query for {split}", "label": "not_read_as_a_target",
               "candidates": candidates[:1], "expanded_candidates": candidates}
        (data / (split + ".jsonl")).write_text(json.dumps(row) + "\n", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"budget": {"max_usd": 100., "max_elapsed_hours": 12.,
                                             "gpu_usd_per_hour": 0., "assumed_tax_multiplier": 1.}}))
    args = SimpleNamespace(run=str(tmp_path / "run"), data_dir=str(data), config=str(config),
                           model="jev-1.13.0", max_usd=1., max_requests=12,
                           max_seconds=10, prior_spend_usd=0., input_price_per_million=.042)
    return script, args


def cli_args(args):
    return ["--run", args.run, "--data-dir", args.data_dir, "--config", args.config,
            "--max-usd", str(args.max_usd), "--max-requests", str(args.max_requests),
            "--max-seconds", str(args.max_seconds)]


def test_selection_uses_file_order_and_discards_targets(benchmark, tmp_path):
    path = tmp_path / "queries.jsonl"
    rows = [{"id": "z", "text": "First", "label": "wrong", "candidates": ["secret"]},
            {"id": "a", "text": "Second", "label": "right"}, {"id": "b", "text": "Third"}]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    assert benchmark.select_queries(path, 2) == [{"id": "z", "text": "First"}, {"id": "a", "text": "Second"}]
    with pytest.raises(ValueError, match="only 3"):
        benchmark.select_queries(path, 4)


def test_measurement_calls_true_route_once_per_query_plus_warmup(benchmark):
    calls, syncs = [], []
    def route(text):
        calls.append(text)
        return {"action": "commit" if text == "First" else "handoff", "expanded": text == "Second",
                "candidate_id": "some_id" if text == "First" else None, "stage_traces": []}
    report = benchmark.measure_queries([{"id": "one", "text": "First"}, {"id": "two", "text": "Second"}],
                                       route, lambda: syncs.append(True), warmup=3)
    assert calls == ["First", "Second", "First", "First", "Second"]
    assert len(syncs) == 10
    assert report["timing_status"] == "measured_sequential_online_latency"
    assert report["action_counts"] == {"commit": 1, "handoff": 1}
    assert report["expanded_requests"] == 1 and report["batch_size"] == 1
    assert 0 <= report["min_ms"] <= report["p50_ms"] <= report["p95_ms"] <= report["max_ms"]
    assert "accuracy" not in report
    json.dumps(report, allow_nan=False)


def test_direct_baseline_uses_one_full_retrieval_and_one_expanded_forecast(benchmark):
    calls = []
    class Retriever:
        def retrieve(self, text, k):
            calls.append(("retrieve", k))
            return [{"id": f"i{i}"} for i in range(k)]
    def forecast(row, expanded):
        assert "candidates" not in row and "label" not in row
        assert expanded is True
        calls.append(("forecast", len(row["expanded_candidates"])))
        return {"probs": [.12] * 5 + [.4]}
    result = benchmark.route_direct("Request", Retriever(), forecast,
                                    {"expanded_temperature": 1.}, Costs(), expanded_k=5)
    assert calls == [("retrieve", 5), ("forecast", 5)]
    assert result["action"] == "handoff" and result["expanded"] is False
    assert result["confidence"] == pytest.approx(.12)
    assert result["stage_traces"][0]["none_probability"] == pytest.approx(.4)
    # Lower wrong-action cost changes the analytical decision without a first pass.
    result = benchmark.route_direct("Request", Retriever(), forecast,
                                    {"expanded_temperature": 1.}, Costs(wrong=.01), expanded_k=5)
    assert result["action"] == "commit"


def test_jev_all_splits_share_budget_and_fitting_precedes_tests(jev_plan):
    script, args = jev_plan
    plan = script.prepare_plan(args)
    assert plan["status"] == "plan_only_not_run"
    assert plan["unique_requests_before_cache"] == 12
    stages = plan["stages"]
    assert len(stages) == 9 and stages[3]["name"] == "jev_fit_and_seal"
    assert stages[4]["name"] == "jev_predict_test_base"
    predictions = [s["command"] for s in stages if s["name"].startswith("jev_predict_")]
    for cmd in predictions:
        assert cmd[cmd.index("--cache") + 1] == plan["shared_api_ledger"]
        assert cmd[cmd.index("--max-usd") + 1] == "1.0"
        assert cmd[cmd.index("--max-requests") + 1] == "12"
    assert not Path(args.run).exists()


def test_jev_parallel_options_reach_every_prediction_stage(jev_plan):
    script, args = jev_plan
    args.workers, args.requests_per_second, args.max_retries = 4, 8., 3
    plan = script.prepare_plan(args)
    assert (plan["workers"], plan["requests_per_second"], plan["max_retries"]) == (4, 8., 3)
    for stage in plan["stages"]:
        if stage["name"].startswith("jev_predict_"):
            command = stage["command"]
            assert command[command.index("--workers") + 1] == "4"
            assert command[command.index("--requests-per-second") + 1] == "8.0"
            assert command[command.index("--max-retries") + 1] == "3"


@pytest.mark.parametrize("field,value,match", [
    ("max_usd", float("nan"), "finite"), ("max_usd", .000001, "reservation"),
    ("max_requests", 11, "distinct uncached"), ("max_seconds", 0, "positive"),
    ("model", "jev", "pinned"), ("prior_spend_usd", 100., "total budget"),
])
def test_jev_rejects_invalid_or_inadequate_limits(jev_plan, field, value, match):
    script, args = jev_plan
    setattr(args, field, value)
    with pytest.raises(ValueError, match=match):
        script.prepare_plan(args)


def test_jev_requires_every_split_before_planning(jev_plan):
    script, args = jev_plan
    (Path(args.data_dir) / "transfer.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        script.prepare_plan(args)


def test_dry_run_needs_no_key_and_never_executes(jev_plan, monkeypatch, capsys):
    script, args = jev_plan
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(script.subprocess, "run", lambda *a, **kw: pytest.fail("No subprocess during dry-run"))
    script.main(cli_args(args) + ["--dry-run"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["status"] == "plan_only_not_run" and len(plan["stages"]) == 9
    assert not Path(args.run).exists()


@pytest.mark.parametrize("flags", [[], ["--execute-paid-api"]])
def test_paid_execution_needs_both_flag_and_key(jev_plan, monkeypatch, flags):
    script, args = jev_plan
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(script.subprocess, "run", lambda *a, **kw: pytest.fail("No execution without explicit paid mode and key"))
    with pytest.raises(SystemExit):
        script.main(cli_args(args) + flags)
    assert not Path(args.run).exists()
