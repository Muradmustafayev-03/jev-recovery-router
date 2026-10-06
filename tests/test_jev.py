import json
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import threading
import time
from types import SimpleNamespace

import pytest

from recovery_router.jev import BudgetExceeded, JevClient, build_request, iter_predictions, predict_record


def record(identifier="one"):
    candidates = [{"id": "alpha", "label": "alpha", "description": "first meaning"},
                  {"id": "beta", "label": "beta", "description": "second meaning"}]
    return {"id": identifier, "text": "test query", "label": "SECRET_GOLD",
            "candidates": candidates[:1], "expanded_candidates": candidates}


class FakeHTTP:
    def __init__(self, wrong_model=False, status=200):
        self.calls = []
        self.wrong_model = wrong_model
        self.status = status

    def __call__(self, payload, key, timeout):
        self.calls.append(payload)
        keys = list(payload["questions"]["decision"]["criteria"])
        probability = {k: .2 / (len(keys)-1) for k in keys[:-1]}
        probability["NONE"] = .8
        # Deliberately reverse response dictionary order.
        return self.status, {"model": "jev-9.9.9" if self.wrong_model else payload["model"],
                             "usage": {"input_tokens": 10}, "answers": {"decision": {
                                 "probabilities": dict(reversed(list(probability.items())))}}}


def test_separate_requests_none_order_and_cache(tmp_path):
    fake = FakeHTTP()
    ledger = tmp_path / "cache.jsonl"
    client = JevClient(ledger, api_key="DO_NOT_PERSIST_KEY", transport=fake)
    prediction = predict_record(client, record())
    assert len(fake.calls) == 2
    assert "SECRET_GOLD" not in json.dumps(fake.calls)
    assert len(fake.calls[0]["questions"]["decision"]["criteria"]) == 2
    assert len(fake.calls[1]["questions"]["decision"]["criteria"]) == 3
    assert prediction["first_probs"] == [.2, .8]
    assert prediction["expanded_probs"] == [.1, .1, .8]
    assert prediction["model_fingerprint"] == "api-version:jev-1.13.0:choice-protocol-v1"
    assert "DO_NOT_PERSIST_KEY" not in ledger.read_text()
    client.close()
    resumed = JevClient(ledger, api_key=None, transport=fake)
    second = predict_record(resumed, record("duplicate-observation"))
    assert second["id"] == "duplicate-observation"
    assert second["first_cache_hit"]
    assert len(fake.calls) == 2


def test_cost_bound_before_send_and_resume(tmp_path):
    fake = FakeHTTP()
    ledger = tmp_path / "cache.jsonl"
    client = JevClient(ledger, api_key="key", transport=fake, max_usd=.002688)
    client.request(build_request(record()))
    assert client.reserved_usd == Decimal("0.00000042")
    with pytest.raises(BudgetExceeded):
        client.request(build_request(record(), True))
    assert len(fake.calls) == 1
    client.close()
    resumed = JevClient(ledger, api_key="key", transport=fake, max_usd=.002688)
    with pytest.raises(BudgetExceeded):
        resumed.request(build_request(record(), True))
    assert len(fake.calls) == 1


def test_model_mismatch_blocks_retry_and_keeps_reservation(tmp_path):
    fake = FakeHTTP(wrong_model=True)
    client = JevClient(tmp_path / "cache.jsonl", api_key="key", transport=fake)
    with pytest.raises(RuntimeError):
        client.request(build_request(record()))
    assert client.request_count == 1
    assert client.reserved_usd == Decimal("0.002688")
    with pytest.raises(RuntimeError, match="automatic retry"):
        client.request(build_request(record()))
    assert len(fake.calls) == 1


def test_no_key_or_request_allowance_no_network(tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    fake = FakeHTTP()
    client = JevClient(tmp_path / "no-key.jsonl", transport=fake)
    with pytest.raises(RuntimeError, match="absent"):
        client.request(build_request(record()))
    client = JevClient(tmp_path / "no-requests.jsonl", api_key="key", transport=fake, max_requests=0)
    with pytest.raises(BudgetExceeded):
        client.request(build_request(record()))
    assert not fake.calls


@pytest.mark.parametrize("max_retries", [0, 3])
def test_uncertain_transport_error_is_never_retried(tmp_path, max_retries):
    calls = []
    def uncertain(payload, key, timeout):
        calls.append(payload)
        raise TimeoutError("sensitive transport information")
    ledger = tmp_path / "cache.jsonl"
    client = JevClient(ledger, api_key="key", transport=uncertain, max_retries=max_retries)
    with pytest.raises(RuntimeError):
        client.request(build_request(record()))
    client.close()
    resumed = JevClient(ledger, api_key="key", transport=uncertain, max_retries=max_retries)
    assert resumed.reserved_usd == Decimal("0.002688")
    with pytest.raises(RuntimeError):
        resumed.request(build_request(record()))
    assert len(calls) == 1
    assert "sensitive transport information" not in ledger.read_text()


def test_success_releases_unused_reservation(tmp_path):
    fake = FakeHTTP()
    ledger = tmp_path / "cache.jsonl"
    with JevClient(ledger, api_key="key", transport=fake, max_usd=.003) as client:
        # Two worst-case reservations would exceed .003, but validated actual usage is tiny.
        prediction = predict_record(client, record())
        assert len(fake.calls) == 2
        assert client.reserved_usd == Decimal("0.00000084")
    with JevClient(ledger, api_key="key", transport=fake, max_usd=.003) as resumed:
        assert resumed.reserved_usd == Decimal("0.00000084")
        assert predict_record(resumed, record())["expanded_probs"] == prediction["expanded_probs"]
    assert len(fake.calls) == 2


def test_ledger_lock_prevents_concurrent_budget_races(tmp_path):
    ledger = tmp_path / "cache.jsonl"
    first = JevClient(ledger, api_key="key", transport=FakeHTTP())
    with pytest.raises(RuntimeError, match="Another process"):
        JevClient(ledger, api_key="key", transport=FakeHTTP())
    first.close()
    with JevClient(ledger, api_key="key", transport=FakeHTTP()) as second:
        assert second.request_count == 0
    with pytest.raises(RuntimeError, match="closed"):
        second.request(build_request(record()))


def distinct_record(index):
    row = record(str(index))
    row["text"] = f"query number {index}"
    return row


def test_concurrent_requests_reserve_before_send_and_share_cache(tmp_path):
    fake = FakeHTTP()
    barrier = threading.Barrier(4)
    counts_lock = threading.Lock()
    active = maximum_active = 0

    def transport(payload, key, timeout):
        nonlocal active, maximum_active
        with counts_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        barrier.wait(timeout=10)
        response = fake(payload, key, timeout)
        with counts_lock:
            active -= 1
        return response

    ledger = tmp_path / "concurrent.jsonl"
    with JevClient(ledger, api_key="key", transport=transport, workers=4) as client:
        payloads = [build_request(distinct_record(i)) for i in range(4)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(client.request, payloads))
        assert maximum_active == 4
        assert client.request_count == 4
        assert client.reserved_usd == Decimal("0.00000168")
        with ThreadPoolExecutor(max_workers=4) as pool:
            cached = list(pool.map(client.request, [payloads[0]] * 4))
        assert all(result["cache_hit"] for result in cached)
        assert len(fake.calls) == 4
        assert all(not result["cache_hit"] for result in results)
    with JevClient(ledger, api_key="key", transport=fake) as resumed:
        assert resumed.request_count == 4
        assert resumed.reserved_usd == Decimal("0.00000168")


def test_duplicate_inflight_requests_only_send_once(tmp_path):
    fake = FakeHTTP()
    started, release = threading.Event(), threading.Event()

    def transport(payload, key, timeout):
        started.set()
        assert release.wait(timeout=10)
        return fake(payload, key, timeout)

    with JevClient(tmp_path / "dedup.jsonl", api_key="key", transport=transport, workers=4) as client:
        payload = build_request(record())
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(client.request, payload) for _ in range(4)]
            assert started.wait(timeout=10)
            release.set()
            results = [future.result(timeout=10) for future in futures]
        assert len(fake.calls) == 1
        assert sum(not result["cache_hit"] for result in results) == 1


def test_inflight_reservation_blocks_concurrent_overspend(tmp_path):
    fake = FakeHTTP()
    started, release = threading.Event(), threading.Event()

    def transport(payload, key, timeout):
        started.set()
        assert release.wait(timeout=10)
        return fake(payload, key, timeout)

    with JevClient(tmp_path / "cap.jsonl", api_key="key", transport=transport,
                   workers=2, max_usd=.003) as client:
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(client.request, build_request(distinct_record(1)))
            try:
                assert started.wait(timeout=10)
                with pytest.raises(BudgetExceeded):
                    client.request(build_request(distinct_record(2)))
                assert client.request_count == 1
                assert client.reserved_usd == Decimal("0.002688")
            finally:
                release.set()
            first.result(timeout=10)
        assert len(fake.calls) == 1


def test_retries_retain_each_failed_reservation_and_resume(tmp_path):
    fake = FakeHTTP()
    calls, delays = [], []

    def transport(payload, key, timeout):
        calls.append(payload)
        if len(calls) <= 2:
            return (429 if len(calls) == 1 else 529), {"retry_after_seconds": 2.}
        return fake(payload, key, timeout)

    ledger = tmp_path / "retries.jsonl"
    with JevClient(ledger, api_key="key", transport=transport, max_retries=2) as client:
        client._wait = delays.append
        result = client.request(build_request(record()))
        assert result["probabilities"] == [.2, .8]
        assert len(calls) == 3
        assert calls[0] == calls[1] == calls[2]
        assert delays == [2., 2.]
        assert client.request_count == 3
        assert client.reserved_usd == Decimal("0.00537642")
    with JevClient(ledger, api_key="key", transport=transport) as resumed:
        assert resumed.request_count == 3
        assert resumed.reserved_usd == Decimal("0.00537642")
        assert resumed.request(build_request(record()))["cache_hit"]
    assert len(calls) == 3


def test_retry_rechecks_budget_and_can_resume_known_rejection(tmp_path):
    fake = FakeHTTP(status=429)
    ledger = tmp_path / "retry-cap.jsonl"
    with JevClient(ledger, api_key="key", transport=fake, max_retries=1, max_usd=.003) as client:
        client._wait = lambda _: None
        with pytest.raises(BudgetExceeded):
            client.request(build_request(record()))
        assert len(fake.calls) == 1
        assert client.reserved_usd == Decimal("0.002688")
    fake.status = 200
    with JevClient(ledger, api_key="key", transport=fake, max_retries=1, max_usd=.006) as client:
        client._wait = lambda _: None
        client.request(build_request(record()))
        assert client.request_count == 2
        assert client.reserved_usd == Decimal("0.00268842")
    assert len(fake.calls) == 2


@pytest.mark.parametrize("status", [401, 422, 500])
def test_nonretryable_http_errors_stop_and_keep_status(tmp_path, status):
    fake = FakeHTTP(status=status)
    ledger = tmp_path / "status.jsonl"
    with JevClient(ledger, api_key="key", transport=fake, max_retries=3) as client:
        with pytest.raises(RuntimeError, match=f"HTTP {status}"):
            client.request(build_request(record()))
        assert len(fake.calls) == 1
    failure = json.loads(ledger.read_text().splitlines()[-1])
    assert failure["http_status"] == status


def test_retry_count_is_bounded_across_resume(tmp_path):
    fake = FakeHTTP(status=429)
    ledger = tmp_path / "retry-limit.jsonl"
    with JevClient(ledger, api_key="key", transport=fake, max_retries=2) as client:
        client._wait = lambda _: None
        with pytest.raises(RuntimeError, match="HTTP 429"):
            client.request(build_request(record()))
        assert client.request_count == 3
        assert client.reserved_usd == Decimal("0.008064")
    with JevClient(ledger, api_key="key", transport=fake, max_retries=2) as resumed:
        with pytest.raises(RuntimeError, match="exhausted"):
            resumed.request(build_request(record()))
    assert len(fake.calls) == 3


def test_global_request_pacing_applies_to_all_workers(tmp_path):
    fake = FakeHTTP()
    starts = []

    def transport(payload, key, timeout):
        starts.append(time.monotonic())
        return fake(payload, key, timeout)

    with JevClient(tmp_path / "rate.jsonl", api_key="key", transport=transport,
                   workers=4, requests_per_second=20) as client:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(client.request, [build_request(distinct_record(i)) for i in range(4)]))
    assert all(later - earlier >= .045 for earlier, later in zip(starts, starts[1:]))


def test_pacing_preserves_spacing_after_delayed_transport_entry(tmp_path, monkeypatch):
    """A worker delayed after admission must not spend its successor's gap."""
    import recovery_router.jev as jev

    clock = [0.]
    starts = []
    fake = FakeHTTP()
    monkeypatch.setattr(jev, "time", SimpleNamespace(
        monotonic=lambda: clock[0], perf_counter=time.perf_counter))

    def transport(payload, key, timeout):
        starts.append(clock[0])
        return fake(payload, key, timeout)

    with JevClient(tmp_path / "late-dispatch.jsonl", api_key="key", transport=transport,
                   workers=4, requests_per_second=20) as client:
        original_pace = client._pace

        def delayed_pace():
            original_pace()
            if not starts:
                # Deterministic stand-in for descheduling between admission
                # and entry into the supplied synchronous transport.
                clock[0] += 10.

        def advance(seconds):
            clock[0] += seconds

        client._pace = delayed_pace
        client._wait = advance
        for i in range(3):
            client.request(build_request(distinct_record(i)))
    assert starts == pytest.approx([10., 10.05, 10.10])


def test_positive_rate_serializes_transport_without_blocking_reservations(tmp_path):
    fake = FakeHTTP()
    entered = threading.Event()
    release = threading.Event()
    second_reserved = threading.Event()
    starts = []

    def transport(payload, key, timeout):
        starts.append(time.monotonic())
        if len(starts) == 1:
            entered.set()
            assert release.wait(timeout=10)
        return fake(payload, key, timeout)

    with JevClient(tmp_path / "paced-serialization.jsonl", api_key="key", transport=transport,
                   workers=2, requests_per_second=20) as client:
        original_append = client._append

        def append(event):
            original_append(event)
            if event.get("event") == "reserved" and event.get("request_hash") != first_hash:
                second_reserved.set()

        first_payload = build_request(distinct_record(0))
        from recovery_router.jev import canonical_hash
        first_hash = canonical_hash(first_payload)
        client._append = append
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(client.request, first_payload)
            try:
                assert entered.wait(timeout=10)
                second = pool.submit(client.request, build_request(distinct_record(1)))
                assert second_reserved.wait(timeout=10)
                assert len(starts) == 1
            finally:
                release.set()
            first.result(timeout=10)
            second.result(timeout=10)
    assert len(starts) == 2
    assert starts[1] - starts[0] >= .045


def test_parallel_predictions_preserve_input_order_and_protocol(tmp_path):
    fake = FakeHTTP()
    rows = [distinct_record(i) for i in range(12)]
    with JevClient(tmp_path / "order.jsonl", api_key="key", transport=fake, workers=4) as client:
        predictions = list(iter_predictions(client, rows))
    assert [prediction["id"] for prediction in predictions] == [row["id"] for row in rows]
    assert all(prediction["first_probs"] == [.2, .8] for prediction in predictions)
    assert all(prediction["expanded_probs"] == [.1, .1, .8] for prediction in predictions)
    assert all(prediction["model_fingerprint"].endswith(":choice-protocol-v1") for prediction in predictions)
    assert all("4 concurrent requests" in prediction["timing_scope"] for prediction in predictions)
    assert len(fake.calls) == 24


def test_parallel_terminal_failure_stops_new_records(tmp_path):
    fake = FakeHTTP(status=401)
    with JevClient(tmp_path / "stop.jsonl", api_key="key", transport=fake, workers=4) as client:
        with pytest.raises(RuntimeError, match="HTTP 401"):
            list(iter_predictions(client, [distinct_record(i) for i in range(100)]))
        assert 1 <= len(fake.calls) <= 4
        assert client.reserved_usd <= 4 * Decimal("0.002688")
