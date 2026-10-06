"""Safe response diagnostics and charged, explicitly bounded recovery."""
import json
from decimal import Decimal

import pytest

from recovery_router.jev import BudgetExceeded, JevClient, build_request, canonical_hash
from test_jev import FakeHTTP, record


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def legacy_failure(path, error_type="ValueError", status=200):
    payload = build_request(record())
    with JevClient(path, api_key="key", transport=FakeHTTP()) as client:
        settings, reservation = client.settings, str(client.reservation)
    failure = {"event": "failed", "request_hash": canonical_hash(payload), "error_type": error_type}
    if status is not None:
        failure["http_status"] = status
    path.write_text("\n".join(json.dumps(event) for event in [settings,
        {"event": "reserved", "request_hash": canonical_hash(payload), "reserved_usd": reservation}, failure]) + "\n")
    return payload


def test_validation_retry_journals_safe_diagnostics_and_every_charge(tmp_path):
    fake, calls = FakeHTTP(), []
    def transport(payload, key, timeout):
        calls.append(payload)
        status, response = fake(payload, key, timeout)
        if len(calls) == 1:
            response["answers"]["decision"]["probabilities"].update(A=.3, SENSITIVE_UNEXPECTED_KEY="SECRET")
            response.update(headers={"Authorization": "SECRET"}, body="SECRET", request="SECRET")
            response["usage"]["arbitrary"] = "SECRET"
        return status, response
    path = tmp_path / "ledger.jsonl"
    with JevClient(path, api_key="SECRET", transport=transport, max_validation_retries=2) as client:
        result = client.request(build_request(record()))
        assert result["probabilities"] == [.2, .8]
        assert result["attempt_count"] == 2
        assert result["rejection_counts"] == {"validation": 1}
        assert client.reserved_usd == Decimal("0.00268842")
    journal = events(path)
    rejected = journal[2]
    assert rejected["http_status"] == 200 and rejected["request_dispatched"] is True
    assert rejected["validation"]["reason"] == "choice_keys"
    assert rejected["validation"]["probabilities"] == {"A": .3, "NONE": .8}
    assert rejected["validation"]["unexpected_choice_count"] == 1
    assert journal[3]["retry_reason"] == "validation"
    assert "SECRET" not in path.read_text() and "SENSITIVE_UNEXPECTED_KEY" not in path.read_text()
    with JevClient(path, api_key="key", transport=fake) as resumed:
        assert resumed.reserved_usd == Decimal("0.00268842")
        assert resumed.request(build_request(record()))["cache_hit"]
    assert len(calls) == 2


@pytest.mark.parametrize("invalid", [None, [], {"model": "SECRET"},
    {"model": "jev-1.13.0", "answers": {"decision": {"probabilities": {"A": "SECRET", "NONE": .8}}}},
    {"model": "jev-1.13.0", "answers": {"decision": {"probabilities": {"A": 10**400, "NONE": .8}}}}])
def test_malformed_validation_stays_strict_and_diagnostics_never_copy_text(tmp_path, invalid):
    path = tmp_path / "bad.jsonl"
    with JevClient(path, api_key="key", transport=lambda *args: (200, invalid)) as client:
        with pytest.raises(RuntimeError):
            client.request(build_request(record()))
        assert client.request_count == 1
    failure = events(path)[2]
    assert failure["validation"]["reason"]
    assert "SECRET" not in path.read_text()


def test_validation_retry_limit_survives_repeated_resume(tmp_path):
    path = tmp_path / "bad.jsonl"
    fake = FakeHTTP(wrong_model=True)
    for _ in range(2):
        with JevClient(path, api_key="key", transport=fake, max_validation_retries=2) as client:
            with pytest.raises(RuntimeError):
                client.request(build_request(record()))
            assert client.request_count == 3
            assert client.reserved_usd == Decimal("0.008064")
    assert len(fake.calls) == 3


def test_validation_retry_reserves_fresh_cost_before_dispatch(tmp_path):
    path = tmp_path / "bounded.jsonl"
    fake = FakeHTTP(wrong_model=True)
    with JevClient(path, api_key="key", transport=fake, max_validation_retries=2, max_usd=.003) as client:
        with pytest.raises(BudgetExceeded):
            client.request(build_request(record()))
        assert client.request_count == 1
        assert client.reserved_usd == Decimal("0.002688")
    assert len(fake.calls) == 1


def test_legacy_http200_validation_can_retry_with_new_reservation(tmp_path):
    path = tmp_path / "legacy.jsonl"
    payload = legacy_failure(path)
    original = path.read_bytes()
    with JevClient(path, api_key="key", transport=FakeHTTP()) as client:
        with pytest.raises(RuntimeError, match="exhausted"):
            client.request(payload)
    with JevClient(path, api_key="key", transport=FakeHTTP(), max_validation_retries=2) as client:
        assert client.request(payload)["attempt_count"] == 2
        assert client.reserved_usd == Decimal("0.00268842")
    assert path.read_bytes().startswith(original)
    assert events(path)[3]["retry_reason"] == "validation"


def test_legacy_no_status_requires_explicit_flag_and_discloses_uncertainty(tmp_path):
    path = tmp_path / "legacy.jsonl"
    payload = legacy_failure(path, "RuntimeError", None)
    with JevClient(path, api_key="key", transport=FakeHTTP(), max_validation_retries=2) as client:
        with pytest.raises(RuntimeError):
            client.request(payload)
    with JevClient(path, api_key="key", transport=FakeHTTP(), retry_legacy_cancelled=True) as client:
        result = client.request(payload)
        assert result["rejection_counts"] == {"legacy_unknown_dispatch_runtime_error": 1}
        assert client.request_count == 2
        assert client.reserved_usd == Decimal("0.00268842")
    assert events(path)[3]["retry_reason"] == "legacy_unknown_dispatch_runtime_error"


def test_dispatched_unknown_runtime_error_remains_terminal_even_with_recovery_flags(tmp_path):
    path = tmp_path / "transport.jsonl"
    calls = []
    def transport(*args):
        calls.append(1)
        raise RuntimeError("SECRET")
    for _ in range(2):
        with JevClient(path, api_key="key", transport=transport, max_validation_retries=2,
                       retry_legacy_cancelled=True, max_retries=3) as client:
            with pytest.raises(RuntimeError):
                client.request(build_request(record()))
    assert len(calls) == 1
    assert events(path)[2]["request_dispatched"] is True
    assert "SECRET" not in path.read_text()


def test_predispatch_cancellation_has_only_one_explicit_recovery_across_resume(tmp_path):
    path = tmp_path / "cancelled.jsonl"
    fake = FakeHTTP()
    def cancelled():
        raise RuntimeError("Jev request scheduling stopped")
    for _ in range(3):
        with JevClient(path, api_key="key", transport=fake, retry_legacy_cancelled=True) as client:
            client._pace = cancelled
            with pytest.raises(RuntimeError):
                client.request(build_request(record()))
    assert not fake.calls
    assert len([event for event in events(path) if event["event"] == "reserved"]) == 2
    assert all(event["request_dispatched"] is False for event in events(path) if event["event"] == "failed")
