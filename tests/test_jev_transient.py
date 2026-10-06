"""Explicit transient HTTP recovery keeps historical failures and shared caps."""
import json
from decimal import Decimal

import pytest

from recovery_router.jev import BudgetExceeded, JevClient, build_request, TRANSIENT_HTTP_STATUSES
from test_jev import FakeHTTP, record


@pytest.mark.parametrize("status", TRANSIENT_HTTP_STATUSES)
def test_transient_retry_is_opt_in_and_resumed_attempt_has_new_reservation(tmp_path, status):
    ledger = tmp_path / "ledger.jsonl"
    fake = FakeHTTP(status=status)
    payload = build_request(record())
    with JevClient(ledger, api_key="key", transport=fake, max_retries=3) as client:
        with pytest.raises(RuntimeError, match=f"HTTP {status}"):
            client.request(payload)
        assert client.request_count == 1
    before = ledger.read_bytes()
    with JevClient(ledger, api_key="key", transport=fake, max_retries=3) as client:
        with pytest.raises(RuntimeError, match="prohibited"):
            client.request(payload)
    assert ledger.read_bytes() == before
    fake.status = 200
    with JevClient(ledger, api_key="key", transport=fake, max_retries=3,
                   retry_transient_http=True) as client:
        client._wait = lambda _: None
        result = client.request(payload)
        assert result["attempt_count"] == 2
        assert client.reserved_usd == Decimal("0.00268842")
        assert client.request_count == 2
    assert ledger.read_bytes().startswith(before)
    events = [json.loads(line) for line in ledger.read_text().splitlines()]
    retry = events[-2]
    assert retry["event"] == "reserved" and retry["attempt"] == 2
    assert retry["retry_policy"]["retry_transient_http"] is True
    assert retry["reserved_usd"] == "0.002688"
    with JevClient(ledger, transport=fake) as client:
        assert client.request(payload)["cache_hit"] is True
        assert client.reserved_usd == Decimal("0.00268842")
    assert len(fake.calls) == 2


@pytest.mark.parametrize("status", [401, 403, 422, 501])
def test_nontransient_http_never_retries_even_when_opted_in(tmp_path, status):
    ledger = tmp_path / "ledger.jsonl"
    fake = FakeHTTP(status=status)
    with JevClient(ledger, api_key="key", transport=fake, max_retries=3,
                   retry_transient_http=True) as client:
        with pytest.raises(RuntimeError, match=f"HTTP {status}"):
            client.request(build_request(record()))
        assert client.reserved_usd == Decimal("0.002688")
    with JevClient(ledger, api_key="key", transport=fake, max_retries=3,
                   retry_transient_http=True) as client:
        with pytest.raises(RuntimeError, match="prohibited"):
            client.request(build_request(record()))
    assert len(fake.calls) == 1


@pytest.mark.parametrize("caps", [{"max_usd": .004}, {"max_requests": 1}])
def test_transient_retry_rechecks_unchanged_caps_across_resume(tmp_path, caps):
    ledger = tmp_path / "ledger.jsonl"
    fake = FakeHTTP(status=520)
    for _ in range(2):
        with JevClient(ledger, api_key="key", transport=fake, max_retries=3,
                       retry_transient_http=True, **caps) as client:
            client._wait = lambda _: None
            with pytest.raises(BudgetExceeded):
                client.request(build_request(record()))
            assert client.request_count == 1
            assert client.reserved_usd == Decimal("0.002688")
    assert len(fake.calls) == 1


def test_transient_and_default_statuses_share_one_bounded_retry_counter(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    statuses, calls = iter([429, 520, 503]), []
    def transport(payload, key, timeout):
        calls.append(payload)
        return next(statuses), {"retry_after_seconds": .1}
    delays = []
    with JevClient(ledger, api_key="key", transport=transport, max_retries=2,
                   retry_transient_http=True) as client:
        client._wait = delays.append
        with pytest.raises(RuntimeError, match="HTTP 503"):
            client.request(build_request(record()))
        assert client.request_count == 3
        assert client.reserved_usd == Decimal("0.008064")
    with JevClient(ledger, api_key="key", transport=transport, max_retries=2,
                   retry_transient_http=True) as client:
        with pytest.raises(RuntimeError, match="exhausted"):
            client.request(build_request(record()))
    assert len(calls) == 3 and calls[0] == calls[1] == calls[2]
    assert delays == [.5, 1.]
