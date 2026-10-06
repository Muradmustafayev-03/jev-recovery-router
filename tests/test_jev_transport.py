"""Uncertain network retries are explicit, classified and fully charged."""
import json
import socket
import ssl
import urllib.error
from decimal import Decimal

import pytest

from recovery_router.jev import BudgetExceeded, JevClient, build_request
from test_jev import FakeHTTP, record


@pytest.mark.parametrize("error", [urllib.error.URLError(TimeoutError("SECRET")),
    urllib.error.URLError(ConnectionResetError("SECRET")),
    urllib.error.URLError(socket.gaierror(socket.EAI_AGAIN, "SECRET")), TimeoutError("SECRET")])
def test_allowlisted_transport_retry_keeps_reservation_and_payload(tmp_path, error):
    path, calls = tmp_path / "ledger.jsonl", []
    fake = FakeHTTP()
    def transport(payload, key, timeout):
        calls.append(payload)
        if len(calls) == 1:
            raise error
        return fake(payload, key, timeout)
    with JevClient(path, api_key="SECRET", transport=transport, max_retries=1,
                   retry_uncertain_transport=True) as client:
        client._wait = lambda _: None
        result = client.request(build_request(record()))
        assert result["attempt_count"] == 2
        assert client.reserved_usd == Decimal("0.00268842")
    assert calls[0] == calls[1]
    assert "SECRET" not in path.read_text()
    with JevClient(path, transport=transport) as client:
        assert client.request(build_request(record()))["cache_hit"]
        assert client.request_count == 2


@pytest.mark.parametrize("reason", [ssl.SSLCertVerificationError(1, "SECRET"),
    ssl.SSLError(1, "SECRET"), socket.gaierror(socket.EAI_NONAME, "SECRET"), "SECRET unknown reason"])
def test_certificate_protocol_permanent_dns_and_unknown_errors_never_retry(tmp_path, reason):
    path, calls = tmp_path / "ledger.jsonl", []
    def transport(*args):
        calls.append(1)
        raise urllib.error.URLError(reason)
    for _ in range(2):
        with JevClient(path, api_key="key", transport=transport, max_retries=3,
                       retry_uncertain_transport=True, retry_legacy_unknown_transport=True) as client:
            with pytest.raises(RuntimeError):
                client.request(build_request(record()))
    assert len(calls) == 1
    assert "SECRET" not in path.read_text()


def test_transport_retry_default_blocks_and_explicit_resume_rechecks_cap(tmp_path):
    path, calls = tmp_path / "ledger.jsonl", []
    def transport(*args):
        calls.append(1)
        raise urllib.error.URLError(TimeoutError())
    with JevClient(path, api_key="key", transport=transport, max_retries=3) as client:
        with pytest.raises(RuntimeError, match="URLError"):
            client.request(build_request(record()))
    for caps in ({"max_usd": .004}, {"max_requests": 1}):
        with JevClient(path, api_key="key", transport=transport, max_retries=3,
                       retry_uncertain_transport=True, **caps) as client:
            client._wait = lambda _: None
            with pytest.raises(BudgetExceeded):
                client.request(build_request(record()))
            assert client.reserved_usd == Decimal("0.002688")
    assert len(calls) == 1


def test_historical_reasonless_urlerror_needs_separate_opt_in(tmp_path):
    path = tmp_path / "ledger.jsonl"
    def fail(*args):
        raise urllib.error.URLError("unknown historical cause")
    with JevClient(path, api_key="key", transport=fail) as client:
        with pytest.raises(RuntimeError):
            client.request(build_request(record()))
    events = [json.loads(line) for line in path.read_text().splitlines()]
    del events[-1]["transport"]  # Simulate the previous client's immutable ledger schema.
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    before = path.read_bytes()
    fake = FakeHTTP()
    with JevClient(path, api_key="key", transport=fake, max_retries=3,
                   retry_uncertain_transport=True) as client:
        with pytest.raises(RuntimeError, match="prohibited"):
            client.request(build_request(record()))
    with JevClient(path, api_key="key", transport=fake, max_retries=3,
                   retry_legacy_unknown_transport=True) as client:
        client._wait = lambda _: None
        assert client.request(build_request(record()))["attempt_count"] == 2
        assert client.reserved_usd == Decimal("0.00268842")
    assert path.read_bytes().startswith(before)
    assert len(fake.calls) == 1


def test_http_and_transport_share_bounded_counter_across_resume(tmp_path):
    path, calls = tmp_path / "ledger.jsonl", []
    def transport(*args):
        calls.append(1)
        if len(calls) == 2:
            raise urllib.error.URLError(ConnectionResetError())
        return (429 if len(calls) == 1 else 520), {}
    with JevClient(path, api_key="key", transport=transport, max_retries=2,
                   retry_uncertain_transport=True, retry_transient_http=True) as client:
        client._wait = lambda _: None
        with pytest.raises(RuntimeError, match="HTTP 520"):
            client.request(build_request(record()))
        assert client.reserved_usd == Decimal("0.008064")
    with JevClient(path, api_key="key", transport=transport, max_retries=2,
                   retry_uncertain_transport=True, retry_transient_http=True) as client:
        with pytest.raises(RuntimeError, match="exhausted"):
            client.request(build_request(record()))
    assert len(calls) == 3
