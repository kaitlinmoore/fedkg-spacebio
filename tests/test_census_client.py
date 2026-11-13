"""OsdrClient: caching, rate limit, retries, Range reads, downloads. No network: a fake session."""

from __future__ import annotations

import json

import pytest

from fedkg_spacebio.census.osdr_client import OsdrClient, OsdrError, canonical_request

PRESIGNED = "https://fake-bucket.s3.amazonaws.com/x.csv?X-Amz-Signature=FAKESECRET"


class FakeResponse:
    def __init__(self, status: int = 200, body: bytes = b"{}", headers: dict | None = None):
        self.status_code = status
        self.content = body
        self.headers = headers or {"Content-Type": "application/json"}
        self.url = PRESIGNED  # what requests reports after following a redirect
        self.closed = False

    def iter_content(self, chunk_size: int = 1):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i:i + chunk_size]

    def close(self):
        self.closed = True


class FakeSession:
    """Returns queued responses in order and records every call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None, stream=False):
        self.calls.append({"url": url, "params": params, "headers": headers, "stream": stream})
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClock:
    def __init__(self):
        self.t = 100.0
        self.sleeps = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def make_client(tmp_path, responses, **kw):
    clock = FakeClock()
    session = FakeSession(responses)
    client = OsdrClient(cache_dir=tmp_path / "cache", data_dir=tmp_path / "data", session=session,
                        clock=clock, sleep=clock.sleep, **kw)
    return client, session, clock


def test_second_identical_request_is_served_from_cache(tmp_path):
    client, session, _ = make_client(tmp_path, [FakeResponse(body=b'{"a": 1}')])
    assert client.get_json("https://example.invalid/x", {"b": "2", "a": "1"}) == {"a": 1}
    # Same parameters in a different order: same cache entry, no second call.
    assert client.get_json("https://example.invalid/x", {"a": "1", "b": "2"}) == {"a": 1}
    assert len(session.calls) == 1 and client.live_requests == 1


def test_cache_meta_records_stable_url_never_the_presigned_one(tmp_path):
    client, _, _ = make_client(tmp_path, [FakeResponse(body=b"x")])
    url = client.download_url("OSD-FAKE-1", "f.csv")
    client.get_bytes(url)
    meta = client.cached_meta(url)
    assert meta["request"] == url
    for f in (tmp_path / "cache").iterdir():
        assert "X-Amz-Signature" not in f.read_text(errors="replace")


def test_rate_limit_spaces_live_requests(tmp_path):
    client, _, clock = make_client(tmp_path, [FakeResponse(), FakeResponse()], min_interval_s=1.0)
    client.get_bytes("https://example.invalid/1")
    client.get_bytes("https://example.invalid/2")
    assert clock.sleeps == [pytest.approx(1.0)]


def test_retries_on_503_then_succeeds(tmp_path):
    client, session, clock = make_client(
        tmp_path, [FakeResponse(503, b""), FakeResponse(200, b"ok")], min_interval_s=0.5)
    assert client.get_bytes("https://example.invalid/r") == b"ok"
    assert len(session.calls) == 2
    assert 0.5 in clock.sleeps  # backoff 0.5 * 2**0


def test_404_raises_without_retry_and_without_presigned_url(tmp_path):
    client, session, _ = make_client(tmp_path, [FakeResponse(404, b"")])
    with pytest.raises(OsdrError) as exc:
        client.get_bytes("https://example.invalid/missing")
    assert len(session.calls) == 1
    assert "X-Amz" not in str(exc.value)


def test_offline_cache_miss_raises(tmp_path):
    client, session, _ = make_client(tmp_path, [], offline=True)
    with pytest.raises(OsdrError):
        client.get_bytes("https://example.invalid/never")
    assert session.calls == []


def test_head_bytes_sends_range_and_truncates_if_server_ignores_it(tmp_path):
    client, session, _ = make_client(tmp_path, [FakeResponse(200, b"0123456789" * 10)])
    head = client.head_bytes("OSD-FAKE-1", "big.csv", n_bytes=25)
    assert head == (b"0123456789" * 3)[:25]
    assert session.calls[0]["headers"]["Range"] == "bytes=0-24"
    # A different range is a different cache key.
    assert canonical_request("u", None, (0, 24)) != canonical_request("u", None, (0, 99))


def test_download_writes_once_and_checks_size(tmp_path):
    client, session, _ = make_client(tmp_path, [FakeResponse(body=b"abc")])
    path = client.download("OSD-FAKE-1", "c.csv", expected_size=3)
    assert path.read_bytes() == b"abc"
    assert json.loads(path.with_name("c.csv.meta.json").read_text())["n_bytes"] == 3
    client.download("OSD-FAKE-1", "c.csv", expected_size=3)  # already there: no call
    assert len(session.calls) == 1


def test_download_size_mismatch_raises_and_leaves_no_file(tmp_path):
    client, _, _ = make_client(tmp_path, [FakeResponse(body=b"abc")])
    with pytest.raises(OsdrError):
        client.download("OSD-FAKE-1", "c.csv", expected_size=99)
    assert not (tmp_path / "data" / "OSD-FAKE-1" / "c.csv").exists()


def test_samples_query_builds_or_filter_and_bare_fields():
    url, params = OsdrClient.samples_query(["OSD-FAKE-1", "OSD-FAKE-2"], ["study.source name"])
    assert url.endswith("/query/samples/")
    assert params["id.accession"] == "OSD-FAKE-1|OSD-FAKE-2"
    assert params["study.source name"] == ""  # bare field = output column, no filter
