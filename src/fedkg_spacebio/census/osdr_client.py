"""A small, cached, rate-limited client for NASA OSDR (WS-06).

AI-drafted (Claude, WS-06) at Kaitlin's request. Read this module top to bottom; it is meant to be
rebuildable from scratch.

What it talks to (verified live from WSL on 2026-11-12)
-------------------------------------------------------
- **Biological Data API v2**, ``https://visualization.osdr.nasa.gov/biodata/api/v2/``:
  ``dataset/<OSD>/`` (study metadata incl. protocol text) and ``query/samples/`` (one tabular call
  that returns every sample of several studies, with chosen metadata fields as columns).
- **File listing API**, ``https://osdr.nasa.gov/osdr/data/osd/files/<n>``: every file of a study
  with its category, size and a download path.
- **Downloads**, ``https://osdr.nasa.gov/geode-py/ws/studies/<OSD>/download?source=datamanager&file=<name>``.
  This answers with a 302 to a *presigned* Amazon S3 URL. The presigned URL carries temporary AWS
  credentials in its query string, so this client never stores, logs or returns it: the cache is
  keyed on the stable geode URL above. S3 honours HTTP ``Range``, which lets us read just the
  header of a 500 MB DE table.

Design choices (and the alternatives)
-------------------------------------
- **Cache every response on disk** (``$FEDKG_WORKDIR/cache/osdr/``), keyed by a SHA-256 of the
  canonical request (URL + sorted query parameters + byte range). Each body sits next to a
  ``.meta.json`` that records what was asked and when, so a cached answer can be traced to its
  request and access date. Alternative: ``requests-cache``. Rejected: a new dependency, and its
  SQLite store hides the per-request provenance we want to cite.
- **Whole-file downloads go to the data directory**, not the cache: ``download()`` writes
  ``$FEDKG_WORKDIR/data/osdr/<OSD>/<file>`` plus a ``.meta.json``. A file that is already there with
  the expected size is not fetched again.
- **Politeness:** at most one live request per ``min_interval_s`` (default 1 s), a generic
  User-Agent (no personal details), and retries with exponential backoff on 429 and 5xx (honouring a
  numeric ``Retry-After``). Cache hits don't count against the rate limit.
- **Testability:** the HTTP session, clock and sleep function are injectable. Tests pass a fake
  session and never touch the network.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

BIODATA_BASE = "https://visualization.osdr.nasa.gov/biodata/api/v2"
FILES_API = "https://osdr.nasa.gov/osdr/data/osd/files/{number}"
DOWNLOAD_URL = "https://osdr.nasa.gov/geode-py/ws/studies/{osd}/download?source=datamanager&file={file}"
USER_AGENT = "fedkg-spacebio-census/0.0.1 (research testbed; python-requests)"

RETRY_STATUS = {429, 500, 502, 503, 504}


class OsdrError(RuntimeError):
    """An OSDR request failed after retries, or returned something we can't use."""


def default_workdir() -> Path:
    """``$FEDKG_WORKDIR`` if set, else ``~/fedkg-work`` (never a path inside the repo; D-010)."""
    return Path(os.environ.get("FEDKG_WORKDIR", str(Path.home() / "fedkg-work")))


def osd_number(osd_id: str) -> str:
    """``"OSD-47"`` → ``"47"``. Raises on anything that isn't an OSD accession."""
    prefix, _, num = osd_id.partition("-")
    if prefix != "OSD" or not num.isdigit():
        raise ValueError(f"not an OSD accession: {osd_id!r}")
    return num


def canonical_request(url: str, params: Mapping[str, Any] | None = None,
                      byte_range: tuple[int, int] | None = None) -> str:
    """One string that identifies a request, used as the cache key's input.

    Parameters are sorted so ``{"a": 1, "b": 2}`` and ``{"b": 2, "a": 1}`` share a cache entry.
    A byte range is part of the key: the first 64 KB of a file is a different answer from the file.
    """
    key = url
    if params:
        key += ("&" if "?" in url else "?") + urlencode(sorted(params.items()), quote_via=quote)
    if byte_range is not None:
        key += f"#bytes={byte_range[0]}-{byte_range[1]}"
    return key


class OsdrClient:
    """Cached, rate-limited access to the OSDR endpoints above.

    Args:
        cache_dir: where response bodies are cached. Default ``$FEDKG_WORKDIR/cache/osdr``.
        data_dir: where whole files are downloaded. Default ``$FEDKG_WORKDIR/data/osdr``.
        session: anything with a ``requests.Session``-style ``get``. Default: a real session.
        min_interval_s: minimum seconds between two live requests.
        max_retries: retries after the first attempt, for 429/5xx and connection errors.
        offline: if True, a cache miss raises instead of going to the network.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        data_dir: Path | None = None,
        session: Any = None,
        min_interval_s: float = 1.0,
        max_retries: int = 3,
        timeout_s: float = 120.0,
        offline: bool = False,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        work = default_workdir()
        self.cache_dir = Path(cache_dir) if cache_dir else work / "cache" / "osdr"
        self.data_dir = Path(data_dir) if data_dir else work / "data" / "osdr"
        if session is None:
            import requests  # imported here so tests with a fake session don't need the network

            session = requests.Session()
        self.session = session
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.offline = offline
        self._clock = clock
        self._sleep = sleep
        self._last_request: float | None = None
        self.live_requests = 0  # how many requests actually went to the network (for tests/logs)

    # ---------------------------------------------------------------- low level

    def _throttle(self) -> None:
        """Wait until ``min_interval_s`` has passed since the previous live request."""
        if self._last_request is not None:
            wait = self.min_interval_s - (self._clock() - self._last_request)
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._clock()

    def _live_get(self, url: str, params: Mapping[str, Any] | None, headers: dict[str, str],
                  stream: bool) -> Any:
        """One GET with throttling and retries. Returns the response object (status < 400)."""
        if self.offline:
            raise OsdrError(f"offline mode and no cached response for {url}")
        headers = {"User-Agent": USER_AGENT, **headers}
        last_error: str = ""
        for attempt in range(self.max_retries + 1):
            self._throttle()
            self.live_requests += 1
            try:
                resp = self.session.get(url, params=params, headers=headers,
                                        timeout=self.timeout_s, stream=stream)
            except Exception as exc:  # connection reset, timeout, DNS: all worth a retry
                last_error = f"{type(exc).__name__}: {exc}"
                resp = None
            if resp is not None:
                if resp.status_code < 400:
                    return resp
                last_error = f"HTTP {resp.status_code}"
                if resp.status_code not in RETRY_STATUS:
                    break  # 404 and friends won't get better by retrying
                retry_after = resp.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    self._sleep(float(retry_after))
                    continue
            if attempt < self.max_retries:
                self._sleep(self.min_interval_s * 2**attempt)  # 1 s, 2 s, 4 s ...
        # Report the stable URL only. Never the presigned redirect target.
        raise OsdrError(f"GET {url} failed: {last_error}")

    def _cache_paths(self, key: str) -> tuple[Path, Path]:
        digest = hashlib.sha256(key.encode()).hexdigest()[:32]
        return self.cache_dir / f"{digest}.body", self.cache_dir / f"{digest}.meta.json"

    def get_bytes(self, url: str, params: Mapping[str, Any] | None = None,
                  byte_range: tuple[int, int] | None = None) -> bytes:
        """GET ``url`` (optionally only bytes ``start..end`` inclusive), via the cache."""
        key = canonical_request(url, params, byte_range)
        body_path, meta_path = self._cache_paths(key)
        if body_path.exists() and meta_path.exists():
            return body_path.read_bytes()

        headers: dict[str, str] = {}
        if byte_range is not None:
            headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
        resp = self._live_get(url, params, headers, stream=byte_range is not None)
        if byte_range is not None:
            # Read at most the requested number of bytes even if the server ignored Range (200).
            want = byte_range[1] - byte_range[0] + 1
            chunks, got = [], 0
            for chunk in resp.iter_content(chunk_size=65536):
                chunks.append(chunk)
                got += len(chunk)
                if got >= want:
                    break
            resp.close()
            body = b"".join(chunks)[:want]
        else:
            body = resp.content

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        body_path.write_bytes(body)
        meta = {
            "request": key,  # the stable URL; the presigned redirect target is never recorded
            "status": resp.status_code,
            "content_type": resp.headers.get("Content-Type"),
            "n_bytes": len(body),
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        meta_path.write_text(json.dumps(meta, indent=1))
        return body

    def get_json(self, url: str, params: Mapping[str, Any] | None = None) -> Any:
        body = self.get_bytes(url, params)
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise OsdrError(f"{url} did not return JSON: {body[:200]!r}") from exc

    def cached_meta(self, url: str, params: Mapping[str, Any] | None = None,
                    byte_range: tuple[int, int] | None = None) -> dict | None:
        """The ``.meta.json`` of a cached request (for access dates), or None if not cached."""
        _, meta_path = self._cache_paths(canonical_request(url, params, byte_range))
        return json.loads(meta_path.read_text()) if meta_path.exists() else None

    # ---------------------------------------------------------------- OSDR endpoints

    def dataset(self, osd_id: str) -> dict:
        """Study-level metadata (``metadata`` holds mission, protocols, identifiers, ...)."""
        data = self.get_json(f"{BIODATA_BASE}/dataset/{osd_id}/")
        if osd_id not in data:
            raise OsdrError(f"dataset response for {osd_id} lacks the {osd_id} key")
        return data[osd_id]

    @staticmethod
    def samples_query(accessions: Iterable[str], fields: Iterable[str]) -> tuple[str, dict]:
        """URL and parameters for one ``query/samples/`` call.

        ``id.accession=A|B|C`` filters to those studies (``|`` is OR). A field given with an empty
        value is an output column without a filter; a prefix such as ``study.factor value`` expands
        to every field under it.
        """
        params = {"id.accession": "|".join(accessions), "format": "csv"}
        params.update({f: "" for f in fields})
        return f"{BIODATA_BASE}/query/samples/", params

    def query_samples(self, accessions: Iterable[str], fields: Iterable[str]) -> str:
        """CSV text: one row per (study, assay, sample) with the requested fields as columns."""
        url, params = self.samples_query(list(accessions), list(fields))
        return self.get_bytes(url, params).decode("utf-8")

    def file_listing(self, osd_id: str) -> list[dict]:
        """Every file of a study: ``file_name``, ``file_size``, ``category``, ``date_created``."""
        data = self.get_json(FILES_API.format(number=osd_number(osd_id)))
        try:
            return data["studies"][osd_id]["study_files"]
        except KeyError as exc:
            raise OsdrError(f"file listing for {osd_id} has no study_files") from exc

    @staticmethod
    def download_url(osd_id: str, file_name: str) -> str:
        return DOWNLOAD_URL.format(osd=osd_id, file=quote(file_name))

    def head_bytes(self, osd_id: str, file_name: str, n_bytes: int = 65536) -> bytes:
        """The first ``n_bytes`` of a study file (an HTTP Range request), via the cache."""
        return self.get_bytes(self.download_url(osd_id, file_name), byte_range=(0, n_bytes - 1))

    def download(self, osd_id: str, file_name: str, expected_size: int | None = None) -> Path:
        """Download a whole study file to ``data_dir/<OSD>/<file>`` (skipped if already there).

        If ``expected_size`` is given (from the file listing), an existing file of a different size
        is fetched again, and a download that ends at the wrong size raises.
        """
        dest = self.data_dir / osd_id / file_name
        meta_path = dest.with_name(dest.name + ".meta.json")
        if dest.exists() and meta_path.exists() and (
            expected_size is None or dest.stat().st_size == expected_size
        ):
            return dest
        url = self.download_url(osd_id, file_name)
        resp = self._live_get(url, None, {}, stream=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
        resp.close()
        size = tmp.stat().st_size
        if expected_size is not None and size != expected_size:
            tmp.unlink()
            raise OsdrError(f"{osd_id}/{file_name}: got {size} bytes, listing says {expected_size}")
        tmp.replace(dest)
        meta_path.write_text(json.dumps({
            "request": url,
            "n_bytes": size,
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }, indent=1))
        return dest
