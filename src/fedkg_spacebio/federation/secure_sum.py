"""Simulated secure sum for the phase 3a fixed effect (CONTRACTS §6.2–§6.3).

**This is a simulation.** It reproduces the *arithmetic* of pairwise additive masking, so we can
see exactly what a server would receive under a "masked contributions only" policy. It is not a
secure protocol: the pairwise seeds come from the job configuration instead of a key exchange,
every client runs on one machine in the FLARE simulator, and there is no dropout recovery (a
missing client fails the job). It makes no security or privacy claim, and the data are public.

What the server needs for a fixed effect
----------------------------------------
For gene g, with site i reporting log2FC y_i and standard error s_i, and w_i = 1/s_i²:

    FE estimate = Σ w_i·y_i / Σ w_i,      FE SE = 1 / √(Σ w_i)

So the server needs two **totals** per gene, never a single site's (y_i, s_i). Cochran's Q needs
one more total (D-039):

    Q = Σ w_i (y_i − ȳ)² = Σ w_i·y_i² − (Σ w_i·y_i)² / Σ w_i

Each client therefore sends masked vectors of w, w·y (and w·y² when ``fe_het`` is requested), one
entry per gene in the job's gene plan. Genes the plan doesn't assign to the client are exact
zeros, so every vector has the same length and position i means the same gene everywhere.

Fixed point (why integers at all)
---------------------------------
Masks must cancel *exactly*. Floating-point addition isn't associative and a mask of size 2^60
would swamp a value of size 1, so masking is done on integers modulo 2^m:

- **encode**: v = round(x · 2^f) (f = ``fixed_point_bits``), stored modulo 2^m (m =
  ``modulus_bits``) in two's complement, so negative values (w·y < 0) work: −3 is 2^m − 3.
- **decode** a sum: s ≥ 2^(m−1) means s − 2^m.
- rounding error per encoded value ≤ 2^−(f+1).

**Overflow is an error, never a wrap.** With K clients in the job, every encoded value must have
|v| ≤ L = ⌊(2^(m−1) − 1)/K⌋. Then any partial or full sum of K such values has magnitude
≤ K·L ≤ 2^(m−1) − 1, so it is representable and the decoded total is the true integer total. A
client can't see the other clients' values, so this per-value limit is the only way it can
guarantee the totals; ``encode`` refuses anything beyond it, and NaN or ±inf.

m is at most 64 here: numpy's uint64 arithmetic is exactly arithmetic modulo 2^64 (it wraps
silently on arrays), so masking and summing are plain vectorized ``+``/``−``. A larger modulus
would need arrays of Python ints. ``Params`` refuses m > 64 (an implementation limit), and so
does the job-spec parser (``jobspec.MAX_MODULUS_BITS``).

Chosen parameters (WS-13; f = 40, m = 64). On the real liver summaries the largest w at any site
is 920 (SpaceX-16-ISST, SE 0.033), the largest |w·y| 119, the largest w·y² 136. With K = 4,
L/2^f = 2^23/4 ≈ 2.1 × 10⁶: about 2,280× above the largest value, and the per-value error is
≤ 2^−41 ≈ 4.5e-13. (f = 32 would leave a 5.8 × 10⁵× margin at ~256× the error; both work.)

How exact the result is (``error_bound``)
-----------------------------------------
Each decoded total is off by at most ε = k·2^−(f+1) (k = contributing clients for the gene; an
exact zero adds no error). Writing A = Σw·y, B = Σw, C = Σw·y² and est = A/B, for errors
|a|, |b|, |c| ≤ ε:

    est' − est = (a − b·est) / (B + b)                 → |Δest| ≤ ε(1 + |est|) / (B − ε)
    |Δse| ≤ ε / (2 (B − ε)^{3/2})                      (mean value theorem on B^{−1/2})
    |ΔQ| ≤ ε + (2ε|est|B + ε² + est²·B·ε) / (B − ε)

These scale with 1/Σw, so on genes with small Σw they exceed 2^−f. (CONTRACTS §6.3 first said
"within 2^−f per gene"; it now states this bound.)

Masks
-----
Clients are put in sorted order. For each pair (i, j) there is a pair key, and from it a stream of
pseudo-random integers modulo 2^m for each quantity. Client i **adds** the stream of every pair
with j after it and **subtracts** the stream of every pair with j before it. In the sum over all
clients, each pair's stream appears once with + and once with −, so it cancels, and only the sum
of the encoded values remains. A single client's masked vector (or any sum over fewer than all
clients) still carries at least one uncancelled stream, which is uniform over [0, 2^m), so it
looks like noise.

- **Pair key:** SHAKE-256 over a domain tag, the run's root seed, the job ID and the two client
  names (sorted), each length-prefixed so no two inputs can produce the same byte string.
- **Stream:** SHAKE-256 over the pair key and the quantity name, read as little-endian uint64.
- **Why SHAKE-256:** it's in Python's standard library (no new dependency), its output is fixed by
  the standard (so it doesn't change with a numpy upgrade), and an extendable-output hash is the
  usual stand-in for a pseudo-random generator. Alternatives: numpy's PCG64 (``Generator`` method
  streams are not guaranteed stable across numpy versions); AES in counter mode (needs a new
  dependency). ``tests/test_secure_sum.py`` pins the first values for a fixed seed.

Exact totals, then floats
-------------------------
Totals are kept as exact integers. FE uses ``A / B`` on Python ints, which Python rounds
correctly. Q is computed as the exact rational (C·B − A²)/(B·2^f) and converted to float once,
so the subtraction in Q = C − A²/B loses nothing to floating point. The only error left is the
clients' rounding to f bits. A rounding residue can make Q very slightly negative when the true
Q is ~0; it is set to 0 and counted.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pandas as pd
from scipy import stats

from fedkg_spacebio.federation.meta import bh_qvalues
from fedkg_spacebio.federation.site_de import META_FIELDS, TESTED_FLAGS

MASK_SCHEME = "pairwise-additive/shake256-le-u64/1"
DOMAIN = b"fedkg-secure-sum/1"
QUANTITIES_FE = ("w", "wy")
QUANTITIES_FE_HET = ("w", "wy", "wy2")
# The masked-contribution metadata: the §4 site metadata (unmasked, as §6.2 says) plus what the
# server needs to check that every client masked the same rows the same way.
MASK_FIELDS = ("job_id", "gene_plan_sha256", "genes_exclude_sha256", "n_rows",
               "fixed_point_bits", "modulus_bits", "quantities", "mask_scheme")
CONTRIBUTION_META_FIELDS = (*META_FIELDS, *MASK_FIELDS)
MIN_K = 2  # same as meta.pool: genes pooled from fewer than 2 sites get no estimate


class SecureSumError(ValueError):
    """A value that can't be encoded, a payload that doesn't match, or a missing client."""


# ------------------------------------------------------------------ parameters


@dataclass(frozen=True)
class Params:
    """``fixed_point_bits`` f and ``modulus_bits`` m, checked: 1 ≤ f < m ≤ 64."""

    fixed_point_bits: int
    modulus_bits: int

    def __post_init__(self):
        f, m = self.fixed_point_bits, self.modulus_bits
        for name, v in (("fixed_point_bits", f), ("modulus_bits", m)):
            if isinstance(v, bool) or not isinstance(v, int | np.integer):
                raise SecureSumError(f"{name} must be an int, got {v!r}")
        if m > 64:
            raise SecureSumError(f"modulus_bits = {m}: this implementation supports moduli up to "
                                 "2^64 (uint64 arithmetic); refused at run time (CONTRACTS §6.3)")
        if not 1 <= f < m:
            raise SecureSumError(f"need 1 <= fixed_point_bits < modulus_bits, got f={f}, m={m}")

    @property
    def scale(self) -> float:
        return float(2 ** self.fixed_point_bits)

    @property
    def low_bits(self) -> np.uint64:
        """2^m − 1 as uint64: ``& low_bits`` reduces a uint64 modulo 2^m."""
        return np.uint64((1 << self.modulus_bits) - 1)

    def to_dict(self) -> dict:
        return {"fixed_point_bits": int(self.fixed_point_bits),
                "modulus_bits": int(self.modulus_bits)}


def value_limit(params: Params, n_clients: int) -> int:
    """L = ⌊(2^(m−1) − 1) / K⌋: the largest |encoded value| that keeps every sum of K in range."""
    if n_clients < 1:
        raise SecureSumError("n_clients must be >= 1")
    return ((1 << (params.modulus_bits - 1)) - 1) // n_clients


# ------------------------------------------------------------------ encode / decode


def encode(x, params: Params, n_clients: int, what: str = "value") -> np.ndarray:
    """Float array → fixed-point integers modulo 2^m (uint64, two's complement).

    Raises ``SecureSumError`` for non-finite input or any |round(x·2^f)| > ``value_limit``.
    Multiplying by 2^f is exact in float64 (a power of two only changes the exponent), so the
    only rounding is ``np.rint`` (round half to even).
    """
    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        raise SecureSumError(f"{what}: non-finite value; nothing is encoded")
    scaled = np.rint(x * params.scale)
    limit = value_limit(params, n_clients)
    # First a float check that the int64 cast below is safe, then an exact integer check
    # (L itself isn't exactly representable as a float64 when it's above 2^53).
    if np.any(np.abs(scaled) >= 2.0 ** 63):
        raise SecureSumError(f"{what}: value too large for 64-bit fixed point")
    ints = scaled.astype(np.int64)
    over = np.abs(ints) > limit
    if np.any(over):
        worst = float(np.max(np.abs(x[over])))
        raise SecureSumError(
            f"{what}: |value| up to {worst:.6g} exceeds the overflow limit "
            f"{limit / params.scale:.6g} for {n_clients} clients at f={params.fixed_point_bits}, "
            f"m={params.modulus_bits}; refusing to encode (it would wrap)")
    # int64 → uint64 is modular in numpy, i.e. two's complement modulo 2^64; & reduces to 2^m.
    return ints.astype(np.uint64) & params.low_bits


def decode_ints(s, params: Params) -> np.ndarray:
    """Sum modulo 2^m (uint64) → exact signed integers (int64)."""
    s = np.asarray(s, dtype=np.uint64) & params.low_bits
    m = params.modulus_bits
    if m == 64:
        return s.astype(np.int64)  # modular cast = two's complement reading
    signed = s.astype(np.int64)
    return np.where(signed >= (1 << (m - 1)), signed - (1 << m), signed)


def decode(s, params: Params) -> np.ndarray:
    """Sum modulo 2^m → float64 (integer / 2^f)."""
    return decode_ints(s, params).astype(np.float64) / params.scale


# ------------------------------------------------------------------ masks


def _absorb(h, *parts: bytes | str) -> None:
    for p in parts:
        b = p.encode("utf-8") if isinstance(p, str) else bytes(p)
        h.update(len(b).to_bytes(4, "big"))
        h.update(b)


def pair_key(root_seed: bytes, job_id: str, a: str, b: str) -> bytes:
    """The key shared by clients a and b (order doesn't matter)."""
    if a == b:
        raise SecureSumError("a client has no mask with itself")
    lo, hi = sorted((a, b))
    h = hashlib.shake_256(DOMAIN)
    _absorb(h, root_seed, job_id, lo, hi)
    return h.digest(32)


def mask_stream(key: bytes, quantity: str, n: int, params: Params) -> np.ndarray:
    """n pseudo-random integers modulo 2^m from a pair key, one stream per quantity."""
    h = hashlib.shake_256(b"stream")
    _absorb(h, key, quantity)
    raw = np.frombuffer(h.digest(8 * n), dtype="<u8").astype(np.uint64)
    return raw & params.low_bits


def mask(client: str, clients: list[str], encoded: Mapping[str, np.ndarray], *,
         root_seed: bytes, job_id: str, params: Params) -> dict[str, np.ndarray]:
    """Add every pair stream with a later client, subtract every one with an earlier client."""
    order = sorted(clients)
    if client not in order or len(set(order)) != len(order):
        raise SecureSumError(f"{client} must appear once in the job's clients")
    me = order.index(client)
    out = {}
    for q, v in encoded.items():
        acc = np.array(v, dtype=np.uint64, copy=True)
        for j, other in enumerate(order):
            if j == me:
                continue
            s = mask_stream(pair_key(root_seed, job_id, client, other), q, len(acc), params)
            acc = acc + s if j > me else acc - s  # uint64: wraps modulo 2^64
        out[q] = acc & params.low_bits
    return out


def sum_mod(vectors: list[np.ndarray], params: Params) -> np.ndarray:
    """Σ vectors modulo 2^m. With every client's vector, the masks cancel."""
    if not vectors:
        raise SecureSumError("nothing to sum")
    acc = np.zeros(len(vectors[0]), dtype=np.uint64)
    for v in vectors:
        acc = acc + np.asarray(v, dtype=np.uint64)
    return acc & params.low_bits


# ------------------------------------------------------------------ client side


def quantities_for(methods) -> tuple[str, ...]:
    """w and w·y for ``fe``; also w·y² when ``fe_het`` (Q and I²) is requested (D-039)."""
    return QUANTITIES_FE_HET if "fe_het" in methods else QUANTITIES_FE


def contribution_values(genes: Mapping[str, np.ndarray], rows: pd.DataFrame, client: str,
                        flag: str, quantities: tuple[str, ...]) -> dict[str, np.ndarray]:
    """The plain (unmasked) vectors one client contributes, over the plan's rows.

    ``genes`` is the client's §4 summary arrays, ``rows`` the gene plan after exclusions (index
    ``ncbi_gene``, one bool column per client), ``flag`` the spec's inclusion flag. A gene the
    plan assigns to this client must be in its summary and flagged; otherwise the plan was built
    from other flags than this site's and the client refuses. Unassigned genes are exact zeros.
    """
    if client not in rows.columns:
        raise SecureSumError(f"{client} isn't a column of the gene plan")
    if flag not in TESTED_FLAGS:
        raise SecureSumError(f"unknown inclusion flag {flag!r}")
    ids = np.asarray(genes["ncbi_gene"], dtype=np.int64)
    pos = pd.Index(ids).get_indexer(rows.index.to_numpy(dtype=np.int64))
    assigned = rows[client].to_numpy(dtype=bool)
    flagged = np.zeros(len(rows), dtype=bool)
    flagged[pos >= 0] = np.asarray(genes[flag], dtype=bool)[pos[pos >= 0]]
    bad = assigned & ~flagged
    if bad.any():
        sample = rows.index[bad][:5].tolist()
        raise SecureSumError(f"{client}: the plan assigns {int(bad.sum())} genes this site didn't "
                             f"flag under {flag} (e.g. {sample}); plan and flags disagree")
    y = np.zeros(len(rows))
    w = np.zeros(len(rows))
    take = pos[assigned]
    se = np.asarray(genes["se"], dtype=np.float64)[take]
    y[assigned] = np.asarray(genes["log2fc"], dtype=np.float64)[take]
    w[assigned] = 1.0 / se**2
    values = {"w": w, "wy": w * y, "wy2": w * y * y}
    return {q: values[q] for q in quantities}


def masked_contribution(values: Mapping[str, np.ndarray], client: str, clients: list[str], *,
                        root_seed: bytes, job_id: str, params: Params
                        ) -> dict[str, np.ndarray]:
    """Encode (overflow-checked) and mask one client's vectors. Returns ``{masked_<q>: uint64}``."""
    k = len(clients)
    encoded = {q: encode(v, params, k, what=f"{client} {q}") for q, v in values.items()}
    masked = mask(client, clients, encoded, root_seed=root_seed, job_id=job_id, params=params)
    return {f"masked_{q}": v for q, v in masked.items()}


# ------------------------------------------------------------------ the boundary check


def validate_contribution(params_in: Mapping, meta: Mapping, expected: Mapping | None = None
                          ) -> None:
    """Raise unless a masked contribution matches the §6.2 contract exactly.

    Called by the client before sending and by the server on receipt. Exact whitelists: the
    arrays are exactly ``masked_<q>`` for the declared quantities, 1-D uint64 of length
    ``n_rows`` and below 2^m; the metadata keys are exactly ``CONTRIBUTION_META_FIELDS`` and plain
    JSON. ``expected`` (server side) pins ``job_id``, plan and exclusion hashes, f, m, ``n_rows``,
    quantities and the scheme to the job's own values.
    """
    if set(meta) != set(CONTRIBUTION_META_FIELDS):
        extra = sorted(set(meta) - set(CONTRIBUTION_META_FIELDS))
        missing = sorted(set(CONTRIBUTION_META_FIELDS) - set(meta))
        raise SecureSumError(f"masked-contribution metadata fields wrong; extra {extra}, "
                             f"missing {missing}")
    json.dumps(meta)  # plain JSON only: no arrays smuggled into the metadata
    quantities = tuple(meta["quantities"])
    if quantities not in (QUANTITIES_FE, QUANTITIES_FE_HET):
        raise SecureSumError(f"quantities must be {QUANTITIES_FE} or {QUANTITIES_FE_HET}")
    want = {f"masked_{q}" for q in quantities}
    if set(params_in) != want:
        raise SecureSumError(f"arrays must be exactly {sorted(want)}; got {sorted(params_in)}")
    p = Params(meta["fixed_point_bits"], meta["modulus_bits"])
    n = meta["n_rows"]
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise SecureSumError("n_rows must be a positive int")
    for key, arr in params_in.items():
        a = np.asarray(arr)
        if a.dtype != np.uint64 or a.ndim != 1 or len(a) != n:
            raise SecureSumError(f"{key}: expected a 1-D uint64 array of length {n}, got "
                                 f"{a.dtype} {a.shape}")
        if np.any(a > p.low_bits):
            raise SecureSumError(f"{key}: values must be below 2^{p.modulus_bits}")
    for f in ("n_flight", "n_ground"):
        if isinstance(meta[f], bool) or not isinstance(meta[f], int) or meta[f] < 1:
            raise SecureSumError(f"{f} must be a positive int")
    if meta["mask_scheme"] != MASK_SCHEME:
        raise SecureSumError(f"mask_scheme {meta['mask_scheme']!r} isn't {MASK_SCHEME!r}")
    for key, value in (expected or {}).items():
        got = tuple(meta[key]) if key == "quantities" else meta[key]
        want_v = tuple(value) if key == "quantities" else value
        if got != want_v:
            raise SecureSumError(f"{meta.get('client')}: {key} = {got!r}, the job has {want_v!r}")


# ------------------------------------------------------------------ server side


@dataclass
class Totals:
    """Decoded per-gene totals over all clients. ``ints`` are exact (units of 2^−f)."""

    genes: np.ndarray            # ncbi_gene, plan order
    k: np.ndarray                # contributing clients per gene (from the plan)
    ints: dict[str, np.ndarray]  # quantity → int64 exact totals
    params: Params

    def value(self, q: str) -> np.ndarray:
        return self.ints[q].astype(np.float64) / self.params.scale


def aggregate(payloads: Mapping[str, Mapping[str, np.ndarray]], rows: pd.DataFrame,
              params: Params) -> Totals:
    """Sum every client's masked vectors and decode. Refuses a missing or unknown client.

    ``payloads`` maps client → its ``masked_<q>`` arrays (already validated). ``rows`` is the
    plan after exclusions: its columns are the job's clients, which must be exactly the
    payloads' clients, because a sum over a subset leaves masks uncancelled (and would be a
    different release).
    """
    clients = sorted(rows.columns)
    got = sorted(payloads)
    if got != clients:
        raise SecureSumError(f"payloads from {got}, the job's clients are {clients}; "
                             "refusing a partial sum (no dropout recovery)")
    keys = sorted(next(iter(payloads.values())))
    ints = {}
    for key in keys:
        vecs = [np.asarray(payloads[c][key]) for c in clients]
        if any(len(v) != len(rows) for v in vecs):
            raise SecureSumError(f"{key}: vector length differs from the plan's {len(rows)} rows")
        ints[key.removeprefix("masked_")] = decode_ints(sum_mod(vecs, params), params)
    k = rows[clients].to_numpy(dtype=bool).sum(axis=1)
    return Totals(rows.index.to_numpy(dtype=np.int64), k, ints, params)


def fe_from_totals(t: Totals) -> pd.DataFrame:
    """FE (and, with w·y², Q, its p and I²) per gene from the totals. Same conventions as
    ``meta.fixed_effect`` / ``meta.pool``: k < 2 → NaN and left out of BH; BH over the rest."""
    A, B = t.ints["wy"], t.ints["w"]
    n = len(t.genes)
    use = (t.k >= MIN_K) & (B > 0)
    est = np.full(n, np.nan)
    sw = np.full(n, np.nan)
    for i in np.flatnonzero(use):
        est[i] = int(A[i]) / int(B[i])           # exact ints, correctly rounded division
        sw[i] = int(B[i]) / t.params.scale
    with np.errstate(invalid="ignore", divide="ignore"):
        se = 1.0 / np.sqrt(sw)
        p = 2.0 * stats.norm.sf(np.abs(est / se))
    out = pd.DataFrame({"ncbi_gene": t.genes, "k": t.k, "fe_est": est, "fe_se": se, "fe_p": p})
    out["fe_q"] = bh_qvalues(out["fe_p"].to_numpy())
    out.attrs["n_q_clamped"] = 0
    if "wy2" in t.ints:
        C = t.ints["wy2"]
        q = np.full(n, np.nan)
        clamped = 0
        for i in np.flatnonzero(use):
            a, b, c = int(A[i]), int(B[i]), int(C[i])
            val = Fraction(c * b - a * a, b * (1 << t.params.fixed_point_bits))
            if val < 0:
                val, clamped = Fraction(0), clamped + 1
            q[i] = float(val)
        df = t.k - 1
        with np.errstate(invalid="ignore", divide="ignore"):
            q_p = np.where(use & (df >= 1), stats.chi2.sf(q, np.maximum(df, 1)), np.nan)
            i2 = np.where(use & (df >= 1) & (q > 0), np.maximum(0.0, (q - df) / q), np.nan)
        i2 = np.where(use & (df >= 1) & (q == 0), 0.0, i2)
        out["i2"], out["q"], out["q_p"] = i2, q, q_p
        out.attrs["n_q_clamped"] = clamped
    return out


def error_bound(est, sum_w, k, fixed_point_bits: int) -> dict[str, np.ndarray]:
    """Worst-case |Δ| of FE estimate, SE and Q from rounding to f bits (module docstring).

    Pass the true (plain) est and Σw when checking; the server reports it from decoded values.
    """
    est = np.abs(np.asarray(est, dtype=np.float64))
    B = np.asarray(sum_w, dtype=np.float64)
    eps = np.asarray(k, dtype=np.float64) * 2.0 ** -(fixed_point_bits + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = B - eps
        return {
            "est": eps * (1 + est) / d,
            "se": eps / (2 * d ** 1.5),
            "q": eps + (2 * eps * est * B + eps**2 + est**2 * B * eps) / d,
        }


__all__ = ["CONTRIBUTION_META_FIELDS", "MASK_FIELDS", "MASK_SCHEME", "Params",
           "QUANTITIES_FE", "QUANTITIES_FE_HET", "SecureSumError", "Totals", "aggregate",
           "contribution_values", "decode", "decode_ints", "encode", "error_bound",
           "fe_from_totals", "mask", "mask_stream", "masked_contribution", "pair_key",
           "quantities_for", "sum_mod", "validate_contribution", "value_limit"]
