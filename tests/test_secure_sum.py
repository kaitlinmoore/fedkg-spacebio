"""Simulated secure sum (``federation/secure_sum.py``): fixed point, masks, totals, the boundary.

Synthetic inputs only; gene IDs are fake (≥ 9,000,000,001), clients are ``SITE-*``.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fedkg_spacebio.federation import meta
from fedkg_spacebio.federation import secure_sum as ss
from fedkg_spacebio.federation.gene_plan import build_gene_plan, read_gene_plan, write_gene_plan

P = ss.Params(40, 64)
JOB = "job:FAKE-0001"
SEED = bytes(range(32))
CLIENTS = ["SITE-A", "SITE-B", "SITE-C", "SITE-D"]


def summaries(n_genes=300, seed=0, drop=None):
    """Four fake §4 summaries. ``drop``: client → fraction of genes it doesn't flag."""
    rng = np.random.default_rng(seed)
    ids = np.arange(9_000_000_001, 9_000_000_001 + n_genes, dtype=np.int64)
    out = {}
    for j, c in enumerate(CLIENTS):
        flag = np.ones(n_genes, dtype=bool)
        if drop and c in drop:
            flag[rng.random(n_genes) < drop[c]] = False
        out[c] = {"ncbi_gene": ids, "log2fc": rng.normal(0, 1.5, n_genes),
                  "se": rng.uniform(0.03, 2.0, n_genes) * (1 + j / 4),
                  "base_mean": np.full(n_genes, 100.0), "tested_mean10": flag,
                  "tested_indep_filter": flag.copy(), "tested_finite_p": flag.copy()}
    return out


def plan_of(s, min_sites="all"):
    return build_gene_plan({c: pd.DataFrame(g) for c, g in s.items()}, "mean10_finite",
                           min_sites)


def run_masked(s, rows, quantities=ss.QUANTITIES_FE, params=P):
    pay = {c: ss.masked_contribution(ss.contribution_values(s[c], rows, c, "tested_mean10",
                                                            quantities),
                                     c, sorted(rows.columns), root_seed=SEED, job_id=JOB,
                                     params=params)
           for c in rows.columns}
    return pay, ss.fe_from_totals(ss.aggregate(pay, rows, params))


def plain(s, rows):
    clients = sorted(rows.columns)
    Y = np.full((len(rows), len(clients)), np.nan)
    V = np.full_like(Y, np.nan)
    for j, c in enumerate(clients):
        pos = pd.Index(s[c]["ncbi_gene"]).get_indexer(rows.index)
        a = rows[c].to_numpy()
        Y[a, j] = s[c]["log2fc"][pos[a]]
        V[a, j] = s[c]["se"][pos[a]] ** 2
    return meta.pool(Y, V, eb=False)


# ----------------------------------------------------------------- parameters, encode, decode


def test_params_refuse_moduli_above_64_and_bad_orders():
    with pytest.raises(ss.SecureSumError, match="up to 2\\^64"):
        ss.Params(40, 65)
    for f, m in ((64, 64), (0, 64), (41, 40)):
        with pytest.raises(ss.SecureSumError):
            ss.Params(f, m)
    ss.Params(16, 32)  # m < 64 is fine


@pytest.mark.parametrize("params", [P, ss.Params(32, 64), ss.Params(12, 40)])
def test_encode_decode_round_trip_negatives_zero_and_limits(params):
    lim = ss.value_limit(params, 4) / params.scale
    x = np.array([0.0, 1.0, -1.0, 3.25, -1e-9, 920.1, -119.0, lim * 0.999, -lim * 0.999])
    back = ss.decode(ss.encode(x, params, 4), params)
    assert np.all(np.abs(back - x) <= 2.0 ** -(params.fixed_point_bits + 1) * (1 + 1e-12))
    assert ss.decode(ss.encode([-3.0], params, 4), params)[0] == -3.0


def test_negative_values_are_twos_complement():
    v = ss.encode([-1.0], P, 1)
    assert int(v[0]) == (1 << 64) - (1 << 40)
    small = ss.Params(4, 16)
    assert int(ss.encode([-1.0], small, 1)[0]) == (1 << 16) - (1 << 4)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_values_are_refused(bad):
    with pytest.raises(ss.SecureSumError, match="non-finite"):
        ss.encode([1.0, bad], P, 4)


def test_overflow_is_an_error_never_a_wrap():
    limit = ss.value_limit(P, 4)
    ok = (limit - 4096) / P.scale  # L ≈ 2^61 isn't a float64; stay a few float steps below it
    ss.encode([ok, -ok], P, 4)
    with pytest.raises(ss.SecureSumError, match="overflow limit"):
        ss.encode([ok * 1.001], P, 4)
    with pytest.raises(ss.SecureSumError, match="overflow limit"):
        ss.encode([-ok * 1.001], P, 4)
    with pytest.raises(ss.SecureSumError):
        ss.encode([1e30], P, 4)
    # The limit depends on the number of clients: a value fine for 2 is refused for 4.
    v = ss.value_limit(P, 2) / P.scale * 0.9
    ss.encode([v], P, 2)
    with pytest.raises(ss.SecureSumError):
        ss.encode([v], P, 4)


@pytest.mark.parametrize("sign", [1, -1])
def test_k_values_at_the_limit_sum_and_decode_exactly(sign):
    # Parameters whose limit L is below 2^53, so L/2^f is an exact float and we hit L itself.
    for params in (ss.Params(8, 20), ss.Params(20, 50)):
        k = 5
        lim = ss.value_limit(params, k)
        enc = [ss.encode([sign * lim / params.scale], params, k) for _ in range(k)]
        assert int(ss.decode_ints(ss.sum_mod(enc, params), params)[0]) == sign * k * lim


# ----------------------------------------------------------------- masks


def test_mask_streams_are_pinned_deterministic_and_distinct():
    key = ss.pair_key(SEED, JOB, "SITE-A", "SITE-B")
    assert key == ss.pair_key(SEED, JOB, "SITE-B", "SITE-A")
    s = ss.mask_stream(key, "w", 3, P)
    # Known answer: a change of generator, encoding or domain tag fails here.
    # Recorded 2026-11-14 (SHAKE-256, little-endian uint64, m = 64).
    assert s.tolist() == [1507537763449952552, 8024671866292430378, 16870399369207202180]
    assert np.array_equal(s, ss.mask_stream(key, "w", 3, P))
    assert not np.array_equal(s, ss.mask_stream(key, "wy", 3, P))
    assert key != ss.pair_key(SEED, JOB, "SITE-A", "SITE-C")
    assert key != ss.pair_key(SEED, "job:FAKE-0002", "SITE-A", "SITE-B")
    assert key != ss.pair_key(bytes(32), JOB, "SITE-A", "SITE-B")
    with pytest.raises(ss.SecureSumError):
        ss.pair_key(SEED, JOB, "SITE-A", "SITE-A")


@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_masks_cancel_over_the_full_set(k):
    rng = np.random.default_rng(k)
    clients = [f"SITE-{i}" for i in range(k)]
    enc = {c: {"w": ss.encode(rng.normal(0, 50, 200), P, k)} for c in clients}
    masked = [ss.mask(c, clients, enc[c], root_seed=SEED, job_id=JOB, params=P)["w"]
              for c in clients]
    assert np.array_equal(ss.sum_mod(masked, P), ss.sum_mod([enc[c]["w"] for c in clients], P))


def test_masks_cancel_only_as_a_full_set():
    s = summaries(500)
    rows = plan_of(s)
    pay, _ = run_masked(s, rows)
    true = plain(s, rows)
    true_sw = 1 / true["fe_se"].to_numpy() ** 2
    # Leave one client out: its pair streams with every other client stay in the sum.
    for out in CLIENTS:
        part = ss.sum_mod([pay[c]["masked_w"] for c in CLIENTS if c != out], P)
        noise = ss.decode(part, P)
        assert np.median(np.abs(noise)) > 1e5          # spread over the range, not near Σw
        assert abs(np.corrcoef(noise, true_sw)[0, 1]) < 0.2
    # A single masked vector says nothing about its own plain values.
    vals = ss.contribution_values(s["SITE-A"], rows, "SITE-A", "tested_mean10", ss.QUANTITIES_FE)
    one = pay["SITE-A"]["masked_w"].astype(np.float64)
    assert abs(np.corrcoef(one, vals["w"])[0, 1]) < 0.2
    top = (pay["SITE-A"]["masked_w"] >> np.uint64(63)).mean()  # uniform high bit ≈ 0.5
    assert 0.4 < top < 0.6
    # The aggregate refuses to decode a partial set at all.
    with pytest.raises(ss.SecureSumError, match="partial sum"):
        ss.aggregate({c: pay[c] for c in CLIENTS[:3]}, rows, P)


def test_mask_refuses_a_client_outside_the_job():
    with pytest.raises(ss.SecureSumError):
        ss.mask("SITE-Z", CLIENTS, {"w": np.zeros(3, np.uint64)}, root_seed=SEED, job_id=JOB,
                params=P)


# ----------------------------------------------------------------- client vectors


def test_client_vectors_follow_the_plan_with_zeros_for_unassigned_genes():
    s = summaries(200, drop={"SITE-B": 0.3, "SITE-D": 0.2})
    rows = plan_of(s, 3)
    assert not rows.all(axis=None)  # some genes have fewer than 4 clients
    v = ss.contribution_values(s["SITE-B"], rows, "SITE-B", "tested_mean10", ss.QUANTITIES_FE_HET)
    assert set(v) == {"w", "wy", "wy2"} and all(len(a) == len(rows) for a in v.values())
    off = ~rows["SITE-B"].to_numpy()
    assert off.any() and np.all(v["w"][off] == 0) and np.all(v["wy"][off] == 0)
    on = ~off
    pos = pd.Index(s["SITE-B"]["ncbi_gene"]).get_indexer(rows.index[on])
    assert np.allclose(v["w"][on], 1 / s["SITE-B"]["se"][pos] ** 2, rtol=0, atol=0)
    assert np.allclose(v["wy2"][on], v["wy"][on] * s["SITE-B"]["log2fc"][pos], rtol=1e-15)


def test_client_refuses_a_plan_gene_it_did_not_flag():
    s = summaries(50)
    rows = plan_of(s)
    s["SITE-C"]["tested_mean10"] = s["SITE-C"]["tested_mean10"].copy()
    s["SITE-C"]["tested_mean10"][7] = False
    with pytest.raises(ss.SecureSumError, match="didn't flag"):
        ss.contribution_values(s["SITE-C"], rows, "SITE-C", "tested_mean10", ss.QUANTITIES_FE)


def test_excluded_genes_are_absent_from_the_vectors(tmp_path):
    s = summaries(80)
    rows = plan_of(s)
    keep = rows.drop(index=rows.index[[3, 10, 11]])
    pay, fe = run_masked(s, keep)
    assert all(len(v) == len(keep) for p in pay.values() for v in p.values())
    assert not set(rows.index[[3, 10, 11]]) & set(fe["ncbi_gene"])


# ----------------------------------------------------------------- totals and FE


@pytest.mark.parametrize("min_sites", ["all", 3, 2])
def test_masked_fe_equals_plain_fe_within_the_bound(min_sites):
    s = summaries(400, seed=3, drop={"SITE-A": 0.25, "SITE-C": 0.35})
    rows = plan_of(s, min_sites)
    _, fe = run_masked(s, rows, ss.QUANTITIES_FE_HET)
    ref = plain(s, rows)
    b = ss.error_bound(ref["fe_est"], 1 / ref["fe_se"] ** 2, ref["k"], 40)
    slack = 64 * np.finfo(float).eps * (1 + np.abs(ref["fe_est"]))
    for col, bound in (("fe_est", b["est"] + slack), ("fe_se", b["se"] + 64e-16 * ref["fe_se"]),
                       ("q", b["q"] + 64e-16 * (1 + ref["q"]))):
        d = np.abs(fe[col].to_numpy() - ref[col].to_numpy())
        assert np.all(d <= bound), col
    assert np.array_equal(fe["k"], ref["k"])
    assert np.array_equal(fe["fe_q"] < 0.05, ref["fe_q"] < 0.05)
    assert np.allclose(fe["fe_p"], ref["fe_p"], rtol=1e-9, atol=0)
    assert np.allclose(fe["i2"], ref["i2"], rtol=0, atol=1e-9, equal_nan=True)
    assert np.array_equal(fe["q_p"] < 0.10, ref["q_p"] < 0.10)


def test_genes_below_two_sites_get_no_estimate_like_pool():
    s = summaries(100, drop={"SITE-A": 0.5, "SITE-B": 0.5, "SITE-C": 0.5, "SITE-D": 0.5})
    rows = plan_of(s, 1)
    _, fe = run_masked(s, rows)
    one = fe["k"] == 1
    assert one.any()
    assert fe.loc[one, ["fe_est", "fe_se", "fe_p", "fe_q"]].isna().all(axis=None)
    assert fe.loc[~one, "fe_q"].notna().all()


def test_negative_q_residue_is_clamped_and_counted():
    # Identical estimates everywhere: the true Q is 0; rounding can leave a tiny negative.
    s = summaries(200, seed=5)
    for c in CLIENTS:
        s[c]["log2fc"] = s["SITE-A"]["log2fc"].copy()
    rows = plan_of(s)
    _, fe = run_masked(s, rows, ss.QUANTITIES_FE_HET)
    assert (fe["q"] >= 0).all()
    assert fe["q"].max() < 1e-9
    assert fe.attrs["n_q_clamped"] >= 0


# ----------------------------------------------------------------- the boundary


def _meta(client, rows, quantities=ss.QUANTITIES_FE):
    base = {k: None for k in ss.CONTRIBUTION_META_FIELDS}
    base.update(client=client, site=client, stratum="iss", n_flight=4, n_ground=4,
                job_id=JOB, gene_plan_sha256="0" * 64, genes_exclude_sha256=None,
                n_rows=len(rows), fixed_point_bits=40, modulus_bits=64,
                quantities=list(quantities), mask_scheme=ss.MASK_SCHEME)
    return base


def test_validator_accepts_a_contribution_and_refuses_everything_else():
    s = summaries(40)
    rows = plan_of(s)
    pay, _ = run_masked(s, rows)
    good = _meta("SITE-A", rows)
    ss.validate_contribution(pay["SITE-A"], good)
    bad_cases = [
        ({**pay["SITE-A"], "log2fc": s["SITE-A"]["log2fc"]}, good, "arrays"),
        ({"masked_w": pay["SITE-A"]["masked_w"]}, good, "arrays"),
        ({**pay["SITE-A"], "masked_w": pay["SITE-A"]["masked_w"].astype(np.float64)}, good,
         "uint64"),
        ({**pay["SITE-A"], "masked_w": pay["SITE-A"]["masked_w"][:-1]}, good, "length"),
        (pay["SITE-A"], {**good, "log2fc": [1.0]}, "fields"),
        (pay["SITE-A"], {k: v for k, v in good.items() if k != "n_rows"}, "fields"),
        (pay["SITE-A"], {**good, "mask_scheme": "none"}, "mask_scheme"),
        (pay["SITE-A"], {**good, "modulus_bits": 128}, "2\\^64"),
    ]
    for arrays, m, match in bad_cases:
        with pytest.raises(ss.SecureSumError, match=match):
            ss.validate_contribution(arrays, m)
    with pytest.raises(ss.SecureSumError, match="gene_plan_sha256"):
        ss.validate_contribution(pay["SITE-A"], good, {"gene_plan_sha256": "1" * 64})
    with pytest.raises(ss.SecureSumError, match="fixed_point_bits"):
        ss.validate_contribution(pay["SITE-A"], good, {"fixed_point_bits": 32})
    with pytest.raises(TypeError):
        ss.validate_contribution(pay["SITE-A"], {**good, "design": np.zeros(3)})


def test_aggregate_refuses_unknown_clients_and_wrong_lengths():
    s = summaries(30)
    rows = plan_of(s)
    pay, _ = run_masked(s, rows)
    with pytest.raises(ss.SecureSumError, match="partial sum"):
        ss.aggregate({**pay, "SITE-Z": pay["SITE-A"]}, rows, P)
    short = {c: {k: v[:-1] for k, v in p.items()} for c, p in pay.items()}
    with pytest.raises(ss.SecureSumError, match="length"):
        ss.aggregate(short, rows, P)


# ----------------------------------------------------------------- nothing per-site reaches it

_OPENED: list[str] | None = None


def _audit(event, args):
    if _OPENED is not None and event == "open" and isinstance(args[0], (str, Path)):
        _OPENED.append(str(Path(args[0]).resolve()))


sys.addaudithook(_audit)


def test_server_side_sees_only_masked_vectors_and_whitelisted_metadata(tmp_path):
    """WS-09's audit-hook pattern. The site side writes its plain summary to its own folder;
    the server side reads the plan and the received payloads only, and what it receives
    contains no per-site value."""
    global _OPENED
    s = summaries(120, seed=9, drop={"SITE-B": 0.2})
    rows = plan_of(s, 3)
    plan_path = tmp_path / "policy" / "plan.csv"
    sha = write_gene_plan(rows, plan_path)
    sites = tmp_path / "sites"
    received = {}
    for c in CLIENTS:  # the client side
        local = sites / c / "local"
        local.mkdir(parents=True)
        np.savez(local / "summary.npz", **s[c])
        vals = ss.contribution_values(s[c], rows, c, "tested_mean10", ss.QUANTITIES_FE_HET)
        arrays = ss.masked_contribution(vals, c, CLIENTS, root_seed=SEED, job_id=JOB, params=P)
        m = _meta(c, rows, ss.QUANTITIES_FE_HET) | {"gene_plan_sha256": sha}
        ss.validate_contribution(arrays, m)
        received[c] = (arrays, m)
        enc = {q: ss.encode(v, P, 4) for q, v in vals.items()}
        for q in vals:  # no received vector is the client's plain encoded vector
            assert not np.array_equal(arrays[f"masked_{q}"], enc[q])

    _OPENED = []
    try:  # the server side
        server_rows = read_gene_plan(plan_path, sha)
        for arrays, m in received.values():
            ss.validate_contribution(arrays, m, {"gene_plan_sha256": sha, "n_rows": len(rows)})
        fe = ss.fe_from_totals(ss.aggregate({c: a for c, (a, _) in received.items()},
                                            server_rows, P))
        opened = list(_OPENED)
    finally:
        _OPENED = None
    ours = {p for p in opened if p.startswith(str(tmp_path.resolve()))}
    assert ours == {str(plan_path.resolve())}
    assert not any(str(sites.resolve()) in p for p in opened)
    for arrays, m in received.values():
        assert all(a.dtype == np.uint64 for a in arrays.values())
        assert set(m) == set(ss.CONTRIBUTION_META_FIELDS)
        assert not any(k in m for k in ("log2fc", "se", "base_mean"))
    assert len(fe) == len(rows) and fe["fe_est"].notna().all()
