"""Gene plan: which clients contribute to which gene (CONTRACTS §5.3, §6.1).

The plan is built by the coordinator from the clients' released inclusion flags (site_metadata,
§5.1) and the spec's inclusion rule. It drives both the per-gene policy checks and, later, what
each client sends (WS-13). The canonical CSV makes its hash reproducible by hand.
"""

import hashlib

import numpy as np
import pandas as pd
import pytest
from fedkg_spacebio.federation.gene_plan import (
    INCLUSION_FLAG,
    GenePlanError,
    build_gene_plan,
    canonical_gene_list_csv,
    canonical_plan_csv,
    exclude_genes,
    gene_plan_sha256,
    read_gene_list,
    read_gene_plan,
    write_gene_list,
    write_gene_plan,
)

from policy_helpers import FIX, G1, G3, G4, A, B, C, D, plan_frame


def example_b_flags(clients=(A, B, C, D)) -> dict[str, pd.DataFrame]:
    """Released flags per client: A tests all 3 genes, B misses FAKE:0003, D misses FAKE:0004."""
    df = pd.read_csv(FIX / "flags_example_b.csv")
    out = {}
    for c in clients:
        sub = df[df["client"] == c].drop(columns="client").reset_index(drop=True)
        for f in INCLUSION_FLAG.values():
            sub[f] = sub[f].astype(bool)
        out[c] = sub
    return out


def as_sets(plan: pd.DataFrame) -> dict[int, set[str]]:
    return {int(g): {c for c in plan.columns if row[c]} for g, row in plan.iterrows()}


def test_inclusion_rules_map_to_flags():
    assert INCLUSION_FLAG == {"mean10_finite": "tested_mean10",
                              "indep_filter": "tested_indep_filter",
                              "finite_p": "tested_finite_p"}


# ------------------------------------------------------------------ example B's rows, rebuilt


def test_r1_complete_case():
    plan = build_gene_plan(example_b_flags(), "mean10_finite", "all")
    assert as_sets(plan) == {G1: {A, B, C, D}}


def test_r2_at_least_three_sites():
    plan = build_gene_plan(example_b_flags(), "mean10_finite", 3)
    assert as_sets(plan) == {G1: {A, B, C, D}, G3: {A, C, D}, G4: {A, B, C}}


def test_j3_without_spacex8():
    plan = build_gene_plan(example_b_flags((A, C, D)), "mean10_finite", 3)
    assert as_sets(plan) == {G1: {A, C, D}, G3: {A, C, D}}  # FAKE:0004 has k = 2 without B


def test_j4_without_spacex16():
    plan = build_gene_plan(example_b_flags((A, B, C)), "mean10_finite", "all")
    assert as_sets(plan) == {G1: {A, B, C}, G4: {A, B, C}}


def test_j5_spacex4_alone():
    plan = build_gene_plan(example_b_flags((A,)), "mean10_finite", "all")
    assert as_sets(plan) == {G1: {A}, G3: {A}, G4: {A}}


def test_rule_selects_the_flag():
    flags = example_b_flags((A, B))
    flags[B]["tested_finite_p"] = True  # B passes finite_p for every gene
    assert G3 not in as_sets(build_gene_plan(flags, "mean10_finite", "all"))
    assert as_sets(build_gene_plan(flags, "finite_p", "all"))[G3] == {A, B}


def test_unreported_gene_counts_as_not_included():
    flags = example_b_flags((A, B))
    flags[B] = flags[B][flags[B]["ncbi_gene"] != G1]  # B doesn't report FAKE:0001 at all
    plan = build_gene_plan(flags, "mean10_finite", 1)
    assert as_sets(plan)[G1] == {A}


def test_flags_may_carry_ncbi_gene_as_index():
    flags = {c: f.set_index("ncbi_gene") for c, f in example_b_flags().items()}
    assert as_sets(build_gene_plan(flags, "mean10_finite", 3))[G3] == {A, C, D}


def test_bad_inputs_raise():
    flags = example_b_flags((A, B))
    with pytest.raises(GenePlanError):
        build_gene_plan(flags, "mean5", "all")
    with pytest.raises(GenePlanError):
        build_gene_plan(flags, "mean10_finite", 0)
    with pytest.raises(GenePlanError):
        build_gene_plan({}, "mean10_finite", "all")
    dup = {A: pd.concat([flags[A], flags[A]]), B: flags[B]}
    with pytest.raises(GenePlanError):
        build_gene_plan(dup, "mean10_finite", "all")


# ------------------------------------------------------------------ canonical form and hash


def test_canonical_plan_csv_by_hand():
    plan = plan_frame({G4: [A, B], G1: [A, B, D]}, [D, A, B])
    expected = (b"ncbi_gene,SpaceX-16-ISST,SpaceX-4,SpaceX-8\n"
                b"9000000001,1,1,1\n"
                b"9000000004,0,1,1\n")
    assert canonical_plan_csv(plan) == expected
    assert gene_plan_sha256(plan) == hashlib.sha256(expected).hexdigest()


def test_column_and_row_order_dont_change_the_hash():
    p1 = plan_frame({G1: [A, B], G3: [A]}, [A, B])
    p2 = p1[[B, A]].iloc[::-1]
    assert gene_plan_sha256(p1) == gene_plan_sha256(p2)


def test_write_and_read_back(tmp_path):
    plan = plan_frame({G1: [A, B, C], G3: [A, C]}, [A, B, C])
    path = tmp_path / "plans" / "x" / "gene_plan.csv"
    sha = write_gene_plan(plan, path)
    assert path.read_bytes() == canonical_plan_csv(plan)
    back = read_gene_plan(path, sha)
    assert as_sets(back) == as_sets(plan)
    assert back.index.dtype == np.int64


def test_read_refuses_wrong_hash_and_noncanonical_bytes(tmp_path):
    plan = plan_frame({G1: [A, B]}, [A, B])
    path = tmp_path / "p.csv"
    write_gene_plan(plan, path)
    with pytest.raises(GenePlanError):
        read_gene_plan(path, "0" * 64)
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    with pytest.raises(GenePlanError):
        read_gene_plan(path)
    path.write_text("ncbi_gene,SpaceX-4,SpaceX-8\n9000000001,0,0\n")  # a gene nobody sends
    with pytest.raises(GenePlanError):
        read_gene_plan(path)


def test_gene_list_by_hand_and_round_trip(tmp_path):
    assert canonical_gene_list_csv([G4, G1]) == b"ncbi_gene\n9000000001\n9000000004\n"
    sha = write_gene_list([G4, G1], tmp_path / "g.csv")
    assert list(read_gene_list(tmp_path / "g.csv", sha)) == [G1, G4]
    with pytest.raises(GenePlanError):
        canonical_gene_list_csv([G1, G1])
    with pytest.raises(GenePlanError):
        read_gene_list(tmp_path / "g.csv", "f" * 64)


def test_exclude_genes():
    plan = plan_frame({G1: [A, B], G3: [A], G4: [B]}, [A, B])
    assert set(exclude_genes(plan, [G1, G4]).index) == {G3}
    assert set(exclude_genes(plan, [9000000099]).index) == {G1, G3, G4}  # not in plan: no-op
