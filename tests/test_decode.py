"""Decoding a release with no language model: the maximum-entropy table of the released counts,
sampled and passed through the selection step (`cortec.decode`). Each test is named after the
failure it prevents. Needs private-pgm (mbi 1.1) and JAX (`pip install 'cortec[pgm]'`); the tests
are skipped without them."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mbi")
pytest.importorskip("jax")

from cortec import Schema, Band, release_statistics, decode_release, fit_release_model, DecodeError
from cortec.release import _cohort_keys


def _schema():
    return Schema(name="sel", numerical={"age": (18, 90)}, categorical={"grp": ["a", "b", "c"]},
                  target="y", positive="YES", negative="NO", bins={"age": [18, 40, 60, 90]},
                  stratify=[("age", [Band("young", 18, 40), Band("old", 40, 90)])],
                  conditional=[("grp",)])


def _private(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 90, n)
    grp = rng.choice(["a", "b", "c"], n, p=[0.5, 0.3, 0.2])
    y = np.where(rng.random(n) < 0.15 + 0.5 * (age > 60) + 0.2 * (grp == "c"), "YES", "NO")
    return pd.DataFrame({"age": age, "grp": grp, "y": y})


@pytest.fixture(scope="module")
def fitted():
    schema = _schema(); private = _private()
    rel = release_statistics(schema, private, epsilon_total=2.0, n_min=150, max_rows_per_person=1,
                             autoconfig=False, seed=3)
    rm = fit_release_model(schema, rel, iters=400, seed=1, verbose=False)
    return schema, rel, rm


def test_decoded_rows_carry_the_released_cohort_sizes_and_class_balance(fitted):
    schema, rel, rm = fitted
    df = decode_release(schema, rel, 300, pool_factor=2, seed=1, model=rm, verbose=False)
    assert len(df) == 300 and list(df.columns) == ["age", "grp", "y"]
    assert df["age"].between(18, 90).all() and (df["age"] % 1 == 0).all(), "integral bounds and edges give integer ages"
    assert df["grp"].isin(["a", "b", "c"]).all() and df["y"].isin(["YES", "NO"]).all()
    coh = _cohort_keys(rel.effective_schema(schema), df)
    total = sum(c["cohort_size"] for c in rel.cohorts)
    for c in rel.cohorts:
        name = c["cohort_name"]
        assert abs((coh == name).mean() - c["cohort_size"] / total) < 0.08, name
        pos = (df.loc[coh == name, "y"] == "YES").mean()
        assert abs(pos - c["class_balance"]["YES"]) < 0.1, (name, pos, c["class_balance"])


def test_the_table_alone_carries_the_released_rates_without_selection(fitted):
    schema, rel, rm = fitted
    df = decode_release(schema, rel, 2000, select=False, seed=2, model=rm, verbose=False)
    assert len(df) == 2000
    total = sum(c["cohort_size"] for c in rel.cohorts)
    want = sum(c["cohort_size"] * c["class_balance"]["YES"] for c in rel.cohorts) / total
    assert abs((df["y"] == "YES").mean() - want) < 0.05
    for key, cell in rel.conditional_levels[0]["cells"].items():
        g = key.split("=", 1)[1]
        got = (df.loc[df["grp"] == g, "y"] == "YES").mean()
        assert abs(got - cell["rate"]) < 0.1, (key, got, cell["rate"])


def test_a_fitted_model_is_reusable_and_seeded(fitted):
    schema, rel, rm = fitted
    a = decode_release(schema, rel, 100, pool_factor=2, seed=7, model=rm, verbose=False)
    b = decode_release(schema, rel, 100, pool_factor=2, seed=7, model=rm, verbose=False)
    pd.testing.assert_frame_equal(a, b)
    assert rm.consistency_mean_tv < 0.15, "the fitted table should reproduce the released per-cohort shares"


def test_a_release_of_another_schema_is_refused(fitted):
    schema, rel, rm = fitted
    other = Schema(name="other", numerical={"age": (18, 90)}, categorical={"grp": ["a", "b", "c"]},
                   target="y", positive="YES", negative="NO", bins={"age": [18, 40, 60, 90]})
    with pytest.raises(DecodeError):
        fit_release_model(other, rel, iters=10, verbose=False)
    with pytest.raises(DecodeError):
        decode_release(schema, rel, 0, model=rm, verbose=False)


def test_the_command_writes_the_draws_and_a_manifest(fitted, tmp_path):
    schema, rel, rm = fitted
    rel.to_json(str(tmp_path / "release.json"))
    (tmp_path / "schema.py").write_text(
        'from cortec import Schema, Band\n'
        'SCHEMA = Schema(name="sel", numerical={"age": (18, 90)}, categorical={"grp": ["a", "b", "c"]}, '
        'target="y", positive="YES", negative="NO", bins={"age": [18, 40, 60, 90]}, '
        'stratify=[("age", [Band("young", 18, 40), Band("old", 40, 90)])], conditional=[("grp",)])\n')
    from cortec.decode import main
    rc = main(["--schema", str(tmp_path / "schema.py"), "--release", str(tmp_path / "release.json"),
               "--n-rows", "120", "--draws", "2", "--pool-factor", "2", "--iters", "300",
               "--out", str(tmp_path / "out")])
    assert rc == 0
    for k in range(2):
        assert len(pd.read_csv(tmp_path / "out" / f"decoded_draw{k}.csv")) == 120
    m = json.load(open(tmp_path / "out" / "decode_manifest.json"))
    assert m["privacy_cost"] == 0.0 and m["private_data_read"] is False and m["selected"] is True
