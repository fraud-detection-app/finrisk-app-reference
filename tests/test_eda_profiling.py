from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from financial_risk.eda import (
    IV_BANDS,
    first_look,
    information_value,
    missingness_report,
    outlier_summary,
    population_stability_index,
    rank_features_by_iv,
    woe_table,
)
from financial_risk.eda.profiling import psi_band


def _frame(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """A tiny risk-shaped table: rare binary target, one signal column, one noise column."""
    rng = np.random.default_rng(seed)
    signal = rng.uniform(0, 1, n)
    return pd.DataFrame(
        {
            "signal": signal,
            "noise": rng.normal(size=n),
            "segment": rng.choice(["a", "b", "c"], n),
            # event probability rises with `signal`, so IV should rank it above noise
            "target": (rng.uniform(0, 1, n) < 0.01 + 0.06 * signal).astype(int),
        }
    )


def test_first_look_reports_every_column_and_counts_missing() -> None:
    df = _frame(200)
    df.loc[:19, "noise"] = np.nan

    profile = first_look(df).set_index("column")

    assert set(profile.index) == set(df.columns)
    assert profile.loc["noise", "missing"] == 20
    assert profile.loc["noise", "missing_pct"] == pytest.approx(10.0)
    assert profile.loc["signal", "missing"] == 0


def test_missingness_report_compares_target_rate_and_skips_complete_columns() -> None:
    df = _frame(600)
    # Make missingness carry signal: every missing row is an event.
    missing_rows = df.index[:100]
    df.loc[missing_rows, "noise"] = np.nan
    df.loc[missing_rows, "target"] = 1

    report = missingness_report(df, target="target").set_index("column")

    assert list(report.index) == ["noise"], "columns without nulls must not appear"
    assert report.loc["noise", "target_rate_when_missing"] == 1.0
    assert report.loc["noise", "target_rate_when_missing"] > report.loc["noise", "target_rate_when_present"]


def test_missingness_report_is_empty_when_nothing_is_missing() -> None:
    assert missingness_report(_frame(100), target="target").empty


def test_outlier_summary_flags_only_the_planted_extremes() -> None:
    df = pd.DataFrame({"x": [10.0] * 100 + [10_000.0, 20_000.0]})

    row = outlier_summary(df, ["x"]).iloc[0]

    assert row["beyond_fences"] == 2
    assert row["max"] == 20_000.0
    assert row["upper_fence"] < 10_000.0


def test_woe_table_bins_are_complete_and_iv_sums_to_information_value() -> None:
    df = _frame()

    table = woe_table(df, "signal", "target", bins=5)

    assert table["n"].sum() == len(df)
    assert table["events"].sum() == df["target"].sum()
    assert information_value(df, "signal", "target", bins=5) == pytest.approx(table["iv"].sum())


def test_woe_survives_a_bin_with_zero_events() -> None:
    # Without smoothing this is a log(0) -> -inf crash; the event-free bin is the point.
    df = pd.DataFrame({"x": list(range(100)), "target": [0] * 90 + [1] * 10})

    table = woe_table(df, "x", "target", bins=5)

    assert np.isfinite(table["woe"]).all()
    assert np.isfinite(table["iv"]).all()


def test_woe_requires_both_classes() -> None:
    df = pd.DataFrame({"x": range(50), "target": [0] * 50})

    with pytest.raises(ValueError, match="only one class"):
        woe_table(df, "x", "target")


def test_information_value_is_invariant_under_monotonic_transform() -> None:
    """IV works on quantile bins, so log() cannot inflate it -- unlike correlation."""
    df = _frame()
    df["log_signal"] = np.log1p(df["signal"])

    assert information_value(df, "signal", "target") == pytest.approx(
        information_value(df, "log_signal", "target")
    )


def test_information_value_sees_a_threshold_effect_that_correlation_misses() -> None:
    """The lab's central claim: Pearson ~0 while IV is clearly predictive."""
    rng = np.random.default_rng(7)
    n = 4000
    # Symmetric feature; only its TAILS carry events, so the linear relationship is ~nil.
    x = rng.normal(size=n)
    target = (rng.uniform(size=n) < np.where(np.abs(x) > 1.5, 0.10, 0.005)).astype(int)
    df = pd.DataFrame({"x": x, "target": target})

    corr = abs(df["x"].corr(df["target"]))
    iv = information_value(df, "x", "target", bins=10)

    assert corr < 0.05, "the feature must look useless to correlation"
    assert iv > 0.10, "IV must still rank it at least medium-predictive"


def test_rank_features_by_iv_orders_signal_above_noise_and_labels_bands() -> None:
    df = _frame()

    ranked = rank_features_by_iv(df, "target")

    assert "target" not in ranked["feature"].values
    order = ranked["feature"].tolist()
    assert order.index("signal") < order.index("noise")
    assert ranked["iv"].is_monotonic_decreasing
    assert set(ranked["band"]) <= {label for _, label in IV_BANDS}


def test_psi_is_zero_for_identical_samples_and_grows_with_the_shift() -> None:
    rng = np.random.default_rng(3)
    base = rng.normal(100, 15, 5000)

    same = population_stability_index(base, base)
    mild = population_stability_index(base, base * 1.10)
    severe = population_stability_index(base, base * 2.0)

    assert same == pytest.approx(0.0, abs=1e-9)
    assert same < mild < severe
    assert psi_band(same) == "stable"
    assert psi_band(severe) == "significant shift - investigate/refit"


def test_psi_rejects_an_empty_sample() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        population_stability_index(pd.Series([1.0, 2.0]), pd.Series([], dtype=float))
