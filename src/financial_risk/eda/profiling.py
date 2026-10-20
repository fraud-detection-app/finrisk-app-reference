"""The EDA checks a risk team runs on every dataset, as functions instead of notebook cells.

Four of these -- WOE/IV, PSI, IQR fences, missingness-vs-target -- are the standard
toolkit of credit-risk and fraud analytics, not general-purpose data science. They exist
because a binary, heavily imbalanced target (default, fraud, churn) breaks the intuitions
that correlation and scatter plots are built on:

* `information_value` ranks a feature by how well it separates the two classes, including
  non-linear and threshold-shaped relationships that Pearson correlation scores near zero.
* `population_stability_index` compares two samples of the SAME feature -- development vs
  this month's production traffic -- and is the standard trigger for "the world moved,
  refit the model".

The cut-offs in IV_BANDS and PSI_BANDS are the conventional industry ones (Siddiqi,
*Credit Risk Scorecards*, and standard model-monitoring practice). They are rules of
thumb agreed on by convention, not laws -- treat a breach as "go investigate", never as
an automatic decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Conventional interpretation bands. Lower bound -> label.
IV_BANDS: list[tuple[float, str]] = [
    (0.50, "suspicious - check for leakage"),
    (0.30, "strong"),
    (0.10, "medium"),
    (0.02, "weak"),
    (0.00, "unpredictive"),
]

PSI_BANDS: list[tuple[float, str]] = [
    (0.25, "significant shift - investigate/refit"),
    (0.10, "moderate shift - monitor"),
    (0.00, "stable"),
]


def _band(value: float, bands: list[tuple[float, str]]) -> str:
    for lower, label in bands:
        if value >= lower:
            return label
    return bands[-1][1]


def first_look(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: dtype, missing, distinct, and an example value.

    The scan that comes before any chart. `df.info()` prints something similar but
    returns None, so it cannot be asserted on in a pipeline or written to a report.
    """
    rows = []
    for col in df.columns:
        s = df[col]
        non_null = s.notna()
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "missing": int((~non_null).sum()),
                "missing_pct": round(float((~non_null).mean() * 100), 2),
                "distinct": int(s.nunique(dropna=True)),
                "example": s[non_null].iloc[0] if non_null.any() else None,
            }
        )
    return pd.DataFrame(rows)


def missingness_report(df: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
    """Missing counts per column -- and, given a target, whether MISSING itself predicts it.

    The second half is the point. A column that is missing 40% of the time is a nuisance;
    a column whose missingness carries a different target rate is a *feature*, and
    imputing it silently throws that signal away.
    """
    rows = []
    for col in df.columns:
        if col == target:
            continue
        is_missing = df[col].isna()
        if not is_missing.any():
            continue
        row = {
            "column": col,
            "missing": int(is_missing.sum()),
            "missing_pct": round(float(is_missing.mean() * 100), 2),
        }
        if target is not None:
            row["target_rate_when_missing"] = round(float(df.loc[is_missing, target].mean()), 4)
            row["target_rate_when_present"] = round(float(df.loc[~is_missing, target].mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def outlier_summary(df: pd.DataFrame, columns: list[str] | None = None, k: float = 1.5) -> pd.DataFrame:
    """Tukey IQR fences per numeric column: how many points sit beyond q1-k*IQR / q3+k*IQR.

    `k=1.5` is Tukey's convention. On heavy-tailed money columns it flags a large share of
    perfectly legitimate rows, which is exactly why the output is a SUMMARY to reason about
    and not a filter -- see the `share_beyond` column before deciding to drop anything.
    """
    columns = columns or df.select_dtypes("number").columns.tolist()
    rows = []
    for col in columns:
        s = df[col].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - k * iqr, q3 + k * iqr
        beyond = int(((s < low) | (s > high)).sum())
        rows.append(
            {
                "column": col,
                "q1": round(float(q1), 2),
                "q3": round(float(q3), 2),
                "lower_fence": round(float(low), 2),
                "upper_fence": round(float(high), 2),
                "beyond_fences": beyond,
                "share_beyond": round(beyond / len(s) * 100, 2),
                "max": round(float(s.max()), 2),
            }
        )
    return pd.DataFrame(rows).sort_values("share_beyond", ascending=False).reset_index(drop=True)


def woe_table(
    df: pd.DataFrame, feature: str, target: str, bins: int = 5, smoothing: float = 0.5
) -> pd.DataFrame:
    """Bin a feature and report event rate, Weight of Evidence and IV contribution per bin.

    WOE = ln( (events in bin / all events) / (non-events in bin / all non-events) ).
    Positive WOE = the bin over-represents the event. A *monotonic* WOE column is what
    scorecard developers look for: it means the feature behaves consistently across its range.

    `smoothing` (0.5 by default) keeps a bin with zero events from producing log(0); with a
    ~1% event rate, empty bins are common on small samples.
    """
    s = df[feature]
    if pd.api.types.is_numeric_dtype(s) and s.nunique() > bins:
        binned = pd.qcut(s, bins, duplicates="drop")
    else:
        binned = s.astype(str)

    g = df.groupby(binned, observed=True)[target].agg(["count", "sum"])
    g.columns = ["n", "events"]
    g["non_events"] = g["n"] - g["events"]

    total_events = g["events"].sum()
    total_non_events = g["non_events"].sum()
    if total_events == 0 or total_non_events == 0:
        raise ValueError(f"{target!r} has only one class -- WOE/IV are undefined")

    event_share = (g["events"] + smoothing) / total_events
    non_event_share = (g["non_events"] + smoothing) / total_non_events

    g["event_rate"] = (g["events"] / g["n"]).round(5)
    g["woe"] = np.log(event_share / non_event_share).round(4)
    g["iv"] = ((event_share - non_event_share) * g["woe"]).round(5)
    return g.reset_index().rename(columns={g.index.name or "index": "bin"})


def information_value(df: pd.DataFrame, feature: str, target: str, bins: int = 5) -> float:
    """Total IV for one feature: the sum of its per-bin IV contributions."""
    return float(woe_table(df, feature, target, bins=bins)["iv"].sum())


def rank_features_by_iv(
    df: pd.DataFrame, target: str, exclude: list[str] | None = None, bins: int = 5
) -> pd.DataFrame:
    """Screen every candidate feature by IV and attach the conventional band label.

    This is the first pass of scorecard development: rank, then read the top AND the
    bottom of the list. An implausibly high IV is a leakage alarm, not a trophy.
    """
    exclude = set(exclude or []) | {target}
    rows = []
    for col in df.columns:
        if col in exclude:
            continue
        s = df[col]
        # Skip free-form identifiers: one bin per row carries no generalisable signal.
        if not pd.api.types.is_numeric_dtype(s) and s.nunique() > 50:
            continue
        try:
            iv = information_value(df, col, target, bins=bins)
        except (ValueError, TypeError):
            continue
        rows.append({"feature": col, "iv": round(iv, 4), "band": _band(iv, IV_BANDS)})
    return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)


def population_stability_index(
    expected: pd.Series | np.ndarray, actual: pd.Series | np.ndarray, bins: int = 10
) -> float:
    """PSI between a baseline sample and a later one, using baseline quantile bins.

    PSI = sum over bins of (actual% - expected%) * ln(actual% / expected%).

    Bin edges come from `expected` (the development sample) so the question is always
    "where did the new traffic land relative to the population the model was built on".
    """
    expected = np.asarray(pd.Series(expected).dropna(), dtype=float)
    actual = np.asarray(pd.Series(actual).dropna(), dtype=float)
    if expected.size == 0 or actual.size == 0:
        raise ValueError("PSI needs non-empty expected and actual samples")

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    exp_share = np.histogram(expected, bins=edges)[0] / expected.size
    act_share = np.histogram(actual, bins=edges)[0] / actual.size
    # Floor both shares: an empty bin on either side would send the log to +/-inf.
    exp_share = np.clip(exp_share, 1e-4, None)
    act_share = np.clip(act_share, 1e-4, None)
    return float(np.sum((act_share - exp_share) * np.log(act_share / exp_share)))


def psi_band(value: float) -> str:
    """Conventional label for a PSI value."""
    return _band(value, PSI_BANDS)
