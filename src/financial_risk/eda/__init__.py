"""Reusable EDA profiling utilities.

Every risk team ends up with an internal library like this one -- the same handful of
checks run on every new dataset and every monthly refresh, so they live in version
control rather than being retyped into a notebook each time.
"""

from financial_risk.eda.profiling import (
    IV_BANDS,
    PSI_BANDS,
    first_look,
    information_value,
    missingness_report,
    outlier_summary,
    population_stability_index,
    rank_features_by_iv,
    woe_table,
)

__all__ = [
    "IV_BANDS",
    "PSI_BANDS",
    "first_look",
    "information_value",
    "missingness_report",
    "outlier_summary",
    "population_stability_index",
    "rank_features_by_iv",
    "woe_table",
]
