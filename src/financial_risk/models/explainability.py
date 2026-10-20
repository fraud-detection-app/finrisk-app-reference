from __future__ import annotations

import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from financial_risk.models.training import FEATURE_COLUMNS


def _dense(x: object) -> object:
    return x.toarray() if hasattr(x, "toarray") else x


def _transformed_feature_names(pipeline: Pipeline) -> list[str]:
    return list(pipeline.named_steps["preprocess"].get_feature_names_out())


def build_explainer(pipeline: Pipeline, background: pd.DataFrame) -> shap.Explainer:
    """`shap.Explainer` auto-selects Tree/Linear/Permutation based on the estimator type, so this
    one call works for the Week 7 logistic-regression baseline and the Week 8 tree/boosted
    optimized model without branching on model type here."""

    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    background_transformed = _dense(preprocess.transform(background[FEATURE_COLUMNS]))
    return shap.Explainer(model, background_transformed, feature_names=_transformed_feature_names(pipeline))


def _path_dependent(pipeline: Pipeline) -> shap.Explainer:
    # shap >= 0.52 refuses interventional TreeSHAP (the background-data mode) on some XGBoost
    # models. Path-dependent TreeSHAP is still exact and needs no background, so a fresh install
    # keeps working without pinning an old shap.
    return shap.TreeExplainer(
        pipeline.named_steps["model"],
        feature_perturbation="tree_path_dependent",
        feature_names=_transformed_feature_names(pipeline),
    )


def make_explainer(pipeline: Pipeline, background: pd.DataFrame) -> shap.Explainer:
    """Build a SHAP explainer ONCE, for reuse on every row.

    Building one costs ~15 ms on the demo model (the background pass, plus -- on shap >= 0.52 --
    a failed interventional attempt before the fallback); evaluating it on one row costs ~0.2 ms.
    A service that builds it per request is paying a reload tax, exactly like reloading the
    model. So we decide which mode works here, once, by probing one background row."""
    explainer = build_explainer(pipeline, background)
    probe = _dense(pipeline.named_steps["preprocess"].transform(background[FEATURE_COLUMNS].head(1)))
    try:
        explainer(probe)
        return explainer
    except NotImplementedError:
        return _path_dependent(pipeline)


def compute_shap_values(
    pipeline: Pipeline,
    x: pd.DataFrame,
    background: pd.DataFrame | None,
    explainer: shap.Explainer | None = None,
) -> pd.DataFrame:
    """Pass a prebuilt ``explainer`` (from ``make_explainer``) to skip rebuilding it per call."""
    preprocess = pipeline.named_steps["preprocess"]
    x_transformed = _dense(preprocess.transform(x[FEATURE_COLUMNS]))
    if explainer is not None:
        explanation = explainer(x_transformed)
    else:
        try:
            explanation = build_explainer(pipeline, background)(x_transformed)
        except NotImplementedError:
            explanation = _path_dependent(pipeline)(x_transformed)

    values = explanation.values
    if values.ndim == 3:
        # Some explainers return (n_rows, n_features, n_classes) for classifiers; keep the
        # positive ("is_fraud"=1) class.
        values = values[:, :, -1]
    return pd.DataFrame(values, columns=explanation.feature_names, index=x.index)


def global_feature_importance(
    pipeline: Pipeline,
    sample: pd.DataFrame,
    background: pd.DataFrame,
    top_n: int = 15,
) -> pd.DataFrame:
    shap_df = compute_shap_values(pipeline, sample, background)
    importance = shap_df.abs().mean().sort_values(ascending=False).head(top_n)
    return importance.rename("mean_abs_shap").reset_index().rename(columns={"index": "feature"})


def local_reasons(
    pipeline: Pipeline,
    row: pd.DataFrame,
    background: pd.DataFrame | None,
    top_n: int = 4,
    explainer: shap.Explainer | None = None,
) -> list[str]:
    """Per-transaction, model-driven explanation: the top SHAP contributors for this one row,
    replacing/augmenting the hand-crafted heuristic reasons in `inference.heuristic_reasons`."""

    shap_df = compute_shap_values(pipeline, row, background, explainer=explainer)
    contributions = shap_df.iloc[0].reindex(shap_df.iloc[0].abs().sort_values(ascending=False).index).head(top_n)
    reasons = []
    for feature, value in contributions.items():
        direction = "pushes risk up" if value > 0 else "pushes risk down"
        # Drop the ColumnTransformer prefix (numeric__ / categorical__): these strings are
        # read by analysts and fed to the LLM explainer, so they should name the feature.
        name = feature.split("__", 1)[1] if "__" in feature else feature
        reasons.append(f"`{name}` {direction} (SHAP contribution {value:+.3f}).")
    return reasons
