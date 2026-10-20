"""The model package: what a trained model looks like on disk, and how to open it.

finrisk-ml writes it, SageMaker unpacks it into ``/opt/ml/model`` when an endpoint starts, and this repo's
code reads it -- on a laptop, in the tests, in the training job (to compare a candidate with the champion)
and inside the endpoint (``serving/sagemaker_server.py``). One reader for all of them, so a package that
loads in one place loads in every place.

The layout is flat, at the top level of ``model.tar.gz``:

    model_v2.pkl / model_v1.pkl   joblib dict: {"model": <sklearn Pipeline>, "threshold", "feature_columns",
                                  "model_type", "metrics", "created_at_utc", ...}
    latest_model.joblib           {"model_path": "model_v2.pkl"}  -- a file NAME, resolved inside the folder
    metrics.json                  the numbers the training job measured (informational)

There is NO ``features.parquet`` any more (13b, SageMaker): six million rows do not belong in a model package.
The service scores ``/api/predict/<row_id>`` from its own feature table instead. An older package that still
carries one is accepted; the file is simply never read.

Public interface (stable; finrisk-ml imports these):

    unpack_model_package(archive_path, dest)   model.tar.gz -> folder, refusing unsafe entries
    load_model_package(model_dir)              folder -> the artifact dict, checked against FEATURE_COLUMNS
"""

from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Any

from financial_risk.models.evaluation import load_latest_artifact
from financial_risk.models.training import FEATURE_COLUMNS

__all__ = ["ModelPackageError", "load_model_package", "unpack_model_package"]


class ModelPackageError(ValueError):
    """The package cannot be used by this code. The message says why, in words a student can act on."""


def _safe_members(archive: tarfile.TarFile, dest: Path) -> list[tarfile.TarInfo]:
    """Only plain files and folders that stay inside ``dest`` (no ../, no absolute paths, no links)."""
    root = dest.resolve()
    members = []
    for m in archive.getmembers():
        target = (dest / m.name).resolve()
        if not (m.isfile() or m.isdir()) or (target != root and root not in target.parents):
            raise ModelPackageError(f"unsafe entry in model.tar.gz: {m.name!r}")
        members.append(m)
    return members


def unpack_model_package(archive_path: Path, dest: Path) -> None:
    """model.tar.gz -> ``dest``. Checks that the package holds a model."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        members = _safe_members(archive, dest)
        extra = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        archive.extractall(dest, members=members, **extra)
    names = sorted(p.name for p in dest.iterdir())
    if "latest_model.joblib" not in names and not any(n.startswith("model_v") for n in names):
        raise ModelPackageError(
            f"model.tar.gz holds no model (expected latest_model.joblib / model_v*.pkl; found {names})")


def load_model_package(model_dir: Path) -> dict[str, Any]:
    """Open an unpacked package: the artifact dict (``model``, ``threshold``, ``feature_columns``, ``metrics`` ...).

    Raises ``ModelPackageError`` when there is no model, or when the model was trained on other columns than this
    code expects -- better a clear refusal at load than a shape error on the first request (or, worse, a score
    computed from the wrong columns)."""
    model_dir = Path(model_dir)
    try:
        artifact = load_latest_artifact(model_dir)
    except FileNotFoundError as exc:
        raise ModelPackageError(f"no model in {model_dir}: {exc}") from exc
    if "model" not in artifact or "threshold" not in artifact:
        raise ModelPackageError(f"the model file in {model_dir} lacks 'model' / 'threshold' (is it a finrisk model?)")
    trained_on = artifact.get("feature_columns")
    if trained_on is not None and list(trained_on) != list(FEATURE_COLUMNS):
        old = sorted(set(trained_on) - set(FEATURE_COLUMNS))
        new = sorted(set(FEATURE_COLUMNS) - set(trained_on))
        raise ModelPackageError(
            "this model was trained on different features than this code scores with "
            f"(it has {old or 'nothing extra'}, it lacks {new or 'nothing'}). "
            "Retrain it with this image: 13b replaced the feature `step` with `hour_of_day`.")
    return artifact
