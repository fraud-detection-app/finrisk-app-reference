"""The project has to survive being moved -- into a container, onto another laptop, onto AWS.

Bugs the first container runs exposed, pinned here so they cannot come back:

1. ``models/latest_model.joblib`` used to store the model's ABSOLUTE path on the training machine. Mounted into a
   container, that path does not exist, so the model "vanished". (finrisk-ml now writes a bare FILE NAME.)
2. ``/ready`` checked that the model FILE exists, so it said "ready" while every prediction failed.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd

from artifact_fakes import write_tfidf_index
from financial_risk.config import get_settings
from financial_risk.models.evaluation import load_latest_artifact
from financial_risk.service.risk_service import ServiceContext

ROOT = Path(__file__).resolve().parents[1]


def test_absolute_pointer_from_another_machine_still_loads(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    joblib.dump({"model": "stand-in", "threshold": 0.5}, model_dir / "model_v2.pkl")
    # A pointer written on someone else's laptop: absolute, and not valid on this machine.
    joblib.dump({"model_path": "/Users/someone/project/models/model_v2.pkl"}, model_dir / "latest_model.joblib")

    assert load_latest_artifact(model_dir)["threshold"] == 0.5


def test_a_bare_file_name_pointer_resolves_inside_the_unpacked_folder(tmp_path, monkeypatch):
    """What finrisk-ml writes: {"model_path": "model_v2.pkl"}. Whatever the current directory is."""
    model_dir = tmp_path / "unpacked"
    model_dir.mkdir()
    joblib.dump({"model": "stand-in", "threshold": 0.7}, model_dir / "model_v2.pkl")
    joblib.dump({"model_path": "model_v2.pkl"}, model_dir / "latest_model.joblib")
    monkeypatch.chdir(tmp_path)
    assert load_latest_artifact(model_dir)["threshold"] == 0.7


def test_ready_is_false_when_the_model_file_exists_but_cannot_load(tmp_path):
    gold = tmp_path / "gold"
    gold.mkdir()
    pd.DataFrame({"row_id": [1, 2], "amount": [10.0, 20.0]}).to_parquet(gold / "transaction_features.parquet")
    index = tmp_path / "vector_index"
    write_tfidf_index(index, version="baked")
    models = tmp_path / "models"
    models.mkdir()
    (models / "latest_model.joblib").write_bytes(b"not a pickle at all")

    settings = dataclasses.replace(get_settings(), pointer_prefix="", gold_path=gold, model_dir=models, vector_index_dir=index)
    info = ServiceContext.build(settings=settings, eager=True).readiness()

    assert info["ready"] is False
    assert info["checks"]["model_artifact"] is False and info["checks"]["vector_index"] is True
    assert info["load_error"] and info["model"]["load_error"] == info["load_error"]
    assert "could not be loaded" in info["hint"] or "See load_error" in info["hint"]


def test_missing_files_are_still_reported_as_missing(tmp_path):
    settings = dataclasses.replace(get_settings(), pointer_prefix="", gold_path=tmp_path / "gold",
                                   model_dir=tmp_path / "models", vector_index_dir=tmp_path / "vi")
    info = ServiceContext.build(settings=settings, eager=True).readiness()

    assert info["ready"] is False
    assert set(info["missing"]) == {"gold_features", "model_artifact", "vector_index"}
    assert info["load_error"] is None and "run_pipeline" in info["hint"]


def test_run_pipeline_honours_the_path_overrides_the_training_image_uses(tmp_path):
    """finrisk-ml's prepare-data step runs `python /app/scripts/run_pipeline.py` inside this image with
    RAW_DATA_PATH / BRONZE_PATH / SILVER_PATH / GOLD_PATH (and MODEL_DIR ...) pointing at a work folder."""
    import os

    env = dict(os.environ, DATA_MODE="demo", LLM_PROVIDER="none", DEEPSEEK_API_KEY="",
               RAW_DATA_PATH=str(tmp_path / "raw" / "paysim.csv"), BRONZE_PATH=str(tmp_path / "bronze" / "t.parquet"),
               SILVER_PATH=str(tmp_path / "silver" / "c.parquet"), GOLD_PATH=str(tmp_path / "gold"),
               MODEL_DIR=str(tmp_path / "models"), REPORT_DIR=str(tmp_path / "reports"),
               VECTOR_INDEX_DIR=str(tmp_path / "index"), RAG_DOCS_DIR=str(tmp_path / "rag_docs"))
    run = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_pipeline.py"), "--demo-rows", "600"],
                         capture_output=True, text=True, env=env, timeout=240)
    assert run.returncode == 0, run.stderr[-800:]
    assert (tmp_path / "gold" / "transaction_features.parquet").exists()
    assert (tmp_path / "models" / "latest_model.joblib").exists()
    assert (tmp_path / "index" / "manifest.json").exists()
    assert json.loads((tmp_path / "index" / "manifest.json").read_text())["embeddings"] == "tfidf"
