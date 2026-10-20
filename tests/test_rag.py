"""Week 9: chunking, retrieval, refusal, and the three RAG measurements."""

from __future__ import annotations

import pandas as pd
import pytest

from financial_risk.rag.evaluation import (
    acceptable_sources,
    eval_score,
    groundedness_report,
    recall_at_k,
    reciprocal_rank,
    retrieval_report,
    retrieval_summary,
    run_eval_set,
    sentence_support,
)
from financial_risk.rag.index import (
    answer_question,
    build_index,
    chunk_text,
    clean_markdown,
    is_out_of_policy,
    retrieve,
    split_sections,
)

DOC = """# Fraud Review Policy

High-risk model outputs are reviewed by an analyst before any action is taken.
The model is a prioritisation aid, not an automated decision system.

## Excluded fields

The `is_flagged_fraud` field is excluded from model features because it encodes an existing
business rule, and `amount_to_origin_balance` measures how much of the balance a transfer moved.

```python
# this code block must not be indexed
secretvalue = 42
```

| column | meaning |
|---|---|
| amount | transaction size |

See [the playbook](https://example.com/playbook) for more.
"""


@pytest.fixture
def index(tmp_path):
    docs = tmp_path / "kb"
    docs.mkdir()
    (docs / "fraud_policy.md").write_text(DOC, encoding="utf-8")
    (docs / "faq.md").write_text(
        "# FAQ\n\nQ: Why PR-AUC?\nA: Fraud is rare, so accuracy is misleading and PR-AUC "
        "focuses on ranking the positive class.\n",
        encoding="utf-8",
    )
    index_dir = tmp_path / "index"
    build_index([docs], index_dir, seed_dir=None)
    return index_dir


# ------------------------------------------------------------------ cleaning & chunking


def test_clean_markdown_strips_noise_but_keeps_identifiers():
    clean = clean_markdown(DOC)
    assert "```" not in clean and "secretvalue" not in clean       # code fences dropped
    assert "|" not in clean                                        # table pipes dropped
    assert "https://" not in clean and "the playbook" in clean     # link text kept, URL gone
    # the regression this cleaner was fixed for: underscores are identifiers, not italics
    assert "is_flagged_fraud" in clean
    assert "amount_to_origin_balance" in clean


def test_split_sections_keeps_headings():
    sections = dict(split_sections(clean_markdown(DOC)))
    assert "Fraud Review Policy" in sections
    assert "Excluded fields" in sections
    assert "analyst" in sections["Fraud Review Policy"]


def test_chunks_carry_source_and_section():
    chunks = chunk_text(DOC, "fraud_policy.md", max_words=40, overlap=10)
    assert chunks, "expected at least one chunk"
    assert all(c.source == "fraud_policy.md" for c in chunks)
    assert any(c.section == "Excluded fields" for c in chunks)
    assert chunks[0].citation().startswith("fraud_policy.md § ")
    assert all(c.n_words > 0 for c in chunks)


def test_overlap_repeats_text_and_must_be_smaller_than_the_window():
    long_doc = "# T\n\n" + " ".join(f"word{i}" for i in range(200))
    no_overlap = chunk_text(long_doc, "t.md", max_words=50, overlap=0)
    overlapping = chunk_text(long_doc, "t.md", max_words=50, overlap=25)
    assert len(overlapping) > len(no_overlap)          # overlap costs index size
    with pytest.raises(ValueError):
        chunk_text(long_doc, "t.md", max_words=50, overlap=50)


def test_eval_artifacts_are_never_indexed(tmp_path):
    docs = tmp_path / "kb"
    docs.mkdir()
    (docs / "faq.md").write_text("# FAQ\n\nThe project uses PaySim transactions.\n", encoding="utf-8")
    for leaky in ("rag_eval_results.md", "rag_retrieval_report.md", "agent_routing_report.md"):
        (docs / leaky).write_text("# Results\n\nanswer key leaking into the corpus\n", encoding="utf-8")
    _, n = build_index([docs], tmp_path / "ix", seed_dir=None)
    sources = {c["source"] for c in retrieve("answer key", tmp_path / "ix", top_k=5)}
    assert sources.isdisjoint({"rag_eval_results.md", "rag_retrieval_report.md", "agent_routing_report.md"})
    assert n >= 1


# ------------------------------------------------------------------------- retrieval


def test_retrieve_ranks_by_score_and_carries_citations(index):
    hits = retrieve("Why is is_flagged_fraud excluded?", index, top_k=3)
    assert hits, "expected a hit for an exact identifier match"
    assert hits[0]["source"] == "fraud_policy.md"
    assert hits[0]["citation"].startswith("fraud_policy.md")
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_recall_and_reciprocal_rank():
    sources = ["a.md", "b.md", "c.md"]
    assert recall_at_k(sources, "b.md", 3) and not recall_at_k(sources, "b.md", 1)
    assert reciprocal_rank(sources, "a.md") == 1.0
    assert reciprocal_rank(sources, "c.md") == pytest.approx(1 / 3)
    assert reciprocal_rank(sources, "zz.md") == 0.0


def test_acceptable_sources_allows_several_right_answers():
    assert acceptable_sources("a.md|b.md") == {"a.md", "b.md"}
    assert recall_at_k(["b.md"], "a.md|b.md", 1)


def test_retrieval_report_skips_unlabelled_rows(index):
    questions = pd.DataFrame({
        "question": ["Why is is_flagged_fraud excluded?", "Which stocks should I buy?"],
        "expected_behavior": ["Answer.", "Refuse: out of policy."],
        "expected_source": ["fraud_policy.md", ""],
    })
    report = retrieval_report(questions, index, ks=(1, 3), top_k=3)
    assert len(report) == 1                      # the refusal row has no right document
    assert bool(report.iloc[0]["hit@1"])
    summary = retrieval_summary(report, ks=(1, 3))
    assert list(summary["k"]) == [1, 3]
    assert 0.0 <= float(summary.iloc[0]["recall"]) <= 1.0


# ---------------------------------------------------------------------- groundedness


def test_sentence_support_separates_grounded_from_invented():
    context = "The model is a prioritisation aid reviewed by an analyst."
    assert sentence_support("The model is a prioritisation aid.", context) == 1.0
    assert sentence_support("It achieves 94% accuracy on live production traffic.", context) < 0.4


def test_groundedness_report_flags_the_invented_sentence():
    chunks = [{"text": "The model is a prioritisation aid reviewed by an analyst."}]
    answer = "The model is a prioritisation aid. It achieves 94% accuracy on live production traffic."
    report = groundedness_report(answer, chunks)
    assert len(report) == 2
    assert bool(report.iloc[0]["grounded"])
    assert not bool(report.iloc[1]["grounded"])


# --------------------------------------------------------------------------- refusal


def test_policy_refusal_happens_before_retrieval(index):
    result = answer_question("Which stocks should I buy with my savings?", index)
    assert result["refused"] and result["refusal_reason"] == "out_of_policy"
    assert result["citations"] == []          # retrieval never ran


def test_low_similarity_refusal(index):
    result = answer_question("What is the capital of France?", index)
    assert result["refused"] and result["refusal_reason"] == "low_similarity"


def test_answers_carry_citations_and_a_disclaimer(index):
    result = answer_question("Why is is_flagged_fraud excluded?", index)
    assert not result["refused"]
    assert result["citations"] and all("source" in c and "section" in c for c in result["citations"])
    assert "not financial" in result["disclaimer"]
    assert result["answer_method"] == "stitched"   # offline default is grounded by construction


def test_extra_patterns_close_the_third_person_gap(index):
    question = "Should this customer receive a loan?"
    assert not is_out_of_policy(question)                     # the gap week 9 finds
    fix = (r"\bshould (this|that|the) \w+\b",)
    assert is_out_of_policy(question, fix)
    # and the fix must not start refusing legitimate questions
    assert not is_out_of_policy("Which features does the fraud model use?", fix)
    assert answer_question(question, index, extra_policy_patterns=fix)["refused"]


def test_eval_set_scores_behaviour_and_records_why(index):
    """A binary pass/fail can hide a bug: a lending question that gets refused only because
    nothing relevant was retrieved LOOKS correct, but the compliance boundary never ran. That
    is why run_eval_set reports the refusal reason as well as the verdict."""
    questions = pd.DataFrame({
        "question": ["Why is is_flagged_fraud excluded?", "Which stocks should I buy?", "Should this customer receive a loan?"],
        "expected_behavior": ["Answer.", "Refuse: out of policy.", "Refuse: out of policy."],
        "expected_source": ["fraud_policy.md", "", ""],
    })
    before = run_eval_set(questions, index)
    assert eval_score(before) == 1.0                       # every row "passes"...
    reasons = dict(zip(before["question"], before["reason"]))
    assert reasons["Which stocks should I buy?"] == "out_of_policy"
    # ...but the third-person lending question only refused because retrieval found nothing:
    assert reasons["Should this customer receive a loan?"] == "low_similarity"

    after = run_eval_set(questions, index, extra_policy_patterns=(r"\bshould (this|that|the) \w+\b",))
    assert eval_score(after) == 1.0
    assert dict(zip(after["question"], after["reason"]))["Should this customer receive a loan?"] == "out_of_policy"


# ------------------------------------------------------------------ the shared index format


def test_build_index_writes_the_shared_folder_layout(index):
    import json

    assert sorted(p.name for p in index.iterdir()) == ["chunks.jsonl", "manifest.json", "tfidf_matrix.joblib", "tfidf_vectorizer.joblib"]
    manifest = json.loads((index / "manifest.json").read_text())
    assert manifest["embeddings"] == "tfidf" and manifest["version"] == "baked"
    assert set(manifest["docs"]) == {"fraud_policy.md", "faq.md"} and all(len(d["sha256"]) == 64 for d in manifest["docs"].values())
    lines = [json.loads(l) for l in (index / "chunks.jsonl").read_text().splitlines()]
    assert manifest["chunks"] == len(lines) == sum(d["chunks"] for d in manifest["docs"].values())
    assert set(lines[0]) == {"id", "doc_path", "section", "text", "sha256"}


def test_the_embedding_method_belongs_to_the_knowledge_repo(tmp_path):
    with pytest.raises(ValueError, match="finrisk-knowledge"):
        build_index([tmp_path], tmp_path / "ix", method="embedding", seed_dir=None)


def test_a_loaded_index_gives_the_same_hits_as_the_folder(index):
    from financial_risk.rag.index import load_index_dir

    loaded = load_index_dir(index)
    by_folder = retrieve("Why use PR-AUC instead of accuracy?", index, top_k=2)
    assert by_folder == retrieve("Why use PR-AUC instead of accuracy?", loaded, top_k=2)
    assert loaded.version == "baked" and loaded.n_docs == 2
