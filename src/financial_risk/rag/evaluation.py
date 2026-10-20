"""Measuring a RAG assistant: retrieval quality, groundedness, and behaviour.

A RAG system has three places it can fail, and they need three different measurements:

1. **Retrieval** -- did the right chunk come back at all? If it did not, nothing downstream
   can save the answer. Measured with ``recall_at_k`` and ``reciprocal_rank`` against a set of
   questions labelled with the document that should answer them.
2. **Groundedness** -- is every sentence of the answer actually supported by the retrieved
   text, or did the model add something? ``groundedness_report`` scores each sentence by how
   much of its content vocabulary appears in the context it was given.
3. **Behaviour** -- did it answer when it should answer, and refuse when it should refuse?
   ``run_eval_set`` replays a fixed question set and scores that, so the whole assistant has a
   regression test you rerun after every index, chunk-size, or policy change.

All three return DataFrames so they can be printed in class, written to ``reports/``, and
asserted on in CI.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from financial_risk.rag.index import answer_question, retrieve

# Words too common to count as evidence that a sentence came from the context.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "been", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "how", "i", "if", "in", "is", "it",
    "its", "not", "of", "on", "or", "should", "so", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "to", "was", "were", "what", "when", "which",
    "while", "who", "why", "will", "with", "would", "you", "your",
}


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9_]+", text.lower()) if w not in STOPWORDS and len(w) > 2}


# ------------------------------------------------------------------------ retrieval quality


def acceptable_sources(expected_source: str) -> set[str]:
    """Split a label into the set of documents that would count as correct.

    Relevance is not always single-source: two documents can legitimately answer the same
    question, and marking one of them "wrong" measures your labelling, not your retriever.
    Write those as ``datasource.md|faq.md``."""
    return {part.strip() for part in str(expected_source).split("|") if part.strip()}


def recall_at_k(retrieved_sources: list[str], expected_source: str, k: int) -> bool:
    """Did an acceptable document appear in the top ``k`` retrieved chunks?

    The single most important RAG metric: if recall@k is low, no prompt, model or reranker
    downstream can fix it -- the evidence never arrived."""
    return bool(acceptable_sources(expected_source) & set(retrieved_sources[:k]))


def reciprocal_rank(retrieved_sources: list[str], expected_source: str) -> float:
    """1/rank of the first correct document (1.0 = top hit, 0.0 = never retrieved).

    Averaged over questions this is MRR. It rewards putting the right chunk FIRST, which
    matters because the answer step reads the top chunks most closely."""
    wanted = acceptable_sources(expected_source)
    for position, source in enumerate(retrieved_sources, start=1):
        if source in wanted:
            return 1.0 / position
    return 0.0


def retrieval_report(
    questions: pd.DataFrame,
    index_dir: Path,
    ks: tuple[int, ...] = (1, 3, 5),
    top_k: int = 5,
) -> pd.DataFrame:
    """Per-question retrieval scores for every question that has an ``expected_source``.

    ``questions`` needs columns ``question`` and ``expected_source``; rows without an expected
    source (the refusal cases) are skipped -- there is no right document for them."""
    rows: list[dict[str, object]] = []
    labelled = questions[questions["expected_source"].notna() & (questions["expected_source"] != "")]
    for _, item in labelled.iterrows():
        hits = retrieve(str(item["question"]), index_dir, top_k=top_k)
        sources = [str(hit["source"]) for hit in hits]
        row: dict[str, object] = {
            "question": str(item["question"]),
            "expected": str(item["expected_source"]),
            "top_hit": sources[0] if sources else "-",
            "rr": round(reciprocal_rank(sources, str(item["expected_source"])), 3),
        }
        for k in ks:
            row[f"hit@{k}"] = recall_at_k(sources, str(item["expected_source"]), k)
        rows.append(row)
    return pd.DataFrame(rows)


def retrieval_summary(report: pd.DataFrame, ks: tuple[int, ...] = (1, 3, 5)) -> pd.DataFrame:
    """One row per k: recall@k over the labelled questions, plus MRR."""
    return pd.DataFrame(
        [{"k": k,
          "recall": round(float(report[f"hit@{k}"].mean()), 3),
          "found": int(report[f"hit@{k}"].sum()),
          "of": len(report),
          "mrr": round(float(report["rr"].mean()), 3)} for k in ks]
    )


# ---------------------------------------------------------------------------- groundedness


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def sentence_support(sentence: str, context: str) -> float:
    """Fraction of a sentence's content words that appear in the context it was given.

    A cheap, explainable proxy for "is this claim grounded". Real systems use an NLI model or
    an LLM judge; the logic being approximated is the same -- a sentence built from words the
    context never contained is a sentence the model invented."""
    words = _content_words(sentence)
    if not words:
        return 1.0
    return len(words & _content_words(context)) / len(words)


def groundedness_report(answer: str, chunks: list[dict[str, object]], floor: float = 0.6) -> pd.DataFrame:
    """Per-sentence support of an answer against the chunks that were retrieved for it.

    Sentences below ``floor`` are the ones to read out loud in class: either the model added
    something the documents do not say, or the retrieval did not supply what it needed."""
    context = " ".join(str(chunk["text"]) for chunk in chunks)
    rows = []
    for sentence in split_sentences(answer):
        score = sentence_support(sentence, context)
        rows.append({
            "sentence": sentence if len(sentence) <= 70 else sentence[:67] + "...",
            "support": round(score, 2),
            "grounded": score >= floor,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ behaviour eval set


def run_eval_set(
    questions: pd.DataFrame,
    index_dir: Path,
    extra_policy_patterns: tuple[str, ...] = (),
    **answer_kwargs: object,
) -> pd.DataFrame:
    """Replay a labelled question set and score answer-vs-refuse behaviour.

    ``questions`` needs ``question`` and ``expected_behavior``; a behaviour containing
    "refuse" means the assistant is supposed to decline. The returned frame carries the
    refusal reason too, because *why* it refused separates a policy boundary working from a
    retrieval failure hiding behind one."""
    rows = []
    for _, item in questions.iterrows():
        expected = str(item["expected_behavior"])
        should_refuse = "refuse" in expected.lower()
        result = answer_question(
            str(item["question"]), index_dir, extra_policy_patterns=extra_policy_patterns, **answer_kwargs
        )
        did_refuse = bool(result.get("refused"))
        rows.append({
            "question": str(item["question"])[:46],
            "expect": "refuse" if should_refuse else "answer",
            "got": "refuse" if did_refuse else "answer",
            "reason": str(result.get("refusal_reason") or ("%d cite" % len(result.get("citations", [])))),
            "pass": should_refuse == did_refuse,
        })
    return pd.DataFrame(rows)


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    """Minimal markdown table -- avoids a hard dependency on `tabulate` for one report."""
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def eval_score(report: pd.DataFrame) -> float:
    return float(report["pass"].mean())


def write_eval_report(report: pd.DataFrame, output_path: Path, score: float | None = None) -> Path:
    """Persist the behaviour report as the week's deliverable."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    if score is not None:
        summary = output_path.with_suffix(".md")
        lines = [
            "# RAG assistant evaluation",
            "",
            f"Behaviour score: **{score:.0%}** ({int(report['pass'].sum())}/{len(report)} questions)",
            "",
            _frame_to_markdown(report),
            "",
            "Rerun after every index, chunk-size, or policy change.",
        ]
        summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
