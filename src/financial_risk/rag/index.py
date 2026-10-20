"""The query side of the knowledge base (RAG): load an index, find the chunks, answer from them.

The INDEX is built by another repo (finrisk-knowledge) and downloaded by service/artifacts.py. This
module only reads it. Both sides agree on one folder layout, the "3.4 format":

    manifest.json   version, created, source_sha, embeddings ("titan" | "tfidf"), embedding_model,
                    dims, docs {path: {sha256, chunks}}, chunks (a count)
    chunks.jsonl    one chunk per line: id, doc_path, section, text, sha256
    embeddings.npy  (titan)  float32, row i = line i of chunks.jsonl
    tfidf_vectorizer.joblib + tfidf_matrix.joblib   (tfidf)

``build_index`` below writes the same layout from local markdown files: that is the small baked
index the image carries so the service also works on a laptop, in tests, and before the knowledge
pipeline has published anything.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity  # noqa: F401 - kept for callers that import it from here


logger = logging.getLogger("financial_risk.rag")

# An injected model call: (messages, purpose) -> (text, call_record). The service passes
# llm.client.complete; this module never imports a provider, so week 9 runs without one.
LLMCall = Callable[[list[dict[str, str]], str], tuple[str, dict[str, Any]]]

DISCLAIMER = "Educational demo only. This is not financial, lending, legal, or investment advice."
DEFAULT_EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"

# "Too far away to answer": below this similarity the assistant refuses instead of guessing. The
# scores of the two methods are not comparable (TF-IDF cosine is low for any real match, a dense
# embedding's is much higher), so each method has its own floor. RAG_MIN_SCORE overrides both.
MIN_SCORE_BY_METHOD = {"tfidf": 0.03, "titan": 0.25}

# Questions that ask the assistant to make or recommend a financial/legal decision are refused
# outright, regardless of retrieval similarity -- this is a compliance boundary, not a "we don't
# have documents about this" case. Week 11 homework asks students to design 3 more of these.
OUT_OF_POLICY_PATTERNS = [
    r"\bshould i\b",
    r"\brecommend(ed|ation)?\b",
    r"\bapprove\b.*\bloan\b",
    r"\bgood idea\b",
    r"\binvest\b",
    r"\bbuy (stock|shares|crypto)\b",
    r"\bsell (stock|shares|crypto)\b",
    r"\bguarantee(d)?\b",
    r"\blegal advice\b",
    r"\btax advice\b",
    r"\bwill (this|it) (definitely|guarantee)\b",
]


@dataclass(frozen=True)
class Chunk:
    """One retrievable piece of a document.

    ``section`` is the markdown heading the text sat under. Carrying it costs nothing at index
    time and makes a citation readable -- "eda_playbook.md § Information Value" tells a
    reviewer where to look; "eda_playbook.md" makes them read the whole file."""

    chunk_id: str
    source: str
    text: str
    section: str = ""
    n_words: int = 0

    def citation(self) -> str:
        return f"{self.source} § {self.section}" if self.section else self.source


SEED_DOCS = {
    "project_overview.md": """# Project Overview

The Financial Fraud Risk Intelligence Assistant is an educational end-to-end data project.
It uses PaySim synthetic financial transactions to demonstrate data engineering, machine learning,
LLM explanation, RAG, dashboards, and AWS lightweight deployment.
""",
    "fraud_policy.md": """# Fraud Review Policy

High-risk model outputs should be reviewed by an analyst. The model is a prioritisation aid, not
an automated decision system. The assistant must not provide financial, credit, legal, or investment
advice.
""",
    "model_card_stub.md": """# Model Card Notes

The fraud model uses transaction amount, transaction type, balance deltas, and time-safe customer
history features. The target is is_fraud. The is_flagged_fraud field is excluded from model features
because it represents an existing business rule.
""",
    "faq.md": """# FAQ

Q: Why use PR-AUC instead of accuracy?
A: Fraud is rare, so a model can achieve high accuracy by predicting non-fraud. PR-AUC focuses on
positive-class ranking performance.

Q: What is the AWS deployment?
A: The project can upload curated data to S3, query it with Athena, run Streamlit on EC2 or locally,
and call Bedrock for LLM summaries.
""",
}


def ensure_seed_docs(rag_docs_dir: Path) -> None:
    rag_docs_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in SEED_DOCS.items():
        path = rag_docs_dir / filename
        if not path.exists():
            path.write_text(content.strip() + "\n", encoding="utf-8")


def clean_markdown(text: str) -> str:
    """Strip what a retriever should never match on, keep what answers questions.

    Fenced code blocks, table pipes, link URLs and heading marks are noise in a similarity
    score: they inflate matches on punctuation and boilerplate while carrying little meaning.
    Cleaning is the cheapest retrieval improvement available, and it happens once at index
    time rather than on every query."""

    out: list[str] = []
    in_code = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if set(stripped) <= {"|", "-", ":", " "} and stripped:
            continue                                            # table rule rows
        stripped = stripped.replace("|", " ")                   # table cell separators
        stripped = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", stripped)   # [text](url) -> text
        stripped = re.sub(r"https?://\S+", "", stripped)
        # NB: `*` and backticks go, `_` STAYS -- in this corpus underscores are identifiers
        # (`is_flagged_fraud`, `amount_to_origin_balance`), not markdown italics, and stripping
        # them makes every feature name unsearchable.
        stripped = re.sub(r"[`*>]", "", stripped)
        out.append(stripped)
    return "\n".join(out)


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) pairs so chunks inherit a section name."""
    sections: list[tuple[str, str]] = []
    heading = ""
    buffer: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.*)$", line.strip())
        if match:
            if buffer:
                sections.append((heading, "\n".join(buffer)))
                buffer = []
            heading = match.group(2).strip()
        else:
            buffer.append(line)
    if buffer:
        sections.append((heading, "\n".join(buffer)))
    return [(h, b) for h, b in sections if b.strip()]


def chunk_text(text: str, source: str, max_words: int = 140, overlap: int = 30) -> list[Chunk]:
    """Split a document into overlapping, section-tagged chunks.

    Two dials the class measures rather than guesses:

    - ``max_words`` -- too large and a chunk carries mostly irrelevant text that dilutes its
      similarity score; too small and a sentence loses the context that made it meaningful.
    - ``overlap`` -- repeating the tail of one chunk at the head of the next stops an answer
      that straddles a boundary from being split beyond recovery. It costs index size.
    """
    if overlap >= max_words:
        raise ValueError(f"overlap ({overlap}) must be smaller than max_words ({max_words})")

    chunks: list[Chunk] = []
    for heading, body in split_sections(clean_markdown(text)):
        words = [w for w in re.split(r"\s+", body.strip()) if w]
        if not words:
            continue
        step = max_words - overlap
        for index in range(0, len(words), step):
            window = words[index : index + max_words]
            segment = " ".join(window).strip()
            if segment:
                chunks.append(Chunk(
                    chunk_id=f"{Path(source).stem}-{len(chunks):03d}",
                    source=source,
                    text=(f"{heading}. {segment}" if heading else segment),
                    section=heading,
                    n_words=len(window),
                ))
            if index + max_words >= len(words):
                break
    return chunks


# Files that must never enter the index. Every evaluation report contains the test questions
# themselves, so indexing one lets each test question retrieve its own answer key -- the RAG
# form of the leakage week 7 spent an hour hunting.
#
# The rule that matters: these reports are WRITTEN INTO a directory that is also INDEXED, so a
# second run silently swallows the first run's answer key. Any evaluation artefact you add
# belongs in this list on the same commit that creates it.
DEFAULT_EXCLUDE: tuple[str, ...] = ("rag_eval*", "rag_retrieval*", "agent_routing*", "*_eval_results*")


def load_chunks(
    rag_docs_dir: Path | list[Path],
    max_words: int = 140,
    overlap: int = 30,
    seed_dir: Path | None = None,
    exclude: tuple[str, ...] = DEFAULT_EXCLUDE,
) -> list[Chunk]:
    """Chunk every markdown file in one or more document directories.

    ``seed_dir`` is the only directory the starter documents may be written into. Passing a
    single directory seeds it (the original behaviour); passing a list seeds nothing unless
    ``seed_dir`` says so, so indexing ``docs/`` never creates files inside it."""
    single = isinstance(rag_docs_dir, (str, Path))
    dirs = [Path(rag_docs_dir)] if single else [Path(d) for d in rag_docs_dir]
    seed_target = Path(seed_dir) if seed_dir is not None else (dirs[0] if single else None)
    if seed_target is not None:
        ensure_seed_docs(seed_target)
    chunks: list[Chunk] = []
    for directory in dirs:
        for path in sorted(Path(directory).glob("*.md")):
            if any(path.match(pattern) for pattern in exclude):
                continue
            chunks.extend(chunk_text(path.read_text(encoding="utf-8"), path.name, max_words, overlap))
    return chunks


class RetrievalUnavailable(RuntimeError):
    """The index cannot be searched right now (for example the Bedrock embedding call failed).
    The API turns it into a 503: nothing is wrong with the question, retry later."""


def _doc_paths(directories: list[Path], exclude: tuple[str, ...]) -> list[Path]:
    paths: list[Path] = []
    for directory in directories:
        for path in sorted(Path(directory).glob("*.md")):
            if not any(path.match(pattern) for pattern in exclude):
                paths.append(path)
    return paths


def build_index(
    rag_docs_dir: Path | list[Path],
    index_dir: Path,
    method: str = "tfidf",
    max_words: int = 140,
    overlap: int = 30,
    seed_dir: Path | None = None,
    exclude: tuple[str, ...] = DEFAULT_EXCLUDE,
) -> tuple[Path, int]:
    """Build the small LOCAL index (TF-IDF) in the shared folder layout. The real knowledge base is
    built by the finrisk-knowledge repo; this one is the fallback baked into the image."""
    if method != "tfidf":
        raise ValueError(f"Unknown RAG index method: {method!r}. Only 'tfidf' is built here; "
                         "embedding indexes (titan) are built by the finrisk-knowledge repo.")

    chunks = load_chunks(rag_docs_dir, max_words=max_words, overlap=overlap, seed_dir=seed_dir, exclude=exclude)
    if not chunks:
        raise ValueError("No RAG documents found.")

    index_dir.mkdir(parents=True, exist_ok=True)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform([chunk.text for chunk in chunks])
    joblib.dump(vectorizer, index_dir / "tfidf_vectorizer.joblib")
    joblib.dump(matrix, index_dir / "tfidf_matrix.joblib")

    single = isinstance(rag_docs_dir, (str, Path))
    dirs = [Path(rag_docs_dir)] if single else [Path(d) for d in rag_docs_dir]
    per_doc: dict[str, dict[str, Any]] = {}
    for path in _doc_paths(dirs, exclude):
        per_doc[path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                              "chunks": sum(1 for c in chunks if c.source == path.name)}
    with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps({
                "id": chunk.chunk_id, "doc_path": chunk.source, "section": chunk.section, "text": chunk.text,
                "sha256": hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
            }) + "\n")
    (index_dir / "manifest.json").write_text(json.dumps({
        "version": "baked",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_sha": None,
        "embeddings": "tfidf",
        "embedding_model": None,
        "dims": None,
        "docs": per_doc,
        "chunks": len(chunks),
        "max_words": max_words,
        "overlap": overlap,
    }, indent=2), encoding="utf-8")
    return index_dir, len(chunks)


def knowledge_dirs(settings: object) -> list[Path]:
    """The directories that make up the LOCAL knowledge base -- defined once, used everywhere.

    The service image bakes a small index of these so it answers questions before any index has been
    published. The real knowledge base lives in the finrisk-knowledge repo (docs/ there)."""
    from financial_risk.config import project_root

    return [project_root() / "docs", settings.rag_docs_dir, settings.report_dir]


# ------------------------------------------------------------------------------ loading an index

Embedder = Callable[[str], "np.ndarray"]


class TitanEmbedder:
    """Embed a question the way the ingest side embedded the chunks: Bedrock InvokeModel with Titan
    Text Embeddings V2. The request body is the one ingest sends -- same ``dimensions``, ``normalize`` true --
    so a question vector and a chunk vector live in the same space and cosine similarity means something."""

    def __init__(self, model_id: str = DEFAULT_EMBEDDING_MODEL, dims: int | None = None,
                 client: Any | None = None, region: str | None = None) -> None:
        self.model_id, self.dims, self.region = model_id, dims, region
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            import boto3
            from botocore.config import Config

            self._client = boto3.client("bedrock-runtime", region_name=self.region, config=Config(
                connect_timeout=3, read_timeout=10, retries={"max_attempts": 2, "mode": "standard"}))
        return self._client

    def request_body(self, text: str) -> dict[str, Any]:
        body: dict[str, Any] = {"inputText": text, "normalize": True}
        if self.dims:
            body["dimensions"] = int(self.dims)
        return body

    def __call__(self, text: str) -> "np.ndarray":
        try:
            response = self.client.invoke_model(
                modelId=self.model_id, body=json.dumps(self.request_body(text)),
                contentType="application/json", accept="application/json")
            vector = json.loads(response["body"].read())["embedding"]
        except Exception as exc:  # noqa: BLE001 - any failure is "cannot search now", not a crash.
            raise RetrievalUnavailable(f"embedding the question failed: {type(exc).__name__}: {exc}"[:240]) from exc
        return np.asarray(vector, dtype=np.float32)


EmbedderFactory = Callable[[dict[str, Any]], Embedder]


def default_embedder_factory(region: str | None = None, default_model: str = DEFAULT_EMBEDDING_MODEL) -> EmbedderFactory:
    """manifest -> TitanEmbedder, using the model and dimensions the manifest names."""
    def make(manifest: dict[str, Any]) -> Embedder:
        return TitanEmbedder(manifest.get("embedding_model") or default_model, manifest.get("dims"), region=region)
    return make


@dataclass
class LoadedIndex:
    """One index, fully in memory. Immutable by convention: a new version is a NEW object, swapped in."""

    chunks: list[Chunk]
    manifest: dict[str, Any]
    method: str                                   # "tfidf" | "titan"
    vectorizer: Any = field(default=None, repr=False)
    matrix: Any = field(default=None, repr=False)
    embeddings: Any = field(default=None, repr=False)   # float32, rows L2-normalised
    embedder: Embedder | None = field(default=None, repr=False)

    @property
    def version(self) -> str:
        return str(self.manifest.get("version"))

    @property
    def n_docs(self) -> int:
        return len(self.manifest.get("docs") or {}) or len({c.source for c in self.chunks})

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    def scores(self, question: str) -> "np.ndarray":
        if self.method == "titan":
            query = np.asarray(self.embedder(question), dtype=np.float32)       # type: ignore[misc]
            norm = float(np.linalg.norm(query))
            if norm == 0.0:
                return np.zeros(len(self.chunks), dtype=np.float32)
            return self.embeddings @ (query / norm)                               # cosine: rows are unit length
        query = self.vectorizer.transform([question])
        return cosine_similarity(query, self.matrix).ravel()


def load_index_dir(index_dir: Path, embedder_factory: EmbedderFactory | None = None) -> LoadedIndex:
    """Read an index folder (see the module docstring). Raises ValueError/FileNotFoundError when it is
    not a complete, consistent index -- the caller reports that and keeps serving the old one."""
    index_dir = Path(index_dir)
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    method = str(manifest.get("embeddings") or "tfidf")
    if method not in {"tfidf", "titan"}:
        raise ValueError(f"manifest.json: unknown embeddings {method!r} (expected 'titan' or 'tfidf')")

    chunks: list[Chunk] = []
    for number, line in enumerate((index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            text = str(item["text"])
            chunks.append(Chunk(chunk_id=str(item["id"]), source=str(item["doc_path"]), text=text,
                                section=str(item.get("section") or ""), n_words=len(text.split())))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"chunks.jsonl line {number} is not a chunk ({type(exc).__name__}: {exc})") from exc
    if not chunks:
        raise ValueError("chunks.jsonl holds no chunks")
    if manifest.get("chunks") is not None and int(manifest["chunks"]) != len(chunks):
        raise ValueError(f"manifest says {manifest['chunks']} chunks, chunks.jsonl has {len(chunks)}")

    loaded = LoadedIndex(chunks=chunks, manifest=manifest, method=method)
    if method == "titan":
        embeddings = np.load(index_dir / "embeddings.npy").astype(np.float32, copy=False)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
            raise ValueError(f"embeddings.npy has shape {embeddings.shape}, expected ({len(chunks)}, dims)")
        if manifest.get("dims") and embeddings.shape[1] != int(manifest["dims"]):
            raise ValueError(f"embeddings.npy has {embeddings.shape[1]} dimensions, manifest says {manifest['dims']}")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        loaded.embeddings = embeddings / np.where(norms == 0, 1.0, norms)
        loaded.embedder = (embedder_factory or default_embedder_factory())(manifest)
    else:
        loaded.vectorizer = joblib.load(index_dir / "tfidf_vectorizer.joblib")
        loaded.matrix = joblib.load(index_dir / "tfidf_matrix.joblib")
        if loaded.matrix.shape[0] != len(chunks):
            raise ValueError(f"tfidf_matrix.joblib has {loaded.matrix.shape[0]} rows, chunks.jsonl has {len(chunks)}")
    return loaded


def index_method(index_dir: Path | LoadedIndex) -> str:
    if isinstance(index_dir, LoadedIndex):
        return index_dir.method
    manifest = Path(index_dir) / "manifest.json"
    if not manifest.exists():
        return "tfidf"
    return str(json.loads(manifest.read_text(encoding="utf-8")).get("embeddings") or "tfidf")


def retrieve(question: str, index: Path | LoadedIndex, top_k: int = 3) -> list[dict[str, object]]:
    """The top chunks for a question. ``index`` is a loaded index (the service) or a folder (scripts,
    notebooks, tests -- loaded on the spot)."""
    loaded = index if isinstance(index, LoadedIndex) else load_index_dir(Path(index))
    chunks = loaded.chunks
    scores = loaded.scores(question)
    ranked = scores.argsort()[::-1][:top_k]
    logger.info("retrieved %s", ", ".join(f"{chunks[i].source} {float(scores[i]):.3f}" for i in ranked if scores[i] > 0)
                or "nothing above zero")
    return [
        {
            "chunk_id": chunks[i].chunk_id,
            "source": chunks[i].source,
            "section": chunks[i].section,
            "citation": chunks[i].citation(),
            "text": chunks[i].text,
            "score": float(scores[i]),
        }
        for i in ranked
        if scores[i] > 0
    ]


def min_score_for(index: Path | LoadedIndex) -> float:
    import os

    override = os.environ.get("RAG_MIN_SCORE")
    if override:
        return float(override)
    return MIN_SCORE_BY_METHOD.get(index_method(index), 0.03)


def is_out_of_policy(question: str, extra_patterns: tuple[str, ...] = ()) -> bool:
    """Deterministic compliance gate, run BEFORE retrieval.

    ``extra_patterns`` lets the week-9 lab close a policy gap and re-score the assistant
    without editing this file; patterns that prove themselves belong in
    ``OUT_OF_POLICY_PATTERNS`` permanently."""
    lowered = question.lower()
    return any(re.search(pattern, lowered) for pattern in (*OUT_OF_POLICY_PATTERNS, *extra_patterns))


def stitch_answer(hits: list[dict[str, object]]) -> str:
    """The naive answer: paste the retrieved chunks together.

    Perfectly grounded -- it cannot say anything the documents do not -- and unreadable. It is
    the baseline the generated answer has to beat on readability without losing that
    grounding."""
    return " ".join(str(hit["text"]) for hit in hits)


_STOP = frozenset(
    "a an and are as at be by can do does for from how i in is it its me my of on or our so that the "
    "their them there this to was we what when where which who why will with you your".split()
)


def _terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9_]+", text.lower()) if len(w) > 2 and w not in _STOP}


def _is_prose(sentence: str) -> bool:
    """False for what cleaning leaves of a markdown table: long runs of numbers with no verb."""
    tokens = sentence.split()
    numeric = sum(bool(re.fullmatch(r"[-+]?\d[\d.,%]*", tok)) for tok in tokens)
    return len(tokens) <= 60 and numeric <= max(2, len(tokens) // 4)


def extract_answer(question: str, hits: list[dict[str, object]], max_sentences: int = 3) -> str:
    """The offline answer the web app uses: the few retrieved sentences that best match.

    Still grounded by construction -- every sentence is copied from a retrieved chunk -- but it
    answers the question instead of pasting three chunks end to end. Sentences are scored by how
    many of the question's content words they contain, lightly favouring the better-ranked
    chunk, then put back in document order so they read in sequence.

    Two things a FAQ or a report would otherwise get wrong: a matching QUESTION in the corpus
    ("Q: Why use PR-AUC?") is not an answer, so its relevance passes to the sentences after it;
    and a flattened table is not prose, so it is skipped."""
    wanted = _terms(question)
    candidates = []
    for rank, hit in enumerate(hits):
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(hit["text"])) if len(s.strip()) > 20]
        carry = 0.0
        for position, sentence in enumerate(sentences):
            body = re.sub(r"^(?:[QA]\s*:\s*)", "", sentence)
            overlap = len(wanted & _terms(body))
            if body.endswith("?"):                  # a question: hand its relevance to the answer
                carry = float(overlap)
                continue
            score = (overlap + carry) / math.sqrt(len(_terms(body)) + 1) + 0.15 / (rank + 1)
            carry *= 0.5                            # the answer is the next sentence or two
            if (overlap or carry) and _is_prose(body):
                candidates.append((score, rank, position, body, hit["citation"]))
    if not candidates:
        first = re.split(r"(?<=[.!?])\s+", str(hits[0]["text"]).strip())
        return " ".join(first[:2]) + f" [{hits[0]['citation']}]"
    chosen = sorted(sorted(candidates, reverse=True)[:max_sentences], key=lambda c: (c[1], c[2]))
    parts, cited = [], set()
    for index, (_, _, _, sentence, citation) in enumerate(chosen):
        last_of_source = index + 1 == len(chosen) or chosen[index + 1][4] != citation
        parts.append(f"{sentence} [{citation}]" if last_of_source and citation not in cited else sentence)
        if last_of_source:
            cited.add(citation)
    return " ".join(parts)


RAG_SYSTEM_PROMPT = (
    "You are the knowledge assistant of an educational fraud-risk data project. Answer ONLY from "
    "the context passages provided. Write 2-5 plain sentences. After each claim, cite the passage "
    "label you used in square brackets, exactly as given, e.g. [faq.md § FAQ]. If the passages do "
    "not contain the answer, say you cannot find it in the project documents. Never give financial, "
    "lending, legal, tax or investment advice."
)


def generate_answer(
    question: str,
    hits: list[dict[str, object]],
    use_bedrock: bool = False,
    llm: LLMCall | None = None,
    offline: str = "stitch",
) -> tuple[str, str]:
    """Ask an LLM to answer the question FROM the retrieved chunks only.

    Returns ``(answer, method)``. This is week 8's harness pointed at a different job: the
    context is retrieved rather than computed, the instruction is "answer only from this", and
    a model failure degrades to an offline answer rather than to an error -- the same fallback
    discipline, so the assistant always returns something grounded.

    ``offline`` picks that answer: ``stitch`` (week 9's baseline, the chunks pasted together) or
    ``extract`` (the best-matching sentences, what the week 10 web app uses)."""
    context = "\n\n".join(f"[{hit['citation']}] {hit['text']}" for hit in hits)

    def offline_answer(suffix: str = "") -> tuple[str, str]:
        if offline == "extract":
            return extract_answer(question, hits), "extracted" + suffix
        return stitch_answer(hits), "stitched" + suffix

    if llm is not None:
        messages = [{"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]
        try:
            text, _ = llm(messages, "rag_answer")
            if text.strip():
                return text.strip(), "generated"
        except Exception as exc:  # noqa: BLE001 - an LLM outage must not break the assistant.
            logger.warning("answer generation failed (%s); answering offline", exc)
        return offline_answer("_fallback")

    if use_bedrock:
        system = (
            "You answer questions about an educational financial-risk data project, using ONLY the "
            "context provided. If the context does not contain the answer, say so. Cite the sources "
            "you used in square brackets. Never give financial, lending, legal, tax, or investment advice."
        )
        try:
            from financial_risk.llm.bedrock_client import BedrockRiskClient

            raw = BedrockRiskClient().invoke(f"{system}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:")
            text = raw.strip()
            if text:
                return text, "generated"
        except Exception:  # noqa: BLE001 - an LLM outage must not break the assistant.
            pass
        return stitch_answer(hits), "stitched_fallback"

    return offline_answer()


def answer_question(
    question: str,
    index_dir: Path | LoadedIndex,
    min_score: float | None = None,
    top_k: int = 3,
    extra_policy_patterns: tuple[str, ...] = (),
    use_bedrock: bool = False,
    llm: LLMCall | None = None,
    offline: str = "stitch",
) -> dict[str, object]:
    if is_out_of_policy(question, extra_policy_patterns):
        logger.info("policy: out-of-policy pattern matched -> refuse before retrieval")
        return {
            "answer": (
                "I can explain how this project's data, model, and RAG index work, but I can't "
                "give financial, lending, legal, tax, or investment advice or decisions. Please "
                "consult a qualified professional or your institution's own policy for that."
            ),
            "citations": [],
            "disclaimer": DISCLAIMER,
            "refused": True,
            "refusal_reason": "out_of_policy",
        }

    hits = retrieve(question, index_dir, top_k=top_k)
    if min_score is None:
        min_score = min_score_for(index_dir)
    usable = [hit for hit in hits if float(hit["score"]) >= min_score]
    if not usable:
        logger.info("no chunk scored above %.2f -> refuse (low similarity)", min_score)
        return {
            "answer": "I cannot answer that from the project knowledge base. Please ask about the dataset, model, features, AWS deployment, or fraud-review policy.",
            "citations": [],
            "disclaimer": DISCLAIMER,
            "refused": True,
            "refusal_reason": "low_similarity",
        }

    answer, answer_method = generate_answer(question, usable, use_bedrock=use_bedrock, llm=llm, offline=offline)
    logger.info("answer: %s from %d chunk(s)", answer_method, len(usable))
    return {
        "answer": answer,
        "answer_method": answer_method,
        "citations": [
            {"source": hit["source"], "section": hit["section"], "chunk_id": hit["chunk_id"],
             "score": round(float(hit["score"]), 4)}
            for hit in usable
        ],
        "chunks": usable,
        "disclaimer": DISCLAIMER,
        "refused": False,
        "refusal_reason": None,
    }
