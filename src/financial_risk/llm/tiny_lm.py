"""Toy language-model machinery for the week-8 class.

Nothing here is production code. It exists so a student can SEE the three ideas an LLM is
built from -- tokens, next-token prediction, attention -- in a few dozen lines of pure
Python/numpy, running on this project's own documentation as the corpus:

- ``bpe_train`` / ``bpe_encode``: byte-pair encoding, the tokeniser family behind GPT, Llama
  and Claude. Starts from characters and repeatedly merges the most frequent adjacent pair.
- ``train_ngram`` / ``next_token_distribution`` / ``sample_text``: a count-based n-gram
  language model. It is the smallest possible "predict the next token from the context"
  machine, and it makes temperature sampling and cross-entropy loss concrete.
- ``perplexity``: the held-out loss an LLM's pre-training minimises, exp(mean cross-entropy).
- ``attention``: scaled dot-product attention, softmax(QK^T / sqrt(d)) V, on tiny matrices.

A transformer replaces the count table with a neural network and the fixed n-token context
with attention over thousands of tokens -- but the *objective* is exactly what these toys
compute.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict

import numpy as np

# --------------------------------------------------------------------------------- tokens

_WORD_RE = re.compile(r"[a-z0-9_]+|[^\sa-z0-9_]")


def clean_prose(markdown: str, min_words: int = 6) -> str:
    """Keep only the prose sentences of a markdown file: drop headings, tables, code blocks,
    list markers and non-ASCII characters, and any line shorter than ``min_words`` words.
    A language model trained on prose produces prose; one trained on pipe tables produces
    pipe tables."""
    lines: list[str] = []
    in_code = False
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not stripped or stripped[0] in "#|>":
            continue
        stripped = re.sub(r"^[-*\d.]+\s+", "", stripped)  # list / numbered markers
        stripped = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", stripped)  # [text](url) -> text
        stripped = re.sub(r"https?://\S+", "", stripped)     # bare URLs
        stripped = re.sub(r"[`*_\[\]()]", "", stripped)     # inline markup
        stripped = re.sub(r"[^\x20-\x7e]", " ", stripped)   # non-ASCII (tables of CJK glyphs)
        if len(stripped.split()) >= min_words:
            lines.append(stripped)
    return "\n".join(lines)


def tokenize_words(text: str) -> list[str]:
    """Lower-case word/punctuation tokens -- the crude tokeniser the n-gram model uses."""
    return _WORD_RE.findall(text.lower())


def bpe_train(text: str, num_merges: int) -> list[tuple[str, str]]:
    """Learn ``num_merges`` byte-pair merges from ``text``.

    Each word starts as a sequence of characters plus an end-of-word marker ``</w>``; every
    merge step joins the most frequent adjacent pair across the corpus. The returned merge
    list, applied in order, is the whole tokeniser."""
    words = Counter(tokenize_words(text))
    vocab: dict[tuple[str, ...], int] = {tuple(w) + ("</w>",): c for w, c in words.items()}
    merges: list[tuple[str, str]] = []
    for _ in range(num_merges):
        pairs: Counter[tuple[str, str]] = Counter()
        for symbols, count in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += count
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = {_merge(symbols, best): c for symbols, c in vocab.items()}
    return merges


def _merge(symbols: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    out: list[str] = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            out.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


def bpe_encode(text: str, merges: list[tuple[str, str]]) -> list[str]:
    """Tokenise ``text`` with a learned merge list (applied in learned order, per word)."""
    tokens: list[str] = []
    for word in tokenize_words(text):
        symbols = tuple(word) + ("</w>",)
        for pair in merges:
            symbols = _merge(symbols, pair)
        tokens.extend(symbols)
    return tokens


# ---------------------------------------------------------------------------- n-gram LM

NGramModel = dict[tuple[str, ...], Counter]


def train_ngram(tokens: list[str], n: int = 3) -> NGramModel:
    """Count, for every (n-1)-token context, which token followed it and how often."""
    model: NGramModel = defaultdict(Counter)
    ctx = n - 1
    for i in range(ctx, len(tokens)):
        model[tuple(tokens[i - ctx : i])][tokens[i]] += 1
    return model


def next_token_distribution(
    model: NGramModel, context: tuple[str, ...], temperature: float = 1.0
) -> dict[str, float]:
    """P(next token | context) after temperature scaling.

    Temperature reshapes the distribution before sampling: below 1 sharpens it toward the
    most frequent continuation (0 = always pick the top one), above 1 flattens it toward
    uniform. Real LLMs apply the same transform to their logits."""
    counts = model.get(tuple(context))
    if not counts:
        return {}
    if temperature <= 0:
        top = counts.most_common(1)[0][0]
        return {top: 1.0}
    weights = {tok: c ** (1.0 / temperature) for tok, c in counts.items()}
    total = sum(weights.values())
    return {tok: w / total for tok, w in sorted(weights.items(), key=lambda kv: -kv[1])}


def sample_text(
    model: NGramModel,
    start: tuple[str, ...],
    length: int = 25,
    temperature: float = 1.0,
    seed: int = 0,
) -> str:
    """Generate ``length`` tokens after ``start`` by repeatedly sampling the next token."""
    rng = random.Random(seed)
    out = list(start)
    for _ in range(length):
        dist = next_token_distribution(model, tuple(out[-(len(start)) :]), temperature)
        if not dist:
            break
        toks, probs = zip(*dist.items())
        out.append(rng.choices(toks, weights=probs, k=1)[0])
    return " ".join(out)


def perplexity(model: NGramModel, tokens: list[str], n: int, vocab_size: int, k: float = 1.0) -> float:
    """exp(mean cross-entropy) of ``tokens`` under the model, with add-k smoothing.

    Cross-entropy -log P(actual next token) is the loss every LLM is pre-trained on; perplexity
    is just its exponential, readable as "the model was as unsure as a fair choice among
    THIS many tokens". Lower is better. Smoothing keeps unseen continuations from scoring
    -log(0) = infinity."""
    ctx = n - 1
    total = 0.0
    count = 0
    for i in range(ctx, len(tokens)):
        counts = model.get(tuple(tokens[i - ctx : i]), Counter())
        prob = (counts.get(tokens[i], 0) + k) / (sum(counts.values()) + k * vocab_size)
        total += -math.log(prob)
        count += 1
    return math.exp(total / max(count, 1))


# --------------------------------------------------------------------------- attention


def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scaled dot-product attention: weights = softmax(Q K^T / sqrt(d)); output = weights V.

    Returns ``(weights, output)``. Row i of ``weights`` says how much token i attends to every
    token (rows sum to 1); row i of ``output`` is token i's new representation, a weighted
    average of the values it attended to."""
    d = q.shape[-1]
    scores = q @ k.T / math.sqrt(d)
    scores = scores - scores.max(axis=1, keepdims=True)  # numerical stability
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return weights, weights @ v
