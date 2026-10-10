"""Book loading, chunking and the exam-vs-training similarity filter (no heavy dependencies)."""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path


def load_book(path: Path, max_words: int) -> str:
    """Project Gutenberg text without licence header/footer, cut to the first `max_words` words at a paragraph end."""
    t = Path(path).read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n")
    a = re.search(r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", t, re.S)
    b = re.search(r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", t)
    t = t[a.end():b.start() if b else None] if a else t
    out, n = [], 0
    for para in re.split(r"\n\s*\n", t.strip()):
        para = " ".join(para.split())
        if not para:
            continue
        w = len(para.split())
        if n + w > max_words and n > 0:
            break
        out.append(para)
        n += w
    return "\n\n".join(out)


def chunk(text: str, target_words: int = 220) -> list[str]:
    chunks, cur, n = [], [], 0
    for para in text.split("\n\n"):
        w = len(para.split())
        if cur and n + w > target_words:
            chunks.append("\n\n".join(cur))
            cur, n = [], 0
        cur.append(para)
        n += w
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


_TOK = re.compile(r"[a-z0-9']+")


def tokens(s: str) -> list[str]:
    return _TOK.findall(s.lower())


def ngrams(s: str, n: int = 4) -> set[tuple[str, ...]]:
    t = tokens(s)
    return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}


def cosine(a: str, b: str) -> float:
    ca, cb = Counter(tokens(a)), Counter(tokens(b))
    dot = sum(ca[k] * cb.get(k, 0) for k in ca)
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    return dot / (na * nb) if na and nb else 0.0


def too_close(question: str, train_questions: list[str], cos_limit: float = 0.8, ngram_limit: float = 0.5) -> bool:
    """True when an exam question is (nearly) a training question: high bag-of-words cosine OR most of its
    4-grams already occur in one training row. Such exam items would measure memorised wording, not knowledge."""
    qg = ngrams(question)
    for t in train_questions:
        if cosine(question, t) >= cos_limit:
            return True
        if qg and len(qg & ngrams(t)) / len(qg) >= ngram_limit:
            return True
    return False
