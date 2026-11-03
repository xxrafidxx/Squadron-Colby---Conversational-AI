"""
Mission 05 — shared archive utilities (used by Parts 1–4 and the notebook)

* load_archive()      Document C (the USSE Noosphere Technical Archive Extract) as 36 records
* tokenize()          lower-case word tokens; codes such as PCV-7731-X stay one token
* split_sentences()   simple sentence splitter for the RAG triad
* Embedder            dense sentence embeddings for the dense retrieval path

Dense embeddings come from WordLlama (l2_supercat, 256 dimensions): static token
embeddings distilled from LLaMA-2 and trained for sentence similarity. The weights
ship inside the pip package, so it runs offline, on CPU, without torch. If WordLlama
is not installed, a TF-IDF + truncated SVD (latent semantic analysis) embedder fitted
on the archive is used instead and the scripts say so in their output.
"""

from __future__ import annotations

import math
import pathlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ARCHIVE_PATH = HERE / "corpus" / "USSE_Noosphere_Technical_Archive_Extract.md"


# ---------------------------------------------------------------------------
# Archive loading
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Chunk:
    id: str
    title: str
    text: str

    @property
    def full_text(self) -> str:
        return f"{self.title}. {self.text}"


_SECTION = re.compile(r"^### (C-\d+) · (.+)$", re.M)


def load_archive(path: Optional[pathlib.Path] = None) -> List[Chunk]:
    """Parses '### C-xx · Title' sections of Document C into retrieval chunks."""
    raw = pathlib.Path(path or ARCHIVE_PATH).read_text(encoding="utf-8")
    marks = list(_SECTION.finditer(raw))
    chunks = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(raw)
        body = " ".join(raw[m.end():end].split())
        chunks.append(Chunk(m.group(1), m.group(2).strip(), body))
    return chunks


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------
STOPWORDS = set("""
a an and are as at be been but by can could did do does for from had has have he her his how i if in into is it its
of on or our she should so than that the their them then there these they this to up was we were what when where which
who why will with would you your about after before any all also more most no not only other out over such very each
between during through under may must one two per
""".split())

_CODE = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)+")
_WORD = re.compile(r"[a-z0-9]+")


def _stem(tok: str) -> str:
    """Very light stemmer: plural -s only (keeps 'pulsars' ~ 'pulsar', leaves codes alone)."""
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith(("ss", "us", "is")) and not tok[-2].isdigit():
        return tok[:-1]
    return tok


def tokenize(text: str, keep_stopwords: bool = False) -> List[str]:
    """
    Lower-case tokens for BM25. A code such as 'PCV-7731-X' produces the exact token
    'pcv-7731-x' plus its parts ('pcv', '7731', 'x'), so an exact code query matches the
    exact code first, and a partial query ('7731') still finds it.
    """
    text = text.lower()
    out: List[str] = []
    for code in _CODE.findall(text):
        out.append(code)
    for w in _WORD.findall(text):
        if keep_stopwords or w not in STOPWORDS:
            out.append(_stem(w))
    return out


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT.split(text.strip()) if s.strip()]


_CRITICAL = re.compile(r"\b(?:[A-Za-z]+-[A-Za-z0-9-]*\d[A-Za-z0-9-]*|\d+(?:[.,]\d+)*)\b")


def critical_tokens(text: str) -> List[str]:
    """Numbers and codes (e.g. '3.2', '2,000', 'PCV-7731-X', 'NAV-ORI-0451') that must be quoted exactly."""
    return [t.lower().replace(",", "") for t in _CRITICAL.findall(text)]


# ---------------------------------------------------------------------------
# Dense embeddings
# ---------------------------------------------------------------------------
def _l2n(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


class Embedder:
    """Dense text embedder: WordLlama (default) or an LSA fallback fitted on the archive."""

    def __init__(self, backend: str = "auto", fit_texts: Optional[Iterable[str]] = None, lsa_dim: int = 64):
        self.backend = None
        if backend not in ("auto", "wordllama", "lsa"):
            raise ValueError(f"unknown embedder backend {backend!r}")
        if backend in ("auto", "wordllama"):
            try:
                import wordllama  # noqa: F401
                from wordllama import WordLlama
                pkg = pathlib.Path(wordllama.__file__).parent
                # weights + tokenizer are bundled in the package: no download needed
                self._wl = WordLlama.load(cache_dir=pkg, disable_download=True)
                self.backend = "wordllama"
                self.name = "WordLlama l2_supercat (256-d, pretrained, offline)"
            except Exception as exc:  # pragma: no cover - depends on the environment
                if backend == "wordllama":
                    raise
                self._why_fallback = repr(exc)
        if self.backend is None:
            self._fit_lsa(list(fit_texts) if fit_texts is not None else _default_fit_texts(), lsa_dim)
            self.backend = "lsa"
            self.name = f"LSA fallback (TF-IDF + SVD fitted on Document C, {self.dim}-d)"

    # -- LSA fallback -------------------------------------------------------
    def _fit_lsa(self, texts: List[str], k: int) -> None:
        docs = [tokenize(t) for t in texts]
        vocab = sorted({w for d in docs for w in d})
        self._vocab = {w: i for i, w in enumerate(vocab)}
        df = Counter(w for d in docs for w in set(d))
        n = len(docs)
        self._idf = np.array([math.log((1 + n) / (1 + df[w])) + 1.0 for w in vocab], dtype=np.float32)
        X = self._tfidf(docs)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        k = min(k, len(S))
        self._proj = (Vt[:k].T / np.maximum(S[:k], 1e-9)).astype(np.float32)
        self.dim = k

    def _tfidf(self, docs: List[List[str]]) -> np.ndarray:
        X = np.zeros((len(docs), len(self._vocab)), dtype=np.float32)
        for i, d in enumerate(docs):
            for w, c in Counter(d).items():
                j = self._vocab.get(w)
                if j is not None:
                    X[i, j] = (1.0 + math.log(c)) * self._idf[j]
        return _l2n(X)

    # -- public API -----------------------------------------------------------
    def embed(self, texts: List[str]) -> np.ndarray:
        """Returns unit-norm float32 vectors, shape (len(texts), dim)."""
        if self.backend == "wordllama":
            return _l2n(self._wl.embed(list(texts), norm=True))
        return _l2n(self._tfidf([tokenize(t) for t in texts]) @ self._proj)

    @property
    def dimension(self) -> int:
        return 256 if self.backend == "wordllama" else self.dim


def _default_fit_texts() -> List[str]:
    chunks = load_archive()
    return [s for c in chunks for s in split_sentences(c.full_text)]


_EMBEDDER: Optional[Embedder] = None


def get_embedder() -> Embedder:
    """Process-wide embedder (loading WordLlama once takes about a second)."""
    global _EMBEDDER
    if _EMBEDDER is None:
        import os
        _EMBEDDER = Embedder(os.environ.get("M05_EMBEDDER", "auto"))   # M05_EMBEDDER=lsa forces the fallback
    return _EMBEDDER


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _l2n(a) @ _l2n(b).T


def describe_environment() -> Dict[str, str]:
    import platform
    import sys
    return {"python": sys.version.split()[0], "numpy": np.__version__, "platform": platform.system()}


def load_part(filename: str):
    """Imports a mission script whose name starts with a digit (e.g. '01_HNSW_Vector_Index.py') once per process."""
    import importlib.util
    import sys
    name = "m05_" + pathlib.Path(filename).stem.lower()
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # registered before exec: dataclasses and pickling need it
    spec.loader.exec_module(mod)
    return mod
