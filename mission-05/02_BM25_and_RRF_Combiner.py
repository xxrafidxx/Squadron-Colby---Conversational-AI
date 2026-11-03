"""
Mission 05 — Part 2: Communications & Protocol Officer
Okapi BM25 lexical retrieval + Reciprocal Rank Fusion (RRF) with the HNSW dense path

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-05-M01 (Crab Pulsar Remnant / Messier 1)
Subsystem 05: Dual-Path Associative Memory Core

  BM25(D, Q) = sum_i IDF(q_i) * f(q_i, D) (k1 + 1) / ( f(q_i, D) + k1 (1 - b + b |D| / avgdl) ),  k1 = 1.5, b = 0.75
  IDF(q)     = ln( 1 + (N - n(q) + 0.5) / (n(q) + 0.5) )          (always >= 0)
  RRF(d)     = sum_{m in {dense, sparse}} 1 / (k + r_m(d)),  k = 60, ranks r start at 1

Corpus: Document C (USSE Noosphere Technical Archive Extract, 36 records).
Dense path: WordLlama sentence embeddings searched with the HNSW index from Part 1.
Benchmark: 30 labelled questions (exact codes, paraphrased concepts, mixed) scored with Hit@1, Hit@3, MRR@10.
Run:  python 02_BM25_and_RRF_Combiner.py
"""

from __future__ import annotations

import json
import math
import pathlib
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import aura_archive as A

HERE = pathlib.Path(__file__).resolve().parent


def _load_part1():
    return A.load_part("01_HNSW_Vector_Index.py")


# ---------------------------------------------------------------------------
# 1. BM25 scoring engine
# ---------------------------------------------------------------------------
class BM25Okapi:
    """Okapi BM25 with term-frequency saturation (k1) and document-length normalisation (b)."""

    def __init__(self, corpus: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [A.tokenize(d) for d in corpus]
        self.tf = [Counter(d) for d in self.docs]
        self.doc_len = np.array([len(d) for d in self.docs], dtype=float)
        self.N = len(self.docs)
        self.avgdl = float(self.doc_len.mean()) if self.N else 0.0
        df = Counter(t for d in self.docs for t in set(d))
        self.idf = {t: math.log(1.0 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def term_score(self, term: str, i: int) -> float:
        f = self.tf[i].get(term, 0)
        if f == 0:
            return 0.0
        norm = self.k1 * (1.0 - self.b + self.b * self.doc_len[i] / self.avgdl)
        return self.idf.get(term, 0.0) * f * (self.k1 + 1.0) / (f + norm)

    def scores(self, query: str) -> np.ndarray:
        q = A.tokenize(query)
        return np.array([sum(self.term_score(t, i) for t in q) for i in range(self.N)])

    def rank(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        s = self.scores(query)
        order = np.argsort(-s, kind="stable")
        return [(int(i), float(s[i])) for i in order[:top_k] if s[i] > 0]


# ---------------------------------------------------------------------------
# 2. Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(rankings: Dict[str, Sequence[int]], k: int = 60,
                           weights: Optional[Dict[str, float]] = None) -> List[Tuple[int, float]]:
    """
    rankings: {"dense": [doc ids best-first], "sparse": [...]}.  RRF(d) = sum_m w_m / (k + r_m(d)), r from 1,
    with all w_m = 1 by default (the plain RRF of Cormack et al., 2009).
    A document missing from one list simply gets no contribution from it.
    Ties are broken by the best single rank, then by doc id (deterministic).
    """
    score: Dict[int, float] = {}
    best: Dict[int, int] = {}
    for name, ranking in rankings.items():
        w = 1.0 if weights is None else weights.get(name, 1.0)
        for r, doc in enumerate(ranking, start=1):
            score[doc] = score.get(doc, 0.0) + w / (k + r)
            best[doc] = min(best.get(doc, r), r)
    return sorted(score.items(), key=lambda kv: (-kv[1], best[kv[0]], kv[0]))


_IDENTIFIER = re.compile(r"\b(?=[A-Za-z0-9-]*\d)(?=[A-Za-z0-9-]*[A-Za-z])[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b")


def has_identifier(query: str) -> bool:
    """True if the query contains a code-like token: letters and digits joined by hyphens (PCV-7731-X, I-503, DT-3)."""
    return bool(_IDENTIFIER.search(query))


def code_aware_weights(query: str) -> Dict[str, float]:
    """
    Extension (not in the brief): weighted RRF. Weights fixed a priori from the two failure modes in
    Document C (C-20, C-21): a code in the query -> trust the exact-token path twice as much;
    no code -> the keyword path only half counts, because paraphrases share few rare words.
    """
    return {"dense": 1.0, "sparse": 2.0} if has_identifier(query) else {"dense": 1.0, "sparse": 0.5}


# ---------------------------------------------------------------------------
# Dense path + hybrid retriever
# ---------------------------------------------------------------------------
class DenseRetriever:
    def __init__(self, texts: Sequence[str], embedder: Optional[A.Embedder] = None):
        self.embedder = embedder or A.get_embedder()
        H = _load_part1()
        self.vectors = self.embedder.embed(list(texts))
        self.index = H.HNSWIndex(self.vectors.shape[1], M=8, ef_construction=32, seed=1).build(self.vectors)

    def rank(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        qv = self.embedder.embed([query])[0]
        return self.index.knn_search(qv, k=top_k, ef=max(32, top_k))


class HybridRetriever:
    """Dual-path retrieval for Subsystem 05: BM25 (exact tokens) + dense HNSW (meaning), fused with RRF."""

    def __init__(self, chunks: Optional[List[A.Chunk]] = None, rrf_k: int = 60, depth: int = 20):
        self.chunks = chunks or A.load_archive()
        texts = [c.full_text for c in self.chunks]
        self.bm25 = BM25Okapi(texts)
        self.dense = DenseRetriever(texts)
        self.rrf_k, self.depth = rrf_k, depth

    def search(self, query: str, top_k: int = 5, rrf_k: Optional[int] = None, code_aware: bool = False) -> List[Dict]:
        sparse = self.bm25.rank(query, self.depth)
        dense = self.dense.rank(query, self.depth)
        fused = reciprocal_rank_fusion({"dense": [i for i, _ in dense], "sparse": [i for i, _ in sparse]},
                                       k=self.rrf_k if rrf_k is None else rrf_k,
                                       weights=code_aware_weights(query) if code_aware else None)
        d_rank = {i: r for r, (i, _) in enumerate(dense, 1)}
        s_rank = {i: r for r, (i, _) in enumerate(sparse, 1)}
        d_sc, s_sc = dict(dense), dict(sparse)
        return [{"id": self.chunks[i].id, "index": i, "rrf": sc,
                 "dense_rank": d_rank.get(i), "dense_cosine": d_sc.get(i),
                 "sparse_rank": s_rank.get(i), "bm25": s_sc.get(i)} for i, sc in fused[:top_k]]

    def ranked_ids(self, query: str, method: str, top_k: int = 10, rrf_k: Optional[int] = None) -> List[str]:
        if method == "bm25":
            return [self.chunks[i].id for i, _ in self.bm25.rank(query, top_k)]
        if method == "dense":
            return [self.chunks[i].id for i, _ in self.dense.rank(query, top_k)]
        return [h["id"] for h in self.search(query, top_k, rrf_k=rrf_k, code_aware=(method == "rrf_code"))]


# ---------------------------------------------------------------------------
# 3. Evaluation benchmark (labels written before any run)
# ---------------------------------------------------------------------------
BENCHMARK: List[Dict] = [
    # exact codes: the identifier is the only thing that pins down the right record
    {"type": "exact code", "q": "Which locker holds the spare PCV-7731-X?", "rel": ["C-08"]},
    {"type": "exact code", "q": "What is the service interval of PCV-7713-X?", "rel": ["C-09"]},
    {"type": "exact code", "q": "Steps of procedure SR-11", "rel": ["C-05"]},
    {"type": "exact code", "q": "ALPHA-9", "rel": ["C-07"]},
    {"type": "exact code", "q": "What does fix NAV-ORI-0451 record?", "rel": ["C-13"]},
    {"type": "exact code", "q": "KST3-FINAL", "rel": ["C-32"]},
    {"type": "exact code", "q": "Field strength of MAG-SHLD-04B", "rel": ["C-04"]},
    {"type": "exact code", "q": "What is TW-TILE-220 used for?", "rel": ["C-10"]},
    {"type": "exact code", "q": "When is LRS-9-DM replaced?", "rel": ["C-15"]},
    {"type": "exact code", "q": "Incident I-503", "rel": ["C-28"]},
    {"type": "exact code", "q": "What is stored in record C-30?", "rel": ["C-30"]},
    {"type": "exact code", "q": "Where is locker E3-17?", "rel": ["C-08"]},
    # paraphrased concepts: few or no rare words shared with the record
    {"type": "concept", "q": "How do we keep people healthy when the neutron star sprays charged radiation at us?", "rel": ["C-04", "C-34"]},
    {"type": "concept", "q": "Why did the ship's AI start making up travel records?", "rel": ["C-03", "C-26", "C-29"]},
    {"type": "concept", "q": "What happens if the reactor's magnetic bottle suddenly fails?", "rel": ["C-07"]},
    {"type": "concept", "q": "How much breathable air do we have if the generators break down?", "rel": ["C-16"]},
    {"type": "concept", "q": "How does the ship work out where it is using spinning dead stars?", "rel": ["C-14"]},
    {"type": "concept", "q": "Can the small transport craft fly while the star is erupting?", "rel": ["C-33"]},
    {"type": "concept", "q": "What should the assistant say when it does not know the answer?", "rel": ["C-24"]},
    {"type": "concept", "q": "Should the AI obey commands hidden inside the documents it reads?", "rel": ["C-25"]},
    {"type": "concept", "q": "How long can we stay here before we run out of propellant?", "rel": ["C-11"]},
    {"type": "concept", "q": "How bad is radio contact with the robot scouts near the star?", "rel": ["C-31"]},
    {"type": "concept", "q": "What happened to the robotic explorer that flew into the nebula clouds?", "rel": ["C-32"]},
    {"type": "concept", "q": "Who is allowed to read the assistant's chat history?", "rel": ["C-35"]},
    # mixed: a code plus a conceptual question about it
    {"type": "mixed", "q": "Can PCV-7713-X replace the main drive's field generator?", "rel": ["C-09", "C-08"]},
    {"type": "mixed", "q": "Why was the course correction citing NAV-ORI-0415 rejected?", "rel": ["C-29", "C-13"]},
    {"type": "mixed", "q": "What is the dose limit the MAG-SHLD-04B protects the crew against?", "rel": ["C-04"]},
    {"type": "mixed", "q": "Which tank gets used first because DT-3 is leaking fuel?", "rel": ["C-11", "C-28"]},
    {"type": "mixed", "q": "How long does SR-11 take and when must it not run?", "rel": ["C-05"]},
    {"type": "mixed", "q": "Where does ALPHA-9 send the leftover plasma?", "rel": ["C-07"]},
]


# Held-out questions: written after the first benchmark run, BEFORE the code-aware variant was
# implemented, and never used to choose its weights. Used only to check that the variant generalises.
HELD_OUT: List[Dict] = [
    {"type": "exact code", "q": "What is the rated output of PCV-7731-X?", "rel": ["C-08"]},
    {"type": "exact code", "q": "Where are the PCV-7713-X spares kept?", "rel": ["C-09"]},
    {"type": "exact code", "q": "Incident I-502", "rel": ["C-27"]},
    {"type": "exact code", "q": "What is QT-2?", "rel": ["C-07"]},
    {"type": "exact code", "q": "DT-1 temperature", "rel": ["C-11"]},
    {"type": "concept", "q": "What do we do if someone gets too much radiation during a shift?", "rel": ["C-34"]},
    {"type": "concept", "q": "Why can't we trust a single reading from the long-distance sensors near the star?", "rel": ["C-27", "C-02"]},
    {"type": "concept", "q": "How many people live aboard the ship?", "rel": ["C-17", "C-16"]},
    {"type": "concept", "q": "How is drinking water recycled?", "rel": ["C-16"]},
    {"type": "concept", "q": "Which star system did the expedition visit before this one?", "rel": ["C-12", "C-13"]},
    {"type": "concept", "q": "What protects the reactor walls from the heat?", "rel": ["C-06", "C-10"]},
    {"type": "mixed", "q": "Is it safe to put PCV-7713-X into the main chamber?", "rel": ["C-09", "C-08"]},
    {"type": "mixed", "q": "How often are TW-TILE-220 tiles checked and what damage do inspectors look for?", "rel": ["C-10"]},
]


def first_relevant_rank(ranked: Sequence[str], rel: Sequence[str]) -> Optional[int]:
    for r, cid in enumerate(ranked, 1):
        if cid in rel:
            return r
    return None


METHODS = ("bm25", "dense", "rrf", "rrf_code")
LABELS = {"bm25": "BM25", "dense": "dense (HNSW)", "rrf": "RRF k=60", "rrf_code": "code-aware RRF"}


def evaluate(retriever: HybridRetriever, methods=METHODS, rrf_k: Optional[int] = None,
             depth: int = 10, questions: Optional[List[Dict]] = None) -> Tuple[List[Dict], Dict]:
    per_query = []
    for item in (BENCHMARK if questions is None else questions):
        row = {"type": item["type"], "q": item["q"], "rel": item["rel"]}
        for m in methods:
            ranked = retriever.ranked_ids(item["q"], m, depth, rrf_k=rrf_k)
            row[m] = first_relevant_rank(ranked, item["rel"])
            row[m + "_top1"] = ranked[0] if ranked else None
        per_query.append(row)
    summary: Dict = {}
    for t in ("exact code", "concept", "mixed", "all"):
        rows = [r for r in per_query if t == "all" or r["type"] == t]
        if not rows:
            continue
        summary[t] = {}
        for m in methods:
            ranks = [r[m] for r in rows]
            summary[t][m] = {
                "n": len(rows),
                "hit@1": float(np.mean([1.0 if rk == 1 else 0.0 for rk in ranks])),
                "hit@3": float(np.mean([1.0 if rk and rk <= 3 else 0.0 for rk in ranks])),
                "mrr@10": float(np.mean([1.0 / rk if rk else 0.0 for rk in ranks])),
            }
    return per_query, summary


def run(output_dir: Optional[pathlib.Path] = None) -> Dict:
    print("=" * 78)
    print("USSE NOOSPHERE • SUBSYSTEM 05 · PART 2: BM25 + RECIPROCAL RANK FUSION")
    print("=" * 78)
    hr = HybridRetriever()
    print(f"Corpus: {len(hr.chunks)} records of Document C · avgdl = {hr.bm25.avgdl:.1f} tokens · k1 = 1.5, b = 0.75")
    print(f"Dense path: {hr.dense.embedder.name}, HNSW (M = 8) · RRF k = 60, depth 20 per path")

    # Demonstration: the near-duplicate part numbers
    q = "Which locker holds the spare PCV-7731-X?"
    print(f"\n[demo] '{q}'")
    print(f"   BM25 top-3 : {[(hr.chunks[i].id, round(s, 2)) for i, s in hr.bm25.rank(q, 3)]}")
    print(f"   dense top-3: {[(hr.chunks[i].id, round(s, 3)) for i, s in hr.dense.rank(q, 3)]}")
    print(f"   RRF top-3  : {[(h['id'], round(h['rrf'], 5)) for h in hr.search(q, 3)]}")

    per_query, summary = evaluate(hr)
    _print_table("Benchmark (30 questions, labels fixed before the first run)", summary)

    print("\nPer query: rank of the first relevant record (– = not in top 10)")
    print(f"{'type':<11}{'BM25':>5}{'dense':>6}{'RRF':>5}{'code':>5}  question")
    fmt = lambda r: "–" if r is None else str(r)
    for r in per_query:
        print(f"{r['type']:<11}{fmt(r['bm25']):>5}{fmt(r['dense']):>6}{fmt(r['rrf']):>5}{fmt(r['rrf_code']):>5}  {r['q']}")

    sweep = {}
    print("\nPlain RRF smoothing constant k (MRR@10 over the 30 questions):")
    for k in (1, 10, 60, 200):
        _, s = evaluate(hr, methods=("rrf",), rrf_k=k)
        sweep[k] = s["all"]["rrf"]["mrr@10"]
        print(f"   k = {k:>3}: MRR {sweep[k]:.3f}")

    ho_rows, ho_summary = evaluate(hr, questions=HELD_OUT)
    _print_table(f"Held-out check ({len(HELD_OUT)} new questions, never used for any choice)", ho_summary)

    wins = {m: sum(1 for r in per_query if r[m] == 1) for m in METHODS}
    print(f"\n[summary] Hit@1 over 30 questions: " + ", ".join(f"{LABELS[m]} {wins[m]}" for m in METHODS) + ".")
    result = {"corpus_records": len(hr.chunks), "embedder": hr.dense.embedder.name, "k1": 1.5, "b": 0.75, "rrf_k": 60,
              "summary": summary, "per_query": per_query, "rrf_k_sweep": sweep,
              "code_aware_weights": {"code in query": {"dense": 1.0, "sparse": 2.0}, "no code": {"dense": 1.0, "sparse": 0.5}},
              "held_out_summary": ho_summary, "held_out_per_query": ho_rows}
    if output_dir:
        _save(result, output_dir)
    return result


def _print_table(title: str, summary: Dict) -> None:
    print(f"\n{title}")
    print(f"{'query type':<12}{'n':>3} " + "".join(f"{LABELS[m]:>20}" for m in METHODS))
    print(f"{'':<16}" + "".join(f"{'Hit@1 Hit@3  MRR':>20}" for _ in METHODS))
    print("-" * (16 + 20 * len(METHODS)))
    for t in ("exact code", "concept", "mixed", "all"):
        if t not in summary:
            continue
        line = f"{t:<12}{summary[t]['bm25']['n']:>3} "
        for m in METHODS:
            v = summary[t][m]
            line += f"{v['hit@1']:>9.2f}{v['hit@3']:>6.2f}{v['mrr@10']:>5.2f}"
        print(line)


def _save(result: Dict, output_dir: pathlib.Path) -> None:
    (output_dir / "results").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "results" / "02_hybrid_retrieval.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    s = result["summary"]
    types = ["exact code", "concept", "mixed", "all"]
    methods = [("bm25", "BM25 (sparse)", "#ff7f0e"), ("dense", "Dense (WordLlama + HNSW)", "#1f77b4"),
               ("rrf", "RRF hybrid, k = 60", "#2ca02c"), ("rrf_code", "code-aware RRF (extension)", "#9467bd")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for ax, metric, title in ((axes[0], "hit@1", "Hit@1: right record ranked first"), (axes[1], "mrr@10", "Mean reciprocal rank (MRR@10)")):
        x = np.arange(len(types))
        for j, (m, label, col) in enumerate(methods):
            vals = [s[t][m][metric] for t in types]
            off = (j - 1.5) * 0.2
            ax.bar(x + off, vals, 0.2, label=label, color=col)
            for xi, v in zip(x + off, vals):
                ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=6.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t}\n(n = {s[t]['bm25']['n']})" for t in types])
        ax.set_ylim(0, 1.12); ax.set_title(title + " · 30 benchmark questions", fontweight="bold", fontsize=10); ax.grid(axis="y", alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.5, frameon=False)
    plt.tight_layout(rect=(0, 0.07, 1, 1))
    plt.savefig(output_dir / "figures" / "02_hybrid_retrieval.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    run(output_dir=HERE)
    print("=" * 78)
