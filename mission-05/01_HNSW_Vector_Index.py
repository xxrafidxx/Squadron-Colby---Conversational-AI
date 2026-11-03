"""
Mission 05 — Part 1: Systems Architect (Lead Engineer)
Hierarchical Navigable Small World (HNSW) vector index in pure Python + NumPy

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-05-M01 (Crab Pulsar Remnant / Messier 1)
Subsystem 05: Dual-Path Associative Memory Core

Implements Malkov & Yashunin (2018), "Efficient and robust approximate nearest neighbor
search using Hierarchical Navigable Small World graphs":
  * cosine similarity core (vectors stored unit-normalised, so Sim = dot product)
  * layer assignment l = floor(-ln(U) * m_L) with m_L = 1 / ln(M)            (paper eq. / Alg. 1)
  * search_layer(entry_points, query, ef, layer)                            (Alg. 2)
  * neighbour selection heuristic                                           (Alg. 4)
  * k-NN search: greedy descent through upper layers, ef-search at layer 0  (Alg. 5)

Benchmark: N = 25,000 vectors of dimension 768 versus brute-force flat search.
Run:  python 01_HNSW_Vector_Index.py            (full benchmark, ~2-4 min on a laptop)
      python 01_HNSW_Vector_Index.py --n 5000   (quick run)
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import pathlib
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# 1. Cosine similarity core
# ---------------------------------------------------------------------------
def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Sim(u, v) = u·v / (||u||2 ||v||2); 0.0 if either vector is zero."""
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def cosine_similarity_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Vectorised cosine similarity of one query against every row of `matrix`."""
    q = query / max(float(np.linalg.norm(query)), 1e-12)
    norms = np.maximum(np.linalg.norm(matrix, axis=1), 1e-12)
    return (matrix @ q) / norms


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


# ---------------------------------------------------------------------------
# 2-3. HNSW index
# ---------------------------------------------------------------------------
class HNSWIndex:
    """
    Multi-layer proximity graph. Distance d(u, v) = 1 - cos(u, v) on unit vectors.

    M               max links per node on layers >= 1 (layer 0 allows M0 = 2M)
    ef_construction size of the dynamic candidate list while inserting
    m_L             1 / ln(M): level-generation normalisation
    """

    def __init__(self, dim: int, M: int = 16, ef_construction: int = 64, seed: int = 42,
                 capacity: int = 1024, use_heuristic: bool = True):
        self.dim, self.M, self.M0 = dim, M, 2 * M
        self.ef_construction = ef_construction
        self.m_L = 1.0 / math.log(M)
        self.rng = np.random.default_rng(seed)
        self.use_heuristic = use_heuristic
        self._data = np.zeros((capacity, dim), dtype=np.float32)
        self.count = 0
        self.levels: List[int] = []                    # level of each node
        self.links: List[Dict[int, List[int]]] = []    # links[layer][node] -> neighbour ids
        self.entry_point: Optional[int] = None
        self.max_level = -1
        self.distance_evals = 0                        # counter (reset by caller)

    # -- helpers ---------------------------------------------------------------
    @property
    def vectors(self) -> np.ndarray:
        return self._data[: self.count]

    def _dist(self, q: np.ndarray, ids: Sequence[int]) -> np.ndarray:
        self.distance_evals += len(ids)
        return 1.0 - self._data[list(ids)] @ q

    def random_level(self) -> int:
        """l = floor(-ln(uniform(0,1)) * m_L): P(l >= L) = M^-L, an exponentially decaying layer."""
        u = 1.0 - self.rng.random()          # in (0, 1], avoids ln(0)
        return int(math.floor(-math.log(u) * self.m_L))

    # -- Algorithm 2: SEARCH-LAYER ---------------------------------------------
    def search_layer(self, entry_points: Sequence[int], query: np.ndarray, ef: int, layer: int) -> List[Tuple[float, int]]:
        """
        Best-first search on one layer. Returns up to `ef` (distance, id) pairs sorted by distance.
        With ef = 1 this is the greedy hop used on the upper layers.
        """
        eps = list(entry_points)
        d0 = self._dist(query, eps)
        visited = set(eps)
        candidates = [(float(d), e) for d, e in zip(d0, eps)]            # min-heap on distance
        heapq.heapify(candidates)
        results = [(-float(d), e) for d, e in zip(d0, eps)]               # max-heap via negation
        heapq.heapify(results)
        while len(results) > ef:
            heapq.heappop(results)
        graph = self.links[layer]
        while candidates:
            d_c, c = heapq.heappop(candidates)
            if d_c > -results[0][0]:          # closest candidate is farther than worst result: stop
                break
            fresh = [n for n in graph.get(c, ()) if n not in visited]
            if not fresh:
                continue
            visited.update(fresh)
            dists = self._dist(query, fresh)
            worst = -results[0][0]
            for d_n, n in zip(dists.tolist(), fresh):
                if len(results) < ef or d_n < worst:
                    heapq.heappush(candidates, (d_n, n))
                    heapq.heappush(results, (-d_n, n))
                    if len(results) > ef:
                        heapq.heappop(results)
                    worst = -results[0][0]
        return sorted((-d, n) for d, n in results)

    # -- Algorithm 4: SELECT-NEIGHBORS-HEURISTIC --------------------------------
    def select_neighbors(self, base: np.ndarray, candidates: List[Tuple[float, int]], m: int) -> List[int]:
        """
        Keep a candidate only if it is closer to the base than to every neighbour already kept.
        This keeps links pointing in different directions (long-range 'highway' edges between
        clusters) instead of M links into the same dense cluster.
        """
        cands = sorted(candidates)
        if not self.use_heuristic or len(cands) <= m:
            return [n for _, n in cands[:m]]
        ids = [n for _, n in cands]
        V = self._data[ids]
        pair = (1.0 - V @ V.T).tolist()          # candidate-to-candidate distances, one matrix product
        self.distance_evals += len(ids) * (len(ids) - 1) // 2
        chosen: List[int] = []                   # positions in `ids`
        discarded: List[int] = []
        for i, (d_c, _) in enumerate(cands):
            if len(chosen) >= m:
                break
            row = pair[i]
            if any(row[j] < d_c for j in chosen):
                discarded.append(i)
            else:
                chosen.append(i)
        for i in discarded:                      # keepPrunedConnections: fill up to m
            if len(chosen) >= m:
                break
            chosen.append(i)
        return [ids[i] for i in chosen]

    # -- Algorithm 1: INSERT ----------------------------------------------------
    def insert(self, vector: np.ndarray) -> int:
        if self.count == self._data.shape[0]:
            self._data = np.vstack([self._data, np.zeros_like(self._data)])
        q = normalize_rows(vector.reshape(1, -1))[0]
        node = self.count
        self._data[node] = q
        self.count += 1
        level = self.random_level()
        self.levels.append(level)
        while len(self.links) <= level:
            self.links.append({})
        for l in range(level + 1):
            self.links[l][node] = []

        if self.entry_point is None:
            self.entry_point, self.max_level = node, level
            return node

        ep = [self.entry_point]
        for l in range(self.max_level, level, -1):                 # greedy descent above the new level
            ep = [self.search_layer(ep, q, ef=1, layer=l)[0][1]]
        for l in range(min(level, self.max_level), -1, -1):
            found = self.search_layer(ep, q, ef=self.ef_construction, layer=l)
            m_max = self.M0 if l == 0 else self.M
            neighbours = self.select_neighbors(q, found, self.M)
            self.links[l][node] = neighbours
            for n in neighbours:                                     # bidirectional links + pruning
                nl = self.links[l][n]
                nl.append(node)
                if len(nl) > m_max:
                    base = self._data[n]
                    d = self._dist(base, nl)
                    self.links[l][n] = self.select_neighbors(base, list(zip(d.tolist(), nl)), m_max)
            ep = [n for _, n in found]
        if level > self.max_level:
            self.entry_point, self.max_level = node, level
        return node

    def build(self, vectors: np.ndarray) -> "HNSWIndex":
        self._data = np.zeros((max(len(vectors), 1), self.dim), dtype=np.float32)
        for v in vectors:
            self.insert(v)
        return self

    # -- Algorithm 5: K-NN-SEARCH -----------------------------------------------
    def knn_search(self, query: np.ndarray, k: int = 10, ef: int = 64) -> List[Tuple[int, float]]:
        """Returns [(id, cosine similarity)] of the approximate k nearest neighbours."""
        if self.entry_point is None:
            return []
        q = normalize_rows(query.reshape(1, -1))[0]
        ep = [self.entry_point]
        for l in range(self.max_level, 0, -1):
            ep = [self.search_layer(ep, q, ef=1, layer=l)[0][1]]
        found = self.search_layer(ep, q, ef=max(ef, k), layer=0)
        return [(n, 1.0 - d) for d, n in found[:k]]

    def layer_histogram(self) -> List[int]:
        return [sum(1 for lv in self.levels if lv >= l) for l in range(self.max_level + 1)]


# ---------------------------------------------------------------------------
# 4. Brute-force baselines
# ---------------------------------------------------------------------------
def flat_search_numpy(query: np.ndarray, vectors: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
    """Exact search with one BLAS matrix-vector product (vectors must be unit-normalised)."""
    sims = vectors @ (query / max(float(np.linalg.norm(query)), 1e-12))
    top = np.argpartition(-sims, k)[:k]
    top = top[np.argsort(-sims[top])]
    return [(int(i), float(sims[i])) for i in top]


def flat_search_python(query: np.ndarray, vectors: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
    """Exact search that checks every vector one at a time (same per-distance cost model as HNSW)."""
    q = query / max(float(np.linalg.norm(query)), 1e-12)
    heap: List[Tuple[float, int]] = []
    for i in range(len(vectors)):
        s = float(np.dot(vectors[i], q))
        if len(heap) < k:
            heapq.heappush(heap, (s, i))
        elif s > heap[0][0]:
            heapq.heapreplace(heap, (s, i))
    return [(i, s) for s, i in sorted(heap, reverse=True)]


# ---------------------------------------------------------------------------
# Benchmark data
# ---------------------------------------------------------------------------
def make_embedding_like_vectors(n: int, dim: int = 768, n_clusters: int = 2000, latent_dim: int = 64,
                                seed: int = 7) -> np.ndarray:
    """
    Synthetic stand-in for text embeddings: points around many fine-grained topic clusters in a
    64-dimensional latent space, mapped into `dim` dimensions, plus small isotropic noise. Real
    sentence embeddings have this kind of structure (low intrinsic dimension, topics). Recall
    depends strongly on this choice: with 250 coarse clusters in 48 dimensions every ef reached
    recall 1.0 (too easy); purely isotropic vectors are the unrealistic worst case.
    """
    rng = np.random.default_rng(seed)
    basis = rng.standard_normal((latent_dim, dim)).astype(np.float32) / math.sqrt(latent_dim)
    centres = rng.standard_normal((n_clusters, latent_dim)).astype(np.float32) * 1.5
    assign = rng.integers(0, n_clusters, n)
    z = centres[assign] + rng.standard_normal((n, latent_dim)).astype(np.float32)
    x = z @ basis + 0.15 * rng.standard_normal((n, dim)).astype(np.float32)
    return normalize_rows(x)


def recall_at_k(approx: Sequence[Tuple[int, float]], exact: Sequence[Tuple[int, float]]) -> float:
    return len({i for i, _ in approx} & {i for i, _ in exact}) / max(len(exact), 1)


def _time_queries(fn, queries, repeat: int = 1) -> Tuple[float, list]:
    out = []
    t0 = time.perf_counter()
    for _ in range(repeat):
        out = [fn(q) for q in queries]
    return (time.perf_counter() - t0) / (repeat * len(queries)) * 1000.0, out


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------
def run_benchmark(n: int = 25_000, dim: int = 768, n_queries: int = 200, k: int = 10, M: int = 16,
                  ef_construction: int = 64, ef_values: Sequence[int] = (10, 20, 40, 80, 160, 320),
                  scaling_sizes: Optional[Sequence[int]] = None, output_dir: Optional[pathlib.Path] = None,
                  seed: int = 42) -> Dict:
    print("=" * 74)
    print("USSE NOOSPHERE • SUBSYSTEM 05 · PART 1: HNSW VECTOR PROXIMITY INDEX")
    print("=" * 74)
    data = make_embedding_like_vectors(n + n_queries, dim)
    base, queries = data[:n], data[n:]
    print(f"Dataset: N = {n:,} unit vectors, d = {dim}, {n_queries} held-out queries, k = {k}")
    print(f"Index:   M = {M} (layer 0: {2 * M}), ef_construction = {ef_construction}, m_L = 1/ln(M) = {1 / math.log(M):.4f}")

    idx = HNSWIndex(dim, M=M, ef_construction=ef_construction, seed=seed)
    t0 = time.perf_counter()
    idx.build(base)
    build_s = time.perf_counter() - t0
    hist = idx.layer_histogram()
    print(f"\n[build] {build_s:.1f} s ({build_s / n * 1000:.2f} ms per insert), top layer = {idx.max_level}")
    print("[layers] nodes on layer >= l:  measured vs expected N * M^-l")
    for l, cnt in enumerate(hist):
        print(f"   l = {l}:  {cnt:>6}   vs {n * M ** -l:>9.1f}")

    # exact ground truth with the NumPy baseline
    t_flat_np, exact = _time_queries(lambda q: flat_search_numpy(q, base, k), queries, repeat=3)
    t_flat_py, exact_py = _time_queries(lambda q: flat_search_python(q, base, k), queries[:50])
    assert all(recall_at_k(a, b) == 1.0 for a, b in zip(exact_py, exact[:50]))

    rows = []
    print(f"\n{'search':<22}{'recall@' + str(k):>10}{'latency ms':>12}{'distances':>12}{'% of N':>9}")
    print("-" * 65)
    print(f"{'flat, Python loop':<22}{1.0:>10.3f}{t_flat_py:>12.3f}{n:>12,}{100.0:>8.1f}%")
    print(f"{'flat, NumPy BLAS':<22}{1.0:>10.3f}{t_flat_np:>12.3f}{n:>12,}{100.0:>8.1f}%")
    for ef in ef_values:
        idx.distance_evals = 0
        lat, approx = _time_queries(lambda q: idx.knn_search(q, k=k, ef=ef), queries)
        dist = idx.distance_evals / len(queries)
        rec = float(np.mean([recall_at_k(a, e) for a, e in zip(approx, exact)]))
        rows.append({"ef": ef, "recall": rec, "latency_ms": lat, "distance_evals": dist})
        print(f"{'HNSW ef = ' + str(ef):<22}{rec:>10.3f}{lat:>12.3f}{dist:>12,.0f}{100 * dist / n:>8.1f}%")

    # scaling: how the work grows with N at a fixed ef
    scaling = []
    sizes = list(scaling_sizes) if scaling_sizes else [s for s in (1_000, 2_500, 5_000, 10_000) if s < n] + [n]
    ef_fixed = 40
    print(f"\n[scaling] fixed ef = {ef_fixed}: distance evaluations per query vs N")
    for s in sizes:
        if s == n:
            sub = idx
        else:
            sub = HNSWIndex(dim, M=M, ef_construction=ef_construction, seed=seed).build(base[:s])
        sub.distance_evals = 0
        lat, approx = _time_queries(lambda q: sub.knn_search(q, k=k, ef=ef_fixed), queries[:100])
        ex = [flat_search_numpy(q, base[:s], k) for q in queries[:100]]
        rec = float(np.mean([recall_at_k(a, e) for a, e in zip(approx, ex)]))
        scaling.append({"n": s, "distance_evals": sub.distance_evals / 100, "latency_ms": lat, "recall": rec})
        print(f"   N = {s:>6,}: {sub.distance_evals / 100:>7,.0f} distances ({100 * sub.distance_evals / 100 / s:5.1f}% of N), "
              f"{lat:6.3f} ms, recall@{k} {rec:.3f}")

    result = {
        "n": n, "dim": dim, "k": k, "M": M, "ef_construction": ef_construction, "m_L": 1 / math.log(M),
        "build_seconds": build_s, "layer_histogram": hist,
        "flat_numpy_ms": t_flat_np, "flat_python_ms": t_flat_py,
        "hnsw": rows, "scaling": scaling,
    }
    best = next((r for r in rows if r["recall"] >= 0.95), rows[-1])   # cheapest ef with recall >= 0.95
    print(f"\n[summary] at ef = {best['ef']}: recall@{k} {best['recall']:.3f}, {best['distance_evals']:,.0f} of {n:,} distances "
          f"({100 * best['distance_evals'] / n:.1f}%), {best['latency_ms']:.2f} ms/query in pure Python "
          f"vs {t_flat_py:.2f} ms (Python flat) and {t_flat_np:.2f} ms (NumPy flat).")
    if output_dir:
        _save(result, output_dir)
    return result


def archive_check(output_dir: Optional[pathlib.Path] = None) -> Dict:
    """Same index on real data: Document C records embedded with the dense embedder."""
    import aura_archive as A
    chunks = A.load_archive()
    emb = A.get_embedder()
    X = emb.embed([c.full_text for c in chunks])
    idx = HNSWIndex(X.shape[1], M=8, ef_construction=32, seed=1).build(X)
    queries = ["How is the crew protected from the pulsar's particle bursts?",
               "What went wrong with AURA's memory when the ship arrived?",
               "Which spare part fixes the main drive's confinement field?"]
    print(f"\n[archive] {len(chunks)} records of Document C embedded with {emb.name}")
    agree = []
    for q in queries:
        qv = emb.embed([q])[0]
        h = idx.knn_search(qv, k=3, ef=16)
        f = flat_search_numpy(qv, idx.vectors, 3)
        agree.append(recall_at_k(h, f))
        print(f"   '{q}'")
        print(f"      HNSW top-3: {[chunks[i].id for i, _ in h]}   exact top-3: {[chunks[i].id for i, _ in f]}")
    out = {"records": len(chunks), "embedder": emb.name, "recall_vs_exact": float(np.mean(agree))}
    if output_dir:
        p = output_dir / "results" / "01_hnsw_benchmark.json"
        data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        data["archive_check"] = out
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out


def _save(result: Dict, output_dir: pathlib.Path) -> None:
    (output_dir / "results").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "results" / "01_hnsw_benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows, sc, n = result["hnsw"], result["scaling"], result["n"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    ax.plot([r["latency_ms"] for r in rows], [r["recall"] for r in rows], "o-", color="#1f77b4", lw=2, label="HNSW (pure Python)")
    for r in rows:
        ax.annotate(f"ef={r['ef']}", (r["latency_ms"], r["recall"]), textcoords="offset points", xytext=(4, -10), fontsize=8)
    ax.axvline(result["flat_python_ms"], color="#d62728", ls="--", label=f"flat, Python loop ({result['flat_python_ms']:.1f} ms)")
    ax.axvline(result["flat_numpy_ms"], color="#7f7f7f", ls=":", label=f"flat, NumPy BLAS ({result['flat_numpy_ms']:.1f} ms)")
    ax.set_xscale("log"); ax.set_xlabel("latency per query (ms, log)"); ax.set_ylabel(f"recall@{result['k']}")
    ax.set_title(f"Recall vs latency, N = {n:,}, d = {result['dim']}", fontweight="bold"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax = axes[1]
    ns = [s["n"] for s in sc]
    ax.plot(ns, ns, "s--", color="#d62728", label="flat search: N distances")
    ax.plot(ns, [s["distance_evals"] for s in sc], "o-", color="#1f77b4", lw=2, label="HNSW, ef = 40")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("archive size N"); ax.set_ylabel("distance evaluations per query")
    ax.set_title("Work per query grows sub-linearly", fontweight="bold"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    ax = axes[2]
    hist = result["layer_histogram"]
    ax.bar(range(len(hist)), hist, color="#2ca02c", label="measured")
    ax.plot(range(len(hist)), [n * result["M"] ** -l for l in range(len(hist))], "k^--", label="expected N·M^-l")
    ax.set_yscale("log"); ax.set_xlabel("layer l"); ax.set_ylabel("nodes on layer ≥ l")
    ax.set_title("Exponential layer assignment", fontweight="bold"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "figures" / "01_hnsw_benchmark.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=25_000)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--queries", type=int, default=200)
    args = ap.parse_args()
    run_benchmark(n=args.n, dim=args.dim, n_queries=args.queries, output_dir=HERE)
    archive_check(output_dir=HERE)
    print("=" * 74)
