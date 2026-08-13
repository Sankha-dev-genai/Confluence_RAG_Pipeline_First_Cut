"""
Hybrid candidate generation: fuse dense (FAISS cosine) + sparse (BM25).

Fusion = weighted min-max: score = alpha * norm(cosine) + (1-alpha) * norm(bm25),
kept on a [0,1] scale so the downstream metadata reranker's additive boosts stay
balanced. Raw cosine_score and bm25_score are preserved for transparency; the
fused value becomes the base `score` the reranker builds on.
"""
from __future__ import annotations

from typing import Any

# Duck-typed to avoid importing the dense stack here: `store` just needs
# .search(query, top_k) and `bm25` needs .search(query, top_k).


def _minmax(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    denom = hi - lo
    out = {}
    for k, v in values.items():
        if denom > 1e-9:
            out[k] = (v - lo) / denom
        else:
            out[k] = 1.0 if v > 0 else 0.0
    return out


class HybridSearcher:
    def __init__(self, store: Any, bm25: Any, alpha: float = 0.5) -> None:
        self.store = store          # dense weight
        self.bm25 = bm25
        self.alpha = alpha

    @staticmethod
    def _key(r: dict) -> str:
        return str(r.get("chunk_id") or r.get("vector_id"))

    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        candidate_k = max(top_k * 3, 30)
        dense = self.store.search(query=query, top_k=candidate_k)
        sparse = self.bm25.search(query=query, top_k=candidate_k)

        records: dict[str, dict] = {}
        cos: dict[str, float] = {}
        bm: dict[str, float] = {}

        for r in dense:
            k = self._key(r)
            records.setdefault(k, dict(r))
            cos[k] = float(r.get("score", 0.0))
        for r in sparse:
            k = self._key(r)
            records.setdefault(k, dict(r))
            bm[k] = float(r.get("bm25_score", 0.0))

        ncos = _minmax(cos)
        nbm = _minmax(bm)

        fused: list[dict[str, Any]] = []
        for k, rec in records.items():
            c = cos.get(k, 0.0)
            b = bm.get(k, 0.0)
            score = self.alpha * ncos.get(k, 0.0) + (1 - self.alpha) * nbm.get(k, 0.0)
            rec["cosine_score"] = c
            rec["bm25_score"] = b
            rec["hybrid_score"] = score
            rec["score"] = score          # base for the metadata reranker
            fused.append(rec)

        fused.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return fused[:candidate_k]
