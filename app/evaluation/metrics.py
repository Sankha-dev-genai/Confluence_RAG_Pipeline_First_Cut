from __future__ import annotations

from typing import Iterable, Sequence


def _dedup_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def precision_at_k(
    retrieved: Sequence[str],
    relevant: set[str],
    k: int,
) -> float:
    """Fraction of the top-k retrieved items that are relevant."""
    if k <= 0:
        return 0.0
    topk = retrieved[:k]
    if not topk:
        return 0.0
    hits = sum(1 for r in topk if r in relevant)
    return hits / min(k, len(topk))


def recall_at_k(
    retrieved: Sequence[str],
    relevant: set[str],
    k: int,
) -> float:
    """Fraction of all relevant items that appear in the top-k."""
    if not relevant:
        return 0.0
    topk = set(retrieved[:k])
    return len(topk & relevant) / len(relevant)


def hit_at_k(
    retrieved: Sequence[str],
    relevant: set[str],
    k: int,
) -> float:
    """1.0 if at least one relevant item is in the top-k, else 0.0."""
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def reciprocal_rank(
    retrieved: Sequence[str],
    relevant: set[str],
) -> float:
    """1 / rank of the first relevant item (0 if none found)."""
    for i, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def average_precision(
    retrieved: Sequence[str],
    relevant: set[str],
) -> float:
    """Average precision over the ranks where relevant items occur."""
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for i, r in enumerate(retrieved, start=1):
        if r in relevant:
            hits += 1
            score += hits / i
    return score / len(relevant)


def evaluate_single(
    retrieved: Sequence[str],
    relevant: set[str],
    k_values: Sequence[int] = (1, 3, 5),
) -> dict[str, float]:
    """All metrics for one query. `retrieved` should be de-duplicated ids."""
    retrieved = _dedup_keep_order(retrieved)
    out: dict[str, float] = {}
    for k in k_values:
        out[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        out[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        out[f"hit@{k}"] = hit_at_k(retrieved, relevant, k)
    out["mrr"] = reciprocal_rank(retrieved, relevant)
    out["ap"] = average_precision(retrieved, relevant)
    return out


def context_sufficiency(retrieved_texts, keywords) -> float:
    """
    Groundedness proxy: fraction of the expected answer keywords that appear
    somewhere in the retrieved context. High value => retrieval surfaced the
    facts needed to answer (a precondition for a faithful, cited answer).
    """
    if not keywords:
        return 0.0
    blob = " \n ".join(retrieved_texts).lower()
    hits = sum(1 for k in keywords if str(k).lower() in blob)
    return hits / len(keywords)
