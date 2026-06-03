"""
Agent 4 — Stage-2 Cross-Encoder Reranker

Replaces the bi-encoder's approximate dot-product similarity with full
cross-attention over [query, chunk] pairs. This eliminates the false
positives that survive RRF and surfaces the most relevant passages
for the final grading stage.

Model: BAAI/bge-reranker-base (runs locally, no API needed)
"""

from typing import List, Tuple
from sentence_transformers import CrossEncoder

from models.schemas import ChunkWithContext, JobRequirements
from config import RERANKER_MODEL, RERANK_TOP_N

_reranker: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        print(f"  Loading reranker model: {RERANKER_MODEL}")
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank_chunks(
    query: str,
    candidates: List[Tuple[ChunkWithContext, float]],
    top_n: int = RERANK_TOP_N,
) -> List[Tuple[ChunkWithContext, float]]:
    """
    Scores every (query, chunk_text) pair with the cross-encoder.
    Returns top_n chunks sorted by cross-attention relevance score (descending).
    """
    reranker = _get_reranker()

    pairs = [(query, chunk.text) for chunk, _ in candidates]
    scores: List[float] = reranker.predict(pairs).tolist()

    ranked = sorted(
        zip([c for c, _ in candidates], scores),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[:top_n]


def build_rerank_query(job_req: JobRequirements) -> str:
    """Compact, structured query string for the cross-encoder."""
    return (
        f"Role: {job_req.job_title}\n"
        f"Domain: {job_req.domain}\n"
        f"Required skills: {', '.join(job_req.required_skills)}\n"
        f"Preferred skills: {', '.join(job_req.preferred_skills)}\n"
        f"Experience: {job_req.min_years_experience}+ years\n"
        f"Education: {job_req.required_education}"
    )
