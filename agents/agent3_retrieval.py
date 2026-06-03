"""
Agent 3 — Stage-1 Hybrid Retrieval Engine

Combines two complementary retrieval strategies:
  • Dense: Qdrant + BAAI/bge-small-en-v1.5 embeddings (semantic similarity)
  • Sparse: BM25Okapi (exact token / keyword matching)

Both streams are merged via Reciprocal Rank Fusion (RRF) to maximize recall —
semantically equivalent phrases AND exact framework names are both captured.
"""

from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi

from models.schemas import ChunkWithContext, JobRequirements
from config import (
    EMBEDDING_MODEL,
    QDRANT_COLLECTION,
    EMBEDDING_DIM,
    DENSE_TOP_K,
    SPARSE_TOP_K,
    HYBRID_TOP_K,
)

# Module-level singletons — loaded once per process
_embedding_model: SentenceTransformer | None = None
_qdrant_client: QdrantClient | None = None
_bm25_index: BM25Okapi | None = None
_indexed_chunks: List[ChunkWithContext] = []
_indexed_texts: List[str] = []


# ---------------------------------------------------------------------------
# Model / client accessors
# ---------------------------------------------------------------------------

def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        print(f"  Loading embedding model: {EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def _get_qdrant() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(":memory:")
        _qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
    return _qdrant_client


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def index_chunks(chunks: List[ChunkWithContext]) -> None:
    """
    Builds both the dense Qdrant index and the sparse BM25 index.
    Each chunk's text is enriched with its global macro-profile before
    encoding so the embedding captures the candidate's full context.
    """
    global _bm25_index, _indexed_chunks, _indexed_texts

    _indexed_chunks = chunks
    _indexed_texts = [_enrich_chunk_text(c) for c in chunks]

    model = _get_embedding_model()
    client = _get_qdrant()

    # Dense index
    print(f"  Encoding {len(chunks)} chunks with {EMBEDDING_MODEL}...")
    embeddings = model.encode(
        _indexed_texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    points = [
        PointStruct(id=i, vector=emb.tolist(), payload={"idx": i})
        for i, emb in enumerate(embeddings)
    ]
    client.upsert(collection_name=QDRANT_COLLECTION, points=points)

    # Sparse BM25 index
    tokenized = [t.lower().split() for t in _indexed_texts]
    _bm25_index = BM25Okapi(tokenized)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def hybrid_retrieve(
    job_req: JobRequirements,
    top_k: int = HYBRID_TOP_K,
) -> List[Tuple[ChunkWithContext, float]]:
    """
    Runs dense + sparse retrieval in parallel and fuses results via RRF.
    Returns up to top_k (chunk, rrf_score) pairs, highest score first.
    """
    query_text = _build_query_text(job_req)

    # --- Dense retrieval ---
    model = _get_embedding_model()
    query_vec = model.encode([query_text], normalize_embeddings=True)[0]
    client = _get_qdrant()

    dense_hits = client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vec.tolist(),
        limit=DENSE_TOP_K,
    )
    dense_ids = [hit.payload["idx"] for hit in dense_hits]

    # --- Sparse BM25 retrieval ---
    bm25_scores = _bm25_index.get_scores(query_text.lower().split())
    sparse_ids = np.argsort(bm25_scores)[::-1][:SPARSE_TOP_K].tolist()

    # --- Reciprocal Rank Fusion ---
    rrf_ranked = _reciprocal_rank_fusion([dense_ids, sparse_ids])

    return [(_indexed_chunks[idx], score) for idx, score in rrf_ranked[:top_k]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reciprocal_rank_fusion(
    ranked_lists: List[List[int]],
    k: int = 60,
) -> List[Tuple[int, float]]:
    """Standard RRF: score(d) = Σ 1 / (k + rank(d))"""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rank + k)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _enrich_chunk_text(chunk: ChunkWithContext) -> str:
    """
    Prepends the candidate's global macro-profile to the chunk text.
    This prevents context fragmentation: every embedding 'knows' who the
    candidate is and what their overall profile looks like.
    """
    p = chunk.global_profile
    header = f"[SECTION: {chunk.section_header}]\n" if chunk.section_header else ""
    return (
        f"[CANDIDATE: {p.candidate_name} | "
        f"Experience: {p.total_years_experience} yrs | "
        f"Stack: {', '.join(p.primary_tech_stack)} | "
        f"Education: {p.highest_education}]\n"
        f"[PROFILE: {p.summary}]\n"
        f"{header}"
        f"{chunk.text}"
    )


def _build_query_text(job_req: JobRequirements) -> str:
    skills = " ".join(job_req.required_skills + job_req.preferred_skills)
    return (
        f"{job_req.job_title} {job_req.domain} {skills} "
        f"{job_req.min_years_experience} years experience "
        f"{job_req.required_education}"
    )
