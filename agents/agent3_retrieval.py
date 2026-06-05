"""
Agent 3 — Stage-1 Hybrid Retrieval Engine (Ollama-powered)

Combines two complementary retrieval strategies:
  • Dense: Ollama embeddings (via mistral-embed or similar)
  • Sparse: BM25Okapi (exact token / keyword matching)

Both streams are merged via Reciprocal Rank Fusion (RRF) to maximize recall.

Embeddings stored in Qdrant Cloud for persistence.
"""

from typing import List, Tuple
import numpy as np
import requests
import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi

from models.schemas import ChunkWithContext, JobRequirements
from config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    EMBEDDING_DIM,
    DENSE_TOP_K,
    SPARSE_TOP_K,
    HYBRID_TOP_K,
)

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "mxbai-embed-large"  # Powerful embedding model for Ollama (1024 dimensions)

# Module-level singletons
_qdrant_client: QdrantClient | None = None
_bm25_index: BM25Okapi | None = None
_indexed_chunks: List[ChunkWithContext] = []
_indexed_texts: List[str] = []


# ---------------------------------------------------------------------------
# Ollama API calls
# ---------------------------------------------------------------------------

def _get_ollama_embedding(text: str) -> List[float]:
    """Get embedding from Ollama via API."""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={
                "model": EMBEDDING_MODEL,
                "input": text,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

        # Ollama returns embeddings in 'embeddings' field
        if "embeddings" in data:
            return data["embeddings"][0] if isinstance(data["embeddings"][0], list) else data["embeddings"]
        else:
            raise ValueError(f"Unexpected Ollama response: {data}")
    except Exception as e:
        print(f"  ERROR: Ollama connection failed: {e}")
        print(f"  Make sure Ollama is running: ollama serve")
        raise


def _get_qdrant() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        print(f"  Connecting to Qdrant Cloud...")
        _qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
    return _qdrant_client


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def index_chunks(chunks: List[ChunkWithContext]) -> None:
    """
    Builds Qdrant index with Ollama embeddings and BM25 sparse index.
    Each chunk is enriched with macro-profile before encoding.
    """
    global _bm25_index, _indexed_chunks, _indexed_texts

    _indexed_chunks = chunks
    _indexed_texts = [_enrich_chunk_text(c) for c in chunks]

    client = _get_qdrant()

    # Get embeddings from Ollama
    print(f"  Generating embeddings with Ollama ({EMBEDDING_MODEL})...")
    embeddings = []
    for i, text in enumerate(_indexed_texts):
        if (i + 1) % 5 == 0:
            print(f"    Embedded {i + 1}/{len(_indexed_texts)} chunks...")
        emb = _get_ollama_embedding(text)
        embeddings.append(emb)

    print(f"  Retrieved {len(embeddings)} embeddings from Ollama")

    # Detect embedding dimension from first embedding
    embedding_dim = len(embeddings[0])
    print(f"  Embedding dimension: {embedding_dim}")

    # Delete existing collection if present
    try:
        client.delete_collection(collection_name=QDRANT_COLLECTION)
        print(f"  Deleted existing collection: {QDRANT_COLLECTION}")
    except Exception:
        pass

    # Create Qdrant collection
    print(f"  Creating Qdrant collection: {QDRANT_COLLECTION}")
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
    )

    # Upload vectors to Qdrant Cloud
    points = [
        PointStruct(id=i, vector=emb, payload={"idx": i})
        for i, emb in enumerate(embeddings)
    ]
    print(f"  Uploading {len(points)} vectors to Qdrant Cloud...")
    client.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)

    # Build BM25 sparse index
    tokenized = [t.lower().split() for t in _indexed_texts]
    _bm25_index = BM25Okapi(tokenized)
    print(f"  Built BM25 index for sparse retrieval\n")


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def hybrid_retrieve(
    job_req: JobRequirements,
    top_k: int = HYBRID_TOP_K,
) -> List[Tuple[ChunkWithContext, float]]:
    """
    Runs dense (Ollama) + sparse (BM25) retrieval and fuses via RRF.
    Returns up to top_k (chunk, rrf_score) pairs, highest score first.
    """
    query_text = _build_query_text(job_req)

    # --- Dense retrieval (Ollama) ---
    query_vec = _get_ollama_embedding(query_text)
    client = _get_qdrant()

    # Use REST API for search
    import requests
    search_url = f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/search"
    search_payload = {
        "vector": query_vec,
        "limit": DENSE_TOP_K,
        "with_payload": True
    }
    headers = {
        "api-key": QDRANT_API_KEY,
        "Content-Type": "application/json"
    }

    response = requests.post(search_url, json=search_payload, headers=headers, timeout=30)
    response.raise_for_status()
    search_result = response.json()

    dense_ids = [hit["payload"]["idx"] for hit in search_result.get("result", [])]

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
    """Prepends candidate macro-profile to chunk."""
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
