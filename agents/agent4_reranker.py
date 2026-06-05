"""
Agent 4 — Stage-2 Reranking with Ollama

Uses Ollama embeddings + cosine similarity to rerank chunks.
Scores each [query, chunk] pair and returns top-N highest-scoring candidates.

This eliminates false positives from the hybrid retrieval stage.
"""

from typing import List, Tuple
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from models.schemas import ChunkWithContext, JobRequirements
from config import RERANK_TOP_N

OLLAMA_BASE_URL = "http://localhost:11434"
RERANKER_MODEL = "mxbai-embed-large"  # Same as dense model for consistency


def _get_ollama_embedding(text: str) -> List[float]:
    """Get embedding from Ollama."""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={
                "model": RERANKER_MODEL,
                "input": text,
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

        if "embeddings" in data:
            return data["embeddings"][0] if isinstance(data["embeddings"][0], list) else data["embeddings"]
        else:
            raise ValueError(f"Unexpected response: {data}")
    except Exception as e:
        print(f"  ERROR: Ollama reranker failed: {e}")
        raise


def rerank_chunks(
    query: str,
    candidates: List[Tuple[ChunkWithContext, float]],
    top_n: int = RERANK_TOP_N,
) -> List[Tuple[ChunkWithContext, float]]:
    """
    Reranks candidate chunks using Ollama embeddings + cosine similarity.
    Returns top_n chunks sorted by relevance score (descending).
    """
    chunks = [chunk for chunk, _ in candidates]

    if not chunks:
        return []

    print(f"  Reranking {len(chunks)} chunks with Ollama...")

    # Get query embedding
    query_embedding = np.array(_get_ollama_embedding(query)).reshape(1, -1)

    # Get chunk embeddings
    chunk_embeddings = []
    for i, chunk in enumerate(chunks):
        if (i + 1) % 5 == 0:
            print(f"    Reranked {i + 1}/{len(chunks)} chunks...")
        emb = _get_ollama_embedding(chunk.text)
        chunk_embeddings.append(emb)

    chunk_embeddings = np.array(chunk_embeddings)

    # Compute cosine similarity
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]

    # Rank by similarity
    ranked_indices = np.argsort(similarities)[::-1][:top_n]
    reranked = [(chunks[i], float(similarities[i])) for i in ranked_indices]

    return reranked


def build_rerank_query(job_req: JobRequirements) -> str:
    """Build a structured query for reranking."""
    return (
        f"Role: {job_req.job_title}\n"
        f"Domain: {job_req.domain}\n"
        f"Required skills: {', '.join(job_req.required_skills)}\n"
        f"Preferred skills: {', '.join(job_req.preferred_skills)}\n"
        f"Experience: {job_req.min_years_experience}+ years\n"
        f"Education: {job_req.required_education}"
    )
