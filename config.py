import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Gemini model names
FLASH_MODEL = "gemini-1.5-flash-latest"
PRO_MODEL = "gemini-1.5-pro-latest"

# Local embedding & reranker models (download automatically via HuggingFace)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"

# Retrieval settings
DENSE_TOP_K = 25       # Top-k chunks from Qdrant dense search
SPARSE_TOP_K = 25      # Top-k chunks from BM25 sparse search
HYBRID_TOP_K = 50      # Top-k after RRF fusion passed to reranker
RERANK_TOP_N = 15      # Top-n after cross-encoder reranking (covers 3-5 candidates)

# Qdrant in-memory settings
QDRANT_COLLECTION = "resume_chunks"
EMBEDDING_DIM = 384    # BAAI/bge-small-en-v1.5 output dimension

# Rate limiting — Gemini 1.5 Pro free tier: ~5 RPM
GEMINI_PRO_CALL_DELAY = 13  # seconds between Pro API calls
