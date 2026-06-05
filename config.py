import os
from dotenv import load_dotenv

load_dotenv()

# OpenRouter API
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Qdrant Cloud
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

# OpenRouter model names
FLASH_MODEL = "openai/gpt-4o-mini"  # Fast, cost-effective
PRO_MODEL = "openai/gpt-4-turbo"     # More powerful for grading

# Local embedding & reranker models (download automatically via HuggingFace)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"

# Retrieval settings
DENSE_TOP_K = 25       # Top-k chunks from Qdrant dense search
SPARSE_TOP_K = 25      # Top-k chunks from BM25 sparse search
HYBRID_TOP_K = 50      # Top-k after RRF fusion passed to reranker
RERANK_TOP_N = 15      # Top-n after cross-encoder reranking (covers 3-5 candidates)

# Qdrant Cloud settings
QDRANT_COLLECTION = "resume_chunks_ubs"
EMBEDDING_DIM = 1024    # mxbai-embed-large output dimension

# Rate limiting — Gemini 1.5 Pro free tier: ~5 RPM
GEMINI_PRO_CALL_DELAY = 13  # seconds between Pro API calls
