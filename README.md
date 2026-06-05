# MeritScan

A production-ready, modular **Multi-Agent Resume Screening Framework** that eliminates the two core flaws of legacy resume screening: semantic blindness from keyword matching, and context fragmentation from naive RAG chunking.

---

## Architecture

```
Job Description (raw text)
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  Agent 2 — Query Refinement (OpenRouter)              │
│  Strips filler text → structured JobRequirements      │
└───────────────────┬───────────────────────────────────┘
                    │
PDF Resumes ────────┼──────────────────────────────────┐
                    │                                  │
                    ▼                                  ▼
┌───────────────────────────────────────────────────────┐
│  Agent 1 — Ingestion & Global Context Enrichment      │
│  pymupdf4llm  →  MarkdownHeaderTextSplitter           │
│  + OpenRouter global profile injected into chunks     │
└───────────────────┬───────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────┐
│  Agent 3 — Hybrid Retrieval (Dense + Sparse + RRF)    │
│  Qdrant (Ollama mxbai-embed-large) ┐                 │
│                                    ├─ RRF → Top 50   │
│  BM25Okapi (rank_bm25)  ───────────┘                 │
└───────────────────┬───────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────┐
│  Agent 4 — Cross-Encoder Reranker                     │
│  BAAI/bge-reranker-base → Top 15 high-precision chunks│
└───────────────────┬───────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────┐
│  Agent 5 — Structured Grading (OpenRouter)            │
│  Instructor + Pydantic → Chain-of-Thought evaluation  │
│  Scores: Technical / Experience / Education / Domain  │
└───────────────────┬───────────────────────────────────┘
                    │
                    ▼
         Ranked JSON results
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo>
cd MeritScan
pip install -r requirements.txt
```

> PyTorch is included in requirements. For GPU acceleration install the CUDA build manually.

### 2. Configure your API keys

```bash
cp .env.example .env
# Edit .env and add your API keys:
# OPENROUTER_API_KEY=your_openrouter_key
# QDRANT_URL=your_qdrant_url
# QDRANT_API_KEY=your_qdrant_api_key
```

Required APIs:
- **OpenRouter** (for candidate grading): Get API key at [openrouter.ai](https://openrouter.ai)
- **Qdrant Cloud** (for vector database): Get free instance at [qdrant.tech](https://qdrant.tech/)
- **Ollama** (for local embeddings): Download from [ollama.ai](https://ollama.ai)

---

## Usage

```bash
python main.py --jd path/to/job_description.txt --resumes path/to/resumes/
```

| Argument | Description |
|---|---|
| `--jd` | Path to a plain-text job description file |
| `--resumes` | Directory containing PDF resumes |
| `--output` | Output JSON path (default: `results.json`) |

### Example

```bash
python main.py \
  --jd data/jd_ml_engineer.txt \
  --resumes data/resumes/ \
  --output data/results/ml_engineer_run1.json
```

---

## Output

Results are printed to stdout and saved as JSON:

```
==============================
  RANKED CANDIDATES
==============================

  #1  Jane Smith  (jane_smith.pdf)
       Score: 8.7/10  [████████░░]  → STRONG_YES
       Technical: 9.0  |  Experience: 8.5  |  Domain: 9.0  |  Education: 7.0
       Strengths: 5 yrs PyTorch production experience • Published NLP research
       Gaps:      No MLOps/deployment experience mentioned
       Rationale: Strong ML background with direct domain match ...
```

---

## Project Structure

```
MeritScan/
├── main.py                    # Pipeline orchestrator & CLI entry point
├── config.py                  # All constants and environment variables
├── requirements.txt
├── .env.example
├── agents/
│   ├── agent1_ingestion.py    # PDF → Markdown → enriched chunks
│   ├── agent2_query.py        # JD → structured JobRequirements
│   ├── agent3_retrieval.py    # Hybrid dense+sparse retrieval (RRF)
│   ├── agent4_reranker.py     # Cross-encoder reranking
│   └── agent5_grader.py       # CoT grading with Gemini Pro
├── models/
│   └── schemas.py             # All Pydantic schemas
└── data/
    ├── resumes/               # Drop PDF resumes here
    └── results/               # Output JSON files land here
```

---

## Why Each Component Was Chosen

| Component | Why |
|---|---|
| **pymupdf4llm** | Preserves tables, bullets, and column layout in academic/professional PDFs |
| **MarkdownHeaderTextSplitter** | Splits at section boundaries (Experience, Projects) not arbitrary token counts |
| **BAAI/bge-small-en-v1.5** | Top-tier open-source English embedding model; runs locally, no API needed |
| **BM25Okapi** | Catches exact framework names (e.g. `FastAPI`, `BioBERT`) that semantic search misses |
| **Reciprocal Rank Fusion** | Merges dense + sparse rankings without needing score normalization |
| **BAAI/bge-reranker-base** | Full cross-attention on [query, chunk] pairs; eliminates false positives from bi-encoder |
| **Gemini 1.5 Flash** | Fast + free for ingestion and JD parsing; 1500 RPD on free tier |
| **Gemini 1.5 Pro** | 2M token context window; strong CoT reasoning for unbiased grading |
| **Instructor + Pydantic** | Guarantees structured JSON output; prevents hallucinated scores |
