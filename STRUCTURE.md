# Project Structure

```
MeritScan/
├── main.py                      # CLI entry point & pipeline orchestrator
├── config.py                    # Configuration & environment variables
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
├── .env                         # (local) Environment variables (git-ignored)
├── .gitignore                   # Git ignore rules
├── README.md                    # Project documentation
├── STRUCTURE.md                 # This file
│
├── agents/                      # Multi-agent pipeline components
│   ├── __init__.py
│   ├── agent1_ingestion.py      # PDF → Markdown chunks with context
│   ├── agent2_query.py          # Job description → structured requirements
│   ├── agent3_retrieval.py      # Hybrid retrieval (dense + sparse + RRF)
│   ├── agent4_reranker.py       # Cross-encoder reranking
│   └── agent5_grader.py         # Chain-of-thought candidate evaluation
│
├── models/                      # Data schemas & types
│   ├── __init__.py
│   └── schemas.py               # Pydantic models for all data structures
│
├── data/                        # Data directory (git-ignored except structure)
│   ├── resumes/                 # Input: PDF resume files
│   │   ├── resume1.pdf
│   │   ├── resume2.pdf
│   │   └── ...
│   │
│   ├── results/                 # Output: Screening results (JSON)
│   │   ├── ubs_screening_results.json
│   │   └── text_extractions.json
│   │
│   └── UBS_Job_Description.txt  # Job description for screening
│
└── scripts/                     # (Optional) Utility scripts
    └── run_with_ollama.sh       # Helper script to start Ollama & run pipeline
```

## File Descriptions

### Core Files
- **main.py**: Entry point. Run with `--jd` and `--resumes` arguments.
- **config.py**: Loads environment variables and defines pipeline constants.

### Agents (Pipeline Components)
1. **Agent 1** (Ingestion): Converts PDFs to markdown chunks, enriches with global context
2. **Agent 2** (Query Refinement): Parses job description into structured requirements
3. **Agent 3** (Retrieval): Hybrid search combining dense (Qdrant) + sparse (BM25) vectors
4. **Agent 4** (Reranker): Cross-encoder reranking for precision filtering
5. **Agent 5** (Grader): Chain-of-thought evaluation with Instructor + Pydantic

### Data Directories
- **data/resumes/**: Place PDF resumes here for screening
- **data/results/**: Output JSON files with candidate rankings and scores
- **data/UBS_Job_Description.txt**: Template for job description

## Configuration

Required environment variables in `.env`:
```
OPENROUTER_API_KEY=...
QDRANT_URL=...
QDRANT_API_KEY=...
```

See `.env.example` for the template.
