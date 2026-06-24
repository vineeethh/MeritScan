# MeritScan AI Agent Instructions

This document provides essential context for AI agents working on the MeritScan codebase.

## Project Overview

MeritScan is a multi-agent resume screening framework designed to overcome the limitations of traditional keyword-based and naive RAG systems. It uses a pipeline of specialized agents to ingest, analyze, and rank candidates against a given job description.

For a visual overview of the architecture, see the diagram in the [README.md](README.md#architecture).

## Key Technologies & Conventions

- **Orchestration**: The main pipeline is orchestrated in `main.py`.
- **Configuration**: All constants, API keys, and model names are managed in `config.py`.
- **Data Models**: All Pydantic schemas are defined in `models/schemas.py`. These are crucial for structured data exchange between agents.
- **LLM Interaction**: The project uses both local (Ollama) and remote (OpenRouter) LLMs.
  - **OpenRouter**: Used for tasks requiring powerful reasoning, such as query refinement (`agent2_query.py`) and candidate grading (`agent5_grader.py`). The `instructor` library is used to ensure structured output from the LLM.
  - **Ollama**: Used for generating embeddings locally for dense retrieval (`agent3_retrieval.py`) and reranking (`agent4_reranker.py`). The model used is `mxbai-embed-large`.
- **Vector Database**: [Qdrant Cloud](https://qdrant.tech/) is used for storing and retrieving embeddings.
- **PDF Parsing**: `pymupdf4llm` is used for layout-aware PDF-to-Markdown conversion in `agent1_ingestion.py`.
- **Text Splitting**: `MarkdownHeaderTextSplitter` is the primary method for semantic chunking.
- **Retrieval Strategy**: A hybrid approach is used:
  1.  **Dense Retrieval**: Ollama embeddings + Qdrant.
  2.  **Sparse Retrieval**: `rank_bm25` for keyword matching.
  3.  **Fusion**: Reciprocal Rank Fusion (RRF) to combine dense and sparse results.
- **Reranking**: A second-stage reranker (`agent4_reranker.py`) uses Ollama embeddings to improve the precision of the retrieved chunks.
- **Guardrails**: The `guardrails/` directory contains modules for input validation and PII masking.

## Development Workflow

1.  **Setup**: Follow the instructions in the [README.md](README.md#setup) to install dependencies and configure API keys.
2.  **Running the pipeline**: Use `main.py` with the `--jd` and `--resumes` arguments as described in the [README.md](README.md#usage).
3.  **Modifying Agents**: Each agent is a self-contained module in the `agents/` directory. When modifying an agent, ensure it adheres to the Pydantic schemas defined in `models/schemas.py`.

## Common Tasks

- **Adding a new agent**: Create a new file in the `agents/` directory and integrate it into the `main.py` pipeline.
- **Changing an LLM model**: Update the model name in `config.py`.
- **Modifying a Pydantic schema**: Update the relevant schema in `models/schemas.py` and ensure all agents that use it are updated accordingly.
- **Improving a prompt**: Prompts are located within the agent files that use them. For example, the grading prompt is in `agent5_grader.py`.
