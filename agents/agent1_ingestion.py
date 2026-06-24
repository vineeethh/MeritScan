"""
Agent 1 — Ingestion & Global Context Enrichment (OpenRouter)

Converts PDF resumes to layout-aware Markdown using pymupdf4llm,
splits by section headers, then uses OpenRouter to generate a holistic global
profile. That profile is injected into every chunk so downstream retrieval
never suffers from context fragmentation.
"""

import os
import uuid
import instructor
from openai import OpenAI
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from typing import List

from models.schemas import CandidateGlobalProfile, ChunkWithContext
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, FLASH_MODEL
from guardrails.pii_masker import mask_pii
from guardrails.input_validator import validate_input

_HEADERS_TO_SPLIT = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3"),
]

_FALLBACK_CHUNK_SIZE = 800
_FALLBACK_CHUNK_OVERLAP = 100
_MIN_CHUNK_LENGTH = 40


def _pdf_to_markdown(pdf_path: str) -> str:
    return pymupdf4llm.to_markdown(
        pdf_path,
        page_chunks=False,
        show_progress=False,
        write_images=False,
    )


def extract_resume_chunks(pdf_path: str) -> List[ChunkWithContext]:
    """
    Full ingestion pipeline for a single PDF resume.
    Returns a list of context-enriched chunks ready for indexing.
    """
    filename = os.path.basename(pdf_path)

    # Step 1: PDF → structured Markdown (pymupdf4llm layout model)
    md_text = _pdf_to_markdown(pdf_path)

    # Guardrail: reject resumes containing adversarial/malicious content
    guard = validate_input(md_text, "resume")
    if not guard.passed:
        raise ValueError(f"Resume '{filename}' blocked [{guard.layer}]: {guard.reason}")

    # Step 2: Split by Markdown headers for semantic continuity
    docs = _split_by_headers(md_text)

    # Fallback: if header splitting produces fewer than 3 chunks, use character splitter
    if len(docs) < 3:
        docs = _fallback_character_split(md_text)

    # Step 3: Generate global macro-profile from the full document (one LLM call)
    global_profile = _generate_global_profile(md_text, filename)

    # Step 4: Inject global context into each chunk
    chunks = []
    for doc in docs:
        text = doc.page_content if hasattr(doc, "page_content") else doc
        if len(text.strip()) < _MIN_CHUNK_LENGTH:
            continue
        header = (
            " > ".join(str(v) for v in doc.metadata.values())
            if hasattr(doc, "metadata") and doc.metadata
            else None
        )
        chunks.append(
            ChunkWithContext(
                chunk_id=str(uuid.uuid4()),
                candidate_file=filename,
                text=text.strip(),
                global_profile=global_profile,
                section_header=header,
            )
        )

    return chunks


def _split_by_headers(md_text: str):
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS_TO_SPLIT,
        strip_headers=False,
    )
    return splitter.split_text(md_text)


def _fallback_character_split(md_text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_FALLBACK_CHUNK_SIZE,
        chunk_overlap=_FALLBACK_CHUNK_OVERLAP,
        separators=["\n\n", "\n", " "],
    )

    class _FakeDoc:
        def __init__(self, text):
            self.page_content = text
            self.metadata = {}

    return [_FakeDoc(chunk) for chunk in splitter.split_text(md_text)]


def _generate_global_profile(full_resume_text: str, filename: str) -> CandidateGlobalProfile:
    """
    Single OpenRouter call over the entire document.
    Produces the structured macro-profile that gets injected into every chunk.
    """
    client = instructor.from_openai(
        OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
    )

    safe_text = mask_pii(full_resume_text[:15000])

    prompt = (
        "You are parsing a resume. Extract structured information precisely. "
        "Do not infer or guess — only report what is explicitly stated.\n\n"
        f"Resume ({filename}):\n{safe_text}"
    )

    return client.chat.completions.create(
        model=FLASH_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_model=CandidateGlobalProfile,
    )
