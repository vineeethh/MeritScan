"""
MeritScan — Multi-Agent Resume Screening Pipeline

Usage:
    python main.py --jd path/to/job_description.txt --resumes path/to/resumes/ [--output results.json]

Pipeline:
    Agent 2 → Agent 1 → Agent 3 → Agent 4 → Agent 5
    (Query)   (Ingest)  (Hybrid   (Rerank)  (Grade)
               (PDF→MD)  Retrieval)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

from agents.agent1_ingestion import extract_resume_chunks
from agents.agent2_query import refine_job_description
from agents.agent3_retrieval import index_chunks, hybrid_retrieve
from agents.agent4_reranker import rerank_chunks, build_rerank_query
from agents.agent5_grader import grade_candidates
from models.schemas import CandidateEvaluation
from config import HYBRID_TOP_K, RERANK_TOP_N


def run_pipeline(jd_path: str, resumes_dir: str, output_path: str = "results.json") -> None:
    print("\n" + "=" * 60)
    print("  MeritScan: Multi-Agent Resume Screening Pipeline")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Agent 2: Refine job description
    # ------------------------------------------------------------------
    print("[Agent 2] Extracting structured job requirements...")
    with open(jd_path, "r", encoding="utf-8") as f:
        jd_text = f.read()

    job_req = refine_job_description(jd_text)
    print(f"  Role:             {job_req.job_title}")
    print(f"  Domain:           {job_req.domain}")
    print(f"  Required Skills:  {', '.join(job_req.required_skills[:6])}")
    print(f"  Min Experience:   {job_req.min_years_experience} yrs")
    print(f"  Education:        {job_req.required_education}\n")

    # ------------------------------------------------------------------
    # Agent 1: Ingest all PDF resumes
    # ------------------------------------------------------------------
    print("[Agent 1] Ingesting resumes and building global profiles...")
    pdf_files = sorted(Path(resumes_dir).glob("*.pdf"))
    if not pdf_files:
        print(f"  ERROR: No PDF files found in '{resumes_dir}'")
        sys.exit(1)

    all_chunks = []
    for pdf_path in pdf_files:
        print(f"  Processing: {pdf_path.name}")
        try:
            chunks = extract_resume_chunks(str(pdf_path))
            all_chunks.extend(chunks)
            p = chunks[0].global_profile
            print(
                f"    Candidate: {p.candidate_name} | "
                f"{p.total_years_experience} yrs exp | "
                f"{len(chunks)} chunks extracted"
            )
        except Exception as exc:
            print(f"    WARNING: Failed to process {pdf_path.name}: {exc}")

    if not all_chunks:
        print("  ERROR: No chunks extracted from any resume.")
        sys.exit(1)

    print(f"\n  Total chunks across all candidates: {len(all_chunks)}\n")

    # ------------------------------------------------------------------
    # Agent 3: Index + Hybrid Retrieval
    # ------------------------------------------------------------------
    print("[Agent 3] Building hybrid index (Qdrant dense + BM25 sparse)...")
    index_chunks(all_chunks)

    print(f"\n[Agent 3] Running hybrid retrieval (RRF fusion, top {HYBRID_TOP_K})...")
    retrieved = hybrid_retrieve(job_req, top_k=HYBRID_TOP_K)

    unique_candidates = {c.candidate_file for c, _ in retrieved}
    print(f"  Retrieved {len(retrieved)} chunks across {len(unique_candidates)} candidates\n")

    # ------------------------------------------------------------------
    # Agent 4: Cross-encoder reranking
    # ------------------------------------------------------------------
    print(f"[Agent 4] Cross-encoder reranking to top {RERANK_TOP_N} chunks...")
    rerank_query = build_rerank_query(job_req)
    reranked = rerank_chunks(rerank_query, retrieved, top_n=RERANK_TOP_N)

    finalists = {c.candidate_file for c, _ in reranked}
    print(f"  Reranked to {len(reranked)} chunks | Finalist candidates: {len(finalists)}")
    print(f"  Finalists: {', '.join(finalists)}\n")

    # ------------------------------------------------------------------
    # Agent 5: Structured grading with Gemini 1.5 Pro
    # ------------------------------------------------------------------
    print("[Agent 5] Grading candidates (Gemini 1.5 Pro — Chain-of-Thought)...")
    evaluations: List[CandidateEvaluation] = grade_candidates(reranked, job_req)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    _print_results(evaluations)
    _save_results(evaluations, output_path)


def _print_results(evaluations: List[CandidateEvaluation]) -> None:
    print("\n" + "=" * 60)
    print("  RANKED CANDIDATES")
    print("=" * 60)
    for rank, ev in enumerate(evaluations, 1):
        bar = "█" * int(ev.overall_score) + "░" * (10 - int(ev.overall_score))
        print(f"\n  #{rank}  {ev.candidate_name}  ({ev.candidate_file})")
        print(f"       Score: {ev.overall_score:.1f}/10  [{bar}]  → {ev.hire_recommendation}")
        print(
            f"       Technical: {ev.technical_skills_score.score:.1f}  |  "
            f"Experience: {ev.experience_score.score:.1f}  |  "
            f"Domain: {ev.domain_relevance_score.score:.1f}  |  "
            f"Education: {ev.education_score.score:.1f}"
        )
        if ev.key_strengths:
            print(f"       Strengths: {' • '.join(ev.key_strengths[:2])}")
        if ev.key_gaps:
            print(f"       Gaps:      {' • '.join(ev.key_gaps[:2])}")
        print(f"       Rationale: {ev.summary_rationale}")
    print()


def _save_results(evaluations: List[CandidateEvaluation], output_path: str) -> None:
    data = [ev.model_dump() for ev in evaluations]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Full results saved to: {output_path}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MeritScan: Multi-Agent Resume Screening Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py --jd jd.txt --resumes data/resumes/\n"
            "  python main.py --jd jd.txt --resumes data/resumes/ --output data/results/run1.json\n"
        ),
    )
    parser.add_argument("--jd", required=True, help="Path to the job description (.txt file)")
    parser.add_argument("--resumes", required=True, help="Directory containing PDF resumes")
    parser.add_argument("--output", default="results.json", help="Output JSON path (default: results.json)")
    args = parser.parse_args()

    run_pipeline(args.jd, args.resumes, args.output)


if __name__ == "__main__":
    main()
