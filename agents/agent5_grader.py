"""
Agent 5 — Structured Grading & Evaluation

Groups the reranked top chunks by candidate, then evaluates each candidate
against the job requirements using Gemini 1.5 Pro. Strict Chain-of-Thought
reasoning and Pydantic validation prevent hallucinated scores.

Rate-limiting: Gemini 1.5 Pro free tier is ~5 RPM. A configurable sleep
is inserted between calls. Pass apply_rate_limit=False in tests.
"""

import time
from collections import defaultdict
from typing import List, Tuple, Dict

import instructor
import google.generativeai as genai

from models.schemas import ChunkWithContext, JobRequirements, CandidateEvaluation
from config import GOOGLE_API_KEY, PRO_MODEL, GEMINI_PRO_CALL_DELAY

genai.configure(api_key=GOOGLE_API_KEY)


def grade_candidates(
    reranked_chunks: List[Tuple[ChunkWithContext, float]],
    job_req: JobRequirements,
    apply_rate_limit: bool = True,
) -> List[CandidateEvaluation]:
    """
    Groups top chunks by candidate file, evaluates each with Gemini Pro,
    and returns evaluations sorted by overall_score descending.
    """
    candidate_chunks: Dict[str, List[ChunkWithContext]] = defaultdict(list)
    for chunk, _ in reranked_chunks:
        candidate_chunks[chunk.candidate_file].append(chunk)

    client = _build_pro_client()
    evaluations: List[CandidateEvaluation] = []

    for i, (candidate_file, chunks) in enumerate(candidate_chunks.items()):
        if i > 0 and apply_rate_limit:
            print(f"  [Rate limit] Waiting {GEMINI_PRO_CALL_DELAY}s before next Pro call...")
            time.sleep(GEMINI_PRO_CALL_DELAY)

        print(f"  Grading: {candidate_file}")
        evaluation = _evaluate_candidate(client, candidate_file, chunks, job_req)
        evaluations.append(evaluation)

    return sorted(evaluations, key=lambda e: e.overall_score, reverse=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_pro_client():
    return instructor.from_gemini(
        client=genai.GenerativeModel(model_name=PRO_MODEL),
        mode=instructor.Mode.GEMINI_JSON,
    )


def _evaluate_candidate(
    client,
    candidate_file: str,
    chunks: List[ChunkWithContext],
    job_req: JobRequirements,
) -> CandidateEvaluation:
    profile = chunks[0].global_profile
    resume_excerpts = "\n\n---\n\n".join(c.text for c in chunks)

    prompt = f"""You are a senior technical recruiter conducting a rigorous, unbiased candidate evaluation.

CRITICAL RULES:
- Score ONLY based on explicit evidence found in the resume excerpts below.
- Do NOT hallucinate skills, projects, or experience that is not clearly stated.
- Use strict Chain-of-Thought: cite evidence first, then reason, then assign a score.
- Scores are 0.0–10.0 where 10.0 = perfectly meets the criterion.

═══════════════════════════════════════════
JOB REQUIREMENTS
═══════════════════════════════════════════
Title:              {job_req.job_title}
Domain:             {job_req.domain}
Required Skills:    {', '.join(job_req.required_skills)}
Preferred Skills:   {', '.join(job_req.preferred_skills)}
Min Experience:     {job_req.min_years_experience} years
Education Required: {job_req.required_education}
Responsibilities:   {'; '.join(job_req.key_responsibilities)}

═══════════════════════════════════════════
CANDIDATE MACRO-PROFILE (from full resume)
═══════════════════════════════════════════
Name:         {profile.candidate_name}
File:         {candidate_file}
Experience:   {profile.total_years_experience} years
Stack:        {', '.join(profile.primary_tech_stack)}
Education:    {profile.highest_education}
Summary:      {profile.summary}

═══════════════════════════════════════════
TOP RELEVANT RESUME EXCERPTS
═══════════════════════════════════════════
{resume_excerpts}

═══════════════════════════════════════════
GRADING CRITERIA
═══════════════════════════════════════════
technical_skills_score  — How many required/preferred skills does the candidate demonstrate?
                          Are they applied in real projects or just listed?
experience_score        — Does total experience meet/exceed the minimum? Is it in a relevant domain?
education_score         — Does the education level and field meet the requirements?
domain_relevance_score  — Is the candidate's past work/research in the same domain as this role?

overall_score (weighted):
  technical_skills = 40%
  experience       = 30%
  domain_relevance = 20%
  education        = 10%

Hire recommendation thresholds:
  STRONG_YES : overall_score >= 8.0
  YES        : overall_score >= 6.5
  MAYBE      : overall_score >= 5.0
  NO         : overall_score <  5.0
"""

    return client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        response_model=CandidateEvaluation,
    )
