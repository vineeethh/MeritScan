from pydantic import BaseModel, Field
from typing import Optional, List


class CandidateGlobalProfile(BaseModel):
    """Holistic 'macro' profile extracted from the full resume by Gemini Flash."""
    candidate_name: str = Field(description="Candidate's full name; 'Unknown' if not found")
    summary: str = Field(description="3-sentence holistic summary of the candidate's overall profile")
    total_years_experience: float = Field(description="Estimated total years of professional/research experience")
    primary_tech_stack: List[str] = Field(description="Top 5-7 primary technologies, frameworks, or tools")
    highest_education: str = Field(description="Highest degree level, e.g. PhD, MS, BS, Associate, High School")


class JobRequirements(BaseModel):
    """Structured extraction of a raw job description — no filler text."""
    job_title: str = Field(description="Job title / role being hired for")
    domain: str = Field(description="Industry or technical domain, e.g. ML/AI, Backend, DevOps, Biotech")
    required_skills: List[str] = Field(description="Non-negotiable technical skills, frameworks, or tools")
    preferred_skills: List[str] = Field(description="Nice-to-have skills that strengthen a candidate's fit")
    min_years_experience: float = Field(description="Minimum years of relevant professional experience required")
    required_education: str = Field(description="Minimum education level required, e.g. BS, MS, PhD")
    key_responsibilities: List[str] = Field(description="Primary day-to-day responsibilities (max 5)")


class ChunkWithContext(BaseModel):
    """A resume text chunk with the candidate's global macro-profile injected."""
    chunk_id: str
    candidate_file: str
    text: str
    global_profile: CandidateGlobalProfile
    section_header: Optional[str] = None


class CriterionScore(BaseModel):
    """Chain-of-thought score for a single evaluation criterion."""
    criterion: str
    score: float = Field(ge=0.0, le=10.0, description="Score from 0.0 (no evidence) to 10.0 (perfect match)")
    evidence: str = Field(description="Direct quote or specific finding from the resume supporting this score")
    reasoning: str = Field(description="Step-by-step reasoning explaining how the evidence maps to the score")


class CandidateEvaluation(BaseModel):
    """Structured grading output for a single candidate."""
    candidate_name: str
    candidate_file: str
    technical_skills_score: CriterionScore
    experience_score: CriterionScore
    education_score: CriterionScore
    domain_relevance_score: CriterionScore
    overall_score: float = Field(ge=0.0, le=10.0, description="Weighted overall score (tech 40%, exp 30%, domain 20%, edu 10%)")
    hire_recommendation: str = Field(description="One of: STRONG_YES / YES / MAYBE / NO")
    key_strengths: List[str] = Field(description="Top 3 candidate strengths most relevant to the role")
    key_gaps: List[str] = Field(description="Top 3 gaps or concerns relative to the job requirements")
    summary_rationale: str = Field(description="2-3 sentence overall hiring rationale")
