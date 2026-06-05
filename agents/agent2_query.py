"""
Agent 2 — Query Refinement & Schema Extraction (OpenRouter)

Converts a raw, unstructured job description into a deterministic
JobRequirements object using OpenRouter API + Instructor/Pydantic.
All corporate filler is stripped; only exact technical parameters survive.
"""

import instructor
from openai import OpenAI

from models.schemas import JobRequirements
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, FLASH_MODEL


def refine_job_description(jd_text: str) -> JobRequirements:
    """
    Parses a job description and returns a structured JobRequirements object.
    Guaranteed to conform to the Pydantic schema via Instructor.
    """
    client = instructor.from_openai(
        OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
    )

    prompt = (
        "You are a senior technical recruiter. Extract precise, actionable hiring requirements "
        "from the job description below. Strip all corporate filler, buzzwords, and vague language. "
        "List only concrete technical skills, explicit package/framework names, numeric experience "
        "thresholds, and structural parameters. Be exhaustive for required_skills.\n\n"
        f"Job Description:\n{jd_text}"
    )

    return client.chat.completions.create(
        model=FLASH_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_model=JobRequirements,
    )
