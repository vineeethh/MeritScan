"""
PII Guardrail — masks identity and location fields before any resume text
reaches an LLM. Goal: prevent name, gender, and geographic bias in scoring.

Fields masked:
  PERSON   → <CANDIDATE>          (name bias: ethnic/gender-coded names)
  EMAIL    → <EMAIL_REDACTED>     (often contains candidate's name)
  PHONE    → <PHONE_REDACTED>     (area code leaks location)
  URL      → <URL_REDACTED>       (LinkedIn/GitHub reveals identity)
  LOCATION → <LOCATION_REDACTED>  (home city/address = geographic bias)

Company names, university names, and technical content are left untouched
so the LLM can still assess skills and domain relevance accurately.

First call lazy-loads the spaCy NLP model; subsequent calls reuse the singleton.
"""

from presidio_anonymizer.entities import OperatorConfig
from langchain_experimental.data_anonymizer import PresidioAnonymizer

_OPERATORS = {
    "PERSON":        OperatorConfig("replace", {"new_value": "<CANDIDATE>"}),
    "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL_REDACTED>"}),
    "PHONE_NUMBER":  OperatorConfig("replace", {"new_value": "<PHONE_REDACTED>"}),
    "URL":           OperatorConfig("replace", {"new_value": "<URL_REDACTED>"}),
    "LOCATION":      OperatorConfig("replace", {"new_value": "<LOCATION_REDACTED>"}),
}

_anonymizer: PresidioAnonymizer | None = None


def _get_anonymizer() -> PresidioAnonymizer:
    global _anonymizer
    if _anonymizer is None:
        _anonymizer = PresidioAnonymizer(
            analyzed_fields=list(_OPERATORS.keys()),
            operators=_OPERATORS,
        )
    return _anonymizer


def mask_pii(text: str) -> str:
    """
    Returns a copy of text with identity/location PII masked.
    Safe to call with any string — passthrough if text is empty.
    """
    if not text or not text.strip():
        return text
    return _get_anonymizer().anonymize(text)
