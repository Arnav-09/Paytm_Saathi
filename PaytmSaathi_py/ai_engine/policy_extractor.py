"""Feature 2: Policy information extraction.

Extracts structured insurance information from a policy document using the
configured LLM. All LLM output is validated with Pydantic; missing values are
reported as null and listed in unknown_fields. Nothing is ever invented.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .exceptions import PolicyNotFoundError, ValidationError as PSAValidationError
from .llm_provider import LLMProvider, get_llm_provider
from .prompts import structured_extraction_prompt
from .schemas import Chunk, PageContent, PolicyDetails
from .utils import (
    POLICIES_DIR,
    ensure_dir,
    logger,
    sanitize_collection_name,
)


KEY_FIELDS = [
    "insurer_name",
    "policy_name",
    "sum_insured",
    "room_rent_limit",
    "copay_percentage",
    "exclusions",
    "waiting_periods",
    "claim_conditions",
]


class PolicyExtractor:
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        self.llm = llm_provider or get_llm_provider()

    def extract(
        self,
        pages: list[PageContent],
        source_filename: Optional[str] = None,
    ) -> PolicyDetails:
        """Extract structured policy details from extracted pages.

        Returns a PolicyDetails instance with nulls for any field the LLM
        could not ground in the document. If the LLM returns invalid JSON or
        data that fails Pydantic validation, a safe empty instance is
        returned instead of raising.
        """
        file_name = source_filename or (pages[0].source_filename if pages else "unknown.pdf")
        document_text = self._format_document(pages)

        if not document_text.strip():
            logger.warning("No text to extract from for file %s", file_name)
            return PolicyDetails(
                source_filename=file_name,
                unknown_fields=[
                    "insurer_name",
                    "policy_name",
                    "sum_insured",
                    "room_rent_limit",
                    "copay_percentage",
                    "exclusions",
                    "waiting_periods",
                    "claim_conditions",
                ],
                extracted_at=datetime.now(timezone.utc).isoformat(),
            )

        prompt = structured_extraction_prompt(document_text)
        schema = PolicyDetails.model_json_schema()

        try:
            data = self.llm.generate_structured(prompt, schema=schema)
        except (PSAValidationError, Exception) as exc:  # LLM or JSON errors
            if not isinstance(exc, PSAValidationError):
                logger.warning("Structured extraction failed for %s: %s", file_name, exc)
            data = {}

        if not isinstance(data, dict):
            logger.warning("LLM structured extraction did not return an object for %s", file_name)
            data = {}

        try:
            details = PolicyDetails.model_validate(data)
        except Exception as exc:  # Pydantic validation failure
            logger.warning(
                "Pydantic validation failed for extraction of %s: %s", file_name, exc
            )
            details = PolicyDetails(
                source_filename=file_name,
                unknown_fields=["extraction_validation_failed"],
                extracted_at=datetime.now(timezone.utc).isoformat(),
            )

        details.source_filename = file_name
        details.extracted_at = details.extracted_at or datetime.now(timezone.utc).isoformat()
        # Don't rely on the LLM to report gaps: derive them from the values themselves.
        missing = [f for f in KEY_FIELDS if getattr(details, f) in (None, [], "")]
        details.unknown_fields = list(dict.fromkeys(list(details.unknown_fields) + missing))
        return details

    @staticmethod
    def _format_document(pages: list[PageContent]) -> str:
        """Join pages with page markers so the LLM can cite page numbers."""
        parts: list[str] = []
        for page in pages:
            if page.page_text.strip():
                parts.append(f"--- Page {page.page_number} ---\n{page.page_text}")
        return "\n\n".join(parts)

    def extract_from_chunks(self, chunks: list[Chunk]) -> PolicyDetails:
        """Convenience: extract from chunks (deduplicated by page)."""
        pages_by_number: dict[int, str] = {}
        source_filename = chunks[0].source_filename if chunks else "unknown.pdf"
        for chunk in chunks:
            existing = pages_by_number.get(chunk.page_number, "")
            pages_by_number[chunk.page_number] = (
                existing + "\n" + chunk.text if existing else chunk.text
            )
        pages = [
            PageContent(
                page_number=page_number,
                page_text=text,
                source_filename=source_filename,
            )
            for page_number, text in sorted(pages_by_number.items())
        ]
        return self.extract(pages, source_filename=source_filename)


def save_policy_details(
    policy_id: str, details: PolicyDetails, policies_dir: Optional[str] = None
) -> str:
    """Persist extracted policy details as JSON for get_policy_details()."""
    directory = ensure_dir(policies_dir or POLICIES_DIR)
    path = directory / f"{sanitize_collection_name(policy_id)}.json"
    path.write_text(details.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Saved policy details to %s", path)
    return str(path)


def load_policy_details(
    policy_id: str, policies_dir: Optional[str] = None
) -> PolicyDetails:
    """Load previously saved policy details. Raises PolicyNotFoundError if absent."""
    directory = Path(policies_dir or POLICIES_DIR)
    path = directory / f"{sanitize_collection_name(policy_id)}.json"
    if not path.exists():
        raise PolicyNotFoundError(policy_id)
    try:
        data = path.read_text(encoding="utf-8")
        return PolicyDetails.model_validate_json(data)
    except PolicyNotFoundError:
        raise
    except Exception as exc:
        raise PolicyNotFoundError(f"Policy {policy_id} details are corrupt: {exc}") from exc


def list_policy_ids(policies_dir: Optional[Path] = None) -> list[str]:
    """Ids (sanitized file stems) of every policy whose details have been saved."""
    directory = Path(policies_dir or POLICIES_DIR)
    if not directory.exists():
        return []
    return sorted(p.stem for p in directory.glob("*.json"))
