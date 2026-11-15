"""Shared utilities for the Paytm Saathi AI engine.

Helpers are intentionally dependency-light: environment loading, logging,
text cleaning, ChromaDB collection-name sanitization, and filesystem helpers.
No API keys or policy contents are logged.
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from a .env file if one exists (never hardcode keys).
load_dotenv()


def _use_system_trust_store() -> None:
    """Make HTTPS use the operating system's trusted certificates.

    Fixes ``CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate`` on machines where Python's
    own certificate list is missing or incomplete: Macs with python.org Python, and networks or antivirus
    products that re-sign HTTPS traffic with a certificate the OS already trusts. Certificate checking stays ON;
    it only changes which trusted list is used. Needs Python 3.10+ and the ``truststore`` package (skipped
    silently otherwise). Set SAATHI_NO_TRUSTSTORE=1 to turn it off.
    """
    if os.getenv("SAATHI_NO_TRUSTSTORE") == "1":
        return
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:  # not installed, or Python < 3.10: fall back to Python's default certificates
        pass


_use_system_trust_store()

logger = logging.getLogger("paytm_saathi")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
# Never log at DEBUG by default to avoid accidentally recording policy contents.
logger.propagate = False

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "data/vector_db")
POLICIES_DIR: str = os.getenv("POLICIES_DIR", "data/policies")
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K: int = int(os.getenv("TOP_K", "5"))
MIN_RELEVANCE: float = float(os.getenv("MIN_RELEVANCE", "0.35"))
MAX_CONVERSATION_TURNS: int = int(os.getenv("MAX_CONVERSATION_TURNS", "6"))


def sanitize_collection_name(name: str) -> str:
    """Make a policy id safe for use as a ChromaDB collection name and filename.

    ChromaDB collection names must match ^[a-zA-Z0-9_-]{3,64}$.
    """
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "_", name.strip())
    cleaned = cleaned.lower()
    if len(cleaned) < 3:
        cleaned = "policy_" + cleaned
    if len(cleaned) > 64:
        cleaned = cleaned[:64]
    return cleaned


def stable_chunk_id(source_filename: str, page_number: int, index: int) -> str:
    """Deterministic chunk id so re-ingestion produces identical ids."""
    raw = f"{source_filename}:{page_number}:{index}".encode("utf-8")
    return __import__("hashlib").sha256(raw).hexdigest()[:24]


def clean_policy_text(text: str) -> str:
    """Normalize whitespace/formatting artifacts without destroying policy wording.

    - Unicode normalization (NFKC)
    - Form feeds/vertical tabs -> space
    - Collapse horizontal runs of spaces/tabs (keeps newlines)
    - Collapse 3+ consecutive newlines to 2
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\f", " ").replace("\v", " ")
    text = re.sub(r"[ \t\r]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (parents included) if missing; return the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def mask_secret(value: Optional[str]) -> str:
    """Return a masked representation of a secret for safe logging."""
    if not value:
        return ""
    if len(value) <= 7:
        return "***"
    return value[:3] + "***" + value[-4:]


def inr(value: float, decimals: int = 0) -> str:
    """Indian rupee format: 486000 -> '₹4,86,000'; 1234.5 with decimals=2 -> '₹1,234.50'."""
    negative = value < 0
    whole, _, frac = f"{abs(value):.{decimals}f}".partition(".")
    head, tail = whole[:-3], whole[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    text = ",".join(groups + [tail]) if groups else tail
    return f"{'-' if negative else ''}₹{text}" + (f".{frac}" if decimals else "")
