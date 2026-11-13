#!/usr/bin/env python3
"""CLI to process a health insurance policy PDF through the Paytm Saathi AI engine.

Example:
    python scripts/process_policy.py --pdf data/policies/sample_policy.pdf --policy-id policy_001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this script directly: python scripts/process_policy.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_engine import process_policy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a health insurance policy PDF into the AI engine."
    )
    parser.add_argument("--pdf", required=True, help="Path to the policy PDF file.")
    parser.add_argument(
        "--policy-id",
        required=True,
        help="Unique id for this policy (used for vector-store isolation).",
    )
    args = parser.parse_args()

    try:
        details = process_policy(args.pdf, args.policy_id)
        print(details.model_dump_json(indent=2))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
