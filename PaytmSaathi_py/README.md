# Paytm Saathi – AI & RAG Engine

AI and Retrieval-Augmented Generation (RAG) module for **Paytm Saathi**, an
AI-powered financial assistant for health insurance. This module implements the
**Claim Pre-Flight** feature: users upload a health insurance policy PDF, ask
questions about it in natural language, get answers grounded in the actual
policy document with citations, and receive structured policy information that a
separate calculation backend can use for payout estimation.

> **Scope:** This package is the AI engine only. The FastAPI backend, hospital
> bill extraction, payout calculation engine, and Streamlit frontend are owned
> by other team members and are deliberately not implemented here.

## Features

1. **Policy PDF processing** – page-by-page text extraction with PyMuPDF, page
   numbers preserved, scanned/image-based PDF detection (OCR required, never
   silently empty).
2. **Policy information extraction** – structured extraction of insurer, policy
   name, sum insured, room/ICU rent limits, co-pay, deductibles, waiting
   periods, exclusions, sub-limits, pre/post-hospitalization coverage, claim
   conditions, and clause references. Validated with Pydantic; missing fields
   are `null`, never invented.
3. **Document chunking** – section/clause-aware chunking with configurable size
   and overlap; every chunk carries page, section, clause, and source metadata.
4. **Embeddings & vector store** – Sentence Transformers
   (`all-MiniLM-L6-v2`) + ChromaDB persisted locally; per-policy isolation;
   duplicate ingestion prevention.
5. **Semantic retrieval** – top-k relevant clauses with relevance scores,
   policy-ID filtering, keyword boost for exact policy wording, weak evidence
   dropped rather than returned to fill `top_k`.
6. **RAG question answering** – grounded answers with structured citations,
   conversational context per conversation ID, mandatory anti-hallucination
   instructions.
7. **Plain-language clause explanations** – original wording plus simple
   English explanation (English first, Hindi-ready design).
8. **Structured integration output** – `process_policy`, `ask_policy_question`,
   `get_policy_details`, `explain_policy_clause`, `get_claim_relevant_clauses`.
9. **Claim Pre-Flight support** – relevant clauses and structured conditions
   returned for the calculation backend; no payout calculations here.

## Architecture

```
ai_engine/                 # The AI & RAG package
├── __init__.py            # Public API (process_policy, ask_policy_question, ...)
├── exceptions.py          # Custom exceptions with error codes
├── utils.py               # Env loading, logging, text cleaning, helpers
├── schemas.py             # Pydantic models (all structured I/O)
├── llm_provider.py        # Provider abstraction: mock / openai-compatible / ollama
├── prompts.py             # Mandatory RAG instructions + task prompts
├── document_processor.py  # PDF text extraction (PyMuPDF)
├── text_chunker.py        # Clause-aware chunking
├── embeddings.py          # Sentence Transformers wrapper
├── vector_store.py        # ChromaDB persistent store, per-policy collections
├── retriever.py           # Semantic + keyword-boosted retrieval
├── rag_engine.py          # RAG QA, explanations, claim-relevant clauses
└── policy_extractor.py    # Structured policy extraction + JSON persistence
data/
├── policies/              # Extracted policy details JSON (created at runtime)
└── vector_db/             # Persisted ChromaDB (created at runtime)
tests/                     # Pytest suite (mocked LLM)
scripts/
└── process_policy.py      # CLI for ingesting a policy PDF
```

### Data flow

1. `process_policy(pdf_path, policy_id)`:
   `extract_pdf_pages` → `chunk_pages` → `embed_texts` →
   `vector_store.add_chunks` (skipped if already ingested) → LLM structured
   extraction → Pydantic validation → save JSON.
2. `ask_policy_question(policy_id, question, conversation_id)`:
   embed question → `retriever.retrieve` (policy-isolated collection) → build
   context → LLM answer → parse `[Source N]` citations → `RAGResponse`.
3. `get_claim_relevant_clauses(policy_id, treatment_details)`: retrieve clauses,
   categorize them (coverage/exclusion/limit/waiting_period/copay/deductible),
   return structured conditions for the calculation backend.

The LLM never performs calculations, never guarantees claim approval, and must
say *"I could not find enough information in the uploaded policy to answer this
reliably."* when evidence is insufficient.

## Project structure (at repo root)

```
paytm-saathi/   (= this repo root)
├── ai_engine/
├── data/           (created at runtime; gitignored)
├── tests/
├── scripts/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

### 1. Install Python 3.11+

This environment did not have Python installed, so install it first:

```bash
# Windows: download from https://python.org or use the Microsoft Store
python --version   # must be 3.11+
```

### 2. Clone and create a virtual environment

```bash
cd c:\Users\HP\OneDrive\Desktop\devsemble
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Packages: `python-dotenv`, `pydantic`, `pymupdf`, `sentence-transformers`,
`chromadb`, `pytest`.

> `sentence-transformers` pulls PyTorch (~several hundred MB) and downloads the
> `all-MiniLM-L6-v2` model (~80 MB) on first use. Both happen automatically
> and are cached locally.

### 4. Environment variables

Copy `.env.example` to `.env`:

```bash
copy .env.example .env     # Windows
cp .env.example .env       # Linux/Mac
```

For offline development/hackathon use, leave `LLM_PROVIDER=mock` — no API key
required. The mock provider is honest about not knowing the policy
(insufficient-evidence answers), which demonstrates the engine's safety
behavior.


To use a real LLM, set in `.env`:

```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

Or a local Ollama model:

```
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

API keys are read only from environment variables and are never logged.

## Usage

### Process a policy PDF

```bash
python scripts/process_policy.py --pdf data/policies/my_policy.pdf --policy-id policy_001
```

Output (structured `PolicyDetails`, JSON):

```json
{
  "insurer_name": "Example Health Insurance",
  "policy_name": "Silver Health Plan",
  "policy_type": "Indemnity",
  "sum_insured": {"value": 500000, "currency": "INR", "source_page": 3, "source_clause": "Section 2", "quoted_text": "...", "is_explicit": true},
  "room_rent_limit": {"value": 5000, "currency": "INR", "source_page": 12, "source_clause": "4.2", "is_explicit": true},
  "icu_room_rent_limit": null,
  "copay_percentage": 10,
  "deductibles": null,
  "waiting_periods": [{"condition": "Hospitalization", "duration": "30 days", "duration_days": 30, ...}],
  "exclusions": [],
  "sub_limits": [],
  "unknown_fields": ["icu_room_rent_limit", ...],
  "extracted_at": "2026-...",
  "source_filename": "my_policy.pdf"
}
```

> Values above are illustrative only. Actual values come from the uploaded
> policy; missing fields are `null` and listed in `unknown_fields`.

### Ask a question (Python)

```python
from ai_engine import ask_policy_question, RAGResponse

response: RAGResponse = ask_policy_question(
    policy_id="policy_001",
    question="What is my room rent limit?",
    conversation_id="conv-1",
)

print(response.answer)
# Your room rent limit is ₹5,000 per day. [Source 1]

print(response.model_dump_json(indent=2))
```

Example response JSON:

```json
{
  "policy_id": "policy_001",
  "question": "What is my room rent limit?",
  "answer": "Your policy specifies a room rent limit of ₹5,000 per day.",
  "sources": [
    {
      "page_number": 12,
      "section_name": "Room Rent",
      "clause_reference": "4.2",
      "quoted_text": "Room rent is limited to ₹5,000 per day..."
    }
  ],
  "confidence": "high",
  "insufficient_evidence": false,
  "conversation_id": "conv-1"
}
```

When evidence is missing:

```json
{
  "policy_id": "policy_001",
  "question": "Is cataract surgery covered?",
  "answer": "I could not find enough information in the uploaded policy to answer this reliably.",
  "sources": [],
  "confidence": "low",
  "insufficient_evidence": true
}
```

### Follow-up questions

```python
r1 = ask_policy_question("policy_001", "Is ICU covered?", conversation_id="conv-1")
r2 = ask_policy_question("policy_001", "What is the limit for that?",
                          conversation_id="conv-1")
```


### Importing in a FastAPI backend

```python
from ai_engine import process_policy, ask_policy_question, get_policy_details
from ai_engine.exceptions import PolicyNotFoundError, ScannedPDFError

# POST /policies/{policy_id}/ingest
details = process_policy("data/policies/file.pdf", "policy_001")

# GET /policies/{policy_id}/details
details = get_policy_details("policy_001")   # raises PolicyNotFoundError if not ingested

# POST /policies/{policy_id}/ask
response = ask_policy_question("policy_001", "What is the co-pay?", "conv-42")
```

## Running tests

```bash
pytest -q
```

Tests mock the LLM provider so no API key or network is needed. They cover:

- PDF text extraction, page-number preservation, scanned PDF detection,
  empty/corrupt PDF handling
- Structured extraction, Pydantic validation, coercion (₹5,00,000 → 500000),
  safe handling of invalid LLM JSON
- Retrieval filtering/ranking, relevance threshold, policy isolation,
  embedding failure
- RAG citations, insufficient evidence, invalid questions, LLM failure,
  conversation history

## What is implemented vs. still to configure

| Item | Status |
|---|---|
| PDF extraction with page numbers | ✅ Implemented |
| Scanned PDF detection | ✅ Implemented |
| Structured policy extraction (Pydantic) | ✅ Implemented |
| Clause-aware chunking with metadata | ✅ Implemented |
| Sentence Transformers embeddings | ✅ Implemented |
| ChromaDB persistent per-policy store | ✅ Implemented |
| Duplicate ingestion prevention | ✅ Implemented |
| Semantic + keyword retrieval with scores | ✅ Implemented |
| RAG QA with citations & conversation | ✅ Implemented |
| Plain-language clause explanation | ✅ Implemented |
| Claim Pre-Flight clauses output | ✅ Implemented |
| LLM provider: mock | ✅ Ready (default) |
| LLM provider: OpenAI-compatible API | ✅ Implemented — set `OPENAI_API_KEY` in `.env` |
| LLM provider: Ollama | ✅ Implemented — run Ollama locally |
| FastAPI integration | ⬜ Wrapper endpoints to be written by backend team |
| Streamlit UI | ⬜ To be written by frontend team |
| OCR for scanned PDFs | ⬜ Integration point (engine reports OCR required) |
| Hospital bill extraction / payout calculation | ⬜ Owned by other team member |

## Security notes

- Only `.pdf` uploads are accepted; uploaded content is never executed.
- API keys live only in environment variables (`.env`, gitignored) and are
  masked in any log output.
- Policy data is isolated per `policy_id` (separate ChromaDB collection).
- No API keys or policy contents are logged unnecessarily.

The second call receives the previous turn as conversational context, but the
answer is still grounded in the retrieved policy clauses.

### Explain a clause

```python
from ai_engine import explain_policy_clause

explanation = explain_policy_clause("policy_001", "4.2")
print(explanation.original_clause)
print(explanation.simple_explanation)
```

### Claim Pre-Flight clauses (for the calculation backend)

```python
from ai_engine import get_claim_relevant_clauses

result = get_claim_relevant_clauses(
    "policy_001",
    treatment_details={"procedure": "cataract surgery", "hospital": "network"},
)
# result.clauses        -> categorized clauses with condition_type
# result.structured_conditions -> sum insured, room rent, copay, waiting periods, exclusions, ...
# result.missing_information -> fields not found in the policy
```

No financial calculations are performed by this module — only rule retrieval
and structuring.
