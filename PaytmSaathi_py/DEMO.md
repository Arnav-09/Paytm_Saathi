# Paytm Saathi – Demo Document

**Paytm Saathi** is an AI-powered financial assistant for health insurance, built as a Retrieval-Augmented Generation (RAG) system. It helps users understand their health insurance policies, ask questions in natural language, estimate claim payouts, check claim readiness, and manage lending and fintech journeys — all with multilingual support and safety guardrails.

---

## 1. Project Overview

| Attribute | Details |
|-----------|---------|
| **Project Name** | Paytm Saathi |
| **Purpose** | AI-powered financial assistant for health insurance claims, lending, and payments |
| **Core Technology** | Python, FastAPI, Streamlit, Sentence Transformers, ChromaDB, PyMuPDF, Pydantic |
| **AI Providers** | Mock (demo), OpenAI-compatible, Ollama, Groq, Gemini |
| **Languages** | English, हिन्दी, मराठी, বাংলা, தமிழ், తెలుగు, ગુજરાતી |
| **License** | Internal / Hackathon |

### What It Does

Paytm Saathi lets a user:

1. **Upload a health insurance policy PDF** (text-based or scanned).
2. **Ask questions** about the policy in natural language (e.g., "What is my room rent limit?").
3. **Get grounded answers** with citations pointing to specific pages and clauses.
4. **Estimate claim payouts** by uploading a hospital bill and walking through deductions (room rent, co-pay, deductible, sub-limits, sum insured cap).
5. **Generate a Claim Readiness Report** — a pre-flight check showing cashless eligibility, readiness score, pre-auth requirements, room-rent impact, and required documents.
6. **Explain confusing clauses** in plain language (original wording + simple explanation).
7. **Calculate loan EMIs, check eligibility, compare offers, and plan prepayments** (lending journey).
8. **Validate KYC fields, create dispute tickets, check scam messages, and attach evidence** (fintech journey).
9. **Chat with a general assistant** (Saathi Chat) backed by RAG over knowledge base + MCP tools, with input/output guardrails.

### What It Does NOT Do

- The LLM **never performs financial calculations** (payouts, EMIs, eligibility are pure Python).
- The LLM **never guarantees claim approval** — every result is an estimate with a disclaimer.
- Missing policy values stay `null` and are listed as "could not check" — nothing is invented.
- No KYC is collected at sign-up (only basic details: name, mobile, email, password, language).

---

## 2. Architecture

```
Paytm_Saathi/
├── ai_engine/                 # Core AI & RAG package
│   ├── __init__.py            # Public API
│   ├── exceptions.py          # Custom exceptions with error codes
│   ├── utils.py               # Env loading, logging, text cleaning
│   ├── schemas.py             # Pydantic models (all structured I/O)
│   ├── llm_provider.py        # LLM abstraction: mock / OpenAI / Ollama / Groq / Gemini
│   ├── prompts.py             # Mandatory RAG instructions + task prompts
│   ├── document_processor.py  # PDF text extraction (PyMuPDF)
│   ├── text_chunker.py        # Clause-aware chunking with metadata
│   ├── embeddings.py          # Sentence Transformers wrapper
│   ├── vector_store.py        # ChromaDB persistent store, per-policy collections
│   ├── retriever.py           # Semantic + keyword-boosted retrieval
│   ├── rag_engine.py          # RAG QA, explanations, claim-relevant clauses
│   ├── policy_extractor.py    # Structured policy extraction + JSON persistence
│   ├── payout_calculator.py   # Deterministic claim payout estimation
│   ├── claim_readiness.py     # Claim Pre-Flight + readiness report
│   ├── policy_insights.py     # Coverage at a glance, room what-if, waiting timeline
│   ├── bill_extractor.py      # Hospital bill extraction (PDF/scan/photo)
│   ├── ocr.py                 # OCR for scanned documents
│   ├── language.py            # Multilingual detection & translation
│   ├── lending.py             # EMI, eligibility, prepayment, offer comparison
│   ├── fintech.py             # KYC, disputes, scam check, attachments
│   ├── accounts.py            # User registration, login, token auth
│   ├── assistant.py           # Saathi Chat assistant (RAG + MCP + guardrails)
│   ├── guardrails.py          # Input/output safety guardrails
│   ├── mcp_bridge.py          # MCP server bridge
│   └── linkcheck.py           # Link/scam analysis
├── backend/                   # FastAPI REST API
│   └── main.py                # All HTTP endpoints
├── frontend/                  # Streamlit UI
│   ├── app.py                 # Entry point
│   ├── home_ui.py             # Home page
│   ├── chat_ui.py             # Saathi Chat
│   ├── insurance_ui.py        # Insurance journey (5 tabs)
│   ├── lending_ui.py          # Loans journey
│   ├── fintech_ui.py          # Payments & KYC journey
│   ├── auth_ui.py             # Login/register
│   ├── common.py              # Shared UI helpers
│   ├── theme.py               # Paytm-style theme
│   ├── i18n.py                # Internationalization
│   └── locales/               # Translation JSON files (bn, gu, mr, ta, te)
├── mcp_server/                # Model Context Protocol server
│   └── server.py              # stdio MCP server (EMI, eligibility, scam check, etc.)
├── knowledge/                 # Markdown guides for RAG
│   ├── health_insurance_basics.md
│   ├── loans_basics.md
│   ├── payments_and_kyc.md
│   ├── claims_and_grievances.md
│   ├── saathi_app_guide.md
│   └── blocklist.txt
├── samples/                   # Demo files
│   ├── demo_health_policy.pdf
│   ├── make_demo_policy.py
│   ├── demo_hospital_bill.pdf # Fictional bill for extraction testing
│   └── make_demo_hospital_bill.py
├── scripts/                   # Utility scripts
│   ├── process_policy.py      # CLI for ingesting a policy PDF
│   └── build_translations.py  # Build translation JSON files
├── tests/                     # Pytest suite
├── data/                      # Runtime data (policies, vector_db, users)
├── app.py                     # Streamlit entry point
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template
├── pytest.ini                 # Pytest configuration
└── README.md                  # Project README
```

### Data Flow — Policy Ingestion

```
process_policy(pdf_path, policy_id)
    → extract_pdf_pages()          # PyMuPDF, page-by-page, scanned detection
    → chunk_pages()                # Clause-aware chunking with page/section metadata
    → embed_texts()                # Sentence Transformers (all-MiniLM-L6-v2)
    → vector_store.add_chunks()    # ChromaDB, per-policy isolation, dedup prevention
    → PolicyExtractor.extract()    # LLM structured extraction + Pydantic validation
    → save_policy_details()        # Persist as JSON
```

### Data Flow — Question Answering

```
ask_policy_question(policy_id, question, conversation_id)
    → detect language / translate to English
    → embed question
    → retriever.retrieve()         # Semantic + keyword boost, policy-isolated
    → build context with [Source N] markers
    → LLM generate (with mandatory anti-hallucination instructions)
    → parse citations, return RAGResponse
    → translate back if non-English (with number/citation safety check)
```

### Data Flow — Claim Payout Estimation

```
estimate_claim_payout(policy_id, bill, context)
    → load PolicyDetails (from JSON)
    → walk deductions in order:
        1. Bill total
        2. Non-payable items removed
        3. Room rent cap (+ proportionate deduction)
        4. ICU rent cap
        5. Treatment sub-limit
        6. Deductible
        7. Co-pay
        8. Sum insured cap
    → eligibility checks (exclusions, waiting periods)
    → return PayoutEstimate with transparent steps
```

---

## 3. Key Features

### 3.1 Policy PDF Processing
- Page-by-page text extraction with PyMuPDF
- Page numbers preserved for citations
- Scanned/image-based PDF detection (raises `ScannedPDFError`, never silently empty)
- Empty/corrupt PDF handling with clear errors

### 3.2 Structured Policy Extraction
Extracts and validates (via Pydantic):
- Insurer name, policy name, policy type
- Sum insured (with currency, source page, clause)
- Room rent limit, ICU room rent limit
- Co-pay percentage, deductibles
- Waiting periods (condition, duration, days)
- Exclusions, sub-limits
- Pre/post-hospitalization coverage
- Claim conditions, clause references
- Missing fields are `null`, listed in `unknown_fields`

### 3.3 RAG Question Answering
- Semantic retrieval with top-k relevant clauses
- Keyword boost for exact policy wording
- Weak evidence dropped (not returned to fill top_k)
- Grounded answers with structured citations `[Source N]`
- Conversational context per `conversation_id` (follow-up questions)
- Mandatory anti-hallucination instructions
- Honest insufficient-evidence response: *"I could not find enough information in the uploaded policy to answer this reliably."*

### 3.4 Clause Explanation
- Original wording + plain-language explanation
- English first, Hindi-ready design
- Can explain by clause number or pasted text

### 3.5 Claim Pre-Flight
- `get_claim_relevant_clauses()` — categorized clauses (coverage/exclusion/limit/waiting_period/copay/deductible)
- `generate_claim_readiness_report()` — full readiness report:
  - Policy Card (sum insured, room rent, co-pay, deductible, etc.)
  - Readiness Score (0-100, green/amber/red)
  - Cashless Eligibility (likely/uncertain/not_eligible)
  - Pre-Authorization requirements
  - Room Rent Impact (proportionate deduction warning)
  - Financial Estimate (out-of-pocket range)
  - Required documents (cashless / reimbursement)
  - Policy citations with source text

### 3.6 Hospital Bill Extraction & Payout
- Extract bills from PDF, scanned PDF, or photo (OCR via rapidocr-onnxruntime)
- Structured line items with categories (room_rent, icu_rent, doctor_fees, surgery, etc.)
- Deterministic payout estimation (pure Python, no LLM arithmetic)
- Transparent step-by-step deductions with clause references
- Room-choice what-if analysis (how much a cheaper room saves)
- Downloadable estimate summary

### 3.7 Lending Journey
- EMI calculator with amortization schedule
- Prepayment planner (extra monthly, yearly extra, lump sum)
- Eligibility check with "what would change the answer" levers
- Loan offer comparison with true-cost chart
- Document checklist with progress
- Agreement explainer and glossary

### 3.8 Fintech Journey
- KYC readiness assessment (format checks only, masked, nothing stored)
- Payment problem helper with escalation dates
- Ready-to-send complaint letters (app / bank nodal officer / RBI Ombudsman)
- Dispute tickets with evidence attachments (screenshots/PDFs, max 5 MB, 6 per ticket)
