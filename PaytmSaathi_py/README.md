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
| FastAPI backend (`backend/main.py`) | ✅ Implemented |
| Streamlit UI (`app.py`, `frontend/`) | ✅ Implemented (Paytm-style theme, English + हिन्दी) |
| OCR for scanned policies and photographed bills | ✅ Implemented (needs `rapidocr-onnxruntime`) |
| Hospital bill extraction + payout estimate | ✅ Implemented (`bill_extractor.py`, `payout_calculator.py`) |
| Hindi support | ✅ Implemented (`language.py`) |
| Lending journeys | ✅ Implemented (`lending.py`) |
| Fintech journeys (KYC, disputes, scam check) | ✅ Implemented (`fintech.py`) |

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

---

## Full app: Insurance + Lending + Fintech

```
ai_engine/   engine (insurance RAG + bills + payout, language, lending, fintech, ocr)
backend/     FastAPI REST API          ->  uvicorn backend.main:app --reload   (docs at /docs)
frontend/    Streamlit pages + theme   ->  streamlit run app.py
```

### Run it

```bash
pip install -r requirements.txt
copy .env.example .env            # then paste your free GROQ_API_KEY (see below)
streamlit run app.py              # UI on http://localhost:8501 - register, then log in
uvicorn backend.main:app --reload # API on http://localhost:8000/docs
pytest -q
```

### Connect a free AI model (Groq)

1. Sign up at https://console.groq.com (free, no credit card) and create an API key.
2. In `.env` set `LLM_PROVIDER=groq` and `GROQ_API_KEY=<your key>`.
3. Restart the app. The chip at the top changes from "Demo mode" to "AI: Groq · openai/gpt-oss-120b".

Free tiers are rate-limited per model. When the main model is busy the app automatically falls back to
`openai/gpt-oss-20b`, then `qwen/qwen3.8-27b` (`GROQ_FALLBACK_MODELS`), and only then shows "busy, try again in a minute".
Google Gemini also has a free tier: `LLM_PROVIDER=gemini` and `GEMINI_API_KEY`.
Without a key the app still runs in demo mode: calculators, checks and dispute help work fully;
policy Q&A and explanations say honestly that they cannot answer.

### Saathi Chat: RAG + MCP + guardrails

A general assistant (any topic) on the **Saathi Chat** page and at `POST /chat` (login required).

```
user message
  -> input guardrails : rate limit · clean text · redact OTP/PIN/CVV/Aadhaar/PAN/cards/passwords/API keys
                        · injection rules + Llama Prompt Guard 2 · blocked topics (fraud help, weapons, malware...)
                        + gpt-oss-safeguard policy classifier · crisis messages get helplines (Tele-MANAS 14416)
  -> RAG              : knowledge/*.md guides (+ the user's own policy, if selected) from ChromaDB,
                        screened for injected instructions, passed as delimited DATA with [K1]/[P1] ids
  -> Groq LLM         : openai/gpt-oss-120b with tool calling (fallback models on rate limits)
  -> MCP tools        : real MCP server (mcp_server/server.py, stdio) - EMI, prepayment, eligibility,
                        offer comparison, scam/link checks, grievance contacts, guide + policy search,
                        policy overview, waiting periods, my tickets. user_id/policy_id are injected by
                        the app and hidden from the model, so it can't read other users' data.
                        Tool results are redacted, screened and wrapped as DATA.
  -> output guardrails: strip HTML/scripts/images · remove secrets & PII · defang untrusted links
                        · rewrite "guaranteed approval" · catch system-prompt leaks (canary)
                        · medical/investment/legal disclaimers
```

The safety models degrade gracefully: if Prompt Guard or the safeguard model is unavailable, the
deterministic rules still run. Tests: `pytest tests/test_guardrails.py tests/test_assistant.py`
(offline) and `set SAATHI_LIVE=1 && pytest tests/test_live_integration.py` (real Groq + real MCP).
The MCP server works with any MCP client: `python -m mcp_server.server`.

To add knowledge, drop a Markdown file with `##` sections into `knowledge/`; it is re-indexed automatically.

### Payments evidence and scam check

* Tickets accept **screenshots/PDFs** (content-verified PNG/JPG/PDF, 5 MB, 6 per ticket) and **pasted SMS**.
  Text is read (OCR), OTPs/PINs are removed, and the UTR, amount and date found are offered to fill the ticket.
  The complaint letter lists the evidence, and an **evidence pack ZIP** (letter + files + SHA-256 index) can be downloaded.
* The **scam check** now also analyses **links** (look-alikes such as `paytrn.com`, brand-stuffed domains,
  shorteners, IP links, `@` tricks, punycode, throwaway TLDs, plus a blocklist in `knowledge/blocklist.txt` /
  `data/blocklist.txt`), **phone numbers** (personal mobiles posing as support, foreign numbers) and the
  **sender** (does the SMS header match the company the message claims to be?). It never claims a link is safe.

### Languages

English, हिन्दी, मराठी, বাংলা, தமிழ், తెలుగు, ગુજરાતી. Chat answers come in the user's language (or the
script they type in). UI text for the five newer languages comes from `frontend/locales/*.json`, built by
`python scripts/build_translations.py` (resumable). These are **machine translations - have native speakers
review them**. Sentences not in the catalog are translated on demand (numbers checked) and cached.

### Accounts

Registration asks only for basic details: full name, mobile, optional email, password and
preferred language. **No KYC at sign-up.** Users are stored in `data/users.json` with
PBKDF2-hashed passwords. 5 wrong passwords lock the account for 15 minutes. Each user sees only their
own policies and dispute tickets. The API offers `/auth/register` and `/auth/login`, which return a
token signed with `APP_SECRET`.

### What each journey does

| Journey | Features |
|---|---|
| **Insurance** | Upload policy (text or scanned) · **Coverage at a glance** (covered / not covered / watch-outs with clause and page) · **waiting-period countdown** from the policy start date · Ask Saathi with cited sources, follow-up chips and "explain simpler" · explain a clause by number **or pasted text** · **Claim Pre-Flight**: bill (PDF/scan/photo) → editable line items → step-by-step payout, **room-choice what-if** (how much a room within the limit saves), cashless/reimbursement **document checklist**, downloadable summary |
| **Loans** | EMI calculator · **prepayment planner** (interest and years saved, balance chart) · eligibility (illustrative rules) with **"what would change the answer"** levers · offer comparison with a **true-cost chart** and a warning when the lowest EMI isn't the cheapest · document checklist with progress · agreement explainer and glossary |
| **Payments & KYC** | Payment problem helper with **escalation dates**, a **ready-to-send complaint** (app / bank nodal officer / RBI Ombudsman) and **My tickets** · scam check that **highlights the risky words** · KYC readiness stepper (format checks only, masked, nothing stored) |

### Design rules kept throughout

- **Money maths is plain Python.** The LLM never calculates payouts, EMIs or eligibility.
- **Estimates, not promises.** Every result carries its assumptions and a disclaimer.
- **Hindi is safe.** Translations are accepted only if every number and `[Source N]` citation survives unchanged; otherwise the English text is shown. The mock LLM never fakes a translation.
- **Nothing invented.** Missing policy values stay empty and are listed as "could not check".
- **Lending rules and rates are illustrative defaults** (`LENDING_RULES` in `lending.py`), not any lender's policy. KYC limits and RBI timelines in `fintech.py` are indicative; confirm with the provider.

### OCR

`pip install rapidocr-onnxruntime` enables OCR for scanned policy PDFs and bill photos (English and digits). Hindi OCR needs Tesseract with the `hin` pack and `OCR_LANGS=eng+hin`.

---

## Money Clarity and Loan Pre-Flight (inside Saathi Chat)

Both features live **inside the Saathi Chat page** - no separate screens. Above the chat box there are two boxes:

* **📎 Add statement or document** - bank or credit-card statements (CSV, Excel `.xlsx`, PDF; several months at once)
  and loan documents (rejection letter, credit report: PDF, image or text). Optional: current balance and next salary date.
* **🏦 Loan details** - amount, tenure, quoted rate, income, existing EMIs, card dues, expenses, savings, credit-score
  range, employment, recent applications. "Save & check" stores the simulation (inputs + results) for the session.

Results appear as **cards inside the chat replies**: the Money Clarity report (alerts, upcoming dues, recurring charges,
spending chart, action plan, clickable transactions -> "Explain this transaction"), the Loan Pre-Flight card (EMI, total
interest, monthly leftover, before/after cash-flow chart, Safe / Tight / High Risk stress test, risks, documents,
explanations) and the rejection decoder (each reason with the exact quoted line).

### How it is built

| Layer | What | Where |
|---|---|---|
| Ingestion | CSV/TSV/Excel/HTML-xls/PDF (+OCR), layout-tolerant -> masked, normalised transactions with `source` (row / page+line) | `ai_engine/statements.py` |
| Analysis (deterministic) | categories, month-over-month (bill-to-bill for cards), recurring, duplicates, overlapping subscriptions, price rises, fees, unusual payments, upcoming dues, cash-flow projection, action plan, evidence + confidence | `ai_engine/money.py` |
| Loan maths (deterministic) | EMI = P x r x (1+r)^n / ((1+r)^n - 1), DTI, before/after cash flow, Safe/Tight/High Risk, rate/income/emergency stress tests, what-if, risk factors, rejection decoder | `ai_engine/lending.py` |
| Rules (editable) | categories, fee keywords, subscription groups, recurring tolerances; stress thresholds, rejection reasons, explanations | `knowledge/money_rules.json`, `knowledge/loan_preflight_rules.json` |
| Chat tools | session tools run in-process on the user's session data (`money_report`, `explain_transaction`, `list_transactions`, `my_loan_preflight`, `my_loan_what_if`, `decode_my_documents`); stateless tools over MCP (`loan_preflight_check`, `decode_rejection_text`) | `ai_engine/session_tools.py`, `ai_engine/assistant_tools.py` |
| Grounding | when a statement/loan/document is in the session and the question needs it, the app runs the tool itself before asking the model; the LLM only explains the numbers. Answers cite `[S#]` (statement rows) / `[D#]` (document lines); uncited answers are labelled general guidance | `ai_engine/assistant.py` |
| API | `POST /money/analyse`, `/money/explain-transaction`, `/loans/preflight`, `/loans/what-if`, `/loans/decode-rejection` (login required, nothing stored) | `backend/main.py` |

Privacy: statements and simulations stay in the user's session only (lost on logout/restart, never written to disk);
card/account numbers, Aadhaar, PAN and OTPs are masked when the file is read. Every money answer carries
"informational only, not financial advice"; every loan answer carries "This is not a loan approval, credit decision,
or guarantee of eligibility." Loan thresholds are **illustrative** - edit `knowledge/loan_preflight_rules.json`.

Sample data: `python samples/make_statement_samples.py` writes fictional files to `samples/statements/`.

**Different bank layouts.** The parser finds the transaction table anywhere near the top of the file and matches
columns by meaning, not position. That covers header names ("Txn Date" / "Value Dt" / "Narration" /
"Withdrawal Amt. (INR)" / "Dr/Cr"), extra columns, any delimiter, every workbook sheet, text or real Excel dates,
Rs/INR/₹ amounts with Cr/Dr markers, brackets or minus signs, card exports where payments are negative, wrapped
descriptions, repeated headers and summary blocks. Files with no header row get their columns guessed from the values
(with a warning). PDF tables are rebuilt from word positions. `python samples/make_statement_variants.py` writes 12
layouts of the same transactions to `samples/statements/variants/`, with `expected.json`:

| File | What is different |
|---|---|
| `hdfc_style.csv` | account details on top, `dd/mm/yy`, Withdrawal/Deposit columns, summary block (labels row + values row) at the end |
| `sbi_style.xlsx` | Summary sheet + Statement sheet, table starts at row 18, "1 Nov 2026" text dates, comma amounts, Total row |
| `icici_style.csv` | semicolon-separated, "(INR )" headers, serial-number column, header repeated mid-file, legend lines |
| `signed_amount.csv` | "Date & Time", signed `-₹18,000.00` / `+₹85,000.00` amounts, extra ID / Category / Mode columns |
| `card_export.csv` | card export: purchases positive, payments and refunds negative |
| `drcr_column.tsv` | tab-separated, Amount + Dr/Cr column, "Nov 01, 2026" dates, balance with " Cr" |
| `bank_table.pdf` | drawn table, right-aligned amounts, descriptions wrapped onto two lines, 2 pages with repeated header |
| `card_layout2.pdf` | card summary as a labels row with values below, "18 Oct 26" dates, "CR" markers, "Total Outstanding" |
| `excel_dates.xlsx` | real Excel dates and numbers, Description + Remarks, blank row, Total row |
| `messy.csv` | BOM + CRLF, blank lines, quoted commas, a line break inside a narration, `01.11.2026`, trailing-minus debits |
| `no_header.csv` | no header row at all |
| `html_export.xls` | ".xls" that is really an HTML table (what many banks' "Download Excel" gives) |

Not supported: the old binary Excel 97-2003 `.xls` format (you get a message asking you to save it as .xlsx or CSV).
Password-protected PDFs aren't supported either. Scanned PDFs need OCR installed.

### Demo script (3 journeys)

Start the app (`streamlit run app.py`), log in, open **Saathi Chat**.

1. **"I am planning my mother's hospitalisation; will this be cashless?"**
   Upload `samples/demo_health_policy.pdf` in Insurance -> My policy, then in Saathi Chat pick it under
   "Use my policy in this chat" and ask: *"I am planning my mother's cataract surgery in a network hospital. Will this be
   cashless, and what will the policy cover?"* -> cashless steps [K#] plus the ₹40,000/eye cataract sub-limit, room-rent
   cap and ICU limit cited to the policy [P#]. For a payout estimate use Insurance -> Claim Pre-Flight.
2. **"Why did my credit-card bill increase and what bills are due soon?"**
   📎 add `credit_card_oct_2026.pdf` and `credit_card_nov_2026.pdf` -> the Money Clarity card appears. Ask the question ->
   bill-to-bill change (Croma ₹12,999 purchase, late fee + finance charges + GST), card due ₹38,450 on 05 Dec,
   utilisation 38.5%, with evidence rows. Click a transaction -> "Explain this transaction". Also try
   `bank_statement_oct_nov_2026.csv` -> *"Will I face a cash crunch before my next salary?"* (Tight: rent + EMI due before
   the 7 Dec salary), duplicate Swiggy charge, Netflix price rise, two music subscriptions.
3. **"Can I afford a ₹5 lakh personal loan right now?"**
   🏦 Loan details: ₹5,00,000, 36 months, 13.5%, income ₹85,000, EMIs ₹12,500, card ₹5,000, expenses ₹40,000,
   savings ₹30,000 -> card: EMI ₹16,968, 41% of income to EMIs, **Tight**. Then ask *"What happens if I choose a 5-year
   tenure?"*, *"Can I afford this loan if my father is hospitalised next month?"*, and (after 📎 adding
   `loan_rejection_letter.pdf`) *"Why might my loan application be rejected?"* -> FOIR, enquiries and credit score, each
   with the quoted line [D#].

Tests: `pytest tests/test_money.py tests/test_loan_preflight.py tests/test_assistant.py` (offline).
Feature 1 (Claim Readiness Score) is pending its specification.
