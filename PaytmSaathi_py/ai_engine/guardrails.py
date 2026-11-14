"""Guardrails for the Saathi assistant (input, retrieved context, tools and output).

Layers, cheapest first:

1. **Deterministic rules - always on.** Length/control-character cleanup, redaction of
   credentials and identity numbers (Aadhaar, PAN, cards, OTP, PIN, passwords, API keys),
   prompt-injection patterns, hard-blocked topics, output sanitising (HTML, links, secrets,
   system-prompt leaks, guarantee language) and a per-user rate limit.
2. **Safety models - when a Groq key is connected.** Llama Prompt Guard scores prompt
   injection (user text, retrieved passages and tool results); gpt-oss-safeguard classifies
   the request against our written policy. If a model call fails, the deterministic result
   stands and the event is recorded - the assistant keeps working but says nothing risky
   passes unchecked by the rules.

Nothing here logs message content - only counts and categories.
"""

from __future__ import annotations

import json
import re
import threading
import time
import unicodedata
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

from .llm_provider import LLMProvider, OpenAICompatibleProvider
from .utils import logger

MAX_INPUT_CHARS = 2000
MAX_OUTPUT_CHARS = 6000
RATE_LIMIT_MESSAGES = 20
RATE_LIMIT_WINDOW_S = 300
INJECTION_THRESHOLD = 0.9  # Prompt Guard score above this blocks (benign ~0.0004, attacks >= 0.99)

PROMPT_GUARD_MODEL = "meta-llama/llama-prompt-guard-2-86m"
SAFEGUARD_MODEL = "openai/gpt-oss-safeguard-20b"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class GuardEvent:
    layer: str  # input | context | tool | output | rate_limit
    kind: str  # e.g. redacted_aadhaar, injection, blocked_topic, secret_removed
    action: str  # redacted | blocked | removed | rewritten | warned
    detail: str = ""


@dataclass
class InputVerdict:
    text: str  # cleaned + redacted text that is safe to send to the model
    blocked: bool = False
    refusal: str = ""  # user-facing message when blocked
    refusal_hi: str = ""
    events: list[GuardEvent] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PII / credential redaction
# ---------------------------------------------------------------------------
def _luhn_ok(digits: str) -> bool:
    total, flip = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if flip:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
        flip = not flip
    return total % 10 == 0


_AADHAAR = re.compile(r"(?<!\d)[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}(?!\d)")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
_OTP = re.compile(r"(?i)\b(otp|one[- ]time (?:password|pin)|verification code|security code)\b[^\d\n]{0,25}(\d{4,8})\b")
# "pin code 560001" is a postal code, not a PIN.
_PIN = re.compile(r"(?i)\b((?:(?:upi|atm|debit|credit|card)\s*)?pin(?!\s*-?\s*code)|cvv|cvc)\b[^\d\n]{0,15}(\d{3,6})\b")
_PASSWORD = re.compile(r"(?i)\b(password|passcode|passwd)\b\s*(?:is|:|=)\s*\S+")
_SECRET = re.compile(r"\b(gsk_[A-Za-z0-9]{20,}|xai-[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{30,}|"
                     r"ghp_[A-Za-z0-9]{30,}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})\b")
_PRIVATE_KEY = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)")


def redact_sensitive(text: str) -> tuple[str, list[GuardEvent]]:
    """Replace credentials and identity numbers with labelled placeholders."""
    events: list[GuardEvent] = []

    def sub(pattern, repl, label, flags=0, check=None):
        nonlocal text
        count = 0

        def _r(m):
            nonlocal count
            if check and not check(m):
                return m.group(0)
            count += 1
            return repl(m) if callable(repl) else repl

        text = pattern.sub(_r, text)
        if count:
            events.append(GuardEvent("input", f"redacted_{label}", "redacted", f"{count}x"))

    sub(_PRIVATE_KEY, "[PRIVATE KEY REMOVED]", "private_key")
    sub(_SECRET, "[API KEY REMOVED]", "api_key")
    sub(_PASSWORD, lambda m: f"{m.group(1)} [PASSWORD REMOVED]", "password")
    sub(_OTP, lambda m: f"{m.group(1)} [OTP REMOVED]", "otp")
    sub(_PIN, lambda m: f"{m.group(1)} [PIN REMOVED]", "pin")
    sub(_CARD, "[CARD NUMBER REMOVED]", "card",
        check=lambda m: 13 <= len(re.sub(r"\D", "", m.group(0))) <= 19 and _luhn_ok(re.sub(r"\D", "", m.group(0))))
    def _aadhaar_like(m) -> bool:
        raw = m.group(0)
        if re.search(r"[\s-]", raw):  # written like an Aadhaar number (4-4-4): always hide
            return True
        from .fintech import _verhoeff_valid  # lazy: fintech imports this module

        return _verhoeff_valid(raw)  # bare 12 digits (e.g. a UTR) only if the checksum matches

    sub(_AADHAAR, lambda m: "XXXX XXXX " + re.sub(r"\D", "", m.group(0))[-4:], "aadhaar", check=_aadhaar_like)
    sub(_PAN, lambda m: "XXXXX" + m.group(0)[5:], "pan")
    return text, events


# ---------------------------------------------------------------------------
# Prompt injection + blocked topics (deterministic)
# ---------------------------------------------------------------------------
_INJECTION_PATTERNS = [
    r"ignore (?:all |any |the )?(?:previous|prior|above|earlier) (?:instructions|rules|prompts?|messages)",
    r"disregard (?:all |any |the )?(?:previous|prior|above|your) (?:instructions|rules|guidelines)",
    r"forget (?:everything|all|your) (?:you|instructions|rules)",
    r"(?:reveal|show|print|repeat|output|tell me) (?:me )?(?:your|the) (?:system|hidden|initial|original) (?:prompt|instructions|rules|message)",
    r"what (?:is|are|were) your (?:system|hidden|initial) (?:prompt|instructions)",
    r"you are now (?:dan|an? unrestricted|an? unfiltered|in developer mode)",
    r"\b(?:dan|jailbreak|developer) mode\b",
    r"act as (?:an? )?(?:unrestricted|unfiltered|evil|jailbroken)",
    r"pretend (?:you have|there are) no (?:rules|restrictions|guidelines)",
    r"(?:bypass|disable|turn off|override) (?:your |the )?(?:safety|guardrails?|filters?|restrictions?)",
    r"<\s*/?\s*(?:system|assistant|tool)\s*>",
    r"\[\s*(?:system|inst)\s*\]",
    r"(?:^|\n)\s*(?:system|assistant)\s*:",
    r"(?:पिछले|ऊपर के) (?:सभी )?निर्देश(?:ों)? (?:को )?(?:अनदेखा|भूल)",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

# Hard-blocked requests. (category, regex, user message en, user message hi)
_BLOCKED_TOPICS: list[tuple[str, re.Pattern, str, str]] = [
    ("fraud_help", re.compile(
        r"(?:fake|forg(?:e|ed|ing)|fabricat\w*|doctor(?:ed)?|tamper\w*|inflat\w*)\W+(?:\w+\W+){0,3}"
        r"(?:bill|invoice|receipt|claim|policy|document|certificate|prescription|salary slip|bank statement|aadhaar|pan|kyc)"
        r"|(?:bypass|skip|avoid|get around|beat)\W+(?:\w+\W+){0,2}kyc"
        r"|launder\w*\W+(?:\w+\W+){0,2}money|money laundering how"
        r"|(?:steal|hack|phish|clone)\w*\W+(?:\w+\W+){0,3}(?:otp|upi|account|card|password|sim)"
        r"|how (?:to|do i) (?:scam|cheat|defraud)",
        re.IGNORECASE),
     "I can't help with creating fake documents, getting around KYC, or taking money or accounts from others. "
     "If you're facing a real claim or loan problem, I can help you do it the right way.",
     "मैं नकली दस्तावेज़ बनाने, केवाईसी से बचने या किसी के पैसे/खाते लेने में मदद नहीं कर सकता। "
     "अगर आपका असली क्लेम या लोन से जुड़ा मसला है, तो मैं सही तरीके से मदद कर सकता हूँ।"),
    ("weapons", re.compile(
        r"(?:make|build|synthesi[sz]e|assemble|manufacture)\W+(?:\w+\W+){0,3}(?:bomb|explosive|nerve agent|chemical weapon|bioweapon|ghost gun|silencer)",
        re.IGNORECASE),
     "I can't help with making weapons or explosives.",
     "मैं हथियार या विस्फोटक बनाने में मदद नहीं कर सकता।"),
    ("malware", re.compile(
        r"(?:write|create|build|code)\W+(?:\w+\W+){0,3}(?:ransomware|keylogger|malware|virus|botnet|phishing (?:page|site|kit))",
        re.IGNORECASE),
     "I can't help with creating malware or phishing tools.",
     "मैं मैलवेयर या फ़िशिंग टूल बनाने में मदद नहीं कर सकता।"),
    ("sexual_minors", re.compile(r"\b(?:child|minor|underage|teen)\w*\W+(?:\w+\W+){0,3}(?:sexual|porn|nude|explicit)", re.IGNORECASE),
     "I can't help with that.", "मैं इसमें मदद नहीं कर सकता।"),
]

_SELF_HARM = re.compile(
    r"(?:kill myself|end my life|want to die|suicid\w*|self[- ]harm|hurt myself|don'?t want to live|"
    r"आत्महत्या|जान दे दूँ|मर जाना चाहता|खुद को खत्म)", re.IGNORECASE)
SELF_HARM_REPLY = (
    "I'm really sorry you're going through this. You don't have to face it alone. "
    "In India you can call **Tele-MANAS at 14416** (free, 24x7, many languages) or **iCall at 9152987821**. "
    "If you are in immediate danger, call **112**. If money worries or a debt are part of this, I can also help "
    "you look at options - but please reach out to someone you trust right now.")
SELF_HARM_REPLY_HI = (
    "मुझे बहुत दुख है कि आप इस दौर से गुज़र रहे हैं। आप अकेले नहीं हैं। भारत में आप **टेली-मानस 14416** (मुफ़्त, 24x7, कई भाषाओं में) "
    "या **आईकॉल 9152987821** पर बात कर सकते हैं। तुरंत खतरे में हों तो **112** पर कॉल करें। अगर पैसों या कर्ज़ की चिंता भी इसकी वजह है, "
    "तो मैं विकल्प समझाने में मदद कर सकता हूँ - पर कृपया अभी किसी भरोसेमंद व्यक्ति से बात करें।")

_INJECTION_REFUSAL = ("That message looks like an attempt to change my rules or read my hidden instructions, so I've set it aside. "
                      "Ask me about your policy, loans, payments or anything else and I'll help.")
_INJECTION_REFUSAL_HI = ("यह संदेश मेरे नियम बदलने या छिपे निर्देश जानने की कोशिश जैसा लगता है, इसलिए मैंने इसे अलग रख दिया। "
                         "अपनी पॉलिसी, लोन, पेमेंट या किसी और विषय पर पूछिए, मैं मदद करूँगा।")


# ---------------------------------------------------------------------------
# Model-based checks (optional)
# ---------------------------------------------------------------------------
_SAFEGUARD_POLICY = (
    "You are a safety classifier for a finance and insurance help assistant used in India. "
    "Classify ONLY the user's message. Reply with JSON: "
    '{"violation":0|1,"category":"none|weapons|self_harm|fraud_help|hate|sexual_minors|malware|illegal_other","reason":"short"}. '
    "violation=1 when the user asks for help to: forge or fake bills, claims, KYC or identity documents; bypass KYC; launder money; "
    "steal or phish OTPs, accounts, cards or SIMs; make weapons or explosives; create malware; sexual content involving minors; "
    "or other serious crimes. NOT violations: questions about how fraud works in order to AVOID it, how to report fraud, general "
    "finance/insurance questions, and users mentioning or sharing their OWN OTP, Aadhaar, PAN, card or account details (the app hides "
    "those numbers; it is a safety mistake by the user, not a crime). Only requests to obtain OTPs or accounts of OTHER people are violations. "
    "When unsure, answer violation 0.")


def _parse_json(text: str) -> Optional[dict]:
    try:
        match = re.search(r"\{.*\}", text or "", re.DOTALL)
        return json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        return None


_PLACEHOLDER = re.compile(r"\[[A-Z ]*REMOVED\]|X{4,}(?:[ -]?X{4})*[ -]?\d{4}|X{5}\d{4}[A-Z]")


def _neutralise_placeholders(text: str) -> str:
    """Prompt Guard mistakes our own redaction markers ([OTP REMOVED], XXXX 0123) for an attack
    (score 0.99 on a harmless question). Swap them for plain words before scoring."""
    return _PLACEHOLDER.sub("a number", text or "")


class ModelGuard:
    """Wraps the Groq safety models. All methods fail open (return None) on any error."""

    def __init__(self, llm: Optional[LLMProvider]):
        self.llm = llm if isinstance(llm, OpenAICompatibleProvider) and llm.label == "Groq" else None

    @property
    def available(self) -> bool:
        return self.llm is not None

    def injection_score(self, text: str) -> Optional[float]:
        text = _neutralise_placeholders(text)
        if not self.llm or not text.strip():
            return None
        try:
            out = self.llm.complete_with_model(PROMPT_GUARD_MODEL, [{"role": "user", "content": text[:2000]}], max_tokens=8)
            return float(out.strip())
        except Exception as exc:  # network, rate limit, model unavailable, bad format
            logger.warning("Prompt Guard unavailable: %s", str(exc)[:80])
            return None

    def classify(self, text: str) -> Optional[dict]:
        text = _neutralise_placeholders(text)
        if not self.llm:
            return None
        try:
            out = self.llm.complete_with_model(
                SAFEGUARD_MODEL, [{"role": "system", "content": _SAFEGUARD_POLICY}, {"role": "user", "content": text[:2000]}],
                max_tokens=500, reasoning_effort="low")
            return _parse_json(out)
        except Exception as exc:
            logger.warning("Safeguard model unavailable: %s", str(exc)[:80])
            return None


# ---------------------------------------------------------------------------
# Input guardrails
# ---------------------------------------------------------------------------
def clean_text(text: str, limit: int = MAX_INPUT_CHARS) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")  # drop control/format chars
    text = re.sub(r"[ \t]{3,}", "  ", text)
    return text.strip()[:limit]


def check_input(text: str, model_guard: Optional[ModelGuard] = None) -> InputVerdict:
    """Clean, redact and screen a user message before it reaches the model."""
    original_len = len(text or "")
    cleaned = clean_text(text)
    events: list[GuardEvent] = []
    if original_len > MAX_INPUT_CHARS:
        events.append(GuardEvent("input", "truncated", "rewritten", f"{original_len}->{MAX_INPUT_CHARS}"))
    if not cleaned:
        return InputVerdict("", True, "Please type a question.", "कृपया अपना सवाल लिखें।", events)

    # Crisis support takes priority over everything else.
    if _SELF_HARM.search(cleaned):
        events.append(GuardEvent("input", "self_harm", "blocked", "support message shown"))
        return InputVerdict(cleaned, True, SELF_HARM_REPLY, SELF_HARM_REPLY_HI, events)

    redacted, red_events = redact_sensitive(cleaned)
    events += red_events

    if _INJECTION_RE.search(redacted):
        events.append(GuardEvent("input", "injection", "blocked", "pattern"))
        return InputVerdict(redacted, True, _INJECTION_REFUSAL, _INJECTION_REFUSAL_HI, events)
    for category, pattern, msg_en, msg_hi in _BLOCKED_TOPICS:
        if pattern.search(redacted):
            events.append(GuardEvent("input", category, "blocked", "rule"))
            return InputVerdict(redacted, True, msg_en, msg_hi, events)

    if model_guard and model_guard.available:
        with ThreadPoolExecutor(max_workers=2) as pool:
            score_f = pool.submit(model_guard.injection_score, redacted)
            class_f = pool.submit(model_guard.classify, redacted)
            score, verdict = score_f.result(), class_f.result()
        if score is not None and score >= INJECTION_THRESHOLD:
            events.append(GuardEvent("input", "injection", "blocked", f"prompt-guard {score:.2f}"))
            return InputVerdict(redacted, True, _INJECTION_REFUSAL, _INJECTION_REFUSAL_HI, events)
        if verdict and int(verdict.get("violation", 0)) == 1:
            category = str(verdict.get("category", "policy"))[:30]
            events.append(GuardEvent("input", category, "blocked", "safeguard model"))
            msg = next((m for c, _, m, _h in _BLOCKED_TOPICS if c == category), "I can't help with that request.")
            msg_hi = next((h for c, _, _m, h in _BLOCKED_TOPICS if c == category), "मैं इस अनुरोध में मदद नहीं कर सकता।")
            return InputVerdict(redacted, True, msg, msg_hi, events)
        if score is None and verdict is None:
            events.append(GuardEvent("input", "safety_models_unavailable", "warned", "rules only"))
    return InputVerdict(redacted, False, events=events)


# ---------------------------------------------------------------------------
# Untrusted context: retrieved passages and tool results
# ---------------------------------------------------------------------------
def screen_context(passages: list[str], model_guard: Optional[ModelGuard] = None) -> tuple[list[bool], list[GuardEvent]]:
    """Return (keep flags, events). Drops passages that carry injected instructions."""
    keep = [not _INJECTION_RE.search(p or "") for p in passages]
    events = [GuardEvent("context", "injection_in_document", "removed", "pattern") for ok in keep if not ok]
    if model_guard and model_guard.available:
        candidates = [i for i, ok in enumerate(keep) if ok]
        if candidates:
            with ThreadPoolExecutor(max_workers=4) as pool:
                scores = list(pool.map(lambda i: model_guard.injection_score(passages[i]), candidates))
            for i, score in zip(candidates, scores):
                if score is not None and score >= INJECTION_THRESHOLD:
                    keep[i] = False
                    events.append(GuardEvent("context", "injection_in_document", "removed", f"prompt-guard {score:.2f}"))
    return keep, events


def wrap_untrusted(label: str, text: str, limit: int = 2500) -> str:
    """Delimit retrieved/tool text so the model treats it strictly as data."""
    safe = clean_text(text, limit).replace("<<", "<").replace(">>", ">")
    return f"<<{label}>>\n{safe}\n<</{label}>>"


# ---------------------------------------------------------------------------
# Output guardrails
# ---------------------------------------------------------------------------
_SCRIPT_BLOCK = re.compile(r"<\s*(script|style|iframe|object|embed)\b[\s\S]*?(?:<\s*/\s*\1\s*>|$)", re.IGNORECASE)
_TAG = re.compile(r"<\s*/?\s*(?:script|style|iframe|object|embed|form|input|button|svg|math|link|meta|base)\b[^>]*>", re.IGNORECASE)
_ANY_HTML = re.compile(r"<\s*/?\s*[a-zA-Z][^>]*>")
_EVENT_ATTR = re.compile(r"\bon\w+\s*=", re.IGNORECASE)
_JS_URL = re.compile(r"(?:javascript|data|vbscript)\s*:", re.IGNORECASE)
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)[^)]*\)")
_BARE_URL = re.compile(r"https?://[^\s)>\]]+", re.IGNORECASE)

# Links the assistant may show as clickable. Everything else is shown inert (defanged).
TRUSTED_DOMAINS = (
    "rbi.org.in", "cms.rbi.org.in", "irdai.gov.in", "bimabharosa.irdai.gov.in", "cybercrime.gov.in",
    "sancharsaathi.gov.in", "npci.org.in", "uidai.gov.in", "incometax.gov.in", "paytm.com", "paytminsurance.co.in",
    "pib.gov.in", "india.gov.in", "sebi.gov.in", "tele-manas.in",
)
_GUARANTEE = re.compile(
    r"\b(?:your|the) claim (?:will|is going to) (?:definitely |certainly |surely )?(?:be )?(?:approved|paid|accepted|settled)\b"
    r"|\bguarantee[sd]? (?:approval|payout|claim|loan|that you)\b|\b100% (?:approved|guaranteed|sure|certain)\b"
    r"|\byou (?:will|are) (?:definitely|certainly|surely) (?:get|receive|qualify)\b", re.IGNORECASE)
_NEGATION = re.compile(r"\b(?:never|not|don'?t|do not|dont|avoid|shouldn'?t|should not|must not|no one should|nobody should|stop|without)\b"
                       r"|कभी नहीं|न करें|मत", re.IGNORECASE)
_SHARE_CREDENTIALS = re.compile(
    r"\b(?:share|send|tell|give|provide|enter|type)\b[^.\n]{0,40}\b(?:your |the )?(?:otp|upi pin|atm pin|cvv|password|aadhaar number|card number)\b",
    re.IGNORECASE)


def _domain_of(url: str) -> str:
    m = re.match(r"https?://([^/\s:?#]+)", url, re.IGNORECASE)
    host = m.group(1).lower() if m else ""
    return host[4:] if host.startswith("www.") else host


def _trusted(domain: str) -> bool:
    return any(domain == d or domain.endswith("." + d) for d in TRUSTED_DOMAINS)


def sanitize_links(text: str, events: list[GuardEvent]) -> str:
    """Remove images, keep trusted links, and make all other URLs inert (non-clickable)."""
    if _MD_IMAGE.search(text):
        text = _MD_IMAGE.sub("[image removed]", text)
        events.append(GuardEvent("output", "image_removed", "removed"))
    defanged = 0

    def md_link(m):
        nonlocal defanged
        label, url = m.group(1), m.group(2)
        if _trusted(_domain_of(url)):
            return f"[{label}]({url})"
        defanged += 1
        return f"{label} ({url.replace('http', 'hxxp', 1).replace('.', '[.]')})"

    text = _MD_LINK.sub(md_link, text)

    def bare(m):
        nonlocal defanged
        url = m.group(0)
        if _trusted(_domain_of(url)):
            return url
        defanged += 1
        return url.replace("http", "hxxp", 1).replace(".", "[.]")

    # Only touch bare URLs that are not already inside a kept markdown link target.
    parts = re.split(r"(\]\(https?://[^)\s]+\))", text)
    text = "".join(p if p.startswith("](") else _BARE_URL.sub(bare, p) for p in parts)
    if defanged:
        events.append(GuardEvent("output", "link_defanged", "rewritten", f"{defanged}x"))
    return text


def check_output(text: str, canary: str = "", disclaimers: Optional[list[str]] = None) -> tuple[str, list[GuardEvent]]:
    """Sanitise a model answer before it is shown. Returns (safe_text, events)."""
    events: list[GuardEvent] = []
    text = clean_text(text or "", MAX_OUTPUT_CHARS)

    if canary and canary in text:
        events.append(GuardEvent("output", "system_prompt_leak", "removed"))
        return ("I can't share my internal instructions, but I'm happy to help with your question.", events)

    text = re.sub(r"<\s*br\s*/?\s*>", "\n", text, flags=re.IGNORECASE)  # models use <br> inside table cells
    stripped = _SCRIPT_BLOCK.sub("", text)  # remove the script/style content, not just the tags
    stripped = _TAG.sub("", stripped)
    stripped = _ANY_HTML.sub("", stripped)
    if stripped != text:
        events.append(GuardEvent("output", "html_removed", "removed"))
        text = stripped
    if _EVENT_ATTR.search(text) or _JS_URL.search(text):
        text = _EVENT_ATTR.sub("", _JS_URL.sub("", text))
        events.append(GuardEvent("output", "script_vector_removed", "removed"))

    cleaned = _PRIVATE_KEY.sub("[removed]", _SECRET.sub("[removed]", text))
    if cleaned != text:
        events.append(GuardEvent("output", "secret_removed", "removed"))
        text = cleaned

    redacted, red = redact_sensitive(text)
    if red:
        events += [GuardEvent("output", e.kind, "redacted", e.detail) for e in red]
        text = redacted

    text = sanitize_links(text, events)

    def _asks_for_credentials(m) -> str:
        before = text[max(0, m.start() - 40): m.start()].lower()
        if _NEGATION.search(before):  # "never share your OTP" is good advice, keep it
            return m.group(0)
        return "[never share this with anyone]"

    rewritten = _SHARE_CREDENTIALS.sub(_asks_for_credentials, text)
    if rewritten != text:
        text = rewritten
        events.append(GuardEvent("output", "credential_request", "rewritten"))
    if _GUARANTEE.search(text):
        events.append(GuardEvent("output", "guarantee_language", "rewritten"))
        text = _GUARANTEE.sub("the outcome depends on the insurer or lender's final decision", text)

    for note in disclaimers or []:
        text = f"{text}\n\n_{note}_"
    return text.strip(), events


# ---------------------------------------------------------------------------
# Sensitive-topic disclaimers (added to the answer, not model-generated)
# ---------------------------------------------------------------------------
_TOPIC_DISCLAIMERS = [
    (re.compile(r"\b(?:symptom|diagnos\w*|medicine|dosage|treatment|disease|cancer|diabetes|pregnan\w*|surgery)\b", re.I),
     "I'm not a doctor. For medical decisions please consult a qualified doctor.",
     "मैं डॉक्टर नहीं हूँ। चिकित्सा संबंधी फ़ैसलों के लिए कृपया योग्य डॉक्टर से सलाह लें।"),
    (re.compile(r"\b(?:invest\w*|stock|mutual fund|sip|crypto|bitcoin|trading|shares?)\b", re.I),
     "This is general information, not investment advice. Past returns don't guarantee future results.",
     "यह सामान्य जानकारी है, निवेश सलाह नहीं। पिछला रिटर्न भविष्य के रिटर्न की गारंटी नहीं है।"),
    (re.compile(r"\b(?:lawyer|legal|court|sue|lawsuit|fir|police case|tax(?:es)?|itr|gst)\b", re.I),
     "This is general information, not legal or tax advice. Please consult a qualified professional for your case.",
     "यह सामान्य जानकारी है, कानूनी या कर सलाह नहीं। अपने मामले के लिए कृपया योग्य विशेषज्ञ से सलाह लें।"),
]


def disclaimers_for(user_text: str, language: str = "en") -> list[str]:
    return [(hi if language == "hi" else en) for pattern, en, hi in _TOPIC_DISCLAIMERS if pattern.search(user_text or "")]


# ---------------------------------------------------------------------------
# Rate limit
# ---------------------------------------------------------------------------
class RateLimiter:
    def __init__(self, limit: int = RATE_LIMIT_MESSAGES, window_s: int = RATE_LIMIT_WINDOW_S):
        self.limit, self.window_s = limit, window_s
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, now: Optional[float] = None) -> tuple[bool, int]:
        """Return (allowed, seconds_until_next_slot)."""
        now = time.time() if now is None else now
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > self.window_s:
                hits.popleft()
            if len(hits) >= self.limit:
                return False, int(self.window_s - (now - hits[0])) + 1
            hits.append(now)
            return True, 0


RATE_LIMITER = RateLimiter()
