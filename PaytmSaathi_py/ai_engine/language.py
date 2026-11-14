"""Language support: English plus Hindi, Marathi, Bengali, Tamil, Telugu and Gujarati.

* Detects the script of a message (Devanagari -> Hindi, Bengali, Gujarati, Tamil, Telugu).
  Marathi shares Devanagari with Hindi, so it is only used when the user picks it.
* Translates text with the configured LLM under strict rules: numbers, rupee amounts, clause
  numbers and ``[Source N]`` markers must survive unchanged and the result must be in the target
  script, otherwise the original text is kept. The offline mock model never fakes a translation.
* Hindi has a curated glossary and fixed phrases that work with no AI model at all.

The embedding model is English-only, so a non-English question is translated to English *for
retrieval only*; the user's original wording is always preserved.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

from .exceptions import LLMProviderError
from .llm_provider import LLMProvider, MockLLMProvider
from .utils import logger

# code -> native name (shown in the language picker)
SUPPORTED_LANGUAGES = {
    "en": "English", "hi": "हिन्दी", "mr": "मराठी", "bn": "বাংলা", "ta": "தமிழ்", "te": "తెలుగు", "gu": "ગુજરાતી",
}
LANGUAGE_NAMES_EN = {
    "en": "English", "hi": "Hindi", "mr": "Marathi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "gu": "Gujarati",
}
# Unicode block of each language's script (Marathi and Hindi share Devanagari).
_SCRIPTS = {
    "hi": (0x0900, 0x097F), "mr": (0x0900, 0x097F), "bn": (0x0980, 0x09FF),
    "gu": (0x0A80, 0x0AFF), "ta": (0x0B80, 0x0BFF), "te": (0x0C00, 0x0C7F),
}
_SCRIPT_TO_CODE = {(0x0900, 0x097F): "hi", (0x0980, 0x09FF): "bn", (0x0A80, 0x0AFF): "gu", (0x0B80, 0x0BFF): "ta", (0x0C00, 0x0C7F): "te"}

INSUFFICIENT_EVIDENCE_EN = (
    "I could not find enough information in the uploaded policy to answer this reliably."
)
INSUFFICIENT_EVIDENCE_HI = (
    "मुझे अपलोड की गई पॉलिसी में इसका भरोसेमंद उत्तर देने के लिए पर्याप्त जानकारी नहीं मिली।"
)

# English -> Hindi glossary (insurance, lending, payments). Longest phrases first at use time.
GLOSSARY_EN_HI: dict[str, str] = {
    "sum insured": "बीमित राशि",
    "room rent": "कमरे का किराया",
    "icu": "आईसीयू",
    "co-pay": "सह-भुगतान",
    "copay": "सह-भुगतान",
    "deductible": "कटौती योग्य राशि",
    "waiting period": "प्रतीक्षा अवधि",
    "pre-existing disease": "पहले से मौजूद बीमारी",
    "pre-existing": "पहले से मौजूद",
    "exclusion": "अपवर्जन (कवर नहीं)",
    "exclusions": "अपवर्जन (कवर नहीं)",
    "sub-limit": "उप-सीमा",
    "claim": "क्लेम",
    "premium": "प्रीमियम",
    "policy": "पॉलिसी",
    "policyholder": "पॉलिसीधारक",
    "hospitalization": "अस्पताल में भर्ती",
    "pre-hospitalization": "भर्ती से पहले का खर्च",
    "post-hospitalization": "छुट्टी के बाद का खर्च",
    "network hospital": "नेटवर्क अस्पताल",
    "cashless": "कैशलेस",
    "reimbursement": "प्रतिपूर्ति (रीइम्बर्समेंट)",
    "insurer": "बीमा कंपनी",
    "estimated payout": "अनुमानित भुगतान",
    "you pay": "आपको देना होगा",
    "non-payable": "भुगतान योग्य नहीं",
    "loan": "लोन",
    "emi": "ईएमआई",
    "interest": "ब्याज",
    "principal": "मूलधन",
    "tenure": "अवधि",
    "processing fee": "प्रोसेसिंग शुल्क",
    "credit score": "क्रेडिट स्कोर",
    "foreclosure": "समय से पहले लोन बंद करना",
    "prepayment": "समय से पहले भुगतान",
    "kyc": "केवाईसी",
    "upi": "यूपीआई",
    "refund": "रिफंड",
    "transaction": "लेन-देन",
    "dispute": "शिकायत",
    "otp": "ओटीपी",
    "pin": "पिन",
}

# Hindi -> English for retrieval-only translation (offline fallback).
GLOSSARY_HI_EN: dict[str, str] = {
    "बीमित राशि": "sum insured",
    "कमरे का किराया": "room rent",
    "रूम रेंट": "room rent",
    "आईसीयू": "ICU",
    "सह-भुगतान": "co-pay",
    "को-पे": "co-pay",
    "कटौती": "deductible",
    "प्रतीक्षा अवधि": "waiting period",
    "वेटिंग पीरियड": "waiting period",
    "पहले से मौजूद बीमारी": "pre-existing disease",
    "अपवर्जन": "exclusion",
    "कवर नहीं": "exclusion not covered",
    "क्लेम": "claim",
    "प्रीमियम": "premium",
    "पॉलिसी": "policy",
    "अस्पताल": "hospital",
    "भर्ती": "hospitalization",
    "कैशलेस": "cashless",
    "मोतियाबिंद": "cataract",
    "ऑपरेशन": "surgery",
    "सर्जरी": "surgery",
    "डायलिसिस": "dialysis",
    "मातृत्व": "maternity",
    "प्रसव": "maternity delivery",
    "दांत": "dental",
    "डेंटल": "dental",
    "कैंसर": "cancer",
    "मधुमेह": "diabetes",
    "डायबिटीज": "diabetes",
    "हृदय": "heart cardiac",
    "कवर": "coverage covered",
    "कितना": "how much",
    "क्या": "what is",
    "सीमा": "limit",
    "लिमिट": "limit",
}

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_SOURCE = re.compile(r"\[Source \d+\]|\[[KP]\d+\]")


# ---------------------------------------------------------------------------
# Script detection and normalisation
# ---------------------------------------------------------------------------
def _script_fraction(text: str, block: tuple[int, int]) -> float:
    letters = [c for c in (text or "") if c.isalpha() or unicodedata.category(c).startswith("M")]
    if not letters:
        return 0.0
    return sum(1 for c in letters if block[0] <= ord(c) <= block[1]) / len(letters)


def contains_devanagari(text: str) -> bool:
    return any(0x0900 <= ord(c) <= 0x097F for c in (text or ""))


def has_indic_script(text: str) -> bool:
    return any(any(lo <= ord(c) <= hi for lo, hi in set(_SCRIPTS.values())) for c in (text or ""))


def in_script(text: str, code: str) -> bool:
    """True when the text is mostly written in ``code``'s script (always True for English)."""
    if code not in _SCRIPTS:
        return True
    return _script_fraction(text, _SCRIPTS[code]) >= 0.3


def detect_language(text: str) -> str:
    """Language code from the dominant script; Latin text is 'en'. Devanagari is reported as 'hi'."""
    for block, code in _SCRIPT_TO_CODE.items():
        if _script_fraction(text, block) >= 0.3:
            return code
    return "en"


def normalize_language(language: Optional[str]) -> str:
    """Accept a code ('hi'), an English name ('Hindi') or a native name ('हिन्दी')."""
    if not language:
        return "en"
    lowered = language.strip().lower()
    if lowered in SUPPORTED_LANGUAGES:
        return lowered
    for code, native in SUPPORTED_LANGUAGES.items():
        if lowered in (native.lower(), LANGUAGE_NAMES_EN[code].lower()):
            return code
    if lowered in ("हिंदी",):
        return "hi"
    return "en"


def _ascii_digits(text: str) -> str:
    """Convert digits of any script (०१२, ০১২, ௦௧௨ ...) to 0-9."""
    return "".join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in text)


def _normalized_numbers(text: str) -> list[str]:
    return sorted(n.replace(",", "") for n in _NUMBER.findall(_ascii_digits(text)))


def _preserves_facts(original: str, translated: str) -> bool:
    """Numbers and citation markers must be identical after translation."""
    if _normalized_numbers(original) != _normalized_numbers(translated):
        return False
    return sorted(_SOURCE.findall(original)) == sorted(_SOURCE.findall(translated))


def glossary_translate(text: str, target: str = "hi") -> str:
    """Replace known terms only (offline, Hindi<->English). Unknown words are left untouched."""
    table = GLOSSARY_EN_HI if target == "hi" else GLOSSARY_HI_EN
    result = text
    for term in sorted(table, key=len, reverse=True):
        if target == "hi":
            result = re.sub(rf"\b{re.escape(term)}\b", table[term], result, flags=re.IGNORECASE)
        else:
            result = result.replace(term, table[term])
    return result


def _is_mock(llm: Optional[LLMProvider]) -> bool:
    return llm is None or isinstance(llm, MockLLMProvider)


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------
def translate_text(
    text: str, target: str, llm: Optional[LLMProvider] = None
) -> tuple[str, bool]:
    """Translate ``text`` into ``target`` (any supported code, or 'en').

    Returns (translated_text, was_translated). On any failure, fact mismatch or wrong script the
    original text is returned with was_translated=False.
    """
    target = normalize_language(target)
    if not text or not text.strip():
        return text, False
    if target == "en":
        if not has_indic_script(text):
            return text, False  # already English
    else:
        if in_script(text, target) and has_indic_script(text) and target != "mr":
            return text, False  # already in the target language
        if target == "hi" and text.strip() == INSUFFICIENT_EVIDENCE_EN:
            return INSUFFICIENT_EVIDENCE_HI, True

    if _is_mock(llm):
        return text, False  # never let a stub LLM "translate"

    system = (
        "You are a careful professional translator for Indian financial services. "
        "Translate faithfully. Never add, remove or change any information."
    )
    name = LANGUAGE_NAMES_EN[target]
    if target == "en":
        prompt = (
            "Translate the text below into plain English. Keep all numbers, amounts "
            "and names exactly. Return only the translation.\n\n"
            f"TEXT:\n{text}"
        )
    else:
        prompt = (
            f"Translate the text below into simple, natural {name} ({SUPPORTED_LANGUAGES[target]} script). "
            "Rules: keep all numbers, rupee amounts (₹), percentages, dates, clause numbers and any "
            "[Source N] / [K1] / [P1] markers EXACTLY as written (use Western digits 0-9); keep brand "
            "names like Paytm, UPI, KYC, EMI, OTP in English letters; do not add explanations.\n\n"
            f"TEXT:\n{text}\n\nReturn only the {name} translation."
        )
    try:
        translated = llm.generate(prompt, system_prompt=system, temperature=0.0, max_tokens=2500)  # type: ignore[union-attr]
    except LLMProviderError as exc:
        logger.warning("Translation failed, keeping original: %s", exc)
        return text, False
    translated = (translated or "").strip()
    if not translated or not _preserves_facts(text, translated):
        logger.warning("Translation altered numbers/citations; keeping original text.")
        return text, False
    if target != "en" and not in_script(translated, target):
        return text, False
    return translated, True


def question_to_english(question: str, llm: Optional[LLMProvider] = None) -> str:
    """Translate a non-English question to English for retrieval. English passes through."""
    if not has_indic_script(question):
        return question
    if not _is_mock(llm):
        translated, ok = translate_text(question, "en", llm)
        if ok:
            return translated
    return glossary_translate(question, "en")  # offline Hindi fallback
