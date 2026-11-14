"""Hindi (and English) language support.

* Detects Hindi (Devanagari) questions so they can be answered in Hindi.
* Translates answers/explanations with the configured LLM, under strict rules:
  numbers, rupee amounts, clause numbers and ``[Source N]`` markers must survive
  translation unchanged, otherwise the English original is kept.
* Works offline: with the mock provider (or on LLM failure) a curated insurance /
  lending / payments glossary and fixed phrases are used, so the app never shows
  a mistranslated or fabricated sentence.

The embedding model is English-only, so a Hindi question is translated to English
*for retrieval only*; the original wording is preserved for the user.
"""

from __future__ import annotations

import re
from typing import Optional

from .exceptions import LLMProviderError
from .llm_provider import LLMProvider, MockLLMProvider
from .utils import logger

SUPPORTED_LANGUAGES = {"en": "English", "hi": "हिन्दी"}

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

_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_SOURCE = re.compile(r"\[Source \d+\]")
_DIGIT_MAP = str.maketrans("०१२३४५६७८९", "0123456789")


def contains_devanagari(text: str) -> bool:
    return bool(text and _DEVANAGARI.search(text))


def detect_language(text: str) -> str:
    """'hi' when the text is mostly Devanagari, else 'en'."""
    if not text:
        return "en"
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "en"
    deva = sum(1 for c in letters if _DEVANAGARI.match(c))
    return "hi" if deva / len(letters) >= 0.3 else "en"


def normalize_language(language: Optional[str]) -> str:
    if not language:
        return "en"
    lowered = language.strip().lower()
    if lowered in ("hi", "hindi", "हिन्दी", "हिंदी"):
        return "hi"
    return "en"


def _normalized_numbers(text: str) -> list[str]:
    text = text.translate(_DIGIT_MAP)
    return sorted(n.replace(",", "") for n in _NUMBER.findall(text))


def _preserves_facts(original: str, translated: str) -> bool:
    """Numbers and [Source N] markers must be identical after translation."""
    if _normalized_numbers(original) != _normalized_numbers(translated):
        return False
    return sorted(_SOURCE.findall(original)) == sorted(_SOURCE.findall(translated))


def glossary_translate(text: str, target: str = "hi") -> str:
    """Replace known terms only (offline). Unknown words are left untouched."""
    table = GLOSSARY_EN_HI if target == "hi" else GLOSSARY_HI_EN
    result = text
    for term in sorted(table, key=len, reverse=True):
        if target == "hi":
            result = re.sub(
                rf"\b{re.escape(term)}\b", table[term], result, flags=re.IGNORECASE
            )
        else:
            result = result.replace(term, table[term])
    return result


def _is_mock(llm: Optional[LLMProvider]) -> bool:
    return llm is None or isinstance(llm, MockLLMProvider)


def translate_text(
    text: str, target: str, llm: Optional[LLMProvider] = None
) -> tuple[str, bool]:
    """Translate ``text`` to ``target`` ('hi' or 'en').

    Returns (translated_text, was_translated). On any failure or fact mismatch the
    original text is returned with was_translated=False.
    """
    target = normalize_language(target)
    if not text or not text.strip():
        return text, False
    if target == "hi":
        if contains_devanagari(text):
            return text, False  # already Hindi
        if text.strip() == INSUFFICIENT_EVIDENCE_EN:
            return INSUFFICIENT_EVIDENCE_HI, True
    elif not contains_devanagari(text):
        return text, False  # already English

    if _is_mock(llm):
        return text, False  # never let a stub LLM "translate"

    system = (
        "You are a careful professional translator for Indian financial services. "
        "Translate faithfully. Never add, remove or change any information."
    )
    if target == "hi":
        prompt = (
            "Translate the text below into simple, natural Hindi (Devanagari script). "
            "Rules: keep all numbers, rupee amounts (₹), percentages, dates, clause "
            "numbers and any [Source N] markers EXACTLY as written (use Western digits); "
            "keep product/brand names like Paytm in English; do not add explanations.\n\n"
            f"TEXT:\n{text}\n\nReturn only the Hindi translation."
        )
    else:
        prompt = (
            "Translate the text below into plain English. Keep all numbers, amounts "
            "and names exactly. Return only the translation.\n\n"
            f"TEXT:\n{text}"
        )
    try:
        translated = llm.generate(prompt, system_prompt=system, temperature=0.0, max_tokens=1500)  # type: ignore[union-attr]
    except LLMProviderError as exc:
        logger.warning("Translation failed, keeping original: %s", exc)
        return text, False
    translated = (translated or "").strip()
    if not translated or not _preserves_facts(text, translated):
        logger.warning("Translation altered numbers/citations; keeping original text.")
        return text, False
    if target == "hi" and not contains_devanagari(translated):
        return text, False
    return translated, True


def question_to_english(question: str, llm: Optional[LLMProvider] = None) -> str:
    """Translate a Hindi question to English for retrieval. English passes through."""
    if not contains_devanagari(question):
        return question
    if not _is_mock(llm):
        translated, ok = translate_text(question, "en", llm)
        if ok:
            return translated
    english = glossary_translate(question, "en")
    return english
