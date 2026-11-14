"""UI translation for languages beyond English and Hindi.

English and Hindi text is written inline in the pages (``L(en, hi)``). For Marathi, Bengali, Tamil,
Telugu and Gujarati, text is looked up in order:

1. ``frontend/locales/<code>.json`` - catalog built by ``scripts/build_translations.py`` (machine
   translated; have a native speaker review before production use).
2. ``data/i18n_cache/<code>.json`` - sentences translated on demand earlier (for dynamic text that
   contains numbers). At most ``LIVE_BUDGET`` new sentences are translated per page run so a page
   never stalls; anything not translated yet is shown in English and filled in on later runs.
3. English.

Translations must keep every number and citation marker (checked in ``ai_engine.language``).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

import streamlit as st

from ai_engine.language import normalize_language, translate_text
from ai_engine.llm_provider import get_llm_provider
from ai_engine.utils import CHROMA_PERSIST_DIR, ensure_dir

LOCALES_DIR = Path(__file__).resolve().parent / "locales"
CACHE_DIR = Path(CHROMA_PERSIST_DIR).parent / "i18n_cache"
LIVE_BUDGET = 4

_LOCK = threading.Lock()
_CATALOGS: dict[str, dict[str, str]] = {}
_CACHES: dict[str, dict[str, str]] = {}


def _load(path: Path) -> dict[str, str]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (json.JSONDecodeError, OSError):
        return {}


def catalog(code: str) -> dict[str, str]:
    if code not in _CATALOGS:
        _CATALOGS[code] = _load(LOCALES_DIR / f"{code}.json")
    return _CATALOGS[code]


def _cache(code: str) -> dict[str, str]:
    if code not in _CACHES:
        _CACHES[code] = _load(CACHE_DIR / f"{code}.json")
    return _CACHES[code]


def _save_cache(code: str) -> None:
    try:
        ensure_dir(CACHE_DIR)
        (CACHE_DIR / f"{code}.json").write_text(json.dumps(_CACHES[code], ensure_ascii=False, indent=0), encoding="utf-8")
    except OSError:
        pass


def reset_budget() -> None:
    """Call once at the top of each page run."""
    st.session_state["_i18n_budget"] = LIVE_BUDGET


def tr(en: str, code: str, live: bool = True) -> str:
    """Translate an English UI string into ``code`` (falls back to English)."""
    code = normalize_language(code)
    if code in ("en", "hi") or not en or not en.strip():
        return en
    hit = catalog(code).get(en) or _cache(code).get(en)
    if hit:
        return hit
    if not live or st.session_state.get("_i18n_budget", LIVE_BUDGET) <= 0:
        return en
    st.session_state["_i18n_budget"] = st.session_state.get("_i18n_budget", LIVE_BUDGET) - 1
    try:
        translated, ok = translate_text(en, code, get_llm_provider())
    except Exception:
        return en
    if ok:
        with _LOCK:
            _cache(code)[en] = translated
            _save_cache(code)
        return translated
    return en


def tr_many(texts: list[str], code: str) -> list[str]:
    return [tr(t, code) for t in texts]
