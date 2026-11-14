"""Build UI translation catalogs (frontend/locales/<code>.json) with the configured LLM.

    python scripts/build_translations.py --dry-run               # just count the strings
    python scripts/build_translations.py --langs mr bn ta te gu  # translate (resumable)

The catalogs are MACHINE translations. Have a native speaker review them before production use;
fix a string by editing the JSON file. Re-running only translates strings that are missing.
Every translated string must keep the original numbers and be written in the right script.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_engine.language import (  # noqa: E402
    LANGUAGE_NAMES_EN, SUPPORTED_LANGUAGES, _preserves_facts, in_script,
)
from ai_engine.llm_provider import get_llm_provider  # noqa: E402

LOCALES = ROOT / "frontend" / "locales"
KEEP = "Paytm, Saathi, UPI, KYC, EMI, OTP, PIN, PAN, Aadhaar, IRDAI, RBI, NPCI, UTR, FAQ, ZIP, PDF, PNG, JPG, SMS, WhatsApp, AI, MCP, IFSC, ICU, GST, ITR"


# ---------------------------------------------------------------------------
# Harvest English strings
# ---------------------------------------------------------------------------
def _fstring_children(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            for child in ast.walk(node):
                if isinstance(child, ast.Constant):
                    ids.add(id(child))
    return ids


def harvest_ui() -> set[str]:
    """First argument of every L("English", "हिन्दी") call in the pages."""
    out: set[str] = set()
    files = [ROOT / "app.py", *sorted((ROOT / "frontend").glob("*.py"))]
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "L"
                    and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
                out.add(node.args[0].value)
    return out


def _looks_like_prose(s: str) -> bool:
    if any("\u0900" <= ch <= "\u097f" for ch in s) or "|" in s:  # already Hindi, or a regex alternation
        return False
    return (len(s) >= 18 and " " in s and not re.search(r"\\[dsbwDSBW(]|\(\?|\[\^|\{\d|\.\*|\\\.", s)
            and not s.startswith(("http", "http://", "SELECT")) and re.search(r"[A-Za-z]{3}", s) is not None)


def harvest_engine() -> set[str]:
    """Static user-facing sentences in the engine modules (dispute, scam, link, KYC, lending, insights)."""
    out: set[str] = set()
    for name in ["fintech.py", "linkcheck.py", "lending.py", "policy_insights.py", "guardrails.py"]:
        path = ROOT / "ai_engine" / name
        tree = ast.parse(path.read_text(encoding="utf-8"))
        skip = _fstring_children(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
                s = node.value.strip()
                if _looks_like_prose(s) and "{" not in s and len(s) < 400:
                    out.add(s)
    return out


def harvest() -> list[str]:
    return sorted(harvest_ui() | harvest_engine())


# ---------------------------------------------------------------------------
# Translate
# ---------------------------------------------------------------------------
def _parse_array(text: str):
    match = re.search(r"\[.*\]", text or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, list) and all(isinstance(x, str) for x in data) else None


def translate_batch(llm, strings: list[str], code: str) -> dict[str, str]:
    name, native = LANGUAGE_NAMES_EN[code], SUPPORTED_LANGUAGES[code]
    prompt = (
        f"Translate each English UI string into {name} ({native} script). Output ONLY a JSON array of strings with exactly "
        f"{len(strings)} items, in the same order.\n"
        "Rules: natural, simple and polite wording; keep all numbers, ₹, %, emojis, markdown characters (** ` _) and any {placeholders} "
        f"exactly; keep these terms in English letters: {KEEP}. Do not add or remove information.\n\n"
        f"INPUT:\n{json.dumps(strings, ensure_ascii=False)}"
    )
    reply = llm.generate(prompt, system_prompt="You are a careful professional translator for Indian financial apps.",
                         temperature=0.1, max_tokens=7000)
    arr = _parse_array(reply)
    good: dict[str, str] = {}
    if arr is None or len(arr) != len(strings):
        return good
    for src, dst in zip(strings, arr):
        dst = dst.strip()
        if dst and _preserves_facts(src, dst) and in_script(dst, code):
            good[src] = dst
    return good


def build(codes: list[str], batch: int, limit: int | None, pause: float) -> None:
    llm = get_llm_provider()
    if llm.label == "Demo mode":
        sys.exit("No AI key configured. Set GROQ_API_KEY in .env first.")
    strings = harvest()
    LOCALES.mkdir(parents=True, exist_ok=True)
    for code in codes:
        path = LOCALES / f"{code}.json"
        done = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        todo = [s for s in strings if s not in done][: limit or None]
        print(f"[{code}] {len(done)} done, {len(todo)} to translate", flush=True)
        failures = 0
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            try:
                got = translate_batch(llm, chunk, code)
            except Exception as exc:  # rate limits etc.: save progress and move on
                print(f"[{code}] batch {i // batch + 1} failed: {str(exc)[:100]}", flush=True)
                time.sleep(10)
                continue
            if len(got) < len(chunk):  # retry the leftovers in smaller pieces
                rest = [s for s in chunk if s not in got]
                for j in range(0, len(rest), 5):
                    try:
                        got.update(translate_batch(llm, rest[j:j + 5], code))
                    except Exception:
                        pass
            done.update(got)
            failures += len(chunk) - len(got)
            path.write_text(json.dumps(done, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
            print(f"[{code}] {min(i + batch, len(todo))}/{len(todo)} (skipped so far: {failures})", flush=True)
            time.sleep(pause)
        print(f"[{code}] finished: {len(done)} strings in {path.name}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["mr", "bn", "ta", "te", "gu"])
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None, help="translate at most N missing strings per language")
    ap.add_argument("--pause", type=float, default=1.5, help="seconds between batches (rate limits)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        ui, eng = harvest_ui(), harvest_engine()
        print(f"UI strings: {len(ui)} | engine strings: {len(eng)} | total unique: {len(ui | eng)}")
        for s in sorted(ui)[:5]:
            print("  e.g.", s[:90])
    else:
        build(args.langs, args.batch, args.limit, args.pause)
