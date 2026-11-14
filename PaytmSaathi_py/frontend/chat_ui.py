"""Saathi Chat: a general assistant (RAG + MCP tools) behind input and output guardrails."""

from __future__ import annotations

import threading

import streamlit as st

from ai_engine.assistant import Assistant, ChatContext, ChatResult, TOOL_LABELS
from frontend.common import (
    L, current_policy, current_user, demo_mode, lang, my_policies, pick, policy_label, run, uid,
)
from frontend.theme import empty_state, esc, page_header

_ASSISTANT: Assistant | None = None
_PREWARM_STARTED = False
_LOCK = threading.Lock()


def assistant() -> Assistant:
    global _ASSISTANT
    with _LOCK:
        if _ASSISTANT is None:
            _ASSISTANT = Assistant()
        return _ASSISTANT


def _prewarm() -> None:
    """Start the MCP server and index the guides in the background so the first answer isn't slow."""
    global _PREWARM_STARTED
    with _LOCK:
        if _PREWARM_STARTED:
            return
        _PREWARM_STARTED = True

    def work():
        try:
            from ai_engine.kb import ensure_kb
            from ai_engine.mcp_bridge import get_bridge

            ensure_kb()
            get_bridge().start()
        except Exception:  # the assistant reports tools as unavailable if this fails
            pass

    threading.Thread(target=work, name="saathi-prewarm", daemon=True).start()


def _bridge_state() -> str:
    try:
        from ai_engine.mcp_bridge import get_bridge

        return "ready" if get_bridge().alive else "starting"
    except Exception:
        return "off"


_REDACTION_LABELS = {
    "redacted_otp": ("OTP", "ओटीपी"), "redacted_pin": ("PIN/CVV", "पिन/सीवीवी"), "redacted_aadhaar": ("Aadhaar number", "आधार नंबर"),
    "redacted_pan": ("PAN", "पैन"), "redacted_card": ("card number", "कार्ड नंबर"), "redacted_password": ("password", "पासवर्ड"),
    "redacted_api_key": ("API key", "एपीआई कुंजी"), "redacted_private_key": ("private key", "प्राइवेट की"),
}
_CLEANUP_KINDS = {"html_removed", "script_vector_removed", "secret_removed", "image_removed", "link_defanged",
                  "credential_request", "guarantee_language", "system_prompt_leak"}


def _notes_text(turn: dict) -> list[str]:
    out, seen = [], set()
    for n in turn.get("notes", []):
        kind = n["kind"]
        if kind in _REDACTION_LABELS and kind not in seen:
            en, hi = _REDACTION_LABELS[kind]
            what = L(en, hi)
            out.append(L(f"🔒 I hid your {what} for your safety. Never share it in chat.",
                         f"🔒 मैंने आपकी सुरक्षा के लिए आपका {what} छिपा दिया। इसे चैट में कभी साझा न करें।"))
            seen.add(kind)
        elif kind == "injection_in_document" and kind not in seen:
            out.append(L("⚠️ I ignored a passage that tried to give me instructions.", "⚠️ मैंने एक ऐसे अंश को अनदेखा किया जो मुझे निर्देश देने की कोशिश कर रहा था।"))
            seen.add(kind)
        elif kind in _CLEANUP_KINDS and "cleanup" not in seen:
            out.append(L("🛡️ I cleaned up part of the reply (links, markup or risky wording) before showing it.",
                         "🛡️ दिखाने से पहले मैंने जवाब के कुछ हिस्से (लिंक, मार्कअप या जोखिम भरे शब्द) साफ़ किए।"))
            seen.add("cleanup")
    for d in turn.get("degraded", []):
        if d == "tools_unavailable":
            out.append(L("ℹ️ My calculators and checkers were unavailable, so I answered without them.",
                         "ℹ️ मेरे कैलकुलेटर और जाँच उपकरण उपलब्ध नहीं थे, इसलिए मैंने उनके बिना जवाब दिया।"))
        if d == "knowledge_unavailable":
            out.append(L("ℹ️ Saathi's guides were unavailable for this answer.", "ℹ️ इस उत्तर के लिए साथी की गाइड उपलब्ध नहीं थी।"))
    return out


def _render_turn(turn: dict) -> None:
    with st.chat_message(turn["role"], avatar="🧑" if turn["role"] == "user" else "🛡️"):
        st.markdown(turn["content"])
        if turn["role"] != "assistant":
            return
        tools = turn.get("tools", [])
        if tools:
            chips = " ".join(f'<span class="ps-chip">{"✅" if t["ok"] else "⚠️"} {esc(L(TOOL_LABELS.get(t["name"], t["name"]), TOOL_LABELS.get(t["name"], t["name"])))}</span>' for t in tools)
            st.markdown(f'<div style="margin:.3rem 0">{L("Used", "इस्तेमाल")}: {chips}</div>', unsafe_allow_html=True)
        if turn.get("sources"):
            with st.expander(L(f"Sources ({len(turn['sources'])})", f"स्रोत ({len(turn['sources'])})")):
                for s in turn["sources"]:
                    where = s["title"] + (f" · {s['detail']}" if s.get("detail") else "")
                    icon = "📄" if s["kind"] == "policy" else "📘"
                    st.markdown(f'<div class="ps-note"><b>{icon} [{esc(s["ref"])}] {esc(where)}</b><br>{esc(s["text"])}</div>', unsafe_allow_html=True)
        for line in _notes_text(turn):
            st.caption(line)


def _suggestions() -> list[tuple[str, str]]:
    base = [
        ("Explain co-pay and deductible with an example", "को-पे और डिडक्टिबल को उदाहरण के साथ समझाइए"),
        ("What's the EMI on ₹5 lakh at 10.5% for 5 years?", "₹5 लाख पर 10.5% ब्याज, 5 साल की ईएमआई कितनी होगी?"),
        ("How do I complain to the Insurance Ombudsman?", "इंश्योरेंस ओम्बड्समैन को शिकायत कैसे करूँ?"),
        ("Is this safe? 'Your KYC expires today, update at paytm-kyc.xyz'", "क्या यह सुरक्षित है? 'आपका केवाईसी आज खत्म, paytm-kyc.xyz पर अपडेट करें'"),
        ("Money was debited but my UPI payment failed. What now?", "पैसे कट गए पर यूपीआई पेमेंट फेल हो गया। अब क्या करूँ?"),
        ("Help me write a polite leave application", "छुट्टी के लिए एक विनम्र आवेदन लिखने में मदद कीजिए"),
    ]
    return base


def _transcript(history: list[dict]) -> str:
    lines = []
    for t in history:
        who = "You" if t["role"] == "user" else "Saathi"
        lines.append(f"{who}: {t['content']}\n")
    return "\n".join(lines)


def render() -> None:
    page_header(L("Saathi Chat", "साथी चैट"),
                L("Ask anything. Saathi uses your policy and its guides, runs calculators and safety checks, and keeps your secrets safe.",
                  "कुछ भी पूछिए। साथी आपकी पॉलिसी और अपनी गाइड का इस्तेमाल करता है, कैलकुलेटर और सुरक्षा जाँच चलाता है और आपके राज़ सुरक्षित रखता है।"))
    _prewarm()

    c1, c2 = st.columns([2, 3], vertical_alignment="center")
    mine = my_policies()
    options = [None] + mine
    current = current_policy()
    idx = options.index(current) if current in options else 0
    chosen = c1.selectbox(L("Use my policy in this chat", "इस चैट में मेरी पॉलिसी इस्तेमाल करें"), options, index=idx,
                          format_func=lambda p: L("None", "कोई नहीं") if p is None else policy_label(p), key="chat_policy")
    state = _bridge_state()
    tools_chip = {"ready": ("#12A26B", L("Tools ready (MCP)", "टूल तैयार (MCP)")),
                  "starting": ("#D98A00", L("Tools starting…", "टूल शुरू हो रहे हैं…")),
                  "off": ("#D64545", L("Tools off", "टूल बंद"))}[state]
    c2.markdown(
        f'<span class="ps-chip"><span class="ps-dot" style="background:{tools_chip[0]}"></span>{esc(tools_chip[1])}</span> '
        f'<span class="ps-chip">🛡️ {esc(L("Safety filters on", "सुरक्षा फ़िल्टर चालू"))}</span> '
        f'<span class="ps-chip">🌐 {esc(L("Answers in your language", "आपकी भाषा में उत्तर"))}</span>',
        unsafe_allow_html=True)
    if demo_mode():
        st.warning(L("Demo mode: add a free Groq key in .env to chat for real.", "डेमो मोड: असली चैट के लिए .env में मुफ़्त Groq कुंजी जोड़ें।"))

    history: list[dict] = st.session_state.setdefault("gchat", {}).setdefault(uid(), [])
    clicked = None
    if not history:
        empty_state("💬", L("How can I help today?", "आज मैं कैसे मदद करूँ?"),
                    L("Try one of these, or type your own question below.", "इनमें से कोई आज़माएँ, या नीचे अपना सवाल लिखें।"))
        cols = st.columns(3)
        for i, (en, hi) in enumerate(_suggestions()):
            if cols[i % 3].button(L(en, hi), key=f"gsg_{i}", width="stretch"):
                clicked = L(en, hi)
    for turn in history:
        _render_turn(turn)

    question = st.chat_input(L("Ask Saathi anything…", "साथी से कुछ भी पूछिए…")) or clicked
    if question:
        user = current_user()
        ctx = ChatContext(user_id=uid(), user_name=user.first_name if user else "", language=lang(),
                          policy_id=chosen, policy_label=policy_label(chosen) if chosen else "")
        model_history = [{"role": t["role"], "content": t["content"]} for t in history]
        with st.spinner(L("Saathi is thinking… (the first message can take a little longer)",
                          "साथी सोच रहा है… (पहले संदेश में थोड़ा समय लग सकता है)")):
            result: ChatResult | None = run(assistant().chat, question, model_history, ctx)
        if result:
            history.append({"role": "user", "content": result.user_text or question})
            history.append({
                "role": "assistant", "content": result.answer,
                "sources": [s.model_dump() for s in result.sources],
                "tools": [{"name": t.name, "ok": t.ok} for t in result.tools_used],
                "notes": [n.model_dump() for n in result.guard_notes], "degraded": result.degraded,
            })
        st.rerun()

    if history:
        a, b = st.columns([1, 1])
        if a.button(L("🗑️ New chat", "🗑️ नई चैट"), key="gchat_clear"):
            st.session_state["gchat"][uid()] = []
            st.rerun()
        b.download_button(L("⬇️ Download chat", "⬇️ चैट डाउनलोड करें"), _transcript(history), file_name="saathi_chat.txt", mime="text/plain")
    st.caption(L("Saathi can make mistakes and is not a doctor, lawyer, tax or investment adviser. Chats stay in this session; OTPs, PINs, Aadhaar, PAN and card numbers are hidden.",
                 "साथी से गलती हो सकती है और वह डॉक्टर, वकील, कर या निवेश सलाहकार नहीं है। चैट इसी सेशन में रहती है; ओटीपी, पिन, आधार, पैन और कार्ड नंबर छिपा दिए जाते हैं।"))
