"""Paytm-style look for Streamlit: navy + cyan palette, white header, cards, steps, trust footer."""

from __future__ import annotations

import html

import streamlit as st

NAVY = "#002E6E"
NAVY_2 = "#0A3D91"
CYAN = "#00BAF2"
BG = "#F4F8FC"
BORDER = "#DFE9F5"
INK = "#1B2B41"
MUTED = "#5B6B82"
GREEN = "#12A26B"
AMBER = "#D98A00"
RED = "#D64545"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+Devanagari:wght@400;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, textarea, select {{
  font-family: 'Inter','Noto Sans Devanagari','Segoe UI',system-ui,sans-serif !important;
}}
/* Keep Streamlit's header (it holds the sidebar toggle and the top navigation). */
header[data-testid="stHeader"] {{ background:#FFFFFF; border-bottom:1px solid {BORDER}; box-shadow:0 1px 6px rgba(0,46,110,.05); }}
#MainMenu, footer[data-testid="stFooter"], .stDeployButton, [data-testid="stToolbarActions"] {{ display:none !important; }}
.block-container {{ padding-top: 4.2rem; padding-bottom: 1rem; max-width: 1180px; }}
h1, h2, h3, h4 {{ color:{NAVY}; letter-spacing:-0.015em; }}
h1 {{ font-weight:800; }}
p, li, label {{ color:{INK}; }}

/* Sidebar */
[data-testid="stSidebar"] {{ background:linear-gradient(180deg,{NAVY} 0%,#01245A 100%); }}
[data-testid="stSidebar"] * {{ color:#EAF3FF; }}
[data-testid="stSidebar"] a {{ color:#9FE4FF !important; }}
[data-testid="stSidebar"] .ps-side-card {{ background:rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.12);
  border-radius:14px; padding:.9rem 1rem; margin-bottom:.8rem; }}

/* Brand bar */
.ps-brand {{ display:flex; align-items:center; gap:.55rem; font-weight:800; font-size:1.35rem; color:{NAVY}; }}
.ps-logo {{ width:34px; height:34px; border-radius:10px; background:linear-gradient(135deg,{CYAN},{NAVY});
  display:inline-flex; align-items:center; justify-content:center; color:#fff; font-size:1.05rem; }}
.ps-brand b {{ color:{CYAN}; }}
.ps-tag {{ color:{MUTED}; font-size:.8rem; font-weight:500; margin-top:-.2rem; }}
.ps-chip {{ display:inline-flex; align-items:center; gap:.35rem; padding:.28rem .7rem; border-radius:999px;
  font-size:.78rem; font-weight:600; border:1px solid {BORDER}; background:#fff; color:{INK}; white-space:nowrap; }}
.ps-dot {{ width:8px; height:8px; border-radius:50%; display:inline-block; }}

/* Hero */
.ps-hero {{ background:radial-gradient(1200px 300px at 100% 0%, rgba(0,186,242,.35), transparent 60%),
  linear-gradient(120deg,{NAVY} 0%,{NAVY_2} 100%); color:#fff; border-radius:22px; padding:2rem 2.2rem;
  margin:.4rem 0 1.4rem; box-shadow:0 10px 30px rgba(0,46,110,.18); }}
.ps-hero h1 {{ color:#fff; margin:0 0 .5rem; font-size:2rem; line-height:1.2; }}
.ps-hero p {{ color:#D5E8FB; margin:0; font-size:1.02rem; max-width:720px; }}
.ps-hero .ps-hello {{ color:#9FE4FF; font-weight:600; font-size:.95rem; margin-bottom:.35rem; }}

/* Section header */
.ps-page {{ margin:.2rem 0 1rem; }}
.ps-page h1 {{ margin:0; font-size:1.75rem; }}
.ps-page p {{ margin:.25rem 0 0; color:{MUTED}; }}

/* Cards */
.ps-card {{ background:#fff; border:1px solid {BORDER}; border-radius:16px; padding:1.15rem 1.25rem;
  height:100%; box-shadow:0 2px 8px rgba(0,46,110,.05); transition:transform .15s, box-shadow .15s; }}
.ps-card:hover {{ transform:translateY(-2px); box-shadow:0 8px 20px rgba(0,46,110,.10); }}
.ps-card h4 {{ margin:.35rem 0 .35rem; font-size:1.08rem; }}
.ps-card p {{ margin:0; color:{MUTED}; font-size:.93rem; line-height:1.5; }}
.ps-card ul {{ margin:.5rem 0 0 1rem; padding:0; color:{INK}; font-size:.9rem; }}
.ps-icon {{ width:44px; height:44px; border-radius:12px; background:#E6F7FE; display:flex; align-items:center;
  justify-content:center; font-size:1.35rem; }}
.ps-stat {{ background:#fff; border:1px solid {BORDER}; border-radius:14px; padding:.85rem 1rem; }}
.ps-stat .v {{ color:{NAVY}; font-weight:800; font-size:1.35rem; }}
.ps-stat .l {{ color:{MUTED}; font-size:.82rem; }}

/* Glance columns */
.ps-glance {{ border-radius:16px; padding:1rem 1.1rem; height:100%; border:1px solid {BORDER}; background:#fff; }}
.ps-glance h5 {{ margin:0 0 .6rem; font-size:.95rem; }}
.ps-glance .item {{ font-size:.88rem; padding:.42rem 0; border-top:1px dashed {BORDER}; color:{INK}; }}
.ps-glance .item:first-of-type {{ border-top:none; }}
.ps-glance .ref {{ color:{MUTED}; font-size:.76rem; }}
.ps-g-ok {{ border-top:4px solid {GREEN}; }} .ps-g-bad {{ border-top:4px solid {RED}; }} .ps-g-warn {{ border-top:4px solid {AMBER}; }}

/* Steps */
.ps-steps {{ display:flex; gap:.5rem; margin:.3rem 0 1rem; flex-wrap:wrap; }}
.ps-step {{ display:flex; align-items:center; gap:.45rem; padding:.4rem .85rem .4rem .45rem; border-radius:999px;
  background:#fff; border:1px solid {BORDER}; font-size:.85rem; color:{MUTED}; font-weight:600; }}
.ps-step .n {{ width:22px; height:22px; border-radius:50%; background:#E9EEF5; color:{MUTED}; display:flex;
  align-items:center; justify-content:center; font-size:.75rem; }}
.ps-step.on {{ border-color:{CYAN}; color:{NAVY}; }} .ps-step.on .n {{ background:{CYAN}; color:{NAVY}; }}
.ps-step.done .n {{ background:{GREEN}; color:#fff; }}

/* Pills, notes */
.ps-pill {{ display:inline-block; padding:.22rem .8rem; border-radius:999px; font-weight:700; font-size:.82rem; }}
.ps-ok {{ background:#E2F6EE; color:{GREEN}; }} .ps-warn {{ background:#FFF2D9; color:{AMBER}; }}
.ps-bad {{ background:#FDE7E7; color:{RED}; }} .ps-info {{ background:#E2F6FD; color:#0077A8; }}
.ps-note {{ border-left:4px solid {CYAN}; background:#EAF8FE; padding:.7rem 1rem; border-radius:10px; color:#12405F; margin:.5rem 0; }}
.ps-alert {{ border-left:4px solid {RED}; background:#FDECEC; padding:.7rem 1rem; border-radius:10px; color:#7A1F1F; margin:.5rem 0; }}
.ps-empty {{ text-align:center; padding:2.2rem 1rem; border:2px dashed {BORDER}; border-radius:18px; background:#fff; color:{MUTED}; }}
.ps-empty .big {{ font-size:2.2rem; }}
.ps-msg {{ background:#fff; border:1px solid {BORDER}; border-radius:12px; padding:.9rem 1rem; line-height:1.7; white-space:pre-wrap; }}
.ps-msg mark {{ background:#FFD7D7; color:#7A1F1F; padding:0 .15rem; border-radius:4px; }}
.ps-big-number {{ font-size:2.1rem; font-weight:800; color:{NAVY}; line-height:1.1; }}

/* Streamlit widgets */
div[data-testid="stMetric"] {{ background:#fff; border:1px solid {BORDER}; border-radius:14px; padding:.8rem 1rem; }}
div[data-testid="stMetricValue"] {{ color:{NAVY}; font-weight:800; }}
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {{
  border-radius:10px; font-weight:600; border:1px solid {NAVY}; color:{NAVY}; background:#fff; }}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{
  background:{CYAN}; color:{NAVY}; border-color:{CYAN}; box-shadow:0 4px 12px rgba(0,186,242,.30); }}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover {{
  background:#2CC8F6; border-color:#2CC8F6; color:{NAVY}; }}
.stTabs [data-baseweb="tab-list"] {{ gap:1.4rem; border-bottom:1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{ font-weight:600; color:{MUTED}; padding:.6rem .1rem; }}
.stTabs [aria-selected="true"] {{ color:{NAVY}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background:{CYAN}; height:3px; border-radius:3px; }}
[data-testid="stChatMessage"] {{ background:#fff; border:1px solid {BORDER}; border-radius:14px; }}
[data-testid="stExpander"] details {{ border-radius:12px; border-color:{BORDER}; background:#fff; }}
[data-testid="stForm"] {{ background:#fff; border:1px solid {BORDER}; border-radius:16px; padding:1.2rem; }}

/* Auth page */
.ps-auth-side {{ background:linear-gradient(150deg,{NAVY} 0%,{NAVY_2} 70%,#0F6FC6 100%); border-radius:22px;
  padding:2.2rem 2rem; color:#fff; min-height:520px; }}
.ps-auth-side h2 {{ color:#fff; font-size:1.8rem; margin:.8rem 0 .5rem; }}
.ps-auth-side p, .ps-auth-side li {{ color:#D5E8FB; }}
.ps-auth-side ul {{ padding-left:1.1rem; line-height:2; }}

/* Footer */
.ps-footer {{ margin-top:2.5rem; padding:1.4rem 1.5rem; border-radius:18px; background:#fff; border:1px solid {BORDER};
  color:{MUTED}; font-size:.8rem; line-height:1.6; }}
.ps-footer b {{ color:{NAVY}; }}
.ps-footer .cols {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:1rem; margin-bottom:.8rem; }}
.ps-footer .warn {{ background:#FFF6E5; border-radius:10px; padding:.55rem .8rem; color:#7A5200; margin:.4rem 0 .8rem; }}
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(text: object) -> str:
    return html.escape(str(text))


def pill(text: str, kind: str = "info") -> str:
    return f'<span class="ps-pill ps-{kind}">{esc(text)}</span>'


def note(text: str) -> None:
    st.markdown(f'<div class="ps-note">{esc(text)}</div>', unsafe_allow_html=True)


def alert(text: str) -> None:
    st.markdown(f'<div class="ps-alert">{esc(text)}</div>', unsafe_allow_html=True)


def card(icon: str, title: str, body: str, bullets: list[str] | None = None) -> str:
    items = "".join(f"<li>{esc(b)}</li>" for b in (bullets or []))
    return (f'<div class="ps-card"><div class="ps-icon">{icon}</div><h4>{esc(title)}</h4>'
            f'<p>{esc(body)}</p>{f"<ul>{items}</ul>" if items else ""}</div>')


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="ps-page"><h1>{esc(title)}</h1>'
                + (f"<p>{esc(subtitle)}</p>" if subtitle else "") + "</div>", unsafe_allow_html=True)


def steps(labels: list[str], current: int) -> None:
    """Numbered step indicator; `current` is 0-based."""
    parts = []
    for i, label in enumerate(labels):
        cls = "done" if i < current else ("on" if i == current else "")
        mark = "✓" if i < current else str(i + 1)
        parts.append(f'<div class="ps-step {cls}"><span class="n">{mark}</span>{esc(label)}</div>')
    st.markdown(f'<div class="ps-steps">{"".join(parts)}</div>', unsafe_allow_html=True)


def empty_state(icon: str, title: str, body: str) -> None:
    st.markdown(f'<div class="ps-empty"><div class="big">{icon}</div><h4>{esc(title)}</h4>'
                f'<p>{esc(body)}</p></div>', unsafe_allow_html=True)


def stat(label: str, value: str) -> str:
    return f'<div class="ps-stat"><div class="v">{esc(value)}</div><div class="l">{esc(label)}</div></div>'


def highlight(text: str, spans: list[tuple[int, int]]) -> str:
    """Escape text and wrap the given character spans in <mark>."""
    out, pos = [], 0
    for start, end in spans:
        out.append(esc(text[pos:start]))
        out.append(f"<mark>{esc(text[start:end])}</mark>")
        pos = end
    out.append(esc(text[pos:]))
    return f'<div class="ps-msg">{"".join(out)}</div>'


def inr(value: float | None) -> str:
    """Format rupees with Indian digit grouping: 1234567.5 -> ₹12,34,567.50"""
    if value is None:
        return "—"
    negative = value < 0
    whole, _, frac = f"{abs(value):.2f}".partition(".")
    head, tail = whole[:-3], whole[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    text = ",".join(groups + [tail]) if groups else tail
    sign = "-" if negative else ""
    return f"{sign}₹{text}" if frac == "00" else f"{sign}₹{text}.{frac}"
