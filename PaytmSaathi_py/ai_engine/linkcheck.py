"""Link, phone-number and sender checks for the scam check. Deterministic and offline.

What this can and cannot do:

* It spots the classic tricks: look-alike or brand-stuffed domains, shorteners, IP links,
  ``paytm.com@evil.xyz`` URL tricks, punycode, throwaway TLDs, plus a local **blocklist**
  you can extend (``data/blocklist.txt``, one domain per line).
* It checks whether a message that *claims* to be from a known company links to one of that
  company's official domains and comes from a plausible sender.
* It can NOT prove a link is safe, and a domain missing from our short official list is
  "unknown", not "fake". The verdict always tells the user to confirm in the official app.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from .utils import POLICIES_DIR

# brand -> (names that appear in messages, official registrable domains, SMS header keywords)
BRANDS: dict[str, dict] = {
    "Paytm": {"names": ["paytm"], "domains": ["paytm.com", "paytmbank.com", "paytminsurance.co.in", "paytmpayments.com"], "headers": ["PAYTM", "PYTM"]},
    "SBI": {"names": ["sbi", "state bank of india"], "domains": ["sbi.co.in", "onlinesbi.sbi", "onlinesbi.com", "sbicard.com", "sbilife.co.in", "sbigeneral.in"], "headers": ["SBI", "SBIN"]},
    "HDFC Bank": {"names": ["hdfc"], "domains": ["hdfcbank.com", "hdfcergo.com", "hdfclife.com"], "headers": ["HDFC", "HDFCBK"]},
    "ICICI Bank": {"names": ["icici"], "domains": ["icicibank.com", "icicilombard.com", "iciciprulife.com"], "headers": ["ICICI", "ICICIB"]},
    "Axis Bank": {"names": ["axis bank", "axisbank"], "domains": ["axisbank.com"], "headers": ["AXIS", "AXISBK"]},
    "Kotak": {"names": ["kotak"], "domains": ["kotak.com"], "headers": ["KOTAK", "KMBL"]},
    "PNB": {"names": ["punjab national bank", "pnb"], "domains": ["pnbindia.in"], "headers": ["PNB"]},
    "Bank of Baroda": {"names": ["bank of baroda", "bob"], "domains": ["bankofbaroda.in"], "headers": ["BOB", "BARODA"]},
    "LIC": {"names": ["lic", "life insurance corporation"], "domains": ["licindia.in"], "headers": ["LIC", "LICIND"]},
    "PhonePe": {"names": ["phonepe"], "domains": ["phonepe.com"], "headers": ["PHONEPE", "PHNEPE"]},
    "RBI": {"names": ["rbi", "reserve bank"], "domains": ["rbi.org.in"], "headers": ["RBI"]},
    "IRDAI": {"names": ["irdai", "irda"], "domains": ["irdai.gov.in"], "headers": ["IRDAI"]},
    "NPCI": {"names": ["npci"], "domains": ["npci.org.in"], "headers": ["NPCI"]},
    "UIDAI / Aadhaar": {"names": ["uidai", "aadhaar", "aadhar"], "domains": ["uidai.gov.in"], "headers": ["UIDAI"]},
    "Income Tax": {"names": ["income tax", "incometax"], "domains": ["incometax.gov.in", "incometaxindia.gov.in"], "headers": ["INCTAX", "ITDEPT"]},
    "EPFO": {"names": ["epfo", "provident fund"], "domains": ["epfindia.gov.in"], "headers": ["EPFO", "EPFIND"]},
}
OFFICIAL_DOMAINS = {d for b in BRANDS.values() for d in b["domains"]}

SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "cutt.ly", "rb.gy", "is.gd", "ow.ly", "shorturl.at", "tiny.cc", "rebrand.ly", "t.ly", "bl.ink", "wa.me", "lnkd.in"}
SUSPICIOUS_TLDS = {"xyz", "top", "click", "link", "work", "loan", "info", "online", "site", "icu", "buzz", "cfd", "live", "vip", "shop", "gq", "tk", "ml", "cf", "ga", "rest", "monster", "cyou", "sbs", "zip", "mov"}
BAIT_WORDS = ("kyc", "verify", "update", "secure", "login", "refund", "reward", "claim", "bonus", "cashback", "support", "helpdesk", "customercare", "unblock", "activate", "prize")
TWO_PART_TLDS = {"co.in", "gov.in", "org.in", "net.in", "ac.in", "nic.in", "res.in", "firm.in", "gen.in", "ind.in", "co.uk", "com.au"}

_URL_RE = re.compile(
    r"(?<![@\w.-])(?:https?://|www\.)[^\s<>\"')\]]+"
    r"|(?<![@\w.-])[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(?:\.[a-z0-9-]+)*\.(?:com|in|co\.in|org|net|gov\.in|xyz|top|click|link|info|online|site|app|me|ly|gl|co|cc|icu|buzz|live|vip|shop|tk|ml|cf|ga|rest|zip)(?:/[^\s<>\"')\]]*)?",
    re.IGNORECASE)
_PHONE_RE = re.compile(r"(?<![\d.])(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)|(?<![\d.])1[8]00[\s-]?\d{3,4}[\s-]?\d{3,4}(?!\d)|(?<![\d.])\+(?!91)\d{1,3}(?:[\s-]?\d{2,6}){2,4}(?!\d)")
_SUPPORT_CONTEXT = re.compile(r"(?:customer care|customer support|helpline|support|call (?:us|now)?|contact|whatsapp|bank|executive|officer)", re.IGNORECASE)


class LinkFinding(BaseModel):
    url: str
    domain: str
    verdict: str = "unknown"  # official | dangerous | suspicious | unknown
    reasons: list[str] = Field(default_factory=list)
    reasons_hi: list[str] = Field(default_factory=list)
    claimed_brand: Optional[str] = None
    score: int = 0


class PhoneFinding(BaseModel):
    number: str
    kind: str  # mobile | toll_free | international
    verdict: str  # ok | suspicious
    note: str
    note_hi: str


class SenderFinding(BaseModel):
    sender: str
    kind: str  # header | mobile | other
    verdict: str  # consistent | mismatch | suspicious | unknown
    note: str
    note_hi: str


# ---------------------------------------------------------------------------
# Blocklist (extendable)
# ---------------------------------------------------------------------------
def _blocklist_paths() -> list[Path]:
    here = Path(__file__).resolve().parent.parent / "knowledge" / "blocklist.txt"
    return [here, Path(os.getenv("BLOCKLIST_FILE", str(Path(POLICIES_DIR).parent / "blocklist.txt")))]


def load_blocklist() -> set[str]:
    domains: set[str] = set()
    for path in _blocklist_paths():
        if path.exists():
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.split("#", 1)[0].strip().lower()
                if line:
                    domains.add(line.removeprefix("www."))
    return domains


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def registrable_domain(host: str) -> str:
    parts = host.lower().strip(".").split(".")
    if len(parts) <= 2:
        return ".".join(parts)
    tail2 = ".".join(parts[-2:])
    return ".".join(parts[-3:]) if tail2 in TWO_PART_TLDS else tail2


def _label(domain: str) -> str:
    parts = domain.split(".")
    return parts[-3] if ".".join(parts[-2:]) in TWO_PART_TLDS and len(parts) >= 3 else parts[0]


def _normalise_lookalike(text: str) -> str:
    text = text.lower().replace("rn", "m").replace("vv", "w")
    return text.translate(str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "4": "a", "$": "s"}))


def _edit_distance(a: str, b: str, cap: int = 3) -> int:
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def extract_urls(text: str) -> list[str]:
    seen, out = set(), []
    for m in _URL_RE.finditer(text or ""):
        url = m.group(0).rstrip(".,;:!?)\"'")
        if url.lower() not in seen:
            seen.add(url.lower())
            out.append(url)
    return out


def detect_brands(text: str) -> list[str]:
    lowered = (text or "").lower()
    found = []
    for brand, info in BRANDS.items():
        if any(re.search(rf"\b{re.escape(n)}\b", lowered) for n in info["names"]):
            found.append(brand)
    return found


# ---------------------------------------------------------------------------
# Link analysis
# ---------------------------------------------------------------------------
def analyse_link(url: str, claimed_brands: Optional[list[str]] = None, blocklist: Optional[set[str]] = None) -> LinkFinding:
    claimed_brands = claimed_brands or []
    blocklist = load_blocklist() if blocklist is None else blocklist
    raw = url if re.match(r"https?://", url, re.IGNORECASE) else "http://" + url
    scheme_given = bool(re.match(r"https?://", url, re.IGNORECASE))
    try:
        parts = urlsplit(raw)
        host = (parts.hostname or "").lower()
    except ValueError:
        return LinkFinding(url=url, domain="", verdict="suspicious", reasons=["Malformed link"], reasons_hi=["गलत बना हुआ लिंक"], score=3)
    host = host.removeprefix("www.")
    domain = registrable_domain(host)
    f = LinkFinding(url=url, domain=domain)
    score = 0
    items: list[tuple[int, str, str]] = []

    def flag(weight: int, en: str, hi: str) -> None:
        nonlocal score
        score += weight
        items.append((weight, en, hi))
        f.reasons.append(en)  # re-sorted by weight before returning
        f.reasons_hi.append(hi)

    def strongest_first() -> None:
        ordered = sorted(items, key=lambda it: -it[0])
        f.reasons = [it[1] for it in ordered]
        f.reasons_hi = [it[2] for it in ordered]

    if domain in blocklist or host in blocklist:
        flag(10, "This domain is on the blocklist of reported scam sites", "यह डोमेन रिपोर्ट की गई ठग साइटों की सूची में है")
    if "@" in parts.netloc:
        flag(5, "The link hides its real destination after an '@' sign", "लिंक '@' के बाद अपना असली पता छिपाता है")
    if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", host):
        flag(5, "The link points to a raw IP address, not a company website", "लिंक किसी कंपनी की साइट नहीं, सीधे आईपी पते पर जाता है")
    if "xn--" in host:
        flag(3, "The domain uses disguised look-alike characters (punycode)", "डोमेन में मिलते-जुलते अक्षर (प्यूनीकोड) छिपाए गए हैं")
    if domain in SHORTENERS:
        flag(2, "A shortened link hides where it really goes", "छोटा लिंक असली मंज़िल छिपाता है")
    if domain in OFFICIAL_DOMAINS and not f.reasons:
        f.verdict, f.score = "official", 0
        brand = next((b for b, i in BRANDS.items() if domain in i["domains"]), None)
        f.claimed_brand = brand
        f.reasons.append(f"Official {brand} domain (from our list). Still, open it from the official app, not from a message.")
        f.reasons_hi.append(f"{brand} का आधिकारिक डोमेन (हमारी सूची के अनुसार)। फिर भी इसे संदेश से नहीं, आधिकारिक ऐप से खोलें।")
        return f

    tld = domain.rsplit(".", 1)[-1]
    if tld in SUSPICIOUS_TLDS:
        flag(2, f"Unusual domain ending '.{tld}' often used by throwaway scam sites", f"असामान्य डोमेन अंत '.{tld}' जो अक्सर ठग साइटें इस्तेमाल करती हैं")
    if not scheme_given or raw.lower().startswith("http://"):
        if scheme_given:
            flag(1, "Not a secure (https) link", "सुरक्षित (https) लिंक नहीं है")
    if host.count(".") >= 3:
        flag(1, "Many sub-domains, a common way to imitate a real site", "कई सब-डोमेन, असली साइट की नकल का आम तरीका")
    if domain.count("-") >= 2:
        flag(1, "Several hyphens in the domain name", "डोमेन नाम में कई हाइफ़न")
    if any(w in host for w in BAIT_WORDS):
        flag(2, "Bait words such as 'kyc', 'verify', 'refund' or 'reward' in the address", "पते में 'kyc', 'verify', 'refund' या 'reward' जैसे लालच वाले शब्द")

    label = _label(domain)
    norm_host, norm_label = _normalise_lookalike(host), _normalise_lookalike(label)
    brand_hit = None
    for brand, info in BRANDS.items():
        if domain in info["domains"]:
            continue
        for name in info["names"]:
            token = name.replace(" ", "")
            if len(token) >= 4 and (token in host.replace("-", "").replace(".", "")):
                brand_hit = (brand, "contains")
                break
            if len(token) >= 4 and token in norm_host.replace("-", "").replace(".", "") and token not in host:
                brand_hit = (brand, "lookalike")
                break
        if not brand_hit:
            for official in info["domains"]:
                ol = _label(official)
                if len(ol) >= 5 and len(norm_label) >= 5 and 0 < _edit_distance(norm_label, ol) <= 1 and norm_label != ol:
                    brand_hit = (brand, "lookalike")
                    break
        if brand_hit:
            break
    if brand_hit:
        brand, how = brand_hit
        f.claimed_brand = brand
        if how == "contains":
            flag(5, f"Uses the name '{brand}' but is NOT {brand}'s official website", f"'{brand}' का नाम इस्तेमाल करता है पर {brand} की आधिकारिक वेबसाइट नहीं है")
        else:
            flag(5, f"Look-alike of the {brand} website address", f"{brand} की वेबसाइट के पते का हूबहू-सा नकली रूप")
    for brand in claimed_brands:
        if brand != f.claimed_brand and domain not in BRANDS.get(brand, {}).get("domains", []):
            flag(3, f"The message says it is from {brand}, but this link is not a {brand} website", f"संदेश खुद को {brand} का बताता है, पर यह लिंक {brand} की वेबसाइट नहीं है")
            f.claimed_brand = brand
            break

    f.score = score
    f.verdict = "dangerous" if score >= 5 else "suspicious" if score >= 2 else "unknown"
    strongest_first()
    if f.verdict == "unknown":
        f.reasons.append("No red flags found, but this site is not on our list of official domains. Check it in the official app.")
        f.reasons_hi.append("कोई खतरे का संकेत नहीं मिला, पर यह साइट हमारी आधिकारिक सूची में नहीं है। आधिकारिक ऐप में जाँचें।")
    return f


# ---------------------------------------------------------------------------
# Phone numbers and sender
# ---------------------------------------------------------------------------
def analyse_phones(text: str) -> list[PhoneFinding]:
    out, seen = [], set()
    for m in _PHONE_RE.finditer(text or ""):
        raw = m.group(0).strip()
        digits = re.sub(r"\D", "", raw)
        if digits in seen:
            continue
        seen.add(digits)
        window = text[max(0, m.start() - 70): m.end() + 70]
        masked = raw if len(digits) < 6 else re.sub(r"\d", "X", raw[:-4]) + raw[-4:]
        if digits.startswith("1800") or digits.startswith("1860"):
            out.append(PhoneFinding(number=masked, kind="toll_free", verdict="ok",
                                    note="Toll-free numbers can be faked in messages. Call only the number printed on your card, statement or official app.",
                                    note_hi="संदेश में टोल-फ़्री नंबर भी नकली हो सकता है। केवल कार्ड, स्टेटमेंट या आधिकारिक ऐप पर छपे नंबर पर कॉल करें।"))
        elif m.group(0).lstrip().startswith("+") and not digits.startswith("91"):
            out.append(PhoneFinding(number=masked, kind="international", verdict="suspicious",
                                    note="An overseas number. Indian banks, insurers and wallets don't contact customers from foreign numbers.",
                                    note_hi="विदेशी नंबर। भारतीय बैंक, बीमा कंपनियाँ और वॉलेट ग्राहकों को विदेशी नंबर से संपर्क नहीं करते।"))
        else:
            support = bool(_SUPPORT_CONTEXT.search(window))
            out.append(PhoneFinding(
                number=masked, kind="mobile", verdict="suspicious" if support else "ok",
                note=("A personal mobile number is being given as 'support' or 'bank' contact. Real customer care uses toll-free or official numbers."
                      if support else "A mobile number appears in the message. Don't share OTPs or money with unknown callers."),
                note_hi=("'सपोर्ट' या 'बैंक' के नाम पर निजी मोबाइल नंबर दिया गया है। असली कस्टमर केयर टोल-फ़्री या आधिकारिक नंबर इस्तेमाल करता है।"
                         if support else "संदेश में एक मोबाइल नंबर है। अनजान कॉलर को ओटीपी या पैसे न दें।")))
    return out


_HEADER_RE = re.compile(r"^(?:[A-Za-z]{2}-)?[A-Za-z0-9]{3,8}(?:-[A-Za-z])?$")


def analyse_sender(sender: str, claimed_brands: list[str]) -> Optional[SenderFinding]:
    sender = (sender or "").strip()
    if not sender:
        return None
    digits = re.sub(r"\D", "", sender)
    if len(digits) == 12 and digits.startswith("91"):  # +91 prefix
        digits = digits[2:]
    elif len(digits) == 11 and digits.startswith("0"):  # trunk prefix
        digits = digits[1:]
    if len(digits) == 10 and re.fullmatch(r"[\d+\s-]+", sender):
        if claimed_brands:
            return SenderFinding(sender=sender, kind="mobile", verdict="suspicious",
                                 note=f"The message claims to be from {', '.join(claimed_brands)}, but it came from an ordinary mobile number. Genuine banks and insurers send SMS from short names like VM-HDFCBK.",
                                 note_hi=f"संदेश खुद को {', '.join(claimed_brands)} का बताता है, पर यह आम मोबाइल नंबर से आया है। असली बैंक और बीमा कंपनियाँ VM-HDFCBK जैसे छोटे नाम से एसएमएस भेजती हैं।")
        return SenderFinding(sender=sender, kind="mobile", verdict="unknown",
                             note="An ordinary mobile number. Be careful with anyone who asks for money, OTPs or documents.",
                             note_hi="आम मोबाइल नंबर। पैसे, ओटीपी या दस्तावेज़ माँगने वाले से सावधान रहें।")
    if _HEADER_RE.match(sender):
        letters = re.sub(r"^[A-Za-z]{2}-", "", sender).upper()
        letters = re.sub(r"-[A-Z]$", "", letters)
        if claimed_brands:
            if any(any(h in letters or letters in h for h in BRANDS[b]["headers"]) for b in claimed_brands if b in BRANDS):
                return SenderFinding(sender=sender, kind="header", verdict="consistent",
                                     note="The sender name matches the company mentioned. Headers are regulated, but still verify important requests in the official app.",
                                     note_hi="भेजने वाले का नाम बताई गई कंपनी से मेल खाता है। फिर भी ज़रूरी अनुरोध आधिकारिक ऐप में जाँचें।")
            return SenderFinding(sender=sender, kind="header", verdict="mismatch",
                                 note=f"The sender name '{sender}' doesn't look like {', '.join(claimed_brands)}. Treat this message with suspicion.",
                                 note_hi=f"भेजने वाले का नाम '{sender}' {', '.join(claimed_brands)} जैसा नहीं लगता। इस संदेश पर शक करें।")
        return SenderFinding(sender=sender, kind="header", verdict="unknown",
                             note="An SMS header. We can't verify who owns it offline.", note_hi="एसएमएस हेडर। हम इसके मालिक की ऑफ़लाइन पुष्टि नहीं कर सकते।")
    return SenderFinding(sender=sender, kind="other", verdict="unknown",
                         note="Unrecognised sender format.", note_hi="भेजने वाले का प्रारूप पहचाना नहीं गया।")
