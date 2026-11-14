"""Scam check upgrades (links, phones, sender, blocklist) and ticket evidence attachments."""

import io
import zipfile

import pytest

from ai_engine import fintech
from ai_engine.exceptions import InvalidFileError, PaytmSaathiError
from ai_engine.fintech import _verhoeff_valid
from ai_engine.linkcheck import (
    analyse_link,
    analyse_phones,
    analyse_sender,
    detect_brands,
    extract_urls,
    load_blocklist,
    registrable_domain,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64
JPG = b"\xff\xd8\xff\xe0" + b"0" * 64
PDF = b"%PDF-1.4\n" + b"0" * 64


# ----------------------------- links --------------------------------------
def test_registrable_domain():
    assert registrable_domain("pay.paytm.com") == "paytm.com"
    assert registrable_domain("www.sbi.co.in") == "sbi.co.in"
    assert registrable_domain("a.b.example.gov.in") == "example.gov.in"


def test_official_domains_are_recognised_but_still_cautioned():
    f = analyse_link("https://www.hdfcbank.com/personal", ["HDFC Bank"], set())
    assert f.verdict == "official" and "official app" in f.reasons[0]
    assert analyse_link("https://netbanking.hdfcbank.com/login", [], set()).verdict == "official"


@pytest.mark.parametrize("url", [
    "http://paytm-kyc-update.xyz/login",   # brand stuffed
    "https://paytrn.com/offer",            # look-alike (rn for m)
    "https://paytm.com@evil.example/x",    # '@' trick
    "http://192.168.4.5/verify",           # raw IP
    "https://xn--paytm-9qa.com",           # punycode
    "https://sbi-secure-login.top",        # brand + bait + throwaway TLD
])
def test_dangerous_links(url):
    f = analyse_link(url, [], set())
    assert f.verdict == "dangerous", (url, f.verdict, f.reasons)
    assert f.reasons and len(f.reasons) == len(f.reasons_hi)


def test_strongest_reason_comes_first():
    f = analyse_link("http://paytm-kyc-update.xyz/login", [], set())
    assert "Paytm" in f.reasons[0]


def test_shortener_is_suspicious_and_plain_site_is_unknown():
    assert analyse_link("https://bit.ly/abc", [], set()).verdict == "suspicious"
    f = analyse_link("https://example.org/menu", [], set())
    assert f.verdict == "unknown" and "official app" in f.reasons[-1]


def test_message_claiming_a_brand_but_linking_elsewhere_is_flagged():
    f = analyse_link("https://refund-center.org/claim", ["SBI"], set())
    assert f.verdict in ("suspicious", "dangerous") and f.claimed_brand == "SBI"


def test_blocklist_from_file(tmp_path, monkeypatch):
    path = tmp_path / "blocklist.txt"
    path.write_text("# comment\nbad-site.example\n  Evil.Test  \n", encoding="utf-8")
    monkeypatch.setenv("BLOCKLIST_FILE", str(path))
    block = load_blocklist()
    assert "bad-site.example" in block and "evil.test" in block
    f = analyse_link("https://sub.bad-site.example/x", [], block)
    assert f.verdict == "dangerous" and "blocklist" in f.reasons[0]


def test_extract_urls_ignores_emails_and_trailing_punctuation():
    urls = extract_urls("Mail me at ravi@example.com or visit http://paytm-kyc.xyz/a, then bit.ly/xyz. Done")
    assert "http://paytm-kyc.xyz/a" in urls and "bit.ly/xyz" in urls
    assert not any("ravi" in u for u in urls)


def test_detect_brands():
    assert set(detect_brands("Your SBI account and Paytm wallet")) == {"SBI", "Paytm"}
    assert detect_brands("Lunch at one") == []


# ----------------------------- phones & sender ----------------------------
def test_phone_findings():
    p = analyse_phones("Call customer care on 98765 43210 now")
    assert p[0].kind == "mobile" and p[0].verdict == "suspicious" and p[0].number.endswith("3210") and "9876" not in p[0].number
    assert analyse_phones("Toll free 1800 123 4567")[0].kind == "toll_free"
    assert analyse_phones("Call +44 7911 123456 for the bank")[0].verdict == "suspicious"
    assert analyse_phones("see you at 5 pm") == []


def test_sender_checks():
    assert analyse_sender("VM-HDFCBK", ["HDFC Bank"]).verdict == "consistent"
    assert analyse_sender("AX-HDFCBK", ["SBI"]).verdict == "mismatch"
    assert analyse_sender("9876501234", ["Paytm"]).verdict == "suspicious"
    assert analyse_sender("9876501234", []).verdict == "unknown"
    assert analyse_sender("", ["SBI"]) is None


def test_scam_assessment_combines_everything():
    r = fintech.check_scam_message("Dear customer, your Paytm KYC expires today. Update at http://paytm-kyc-update.xyz or call 9876501234 support",
                                   sender="9812345678")
    assert r.risk == "high" and "Paytm" in r.claimed_brands
    assert r.links[0].verdict == "dangerous" and r.phones and r.sender.verdict == "suspicious"
    assert len(r.reasons) == len(r.reasons_hi)
    assert any("Chakshu" in a for a in r.advice)
    assert any("paytm-kyc-update.xyz" in "Dear customer, your Paytm KYC expires today. Update at http://paytm-kyc-update.xyz"[a:b] for a, b in r.spans)


def test_genuine_looking_sms_stays_low():
    r = fintech.check_scam_message("HDFC Bank: Rs 2,000 spent on card ending 1234. Not you? Visit https://www.hdfcbank.com/help", "VM-HDFCBK")
    assert r.risk == "low" and r.links[0].verdict == "official" and r.sender.verdict == "consistent"


