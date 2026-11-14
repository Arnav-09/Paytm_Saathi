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


# ----------------------------- attachments --------------------------------
@pytest.fixture()
def ticket(tmp_path):
    t = fintech.create_dispute_ticket("money debited but payment failed", directory=str(tmp_path), user_id="U1")
    return t, str(tmp_path)


def test_attach_valid_files(ticket):
    t, d = ticket
    a = fintech.add_attachment(t.ticket_id, "U1", "debit screenshot.png", PNG, d, run_ocr=False)
    b = fintech.add_attachment(t.ticket_id, "U1", "statement.PDF", PDF, d, run_ocr=False)
    assert a.kind == "screenshot" and a.filename == "debit_screenshot.png" and len(a.sha256) == 64
    assert b.kind == "document"
    loaded = fintech.get_dispute_ticket(t.ticket_id, d)
    assert [x.attachment_id for x in loaded.attachments] == [a.attachment_id, b.attachment_id]


@pytest.mark.parametrize("name,data", [
    ("evil.png", b"MZ\x90\x00 not a png"),        # wrong content
    ("script.exe", PNG),                           # wrong extension
    ("noext", PNG),
    ("empty.png", b""),
])
def test_attachment_rejects_wrong_or_fake_files(ticket, name, data):
    t, d = ticket
    with pytest.raises(InvalidFileError):
        fintech.add_attachment(t.ticket_id, "U1", name, data, d, run_ocr=False)


def test_attachment_size_and_count_limits(ticket):
    t, d = ticket
    with pytest.raises(InvalidFileError):
        fintech.add_attachment(t.ticket_id, "U1", "big.png", PNG + b"0" * (fintech.MAX_ATTACHMENT_BYTES + 1), d, run_ocr=False)
    for i in range(fintech.MAX_ATTACHMENTS):
        fintech.add_attachment(t.ticket_id, "U1", f"{i}.jpg", JPG, d, run_ocr=False)
    with pytest.raises(InvalidFileError):
        fintech.add_attachment(t.ticket_id, "U1", "one_more.jpg", JPG, d, run_ocr=False)


def test_users_cannot_touch_other_users_tickets(ticket):
    t, d = ticket
    for call in (lambda: fintech.add_attachment(t.ticket_id, "U2", "x.png", PNG, d, run_ocr=False),
                 lambda: fintech.add_sms_text(t.ticket_id, "U2", "some sms text here", d),
                 lambda: fintech.remove_attachment(t.ticket_id, "U2", "AAAA", d),
                 lambda: fintech.add_attachment(t.ticket_id, "", "x.png", PNG, d, run_ocr=False)):
        with pytest.raises(PaytmSaathiError) as err:
            call()
        assert err.value.error_code == "ticket_not_found"


def test_sms_text_is_redacted_and_hints_extracted(ticket):
    t, d = ticket
    utr = next(str(412345678900 + i) for i in range(10) if not _verhoeff_valid(str(412345678900 + i)))  # not Aadhaar-like
    a = fintech.add_sms_text(t.ticket_id, "U1", f"Rs. 2,000.00 debited on 01/11/2026. UPI Ref No {utr}. OTP is 482913", d)
    assert "482913" not in a.text and "[OTP REMOVED]" in a.text
    assert a.hints == {"utr": utr, "amount": 2000.0, "date": "2026-11-01"}
    with pytest.raises(InvalidFileError):
        fintech.add_sms_text(t.ticket_id, "U1", "hi", d)


def test_hints_fill_only_empty_ticket_fields(ticket):
    t, d = ticket
    utr = next(str(998877665540 + i) for i in range(10) if not _verhoeff_valid(str(998877665540 + i)))
    fintech.add_sms_text(t.ticket_id, "U1", f"Rs 750 debited 05-11-2026 Txn ID {utr}", d)
    updated = fintech.apply_hints_to_ticket(t.ticket_id, "U1", d)
    assert updated.amount == 750 and updated.transaction_id == utr and updated.transaction_date == "2026-11-05"
    fintech.add_sms_text(t.ticket_id, "U1", "Rs 1 debited 06-11-2026 Txn ID 111111111111", d)
    again = fintech.apply_hints_to_ticket(t.ticket_id, "U1", d)
    assert again.amount == 750 and again.transaction_id == utr  # never overwrites what the user set


def test_remove_attachment_deletes_file(ticket, tmp_path):
    t, d = ticket
    a = fintech.add_attachment(t.ticket_id, "U1", "a.png", PNG, d, run_ocr=False)
    fintech.remove_attachment(t.ticket_id, "U1", a.attachment_id, d)
    assert fintech.get_dispute_ticket(t.ticket_id, d).attachments == []
    assert not list((tmp_path / "attachments" / t.ticket_id).glob(f"{a.attachment_id}.*"))
    with pytest.raises(PaytmSaathiError):
        fintech.remove_attachment(t.ticket_id, "U1", a.attachment_id, d)


def test_complaint_mentions_attachments_and_sms_text(ticket):
    t, d = ticket
    fintech.add_attachment(t.ticket_id, "U1", "debit.png", PNG, d, run_ocr=False)
    fintech.add_sms_text(t.ticket_id, "U1", "Rs 500 debited from your account for UPI payment", d)
    loaded = fintech.get_dispute_ticket(t.ticket_id, d)
    letter = fintech.draft_complaint(loaded, "Ravi Kumar", "bank")
    assert "Supporting evidence attached" in letter and "debit.png" in letter and "Rs 500 debited" in letter
    assert "(to be attached)" in fintech.draft_complaint(t, "Ravi", "bank")  # no evidence yet


def test_evidence_pack_zip(ticket):
    t, d = ticket
    fintech.add_attachment(t.ticket_id, "U1", "debit.png", PNG, d, run_ocr=False)
    fintech.add_sms_text(t.ticket_id, "U1", "Rs 500 debited from your account", d)
    loaded = fintech.get_dispute_ticket(t.ticket_id, d)
    data = fintech.build_evidence_pack(loaded, "Ravi Kumar", "ombudsman", "XXXXXX3210", d)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        assert "complaint_ombudsman.txt" in names and "evidence_index.txt" in names
        assert any(n.startswith("evidence/") and n.endswith("debit.png") for n in names)
        assert any(n.endswith("_sms.txt") for n in names)
        assert "sha256=" in zf.read("evidence_index.txt").decode()
        assert zf.read([n for n in names if n.endswith("debit.png")][0]) == PNG
