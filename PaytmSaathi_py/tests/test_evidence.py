"""Ticket evidence: file checks, verification against the ticket's transaction ID and the user's name,
SMS date + transaction-ID rules, redaction, hints, complaint letter and evidence pack."""

import io
import zipfile

import pytest

from ai_engine import fintech
from ai_engine.exceptions import CalculationInputError, EvidenceRejectedError, InvalidFileError, PaytmSaathiError
from ai_engine.fintech import _verhoeff_valid

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64
JPG = b"\xff\xd8\xff\xe0" + b"0" * 64
PDF = b"%PDF-1.4\n" + b"0" * 64
NAME = "Ravi Kumar"
TXN = next(str(412345678900 + i) for i in range(10) if not _verhoeff_valid(str(412345678900 + i)))  # not Aadhaar-like
GOOD_TEXT = f"Payment failed. From: RAVI KUMAR. UPI Ref No: {TXN}. Rs 2,000.00 on 01/11/2026"
GOOD_SMS = f"Rs 2,000.00 debited on 01/11/2026 10:42. UPI Ref No {TXN}. OTP is 482913"


@pytest.fixture()
def ticket(tmp_path):
    t = fintech.create_dispute_ticket("money debited but payment failed", transaction_id=TXN, directory=str(tmp_path), user_id="U1")
    return t, str(tmp_path)


@pytest.fixture()
def reads(monkeypatch):
    """Control what text the 'OCR' reads from an uploaded file."""
    box = {"text": GOOD_TEXT}
    monkeypatch.setattr(fintech, "_read_document_text", lambda ext, data: box["text"])
    return box


def attach(ticket, name="shot.png", data=PNG, who="U1", expected=NAME):
    t, d = ticket
    return fintech.add_attachment(t.ticket_id, who, name, data, d, expected_name=expected)


# ----------------------------- file checks ---------------------------------
def test_matching_screenshot_and_pdf_are_accepted(ticket, reads):
    a = attach(ticket, "debit screenshot.png")
    b = attach(ticket, "statement.PDF", PDF)
    assert a.verified and a.kind == "screenshot" and a.filename == "debit_screenshot.png" and len(a.sha256) == 64
    assert any("Transaction ID" in c for c in a.checks) and "Your name matches" in a.checks
    assert b.kind == "document" and b.verified
    t, d = ticket
    assert len(fintech.get_dispute_ticket(t.ticket_id, d).attachments) == 2


@pytest.mark.parametrize("name,data", [("evil.png", b"MZ\x90\x00 not a png"), ("script.exe", PNG), ("noext", PNG), ("empty.png", b"")])
def test_wrong_or_fake_files_rejected(ticket, reads, name, data):
    with pytest.raises(InvalidFileError):
        attach(ticket, name, data)


def test_size_and_count_limits(ticket, reads):
    with pytest.raises(InvalidFileError):
        attach(ticket, "big.png", PNG + b"0" * (fintech.MAX_ATTACHMENT_BYTES + 1))
    for i in range(fintech.MAX_ATTACHMENTS):
        attach(ticket, f"{i}.jpg", JPG)
    with pytest.raises(InvalidFileError):
        attach(ticket, "one_more.jpg", JPG)


# ----------------------------- verification --------------------------------
def test_screenshot_with_another_transaction_id_is_rejected(ticket, reads):
    reads["text"] = f"From: Ravi Kumar. UPI Ref No: 998877665544. Rs 2,000"
    with pytest.raises(EvidenceRejectedError) as err:
        attach(ticket)
    assert "998877665544" in err.value.reasons[0] and TXN in err.value.reasons[0]
    assert err.value.reasons_hi and err.value.error_code == "evidence_rejected"


def test_screenshot_without_transaction_id_is_rejected(ticket, reads):
    reads["text"] = "From: Ravi Kumar. Payment failed."
    with pytest.raises(EvidenceRejectedError) as err:
        attach(ticket)
    assert f"does not show your transaction ID {TXN}" in err.value.reasons[0]


def test_screenshot_with_someone_elses_name_is_rejected(ticket, reads):
    reads["text"] = f"From: Amit Shah. UPI Ref No: {TXN}"
    with pytest.raises(EvidenceRejectedError) as err:
        attach(ticket)
    assert "does not show your name" in err.value.reasons[0] and "Ravi" in err.value.reasons[0]


def test_both_problems_are_reported_together(ticket, reads):
    reads["text"] = "From: Amit Shah. Ref No 111222333444"
    with pytest.raises(EvidenceRejectedError) as err:
        attach(ticket)
    assert len(err.value.reasons) == 2


def test_unreadable_screenshot_is_rejected(ticket, reads):
    reads["text"] = None
    with pytest.raises(EvidenceRejectedError) as err:
        attach(ticket)
    assert "could not read any text" in err.value.reasons[0]


def test_rejected_files_are_not_stored(ticket, reads, tmp_path):
    reads["text"] = "nothing useful"
    with pytest.raises(EvidenceRejectedError):
        attach(ticket)
    t, d = ticket
    assert fintech.get_dispute_ticket(t.ticket_id, d).attachments == []
    folder = tmp_path / "attachments" / t.ticket_id
    assert not folder.exists() or not any(folder.iterdir())  # nothing written to disk


def test_ocr_quirks_still_match():
    # words glued together, digits read as letters, one-letter slip in a long name part
    assert fintech.txn_id_in_text(TXN, "UPIRefNo" + TXN + "Rs2000")
    assert fintech.txn_id_in_text("412345678900", "Ref No 4I23456789OO")  # OCR read 1 as I and 0 as O
    assert fintech.txn_id_in_text("AXIS-123456", "txn axis 123456")
    assert not fintech.txn_id_in_text(TXN, "Ref No 412345678")
    assert fintech.name_in_text("Ravi Kumar", "FROM:RAVIKUMAR")[0]
    assert fintech.name_in_text("Panchal Nayak", "From Panchai Nayak")[0]  # OCR slip in a 7-letter part
    ok, missing = fintech.name_in_text("Ravi Kumar", "From: Ravi Sharma")
    assert not ok and missing == ["Kumar"]


@pytest.mark.parametrize("text,ok", [
    ("on 01/11/2026", True), ("on 01-11-26 at 10:42", True), ("2026-11-01", True), ("01 Nov 2026", True),
    ("01-Nov-26", True), ("Nov 1, 2026", True), ("on01/11/2026to", True), ("today at noon", False), ("Ref 412345678900", False),
])
def test_timestamp_detection(text, ok):
    assert fintech.has_timestamp(text) is ok


def test_ticket_without_transaction_id_blocks_evidence_until_added(tmp_path, reads):
    t = fintech.create_dispute_ticket("old ticket", directory=str(tmp_path), user_id="U1")
    with pytest.raises(EvidenceRejectedError):
        fintech.add_sms_text(t.ticket_id, "U1", GOOD_SMS, str(tmp_path))
    with pytest.raises(EvidenceRejectedError):
        fintech.add_attachment(t.ticket_id, "U1", "a.png", PNG, str(tmp_path), expected_name=NAME)
    fintech.set_transaction_id(t.ticket_id, "U1", TXN, str(tmp_path))
    assert fintech.add_sms_text(t.ticket_id, "U1", GOOD_SMS, str(tmp_path)).verified


def test_transaction_id_format_and_lock(ticket, reads):
    t, d = ticket
    with pytest.raises(CalculationInputError):
        fintech.create_dispute_ticket("x", transaction_id="12", directory=d, user_id="U1")
    with pytest.raises(CalculationInputError):
        fintech.set_transaction_id(t.ticket_id, "U1", "bad id!", d)
    attach(ticket)
    with pytest.raises(CalculationInputError):  # evidence was checked against the old ID
        fintech.set_transaction_id(t.ticket_id, "U1", "999888777666", d)


# ----------------------------- SMS -----------------------------------------
def test_good_sms_is_accepted_and_redacted(ticket):
    t, d = ticket
    a = fintech.add_sms_text(t.ticket_id, "U1", GOOD_SMS, d)
    assert a.verified and "Date/time found" in a.checks
    assert "482913" not in a.text and "[OTP REMOVED]" in a.text
    assert a.hints == {"utr": TXN, "amount": 2000.0, "date": "2026-11-01"}


@pytest.mark.parametrize("sms,reason", [
    (f"Rs 2,000.00 debited from your account. UPI Ref No {TXN}", "must include the transaction date"),
    ("Rs 2,000.00 debited on 01/11/2026. UPI Ref No 998877665544", "does not match your ticket"),
    ("Rs 2,000.00 debited from your account on 01/11/2026", "does not show your transaction ID"),
])
def test_bad_sms_is_rejected(ticket, sms, reason):
    t, d = ticket
    with pytest.raises(EvidenceRejectedError) as err:
        fintech.add_sms_text(t.ticket_id, "U1", sms, d)
    assert any(reason in r for r in err.value.reasons)
    with pytest.raises(InvalidFileError):
        fintech.add_sms_text(t.ticket_id, "U1", "hi", d)


# ----------------------------- ownership, hints, letter, pack --------------
def test_users_cannot_touch_other_users_tickets(ticket, reads):
    t, d = ticket
    for call in (lambda: attach(ticket, who="U2"),
                 lambda: fintech.add_sms_text(t.ticket_id, "U2", GOOD_SMS, d),
                 lambda: fintech.remove_attachment(t.ticket_id, "U2", "AAAA", d),
                 lambda: fintech.set_transaction_id(t.ticket_id, "U2", TXN, d),
                 lambda: attach(ticket, who="")):
        with pytest.raises(PaytmSaathiError) as err:
            call()
        assert err.value.error_code == "ticket_not_found"


def test_hints_fill_only_empty_ticket_fields(ticket):
    t, d = ticket
    fintech.add_sms_text(t.ticket_id, "U1", f"Rs 750 debited 05-11-2026 Txn ID {TXN}", d)
    updated = fintech.apply_hints_to_ticket(t.ticket_id, "U1", d)
    assert updated.amount == 750 and updated.transaction_id == TXN and updated.transaction_date == "2026-11-05"
    fintech.add_sms_text(t.ticket_id, "U1", f"Rs 1 debited 06-11-2026 Txn ID {TXN}", d)
    again = fintech.apply_hints_to_ticket(t.ticket_id, "U1", d)
    assert again.amount == 750 and again.transaction_date == "2026-11-05"  # never overwrites


def test_remove_attachment_deletes_file(ticket, reads, tmp_path):
    t, d = ticket
    a = attach(ticket, "a.png")
    fintech.remove_attachment(t.ticket_id, "U1", a.attachment_id, d)
    assert fintech.get_dispute_ticket(t.ticket_id, d).attachments == []
    assert not list((tmp_path / "attachments" / t.ticket_id).glob(f"{a.attachment_id}.*"))
    with pytest.raises(PaytmSaathiError):
        fintech.remove_attachment(t.ticket_id, "U1", a.attachment_id, d)


def test_complaint_lists_verified_evidence(ticket, reads):
    t, d = ticket
    attach(ticket, "debit.png")
    fintech.add_sms_text(t.ticket_id, "U1", GOOD_SMS, d)
    letter = fintech.draft_complaint(fintech.get_dispute_ticket(t.ticket_id, d), "Ravi Kumar", "bank")
    assert "Supporting evidence attached" in letter and "debit.png" in letter
    assert "checked: Transaction ID" in letter and "Your name matches" in letter
    assert "482913" not in letter
    assert "(to be attached)" in fintech.draft_complaint(t, "Ravi", "bank")


def test_evidence_pack_zip(ticket, reads):
    t, d = ticket
    attach(ticket, "debit.png")
    fintech.add_sms_text(t.ticket_id, "U1", GOOD_SMS, d)
    data = fintech.build_evidence_pack(fintech.get_dispute_ticket(t.ticket_id, d), "Ravi Kumar", "ombudsman", "XXXXXX3210", d)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        assert "complaint_ombudsman.txt" in names and "evidence_index.txt" in names
        assert any(n.startswith("evidence/") and n.endswith("debit.png") for n in names)
        assert any(n.endswith("_sms.txt") for n in names)
        assert zf.read([n for n in names if n.endswith("debit.png")][0]) == PNG
