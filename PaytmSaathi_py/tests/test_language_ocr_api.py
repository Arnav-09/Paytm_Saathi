"""Hindi support, OCR availability handling, and the REST API."""

import pytest

from ai_engine import language
from ai_engine.llm_provider import LLMProvider, MockLLMProvider


class FakeLLM(LLMProvider):
    def __init__(self, reply):
        self.reply = reply

    def generate(self, prompt, system_prompt=None, temperature=0.2, max_tokens=1024):
        return self.reply

    def generate_structured(self, prompt, schema, system_prompt=None, temperature=0.1):
        return {}


def test_language_detection():
    assert language.detect_language("मेरा क्लेम कितना मिलेगा?") == "hi"
    assert language.detect_language("What is my co-pay?") == "en"
    assert language.detect_language("") == "en"
    assert language.normalize_language("Hindi") == "hi"
    assert language.normalize_language(None) == "en"


def test_fixed_insufficient_answer_translates_offline():
    text, ok = language.translate_text(language.INSUFFICIENT_EVIDENCE_EN, "hi", MockLLMProvider())
    assert ok and text == language.INSUFFICIENT_EVIDENCE_HI


def test_mock_llm_never_fakes_a_translation():
    text, ok = language.translate_text("Your room rent limit is ₹5,000. [Source 1]", "hi", MockLLMProvider())
    assert not ok and text.startswith("Your room rent")


def test_translation_kept_only_if_facts_preserved():
    original = "Your room rent limit is ₹5,000 per day. [Source 1]"
    good = "आपके कमरे के किराए की सीमा ₹5,000 प्रति दिन है। [Source 1]"
    bad_number = "आपके कमरे के किराए की सीमा ₹6,000 प्रति दिन है। [Source 1]"
    missing_cite = "आपके कमरे के किराए की सीमा ₹5,000 प्रति दिन है।"
    assert language.translate_text(original, "hi", FakeLLM(good)) == (good, True)
    assert language.translate_text(original, "hi", FakeLLM(bad_number)) == (original, False)
    assert language.translate_text(original, "hi", FakeLLM(missing_cite)) == (original, False)
    assert language.translate_text(original, "hi", FakeLLM("still english 5000 [Source 1]")) == (original, False)


def test_hindi_question_to_english_for_retrieval():
    english = language.question_to_english("मोतियाबिंद की सर्जरी कवर है क्या", MockLLMProvider())
    assert "cataract" in english
    assert language.question_to_english("Is ICU covered?", None) == "Is ICU covered?"


def test_ocr_unavailable_gives_clear_error_for_images(tmp_path, monkeypatch):
    from ai_engine import bill_extractor, ocr
    from ai_engine.exceptions import OCRUnavailableError

    monkeypatch.setattr(ocr, "available_backend", lambda: None)
    monkeypatch.setattr(bill_extractor, "ocr_image_bytes", ocr.ocr_image_bytes)
    image = tmp_path / "bill.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 20)
    with pytest.raises(OCRUnavailableError):
        bill_extractor.extract_bill(str(image))


def test_bill_rejects_unsupported_type(tmp_path):
    from ai_engine.bill_extractor import extract_bill
    from ai_engine.exceptions import InvalidFileError

    doc = tmp_path / "bill.docx"
    doc.write_bytes(b"x")
    with pytest.raises(InvalidFileError):
        extract_bill(str(doc))


# ------------------------------- API --------------------------------------
@pytest.fixture()
def client():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from backend.main import app

    return TestClient(app)


def test_api_health_and_emi(client):
    assert client.get("/health").json()["status"] == "ok"
    res = client.post("/lending/emi", json={"principal": 1000000, "annual_rate_pct": 12, "tenure_months": 12})
    assert res.status_code == 200 and res.json()["emi"] == pytest.approx(88848.79, abs=0.01)


def test_api_engine_errors_map_to_http_status(client):
    res = client.post("/lending/emi", json={"principal": 0, "annual_rate_pct": 12, "tenure_months": 12})
    assert res.status_code == 422 and res.json()["error"] == "calculation_input_error"
    assert client.get("/fintech/disputes/DSP-20250101-ABCDEF").status_code == 404


def test_api_scam_and_validate(client):
    res = client.post("/fintech/scam-check", json={"text": "Share your OTP now, last chance, click bit.ly/x"})
    assert res.json()["risk"] == "high"
    res = client.post("/fintech/validate", json={"field": "ifsc", "value": "SBIN0001234"})
    assert res.json()["valid"] is True


def test_api_rejects_non_pdf_policy_upload(client):
    res = client.post("/policies/test_policy/ingest", files={"file": ("x.txt", b"hello", "text/plain")})
    assert res.status_code == 400 and res.json()["error"] == "invalid_file"
