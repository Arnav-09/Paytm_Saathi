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
    assert client.get("/fintech/disputes/DSP-20250101-ABCDEF").status_code == 401  # tickets now require login


def test_api_scam_and_validate(client):
    res = client.post("/fintech/scam-check", json={"text": "Share your OTP now, last chance, click bit.ly/x"})
    assert res.json()["risk"] == "high"
    res = client.post("/fintech/validate", json={"field": "ifsc", "value": "SBIN0001234"})
    assert res.json()["valid"] is True


def test_api_rejects_non_pdf_policy_upload(client):
    res = client.post("/policies/test_policy/ingest", files={"file": ("x.txt", b"hello", "text/plain")})
    assert res.status_code == 400 and res.json()["error"] == "invalid_file"


def test_api_ticket_ownership_and_evidence(client, tmp_path, monkeypatch):
    monkeypatch.setenv("APP_SECRET", "test-secret")
    monkeypatch.setattr("ai_engine.accounts.USERS_FILE", str(tmp_path / "users.json"))
    monkeypatch.setattr("ai_engine.fintech.DISPUTES_DIR", str(tmp_path / "disputes"))

    def register(mobile):
        r = client.post("/auth/register", json={"full_name": "Test User", "mobile": mobile, "password": "secret123",
                                                "confirm_password": "secret123", "accepted_terms": True})
        assert r.status_code == 200, r.text
        return {"Authorization": "Bearer " + r.json()["token"]}

    alice, bob = register("9876500001"), register("9876500002")
    assert client.post("/fintech/disputes", json={"description": "no txn id"}, headers=alice).status_code == 422  # txn ID required
    monkeypatch.setattr("ai_engine.fintech._read_document_text", lambda ext, data: "From: Test User. UPI Ref No 998877665540")
    t = client.post("/fintech/disputes", json={"description": "money debited but payment failed", "transaction_id": "998877665540"}, headers=alice).json()
    tid = t["ticket_id"]
    assert client.get(f"/fintech/disputes/{tid}", headers=alice).status_code == 200
    assert client.get(f"/fintech/disputes/{tid}", headers=bob).status_code == 404  # not someone else's
    r = client.post(f"/fintech/disputes/{tid}/attachments", headers=alice,
                    files={"file": ("shot.png", b"\x89PNG\r\n\x1a\n" + b"0" * 40, "image/png")})
    assert r.status_code == 200 and r.json()["kind"] == "screenshot"
    fake = client.post(f"/fintech/disputes/{tid}/attachments", headers=alice, files={"file": ("x.png", b"MZ evil", "image/png")})
    assert fake.status_code == 400
    bad = client.post(f"/fintech/disputes/{tid}/sms", headers=alice, json={"text": "Rs 500 debited. OTP is 482913"})
    assert bad.status_code == 422 and bad.json()["error"] == "evidence_rejected" and len(bad.json()["reasons"]) == 2
    sms = client.post(f"/fintech/disputes/{tid}/sms", headers=alice, json={"text": "Rs 500 debited on 01/11/2026. Ref No 998877665540. OTP is 482913"})
    assert sms.status_code == 200, sms.text
    assert "482913" not in sms.json()["text"]
    pack = client.get(f"/fintech/disputes/{tid}/evidence-pack", headers=alice)
    assert pack.status_code == 200 and pack.headers["content-type"] == "application/zip"
    assert client.get(f"/fintech/disputes/{tid}/evidence-pack", headers=bob).status_code == 404


def test_api_chat_requires_login_and_own_policy(client, tmp_path, monkeypatch):
    monkeypatch.setenv("APP_SECRET", "test-secret")
    monkeypatch.setattr("ai_engine.accounts.USERS_FILE", str(tmp_path / "users.json"))
    assert client.post("/chat", json={"message": "hi"}).status_code == 401
    r = client.post("/auth/register", json={"full_name": "Chat User", "mobile": "9876500003", "password": "secret123",
                                            "confirm_password": "secret123", "accepted_terms": True})
    headers = {"Authorization": "Bearer " + r.json()["token"]}
    assert client.post("/chat", json={"message": "hi", "policy_id": "someone_elses_policy"}, headers=headers).status_code == 404

    import backend.main as api
    from ai_engine.assistant import ChatResult

    class FakeAssistant:
        def chat(self, message, history, ctx):
            return ChatResult(answer=f"echo:{ctx.user_id}:{len(history)}", user_text=message)

    monkeypatch.setattr(api, "_ASSISTANT", FakeAssistant())
    out = client.post("/chat", json={"message": "hi", "history": [{"role": "user", "content": "x"}]}, headers=headers).json()
    assert out["answer"].startswith("echo:U") and out["answer"].endswith(":1")


def test_api_languages(client):
    assert set(client.get("/languages").json()) == {"en", "hi", "mr", "bn", "ta", "te", "gu"}


def test_more_languages_detection_and_normalisation():
    assert language.detect_language("எனது பாலிசியில் அறை வாடகை எவ்வளவு?") == "ta"
    assert language.detect_language("আমার পলিসিতে রুম ভাড়া কত?") == "bn"
    assert language.detect_language("મારી પોલિસીમાં રૂમ ભાડું કેટલું છે?") == "gu"
    assert language.detect_language("నా పాలసీలో గది అద్దె ఎంత?") == "te"
    assert language.normalize_language("Marathi") == "mr" and language.normalize_language("தமிழ்") == "ta"
    assert language.normalize_language("klingon") == "en"


def test_translation_to_tamil_checks_script_and_native_digits():
    original = "Room rent limit is ₹5,000 per day [K1]"
    good = "அறை வாடகை வரம்பு ஒரு நாளைக்கு ₹௫,௦௦௦ [K1]"  # Tamil digits are accepted
    assert language.translate_text(original, "ta", FakeLLM(good)) == (good, True)
    assert language.translate_text(original, "ta", FakeLLM("अभी हिंदी में ₹5,000 [K1]")) == (original, False)  # wrong script
    assert language.translate_text(original, "ta", FakeLLM("அறை வாடகை ₹6,000 [K1]")) == (original, False)  # number changed


def test_ui_catalog_lookup_without_live_translation(tmp_path, monkeypatch):
    import json as _json

    from frontend import i18n

    (tmp_path / "ta.json").write_text(_json.dumps({"Log in": "உள்நுழை"}), encoding="utf-8")
    monkeypatch.setattr(i18n, "LOCALES_DIR", tmp_path)
    monkeypatch.setattr(i18n, "_CATALOGS", {})
    assert i18n.tr("Log in", "ta", live=False) == "உள்நுழை"
    assert i18n.tr("Not translated yet", "ta", live=False) == "Not translated yet"
    assert i18n.tr("Log in", "en") == "Log in" and i18n.tr("Log in", "hi") == "Log in"
