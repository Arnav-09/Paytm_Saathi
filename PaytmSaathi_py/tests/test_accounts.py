"""Accounts: registration rules, password hashing, login, lockout, tokens."""

import json

import pytest

from ai_engine import accounts
from ai_engine.accounts import AccountError


@pytest.fixture()
def users_file(tmp_path):
    return str(tmp_path / "users.json")


def _register(users_file, **kw):
    data = dict(full_name="Ravi Kumar", mobile="+91 98765 43210", password="secret123",
                confirm_password="secret123", email="Ravi@Example.com", preferred_language="hi",
                accepted_terms=True, users_file=users_file)
    data.update(kw)
    return accounts.register_user(**data)


def test_register_stores_hash_not_password(users_file):
    user = _register(users_file)
    assert user.mobile == "9876543210" and user.email == "ravi@example.com"
    assert user.preferred_language == "hi" and user.first_name == "Ravi"
    assert user.masked_mobile == "XXXXXX3210"
    raw = open(users_file, encoding="utf-8").read()
    assert "secret123" not in raw
    record = json.loads(raw)["users"][user.user_id]
    assert record["password_hash"] and record["salt"]


@pytest.mark.parametrize("override,code", [
    ({"full_name": "R"}, "invalid_name"),
    ({"mobile": "12345"}, "invalid_mobile"),
    ({"email": "not-an-email"}, "invalid_email"),
    ({"password": "short1", "confirm_password": "short1"}, "weak_password"),
    ({"password": "onlyletters", "confirm_password": "onlyletters"}, "weak_password"),
    ({"confirm_password": "different1"}, "password_mismatch"),
    ({"accepted_terms": False}, "terms_not_accepted"),
])
def test_register_validation(users_file, override, code):
    with pytest.raises(AccountError) as err:
        _register(users_file, **override)
    assert err.value.error_code == code
    assert err.value.message_hi  # every message has a Hindi version


def test_duplicate_mobile_and_email_rejected(users_file):
    _register(users_file)
    with pytest.raises(AccountError) as err:
        _register(users_file, email=None)
    assert err.value.error_code == "mobile_taken"
    with pytest.raises(AccountError) as err:
        _register(users_file, mobile="9123456780")
    assert err.value.error_code == "email_taken"


def test_login_by_mobile_or_email(users_file):
    _register(users_file)
    assert accounts.authenticate("98765 43210", "secret123", users_file).full_name == "Ravi Kumar"
    assert accounts.authenticate("RAVI@example.com", "secret123", users_file).mobile == "9876543210"


def test_wrong_password_and_unknown_user_give_same_error(users_file):
    _register(users_file)
    with pytest.raises(AccountError) as a:
        accounts.authenticate("9876543210", "wrongpass1", users_file)
    with pytest.raises(AccountError) as b:
        accounts.authenticate("9000000000", "wrongpass1", users_file)
    assert a.value.message == b.value.message


def test_lockout_after_repeated_failures(users_file):
    _register(users_file)
    for _ in range(accounts.MAX_FAILED_ATTEMPTS):
        with pytest.raises(AccountError):
            accounts.authenticate("9876543210", "wrongpass1", users_file)
    with pytest.raises(AccountError) as err:
        accounts.authenticate("9876543210", "secret123", users_file)  # even the right password
    assert err.value.error_code == "account_locked"


def test_policy_links_and_language(users_file):
    user = _register(users_file)
    accounts.link_policy(user.user_id, "p1", users_file)
    accounts.link_policy(user.user_id, "p1", users_file)
    assert accounts.user_policy_ids(user.user_id, users_file) == ["p1"]
    accounts.update_language(user.user_id, "en", users_file)
    assert accounts.get_user(user.user_id, users_file).preferred_language == "en"


def test_tokens(monkeypatch):
    monkeypatch.setenv("APP_SECRET", "test-secret")
    token = accounts.issue_token("U123")
    assert accounts.verify_token(token) == "U123"
    with pytest.raises(AccountError):
        accounts.verify_token(token[:-1] + ("0" if token[-1] != "0" else "1"))
    with pytest.raises(AccountError):
        accounts.verify_token(accounts.issue_token("U123", ttl_seconds=-10))
    with pytest.raises(AccountError):
        accounts.verify_token("garbage")
