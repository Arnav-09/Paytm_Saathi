"""User accounts: registration and login with basic details only.

No KYC data is collected here (that happens later, in the KYC journey). Users are
stored in a local JSON file; passwords are hashed with PBKDF2-SHA256 and a per-user
salt and are never logged. Repeated wrong passwords lock the account for a while.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from .exceptions import PaytmSaathiError
from .utils import POLICIES_DIR, ensure_dir

USERS_FILE = os.getenv("USERS_FILE", str(Path(POLICIES_DIR).parent / "users.json"))
PBKDF2_ITERATIONS = 200_000
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_SECONDS = 15 * 60
TOKEN_TTL_SECONDS = 7 * 24 * 3600

_LOCK = threading.Lock()


class AccountError(PaytmSaathiError):
    """Registration/login problem with a user-safe message (and a Hindi version)."""

    def __init__(self, message: str, message_hi: str, code: str = "account_error"):
        super().__init__(message, code)
        self.message_hi = message_hi


class PublicUser(BaseModel):
    """What the app may show about a user (never the password hash)."""

    user_id: str
    full_name: str
    mobile: str
    email: Optional[str] = None
    preferred_language: str = "en"
    created_at: str

    @property
    def first_name(self) -> str:
        return self.full_name.split()[0] if self.full_name else ""

    @property
    def masked_mobile(self) -> str:
        return "XXXXXX" + self.mobile[-4:]


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
def _path(users_file: Optional[str]) -> Path:
    return Path(users_file or USERS_FILE)


def _load(users_file: Optional[str]) -> dict:
    path = _path(users_file)
    if not path.exists():
        return {"users": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        raise AccountError("Account storage is unreadable. Please contact support.",
                           "खाता डेटा पढ़ा नहीं जा सका। कृपया सपोर्ट से संपर्क करें।", "account_storage_error")


def _save(data: dict, users_file: Optional[str]) -> None:
    path = _path(users_file)
    ensure_dir(path.parent)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)  # atomic on the same filesystem


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def normalize_mobile(mobile: str) -> str:
    digits = re.sub(r"[\s-]", "", mobile or "")
    return re.sub(r"^(\+91|91|0)(?=\d{10}$)", "", digits)


def _hash_password(password: str, salt: bytes) -> str:
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return base64.b64encode(digest).decode("ascii")


def password_problems(password: str) -> list[tuple[str, str]]:
    """Return (en, hi) messages for every rule the password breaks."""
    problems = []
    if len(password or "") < 8:
        problems.append(("Use at least 8 characters.", "कम से कम 8 अक्षर रखें।"))
    if not re.search(r"[A-Za-z]", password or ""):
        problems.append(("Include at least one letter.", "कम से कम एक अक्षर (A-Z) रखें।"))
    if not re.search(r"\d", password or ""):
        problems.append(("Include at least one number.", "कम से कम एक अंक रखें।"))
    return problems


def _public(record: dict) -> PublicUser:
    return PublicUser(**{k: record.get(k) for k in PublicUser.model_fields})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def register_user(
    full_name: str,
    mobile: str,
    password: str,
    confirm_password: str,
    email: Optional[str] = None,
    preferred_language: str = "en",
    accepted_terms: bool = False,
    users_file: Optional[str] = None,
) -> PublicUser:
    """Create an account. Basic details only - no KYC documents."""
    name = re.sub(r"\s+", " ", (full_name or "").strip())
    if not re.fullmatch(r"[A-Za-zऀ-ॿ .'-]{2,60}", name):
        raise AccountError("Enter your full name (letters only, 2-60 characters).",
                           "अपना पूरा नाम लिखें (केवल अक्षर, 2-60)।", "invalid_name")
    mobile_n = normalize_mobile(mobile)
    if not re.fullmatch(r"[6-9]\d{9}", mobile_n):
        raise AccountError("Enter a valid 10-digit Indian mobile number.",
                           "सही 10 अंकों का भारतीय मोबाइल नंबर लिखें।", "invalid_mobile")
    email_n = (email or "").strip().lower() or None
    if email_n and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[a-z]{2,}", email_n):
        raise AccountError("Enter a valid email address or leave it blank.",
                           "सही ईमेल लिखें या खाली छोड़ दें।", "invalid_email")
    problems = password_problems(password)
    if problems:
        raise AccountError(" ".join(p[0] for p in problems), " ".join(p[1] for p in problems), "weak_password")
    if password != confirm_password:
        raise AccountError("Passwords do not match.", "पासवर्ड मेल नहीं खाते।", "password_mismatch")
    if not accepted_terms:
        raise AccountError("Please accept the terms and privacy policy to continue.",
                           "आगे बढ़ने के लिए शर्तें और गोपनीयता नीति स्वीकार करें।", "terms_not_accepted")
    language = "hi" if preferred_language == "hi" else "en"

    with _LOCK:
        data = _load(users_file)
        for record in data["users"].values():
            if record["mobile"] == mobile_n:
                raise AccountError("This mobile number is already registered. Please log in.",
                                   "यह मोबाइल नंबर पहले से रजिस्टर्ड है। कृपया लॉग इन करें।", "mobile_taken")
            if email_n and record.get("email") == email_n:
                raise AccountError("This email is already registered. Please log in.",
                                   "यह ईमेल पहले से रजिस्टर्ड है। कृपया लॉग इन करें।", "email_taken")
        salt = secrets.token_bytes(16)
        user_id = "U" + secrets.token_hex(6).upper()
        record = {
            "user_id": user_id, "full_name": name, "mobile": mobile_n, "email": email_n,
            "preferred_language": language,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "salt": base64.b64encode(salt).decode("ascii"),
            "password_hash": _hash_password(password, salt),
            "failed_attempts": 0, "locked_until": 0,
        }
        data["users"][user_id] = record
        _save(data, users_file)
    return _public(record)


def authenticate(identifier: str, password: str, users_file: Optional[str] = None) -> PublicUser:
    """Log in with mobile number or email plus password."""
    ident = (identifier or "").strip().lower()
    if not ident or not password:
        raise AccountError("Enter your mobile number (or email) and password.",
                           "अपना मोबाइल नंबर (या ईमेल) और पासवर्ड लिखें।", "missing_credentials")
    mobile_n = normalize_mobile(ident)
    generic = AccountError("Mobile/email or password is incorrect.",
                           "मोबाइल/ईमेल या पासवर्ड गलत है।", "invalid_credentials")
    with _LOCK:
        data = _load(users_file)
        record = next((r for r in data["users"].values()
                       if r["mobile"] == mobile_n or (r.get("email") and r["email"] == ident)), None)
        if record is None:
            # Same message and similar cost as a wrong password: don't reveal who is registered.
            _hash_password(password, b"0" * 16)
            raise generic
        now = time.time()
        if record.get("locked_until", 0) > now:
            minutes = int((record["locked_until"] - now) // 60) + 1
            raise AccountError(f"Too many wrong attempts. Try again in {minutes} minutes.",
                               f"बहुत ज़्यादा गलत कोशिशें। {minutes} मिनट बाद फिर कोशिश करें।", "account_locked")
        salt = base64.b64decode(record["salt"])
        if not hmac.compare_digest(_hash_password(password, salt), record["password_hash"]):
            record["failed_attempts"] = record.get("failed_attempts", 0) + 1
            if record["failed_attempts"] >= MAX_FAILED_ATTEMPTS:
                record["failed_attempts"] = 0
                record["locked_until"] = now + LOCKOUT_SECONDS
            _save(data, users_file)
            raise generic
        record["failed_attempts"], record["locked_until"] = 0, 0
        _save(data, users_file)
    return _public(record)


def get_user(user_id: str, users_file: Optional[str] = None) -> Optional[PublicUser]:
    record = _load(users_file)["users"].get(user_id)
    return _public(record) if record else None


def update_language(user_id: str, language: str, users_file: Optional[str] = None) -> None:
    with _LOCK:
        data = _load(users_file)
        if user_id in data["users"]:
            data["users"][user_id]["preferred_language"] = "hi" if language == "hi" else "en"
            _save(data, users_file)


def link_policy(user_id: str, policy_id: str, users_file: Optional[str] = None) -> None:
    """Remember that this user uploaded this policy ("My policies")."""
    with _LOCK:
        data = _load(users_file)
        record = data["users"].get(user_id)
        if record is None:
            return
        policies = record.setdefault("policy_ids", [])
        if policy_id not in policies:
            policies.append(policy_id)
            _save(data, users_file)


def user_policy_ids(user_id: str, users_file: Optional[str] = None) -> list[str]:
    record = _load(users_file)["users"].get(user_id)
    return list(record.get("policy_ids", [])) if record else []


# ---------------------------------------------------------------------------
# Signed tokens for the REST API
# ---------------------------------------------------------------------------
def _secret() -> bytes:
    secret = os.getenv("APP_SECRET")
    if not secret:
        raise AccountError("APP_SECRET is not configured on the server.",
                           "सर्वर पर APP_SECRET सेट नहीं है।", "server_misconfigured")
    return secret.encode("utf-8")


def issue_token(user_id: str, ttl_seconds: int = TOKEN_TTL_SECONDS) -> str:
    payload = f"{user_id}.{int(time.time()) + ttl_seconds}"
    sig = hmac.new(_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def verify_token(token: str) -> str:
    """Return the user_id for a valid, unexpired token."""
    invalid = AccountError("Your session has expired. Please log in again.",
                           "आपका सेशन खत्म हो गया। कृपया फिर से लॉग इन करें।", "invalid_token")
    try:
        user_id, expires, sig = (token or "").split(".")
        expected = hmac.new(_secret(), f"{user_id}.{expires}".encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected) or int(expires) < time.time():
            raise invalid
    except ValueError:
        raise invalid
    return user_id
