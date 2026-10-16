"""Accounts and refresh tokens (Step 11). Passwords: scrypt with a per-user salt. Works on SQLite and Postgres."""
import hashlib, hmac, os, re, secrets, time, uuid

from state.db import connect

EMAIL = re.compile(r"[^@\s]{1,64}@[^@\s]{1,190}\.[^@\s]{2,24}")
MIN_PW, MAX_PW, MAX_FAILS, LOCK_S, REFRESH_TTL = 10, 128, 5, 900, 30 * 86400
_N, _R, _P = 2 ** 14, 8, 1


class AccountError(Exception):
    def __init__(self, kind, message=""):
        super().__init__(message or kind)
        self.kind = kind


def hash_password(pw):
    salt = os.urandom(16)
    h = hashlib.scrypt(pw.encode("utf-8"), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    return f"scrypt${_N}${_R}${_P}${salt.hex()}${h.hex()}"


def verify_password(pw, stored):
    try:
        _, n, r, p, salt, h = stored.split("$")
        calc = hashlib.scrypt(pw.encode("utf-8"), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p), dklen=32)
        return hmac.compare_digest(calc, bytes.fromhex(h))
    except Exception:
        return False


_DUMMY = []


def _dummy():
    if not _DUMMY:
        _DUMMY.append(hash_password("not-a-real-password"))
    return _DUMMY[0]


def normalize_email(email):
    e = email.strip().lower() if isinstance(email, str) else ""
    if len(e) > 254 or not EMAIL.fullmatch(e):
        raise AccountError("invalid", "enter a valid email address")
    return e


def check_password(pw, email):
    if not isinstance(pw, str) or not MIN_PW <= len(pw) <= MAX_PW:
        raise AccountError("invalid", f"password must be {MIN_PW} to {MAX_PW} characters")
    if pw.strip().lower() == email:
        raise AccountError("invalid", "password must not be your email address")


def _h(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class AccountStore:
    def __init__(self, path=":memory:"):
        self.conn = connect(path)

    def register(self, email, password, now=None):
        email = normalize_email(email)
        check_password(password, email)
        uid, t, pw = "u" + uuid.uuid4().hex[:24], int(now if now is not None else time.time()), hash_password(password)
        taken = AccountError("exists", "an account with this email already exists")
        try:
            with self.conn:
                if self.conn.execute("SELECT 1 FROM accounts WHERE email = ?", (email,)).fetchone():
                    raise taken
                self.conn.execute("INSERT INTO accounts (user_id, email, pw_hash, created_at) VALUES (?, ?, ?, ?)", (uid, email, pw, t))
        except AccountError:
            raise
        except Exception:
            if self.conn.execute("SELECT 1 FROM accounts WHERE email = ?", (email,)).fetchone():
                raise taken
            raise
        return uid

    def login(self, email, password, now=None):
        t = int(now if now is not None else time.time())
        try:
            email = normalize_email(email)
        except AccountError:
            email = None
        row = self.conn.execute("SELECT user_id, pw_hash, failed, locked_until FROM accounts WHERE email = ?", (email,)).fetchone() if email else None
        bad = AccountError("bad_credentials", "invalid email or password")
        if row is None:
            verify_password(password if isinstance(password, str) else "", _dummy())   # same work as a real check
            raise bad
        if row["locked_until"] > t:
            raise AccountError("locked", "too many failed attempts, try again later")
        if not verify_password(password if isinstance(password, str) else "", row["pw_hash"]):
            failed = row["failed"] + 1
            lock = t + LOCK_S if failed >= MAX_FAILS else 0
            with self.conn:
                self.conn.execute("UPDATE accounts SET failed = ?, locked_until = ? WHERE user_id = ?", (0 if lock else failed, lock, row["user_id"]))
            raise bad
        if row["failed"] or row["locked_until"]:
            with self.conn:
                self.conn.execute("UPDATE accounts SET failed = 0, locked_until = 0 WHERE user_id = ?", (row["user_id"],))
        return row["user_id"]

    def issue_refresh(self, user_id, now=None):
        t, token = int(now if now is not None else time.time()), secrets.token_urlsafe(32)
        with self.conn:
            self.conn.execute("DELETE FROM refresh_tokens WHERE expires_at < ?", (t - 86400,))
            self.conn.execute("INSERT INTO refresh_tokens (token_hash, user_id, expires_at) VALUES (?, ?, ?)", (_h(token), user_id, t + REFRESH_TTL))
        return token

    def rotate_refresh(self, token, now=None):
        t = int(now if now is not None else time.time())
        if not isinstance(token, str) or not 20 <= len(token) <= 200:
            raise AccountError("bad_token", "invalid or expired token")
        h, user = _h(token), None
        with self.conn:
            ok = self.conn.execute("UPDATE refresh_tokens SET revoked = 1 WHERE token_hash = ? AND revoked = 0 AND expires_at > ?", (h, t)).rowcount == 1
            row = self.conn.execute("SELECT user_id, revoked FROM refresh_tokens WHERE token_hash = ?", (h,)).fetchone()
            if ok:
                user = row["user_id"]
            elif row is not None and row["revoked"]:   # a spent token was presented again: revoke the whole family
                self.conn.execute("UPDATE refresh_tokens SET revoked = 1 WHERE user_id = ?", (row["user_id"],))
        if user is None:
            raise AccountError("bad_token", "invalid or expired token")
        return user, self.issue_refresh(user, t)

    def revoke(self, token):
        if isinstance(token, str) and 20 <= len(token) <= 200:
            with self.conn:
                self.conn.execute("UPDATE refresh_tokens SET revoked = 1 WHERE token_hash = ?", (_h(token),))

    def close(self):
        self.conn.close()
