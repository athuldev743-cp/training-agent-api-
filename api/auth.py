"""Signed access tokens (HS256 JWT, standard library only). The user id lives in the signed 'sub' claim."""
import base64, hashlib, hmac, json, os, re, time

_USER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:@-]{0,63}")
TTL_S = 3600


class AuthError(Exception):
    pass


def _b64(b):
    return base64.urlsafe_b64encode(b).rstrip(b"=")


def _unb64(s):
    return base64.urlsafe_b64decode(s + b"=" * (-len(s) % 4))


def secret():
    s = os.getenv("API_TOKEN_SECRET", "")
    if len(s) < 32:
        raise AuthError("API_TOKEN_SECRET must be set to at least 32 characters")
    return s.encode()


def issue(user_id, ttl=TTL_S, now=None, key=None):
    if not isinstance(user_id, str) or not _USER.fullmatch(user_id):
        raise AuthError("bad user id")
    t = int(now if now is not None else time.time())
    head = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    body = _b64(json.dumps({"sub": user_id, "iat": t, "exp": t + ttl}, separators=(",", ":")).encode())
    sig = _b64(hmac.new(key or secret(), head + b"." + body, hashlib.sha256).digest())
    return (head + b"." + body + b"." + sig).decode()


def verify(token, now=None, key=None):
    try:
        head, body, sig = token.encode().split(b".")
        good = _b64(hmac.new(key or secret(), head + b"." + body, hashlib.sha256).digest())
        if not hmac.compare_digest(sig, good):
            raise AuthError("bad signature")
        if json.loads(_unb64(head)).get("alg") != "HS256":
            raise AuthError("bad algorithm")
        claims = json.loads(_unb64(body))
    except AuthError:
        raise
    except Exception:
        raise AuthError("malformed token")
    if int(now if now is not None else time.time()) >= int(claims.get("exp", 0)):
        raise AuthError("token expired")
    sub = claims.get("sub")
    if not isinstance(sub, str) or not _USER.fullmatch(sub):
        raise AuthError("bad subject")
    return sub
