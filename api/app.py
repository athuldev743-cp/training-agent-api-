"""HTTP API (Step 11/10 contract). The user id always comes from the verified token."""
import json, os, re, time
from datetime import date
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from api.auth import AuthError, verify
from state.history import HistoryStore
from state.programs import ProgramStore
from state.store import ProfileStore
from state.sync import apply_changes, export_changes
from state.traces import TraceStore

MAX_BODY = 2_000_000
DB = Path(os.getenv("TRAINING_DB") or Path(__file__).resolve().parent.parent / "data" / "server.db")
app = FastAPI(title="Local Training Agent API", version="1")
app.state.llm = None
app.state.exercises = None


class ChatIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=1000)
    today: date | None = None


class SyncIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    workouts: list = Field(default_factory=list, max_length=5000)
    programs: list = Field(default_factory=list, max_length=5000)
    deleted: list = Field(default_factory=list, max_length=5000)


@app.middleware("http")
async def limit_body(request: Request, call_next):
    if int(request.headers.get("content-length") or 0) > MAX_BODY:
        return JSONResponse({"error": "request body too large"}, status_code=413)
    return await call_next(request)


@app.exception_handler(Exception)
async def unhandled(request, exc):
    return JSONResponse({"error": "internal error"}, status_code=500)  # no stack trace, no details


def current_user(authorization: str | None = Header(default=None)) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    try:
        return verify(authorization[7:])
    except AuthError:
        raise HTTPException(401, "invalid or expired token")


def stores(user):
    return {"history": HistoryStore(DB, user_id=user), "programs": ProgramStore(DB, user_id=user),
            "profile": ProfileStore(DB, user_id=user), "traces": TraceStore(DB, user_id=user)}


def close(s):
    for x in s.values():
        x.close()


def exercises():
    if app.state.exercises is None:
        from rag.store import load_exercises
        app.state.exercises = load_exercises()
    return app.state.exercises


def model():
    if app.state.llm is None:
        if os.getenv("LLM_MODE", "cloud") == "local":
            from agent.cloud_llm import HybridLLM
            from agent.local_model import load_model
            app.state.llm = HybridLLM(load_model())
        else:
            from agent.cloud_llm import CloudOnlyLLM
            app.state.llm = CloudOnlyLLM()
    return app.state.llm


@app.get("/health")
def health():
    return {"status": "ok", "time": time.strftime("%Y-%m-%dT%H:%M:%S")}


@app.get("/v1/sync")
def sync_down(user: str = Depends(current_user)):
    s = stores(user)
    try:
        return export_changes(s["history"].conn, user)
    finally:
        close(s)


@app.post("/v1/sync")
def sync_up(body: SyncIn, user: str = Depends(current_user)):
    s = stores(user)
    try:
        return apply_changes(s["history"].conn, user, body.model_dump())
    except ValueError as e:
        raise HTTPException(422, str(e))
    finally:
        close(s)


def run_chat(user, text, today=None):
    from agent.dispatcher import handle_message
    s = stores(user)
    try:
        r = handle_message(model(), text, exercises(), s["profile"].get(), s["programs"], today or date.today(), s["history"])
        out = {"status": r.get("status"), "tool": r.get("tool"), "reply": r.get("reply"), "confirm_token": None}
        pending = (r.get("data") or {}).get("pending") if r.get("status") == "confirm" else None
        if isinstance(pending, dict) and pending.get("tool") in CONFIRMABLE:
            out["confirm_token"] = issue_confirm(user, pending, _marker(s["programs"]))
            out["confirm_expires_in"] = 300
        return out
    finally:
        close(s)


@app.post("/v1/chat")
def chat(body: ChatIn, user: str = Depends(current_user)):
    return run_chat(user, body.message, _client_today(body.today))


_SENT = re.compile(r"(?<=[.!?])\s+")


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/v1/chat/stream")
def chat_stream(body: ChatIn, user: str = Depends(current_user)):
    result = run_chat(user, body.message, _client_today(body.today))

    def gen():
        # safety / scope / plans / tools are deterministic and already validated: one complete event.
        # Plain chat text is delivered sentence by sentence so voice output can start early.
        if result["status"] == "ok" and result["tool"] in (None, "general_chat"):
            for part in _SENT.split(result["reply"] or ""):
                if part.strip():
                    yield sse("sentence", {"text": part.strip()})
        else:
            yield sse("message", {"text": result["reply"], "status": result["status"], "confirm_token": result.get("confirm_token")})
        yield sse("done", {"status": result["status"]})

    return StreamingResponse(gen(), media_type="text/event-stream")


import collections, hashlib, threading
from state.account import delete_account, export_account
from state.profile import UserProfile

RATE = {"limit": int(os.getenv("API_RATE_LIMIT", "120")), "auth_limit": int(os.getenv("API_AUTH_RATE_LIMIT", "10")), "window": 60.0}
_hits = collections.defaultdict(collections.deque)
_rl = threading.Lock()


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.url.path in ("/health", "/ready"):
        return await call_next(request)
    who = None
    h = request.headers.get("authorization") or ""
    if h.startswith("Bearer "):
        try:
            who = "u:" + verify(h[7:])
        except AuthError:
            who = None
    auth_path = request.url.path.startswith("/v1/auth/")
    key = ("auth:" + client_ip(request)) if auth_path else (who or "ip:" + client_ip(request))
    limit = RATE["auth_limit"] if auth_path else RATE["limit"]
    now = time.monotonic()
    with _rl:
        if len(_hits) > 10000:
            for k in [k for k, q in _hits.items() if not q or now - q[-1] > RATE["window"]]:
                del _hits[k]
        q = _hits[key]
        while q and now - q[0] > RATE["window"]:
            q.popleft()
        if len(q) >= limit:
            return JSONResponse({"error": "rate limit exceeded"}, status_code=429, headers={"Retry-After": "60"})
        q.append(now)
    return await call_next(request)


class ProfileIn(UserProfile):
    model_config = ConfigDict(extra="forbid")


class DeleteIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confirm: str


@app.get("/v1/profile")
def get_profile(user: str = Depends(current_user)):
    s = stores(user)
    try:
        p = s["profile"].get()
    finally:
        close(s)
    if p is None:
        raise HTTPException(404, "no profile")
    return p.model_dump()


@app.put("/v1/profile")
def put_profile(body: ProfileIn, user: str = Depends(current_user)):
    s = stores(user)
    try:
        s["profile"].save(UserProfile(**body.model_dump()))
        return {"status": "saved"}
    finally:
        close(s)


@app.get("/v1/account/export")
def account_export(user: str = Depends(current_user)):
    s = stores(user)
    try:
        return export_account(s["history"].conn, user)
    finally:
        close(s)


@app.delete("/v1/account")
def account_delete(body: DeleteIn, user: str = Depends(current_user)):
    if body.confirm != "DELETE":
        raise HTTPException(422, 'send {"confirm": "DELETE"} to delete the account')
    s = stores(user)
    try:
        return {"deleted": delete_account(s["history"].conn, user)}
    finally:
        close(s)


@app.get("/ready")
def ready():
    from agent.cloud_llm import PROVIDERS
    checks = {"token_secret": len(os.getenv("API_TOKEN_SECRET", "")) >= 32,
              "providers_configured": [n for n, _, k, _, _ in PROVIDERS if os.getenv(k)],
              "state_backend": os.getenv("STATE_BACKEND", "sqlite")}
    good = checks["token_secret"] and bool(checks["providers_configured"])
    return JSONResponse({"ready": good, **checks}, status_code=200 if good else 503)


def client_ip(request):
    """Client address. X-Forwarded-For is trusted only when TRUST_PROXY=1 (set it only behind a proxy you control)."""
    if os.getenv("TRUST_PROXY") == "1":
        fwd = request.headers.get("x-forwarded-for", "")
        if fwd:
            return fwd.split(",")[0].strip()[:64]
    return request.client.host if request.client else "unknown"


from api.auth import TTL_S, issue
from state.accounts import AccountError, AccountStore

_AUTH_STATUS = {"invalid": 422, "exists": 409, "bad_credentials": 401, "locked": 429, "bad_token": 401}


class CredsIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=128)


class RefreshIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    refresh_token: str = Field(min_length=20, max_length=200)


def _token_response(store, uid, refresh=None):
    return {"user_id": uid, "token_type": "bearer", "expires_in": TTL_S, "access_token": issue(uid),
            "refresh_token": refresh or store.issue_refresh(uid)}


def _auth_call(fn):
    store = AccountStore(DB)
    try:
        return fn(store)
    except AccountError as e:
        raise HTTPException(_AUTH_STATUS.get(e.kind, 400), str(e))
    finally:
        store.close()


@app.post("/v1/auth/register", status_code=201)
def auth_register(body: CredsIn):
    return _auth_call(lambda s: _token_response(s, s.register(body.email, body.password)))


@app.post("/v1/auth/login")
def auth_login(body: CredsIn):
    return _auth_call(lambda s: _token_response(s, s.login(body.email, body.password)))


@app.post("/v1/auth/refresh")
def auth_refresh(body: RefreshIn):
    def go(s):
        uid, new = s.rotate_refresh(body.refresh_token)
        return _token_response(s, uid, refresh=new)
    return _auth_call(go)


@app.post("/v1/auth/logout")
def auth_logout(body: RefreshIn):
    _auth_call(lambda s: s.revoke(body.refresh_token))
    return {"status": "ok"}


from datetime import timedelta
from api.auth import issue_confirm, verify_confirm

CONFIRMABLE = {"build_program", "finish_session", "skip_session"}


def _marker(programs):
    """Where the user's programme stands: (active programme id, next planned session id). A confirmation is only valid in this exact state."""
    active = programs.active_program()
    if active is None:
        return [None, None]
    nxt = programs.next_session(active["id"])
    return [active["id"], nxt["id"] if nxt else None]


def _client_today(d):
    if d is None:
        return date.today()
    if abs((d - date.today()).days) > 2:
        raise HTTPException(422, "today must be within 2 days of the server date")
    return d


class ConfirmIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    token: str = Field(min_length=20, max_length=2000)
    accept: bool
    today: date | None = None


@app.post("/v1/chat/confirm")
def chat_confirm(body: ConfirmIn, user: str = Depends(current_user)):
    try:
        action, marker = verify_confirm(body.token, user)
    except AuthError:
        raise HTTPException(400, "invalid or expired confirmation")
    if not isinstance(action, dict) or action.get("tool") not in CONFIRMABLE:
        raise HTTPException(400, "invalid or expired confirmation")
    today = _client_today(body.today)
    if not body.accept:
        return {"status": "cancelled", "tool": action["tool"], "reply": "Okay, I cancelled that. Nothing was changed."}
    from agent.coach_tools import run_coach_tool
    s = stores(user)
    try:
        if _marker(s["programs"]) != marker:
            raise HTTPException(409, "your plan changed since I asked, so please ask again")
        r = run_coach_tool(action, exercises(), s["profile"].get(), s["programs"], s["history"], today, confirmed=True)
        return {"status": r.get("status"), "tool": r.get("tool"), "reply": r.get("reply")}
    finally:
        close(s)
