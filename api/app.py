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


from engine.safety import assess_policy, assess_symptoms
from engine.swap import swap_exercise
from engine.swap_checks import check_swap
from rag.search import DIFFICULTY_RANK, search_exercises
from state.profile import KNOWN_LIMITATIONS, UserProfile

_MEDIA = {}


def _media(ex_id):
    if "data" not in _MEDIA:
        try:
            _MEDIA["data"] = json.loads((Path(__file__).resolve().parent.parent / "data" / "exercise_media.json").read_text(encoding="utf-8-sig"))
        except Exception:
            _MEDIA["data"] = {}
    return _MEDIA["data"].get(ex_id)


def _ex_summary(e):
    return {"id": e.id, "name": e.name, "primary": list(e.primary), "equipment": list(e.equipment),
            "difficulty": e.difficulty, "movement": e.movement, "media": _media(e.id)}


def _ex_detail(e):
    secondary = [m for m in e.muscles if m not in e.primary]
    return {**e.model_dump(mode="json"), "secondary": secondary,
            "muscle_map": {"primary": list(e.primary), "secondary": secondary}, "media": _media(e.id)}


def _swap_context(profile):
    equip = list(profile.equipment) if profile is not None and profile.equipment is not None else []
    return equip, (list(profile.limitations) if profile is not None else []), (profile.experience if profile is not None else "beginner")


_SESSION_SQL = ("SELECT ps.id AS id, ps.uid AS uid, ps.week AS week, ps.day_number AS day_number, ps.session_number AS session_number, "
                "ps.planned_date AS planned_date, ps.status AS status, ps.completed_at AS completed_at, w.uid AS workout_uid "
                "FROM program_sessions ps JOIN programs p ON p.id = ps.program_id LEFT JOIN workouts w ON w.id = ps.workout_id "
                "WHERE ps.program_id = ? AND p.user_id = ? ORDER BY ps.week, ps.day_number, ps.session_number")


def _rows(s, program_id):
    return [dict(r) for r in s["programs"].conn.execute(_SESSION_SQL, (program_id, s["programs"].user_id)).fetchall()]


def _public(row):
    return {k: v for k, v in row.items() if k != "id"}


def _payload(s, latest=False):
    prog = s["programs"].latest_program() if latest else s["programs"].active_program()
    if prog is None:
        return {}
    rows = _rows(s, prog["id"])
    uid = s["programs"].conn.execute("SELECT uid FROM programs WHERE id = ?", (prog["id"],)).fetchone()[0]
    data = prog["program"].model_dump(mode="json")
    by = {e.id: e for e in exercises()}
    ids = {x["exercise_id"] for w in data["weeks"] for sess in w["sessions"] for x in sess["exercises"]}
    return {"program_uid": uid, "start_date": prog["start_date"], "status": prog["status"], "program": data,
            "sessions": [_public(r) for r in rows], "progress": s["programs"].progress(prog["id"]),
            "exercises": {i: _ex_summary(by[i]) for i in sorted(ids) if i in by}}


def _state(s):
    latest = s["programs"].latest_program()
    if latest is None:
        return {"progress": None, "next_session": None}
    nxt = s["programs"].next_session(latest["id"])
    uid = next((r["uid"] for r in _rows(s, latest["id"]) if nxt and r["id"] == nxt["id"]), None)
    return {"progress": s["programs"].progress(latest["id"]),
            "next_session": ({"uid": uid, "week": nxt["week"], "day_number": nxt["day_number"], "session_number": nxt["session_number"], "planned_date": nxt["planned_date"]} if nxt else None)}


def _detail(payload, row):
    wk = next((w for w in payload["program"]["weeks"] if w["number"] == row["week"]), {})
    sess = next((q for q in wk.get("sessions", []) if q["day_number"] == row["day_number"] and q.get("session_number", 1) == row["session_number"]), None) or {}
    return {**row, "weekday": sess.get("weekday"), "label": sess.get("label"), "muscles": sess.get("muscles", []),
            "est_minutes": sess.get("est_minutes"), "phase": wk.get("phase"), "effort": wk.get("effort"), "exercises": sess.get("exercises", [])}


@app.get("/v1/meta")
def v1_meta():
    from engine.health import DISCLAIMER, RED_FLAG_QUESTION
    from engine.taxonomy import MUSCLES
    props = UserProfile.model_json_schema().get("properties", {})
    lib = exercises()
    has_media = lambda e: bool((_media(e.id) or {}).get("video_url") or (_media(e.id) or {}).get("image_urls"))
    return {"goals": props.get("goal", {}).get("enum"), "experience": props.get("experience", {}).get("enum"),
            "screening": props.get("screening", {}).get("enum"), "equipment": sorted({q for e in lib for q in e.equipment}),
            "limitations": sorted(KNOWN_LIMITATIONS), "muscles": list(MUSCLES), "difficulties": list(DIFFICULTY_RANK),
            "exercise_count": len(lib), "media_available": sum(1 for e in lib if has_media(e)), "profile_fields": props,
            "disclaimer": DISCLAIMER, "health_question": RED_FLAG_QUESTION, "days_per_week": {"min": 1, "max": 7}}


@app.get("/v1/exercises")
def v1_exercises(muscle: str | None = None, equipment: str | None = None, max_difficulty: str | None = None,
                 q: str | None = None, limit: int = 200, user: str = Depends(current_user)):
    if not 1 <= limit <= 500:
        raise HTTPException(422, "limit must be 1 to 500")
    if max_difficulty and max_difficulty.lower() not in DIFFICULTY_RANK and max_difficulty.lower() != "any":
        raise HTTPException(422, "unknown difficulty")
    if q and len(q) > 60:
        raise HTTPException(422, "q is too long")
    equip = None if equipment is None else [x.strip() for x in equipment.split(",") if x.strip()]
    found = search_exercises(exercises(), muscle=muscle, available_equipment=equip, max_difficulty=max_difficulty, query=q, limit=limit)
    return {"count": len(found), "exercises": [_ex_summary(e) for e in found]}


@app.get("/v1/exercises/{exercise_id}")
def v1_exercise(exercise_id: str, user: str = Depends(current_user)):
    e = next((x for x in exercises() if x.id == exercise_id), None)
    if e is None:
        raise HTTPException(404, "unknown exercise")
    return _ex_detail(e)


@app.get("/v1/exercises/{exercise_id}/alternatives")
def v1_alternatives(exercise_id: str, user: str = Depends(current_user)):
    lib = exercises()
    if not any(x.id == exercise_id for x in lib):
        raise HTTPException(404, "unknown exercise")
    s = stores(user)
    try:
        profile = s["profile"].get()
    finally:
        close(s)
    equip, lim, level = _swap_context(profile)
    found = [e for e in lib if e.id != exercise_id and not check_swap({exercise_id}, e.id, lib, equip, lim, level)]
    found.sort(key=lambda e: (DIFFICULTY_RANK[e.difficulty], e.name))
    return {"exercise_id": exercise_id, "using_profile": profile is not None, "alternatives": [_ex_summary(e) for e in found[:20]]}


class BuildIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    days: int | None = Field(default=None, ge=1, le=7)
    today: date | None = None


@app.post("/v1/program")
def v1_build_program(body: BuildIn, user: str = Depends(current_user)):
    from agent.coach import start_program
    s = stores(user)
    try:
        profile = s["profile"].get()
        if profile is None:
            raise HTTPException(409, "create your profile first")
        r = start_program(exercises(), profile, s["programs"], _client_today(body.today), body.days)
        out = {"status": r["status"], "reply": r["reply"]}
        if r["status"] in ("ok", "active"):
            out.update(_payload(s))
        return out
    finally:
        close(s)


@app.get("/v1/program")
def v1_program(user: str = Depends(current_user)):
    s = stores(user)
    try:
        p = _payload(s, latest=True)
        if not p:
            raise HTTPException(404, "no programme")
        return p
    finally:
        close(s)


@app.get("/v1/program/explain")
def v1_explain(user: str = Depends(current_user)):
    from engine.explain import explain_programme
    s = stores(user)
    try:
        latest = s["programs"].latest_program()
        if latest is None:
            raise HTTPException(404, "no programme")
        return {"text": explain_programme(latest["program"], exercises(), s["profile"].get())}
    finally:
        close(s)


@app.get("/v1/today")
def v1_today(today: date | None = None, user: str = Depends(current_user)):
    t = _client_today(today).isoformat()
    s = stores(user)
    try:
        p = _payload(s)
        if not p:
            raise HTTPException(404, "no active programme")
        planned = [r for r in p["sessions"] if r["status"] == "planned"]
        late = [r for r in planned if r["planned_date"] < t]
        return {"date": t, "today": [_detail(p, r) for r in planned if r["planned_date"] == t],
                "overdue": [_detail(p, r) for r in late[:5]], "overdue_count": len(late),
                "next_session": _detail(p, planned[0]) if planned else None, "progress": p["progress"], "exercises": p["exercises"]}
    finally:
        close(s)


class SessionActionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_uid: str = Field(min_length=32, max_length=32, pattern=r"^[0-9a-f]{32}$")
    today: date | None = None


class DayIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    today: date | None = None


def _require_next(s, uid):
    active = s["programs"].active_program()
    if active is None:
        raise HTTPException(409, "no active programme")
    nxt = s["programs"].next_session(active["id"])
    cur = next((r["uid"] for r in _rows(s, active["id"]) if nxt and r["id"] == nxt["id"]), None)
    if cur != uid:
        raise HTTPException(409, "that is not the next planned session; sync your session statuses instead")


@app.post("/v1/session/finish")
def v1_finish(body: SessionActionIn, user: str = Depends(current_user)):
    from agent.coach import finish_session
    s = stores(user)
    try:
        profile = s["profile"].get()
        if profile is None:
            raise HTTPException(409, "create your profile first")
        _require_next(s, body.session_uid)
        text = finish_session(exercises(), profile, s["programs"], s["history"], _client_today(body.today))
        return {"status": "ok", "reply": text, **_state(s)}
    finally:
        close(s)


@app.post("/v1/session/skip")
def v1_skip(body: SessionActionIn, user: str = Depends(current_user)):
    from agent.coach import skip_session
    s = stores(user)
    try:
        _require_next(s, body.session_uid)
        return {"status": "ok", "reply": skip_session(s["programs"]), **_state(s)}
    finally:
        close(s)


@app.post("/v1/session/undo")
def v1_undo(body: DayIn, user: str = Depends(current_user)):
    from agent.coach import reopen_last
    s = stores(user)
    try:
        return {"status": "ok", "reply": reopen_last(s["programs"]), **_state(s)}
    finally:
        close(s)


class SwapIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    exercise_id: str = Field(min_length=1, max_length=64)
    replacement_id: str = Field(min_length=1, max_length=64)


@app.post("/v1/program/swap")
def v1_swap(body: SwapIn, user: str = Depends(current_user)):
    s = stores(user)
    try:
        profile, active = s["profile"].get(), s["programs"].active_program()
        if profile is None or active is None:
            raise HTTPException(409, "you need a profile and an active programme")
        nxt = s["programs"].next_session(active["id"])
        if nxt is None:
            raise HTTPException(409, "no sessions left to change")
        lib = exercises()
        equip, lim, level = _swap_context(profile)
        problems = check_swap({body.exercise_id}, body.replacement_id, lib, equip, lim, level)
        if problems:
            raise HTTPException(422, {"problems": problems})
        new, count = swap_exercise(active["program"], {body.exercise_id}, body.replacement_id, level, lib, (nxt["week"], nxt["day_number"] - 1))
        if count == 0:
            raise HTTPException(409, "that exercise is not in your remaining sessions")
        s["programs"].update_program(active["id"], new)
        return {"status": "ok", "changes": count, **_payload(s)}
    finally:
        close(s)


@app.get("/v1/reminders")
def v1_reminders(today: date | None = None, user: str = Depends(current_user)):
    from agent.coach import current_reminders
    s = stores(user)
    try:
        latest = s["programs"].latest_program()
        uids = {r["id"]: r["uid"] for r in _rows(s, latest["id"])} if latest else {}
        items = current_reminders(s["programs"], _client_today(today))
        return {"reminders": [{"kind": r.kind, "message": r.message, "session_uid": uids.get(r.session_id)} for r in items]}
    finally:
        close(s)


@app.get("/v1/history")
def v1_history(limit: int = 10, user: str = Depends(current_user)):
    if not 1 <= limit <= 100:
        raise HTTPException(422, "limit must be 1 to 100")
    s = stores(user)
    try:
        out = []
        for w in s["history"].recent_workouts(limit):
            uid = s["history"].conn.execute("SELECT uid FROM workouts WHERE id = ?", (w["id"],)).fetchone()[0]
            out.append({"uid": uid, **{k: v for k, v in w.items() if k != "id"}})
        return {"workouts": out}
    finally:
        close(s)


@app.get("/v1/history/exercise/{exercise_id}")
def v1_exercise_history(exercise_id: str, limit: int = 5, user: str = Depends(current_user)):
    if not 1 <= limit <= 50:
        raise HTTPException(422, "limit must be 1 to 50")
    s = stores(user)
    try:
        return {"exercise_id": exercise_id, "sessions": [{"performed_at": x["performed_at"], "sets": [t.model_dump(mode="json") for t in x["sets"]]}
                                                         for x in s["history"].sessions_for(exercise_id, limit)]}
    finally:
        close(s)


class SafetyIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=1000)


@app.post("/v1/safety/check")
def v1_safety(body: SafetyIn, user: str = Depends(current_user)):
    a = assess_symptoms(body.text)
    if a.level != "none":
        return {"level": a.level, "policy": None, "reply": a.reply}
    kind, reply = assess_policy(body.text)
    return {"level": "none", "policy": kind, "reply": reply}
