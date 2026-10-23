"""Deterministic 4-week programme builder (no model involved, so the same inputs always give the same programme).
Weeks 1-2 use only foundation-tier exercises; weeks 3-4 add the build tier. Everything respects the
user's level, equipment, limitations and health check.
Quality rules: one exercise per movement pattern and main muscle in a session; beginners start on the lower rung of a
ladder, intermediate and advanced users on a higher rung, and everyone moves up a rung in the build weeks."""
import json
from pathlib import Path

from pydantic import BaseModel

from engine.equipment_hints import unlock_hint
from engine.health import DISCLAIMER, screening_gate
from engine.prescription import Prescription, load_templates, prescription_for
from engine.taxonomy import SPLITS, day_muscles
from engine.validator import SECONDS_PER_REP
from rag.search import DIFFICULTY_RANK
from rag.store import Exercise, safe_exercises
from state.profile import KNOWN_EQUIPMENT, UserProfile

TIER_PATH = Path(__file__).resolve().parent.parent / "data" / "exercise_tiers.json"

# Compound-heavy muscles first. ANCHORS are trained in every full-body session.
PRIORITY = ["chest", "back", "quads", "hamstrings", "glutes", "core", "shoulders", "triceps", "biceps", "calves"]
ANCHORS = ["chest", "back", "quads"]
MAX_EXERCISES = {"beginner": 6, "intermediate": 7, "advanced": 8}
MAX_PER_MUSCLE = 4
DAY_RANGE = {"beginner": (2, 3), "intermediate": (3, 4), "advanced": (4, 5)}
DEFAULT_DAYS = {"beginner": 3, "intermediate": 4, "advanced": 4}
WEEKDAYS = {
    2: ["Monday", "Thursday"],
    3: ["Monday", "Wednesday", "Friday"],
    4: ["Monday", "Tuesday", "Thursday", "Friday"],
    5: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
}
EFFORT = {
    1: "RPE 6-7 (about 3 reps left in the tank)",
    2: "RPE 7 (about 3 reps left)",
    3: "RPE 7-8 (2-3 reps left)",
    4: "RPE 8 (about 2 reps left)",
}
SPLIT_NAMES = {"full_body": "full body", "upper_lower": "upper/lower",
               "push_pull_legs": "push/pull/legs", "bro_split": "body-part split"}


class ProgramExercise(BaseModel):
    exercise_id: str
    name: str
    tier: str
    unit: str
    sets: int
    rep_min: int
    rep_max: int
    aim_low: int
    aim_high: int
    rest_seconds: int


class ProgramSession(BaseModel):
    day_number: int
    session_number: int = 1
    weekday: str
    label: str
    muscles: list[str]
    exercises: list[ProgramExercise]
    est_minutes: float


class ProgramWeek(BaseModel):
    number: int
    phase: str
    effort: str
    sessions: list[ProgramSession]


class Program(BaseModel):
    level: str
    split: str
    days_per_week: int
    goal: str = "general"
    equipment: list[str]
    limitations: list[str]
    weeks: list[ProgramWeek]
    notes: list[str]


def load_tiers(path: Path = TIER_PATH) -> dict[str, str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def choose_split(level: str, days: int | None) -> tuple[str, int, str | None]:
    """Returns (split name, days actually used, optional note when the request was adjusted)."""
    lo, hi = DAY_RANGE[level]
    wanted = DEFAULT_DAYS[level] if days is None else days
    used = max(lo, min(hi, wanted))
    note = None
    if used != wanted:
        note = (f"{wanted} days a week is outside the usual range for a {level} ({lo}-{hi}), "
                f"so I used {used}.")
    if level == "beginner":
        name = "full_body"
    elif used == 3:
        name = "push_pull_legs"
    elif used == 4:
        name = "upper_lower"
    else:
        name = "bro_split"
    return name, used, note


def allowed_exercises(exercises, level, equipment, limitations, phase, tiers) -> list[Exercise]:
    rank = DIFFICULTY_RANK[level]
    wanted = {"foundation"} if (phase == "foundation" and level == "beginner") else {"foundation", "build"}
    have = set(KNOWN_EQUIPMENT) if equipment is None else set(equipment) | {"bodyweight"}
    return [
        e for e in safe_exercises(exercises, limitations)
        if DIFFICULTY_RANK[e.difficulty] <= rank and tiers[e.id] in wanted and set(e.equipment) <= have
    ]


def exercise_families(exercises: list[Exercise]) -> dict[str, str]:
    """Groups exercises into families: everything connected by regression / progression links
    (a ladder such as knee push-up, push-up, decline push-up). Returns exercise id -> family id."""
    parent = {e.id: e.id for e in exercises}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in exercises:
        for other in (e.regression, e.progression):
            if other in parent:
                parent[find(e.id)] = find(other)
    return {e.id: find(e.id) for e in exercises}


def ladder_depth(ex: Exercise, by_id: dict) -> int:
    """How many easier steps sit below this exercise in its regression chain (0 = bottom rung)."""
    depth = 0
    current = ex
    while current.regression is not None and current.regression in by_id and depth < 10:
        current = by_id[current.regression]
        depth += 1
    return depth


def _redundant(e: Exercise, chosen: list[Exercise], families: dict[str, str]) -> bool:
    """Same movement pattern and a shared main muscle as something already in the session."""
    return any(c.movement == e.movement and bool(set(c.primary) & set(e.primary)) for c in chosen)


def _day_muscle_order(groups: list[str], day_number: int, cap: int) -> list[str]:
    """Muscles of a day, most important first. Days with too many muscles keep the anchors
    and rotate the rest from day to day, so a whole week still covers everything."""
    muscles = sorted(day_muscles(groups), key=PRIORITY.index)
    if len(muscles) <= cap:
        return muscles
    anchors = [m for m in ANCHORS if m in muscles]
    rest = [m for m in muscles if m not in anchors]
    slots = max(1, cap - len(anchors))
    shift = ((day_number - 1) * slots) % len(rest)
    return anchors + rest[shift:] + rest[:shift]


def _sets_for_week(rx: Prescription, week: int) -> int:
    if week <= 2:
        return rx.sets_start
    if week == 3:
        return min(rx.sets_start + 1, rx.sets_max)
    return rx.sets_max


def _aim_range(rx: Prescription, week: int) -> tuple[int, int]:
    mid = (rx.rep_min + rx.rep_max) // 2
    return (rx.rep_min, mid) if week in (1, 3) else (mid, rx.rep_max)


def _exercise_minutes(rx: Prescription, sets: int) -> float:
    mid = (rx.rep_min + rx.rep_max) / 2
    work = mid if rx.unit == "seconds" else mid * SECONDS_PER_REP
    return sets * (work + rx.rest_seconds) / 60


def _pick(muscle, pool, chosen, used_patterns, owned, phase, tiers, level, by_id, families):
    candidates = [e for e in pool if muscle in e.primary and not _redundant(e, chosen, families)]
    if not candidates:
        return None

    def key(e):
        depth = ladder_depth(e, by_id)
        # Foundation weeks: beginners start on the lower rung, everyone else higher.
        # Build weeks: everyone moves up to the highest rung they are allowed.
        placement = depth if (phase == "foundation" and level == "beginner") else -depth
        return (
            e.movement == "isolation",                                  # compound lifts first
            0 if set(e.equipment) & owned else 1,                       # use the equipment they own
            0 if (phase == "build" and tiers[e.id] == "build") else 1,  # build weeks: harder tier first
            e.movement in used_patterns,                                # variety within the session
            placement,
            -DIFFICULTY_RANK[e.difficulty] if phase == "build" else 0,
            e.name,
        )

    return min(candidates, key=key)


def _build_block(phase, split_days, level, pool, owned, budget, cap, tiers, templates, goal,
                 by_id, families, day_caps=None) -> dict[int, list[Exercise]]:
    last_week = 2 if phase == "foundation" else 4  # the block's heaviest week must still fit the budget
    block: dict[int, list[Exercise]] = {}
    seen_by_label: dict[str, set[str]] = {}
    for day_number, groups in enumerate(split_days, 1):
        order = _day_muscle_order(groups, day_number, cap)
        prior = seen_by_label.get(groups[0], set())  # exercises used on earlier days of the same kind
        day_cap = cap if not day_caps else min(cap, max(1, len(day_caps.get(day_number, [])) or cap))
        chosen: list[Exercise] = []
        patterns: set[str] = set()
        per_muscle: dict[str, int] = {}
        minutes = 0.0

        def try_add(muscle: str) -> bool:
            nonlocal minutes
            if len(chosen) >= day_cap or per_muscle.get(muscle, 0) >= MAX_PER_MUSCLE:
                return False
            fresh = [e for e in pool if e.id not in prior]
            ex = (_pick(muscle, fresh, chosen, patterns, owned, phase, tiers, level, by_id, families) if prior else None) \
                or _pick(muscle, pool, chosen, patterns, owned, phase, tiers, level, by_id, families)
            if ex is None:
                return False
            rx = prescription_for(ex.id, level, templates, goal)
            cost = _exercise_minutes(rx, _sets_for_week(rx, last_week))
            if minutes + cost > budget:
                return False
            chosen.append(ex)
            patterns.add(ex.movement)
            per_muscle[muscle] = per_muscle.get(muscle, 0) + 1
            minutes += cost
            return True

        while True:
            added = False
            for muscle in order:
                if try_add(muscle):
                    added = True
            if not added:
                break
        # Core finisher (Step 40): close a session that already has other work but no core work, if the cap and the time ceiling allow.
        if chosen and "core" not in {m for e in chosen for m in e.primary} and len(chosen) < day_cap:
            ex = _pick("core", pool, chosen, patterns, owned, phase, tiers, level, by_id, families)
            if ex is not None:
                rx = prescription_for(ex.id, level, templates, goal)
                cost = _exercise_minutes(rx, _sets_for_week(rx, last_week))
                if minutes + cost <= budget:
                    chosen.append(ex)
                    patterns.add(ex.movement)
                    minutes += cost
        # Core finisher fallback (Step 40): when the preferred core exercise did not fit the time, try the next best that does.
        if chosen and "core" not in {m for e in chosen for m in e.primary} and len(chosen) < day_cap:
            remaining = list(pool)
            while remaining:
                ex = _pick("core", remaining, chosen, patterns, owned, phase, tiers, level, by_id, families)
                if ex is None:
                    break
                rx = prescription_for(ex.id, level, templates, goal)
                cost = _exercise_minutes(rx, _sets_for_week(rx, last_week))
                if minutes + cost <= budget:
                    chosen.append(ex)
                    patterns.add(ex.movement)
                    minutes += cost
                    break
                remaining = [e for e in remaining if e.id != ex.id]
        seen_by_label.setdefault(groups[0], set()).update(e.id for e in chosen)
        block[day_number] = chosen
    return block


def _coverage_notes(exercises, level, equipment, limitations, muscles, tiers) -> list[str]:
    safe_pool = allowed_exercises(exercises, level, equipment, limitations, "build", tiers)
    open_pool = allowed_exercises(exercises, level, equipment, [], "build", tiers)
    areas = ", ".join(a.replace("_", " ") for a in limitations)
    notes = []
    for m in sorted(muscles, key=PRIORITY.index):
        if any(m in e.primary for e in safe_pool):
            continue
        if any(m in e.primary for e in open_pool):
            notes.append(f"Your limitations ({areas}) removed every exercise that mainly trains {m}.")
        else:
            notes.append(unlock_hint(exercises, m, equipment, level))
    return notes


def _label(split: str, groups: list[str]) -> str:
    if split == "full_body":
        return "Full body"
    return " + ".join(g.replace("_", " ") for g in groups).capitalize()


def build_program(exercises: list[Exercise], profile: UserProfile, days_per_week: int | None = None) -> dict:
    gate = screening_gate(profile)
    if gate:
        return {"status": "blocked_health", "reply": gate, "program": None}

    from engine.profile_checks import check_profile
    _problem = check_profile(profile)
    if _problem:
        return {"status": "needs_info", "reply": _problem, "program": None}
    level = profile.experience
    notes: list[str] = []
    equipment = profile.equipment
    if equipment is None:
        equipment = []
        notes.append("Your profile does not say what equipment you have, so I assumed bodyweight only.")
    split, used_days, day_note = choose_split(level, days_per_week)
    if day_note:
        notes.append(day_note)

    tiers = load_tiers()
    templates = load_templates()
    by_id = {e.id: e for e in exercises}
    families = exercise_families(exercises)
    cap = MAX_EXERCISES[level]
    goal = profile.goal
    split_days = SPLITS[split][:used_days]
    owned = set(equipment) - {"bodyweight"}

    blocks = {}
    for phase in ("foundation", "build"):
        pool = allowed_exercises(exercises, level, equipment, profile.limitations, phase, tiers)
        blocks[phase] = _build_block(phase, split_days, level, pool, owned, profile.session_minutes,
                                     cap, tiers, templates, goal, by_id, families, blocks.get("foundation"))
    if not all(len(day) > 0 for block in blocks.values() for day in block.values()):
        return {"status": "no_exercises", "program": None,
                "reply": "I couldn't build a full programme with your equipment, level and limitations: some training days would have no exercises. A doctor or physiotherapist can tell you which exercises suit you."}

    weeks = []
    for week in (1, 2, 3, 4):
        phase = "foundation" if week <= 2 else "build"
        sessions = []
        for day_number, groups in enumerate(split_days, 1):
            items = []
            minutes = 0.0
            for ex in blocks[phase][day_number]:
                rx = prescription_for(ex.id, level, templates, goal)
                sets = _sets_for_week(rx, week)
                lo, hi = _aim_range(rx, week)
                minutes += _exercise_minutes(rx, sets)
                items.append(ProgramExercise(
                    exercise_id=ex.id, name=ex.name, tier=tiers[ex.id], unit=rx.unit, sets=sets,
                    rep_min=rx.rep_min, rep_max=rx.rep_max, aim_low=lo, aim_high=hi,
                    rest_seconds=rx.rest_seconds))
            sessions.append(ProgramSession(
                day_number=day_number, weekday=WEEKDAYS[used_days][day_number - 1],
                label=_label(split, groups), muscles=_day_muscle_order(groups, day_number, cap),
                exercises=items, est_minutes=round(minutes, 1)))
        weeks.append(ProgramWeek(number=week, phase=phase, effort=EFFORT[week], sessions=sessions))

    split_muscles = set().union(*(day_muscles(d) for d in split_days))
    notes += _coverage_notes(exercises, level, equipment, profile.limitations, split_muscles, tiers)
    covered = {m for wk in weeks for s in wk.sessions for e in s.exercises for m in by_id[e.exercise_id].primary}
    explained = " ".join(notes).lower()
    for m in sorted(split_muscles - covered, key=PRIORITY.index):
        name = m.replace("_", " ")
        if name not in explained:
            notes.append(f"{name.capitalize()} did not fit: your {profile.session_minutes}-minute sessions were "
                         "filled by other muscles first. A longer session would make room for it.")
    avg_minutes = sum(s.est_minutes for s in weeks[0].sessions) / len(weeks[0].sessions)
    if avg_minutes < 0.5 * profile.session_minutes:
        notes.append(f"Your sessions are about {avg_minutes:.0f} minutes, shorter than the {profile.session_minutes} you set, because I cap a {level} session at {cap} exercises. Use the spare time for warm-up and mobility." if any(len(s.exercises) >= cap for wk in weeks for s in wk.sessions) else f"Your sessions are about {avg_minutes:.0f} minutes, shorter than the {profile.session_minutes} you set, because few exercises in my library match your level and equipment. More equipment would unlock longer sessions.")
    peak = max(s.est_minutes for wk in weeks for s in wk.sessions)
    capped = any(len(s.exercises) >= cap for wk in weeks for s in wk.sessions)
    if capped and avg_minutes >= 0.5 * profile.session_minutes and peak < 0.75 * profile.session_minutes:
        notes.append(f"Sessions top out at about {peak:.0f} minutes: I cap a {level} session at {cap} exercises, so the rest of your {profile.session_minutes} minutes is unused. Spend it on warm-up and mobility.")
    notes.append("After week 4, take an easier week (fewer sets, easy effort) before starting the next programme.")

    program = Program(goal=profile.goal, level=level, split=split, days_per_week=used_days, equipment=list(equipment),
                      limitations=list(profile.limitations), weeks=weeks, notes=notes)
    return {"status": "ok", "reply": format_overview(program), "program": program}


def format_overview(program: Program) -> str:
    days = ", ".join(s.weekday for s in program.weeks[0].sessions)
    equipment = ", ".join(e.replace("_", " ") for e in program.equipment) or "bodyweight only"
    avoid = ", ".join(a.replace("_", " ") for a in program.limitations) or "nothing"
    lines = [
        f"Your 4-week {SPLIT_NAMES[program.split]} programme: {program.days_per_week} days a week ({days}).",
        f"Level: {program.level}. Goal: {program.goal.replace('_', ' ')}. Equipment: {equipment}. Avoiding exercises that load: {avoid}.",
        "Weeks 1-2: lower effort and fewer sets (beginners also stay on foundation exercises). Weeks 3-4: harder variations and more sets.",
        "Control every rep: about 2 seconds down and 1 second up, no bouncing.",
    ]
    lines += [f"Note: {n}" for n in program.notes]
    lines.append(DISCLAIMER)
    return "\n".join(lines)


def format_week(program: Program, week_number: int) -> str:
    wk = program.weeks[week_number - 1]
    lines = [f"WEEK {wk.number} ({wk.phase}) - effort: {wk.effort}"]
    for s in wk.sessions:
        lines.append(f"  Day {s.day_number} ({s.weekday}) - {s.label}, about {s.est_minutes:g} min")
        for i, e in enumerate(s.exercises, 1):
            u = " s" if e.unit == "seconds" else ""
            lines.append(f"    {i}. {e.name}: {e.sets} sets x {e.aim_low}-{e.aim_high}{u} "
                         f"(range {e.rep_min}-{e.rep_max}{u}), rest {e.rest_seconds} s")
    return "\n".join(lines)


# --- Plan upgrade: honour the requested number of days (2 to 6) ---
DAY_RANGE = {_lv: (2, 6) for _lv in DAY_RANGE}
WEEKDAYS = {**WEEKDAYS, 6: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]}
SPLITS = {**SPLITS, "push_pull_legs": list(SPLITS["push_pull_legs"]) * 2}


def choose_split(level: str, days: int | None) -> tuple[str, int, str | None]:
    # Returns (split name, days used, optional note). Any request from 2 to 6 days is honoured.
    wanted = DEFAULT_DAYS[level] if days is None else days
    used = max(2, min(6, wanted))
    notes = []
    if used != wanted:
        notes.append(f"{wanted} day(s) a week cannot be scheduled with proper rest days (2 to 6 works), so I used {used}.")
    if used > {"beginner": 4, "intermediate": 5, "advanced": 6}[level]:
        notes.append(f"{used} days a week is a lot for a {level}: drop a day if you feel run down.")
    if used == 2 or (used == 3 and level == "beginner"):
        name = "full_body"
    elif used == 3:
        name = "push_pull_legs"
    elif used == 4:
        name = "upper_lower"
    elif used == 5:
        name = "bro_split"
    else:
        name = "push_pull_legs"
    return name, used, (" ".join(notes) or None)
