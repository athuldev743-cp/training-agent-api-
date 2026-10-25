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
    goal: str | None = None


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
    goals: list[str] = []
    time_warning: str | None = None
    recommended_minutes: int | None = None
    block_mode: str | None = None


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


def _pick(muscle, pool, chosen, used_patterns, owned, phase, tiers, level, by_id, families, goal="general"):
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
            _goal_rank(e, goal),
            0 if (phase == "build" and tiers[e.id] == "build") else 1,  # build weeks: harder tier first
            e.movement in used_patterns,                                # variety within the session
            placement,
            -DIFFICULTY_RANK[e.difficulty] if phase == "build" else 0,
            e.name,
        )

    return min(candidates, key=key)


def _build_block(phase, split_days, level, pool, owned, budget, cap, tiers, templates, goal,
                 by_id, families, day_caps=None, day_goals=None, avoid=None, anchors=None) -> dict[int, list[Exercise]]:
    last_week = 2 if phase == "foundation" else 4  # the block's heaviest week must still fit the budget
    block: dict[int, list[Exercise]] = {}
    seen_by_label: dict[str, set[str]] = {}
    def _pick(*a, **k):  # goal-aware pick
        a = list(a)
        if any(c.movement == "hinge" and set(c.equipment) & LOADED_EQUIPMENT for c in a[2]):  # one heavy hinge per session
            a[1] = [e for e in a[1] if not (e.movement == "hinge" and set(e.equipment) & LOADED_EQUIPMENT)]
        return _pick_goal(*a, goal=goal, **k)
    cond_pool = [e for e in pool if e.movement in CONDITIONING_MOVEMENTS]
    pool = [e for e in pool if e.movement not in CONDITIONING_MOVEMENTS]
    n_cond = GOAL_CONDITIONING.get(goal, 0)
    used_cond: set[str] = set()
    goal0 = goal
    _core = [e for e in pool if "core" in e.primary]

    def _wk(m):  # sets this muscle already has in the week being built (heaviest week of the block)
        tot = sum(_sets_for_week(prescription_for(e.id, level, templates, (day_goals[dn - 1] if day_goals else goal0)), last_week)
                  for dn, d in block.items() for e in d if m in e.primary)
        return tot + sum(_sets_for_week(prescription_for(e.id, level, templates, goal), last_week) for e in chosen if m in e.primary)
    core_reserve = min((_exercise_minutes(prescription_for(e.id, level, templates, goal), _sets_for_week(prescription_for(e.id, level, templates, goal), last_week)) for e in _core), default=0.0)
    for day_number, groups in enumerate(split_days, 1):
        goal = day_goals[day_number - 1] if day_goals else goal0
        rem = {m: sum(1 for g in split_days[day_number - 1:] if m in day_muscles(g)) for m in PRIORITY}

        def _over(m, rx):  # weekly volume is shared over the days that train a muscle; the first exercise per muscle each day is always allowed
            s = _sets_for_week(rx, last_week)
            today = sum(_sets_for_week(prescription_for(c.id, level, templates, goal), last_week) for c in chosen if m in c.primary)
            if today == 0:
                return False
            used = _wk(m) - today
            return today + s > (WEEKLY_SET_CAP - used) // max(1, rem.get(m, 1))
        n_cond = GOAL_CONDITIONING.get(goal, 0)
        order = _day_muscle_order(groups, day_number, cap)
        _cc = [e for e in cond_pool if set(e.equipment) & owned]
        cond_reserve = min((_exercise_minutes(prescription_for(e.id, level, templates, goal), _sets_for_week(prescription_for(e.id, level, templates, goal), last_week)) for e in _cc), default=0.0) if n_cond else 0.0
        prior = set(seen_by_label.get(groups[0], set())) | set((avoid or {}).get(day_number, set()))  # earlier days of the same kind, plus last week's accessories
        day_cap = cap if not day_caps else min(cap, max(1, len(day_caps.get(day_number, [])) or cap))
        chosen: list[Exercise] = []
        patterns: set[str] = set()
        per_muscle: dict[str, int] = {}
        minutes = 0.0

        for a_ in (anchors or {}).get(day_number, []):  # keep last week's big lifts
            _rx = prescription_for(a_.id, level, templates, goal)
            chosen.append(a_); patterns.add(a_.movement); minutes += _exercise_minutes(_rx, _sets_for_week(_rx, last_week))
            for _m in a_.primary: per_muscle[_m] = per_muscle.get(_m, 0) + 1
        def try_add(muscle: str) -> bool:
            nonlocal minutes
            if len(chosen) >= max(1, day_cap - n_cond) or per_muscle.get(muscle, 0) >= (1 if muscle == "core" else MAX_PER_MUSCLE):
                return False
            fresh = [e for e in pool if e.id not in prior]
            ex = (_pick(muscle, fresh, chosen, patterns, owned, phase, tiers, level, by_id, families) if prior else None) \
                or _pick(muscle, pool, chosen, patterns, owned, phase, tiers, level, by_id, families)
            if ex is None:
                return False
            rx = prescription_for(ex.id, level, templates, goal)
            cost = _exercise_minutes(rx, _sets_for_week(rx, last_week))
            if minutes + cost > budget - cond_reserve - (core_reserve if (muscle != "core" and not any("core" in c.primary for c in chosen)) else 0.0):
                return False
            if any(_over(m, rx) for m in ex.primary):
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
        # Goal extras: conditioning / plyometric work for goals that ask for it (endurance, fat loss)
        for _ in range(n_cond):
            if len(chosen) >= day_cap:
                break
            taken = {c.id for c in chosen}
            covered = {m for c in chosen for m in c.primary}
            day_m = day_muscles(groups) | {"core"}
            has_ply = any(c.movement == "plyometric" for c in chosen)
            fresh_c = [e for e in cond_pool if e.id not in taken and not _redundant(e, chosen, families)
                       and not ("core" in e.primary and "core" in covered)
                       and (e.movement != "plyometric" or (not has_ply and set(e.primary) <= day_m))
                       and (e.movement == "conditioning" or not any(_over(m, prescription_for(e.id, level, templates, goal)) for m in e.primary))]
            if not fresh_c:
                break
            ex = min(fresh_c, key=lambda e: (0 if (set(e.equipment) & owned and not any(c.movement in CONDITIONING_MOVEMENTS for c in chosen)) else 1, _cond_pref(e, goal), bool(set(e.primary) & covered), e.id in used_cond, e.id in prior, -DIFFICULTY_RANK[e.difficulty] if level != "beginner" else 0, e.name))
            rx = prescription_for(ex.id, level, templates, goal)
            cost = _exercise_minutes(rx, _sets_for_week(rx, last_week))
            if minutes + cost > budget:
                break
            chosen.append(ex)
            patterns.add(ex.movement)
            minutes += cost
            used_cond.add(ex.id)
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


def build_program(exercises: list[Exercise], profile: UserProfile, days_per_week: int | None = None, goals: list[str] | None = None, _probe: bool = False, _force_split: str | None = None, avoid_ids: set[str] | None = None) -> dict:
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
    if _force_split:
        split = _force_split
    elif split == "push_pull_legs":
        _sp = build_program(exercises, profile.model_copy(update={"session_minutes": 90}), days_per_week, goals, _probe=True, _force_split="push_pull_legs")
        _thin = _thin_days(_sp["program"], exercises) if _sp["program"] is not None else []
        if _thin:
            split = "full_body" if used_days <= 3 else "upper_lower"
            notes.append(f"With your equipment the {' and '.join(_thin)} day of a push/pull/legs split would have too few exercises, so I used the {SPLIT_NAMES[split]} split instead. A pull-up bar, a row bar or dumbbells would unlock proper pulling.")

    tiers = load_tiers()
    templates = load_templates()
    by_id = {e.id: e for e in exercises}
    families = exercise_families(exercises)
    cap = MAX_EXERCISES[level]
    goals = _clean_goals(goals, profile.goal)
    goal = goals[0]
    split_days = SPLITS[split][:used_days]
    day_goals, dropped = assign_day_goals(split_days, goals)
    if dropped:
        notes.append(f"Only {used_days} training days: {', '.join(g.replace('_', ' ') for g in dropped)} did not fit. Add a training day to include it.")
    owned = set(equipment) - {"bodyweight"}

    blocks = {}
    for phase in ("foundation", "build"):
        pool = allowed_exercises(exercises, level, equipment, profile.limitations, phase, tiers)
        blocks[phase] = _build_block(phase, split_days, level, pool, owned, profile.session_minutes,
                                     cap, tiers, templates, goal, by_id, families, blocks.get("foundation"), day_goals=day_goals, avoid={d: set(avoid_ids or ()) for d in range(1, len(split_days) + 1)})
        blocks[phase + "_b"] = _build_block(phase, split_days, level, pool, owned, profile.session_minutes,
                                            cap, tiers, templates, goal, by_id, families, blocks.get("foundation"), day_goals=day_goals,
                                            avoid={d: ids | set(avoid_ids or ()) for d, ids in _variant_avoid(blocks[phase]).items()}, anchors={d: x[:KEEP_ANCHORS] for d, x in blocks[phase].items()})
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
            for ex in blocks[phase if week in (1, 3) else phase + "_b"][day_number]:
                rx = prescription_for(ex.id, level, templates, day_goals[day_number - 1])
                sets = _sets_for_week(rx, week)
                lo, hi = _aim_range(rx, week)
                minutes += _exercise_minutes(rx, sets)
                items.append(ProgramExercise(
                    exercise_id=ex.id, name=ex.name, tier=tiers[ex.id], unit=rx.unit, sets=sets,
                    rep_min=rx.rep_min, rep_max=rx.rep_max, aim_low=lo, aim_high=hi,
                    rest_seconds=rx.rest_seconds))
            sessions.append(ProgramSession(
                day_number=day_number, goal=day_goals[day_number - 1], weekday=WEEKDAYS[used_days][day_number - 1],
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
        notes.append(f"Your sessions are about {avg_minutes:.0f} minutes, shorter than the {profile.session_minutes} you set, because I cap a {level} session at {cap} exercises. Use the spare time for warm-up and mobility." if (sum(len(s.exercises) >= cap for wk in weeks for s in wk.sessions) >= 0.5 * sum(len(wk.sessions) for wk in weeks)) else f"Your sessions are about {avg_minutes:.0f} minutes, shorter than the {profile.session_minutes} you set, because few exercises in my library match your level and equipment. More equipment would unlock longer sessions.")
    peak = max(s.est_minutes for wk in weeks for s in wk.sessions)
    capped = any(len(s.exercises) >= cap for wk in weeks for s in wk.sessions)
    if capped and avg_minutes >= 0.5 * profile.session_minutes and peak < 0.75 * profile.session_minutes:
        notes.append(f"Sessions top out at about {peak:.0f} minutes: I cap a {level} session at {cap} exercises, so the rest of your {profile.session_minutes} minutes is unused. Spend it on warm-up and mobility.")
    notes.append("After week 4, take an easier week (fewer sets, easy effort) before starting the next programme.")

    program = Program(goal=goal, goals=[g for g in goals if g in day_goals], level=level, split=split, days_per_week=used_days, equipment=list(equipment),
                      limitations=list(profile.limitations), weeks=weeks, notes=notes)
    if not _probe and profile.session_minutes < 90:
        probe = build_program(exercises, profile.model_copy(update={"session_minutes": 90}), days_per_week, goals, _probe=True, _force_split=split)
        if probe["program"] is not None:
            pw, mine = probe["program"].weeks[0].sessions, weeks[0].sessions
            need = sum(s.est_minutes for s in pw) / len(pw)
            lost = sum(len(s.exercises) for s in pw) / len(pw) - sum(len(s.exercises) for s in mine) / len(mine)
            if lost >= 1 and profile.session_minutes < 0.85 * need:
                program.recommended_minutes = ((int(need) + 4) // 5) * 5
                program.time_warning = (f"{' + '.join(g.replace('_', ' ') for g in program.goals)} needs about {program.recommended_minutes} minutes per session "
                                        f"to cover everything. You set {profile.session_minutes}, so each session has about {lost:.0f} fewer exercises than the full plan. "
                                        f"Train at least {program.recommended_minutes} minutes, or drop a goal.")
                program.notes.append("Time check: " + program.time_warning)
    return {"status": "ok", "reply": format_overview(program), "program": program}


def format_overview(program: Program) -> str:
    days = ", ".join(s.weekday for s in program.weeks[0].sessions)
    equipment = ", ".join(e.replace("_", " ") for e in program.equipment) or "bodyweight only"
    avoid = ", ".join(a.replace("_", " ") for a in program.limitations) or "nothing"
    lines = [
        f"Your 4-week {SPLIT_NAMES[program.split]} programme: {program.days_per_week} days a week ({days}).",
        f"Level: {program.level}. {_goal_text(program)}. Equipment: {equipment}. Avoiding exercises that load: {avoid}.",
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


from functools import lru_cache as _lru_cache
CONDITIONING_MOVEMENTS = {"conditioning", "plyometric"}
WEEKLY_SET_CAP = 20  # sets per muscle per week in the heaviest week (the audited baseline)
GOAL_CONDITIONING = {"endurance": 2, "fat_loss": 2}  # conditioning exercises added per session


@_lru_cache(maxsize=1)
def _exercise_templates():
    return load_templates()


def _goal_rank(e, goal):
    """0 = preferred for this goal. Strength likes hard low-rep work; endurance and fat loss avoid skill moves."""
    try:
        t = _exercise_templates().get(e.id, "")
    except Exception:
        t = ""
    if goal == "strength":
        return 0 if t in ("hard_bodyweight", "skill_bodyweight", "loaded_compound") else 1
    if goal == "fat_loss":  # metabolic work: big multi-muscle exercises first
        return 0 if t in ("bodyweight_compound", "loaded_compound", "loaded_unilateral") and len(e.muscles) >= 2 else 1
    if goal == "endurance":
        return 0 if t in ("bodyweight_compound", "core_reps", "loaded_compound", "loaded_unilateral", "isolation", "hold") else 1
    if goal == "muscle":
        return 0 if t in ("loaded_compound", "bodyweight_compound", "isolation") else 1
    return 0


_pick_goal = _pick


def _clean_goals(goals, default):
    from engine.prescription import GOALS as _GOALS
    out = [g for g in dict.fromkeys(goals or []) if g in _GOALS][:3]
    return out or [default]


def assign_day_goals(split_days, goals):
    """Goal per training day. The nth day of the same kind cycles through the goals, so each goal gets upper and lower
    days. A goal left without a day takes a day from the end that another goal has twice; if none fits it is dropped."""
    seen, out = {}, []
    for groups in split_days:
        k = groups[0]
        out.append(goals[seen.get(k, 0) % len(goals)])
        seen[k] = seen.get(k, 0) + 1
    dropped = []
    for g in [g for g in goals if g not in out]:
        j = next((j for j in range(len(out) - 1, 0, -1) if out.count(out[j]) > 1), None)
        if j is None:
            dropped.append(g)
        else:
            out[j] = g
    return out, dropped


def _goal_text(program):
    gs = [g.replace("_", " ") for g in (program.goals or [program.goal])]
    return ("Goals: " + ", ".join(gs)) if len(gs) > 1 else "Goal: " + gs[0]


SPLITS = {**SPLITS, "upper_lower": list(SPLITS["upper_lower"][:4]) + [["upper_body"], ["lower_body", "core"]]}


def _thin_days(program, exercises, min_exercises=4):
    """Labels of days that end up with fewer than min_exercises non-conditioning exercises in week 4."""
    by_id = {e.id: e for e in exercises}
    out = set()
    for s in program.weeks[3].sessions:
        if sum(1 for e in s.exercises if by_id[e.exercise_id].movement not in CONDITIONING_MOVEMENTS) < min_exercises:
            out.add(s.label.lower())
    return sorted(out)


LOADED_EQUIPMENT = {"barbell", "dumbbells", "machine", "cable"}


HIIT_CONDITIONING = {"burpee", "mountain-climber", "high-knees", "plank-jack", "skater-jump"}


def _cond_pref(e, goal):
    """Fat loss prefers high-intensity bursts; endurance prefers sustained, lower-impact work. 0 = preferred."""
    hi = e.movement == "plyometric" or e.id in HIIT_CONDITIONING
    return 0 if (goal == "fat_loss") == hi else 1


KEEP_ANCHORS = 3


def _variant_avoid(block):
    """Per day: the exercises after the first KEEP_ANCHORS. The big lifts stay for the second week of a phase so they
    can progress; the accessories rotate."""
    return {d: {e.id for e in ex[KEEP_ANCHORS:]} for d, ex in block.items()}
