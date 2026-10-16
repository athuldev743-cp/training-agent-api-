"""What-if answers by recomputation (Step 38a, v2). The builder is a pure function, so a question such as 'if I had
dumbbells', 'if I trained 5 days' or 'if I only had 20 minutes' is answered by building two programmes from the profile (as
it is, and with the change) and comparing them, with the reasons taken from the builder's rules. No model, nothing stored,
the profile is not changed. Not routed from chat yet."""
import re
from statistics import mean

from engine.extract import extract_equipment
from engine.extract_coach import extract_days
from engine.program import ANCHORS, PRIORITY, SPLIT_NAMES, Program, build_program
from rag.store import Exercise
from state.profile import UserProfile

MAX_REPLY_CHARS = 1800
_MINUTES = re.compile(r"\b(\d{1,3})\s*-?\s*(?:mins?|minutes?)\b", re.IGNORECASE)
_ROW_BAR = re.compile(r"\brow\s*-?\s*bar\b", re.IGNORECASE)


def extract_minutes(text: str) -> int | None:
    m = _MINUTES.search(text)
    if not m:
        return None
    value = int(m.group(1))
    return value if 5 <= value <= 180 else None


def extract_whatif(text: str) -> dict:
    equipment = list(extract_equipment(text))
    if _ROW_BAR.search(text):
        equipment.append("row_bar")
    return {"add_equipment": equipment, "days": extract_days(text), "minutes": extract_minutes(text)}


def _names(p: Program) -> list[str]:
    seen: list[str] = []
    for wk in p.weeks:
        for s in wk.sessions:
            for it in s.exercises:
                if it.name not in seen:
                    seen.append(it.name)
    return seen


def _muscles(p: Program, by_id: dict) -> set[str]:
    return {m for wk in p.weeks for s in wk.sessions for it in s.exercises for m in by_id[it.exercise_id].primary}


def _list(items: list[str], limit: int = 6) -> str:
    shown = ", ".join(items[:limit])
    return shown + (f" and {len(items) - limit} more" if len(items) > limit else "")


def _key(note: str) -> str:
    return re.sub(r"\d+", "#", note)


def _short(note: str) -> str:
    return note if len(note) <= 110 else note[:107] + "..."


def what_if(exercises: list[Exercise], profile: UserProfile, current_days: int | None = None,
            add_equipment: list[str] | None = None, days: int | None = None, minutes: int | None = None) -> dict:
    have = list(profile.equipment or [])
    add = [e for e in (add_equipment or []) if e not in have]
    if not add and days is None and minutes is None:
        return {"status": "needs_info", "reply": "Tell me what to change, for example 'if I had dumbbells', "
                                                  "'if I trained 5 days' or 'if I only had 20 minutes'."}
    base = build_program(exercises, profile, current_days)
    if base["status"] != "ok":
        return {"status": base["status"], "reply": base["reply"]}
    changed = profile.model_copy(update={"equipment": sorted(set(have) | set(add)) if add else profile.equipment,
                                         "session_minutes": minutes or profile.session_minutes})
    alt = build_program(exercises, changed, days if days is not None else current_days)
    if alt["status"] != "ok":
        return {"status": alt["status"], "reply": "With that change: " + alt["reply"]}

    b, a = base["program"], alt["program"]
    by_id = {e.id: e for e in exercises}
    changes = []
    if add:
        changes.append("you also had " + " and ".join(e.replace("_", " ") for e in add))
    if days is not None:
        changes.append(f"you trained {a.days_per_week} days a week")
    if minutes:
        changes.append(f"your sessions were limited to {minutes} minutes")
    mb = mean(s.est_minutes for s in b.weeks[0].sessions)
    ma = mean(s.est_minutes for s in a.weeks[0].sessions)
    tb = sum(s.est_minutes for s in b.weeks[0].sessions)
    ta = sum(s.est_minutes for s in a.weeks[0].sessions)
    lines = ["If " + " and ".join(changes) + ":"]
    if b.split != a.split:
        lines.append(f"- The split changes from {SPLIT_NAMES[b.split]} to {SPLIT_NAMES[a.split]}.")
    else:
        lines.append(f"- The split stays {SPLIT_NAMES[b.split]}.")
    lines.append(f"- Sessions: {b.days_per_week} -> {a.days_per_week} a week, about {mb:.0f} -> {ma:.0f} minutes each.")
    lines.append(f"- Total training time per week: {tb:.0f} -> {ta:.0f} minutes.")
    nb, na = _names(b), _names(a)
    added = [n for n in na if n not in nb]
    dropped = [n for n in nb if n not in na]
    if added:
        lines.append("- Exercises added: " + _list(added) + ".")
    if dropped:
        lines.append("- Exercises dropped: " + _list(dropped) + ".")
    mus_b, mus_a = _muscles(b, by_id), _muscles(a, by_id)
    new = sorted(mus_a - mus_b, key=PRIORITY.index)
    lost = sorted(mus_b - mus_a, key=PRIORITY.index)
    if new:
        lines.append("- Newly trained mainly: " + ", ".join(new) + ".")
    if lost:
        lines.append("- No longer trained mainly: " + ", ".join(lost) + ".")
    if minutes and b.split == a.split and b.days_per_week == a.days_per_week:
        for sb, sa in zip(b.weeks[0].sessions, a.weeks[0].sessions):
            gone = [e.name for e in sb.exercises if e.name not in {x.name for x in sa.exercises}]
            if gone:
                lines.append(f"- Day {sb.day_number} {sb.weekday} drops: {_list(gone, 4)}.")
    b_keys, a_keys = {_key(n) for n in b.notes}, {_key(n) for n in a.notes}
    for n in [n for n in b.notes if _key(n) not in a_keys and not n.startswith("After week 4")][:2]:
        lines.append("- No longer applies: " + _short(n))
    for n in [n for n in a.notes if _key(n) not in b_keys and not n.startswith("After week 4")][:2]:
        lines.append("- New: " + _short(n))
    if add and (added or dropped):
        lines.append("Why: when you own equipment I prefer exercises that use it, and I allow only one exercise per "
                     "movement type for a muscle in a session, so an exercise with your new equipment can replace a "
                     "bodyweight one for the same muscle.")
    if b.split != a.split:
        lines.append(f"Why: the split depends on your level and days per week; at {a.level} level {a.days_per_week} "
                     f"days uses {SPLIT_NAMES[a.split]}, which groups the muscles differently.")
    if minutes:
        lines.append(f"How I decide: I fill each session muscle by muscle in this order: {', '.join(PRIORITY)}; "
                     f"{', '.join(ANCHORS[:-1])} and {ANCHORS[-1]} are kept every day. An exercise that would push a "
                     "session past your limit is skipped, so muscles later in the order tend to drop out first. The "
                     "limit is applied to every day.")
    footer = []
    if days is not None:
        footer.append("More days is not automatically faster: the app has no rule for that and I have not checked the research.")
    footer.append("Minutes are for week 1; later weeks add sets. This compares two fresh builds from your profile, so it ignores swaps you have made, and nothing was "
                  "changed." + (" To use it, add the equipment to your profile and type /program." if add else ""))
    footer.append("If this is not what you meant, ask again in other words.")
    foot = "\n".join(footer)
    body = "\n".join(lines)
    if len(body) + len(foot) + 1 > MAX_REPLY_CHARS:
        body = body[:MAX_REPLY_CHARS - len(foot) - 1].rsplit("\n", 1)[0]
    return {"status": "ok", "reply": body + "\n" + foot, "base": b, "alt": a}
