STOP = set("a an the is are was were be been am do does did can could would should will shall may might must i me my mine we us our you your yours it its this that these those there here of in on at to for from with by about into over under as and or but if then than so not no yes very just also".split())
CONVO = set("hello hi hey thanks thank thx bye goodbye ok okay cool great nice good morning evening afternoon night sure please sorry help explain again more tell repeat simplify detail details example examples another instead change why how what who when where which mean means meaning name think say said ask answer question questions able want need like know understand got get give show make let lets way thing things really helpful sounds fine alright lot lots much many some any all bit little well too now today tomorrow yesterday day days week weeks".split())
PREFIX = ("workout", "exerc", "train", "muscl", "strength", "squat", "plank", "cardio", "stretch", "mobil", "flexib", "protein", "calor", "nutrit", "weight", "recover", "soreness", "hypertro", "endur", "stamin", "sprint", "warm", "program", "progress", "bench", "deadlift", "glute", "quad", "hamstr", "bicep", "tricep", "shoulder", "chest", "health", "fitness", "hydrat", "injur", "technique", "session", "routine", "aerobic", "interval", "pilates", "posture", "creatine", "supplement", "caffeine", "vitamin", "fatigue", "overtrain", "periodi", "taper", "plyo", "agility", "balance", "ankle", "tendon", "joint", "calisthen", "bodyweight", "dumbbell", "barbell", "kettlebell", "pushup", "pullup", "situp", "crunch", "lunge", "runn", "lifting", "jogg", "swimm", "cycl", "walki", "stress", "energy", "deload", "gains", "athlet", "sport", "reps", "sets")
EXACT = set("rep set gym abs arm run fat cut dip row hip band step form core back leg bmi rpe rir rm hiit yoga bike walk sore rest diet lift pull push curl bulk lean mass body burn jump knee neck pain heart sleep water power speed coach plan split press raise fly swim jog elbow wrist spine tendon".split())
SCOPE_REPLY = "I can only help with fitness, training and healthy habits, so I can't answer that. Ask me about your workouts, exercises, recovery or your plan."


def _fit(t):
    return t in EXACT or t.rstrip("s") in EXACT or t.startswith(PREFIX)


def out_of_scope(text: str) -> bool:
    """True when the message has real content words but none are fitness related. Heuristic: a novel fitness word may be missed (it is then refused, the safe side)."""
    import re
    tokens = re.findall(r"[a-z]+", text.lower().replace("\u2019", "'"))
    content = [t for t in tokens if len(t) >= 2 and t not in STOP and t not in CONVO]
    return bool(content) and not any(_fit(t) for t in tokens)

import difflib
CONTEXT = set("monday tuesday wednesday thursday friday saturday sunday plan programme program session sessions schedule routine week weekly workout workouts reminder reminders".split())
_LONG = sorted({w for w in (CONTEXT | set("exercise exercises training muscle muscles stretching strength protein nutrition cardio recovery squat squats deadlift fitness".split())) if len(w) >= 6})


def _fit(t):
    if t in EXACT or t.rstrip("s") in EXACT or t in CONTEXT or t.startswith(PREFIX):
        return True
    return len(t) >= 6 and bool(difflib.get_close_matches(t, _LONG, n=1, cutoff=0.84))
