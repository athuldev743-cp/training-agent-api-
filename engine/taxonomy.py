"""Single source of truth: muscles, muscle groups, aliases, and split routines."""

MUSCLES = ["chest", "shoulders", "triceps", "back", "biceps",
           "core", "quads", "hamstrings", "glutes", "calves"]

GROUPS = {
    "upper_body": ["chest", "shoulders", "triceps", "back", "biceps"],
    "lower_body": ["glutes", "quads", "hamstrings", "calves"],
    "push": ["chest", "shoulders", "triceps"],
    "pull": ["back", "biceps"],
    "arms": ["biceps", "triceps"],
    "thighs": ["quads", "hamstrings"],
}

ALIASES = {
    "legs": "lower_body", "leg": "lower_body",
    "hams": "hamstrings", "hamstring": "hamstrings",
    "quad": "quads", "quadriceps": "quads",
    "glute": "glutes", "calf": "calves",
    "abs": "core", "abdominals": "core",
    "delts": "shoulders", "deltoids": "shoulders", "shoulder": "shoulders",
    "lats": "back", "pecs": "chest",
    "bicep": "biceps", "tricep": "triceps",
}

# Each split is a list of training days; each day lists groups or muscles.
SPLITS = {
    "full_body": [["upper_body", "lower_body", "core"] for _ in range(3)],
    "upper_lower": [["upper_body"], ["lower_body"], ["upper_body"], ["lower_body", "core"]],
    "push_pull_legs": [["push"], ["pull"], ["lower_body", "core"]],
    "bro_split": [["chest"], ["back"], ["shoulders"], ["arms"], ["lower_body", "core"]],
}


def _key(term: str) -> str:
    return term.strip().lower().replace("-", "_").replace(" ", "_")


def expand(term: str) -> set[str]:
    """Turn a muscle, group, or alias into a set of canonical muscles. Unknown -> empty set."""
    key = _key(term)
    key = ALIASES.get(key, key)
    if key in GROUPS:
        return set(GROUPS[key])
    if key in MUSCLES:
        return {key}
    return set()


def day_muscles(day: list[str]) -> set[str]:
    return set().union(*(expand(item) for item in day))


def split_muscles(name: str) -> set[str]:
    return set().union(*(day_muscles(day) for day in SPLITS[name]))


def missing_muscles(name: str) -> set[str]:
    return set(MUSCLES) - split_muscles(name)
