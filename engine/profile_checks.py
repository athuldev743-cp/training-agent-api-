"""Deterministic consistency checks on a profile (Step 6.6, contradictory information). Heuristics, not medical rules."""


def check_profile(profile):
    """A message to show instead of a plan when the profile looks self-contradictory, else None."""
    h, w = profile.height_cm, profile.weight_kg
    if h and w:
        bmi = w / (h / 100) ** 2
        if bmi < 12 or bmi > 60:
            return ("Your height and weight don't look consistent with each other (BMI about "
                    f"{bmi:.0f}). Please check them in your profile so I can build a safe plan.")
    if profile.age is not None and profile.age < 16 and profile.experience == "advanced":
        return ("Advanced programming isn't suitable at your age. Please set your experience to beginner "
                "or intermediate, or ask a coach or doctor.")
    return None
