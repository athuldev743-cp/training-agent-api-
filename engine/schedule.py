
"""Turns a programme's weekly pattern into real calendar dates."""
from datetime import date, timedelta

from engine.program import Program

WEEKDAY_INDEX = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}


def schedule_dates(program: Program, start: date) -> list[tuple[int, int, date]]:
    """Return (week, day_number, date) for every session.

    Sessions sharing a day_number receive the same date.
    The weekly weekday pattern remains compatible with existing programmes.
    """
    if not program.weeks:
        return []

    first_week = program.weeks[0]
    training_days = {}
    for session in first_week.sessions:
        training_days.setdefault(session.day_number, session.weekday)

    cycle = list(dict.fromkeys(
        WEEKDAY_INDEX[weekday] for weekday in training_days.values()
    ))
    if not cycle:
        return []

    # Find the first training weekday on or after the requested start date.
    current_date = start
    while current_date.weekday() not in cycle:
        current_date += timedelta(days=1)

    dates_by_day = {}
    position = cycle.index(current_date.weekday())

    for day_number in training_days:
        dates_by_day[day_number] = current_date
        position = (position + 1) % len(cycle)
        current_date += timedelta(days=1)
        while current_date.weekday() != cycle[position]:
            current_date += timedelta(days=1)

    result = []
    first_week_number = first_week.number

    for week in program.weeks:
        week_offset = week.number - first_week_number
        for session in week.sessions:
            planned_date = (
                dates_by_day[session.day_number]
                + timedelta(days=7 * week_offset)
            )
            result.append((week.number, session.day_number, planned_date))

    return result
