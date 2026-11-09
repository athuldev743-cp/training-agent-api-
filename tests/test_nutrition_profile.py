import sqlite3

import pytest
from pydantic import ValidationError

from state.profile import UserProfile
from state.store import ProfileStore


def test_profile_store_round_trips_nutrition_preferences():
    store = ProfileStore()
    try:
        profile = UserProfile(
            name="Nutrition Test",
            age=26,
            height_cm=174,
            weight_kg=76,
            goal="fat_loss",
            experience="intermediate",
            equipment=["dumbbells"],
            limitations=["knee"],
            activity_level="moderately_active",
            dietary_pattern="eggetarian",
            preferred_foods=["eggs", "rice"],
            excluded_foods=["mushroom"],
            allergies=["peanut"],
            intolerances=["lactose"],
            meals_per_day=4,
        )
        store.save(profile)
        loaded = store.get()

        assert loaded is not None
        assert loaded.model_dump() == profile.model_dump()
    finally:
        store.close()


def test_v9_profile_migrates_without_losing_existing_workout_data(tmp_path):
    path = tmp_path / "legacy-v9.db"
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE user_profile (
            user_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            age INTEGER,
            height_cm REAL,
            weight_kg REAL,
            goal TEXT NOT NULL,
            experience TEXT NOT NULL,
            equipment TEXT,
            session_minutes INTEGER NOT NULL,
            limitations TEXT,
            screening TEXT NOT NULL DEFAULT 'not_asked'
        )"""
    )
    conn.execute(
        """INSERT INTO user_profile
           (user_id, name, age, height_cm, weight_kg, goal, experience,
            equipment, session_minutes, limitations, screening)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("legacy-user", "Existing User", 31, 170, 72, "strength", "beginner",
         '["dumbbells"]', 45, '["knee"]', "clear"),
    )
    conn.execute("PRAGMA user_version = 9")
    conn.commit()
    conn.close()

    store = ProfileStore(path, user_id="legacy-user")
    try:
        loaded = store.get()
        assert loaded is not None
        assert loaded.name == "Existing User"
        assert loaded.age == 31
        assert loaded.goal == "strength"
        assert loaded.equipment == ["dumbbells"]
        assert loaded.limitations == ["knee"]
        assert loaded.activity_level == "unspecified"
        assert loaded.dietary_pattern == "unspecified"
        assert loaded.preferred_foods == []
        assert loaded.allergies == []
        assert loaded.meals_per_day == 3
        assert store.conn.execute("PRAGMA user_version").fetchone()[0] == 10
    finally:
        store.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("activity_level", "extremely_active"),
        ("dietary_pattern", "carnivore"),
        ("meals_per_day", 0),
        ("allergies", [""]),
        ("preferred_foods", ["x" * 61]),
    ],
)
def test_profile_rejects_invalid_nutrition_fields(field, value):
    with pytest.raises(ValidationError):
        UserProfile(name="Invalid", **{field: value})
