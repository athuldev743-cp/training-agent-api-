"""SQLite storage for one user's profile. Mirrors what Room will do on Android."""
import json

from state.db import LOCAL_USER_ID, SCHEMA_VERSION, check_user_id, connect  # SCHEMA_VERSION is re-exported for tests
from state.profile import UserProfile


class ProfileStore:
    def __init__(self, path=":memory:", read_only=False, user_id=LOCAL_USER_ID):
        self.user_id = check_user_id(user_id)
        self.conn = connect(path, read_only=read_only)

    def get(self) -> UserProfile | None:
        row = self.conn.execute(
            "SELECT * FROM user_profile WHERE user_id = ?", (self.user_id,)
        ).fetchone()
        if row is None:
            return None
        equipment = json.loads(row["equipment"]) if row["equipment"] is not None else None
        limitations = json.loads(row["limitations"]) if row["limitations"] is not None else []
        return UserProfile(
            name=row["name"],
            age=row["age"],
            height_cm=row["height_cm"],
            weight_kg=row["weight_kg"],
            goal=row["goal"],
            experience=row["experience"],
            equipment=equipment,
            session_minutes=row["session_minutes"],
            limitations=limitations,
            screening=row["screening"],
        )

    def save(self, profile: UserProfile) -> None:
        equipment = json.dumps(profile.equipment) if profile.equipment is not None else None
        self.conn.execute(
            """INSERT OR REPLACE INTO user_profile
                   (user_id, name, age, height_cm, weight_kg, goal, experience, equipment,
                    session_minutes, limitations, screening)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (self.user_id, profile.name, profile.age, profile.height_cm, profile.weight_kg,
             profile.goal, profile.experience, equipment, profile.session_minutes,
             json.dumps(profile.limitations), profile.screening),
        )
        self.conn.commit()

    def update(self, **fields) -> UserProfile:
        current = self.get()
        if current is None:
            raise LookupError("No profile saved yet")
        updated = UserProfile(**{**current.model_dump(), **fields})
        self.save(updated)
        return updated

    def close(self) -> None:
        self.conn.close()