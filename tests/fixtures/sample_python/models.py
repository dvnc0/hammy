import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class User:
    id: int
    name: str
    email: str

class UserService:
    def __init__(self, db):
        self.db = db

    async def get_user(self, user_id: int) -> dict:
        return self.db.find(user_id)

    def list_users(self) -> list:
        return self.db.find_all()

    def create_user(self, name: str, email: str) -> User:
        return self.db.create(name=name, email=email)

def format_user(user: User) -> str:
    return f"{user.name} <{user.email}>"
