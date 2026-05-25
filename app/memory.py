from __future__ import annotations
from collections import deque
from datetime import datetime
from typing import Dict, List

MAX_SESSION_HISTORY = 12

class SessionMemory:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history = deque(maxlen=MAX_SESSION_HISTORY)
        self.updated_at = datetime.utcnow()

    def append(self, role: str, content: str) -> None:
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        })
        self.updated_at = datetime.utcnow()

    def recent(self, count: int = 4) -> List[Dict[str, str]]:
        return list(self.history)[-count:]

    def summary(self) -> str:
        if not self.history:
            return "No session memory initialized yet."

        recent_items = self.recent(4)
        lines = []
        for item in recent_items:
            label = "User" if item["role"] == "user" else "Assistant"
            snippet = item["content"].strip().replace("\n", " ")
            if len(snippet) > 120:
                snippet = snippet[:117].rstrip() + "..."
            lines.append(f"{label}: {snippet}")

        return " | ".join(lines)


_MEM0: Dict[str, SessionMemory] = {}


def get_session_memory(session_id: str) -> SessionMemory:
    if session_id not in _MEM0:
        _MEM0[session_id] = SessionMemory(session_id)
    return _MEM0[session_id]


def append_session_memory(session_id: str, role: str, content: str) -> None:
    memory = get_session_memory(session_id)
    memory.append(role, content)


def get_mem0_summary(session_id: str) -> str:
    return get_session_memory(session_id).summary()
