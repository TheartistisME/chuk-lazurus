"""Durable JSONL memory stores for David runtime artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable
from uuid import uuid4


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9_./-]+", text)}


@dataclass(frozen=True)
class MemoryArtifact:
    family: str
    text: str
    kind: str = "observation"
    timestamp: str = field(default_factory=utc_now_iso)
    session_id: str = "default"
    user_id: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)
    artifact_id: str = field(default_factory=lambda: uuid4().hex)

    def to_json(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "family": self.family,
            "kind": self.kind,
            "text": self.text,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "MemoryArtifact":
        return cls(
            artifact_id=str(data.get("artifact_id") or uuid4().hex),
            family=str(data["family"]),
            kind=str(data.get("kind") or "observation"),
            text=str(data.get("text") or ""),
            timestamp=str(data.get("timestamp") or utc_now_iso()),
            session_id=str(data.get("session_id") or "default"),
            user_id=str(data.get("user_id") or "default"),
            metadata=dict(data.get("metadata") or {}),
        )


class JsonlMemoryStore:
    def __init__(self, path: Path, family: str) -> None:
        self.path = Path(path)
        self.family = family

    def append(self, artifact: MemoryArtifact) -> MemoryArtifact:
        if artifact.family != self.family:
            raise ValueError(f"cannot write {artifact.family!r} into {self.family!r} store")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(artifact.to_json(), sort_keys=True) + "\n")
        return artifact

    def all(self) -> list[MemoryArtifact]:
        if not self.path.exists():
            return []
        artifacts: list[MemoryArtifact] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    artifacts.append(MemoryArtifact.from_json(json.loads(line)))
        return artifacts

    def recall(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        query_tokens = _tokens(query)
        scored: list[tuple[int, int, MemoryArtifact]] = []
        for ordinal, artifact in enumerate(self.all()):
            overlap = len(query_tokens & _tokens(artifact.text))
            kind_bonus = 2 if artifact.kind in query_tokens else 0
            if overlap or kind_bonus or not query_tokens:
                scored.append((overlap + kind_bonus, ordinal, artifact))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [self._evidence(artifact, score, ordinal) for score, ordinal, artifact in scored[:limit]]

    def temporal(self, query: str, *, ordinal: str = "latest") -> dict[str, Any] | None:
        matches = self.recall(query, limit=100)
        if not matches:
            return None
        if ordinal in {"first", "earliest"}:
            return matches[-1]
        return matches[0]

    @staticmethod
    def _evidence(artifact: MemoryArtifact, score: int, ordinal: int) -> dict[str, Any]:
        return {
            "artifact_id": artifact.artifact_id,
            "family": artifact.family,
            "kind": artifact.kind,
            "text": artifact.text,
            "timestamp": artifact.timestamp,
            "score": score,
            "ordinal": ordinal,
            "provenance": artifact.metadata.get("provenance", "jsonl"),
        }


class MemoryBank:
    def __init__(self, user_store: JsonlMemoryStore, task_store: JsonlMemoryStore) -> None:
        self.user = user_store
        self.task = task_store

    @staticmethod
    def family_for_method(method: str) -> str:
        return "user" if method in {"user_continuity", "temporal_recall"} else "task"

    def store_for_method(self, method: str) -> JsonlMemoryStore:
        return self.user if self.family_for_method(method) == "user" else self.task

    def writeback(self, *, method: str, user_id: str, session_id: str, text: str, metadata: dict[str, Any]) -> MemoryArtifact:
        family = self.family_for_method(method)
        artifact = MemoryArtifact(
            family=family,
            kind=method,
            text=text,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
        )
        return (self.user if family == "user" else self.task).append(artifact)

    def recall_for_method(self, method: str, query: str) -> list[dict[str, Any]]:
        return self.store_for_method(method).recall(query)

    def symbolic_chain(self, query: str) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        seen: set[str] = set()
        for store in (self.task, self.user):
            for item in store.recall(query, limit=5):
                if item["artifact_id"] not in seen:
                    seen.add(item["artifact_id"])
                    evidence.append(item)
        return evidence

    def stores(self) -> Iterable[JsonlMemoryStore]:
        return (self.user, self.task)
