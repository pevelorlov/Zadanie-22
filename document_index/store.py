from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np

from .models import Chunk, LoadedDocument


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS index_runs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    model_name TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    settings_json TEXT NOT NULL,
    document_count INTEGER NOT NULL,
    failed_document_count INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    duration_seconds REAL NOT NULL,
    errors_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    run_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    document_type TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    PRIMARY KEY (run_id, document_id),
    FOREIGN KEY (run_id) REFERENCES index_runs(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    strategy TEXT NOT NULL CHECK(strategy IN ('fixed', 'structural')),
    document_id TEXT NOT NULL,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    section TEXT NOT NULL,
    page INTEGER,
    chunk_order INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    text TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    PRIMARY KEY (run_id, chunk_id),
    FOREIGN KEY (run_id, document_id) REFERENCES documents(run_id, document_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS embeddings (
    run_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (run_id, chunk_id),
    FOREIGN KEY (run_id, chunk_id) REFERENCES chunks(run_id, chunk_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_chunks_run_strategy ON chunks(run_id, strategy, chunk_order);
CREATE INDEX IF NOT EXISTS idx_chunks_run_source ON chunks(run_id, source);
"""


class RagIndexStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def save_run(
        self,
        run_id: str,
        model_name: str,
        dimensions: int,
        settings: dict,
        documents: list[LoadedDocument],
        chunks: list[Chunk],
        vectors: np.ndarray,
        errors: list[dict],
        started_at: datetime,
        duration_seconds: float,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Количество чанков и эмбеддингов не совпадает.")
        completed = datetime.now(timezone.utc)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO index_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, started_at.isoformat(), completed.isoformat(), model_name, dimensions,
                 json.dumps(settings, ensure_ascii=False), len(documents), len(errors), len(chunks),
                 duration_seconds, json.dumps(errors, ensure_ascii=False)),
            )
            connection.executemany(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)",
                [(run_id, item.document_id, item.source, item.title, item.document_type, item.content_hash) for item in documents],
            )
            connection.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(item.chunk_id, run_id, item.strategy, item.document_id, item.source, item.title,
                  item.section, item.page, item.chunk_order, item.token_count, item.text, item.text_hash) for item in chunks],
            )
            connection.executemany(
                "INSERT INTO embeddings VALUES (?, ?, ?, ?, ?)",
                [(run_id, chunk.chunk_id, model_name, dimensions,
                  np.asarray(vector, dtype=np.float32).tobytes()) for chunk, vector in zip(chunks, vectors, strict=True)],
            )

    def latest_run(self) -> dict | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM index_runs ORDER BY completed_at DESC LIMIT 1").fetchone()
            if not row:
                return None
            result = dict(row)
            result["settings"] = json.loads(result.pop("settings_json"))
            result["errors"] = json.loads(result.pop("errors_json"))
            result["database_size_bytes"] = self.database_path.stat().st_size if self.database_path.exists() else 0
            result["strategies"] = {}
            for strategy in ("fixed", "structural"):
                stats = connection.execute(
                    """SELECT COUNT(*) count, COALESCE(AVG(token_count), 0) average_tokens,
                       COALESCE(MIN(token_count), 0) min_tokens, COALESCE(MAX(token_count), 0) max_tokens
                       FROM chunks WHERE run_id=? AND strategy=?""",
                    (result["id"], strategy),
                ).fetchone()
                result["strategies"][strategy] = dict(stats)
            return result

    def list_chunks(self, strategy: str, page: int, page_size: int, source: str = "") -> dict:
        run = self.latest_run()
        if not run:
            return {"items": [], "total": 0, "page": page, "page_size": page_size, "run_id": None}
        where = "run_id=? AND strategy=?"
        params: list = [run["id"], strategy]
        if source:
            where += " AND source=?"
            params.append(source)
        with self.connect() as connection:
            total = connection.execute(f"SELECT COUNT(*) FROM chunks WHERE {where}", params).fetchone()[0]
            rows = connection.execute(
                f"""SELECT chunk_id, strategy, document_id, source, title, section, page, chunk_order,
                    token_count, text, text_hash FROM chunks WHERE {where}
                    ORDER BY source, chunk_order LIMIT ? OFFSET ?""",
                [*params, page_size, (page - 1) * page_size],
            ).fetchall()
        return {"items": [dict(row) for row in rows], "total": total, "page": page, "page_size": page_size, "run_id": run["id"]}

    def sources(self) -> list[str]:
        run = self.latest_run()
        if not run:
            return []
        with self.connect() as connection:
            return [row[0] for row in connection.execute(
                "SELECT source FROM documents WHERE run_id=? ORDER BY source", (run["id"],)
            ).fetchall()]

    def search_vectors(self, strategy: str) -> tuple[list[dict], np.ndarray]:
        run = self.latest_run()
        if not run:
            return [], np.empty((0, 0), dtype=np.float32)
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT c.chunk_id, c.strategy, c.source, c.title, c.section, c.page,
                    c.chunk_order, c.token_count, c.text, c.text_hash, e.dimensions, e.vector
                    FROM chunks c JOIN embeddings e ON e.run_id=c.run_id AND e.chunk_id=c.chunk_id
                    WHERE c.run_id=? AND c.strategy=? ORDER BY c.source, c.chunk_order""",
                (run["id"], strategy),
            ).fetchall()
        items: list[dict] = []
        vectors: list[np.ndarray] = []
        for row in rows:
            item = dict(row)
            vector = np.frombuffer(item.pop("vector"), dtype=np.float32, count=int(item.pop("dimensions"))).copy()
            items.append(item)
            vectors.append(vector)
        matrix = np.stack(vectors) if vectors else np.empty((0, int(run["dimensions"])), dtype=np.float32)
        return items, matrix
