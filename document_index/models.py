from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class SourceBlock:
    section: str
    text: str
    page: int | None = None


@dataclass(slots=True)
class LoadedDocument:
    document_id: str
    source: str
    title: str
    document_type: str
    path: Path
    content_hash: str
    blocks: list[SourceBlock] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(block.text for block in self.blocks if block.text.strip())


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    strategy: str
    document_id: str
    source: str
    title: str
    section: str
    page: int | None
    chunk_order: int
    token_count: int
    text: str
    text_hash: str

