from __future__ import annotations

import hashlib
from typing import Protocol

from .models import Chunk, LoadedDocument, SourceBlock


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...
    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str: ...


def fixed_chunks(document: LoadedDocument, tokenizer: Tokenizer, size: int, overlap: int) -> list[Chunk]:
    _validate(size, overlap)
    return _split_block(document, SourceBlock("Весь документ", document.text), "fixed", tokenizer, size, overlap, 0)


def structural_chunks(document: LoadedDocument, tokenizer: Tokenizer, max_size: int, overlap: int) -> list[Chunk]:
    _validate(max_size, overlap)
    chunks: list[Chunk] = []
    order = 0
    for block in document.blocks:
        created = _split_block(document, block, "structural", tokenizer, max_size, overlap, order)
        chunks.extend(created)
        order += len(created)
    return chunks


def _split_block(
    document: LoadedDocument,
    block: SourceBlock,
    strategy: str,
    tokenizer: Tokenizer,
    size: int,
    overlap: int,
    start_order: int,
) -> list[Chunk]:
    token_ids = list(tokenizer.encode(block.text, add_special_tokens=False))
    if not token_ids:
        return []
    step = size - overlap
    chunks: list[Chunk] = []
    for offset in range(0, len(token_ids), step):
        part_ids = token_ids[offset: offset + size]
        text = tokenizer.decode(part_ids, skip_special_tokens=True).strip()
        if not text:
            continue
        order = start_order + len(chunks)
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        identity = f"{document.document_id}|{document.content_hash}|{strategy}|{order}|{text_hash}"
        chunks.append(Chunk(
            chunk_id=hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32],
            strategy=strategy,
            document_id=document.document_id,
            source=document.source,
            title=document.title,
            section=block.section,
            page=block.page,
            chunk_order=order,
            token_count=len(part_ids),
            text=text,
            text_hash=text_hash,
        ))
        if offset + size >= len(token_ids):
            break
    return chunks


def _validate(size: int, overlap: int) -> None:
    if size <= 0:
        raise ValueError("Размер чанка должен быть больше нуля.")
    if overlap < 0:
        raise ValueError("Overlap не может быть отрицательным.")
    if overlap >= size:
        raise ValueError("Overlap должен быть меньше размера чанка.")
