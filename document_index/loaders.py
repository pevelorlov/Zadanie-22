from __future__ import annotations

import hashlib
import html
import re
from datetime import datetime, timezone
from pathlib import Path

from .models import LoadedDocument, SourceBlock


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}


def scan_document_files(root: Path) -> list[dict]:
    root.mkdir(parents=True, exist_ok=True)
    files: list[dict] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().lower()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        stat = path.stat()
        files.append({
            "source": path.relative_to(root).as_posix(),
            "name": path.name,
            "extension": path.suffix.lower(),
            "size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        })
    return files


def load_document(path: Path, root: Path) -> LoadedDocument:
    resolved_root = root.resolve()
    resolved = path.resolve()
    if resolved_root != resolved and resolved_root not in resolved.parents:
        raise ValueError("Документ находится вне разрешённой папки.")
    extension = resolved.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Формат {extension or '(без расширения)'} не поддерживается.")

    raw = resolved.read_bytes()
    source = resolved.relative_to(resolved_root).as_posix()
    digest = hashlib.sha256(raw).hexdigest()
    document_id = hashlib.sha256(source.lower().encode("utf-8")).hexdigest()[:24]

    if extension == ".md":
        text = _decode_text(raw)
        title, blocks = _markdown_blocks(text, resolved.stem)
    elif extension in {".html", ".htm"}:
        title, blocks = _html_blocks(raw, resolved.stem)
    elif extension == ".pdf":
        title, blocks = _pdf_blocks(resolved, resolved.stem)
    elif extension == ".docx":
        title, blocks = _docx_blocks(resolved, resolved.stem)
    else:
        text = _decode_text(raw)
        title, blocks = resolved.stem, _plain_text_blocks(text)

    blocks = [SourceBlock(block.section, _clean_text(block.text), block.page) for block in blocks if _clean_text(block.text)]
    if not blocks:
        raise ValueError("В документе не найден пригодный для индексации текст.")
    return LoadedDocument(document_id, source, title.strip() or resolved.stem, extension.lstrip("."), resolved, digest, blocks)


def _decode_text(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _clean_text(value: str) -> str:
    value = html.unescape(value).replace("\u00a0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    result: list[str] = []
    blank = False
    for line in lines:
        if line:
            result.append(line)
            blank = False
        elif result and not blank:
            result.append("")
            blank = True
    return "\n".join(result).strip()


def _markdown_blocks(text: str, fallback_title: str) -> tuple[str, list[SourceBlock]]:
    headings: list[str] = []
    current: list[str] = []
    current_section = fallback_title
    title = fallback_title
    blocks: list[SourceBlock] = []

    def flush() -> None:
        nonlocal current
        body = "\n".join(current).strip()
        if body:
            blocks.append(SourceBlock(current_section, body))
        current = []

    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if not match:
            current.append(line)
            continue
        flush()
        level = len(match.group(1))
        heading = match.group(2).strip()
        if level == 1 and title == fallback_title:
            title = heading
        headings[:] = headings[: level - 1]
        headings.append(heading)
        current_section = " > ".join(headings)
        current.append(heading)
    flush()
    return title, blocks


def _plain_text_blocks(text: str) -> list[SourceBlock]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    return [SourceBlock(f"Абзац {index}", paragraph) for index, paragraph in enumerate(paragraphs, 1)]


def _html_blocks(raw: bytes, fallback_title: str) -> tuple[str, list[SourceBlock]]:
    try:
        from bs4 import BeautifulSoup
    except ImportError as error:
        raise RuntimeError("Для HTML требуется пакет beautifulsoup4.") from error

    soup = BeautifulSoup(_decode_text(raw), "html.parser")
    for node in soup(["script", "style", "noscript", "template", "svg", "canvas", "form", "button"]):
        node.decompose()
    title = soup.title.get_text(" ", strip=True) if soup.title else fallback_title
    container = soup.find("main") or soup.find("article") or soup.body or soup
    headings: list[str] = []
    current_section = title
    current: list[str] = []
    blocks: list[SourceBlock] = []

    def flush() -> None:
        nonlocal current
        body = "\n".join(current).strip()
        if body:
            blocks.append(SourceBlock(current_section, body))
        current = []

    for node in container.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "code", "td", "th"]):
        text = node.get_text(" ", strip=True)
        if not text:
            continue
        if node.name.startswith("h"):
            flush()
            level = int(node.name[1])
            headings[:] = headings[: level - 1]
            headings.append(text)
            current_section = " > ".join(headings)
            current.append(text)
        elif not any(parent.name in {"p", "li", "pre", "td", "th"} for parent in node.parents if parent is not node):
            current.append(text)
    flush()
    if not blocks:
        body = container.get_text("\n", strip=True)
        blocks = [SourceBlock(title, body)] if body else []
    return title, blocks


def _pdf_blocks(path: Path, fallback_title: str) -> tuple[str, list[SourceBlock]]:
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError("Для PDF требуется пакет pypdf.") from error
    reader = PdfReader(str(path))
    metadata_title = ""
    if reader.metadata:
        metadata_title = str(reader.metadata.get("/Title") or "").strip()
    title = metadata_title or fallback_title
    blocks: list[SourceBlock] = []
    for page_number, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        for paragraph_number, paragraph in enumerate(re.split(r"\n\s*\n+", text), 1):
            if paragraph.strip():
                blocks.append(SourceBlock(f"Страница {page_number} · блок {paragraph_number}", paragraph, page_number))
    return title, blocks


def _docx_blocks(path: Path, fallback_title: str) -> tuple[str, list[SourceBlock]]:
    try:
        from docx import Document
        from docx.table import Table
    except ImportError as error:
        raise RuntimeError("Для DOCX требуется пакет python-docx.") from error

    document = Document(str(path))
    title = str(document.core_properties.title or "").strip() or fallback_title
    headings: list[str] = []
    current_section = title
    current: list[str] = []
    blocks: list[SourceBlock] = []
    table_number = 0

    def flush() -> None:
        nonlocal current
        body = "\n".join(current).strip()
        if body:
            blocks.append(SourceBlock(current_section, body))
        current = []

    for item in document.iter_inner_content():
        if isinstance(item, Table):
            flush()
            table_number += 1
            rows: list[str] = []
            for row in item.rows:
                cells = [_clean_text(cell.text) for cell in row.cells]
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                blocks.append(SourceBlock(f"{current_section} > Таблица {table_number}", "\n".join(rows)))
            continue

        text = _clean_text(item.text)
        if not text:
            continue
        heading_level = _docx_heading_level(item)
        if heading_level is not None:
            flush()
            if heading_level == 1 and title == fallback_title:
                title = text
            headings[:] = headings[: heading_level - 1]
            headings.append(text)
            current_section = " > ".join(headings)
            current.append(text)
            continue
        if _docx_is_title(item):
            if title == fallback_title:
                title = text
            if not headings:
                current_section = title
        prefix = "• " if _docx_is_list(item) else ""
        current.append(prefix + text)

    flush()
    if title != fallback_title:
        for block in blocks:
            if block.section == fallback_title:
                block.section = title
            elif block.section.startswith(fallback_title + " > "):
                block.section = title + block.section[len(fallback_title):]
    return title, blocks


def _docx_heading_level(paragraph) -> int | None:
    style = paragraph.style
    style_id = str(getattr(style, "style_id", "") or "")
    style_name = str(getattr(style, "name", "") or "")
    match = re.fullmatch(r"Heading([1-6])", style_id, flags=re.IGNORECASE)
    if not match:
        match = re.fullmatch(r"(?:Heading|Заголовок)\s*([1-6])", style_name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _docx_is_title(paragraph) -> bool:
    style = paragraph.style
    style_id = str(getattr(style, "style_id", "") or "")
    style_name = str(getattr(style, "name", "") or "")
    return style_id.lower() == "title" or style_name.lower() in {"title", "название"}


def _docx_is_list(paragraph) -> bool:
    properties = paragraph._p.pPr
    if properties is not None and properties.numPr is not None:
        return True
    style = paragraph.style
    style_id = str(getattr(style, "style_id", "") or "").lower()
    style_name = str(getattr(style, "name", "") or "").lower()
    return style_id.startswith("list") or style_name.startswith(("list", "список"))
