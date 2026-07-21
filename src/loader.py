import re
from dataclasses import dataclass
from pathlib import Path

from config import CHUNK_OVERLAP, CHUNK_SIZE, NOVELS_DIR

# 国内旧小说站的 txt 绝大多数是 UTF-8 或 GBK/GB2312。优先严格尝试 UTF-8，
# 失败再严格尝试 GB18030（GBK 的超集）；两者都失败才退化为“忽略错误字节”，
# 避免把 GBK 文本当 UTF-8 硬解，静默产出满篇乱码却不报错。
_CANDIDATE_ENCODINGS = ("utf-8", "gb18030")


@dataclass
class Chunk:
    novel: str
    chunk_id: int
    text: str


def _read_text(path: Path) -> str:
    raw_bytes = path.read_bytes()
    for enc in _CANDIDATE_ENCODINGS:
        try:
            return raw_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("utf-8", errors="ignore")


def _clean_text(raw: str) -> str:
    text = raw.replace("﻿", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t　]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in text.split("\n\n")]
    return [p for p in parts if p]


def _chunk_paragraphs(paragraphs: list[str]) -> list[str]:
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= CHUNK_SIZE or not current:
            current = candidate
        else:
            chunks.append(current)
            overlap_tail = current[-CHUNK_OVERLAP:] if CHUNK_OVERLAP else ""
            current = f"{overlap_tail}\n\n{para}" if overlap_tail else para
    if current:
        chunks.append(current)
    return chunks


def load_novel_chunks(novels_dir: Path = NOVELS_DIR) -> list[Chunk]:
    """读取 novels_dir 下所有 .txt 文件，清洗并切分为 Chunk 列表。"""
    chunks: list[Chunk] = []
    for path in sorted(novels_dir.glob("*.txt")):
        raw = _read_text(path)
        text = _clean_text(raw)
        paragraphs = _split_paragraphs(text)
        for i, chunk_text in enumerate(_chunk_paragraphs(paragraphs)):
            chunks.append(Chunk(novel=path.stem, chunk_id=i, text=chunk_text))
    return chunks
