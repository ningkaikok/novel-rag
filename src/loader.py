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
    """按段落切分。

    注意：不能只按空行（\\n\\n）切——国内小说站的 txt 大多用**单个换行**分段，
    整本书可能只有几千个空行。只按空行切会得到平均数千字的巨块，
    远超 embedding 模型的长度上限（bge-small-zh 为 512 token），
    导致片段后半部分在向量里完全没有表示、检索不到。
    因此按任意换行切分，段落间关系交由 _chunk_paragraphs 重新聚合。
    """
    parts = [p.strip() for p in re.split(r"\n+", text)]
    return [p for p in parts if p]


def _split_long_paragraph(para: str) -> list[str]:
    """把单段就超过 CHUNK_SIZE 的长段落按句末标点切成不超限的小片。

    优先在句末标点处断开以保住语义；实在没有标点可断（如超长无标点文本）
    则硬切，保证每片都不超过 CHUNK_SIZE。
    """
    if len(para) <= CHUNK_SIZE:
        return [para]

    # 在句末标点后断句，标点保留在前一句末尾
    sentences = re.findall(r"[^。！？…；\n]*[。！？…；]+|[^。！？…；\n]+", para)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        # 单句本身就超限：先收掉已积累的内容，再把这句硬切
        if len(sentence) > CHUNK_SIZE:
            if current:
                pieces.append(current)
                current = ""
            for i in range(0, len(sentence), CHUNK_SIZE):
                pieces.append(sentence[i : i + CHUNK_SIZE])
            continue
        if len(current) + len(sentence) <= CHUNK_SIZE:
            current += sentence
        else:
            if current:
                pieces.append(current)
            current = sentence
    if current:
        pieces.append(current)
    return pieces


def _chunk_paragraphs(paragraphs: list[str]) -> list[str]:
    """把段落聚合成约 CHUNK_SIZE 字的片段，片段间保留 CHUNK_OVERLAP 字重叠。

    硬保证：任何片段都不超过 CHUNK_SIZE，否则超出部分不会进入 embedding。
    """
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        # 先把超长段落拆开，避免"单段直接成为一个超限片段"
        for piece in _split_long_paragraph(para):
            candidate = f"{current}\n\n{piece}" if current else piece
            if len(candidate) <= CHUNK_SIZE:
                current = candidate
                continue
            if current:
                chunks.append(current)
            overlap_tail = current[-CHUNK_OVERLAP:] if CHUNK_OVERLAP else ""
            # 重叠 + 新片仍可能超限，超了就丢弃重叠只保留新片
            merged = f"{overlap_tail}\n\n{piece}" if overlap_tail else piece
            current = merged if len(merged) <= CHUNK_SIZE else piece
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
