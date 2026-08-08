"""文本切分：把整本小说切成适合检索的片段。

这是 RAG 流水线的第一步，也是最容易被低估的一步
------------------------------------------------
业界把 RAG 的质量归结为四个杠杆：**切分**、混合检索、重排、长上下文取舍。
切分排第一不是偶然——**后面所有环节都只能在切分的结果上做文章**。切错了，
再好的检索和重排也救不回来。

为什么不能整本书直接塞进向量
-----------------------------
embedding 模型有长度上限（本项目用的 `bge-small-zh-v1.5` 是 512 token）。
超出的部分**不会报错，只是静默地不进入向量**——文本在库里，却永远检索不到。
这个项目真踩过这个坑：早期版本按空行切分，切出平均 3108 字的巨块，
**每块只有开头约 500 字进了向量，后面 84% 的内容在检索中完全不可见**。

切分要平衡两件互相矛盾的事
---------------------------
- **切太大**：超过模型上限（信息丢失），且一个片段里混了多个话题，
  向量变成"什么都沾一点"的平均值，检索精度下降
- **切太小**：语义被切断（"他说" 和 "说了什么" 分到两个片段），
  单个片段脱离上下文后看不懂

本项目的策略：按段落聚合到约 `CHUNK_SIZE` 字，相邻片段保留 `CHUNK_OVERLAP`
字的重叠。重叠是为了缓解"答案正好跨在两个片段边界上"的情况——
边界附近的内容在两个片段里各出现一次，至少有一个片段包含完整语义。

> 这是**固定尺寸切分**，不是语义切分。更高级的做法（按语义边界切、
> Late Chunking 等）见 docs/rag-techniques.md。
"""
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
    """一个可检索的最小单位。

    `chunk_id` 是它在这本书里的**顺序编号**，不只是主键——检索时靠它做两件事：
    - 取相邻片段补全上下文（`rag.expand_neighbors`）
    - 定位书的开头/结尾（`rag.positional_retrieve`，回答"结局是什么"这类问题）
    """

    novel: str
    chunk_id: int
    text: str


def _read_text(path: Path) -> str:
    """按候选编码依次严格解码，全失败才降级为忽略错误字节。

    为什么不直接 `open(path)` 用默认编码：Python 默认按 UTF-8 解，遇到 GBK
    文件会抛异常或（用 errors="ignore" 时）**静默产出满篇乱码**。
    乱码不会报错，只会让检索莫名其妙搜不到——非常难查。
    本项目的《降龙》就是 GB18030 编码的，靠这个回退才读对。
    """
    raw_bytes = path.read_bytes()
    for enc in _CANDIDATE_ENCODINGS:
        try:
            return raw_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("utf-8", errors="ignore")


def _clean_text(raw: str) -> str:
    """统一换行、压缩空白，为后面的切分做准备。

    逐项的用意：
    - `﻿` 是 BOM（字节序标记），不去掉会混进第一个片段的开头
    - `\\r\\n` / `\\r` 统一成 `\\n`：Windows 和老 Mac 的换行风格不同，
      不统一的话后面按 `\\n` 切分会漏切
    - 连续空格/制表符/全角空格压成一个：小说站的 txt 常用大量空格做缩进，
      留着会白白占用片段的字数配额
    - 三个以上连续换行压成两个：保留段落间隔，但不浪费空间
    """
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

    聚合的思路是"贪心装箱"：不断往当前片段里塞段落，塞不下了就封箱、开新箱。

    **重叠（overlap）是为了什么**：答案有可能正好跨在两个片段的边界上——
    比如"墨大夫把长春功传给了"在片段 A 结尾，"韩立"在片段 B 开头。不做重叠的话
    两个片段谁都答不完整。让新片段的开头带上前一段的尾巴，边界附近的内容就在
    两个片段里各出现一次，至少有一个包含完整语义。

    代价是索引会略微变大（重叠部分被存了两次），这是标准的空间换质量。

    **硬保证：任何片段都不超过 CHUNK_SIZE**，否则超出部分不会进入 embedding
    （静默丢失，见模块 docstring 里那个 84% 内容不可见的真实案例）。
    """
    chunks: list[str] = []
    current = ""  # 正在装的那个"箱子"
    for para in paragraphs:
        # 先把超长段落拆开，避免"单段直接成为一个超限片段"
        for piece in _split_long_paragraph(para):
            candidate = f"{current}\n\n{piece}" if current else piece
            if len(candidate) <= CHUNK_SIZE:
                current = candidate  # 还装得下，继续装
                continue
            # 装不下了：把当前箱子封存，开一个新箱子
            if current:
                chunks.append(current)
            # 新箱子的开头带上旧箱子的尾巴，这就是"重叠"
            overlap_tail = current[-CHUNK_OVERLAP:] if CHUNK_OVERLAP else ""
            merged = f"{overlap_tail}\n\n{piece}" if overlap_tail else piece
            # 边界情况：重叠 + 新片可能又超限了。这时宁可放弃重叠也要守住上限——
            # 超限是硬伤（内容静默丢失），少一次重叠只是小损失。
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
