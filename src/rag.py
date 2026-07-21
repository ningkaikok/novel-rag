"""检索 + 生成：从本地 Chroma 库中检索相关片段，调用本地 Ollama 生成回答。"""
import json
from collections.abc import Iterator
from dataclasses import dataclass

import chromadb
import requests
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    TOP_K,
)

PROMPT_TEMPLATE = """你是一个小说问答助手。请仅根据下面提供的原文片段回答问题。
如果片段中没有足够信息回答，请明确说“根据提供的片段无法确定”，不要编造内容。

原文片段：
{context}

问题：{question}

回答："""


@dataclass
class SourceChunk:
    novel: str
    chunk_id: int
    text: str
    distance: float


class NovelRAG:
    def __init__(self, embedder: SentenceTransformer | None = None):
        self.embedder = embedder or SentenceTransformer(EMBEDDING_MODEL)
        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self._collection = client.get_collection(COLLECTION_NAME)

    def retrieve(self, question: str, top_k: int = TOP_K) -> list[SourceChunk]:
        query_embedding = self.embedder.encode([question], normalize_embeddings=True)
        result = self._collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
        )
        sources = []
        for text, meta, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            sources.append(
                SourceChunk(
                    novel=meta["novel"],
                    chunk_id=meta["chunk_id"],
                    text=text,
                    distance=distance,
                )
            )
        return sources

    def build_prompt(self, question: str, sources: list[SourceChunk]) -> str:
        """拼装检索片段 + 问题成完整 prompt。Ollama 和其他生成后端（如 Claude CLI）共用。"""
        context = "\n\n---\n\n".join(
            f"[{s.novel} #{s.chunk_id}]\n{s.text}" for s in sources
        )
        return PROMPT_TEMPLATE.format(context=context, question=question)

    def generate(
        self, question: str, sources: list[SourceChunk], model: str = OLLAMA_MODEL
    ) -> str:
        prompt = self.build_prompt(question, sources)
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    def generate_stream(
        self, question: str, sources: list[SourceChunk], model: str = OLLAMA_MODEL
    ) -> Iterator[str]:
        """逐字（token）流式返回回答，供界面实时展示。model 可按次调用覆盖，便于前端切换模型。"""
        prompt = self.build_prompt(question, sources)
        with requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": model, "prompt": prompt, "stream": True},
            stream=True,
            timeout=120,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line).get("response", "")
                if chunk:
                    yield chunk

    def query(
        self, question: str, top_k: int = TOP_K, model: str = OLLAMA_MODEL
    ) -> tuple[str, list[SourceChunk]]:
        sources = self.retrieve(question, top_k)
        answer = self.generate(question, sources, model=model)
        return answer, sources
