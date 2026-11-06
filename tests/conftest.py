"""pytest 公共 fixtures：fake embeddings、内存知识库、retriever、golden QA。"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from agent.rag import KnowledgeBase, load_documents, split_documents

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "data" / "docs"
FIXTURES = Path(__file__).resolve().parent / "fixtures"

# 中英文混合分词：连续 ASCII 词作为一个 token，CJK 单字作为 token。
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")


def _tokens(text: str):
    return _TOKEN_RE.findall((text or "").lower())


class FakeEmbeddings(Embeddings):
    """确定性「关键词重叠」假向量：

    - 同一字符串永远得到同一向量（可复现、离线、零成本）。
    - 共享 token 越多，余弦相似度越高 —— 因此检索结果对关键词敏感，
      既验证了 RAG 管道，又不依赖真实语义向量。
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _tokens(text):
            idx = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % self.dim
            vec[idx] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def fake_embeddings():
    return FakeEmbeddings(dim=64)


@pytest.fixture
def sample_docs():
    return load_documents(DOCS_DIR)


@pytest.fixture
def sample_chunks(sample_docs):
    return split_documents(sample_docs)


@pytest.fixture
def knowledge_base(sample_chunks, fake_embeddings):
    return KnowledgeBase.from_documents(sample_chunks, fake_embeddings)


@pytest.fixture
def retriever(knowledge_base):
    # k 取大一些，确保证据片段能被检索召回（与真实语义无关，只验证管道）
    return knowledge_base.as_retriever(k=6)


@pytest.fixture
def golden_qa():
    with open(FIXTURES / "golden_qa.json", encoding="utf-8") as f:
        return json.load(f)
