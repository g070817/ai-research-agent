"""LLM / Embeddings 工厂的离线边界测试。

关键点：本文件**不触发任何真实网络调用**。
LangChain 的 ChatOpenAI / ChatOllama / OpenAIEmbeddings / OllamaEmbeddings
在「构造」时都是惰性初始化（lazy），只有真正 invoke 才会联网。
因此这里只对工厂函数做「成功构造 / 缺 key / 未知 provider」三类断言，
既能把 llm.py 的覆盖拉满，又保持离线、零成本、可重复。
"""
from __future__ import annotations

import pytest

from agent.config import Settings
from agent.llm import build_embeddings, build_llm


# ---------------- build_llm ----------------

def test_build_llm_unknown_provider_raises():
    # 未知 provider 应直接报 ValueError（llm.py 末尾的兜底分支）
    with pytest.raises(ValueError):
        build_llm(Settings(llm_provider="gemini"))


def test_build_llm_missing_api_key_raises():
    # deepseek / openai 都依赖 API key，缺失时应抛 RuntimeError 并给出清晰提示
    with pytest.raises(RuntimeError):
        build_llm(Settings(llm_provider="deepseek", api_key=None))
    with pytest.raises(RuntimeError):
        build_llm(Settings(llm_provider="openai", api_key=None))


def test_build_llm_deepseek_constructs_with_key():
    # 仅构造对象（不联网）；显式传 llm_model 以隔绝 .env 里的 LLM_MODEL 影响断言
    s = Settings(llm_provider="deepseek", api_key="dummy-key", llm_model="deepseek-chat", llm_base_url=None)
    llm = build_llm(s)
    assert llm is not None
    assert s.model_for() == "deepseek-chat"
    assert s.base_url_for() == "https://api.deepseek.com/v1"


def test_build_llm_openai_constructs_with_key():
    s = Settings(llm_provider="openai", api_key="dummy-key", llm_model="gpt-4o-mini")
    llm = build_llm(s)
    assert llm is not None
    assert s.model_for() == "gpt-4o-mini"


def test_build_llm_ollama_constructs_without_key():
    # ollama 走本地、无需 key，可完全离线构造
    s = Settings(llm_provider="ollama", api_key=None)
    llm = build_llm(s)
    assert llm is not None


# ---------------- build_embeddings ----------------

def test_build_embeddings_unknown_provider_raises():
    with pytest.raises(ValueError):
        build_embeddings(Settings(embeddings_provider="qdrant"))


def test_build_embeddings_openai_missing_key_raises():
    with pytest.raises(RuntimeError):
        build_embeddings(Settings(embeddings_provider="openai", api_key=None))


def test_build_embeddings_ollama_constructs():
    # ollama embeddings 默认本地，无需 key
    emb = build_embeddings(Settings(embeddings_provider="ollama", api_key=None))
    assert emb is not None


def test_build_embeddings_openai_constructs_with_key():
    emb = build_embeddings(Settings(embeddings_provider="openai", api_key="dummy-key"))
    assert emb is not None
