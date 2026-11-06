"""集成测试：需要真实 LLM API key 和/或本地 Embeddings（Ollama）。

默认 pytest 通过 `addopts = -m 'not integration'` 排除本文件；
显式运行：pytest -m integration
无 API key 或未启动 Ollama 时，用例会自动跳过（skip），不会失败、也不会产生费用。
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration

from agent.agent import build_agent
from agent.config import Settings
from agent.llm import build_embeddings, build_llm
from agent.rag import KnowledgeBase, load_documents, split_documents


def _has_llm_key() -> bool:
    return bool(
        os.getenv("LLM_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )


def _embeddings_reachable(settings: Settings) -> bool:
    """默认 embeddings 走 Ollama；可达才进入在线流程，否则跳过。"""
    if settings.embeddings_provider != "ollama":
        return True
    try:
        emb = build_embeddings(settings)
        emb.embed_query("ping")
        return True
    except Exception:  # noqa: BLE001 — 网络/服务不可达，跳过而非失败
        return False


@pytest.fixture(scope="module")
def real_stack():
    settings = Settings()
    if not _has_llm_key():
        pytest.skip("集成测试需要 LLM_API_KEY / DEEPSEEK_API_KEY（在 .env 中配置）")
    if not _embeddings_reachable(settings):
        pytest.skip(
            "集成测试需要可用的 Embeddings（默认 Ollama；请先 "
            "`ollama pull nomic-embed-text` 并启动 Ollama）"
        )
    llm = build_llm(settings)
    embeddings = build_embeddings(settings)
    docs = split_documents(load_documents(settings.docs_dir))
    kb = KnowledgeBase.from_documents(docs, embeddings)
    return settings, llm, kb


def test_agent_answers_from_local_kb(real_stack):
    """端到端：真实 LLM + 真实向量库，应能基于本地知识库给出含确定性事实的回答。"""
    settings, llm, kb = real_stack
    agent = build_agent(llm, retriever=kb.as_retriever(settings.retriever_k))
    result = agent.invoke({"input": "星澜科技成立于哪一年？总部在哪里？"})
    answer = result["output"]
    assert answer, "Agent 必须返回非空答案"
    # 来自 company_faq.md 的确定性事实：成立 2018 年、总部杭州
    assert ("2018" in answer) or ("杭州" in answer)
