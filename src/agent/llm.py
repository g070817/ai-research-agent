"""LLM 与 Embeddings 工厂：根据 Settings 构造可插拔的模型后端。

支持三种 LLM 后端：
- deepseek：通过 OpenAI 兼容接口访问（默认，国内可用、成本低）
- openai：GPT 系列
- ollama：本地模型（免费、离线）

Embeddings 支持 ollama（本地，默认）与 openai。
"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from .config import Settings


def build_llm(settings: Settings) -> BaseChatModel:
    """根据配置构造一个 ChatModel。"""
    provider = settings.llm_provider.lower()

    if provider in ("deepseek", "openai"):
        from langchain_openai import ChatOpenAI

        if not settings.api_key:
            raise RuntimeError(
                f"LLM provider '{provider}' 需要 API key。"
                "请在 .env 中设置 LLM_API_KEY（或 DEEPSEEK_API_KEY / OPENAI_API_KEY）。"
            )
        return ChatOpenAI(
            model=settings.model_for(),
            temperature=settings.llm_temperature,
            api_key=settings.api_key,
            base_url=settings.base_url_for(),
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.model_for(),
            temperature=settings.llm_temperature,
            base_url=settings.ollama_base_url,
        )

    raise ValueError(f"未知的 LLM provider: {provider}。请使用 deepseek | openai | ollama。")


def build_embeddings(settings: Settings):
    """根据配置构造一个 Embeddings 模型，用于 RAG 检索。"""
    provider = settings.embeddings_provider.lower()

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=settings.embeddings_model_for(),
            base_url=settings.ollama_base_url,
        )

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        if not settings.api_key:
            raise RuntimeError("使用 OpenAI embeddings 需要设置 LLM_API_KEY。")
        return OpenAIEmbeddings(
            model=settings.embeddings_model_for(),
            api_key=settings.api_key,
        )

    raise ValueError(f"未知的 embeddings provider: {provider}。请使用 ollama | openai。")
