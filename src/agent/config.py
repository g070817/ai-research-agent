"""配置层：从环境变量读取所有可插拔设置，集中管理，便于测试时覆盖。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    """全局配置。默认值来自环境变量，缺失时使用合理默认。"""

    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "deepseek"))
    llm_model: str | None = field(default_factory=lambda: os.getenv("LLM_MODEL") or None)
    llm_temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.0"))
    )
    llm_base_url: str | None = field(default_factory=lambda: os.getenv("LLM_BASE_URL") or None)
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    api_key: str | None = field(
        default_factory=lambda: (
            os.getenv("LLM_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or None
        )
    )
    embeddings_provider: str = field(
        default_factory=lambda: os.getenv("EMBEDDINGS_PROVIDER", "ollama")
    )
    kb_path: str = field(default_factory=lambda: os.getenv("KB_PATH", ".cache/kb"))
    docs_dir: str = field(default_factory=lambda: os.getenv("DOCS_DIR", "data/docs"))
    memory_k: int = field(default_factory=lambda: int(os.getenv("MEMORY_K", "6")))
    retriever_k: int = field(default_factory=lambda: int(os.getenv("RETRIEVER_K", "4")))
    verbose: bool = field(
        default_factory=lambda: os.getenv("AGENT_VERBOSE", "false").lower() == "true"
    )
    web_search_timeout: int = field(
        default_factory=lambda: int(os.getenv("WEB_SEARCH_TIMEOUT", "8"))
    )

    # 各 provider 的默认模型名
    _DEFAULT_LLM_MODELS = {"deepseek": "deepseek-chat", "openai": "gpt-4o-mini", "ollama": "qwen2.5"}
    _DEFAULT_EMB_MODELS = {"ollama": "nomic-embed-text", "openai": "text-embedding-3-small"}

    def model_for(self) -> str:
        if self.llm_model:
            return self.llm_model
        return self._DEFAULT_LLM_MODELS.get(self.llm_provider.lower(), "deepseek-chat")

    def base_url_for(self) -> str | None:
        if self.llm_base_url:
            return self.llm_base_url
        if self.llm_provider.lower() == "deepseek":
            # DeepSeek 兼容 OpenAI 接口，v1 路径已包含
            return "https://api.deepseek.com/v1"
        return None

    def embeddings_model_for(self) -> str:
        return os.getenv(
            "EMBEDDINGS_MODEL",
            self._DEFAULT_EMB_MODELS.get(self.embeddings_provider.lower(), "nomic-embed-text"),
        )


def get_settings() -> Settings:
    """返回一份 Settings 实例。测试可通过直接构造 Settings(...) 覆盖任意字段。"""
    return Settings()
