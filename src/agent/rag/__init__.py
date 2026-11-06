"""RAG 模块：知识库的构建（ingest）与存储（store）。"""
from .store import KnowledgeBase
from .ingest import load_documents, split_documents

__all__ = ["KnowledgeBase", "load_documents", "split_documents"]
