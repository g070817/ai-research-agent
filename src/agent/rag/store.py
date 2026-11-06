"""知识库（KnowledgeBase）：对 FAISS 向量库的轻封装，支持构建、加载、持久化与检索。

========================================================================
三个最该先搞懂的概念
========================================================================
1) Embeddings（嵌入模型）：一个把「文本 → 一串浮点数向量」的模型。
   语义相近的文本，向量在空间里也离得近；这就是「语义检索」的数学基础。
   本项目默认用本地 Ollama 的 nomic-embed-text（免费、离线），见 llm.py。

2) Vector（向量）：上一步产出的浮点数组，是文本在「语义空间」里的坐标。

3) FAISS：Facebook 开源的向量索引库。它把无数个向量块组织成可快速查找的
   结构，给定「查询向量」，能在毫秒级返回「最相似的 k 个片段」。

========================================================================
两条构建路径
========================================================================
- from_documents：把切好的 chunk 逐个向量化 → 建索引（首次建库用）。
- load：从磁盘反序列化已保存的索引（后续问答直接复用，不必重新向量化）。

关于 allow_dangerous_deserialization=True：
  FAISS 用 pickle 序列化索引，而 pickle 反序列化理论上可执行任意代码，
  所以 LangChain 默认禁止。本场景里索引文件是「我们自己 ingest 生成的
  .cache/kb」，来源完全可信，开启此开关是安全的——这也提醒我们：
  绝不要用它去加载来历不明的 .cache/kb 目录。
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

# langchain-community 已进入 sunset 阶段（官方 2.0 将移除该包）。在 1.x 内，其 FAISS 封装
# 仍完全可用且是当前最稳定的选择；2.0 迁移事项已记入 README。此处仅精准抑制该 sunset 通知，
# 不影响其他 DeprecationWarning，避免污染测试/运行输出。
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r".*langchain-community.*sunset.*",
        category=DeprecationWarning,
    )
    from langchain_community.vectorstores import FAISS


class KnowledgeBase:
    """对 FAISS 向量库的薄封装，只暴露项目真正需要的四个操作。"""

    def __init__(self, vectorstore: VectorStore):
        # 内部就是真实的 FAISS 实例，外面不需要关心它的具体类型。
        self.vectorstore = vectorstore

    @classmethod
    def from_documents(cls, documents: list[Document], embeddings) -> "KnowledgeBase":
        """首次建库：把每个 chunk 向量化并构建 FAISS 索引。

        documents 来自 ingest.split_documents 的输出；embeddings 是嵌入模型实例。
        FAISS.from_documents 内部会：对每段 page_content 调 embeddings 得到向量，
        再把「向量 + 原文 + metadata」一起建进索引。
        """
        vs = FAISS.from_documents(documents, embeddings)
        return cls(vs)

    @classmethod
    def load(cls, path: str | Path, embeddings) -> "KnowledgeBase":
        """从磁盘加载已保存的索引（问答阶段复用，省去重新向量化）。

        embeddings 必须与建库时同模型——FAISS 只存向量，查询时要先用同一模型
        把用户问题也转成向量，才能在同一空间里比距离。
        """
        vs = FAISS.load_local(
            str(path), embeddings, allow_dangerous_deserialization=True
        )
        return cls(vs)

    def save(self, path: str | Path) -> None:
        """把索引持久化到磁盘（自动创建父目录）。之后 load 即可复用。"""
        Path(path).mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(path))

    def as_retriever(self, k: int = 4):
        """包装成 LangChain 的 retriever 接口，供 Agent 工具调用。

        k 表示每次返回「最相似的 k 个片段」。retriever 背后就是 similarity_search。
        """
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """直接做语义检索：把 query 向量化 → 返回最相似的 k 个 chunk。

        返回的 Document 含 page_content 与 metadata（含 source 文件名），
        Agent 会把它们拼进 prompt 作为回答证据。
        """
        return self.vectorstore.similarity_search(query, k=k)
