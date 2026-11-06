"""文档摄入（ingest）：加载源文档并切分为可嵌入的 chunk（片段）。

========================================================================
为什么必须「切分」？—— 这是 RAG 检索粒度的关键
========================================================================
RAG 检索的本质是：把知识库里每段文字都转成一个向量，用户提问时也转成向量，
找「距离最近」的片段返回给 LLM 当证据。问题在于：

  - 一整篇文档（几千字）转成一个向量 → 语义被「平均」掉了，检索很模糊；
  - 切成若干 800 字左右的小片段 → 每段语义更聚焦，检索命中更精准。

所以「加载 → 切分 → 逐片段向量化」是 RAG 的标准前置流程。

切分参数（默认值）怎么理解？
  chunk_size = 800     每个片段最多约 800 字符（中英文混排大致够讲清一个小主题）
  chunk_overlap = 120  相邻片段重叠 120 字符，好处是：如果一个知识点恰好
                      被切在边界上，它至少会完整地出现在「前一段的尾部」
                      或「后一段的头部」，不会被拦腰斩断而导致两边都检索不全。

RecursiveCharacterTextSplitter 的「递归」含义：
  它先按优先级尝试分隔符（\n\n 段落 > \n 换行 > 句号 > 空格 …）切，
  尽量在「语义边界」切，而不是生硬地数到第 800 个字符就一刀。
  只有当某一段本身超过 chunk_size 时，才逐级降级到更细的分隔符继续切。
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(docs_dir: str | Path) -> list[Document]:
    """递归加载 docs_dir 下所有 .md / .txt / .markdown 文件为 Document。

    Document 是 LangChain 的统一文档结构：含 page_content（正文）与
    metadata（元数据，这里至少写入 source=相对路径，便于回答时标注出处）。
    """
    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        raise FileNotFoundError(f"文档目录不存在: {docs_dir}")

    documents: list[Document] = []
    # 只取三类纯文本后缀；sorted 保证加载顺序稳定（可复现）。
    paths = sorted(
        p for p in docs_dir.rglob("*") if p.is_file() and p.suffix.lower() in (".md", ".txt", ".markdown")
    )
    for p in paths:
        try:
            # Markdown 本质是文本，无需引入重量级 unstructured 依赖，
            # 用通用 TextLoader 读 UTF-8 即可，质量足够做知识库。
            loader = TextLoader(str(p), encoding="utf-8")
            loaded = loader.load()
            for d in loaded:
                # 记录「它来自哪个文件」，让 Agent 回答时能引用来源文件名。
                d.metadata.setdefault("source", str(p.relative_to(docs_dir)))
            documents.extend(loaded)
        except Exception as exc:  # noqa: BLE001 — 单个文件失败不应中断整库构建
            print(f"[warn] 加载失败 {p}: {exc}")
    return documents


def split_documents(
    documents: list[Document], chunk_size: int = 800, chunk_overlap: int = 120
) -> list[Document]:
    """用递归字符切分器把长文档切成重叠的 chunk，利于检索粒度。

    返回的每段 Document 会继承原文档的 metadata（含 source），
    同时 page_content 变为切出来的小片段。这些片段随后会被逐个向量化并入库。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)
