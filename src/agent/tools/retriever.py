"""retriever 工具：把本地知识库的相似性检索封装成一个 Agent 工具。

用工厂函数 make_retriever_tool 生成，把 retriever 通过闭包注入，
避免用全局变量，便于在测试里替换为内存型 retriever。
"""
from __future__ import annotations

from langchain_core.tools import StructuredTool


def make_retriever_tool(retriever, name: str = "knowledge_search", k: int = 4) -> StructuredTool:
    """根据给定的 retriever 生成一个可被 Agent 调用的检索工具。"""

    def _run(query: str) -> str:
        docs = retriever.invoke(query)
        if not docs:
            return "本地知识库中未检索到相关段落。"
        blocks = []
        for d in docs[:k]:
            source = d.metadata.get("source", "未知来源")
            blocks.append(f"[来源: {source}]\n{d.page_content}")
        return "\n\n".join(blocks)

    return StructuredTool.from_function(
        func=_run,
        name=name,
        description=(
            "在本地知识库中检索与用户问题相关的段落。对于知识库可能覆盖的问题，"
            "应优先调用本工具获取证据。"
        ),
    )
