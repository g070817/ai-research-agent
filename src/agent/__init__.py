"""AI Research Agent — 一个可插拔的 LangChain 工具调用 Agent，用于本地知识库 RAG 问答。

本包同时作为「学习 agent 开发」与「AI 测试工程」的范例：
- agent 主循环基于 LangChain 的 create_tool_calling_agent（工具调用决策链）
- 支持 Memory（多轮对话）、RAG 检索、可插拔 LLM 后端
- 全部核心逻辑（工具、检索、agent 循环）均可用 Fake 模型做确定性单测
"""

__version__ = "0.1.0"
