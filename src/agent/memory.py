"""对话记忆：用普通消息列表维护多轮对话（无废弃依赖）。

LangChain 1.x 中 ConversationBufferWindowMemory / ChatMessageHistory 等记忆类均已废弃，
且最终指向 LangGraph 的 Store/checkpointing（对教学项目过重）。最干净的做法是：
直接用 list[BaseMessage] 维护历史，按窗口裁剪后传入 prompt 的 chat_history。
这样既零依赖、零废弃警告，也让记忆管理对学习者完全透明。
"""
from __future__ import annotations

from langchain_core.messages import BaseMessage


def build_history() -> list[BaseMessage]:
    """构造一个空的多轮对话历史（消息列表）。"""
    return []


def window_messages(history: list[BaseMessage], k: int) -> list[BaseMessage]:
    """返回最近 k 轮对话消息（每轮 = 1 条 human + 1 条 ai，共 k*2 条）。

    为什么是 k*2？因为「一轮对话」由用户的一条 HumanMessage 和 Agent 的一条
    AIMessage 组成（见 agent.py 的 invoke）。只保留最近 k 轮，既能给模型足够
    的上下文做指代/连贯回答，又避免历史无限增长、撑爆上下文窗口或增加费用。
    取最后 k*2 条即用列表切片 history[-k*2:] 实现。
    """
    return history[-k * 2 :]
