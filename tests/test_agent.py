"""Agent 主循环测试：用 LangChain Fake 模型预置工具调用，验证 Agent 决策链，**无需真实 API**。

这是 AI 测试里最关键的技巧之一：把不确定性的 LLM 替换成确定性 Fake，
使「模型是否正确地选择了工具、工具是否真的被执行」可被稳定断言。
"""
from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from agent.agent import build_agent


class FakeToolCallingLLM(FakeMessagesListChatModel):
    """FakeMessagesListChatModel 的 bind_tools 默认抛 NotImplementedError，
    导致 create_tool_calling_agent 装配失败。覆写 bind_tools 返回自身即可：
    预设的 tool_calls 仍由 responses 提供，从而可确定性地测试 Agent 决策链。"""

    def bind_tools(self, tools, **kwargs):
        return self


def _fake_llm_that_calls(tool_name: str, tool_args: dict, final_answer: str):
    """构造一个 Fake LLM：第一步要求调用指定工具，第二步给出最终答案。"""
    return FakeToolCallingLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": tool_name, "args": tool_args, "id": "call_1"}
                ],
            ),
            AIMessage(content=final_answer),
        ]
    )


def test_agent_invokes_calculator(retriever):
    llm = _fake_llm_that_calls(
        "calculator", {"expression": "2**10"}, "2 的 10 次方等于 1024。"
    )
    agent = build_agent(llm, retriever=retriever, verbose=False)
    result = agent.invoke({"input": "2 的 10 次方是多少？"})

    assert "1024" in result["output"]
    # intermediate_steps 的元素是 ToolAgentAction（langchain_classic 工具调用 Agent）
    tool_names = [getattr(step[0], "tool", None) or getattr(step[0], "name", None)
                  for step in result["intermediate_steps"]]
    assert "calculator" in tool_names


def test_agent_invokes_retriever(retriever):
    llm = _fake_llm_that_calls(
        "knowledge_search",
        {"query": "LangChain 是什么"},
        "LangChain 是一个用于构建 LLM 应用的框架。",
    )
    agent = build_agent(llm, retriever=retriever, verbose=False)
    result = agent.invoke({"input": "LangChain 是什么？"})

    assert "LangChain" in result["output"]
    tool_names = [getattr(step[0], "tool", None) or getattr(step[0], "name", None)
                  for step in result["intermediate_steps"]]
    assert "knowledge_search" in tool_names


def test_agent_without_retriever_has_no_search_tool():
    llm = _fake_llm_that_calls("calculator", {"expression": "1+1"}, "等于 2。")
    agent = build_agent(llm, retriever=None, verbose=False)
    tool_names = {t.name for t in agent.tools}
    assert "calculator" in tool_names
    assert "knowledge_search" not in tool_names
    result = agent.invoke({"input": "1+1 等于几？"})
    assert "2" in result["output"]


def test_memory_accumulates_across_turns():
    """验证多轮对话时历史会累积（每次 invoke 追加 1 条 human + 1 条 ai）。"""
    llm = FakeToolCallingLLM(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "calculator", "args": {"expression": "1+1"}, "id": "c1"}]),
            AIMessage(content="答案是 2。"),
            AIMessage(content="", tool_calls=[{"name": "calculator", "args": {"expression": "2+2"}, "id": "c2"}]),
            AIMessage(content="答案是 4。"),
        ]
    )
    agent = build_agent(llm, retriever=None, verbose=False, memory_k=10)
    assert agent.history == []
    agent.invoke({"input": "1+1 等于几？"})
    assert len(agent.history) == 2
    agent.invoke({"input": "那 2+2 呢？"})
    assert len(agent.history) == 4


def test_memory_window_trims_old_turns():
    """验证窗口裁剪：只保留最近 k 轮（k*2 条消息）。"""
    from agent.memory import window_messages
    from langchain_core.messages import AIMessage, HumanMessage

    mixed = []
    for i in range(5):
        mixed.append(HumanMessage(content=f"h{i}"))
        mixed.append(AIMessage(content=f"a{i}"))

    assert len(window_messages(mixed, k=2)) == 4
    assert window_messages(mixed, k=2)[0].content == "h3"  # 仅保留最后 2 轮
    assert window_messages(mixed, k=10) == mixed  # 窗口足够大时不丢消息
