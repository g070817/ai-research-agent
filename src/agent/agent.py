"""Agent 装配：把 LLM、工具、记忆、Prompt 组装成可执行的 Agent。

========================================================================
Agent 主循环（工具调用 / Tool-Calling）是怎么一步步工作的？
========================================================================
这是理解整个项目的核心。所谓「工具调用型 Agent」，本质是让 LLM 在一个
循环里反复做「决策 → 执行 → 再看结果」：

  第 0 步  组装 prompt：
            system（角色与原则）
            + chat_history（最近几轮对话，见 memory.py）
            + input（用户本轮问题）
            + agent_scratchpad（工具调用与结果的暂存区，模型自动维护）

  第 1 步  把 prompt 交给 LLM，LLM 二选一：
            (a) 直接输出最终答案 → 循环结束；
            (b) 决定「调用某个工具」并给出参数（tool_calls），
                例如 knowledge_search("LangChain 是什么")。

  第 2 步  运行时（AgentExecutor）执行该工具，拿到结果字符串，
            把「调用 + 结果」写回 agent_scratchpad。

  第 3 步  再次把整段 prompt（含刚写入的工具结果）交给 LLM，
            LLM 据此决定：继续调下一个工具，还是已经能给出最终答案。

  ……重复第 1~3 步，直到 LLM 给出最终答案，或达到 max_iterations（本例 8 次）。

这种范式的优点：模型只负责「想调用什么、传什么参数」，真正干活的是
我们写好的工具函数；模型永不接触危险操作（如直接 eval 用户输入），安全可控。

本文件用 LangChain 的 create_tool_calling_agent 实现上面这套循环，
再用 ConversationalAgent 在外面套一层「多轮记忆」的管理。
"""
from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
# LangChain 1.x 把经典 Agent 实现（AgentExecutor / create_tool_calling_agent）
# 迁移到了 langchain_classic；功能与原 langchain.agents 完全一致。
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

from .memory import build_history, window_messages
from .prompts import SYSTEM_PROMPT
from .tools import calculator, current_datetime, make_retriever_tool, make_web_search_tool


def build_prompt() -> ChatPromptTemplate:
    """构造 Agent 的 prompt 模板，决定「信息以什么顺序喂给模型」。

    四个槽位（按模型看到的顺序）：
      1. system            —— 角色与行为原则（见 prompts.py 的 SYSTEM_PROMPT）
      2. chat_history      —— 历史对话（多轮记忆），optional 表示首轮可省略
      3. input             —— 用户本轮问题
      4. agent_scratchpad  —— LangChain 自动维护的「工具调用 ↔ 结果」暂存区

    MessagesPlaceholder 是「把一组消息原样展开塞进这里」的占位符，
    与普通的 {变量} 文本替换不同，它保留每条消息的角色（human/ai/system）。
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )


class ConversationalAgent:
    """围绕 AgentExecutor 的对话包装器，用普通消息列表维护多轮记忆。

    为什么不直接用 LangChain 旧的 ConversationBufferWindowMemory？
      —— 1.x 中它已被废弃，且最终指向 LangGraph 的 Store（对教学项目过重）。
    所以这里用最直白的方式：一个 list[BaseMessage] 存全部历史，
    每轮取最近 memory_k 轮作为 chat_history 喂给模型。逻辑完全透明、零废弃依赖。

    对外暴露 .invoke() / .ainvoke() / .tools，行为与 AgentExecutor 兼容，
    调用方（CLI）无需关心内部是「经典 Executor」还是「本包装器」。
    """

    def __init__(
        self,
        executor: AgentExecutor,
        *,
        history: list[BaseMessage] | None = None,
        memory_k: int = 6,
    ):
        self.executor = executor                      # 底层工具调用循环
        # 历史存成一个列表；首轮为空列表。memory_k 决定每轮往前看多少轮。
        self.history: list[BaseMessage] = history if history is not None else build_history()
        self.memory_k = memory_k
        # 暴露底层工具列表，便于运行时检查或测试断言。
        self.tools = executor.tools

    def _history_window(self) -> list[BaseMessage]:
        """取最近 memory_k 轮消息，作为本轮 chat_history 传入。"""
        return window_messages(self.history, self.memory_k)

    def invoke(self, inputs, config=None, **kwargs):
        """执行一轮对话（同步）。

        数据流：
          1) 从 inputs 取出用户文本；
          2) 把「问题 + 最近记忆窗口」交给底层 AgentExecutor，
             由其完成「模型决策 ↔ 工具执行」的循环，返回最终答案；
          3) 把本轮 human/ai 两条消息追加进 history，供下一轮使用。
        """
        input_text = inputs["input"] if isinstance(inputs, dict) else inputs
        result = self.executor.invoke(
            {"input": input_text, "chat_history": self._history_window()},
            config=config,
            **kwargs,
        )
        self.history.append(HumanMessage(content=input_text))
        self.history.append(AIMessage(content=result["output"]))
        return result

    async def ainvoke(self, inputs, config=None, **kwargs):
        """执行一轮对话（异步版本），逻辑与 invoke() 完全一致。"""
        input_text = inputs["input"] if isinstance(inputs, dict) else inputs
        result = await self.executor.ainvoke(
            {"input": input_text, "chat_history": self._history_window()},
            config=config,
            **kwargs,
        )
        self.history.append(HumanMessage(content=input_text))
        self.history.append(AIMessage(content=result["output"]))
        return result


def build_agent(
    llm: BaseChatModel,
    *,
    retriever=None,
    enable_web_search: bool = False,
    web_search_tool: StructuredTool | None = None,
    memory_k: int = 6,
    retriever_k: int = 4,
    verbose: bool = False,
) -> ConversationalAgent:
    """构造一个工具调用型 Agent。

    步骤：
      1) 准备工具清单：计算、时间永远有；有 retriever 就加 knowledge_search；
         显式启用联网且依赖可用就加 web_search。
      2) create_tool_calling_agent：把「模型 + 工具 + prompt」绑成一个
         知道「何时调哪个工具」的决策器。
      3) AgentExecutor：在决策器外面包一层循环控制器（max_iterations 防死循环、
         handle_parsing_errors 容错、return_intermediate_steps 暴露每一步）。

    参数:
        llm: 已构造好的 ChatModel（见 llm.py）
        retriever: 可选的向量检索器；提供后 Agent 获得 knowledge_search 工具
        enable_web_search: 是否启用联网检索（需网络）
        web_search_tool: 外部已构造好的联网工具（None 表示不启用）
        memory_k: 记忆窗口轮数（仅保留最近 k 轮，避免上下文无限膨胀）
        retriever_k: 每次检索返回段落数
        verbose: 是否打印 Agent 的思考/调用过程（调试用）
    """
    # 1) 基础工具：任何问题都可能用到计算与时间，始终注册。
    tools = [calculator, current_datetime]
    # 2) 有知识库就给 Agent「查知识库」的能力。
    if retriever is not None:
        tools.append(make_retriever_tool(retriever, k=retriever_k))
    # 3) 显式 --web 且依赖已安装时，补充联网检索能力。
    if enable_web_search:
        ws = web_search_tool if web_search_tool is not None else make_web_search_tool()
        if ws is not None:
            tools.append(ws)

    # 把模型、工具、提示词组装成「决策器」。
    agent = create_tool_calling_agent(llm, tools, build_prompt())
    # 用 Executor 驱动循环：最多 8 轮，避免模型反复调工具卡死；
    # 解析失败时尝试自我纠正，而不是抛异常中断。
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        max_iterations=8,
    )
    return ConversationalAgent(executor, memory_k=memory_k)
