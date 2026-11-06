# Agent 架构：从 ReAct 到 Tool Calling

Agent 的本质是一个「感知—思考—行动」的循环：模型根据当前状态决定下一步动作，执行后再观察结果，直到完成任务。

## ReAct 范式

ReAct（Reason + Act）让模型交替进行**推理（Thought）**与**行动（Action）**：

- **Reason**：模型用自然语言阐述当前思路。
- **Act**：模型选择一个工具并给出参数。
- **Observation**：环境返回工具执行结果，模型据此继续推理。

ReAct 把「思考过程」显式暴露出来，可解释性强，是学习 Agent 的经典入口。

## Tool Calling（工具调用）

现代 Agent 更多采用 Tool Calling（函数调用）范式：模型直接输出结构化的工具调用请求（工具名 + 参数 JSON），由运行时解析并执行。相比纯文本 ReAct，结构化调用更稳定、更易解析、更不容易出错。

## Agent 循环（Agent Loop）

一个标准工具调用 Agent 的执行流程：

1. 把用户输入、历史记忆、可用工具 schema 传给模型。
2. 模型返回：要么直接给出最终答案，要么返回一个工具调用。
3. 运行时执行该工具，把结果作为 Observation 回填。
4. 重复 2-3，直到模型给出最终答案或达到最大迭代次数。

> 关键词：ReAct、Reason、Act、Tool Calling、Agent Loop、推理、行动。
