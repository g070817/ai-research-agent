"""命令行入口（Typer）：ingest / chat / ask 三个子命令。

========================================================================
三条命令共享的装配流程
========================================================================
每个命令都遵循「读配置 → 造模型 → 准备知识库 → 组装 Agent → 执行」的套路：

  ingest  只做知识库：读文档 → 切分 → 向量化 → 存盘 .cache/kb（不需要 LLM）
  ask     单轮问答：加载知识库 → 造 LLM → 组装 Agent → 问一个问题
  chat    多轮对话：同 ask，但进入循环，每轮把 human/ai 写入记忆（见 agent.py）

为减少重复，本文件把「易失败的步骤」包成 *_or_exit 辅助函数：初始化 LLM /
Embeddings 时一旦缺少 key 或 Ollama 未启动，就打印友好提示并干净退出，
而不是把原始异常栈甩给用户。
"""
from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown

from .agent import build_agent
from .config import Settings, get_settings
from .llm import build_embeddings, build_llm
from .rag import KnowledgeBase, load_documents, split_documents
from .tools import make_web_search_tool

app = typer.Typer(
    help="AI Research Agent —— 基于 LangChain 的本地知识库 RAG 问答 Agent（可插拔 LLM）。"
)
console = Console()


def _build_llm_or_exit(settings: Settings):
    try:
        return build_llm(settings)
    except RuntimeError as exc:
        console.print(f"[red]无法初始化 LLM: {exc}[/red]")
        raise typer.Exit(code=1)


def _build_embeddings_or_exit(settings: Settings):
    """构造 Embeddings，失败时用友好提示替代原始异常栈。"""
    try:
        return build_embeddings(settings)
    except RuntimeError as exc:
        console.print(f"[red]无法初始化 Embeddings: {exc}[/red]")
        raise typer.Exit(code=1)
    except Exception as exc:  # 例如 Ollama 未启动 / 模型未拉取
        _report_embedding_error(exc, settings)
        raise typer.Exit(code=1)


def _report_embedding_error(exc: Exception, settings: Settings) -> None:
    msg = str(exc)
    model = settings.embeddings_model_for()
    if "not found" in msg.lower() or "pull" in msg.lower():
        console.print(
            f"[red]Embedding 模型 '{model}' 未找到。[/red]\n"
            f"请先拉取模型（只需一次）：\n"
            f"  [bold cyan]ollama pull {model}[/bold cyan]\n"
            f"然后重新运行本命令。"
        )
    elif "connection" in msg.lower() or "connect" in msg.lower():
        console.print(
            "[red]无法连接 Ollama。[/red]\n"
            "请确认 Ollama 已启动（运行 [bold cyan]ollama serve[/bold cyan] 或打开 Ollama 桌面应用），\n"
            f"或检查环境变量 OLLAMA_BASE_URL（当前：{settings.ollama_base_url}）。"
        )
    else:
        console.print(f"[red]初始化 Embeddings 失败: {exc}[/red]")


def _resolve_web_tool(web: bool, settings: Settings):
    """解析联网搜索工具；依赖缺失时给友好提示并返回 None（Agent 退化为仅本地知识库）。"""
    if not web:
        return None
    tool = make_web_search_tool(timeout=settings.web_search_timeout)
    if tool is None:
        console.print(
            "[yellow]⚠️ 已指定 --web，但联网搜索依赖未安装。"
            "运行 [bold cyan]pip install ddgs[/bold cyan] 后重试，本次将仅使用本地知识库。[/yellow]"
        )
    return tool


def _load_kb(settings: Settings):
    embeddings = _build_embeddings_or_exit(settings)
    if not Path(settings.kb_path).exists():
        console.print(
            f"[yellow]未找到知识库（{settings.kb_path}）。请先运行: python -m agent ingest[/yellow]"
        )
        return None, embeddings
    return KnowledgeBase.load(settings.kb_path, embeddings), embeddings


@app.command()
def ingest(
    docs_dir: str = typer.Option(None, "--docs", "-d", help="源文档目录"),
    kb_path: str = typer.Option(None, "--out", "-o", help="向量库输出路径"),
):
    """从文档构建（或重建）本地知识库。"""
    settings = get_settings()
    docs_dir = docs_dir or settings.docs_dir
    kb_path = kb_path or settings.kb_path
    embeddings = build_embeddings(settings)

    console.print(f"正在从 [cyan]{docs_dir}[/cyan] 加载文档 ...")
    raw = load_documents(docs_dir)
    chunks = split_documents(raw)
    console.print(f"已加载 {len(raw)} 篇文档，切分为 {len(chunks)} 个 chunk。")

    try:
        kb = KnowledgeBase.from_documents(chunks, embeddings)
    except Exception as exc:
        _report_embedding_error(exc, settings)
        raise typer.Exit(code=1)
    kb.save(kb_path)
    console.print(f"[green]知识库已保存到 {kb_path}[/green]")


@app.command()
def chat(
    web: bool = typer.Option(False, "--web", help="启用联网检索工具（需网络）"),
):
    """进入交互式对话（支持多轮记忆）。"""
    settings = get_settings()
    llm = _build_llm_or_exit(settings)
    kb, _ = _load_kb(settings)
    retriever = kb.as_retriever(settings.retriever_k) if kb else None

    web_tool = _resolve_web_tool(web, settings)
    agent = build_agent(
        llm,
        retriever=retriever,
        enable_web_search=web,
        web_search_tool=web_tool,
        memory_k=settings.memory_k,
        retriever_k=settings.retriever_k,
        verbose=settings.verbose,
    )

    console.print("[bold cyan]AI Research Agent[/bold cyan] —— 输入 exit/quit 退出，Ctrl+C 中断。")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]再见。[/dim]")
            break
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        result = agent.invoke({"input": q})
        console.print(Markdown(result["output"]))


@app.command()
def ask(
    question: str = typer.Argument(..., help="要提问的问题"),
    web: bool = typer.Option(False, "--web", help="启用联网检索工具（需网络）"),
):
    """对单个问题给出回答。"""
    settings = get_settings()
    llm = _build_llm_or_exit(settings)
    kb, _ = _load_kb(settings)
    retriever = kb.as_retriever(settings.retriever_k) if kb else None

    web_tool = _resolve_web_tool(web, settings)
    agent = build_agent(
        llm,
        retriever=retriever,
        enable_web_search=web,
        web_search_tool=web_tool,
        memory_k=settings.memory_k,
        retriever_k=settings.retriever_k,
        verbose=settings.verbose,
    )
    result = agent.invoke({"input": question})
    console.print(Markdown(result["output"]))


if __name__ == "__main__":
    sys.exit(app())
