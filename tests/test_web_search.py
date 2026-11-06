"""web_search 工具的离线边界测试。

真实联网检索依赖 ddgs + 外网，无法在离线 CI 中稳定跑。
本文件用 monkeypatch 把模块内的 `DDGS` 替换成「假检索器」，从而**不联网**
就能覆盖 web_search.py 的每一条分支：
  - 不支持的后端 -> 返回 None（优雅降级）
  - ddgs 缺失（DDGS=None）-> 返回 None
  - 正常返回 -> 格式化标题/链接/摘要
  - 空结果 -> 友好提示
  - 整体超时 -> 返回超时提示（不抛异常、不卡死）
  - ddgs 抛 TimeoutException -> 返回超时提示
  - ddgs 抛其它异常 -> 返回失败提示

⚠️ 关键点：patch 必须在「构造工具」和「调用工具」的**整个期间**生效，
因为真正的检索发生在 _run -> _search_once -> DDGS(...) 的调用时，
而非构造时。若只在构造期 patch，invoke 时 DDGS 会变回真实实现而去联网。
"""
from __future__ import annotations

import time
from unittest.mock import patch

import agent.tools.web_search as ws
from agent.tools.web_search import make_web_search_tool


def test_make_web_search_unsupported_provider_returns_none():
    # 非 duckduckgo 的后端尚未实现，应优雅降级为 None 而非报错
    assert make_web_search_tool(provider="tavily") is None


def test_make_web_search_without_ddgs_returns_none():
    # ddgs 未安装时 DDGS=None，应返回 None（由调用方决定是否启用联网）
    with patch.object(ws, "DDGS", None):
        assert make_web_search_tool() is None


def test_web_search_happy_path_formats_results():
    class _FakeDDGS:
        def __init__(self, timeout=None):
            pass

        def text(self, query, max_results=3):
            return [
                {"title": "T1", "href": "https://a.example", "body": "摘要一"},
                {"title": "T2", "href": "https://b.example", "body": "摘要二"},
            ]

    # patch 作用域必须覆盖 invoke：真正的检索发生在调用时
    with patch.object(ws, "DDGS", _FakeDDGS):
        tool = make_web_search_tool(timeout=1)
        out = tool.invoke({"query": "今天上海天气"})
    assert "联网检索结果：" in out
    assert "T1" in out and "https://a.example" in out and "摘要一" in out
    assert "T2" in out


def test_web_search_empty_results():
    class _FakeDDGS:
        def __init__(self, timeout=None):
            pass

        def text(self, query, max_results=3):
            return []

    with patch.object(ws, "DDGS", _FakeDDGS):
        tool = make_web_search_tool(timeout=1)
        out = tool.invoke({"query": "x"})
    assert "未返回相关结果" in out


def test_web_search_overall_timeout():
    # 假检索器睡眠超过整体硬性超时 -> 触发 ThreadPoolExecutor 超时分支
    class _SlowDDGS:
        def __init__(self, timeout=None):
            pass

        def text(self, query, max_results=3):
            time.sleep(5)  # 久于整体超时
            return []

    # timeout=1 -> 实际外层 timeout = max(1, 内部超时+1)=3s，约 3s 后触发超时
    with patch.object(ws, "DDGS", _SlowDDGS):
        tool = make_web_search_tool(timeout=1)
        out = tool.invoke({"query": "x"})
    assert "超时" in out


def test_web_search_ddgs_timeout_exception():
    # ddgs 内部抛出 TimeoutException -> 命中专门的超时分支
    class _BoomDDGS:
        def __init__(self, timeout=None):
            pass

        def text(self, query, max_results=3):
            raise ws.TimeoutException("ddgs backend timeout")

    with patch.object(ws, "DDGS", _BoomDDGS):
        tool = make_web_search_tool(timeout=1)
        out = tool.invoke({"query": "x"})
    assert "超时" in out


def test_web_search_generic_exception():
    # 其它网络/解析异常 -> 命中通用异常分支，返回失败提示而非崩溃
    class _BoomDDGS:
        def __init__(self, timeout=None):
            pass

        def text(self, query, max_results=3):
            raise RuntimeError("connection reset")

    with patch.object(ws, "DDGS", _BoomDDGS):
        tool = make_web_search_tool(timeout=1)
        out = tool.invoke({"query": "x"})
    assert "联网检索失败" in out
