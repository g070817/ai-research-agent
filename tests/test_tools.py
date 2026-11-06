"""工具单测：纯函数工具（calculator / datetime）的确定性测试。"""
from __future__ import annotations

import datetime as dt
from unittest.mock import patch

import agent.tools.datetime_tool as dt_tool
from agent.tools import calculator, current_datetime, make_web_search_tool
from agent.tools.calculator import safe_calculate


def test_calculator_basic_arithmetic():
    out = calculator.invoke({"expression": "2 + 3 * 4"})
    assert "14" in out


def test_calculator_functions_and_constants():
    assert "12" in calculator.invoke({"expression": "sqrt(144)"})
    assert "3.14" in calculator.invoke({"expression": "round(pi, 2)"})


def test_calculator_pow_and_parens():
    out = calculator.invoke({"expression": "(2 + 3) ** 2"})
    assert "25" in out


def test_calculator_rejects_code_injection():
    # 任何尝试执行任意代码的输入都应被拒绝（抛错或返回错误，绝不执行）
    dangerous = [
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd').read()",
        "exec('import os')",
        "__import__('subprocess').call(['ls'])",
    ]
    for payload in dangerous:
        try:
            result = safe_calculate(payload)
        except Exception:
            result = None  # 抛错 = 安全
        assert result is None or "出错" in str(result)


def test_datetime_tool_returns_iso_string():
    out = current_datetime.invoke({"query": ""})
    assert "T" in out  # ISO 8601 含 T 分隔符


def test_datetime_tool_uses_real_clock_when_mocked():
    # 固定一个 UTC 瞬时：2026-11-03 03:30 UTC == 2026-11-03 11:30 +08:00
    fixed_utc = dt.datetime(2026, 11, 3, 3, 30, 0, tzinfo=dt.timezone.utc)

    # datetime.datetime 是 C 类型、不可变，不能直接 patch 其 now 方法；
    # 改为 patch 模块上的 `datetime` 属性（模块属性可重写）。
    # 关键：假 now 必须“尊重 tz 参数”——真正的 datetime.now(tz) 会返回
    # 该时区下的同一瞬时；否则 .astimezone() 会把时间错误换算成本机时区。
    class _FakeDatetime:
        @classmethod
        def now(cls, tz=None):
            return fixed_utc if tz is None else fixed_utc.astimezone(tz)

    with patch.object(dt_tool.dt, "datetime", _FakeDatetime):
        out = current_datetime.invoke({"query": ""})

    # 不依赖运行机时区：把输出解析回 UTC 瞬时，验证“被 mock 的时钟确实被采用”。
    # 断言瞬时而非本地墙钟字符串（如 "11:30"）——后者在 UTC 机器上会变 "03:30"，
    # 导致 CI（GitHub Actions 为 UTC）与本机（UTC+8）结果不一致。
    parsed = dt.datetime.fromisoformat(out)
    assert parsed.astimezone(dt.timezone.utc) == fixed_utc


def test_web_search_tool_never_raises_without_dependency():
    """联网工具在 ddgs 缺失时必须优雅返回 None，而不是抛异常导致 Agent 启动崩溃。"""
    result = make_web_search_tool()
    assert result is None or hasattr(result, "name")
