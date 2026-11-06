"""datetime 工具：返回当前准确时间，供时间敏感类问题使用。"""
from __future__ import annotations

import datetime as dt

from langchain_core.tools import tool


@tool
def current_datetime(query: str = "") -> str:
    """返回当前日期与时间（ISO 8601，含本地时区偏移）。涉及「今天/现在/几号」等时间问题时调用。

    实现说明：先取 UTC 当前时刻，再用 .astimezone() 转换为本地时区——
    这样返回的 ISO 字符串会带时区偏移（如 2026-11-03T14:05:00+08:00），
    既准确又便于模型/用户判断时区，避免「服务器在 UTC、用户在中国」造成的日期错位。
    query 参数保留以兼容工具调用约定，本工具无需解析它。
    """
    now = dt.datetime.now(dt.timezone.utc).astimezone()
    return now.isoformat()
