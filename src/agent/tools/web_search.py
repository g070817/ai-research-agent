"""web_search 工具：可选联网检索（仅当显式启用时使用）。

默认后端为 DuckDuckGo（通过 ddgs 库，免费、无需 API key）。

⚠️ 已知限制（中国大陆用户务必阅读）：
DuckDuckGo 的多个检索后端（startpage / mojeek / duckduckgo 等）在中国大陆常因网络不可达而超时。
ddgs 自带的 `timeout` 是「每次后端尝试」的超时，且会跨多个后端重试，因此单次调用整体可能卡很久。
为此本工具做两层防护：
  1. 内部对 ddgs 设较小的单次超时，外层再套一个「整体硬性超时」（默认 8s），确保 Agent 不会长时间卡死；
  2. 失败时返回一句人话（而非抛异常），并明确提示 LLM「不要再盲目重试」，由它基于本地知识库作答或建议用户查官方渠道。

若需要在中国大陆稳定可用，可接入基于 API key 的检索服务（如 Tavily / SerpAPI / Bing）：
在 `make_web_search_tool` 中扩展 `provider` 分支即可（见 README 的「接入可插拔检索后端」）。
"""
from __future__ import annotations

import concurrent.futures

from langchain_core.tools import StructuredTool

# ddgs 在缺失时整段导入失败；缺失时 make_web_search_tool 返回 None，由调用方决定是否启用。
try:
    from ddgs import DDGS

    try:
        from ddgs.exceptions import TimeoutException
    except ImportError:  # 老版本 ddgs 没有独立异常类时，退化为捕获通用 Exception
        TimeoutException = Exception
except Exception:  # noqa: BLE001
    DDGS = None  # type: ignore[assignment]
    TimeoutException = Exception

# 单次后端尝试的超时（秒）。ddgs 会跨多个后端重试，故取小值，外层再做整体超时。
_DDGS_INTERNAL_TIMEOUT = 2


def make_web_search_tool(
    max_results: int = 3,
    timeout: int = 8,
    provider: str = "duckduckgo",
) -> StructuredTool | None:
    """构造联网检索工具。

    参数:
        max_results: 返回结果条数
        timeout: 整体硬性超时（秒），防止 Agent 长时间卡死在不可达的外网
        provider: 检索后端；当前仅 "duckduckgo" 实现，其余值优雅降级为 None
    返回:
        StructuredTool，或在依赖缺失 / provider 不支持时返回 None
    """
    if provider != "duckduckgo":
        # 预留可插拔后端（如 tavily / serpapi）。未实现时优雅降级，不报错。
        return None
    if DDGS is None:
        return None

    def _search_once(query: str) -> list[dict]:
        ddgs = DDGS(timeout=_DDGS_INTERNAL_TIMEOUT)
        return ddgs.text(query, max_results=max_results)

    def _run(query: str) -> str:
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(_search_once, query)
                # 整体硬性超时：必须 > 内部单次超时，但尊重用户传入的更小 timeout（原代码写死 6s 会让该参数失效）
                results = future.result(timeout=max(timeout, _DDGS_INTERNAL_TIMEOUT + 1))
        except concurrent.futures.TimeoutError:
            return (
                "联网检索超时（网络不可达）。请勿重复调用本工具；"
                "若本地知识库无答案，请直接建议用户查阅官方来源（如官方网站、气象台等）。"
            )
        except TimeoutException:
            return (
                "联网检索超时（网络不可达）。请勿重复调用本工具；"
                "若本地知识库无答案，请直接建议用户查阅官方来源。"
            )
        except Exception as exc:  # noqa: BLE001 — 任何网络/解析错误都不应让 Agent 崩溃
            return (
                f"联网检索失败（{type(exc).__name__}）。请勿重复调用本工具；"
                "若本地知识库无答案，请直接建议用户查阅官方渠道。"
            )

        if not results:
            return "联网检索未返回相关结果。"
        lines = []
        for r in results[:max_results]:
            title = r.get("title", "")
            href = r.get("href", "")
            body = r.get("body", "")
            lines.append(f"- {title}\n  {href}\n  {body}")
        return "联网检索结果：\n" + "\n".join(lines)

    return StructuredTool.from_function(
        func=_run,
        name="web_search",
        description=(
            "当本地知识库与常识不足时，用于检索公开网络的最新信息。"
            "若调用后返回『超时/失败』，说明当前环境无法访问外网；"
            "此时请勿重复调用，直接基于已有信息作答或建议用户查阅官方渠道。"
        ),
    )
