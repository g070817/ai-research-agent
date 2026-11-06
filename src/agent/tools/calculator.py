"""calculator 工具：安全算术求值。

为什么不用 eval/exec？因为 LLM 生成的字符串直接丢给 Python 的 `eval()` 执行是严重安全隐患（命令注入）。
（注：此处的 `eval()` 是「把字符串当代码跑」的危险内置函数，和项目 `eval/` 目录里的
「evaluation 评估」、以及 `tests/` 里的「测试」不是一回事——前者执行代码，后两者是度量与验证。）
这里用 AST 白名单解析，只允许数字、基础运算符、少量数学函数与常量，从根上杜绝任意代码执行。
这本身也是 AI 测试 / AI 安全里一个很好的「工具必须防注入」范例。
"""
from __future__ import annotations

import ast
import math
import operator

from langchain_core.tools import tool

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_ALLOWED_FUNCS = {
    "sqrt": math.sqrt,
    "pow": math.pow,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "ceil": math.ceil,
    "floor": math.floor,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}
_ALLOWED_NAMES = {"pi": math.pi, "e": math.e}


def _eval_node(node: ast.AST):
    """递归遍历 AST 并「按白名单」求值——这是安全防护的核心。

    思路：AST（抽象语法树）把表达式拆成节点树，例如 'sqrt(144) + 2**3' 会被拆成：
        BinOp(+)
        ├─ Call(sqrt, [144])
        └─ BinOp(**)
           ├─ 2
           └─ 3
    我们逐节点下钻（递归）：
      - 每遇到一种节点，先检查它是否在「允许名单」里，不在立刻拒绝；
      - 对 BinOp/UnaryOp/Call，会先递归求出子节点的值，再套用对应运算。
    由于只允许数值、白名单运算符/函数/常量，像 __import__('os').system(...)
    这种节点根本不在名单内，从根上无法被执行——这就是「白名单优于黑名单」。
    """
    if isinstance(node, ast.Expression):
        # 顶层包了一层 Expression，先剥掉再处理真正的表达式。
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):  # 仅允许数值常量
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("只允许数值常量。")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        # 二元运算：先分别求左右子树的值，再套用对应运算符。
        return _ALLOWED_BINOPS[type(node.op)](
            _eval_node(node.left), _eval_node(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        # 一元运算：正负号取反等（如 -5、+3）。
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Call):
        # 函数调用：函数名必须在白名单内，参数同样递归求值。
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise ValueError("不允许调用的函数。")
        args = [_eval_node(a) for a in node.args]
        return _ALLOWED_FUNCS[node.func.id](*args)
    if isinstance(node, ast.Name):
        # 只允许 pi / e 这类常量名，其余标识符（如变量、模块）一律拒绝。
        if node.id in _ALLOWED_NAMES:
            return _ALLOWED_NAMES[node.id]
        raise ValueError(f"不允许使用的名称: {node.id}")
    raise ValueError("不支持的表达式语法。")


def safe_calculate(expression: str) -> float:
    """解析并求值一个算术表达式。非法/危险输入会抛 ValueError。

    流程：先用 ast.parse 把字符串编译成 AST（语法错误在此抛出），
    再交给 _eval_node 按白名单递归求值（语义/安全违规在此抛出）。
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        return _eval_node(tree)
    except SyntaxError as exc:
        raise ValueError(f"表达式语法错误: {exc}") from exc


@tool
def calculator(expression: str) -> str:
    """计算单个算术表达式，支持 + - * / ** % 以及 sqrt/log/exp/sin/cos/tan/pi/e。

    当用户提出任何数值计算需求时使用本工具，不要心算。
    示例输入：'sqrt(144) + 2**3'
    """
    try:
        result = safe_calculate(expression)
        return f"{expression} = {result}"
    except Exception as exc:  # noqa: BLE001 — 把错误以字符串返回给 LLM，而非崩溃
        return f"计算 '{expression}' 出错: {exc}"
