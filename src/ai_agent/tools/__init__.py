"""Agent 工具集 — LLM 可以调用的外部能力。

每个工具用 @tool 装饰器注册，LangChain 会自动：
1. 生成 JSON Schema 供 LLM 理解参数
2. 在 ToolNode 中执行对应的 Python 函数
"""

import ast
import operator
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

# 安全的数学运算，只允许字面量和基本运算符
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    raise ValueError(f"不支持的表达式: {ast.dump(node)}")


@tool
def calculator(expression: str) -> str:
    """计算数学表达式。支持 +、-、*、/、//、%、** 和括号。

    Args:
        expression: 数学表达式，例如 "2 + 3 * 4" 或 "(10 - 2) / 4"
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval_node(tree.body)
        if result == int(result):
            return str(int(result))
        return str(result)
    except Exception as e:
        return f"计算失败: {e}"


@tool
def get_current_time() -> str:
    """获取当前的日期和时间（本地时区）。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def read_file(path: str) -> str:
    """读取文本文件内容。路径相对于当前工作目录。

    Args:
        path: 文件路径，例如 "README.md"
    """
    try:
        file_path = Path(path).resolve()
        cwd = Path.cwd().resolve()
        if not str(file_path).startswith(str(cwd)):
            return "错误: 不允许读取工作目录之外的文件"
        if not file_path.exists():
            return f"错误: 文件不存在 — {path}"
        if not file_path.is_file():
            return f"错误: 不是文件 — {path}"
        content = file_path.read_text(encoding="utf-8")
        if len(content) > 8000:
            return content[:8000] + "\n\n... (内容已截断，超过 8000 字符)"
        return content
    except Exception as e:
        return f"读取失败: {e}"


ALL_TOOLS = [calculator, get_current_time, read_file]
