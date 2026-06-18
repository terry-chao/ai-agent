"""工具单元测试 — 不依赖 LLM API。"""

from ai_agent.tools import calculator, get_current_time, read_file


def test_calculator_basic():
    assert calculator.invoke({"expression": "2 + 3"}) == "5"


def test_calculator_complex():
    assert calculator.invoke({"expression": "(10 - 2) / 4"}) == "2"


def test_calculator_power():
    assert calculator.invoke({"expression": "2 ** 10"}) == "1024"


def test_calculator_invalid():
    result = calculator.invoke({"expression": "import os"})
    assert "失败" in result


def test_get_current_time():
    result = get_current_time.invoke({})
    assert len(result) == 19  # "YYYY-MM-DD HH:MM:SS"


def test_read_file_exists():
    result = read_file.invoke({"path": "README.md"})
    assert "ai-agent" in result.lower() or "agent" in result.lower()


def test_read_file_not_found():
    result = read_file.invoke({"path": "nonexistent_file_xyz.txt"})
    assert "不存在" in result
