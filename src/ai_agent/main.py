"""CLI 入口 — 交互式对话 + 单次查询 + 图结构展示。"""

from typing import Annotated

import typer
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from ai_agent.agent.graph import build_agent, get_graph_mermaid
from ai_agent.config import settings

app = typer.Typer(
    name="ai-agent",
    help="基于 LangGraph 的学习型 AI Agent",
    no_args_is_help=True,
)
console = Console()


def _check_api_key() -> None:
    if not settings.openai_api_key:
        console.print(
            "[yellow]警告:[/yellow] 未设置 OPENAI_API_KEY。"
            "请复制 .env.example 为 .env 并填入 API Key。"
        )
        console.print("  cp .env.example .env\n")


def _print_tool_call(msg: AIMessage) -> None:
    for tc in msg.tool_calls:
        args = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
        console.print(f"  [dim cyan]🔧 调用工具[/dim cyan] [bold]{tc['name']}[/bold]({args})")


def _print_tool_result(msg: ToolMessage) -> None:
    preview = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
    console.print(f"  [dim green]📋 工具结果[/dim green] {preview}")


def _print_response(content: str) -> None:
    console.print()
    console.print(Panel(Markdown(content), title="Agent", border_style="green"))
    console.print()


def _run_agent_streaming(user_input: str, history: list) -> str:
    """运行 Agent 并流式打印中间步骤，返回最终回复。"""
    agent = build_agent()
    history.append(HumanMessage(content=user_input))

    final_content = ""
    for event in agent.stream({"messages": history}, stream_mode="updates"):
        for node_name, update in event.items():
            if node_name == "agent":
                for msg in update.get("messages", []):
                    if isinstance(msg, AIMessage):
                        if msg.tool_calls:
                            _print_tool_call(msg)
                        elif msg.content:
                            final_content = msg.content
            elif node_name == "tools":
                for msg in update.get("messages", []):
                    if isinstance(msg, ToolMessage):
                        _print_tool_result(msg)

    # 同步 history 到最新状态
    result = agent.invoke({"messages": history})
    history.clear()
    history.extend(result["messages"])

    return final_content


@app.command()
def chat() -> None:
    """启动交互式对话模式。"""
    _check_api_key()
    console.print(Panel.fit(
        "[bold]AI Agent 交互模式[/bold]\n"
        "输入问题与 Agent 对话，输入 [dim]quit[/dim] / [dim]exit[/dim] 退出",
        border_style="blue",
    ))

    history: list = []
    while True:
        try:
            user_input = console.input("\n[bold blue]You[/bold blue] > ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            console.print("再见！")
            break

        try:
            response = _run_agent_streaming(user_input, history)
            if response:
                _print_response(response)
        except Exception as e:
            console.print(f"[red]错误:[/red] {e}")


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="要问 Agent 的问题")],
) -> None:
    """单次提问模式（非交互）。"""
    _check_api_key()
    try:
        response = _run_agent_streaming(question, [])
        if response:
            _print_response(response)
    except Exception as e:
        console.print(f"[red]错误:[/red] {e}")
        raise typer.Exit(1) from e


@app.command()
def graph() -> None:
    """展示 Agent 状态图结构（Mermaid 格式）。"""
    mermaid = get_graph_mermaid()
    console.print(Panel(mermaid, title="Agent State Graph (Mermaid)", border_style="magenta"))
    console.print("\n[dim]可将上述 Mermaid 代码粘贴到 https://mermaid.live 可视化[/dim]")


@app.command()
def tools() -> None:
    """列出 Agent 可用的工具。"""
    from ai_agent.tools import ALL_TOOLS

    table = Table(title="可用工具", show_header=True)
    table.add_column("名称", style="cyan")
    table.add_column("描述", style="white")

    for t in ALL_TOOLS:
        table.add_row(t.name, t.description or "")

    console.print(table)


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="监听地址")] = "0.0.0.0",
    port: Annotated[int, typer.Option(help="监听端口")] = 8000,
    reload: Annotated[bool, typer.Option(help="热重载")] = False,
) -> None:
    """启动 FastAPI HTTP 服务。"""
    import uvicorn

    _check_api_key()
    console.print(f"[green]启动 API 服务[/green] http://{host}:{port}")
    console.print(f"[dim]文档:[/dim] http://{host}:{port}/docs")
    uvicorn.run("ai_agent.api.server:app", host=host, port=port, reload=reload)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
