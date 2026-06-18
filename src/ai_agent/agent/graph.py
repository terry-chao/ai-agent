"""Agent 状态图 — 用 LangGraph 编排 ReAct 循环。

图结构:

    START → agent ──(有 tool_calls?)──→ tools
              ↑                            │
              └────────(继续推理)────────────┘
              │
              └──(无 tool_calls)──→ END

这是业界最经典的 Agent 控制流，LangGraph 将其显式建模为有向图，
便于调试、可视化、以及后续扩展（如 human-in-the-loop、checkpoint）。
"""

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from ai_agent.agent.nodes import agent_node, tools_node
from ai_agent.agent.state import AgentState
from ai_agent.config import settings


def _should_continue(state: AgentState) -> str:
    """路由函数 — 决定 agent 节点之后走哪条边。"""
    last = state["messages"][-1]

    if not isinstance(last, AIMessage):
        return END

    if last.tool_calls:
        return "tools"

    return END


def build_agent():
    """构建并编译 Agent 状态图。"""
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile(
        interrupt_before=[],
        debug=False,
    )


def get_graph_mermaid() -> str:
    """返回 Mermaid 格式的图结构，便于在 README 或文档中展示。"""
    agent = build_agent()
    return agent.get_graph().draw_mermaid()
