"""Agent 图节点 — ReAct 循环中的两个核心节点。

ReAct 模式 (Reason + Act):
  1. agent 节点: LLM 推理，决定是直接回答还是调用工具
  2. tools 节点: 执行工具，将结果作为 ToolMessage 返回
  3. 回到 agent 节点，LLM 根据工具结果继续推理
  4. 重复直到 LLM 不再请求工具
"""

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode

from ai_agent.agent.state import AgentState
from ai_agent.config import settings
from ai_agent.tools import ALL_TOOLS


def create_llm() -> ChatOpenAI:
    """创建 LLM 实例，绑定工具 schema。"""
    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key or None,
        base_url=settings.openai_base_url,
        temperature=0,
    )
    return llm.bind_tools(ALL_TOOLS)


def agent_node(state: AgentState, config: RunnableConfig) -> dict:
    """Agent 推理节点 — 调用 LLM，可能产生文本回复或 tool_calls。"""
    llm = create_llm()
    messages = state["messages"]

    # 确保 system prompt 在消息列表最前面
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=settings.agent_system_prompt), *messages]

    response = llm.invoke(messages, config)
    return {"messages": [response]}


# ToolNode 是 LangGraph 预置节点，自动执行 tool_calls 并返回 ToolMessage
tools_node = ToolNode(ALL_TOOLS)
