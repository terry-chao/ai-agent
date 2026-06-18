"""Agent 状态定义 — LangGraph 的核心数据结构。

Agent 的每一步都会读取/更新这个 State，并在节点之间传递。
`add_messages` reducer 会自动将新消息追加到历史记录，而不是覆盖。
"""

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Agent 运行时状态。

    Attributes:
        messages: 对话历史（Human / AI / Tool 消息），使用 add_messages 自动合并。
    """

    messages: Annotated[list[BaseMessage], add_messages]
