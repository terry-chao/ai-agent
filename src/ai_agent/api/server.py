"""FastAPI HTTP 服务 — 将 Agent 暴露为 REST API。"""

from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from ai_agent.agent.graph import build_agent
from ai_agent.config import settings

app = FastAPI(
    title="AI Agent API",
    description="基于 LangGraph 的学习型 Agent REST API",
    version="0.1.0",
)

_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="用户消息")
    thread_id: str | None = Field(None, description="会话 ID（预留，用于 checkpoint）")


class ChatResponse(BaseModel):
    reply: str
    model: str


class HealthResponse(BaseModel):
    status: str
    model: str


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", model=settings.openai_model)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if not settings.openai_api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY 未配置")

    agent = _get_agent()
    result = agent.invoke({"messages": [HumanMessage(content=req.message)]})

    last = result["messages"][-1]
    reply = last.content if hasattr(last, "content") else str(last)

    return ChatResponse(reply=reply, model=settings.openai_model)
