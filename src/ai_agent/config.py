"""Agent 配置 — 通过环境变量加载，支持 .env 文件。"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    agent_max_iterations: int = 10
    agent_system_prompt: str = (
        "你是一个 helpful 的 AI 助手，可以使用工具来完成任务。"
        "回答时请使用中文，除非用户要求其他语言。"
    )


settings = Settings()
