from datetime import datetime

from pydantic import BaseModel, Field


class AgentPromptDTO(BaseModel):
    agent_id: str
    content: str
    version: int
    updated_at: datetime
    updated_by: str | None = None


class PromptUpdateRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=8000)
    updated_by: str | None = Field(default=None, max_length=128)


class PromptListResponse(BaseModel):
    prompts: list[AgentPromptDTO]
