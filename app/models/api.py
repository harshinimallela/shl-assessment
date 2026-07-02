"""
API models: the public contract. These schemas are non-negotiable per SHL spec.

All internal models (CatalogEntry, HiringIntent) are separate — this file
contains only what crosses the API boundary.
"""
from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, field_validator, model_validator


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]

    @field_validator("messages")
    @classmethod
    def messages_not_empty(cls, v: List[Message]) -> List[Message]:
        if not v:
            raise ValueError("messages must not be empty")
        return v

    @model_validator(mode="after")
    def first_message_must_be_user(self) -> "ChatRequest":
        if self.messages and self.messages[0].role != "user":
            raise ValueError("First message must be from user")
        return self


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool
