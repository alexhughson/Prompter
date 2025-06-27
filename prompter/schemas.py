from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Type
from uuid import uuid4

from pydantic import BaseModel


@dataclass
class Text:
    content: str


@dataclass
class Image:
    source: str
    media_type: str = "image/jpeg"
    detail: Literal["low", "high", "auto"] = "auto"

    @classmethod
    def file(
        cls,
        path: str | Path,
        media_type: str = "image/jpeg",
        detail: Literal["low", "high", "auto"] = "auto",
    ) -> "Image":
        return cls(source=str(path), media_type=media_type, detail=detail)

    @classmethod
    def url(cls, url: str, detail: Literal["low", "high", "auto"] = "auto") -> "Image":
        return cls(source=url, detail=detail)

    @classmethod
    def base64(cls, data: str, media_type: str = "image/jpeg") -> "Image":
        return cls(source=f"data:{media_type};base64,{data}", media_type=media_type)


@dataclass
class Document:
    source: str
    doc_type: str
    extract_text: bool = True
    cache: bool = False

    @classmethod
    def from_pdf(
        cls, path: str | Path, extract_text: bool = True, cache: bool = False
    ) -> "Document":
        return cls(
            source=str(path), doc_type="pdf", extract_text=extract_text, cache=cache
        )

    @classmethod
    def from_url(cls, url: str, cache: bool = False) -> "Document":
        return cls(source=url, doc_type="url", cache=cache)


@dataclass
class User:
    content: list[str | Text | Image | Document]

    def __init__(self, *content: str | Text | Image | Document):
        self.content = []
        for item in content:
            if isinstance(item, str):
                self.content.append(Text(item))
            else:
                self.content.append(item)


@dataclass
class Assistant:
    content: list[str | Text | Image | Document]

    def __init__(self, *content: str | Text | Image | Document):
        self.content = []
        for item in content:
            if isinstance(item, str):
                self.content.append(Text(item))
            else:
                self.content.append(item)


@dataclass
class System:
    content: str


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str | None = None

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid4())


# @dataclass
# class ToolCallResponse:
#     call: ToolCall
#     response: Any | None = None
#     error: str | None = None


@dataclass
class ToolUse:
    name: str
    arguments: Any
    result: Any | None = None
    error: str | None = None
    id: str | None = None

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid4())


@dataclass
class Tool:
    name: str
    description: str
    params: Type[BaseModel] | dict[str, Any] | None = None

    def __init__(self, name: str, description: str, params=None):
        self.name = name
        self.description = description
        self.params = params

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if isinstance(self.params, type) and issubclass(self.params, BaseModel):
            return self.params(**arguments).model_dump()
        return arguments


Block = Text | Image | Document | User | Assistant | System | ToolCall | ToolUse


@dataclass
class Prompt:
    def __init__(
        self,
        conversation: Optional[list[Block]] = None,
        system: str | None = None,
        tools: list[Tool] | None = None,
        response_format: Type[BaseModel] | None = None,
        tool_choice: Literal["auto", "none", "required"] | str = "auto",
        **kwargs,
    ):
        self.conversation = conversation or []
        self.system = system
        self.tools = tools or []
        self.response_format = response_format
        self.tool_choice = tool_choice
        self.extra = kwargs

    @classmethod
    def simple(cls, system: str, user: str) -> "Prompt":
        return cls(system=system, conversation=[User(user)])

    @classmethod
    def with_tools(
        cls, tools: list[Tool], user: str, system: str = "You are a helpful assistant"
    ) -> "Prompt":
        return cls(system=system, conversation=[User(user)], tools=tools)

    @classmethod
    def with_schema(
        cls, schema: Type[BaseModel], user: str, system: str = "Extract structured data"
    ) -> "Prompt":
        return cls(system=system, conversation=[User(user)], response_format=schema)

    def serialize(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "conversation": [block for block in self.conversation],
            "tools": [
                {"name": t.name, "description": t.description} for t in self.tools
            ],
            "response_format": (
                self.response_format.__name__ if self.response_format else None
            ),
            "tool_choice": self.tool_choice,
            **self.extra,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> "Prompt":
        return cls(**data)


@dataclass
class Message:
    content: str


@dataclass
class TextMessage:
    content: str


@dataclass
class LLMResponse:
    raw_response: Any
    tools: list[Tool]
    _text_content: str
    _tool_calls: list[ToolCall]

    def raise_for_status(self):
        pass

    def text(self) -> str:
        return self._text_content

    def messages(self) -> list[Message]:
        return [Message(content=self._text_content)] if self._text_content else []

    def tool_call(self) -> Optional[ToolCall]:
        return self._tool_calls[0] if self._tool_calls else None

    def tool_calls(self) -> list[ToolCall]:
        return self._tool_calls

    def text_messages(self) -> list[TextMessage]:
        return [TextMessage(content=self._text_content)] if self._text_content else []
