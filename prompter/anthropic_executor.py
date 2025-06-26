from dataclasses import dataclass
from typing import Any, Optional

import anthropic

from prompter.image_data import url_to_b64
from prompter.schemas import (
    Assistant,
    Block,
    Document,
    Image,
    LLMResponse,
    Prompt,
    System,
    Text,
    Tool,
    ToolCall,
    ToolUse,
    User,
)


@dataclass
class AnthropicParams:
    max_tokens: int = 1024
    model: str = "claude-3-5-sonnet-latest"
    temperature: float = 0.0


def block_to_anthropic_content(block: Block) -> dict[str, Any] | list[dict[str, Any]]:
    if isinstance(block, User):
        return {"role": "user", "content": content_list_to_anthropic(block.content)}
    elif isinstance(block, Assistant):
        return {
            "role": "assistant",
            "content": content_list_to_anthropic(block.content),
        }
    elif isinstance(block, ToolUse):
        return tool_use_to_anthropic_messages(block)
    elif isinstance(block, ToolCall):
        return {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.arguments,
                }
            ],
        }
    elif isinstance(block, System):
        raise ValueError("System blocks should be handled separately")
    else:
        raise ValueError(f"Unknown block type: {type(block)}")


def content_list_to_anthropic(content: list) -> list[dict[str, Any]]:
    result = []
    for item in content:
        if isinstance(item, Text):
            result.append({"type": "text", "text": item.content})
        elif isinstance(item, str):
            result.append({"type": "text", "text": item})
        elif isinstance(item, Image):
            result.append(image_to_anthropic(item))
        elif isinstance(item, Document):
            raise NotImplementedError("Document support not implemented")
    return result


def image_to_anthropic(image: Image) -> dict[str, Any]:
    if image.source.startswith("data:"):
        media_type, data = image.source.split(";base64,", 1)
        media_type = media_type.replace("data:", "")
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    elif image.source.startswith(("http://", "https://")):
        image_data = url_to_b64(image.source)
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image_data.content_type,
                "data": image_data.base64_data,
            },
        }
    else:
        import base64
        from pathlib import Path

        path = Path(image.source)
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": image.media_type, "data": data},
        }


def tool_use_to_anthropic_messages(tool_use: ToolUse) -> list[dict[str, Any]]:
    messages = []
    messages.append(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_use.id,
                    "name": tool_use.name,
                    "input": tool_use.arguments,
                }
            ],
        }
    )

    content = []
    if tool_use.result is not None:
        import json

        content.append(
            {
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": (
                    json.dumps(tool_use.result)
                    if not isinstance(tool_use.result, str)
                    else tool_use.result
                ),
            }
        )
    elif tool_use.error:
        content.append(
            {
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": tool_use.error,
            }
        )

    if content:
        messages.append({"role": "user", "content": content})

    return messages


def tool_to_anthropic(tool: Tool) -> dict[str, Any]:
    schema: dict[str, Any] = {}
    if tool.params:
        if hasattr(tool.params, "model_json_schema"):
            schema = tool.params.model_json_schema()
        elif isinstance(tool.params, dict):
            schema = tool.params
        else:
            schema = {"type": "object", "properties": {}}

    return {"name": tool.name, "description": tool.description, "input_schema": schema}


def flatten_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened = []
    for msg in messages:
        if isinstance(msg, list):
            flattened.extend(msg)
        else:
            flattened.append(msg)
    return flattened


def merge_consecutive_roles(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not messages:
        return []

    merged = []
    current = messages[0].copy()

    for msg in messages[1:]:
        if msg["role"] == current["role"]:
            current["content"].extend(msg["content"])
        else:
            merged.append(current)
            current = msg.copy()

    merged.append(current)
    return merged


def parse_anthropic_response(response, tools: list[Tool]) -> LLMResponse:
    text_parts = []
    tool_calls = []

    for content in response.content:
        if content.type == "text":
            text_parts.append(content.text)
        elif content.type == "tool_use":
            tool_calls.append(
                ToolCall(name=content.name, arguments=content.input, id=content.id)
            )

    return LLMResponse(
        raw_response=response,
        tools=tools,
        _text_content=" ".join(text_parts),
        _tool_calls=tool_calls,
    )


class ClaudeExecutor:
    def __init__(self, client=None, params: Optional[AnthropicParams] = None):
        self.client = client or anthropic.Anthropic()
        self.params = params or AnthropicParams()

    def execute(
        self, prompt: Prompt, params: Optional[AnthropicParams] = None
    ) -> LLMResponse:
        params = params or self.params

        messages = []
        system_message = None

        for block in prompt.conversation:
            if isinstance(block, System):
                system_message = block.content
            else:
                msg = block_to_anthropic_content(block)
                if isinstance(msg, list):
                    messages.extend(msg)
                else:
                    messages.append(msg)

        if prompt.system:
            system_message = prompt.system

        messages = merge_consecutive_roles(messages)

        tools = None
        if prompt.tools:
            tools = [tool_to_anthropic(tool) for tool in prompt.tools]

        kwargs = {
            "model": params.model,
            "max_tokens": params.max_tokens,
            "messages": messages,
        }

        if system_message:
            kwargs["system"] = system_message

        if tools:
            kwargs["tools"] = tools

        if hasattr(params, "temperature"):
            kwargs["temperature"] = params.temperature

        response = self.client.messages.create(**kwargs)  # type: ignore

        return parse_anthropic_response(response, prompt.tools or [])
