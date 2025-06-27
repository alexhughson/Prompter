from dataclasses import dataclass
from typing import Any, Optional
import json
from openai import OpenAI

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
class OpenAIParams:
    max_tokens: int = 1024
    model: str = "gpt-4o"
    temperature: float = 0.0


def block_to_openai_messages(block: Block) -> list[dict[str, Any]]:
    if isinstance(block, User):
        return [_user_block_to_openai(block)]
    elif isinstance(block, Assistant):
        if len(block.content) == 1:
            if isinstance(block.content[0], str):
                return [{"role": "assistant", "content": block.content[0]}]
            elif isinstance(block.content[0], Text):
                return [{"role": "assistant", "content": block.content[0].content}]
        else:
            content = content_list_to_openai(block.content)
            return [{"role": "assistant", "content": content}]
    elif isinstance(block, ToolUse):
        return tool_use_to_openai_messages(block)
    elif isinstance(block, ToolCall):
        return [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.arguments) if isinstance(block.arguments, dict) else block.arguments,
                        },
                    }
                ],
            }
        ]
    elif isinstance(block, System):
        return [{"role": "system", "content": block.content}]
    else:
        raise ValueError(f"Unknown block type: {type(block)}")


def _user_block_to_openai(block: User) -> dict[str, Any]:
    if len(block.content) == 1:
        if isinstance(block.content[0], str):
            return {"role": "user", "content": block.content[0]}
        elif isinstance(block.content[0], Text):
            return {"role": "user", "content": block.content[0].content}
    else:
        return {"role": "user", "content": content_list_to_openai(block.content)}


def content_list_to_openai(content: list) -> str | list[dict[str, Any]]:
    result = []
    for item in content:
        if isinstance(item, Text):
            result.append({"type": "text", "text": item.content})
        elif isinstance(item, str):
            result.append({"type": "text", "text": item})
        elif isinstance(item, Image):
            result.append(image_to_openai(item))
        elif isinstance(item, Document):
            raise NotImplementedError("Document support not implemented")
    return result


def image_to_openai(image: Image) -> dict[str, Any]:
    if image.source.startswith("data:"):
        return {
            "type": "image_url",
            "image_url": {
                "url": image.source,
                "detail": image.detail,
            },
        }
    elif image.source.startswith(("http://", "https://")):
        return {
            "type": "image_url",
            "image_url": {
                "url": image.source,
                "detail": image.detail,
            },
        }
    else:
        import base64
        from pathlib import Path

        path = Path(image.source)
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        data_url = f"data:{image.media_type};base64,{data}"
        return {
            "type": "image_url",
            "image_url": {
                "url": data_url,
                "detail": image.detail,
            },
        }


def tool_use_to_openai_messages(tool_use: ToolUse) -> list[dict[str, Any]]:
    messages = []

    messages.append(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tool_use.id,
                    "type": "function",
                    "function": {
                        "name": tool_use.name,
                        "arguments": json.dumps(tool_use.arguments) if isinstance(tool_use.arguments, dict) else tool_use.arguments,
                    },
                }
            ],
        }
    )

    content = None
    if tool_use.result is not None:
        content = (
            json.dumps(tool_use.result)
            if not isinstance(tool_use.result, str)
            else tool_use.result
        )
    elif tool_use.error:
        content = tool_use.error

    if content is not None:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_use.id,
                "content": content,
            }
        )

    return messages


def tool_to_openai(tool: Tool) -> dict[str, Any]:
    parameters = {}
    if tool.params:
        if hasattr(tool.params, "model_json_schema"):
            parameters = tool.params.model_json_schema()
        elif isinstance(tool.params, dict):
            parameters = tool.params
        else:
            parameters = {"type": "object", "properties": {}}

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters,
        },
    }


def flatten_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened = []
    for msg in messages:
        if isinstance(msg, list):
            flattened.extend(msg)
        else:
            flattened.append(msg)
    return flattened


def parse_openai_response(response, tools: list[Tool]) -> LLMResponse:
    choice = response.choices[0]
    message = choice.message

    text_content = message.content or ""
    tool_calls = []

    if hasattr(message, "tool_calls") and message.tool_calls:
        for tc in message.tool_calls:
            import json

            arguments = tc.function.arguments
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass

            tool_calls.append(
                ToolCall(
                    name=tc.function.name,
                    arguments=arguments,
                    id=tc.id,
                )
            )

    return LLMResponse(
        raw_response=response,
        tools=tools,
        _text_content=text_content,
        _tool_calls=tool_calls,
    )


class OpenAIExecutor:
    def __init__(self, client=None, params: Optional[OpenAIParams] = None):
        self.client = client or OpenAI()
        self.params = params or OpenAIParams()

    def execute(
        self, prompt: Prompt, params: Optional[OpenAIParams] = None
    ) -> LLMResponse:
        params = params or self.params

        messages = []

        for block in prompt.conversation:
            messages.extend(block_to_openai_messages(block))

        if prompt.system and not any(msg.get("role") == "system" for msg in messages):
            messages.insert(0, {"role": "system", "content": prompt.system})

        messages = flatten_messages(messages)

        kwargs = {
            "model": params.model,
            "messages": messages,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
        }

        if prompt.tools:
            kwargs["tools"] = [tool_to_openai(tool) for tool in prompt.tools]

            if prompt.tool_choice == "required":
                kwargs["tool_choice"] = "required"
            elif prompt.tool_choice == "none":
                kwargs["tool_choice"] = "none"
            elif prompt.tool_choice == "auto":
                kwargs["tool_choice"] = "auto"
            elif isinstance(prompt.tool_choice, str):
                kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": prompt.tool_choice},
                }

        if prompt.response_format:
            if hasattr(prompt.response_format, "model_json_schema"):
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": prompt.response_format.__name__,
                        "schema": prompt.response_format.model_json_schema(),
                        "strict": True,
                    },
                }
            else:
                kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)

        return parse_openai_response(response, prompt.tools or [])
