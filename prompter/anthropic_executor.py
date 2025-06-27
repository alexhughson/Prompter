import base64
import json
from dataclasses import dataclass
from pathlib import Path
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


class ClaudeExecutor:
    class Converters:
        @staticmethod
        def block_to_anthropic_content(block: Block) -> list[dict[str, Any]]:
            block_converters = {
                User: ClaudeExecutor.Converters._convert_user_block,
                Assistant: ClaudeExecutor.Converters._convert_assistant_block,
                ToolUse: ClaudeExecutor.Converters._convert_tool_use_block,
                ToolCall: ClaudeExecutor.Converters._convert_tool_call_block,
                System: ClaudeExecutor.Converters._handle_system_block,
            }
            
            for block_type, converter in block_converters.items():
                if isinstance(block, block_type):
                    return converter(block)
            
            raise ValueError(f"Unknown block type: {type(block)}")

        @staticmethod
        def _convert_user_block(block: User) -> list[dict[str, Any]]:
            return [
                {
                    "role": "user",
                    "content": ClaudeExecutor.Converters.content_list_to_anthropic(
                        block.content
                    ),
                }
            ]

        @staticmethod
        def _convert_assistant_block(block: Assistant) -> list[dict[str, Any]]:
            return [
                {
                    "role": "assistant",
                    "content": ClaudeExecutor.Converters.content_list_to_anthropic(
                        block.content
                    ),
                }
            ]

        @staticmethod
        def _convert_tool_use_block(block: ToolUse) -> list[dict[str, Any]]:
            tool_use_msg = ClaudeExecutor.Converters._create_tool_use_message(
                block.id, block.name, block.arguments
            )
            messages = [tool_use_msg]

            if block.result is not None or block.error:
                result_msg = ClaudeExecutor.Converters._create_tool_result_message(
                    block.id, block.result, block.error
                )
                messages.append(result_msg)

            return messages

        @staticmethod
        def _convert_tool_call_block(block: ToolCall) -> list[dict[str, Any]]:
            return [ClaudeExecutor.Converters._create_tool_use_message(
                block.id, block.name, block.arguments
            )]

        @staticmethod
        def _handle_system_block(block: System) -> list[dict[str, Any]]:
            raise ValueError("System blocks should be handled separately")

        @staticmethod
        def _create_tool_use_message(
            tool_id: str, name: str, arguments: dict[str, Any]
        ) -> dict[str, Any]:
            return {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": name,
                        "input": arguments,
                    }
                ],
            }

        @staticmethod
        def _create_tool_result_message(
            tool_id: str, result: Any, error: Optional[str]
        ) -> dict[str, Any]:
            content = ClaudeExecutor.Converters._format_tool_result_content(
                tool_id, result, error
            )
            return {"role": "user", "content": content}

        @staticmethod
        def _format_tool_result_content(
            tool_id: str, result: Any, error: Optional[str]
        ) -> list[dict[str, Any]]:
            if error:
                return [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": error,
                    }
                ]
            elif result is not None:
                content = (
                    json.dumps(result) if not isinstance(result, str) else result
                )
                return [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": content,
                    }
                ]
            return []

        @staticmethod
        def content_list_to_anthropic(content: list) -> list[dict[str, Any]]:
            content_converters = {
                Text: lambda item: {"type": "text", "text": item.content},
                str: lambda item: {"type": "text", "text": item},
                Image: ClaudeExecutor.Converters._convert_image,
                Document: lambda item: ClaudeExecutor.Converters._handle_document(),
            }
            
            result = []
            for item in content:
                for content_type, converter in content_converters.items():
                    if isinstance(item, content_type):
                        result.append(converter(item))
                        break
            
            return result

        @staticmethod
        def _convert_image(image: Image) -> dict[str, Any]:
            if image.source.startswith("data:"):
                return ClaudeExecutor.Converters._convert_data_url_image(image)
            elif image.source.startswith(("http://", "https://")):
                return ClaudeExecutor.Converters._convert_url_image(image)
            else:
                return ClaudeExecutor.Converters._convert_file_image(image)

        @staticmethod
        def _convert_data_url_image(image: Image) -> dict[str, Any]:
            media_type, data = image.source.split(";base64,", 1)
            media_type = media_type.replace("data:", "")
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": data,
                },
            }

        @staticmethod
        def _convert_url_image(image: Image) -> dict[str, Any]:
            image_data = url_to_b64(image.source)
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image_data.content_type,
                    "data": image_data.base64_data,
                },
            }

        @staticmethod
        def _convert_file_image(image: Image) -> dict[str, Any]:
            path = Path(image.source)
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image.media_type,
                    "data": data,
                },
            }

        @staticmethod
        def _handle_document() -> None:
            raise NotImplementedError("Document support not implemented")

        @staticmethod
        def tool_to_anthropic(tool: Tool) -> dict[str, Any]:
            schema = ClaudeExecutor.Converters._extract_tool_schema(tool.params)
            return {
                "name": tool.name,
                "description": tool.description,
                "input_schema": schema,
            }

        @staticmethod
        def _extract_tool_schema(params: Any) -> dict[str, Any]:
            if not params:
                return {"type": "object", "properties": {}}
            
            if hasattr(params, "model_json_schema"):
                return params.model_json_schema()
            elif isinstance(params, dict):
                return params
            else:
                return {"type": "object", "properties": {}}

        @staticmethod
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


    @staticmethod
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

    def __init__(self, client=None, params: Optional[AnthropicParams] = None):
        self.client = client or anthropic.Anthropic()
        self.params = params or AnthropicParams()

    def execute(
        self, prompt: Prompt, params: Optional[AnthropicParams] = None
    ) -> LLMResponse:
        params = params or self.params

        messages = self._build_messages(prompt)
        system_message = self._extract_system_message(prompt)
        tools = self._convert_tools(prompt.tools)

        kwargs = self._build_api_kwargs(params, messages, system_message, tools)

        response = self.client.messages.create(**kwargs)  # type: ignore

        return self.parse_anthropic_response(response, prompt.tools or [])

    def _build_messages(self, prompt: Prompt) -> list[dict[str, Any]]:
        messages = []
        for block in prompt.conversation:
            if not isinstance(block, System):
                msg = self.Converters.block_to_anthropic_content(block)
                messages.extend(msg)
        return self.Converters.merge_consecutive_roles(messages)

    def _extract_system_message(self, prompt: Prompt) -> Optional[str]:
        if prompt.system:
            return prompt.system
        
        for block in prompt.conversation:
            if isinstance(block, System):
                return block.content
        
        return None

    def _convert_tools(self, tools: Optional[list[Tool]]) -> Optional[list[dict[str, Any]]]:
        if not tools:
            return None
        return [self.Converters.tool_to_anthropic(tool) for tool in tools]

    def _build_api_kwargs(
        self, 
        params: AnthropicParams, 
        messages: list[dict[str, Any]], 
        system_message: Optional[str],
        tools: Optional[list[dict[str, Any]]]
    ) -> dict[str, Any]:
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

        return kwargs


# Compatibility layer for existing code
block_to_anthropic_content = ClaudeExecutor.Converters.block_to_anthropic_content
content_list_to_anthropic = ClaudeExecutor.Converters.content_list_to_anthropic
tool_to_anthropic = ClaudeExecutor.Converters.tool_to_anthropic
merge_consecutive_roles = ClaudeExecutor.Converters.merge_consecutive_roles
parse_anthropic_response = ClaudeExecutor.parse_anthropic_response
