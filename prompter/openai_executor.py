import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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


class OpenAIExecutor:
    class Converters:
        @staticmethod
        def block_to_openai_messages(block: Block) -> list[dict[str, Any]]:
            block_converters = {
                User: OpenAIExecutor.Converters._convert_user_block,
                Assistant: OpenAIExecutor.Converters._convert_assistant_block,
                ToolUse: OpenAIExecutor.Converters._convert_tool_use_block,
                ToolCall: OpenAIExecutor.Converters._convert_tool_call_block,
                System: OpenAIExecutor.Converters._convert_system_block,
            }
            
            for block_type, converter in block_converters.items():
                if isinstance(block, block_type):
                    return converter(block)
            
            raise ValueError(f"Unknown block type: {type(block)}")

        @staticmethod
        def _convert_user_block(block: User) -> list[dict[str, Any]]:
            return [OpenAIExecutor.Converters._build_user_message(block)]

        @staticmethod
        def _convert_assistant_block(block: Assistant) -> list[dict[str, Any]]:
            content = OpenAIExecutor.Converters._process_assistant_content(block.content)
            return [{"role": "assistant", "content": content}]

        @staticmethod
        def _convert_tool_use_block(block: ToolUse) -> list[dict[str, Any]]:
            tool_call_msg = OpenAIExecutor.Converters._create_tool_call_message(
                block.id, block.name, block.arguments
            )
            messages = [tool_call_msg]

            tool_result = OpenAIExecutor.Converters._create_tool_result(
                block.id, block.result, block.error
            )
            if tool_result:
                messages.append(tool_result)

            return messages

        @staticmethod
        def _convert_tool_call_block(block: ToolCall) -> list[dict[str, Any]]:
            return [OpenAIExecutor.Converters._create_tool_call_message(
                block.id, block.name, block.arguments
            )]

        @staticmethod
        def _convert_system_block(block: System) -> list[dict[str, Any]]:
            return [{"role": "system", "content": block.content}]

        @staticmethod
        def _build_user_message(block: User) -> dict[str, Any]:
            content = OpenAIExecutor.Converters._process_user_content(block.content)
            return {"role": "user", "content": content}

        @staticmethod
        def _process_user_content(content: list) -> str | list[dict[str, Any]]:
            if len(content) == 1:
                item = content[0]
                if isinstance(item, str):
                    return item
                elif isinstance(item, Text):
                    return item.content
            
            return OpenAIExecutor.Converters.content_list_to_openai(content)

        @staticmethod
        def _process_assistant_content(content: list) -> str | list[dict[str, Any]]:
            if len(content) == 1:
                item = content[0]
                if isinstance(item, str):
                    return item
                elif isinstance(item, Text):
                    return item.content
            
            return OpenAIExecutor.Converters.content_list_to_openai(content)

        @staticmethod
        def _create_tool_call_message(
            tool_id: str, name: str, arguments: Any
        ) -> dict[str, Any]:
            args_str = (
                json.dumps(arguments) if isinstance(arguments, dict) else arguments
            )
            return {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": args_str,
                        },
                    }
                ],
            }

        @staticmethod
        def _create_tool_result(
            tool_id: str, result: Any, error: Optional[str]
        ) -> Optional[dict[str, Any]]:
            content = OpenAIExecutor.Converters._format_tool_result_content(result, error)
            if content is not None:
                return {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": content,
                }
            return None

        @staticmethod
        def _format_tool_result_content(result: Any, error: Optional[str]) -> Optional[str]:
            if error:
                return error
            elif result is not None:
                return json.dumps(result) if not isinstance(result, str) else result
            return None

        @staticmethod
        def content_list_to_openai(content: list) -> list[dict[str, Any]]:
            content_converters = {
                Text: lambda item: {"type": "text", "text": item.content},
                str: lambda item: {"type": "text", "text": item},
                Image: OpenAIExecutor.Converters._convert_image,
                Document: lambda item: OpenAIExecutor.Converters._handle_document(),
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
                return OpenAIExecutor.Converters._create_image_url_dict(
                    image.source, image.detail
                )
            elif image.source.startswith(("http://", "https://")):
                return OpenAIExecutor.Converters._create_image_url_dict(
                    image.source, image.detail
                )
            else:
                data_url = OpenAIExecutor.Converters._file_to_data_url(image)
                return OpenAIExecutor.Converters._create_image_url_dict(
                    data_url, image.detail
                )

        @staticmethod
        def _create_image_url_dict(url: str, detail: str) -> dict[str, Any]:
            return {
                "type": "image_url",
                "image_url": {
                    "url": url,
                    "detail": detail,
                },
            }

        @staticmethod
        def _file_to_data_url(image: Image) -> str:
            path = Path(image.source)
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            return f"data:{image.media_type};base64,{data}"

        @staticmethod
        def _handle_document() -> None:
            raise NotImplementedError("Document support not implemented")

        @staticmethod
        def tool_to_openai(tool: Tool) -> dict[str, Any]:
            parameters = OpenAIExecutor.Converters._extract_tool_parameters(tool.params)
            return {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": parameters,
                },
            }

        @staticmethod
        def _extract_tool_parameters(params: Any) -> dict[str, Any]:
            if not params:
                return {"type": "object", "properties": {}}
            
            if hasattr(params, "model_json_schema"):
                return params.model_json_schema()
            elif isinstance(params, dict):
                return params
            else:
                return {"type": "object", "properties": {}}

        @staticmethod
        def flatten_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
            flattened = []
            for msg in messages:
                if isinstance(msg, list):
                    flattened.extend(msg)
                else:
                    flattened.append(msg)
            return flattened


    @staticmethod
    def parse_openai_response(response, tools: list[Tool]) -> LLMResponse:
        choice = response.choices[0]
        message = choice.message

        text_content = message.content or ""
        tool_calls = OpenAIExecutor._extract_tool_calls(message)

        return LLMResponse(
            raw_response=response,
            tools=tools,
            _text_content=text_content,
            _tool_calls=tool_calls,
        )

    @staticmethod
    def _extract_tool_calls(message) -> list[ToolCall]:
        tool_calls = []
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                arguments = OpenAIExecutor._parse_tool_arguments(tc.function.arguments)
                tool_calls.append(
                    ToolCall(
                        name=tc.function.name,
                        arguments=arguments,
                        id=tc.id,
                    )
                )
        return tool_calls

    @staticmethod
    def _parse_tool_arguments(arguments: Any) -> Any:
        if isinstance(arguments, str):
            try:
                return json.loads(arguments)
            except json.JSONDecodeError:
                return arguments
        return arguments

    def __init__(self, client=None, params: Optional[OpenAIParams] = None):
        self.client = client or OpenAI()
        self.params = params or OpenAIParams()

    def execute(
        self, prompt: Prompt, params: Optional[OpenAIParams] = None
    ) -> LLMResponse:
        params = params or self.params

        messages = self._build_messages(prompt)
        kwargs = self._build_api_kwargs(params, messages, prompt)

        response = self.client.chat.completions.create(**kwargs)

        return self.parse_openai_response(response, prompt.tools or [])

    def _build_messages(self, prompt: Prompt) -> list[dict[str, Any]]:
        messages = []
        
        for block in prompt.conversation:
            messages.extend(self.Converters.block_to_openai_messages(block))

        if prompt.system and not self._has_system_message(messages):
            messages.insert(0, {"role": "system", "content": prompt.system})

        return self.Converters.flatten_messages(messages)

    def _has_system_message(self, messages: list[dict[str, Any]]) -> bool:
        return any(msg.get("role") == "system" for msg in messages)

    def _build_api_kwargs(
        self, 
        params: OpenAIParams, 
        messages: list[dict[str, Any]], 
        prompt: Prompt
    ) -> dict[str, Any]:
        kwargs = {
            "model": params.model,
            "messages": messages,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
        }

        if prompt.tools:
            kwargs["tools"] = self._convert_tools(prompt.tools)
            tool_choice = self._format_tool_choice(prompt.tool_choice)
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

        response_format = self._format_response_format(prompt.response_format)
        if response_format:
            kwargs["response_format"] = response_format

        return kwargs

    def _convert_tools(self, tools: list[Tool]) -> list[dict[str, Any]]:
        return [self.Converters.tool_to_openai(tool) for tool in tools]

    def _format_tool_choice(self, tool_choice: Optional[str]) -> Optional[Any]:
        if tool_choice in ["required", "none", "auto"]:
            return tool_choice
        elif isinstance(tool_choice, str):
            return {
                "type": "function",
                "function": {"name": tool_choice},
            }
        return None

    def _format_response_format(self, response_format: Any) -> Optional[dict[str, Any]]:
        if not response_format:
            return None
        
        if hasattr(response_format, "model_json_schema"):
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "schema": response_format.model_json_schema(),
                    "strict": True,
                },
            }
        else:
            return {"type": "json_object"}


# Compatibility layer for existing code
block_to_openai_messages = OpenAIExecutor.Converters.block_to_openai_messages
content_list_to_openai = OpenAIExecutor.Converters.content_list_to_openai
tool_to_openai = OpenAIExecutor.Converters.tool_to_openai
flatten_messages = OpenAIExecutor.Converters.flatten_messages
parse_openai_response = OpenAIExecutor.parse_openai_response
