import json
import uuid
from dataclasses import asdict, dataclass
from typing import Optional, Type

import anthropic
from pydantic import BaseModel

from prompter.image_data import url_to_b64
from prompter.schemas import (
    CompletionInfo,
    ImageMessage,
    LLMResponse,
    Message,
    OutputMessage,
    Prompt,
    TextMessage,
    TextOutputMessage,
    ThoughtOutputMessage,
    Tool,
    ToolCallMessage,
    ToolCallOutputMessage,
)


@dataclass
class ClaudeModelParams:
    max_tokens: int = 1024
    model: str = "claude-3-5-sonnet-latest"


class ClaudeExecutor:
    def __init__(
        self,
        connector=None,
        model_params: Optional[ClaudeModelParams] = None,
    ):
        if connector is None:
            self.connector = anthropic.Anthropic()
        else:
            self.connector = connector
        if model_params is None:
            model_params = ClaudeModelParams()
        self.model_params = model_params

    def _message_to_message(self, input_message: Message) -> list[dict]:
        if isinstance(input_message, TextMessage):
            return [self._textmessage_to_message(input_message)]
        elif isinstance(input_message, ImageMessage):
            return [self._imagemessage_to_message(input_message)]
        elif isinstance(input_message, ToolCallMessage):
            return self._tool_call_message_to_message(input_message)

    def _textmessage_to_message(self, input_message: TextMessage) -> dict:
        return {
            "role": input_message.role,
            "content": [{"type": "text", "text": input_message.content}],
        }

    def _imagemessage_to_message(self, input_message: ImageMessage) -> dict:
        image_data = url_to_b64(input_message.url)

        return {
            "role": input_message.role,
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "data": image_data.base64_data,
                        "media_type": image_data.content_type,
                    },
                }
            ],
        }

    def _tool_call_message_to_message(
        self, input_message: ToolCallMessage
    ) -> list[dict]:
        ret = []
        tool_call_id = input_message.tool_call_id or "tooluse_" + str(uuid.uuid4())
        ret.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": input_message.tool_name,
                        "id": tool_call_id,
                        "input": input_message.arguments,
                    }
                ],
            }
        )

        ret.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": (json.dumps(input_message.result)),
                    }
                ],
            },
        )
        return ret

    def _tools_to_tools(self, input_tools: list[Tool]) -> list[dict]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.get_schema(),
            }
            for tool in input_tools
        ]

    def execute(
        self, prompt: Prompt, model_params: Optional[ClaudeModelParams] = None
    ) -> LLMResponse:
        if model_params is None:
            model_params = self.model_params
        # Flatten the list of lists using list comprehension
        messages = [
            output_msg
            for input_message in prompt.messages
            for output_msg in self._message_to_message(input_message)
        ]
        tools = self._tools_to_tools(prompt.tools) if prompt.tools else []
        raw_llm_response = self.connector.messages.create(
            system=prompt.system_message,
            messages=messages,
            tools=tools,
            **asdict(model_params),
        )

        response_messages = self._convert_api_response_to_messages(
            raw_llm_response, prompt.tools
        )

        response_usage, response_cost = self._get_api_response_meta(raw_llm_response)
        return LLMResponse(
            messages=response_messages,
            cost=response_cost,
            completion_info=response_usage,
            response_object=raw_llm_response,
        )

    def execute_format(self, prompt: Prompt, format: Type[BaseModel]):
        pass

    def _get_api_response_meta(self, response) -> tuple[CompletionInfo, float]:
        cost = self._compute_cost(response, self.model_params)

        return (
            CompletionInfo(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                finish_reason=response.stop_reason,
                refusal=(
                    response.stop_reason_details
                    if hasattr(response, "stop_reason_details")
                    else None
                ),
            ),
            cost,
        )

    def _compute_cost(self, response, model_params) -> float:
        pricing = PRICE_TABLE[model_params.model]
        input_cost = pricing.input_price * response.usage.input_tokens / pricing.per
        output_cost = pricing.output_price * response.usage.output_tokens / pricing.per
        return input_cost + output_cost

    def _convert_api_response_to_messages(
        self, response, tools: Optional[list[Tool]]
    ) -> list[OutputMessage]:
        if tools is None:
            tools = []

        def _schema_from_name(
            name: str, tools: list[Tool]
        ) -> Optional[Type[BaseModel]]:
            for tool in tools:
                if tool.name == name:
                    return tool.argument_schema
            return None

        def _content_message_to_output_message(anthropic_content):

            if anthropic_content.type == "text":
                return TextOutputMessage(content=anthropic_content.text)
            elif anthropic_content.type == "tool_use":
                schema = _schema_from_name(anthropic_content.name, tools)
                assert schema is not None, f"Tool {anthropic_content.name} not found"
                return ToolCallOutputMessage(
                    name=anthropic_content.name,
                    arguments=anthropic_content.input,
                    schema=schema,
                )
            elif (
                anthropic_content.type == "thinking"
                or anthropic_content.type == "redacted_thinking"
            ):
                return ThoughtOutputMessage(content=anthropic_content.text)
            else:
                raise ValueError(f"Unknown content type: {anthropic_content.type}")

        return [
            _content_message_to_output_message(anthropic_content)
            for anthropic_content in response.content
        ]


@dataclass
class AnthropicPricing:
    input_price: float
    output_price: float
    per: int = 1000000


DEFAULT_CLAUDE_PRICE = AnthropicPricing(
    input_price=3.0,
    output_price=15.0,
)

PRICE_TABLE = {
    "claude-3-5-sonnet-latest": DEFAULT_CLAUDE_PRICE,
    "claude-3-7-sonnet-20250219": DEFAULT_CLAUDE_PRICE,
    "claude-3-5-sonnet-20240620": DEFAULT_CLAUDE_PRICE,
    "claude-3-5-haiku-20241022": AnthropicPricing(
        input_price=0.8,
        output_price=3.0,
    ),
    "claude-3-haiku-20240307": AnthropicPricing(
        input_price=0.25,
        output_price=1.25,
    ),
}
