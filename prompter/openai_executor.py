import json
import os
import uuid
from typing import Dict, List

from openai import OpenAI

from .base_executor import BaseLLMExecutor
from .schemas import (
    ImageMessage,
    SchemaResult,
    LLMResponse,
    ModelPricing,
    Prompt,
    TextOutputMessage,
    Tool,
    ToolCallMessage,
    ToolCallOutputMessage,
    UsageInfo,
    CompletionInfo,
    ResponseFormat,
)


class OpenAIExecutor(BaseLLMExecutor):
    """OpenAI-specific implementation of the LLM executor"""

    def __init__(self, api_key: str = None, model: str = "gpt-4o", **kwargs):
        super().__init__(api_key, model, **kwargs)
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        self.client = OpenAI(api_key=api_key)
        self.pricing = self._get_model_pricing()

    def _get_model_pricing(self) -> ModelPricing:
        """Get the pricing for the current model"""
        prices = {
            "gpt-4-turbo-preview": ModelPricing(
                input_price=0.01, output_price=0.03  # per 1K tokens  # per 1K tokens
            ),
            "gpt-4": ModelPricing(input_price=0.03, output_price=0.06),
            "gpt-3.5-turbo": ModelPricing(input_price=0.0005, output_price=0.0015),
        }
        return prices.get(self.model, ModelPricing(input_price=0.0, output_price=0.0))

    def get_api_params(self, prompt: Prompt) -> Dict:
        """Get the API parameters for the prompt"""
        messages = self._convert_prompt_to_api_messages(prompt)
        params = {"model": self.model, "messages": messages, **self.kwargs}

        # Handle response formats
        if prompt.response_schema:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": prompt.response_schema.model_json_schema(),
                },
            }
            # if self._supports_structured_output():
            #     params["response_format"] = prompt.response_schema
            # else:
            #     # Fall back to JSON mode for older models
            #     params["response_format"] = {"type": "json_object"}
            #     if not any("JSON" in msg["content"] for msg in messages):
            #         # Add JSON instruction if not present
            #         messages.insert(
            #             0,
            #             {
            #                 "role": "system",
            #                 "content": "Please respond with valid JSON matching the required schema.",
            #             },
            #         )

        if prompt.tools:
            params["tools"] = self._convert_tools_to_api_format(prompt.tools)
            params["tool_choice"] = "auto"

        return params

    def execute(self, prompt: Prompt) -> LLMResponse:
        """Execute the prompt using OpenAI's API"""
        params = self.get_api_params(prompt)
        response = self.client.chat.completions.create(**params)

        return self._convert_api_response_to_messages(prompt, response)

    def _calculate_cost(self, usage: UsageInfo) -> float:
        """Calculate the cost based on OpenAI's pricing"""
        prompt_cost = (usage.prompt_tokens / 1000) * self.pricing.input_price
        completion_cost = (usage.completion_tokens / 1000) * self.pricing.output_price
        return prompt_cost + completion_cost

    def _convert_prompt_to_api_messages(self, prompt: Prompt) -> list:
        """Convert our prompt format to OpenAI's expected format"""
        messages = []

        # Add system message if present
        if prompt.system_message:
            messages.append({"role": "system", "content": prompt.system_message})

        # Add conversation messages
        for msg in prompt.messages:
            if msg.message_type() == "image":
                messages.append(self._convert_imagemessage_to_api_format(msg))
            elif msg.message_type() == "text":
                messages.append({"role": msg.role, "content": msg.content})
            elif msg.message_type() == "tool_call":
                messages.extend(self._convert_tool_call_to_api_format(msg))
            else:
                raise ValueError(f"Unsupported message type: {msg.message_type()}")

        return messages

    def _convert_imagemessage_to_api_format(self, image: ImageMessage) -> Dict:
        content = []
        if image.content:
            content.append({"type": "text", "text": image.content})
        content.append({"type": "image_url", "image_url": {"url": image.url}})
        return {
            "role": image.role,
            "content": content,
        }

    def _convert_tool_call_to_api_format(self, tool_call: ToolCallMessage) -> Dict:
        tool_call_id = tool_call.tool_call_id or str(uuid.uuid4())
        return [
            {
                "role": tool_call.role,
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_call.tool_name,
                            "arguments": json.dumps(tool_call.arguments),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": (json.dumps(tool_call.result)),
                "tool_call_id": tool_call_id,
            },
        ]

    def _convert_tools_to_api_format(self, tools: List[Tool]) -> List[Dict]:
        """Convert our tool format to OpenAI's expected format"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.get_schema(),
                },
            }
            for tool in tools
        ]

    def _convert_api_response_to_messages(
        self, prompt: Prompt, response
    ) -> LLMResponse:
        """Convert OpenAI's response format to our internal format"""
        messages = []
        tool_schemas = {tool.name: tool.argument_schema for tool in prompt.tools}
        finish_type = {
            "stop": CompletionInfo.FinishType.SUCCESS,
            "tool_calls": CompletionInfo.FinishType.TOOL_USE,
            "length": CompletionInfo.FinishType.FAIL_LENGTH,
            "content_filter": CompletionInfo.FinishType.FAIL_FILTER,
            "error": CompletionInfo.FinishType.FAIL_ERROR,
        }.get(response.choices[0].finish_reason, CompletionInfo.FinishType.FAIL_ERROR)

        completion_info = CompletionInfo(
            finish_reason=finish_type,
        )
        response_object = None
        for choice in response.choices:
            message = choice.message
            if prompt.response_schema:
                response_object = SchemaResult(message.content, prompt.response_schema)
            # Handle refusals
            if hasattr(message, "refusal") and isinstance(message.refusal, str):
                completion_info.refusal = message.refusal
                continue

            # Handle regular text responses
            if message.content:
                messages.append(TextOutputMessage(content=message.content))

            # Handle structured/parsed responses
            if hasattr(message, "parsed"):
                messages.append(TextOutputMessage(content=json.dumps(message.parsed)))

            # Handle tool calls
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    schema = tool_schemas.get(tool_call.function.name)
                    messages.append(
                        ToolCallOutputMessage(
                            tool_call_id=tool_call.id,
                            name=tool_call.function.name,
                            arguments=tool_call.function.arguments,
                            schema=schema,
                        )
                    )

        return LLMResponse(
            messages=messages,
            cost=self._calculate_cost(response.usage),
            usage=UsageInfo(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            ),
            completion_info=completion_info,
            response_object=response_object,
        )

    def _supports_structured_output(self) -> bool:
        """Check if the current model supports structured output"""
        structured_models = {
            "gpt-4o-2024-08-06",
            "gpt-4o-mini-2024-07-18",
            "o1-2024-12-17",
        }
        return self.model in structured_models
