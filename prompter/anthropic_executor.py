import uuid

from typing import Dict, List
import anthropic
import json
from .image_data import url_to_b64
from .base_executor import BaseLLMExecutor
from .schemas import (
    Prompt,
    ImageMessage,
    ToolCallMessage,
    LLMResponse,
    Tool,
    TextOutputMessage,
    ToolCallOutputMessage,
    UsageInfo,
    ModelPricing,
    ToolCallArguments,
)


class AnthropicExecutor(BaseLLMExecutor):
    """Anthropic-specific implementation of the LLM executor"""

    def __init__(
        self, api_key: str = None, model: str = "claude-3-5-sonnet-latest", **kwargs
    ):
        super().__init__(api_key, model, **kwargs)
        if api_key is None:
            self.client = anthropic.Anthropic()
        else:
            self.client = anthropic.Anthropic(api_key=api_key)
        self.pricing = self._get_model_pricing()

    def _get_model_pricing(self) -> ModelPricing:
        """Get the pricing for the current model"""
        prices = {
            "claude-3-opus-20240229": ModelPricing(
                input_price=0.015, output_price=0.075  # per 1K tokens  # per 1K tokens
            ),
            "claude-3-sonnet-20240229": ModelPricing(
                input_price=0.003, output_price=0.015
            ),
            "claude-2.1": ModelPricing(input_price=0.008, output_price=0.024),
        }
        return prices.get(self.model, ModelPricing(input_price=0.0, output_price=0.0))

    def get_api_params(self, prompt: Prompt) -> Dict:
        """Get the API parameters for the prompt"""
        """Execute the prompt using Anthropic's API"""
        messages = self._convert_prompt_to_api_messages(prompt)

        # Prepare the API call parameters
        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.kwargs.get("max_tokens", 4096),
            **self.kwargs,
        }
        if prompt.system_message:
            params["system"] = prompt.system_message

        # Add tools if they exist
        if prompt.tools:
            params["tools"] = self._convert_tools_to_api_format(prompt.tools)

        return params

    def execute(self, prompt: Prompt) -> LLMResponse:
        params = self.get_api_params(prompt)
        # Make the API call
        response = self.client.messages.create(**params)

        # Convert and return the response
        return self._convert_api_response_to_messages(prompt, response)

    def _calculate_cost(self, usage: UsageInfo) -> float:
        """Calculate the cost based on Anthropic's pricing"""
        prompt_cost = (usage.input_tokens / 1000) * self.pricing.input_price
        completion_cost = (usage.output_tokens / 1000) * self.pricing.output_price
        return prompt_cost + completion_cost

    def _convert_prompt_to_api_messages(self, prompt: Prompt) -> list:
        """Convert our prompt format to Anthropic's expected format"""
        messages = []

        # Add conversation messages
        for msg in prompt.messages:
            if msg.message_type() == "image":
                messages.append(self._convert_imagemessage_to_api_format(msg))
            elif msg.message_type() == "text":
                # Map our roles to Anthropic's roles
                role = "assistant" if msg.role == "assistant" else "user"
                messages.append(
                    {"role": role, "content": [{"type": "text", "text": msg.content}]}
                )
            elif msg.message_type() == "tool_call":
                messages.extend(self._convert_tool_call_to_api_format(msg))
            else:
                raise ValueError(f"Unsupported message type: {msg.message_type()}")

        return messages

    def _convert_imagemessage_to_api_format(self, msg: ImageMessage) -> Dict:
        """Convert an image message to Anthropic's expected format"""
        content = []
        if msg.content:
            content.append({"type": "text", "text": msg.content})

        image_data = url_to_b64(msg.url)
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "data": image_data.base64_data,
                    "media_type": image_data.content_type,
                },
            }
        )

        return {
            "role": "user",
            "content": content,
        }

    def _convert_tools_to_api_format(self, tools: List[Tool]) -> List[Dict]:
        """Convert our tool format to Anthropic's expected format"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.get_schema(),
            }
            for tool in tools
        ]

    def _convert_tool_call_to_api_format(self, msg: ToolCallMessage) -> Dict:
        """Convert a tool call message to Anthropic's expected format"""
        ret = []
        tool_call_id = msg.tool_call_id or "tooluse_" + str(uuid.uuid4())
        ret.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": msg.tool_name,
                        "id": tool_call_id,
                        "input": msg.arguments,
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
                        "content": (json.dumps(msg.result.result)),
                    }
                ],
            },
        )
        return ret

    def _convert_api_response_to_messages(
        self, prompt: Prompt, response
    ) -> LLMResponse:
        """Convert Anthropic's response format to our internal format"""
        messages = []
        tool_schemas = {tool.name: tool.argument_schema for tool in prompt.tools}

        # Handle each content block
        for content in response.content:

            if content.type == "text":
                messages.append(TextOutputMessage(type="text", content=content.text))
            elif content.type == "tool_use":

                tool_args = ToolCallArguments(arguments=content.input)
                # Set the schema if we have one
                if schema := tool_schemas.get(content.name):
                    tool_args.set_schema(schema)

                messages.append(
                    ToolCallOutputMessage(
                        name=content.name,
                        arguments=tool_args,
                    )
                )

        # Create usage info
        usage = UsageInfo(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

        return LLMResponse(
            messages=messages,
            cost=self._calculate_cost(usage),
            usage=usage,
        )
