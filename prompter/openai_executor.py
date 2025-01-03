import os
from typing import Dict, List
from openai import OpenAI

from .base_executor import BaseLLMExecutor
from .schemas import (
    Prompt,
    LLMResponse,
    TextOutputMessage,
    ToolCallOutputMessage,
    OutputMessage,
    UsageInfo,
    ModelPricing,
    ToolCallArguments,
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

    def execute(self, prompt: Prompt) -> LLMResponse:
        """Execute the prompt using OpenAI's API"""
        # Store prompt for use in response parsing
        self.prompt = prompt

        messages = self._convert_prompt_to_api_messages(prompt)

        # Prepare the API call parameters
        params = {"model": self.model, "messages": messages, **self.kwargs}

        # Add tools if they exist
        if prompt.tools:
            params["tools"] = self._convert_tools_to_api_format(prompt.tools)
            params["tool_choice"] = "auto"

        # Make the API call
        response = self.client.chat.completions.create(**params)

        # Convert and return the response
        return self._convert_api_response_to_messages(response)

    def _calculate_cost(self, usage: UsageInfo) -> float:
        """Calculate the cost based on OpenAI's pricing"""
        prompt_cost = (usage.input_tokens / 1000) * self.pricing.input_price
        completion_cost = (usage.output_tokens / 1000) * self.pricing.output_price
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
                # Handle image messages (assuming OpenAI's format)
                messages.append(
                    {
                        "role": msg.role,
                        "content": [
                            *(
                                [{"type": "text", "text": msg.content}]
                                if msg.content
                                else []
                            ),
                            {"type": "image_url", "image_url": {"url": msg.url}},
                        ],
                    }
                )
            else:
                messages.append({"role": msg.role, "content": msg.content})

        return messages

    def _convert_tools_to_api_format(self, tools: List) -> List[Dict]:
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

    def _convert_api_response_to_messages(self, response) -> LLMResponse:
        """Convert OpenAI's response format to our internal format"""
        messages = []
        tool_schemas = {tool.name: tool.argument_schema for tool in self.prompt.tools}

        for choice in response.choices:
            message = choice.message

            # Handle regular text responses
            if message.content:
                messages.append(TextOutputMessage(type="text", content=message.content))

            # Handle tool calls
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_args = ToolCallArguments(
                        name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                    )
                    # Set the schema if we have one
                    if schema := tool_schemas.get(tool_call.function.name):
                        tool_args.set_schema(schema)

                    messages.append(
                        ToolCallOutputMessage(
                            name=tool_call.function.name,
                            arguments=tool_args,
                        )
                    )

        return LLMResponse(
            messages=messages,
            cost=None,
            usage=UsageInfo(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            ),
        )
