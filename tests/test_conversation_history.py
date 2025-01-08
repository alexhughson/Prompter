from prompter.schemas import (
    Prompt,
    UserMessage,
    AssistantMessage,
    Tool,
    ToolCallMessage,
)
from pydantic import BaseModel


class WeatherArgs(BaseModel):
    location: str
    units: str = "celsius"


def test_conversation_with_completed_tool_calls(llm_executor):
    """Test that the LLM can understand context from previous tool call results"""
    weather_tool = Tool(
        name="get_weather",
        description="Get the weather in a given location",
        argument_schema=WeatherArgs,
    )

    prompt = Prompt(
        system_message="You are a helpful weather assistant.",
        messages=[
            UserMessage(content="What's the weather like in Tokyo?"),
            AssistantMessage(content="Let me check that for you."),
            ToolCallMessage(
                tool_name="get_weather",
                arguments={"location": "Tokyo", "units": "celsius"},
                result={"temperature": 22, "conditions": "sunny"},
            ),
        ],
        tools=[weather_tool],
    )

    response = llm_executor.execute(prompt)
    response.raise_for_status()

    message = response.text()
    assert "22" in message


def test_conversation_with_error_tool_calls(llm_executor):
    """Test that the LLM can handle tool calls that resulted in errors"""
    weather_tool = Tool(
        name="get_weather",
        description="Get the weather in a given location",
        argument_schema=WeatherArgs,
    )

    prompt = Prompt(
        system_message="You are a helpful weather assistant.",
        messages=[
            UserMessage(content="What's the weather like in Tokyo and InvalidCity?"),
            AssistantMessage(content="I'll check both locations."),
            ToolCallMessage(
                tool_name="get_weather",
                arguments={"location": "Tokyo", "units": "celsius"},
                result={"temperature": 22, "conditions": "sunny"},
            ),
            ToolCallMessage(
                tool_name="get_weather",
                arguments={"location": "InvalidCity", "units": "celsius"},
                result="Location 'InvalidCity' not found",
            ),
            AssistantMessage(
                content="I found the weather for Tokyo, but InvalidCity isn't a valid location."
            ),
            UserMessage(content="Ok, check London instead of InvalidCity"),
        ],
        tools=[weather_tool],
    )

    response = llm_executor.execute(prompt)
    response.raise_for_status()

    # The LLM should make a tool call for London
    cities = {call.arguments.parse().location.lower() for call in response.tool_calls()}
    assert "london" in cities
