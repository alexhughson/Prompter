from prompter.schemas import Prompt, UserMessage, Tool
from pydantic import BaseModel


class WeatherArgs(BaseModel):
    location: str
    units: str = "celsius"


def test_weather_tool_call(llm_executor):
    """Test that the LLM correctly formats tool calls"""
    prompt = Prompt(
        system_message="You are a helpful assistant. Always use the weather tool when asked about weather.",
        messages=[
            UserMessage(content="What's the weather like in Tokyo?"),
        ],
        tools=[
            Tool(
                name="get_weather",
                description="Get the weather in a given location",
                argument_schema=WeatherArgs,
            )
        ],
    )

    response = llm_executor.execute(prompt)
    response.raise_for_status()

    tool_call = response.tool_call()
    assert tool_call.name == "get_weather"
    assert isinstance(tool_call.arguments, WeatherArgs)
    assert tool_call.arguments.location.lower() == "tokyo"


def test_multiple_tool_calls(llm_executor):
    """Test handling multiple tool calls in one response"""
    prompt = Prompt(
        system_message="You are a helpful assistant. Compare the weather in two cities.",
        messages=[
            UserMessage(content="Compare the weather in New York and London"),
        ],
        tools=[
            Tool(
                name="get_weather",
                description="Get the weather in a given location",
                argument_schema=WeatherArgs,
            )
        ],
    )

    response = llm_executor.execute(prompt)
    response.raise_for_status()

    tool_calls = response.tool_calls()
    assert len(tool_calls) == 2

    locations = {call.arguments.location.lower() for call in tool_calls}
    assert locations == {"new york", "london"}
