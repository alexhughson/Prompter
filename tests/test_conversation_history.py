from pydantic import BaseModel

from prompter.schemas import Assistant, Prompt, Tool, ToolUse, User


class WeatherArgs(BaseModel):
    location: str
    units: str = "celsius"


def test_conversation_with_completed_tool_calls(llm_executor):
    """Test that the LLM can understand context from previous tool call results"""
    weather_tool = Tool(
        name="get_weather",
        description="Get the weather in a given location",
        params=WeatherArgs,
    )

    prompt = Prompt(
        system="You are a helpful weather assistant.",
        conversation=[
            User("What's the weather like in Tokyo?"),
            Assistant("Let me check that for you."),
            ToolUse(
                name="get_weather",
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
        params=WeatherArgs,
    )

    prompt = Prompt(
        system="You are a helpful weather assistant.",
        conversation=[
            User("What's the weather like in Tokyo and InvalidCity?"),
            Assistant("I'll check both locations."),
            ToolUse(
                name="get_weather",
                arguments={"location": "Tokyo", "units": "celsius"},
                result={"temperature": 22, "conditions": "sunny"},
            ),
            ToolUse(
                name="get_weather",
                arguments={"location": "InvalidCity", "units": "celsius"},
                error="Location 'InvalidCity' not found",
            ),
            Assistant(
                content="I found the weather for Tokyo, but InvalidCity isn't a valid location."
            ),
            User("Ok, check London instead of InvalidCity"),
        ],
        tools=[weather_tool],
    )

    response = llm_executor.execute(prompt)
    response.raise_for_status()

    # The LLM should make a tool call for London
    cities = {call.arguments["location"].lower() for call in response.tool_calls()}
    assert "london" in cities
