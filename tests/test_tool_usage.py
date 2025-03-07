import json
from typing import Optional

import pytest
from pydantic import BaseModel

from prompter.schemas import (
    Prompt,
    SchemaValidationError,
    TextMessage,
    Tool,
    ToolCallMessage,
    ToolCallOutputMessage,
)

# from .stub_executor import StubExecutor


class WeatherArgs(BaseModel):
    location: str
    units: str = "celsius"


WEATHER_TOOL = Tool(
    name="get_weather",
    description="Get the weather in a given location",
    argument_schema=WeatherArgs,
)


def test_weather_tool_call(llm_executor):
    """Test that the LLM correctly formats tool calls"""
    prompt = Prompt(
        system_message="You are a helpful assistant. Always use the weather tool when asked about weather.",
        messages=[
            TextMessage.user("What's the weather like in Tokyo?"),
        ],
        tools=[WEATHER_TOOL],
    )

    response = llm_executor.execute(prompt)
    # response.raise_for_status()

    tool_call = response.tool_call()
    assert tool_call.name == "get_weather"
    assert isinstance(tool_call.arguments.parse(), WeatherArgs)
    assert tool_call.arguments.parse().location.lower() == "tokyo"


def test_multiple_tool_calls(llm_executor):
    """Test handling multiple tool calls in one response"""
    prompt = Prompt(
        system_message="You are a helpful assistant. Compare the weather in two cities.  Make as many tool usage calls as possible at one time",
        messages=[
            TextMessage.user(content="Compare the weather in New York and London"),
        ],
        tools=[WEATHER_TOOL],
    )

    response = llm_executor.execute(prompt)
    # response.raise_for_status()

    tool_calls = response.tool_calls()
    assert len(tool_calls) == 2

    locations = {call.arguments.parse().location.lower() for call in tool_calls}
    assert locations == {"new york", "london"}


def test_tool_call_result_handling(llm_executor):
    """Test handling of tool call results"""
    prompt = Prompt(
        system_message="You are a weather assistant",
        messages=[TextMessage.user("What's the weather in Toronto?")],
        tools=[WEATHER_TOOL],
    )

    response = llm_executor.execute(prompt)

    # Get the tool call
    tool_call = response.tool_call()

    # Verify tool arguments
    assert tool_call.arguments.valid()
    args = tool_call.arguments.parse()
    assert isinstance(args, WeatherArgs)

    # Add a mock result
    weather_data = {"temperature": 20, "conditions": "sunny"}
    prompt.messages.append(tool_call.to_input_message(weather_data))

    # Execute again
    response = llm_executor.execute(prompt)
    # response.raise_for_status()

    # Verify final response includes tool results

    # Verify final response includes tool results
    final_text = response.text(include_tools=True)
    assert "20" in final_text
    assert "sunny" in final_text


def test_tool_call_explicit_result_message(llm_executor):
    """Test that tool calls can be added as messages"""
    prompt = Prompt(
        system_message="You are a weather assistant",
        messages=[
            TextMessage.user("What's the weather in Toronto?"),
            ToolCallMessage(
                tool_name="get_weather",
                arguments={"location": "Toronto"},
                result={"temperature": 20, "conditions": "sunny"},
            ),
        ],
        tools=[WEATHER_TOOL],
    )

    response = llm_executor.execute(prompt)
    # response.raise_for_status()
    assert "20" in response.text()
    assert "sunny" in response.text()
    assert "Toronto" in response.text()


def test_tool_call_with_message(llm_executor):
    """Test that tool calls can be added as messages"""
    prompt = Prompt(
        system_message="You are a weather assistant.  Always announce what you are doing before you use a tool.",
        messages=[TextMessage.user("What's the weather in Toronto?")],
        tools=[WEATHER_TOOL],
    )

    response = llm_executor.execute(prompt)
    # response.raise_for_status()

    tool_call = response.tool_call()
    assert tool_call.name == "get_weather"
    assert isinstance(tool_call.arguments.parse(), WeatherArgs)
    assert tool_call.arguments.parse().location.lower() == "toronto"

    text_messages = response.text_messages()
    assert len(text_messages) > 0


# def test_invalid_tool_arguments():
#     """Test handling of invalid tool call arguments"""
#     executor = StubExecutor(invalid_json=True)

#     prompt = Prompt(
#         system_message="You are a weather assistant",
#         messages=[UserMessage("What's the weather in Toronto?")],
#         tools=[WEATHER_TOOL],
#     )

#     response = executor.execute(prompt)
#     tool_call = response.tool_call()

#     assert not tool_call.arguments.valid()
#     with pytest.raises(json.JSONDecodeError):
#         tool_call.arguments.parse()


# def test_schema_mismatch_tool_arguments():
#     """Test handling of tool arguments that don't match schema"""
#     executor = StubExecutor(invalid_schema=True)

#     prompt = Prompt(
#         system_message="You are a weather assistant",
#         messages=[UserMessage("What's the weather in Toronto?")],
#         tools=[WEATHER_TOOL],
#     )

#     response = executor.execute(prompt)
#     tool_call = response.tool_call()

#     assert tool_call.arguments.valid_json()
#     assert not tool_call.arguments.valid()

#     # Should be able to get raw dict
#     args_dict = tool_call.arguments.parse_obj()
#     assert "wrong_field" in args_dict

#     # But parsing to schema should fail
#     with pytest.raises(SchemaValidationError):
#         tool_call.arguments.parse()
