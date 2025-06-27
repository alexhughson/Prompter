from pydantic import BaseModel

from prompter.schemas import Prompt, Tool, ToolUse, User
from prompter.tool_belt import ToolBelt

# from .stub_executor import StubExecutor


class WeatherArgs(BaseModel):
    location: str
    units: str = "celsius"


WEATHER_TOOL = Tool(
    name="get_weather",
    description="Get the weather in a given location",
    params=WeatherArgs,
)


def test_weather_tool_call(llm_executor):
    """Test that the LLM correctly formats tool calls"""
    tool_belt = ToolBelt([WEATHER_TOOL])

    prompt = Prompt(
        system="You are a helpful assistant. Always use the weather tool when asked about weather.",
        conversation=[
            User("What's the weather like in Tokyo?"),
        ],
        tools=tool_belt.tool_list(),
    )

    response = llm_executor.execute(prompt)
    # response.raise_for_status()

    tool_call = response.tool_call()
    assert tool_call.name == "get_weather"

    # Parse with ToolBelt
    parsed = tool_belt.parse_call(tool_call)
    assert isinstance(parsed, WeatherArgs)
    assert parsed.location.lower() == "tokyo"


def test_multiple_tool_calls(llm_executor):
    """Test handling multiple tool calls in one response"""
    tool_belt = ToolBelt([WEATHER_TOOL])

    prompt = Prompt(
        system="You are a helpful assistant. Compare the weather in two cities.  Make as many tool usage calls as possible at one time",
        conversation=[
            User("Compare the weather in New York and London"),
        ],
        tools=tool_belt.tool_list(),
    )

    response = llm_executor.execute(prompt)
    # response.raise_for_status()

    tool_calls = response.tool_calls()
    assert len(tool_calls) == 2

    # Parse all calls with ToolBelt
    parsed_calls = [tool_belt.parse_call(call) for call in tool_calls]
    locations = {call.location.lower() for call in parsed_calls}
    assert locations == {"new york", "london"}


def test_tool_call_result_handling(llm_executor):
    """Test handling of tool call results"""
    prompt = Prompt(
        system="You are a weather assistant",
        conversation=[User("What's the weather in Toronto?")],
        tools=[WEATHER_TOOL],
    )

    response = llm_executor.execute(prompt)

    # Get the tool call
    tool_call = response.tool_call()

    # Verify tool arguments
    args = WEATHER_TOOL.validate_arguments(tool_call.arguments)
    assert args["location"].lower() == "toronto"

    # Add a mock result
    weather_data = {"temperature": 20, "conditions": "sunny"}
    prompt.conversation.append(
        ToolUse(name=tool_call.name, arguments=tool_call.arguments, result=weather_data)
    )

    # Execute again
    response = llm_executor.execute(prompt)
    # response.raise_for_status()

    # Verify final response includes tool results
    final_text = response.text()
    assert "20" in final_text
    assert "sunny" in final_text


def test_tool_call_explicit_result_message(llm_executor):
    """Test that tool calls can be added as messages"""
    prompt = Prompt(
        system="You are a weather assistant",
        conversation=[
            User("What's the weather in Toronto?"),
            ToolUse(
                name="get_weather",
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


# This test is unreliable in most LLMs, but often passes
# def test_tool_call_with_message(llm_executor):
#     """Test that tool calls can be added as messages"""
#     tool_belt = ToolBelt([WEATHER_TOOL])

#     prompt = Prompt(
#         system="You are a weather assistant.  Always announce what you are doing before you use a tool.",
#         conversation=[User("What's the weather in Toronto?")],
#         tools=tool_belt.tool_list(),
#     )

#     response = llm_executor.execute(prompt)
#     # response.raise_for_status()

#     tool_call = response.tool_call()
#     assert tool_call.name == "get_weather"

#     # Parse with ToolBelt
#     parsed = tool_belt.parse_call(tool_call)
#     assert parsed.location.lower() == "toronto"

#     text_messages = response.text_messages()
#     assert len(text_messages) > 0


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


def test_tool_belt_workflow_with_response(llm_executor):
    """Test complete workflow: ToolBelt -> Prompt -> Response -> Parse"""

    # Create tool belt with multiple tools
    class SearchArgs(BaseModel):
        query: str
        max_results: int = 10

    search_tool = Tool(
        name="web_search", description="Search the web", params=SearchArgs
    )

    tool_belt = ToolBelt([WEATHER_TOOL, search_tool])

    prompt = Prompt(
        system="You are a helpful assistant.",
        conversation=[
            User(
                "What's the weather in Berlin and search for tourist attractions there"
            )
        ],
        tools=tool_belt.tool_list(),
    )

    response = llm_executor.execute(prompt)

    # Parse all tool calls from response
    for tool_call in response.tool_calls():
        parsed = tool_belt.parse_call(tool_call)

        if tool_call.name == "get_weather":
            assert isinstance(parsed, WeatherArgs)
            assert "berlin" in parsed.location.lower()
        elif tool_call.name == "web_search":
            assert isinstance(parsed, SearchArgs)
            assert (
                "tourist" in parsed.query.lower()
                or "attractions" in parsed.query.lower()
            )
