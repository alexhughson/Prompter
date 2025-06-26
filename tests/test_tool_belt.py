import pytest
from pydantic import BaseModel, ValidationError

from prompter.schemas import Tool, ToolCall
from prompter.tool_belt import ToolBelt


class WeatherParams(BaseModel):
    location: str
    units: str = "celsius"


class CalculatorParams(BaseModel):
    operation: str
    a: float
    b: float


def test_tool_belt_creation():
    """Test creating a ToolBelt with tools"""
    weather_tool = Tool(
        name="get_weather",
        description="Get weather for a location",
        params=WeatherParams,
    )

    calc_tool = Tool(
        name="calculator", description="Perform calculations", params=CalculatorParams
    )

    tool_belt = ToolBelt([weather_tool, calc_tool])

    tools = tool_belt.tool_list()
    assert len(tools) == 2
    assert tools[0].name == "get_weather"
    assert tools[1].name == "calculator"


def test_tool_belt_add_tool():
    """Test adding tools to ToolBelt"""
    tool_belt = ToolBelt()

    weather_tool = Tool(
        name="get_weather",
        description="Get weather for a location",
        params=WeatherParams,
    )

    tool_belt.add(weather_tool)

    tools = tool_belt.tool_list()
    assert len(tools) == 1
    assert tools[0].name == "get_weather"


def test_parse_tool_call_valid():
    """Test parsing a valid tool call into pydantic object"""
    weather_tool = Tool(
        name="get_weather",
        description="Get weather for a location",
        params=WeatherParams,
    )

    tool_belt = ToolBelt([weather_tool])

    tool_call = ToolCall(
        name="get_weather", arguments={"location": "Tokyo", "units": "fahrenheit"}
    )

    parsed = tool_belt.parse_call(tool_call)

    assert isinstance(parsed, WeatherParams)
    assert parsed.location == "Tokyo"
    assert parsed.units == "fahrenheit"


def test_parse_tool_call_with_defaults():
    """Test parsing tool call uses default values"""
    weather_tool = Tool(
        name="get_weather",
        description="Get weather for a location",
        params=WeatherParams,
    )

    tool_belt = ToolBelt([weather_tool])

    tool_call = ToolCall(name="get_weather", arguments={"location": "Paris"})

    parsed = tool_belt.parse_call(tool_call)

    assert isinstance(parsed, WeatherParams)
    assert parsed.location == "Paris"
    assert parsed.units == "celsius"  # default value


def test_parse_tool_call_invalid_args():
    """Test parsing tool call with invalid arguments raises ValidationError"""
    weather_tool = Tool(
        name="get_weather",
        description="Get weather for a location",
        params=WeatherParams,
    )

    tool_belt = ToolBelt([weather_tool])

    tool_call = ToolCall(
        name="get_weather",
        arguments={"invalid_field": "value"},  # missing required 'location'
    )

    with pytest.raises(ValidationError):
        tool_belt.parse_call(tool_call)


def test_parse_tool_call_unknown_tool():
    """Test parsing tool call for unknown tool raises error"""
    tool_belt = ToolBelt()

    tool_call = ToolCall(name="unknown_tool", arguments={"some": "args"})

    with pytest.raises(ValueError, match="Unknown tool"):
        tool_belt.parse_call(tool_call)


def test_parse_multiple_tool_calls():
    """Test parsing multiple tool calls from a response"""
    weather_tool = Tool(
        name="get_weather",
        description="Get weather for a location",
        params=WeatherParams,
    )

    calc_tool = Tool(
        name="calculator", description="Perform calculations", params=CalculatorParams
    )

    tool_belt = ToolBelt([weather_tool, calc_tool])

    tool_calls = [
        ToolCall(name="get_weather", arguments={"location": "London"}),
        ToolCall(name="calculator", arguments={"operation": "add", "a": 5, "b": 3}),
    ]

    parsed_calls = []
    for call in tool_calls:
        parsed = tool_belt.parse_call(call)
        parsed_calls.append(parsed)

    assert len(parsed_calls) == 2
    assert isinstance(parsed_calls[0], WeatherParams)
    assert parsed_calls[0].location == "London"
    assert isinstance(parsed_calls[1], CalculatorParams)
    assert parsed_calls[1].operation == "add"
    assert parsed_calls[1].a == 5
    assert parsed_calls[1].b == 3


def test_tool_with_dict_params():
    """Test tool with dict params instead of pydantic model"""
    tool = Tool(
        name="simple_tool",
        description="A simple tool",
        params={"type": "object", "properties": {"name": {"type": "string"}}},
    )

    tool_belt = ToolBelt([tool])

    tool_call = ToolCall(name="simple_tool", arguments={"name": "test"})

    # Should return raw arguments when no pydantic model
    parsed = tool_belt.parse_call(tool_call)
    assert parsed == {"name": "test"}


def test_tool_with_no_params():
    """Test tool with no params"""
    tool = Tool(name="no_params_tool", description="A tool with no parameters")

    tool_belt = ToolBelt([tool])

    tool_call = ToolCall(name="no_params_tool", arguments={})

    parsed = tool_belt.parse_call(tool_call)
    assert parsed == {}
