import json
from typing import Dict, List, Optional

import pytest
from prompter.schemas import (
    AssistantMessage,
    ImageMessage,
    LLMResponse,
    Prompt,
    TextOutputMessage,
    Tool,
    ToolCallMessage,
    ToolCallOutputMessage,
    ToolCallResult,
    UsageInfo,
    UserMessage,
)
from pydantic import BaseModel, ValidationError


# Test data models
class WeatherArgs(BaseModel):
    location: str
    units: Optional[str] = "celsius"


def test_message_types():
    """Test that all message types return correct message_type and have expected attributes"""
    user_msg = UserMessage(content="Hello")
    assert user_msg.message_type() == "text"
    assert user_msg.role == "user"
    assert user_msg.content == "Hello"

    assistant_msg = AssistantMessage(content="Hi there")
    assert assistant_msg.message_type() == "text"
    assert assistant_msg.role == "assistant"
    assert assistant_msg.content == "Hi there"

    image_msg = ImageMessage(url="http://example.com/image.jpg")
    assert image_msg.message_type() == "image"
    assert image_msg.role == "user"
    assert image_msg.url == "http://example.com/image.jpg"
    assert image_msg.media_type == "image/jpeg"


def test_tool_call_message():
    """Test ToolCallMessage creation and attributes"""
    result = ToolCallResult(result={"temperature": 25})
    tool_msg = ToolCallMessage(
        tool_name="get_weather", arguments={"location": "London"}, result=result
    )

    assert tool_msg.message_type() == "tool_call"
    assert tool_msg.role == "assistant"
    assert tool_msg.tool_name == "get_weather"
    assert tool_msg.arguments == {"location": "London"}
    assert tool_msg.result.result == {"temperature": 25}
    assert tool_msg.result.error is None


def test_tool_definition():
    """Test Tool class with both dict and pydantic schema"""
    # Test with Pydantic model
    tool_with_model = Tool(
        name="get_weather",
        description="Get weather for a location",
        argument_schema=WeatherArgs,
    )
    schema = tool_with_model.get_schema()
    assert schema["properties"]["location"]["type"] == "string"
    assert schema["properties"]["units"]["default"] == "celsius"

    # Test with dict schema
    dict_schema = {"type": "object", "properties": {"location": {"type": "string"}}}
    tool_with_dict = Tool(
        name="get_weather",
        description="Get weather for a location",
        argument_schema=dict_schema,
    )
    assert tool_with_dict.get_schema() == dict_schema


def test_prompt_creation():
    """Test Prompt class creation and attributes"""
    messages = [
        UserMessage(content="What's the weather?"),
        AssistantMessage(content="Let me check that for you"),
    ]
    tools = [
        Tool(
            name="get_weather",
            description="Get weather for a location",
            argument_schema=WeatherArgs,
        )
    ]

    prompt = Prompt(
        system_message="You are a weather assistant", messages=messages, tools=tools
    )

    assert prompt.system_message == "You are a weather assistant"
    assert len(prompt.messages) == 2
    assert len(prompt.tools) == 1
    assert isinstance(prompt.messages[0], UserMessage)


def test_output_messages():
    """Test OutputMessage implementations"""
    text_msg = TextOutputMessage(content="Hello")
    assert text_msg.text() == "Hello"
    assert text_msg.is_text() is True
    assert text_msg.is_tool_call() is False

    tool_call = ToolCallOutputMessage(
        name="get_weather", arguments={"location": "London"}, schema=WeatherArgs
    )
    assert tool_call.is_tool_call() is True
    assert tool_call.is_text() is False
    assert isinstance(tool_call.arguments, WeatherArgs)

    # Test with schema validation
    parsed_args = tool_call.parse()
    assert isinstance(parsed_args, WeatherArgs)
    assert parsed_args.location == "London"
    assert parsed_args.units == "celsius"  # default value


def test_llm_response():
    """Test LLMResponse functionality"""
    messages = [
        TextOutputMessage(content="The weather is:"),
        ToolCallOutputMessage(
            name="get_weather", arguments={"location": "London"}, schema=WeatherArgs
        ),
    ]

    usage = UsageInfo(input_tokens=10, output_tokens=20)
    response = LLMResponse(messages=messages, usage=usage, cost=0.001)

    assert len(response.text_messages()) == 1
    assert len(response.tool_calls()) == 1

    # Test text concatenation
    assert "The weather is:" in response.text()


def test_llm_response_tool_call_validation():
    """Test LLMResponse tool call validation"""
    response = LLMResponse(
        messages=[
            ToolCallOutputMessage(
                name="get_weather", arguments={"location": "London"}, schema=WeatherArgs
            )
        ],
        usage=UsageInfo(),
    )

    # Test single tool call extraction
    tool_call = response.tool_call()
    assert tool_call.name == "get_weather"

    # Test error when no tool calls
    response_no_tools = LLMResponse(
        messages=[TextOutputMessage(content="Hello")], usage=UsageInfo()
    )
    with pytest.raises(ValueError, match="No tool calls found"):
        response_no_tools.tool_call()

    # Test error when multiple tool calls
    response_multiple_tools = LLMResponse(
        messages=[
            ToolCallOutputMessage(name="tool1", arguments={}),
            ToolCallOutputMessage(name="tool2", arguments={}),
        ],
        usage=UsageInfo(),
    )
    with pytest.raises(ValueError, match="Multiple tool calls found"):
        response_multiple_tools.tool_call()


def test_tool_call_json_parsing_errors():
    """Test ToolCallOutputMessage handling of invalid JSON arguments"""
    # Invalid JSON string
    tool_call = ToolCallOutputMessage(
        name="get_weather", arguments="{invalid_json", schema=WeatherArgs
    )

    with pytest.raises(json.JSONDecodeError):
        tool_call.parse()

    # Valid JSON but wrong type (string instead of dict)
    tool_call = ToolCallOutputMessage(
        name="get_weather", arguments='"not_a_dict"', schema=WeatherArgs
    )

    with pytest.raises(ValidationError):
        tool_call.parse()


def test_tool_call_schema_validation_errors():
    """Test ToolCallOutputMessage handling of schema validation errors"""
    # Missing required field
    tool_call = ToolCallOutputMessage(
        name="get_weather",
        arguments={"units": "fahrenheit"},  # missing required 'location'
        schema=WeatherArgs,
    )

    with pytest.raises(ValidationError) as exc_info:
        tool_call.parse()
    assert "location" in str(exc_info.value)  # Error should mention missing field

    # Wrong type for field
    tool_call = ToolCallOutputMessage(
        name="get_weather",
        arguments={"location": 123, "units": "celsius"},  # location should be string
        schema=WeatherArgs,
    )

    with pytest.raises(ValidationError) as exc_info:
        tool_call.parse()
    assert "should be a valid string" in str(exc_info.value)

    # Invalid enum value for units
    tool_call = ToolCallOutputMessage(
        name="get_weather",
        arguments={"location": "London", "units": "invalid_unit"},
        schema=WeatherArgs,
    )

    # This should still parse since units is not restricted to specific values
    parsed = tool_call.parse()
    assert parsed.units == "invalid_unit"
