from prompter.schemas import (
    ImageMessage,
    Tool,
    ToolCallMessage,
    Prompt,
    UserMessage,
)
from prompter.anthropic_executor import AnthropicExecutor
from pydantic import BaseModel
import json

# Use a small, reliable test image
TEST_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"


def test_convert_image_message():
    """Test conversion of a single image message"""
    executor = AnthropicExecutor()

    message = ImageMessage(
        url=TEST_IMAGE_URL,
        content="This is a caption",
    )

    converted = executor._convert_imagemessage_to_api_format(message)
    assert converted["role"] == "user"
    assert len(converted["content"]) == 2
    assert converted["content"][0] == {
        "type": "text",
        "text": "This is a caption",
    }

    # Verify image structure but not exact base64 content
    image_content = converted["content"][1]

    assert image_content["type"] == "image"
    assert image_content["source"]["type"] == "base64"
    assert image_content["source"]["media_type"].startswith("image/")
    assert isinstance(image_content["source"]["data"], str)


def test_convert_image_message_without_caption():
    """Test conversion of an image message without caption"""
    executor = AnthropicExecutor()

    message = ImageMessage(url=TEST_IMAGE_URL)

    converted = executor._convert_imagemessage_to_api_format(message)
    assert converted["role"] == "user"
    assert len(converted["content"]) == 1

    # Verify image structure but not exact base64 content
    image_content = converted["content"][0]
    assert image_content["type"] == "image"
    assert image_content["source"]["type"] == "base64"
    assert image_content["source"]["media_type"].startswith("image/")
    assert isinstance(image_content["source"]["data"], str)


class WeatherArgs(BaseModel):
    location: str
    units: str = "celsius"


def test_convert_tool_call_with_result():
    """Test conversion of a completed tool call"""
    executor = AnthropicExecutor()

    message = ToolCallMessage(
        tool_name="get_weather",
        tool_call_id="call_123",
        arguments={"location": "London", "units": "celsius"},
        result={"temperature": 20, "conditions": "cloudy"},
    )

    converted = executor._convert_tool_call_to_api_format(message)
    assert len(converted) == 2

    # Check tool use message
    tool_use = converted[0]
    assert tool_use["role"] == "assistant"
    assert tool_use["content"][0]["type"] == "tool_use"
    assert tool_use["content"][0]["name"] == "get_weather"
    assert tool_use["content"][0]["id"] == "call_123"
    assert tool_use["content"][0]["input"] == {
        "location": "London",
        "units": "celsius",
    }

    # Check tool result message
    tool_result = converted[1]
    assert tool_result["role"] == "user"
    assert tool_result["content"][0]["type"] == "tool_result"
    assert tool_result["content"][0]["tool_use_id"] == "call_123"
    assert (
        tool_result["content"][0]["content"]
        == '{"temperature": 20, "conditions": "cloudy"}'
    )


def test_convert_tool_definition():
    """Test conversion of a tool definition"""
    executor = AnthropicExecutor()

    class ComplexArgs(BaseModel):
        query: str
        filters: dict
        limit: int = 10

    tool = Tool(
        name="search",
        description="Search for items",
        argument_schema=ComplexArgs,
    )

    converted = executor._convert_tools_to_api_format([tool])[0]

    assert converted == {
        "name": "search",
        "description": "Search for items",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "title": "Query"},
                "filters": {"type": "object", "title": "Filters"},
                "limit": {"type": "integer", "title": "Limit", "default": 10},
            },
            "required": ["query", "filters"],
            "title": "ComplexArgs",
        },
    }


def test_convert_tool_call_with_error():
    """Test conversion of a tool call that resulted in an error"""
    executor = AnthropicExecutor()

    message = ToolCallMessage(
        tool_name="get_weather",
        tool_call_id="error_call",
        arguments={"location": "InvalidCity"},
        result=ToolCallResult(
            result=None,
            error="Location 'InvalidCity' not found",
        ),
    )

    converted = executor._convert_tool_call_to_api_format(message)
    assert len(converted) == 2

    # Check tool use message
    tool_use = converted[0]
    assert tool_use["role"] == "assistant"
    assert tool_use["content"][0]["type"] == "tool_use"
    assert tool_use["content"][0]["name"] == "get_weather"
    assert tool_use["content"][0]["id"] == "error_call"
    assert tool_use["content"][0]["input"] == {"location": "InvalidCity"}

    # Check error message
    error_result = converted[1]
    assert error_result["role"] == "user"
    assert error_result["content"][0]["type"] == "tool_result"
    assert error_result["content"][0]["tool_use_id"] == "error_call"
    assert (
        error_result["content"][0]["content"] == "null"
    )  # Error results have null content


def test_stop_reason_mapping():
    anthropic_executor = AnthropicExecutor()
    """Test that Anthropic's stop reasons are correctly mapped"""
    prompt = Prompt(
        system_message="You are a helpful assistant.",
        messages=[UserMessage("Hello")],
    )

    response = anthropic_executor.execute(prompt)

    assert response.status == "success"
    response.raise_for_status()  # Should not raise
