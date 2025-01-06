from prompter.schemas import (
    ImageMessage,
    Tool,
    ToolCallMessage,
    ToolCallResult,
)
from prompter.openai_executor import OpenAIExecutor
from pydantic import BaseModel

# Use a small, reliable test image
TEST_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"


def test_convert_image_message():
    """Test conversion of a single image message"""
    executor = OpenAIExecutor()

    message = ImageMessage(
        url=TEST_IMAGE_URL,
        content="This is a caption",
    )

    converted = executor._convert_imagemessage_to_api_format(message)
    assert converted == {
        "role": "user",
        "content": [
            {"type": "text", "text": "This is a caption"},
            {"type": "image_url", "image_url": {"url": TEST_IMAGE_URL}},
        ],
    }


def test_convert_image_message_without_caption():
    """Test conversion of an image message without caption"""
    executor = OpenAIExecutor()

    message = ImageMessage(url=TEST_IMAGE_URL)

    converted = executor._convert_imagemessage_to_api_format(message)
    assert converted == {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": TEST_IMAGE_URL}},
        ],
    }


class WeatherArgs(BaseModel):
    location: str
    units: str = "celsius"


def test_convert_tool_call_with_result():
    """Test conversion of a completed tool call"""
    executor = OpenAIExecutor()

    message = ToolCallMessage(
        tool_name="get_weather",
        tool_call_id="call_123",
        arguments={"location": "London", "units": "celsius"},
        result=ToolCallResult(result={"temperature": 20, "conditions": "cloudy"}),
    )

    converted = executor._convert_tool_call_to_api_format(message)
    assert converted == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "London", "units": "celsius"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"temperature": 20, "conditions": "cloudy"}',
            "tool_call_id": "call_123",
        },
    ]


def test_convert_tool_definition():
    """Test conversion of a tool definition"""
    executor = OpenAIExecutor()

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
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for items",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "title": "Query"},
                    "filters": {"type": "object", "title": "Filters"},
                    "limit": {"type": "integer", "title": "Limit", "default": 10},
                },
                "required": ["query", "filters"],
                "title": "ComplexArgs",
            },
        },
    }
