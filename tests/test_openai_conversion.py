from pydantic import BaseModel

from prompter.schemas import (
    Assistant,
    Image,
    Prompt,
    System,
    Text,
    Tool,
    ToolCall,
    ToolUse,
    User,
)
from prompter.openai_executor import (
    block_to_openai_messages,
    tool_to_openai,
)


class WeatherParams(BaseModel):
    location: str
    units: str = "celsius"


def test_simple_text_prompt():

    prompt = Prompt(
        system="You are helpful",
        conversation=[User("Hello")],
    )

    expected_system = {"role": "system", "content": "You are helpful"}
    expected_user = {"role": "user", "content": "Hello"}

    system_msg = block_to_openai_messages(System(prompt.system))
    assert system_msg == [expected_system]

    user_msg = block_to_openai_messages(prompt.conversation[0])
    assert user_msg == [expected_user]


def test_multi_part_user_content():
    user = User(Text("Hello"), Text(" "), Text("world"))

    expected = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": " "},
            {"type": "text", "text": "world"},
        ],
    }

    actual = block_to_openai_messages(user)
    assert actual == [expected]


def test_image_prompt():
    user = User("What's this?", Image.url("https://example.com/cat.jpg"))

    expected = {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's this?"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/cat.jpg", "detail": "auto"},
            },
        ],
    }

    actual = block_to_openai_messages(user)
    assert actual == [expected]


def test_image_with_detail():
    img = Image(source="https://example.com/cat.jpg", detail="high")
    user = User("Analyze this", img)
    expected = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Analyze this"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/cat.jpg", "detail": "high"},
            },
        ],
    }

    actual = block_to_openai_messages(user)
    assert actual == [expected]


def test_tool_conversion():
    tool = Tool(
        name="get_weather",
        description="Get weather for a location",
        params=WeatherParams,
    )

    expected = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "title": "Location"},
                    "units": {"type": "string", "title": "Units", "default": "celsius"},
                },
                "required": ["location"],
                "title": "WeatherParams",
            },
        },
    }

    actual = tool_to_openai(tool)
    assert actual == expected


def test_tool_with_dict_params():

    params_dict = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 10},
        },
        "required": ["query"],
    }
    tool = Tool(name="search", description="Search for items", params=params_dict)

    expected = {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for items",
            "parameters": params_dict,
        },
    }

    actual = tool_to_openai(tool)
    assert actual == expected


def test_tool_call_conversion():
    tool_call = ToolCall(
        name="get_weather", arguments={"location": "Paris"}, id="call_123"
    )

    expected = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "Paris"}',
                },
            }
        ],
    }

    actual = block_to_openai_messages(tool_call)
    assert actual == [expected]


def test_tool_use_conversion():
    """Test tool use (with result) conversion"""
    tool_use = ToolUse(
        name="get_weather",
        arguments={"location": "Paris"},
        id="call_123",
        result={"temperature": 20, "conditions": "sunny"},
    )

    expected = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Paris"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": '{"temperature": 20, "conditions": "sunny"}',
        },
    ]

    actual = block_to_openai_messages(tool_use)
    assert isinstance(actual, list)
    assert len(actual) == 2
    assert actual == expected


def test_tool_use_with_error():

    tool_use = ToolUse(
        name="get_weather",
        arguments={"location": "InvalidCity"},
        id="call_456",
        error="City not found",
    )

    expected = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_456",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "InvalidCity"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_456",
            "content": "City not found",
        },
    ]

    actual = block_to_openai_messages(tool_use)
    assert isinstance(actual, list)
    assert len(actual) == 2
    assert actual == expected


def test_assistant_message():
    assistant = Assistant("I can help with that.")

    expected = {"role": "assistant", "content": "I can help with that."}

    actual = block_to_openai_messages(assistant)
    assert actual == [expected]


def test_conversation_with_mixed_content():

    conversation = [
        System("You are a weather assistant"),
        User("What's the weather like?"),
        Assistant("I'll check that for you."),
        ToolUse(
            name="get_weather",
            arguments={"location": "Paris"},
            id="call_1",
            result={"temp": 20},
        ),
        Assistant("It's 20°C in Paris."),
    ]

    expected_structure = [
        {"role": "system", "content": "You are a weather assistant"},
        {"role": "user", "content": "What's the weather like?"},
        {"role": "assistant", "content": "I'll check that for you."},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Paris"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 20}'},
        {"role": "assistant", "content": "It's 20°C in Paris."},
    ]

    messages = []
    for block in conversation:
        messages.extend(block_to_openai_messages(block))

    assert messages == expected_structure


def test_multiple_images_in_user_message():
    """Test user message with multiple images"""
    # Input: User with text and multiple images
    user = User(
        "Compare these images",
        Image.url("https://example.com/image1.jpg"),
        Text(" and "),
        Image.url("https://example.com/image2.jpg"),
    )

    # Expected: Content array with text and multiple images
    expected = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Compare these images"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/image1.jpg",
                    "detail": "auto",
                },
            },
            {"type": "text", "text": " and "},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/image2.jpg",
                    "detail": "auto",
                },
            },
        ],
    }

    actual = block_to_openai_messages(user)
    assert actual == [expected]


def test_tool_call_with_string_arguments():
    tool_call = ToolCall(
        name="custom_tool", arguments='{"key": "value"}', id="call_999"
    )

    expected = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_999",
                "type": "function",
                "function": {"name": "custom_tool", "arguments": '{"key": "value"}'},
            }
        ],
    }

    actual = block_to_openai_messages(tool_call)
    assert actual == [expected]
