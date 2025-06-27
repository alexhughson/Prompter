"""Unit tests for Anthropic prompt to API format conversion logic"""

from pydantic import BaseModel

from prompter.schemas import (
    Assistant,
    Block,
    Image,
    Prompt,
    System,
    Text,
    Tool,
    ToolCall,
    ToolUse,
    User,
)
from prompter.anthropic_executor import (
    block_to_anthropic_content,
    content_list_to_anthropic,
    tool_to_anthropic,
    merge_consecutive_roles,
)


class WeatherParams(BaseModel):
    location: str
    units: str = "celsius"


def test_simple_text_prompt():
    user = User("Hello")

    expected = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]

    actual = block_to_anthropic_content(user)
    assert actual == expected


def test_multi_part_user_content():
    user = User(Text("Hello"), " ", Text("world"))
    
    expected = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": " "},
                {"type": "text", "text": "world"},
            ],
        }
    ]
    
    actual = block_to_anthropic_content(user)
    assert actual == expected


def test_image_prompt():
    """Test image prompt conversion with data URL"""
    # Using data URL for predictable test
    img = Image(source="data:image/png;base64,iVBORw0KGgo=", media_type="image/png")
    user = User("What's this?", img)

    msg = block_to_anthropic_content(user)
    assert len(msg) == 1
    assert msg[0]["role"] == "user"
    assert len(msg[0]["content"]) == 2
    assert msg[0]["content"][0] == {"type": "text", "text": "What's this?"}
    assert msg[0]["content"][1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="},
    }


def test_tool_conversion():
    tool = Tool(
        name="get_weather",
        description="Get weather for a location",
        params=WeatherParams,
    )

    expected = {
        "name": "get_weather",
        "description": "Get weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "title": "Location"},
                "units": {"type": "string", "default": "celsius", "title": "Units"},
            },
            "required": ["location"],
            "title": "WeatherParams",
        },
    }

    actual = tool_to_anthropic(tool)
    assert actual == expected


def test_tool_with_dict_params():
    """Test tool with dictionary parameters"""
    tool = Tool(
        name="search",
        description="Search for items",
        params={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    )

    anthropic_tool = tool_to_anthropic(tool)
    assert anthropic_tool["input_schema"] == tool.params


def test_tool_call_conversion():
    """Test standalone tool call conversion"""
    tool_call = ToolCall(
        name="get_weather", arguments={"location": "Paris"}, id="call_123"
    )

    msg = block_to_anthropic_content(tool_call)
    assert msg == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_123",
                    "name": "get_weather",
                    "input": {"location": "Paris"},
                }
            ],
        }
    ]


def test_tool_use_conversion():
    """Test tool use with result conversion"""
    tool_use = ToolUse(
        name="get_weather",
        arguments={"location": "Paris"},
        id="call_123",
        result={"temperature": 20, "conditions": "sunny"},
    )

    msgs = block_to_anthropic_content(tool_use)
    assert isinstance(msgs, list)
    assert len(msgs) == 2

    # First message is the tool use
    assert msgs[0] == {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "call_123",
                "name": "get_weather",
                "input": {"location": "Paris"},
            }
        ],
    }

    # Second message is the tool result
    assert msgs[1] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "call_123",
                "content": '{"temperature": 20, "conditions": "sunny"}',
            }
        ],
    }


def test_tool_use_with_error():
    """Test tool use with error"""
    tool_use = ToolUse(
        name="get_weather",
        arguments={"location": "InvalidCity"},
        id="call_456",
        error="City not found",
    )

    msgs = block_to_anthropic_content(tool_use)
    assert isinstance(msgs, list)
    assert len(msgs) == 2
    assert msgs[1]["content"][0]["content"] == "City not found"


def test_assistant_message():
    """Test assistant message conversion"""
    assistant = Assistant("I can help with that.")
    msg = block_to_anthropic_content(assistant)
    assert msg == [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "I can help with that."}],
        }
    ]


def test_merge_consecutive_roles():
    """Test merging consecutive messages with same role"""
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        {"role": "user", "content": [{"type": "text", "text": "How are you?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "I'm well."}]},
    ]

    merged = merge_consecutive_roles(messages)
    assert len(merged) == 2
    assert merged[0]["role"] == "user"
    assert len(merged[0]["content"]) == 2
    assert merged[1]["role"] == "assistant"
    assert len(merged[1]["content"]) == 2


def test_system_block_error():
    """Test that system blocks raise error when converted directly"""
    system = System("You are helpful")
    try:
        block_to_anthropic_content(system)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "System blocks should be handled separately" in str(e)


def test_conversation_with_mixed_content():
    """Test full conversation with various content types"""
    # Note: System is handled separately in the executor
    conversation = [
        User("What's the weather like?"),
        Assistant("I'll check that for you."),
        ToolCall(name="get_weather", arguments={"location": "Paris"}, id="call_1"),
        # In real usage, ToolUse would include the call and result together
    ]

    messages = []
    for block in conversation:
        msg = block_to_anthropic_content(block)
        messages.extend(msg)

    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[2]["role"] == "assistant"  # tool call
    assert messages[2]["content"][0]["type"] == "tool_use"
