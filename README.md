# Prompter

Prompter is a Python library for structured interactions with LLMs, built on a unified block-based architecture where everything is composable.

## Installation

```bash
pip install prompter

# With specific LLM support
pip install prompter[anthropic]
pip install prompter[openai]
pip install prompter[all]  # Install all providers
```

## Basic Usage

```python
from prompter.schemas import Prompt, User
from prompter.anthropic_executor import ClaudeExecutor

# Simple text response
prompt = Prompt(
    system="You are a helpful assistant",
    conversation=[
        User("Write a haiku about coding")
    ]
)

executor = ClaudeExecutor()
response = executor.execute(prompt)
print(response.text())
```

## Unified Block Architecture

Everything in Prompter is a "block" - a composable unit that can be combined in conversations:

```python
from prompter.schemas import Prompt, User, Assistant, Image, Tool, ToolUse

# Multi-modal conversation
prompt = Prompt(
    system="You are an image analyst",
    conversation=[
        User(
            "What's in this image?",
            Image.url("https://example.com/cat.jpg"),
            "Is it cute?"
        ),
        Assistant("I see a cute orange cat in the image!"),
        User("Describe it in detail")
    ]
)
```

## Tool Usage with ToolBelt

The ToolBelt class manages tool definitions and parsing:

```python
from pydantic import BaseModel
from prompter.schemas import Tool, Prompt, User, ToolUse
from prompter.tool_belt import ToolBelt

class WeatherParams(BaseModel):
    location: str
    units: str = "celsius"

weather_tool = Tool(
    name="get_weather",
    description="Get the weather for a location",
    params=WeatherParams
)

# Create a tool belt to manage tools
tool_belt = ToolBelt([weather_tool])

prompt = Prompt(
    system="You are a weather assistant",
    conversation=[User("What's the weather in Tokyo?")],
    tools=tool_belt.tool_list()
)

response = executor.execute(prompt)

# Parse tool calls with proper validation
for tool_call in response.tool_calls():
    # ToolBelt parses and validates arguments
    parsed_args = tool_belt.parse_call(tool_call)
    print(f"Location: {parsed_args.location}, Units: {parsed_args.units}")
    
    # Execute your tool logic
    result = {"temperature": 22, "conditions": "sunny"}
    
    # Add result to conversation
    prompt.conversation.append(
        ToolUse(
            name=tool_call.name,
            arguments=tool_call.arguments,
            result=result
        )
    )

# Continue conversation with results
response = executor.execute(prompt)
print(response.text())
```

## Working with Images

```python
# Multiple ways to add images
prompt = Prompt(
    conversation=[
        User(
            "Compare these images",
            Image.url("https://example.com/image1.jpg"),
            Image.file("/path/to/local/image.png"),
            Image.base64(encoded_data, media_type="image/jpeg")
        )
    ]
)
```

## Conversation History

Build complex conversations with tool results:

```python
prompt = Prompt(
    system="You are a helpful weather assistant",
    conversation=[
        User("What's the weather in Tokyo and London?"),
        ToolUse(
            name="get_weather",
            arguments={"location": "Tokyo", "units": "celsius"},
            result={"temperature": 22, "conditions": "sunny"}
        ),
        ToolUse(
            name="get_weather",
            arguments={"location": "London", "units": "celsius"},
            result={"temperature": 15, "conditions": "rainy"}
        ),
        Assistant("Tokyo is 22°C and sunny, while London is 15°C and rainy."),
        User("Should I bring an umbrella to London?")
    ]
)
```

## Error Handling

Handle tool errors gracefully:

```python
prompt = Prompt(
    conversation=[
        User("What's the weather on Mars?"),
        ToolUse(
            name="get_weather",
            arguments={"location": "Mars"},
            error="Location not found"
        ),
        Assistant("I couldn't get weather data for Mars.")
    ]
)
```

## Common Patterns

```python
# Quick constructors
simple_prompt = Prompt.simple("You are helpful", "Explain quantum computing")

prompt_with_tools = Prompt.with_tools(
    [weather_tool, time_tool],
    "What's the weather and time in Paris?"
)

# Response format (coming soon)
prompt_with_schema = Prompt.with_schema(
    PersonInfo,
    "Extract: John Doe is 25 years old"
)
```

## LLM Response API

The unified `LLMResponse` class provides consistent access to results:

```python
response = executor.execute(prompt)

# Get text content
text = response.text()

# Get all tool calls
for tool_call in response.tool_calls():
    print(f"Tool: {tool_call.name}")
    print(f"Args: {tool_call.arguments}")

# Get single tool call (if expecting one)
tool_call = response.tool_call()

# Access raw response
raw = response.raw_response
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
make test

# Format code
make format

# Type checking
make type

# Run all checks
make check
```

## Key Concepts

- **Blocks**: Unified content types (User, Assistant, Image, ToolUse, etc.)
- **ToolBelt**: Manages tool definitions and validates/parses tool calls
- **Conversation**: List of blocks representing dialogue history
- **Tools**: Functions with Pydantic models for parameter validation
- **ToolUse**: Records of tool executions with results or errors

## Design Philosophy

Prompter follows functional programming principles:
- Pure functions for transformations
- Composable blocks
- Immutable prompt construction
- Explicit error handling

See [DESIGN.md](DESIGN.md) for architectural details.