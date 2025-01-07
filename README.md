# Prompter

Prompter is a Python library for structured interactions with LLMs, providing type-safe parsing and validation of responses.

## Basic Usage

```python
from prompter import Prompt, UserMessage, OpenAIExecutor, Schema

# Simple text response
prompt = Prompt(
    system_message="You are a helpful assistant",
    messages=[UserMessage("Write a haiku about coding")]
)

response = OpenAIExecutor().execute(prompt)
print(response.text())  # All text from output messages as a string
print(response.text_messages())  # iterable of text messages
```

## Structured Outputs

```python
# With schema
class Character(Schema):
    name: str
    species: str
    age: int

prompt = Prompt(
    system_message="You are a character creator",
    messages=[UserMessage("Create a sci-fi character")],
    response_schema=Character
)

response = OpenAIExecutor().execute(prompt)
result = response.result()  # Returns SchemaResult, raises if no schema was specified

# Different ways to handle the result:
try:
    # Raises SchemaValidationError if invalid
    character = result.parse()  # Returns Character instance
    print(f"Created {character.name}")
except SchemaValidationError as e:
    print(f"Invalid response: {e}")

# Or check validity first
if result.valid():
    character = result.parse()  # Safe to call
else:
    print("Response didn't match schema")

# Get raw text
print(result.raw())  # Original LLM response

# Get dict representation (raises if not valid JSON)
try:
    char_dict = result.parse_obj()
    print(f"As dict: {char_dict}")
except JSONDecodeError:
    print("Response wasn't valid JSON")

if result.valid_json():
    # safe to call
    dict = result.parse_obj()
```



## Tool Usage

```python
class WeatherArgs(Schema):
    location: str
    units: str = "celsius"

prompt = Prompt(
    system_message="You are a weather assistant",
    messages=[UserMessage("What's the weather in Toronto?")],
    tools=[
        Tool(
            name="get_weather",
            description="Get weather forecast",
            argument_schema=WeatherArgs
        )
    ]
)

response = OpenAIExecutor().execute(prompt)

# Handle tool calls
for tool_call in response.tool_calls():
    # Tool arguments are a SchemaResult
    if tool_call.arguments.valid():
        args = tool_call.arguments.parse()  # Returns WeatherArgs instance
        weather = get_weather(**args.dict())
        response.add_tool_result(tool_call.id, weather)
    else:
        print(f"Invalid tool arguments: {tool_call.arguments.raw()}")

# Get all messages including tool results
print(response.text(include_tools=True))
```

## Error Handling Patterns

```python
def get_character(retries: int = 3) -> Character:
    prompt = Prompt(
        system_message="You are a character creator",
        messages=[UserMessage("Create a sci-fi character")],
        response_schema=Character
    )

    for attempt in range(retries):
        response = OpenAIExecutor().execute(prompt)
        result = response.result()

        if result.valid():
            return result.parse()

        if result.valid_json():
            # Maybe we can work with partial data
            data = result.parse_obj()
            prompt.messages.append(
                UserMessage(f"Almost correct, but had errors: {data}")
            )
        else:
            prompt.messages.append(
                UserMessage("Please respond with valid JSON")
            )

    raise ValueError(f"Failed to get valid response after {retries} attempts")
```

## Key Concepts

- Response methods:
  - `.result()` - Returns SchemaResult when response_schema was specified
  - `.messages()` - All messages
  - `.tool_calls()` - Just the tool calls
  - `.text()` - All text including optional tool results
  - `.text_messages()` - Just the text messages
- SchemaResult methods:
  - `.parse()` - Returns instance of schema class
  - `.parse_obj()` - Returns dict/list representation
  - `.valid()` - Schema validation check
  - `.valid_json()` - JSON validation check
  - `.raw()` - Original text
- Tool calls have `.name` and `.arguments` (SchemaResult) properties

## Installation

```bash
pip install prompter
```