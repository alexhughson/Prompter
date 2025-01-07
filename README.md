# Prompter

Prompter is a small class based python representation of LLM Prompts, and some adapters to make it easier to use with different LLM providers.

# VERY MUCH A WORK IN PROGRESS

## Usage

```python
from prompter import Prompt, UserMessage, OpenAIExecutor

prompt = Prompt(
    system_message="You are a helpful assistant",
    messages=[
        UserMessage(content="What is this?"),
        ImageMessage(
            url=URL_OF_CLOUD_PICTURE,
        ),
    ],
)
executor = OpenAIExecutor()

response = executor.execute(prompt)
response.raise_for_status()

print("Cloud test response: " + response.text())
assert "cloud" in response.text().lower()
```

## Tool Usage

```python
prompt = Prompt(
    system_message="You are a helpful assistant",
    messages=[
        UserMessage(
            content="Please use the get_weather tool to get the weather in toronto"
        ),
    ],
    tools=[
        Tool(
            name="get_weather",
            description="Get the weather in a given location",
            argument_schema=WeatherArgs,
        )
    ],
    tool_use=Prompt.TOOL_USE_REQUIRED,
)
executor = AnthropicExecutor()
response = executor.execute(prompt)
response.raise_for_status()
tool_call = response.tool_call()

print("Tool call: " + str(tool_call.arguments))

assert tool_call.name == "get_weather"
assert isinstance(tool_call.arguments, WeatherArgs)
assert tool_call.arguments.location.lower() == "toronto"
```

## Build your own Adapters

Ultimately the point of this is that you can quickly build your own adapters for different LLM providers.

Just take in a Prompt object and return a LLMResponse object.

## Documentation

Full documentation is available at: https://alexhughson.github.io/Prompter/

