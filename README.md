# Prompter

Prompter is a small class based python representation of LLM Prompts, and some adapters to make it easier to use with different LLM providers.

# VERY MUCH A WORK IN PROGRESS

## Usage

```python
from prompter import Prompt, UserMessage, OpenAIAdapter

prompt = Prompt(
    system_prompt="You are a helpful assistant. You are given a question and you need to answer it.",
    messages=[UserMessage(content="How's it going?")],
)

adapter = OpenAIAdapter()

response = adapter.execute(prompt)

for message in response.messages:
    print(str(message))
```

## Sending Images

```python
from prompter import Prompt, ImageMessage, OpenAIAdapter

prompt = Prompt(
    system_prompt="You are an image classifier.  Sorry about that",
    messages=[ImageMessage(image_path="image.png")],
)

adapter = OpenAIAdapter()

response = adapter.execute(prompt)

for message in response.messages:
    print(str(message))
```

## Build your own Adapters

Ultimately the point of this is that you can quickly build your own adapters for different LLM providers.

Just take in a Prompt object and return a LLMResponse object.

